/**
 * @file run_page_test.cpp
 * @brief The run report (report.html): built from the run's records, the same page by the run and by
 *        mallard info --html, with the facts of the run, bounded in size, and safe for any text it shows.
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 */

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "check.h"
#include "comm.h"
#include "json.h"
#include "provenance.h"
#include "run_page.h"
#include "solver.h"
#include "test_fixtures.h"

namespace {

namespace fs = std::filesystem;

std::string case_input(const std::string & dir, const std::string & run, const std::string & output) {
    std::ostringstream s;
    s << "[run]\n" << run
      << "[mesh]\ntype = \"cartesian\"\nNx = 12\nNy = 10\nLx = 1.0\nLy = 1.0\n"
      << (N_DIM == 3 ? "Nz = 3\nLz = 0.25\n[[periodic]]\nzones = [\"back\", \"front\"]\n" : "")
      << "[initialize]\ntype = \"analytical\"\n"
      << "rho = \"1.0 + (x < 0.4 ? 1.0 : 0.0)\"\nu = [\"0.2\", \"0.1\"" << (N_DIM == 3 ? ", \"0.0\"" : "")
      << "]\np = \"x < 0.4 ? 2.0 : 1.0\"\n"
      << "[[boundaries]]\nname = \"left\"\ntype = \"outlet_extrapolation\"\n"
      << "[[boundaries]]\nname = \"right\"\ntype = \"outlet_extrapolation\"\n"
      << "[[boundaries]]\nname = \"top\"\ntype = \"symmetry\"\n"
      << "[[boundaries]]\nname = \"bottom\"\ntype = \"wall_adiabatic\"\n"
      << "[numerics]\nriemann_solver = \"HLLC\"\ntime_integrator = \"SSPRK3\"\n"
      << "[numerics.face_reconstruction]\ntype = \"MUSCL\"\n"
      << "[physics]\ntype = \"euler\"\ngamma = 1.4\np_ref = 1.0\nT_ref = 1.0\nrho_ref = 1.0\n"
      << "[output]\ncheck_interval = 1\n" << output
      << "[[write_data]]\nprefix = \"" << dir << "/restart\"\nformat = \"restart\"\ninterval = 3\n";
    return s.str();
}

std::string fresh_dir(const std::string & name) {
    if (comm::is_root()) fs::create_directories(name);
    comm::barrier();
    return name;
}

std::string logged(const std::string & dir) {
    return "run_log = \"" + dir + "/mallard.jsonl\"\nreport = \"" + dir + "/report.html\"\n";
}

std::string read_file(const std::string & path) {
    std::ifstream in(path, std::ios::binary);
    return {std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>()};
}

/** @brief The last record of this type in a run log. */
json::Value last_record(const std::string & log, const std::string & type) {
    json::Value found;
    std::ifstream in(log);
    for (std::string line; std::getline(in, line);) {
        json::Value v = json::parse(line);
        if (v["type"].string == type) found = v;
    }
    return found;
}

/** @brief The series data embedded in a page. */
json::Value embedded_series(const std::string & page) {
    const std::string open = "id=\"mallard-series\">";
    const size_t begin = page.find(open) + open.size();
    return json::parse(page.substr(begin, page.find("</script>", begin) - begin));
}

const json::Value * find_series(const json::Value & data, const std::string & name) {
    for (const json::Value & g : data["groups"].array) {
        for (const json::Value & s : g["series"].array) {
            if (s["name"].string == name) return &s;
        }
    }
    return nullptr;
}

/** @brief The times or steps of a series' buckets, from the axis it shares. */
const std::vector<json::Value> & axis(const json::Value & data, const json::Value & series, const char * what) {
    return data["axes"].array[size_t(series["axis"].number)][what].array;
}

} // namespace

TEST(RunPageTest, ARunWritesItsReportFromItsRecordsAndInfoRebuildsTheSamePage) {
    const std::string dir = fresh_dir("report");
    Solver solver;
    const std::string monitors = "[integrals]\ninterval = 1\n[[forces]]\nzone = \"bottom\"\ninterval = 1\nfile = \"" +
                                 dir + "/forces_bottom.csv\"\n";
    solver.init(parse_toml(case_input(dir, "n_steps = 7\ncfl = 0.25\n", logged(dir) + monitors)));
    EXPECT_EQ(solver.run(), 0);
    if (!comm::is_root()) return;
    const std::string log = dir + "/mallard.jsonl";
    const std::string page = read_file(dir + "/report.html");
    ASSERT_FALSE(page.empty());
    EXPECT_FALSE(fs::exists(dir + "/report.html.partial"));

    const json::Value start = last_record(log, "start");
    const json::Value end = last_record(log, "end");
    EXPECT_NE(page.find(start["run_id"].string), std::string::npos);
    EXPECT_NE(page.find("<strong>finished (n_steps)</strong> at step 7"), std::string::npos);
    const json::Value & restart = end["last_restart"];
    EXPECT_NE(page.find(restart["path"].string + ", step " + std::to_string(uint64_t(restart["step"].number))),
              std::string::npos);
    EXPECT_NE(page.find("<td>SSPRK3</td>"), std::string::npos);
    EXPECT_NE(page.find("<td>wall_adiabatic</td>"), std::string::npos);

    const json::Value data = embedded_series(page);
    const json::Value * min_rho = find_series(data, "min_rho");
    ASSERT_NE(min_rho, nullptr);
    EXPECT_EQ(axis(data, *min_rho, "step").back().number, 7.0);

    // Every plot has its axes labeled with the units the run recorded
    EXPECT_EQ(data["xlabel"].string, "t [s]");
    size_t n_series = 0;
    std::vector<std::string> groups;
    for (const json::Value & g : data["groups"].array) {
        groups.push_back(g["title"].string);
        for (const json::Value & s : g["series"].array) {
            n_series++;
            const std::string & unit = s["unit"].string;
            EXPECT_FALSE(unit.empty()) << s["name"].string;
            EXPECT_EQ(s["label"].string, s["name"].string + " [" + unit + "]");
        }
    }
    EXPECT_NE(std::find(groups.begin(), groups.end(), "Integrals over the domain"), groups.end());
    EXPECT_NE(std::find(groups.begin(), groups.end(), "Forces on bottom"), groups.end());
    EXPECT_GT(n_series, 10u);
    EXPECT_EQ(find_series(data, "min_p")->find("unit")->string, "kg m-1 s-2");
    EXPECT_EQ(find_series(data, "Fx_pressure")->find("unit")->string, N_DIM == 2 ? "kg s-2" : "kg m s-2");

    EXPECT_EQ(run_page::from_file(log).html(), page);
}

TEST(RunPageTest, ADivergedRunsReportShowsTheLikelyCauseAndItsFix) {
    const std::string dir = fresh_dir("report_diverged");
    Solver solver;
    solver.init(parse_toml(case_input(dir, "n_steps = 500\ndt = 0.2\n", logged(dir))));
    EXPECT_THROW(solver.run(), std::runtime_error);
    if (!comm::is_root()) return;
    const std::string page = read_file(dir + "/report.html");
    const json::Value divergence = last_record(dir + "/mallard.jsonl", "divergence");
    ASSERT_TRUE(divergence.has("causes"));
    const json::Value & cause = divergence["causes"].array.front();
    EXPECT_NE(page.find("<section id=\"divergence\">"), std::string::npos);
    EXPECT_NE(page.find("<td>" + cause["title"].string + "</td>"), std::string::npos);
    EXPECT_NE(page.find("banner error"), std::string::npos);
}

TEST(RunPageTest, ARunWithoutARunLogStillWritesItsReport) {
    const std::string dir = fresh_dir("report_no_log");
    Solver solver;
    solver.init(parse_toml(case_input(dir, "n_steps = 4\ncfl = 0.25\n",
                                      "run_log = \"\"\nreport = \"" + dir + "/report.html\"\n")));
    EXPECT_EQ(solver.run(), 0);
    if (!comm::is_root()) return;
    EXPECT_FALSE(fs::exists(dir + "/mallard.jsonl"));
    const std::string page = read_file(dir + "/report.html");
    EXPECT_NE(page.find(provenance::run_id()), std::string::npos);
    EXPECT_NE(page.find("<strong>finished (n_steps)</strong> at step 4"), std::string::npos);
}

TEST(RunPageTest, ALongRunsPageStaysSmallAndKeepsItsExtremes) {
    run_page::Page page;
    page.add(json::parse(R"({"type":"start","run_id":"r1","provenance":{"input":{"file":"</script><b>x"}}})"));
    page.add(json::parse(R"({"type":"warning","message":"<b>bold</b>"})"));
    const uint64_t rows = 200000, spike = 123457;
    for (uint64_t i = 1; i <= rows; i++) {
        const double t = 1e-3 * double(i);
        json::Object r;
        r.add("type", "progress").add("step", 5 * i).add("t", t).add("dt", 1e-3).add("cfl", 0.5);
        r.add_numbers("monitors", {{"min_rho", i == spike ? -0.5 : 1.0 + 1e-6 * double(i % 97)},
                                   {"min_p", 1.0},
                                   {"max_mach", 0.3}});
        json::Object integrals, forces;
        integrals.add("step", 5 * i).add("t", t);
        forces.add("step", 5 * i).add("t", t);
        for (const char * c : {"mass", "momentum_x", "momentum_y", "energy", "kinetic"}) integrals.add(c, 1.0 + t);
        for (const char * c : {"Fx", "Fy", "Fx_p", "Fy_p", "Fx_v", "Fy_v"}) forces.add(c, std::sin(t));
        r.raw("integrals", integrals.str()).raw("forces_wall", forces.str());
        page.add(json::parse(r.str()));
    }
    const std::string html = page.html();
    EXPECT_LT(html.size(), 2u << 20);
    EXPECT_EQ(html.find("</script><b>"), std::string::npos);
    EXPECT_EQ(html.find("<b>bold</b>"), std::string::npos);
    EXPECT_NE(html.find("&lt;b&gt;bold&lt;/b&gt;"), std::string::npos);
    EXPECT_EQ(html.find(" src="), std::string::npos);
    EXPECT_EQ(html.find("<link"), std::string::npos);

    const json::Value data = embedded_series(html);
    const json::Value * min_rho = find_series(data, "min_rho");
    ASSERT_NE(min_rho, nullptr);
    EXPECT_LE(axis(data, *min_rho, "t").size(), run_page::BUCKETS);
    double least = 1.0;
    for (const json::Value & v : (*min_rho)["lo"].array) least = std::min(least, v.number);
    EXPECT_EQ(least, -0.5);
    EXPECT_EQ(axis(data, *min_rho, "step").back().number, double(5 * rows));
}

TEST(RunPageTest, ARewoundRestartDropsTheRowsAfterIt) {
    run_page::Page page;
    page.add(json::parse(R"({"type":"start","run_id":"a"})"));
    for (int i = 1; i <= 10; i++) {
        page.add(json::parse(R"({"type":"progress","step":)" + std::to_string(10 * i) + R"(,"t":)" +
                             std::to_string(i) + R"(,"dt":1,"monitors":{"min_rho":)" + std::to_string(i) + "}}"));
    }
    page.add(json::parse(R"({"type":"start","run_id":"b","restart":{"file":"r","step":50,"t":5,"rewound_from":100}})"));
    page.add(json::parse(R"({"type":"progress","step":60,"t":6,"dt":1,"monitors":{"min_rho":0.25}})"));
    const json::Value data = embedded_series(page.html());
    const json::Value * min_rho = find_series(data, "min_rho");
    ASSERT_NE(min_rho, nullptr);
    std::vector<double> steps, values;
    for (const json::Value & v : axis(data, *min_rho, "step")) steps.push_back(v.number);
    for (const json::Value & v : (*min_rho)["y"].array) values.push_back(v.number);
    EXPECT_EQ(steps, (std::vector<double>{10, 20, 30, 40, 50, 60}));
    EXPECT_EQ(values, (std::vector<double>{1, 2, 3, 4, 5, 0.25}));
}

TEST(RunPageTest, ActivityShowsTheShareOfAllRanksAndOfTheLeastAndMostActiveRank) {
    run_page::Page page;
    page.add(json::parse(R"({"type":"start","run_id":"a"})"));
    page.add(json::parse(R"({"type":"progress","step":10,"t":1,"dt":1,"health":{"activity":{)"
                         R"("troubled_cells":{"all":0.5,"rank_min":0.25,"rank_max":0.75},)"
                         R"("limited_cells":{"all":1,"rank_min":1,"rank_max":1}}}})"));
    const std::string html = page.html();
    const auto row = [&](const std::string & name) {
        const size_t begin = html.find("<tr><td>" + name + "</td>");
        return begin == std::string::npos ? std::string() : html.substr(begin, html.find("</tr>", begin) - begin);
    };
    const auto shares = [](const char * all, const char * least, const char * most) {
        return "<td class=\"num\">" + std::string(all) + "</td><td class=\"num\">" + least +
               "</td><td class=\"num\">" + most + "</td>";
    };
    EXPECT_TRUE(row("troubled_cells").ends_with(shares("50.00", "25.00", "75.00"))) << row("troubled_cells");
    EXPECT_TRUE(row("limited_cells").ends_with(shares("100.00", "100.00", "100.00"))) << row("limited_cells");
}

TEST(RunPageTest, ARunRecordsTheCheckAtSetupAsMallardCheckMeasuresIt) {
    const std::string dir = fresh_dir("report_check");
    // CFL 1.3 is above SSPRK3's limit for smooth flow (1.2): a warning of the check
    const std::string input = case_input(dir, "n_steps = 2\ncfl = 1.3\n", logged(dir));
    Solver solver;
    solver.init(parse_toml(input));
    EXPECT_EQ(solver.run(), 0);
    Solver checker;
    const CheckReport checked = checker.check(parse_toml(input), CheckOptions{});
    if (!comm::is_root()) return;

    const json::Value check = last_record(dir + "/mallard.jsonl", "start")["check"];
    ASSERT_EQ(check.type, json::Value::Type::OBJECT);
    for (size_t m = 0; m < MeshQuality::N_METRICS; m++) {
        const json::Value & q = check["mesh_quality"][MeshQuality::FIELDS[m]];
        EXPECT_EQ(q["max"].number, checked.quality[m].max) << MeshQuality::NAMES[m];
        EXPECT_EQ(q["units"].string, MeshQuality::UNITS[m]);
    }
    ASSERT_FALSE(checked.resolution.empty());
    for (const CheckEstimate & e : checked.resolution) {
        const json::Value & recorded = check["resolution"][e.id];
        EXPECT_EQ(recorded["quantity"].string, e.quantity);
        if (!std::isnan(e.value)) {
            EXPECT_EQ(recorded["value"].number, e.value) << e.id;
        }
    }
    EXPECT_EQ(check["time_step"]["cfl_limit"].number, 1.2);
    EXPECT_EQ(check["status"].string, "warning");
    ASSERT_EQ(check["diagnostics"].array.size(), 1u);
    EXPECT_EQ(check["diagnostics"].array[0]["code"].string, check_code::CFL_LIMIT);

    const run_page::Page page = run_page::from_file(dir + "/mallard.jsonl");
    const std::string text = page.text(false);
    const size_t section = text.find("== Check at setup");
    ASSERT_NE(section, std::string::npos);
    EXPECT_NE(text.find("WARN", section), std::string::npos);
    const size_t row = text.find("CFL number, run.cfl", section);
    ASSERT_NE(row, std::string::npos);
    EXPECT_EQ(text.substr(text.rfind('\n', row) + 1, 6), "  WARN");
    EXPECT_NE(text.find(check_code::CFL_LIMIT, section), std::string::npos);
    const std::string html = page.html();
    EXPECT_NE(html.find("<section id=\"check\"><h2>Check at setup <span class=\"status warning\">"), std::string::npos);
    EXPECT_NE(html.find("<td>aspect ratio</td>"), std::string::npos);
}

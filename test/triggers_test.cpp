/**
 * @file triggers_test.cpp
 * @author Matthew Bonanni (mbonanni001@gmail.com)
 * @brief [[triggers]]: each action at the step its condition says, once and
 *        repeated, held conditions, the state across restarts, rank agreement
 *        and errors in conditions.
 * @version 0.1
 * @date 2026-11-04
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 *
 */

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iterator>
#include <map>
#include <sstream>
#include <string>
#include <vector>

#include "comm.h"
#include "output_reader.h"
#include "scratch.h"
#include "solver.h"
#include "test_fixtures.h"

namespace {

namespace fs = std::filesystem;

std::string case_input(const std::string & dir, const std::string & run, const std::string & extra = "",
                       uint64_t restart_interval = 1000) {
    std::ostringstream s;
    s << "[run]\n" << run
      << "[mesh]\ntype = \"cartesian_tri\"\nNx = 12\nNy = 10\nLx = 1.0\nLy = 1.0\n"
      << "[initialize]\ntype = \"analytical\"\n"
      << "rho = \"1.0 + (x < 0.4 ? 1.0 : 0.0)\"\nu = [\"0.2\", \"0.1\"]\np = \"x < 0.4 ? 2.0 : 1.0\"\n"
      << "[[boundaries]]\nname = \"left\"\ntype = \"outlet_extrapolation\"\n"
      << "[[boundaries]]\nname = \"right\"\ntype = \"outlet_extrapolation\"\n"
      << "[[boundaries]]\nname = \"top\"\ntype = \"symmetry\"\n"
      << "[[boundaries]]\nname = \"bottom\"\ntype = \"wall_adiabatic\"\n"
      << "[numerics]\nriemann_solver = \"HLLC\"\ntime_integrator = \"SSPRK3\"\n"
      << "[numerics.face_reconstruction]\ntype = \"MUSCL\"\n"
      << "[physics]\ntype = \"euler\"\ngamma = 1.4\np_ref = 1.0\nT_ref = 1.0\nrho_ref = 1.0\n"
      << "[output]\ncheck_interval = 1000000\nrun_log = \"" << dir << "/mallard.jsonl\"\n"
      << "[[write_data]]\nprefix = \"" << dir << "/restart\"\nformat = \"restart\"\ninterval = " << restart_interval
      << "\n"
      << extra;
    return s.str();
}

std::string fresh_dir(const std::string & name) {
    const std::string dir = (scratch_dir() / ("triggers_" + name)).string();
    comm::barrier();
    if (comm::is_root()) {
        fs::remove_all(dir);
        fs::create_directories(dir);
    }
    comm::barrier();
    return dir;
}

std::string restart_file(const std::string & dir, uint64_t step) {
    std::ostringstream s;
    s << dir << "/restart_" << std::setw(6) << std::setfill('0') << step << ".restart";
    return s.str();
}

/** @brief Steps with a restart file in dir. */
std::vector<uint64_t> restart_steps(const std::string & dir) {
    comm::barrier();
    std::vector<uint64_t> steps;
    for (const auto & entry : fs::directory_iterator(dir)) {
        const std::string name = entry.path().filename().string();
        if (name.rfind("restart_", 0) == 0 && entry.path().extension() == ".restart") {
            steps.push_back(std::stoull(name.substr(8, 6)));
        }
    }
    std::sort(steps.begin(), steps.end());
    return steps;
}

int run(Solver & solver) {
    const int status = solver.run();
    comm::barrier();
    return status;
}

const std::string TRIGGER_CASE_PROBE =
    "[[probes]]\nname = \"gauge\"\npoint = [0.45, 0.5]\nvariables = [\"P\"]\ninterval = 1\n";

} // namespace

TEST(TriggersTest, ProbeThresholdStopsAtTheFirstStepPastItOnEveryRank) {
    // The signal: the pressure at a point as the expansion of the high-pressure side reaches it
    const std::string ref = fresh_dir("threshold_ref");
    Solver reference;
    reference.init(parse_toml(case_input(ref, "n_steps = 12\ncfl = 0.25\n",
                                         TRIGGER_CASE_PROBE + "file = \"" + ref + "/gauge.csv\"\n")));
    run(reference);
    std::vector<double> p(13, 0.0);
    {
        std::ifstream in(ref + "/gauge.csv");
        std::string line;
        while (std::getline(in, line)) {
            if (line.empty() || !std::isdigit(static_cast<unsigned char>(line[0]))) continue;
            std::vector<std::string> cells;
            std::stringstream row(line);
            for (std::string cell; std::getline(row, cell, ',');) cells.push_back(cell);
            p[std::stoull(cells[0])] = std::stod(cells.back());
        }
    }
    const double threshold = 0.5 * (p[5] + p[6]);
    uint64_t expected = 0;
    for (uint64_t s = 1; s <= 12 && expected == 0; s++) {
        if (p[s] > threshold) expected = s;
    }
    ASSERT_GE(expected, 2u);
    ASSERT_LT(expected, 12u);

    const std::string dir = fresh_dir("threshold");
    std::ostringstream trigger;
    trigger << std::setprecision(17) << "[[triggers]]\nname = \"pressure_rise\"\nwhen = \"gauge_P > " << threshold
            << " and wall >= 0\"\ndo = \"stop\"\n";
    Solver solver;
    solver.init(parse_toml(case_input(dir, "n_steps = 12\ncfl = 0.25\n",
                                      TRIGGER_CASE_PROBE + "file = \"" + dir + "/gauge.csv\"\n" + trigger.str())));
    EXPECT_EQ(run(solver), 0);
    EXPECT_EQ(solver.get_step(), expected);
    const double step = static_cast<double>(solver.get_step());
    EXPECT_EQ(comm::allreduce(step, comm::Op::MIN), comm::allreduce(step, comm::Op::MAX));
    EXPECT_TRUE(fs::exists(restart_file(dir, expected)));
    std::ifstream log(dir + "/mallard.jsonl");
    const std::string text((std::istreambuf_iterator<char>(log)), std::istreambuf_iterator<char>());
    EXPECT_NE(text.find("\"kind\":\"trigger\""), std::string::npos);
    EXPECT_NE(text.find("\"name\":\"pressure_rise\",\"firing\":1,\"values\":{"), std::string::npos);
    EXPECT_NE(text.find("\"gauge_P\":"), std::string::npos);
    EXPECT_NE(text.find("\"outcome\":\"stopped\",\"cause\":\"trigger\",\"trigger\":\"pressure_rise\",\"reason\":"
                        "\"trigger pressure_rise"),
              std::string::npos)
        << text;
}

TEST(TriggersTest, OnceFiresOnTheFirstStepAndRepeatOnEveryRisingEdge) {
    for (const bool repeat : {false, true}) {
        const std::string dir = fresh_dir(repeat ? "repeat" : "once");
        Solver solver;
        solver.init(parse_toml(case_input(dir, "n_steps = 10\ncfl = 0.25\n",
                                          std::string("[[triggers]]\nwhen = \"step % 3 == 0\"\ndo = \"checkpoint\"\n") +
                                              "repeat = " + (repeat ? "true" : "false") + "\n")));
        EXPECT_EQ(run(solver), 0);
        // The first and last restarts are the run's own
        EXPECT_EQ(restart_steps(dir), repeat ? std::vector<uint64_t>({0, 3, 6, 9, 10}) : std::vector<uint64_t>({0, 3, 10}));
    }
}

TEST(TriggersTest, HeldConditionFiresAfterItsDurationCountedFromItsLastRise) {
    // True at steps 2-3 and from 5: held 2 ms from t = 5 ms, not from t = 2 ms (which would fire at 5)
    const std::string dir = fresh_dir("hold");
    Solver solver;
    solver.init(parse_toml(case_input(dir, "n_steps = 10\ndt = 0.001\n",
                                      "[[triggers]]\nwhen = \"step == 2 or step == 3 or step >= 5\"\n"
                                      "for = 0.002\ndo = \"checkpoint\"\n")));
    run(solver);
    EXPECT_EQ(restart_steps(dir), std::vector<uint64_t>({0, 7, 10}));
}

TEST(TriggersTest, WriteActionWritesTheNamedSnapshotAtThatStep) {
    const std::string dir = fresh_dir("write");
    Solver solver;
    solver.init(parse_toml(case_input(dir, "n_steps = 6\ncfl = 0.25\n",
                                      "[[write_data]]\nprefix = \"" + dir + "/snap\"\nformat = \"vtu\"\nvariables = [\"P\"]\ninterval = 1000\n"
                                      "[[triggers]]\nwhen = \"step == 4\"\ndo = [\"write " + dir + "/snap\"]\n")));
    run(solver);
    EXPECT_TRUE(fs::exists(dir + "/snap_000004.vtu") || fs::exists(dir + "/snap_000004.pvtu"));
    EXPECT_FALSE(fs::exists(dir + "/snap_000003.vtu") || fs::exists(dir + "/snap_000003.pvtu"));
    const std::string written = fs::exists(dir + "/snap_000004.vtu") ? dir + "/snap_000004.vtu" : dir + "/snap_000004.pvtu";
    const output::Contents file = output::open(written);
    ASSERT_EQ(file.snapshots.size(), 1u);
    EXPECT_EQ(file.snapshots[0].block.reason, "trigger");
}

TEST(TriggersTest, StartedStatisticsEqualThoseOfTheSameStartTime) {
    const std::string dir = fresh_dir("statistics");
    Solver triggered;
    triggered.init(parse_toml(case_input(dir + "/a", "n_steps = 10\ncfl = 0.25\n",
                                         "[statistics]\nfields = [\"P\"]\nproducts = [\"P*U_X\"]\n"
                                         "[[triggers]]\nwhen = \"step == 4\"\ndo = [\"start statistics\", \"checkpoint\"]\n")));
    run(triggered);
    const RestartData at_start = read_restart(restart_file(dir + "/a", 4));
    std::ostringstream t_start;
    t_start << std::setprecision(17) << at_start.t;
    Solver scheduled;
    scheduled.init(parse_toml(case_input(dir + "/b", "n_steps = 10\ncfl = 0.25\n",
                                         "[statistics]\nfields = [\"P\"]\nproducts = [\"P*U_X\"]\nt_start = " +
                                             t_start.str() + "\n")));
    run(scheduled);
    const RestartData a = read_restart(restart_file(dir + "/a", 10)), b = read_restart(restart_file(dir + "/b", 10));
    for (const char * name : {"MEAN_P", "COV_P_U_X"}) {
        ASSERT_NE(a.find(name), nullptr) << name;
        ASSERT_NE(b.find(name), nullptr) << name;
        EXPECT_EQ(*a.find(name), *b.find(name)) << name;
    }
    EXPECT_NE(*a.attribute("STATISTICS_WEIGHT"), 0.0);
}

TEST(TriggersTest, ResumedRunKeepsWhatFiredAndHowLongAConditionHeld) {
    // once fired at step 3 must not fire again at 6 or 9; held fires at 8, its clock started at 4 before the restart
    const std::string triggers = "[[triggers]]\nname = \"once\"\nwhen = \"step % 3 == 0\"\ndo = \"checkpoint\"\n"
                                 "[[triggers]]\nname = \"held\"\nwhen = \"step >= 4\"\nfor = 0.004\ndo = \"checkpoint\"\n";
    const std::string first = fresh_dir("resume_first"), second = fresh_dir("resume_second");
    Solver before;
    before.init(parse_toml(case_input(first, "n_steps = 6\ndt = 0.001\n", triggers)));
    run(before);
    EXPECT_EQ(restart_steps(first), std::vector<uint64_t>({0, 3, 6}));
    std::string input = case_input(second, "n_steps = 12\ndt = 0.001\n", triggers);
    const size_t begin = input.find("type = \"analytical\""), end = input.find("[[boundaries]]");
    input.replace(begin, end - begin, "type = \"restart\"\nfile = \"" + restart_file(first, 6) + "\"\n");
    Solver resumed;
    resumed.init(parse_toml(input));
    run(resumed);
    EXPECT_EQ(restart_steps(second), std::vector<uint64_t>({8, 12}));
}

TEST(TriggersTest, UnknownNameInAConditionIsAnInputErrorWithItsLine) {
    const std::string dir = fresh_dir("unknown");
    const std::string input = case_input(dir, "n_steps = 2\ncfl = 0.25\n",
                                         TRIGGER_CASE_PROBE + "[[triggers]]\nwhen = \"gauge_T > 2\"\ndo = \"stop\"\n");
    const size_t line = static_cast<size_t>(std::count(input.begin(), input.begin() + long(input.find("when =")), '\n')) + 1;
    Solver solver;
    try {
        solver.init(parse_toml(input));
        FAIL() << "no error";
    } catch (const InputError & e) {
        const std::string what = e.what();
        for (const std::string & part : {"line " + std::to_string(line) + ": triggers[0].when", std::string("'gauge_T'"),
                                       std::string("did you mean \"gauge_P\"")}) {
            EXPECT_NE(what.find(part), std::string::npos) << part << " not in: " << what;
        }
    }
}

TEST(TriggersTest, UnknownActionOrWriterIsAnInputError) {
    const std::string dir = fresh_dir("unknown_action");
    for (const char * action : {"halt", "write nowhere"}) {
        Solver solver;
        EXPECT_THROW(solver.init(parse_toml(case_input(dir, "n_steps = 2\ncfl = 0.25\n",
                                                       std::string("[[triggers]]\nwhen = \"t > 1\"\ndo = \"") + action +
                                                           "\"\n"))),
                     InputError)
            << action;
    }
}

namespace {

/** @brief The data rows of a CSV file, by step (header and comment lines left out). */
std::map<uint64_t, std::string> csv_rows(const std::string & file) {
    std::map<uint64_t, std::string> rows;
    std::ifstream in(file);
    for (std::string line; std::getline(in, line);) {
        if (!line.empty() && std::isdigit(static_cast<unsigned char>(line[0]))) rows[std::stoull(line)] = line;
    }
    return rows;
}

std::string convergence_case(const std::string & dir) {
    // The pressure ahead of the expansion is steady until it arrives
    return "[[probes]]\nname = \"far\"\npoint = [0.95, 0.5]\nvariables = [\"P\"]\ninterval = 1\nfile = \"" + dir +
           "/far.csv\"\n[statistics]\nfields = [\"P\"]\n[statistics.convergence]\nsignals = [\"far_P\"]\n"
           "min_samples = 10\nbatches = 4\ntolerance = 1e-3\nfile = \"" + dir + "/convergence.csv\"\n"
           "[[triggers]]\nname = \"steady\"\nwhen = \"convergence_stationary\"\ndo = \"start statistics\"\n";
}

} // namespace

TEST(TriggersTest, ConvergenceStartsStatisticsWhenStationaryAndResumesBitwise) {
    const std::string first = fresh_dir("convergence_first"), second = fresh_dir("convergence_second");
    Solver whole;
    whole.init(parse_toml(case_input(first, "n_steps = 40\ncfl = 0.25\n", convergence_case(first), 20)));
    EXPECT_EQ(run(whole), 0);
    const auto rows = csv_rows(first + "/convergence.csv");
    ASSERT_EQ(rows.size(), 41u);
    uint64_t stationary = 0;
    double t_stationary = 0.0;
    for (const auto & [step, row] : rows) {
        std::vector<std::string> cells;
        std::stringstream s(row);
        for (std::string cell; std::getline(s, cell, ',');) cells.push_back(cell);
        if (cells[2] == "1") {
            stationary = step;
            t_stationary = std::stod(cells[1]);
            break;
        }
    }
    ASSERT_GT(stationary, 0u);
    ASSERT_LT(stationary, 20u);
    const RestartData end = read_restart(restart_file(first, 40));
    ASSERT_NE(end.attribute("TRIGGER_steady_T_FIRST"), nullptr);
    EXPECT_EQ(*end.attribute("TRIGGER_steady_T_FIRST"), t_stationary);

    std::string input = case_input(second, "n_steps = 40\ncfl = 0.25\n", convergence_case(second), 20);
    const size_t begin = input.find("type = \"analytical\""), stop = input.find("[[boundaries]]");
    input.replace(begin, stop - begin, "type = \"restart\"\nfile = \"" + restart_file(first, 20) + "\"\n");
    Solver resumed;
    resumed.init(parse_toml(input));
    EXPECT_EQ(run(resumed), 0);
    const auto continued = csv_rows(second + "/convergence.csv");
    ASSERT_EQ(continued.size(), 20u);
    for (const auto & [step, row] : continued) EXPECT_EQ(row, rows.at(step)) << step;
}

TEST(TriggersTest, UnknownConvergenceSignalIsAnInputErrorListingTheMonitoredValues) {
    const std::string dir = fresh_dir("convergence_unknown");
    Solver solver;
    try {
        solver.init(parse_toml(case_input(dir, "n_steps = 2\ncfl = 0.25\n",
                                          TRIGGER_CASE_PROBE + "[statistics.convergence]\nsignals = [\"gauge_T\"]\n")));
        FAIL() << "no error";
    } catch (const InputError & e) {
        const std::string what = e.what();
        EXPECT_NE(what.find("statistics.convergence.signals"), std::string::npos) << what;
        EXPECT_NE(what.find("did you mean"), std::string::npos) << what;
        EXPECT_NE(what.find("gauge_P"), std::string::npos) << what;
    }
}

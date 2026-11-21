/**
 * @file session_test.cpp
 * @brief mallard session answers as its one-shot commands do, rereads a mesh only when it changes, answers every
 *        message of its protocol, and all its output conforms to the schemas of schemas/.
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 */

#include <gtest/gtest.h>

#include <cctype>
#include <cmath>
#include <chrono>
#include <csignal>
#include <filesystem>
#include <fstream>
#include <map>
#include <numbers>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include <fcntl.h>
#include <sys/wait.h>
#include <unistd.h>

#include "cli.h"
#include "comm.h"
#include "common_typedef.h"
#include "data_paths.h"
#include "json.h"
#include "json_schema.h"
#include "provenance.h"
#include "run_control.h"
#include "session/session.h"

extern char ** environ;

namespace {

namespace fs = std::filesystem;

#define SKIP_UNDER_MPI() \
    if (comm::size() > 1) GTEST_SKIP() << "runs mallard itself"

struct Result {
    int status = -1;
    std::string out, err;
};

std::string read_file(const fs::path & path) {
    std::ifstream in(path);
    std::stringstream text;
    text << in.rdbuf();
    return text.str();
}

std::string mallard() {
    const fs::path solver = mallard_executable();
    const fs::path front_end = solver.parent_path() / "mallard";
    return fs::exists(front_end) ? front_end.string() : solver.string();
}

/** @brief Runs mallard with these arguments and this standard input, outside any MPI launcher; killed after two
 *         minutes. hold_input: the input stays open until mallard exits; kokkos: with one thread. */
Result run(std::vector<std::string> args, const std::string & input = "", bool hold_input = false, bool kokkos = true) {
    args.insert(args.begin(), mallard());
    if (kokkos) args.push_back("--kokkos-num-threads=1");
    std::vector<std::string> env;
    for (char ** e = environ; *e; e++) {
        const std::string var = *e;
        bool launcher = false;
        for (const char * prefix : {"OMPI_", "PMIX_", "PRTE_", "PMI_", "HYDRA_", "MPIR_", "I_MPI_"}) {
            launcher = launcher || var.rfind(prefix, 0) == 0;
        }
        if (!launcher) env.push_back(var);
    }
    std::vector<char *> argv, envp;
    for (auto & a : args) argv.push_back(a.data());
    for (auto & e : env) envp.push_back(e.data());
    argv.push_back(nullptr);
    envp.push_back(nullptr);
    int to_child[2];
    if (pipe(to_child) != 0) return {};
    const pid_t pid = fork();
    if (pid == 0) {
        setpgid(0, 0);
        const int out = open("out.txt", O_WRONLY | O_CREAT | O_TRUNC, 0644);
        const int err = open("err.txt", O_WRONLY | O_CREAT | O_TRUNC, 0644);
        if (out < 0 || err < 0) _exit(126);
        dup2(to_child[0], 0);
        close(to_child[0]);
        close(to_child[1]);
        dup2(out, 1);
        dup2(err, 2);
        execve(argv[0], argv.data(), envp.data());
        _exit(127);
    }
    close(to_child[0]);
    Result r;
    if (pid < 0) {
        close(to_child[1]);
        return r;
    }
    signal(SIGPIPE, SIG_IGN);
    for (size_t done = 0; done < input.size();) {
        const ssize_t n = write(to_child[1], input.data() + done, input.size() - done);
        if (n <= 0) break;
        done += static_cast<size_t>(n);
    }
    if (!hold_input) close(to_child[1]);
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::minutes(2);
    int wstatus = 0;
    bool hung = false;
    while (waitpid(pid, &wstatus, WNOHANG) == 0) {
        if (std::chrono::steady_clock::now() > deadline) {
            hung = true;
            kill(-pid, SIGKILL);
            waitpid(pid, &wstatus, 0);
            break;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    if (hold_input) close(to_child[1]);
    if (!hung && WIFEXITED(wstatus)) r.status = WEXITSTATUS(wstatus);
    r.out = read_file("out.txt");
    r.err = read_file("err.txt");
    if (hung) r.err += "\n(killed: no exit within two minutes)";
    return r;
}

/** @brief A request line. */
std::string request(int id, const std::string & method, const std::string & params = "{}") {
    return "{\"jsonrpc\":\"2.0\",\"id\":" + std::to_string(id) + ",\"method\":\"" + method + "\",\"params\":" + params +
           "}\n";
}

/** @brief The messages a session wrote, one per line. */
std::vector<json::Value> messages(const std::string & out) {
    std::vector<json::Value> list;
    std::istringstream lines(out);
    for (std::string line; std::getline(lines, line);) list.push_back(json::parse(line));
    return list;
}

/** @brief The response to the request id; a null value if there is none. */
json::Value response(const std::vector<json::Value> & list, int id) {
    for (const json::Value & m : list) {
        if (m.has("id") && !m.has("method") && m["id"].type == json::Value::Type::NUMBER && m["id"].number == id) return m;
    }
    return {};
}

const json_schema::Schemas & schemas() {
    static const json_schema::Schemas s(source_dir() + "/schemas");
    return s;
}

const json_schema::Schemas & schemas();

/** @brief The file of the schema named, of whatever version schemas/ has. */
std::string schema_file(const std::string & name) {
    for (const std::string & file : schemas().files()) {
        if (file.rfind(name + ".v", 0) == 0) return file;
    }
    return name + ".v1.schema.json";
}

const std::string MESSAGE = "mallard-session.v1.schema.json#/$defs/message";

/** @brief Where document breaks the schema named (or MESSAGE), joined; empty if it conforms. */
std::string violations(const std::string & name, const json::Value & document) {
    std::string text;
    for (const std::string & e : schemas().validate(name == MESSAGE ? name : schema_file(name), document)) text += e + "\n";
    return text;
}

/** @brief A report without what differs from run to run: the memory measured. */
std::string comparable(json::Value report) {
    for (size_t i = 0; i < report.names.size(); i++) {
        if (report.names[i] == "memory") report.members[i] = json::Value();
    }
    return json::dump(report);
}

std::string quoted(const std::string & text) { return json::quote(text); }

/** @brief A flow past a cylinder read from mesh. */
std::string cylinder_input(const std::string & mesh, double cfl = 0.5) {
    const bool three = Mallard_DIM == 3;
    std::ostringstream input;
    const std::string u = three ? "[0.2, 0.0, 0.0]" : "[0.2, 0.0]";
    input << "[run]\nn_steps = 4\ncfl = " << cfl << "\n[mesh]\ntype = \"file\"\nfilename = \"" << mesh << "\"\n"
          << "[initialize]\ntype = \"constant\"\nu = " << u << "\np = 0.7142857142857143\nT = 0.7142857142857143\n"
          << "[physics]\ntype = \"euler\"\ngamma = 1.4\np_ref = 0.7142857142857143\nT_ref = 0.7142857142857143\n"
          << "rho_ref = 1.0\n";
    for (const char * zone : {"inlet", "outlet"}) {
        input << "[[boundaries]]\nname = \"" << zone << "\"\ntype = \"farfield\"\nu = " << u
              << "\np = 0.7142857142857143\nT = 0.7142857142857143\n";
    }
    for (const std::string zone : {"walls", "cylinder", "back", "front"}) {
        if (three || (zone != "back" && zone != "front")) input << "[[boundaries]]\nname = \"" << zone << "\"\ntype = \"symmetry\"\n";
    }
    return input.str();
}

void write_cylinder_case() {
    const std::string mesh = source_dir() + (Mallard_DIM == 3 ? "/test/data/meshes/cyl3d.msh" : "/test/data/meshes/cyl2d.msh");
    fs::copy_file(mesh, "cyl.msh", fs::copy_options::overwrite_existing);
    fs::copy_file(mesh, "other.msh", fs::copy_options::overwrite_existing);
    std::ofstream("case.toml") << cylinder_input("cyl.msh");
}

/** @brief Every message conforms to the protocol's schema, and every result to the schema of its method. */
void expect_conforms(const std::vector<json::Value> & list, const std::vector<std::pair<int, std::string>> & results) {
    for (const json::Value & m : list) EXPECT_EQ(violations(MESSAGE, m), "") << json::dump(m).substr(0, 400);
    for (const auto & [id, name] : results) {
        const json::Value r = response(list, id);
        ASSERT_TRUE(r.has("result")) << id << ": " << json::dump(r);
        EXPECT_EQ(violations(name, r["result"]), "") << name;
    }
}

} // namespace

TEST(SessionTest, AnswersAsItsOneShotCommandsDo) {
    SKIP_UNDER_MPI();
    write_cylinder_case();
    const Result session = run({"session", "case.toml"},
                               request(1, "initialize") + request(2, "case.check") + request(3, "case.keys") +
                                   request(4, "keys.catalog", "{\"section\":\"numerics\"}") + request(5, "shutdown"));
    ASSERT_EQ(session.status, 0) << session.err;
    const std::vector<json::Value> list = messages(session.out);
    const json::Value init = response(list, 1)["result"];
    EXPECT_EQ(init["dimension"].number, Mallard_DIM);
    EXPECT_EQ(init["case"]["input"].string, "case.toml");
    std::vector<std::pair<int, std::string>> results = {{1, "mallard-capabilities"}};
    for (const json::Value & m : init["methods"].array) {
        if (m["name"].string == "case.check") results.emplace_back(2, m["result"].string);
        if (m["name"].string == "case.keys") results.emplace_back(3, m["result"].string);
        if (m["name"].string == "keys.catalog") results.emplace_back(4, m["result"].string);
    }
    ASSERT_EQ(results.size(), 4u) << json::dump(init);
    expect_conforms(list, results);

    const Result check = run({"check", "case.toml", "--format", "json"});
    ASSERT_EQ(check.status, 0) << check.err;
    EXPECT_EQ(comparable(response(list, 2)["result"]), comparable(json::parse(check.out)));
    EXPECT_EQ(violations("mallard-check", json::parse(check.out)), "");
    const Result keys = run({"list", "keys", "case.toml", "--format", "json"});
    ASSERT_EQ(keys.status, 0) << keys.err;
    EXPECT_EQ(json::dump(response(list, 3)["result"]), json::dump(json::parse(keys.out)));
    EXPECT_EQ(violations("mallard-key-catalog", json::parse(keys.out)), "");
    const Result catalog = run({"list", "keys", "numerics", "--format", "json"});
    ASSERT_EQ(catalog.status, 0) << catalog.err;
    EXPECT_EQ(json::dump(response(list, 4)["result"]), json::dump(json::parse(catalog.out)));
    // Progress names the request it reports on
    bool progress = false;
    for (const json::Value & m : list) progress = progress || (m.has("method") && m["method"].string == "$/progress" && m["params"]["id"].number == 2);
    EXPECT_TRUE(progress) << session.out;
}

TEST(SessionTest, InitializeAnswersAsVersionJsonWithTheDevicesAndTheSettingsOfControl) {
    SKIP_UNDER_MPI();
    const Result session = run({"session"}, request(1, "initialize") + request(2, "shutdown"));
    ASSERT_EQ(session.status, 0) << session.err;
    json::Value init = response(messages(session.out), 1)["result"];
    EXPECT_EQ(violations("mallard-capabilities", init), "");
    EXPECT_TRUE(init["case"].is_null());
    const Result version = run({"--dim", std::to_string(Mallard_DIM), "version", "--format", "json"}, "", false, false);
    ASSERT_EQ(version.status, 0) << version.err;
    const json::Value one_shot = json::parse(version.out);
    EXPECT_EQ(violations("mallard-capabilities", one_shot), "");
    for (size_t i = 0; i < init.names.size(); i++) {
        if (init.names[i] != "case") continue;
        init.names.erase(init.names.begin() + static_cast<std::ptrdiff_t>(i));
        init.members.erase(init.members.begin() + static_cast<std::ptrdiff_t>(i));
    }
    EXPECT_EQ(json::dump(init), json::dump(one_shot));

    const json::Value & devices = one_shot["devices"];
    EXPECT_FALSE(devices["backend"].string.empty());
    EXPECT_EQ(devices["count"].number, static_cast<double>(devices["visible"].array.size()));
    // ctl checks set against the table the run applies, and capabilities list that table
    std::vector<std::string> listed, table;
    for (const json::Value & s : one_shot["control"]["settings"].array) listed.push_back(s["key"].string);
    for (const run_control::Setting & s : run_control::SETTINGS) table.push_back(s.key());
    EXPECT_EQ(listed, table);
    std::vector<std::string> methods;
    for (const json::Value & m : one_shot["methods"].array) methods.push_back(m["name"].string);
    EXPECT_EQ(methods.size(), session::methods().size());
}

TEST(SessionTest, CaseViewAnswersAsCheckViewWithTheFrameOfEachBoundary) {
    SKIP_UNDER_MPI();
    write_cylinder_case();
    const Result session = run({"session", "case.toml"}, request(1, "case.view", "{\"dir\":\"view\"}") +
                                                             request(2, "case.view") + request(3, "shutdown"));
    ASSERT_EQ(session.status, 0) << session.err;
    const std::vector<json::Value> list = messages(session.out);
    expect_conforms(list, {{1, "mallard-case-view"}});
    EXPECT_EQ(response(list, 2)["error"]["code"].number, session::code::INVALID_PARAMS);
    const json::Value view = response(list, 1)["result"];
    const Result check = run({"check", "case.toml", "--view", "view", "--format", "json"});
    ASSERT_EQ(check.status, 0) << check.err;
    EXPECT_EQ(json::dump(view), json::dump(json::parse(check.out)["view"]));
    EXPECT_EQ(violations("mallard-check", json::parse(check.out)), "");
    for (const json::Value & f : view["files"].array) EXPECT_TRUE(fs::exists(f.string)) << f.string;

    std::map<std::string, json::Value> frames;
    for (const json::Value & f : view["frames"].array) frames[f["zone"].string] = f;
    ASSERT_EQ(frames.size(), Mallard_DIM == 3 ? 6u : 4u) << json::dump(view);
    for (const auto & [zone, f] : frames) {
        if (!f["error"].is_null()) continue;
        double along = 0.0, u2 = 0.0, n2 = 0.0;
        for (size_t d = 0; d < 3; d++) {
            along += f["u_direction"].array[d].number * f["normal"].array[d].number;
            u2 += f["u_direction"].array[d].number * f["u_direction"].array[d].number;
            n2 += f["normal"].array[d].number * f["normal"].array[d].number;
        }
        EXPECT_NEAR(u2, 1.0, 1e-5) << zone;
        EXPECT_NEAR(n2, 1.0, 1e-5) << zone;
        EXPECT_NEAR(along, 0.0, 1e-5) << zone;  // u runs along the boundary
        EXPECT_GT(f["length"].number, 0.0) << zone;
    }
    // The inlet, at the domain's low x, faces +x
    ASSERT_TRUE(frames["inlet"]["error"].is_null()) << json::dump(frames["inlet"]);
    EXPECT_NEAR(frames["inlet"]["normal"].array[0].number, 1.0, 1e-5);
    EXPECT_FALSE(frames["cylinder"]["error"].is_null() && Mallard_DIM == 2) << "a closed curve has no start";

    if (Mallard_DIM == 2) {
        // origin sets the end u starts from
        std::string flipped = cylinder_input("cyl.msh");
        const std::string inlet = "name = \"inlet\"\ntype = \"farfield\"\nu = [0.2, 0.0]\np = 0.7142857142857143\nT = 0.7142857142857143\n";
        ASSERT_NE(flipped.find(inlet), std::string::npos);
        flipped.replace(flipped.find(inlet), inlet.size(),
                        "name = \"inlet\"\ntype = \"farfield\"\nu = [0.2, 0.0]\np = 0.7142857142857143\n"
                        "T = \"0.7142857142857143 * (1 + 0.1 * u)\"\norigin = [0.0, 1.0]\n");
        const Result edited = run({"session", "case.toml"}, request(1, "case.view", "{\"dir\":\"view\"}") +
                                                                request(2, "case.update", "{\"text\":" + json::quote(flipped) + "}") +
                                                                request(3, "case.view", "{\"dir\":\"view\"}"));
        ASSERT_EQ(edited.status, 0) << edited.err;
        const std::vector<json::Value> edits = messages(edited.out);
        const auto inlet_frame = [&](int id) {
            const json::Value r = response(edits, id);
            for (const json::Value & f : r["result"]["frames"].array) {
                if (f["zone"].string == "inlet") return f;
            }
            return json::Value();
        };
        const json::Value before = inlet_frame(1), after = inlet_frame(3);
        ASSERT_FALSE(before.is_null() || after.is_null()) << edited.out;
        EXPECT_EQ(json::dump(after["normal"]), json::dump(before["normal"]));
        EXPECT_NEAR(after["u_direction"].array[1].number, -before["u_direction"].array[1].number, 1e-6);
        EXPECT_NEAR(after["origin"].array[1].number - before["origin"].array[1].number,
                    before["u_direction"].array[1].number * before["length"].number, 1e-5);
    }
}

/** @brief A CSV file as columns by name. */
std::map<std::string, std::vector<double>> csv_columns(const std::string & path) {
    std::ifstream in(path);
    std::string line, cell;
    std::getline(in, line);
    std::vector<std::string> names;
    for (std::istringstream header(line); std::getline(header, cell, ',');) names.push_back(cell);
    std::map<std::string, std::vector<double>> columns;
    while (std::getline(in, line)) {
        std::istringstream row(line);
        for (const std::string & name : names) {
            std::getline(row, cell, ',');
            columns[name].push_back(cell == "null" ? std::nan("") : std::stod(cell));
        }
    }
    return columns;
}

TEST(SessionTest, BoundaryValuesAreTheBoundariesOwnWithTheInflowOfMassFlowInlets) {
    SKIP_UNDER_MPI();
    write_cylinder_case();
    std::string text = cylinder_input("cyl.msh");
    const std::string u = Mallard_DIM == 3 ? "[0.2, 0.0, 0.0]" : "[0.2, 0.0]";
    const std::string inlet = "name = \"inlet\"\ntype = \"farfield\"\nu = " + u + "\np = 0.7142857142857143\nT = 0.7142857142857143\n";
    const std::string outlet = "name = \"outlet\"\ntype = \"farfield\"\nu = " + u + "\np = 0.7142857142857143\nT = 0.7142857142857143\n";
    ASSERT_NE(text.find(inlet), std::string::npos);
    text.replace(text.find(inlet), inlet.size(),
                 "name = \"inlet\"\ntype = \"farfield\"\nu = " + u + "\np = 0.7142857142857143\n"
                 "T = \"0.7142857142857143 * (1 + 0.1 * u)\"\n");
    ASSERT_NE(text.find(outlet), std::string::npos);
    text.replace(text.find(outlet), outlet.size(),
                 "name = \"outlet\"\ntype = \"inlet_mass_flow\"\nmass_flux = 0.15\nT = 0.7142857142857143\n");
    std::ofstream("case.toml") << text;
    const Result session = run({"session", "case.toml"}, request(1, "case.boundary_values", "{\"dir\":\"values\"}"));
    ASSERT_EQ(session.status, 0) << session.err;
    const std::vector<json::Value> list = messages(session.out);
    expect_conforms(list, {{1, "mallard-boundary-values"}});
    const json::Value values = response(list, 1)["result"];
    const Result check = run({"check", "case.toml", "--boundary-values", "values", "--format", "json"});
    ASSERT_EQ(check.status, 0) << check.err;
    EXPECT_EQ(json::dump(values), json::dump(json::parse(check.out)["boundary_values"]));
    EXPECT_EQ(violations("mallard-check", json::parse(check.out)), "");

    std::map<std::string, json::Value> entries;
    for (const json::Value & e : values["entries"].array) entries[e["zone"].string] = e;
    ASSERT_EQ(entries.size(), Mallard_DIM == 3 ? 6u : 4u) << json::dump(values);
    for (const auto & [zone, e] : entries) {
        const auto columns = csv_columns(e["file"].string);
        ASSERT_EQ(columns.size(), e["columns"].array.size() + 1) << zone;
        EXPECT_EQ(columns.at("face").size(), static_cast<size_t>(e["faces"].number)) << zone;
    }
    // The profile, as the boundary evaluates it at each face from its own u
    const auto in = csv_columns(entries["inlet"]["file"].string);
    ASSERT_TRUE(in.count("T") && in.count("u")) << json::dump(entries["inlet"]);
    for (size_t k = 0; k < in.at("T").size(); k++) {
        EXPECT_NEAR(in.at("T")[k], 0.7142857142857143 * (1 + 0.1 * in.at("u")[k]), 1e-6);
    }
    // The mass-flow inlet's flux makes the prescribed mass flux of the initial state
    const auto out = csv_columns(entries["outlet"]["file"].string);
    ASSERT_TRUE(out.count("rho_u_n") && out.count("speed")) << json::dump(entries["outlet"]);
    for (size_t k = 0; k < out.at("rho_u_n").size(); k++) {
        EXPECT_NEAR(out.at("rho_u_n")[k], 0.15, 1e-6);
        EXPECT_GT(out.at("speed")[k], 0.0);
    }
    for (const json::Value & c : entries["outlet"]["columns"].array) {
        EXPECT_EQ(c["estimated"].boolean, c["name"].string == "rho_u_n" || c["name"].string == "speed" ||
                                              c["name"].string == "rho" || c["name"].string == "p" ||
                                              c["name"].string.rfind("u_", 0) == 0) << c["name"].string;
    }
}

TEST(SessionTest, UnitsCatalogAnswersAsListUnitsWithEachFactorAsDefined) {
    SKIP_UNDER_MPI();
    const Result session = run({"session"}, request(1, "units.catalog") + request(2, "shutdown"));
    ASSERT_EQ(session.status, 0) << session.err;
    const std::vector<json::Value> list = messages(session.out);
    expect_conforms(list, {{1, "mallard-unit-catalog"}});
    const Result one_shot = run({"list", "units", "--format", "json"});
    ASSERT_EQ(one_shot.status, 0) << one_shot.err;
    const json::Value catalog = json::parse(one_shot.out);
    EXPECT_EQ(json::dump(response(list, 1)["result"]), json::dump(catalog));
    EXPECT_EQ(violations("mallard-unit-catalog", catalog), "");
    // The exact definition is the factor Mallard converts with
    const auto decimal = [](const std::string & text) { return text == "pi" ? std::numbers::pi : std::stod(text); };
    for (const json::Value & u : catalog["units"].array) {
        const std::string & factor = u["factor"].string;
        const size_t slash = factor.find('/');
        const double exact = slash == std::string::npos
                                 ? decimal(factor)
                                 : decimal(factor.substr(0, slash)) / decimal(factor.substr(slash + 1));
        EXPECT_DOUBLE_EQ(exact, u["value"].number) << u["symbol"].string << " = " << factor;
    }
    const Result text = run({"list", "units"});
    ASSERT_EQ(text.status, 0) << text.err;
    EXPECT_NE(text.out.find("101325/760"), std::string::npos) << text.out;
}

TEST(SessionTest, FileInfoAnswersAsInfoJsonWithWhetherThisBuildContinuesARestart) {
    SKIP_UNDER_MPI();
    const std::string restart = source_dir() + "/test/data/formats/v0.8.0/gas_000002.restart";  // 2D, double
    const Result session = run({"session"}, request(1, "file.info", "{\"file\":" + json::quote(restart) + "}") +
                                                request(2, "file.info", "{\"file\":\"no_such.restart\"}") +
                                                request(3, "file.info", "{\"file\":\".\"}") + request(4, "shutdown"));
    ASSERT_EQ(session.status, 0) << session.err;
    const std::vector<json::Value> list = messages(session.out);
    expect_conforms(list, {{1, "mallard-file-info"}});
    EXPECT_EQ(response(list, 2)["error"]["code"].number, session::code::UNREADABLE);
    EXPECT_EQ(response(list, 3)["error"]["code"].number, session::code::INVALID_PARAMS);
    const Result one_shot = run({"info", restart, "--format", "json"});
    ASSERT_EQ(one_shot.status, 0) << one_shot.err;
    const json::Value info = json::parse(one_shot.out);
    EXPECT_EQ(json::dump(response(list, 1)["result"]), json::dump(info));
    EXPECT_EQ(violations("mallard-file-info", info), "");
    // The reasons are those the run gives when it refuses the file
    std::vector<std::string> expected;
    if (sizeof(rtype) != 8) expected.push_back("Restart file " + restart + " was written with a different floating-point precision.");
    if (Mallard_DIM == 3) expected.push_back("Restart file " + restart + " holds a 2D run, but Mallard was built with Mallard_DIM = 3.");
    EXPECT_EQ(info["continuable"].boolean, expected.empty());
    ASSERT_EQ(info["incompatibilities"].array.size(), expected.size()) << one_shot.out;
    for (size_t k = 0; k < expected.size(); k++) EXPECT_EQ(info["incompatibilities"].array[k].string, expected[k]);
    const Result text = run({"info", restart});
    EXPECT_NE(text.out.find(expected.empty() ? "yes, by this build" : "no, by this build"), std::string::npos) << text.out;
}

TEST(SessionTest, CaseFilesAnswersAsFilesJsonWithEveryFileTheSetupRead) {
    SKIP_UNDER_MPI();
    write_cylinder_case();
    const Result session = run({"session", "case.toml"}, request(1, "case.files", "{\"sha256\":true}") +
                                                             request(2, "case.files") + request(3, "shutdown"));
    ASSERT_EQ(session.status, 0) << session.err;
    const std::vector<json::Value> list = messages(session.out);
    expect_conforms(list, {{1, "mallard-case-files"}, {2, "mallard-case-files"}});
    const Result one_shot = run({"files", "case.toml", "--sha256", "--format", "json"});
    ASSERT_EQ(one_shot.status, 0) << one_shot.err;
    const json::Value files = json::parse(one_shot.out);
    EXPECT_EQ(json::dump(response(list, 1)["result"]), json::dump(files));
    EXPECT_EQ(violations("mallard-case-files", files), "");
    EXPECT_TRUE(files["complete"].boolean);
    const auto & entries = files["files"].array;
    ASSERT_EQ(entries.size(), 2u) << one_shot.out;
    EXPECT_EQ(entries[0]["role"].string, "input");
    EXPECT_EQ(entries[0]["path"].string, "case.toml");
    EXPECT_EQ(entries[1]["role"].string, "mesh");
    EXPECT_EQ(entries[1]["path"].string, "cyl.msh");
    EXPECT_EQ(entries[1]["resolved"].string, fs::absolute("cyl.msh").lexically_normal().string());
    EXPECT_EQ(entries[1]["bytes"].number, static_cast<double>(fs::file_size("cyl.msh")));
    EXPECT_EQ(entries[1]["sha256"].string, provenance::sha256_file("cyl.msh"));
    // Without sha256, the same files unhashed
    const json::Value plain = response(list, 2)["result"];
    EXPECT_TRUE(plain["files"].array[1]["sha256"].is_null());
    EXPECT_EQ(plain["files"].array[1]["bytes"].number, entries[1]["bytes"].number);
}

TEST(SessionTest, EditsRereadTheMeshOnlyWhenItChanges) {
    SKIP_UNDER_MPI();
    write_cylinder_case();
    const std::string faster = cylinder_input("cyl.msh", 0.25), moved = cylinder_input("other.msh", 0.25);
    const Result session = run({"session"}, request(1, "case.open", "{\"input\":\"case.toml\"}") +
                                                request(2, "case.check") +
                                                request(3, "case.update", "{\"text\":" + quoted(faster) + "}") +
                                                request(4, "case.check") +
                                                request(5, "case.update", "{\"text\":" + quoted(moved) + "}") +
                                                request(6, "case.check") + request(7, "initialize"));
    ASSERT_EQ(session.status, 0) << session.err;
    const std::vector<json::Value> list = messages(session.out);
    expect_conforms(list, {{1, "mallard-case"}, {2, "mallard-check"}, {3, "mallard-case"}, {4, "mallard-check"},
                           {5, "mallard-case"}, {6, "mallard-check"}});
    const json::Value edit = response(list, 3)["result"];
    ASSERT_EQ(edit["changed"].array.size(), 1u) << json::dump(edit);
    EXPECT_EQ(edit["changed"].array[0].string, "run.cfl");
    EXPECT_EQ(edit["mesh_files_read"].number, 1);
    EXPECT_EQ(response(list, 5)["result"]["mesh_files_read"].number, 1);
    EXPECT_EQ(response(list, 7)["result"]["case"]["mesh_files_read"].number, 2);

    // The check of an edit is the one-shot check of the edited file
    std::ofstream("case.toml") << faster;
    const Result check = run({"check", "case.toml", "--format", "json"});
    ASSERT_EQ(check.status, 0) << check.err;
    EXPECT_EQ(comparable(response(list, 4)["result"]), comparable(json::parse(check.out)));
    EXPECT_NE(comparable(response(list, 2)["result"]), comparable(json::parse(check.out)));
}

TEST(SessionTest, AnswersEveryMessageOfItsProtocol) {
    SKIP_UNDER_MPI();
    write_cylinder_case();
    const Result session =
        run({"session"}, std::string("not json\n") + "[" + request(1, "initialize").substr(0, request(1, "initialize").size() - 1) +
                             "]\n" + "{\"jsonrpc\":\"2.0\",\"method\":\"no.such\"}\n" + request(2, "no.such") +
                             request(3, "case.check") + request(4, "case.open", "{\"input\":\"missing.toml\"}") +
                             request(5, "keys.catalog", "{\"no_such\":true}") +
                             request(6, "case.open", "{\"input\":\"case.toml\"}") + request(7, "case.check") +
                             request(8, "case.check", "{\"partial\":true}") +
                             "{\"jsonrpc\":\"2.0\",\"method\":\"$/cancel\",\"params\":{\"id\":8}}\n");
    // The end of the input ends the session as shutdown does
    ASSERT_EQ(session.status, 0) << session.err;
    const std::vector<json::Value> list = messages(session.out);
    for (const json::Value & m : list) EXPECT_EQ(violations(MESSAGE, m), "") << json::dump(m);
    std::vector<int> codes;
    for (const json::Value & m : list) {
        if (m.has("error") && m["id"].is_null()) codes.push_back(int(m["error"]["code"].number));
    }
    EXPECT_EQ(codes, (std::vector<int>{session::code::PARSE, session::code::INVALID_REQUEST}));
    const auto code_of = [&](int id) { return int(response(list, id)["error"]["code"].number); };
    EXPECT_EQ(code_of(2), session::code::NO_METHOD);
    EXPECT_EQ(code_of(3), session::code::NO_CASE);
    EXPECT_EQ(code_of(4), session::code::UNREADABLE);
    EXPECT_EQ(code_of(5), session::code::INVALID_PARAMS);
    EXPECT_TRUE(response(list, 7).has("result"));
    EXPECT_EQ(code_of(8), session::code::CANCELED);
    // A notification is never answered
    size_t answers = 0;
    for (const json::Value & m : list) answers += !m.has("method");
    EXPECT_EQ(answers, 9u) << session.out;
}

TEST(SessionTest, ShutdownEndsTheSessionWhileItsInputStaysOpen) {
    SKIP_UNDER_MPI();
    const Result session = run({"session"}, request(1, "initialize") + request(2, "shutdown"), true);
    EXPECT_EQ(session.status, 0) << session.err;
    EXPECT_TRUE(response(messages(session.out), 2)["result"].is_null()) << session.out;
}

TEST(SessionTest, EveryRequestNamesAOneShotCommandAndASchemaThatExist) {
    for (const session::Method & m : session::methods()) {
        if (!m.result.empty()) {
            EXPECT_TRUE(fs::exists(source_dir() + "/schemas/" + schema_file(std::string(m.result)))) << m.name;
        }
        if (m.command.empty()) continue;
        std::istringstream words{std::string(m.command)};
        std::string word, name;
        words >> word;
        ASSERT_EQ(word, "mallard") << m.command;
        while (words >> word && word.find_first_of("[]-") != 0 && !std::isupper(static_cast<unsigned char>(word[0]))) {
            name += (name.empty() ? "" : " ") + word;
        }
        EXPECT_NE(cli::command(name), nullptr) << m.name << ": " << m.command;
    }
}

TEST(SessionTest, SchemasUseOnlyWhatTheConformanceTestsValidate) {
    ASSERT_FALSE(schemas().files().empty());
    for (const std::string & file : schemas().files()) {
        std::string unsupported;
        for (const std::string & u : schemas().unsupported(file)) unsupported += u + "\n";
        EXPECT_EQ(unsupported, "") << file;
    }
    // and what they validate is checked
    EXPECT_NE(violations("mallard-check", json::parse("{\"version\": 1}")), "");
    EXPECT_NE(violations(MESSAGE, json::parse("{\"jsonrpc\": \"2.0\", \"id\": 1}")), "");
    EXPECT_NE(violations("mallard-case", json::parse(R"({"schema": "mallard-case", "version": 1, "input": "a.toml",
                                                        "dimension": 2, "mesh_files_read": -1})")), "");
}

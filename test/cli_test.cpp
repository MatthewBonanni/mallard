/**
 * @file cli_test.cpp
 * @brief Every command of the mallard executable has a help page that covers
 *        its options, usage errors point at it, and the CLI docs match it.
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 */

#include <gtest/gtest.h>

#include <algorithm>
#include <chrono>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <limits>
#include <map>
#include <regex>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include <fcntl.h>
#include <sys/wait.h>
#include <unistd.h>

#include "archive.h"
#include "build_info.h"
#include "cli.h"
#include "comm.h"
#include "common_typedef.h"
#include "data_paths.h"
#include "json.h"
#include "output_reader.h"

extern char ** environ;

namespace {

namespace fs = std::filesystem;

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

/** @brief The front end next to the solver of this build, else the solver. */
std::string mallard() {
    const fs::path solver = mallard_executable();
    const fs::path front_end = solver.parent_path() / "mallard";
    return fs::exists(front_end) ? front_end.string() : solver.string();
}

/**
 * @brief Runs mallard with these arguments, outside any MPI launcher this test runs under; killed after a minute.
 *        launched: under the MPI launcher of the build, for commands that start MPI (an MPI library need not
 *        start a process without its launcher).
 */
Result run(const std::vector<std::string> & args, bool launched = false) {
    std::vector<std::string> command;
#ifdef MALLARD_MPIEXEC
    if (launched) command = {MALLARD_MPIEXEC, "-n", "1"};
#else
    (void)launched;
#endif
    command.push_back(mallard());
    command.insert(command.end(), args.begin(), args.end());
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
    for (auto & a : command) argv.push_back(a.data());
    for (auto & e : env) envp.push_back(e.data());
    argv.push_back(nullptr);
    envp.push_back(nullptr);
    const pid_t pid = fork();
    if (pid == 0) {
        setpgid(0, 0);
        const int out = open("out.txt", O_WRONLY | O_CREAT | O_TRUNC, 0644);
        const int err = open("err.txt", O_WRONLY | O_CREAT | O_TRUNC, 0644);
        if (out < 0 || err < 0) _exit(126);
        dup2(out, 1);
        dup2(err, 2);
        execve(argv[0], argv.data(), envp.data());
        _exit(127);
    }
    Result r;
    if (pid < 0) return r;
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::minutes(1);
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
    if (!hung && WIFEXITED(wstatus)) r.status = WEXITSTATUS(wstatus);
    r.out = read_file("out.txt");
    r.err = read_file("err.txt");
    if (hung) r.err += "\n(killed: no exit within a minute)";
    return r;
}

std::vector<std::string> words(std::string_view name) {
    std::vector<std::string> out;
    std::istringstream in{std::string(name)};
    for (std::string w; in >> w;) out.push_back(w);
    return out;
}

bool contains(const std::string & text, std::string_view part) { return text.find(part) != std::string::npos; }

/** @brief A time as mallard info prints it: the run's time in this build's precision, %.9g. */
std::string printed_time(double t) {
    char out[32];
    std::snprintf(out, sizeof out, "%.9g", static_cast<double>(static_cast<rtype>(t)));
    return out;
}

/** @brief Whether a line of text starts with these cells, columns apart (two spaces or more). */
bool has_row(const std::string & text, const std::vector<std::string> & cells) {
    std::istringstream in(text);
    for (std::string line; std::getline(in, line);) {
        std::vector<std::string> found;
        size_t at = line.find_first_not_of(' ');
        while (at != std::string::npos) {
            const size_t end = line.find("  ", at);
            found.push_back(line.substr(at, end == std::string::npos ? std::string::npos : end - at));
            at = end == std::string::npos ? end : line.find_first_not_of(' ', end);
        }
        if (found.size() >= cells.size() && std::equal(cells.begin(), cells.end(), found.begin())) return true;
    }
    return false;
}

/** @brief The help page as the CLI docs show it: the build's version as <version>. */
template <typename T>
std::string docs_page(const T & command) {
    std::ostringstream out;
    cli::print_help(out, command);
    std::string page = out.str();
    const std::string version = mallard_version();
    for (size_t pos = page.find(version); pos != std::string::npos; pos = page.find(version, pos)) {
        page.replace(pos, version.size(), "<version>");
    }
    return page;
}

#define SKIP_UNDER_MPI() \
    if (comm::size() > 1) GTEST_SKIP() << "runs mallard itself"

} // namespace

TEST(CliTest, EveryCommandIsComplete) {
    const auto groups = cli::groups();
    for (const cli::Command & c : cli::commands()) {
        SCOPED_TRACE(cli::title(c));
        EXPECT_NE(std::find(groups.begin(), groups.end(), c.group), groups.end()) << c.group;
        EXPECT_FALSE(c.summary.empty());
        EXPECT_FALSE(c.synopsis.empty());
        EXPECT_FALSE(c.description.empty());
        EXPECT_FALSE(c.examples.empty());
        EXPECT_FALSE(c.exit_status.empty());
        EXPECT_NE(c.run, nullptr);
        const auto w = words(c.name);
        if (w.size() > 1) {
            const auto families = cli::families();
            EXPECT_TRUE(std::any_of(families.begin(), families.end(), [&](const cli::Family & f) {
                return f.name == w[0];
            })) << "no family " << w[0];
        }
    }
}

TEST(CliTest, HelpOfEveryCommandCoversItsOptions) {
    SKIP_UNDER_MPI();
    for (const cli::Command & c : cli::commands()) {
        const std::string title = cli::title(c);
        SCOPED_TRACE(title);
        for (const std::vector<std::string> & tail : {std::vector<std::string>{"--help"}, {"-h"}}) {
            std::vector<std::string> args = words(c.name);
            args.insert(args.end(), tail.begin(), tail.end());
            const Result r = run(args);
            EXPECT_EQ(r.status, 0) << r.err;
            EXPECT_TRUE(contains(r.out, "Usage: " + title)) << r.out;
            EXPECT_TRUE(contains(r.out, cli::docs_url(c))) << r.out;
            for (const cli::Option & o : c.options) {
                EXPECT_TRUE(contains(r.out, o.name)) << o.name;
                if (!o.alias.empty()) {
                    EXPECT_TRUE(contains(r.out, o.alias)) << o.alias;
                }
            }
        }
    }
}

TEST(CliTest, HelpAnywhereInTheArgumentsShowsThePage) {
    SKIP_UNDER_MPI();
    const Result r = run({"check", "missing.toml", "--ranks", "4", "--help"});
    EXPECT_EQ(r.status, 0) << r.err;
    EXPECT_TRUE(contains(r.out, "Usage: mallard check")) << r.out;
    const Result convert = run({"mesh", "convert", "missing.msh", "--help"});
    EXPECT_EQ(convert.status, 0) << convert.err;
    EXPECT_TRUE(contains(convert.out, "Usage: mallard mesh convert")) << convert.out;
}

TEST(CliTest, TopLevelPageListsEveryCommandAndFamily) {
    SKIP_UNDER_MPI();
    const Result r = run({"--help"});
    EXPECT_EQ(r.status, 0) << r.err;
    for (const cli::Command & c : cli::commands()) {
        if (!c.name.empty()) {
            EXPECT_TRUE(contains(r.out, cli::title(c) + " ")) << c.name;
        }
    }
    for (const cli::Option & o : cli::global_options()) EXPECT_TRUE(contains(r.out, o.name)) << o.name;
    for (const cli::Family & f : cli::families()) {
        const Result page = run({std::string(f.name), "--help"});
        EXPECT_EQ(page.status, 0) << f.name << ": " << page.err;
        for (const cli::Command & c : cli::commands()) {
            const auto w = words(c.name);
            if (w.size() > 1 && w[0] == f.name) {
                EXPECT_TRUE(contains(page.out, cli::title(c))) << c.name;
            }
        }
    }
}

TEST(CliTest, UnknownOptionIsAUsageErrorPointingAtHelp) {
    SKIP_UNDER_MPI();
    for (const cli::Command & c : cli::commands()) {
        const std::string title = cli::title(c);
        SCOPED_TRACE(title);
        std::vector<std::string> args = words(c.name);
        args.push_back("--no-such-option");
        const Result r = run(args);
        EXPECT_EQ(r.status, 2);
        EXPECT_EQ(r.err, "error E-USAGE: " + title + ": unknown option --no-such-option (see " + title +
                             " --help)\n  docs/guide/running.md#e-usage\n");
    }
    const Result r = run({"no-such-command"});
    EXPECT_EQ(r.status, 2);
    EXPECT_EQ(r.err, "error E-USAGE: mallard: unknown command no-such-command; to run an input file, give it with -i: "
                     "mallard -i no-such-command (see mallard --help)\n  docs/guide/running.md#e-usage\n");
    const Result typo = run({"chek", "input.toml"});
    EXPECT_EQ(typo.status, 2);
    EXPECT_EQ(typo.err, "error E-USAGE: mallard: unknown command chek (did you mean \"check\"?)\n"
                        "  docs/guide/running.md#e-usage\n");
    const Result dim = run({"--dim", "4", "-i", "input.toml"});
    EXPECT_EQ(dim.status, 2);
    EXPECT_EQ(dim.err, "error E-USAGE: mallard: --dim takes 2 or 3 (see mallard --help)\n"
                       "  docs/guide/running.md#e-usage\n");
}

TEST(CliTest, CommandErrorsHaveTheirCode) {
    SKIP_UNDER_MPI();
    const Result r = run({"info", "no_such_file.vtkhdf"});
    EXPECT_EQ(r.status, 1);
    EXPECT_EQ(r.err.rfind("error E-INFO: mallard info: ", 0), 0u) << r.err;
    EXPECT_NE(r.err.find("\n  docs/guide/running.md#e-info\n"), std::string::npos) << r.err;
}

TEST(CliTest, UserDocsShowEveryHelpPage) {
    const fs::path docs = fs::path(source_dir()) / "docs" / "guide" / "cli.md";
    if (!fs::exists(docs)) GTEST_SKIP() << "no docs in " << source_dir();
    const std::string text = read_file(docs);
    for (const cli::Command & c : cli::commands()) {
        EXPECT_TRUE(contains(text, "```text\n" + docs_page(c) + "```\n"))
            << cli::title(c) << ": docs/guide/cli.md is out of date; "
            << "python tools/docs/cli.py --mallard build/src/mallard updates it";
    }
    for (const cli::Family & f : cli::families()) {
        EXPECT_TRUE(contains(text, "```text\n" + docs_page(f) + "```\n")) << f.name << ": docs/guide/cli.md is out of date";
    }
}

TEST(CliTest, TestOptionsOnItsPageReachGoogleTest) {
    const cli::Command & test = *cli::command("test");
    cli::Args parsed;
    std::string error;
    ASSERT_TRUE(cli::parse(test, {"--gtest_filter=Sod*", "--gtest_list_tests", "--gtest_shuffle", "--dim", "2"},
                           parsed, error))
        << error;
    const std::vector<std::string> gtest = {"--gtest_filter=Sod*", "--gtest_list_tests", "--gtest_shuffle"};
    EXPECT_EQ(parsed.passthrough, gtest);
    EXPECT_EQ(parsed.value_or("--dim", ""), "2");
}

TEST(CliTest, TestHelpIsMallardsPageAlone) {
    SKIP_UNDER_MPI();
    const Result r = run({"test", "--help"});
    EXPECT_EQ(r.status, 0) << r.err;
    std::ostringstream page;
    cli::print_help(page, *cli::command("test"));
    EXPECT_EQ(r.out, page.str());
}

TEST(CliTest, InputUpgradeSaysWhereItLookedForTheMesh) {
    SKIP_UNDER_MPI();
    // The mesh sits next to the input, but relative paths start from the working directory, as in a run
    const fs::path dir = "upgrade_case";
    fs::create_directories(dir);
    fs::copy_file(source_dir() + "/test/data/meshes/cyl3d.msh", dir / "mesh.msh", fs::copy_options::overwrite_existing);
    std::ofstream(dir / "input.toml") << "[mesh]\nfilename = \"mesh.msh\"\n[[mesh.curved]]\nzone = \"wall\"\n"
                                         "shape = \"sphere\"\ncenter = [0, 0, 0]\nradius = 1\n";
    const std::string input = (dir / "input.toml").string();
    const Result r = run({"input", "upgrade", input});
    EXPECT_EQ(r.status, 1);
    EXPECT_TRUE(contains(r.err, "can't tell 2D or 3D: mesh file 'mesh.msh' not found relative to the working "
                                "directory (" + fs::current_path().string() + "); run from the directory you run "
                                "the case in, or pass --dim 2|3"))
        << r.err;
    const Result stated = run({"input", "upgrade", "--dim", "3", input});
    EXPECT_EQ(stated.status, 0) << stated.err;
    EXPECT_TRUE(contains(stated.out, "type = \"sphere\"\nfilled = false")) << stated.out;
}

namespace {

/** @brief Runs a two-step case in the working directory: VTU snapshots and restart files under case/. */
Result small_run() {
    const bool three = Mallard_DIM == 3;
    std::ofstream("small.toml")
        << "[run]\nn_steps = 2\ndt = 1.0e-3\n[mesh]\ntype = \"cartesian\"\nNx = 4\nNy = 3\nLx = 1.0\nLy = 1.0\n"
        << (three ? "Nz = 3\nLz = 1.0\n[[periodic]]\nzones = [\"back\", \"front\"]\n" : "")
        << "[[periodic]]\nzones = [\"left\", \"right\"]\n[[periodic]]\nzones = [\"bottom\", \"top\"]\n"
        << "[initialize]\ntype = \"analytical\"\nrho = \"1.0 + 0.3 * x\"\n"
        << (three ? "u = [\"0.3\", \"0.0\", \"0.0\"]\n" : "u = [\"0.3\", \"0.0\"]\n")
        << "p = \"1.0\"\n[physics]\ntype = \"euler\"\ngamma = 1.4\np_ref = 1.0\nT_ref = 1.0\nrho_ref = 1.0\n"
        << "[[write_data]]\nprefix = \"case/flow\"\nformat = \"vtu\"\ninterval = 2\nfields = [\"RHO\", \"U\"]\n"
        << "[[write_data]]\nprefix = \"case/restart\"\nformat = \"restart\"\ninterval = 2\n";
    return run({"-i", "small.toml", "--kokkos-num-threads=1"}, true);
}

} // namespace

TEST(CliTest, InfoOfAPastReleasesRestartFileSaysWhatIsInIt) {
    SKIP_UNDER_MPI();
    const std::string file = source_dir() + "/test/data/formats/v0.8.0/gas_000002.restart";
    const Result r = run({"info", file}, true);
    EXPECT_EQ(r.status, 0) << r.err;
    const std::vector<std::vector<std::string>> rows = {
        {"Kind", "restart"},
        {"Format", "restart, version 2"},
        {"Values", "2D, double precision"},
        {"Simulated time", "t = 0.0288616582 s, step 2"},
        {"Note", "no run block: written before Mallard 0.9, or not by Mallard"},
        {"Note", "no snapshot block (written before Mallard 0.10): step and time are the format's own records; "
                 "reason, run and stored precision are unknown"},
        {"cells"},
        {"30"},
        {"field", "units", "components", "min", "max", "description"},
        {"RHO", "kg m-3", "1"},
        {"RHOE", "kg m-1 s-2", "1"},
    };
    for (const auto & row : rows) EXPECT_TRUE(has_row(r.out, row)) << row[0] << "\n" << r.out;
    const Result input = run({"info", "--input", file}, true);
    EXPECT_EQ(input.status, 1);
    EXPECT_TRUE(contains(input.err, "has no run block")) << input.err;
    EXPECT_FALSE(contains(r.out, "This snapshot")) << r.out;
}

TEST(CliTest, InfoOfARunsFilesGivesItsInputAndFields) {
    SKIP_UNDER_MPI();
    const Result ran = small_run();
    ASSERT_EQ(ran.status, 0) << ran.err;
    const Result resolved = run({"input", "resolve", "small.toml"}, true);
    ASSERT_EQ(resolved.status, 0) << resolved.err;
    const Result input = run({"info", "--input", "case/flow_000002.vtu"}, true);
    EXPECT_EQ(input.status, 0) << input.err;
    EXPECT_EQ(input.out, resolved.out);
    EXPECT_EQ(run({"info", "--input", "case/restart_000002.restart"}, true).out, resolved.out);

    const Result text = run({"info", "case/flow_000002.vtu"}, true);
    ASSERT_EQ(text.status, 0) << text.err;
    const std::vector<std::vector<std::string>> rows = {
        {"Kind", "snapshot"},
        {"Simulated time", "t = " + printed_time(2.0e-3) + " s, step 2"},
        {"cells", "points"},
        {Mallard_DIM == 3 ? "36" : "12"},
        {"axis", "min [m]", "max [m]"},
        {"x", "0", "1"},
        {"field", "units", "components", "min", "max", "value", "stored", "description"},
        {"RHO", "kg m-3", "1"},
        {"U", "m s-1", "3"},
        {"Input file", "small.toml"},
        {"Input recorded", "resolved; mallard info --input prints it"},
        {"Restarted from", "no: a fresh start"},
        {"Step", "2"},
        {"Time", printed_time(2.0e-3) + " s"},
        {"Reason", "interval"},
        {"Format", "vtu, version 1"},
    };
    for (const auto & row : rows) EXPECT_TRUE(has_row(text.out, row)) << row[0] << "\n" << text.out;
    // The run's block and this snapshot's, in that order, each its own section
    const size_t run_section = text.out.find("Run ="), snapshot_section = text.out.find("This snapshot =");
    ASSERT_NE(run_section, std::string::npos) << text.out;
    ASSERT_NE(snapshot_section, std::string::npos) << text.out;
    EXPECT_LT(run_section, snapshot_section);
    EXPECT_TRUE(contains(text.out, "same for every file of the run")) << text.out;
    EXPECT_TRUE(contains(text.out, "ranges at t = " + printed_time(2.0e-3) + " s")) << text.out;
    EXPECT_EQ(run({"info", "--format", "yaml", "case/flow_000002.vtu"}, true).status, 2);

    const Result described = run({"info", "--format", "json", "case/flow_000002.vtu"}, true);
    ASSERT_EQ(described.status, 0) << described.err;
    const json::Value v = json::parse(described.out);
    EXPECT_EQ(v["kind"].string, "snapshot");
    EXPECT_EQ(v["format"].string, "VTU");
    EXPECT_EQ(v["dim"].number, Mallard_DIM);
    EXPECT_EQ(v["mesh"]["cells"].number, Mallard_DIM == 3 ? 36 : 12);
    EXPECT_NEAR(v["snapshots"].array.at(0)["t"].number, 2.0e-3, 1e-8);
    const auto & fields = v["fields"].array;
    ASSERT_EQ(fields.size(), 2u);
    EXPECT_EQ(fields[0]["name"].string, "RHO");
    EXPECT_EQ(fields[0]["units"].string, "kg m-3");
    EXPECT_EQ(fields[1]["name"].string, "U");
    EXPECT_EQ(fields[1]["components"].number, 3);
    EXPECT_NEAR(fields[0]["min"].number, 1.0, 0.1);
    EXPECT_EQ(v["run"]["input"]["file"].string, "small.toml");
    const json::Value & snapshot = v["snapshots"].array.at(0);
    EXPECT_EQ(snapshot["step"].number, 2);
    EXPECT_EQ(snapshot["snapshot"]["step"].number, 2);
    EXPECT_EQ(snapshot["snapshot"]["reason"].string, "interval");
    EXPECT_EQ(snapshot["run_id"].string, v["run"]["run"]["run_id"].string);
    EXPECT_EQ(snapshot["snapshot"]["run_id"].string, v["run"]["run"]["run_id"].string);
    const auto & recorded = snapshot["snapshot"]["fields"].array;
    ASSERT_EQ(recorded.size(), 2u);
    EXPECT_EQ(recorded[0]["value"].string, "cell_average");
    EXPECT_EQ(recorded[1]["value"].string, "cell_state");
}

TEST(CliTest, InfoAndDiffSayWhetherProbeValuesArePointValuesOrOfCells) {
    SKIP_UNDER_MPI();
    const bool three = Mallard_DIM == 3;
    std::ofstream input("probes.toml");
    input << "[run]\nn_steps = 2\ndt = 1.0e-3\n[mesh]\ntype = \"cartesian\"\nNx = 8\nNy = 8\nLx = 1.0\nLy = 1.0\n"
          << (three ? "Nz = 4\nLz = 1.0\n[[periodic]]\nzones = [\"back\", \"front\"]\n" : "")
          << "[[periodic]]\nzones = [\"left\", \"right\"]\n[[periodic]]\nzones = [\"bottom\", \"top\"]\n"
          << "[initialize]\ntype = \"analytical\"\nrho = \"1.0 + 0.3 * x * x\"\n"
          << (three ? "u = [\"0.3\", \"0.0\", \"0.0\"]\n" : "u = [\"0.3\", \"0.0\"]\n")
          << "p = \"1.0\"\n[physics]\ntype = \"euler\"\ngamma = 1.4\np_ref = 1.0\nT_ref = 1.0\nrho_ref = 1.0\n"
          << "[numerics.face_reconstruction]\ntype = \"TENO\"\norder = 3\n";
    for (const char * sampling : {"cell", "point"}) {
        input << "[[probes]]\nname = \"" << sampling << "\"\nsampling = \"" << sampling << "\"\nformat = \"vtp\"\n"
              << "file = \"case/" << sampling << ".pvd\"\nfields = [\"RHO\", \"P\"]\nn_points = 5\n"
              << "start = " << (three ? "[0.1, 0.3, 0.4]" : "[0.1, 0.3]") << "\nend = "
              << (three ? "[0.9, 0.3, 0.4]" : "[0.9, 0.3]") << "\ninterval = 2\n";
    }
    input.close();
    const Result ran = run({"-i", "probes.toml", "--kokkos-num-threads=1"}, true);
    ASSERT_EQ(ran.status, 0) << ran.err;

    const Result point = run({"info", "case/point.pvd"}, true);
    ASSERT_EQ(point.status, 0) << point.err;
    EXPECT_TRUE(has_row(point.out, {"Point values", "TENO, order 3; falling back to linear_least_squares, then "
                                                    "cell_average"}))
        << point.out;
    const Result cell = run({"info", "case/cell.pvd"}, true);
    ASSERT_EQ(cell.status, 0) << cell.err;
    EXPECT_FALSE(contains(cell.out, "Point values")) << cell.out;

    const Result compared = run({"diff", "case/cell.pvd", "case/point.pvd", "--tol", "1", "--format", "json"}, true);
    EXPECT_EQ(compared.status, 0) << compared.err;
    const json::Value v = json::parse(compared.out);
    EXPECT_EQ(v["result"].string, "warning");
    std::map<std::string, const json::Value *> fields;
    for (const json::Value & f : v["fields"].array) fields[f["name"].string] = &f;
    ASSERT_TRUE(fields.count("RHO") && fields.count("P")) << compared.out;
    EXPECT_EQ((*fields["RHO"])["value_a"].string, "cell_average");
    EXPECT_EQ((*fields["RHO"])["value_b"].string, "point_value");
    EXPECT_EQ((*fields["RHO"])["status"].string, "warning");
    EXPECT_EQ((*fields["P"])["value_a"].string, "cell_state");
    const Result text = run({"diff", "case/cell.pvd", "case/point.pvd", "--tol", "1"}, true);
    EXPECT_TRUE(has_row(text.out, {"WARN", "RHO", "cell_average", "point_value"})) << text.out;
    EXPECT_TRUE(contains(text.out, "different kinds of values")) << text.out;
    const Result itself = run({"diff", "case/point.pvd", "case/point.pvd", "--format", "json"}, true);
    EXPECT_EQ(json::parse(itself.out)["result"].string, "ok") << itself.out;
}

TEST(CliTest, InfoOfARunDirectoryListsItsRunAndFiles) {
    SKIP_UNDER_MPI();
    ASSERT_EQ(small_run().status, 0);
    const Result r = run({"info", "."}, true);
    EXPECT_EQ(r.status, 0) << r.err;
    const std::vector<std::vector<std::string>> rows = {
        {"Ended (finished): n_steps = 2 reached, at step 2, t = 0.002 s"},
        {"Last snapshot: case/flow_000002.vtu, step 2, t = 0.002 s"},
        {"Last restart: case/restart_000002.restart, step 2, t = 0.002 s"},
        {"file", "kind", "times", "first t [s]", "last t [s]", "last step", "fields", "format"},
        {"case/flow.pvd", "snapshot", "2", "0", printed_time(2.0e-3), "2", "2", "PVD series index (VTU snapshots)"},
        {"case/restart_000002.restart", "restart", "1", printed_time(2.0e-3), printed_time(2.0e-3), "2", Mallard_DIM == 3 ? "5" : "4", "restart"},
    };
    for (const auto & row : rows) EXPECT_TRUE(has_row(r.out, row)) << row[0] << "\n" << r.out;
    // A series is listed by its index, not its snapshots
    EXPECT_FALSE(has_row(r.out, {"case/flow_000002.vtu"})) << r.out;
    EXPECT_EQ(run({"info", "--input", "."}, true).out, run({"input", "resolve", "small.toml"}, true).out);
}

TEST(CliTest, InfoOfARunDirectorySummarizesItsOutcomeHealthCostAndWarnings) {
    SKIP_UNDER_MPI();
    ASSERT_EQ(small_run().status, 0);
    std::ofstream("mallard.jsonl", std::ios::app) << R"({"type":"warning","message":"a recorded warning"})" << "\n";
    const Result r = run({"info", "."}, true);
    EXPECT_EQ(r.status, 0) << r.err;
    const auto heading = [&](const std::string & title, const std::string & status) {
        return std::regex_search(r.out, std::regex("== " + title + " =+ " + status + "\n"));
    };
    EXPECT_TRUE(heading("Outcome", "WARN")) << r.out;
    EXPECT_TRUE(has_row(r.out, {"Ended (finished): n_steps = 2 reached, at step 2, t = 0.002 s"})) << r.out;
    EXPECT_TRUE(heading("Health", "ok")) << r.out;
    EXPECT_TRUE(has_row(r.out, {"quantity", "value", "units", "step", "at"})) << r.out;
    EXPECT_TRUE(has_row(r.out, {"steps", "2", "steps", "this run's"})) << r.out;
    EXPECT_TRUE(heading("Warnings", "WARN")) << r.out;
    EXPECT_TRUE(has_row(r.out, {"- a recorded warning"})) << r.out;
    std::istringstream lines(r.out);
    std::string line, previous = "";
    for (bool first = true; std::getline(lines, line); first = false, previous = line) {
        if (!first && line.starts_with("== ")) {
            EXPECT_EQ(previous, "") << line << "\n" << r.out;
        }
    }
}

TEST(CliTest, InfoOfARunDirectoryChecksItsFilesAgainstTheirRecords) {
    SKIP_UNDER_MPI();
    ASSERT_EQ(small_run().status, 0);
    const Result clean = run({"info", "--verify", "."}, true);
    EXPECT_EQ(clean.status, 0) << clean.out;
    EXPECT_TRUE(has_row(clean.out, {"Recorded files", "all present, as written"})) << clean.out;

    // Same size, other contents: only the hash tells
    const std::string vtu = "case/flow_000000.vtu";
    {
        std::fstream f(vtu, std::ios::in | std::ios::out | std::ios::binary);
        f.seekg(-2, std::ios::end);
        const char c = static_cast<char>(f.get());
        f.seekp(-2, std::ios::end);
        f.put(c == 'x' ? 'y' : 'x');
    }
    EXPECT_EQ(run({"info", "."}, true).status, 0);
    const Result changed = run({"info", "--verify", "."}, true);
    EXPECT_EQ(changed.status, 1);
    EXPECT_TRUE(has_row(changed.out, {vtu, "its contents differ from what the run wrote (SHA-256)"})) << changed.out;

    const std::string restart = "case/restart_000002.restart";
    fs::resize_file(restart, fs::file_size(restart) - 8);
    const Result cut = run({"info", "."}, true);
    EXPECT_EQ(cut.status, 1);
    EXPECT_TRUE(contains(cut.out, "the run wrote")) << cut.out;
    EXPECT_TRUE(contains(cut.out, "changed or cut short since")) << cut.out;

    // A file whose write did not finish keeps its temporary name, and is refused
    fs::copy_file("case/flow_000002.vtu", "case/flow_000004.vtu.partial");
    const Result partial = run({"info", "case/flow_000004.vtu.partial"}, true);
    EXPECT_EQ(partial.status, 1);
    EXPECT_TRUE(contains(partial.err, "its write did not finish")) << partial.err;
    EXPECT_TRUE(contains(run({"info", "."}, true).out, "its write did not finish"));
}

TEST(CliTest, InfoOfAMeshFileReportsItsQualityAndWritesIt) {
    SKIP_UNDER_MPI();
    const std::string mesh =
        source_dir() + (Mallard_DIM == 3 ? "/test/data/meshes/cyl3d.msh" : "/test/data/meshes/cyl2d.msh");
    const Result r = run({"info", "--mesh-fields", "quality", mesh}, true);
    EXPECT_EQ(r.status, 0) << r.err;
    const std::vector<std::vector<std::string>> rows = {
        {"Format", "Gmsh"},
        {"boundary zone", "faces"},
        {"metric", "units", "max", "mean", "limit", "cells over limit"},
        {"non-orthogonality", "deg"},
        {"skewness", "dimensionless"},
        {"aspect ratio", "dimensionless"},
        {"volume ratio", "dimensionless"},
        {"metric", "value", "cell", "x [m]", "y [m]"},
    };
    for (const auto & row : rows) EXPECT_TRUE(has_row(r.out, row)) << row[0] << "\n" << r.out;
    size_t written = 0;
    for (const auto & entry : fs::directory_iterator(".")) {
        written += entry.path().filename().string().starts_with("quality_");
    }
    EXPECT_EQ(written, 1u);
    const Result not_a_mesh =
        run({"info", "--mesh-fields", "quality", source_dir() + "/test/data/formats/gas.toml"}, true);
    EXPECT_EQ(not_a_mesh.status, 2);
    EXPECT_TRUE(contains(not_a_mesh.err, "--mesh-fields takes a mesh file")) << not_a_mesh.err;
}

namespace {

/** @brief The working directory moved to dir while it lives. */
struct InDirectory {
    fs::path previous = fs::current_path();
    explicit InDirectory(const fs::path & dir) { fs::current_path(dir); }
    ~InDirectory() { fs::current_path(previous); }
};

/** @brief A flow past a cylinder whose mesh is outside its run directory, and whose probe points are inside it. */
void write_pack_case() {
    const bool three = Mallard_DIM == 3;
    fs::create_directories("study/data");
    fs::create_directories("study/run");
    fs::copy_file(source_dir() + (three ? "/test/data/meshes/cyl3d.msh" : "/test/data/meshes/cyl2d.msh"),
                  "study/data/cyl.msh");
    std::ofstream("study/run/points.txt") << (three ? "0.6 0.0 0.15\n0.0 0.7 0.15\n" : "0.6 0.0\n0.0 0.7\n");
    const std::string u = three ? "[0.2, 0.0, 0.0]" : "[0.2, 0.0]";
    std::ofstream input("study/run/input.toml");
    input << "[run]\nn_steps = 4\ncfl = 0.5\n[mesh]\ntype = \"file\"\nfilename = \"../data/cyl.msh\"\n"
          << "[initialize]\ntype = \"analytical\"\nT = \"0.7142857142857143\"\np = \"0.7142857142857143 * (1 + "
             "0.1 * x)\"\nu = "
          << (three ? "[\"0.2\", \"0.05 * y\", \"0.0\"]" : "[\"0.2\", \"0.05 * y\"]") << "\n"
          << "[physics]\ntype = \"euler\"\ngamma = 1.4\np_ref = 0.7142857142857143\nT_ref = 0.7142857142857143\n"
          << "rho_ref = 1.0\n[numerics]\nriemann_solver = \"HLLC\"\n[numerics.face_reconstruction]\ntype = \"MUSCL\"\n";
    for (const char * zone : {"inlet", "outlet"}) {
        input << "[[boundaries]]\nname = \"" << zone << "\"\ntype = \"farfield\"\nu = " << u
              << "\np = 0.7142857142857143\nT = 0.7142857142857143\n";
    }
    for (const std::string zone : {"walls", "cylinder", "back", "front"}) {
        if (three || (zone != "back" && zone != "front")) input << "[[boundaries]]\nname = \"" << zone << "\"\ntype = \"symmetry\"\n";
    }
    input << "[[forces]]\nzone = \"cylinder\"\ninterval = 1\nfile = \"solut/forces.csv\"\n"
          << "[[probes]]\nname = \"pts\"\ntype = \"points\"\npoints_file = \"points.txt\"\ninterval = 1\n"
          << "variables = [\"RHO\"]\n"
          << "[[write_data]]\nprefix = \"solut/flow\"\nformat = \"vtu\"\ninterval = 2\nvariables = [\"RHO\"]\n"
          << "[[write_data]]\nprefix = \"solut/restart\"\nformat = \"restart\"\ninterval = 2\n";
}

std::vector<std::string> words_of(const std::string & command) {
    std::istringstream in(command);
    std::vector<std::string> out;
    for (std::string w; in >> w;) out.push_back(w);
    return out;
}

/** @brief The manifest's files of a role, by path. */
std::vector<std::string> paths_of(const json::Value & manifest, const std::string & role) {
    std::vector<std::string> out;
    for (const auto & f : manifest["files"].array) {
        if (f["role"].string == role) out.push_back(f["path"].string);
    }
    return out;
}

} // namespace

TEST(CliTest, PackReproducesARunBitForBit) {
    SKIP_UNDER_MPI();
    if (std::system("tar --version > /dev/null 2>&1") != 0) GTEST_SKIP() << "no tar";
    write_pack_case();
    {
        // A run, then one continued from its restart file given on the command line
        const InDirectory in("study/run");
        ASSERT_EQ(run({"-i", "input.toml", "--kokkos-num-threads=1"}, true).status, 0) << read_file("err.txt");
        const Result continued =
            run({"-i", "input.toml", "--restart", "solut/restart_000002.restart", "--kokkos-num-threads=1"}, true);
        ASSERT_EQ(continued.status, 0) << continued.err;
    }
    const std::string name = std::string("study") + (archive::have_gzip() ? ".tar.gz" : ".tar");
    const Result packed = run({"pack", "study/run", "-o", name}, true);
    ASSERT_EQ(packed.status, 0) << packed.err;
    const Result verified = run({"pack", "--verify", name}, true);
    EXPECT_EQ(verified.status, 0) << verified.out << verified.err;
    EXPECT_TRUE(has_row(verified.out, {"Against the run", "4 files checked against the SHA-256 the run recorded"}))
        << verified.out;

    fs::create_directories("elsewhere");
    ASSERT_EQ(std::system(("tar -xf " + name + " -C elsewhere").c_str()), 0);
    const json::Value manifest = json::parse(read_file("elsewhere/study/manifest.json"));
    // The mesh is one level up, so the case keeps the run directory's name below the layout around it
    EXPECT_EQ(manifest["rerun"]["directory"].string, "case/run");
    EXPECT_EQ(manifest["rerun"]["command"].string,
              "mallard -i input.toml --restart solut/restart_000002.restart");
    EXPECT_EQ(paths_of(manifest, "mesh"), std::vector<std::string>{"case/data/cyl.msh"});
    std::vector<std::string> read = paths_of(manifest, "read");
    std::sort(read.begin(), read.end());
    EXPECT_EQ(read, (std::vector<std::string>{"case/run/points.txt", "case/run/solut/restart_000002.restart"}));
    // Small outputs: the CSV files, not the snapshots
    const std::vector<std::string> outputs = paths_of(manifest, "output");
    EXPECT_EQ(outputs, (std::vector<std::string>{"output/solut/forces.csv", "output/probe_pts.csv"}));

    {
        const InDirectory in("elsewhere/study/case/run");
        std::vector<std::string> command = words_of(manifest["rerun"]["command"].string);
        command.erase(command.begin());
        command.push_back("--kokkos-num-threads=1");
        const Result rerun = run(command, true);
        ASSERT_EQ(rerun.status, 0) << rerun.err;
        const fs::path original = fs::path("../../../../study/run/solut");
        for (const char * file : {"restart_000004.restart", "flow_000004.vtu"}) {
            const Result same = run({"diff", (original / file).string(), (fs::path("solut") / file).string()}, true);
            EXPECT_EQ(same.status, 0) << file << "\n" << same.out << same.err;
        }
    }

    // A file changed since the archive was made, and one added, are found; so is one changed since the run read it
    {
        std::fstream points("elsewhere/study/case/run/points.txt", std::ios::in | std::ios::out | std::ios::binary);
        points.seekp(0);
        points.put('1');
    }
    ASSERT_EQ(std::system("cd elsewhere && COPYFILE_DISABLE=1 tar -cf ../tampered.tar study"), 0);
    const Result tampered = run({"pack", "--verify", "tampered.tar"}, true);
    EXPECT_EQ(tampered.status, 1);
    EXPECT_TRUE(has_row(tampered.out, {"case/run/points.txt", "its SHA-256 differs from the manifest's"}))
        << tampered.out;
    EXPECT_TRUE(has_row(tampered.out, {"case/run/points.txt", "differs from what the run read (its run block's SHA-256)"}))
        << tampered.out;
    EXPECT_TRUE(has_row(tampered.out, {"case/run/mallard.jsonl", "not in the manifest"})) << tampered.out;
    std::ofstream("study/run/points.txt", std::ios::app) << "\n";
    const Result changed = run({"pack", "study/run", "-o", "changed.tar"}, true);
    EXPECT_EQ(changed.status, 1);
    EXPECT_TRUE(contains(changed.err, "changed since the run read it")) << changed.err;
    EXPECT_FALSE(fs::exists("changed.tar"));
    EXPECT_FALSE(fs::exists("changed.tar.partial"));
}

TEST(CliTest, ListFieldsGivesTheCatalogOrOneCasesFields) {
    SKIP_UNDER_MPI();
    const Result catalog = run({"list", "fields"}, true);
    EXPECT_EQ(catalog.status, 0) << catalog.err;
    EXPECT_TRUE(contains(catalog.out, "== Fields: conservative")) << catalog.out;
    const std::vector<std::vector<std::string>> catalog_rows = {
        {"field", "units", "description", "needs", "available in"},
        {"RHO", "kg m-3", "density", "every case", "write_data, probes, statistics"},
        {"CFL", "dimensionless", "local CFL number", "every case", "write_data"},
        {"MU_T", "kg m-1 s-1", "eddy viscosity", "[les]", "write_data"},
        {"MACH", "dimensionless", "Mach number", "every case", "write_data, post derive"},
        {"U", "m s-1", "velocity (vector)", "every case", "write_data, probes, statistics"},
        {"Y_<s>", "dimensionless", "mass fraction of species s", "a mixture", "write_data, probes, statistics"},
        {"MASS_FLUX", "kg m-2 s-1", "normal mass flux", "[[probes]] on a surface or boundary zone", "probes"},
        {"MEAN_<A>", "A's", "time average of A", "[statistics] fields with A, or [[probes]] averaging A",
         "write_data, probes"},
        {"SOURCE_<A>", "A's s-1", "source of A, per second", "[[sources]] or [source]", "write_data"},
        {"SKEWNESS", "dimensionless", "largest face skewness", "every case", "write_data"},
        {"DELTA_OVER_ETA", "dimensionless", "cell size over the estimated Kolmogorov length",
         "mallard check --mesh-fields (resolution estimates)", "-"},
    };
    for (const auto & row : catalog_rows) EXPECT_TRUE(has_row(catalog.out, row)) << row[0] << "\n" << catalog.out;
    EXPECT_TRUE(contains(catalog.out, "MEAN_<A>")) << catalog.out;
    ASSERT_EQ(small_run().status, 0);
    const Result text = run({"list", "fields", "small.toml"}, true);
    ASSERT_EQ(text.status, 0) << text.err;
    const std::vector<std::vector<std::string>> case_rows = {
        {"field", "units", "description", "available in"},
        {"RHO", "kg m-3", "density", "write_data, probes, statistics"},
        {"U_X", "m s-1", "x component of velocity", "write_data, probes, statistics"},
        {"Q_CRITERION", "s-2", "Q-criterion", "write_data, post derive"},
        {"P_MAX", "kg m-1 s-2", "largest pressure seen", "write_data"},
        {"field", "units", "needs"},
        {"MU_T", "kg m-1 s-1", "[les]"},
    };
    for (const auto & row : case_rows) EXPECT_TRUE(has_row(text.out, row)) << row[0] << "\n" << text.out;
    EXPECT_EQ(run({"list", "fields", "--format", "yaml"}, true).status, 2);
    const Result listed = run({"list", "fields", "small.toml", "--format", "json"}, true);
    ASSERT_EQ(listed.status, 0) << listed.err;
    const json::Value v = json::parse(listed.out);
    EXPECT_EQ(v["dim"].number, Mallard_DIM);
    std::vector<std::string> names;
    for (const auto & f : v["fields"].array) names.push_back(f["name"].string);
    for (const char * name : {"RHO", "U_X", "P", "Q_CRITERION", "MACH", "LAMBDA2", "SCHLIEREN", "P_MAX"}) {
        EXPECT_NE(std::find(names.begin(), names.end(), name), names.end()) << name;
    }
    EXPECT_EQ(std::count(names.begin(), names.end(), "U_Z"), Mallard_DIM == 3 ? 1 : 0);
    std::vector<std::string> missing;
    for (const auto & f : v["missing"].array) missing.push_back(f["name"].string + ": " + f["needs"].string);
    EXPECT_NE(std::find(missing.begin(), missing.end(), "MU_T: [les]"), missing.end());
    EXPECT_NE(std::find(missing.begin(), missing.end(), "Y_<s>: a mixture"), missing.end());
    for (const auto & f : v["fields"].array) {
        if (f["name"].string != "P") continue;
        std::vector<std::string> where;
        for (const auto & w : f["available"].array) where.push_back(w.string);
        EXPECT_EQ(where, (std::vector<std::string>{"write_data", "probes", "statistics"}));
    }
}

TEST(CliTest, OutputGuideTableIsTheFieldCatalog) {
    SKIP_UNDER_MPI();
    const fs::path docs = fs::path(source_dir()) / "docs" / "guide" / "output.md";
    if (!fs::exists(docs)) GTEST_SKIP() << "no docs in " << source_dir();
    const std::string text = read_file(docs);
    const std::string begin = "<!-- BEGIN GENERATED fields -->\n", end = "<!-- END GENERATED fields -->";
    ASSERT_TRUE(contains(text, begin) && contains(text, end));
    const size_t from = text.find(begin) + begin.size();
    std::istringstream table(text.substr(from, text.find(end) - from));
    std::vector<std::vector<std::string>> rows;
    for (std::string line; std::getline(table, line);) {
        std::vector<std::string> cells;
        const std::string inner = line.size() > 4 ? line.substr(2, line.size() - 4) : "";
        for (size_t at = 0;;) {
            const size_t stop = inner.find(" | ", at);
            std::string c = inner.substr(at, stop == std::string::npos ? std::string::npos : stop - at);
            if (c.size() > 1 && c.front() == '`' && c.back() == '`') c = c.substr(1, c.size() - 2);
            cells.push_back(c);
            if (stop == std::string::npos) break;
            at = stop + 3;
        }
        rows.push_back(cells);
    }
    const Result listed = run({"list", "fields", "--format", "json"}, true);
    ASSERT_EQ(listed.status, 0) << listed.err;
    const json::Value v = json::parse(listed.out);
    ASSERT_EQ(rows.size(), v["fields"].array.size() + 2)
        << "docs/guide/output.md is out of date; python tools/docs/fields.py --mallard build/src/mallard updates it";
    std::string group;
    for (size_t i = 0; i < v["fields"].array.size(); i++) {
        const json::Value & f = v["fields"].array[i];
        std::string where;
        for (const auto & w : f["available"].array) where += (where.empty() ? "" : ", ") + w.string;
        const std::string needs = f.has("needs") ? f["needs"].string : "every case";
        const std::vector<std::string> expected = {f["group"].string == group ? "" : f["group"].string,
                                                   f["name"].string,
                                                   f["units"].string,
                                                   f["long_name"].string,
                                                   needs,
                                                   where.empty() ? "-" : where};
        group = f["group"].string;
        EXPECT_EQ(rows[i + 2], expected)
            << "docs/guide/output.md is out of date; python tools/docs/fields.py --mallard build/src/mallard updates it";
    }
}

TEST(CliTest, ListSpeciesGivesTheMechanismsSpecies) {
    SKIP_UNDER_MPI();
    if (Mallard_DIM != 2) GTEST_SKIP() << "the mixture fixture is a 2D case";
    std::string mixture = read_file(source_dir() + "/test/data/formats/mixture.toml");
    const std::string relative = "\"../../../mechanisms/h2o2.yaml\"";
    ASSERT_NE(mixture.find(relative), std::string::npos);
    mixture.replace(mixture.find(relative), relative.size(), "\"" + source_dir() + "/mechanisms/h2o2.yaml\"");
    std::ofstream("mixture.toml") << mixture;
    const Result r = run({"list", "species", "mixture.toml"}, true);
    EXPECT_EQ(r.status, 0) << r.err;
    EXPECT_TRUE(has_row(r.out, {"#", "species"})) << r.out;
    const std::vector<std::string> species = {"H2", "H", "O", "O2", "OH", "H2O", "HO2", "H2O2", "AR", "N2"};
    for (size_t i = 0; i < species.size(); i++) {
        EXPECT_TRUE(has_row(r.out, {std::to_string(i), species[i]})) << species[i] << "\n" << r.out;
    }
    const Result listed = run({"list", "species", "mixture.toml", "--format", "json"}, true);
    ASSERT_EQ(listed.status, 0) << listed.err;
    std::vector<std::string> names;
    const json::Value v = json::parse(listed.out);
    for (const auto & s : v["species"].array) names.push_back(s.string);
    EXPECT_EQ(names, species);
    ASSERT_EQ(small_run().status, 0);
    const Result gas = run({"list", "species", "small.toml"}, true);
    EXPECT_EQ(gas.status, 0) << gas.err;
    EXPECT_TRUE(contains(gas.out, "a single gas, no species")) << gas.out;
    const Result broken = run({"list", "species", "missing.toml"}, true);
    EXPECT_EQ(broken.status, 1);
}

TEST(CliTest, ListFieldsAndSpeciesNeedNoMesh) {
    SKIP_UNDER_MPI();
    ASSERT_EQ(small_run().status, 0);
    std::string text = read_file("small.toml");
    const std::string generated = "type = \"cartesian\"\nNx = 4\nNy = 3\nLx = 1.0\nLy = 1.0\n";
    ASSERT_NE(text.find(generated), std::string::npos);
    text.replace(text.find(generated), generated.size(), "type = \"file\"\nfilename = \"absent.msh\"\n");
    std::ofstream("absent.toml") << text;
    ASSERT_FALSE(fs::exists("absent.msh"));
    const Result with_mesh = run({"list", "fields", "small.toml", "--format", "json"}, true);
    const Result without = run({"list", "fields", "absent.toml", "--format", "json"}, true);
    ASSERT_EQ(without.status, 0) << without.err;
    std::string expected = with_mesh.out;
    expected.replace(expected.find("small.toml"), 10, "absent.toml");
    EXPECT_EQ(without.out, expected);
    EXPECT_EQ(run({"list", "fields", "absent.toml"}, true).status, 0);

    std::string mixture = read_file(source_dir() + "/test/data/formats/mixture.toml");
    const std::string relative = "\"../../../mechanisms/h2o2.yaml\"";
    mixture.replace(mixture.find(relative), relative.size(), "\"" + source_dir() + "/mechanisms/h2o2.yaml\"");
    const std::string mesh = "type = \"cartesian\"\n";
    ASSERT_NE(mixture.find(mesh), std::string::npos);
    mixture.replace(mixture.find(mesh), mesh.size(), "type = \"file\"\nfilename = \"absent.msh\"\n");
    std::ofstream("mixture_absent.toml") << mixture;
    const Result species = run({"list", "species", "mixture_absent.toml", "--format", "json"}, true);
    ASSERT_EQ(species.status, 0) << species.err;
    std::vector<std::string> names;
    const json::Value listed = json::parse(species.out);
    for (const auto & s : listed["species"].array) names.push_back(s.string);
    EXPECT_EQ(names, (std::vector<std::string>{"H2", "H", "O", "O2", "OH", "H2O", "HO2", "H2O2", "AR", "N2"}));
}

TEST(CliTest, CheckWithoutAnInputChecksTheEnvironment) {
    SKIP_UNDER_MPI();
    const Result text = run({"check", "--kokkos-num-threads=1"}, true);
    ASSERT_EQ(text.status, 0) << text.err;
    EXPECT_TRUE(contains(text.out, "== Environment")) << text.out;
    EXPECT_FALSE(contains(text.out, "== Input")) << text.out;
    for (const char * check : {"Kokkos", "MPI", "GPU-aware MPI", "NCCL", "HDF5", "CGNS", "zlib", "KaMinPar"}) {
        EXPECT_TRUE(has_row(text.out, {"ok", check})) << check << "\n" << text.out;
    }
    EXPECT_TRUE(has_row(text.out, {"rank", "host", "device", "GPU", "CPUs", "threads", "binding"})) << text.out;
    EXPECT_TRUE(has_row(text.out, {"0"})) << text.out;
    EXPECT_EQ(run({"check", "--ranks", "4"}, true).status, 2);

    // One source for the build: the run block of every output file
    ASSERT_EQ(small_run().status, 0);
    const Result listed = run({"check", "small.toml", "--format", "json", "--kokkos-num-threads=1"}, true);
    ASSERT_EQ(listed.status, 0) << listed.err;
    const json::Value environment = json::parse(listed.out)["environment"];
    const json::Value file = json::parse(run({"info", "--format", "json", "case/flow_000002.vtu"}, true).out);
    EXPECT_EQ(environment["build"].keys(), file["run"]["build"].keys());
    EXPECT_EQ(environment["build"]["dim"].number, file["run"]["build"]["dim"].number);
    EXPECT_EQ(environment["dependencies"]["mpi"].string, file["run"]["dependencies"]["mpi"].string);
    EXPECT_EQ(environment["placement"].array.size(), 1u);
    EXPECT_EQ(environment["placement"].array[0]["threads"].number, 1);
}

TEST(CliTest, ListKeysGivesTheCatalogOrOneCasesKeys) {
    SKIP_UNDER_MPI();
    const Result numerics = run({"list", "keys", "numerics"}, true);
    ASSERT_EQ(numerics.status, 0) << numerics.err;
    EXPECT_TRUE(contains(numerics.out, "== Keys: [numerics]")) << numerics.out;
    EXPECT_TRUE(has_row(numerics.out, {"key", "type", "units", "default", "values"})) << numerics.out;
    EXPECT_TRUE(has_row(numerics.out, {"numerics.riemann_solver", "string", "\"HLLC\"", "HLL, HLLC, RHLL, Roe, Rusanov"}))
        << numerics.out;
    EXPECT_FALSE(contains(numerics.out, "mesh.Nx")) << numerics.out;
    const Result typo = run({"list", "keys", "numerix"}, true);
    EXPECT_EQ(typo.status, 2);
    EXPECT_TRUE(contains(typo.err, "did you mean \"numerics\"")) << typo.err;
    // Every file of the catalog is a key, including those whose file names are percent-encoded
    const Result all = run({"list", "keys", "--format", "json"}, true);
    ASSERT_EQ(all.status, 0) << all.err;
    const json::Value catalog = json::parse(all.out)["keys"];
    for (const fs::path & dir : data_dirs()) {
        if (!fs::is_directory(dir / "docs" / "input-keys")) continue;
        const auto files = std::distance(fs::directory_iterator(dir / "docs" / "input-keys"), fs::directory_iterator());
        EXPECT_EQ(catalog.keys().size(), static_cast<size_t>(files));
        break;
    }
    for (const char * key : {"boundaries.nscbc.sigma_t", "boundaries.nscbc.sigma_T", "regions.<name>.type"}) {
        EXPECT_TRUE(catalog.has(key)) << key;
    }

    ASSERT_EQ(small_run().status, 0);
    const Result text = run({"list", "keys", "[write_data]", "small.toml"}, true);
    ASSERT_EQ(text.status, 0) << text.err;
    EXPECT_TRUE(has_row(text.out, {"key", "set", "type", "units", "default", "values"})) << text.out;
    EXPECT_TRUE(has_row(text.out, {"write_data.fields", "yes", "array"})) << text.out;
    EXPECT_FALSE(contains(text.out, "physics.gamma")) << text.out;
    // The same keys mallard check --catalog writes
    ASSERT_EQ(run({"check", "small.toml", "--catalog", "keys.json"}, true).status, 0);
    const Result listed = run({"list", "keys", "small.toml", "--format", "json"}, true);
    ASSERT_EQ(listed.status, 0) << listed.err;
    const json::Value keys = json::parse(listed.out)["keys"];
    EXPECT_EQ(keys.keys(), json::parse(read_file("keys.json")).keys());
    EXPECT_TRUE(keys["physics.gamma"]["set"].boolean);
    EXPECT_FALSE(keys["run.cfl"]["set"].boolean);
}

TEST(CliTest, ListTemplatesIsWhatNewListShowedWhichStillWorks) {
    SKIP_UNDER_MPI();
    const Result listed = run({"list", "templates"});
    EXPECT_EQ(listed.status, 0) << listed.err;
    EXPECT_TRUE(contains(listed.out, "cylinder")) << listed.out;
    const Result old = run({"new", "--list"});
    EXPECT_EQ(old.status, 0);
    EXPECT_EQ(old.out, listed.out);
    EXPECT_TRUE(contains(old.err, "use mallard list templates")) << old.err;
}

namespace {

/** @brief The largest |B - A| over the cells of a field of two files, and its cell, from the reader. */
std::pair<double, size_t> largest_difference(const std::string & a, const std::string & b, const std::string & field) {
    const std::vector<double> x = output::read(output::open(a), field, 0);
    const std::vector<double> y = output::read(output::open(b), field, 0);
    std::pair<double, size_t> out = {0.0, 0};
    for (size_t i = 0; i < x.size(); i++) {
        if (std::abs(y[i] - x[i]) > out.first) out = {std::abs(y[i] - x[i]), i};
    }
    return out;
}

/** @brief Runs steps (two) steps of a periodic case on an N x (3N/4) mesh (x 3N/4 in 3D), writing restart files and VTU snapshots to dir/. */
Result diff_case(const std::string & dir, uint32_t n, const std::string & rho, int steps = 2) {
    const bool three = Mallard_DIM == 3;
    fs::create_directories(dir);
    std::ofstream(dir + "/input.toml")
        << "[run]\nn_steps = " << steps << "\ndt = 1.0e-3\n[mesh]\ntype = \"cartesian\"\nNx = " << n << "\nNy = " << 3 * n / 4
        << "\nLx = 1.0\nLy = 1.0\n" << (three ? "Nz = " + std::to_string(3 * n / 4) + "\nLz = 1.0\n[[periodic]]\nzones = [\"back\", \"front\"]\n" : "")
        << "[[periodic]]\nzones = [\"left\", \"right\"]\n[[periodic]]\nzones = [\"bottom\", \"top\"]\n"
        << "[initialize]\ntype = \"analytical\"\nrho = \"" << rho << "\"\n"
        << (three ? "u = [\"0.3\", \"0.0\", \"0.0\"]\n" : "u = [\"0.3\", \"0.0\"]\n")
        << "p = \"1.0\"\n[physics]\ntype = \"euler\"\ngamma = 1.4\np_ref = 1.0\nT_ref = 1.0\nrho_ref = 1.0\n"
        << "[[write_data]]\nprefix = \"" << dir << "/flow\"\nformat = \"vtu\"\ninterval = 2\nfields = [\"RHO\", \"U\"]\n"
        << "[[write_data]]\nprefix = \"" << dir << "/restart\"\nformat = \"restart\"\ninterval = 2\n";
    return run({"-i", dir + "/input.toml", "--kokkos-num-threads=1"}, true);
}

} // namespace

TEST(CliTest, DiffOfRestartFilesSaysWhetherAndWhereTheirStatesDiffer) {
    SKIP_UNDER_MPI();
    const std::string dir = source_dir() + "/test/data/formats/";
    const std::string first = dir + "v0.8.0/gas_000000.restart", later = dir + "v0.8.0/gas_000002.restart";
    const Result same = run({"diff", later, dir + "v0.7.0/gas_000002.restart"}, true);
    EXPECT_EQ(same.status, 0) << same.err;
    EXPECT_TRUE(contains(same.out, "Result: ok, identical, the same state bit for bit (exit status 0)")) << same.out;
    EXPECT_TRUE(has_row(same.out, {"ok", "RHOE", "kg m-1 s-2", "0 of 30", "0", "0", "0"})) << same.out;

    const Result differ = run({"diff", first, later, "--format", "json"}, true);
    EXPECT_EQ(differ.status, 1) << differ.err;
    const json::Value v = json::parse(differ.out);
    EXPECT_EQ(v["result"].string, "error");
    EXPECT_EQ(v["time"]["status"].string, "error");
    ASSERT_EQ(v["fields"].array.size(), 4u);
    for (const json::Value & f : v["fields"].array) {
        SCOPED_TRACE(f["name"].string);
        const auto [linf, cell] = largest_difference(first, later, f["name"].string);
        EXPECT_GT(linf, 0.0);
        EXPECT_DOUBLE_EQ(f["absolute"]["linf"].number, linf);
        EXPECT_EQ(f["max_at"]["index"].number, static_cast<double>(cell));
        EXPECT_EQ(f["status"].string, "error");
    }
    const Result loose = run({"diff", first, later, "--tol", "1"}, true);
    EXPECT_EQ(loose.status, 0) << loose.err;
    EXPECT_TRUE(contains(loose.out, "Result: WARN, within tolerance, differences but none beyond it (exit status 0)"))
        << loose.out;
    EXPECT_EQ(run({"diff", first, later, "--tol", "1,RHOE=1e-6"}, true).status, 1);
    EXPECT_EQ(run({"diff", first, later, "--tol", "small"}, true).status, 2);

    const std::string nscbc = dir + "v0.8.0/nscbc_000002.restart";
    const Result faces = run({"diff", nscbc, dir + "v0.7.0/nscbc_000002.restart", "--format", "json"}, true);
    EXPECT_EQ(faces.status, 0) << faces.err;
    const json::Value f = json::parse(faces.out)["restart"]["faces"];
    EXPECT_EQ(f["values"].number, static_cast<double>(output::open(nscbc).restart.n_faces));
    EXPECT_EQ(f["status"].string, "ok");
    const Result no_faces = run({"diff", dir + "v0.8.0/nscbc_000000.restart", nscbc}, true);
    EXPECT_EQ(no_faces.status, 1);
    EXPECT_TRUE(has_row(no_faces.out, {"ERROR", "face records", "0 faces in A, 10 in B"})) << no_faces.out;
    EXPECT_TRUE(contains(no_faces.out, "Result: ERROR, beyond tolerance: RHO, RHOU_X, RHOU_Y, RHOE (exit status 1)")) << no_faces.out;

    const Result mesh = run({"diff", dir + "annulus.msh", later}, true);
    EXPECT_EQ(mesh.status, 3);
    EXPECT_TRUE(contains(mesh.err, "is a mesh file, which holds no field values")) << mesh.err;
}

TEST(CliTest, DiffMatchesCellsByTheirPlaceAndRemapsAcrossMeshes) {
    SKIP_UNDER_MPI();
    const std::string linear = "1.0 + 0.3 * x";
    ASSERT_EQ(diff_case("coarse", 4, linear).status, 0);
    ASSERT_EQ(diff_case("other", 4, linear + " + 0.1 * x * y").status, 0);
    ASSERT_EQ(diff_case("fine", 8, linear).status, 0);

    const Result located = run({"diff", "coarse/restart_000000.restart", "other/restart_000000.restart", "--format", "json"}, true);
    EXPECT_EQ(located.status, 1) << located.err;
    const json::Value rho = json::parse(located.out)["fields"].array.at(0);
    ASSERT_EQ(rho["name"].string, "RHO");
    const auto & x = rho["max_at"]["x"].array;
    ASSERT_EQ(x.size(), size_t(Mallard_DIM));
    EXPECT_NEAR(x[0].number, 0.875, 1e-6);
    EXPECT_NEAR(x[1].number, 5.0 / 6.0, 1e-6);

    // Series of different lengths are matched by time, not position
    ASSERT_EQ(diff_case("longer", 4, linear, 4).status, 0);
    const Result timed = run({"diff", "longer/flow.pvd", "coarse/flow.pvd", "--time", "0.0021", "--format", "json"}, true);
    EXPECT_EQ(timed.status, 0) << timed.err;
    const json::Value t = json::parse(timed.out);
    EXPECT_EQ(t["a"]["snapshot"].number, 1);
    EXPECT_EQ(t["b"]["snapshot"].number, 1);
    EXPECT_EQ(run({"diff", "longer/flow.pvd", "coarse/flow.pvd", "--format", "json"}, true).status, 1);
    const Result no_step = run({"diff", "coarse/restart_000002.restart", "other/restart_000002.restart", "--step", "3"}, true);
    EXPECT_EQ(no_step.status, 3);
    EXPECT_TRUE(contains(no_step.err, "has no snapshot of step 3")) << no_step.err;
    EXPECT_EQ(run({"diff", "longer/flow.pvd", "coarse/flow.pvd", "--step", "2", "--time", "0"}, true).status, 2);

    const Result mixed = run({"diff", "coarse/restart_000002.restart", "coarse/flow_000002.vtu"}, true);
    EXPECT_EQ(mixed.status, 3);
    EXPECT_TRUE(contains(mixed.err, "not the mesh's global cell order")) << mixed.err;

    const Result refused = run({"diff", "coarse/restart_000000.restart", "fine/restart_000000.restart"}, true);
    EXPECT_EQ(refused.status, 3);
    EXPECT_TRUE(contains(refused.err, "different meshes; --remap compares B on A's mesh")) << refused.err;

    // The fine mesh nests in the coarse one: its averages, remapped conservatively, are the coarse averages
    const double rounding = 100.0 * static_cast<double>(std::numeric_limits<rtype>::epsilon());
    char tol[32];
    std::snprintf(tol, sizeof tol, "%.3g", rounding);
    const Result remapped = run({"diff", "coarse/restart_000000.restart", "fine/restart_000000.restart", "--remap", "--tol",
                                 tol, "--format", "json"}, true);
    EXPECT_EQ(remapped.status, 0) << remapped.err << remapped.out;
    const json::Value r = json::parse(remapped.out);
    EXPECT_EQ(r["compared"].string, "remap");
    EXPECT_EQ(r["remap"]["source_cells"].number, Mallard_DIM == 3 ? 8 * 6 * 6 : 8 * 6);
    for (const json::Value & f : r["fields"].array) {
        EXPECT_LT(f["relative"]["linf"].number, rounding) << f["name"].string;
    }
}

TEST(CliTest, CompletionCoversEveryCommandAndOption) {
    for (const std::string shell : {"bash", "zsh", "fish"}) {
        SCOPED_TRACE(shell);
        const std::string script = cli::completion_script(shell);
        ASSERT_FALSE(script.empty());
        auto has = [&](std::string_view name) {
            std::string spelled(name);
            if (shell == "fish" && name.starts_with("--")) {
                spelled = "-l " + spelled.substr(2);
            } else if (shell == "fish" && name.starts_with('-')) {
                spelled = "-s " + spelled.substr(1);
            }
            return std::regex_search(script, std::regex("(^|[ '(])" + spelled + "([ ':=)]|$)"));
        };
        for (const cli::Command & c : cli::commands()) {
            for (const std::string & w : words(c.name)) EXPECT_TRUE(has(w)) << c.name;
            for (const cli::Option & o : c.options) {
                if (o.name.ends_with('*')) continue;
                EXPECT_TRUE(has(o.name)) << c.name << " " << o.name;
                if (!o.alias.empty()) {
                    EXPECT_TRUE(has(o.alias)) << c.name << " " << o.alias;
                }
            }
        }
        for (const cli::Option & o : cli::global_options()) EXPECT_TRUE(has(o.name)) << o.name;
    }
    EXPECT_TRUE(cli::completion_script("tcsh").empty());
}

namespace {

/** @brief Runs a script with a shell, if installed (else -1): its exit status, out its output. */
int run_shell(const std::string & shell, const std::string & flags, const fs::path & script, std::string & out) {
    if (std::system(("command -v " + shell + " > /dev/null 2>&1").c_str()) != 0) return -1;
    const int status = std::system((shell + " " + flags + " " + script.string() + " > shell_out.txt 2>&1").c_str());
    out = read_file("shell_out.txt");
    return WIFEXITED(status) ? WEXITSTATUS(status) : -2;
}

/**
 * @brief Completes each line (the words after mallard; the last is the one completed) with the bash completion,
 *        in dir: got, one line each, " " + the completions sorted, each followed by a space. The status of bash, -1
 *        if it is not installed.
 */
int bash_complete(const fs::path & dir, const std::vector<std::vector<std::string>> & lines,
                  std::vector<std::string> & got) {
    const Result script = run({"completion", "bash"});
    EXPECT_EQ(script.status, 0) << script.err;
    std::ofstream(dir / "mallard.bash") << script.out;
    std::ofstream driver(dir / "driver.sh");
    driver << "export LC_ALL=C\ncd '" << dir.string() << "' && source mallard.bash || exit 1\n";
    for (const auto & line : lines) {
        driver << "COMP_WORDS=('" << mallard() << "'";
        for (const std::string & w : line) driver << " '" << w << "'";
        driver << "); COMP_CWORD=" << line.size() << "; COMPREPLY=(); _mallard\n"
               << "printf '%s\\n' \"${COMPREPLY[@]}\" | sort | tr '\\n' ' '; echo\n";
    }
    driver.close();
    std::string out;
    const int status = run_shell("bash", "--norc --noprofile", dir / "driver.sh", out);
    got.clear();
    std::istringstream in(out);
    for (std::string l; std::getline(in, l);) got.push_back(" " + l);
    if (status == 0) {
        EXPECT_EQ(got.size(), lines.size()) << out;
    } else if (status != -1) {
        ADD_FAILURE() << out;
    }
    got.resize(lines.size());
    return status;
}

} // namespace

TEST(CliTest, BashCompletesCommandsOptionsAndValues) {
    SKIP_UNDER_MPI();
    const fs::path dir = fs::absolute("completion_case");
    fs::remove_all(dir);
    fs::create_directories(dir / "solut");
    fs::create_directories(dir / "run1");
    for (const char * f : {"case.toml", "notes.txt", "mesh.h5", "solut/restart_000010.restart"}) std::ofstream(dir / f);
    const std::vector<std::vector<std::string>> lines = {
        {""},
        {"input", ""},
        {"--res"},
        {"-i", ""},
        {"-i", "case.toml", "--restart", ""},
        {"--restart", "solut/"},
        {"--restart", "=", ""},
        {"ctl", ""},
        {"ctl", "run1", ""},
        {"ctl", "run1", "set", ""},
        {"new", ""},
        {"test", "--gtest_f"},
        {"check", "case.toml", "--kokkos-n"},
        {"--dim", ""},
        {"--dim", "3", "check", "--ranks", ""},
        {"completion", ""},
        {"mesh", "convert", "case.toml", ""},
        {"input", "upgrade", "--dim", ""},
    };
    std::vector<std::string> got;
    const int status = bash_complete(dir, lines, got);
    if (status == -1) GTEST_SKIP() << "no bash";
    ASSERT_EQ(status, 0);
    for (const char * w : {"ctl", "check", "new", "input", "mesh", "test", "completion"}) {
        EXPECT_TRUE(contains(got[0], std::string(" ") + w + " ")) << w << ":" << got[0];
    }
    EXPECT_FALSE(contains(got[0], " upgrade ")) << got[0];
    EXPECT_TRUE(contains(got[1], " upgrade ")) << got[1];
    EXPECT_FALSE(contains(got[1], " check ")) << got[1];
    EXPECT_EQ(got[2], " --restart ");
    EXPECT_EQ(got[3], " case.toml run1 solut ");
    EXPECT_EQ(got[4], " latest run1 solut ");
    EXPECT_EQ(got[5], " solut/restart_000010.restart ");
    EXPECT_EQ(got[6], " latest run1 solut ");
    EXPECT_EQ(got[7], " run1 solut ");
    EXPECT_EQ(got[8], " checkpoint set status stop ");
    EXPECT_EQ(got[9], "  ");
    EXPECT_TRUE(contains(got[10], " cylinder ")) << got[10];
    EXPECT_TRUE(contains(got[10], " sod ")) << got[10];
    EXPECT_EQ(got[11], " --gtest_fail_fast --gtest_filter= ");
    EXPECT_EQ(got[12], " --kokkos-num-threads= ");
    EXPECT_EQ(got[13], " 2 3 ");
    EXPECT_EQ(got[14], "  ");
    EXPECT_EQ(got[15], " bash fish zsh ");
    EXPECT_EQ(got[16], " mesh.h5 run1 solut ");
    EXPECT_EQ(got[17], " 2 3 ");
}

TEST(CliTest, ZshAndFishCompletionParse) {
    SKIP_UNDER_MPI();
    for (const std::string shell : {"zsh", "fish"}) {
        SCOPED_TRACE(shell);
        const Result script = run({"completion", shell});
        ASSERT_EQ(script.status, 0) << script.err;
        const std::string file = "completion." + shell;
        std::ofstream(file) << script.out;
        std::string out;
        const int status = run_shell(shell, "-n", file, out);
        if (status != -1) {
            EXPECT_EQ(status, 0) << out;
        }
    }
}

TEST(CliTest, CompletionOffersTheKeyAndFieldCatalogs) {
    SKIP_UNDER_MPI();
    const Result listed_keys = run({"list", "keys", "--format", "json"}, true);
    ASSERT_EQ(listed_keys.status, 0) << listed_keys.err;
    std::vector<std::string> keys, numerics;
    for (const std::string & key : json::parse(listed_keys.out)["keys"].keys()) {
        const std::string section = key.substr(0, key.find('.'));
        if (std::find(keys.begin(), keys.end(), section) == keys.end()) keys.push_back(section);
        if (key.find('<') != std::string::npos || key == section) continue;
        keys.push_back(key);
        if (key.starts_with("numerics.f")) numerics.push_back(key);
    }
    const Result listed_fields = run({"list", "fields", "--format", "json"}, true);
    ASSERT_EQ(listed_fields.status, 0) << listed_fields.err;
    std::vector<std::string> fields;
    const json::Value field_catalog = json::parse(listed_fields.out);
    for (const json::Value & f : field_catalog["fields"].array) {
        if (f["name"].string.find('<') == std::string::npos) fields.push_back(f["name"].string);
    }
    ASSERT_FALSE(numerics.empty());
    ASSERT_TRUE(std::find(fields.begin(), fields.end(), "MACH") != fields.end());

    const fs::path dir = fs::absolute("completion_catalogs");
    fs::remove_all(dir);
    fs::create_directories(dir);
    std::ofstream(dir / "case.toml");
    const std::vector<std::vector<std::string>> lines = {
        {"list", "keys", ""},
        {"list", "keys", "numerics.f"},
        {"post", "extract", "flow.vtkhdf.series", "--fields", ""},
        {"post", "extract", "flow.vtkhdf.series", "--fields", "RHO,U,"},
        {"diff", "a.restart", "b.restart", "--fields", "=", "RHO,"},
    };
    std::vector<std::string> got;
    const int status = bash_complete(dir, lines, got);
    if (status == -1) GTEST_SKIP() << "no bash";
    ASSERT_EQ(status, 0);
    auto expected = [](std::vector<std::string> words, const std::string & head) {
        for (std::string & w : words) w = head + w;
        std::sort(words.begin(), words.end());
        std::string out = " ";
        for (const std::string & w : words) out += w + " ";
        return out;
    };
    std::vector<std::string> keys_or_input = keys;
    keys_or_input.push_back("case.toml");
    EXPECT_EQ(got[0], expected(keys_or_input, ""));
    EXPECT_EQ(got[1], expected(numerics, ""));
    EXPECT_EQ(got[2], expected(fields, ""));
    EXPECT_EQ(got[3], expected(fields, "RHO,U,"));
    EXPECT_EQ(got[4], expected(fields, "RHO,"));

    for (const std::string shell : {"zsh", "fish"}) {
        SCOPED_TRACE(shell);
        const Result script = run({"completion", shell});
        ASSERT_EQ(script.status, 0) << script.err;
        for (const std::string & w : keys) EXPECT_TRUE(contains(script.out, "'" + w + "'")) << w;
        for (const std::string & w : fields) EXPECT_TRUE(contains(script.out, "'" + w + "'")) << w;
    }
}

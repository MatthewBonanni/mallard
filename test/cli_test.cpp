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
#include <regex>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include <fcntl.h>
#include <sys/wait.h>
#include <unistd.h>

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

TEST(CliTest, UnknownOptionIsOneLinePointingAtHelp) {
    SKIP_UNDER_MPI();
    for (const cli::Command & c : cli::commands()) {
        const std::string title = cli::title(c);
        SCOPED_TRACE(title);
        std::vector<std::string> args = words(c.name);
        args.push_back("--no-such-option");
        const Result r = run(args);
        EXPECT_EQ(r.status, 2);
        EXPECT_EQ(r.err, title + ": unknown option --no-such-option (see " + title + " --help)\n");
    }
    const Result r = run({"no-such-command"});
    EXPECT_EQ(r.status, 2);
    EXPECT_EQ(r.err, "mallard: unknown command no-such-command; to run an input file, give it with -i: "
                     "mallard -i no-such-command (see mallard --help)\n");
    const Result typo = run({"chek", "input.toml"});
    EXPECT_EQ(typo.status, 2);
    EXPECT_EQ(typo.err, "mallard: unknown command chek (did you mean \"check\"?)\n");
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
        << "[[write_data]]\nprefix = \"case/flow\"\nformat = \"vtu\"\ninterval = 2\nvariables = [\"RHO\", \"U\"]\n"
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
        {"field", "units", "components", "min", "max", "stored", "description"},
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
}

TEST(CliTest, InfoOfARunDirectoryListsItsRunAndFiles) {
    SKIP_UNDER_MPI();
    ASSERT_EQ(small_run().status, 0);
    const Result r = run({"info", "."}, true);
    EXPECT_EQ(r.status, 0) << r.err;
    const std::vector<std::vector<std::string>> rows = {
        {"Ended", "finished: n_steps = 2 reached"},
        {"Last step", "2"},
        {"Last snapshot", "case/flow_000002.vtu, step 2, t = 0.002 s"},
        {"Last restart", "case/restart_000002.restart, step 2, t = 0.002 s"},
        {"Simulated time", "t = 0.002 s at the end"},
        {"file", "kind", "times", "first t [s]", "last t [s]", "last step", "fields", "format"},
        {"case/flow.pvd", "snapshot", "2", "0", printed_time(2.0e-3), "2", "2", "PVD series index (VTU snapshots)"},
        {"case/restart_000002.restart", "restart", "1", printed_time(2.0e-3), printed_time(2.0e-3), "2", Mallard_DIM == 3 ? "5" : "4", "restart"},
    };
    for (const auto & row : rows) EXPECT_TRUE(has_row(r.out, row)) << row[0] << "\n" << r.out;
    // A series is listed by its index, not its snapshots
    EXPECT_FALSE(has_row(r.out, {"case/flow_000002.vtu"})) << r.out;
    EXPECT_EQ(run({"info", "--input", "."}, true).out, run({"input", "resolve", "small.toml"}, true).out);
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
    const Result r = run({"info", "--quality", "quality", mesh}, true);
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
        run({"info", "--quality", "quality", source_dir() + "/test/data/formats/gas.toml"}, true);
    EXPECT_EQ(not_a_mesh.status, 2);
    EXPECT_TRUE(contains(not_a_mesh.err, "--quality takes a mesh file")) << not_a_mesh.err;
}

TEST(CliTest, ListFieldsGivesTheCatalogOrOneCasesFields) {
    SKIP_UNDER_MPI();
    const Result catalog = run({"list", "fields"}, true);
    EXPECT_EQ(catalog.status, 0) << catalog.err;
    EXPECT_TRUE(contains(catalog.out, "== Fields: conservative")) << catalog.out;
    const std::vector<std::vector<std::string>> catalog_rows = {
        {"field", "units", "description", "needs"},
        {"RHO", "kg m-3", "density", "every case"},
        {"CFL", "dimensionless", "local CFL number", "every case"},
        {"MU_T", "kg m-1 s-1", "eddy viscosity", "[les]"},
    };
    for (const auto & row : catalog_rows) EXPECT_TRUE(has_row(catalog.out, row)) << row[0] << "\n" << catalog.out;
    EXPECT_TRUE(contains(catalog.out, "MEAN_<A>")) << catalog.out;
    ASSERT_EQ(small_run().status, 0);
    const Result text = run({"list", "fields", "small.toml"}, true);
    ASSERT_EQ(text.status, 0) << text.err;
    const std::vector<std::vector<std::string>> case_rows = {
        {"field", "units", "description"},
        {"RHO", "kg m-3", "density"},
        {"U_X", "m s-1", "x component of velocity"},
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
        << "[[write_data]]\nprefix = \"" << dir << "/flow\"\nformat = \"vtu\"\ninterval = 2\nvariables = [\"RHO\", \"U\"]\n"
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

} // namespace

TEST(CliTest, BashCompletesCommandsOptionsAndValues) {
    SKIP_UNDER_MPI();
    const fs::path dir = fs::absolute("completion_case");
    fs::remove_all(dir);
    fs::create_directories(dir / "solut");
    fs::create_directories(dir / "run1");
    for (const char * f : {"case.toml", "notes.txt", "mesh.h5", "solut/restart_000010.restart"}) std::ofstream(dir / f);
    const Result script = run({"completion", "bash"});
    ASSERT_EQ(script.status, 0) << script.err;
    std::ofstream(dir / "mallard.bash") << script.out;
    // The words after mallard; the last is the one completed
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
    if (status == -1) GTEST_SKIP() << "no bash";
    ASSERT_EQ(status, 0) << out;
    std::vector<std::string> got;
    std::istringstream in(out);
    for (std::string l; std::getline(in, l);) got.push_back(" " + l);
    ASSERT_EQ(got.size(), lines.size()) << out;
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

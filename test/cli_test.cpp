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
#include <filesystem>
#include <fstream>
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
#include "data_paths.h"

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

/** @brief Runs mallard with these arguments, outside any MPI launcher this test runs under; killed after a minute. */
Result run(const std::vector<std::string> & args) {
    std::vector<std::string> command = {mallard()};
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

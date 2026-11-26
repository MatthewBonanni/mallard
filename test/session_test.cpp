/**
 * @file session_test.cpp
 * @brief mallard session answers as its one-shot commands do, rereads a mesh only when it changes, answers every
 *        message of its protocol, and all its output conforms to the schemas of schemas/.
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 */

#include <gtest/gtest.h>

#include <cctype>
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

#include "cli.h"
#include "comm.h"
#include "common_typedef.h"
#include "data_paths.h"
#include "json.h"
#include "json_schema.h"
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
 *         minutes. hold_input: the input stays open until mallard exits. */
Result run(std::vector<std::string> args, const std::string & input = "", bool hold_input = false) {
    args.insert(args.begin(), mallard());
    args.push_back("--kokkos-num-threads=1");
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

/**
 * @brief A report without what differs from process to process: the memory measured, and the GPU sizing
 *        but its inputs, as it adds the device memory measured outside Kokkos (a long-lived session's
 *        CUDA context holds more than a fresh process's).
 */
std::string comparable(json::Value report) {
    for (size_t i = 0; i < report.names.size(); i++) {
        if (report.names[i] == "memory") report.members[i] = json::Value();
        if (report.names[i] != "sizing" || report.members[i].type != json::Value::Type::OBJECT) continue;
        json::Value & sizing = report.members[i];
        for (size_t k = 0; k < sizing.names.size(); k++) {
            const std::string & name = sizing.names[k];
            if (name != "name" && name != "gpu" && name != "memory" && name != "margin" && name != "halo_fraction") {
                sizing.members[k] = json::Value();
            }
        }
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
    std::vector<std::pair<int, std::string>> results = {{1, "mallard-session"}};
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

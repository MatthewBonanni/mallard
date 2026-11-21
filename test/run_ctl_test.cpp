/**
 * @file run_ctl_test.cpp
 * @brief Runs reached from outside: the endpoint record, the routes mallard ctl takes to a run (socket, Slurm,
 *        Kubernetes, SSH, signals, files), the acknowledgments and control records of protocol version 2, and
 *        runs launched and controlled through mallard session.
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 */

#include <gtest/gtest.h>

#include <chrono>
#include <csignal>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <functional>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include <fcntl.h>
#include <poll.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <unistd.h>

#include "comm.h"
#include "common_typedef.h"
#include "data_paths.h"
#include "json.h"
#include "json_schema.h"
#include "run_control.h"
#include "run_endpoint.h"
#include "session/run_ctl.h"
#include "session/session.h"

extern char ** environ;

namespace {

namespace fs = std::filesystem;

const json_schema::Schemas & schemas() {
    static const json_schema::Schemas s(source_dir() + "/schemas");
    return s;
}

std::string violations(const std::string & file, const json::Value & document) {
    std::string text;
    for (const std::string & e : schemas().validate(file, document)) text += e + "\n";
    return text;
}

const std::string CONTROL = "mallard-run-control.v2.schema.json";
const std::string RECORD = "mallard-run-control.v2.schema.json#/$defs/record";
const std::string ENDPOINT = "mallard-run-endpoint.v1.schema.json";
const std::string LAUNCH = "mallard-run-launch.v1.schema.json";

json::Value endpoint(const std::string & host, const std::string & scheduler, bool socket = true) {
    return json::parse(R"({"schema": "mallard-run-endpoint", "version": 1, "run_id": "r1",
        "mallard": {"version": "0", "executable": "/opt/mallard/bin/mallard3d"}, "host": ")" + host +
                       R"(", "pid": 4242, "ranks": 8, "directory": "/scratch/run",
        "socket": )" + (socket ? std::string("\"/scratch/run/mallard.sock\"") : std::string("null")) +
                       R"(, "control_file_interval": 10, "run_log": "mallard.jsonl", "scheduler": )" + scheduler +
                       R"(, "started": "2026-11-20T00:00:00Z", "state": "running"})");
}

std::vector<std::string> kinds(const std::vector<run_ctl::Route> & routes) {
    std::vector<std::string> out;
    for (const auto & r : routes) out.push_back(r.kind + (r.runnable ? "" : "?"));
    return out;
}

run_ctl::Here somewhere(std::vector<std::string> programs) {
    return {"login1", [programs](const std::string & p) { return std::find(programs.begin(), programs.end(), p) != programs.end(); }};
}

std::string joined(const std::vector<std::string> & words) {
    std::string out;
    for (const auto & w : words) out += (out.empty() ? "" : " ") + w;
    return out;
}

} // namespace

TEST(RunCtlTest, RoutesFollowTheEndpointRecord) {
    run_ctl::Request stop;
    stop.action = "stop";
    stop.id = "q1";
    run_ctl::Request status = stop;
    status.action = "status";
    const std::string none = R"({"kind": "none"})";
    const std::string slurm = R"({"kind": "slurm", "job_id": "48213", "step_id": "0", "node": "node0412"})";

    // On the run's host: its socket, then a signal to its rank 0
    EXPECT_EQ(kinds(run_ctl::routes(endpoint("login1", none), stop, somewhere({}))),
              (std::vector<std::string>{"socket", "signal"}));
    EXPECT_EQ(joined(run_ctl::routes(endpoint("login1", none), stop, somewhere({}))[1].argv), "kill -USR2 4242");
    EXPECT_EQ(kinds(run_ctl::routes(endpoint("login1", none, false), status, somewhere({}))), std::vector<std::string>{});

    // In a Slurm job: a step overlapping it on rank 0's node, then scancel; srun not here is a hint
    const auto slurm_routes = run_ctl::routes(endpoint("node0412", slurm), stop, somewhere({"srun", "scancel"}));
    ASSERT_EQ(kinds(slurm_routes), (std::vector<std::string>{"srun", "signal"}));
    EXPECT_EQ(joined(slurm_routes[0].argv),
              "srun --jobid=48213 --overlap --nodes=1 --ntasks=1 -w node0412 /opt/mallard/bin/mallard3d ctl --direct "
              "/scratch/run stop --id q1 --wait done --format json");
    EXPECT_EQ(joined(slurm_routes[1].argv), "scancel --signal=USR2 48213.0");
    EXPECT_EQ(kinds(run_ctl::routes(endpoint("node0412", slurm), status, somewhere({}))), std::vector<std::string>{"srun?"});

    // In a pod: kubectl exec
    const auto pod = run_ctl::routes(endpoint("pod-7", R"({"kind": "kubernetes", "pod": "pod-7", "namespace": "cfd"})"),
                                     status, somewhere({"kubectl"}));
    ASSERT_EQ(kinds(pod), std::vector<std::string>{"kubectl"});
    EXPECT_EQ(joined(pod[0].argv),
              "kubectl exec -n cfd pod-7 -- /opt/mallard/bin/mallard3d ctl --direct /scratch/run status --id q1 --wait "
              "done --format json");

    // From inside the pod or the job's node, they are not needed: a run still without its socket is not reachable
    EXPECT_EQ(kinds(run_ctl::routes(endpoint("login1", R"({"kind": "kubernetes", "pod": "login1", "namespace": "cfd"})", false),
                                    status, somewhere({"kubectl"}))),
              std::vector<std::string>{});
    EXPECT_EQ(kinds(run_ctl::routes(endpoint("login1", slurm, false), stop, somewhere({"srun", "scancel"}))),
              std::vector<std::string>{"signal"});

    // PBS and LSF: ssh to the node, then their signals; control files only when asked for
    run_ctl::Request filed = stop;
    filed.allow_files = true;
    EXPECT_EQ(kinds(run_ctl::routes(endpoint("cn12", R"({"kind": "pbs", "job_id": "991.pbs"})"), filed,
                                    somewhere({"ssh", "qsig"}))),
              (std::vector<std::string>{"ssh", "signal", "file"}));
    EXPECT_EQ(joined(run_ctl::routes(endpoint("cn12", R"({"kind": "lsf", "job_id": "77"})"), stop, somewhere({"bkill"}))[1].argv),
              "bkill -s USR2 77");
}

#ifdef MALLARD_EXECUTABLE

namespace {

std::vector<std::string> child_environment() {
    std::vector<std::string> env;
    for (char ** e = environ; *e; e++) {
        const std::string var = *e;
        bool launcher = false;
        for (const char * prefix : {"OMPI_", "PMIX_", "PRTE_", "PMI_", "HYDRA_", "MPIR_", "I_MPI_"}) {
            launcher = launcher || var.rfind(prefix, 0) == 0;
        }
        if (!launcher) env.push_back(var);
    }
    if (geteuid() == 0) {
        env.push_back("OMPI_ALLOW_RUN_AS_ROOT=1");
        env.push_back("OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1");
    }
    return env;
}

pid_t spawn(const std::vector<std::string> & args, const std::string & dir, int in_fd = -1, int out_fd = -1) {
    std::vector<std::string> env = child_environment();
    std::vector<char *> argv, envp;
    for (const auto & a : args) argv.push_back(const_cast<char *>(a.c_str()));
    for (const auto & e : env) envp.push_back(const_cast<char *>(e.c_str()));
    argv.push_back(nullptr);
    envp.push_back(nullptr);
    const std::string log = dir + "/run.log";
    const pid_t pid = fork();
    if (pid == 0) {
        setpgid(0, 0);
        const int fd = open(log.c_str(), O_WRONLY | O_CREAT | O_APPEND, 0644);
        if (fd < 0 || chdir(dir.c_str()) != 0) _exit(126);
        if (in_fd >= 0) dup2(in_fd, 0);
        dup2(out_fd >= 0 ? out_fd : fd, 1);
        dup2(fd, 2);
        for (int other = 3; other < 1024; other++) close(other);
        execve(argv[0], argv.data(), envp.data());
        _exit(127);
    }
    return pid;
}

std::string read_text(const std::string & path) {
    std::ifstream in(path);
    return std::string(std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>());
}

int wait_exit(pid_t pid, double seconds) {
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::duration<double>(seconds);
    int status = 0;
    while (waitpid(pid, &status, WNOHANG) == 0) {
        if (std::chrono::steady_clock::now() > deadline) {
            kill(-pid, SIGKILL);
            waitpid(pid, &status, 0);
            return -1;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    return WIFEXITED(status) ? WEXITSTATUS(status) : 128 + WTERMSIG(status);
}

bool wait_until(const std::function<bool()> & done, double seconds = 120.0) {
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::duration<double>(seconds);
    while (!done()) {
        if (std::chrono::steady_clock::now() > deadline) return false;
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
    }
    return true;
}

json::Value record_of(const std::string & dir) {
    try {
        return json::parse(read_text(dir + "/" + run_endpoint::FILE_NAME));
    } catch (const std::exception &) {
        return {};
    }
}

std::string state_of(const std::string & dir) {
    const json::Value r = record_of(dir);
    return r.type == json::Value::Type::OBJECT && r.has("state") ? r["state"].string : "";
}

std::vector<json::Value> control_records(const std::string & dir) {
    std::vector<json::Value> out;
    std::ifstream in(dir + "/mallard.jsonl");
    for (std::string line; std::getline(in, line);) {
        const json::Value v = json::parse(line);
        if (v.has("type") && v["type"].string == "control") out.push_back(v);
    }
    return out;
}

/** @brief A long run in a fresh directory: the input's [run] extra lines, its registry in the test's directory. */
std::string write_case(const std::string & name, const std::string & run_extra = "") {
    const bool three = Mallard_DIM == 3;
    fs::create_directories(name);
    std::ofstream(name + "/input.toml")
        << "[run]\nn_steps = 100000000\nt_wall_stop = 300\ncfl = 0.25\n" << run_extra
        << "[mesh]\ntype = \"cartesian\"\nNx = 6\nNy = 5\nLx = 1.0\nLy = 1.0\n"
        << (three ? "Nz = 2\nLz = 1.0\n[[periodic]]\nzones = [\"back\", \"front\"]\n" : "")
        << "[[periodic]]\nzones = [\"left\", \"right\"]\n[[periodic]]\nzones = [\"bottom\", \"top\"]\n"
        << "[initialize]\ntype = \"analytical\"\nrho = \"1.0 + 0.3 * x\"\n"
        << (three ? "u = [\"0.2\", \"0.1\", \"0.0\"]\n" : "u = [\"0.2\", \"0.1\"]\n") << "p = \"1.0\"\n"
        << "[physics]\ntype = \"euler\"\ngamma = 1.4\np_ref = 1.0\nT_ref = 1.0\nrho_ref = 1.0\n"
        << "[output]\ncheck_interval = 1000000\n"
        << "[[write_data]]\nprefix = \"restart\"\nformat = \"restart\"\ninterval = 100000000\n";
    return name;
}

std::vector<std::string> launcher() {
#ifdef MALLARD_MPIEXEC
    std::vector<std::string> args = {MALLARD_MPIEXEC, "-n", "1"};
#ifdef MALLARD_MPIEXEC_OPEN_MPI
    args.push_back("--oversubscribe");
#endif
    return args;
#else
    return {};
#endif
}

pid_t start_run(const std::string & dir) {
    std::vector<std::string> args = launcher();
    args.insert(args.end(), {mallard_executable(), "-i", "input.toml"});
    const pid_t pid = spawn(args, dir);
    EXPECT_TRUE(wait_until([&] { return state_of(dir) == "running" || waitpid(pid, nullptr, WNOHANG) != 0; }))
        << read_text(dir + "/run.log");
    EXPECT_EQ(state_of(dir), "running") << read_text(dir + "/run.log");
    return pid;
}

/** @brief The run registry of the test, in its own directory. */
struct Registry {
    std::string previous;
    explicit Registry(const std::string & dir) {
        const char * p = std::getenv("MALLARD_RUN_REGISTRY");
        previous = p ? p : "";
        setenv("MALLARD_RUN_REGISTRY", (fs::current_path() / dir).c_str(), 1);
    }
    ~Registry() {
        if (previous.empty()) {
            unsetenv("MALLARD_RUN_REGISTRY");
        } else {
            setenv("MALLARD_RUN_REGISTRY", previous.c_str(), 1);
        }
    }
};

json::Value control(const std::string & run, const std::string & action, std::vector<std::string> settings = {},
                    const std::string & wait = "done") {
    run_ctl::Request r;
    r.run = run;
    r.action = action;
    r.settings = std::move(settings);
    r.wait = wait;
    r.timeout = wait == "done" ? 120.0 : 30.0;
    return json::parse(run_ctl::control(r));
}

#define SKIP_UNLESS_ALONE() \
    if (comm::size() > 1) GTEST_SKIP() << "starts its own run"

} // namespace

TEST(RunCtlProcessTest, RunsAreFoundByIdAcknowledgeRequestsAndRecordThem) {
    SKIP_UNLESS_ALONE();
    Registry registry("registry");
    const std::string dir = write_case("by_id");
    const pid_t pid = start_run(dir);
    const json::Value record = record_of(dir);
    EXPECT_EQ(violations(ENDPOINT, record), "");
    EXPECT_EQ(record["pid"].number > 0, true);
    EXPECT_EQ(record["directory"].string, fs::absolute(dir).string());
    const std::string id = record["run_id"].string;
    EXPECT_EQ(read_text("registry/" + id + ".json"), read_text(dir + "/mallard.run.json"));

    // By its ID, from anywhere the registry is
    const json::Value status = control(id, "status");
    EXPECT_EQ(violations(CONTROL, status), "");
    EXPECT_EQ(status["route"].string, "socket");
    EXPECT_EQ(status["status"].string, "done");
    EXPECT_NE(status["replies"].array[0]["values"]["step"].string, "");

    const json::Value set = control(dir, "set", {"check_interval=1000", "cfl=0.5"});
    EXPECT_EQ(violations(CONTROL, set), "");
    ASSERT_EQ(set["replies"].array.size(), 2u) << json::dump(set);
    EXPECT_EQ(set["replies"].array[0]["status"].string, "accepted");
    EXPECT_EQ(set["status"].string, "refused");  // cfl is not settable while running
    const json::Value & applied = set["replies"].array[1]["result"]["settings"];
    ASSERT_EQ(applied.array.size(), 2u);
    EXPECT_TRUE(applied.array[0]["applied"].boolean);
    EXPECT_FALSE(applied.array[1]["applied"].boolean);

    // Accepted at once; done once acted on, and recorded with its ID
    const json::Value accepted = control(id, "checkpoint", {}, "accepted");
    EXPECT_EQ(accepted["status"].string, "accepted");
    ASSERT_TRUE(wait_until([&] {
        for (const json::Value & r : control_records(dir)) {
            if (r["id"].is_null() || r["id"].string != accepted["id"].string) continue;
            return r["status"].string == "done";
        }
        return false;
    })) << read_text(dir + "/mallard.jsonl");
    const json::Value stopped = control(id, "stop");
    EXPECT_EQ(violations(CONTROL, stopped), "");
    EXPECT_EQ(stopped["status"].string, "done");
    const std::string restart = stopped["replies"].array[1]["result"]["restart"]["path"].string;
    EXPECT_TRUE(fs::exists(dir + "/" + restart)) << json::dump(stopped);
    EXPECT_EQ(wait_exit(pid, 120.0), 3) << read_text(dir + "/run.log");

    std::vector<std::string> seen;
    for (const json::Value & r : control_records(dir)) {
        EXPECT_EQ(violations(RECORD, r), "") << json::dump(r);
        seen.push_back(r["action"].string + " " + r["source"].string + " " + r["status"].string);
    }
    EXPECT_EQ(seen, (std::vector<std::string>{"set socket refused", "checkpoint socket done", "stop socket done"}));
    EXPECT_EQ(state_of(dir), "ended");
    EXPECT_EQ(record_of(dir)["outcome"].string, "stopped");
    EXPECT_TRUE(record_of(dir)["socket"].is_null());
    // An ended run answers no more requests, and says how it ended
    try {
        control(id, "status");
        ADD_FAILURE() << "an ended run answered";
    } catch (const session::Error & e) {
        EXPECT_EQ(e.code, session::code::NO_RUN);
        EXPECT_NE(std::string(e.what()).find("stopped"), std::string::npos) << e.what();
    }
}

TEST(RunCtlProcessTest, WithoutASocketStopAndCheckpointGoBySignalsAndAreAcknowledgedInTheRunLog) {
    SKIP_UNLESS_ALONE();
    Registry registry("registry");
    const std::string dir = write_case("by_signal", "control_socket = false\n");
    const pid_t pid = start_run(dir);
    EXPECT_TRUE(record_of(dir)["socket"].is_null());
    const json::Value checkpoint = control(dir, "checkpoint");
    EXPECT_EQ(violations(CONTROL, checkpoint), "");
    EXPECT_EQ(checkpoint["route"].string, "signal");
    EXPECT_EQ(checkpoint["status"].string, "done");
    EXPECT_TRUE(fs::exists(dir + "/" + checkpoint["replies"].array[0]["result"]["restart"]["path"].string))
        << json::dump(checkpoint);
    const json::Value stop = control(dir, "stop");
    EXPECT_EQ(stop["route"].string, "signal");
    EXPECT_EQ(stop["status"].string, "done");
    EXPECT_EQ(wait_exit(pid, 120.0), 3) << read_text(dir + "/run.log");
    std::vector<std::string> seen;
    for (const json::Value & r : control_records(dir)) {
        EXPECT_EQ(violations(RECORD, r), "") << json::dump(r);
        EXPECT_TRUE(r["id"].is_null());
        seen.push_back(r["action"].string + " " + r["source"].string + " " + r["status"].string);
    }
    EXPECT_EQ(seen, (std::vector<std::string>{"checkpoint signal accepted", "checkpoint signal done",
                                              "stop signal accepted", "stop signal done"}));
}

TEST(RunCtlProcessTest, ARunInAPodIsReachedByKubectlExecOrNamedAsTheCommandToRun) {
    SKIP_UNLESS_ALONE();
    Registry registry("registry");
    const std::string dir = write_case("in_pod");
    const pid_t pid = start_run(dir);
    // The record of a run elsewhere, in a pod
    json::Value r = record_of(dir);
    for (size_t i = 0; i < r.names.size(); i++) {
        if (r.names[i] == "host") r.members[i].string = "elsewhere";
        if (r.names[i] == "scheduler") r.members[i] = json::parse(R"({"kind": "kubernetes", "pod": "run-0", "namespace": "cfd"})");
    }
    std::ofstream(dir + "/mallard.run.json") << json::dump(r) << "\n";

    const char * old_path = std::getenv("PATH");
    const std::string path = old_path ? old_path : "/usr/bin:/bin";
    try {
        control(dir, "status");
        ADD_FAILURE() << "reached without kubectl";
    } catch (const session::Error & e) {
        EXPECT_EQ(e.code, session::code::UNREACHABLE) << e.what();
        const json::Value hint = json::parse(e.data)["route_hint"];
        EXPECT_EQ(hint["kind"].string, "kubectl");
        EXPECT_EQ(hint["argv"].array[0].string, "kubectl");
    }
    // A kubectl that runs the command here, as kubectl exec runs it in the pod
    fs::create_directories("bin");
    std::ofstream("bin/kubectl") << "#!/bin/sh\necho \"$@\" >> \"$(dirname \"$0\")/kubectl.log\"\n"
                                    "while [ \"$1\" != \"--\" ]; do shift; done\nshift\nexec \"$@\"\n";
    fs::permissions("bin/kubectl", fs::perms::owner_all);
    setenv("PATH", (fs::absolute("bin").string() + ":" + path).c_str(), 1);
    const json::Value status = control(dir, "status");
    const json::Value stopped = control(dir, "stop");
    setenv("PATH", path.c_str(), 1);
    EXPECT_EQ(violations(CONTROL, status), "");
    EXPECT_EQ(status["route"].string, "kubectl");
    EXPECT_EQ(status["status"].string, "done");
    EXPECT_EQ(stopped["route"].string, "kubectl");
    EXPECT_EQ(stopped["status"].string, "done");
    EXPECT_EQ(wait_exit(pid, 120.0), 3) << read_text(dir + "/run.log");
    const std::string calls = read_text("bin/kubectl.log");
    EXPECT_NE(calls.find("exec -n cfd run-0 -- "), std::string::npos) << calls;
    EXPECT_NE(calls.find(" ctl --direct " + fs::absolute(dir).string() + " stop --id " + stopped["id"].string),
              std::string::npos)
        << calls;
}

namespace {

/** @brief mallard session, driven one request at a time. */
class Session {
    public:
        explicit Session(const std::string & dir) {
            int in[2], out[2];
            if (pipe(in) != 0 || pipe(out) != 0) throw std::runtime_error("pipe");
            std::vector<std::string> args = {mallard_executable(), "session", "--kokkos-num-threads=1"};
            pid = spawn(args, dir, in[0], out[1]);
            close(in[0]);
            close(out[1]);
            to = in[1];
            from = out[0];
        }
        ~Session() {
            close(to);
            wait_exit(pid, 30.0);
            close(from);
        }
        json::Value call(int id, const std::string & method, const std::string & params) {
            const std::string line = "{\"jsonrpc\":\"2.0\",\"id\":" + std::to_string(id) + ",\"method\":\"" + method +
                                     "\",\"params\":" + params + "}\n";
            if (write(to, line.data(), line.size()) != static_cast<ssize_t>(line.size())) return {};
            while (true) {
                const size_t nl = buffer.find('\n');
                if (nl != std::string::npos) {
                    const json::Value m = json::parse(buffer.substr(0, nl));
                    buffer.erase(0, nl + 1);
                    if (m.has("id") && !m.has("method")) return m;
                    continue;
                }
                pollfd p{from, POLLIN, 0};
                if (poll(&p, 1, 180000) <= 0) return {};
                char chunk[4096];
                const ssize_t n = read(from, chunk, sizeof(chunk));
                if (n <= 0) return {};
                buffer.append(chunk, static_cast<size_t>(n));
            }
        }

    private:
        pid_t pid = -1;
        int to = -1, from = -1;
        std::string buffer;
};

} // namespace

TEST(RunCtlProcessTest, ASessionLaunchesARunThatOutlivesItAndControlsIt) {
    SKIP_UNLESS_ALONE();
    if (!launcher().empty()) GTEST_SKIP() << "an MPI build's solver starts under its launcher";
    Registry registry("registry");
    const std::string dir = write_case("launched");
    json::Value launched;
    {
        Session session(".");
        launched = session.call(1, "run.launch", "{\"input\":" + json::quote(dir + "/input.toml") + ",\"run_dir\":" +
                                                     json::quote(dir) + "}");
        ASSERT_TRUE(launched.has("result")) << json::dump(launched) << read_text(dir + "/mallard.out");
        EXPECT_EQ(violations(LAUNCH, launched["result"]), "");
        EXPECT_EQ(launched["result"]["run_id"].string, record_of(dir)["run_id"].string);
        // Unsaved text is not launched
        session.call(2, "case.open", "{\"input\":" + json::quote(dir + "/input.toml") + ",\"text\":\"[run]\\n\"}");
        EXPECT_EQ(session.call(3, "run.launch", "{}")["error"]["code"].number, session::code::UNSAVED);
    }
    // The session has ended; the run goes on
    wait_until([&] { return state_of(dir) != "setup"; });
    ASSERT_EQ(state_of(dir), "running") << read_text(dir + "/mallard.out");
    {
        Session session(".");
        const json::Value stopped =
            session.call(1, "run.control", "{\"run\":" + json::quote(launched["result"]["run_id"].string) + ",\"action\":\"stop\"}");
        ASSERT_TRUE(stopped.has("result")) << json::dump(stopped);
        EXPECT_EQ(violations(CONTROL, stopped["result"]), "");
        EXPECT_EQ(stopped["result"]["status"].string, "done");
    }
    EXPECT_TRUE(wait_until([&] { return state_of(dir) == "ended"; })) << read_text(dir + "/mallard.out");
    EXPECT_EQ(record_of(dir)["outcome"].string, "stopped");
}

#endif

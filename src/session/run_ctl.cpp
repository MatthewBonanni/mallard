/**
 * @file run_ctl.cpp
 * @brief Routing of control requests to runs, and detached launches.
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 */

#include "run_ctl.h"

#include <cerrno>
#include <chrono>
#include <csignal>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <optional>
#include <sstream>
#include <thread>

#include <fcntl.h>
#include <poll.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <sys/wait.h>
#include <unistd.h>

#include "build_info.h"
#include "provenance.h"
#include "run_control.h"
#include "run_endpoint.h"
#include "session.h"

extern char ** environ;

namespace run_ctl {

namespace fs = std::filesystem;
using session::Error;
namespace code = session::code;

namespace {

using Clock = std::chrono::steady_clock;

double epoch_seconds() {
    return std::chrono::duration<double>(std::chrono::system_clock::now().time_since_epoch()).count();
}

std::string text_or_null(const json::Value * v) {
    return v && v->type == json::Value::Type::STRING ? v->string : "";
}

std::string read_file(const fs::path & path) {
    std::ifstream in(path);
    std::stringstream text;
    text << in.rdbuf();
    return text.str();
}

std::string json_list(const std::vector<std::string> & items, bool quote) {
    std::string out = "[";
    for (const std::string & item : items) out += (out.size() > 1 ? "," : "") + (quote ? json::quote(item) : item);
    return out + "]";
}

/** @brief Seconds left before the deadline, for poll; -1 for none. */
int poll_ms(const std::optional<Clock::time_point> & deadline) {
    if (!deadline) return -1;
    const auto left = std::chrono::duration_cast<std::chrono::milliseconds>(*deadline - Clock::now()).count();
    return left > 0 ? static_cast<int>(left) : 0;
}

std::optional<Clock::time_point> deadline_after(double seconds) {
    if (seconds < 0.0) return std::nullopt;
    return Clock::now() + std::chrono::duration_cast<Clock::duration>(std::chrono::duration<double>(seconds));
}

/** @brief A connection to the socket in dir, or -1; long paths are reached from within dir. */
int connect_in(const std::string & dir) {
    const fs::path path = fs::path(dir) / run_control::SOCKET_NAME;
    sockaddr_un address{};
    address.sun_family = AF_UNIX;
    const int fd = socket(AF_UNIX, SOCK_STREAM, 0);
    if (fd < 0) return -1;
    int here = -1;
    std::string name = path.string();
    if (name.size() >= sizeof(address.sun_path)) {
        here = open(".", O_RDONLY | O_CLOEXEC);
        if (here < 0 || chdir(dir.c_str()) != 0) {
            if (here >= 0) close(here);
            close(fd);
            return -1;
        }
        name = run_control::SOCKET_NAME;
    }
    std::memcpy(address.sun_path, name.c_str(), name.size() + 1);
    const int connected = connect(fd, reinterpret_cast<const sockaddr *>(&address), sizeof(address));
    if (here >= 0) {
        if (fchdir(here) != 0) {
            // The working directory cannot move back: nothing better to do than go on
        }
        close(here);
    }
    if (connected != 0) {
        close(fd);
        return -1;
    }
    return fd;
}

bool send_all(int fd, const std::string & text) {
    size_t done = 0;
    while (done < text.size()) {
        const ssize_t n = ::send(fd, text.data() + done, text.size() - done,
#ifdef MSG_NOSIGNAL
                                 MSG_NOSIGNAL
#else
                                 0
#endif
        );
        if (n < 0 && errno == EINTR) continue;
        if (n <= 0) return false;
        done += static_cast<size_t>(n);
    }
    return true;
}

/** @brief The next line from fd: nothing at the end of the connection; throws TIMEOUT at the deadline. */
std::optional<std::string> read_line(int fd, std::string & buffer, const std::optional<Clock::time_point> & deadline,
                                     const std::string & phase) {
    while (true) {
        const size_t newline = buffer.find('\n');
        if (newline != std::string::npos) {
            std::string line = buffer.substr(0, newline);
            buffer.erase(0, newline + 1);
            return line;
        }
        pollfd p{fd, POLLIN, 0};
        const int ready = poll(&p, 1, poll_ms(deadline));
        if (ready < 0 && errno == EINTR) continue;
        if (ready == 0) {
            throw Error(code::TIMEOUT, "no " + phase + " reply before the timeout",
                        json::Object().add("phase", phase).str());
        }
        char chunk[4096];
        const ssize_t n = recv(fd, chunk, sizeof(chunk), 0);
        if (n < 0 && errno == EINTR) continue;
        if (n <= 0) {
            if (buffer.empty()) return std::nullopt;
            return std::exchange(buffer, "");
        }
        buffer.append(chunk, static_cast<size_t>(n));
    }
}

std::string request_json(const Request & r) {
    json::Object o;
    o.add("schema", run_control::SCHEMA).add("version", run_control::VERSION).add("id", r.id).add("action", r.action);
    if (r.action == "set") {
        json::Object settings;
        for (const std::string & s : r.settings) {
            const size_t eq = s.find('=');
            settings.add(s.substr(0, eq), eq == std::string::npos ? std::string() : s.substr(eq + 1));
        }
        o.raw("settings", settings.str());
    }
    return o.str();
}

/** @brief The text line of protocol version 1. */
std::string request_text(const Request & r) {
    std::string text = r.action;
    for (const std::string & s : r.settings) text += " " + s;
    return text;
}

struct Result {
    std::string id, run_id, directory, route, status;
    std::vector<std::string> replies;  // JSON objects

    std::string json() const {
        return json::Object()
            .add("schema", run_control::SCHEMA)
            .add("version", run_control::VERSION)
            .add("id", id)
            .raw("run_id", run_id.empty() ? "null" : json::quote(run_id))
            .add("directory", directory)
            .add("route", route)
            .add("status", status)
            .raw("replies", json_list(replies, false))
            .str();
    }
};

/** @brief Protocol version 2 on the socket of dir; throws UNREACHABLE if nothing answers there. */
void via_socket(const std::string & dir, const Request & r, Result & result) {
    const int fd = connect_in(dir);
    if (fd < 0) {
        throw Error(code::UNREACHABLE, "no run answers on " + (fs::path(dir) / run_control::SOCKET_NAME).string(),
                    json::Object().add("errno", std::string(std::strerror(errno))).str());
    }
    struct Closer {
        int fd;
        ~Closer() { close(fd); }
    } closer{fd};
    if (!send_all(fd, request_json(r) + "\n")) throw Error(code::UNREACHABLE, "cannot send to the run's socket");
    const double accepted_timeout = r.timeout >= 0.0 ? r.timeout : 30.0;
    std::string buffer;
    std::optional<std::string> first = read_line(fd, buffer, deadline_after(accepted_timeout), "accepted");
    if (!first) throw Error(code::UNREACHABLE, "the run closed the connection without replying");
    if (first->empty() || (*first)[0] != '{') {
        // A run of protocol version 1 refuses JSON: ask again in its text
        const std::string text = run_control::send(dir, request_text(r));
        const bool refused = text.rfind("error", 0) == 0;
        result.status = refused ? "refused" : "done";
        result.replies.push_back(json::Object()
                                     .add("id", r.id)
                                     .add("status", result.status)
                                     .add("action", r.action)
                                     .add("message", text)
                                     .str());
        return;
    }
    const json::Value reply = json::parse(*first);
    result.replies.push_back(*first);
    result.status = text_or_null(reply.find("status"));
    if (result.status != "accepted" || r.wait == "accepted") return;
    std::optional<std::string> second = read_line(fd, buffer, deadline_after(r.timeout), "done");
    if (!second) {
        throw Error(code::UNREACHABLE, "the run closed the connection after accepting the request",
                    json::Object().add("phase", "done").str());
    }
    result.replies.push_back(*second);
    result.status = text_or_null(json::parse(*second).find("status"));
}

struct Output {
    int status = -1;
    std::string out, err;
};

/** @brief Runs argv and collects its output; killed at the deadline (status -1). */
Output run_program(const std::vector<std::string> & argv, const std::optional<Clock::time_point> & deadline) {
    int out_pipe[2], err_pipe[2];
    Output result;
    if (pipe(out_pipe) != 0) return result;
    if (pipe(err_pipe) != 0) {
        close(out_pipe[0]);
        close(out_pipe[1]);
        return result;
    }
    std::vector<char *> args;
    for (const auto & a : argv) args.push_back(const_cast<char *>(a.c_str()));
    args.push_back(nullptr);
    const pid_t pid = fork();
    if (pid == 0) {
        setpgid(0, 0);
        const int null = open("/dev/null", O_RDONLY);
        if (null >= 0) dup2(null, 0);
        dup2(out_pipe[1], 1);
        dup2(err_pipe[1], 2);
        close(out_pipe[0]);
        close(err_pipe[0]);
        execvp(args[0], args.data());
        _exit(127);
    }
    close(out_pipe[1]);
    close(err_pipe[1]);
    if (pid < 0) {
        close(out_pipe[0]);
        close(err_pipe[0]);
        return result;
    }
    pollfd fds[2] = {{out_pipe[0], POLLIN, 0}, {err_pipe[0], POLLIN, 0}};
    int open_pipes = 2;
    bool killed = false;
    while (open_pipes > 0) {
        const int ready = poll(fds, 2, poll_ms(deadline));
        if (ready < 0 && errno == EINTR) continue;
        if (ready == 0) {
            kill(-pid, SIGKILL);
            killed = true;
            break;
        }
        for (int i = 0; i < 2; i++) {
            if (fds[i].fd < 0 || !(fds[i].revents & (POLLIN | POLLHUP))) continue;
            char chunk[4096];
            const ssize_t n = read(fds[i].fd, chunk, sizeof(chunk));
            if (n <= 0) {
                close(fds[i].fd);
                fds[i].fd = -1;
                open_pipes--;
            } else {
                (i == 0 ? result.out : result.err).append(chunk, static_cast<size_t>(n));
            }
        }
    }
    for (const pollfd & p : fds) {
        if (p.fd >= 0) close(p.fd);
    }
    int status = 0;
    waitpid(pid, &status, 0);
    result.status = !killed && WIFEXITED(status) ? WEXITSTATUS(status) : -1;
    return result;
}

/** @brief The last line of text that is a JSON object, or nothing. */
std::optional<json::Value> last_json(const std::string & text) {
    std::istringstream lines(text);
    std::optional<json::Value> found;
    for (std::string line; std::getline(lines, line);) {
        if (line.empty() || line[0] != '{') continue;
        try {
            found = json::parse(line);
        } catch (const std::exception &) {
            // not ours
        }
    }
    return found;
}

std::string tail(const std::string & text, size_t n = 400) { return text.size() > n ? text.substr(text.size() - n) : text; }

/** @brief Waits for the control record of the action from source in the run log, written at or after since. */
bool pid_alive(long pid);

std::optional<std::string> wait_record(const fs::path & log, const Request & r, const std::string & source, double since,
                                       const std::optional<Clock::time_point> & deadline, const fs::path & record,
                                       long local_pid) {
    const std::string wanted = r.wait == "accepted" ? "accepted" : "done";
    bool gone = false;
    while (true) {
        std::ifstream in(log);
        for (std::string line; std::getline(in, line);) {
            if (line.find("\"control\"") == std::string::npos) continue;
            try {
                const json::Value v = json::parse(line);
                if (text_or_null(v.find("type")) != "control" || text_or_null(v.find("action")) != r.action ||
                    text_or_null(v.find("source")) != source)
                    continue;
                const json::Value * time = v.find("time");
                if (!time || time->number < since - 1.0) continue;
                const std::string status = text_or_null(v.find("status"));
                if (status == wanted || status == "refused") return line;
            } catch (const std::exception &) {
                // a line being written
            }
        }
        try {
            if (text_or_null(json::parse(read_file(record)).find("state")) == "ended") {
                // The record of a stop is written before the run ends; one more look finds it
                std::ifstream again(log);
                std::string last;
                for (std::string line; std::getline(again, line);) {
                    if (line.find("\"control\"") != std::string::npos && line.find("\"" + source + "\"") != std::string::npos &&
                        line.find("\"" + wanted + "\"") != std::string::npos)
                        last = line;
                }
                if (!last.empty()) return last;
                throw Error(code::NO_RUN, "the run ended before acting on the request");
            }
        } catch (const Error &) {
            throw;
        } catch (const std::exception &) {
            // the record is being replaced
        }
        if (local_pid > 0 && !pid_alive(local_pid)) {
            // Its last records were written before it exited: one more look
            if (!gone) {
                gone = true;
                continue;
            }
            throw Error(code::NO_RUN, "the run ended without acting on the request (a stop signal during its setup ends it at once)");
        }
        if (deadline && Clock::now() > *deadline) {
            throw Error(code::TIMEOUT, "no " + wanted + " control record in " + log.string() + " before the timeout",
                        json::Object().add("phase", wanted).str());
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
}

/** @brief The arguments that run the request on the run's node: mallard ctl --direct DIR ... */
std::vector<std::string> direct_command(const json::Value & endpoint, const Request & r) {
    const json::Value * mallard = endpoint.find("mallard");
    std::string exe = mallard ? text_or_null(mallard->find("executable")) : "";
    if (exe.empty()) exe = "mallard";
    std::vector<std::string> argv = {exe, "ctl", "--direct", text_or_null(endpoint.find("directory")), r.action};
    argv.insert(argv.end(), r.settings.begin(), r.settings.end());
    argv.insert(argv.end(), {"--id", r.id, "--wait", r.wait, "--format", "json"});
    if (r.timeout >= 0.0) argv.insert(argv.end(), {"--timeout", json::number(r.timeout)});
    return argv;
}

bool pid_alive(long pid) { return pid > 0 && (kill(static_cast<pid_t>(pid), 0) == 0 || errno == EPERM); }

} // namespace

Here here() {
    char host[256] = {};
    gethostname(host, sizeof(host) - 1);
    return {host, [](const std::string & program) {
                const char * path = std::getenv("PATH");
                std::istringstream dirs(path ? path : "");
                for (std::string dir; std::getline(dirs, dir, ':');) {
                    if (!dir.empty() && access((fs::path(dir) / program).c_str(), X_OK) == 0) return true;
                }
                return false;
            }};
}

std::vector<Route> routes(const json::Value & endpoint, const Request & r, const Here & h) {
    std::vector<Route> out;
    const std::string host = text_or_null(endpoint.find("host"));
    const bool same_host = host == h.host;
    const json::Value * scheduler = endpoint.find("scheduler");
    const std::string kind = scheduler ? text_or_null(scheduler->find("kind")) : "none";
    const std::string job = scheduler ? text_or_null(scheduler->find("job_id")) : "";
    const bool socket = endpoint.find("socket") && !endpoint["socket"].is_null();
    if (same_host && socket) out.push_back({"socket", {}, true});
    const std::vector<std::string> direct = direct_command(endpoint, r);
    if (kind == "slurm" && !job.empty() && !same_host) {
        std::vector<std::string> argv = {"srun", "--jobid=" + job, "--overlap", "--nodes=1", "--ntasks=1"};
        const std::string node = text_or_null(scheduler->find("node"));
        if (!node.empty() || !host.empty()) argv.insert(argv.end(), {"-w", node.empty() ? host : node});
        argv.insert(argv.end(), direct.begin(), direct.end());
        out.push_back({"srun", argv, h.has_program("srun")});
    }
    if (kind == "kubernetes" && !same_host) {
        std::vector<std::string> argv = {"kubectl", "exec"};
        const std::string ns = text_or_null(scheduler->find("namespace"));
        if (!ns.empty()) argv.insert(argv.end(), {"-n", ns});
        argv.insert(argv.end(), {text_or_null(scheduler->find("pod")), "--"});
        argv.insert(argv.end(), direct.begin(), direct.end());
        out.push_back({"kubectl", argv, h.has_program("kubectl")});
    }
    if ((kind == "pbs" || kind == "lsf") && !same_host && !host.empty()) {
        std::vector<std::string> argv = {"ssh", "-o", "BatchMode=yes", host};
        argv.insert(argv.end(), direct.begin(), direct.end());
        out.push_back({"ssh", argv, h.has_program("ssh")});
    }
    if (r.action == "stop" || r.action == "checkpoint") {
        const std::string sig = r.action == "stop" ? "USR2" : "USR1";
        if (same_host) {
            out.push_back({"signal", {"kill", "-" + sig, std::to_string(static_cast<long>(endpoint["pid"].number))}, true});
        } else if (kind == "slurm" && !job.empty()) {
            const std::string step = text_or_null(scheduler->find("step_id"));
            out.push_back({"signal", {"scancel", "--signal=" + sig, step.empty() ? job : job + "." + step}, h.has_program("scancel")});
        } else if (kind == "pbs" && !job.empty()) {
            out.push_back({"signal", {"qsig", "-s", sig, job}, h.has_program("qsig")});
        } else if (kind == "lsf" && !job.empty()) {
            out.push_back({"signal", {"bkill", "-s", sig, job}, h.has_program("bkill")});
        }
        const json::Value * interval = endpoint.find("control_file_interval");
        if (r.allow_files && interval && interval->number > 0) {
            out.push_back({"file", {r.action == "stop" ? "STOP" : "CHECKPOINT"}, true});
        }
    }
    return out;
}

std::string control(Request r) {
    if (r.action != "status" && r.action != "checkpoint" && r.action != "stop" && r.action != "set") {
        throw Error(code::INVALID_PARAMS, "unknown action " + r.action + " (status, checkpoint, stop, set)");
    }
    if ((r.action == "set") == r.settings.empty()) {
        throw Error(code::INVALID_PARAMS, r.action == "set" ? "set takes settings, key=value" : r.action + " takes no settings");
    }
    if (r.wait != "accepted" && r.wait != "done") throw Error(code::INVALID_PARAMS, "wait is accepted or done");
    if (r.id.empty()) r.id = provenance::new_uuid();
    Result result;
    result.id = r.id;
    if (r.direct) {
        result.directory = r.run;
        result.route = "socket";
        via_socket(r.run, r, result);
        return result.json();
    }
    json::Value endpoint;
    try {
        endpoint = run_endpoint::find(r.run);
    } catch (const std::runtime_error & e) {
        std::error_code error;
        if (fs::exists(fs::path(r.run) / run_control::SOCKET_NAME, error)) {
            // A run that wrote no endpoint record (an older version): its socket alone
            result.directory = r.run;
            result.route = "socket";
            via_socket(r.run, r, result);
            return result.json();
        }
        throw Error(code::NO_RUN, e.what());
    }
    result.run_id = text_or_null(endpoint.find("run_id"));
    result.directory = text_or_null(endpoint.find("directory"));
    const fs::path record = fs::path(result.directory) / run_endpoint::FILE_NAME;
    std::string state = text_or_null(endpoint.find("state"));
    const Here h = here();
    // A run opens its socket when its setup ends: requests made before then wait for it
    const std::optional<Clock::time_point> setup_deadline = deadline_after(r.timeout);
    while (state == "setup") {
        const long pid = static_cast<long>(endpoint["pid"].number);
        if (text_or_null(endpoint.find("host")) == h.host && !pid_alive(pid)) break;
        if (setup_deadline && Clock::now() >= *setup_deadline) {
            throw Error(code::TIMEOUT, "the run is still in its setup after " + std::to_string(r.timeout) + " s");
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
        try {
            endpoint = run_endpoint::find(r.run);
        } catch (const std::exception &) {
            // Mid-rename on a file system without atomic renames: read it again
        }
        state = text_or_null(endpoint.find("state"));
    }
    if (state == "ended") {
        throw Error(code::NO_RUN, "the run ended (" + text_or_null(endpoint.find("outcome")) + ")",
                    json::Object().add("outcome", text_or_null(endpoint.find("outcome"))).str());
    }
    if (text_or_null(endpoint.find("host")) == h.host && !pid_alive(static_cast<long>(endpoint["pid"].number))) {
        throw Error(code::NO_RUN, "the run is gone: no process " + std::to_string(static_cast<long>(endpoint["pid"].number)) +
                                      " on " + h.host + ", and its record says " + state);
    }
    std::vector<std::string> tried;
    std::string hint;
    for (const Route & route : routes(endpoint, r, h)) {
        if (!route.runnable) {
            if (hint.empty()) {
                hint = json::Object().add("kind", route.kind).raw("argv", json_list(route.argv, true)).str();
            }
            tried.push_back(route.kind + ": " + route.argv[0] + " is not on PATH");
            continue;
        }
        result.route = route.kind;
        if (route.kind == "socket") {
            try {
                via_socket(result.directory, r, result);
                return result.json();
            } catch (const Error & e) {
                if (e.code != code::UNREACHABLE) throw;
                tried.push_back("socket: " + std::string(e.what()));
                result.replies.clear();
                continue;
            }
        }
        if (route.kind == "signal" || route.kind == "file") {
            const double since = epoch_seconds();
            if (route.kind == "file") {
                std::ofstream(fs::path(result.directory) / route.argv[0]) << r.id << "\n";
            } else if (route.argv[0] == "kill") {
                if (kill(static_cast<pid_t>(endpoint["pid"].number), r.action == "stop" ? SIGUSR2 : SIGUSR1) != 0) {
                    tried.push_back("signal: " + std::string(std::strerror(errno)));
                    continue;
                }
            } else {
                const Output o = run_program(route.argv, deadline_after(30.0));
                if (o.status != 0) {
                    tried.push_back(route.kind + ": " + route.argv[0] + " failed: " + tail(o.err));
                    continue;
                }
            }
            const json::Value * log = endpoint.find("run_log");
            const fs::path log_path = log && !log->is_null() ? fs::path(result.directory) / log->string : fs::path();
            std::error_code error;
            if (log_path.empty() || !fs::exists(log_path, error)) {
                result.status = "sent";
                return result.json();
            }
            const double timeout = r.timeout >= 0.0 ? r.timeout : r.wait == "accepted" ? 30.0 : -1.0;
            const std::optional<std::string> found =
                wait_record(log_path, r, route.kind, since, deadline_after(timeout), record,
                            text_or_null(endpoint.find("host")) == h.host ? static_cast<long>(endpoint["pid"].number) : 0);
            result.replies.push_back(*found);
            result.status = text_or_null(json::parse(*found).find("status"));
            return result.json();
        }
        const Output o = run_program(route.argv, deadline_after(r.timeout >= 0.0 ? r.timeout + 30.0 : -1.0));
        const std::optional<json::Value> reply = last_json(o.out);
        if (reply && text_or_null(reply->find("schema")) == run_control::SCHEMA) {
            result.status = text_or_null(reply->find("status"));
            if (const json::Value * replies = reply->find("replies")) {
                for (const json::Value & v : replies->array) result.replies.push_back(json::dump(v));
            }
            return result.json();
        }
        if (reply && reply->find("error")) {
            const json::Value & e = (*reply)["error"];
            const int c = static_cast<int>(e["code"].number);
            if (c != code::UNREACHABLE) throw Error(c, text_or_null(e.find("message")), e.find("data") ? json::dump(e["data"]) : "");
        }
        tried.push_back(route.kind + ": exit " + std::to_string(o.status) + (o.err.empty() ? "" : ": " + tail(o.err)));
    }
    json::Object data;
    data.raw("tried", json_list(tried, true));
    if (!hint.empty()) data.raw("route_hint", hint);
    throw Error(code::UNREACHABLE, "no route reached the run" + std::string(hint.empty() ? "" : "; run route_hint.argv where it can"),
                data.str());
}

std::string launch(const Launch & l) {
    if (l.input.empty()) throw Error(code::INVALID_PARAMS, "launch takes input, the input file");
    std::error_code error;
    if (!fs::is_regular_file(l.input, error)) {
        throw Error(code::UNREADABLE, "no input file " + l.input, json::Object().add("path", l.input).str());
    }
    if (!l.launcher.empty()) {
        const std::string first = fs::path(l.launcher[0]).filename().string();
        if (first != "mpirun" && first != "mpiexec" && first != "srun") {
            throw Error(code::INVALID_PARAMS, "launcher starts with mpirun, mpiexec or srun, not " + first);
        }
    }
    const fs::path input = fs::absolute(l.input);
    const fs::path dir = fs::absolute(l.run_dir.empty() ? fs::current_path() : fs::path(l.run_dir));
    fs::create_directories(dir, error);
    if (error) throw Error(code::UNREADABLE, "cannot create " + dir.string() + ": " + error.message());
    const fs::path relative = input.lexically_relative(dir);
    const bool inside = !relative.empty() && relative.begin()->string() != "..";
    std::vector<std::string> argv = l.launcher;
    argv.insert(argv.end(), {executable_path().string(), "-i", inside ? relative.string() : input.string()});
    if (!l.restart.empty()) argv.insert(argv.end(), {"--restart", l.restart});
    argv.insert(argv.end(), l.options.begin(), l.options.end());
    const std::string launch_id = provenance::new_uuid();
    const fs::path output = dir / "mallard.out";

    std::vector<std::string> env;
    for (char ** e = environ; *e; e++) {
        if (std::string(*e).rfind("MALLARD_LAUNCH_ID=", 0) != 0) env.emplace_back(*e);
    }
    env.push_back("MALLARD_LAUNCH_ID=" + launch_id);
    std::vector<char *> args, envp;
    for (auto & a : argv) args.push_back(a.data());
    for (auto & e : env) envp.push_back(e.data());
    args.push_back(nullptr);
    envp.push_back(nullptr);

    // Two forks, so that the run is no child of the caller's; the pipe carries the run's PID, then closes at exec
    int report[2];
    if (pipe(report) != 0) throw Error(code::INTERNAL, std::string("pipe: ") + std::strerror(errno));
    const pid_t child = fork();
    if (child == 0) {
        close(report[0]);
        if (setsid() < 0) _exit(126);
        const pid_t run = fork();
        if (run != 0) _exit(run < 0 ? 126 : 0);
        fcntl(report[1], F_SETFD, FD_CLOEXEC);
        const long pid = static_cast<long>(getpid());
        if (write(report[1], &pid, sizeof(pid)) != sizeof(pid)) _exit(126);
        // Nothing of the caller's stays open in the run: a client waiting for the caller's output to end would wait
        // for the run
        for (int fd = 3; fd < 1024; fd++) {
            if (fd != report[1]) close(fd);
        }
        if (chdir(dir.c_str()) != 0) _exit(126);
        const int null = open("/dev/null", O_RDONLY);
        const int out = open(output.c_str(), O_WRONLY | O_CREAT | O_APPEND, 0644);
        if (null < 0 || out < 0) _exit(126);
        dup2(null, 0);
        dup2(out, 1);
        dup2(out, 2);
        execve(args[0], args.data(), envp.data());
        const int err = errno;
        if (write(report[1], &err, sizeof(err)) < 0) _exit(127);
        _exit(127);
    }
    close(report[1]);
    if (child < 0) {
        close(report[0]);
        throw Error(code::INTERNAL, std::string("fork: ") + std::strerror(errno));
    }
    int status = 0;
    waitpid(child, &status, 0);
    long pid = 0;
    const bool got_pid = read(report[0], &pid, sizeof(pid)) == static_cast<ssize_t>(sizeof(pid));
    int exec_errno = 0;
    const bool exec_failed = read(report[0], &exec_errno, sizeof(exec_errno)) == static_cast<ssize_t>(sizeof(exec_errno));
    close(report[0]);
    if (!got_pid || exec_failed) {
        throw Error(code::INTERNAL, "cannot start " + argv[0] + (exec_failed ? ": " + std::string(std::strerror(exec_errno)) : ""));
    }
    const fs::path record = dir / run_endpoint::FILE_NAME;
    const auto deadline = deadline_after(l.timeout);
    while (true) {
        try {
            const json::Value v = json::parse(read_file(record));
            if (text_or_null(v.find("launch_id")) == launch_id) {
                return json::Object()
                    .add("schema", LAUNCH_SCHEMA)
                    .add("version", LAUNCH_VERSION)
                    .add("run_id", text_or_null(v.find("run_id")))
                    .add("directory", dir.string())
                    .add("pid", static_cast<int64_t>(v["pid"].number))
                    .add("launcher_pid", static_cast<int64_t>(pid))
                    .add("endpoint", record.string())
                    .add("output", output.string())
                    .raw("argv", json_list(argv, true))
                    .str();
            }
        } catch (const std::exception &) {
            // not written yet, or another run's
        }
        if (!pid_alive(pid)) {
            throw Error(code::NO_RUN, "the run ended before writing its endpoint record; " + output.string() + ": " +
                                          tail(read_file(output), 800),
                        json::Object().add("output", output.string()).str());
        }
        if (deadline && Clock::now() > *deadline) {
            throw Error(code::TIMEOUT, "no endpoint record from the run within the timeout",
                        json::Object().add("output", output.string()).add("launcher_pid", static_cast<int64_t>(pid)).str());
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
    }
}

} // namespace run_ctl

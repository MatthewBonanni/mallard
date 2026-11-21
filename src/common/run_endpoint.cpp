/**
 * @file run_endpoint.cpp
 * @brief The endpoint record of a run and the run registry.
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 */

#include "run_endpoint.h"

#include <chrono>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <sstream>
#include <stdexcept>

#include <unistd.h>

#include "build_info.h"
#include "diagnostic_code.h"
#include "log.h"
#include "run_control.h"

namespace run_endpoint {

namespace fs = std::filesystem;

namespace {

struct Fixed {
    std::string run_id, host, started, directory, scheduler, launch_id;
    int ranks = 1;
    long pid = 0;
    bool ended = false;
    State last;
};

std::optional<Fixed> fixed;

std::string env(const char * name) {
    const char * v = std::getenv(name);
    return v ? v : "";
}

std::string read_file(const fs::path & path) {
    std::ifstream in(path);
    std::stringstream text;
    text << in.rdbuf();
    return text.str();
}

/** @brief Writes text to path through path.partial and a rename, so that a reader never sees half a record. */
bool write_atomically(const fs::path & path, const std::string & text) {
    const fs::path partial = path.string() + ".partial";
    {
        std::ofstream out(partial, std::ios::trunc);
        if (!(out << text << '\n')) return false;
    }
    std::error_code error;
    fs::rename(partial, path, error);
    return !error;
}

/** @brief Removes the registry's entries of runs that ended more than 30 days ago. */
void prune(const fs::path & dir) {
    std::error_code error;
    const auto month_ago = fs::file_time_type::clock::now() - std::chrono::hours(24 * 30);
    for (const auto & entry : fs::directory_iterator(dir, error)) {
        if (entry.path().extension() != ".json" || entry.last_write_time(error) > month_ago) continue;
        try {
            if (json::parse(read_file(entry.path()))["state"].string == "ended") fs::remove(entry.path(), error);
        } catch (const std::exception &) {
            // Not a record: not ours to remove
        }
    }
}

std::string record_json(const Fixed & f, const State & s) {
    json::Object mallard;
    mallard.add("version", mallard_version()).add("executable", executable_path().string());
    json::Object o;
    o.add("schema", SCHEMA).add("version", VERSION).add("run_id", f.run_id).raw("mallard", mallard.str());
    o.add("host", f.host).add("pid", static_cast<int64_t>(f.pid)).add("ranks", f.ranks).add("directory", f.directory);
    o.raw("socket", s.socket ? json::quote((fs::path(f.directory) / run_control::SOCKET_NAME).string()) : "null");
    o.add("control_file_interval", s.control_file_interval);
    o.raw("run_log", s.run_log.empty() ? "null" : json::quote(s.run_log));
    o.raw("scheduler", f.scheduler).add("started", f.started).add("state", s.state);
    if (!s.outcome.empty()) o.add("outcome", s.outcome);
    if (!f.launch_id.empty()) o.add("launch_id", f.launch_id);
    return o.str();
}

} // namespace

std::optional<fs::path> registry() {
    const std::string own = env("MALLARD_RUN_REGISTRY");
    if (own == "off") return std::nullopt;
    if (!own.empty()) return fs::path(own);
    const std::string state = env("XDG_STATE_HOME");
    if (!state.empty()) return fs::path(state) / "mallard" / "runs";
    const std::string home = env("HOME");
    if (home.empty()) return std::nullopt;
    return fs::path(home) / ".local" / "state" / "mallard" / "runs";
}

std::string scheduler_json() {
    json::Object o;
    if (!env("SLURM_JOB_ID").empty()) {
        o.add("kind", "slurm").add("job_id", env("SLURM_JOB_ID"));
        o.raw("step_id", env("SLURM_STEP_ID").empty() ? "null" : json::quote(env("SLURM_STEP_ID")));
        const std::string node = env("SLURMD_NODENAME");
        o.raw("node", node.empty() ? "null" : json::quote(node));
    } else if (!env("PBS_JOBID").empty()) {
        o.add("kind", "pbs").add("job_id", env("PBS_JOBID"));
    } else if (!env("LSB_JOBID").empty()) {
        o.add("kind", "lsf").add("job_id", env("LSB_JOBID"));
    } else if (!env("KUBERNETES_SERVICE_HOST").empty()) {
        std::string ns = env("MALLARD_NAMESPACE");
        if (ns.empty()) {
            ns = read_file("/var/run/secrets/kubernetes.io/serviceaccount/namespace");
            while (!ns.empty() && (ns.back() == '\n' || ns.back() == ' ')) ns.pop_back();
        }
        const std::string pod = env("MALLARD_POD").empty() ? env("HOSTNAME") : env("MALLARD_POD");
        o.add("kind", "kubernetes").add("pod", pod).raw("namespace", ns.empty() ? "null" : json::quote(ns));
    } else {
        o.add("kind", "none");
    }
    return o.str();
}

void write(const std::string & run_id, int ranks, const State & state) {
    if (!fixed) {
        Fixed f;
        char host[256] = {};
        gethostname(host, sizeof(host) - 1);
        char started[32] = {};
        const std::time_t now = std::time(nullptr);
        std::strftime(started, sizeof(started), "%Y-%m-%dT%H:%M:%SZ", std::gmtime(&now));
        std::error_code error;
        f.run_id = run_id;
        f.host = host;
        f.started = started;
        f.directory = fs::current_path(error).string();
        f.scheduler = scheduler_json();
        f.launch_id = env("MALLARD_LAUNCH_ID");
        f.ranks = ranks;
        f.pid = static_cast<long>(getpid());
        fixed = f;
    }
    fixed->last = state;
    fixed->ended = state.state == "ended";
    const std::string text = record_json(*fixed, state);
    if (!write_atomically(FILE_NAME, text)) {
        logging::warning(diagnostic_code::CONTROL_SOCKET, std::string("cannot write ") + FILE_NAME +
                                                              ": mallard ctl finds this run only by its socket");
    }
    if (const std::optional<fs::path> dir = registry()) {
        std::error_code error;
        fs::create_directories(*dir, error);
        if (!error && write_atomically(*dir / (fixed->run_id + ".json"), text) && state.state == "setup") prune(*dir);
    }
}

bool open() { return fixed && !fixed->ended; }

void end(const std::string & outcome) {
    if (!open()) return;
    State s = fixed->last;
    s.state = "ended";
    s.outcome = outcome;
    s.socket = false;
    write(fixed->run_id, fixed->ranks, s);
}

json::Value find(const std::string & run) {
    std::error_code error;
    if (fs::is_directory(run, error)) {
        const fs::path record = fs::path(run) / FILE_NAME;
        if (!fs::exists(record, error)) throw std::runtime_error("no " + record.string() + ": no run of this version started there");
        return json::parse(read_file(record));
    }
    const std::optional<fs::path> dir = registry();
    const fs::path entry = dir ? *dir / (run + ".json") : fs::path();
    if (!dir || !fs::exists(entry, error)) {
        throw std::runtime_error("no run directory " + run + ", and no run of that ID in the registry" +
                                 (dir ? " (" + dir->string() + ")" : ""));
    }
    return json::parse(read_file(entry));
}

} // namespace run_endpoint

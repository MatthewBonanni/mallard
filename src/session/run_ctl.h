/**
 * @file run_ctl.h
 * @brief Control and launch of runs from outside them: `mallard ctl` and the session's run.control route a request
 *        to a run by its endpoint record (socket, Slurm step, Kubernetes, SSH, scheduler signal, control file) and
 *        wait for its acknowledgments; `mallard -i INPUT --detach` and run.launch start a run that outlives them
 *        (docs/guide/running.md#controlling-a-run-from-anywhere).
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 */

#ifndef RUN_CTL_H
#define RUN_CTL_H

#include <functional>
#include <string>
#include <vector>

#include "json.h"

namespace run_ctl {

inline constexpr const char * LAUNCH_SCHEMA = "mallard-run-launch";
inline constexpr int LAUNCH_VERSION = 1;

/** @brief A control request for a run. */
struct Request {
    std::string run;                    // the run directory, or the run ID (the registry finds it)
    std::string action;                 // status, checkpoint, stop, set
    std::vector<std::string> settings;  // set: "key=value"
    std::string id;                     // the request ID; made if empty
    std::string wait = "done";          // accepted or done
    double timeout = -1.0;              // seconds; negative: 30 for accepted, none for done
    bool direct = false;                // the socket in run alone: the remote end of a route
    bool allow_files = false;           // control files as the last route
};

/** @brief A way to reach a run. */
struct Route {
    std::string kind;               // socket, srun, kubectl, ssh, signal, file
    std::vector<std::string> argv;  // the command of srun, kubectl, ssh and scheduler signals
    bool runnable = true;           // false: its program is not here, so the caller must run argv (a hint)
};

/** @brief Where routes are looked for: this host and the programs on PATH. */
struct Here {
    std::string host;
    std::function<bool(const std::string &)> has_program;
};

/** @brief This host, and PATH. */
Here here();

/** @brief The routes to the run of the endpoint record, in the order to try them (docs/guide/running.md#controlling-a-run-from-anywhere). */
std::vector<Route> routes(const json::Value & endpoint, const Request & request, const Here & here);

/**
 * @brief Sends the request by the first route that reaches the run and waits as it asks: the result
 *        (mallard-run-control), whose status is accepted, done, refused or sent (a signal or file without a
 *        run log to confirm it). @throws session::Error: NO_RUN, UNREACHABLE (with a route_hint when a route
 *        must run elsewhere), TIMEOUT, INVALID_PARAMS
 */
std::string control(Request request);

/** @brief A run to start, detached from its caller. */
struct Launch {
    std::string input;                  // the input file
    std::string run_dir;                // where the run runs; default: the working directory
    std::string restart;                // --restart
    std::vector<std::string> launcher;  // e.g. {"mpirun", "-n", "4"}: mpirun, mpiexec or srun
    std::vector<std::string> options;   // more arguments of mallard -i (Kokkos options)
    double timeout = 60.0;              // seconds to wait for the run's endpoint record
};

/**
 * @brief Starts the run in a session of its own, with stdin from /dev/null and its output in mallard.out of the
 *        run directory, and returns once it has written its endpoint record: the result (mallard-run-launch).
 *        @throws session::Error
 */
std::string launch(const Launch & launch);

} // namespace run_ctl

#endif // RUN_CTL_H

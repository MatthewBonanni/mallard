/**
 * @file run_control.h
 * @author Matthew Bonanni (mbonanni001@gmail.com)
 * @brief Control of a running solver: signals, the control socket served to
 *        `mallard ctl`, and the requests they leave for the time loop.
 * @version 0.1
 * @date 2026-10-15
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 *
 * Signal handlers and the socket's thread only record requests; the time loop
 * reads them once per step, on every rank, through the time-step reduction,
 * so all ranks act on them after the same step.
 */

#ifndef RUN_CONTROL_H
#define RUN_CONTROL_H

#include <cstdint>
#include <string>
#include <thread>
#include <vector>

#include "run_status.h"

namespace run_control {

/**
 * @brief Requests, as bits: write a restart and continue, write one and stop, commands queued, the wall time
 *        limit reached (on any rank: every rank then stops after the same step).
 */
enum Request : int { CHECKPOINT = 1, STOP = 2, COMMAND = 4, WALL_TIME = 8 };

/** @brief Number of request bits, each carried by one value of the time-step reduction. */
constexpr int N_REQUESTS = 4;

/** @brief Name of the control socket in the run directory. */
inline constexpr const char * SOCKET_NAME = "mallard.sock";

/**
 * @brief Handlers of SIGUSR1 (checkpoint), and SIGTERM, SIGINT and SIGUSR2
 *        (stop). Until accepting(true), the stop signals end the process at
 *        once, as there is nothing to save; a second SIGINT always does.
 */
void install_signal_handlers();

/**
 * @brief Whether stop signals are deferred to the time loop. Turning it on
 *        restores the handlers if a library replaced them during setup.
 */
void set_accepting(bool on);

/** @brief Records requests as a signal would. */
void request(int bits);

/** @brief Requests not yet acted on. */
int pending();

/** @brief Marks requests as acted on. */
void clear(int bits);

/** @brief Signal that asked this process to stop, or 0. */
int stop_signal();

/** @brief "SIGTERM" etc. */
std::string signal_name(int signal);

/** @brief Name and version of the control protocol's JSON requests and replies (version 1: text lines). */
inline constexpr const char * SCHEMA = "mallard-run-control";
inline constexpr int VERSION = 2;

/** @brief A control request: from the socket, or from a signal or control file (no connection, no ID). */
struct Command {
    std::string text;              // "checkpoint", "stop" or "set key=value ..."
    int fd = -1;                   // connection to answer, closed by reply() or finish()
    std::string id;                // the client's request ID (protocol version 2), else empty
    bool json = false;             // answered with JSON lines (version 2)
    std::string source = "socket"; // socket, signal or file
    /** @brief "checkpoint", "stop" or "set". */
    std::string action() const;
};

/** @brief How a request ended, for its reply and its control record of the run log. */
struct Done {
    bool ok = true;
    std::string message;           // the text reply of protocol version 1
    uint64_t step = 0;
    double t = 0.0;
    std::string result = "null";   // JSON: {"restart": {...}} for a checkpoint or stop, {"settings": [...]} for set
};

/**
 * @brief Answers the request (version 1: the message; version 2: a done or refused line) and closes its
 *        connection; rank 0 also writes the request's control record to the run log.
 */
void finish(Command & command, const Done & done);

/** @brief The result of a checkpoint or stop: {"restart": {"path": file}}, or {"restart": null} without one. */
std::string restart_result(const std::string & file);

/** @brief A control record of the run log for a request accepted (signals and control files, which have no
 *         connection to answer). */
void record_accepted(const Command & command, uint64_t step, double t);

/** @brief Commands received since the last call; clears the COMMAND request. */
std::vector<Command> take_commands();

/** @brief Queues a command as the socket would, without a connection (tests). */
void submit(const std::string & text);

/**
 * @brief A request line of protocol version 2 as a command: {"schema": "mallard-run-control", "version": 2,
 *        "id": ..., "action": "stop"|"checkpoint"|"status"|"set", "settings": {...}}. @throws std::runtime_error
 *        saying what is wrong
 */
Command parse_request(const std::string & line);

/** @brief Sends the reply and closes the connection. */
void reply(Command & command, const std::string & text);

/**
 * @brief What `status` reports, published by the time loop at each step.
 */
struct Status {
    RunStatus run;                 // memory and monitored values are not used
    int device = -1;               // CUDA or HIP device of rank 0, -1 for other backends
    std::string stop_conditions;
    std::vector<std::pair<std::string, std::string>> monitors;  // last progress row's diagnostics
};

void publish(const Status & status);

/**
 * @brief The control socket of rank 0, answered by a thread of its own:
 *        `status` at once, other commands queued for the time loop.
 */
class Server {
    public:
        Server() = default;
        ~Server();
        Server(const Server &) = delete;
        Server & operator=(const Server &) = delete;

        /** @brief Listens on SOCKET_NAME in the working directory; false (with a warning) if it cannot. */
        bool start();
        /** @brief Stops listening, answers queued commands and removes the socket. */
        void stop();
        bool running() const { return listen_fd >= 0; }

    private:
        void serve();
        int listen_fd = -1;
        int wake[2] = {-1, -1};
        std::thread thread;
};

/**
 * @brief `mallard ctl <run-dir> <command>`: sends the command to the socket of
 *        the run in run_dir and prints the reply. Returns the exit status.
 */
int ctl(const std::string & run_dir, const std::string & command);

/** @brief Sends one command line to the socket in run_dir; the reply, or throws. */
std::string send(const std::string & run_dir, const std::string & command);

} // namespace run_control

#endif // RUN_CONTROL_H

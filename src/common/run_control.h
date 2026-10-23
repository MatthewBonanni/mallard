/**
 * @file run_control.h
 * @author Matthew Bonanni (mbonanni001@gmail.com)
 * @brief Control of a running solver: signals, the control socket served to
 *        `Mallard ctl`, and the requests they leave for the time loop.
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

/** @brief Requests, as bits: write a restart and continue, write one and stop, commands queued. */
enum Request : int { CHECKPOINT = 1, STOP = 2, COMMAND = 4 };

/** @brief Number of request bits, each carried by one value of the time-step reduction. */
constexpr int N_REQUESTS = 3;

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

/** @brief A command received on the control socket. */
struct Command {
    std::string text;  // "checkpoint", "stop" or "set key=value ..."
    int fd = -1;       // connection to answer, closed by reply()
};

/** @brief Commands received since the last call; clears the COMMAND request. */
std::vector<Command> take_commands();

/** @brief Queues a command as the socket would, without a connection (tests). */
void submit(const std::string & text);

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
 * @brief `Mallard ctl <run-dir> <command>`: sends the command to the run's
 *        socket and prints the reply. Returns the exit status.
 */
int ctl_main(int argc, char ** argv);

/** @brief Sends one command line to the socket in run_dir; the reply, or throws. */
std::string send(const std::string & run_dir, const std::string & command);

} // namespace run_control

#endif // RUN_CONTROL_H

/**
 * @file run_control.cpp
 * @author Matthew Bonanni (mbonanni001@gmail.com)
 * @brief Signals, the control socket and `mallard ctl`.
 * @version 0.1
 * @date 2026-10-15
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 *
 */

#include "run_control.h"

#include <atomic>
#include <cerrno>
#include <csignal>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <mutex>
#include <stdexcept>
#include <utility>

#include <fcntl.h>
#include <poll.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <sys/un.h>
#include <unistd.h>

#include <Kokkos_Macros.hpp>
#if defined(KOKKOS_ENABLE_CUDA)
#include <cuda_runtime_api.h>
#elif defined(KOKKOS_ENABLE_HIP)
#include <hip/hip_runtime_api.h>
#endif

#include "log.h"

namespace run_control {

namespace {

static_assert(std::atomic<int>::is_always_lock_free, "signal handlers need lock-free atomics");

std::atomic<int> requests{0};
std::atomic<int> stop_signo{0};
std::atomic<int> n_interrupts{0};
std::atomic<bool> accepting{false};
bool installed = false;
constexpr int HANDLED[] = {SIGTERM, SIGINT, SIGUSR1, SIGUSR2};

std::mutex queue_mutex;
std::vector<Command> queue;

std::mutex status_mutex;
Status status;
bool status_published = false;

extern "C" void on_signal(int signo) {
    if (signo == SIGUSR1) {
        requests.fetch_or(CHECKPOINT);
        return;
    }
    if (!accepting.load()) _exit(128 + signo);
    if (signo == SIGINT && n_interrupts.fetch_add(1) > 0) _exit(128 + signo);
    int expected = 0;
    stop_signo.compare_exchange_strong(expected, signo);
    requests.fetch_or(STOP);
}

std::string trim(std::string s) {
    while (!s.empty() && (s.back() == '\n' || s.back() == '\r' || s.back() == ' ')) s.pop_back();
    size_t a = 0;
    while (a < s.size() && s[a] == ' ') a++;
    return s.substr(a);
}

bool write_all(int fd, const std::string & text) {
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

bool same_user(int fd) {
#if defined(__linux__)
    struct ucred cred {};
    socklen_t len = sizeof(cred);
    if (getsockopt(fd, SOL_SOCKET, SO_PEERCRED, &cred, &len) != 0) return false;
    return cred.uid == geteuid();
#else
    uid_t uid = 0;
    gid_t gid = 0;
    if (getpeereid(fd, &uid, &gid) != 0) return false;
    return uid == geteuid();
#endif
}

sockaddr_un socket_address(const std::string & path) {
    sockaddr_un address {};
    address.sun_family = AF_UNIX;
    if (path.size() >= sizeof(address.sun_path)) {
        throw std::runtime_error("socket path too long (" + std::to_string(path.size()) + " characters): " + path);
    }
    std::memcpy(address.sun_path, path.c_str(), path.size() + 1);
    return address;
}

int connect_to(const std::string & path) {
    const sockaddr_un address = socket_address(path);
    const int fd = socket(AF_UNIX, SOCK_STREAM, 0);
    if (fd < 0) throw std::runtime_error(std::string("socket: ") + std::strerror(errno));
    if (connect(fd, reinterpret_cast<const sockaddr *>(&address), sizeof(address)) != 0) {
        const int err = errno;
        close(fd);
        errno = err;
        return -1;
    }
    return fd;
}

void set_cloexec(int fd) { fcntl(fd, F_SETFD, FD_CLOEXEC); }

std::string bytes(double b) {
    const char * units[] = {"B", "KiB", "MiB", "GiB", "TiB"};
    int u = 0;
    while (b >= 1024.0 && u < 4) {
        b /= 1024.0;
        u++;
    }
    return u == 0 ? logging::format("%.0f B", b) : logging::format("%.2f %s", b, units[u]);
}

std::string memory_lines(int device) {
    // The device of rank 0's solver, not this thread's default
    bool on_device = device >= 0;
#if defined(KOKKOS_ENABLE_CUDA)
    on_device = on_device && cudaSetDevice(device) == cudaSuccess;
#elif defined(KOKKOS_ENABLE_HIP)
    on_device = on_device && hipSetDevice(device) == hipSuccess;
#endif
    const MemoryUsage m = memory_usage();
    std::string text;
    if (m.host >= 0.0) text += "memory_host: " + bytes(m.host) + " (peak " + bytes(m.host_peak) + ", rank 0)\n";
    if (on_device && m.device >= 0.0) {
        text += "memory_device: " + bytes(m.device) + " of " + bytes(m.device_total) + " (device " +
                std::to_string(device) + ", all processes)\n";
    }
    return text;
}

std::string status_text() {
    Status s;
    {
        std::lock_guard<std::mutex> lock(status_mutex);
        if (!status_published) return "state: starting\n";
        s = status;
    }
    using logging::format;
    std::string text = "state: running\n";
    text += format("step: %llu\n", static_cast<unsigned long long>(s.run.step));
    text += format("t: %.10g\n", s.run.t);
    text += format("dt: %.6g\n", s.run.dt);
    text += format("cfl: %.4g\n", s.run.cfl);
    text += format("progress: %.2f%%\n", 100.0 * s.run.progress);
    text += "stop_at: " + s.stop_conditions + "\n";
    text += "wall: " + logging::duration(s.run.wall) + "\n";
    if (s.run.wall_per_step > 0.0) {
        text += "wall_per_step: " + logging::duration(s.run.wall_per_step) + "\n";
        text += "cells_per_second: " + logging::si(s.run.cells_per_second) + "\n";
    }
    text += "eta: " + (s.run.eta >= 0.0 ? logging::duration(s.run.eta) : std::string("--")) + "\n";
    text += format("cells: %llu\nranks: %d\n", static_cast<unsigned long long>(s.run.n_cells), s.run.n_ranks);
    text += memory_lines(s.device);
    for (const auto & [key, value] : s.monitors) text += key + ": " + value + "\n";
    return text;
}

} // namespace

void install_signal_handlers() {
    struct sigaction action {};
    action.sa_handler = on_signal;
    sigemptyset(&action.sa_mask);
    action.sa_flags = SA_RESTART;
    for (int signo : HANDLED) sigaction(signo, &action, nullptr);
    installed = true;
}

void set_accepting(bool on) {
    if (on && installed) {
        // MPI, CUDA and their libraries may take a signal over during setup
        std::string replaced;
        for (int signo : HANDLED) {
            struct sigaction current {};
            sigaction(signo, nullptr, &current);
            if (current.sa_handler != on_signal) replaced += (replaced.empty() ? "" : ", ") + signal_name(signo);
        }
        if (!replaced.empty()) {
            install_signal_handlers();
            logging::warning("the handlers of " + replaced + " were replaced during setup; Mallard's are restored");
        }
    }
    accepting.store(on);
}

void request(int bits) { requests.fetch_or(bits); }

int pending() { return requests.load(); }

void clear(int bits) { requests.fetch_and(~bits); }

int stop_signal() { return stop_signo.load(); }

std::string signal_name(int signo) {
    switch (signo) {
        case SIGTERM: return "SIGTERM";
        case SIGINT: return "SIGINT";
        case SIGUSR1: return "SIGUSR1";
        case SIGUSR2: return "SIGUSR2";
        default: return "signal " + std::to_string(signo);
    }
}

std::vector<Command> take_commands() {
    std::lock_guard<std::mutex> lock(queue_mutex);
    requests.fetch_and(~COMMAND);
    return std::exchange(queue, {});
}

void submit(const std::string & text) {
    std::lock_guard<std::mutex> lock(queue_mutex);
    queue.push_back(Command{text, -1});
    requests.fetch_or(COMMAND);
}

void reply(Command & command, const std::string & text) {
    if (command.fd < 0) return;
    write_all(command.fd, text.empty() || text.back() == '\n' ? text : text + "\n");
    close(command.fd);
    command.fd = -1;
}

void publish(const Status & s) {
    std::lock_guard<std::mutex> lock(status_mutex);
    status = s;
    status_published = true;
}

Server::~Server() { stop(); }

bool Server::start() {
    struct stat info {};
    if (lstat(SOCKET_NAME, &info) == 0) {
        if (!S_ISSOCK(info.st_mode)) {
            logging::warning(std::string(SOCKET_NAME) + " exists and is not a socket; no control socket for this run");
            return false;
        }
        const int fd = connect_to(SOCKET_NAME);
        if (fd >= 0) {
            close(fd);
            logging::warning(std::string("another run answers on ") + SOCKET_NAME +
                             " in this directory; no control socket for this run");
            return false;
        }
        unlink(SOCKET_NAME);  // stale: left by a run that ended without removing it
    }
    const sockaddr_un address = socket_address(SOCKET_NAME);
    listen_fd = socket(AF_UNIX, SOCK_STREAM, 0);
    if (listen_fd < 0) {
        logging::warning(std::string("no control socket: ") + std::strerror(errno));
        return false;
    }
    set_cloexec(listen_fd);
    const mode_t mask = umask(0077);
    const int bound = bind(listen_fd, reinterpret_cast<const sockaddr *>(&address), sizeof(address));
    umask(mask);
    if (bound != 0 || chmod(SOCKET_NAME, 0600) != 0 || listen(listen_fd, 8) != 0 || pipe(wake) != 0) {
        logging::warning(std::string("no control socket: ") + std::strerror(errno));
        close(listen_fd);
        listen_fd = -1;
        if (bound == 0) unlink(SOCKET_NAME);
        return false;
    }
    set_cloexec(wake[0]);
    set_cloexec(wake[1]);
    // Signals go to the other threads, so that they never interrupt this one
    sigset_t all, previous;
    sigfillset(&all);
    pthread_sigmask(SIG_SETMASK, &all, &previous);
    thread = std::thread(&Server::serve, this);
    pthread_sigmask(SIG_SETMASK, &previous, nullptr);
    return true;
}

void Server::stop() {
    if (listen_fd < 0) return;
    const char byte = 0;
    if (write(wake[1], &byte, 1) < 0) {
        // The thread polls the pipe; a full pipe already wakes it
    }
    thread.join();
    close(listen_fd);
    close(wake[0]);
    close(wake[1]);
    listen_fd = -1;
    unlink(SOCKET_NAME);
    for (auto & command : take_commands()) reply(command, "error: the run ended before acting on the command");
}

void Server::serve() {
    while (true) {
        pollfd fds[2] = {{listen_fd, POLLIN, 0}, {wake[0], POLLIN, 0}};
        if (poll(fds, 2, -1) < 0) {
            if (errno == EINTR) continue;
            return;
        }
        if (fds[1].revents) return;
        if (!(fds[0].revents & POLLIN)) continue;
        const int fd = accept(listen_fd, nullptr, nullptr);
        if (fd < 0) continue;
        set_cloexec(fd);
        if (!same_user(fd)) {
            close(fd);
            continue;
        }
        timeval timeout {5, 0};
        setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &timeout, sizeof(timeout));
        std::string line;
        char buffer[512];
        while (line.find('\n') == std::string::npos && line.size() < 4096) {
            const ssize_t n = recv(fd, buffer, sizeof(buffer), 0);
            if (n < 0 && errno == EINTR) continue;
            if (n <= 0) break;
            line.append(buffer, static_cast<size_t>(n));
        }
        line = trim(line.substr(0, line.find('\n')));
        Command command{line, fd};
        if (line == "status") {
            reply(command, status_text());
        } else if (line == "checkpoint" || line == "stop" || line.rfind("set ", 0) == 0) {
            std::lock_guard<std::mutex> lock(queue_mutex);
            queue.push_back(command);
            requests.fetch_or(COMMAND);
        } else {
            reply(command, "error: unknown command '" + line + "' (status, checkpoint, stop, set key=value)");
        }
    }
}

std::string send(const std::string & run_dir, const std::string & command) {
    const std::string path = run_dir.empty() || run_dir == "." ? SOCKET_NAME : run_dir + "/" + SOCKET_NAME;
    const int fd = connect_to(path);
    if (fd < 0) {
        if (errno == ENOENT) throw std::runtime_error("no control socket " + path + " (is a run active there?)");
        if (errno == ECONNREFUSED) throw std::runtime_error(path + " is stale: no run answers on it");
        throw std::runtime_error("cannot connect to " + path + ": " + std::strerror(errno));
    }
    if (!write_all(fd, command + "\n")) {
        close(fd);
        throw std::runtime_error("cannot send to " + path);
    }
    shutdown(fd, SHUT_WR);
    std::string text;
    char buffer[4096];
    while (true) {
        const ssize_t n = recv(fd, buffer, sizeof(buffer), 0);
        if (n < 0 && errno == EINTR) continue;
        if (n <= 0) break;
        text.append(buffer, static_cast<size_t>(n));
    }
    close(fd);
    if (text.empty()) throw std::runtime_error("the run closed the connection without replying");
    return text;
}

int ctl(const std::string & run_dir, const std::string & command) {
    if (chdir(run_dir.c_str()) != 0) {
        std::cerr << "mallard ctl: cannot enter " << run_dir << ": " << std::strerror(errno) << "\n";
        return 1;
    }
    try {
        const std::string text = send(".", command);
        std::cout << text << std::flush;
        return text.rfind("error", 0) == 0 ? 1 : 0;
    } catch (const std::exception & e) {
        std::cerr << "mallard ctl: " << e.what() << "\n";
        return 1;
    }
}

} // namespace run_control

/**
 * @file run_control_test.cpp
 * @author Matthew Bonanni (mbonanni001@gmail.com)
 * @brief Run control: requests acted on at step boundaries on every rank,
 *        runtime settings, divergence, the control socket, and the solver
 *        driven by signals and `Mallard ctl` as a separate process.
 * @version 0.1
 * @date 2026-10-15
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 *
 */

#include <gtest/gtest.h>
#include <Kokkos_Core.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <csignal>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iterator>
#include <regex>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include <fcntl.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/un.h>
#include <sys/wait.h>
#include <unistd.h>

#include "comm.h"
#include "run_control.h"
#include "solver.h"
#include "test_fixtures.h"

extern char ** environ;

namespace {

namespace fs = std::filesystem;

std::string case_input(const std::string & dir, const std::string & run, uint64_t restart_interval) {
    std::ostringstream s;
    s << "[run]\n" << run
      << "[mesh]\ntype = \"cartesian_tri\"\nNx = 12\nNy = 10\nLx = 1.0\nLy = 1.0\n"
      << "[initialize]\ntype = \"analytical\"\n"
      << "rho = \"1.0 + (x < 0.4 ? 1.0 : 0.0)\"\nu = [\"0.2\", \"0.1\"]\np = \"x < 0.4 ? 2.0 : 1.0\"\n"
      << "[[boundaries]]\nname = \"left\"\ntype = \"extrapolation\"\n"
      << "[[boundaries]]\nname = \"right\"\ntype = \"extrapolation\"\n"
      << "[[boundaries]]\nname = \"top\"\ntype = \"symmetry\"\n"
      << "[[boundaries]]\nname = \"bottom\"\ntype = \"wall_adiabatic\"\n"
      << "[numerics]\nriemann_solver = \"HLLC\"\ntime_integrator = \"SSPRK3\"\n"
      << "[numerics.face_reconstruction]\ntype = \"MUSCL\"\n"
      << "[physics]\ntype = \"euler\"\ngamma = 1.4\np_ref = 1.0\nT_ref = 1.0\nrho_ref = 1.0\n"
      << "[output]\ncheck_interval = 1000000\n"
      << "[[write_data]]\nprefix = \"" << dir << "/restart\"\nformat = \"restart\"\ninterval = " << restart_interval << "\n";
    return s.str();
}

/**
 * @brief A directory of the test's scratch directory, shared by all ranks.
 *        Relative, so the control socket's path stays within its length limit.
 */
std::string fresh_dir(const std::string & name) {
    if (comm::is_root()) fs::create_directories(name);
    comm::barrier();
    return name;
}

std::string restart_file(const std::string & dir, uint64_t step) {
    std::ostringstream s;
    s << dir << "/restart_" << std::setw(6) << std::setfill('0') << step << ".restart";
    return s.str();
}

std::vector<rtype> owned_state(Solver & solver) {
    solver.copy_device_to_host();
    std::vector<rtype> U;
    for (uint32_t i = 0; i < solver.get_mesh()->n_owned(); i++) {
        for (uint8_t v = 0; v < N_CONSERVATIVE; v++) U.push_back(solver.h_conservatives(i, v));
    }
    return U;
}

/** @brief The state after n steps, straight through, and resumed from a restart file. */
void expect_restart_continues_exactly(const std::string & dir, const std::string & file, uint64_t n) {
    Solver straight;
    straight.init(parse_toml(case_input(dir + "/straight", "n_steps = " + std::to_string(n) + "\ncfl = 0.25\n", 1000)));
    straight.run();
    Solver resumed;
    std::string input = case_input(dir + "/resumed", "n_steps = " + std::to_string(n) + "\ncfl = 0.25\n", 1000);
    const size_t begin = input.find("type = \"analytical\""), end = input.find("[[boundaries]]");
    input.replace(begin, end - begin, "type = \"restart\"\nfile = \"" + file + "\"\n");
    resumed.init(parse_toml(input));
    resumed.run();
    EXPECT_EQ(resumed.get_step(), straight.get_step());
    EXPECT_EQ(resumed.get_time(), straight.get_time());
    EXPECT_EQ(owned_state(resumed), owned_state(straight));
}

void clear_requests() {
    run_control::clear(run_control::CHECKPOINT | run_control::STOP);
    run_control::take_commands();
}

} // namespace

TEST(RunControlTest, CheckpointRequestWritesARestartAfterTheNextStepAndTheRunContinues) {
    const std::string dir = fresh_dir("checkpoint");
    clear_requests();
    Solver solver;
    solver.init(parse_toml(case_input(dir, "n_steps = 6\ncfl = 0.25\n", 1000)));
    run_control::request(run_control::CHECKPOINT);
    EXPECT_EQ(solver.run(), 0);
    EXPECT_EQ(solver.get_step(), 6u);
    EXPECT_EQ(run_control::pending(), 0);
    ASSERT_TRUE(fs::exists(restart_file(dir, 1)));
    EXPECT_EQ(read_restart(restart_file(dir, 1)).step, 1u);
    expect_restart_continues_exactly(dir, restart_file(dir, 1), 6);
}

TEST(RunControlTest, StopRequestEndsTheRunAfterTheNextStepWithARestart) {
    const std::string dir = fresh_dir("stop");
    clear_requests();
    Solver solver;
    solver.init(parse_toml(case_input(dir, "n_steps = 50\ncfl = 0.25\n", 1000)));
    run_control::request(run_control::STOP);
    EXPECT_EQ(solver.run(), EXIT_STOPPED);
    EXPECT_EQ(solver.get_step(), 1u);
    EXPECT_EQ(run_control::pending(), 0);
    ASSERT_TRUE(fs::exists(restart_file(dir, 1)));
    expect_restart_continues_exactly(dir, restart_file(dir, 1), 5);
}

TEST(RunControlTest, CommandsChangeRuntimeSettingsAtTheNextStep) {
    const std::string dir = fresh_dir("set");
    clear_requests();
    Solver solver;
    solver.init(parse_toml(case_input(dir, "n_steps = 50\ncfl = 0.25\n", 1000)));
    run_control::submit("set n_steps=7 write_data.0.interval=3 cfl=0.5");
    EXPECT_EQ(solver.run(), 0);
    EXPECT_EQ(solver.get_step(), 7u);
    for (uint64_t step : {3u, 6u}) EXPECT_TRUE(fs::exists(restart_file(dir, step))) << step;
    EXPECT_FALSE(fs::exists(restart_file(dir, 4)));
}

TEST(RunControlTest, SettingsOutsideTheRuntimeSafeSetAreRejected) {
    const std::string dir = fresh_dir("reject");
    Solver solver;
    solver.init(parse_toml(case_input(dir, "n_steps = 5\ncfl = 0.25\n", 1000)));
    const auto [cfl_ok, cfl_why] = solver.apply_setting("cfl", "0.5");
    EXPECT_FALSE(cfl_ok);
    EXPECT_NE(cfl_why.find("cannot be changed"), std::string::npos) << cfl_why;
    EXPECT_FALSE(solver.apply_setting("n_steps", "-3").first);
    EXPECT_FALSE(solver.apply_setting("write_data.flow.interval", "3").first);
    EXPECT_FALSE(solver.apply_setting("probes.p.interval", "3").first);
    EXPECT_FALSE(solver.apply_setting("integrals.interval", "3").first);
    // The only stop condition cannot be removed
    EXPECT_FALSE(solver.apply_setting("n_steps", "0").first);
    EXPECT_FALSE(solver.apply_setting("t_stop", "10mm").first);
    EXPECT_TRUE(solver.apply_setting("t_stop", "10ms").first);
    EXPECT_TRUE(solver.apply_setting("n_steps", "0").first);
    solver.run();
    EXPECT_NEAR(double(solver.get_time()), 0.01, 1e-6);
}

TEST(RunControlTest, ProbeIntervalsChangeOnlyInTheirOwnKind) {
    const std::string dir = fresh_dir("probe_set");
    clear_requests();
    Solver solver;
    solver.init(parse_toml(case_input(dir, "n_steps = 5\ncfl = 0.25\n", 1000) +
                           "[[probes]]\nname = \"mid\"\nvariables = [\"P\"]\npoint = [0.5, 0.5]\n"
                           "time_interval = 0.1\nfile = \"" + dir + "/mid.csv\"\n"));
    EXPECT_FALSE(solver.apply_setting("probes.mid.interval", "3").first);
    EXPECT_FALSE(solver.apply_setting("probes.other.time_interval", "0.05").first);
    EXPECT_TRUE(solver.apply_setting("probes.mid.time_interval", "50ms").first);
}

TEST(RunControlTest, DivergenceReportsTheCellAndWritesTheLastValidState) {
    const std::string dir = fresh_dir("diverge");
    Solver solver;
    solver.init(parse_toml(case_input(dir, "n_steps = 500\ndt = 0.04\n", 1000)));
    std::string message;
    try {
        solver.run();
    } catch (const std::runtime_error & e) {
        message = e.what();
    }
    std::smatch m;
    ASSERT_TRUE(std::regex_search(message, m, std::regex("diverged at step ([0-9]+).*: ([0-9,]+) cells? ha")))
        << message;
    const uint64_t bad = std::stoull(m[1]);
    EXPECT_GT(bad, 1u);
    EXPECT_TRUE(std::regex_search(message, std::regex("(RHO|RHOU_X|RHOU_Y|RHOE|P) = \\S+ in cell [0-9]+ at \\(")))
        << message;
    const std::string last = restart_file(dir, bad - 1);
    EXPECT_NE(message.find("last valid state (step " + std::to_string(bad - 1)), std::string::npos) << message;
    ASSERT_TRUE(fs::exists(last)) << message;
    const RestartData restart = read_restart(last);
    EXPECT_EQ(restart.step, bad - 1);
    for (const auto & field : restart.fields) {
        for (rtype v : field) ASSERT_TRUE(std::isfinite(double(v)));
    }
}

TEST(RunControlTest, ControlSocketIsPrivateAnswersStatusAndQueuesCommands) {
    if (!comm::is_root()) GTEST_SKIP() << "the socket is rank 0's";
    const std::string dir = ".";
    clear_requests();
    {
        // A socket left by a run that died is replaced
        const int stale = socket(AF_UNIX, SOCK_STREAM, 0);
        sockaddr_un address {};
        address.sun_family = AF_UNIX;
        std::strcpy(address.sun_path, run_control::SOCKET_NAME);
        ASSERT_EQ(bind(stale, reinterpret_cast<const sockaddr *>(&address), sizeof(address)), 0);
        close(stale);
    }
    {
        run_control::Server server;
        ASSERT_TRUE(server.start());
        struct stat info {};
        ASSERT_EQ(stat(run_control::SOCKET_NAME, &info), 0);
        EXPECT_EQ(info.st_mode & 0777, 0600u);
        run_control::Server second;
        EXPECT_FALSE(second.start());

        EXPECT_EQ(run_control::send(dir, "status"), "state: starting\n");
        run_control::Status status;
        status.run.step = 42;
        status.monitors = {{"min_p", "0.5"}};
        run_control::publish(status);
        const std::string text = run_control::send(dir, "status");
        EXPECT_NE(text.find("step: 42\n"), std::string::npos) << text;
        EXPECT_NE(text.find("min_p: 0.5\n"), std::string::npos) << text;
        EXPECT_EQ(run_control::send(dir, "frobnicate").rfind("error", 0), 0u);

        std::string answer;
        std::thread client([&] { answer = run_control::send(dir, "checkpoint"); });
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
        while (!(run_control::pending() & run_control::COMMAND) && std::chrono::steady_clock::now() < deadline) {
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
        }
        auto commands = run_control::take_commands();
        ASSERT_EQ(commands.size(), 1u);
        EXPECT_EQ(commands[0].text, "checkpoint");
        run_control::reply(commands[0], "done");
        client.join();
        EXPECT_EQ(answer, "done\n");
    }
    EXPECT_FALSE(fs::exists(run_control::SOCKET_NAME));
}

namespace {

/** @brief Runs body in a forked child and returns its exit status, or 128 + the signal that ended it. */
int in_child(const std::function<int()> & body) {
    const pid_t pid = fork();
    if (pid == 0) _exit(body());
    int status = 0;
    waitpid(pid, &status, 0);
    return WIFEXITED(status) ? WEXITSTATUS(status) : 128 + WTERMSIG(status);
}

} // namespace

TEST(RunControlTest, StopSignalsAreDeferredWhileRunningAndASecondInterruptEndsAtOnce) {
    if (comm::size() > 1) GTEST_SKIP() << "forks";
    // Before the time loop, there is nothing to save
    EXPECT_EQ(in_child([] {
        run_control::install_signal_handlers();
        raise(SIGTERM);
        return 0;
    }), 128 + SIGTERM);
    EXPECT_EQ(in_child([] {
        run_control::install_signal_handlers();
        run_control::clear(run_control::STOP | run_control::CHECKPOINT);
        run_control::set_accepting(true);
        raise(SIGTERM);
        raise(SIGUSR1);
        const bool deferred = run_control::pending() == (run_control::STOP | run_control::CHECKPOINT) &&
                              run_control::stop_signal() == SIGTERM;
        return deferred ? 0 : 1;
    }), 0);
    // A library that takes a signal over during setup loses it when the loop starts
    EXPECT_EQ(in_child([] {
        run_control::install_signal_handlers();
        run_control::clear(run_control::STOP | run_control::CHECKPOINT);
        signal(SIGUSR1, SIG_DFL);
        run_control::set_accepting(true);
        raise(SIGUSR1);
        return run_control::pending() == run_control::CHECKPOINT ? 0 : 1;
    }), 0);
    EXPECT_EQ(in_child([] {
        run_control::install_signal_handlers();
        run_control::set_accepting(true);
        raise(SIGINT);
        if (!(run_control::pending() & run_control::STOP)) return 1;
        raise(SIGINT);
        return 2;
    }), 128 + SIGINT);
}

TEST(MPITest, RequestsOfAnyRankStopEveryRankAfterTheSameStep) {
    const std::string dir = fresh_dir("mpi_stop");
    clear_requests();
    Solver solver;
    solver.init(parse_toml(case_input(dir, "n_steps = 50\ncfl = 0.25\n", 1000)));
    if (comm::rank() == comm::size() - 1) run_control::request(run_control::STOP | run_control::CHECKPOINT);
    EXPECT_EQ(solver.run(), EXIT_STOPPED);
    EXPECT_EQ(comm::allreduce(solver.get_step(), comm::Op::MAX), 1u);
    EXPECT_EQ(comm::allreduce(solver.get_step(), comm::Op::MIN), 1u);
    EXPECT_EQ(run_control::pending(), 0);
    EXPECT_EQ(read_restart(restart_file(dir, 1)).step, 1u);
}

TEST(MPITest, CommandsOfRankZeroReachEveryRank) {
    const std::string dir = fresh_dir("mpi_set");
    clear_requests();
    Solver solver;
    solver.init(parse_toml(case_input(dir, "n_steps = 50\ncfl = 0.25\n", 1000)));
    if (comm::is_root()) run_control::submit("set n_steps=4 write_data.0.interval=2");
    EXPECT_EQ(solver.run(), 0);
    EXPECT_EQ(comm::allreduce(solver.get_step(), comm::Op::MAX), 4u);
    EXPECT_EQ(comm::allreduce(solver.get_step(), comm::Op::MIN), 4u);
    EXPECT_TRUE(fs::exists(restart_file(dir, 2)));
}

TEST(MPITest, DivergenceIsReportedAlikeOnEveryRank) {
    const std::string dir = fresh_dir("mpi_diverge");
    Solver solver;
    solver.init(parse_toml(case_input(dir, "n_steps = 500\ndt = 0.04\n", 1000)));
    std::string message;
    try {
        solver.run();
    } catch (const std::runtime_error & e) {
        message = e.what();
    }
    ASSERT_NE(message.find("diverged at step"), std::string::npos) << message;
    std::string root = message;
    root = comm::broadcast(root);
    EXPECT_EQ(message, root);
}

namespace {

#ifdef MALLARD_EXECUTABLE

/** @brief Environment without the variables of the MPI launcher this test may run under. */
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

/** @brief Starts the command in dir with its output in dir/run.log. */
pid_t spawn(const std::vector<std::string> & args, const std::string & dir) {
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
        const int fd = open(log.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
        if (fd < 0 || chdir(dir.c_str()) != 0) _exit(126);
        dup2(fd, 1);
        dup2(fd, 2);
        execve(argv[0], argv.data(), envp.data());
        _exit(127);
    }
    return pid;
}

std::string read_text(const std::string & path) {
    std::ifstream in(path);
    return std::string(std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>());
}

/** @brief Waits for the process to end: its exit status, 128 + signal, or -1 (killed after the timeout). */
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
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    return true;
}

std::string status_of(const std::string & dir) {
    try {
        return run_control::send(dir, "status");
    } catch (const std::exception &) {
        return "";
    }
}

uint64_t status_step(const std::string & text) {
    std::smatch m;
    return std::regex_search(text, m, std::regex("step: ([0-9]+)")) ? std::stoull(m[1]) : 0;
}

std::vector<std::string> restart_files(const std::string & dir) {
    std::vector<std::string> files;
    for (const auto & entry : fs::directory_iterator(dir)) {
        if (entry.path().extension() == ".restart") files.push_back(entry.path().string());
    }
    std::sort(files.begin(), files.end());
    return files;
}

/** @brief Starts a long run in a fresh directory and waits until it answers on its socket. */
pid_t start_run(const std::string & dir, [[maybe_unused]] int ranks) {
    std::ofstream(dir + "/input.toml") << case_input(".", "n_steps = 100000000\nt_wall_stop = 300\ncfl = 0.25\n",
                                                    100000000);
    std::vector<std::string> args;
#ifdef MALLARD_MPIEXEC
    // An MPI library need not start a process without its launcher (a singleton)
    args = {MALLARD_MPIEXEC, "-n", std::to_string(ranks)};
#ifdef MALLARD_MPIEXEC_OPEN_MPI
    args.push_back("--oversubscribe");
#endif
#endif
    args.insert(args.end(), {MALLARD_EXECUTABLE, "-i", "input.toml"});
    const pid_t pid = spawn(args, dir);
    int status = 0;
    pid_t ended = 0;
    const bool up = wait_until([&] {
        ended = waitpid(pid, &status, WNOHANG);
        return ended != 0 || status_step(status_of(dir)) > 2;
    });
    EXPECT_TRUE(up && ended == 0) << (ended == 0 ? "no status after 120 s" : "ended with status " + std::to_string(status))
                                  << "\n" << read_text(dir + "/run.log");
    return pid;
}

void signals_checkpoint_and_stop(const std::string & name, int ranks, int stop_signal) {
    const std::string dir = fresh_dir(name);
    const pid_t pid = start_run(dir, ranks);
    const size_t before = restart_files(dir).size();  // the one of step 0
    ASSERT_EQ(kill(pid, SIGUSR1), 0);
    ASSERT_TRUE(wait_until([&] { return restart_files(dir).size() > before; })) << read_text(dir + "/run.log");
    const std::string checkpoint = restart_files(dir).back();
    const uint64_t checkpoint_step = read_restart(checkpoint).step;
    EXPECT_GT(status_step(status_of(dir)), 0u);
    ASSERT_EQ(kill(pid, stop_signal), 0);
    EXPECT_EQ(wait_exit(pid, 120.0), EXIT_STOPPED) << read_text(dir + "/run.log");
    const std::string log = read_text(dir + "/run.log");
    EXPECT_NE(log.find(run_control::signal_name(stop_signal) + " received"), std::string::npos) << log;
    const std::string last = restart_files(dir).back();
    const RestartData restart = read_restart(last);
    EXPECT_GT(restart.step, checkpoint_step);
    EXPECT_EQ(restart.n_cells, 240u);
    for (const auto & field : restart.fields) {
        for (rtype v : field) ASSERT_TRUE(std::isfinite(double(v)));
    }
    EXPECT_FALSE(fs::exists(dir + "/" + run_control::SOCKET_NAME));
}

void ctl_set_checkpoint_and_stop(const std::string & name, int ranks) {
    const std::string dir = fresh_dir(name);
    const pid_t pid = start_run(dir, ranks);
    const std::string status = status_of(dir);
    for (const char * key : {"step:", "t:", "dt:", "cfl: 0.25", "eta:", "memory_host:", "min_p:"}) {
        EXPECT_NE(status.find(key), std::string::npos) << key << "\n" << status;
    }
    EXPECT_EQ(run_control::send(dir, "set cfl=0.5").rfind("error", 0), 0u);
    EXPECT_EQ(run_control::send(dir, "set check_interval=1000"), "check_interval = 1000 (was 1000000)\n");
    const std::string checkpoint = run_control::send(dir, "checkpoint");
    std::smatch m;
    ASSERT_TRUE(std::regex_search(checkpoint, m, std::regex("checkpoint at step ([0-9]+), t = \\S+: (\\S+)")))
        << checkpoint;
    EXPECT_EQ(read_restart(dir + "/" + m[2].str()).step, std::stoull(m[1]));
    const std::string stopped = run_control::send(dir, "stop");
    ASSERT_TRUE(std::regex_search(stopped, m, std::regex("stopped at step ([0-9]+), t = \\S+; restart: (\\S+)")))
        << stopped;
    EXPECT_EQ(read_restart(dir + "/" + m[2].str()).step, std::stoull(m[1]));
    EXPECT_EQ(wait_exit(pid, 120.0), EXIT_STOPPED) << read_text(dir + "/run.log");
    EXPECT_FALSE(fs::exists(dir + "/" + run_control::SOCKET_NAME));
}

#endif

} // namespace

TEST(RunControlProcessTest, SignalsCheckpointAndStopARun) {
#ifndef MALLARD_EXECUTABLE
    GTEST_SKIP() << "no solver executable";
#else
    if (comm::size() > 1) GTEST_SKIP() << "starts its own run";
#ifdef MALLARD_MPIEXEC
    // Launched by mpiexec, as an MPI build is; Open MPI ends the job on SIGTERM
    signals_checkpoint_and_stop("process_signals", 1, SIGUSR2);
#else
    signals_checkpoint_and_stop("process_signals", 1, SIGTERM);
#endif
#endif
}

TEST(RunControlProcessTest, CtlSetsCheckpointsAndStopsARun) {
#ifndef MALLARD_EXECUTABLE
    GTEST_SKIP() << "no solver executable";
#else
    if (comm::size() > 1) GTEST_SKIP() << "starts its own run";
    ctl_set_checkpoint_and_stop("process_ctl", 1);
#endif
}

TEST(RunControlProcessTest, SignalsForwardedByMpiexecCheckpointAndStopEveryRank) {
#if !defined(MALLARD_EXECUTABLE) || !defined(MALLARD_MPIEXEC)
    GTEST_SKIP() << "no MPI launcher";
#else
    if (comm::size() > 1) GTEST_SKIP() << "starts its own run";
    // mpiexec forwards SIGUSR1 and SIGUSR2; Open MPI ends the job on SIGTERM
    signals_checkpoint_and_stop("process_mpi_signals", 2, SIGUSR2);
#endif
}

TEST(RunControlProcessTest, CtlControlsAnMpiRun) {
#if !defined(MALLARD_EXECUTABLE) || !defined(MALLARD_MPIEXEC)
    GTEST_SKIP() << "no MPI launcher";
#else
    if (comm::size() > 1) GTEST_SKIP() << "starts its own run";
    ctl_set_checkpoint_and_stop("process_mpi_ctl", 2);
#endif
}

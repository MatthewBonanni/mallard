/**
 * @file main.cpp
 * @brief Main file for Mallard.
 * @version 0.1
 * @date 2023-12-17
 *
 * @copyright Copyright (c) 2023 Matthew Bonanni
 */

#include <cerrno>
#include <chrono>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <thread>
#include <vector>

#include <unistd.h>

#include <Kokkos_Core.hpp>

#include "build_info.h"
#include "case_template.h"
#include "commands.h"
#include "input_upgrade.h"
#include "input.h"
#include "log.h"
#include "parallel/binding.h"
#include "parallel/comm.h"
#include "run_control.h"
#include "solver/solver.h"

namespace {

void usage() {
    std::cerr << "Usage: mallard -i input.toml [--restart FILE] [Kokkos options, e.g. --kokkos-num-threads=8]\n"
              << "       mallard check input.toml [--ranks N] [--throughput CELLS_PER_S | --steps N]"
              << " [--quality-vtu PREFIX] [--catalog KEYS.json]\n"
              << "       mallard new <template> [directory]   (mallard new --list)\n"
              << "       mallard ctl <run-dir> status | checkpoint | stop | set <key>=<value> ...\n"
              << "       mallard upgrade-input old.toml > new.toml\n"
              << "       mallard reactor -i input.toml\n"
              << "       mallard mesh convert INPUT OUTPUT.h5\n"
              << "       mallard test [GoogleTest options, e.g. --gtest_filter='Sod*']\n"
              << "       mallard --version\n";
}

/**
 * @brief `mallard test`: runs mallard-test from the same build or install tree.
 */
int test_main(int argc, char ** argv) {
    const std::string name = std::string("mallard-test") + MALLARD_EXECUTABLE_SUFFIX;
    const std::filesystem::path bin = executable_path().parent_path();
    const std::vector<std::filesystem::path> candidates = {bin / name, bin.parent_path() / "test" / name};
    for (const auto & path : candidates) {
        if (!std::filesystem::exists(path)) continue;
        std::string program = path.string();
        std::vector<char *> args = {program.data()};
        for (int i = 0; i < argc; ++i) args.push_back(argv[i]);
        args.push_back(nullptr);
        execv(program.c_str(), args.data());
        std::cerr << "mallard test: cannot run " << program << ": " << std::strerror(errno) << "\n";
        return 1;
    }
    std::cerr << "mallard test: no " << name << " in";
    for (const auto & path : candidates) std::cerr << "\n  " << path.parent_path().string();
    std::cerr << "\nBuild it with -DMallard_ENABLE_TESTS=ON and install it with -DMallard_INSTALL_TESTS=ON"
                 " (both the default).\n";
    return 1;
}

/**
 * @brief Commands given as the first argument, each with the arguments after it.
 */
struct Subcommand {
    const char * name;
    int (*run)(int argc, char ** argv);
};

constexpr Subcommand subcommands[] = {
    {"new", case_template::new_main},
    {"ctl", run_control::ctl_main},
    {"upgrade-input", input_upgrade::main},
    {"reactor", reactor_main},
    {"mesh", mesh_main},
    {"test", test_main},
};

/**
 * @brief Exception text without toml11's "[error] " prefix and trailing newlines.
 */
std::string error_text(const std::exception & e) {
    std::string text = e.what();
    if (text.rfind("[error] ", 0) == 0) text.erase(0, 8);
    while (!text.empty() && text.back() == '\n') text.pop_back();
    return text;
}

/**
 * @brief Reports a fatal error and ends all ranks of a parallel run.
 */
[[noreturn]] void abort_parallel(const std::string & message) {
    // Most errors (bad input) hit every rank; rank 0 reports and aborts first,
    // so the other ranks wait briefly rather than print the same message again
    if (!comm::is_root()) std::this_thread::sleep_for(std::chrono::seconds(2));
    logging::error(message);
    comm::abort(1);
}

} // namespace

/**
 * @brief Entry point of the program.
 * @param argc Number of command-line arguments.
 * @param argv Array of command-line arguments.
 * @return Exit status.
 */
int main(int argc, char* argv[]) {
    if (argc > 1) {
        for (const auto & command : subcommands) {
            if (strcmp(argv[1], command.name) == 0) return command.run(argc - 2, argv + 2);
        }
    }
    std::string input_file, restart_file;
    const bool check = argc > 1 && strcmp(argv[1], "check") == 0;
    CheckOptions options;
    try {
        for (int i = check ? 2 : 1; i < argc; ++i) {
            const bool has_value = i + 1 < argc;
            if (strcmp(argv[i], "-i") == 0 && has_value) {
                input_file = argv[++i];
            } else if (strcmp(argv[i], "--restart") == 0 && has_value) {
                restart_file = argv[++i];
            } else if (check && input_file.empty() && argv[i][0] != '-') {
                input_file = argv[i];
            } else if (strcmp(argv[i], "--ranks") == 0 && has_value) {
                options.ranks = std::stoi(argv[++i]);
            } else if (strcmp(argv[i], "--throughput") == 0 && has_value) {
                options.throughput = std::stod(argv[++i]);
            } else if (strcmp(argv[i], "--steps") == 0 && has_value) {
                options.steps = std::stoull(argv[++i]);
            } else if (strcmp(argv[i], "--quality-vtu") == 0 && has_value) {
                options.quality_file = argv[++i];
            } else if (strcmp(argv[i], "--catalog") == 0 && has_value) {
                options.catalog_file = argv[++i];
            } else if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
                usage();
                return 0;
            } else if (strcmp(argv[i], "--version") == 0) {
                std::cout << "Mallard " << mallard_version() << std::endl;
                return 0;
            }
        }
    } catch (const std::exception &) {
        usage();
        return 1;
    }
    if (input_file.empty() || options.ranks < 0 || options.throughput < 0.0) {
        usage();
        return 1;
    }

    // A checkpoint requested while MPI and the device start up is kept for the first step
    run_control::install_signal_handlers();
    // MPI before Kokkos, so Kokkos can map each rank to its own device
    comm::Session session(argc, argv);
    if (!comm::is_root()) std::cout.rdbuf(nullptr);

    binding::select_device(argc, argv);
    Kokkos::initialize(argc, argv);
    int status = 0;
    {
        std::string error;
        try {
            print_header(input_file);
            Solver solver;
            if (check) {
                Input input = parse_input(input_file);
                if (!restart_file.empty()) Solver::start_from_restart(input, restart_file);
                const CheckReport report = solver.check(input, options);
                print_check(report);
                status = report.ok() ? 0 : 1;
            } else {
                binding::report();
                solver.enable_control(true);
                solver.init(input_file, restart_file);
                status = solver.run();
            }
        } catch (const InputError & e) {
            error = input_file + ": " + error_text(e);
        } catch (const std::exception & e) {
            error = error_text(e);
        }
        if (!error.empty()) {
            if (comm::size() > 1) abort_parallel(error);
            logging::error(error);
            status = 1;
        }
    }
    Kokkos::finalize();
    return status;
}

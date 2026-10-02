/**
 * @file main.cpp
 * @brief Main file for Mallard.
 * @version 0.1
 * @date 2023-12-17
 *
 * @copyright Copyright (c) 2023 Matthew Bonanni
 */

#include <chrono>
#include <cstring>
#include <iostream>
#include <thread>

#include <Kokkos_Core.hpp>

#include "build_info.h"
#include "input.h"
#include "log.h"
#include "parallel/comm.h"
#include "solver/solver.h"

namespace {

void usage(const char * program) {
    std::cerr << "Usage: " << program << " -i input.toml [Kokkos options, e.g. --kokkos-num-threads=8]\n"
              << "       " << program << " --version\n";
}

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
    std::string input_file;
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-i") == 0 && i + 1 < argc) {
            input_file = argv[++i];
        } else if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
            usage(argv[0]);
            return 0;
        } else if (strcmp(argv[i], "--version") == 0) {
            std::cout << "Mallard " << mallard_version() << std::endl;
            return 0;
        }
    }
    if (input_file.empty()) {
        usage(argv[0]);
        return 1;
    }

    // MPI before Kokkos, so Kokkos can map each rank to its own device
    comm::Session session(argc, argv);
    if (!comm::is_root()) std::cout.rdbuf(nullptr);

    Kokkos::initialize(argc, argv);
    int status = 0;
    {
        std::string error;
        try {
            print_header(input_file);
            Solver solver;
            solver.init(input_file);
            solver.run();
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

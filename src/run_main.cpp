/**
 * @file run_main.cpp
 * @brief `mallard -i input.toml` and `mallard check`: set up a case, then run or check it.
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 */

#include <chrono>
#include <exception>
#include <iostream>
#include <string>
#include <thread>
#include <type_traits>

#include <Kokkos_Core.hpp>

#include "build_info.h"
#include "cli.h"
#include "commands.h"
#include "input.h"
#include "log.h"
#include "parallel/binding.h"
#include "parallel/comm.h"
#include "run_control.h"
#include "solver/solver.h"

namespace {

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

int solve(const cli::Args & args, const std::string & input_file, const CheckOptions * check) {
    const std::string restart_file = args.value_or("--restart", "");
    int argc = args.argc;
    char ** argv = args.argv;
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
                const CheckReport report = solver.check(input, *check);
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

/** @brief The option's value as a number of type T, or a usage error. */
template <typename T>
bool number(const cli::Args & args, std::string_view name, T & value, std::string & error) {
    if (!args.has(name)) return true;
    const std::string text = args.value_or(name, "");
    try {
        size_t used = 0;
        double x = std::stod(text, &used);
        if (used == text.size() && x >= 0.0) {
            value = static_cast<T>(x);
            if (static_cast<double>(value) == x) return true;
        }
    } catch (const std::exception &) {
    }
    error = std::string(name) + " takes a non-negative " + (std::is_integral_v<T> ? "integer" : "number") +
            ", not " + text;
    return false;
}

} // namespace

int run_main(const cli::Args & args) {
    if (!args.has("-i")) return cli::usage_error(*cli::command(""), "missing -i input.toml");
    return solve(args, args.value_or("-i", ""), nullptr);
}

int check_main(const cli::Args & args) {
    const cli::Command & command = *cli::command("check");
    if (args.has("-i") == !args.positional.empty()) {
        return cli::usage_error(command, args.positional.empty() ? "missing the input file" : "two input files");
    }
    const std::string input_file = args.has("-i") ? args.value_or("-i", "") : args.positional[0];
    CheckOptions options;
    options.quality_file = args.value_or("--quality", "");
    options.catalog_file = args.value_or("--catalog", "");
    std::string error;
    if (!number(args, "--ranks", options.ranks, error) || !number(args, "--throughput", options.throughput, error) ||
        !number(args, "--steps", options.steps, error)) {
        return cli::usage_error(command, error);
    }
    return solve(args, input_file, &options);
}

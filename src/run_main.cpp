/**
 * @file run_main.cpp
 * @brief `mallard -i input.toml` and `mallard check`: set up a case, then run or check it.
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 */

#include <chrono>
#include <exception>
#include <iostream>
#include <new>
#include <string>
#include <thread>
#include <type_traits>

#include <Kokkos_Core.hpp>

#include "build_info.h"
#include "cli.h"
#include "commands.h"
#include "diagnostic_code.h"
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
    if (dynamic_cast<const std::bad_alloc *>(&e) || text.find("failed to allocate memory") != std::string::npos) {
        text += "\nThe run is out of memory. mallard check input.toml --ranks N estimates the memory per rank: "
                "run on more ranks or GPUs, or on a coarser mesh. See docs/guide/input_files.md#checking-an-input.";
    }
    return text;
}

/**
 * @brief Reports a fatal error and ends all ranks of a parallel run.
 */
[[noreturn]] void abort_parallel(const std::string & code, const std::string & message) {
    // Most errors (bad input) hit every rank; rank 0 reports and aborts first,
    // so the other ranks wait briefly rather than print the same message again
    if (!comm::is_root()) std::this_thread::sleep_for(std::chrono::seconds(2));
    logging::error(code, message);
    comm::abort(1);
}

/** @brief With check: checks the case, and with resolve writes its resolved input (and nothing else) to stdout. */
int solve(const cli::Args & args, const std::string & input_file, const CheckOptions * check, bool resolve = false) {
    const std::string restart_file = args.value_or("--restart", "");
    logging::set_verbose(args.has("--verbose"));
    int argc = args.argc;
    char ** argv = args.argv;
    // A checkpoint requested while MPI and the device start up is kept for the first step
    run_control::install_signal_handlers();
    // MPI before Kokkos, so Kokkos can map each rank to its own device
    comm::Session session(argc, argv);
    std::streambuf * const out = std::cout.rdbuf();
    const bool json = check && args.value_or("--format", "text") == "json";
    if (!comm::is_root() || resolve || json) std::cout.rdbuf(nullptr);

    binding::select_device(argc, argv);
    Kokkos::initialize(argc, argv);
    int status = 0;
    {
        std::string error;
        std::string code = diagnostic_code::SETUP;
        try {
            print_header(input_file);
            Solver solver;
            if (check) {
                CheckReport report;
                try {
                    if (!input_file.empty()) {
                        Input input = parse_input(input_file);
                        if (!restart_file.empty()) Solver::start_from_restart(input, restart_file);
                        report = solver.check(input, *check);
                    }
                } catch (const InputError & e) {
                    if (resolve) throw;
                    report.input_file = input_file;
                    report.error(diagnostic_code::INPUT, input_file + ": " + error_text(e));
                }
                if (!resolve) check_environment(report);
                if (resolve && report.ok()) {
                    if (comm::is_root()) std::cout.rdbuf(out);
                    std::cout << report.resolved << std::flush;
                } else if (resolve) {
                    for (const CheckDiagnostic & d : report.diagnostics) {
                        if (d.severity != report::Status::ERROR || d.code == diagnostic_code::KEY_UNUSED) continue;
                        if (error.empty()) code = d.code;
                        error += (error.empty() ? "" : "\n") + d.message;
                    }
                    if (error.empty()) code = diagnostic_code::KEY_UNUSED;
                    if (!report.unused.empty()) {
                        error += std::string(error.empty() ? "" : "\n") +
                                 "keys that are unknown or unused with these settings (mallard check lists them all):";
                        for (const Diagnostic & key : report.unused) error += "\n  " + describe(key);
                    }
                } else {
                    if (json && comm::is_root()) std::cout.rdbuf(out);
                    print_check(report, json);
                }
                status = report.ok() ? 0 : 1;
            } else {
                binding::report();
                solver.enable_control(true);
                solver.init(input_file, restart_file);
                code = diagnostic_code::RUN;
                status = solver.run();
            }
        } catch (const InputError & e) {
            code = e.code;
            error = input_file + ": " + error_text(e);
        } catch (const diagnostic_code::Error & e) {
            code = e.code;
            error = error_text(e);
        } catch (const std::exception & e) {
            error = error_text(e);
        }
        if (!error.empty()) {
            if (comm::size() > 1) abort_parallel(code, error);
            logging::error(code, error);
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

/** @brief The mesh-field options of mallard check into options; an error, or empty. */
std::string mesh_field_options(const cli::Args & args, CheckOptions & options) {
    if (args.has("--quality")) {
        if (comm::is_root()) std::cerr << "mallard check --quality is deprecated: use --mesh-fields\n";
        if (!args.has("--mesh-fields")) options.mesh_fields = args.value_or("--quality", "");
    }
    if (args.has("--mesh-fields")) options.mesh_fields = args.value_or("--mesh-fields", "");
    const int selections = int(args.has("--stencils")) + int(args.has("--cells")) + int(args.has("--region"));
    if (selections > 1) return "--stencils, --cells and --region each choose the stencils written: give one";
    if ((selections > 0 || args.has("--weights")) && options.mesh_fields.empty()) {
        return "--stencils, --cells, --region and --weights write with the mesh fields: give --mesh-fields PREFIX";
    }
    if (args.has("--weights") && selections == 0) return "--weights takes --stencils all, --cells or --region";
    if (args.has("--stencils")) {
        const std::string value = args.value_or("--stencils", "");
        if (value != "all") return "--stencils takes all, not " + value + " (--cells or --region choose some)";
        options.all_stencils = true;
    }
    if (args.has("--cells")) {
        const std::string text = args.value_or("--cells", "");
        size_t at = 0;
        while (at <= text.size()) {
            const size_t end = std::min(text.find(',', at), text.size());
            const std::string id = text.substr(at, end - at);
            if (id.empty() || id.find_first_not_of("0123456789") != std::string::npos) {
                return "--cells takes global cell ids separated by commas (12,40), not " + text;
            }
            options.stencil_cells.push_back(std::stoull(id));
            at = end + 1;
        }
    }
    options.stencil_region = args.value_or("--region", "");
    options.stencil_weights = args.has("--weights");
    return "";
}

} // namespace

int run_main(const cli::Args & args) {
    if (!args.has("-i")) return cli::usage_error(*cli::command(""), "missing -i input.toml");
    return solve(args, args.value_or("-i", ""), nullptr);
}

int check_main(const cli::Args & args) {
    const cli::Command & command = *cli::command("check");
    if (args.has("-i") && !args.positional.empty()) return cli::usage_error(command, "two input files");
    const std::string input_file =
        args.has("-i") ? args.value_or("-i", "") : args.positional.empty() ? "" : args.positional[0];
    if (input_file.empty()) {
        for (const char * option : {"--restart", "--ranks", "--throughput", "--steps", "--memory", "--quality",
                                    "--catalog", "--view"}) {
            if (args.has(option)) return cli::usage_error(command, std::string(option) + " needs an input file");
        }
    }
    CheckOptions options;
    options.input_file = input_file;
    options.catalog_file = args.value_or("--catalog", "");
    options.view_dir = args.value_or("--view", "");
    if (const std::string f = args.value_or("--format", "text"); f != "text" && f != "json") {
        return cli::usage_error(command, "--format takes text or json, not " + f);
    }
    std::string error = mesh_field_options(args, options);
    if (error.empty() && args.has("--memory")) error = set_memory_target(args.value_or("--memory", ""), options);
    if (!error.empty()) return cli::usage_error(command, error);
    if (!number(args, "--ranks", options.ranks, error) || !number(args, "--throughput", options.throughput, error) ||
        !number(args, "--steps", options.steps, error)) {
        return cli::usage_error(command, error);
    }
    return solve(args, input_file, &options);
}

int resolve_main(const cli::Args & args) {
    CheckOptions options;
    options.input_file = args.positional[0];
    return solve(args, options.input_file, &options, true);
}

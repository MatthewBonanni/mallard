/**
 * @file restart_convert.cpp
 * @brief `mallard restart convert`: a restart file of one gas as a restart file of another case's gas
 *        (finite-rate chemistry of another mechanism, or a flamelet table), after the user accepts the mapping.
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 */

#include <cstdio>
#include <exception>
#include <filesystem>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unistd.h>

#include <Kokkos_Core.hpp>

#include "cli.h"
#include "commands.h"
#include "composition_map.h"
#include "diagnostic_code.h"
#include "input.h"
#include "json.h"
#include "log.h"
#include "parallel/binding.h"
#include "parallel/comm.h"
#include "solver/solver.h"

namespace {

/** @brief A usage error, exit status 2. */
struct Usage : std::runtime_error {
    using std::runtime_error::runtime_error;
};

std::vector<std::string> split(const std::string & text, const char by) {
    std::vector<std::string> parts;
    std::string part;
    std::istringstream in(text);
    while (std::getline(in, part, by)) {
        if (!part.empty()) parts.push_back(part);
    }
    return parts;
}

/** @brief "A=B,C=D" as a TOML inline table of names. */
std::string aliases(const std::string & text) {
    std::string table;
    for (const std::string & entry : split(text, ',')) {
        const size_t at = entry.find('=');
        if (at == std::string::npos || at == 0 || at + 1 == entry.size()) {
            throw Usage("--alias takes SOURCE=TARGET pairs separated by commas, not \"" + entry + "\"");
        }
        table += (table.empty() ? "" : ", ") + json::quote(entry.substr(0, at)) + " = " + json::quote(entry.substr(at + 1));
    }
    return "{" + table + "}";
}

/** @brief The flags' keys of [initialize.convert], added to the input's. */
void add_options(Input & input, const cli::Args & args) {
    std::string text;
    auto key = [&](const char * option, const char * name, const std::string & value) {
        if (args.has(option)) text += std::string(name) + " = " + value + "\n";
    };
    key("--mechanism", "mechanism", json::quote(args.value_or("--mechanism", "")));
    key("--table", "table", json::quote(args.value_or("--table", "")));
    if (args.has("--alias")) key("--alias", "aliases", aliases(args.value_or("--alias", "")));
    if (text.empty()) return;
    const Input flags = parse_input_string("[initialize.convert]\n" + text, "the options");
    if (!input.contains("initialize") || !input.at("initialize").is_table()) {
        input.as_table()["initialize"] = Input(Input::table_type{});
    }
    Input & init = input.as_table()["initialize"];
    if (!init.as_table().entries().contains("convert")) init.as_table()["convert"] = Input(Input::table_type{});
    for (const auto & [name, value] : flags.at("initialize").at("convert").as_table().entries()) {
        init.as_table()["convert"].as_table()[name] = value;
    }
}

std::string default_output(const std::string & source, const std::string & gas) {
    const std::filesystem::path p(source);
    return (p.parent_path() / (p.stem().string() + "_" + gas + ".restart")).string();
}

} // namespace

int restart_convert(const cli::Args & args) {
    const std::string source = args.positional.at(0);
    const std::string input_file = args.value_or("-i", "");
    const bool dry_run = args.has("--dry-run"), json = args.value_or("--format", "text") == "json";
    const std::string accepted = args.value_or("--accept-mapping", "");
    if (input_file.empty()) throw Usage("give the target case's input file with -i");
    if (args.value_or("--format", "text") != "text" && !json) throw Usage("--format takes text or json");
    if (!accepted.empty() && accepted != chemistry::CompositionMap::RULE) {
        throw Usage("--accept-mapping names the rule it accepts: " +
                              std::string(chemistry::CompositionMap::RULE) + ", not \"" + accepted + "\"");
    }
    std::string output = args.value_or("-o", "");
    std::streambuf * const out = std::cout.rdbuf();
    std::ostream terminal(out);
    if (!args.has("--verbose")) std::cout.rdbuf(nullptr);
    int status = 0;
    std::string error, refusal;
    bool written = false, approved = false;
    std::string summary_json, summary_text;
    try {
        Input input = parse_input(input_file);
        add_options(input, args);
        Solver solver;
        auto approve = [&](const gas_conversion::Summary & s) {
            if (output.empty()) output = default_output(source, s.target_gas);
            std::string decision = "no";
            if (comm::is_root()) {
                summary_text = s.text();
                summary_json = s.json();
                if (!json) terminal << summary_text << std::flush;
                if (dry_run) {
                    decision = "dry";
                } else if (s.exact || accepted == chemistry::CompositionMap::RULE) {
                    decision = "yes";
                } else if (!json && isatty(STDIN_FILENO)) {
                    terminal << "Accept the mapping rule " << s.rule << " and write " << output << "? [y/N] " << std::flush;
                    std::string answer;
                    std::getline(std::cin, answer);
                    decision = answer == "y" || answer == "Y" || answer == "yes" ? "yes" : "no";
                } else {
                    decision = "refused";
                }
            }
            decision = comm::broadcast(decision);
            approved = decision == "yes";
            if (decision == "refused") {
                refusal = "the conversion is not exact (see the summary), so nothing is written: pass "
                          "--accept-mapping " + s.rule + " to accept its rule, or --dry-run to print the summary only";
            } else if (decision == "no") {
                refusal = "not accepted: nothing written";
            }
            return approved;
        };
        const std::optional<gas_conversion::Summary> s = solver.convert_restart(input, source, approve, output);
        if (s) {
            written = true;
            if (comm::is_root() && !json) terminal << "Wrote " << output << "\n";
        } else if (comm::is_root() && !json) {
            terminal << source << " holds the species of " << input_file << ": nothing to convert; start the case "
                     << "from it with --restart " << source << "\n";
        }
    } catch (const gas_conversion::Declined &) {
        status = dry_run ? 0 : 1;
    } catch (const Usage &) {
        std::cout.rdbuf(out);
        throw;
    } catch (const std::exception & e) {
        error = e.what();
        status = 1;
    }
    std::cout.rdbuf(out);
    if (comm::is_root()) {
        if (json) {
            std::string text = summary_json.empty() ? "{\"schema\":\"" + std::string(gas_conversion::SCHEMA) +
                                                          "\",\"version\":" + std::to_string(gas_conversion::VERSION) + "}"
                                                    : summary_json;
            text.pop_back();
            text += std::string(",\"dry_run\":") + (dry_run ? "true" : "false") +
                    ",\"accepted\":" + (approved ? "true" : "false") +
                    ",\"written\":" + (written ? json::quote(output) : "null") +
                    ",\"error\":" + (!error.empty() ? json::quote(error) : !refusal.empty() && !dry_run ? json::quote(refusal) : "null") + "}";
            std::cout << text << "\n";
        }
        if (!error.empty()) logging::error(diagnostic_code::SETUP, "mallard restart convert: " + error);
        if (error.empty() && !refusal.empty() && !dry_run) logging::error(diagnostic_code::SETUP, "mallard restart convert: " + refusal);
    }
    return status;
}

int restart_convert_main(const cli::Args & args) {
    int argc = args.argc;
    char ** argv = args.argv;
    comm::Session session(argc, argv);
    binding::select_device(argc, argv);
    Kokkos::initialize(argc, argv);
    int status = 0;
    try {
        status = restart_convert(args);
    } catch (const Usage & e) {
        if (comm::is_root()) logging::error(diagnostic_code::USAGE, std::string("mallard restart convert: ") + e.what());
        status = 2;
    }
    Kokkos::finalize();
    return status;
}

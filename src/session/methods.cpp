/**
 * @file methods.cpp
 * @brief The requests of mallard session (docs/guide/session.md#requests) and the command that serves them.
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 */

#include <algorithm>
#include <cerrno>
#include <filesystem>
#include <iostream>
#include <sstream>
#include <streambuf>

#include <unistd.h>

#include <Kokkos_Core.hpp>

#include "build_info.h"
#include "cli.h"
#include "comm.h"
#include "commands.h"
#include "common_typedef.h"
#include "interfaces.h"
#include "log.h"
#include "output_reader.h"
#include "run_control.h"
#include "run_ctl.h"
#include "session.h"
#include "units.h"

namespace session {

namespace {

/** @brief params as the arguments of a command whose options are named as them: "mesh_fields" is --mesh-fields. */
cli::Args as_options(const json::Value & params, std::initializer_list<std::string_view> names) {
    cli::Args args;
    for (size_t i = 0; i < params.names.size(); i++) {
        const std::string & name = params.names[i];
        const json::Value & v = params.members[i];
        if (std::find(names.begin(), names.end(), name) == names.end()) {
            throw Error(code::INVALID_PARAMS, "no parameter " + name);
        }
        std::string option = "--" + name;
        std::replace(option.begin(), option.end(), '_', '-');
        switch (v.type) {
            case json::Value::Type::BOOL:
                if (v.boolean) args.values[option] = "";
                break;
            case json::Value::Type::NUMBER: args.values[option] = json::number(v.number); break;
            case json::Value::Type::STRING: args.values[option] = v.string; break;
            case json::Value::Type::NUL: break;
            default: throw Error(code::INVALID_PARAMS, name + " takes a number, a string or a boolean");
        }
    }
    return args;
}

/** @brief The case's description: its input, dimension and what its setups have read. */
std::string case_json(const Context & context, const std::vector<std::string> * changed = nullptr) {
    json::Object o;
    o.add("schema", CASE_SCHEMA).add("version", CASE_VERSION).add("input", context.path()).add("dimension", N_DIM);
    o.raw("mesh_files_read", std::to_string(context.mesh_files_read()));
    if (changed) {
        std::string list = "[";
        for (const std::string & key : *changed) list += (list.size() > 1 ? "," : "") + json::quote(key);
        o.raw("changed", list + "]");
    }
    return o.str();
}

std::string initialize(Context & context, const json::Value &) {
    std::string result = capabilities_json();
    result.pop_back();
    return result + ",\"case\":" + (context.is_open() ? case_json(context) : "null") + "}";
}

std::string shutdown(Context &, const json::Value &) { return "null"; }

std::string case_open(Context & context, const json::Value & params) {
    const std::string input = string_param(params, "input");
    if (input.empty()) throw Error(code::INVALID_PARAMS, "case.open takes input, the input file's path");
    const json::Value * text = params.find("text");
    if (text && text->type != json::Value::Type::STRING) throw Error(code::INVALID_PARAMS, "text takes a string");
    if (!text && !std::filesystem::is_regular_file(input)) {
        throw Error(code::UNREADABLE, "no input file " + input, json::Object().add("path", input).str());
    }
    context.open(input, text ? std::optional<std::string>(text->string) : std::nullopt);
    return case_json(context);
}

std::string case_update(Context & context, const json::Value & params) {
    const json::Value * text = params.find("text");
    if (!text || text->type != json::Value::Type::STRING) {
        throw Error(code::INVALID_PARAMS, "case.update takes text, the whole input");
    }
    const std::vector<std::string> changed = context.update(text->string);
    return case_json(context, &changed);
}

std::string case_close(Context & context, const json::Value &) {
    context.close();
    return "null";
}

std::string case_check(Context & context, const json::Value & params) {
    const std::string key = "case.check " + json::dump(params);
    if (const std::optional<std::string> known = context.remembered(key)) return *known;
    const cli::Args args =
        as_options(params, {"partial", "ranks", "memory", "throughput", "steps", "mesh_fields", "stencils", "cells",
                            "region", "weights", "catalog", "view", "boundary_values", "restart"});
    CheckOptions options;
    options.input_file = context.path();
    if (const std::string error = check_options(args, options); !error.empty()) throw Error(code::INVALID_PARAMS, error);
    const std::string result = check_json(check_case(context, options, args.value_or("--restart", "")).report);
    context.remember(key, result);
    return result;
}

std::string case_view(Context & context, const json::Value & params) {
    as_options(params, {"dir"});
    const std::string dir = string_param(params, "dir");
    if (dir.empty()) throw Error(code::INVALID_PARAMS, "case.view takes dir, the directory to write the view in");
    const std::string key = "case.view " + json::dump(params);
    if (const std::optional<std::string> known = context.remembered(key)) return *known;
    CheckOptions options;
    options.input_file = context.path();
    options.view_dir = dir;
    const CheckReport report = check_case(context, options).report;
    const std::string view = case_view_json(report);
    if (view == "null") {
        throw Error(code::INVALID_PARAMS, "the case cannot be set up, so it has no view",
                    json::Object().raw("diagnostics", diagnostics_json(report.diagnostics)).str());
    }
    context.remember(key, view);
    return view;
}

std::string case_boundary_values(Context & context, const json::Value & params) {
    as_options(params, {"dir"});
    const std::string dir = string_param(params, "dir");
    if (dir.empty()) throw Error(code::INVALID_PARAMS, "case.boundary_values takes dir, the directory to write them in");
    const std::string key = "case.boundary_values " + json::dump(params);
    if (const std::optional<std::string> known = context.remembered(key)) return *known;
    CheckOptions options;
    options.input_file = context.path();
    options.boundary_values_dir = dir;
    const CheckReport report = check_case(context, options).report;
    if (report.boundary_values.empty()) {
        throw Error(code::INVALID_PARAMS, "the case cannot be set up, so its boundaries have no values",
                    json::Object().raw("diagnostics", diagnostics_json(report.diagnostics)).str());
    }
    context.remember(key, report.boundary_values);
    return report.boundary_values;
}

std::string listing(const json::Value & catalog, const json::Value & params, const KeyCase * per_case) {
    const std::string section = section_argument(string_param(params, "section"));
    try {
        return key_listing_json(catalog, keys_in_section(catalog, section, per_case), section, per_case);
    } catch (const std::invalid_argument & e) {
        throw Error(code::INVALID_PARAMS, e.what());
    }
}

std::string case_keys(Context & context, const json::Value & params) {
    as_options(params, {"section"});
    const std::string key = "case.keys " + json::dump(params);
    if (const std::optional<std::string> known = context.remembered(key)) return *known;
    KeyCase key_case{.input = context.path(), .complete = false, .diagnostics = {}};
    CheckOptions options;
    options.input_file = context.path();
    const CheckResult result = check_case(context, options, "", false);
    key_case.complete = result.report.setup_done;
    key_case.diagnostics = result.report.diagnostics;
    if (result.catalog.empty()) {
        throw Error(code::INVALID_PARAMS, "the case cannot be set up",
                    json::Object().raw("diagnostics", diagnostics_json(result.report.diagnostics)).str());
    }
    const std::string listed = listing(json::parse(result.catalog), params, &key_case);
    context.remember(key, listed);
    return listed;
}

std::string keys_catalog(Context &, const json::Value & params) {
    as_options(params, {"section"});
    try {
        return listing(general_key_catalog(), params, nullptr);
    } catch (const std::runtime_error & e) {
        if (dynamic_cast<const Error *>(&e)) throw;
        throw Error(code::UNREADABLE, e.what());
    }
}

std::string file_info(Context &, const json::Value & params) {
    as_options(params, {"file"});
    const std::string file = string_param(params, "file");
    if (file.empty()) throw Error(code::INVALID_PARAMS, "file.info takes file, the output or restart file to describe");
    std::error_code error;
    if (std::filesystem::is_directory(file, error) || output::is_mesh_file(file)) {
        throw Error(code::INVALID_PARAMS, file + " is a run directory or a mesh file: file.info describes output and "
                                                 "restart files (mallard info describes the others)");
    }
    try {
        return file_info_json(file);
    } catch (const std::exception & e) {
        throw Error(code::UNREADABLE, e.what());
    }
}

std::string units_catalog(Context &, const json::Value & params) {
    as_options(params, {});
    return units::catalog_json();
}

std::vector<std::string> strings_param(const json::Value & params, const std::string & name) {
    std::vector<std::string> out;
    const json::Value * v = params.find(name);
    if (!v || v->is_null()) return out;
    if (v->type != json::Value::Type::ARRAY) throw Error(code::INVALID_PARAMS, name + " takes an array of strings");
    for (const json::Value & item : v->array) {
        if (item.type != json::Value::Type::STRING) throw Error(code::INVALID_PARAMS, name + " takes an array of strings");
        out.push_back(item.string);
    }
    return out;
}

double number_param(const json::Value & params, const std::string & name, double fallback) {
    const json::Value * v = params.find(name);
    if (!v || v->is_null()) return fallback;
    if (v->type != json::Value::Type::NUMBER || v->number < 0) throw Error(code::INVALID_PARAMS, name + " takes a non-negative number");
    return v->number;
}

void only(const json::Value & params, std::initializer_list<std::string_view> names) {
    for (const std::string & name : params.names) {
        if (std::find(names.begin(), names.end(), name) == names.end()) throw Error(code::INVALID_PARAMS, "no parameter " + name);
    }
}

std::string control_run(Context &, const json::Value & params) {
    only(params, {"run", "action", "settings", "id", "wait", "timeout", "allow_files"});
    run_ctl::Request request;
    request.run = string_param(params, "run");
    if (request.run.empty()) throw Error(code::INVALID_PARAMS, "run.control takes run, a run directory or run ID");
    request.action = string_param(params, "action");
    if (const json::Value * settings = params.find("settings"); settings && settings->type == json::Value::Type::OBJECT) {
        for (size_t i = 0; i < settings->names.size(); i++) {
            const json::Value & v = settings->members[i];
            request.settings.push_back(settings->names[i] + "=" + (v.type == json::Value::Type::STRING ? v.string : json::dump(v)));
        }
    } else if (settings && !settings->is_null()) {
        throw Error(code::INVALID_PARAMS, "settings takes an object of key: value");
    }
    request.id = string_param(params, "id");
    request.wait = string_param(params, "wait", "done");
    request.timeout = number_param(params, "timeout", -1.0);
    request.allow_files = bool_param(params, "allow_files", false);
    return run_ctl::control(request);
}

std::string launch_run(Context & context, const json::Value & params) {
    only(params, {"input", "run_dir", "restart", "launcher", "options", "timeout"});
    run_ctl::Launch launch;
    launch.input = string_param(params, "input");
    if (launch.input.empty()) {
        if (!context.is_open()) throw Error(code::INVALID_PARAMS, "run.launch takes input, or runs the open case");
        if (context.unsaved()) {
            throw Error(code::UNSAVED, "the case's text differs from " + context.path() + ": save it, so that the run runs a file",
                        json::Object().add("path", context.path()).str());
        }
        launch.input = context.path();
    }
    launch.run_dir = string_param(params, "run_dir");
    launch.restart = string_param(params, "restart");
    launch.launcher = strings_param(params, "launcher");
    launch.options = strings_param(params, "options");
    launch.timeout = number_param(params, "timeout", 60.0);
    return run_ctl::launch(launch);
}

constexpr Method METHODS[] = {
    {"initialize", "mallard version --format json", interfaces::CAPABILITIES.name, false, initialize},
    {"shutdown", "", "", false, shutdown},
    {"case.open", "", CASE_SCHEMA, false, case_open},
    {"case.update", "", CASE_SCHEMA, true, case_update},
    {"case.close", "", "", false, case_close},
    {"case.check", "mallard check INPUT --format json", "mallard-check", true, case_check},
    {"case.view", "mallard check INPUT --view DIR --format json", interfaces::CASE_VIEW.name, true, case_view},
    {"case.boundary_values", "mallard check INPUT --boundary-values DIR --format json", interfaces::BOUNDARY_VALUES.name,
     true, case_boundary_values},
    {"case.keys", "mallard list keys [SECTION] INPUT --format json", "mallard-key-catalog", true, case_keys},
    {"keys.catalog", "mallard list keys [SECTION] --format json", "mallard-key-catalog", false, keys_catalog},
    {"file.info", "mallard info FILE --format json", interfaces::FILE_INFO.name, false, file_info},
    {"units.catalog", "mallard list units --format json", interfaces::UNIT_CATALOG.name, false, units_catalog},
    {"run.launch", "mallard -i INPUT --detach", run_ctl::LAUNCH_SCHEMA, false, launch_run},
    {"run.control", "mallard ctl RUN ACTION [KEY=VALUE...] --format json", run_control::SCHEMA, false, control_run},
};

} // namespace

std::span<const Method> methods() { return METHODS; }

std::string methods_json() {
    std::string list = "[";
    for (const Method & m : methods()) {
        json::Object entry;
        entry.add("name", std::string(m.name));
        entry.raw("command", m.command.empty() ? "null" : json::quote(m.command));
        entry.raw("result", m.result.empty() ? "null" : json::quote(m.result));
        list += (list.size() > 1 ? "," : "") + entry.str();
    }
    return list + "]";
}

const Method * method(std::string_view name) {
    for (const Method & m : METHODS) {
        if (m.name == name) return &m;
    }
    return nullptr;
}

/**
 * @brief Standard input read with read(2), bypassing stdio: the reader thread blocked in it then holds no lock that
 *        the exit after shutdown waits for.
 */
class StdinBuffer : public std::streambuf {
    protected:
        int_type underflow() override {
            ssize_t n = 0;
            do {
                n = ::read(0, buffer, sizeof buffer);
            } while (n < 0 && errno == EINTR);
            if (n <= 0) return traits_type::eof();
            setg(buffer, buffer, buffer + n);
            return traits_type::to_int_type(buffer[0]);
        }

    private:
        char buffer[1 << 16];
};

} // namespace session

int session_main(const cli::Args & args) {
    if (args.positional.size() > 1) return cli::usage_error(*cli::command("session"), "give one input file");
    int argc = args.argc;
    char ** argv = args.argv;
    comm::Session mpi(argc, argv);
    // stdout carries the protocol alone: setup text goes to stderr
    std::ostream protocol(std::cout.rdbuf());
    std::cout.rdbuf(std::cerr.rdbuf());
    int status = 0;
    {
        Kokkos::ScopeGuard kokkos(argc, argv);
        session::Context context(true);
        if (!args.positional.empty()) context.open(args.positional[0]);
        // never destroyed: the reader thread may still be blocked reading them at exit
        std::istream * requests = new std::istream(new session::StdinBuffer);
        status = session::serve(*requests, protocol, context);
    }
    std::cout.rdbuf(protocol.rdbuf());
    return status;
}

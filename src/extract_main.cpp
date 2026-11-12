/**
 * @file extract_main.cpp
 * @brief `mallard post extract`: the probes, regions and snapshots of existing output,
 *        `mallard post derive`: snapshots of derived fields of it, and `mallard post average`: its
 *        statistics over snapshots, all by the run's own code on a case set up from the output's input.
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 */

#include <algorithm>
#include <chrono>
#include <cmath>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include <Kokkos_Core.hpp>

#include "cli.h"
#include "commands.h"
#include "derived_catalog.h"
#include "diagnostic_code.h"
#include "favre_average.h"
#include "homogeneous_average.h"
#include "input.h"
#include "input_upgrade.h"
#include "json.h"
#include "log.h"
#include "mesh_block.h"
#include "output_reader.h"
#include "parallel/binding.h"
#include "parallel/comm.h"
#include "provenance.h"
#include "region_input.h"
#include "remap.h"
#include "solver/solver.h"

namespace fs = std::filesystem;

namespace {

using logging::count;
using logging::format;
using logging::item;


/** @brief A usage error found after parsing: printed by rank 0 as cli::usage_error does. */
struct UsageError : std::runtime_error {
    using std::runtime_error::runtime_error;
};

std::vector<std::string> split_list(const std::string & text) {
    std::vector<std::string> out;
    std::stringstream in(text);
    std::string item;
    while (std::getline(in, item, ',')) {
        const size_t a = item.find_first_not_of(" \t"), b = item.find_last_not_of(" \t");
        if (a != std::string::npos) out.push_back(item.substr(a, b - a + 1));
    }
    return out;
}

std::string list_text(const std::vector<std::string> & items) {
    std::string out;
    for (const auto & s : items) out += (out.empty() ? "" : ", ") + s;
    return out;
}

std::string seconds(double t) { return format("%.9g", t); }

std::string units_text(const std::string & units) {
    if (units.empty()) return "not recorded";
    return units == "1" ? "dimensionless" : units;
}

/** @brief One snapshot to extract: of which file, its index there, and its step as the file records it
 *        (its position only in files that record none, written before 0.10). */
struct Selected {
    size_t file = 0;
    size_t index = 0;  // in the file's snapshots
    output::Snapshot snapshot;
    int64_t step = 0;
    bool step_recorded = false;
};

double parse_number(const std::string & text, const std::string & option) {
    try {
        size_t used = 0;
        const double x = std::stod(text, &used);
        if (used == text.size() && std::isfinite(x)) return x;
    } catch (const std::exception &) {
    }
    throw UsageError(option + " takes numbers, not " + text);
}

uint64_t parse_count(const std::string & text, const std::string & option) {
    const double x = parse_number(text, option);
    if (x < 0.0 || x != std::floor(x)) throw UsageError(option + " takes non-negative integers, not " + text);
    return static_cast<uint64_t>(x);
}

/** @brief The snapshots of the files, in order, then those --steps, --times, --every and --latest choose. */
std::vector<Selected> select(const std::vector<output::Contents> & files, const cli::Args & args) {
    std::vector<Selected> all;
    for (size_t f = 0; f < files.size(); f++) {
        const output::Contents & c = files[f];
        const bool xdmf = c.format.find("XDMF") != std::string::npos;
        for (size_t k = 0; k < c.snapshots.size(); k++) {
            const output::Snapshot & s = c.snapshots[k];
            Selected e{f, k, s, s.step, s.step >= 0};
            // An XDMF index records the step of its last snapshot only; each member file has its own
            if (!e.step_recorded && xdmf && s.file != c.path) {
                const output::Contents member = output::open(s.file);
                if (!member.snapshots.empty() && member.snapshots[0].step >= 0) {
                    e.step = member.snapshots[0].step;
                    e.step_recorded = true;
                }
            }
            all.push_back(e);
        }
    }
    const bool recorded = std::all_of(all.begin(), all.end(), [](const Selected & e) { return e.step_recorded; });
    for (size_t i = 0; i < all.size() && !recorded; i++) {
        all[i].step = static_cast<int64_t>(i);
        all[i].step_recorded = false;
    }
    std::vector<Selected> out = all;
    if (args.has("--steps")) {
        if (!recorded) {
            throw UsageError("--steps needs files that record their steps, which VTKHDF files do not; choose with "
                             "--times");
        }
        std::vector<Selected> chosen;
        for (const std::string & s : split_list(args.value_or("--steps", ""))) {
            const auto step = static_cast<int64_t>(parse_count(s, "--steps"));
            const auto it = std::find_if(out.begin(), out.end(), [&](const Selected & e) { return e.step == step; });
            if (it == out.end()) throw std::runtime_error("no snapshot of step " + s + " in the files given");
            chosen.push_back(*it);
        }
        out = chosen;
    }
    if (args.has("--times")) {
        const std::string text = args.value_or("--times", "");
        const size_t colon = text.find(':');
        std::vector<Selected> chosen;
        if (colon == std::string::npos) {
            const double t = parse_number(text, "--times");
            const auto nearest = std::min_element(out.begin(), out.end(), [&](const Selected & a, const Selected & b) {
                return std::abs(a.snapshot.t - t) < std::abs(b.snapshot.t - t);
            });
            if (nearest != out.end()) chosen.push_back(*nearest);
        } else {
            const double lo = parse_number(text.substr(0, colon), "--times");
            const double hi = parse_number(text.substr(colon + 1), "--times");
            for (const Selected & e : out) {
                if (e.snapshot.t >= lo && e.snapshot.t <= hi) chosen.push_back(e);
            }
        }
        out = chosen;
    }
    if (args.has("--every")) {
        const uint64_t n = parse_count(args.value_or("--every", ""), "--every");
        if (n == 0) throw UsageError("--every takes a positive integer");
        std::vector<Selected> chosen;
        for (size_t i = 0; i < out.size(); i += n) chosen.push_back(out[i]);
        out = chosen;
    }
    if (args.has("--latest") && !out.empty()) out = {out.back()};
    if (out.empty()) throw std::runtime_error("no snapshot of the files given matches the selection");
    return out;
}

/** @brief The input recorded in an output file's run block: resolved where recorded so, else as written. */
std::string recorded_input(const output::Contents & c, bool & resolved) {
    std::string block = c.run;
    if (block.empty() && !c.snapshots.empty() && c.snapshots[0].file != c.path) {
        block = output::open(c.snapshots[0].file).run;
    }
    if (block.empty()) return "";
    const json::Value p = json::parse(block);
    const json::Value * input = p.find("input");
    for (const char * key : {"resolved", "text"}) {
        const json::Value * text = input ? input->find(key) : nullptr;
        if (text && text->type == json::Value::Type::STRING && !text->string.empty()) {
            resolved = std::string(key) == "resolved";
            return text->string;
        }
    }
    return "";
}

/** @brief path under the output directory: relative paths keep their directories, absolute ones their name. */
std::string under(const std::string & dir, const std::string & path) {
    const fs::path p(path);
    return (fs::path(dir) / (p.is_absolute() ? p.filename() : p)).lexically_normal().string();
}

Input string_array(const std::vector<std::string> & items) {
    Input::array_type a;
    for (const auto & s : items) a.emplace_back(s);
    return Input(a);
}

/** @brief The value of --region: a name, or an inline TOML table. */
Input region_value(const std::string & text) {
    if (text.find('{') == std::string::npos) return Input(text);
    const Input parsed = parse_input_string("region = " + text, "--region");
    return parsed.as_table().entries().at("region");
}

/** @brief Whether a [[probes]] entry writes samples, not only reductions: a region probe writes none without n_points. */
bool writes_samples(const Input & input, const Input::table_type & entry) {
    if (!entry.entries().contains("region")) return true;
    const Input & value = entry.entries().at("region");
    const Input * names = region::named_regions(input);
    const bool named = value.is_string() && names && names->as_table().entries().contains(value.as_string());
    const Input & table = named ? names->as_table().entries().at(value.as_string()) : value;
    return probe::sampled(table) || entry.entries().contains("n_points");
}

std::string probe_extension(const Input::table_type & entry) {
    const auto it = entry.entries().find("format");
    const std::string format = it != entry.entries().end() && it->second.is_string() ? it->second.as_string() : "csv";
    return format == "hdf5" ? ".h5" : format == "vtkhdf" ? ".vtkhdf" : format == "vtp" ? ".pvd" : ".csv";
}

/** @brief Whether a probe entry averages in time: average = true, or [statistics] averages as variables. */
bool time_averaged(const Input & entry) {
    const auto & t = entry.as_table().entries();
    if (t.contains("average") && t.at("average").is_boolean() && t.at("average").as_boolean()) return true;
    if (!t.contains("fields") || !t.at("fields").is_array()) return false;
    for (const Input & v : t.at("fields").as_array()) {
        if (v.is_string() && (v.as_string().starts_with("MEAN_") || v.as_string().starts_with("COV_"))) return true;
    }
    return false;
}

/** @brief What the extraction writes, as the case input sets it up. */
struct Plan {
    std::vector<std::string> probes;  // names
    std::vector<std::string> files;   // that may be written: probe files, reductions files, snapshot indexes
    std::string snapshot_prefix;
    std::vector<std::string> conditional;  // the conditional statistics files
};

/**
 * @brief The case input of a run made to write only what is extracted: the probes chosen, written
 *        every time to the output directory, and the snapshots of --snapshot; no other output, no
 *        [initialize] (the state is the file's) and no statistics.
 */
Plan prepare(Input & input, const cli::Args & args, const std::string & dir) {
    Plan plan;
    auto & top = input.as_table().entries();
    for (const char * key : {"write_data", "forces", "integrals", "statistics", "initialize"}) top.erase(key);
    if (!top.contains("output") || !top.at("output").is_table()) top["output"] = Input(Input::table_type{});
    auto & output = top.at("output").as_table().entries();
    output["run_log"] = Input("");
    output["health"] = Input(false);  // Its ledger and repair audit follow time steps

    const bool chosen = args.has("--probe");
    const bool others = args.has("--region") || args.has("--snapshot");
    std::vector<std::string> wanted = split_list(args.value_or("--probe", ""));
    Input::array_type probes;
    std::vector<std::string> available, kept, averaged;
    if (top.contains("probes") && top.at("probes").is_array()) {
        for (const Input & entry : top.at("probes").as_array()) {
            const auto & t = entry.as_table().entries();
            const std::string name = t.contains("name") && t.at("name").is_string() ? t.at("name").as_string() : "";
            available.push_back(name);
            if (chosen ? std::find(wanted.begin(), wanted.end(), name) == wanted.end() : others) continue;
            // Averages over the run's steps are not in a snapshot's state
            if (time_averaged(entry)) {
                if (chosen) {
                    throw UsageError("the probe " + name + " averages in time (average = true, or MEAN_/COV_ "
                                     "variables), over the run's steps, which output does not hold");
                }
                averaged.push_back(name);
                continue;
            }
            probes.push_back(entry);
            kept.push_back(name);
        }
    }
    for (const std::string & name : wanted) {
        if (std::find(available.begin(), available.end(), name) == available.end()) {
            throw std::runtime_error("the input has no probe named " + name + " (its [[probes]]: " +
                                     (available.empty() ? "none" : list_text(available)) + ")");
        }
    }
    const std::string name = args.value_or("--name", "extract");
    const std::vector<std::string> fields = split_list(args.value_or("--fields", "RHO,U,P,T"));
    if (fields.empty()) throw UsageError("--fields lists no variable");
    if (args.has("--region")) {
        Input::table_type entry;
        entry["name"] = Input(name);
        entry["region"] = region_value(args.value_or("--region", ""));
        entry["fields"] = string_array(fields);
        if (args.has("--reductions")) entry["reductions"] = string_array(split_list(args.value_or("--reductions", "")));
        if (args.has("--sampling")) entry["sampling"] = Input(args.value_or("--sampling", ""));
        if (std::find(kept.begin(), kept.end(), name) != kept.end()) {
            throw UsageError("the probe " + name + " of the input is extracted too: give --region another --name");
        }
        probes.emplace_back(entry);
    } else if (args.has("--reductions") || args.has("--sampling")) {
        throw UsageError(std::string(args.has("--reductions") ? "--reductions" : "--sampling") + " applies to --region");
    }
    if (!averaged.empty() && comm::is_root()) {
        logging::warning(diagnostic_code::EXTRACT_AVERAGED, "not extracted: the probes " + list_text(averaged) + ", which average in time over the "
                         "run's steps");
    }
    if (!others && probes.empty()) {
        throw UsageError("the input has no [[probes]]: give --region or --snapshot, or an input with [[probes]] (-i)");
    }
    for (Input & entry : probes) {
        auto & t = entry.as_table().entries();
        if (!t.contains("name") || !t.at("name").is_string()) continue;
        const std::string probe = t.at("name").as_string();
        plan.probes.push_back(probe);
        t.erase("time_interval");
        t["interval"] = Input(1);
        const auto file = t.find("file");
        if (file != t.end() || writes_samples(input, entry.as_table())) {
            const std::string path = under(dir, file != t.end() ? file->second.as_string()
                                                                : "probe_" + probe + probe_extension(entry.as_table()));
            t["file"] = Input(path);
            plan.files.push_back(path);
        }
        const auto reductions = t.find("reductions_file");
        const std::string path = under(dir, reductions != t.end() ? reductions->second.as_string()
                                                                  : "probe_" + probe + "_reductions.csv");
        t["reductions_file"] = Input(path);
        plan.files.push_back(path);
    }
    if (probes.empty()) {
        top.erase("probes");
    } else {
        top["probes"] = Input(probes);
    }
    if (args.has("--snapshot")) {
        const std::string format = args.value_or("--snapshot", "");
        if (format != "vtkhdf" && format != "vtu") throw UsageError("--snapshot takes vtkhdf or vtu, not " + format);
        Input::table_type entry;
        plan.snapshot_prefix = under(dir, name);
        entry["prefix"] = Input(plan.snapshot_prefix);
        entry["fields"] = string_array(fields);
        entry["interval"] = Input(1);
        entry["format"] = Input(format);
        if (args.has("--region")) entry["region"] = region_value(args.value_or("--region", ""));
        top["write_data"] = Input(Input::array_type{Input(entry)});
        plan.files.push_back(plan.snapshot_prefix + (format == "vtkhdf" ? ".vtkhdf.series" : ".pvd"));
        plan.files.push_back(plan.snapshot_prefix + ".vtkhdf");
    }
    return plan;
}

/** @brief The value of an option that takes an inline TOML table, as the input has it. */
Input inline_table(const std::string & text, const std::string & option) {
    const Input parsed = parse_input_string("value = " + text, option);
    const Input & value = parsed.as_table().entries().at("value");
    if (!value.is_table()) throw UsageError(option + " takes an inline TOML table ({...}), not " + text);
    return value;
}

/** @brief An example of the [output] table a derived-field need takes, for messages. */
std::string need_hint(uint16_t need) {
    if (need == derived::need::REFERENCE) return "--reference '{rho = 1.2, u = [30, 0], p = 101325}'";
    if (need == derived::need::PROGRESS) return "--mixture '{progress = {T_unburnt = 300, T_burnt = 2200}}'";
    return "--mixture '{fuel = {X = {CH4 = 1}}, oxidizer = {X = {O2 = 1, N2 = 3.76}}}'";
}

/**
 * @brief The case input of mallard post derive: no probes, one snapshot of the --fields (with the
 *        state, with --with-state) at each step, and [output] reference and mixture from --reference
 *        and --mixture, which replace the input's.
 */
Plan prepare_derive(Input & input, const cli::Args & args, const std::string & dir) {
    Plan plan;
    auto & top = input.as_table().entries();
    for (const char * key : {"write_data", "forces", "integrals", "statistics", "initialize", "probes"}) top.erase(key);
    if (!top.contains("output") || !top.at("output").is_table()) top["output"] = Input(Input::table_type{});
    auto & output = top.at("output").as_table().entries();
    output["run_log"] = Input("");
    output["health"] = Input(false);
    if (args.has("--reference")) output["reference"] = inline_table(args.value_or("--reference", ""), "--reference");
    if (args.has("--mixture")) output["mixture"] = inline_table(args.value_or("--mixture", ""), "--mixture");

    if (!args.has("--fields")) throw UsageError("--fields is required: the derived fields to write (MACH,P0,CP)");
    const std::vector<std::string> fields = split_list(args.value_or("--fields", ""));
    if (fields.empty()) throw UsageError("--fields lists no field");
    const auto has = [&](const char * table, const char * key) {
        if (!output.contains(table)) return false;
        return key == nullptr || (output.at(table).is_table() && output.at(table).contains(key));
    };
    for (const std::string & name : fields) {
        const derived::Field f = derived::find(name.c_str());
        if (f == derived::N_FIELDS) {
            std::vector<std::string> names;
            for (const auto & d : derived::FIELDS) names.emplace_back(d.name);
            const std::string like = suggest_value(name, names);
            throw UsageError("--fields: " + name + " is not a derived field" +
                             (like.empty() ? "" : " (did you mean " + like + "?)") +
                             "; mallard list fields lists them, in its derived and mixture groups");
        }
        const uint16_t needs = derived::FIELDS[f].needs;
        uint16_t missing = 0;
        if ((needs & derived::need::REFERENCE) && !has("reference", nullptr)) missing = derived::need::REFERENCE;
        if ((needs & derived::need::STREAMS) && !has("mixture", "fuel")) missing = derived::need::STREAMS;
        if ((needs & derived::need::PROGRESS) && !has("mixture", "progress")) missing = derived::need::PROGRESS;
        if ((needs & derived::need::SPECIES_PAIR) && !has("mixture", "fuel") &&
            !(has("mixture", "fuel_species") && has("mixture", "oxidizer_species"))) {
            missing = derived::need::STREAMS;
        }
        if (missing) {
            throw std::runtime_error(name + " needs [output] " +
                                     (missing == derived::need::REFERENCE ? "reference" : "mixture keys") +
                                     " (" + derived::FIELDS[f].needs_text + "), which the run's input does not " +
                                     "have: give them, e.g. " + need_hint(missing));
        }
    }
    std::vector<std::string> variables;
    if (args.has("--with-state")) variables = {"RHO", "U", "P", "T"};
    variables.insert(variables.end(), fields.begin(), fields.end());
    const std::string format = args.value_or(
        "--snapshot", have_hdf5() && (comm::size() == 1 || have_parallel_hdf5()) ? "vtkhdf" : "vtu");
    if (format != "vtkhdf" && format != "vtu") throw UsageError("--snapshot takes vtkhdf or vtu, not " + format);
    Input::table_type entry;
    plan.snapshot_prefix = under(dir, args.value_or("--name", "derived"));
    entry["prefix"] = Input(plan.snapshot_prefix);
    entry["fields"] = string_array(variables);
    entry["interval"] = Input(1);
    entry["format"] = Input(format);
    if (args.has("--region")) entry["region"] = region_value(args.value_or("--region", ""));
    top["write_data"] = Input(Input::array_type{Input(entry)});
    plan.files.push_back(plan.snapshot_prefix + (format == "vtkhdf" ? ".vtkhdf.series" : ".pvd"));
    plan.files.push_back(plan.snapshot_prefix + ".vtkhdf");
    return plan;
}

/** @brief Fields the state needs, read from a snapshot: the conservative variables if it holds them, else primitives. */
std::vector<std::string> state_fields(const output::Contents & c) {
    const bool conservative = c.field("RHO") && c.field("RHOE") && (c.field("RHOU") || c.field("RHOU_X"));
    std::vector<std::string> out;
    for (const output::Field & f : c.fields) {
        const std::string & n = f.name;
        const bool flow = conservative ? n == "RHO" || n == "RHOE" || n.starts_with("RHOU") || n.starts_with("RHOY_")
                                       : n == "RHO" || n == "U" || n.starts_with("U_") || n == "P" || n.starts_with("Y_");
        if (flow || n == "T") out.push_back(n);
    }
    return out;
}

/** @brief The state of a snapshot or restart file on the mesh's local cells (collective for nothing: each rank reads). */
RestartData read_state(const output::Contents & c, const Selected & e, const Mesh & mesh) {
    const bool distributed = mesh.n_global_cells > 0;
    const uint64_t n_global = distributed ? mesh.n_global_cells : mesh.n_cells;
    const auto wrong_mesh = [&](uint64_t n) {
        return std::runtime_error("has " + count(n) + " cells, but the case's mesh has " +
                                  count(n_global) + ": give the input of the run that wrote it (-i)");
    };
    if (c.kind == output::Kind::restart) {
        RestartData data = read_restart(c.path, mesh.h_global_cell_id.empty() ? nullptr : &mesh.h_global_cell_id);
        if (data.n_cells != n_global) throw wrong_mesh(data.n_cells);
        return data;
    }
    RestartData data;
    data.step = static_cast<uint64_t>(e.step);
    data.t = e.snapshot.t;
    data.n_cells = n_global;
    const std::vector<std::string> fields = state_fields(c);
    for (const std::string & name : fields) {
        const std::vector<double> values = output::read(c, name, e.index);
        const uint32_t components = c.field(name)->components;
        if (values.size() != n_global * components) throw wrong_mesh(values.size() / components);
        for (uint32_t d = 0; d < components && d < N_DIM; d++) {
            data.names.push_back(components == 1 ? name : name + "_" + "XYZ"[d]);
            std::vector<rtype> & local = data.fields.emplace_back(mesh.n_cells);
            for (uint32_t i = 0; i < mesh.n_cells; i++) {
                local[i] = static_cast<rtype>(values[mesh.h_global_cell(i) * components + d]);
            }
        }
    }
    return data;
}

/** @brief The run block a snapshot is recorded by: its file's, with the run id its own snapshot block gives. */
std::string source_block(const output::Contents & c, const Selected & e) {
    std::string block = c.run;
    if (e.snapshot.file != c.path) {
        try {
            block = output::open(e.snapshot.file).run;
        } catch (const std::exception &) {
        }
    }
    if (e.snapshot.run_id.empty() || provenance::find_string(block, "run", "run_id") == e.snapshot.run_id) return block;
    json::Object run;
    run.add("run_id", e.snapshot.run_id);
    const std::string directory = provenance::find_string(block, "run", "directory");
    if (!directory.empty()) run.add("directory", directory);
    return json::Object().raw("run", run.str()).str();
}

/** @brief A file the extraction wrote, as the reader reads it back. */
struct Written {
    output::Contents contents;
    std::vector<std::pair<std::string, double>> last;  // tables: every column of the last time, after step and t
};

std::vector<Written> read_written(const Plan & plan, fs::file_time_type since) {
    std::vector<Written> out;
    for (const std::string & file : plan.files) {
        std::error_code error;
        if (!fs::exists(file, error) || fs::last_write_time(file, error) < since) continue;
        Written w{output::open(file), {}};
        if (w.contents.kind == output::Kind::table && !w.contents.snapshots.empty()) {
            const size_t k = w.contents.snapshots.size() - 1;
            for (const output::Field & f : w.contents.fields) {
                if (f.name == "step" || f.name == "t") continue;
                const std::vector<double> v = output::read(w.contents, f.name, k);
                if (v.size() == 1) w.last.emplace_back(f.name, v[0]);
            }
        }
        out.push_back(std::move(w));
    }
    return out;
}

std::string snapshot_json(const output::Contents & c, const Selected & e) {
    return json::Object()
        .add("file", c.path)
        .add("member", e.snapshot.file)
        .add("index", e.snapshot.index)
        .add("t", e.snapshot.t)
        .add("step", e.step)
        .add("step_recorded", e.step_recorded)
        .str();
}

void print_json(const std::vector<output::Contents> & files, const std::vector<Selected> & selected,
                const std::string & case_from, const std::string & dir, const std::vector<Written> & written) {
    json::Object o;
    std::string list = "[";
    for (const auto & c : files) list += (list.size() > 1 ? "," : "") + json::quote(c.path);
    o.raw("files", list + "]").add("input", case_from).add("output", dir);
    std::string snaps = "[";
    for (const auto & e : selected) snaps += (snaps.size() > 1 ? "," : "") + snapshot_json(files[e.file], e);
    o.raw("snapshots", snaps + "]");
    std::string out = "[";
    for (const Written & w : written) {
        const output::Contents & c = w.contents;
        json::Object f;
        f.add("path", c.path).add("kind", std::string(output::kind_name(c.kind))).add("format", c.format);
        f.add("times", c.snapshots.size());
        std::string fields = "[";
        for (const auto & field : c.fields) {
            fields += (fields.size() > 1 ? "," : "") +
                      json::Object().add("name", field.name).add("units", field.units).add("components", field.components).str();
        }
        f.raw("fields", fields + "]");
        if (!w.last.empty()) {
            json::Object last;
            last.add("t", c.snapshots.back().t);
            json::Object values;
            for (const auto & [name, value] : w.last) {
                const output::Field * field = c.field(name);
                values.raw(name, json::Object().add("value", value).add("units", field ? field->units : "").str());
            }
            last.raw("values", values.str());
            f.raw("last", last.str());
        }
        out += (out.size() > 1 ? "," : "") + f.str();
    }
    o.raw("written", out + "]");
    std::cout << o.str() << "\n";
}

void print_text(const std::vector<output::Contents> & files, const std::vector<Selected> & selected,
                const std::string & case_from, const std::string & dir, const std::vector<Written> & written,
                size_t n_available, bool derive) {
    logging::section(derive ? "Derive" : "Extract");
    std::vector<std::string> names;
    for (const auto & c : files) names.push_back(c.path + " (" + c.format + ")");
    item(files.size() == 1 ? "File" : "Files", list_text(names));
    item("Case", case_from);
    const Selected & first = selected.front();
    const Selected & last = selected.back();
    item("Snapshots", count(selected.size()) + " of " + count(n_available) + ", t = " + seconds(first.snapshot.t) +
                          (selected.size() > 1 ? " to " + seconds(last.snapshot.t) : "") + " s");
    if (!first.step_recorded) item("Steps", "not recorded in the files: numbered by position, from 0");
    item("Output", dir);

    logging::section("Snapshots", count(selected.size()) + (derive ? " derived" : " extracted"));
    std::vector<std::vector<std::string>> rows;
    for (size_t i = 0; i < selected.size(); i++) {
        const Selected & e = selected[i];
        rows.push_back({std::to_string(i), seconds(e.snapshot.t), std::to_string(e.step) + (e.step_recorded ? "" : " *"),
                        fs::path(e.snapshot.file).filename().string() +
                            (e.snapshot.index ? ", index " + std::to_string(e.snapshot.index) : "")});
    }
    logging::table({"#", "t [s]", "step", "file"}, rows);
    if (!first.step_recorded) logging::line("  *: position in the files, which record no step.");

    logging::section("Written", count(written.size(), "file"));
    rows.clear();
    for (const Written & w : written) {
        const output::Contents & c = w.contents;
        const bool timed = !c.snapshots.empty();
        rows.push_back({c.path, output::kind_name(c.kind), timed ? count(c.snapshots.size()) : "-",
                        timed ? seconds(c.snapshots.front().t) : "-", timed ? seconds(c.snapshots.back().t) : "-",
                        count(c.fields.size()), c.format});
    }
    logging::table({"file", "kind", "times", "first t [s]", "last t [s]", "fields", "format"}, rows);

    for (const Written & w : written) {
        if (w.last.empty()) continue;
        const output::Contents & c = w.contents;
        const output::Snapshot & s = c.snapshots.back();
        logging::section(fs::path(c.path).filename().string(),
                         "t = " + seconds(s.t) + " s" + (c.snapshots.size() > 1 ? ", the last" : ""));
        rows.clear();
        for (const auto & [name, value] : w.last) {
            const output::Field * f = c.field(name);
            rows.push_back({name, format("%.9g", value), units_text(f ? f->units : "")});
        }
        logging::table({"column", "value", "units"}, rows);
    }
}

/** @brief The files of the command line: restart files and snapshots of the whole mesh, in its order. */
std::vector<output::Contents> open_files(const cli::Args & args, const std::string & verb) {
    std::vector<output::Contents> files;
    for (const std::string & path : args.positional) {
        output::Contents c = output::open(path);
        if (c.kind != output::Kind::snapshot && c.kind != output::Kind::restart) {
            throw std::runtime_error(path + " is a " + output::kind_name(c.kind) +
                                     " file; " + verb + " takes restart files and snapshots of the whole mesh");
        }
        if (!c.global_order) {
            throw std::runtime_error(path + " holds cells in an order of its own (" + c.format +
                                     "), not the mesh's: " + verb + " from restart files, VTKHDF or XDMF snapshots of the "
                                     "whole mesh");
        }
        files.push_back(std::move(c));
    }
    return files;
}

/** @brief The case's input: -i, else the input in the first file's run block; case_from says which. */
Input case_input(const cli::Args & args, const output::Contents & first, std::string & case_from) {
    Input input;
    if (args.has("-i")) {
        case_from = args.value_or("-i", "");
        input = parse_input(case_from);
    } else {
        bool resolved = false;
        const std::string text = recorded_input(first, resolved);
        if (text.empty()) {
            throw std::runtime_error(first.path + " records no input (files of Mallard 0.9 and later do, in their "
                                     "run block): give the input file with -i");
        }
        case_from = std::string(resolved ? "the resolved input" : "the input as written") + " in the run block of " +
                    first.path;
        input = parse_input_string(text, first.path + " (its run block)");
    }
    for (const auto & finding : input_upgrade::apply(input)) {
        if (comm::is_root()) logging::warning(diagnostic_code::KEY_DEPRECATED, input_upgrade::advice(finding));
    }
    return input;
}

int extract(const cli::Args & args, const bool derive) {
    const std::string verb = derive ? "derive" : "extract";
    const std::string dir = args.value_or("--output", derive ? "derived" : "extract");
    const std::string report = args.value_or("--format", "text");
    if (report != "text" && report != "json") throw UsageError("--format takes text or json, not " + report);
    if (args.has("--sampling") && args.value_or("--sampling", "") != "cell" && args.value_or("--sampling", "") != "point") {
        throw UsageError("--sampling takes cell or point, not " + args.value_or("--sampling", ""));
    }
    const std::vector<output::Contents> files = open_files(args, verb);
    size_t n_available = 0;
    for (const auto & c : files) n_available += c.snapshots.size();
    const std::vector<Selected> selected = select(files, args);

    std::string case_from;
    Input input = case_input(args, files[0], case_from);
    const Plan plan = derive ? prepare_derive(input, args, dir) : prepare(input, args, dir);

    // The setup's report and the writers' events are the run's, not this command's: errors and warnings still show
    std::ostringstream quiet;
    std::streambuf * const out = std::cout.rdbuf(quiet.rdbuf());
    struct Restore {
        std::streambuf * out;
        ~Restore() { std::cout.rdbuf(out); }
    } restore{out};
    if (comm::is_root()) fs::create_directories(dir);
    const auto since = fs::file_time_type::clock::now() - std::chrono::seconds(1);
    Solver solver;
    size_t current = 0;
    solver.set_state_reader([&](const Mesh & mesh) {
        // The first state is read within init, after the run block starts and before it lists the files read
        if (current == 0 && comm::is_root()) {
            for (const Selected & e : selected) {
                provenance::read_start_file("snapshot", e.snapshot.file, source_block(files[e.file], e),
                                            static_cast<uint64_t>(e.step), e.snapshot.t);
            }
        }
        try {
            return read_state(files[selected[current].file], selected[current], mesh);
        } catch (const std::exception & e) {
            throw std::runtime_error(selected[current].snapshot.file + ": " + e.what());
        }
    });
    solver.init(input);
    const auto hash = [](const std::string & block) { return provenance::find_string(block, "mesh", "sha256"); };
    const std::string then = hash(files[0].run), now = hash(provenance::json());
    if (comm::is_root() && !then.empty() && !now.empty() && then != now) {
        logging::warning(diagnostic_code::MESH_DIFFERS, files[0].path + " was written on a mesh file of other contents (the sha256 in its run block "
                         "differs) with as many cells: check that the case's mesh is the run's.");
    }
    for (current = 0; current < selected.size(); current++) {
        try {
            solver.write_output_of_state(current > 0);
        } catch (const std::exception & e) {
            const std::string what = e.what();
            if (what.starts_with(selected[current].snapshot.file)) throw;
            throw std::runtime_error(selected[current].snapshot.file + ": " + what);
        }
    }
    comm::barrier();
    std::cout.rdbuf(out);
    if (!comm::is_root()) return 0;
    const std::vector<Written> written = read_written(plan, since);
    if (report == "json") {
        print_json(files, selected, case_from, dir, written);
    } else {
        print_text(files, selected, case_from, dir, written, n_available, derive);
    }
    return 0;
}

/** @brief What mallard post average computes. */
struct Averages {
    std::vector<std::string> fields;
    std::vector<std::pair<uint32_t, uint32_t>> products;  // indices into fields
    bool favre = false;
    std::vector<Input> conditional;  // [[statistics.conditional]] entries
    std::string homogeneous;  // the directions averaged over ("xz"); empty: a snapshot of every cell
    double rtol = 1e-6;

    std::string product(const std::pair<uint32_t, uint32_t> & p) const { return fields[p.first] + "_" + fields[p.second]; }
    /** @brief Favre averages: of the fields and products without RHO, as indices into fields. */
    std::vector<uint32_t> favre_fields() const {
        std::vector<uint32_t> out;
        for (uint32_t i = 0; i < fields.size(); i++) {
            if (fields[i] != "RHO") out.push_back(i);
        }
        return out;
    }
    std::vector<std::pair<uint32_t, uint32_t>> favre_products() const {
        std::vector<std::pair<uint32_t, uint32_t>> out;
        for (const auto & p : products) {
            if (fields[p.first] != "RHO" && fields[p.second] != "RHO") out.push_back(p);
        }
        return out;
    }
    /** @brief The products of a field with itself, whose square roots are RMS_<field>. */
    std::vector<uint32_t> variances() const {
        std::vector<uint32_t> out;
        for (uint32_t p = 0; p < products.size(); p++) {
            if (products[p].first == products[p].second) out.push_back(p);
        }
        return out;
    }
    /** @brief The fields written: MEAN_, COV_, RMS_, then FAVRE_MEAN_ and FAVRE_COV_. */
    std::vector<std::string> variables() const {
        std::vector<std::string> out;
        for (const auto & f : fields) out.push_back("MEAN_" + f);
        for (const auto & p : products) out.push_back("COV_" + product(p));
        for (const uint32_t p : variances()) out.push_back("RMS_" + fields[products[p].first]);
        if (!favre) return out;
        for (const uint32_t i : favre_fields()) out.push_back("FAVRE_MEAN_" + fields[i]);
        for (const auto & p : favre_products()) out.push_back("FAVRE_COV_" + product(p));
        return out;
    }
};

std::vector<std::string> velocity_names() {
    std::vector<std::string> out;
    for (size_t d = 0; d < N_DIM; d++) out.push_back(std::string("U_") + "XYZ"[d]);
    return out;
}

/** @brief Whether [statistics] fields take name (species variables are checked against the case later). */
void check_variable(const std::string & name, const std::string & option) {
    if (name.starts_with("Y_") || name.starts_with("RHOY_") || name.starts_with("OMEGA_") || name == "HRR" ||
        derived::find(name.c_str()) != derived::N_FIELDS) {
        return;
    }
    std::vector<std::string> known = {"RHO"};
    known.insert(known.end(), PRIMITIVE_NAMES.begin(), PRIMITIVE_NAMES.end());
    known.insert(known.end(), CONSERVATIVE_NAMES.begin() + 1, CONSERVATIVE_NAMES.end());
    if (std::find(known.begin(), known.end(), name) != known.end()) return;
    std::vector<std::string> names = known;
    for (const auto & d : derived::FIELDS) names.emplace_back(d.name);
    const std::string like = suggest_value(name, names);
    throw UsageError(option + ": " + name + " is not a variable [statistics] averages" +
                     (like.empty() ? "" : " (did you mean " + like + "?)") + "; it takes " + list_text(known) +
                     ", Y_<species>, RHOY_<species>, HRR, OMEGA_<species> and the derived fields of mallard list "
                     "fields");
}

/** @brief --conditional: an inline table, or an array of them, of [[statistics.conditional]] entries. */
std::vector<Input> conditional_entries(const std::string & text) {
    const Input parsed = parse_input_string("value = " + text, "--conditional");
    const Input & value = parsed.as_table().entries().at("value");
    std::vector<Input> out;
    if (value.is_table()) out.push_back(value);
    if (value.is_array()) {
        for (const Input & item : value.as_array()) {
            if (!item.is_table()) break;
            out.push_back(item);
        }
        if (out.size() != value.as_array().size()) out.clear();
    }
    if (out.empty()) {
        throw UsageError("--conditional takes an inline TOML table ({on = \"MIXTURE_FRACTION\", fields = [\"T\"]}) "
                         "or an array of them, not " + text);
    }
    for (const Input & entry : out) {
        for (const std::string & name : ConditionalStatistics::sampled(entry)) check_variable(name, "--conditional");
    }
    return out;
}

Averages choose_averages(const cli::Args & args) {
    Averages a;
    a.favre = args.has("--favre");
    if (args.has("--conditional")) a.conditional = conditional_entries(args.value_or("--conditional", ""));
    std::vector<std::string> fields = {"RHO"};
    for (const auto & u : velocity_names()) fields.push_back(u);
    fields.insert(fields.end(), {"P", "T"});
    if (!a.conditional.empty() && !a.favre && !args.has("--products")) fields.clear();
    if (args.has("--fields")) fields = split_list(args.value_or("--fields", ""));
    if (fields.empty() && (args.has("--fields") || a.conditional.empty())) throw UsageError("--fields lists no variable");
    const auto add_field = [&](const std::string & name, const std::string & option) {
        check_variable(name, option);
        const auto it = std::find(a.fields.begin(), a.fields.end(), name);
        if (it != a.fields.end()) return static_cast<uint32_t>(it - a.fields.begin());
        a.fields.push_back(name);
        return static_cast<uint32_t>(a.fields.size() - 1);
    };
    for (size_t i = 0; i < fields.size(); i++) {
        if (std::find(fields.begin(), fields.begin() + static_cast<std::ptrdiff_t>(i), fields[i]) !=
            fields.begin() + static_cast<std::ptrdiff_t>(i)) {
            throw UsageError("--fields: " + fields[i] + " is listed twice");
        }
    }
    if (a.favre && std::find(fields.begin(), fields.end(), "RHO") == fields.end()) a.fields.push_back("RHO");
    for (const auto & f : fields) add_field(f, "--fields");
    const auto add_product = [&](uint32_t i, uint32_t j) {
        for (const auto & p : a.products) {
            if ((p.first == i && p.second == j) || (p.first == j && p.second == i)) return false;
        }
        a.products.emplace_back(i, j);
        return true;
    };
    if (args.has("--products")) {
        const std::string text = args.value_or("--products", "");
        for (const std::string & product : text == "none" ? std::vector<std::string>{} : split_list(text)) {
            const size_t star = product.find('*');
            if (star == std::string::npos || product.find('*', star + 1) != std::string::npos) {
                throw UsageError("--products: " + product + " is not of the form A*B");
            }
            const uint32_t i = add_field(product.substr(0, star), "--products");
            if (!add_product(i, add_field(product.substr(star + 1), "--products"))) {
                throw UsageError("--products: " + product + " is listed twice");
            }
        }
    } else {
        for (uint32_t i = 0; i < a.fields.size(); i++) add_product(i, i);
        const std::vector<std::string> u = velocity_names();
        for (size_t d = 0; d < u.size(); d++) {
            for (size_t e = d + 1; e < u.size(); e++) {
                const auto i = std::find(a.fields.begin(), a.fields.end(), u[d]);
                const auto j = std::find(a.fields.begin(), a.fields.end(), u[e]);
                if (i == a.fields.end() || j == a.fields.end()) continue;
                add_product(static_cast<uint32_t>(i - a.fields.begin()), static_cast<uint32_t>(j - a.fields.begin()));
            }
        }
    }
    for (const std::string & d : split_list(args.value_or("--homogeneous", ""))) {
        if (d.size() != 1 || d[0] < 'x' || d[0] > 'z' || a.homogeneous.find(d) != std::string::npos) {
            throw UsageError("--homogeneous takes directions x, y and z, each once, separated by commas, not " +
                             args.value_or("--homogeneous", ""));
        }
        a.homogeneous += d;
    }
    if (args.has("--homogeneous") && a.homogeneous.empty()) throw UsageError("--homogeneous lists no direction");
    if (!a.homogeneous.empty() && a.fields.empty()) {
        throw UsageError("--homogeneous averages fields over directions: give --fields with --conditional");
    }
    if (args.has("--rtol")) {
        a.rtol = parse_number(args.value_or("--rtol", ""), "--rtol");
        if (a.rtol <= 0.0) throw UsageError("--rtol takes a positive number");
    }
    return a;
}

/**
 * @brief The case input of mallard post average: [statistics] of the fields and products, from
 *        t_start at every state read, [output] reference and mixture from --reference and
 *        --mixture, no other output, and a snapshot of the averages unless they are averaged over
 *        homogeneous directions.
 */
Plan prepare_average(Input & input, const cli::Args & args, const std::string & dir, const Averages & a, double t_start) {
    Plan plan;
    auto & top = input.as_table().entries();
    for (const char * key : {"write_data", "forces", "integrals", "statistics", "initialize", "probes"}) top.erase(key);
    if (!top.contains("output") || !top.at("output").is_table()) top["output"] = Input(Input::table_type{});
    auto & output = top.at("output").as_table().entries();
    output["run_log"] = Input("");
    output["health"] = Input(false);
    if (args.has("--reference")) output["reference"] = inline_table(args.value_or("--reference", ""), "--reference");
    if (args.has("--mixture")) output["mixture"] = inline_table(args.value_or("--mixture", ""), "--mixture");
    std::vector<std::string> products;
    for (const auto & [i, j] : a.products) products.push_back(a.fields[i] + "*" + a.fields[j]);
    Input::table_type statistics;
    statistics["fields"] = string_array(a.fields);
    statistics["products"] = string_array(products);
    statistics["t_start"] = Input(t_start);
    statistics["interval"] = Input(1);
    if (!a.conditional.empty()) {
        Input::array_type entries;
        for (Input entry : a.conditional) {
            auto & e = entry.as_table().entries();
            if (!e.contains("file")) {
                e["file"] = Input(under(dir, ConditionalStatistics::default_file(ConditionalStatistics::on_of(entry))));
            }
            plan.conditional.push_back(e.at("file").as_string());
            entries.push_back(entry);
        }
        statistics["conditional"] = Input(entries);
    }
    top["statistics"] = Input(statistics);
    if (!a.homogeneous.empty() || a.fields.empty()) return plan;
    const std::string format = args.value_or(
        "--snapshot", have_hdf5() && (comm::size() == 1 || have_parallel_hdf5()) ? "vtkhdf" : "vtu");
    if (format != "vtkhdf" && format != "vtu") throw UsageError("--snapshot takes vtkhdf or vtu, not " + format);
    Input::table_type entry;
    plan.snapshot_prefix = under(dir, args.value_or("--name", "average"));
    entry["prefix"] = Input(plan.snapshot_prefix);
    entry["fields"] = string_array(a.variables());
    entry["interval"] = Input(1);
    entry["format"] = Input(format);
    if (args.has("--region")) entry["region"] = region_value(args.value_or("--region", ""));
    top["write_data"] = Input(Input::array_type{Input(entry)});
    plan.files.push_back(plan.snapshot_prefix + (format == "vtkhdf" ? ".vtkhdf.series" : ".pvd"));
    plan.files.push_back(plan.snapshot_prefix + ".vtkhdf");
    return plan;
}

/** @brief Each owned cell's weight in averages over homogeneous directions: its volume, times its fraction of --region. */
std::vector<double> cell_weights(const cli::Args & args, const Input & input, const Mesh & mesh) {
    std::vector<double> w(mesh.n_owned());
    for (uint32_t c = 0; c < mesh.n_owned(); c++) w[c] = double(mesh.h_cell_measure(c));
    if (!args.has("--region")) return w;
    region::Regions regions(input, mesh);
    const Input value = region_value(args.value_or("--region", ""));
    region::require_full_dimension(regions.get(value, "--region"), value, "--region");
    const std::vector<double> & fractions = regions.fractions(value, "--region");
    for (uint32_t c = 0; c < mesh.n_owned(); c++) w[c] *= fractions[c];
    double total = 0.0;
    for (const double x : w) total += x;
    if (comm::allreduce(total, comm::Op::SUM) <= 0.0) throw std::runtime_error("--region contains no part of any cell");
    return w;
}

std::vector<std::array<double, N_DIM>> cell_centers(const Mesh & mesh) {
    std::vector<std::array<double, N_DIM>> out(mesh.n_owned());
    for (uint32_t c = 0; c < mesh.n_owned(); c++) FOR_I_DIM out[c][i] = double(mesh.h_cell_coords(c, i));
    return out;
}

/** @brief RMS_<a> = sqrt(COV_a_a) of every averaged covariance of a field with itself, as columns of the profile. */
void add_rms(homogeneous::Profile & p) {
    const size_t n = p.names.size();
    std::vector<std::pair<std::string, size_t>> rms;
    for (size_t j = 0; j < n; j++) {
        const std::string & name = p.names[j];
        if (!name.starts_with("COV_")) continue;
        const std::string body = name.substr(4);
        const size_t k = body.size() / 2;
        if (body.size() % 2 == 1 && body[k] == '_' && body.substr(0, k) == body.substr(k + 1)) {
            rms.emplace_back("RMS_" + body.substr(0, k), j);
        }
    }
    if (rms.empty()) return;
    std::vector<double> values;
    for (size_t r = 0; r < p.rows(); r++) {
        for (size_t j = 0; j < n; j++) values.push_back(p.values[r * n + j]);
        for (const auto & entry : rms) values.push_back(std::sqrt(std::max(0.0, p.values[r * n + entry.second])));
    }
    for (const auto & entry : rms) p.names.push_back(entry.first);
    p.values = std::move(values);
}

/** @brief Writes the profile to OUTPUT/NAME.csv (rank 0); its path. */
std::string write_profile(const homogeneous::Profile & p, const cli::Args & args, const std::string & dir) {
    const std::string path = under(dir, args.value_or("--name", "average") + ".csv");
    if (!comm::is_root()) return path;
    fs::create_directories(dir);
    std::ofstream out(path);
    out << p.csv();
    if (!out) throw std::runtime_error("cannot write " + path);
    return path;
}

std::string json_strings(const std::vector<std::string> & items) {
    std::string out = "[";
    for (const auto & s : items) out += (out.size() > 1 ? "," : "") + json::quote(s);
    return out + "]";
}

std::string directions_text(const std::string & directions) {
    std::vector<std::string> out;
    for (const char d : directions) out.emplace_back(1, d);
    return list_text(out);
}

void print_profile(const homogeneous::Profile & p, const std::string & path, const std::string & homogeneous) {
    logging::section("Profile", count(p.rows()) + " rows");
    item("Averaged over", directions_text(homogeneous) + ", weighted by cell volume; a row per position in " +
                              list_text(p.axes));
    item("Columns", list_text(p.names));
    item("Written", path);
}

/**
 * @brief mallard post average --statistics: the [statistics] fields one snapshot holds (MEAN_,
 *        COV_, FAVRE_MEAN_, FAVRE_COV_), averaged over homogeneous directions, on the case's mesh.
 */
int average_statistics(const cli::Args & args, const std::string & dir, const std::string & report) {
    for (const char * option : {"--fields", "--products", "--favre", "--t-start", "--snapshot", "--reference", "--mixture"}) {
        if (args.has(option)) {
            throw UsageError(std::string(option) + " does not apply to --statistics, which averages the time "
                             "averages the file holds");
        }
    }
    const Averages a = choose_averages(args);
    if (a.homogeneous.empty()) {
        throw UsageError("--statistics needs --homogeneous: the [statistics] fields are time averages already");
    }
    const std::vector<output::Contents> files = open_files(args, "average");
    const std::vector<Selected> selected = select(files, args);
    if (selected.size() > 1) {
        throw UsageError("--statistics averages the [statistics] fields of one snapshot, and " +
                         count(selected.size()) + " are selected: choose one with --latest, --steps or --times");
    }
    const Selected & e = selected.front();
    const output::Contents & c = files[e.file];
    std::string case_from;
    const Input input = case_input(args, files[0], case_from);
    Input setup(Input::table_type{});
    for (const char * key : {"mesh", "periodic", "regions"}) {
        if (input.contains(key)) setup.as_table()[key] = input.at(key);
    }
    const bool axisymmetric = find_value_or<bool>(input, "physics", "axisymmetric", false);
    // Centers and weights of the owned cells, and their global ids; faces only for --region and curved cells
    std::vector<uint64_t> global;
    std::vector<std::array<double, N_DIM>> centers;
    std::vector<double> weights;
    uint64_t n_global = 0;
    {
        std::ostringstream quiet;
        std::streambuf * const out = std::cout.rdbuf(quiet.rdbuf());
        struct Restore {
            std::streambuf * out;
            ~Restore() { std::cout.rdbuf(out); }
        } restore{out};
        const MeshBlock block = read_mesh_block(setup);
        const bool curved = comm::allreduce(uint32_t(!block.face_high_order_offsets.empty()), comm::Op::MAX) > 0;
        if (args.has("--region") || curved) {
            const std::shared_ptr<Mesh> mesh = remap::source_mesh(setup, 1);
            if (axisymmetric) mesh->make_axisymmetric();
            n_global = mesh->n_global_cells > 0 ? mesh->n_global_cells : mesh->n_cells;
            for (uint32_t i = 0; i < mesh->n_owned(); i++) global.push_back(mesh->h_global_cell(i));
            centers = cell_centers(*mesh);
            weights = cell_weights(args, input, *mesh);
        } else {
            n_global = comm::allreduce(block.n_cells(), comm::Op::SUM);
            for (const CellGeometry & g : block_cell_geometry(block, axisymmetric)) {
                global.push_back(block.first_cell + global.size());
                std::array<double, N_DIM> x;
                FOR_I_DIM x[i] = double(g.centroid[i]);
                centers.push_back(x);
                weights.push_back(double(g.measure));
            }
        }
    }
    const uint32_t n_owned = static_cast<uint32_t>(global.size());
    std::vector<homogeneous::Column> columns;
    const auto index = [&](const std::string & name) {
        for (size_t j = 0; j < columns.size(); j++) {
            if (columns[j].name == name) return static_cast<int32_t>(j);
        }
        return int32_t(-1);
    };
    for (const std::string prefix : {"MEAN_", "FAVRE_MEAN_", "COV_", "FAVRE_COV_"}) {
        for (const output::Field & f : c.fields) {
            if (!f.name.starts_with(prefix)) continue;
            const std::vector<double> values = output::read(c, f.name, e.index);
            if (values.size() != n_global * f.components) {
                throw std::runtime_error(c.path + " has " + count(values.size() / f.components) + " cells, but the "
                                         "case's mesh has " + count(n_global) + ": give the input of the run that "
                                         "wrote it (-i)");
            }
            for (uint32_t d = 0; d < f.components && d < N_DIM; d++) {
                homogeneous::Column col;
                col.name = f.components == 1 ? f.name : f.name + "_" + "XYZ"[d];
                col.favre = prefix.starts_with("FAVRE_");
                col.values.resize(n_owned);
                for (uint32_t i = 0; i < n_owned; i++) {
                    col.values[i] = values[global[i] * f.components + d];
                }
                columns.push_back(std::move(col));
            }
        }
    }
    if (columns.empty()) {
        throw std::runtime_error(c.path + " holds no [statistics] fields (MEAN_, COV_): average its states "
                                 "without --statistics, or write them with variables = [\"MEAN_*\", \"COV_*\"]");
    }
    // COV_<a>_<b> pairs with MEAN_<a> and MEAN_<b> where the file holds both
    for (homogeneous::Column & col : columns) {
        const std::string cov = col.favre ? "FAVRE_COV_" : "COV_", mean = col.favre ? "FAVRE_MEAN_" : "MEAN_";
        if (!col.name.starts_with(cov)) continue;
        const std::string body = col.name.substr(cov.size());
        for (size_t k = 1; k + 1 < body.size(); k++) {
            if (body[k] != '_') continue;
            const int32_t i = index(mean + body.substr(0, k)), j = index(mean + body.substr(k + 1));
            if (i >= 0 && j >= 0) {
                col.mean_a = i;
                col.mean_b = j;
                break;
            }
        }
    }
    std::vector<double> density;
    if (std::any_of(columns.begin(), columns.end(), [](const auto & col) { return col.favre; })) {
        const int32_t rho = index("MEAN_RHO");
        if (rho < 0) throw std::runtime_error(c.path + " holds Favre averages but not MEAN_RHO, which weighs them");
        density = columns[static_cast<size_t>(rho)].values;
    }
    homogeneous::Profile profile = homogeneous::average(centers, weights, density,
                                                        columns, a.homogeneous, a.rtol);
    add_rms(profile);
    const std::string path = write_profile(profile, args, dir);
    if (!comm::is_root()) return 0;
    if (report == "json") {
        json::Object o;
        o.add("file", c.path).add("input", case_from).raw("snapshot", snapshot_json(c, e));
        o.add("homogeneous", a.homogeneous).add("rows", profile.rows()).raw("columns", json_strings(profile.names));
        o.raw("written", json_strings({path}));
        std::cout << o.str() << "\n";
        return 0;
    }
    logging::section("Average");
    item("File", c.path + " (" + c.format + ")");
    item("Case", case_from);
    item("Snapshot", "t = " + seconds(e.snapshot.t) + " s, step " + std::to_string(e.step) +
                         (e.step_recorded ? "" : " (its position: the file records no step)"));
    item("Averaged", "its [statistics] fields, over space only");
    print_profile(profile, path, a.homogeneous);
    return 0;
}

int average(const cli::Args & args) {
    const std::string dir = args.value_or("--output", "average");
    const std::string report = args.value_or("--format", "text");
    if (report != "text" && report != "json") throw UsageError("--format takes text or json, not " + report);
    if (args.has("--statistics")) return average_statistics(args, dir, report);
    const Averages a = choose_averages(args);
    if (!a.homogeneous.empty() && args.has("--snapshot")) {
        throw UsageError("--snapshot does not apply to --homogeneous, which writes a CSV profile");
    }
    const std::vector<output::Contents> files = open_files(args, "average");
    size_t n_available = 0;
    for (const auto & c : files) n_available += c.snapshots.size();
    const std::vector<Selected> selected = select(files, args);
    for (size_t k = 1; k < selected.size(); k++) {
        if (selected[k].snapshot.t <= selected[k - 1].snapshot.t) {
            throw std::runtime_error(selected[k].snapshot.file + " (t = " + seconds(selected[k].snapshot.t) +
                                     " s) follows " + selected[k - 1].snapshot.file + " (t = " +
                                     seconds(selected[k - 1].snapshot.t) + " s): give the snapshots in increasing "
                                     "time, each once");
        }
    }
    const double t_first = selected.front().snapshot.t;
    double t_start = 0.0;
    if (args.has("--t-start")) {
        t_start = parse_number(args.value_or("--t-start", ""), "--t-start");
        if (t_start >= t_first) {
            throw UsageError("--t-start must come before the first snapshot (t = " + seconds(t_first) + " s), where "
                             "its weight starts");
        }
    } else if (selected.size() < 2) {
        throw UsageError("one snapshot has no time since a previous one to weigh it by: give --t-start");
    } else {
        t_start = t_first - (selected[1].snapshot.t - t_first);
    }

    std::string case_from;
    Input input = case_input(args, files[0], case_from);
    const Plan plan = prepare_average(input, args, dir, a, t_start);

    std::ostringstream quiet;
    std::streambuf * const out = std::cout.rdbuf(quiet.rdbuf());
    struct Restore {
        std::streambuf * out;
        ~Restore() { std::cout.rdbuf(out); }
    } restore{out};
    if (comm::is_root()) fs::create_directories(dir);
    const auto since = fs::file_time_type::clock::now() - std::chrono::seconds(1);
    Solver solver;
    size_t current = 0;
    solver.set_state_reader([&](const Mesh & mesh) {
        try {
            return read_state(files[selected[current].file], selected[current], mesh);
        } catch (const std::exception & e) {
            throw std::runtime_error(selected[current].snapshot.file + ": " + e.what());
        }
    });
    const std::vector<uint32_t> variances = a.variances();
    const std::vector<uint32_t> favre_fields = a.favre_fields();
    const std::vector<std::pair<uint32_t, uint32_t>> favre_products = a.favre_products();
    Kokkos::View<rtype **, Kokkos::LayoutLeft, Kokkos::HostSpace> rms;
    std::unique_ptr<FavreAverage> favre;
    solver.set_output_data([&](const Mesh & mesh, std::vector<Data> & data) {
        rms = decltype(rms)("rms", mesh.n_cells, variances.size());
        for (size_t k = 0; k < variances.size(); k++) {
            data.emplace_back("RMS_" + a.fields[a.products[variances[k]].first], Kokkos::subview(rms, Kokkos::ALL(), k));
        }
        if (!a.favre) return;
        std::vector<std::string> fields;
        std::vector<uint32_t> column(a.fields.size());
        for (const uint32_t i : favre_fields) {
            column[i] = static_cast<uint32_t>(fields.size());
            fields.push_back(a.fields[i]);
        }
        std::vector<std::pair<uint32_t, uint32_t>> products;
        for (const auto & [i, j] : favre_products) products.emplace_back(column[i], column[j]);
        favre = std::make_unique<FavreAverage>(fields, products, solver.get_species_names(), solver.derived_selection,
                                               mesh.n_cells);
        favre->register_data(data);
    });
    solver.init(input);
    const auto hash = [](const std::string & block) { return provenance::find_string(block, "mesh", "sha256"); };
    const std::string then = hash(files[0].run), now = hash(provenance::json());
    if (comm::is_root() && !then.empty() && !now.empty() && then != now) {
        logging::warning(diagnostic_code::MESH_DIFFERS, files[0].path + " was written on a mesh file of other contents (the sha256 in its run block "
                         "differs) with as many cells: check that the case's mesh is the run's.");
    }
    const Mesh & mesh = *solver.get_mesh();
    std::vector<double> weights;
    rtype t_last = static_cast<rtype>(t_start);
    for (current = 0; current < selected.size(); current++) {
        try {
            const CellSampler sampler = solver.sample_statistics_of_state(current > 0);
            const rtype w = solver.get_time() - t_last;
            t_last = solver.get_time();
            weights.push_back(double(w));
            if (favre) favre->sample(w, sampler, mesh.n_owned());
        } catch (const std::exception & e) {
            const std::string what = e.what();
            if (what.starts_with(selected[current].snapshot.file)) throw;
            throw std::runtime_error(selected[current].snapshot.file + ": " + what);
        }
    }
    const Statistics & statistics = solver.get_statistics();
    const auto & means = statistics.host_means();
    const auto & cov = statistics.host_covariances();
    for (size_t k = 0; k < variances.size(); k++) {
        for (uint32_t c = 0; c < mesh.n_cells; c++) rms(c, k) = std::sqrt(std::max(0.0_r, cov(c, variances[k])));
    }
    if (favre) favre->copy_device_to_host();

    homogeneous::Profile profile;
    std::string profile_path;
    if (a.homogeneous.empty() && !a.fields.empty()) {
        solver.write_output_of_state(false);
    } else if (!a.conditional.empty()) {
        statistics.write(solver.get_step(), solver.get_time(), snapshot_block::current_reason(true));
    }
    if (!a.homogeneous.empty()) {
        std::vector<homogeneous::Column> columns;
        const auto add_column = [&](const std::string & name, const auto & view, uint32_t k, bool is_favre) {
            homogeneous::Column col;
            col.name = name;
            col.favre = is_favre;
            col.values.resize(mesh.n_owned());
            for (uint32_t c = 0; c < mesh.n_owned(); c++) col.values[c] = double(view(c, k));
            columns.push_back(std::move(col));
        };
        for (uint32_t i = 0; i < a.fields.size(); i++) add_column("MEAN_" + a.fields[i], means, i, false);
        for (uint32_t p = 0; p < a.products.size(); p++) {
            add_column("COV_" + a.product(a.products[p]), cov, p, false);
            columns.back().mean_a = static_cast<int32_t>(a.products[p].first);
            columns.back().mean_b = static_cast<int32_t>(a.products[p].second);
        }
        std::vector<double> density;
        if (favre) {
            const int32_t first = static_cast<int32_t>(columns.size());
            const auto favre_column = [&](uint32_t i) {
                return first + static_cast<int32_t>(std::find(favre_fields.begin(), favre_fields.end(), i) -
                                                    favre_fields.begin());
            };
            for (uint32_t k = 0; k < favre_fields.size(); k++) {
                add_column("FAVRE_MEAN_" + a.fields[favre_fields[k]], favre->host_means(), k, true);
            }
            for (uint32_t p = 0; p < favre_products.size(); p++) {
                add_column("FAVRE_COV_" + a.product(favre_products[p]), favre->host_covariances(), p, true);
                columns.back().mean_a = favre_column(favre_products[p].first);
                columns.back().mean_b = favre_column(favre_products[p].second);
            }
            const auto rho = std::find(a.fields.begin(), a.fields.end(), "RHO");
            density = columns[static_cast<size_t>(rho - a.fields.begin())].values;
        }
        profile = homogeneous::average(cell_centers(mesh), cell_weights(args, input, mesh), density, columns,
                                       a.homogeneous, a.rtol);
        add_rms(profile);
        profile_path = write_profile(profile, args, dir);
    }
    comm::barrier();
    std::cout.rdbuf(out);
    if (!comm::is_root()) return 0;
    const std::vector<Written> written =
        a.homogeneous.empty() && !a.fields.empty() ? read_written(plan, since) : std::vector<Written>{};
    std::vector<std::string> products;
    for (const auto & [i, j] : a.products) products.push_back(a.fields[i] + "*" + a.fields[j]);
    std::vector<std::string> paths;
    for (const Written & w : written) {
        paths.push_back(w.contents.path);
        for (const output::Snapshot & snapshot : w.contents.snapshots) {
            if (snapshot.file != w.contents.path) paths.push_back(snapshot.file);
        }
    }
    if (!profile_path.empty()) paths.push_back(profile_path);
    for (const std::string & path : plan.conditional) paths.push_back(path);
    // The window of the means, or of the conditional statistics without them
    double window_start = statistics.average_start(), window_end = statistics.average_end();
    uint64_t samples = statistics.samples();
    if (a.fields.empty() && !statistics.get_conditional().empty()) {
        const ConditionalStatistics::Result r = statistics.get_conditional().front().result();
        window_start = r.t_first;
        window_end = r.t_last;
        samples = r.samples;
    }
    if (report == "json") {
        json::Object o;
        std::vector<std::string> names;
        for (const auto & c : files) names.push_back(c.path);
        o.raw("files", json_strings(names)).add("input", case_from).add("output", dir);
        o.raw("window", json::Object()
                            .add("start", window_start)
                            .add("end", window_end)
                            .add("samples", samples)
                            .str());
        o.raw("fields", json_strings(a.fields)).raw("products", json_strings(products));
        o.add("favre", a.favre).add("homogeneous", a.homogeneous).raw("conditional", json_strings(plan.conditional));
        std::string snaps = "[";
        for (size_t k = 0; k < selected.size(); k++) {
            json::Object s;
            s.raw("snapshot", snapshot_json(files[selected[k].file], selected[k])).add("weight", weights[k]);
            snaps += (snaps.size() > 1 ? "," : "") + s.str();
        }
        o.raw("snapshots", snaps + "]").raw("written", json_strings(paths));
        std::cout << o.str() << "\n";
        return 0;
    }
    logging::section("Average");
    std::vector<std::string> names;
    for (const auto & c : files) names.push_back(c.path + " (" + c.format + ")");
    item(files.size() == 1 ? "File" : "Files", list_text(names));
    item("Case", case_from);
    item("Snapshots", count(selected.size()) + " of " + count(n_available) + ", t = " + seconds(t_first) +
                          (selected.size() > 1 ? " to " + seconds(selected.back().snapshot.t) : "") + " s");
    item("Window", seconds(window_start) + " to " + seconds(window_end) +
                       " s; each snapshot weighs the time since the previous one (the first: since " +
                       (args.has("--t-start") ? "--t-start" : "as long before it as the second is after it") + ")");
    if (!a.fields.empty()) {
        item("Fields", list_text(a.fields));
        item("Covariances", products.empty() ? "none" : list_text(products));
        item("Favre", a.favre ? "means and covariances, weighted by density" : "no (--favre adds them)");
    }
    for (const ConditionalStatistics & c : statistics.get_conditional()) {
        const OutputSummary o = c.summary();
        item("Conditional", o.contents + ", every " + count(o.steps) + " steps -> " + o.file);
    }
    logging::section("Snapshots", count(selected.size()) + " averaged");
    std::vector<std::vector<std::string>> rows;
    for (size_t i = 0; i < selected.size(); i++) {
        const Selected & e = selected[i];
        rows.push_back({std::to_string(i), seconds(e.snapshot.t), std::to_string(e.step) + (e.step_recorded ? "" : " *"),
                        format("%.6g", weights[i]),
                        fs::path(e.snapshot.file).filename().string() +
                            (e.snapshot.index ? ", index " + std::to_string(e.snapshot.index) : "")});
    }
    logging::table({"#", "t [s]", "step", "weight [s]", "file"}, rows);
    if (!selected.front().step_recorded) logging::line("  *: position in the files, which record no step.");
    if (!a.homogeneous.empty()) {
        print_profile(profile, profile_path, a.homogeneous);
        for (const std::string & path : plan.conditional) item("Written", path);
        return 0;
    }
    logging::section("Written", count(paths.size()) + " files");
    for (const auto & p : paths) logging::line("  " + p);
    return 0;
}

} // namespace

namespace {

enum class Post { extract, derive, average };

int post_command(const cli::Args & args, const Post post) {
    const std::string command = post == Post::extract ? "post extract" : post == Post::derive ? "post derive" : "post average";
    try {
        return post == Post::average ? average(args) : extract(args, post == Post::derive);
    } catch (const UsageError & e) {
        return comm::is_root() ? cli::usage_error(*cli::command(command), e.what()) : 2;
    } catch (const InputError & e) {
        if (comm::is_root()) logging::error(diagnostic_code::EXTRACT, "mallard " + command + ": the case's input: " + e.what());
    } catch (const std::exception & e) {
        if (comm::is_root()) logging::error(diagnostic_code::EXTRACT, "mallard " + command + ": " + e.what());
    }
    return 1;
}

int post_main(const cli::Args & args, const Post post) {
    int argc = args.argc;
    char ** argv = args.argv;
    comm::Session session(argc, argv);
    binding::select_device(argc, argv);
    Kokkos::initialize(argc, argv);
    const int status = post_command(args, post);
    // A rank that failed alone would leave the others waiting in a collective
    if (status == 1 && comm::size() > 1) comm::abort(1);
    Kokkos::finalize();
    return status;
}

} // namespace

int post_extract(const cli::Args & args) { return post_command(args, Post::extract); }

int post_derive(const cli::Args & args) { return post_command(args, Post::derive); }

int post_average(const cli::Args & args) { return post_command(args, Post::average); }

int extract_main(const cli::Args & args) { return post_main(args, Post::extract); }

int derive_main(const cli::Args & args) { return post_main(args, Post::derive); }

int average_main(const cli::Args & args) { return post_main(args, Post::average); }

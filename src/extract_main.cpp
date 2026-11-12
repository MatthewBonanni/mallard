/**
 * @file extract_main.cpp
 * @brief `mallard post extract`: the probes, regions and snapshots of existing output, and
 *        `mallard post derive`: snapshots of derived fields of it, both written by the run's own
 *        code on a case set up from the output's input.
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 */

#include <algorithm>
#include <chrono>
#include <cmath>
#include <exception>
#include <filesystem>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include <Kokkos_Core.hpp>

#include "cli.h"
#include "commands.h"
#include "derived_catalog.h"
#include "diagnostic_code.h"
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

/** @brief Whether a [[probes]] entry writes samples, not only reductions: a probe with region writes none. */
bool writes_samples(const Input & input, const Input::table_type & entry) {
    if (!entry.entries().contains("region")) return true;
    const Input & value = entry.entries().at("region");
    const Input * names = region::named_regions(input);
    const bool named = value.is_string() && names && names->as_table().entries().contains(value.as_string());
    const Input & table = named ? names->as_table().entries().at(value.as_string()) : value;
    return probe::sampled(table) || (N_DIM == 3 && entry.entries().contains("n_points"));
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

    logging::section("Written", count(written.size()) + " files");
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

int extract(const cli::Args & args, const bool derive) {
    const std::string verb = derive ? "derive" : "extract";
    const std::string dir = args.value_or("--output", derive ? "derived" : "extract");
    const std::string report = args.value_or("--format", "text");
    if (report != "text" && report != "json") throw UsageError("--format takes text or json, not " + report);
    if (args.has("--sampling") && args.value_or("--sampling", "") != "cell" && args.value_or("--sampling", "") != "point") {
        throw UsageError("--sampling takes cell or point, not " + args.value_or("--sampling", ""));
    }
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
    size_t n_available = 0;
    for (const auto & c : files) n_available += c.snapshots.size();
    const std::vector<Selected> selected = select(files, args);

    Input input;
    std::string case_from;
    if (args.has("-i")) {
        case_from = args.value_or("-i", "");
        input = parse_input(case_from);
    } else {
        bool resolved = false;
        const std::string text = recorded_input(files[0], resolved);
        if (text.empty()) {
            throw std::runtime_error(files[0].path + " records no input (files of Mallard 0.9 and later do, in their "
                                     "run block): give the input file with -i");
        }
        case_from = std::string(resolved ? "the resolved input" : "the input as written") + " in the run block of " +
                    files[0].path;
        input = parse_input_string(text, files[0].path + " (its run block)");
    }
    for (const auto & finding : input_upgrade::apply(input)) {
        if (comm::is_root()) logging::warning(diagnostic_code::KEY_DEPRECATED, input_upgrade::advice(finding));
    }
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

} // namespace

namespace {

int post_command(const cli::Args & args, const bool derive) {
    const std::string command = derive ? "post derive" : "post extract";
    try {
        return extract(args, derive);
    } catch (const UsageError & e) {
        return comm::is_root() ? cli::usage_error(*cli::command(command), e.what()) : 2;
    } catch (const InputError & e) {
        if (comm::is_root()) std::cerr << "mallard " << command << ": the case's input: " << e.what() << "\n";
    } catch (const std::exception & e) {
        if (comm::is_root()) std::cerr << "mallard " << command << ": " << e.what() << "\n";
    }
    return 1;
}

int post_main(const cli::Args & args, const bool derive) {
    int argc = args.argc;
    char ** argv = args.argv;
    comm::Session session(argc, argv);
    binding::select_device(argc, argv);
    Kokkos::initialize(argc, argv);
    const int status = post_command(args, derive);
    // A rank that failed alone would leave the others waiting in a collective
    if (status == 1 && comm::size() > 1) comm::abort(1);
    Kokkos::finalize();
    return status;
}

} // namespace

int post_extract(const cli::Args & args) { return post_command(args, false); }

int post_derive(const cli::Args & args) { return post_command(args, true); }

int extract_main(const cli::Args & args) { return post_main(args, false); }

int derive_main(const cli::Args & args) { return post_main(args, true); }

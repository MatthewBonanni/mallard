/**
 * @file restart_changes.cpp
 * @brief The classification of the keys a restarted run changes (restart_changes.h).
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 */

#include "restart_changes.h"

#include <algorithm>
#include <array>
#include <fstream>
#include <map>
#include <stdexcept>
#include <string_view>
#include <utility>

#include "data_writer.h"
#include "json.h"
#include "provenance.h"

namespace restart_changes {

namespace {

struct Leaf {
    std::string catalog;  // the key as the catalog names it: boundaries.type
    std::string text;     // its value as TOML, on one line
};

using Leaves = std::map<std::string, Leaf>;

/** @brief A value as TOML, on one line. */
std::string text_of(const Input & v) {
    if (v.is_boolean()) return input_detail::toml_text(v.as_boolean());
    if (v.is_integer()) return input_detail::toml_text(v.as_integer());
    if (v.is_floating()) return input_detail::toml_text(v.as_floating());
    if (v.is_string()) return input_detail::toml_text(v.as_string());
    if (v.is_array()) {
        std::string t;
        for (const Input & item : v.as_array()) t += (t.empty() ? "" : ", ") + text_of(item);
        return "[" + t + "]";
    }
    if (v.is_table()) {
        std::string t;
        for (const auto & [key, item] : v.as_table()) t += (t.empty() ? "" : ", ") + key + " = " + text_of(item);
        return "{" + t + "}";
    }
    return "?";  // dates and times: no key takes one
}

/** @brief The name of an entry of an array of tables, or empty if it has none. */
std::string entry_name(const Input & entry) {
    for (const auto & [key, value] : entry.as_table()) {
        if (key == "name" && value.is_string()) return value.as_string();
    }
    return "";
}

void flatten(const Input & v, const std::string & key, const std::string & catalog, Leaves & out) {
    const auto join = [](const std::string & a, const std::string & b) { return a.empty() ? b : a + "." + b; };
    if (v.is_table()) {
        for (const auto & [k, sub] : v.as_table()) flatten(sub, join(key, k), join(catalog, k), out);
        return;
    }
    if (v.is_array() && !v.as_array().empty() &&
        std::all_of(v.as_array().begin(), v.as_array().end(), [](const Input & e) { return e.is_table(); })) {
        const auto & entries = v.as_array();
        const bool named = std::all_of(entries.begin(), entries.end(),
                                       [](const Input & e) { return !entry_name(e).empty(); });
        for (size_t i = 0; i < entries.size(); i++) {
            const std::string at = named ? "[name = \"" + entry_name(entries[i]) + "\"]" : "[" + std::to_string(i) + "]";
            flatten(entries[i], key + at, catalog, out);
        }
        return;
    }
    out[key] = {catalog, text_of(v)};
}

bool starts_with_any(const std::string & s, std::initializer_list<std::string_view> prefixes) {
    return std::any_of(prefixes.begin(), prefixes.end(), [&](std::string_view p) { return s.starts_with(p); });
}

/** @brief A catalog path at or under a table: "statistics.fields" is under "statistics". */
bool under(const std::string & catalog, std::string_view table) {
    return catalog == table || (catalog.starts_with(table) && catalog.size() > table.size() && catalog[table.size()] == '.');
}

// Variables of the cells that hold running averages, and those that only seed a run (Solver::apply_restart)
bool is_average(const std::string & name) {
    return starts_with_any(name, {"MEAN_", "COV_", "PHASE_", "BATCH_"});
}
bool is_seed(const std::string & name) { return name == "CHEM_H" || name == "P_MAX"; }

std::string list(const std::vector<std::string> & names) {
    std::string out;
    for (const std::string & n : names) out += (out.empty() ? "" : ", ") + n;
    return out.empty() ? "none" : out;
}

struct Context {
    bool averages = false;   // the file holds running averages (statistics, spectra, phase, conditional)
    bool particles = false;  // the file holds particles
    bool faces = false;      // the file holds characteristic boundary states
    bool reset = false;      // the case sets statistics.reset = true
    std::string variables;   // why the file's variables are not the case's, or empty
};

constexpr std::string_view RESET =
    "statistics.reset: the running averages (means, covariances, batches, spectra, phase and conditional "
    "averages), the probes' averages and spectra and the convergence monitor start afresh at the restart's time";
constexpr std::string_view MESH =
    "the restart file holds the cells of the mesh it was written on, in that mesh's order; to start from it on "
    "another mesh, use [initialize] type = \"remap\"";

std::pair<Effect, std::string> effect_of(const std::string & key, const Leaf & leaf, const Leaves & then,
                                         const Leaves & now, const Context & c) {
    const std::string & k = leaf.catalog;
    if (under(k, "mesh")) {
        if (k == "mesh.cell_order") return {Effect::ACCEPTED, "the file holds the cells in the mesh file's order"};
        if (k == "mesh.curved" || k == "mesh.curved_geometry") return {Effect::ACCEPTED, ""};
        return {Effect::INCOMPATIBLE, std::string(MESH)};
    }
    if (k == "physics.units") {
        return {Effect::INCOMPATIBLE, "the restart file's values are in the units of the run that wrote it"};
    }
    if (under(k, "physics") && !c.variables.empty()) return {Effect::INCOMPATIBLE, c.variables};
    if (under(k, "statistics")) {
        if (c.reset) return {Effect::RESETS, std::string(RESET)};
        if (starts_with_any(k, {"statistics.fields", "statistics.products", "statistics.batch", "statistics.spectra",
                                "statistics.phase", "statistics.conditional"})) {
            if (!c.averages) return {Effect::ACCEPTED, "the restart file holds no averages: they start at its time"};
            if (!now.contains(key)) return {Effect::ACCEPTED, "the restart file's averages of it are not continued"};
            return {Effect::INCOMPATIBLE, "the restart file holds the averages of the statistics it was written with, "
                                          "which a run continues; set statistics.reset = true to start them afresh"};
        }
        return {Effect::ACCEPTED, ""};
    }
    if (under(k, "particles")) {
        if (!c.particles) return {Effect::ACCEPTED, "the restart file holds no particles"};
        return {Effect::RESETS, "the restart file's particles continue where they are; what the change adds starts "
                                "afresh: velocities and temperatures at the gas's, new quantities at their initial "
                                "values"};
    }
    if (k == "boundaries.type" && c.faces) {
        return {Effect::RESETS, "the characteristic boundaries' wave states start afresh from the solution (all of "
                                "them, when one boundary's faces have none in the file)"};
    }
    if (under(k, "probes")) {
        const std::string entry = key.substr(0, key.find("]") + 1);
        const auto averaged = [&](const Leaves & in) {
            const auto a = in.find(entry + ".average");
            const bool spectra = std::any_of(in.begin(), in.end(), [&](const auto & l) {
                return l.first.starts_with(entry + ".spectra") && l.second.text != "false";
            });
            return (a != in.end() && a->second.text == "true") || spectra;
        };
        if (averaged(then) || averaged(now)) {
            return {Effect::RESETS, "this probe's running average and spectra start afresh at the restart's time"};
        }
        return {Effect::ACCEPTED, ""};
    }
    if (k == "triggers.name") {
        return {Effect::RESETS, "a trigger keeps its firings and state by name: one of a new name starts with none"};
    }
    return {Effect::ACCEPTED, ""};
}

}  // namespace

const char * name(const Effect effect) {
    switch (effect) {
        case Effect::RESETS: return "resets";
        case Effect::INCOMPATIBLE: return "incompatible";
        case Effect::ACCEPTED: break;
    }
    return "accepted";
}

Restart read(const std::string & file) {
    std::ifstream in(file, std::ios::binary);
    if (!in) throw std::runtime_error("cannot open restart file " + file);
    const RestartHeader h = read_restart_header(in, file);
    Restart r;
    r.file = file;
    r.step = h.step;
    r.t = h.t;
    r.real_size = h.real_size;
    r.variables = h.names;
    for (const auto & [key, value] : h.attributes) r.attributes.push_back(key);
    for (const auto & [key, values] : h.arrays) r.arrays.push_back(key);
    r.faces = h.n_faces;
    if (!h.provenance.empty()) {
        r.run_id = provenance::find_string(h.provenance, "run", "run_id");
        const json::Value block = json::parse(h.provenance);
        if (const json::Value * build = block.find("build"); build && build->find("dim")) {
            r.dim = static_cast<int>(build->find("dim")->number);
        }
        if (const json::Value * input = block.find("input"); input && input->find("resolved")) {
            r.resolved = input->find("resolved")->string;
        }
    }
    return r;
}

std::vector<Change> classify(const Restart & restart, const Input & case_resolved,
                             const std::vector<std::string> & case_variables, const bool case_complete) {
    std::vector<Change> changes;
    if (restart.dim != 0 && restart.dim != N_DIM) {
        changes.push_back({"build.dimension", std::to_string(restart.dim), std::to_string(N_DIM), Effect::INCOMPATIBLE,
                           "the restart file is of a " + std::to_string(restart.dim) + "D run; this is mallard" +
                               std::to_string(N_DIM) + "d"});
    }
    if (restart.real_size != 0 && restart.real_size != sizeof(rtype)) {
        const auto precision = [](uint32_t size) { return std::string(size == 8 ? "double" : "single"); };
        changes.push_back({"build.precision", precision(restart.real_size), precision(sizeof(rtype)),
                           Effect::INCOMPATIBLE, "the restart file's values are in another floating-point precision"});
    }

    Context c;
    c.averages = std::any_of(restart.variables.begin(), restart.variables.end(), is_average) ||
                 std::any_of(restart.attributes.begin(), restart.attributes.end(), [](const std::string & a) {
                     return starts_with_any(a, {"STATISTICS_", "SPECTRA_", "PHASE_", "CONDITIONAL_"});
                 });
    c.particles = std::find(restart.arrays.begin(), restart.arrays.end(), "PARTICLE_ID") != restart.arrays.end();
    c.faces = restart.faces > 0;
    if (!case_variables.empty()) {
        std::vector<std::string> missing, extra;
        for (const std::string & v : case_variables) {
            if (is_average(v) || is_seed(v)) continue;
            if (std::find(restart.variables.begin(), restart.variables.end(), v) == restart.variables.end()) {
                missing.push_back(v);
            }
        }
        for (const std::string & v : restart.variables) {
            if (is_average(v) || is_seed(v)) continue;
            if (std::find(case_variables.begin(), case_variables.end(), v) == case_variables.end()) extra.push_back(v);
        }
        if (!missing.empty() || !extra.empty()) {
            c.variables = "the restart file's variables are not this case's: it has no " + list(missing) +
                          ", and this case has no " + list(extra);
            changes.push_back({"variables", list(restart.variables), list(case_variables), Effect::INCOMPATIBLE,
                               c.variables});
        }
    }
    if (restart.resolved.empty()) return changes;

    Leaves then, now;
    const Input before = parse_input_string(restart.resolved, restart.file + " (its input)");
    flatten(before, "", "", then);
    flatten(case_resolved, "", "", now);
    for (const auto & [key, leaf] : now) {
        if (leaf.catalog == "statistics.reset" && leaf.text == "true") c.reset = true;
    }
    std::map<std::string, std::pair<std::string, std::string>> differ;
    for (const auto & [key, leaf] : then) {
        const auto other = now.find(key);
        if (other == now.end() && !case_complete) continue;  // a key the setup did not get to
        if (other == now.end() || other->second.text != leaf.text) {
            differ[key] = {leaf.text, other == now.end() ? "" : other->second.text};
        }
    }
    for (const auto & [key, leaf] : now) {
        if (!then.contains(key)) differ[key] = {"", leaf.text};
    }
    for (const auto & [key, values] : differ) {
        const Leaf & leaf = now.contains(key) ? now.at(key) : then.at(key);
        if (under(leaf.catalog, "initialize")) continue;
        auto [effect, detail] = effect_of(key, leaf, then, now, c);
        changes.push_back({key, values.first, values.second, effect, detail});
    }
    // Set in both: a reset all the same
    if (c.reset && c.averages && !differ.contains("statistics.reset")) {
        changes.push_back({"statistics.reset", "true", "true", Effect::RESETS, std::string(RESET)});
    }
    return changes;
}

}  // namespace restart_changes

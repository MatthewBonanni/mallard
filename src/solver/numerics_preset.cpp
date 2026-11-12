/**
 * @file numerics_preset.cpp
 * @author Matthew Bonanni (mbonanni001@gmail.com)
 * @brief Numerics presets. Their choices and the validation behind them:
 *        docs/guide/numerics.md#presets.
 * @version 0.1
 * @date 2026-10-16
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 *
 */

#include "numerics_preset.h"

#include <unordered_map>

#include "input.h"

namespace numerics_preset {

namespace {

struct Entry {
    std::vector<std::string> path;
    Input value;
};

struct Preset {
    std::string name;
    std::vector<Entry> entries;
};

const std::vector<std::string> RECONSTRUCTION = {"numerics", "face_reconstruction"};

std::vector<std::string> reconstruction(const std::string & key) {
    std::vector<std::string> path = RECONSTRUCTION;
    path.push_back(key);
    return path;
}

const std::vector<Preset> & presets() {
    static const std::vector<Preset> list = {
        {"high_order", {
            {reconstruction("type"), Input("TENO")},
            {reconstruction("order"), Input(5)},
            {{"numerics", "riemann_solver"}, Input("HLLC")},
            {{"numerics", "convective_flux"}, Input("riemann")},
            {{"numerics", "time_integrator"}, Input("SSPRK3")},
            {{"run", "cfl"}, Input(1.0)},
        }},
        {"second_order", {
            {reconstruction("type"), Input("MUSCL")},
            {reconstruction("limiter"), Input("venkatakrishnan")},
            {{"numerics", "riemann_solver"}, Input("RHLL")},
            {{"numerics", "convective_flux"}, Input("riemann")},
            {{"numerics", "time_integrator"}, Input("SSPRK3")},
            {{"run", "cfl"}, Input(0.5)},
        }},
        {"first_order", {
            {reconstruction("type"), Input("FO")},
            {{"numerics", "riemann_solver"}, Input("HLL")},
            {{"numerics", "convective_flux"}, Input("riemann")},
            {{"numerics", "time_integrator"}, Input("SSPRK3")},
            {{"run", "cfl"}, Input(0.5)},
        }},
        {"les", {
            {reconstruction("type"), Input("MUSCL")},
            {reconstruction("limiter"), Input("none")},
            {{"numerics", "riemann_solver"}, Input("HLLC")},
            {{"numerics", "convective_flux"}, Input("hybrid")},
            {{"numerics", "time_integrator"}, Input("SSPRK3")},
            {{"run", "cfl"}, Input(0.8)},
        }},
        {"reacting", {
            {reconstruction("type"), Input("MUSCL")},
            {reconstruction("limiter"), Input("venkatakrishnan")},
            {{"numerics", "riemann_solver"}, Input("HLLC")},
            {{"numerics", "convective_flux"}, Input("riemann")},
            {{"numerics", "time_integrator"}, Input("SSPRK3")},
            {{"run", "cfl"}, Input(0.35)},
        }},
    };
    return list;
}

std::string join(const std::vector<std::string> & path) {
    std::string key;
    for (const auto & part : path) key += (key.empty() ? "" : ".") + part;
    return key;
}

const Input * lookup(const Input & input, const std::vector<std::string> & path) {
    const Input * v = &input;
    for (const auto & part : path) {
        if (!v->is_table() || !v->contains(part)) return nullptr;
        v = &v->at(part);
    }
    return v;
}

Input & table_at(Input & input, const std::vector<std::string> & path) {
    Input * v = &input;
    for (const auto & part : path) {
        if (!v->contains(part)) v->as_table().emplace(part, Input::table_type{});
        v = &v->at(part);
        if (!v->is_table()) throw InputError(join(path) + " must be a table.");
    }
    return *v;
}

} // namespace

std::vector<std::string> names() {
    std::vector<std::string> list;
    for (const auto & preset : presets()) list.push_back(preset.name);
    return list;
}

std::string Resolution::summary(const ResolvedInput & resolved) const {
    std::string overrides;
    for (const auto & s : settings) {
        if (s.from_preset) continue;
        const KeyRecord * record = resolved.find(s.key);
        overrides += (overrides.empty() ? "" : ", ") + s.key + " = " + (record ? record->value : "");
    }
    std::string note = is_default ? "the default since 0.10.0, first order before: the input sets no preset and no "
                                    "[numerics.face_reconstruction]" : "";
    if (!overrides.empty()) note += (note.empty() ? "" : "; ") + std::string("the input sets ") + overrides;
    return note.empty() ? name : name + " (" + note + ")";
}

Resolution apply(Input & input) {
    Resolution resolution;
    const Input * name = lookup(input, {"numerics", "preset"});
    if (!name && lookup(input, RECONSTRUCTION)) return resolution;
    std::unordered_map<std::string, const Preset *> by_name;
    for (const auto & preset : presets()) by_name.emplace(preset.name, &preset);
    const Preset * found = by_name.at(DEFAULT);
    if (name) {
        if (!name->is_string()) throw InputError(toml::format_error("numerics.preset must be a string", *name, "here"));
        found = find_option(by_name, "numerics.preset", name->as_string(), line_of(*name))->second;
    } else {
        resolution.is_default = true;
    }
    const Preset & preset = *found;
    resolution.name = preset.name;

    const Input * type = lookup(input, reconstruction("type"));
    std::string preset_type;
    for (const auto & entry : preset.entries) {
        if (entry.path == reconstruction("type")) preset_type = entry.value.as_string();
    }
    const bool own_reconstruction = !type || (type->is_string() && type->as_string() == preset_type);
    for (const auto & entry : preset.entries) {
        const std::string key = join(entry.path);
        const std::vector<std::string> parent(entry.path.begin(), entry.path.end() - 1);
        const bool reconstruction_option = parent == RECONSTRUCTION && entry.path.back() != "type";
        if (reconstruction_option && !own_reconstruction) continue;
        const bool set = lookup(input, entry.path);
        if (key == "run.cfl" && !set && lookup(input, {"run", "dt"})) {
            resolution.settings.push_back({"run.dt", false});
            continue;
        }
        if (!set) table_at(input, parent).as_table().emplace(entry.path.back(), entry.value);
        resolution.settings.push_back({key, !set});
    }
    return resolution;
}

} // namespace numerics_preset

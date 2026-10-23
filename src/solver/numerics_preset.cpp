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
#include "log.h"

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
        {"robust", {
            {reconstruction("type"), Input("MUSCL")},
            {reconstruction("limiter"), Input("venkatakrishnan")},
            {{"numerics", "riemann_solver"}, Input("RHLL")},
            {{"numerics", "convective_flux"}, Input("riemann")},
            {{"numerics", "time_integrator"}, Input("SSPRK3")},
            {{"run", "cfl"}, Input(0.5)},
        }},
        {"accurate", {
            {reconstruction("type"), Input("TENO")},
            {reconstruction("order"), Input(5)},
            {{"numerics", "riemann_solver"}, Input("HLLC")},
            {{"numerics", "convective_flux"}, Input("riemann")},
            {{"numerics", "time_integrator"}, Input("SSPRK3")},
            {{"run", "cfl"}, Input(1.0)},
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

std::string text(const Input & v) {
    if (v.is_string()) return v.as_string();
    if (v.is_floating()) return logging::real(v.as_floating());
    return toml::format(v);
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

std::string Resolution::summary() const {
    std::string overrides;
    for (const auto & s : settings) {
        if (!s.from_preset) overrides += (overrides.empty() ? "" : ", ") + s.key + " = " + s.value;
    }
    return overrides.empty() ? name : name + " (the input sets " + overrides + ")";
}

Resolution apply(Input & input) {
    Resolution resolution;
    const Input * name = lookup(input, {"numerics", "preset"});
    if (!name) return resolution;
    std::unordered_map<std::string, const Preset *> by_name;
    for (const auto & preset : presets()) by_name.emplace(preset.name, &preset);
    if (!name->is_string()) throw InputError(toml::format_error("numerics.preset must be a string", *name, "here"));
    const auto it = find_option(by_name, "numerics.preset", name->as_string());
    const Preset & preset = *it->second;
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
        const Input * set = lookup(input, entry.path);
        if (key == "run.cfl" && !set) {
            const std::vector<std::string> dt_path = {"run", "dt"};
            if (const Input * dt = lookup(input, dt_path)) {
                resolution.settings.push_back({join(dt_path), text(*dt), false});
                continue;
            }
        }
        if (set) {
            resolution.settings.push_back({key, text(*set), false});
        } else {
            table_at(input, parent).as_table().emplace(entry.path.back(), entry.value);
            resolution.settings.push_back({key, text(entry.value), true});
        }
    }
    return resolution;
}

} // namespace numerics_preset

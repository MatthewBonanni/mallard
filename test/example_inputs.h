/**
 * @file example_inputs.h
 * @author Matthew Bonanni (mbonanni001@gmail.com)
 * @brief The inputs of examples/ with their numerics replaced by a preset,
 *        reduced to a few steps on a small mesh that needs no generated files.
 * @version 0.1
 * @date 2026-10-16
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 *
 */

#ifndef EXAMPLE_INPUTS_H
#define EXAMPLE_INPUTS_H

#include <algorithm>
#include <array>
#include <filesystem>
#include <ostream>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

#include "data_paths.h"
#include "input.h"

#include "test_fixtures.h"

struct ExampleCase {
    std::string file;    ///< Under examples/
    std::string preset;  ///< Its closest numerics preset
    int dim;             ///< Spatial dimension of the build it needs
    /// Replaces an [initialize] that reads a generated file (restart, remap or
    /// data); empty, for a restart or remap: the state of the first boundary
    /// with p and T (mixtures) or the reference state (perfect gas), at rest
    std::string init;
    /// File meshes: (zone of the replacement mesh, example boundary it takes)
    std::vector<std::array<std::string, 2>> zones;
    /// File meshes: the replacement [mesh] table, a file in test/data/meshes;
    /// empty: test/data/meshes/cyl{2,3}d.msh
    std::string mesh;
};

/** @brief Every solver example of this build's dimension (numerics_preset_test.cpp). */
std::vector<ExampleCase> examples_of_this_build();

inline void PrintTo(const ExampleCase & c, std::ostream * os) { *os << c.file << " (" << c.preset << ")"; }

inline std::filesystem::path example_path(const std::string & file) {
    return std::filesystem::path(source_dir()) / "examples" / file;
}

inline void collect_strings(const Input & v, std::set<std::string> & out) {
    if (v.is_string()) {
        out.insert(v.as_string());
    } else if (v.is_array()) {
        for (const auto & e : v.as_array()) collect_strings(e, out);
    } else if (v.is_table()) {
        for (const auto & [key, e] : v.as_table().entries()) collect_strings(e, out);
    }
}

/** @brief Removes the named regions that only the removed outputs named, which would be unused keys. */
inline void erase_unreferenced_regions(Input & input) {
    auto & table = input.as_table().entries();
    const auto it = table.find("regions");
    if (it == table.end() || !it->second.is_table()) return;
    auto & regions = it->second.as_table().entries();
    std::set<std::string> named;
    for (const auto & [key, v] : table) {
        if (key != "regions") collect_strings(v, named);
    }
    for (bool grew = true; grew;) {
        grew = false;
        for (const auto & [name, v] : regions) {
            if (!named.count(name)) continue;
            const size_t before = named.size();
            collect_strings(v, named);
            grew = grew || named.size() != before;
        }
    }
    std::erase_if(regions, [&](const auto & r) { return !named.count(r.first); });
    if (regions.empty()) table.erase(it);
}

/** @brief The replacement mesh's zones that took the example's boundary `name`; an error if none. */
inline std::vector<std::string> replacement_zones(const ExampleCase & c, const std::string & name) {
    std::vector<std::string> zones;
    for (const auto & [zone, source] : c.zones) {
        if (source == name) zones.push_back(zone);
    }
    if (zones.empty()) throw std::runtime_error(c.file + ": no zone of the replacement mesh takes boundary " + name);
    return zones;
}

/**
 * @brief Names the replacement mesh's zones where the input names the example's
 *        boundaries as zones (`zone`), boundaries (`boundary`) or regions of the
 *        mesh (`region`, `regions`).
 */
inline void rename_zones(const ExampleCase & c, const std::set<std::string> & boundaries, Input & v,
                         const std::string & key = "") {
    if (v.is_table()) {
        for (auto & [k, e] : v.as_table().entries()) rename_zones(c, boundaries, e, k);
    } else if (v.is_array()) {
        if (key == "boundary") {
            Input::array_type renamed;
            for (const auto & b : v.as_array()) {
                if (!b.is_string() || !boundaries.count(b.as_string())) {
                    renamed.push_back(b);
                    continue;
                }
                for (const auto & zone : replacement_zones(c, b.as_string())) renamed.push_back(zone);
            }
            v = renamed;
        } else {
            for (auto & e : v.as_array()) rename_zones(c, boundaries, e, key);
        }
    } else if (v.is_string() && boundaries.count(v.as_string()) &&
               (key == "zone" || key == "boundary" || key == "region" || key == "regions")) {
        const auto zones = replacement_zones(c, v.as_string());
        if (zones.size() == 1) {
            v = zones[0];
        } else if (key == "boundary") {
            v = Input::array_type(zones.begin(), zones.end());
        } else {
            throw std::runtime_error(c.file + ": " + key + " names boundary " + v.as_string() + ", which several zones take");
        }
    }
}

/**
 * @brief The example's input with `[numerics] preset` in place of its scheme
 *        keys (riemann_solver, time_integrator, convective_flux and the
 *        reconstruction tables), no `cfl` or `dt`, `n_steps` steps, at most
 *        `max_cells` cells per direction of a generated mesh, and paths made
 *        absolute. Without `outputs`, no file output besides `[integrals]`;
 *        with it, every output of the example, writing in the working
 *        directory. On a replacement mesh, boundary zones are those it takes.
 */
inline Input example_under_preset(const ExampleCase & c, uint64_t n_steps, int64_t max_cells, bool outputs = false) {
    const std::filesystem::path path = example_path(c.file);
    const std::filesystem::path dir = path.parent_path();
    Input input = parse_input(path.string());
    auto & table = input.as_table();

    auto & numerics = input.at("numerics").as_table();
    for (const char * key : {"riemann_solver", "time_integrator", "convective_flux", "face_reconstruction", "hybrid"}) {
        numerics.erase(key);
    }
    numerics["preset"] = c.preset;
    numerics["check_nan"] = true;

    table["run"] = Input::table_type{{"n_steps", static_cast<Input::integer_type>(n_steps)}};
    if (outputs) {
        if (!input.contains("output")) table["output"] = Input::table_type{};
        table["output"].as_table()["check_interval"] = 1000000;
    } else {
        table["output"] = Input::table_type{{"check_interval", 1000000}};
        for (const char * key : {"write_data", "probes", "statistics", "forces", "triggers"}) table.erase(key);
        erase_unreferenced_regions(input);
    }

    auto & mesh = input.at("mesh").as_table();
    if (mesh.at("type").as_string() == "file") {
        const std::string file =
            source_dir() + "/test/data/meshes/" + (c.dim == 3 ? "cyl3d.msh" : "cyl2d.msh");
        table["mesh"] =
            c.mesh.empty() ? Input(Input::table_type{{"type", "file"}, {"filename", file}}) : parse_toml(c.mesh);
        if (table["mesh"].contains("filename")) {
            auto & filename = table["mesh"].as_table()["filename"];
            filename = (std::filesystem::path(source_dir()) / "test/data/meshes" / filename.as_string()).string();
        }
        table.erase("periodic");
        Input::array_type boundaries;
        for (const auto & [zone, source] : c.zones) {
            for (const auto & b : input.at("boundaries").as_array()) {
                if (b.at("name").as_string() != source) continue;
                Input copy = b;
                copy.as_table()["name"] = zone;
                copy.as_table().erase("where");
                boundaries.push_back(copy);
                break;
            }
        }
        std::set<std::string> names;
        for (const auto & b : input.at("boundaries").as_array()) names.insert(b.at("name").as_string());
        if (input.contains("regions")) {
            for (const auto & [name, region] : input.at("regions").as_table().entries()) names.erase(name);
        }
        for (auto & [key, v] : table.entries()) {
            if (key != "boundaries") rename_zones(c, names, v);
        }
        table["boundaries"] = boundaries;
    } else {
        for (const char * key : {"Nx", "Ny", "Nz"}) {
            if (mesh.count(key)) mesh[key] = std::min<int64_t>(mesh.at(key).as_integer(), max_cells);
        }
    }

    if (!input.contains("boundaries")) table["boundaries"] = Input::array_type{};
    auto & physics = input.at("physics").as_table();
    if (physics.count("mechanism")) physics["mechanism"] = (dir / physics.at("mechanism").as_string()).string();
    for (auto & b : input.at("boundaries").as_array()) {
        if (b.contains("data")) {
            for (auto & d : b.as_table().at("data").as_array()) {
                d.as_table()["file"] = (dir / d.at("file").as_string()).string();
            }
        }
        if (b.contains("turbulence") && b.at("turbulence").contains("profile")) {
            auto & turbulence = b.as_table().at("turbulence").as_table();
            turbulence["profile"] = (dir / turbulence.at("profile").as_string()).string();
        }
    }

    if (input.contains("les") && input.at("les").contains("combustion") &&
        input.at("les").at("combustion").contains("flame_table")) {
        auto & combustion = table["les"].as_table().at("combustion").as_table();
        combustion["flame_table"] = (dir / combustion.at("flame_table").as_string()).string();
    }

    if (input.contains("initialize") && input.at("initialize").contains("data")) {
        for (auto & d : table["initialize"].as_table().at("data").as_array()) {
            d.as_table()["file"] = (dir / d.at("file").as_string()).string();
        }
    }

    if (!c.init.empty()) {
        table["initialize"] = parse_toml(c.init);
    } else if (const auto type = find_value_or<std::string>(input, "initialize", "type", "constant");
               type == "restart" || type == "remap") {
        if (physics.count("mechanism")) {
            for (const auto & b : input.at("boundaries").as_array()) {
                if (!b.contains("p") || !b.contains("T") || !(b.contains("X") || b.contains("Y"))) continue;
                auto expression = [](const Input & v) {
                    return v.is_string() ? v : Input(toml::format(v));
                };
                Input::table_type init{{"type", "analytical"}, {"p", expression(b.at("p"))}, {"T", expression(b.at("T"))}};
                init[b.contains("X") ? "X" : "Y"] = b.at(b.contains("X") ? "X" : "Y");
                init["u"] = Input::array_type(static_cast<size_t>(c.dim), Input("0.0"));
                table["initialize"] = init;
                break;
            }
        } else {
            Input::table_type init{{"type", "constant"}, {"p", physics.at("p_ref")}, {"T", physics.at("T_ref")}};
            init["u"] = Input::array_type(static_cast<size_t>(c.dim), Input(0.0));
            table["initialize"] = init;
        }
    }
    return input;
}

#endif // EXAMPLE_INPUTS_H

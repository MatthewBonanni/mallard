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
#include <string>
#include <vector>

#include <toml.hpp>

#include "test_fixtures.h"

struct ExampleCase {
    std::string file;    ///< Under examples/
    std::string preset;  ///< Its closest numerics preset
    int dim;             ///< Spatial dimension of the build it needs
    /// Replaces an [initialize] that restarts from a generated file; empty:
    /// the state of the first boundary with p and T (mixtures) or the
    /// reference state (perfect gas), at rest
    std::string init;
    /// File meshes: (zone of test/data/meshes/cyl{2,3}d.msh, example boundary it takes)
    std::vector<std::array<std::string, 2>> zones;
};

inline void PrintTo(const ExampleCase & c, std::ostream * os) { *os << c.file << " (" << c.preset << ")"; }

inline std::filesystem::path example_path(const std::string & file) {
    return std::filesystem::path(MALLARD_SOURCE_DIR) / "examples" / file;
}

/**
 * @brief The example's input with `[numerics] preset` in place of its scheme
 *        keys (riemann_solver, time_integrator, convective_flux and the
 *        reconstruction tables), no `cfl` or `dt`, `n_steps` steps, at most
 *        `max_cells` cells per direction of a generated mesh, no file output
 *        besides `[integrals]`, and paths made absolute.
 */
inline toml::value example_under_preset(const ExampleCase & c, uint64_t n_steps, int64_t max_cells) {
    const std::filesystem::path path = example_path(c.file);
    const std::filesystem::path dir = path.parent_path();
    toml::value input = toml::parse(path.string());
    auto & table = input.as_table();

    auto & numerics = input.at("numerics").as_table();
    for (const char * key : {"riemann_solver", "time_integrator", "convective_flux", "face_reconstruction", "hybrid"}) {
        numerics.erase(key);
    }
    numerics["preset"] = c.preset;
    numerics["check_nan"] = true;

    table["run"] = toml::table{{"n_steps", static_cast<toml::value::integer_type>(n_steps)}};
    table["output"] = toml::table{{"check_interval", 1000000}};
    for (const char * key : {"write_data", "probes", "statistics", "forces"}) table.erase(key);

    auto & mesh = input.at("mesh").as_table();
    if (mesh.at("type").as_string() == "file") {
        const std::string name = c.dim == 3 ? "cyl3d.msh" : "cyl2d.msh";
        table["mesh"] = toml::table{{"type", "file"}, {"filename", std::string(MALLARD_SOURCE_DIR) + "/test/data/meshes/" + name}};
        table.erase("periodic");
        toml::array boundaries;
        for (const auto & [zone, source] : c.zones) {
            for (const auto & b : input.at("boundaries").as_array()) {
                if (b.at("name").as_string() != source) continue;
                toml::value copy = b;
                copy.as_table()["name"] = zone;
                boundaries.push_back(copy);
            }
        }
        table["boundaries"] = boundaries;
    } else {
        for (const char * key : {"Nx", "Ny", "Nz"}) {
            if (mesh.count(key)) mesh[key] = std::min<int64_t>(mesh.at(key).as_integer(), max_cells);
        }
    }

    if (!input.contains("boundaries")) table["boundaries"] = toml::array{};
    auto & physics = input.at("physics").as_table();
    if (physics.count("mechanism")) physics["mechanism"] = (dir / physics.at("mechanism").as_string()).string();
    for (auto & b : input.at("boundaries").as_array()) {
        if (b.contains("turbulence") && b.at("turbulence").contains("profile")) {
            auto & turbulence = b.as_table().at("turbulence").as_table();
            turbulence["profile"] = (dir / turbulence.at("profile").as_string()).string();
        }
    }

    if (toml::find_or<std::string>(input, "initialize", "type", "constant") == "restart") {
        if (!c.init.empty()) {
            table["initialize"] = parse_toml(c.init);
        } else if (physics.count("mechanism")) {
            for (const auto & b : input.at("boundaries").as_array()) {
                if (!b.contains("p") || !b.contains("T") || !(b.contains("X") || b.contains("Y"))) continue;
                auto expression = [](const toml::value & v) { return v.is_string() ? v : toml::value(toml::format(v)); };
                toml::table init{{"type", "analytical"}, {"p", expression(b.at("p"))}, {"T", expression(b.at("T"))}};
                init[b.contains("X") ? "X" : "Y"] = b.at(b.contains("X") ? "X" : "Y");
                init["u"] = toml::array(static_cast<size_t>(c.dim), toml::value("0.0"));
                table["initialize"] = init;
                break;
            }
        } else {
            toml::table init{{"type", "constant"}, {"p", physics.at("p_ref")}, {"T", physics.at("T_ref")}};
            init["u"] = toml::array(static_cast<size_t>(c.dim), toml::value(0.0));
            table["initialize"] = init;
        }
    }
    return input;
}

#endif // EXAMPLE_INPUTS_H

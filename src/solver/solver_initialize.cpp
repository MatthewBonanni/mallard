/**
 * @file solver_initialize.cpp
 * @author Matthew Bonanni (mbonanni001@gmail.com)
 * @brief Implementation of solution initialization methods for the Solver class.
 * @version 0.2
 * @date 2024-01-11
 *
 * @copyright Copyright (c) 2024 Matthew Bonanni
 *
 */

#include "solver.h"

#include "input.h"

#include <algorithm>
#include <string>
#include <unordered_map>
#include <vector>

#include <Kokkos_Core.hpp>

#include "initial_state.h"
#include "remap.h"

enum class InitType {
    CONSTANT,
    ANALYTICAL,
    RESTART,
    REMAP
};

static const std::unordered_map<std::string, InitType> INIT_TYPES = {
    {"constant", InitType::CONSTANT},
    {"analytical", InitType::ANALYTICAL},
    {"restart", InitType::RESTART},
    {"remap", InitType::REMAP}
};

void Solver::init_solution() {
    if (!input.contains("initialize")) {
        throw InputError({.key = "initialize",
                          .message = "the [initialize] table is missing; it sets the initial flow",
                          .fix = N_DIM == 3 ? "Add [initialize] with u = [ux, uy, uz], p and T (a uniform flow), or "
                                              "type = \"analytical\" for expressions of x, y, z"
                                            : "Add [initialize] with u = [ux, uy], p and T (a uniform flow), or "
                                              "type = \"analytical\" for expressions of x, y"});
    }
    const std::string type_str = find_value_or<std::string>(input, "initialize", "type", "constant");
    auto it = find_option(INIT_TYPES, "initialize.type", type_str, line_of(input.at("initialize"), "type"));
    initial_state = type_str;
    const bool from_file = it->second == InitType::RESTART || it->second == InitType::REMAP;
    const bool layered = InitialState::layered(input);
    if (it->second == InitType::CONSTANT && !layered) {
        init_solution_constant();
    } else if (it->second == InitType::RESTART) {
        init_solution_restart();
    } else if (it->second == InitType::REMAP) {
        init_solution_remap();
    }
    if (!from_file && !(it->second == InitType::CONSTANT && !layered)) {
        const InitialState state(input, *mesh, physics, mixture_model.get(), false);
        state.apply(h_conservatives, h_species, h_T_seed);
        if (layered) initial_state = state.summary();
    } else if (layered) {
        const InitialState state(input, *mesh, physics, mixture_model.get(), true);
        state.apply(h_conservatives, h_species, h_T_seed);
        initial_state += "; " + state.summary();
    }
    copy_host_to_device();
    if (!from_file) statistics.start(t);
    if (is_mixture() && (!from_file || layered)) init_temperature_seed();
    update_primitives();
    if (p_max.is_allocated()) update_p_max();
}

void Solver::init_solution_restart() {
    if (!input.at("initialize").contains("file")) {
        throw missing_key(input.at("initialize"), "initialize.file",
                          "Add file = \"...\": a restart file written by [[write_data]] with format = \"restart\"");
    }
    const std::string file = find_value<std::string>(input, "initialize", "file");
    // Restart files hold cells in global order, so any number of ranks can read
    // them; each rank reads only its local cells
    const bool distributed = mesh->n_global_cells > 0;
    RestartData restart = read_restart(file, distributed ? &mesh->h_global_cell_id : nullptr);
    const uint64_t n_expected = distributed ? mesh->n_global_cells : mesh->n_cells;
    if (restart.n_cells != n_expected) {
        throw InputError({.key = "initialize.file",
                          .line = line_of(input.at("initialize"), "file"),
                          .message = "restart file " + file + " has " + std::to_string(restart.n_cells) +
                                     " cells, but the mesh has " + std::to_string(n_expected),
                          .fix = "Restart on the mesh that wrote the file; to start from a solution on another mesh, "
                                 "use type = \"remap\"",
                          .docs = "docs/input.md#type--remap"});
    }
    apply_restart(restart, file, true);
    initial_state = "restart from " + file + " (step " + std::to_string(step) + ", t = " + logging::real(double(t)) + ")";
}

void Solver::init_solution_remap() {
    remap::Report report;
    const RestartData restart = remap::remap_restart(input, *mesh, mesh->axisymmetric, report);
    const std::string file = find_value<std::string>(input, "initialize", "file");
    apply_restart(restart, file, false);
    initial_state = "remap of " + file + " (step " + std::to_string(step) + ", t = " + logging::real(double(t)) +
                    "): " + remap::describe(report, restart.names);
}

void Solver::apply_restart(const RestartData & restart, const std::string & file, bool resume_output) {
    // Variables map by name: the flow block, then one RHOY_<name> per species
    // CHEM_H (last chemistry sub-step) only seeds the integrator: a reacting run
    // may start from a non-reacting one, and a non-reacting run ignores it
    std::vector<std::string> expected = restart_variables();
    const std::vector<std::string> averages = statistics.variables();
    auto is_average = [&](const std::string & name) {
        return std::find(averages.begin(), averages.end(), name) != averages.end();
    };
    for (const auto & name : restart.names) {
        // Averages carry over only into a run that keeps them
        if (name == "CHEM_H" || name == "P_MAX" || name.rfind("MEAN_", 0) == 0 || name.rfind("COV_", 0) == 0 ||
            name.rfind("PHASE_", 0) == 0) continue;
        if (std::find(expected.begin(), expected.end(), name) == expected.end()) {
            throw std::runtime_error("Restart file " + file + " has variable " + name + ", which this run does not " +
                                     (name.rfind("RHOY_", 0) == 0 ? "transport." : "know."));
        }
    }
    for (uint32_t v = 0; v < expected.size(); v++) {
        if (is_average(expected[v])) continue;
        const std::vector<rtype> * values = restart.find(expected[v]);
        if (expected[v] == "P_MAX") {
            for (uint32_t i_cell = 0; i_cell < mesh->n_cells; ++i_cell) {
                h_p_max(i_cell) = values ? (*values)[i_cell] : 0.0_r;
            }
            continue;
        }
        if (expected[v] == "CHEM_H") {
            for (uint32_t i_cell = 0; i_cell < mesh->n_cells; ++i_cell) {
                h_chem_h(i_cell) = values ? (*values)[i_cell] : 0.0_r;
            }
            continue;
        }
        if (values == nullptr) {
            throw std::runtime_error("Restart file " + file + " has no variable " + expected[v] + ".");
        }
        const uint32_t n_species = static_cast<uint32_t>(species_names.size());
        for (uint32_t i_cell = 0; i_cell < mesh->n_cells; ++i_cell) {
            if (v < N_CONSERVATIVE) {
                h_conservatives(i_cell, v) = (*values)[i_cell];
            } else if (v < N_CONSERVATIVE + n_species) {
                h_species(i_cell, v - N_CONSERVATIVE) = (*values)[i_cell];
            } else {
                h_T_seed(i_cell) = (*values)[i_cell];
            }
        }
    }
    step = restart.step;
    t = static_cast<rtype>(restart.t);
    statistics.restore(restart, file, t);
    if (step_controller.enabled()) step_controller.restore(restart.attributes);
    restore_characteristic_state(restart);
    if (!resume_output) return;
    for (auto & writer : data_writers) {
        const std::string name = writer->snapshot_attribute();
        writer->resume(step, t, name.empty() ? nullptr : restart.attribute(name));
    }
    if (checkpoint_writer) checkpoint_writer->resume(step, t);
    probes.resume(step, t);
}

void Solver::init_solution_constant() {
    const Input & init = input.at("initialize");
    for (const char * key : {"u", "p", "T"}) {
        if (!init.contains(key)) {
            Diagnostic d = missing_key(init, std::string("initialize.") + key).diagnostic;
            d.message = "missing; a uniform initial state (type = \"constant\") needs u, p and T";
            throw InputError(d);
        }
    }
    std::vector<rtype> u = find_real_vector(input, "initialize", "u", dim::velocity);
    if (u.size() != N_DIM) {
        throw InputError({.key = "initialize.u",
                          .line = line_of(init, "u"),
                          .message = "has " + std::to_string(u.size()) + " components, but this build is " +
                                     std::to_string(N_DIM) + "D",
                          .fix = N_DIM == 3 ? "Write u = [ux, uy, uz]" : "Write u = [ux, uy]"});
    }
    const rtype p = find_real(input, "initialize", "p", dim::pressure);
    const rtype T = find_real(input, "initialize", "T", dim::temperature);
    if (is_mixture()) {
        const std::vector<double> Y = mixture_model->mass_fractions(init, "initialize");
        rtype cons[N_CONSERVATIVE];
        std::vector<rtype> rhoY(species_names.size());
        mixture_model->conservatives(static_cast<double>(p), static_cast<double>(T), u.data(), Y, cons, rhoY.data());
        for (uint32_t i_cell = 0; i_cell < mesh->n_cells; ++i_cell) {
            FOR_I_CONSERVATIVE h_conservatives(i_cell, i) = cons[i];
            for (size_t k = 0; k < rhoY.size(); k++) h_species(i_cell, k) = rhoY[k];
        }
        return;
    }
    rtype W[N_CONSERVATIVE];
    W[0] = physics.get_density_from_pressure_temperature(p, T);
    FOR_I_DIM W[1 + i] = u[i];
    W[N_DIM + 1] = p;
    rtype cons[N_CONSERVATIVE];
    physics.compute_conservatives_from_W(cons, W);
    for (uint32_t i_cell = 0; i_cell < mesh->n_cells; ++i_cell) {
        FOR_I_CONSERVATIVE h_conservatives(i_cell, i) = cons[i];
    }
}

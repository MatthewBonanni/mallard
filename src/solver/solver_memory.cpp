/**
 * @file solver_memory.cpp
 * @author Matthew Bonanni (mbonanni001@gmail.com)
 * @brief Device memory of the Solver: the solution, stage and time-stepping arrays.
 * @version 0.1
 * @date 2026-11-24
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 *
 */

#include "solver.h"

#include "diagnostic_code.h"
#include "input.h"

#include <algorithm>
#include <cstddef>
#include <optional>

#include <Kokkos_Core.hpp>

#include "field_units.h"
#include "snapshot_block.h"
#include "run_report.h"
#include "gradient.h"

bool writes_variable(const Input & input, const std::string & name) {
    if (!contains_optional(input, "write_data")) return false;
    const auto outputs = find_value<std::vector<Input>>(input, "write_data");
    for (const auto & output : outputs) {
        if (!output.contains("fields")) continue;
        const auto variables = find_value<std::vector<std::string>>(output, "fields");
        if (std::find(variables.begin(), variables.end(), name) != variables.end()) return true;
    }
    return false;
}

void Solver::allocate_memory() {
    const uint32_t n_species = static_cast<uint32_t>(species_names.size());
    conservatives = StateView("conservatives", mesh->n_cells);
    species = SpeciesView("species", mesh->n_cells, n_species);
    primitives = Kokkos::View<rtype *[N_PRIMITIVE]>("primitives", mesh->n_cells);
    W_cells = Kokkos::View<rtype *[N_CONSERVATIVE]>("W_cells", mesh->n_cells);
    cfl_local = Kokkos::View<rtype *>("cfl_local", mesh->n_cells);
    if (writes_variable(input, "P_MAX")) {
        p_max = Kokkos::View<rtype *>("p_max", mesh->n_cells);
        h_p_max = pinned_mirrors(p_max);
    }
    bool vortex_output = false;
    for (const char * name : {"VORTICITY", "VORTICITY_X", "VORTICITY_Y", "VORTICITY_Z"}) {
        vortex_output = vortex_output || writes_variable(input, name);
    }
    if (vortex_output) {
        vortex_fields = Kokkos::View<rtype **>("vortex_fields", mesh->n_cells, N_DIM == 2 ? 1 : 3);
        h_vortex_fields = pinned_mirrors(vortex_fields);
    }
    derived_settings = derived::Settings::from_input(input, physics, mixture_model.get(), is_viscous());
    derived_selection = {};
    for (uint8_t f = 0; f < derived::N_FIELDS; f++) {
        const auto field = static_cast<derived::Field>(f);
        if (!writes_variable(input, derived::FIELDS[f].name)) continue;
        derived_settings.require(field, input);
        derived_selection.add(field);
    }
    for (const derived::Field field : Statistics::derived_fields(input)) {
        derived_settings.require(field, input);
        derived_selection.add(field);
    }
    if (derived_selection.any()) {
        derived_fields = Kokkos::View<rtype **>("derived_fields", mesh->n_cells, derived_selection.n_columns);
        h_derived_fields = pinned_mirrors(derived_fields);
    }
    if (derived_selection.need(derived::need::SPECIES_GRADIENTS)) derived_workspace = derived::Workspace(mesh->n_cells);
    if (is_viscous() || vortex_output || derived_selection.any()) allocate_viscous_gradients();
    if (positivity_limiter) positivity_kappa = Kokkos::View<rtype *>("positivity_kappa", mesh->n_cells);
    if (hybrid_flux) {
        cell_upwind = Kokkos::View<rtype *>("cell_upwind", mesh->n_cells);
        allocate_viscous_gradients();
    }
    allocate_stage_scratch();
    if (is_mixture() && is_viscous()) {
        cell_transport =Kokkos::View<rtype *[3]>("cell_transport", mesh->n_cells);
        h_cell_transport = pinned_mirrors(cell_transport);
        h_cell_diffusion = pinned_mirrors(cell_diffusion, "cell_diffusion");
        h_D = Kokkos::View<rtype **, Kokkos::LayoutRight, Kokkos::HostSpace>("D", mesh->n_cells, cell_diffusion.extent(1));
    }
    if (is_mixture()) {
        cell_scalars = ScalarView("cell_scalars", mesh->n_cells, n_species + 2);
        T_seed = Kokkos::View<rtype *>("T_seed", mesh->n_cells);
        h_T_seed = pinned_mirrors(T_seed);
        if (double_flux) frozen_thermo = Kokkos::View<rtype *[2]>("frozen_thermo", mesh->n_cells);
        if (auto * teno = dynamic_cast<TENO *>(face_reconstruction.get())) {
            cell_molar_mass = Kokkos::View<rtype *>("cell_molar_mass", mesh->n_cells);
            teno->set_mixture(Kokkos::subview(cell_scalars, Kokkos::ALL(), n_species), cell_molar_mass);
        }
        if (mixture.tabulated()) {
            h_table_fields = Kokkos::View<rtype **, Kokkos::LayoutRight, Kokkos::HostSpace>(
                "table_fields", mesh->n_cells, mixture_model->flamelet().output_names().size());
        }
        if (reacting) allocate_chemistry();
        if (radiating) h_qrad = Kokkos::View<rtype *, Kokkos::HostSpace>("QRAD", mesh->n_cells);
    }
    h_conservatives = pinned_mirrors(conservatives);
    h_species = pinned_mirrors(species);
    h_primitives = pinned_mirrors(primitives);
    h_cfl_local = pinned_mirrors(cfl_local);
}

void Solver::allocate_stage_scratch() {
    // The convective part lives from the reconstruction to the species slots, the diffusive part from
    // update_transport (or the time step's transport) to its last reader. Gas mixtures never start a
    // reconstruction before the time step (exchanges_partition_faces), which would overlap the two
    const size_t CONVECTIVE = 0, DIFFUSIVE = shared_stage_scratch ? 1 : 0;
    memory::ScratchArena & arena = stage_scratch;
    arena = memory::ScratchArena();
    const bool stepping = !state_reader, mixture_stepping = stepping && is_mixture();
    const uint32_t n_cells = mesh->n_cells, n_faces = mesh->n_state_faces();
    const uint32_t n_quad = face_reconstruction->n_face_quadrature_points();
    const uint32_t n_species = static_cast<uint32_t>(species_names.size());
    const uint32_t n_diffusion = (mixture.transport.thermal_diffusion ? 2 : 1) * n_species;
    const uint32_t n_values = N_DIM + 1 + n_species;
    const bool transport = is_mixture() && is_viscous();
    size_t reconstruction = 0, scalars = 0, thermo = 0, mdot = 0, diffusion = 0, values = 0, gradients = 0;
    if (stepping) reconstruction = arena.reserve<std::byte>(CONVECTIVE, face_reconstruction->scratch_bytes());
    if (mixture_stepping) {
        scalars = arena.reserve<std::byte>(CONVECTIVE, scalar_reconstruction.scratch_bytes());
        thermo = arena.reserve<rtype>(CONVECTIVE, size_t(n_faces) * n_quad * 2 * 2);
        mdot = arena.reserve<rtype>(CONVECTIVE, size_t(n_faces) * n_quad);
    }
    if (transport) {
        diffusion = arena.reserve<double>(DIFFUSIVE, size_t(n_cells) * n_diffusion);
        values = arena.reserve<rtype>(DIFFUSIVE, size_t(n_cells) * n_values);
        gradients = arena.reserve<rtype>(DIFFUSIVE, size_t(n_cells) * n_values * N_DIM);
    }
    arena.allocate("stage_scratch");
    if (stepping) face_reconstruction->place_scratch(arena.data(reconstruction));
    if (mixture_stepping) {
        scalar_reconstruction.place_scratch(arena.data(scalars));
        face_thermo = arena.view<Kokkos::View<rtype **[2][2]>>("face_thermo", thermo, n_faces, n_quad);
        face_mdot = arena.view<Kokkos::View<rtype **>>("face_mdot", mdot, n_faces, n_quad);
    }
    if (transport) {
        cell_diffusion = arena.view<Kokkos::View<double **, Kokkos::LayoutRight>>("cell_diffusion", diffusion,
                                                                                  n_cells, n_diffusion);
        transport_values = arena.view<Kokkos::View<rtype **, Kokkos::LayoutRight>>("transport_values", values,
                                                                                   n_cells, n_values);
        transport_gradients = arena.view<Kokkos::View<rtype ***, Kokkos::LayoutRight>>(
            "transport_gradients", gradients, n_cells, n_values, uint32_t(N_DIM));
    }
}

void Solver::allocate_time_stepping() {
    const uint32_t n_species = static_cast<uint32_t>(species_names.size());
    const uint32_t n_quad = face_reconstruction->n_face_quadrature_points();
    const uint32_t n_faces = mesh->n_state_faces();
    // The species slots are filled after the fluxes, the last to read the face states of a stage
    const size_t n_states = size_t(n_faces) * n_quad * 2 * N_CONSERVATIVE;
    const size_t n_slots = is_mixture() ? size_t(n_faces) * 2 * n_species : 0;
    face_storage = Kokkos::View<rtype *>("face_solution", std::max(n_states, n_slots));
    face_solution = Kokkos::View<rtype **[2][N_CONSERVATIVE]>(face_storage.data(), n_faces, n_quad);
    face_flux = Kokkos::View<rtype *[N_CONSERVATIVE]>("face_flux", n_faces);
    const size_t n_particle_values = particles ? particles->n_values() : 0;
    if (keep_last_good) last_good = State("last_good", mesh->n_cells, n_species, n_particle_values);
    if (is_mixture()) {
        species_slots = Kokkos::View<rtype ***, Kokkos::LayoutRight>(face_storage.data(), n_faces, 2, n_species);
        if (double_flux) face_energy_1 = Kokkos::View<rtype *>("face_energy_1", n_faces);
    }
    solution_vec.clear();
    rhs_vec.clear();
    solution_vec.push_back(state());
    for (uint8_t i = 1; i < time_integrator->get_n_solution_vectors(); i++) {
        solution_vec.emplace_back("solution", mesh->n_cells, n_species, n_particle_values);
    }
    for (uint8_t i = 0; i < time_integrator->get_n_rhs_vectors(); i++) {
        rhs_vec.emplace_back("rhs", mesh->n_cells, n_species, n_particle_values);
    }
}

void Solver::copy_host_to_device() {
    Kokkos::deep_copy(conservatives, h_conservatives);
    if (species.span() > 0) Kokkos::deep_copy(species, h_species);
    if (is_mixture()) Kokkos::deep_copy(T_seed, h_T_seed);
    if (reacting) Kokkos::deep_copy(chem_h, h_chem_h);
    if (p_max.is_allocated()) Kokkos::deep_copy(p_max, h_p_max);
    Kokkos::deep_copy(primitives, h_primitives);
}

void Solver::copy_device_to_host() {
    mirrors_unflushed = chemistry_flushed;
    output_arrays_to_host();
    Kokkos::fence("copy_device_to_host");
    if (source_terms) source_terms->check_loops();
    finish_output_arrays();
}

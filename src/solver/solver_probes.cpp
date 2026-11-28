/**
 * @file solver_probes.cpp
 * @author Matthew Bonanni (mbonanni001@gmail.com)
 * @brief Probes of the Solver, and the vertex gradients that they and the viscous fluxes read.
 * @version 0.1
 * @date 2026-11-24
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 *
 */

#include "solver.h"
#include "gated.h"

#include "diagnostic_code.h"

#include <stdexcept>

#include <Kokkos_Core.hpp>

#include "region_input.h"
#include "gradient.h"
#include "viscous_flux.h"
#include "probe_fluxes.h"

void Solver::allocate_viscous_gradients() {
    if (!viscous_gradients.is_allocated()) {
        viscous_gradients = Kokkos::View<rtype *[N_CONSERVATIVE][N_DIM]>("viscous_gradients", mesh->n_cells);
    }
}

void Solver::init_vertex_gradient() {
    // Its readers: the right-hand side's viscous fluxes and upwind sensor, the velocity gradients of schemes
    // without cell gradients, mixture transport, LES, probes, mass-fraction gradients, viscous forces and particles
    const bool read = (!state_reader && (is_viscous() || hybrid_flux)) || !face_reconstruction->has_cell_gradients() ||
                      (is_mixture() && is_viscous()) || les_on || tfles_on || probes.any_gradients() ||
                      derived_selection.need(derived::need::SPECIES_GRADIENTS) ||
                      (is_viscous() && !force_monitors.empty()) || particles;
    if (!viscous_gradients.is_allocated() || !read || vertex_gradient.weights.ghost_offsets.is_allocated()) return;
    vertex_gradient = make_vertex_gradient(make_gradient(*mesh, boundary_data, W_cells, viscous_gradients), *mesh, true,
                                           n_gradient_cells());
}

const LSQVertexGradientFunctor & Solver::viscous_gradient() const {
    if (!vertex_gradient.weights.ghost_offsets.is_allocated()) {
        throw std::logic_error("Solver: the vertex gradient is read, but init_vertex_gradient found no reader");
    }
    return vertex_gradient;
}

void Solver::write_probes() {
    if (!probes.probes_due(step, t)) return;
    if (probes.point_values_due(step, t) || probes.gradients_due(step, t)) {
        update_cell_W();
        if (is_mixture() && probes.transport_due(step, t)) update_transport();
        if (probes.all_gradients_due(step, t, wall_model_on)) update_probe_gradients();
    } else {
        update_primitives();
    }
    if (probes.statistics_due(step, t)) statistics.copy_device_to_host();
    ProbeFluxes fluxes;
    probes.write(step, t, probe_fields(fluxes));
}

void Solver::write_particle_probes() {
    if (!particles || !particles->probes_due(step)) return;
    update_cell_W();
    particles->sample(W_cells, *face_reconstruction, viscous_gradient(), viscous_gradients);
    particles->write_probes(step, static_cast<double>(t));
}

void Solver::sample_probe_averages() {
    const auto [cell_W, transport, all_gradients] = probes.average_needs(wall_model_on);
    if (cell_W) {
        update_cell_W();
        if (is_mixture() && transport) update_transport();
        if (all_gradients) update_probe_gradients();
    } else {
        update_primitives();
    }
    ProbeFluxes fluxes;
    probes.sample_averages(step, t, probe_fields(fluxes));
}

void Solver::update_probe_gradients() {
    // Diffusive fluxes read the gradients of image cells, and modeled walls those of their exchange points
    gated::parallel_for("probe_all_gradients", n_gradient_cells(), viscous_gradient());
    if (wall_model_on) update_wall_model();
}

ProbeFields Solver::probe_fields(ProbeFluxes & fluxes) {
    ProbeFields fields{cell_sampler(),
                       W_cells,
                       probes.any_gradients() ? viscous_gradient() : LSQVertexGradientFunctor{},
                       viscous_gradients,
                       face_reconstruction.get(),
                       physics,
                       boundary_data,
                       is_viscous(),
                       axisymmetric,
                       mesh->face_coords,
                       {},
                       {}};
    if (is_mixture()) {
        fields.cell_gamma = Kokkos::subview(cell_scalars, Kokkos::ALL(), species_names.size());
        if (is_viscous()) fields.transport = cell_transport;
    }
    if (wall_model_on) {
        fields.wall_model = WallModelFunctor{wall_model,        mesh->face_normals, mesh->face_area, W_cells,
                                             viscous_gradients, boundary_data,      wall_model::Gas{physics}};
    }
    fields.fluxes = &fluxes;
    if (is_viscous() && is_mixture()) {
        fluxes.mixture = MixtureViscousFluxFunctor<true>{
            mesh->face_normals, mesh->face_measure, mesh->face_coords, mesh->cell_coords, mesh->cells_of_face,
            mesh->periodic_maps(), mesh->face_shift, boundary_data, mixture, cell_scalars, transport_values,
            transport_gradients, cell_transport, cell_diffusion, {}, {}, axisymmetric, mesh->cell_covariance,
            {}, false, {}};
    } else if (is_viscous()) {
        fluxes.viscous = ViscousFluxFunctor{
            mesh->face_normals, mesh->face_measure, mesh->face_coords, mesh->cell_coords, mesh->cells_of_face,
            mesh->periodic_maps(), mesh->face_shift, W_cells, viscous_gradients, boundary_data, {}, physics,
            axisymmetric, mesh->cell_covariance, {}, false, wall_model.out};
    }
    return fields;
}

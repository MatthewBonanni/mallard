/**
 * @file solver_sponge.cpp
 * @author Matthew Bonanni (mbonanni001@gmail.com)
 * @brief Sponge layers, and the per-step update of characteristic boundaries.
 * @version 0.1
 * @date 2026-10-04
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 *
 */

#include "solver.h"

#include <limits>

#include <Kokkos_Core.hpp>

#include "characteristic.h"
#include "input.h"
#include "region_input.h"
#include "region_motion.h"

namespace {

/** @brief An expression in x, y, z from a number, a quantity with a unit or an expression. */
Expression number_or_expression(const Input & value, const std::string & name, const Dimension & dimension) {
    return Expression(name, quantity_or_expression(value, name, dimension));
}

} // namespace

void Solver::init_sponges(region::Regions & regions) {
    sponge_cells = Kokkos::View<uint32_t *>();
    moving_sponges.clear();
    sponge_dt_max = std::numeric_limits<rtype>::infinity();
    if (!input.contains("sponges")) return;
    const std::vector<Input> sponges = find_value<std::vector<Input>>(input, "sponges");
    const uint32_t n_owned = mesh->n_owned();
    const uint32_t n_species = static_cast<uint32_t>(species_names.size());
    const uint32_t n_vars = N_CONSERVATIVE + n_species;
    std::vector<double> strength(n_owned, 0.0);
    std::vector<double> target(size_t(n_owned) * n_vars, 0.0);
    std::vector<double> moving_strength(n_owned, 0.0);  // the sum of moving sponges' strengths, for the time step
    for (size_t i_s = 0; i_s < sponges.size(); i_s++) {
        const Input & sponge = sponges[i_s];
        const std::string where = "sponges[" + std::to_string(i_s) + "]";
        for (const char * key : {"strength", "u", "p", "T"}) {
            if (!sponge.contains(key)) throw InputError(where + ": missing " + key + ".");
        }
        const Expression sigma = number_or_expression(sponge.at("strength"), where + ".strength", dim::rate);
        const auto & u_in = sponge.at("u").as_array();
        if (u_in.size() != N_DIM) throw InputError(where + ".u must have " + std::to_string(N_DIM) + " components.");
        std::vector<Expression> u;
        FOR_I_DIM u.push_back(number_or_expression(u_in[i], where + ".u[" + std::to_string(i) + "]", dim::velocity));
        const Expression p = number_or_expression(sponge.at("p"), where + ".p", dim::pressure);
        const Expression T = number_or_expression(sponge.at("T"), where + ".T", dim::temperature);

        const std::vector<double> * fractions = nullptr;
        std::shared_ptr<region::MovingFractions> moving;
        if (sponge.contains("region")) {
            const region::Region & r = regions.get(sponge.at("region"), where + ".region");
            region::require_full_dimension(r, sponge.at("region"), where + ".region");
            if (r.moving() || r.solution()) {
                std::shared_ptr<const region::SolutionLevels> levels;
                if (r.solution()) {
                    levels = std::make_shared<region::SolutionLevels>(*mesh, r.conditions(), species_names, cell_fields());
                }
                moving = std::make_shared<region::MovingFractions>(*mesh, r, levels);
            } else {
                fractions = &regions.fractions(sponge.at("region"), where + ".region");
            }
        }
        // Moving sponges: every cell, weighted by the fraction at each stage
        std::vector<double> own_strength, own_target;
        if (moving) {
            own_strength.assign(n_owned, 0.0);
            own_target.assign(size_t(n_owned) * n_vars, 0.0);
        }
        std::vector<double> & S = moving ? own_strength : strength;
        std::vector<double> & T_s = moving ? own_target : target;

        std::vector<uint32_t> candidates;
        for (uint32_t c = 0; c < n_owned; c++) {
            if (!fractions || (*fractions)[c] != 0.0) candidates.push_back(c);
        }
        const std::vector<double> sigma_candidates = evaluate_at({&sigma}, mesh->cell_coords, candidates);
        std::vector<uint32_t> cells;
        std::vector<double> sigma_cells;
        for (size_t k = 0; k < candidates.size(); k++) {
            const uint32_t c = candidates[k];
            double s = sigma_candidates[k];
            if (!(s >= 0.0)) throw InputError(where + ".strength must be non-negative.");
            if (fractions) s *= (*fractions)[c];
            if (s == 0.0) continue;
            cells.push_back(c);
            sigma_cells.push_back(s);
        }
        const std::vector<std::vector<double>> Y =
            is_mixture() ? composition_at_cells(sponge, where, cells) : std::vector<std::vector<double>>();
        std::vector<const Expression *> state = {&p, &T};
        FOR_I_DIM state.push_back(&u[i]);
        const std::vector<double> state_cells = evaluate_at(state, mesh->cell_coords, cells);
        std::vector<rtype> U(n_vars);
        for (size_t j = 0; j < cells.size(); j++) {
            const uint32_t c = cells[j];
            const double * s = state_cells.data() + j * state.size();
            rtype u_c[N_DIM];
            FOR_I_DIM u_c[i] = static_cast<rtype>(s[2 + i]);
            const double p_c = s[0], T_c = s[1];
            if (is_mixture()) {
                mixture_model->conservatives(p_c, T_c, u_c, Y[j], U.data(), U.data() + N_CONSERVATIVE);
            } else {
                rtype W[N_CONSERVATIVE];
                W[0] = physics.get_density_from_pressure_temperature(static_cast<rtype>(p_c), static_cast<rtype>(T_c));
                FOR_I_DIM W[1 + i] = u_c[i];
                W[N_DIM + 1] = static_cast<rtype>(p_c);
                physics.compute_conservatives_from_W(U.data(), W);
            }
            S[c] += sigma_cells[j];
            for (uint32_t v = 0; v < n_vars; v++) T_s[size_t(c) * n_vars + v] += sigma_cells[j] * double(U[v]);
        }
        if (moving) {
            MovingSponge m{moving, Kokkos::View<rtype *>("sponge_moving_strength", n_owned),
                           Kokkos::View<rtype **, Kokkos::LayoutRight>("sponge_moving_target", n_owned, n_vars)};
            auto h_s = Kokkos::create_mirror_view(m.strength);
            auto h_t = Kokkos::create_mirror_view(m.target);
            for (uint32_t c = 0; c < n_owned; c++) {
                h_s(c) = static_cast<rtype>(S[c]);
                moving_strength[c] += S[c];
                for (uint32_t v = 0; v < n_vars; v++) h_t(c, v) = static_cast<rtype>(T_s[size_t(c) * n_vars + v]);
            }
            Kokkos::deep_copy(m.strength, h_s);
            Kokkos::deep_copy(m.target, h_t);
            moving_sponges.push_back(std::move(m));
        }
    }

    std::vector<uint32_t> cells;
    double max_strength = 0.0;
    for (uint32_t c = 0; c < n_owned; c++) {
        if (strength[c] > 0.0) cells.push_back(c);
        max_strength = std::max(max_strength, strength[c] + moving_strength[c]);
    }
    max_strength = comm::allreduce(max_strength, comm::Op::MAX);
    const uint64_t n_cells = comm::allreduce(uint64_t(cells.size()), comm::Op::SUM);
    sponge_cells = Kokkos::View<uint32_t *>("sponge_cells", cells.size());
    sponge_strength = Kokkos::View<rtype *>("sponge_strength", cells.size());
    sponge_target = Kokkos::View<rtype **, Kokkos::LayoutRight>("sponge_target", cells.size(), n_vars);
    auto h_cells = Kokkos::create_mirror_view(sponge_cells);
    auto h_strength = Kokkos::create_mirror_view(sponge_strength);
    auto h_target = Kokkos::create_mirror_view(sponge_target);
    for (size_t k = 0; k < cells.size(); k++) {
        h_cells(k) = cells[k];
        h_strength(k) = static_cast<rtype>(strength[cells[k]]);
        for (uint32_t v = 0; v < n_vars; v++) h_target(k, v) = static_cast<rtype>(target[size_t(cells[k]) * n_vars + v]);
    }
    Kokkos::deep_copy(sponge_cells, h_cells);
    Kokkos::deep_copy(sponge_strength, h_strength);
    Kokkos::deep_copy(sponge_target, h_target);
    if (max_strength > 0.0) sponge_dt_max = static_cast<rtype>(1.0 / max_strength);
    if (!moving_sponges.empty() && !stage_time.is_allocated()) stage_time = Kokkos::View<rtype>("stage_time");
    source_summary.quantity("sponges", std::to_string(sponges.size()), "sponges",
                            moving_sponges.empty() ? "" : std::to_string(moving_sponges.size()) + " moving");
    source_summary.quantity("sponge cells", logging::count(n_cells), "cells", "");
    source_summary.quantity("sponge strength", logging::real(max_strength), units::column("1/s"), "largest");
}

void Solver::apply_sponges(const State & solution, const State & rhs, const rtype t_stage, const rtype c_stage) {
    apply_moving_sponges(solution, rhs, t_stage, c_stage);
    if (sponge_cells.extent(0) == 0) return;
    const Kokkos::View<uint32_t *> cells = sponge_cells;
    const Kokkos::View<rtype *> strength = sponge_strength;
    const Kokkos::View<rtype **, Kokkos::LayoutRight> target = sponge_target;
    const StateView U = solution.flow, dU = rhs.flow;
    const SpeciesView rhoY = solution.species, d_rhoY = rhs.species;
    const uint32_t n_species = static_cast<uint32_t>(species_names.size());
    Kokkos::parallel_for("sponges", cells.extent(0), KOKKOS_LAMBDA(const uint32_t k) {
        const uint32_t c = cells(k);
        const rtype s = strength(k);
        FOR_I_CONSERVATIVE dU(c, i) += target(k, i) - s * U(c, i);
        for (uint32_t j = 0; j < n_species; j++) d_rhoY(c, j) += target(k, N_CONSERVATIVE + j) - s * rhoY(c, j);
    });
}

void Solver::apply_moving_sponges(const State & solution, const State & rhs, const rtype t_stage, const rtype c_stage) {
    if (moving_sponges.empty()) return;
    set_stage_time(t_stage, c_stage);
    const StateView U = solution.flow, dU = rhs.flow;
    const SpeciesView rhoY = solution.species, d_rhoY = rhs.species;
    const uint32_t n_species = static_cast<uint32_t>(species_names.size());
    for (const MovingSponge & m : moving_sponges) {
        m.fractions->update(stage_time);
        const Kokkos::View<double *> f = m.fractions->fractions();
        const Kokkos::View<rtype *> strength = m.strength;
        const Kokkos::View<rtype **, Kokkos::LayoutRight> target = m.target;
        Kokkos::parallel_for("moving_sponges", strength.extent(0), KOKKOS_LAMBDA(const uint32_t c) {
            const rtype w = static_cast<rtype>(f(c)), s = strength(c);
            if (w == 0.0_r || s == 0.0_r) return;
            FOR_I_CONSERVATIVE dU(c, i) += w * (target(c, i) - s * U(c, i));
            for (uint32_t j = 0; j < n_species; j++) d_rhoY(c, j) += w * (target(c, N_CONSERVATIVE + j) - s * rhoY(c, j));
        });
    }
}

void Solver::update_characteristic_boundaries(const rtype t_stage) {
    // Once per step, at the first stage
    if (boundary_data.char_faces.extent(0) == 0 || t_stage != t || t_stage == t_characteristic) return;
    if (characteristic_transverse) {
        const CharacteristicTransverseFunctor transverse{boundary_data,      mesh->face_coords,   mesh->face_normals,
                                                         mesh->face_area,    mesh->cells_of_face, W_cells};
        Kokkos::parallel_for("characteristic_transverse", boundary_data.char_faces.extent(0), transverse);
    }
    // Inlet targets at the end of the step, whose change enters the incoming wave
    if (!inflows.empty() || !timed_inlets.empty()) {
        if (!characteristic_time.is_allocated()) characteristic_time = Kokkos::View<rtype>("char_target_time");
        const Kokkos::View<rtype> t_next = characteristic_time;
        if (recording_step) {
            const rtype * s = step_dt_device();
            const rtype * t_step = step_time_device();
            Kokkos::parallel_for("characteristic_time", 1, KOKKOS_LAMBDA(const int) { t_next() = *t_step + *s; });
        } else {
            const rtype t_end = t + dt;
            Kokkos::parallel_for("characteristic_time", 1, KOKKOS_LAMBDA(const int) { t_next() = t_end; });
        }
        timed_inlets.fill(t_next, boundary_data.char_target_next);
        for (auto & inflow : inflows) inflow->fill(t_next, boundary_data.char_target_next);
    }
    const CharacteristicStateFunctor state{boundary_data,
                                           mesh->face_normals,
                                           face_solution,
                                           is_mixture() ? face_thermo : Kokkos::View<rtype **[2][2]>(),
                                           face_reconstruction->quadrature_face.weights,
                                           face_reconstruction->face_quad_weights,
                                           dt,
                                           !characteristic_state_set,
                                           recording_step ? step_dt_device() : nullptr,
                                           characteristic_start,
                                           step_controller.enabled()
                                               ? step_controller.slots().data() + StepController::ACCEPTED
                                               : nullptr};
    Kokkos::parallel_for("characteristic_state", boundary_data.char_faces.extent(0), state);
    t_characteristic = t_stage;
    characteristic_state_set = true;
}

uint64_t Solver::restart_face_key(const uint32_t i_face) const {
    const uint32_t c = static_cast<uint32_t>(mesh->h_cells_of_face(i_face, 0));
    uint8_t local = 0;
    while (mesh->h_face_of_cell(c, local) != i_face) local++;
    return mesh->h_global_cell(c) * RESTART_FACES_PER_CELL + local;
}

RestartFaces Solver::characteristic_restart_faces() const {
    RestartFaces faces;
    if (!characteristic_state_set) return faces;
    // The mirrors output_arrays_to_host filled
    for (uint32_t k = 0; k < h_char_faces.extent(0); k++) {
        if (static_cast<uint32_t>(mesh->h_cells_of_face(h_char_faces(k), 0)) >= mesh->n_owned()) continue;
        faces.keys.push_back(restart_face_key(h_char_faces(k)));
        for (uint32_t w = 0; w < RestartFaces::WIDTH; w++) faces.values.push_back(h_char_state(k, w));
    }
    return faces;
}

void Solver::restore_characteristic_state(const RestartData & restart) {
    // A file without some face (older versions, or a boundary that was not
    // characteristic) starts every face afresh from the solution
    const auto h_faces = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), boundary_data.char_faces);
    const auto h_state = Kokkos::create_mirror_view(boundary_data.char_state);
    uint32_t missing = 0;
    for (uint32_t k = 0; k < h_faces.extent(0) && !missing; k++) {
        const rtype * values = restart.face(restart_face_key(h_faces(k)));
        missing = values == nullptr;
        for (uint32_t w = 0; w < RestartFaces::WIDTH && values; w++) h_state(k, w) = values[w];
    }
    if (mesh->n_global_cells > 0) missing = comm::allreduce(missing, comm::Op::MAX);
    if (missing) return;
    Kokkos::deep_copy(boundary_data.char_state, h_state);
    characteristic_state_set = true;
}

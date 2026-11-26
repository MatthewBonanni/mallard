/**
 * @file solver_time_step.cpp
 * @author Matthew Bonanni (mbonanni001@gmail.com)
 * @brief Time step of the Solver: the CFL time step and the target times.
 * @version 0.1
 * @date 2026-11-24
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 *
 */

#include "solver.h"
#include "gated.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

#include <Kokkos_Core.hpp>

#include "comm.h"
#include "device_comm.h"
#include "log.h"
#include "gradient.h"

void Solver::update_p_max() {
    update_primitives();
    Kokkos::View<rtype *> peak = p_max;
    Kokkos::View<rtype *[N_PRIMITIVE]> P = primitives;
    gated::parallel_for("update_p_max", mesh->n_cells, KOKKOS_LAMBDA(const uint32_t i_cell) {
        peak(i_cell) = Kokkos::max(peak(i_cell), P(i_cell, N_DIM));
    });
}

rtype Solver::next_target_time(const bool unstaged) const {
    rtype t_target = (t_stop > 0) ? t_stop : std::numeric_limits<rtype>::infinity();
    t_target = std::min({t_target, next_landing_time(), pause_time});
    for (const auto & writer : data_writers) {
        if (!(unstaged && staged_outputs)) t_target = std::min(t_target, writer->next_time());
    }
    return std::min(t_target, probes.next_time(unstaged));
}

void Solver::calc_dt(const bool overlap) {
    // Halo values are stale after the last stage of the previous step; the
    // first stage of the next one reuses them
    exchange_for_first_stage(overlap);
    update_boundary_states(t);
    dt_cfl1 = calc_dt_cfl1();
    if (!(dt_cfl1 > 0.0_r)) throw DivergedError();
    save_last_good();
    dt = use_cfl ? cfl * dt_cfl1 : dt_fixed;
    if (step_controller.enabled()) {
        // The first step takes the cfl or dt of [run], later ones the controller's
        const rtype dt_next = step_controller.dt_next();
        if (dt_next > 0.0_r) dt = dt_next;
    }
    dt = std::min(dt, sponge_dt_max);
    // Land exactly on t_stop and on time-based output times
    const rtype t_target = next_target_time();
    if (t + dt > t_target && t_target > t) {
        dt = t_target - t;
    }
    if (!(dt > 0.0_r)) {
        throw std::runtime_error("the time step is " + logging::real(double(dt)) + " at step " + std::to_string(step) +
                                 ", t = " + logging::real(double(t)) +
                                 ": it must be positive. Check run.dt, run.cfl, the sponges' and the step "
                                 "controller's limits, and that t_stop is after the initial time.");
    }
    // cfl_local holds dt_cfl1 per cell; convert to local CFL number
    Kokkos::View<rtype *> c = cfl_local;
    const rtype dt_ = dt;
    gated::parallel_for("local_cfl", mesh->n_cells, KOKKOS_LAMBDA(const uint32_t i) {
        c(i) = dt_ / c(i);
    });
}

/**
 * @brief Per-cell stable time step for CFL = 1 (Blazek eqs. 6.20-6.21, C = 4):
 *        dt_i = 2 V_i / (sum_f (|u_n| + a)_f A_f + 4 nu_eff sum_f A_f^2 / V_i),
 *        with the face wave speed taken as the max over the two adjacent cells
 *        (at boundary faces, the cell and the exterior state) and nu_eff = max(4/3, gamma/Pr) mu / rho for viscous flow. Half the
 *        sums over faces stand for Blazek's sums of projected areas.
 */
struct TimeStepFunctor {
    Kokkos::View<uint32_t *> offsets_faces_of_cell;
    Kokkos::View<uint32_t *> faces_of_cell;
    Kokkos::View<int32_t *[2]> cells_of_face;
    Kokkos::View<rtype *[N_DIM]> face_normals;
    Kokkos::View<rtype *> face_area;
    Kokkos::View<rtype *> cell_volume;
    StateView conservatives;
    Kokkos::View<rtype *> dt_local;
    Euler physics;
    Kokkos::View<rtype *[N_DIM]> radius_coords;  // axisymmetric runs: cell_coords, else empty
    Kokkos::View<rtype *[3]> sgs;                // LES: (cell, [mu_t, ...]), else empty
    rtype Pr_t = 1.0_r;
    BoundaryData boundaries;
    PeriodicMaps periodic;
    Kokkos::View<uint8_t *> face_shift;
    rtype viscous_stiffness = 1.0_r;  // the viscous operator's spectral radius over the compact one's

    KOKKOS_INLINE_FUNCTION
    void cell_W(const int32_t i_cell, rtype * W) const {
        rtype cons[N_CONSERVATIVE];
        FOR_I_CONSERVATIVE cons[i] = conservatives(i_cell, i);
        physics.compute_W_from_conservatives(W, cons);
    }

    KOKKOS_INLINE_FUNCTION
    rtype wave_speed(const int32_t i_cell, const rtype * n) const {
        rtype W[N_CONSERVATIVE];
        cell_W(i_cell, W);
        const rtype u_n = dot<N_DIM>(W + 1, n);
        return Kokkos::fabs(u_n) + physics.get_sound_speed_from_pressure_density(W[N_DIM + 1], W[0]);
    }

    KOKKOS_INLINE_FUNCTION
    bool finite(const uint32_t i_cell) const {
        bool all = true;
        FOR_I_CONSERVATIVE all = all && Kokkos::isfinite(conservatives(i_cell, i));
        return all;
    }

    KOKKOS_INLINE_FUNCTION
    void operator()(const uint32_t i_cell, rtype & dt_min) const {
        rtype sum = 0.0;
        rtype sum_area2 = 0.0;
        for (uint32_t k = offsets_faces_of_cell(i_cell); k < offsets_faces_of_cell(i_cell + 1); k++) {
            const uint32_t i_face = faces_of_cell(k);
            rtype n[N_DIM];
            rtype n_vec[N_DIM];
            FOR_I_DIM n_vec[i] = face_normals(i_face, i);
            unit<N_DIM>(n_vec, n);
            const int32_t c0 = cells_of_face(i_face, 0);
            const int32_t c1 = cells_of_face(i_face, 1);
            rtype lambda = wave_speed(c0, n);
            if (c1 >= 0) {
                // The normal in cell 1's frame
                if (periodic.rotated()) periodic.unrotate(face_shift(i_face), n);
                lambda = Kokkos::fmax(lambda, wave_speed(c1, n));
            } else {
                rtype W[N_CONSERVATIVE];
                cell_W(c0, W);
                lambda = Kokkos::fmax(lambda, boundaries.exterior_wave_speed(i_face, W, n, physics.gamma));
            }
            sum += lambda * face_area(i_face);
            sum_area2 += face_area(i_face) * face_area(i_face);
        }
        if (physics.is_viscous()) {
            rtype cons[N_CONSERVATIVE], W[N_CONSERVATIVE];
            FOR_I_CONSERVATIVE cons[i] = conservatives(i_cell, i);
            physics.compute_W_from_conservatives(W, cons);
            const rtype T = W[N_DIM + 1] / (W[0] * physics.R);
            const rtype mu = physics.viscosity(T);
            rtype coeff = Kokkos::fmax(4.0_r / 3.0_r, physics.gamma / physics.Pr) * mu / W[0];
            if (sgs.extent(0) > 0) {
                const rtype mu_t = sgs(i_cell, 0);
                coeff = Kokkos::fmax(4.0_r / 3.0_r * (mu + mu_t), physics.gamma * (mu / physics.Pr + mu_t / Pr_t)) /
                        W[0];
            }
            sum += 4.0_r * viscous_stiffness * coeff * sum_area2 / cell_volume(i_cell);
            if (radius_coords.extent(0) > 0) {
                // Decay of u_r by the hoop stress
                const rtype r = radius_coords(i_cell, 1);
                sum += coeff * cell_volume(i_cell) / (r * r);
            }
        }
        const rtype dt_i = 2.0_r * cell_volume(i_cell) / sum;
        dt_local(i_cell) = dt_i;
        // A negative minimum flags a non-finite state to the run
        dt_min = Kokkos::fmin(dt_min, finite(i_cell) ? dt_i : -1.0_r);
    }
};

rtype Solver::calc_dt_cfl1(Kokkos::View<rtype> on_device) {
    if (is_mixture()) return calc_dt_cfl1_mixture(on_device);
    TimeStepFunctor functor{mesh->offsets_faces_of_cell,
                            mesh->faces_of_cell,
                            mesh->cells_of_face,
                            mesh->face_normals,
                            mesh->face_measure,
                            mesh->cell_measure,
                            conservatives,
                            cfl_local,
                            physics,
                            axisymmetric ? mesh->cell_coords : Kokkos::View<rtype *[N_DIM]>(),
                            les_coefficients,
                            les.Pr_t,
                            boundary_data,
                            mesh->periodic_maps(),
                            mesh->face_shift};
    // Reconstruction gradients: 1D symbol up to 8.2 / h^2 (central stencils fitted exactly; 5.5 by least squares)
    // against the compact gradients' 4 / h^2, with the jump damping of 4/3
    if (polynomial_viscous) functor.viscous_stiffness = 2.05_r;
    if (les_on) {
        eddy_viscosity_of_state(mesh->n_owned());
        if (les.dynamic) {
            update_dynamic_constant();
            update_eddy_viscosity(mesh->n_owned(), true);
        }
    }
    // On comm_space, so that the reduction ahead of the allreduce does not wait
    // for the reconstruction that exchange_for_first_stage() started
    comm::enqueue_wait(comm_space, Kokkos::DefaultExecutionSpace());
    const Kokkos::RangePolicy<> cells(comm_space, 0, mesh->n_owned());
    if (on_device.data()) {
        gated::parallel_reduce("time_step", cells, functor,
                                Kokkos::Min<rtype, Kokkos::DefaultExecutionSpace::memory_space>(on_device));
        comm::enqueue_wait(Kokkos::DefaultExecutionSpace(), comm_space);
        return 0.0;
    }
    gated::parallel_reduce("time_step", cells, functor,
                            Kokkos::Min<rtype, Kokkos::DefaultExecutionSpace::memory_space>(dt_reduce_slot()));
    comm::enqueue_wait(Kokkos::DefaultExecutionSpace(), comm_space);
    return reduce_dt();
}

Kokkos::View<rtype> Solver::dt_reduce_slot() {
    if (!step_reduce.is_allocated()) step_reduce = Kokkos::View<rtype *>("step_reduce", N_STEP_REDUCE);
    return Kokkos::View<rtype>(step_reduce.data());
}

rtype Solver::reduce_dt() {
    control_contributed = run_control::pending();
    const Kokkos::View<rtype *> values = step_reduce;
    const int contributed = control_contributed;
    gated::parallel_for("time_step_requests", run_control::N_REQUESTS, KOKKOS_LAMBDA(const int k) {
        values(k + 1) = (contributed >> k) & 1 ? rtype(-1) : rtype(0);
    });
    if (is_distributed()) comm::allreduce_device(values, comm::Op::MIN, halo.uses_nccl());
    const auto reduced = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), values);
    control_reduced = 0;
    for (int k = 0; k < run_control::N_REQUESTS; k++) {
        if (reduced(k + 1) < 0) control_reduced |= 1 << k;
    }
    return reduced(0);
}

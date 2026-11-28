/**
 * @file solver_forces.cpp
 * @author Matthew Bonanni (mbonanni001@gmail.com)
 * @brief Forces on boundary zones ([[forces]]).
 * @version 0.1
 * @date 2026-11-24
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 *
 */

#include "solver.h"
#include "gated.h"

#include <cmath>
#include <cstddef>
#include <iomanip>

#include <Kokkos_Core.hpp>

#include "comm.h"
#include "device_comm.h"
#include "gradient.h"
#include "viscous_flux.h"

/**
 * @brief Traction of the fluid on boundary faces, t = p n - tau . n with n
 *        pointing out of the fluid. Pressure is the adjacent cell's; on walls
 *        the velocity gradient is corrected with the wall velocity as in the
 *        viscous flux.
 */
struct ForceFunctor {
    Kokkos::View<uint32_t *> faces;
    Kokkos::View<rtype *[N_DIM]> normals;
    Kokkos::View<rtype *[N_DIM]> face_coords;
    Kokkos::View<rtype *[N_DIM]> cell_coords;
    Kokkos::View<int32_t *[2]> cells_of_face;
    Kokkos::View<rtype *[N_CONSERVATIVE]> W;
    Kokkos::View<rtype *[N_CONSERVATIVE][N_DIM]> gradients;
    BoundaryData boundaries;
    Euler physics;
    bool viscous;
    Kokkos::View<rtype *[3]> mixture_transport;  // viscous mixtures: (cell, [mu, ...]), else empty
    bool axisymmetric = false;  // per radian: tractions weighted by the face radius

    WallModelFlux wall_model = {};  // modeled walls: their stress and heat flux, else empty

    struct value_type {
        rtype v[2 * N_DIM + 1];
    };

    KOKKOS_INLINE_FUNCTION
    void init(value_type & sum) const {
        for (int i = 0; i < 2 * N_DIM + 1; i++) sum.v[i] = 0.0;
    }

    KOKKOS_INLINE_FUNCTION
    void join(value_type & dst, const value_type & src) const {
        for (int i = 0; i < 2 * N_DIM + 1; i++) dst.v[i] += src.v[i];
    }

    KOKKOS_INLINE_FUNCTION
    void operator()(const uint32_t k, value_type & total) const {
        rtype * sum = total.v;
        const uint32_t f = faces(k);
        const int32_t c = cells_of_face(f, 0);
        rtype n_A[N_DIM];
        FOR_I_DIM n_A[i] = normals(f, i);
        rtype n[N_DIM];
        unit<N_DIM>(n_A, n);
        if (axisymmetric) FOR_I_DIM n_A[i] *= face_coords(f, 1);
        FOR_I_DIM sum[i] += W(c, N_DIM + 1) * n_A[i];
        if (!viscous) return;
        const BoundaryCondition & bc = boundaries.bcs(boundaries.face_bc(f));
        if (bc.modeled && wall_model.active()) {
            const int32_t row = wall_model.face_index(f);
            if (row < 0) return;
            const rtype A = Kokkos::sqrt(dot<N_DIM>(n_A, n_A));
            FOR_I_DIM sum[N_DIM + i] -= A * wall_model.flux(row, i);
            sum[2 * N_DIM] += A * wall_model.flux(row, N_DIM);
            return;
        }
        rtype g[N_DIM][N_DIM];
        for (uint8_t v = 0; v < N_DIM; v++) {
            FOR_I_DIM g[v][i] = gradients(c, 1 + v, i);
        }
        if (bc.is_wall()) {
            rtype dn = 0.0;
            FOR_I_DIM dn += (face_coords(f, i) - cell_coords(c, i)) * n[i];
            for (uint8_t v = 0; v < N_DIM; v++) {
                const rtype correction = (bc.data[1 + v] - W(c, 1 + v)) / dn - dot<N_DIM>(g[v], n);
                FOR_I_DIM g[v][i] += correction * n[i];
            }
        }
        const rtype mu = mixture_transport.extent(0) > 0 ? mixture_transport(c, 0)
                                                         : physics.viscosity(W(c, N_DIM + 1) / (W(c, 0) * physics.R));
        rtype tau_n[N_DIM];
        rtype u[N_DIM];
        FOR_I_DIM u[i] = bc.is_wall() ? bc.data[1 + i] : W(c, 1 + i);
        viscous_traction(mu, g, n_A, tau_n, axisymmetric ? hoop_divergence(face_coords(f, 1), u, g) : 0.0_r);
        FOR_I_DIM sum[N_DIM + i] -= tau_n[i];
    }
};

void Solver::reduce_force(const Kokkos::View<uint32_t *> & faces, double * out) {
    const comm::SelfScope scope = communication_scope();
    // The gradients and the wall model read the halo, stale after the last stage
    update_cell_W();
    if (is_viscous()) {
        gated::parallel_for("force_gradients", n_gradient_cells(), viscous_gradient());
        if (is_mixture()) update_transport();
        if (wall_model_on) update_wall_model();
    }
    ForceFunctor functor{faces, mesh->face_normals, mesh->face_coords, mesh->cell_coords, mesh->cells_of_face,
                         W_cells, viscous_gradients, boundary_data, physics, is_viscous(),
                         is_mixture() ? cell_transport : Kokkos::View<rtype *[3]>(), axisymmetric, wall_model.out};
    using DeviceMemory = Kokkos::DefaultExecutionSpace::memory_space;
    constexpr size_t n = 2 * N_DIM + 1;
    const Kokkos::View<rtype *> sum = monitor_sums;
    gated::parallel_reduce("force", faces.extent(0), functor,
                            Kokkos::View<ForceFunctor::value_type, DeviceMemory, Kokkos::MemoryUnmanaged>(
                                reinterpret_cast<ForceFunctor::value_type *>(sum.data())));
    if (is_distributed()) comm::allreduce_device(Kokkos::View<rtype *>(sum.data(), n), comm::Op::SUM, halo.uses_nccl());
    gated::parallel_for("force_out", 1, KOKKOS_LAMBDA(const int) {
        for (size_t i = 0; i < n; i++) out[i] = static_cast<double>(sum(i));
    });
}

std::array<rtype, 2 * N_DIM + 1> Solver::calc_force(const Kokkos::View<uint32_t *> & faces) {
    const comm::SelfScope scope = communication_scope();
    allocate_monitors();
    reduce_force(faces, monitor_results.data());
    const auto values = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), monitor_results);
    std::array<rtype, 2 * N_DIM + 1> F;
    for (size_t i = 0; i < 2 * N_DIM + 1; i++) F[i] = static_cast<rtype>(values(i));
    return F;
}

void Solver::write_force_row(ForceMonitor & monitor, const double * values) {
    monitor.last = {"forces_" + monitor.zone, step, double(t), {}};
    const auto columns = force_columns(monitor.modeled);
    for (size_t i = 0; i < columns.size(); i++) monitor.last.values.emplace_back(columns[i].first, values[i]);
    if (!monitor.out) return;
    *monitor.out << step << "," << std::setprecision(12) << t;
    for (size_t i = 0; i < 2 * N_DIM; i++) *monitor.out << "," << static_cast<rtype>(values[i]);
    if (monitor.modeled) *monitor.out << "," << static_cast<rtype>(values[2 * N_DIM]);
    *monitor.out << "\n";
    monitor.out->flush();
}

void Solver::write_forces() {
    for (size_t k = 0; k < force_monitors.size(); k++) {
        if (step % force_monitors[k].interval != 0) continue;
        allocate_monitors();
        monitor_reductions(k);
        const auto all = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), monitor_results);
        write_force_row(force_monitors[k], all.data() + monitor_offsets[k]);
    }
}

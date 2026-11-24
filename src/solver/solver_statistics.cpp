/**
 * @file solver_statistics.cpp
 * @author Matthew Bonanni (mbonanni001@gmail.com)
 * @brief Statistics, integrals and derived fields of the Solver.
 * @version 0.1
 * @date 2026-11-24
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 *
 */

#include "solver.h"
#include "gated.h"

#include "diagnostic_code.h"

#include <cstddef>
#include <iomanip>

#include <Kokkos_Core.hpp>

#include "comm.h"
#include "device_comm.h"
#include "gradient.h"

void Solver::sample_statistics() {
    update_statistics_fields();
    statistics.sample(step, t, cell_sampler(), mesh->n_owned());
}

void Solver::update_statistics_fields() {
    if (statistics.samples_derived()) {
        update_velocity_gradients();
        update_derived_fields();
    } else {
        update_primitives();
    }
    if (statistics.samples_chemistry()) update_heat_release_rate();
}

CellSampler Solver::sample_statistics_of_state(bool next) {
    const comm::SelfScope scope = communication_scope();
    if (next) apply_output_state(state_reader(*mesh));
    sample_statistics();
    if (probes.averages_due(step, t)) sample_probe_averages();
    statistics.copy_device_to_host();
    return cell_sampler();
}

std::array<rtype, N_CONSERVATIVE> Solver::integrate_conservatives() {
    const comm::SelfScope scope = communication_scope();
    std::array<rtype, N_CONSERVATIVE> total;
    StateView U = conservatives;
    Kokkos::View<rtype *> vol = mesh->cell_measure;
    for (uint8_t i_var = 0; i_var < N_CONSERVATIVE; i_var++) {
        rtype sum = 0.0;
        gated::parallel_reduce("integrate", mesh->n_owned(), KOKKOS_LAMBDA(const uint32_t i, rtype & s) {
            s += U(i, i_var) * vol(i);
        }, sum);
        total[i_var] = sum;
    }
    return is_distributed() ? comm::allreduce(total, comm::Op::SUM) : total;
}

/**
 * @brief Per-cell contributions to Solver::integrate_flow_statistics.
 */
struct FlowStatisticsFunctor {
    Kokkos::View<rtype *[N_CONSERVATIVE]> W;
    Kokkos::View<rtype *[N_PRIMITIVE]> primitives;
    Kokkos::View<rtype *[N_CONSERVATIVE][N_DIM]> gradients;
    Kokkos::View<rtype *> volume;
    Kokkos::View<rtype *[N_DIM]> radius_coords;  // axisymmetric runs: cell_coords, else empty

    struct value_type {
        rtype v[N_FLOW_STATISTICS];
    };

    KOKKOS_INLINE_FUNCTION
    void init(value_type & sum) const {
        for (int i = 0; i < N_FLOW_STATISTICS; i++) sum.v[i] = 0.0;
    }

    KOKKOS_INLINE_FUNCTION
    void join(value_type & dst, const value_type & src) const {
        for (int i = 0; i < N_FLOW_STATISTICS; i++) dst.v[i] += src.v[i];
    }

    KOKKOS_INLINE_FUNCTION
    void operator()(const uint32_t c, value_type & sum) const {
        const rtype rho = W(c, 0);
        const rtype T = primitives(c, N_DIM + 1);
        const rtype V = volume(c);
        rtype u2 = 0.0, div = 0.0;
        FOR_I_DIM {
            u2 += W(c, 1 + i) * W(c, 1 + i);
            div += gradients(c, 1 + i, i);
        }
        if (radius_coords.extent(0) > 0) div += W(c, 2) / radius_coords(c, 1);
        // gradients(c, 1 + k, i) = d u_k / d x_i
        rtype omega2 = 0.0;
        if constexpr (N_DIM == 2) {
            const rtype w = gradients(c, 2, 0) - gradients(c, 1, 1);
            omega2 = w * w;
        } else {
            for (uint8_t k = 0; k < 3; k++) {
                const uint8_t a = static_cast<uint8_t>((k + 1) % 3), b = static_cast<uint8_t>((k + 2) % 3);
                const rtype w = gradients(c, 1 + b, a) - gradients(c, 1 + a, b);
                omega2 += w * w;
            }
        }
        sum.v[0] += 0.5_r * rho * u2 * V;
        sum.v[1] += 0.5_r * rho * omega2 * V;
        sum.v[2] += div * div * V;
        sum.v[3] += W(c, N_DIM + 1) * div * V;
        sum.v[4] += u2 * V;
        sum.v[5] += omega2 * V;
        sum.v[6] += rho * rho * V;
        sum.v[7] += T * V;
        sum.v[8] += T * T * V;
    }
};

void Solver::update_velocity_gradients() {
    update_cell_W();
    if (face_reconstruction->has_cell_gradients()) {
        face_reconstruction->cell_gradients(W_cells, viscous_gradients, mesh->n_owned());
    } else {
        gated::parallel_for("statistics_gradients", mesh->n_owned(), viscous_gradient());
    }
}

void Solver::update_cell_W() {
    halo.exchange(state());
    update_boundary_states(t);
    if (is_mixture()) {
        update_cell_states_and_primitives();
        return;
    }
    const Euler phys = physics;
    StateView U = conservatives;
    Kokkos::View<rtype *[N_CONSERVATIVE]> W = W_cells;
    gated::parallel_for("statistics_W", mesh->n_cells, KOKKOS_LAMBDA(const uint32_t i_cell) {
        rtype cons[N_CONSERVATIVE], W_c[N_CONSERVATIVE];
        FOR_I_CONSERVATIVE cons[i] = U(i_cell, i);
        phys.compute_W_from_conservatives(W_c, cons);
        FOR_I_CONSERVATIVE W(i_cell, i) = W_c[i];
    });
    update_primitives();
}

void Solver::reduce_flow_statistics(double * out) {
    const comm::SelfScope scope = communication_scope();
    update_velocity_gradients();
    FlowStatisticsFunctor functor{W_cells, primitives, viscous_gradients, mesh->cell_measure,
                                  axisymmetric ? mesh->cell_coords : Kokkos::View<rtype *[N_DIM]>()};
    using DeviceMemory = Kokkos::DefaultExecutionSpace::memory_space;
    const Kokkos::View<rtype *> sum = monitor_sums;
    gated::parallel_reduce("statistics", mesh->n_owned(), functor,
                            Kokkos::View<FlowStatisticsFunctor::value_type, DeviceMemory, Kokkos::MemoryUnmanaged>(
                                reinterpret_cast<FlowStatisticsFunctor::value_type *>(sum.data())));
    if (is_distributed()) {
        comm::allreduce_device(Kokkos::View<rtype *>(sum.data(), N_FLOW_STATISTICS), comm::Op::SUM, halo.uses_nccl());
    }
    gated::parallel_for("statistics_out", 1, KOKKOS_LAMBDA(const int) {
        for (size_t i = 0; i < N_FLOW_STATISTICS; i++) out[i] = static_cast<double>(sum(i));
    });
}

std::array<rtype, N_FLOW_STATISTICS> Solver::integrate_flow_statistics() {
    const comm::SelfScope scope = communication_scope();
    allocate_monitors();
    reduce_flow_statistics(monitor_results.data());
    const auto values = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), monitor_results);
    std::array<rtype, N_FLOW_STATISTICS> sums;
    for (size_t i = 0; i < N_FLOW_STATISTICS; i++) sums[i] = static_cast<rtype>(values(i));
    return sums;
}

void Solver::update_vortex_fields() {
    Kokkos::View<rtype *[N_CONSERVATIVE][N_DIM]> g = viscous_gradients;
    Kokkos::View<rtype **> out = vortex_fields;
    gated::parallel_for("vortex_fields", mesh->n_owned(), KOKKOS_LAMBDA(const uint32_t c) {
#if N_DIM == 2
        out(c, 0) = g(c, 2, 0) - g(c, 1, 1);
#else
        for (uint8_t k = 0; k < 3; k++) {
            const uint8_t a = static_cast<uint8_t>((k + 1) % 3), b = static_cast<uint8_t>((k + 2) % 3);
            out(c, k) = g(c, 1 + b, a) - g(c, 1 + a, b);
        }
#endif
    });
}

void Solver::update_derived_fields() {
    derived::Inputs inputs{.W = W_cells,
                           .gradients = viscous_gradients,
                           .gamma = physics.get_gamma(),
                           .radius_coords = axisymmetric ? mesh->cell_coords : Kokkos::View<rtype *[N_DIM]>(),
                           .primitives = primitives};
    if (is_mixture()) {
        inputs.cell_gamma = Kokkos::subview(cell_scalars, Kokkos::ALL(), species_names.size());
        inputs.Y = cell_scalars;
        inputs.thermo = mixture.thermo;
        inputs.flamelet = mixture.flamelet;
        if (derived_selection.need(derived::need::VISCOUS)) {
            update_transport();
            inputs.transport = cell_transport;
        }
        if (derived_selection.need(derived::need::SPECIES_GRADIENTS)) {
            derived::scalar_gradients(derived_settings, cell_scalars, viscous_gradient(), boundary_data.bc_Y,
                                      mesh->n_cells, mesh->n_owned(), derived_workspace);
            inputs.scalar_gradients = derived_workspace.gradients;
        }
    }
    derived::compute(derived_selection, derived_settings, inputs, mesh->n_owned(), derived_fields);
}

void Solver::write_integrals() {
    if (integral_monitor.interval == 0 || step % integral_monitor.interval != 0) return;
    allocate_monitors();
    monitor_reductions(force_monitors.size());
    const auto all = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), monitor_results);
    write_integral_row(all.data() + monitor_offsets[force_monitors.size()]);
}

void Solver::write_integral_row(const double * monitored) {
    std::vector<double> values;
    for (size_t i = 0; i < N_FLOW_STATISTICS; i++) values.push_back(double(static_cast<rtype>(monitored[i])));
    if (integral_monitor.budget) values.insert(values.end(), monitored + N_FLOW_STATISTICS + 2,
                                               monitored + N_FLOW_STATISTICS + 2 + N_BUDGET);
    if (les_on && les.dynamic) values.push_back(double(static_cast<rtype>(monitored[N_FLOW_STATISTICS])));
    if (dynamic_efficiency()) values.push_back(monitored[N_FLOW_STATISTICS + 1]);
    const auto columns = integral_columns();
    integral_monitor.last = {"integrals", step, double(t), {}};
    for (size_t i = 0; i < values.size(); i++) integral_monitor.last.values.emplace_back(columns[i].first, values[i]);
    if (!integral_monitor.out) return;
    *integral_monitor.out << step << "," << std::setprecision(12) << t;
    for (const double v : values) *integral_monitor.out << "," << v;
    *integral_monitor.out << "\n";
    integral_monitor.out->flush();
}

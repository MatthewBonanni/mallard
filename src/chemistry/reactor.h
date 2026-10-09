/**
 * @file reactor.h
 * @author Matthew Bonanni (mbonanni001@gmail.com)
 * @brief Adiabatic constant-volume reactor: the chemistry of one cell over a
 *        splitting step, integrated with RODAS.
 * @version 0.3
 * @date 2026-10-03
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 *
 */

#ifndef CHEMISTRY_REACTOR_H
#define CHEMISTRY_REACTOR_H

#include <Kokkos_Core.hpp>

#include <cstdint>

#include "kinetics.h"
#include "rosenbrock.h"
#include "sparse_lu.h"
#include "thermo.h"

namespace chemistry {

struct ReactorOptions {
    RosenbrockOptions integrator;
    double atol_Y = 1e-10;
    int sparse = -1;  // linear solver: 0 dense LU, 1 sparse LU, -1 automatic (use_sparse_lu)
};

/**
 * @brief Whether a mechanism's reactors use the sparse LU: as options.sparse
 *        says, or automatically from 30 species when the factors of the
 *        static pattern fill at most 60% of the dense matrix (GRI-3.0 53%:
 *        sparse; on an A100 the sparse LU is about 3x faster at 100 species).
 */
inline bool use_sparse_lu(const ReactorOptions & options, const Mechanism & mechanism) {
    if (options.sparse >= 0) return options.sparse == 1;
    const uint32_t n = static_cast<uint32_t>(mechanism.n_species()) + 1;
    if (n < 31) return false;
    const SparseLUPattern<Kokkos::HostSpace> pattern = make_sparse_lu_pattern<Kokkos::HostSpace>(mechanism);
    return pattern.nnz <= 0.6 * static_cast<double>(n) * n;
}

/**
 * @brief Adiabatic, constant-volume reactor at density rho with state
 *        y = (Y_1 .. Y_Ns, T):
 *        dY_k/dt = W_k omega_k / rho,
 *        dT/dt = -sum_k u_k omega_k / (rho cv),
 *        with u_k the molar internal energies. The ODE system of integrate().
 *
 * With a forcing g (n_species + 1 values), the reactor also receives the
 * constant sources rho g_k of its partial densities and rho g_e of its
 * internal energy per volume, rho being the initial density: y_k = rho_k /
 * rho then sums to the current density over rho, and
 *        dy_k/dt = W_k omega_k / rho + g_k,
 *        dT/dt = (g_e - sum_k e_k g_k - sum_k u_k omega_k / rho) / sum_k y_k cv_k,
 * still autonomous, with the species rows of the Jacobian unchanged. Over a
 * call of length forcing_time, the exact solution can fall below zero by
 * forcing_time max(0, -g_k) at most where the forcing removes a species
 * faster than chemistry restores it, so that much is admissible.
 */
template <typename MemorySpace = Kokkos::DefaultExecutionSpace::memory_space>
struct ConstantVolumeReactor {
    const ThermoTable<MemorySpace> & thermo;
    const KineticsTable<MemorySpace> & kinetics;
    double rho;
    double atol_Y;
    double * scratch;   // scratch_size(kinetics) doubles
    double * rank_one;  // null: J in full; else (n_species + 1) doubles: J = J_s + u v^T with u here, v_j = 1 / W_j
    const SparseLUPattern<MemorySpace> * sparse = nullptr;  // with rank_one: J_s in the pattern's compact layout
    const double * forcing = nullptr;  // null, or g_1 .. g_Ns, g_e
    double forcing_time = 0.0;         // length of the call (with forcing)

    KOKKOS_INLINE_FUNCTION
    static uint32_t scratch_size(const KineticsTable<MemorySpace> & kinetics) {
        return 6 * kinetics.n_species + kinetics.n_reactions + kinetics.derivatives_size() +
               static_cast<uint32_t>(kinetics.chunk_end.extent(0));
    }

    KOKKOS_INLINE_FUNCTION uint32_t size() const { return thermo.n_species + 1; }

    KOKKOS_INLINE_FUNCTION double atol(const uint32_t i) const { return i < thermo.n_species ? atol_Y : 0.0; }

    template <typename Lanes>
    KOKKOS_INLINE_FUNCTION bool admissible(const Lanes & lanes, const double * y) const {
        if (forcing) {
            const double negative = lanes.sum(thermo.n_species, [&](const uint32_t k) {
                return y[k] < -atol_Y - forcing_time * Kokkos::fmax(0.0, -forcing[k]) ? 1.0 : 0.0;
            });
            return negative == 0.0 && y[thermo.n_species] > 0.0;
        }
        const double negative =
            lanes.sum(thermo.n_species, [&](const uint32_t k) { return y[k] < -atol_Y ? 1.0 : 0.0; });
        return negative == 0.0 && y[thermo.n_species] > 0.0;
    }

    template <typename Lanes>
    KOKKOS_INLINE_FUNCTION void rhs(const Lanes & lanes, const double * y, double * f) const {
        evaluate(lanes, y, f, nullptr);
    }

    template <typename Lanes>
    KOKKOS_INLINE_FUNCTION void rhs_jacobian(const Lanes & lanes, const double * y, double * f, double * J) const {
        evaluate(lanes, y, f, J);
    }

    /** @brief d(sum_k e_k g_k) / dT = sum_k cv_k g_k. */
    template <typename Lanes>
    KOKKOS_INLINE_FUNCTION double forcing_temperature_derivative(const Lanes & lanes, const double * cp_R) const {
        return GAS_CONSTANT *
               lanes.sum(thermo.n_species, [&](const uint32_t k) { return (cp_R[k] - 1.0) * thermo.inv_W(k) * forcing[k]; });
    }

    /**
     * @brief f(y) and, if J is not null, the Jacobian df/dy (in the lanes'
     *        layout, see dense_index), from d omega / dC and d q / dT by the
     *        chain rule (C_k = rho Y_k / W_k).
     */
    /**
     * @brief The Jacobian in the sparse pattern's compact layout (J_s, with
     *        u in rank_one): the species block's entries, the T column and
     *        the T row, by the chain rule as in evaluate().
     */
    template <typename Lanes, typename Powers>
    KOKKOS_INLINE_FUNCTION void compact_jacobian(const Lanes & lanes, const double * y, double * J, const double T,
                                                 const Powers & p, const double cv, const double inv_rho_cv,
                                                 const double dT_dt, const double * h_RT, const double * cp_R,
                                                 const double * omega, const double * domega_dT,
                                                 const ReactionDerivatives & d) const {
        const uint32_t ns = thermo.n_species, ne = sparse->n_entries;
        double * T_column = J + ne;
        double * T_row = T_column + ns;
        kinetics.production_jacobian_entries(lanes, d, J, rank_one);
        const double sum_a =
            lanes.sum(ns, [&](const uint32_t k) { return GAS_CONSTANT * T * (h_RT[k] - 1.0) * rank_one[k]; });
        lanes.for_each(ns, [&](const uint32_t j) {
            double sum = sum_a;
            for (uint32_t c = sparse->column_offset(j); c < sparse->column_offset(j + 1); c++) {
                const uint32_t e = sparse->column_entry(c);
                sum += GAS_CONSTANT * T * (h_RT[sparse->entry_row(e)] - 1.0) * J[e];
            }
            const double cv_j = GAS_CONSTANT * (cp_R[j] - 1.0) * thermo.inv_W(j);
            T_row[j] = -sum * rho * thermo.inv_W(j) * inv_rho_cv - dT_dt * cv_j / cv;
        });
        lanes.sync();
        const double d_sum_dT = lanes.sum(ns, [&](const uint32_t k) {
            return GAS_CONSTANT * ((cp_R[k] - 1.0) * omega[k] + T * (h_RT[k] - 1.0) * domega_dT[k]);
        });
        const double dcv_dT =
            GAS_CONSTANT * lanes.sum(ns, [&](const uint32_t k) { return y[k] * thermo.inv_W(k) * thermo.dcp_R_dT(k, p); });
        if (forcing) {
            const double d_forcing_dT = forcing_temperature_derivative(lanes, cp_R);
            lanes.single([&]() { T_row[ns] = -d_sum_dT * inv_rho_cv - dT_dt * dcv_dT / cv - d_forcing_dT / cv; });
        } else {
            lanes.single([&]() { T_row[ns] = -d_sum_dT * inv_rho_cv - dT_dt * dcv_dT / cv; });
        }
        lanes.for_each(ne, [&](const uint32_t e) {
            J[e] *= thermo.inv_W(sparse->entry_column(e)) / thermo.inv_W(sparse->entry_row(e));
        });
        lanes.for_each(ns, [&](const uint32_t k) {
            const double W_k = 1.0 / thermo.inv_W(k);
            T_column[k] = W_k * domega_dT[k] / rho;
            rank_one[k] *= W_k;  // u_k = W_k a_k, with v_j = 1 / W_j
        });
        lanes.single([&]() { rank_one[ns] = 0.0; });
        lanes.sync();
    }

    template <typename Lanes>
    KOKKOS_INLINE_FUNCTION void evaluate(const Lanes & lanes, const double * y, double * f, double * J) const {
        const uint32_t ns = thermo.n_species, nr = kinetics.n_reactions, n = ns + 1;
        double * C = scratch;
        double * g_RT = C + ns;
        double * h_RT = g_RT + ns;
        double * cp_R = h_RT + ns;
        double * omega = cp_R + ns;
        double * domega_dT = omega + ns;
        double * q = domega_dT + ns;
        const ReactionDerivatives d =
            ReactionDerivatives::at(q + nr, nr, static_cast<uint32_t>(kinetics.forward_species.extent(0)),
                                    static_cast<uint32_t>(kinetics.reverse_species.extent(0)));
        double * partial = q + nr + kinetics.derivatives_size();

        const double T = y[ns];
        const auto p = ThermoTable<MemorySpace>::powers(T);
        lanes.for_each(ns, [&](const uint32_t k) {
            C[k] = rho * y[k] * thermo.inv_W(k);
            h_RT[k] = thermo.h_RT(k, p);
            g_RT[k] = h_RT[k] - thermo.s_R(k, p);
            cp_R[k] = thermo.cp_R(k, p);
        });
        lanes.sync();
        const double cv =
            GAS_CONSTANT * lanes.sum(ns, [&](const uint32_t k) { return y[k] * thermo.inv_W(k) * (cp_R[k] - 1.0); });
        const double C_total = lanes.sum(ns, [&](const uint32_t k) { return C[k]; });
        kinetics.rates_of_progress(lanes, T, C, C_total, g_RT, h_RT, q, J ? &d : nullptr);
        kinetics.production_rates(lanes, q, omega, partial);
        // u_k = R T (h_k / RT - 1)
        lanes.for_each(ns, [&](const uint32_t k) { f[k] = omega[k] / (rho * thermo.inv_W(k)); });
        const double sum_u_omega =
            lanes.sum(ns, [&](const uint32_t k) { return GAS_CONSTANT * T * (h_RT[k] - 1.0) * omega[k]; });
        const double inv_rho_cv = 1.0 / (rho * cv);
        double dT_dt = -sum_u_omega * inv_rho_cv;
        if (forcing) {
            lanes.sync();
            lanes.for_each(ns, [&](const uint32_t k) { f[k] += forcing[k]; });
            const double sum_e_g = lanes.sum(
                ns, [&](const uint32_t k) { return GAS_CONSTANT * T * (h_RT[k] - 1.0) * thermo.inv_W(k) * forcing[k]; });
            dT_dt += (forcing[ns] - sum_e_g) / cv;
        }
        lanes.single([&]() { f[ns] = dT_dt; });
        lanes.sync();
        if (!J) return;

        kinetics.production_rates(lanes, d.dq_dT, domega_dT, partial);
        if (sparse) {
            compact_jacobian(lanes, y, J, T, p, cv, inv_rho_cv, dT_dt, h_RT, cp_R, omega, domega_dT, d);
            return;
        }
        kinetics.production_jacobian(lanes, d, J, n, rank_one);
        // With the rank-one part apart, its share of the T row: sum_k u_k a_k
        const double sum_a =
            rank_one ? lanes.sum(ns, [&](const uint32_t k) { return GAS_CONSTANT * T * (h_RT[k] - 1.0) * rank_one[k]; })
                     : 0.0;

        // T row: d(sum u_k omega_k)/dY_j = rho / W_j sum_k u_k dw_kj; d cv / dY_j = cv_j
        constexpr bool CM = Lanes::column_major;
        auto Jat = [&](const uint32_t r, const uint32_t c) -> double & { return J[dense_index<CM>(n, r, c)]; };
        lanes.for_each(ns, [&](const uint32_t j) {
            double sum = sum_a;
            for (uint32_t k = 0; k < ns; k++) sum += GAS_CONSTANT * T * (h_RT[k] - 1.0) * Jat(k, j);
            const double cv_j = GAS_CONSTANT * (cp_R[j] - 1.0) * thermo.inv_W(j);
            Jat(ns, j) = -sum * rho * thermo.inv_W(j) * inv_rho_cv - dT_dt * cv_j / cv;
        });
        lanes.sync();
        const double d_sum_dT = lanes.sum(ns, [&](const uint32_t k) {
            return GAS_CONSTANT * ((cp_R[k] - 1.0) * omega[k] + T * (h_RT[k] - 1.0) * domega_dT[k]);
        });
        const double dcv_dT =
            GAS_CONSTANT * lanes.sum(ns, [&](const uint32_t k) { return y[k] * thermo.inv_W(k) * thermo.dcp_R_dT(k, p); });
        if (forcing) {
            const double d_forcing_dT = forcing_temperature_derivative(lanes, cp_R);
            lanes.single([&]() { Jat(ns, ns) = -d_sum_dT * inv_rho_cv - dT_dt * dcv_dT / cv - d_forcing_dT / cv; });
        } else {
            lanes.single([&]() { Jat(ns, ns) = -d_sum_dT * inv_rho_cv - dT_dt * dcv_dT / cv; });
        }

        // Species rows: df_k/dY_j = W_k / W_j dw_kj, df_k/dT = W_k / rho domega_k/dT
        lanes.for_each(ns, [&](const uint32_t k) {
            const double W_k = 1.0 / thermo.inv_W(k);
            for (uint32_t j = 0; j < ns; j++) Jat(k, j) *= W_k * thermo.inv_W(j);
            Jat(k, ns) = W_k * domega_dT[k] / rho;
            if (rank_one) rank_one[k] *= W_k;  // u_k = W_k a_k, with v_j = 1 / W_j
        });
        if (rank_one) lanes.single([&]() { rank_one[ns] = 0.0; });
        lanes.sync();
    }
};

/**
 * @brief Doubles of the part of advance_reactor()'s work memory that can be
 *        in fast (team scratch) memory: the state, the factors and the
 *        integrator's vectors, the most used in the factorization and solves.
 */
template <typename MemorySpace>
KOKKOS_INLINE_FUNCTION uint32_t reactor_fast_size(const KineticsTable<MemorySpace> & kinetics,
                                                  const SparseLUPattern<MemorySpace> * sparse = nullptr) {
    const uint32_t n = kinetics.n_species + 1;
    const uint32_t solver = sparse ? sparse->work_size() + 1 : n * n;
    return n + solver + rosenbrock_vectors_size(n);
}

/**
 * @brief Doubles of work memory advance_reactor() needs, the linear solver's
 *        included: dense, or sparse with a SparseLUPattern.
 */
template <typename MemorySpace>
KOKKOS_INLINE_FUNCTION uint32_t reactor_work_size(const KineticsTable<MemorySpace> & kinetics,
                                                  const SparseLUPattern<MemorySpace> * sparse = nullptr) {
    const uint32_t n = kinetics.n_species + 1;
    return (sparse ? sparse->jacobian_size() : n * n) + ConstantVolumeReactor<MemorySpace>::scratch_size(kinetics) +
           reactor_fast_size(kinetics, sparse);
}

/**
 * @brief Advance one adiabatic constant-volume reactor over dt: integrate
 *        (Y, T), then clip negative mass fractions, renormalize, and take T
 *        from the initial internal energy (so energy is exact). Negative
 *        initial mass fractions are clipped the same way before integrating.
 * @param Y Mass fractions (n_species), T temperature: updated in place.
 * @param h Sub-step size: first guess in, proposal for the next call out.
 * @param work reactor_work_size(kinetics, sparse) doubles (with fast, only the
 *        Jacobian and the reactor's scratch are used).
 * @param pivot n_species + 1 integers (dense LU).
 * @param sparse The pattern of the sparse LU (Sparse = true; else unused).
 * @param fast Null, or reactor_fast_size(kinetics, sparse) doubles of faster
 *        memory for the rest of the work memory.
 * @param forcing Null, or the constant sources g_1 .. g_Ns, g_e per unit
 *        initial density (see ConstantVolumeReactor): Y then holds the
 *        mass fractions at the end, at density rho (1 + dt sum_k g_k), and T
 *        follows from the internal energy e + dt g_e per unit initial mass.
 */
template <bool Sparse = false, typename Lanes, typename MemorySpace, typename Observer = NoObserver>
KOKKOS_INLINE_FUNCTION RosenbrockResult advance_reactor(const Lanes & lanes, const ThermoTable<MemorySpace> & thermo,
                                                        const KineticsTable<MemorySpace> & kinetics, const double rho,
                                                        const double dt, double * Y, double & T, double & h,
                                                        const ReactorOptions & options, double * work,
                                                        uint32_t * pivot, Observer && observer = Observer(),
                                                        const SparseLUPattern<MemorySpace> * sparse = nullptr,
                                                        double * fast = nullptr, const double * forcing = nullptr) {
    const uint32_t ns = thermo.n_species, n = ns + 1;
    double * J = work;
    double * scratch = J + (Sparse ? sparse->jacobian_size() : n * n);
    double * y = fast ? fast : scratch + ConstantVolumeReactor<MemorySpace>::scratch_size(kinetics);
    double * solver_work = y + n;
    double * vectors = solver_work + (Sparse ? sparse->work_size() + 1 : n * n);
    // Scalar code runs on every lane with the same values; writes go through for_each or single
    double e, cv;
    thermo.e_cv(T, MassFractions{Y}, e, cv);
    lanes.for_each(ns, [&](const uint32_t k) { y[k] = Kokkos::fmax(Y[k], 0.0); });
    lanes.sync();
    const double negative = lanes.sum(ns, [&](const uint32_t k) { return Y[k] < 0.0 ? 1.0 : 0.0; });
    double T0 = T;
    if (negative > 0.0) {
        // Small negative mass fractions (e.g. from explicit diffusion) would
        // reject every sub-step: start from the clipped composition at the same energy
        const double sum = lanes.sum(ns, [&](const uint32_t k) { return y[k]; });
        lanes.for_each(ns, [&](const uint32_t k) { y[k] /= sum; });
        lanes.sync();
        T0 = thermo.T_from_e(e, MassFractions{y}, T);
    }
    lanes.single([&]() { y[ns] = T0; });
    lanes.sync();
    RosenbrockResult result;
    if constexpr (Sparse) {
        // LU values, z = A_s^-1 u, u, x, beta
        double * values = solver_work;
        double * z = values + sparse->nnz;
        double * u = z + n;
        double * x = u + n;
        const ConstantVolumeReactor<MemorySpace> reactor{thermo,  kinetics, rho,    options.atol_Y, scratch,
                                                         u,       sparse,   forcing, dt};
        const SparseLU<MemorySpace> solver{*sparse, values, z, u, x, x + n};
        result = integrate(lanes, reactor, solver, 0.0, dt, y, h, options.integrator, J, vectors,
                           static_cast<Observer &&>(observer));
    } else {
        const ConstantVolumeReactor<MemorySpace> reactor{thermo,  kinetics, rho,     options.atol_Y, scratch,
                                                         nullptr, nullptr,  forcing, dt};
        const DenseLU solver{n, solver_work, pivot};
        result = integrate(lanes, reactor, solver, 0.0, dt, y, h, options.integrator, J, vectors,
                           static_cast<Observer &&>(observer));
    }
    lanes.for_each(ns, [&](const uint32_t k) { y[k] = Kokkos::fmax(y[k], 0.0); });
    lanes.sync();
    const double total = lanes.sum(ns, [&](const uint32_t k) { return y[k]; });
    lanes.for_each(ns, [&](const uint32_t k) { Y[k] = y[k] / total; });
    lanes.sync();
    if (forcing) {
        const double growth = 1.0 + dt * lanes.sum(ns, [&](const uint32_t k) { return forcing[k]; });
        T = thermo.T_from_e((e + dt * forcing[ns]) / growth, MassFractions{Y}, y[ns]);
        return result;
    }
    T = thermo.T_from_e(e, MassFractions{Y}, y[ns]);
    return result;
}

/** @brief advance_reactor() by one thread. */
template <typename MemorySpace, typename Observer = NoObserver>
KOKKOS_INLINE_FUNCTION RosenbrockResult advance_reactor(const ThermoTable<MemorySpace> & thermo,
                                                        const KineticsTable<MemorySpace> & kinetics, const double rho,
                                                        const double dt, double * Y, double & T, double & h,
                                                        const ReactorOptions & options, double * work,
                                                        uint32_t * pivot, Observer && observer = Observer(),
                                                        const SparseLUPattern<MemorySpace> * sparse = nullptr) {
    if (sparse) {
        return advance_reactor<true>(SerialLanes(), thermo, kinetics, rho, dt, Y, T, h, options, work, pivot,
                                     static_cast<Observer &&>(observer), sparse);
    }
    return advance_reactor<false>(SerialLanes(), thermo, kinetics, rho, dt, Y, T, h, options, work, pivot,
                                  static_cast<Observer &&>(observer), sparse);
}

/**
 * @brief Ignition delay as the time of the maximum of dT/dt: the vertex of
 *        the parabola through the largest sample and its two neighbors. The
 *        reactor has ignited once dT/dt has since fallen below half that
 *        maximum: the thermal runaway is over, however small the temperature
 *        rise (very lean mixtures).
 */
struct IgnitionObserver {
    uint32_t index = 0;     // of T in the state
    double t_offset = 0.0;  // added to the integrator's time
    double t_ignition = 0.0;
    double peak = 0.0;      // largest sampled dT/dt
    double t[3] = {0.0, 0.0, 0.0};
    double g[3] = {0.0, 0.0, 0.0};
    uint32_t count = 0;

    KOKKOS_INLINE_FUNCTION void operator()(const double time, const double *, const double * f) {
        for (int i = 0; i < 2; i++) {
            t[i] = t[i + 1];
            g[i] = g[i + 1];
        }
        t[2] = t_offset + time;
        g[2] = f[index];
        if (++count < 3 || !(g[1] > peak && g[1] >= g[0] && g[1] >= g[2])) return;
        peak = g[1];
        const double num = (t[1] - t[0]) * (t[1] - t[0]) * (g[1] - g[2]) - (t[1] - t[2]) * (t[1] - t[2]) * (g[1] - g[0]);
        const double den = (t[1] - t[0]) * (g[1] - g[2]) - (t[1] - t[2]) * (g[1] - g[0]);
        t_ignition = den != 0.0 ? t[1] - 0.5 * num / den : t[1];
    }

    /** @brief Whether the reactor has ignited, at t_ignition. */
    KOKKOS_INLINE_FUNCTION bool ignited() const { return peak > 0.0 && g[2] < 0.5 * peak; }
};

} // namespace chemistry

#endif // CHEMISTRY_REACTOR_H

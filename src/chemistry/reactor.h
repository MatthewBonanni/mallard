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
#include "thermo.h"

namespace chemistry {

struct ReactorOptions {
    RosenbrockOptions integrator;
    double atol_Y = 1e-10;
};

/**
 * @brief Adiabatic, constant-volume reactor at density rho with state
 *        y = (Y_1 .. Y_Ns, T):
 *        dY_k/dt = W_k omega_k / rho,
 *        dT/dt = -sum_k u_k omega_k / (rho cv),
 *        with u_k the molar internal energies. The ODE system of integrate().
 */
template <typename MemorySpace = Kokkos::DefaultExecutionSpace::memory_space>
struct ConstantVolumeReactor {
    const ThermoTable<MemorySpace> & thermo;
    const KineticsTable<MemorySpace> & kinetics;
    double rho;
    double atol_Y;
    double * scratch;  // scratch_size(n_species, n_reactions) doubles

    KOKKOS_INLINE_FUNCTION
    static constexpr uint32_t scratch_size(const uint32_t ns, const uint32_t nr) { return 6 * ns + 4 * nr; }

    KOKKOS_INLINE_FUNCTION uint32_t size() const { return thermo.n_species + 1; }

    KOKKOS_INLINE_FUNCTION double atol(const uint32_t i) const { return i < thermo.n_species ? atol_Y : 0.0; }

    KOKKOS_INLINE_FUNCTION bool admissible(const double * y) const {
        for (uint32_t k = 0; k < thermo.n_species; k++) {
            if (y[k] < -atol_Y) return false;
        }
        return y[thermo.n_species] > 0.0;
    }

    KOKKOS_INLINE_FUNCTION void rhs(const double * y, double * f) const { evaluate(y, f, nullptr); }

    KOKKOS_INLINE_FUNCTION void rhs_jacobian(const double * y, double * f, double * J) const { evaluate(y, f, J); }

    /**
     * @brief f(y) and, if J is not null, the row-major Jacobian df/dy, from
     *        d omega / dC and d q / dT by the chain rule (C_k = rho Y_k / W_k).
     */
    KOKKOS_INLINE_FUNCTION void evaluate(const double * y, double * f, double * J) const {
        const uint32_t ns = thermo.n_species, nr = kinetics.n_reactions, n = ns + 1;
        double * C = scratch;
        double * g_RT = C + ns;
        double * h_RT = g_RT + ns;
        double * cp_R = h_RT + ns;
        double * omega = cp_R + ns;
        double * domega_dT = omega + ns;
        double * q = domega_dT + ns;
        double * dq_dT = q + nr;
        double * kf = dq_dT + nr;
        double * kr = kf + nr;

        const double T = y[ns];
        const auto p = ThermoTable<MemorySpace>::powers(T);
        double cv = 0.0;
        for (uint32_t k = 0; k < ns; k++) {
            C[k] = rho * y[k] * thermo.inv_W(k);
            h_RT[k] = thermo.h_RT(k, p);
            g_RT[k] = h_RT[k] - thermo.s_R(k, p);
            cp_R[k] = thermo.cp_R(k, p);
            cv += y[k] * thermo.inv_W(k) * (cp_R[k] - 1.0);
        }
        cv *= GAS_CONSTANT;
        kinetics.rates_of_progress(T, C, g_RT, h_RT, q, J ? dq_dT : nullptr, J ? kf : nullptr, J ? kr : nullptr);
        kinetics.production_rates(q, omega);
        // u_k = R T (h_k / RT - 1)
        double sum_u_omega = 0.0;
        for (uint32_t k = 0; k < ns; k++) {
            f[k] = omega[k] / (rho * thermo.inv_W(k));
            sum_u_omega += GAS_CONSTANT * T * (h_RT[k] - 1.0) * omega[k];
        }
        const double inv_rho_cv = 1.0 / (rho * cv);
        const double dT_dt = -sum_u_omega * inv_rho_cv;
        f[ns] = dT_dt;
        if (!J) return;

        kinetics.production_rates(dq_dT, domega_dT);
        // d omega / dC into the top-left of J with row stride ns, then spread to stride n
        kinetics.production_jacobian(T, C, kf, kr, J);
        for (uint32_t a = ns * ns; a-- > 0;) J[(a / ns) * n + a % ns] = J[a];

        // T row: d(sum u_k omega_k)/dY_j = rho / W_j sum_k u_k dw_kj; d cv / dY_j = cv_j
        for (uint32_t j = 0; j < ns; j++) {
            double sum = 0.0;
            for (uint32_t k = 0; k < ns; k++) sum += GAS_CONSTANT * T * (h_RT[k] - 1.0) * J[k * n + j];
            const double cv_j = GAS_CONSTANT * (cp_R[j] - 1.0) * thermo.inv_W(j);
            J[ns * n + j] = -sum * rho * thermo.inv_W(j) * inv_rho_cv - dT_dt * cv_j / cv;
        }
        double d_sum_dT = 0.0, dcv_dT = 0.0;
        for (uint32_t k = 0; k < ns; k++) {
            d_sum_dT += GAS_CONSTANT * ((cp_R[k] - 1.0) * omega[k] + T * (h_RT[k] - 1.0) * domega_dT[k]);
            dcv_dT += y[k] * thermo.inv_W(k) * thermo.dcp_R_dT(k, p);
        }
        dcv_dT *= GAS_CONSTANT;
        J[ns * n + ns] = -d_sum_dT * inv_rho_cv - dT_dt * dcv_dT / cv;

        // Species rows: df_k/dY_j = W_k / W_j dw_kj, df_k/dT = W_k / rho domega_k/dT
        for (uint32_t k = 0; k < ns; k++) {
            const double W_k = 1.0 / thermo.inv_W(k);
            for (uint32_t j = 0; j < ns; j++) J[k * n + j] *= W_k * thermo.inv_W(j);
            J[k * n + ns] = W_k * domega_dT[k] / rho;
        }
    }
};

/**
 * @brief Ignition delay as the time of the maximum of dT/dt: the vertex of
 *        the parabola through the largest sample and its two neighbors.
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
};

/** @brief Doubles of work memory advance_reactor() needs. */
KOKKOS_INLINE_FUNCTION
constexpr uint32_t reactor_work_size(const uint32_t ns, const uint32_t nr) {
    return rosenbrock_work_size(ns + 1) + ConstantVolumeReactor<>::scratch_size(ns, nr) + ns + 1;
}

/**
 * @brief Advance one adiabatic constant-volume reactor over dt: integrate
 *        (Y, T), then clip negative mass fractions, renormalize, and take T
 *        from the initial internal energy (so energy is exact). Negative
 *        initial mass fractions are clipped the same way before integrating.
 * @param Y Mass fractions (n_species), T temperature: updated in place.
 * @param h Sub-step size: first guess in, proposal for the next call out.
 * @param work reactor_work_size(n_species, n_reactions) doubles.
 * @param pivot n_species + 1 integers.
 */
template <typename MemorySpace, typename Observer = NoObserver>
KOKKOS_INLINE_FUNCTION RosenbrockResult advance_reactor(const ThermoTable<MemorySpace> & thermo,
                                                        const KineticsTable<MemorySpace> & kinetics, const double rho,
                                                        const double dt, double * Y, double & T, double & h,
                                                        const ReactorOptions & options, double * work,
                                                        uint32_t * pivot, Observer && observer = Observer()) {
    const uint32_t ns = thermo.n_species, n = ns + 1;
    double * y = work;
    double * scratch = y + n;
    double * integrator_work = scratch + ConstantVolumeReactor<MemorySpace>::scratch_size(ns, kinetics.n_reactions);
    double e, cv;
    thermo.e_cv(T, MassFractions{Y}, e, cv);
    double sum = 0.0;
    bool negative = false;
    for (uint32_t k = 0; k < ns; k++) {
        y[k] = Kokkos::fmax(Y[k], 0.0);
        negative = negative || Y[k] < 0.0;
        sum += y[k];
    }
    y[ns] = T;
    if (negative) {
        // Small negative mass fractions (e.g. from explicit diffusion) would
        // reject every sub-step: start from the clipped composition at the same energy
        for (uint32_t k = 0; k < ns; k++) y[k] /= sum;
        y[ns] = thermo.T_from_e(e, MassFractions{y}, T);
    }
    const ConstantVolumeReactor<MemorySpace> reactor{thermo, kinetics, rho, options.atol_Y, scratch};
    const RosenbrockResult result = integrate(reactor, 0.0, dt, y, h, options.integrator, integrator_work, pivot,
                                              static_cast<Observer &&>(observer));
    sum = 0.0;
    for (uint32_t k = 0; k < ns; k++) {
        Y[k] = Kokkos::fmax(y[k], 0.0);
        sum += Y[k];
    }
    for (uint32_t k = 0; k < ns; k++) Y[k] /= sum;
    T = thermo.T_from_e(e, MassFractions{Y}, y[ns]);
    return result;
}

} // namespace chemistry

#endif // CHEMISTRY_REACTOR_H

/**
 * @file inflow.h
 * @author Matthew Bonanni (mbonanni001@gmail.com)
 * @brief Boundary states of subsonic inlets given stagnation conditions or
 *        for calorically and thermally perfect gases.
 * @version 0.1
 * @date 2026-10-15
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 *
 */

#ifndef INFLOW_H
#define INFLOW_H

#include <Kokkos_Core.hpp>

#include "common.h"
#include "thermo.h"

namespace inflow {

/** @brief Isentrope of a calorically perfect gas through (T0, p0). */
struct PerfectGas {
    double gamma, R;

    KOKKOS_INLINE_FUNCTION double cp(const double) const { return gamma * R / (gamma - 1.0); }
    KOKKOS_INLINE_FUNCTION double h(const double T) const { return cp(T) * T; }
    /** @brief ln(p / p0) along the isentrope. */
    KOKKOS_INLINE_FUNCTION double log_p_ratio(const double T, const double T0) const {
        return gamma / (gamma - 1.0) * Kokkos::log(T / T0);
    }
};

/** @brief Isentrope of a thermally perfect mixture of fixed composition y. */
template <typename Table, typename F_Y>
struct ThermallyPerfectGas {
    const Table & thermo;
    F_Y y;
    double R;

    KOKKOS_INLINE_FUNCTION double cp(const double T) const { return thermo.cp_mass(T, y); }
    KOKKOS_INLINE_FUNCTION double h(const double T) const { return thermo.h_mass(T, y); }
    KOKKOS_INLINE_FUNCTION double log_p_ratio(const double T, const double T0) const {
        return (thermo.s0_mass(T, y) - thermo.s0_mass(T0, y)) / R;
    }
};

/**
 * @brief Subsonic inflow state at stagnation temperature T0 and pressure p0
 *        flowing along the unit vector d (zero: along -n), given the interior
 *        state W_i: the outgoing Riemann invariant u_n + 2 c / (gamma_c - 1)
 *        of W_i, with c^2 = gamma_c p / rho on both sides, fixes the speed;
 *        the state lies on the isentrope through (T0, p0) with
 *        h(T) + |u|^2 / 2 = h(T0). gamma_c is the ratio of specific heats at
 *        T0 (exact for a calorically perfect gas). Interiors flowing out, or
 *        at a pressure above p0, see the stagnation state at rest. Supersonic
 *        inflow keeps the interior speed, supersonic outflow the interior
 *        state.
 * @param T_b Temperature of the boundary state (output).
 */
template <typename Gas>
KOKKOS_INLINE_FUNCTION void stagnation_state(const Gas & gas, const double gamma_c, const double T0, const double p0,
                                             const rtype * d, const rtype * W_i, const rtype * n, rtype * W_g,
                                             double & T_b) {
    constexpr uint8_t E = N_DIM + 1;
    double dir[N_DIM], d2 = 0.0;
    FOR_I_DIM d2 += double(d[i]) * double(d[i]);
    FOR_I_DIM dir[i] = (d2 > 0.0) ? double(d[i]) : -double(n[i]);
    double c_dn = 0.0, u_n = 0.0, u2 = 0.0;
    FOR_I_DIM {
        c_dn += dir[i] * double(n[i]);
        u_n += double(W_i[1 + i]) * double(n[i]);
        u2 += double(W_i[1 + i]) * double(W_i[1 + i]);
    }
    c_dn = Kokkos::fmin(c_dn, 0.0);
    const double a_i = Kokkos::sqrt(gamma_c * double(W_i[E]) / double(W_i[0]));
    if (u_n >= a_i) {
        for (uint8_t i = 0; i < N_DIM + 2; i++) W_g[i] = W_i[i];
        T_b = double(W_i[E]) / (double(W_i[0]) * gas.R);
        return;
    }
    const double h0 = gas.h(T0);
    const double T_low = 1e-3 * T0;
    const double speed_max2 = 2.0 * (h0 - gas.h(T_low));
    auto speed = [&](const double T) { return Kokkos::sqrt(Kokkos::fmax(2.0 * (h0 - gas.h(T)), 0.0)); };
    double T = T0;
    if (u_n <= -a_i) {
        // Supersonic inflow: T from h(T) = h0 - |u_i|^2 / 2
        const double target = h0 - 0.5 * Kokkos::fmin(u2, speed_max2);
        double lo = T_low, hi = T0;
        T = Kokkos::fmin(Kokkos::fmax(double(W_i[E]) / (double(W_i[0]) * gas.R), lo), hi);
        for (int it = 0; it < 100; it++) {
            const double f = gas.h(T) - target;
            if (f > 0.0) hi = T; else lo = T;
            double T_new = T - f / gas.cp(T);
            if (!(T_new > lo && T_new < hi)) T_new = 0.5 * (lo + hi);
            const bool done = Kokkos::fabs(T_new - T) <= 1e-13 * T0;
            T = T_new;
            if (done) break;
        }
    } else {
        // f(T) = c_dn V(T) + 2 c(T) / (gamma_c - 1) - J rises with T on [T_low, T0]
        const double J = u_n + 2.0 * a_i / (gamma_c - 1.0);
        const double k = 2.0 * Kokkos::sqrt(gamma_c * gas.R) / (gamma_c - 1.0);
        auto f = [&](const double t) { return c_dn * speed(t) + k * Kokkos::sqrt(t) - J; };
        if (f(T0) > 0.0) {
            double lo = T_low, hi = T0;
            T = Kokkos::fmin(Kokkos::fmax(double(W_i[E]) / (double(W_i[0]) * gas.R), lo), hi);
            for (int it = 0; it < 100; it++) {
                const double V = speed(T);
                const double r = f(T);
                if (r > 0.0) hi = T; else lo = T;
                const double slope = (V > 0.0 ? -c_dn * gas.cp(T) / V : 0.0) + 0.5 * k / Kokkos::sqrt(T);
                double T_new = T - r / slope;
                if (!(T_new > lo && T_new < hi)) T_new = 0.5 * (lo + hi);
                const bool done = Kokkos::fabs(T_new - T) <= 1e-13 * T0;
                T = T_new;
                if (done) break;
            }
        }
    }
    const double V = speed(T);
    const double p = p0 * Kokkos::exp(gas.log_p_ratio(T, T0));
    W_g[0] = static_cast<rtype>(p / (gas.R * T));
    FOR_I_DIM W_g[1 + i] = static_cast<rtype>(V * dir[i]);
    W_g[E] = static_cast<rtype>(p);
    T_b = T;
}

} // namespace inflow

#endif // INFLOW_H

/**
 * @file particle_forces.h
 * @author Matthew Bonanni (mbonanni001@gmail.com)
 * @brief Forces on and heat transfer to solid particles: drag correlations,
 *        gravity and buoyancy, the pressure-gradient, added-mass and Saffman
 *        lift forces, Ranz-Marshall heat transfer, and the exponential
 *        integration of the relaxations stiffer than the step. Design:
 *        docs/design/particles.md.
 * @version 0.1
 * @date 2026-11-20
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 *
 */

#ifndef PARTICLE_FORCES_H
#define PARTICLE_FORCES_H

#include <cstdint>

#include <Kokkos_Core.hpp>

#include "common.h"
#include "physics.h"

namespace particles {

/** @brief Drag correlation of a sphere: f = C_D Re / 24, the drag over Stokes drag. */
enum class Drag : uint8_t { STOKES, SCHILLER_NAUMANN, SINGH };

/**
 * @brief C_D Re / 24 of Singh et al. (AIAA J. 60(2), 2022, Appendix C), from
 *        continuum to free-molecular flow and from subsonic to hypersonic slip:
 *        at the slip Reynolds number Re, slip Mach number M, Knudsen number Kn
 *        (mean free path over the diameter, Kn = sqrt(gamma pi / 2) M / Re),
 *        particle over gas temperature Tp_T and heat capacity ratio gamma.
 *        Below Mach 1 there is no shock, and the post-shock state is the gas's.
 *        Written in Kn rather than M / Re so that it is finite at zero slip.
 */
KOKKOS_INLINE_FUNCTION rtype singh_drag_factor(const rtype Re, const rtype M, const rtype Kn, const rtype Tp_T,
                                               const rtype gamma) {
    constexpr rtype omega = 0.74_r, delta0 = 9.4_r, alpha0 = 0.356_r, A1 = 2.514_r, A2 = 0.8_r, A3 = 0.55_r,
                    eta = 1.8_r, alpha_hoc = 1.27_r, Cd_hypersonic = 0.9_r;
    constexpr rtype C0 = 24.0_r / (delta0 * delta0);
    const rtype pi = Kokkos::numbers::pi_v<rtype>;
    const rtype gm1 = gamma - 1.0_r, exponent = gamma / gm1;
    // Normal shock, curved by alpha (Eqs. 23, 25, 26, C 12)
    rtype Us_U = 1.0_r, Ms2 = M * M, Ts_T = 1.0_r, alpha = 1.0_r, pressure = 0.0_r;
    if (M > 1.0_r) {
        const rtype M2 = M * M;
        Us_U = (2.0_r + gm1 * M2) / ((gamma + 1.0_r) * M2);
        Ms2 = (gm1 * M2 + 2.0_r) / (2.0_r * gamma * M2 - gm1);
        Ts_T = (gm1 * M2 + 2.0_r) * (2.0_r * gamma * M2 - gm1) / ((gamma + 1.0_r) * (gamma + 1.0_r) * M2);
        alpha = 1.0_r / (alpha0 * M + 1.0_r - alpha0);
        const rtype C1 = (Cd_hypersonic - C0 * Kokkos::pow(1.0_r + gm1 * gm1 / (4.0_r * gamma), exponent)) /
                         (1.0_r - gm1 / ((gamma + 1.0_r) * alpha0 * M));
        pressure = C1 * (1.0_r - alpha * Us_U);
    }
    // Continuum (C 8, C 10), times Re: Re / Re~_s = (alpha^2 Ts / T)^omega Theta_s^-p
    const rtype theta_s = Kokkos::pow(1.0_r + 0.5_r * gm1 * Ms2, exponent);
    const rtype p = (gamma + 1.0_r) / (2.0_r * gamma) - gm1 * omega / gamma;
    const rtype Re_ratio = Kokkos::pow(alpha * alpha * Ts_T, omega) * Kokkos::pow(theta_s, -p);
    const rtype root = Kokkos::sqrt(Re) + delta0 * Kokkos::sqrt(Re_ratio);
    const rtype continuum = pressure * Re + C0 * theta_s * root * root;
    // Rarefaction (C 3 - C 5) and the bridge to free-molecular flow (C 1, C 2)
    const rtype Mw = Kokkos::pow(M, 2.0_r * omega - 1.0_r);
    const rtype wall = Kokkos::pow(1.0_r + Tp_T / Ts_T, omega);
    const rtype Wr_T = Kn * Mw * Kokkos::sqrt(2.0_r / (pi * gamma)) * wall;
    const rtype Br = Kn * Kokkos::sqrt(2.0_r / (pi * gamma)) * wall * (Mw + 1.0_r);
    const rtype f_Kn = 1.0_r / ((1.0_r + Kn * (A1 + A2 * Kokkos::exp(-A3 / Kn))) * (1.0_r + alpha_hoc * Wr_T));
    // Free molecular (C 6, C 7) times s, by its series at small s where the terms cancel; Re / s = sqrt(pi) / Kn
    const rtype s = M * Kokkos::sqrt(0.5_r * gamma), s2 = s * s;
    rtype g;
    if (s < 0.3_r) {
        g = (16.0_r / 3.0_r +
             s2 * (16.0_r / 15.0_r + s2 * (-8.0_r / 105.0_r + s2 * (8.0_r / 945.0_r - s2 * 2.0_r / 2079.0_r)))) /
            Kokkos::sqrt(pi);
    } else {
        g = (1.0_r + 2.0_r * s2) * Kokkos::exp(-s2) / (s2 * Kokkos::sqrt(pi)) +
            (4.0_r * s2 * s2 + 4.0_r * s2 - 1.0_r) * Kokkos::erf(s) / (2.0_r * s2 * s);
    }
    const rtype free_molecular = (g + 2.0_r / 3.0_r * Kokkos::sqrt(pi * Tp_T)) * Kokkos::sqrt(pi) / Kn;
    const rtype B = Kokkos::pow(Br, eta);
    return (continuum * f_Kn + free_molecular * B) / (24.0_r * (1.0_r + B));
}

/**
 * @brief The drag correction f = C_D Re / 24 of a correlation that depends on
 *        the slip Reynolds number Re alone (Stokes, Schiller-Naumann).
 */
KOKKOS_INLINE_FUNCTION rtype drag_factor(const Drag drag, const rtype Re) {
    switch (drag) {
        case Drag::SCHILLER_NAUMANN:
            // C_D = 24 / Re (1 + 0.15 Re^0.687) below Re = 1000, 0.44 above
            return Re < 1000.0_r ? 1.0_r + 0.15_r * Kokkos::pow(Re, 0.687_r) : 0.44_r * Re / 24.0_r;
        default: return 1.0_r;
    }
}

/**
 * @brief Exponential integration over a step of y' = (y_eq(t) - y) / tau, with
 *        h = dt / tau frozen at the step's start. A Runge-Kutta scheme with
 *        sum b_i = 1 and sum b_i c_i = 1/2 (every integrator of order 2 or
 *        more) that integrates the stage rates
 *          y' = [decay (y_eq,0 - y_0) + forcing (y_eq(stage) - y_eq,0)] / dt,
 *        of the step's start values y_0, y_eq,0 and the stage's y_eq, gives
 *          y_1 = y_eq,1 - (y_eq,1 - y_eq,0) phi1(h) + (y_0 - y_eq,0) e^{-h},
 *        phi1(h) = (1 - e^{-h}) / h, the exact solution wherever y_eq is linear
 *        in time over the step; likewise the rates
 *          x' = y_eq,0 + carry (y_0 - y_eq,0) + drift (y_eq(stage) - y_eq,0)
 *        integrate x' = y exactly. Forward Euler, whose one stage is the
 *        step's start, is exact for constant y_eq.
 */
struct Exponential {
    rtype decay, forcing, carry, drift;

    KOKKOS_INLINE_FUNCTION explicit Exponential(const rtype h) {
        const rtype e = Kokkos::exp(-h);
        const rtype phi1 = (1.0_r - e) / h;
        decay = 1.0_r - e;
        forcing = 2.0_r * (1.0_r - phi1);
        carry = phi1;
        drift = 1.0_r - forcing / h;
    }
};

/**
 * @brief The step dt / tau above which a relaxation is integrated
 *        exponentially, and at or below which explicitly, at the integrator's order.
 */
inline constexpr rtype EXPONENTIAL_ABOVE = 1.0_r;

/**
 * @brief Solid spheres of one diameter and density in a single gas: the
 *        right-hand sides of dx/dt, dv/dt and, with heat transfer, dT_p/dt from
 *        the gas state at the particle.
 *
 * The velocity relaxes to the gas velocity u in tau = (rho_p + C_A rho) d^2 /
 * (18 mu f), C_A = 1/2 with added mass and 0 without, and the other forces
 * give an acceleration a: dv/dt = (u - v) / tau + a. The temperature relaxes to
 * the gas temperature T in tau_T = rho_p c_p d^2 / (6 k Nu), with the Nusselt
 * number of Ranz and Marshall. A relaxation whose dt / tau is at most
 * EXPONENTIAL_ABOVE is integrated at the integrator's order; above, it is
 * integrated exponentially (Exponential) toward its equilibrium u + tau a
 * (respectively T), so that the flow's step never depends on tau.
 */
struct Solid {
    rtype diameter = 0.0;
    rtype density = 0.0;
    Drag drag = Drag::STOKES;
    rtype gravity[N_DIM] = {};
    rtype gas_gravity[N_DIM] = {};  // of [source], in the fluid acceleration of the added mass
    bool pressure_gradient = false;
    bool added_mass = false;
    bool lift = false;
    rtype specific_heat = 0.0;  // 0: no heat transfer
    Euler gas;

    /**
     * @brief The values the first stage of a step keeps per particle: its
     *        velocity v, the gas velocity u, the acceleration a, tau, dt, then
     *        with heat transfer T_p, T and tau_T.
     */
    enum Start : uint32_t { V = 0, U = N_DIM, A = 2 * N_DIM, TAU = 3 * N_DIM, DT = 3 * N_DIM + 1, TP = 3 * N_DIM + 2,
                            T = 3 * N_DIM + 3, TAU_T = 3 * N_DIM + 4 };

    KOKKOS_INLINE_FUNCTION bool heated() const { return specific_heat > 0.0_r; }

    /** @brief Whether the forces read the gradients of the gas state. */
    KOKKOS_INLINE_FUNCTION bool reads_gradients() const { return pressure_gradient || added_mass || lift; }

    /** @brief Values per particle that the first stage of a step keeps. */
    KOKKOS_INLINE_FUNCTION uint32_t n_start() const { return heated() ? TAU_T + 1 : DT + 1; }

    /** @brief The relaxation times of a particle's velocity and temperature, and its acceleration by the other forces. */
    struct Response {
        rtype tau;
        rtype a[N_DIM];
        rtype tau_T;
    };

    /**
     * @brief The response of a particle at velocity v and temperature Tp to
     *        the gas state W = [rho, u, p] with gradients G (G[k N_DIM + d] =
     *        dW_k/dx_d), which only the forces of reads_gradients() read.
     */
    KOKKOS_INLINE_FUNCTION Response response(const rtype * W, const rtype * G, const rtype * v, const rtype Tp) const {
        const rtype rho = W[0], p = W[N_DIM + 1];
        const rtype T_gas = p / (rho * gas.R);
        const rtype mu = gas.viscosity(T_gas);
        const rtype pi = Kokkos::numbers::pi_v<rtype>;
        rtype slip[N_DIM], slip2 = 0.0_r;
        FOR_I_DIM {
            slip[i] = W[1 + i] - v[i];
            slip2 += slip[i] * slip[i];
        }
        const rtype speed = Kokkos::sqrt(slip2);
        const rtype Re = rho * speed * diameter / mu;
        rtype f;
        if (drag == Drag::SINGH) {
            const rtype c = Kokkos::sqrt(gas.gamma * p / rho);
            const rtype Kn = mu / (rho * c * diameter) * Kokkos::sqrt(0.5_r * pi * gas.gamma);
            f = singh_drag_factor(Re, speed / c, Kn, Tp / T_gas, gas.gamma);
        } else {
            f = drag_factor(drag, Re);
        }
        const rtype C_A = added_mass ? 0.5_r : 0.0_r;
        const rtype mass = density + C_A * rho;  // per volume, with the added mass
        Response r;
        r.tau = mass * diameter * diameter / (18.0_r * mu * f);
        // Per volume: gravity, buoyancy or the pressure gradient, the fluid acceleration of the added mass, lift
        rtype force[N_DIM];
        FOR_I_DIM {
            const rtype dp = reads_gradients() ? G[(N_DIM + 1) * N_DIM + i] : 0.0_r;
            force[i] = density * gravity[i] + (pressure_gradient ? -dp : -rho * gravity[i]) +
                       C_A * (rho * gas_gravity[i] - dp);
        }
        if (lift) {
            // Saffman: F / V_p = 1.615 (6 / (pi d)) sqrt(rho mu / |w|) (u - v) x w
            const auto du = [&](const int i, const int d) { return G[(1 + i) * N_DIM + d]; };
            rtype w[3] = {0.0_r, 0.0_r, du(1, 0) - du(0, 1)};
            if constexpr (N_DIM == 3) {
                w[0] = du(2, 1) - du(1, 2);
                w[1] = du(0, 2) - du(2, 0);
            }
            const rtype w_norm = Kokkos::sqrt(w[0] * w[0] + w[1] * w[1] + w[2] * w[2]);
            if (w_norm > 0.0_r) {
                const rtype k = 1.615_r * 6.0_r / (pi * diameter) * Kokkos::sqrt(rho * mu / w_norm);
                const rtype s[3] = {slip[0], slip[1], N_DIM == 3 ? slip[N_DIM - 1] : 0.0_r};
                const rtype cross[3] = {s[1] * w[2] - s[2] * w[1], s[2] * w[0] - s[0] * w[2], s[0] * w[1] - s[1] * w[0]};
                FOR_I_DIM force[i] += k * cross[i];
            }
        }
        FOR_I_DIM r.a[i] = force[i] / mass;
        r.tau_T = 0.0_r;
        if (heated()) {
            const rtype Nu = 2.0_r + 0.6_r * Kokkos::sqrt(Re) * Kokkos::cbrt(gas.Pr);
            r.tau_T = density * specific_heat * diameter * diameter / (6.0_r * gas.conductivity(mu) * Nu);
        }
        return r;
    }

    /**
     * @brief The rates of a stage: dx/dt into rate[0, N_DIM), dv/dt into
     *        rate[N_DIM, 2 N_DIM) and with heat transfer dT_p/dt into
     *        rate[2 N_DIM], of a particle whose velocity (then temperature) is
     *        y, in gas W with gradients G, for a step dt. The first stage of a
     *        step (its start) writes start (n_start() values), which every
     *        stage reads: the relaxation times there decide which relaxations
     *        are exponential over the step.
     */
    KOKKOS_INLINE_FUNCTION void rates(const rtype * W, const rtype * G, const rtype * y, const rtype dt, const bool first,
                                      rtype * start, rtype * rate) const {
        const rtype T_gas = W[N_DIM + 1] / (W[0] * gas.R);
        const rtype Tp = heated() ? y[N_DIM] : T_gas;
        const Response r = response(W, G, y, Tp);
        const rtype * u = W + 1;
        if (first) {
            FOR_I_DIM {
                start[V + i] = y[i];
                start[U + i] = u[i];
                start[A + i] = r.a[i];
            }
            start[TAU] = r.tau;
            start[DT] = dt;
            if (heated()) {
                start[TP] = Tp;
                start[T] = T_gas;
                start[TAU_T] = r.tau_T;
            }
        }
        const rtype h = dt / start[TAU];
        if (h > EXPONENTIAL_ABOVE) {
            const Exponential e(h);
            FOR_I_DIM {
                const rtype v_eq = u[i] + start[TAU] * r.a[i];
                const rtype v_eq0 = start[U + i] + start[TAU] * start[A + i], v0 = start[V + i];
                rate[i] = v_eq0 + e.carry * (v0 - v_eq0) + e.drift * (v_eq - v_eq0);
                rate[N_DIM + i] = (e.decay * (v_eq0 - v0) + e.forcing * (v_eq - v_eq0)) / dt;
            }
        } else {
            FOR_I_DIM {
                rate[i] = y[i];
                rate[N_DIM + i] = (u[i] - y[i]) / r.tau + r.a[i];
            }
        }
        if (heated()) {
            const rtype h_T = dt / start[TAU_T];
            if (h_T > EXPONENTIAL_ABOVE) {
                const Exponential e(h_T);
                rate[2 * N_DIM] = (e.decay * (start[T] - start[TP]) + e.forcing * (T_gas - start[T])) / dt;
            } else {
                rate[2 * N_DIM] = (T_gas - Tp) / r.tau_T;
            }
        }
    }
};

/**
 * @brief Rebound of a solid particle found a depth d past a wall of unit normal
 *        n (out of the domain) at the end of a step, from the time within the
 *        step at which it reached the wall. Over the step dt the particle moved
 *        as dv/dt = (u - v) / tau + a with u, a and tau of the step's start
 *        (Solid::Start): it reached the wall a time b before the end
 *        (the root of the distance it then travels along n), left it with the
 *        normal velocity reversed times e_n and the tangential velocity times
 *        e_t, and relaxed on for b. Exact for that motion, constant
 *        acceleration (tau -> infinity) and drag in gas at rest among it. A
 *        particle in its stiff limit (tau below the step) follows the gas
 *        within the step, and rebounds at its end. Updates the position p and
 *        the velocity v.
 */
KOKKOS_INLINE_FUNCTION void rebound(const rtype * u, const rtype * a, const rtype tau, const rtype dt, const rtype * n,
                                    const rtype d, const rtype e_n, const rtype e_t, rtype * p, rtype * v) {
    // (R - I) x for the rebound R x = e_t x_t - e_n x_n n
    const auto change = [&](const rtype * x, rtype * out) {
        rtype x_n = 0.0_r;
        FOR_I_DIM x_n += x[i] * n[i];
        FOR_I_DIM out[i] = (e_t - 1.0_r) * (x[i] - x_n * n[i]) - (e_n + 1.0_r) * x_n * n[i];
    };
    rtype v_n = 0.0_r;
    FOR_I_DIM v_n += v[i] * n[i];
    if (!(dt > 0.0_r) || tau < dt || !(v_n > 0.0_r)) {
        rtype dv[N_DIM];
        change(v, dv);
        FOR_I_DIM {
            v[i] += dv[i];
            p[i] -= (1.0_r + e_n) * d * n[i];
        }
        return;
    }
    // z = b / tau <= 1: phi1 = (1 - e^-z) / z, psi = (e^z - 1) / z, phi2 = (1 - phi1) / z, by series at small z
    const auto functions = [](const rtype z, rtype & phi1, rtype & psi, rtype & phi2) {
        if (z < 0.1_r) {
            phi1 = 1.0_r - z * (0.5_r - z * (1.0_r / 6.0_r - z * (1.0_r / 24.0_r - z / 120.0_r)));
            psi = 1.0_r + z * (0.5_r + z * (1.0_r / 6.0_r + z * (1.0_r / 24.0_r + z / 120.0_r)));
            phi2 = 0.5_r - z * (1.0_r / 6.0_r - z * (1.0_r / 24.0_r - z * (1.0_r / 120.0_r - z / 720.0_r)));
        } else {
            phi1 = -Kokkos::expm1(-z) / z;
            psi = Kokkos::expm1(z) / z;
            phi2 = (1.0_r - phi1) / z;
        }
    };
    rtype u_n = 0.0_r, a_n = 0.0_r;
    FOR_I_DIM {
        u_n += u[i] * n[i];
        a_n += a[i] * n[i];
    }
    const rtype w_n = v_n - u_n;
    // d = u_n b + w_n b psi(z) - a_n b^2 (psi phi1 - phi2)(z), from b = d / v_n
    rtype b = d / v_n, phi1, psi, phi2;
    for (int k = 0; k < 6; k++) {
        functions(Kokkos::min(b, dt) / tau, phi1, psi, phi2);
        const rtype rate = u_n + w_n * psi - a_n * b * (psi * phi1 - phi2);
        if (!(rate > 0.0_r)) break;
        b = d / rate;
    }
    b = Kokkos::min(b, dt);
    const rtype z = b / tau, decay = Kokkos::exp(-z);
    functions(z, phi1, psi, phi2);
    // The velocity at the wall is v_c = u + (v - u) e^z - a b psi; v gains (R - I) v_c e^-z and p (R - I) v_c b phi1
    rtype c[N_DIM], q[N_DIM], dv[N_DIM], dp[N_DIM];
    FOR_I_DIM {
        const rtype w = v[i] - u[i];
        c[i] = u[i] * decay + w - a[i] * b * phi1;
        q[i] = u[i] * b * phi1 + b * psi * (w - a[i] * b * phi1);
    }
    change(c, dv);
    change(q, dp);
    FOR_I_DIM {
        v[i] += dv[i];
        p[i] += dp[i];
    }
}

} // namespace particles

#endif // PARTICLE_FORCES_H

/**
 * @file wall_model.h
 * @brief Wall-stress models of wall-modeled LES: the equilibrium ODE model
 *        and the inverse-velocity-transformation model of Griffin, Fu & Moin
 *        (2023). See docs/design/wall_model.md.
 * @version 0.1
 * @date 2026-10-15
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 *
 */

#ifndef WALL_MODEL_H
#define WALL_MODEL_H

#include <string>
#include <unordered_map>

#include <Kokkos_Core.hpp>

#include "common.h"
#include "physics.h"

enum class WallModelType : uint8_t { NONE = 0, EQUILIBRIUM, GFM };

static const std::unordered_map<std::string, WallModelType> WALL_MODEL_TYPES = {
    {"equilibrium", WallModelType::EQUILIBRIUM},
    {"gfm", WallModelType::GFM},
};

/** @brief Thermal condition of a modeled wall. */
enum class WallThermal : uint8_t { ISOTHERMAL, HEAT_FLUX };

/** @brief Constants of a wall model (plain aggregate, by boundary condition). */
struct WallModelParameters {
    WallModelType type = WallModelType::NONE;
    double kappa = 0.41;  // von Karman constant
    double A_plus = 17.0; // damping constant
    double Pr_t = 0.9;    // turbulent Prandtl number (ODE model)
    double s = 1.14;      // Reynolds analogy factor (GFM)
};

namespace wall_model {

constexpr int N_NODES = 64;       // nodes from the wall (0) to the exchange height (N_NODES - 1)
constexpr double STRETCH = 1.1;   // ratio of consecutive spacings
constexpr int GFM_CORRECTIONS = 3; // corrections of each node's predicted velocity (GFM)

/** @brief State at the exchange point and the edge, and the wall's condition. */
struct Input {
    double h = 0.0;    // exchange height
    double U = 0.0;    // tangential speed relative to the wall at the exchange point
    double T = 0.0;    // temperature at the exchange point
    double p = 0.0;    // pressure (constant across the layer)
    double U_e = 0.0;  // GFM: edge speed
    double T_e = 0.0;  // GFM: edge temperature
    WallThermal thermal = WallThermal::ISOTHERMAL;
    double T_w = 0.0;  // ISOTHERMAL: wall temperature
    double q = 0.0;    // HEAT_FLUX: heat flux into the fluid (0: adiabatic)
};

/** @brief Wall shear stress, heat flux into the fluid and wall temperature. */
struct Output {
    double tau_w = 0.0;
    double q_w = 0.0;
    double T_w = 0.0;
    int iterations = 0;
    bool converged = false;
};

/** @brief Gas properties in double: viscosity law, cp, R, Pr. */
struct Gas {
    Euler gas;

    KOKKOS_INLINE_FUNCTION
    double mu(const double T) const {
        switch (gas.viscosity_model) {
            case ViscosityModel::CONSTANT:
                return static_cast<double>(gas.mu_ref);
            case ViscosityModel::SUTHERLAND: {
                const double T0 = static_cast<double>(gas.T_mu_ref), S = static_cast<double>(gas.S_mu);
                return static_cast<double>(gas.mu_ref) * Kokkos::pow(T / T0, 1.5) * (T0 + S) / (T + S);
            }
            case ViscosityModel::POWER_LAW:
                return static_cast<double>(gas.mu_ref) *
                       Kokkos::pow(T / static_cast<double>(gas.T_mu_ref), static_cast<double>(gas.n_mu));
            default:
                return 0.0;
        }
    }
    KOKKOS_INLINE_FUNCTION double cp() const { return static_cast<double>(gas.cp); }
    KOKKOS_INLINE_FUNCTION double R() const { return static_cast<double>(gas.R); }
    KOKKOS_INLINE_FUNCTION double Pr() const { return static_cast<double>(gas.Pr); }
};

/**
 * @brief Node j of the wall-normal grid over [0, h] is grid_scale(h) (r^j -
 *        1), r = STRETCH, the last one h; powers by repeated products, so that
 *        every build gets the same nodes.
 */
KOKKOS_INLINE_FUNCTION
double grid_scale(const double h) {
    double rN = 1.0;
    for (int i = 0; i < N_NODES - 1; i++) rN *= STRETCH;
    return h / (rN - 1.0);
}

/** @brief Van Driest damping [1 - exp(-y / A+)]^2. */
KOKKOS_INLINE_FUNCTION
double damping(const double y_plus, const double A_plus) {
    const double d = 1.0 - Kokkos::exp(-y_plus / A_plus);
    return d * d;
}

/**
 * @brief Initial estimate of tau_w: the larger of the laminar stress mu_w U /
 *        h and the log law U / u_tau = ln(h u_tau / nu_w) / kappa + 5.2 at
 *        the wall's properties.
 */
KOKKOS_INLINE_FUNCTION
double initial_stress(const WallModelParameters & P, const double rho_w, const double mu_w, const double U,
                      const double h) {
    const double nu = mu_w / rho_w;
    double u_tau = Kokkos::sqrt(nu * U / h);
    for (int i = 0; i < 20; i++) {
        const double h_plus = h * u_tau / nu;
        if (h_plus < 11.0) break;
        u_tau = U / (Kokkos::log(h_plus) / P.kappa + 5.2);
    }
    return Kokkos::fmax(rho_w * u_tau * u_tau, mu_w * U / h);
}

/**
 * @brief Equilibrium ODE model (Kawai & Larsson 2012): the thin-layer
 *        momentum and energy equations
 *          d/dy [(mu + mu_t) du/dy] = 0,
 *          d/dy [(mu + mu_t) u du/dy + cp (mu / Pr + mu_t / Pr_t) dT/dy] = 0,
 *        mu_t = kappa y sqrt(rho tau_w) [1 - exp(-y* / A+)]^2 with y* = y
 *        sqrt(rho tau_w) / mu, at constant pressure, from the wall (u = 0, T
 *        = T_w or -k dT/dy = q) to the exchange height (U, T). Both equations
 *        say a flux is constant across the layer: c du/dy = tau_w and
 *        k dT/dy + tau_w u = -q_w. Given mu_t and the properties they are
 *        integrated by Simpson's rule on each interval of the grid; mu_t, the
 *        properties and tau_w are iterated to a fixed point.
 */
KOKKOS_INLINE_FUNCTION
Output equilibrium(const WallModelParameters & P, const Gas & gas, const Input & in) {
    constexpr int N = N_NODES;
    const double cp = gas.cp(), R = gas.R(), Pr = gas.Pr();
    // Nodes, then interval midpoints (index N + j for interval j)
    double y[2 * N - 1], u[2 * N - 1], T[2 * N - 1], inv_c[2 * N - 1], inv_k[2 * N - 1];
    const double scale = grid_scale(in.h);
    double rj = 1.0;
    for (int j = 0; j < N; j++) {
        y[j] = j == N - 1 ? in.h : scale * (rj - 1.0);
        rj *= STRETCH;
    }
    for (int j = 0; j < N - 1; j++) y[N + j] = 0.5 * (y[j] + y[j + 1]);
    const bool isothermal = in.thermal == WallThermal::ISOTHERMAL;
    const double T_wall0 = isothermal ? in.T_w : in.T;
    for (int j = 0; j < 2 * N - 1; j++) {
        u[j] = in.U * y[j] / in.h;
        T[j] = T_wall0 + (in.T - T_wall0) * y[j] / in.h;
    }
    Output out;
    double tau = initial_stress(P, in.p / (R * T_wall0), gas.mu(T_wall0), in.U, in.h);
    double tau_prev = 0.0, g_prev = 0.0;
    for (int it = 1; it <= 200; it++) {
        for (int j = 0; j < 2 * N - 1; j++) {
            const double rho = in.p / (R * T[j]);
            const double mu = gas.mu(T[j]);
            const double y_star = y[j] * Kokkos::sqrt(rho * tau) / mu;
            const double mu_t = P.kappa * y[j] * Kokkos::sqrt(rho * tau) * damping(y_star, P.A_plus);
            inv_c[j] = 1.0 / (mu + mu_t);
            inv_k[j] = 1.0 / (cp * (mu / Pr + mu_t / P.Pr_t));
        }
        // Momentum: u = tau_w int dy / c
        double sum_c = 0.0;
        for (int j = 0; j < N - 1; j++) sum_c += (y[j + 1] - y[j]) / 6.0 * (inv_c[j] + 4.0 * inv_c[N + j] + inv_c[j + 1]);
        const double tau_new = in.U / sum_c;
        u[0] = 0.0;
        for (int j = 0; j < N - 1; j++) {
            const double dy = y[j + 1] - y[j];
            u[N + j] = u[j] + tau_new * 0.25 * dy * (inv_c[j] + inv_c[N + j]);
            u[j + 1] = u[j] + tau_new * dy / 6.0 * (inv_c[j] + 4.0 * inv_c[N + j] + inv_c[j + 1]);
        }
        // Energy: T = T_w + C int dy / k - tau_w int u dy / k, C = -q_w
        double sum_k = 0.0, sum_work = 0.0;
        for (int j = 0; j < N - 1; j++) {
            const double w = (y[j + 1] - y[j]) / 6.0;
            sum_k += w * (inv_k[j] + 4.0 * inv_k[N + j] + inv_k[j + 1]);
            sum_work += w * (u[j] * inv_k[j] + 4.0 * u[N + j] * inv_k[N + j] + u[j + 1] * inv_k[j + 1]);
        }
        double C, T_w;
        if (isothermal) {
            T_w = in.T_w;
            C = (in.T - T_w + tau_new * sum_work) / sum_k;
        } else {
            C = -in.q;
            T_w = in.T - C * sum_k + tau_new * sum_work;
        }
        double dT_max = 0.0;
        auto set_T = [&](const int j, const double T_new) {
            dT_max = Kokkos::fmax(dT_max, Kokkos::fabs(T_new - T[j]) / T_new);
            T[j] = T_new;
        };
        double T_a = T_w;
        set_T(0, T_w);
        for (int j = 0; j < N - 1; j++) {
            const double dy = y[j + 1] - y[j];
            const double half = 0.25 * dy * (C * (inv_k[j] + inv_k[N + j]) - tau_new * (u[j] * inv_k[j] + u[N + j] * inv_k[N + j]));
            const double full = dy / 6.0 * (C * (inv_k[j] + 4.0 * inv_k[N + j] + inv_k[j + 1]) -
                                            tau_new * (u[j] * inv_k[j] + 4.0 * u[N + j] * inv_k[N + j] + u[j + 1] * inv_k[j + 1]));
            set_T(N + j, T_a + half);
            T_a += full;
            set_T(j + 1, j == N - 2 ? in.T : T_a);
        }
        // Secant steps on the fixed point's residual tau_new - tau
        const double g = tau_new - tau;
        const bool done = Kokkos::fabs(g) <= 1e-10 * tau_new && dT_max <= 1e-10;
        out.tau_w = tau_new;
        out.q_w = -C;
        out.T_w = T_w;
        out.iterations = it;
        if (done || !(tau_new > 0.0)) {
            out.converged = done || !(tau_new > 0.0);
            break;
        }
        double tau_next = tau_new;
        if (it > 1 && g != g_prev) {
            const double secant = tau - g * (tau - tau_prev) / (g - g_prev);
            if (secant > 0.5 * tau_new && secant < 2.0 * tau_new) tau_next = secant;
        }
        tau_prev = tau;
        g_prev = g;
        tau = tau_next;
    }
    return out;
}

/**
 * @brief Temperature of the generalized Reynolds analogy through the wall and
 *        the exchange point (Griffin, Fu & Moin 2023, eq. 8):
 *        T = T_w + s Pr (T_r - T_w) (U / U_e) (1 - U / U_m) + (U / U_m)^2 (T_m - T_w).
 */
struct ReynoldsAnalogy {
    double T_w, T_m, U_m, U_e, a;  // a = s Pr (T_r - T_w)

    KOKKOS_INLINE_FUNCTION
    double operator()(const double U) const {
        const double r = U / U_m;
        return T_w + a * (U / U_e) * (1.0 - r) + r * r * (T_m - T_w);
    }
};

/**
 * @brief Velocity at the exchange height of the inverse-transformation
 *        profile for the wall stress tau (see gfm).
 */
KOKKOS_INLINE_FUNCTION
double gfm_velocity(const WallModelParameters & P, const Gas & gas, const Input & in, const ReynoldsAnalogy & T_of_U,
                    const double rho_w, const double mu_w, const double tau) {
    const double pR = in.p / gas.R();
    const double u_tau = Kokkos::sqrt(tau / rho_w);
    const double plus = rho_w * u_tau / mu_w;  // y+ per unit length
    const double scale = grid_scale(in.h);
    auto S = [&](const double y_star) { return 1.0 / (1.0 + P.kappa * y_star * damping(y_star, P.A_plus)); };
    double U = 0.0, y_prev = 0.0, mu_prev = mu_w, y_star_prev = 0.0, S_prev = 1.0, dU = 0.0, rj = 1.0;
    for (int j = 1; j < N_NODES; j++) {
        rj *= STRETCH;
        const double y = j == N_NODES - 1 ? in.h : scale * (rj - 1.0);
        const double dy_plus = (y - y_prev) * plus;
        // Node j's temperature sets its y* and mu+: predict its velocity from the
        // previous increment (constant in the log layer on this grid), then correct.
        // S_t is averaged over the interval in y* by Simpson's rule
        double U_next = U + dU, mu = mu_prev, y_star = y_star_prev, S_next = S_prev;
        for (int it = 0; it < GFM_CORRECTIONS; it++) {
            const double T = T_of_U(U_next);
            mu = gas.mu(T);
            y_star = y * Kokkos::sqrt(tau * pR / T) / mu;
            const double dy_star = Kokkos::fmax(y_star - y_star_prev, 1e-12 * dy_plus);
            S_next = S(y_star);
            const double S_t = (S_prev + 4.0 * S(0.5 * (y_star + y_star_prev)) + S_next) / 6.0;
            const double mu_plus = 0.5 * (mu + mu_prev) / mu_w;
            U_next = U + u_tau * S_t / ((1.0 - S_t) / (mu_plus * dy_star) + S_t * mu_plus / dy_plus);
        }
        mu_prev = mu;
        y_star_prev = y_star;
        S_prev = S_next;
        dU = U_next - U;
        U = U_next;
        y_prev = y;
    }
    return U;
}

/**
 * @brief Griffin, Fu & Moin (2023): the inverse of their velocity
 *        transformation applied to the constant-property law
 *        S_t+ = 1 / (1 + kappa y* [1 - exp(-y* / A+)]^2), with the temperature
 *        of the generalized Reynolds analogy (ReynoldsAnalogy, recovery factor
 *        Pr^(1/3)) through the wall and the exchange point. tau_w by shooting
 *        (secant method on log tau_w, which U(h) increases with);
 *        q_w = s tau_w cp (T_w - T_r) / U_e. Adiabatic walls (HEAT_FLUX with
 *        q = 0) take T_w = T_r; other heat fluxes are not supported (the
 *        caller rejects them).
 */
KOKKOS_INLINE_FUNCTION
Output gfm(const WallModelParameters & P, const Gas & gas, const Input & in) {
    const double cp = gas.cp(), R = gas.R(), Pr = gas.Pr();
    const double U_e = Kokkos::fmax(in.U_e, in.U);
    const double T_r = in.T_e + Kokkos::cbrt(Pr) * U_e * U_e / (2.0 * cp);
    const double T_w = in.thermal == WallThermal::ISOTHERMAL ? in.T_w : T_r;
    const double rho_w = in.p / (R * T_w), mu_w = gas.mu(T_w);
    Output out;
    out.T_w = T_w;
    if (!(in.U > 0.0) || !(U_e > 0.0)) {
        // No shear: conduction across the layer
        out.q_w = -cp * mu_w / Pr * (in.T - T_w) / in.h;
        out.converged = true;
        return out;
    }
    const ReynoldsAnalogy T_of_U{T_w, in.T, in.U, U_e, P.s * Pr * (T_r - T_w)};
    // Residual in log U, which is close to linear in log tau_w
    auto residual = [&](const double log_tau) {
        return Kokkos::log(gfm_velocity(P, gas, in, T_of_U, rho_w, mu_w, Kokkos::exp(log_tau)) / in.U);
    };
    double x0 = Kokkos::log(initial_stress(P, rho_w, mu_w, in.U, in.h));
    double f0 = residual(x0);
    double x1 = x0 - 2.0 * f0, f1 = residual(x1);  // dlog U / dlog tau is 1/2 to 1
    out.iterations = 2;
    for (int i = 0; i < 50; i++) {
        if (f1 == f0 || Kokkos::fabs(x1 - x0) <= 1e-9) {
            out.converged = Kokkos::fabs(f1) <= 1e-9;
            break;
        }
        double x2 = x1 - f1 * (x1 - x0) / (f1 - f0);
        x2 = Kokkos::fmin(Kokkos::fmax(x2, x1 - 2.0), x1 + 2.0);
        x0 = x1;
        f0 = f1;
        x1 = x2;
        f1 = residual(x1);
        out.iterations++;
    }
    out.tau_w = Kokkos::exp(x1);
    out.q_w = P.s * out.tau_w * cp * (T_w - T_r) / U_e;
    return out;
}

/** @brief The model of P. */
KOKKOS_INLINE_FUNCTION
Output evaluate(const WallModelParameters & P, const Gas & gas, const Input & in) {
    return P.type == WallModelType::GFM ? gfm(P, gas, in) : equilibrium(P, gas, in);
}

} // namespace wall_model

#endif // WALL_MODEL_H

/**
 * @file riemann_solver.h
 * @author Matthew Bonanni (mbonanni001@gmail.com)
 * @brief Approximate Riemann solvers.
 * @version 0.2
 * @date 2024-01-17
 *
 * @copyright Copyright (c) 2024 Matthew Bonanni
 *
 */

#ifndef RIEMANN_SOLVER_H
#define RIEMANN_SOLVER_H

#include <string>
#include <unordered_map>

#include <Kokkos_Core.hpp>

#include "common_typedef.h"
#include "common_math.h"

enum class RiemannSolverType {
    RUSANOV,
    HLL,
    HLLC,
};

static const std::unordered_map<std::string, RiemannSolverType> RIEMANN_SOLVER_TYPES = {
    {"Rusanov", RiemannSolverType::RUSANOV},
    {"HLL", RiemannSolverType::HLL},
    {"HLLC", RiemannSolverType::HLLC}
};

static const std::unordered_map<RiemannSolverType, std::string> RIEMANN_SOLVER_NAMES = {
    {RiemannSolverType::RUSANOV, "Rusanov"},
    {RiemannSolverType::HLL, "HLL"},
    {RiemannSolverType::HLLC, "HLLC"}
};

/**
 * All solvers take left/right states as W = [rho, u_x, u_y, p] and a unit
 * normal pointing from left to right, and return the flux of
 * [rho, rho u_x, rho u_y, rho E] through the face per unit area.
 *
 * The 1D star-region estimators (PVRS, TRRS, TSRS, ANRS) take
 * W = [rho, u_n, p] and follow Toro, "Riemann Solvers and Numerical
 * Methods for Fluid Dynamics", 3rd ed., chapter 9.
 */
namespace riemann {

/**
 * @brief Physical Euler flux of state W through unit normal n.
 */
KOKKOS_INLINE_FUNCTION
void physical_flux(const rtype * W, const rtype * n, const rtype gamma,
                   rtype * U, rtype * F) {
    const rtype u_n = W[1] * n[0] + W[2] * n[1];
    U[0] = W[0];
    U[1] = W[0] * W[1];
    U[2] = W[0] * W[2];
    U[3] = W[3] / (gamma - 1.0) + 0.5 * W[0] * (W[1] * W[1] + W[2] * W[2]);
    F[0] = U[0] * u_n;
    F[1] = U[1] * u_n + W[3] * n[0];
    F[2] = U[2] * u_n + W[3] * n[1];
    F[3] = (U[3] + W[3]) * u_n;
}

/**
 * @brief Primitive variable Riemann solver (Toro 9.20, 9.28).
 */
KOKKOS_INLINE_FUNCTION
rtype PVRS(const rtype * W_l, const rtype * W_r, const rtype gamma) {
    const rtype a_l = Kokkos::sqrt(gamma * W_l[2] / W_l[0]);
    const rtype a_r = Kokkos::sqrt(gamma * W_r[2] / W_r[0]);
    const rtype rho_avg = 0.5 * (W_l[0] + W_r[0]);
    const rtype a_avg = 0.5 * (a_l + a_r);
    return 0.5 * (W_l[2] + W_r[2]) + 0.5 * (W_l[1] - W_r[1]) * rho_avg * a_avg;
}

/**
 * @brief Two-rarefaction Riemann solver (Toro 9.32).
 */
KOKKOS_INLINE_FUNCTION
rtype TRRS(const rtype * W_l, const rtype * W_r, const rtype gamma) {
    const rtype a_l = Kokkos::sqrt(gamma * W_l[2] / W_l[0]);
    const rtype a_r = Kokkos::sqrt(gamma * W_r[2] / W_r[0]);
    const rtype z = (gamma - 1.0) / (2.0 * gamma);
    const rtype num = a_l + a_r - 0.5 * (gamma - 1.0) * (W_r[1] - W_l[1]);
    const rtype den = a_l / Kokkos::pow(W_l[2], z) + a_r / Kokkos::pow(W_r[2], z);
    return Kokkos::pow(Kokkos::fmax(num, 0.0) / den, 1.0 / z);
}

/**
 * @brief Two-shock Riemann solver (Toro 9.42), linearized about p_0.
 */
KOKKOS_INLINE_FUNCTION
rtype TSRS(const rtype * W_l, const rtype * W_r, const rtype gamma, const rtype p_0) {
    const rtype A_l = 2.0 / ((gamma + 1.0) * W_l[0]);
    const rtype A_r = 2.0 / ((gamma + 1.0) * W_r[0]);
    const rtype B_l = (gamma - 1.0) / (gamma + 1.0) * W_l[2];
    const rtype B_r = (gamma - 1.0) / (gamma + 1.0) * W_r[2];
    const rtype p = Kokkos::fmax(0.0, p_0);
    const rtype g_l = Kokkos::sqrt(A_l / (p + B_l));
    const rtype g_r = Kokkos::sqrt(A_r / (p + B_r));
    return (g_l * W_l[2] + g_r * W_r[2] - (W_r[1] - W_l[1])) / (g_l + g_r);
}

/**
 * @brief Adaptive noniterative Riemann solver (Toro 9.5.2): estimate p*.
 */
KOKKOS_INLINE_FUNCTION
rtype ANRS(const rtype * W_l, const rtype * W_r, const rtype gamma) {
    constexpr rtype q_user = 2.0;
    const rtype p_min = Kokkos::fmin(W_l[2], W_r[2]);
    const rtype p_max = Kokkos::fmax(W_l[2], W_r[2]);
    const rtype p_pv = Kokkos::fmax(0.0, PVRS(W_l, W_r, gamma));
    if ((p_max / p_min < q_user) && (p_min <= p_pv) && (p_pv <= p_max)) {
        return p_pv;
    } else if (p_pv < p_min) {
        return TRRS(W_l, W_r, gamma);
    } else {
        return TSRS(W_l, W_r, gamma, p_pv);
    }
}

/**
 * @brief Pressure-based wave speed estimates (Toro 10.59-10.60).
 */
KOKKOS_INLINE_FUNCTION
void wave_speeds(const rtype * W_l, const rtype * W_r, const rtype u_l_n, const rtype u_r_n,
                 const rtype gamma, rtype & S_l, rtype & S_r) {
    const rtype w_l[3] = {W_l[0], u_l_n, W_l[3]};
    const rtype w_r[3] = {W_r[0], u_r_n, W_r[3]};
    const rtype p_star = ANRS(w_l, w_r, gamma);
    const rtype a_l = Kokkos::sqrt(gamma * W_l[3] / W_l[0]);
    const rtype a_r = Kokkos::sqrt(gamma * W_r[3] / W_r[0]);
    const rtype c = (gamma + 1.0) / (2.0 * gamma);
    const rtype q_l = (p_star <= W_l[3]) ? 1.0 : Kokkos::sqrt(1.0 + c * (p_star / W_l[3] - 1.0));
    const rtype q_r = (p_star <= W_r[3]) ? 1.0 : Kokkos::sqrt(1.0 + c * (p_star / W_r[3] - 1.0));
    S_l = u_l_n - a_l * q_l;
    S_r = u_r_n + a_r * q_r;
}

struct Rusanov {
    KOKKOS_INLINE_FUNCTION
    static void calc_flux(rtype * flux, const rtype * n,
                          const rtype * W_l, const rtype * W_r, const rtype gamma) {
        rtype U_l[N_CONSERVATIVE], U_r[N_CONSERVATIVE];
        rtype F_l[N_CONSERVATIVE], F_r[N_CONSERVATIVE];
        physical_flux(W_l, n, gamma, U_l, F_l);
        physical_flux(W_r, n, gamma, U_r, F_r);
        const rtype u_l_n = W_l[1] * n[0] + W_l[2] * n[1];
        const rtype u_r_n = W_r[1] * n[0] + W_r[2] * n[1];
        const rtype a_l = Kokkos::sqrt(gamma * W_l[3] / W_l[0]);
        const rtype a_r = Kokkos::sqrt(gamma * W_r[3] / W_r[0]);
        const rtype S_max = Kokkos::fmax(Kokkos::fabs(u_l_n) + a_l, Kokkos::fabs(u_r_n) + a_r);
        FOR_I_CONSERVATIVE flux[i] = 0.5 * (F_l[i] + F_r[i] - S_max * (U_r[i] - U_l[i]));
    }
};

struct HLL {
    KOKKOS_INLINE_FUNCTION
    static void calc_flux(rtype * flux, const rtype * n,
                          const rtype * W_l, const rtype * W_r, const rtype gamma) {
        rtype U_l[N_CONSERVATIVE], U_r[N_CONSERVATIVE];
        rtype F_l[N_CONSERVATIVE], F_r[N_CONSERVATIVE];
        physical_flux(W_l, n, gamma, U_l, F_l);
        physical_flux(W_r, n, gamma, U_r, F_r);
        const rtype u_l_n = W_l[1] * n[0] + W_l[2] * n[1];
        const rtype u_r_n = W_r[1] * n[0] + W_r[2] * n[1];
        rtype S_l, S_r;
        wave_speeds(W_l, W_r, u_l_n, u_r_n, gamma, S_l, S_r);
        if (0.0 <= S_l) {
            FOR_I_CONSERVATIVE flux[i] = F_l[i];
        } else if (S_r <= 0.0) {
            FOR_I_CONSERVATIVE flux[i] = F_r[i];
        } else {
            FOR_I_CONSERVATIVE {
                flux[i] = (S_r * F_l[i] - S_l * F_r[i] + S_l * S_r * (U_r[i] - U_l[i])) / (S_r - S_l);
            }
        }
    }
};

struct HLLC {
    KOKKOS_INLINE_FUNCTION
    static void calc_flux(rtype * flux, const rtype * n,
                          const rtype * W_l, const rtype * W_r, const rtype gamma) {
        rtype U_l[N_CONSERVATIVE], U_r[N_CONSERVATIVE];
        rtype F_l[N_CONSERVATIVE], F_r[N_CONSERVATIVE];
        physical_flux(W_l, n, gamma, U_l, F_l);
        physical_flux(W_r, n, gamma, U_r, F_r);
        const rtype u_l_n = W_l[1] * n[0] + W_l[2] * n[1];
        const rtype u_r_n = W_r[1] * n[0] + W_r[2] * n[1];
        rtype S_l, S_r;
        wave_speeds(W_l, W_r, u_l_n, u_r_n, gamma, S_l, S_r);
        if (0.0 <= S_l) {
            FOR_I_CONSERVATIVE flux[i] = F_l[i];
            return;
        }
        if (S_r <= 0.0) {
            FOR_I_CONSERVATIVE flux[i] = F_r[i];
            return;
        }
        // Toro 10.37 and 10.38-10.39 (star states, "variant 1")
        const rtype m_l = W_l[0] * (S_l - u_l_n);
        const rtype m_r = W_r[0] * (S_r - u_r_n);
        const rtype S_star = (W_r[3] - W_l[3] + u_l_n * m_l - u_r_n * m_r) / (m_l - m_r);
        const bool left = (S_star >= 0.0);
        const rtype * W = left ? W_l : W_r;
        const rtype * U = left ? U_l : U_r;
        const rtype * F = left ? F_l : F_r;
        const rtype S = left ? S_l : S_r;
        const rtype u_n = left ? u_l_n : u_r_n;
        const rtype coeff = W[0] * (S - u_n) / (S - S_star);
        rtype U_star[N_CONSERVATIVE];
        U_star[0] = coeff;
        U_star[1] = coeff * (W[1] + (S_star - u_n) * n[0]);
        U_star[2] = coeff * (W[2] + (S_star - u_n) * n[1]);
        U_star[3] = coeff * (U[3] / W[0] + (S_star - u_n) * (S_star + W[3] / (W[0] * (S - u_n))));
        FOR_I_CONSERVATIVE flux[i] = F[i] + S * (U_star[i] - U[i]);
    }
};

} // namespace riemann

#endif // RIEMANN_SOLVER_H

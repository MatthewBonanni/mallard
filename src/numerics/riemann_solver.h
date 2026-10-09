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
#include "teno.h"

enum class RiemannSolverType {
    RUSANOV,
    HLL,
    HLLC,
    ROE,
    RHLL,
};

static const std::unordered_map<std::string, RiemannSolverType> RIEMANN_SOLVER_TYPES = {
    {"Rusanov", RiemannSolverType::RUSANOV},
    {"HLL", RiemannSolverType::HLL},
    {"HLLC", RiemannSolverType::HLLC},
    {"Roe", RiemannSolverType::ROE},
    {"RHLL", RiemannSolverType::RHLL}
};

static const std::unordered_map<RiemannSolverType, std::string> RIEMANN_SOLVER_NAMES = {
    {RiemannSolverType::RUSANOV, "Rusanov"},
    {RiemannSolverType::HLL, "HLL"},
    {RiemannSolverType::HLLC, "HLLC"},
    {RiemannSolverType::ROE, "Roe"},
    {RiemannSolverType::RHLL, "RHLL"}
};

/**
 * All solvers take left/right states as W = [rho, u, p] (u with N_DIM
 * components) and a unit normal pointing from left to right, and return the
 * flux of [rho, rho u, rho E] through the face per unit area.
 */
namespace riemann {

/**
 * @brief Physical Euler flux of state W through unit normal n.
 */
KOKKOS_INLINE_FUNCTION
void physical_flux(const rtype * W, const rtype * n, const rtype gamma,
                   rtype * U, rtype * F) {
    constexpr uint8_t E = N_DIM + 1;
    const rtype u_n = dot<N_DIM>(W + 1, n);
    U[0] = W[0];
    FOR_I_DIM U[1 + i] = W[0] * W[1 + i];
    U[E] = W[E] / (gamma - 1.0_r) + 0.5_r * W[0] * dot<N_DIM>(W + 1, W + 1);
    F[0] = U[0] * u_n;
    FOR_I_DIM F[1 + i] = U[1 + i] * u_n + W[E] * n[i];
    F[E] = (U[E] + W[E]) * u_n;
}

/**
 * @brief Einfeldt (HLLE) wave speed estimates using Roe averages
 *        (Einfeldt et al. 1991; Toro 10.52).
 */
KOKKOS_INLINE_FUNCTION
void wave_speeds_einfeldt(const rtype * W_l, const rtype * W_r, const rtype u_l_n, const rtype u_r_n,
                          const rtype gamma, rtype & S_l, rtype & S_r) {
    const rtype a_l = Kokkos::sqrt(gamma * W_l[N_DIM + 1] / W_l[0]);
    const rtype a_r = Kokkos::sqrt(gamma * W_r[N_DIM + 1] / W_r[0]);
    const rtype s_l = Kokkos::sqrt(W_l[0]);
    const rtype s_r = Kokkos::sqrt(W_r[0]);
    const rtype H_l = a_l * a_l / (gamma - 1.0_r) + 0.5_r * dot<N_DIM>(W_l + 1, W_l + 1);
    const rtype H_r = a_r * a_r / (gamma - 1.0_r) + 0.5_r * dot<N_DIM>(W_r + 1, W_r + 1);
    rtype u_roe[N_DIM];
    FOR_I_DIM u_roe[i] = (s_l * W_l[1 + i] + s_r * W_r[1 + i]) / (s_l + s_r);
    const rtype H_roe = (s_l * H_l + s_r * H_r) / (s_l + s_r);
    const rtype un_roe = (s_l * u_l_n + s_r * u_r_n) / (s_l + s_r);
    const rtype a_roe = Kokkos::sqrt(Kokkos::fmax((gamma - 1.0_r) * (H_roe - 0.5_r * dot<N_DIM>(u_roe, u_roe)), 0.0_r));
    S_l = Kokkos::fmin(u_l_n - a_l, un_roe - a_roe);
    S_r = Kokkos::fmax(u_r_n + a_r, un_roe + a_roe);
}

/**
 * @brief Thermodynamics of one side of a face for a gas mixture: the frozen
 *        ratio of specific heats and the energy offset, with
 *        rho E = p / (gamma - 1) + rho e0 + rho |u|^2 / 2 (e0 = 0 for a
 *        calorically perfect gas), and the pair the wave-speed estimates use:
 *        the same but in double flux, where both fluxes of a face estimate
 *        the waves with each side's own cell's, so that their mass fluxes
 *        agree at contacts.
 */
struct SideThermo {
    rtype gamma;
    rtype e0;
    rtype gamma_waves;
    rtype e0_waves;

    KOKKOS_INLINE_FUNCTION
    SideThermo(const rtype gamma_in, const rtype e0_in)
        : gamma(gamma_in), e0(e0_in), gamma_waves(gamma_in), e0_waves(e0_in) {}

    KOKKOS_INLINE_FUNCTION
    SideThermo(const SideThermo & eos, const SideThermo & waves)
        : gamma(eos.gamma), e0(eos.e0), gamma_waves(waves.gamma), e0_waves(waves.e0) {}
};

/**
 * @brief Physical Euler flux of state W of a mixture side through unit normal n.
 */
KOKKOS_INLINE_FUNCTION
void physical_flux(const rtype * W, const rtype * n, const SideThermo & th, rtype * U, rtype * F) {
    constexpr uint8_t E = N_DIM + 1;
    const rtype u_n = dot<N_DIM>(W + 1, n);
    U[0] = W[0];
    FOR_I_DIM U[1 + i] = W[0] * W[1 + i];
    U[E] = W[E] / (th.gamma - 1.0_r) + W[0] * th.e0 + 0.5_r * W[0] * dot<N_DIM>(W + 1, W + 1);
    F[0] = U[0] * u_n;
    FOR_I_DIM F[1 + i] = U[1 + i] * u_n + W[E] * n[i];
    F[E] = (U[E] + W[E]) * u_n;
}

/**
 * @brief Einfeldt wave speed estimates for a mixture: the Roe average of the
 *        perfect-gas estimates with sqrt(rho)-weighted averages of gamma and
 *        e0, a^2 = (gamma - 1) (H - e0 - |u|^2 / 2). Equal to the perfect-gas
 *        estimates when both sides share gamma and e0 = 0.
 */
KOKKOS_INLINE_FUNCTION
void wave_speeds_einfeldt(const rtype * W_l, const rtype * W_r, const rtype u_l_n, const rtype u_r_n,
                          const SideThermo & th_l, const SideThermo & th_r, rtype & S_l, rtype & S_r) {
    const rtype a_l = Kokkos::sqrt(th_l.gamma_waves * W_l[N_DIM + 1] / W_l[0]);
    const rtype a_r = Kokkos::sqrt(th_r.gamma_waves * W_r[N_DIM + 1] / W_r[0]);
    const rtype s_l = Kokkos::sqrt(W_l[0]);
    const rtype s_r = Kokkos::sqrt(W_r[0]);
    const rtype H_l = a_l * a_l / (th_l.gamma_waves - 1.0_r) + th_l.e0_waves + 0.5_r * dot<N_DIM>(W_l + 1, W_l + 1);
    const rtype H_r = a_r * a_r / (th_r.gamma_waves - 1.0_r) + th_r.e0_waves + 0.5_r * dot<N_DIM>(W_r + 1, W_r + 1);
    rtype u_roe[N_DIM];
    FOR_I_DIM u_roe[i] = (s_l * W_l[1 + i] + s_r * W_r[1 + i]) / (s_l + s_r);
    const rtype H_roe = (s_l * H_l + s_r * H_r) / (s_l + s_r);
    const rtype gamma_roe = (s_l * th_l.gamma_waves + s_r * th_r.gamma_waves) / (s_l + s_r);
    const rtype e0_roe = (s_l * th_l.e0_waves + s_r * th_r.e0_waves) / (s_l + s_r);
    const rtype un_roe = (s_l * u_l_n + s_r * u_r_n) / (s_l + s_r);
    const rtype a_roe = Kokkos::sqrt(
        Kokkos::fmax((gamma_roe - 1.0_r) * (H_roe - e0_roe - 0.5_r * dot<N_DIM>(u_roe, u_roe)), 0.0_r));
    S_l = Kokkos::fmin(u_l_n - a_l, un_roe - a_roe);
    S_r = Kokkos::fmax(u_r_n + a_r, un_roe + a_roe);
}

struct Rusanov {
    /** @brief Mixture flux with per-side thermodynamics. */
    KOKKOS_INLINE_FUNCTION
    static void calc_flux(rtype * flux, const rtype * n, const rtype * W_l, const rtype * W_r,
                          const SideThermo & th_l, const SideThermo & th_r) {
        rtype U_l[N_CONSERVATIVE], U_r[N_CONSERVATIVE];
        rtype F_l[N_CONSERVATIVE], F_r[N_CONSERVATIVE];
        physical_flux(W_l, n, th_l, U_l, F_l);
        physical_flux(W_r, n, th_r, U_r, F_r);
        const rtype u_l_n = dot<N_DIM>(W_l + 1, n);
        const rtype u_r_n = dot<N_DIM>(W_r + 1, n);
        const rtype a_l = Kokkos::sqrt(th_l.gamma_waves * W_l[N_DIM + 1] / W_l[0]);
        const rtype a_r = Kokkos::sqrt(th_r.gamma_waves * W_r[N_DIM + 1] / W_r[0]);
        const rtype S_max = Kokkos::fmax(Kokkos::fabs(u_l_n) + a_l, Kokkos::fabs(u_r_n) + a_r);
        FOR_I_CONSERVATIVE flux[i] = 0.5_r * (F_l[i] + F_r[i] - S_max * (U_r[i] - U_l[i]));
    }

    KOKKOS_INLINE_FUNCTION
    static void calc_flux(rtype * flux, const rtype * n,
                          const rtype * W_l, const rtype * W_r, const rtype gamma) {
        rtype U_l[N_CONSERVATIVE], U_r[N_CONSERVATIVE];
        rtype F_l[N_CONSERVATIVE], F_r[N_CONSERVATIVE];
        physical_flux(W_l, n, gamma, U_l, F_l);
        physical_flux(W_r, n, gamma, U_r, F_r);
        const rtype u_l_n = dot<N_DIM>(W_l + 1, n);
        const rtype u_r_n = dot<N_DIM>(W_r + 1, n);
        const rtype a_l = Kokkos::sqrt(gamma * W_l[N_DIM + 1] / W_l[0]);
        const rtype a_r = Kokkos::sqrt(gamma * W_r[N_DIM + 1] / W_r[0]);
        const rtype S_max = Kokkos::fmax(Kokkos::fabs(u_l_n) + a_l, Kokkos::fabs(u_r_n) + a_r);
        FOR_I_CONSERVATIVE flux[i] = 0.5_r * (F_l[i] + F_r[i] - S_max * (U_r[i] - U_l[i]));
    }
};

struct HLL {
    /** @brief Mixture flux with per-side thermodynamics. */
    KOKKOS_INLINE_FUNCTION
    static void calc_flux(rtype * flux, const rtype * n, const rtype * W_l, const rtype * W_r,
                          const SideThermo & th_l, const SideThermo & th_r) {
        rtype U_l[N_CONSERVATIVE], U_r[N_CONSERVATIVE];
        rtype F_l[N_CONSERVATIVE], F_r[N_CONSERVATIVE];
        physical_flux(W_l, n, th_l, U_l, F_l);
        physical_flux(W_r, n, th_r, U_r, F_r);
        const rtype u_l_n = dot<N_DIM>(W_l + 1, n);
        const rtype u_r_n = dot<N_DIM>(W_r + 1, n);
        rtype S_l, S_r;
        wave_speeds_einfeldt(W_l, W_r, u_l_n, u_r_n, th_l, th_r, S_l, S_r);
        if (0.0_r <= S_l) {
            FOR_I_CONSERVATIVE flux[i] = F_l[i];
        } else if (S_r <= 0.0_r) {
            FOR_I_CONSERVATIVE flux[i] = F_r[i];
        } else {
            FOR_I_CONSERVATIVE {
                flux[i] = (S_r * F_l[i] - S_l * F_r[i] + S_l * S_r * (U_r[i] - U_l[i])) / (S_r - S_l);
            }
        }
    }

    KOKKOS_INLINE_FUNCTION
    static void calc_flux(rtype * flux, const rtype * n,
                          const rtype * W_l, const rtype * W_r, const rtype gamma) {
        rtype U_l[N_CONSERVATIVE], U_r[N_CONSERVATIVE];
        rtype F_l[N_CONSERVATIVE], F_r[N_CONSERVATIVE];
        physical_flux(W_l, n, gamma, U_l, F_l);
        physical_flux(W_r, n, gamma, U_r, F_r);
        const rtype u_l_n = dot<N_DIM>(W_l + 1, n);
        const rtype u_r_n = dot<N_DIM>(W_r + 1, n);
        rtype S_l, S_r;
        wave_speeds_einfeldt(W_l, W_r, u_l_n, u_r_n, gamma, S_l, S_r);
        if (0.0_r <= S_l) {
            FOR_I_CONSERVATIVE flux[i] = F_l[i];
        } else if (S_r <= 0.0_r) {
            FOR_I_CONSERVATIVE flux[i] = F_r[i];
        } else {
            FOR_I_CONSERVATIVE {
                flux[i] = (S_r * F_l[i] - S_l * F_r[i] + S_l * S_r * (U_r[i] - U_l[i])) / (S_r - S_l);
            }
        }
    }
};

struct HLLC {
    /**
     * @brief Mixture flux with per-side thermodynamics. The star states
     *        (Toro 10.38-10.39) hold for any equation of state.
     */
    KOKKOS_INLINE_FUNCTION
    static void calc_flux(rtype * flux, const rtype * n, const rtype * W_l, const rtype * W_r,
                          const SideThermo & th_l, const SideThermo & th_r) {
        rtype U_l[N_CONSERVATIVE], U_r[N_CONSERVATIVE];
        rtype F_l[N_CONSERVATIVE], F_r[N_CONSERVATIVE];
        physical_flux(W_l, n, th_l, U_l, F_l);
        physical_flux(W_r, n, th_r, U_r, F_r);
        const rtype u_l_n = dot<N_DIM>(W_l + 1, n);
        const rtype u_r_n = dot<N_DIM>(W_r + 1, n);
        rtype S_l, S_r;
        wave_speeds_einfeldt(W_l, W_r, u_l_n, u_r_n, th_l, th_r, S_l, S_r);
        if (0.0_r <= S_l) {
            FOR_I_CONSERVATIVE flux[i] = F_l[i];
            return;
        }
        if (S_r <= 0.0_r) {
            FOR_I_CONSERVATIVE flux[i] = F_r[i];
            return;
        }
        const rtype m_l = W_l[0] * (S_l - u_l_n);
        const rtype m_r = W_r[0] * (S_r - u_r_n);
        constexpr uint8_t E = N_DIM + 1;
        const rtype S_star = (W_r[E] - W_l[E] + u_l_n * m_l - u_r_n * m_r) / (m_l - m_r);
        const bool left = (S_star >= 0.0_r);
        const rtype * W = left ? W_l : W_r;
        const rtype * U = left ? U_l : U_r;
        const rtype * F = left ? F_l : F_r;
        const rtype S = left ? S_l : S_r;
        const rtype u_n = left ? u_l_n : u_r_n;
        const rtype coeff = W[0] * (S - u_n) / (S - S_star);
        rtype U_star[N_CONSERVATIVE];
        U_star[0] = coeff;
        FOR_I_DIM U_star[1 + i] = coeff * (W[1 + i] + (S_star - u_n) * n[i]);
        U_star[E] = coeff * (U[E] / W[0] + (S_star - u_n) * (S_star + W[E] / (W[0] * (S - u_n))));
        FOR_I_CONSERVATIVE flux[i] = F[i] + S * (U_star[i] - U[i]);
    }

    KOKKOS_INLINE_FUNCTION
    static void calc_flux(rtype * flux, const rtype * n,
                          const rtype * W_l, const rtype * W_r, const rtype gamma) {
        rtype U_l[N_CONSERVATIVE], U_r[N_CONSERVATIVE];
        rtype F_l[N_CONSERVATIVE], F_r[N_CONSERVATIVE];
        physical_flux(W_l, n, gamma, U_l, F_l);
        physical_flux(W_r, n, gamma, U_r, F_r);
        const rtype u_l_n = dot<N_DIM>(W_l + 1, n);
        const rtype u_r_n = dot<N_DIM>(W_r + 1, n);
        rtype S_l, S_r;
        wave_speeds_einfeldt(W_l, W_r, u_l_n, u_r_n, gamma, S_l, S_r);
        if (0.0_r <= S_l) {
            FOR_I_CONSERVATIVE flux[i] = F_l[i];
            return;
        }
        if (S_r <= 0.0_r) {
            FOR_I_CONSERVATIVE flux[i] = F_r[i];
            return;
        }
        // Toro 10.37 and 10.38-10.39 (star states, "variant 1")
        const rtype m_l = W_l[0] * (S_l - u_l_n);
        const rtype m_r = W_r[0] * (S_r - u_r_n);
        constexpr uint8_t E = N_DIM + 1;
        const rtype S_star = (W_r[E] - W_l[E] + u_l_n * m_l - u_r_n * m_r) / (m_l - m_r);
        const bool left = (S_star >= 0.0_r);
        const rtype * W = left ? W_l : W_r;
        const rtype * U = left ? U_l : U_r;
        const rtype * F = left ? F_l : F_r;
        const rtype S = left ? S_l : S_r;
        const rtype u_n = left ? u_l_n : u_r_n;
        const rtype coeff = W[0] * (S - u_n) / (S - S_star);
        rtype U_star[N_CONSERVATIVE];
        U_star[0] = coeff;
        FOR_I_DIM U_star[1 + i] = coeff * (W[1 + i] + (S_star - u_n) * n[i]);
        U_star[E] = coeff * (U[E] / W[0] + (S_star - u_n) * (S_star + W[E] / (W[0] * (S - u_n))));
        FOR_I_CONSERVATIVE flux[i] = F[i] + S * (U_star[i] - U[i]);
    }
};

/**
 * @brief Roe's approximate Riemann solver with Harten's entropy fix on the
 *        acoustic waves.
 */
struct Roe {
    /**
     * @brief Mixture flux with per-side thermodynamics, in jump form:
     *        F = (F_l + F_r - |u_n| dU - sum_+- (|lambda_+-| - |u_n|) alpha_+- r_+-) / 2.
     *        The entropy, shear and composition waves (a jump of gamma and
     *        e0 at fixed p, u) all move at u_n, so together they carry
     *        dU minus the acoustic waves whatever the equation of state
     *        (Glaister 1988; Shuen, Liou & van Leer 1990), and the Roe
     *        property holds for any averaged sound speed: a_roe of
     *        wave_speeds_einfeldt, from gamma and e0. The perfect-gas solver
     *        up to round-off when both sides share gamma and e0 = 0.
     */
    KOKKOS_INLINE_FUNCTION
    static void calc_flux(rtype * flux, const rtype * n, const rtype * W_l, const rtype * W_r,
                          const SideThermo & th_l, const SideThermo & th_r) {
        constexpr uint8_t E = N_DIM + 1;
        rtype U_l[N_CONSERVATIVE], U_r[N_CONSERVATIVE];
        rtype F_l[N_CONSERVATIVE], F_r[N_CONSERVATIVE];
        physical_flux(W_l, n, th_l, U_l, F_l);
        physical_flux(W_r, n, th_r, U_r, F_r);

        const rtype s_l = Kokkos::sqrt(W_l[0]);
        const rtype s_r = Kokkos::sqrt(W_r[0]);
        const rtype w_l = s_l / (s_l + s_r);
        const rtype w_r = s_r / (s_l + s_r);
        rtype u[N_DIM];
        FOR_I_DIM u[i] = w_l * W_l[1 + i] + w_r * W_r[1 + i];
        const rtype H = w_l * (U_l[E] + W_l[E]) / W_l[0] + w_r * (U_r[E] + W_r[E]) / W_r[0];
        const rtype gamma = w_l * th_l.gamma + w_r * th_r.gamma;
        const rtype e0 = w_l * th_l.e0 + w_r * th_r.e0;
        const rtype a2 = Kokkos::fmax((gamma - 1.0_r) * (H - e0 - 0.5_r * dot<N_DIM>(u, u)), 1e-14_r);
        const rtype a = Kokkos::sqrt(a2);
        const rtype rho = s_l * s_r;
        const rtype u_n = dot<N_DIM>(u, n);

        const rtype dp = W_r[E] - W_l[E];
        const rtype du_n = dot<N_DIM>(W_r + 1, n) - dot<N_DIM>(W_l + 1, n);
        const rtype delta = 0.1_r * a;
        const rtype l_c = Kokkos::fabs(u_n);
        FOR_I_CONSERVATIVE flux[i] = l_c * (U_r[i] - U_l[i]);
        for (uint8_t k = 0; k < 2; k++) {
            const rtype sign = k ? 1.0_r : -1.0_r;
            rtype l = Kokkos::fabs(u_n + sign * a);
            if (l < delta) l = 0.5_r * (l * l + delta * delta) / delta;
            const rtype c = (l - l_c) * (dp + sign * rho * a * du_n) / (2.0_r * a2);
            flux[0] += c;
            FOR_I_DIM flux[1 + i] += c * (u[i] + sign * a * n[i]);
            flux[E] += c * (H + sign * a * u_n);
        }
        FOR_I_CONSERVATIVE flux[i] = 0.5_r * (F_l[i] + F_r[i] - flux[i]);
    }

    KOKKOS_INLINE_FUNCTION
    static void calc_flux(rtype * flux, const rtype * n,
                          const rtype * W_l, const rtype * W_r, const rtype gamma) {
        rtype U_l[N_CONSERVATIVE], U_r[N_CONSERVATIVE];
        rtype F_l[N_CONSERVATIVE], F_r[N_CONSERVATIVE];
        physical_flux(W_l, n, gamma, U_l, F_l);
        physical_flux(W_r, n, gamma, U_r, F_r);

        // Roe-averaged state, expressed as W = [rho, u, p] with a matching sound speed
        constexpr uint8_t E = N_DIM + 1;
        const rtype s_l = Kokkos::sqrt(W_l[0]);
        const rtype s_r = Kokkos::sqrt(W_r[0]);
        const rtype H_l = (U_l[E] + W_l[E]) / W_l[0];
        const rtype H_r = (U_r[E] + W_r[E]) / W_r[0];
        rtype W_roe[N_CONSERVATIVE];
        W_roe[0] = s_l * s_r;
        FOR_I_DIM W_roe[1 + i] = (s_l * W_l[1 + i] + s_r * W_r[1 + i]) / (s_l + s_r);
        const rtype H = (s_l * H_l + s_r * H_r) / (s_l + s_r);
        const rtype a2 = Kokkos::fmax((gamma - 1.0_r) * (H - 0.5_r * dot<N_DIM>(W_roe + 1, W_roe + 1)), 1e-14_r);
        W_roe[E] = W_roe[0] * a2 / gamma;
        const rtype a = Kokkos::sqrt(a2);

        rtype L[N_CONSERVATIVE][N_CONSERVATIVE], R[N_CONSERVATIVE][N_CONSERVATIVE];
        teno::eigenvectors(W_roe, n, gamma, L, R);
        const rtype u_n = dot<N_DIM>(W_roe + 1, n);
        rtype lambda[N_CONSERVATIVE];
        FOR_I_CONSERVATIVE lambda[i] = u_n;
        lambda[0] = u_n - a;
        lambda[2] = u_n + a;
        const rtype delta = 0.1_r * a;
        for (uint8_t k = 0; k < N_CONSERVATIVE; k++) {
            rtype l = Kokkos::fabs(lambda[k]);
            if ((k == 0 || k == 2) && l < delta) l = 0.5_r * (l * l + delta * delta) / delta;
            lambda[k] = l;
        }
        rtype strength[N_CONSERVATIVE];
        FOR_I_CONSERVATIVE {
            strength[i] = 0.0;
            for (uint8_t j = 0; j < N_CONSERVATIVE; j++) strength[i] += L[i][j] * (U_r[j] - U_l[j]);
        }
        FOR_I_CONSERVATIVE {
            rtype dissipation = 0.0;
            for (uint8_t k = 0; k < N_CONSERVATIVE; k++) dissipation += R[i][k] * lambda[k] * strength[k];
            flux[i] = 0.5_r * (F_l[i] + F_r[i] - dissipation);
        }
    }
};

/**
 * @brief Rotated-hybrid HLL-Roe solver (Nishikawa & Kitamura, J. Comput.
 *        Phys. 227, 2008): HLL along the direction of the velocity difference
 *        (normal to shocks, where Roe's lack of dissipation causes carbuncles)
 *        and Roe across it (shear layers and contacts stay sharp).
 */
struct RHLL {
    KOKKOS_INLINE_FUNCTION
    static void calc_flux(rtype * flux, const rtype * n,
                          const rtype * W_l, const rtype * W_r, const rtype gamma) {
        const rtype a_ref = Kokkos::sqrt(gamma * Kokkos::fmax(W_l[N_DIM + 1] / W_l[0], W_r[N_DIM + 1] / W_r[0]));
        shock_blended(flux, n, W_l, W_r, a_ref, gamma);
    }

    /** @brief Mixture flux with per-side thermodynamics. */
    KOKKOS_INLINE_FUNCTION
    static void calc_flux(rtype * flux, const rtype * n, const rtype * W_l, const rtype * W_r,
                          const SideThermo & th_l, const SideThermo & th_r) {
        const rtype a_ref = Kokkos::sqrt(Kokkos::fmax(th_l.gamma_waves * W_l[N_DIM + 1] / W_l[0],
                                                      th_r.gamma_waves * W_r[N_DIM + 1] / W_r[0]));
        shock_blended(flux, n, W_l, W_r, a_ref, th_l, th_r);
    }

    /**
     * @brief The rotated hybrid, blended toward HLL along the face normal by
     *        the pressure jump across the face: w F_rotated + (1 - w) F_HLL(n),
     *        w = min(p_l / p_r, p_r / p_l)^3 (the pressure weight of AUSMPW+:
     *        Kim, Kim & Rho, J. Comput. Phys. 174, 2001). On faces aligned with a
     *        shock the rotation is already HLL along n; on tetrahedra it turns
     *        HLL toward the shock normal, which grows a carbuncle on the
     *        stagnation line with MUSCL or TENO states (#80). Contacts and shear
     *        layers carry no pressure jump and keep the rotated flux.
     */
    template <typename... Thermo>
    KOKKOS_INLINE_FUNCTION static void shock_blended(rtype * flux, const rtype * n, const rtype * W_l,
                                                     const rtype * W_r, const rtype a_ref, const Thermo &... th) {
        rotated(flux, n, W_l, W_r, a_ref, th...);
        const rtype ratio = Kokkos::fmin(W_l[N_DIM + 1] / W_r[N_DIM + 1], W_r[N_DIM + 1] / W_l[N_DIM + 1]);
        const rtype w = ratio * ratio * ratio;
        if (w >= 1.0_r) return;
        rtype f_hll[N_CONSERVATIVE];
        HLL::calc_flux(f_hll, n, W_l, W_r, th...);
        FOR_I_CONSERVATIVE flux[i] = w * flux[i] + (1.0_r - w) * f_hll[i];
    }

    /**
     * @brief The hybrid for either form of the thermodynamics th, passed on
     *        to HLL and Roe; a_ref scales the threshold below which the
     *        velocity jump falls back to the face normal.
     */
    template <typename... Thermo>
    KOKKOS_INLINE_FUNCTION static void rotated(rtype * flux, const rtype * n, const rtype * W_l, const rtype * W_r,
                                               const rtype a_ref, const Thermo &... th) {
        if constexpr (N_DIM == 3) {
            rotated_3d(flux, n, W_l, W_r, a_ref, th...);
            return;
        }
        const rtype dq[N_DIM] = {W_r[1] - W_l[1], W_r[2] - W_l[2]};
        const rtype dq_mag = Kokkos::sqrt(dq[0] * dq[0] + dq[1] * dq[1]);
        rtype n1[N_DIM];
        if (dq_mag > precision_tol(1e-12, 1e-5) * a_ref) {
            n1[0] = dq[0] / dq_mag;
            n1[1] = dq[1] / dq_mag;
        } else {
            // No velocity jump: fall back to the face normal (pure HLL)
            n1[0] = n[0];
            n1[1] = n[1];
        }
        rtype alpha1 = n1[0] * n[0] + n1[1] * n[1];
        if (alpha1 < 0.0_r) {
            n1[0] = -n1[0];
            n1[1] = -n1[1];
            alpha1 = -alpha1;
        }
        rtype n2[N_DIM] = {-n1[1], n1[0]};
        rtype alpha2 = n2[0] * n[0] + n2[1] * n[1];
        if (alpha2 < 0.0_r) {
            n2[0] = -n2[0];
            n2[1] = -n2[1];
            alpha2 = -alpha2;
        }
        rtype f1[N_CONSERVATIVE], f2[N_CONSERVATIVE];
        HLL::calc_flux(f1, n1, W_l, W_r, th...);
        Roe::calc_flux(f2, n2, W_l, W_r, th...);
        FOR_I_CONSERVATIVE flux[i] = alpha1 * f1[i] + alpha2 * f2[i];
    }

    /**
     * @brief 3D variant: n1 is the unit velocity difference (or n), n2 the unit
     *        vector orthogonal to n1 in the plane of n1 and n, both oriented
     *        along n. Roe is dropped when n1 is parallel to n.
     */
    template <typename... Thermo>
    KOKKOS_INLINE_FUNCTION static void rotated_3d(rtype * flux, const rtype * n, const rtype * W_l,
                                                  const rtype * W_r, const rtype a_ref, const Thermo &... th) {
        rtype dq[N_DIM], n1[N_DIM], n2[N_DIM];
        FOR_I_DIM dq[i] = W_r[1 + i] - W_l[1 + i];
        const rtype dq_mag = norm_2<N_DIM>(dq);
        FOR_I_DIM n1[i] = (dq_mag > precision_tol(1e-12, 1e-5) * a_ref) ? dq[i] / dq_mag : n[i];
        rtype alpha1 = dot<N_DIM>(n1, n);
        if (alpha1 < 0.0_r) {
            FOR_I_DIM n1[i] = -n1[i];
            alpha1 = -alpha1;
        }
        FOR_I_DIM n2[i] = n[i] - alpha1 * n1[i];
        const rtype alpha2 = norm_2<N_DIM>(n2);
        rtype f1[N_CONSERVATIVE];
        HLL::calc_flux(f1, n1, W_l, W_r, th...);
        if (alpha2 <= precision_tol(1e-12, 1e-6)) {
            FOR_I_CONSERVATIVE flux[i] = alpha1 * f1[i];
            return;
        }
        FOR_I_DIM n2[i] /= alpha2;
        rtype f2[N_CONSERVATIVE];
        Roe::calc_flux(f2, n2, W_l, W_r, th...);
        FOR_I_CONSERVATIVE flux[i] = alpha1 * f1[i] + alpha2 * f2[i];
    }
};

/**
 * @brief Kinetic-energy- and entropy-preserving central flux (KEEP; Kuya,
 *        Totani & Kawai, J. Comput. Phys. 375, 2018) with arithmetic means:
 *        mass C = mean(rho) mean(u).n, momentum C mean(u) + mean(p) n, energy
 *        C (mean(e) + u_L . u_R / 2) + (p_L u_R.n + p_R u_L.n) / 2. Its
 *        momentum flux has Jameson's form, so with cell values on both sides
 *        the convective fluxes change the kinetic energy only by the pressure
 *        work mean(p) (u_R - u_L).n per face. No dissipation: for the hybrid
 *        convective flux (docs/design/les.md, section 4.3).
 */
struct KEEP {
    KOKKOS_INLINE_FUNCTION
    static void central(rtype * flux, const rtype * n, const rtype * W_l, const rtype * W_r, const rtype e_l,
                        const rtype e_r) {
        constexpr uint8_t E = N_DIM + 1;
        rtype u[N_DIM];
        FOR_I_DIM u[i] = 0.5_r * (W_l[1 + i] + W_r[1 + i]);
        const rtype C = 0.5_r * (W_l[0] + W_r[0]) * dot<N_DIM>(u, n);
        const rtype p = 0.5_r * (W_l[E] + W_r[E]);
        flux[0] = C;
        FOR_I_DIM flux[1 + i] = C * u[i] + p * n[i];
        flux[E] = C * (0.5_r * (e_l + e_r) + 0.5_r * dot<N_DIM>(W_l + 1, W_r + 1)) +
                  0.5_r * (W_l[E] * dot<N_DIM>(W_r + 1, n) + W_r[E] * dot<N_DIM>(W_l + 1, n));
    }

    KOKKOS_INLINE_FUNCTION
    static void calc_flux(rtype * flux, const rtype * n, const rtype * W_l, const rtype * W_r, const rtype gamma) {
        constexpr uint8_t E = N_DIM + 1;
        central(flux, n, W_l, W_r, W_l[E] / ((gamma - 1.0_r) * W_l[0]), W_r[E] / ((gamma - 1.0_r) * W_r[0]));
    }

    /** @brief Mixture flux with per-side thermodynamics, e = p / ((gamma - 1) rho) + e0. */
    KOKKOS_INLINE_FUNCTION
    static void calc_flux(rtype * flux, const rtype * n, const rtype * W_l, const rtype * W_r,
                          const SideThermo & th_l, const SideThermo & th_r) {
        constexpr uint8_t E = N_DIM + 1;
        central(flux, n, W_l, W_r, W_l[E] / ((th_l.gamma - 1.0_r) * W_l[0]) + th_l.e0,
                W_r[E] / ((th_r.gamma - 1.0_r) * W_r[0]) + th_r.e0);
    }
};

} // namespace riemann

#endif // RIEMANN_SOLVER_H

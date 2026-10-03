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
 * @brief Primitive variable Riemann solver (Toro 9.20, 9.28).
 */
KOKKOS_INLINE_FUNCTION
rtype PVRS(const rtype * W_l, const rtype * W_r, const rtype gamma) {
    const rtype a_l = Kokkos::sqrt(gamma * W_l[2] / W_l[0]);
    const rtype a_r = Kokkos::sqrt(gamma * W_r[2] / W_r[0]);
    const rtype rho_avg = 0.5_r * (W_l[0] + W_r[0]);
    const rtype a_avg = 0.5_r * (a_l + a_r);
    return 0.5_r * (W_l[2] + W_r[2]) + 0.5_r * (W_l[1] - W_r[1]) * rho_avg * a_avg;
}

/**
 * @brief Two-rarefaction Riemann solver (Toro 9.32).
 */
KOKKOS_INLINE_FUNCTION
rtype TRRS(const rtype * W_l, const rtype * W_r, const rtype gamma) {
    const rtype a_l = Kokkos::sqrt(gamma * W_l[2] / W_l[0]);
    const rtype a_r = Kokkos::sqrt(gamma * W_r[2] / W_r[0]);
    const rtype z = (gamma - 1.0_r) / (2.0_r * gamma);
    const rtype num = a_l + a_r - 0.5_r * (gamma - 1.0_r) * (W_r[1] - W_l[1]);
    const rtype den = a_l / Kokkos::pow(W_l[2], z) + a_r / Kokkos::pow(W_r[2], z);
    return Kokkos::pow(Kokkos::fmax(num, 0.0_r) / den, 1.0_r / z);
}

/**
 * @brief Two-shock Riemann solver (Toro 9.42), linearized about p_0.
 */
KOKKOS_INLINE_FUNCTION
rtype TSRS(const rtype * W_l, const rtype * W_r, const rtype gamma, const rtype p_0) {
    const rtype A_l = 2.0_r / ((gamma + 1.0_r) * W_l[0]);
    const rtype A_r = 2.0_r / ((gamma + 1.0_r) * W_r[0]);
    const rtype B_l = (gamma - 1.0_r) / (gamma + 1.0_r) * W_l[2];
    const rtype B_r = (gamma - 1.0_r) / (gamma + 1.0_r) * W_r[2];
    const rtype p = Kokkos::fmax(0.0_r, p_0);
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
    const rtype p_pv = Kokkos::fmax(0.0_r, PVRS(W_l, W_r, gamma));
    if ((p_max / p_min < q_user) && (p_min <= p_pv) && (p_pv <= p_max)) {
        return p_pv;
    } else if (p_pv < p_min) {
        return TRRS(W_l, W_r, gamma);
    } else {
        return TSRS(W_l, W_r, gamma, p_pv);
    }
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
 * @brief Pressure-based wave speed estimates (Toro 10.59-10.60).
 */
KOKKOS_INLINE_FUNCTION
void wave_speeds_pressure(const rtype * W_l, const rtype * W_r, const rtype u_l_n, const rtype u_r_n,
                          const rtype gamma, rtype & S_l, rtype & S_r) {
    constexpr uint8_t E = N_DIM + 1;
    const rtype w_l[3] = {W_l[0], u_l_n, W_l[E]};
    const rtype w_r[3] = {W_r[0], u_r_n, W_r[E]};
    const rtype p_star = ANRS(w_l, w_r, gamma);
    const rtype a_l = Kokkos::sqrt(gamma * W_l[E] / W_l[0]);
    const rtype a_r = Kokkos::sqrt(gamma * W_r[E] / W_r[0]);
    const rtype c = (gamma + 1.0_r) / (2.0_r * gamma);
    const rtype q_l = (p_star <= W_l[E]) ? 1.0_r : Kokkos::sqrt(1.0_r + c * (p_star / W_l[E] - 1.0_r));
    const rtype q_r = (p_star <= W_r[E]) ? 1.0_r : Kokkos::sqrt(1.0_r + c * (p_star / W_r[E] - 1.0_r));
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
        const rtype u_l_n = dot<N_DIM>(W_l + 1, n);
        const rtype u_r_n = dot<N_DIM>(W_r + 1, n);
        const rtype a_l = Kokkos::sqrt(gamma * W_l[N_DIM + 1] / W_l[0]);
        const rtype a_r = Kokkos::sqrt(gamma * W_r[N_DIM + 1] / W_r[0]);
        const rtype S_max = Kokkos::fmax(Kokkos::fabs(u_l_n) + a_l, Kokkos::fabs(u_r_n) + a_r);
        FOR_I_CONSERVATIVE flux[i] = 0.5_r * (F_l[i] + F_r[i] - S_max * (U_r[i] - U_l[i]));
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
        if constexpr (N_DIM == 3) {
            calc_flux_3d(flux, n, W_l, W_r, gamma);
            return;
        }
        const rtype dq[N_DIM] = {W_r[1] - W_l[1], W_r[2] - W_l[2]};
        const rtype dq_mag = Kokkos::sqrt(dq[0] * dq[0] + dq[1] * dq[1]);
        const rtype a_ref = Kokkos::sqrt(gamma * Kokkos::fmax(W_l[3] / W_l[0], W_r[3] / W_r[0]));
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
        HLL::calc_flux(f1, n1, W_l, W_r, gamma);
        Roe::calc_flux(f2, n2, W_l, W_r, gamma);
        FOR_I_CONSERVATIVE flux[i] = alpha1 * f1[i] + alpha2 * f2[i];
    }

    /**
     * @brief 3D variant: n1 is the unit velocity difference (or n), n2 the unit
     *        vector orthogonal to n1 in the plane of n1 and n, both oriented
     *        along n. Roe is dropped when n1 is parallel to n.
     */
    KOKKOS_INLINE_FUNCTION
    static void calc_flux_3d(rtype * flux, const rtype * n,
                             const rtype * W_l, const rtype * W_r, const rtype gamma) {
        rtype dq[N_DIM], n1[N_DIM], n2[N_DIM];
        FOR_I_DIM dq[i] = W_r[1 + i] - W_l[1 + i];
        const rtype dq_mag = norm_2<N_DIM>(dq);
        const rtype a_ref = Kokkos::sqrt(gamma * Kokkos::fmax(W_l[N_DIM + 1] / W_l[0], W_r[N_DIM + 1] / W_r[0]));
        FOR_I_DIM n1[i] = (dq_mag > precision_tol(1e-12, 1e-5) * a_ref) ? dq[i] / dq_mag : n[i];
        rtype alpha1 = dot<N_DIM>(n1, n);
        if (alpha1 < 0.0_r) {
            FOR_I_DIM n1[i] = -n1[i];
            alpha1 = -alpha1;
        }
        FOR_I_DIM n2[i] = n[i] - alpha1 * n1[i];
        const rtype alpha2 = norm_2<N_DIM>(n2);
        rtype f1[N_CONSERVATIVE];
        HLL::calc_flux(f1, n1, W_l, W_r, gamma);
        if (alpha2 <= precision_tol(1e-12, 1e-6)) {
            FOR_I_CONSERVATIVE flux[i] = alpha1 * f1[i];
            return;
        }
        FOR_I_DIM n2[i] /= alpha2;
        rtype f2[N_CONSERVATIVE];
        Roe::calc_flux(f2, n2, W_l, W_r, gamma);
        FOR_I_CONSERVATIVE flux[i] = alpha1 * f1[i] + alpha2 * f2[i];
    }
};

} // namespace riemann

#endif // RIEMANN_SOLVER_H

/**
 * @file tfles.h
 * @author Matthew Bonanni (mbonanni001@gmail.com)
 * @brief Dynamically thickened flame model (TFLES) of turbulence-chemistry
 *        interaction (docs/design/les.md, section 6).
 * @version 0.1
 * @date 2026-10-05
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 *
 */

#ifndef TFLES_H
#define TFLES_H

#include <Kokkos_Core.hpp>
#include <toml.hpp>

#include "common.h"
#include "log.h"

/**
 * @brief Thickening F, efficiency E and flame sensor Omega of a cell. The
 *        flame is thickened by F where Omega = 1 so that it spans n_res cells:
 *        diffusivities and conductivity times E F, reaction rates times E / F
 *        (laminar speed times E, thickness times F). E is the SGS wrinkling of
 *        Charlette, Meneveau & Veynante (Combust. Flame 131, 2002) with
 *        beta = 0.5 at the filter width F delta_L. Plain aggregate, captured
 *        by value in kernels.
 */
struct ThickenedFlame {
    double delta_L = 0.0;  // laminar thermal thickness (m)
    double s_L = 0.0;      // laminar flame speed (m/s)
    double T_u = 0.0, T_b = 0.0;
    double n_res = 5.0;
    double beta = 0.5;
    bool efficiency = true;
    bool eddy_viscosity_velocity = false;  // u' = C_u nu_t / Delta instead of Colin et al.'s operator
    double C_u = 28.0;

    static constexpr double C_SENSOR = 0.05;  // progress variable at which Omega reaches 1
    static constexpr double C_K = 1.5;        // Kolmogorov constant of the efficiency function

    /** @brief [les.combustion] of an input with one. */
    static ThickenedFlame from_input(const toml::value & table);

    logging::Items summary() const;

    /** @brief Flame sensor: 1 inside the flame (preheat zone included), 0 in fresh and burnt gas. */
    KOKKOS_INLINE_FUNCTION
    double sensor(const double T) const {
        const double c = Kokkos::fmin(1.0, Kokkos::fmax(0.0, (T - T_u) / (T_b - T_u)));
        return Kokkos::fmin(1.0, c * (1.0 - c) / (C_SENSOR * (1.0 - C_SENSOR)));
    }

    /** @brief Thickening of a cell of filter width delta with sensor omega. */
    KOKKOS_INLINE_FUNCTION
    double thickening(const double delta, const double omega) const {
        const double F_max = Kokkos::fmax(1.0, n_res * delta / delta_L);
        return 1.0 + (F_max - 1.0) * omega;
    }

    /**
     * @brief Charlette efficiency (1 + min[F - 1, Gamma u' / s_L])^beta for
     *        the SGS velocity u_prime at the filter width F delta_L, with the
     *        kinematic viscosity nu (their equations for Gamma, f_u, f_Delta
     *        and f_Re with C_k = 1.5, b = 1.4).
     */
    KOKKOS_INLINE_FUNCTION
    double wrinkling(const double F, const double u_prime, const double nu) const {
        if (!efficiency || !(F > 1.0) || !(u_prime > 0.0)) return 1.0;
        const double pi43 = Kokkos::pow(Kokkos::numbers::pi, 4.0 / 3.0);
        const double r = u_prime / s_L;
        const double re = u_prime * F * delta_L / nu;
        const double f_u = 4.0 * Kokkos::sqrt(27.0 * C_K / 110.0) * (18.0 * C_K / 55.0) * r * r;
        const double f_d = Kokkos::sqrt(27.0 * C_K * pi43 / 110.0 * (Kokkos::pow(F, 4.0 / 3.0) - 1.0));
        const double f_re = Kokkos::sqrt(9.0 / 55.0 * Kokkos::exp(-1.5 * C_K * pi43 / re)) * Kokkos::sqrt(re);
        const double a = 0.6 + 0.2 * Kokkos::exp(-0.1 * r) - 0.2 * Kokkos::exp(-0.01 * F);
        const double b = 1.4;
        const double inner = Kokkos::pow(Kokkos::pow(f_u, -a) + Kokkos::pow(f_d, -a), -1.0 / a);
        const double gamma = Kokkos::pow(Kokkos::pow(inner, -b) + Kokkos::pow(f_re, -b), -1.0 / b);
        return Kokkos::pow(1.0 + Kokkos::fmin(F - 1.0, gamma * r), beta);
    }
};

#endif // TFLES_H

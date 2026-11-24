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
#include "input.h"

#include "common.h"
#include "log.h"
#include "report.h"

/**
 * @brief Thickening F, efficiency E and flame sensor Omega of a cell. The
 *        flame is thickened by F where Omega = 1 so that it spans n_res cells:
 *        diffusivities and conductivity times E F, reaction rates times E / F
 *        (laminar speed times E, thickness times F). E is the SGS wrinkling of
 *        Charlette, Meneveau & Veynante (Combust. Flame 131, 2002) at the
 *        filter width F delta_L: their saturated power law F^beta with beta
 *        from the dynamic procedure of Wang, Boileau & Veynante (Combust.
 *        Flame 158, 2011), or their efficiency function of a subgrid velocity
 *        with beta = 0.5, optionally capped. Non-premixed and partially premixed
 *        flames take delta_L and s_L of the premixed flame of each cell's mixture
 *        fraction (docs/design/les.md, section 6.7). Plain aggregate, captured
 *        by value in kernels.
 */
struct ThickenedFlame {
    double delta_L = 0.0;  // laminar thermal thickness (m); REACTION: the table's thinnest
    double s_L = 0.0;      // laminar flame speed (m/s); REACTION: the table's fastest
    double T_u = 0.0, T_b = 0.0;
    double n_res = 5.0;
    enum class Efficiency : uint8_t { NONE, CHARLETTE, DYNAMIC };
    Efficiency efficiency = Efficiency::DYNAMIC;
    double beta = 0.0;  // CHARLETTE: input (default 0.5); DYNAMIC: the initial exponent (then the solver's, on the device)
    bool eddy_viscosity_velocity = false;  // u' = C_u nu_t / Delta instead of Colin et al.'s operator
    double C_u = 28.0;
    double efficiency_max = 0.0;  // E <= efficiency_max where positive

    /**
     * @brief The flame sensor: TEMPERATURE, of the progress variable between T_u and T_b (one fresh-gas state);
     *        REACTION (Legier et al. 2002), of the consumption rate of one species over its largest
     *        in the premixed flame of the local mixture fraction Z, with the s_L and delta_L of that flame, where
     *        the flame is premixed (Takeno's flame index: fuel and oxidizer gradients point the same way);
     *        PROGRESS (gas = "flamelet"), of the table's normalized progress variable Lambda.
     */
    enum class Sensor : uint8_t { TEMPERATURE, REACTION, PROGRESS };
    Sensor sensor_type = Sensor::TEMPERATURE;
    Kokkos::View<double *[4]> table{};   // REACTION: rows (Z, s_L, delta_L, omega_max), Z increasing
    Kokkos::View<double *> z_weights{};  // REACTION: Z = z_offset + sum_k z_weights(k) Y_k (Bilger)
    double z_offset = 0.0;
    uint32_t sensor_species = 0;       // REACTION: the species whose consumption rate the sensor reads
    uint32_t oxidizer_species = 0;     // REACTION: with sensor_species, the flame index's species

    static constexpr double C_SENSOR = 0.05;  // progress variable at which Omega reaches 1
    static constexpr double C_K = 1.5;        // Kolmogorov constant of the efficiency function
    // REACTION: the sensor rises from 0 to 1 between these fractions of the flame's largest consumption rate, is
    // filtered by SENSOR_PASSES passes of the vertex-neighbor average and amplified by SENSOR_GAIN (clipped at 1)
    static constexpr double RATE_ONSET = 0.01, RATE_FULL = 0.05;
    static constexpr uint32_t SENSOR_PASSES = 2;
    static constexpr double SENSOR_GAIN = 2.0;

    /** @brief [les.combustion] of an input with one; flamelet: of a gas = "flamelet" run. */
    static ThickenedFlame from_input(const Input & table, bool flamelet = false);

    report::Summary summary() const;

    /** @brief Flame sensor: 1 inside the flame (preheat zone included), 0 in fresh and burnt gas. */
    KOKKOS_INLINE_FUNCTION
    double sensor(const double T) const { return progress_sensor(progress(T)); }

    /** @brief The progress variable of temperature T, between T_u and T_b. */
    KOKKOS_INLINE_FUNCTION
    double progress(const double T) const { return Kokkos::fmin(1.0, Kokkos::fmax(0.0, (T - T_u) / (T_b - T_u))); }

    /** @brief Flame sensor of a progress variable c in [0, 1]. */
    KOKKOS_INLINE_FUNCTION
    static double progress_sensor(const double c) {
        return Kokkos::fmin(1.0, c * (1.0 - c) / (C_SENSOR * (1.0 - C_SENSOR)));
    }

    /** @brief Thickening of a cell of filter width delta with sensor omega, for a flame of thickness delta_L_. */
    KOKKOS_INLINE_FUNCTION
    double thickening(const double delta, const double omega, const double delta_L_) const {
        const double F_max = Kokkos::fmax(1.0, n_res * delta / delta_L_);
        return 1.0 + (F_max - 1.0) * omega;
    }

    KOKKOS_INLINE_FUNCTION
    double thickening(const double delta, const double omega) const { return thickening(delta, omega, delta_L); }

    /** @brief REACTION: the premixed flame of mixture fraction Z, interpolated in the table; flammable = within it. */
    struct Flame {
        double s_L = 0.0, delta_L = 0.0, omega_max = 0.0;
        bool flammable = false;
    };

    KOKKOS_INLINE_FUNCTION
    Flame flame_of(const double Z) const {
        const uint32_t n = static_cast<uint32_t>(table.extent(0));
        if (!(Z >= table(0, 0) && Z <= table(n - 1, 0))) return Flame{};
        uint32_t lo = 0, hi = n - 1;
        while (hi - lo > 1) {
            const uint32_t mid = lo + (hi - lo) / 2;
            if (table(mid, 0) <= Z) {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        const double t = (Z - table(lo, 0)) / (table(hi, 0) - table(lo, 0));
        const auto at = [&](const int k) { return (1.0 - t) * table(lo, k) + t * table(hi, k); };
        return Flame{at(1), at(2), at(3), true};
    }

    /** @brief REACTION: the unfiltered sensor of a consumption rate in a flame whose largest rate is omega_max. */
    KOKKOS_INLINE_FUNCTION
    static double reaction_sensor(const double rate, const double omega_max) {
        const double s = (rate / omega_max - RATE_ONSET) / (RATE_FULL - RATE_ONSET);
        return s > 0.0 ? (s < 1.0 ? s : 1.0) : 0.0;
    }

    /** @brief The efficiency E clipped at efficiency_max. */
    KOKKOS_INLINE_FUNCTION
    double capped(const double E) const { return efficiency_max > 0.0 && E > efficiency_max ? efficiency_max : E; }

    /**
     * @brief Charlette efficiency (1 + min[F - 1, Gamma u' / s_L])^beta for
     *        the SGS velocity u_prime at the filter width F delta_L, with the
     *        kinematic viscosity nu (their equations for Gamma, f_u, f_Delta
     *        and f_Re with C_k = 1.5, b = 1.4).
     */
    KOKKOS_INLINE_FUNCTION
    double wrinkling(const double F, const double u_prime, const double nu) const {
        return wrinkling(F, u_prime, nu, s_L, delta_L);
    }

    /** @brief The Charlette efficiency of a flame of speed s_L_ and thickness delta_L_. */
    KOKKOS_INLINE_FUNCTION
    double wrinkling(const double F, const double u_prime, const double nu, const double s_L_,
                     const double delta_L_) const {
        if (efficiency != Efficiency::CHARLETTE || !(F > 1.0) || !(u_prime > 0.0)) return 1.0;
        const double pi43 = Kokkos::pow(Kokkos::numbers::pi, 4.0 / 3.0);
        const double r = u_prime / s_L_;
        const double re = u_prime * F * delta_L_ / nu;
        const double f_u = 4.0 * Kokkos::sqrt(27.0 * C_K / 110.0) * (18.0 * C_K / 55.0) * r * r;
        const double f_d = Kokkos::sqrt(27.0 * C_K * pi43 / 110.0 * (Kokkos::pow(F, 4.0 / 3.0) - 1.0));
        const double f_re = Kokkos::sqrt(9.0 / 55.0 * Kokkos::exp(-1.5 * C_K * pi43 / re)) * Kokkos::sqrt(re);
        const double a = 0.6 + 0.2 * Kokkos::exp(-0.1 * r) - 0.2 * Kokkos::exp(-0.01 * F);
        const double b = 1.4;
        const double inner = Kokkos::pow(Kokkos::pow(f_u, -a) + Kokkos::pow(f_d, -a), -1.0 / a);
        const double gamma = Kokkos::pow(Kokkos::pow(inner, -b) + Kokkos::pow(f_re, -b), -1.0 / b);
        return Kokkos::pow(1.0 + Kokkos::fmin(F - 1.0, gamma * r), beta);
    }

    /** @brief Dynamic efficiency F^beta_: Charlette et al.'s saturated form at Delta_e / delta_L = F. */
    KOKKOS_INLINE_FUNCTION
    static double dynamic_wrinkling(const double F, const double beta_) { return F > 1.0 ? Kokkos::pow(F, beta_) : 1.0; }
};

#endif // TFLES_H

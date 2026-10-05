/**
 * @file radiation.h
 * @author Matthew Bonanni (mbonanni001@gmail.com)
 * @brief Planck-mean absorption coefficients of the optically thin radiation
 *        model of the TNF workshop (RADCAL fits; Barlow et al. 2001).
 * @version 0.3
 * @date 2026-10-05
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 *
 */

#ifndef CHEMISTRY_RADIATION_H
#define CHEMISTRY_RADIATION_H

#include <Kokkos_Core.hpp>

#include "mechanism.h"

namespace chemistry {

enum class RadiatingSpecies { H2O = 0, CO2 = 1, CO = 2, CH4 = 3 };
constexpr int N_RADIATING_SPECIES = 4;

/** @brief Planck-mean absorption coefficient, 1 / (m atm), of species s at T (fits valid over 300-2500 K). */
KOKKOS_INLINE_FUNCTION
double planck_mean_absorption(const RadiatingSpecies s, const double T) {
    switch (s) {
        case RadiatingSpecies::H2O: {
            const double z = 1000.0 / T;
            return -0.23093 + z * (-1.12390 + z * (9.41530 + z * (-2.99880 + z * (0.51382 + z * -1.86840e-5))));
        }
        case RadiatingSpecies::CO2: {
            const double z = 1000.0 / T;
            return 18.741 + z * (-121.310 + z * (273.500 + z * (-194.050 + z * (56.310 + z * -5.8169))));
        }
        case RadiatingSpecies::CO:
            if (T <= 750.0) return 4.7869 + T * (-0.06953 + T * (2.95775e-4 + T * (-4.25732e-7 + T * 2.02894e-10)));
            return 10.09 + T * (-0.01183 + T * (4.7753e-6 + T * (-5.87209e-10 + T * -2.5334e-14)));
        case RadiatingSpecies::CH4:
            return 6.6334 + T * (-0.0035686 + T * (1.6682e-8 + T * (2.5611e-10 + T * -2.6558e-14)));
    }
    return 0.0;
}

constexpr double STEFAN_BOLTZMANN = 5.670374419e-8;  // W/(m^2 K^4), CODATA 2018 (Cantera's value)

/**
 * @brief Optically thin radiative loss q = 4 sigma sum_i p_i a_i(T) (T^4 - T_amb^4),
 *        p_i the partial pressures [atm] of the radiating species present.
 */
struct OpticallyThinRadiation {
    Kokkos::Array<int32_t, N_RADIATING_SPECIES> index{-1, -1, -1, -1};  // in the mechanism, -1 if not radiating
    Kokkos::Array<double, N_RADIATING_SPECIES> inv_W{};                 // kmol/kg
    double T_ambient4 = 0.0;

    /** @brief Power lost per unit volume [W/m^3] at T, rho_k(k) the partial densities [kg/m^3]. */
    template <typename F_rho>
    KOKKOS_INLINE_FUNCTION double loss(const double T, const F_rho & rho_k) const {
        const double RT_atm = GAS_CONSTANT * T / ONE_ATM;
        double kappa = 0.0;  // 1/m
        for (int s = 0; s < N_RADIATING_SPECIES; s++) {
            if (index[s] < 0) continue;
            kappa += rho_k(index[s]) * inv_W[s] * RT_atm * planck_mean_absorption(static_cast<RadiatingSpecies>(s), T);
        }
        const double T2 = T * T;
        return 4.0 * STEFAN_BOLTZMANN * kappa * (T2 * T2 - T_ambient4);
    }
};

} // namespace chemistry

#endif // CHEMISTRY_RADIATION_H

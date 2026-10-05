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

} // namespace chemistry

#endif // CHEMISTRY_RADIATION_H

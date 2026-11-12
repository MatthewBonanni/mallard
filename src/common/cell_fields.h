/**
 * @file cell_fields.h
 * @author Matthew Bonanni (mbonanni001@gmail.com)
 * @brief The cell values that expressions of the solution read: W = [rho, u,
 *        p] and, for mixtures, the mass fractions, in every cell.
 * @version 0.1
 * @date 2026-11-11
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 *
 */

#ifndef CELL_FIELDS_H
#define CELL_FIELDS_H

#include <Kokkos_Core.hpp>
#include "common_typedef.h"

namespace region {

/** @brief What expressions of the solution read in each cell: W = [rho, u, p], and for mixtures the mass fractions. */
struct CellFields {
    Kokkos::View<rtype *[N_DIM]> coords;
    Kokkos::View<rtype *[N_CONSERVATIVE]> W;
    Kokkos::View<rtype **, Kokkos::LayoutRight> Y;  // (cell, k), mixtures
    Kokkos::View<double *> inv_W;                   // kmol/kg, mixtures
    double R = 0.0;                                 // single gas: gas constant
    double R_universal = 0.0;                       // mixtures: J/(kmol K)
};

} // namespace region

#endif // CELL_FIELDS_H

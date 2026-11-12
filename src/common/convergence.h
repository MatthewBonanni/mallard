/**
 * @file convergence.h
 * @brief Grid convergence of a quantity from its values on several grids: the observed order of accuracy,
 *        the Richardson-extrapolated value and the grid convergence index (GCI) of Roache, by the procedure
 *        of Celik et al. (2008); and observed orders of errors against an exact solution.
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 */

#ifndef CONVERGENCE_H
#define CONVERGENCE_H

#include <array>
#include <limits>
#include <vector>

namespace convergence {

/** @brief Roache's factors of safety of the GCI: with an order observed on three or more grids, or assumed on two. */
inline constexpr double SAFETY_THREE_GRIDS = 1.25;
inline constexpr double SAFETY_TWO_GRIDS = 3.0;

/** @brief Differences between grids at most this fraction of the largest value are round-off: no order follows. */
inline constexpr double ROUND_OFF = 1e-10;

/** @brief How a quantity changes from the coarsest to the finest grid (Celik's R = e21 / e32). */
enum class Behavior {
    monotonic,      ///< 0 < R < 1: the changes shrink, with one sign
    oscillatory,    ///< R < 0: the changes alternate in sign
    divergent,      ///< R >= 1: the changes grow with refinement
    unchanged,      ///< every change is round-off: the value is grid independent
    indeterminate,  ///< the finest change is round-off, the coarser one not: no order follows
    assumed,        ///< two grids: the order is assumed, not observed
};

/** @brief "monotonic", "oscillatory", "divergent", "unchanged", "indeterminate" or "assumed". */
const char * behavior_name(Behavior b);

constexpr double NaN = std::numeric_limits<double>::quiet_NaN();

/** @brief The estimate of grid 1, the finest of the grids it is computed from. */
struct Estimate {
    Behavior behavior = Behavior::indeterminate;
    double p = NaN;             ///< observed order (assumed with two grids)
    double r21 = NaN, r32 = NaN;  ///< refinement ratios h2 / h1 and h3 / h2 (r32 NaN with two grids)
    double R = NaN;             ///< e21 / e32 (NaN with two grids)
    double extrapolated = NaN;  ///< Richardson-extrapolated value
    double e_a = NaN;           ///< approximate relative error |(phi1 - phi2) / phi1|
    double e_ext = NaN;         ///< extrapolated relative error |(phi_ext - phi1) / phi_ext|
    double gci = NaN;           ///< fine-grid convergence index, relative: safety e_a / (r21^p - 1)
    double safety = NaN;        ///< the factor of safety of the GCI
    bool iterated = true;       ///< whether the fixed-point iteration for p converged (always, for constant r)
};

/**
 * @brief The estimate from three grids, finest first (h[0] < h[1] < h[2]): Celik et al. (2008), eqs. (3)-(7).
 *        The order solves p = |ln|e32 / e21| + q(p)| / ln r21, q(p) = ln((r21^p - s) / (r32^p - s)),
 *        s = sign(e32 / e21), by fixed-point iteration from q = 0. Changes at most round_off of the largest
 *        value are round-off.
 * @throws std::invalid_argument if the sizes are not increasing and positive
 */
Estimate three_grids(const std::array<double, 3> & h, const std::array<double, 3> & phi,
                     double safety = SAFETY_THREE_GRIDS, double round_off = ROUND_OFF);

/**
 * @brief The estimate from two grids, finest first, with the order p assumed (Roache: factor of safety 3).
 * @throws std::invalid_argument if the sizes are not increasing and positive, or p is not positive
 */
Estimate two_grids(const std::array<double, 2> & h, const std::array<double, 2> & phi, double p,
                   double safety = SAFETY_TWO_GRIDS);

/** @brief The observed order of errors e_coarse on a grid of size h_coarse and e_fine on h_fine. */
double pair_order(double h_coarse, double h_fine, double e_coarse, double e_fine);

/** @brief The least-squares slope of ln e against ln h; NaN with fewer than two positive errors. */
double fitted_order(const std::vector<double> & h, const std::vector<double> & e);

} // namespace convergence

#endif // CONVERGENCE_H

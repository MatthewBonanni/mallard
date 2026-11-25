/**
 * @file liquid_curve.h
 * @author Matthew Bonanni (mbonanni001@gmail.com)
 * @brief Tabulated property curves of a liquid: cubic Hermite interpolation on
 *        knots that crowd toward the end of the range, evaluated alike on the
 *        host and in device kernels. Design: docs/design/liquid_library.md.
 * @version 0.1
 * @date 2026-11-25
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 *
 */

#ifndef LIQUID_CURVE_H
#define LIQUID_CURVE_H

#include <cstdint>

#include <Kokkos_Core.hpp>

namespace particles {

/**
 * @brief A function y(s) on [s_min, s_max], tabulated: n knots s_k = s_max - (s_max - s_min) g(1 - k / (n - 1)),
 *        g(u) = u (u + a) / (1 + a), which crowd toward s_max (where a liquid nears its critical point) with a
 *        spacing that stays positive there (a > 0), and at each knot y_k and the slope dy/dx_k, x = k / (n - 1),
 *        stored at offset in a table of values: y on [x_k, x_k+1] is the cubic Hermite polynomial of these.
 */
struct Curve {
    double s_min = 0.0;
    double s_max = 0.0;
    double a = 1.0;
    uint32_t n = 0;       // knots; 0: no curve
    uint32_t offset = 0;  // y_k at table[offset + 2 k], dy/dx_k at table[offset + 2 k + 1]

    /** @brief Whether the curve is tabulated. */
    KOKKOS_INLINE_FUNCTION bool given() const { return n > 1; }

    /** @brief x(s) in [0, 1], with s clamped to [s_min, s_max]. */
    KOKKOS_INLINE_FUNCTION double coordinate(const double s) const {
        const double r = Kokkos::fmin(Kokkos::fmax((s_max - s) / (s_max - s_min), 0.0), 1.0);
        return 1.0 - 2.0 * (1.0 + a) * r / (a + Kokkos::sqrt(a * a + 4.0 * (1.0 + a) * r));
    }

    /** @brief s(x), the inverse of coordinate(). */
    KOKKOS_INLINE_FUNCTION double at(const double x) const {
        const double u = 1.0 - x;
        return s_max - (s_max - s_min) * u * (u + a) / (1.0 + a);
    }

    /** @brief ds/dx at x. */
    KOKKOS_INLINE_FUNCTION double stretch(const double x) const {
        return (s_max - s_min) * (2.0 * (1.0 - x) + a) / (1.0 + a);
    }

    /**
     * @brief y at s clamped to [s_min, s_max], and its slope dy/ds there, from the curve's table.
     */
    KOKKOS_INLINE_FUNCTION double evaluate(const double * table, const double s, double & slope) const {
        const double x = coordinate(s);
        const double scaled = x * static_cast<double>(n - 1);
        const uint32_t k = Kokkos::min(static_cast<uint32_t>(scaled), n - 2);
        const double t = scaled - static_cast<double>(k), h = 1.0 / static_cast<double>(n - 1);
        const double * c = table + offset + 2 * k;
        const double y0 = c[0], d0 = c[1] * h, y1 = c[2], d1 = c[3] * h;
        const double t2 = t * t, t3 = t2 * t;
        const double y = (2.0 * t3 - 3.0 * t2 + 1.0) * y0 + (t3 - 2.0 * t2 + t) * d0 + (3.0 * t2 - 2.0 * t3) * y1 +
                         (t3 - t2) * d1;
        const double dydt = (6.0 * t2 - 6.0 * t) * (y0 - y1) + (3.0 * t2 - 4.0 * t + 1.0) * d0 + (3.0 * t2 - 2.0 * t) * d1;
        slope = dydt / (h * stretch(x));
        return y;
    }

    /** @brief y at s clamped to [s_min, s_max]. */
    KOKKOS_INLINE_FUNCTION double value(const double * table, const double s) const {
        double slope;
        return evaluate(table, s, slope);
    }
};

} // namespace particles

#endif // LIQUID_CURVE_H

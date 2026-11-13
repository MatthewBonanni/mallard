/**
 * @file batch_means.h
 * @author Matthew Bonanni (mbonanni001@gmail.com)
 * @brief Confidence intervals by batch means: the mean and the sum of squared
 *        deviations of batch values (Welford's update), and the half width
 *        t_{n-1} s / sqrt(n) of their mean. Shared by the intervals of
 *        monitored values, of the per-cell [statistics] averages and of
 *        averages over snapshots.
 * @version 0.1
 * @date 2026-11-10
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 *
 */

#ifndef BATCH_MEANS_H
#define BATCH_MEANS_H

#include <cmath>
#include <cstdint>
#include <limits>

#include <Kokkos_Core.hpp>

namespace batch_means {

/**
 * @brief Adds the n-th batch value v (n counted with it) to the mean and the
 *        sum of squared deviations m2 of the first n - 1 (Welford 1962).
 */
template <typename T>
KOKKOS_INLINE_FUNCTION void add(T & mean, T & m2, T n, T v) {
    const T d = v - mean;
    mean += d / n;
    m2 += d * (v - mean);
}

/**
 * @brief Half width of the confidence interval of the mean of n batch values with
 *        sum of squared deviations m2, t the Student's t quantile for n - 1 degrees
 *        of freedom: t sqrt(m2 / (n - 1) / n); NaN for fewer than 2.
 */
inline double half_width(double m2, uint64_t n, double t) {
    if (n < 2) return std::numeric_limits<double>::quiet_NaN();
    return t * std::sqrt(m2 / static_cast<double>(n - 1) / static_cast<double>(n));
}

} // namespace batch_means

#endif // BATCH_MEANS_H

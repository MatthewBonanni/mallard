/**
 * @file exact_sums.h
 * @author Matthew Bonanni (mbonanni001@gmail.com)
 * @brief Global sums of per-cell terms that are the same bit for bit on any
 *        number of ranks, threads or backends.
 * @version 0.1
 * @date 2026-10-29
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 *
 */

#ifndef EXACT_SUMS_H
#define EXACT_SUMS_H

#include <array>
#include <cmath>
#include <cstdint>

#include <Kokkos_Core.hpp>

#include "comm.h"

/**
 * @brief Sums over all ranks of the owned cells' terms, exact in fixed point (each term rounded to 2^-b of
 *        the largest, b leaving room for every cell in 63 bits), so independent of the rank count and order.
 */
template <size_t N>
std::array<double, N> exact_sums(const Kokkos::View<double *[N]> & terms, const uint32_t n_owned, const bool distributed) {
    std::array<double, N> largest = {};
    for (size_t k = 0; k < N; k++) {
        double local = 0.0;
        Kokkos::parallel_reduce(
            "exact_sums_largest", n_owned,
            KOKKOS_LAMBDA(const uint32_t c, double & m) { m = Kokkos::fmax(m, Kokkos::fabs(terms(c, k))); },
            Kokkos::Max<double>(local));
        largest[k] = local;
    }
    if (distributed) largest = comm::allreduce(largest, comm::Op::MAX);
    const uint64_t n_global =
        distributed ? comm::allreduce(static_cast<uint64_t>(n_owned), comm::Op::SUM) : static_cast<uint64_t>(n_owned);
    const int bits = 62 - static_cast<int>(std::ceil(std::log2(static_cast<double>(n_global) + 1.0)));
    std::array<int64_t, N> sums = {};
    std::array<double, N> scale = {};
    for (size_t k = 0; k < N; k++) {
        if (!(largest[k] > 0.0)) continue;
        const double s = std::ldexp(1.0, bits) / largest[k];
        scale[k] = s;
        int64_t local = 0;
        Kokkos::parallel_reduce(
            "exact_sums", n_owned,
            KOKKOS_LAMBDA(const uint32_t c, int64_t & sum) {
                sum += static_cast<int64_t>(Kokkos::round(terms(c, k) * s));
            },
            Kokkos::Sum<int64_t>(local));
        sums[k] = local;
    }
    if (distributed) sums = comm::allreduce(sums, comm::Op::SUM);
    std::array<double, N> result = {};
    for (size_t k = 0; k < N; k++) result[k] = scale[k] > 0.0 ? static_cast<double>(sums[k]) / scale[k] : 0.0;
    return result;
}

#endif // EXACT_SUMS_H

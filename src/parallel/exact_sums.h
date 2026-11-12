/**
 * @file exact_sums.h
 * @author Matthew Bonanni (mbonanni001@gmail.com)
 * @brief Global sums of per-cell terms that are the same bit for bit on any
 *        number of ranks, threads or backends.
 * @version 0.3
 * @date 2026-11-11
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
#include "device_comm.h"

/**
 * @brief 2^b for terms summed over n_global rows, b leaving room for every row in 63 bits.
 */
inline double exact_sums_unit(const uint64_t n_global) {
    return std::ldexp(1.0, 62 - static_cast<int>(std::ceil(std::log2(static_cast<double>(n_global) + 1.0))));
}

/**
 * @brief Sums over all ranks of the owned cells' terms, exact in fixed point (each term rounded to 2^-b of
 *        the largest, see exact_sums_unit), so independent of the rank count and order.
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
    const double unit = exact_sums_unit(n_global);
    std::array<int64_t, N> sums = {};
    std::array<double, N> scale = {};
    for (size_t k = 0; k < N; k++) {
        if (!(largest[k] > 0.0)) continue;
        const double s = unit / largest[k];
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

/**
 * @brief The sums of exact_sums, bit for bit, for a number of columns set at
 *        run time, kept on the device: reduced over the ranks with NCCL,
 *        nothing waits for the host, so a CUDA graph can capture them.
 */
class ExactSums {
    public:
        ExactSums() = default;

        /**
         * @param n_columns Columns of the terms.
         * @param n_rows Rows of the terms on this rank (collective if distributed).
         * @param distributed_ Whether to sum over the ranks.
         * @param use_nccl_ Whether to reduce over the ranks with NCCL (comm::allreduce_device).
         */
        ExactSums(const uint32_t n_columns, const uint32_t n_rows, const bool distributed_, const bool use_nccl_)
            : largest("exact_sums_largest", n_columns), fixed("exact_sums_fixed", n_columns),
              totals("exact_sums", n_columns), distributed(distributed_), use_nccl(use_nccl_),
              unit(exact_sums_unit(distributed_ ? comm::allreduce(static_cast<uint64_t>(n_rows), comm::Op::SUM)
                                               : static_cast<uint64_t>(n_rows))) {}

        /** @brief Sum the columns of terms (row, column) into sums(), enqueued on the default execution space. */
        template <typename Terms>
        void sum(const Terms & terms) const {
            using DeviceMemory = Kokkos::DefaultExecutionSpace::memory_space;
            const uint32_t n_rows = static_cast<uint32_t>(terms.extent(0));
            const uint32_t n_columns = static_cast<uint32_t>(totals.extent(0));
            const Kokkos::View<double *> m = largest;
            for (uint32_t k = 0; k < n_columns; k++) {
                Kokkos::parallel_reduce("exact_sums_largest", n_rows, KOKKOS_LAMBDA(const uint32_t c, double & x) {
                    x = Kokkos::fmax(x, Kokkos::fabs(terms(c, k)));
                }, Kokkos::Max<double, DeviceMemory>(Kokkos::View<double, DeviceMemory>(m.data() + k)));
            }
            if (distributed) comm::allreduce_device(m, comm::Op::MAX, use_nccl);
            const Kokkos::View<int64_t *> f = fixed;
            const double unit_ = unit;
            for (uint32_t k = 0; k < n_columns; k++) {
                Kokkos::parallel_reduce("exact_sums", n_rows, KOKKOS_LAMBDA(const uint32_t c, int64_t & sum) {
                    if (m(k) > 0.0) sum += static_cast<int64_t>(Kokkos::round(terms(c, k) * (unit_ / m(k))));
                }, Kokkos::Sum<int64_t, DeviceMemory>(Kokkos::View<int64_t, DeviceMemory>(f.data() + k)));
            }
            if (distributed) comm::allreduce_device(f, comm::Op::SUM, use_nccl);
            const Kokkos::View<double *> out = totals;
            Kokkos::parallel_for("exact_sums_totals", n_columns, KOKKOS_LAMBDA(const uint32_t k) {
                out(k) = m(k) > 0.0 ? static_cast<double>(f(k)) / (unit_ / m(k)) : 0.0;
            });
        }

        /** @brief The sums of the last sum(), on the device. */
        const Kokkos::View<double *> & sums() const { return totals; }

    private:
        Kokkos::View<double *> largest;
        Kokkos::View<int64_t *> fixed;
        Kokkos::View<double *> totals;
        bool distributed = false, use_nccl = false;
        double unit = 0.0;
};

#endif // EXACT_SUMS_H

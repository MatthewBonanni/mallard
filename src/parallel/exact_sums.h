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
#include "gated.h"

/**
 * @brief 2^b for terms summed over n_global rows, b leaving room for every row in 63 bits: b = 62 - ceil(log2(n_global
 *        + 1)), the bits of n_global.
 */
KOKKOS_INLINE_FUNCTION double exact_sums_unit(const uint64_t n_global) {
    int bits = 0;
    while (bits < 64 && (n_global >> bits) != 0) bits++;
    return Kokkos::bit_cast<double>(static_cast<uint64_t>(1023 + 62 - bits) << 52);
}

/** @brief A term's magnitude for the largest of its column: a non-finite term makes the column's sum NaN. */
KOKKOS_INLINE_FUNCTION double exact_sums_magnitude(const double term) {
    return Kokkos::isfinite(term) ? Kokkos::fabs(term) : Kokkos::Experimental::infinity_v<double>;
}

/** @brief A term in fixed point, in units of largest / unit. */
KOKKOS_INLINE_FUNCTION int64_t exact_sums_fixed(const double term, const double unit, const double largest) {
    return largest > 0.0 && Kokkos::isfinite(largest) ? static_cast<int64_t>(Kokkos::round(term * (unit / largest))) : 0;
}

/** @brief The sum of a column's fixed-point terms. */
KOKKOS_INLINE_FUNCTION double exact_sums_total(const int64_t fixed, const double unit, const double largest) {
    if (!Kokkos::isfinite(largest)) return Kokkos::Experimental::quiet_NaN_v<double>;
    return largest > 0.0 ? static_cast<double>(fixed) / (unit / largest) : 0.0;
}

/**
 * @brief On the host, the sum of n_rows terms (a column of stride `stride`) as ExactSums sums a column of n_rows rows
 *        over every rank, bit for bit.
 */
inline double exact_column_sum(const double * terms, const uint64_t n_rows, const uint64_t stride = 1) {
    double largest = 0.0;
    for (uint64_t i = 0; i < n_rows; i++) largest = Kokkos::fmax(largest, exact_sums_magnitude(terms[i * stride]));
    const double unit = exact_sums_unit(n_rows);
    int64_t fixed = 0;
    for (uint64_t i = 0; i < n_rows; i++) fixed += exact_sums_fixed(terms[i * stride], unit, largest);
    return exact_sums_total(fixed, unit, largest);
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
        gated::parallel_reduce(
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
        gated::parallel_reduce(
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
              unit("exact_sums_unit"), counted_unit("exact_sums_counted_unit"), rows("exact_sums_rows", 1) {
            const double u = exact_sums_unit(distributed_ ? comm::allreduce(static_cast<uint64_t>(n_rows), comm::Op::SUM)
                                                         : static_cast<uint64_t>(n_rows));
            Kokkos::deep_copy(unit, u);
        }

        /** @brief Sum the columns of terms (row, column) into sums(), enqueued on the default execution space. */
        template <typename Terms>
        void sum(const Terms & terms) const {
            sum_rows(terms, Kokkos::View<const uint32_t>(), unit);
        }

        /**
         * @brief As sum, of the first count() rows of terms on each rank, a number known only on the device: the
         *        unit is that of their total over the ranks, as exact_sums of those rows alone has it.
         */
        template <typename Terms>
        void sum(const Terms & terms, const Kokkos::View<const uint32_t> & count) const {
            const Kokkos::View<int64_t *> n = rows;
            gated::parallel_for("exact_sums_rows", 1, KOKKOS_LAMBDA(const int) { n(0) = count(); });
            if (distributed) comm::allreduce_device(n, comm::Op::SUM, use_nccl);
            const Kokkos::View<double> u = counted_unit;
            gated::parallel_for("exact_sums_unit", 1, KOKKOS_LAMBDA(const int) {
                u() = exact_sums_unit(static_cast<uint64_t>(n(0)));
            });
            sum_rows(terms, count, counted_unit);
        }

        /** @brief The sums of the rows of terms (the first count() if counted), in fixed point of unit_() (public: nvcc
         *         rejects device lambdas in private members). */
        template <typename Terms>
        void sum_rows(const Terms & terms, const Kokkos::View<const uint32_t> & count,
                      const Kokkos::View<const double> & unit_) const {
            using DeviceMemory = Kokkos::DefaultExecutionSpace::memory_space;
            const uint32_t n_rows = static_cast<uint32_t>(terms.extent(0));
            const uint32_t n_columns = static_cast<uint32_t>(totals.extent(0));
            const bool counted = count.is_allocated();
            const Kokkos::View<double *> m = largest;
            for (uint32_t k = 0; k < n_columns; k++) {
                gated::parallel_reduce("exact_sums_largest", n_rows, KOKKOS_LAMBDA(const uint32_t c, double & x) {
                    if (!counted || c < count()) x = Kokkos::fmax(x, exact_sums_magnitude(terms(c, k)));
                }, Kokkos::Max<double, DeviceMemory>(Kokkos::View<double, DeviceMemory>(m.data() + k)));
            }
            if (distributed) comm::allreduce_device(m, comm::Op::MAX, use_nccl);
            const Kokkos::View<int64_t *> f = fixed;
            for (uint32_t k = 0; k < n_columns; k++) {
                gated::parallel_reduce("exact_sums", n_rows, KOKKOS_LAMBDA(const uint32_t c, int64_t & sum) {
                    if (!counted || c < count()) sum += exact_sums_fixed(terms(c, k), unit_(), m(k));
                }, Kokkos::Sum<int64_t, DeviceMemory>(Kokkos::View<int64_t, DeviceMemory>(f.data() + k)));
            }
            if (distributed) comm::allreduce_device(f, comm::Op::SUM, use_nccl);
            const Kokkos::View<double *> out = totals;
            gated::parallel_for("exact_sums_totals", n_columns, KOKKOS_LAMBDA(const uint32_t k) {
                out(k) = exact_sums_total(f(k), unit_(), m(k));
            });
        }

        /** @brief The sums of the last sum(), on the device. */
        const Kokkos::View<double *> & sums() const { return totals; }

    private:
        Kokkos::View<double *> largest;
        Kokkos::View<int64_t *> fixed;
        Kokkos::View<double *> totals;
        bool distributed = false, use_nccl = false;
        Kokkos::View<double> unit, counted_unit;
        Kokkos::View<int64_t *> rows;

};

#endif // EXACT_SUMS_H

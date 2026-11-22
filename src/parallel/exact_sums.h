/**
 * @file exact_sums.h
 * @author Matthew Bonanni (mbonanni001@gmail.com)
 * @brief Global sums of per-cell terms that are the same bit for bit on any
 *        number of ranks, threads or backends.
 * @version 0.4
 * @date 2026-11-22
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 *
 */

#ifndef EXACT_SUMS_H
#define EXACT_SUMS_H

#include <array>
#include <cmath>
#include <cstdint>
#include <stdexcept>

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

/** @brief Integers of 32 bits of an exact sum, from 2^-1074 up: room for 2^31 terms below 2^1024 each. */
constexpr uint32_t EXACT_SUMS_LIMBS = 67;

/** @brief Integers an exact sum keeps: its limbs, then its count of terms that are not finite. */
constexpr uint32_t EXACT_SUMS_WIDTH = EXACT_SUMS_LIMBS + 1;

/**
 * @brief Adds term exactly to the sum acc (EXACT_SUMS_WIDTH integers): its 53 bits to the three limbs they cover,
 *        each by less than 2^32, so 2^31 terms fit in 63 bits.
 */
KOKKOS_INLINE_FUNCTION void exact_sums_add(int64_t * acc, const double term) {
    const uint64_t bits = Kokkos::bit_cast<uint64_t>(term);
    const uint32_t biased = static_cast<uint32_t>((bits >> 52) & 0x7ff);
    if (biased == 0x7ff) {
        acc[EXACT_SUMS_LIMBS]++;
        return;
    }
    const uint64_t m = (bits & ((uint64_t(1) << 52) - 1)) | (biased ? uint64_t(1) << 52 : 0);
    const uint32_t p = biased ? biased - 1 : 0;  // of m's lowest bit above 2^-1074
    const uint32_t j = p / 32, r = p % 32;
    const uint64_t a = (m & 0xffffffff) << r, b = (m >> 32) << r, t = (a >> 32) + (b & 0xffffffff);
    const int64_t c0 = static_cast<int64_t>(a & 0xffffffff), c1 = static_cast<int64_t>(t & 0xffffffff),
                  c2 = static_cast<int64_t>((b >> 32) + (t >> 32));
    if (bits >> 63) {
        acc[j] -= c0;
        acc[j + 1] -= c1;
        acc[j + 2] -= c2;
    } else {
        acc[j] += c0;
        acc[j + 1] += c1;
        acc[j + 2] += c2;
    }
}

/** @brief The same sum acc with every limb but the last in [0, 2^32), so that sums of 2^31 such sums fit too. */
KOKKOS_INLINE_FUNCTION void exact_sums_carry(int64_t * acc) {
    int64_t carry = 0;
    for (uint32_t i = 0; i + 1 < EXACT_SUMS_LIMBS; i++) {
        const int64_t v = acc[i] + carry;
        carry = v >> 32;
        acc[i] = v & 0xffffffff;
    }
    acc[EXACT_SUMS_LIMBS - 1] += carry;
}

/** @brief The sum acc rounded to the nearest double (ties to even); NaN if a term was not finite. */
KOKKOS_INLINE_FUNCTION double exact_sums_total(const int64_t * acc) {
    if (acc[EXACT_SUMS_LIMBS] != 0) return Kokkos::Experimental::quiet_NaN_v<double>;
    int64_t d[EXACT_SUMS_LIMBS];
    for (uint32_t i = 0; i < EXACT_SUMS_LIMBS; i++) d[i] = acc[i];
    exact_sums_carry(d);
    const bool negative = d[EXACT_SUMS_LIMBS - 1] < 0;
    if (negative) {
        for (uint32_t i = 0; i < EXACT_SUMS_LIMBS; i++) d[i] = -d[i];
        exact_sums_carry(d);
    }
    int J = static_cast<int>(EXACT_SUMS_LIMBS) - 1;
    while (J >= 0 && d[J] == 0) J--;
    if (J < 0) return 0.0;
    const auto digit = [&](const int i) { return i >= 0 ? static_cast<uint64_t>(d[i]) : uint64_t(0); };
    const uint64_t top = digit(J), next = digit(J - 1), low = digit(J - 2);
    const int lz = Kokkos::countl_zero(static_cast<uint32_t>(top));
    const int h = 32 * J + 31 - lz;  // the highest bit, above 2^-1074
    uint64_t bits = 0;
    if (h < 53) {
        bits = J == 1 ? (top << 32 | next) : top;
    } else {
        const uint64_t v = top << (32 + lz) | next << lz | (lz ? low >> (32 - lz) : 0);
        bool sticky = (lz ? (low << lz) & 0xffffffff : low) != 0;
        for (int i = 0; i < J - 2 && !sticky; i++) sticky = d[i] != 0;
        const uint64_t mantissa = v >> 11, rest = v & 0x7ff;
        const bool up = rest > 0x400 || (rest == 0x400 && (sticky || (mantissa & 1)));
        bits = Kokkos::min((static_cast<uint64_t>(h - 52) << 52) + mantissa + (up ? 1 : 0), uint64_t(0x7ff) << 52);
    }
    return Kokkos::bit_cast<double>(bits | (negative ? uint64_t(1) << 63 : 0));
}

/**
 * @brief On the host, the sum of n_rows terms (a column of stride `stride`) as ExactSums sums a column of n_rows rows
 *        over every rank, bit for bit.
 */
inline double exact_column_sum(const double * terms, const uint64_t n_rows, const uint64_t stride = 1) {
    int64_t acc[EXACT_SUMS_WIDTH] = {};
    for (uint64_t i = 0; i < n_rows; i++) exact_sums_add(acc, terms[i * stride]);
    return exact_sums_total(acc);
}

/** @brief A reduction of column k of terms (its first count() rows if counted) into an exact sum. */
template <typename Terms>
struct ExactSumsColumn {
    using value_type = int64_t[];
    uint32_t value_count = EXACT_SUMS_WIDTH;
    Terms terms;
    Kokkos::View<const uint32_t> count = {};
    uint32_t k = 0;
    bool counted = false;

    KOKKOS_FUNCTION void operator()(const uint32_t c, int64_t * acc) const {
        if (!counted || c < count()) exact_sums_add(acc, static_cast<double>(terms(c, k)));
    }
};

/**
 * @brief Sums over all ranks of the owned cells' terms, each the exact sum rounded once, so independent of the rank
 *        count and order.
 */
template <size_t N>
std::array<double, N> exact_sums(const Kokkos::View<double *[N]> & terms, const uint32_t n_owned, const bool distributed) {
    using Limbs = Kokkos::View<int64_t *, Kokkos::HostSpace, Kokkos::MemoryUnmanaged>;
    std::array<int64_t, N * EXACT_SUMS_WIDTH> acc = {};
    for (uint32_t k = 0; k < N; k++) {
        gated::parallel_reduce("exact_sums", n_owned, ExactSumsColumn<Kokkos::View<double *[N]>>{.terms = terms, .k = k},
                               Limbs(acc.data() + k * EXACT_SUMS_WIDTH, EXACT_SUMS_WIDTH));
        if (distributed) exact_sums_carry(acc.data() + k * EXACT_SUMS_WIDTH);
    }
    if (distributed) acc = comm::allreduce(acc, comm::Op::SUM);
    std::array<double, N> result = {};
    for (size_t k = 0; k < N; k++) result[k] = exact_sums_total(acc.data() + k * EXACT_SUMS_WIDTH);
    return result;
}

/**
 * @brief The sums of exact_sums, bit for bit, for a number of columns set at
 *        run time, kept on the device: reduced over the ranks in one all-reduce
 *        with NCCL, nothing waits for the host, so a CUDA graph can capture them.
 */
class ExactSums {
    public:
        ExactSums() = default;

        /**
         * @param n_columns Columns of the terms.
         * @param n_rows Rows of the terms on this rank (fewer than 2^31).
         * @param distributed_ Whether to sum over the ranks.
         * @param use_nccl_ Whether to reduce over the ranks with NCCL (comm::allreduce_device).
         */
        ExactSums(const uint32_t n_columns, const uint32_t n_rows, const bool distributed_, const bool use_nccl_)
            : limbs("exact_sums_limbs", size_t(n_columns) * EXACT_SUMS_WIDTH), totals("exact_sums", n_columns),
              distributed(distributed_), use_nccl(use_nccl_) {
            if (n_rows >= uint32_t(1) << 31) throw std::invalid_argument("ExactSums: 2^31 rows or more on a rank");
        }

        /** @brief Sum the columns of terms (row, column) into sums(), enqueued on the default execution space. */
        template <typename Terms>
        void sum(const Terms & terms) const {
            sum_rows(terms, Kokkos::View<const uint32_t>());
        }

        /** @brief As sum, of the first count() rows of terms on each rank, a number known only on the device. */
        template <typename Terms>
        void sum(const Terms & terms, const Kokkos::View<const uint32_t> & count) const {
            sum_rows(terms, count);
        }

        /** @brief The sums of the rows of terms (the first count() if counted) (public: nvcc rejects device lambdas in
         *         private members). */
        template <typename Terms>
        void sum_rows(const Terms & terms, const Kokkos::View<const uint32_t> & count) const {
            const uint32_t n_rows = static_cast<uint32_t>(terms.extent(0));
            const uint32_t n_columns = static_cast<uint32_t>(totals.extent(0));
            const Kokkos::View<int64_t *> acc = limbs;
            for (uint32_t k = 0; k < n_columns; k++) {
                gated::parallel_reduce("exact_sums", n_rows,
                                       ExactSumsColumn<Terms>{.terms = terms, .count = count, .k = k,
                                                              .counted = count.is_allocated()},
                                       Kokkos::View<int64_t *>(acc.data() + size_t(k) * EXACT_SUMS_WIDTH, EXACT_SUMS_WIDTH));
            }
            if (distributed) {
                gated::parallel_for("exact_sums_carry", n_columns, KOKKOS_LAMBDA(const uint32_t k) {
                    exact_sums_carry(acc.data() + size_t(k) * EXACT_SUMS_WIDTH);
                });
                comm::allreduce_device(acc, comm::Op::SUM, use_nccl);
            }
            const Kokkos::View<double *> out = totals;
            gated::parallel_for("exact_sums_totals", n_columns, KOKKOS_LAMBDA(const uint32_t k) {
                out(k) = exact_sums_total(acc.data() + size_t(k) * EXACT_SUMS_WIDTH);
            });
        }

        /** @brief The sums of the last sum(), on the device. */
        const Kokkos::View<double *> & sums() const { return totals; }

    private:
        Kokkos::View<int64_t *> limbs;  // EXACT_SUMS_WIDTH a column
        Kokkos::View<double *> totals;
        bool distributed = false, use_nccl = false;
};

#endif // EXACT_SUMS_H

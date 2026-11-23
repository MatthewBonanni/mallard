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
 * @brief Splits a finite term into the three limbs its 53 bits cover, from limb j up, each part below 2^32 in
 *        magnitude (so 2^31 terms fit in 63 bits); false for a term that is not finite.
 */
KOKKOS_INLINE_FUNCTION bool exact_sums_split(const double term, uint32_t & j, int64_t (&parts)[3]) {
    const uint64_t bits = Kokkos::bit_cast<uint64_t>(term);
    const uint32_t biased = static_cast<uint32_t>((bits >> 52) & 0x7ff);
    if (biased == 0x7ff) return false;
    const uint64_t m = (bits & ((uint64_t(1) << 52) - 1)) | (biased ? uint64_t(1) << 52 : 0);
    const uint32_t p = biased ? biased - 1 : 0;  // of m's lowest bit above 2^-1074
    const uint32_t r = p % 32;
    j = p / 32;
    const uint64_t a = (m & 0xffffffff) << r, b = (m >> 32) << r, t = (a >> 32) + (b & 0xffffffff);
    const int64_t sign = (bits >> 63) ? -1 : 1;
    parts[0] = sign * static_cast<int64_t>(a & 0xffffffff);
    parts[1] = sign * static_cast<int64_t>(t & 0xffffffff);
    parts[2] = sign * static_cast<int64_t>((b >> 32) + (t >> 32));
    return true;
}

/** @brief Adds term exactly to the sum acc (EXACT_SUMS_WIDTH integers). */
KOKKOS_INLINE_FUNCTION void exact_sums_add(int64_t * acc, const double term) {
    uint32_t j = 0;
    int64_t parts[3];
    if (!exact_sums_split(term, j, parts)) {
        acc[EXACT_SUMS_LIMBS]++;
        return;
    }
    for (uint32_t x = 0; x < 3; x++) acc[j + x] += parts[x];
}

/** @brief Limbs a column's rows add to in registers, from one below its largest term's lowest limb up. */
constexpr uint32_t EXACT_SUMS_WINDOW = 4;

/** @brief The lowest limb of the window of a column whose largest magnitude is largest. */
KOKKOS_INLINE_FUNCTION uint32_t exact_sums_window_base(const double largest) {
    if (!(largest > 0.0) || !Kokkos::isfinite(largest)) return 0;
    const uint32_t biased = static_cast<uint32_t>(Kokkos::bit_cast<uint64_t>(largest) >> 52);
    const uint32_t top = (biased ? biased - 1 : 0) / 32 + 3;  // past the largest term's highest limb
    return top < EXACT_SUMS_WINDOW ? 0 : top - EXACT_SUMS_WINDOW;
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

/** @brief The largest magnitudes of C columns. */
template <uint32_t C>
struct ExactSumsMaxima {
    double m[C];
};

/** @brief A reduction of the largest magnitudes of columns k0 to k0 + C of terms (their first count() rows if counted). */
template <typename Terms, uint32_t C>
struct ExactSumsLargest {
    using value_type = ExactSumsMaxima<C>;
    Terms terms;
    Kokkos::View<const uint32_t> count = {};
    uint32_t k0 = 0;
    bool counted = false;

    KOKKOS_FUNCTION void operator()(const uint32_t c, value_type & v) const {
        if (counted && c >= count()) return;
        for (uint32_t q = 0; q < C; q++) v.m[q] = Kokkos::fmax(v.m[q], Kokkos::fabs(static_cast<double>(terms(c, k0 + q))));
    }
    KOKKOS_FUNCTION void init(value_type & v) const {
        for (uint32_t q = 0; q < C; q++) v.m[q] = 0.0;
    }
    KOKKOS_FUNCTION void join(value_type & dst, const value_type & src) const {
        for (uint32_t q = 0; q < C; q++) dst.m[q] = Kokkos::fmax(dst.m[q], src.m[q]);
    }
};

/** @brief Terms that are all zero, of n rows. */
struct ExactSumsZeros {
    uint32_t n = 0;
    KOKKOS_INLINE_FUNCTION double operator()(uint32_t, uint32_t) const { return 0.0; }
    uint32_t extent(int) const { return n; }
};

/** @brief Integers of a column's window: its limbs, then its count of terms that are not finite. */
constexpr uint32_t EXACT_SUMS_WINDOW_WIDTH = EXACT_SUMS_WINDOW + 1;

/** @brief The windows of C columns. */
template <uint32_t C>
struct ExactSumsWindows {
    int64_t limbs[C][EXACT_SUMS_WINDOW_WIDTH];
};

/**
 * @brief A reduction of columns k0 to k0 + C of terms (their first count() rows if counted) into the windows of
 *        limbs below their largest magnitudes largest(k), in registers; parts outside them (of terms 2^-32 or less
 *        of the largest) go to the columns' limbs acc with atomics.
 */
template <typename Terms, uint32_t C>
struct ExactSumsWindowed {
    using value_type = ExactSumsWindows<C>;
    Terms terms;
    Kokkos::View<const uint32_t> count = {};
    Kokkos::View<const double *> largest = {};
    Kokkos::View<int64_t *> acc = {};  // EXACT_SUMS_WIDTH a column
    uint32_t k0 = 0;
    bool counted = false;

    KOKKOS_FUNCTION void operator()(const uint32_t c, value_type & v) const {
        if (counted && c >= count()) return;
        for (uint32_t q = 0; q < C; q++) {
            const uint32_t k = k0 + q;
            uint32_t j = 0;
            int64_t parts[3];
            if (!exact_sums_split(static_cast<double>(terms(c, k)), j, parts)) {
                v.limbs[q][EXACT_SUMS_WINDOW]++;
                continue;
            }
            const int32_t base = static_cast<int32_t>(exact_sums_window_base(largest(k)));
            for (uint32_t x = 0; x < 3; x++) {
                const int32_t i = static_cast<int32_t>(j + x) - base;
                if (i >= 0 && i < static_cast<int32_t>(EXACT_SUMS_WINDOW)) {
                    for (int32_t l = 0; l < static_cast<int32_t>(EXACT_SUMS_WINDOW); l++) v.limbs[q][l] += i == l ? parts[x] : 0;
                } else if (parts[x] != 0) {
                    Kokkos::atomic_add(&acc(size_t(k) * EXACT_SUMS_WIDTH + j + x), parts[x]);
                }
            }
        }
    }
    KOKKOS_FUNCTION void init(value_type & v) const {
        for (uint32_t q = 0; q < C; q++) {
            for (uint32_t l = 0; l < EXACT_SUMS_WINDOW_WIDTH; l++) v.limbs[q][l] = 0;
        }
    }
    KOKKOS_FUNCTION void join(value_type & dst, const value_type & src) const {
        for (uint32_t q = 0; q < C; q++) {
            for (uint32_t l = 0; l < EXACT_SUMS_WINDOW_WIDTH; l++) dst.limbs[q][l] += src.limbs[q][l];
        }
    }
};

/** @brief Adds a column's window (EXACT_SUMS_WINDOW_WIDTH integers, of largest magnitude largest) to its limbs acc. */
KOKKOS_INLINE_FUNCTION void exact_sums_merge(int64_t * acc, const int64_t * window, const double largest) {
    const uint32_t base = exact_sums_window_base(largest);
    for (uint32_t q = 0; q < EXACT_SUMS_WINDOW; q++) acc[base + q] += window[q];
    acc[EXACT_SUMS_LIMBS] += window[EXACT_SUMS_WINDOW];
}

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
            : largest("exact_sums_largest", n_columns),
              windows("exact_sums_windows", size_t(n_columns) * EXACT_SUMS_WINDOW_WIDTH),
              limbs("exact_sums_limbs", size_t(n_columns) * EXACT_SUMS_WIDTH), totals("exact_sums", n_columns),
              distributed(distributed_), use_nccl(use_nccl_) {
            if (n_rows >= uint32_t(1) << 31) throw std::invalid_argument("ExactSums: 2^31 rows or more on a rank");
            reserve_reduction_scratch(n_columns, n_rows);
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

        /**
         * @brief The sums of the rows of terms (the first count() if counted): each column's largest magnitude
         *        places its window, its rows add to the window and their few parts below it to its limbs, which
         *        sums() leaves zero for the next call (public: nvcc rejects device lambdas in private members).
         */
        template <typename Terms>
        void sum_rows(const Terms & terms, const Kokkos::View<const uint32_t> & count) const {
            const uint32_t n_columns = static_cast<uint32_t>(totals.extent(0));
            const Kokkos::View<double *> m = largest;
            const Kokkos::View<int64_t *> w = windows, acc = limbs;
            for (uint32_t k0 = 0; k0 < n_columns;) {
                const uint32_t left = n_columns - k0;
                if (left >= 4) {
                    sum_columns<4>(terms, count, k0);
                    k0 += 4;
                } else if (left >= 2) {
                    sum_columns<2>(terms, count, k0);
                    k0 += 2;
                } else {
                    sum_columns<1>(terms, count, k0);
                    k0 += 1;
                }
            }
            if (distributed) {
                gated::parallel_for("exact_sums_carry", n_columns, KOKKOS_LAMBDA(const uint32_t k) {
                    exact_sums_merge(acc.data() + size_t(k) * EXACT_SUMS_WIDTH, w.data() + size_t(k) * EXACT_SUMS_WINDOW_WIDTH, m(k));
                    exact_sums_carry(acc.data() + size_t(k) * EXACT_SUMS_WIDTH);
                });
                comm::allreduce_device(acc, comm::Op::SUM, use_nccl);
            }
            const Kokkos::View<double *> out = totals;
            const bool merged = distributed;
            gated::parallel_for("exact_sums_totals", n_columns, KOKKOS_LAMBDA(const uint32_t k) {
                int64_t * a = acc.data() + size_t(k) * EXACT_SUMS_WIDTH;
                if (!merged) exact_sums_merge(a, w.data() + size_t(k) * EXACT_SUMS_WINDOW_WIDTH, m(k));
                out(k) = exact_sums_total(a);
                for (uint32_t i = 0; i < EXACT_SUMS_WIDTH; i++) a[i] = 0;
            });
        }

        /** @brief Columns k0 to k0 + C of sum_rows: their largest magnitudes, then their windows, each in one kernel. */
        template <uint32_t C, typename Terms>
        void sum_columns(const Terms & terms, const Kokkos::View<const uint32_t> & count, const uint32_t k0) const {
            using DeviceMemory = Kokkos::DefaultExecutionSpace::memory_space;
            const uint32_t n_rows = static_cast<uint32_t>(terms.extent(0));
            const bool counted = count.is_allocated();
            gated::parallel_reduce("exact_sums_largest", n_rows,
                                   ExactSumsLargest<Terms, C>{.terms = terms, .count = count, .k0 = k0, .counted = counted},
                                   Kokkos::View<ExactSumsMaxima<C>, DeviceMemory>(
                                       reinterpret_cast<ExactSumsMaxima<C> *>(largest.data() + k0)));
            gated::parallel_reduce("exact_sums", n_rows,
                                   ExactSumsWindowed<Terms, C>{.terms = terms, .count = count, .largest = largest,
                                                               .acc = limbs, .k0 = k0, .counted = counted},
                                   Kokkos::View<ExactSumsWindows<C>, DeviceMemory>(reinterpret_cast<ExactSumsWindows<C> *>(
                                       windows.data() + size_t(k0) * EXACT_SUMS_WINDOW_WIDTH)));
        }

        /** @brief The sums of the last sum(), on the device. */
        const Kokkos::View<double *> & sums() const { return totals; }

    private:
        /**
         * @brief Sums zeros as the widest columns of sum_rows do, on enough rows for the most blocks, so that
         *        Kokkos grows its reduction scratch for their values here: grown in a step graph's capture, it is an
         *        allocation there, and graphs captured before it keep the scratch it frees.
         */
        void reserve_reduction_scratch(const uint32_t n_columns, const uint32_t n_rows) const {
            const uint32_t n = Kokkos::max(n_rows, static_cast<uint32_t>(Kokkos::DefaultExecutionSpace().concurrency()));
            const ExactSumsZeros zeros{n};
            if (n_columns >= 4) {
                sum_columns<4>(zeros, Kokkos::View<const uint32_t>(), 0);
            } else if (n_columns >= 2) {
                sum_columns<2>(zeros, Kokkos::View<const uint32_t>(), 0);
            } else if (n_columns == 1) {
                sum_columns<1>(zeros, Kokkos::View<const uint32_t>(), 0);
            }
        }

        Kokkos::View<double *> largest;
        Kokkos::View<int64_t *> windows;  // EXACT_SUMS_WINDOW_WIDTH a column
        Kokkos::View<int64_t *> limbs;  // EXACT_SUMS_WIDTH a column, zero between calls
        Kokkos::View<double *> totals;
        bool distributed = false, use_nccl = false;
};

#endif // EXACT_SUMS_H

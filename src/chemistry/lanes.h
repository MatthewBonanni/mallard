/**
 * @file lanes.h
 * @author Matthew Bonanni (mbonanni001@gmail.com)
 * @brief Execution of one cell's chemistry by one thread or by the vector
 *        lanes of a team: loops over species, reactions and matrix rows, and
 *        reductions in a fixed order.
 * @version 0.3
 * @date 2026-10-03
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 *
 */

#ifndef CHEMISTRY_LANES_H
#define CHEMISTRY_LANES_H

#include <cstdint>

#include <Kokkos_Core.hpp>

namespace chemistry {

/** @brief Index of entry (r, c) of an n x n matrix stored by rows, or by columns. */
template <bool ColumnMajor>
KOKKOS_INLINE_FUNCTION constexpr uint32_t dense_index(const uint32_t n, const uint32_t r, const uint32_t c) {
    return ColumnMajor ? c * n + r : r * n + c;
}

/**
 * @brief One thread does all the work of a cell.
 *
 * The interface every cell kernel uses: for_each(n, f) calls f(i) for
 * 0 <= i < n, possibly concurrently; sync() makes the writes of a for_each
 * visible to all lanes; single(f) runs f once; sum and argmax_abs reduce in
 * an order that depends only on the lane count. column_major selects the
 * layout of dense matrices that suits the lanes' access.
 */
struct SerialLanes {
    static constexpr uint32_t lanes = 1;
    static constexpr bool column_major = false;  // dense matrices: rows contiguous for one thread's inner loops
    static constexpr bool parallel = false;      // loops that balance work across lanes are not worth it

    template <typename F>
    KOKKOS_INLINE_FUNCTION void for_each(const uint32_t n, const F & f) const {
        for (uint32_t i = 0; i < n; i++) f(i);
    }

    KOKKOS_INLINE_FUNCTION void sync() const {}

    template <typename F>
    KOKKOS_INLINE_FUNCTION void single(const F & f) const {
        f();
    }

    /** @brief sum_i f(i), in order. */
    template <typename F>
    KOKKOS_INLINE_FUNCTION double sum(const uint32_t n, const F & f) const {
        double s = 0.0;
        for (uint32_t i = 0; i < n; i++) s += f(i);
        return s;
    }

    /** @brief The largest |f(i)| for first <= i < n and the first i that has it. */
    template <typename F>
    KOKKOS_INLINE_FUNCTION double argmax_abs(const uint32_t first, const uint32_t n, const F & f,
                                             uint32_t & index) const {
        index = first;
        double largest = Kokkos::fabs(f(first));
        for (uint32_t i = first + 1; i < n; i++) {
            const double v = Kokkos::fabs(f(i));
            if (v > largest) {
                largest = v;
                index = i;
            }
        }
        return largest;
    }
};

/**
 * @brief All threads and vector lanes of a team (TeamPolicy(cells, T, V),
 *        T * V lanes) share the work of a cell. Scalar code runs on every
 *        lane with the same values; reductions accumulate each lane's strided
 *        share in index order and then the shares in lane order, so their
 *        result depends on T * V only. Needs scratch_bytes(T * V) of level-0
 *        team scratch.
 */
template <typename Member>
struct TeamLanes {
    static constexpr bool column_major = true;  // dense matrices: a column's rows contiguous across lanes
    static constexpr bool parallel = true;      // balance work across lanes (e.g. Jacobian entries, not rows)
    const Member & member;
    uint32_t lanes;
    double * partial;    // (lanes)
    uint32_t * where;    // (lanes)

    static constexpr size_t scratch_bytes(const uint32_t L) { return L * (sizeof(double) + sizeof(uint32_t)) + 16; }

    KOKKOS_INLINE_FUNCTION TeamLanes(const Member & m, const uint32_t L) : member(m), lanes(L) {
        partial = static_cast<double *>(m.team_scratch(0).get_shmem(L * sizeof(double)));
        where = static_cast<uint32_t *>(m.team_scratch(0).get_shmem(L * sizeof(uint32_t)));
    }

    template <typename F>
    KOKKOS_INLINE_FUNCTION void for_each(const uint32_t n, const F & f) const {
        Kokkos::parallel_for(Kokkos::TeamVectorRange(member, n), f);
    }

    KOKKOS_INLINE_FUNCTION void sync() const { member.team_barrier(); }

    template <typename F>
    KOKKOS_INLINE_FUNCTION void single(const F & f) const {
        Kokkos::single(Kokkos::PerTeam(member), f);
    }

    template <typename F>
    KOKKOS_INLINE_FUNCTION double sum(const uint32_t n, const F & f) const {
        const uint32_t L = lanes;
        Kokkos::parallel_for(Kokkos::TeamVectorRange(member, L), [&](const uint32_t l) {
            double s = 0.0;
            for (uint32_t i = l; i < n; i += L) s += f(i);
            partial[l] = s;
        });
        member.team_barrier();
        double s = 0.0;
        for (uint32_t l = 0; l < L; l++) s += partial[l];
        member.team_barrier();
        return s;
    }

    template <typename F>
    KOKKOS_INLINE_FUNCTION double argmax_abs(const uint32_t first, const uint32_t n, const F & f,
                                             uint32_t & index) const {
        const uint32_t L = lanes;
        Kokkos::parallel_for(Kokkos::TeamVectorRange(member, L), [&](const uint32_t l) {
            double largest = -1.0;
            uint32_t at = n;
            for (uint32_t i = first + l; i < n; i += L) {
                const double v = Kokkos::fabs(f(i));
                if (v > largest) {
                    largest = v;
                    at = i;
                }
            }
            partial[l] = largest;
            where[l] = at;
        });
        member.team_barrier();
        double largest = -1.0;
        index = n;
        for (uint32_t l = 0; l < L; l++) {
            if (partial[l] > largest || (partial[l] == largest && where[l] < index)) {
                largest = partial[l];
                index = where[l];
            }
        }
        member.team_barrier();
        return largest;
    }
};

} // namespace chemistry

#endif // CHEMISTRY_LANES_H

/**
 * @file view_entries.h
 * @brief The most entries a multi-dimensional Kokkos View holds, checked at every
 *        allocation by Mallard's copy of Kokkos_BasicView.hpp (src/CMakeLists.txt).
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 */

#ifndef VIEW_ENTRIES_H
#define VIEW_ENTRIES_H

#include <cstddef>
#include <cstdint>
#include <string_view>
#include <type_traits>

#include <Kokkos_Macros.hpp>

// Kokkos's View header includes this one, so its public headers can't be included here
namespace Kokkos {
struct LayoutLeft;
struct LayoutRight;
} // namespace Kokkos

namespace view_entries {

/**
 * @brief The most entries of a View of rank 2 or more, or a strided one. Kokkos computes
 *        such a View's offset in the common type of the indices, at least 32 bits, which
 *        src/CMakeLists.txt makes unsigned where it would be int, so offsets from 2^32 on
 *        would wrap and alias earlier entries (from 2^31 on with an unpatched Kokkos). A
 *        rank-1 contiguous View has no such limit: its offset is the index.
 */
#ifdef Mallard_UNSIGNED_VIEW_OFFSETS
inline constexpr uint64_t MAX = uint64_t(1) << 32;
#else
inline constexpr uint64_t MAX = uint64_t(1) << 31;
#endif

/**
 * @brief Entries (r, 0) to (r, N - 1) of the rank-2 View v into out, stepping a pointer
 *        along the View's stride. Kokkos's unsigned 32-bit offsets (see MAX) cost devices
 *        an extra add per entry; stepping by a signed 32-bit stride does not. N >= 3 keeps
 *        the stride of a View of at most MAX entries below 2^31.
 */
template <int N, typename View, typename T>
KOKKOS_INLINE_FUNCTION void load_row(const View & v, const uint32_t r, T * out) {
    static_assert(View::rank() == 2);
    static_assert(N >= 3);
    const auto * p = &v(r, 0);
    const int32_t stride = static_cast<int32_t>(v.stride(1));
    out[0] = p[0];
    for (int i = 1; i < N; i++) {
        p += stride;
        out[i] = *p;
    }
}

/**
 * @brief Entries (r, k, i) of the rank-3 View v into out[k][i], as the rank-2 load_row:
 *        in either contiguous layout a row's N1 * N2 entries are one stride apart, and
 *        N1 * N2 >= 3 keeps it below 2^31.
 */
template <size_t N1, size_t N2, typename View, typename T>
KOKKOS_INLINE_FUNCTION void load_row(const View & v, const uint32_t r, T (&out)[N1][N2]) {
    using Layout = typename View::array_layout;
    constexpr bool left = std::is_same_v<Layout, Kokkos::LayoutLeft>;
    static_assert(View::rank() == 3);
    static_assert(left || std::is_same_v<Layout, Kokkos::LayoutRight>);
    static_assert(View::static_extent(1) == N1 && View::static_extent(2) == N2);
    static_assert(N1 * N2 >= 3);
    const auto * p = &v(r, 0, 0);
    const int32_t stride = static_cast<int32_t>(left ? v.stride(1) : v.stride(2));
    for (size_t j = 0; j < N1 * N2; j++) {
        if (j > 0) p += stride;
        out[left ? j % N1 : j / N2][left ? j / N1 : j % N2] = *p;
    }
}

/** @brief Record the cells on this rank, halo included, for the limit check() reports. */
void set_cells(uint64_t n);

/**
 * @brief Reject a View that needs more than MAX entries.
 * @throws diagnostic_code::Error naming the View and, once set_cells() has been called,
 *         the cells per rank that fit
 */
void check(std::string_view label, uint64_t entries);

} // namespace view_entries

#endif // VIEW_ENTRIES_H

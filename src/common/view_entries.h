/**
 * @file view_entries.h
 * @brief The most entries a multi-dimensional Kokkos View holds, checked at every
 *        allocation by Mallard's copy of Kokkos_BasicView.hpp (src/CMakeLists.txt).
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 */

#ifndef VIEW_ENTRIES_H
#define VIEW_ENTRIES_H

#include <cstdint>
#include <string_view>

namespace view_entries {

/**
 * @brief The most entries of a View of rank 2 or more, or a strided one. Kokkos computes
 *        such a View's offset in the common type of the indices, which can be int in
 *        Mallard, so offsets from 2^31 on would wrap and alias earlier entries. A rank-1
 *        contiguous View has no such limit: its offset is the index.
 */
inline constexpr uint64_t MAX = uint64_t(1) << 31;

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

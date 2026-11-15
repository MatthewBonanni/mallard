/**
 * @file index_set.h
 * @brief Cells or faces that a kernel visits: a range, or a list when they are not contiguous.
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 */

#ifndef INDEX_SET_H
#define INDEX_SET_H

#include <cstdint>
#include <vector>

#include <Kokkos_Core.hpp>

/** @brief The listed indices, or [begin, end) when the list is empty. */
struct IndexSet {
    Kokkos::View<uint32_t *> list;
    uint32_t begin = 0, end = 0;

    KOKKOS_INLINE_FUNCTION
    uint32_t size() const { return list.extent(0) ? static_cast<uint32_t>(list.extent(0)) : end - begin; }

    KOKKOS_INLINE_FUNCTION
    uint32_t operator()(const uint32_t k) const { return list.extent(0) ? list(k) : begin + k; }

    /** @brief A range if the indices (ascending) are consecutive, otherwise a list on the device. */
    static IndexSet of(const std::vector<uint32_t> & indices, const char * label) {
        IndexSet set;
        if (indices.empty() || indices.back() - indices.front() + 1 == indices.size()) {
            set.begin = indices.empty() ? 0 : indices.front();
            set.end = set.begin + static_cast<uint32_t>(indices.size());
            return set;
        }
        set.list = Kokkos::View<uint32_t *>(label, indices.size());
        Kokkos::deep_copy(set.list, Kokkos::View<const uint32_t *, Kokkos::HostSpace>(indices.data(), indices.size()));
        return set;
    }
};

/** @brief Applies f to the entries of a list. */
template <typename F>
struct OverList {
    F f;
    Kokkos::View<uint32_t *> list;

    KOKKOS_INLINE_FUNCTION
    void operator()(const uint32_t k) const { f(list(k)); }
};

/** @brief Calls f(i) for every index i of a set, under a range policy with the given traits. */
template <typename... Traits, typename F>
void parallel_for_set(const char * label, const IndexSet & set, const F & f) {
    if (set.list.extent(0) == 0) {
        Kokkos::parallel_for(label, Kokkos::RangePolicy<Traits...>(set.begin, set.end), f);
    } else {
        Kokkos::parallel_for(label, Kokkos::RangePolicy<Traits...>(0, set.list.extent(0)), OverList<F>{f, set.list});
    }
}

#endif // INDEX_SET_H

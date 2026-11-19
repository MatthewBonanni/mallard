/**
 * @file scratch_arena.h
 * @brief One allocation for the work buffers of parts of a step that never run at the same time.
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 */

#ifndef SCRATCH_ARENA_H
#define SCRATCH_ARENA_H

#include <algorithm>
#include <array>
#include <cstddef>
#include <stdexcept>
#include <string>

#include <Kokkos_Core.hpp>

#include "view_entries.h"

namespace memory {

/**
 * @brief Work buffers of parts of a step whose lifetimes never overlap, in one device allocation as large as
 *        the largest part: each part's buffers lie one after another from the start. Reserve every buffer,
 *        allocate once, then take each buffer's view. The views do not own their memory: the arena outlives
 *        them. The memory tracker counts the allocation once, under the arena's label.
 */
class ScratchArena {
    public:
        static constexpr size_t N_PARTS = 2;
        /** @brief Alignment of every buffer, in bytes. */
        static constexpr size_t ALIGNMENT = 256;
        /** @brief bytes rounded up to the alignment: the room a buffer takes. */
        static constexpr size_t aligned(const size_t bytes) { return (bytes + ALIGNMENT - 1) / ALIGNMENT * ALIGNMENT; }

        /**
         * @brief Room for n values of T at the end of a part.
         * @return The buffer's offset in bytes, for view() and data().
         */
        template <typename T>
        size_t reserve(const size_t part, const size_t n) {
            if (part >= N_PARTS || storage.is_allocated()) throw std::logic_error("ScratchArena::reserve");
            const size_t offset = ends[part];
            ends[part] += aligned(n * sizeof(T));
            return offset;
        }

        /** @brief The one allocation, of the largest part; none if nothing was reserved. */
        void allocate(const std::string & label) {
            const size_t bytes = *std::max_element(ends.begin(), ends.end());
            if (bytes > 0) storage = Kokkos::View<double *>(label, bytes / sizeof(double));
        }

        /** @brief Bytes of the allocation. */
        size_t bytes() const { return storage.span() * sizeof(double); }

        /** @brief The memory of the buffer reserved at offset. */
        void * data(const size_t offset) const { return reinterpret_cast<std::byte *>(storage.data()) + offset; }

        /**
         * @brief The buffer reserved at offset as a View of type V, checked as an allocation of that many
         *        entries labeled label would be (view_entries::check).
         */
        template <typename V, typename... Extents>
        V view(const char * label, const size_t offset, const Extents... extents) const {
            const V v(reinterpret_cast<typename V::pointer_type>(data(offset)), extents...);
            if (offset + v.span() * sizeof(typename V::value_type) > bytes()) {
                throw std::logic_error(std::string("ScratchArena::view: ") + label + " exceeds the arena");
            }
            if constexpr (V::rank() > 1) view_entries::check(label, v.span());
            return v;
        }

    private:
        std::array<size_t, N_PARTS> ends{};
        Kokkos::View<double *> storage;
};

/**
 * @brief A View of type V with these extents, allocated under label when own, else without memory until its
 *        owner places it in a ScratchArena; checked as an allocation either way (view_entries::check).
 */
template <typename V, typename... Extents>
V scratch_view(const bool own, const char * label, const Extents... extents) {
    if (own) return V(label, extents...);
    const V v(static_cast<typename V::pointer_type>(nullptr), extents...);
    if constexpr (V::rank() > 1) view_entries::check(label, v.span());
    return v;
}

} // namespace memory

#endif // SCRATCH_ARENA_H

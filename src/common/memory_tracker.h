/**
 * @file memory_tracker.h
 * @brief Bytes held in Kokkos allocations, by memory space, from Kokkos' tool hooks.
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 */

#ifndef MEMORY_TRACKER_H
#define MEMORY_TRACKER_H

#include <cstdint>

namespace memory {

struct Usage {
    uint64_t current = 0;
    uint64_t peak = 0;
};

/**
 * @brief Count Kokkos allocations from now on until the tracker is destroyed
 *        (allocations made before are never subtracted). One at a time.
 */
class Tracker {
    public:
        Tracker();
        ~Tracker();
        Tracker(const Tracker &) = delete;
        Tracker & operator=(const Tracker &) = delete;

        /** @brief Memory of the device's own spaces (none in host builds). */
        Usage device() const;
        /** @brief Host memory, pinned included. */
        Usage host() const;
};

} // namespace memory

#endif // MEMORY_TRACKER_H

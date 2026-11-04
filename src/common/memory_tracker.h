/**
 * @file memory_tracker.h
 * @brief Bytes held in Kokkos allocations, by memory space and by category of
 *        their labels, from Kokkos' tool hooks.
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 */

#ifndef MEMORY_TRACKER_H
#define MEMORY_TRACKER_H

#include <array>
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace memory {

/** @brief What an allocation holds, from its label (category()). */
enum class Category : uint8_t {
    STATE,           ///< Solution, Runge-Kutta stages and right-hand sides, cell fields
    FACES,           ///< Face states and fluxes, gradients, limiters, reconstruction buffers
    TABLES,          ///< Reconstruction tables: TENO stencils, pseudo-inverses, indicator matrices
    MESH,            ///< Connectivity, geometry, quadrature, zones
    MODELS,          ///< Boundary conditions, sources, sponges, LES and wall models
    CHEMISTRY,       ///< Mechanism, thermodynamic and transport tables, chemistry workspace
    DIAGNOSTICS,     ///< Statistics, probes, spectra, conservation ledger, health counters
    OUTPUT,          ///< Host staging of output fields and device mirrors
    HALO,            ///< Halo exchange buffers
    WORK,            ///< Work memory in rows of a number set by the device, MAX_WORK_BYTES at most: chemistry's
    SETUP,           ///< Setup workspace, freed after setup: the initial solution's chunks, TENO's batches
    RUNTIME,         ///< Kokkos' own scratch
    OTHER,           ///< A label no rule names: give it a category in memory_tracker.cpp
};
constexpr size_t N_CATEGORIES = static_cast<size_t>(Category::OTHER) + 1;

/** @brief The most bytes one work-memory allocation (Category::WORK) takes on any device. */
constexpr uint64_t MAX_WORK_BYTES = uint64_t(1) << 30;

/** @brief The category's name in reports. */
const char * category_name(Category c);

/**
 * @brief Whether the category's bytes are proportional to the cells: all but the setup's and the work
 *        memory, which work in chunks and batches of a size set by the device, and Kokkos' scratch.
 */
bool scales_with_cells(Category c);

/** @brief The category of an allocation's label. */
Category category(std::string_view label);

using Bytes = std::array<uint64_t, N_CATEGORIES>;

struct Usage {
    uint64_t current = 0;
    uint64_t peak = 0;
    Bytes by_category{};       ///< current
    Bytes at_peak{};           ///< by category when the peak was reached
    std::vector<Bytes> phases;  ///< by category at the peak of each phase (end_phase()), the open one last
};

/** @brief End a phase of the setup for the active tracker, if any: its peak becomes one of Usage::phases. */
void end_phase();

/**
 * @brief Count Kokkos allocations from now on until the tracker is destroyed
 *        (allocations made before are never subtracted). One at a time; tools
 *        loaded through KOKKOS_TOOLS_LIBS still see every allocation.
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
        /** @brief Start a new peak, and a single phase, from the current usage. */
        void reset_peak();
        /** @brief Labels of category OTHER allocated so far, sorted. */
        std::vector<std::string> unclassified() const;
};

} // namespace memory

#endif // MEMORY_TRACKER_H

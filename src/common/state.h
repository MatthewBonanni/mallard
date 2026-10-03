/**
 * @file state.h
 * @author Matthew Bonanni (mbonanni001@gmail.com)
 * @brief Conservative state: the flow block and the species partial densities.
 * @version 0.3
 * @date 2026-10-02
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 *
 */

#ifndef STATE_H
#define STATE_H

#include <cstdint>
#include <string>

#include <Kokkos_Core.hpp>

#include "common_typedef.h"

/** @brief Flow block [rho, rho u, rho E] per cell. */
using StateView = Kokkos::View<rtype *[N_CONSERVATIVE]>;

/** @brief Layout of the species block; LayoutRight keeps a cell's species contiguous. */
using SpeciesLayout = Kokkos::LayoutRight;

/** @brief Partial densities rho Y_k per (cell, species). */
using SpeciesView = Kokkos::View<rtype **, SpeciesLayout>;

/**
 * @brief Conservative state of every cell: the flow block and the partial
 *        densities of a mixture's species. A single gas has no species
 *        columns, and every loop over them is empty.
 */
struct State {
    StateView flow;
    SpeciesView species;

    State() = default;
    State(StateView flow_, SpeciesView species_) : flow(flow_), species(species_) {}
    State(const std::string & label, uint32_t n_cells, uint32_t n_species)
        : flow(label, n_cells), species(label + "_species", n_cells, n_species) {}

    uint32_t n_species() const { return static_cast<uint32_t>(species.extent(1)); }
};

/** @brief Copy both blocks of src into dst. */
inline void deep_copy(const State & dst, const State & src) {
    Kokkos::deep_copy(dst.flow, src.flow);
    if (src.species.span() > 0) Kokkos::deep_copy(dst.species, src.species);
}

#endif // STATE_H

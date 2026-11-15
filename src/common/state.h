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

/**
 * @brief Layout of the species block: LayoutRight keeps a cell's species
 *        contiguous (the default); LayoutLeft (CMake Mallard_SPECIES_LAYOUT_LEFT)
 *        a species' cells.
 */
#ifdef Mallard_SPECIES_LAYOUT_LEFT
using SpeciesLayout = Kokkos::LayoutLeft;
#else
using SpeciesLayout = Kokkos::LayoutRight;
#endif

/** @brief Partial densities rho Y_k per (cell, species). */
using SpeciesView = Kokkos::View<rtype **, SpeciesLayout>;

/** @brief Positions of the Lagrangian particles: N_DIM values per particle slot (src/particles). */
using ParticleView = Kokkos::View<rtype *>;

/** @brief The partial densities of one cell, in any layout. */
struct CellSpecies {
    const rtype * first;
    size_t stride;
    KOKKOS_INLINE_FUNCTION rtype operator[](const uint32_t k) const { return first[k * stride]; }
};

KOKKOS_INLINE_FUNCTION
CellSpecies cell_species(const SpeciesView & rhoY, const uint32_t c) {
    return CellSpecies{&rhoY(c, 0), rhoY.stride(1)};
}

/**
 * @brief Conservative state of every cell: the flow block and the partial
 *        densities of a mixture's species, and the positions of the
 *        particles, which the time integrators advance with the flow. A single
 *        gas has no species columns and a run without particles no particle
 *        block: every loop over them is empty.
 */
struct State {
    StateView flow;
    SpeciesView species;
    ParticleView particles;

    State() = default;
    State(StateView flow_, SpeciesView species_, ParticleView particles_ = {})
        : flow(flow_), species(species_), particles(particles_) {}
    State(const std::string & label, uint32_t n_cells, uint32_t n_species, size_t n_particle_values = 0)
        : flow(label, n_cells), species(label + "_species", n_cells, n_species),
          particles(n_particle_values ? ParticleView(label + "_particles", n_particle_values) : ParticleView()) {}

    uint32_t n_species() const { return static_cast<uint32_t>(species.extent(1)); }
};

/** @brief Copy the flow and species blocks of src into dst, and the particle block where both have one. */
inline void deep_copy(const State & dst, const State & src) {
    Kokkos::deep_copy(dst.flow, src.flow);
    if (src.species.span() > 0) Kokkos::deep_copy(dst.species, src.species);
    if (src.particles.span() > 0 && dst.particles.span() > 0) Kokkos::deep_copy(dst.particles, src.particles);
}

/** @brief deep_copy(dst, src), ordered on exec without a fence. */
template <typename ExecSpace>
inline void deep_copy(const ExecSpace & exec, const State & dst, const State & src) {
    Kokkos::deep_copy(exec, dst.flow, src.flow);
    if (src.species.span() > 0) Kokkos::deep_copy(exec, dst.species, src.species);
    if (src.particles.span() > 0 && dst.particles.span() > 0) Kokkos::deep_copy(exec, dst.particles, src.particles);
}

#endif // STATE_H

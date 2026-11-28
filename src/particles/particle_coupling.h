/**
 * @file particle_coupling.h
 * @author Matthew Bonanni (mbonanni001@gmail.com)
 * @brief Deposition of per-particle values into the cells: exact sums per
 *        cell, bitwise independent of the slot order and the rank count, and a
 *        conservative spreading over a kernel width in physical units. The
 *        two-way coupling sources and the dispersed-phase fields go through it.
 *        Design: docs/design/particles.md.
 * @version 0.1
 * @date 2026-11-21
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 *
 */

#ifndef PARTICLE_COUPLING_H
#define PARTICLE_COUPLING_H

#include <cstdint>
#include <functional>
#include <vector>

#include <Kokkos_Core.hpp>

#include "common.h"
#include "periodic_maps.h"
#include "state.h"

class Mesh;

namespace particles {

struct Walker;

/**
 * @brief Sums values per slot into the cells holding the slots, then spreads
 *        them over a Gaussian of standard deviation `width`.
 *
 * With a width the cells also sum each value's first and second moments about
 * their centers, and before spreading move each sum by minus the divergence of
 * its first-moment density plus half the double divergence of its
 * second-moment density (face averaged, in flux form), so that the deposit has
 * the particles' first and second moments rather than the cell centers': the
 * spread then converges at second order in the cell size instead of first, and
 * its error does not depend on where the particles sit in their cells.
 *
 * Each cell's sum is exact, rounded once (exact_sums.h), so it does not
 * depend on the order of the slots, which differs between rank counts. The spreading is width^2 / 2 of diffusion with
 * unit diffusivity, by explicit steps of the face-flux Laplacian over the mesh:
 * its continuum limit is the Gaussian, so deposited fields converge under
 * refinement at fixed width, and it conserves the sums to round-off, since a
 * face moves the same amount out of one cell and into the other. Each cell
 * sums its faces in the order of its neighbors' global ids, and halo values
 * come from their owners before each step, so the spread is bitwise
 * independent of the rank count too. A width of 0 leaves the sums in their
 * cells.
 */
class Deposition {
    public:
        /**
         * @brief Allocate for n_columns values per slot (at most N_CONSERVATIVE + n_species), and set up the
         *        spreading over width (collective in distributed runs).
         * @param global_cell The global id of each local cell.
         */
        void init(const Mesh & mesh, const Walker & walker, const std::vector<uint64_t> & global_cell,
                  uint32_t n_slots, uint32_t n_species, uint32_t n_columns, rtype width, bool distributed);

        /** @brief Per-slot values (slot, column) for deposit(). */
        const Kokkos::View<double **, Kokkos::LayoutRight> & slot_values() const { return values; }

        /**
         * @brief Sum slot_values() of the occupied slots (ids not EMPTY) into their cells, then spread them:
         *        result() holds per owned cell the flow-block columns in flow and the rest in species.
         * @param positions The slots' positions (in their cells' frames), the first N_DIM of every stride values.
         * @param exchange Fills the halo cells of a State from their owners (a no-op undistributed).
         */
        void deposit(const Kokkos::View<uint64_t *> & ids, const Kokkos::View<uint32_t *> & cells,
                     const ParticleView & positions, uint32_t stride,
                     const std::function<void(const State &)> & exchange);

        /** @brief The deposited sums (or their spread), per owned cell. */
        const State & result() const { return sums; }

        /** @brief Spreading steps per deposit (0 without a width). */
        uint32_t n_steps() const { return steps; }

        /** @brief The kernel's standard deviation. */
        rtype get_width() const { return width; }

    private:
        uint32_t n_owned = 0;
        uint32_t n_columns = 0;
        uint32_t n_sums = 0;
        uint32_t steps = 0;
        rtype width = 0.0_r;
        Kokkos::View<double **, Kokkos::LayoutRight> values;  // (slot, column)
        Kokkos::View<uint32_t *> count;                         // (cell): its occupied slots
        Kokkos::View<uint32_t *> ends;                          // (cell): the end of its slots in order
        Kokkos::View<uint32_t *> order;                         // occupied slots, grouped by cell
        State sums;                                             // (cell, column)
        State density;                                          // spreading: sums / measure, halo included
        Kokkos::View<rtype *> measure;
        Kokkos::View<rtype *[N_DIM]> cell_coords;
        Kokkos::View<uint8_t *> face_shift;
        PeriodicMaps maps;
        State moments;                        // spreading: first moments per volume, (d n_columns + column)
        State second_moments;                 // spreading: second moments per volume, (pair n_columns + column)
        Kokkos::View<uint32_t *> faces_of_neighbor;  // the face to each neighbor
        Kokkos::View<rtype *[N_DIM]> face_vectors;   // its unit normal out of the cell, in the cell's frame, times its area
        Kokkos::View<int8_t *> turns;         // 1: rotate the neighbor's vectors into the cell's frame, -1 unrotate
        Kokkos::View<uint32_t *> offsets;     // (owned cell + 1): its neighbors in neighbors and weights
        Kokkos::View<uint32_t *> neighbors;   // local cell, by increasing global id
        Kokkos::View<rtype *> weights;        // diffusion step times face area over centroid distance
};

} // namespace particles

#endif // PARTICLE_COUPLING_H

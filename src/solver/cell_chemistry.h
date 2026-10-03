/**
 * @file cell_chemistry.h
 * @author Matthew Bonanni (mbonanni001@gmail.com)
 * @brief Finite-rate chemistry of the cells of a mesh: each cell that needs
 *        it is an adiabatic constant-volume reactor over a splitting step,
 *        integrated by one thread or by the vector lanes of a team.
 * @version 0.3
 * @date 2026-10-03
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 *
 */

#ifndef CELL_CHEMISTRY_H
#define CELL_CHEMISTRY_H

#include <cstdint>

#include <Kokkos_Core.hpp>

#include "kinetics.h"
#include "mixture.h"
#include "reactor.h"
#include "sparse_lu.h"
#include "state.h"

/**
 * @brief Options of CellChemistry: the reactor's tolerances, the temperature
 *        below which cells are frozen, the vector lanes and threads of the
 *        team of each cell (0: automatic; lanes = 1: one thread per cell),
 *        whether a team keeps its factors and vectors in scratch (shared)
 *        memory where they fit (1; default no), and whether
 *        cells integrated by one thread are ordered by their last cost
 *        (binning; on GPUs only).
 */
struct CellChemistryOptions {
    chemistry::ReactorOptions reactor;
    double T_frozen = 0.0;
    uint32_t lanes = 0;
    uint32_t threads = 0;
    int shared = 0;
    bool bin_by_cost = true;
};

/**
 * @brief Work memory and kernels that advance the cells of a state as
 *        adiabatic constant-volume reactors at fixed (rho, e).
 *
 * Cells are processed in chunks that bound the work memory; in each chunk a
 * kernel flags the cells that need chemistry (T >= T_frozen and a mass
 * fraction that would change by more than 1e-2 atol over dt at the current
 * rates), a scan compacts them into a queue, and the queued cells are
 * integrated. Each cell's result depends only on its own state, so neither
 * the order of the queue nor the chunking changes it: per mechanism and
 * build, every cell takes the same code path (one thread per cell, or a
 * fixed number of lanes).
 */
class CellChemistry {
    public:
        /**
         * @param n_cells Most cells advanced in one call (sizes the chunks).
         */
        void init(const Mixture & gas, const chemistry::Mechanism & mechanism,
                  const chemistry::KineticsTable<> & kinetics, const CellChemistryOptions & options,
                  uint32_t n_cells);

        struct Statistics {
            uint64_t active = 0;    // cells integrated
            uint32_t failures = 0;  // cells whose integration failed
        };

        /**
         * @brief Advance cells [0, n) over dt. Partial densities change;
         *        rho, rho u, rho E and the temperature seed do not.
         * @param chem_h Last sub-step of each cell, in and out.
         * @param chem_cost Sub-steps of each cell, incremented.
         */
        Statistics advance(const StateView & U, const SpeciesView & rhoY, const Kokkos::View<rtype *> & T_seed,
                           const Kokkos::View<rtype *> & chem_h, const Kokkos::View<rtype *> & chem_cost, uint32_t n,
                           double dt);

        /** @brief Heat release rate and mass production rates W_k omega_k of cells [0, n). */
        void heat_release(const StateView & U, const SpeciesView & rhoY, const Kokkos::View<rtype *> & T_seed,
                          const Kokkos::View<rtype *> & hrr, const Kokkos::View<rtype **, Kokkos::LayoutRight> & production,
                          uint32_t n);

        /** @brief Vector lanes per cell thread (1: one thread per cell). */
        uint32_t lanes() const { return n_lanes; }

        /** @brief Threads of a cell's team. */
        uint32_t threads() const { return n_threads; }

        /** @brief Whether cells integrated one per thread are ordered by their last cost (GPUs). */
        bool binned() const { return bin_by_cost; }

        /** @brief Bytes of team scratch memory per cell holding its work memory, 0 if global memory. */
        size_t shared_bytes() const { return fast_bytes; }

        /** @brief Entries of L + U of the sparse LU, 0 for the dense one. */
        uint32_t sparse_entries() const { return sparse ? pattern.nnz : 0; }

    private:
        Mixture gas;
        chemistry::KineticsTable<> kinetics;
        CellChemistryOptions options;
        uint32_t n_lanes = 1;
        uint32_t n_threads = 1;
        bool bin_by_cost = false;
        size_t fast_bytes = 0;
        bool sparse = false;
        chemistry::SparseLUPattern<> pattern;
        Kokkos::View<double **, Kokkos::LayoutRight> work;  // (cell of a chunk, work)
        Kokkos::View<uint32_t **, Kokkos::LayoutRight> pivot;
        Kokkos::View<uint32_t *> active, queue;
        Kokkos::View<float *> cost;  // binning: the last cost of each queued cell
        Kokkos::View<float *> previous_cost;  // (cell)
};

#endif // CELL_CHEMISTRY_H

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
#include <memory>

#include <Kokkos_Core.hpp>

#include "kinetics.h"
#include "mixture.h"
#include "reactor.h"
#include "sparse_lu.h"
#include "state.h"

/**
 * @brief Options of CellChemistry: the reactor's tolerances, the temperature
 *        below which cells are frozen, the vector lanes and threads of the
 *        team of each cell (0: automatic; lanes = 1: one thread per cell;
 *        automatic threads on GPUs: one, and several warps for the cells
 *        whose last call was expensive),
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
    bool forced = false;  // calls may take a ChemistryForcing
};

/**
 * @brief A constant forcing of a chemistry call: the rates dU/dt of every
 *        conservative variable of each cell (rho, rho u, rho E, rho Y_k),
 *        e.g. the transport tendency of balanced splitting or of simplified
 *        spectral deferred corrections, and a state of the same size for
 *        scratch.
 */
struct ChemistryForcing {
    State rate;
    State scratch;
};

/**
 * @brief The per-cell views a chemistry call reads and writes: the state,
 *        the temperature seed, the last sub-step, the sub-steps of the step
 *        and of the last call, the optional multiplier of dt, and with a
 *        forcing its species rates and U + dt rate, whose partial densities
 *        receive the result instead of rhoY (all empty without one).
 */
struct ChemistryCells {
    StateView U;
    SpeciesView rhoY;
    Kokkos::View<rtype *> T_seed;
    Kokkos::View<rtype *> chem_h;
    Kokkos::View<rtype *> chem_cost;
    Kokkos::View<float *> previous_cost;
    Kokkos::View<rtype *> time_scale;
    SpeciesView rate;
    StateView U_end;
    SpeciesView rhoY_end;

    bool forced() const { return U_end.extent(0) > 0; }
};

class ChemistryBalance;

/**
 * @brief Work memory and kernels that advance the cells of a state as
 *        adiabatic constant-volume reactors at fixed (rho, e).
 *
 * A kernel flags the cells that need chemistry (T >= T_frozen and a mass
 * fraction that would change by more than 1e-2 atol over dt at the current
 * rates), a scan compacts them into a queue, and the queued cells are
 * integrated in batches that bound the work memory, after the ranks have
 * balanced the queues if asked to (ChemistryBalance). Each cell's result
 * depends only on its own state, so neither the order of the queue, the
 * batches nor the rank that integrates it changes it: per mechanism and
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
            uint64_t active = 0;    // cells that needed chemistry
            uint32_t failures = 0;  // cells whose integration here failed (sent cells: where they ran)
            uint64_t sent = 0;      // active cells integrated by another rank
        };

        /**
         * @brief Balances the integration across ranks from now on (collective:
         *        every rank of the run, every call): see ChemistryBalance.
         */
        void balance_across_ranks(double threshold);

        /**
         * @brief Advance cells [0, n) over dt. Partial densities change;
         *        rho, rho u, rho E and the temperature seed do not.
         * @param chem_h Last sub-step of each cell, in and out.
         * @param chem_cost Sub-steps of each cell, incremented.
         * @param time_scale Optional per-cell multiplier of dt: the rates of an
         *        autonomous constant-volume reactor scaled by s give exactly its
         *        state after s dt (combustion models, docs/design/les.md 6.4).
         * @param forcing Optional (options.forced): integrate
         *        dU/dt = s R(U) + forcing->rate instead, so that rho, rho u
         *        and rho E change by dt times their rates and every cell
         *        moves, reacting or not. A cell reacts if its chemistry is
         *        active at U or at U + dt rate; its internal energy per
         *        volume is taken linear in time between the two.
         */
        Statistics advance(const StateView & U, const SpeciesView & rhoY, const Kokkos::View<rtype *> & T_seed,
                           const Kokkos::View<rtype *> & chem_h, const Kokkos::View<rtype *> & chem_cost, uint32_t n,
                           double dt, const Kokkos::View<rtype *> & time_scale = {},
                           const ChemistryForcing * forcing = nullptr);

        /** @brief Heat release rate and mass production rates W_k omega_k of cells [0, n). */
        void heat_release(const StateView & U, const SpeciesView & rhoY, const Kokkos::View<rtype *> & T_seed,
                          const Kokkos::View<rtype *> & hrr, const Kokkos::View<rtype **, Kokkos::LayoutRight> & production,
                          uint32_t n);

        /** @brief Vector lanes per cell thread (1: one thread per cell). */
        uint32_t lanes() const { return n_lanes; }

        /** @brief Threads of a cell's team. */
        uint32_t threads() const { return n_threads; }

        /** @brief Threads of the teams of the cells whose last call was expensive (GPUs), 0 if the same. */
        uint32_t wide_threads() const { return n_wide_threads; }

        /** @brief Whether cells integrated one per thread are ordered by their last cost (GPUs). */
        bool binned() const { return bin_by_cost; }

        /** @brief Bytes of team scratch memory per cell holding its work memory, 0 if global memory. */
        size_t shared_bytes() const { return fast_bytes; }

        /** @brief Entries of L + U of the sparse LU, 0 for the dense one. */
        uint32_t sparse_entries() const { return sparse ? pattern.nnz : 0; }

    private:
        /**
         * @brief Integrates queued cells [0, n) of `cells` in batches of the work memory, with
         *        their forcing if any; returns failures.
         */
        uint32_t integrate(const ChemistryCells & cells, const Kokkos::View<uint32_t *> & queued,
                           const Kokkos::View<float *> & queued_cost, uint32_t n, double dt);

        std::shared_ptr<ChemistryBalance> balance;
        Mixture gas;
        chemistry::KineticsTable<> kinetics;
        CellChemistryOptions options;
        uint32_t n_lanes = 1;
        uint32_t n_threads = 1;
        uint32_t n_wide_threads = 0;  // threads of the expensive cells' teams, 0: the same teams for all
        Kokkos::DefaultExecutionSpace narrow_space, wide_space;
        Kokkos::View<uint32_t> team_failures[2];  // failures of the narrow and wide teams
        bool bin_by_cost = false;
        size_t fast_bytes = 0;
        bool sparse = false;
        uint32_t forcing_offset = 0;  // of the forcing in a cell's work memory, 0: none
        chemistry::SparseLUPattern<> pattern;
        Kokkos::View<double **, Kokkos::LayoutRight> work;  // (cell of a chunk, work)
        Kokkos::View<uint32_t **, Kokkos::LayoutRight> pivot;
        Kokkos::View<uint32_t *> active, queue;  // (cell): flags, then the cells that need chemistry
        Kokkos::View<float *> cost;  // binning: minus the last cost of each queued cell
        Kokkos::View<float *> previous_cost;  // (cell)
};

#endif // CELL_CHEMISTRY_H

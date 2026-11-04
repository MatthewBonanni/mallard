/**
 * @file initial_state.h
 * @author Matthew Bonanni (mbonanni001@gmail.com)
 * @brief Initial states from expressions, layered regions and tabulated data,
 *        integrated over each cell on the device.
 * @version 0.1
 * @date 2026-10-23
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 *
 */

#ifndef INITIAL_STATE_H
#define INITIAL_STATE_H

#include <memory>
#include <string>
#include <vector>

#include "input.h"
#include "physics.h"
#include "state.h"

class Mesh;
class MixtureModel;

/**
 * @brief The state [initialize] describes over a base: the base state (the
 *        expressions or numbers of an analytical or constant state, or the
 *        cells of a restart or remap), then [[initialize.regions]] layered
 *        over it by priority, each setting some of u, rho, p, T and the
 *        composition inside a region; expressions read x, y, z and the
 *        fields of [[initialize.data]] (tables and point clouds). Cell
 *        averages are integrated on the device: each cell's simplices are
 *        subdivided n_subdivisions times, then adaptively where a region's
 *        surface or smooth edge crosses them, and finally cut along the
 *        surface (exact for flat ones). See docs/guide/initialization.md.
 */
class InitialState {
    public:
        /**
         * @param input The run's input.
         * @param mesh The mesh.
         * @param physics The single gas (unused for mixtures).
         * @param mixture The gas mixture, or null.
         * @param base_from_cells The base is the cells' current state (a
         *        restart or remap), else the expressions of [initialize].
         */
        InitialState(const Input & input, const Mesh & mesh, const Euler & physics, const MixtureModel * mixture,
                     bool base_from_cells);
        ~InitialState();
        InitialState(const InitialState &) = delete;
        InitialState & operator=(const InitialState &) = delete;

        /** @brief Whether [initialize] has regions or data. */
        static bool layered(const Input & input);

        /**
         * @brief Set every cell (owned and halo) to the cell average of the
         *        state. With a base from the cells, U, rhoY and T_seed hold
         *        it on entry, and cells no region reaches keep it exactly.
         * @param U Flow block (host).
         * @param rhoY Partial densities (host).
         * @param T_seed Temperature guesses of a mixture's base (host).
         */
        void apply(const StateView::host_mirror_type & U, const SpeciesView::host_mirror_type & rhoY,
                   const Kokkos::View<rtype *>::host_mirror_type & T_seed) const;

        /** @brief Description for the run log. */
        const std::string & summary() const { return m_summary; }

    private:
        struct Impl;
        std::unique_ptr<Impl> impl;
        std::string m_summary;
};

#endif // INITIAL_STATE_H

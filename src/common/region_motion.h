/**
 * @file region_motion.h
 * @author Matthew Bonanni (mbonanni001@gmail.com)
 * @brief The cell fractions of a moving region, or of one defined by the
 *        solution, recomputed on the device at a time read from the device:
 *        no host work, so a CUDA graph can capture it.
 * @version 0.1
 * @date 2026-11-10
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 *
 */

#ifndef REGION_MOTION_H
#define REGION_MOTION_H

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include <Kokkos_Core.hpp>
#include "cell_fields.h"
#include "common_typedef.h"
#include "periodic_maps.h"
#include "region.h"
#include "region_mesh.h"

class Mesh;

namespace region {

/**
 * @brief The levels of a region's conditions in every cell, from the
 *        solution: each one's value at the cell's centroid and its gradient,
 *        by inverse-distance-squared least squares over the cells sharing a
 *        node with it (across periodic boundaries, their images).
 *        A condition's cell fractions and isosurfaces are cut from this
 *        linear reconstruction, exact for a linear level. Halo cells get
 *        their value and no gradient.
 */
class SolutionLevels {
    public:
        /** @brief Compiles the conditions' levels in the variables of condition_variables(species). */
        SolutionLevels(const Mesh & mesh, const std::vector<Condition> & conditions,
                       const std::vector<std::string> & species, const CellFields & fields);

        /** @brief levels() at the time time() on the device, enqueued without host work or allocation. */
        void update(const Kokkos::View<const rtype> & time) const;
        /** @brief levels() at time t, given on the host. */
        void update(double t) const;

        /** @brief (cell, condition, [value, gradient]) of every cell. */
        const Kokkos::View<double **[N_DIM + 1]> & levels() const { return m_levels; }
        const Kokkos::View<rtype *[N_DIM]> & centroids() const { return m_centroids; }

        uint32_t size() const { return n_conditions; }

    private:
        uint32_t n_cells = 0, n_owned = 0, n_conditions = 0, n_species = 0;
        std::vector<kexpr::Program> programs;
        kexpr::ProgramTable<> table;
        CellFields fields;
        Kokkos::View<rtype *[N_DIM]> m_centroids;
        Kokkos::View<uint32_t *> offsets_cells_of_cell, cells_of_cell;
        Kokkos::View<uint8_t *> cells_of_cell_shift;
        PeriodicMaps periodic;
        Kokkos::View<rtype> host_time;
        Kokkos::View<double **[N_DIM + 1]> m_levels;
};

/**
 * @brief The levels a surface or curve of the solution is cut from
 *        (region::elements with levels), in the owned cells level_cells reads
 *        them in, selected on the device: only those cells' levels reach the
 *        host.
 */
class CrossedLevels {
    public:
        CrossedLevels(const Mesh & mesh, const Region & region, std::shared_ptr<const SolutionLevels> levels);

        /** @brief The levels at time t in the cells, on the host. */
        Levels update(double t) const;

    private:
        uint32_t n_owned = 0;
        bool every = false;
        std::shared_ptr<const SolutionLevels> m_levels;
        Kokkos::View<const uint32_t *> offsets, conditions;  // of each piece cut from conditions' levels
        Kokkos::View<const double *> radii;
        Kokkos::View<uint32_t *> cells;
        Kokkos::View<double **[N_DIM + 1], Kokkos::LayoutRight> values;
};

/**
 * @brief The fractions of every cell (owned and halo) inside a filled region
 *        that moves, as cell_fractions gives them at any time: its moving
 *        nodes placed at the time, then each cell sampled as cell_fractions
 *        samples it. Curved cells are sampled on their cones, through
 *        Chebyshev interpolants of degree DEGREE of their faces' maps; parts
 *        of the mesh (cell zones, the domain) are the fixed fractions of
 *        cell_fractions.
 */
class MovingFractions {
    public:
        static constexpr int DEGREE = 8;

        /**
         * @brief For filled part i of the region (its root by default), with
         *        the levels of its conditions if it has any. Collective (fixed
         *        parts of the mesh).
         */
        MovingFractions(const Mesh & mesh, const Region & region,
                        std::shared_ptr<const SolutionLevels> levels = nullptr);
        MovingFractions(const Mesh & mesh, const Region & region, uint32_t filled,
                        std::shared_ptr<const SolutionLevels> levels = nullptr);

        /**
         * @brief fractions() at the time time() on the device (and the levels
         *        of conditions on the current solution), enqueued on the default
         *        execution space without host work or allocation.
         */
        void update(const Kokkos::View<const rtype> & time) const;

        /** @brief fractions() at time t, given on the host: a copy of t to the device, then update. */
        void update(double t) const;

        const Kokkos::View<double *> & fractions() const { return m_fractions; }

        struct Leaf {
            uint32_t first = 0, count = 0;  // its nodes
            double edge = 0.0;
        };
        struct Op {
            enum class Kind : uint8_t { SHAPES, CELLS, CONDITION, UNION, INTERSECTION, DIFFERENCE };
            Kind kind = Kind::SHAPES;
            uint32_t index = 0;  // SHAPES: a leaf; CELLS: a column of fixed fractions; CONDITION: its level
        };
        struct CurvedFace {
            double lo[2] = {0.0, 0.0}, hi[2] = {1.0, 1.0};  // its box of face coordinates
            double o[3] = {0.0, 0.0, 0.0};                  // the cell's vertex average, the cones' apex
            double sign = 1.0;
            uint32_t first = 0;                             // its Chebyshev coefficients
        };

    private:
        uint32_t n_cells = 0;
        bool bounded = true;
        bool axisymmetric = false;
        Kokkos::View<Node *> nodes;
        Kokkos::View<const double *> vertices;
        kexpr::ProgramTable<> programs, times;
        Kokkos::View<const Motion *> motions;
        Kokkos::View<const Step *> steps;
        Kokkos::View<const Leaf *> leaves;
        Kokkos::View<const Op *> ops;
        Kokkos::View<const double **, Kokkos::LayoutLeft> fixed;  // columns of fixed fractions
        Kokkos::View<const double *[4]> spheres;                  // per cell: a center and the radius around it
        Kokkos::View<const rtype *[N_DIM]> node_coords;
        Kokkos::View<const uint32_t *> nodes_of_cell, offsets_nodes_of_cell;
        // Curved cells: their faces, from offsets(curved(c)), and the coefficients of the faces' maps
        Kokkos::View<const int32_t *> curved;
        Kokkos::View<const uint32_t *> curved_offsets;
        Kokkos::View<const CurvedFace *> curved_faces;
        Kokkos::View<const double *[3]> coefficients;
        Kokkos::View<rtype> host_time;
        Kokkos::View<double *> m_fractions;
        std::shared_ptr<const SolutionLevels> m_levels;
};

} // namespace region

#endif // REGION_MOTION_H

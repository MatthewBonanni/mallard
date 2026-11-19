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
 *        their value and no gradient. Fields of the velocity gradient
 *        (CONDITION_GRADIENT_FIELDS) take it from a first fit of the same
 *        kind, in every cell with all its vertex neighbors (Mesh::n_complete).
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
        uint32_t n_cells = 0, n_owned = 0, n_complete = 0, n_conditions = 0, n_species = 0;
        bool axisymmetric = false;
        std::vector<kexpr::Program> programs;
        kexpr::ProgramTable<> table;
        CellFields fields;
        Kokkos::View<rtype *[N_DIM]> m_centroids;
        Kokkos::View<uint32_t *> offsets_cells_of_cell, cells_of_cell;
        ShiftRows cells_of_cell_shift;
        PeriodicMaps periodic;
        Kokkos::View<rtype> host_time;
        Kokkos::View<double **[N_DIM + 1]> m_levels;
        Kokkos::View<rtype *[N_DIM][N_DIM]> velocity_gradients;  // (cell, k, i) = d u_k / d x_i, if read
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
        /**
         * @brief The levels at the time time() on the device, in the first count() of cells() and values(), enqueued
         *        without host work or allocation.
         */
        void select(const Kokkos::View<const rtype> & time) const;

        const Kokkos::View<uint32_t *> & cells() const { return m_cells; }
        const Kokkos::View<uint32_t> & count() const { return m_count; }
        const Kokkos::View<double **[N_DIM + 1], Kokkos::LayoutRight> & values() const { return m_values; }
        uint32_t size() const { return m_levels->size(); }
        /** @brief The cells and values of the current levels(), and their count (public for nvcc's device lambdas). */
        void select_cells() const;

    private:
        uint32_t n_owned = 0;
        bool every = false;
        std::shared_ptr<const SolutionLevels> m_levels;
        Kokkos::View<const uint32_t *> offsets, conditions;  // of each piece cut from conditions' levels
        Kokkos::View<const double *> radii;
        Kokkos::View<uint32_t *> m_cells;
        Kokkos::View<uint32_t> m_count;
        Kokkos::View<double **[N_DIM + 1], Kokkos::LayoutRight> m_values;
};

/**
 * @brief The quadrature points of a surface cut from one condition alone
 *        (single_level) in the first count() of a CrossedLevels' cells, on
 *        the device: those elements(mesh, region, levels) gives in these
 *        cells, bit for bit and in its order, enqueued without host work or
 *        allocation.
 */
class LevelCut {
    public:
        /** @brief Of the condition single_level gives, with room for capacity points. */
        LevelCut(const Mesh & mesh, uint32_t condition, std::shared_ptr<const CrossedLevels> crossed, uint32_t capacity);

        /**
         * @brief Cuts the crossed cells' current levels (after CrossedLevels::select): total() points, kept in the
         *        first total() of cells(), points(), weights() and normals() if they fit in capacity().
         */
        void select() const;

        const Kokkos::View<uint32_t *> & cells() const { return m_cells; }
        const Kokkos::View<double *[3]> & points() const { return m_points; }
        const Kokkos::View<double *> & weights() const { return m_weights; }
        const Kokkos::View<double *[3]> & normals() const { return m_normals; }
        const Kokkos::View<uint32_t> & total() const { return m_total; }
        uint32_t capacity() const { return static_cast<uint32_t>(m_cells.extent(0)); }

    private:
        std::shared_ptr<const CrossedLevels> m_crossed;
        uint32_t m_condition = 0;
        Kokkos::View<const uint32_t *> offsets_nodes, nodes;  // of the owned cells
        Kokkos::View<const double *[3]> node_coords;
        Kokkos::View<const rtype *[N_DIM]> cell_coords;
        Kokkos::View<uint32_t *> row_offsets;  // of each crossed cell's points, then their total
        Kokkos::View<uint32_t *> m_cells;
        Kokkos::View<double *[3]> m_points, m_normals;
        Kokkos::View<double *> m_weights;
        Kokkos::View<uint32_t> m_total;
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
        bool implicit_shapes = false;
        bool sharp_leaves = false, smooth_leaves = false;  // whether the straight cells' sharp or SMOOTH ShapeSampler has leaves
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
        Kokkos::View<double **, Kokkos::LayoutLeft> shapes;  // (cell, leaf): the fractions of the shapes
        Kokkos::View<double *> m_fractions;
        std::shared_ptr<const SolutionLevels> m_levels;
};

} // namespace region

#endif // REGION_MOTION_H

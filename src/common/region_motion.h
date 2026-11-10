/**
 * @file region_motion.h
 * @author Matthew Bonanni (mbonanni001@gmail.com)
 * @brief The cell fractions of a moving region, recomputed on the device at
 *        a time read from the device: no host work, so a CUDA graph can
 *        capture it.
 * @version 0.1
 * @date 2026-11-10
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 *
 */

#ifndef REGION_MOTION_H
#define REGION_MOTION_H

#include <cstdint>

#include <Kokkos_Core.hpp>
#include "common_typedef.h"
#include "region.h"

class Mesh;

namespace region {

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

        /** @brief For filled part i of the region (its root by default). Collective (fixed parts of the mesh). */
        MovingFractions(const Mesh & mesh, const Region & region);
        MovingFractions(const Mesh & mesh, const Region & region, uint32_t filled);

        /**
         * @brief fractions() at the time time() on the device, enqueued on
         *        the default execution space without host work or allocation.
         */
        void update(const Kokkos::View<const rtype> & time) const;

        /** @brief fractions() at time t, given on the host: a copy of t to the device, then update. */
        void update(double t) const;

        const Kokkos::View<double *> & fractions() const { return m_fractions; }

        /** @brief Bytes of device memory it holds. */
        size_t bytes() const;

        struct Leaf {
            uint32_t first = 0, count = 0;  // its nodes
            double edge = 0.0;
        };
        struct Op {
            enum class Kind : uint8_t { SHAPES, CELLS, UNION, INTERSECTION, DIFFERENCE };
            Kind kind = Kind::SHAPES;
            uint32_t index = 0;  // SHAPES: a leaf; CELLS: a column of fixed fractions
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
};

} // namespace region

#endif // REGION_MOTION_H

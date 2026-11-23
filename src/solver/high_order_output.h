/**
 * @file high_order_output.h
 * @brief The high-order view of the snapshots: the reconstruction polynomials
 *        sampled at the nodes of VTK Lagrange cells (docs/design/high_order_output.md).
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 */

#ifndef HIGH_ORDER_OUTPUT_H
#define HIGH_ORDER_OUTPUT_H

#include <memory>
#include <vector>

#include <Kokkos_Core.hpp>

#include "common.h"
#include "data_writer.h"
#include "face_reconstruction.h"
#include "mesh.h"
#include "physics.h"

/**
 * @brief Fills a LagrangeView: every owned cell as a Lagrange cell of the
 *        degree of the reconstruction polynomials, and at its nodes the
 *        requested fields from its polynomial, or from its average where the
 *        reconstruction has no admissible value there (troubled cells).
 */
class HighOrderOutput {
    public:
        /**
         * @brief Sets the view's order, method and fields: those of TENO's polynomials,
         *        none for reconstructions without (order 0).
         */
        void init(std::shared_ptr<Mesh> mesh, const FaceReconstruction * reconstruction, bool mixture,
                  LagrangeView & view);

        /** @brief Once the writers requested their fields: the nodes, their geometry and the arrays of their values. */
        void setup();

        /** @brief Whether any writer writes the view. */
        bool active() const { return n_nodes > 0; }

        /**
         * @brief Enqueues the requested fields at every node from the polynomials of
         *        W_cells, current in the halo too; the view's values hold them after a fence.
         */
        void update(FaceReconstruction & reconstruction, Kokkos::View<rtype *[N_CONSERVATIVE]> W_cells,
                    const Euler & physics);

    private:
        /** @brief The cells of one shape, with the same nodes per cell, evaluated chunk by chunk. */
        struct Group {
            uint32_t m = 0;        // nodes per cell
            uint32_t n_cells = 0;
            uint32_t chunk = 0;    // cells per chunk
            uint32_t n_basis[2] = {0, 0};
            Kokkos::View<uint32_t *> cells;        // padded to whole chunks with the last cell
            Kokkos::View<uint32_t *> geometry;     // where its geometry nodes start
            Kokkos::View<uint8_t *> curved;        // 1: quadratic geometry
            Kokkos::View<uint64_t *> first_node;   // in the view
            Kokkos::View<rtype **> basis[2];       // [node][geometry node] of linear and quadratic geometry
            Kokkos::View<rtype *[N_DIM]> offsets;  // chunk work: nodes from the centroid
            Kokkos::View<rtype *[N_CONSERVATIVE]> W;
        };

        std::shared_ptr<Mesh> mesh;
        LagrangeView * view = nullptr;
        uint64_t n_nodes = 0;
        uint32_t n_columns = 0;
        std::vector<Group> groups;
        Kokkos::View<rtype *[N_DIM]> geometry;  // the cells' geometry nodes
        Kokkos::View<uint8_t *> codes;          // per requested field
        Kokkos::View<uint32_t *> columns;       // per requested field: its first column
        Kokkos::View<rtype *, Kokkos::SharedHostPinnedSpace> values;  // [field][node][component]
};

#endif // HIGH_ORDER_OUTPUT_H

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
#include <optional>
#include <vector>

#include <Kokkos_Core.hpp>

#include "common.h"
#include "data_writer.h"
#include "face_reconstruction.h"
#include "mesh.h"
#include "physics.h"

/**
 * @brief Fills the LagrangeView the writers write: every owned cell as
 *        Lagrange cells of the polynomials' degree, and at their nodes the
 *        requested fields from its polynomial, or from its average where the
 *        reconstruction has no admissible value there (troubled cells).
 */
class HighOrderOutput {
    public:
        /**
         * @brief Sets the view's degree (that of TENO's polynomials), method and fields;
         *        degree 0 for reconstructions without.
         */
        void init(std::shared_ptr<Mesh> mesh, const FaceReconstruction * reconstruction, bool mixture,
                  LagrangeView & view);

        /** @brief Once the writers requested their fields: the nodes, their geometry and the arrays of their values. */
        void setup();

        /** @brief Whether any writer writes a view. */
        bool active() const { return built.has_value(); }

        /**
         * @brief Enqueues the requested fields at every node from the polynomials of
         *        W_cells, current in the halo too; the view's values hold them after a fence.
         */
        void update(FaceReconstruction & reconstruction, Kokkos::View<rtype *[N_CONSERVATIVE]> W_cells,
                    const Euler & physics);

    private:
        /** @brief Lagrange cells of one shape, with the same nodes each, evaluated chunk by chunk. */
        struct Group {
            uint32_t m = 0;        // nodes per Lagrange cell
            uint32_t n_cells = 0;
            uint32_t chunk = 0;    // Lagrange cells per chunk
            uint32_t n_basis = 0;  // vertices
            bool exact = false;    // the nodes are given (curved cells), not the vertices
            Kokkos::View<uint32_t *> cells;        // their mesh cells, padded to whole chunks with the last
            Kokkos::View<uint32_t *> geometry;     // where their vertices, or (exact) nodes, start
            Kokkos::View<uint64_t *> first_node;   // in the view
            Kokkos::View<rtype **> basis;          // [node][vertex], but exact
            Kokkos::View<rtype *[N_DIM]> offsets;  // chunk work: nodes from the centroid
            Kokkos::View<rtype *[N_CONSERVATIVE]> W;
        };
        /** @brief The view the writers write, and its device data. */
        struct Built {
            LagrangeView * view = nullptr;
            uint64_t n_nodes = 0;
            std::vector<Group> groups;
            Kokkos::View<rtype *[N_DIM]> geometry;  // the vertices and given nodes
            Kokkos::View<uint8_t *> codes;          // per requested field
            Kokkos::View<uint32_t *> columns;       // per requested field: its first column
            Kokkos::View<rtype *, Kokkos::SharedHostPinnedSpace> values;  // [field][node][component]
        };

        void build(Built & b);

        std::shared_ptr<Mesh> mesh;
        LagrangeView * view = nullptr;
        std::optional<Built> built;
};

#endif // HIGH_ORDER_OUTPUT_H

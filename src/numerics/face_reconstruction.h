/**
 * @file face_reconstruction.h
 * @author Matthew Bonanni (mbonanni001@gmail.com)
 * @brief Face reconstruction class declaration.
 * @version 0.1
 * @date 2023-12-24
 * 
 * @copyright Copyright (c) 2023 Matthew Bonanni
 * 
 */

#ifndef FACE_RECONSTRUCTION_H
#define FACE_RECONSTRUCTION_H

#include <array>
#include <functional>
#include <memory>
#include <unordered_map>
#include <vector>

#include "input.h"

#include "common_math.h"
#include "common_typedef.h"
#include "index_set.h"
#include "mesh.h"
#include "face_point_gradients.h"
#include "face_weights.h"
#include "quadrature.h"
#include "boundary.h"
#include "gradient.h"
#include "log.h"
#include "report.h"
#include "teno.h"

/**
 * @brief Point (a, b) of a planar 3D face's quadrature rule from its nodes v:
 *        v0 + a (v1 - v0) + b (v2 - v0) on a triangle, the bilinear map at
 *        (s, t) = (a, b) on a quadrilateral. Every product is rounded on its
 *        own, so host and device place the points alike.
 */
template <typename V>
KOKKOS_INLINE_FUNCTION void planar_face_point(const bool triangle, const double a, const double b, const V & v,
                                              double * p) {
    if (triangle) {
        for (uint8_t i = 0; i < 3; i++) {
            p[i] = v[0][i] + rounded_product(a, v[1][i] - v[0][i]) + rounded_product(b, v[2][i] - v[0][i]);
        }
        return;
    }
    const double N[4] = {0.25 * (1 - a) * (1 - b), 0.25 * (1 + a) * (1 - b), 0.25 * (1 + a) * (1 + b),
                         0.25 * (1 - a) * (1 + b)};
    for (uint8_t i = 0; i < 3; i++) {
        p[i] = 0.0;
        for (uint8_t k = 0; k < 4; k++) p[i] += rounded_product(N[k], v[k][i]);
    }
}

// The faces' nodes (face, k), a triangle's last node repeated, and the nodes' coordinates
using FaceNodes = Kokkos::View<uint32_t *[2 * (N_DIM - 1)], Kokkos::LayoutRight>;
using NodeCoords = Kokkos::View<rtype *[N_DIM], Kokkos::LayoutRight>;

/** @brief Nodes of a planar 3D face, from which planar_face_point places its quadrature points. */
struct PlanarFace {
    rtype v[4][3] = {};
    bool triangle = false;
};

/** @brief The nodes of face f (3D), from a FaceNodes-like table. */
template <typename Nodes, typename Coords>
KOKKOS_INLINE_FUNCTION PlanarFace planar_face(const Nodes & face_nodes, const Coords & node_coords, const uint32_t f) {
    PlanarFace face;
    uint32_t n[4];
    for (uint32_t k = 0; k < 4; k++) n[k] = face_nodes(f, k);
    face.triangle = n[3] == n[2];
    for (uint32_t k = 0; k < 4; k++) {
        FOR_I_DIM face.v[k][i] = node_coords(n[k], i);
    }
    return face;
}

/**
 * @brief Point q of a planar 3D face (FaceReconstruction::face_rule), as
 *        FaceReconstruction::init_face_quadrature_3d places it.
 */
template <typename Rule>
KOKKOS_INLINE_FUNCTION void planar_face_point(const PlanarFace & face, const Rule & rule, const uint32_t q, rtype * x) {
    double v[4][3];
    for (uint8_t k = 0; k < 4; k++) {
        for (uint8_t i = 0; i < 3; i++) v[k][i] = double(face.v[k][i]);
    }
    double p[3];
    planar_face_point(face.triangle, double(rule(q, face.triangle ? 2 : 0)), double(rule(q, face.triangle ? 3 : 1)), v, p);
    for (uint8_t i = 0; i < 3; i++) x[i] = static_cast<rtype>(p[i]);
}

enum class FaceReconstructionType {
    FIRST_ORDER,
    MUSCL,
    TENO,
};

static const std::unordered_map<std::string, FaceReconstructionType> FACE_RECONSTRUCTION_TYPES = {
    {"FO", FaceReconstructionType::FIRST_ORDER},
    {"MUSCL", FaceReconstructionType::MUSCL},
    {"TENO", FaceReconstructionType::TENO},
};

static const std::unordered_map<FaceReconstructionType, std::string> FACE_RECONSTRUCTION_NAMES = {
    {FaceReconstructionType::FIRST_ORDER, "FO"},
    {FaceReconstructionType::MUSCL, "MUSCL"},
    {FaceReconstructionType::TENO, "TENO"},
};

/**
 * @brief Face reconstruction class.
 */
class GrowingBuffer;

class FaceReconstruction {
    public:
        /**
         * @brief Construct a new Face Reconstruction object
         */
        FaceReconstruction();

        /**
         * @brief Destroy the Face Reconstruction object
         */
        virtual ~FaceReconstruction();

        /**
         * @brief Initialize the face reconstruction.
         */
        virtual void init(const Input & input) = 0;

        /**
         * @brief Display lines for the run log.
         */
        virtual report::Summary summary() const;

        /**
         * @brief Set the mesh.
         * @param mesh Pointer to the mesh.
         */
        void set_mesh(std::shared_ptr<Mesh> mesh);

        /**
         * @brief Set the boundary data used for ghost states.
         * @param boundaries Boundary data.
         */
        void set_boundaries(const BoundaryData & boundaries);

        /**
         * @brief Get the face reconstruction type.
         * @return Face reconstruction type.
         */
        FaceReconstructionType get_type() const { return type; }

        /**
         * @brief Get the number of quadrature points per face.
         * @return Number of quadrature points per face.
         */
        virtual uint8_t n_face_quadrature_points() const = 0;

        /**
         * @brief Reconstruct the face values.
         * @param solution Cell states W = [rho, u_x, u_y, p].
         * @param face_solution Face states W, indexed (face, quadrature point, side, variable).
         *                      Side 1 of boundary faces is left untouched.
         */
        virtual void calc_face_values(Kokkos::View<rtype *[N_CONSERVATIVE]> solution,
                                      Kokkos::View<rtype **[2][N_CONSERVATIVE]> face_solution) = 0;

        /**
         * @brief Cells below n_owned whose reconstruction reads no cell at or
         *        above n_owned, so that calc_cell_face_values can reconstruct
         *        them before the halo is filled. Empty if this reconstruction
         *        cannot be split by cells.
         */
        virtual std::vector<uint32_t> cells_independent_of_halo(uint32_t n_owned) const;

        /**
         * @brief Whether distributed runs reconstruct owned cells only and get
         *        the halo side of each partition face from the rank owning it
         *        (FaceExchange), rather than reconstructing halo layer 1 too.
         *        Requires cells_independent_of_halo() and calc_cell_face_values().
         */
        virtual bool exchanges_partition_faces() const { return false; }

        uint64_t n_block_cells = 0;   // reconstructed cells on the block path (this rank)
        uint32_t n_block_shapes = 0;  // their block shapes (this rank)

        /**
         * @brief Reconstruct the listed cells' sides of their faces, on the
         *        given execution space instance, as far as each cell can on its
         *        own (see cells_independent_of_halo()). Once every reconstructed
         *        cell went through this and those instances are fenced,
         *        finish_cell_face_values() completes the face values.
         * @param exec Execution space instance to launch on.
         * @param solution Cell states W = [rho, u_x, u_y, p].
         * @param face_solution Face states W, as in calc_face_values.
         * @param cells Local cells to reconstruct.
         */
        virtual void calc_cell_face_values(const Kokkos::DefaultExecutionSpace & exec,
                                           Kokkos::View<rtype *[N_CONSERVATIVE]> solution,
                                           Kokkos::View<rtype **[2][N_CONSERVATIVE]> face_solution,
                                           const IndexSet & cells);

        /** @brief Complete the face values after calc_cell_face_values(). */
        virtual void finish_cell_face_values(Kokkos::View<rtype *[N_CONSERVATIVE]> solution,
                                             Kokkos::View<rtype **[2][N_CONSERVATIVE]> face_solution);

        /**
         * @brief Drop what calc_cell_face_values() left for
         *        finish_cell_face_values(), on the default instance, when the
         *        face values will not be completed.
         */
        virtual void discard_cell_face_values() {}

        /**
         * @brief Bytes of the work memory needed only while face values are reconstructed (calc_face_values,
         *        or from calc_cell_face_values to finish_cell_face_values); 0 if none.
         */
        virtual size_t scratch_bytes() const { return 0; }

        /**
         * @brief Keep that work memory in storage that other work uses at other times (memory::ScratchArena),
         *        instead of the reconstruction's own allocation.
         * @param storage scratch_bytes() bytes, aligned for rtype, that outlive the reconstruction's use.
         */
        virtual void place_scratch(void * /*storage*/) {}

        /**
         * @brief Turn the velocity of the cell 1 sides of faces across
         *        rotational periodic boundaries, which the schemes reconstruct
         *        in cell 1's frame, into the face's (its cell 0's) frame, so
         *        that the flux sees both sides in one frame. Call once the face
         *        values are complete; nothing happens without rotational pairs.
         */
        void rotate_periodic_faces(Kokkos::View<rtype **[2][N_CONSERVATIVE]> face_solution);

        /**
         * @brief Exterior states of transmissive boundary faces without an
         *        image face, into their side 1: W at each quadrature point
         *        moved onto the image cell (BoundaryData::face_image_offset),
         *        from the image cell's reconstruction polynomial (point_values),
         *        or the image cell's average where the scheme has none. Call
         *        once the face values are complete.
         * @param solution Cell states W, as in calc_face_values.
         */
        void image_point_values(Kokkos::View<rtype *[N_CONSERVATIVE]> solution,
                                Kokkos::View<rtype **[2][N_CONSERVATIVE]> face_solution);

        /**
         * @brief Halo cells whose polynomials image_point_values() reads: the
         *        image cells of owned cells' transmissive faces without an image
         *        face (sorted, distinct).
         */
        std::vector<uint32_t> halo_image_cells() const;

        /**
         * @brief Gradients of W = [rho, u, p] at the centroids of the first n_cells
         *        cells from the reconstruction polynomials, if the scheme has
         *        higher-order ones than least squares on face neighbors.
         * @return Whether gradients were written.
         */
        virtual bool cell_gradients(Kokkos::View<rtype *[N_CONSERVATIVE]> /*solution*/,
                                    Kokkos::View<rtype *[N_CONSERVATIVE][N_DIM]> /*gradients*/,
                                    uint32_t /*n_cells*/) {
            return false;
        }
        
        /**
         * @brief Values of W = [rho, u, p] at points inside cells from the
         *        reconstruction polynomials: out(i) = W of cells(i / m) at
         *        offset offsets(i) from its centroid, with m =
         *        offsets.extent(0) / cells.extent(0) points per cell, for cells
         *        the scheme deems smooth and where the value is admissible;
         *        out(i) of other points is left as it is. by_cell: equal
         *        cells come in runs, whose polynomials may then be fitted once.
         * @return Whether the scheme has polynomials of higher order than
         *         least squares on face neighbors (nothing written otherwise).
         */
        virtual bool point_values(Kokkos::View<rtype *[N_CONSERVATIVE]> /*solution*/,
                                  Kokkos::View<uint32_t *> /*cells*/, Kokkos::View<rtype *[N_DIM]> /*offsets*/,
                                  Kokkos::View<rtype *[N_CONSERVATIVE]> /*out*/, bool /*by_cell*/) {
            return false;
        }

        /**
         * @brief Axisymmetric runs: for each of the first n_cells cells whose
         *        reconstruction has a smooth polynomial, replace source(c) by
         *        int (p - tau_thetatheta) dA over the planar cell, with p and the
         *        velocity of the polynomial at cell quadrature points and the cell
         *        viscosity mu(c) (inviscid if mu is empty). Other cells keep
         *        their source, which the caller set from cell values.
         */
        virtual void axisymmetric_source(Kokkos::View<rtype *[N_CONSERVATIVE]> /*solution*/,
                                         Kokkos::View<rtype *, Kokkos::LayoutStride> /*mu*/,
                                         Kokkos::View<rtype *> /*source*/, uint32_t /*n_cells*/) {}

        /**
         * @brief Set up per-face quadrature (3D): Dunavant rules on triangles and
         *        Gauss rules mapped bilinearly onto quadrilaterals, exact for
         *        polynomials of the given degree on planar faces (degree <= 1
         *        uses the face centroid). Also maps each quadrature point of a
         *        transmissive boundary face to the matching point of its image
         *        face (BoundaryData::face_image_quad).
         * @param degree Polynomial degree to integrate exactly.
         */
        void init_face_quadrature_3d(uint8_t degree);

        /**
         * @brief Curved meshes in 2D: per-face quadrature points, weights and
         *        normals for quadrature_face's Gauss rule on every face, the
         *        curved faces' mapped onto their curves (nothing otherwise).
         */
        void init_face_quadrature_2d_curved();

        /**
         * @brief 3D, and 2D curved meshes: the faces' quadrature points (face,
         *        q, dim) on the host, the stored ones or, on planar faces whose
         *        points were released, placed from the nodes as the kernels do.
         */
        Kokkos::View<rtype ***>::host_mirror_type host_face_points() const;
        /** @brief Node k of face f as in a FaceNodes table, on the host. */
        uint32_t host_face_node(uint32_t f, uint32_t k) const;
        /** @brief 3D: the nodes of face f, as planar_face reads them, on the host. */
        PlanarFace host_planar_face(uint32_t f) const;

        Quadrature quadrature_face;
        // 3D, and 2D curved meshes: (face, q, dim) and (face, q). Weights sum to 2 per face
        // (twice the area over face_area on curved faces) and are zero on padding points
        Kokkos::View<rtype ***> face_quad_points;
        FaceWeights<> face_quad_weights;
        Kokkos::View<rtype ***> face_quad_normals;  // curved meshes: (face, q, dim) unit normals out of cell 0
        // 3D: reference coordinates of point q on planar faces, (s, t) on quadrilaterals and (a, b) on triangles,
        // from which planar_face_point places it
        Kokkos::View<rtype *[4]> face_rule;
        bool share_weights = true;  // false keeps a row of face_quad_weights per face (tests)
    protected:
        FaceReconstructionType type;
        std::shared_ptr<Mesh> mesh;
        BoundaryData boundaries;
    private:
        Kokkos::View<uint32_t *> rotated_faces;  // faces whose shift row rotates
        bool rotated_faces_found = false;
        // image_point_values: boundary faces and their image cells; offsets
        // from that cell's centroid and values of each face's quadrature points
        Kokkos::View<uint32_t *> image_point_faces;
        Kokkos::View<uint32_t *> image_point_cells;
        Kokkos::View<rtype *[N_DIM]> image_point_offsets;
        Kokkos::View<rtype *[N_CONSERVATIVE]> image_point_W;
        bool image_points_found = false;
};

class FirstOrder : public FaceReconstruction {
    public:
        /**
         * @brief Construct a new First Order object
         */
        FirstOrder();

        /**
         * @brief Destroy the First Order object
         */
        ~FirstOrder();

        /**
         * @brief Initialize the first order face reconstruction.
         */
        void init(const Input & input) override;

        /**
         * @brief Get the number of quadrature points per face.
         * @return Number of quadrature points per face.
         */
        uint8_t n_face_quadrature_points() const override;

        /**
         * @brief Reconstruct the face values.
         * @param solution Cell states W = [rho, u_x, u_y, p].
         * @param face_solution Face states W, indexed (face, quadrature point, side, variable).
         *                      Side 1 of boundary faces is left untouched.
         */
        void calc_face_values(Kokkos::View<rtype *[N_CONSERVATIVE]> solution,
                              Kokkos::View<rtype **[2][N_CONSERVATIVE]> face_solution) override;
    protected:
    private:
};

enum class LimiterType {
    NONE,
    BARTH_JESPERSEN,
    VENKATAKRISHNAN,
};

static const std::unordered_map<std::string, LimiterType> LIMITER_TYPES = {
    {"none", LimiterType::NONE},
    {"barth_jespersen", LimiterType::BARTH_JESPERSEN},
    {"venkatakrishnan", LimiterType::VENKATAKRISHNAN},
};

static const std::unordered_map<LimiterType, std::string> LIMITER_NAMES = {
    {LimiterType::NONE, "none"},
    {LimiterType::BARTH_JESPERSEN, "barth_jespersen"},
    {LimiterType::VENKATAKRISHNAN, "venkatakrishnan"},
};

/**
 * @brief The geometry of MUSCL's face-neighbor stencils, for the reconstructed
 *        cells that take the face-neighbor fit (not tetrahedra, no face of a
 *        characteristic boundary), have no rotational periodic face and have
 *        all their faces (not the outermost halo layer): per face slot (in
 *        faces_of_cell order), the gradient weights (LSQGradientFunctor::weights)
 *        and the offset from the cell centroid to the face centroid. The
 *        cells whose rows are equal on teno::TableStore's grid share the row
 *        of the lowest global id, agreed over the ranks, so that a cell's
 *        row does not depend on the partition. Boundary faces' states still
 *        come from LSQGradientFunctor::neighbor. The cells of structured
 *        blocks whose faces all have a neighbor take their neighbors as the
 *        cell plus their block shape's offsets (one per face slot).
 */
struct MUSCLTables {
    static constexpr uint32_t NONE = ~0u;
    Kokkos::View<uint32_t *[2]> row;  // (cell): its weights row and offsets row, or NONE; empty if no cell has one
    Kokkos::View<rtype *> weights;    // ((row stride + slot) N_DIM + d)
    Kokkos::View<rtype *> offsets;    // ((row stride + slot) N_DIM + d)
    uint32_t stride = 0;
    teno::BlockStencils block;
    Kokkos::View<uint32_t *> offsets_faces_of_cell;
    Kokkos::View<uint32_t *> faces_of_cell;
    Kokkos::View<int32_t *[2]> cells_of_face;

    KOKKOS_INLINE_FUNCTION
    bool has(const uint32_t c) const { return row.extent(0) && row(c, 0) != NONE; }

    /** @brief Cell c's block offsets, or null if it reads its faces. */
    KOKKOS_INLINE_FUNCTION
    const int32_t * block_offsets(const uint32_t c) const {
        const uint32_t k = block.shape_of(c);
        return k == teno::BlockStencils::NONE ? nullptr : block.offsets_of(k);
    }

    /** @brief The neighbor of cell c across its face slot k, by offsets o or by its face; -1 across a boundary. */
    KOKKOS_INLINE_FUNCTION
    int32_t neighbor(const uint32_t c, const uint32_t k, const int32_t * o) const {
        if (o) return int32_t(c) + o[k];
        const uint32_t f = faces_of_cell(offsets_faces_of_cell(c) + k);
        const int32_t c0 = cells_of_face(f, 0);
        return c0 == int32_t(c) ? cells_of_face(f, 1) : c0;
    }

    /** @brief Cell c's weights, (slot N_DIM + d). */
    KOKKOS_INLINE_FUNCTION
    const rtype * weights_of(const uint32_t c) const { return weights.data() + size_t(row(c, 0)) * stride * N_DIM; }

    /** @brief Cell c's face-centroid offsets, (slot N_DIM + d). */
    KOKKOS_INLINE_FUNCTION
    const rtype * offsets_of(const uint32_t c) const { return offsets.data() + size_t(row(c, 1)) * stride * N_DIM; }
};

/**
 * @brief Second-order MUSCL reconstruction of W = [rho, u_x, u_y, p] using
 *        least-squares gradients and a slope limiter.
 */
class MUSCL : public FaceReconstruction {
    public:
        MUSCL();
        ~MUSCL();
        void init(const Input & input) override;
        report::Summary summary() const override;
        uint8_t n_face_quadrature_points() const override;
        void calc_face_values(Kokkos::View<rtype *[N_CONSERVATIVE]> solution,
                              Kokkos::View<rtype **[2][N_CONSERVATIVE]> face_solution) override;
        std::vector<uint32_t> cells_independent_of_halo(uint32_t n_owned) const override;
        void calc_cell_face_values(const Kokkos::DefaultExecutionSpace & exec,
                                   Kokkos::View<rtype *[N_CONSERVATIVE]> solution,
                                   Kokkos::View<rtype **[2][N_CONSERVATIVE]> face_solution,
                                   const IndexSet & cells) override;
        void finish_cell_face_values(Kokkos::View<rtype *[N_CONSERVATIVE]> solution,
                                     Kokkos::View<rtype **[2][N_CONSERVATIVE]> face_solution) override;

        LimiterType limiter = LimiterType::VENKATAKRISHNAN;
        rtype venkat_K = 5.0;
        Kokkos::View<rtype *[N_CONSERVATIVE][N_DIM]> gradients;
        Kokkos::View<rtype *[N_CONSERVATIVE]> limiters;
        LSQVertexGradientFunctor gradient;  // Its linear vertex fit serves tetrahedra
        // The cells of structured blocks read their neighbors by offsets (MUSCLTables::block);
        // false (tests only) reads every cell's faces
        bool block_path = true;
        MUSCLTables tables;
        // Cells that could have a row per distinct row, over the ranks, below which no cell gets one: the rows
        // would cost more memory than the per-step fit
        static constexpr uint64_t MIN_CELLS_PER_ROW = 8;
        uint64_t n_table_cells = 0;           // cells with a row (this rank)
        uint64_t n_candidates_global = 0;     // owned cells that could have a row, over the ranks
        uint64_t n_rows_global = 0;           // distinct rows over the ranks (weights or offsets, the more)
        /** @brief The tables of the cells that have one (MUSCLTables), and the block shapes (collective). */
        void find_tables();
        // Measured on A100s: one fused kernel per cell is faster from about 2^17 cells, separate kernels (more
        // parallel, shorter chains) below. Tests lower it to run the fused kernel on small meshes
        static inline uint32_t fused_cells = 1u << 17;
};

/**
 * @brief High-order TENO-E reconstruction on unstructured meshes
 *        (Liang, Shyy & Fu, J. Sci. Comput. 104:1, 2025; see docs/numerics/teno_e.md).
 *
 * Each cell carries one large central stencil (degree r) and one small
 * degree-2 sector stencil per face. Cells flagged smooth by a density-based
 * indicator use the large-stencil polynomial directly on the conservative
 * variables; troubled cells perform TENO stencil selection on characteristic
 * variables per face. Works on triangles and quadrilaterals.
 */
class TENO : public FaceReconstruction {
    public:
        TENO();
        ~TENO();
        void init(const Input & input) override;
        report::Summary summary() const override;
        uint8_t n_face_quadrature_points() const override;
        void calc_face_values(Kokkos::View<rtype *[N_CONSERVATIVE]> solution,
                              Kokkos::View<rtype **[2][N_CONSERVATIVE]> face_solution) override;
        std::vector<uint32_t> cells_independent_of_halo(uint32_t n_owned) const override;
        void calc_cell_face_values(const Kokkos::DefaultExecutionSpace & exec,
                                   Kokkos::View<rtype *[N_CONSERVATIVE]> solution,
                                   Kokkos::View<rtype **[2][N_CONSERVATIVE]> face_solution,
                                   const IndexSet & cells) override;
        void finish_cell_face_values(Kokkos::View<rtype *[N_CONSERVATIVE]> solution,
                                     Kokkos::View<rtype **[2][N_CONSERVATIVE]> face_solution) override;
        bool exchanges_partition_faces() const override { return true; }
        void discard_cell_face_values() override;
        size_t scratch_bytes() const override;
        void place_scratch(void * storage) override;
        bool cell_gradients(Kokkos::View<rtype *[N_CONSERVATIVE]> solution,
                            Kokkos::View<rtype *[N_CONSERVATIVE][N_DIM]> gradients, uint32_t n_cells) override;
        void axisymmetric_source(Kokkos::View<rtype *[N_CONSERVATIVE]> solution,
                                 Kokkos::View<rtype *, Kokkos::LayoutStride> mu, Kokkos::View<rtype *> source,
                                 uint32_t n_cells) override;
        bool point_values(Kokkos::View<rtype *[N_CONSERVATIVE]> solution, Kokkos::View<uint32_t *> cells,
                          Kokkos::View<rtype *[N_DIM]> offsets, Kokkos::View<rtype *[N_CONSERVATIVE]> out,
                          bool by_cell) override;

        uint8_t degree = 4;
        uint8_t n_dof_large = 0;
        uint16_t n_stencil_large = 0;
        uint16_t n_stencil_small = 10;
        rtype stencil_factor = 2.0;
        rtype sigma_threshold = 1.0e-3_r;
        rtype sigma_upper = 1.0e-2_r;
        rtype C_T = -1.0;  // Fixed cutoff; negative selects the adaptive cutoff
        bool characteristic = true;
        bool bound_preserving = false;
        // Stencils take mirror images across curved wall and symmetry faces too
        // (across their planes); by default they stay one-sided there
        bool curved_mirrors = false;
        // Largest degree of the central polynomials of cells whose stencils reach a
        // curved wall without mirror images: one-sided fits of higher degree grow there
        uint8_t curved_wall_degree = 3;
        rtype max_condition = 1.0e8;
        uint8_t slice_shift = teno::SLICE_SHIFT;  // log2 of the cells per slice of the packed stencils
        // Pseudo-inverses stored in single precision (opt-in: halves the
        // largest tables, and changes results at single-precision round-off)
        bool single_tables = false;
        // The cells of a stencil shape share one table of pseudo-inverses, and the
        // cells of a cell shape one row of basis_mean and si_matrix or moments
        // (teno::TableStore); false (tests only) keeps a table and a row per cell
        bool share_tables = true;
        // This rank's cells whose tables the last full setup fitted; with shared tables the
        // others take those of their stencil shape's cell with the lowest global id
        uint32_t n_cells_fitted = 0;
        // false: the central stencils alone, which cell_gradients and point_values read, and no
        // face values (output of states read from files); such tables are never written to the cache
        bool sector_stencils = true;
        // false, set before init() by a caller that places the troubled passes' work memory (place_scratch):
        // init() leaves it without memory
        bool own_scratch = true;
        // Set before init() when distributed runs get the halo side of partition faces from their owners
        // (Solver::exchanges_partition_faces): the tables then cover only the owned cells and the halo cells
        // of halo_image_cells(), which are all whose reconstruction is evaluated
        bool partition_faces_exchanged = false;
        // Rows of the per-cell tables: cells [0, n_direct_rows) in order, then extra_cells (halo cells, sorted)
        uint32_t n_direct_rows = 0;
        std::vector<uint32_t> extra_cells;
        Kokkos::View<uint32_t *> extra_cells_device;
        /** @brief Rows of the per-cell tables. */
        uint32_t n_rows() const { return n_direct_rows + static_cast<uint32_t>(extra_cells.size()); }
        /** @brief The local cell of table row r. */
        uint32_t cell_of_row(const uint32_t r) const { return r < n_direct_rows ? r : extra_cells[r - n_direct_rows]; }
        /** @brief The table row of local cell c, or -1 if it has none. */
        int64_t row_of_cell(uint32_t c) const;
        // The cells of structured blocks with a block shape read their central stencils by offsets
        // (teno::BlockStencils); false (tests only) reads every cell's slots
        bool block_path = true;
        teno::BlockStencils block_stencils;
        /**
         * @brief Block shapes of the cells of the structured blocks, from their packed central stencils
         *        (after the tables are set up or rebuilt; a no-op unless block_path).
         */
        void find_block_stencils();
        /** @brief Each table row's global id, or null if the mesh is not distributed (its row then). */
        const uint64_t * global_cell_ids() const {
            if (mesh->h_global_cell_id.empty()) return nullptr;
            return extra_cells.empty() ? mesh->h_global_cell_id.data() : row_global_ids.data();
        }
        // When allocated (cell, var, d), the reconstruction also writes there the gradients of W that
        // cell_gradients gives, for the cells it finds smooth (sigma < sigma_threshold)
        Kokkos::View<rtype *[N_CONSERVATIVE][N_DIM]> smooth_gradients;
        // When allocated, the reconstruction also writes there the gradients of u and T of the central
        // polynomials of the cells it finds smooth at their face points, and marks the sides of the troubled
        // cells and of those whose face values fall back to the cell mean (inadmissible at a face point)
        FacePointGradients point_gradients;

        // Per-cell precomputed data, by table row (n_rows(); cell_of_row)
        Kokkos::View<rtype *> scale;                       // h = sqrt(V)
        // Rotational periodicity: (turns + teno::TURN_BIAS, i, j) the rotation
        // by turns times the periodic angle, and the boundary faces in
        // increasing order, by boundary ordinal (empty otherwise)
        Kokkos::View<rtype *[N_DIM][N_DIM]> entry_rotations;
        Kokkos::View<int32_t *> boundary_faces;
        // (cell): its cell shape, the row of basis_mean and si_matrix or moments it reads
        Kokkos::View<uint32_t *> cell_shape;
        std::vector<uint64_t> cell_shape_owner;            // (cell shape): global id of the cell whose row it is
        Kokkos::View<rtype **> basis_mean;                 // (cell shape, l): mean of phi_l over the cell
        Kokkos::View<uint16_t *> stencil_large_size;       // (cell)
        teno::PackedStencils stencil_large;                // pseudo-inverse width: nk
        Kokkos::View<uint16_t **> stencil_small_size;      // (cell, face); 0 if stencil invalid
        teno::PackedStencils stencil_small;                // the faces' stencils one after another; width NK_SMALL
        Kokkos::View<rtype **> si_matrix;                  // 2D: (cell shape, upper_index(l, m)): symmetric
        teno::Moments moments;                             // 3D: (cell shape, k), from which the troubled passes form it
        teno::SmoothnessTerms si_terms;                    // 3D
        Kokkos::View<rtype *> troubled;                    // (local cell): sigma, for diagnostics
        Kokkos::View<rtype ***> troubled_coeffs;           // (queue slot, l, var): scratch for the troubled passes
        Kokkos::View<rtype ****> troubled_small_coeffs;    // (queue slot, sector, l, var): scratch for the troubled passes
        // Queued cells per round of the troubled passes, as a fraction of the
        // reconstructed cells (scratch slots); later rounds recompute the
        // central coefficients the smooth pass could not keep
        rtype troubled_capacity = N_DIM == 3 ? 0.25 : 1.0;
        Kokkos::View<uint32_t *> troubled_cells;           // queue of troubled cells
        Kokkos::View<uint32_t> n_troubled;
        // Vertex-neighbor layers each table row's stencil search visited (host);
        // a distributed run needs this many complete layers around the row's cell
        std::vector<uint8_t> gather_depth;
        // Axisymmetric runs: rule on the reference triangle (q, [xi, eta, w]) of the geometric source,
        // and the curved cells' own rules (q, [x, y, w]), cell c's from curved_rule_start(c)
        Kokkos::View<rtype *[3]> cell_rule;
        Kokkos::View<uint32_t *> curved_rule_start;
        Kokkos::View<rtype *[3]> curved_rule;
        void init_curved_source_rules();
        // The faces' nodes, from which the kernels place the face points of
        // straight (2D) and planar (3D, meshes without curved faces) faces, and
        // (axisymmetric runs) the cells' nodes; the mesh's own views are
        // released after setup
        FaceNodes face_nodes;
        Kokkos::View<uint32_t *> offsets_nodes_of_cell, nodes_of_cell;
        NodeCoords node_coords;
        // Device memory of the packed stencils when they were streamed to the device (CUDA)
        std::vector<std::shared_ptr<GrowingBuffer>> table_memory;

        /**
         * @brief Stencil cells and mirror faces of every reconstructed cell, row
         *        by row (host copy, for diagnostics).
         */
        struct Stencils {
            std::vector<uint64_t> offsets;  // (cell + 1)
            std::vector<int32_t> cells;
            std::vector<int32_t> faces;
        };
        Stencils large_stencils() const;

        /** @brief One reconstructed cell's stencils, as the reconstruction reads them (host copy). */
        struct CellStencils {
            std::vector<int32_t> cells, faces;  // the central stencil's slots, then each face's sector stencil's
            uint16_t central = 0;               // slots of the central stencil
            std::array<uint16_t, teno::MAX_FACES> sectors{};  // slots of each face's sector stencil, 0 if left out
            std::vector<rtype> pinv;  // with pinv: n_dof_large entries per central slot, then NK_SMALL per sector slot
        };

        /**
         * @brief Calls visit(c, stencils of c) for every cell c with tables, in
         *        the order of their rows, copying the tables to the host chunk by chunk.
         */
        void for_each_stencil(const std::function<void(uint32_t, const CellStencils &)> & visit, bool pinv) const;

        /** @brief Local cells that some stencil (central or sector) of the cells with tables flagged in cells reads. */
        std::vector<uint8_t> stencil_cells(const std::vector<uint8_t> & cells) const;

        /**
         * @brief Metric (3 x 3, row-major) in which the stencil candidates of
         *        cell i are ranked by distance: the identity unless the
         *        mesh spacing around the cell is markedly anisotropic.
         */
        std::array<double, 9> stencil_metric(uint32_t i) const;

        /**
         * @brief Halo layers recorded in this rank's cache file (0 if there is
         *        none, or it was made with other TENO options or rank count),
         *        so a distributed run can build its halo at that depth at once.
         */
        static uint8_t cached_halo_layers(const Input & input);

        /**
         * @brief Write the precomputed data to the cache file, unless init()
         *        loaded it from there. A distributed run calls this once its halo
         *        is final, with its halo layers.
         */
        void save_cache(uint8_t halo_layers = 0);

        /**
         * @brief Recompute the tables of the given cells with tables and keep
         *        every other cell's, after the mesh geometry changed (node
         *        coordinates moved, geometry recomputed and on the device).
         *        Each cell's tables are a function of the mesh alone, so with
         *        the cells of cells_within_reach() the result is bitwise that
         *        of a full setup. Collective when tables are shared.
         */
        void rebuild_cells(const std::vector<uint32_t> & cells);

        /**
         * @brief Cells with tables whose stencil searches reach any of the
         *        given cells (within their gather_depth vertex-neighbor layers),
         *        i.e. whose tables depend on those cells' geometry.
         */
        std::vector<uint32_t> cells_within_reach(const std::vector<uint32_t> & changed) const;

        // Setup: on the host only (tests), the cells per device batch (0: from the device memory), and
        // cells refitted after taking their stencil shape's tables (tests)
        bool setup_on_host = false;
        uint32_t setup_batch_cells = 0;
        std::vector<uint32_t> setup_refit_cells;

        // Gas mixtures: the flow block is reconstructed in primitive variables
        // W with characteristic projections on the primitive system, whose
        // sound speed comes from each cell's frozen gamma; troubled cells are
        // also found by jumps of the mixture molar mass; and the stencil
        // choice of the entropy field on each face side (SELECT_LARGE or a
        // bitmask of the kept sector stencils) is left for the species
        static constexpr uint8_t SELECT_LARGE = 0xFF;
        bool primitive = false;
        Kokkos::View<rtype *, Kokkos::LayoutStride> cell_gamma;  // (cell)
        Kokkos::View<rtype *> cell_molar_mass;                   // (cell)
        Kokkos::View<uint8_t **> selection;                      // (face, side)

        /** @brief Switch to the mixture variant with these per-cell views (filled every stage). */
        void set_mixture(Kokkos::View<rtype *, Kokkos::LayoutStride> gamma, Kokkos::View<rtype *> molar_mass);

    private:
        template <uint8_t DEG, bool PRIM, typename TABLE>
        void launch_source(Kokkos::View<rtype *[N_CONSERVATIVE]> solution, Kokkos::View<rtype *, Kokkos::LayoutStride> mu,
                           Kokkos::View<rtype *> source, uint32_t n_cells);
        template <uint8_t DEG, bool PRIM, typename TABLE>
        void launch_points(Kokkos::View<rtype *[N_CONSERVATIVE]> solution, Kokkos::View<uint32_t *> cells,
                           Kokkos::View<rtype *[N_DIM]> offsets, Kokkos::View<rtype *[N_CONSERVATIVE]> out,
                           bool by_cell);
        template <uint8_t DEG, bool PRIM, typename TABLE>
        void launch_gradients(Kokkos::View<rtype *[N_CONSERVATIVE]> solution,
                              Kokkos::View<rtype *[N_CONSERVATIVE][N_DIM]> gradients, uint32_t n_cells);
        template <uint8_t DEG, bool PRIM, typename TABLE>
        void launch_reconstruction(const Kokkos::DefaultExecutionSpace & exec,
                                   Kokkos::View<rtype *[N_CONSERVATIVE]> solution,
                                   Kokkos::View<rtype **[2][N_CONSERVATIVE]> face_solution,
                                   const IndexSet & cells, bool troubled_pass);
        void dispatch(const Kokkos::DefaultExecutionSpace & exec, Kokkos::View<rtype *[N_CONSERVATIVE]> solution,
                      Kokkos::View<rtype **[2][N_CONSERVATIVE]> face_solution, const IndexSet & cells,
                      bool troubled_pass);

        void read_options(const Input & input);
        void compute_stencils_and_matrices(const std::vector<uint32_t> * subset = nullptr);
        /** @brief 3D meshes without curved faces: frees the face points, which only the setup reads. */
        void release_planar_face_points();
        void allocate_scratch();
        uint64_t options_key() const;
        uint64_t cache_key() const;
        bool load_cache();
        void round_to_single();

        void choose_rows();

        std::vector<uint64_t> row_global_ids;  // with extra_cells: global_cell_ids()
        std::string cache_file;  // this rank's
        bool cache_single = false;  // pseudo-inverses and smoothness-indicator matrices (3D: moments) cached in single precision
        bool cache_loaded = false;

        uint32_t largest_stencil = 0;        // Largest central stencil on any rank (cells)
        int64_t n_sector_unavailable = -1;   // Small sector stencils cut by boundaries, -1 if unknown
        std::string cache_status;          // Stencil cache outcome, empty without a cache file
        std::string setup_status;          // Where and how fast the tables were set up
        std::string setup_producer = "host";  // Execution space that set up the tables, or "host"
};

#endif // FACE_RECONSTRUCTION_H
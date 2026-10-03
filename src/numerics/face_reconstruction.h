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

#include <memory>
#include <unordered_map>
#include <vector>

#include <toml.hpp>

#include "common_typedef.h"
#include "mesh.h"
#include "quadrature.h"
#include "boundary.h"
#include "gradient.h"
#include "log.h"
#include "teno.h"

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
        virtual void init(const toml::value & input) = 0;

        /**
         * @brief Display lines for the run log.
         */
        virtual logging::Items summary() const;

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
         * @brief Reconstruct the listed cells' sides of their faces, on the
         *        given execution space instance, as far as each cell can on its
         *        own (see cells_independent_of_halo()). Once every reconstructed
         *        cell went through this and those instances are fenced,
         *        finish_cell_face_values() completes the face values.
         * @param exec Execution space instance to launch on.
         * @param solution Cell states W = [rho, u_x, u_y, p].
         * @param face_solution Face states W, as in calc_face_values.
         * @param cells Local cells to reconstruct (not empty).
         */
        virtual void calc_cell_face_values(const Kokkos::DefaultExecutionSpace & exec,
                                           Kokkos::View<rtype *[N_CONSERVATIVE]> solution,
                                           Kokkos::View<rtype **[2][N_CONSERVATIVE]> face_solution,
                                           Kokkos::View<uint32_t *> cells);

        /** @brief Complete the face values after calc_cell_face_values(). */
        virtual void finish_cell_face_values(Kokkos::View<rtype *[N_CONSERVATIVE]> solution,
                                             Kokkos::View<rtype **[2][N_CONSERVATIVE]> face_solution);

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
         * @brief Set up per-face quadrature (3D): Dunavant rules on triangles and
         *        Gauss rules mapped bilinearly onto quadrilaterals, exact for
         *        polynomials of the given degree on planar faces (degree <= 1
         *        uses the face centroid). Also maps each quadrature point of a
         *        transmissive boundary face to the matching point of its image
         *        face (BoundaryData::face_image_quad).
         * @param degree Polynomial degree to integrate exactly.
         */
        void init_face_quadrature_3d(uint8_t degree);

        Quadrature quadrature_face;
        Kokkos::View<rtype ***> face_quad_points;   // 3D: (face, q, dim)
        Kokkos::View<rtype **> face_quad_weights;   // 3D: (face, q), sum 2 per face, zero on padding
    protected:
        FaceReconstructionType type;
        std::shared_ptr<Mesh> mesh;
        BoundaryData boundaries;
    private:
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
        void init(const toml::value & input) override;

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
 * @brief Second-order MUSCL reconstruction of W = [rho, u_x, u_y, p] using
 *        least-squares gradients and a slope limiter.
 */
class MUSCL : public FaceReconstruction {
    public:
        MUSCL();
        ~MUSCL();
        void init(const toml::value & input) override;
        logging::Items summary() const override;
        uint8_t n_face_quadrature_points() const override;
        void calc_face_values(Kokkos::View<rtype *[N_CONSERVATIVE]> solution,
                              Kokkos::View<rtype **[2][N_CONSERVATIVE]> face_solution) override;

        LimiterType limiter = LimiterType::VENKATAKRISHNAN;
        rtype venkat_K = 5.0;
        Kokkos::View<rtype *[N_CONSERVATIVE][N_DIM]> gradients;
        Kokkos::View<rtype *[N_CONSERVATIVE]> limiters;
        LSQVertexGradientFunctor gradient;  // Its linear vertex fit serves tetrahedra
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
        void init(const toml::value & input) override;
        logging::Items summary() const override;
        uint8_t n_face_quadrature_points() const override;
        void calc_face_values(Kokkos::View<rtype *[N_CONSERVATIVE]> solution,
                              Kokkos::View<rtype **[2][N_CONSERVATIVE]> face_solution) override;
        std::vector<uint32_t> cells_independent_of_halo(uint32_t n_owned) const override;
        void calc_cell_face_values(const Kokkos::DefaultExecutionSpace & exec,
                                   Kokkos::View<rtype *[N_CONSERVATIVE]> solution,
                                   Kokkos::View<rtype **[2][N_CONSERVATIVE]> face_solution,
                                   Kokkos::View<uint32_t *> cells) override;
        void finish_cell_face_values(Kokkos::View<rtype *[N_CONSERVATIVE]> solution,
                                     Kokkos::View<rtype **[2][N_CONSERVATIVE]> face_solution) override;
        bool cell_gradients(Kokkos::View<rtype *[N_CONSERVATIVE]> solution,
                            Kokkos::View<rtype *[N_CONSERVATIVE][N_DIM]> gradients, uint32_t n_cells) override;

        uint8_t degree = 4;
        uint8_t n_dof_large = 0;
        uint16_t n_stencil_large = 0;
        uint16_t n_stencil_small = 10;
        rtype stencil_factor = 2.0;
        rtype sigma_threshold = 1.0e-3;
        rtype sigma_upper = 1.0e-2;
        rtype C_T = -1.0;  // Fixed cutoff; negative selects the adaptive cutoff
        bool characteristic = true;
        bool bound_preserving = false;
        rtype max_condition = 1.0e8;
        uint8_t slice_shift = teno::SLICE_SHIFT;  // log2 of the cells per slice of the packed stencils

        // Per-cell precomputed data, for the reconstructed cells [0, n_reconstructed)
        Kokkos::View<rtype *> scale;                       // h = sqrt(V)
        Kokkos::View<rtype **> basis_mean;                 // (cell, l): mean of phi_l over the cell
        Kokkos::View<uint16_t *> stencil_large_size;       // (cell)
        teno::PackedStencils stencil_large;                // pseudo-inverse width: nk
        Kokkos::View<uint16_t **> stencil_small_size;      // (cell, face); 0 if stencil invalid
        teno::PackedStencils stencil_small;                // the faces' stencils one after another; width NK_SMALL
        Kokkos::View<rtype **> si_matrix;                  // (cell, upper_index(l, m)): symmetric
        Kokkos::View<rtype *> troubled;                    // (local cell): sigma, for diagnostics
        Kokkos::View<rtype ***> troubled_coeffs;           // (cell, l, var): scratch for the troubled passes
        Kokkos::View<rtype ****> troubled_small_coeffs;    // (cell, sector, l, var): scratch for the troubled passes
        Kokkos::View<uint32_t *> troubled_cells;           // queue of troubled cells
        Kokkos::View<uint32_t> n_troubled;
        // Vertex-neighbor layers each reconstructed cell's stencil search visited
        // (host); a distributed run needs this many complete layers around the cell
        std::vector<uint8_t> gather_depth;

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

        /**
         * @brief Halo layers recorded in this rank's cache file (0 if there is
         *        none, or it was made with other TENO options or rank count),
         *        so a distributed run can build its halo at that depth at once.
         */
        static uint8_t cached_halo_layers(const toml::value & input);

        /**
         * @brief Write the precomputed data to the cache file, unless init()
         *        loaded it from there. A distributed run calls this once its halo
         *        is final, with its halo layers.
         */
        void save_cache(uint8_t halo_layers = 0);

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
        template <uint8_t DEG, bool PRIM>
        void launch_gradients(Kokkos::View<rtype *[N_CONSERVATIVE]> solution,
                              Kokkos::View<rtype *[N_CONSERVATIVE][N_DIM]> gradients, uint32_t n_cells);
        template <uint8_t DEG, bool PRIM>
        void launch_reconstruction(const Kokkos::DefaultExecutionSpace & exec,
                                   Kokkos::View<rtype *[N_CONSERVATIVE]> solution,
                                   Kokkos::View<rtype **[2][N_CONSERVATIVE]> face_solution,
                                   Kokkos::View<uint32_t *> cells, bool troubled_pass);
        void dispatch(const Kokkos::DefaultExecutionSpace & exec, Kokkos::View<rtype *[N_CONSERVATIVE]> solution,
                      Kokkos::View<rtype **[2][N_CONSERVATIVE]> face_solution, Kokkos::View<uint32_t *> cells,
                      bool troubled_pass);

        void read_options(const toml::value & input);
        void compute_stencils_and_matrices();
        void compute_stencils_and_matrices_3d();
        void allocate_scratch();
        uint64_t options_key() const;
        uint64_t cache_key() const;
        bool load_cache();

        std::string cache_file;  // this rank's
        bool cache_loaded = false;

        uint32_t largest_stencil = 0;        // Largest central stencil on any rank (cells)
        int64_t n_sector_unavailable = -1;   // Small sector stencils cut by boundaries, -1 if unknown
        std::string cache_status;          // Stencil cache outcome, empty without a cache file
};

#endif // FACE_RECONSTRUCTION_H
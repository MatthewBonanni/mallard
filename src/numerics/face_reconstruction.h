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

#include <toml.hpp>

#include "common_typedef.h"
#include "mesh.h"
#include "quadrature.h"
#include "boundary.h"
#include "gradient.h"
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
         * @brief Print the face reconstruction.
         */
        virtual void print() const;

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
        
        Quadrature quadrature_face;
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
        void print() const override;
        uint8_t n_face_quadrature_points() const override;
        void calc_face_values(Kokkos::View<rtype *[N_CONSERVATIVE]> solution,
                              Kokkos::View<rtype **[2][N_CONSERVATIVE]> face_solution) override;

        LimiterType limiter = LimiterType::VENKATAKRISHNAN;
        rtype venkat_K = 5.0;
        Kokkos::View<rtype *[N_CONSERVATIVE][N_DIM]> gradients;
        Kokkos::View<rtype *[N_CONSERVATIVE]> limiters;
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
        void print() const override;
        uint8_t n_face_quadrature_points() const override;
        void calc_face_values(Kokkos::View<rtype *[N_CONSERVATIVE]> solution,
                              Kokkos::View<rtype **[2][N_CONSERVATIVE]> face_solution) override;

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

        // Per-cell precomputed data
        Kokkos::View<rtype *> scale;                       // h = sqrt(V)
        Kokkos::View<rtype **> basis_mean;                 // (cell, l): mean of phi_l over the cell
        Kokkos::View<uint16_t *> stencil_large_size;       // (cell)
        Kokkos::View<int32_t **> stencil_large;            // (cell, s)
        Kokkos::View<int32_t **> stencil_large_face;       // (cell, s): mirror boundary face or -1
        Kokkos::View<rtype ***> pinv_large;                // (cell, l, s)
        Kokkos::View<uint16_t **> stencil_small_size;      // (cell, face); 0 if stencil invalid
        Kokkos::View<int32_t ***> stencil_small;           // (cell, face, s)
        Kokkos::View<int32_t ***> stencil_small_face;      // (cell, face, s): mirror boundary face or -1
        Kokkos::View<rtype ****> pinv_small;               // (cell, face, l, s)
        Kokkos::View<rtype ***> si_matrix;                 // (cell, l, m)
        Kokkos::View<rtype *> troubled;                    // (cell): sigma, for diagnostics
        Kokkos::View<rtype ***> troubled_coeffs;           // (cell, l, var): scratch for the troubled pass
        Kokkos::View<uint32_t *> troubled_cells;           // queue of troubled cells
        Kokkos::View<uint32_t> n_troubled;
        // Vertex-neighbor layers each cell's stencil search visited (host); a
        // distributed run needs this many complete layers around the cell
        std::vector<uint8_t> gather_depth;

    private:
        template <uint8_t DEG>
        void launch_reconstruction(Kokkos::View<rtype *[N_CONSERVATIVE]> solution,
                                   Kokkos::View<rtype **[2][N_CONSERVATIVE]> face_solution);

        void compute_stencils_and_matrices();
        uint64_t cache_key() const;
        void save_cache(const std::string & filename) const;
        bool load_cache(const std::string & filename);
};

#endif // FACE_RECONSTRUCTION_H
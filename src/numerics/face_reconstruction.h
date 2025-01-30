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
#include "basis.h"
#include "quadrature.h"

enum class FaceReconstructionType {
    FirstOrder,
    TENO,
};

static const std::unordered_map<std::string, FaceReconstructionType> FACE_RECONSTRUCTION_TYPES = {
    {"FO", FaceReconstructionType::FirstOrder},
    {"TENO", FaceReconstructionType::TENO},
};

static const std::unordered_map<FaceReconstructionType, std::string> FACE_RECONSTRUCTION_NAMES = {
    {FaceReconstructionType::FirstOrder, "FO"},
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
         * @brief Set the cell conservatives.
         * @param cell_conservatives Pointer to the cell conservatives.
         */
        void set_cell_conservatives(Kokkos::View<rtype *[N_CONSERVATIVE]> * cell_conservatives);

        /**
         * @brief Set the face conservatives.
         * @param face_conservatives Pointer to the face conservatives.
         */
        void set_face_conservatives(Kokkos::View<rtype *[2][N_CONSERVATIVE]> * face_conservatives);

        /**
         * @brief Set the mesh.
         * @param mesh Pointer to the mesh.
         */
        void set_mesh(std::shared_ptr<Mesh> mesh);

        /**
         * @brief Reconstruct the face values.
         * @param solution View of the solution.
         * @param face_solution View of the face solution.
         */
        virtual void calc_face_values(Kokkos::View<rtype *[N_CONSERVATIVE]> solution,
                                      Kokkos::View<rtype *[2][N_CONSERVATIVE]> face_solution) = 0;
    protected:
        FaceReconstructionType type;
        Kokkos::View<rtype *[N_CONSERVATIVE]> * cell_conservatives;
        Kokkos::View<rtype *[2][N_CONSERVATIVE]> * face_conservatives;
        std::shared_ptr<Mesh> mesh;
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
         * @brief Reconstruct the face values.
         * @param solution View of the solution.
         * @param face_solution View of the face solution.
         */
        void calc_face_values(Kokkos::View<rtype *[N_CONSERVATIVE]> solution,
                              Kokkos::View<rtype *[2][N_CONSERVATIVE]> face_solution) override;
    protected:
    private:
};

class TENO : public FaceReconstruction {
    public:
        /**
         * @brief Construct a new TENO object
         */
        TENO();

        /**
         * @brief Destroy the TENO object
         */
        ~TENO();

        /**
         * @brief Initialize the TENO face reconstruction.
         */
        void init(const toml::value & input) override;

        /**
         * @brief Print the TENO face reconstruction.
         */
        void print() const override;

        /**
         * @brief Reconstruct the face values.
         * @param solution View of the solution.
         * @param face_solution View of the face solution.
         */
        void calc_face_values(Kokkos::View<rtype *[N_CONSERVATIVE]> solution,
                              Kokkos::View<rtype *[2][N_CONSERVATIVE]> face_solution) override;
    protected:
    private:
        void calc_max_stencil_size();
        void calc_polynomial_indices();
        std::vector<u_int32_t> compute_stencil_of_cell_centered(u_int32_t i_cell);
        std::vector<std::vector<u_int32_t>> compute_stencils_of_cell_directional(u_int32_t i_cell);
        void compute_stencils_of_cell(u_int32_t i_cell,
                                      std::vector<u_int32_t> & v_offsets_stencil_groups,
                                      std::vector<u_int32_t> & v_offsets_stencils,
                                      std::vector<u_int32_t> & v_stencils);
        void compute_stencils();
        void compute_reconstruction_matrices();
        void compute_oscillation_indicators();

        KOKKOS_INLINE_FUNCTION
        rtype basis_compute_1D(u_int8_t n, rtype x) const {
            return dispatch_compute_1D(basis_type, n, x);
        }

        KOKKOS_INLINE_FUNCTION
        rtype basis_gradient_1D(u_int8_t n, rtype x) const {
            return dispatch_gradient_1D(basis_type, n, x);
        }

        KOKKOS_INLINE_FUNCTION
        rtype basis_compute_2D(u_int8_t nx, u_int8_t ny, rtype x, rtype y) const {
            return dispatch_compute_2D(basis_type, nx, ny, x, y);
        }

        KOKKOS_INLINE_FUNCTION
        void basis_gradient_2D(u_int8_t nx, u_int8_t ny, rtype x, rtype y,
                               rtype &grad_x, rtype &grad_y) const {
            dispatch_gradient_2D(basis_type, nx, ny, x, y, grad_x, grad_y);
        }

        BasisType basis_type;
        u_int8_t poly_order;
        Quadrature quadrature;
        u_int16_t n_dof;
        Kokkos::View<u_int32_t *[N_DIM]> poly_indices;
        Kokkos::View<u_int32_t *[N_DIM]>::HostMirror h_poly_indices;
        // ^ Contains the polynomial powers for each dimension for each degree of freedom,
        //   precomputed for easy lookup
        rtype max_stencil_size_factor;
        u_int16_t max_cells_per_stencil;
        Kokkos::View<u_int32_t *> offsets_stencil_groups;
        Kokkos::View<u_int32_t *>::HostMirror h_offsets_stencil_groups;
        // ^ Used to get the IDs of the stencils associated with a given cell.
        // - The difference between two entries is the number of stencils for the target cell.
        // - The last entry is the total number of stencils.
        // - Indexed by:
        //   - i_cell
        // - Indexes the following:
        //   - offsets_stencils
        //   - offsets_reconstruction_matrices
        Kokkos::View<u_int32_t *> offsets_stencils;
        Kokkos::View<u_int32_t *>::HostMirror h_offsets_stencils;
        // ^ Used to get a particular stencil from the stencils array.
        // - The difference between two entries is the number of neighbor cells in a given stencil.
        // - The last entry is the total number of neighbor cells in all stencils.
        // - Indexed by:
        //   - offsets_stencil_groups
        // - Indexes the following:
        //   - stencils
        //   - transformed_areas
        Kokkos::View<u_int32_t *> stencils;
        Kokkos::View<u_int32_t *>::HostMirror h_stencils;
        // ^ Contains the IDs of the neighbor cells in a given stencil.
        // - Indexed by:
        //   - offsets_stencils
        Kokkos::View<u_int32_t *> offsets_reconstruction_matrices;
        Kokkos::View<u_int32_t *>::HostMirror h_offsets_reconstruction_matrices;
        // ^ Used to get the a particular reconstruction matrix from the reconstruction_matrices array.
        // - Each reconstruction matrix is associated with one stencil.
        // - The difference between two entries is the number of matrix elements for the stencil,
        //   which is MxK, where M is the number of cells in the stencil and K is the number of degrees
        //   of freedom.
        // - Indexed by:
        //   - offsets_stencil_groups
        // - Indexes the following:
        //   - reconstruction_matrices
        Kokkos::View<rtype *> reconstruction_matrices;
        Kokkos::View<rtype *>::HostMirror h_reconstruction_matrices;
        // ^ Contains the reconstruction matrices (stored as Moore-Penrose pseudoinverses) for each stencil.
        // - The matrices are stored in row-major order.
        // - Each matrix is MxK, where M is the number of cells in the stencil and K is the number of degrees
        //   of freedom.
        // - Indexed by:
        //   - offsets_reconstruction_matrices
        Kokkos::View<rtype *> transformed_areas;
        Kokkos::View<rtype *>::HostMirror h_transformed_areas;
        // ^ Contains the areas of the transformed triangles for each neighbor cell in each stencil.
        // - Indexed by:
        //   - offsets_stencils
};

#endif // FACE_RECONSTRUCTION_H
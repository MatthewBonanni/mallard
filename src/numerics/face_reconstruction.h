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

#include "common_typedef.h"
#include "mesh.h"

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
        virtual void init();

        /**
         * @brief Print the face reconstruction.
         */
        void print() const;

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
        TENO(u_int8_t poly_order, rtype max_stencil_size_factor);

        /**
         * @brief Destroy the TENO object
         */
        ~TENO();

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
                                      std::vector<u_int32_t> & v_offsets_stencils_of_cell,
                                      std::vector<u_int32_t> & v_stencils_of_cell,
                                      std::vector<u_int32_t> & v_stencils);
        void compute_stencils();
        void compute_reconstruction_matrices();
        void compute_oscillation_indicators();

        u_int8_t poly_order;
        u_int16_t n_dof;
        Kokkos::View<u_int32_t *[N_DIM]> poly_indices;
        Kokkos::View<u_int32_t *[N_DIM]>::HostMirror h_poly_indices;
        // ^ contains the polynomial powers for each dimension for each degree of freedom,
        // precompute for easy lookup
        rtype max_stencil_size_factor;
        u_int16_t max_cells_per_stencil;
        Kokkos::View<u_int32_t *> offsets_stencils_of_cell;
        Kokkos::View<u_int32_t *>::HostMirror h_offsets_stencils_of_cell;
        // ^ vector of offsets into stencils_of_cell, to get the group of stencils for the target
        // cell. The difference between two entries is the number of stencils
        // for the target cell
        Kokkos::View<u_int32_t *> stencils_of_cell;
        Kokkos::View<u_int32_t *>::HostMirror h_stencils_of_cell;
        // ^ vector of offsets into stencils, to get a particular stencil from the group.
        // The difference between two offsets is the number of cells in a given stencil
        Kokkos::View<u_int32_t *> stencils;
        Kokkos::View<u_int32_t *>::HostMirror h_stencils;
        // ^ vector of cell indices that are part of each stencil.
        Kokkos::View<u_int32_t *> weights_of_cell;
        Kokkos::View<u_int32_t *>::HostMirror h_weights_of_cell;
        // ^ vector of offsets into reconstruction_matrices, to get the group of matrices for a
        // stencil. The difference between two entries is the number of matrix elements for the
        // stencil, which is MxK, where M is the number of cells in the stencil and K is the number
        // of degrees of freedom. This vector can be indexed by offsets_stencils_of_cell.
        Kokkos::View<rtype *> reconstruction_matrices;
        Kokkos::View<rtype *>::HostMirror h_reconstruction_matrices;
        // ^ vector of reconstruction matrices for each cell in the stencil. Matrices are stored in
        // row-major order.
};

#endif // FACE_RECONSTRUCTION_H
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
#include "boundary.h"
#include "gradient.h"

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

#endif // FACE_RECONSTRUCTION_H
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

#include "common_typedef.h"
#include "mesh.h"

enum class FaceReconstructionType {
    FirstOrder,
    WENO,
};

static const std::unordered_map<std::string, FaceReconstructionType> FACE_RECONSTRUCTION_TYPES = {
    {"FO", FaceReconstructionType::FirstOrder},
    {"WENO", FaceReconstructionType::WENO}
};

static const std::unordered_map<FaceReconstructionType, std::string> FACE_RECONSTRUCTION_NAMES = {
    {FaceReconstructionType::FirstOrder, "FO"},
    {FaceReconstructionType::WENO, "WENO"}
};

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
         * @brief Set the cell conservatives.
         * @param cell_conservatives Pointer to the cell conservatives.
         */
        void set_cell_conservatives(StateVector * cell_conservatives);

        /**
         * @brief Set the face conservatives.
         * @param face_conservatives Pointer to the face conservatives.
         */
        void set_face_conservatives(FaceStateVector * face_conservatives);

        /**
         * @brief Set the mesh.
         * @param mesh Pointer to the mesh.
         */
        void set_mesh(std::shared_ptr<Mesh> mesh);

        /**
         * @brief Reconstruct the face values.
         * @param solution Pointer to the solution.
         * @param face_solution Pointer to the face solution.
         */
        virtual void calc_face_values(StateVector * solution,
                                      FaceStateVector * face_solution) = 0;
    protected:
        FaceReconstructionType type;
        StateVector * cell_conservatives;
        FaceStateVector * face_conservatives;
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
         * @param solution Pointer to the solution.
         * @param face_solution Pointer to the face solution.
         */
        void calc_face_values(StateVector * solution,
                              FaceStateVector * face_solution) override;
    protected:
    private:
};

class WENO : public FaceReconstruction {
    public:
        /**
         * @brief Construct a new WENO object
         */
        WENO();

        /**
         * @brief Destroy the WENO object
         */
        ~WENO();

        /**
         * @brief Reconstruct the face values.
         * @param solution Pointer to the solution.
         * @param face_solution Pointer to the face solution.
         */
        void calc_face_values(StateVector * solution,
                              FaceStateVector * face_solution) override;
    protected:
    private:
};

#endif // FACE_RECONSTRUCTION_H
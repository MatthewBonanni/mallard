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

#include "common/common_typedef.h"
#include "mesh/mesh.h"

class FaceReconstruction {
    public:
        /**
         * @brief Construct a new Face Reconstruction object
         */
        FaceReconstruction();

        /**
         * @brief Destroy the Face Reconstruction object
         */
        ~FaceReconstruction();

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
        void set_mesh(Mesh * mesh);

        /**
         * @brief Reconstruct the face values.
         */
        virtual void calc_face_values();
    protected:
        StateVector * cell_conservatives;
        FaceStateVector * face_conservatives;
        Mesh * mesh;
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
         */
        void calc_face_values() override;
    protected:
    private:
};

#endif // FACE_RECONSTRUCTION_H
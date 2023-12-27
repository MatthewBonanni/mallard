/**
 * @file face_reconstruction.cpp
 * @author Matthew Bonanni (mbonanni001@gmail.com)
 * @brief Face reconstruction class implementation.
 * @version 0.1
 * @date 2023-12-24
 * 
 * @copyright Copyright (c) 2023 Matthew Bonanni
 * 
 */

#include "face_reconstruction.h"

FaceReconstruction::FaceReconstruction() {
    // Empty
}

FaceReconstruction::~FaceReconstruction() {
    // Empty
}

void FaceReconstruction::set_cell_conservatives(StateVector * cell_conservatives) {
    this->cell_conservatives = cell_conservatives;
}

void FaceReconstruction::set_face_conservatives(FaceStateVector * face_conservatives) {
    this->face_conservatives = face_conservatives;
}

void FaceReconstruction::set_mesh(Mesh * mesh) {
    this->mesh = mesh;
}

void FaceReconstruction::calc_face_values() {
    throw std::runtime_error("FaceReconstruction::calc_face_values() not implemented.");
}

FirstOrder::FirstOrder() {
    // Empty
}

FirstOrder::~FirstOrder() {
    // Empty
}

void FirstOrder::calc_face_values() {
    for (int i_face = 0; i_face < mesh->n_faces(); i_face++) {
        int i_cell_0 = mesh->cells_of_face(i_face)[0];
        int i_cell_1 = mesh->cells_of_face(i_face)[1];

        (*face_conservatives)[i_face][0] = (*cell_conservatives)[i_cell_0];
        (*face_conservatives)[i_face][1] = (*cell_conservatives)[i_cell_1];
    }
}
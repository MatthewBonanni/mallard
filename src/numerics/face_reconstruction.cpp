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

#include <iostream>

FaceReconstruction::FaceReconstruction() {
    // Empty
}

FaceReconstruction::~FaceReconstruction() {
    std::cout << "Destroying face reconstruction: " << FACE_RECONSTRUCTION_NAMES.at(type) << std::endl;
}

void FaceReconstruction::set_cell_conservatives(StateVector * cell_conservatives) {
    this->cell_conservatives = cell_conservatives;
}

void FaceReconstruction::set_face_conservatives(FaceStateVector * face_conservatives) {
    this->face_conservatives = face_conservatives;
}

void FaceReconstruction::set_mesh(std::shared_ptr<Mesh> mesh) {
    this->mesh = mesh;
}

FirstOrder::FirstOrder() {
    type = FaceReconstructionType::FirstOrder;
}

FirstOrder::~FirstOrder() {
    // Empty
}

void FirstOrder::calc_face_values(StateVector * solution,
                                  FaceStateVector * face_solution) {
    for (int i_face = 0; i_face < mesh->n_faces(); i_face++) {
        int i_cell_l = mesh->cells_of_face(i_face)[0];
        int i_cell_r = mesh->cells_of_face(i_face)[1];

        (*face_solution)[i_face][0] = (*solution)[i_cell_l];
        (*face_solution)[i_face][1] = (*solution)[i_cell_r];
    }
}

WENO::WENO() {
    type = FaceReconstructionType::WENO;
}

WENO::~WENO() {
    // Empty
}

void WENO::calc_face_values(StateVector * solution,
                            FaceStateVector * face_solution) {
    throw std::runtime_error("WENO::calc_face_values() not implemented.");
}
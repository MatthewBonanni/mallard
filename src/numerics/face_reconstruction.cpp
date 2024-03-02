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
#include "face_reconstruction_functors.h"

#include <iostream>

#include "common_math.h"

FaceReconstruction::FaceReconstruction() {
    // Empty
}

FaceReconstruction::~FaceReconstruction() {
    std::cout << "Destroying face reconstruction: " << FACE_RECONSTRUCTION_NAMES.at(type) << std::endl;
}

void FaceReconstruction::init() {
    print();
}

void FaceReconstruction::print() const {
    std::cout << LOG_SEPARATOR << std::endl;
    std::cout << "Face reconstruction: " << FACE_RECONSTRUCTION_NAMES.at(type) << std::endl;
    std::cout << LOG_SEPARATOR << std::endl;
}

void FaceReconstruction::set_cell_conservatives(view_2d * cell_conservatives) {
    this->cell_conservatives = cell_conservatives;
}

void FaceReconstruction::set_face_conservatives(view_3d * face_conservatives) {
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

void FirstOrder::calc_face_values(view_2d * solution,
                                  view_3d * face_solution) {
    FaceReconstructionFunctors::FirstOrderFunctor functor(mesh->cells_of_face,
                                                          *face_solution,
                                                          *solution);
    Kokkos::parallel_for(mesh->n_faces(), functor);
}

WENO3_JS::WENO3_JS() {
    type = FaceReconstructionType::WENO3_JS;
}

WENO3_JS::~WENO3_JS() {
    // Empty
}

void WENO3_JS::calc_face_values(view_2d * solution,
                                view_3d * face_solution) {
    FaceReconstructionFunctors::WENO3_JSFunctor functor(mesh->cells_of_face,
                                                        mesh->face_normals,
                                                        mesh->n_cells_x(),
                                                        mesh->n_cells_y(),
                                                        *face_solution,
                                                        *solution);
    Kokkos::parallel_for(mesh->n_faces(), functor);
}

WENO5_JS::WENO5_JS() {
    type = FaceReconstructionType::WENO5_JS;
}

WENO5_JS::~WENO5_JS() {
    // Empty
}

void WENO5_JS::calc_face_values(view_2d * solution,
                                view_3d * face_solution) {
    FaceReconstructionFunctors::WENO5_JSFunctor functor(mesh->cells_of_face,
                                                        mesh->face_normals,
                                                        mesh->n_cells_x(),
                                                        mesh->n_cells_y(),
                                                        *face_solution,
                                                        *solution);
    Kokkos::parallel_for(mesh->n_faces(), functor);
}
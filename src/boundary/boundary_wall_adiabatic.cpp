/**
 * @file boundary_wall_adiabatic.cpp
 * @author Matthew Bonanni (mbonanni001@gmail.com)
 * @brief Adiabatic wall boundary condition class implementation.
 * @version 0.1
 * @date 2023-12-20
 * 
 * @copyright Copyright (c) 2023 Matthew Bonanni
 * 
 */

#include "boundary_wall_adiabatic.h"

#include <iostream>

#include "common.h"

BoundaryWallAdiabatic::BoundaryWallAdiabatic() {
    type = BoundaryType::WALL_ADIABATIC;
}

BoundaryWallAdiabatic::~BoundaryWallAdiabatic() {
    // Empty
}

void BoundaryWallAdiabatic::print() {
    Boundary::print();
    std::cout << LOG_SEPARATOR << std::endl;
}

void BoundaryWallAdiabatic::init(const toml::table & input) {
    print();
}

void BoundaryWallAdiabatic::apply(view_3d * face_solution,
                                  view_2d * rhs) {
    Kokkos::parallel_for(zone->n_faces(), KOKKOS_LAMBDA(const int i_local) {
        State flux;
        State conservatives_l;
        Primitives primitives_l;

        int i_face = (*zone->faces())[i_local];
        int i_cell_l = mesh->cells_of_face(i_face)[0];
        NVector n_unit = unit(mesh->face_normal(i_face));

        // Get cell conservatives
        for (int j = 0; j < N_CONSERVATIVE; j++) {
            conservatives_l[j] = (*face_solution)(i_face, 0, j);
        }

        // Compute relevant primitive variables
        physics->compute_primitives_from_conservatives(primitives_l.data(), conservatives_l.data());

        // Compute flux
        flux[0] = 0.0;
        flux[1] = primitives_l[2] * n_unit[0];
        flux[2] = primitives_l[2] * n_unit[1];
        flux[3] = 0.0;

        // Add flux to RHS
        for (int j = 0; j < N_CONSERVATIVE; j++) {
            Kokkos::atomic_add(&(*rhs)(i_cell_l, j), -mesh->face_area(i_face) * flux[j]);
        }
    });
}
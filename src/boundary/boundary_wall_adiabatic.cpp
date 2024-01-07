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

#include "common/common.h"

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
    // Empty
}

void BoundaryWallAdiabatic::apply(FaceStateVector * face_solution,
                                  StateVector * rhs) {
    int i_face, i_cell_l;
    State flux;
    State * conservatives_l;
    Primitives primitives_l;
    NVector n_unit;
    for (int i_local = 0; i_local < zone->n_faces(); i_local++) {
        i_face = (*zone->faces())[i_local];
        i_cell_l = mesh->cells_of_face(i_face)[0];
        n_unit = unit(mesh->face_normal(i_face));

        // Get cell conservatives
        conservatives_l = &(*face_solution)[i_face][0];

        // Compute relevant primitive variables
        physics->compute_primitives_from_conservatives(primitives_l, *conservatives_l);

        // Compute flux
        flux[0] = 0.0;
        flux[1] = primitives_l[2] * n_unit[0];
        flux[2] = primitives_l[2] * n_unit[1];
        flux[3] = 0.0;

        // Add flux to RHS
        for (int j = 0; j < 4; j++) {
            (*rhs)[i_cell_l][j] -= mesh->face_area(i_face) * flux[j];
        }
    }
}
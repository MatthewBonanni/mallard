/**
 * @file boundary_symmetry.cpp
 * @author Matthew Bonanni (mbonanni001@gmail.com)
 * @brief Symmetry boundary condition class implementation.
 * @version 0.1
 * @date 2024-01-04
 * 
 * @copyright Copyright (c) 2024 Matthew Bonanni
 * 
 */

#include "boundary_symmetry.h"

#include <iostream>

#include "common/common.h"

BoundarySymmetry::BoundarySymmetry() {
    type = BoundaryType::SYMMETRY;
}

BoundarySymmetry::~BoundarySymmetry() {
    // Empty
}

void BoundarySymmetry::print() {
    Boundary::print();
    std::cout << LOG_SEPARATOR << std::endl;
}

void BoundarySymmetry::init(const toml::table & input) {
    // Empty
}

void BoundarySymmetry::apply(StateVector * solution,
                             StateVector * rhs) {
    State flux;
    Primitives primitives_l, primitives_r;
    double u_n;
    NVector u_l, u_r, n_unit;
    for (int i = 0; i < zone->n_faces(); i++) {
        int i_face = (*zone->faces())[i];
        int i_cell_l = mesh->cells_of_face(i_face)[0];
        
        // Get face normal vector
        n_unit = unit(mesh->face_normal(i));

        // Compute relevant primitive variables
        physics->compute_primitives_from_conservatives(primitives_l, (*solution)[i_cell_l]);

        // Right state = left state, but reflect velocity vector
        primitives_r = primitives_l;
        u_l[0] = primitives_l[0];
        u_l[1] = primitives_l[1];
        u_n = dot<double>(u_l.data(), n_unit.data(), 2);
        for (int j = 0; j < 2; j++) {
            u_r[j] = u_l[j] - 2.0 * u_n * n_unit[j];
        }
        primitives_r[0] = u_r[0];
        primitives_r[1] = u_r[1];

        // Calculate flux
        physics->calc_euler_flux(flux, n_unit,
                                 (*solution)[i_cell_l][0],
                                 (*solution)[i_cell_l][0],
                                 primitives_l, primitives_r);

        // Add flux to RHS
        for (int j = 0; j < 4; j++) {
            (*rhs)[i_cell_l][j] -= mesh->face_area(i) * flux[j];
        }
    }
}
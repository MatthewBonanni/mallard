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

void BoundaryWallAdiabatic::apply(StateVector * solution,
                                  StateVector * rhs) {
    State flux;
    double rho_l, E_l, e_l, gamma_l, p_l;
    NVector u_l;
    for (int i = 0; i < zone->n_faces(); i++) {
        int i_face = (*zone->faces())[i];
        int i_cell_l = mesh->cells_of_face(i_face)[0];

        // Compute relevant primitive variables
        rho_l = (*solution)[i_cell_l][0];
        u_l[0] = (*solution)[i_cell_l][1] / rho_l;
        u_l[1] = (*solution)[i_cell_l][2] / rho_l;
        E_l = (*solution)[i_cell_l][3] / rho_l;
        e_l = E_l - 0.5 * dot_self(u_l);
        gamma_l = physics->get_gamma();
        p_l = (gamma_l - 1.0) * rho_l * e_l;

        flux[0] = 0.0;
        flux[1] = -p_l * mesh->face_normal(i_face)[0];
        flux[2] = -p_l * mesh->face_normal(i_face)[1];
        flux[3] = 0.0;

        // Add flux to RHS
        for (int j = 0; j < 4; j++) {
            (*rhs)[i_cell_l][j] -= mesh->face_area(i) * flux[j];
        }
    }
}
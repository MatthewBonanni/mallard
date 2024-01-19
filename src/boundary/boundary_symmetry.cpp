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

#include "common.h"

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
    print();
}

void BoundarySymmetry::apply(view_3d * face_solution,
                             view_2d * rhs) {
    Kokkos::parallel_for(zone->n_faces(), KOKKOS_LAMBDA(const int i_local) {
        State flux;
        State conservatives_l;
        Primitives primitives_l, primitives_r;
        rtype u_n;
        NVector u_l, u_r, n_unit;

        int i_face = (*zone->faces())[i_local];
        int i_cell_l = mesh->cells_of_face(i_face)[0];
        n_unit = unit(mesh->face_normal(i_face));

        // Get cell conservatives
        for (int j = 0; j < N_CONSERVATIVE; j++) {
            conservatives_l[j] = (*face_solution)(i_face, 0, j);
        }

        // Compute relevant primitive variables
        physics->compute_primitives_from_conservatives(primitives_l, conservatives_l);

        // Right state = left state, but reflect velocity vector
        primitives_r = primitives_l;
        u_l[0] = primitives_l[0];
        u_l[1] = primitives_l[1];
        u_n = dot<N_DIM>(u_l.data(), n_unit.data());
        for (int j = 0; j < N_DIM; j++) {
            u_r[j] = u_l[j] - 2.0 * u_n * n_unit[j];
        }
        primitives_r[0] = u_r[0];
        primitives_r[1] = u_r[1];

        // Calculate flux
        riemann_solver->calc_flux(flux, n_unit,
                                  conservatives_l[0], primitives_l.data(),
                                  primitives_l[2], physics->get_gamma(), primitives_l[4],
                                  conservatives_l[0], primitives_r.data(),
                                  primitives_r[2], physics->get_gamma(), primitives_r[4]);

        // Add flux to RHS
        for (int j = 0; j < N_CONSERVATIVE; j++) {
            Kokkos::atomic_add(&(*rhs)(i_cell_l, j), -mesh->face_area(i_face) * flux[j]);
        }
    });
}
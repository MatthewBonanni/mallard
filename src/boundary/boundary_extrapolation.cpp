/**
 * @file boundary_extrapolation.cpp
 * @author Matthew Bonanni (mbonanni001@gmail.com)
 * @brief Extrapolation boundary condition class implementation.
 * @version 0.1
 * @date 2024-01-18
 * 
 * @copyright Copyright (c) 2024 Matthew Bonanni
 * 
 */

#include "boundary_extrapolation.h"

#include <iostream>

#include "common.h"

BoundaryExtrapolation::BoundaryExtrapolation() {
    type = BoundaryType::EXTRAPOLATION;
}

BoundaryExtrapolation::~BoundaryExtrapolation() {
    // Empty
}

void BoundaryExtrapolation::print() {
    Boundary::print();
    std::cout << LOG_SEPARATOR << std::endl;
}

void BoundaryExtrapolation::init(const toml::value & input) {
    (void)(input);
    print();
}

void BoundaryExtrapolation::apply(view_3d * face_solution,
                                  view_2d * rhs) {
    Kokkos::parallel_for(zone->n_faces(), KOKKOS_LAMBDA(const u_int32_t i_local) {
        State flux;
        State conservatives_l, conservatives_r;
        Primitives primitives_l, primitives_r;
        NVector n_unit;

        u_int32_t i_face = (*zone->faces())[i_local];
        int32_t i_cell_l = mesh->cells_of_face(i_face)[0];
        n_unit = unit(mesh->face_normal(i_face));

        // Get cell conservatives
        for (u_int16_t j = 0; j < N_CONSERVATIVE; j++) {
            conservatives_l[j] = (*face_solution)(i_face, 0, j);
        }

        // Compute relevant primitive variables
        physics->compute_primitives_from_conservatives(primitives_l.data(), conservatives_l.data());

        // Zero-order extrapolation of conservative variables
        conservatives_r = conservatives_l;
        primitives_r = primitives_l; // Don't need to recompute primitives

        // Calculate flux
        riemann_solver->calc_flux(flux.data(), n_unit.data(),
                                  conservatives_l[0], primitives_l.data(),
                                  primitives_l[2], physics->get_gamma(), primitives_l[4],
                                  conservatives_r[0], primitives_r.data(),
                                  primitives_r[2], physics->get_gamma(), primitives_r[4]);

        // Add flux to RHS
        for (u_int16_t j = 0; j < N_CONSERVATIVE; j++) {
            Kokkos::atomic_add(&(*rhs)(i_cell_l, j), -mesh->face_area(i_face) * flux[j]);
        }
    });
}
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
        rtype flux[N_CONSERVATIVE];
        rtype conservatives_l[N_CONSERVATIVE];
        rtype conservatives_r[N_CONSERVATIVE];
        rtype primitives_l[N_PRIMITIVE];
        rtype primitives_r[N_PRIMITIVE];
        rtype n_vec[N_DIM];
        rtype n_unit[N_DIM];

        u_int32_t i_face = (*zone->faces())[i_local];
        int32_t i_cell_l = mesh->cells_of_face(i_face)[0];
        FOR_I_DIM n_vec[i] = mesh->face_normals(i_face, i);
        unit<N_DIM>(n_vec, n_unit);

        // Get cell conservatives
        for (u_int16_t j = 0; j < N_CONSERVATIVE; j++) {
            conservatives_l[j] = (*face_solution)(i_face, 0, j);
        }

        // Compute relevant primitive variables
        physics->compute_primitives_from_conservatives(primitives_l, conservatives_l);

        // Zero-order extrapolation of conservative variables
        FOR_I_CONSERVATIVE conservatives_r[i] = conservatives_l[i];
        FOR_I_PRIMITIVE primitives_r[i] = primitives_l[i]; // Don't need to recompute primitives

        // Calculate flux
        riemann_solver->calc_flux(flux, n_unit,
                                  conservatives_l[0], primitives_l,
                                  primitives_l[2], physics->get_gamma(), primitives_l[4],
                                  conservatives_r[0], primitives_r,
                                  primitives_r[2], physics->get_gamma(), primitives_r[4]);

        // Add flux to RHS
        for (u_int16_t j = 0; j < N_CONSERVATIVE; j++) {
            Kokkos::atomic_add(&(*rhs)(i_cell_l, j), -mesh->face_area(i_face) * flux[j]);
        }
    });
}
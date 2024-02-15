/**
 * @file solver_rhs.cpp
 * @author Matthew Bonanni (mbonanni001@gmail.com)
 * @brief Implementation of RHS methods for the Solver class.
 * @version 0.1
 * @date 2024-01-11
 * 
 * @copyright Copyright (c) 2024 Matthew Bonanni
 * 
 */

#include "solver.h"

void Solver::calc_rhs(view_2d * solution,
                      view_3d * face_solution,
                      view_2d * rhs) {
    pre_rhs(solution, face_solution, rhs);
    calc_rhs_source(solution, rhs);
    calc_rhs_interior(face_solution, rhs);
    calc_rhs_boundaries(face_solution, rhs);

    // Divide by cell volume
    Kokkos::parallel_for(mesh->n_cells(), KOKKOS_LAMBDA(const u_int32_t i) {
        for (u_int16_t j = 0; j < N_CONSERVATIVE; j++) {
            (*rhs)(i, j) /= mesh->cell_volume(i);
        }
    });
}

void Solver::pre_rhs(view_2d * solution,
                     view_3d * face_solution,
                     view_2d * rhs) {
    // Zero out RHS
    Kokkos::parallel_for(mesh->n_cells(), KOKKOS_LAMBDA(const u_int32_t i) {
        for (u_int16_t j = 0; j < N_CONSERVATIVE; j++) {
            (*rhs)(i, j) = 0.0;
        }
    });

    face_reconstruction->calc_face_values(solution, face_solution);
}

void Solver::calc_rhs_source(view_2d * solution,
                             view_2d * rhs) {
    (void)(solution);
    (void)(rhs);
    /** \todo Implement source terms. */
}

void Solver::calc_rhs_interior(view_3d * face_solution,
                               view_2d * rhs) {
    Kokkos::parallel_for(mesh->n_faces(), KOKKOS_LAMBDA(const u_int32_t i_face) {
        State flux;
        State conservatives_l;
        State conservatives_r;
        Primitives primitives_l;
        Primitives primitives_r;
        /** \todo Don't reallocate these every time. */

        /** \todo Iterate only over interior faces to save time. */
        if (mesh->cells_of_face(i_face)[1] == -1) {
            // Boundary face
            return;
        }

        int32_t i_cell_l = mesh->cells_of_face(i_face)[0];
        int32_t i_cell_r = mesh->cells_of_face(i_face)[1];
        NVector n_unit = unit(mesh->face_normal(i_face));

        // Get face conservatives
        for (u_int16_t j = 0; j < N_CONSERVATIVE; j++) {
            conservatives_l[j] = (*face_solution)(i_face, 0, j);
            conservatives_r[j] = (*face_solution)(i_face, 1, j);
        }

        // Compute relevant primitive variables
        physics->compute_primitives_from_conservatives(primitives_l.data(), conservatives_l.data());
        physics->compute_primitives_from_conservatives(primitives_r.data(), conservatives_r.data());

        // Calculate flux
        riemann_solver->calc_flux(flux, n_unit,
                                  conservatives_l[0], primitives_l.data(),
                                  primitives_l[2], physics->get_gamma(), primitives_l[4],
                                  conservatives_r[0], primitives_r.data(),
                                  primitives_r[2], physics->get_gamma(), primitives_r[4]);
        
        // Add flux to RHS
        for (u_int16_t j = 0; j < N_CONSERVATIVE; j++) {
            Kokkos::atomic_add(&(*rhs)(i_cell_l, j), -mesh->face_area(i_face) * flux[j]);
            Kokkos::atomic_add(&(*rhs)(i_cell_r, j),  mesh->face_area(i_face) * flux[j]);
        }
    });
}

void Solver::calc_rhs_boundaries(view_3d * face_solution,
                                 view_2d * rhs) {
    for (auto& boundary : boundaries) {
        boundary->apply(face_solution, rhs);
    }
}
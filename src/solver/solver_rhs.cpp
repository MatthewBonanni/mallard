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

void Solver::calc_rhs(StateVector * solution,
                      FaceStateVector * face_solution,
                      StateVector * rhs) {
    pre_rhs(solution, face_solution, rhs);
    calc_rhs_source(solution, rhs);
    calc_rhs_interior(face_solution, rhs);
    calc_rhs_boundaries(face_solution, rhs);

    for (int i = 0; i < mesh->n_cells(); i++) {
        for (int j = 0; j < 4; j++) {
            (*rhs)[i][j] /= mesh->cell_volume(i);
        }
    }
}

void Solver::pre_rhs(StateVector * solution,
                     FaceStateVector * face_solution,
                     StateVector * rhs) {
    for (int i = 0; i < mesh->n_cells(); i++) {
        for (int j = 0; j < 4; j++) {
            (*rhs)[i][j] = 0.0;
        }
    }

    face_reconstruction->calc_face_values(solution, face_solution);
}

void Solver::calc_rhs_source(StateVector * solution,
                             StateVector * rhs) {
    // \todo Sources not implemented yet.
    for (int i = 0; i < mesh->n_cells(); i++) {
        (*rhs)[i][0] += 0.0;
        (*rhs)[i][1] += 0.0;
        (*rhs)[i][2] += 0.0;
        (*rhs)[i][3] += 0.0;
    }
}

void Solver::calc_rhs_interior(FaceStateVector * face_solution,
                               StateVector * rhs) {
    State flux;
    NVector n_unit;
    State * conservatives_l;
    State * conservatives_r;
    Primitives primitives_l;
    Primitives primitives_r;
    for (int i_face = 0; i_face < mesh->n_faces(); i_face++) {
        // \todo iterate only over interior faces to save time.
        if (mesh->cells_of_face(i_face)[1] == -1) {
            // Boundary face
            continue;
        }

        int i_cell_l = mesh->cells_of_face(i_face)[0];
        int i_cell_r = mesh->cells_of_face(i_face)[1];

        // Get face conservatives
        conservatives_l = &(*face_solution)[i_face][0];
        conservatives_r = &(*face_solution)[i_face][1];

        // Compute relevant primitive variables
        physics->compute_primitives_from_conservatives(primitives_l, *conservatives_l);
        physics->compute_primitives_from_conservatives(primitives_r, *conservatives_r);

        // Get face normal vector
        n_unit = unit(mesh->face_normal(i_face));

        // Calculate flux
        physics->calc_euler_flux(flux, n_unit,
                                 (*conservatives_l)[0], (*conservatives_r)[0],
                                 primitives_l, primitives_r);
        
        // Add flux to RHS
        for (int j = 0; j < 4; j++) {
            (*rhs)[i_cell_l][j] -= mesh->face_area(i_face) * flux[j];
            (*rhs)[i_cell_r][j] += mesh->face_area(i_face) * flux[j];
        }
    }
}

void Solver::calc_rhs_boundaries(FaceStateVector * face_solution,
                                 StateVector * rhs) {
    for (auto& boundary : boundaries) {
        boundary->apply(face_solution, rhs);
    }
}
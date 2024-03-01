/**
 * @file solver_rhs.h
 * @author Matthew Bonanni (mbonanni001@gmail.com)
 * @brief Implementation of RHS methods for the Solver class.
 * @version 0.1
 * @date 2024-01-11
 * 
 * @copyright Copyright (c) 2024 Matthew Bonanni
 * 
 */

#include "solver.h"
#include "solver_functors.h"

void Solver::calc_rhs(view_2d * solution,
                      view_3d * face_solution,
                      view_2d * rhs) {
    pre_rhs(solution, face_solution, rhs);
    calc_rhs_source(solution, rhs);
    calc_rhs_interior(face_solution, rhs);
    calc_rhs_boundaries(face_solution, rhs);

    // Divide by cell volume
    DivideVolumeFunctor divide_volume_functor(mesh->cell_volume, *rhs);
    Kokkos::parallel_for(mesh->n_cells(), divide_volume_functor);
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

    FluxFunctor flux_functor(mesh->face_normals,
                             mesh->face_area,
                             mesh->cells_of_face,
                             *face_solution,
                             *rhs,
                             *riemann_solver,
                             *physics);
    Kokkos::parallel_for(mesh->n_faces(), flux_functor);
}

void Solver::calc_rhs_boundaries(view_3d * face_solution,
                                 view_2d * rhs) {
    for (auto& boundary : boundaries) {
        boundary->apply(face_solution, rhs);
    }
}
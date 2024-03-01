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

#ifndef SOLVER_RHS_H
#define SOLVER_RHS_H

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
    /** \todo Figure out a way to clean this up... */
    if (physics->get_type() == PhysicsType::EULER) {
        if (riemann_solver->get_type() == RiemannSolverType::Rusanov) {
            // Rusanov Riemann solver
            FluxFunctor<Euler, Rusanov> flux_functor(mesh->face_normals,
                                                     mesh->face_area,
                                                     mesh->cells_of_face,
                                                     *face_solution,
                                                     *rhs,
                                                     dynamic_cast<Rusanov &>(*riemann_solver),
                                                     dynamic_cast<Euler &>(*physics));
            Kokkos::parallel_for(mesh->n_faces(), flux_functor);
        } else if (riemann_solver->get_type() == RiemannSolverType::Roe) {
            // Roe Riemann solver
            FluxFunctor<Euler, Roe> flux_functor(mesh->face_normals,
                                                 mesh->face_area,
                                                 mesh->cells_of_face,
                                                 *face_solution,
                                                 *rhs,
                                                 dynamic_cast<Roe &>(*riemann_solver),
                                                 dynamic_cast<Euler &>(*physics));
            Kokkos::parallel_for(mesh->n_faces(), flux_functor);
        } else if (riemann_solver->get_type() == RiemannSolverType::HLL) {
            // HLL Riemann solver
            FluxFunctor<Euler, HLL> flux_functor(mesh->face_normals,
                                                 mesh->face_area,
                                                 mesh->cells_of_face,
                                                 *face_solution,
                                                 *rhs,
                                                 dynamic_cast<HLL &>(*riemann_solver),
                                                 dynamic_cast<Euler &>(*physics));
            Kokkos::parallel_for(mesh->n_faces(), flux_functor);
        } else if (riemann_solver->get_type() == RiemannSolverType::HLLC) {
            // HLLC Riemann solver
            FluxFunctor<Euler, HLLC> flux_functor(mesh->face_normals,
                                                  mesh->face_area,
                                                  mesh->cells_of_face,
                                                  *face_solution,
                                                  *rhs,
                                                  dynamic_cast<HLLC &>(*riemann_solver),
                                                  dynamic_cast<Euler &>(*physics));
            Kokkos::parallel_for(mesh->n_faces(), flux_functor);
        } else {
            // Should never get here due to the enum class.
            throw std::runtime_error("Unknown Riemann solver type.");
        }
    } else {
        // Should never get here due to the enum class.
        throw std::runtime_error("Unknown physics type.");
    }
}

void Solver::calc_rhs_boundaries(view_3d * face_solution,
                                 view_2d * rhs) {
    for (auto& boundary : boundaries) {
        boundary->apply(face_solution, rhs);
    }
}

#endif // SOLVER_RHS_H
/**
 * @file solver_rhs.cpp
 * @author Matthew Bonanni (mbonanni001@gmail.com)
 * @brief Implementation of RHS methods for the Solver class.
 * @version 0.2
 * @date 2024-01-11
 *
 * @copyright Copyright (c) 2024 Matthew Bonanni
 *
 */

#include "solver.h"

#include <Kokkos_Core.hpp>

#include "flux_functor.h"

void Solver::calc_rhs(StateView solution, StateView rhs) {
    const Euler phys = physics;
    Kokkos::View<rtype *[N_CONSERVATIVE]> W = W_cells;
    Kokkos::parallel_for("rhs_init", mesh->n_cells, KOKKOS_LAMBDA(const uint32_t i_cell) {
        rtype cons[N_CONSERVATIVE], W_c[N_CONSERVATIVE];
        FOR_I_CONSERVATIVE {
            cons[i] = solution(i_cell, i);
            rhs(i_cell, i) = 0.0;
        }
        phys.compute_W_from_conservatives(W_c, cons);
        FOR_I_CONSERVATIVE W(i_cell, i) = W_c[i];
    });

    face_reconstruction->calc_face_values(W_cells, face_solution);

    switch (riemann_solver_type) {
        case RiemannSolverType::RUSANOV:
            launch_flux_functor<riemann::Rusanov>(rhs);
            break;
        case RiemannSolverType::HLL:
            launch_flux_functor<riemann::HLL>(rhs);
            break;
        case RiemannSolverType::HLLC:
            launch_flux_functor<riemann::HLLC>(rhs);
            break;
    }

    Kokkos::View<rtype *> vol = mesh->cell_volume;
    Kokkos::parallel_for("rhs_divide_volume", mesh->n_cells, KOKKOS_LAMBDA(const uint32_t i_cell) {
        FOR_I_CONSERVATIVE rhs(i_cell, i) /= vol(i_cell);
    });
}

template <typename T_riemann_solver>
void Solver::launch_flux_functor(StateView rhs) {
    ConvectiveFluxFunctor<T_riemann_solver> functor{mesh->face_normals,
                                                    mesh->face_area,
                                                    mesh->cells_of_face,
                                                    face_reconstruction->quadrature_face.weights,
                                                    face_solution,
                                                    boundary_data,
                                                    rhs,
                                                    physics.gamma};
    Kokkos::parallel_for("convective_flux", mesh->n_faces, functor);
}

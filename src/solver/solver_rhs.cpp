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
#include "gradient.h"
#include "viscous_flux.h"

void Solver::calc_rhs(StateView solution, StateView rhs, rtype t_stage) {
    halo.exchange(solution);
    update_boundary_states(t_stage);
    if (!average_pressure_outlets.empty()) {
        update_average_pressure_outlets(solution);
    }

    const Euler phys = physics;
    Kokkos::View<rtype *[N_CONSERVATIVE]> W = W_cells;
    Kokkos::parallel_for("rhs_W", mesh->n_cells, KOKKOS_LAMBDA(const uint32_t i_cell) {
        rtype cons[N_CONSERVATIVE], W_c[N_CONSERVATIVE];
        FOR_I_CONSERVATIVE cons[i] = solution(i_cell, i);
        phys.compute_W_from_conservatives(W_c, cons);
        FOR_I_CONSERVATIVE W(i_cell, i) = W_c[i];
    });

    face_reconstruction->calc_face_values(W_cells, face_solution);

    switch (riemann_solver_type) {
        case RiemannSolverType::RUSANOV:
            launch_flux_functor<riemann::Rusanov>();
            break;
        case RiemannSolverType::HLL:
            launch_flux_functor<riemann::HLL>();
            break;
        case RiemannSolverType::HLLC:
            launch_flux_functor<riemann::HLLC>();
            break;
        case RiemannSolverType::ROE:
            launch_flux_functor<riemann::Roe>();
            break;
        case RiemannSolverType::RHLL:
            launch_flux_functor<riemann::RHLL>();
            break;
    }

    if (physics.is_viscous()) {
        LSQGradientFunctor gradient_functor{mesh->offsets_faces_of_cell, mesh->faces_of_cell,
                                            mesh->cells_of_face, mesh->cell_coords, mesh->face_coords,
                                            mesh->face_normals, boundary_data, W_cells, viscous_gradients};
        LSQVertexGradientFunctor vertex_gradient_functor{gradient_functor, mesh->offsets_cells_of_cell,
                                                         mesh->cells_of_cell};
        Kokkos::parallel_for("viscous_gradients", mesh->n_cells, vertex_gradient_functor);
        ViscousFluxFunctor viscous_functor{mesh->face_normals, mesh->face_area, mesh->face_coords,
                                           mesh->cell_coords, mesh->cells_of_face, W_cells,
                                           viscous_gradients, boundary_data, face_flux, physics};
        Kokkos::parallel_for("viscous_flux", mesh->n_faces, viscous_functor);
    }

    FaceFluxSumFunctor sum_functor{mesh->offsets_faces_of_cell, mesh->faces_of_cell, mesh->cells_of_face,
                                   face_flux, rhs};
    Kokkos::parallel_for("face_flux_sum", mesh->n_cells, sum_functor);

    Kokkos::View<rtype *> vol = mesh->cell_volume;
    if (has_gravity || !source_expressions.empty()) {
        update_source_field(t_stage);
        const bool gravity_on = has_gravity;
        const bool field_on = !source_expressions.empty();
        Kokkos::Array<rtype, N_DIM> g;
        FOR_I_DIM g[i] = gravity[i];
        StateView S = source_field;
        Kokkos::parallel_for("rhs_sources", mesh->n_cells, KOKKOS_LAMBDA(const uint32_t i_cell) {
            const rtype V = vol(i_cell);
            if (gravity_on) {
                rtype rhou[N_DIM];
                FOR_I_DIM {
                    rhs(i_cell, 1 + i) += solution(i_cell, 0) * g[i] * V;
                    rhou[i] = solution(i_cell, 1 + i);
                }
                rhs(i_cell, N_DIM + 1) += dot<N_DIM>(rhou, g.data()) * V;
            }
            if (field_on) {
                FOR_I_CONSERVATIVE rhs(i_cell, i) += S(i_cell, i) * V;
            }
        });
    }

    Kokkos::parallel_for("rhs_divide_volume", mesh->n_cells, KOKKOS_LAMBDA(const uint32_t i_cell) {
        FOR_I_CONSERVATIVE rhs(i_cell, i) /= vol(i_cell);
    });
}

template <typename T_riemann_solver>
void Solver::launch_flux_functor() {
    ConvectiveFluxFunctor<T_riemann_solver> functor{mesh->face_normals,
                                                    mesh->face_area,
                                                    mesh->cells_of_face,
                                                    face_reconstruction->quadrature_face.weights,
                                                    face_reconstruction->face_quad_weights,
                                                    face_solution,
                                                    boundary_data,
                                                    W_cells,
                                                    face_flux,
                                                    physics.gamma};
    Kokkos::parallel_for("convective_flux", mesh->n_faces, functor);
}

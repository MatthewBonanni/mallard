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

#include <Kokkos_Core.hpp>

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

void BoundarySymmetry::init(const toml::value & input) {
    (void)(input);
    print();
}

template <typename T_physics, typename T_riemann_solver>
void BoundarySymmetry::SymmetryFluxFunctor<T_physics, T_riemann_solver>::calc_lr_states_impl(const uint32_t i_face,
                                                                                             const uint8_t i_quad,
                                                                                             rtype * conservatives_l,
                                                                                             rtype * conservatives_r,
                                                                                             rtype * primitives_l,
                                                                                             rtype * primitives_r) const {
    rtype u_n;
    rtype u_l[N_DIM];
    rtype u_r[N_DIM];
    rtype n_vec[N_DIM];
    rtype n_unit[N_DIM];

    FOR_I_DIM n_vec[i] = this->normals(i_face, i);
    unit<N_DIM>(n_vec, n_unit);

    FOR_I_CONSERVATIVE {
        conservatives_l[i] = this->face_solution(i_face, i_quad, 0, i);
        conservatives_r[i] = conservatives_l[i]; // Only density will be used
    }

    this->physics.compute_primitives_from_conservatives(primitives_l, conservatives_l);

    // Right state = left state, but reflect velocity vector
    FOR_I_PRIMITIVE primitives_r[i] = primitives_l[i];
    u_l[0] = primitives_l[0];
    u_l[1] = primitives_l[1];
    u_n = dot<N_DIM>(u_l, n_unit);
    FOR_I_DIM u_r[i] = u_l[i] - 2.0 * u_n * n_unit[i];
    primitives_r[0] = u_r[0];
    primitives_r[1] = u_r[1];
}

template <typename T_physics, typename T_riemann_solver>
void BoundarySymmetry::launch_flux_functor(Kokkos::View<rtype **[2][N_CONSERVATIVE]> face_solution,
                                           Kokkos::View<rtype *[N_CONSERVATIVE]> rhs) {
    SymmetryFluxFunctor<T_physics, T_riemann_solver> flux_functor(zone->faces,
                                                                  mesh->face_normals,
                                                                  mesh->face_area,
                                                                  mesh->cells_of_face,
                                                                  face_quad_weights,
                                                                  face_solution,
                                                                  rhs,
                                                                  *physics->get_as<T_physics>(),
                                                                  dynamic_cast<T_riemann_solver &>(*riemann_solver));
    Kokkos::parallel_for(zone->n_faces(), flux_functor);
}

void BoundarySymmetry::apply(Kokkos::View<rtype **[2][N_CONSERVATIVE]> face_solution,
                             Kokkos::View<rtype *[N_CONSERVATIVE]> rhs) {
    if (physics->get_type() == PhysicsType::EULER) {
        switch (riemann_solver->get_type()) {
            case RiemannSolverType::RUSANOV:
                launch_flux_functor<Euler, Rusanov>(face_solution, rhs);
                break;
            case RiemannSolverType::HLL:
                launch_flux_functor<Euler, HLL>(face_solution, rhs);
                break;
            case RiemannSolverType::HLLC:
                launch_flux_functor<Euler, HLLC>(face_solution, rhs);
                break;
            default:
                throw std::runtime_error("Unknown Riemann solver type.");
        }
    } else {
        throw std::runtime_error("Unknown physics type.");
    }
}
/**
 * @file boundary_wall_adiabatic.cpp
 * @author Matthew Bonanni (mbonanni001@gmail.com)
 * @brief Adiabatic wall boundary condition class implementation.
 * @version 0.1
 * @date 2023-12-20
 * 
 * @copyright Copyright (c) 2023 Matthew Bonanni
 * 
 */

#include "boundary_wall_adiabatic.h"

#include <iostream>

#include <Kokkos_Core.hpp>

#include "common.h"

BoundaryWallAdiabatic::BoundaryWallAdiabatic() {
    type = BoundaryType::WALL_ADIABATIC;
}

BoundaryWallAdiabatic::~BoundaryWallAdiabatic() {
    // Empty
}

void BoundaryWallAdiabatic::print() {
    Boundary::print();
    std::cout << LOG_SEPARATOR << std::endl;
}

void BoundaryWallAdiabatic::init(const toml::value & input) {
    (void)(input);
    print();
}

template <typename T_physics, typename T_riemann_solver>
void BoundaryWallAdiabatic::WallAdiabaticFluxFunctor<T_physics, T_riemann_solver>::call_impl(const u_int32_t i_face_local) const {
    rtype flux[N_CONSERVATIVE];
    rtype conservatives_l[N_CONSERVATIVE];
    rtype primitives_l[N_PRIMITIVE];
    rtype n_vec[N_DIM];
    rtype n_unit[N_DIM];

    u_int32_t i_face = this->faces(i_face_local);

    // Only need left state
    FOR_I_CONSERVATIVE conservatives_l[i] = this->face_solution(i_face, 0, i);
    this->physics.compute_primitives_from_conservatives(primitives_l, conservatives_l);

    FOR_I_DIM n_vec[i] = this->normals(i_face, i);
    unit<N_DIM>(n_vec, n_unit);

    // Compute flux
    flux[0] = 0.0;
    flux[1] = primitives_l[2] * n_unit[0];
    flux[2] = primitives_l[2] * n_unit[1];
    flux[3] = 0.0;

    // Add flux to RHS
    FOR_I_CONSERVATIVE Kokkos::atomic_add(&this->rhs(this->cells_of_face(i_face, 0), i),
                                          -this->face_area(i_face) * flux[i]);
}

template <typename T_physics, typename T_riemann_solver>
void BoundaryWallAdiabatic::launch_flux_functor(Kokkos::View<rtype *[2][N_CONSERVATIVE]> face_solution,
                                                Kokkos::View<rtype *[N_CONSERVATIVE]> rhs) {
    WallAdiabaticFluxFunctor<T_physics, T_riemann_solver> flux_functor(zone->faces,
                                                                       mesh->face_normals,
                                                                       mesh->face_area,
                                                                       mesh->cells_of_face,
                                                                       face_solution,
                                                                       rhs,
                                                                       *physics->get_as<T_physics>(),
                                                                       dynamic_cast<T_riemann_solver &>(*riemann_solver));
    Kokkos::parallel_for(zone->n_faces(), flux_functor);
}

void BoundaryWallAdiabatic::apply(Kokkos::View<rtype *[2][N_CONSERVATIVE]> face_solution,
                                  Kokkos::View<rtype *[N_CONSERVATIVE]> rhs) {
    if (physics->get_type() == PhysicsType::EULER) {
        switch (riemann_solver->get_type()) {
            case RiemannSolverType::Rusanov:
                launch_flux_functor<Euler, Rusanov>(face_solution, rhs);
                break;
            case RiemannSolverType::HLL:
                launch_flux_functor<Euler, HLL>(face_solution, rhs);
                break;
            case RiemannSolverType::HLLE:
                launch_flux_functor<Euler, HLLE>(face_solution, rhs);
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
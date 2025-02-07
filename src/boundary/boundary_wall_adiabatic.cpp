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
void BoundaryWallAdiabatic::WallAdiabaticFluxFunctor<T_physics, T_riemann_solver>::call_impl(const uint32_t i_face_local) const {
    const uint8_t n_quad = this->quad_weights.extent(0);
    rtype flux_temp[N_CONSERVATIVE];
    rtype flux[N_CONSERVATIVE];
    rtype conservatives_l[N_CONSERVATIVE];
    rtype primitives_l[N_PRIMITIVE];
    rtype n_vec[N_DIM];
    rtype n_unit[N_DIM];

    uint32_t i_face = this->faces(i_face_local);
    FOR_I_DIM n_vec[i] = this->normals(i_face, i);
    unit<N_DIM>(n_vec, n_unit);

    FOR_I_CONSERVATIVE flux[i] = 0.0;
    for (uint8_t i_quad = 0; i_quad < n_quad; i_quad++) {
        // Compute the flux at the quadrature point
        FOR_I_CONSERVATIVE conservatives_l[i] = this->face_solution(i_face, i_quad, 0, i);
        this->physics.compute_primitives_from_conservatives(primitives_l, conservatives_l);
        flux_temp[0] = 0.0;
        flux_temp[1] = primitives_l[2] * n_unit[0];
        flux_temp[2] = primitives_l[2] * n_unit[1];
        flux_temp[3] = 0.0;

        // Add this point's contribution to the integral
        FOR_I_CONSERVATIVE flux[i] += this->quad_weights(i_quad) * flux_temp[i];
    }
    FOR_I_CONSERVATIVE flux[i] *= 0.5;

    // Add flux to RHS
    FOR_I_CONSERVATIVE Kokkos::atomic_add(&this->rhs(this->cells_of_face(i_face, 0), i),
                                          -this->face_area(i_face) * flux[i]);
}

template <typename T_physics, typename T_riemann_solver>
void BoundaryWallAdiabatic::launch_flux_functor(Kokkos::View<rtype **[2][N_CONSERVATIVE]> face_solution,
                                                Kokkos::View<rtype *[N_CONSERVATIVE]> rhs) {
    WallAdiabaticFluxFunctor<T_physics, T_riemann_solver> flux_functor(zone->faces,
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

void BoundaryWallAdiabatic::apply(Kokkos::View<rtype **[2][N_CONSERVATIVE]> face_solution,
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
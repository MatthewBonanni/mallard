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

#include <Kokkos_Core.hpp>

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

template <typename T_physics, typename T_riemann_solver>
void BoundaryExtrapolation::ExtrapolationFluxFunctor<T_physics, T_riemann_solver>::calc_lr_states_impl(const u_int32_t i_face,
                                                                                                       rtype * conservatives_l,
                                                                                                       rtype * conservatives_r,
                                                                                                       rtype * primitives_l,
                                                                                                       rtype * primitives_r) const {
    FOR_I_CONSERVATIVE {
        conservatives_l[i] = this->face_solution(i_face, 0, i);
        conservatives_r[i] = conservatives_l[i];
    }

    this->physics.compute_primitives_from_conservatives(primitives_l, conservatives_l);

    FOR_I_PRIMITIVE {
        primitives_r[i] = primitives_l[i];
    }
}

template <typename T_physics, typename T_riemann_solver>
void BoundaryExtrapolation::launch_flux_functor(Kokkos::View<rtype *[2][N_CONSERVATIVE]> face_solution,
                                                Kokkos::View<rtype *[N_CONSERVATIVE]> rhs) {
    ExtrapolationFluxFunctor<T_physics, T_riemann_solver> flux_functor(zone->faces,
                                                                       mesh->face_normals,
                                                                       mesh->face_area,
                                                                       mesh->cells_of_face,
                                                                       face_solution,
                                                                       rhs,
                                                                       *physics->get_as<T_physics>(),
                                                                       dynamic_cast<T_riemann_solver &>(*riemann_solver));
    Kokkos::parallel_for(zone->n_faces(), flux_functor);
}

void BoundaryExtrapolation::apply(Kokkos::View<rtype *[2][N_CONSERVATIVE]> face_solution,
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
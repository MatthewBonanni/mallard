/**
 * @file boundary_upt.cpp
 * @author Matthew Bonanni (mbonanni001@gmail.com)
 * @brief UPT boundary condition class implementation.
 * @version 0.1
 * @date 2023-12-20
 * 
 * @copyright Copyright (c) 2023 Matthew Bonanni
 * 
 */

#include "boundary_upt.h"

#include <iostream>

#include <Kokkos_Core.hpp>
#include <toml.hpp>

#include "common.h"

BoundaryUPT::BoundaryUPT() {
    type = BoundaryType::UPT;
}

BoundaryUPT::~BoundaryUPT() {
    // Empty
}

void BoundaryUPT::print() {
    Boundary::print();
    std::cout << "Parameters:" << std::endl;
    std::cout << "> u: " << u_bc[0] << ", " << u_bc[1] << std::endl;
    std::cout << "> p: " << p_bc << std::endl;
    std::cout << "> T: " << T_bc << std::endl;
    std::cout << LOG_SEPARATOR << std::endl;
}

void BoundaryUPT::init(const toml::value & input) {
    if (!input.contains("u_in")) {
        throw std::runtime_error("Missing u for boundary: " + zone->get_name() + ".");
    }
    if (!input.contains("p_in")) {
        throw std::runtime_error("Missing p for boundary: " + zone->get_name() + ".");
    }
    if (!input.contains("T_in")) {
        throw std::runtime_error("Missing T for boundary: " + zone->get_name() + ".");
    }

    std::vector<rtype> u_in = toml::find<std::vector<rtype>>(input, "u_in");
    if (u_in.size() != N_DIM) {
        throw std::runtime_error("Invalid u for boundary: " + zone->get_name() + ".");
    }

    FOR_I_DIM u_bc[i] = u_in[i];
    p_bc = toml::find<rtype>(input, "p_in");
    T_bc = toml::find<rtype>(input, "T_in");

    rtype rho_bc = physics->h_get_density_from_pressure_temperature(p_bc, T_bc);
    rtype e_bc = physics->h_get_energy_from_temperature(T_bc);
    rtype h_bc = e_bc + p_bc / rho_bc;

    data_bc = Kokkos::View<rtype [N_PRIMITIVE+1]>("data_bc");
    h_data_bc = Kokkos::create_mirror_view(data_bc);

    h_data_bc(0) = rho_bc;
    h_data_bc(1) = u_bc[0];
    h_data_bc(2) = u_bc[1];
    h_data_bc(3) = p_bc;
    h_data_bc(4) = T_bc;
    h_data_bc(5) = h_bc;

    print();
}

void BoundaryUPT::copy_host_to_device() {
    Kokkos::deep_copy(data_bc, h_data_bc);
}

void BoundaryUPT::copy_device_to_host() {
    Kokkos::deep_copy(h_data_bc, data_bc);
}

template <typename T_physics, typename T_riemann_solver>
void BoundaryUPT::UPTFluxFunctor<T_physics, T_riemann_solver>::calc_lr_states_impl(const u_int32_t i_face,
                                                                                   rtype * conservatives_l,
                                                                                   rtype * conservatives_r,
                                                                                   rtype * primitives_l,
                                                                                   rtype * primitives_r) const {
    FOR_I_CONSERVATIVE {
        conservatives_l[i] = this->face_solution(i_face, 0, i);
    }

    this->physics.compute_primitives_from_conservatives(primitives_l, conservatives_l);

    conservatives_r[0] = this->data_bc(0);
    FOR_I_PRIMITIVE {
        primitives_r[i] = this->data_bc(i+i);
    }
}

template <typename T_physics, typename T_riemann_solver>
void BoundaryUPT::launch_flux_functor(Kokkos::View<rtype *[2][N_CONSERVATIVE]> face_solution,
                                      Kokkos::View<rtype *[N_CONSERVATIVE]> rhs) {
    UPTFluxFunctor<T_physics, T_riemann_solver> flux_functor(zone->faces,
                                                             mesh->face_normals,
                                                             mesh->face_area,
                                                             mesh->cells_of_face,
                                                             face_solution,
                                                             rhs,
                                                             *physics->get_as<T_physics>(),
                                                             dynamic_cast<T_riemann_solver &>(*riemann_solver),
                                                             data_bc);
    Kokkos::parallel_for(zone->n_faces(), flux_functor);
}

void BoundaryUPT::apply(Kokkos::View<rtype *[2][N_CONSERVATIVE]> face_solution,
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
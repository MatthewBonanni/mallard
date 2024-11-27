/**
 * @file boundary_p_out.cpp
 * @author Matthew Bonanni (mbonanni001@gmail.com)
 * @brief Outflow pressure boundary condition class implementation.
 * @version 0.1
 * @date 2023-12-20
 * 
 * @copyright Copyright (c) 2023 Matthew Bonanni
 * 
 */

#include "boundary_p_out.h"

#include <iostream>

#include <Kokkos_Core.hpp>
#include <toml.hpp>

#include "common.h"

BoundaryPOut::BoundaryPOut() {
    type = BoundaryType::P_OUT;
}

BoundaryPOut::~BoundaryPOut() {
    // Empty
}

void BoundaryPOut::print() {
    Boundary::print();
    std::cout << "Parameters:" << std::endl;
    std::cout << "> p: " << h_data_bc(0) << std::endl;
    std::cout << LOG_SEPARATOR << std::endl;
}

void BoundaryPOut::init(const toml::value & input) {
    if (!input.contains("p")) {
        throw std::runtime_error("Missing p for boundary: " + zone->get_name() + ".");
    }

    data_bc = Kokkos::View<rtype [1]>("data_bc");
    h_data_bc = Kokkos::create_mirror_view(data_bc);
    
    h_data_bc(0) = toml::find<rtype>(input, "initialize", "p");

    print();
}

void BoundaryPOut::copy_host_to_device() {
    Kokkos::deep_copy(data_bc, h_data_bc);
}

void BoundaryPOut::copy_device_to_host() {
    Kokkos::deep_copy(h_data_bc, data_bc);
}

template <typename T_physics, typename T_riemann_solver>
void BoundaryPOut::POutFluxFunctor<T_physics, T_riemann_solver>::calc_lr_states_impl(const u_int32_t i_face,
                                                                                     rtype * conservatives_l,
                                                                                     rtype * conservatives_r,
                                                                                     rtype * primitives_l,
                                                                                     rtype * primitives_r) const {
    FOR_I_CONSERVATIVE {
        conservatives_l[i] = this->face_solution(i_face, 0, i);
    }

    this->physics.compute_primitives_from_conservatives(primitives_l, conservatives_l);

    // Determine if subsonic or supersonic
    const rtype rho_l = conservatives_l[0];
    rtype u_l[N_DIM];
    FOR_I_DIM u_l[i] = primitives_l[i];
    const rtype p_l = primitives_l[2];
    const rtype T_l = primitives_l[3];
    const rtype u_mag_l = norm_2<N_DIM>(u_l);
    const rtype sos_l = this->physics.get_sound_speed_from_pressure_density(p_l, rho_l);
    rtype p_out;
    if (u_mag_l < sos_l) {
        p_out = data_bc(0); // Use the set boundary pressure
    } else {
        p_out = p_l; // Extrapolate pressure
    }

    // Extrapolate temperature and velocity, use these to calculate
    // the remaining primitive variables
    const rtype T_bc = T_l;
    rtype u_bc[N_DIM];
    FOR_I_DIM u_bc[i] = u_l[i];
    const rtype rho_bc = this->physics.get_density_from_pressure_temperature(p_out, T_bc);
    const rtype e_bc = this->physics.get_energy_from_temperature(T_bc);
    const rtype h_bc = e_bc + p_out / rho_bc;

    // Calculate right state
    conservatives_r[0] = rho_bc;
    FOR_I_DIM primitives_r[i] = u_bc[i];
    primitives_r[2] = p_out;
    primitives_r[3] = T_bc;
    primitives_r[4] = h_bc;
}

template <typename T_physics, typename T_riemann_solver>
void BoundaryPOut::launch_flux_functor(Kokkos::View<rtype *[2][N_CONSERVATIVE]> face_solution,
                                       Kokkos::View<rtype *[N_CONSERVATIVE]> rhs) {
    POutFluxFunctor<T_physics, T_riemann_solver> flux_functor(zone->faces,
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

void BoundaryPOut::apply(Kokkos::View<rtype *[2][N_CONSERVATIVE]> face_solution,
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
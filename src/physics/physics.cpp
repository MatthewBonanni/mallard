/**
 * @file physics.cpp
 * @author Matthew Bonanni (mbonanni001@gmail.com)
 * @brief Physics class implementation.
 * @version 0.1
 * @date 2023-12-27
 * 
 * @copyright Copyright (c) 2023 Matthew Bonanni
 * 
 */

#include "physics.h"

#include <iostream>

#include "common.h"

Physics::Physics() : p_bounds({-1e20, 1e20}) {
    // Empty
}

Physics::~Physics() {
    std::cout << "Destroying physics: " << PHYSICS_NAMES.at(type) << std::endl;
}

void Physics::init(const toml::table & input) {
    rtype p_min = input["physics"]["p_min"].value_or(-1e20);
    rtype p_max = input["physics"]["p_max"].value_or(1e20);
    p_bounds = {p_min, p_max};
}

void Physics::print() const {
    std::cout << LOG_SEPARATOR << std::endl;
    std::cout << "Physics: " << PHYSICS_NAMES.at(type) << std::endl;
    std::cout << "> p_min: " << p_bounds[0] << std::endl;
    std::cout << "> p_max: " << p_bounds[1] << std::endl;
}

Euler::Euler() {
    type = PhysicsType::EULER;
}

Euler::~Euler() {
    // Empty
}

void Euler::init(const toml::table & input) {
    Physics::init(input);

    std::optional<rtype> gamma_in = input["physics"]["gamma"].value<rtype>();
    std::optional<rtype> p_ref_in = input["physics"]["p_ref"].value<rtype>();
    std::optional<rtype> T_ref_in = input["physics"]["T_ref"].value<rtype>();
    std::optional<rtype> rho_ref_in = input["physics"]["rho_ref"].value<rtype>();

    if (!gamma_in.has_value()) {
        throw std::runtime_error("Missing gamma for physics: " + PHYSICS_NAMES.at(type) + ".");
    }
    if (!p_ref_in.has_value()) {
        throw std::runtime_error("Missing p_ref for physics: " + PHYSICS_NAMES.at(type) + ".");
    }
    if (!T_ref_in.has_value()) {
        throw std::runtime_error("Missing T_ref for physics: " + PHYSICS_NAMES.at(type) + ".");
    }
    if (!rho_ref_in.has_value()) {
        throw std::runtime_error("Missing rho_ref for physics: " + PHYSICS_NAMES.at(type) + ".");
    }

    gamma = gamma_in.value();
    p_ref = p_ref_in.value();
    T_ref = T_ref_in.value();
    rho_ref = rho_ref_in.value();

    set_R_cp_cv();

    print();
}

void Euler::init(const rtype & gamma_in,
                 const rtype & p_ref_in,
                 const rtype & T_ref_in,
                 const rtype & rho_ref_in) {
    gamma = gamma_in;
    p_ref = p_ref_in;
    T_ref = T_ref_in;
    rho_ref = rho_ref_in;

    set_R_cp_cv();
}

void Euler::set_R_cp_cv() {
    R = p_ref / (T_ref * rho_ref);
    cp = R * gamma / (gamma - 1.0);
    cv = cp / gamma;
}

void Euler::print() const {
    Physics::print();
    std::cout << "> gamma: " << gamma << std::endl;
    std::cout << "> p_ref: " << p_ref << std::endl;
    std::cout << "> T_ref: " << T_ref << std::endl;
    std::cout << "> rho_ref: " << rho_ref << std::endl;
    std::cout << LOG_SEPARATOR << std::endl;
}

KOKKOS_INLINE_FUNCTION
rtype Euler::get_energy_from_temperature(const rtype & T) const {
    return cv * T;
}

KOKKOS_INLINE_FUNCTION
rtype Euler::get_temperature_from_energy(const rtype & e) const {
    return e / cv;
}

KOKKOS_INLINE_FUNCTION
rtype Euler::get_density_from_pressure_temperature(const rtype & p,
                                                   const rtype & T) const {
    return p / (T * R);
}

KOKKOS_INLINE_FUNCTION
rtype Euler::get_temperature_from_density_pressure(const rtype & rho,
                                                   const rtype & p) const {
    return p / (rho * R);
}

KOKKOS_INLINE_FUNCTION
rtype Euler::get_pressure_from_density_temperature(const rtype & rho,
                                                   const rtype & T) const {
    return rho * R * T;
}

KOKKOS_INLINE_FUNCTION
rtype Euler::get_pressure_from_density_energy(const rtype & rho,
                                              const rtype & e) const {
    return Kokkos::fmax(p_bounds[0], Kokkos::fmin(p_bounds[1], (gamma - 1.0) * rho * e));
}

KOKKOS_INLINE_FUNCTION
rtype Euler::get_sound_speed_from_pressure_density(const rtype & p,
                                                   const rtype & rho) const {
    return Kokkos::sqrt(gamma * p / rho);
}

KOKKOS_INLINE_FUNCTION
void Euler::compute_primitives_from_conservatives(Primitives & primitives,
                                                  const State & conservatives) const {
    rtype rho = conservatives[0];
    NVector u = {conservatives[1] / rho,
                 conservatives[2] / rho};
    rtype E = conservatives[3] / rho;
    rtype e = E - 0.5 * dot<N_DIM>(u.data(), u.data());
    rtype p = get_pressure_from_density_energy(rho, e);
    rtype T = get_temperature_from_energy(e);
    rtype h = e + p / rho;
    primitives[0] = u[0];
    primitives[1] = u[1];
    primitives[2] = p;
    primitives[3] = T;
    primitives[4] = h;
}

void Euler::calc_diffusive_flux(State & flux) {
    flux[0] = 0.0;
    flux[1] = 0.0;
    flux[2] = 0.0;
    flux[3] = 0.0;
}
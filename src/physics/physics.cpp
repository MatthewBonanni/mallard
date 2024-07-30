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

Physics::Physics() {
    p_bounds = Kokkos::View<rtype [2]>("p_bounds");
    h_p_bounds = Kokkos::create_mirror_view(p_bounds);
}

Physics::~Physics() {
    // Empty
}

void Physics::init(const toml::value & input) {
    h_p_bounds(0) = toml::find_or<rtype>(input, "physics", "p_min", -1e20);
    h_p_bounds(1) = toml::find_or<rtype>(input, "physics", "p_max", 1e20);
}

void Physics::print() const {
    std::cout << LOG_SEPARATOR << std::endl;
    std::cout << "Physics: " << PHYSICS_NAMES.at(type) << std::endl;
    std::cout << "> p_min: " << h_p_bounds(0) << std::endl;
    std::cout << "> p_max: " << h_p_bounds(1) << std::endl;
}

void Physics::copy_host_to_device() {
    Kokkos::deep_copy(p_bounds, h_p_bounds);
}

void Physics::copy_device_to_host() {
    Kokkos::deep_copy(h_p_bounds, p_bounds);
}

Euler::Euler() {
    type = PhysicsType::EULER;

    constants = Kokkos::View<rtype [7]>("constants");
    h_constants = Kokkos::create_mirror_view(constants);
}

Euler::~Euler() {
    // Empty
}

void Euler::init(const toml::value & input) {
    Physics::init(input);

    toml::value input_physics = toml::find(input, "physics");

    if (!input_physics.contains("gamma")) {
        throw std::runtime_error("Missing gamma for physics: " + PHYSICS_NAMES.at(type) + ".");
    }
    if (!input_physics.contains("p_ref")) {
        throw std::runtime_error("Missing p_ref for physics: " + PHYSICS_NAMES.at(type) + ".");
    }
    if (!input_physics.contains("T_ref")) {
        throw std::runtime_error("Missing T_ref for physics: " + PHYSICS_NAMES.at(type) + ".");
    }
    if (!input_physics.contains("rho_ref")) {
        throw std::runtime_error("Missing rho_ref for physics: " + PHYSICS_NAMES.at(type) + ".");
    }

    h_constants(i_gamma) = toml::find<rtype>(input, "physics", "gamma");
    h_constants(i_p_ref) = toml::find<rtype>(input, "physics", "p_ref");
    h_constants(i_T_ref) = toml::find<rtype>(input, "physics", "T_ref");
    h_constants(i_rho_ref) = toml::find<rtype>(input, "physics", "rho_ref");

    set_R_cp_cv();

    print();
}


// FOR TESTING PURPOSES ONLY
void Euler::init(rtype p_min, rtype p_max, rtype gamma,
                 rtype p_ref, rtype T_ref, rtype rho_ref) {
    h_p_bounds(0) = p_min;
    h_p_bounds(1) = p_max;
    h_constants(i_gamma) = gamma;
    h_constants(i_p_ref) = p_ref;
    h_constants(i_T_ref) = T_ref;
    h_constants(i_rho_ref) = rho_ref;

    set_R_cp_cv();
}

void Euler::set_R_cp_cv() {
    h_constants(i_R) = h_constants(i_p_ref) / (h_constants(i_T_ref) * h_constants(i_rho_ref));
    h_constants(i_cp) = h_constants(i_R) * h_constants(i_gamma) / (h_constants(i_gamma) - 1.0);
    h_constants(i_cv) = h_constants(i_cp) / h_constants(i_gamma);
}

void Euler::print() const {
    Physics::print();
    std::cout << "> gamma: " << h_constants(i_gamma) << std::endl;
    std::cout << "> p_ref: " << h_constants(i_p_ref) << std::endl;
    std::cout << "> T_ref: " << h_constants(i_T_ref) << std::endl;
    std::cout << "> rho_ref: " << h_constants(i_rho_ref) << std::endl;
    std::cout << LOG_SEPARATOR << std::endl;
}

void Euler::copy_host_to_device() {
    Physics::copy_host_to_device();

    Kokkos::deep_copy(constants, h_constants);
}

void Euler::copy_device_to_host() {
    Physics::copy_device_to_host();

    Kokkos::deep_copy(h_constants, constants);
}
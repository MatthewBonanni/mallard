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
    p_bounds = Kokkos::View<rtype[2]>("p_bounds");
    h_p_bounds = Kokkos::create_mirror_view(p_bounds);
}

Physics::~Physics() {
    std::cout << "Destroying physics: " << PHYSICS_NAMES.at(type) << std::endl;
}

void Physics::init(const toml::table & input) {
    h_p_bounds(0) = input["physics"]["p_min"].value_or(-1e20);
    h_p_bounds(1) = input["physics"]["p_max"].value_or(1e20);
    Kokkos::deep_copy(p_bounds, h_p_bounds);
}

void Physics::print() const {
    std::cout << LOG_SEPARATOR << std::endl;
    std::cout << "Physics: " << PHYSICS_NAMES.at(type) << std::endl;
    std::cout << "> p_min: " << h_p_bounds(0) << std::endl;
    std::cout << "> p_max: " << h_p_bounds(1) << std::endl;
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


// FOR TESTING PURPOSES ONLY
void Euler::init(rtype p_min, rtype p_max, rtype gamma,
                 rtype p_ref, rtype T_ref, rtype rho_ref) {
    h_p_bounds(0) = p_min;
    h_p_bounds(1) = p_max;
    Kokkos::deep_copy(p_bounds, h_p_bounds);
    this->gamma = gamma;
    this->p_ref = p_ref;
    this->T_ref = T_ref;
    this->rho_ref = rho_ref;

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
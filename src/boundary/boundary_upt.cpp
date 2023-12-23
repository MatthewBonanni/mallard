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

#include <toml++/toml.h>

#include "common/common.h"

BoundaryUPT::BoundaryUPT() {
    type = BoundaryType::UPT;
}

BoundaryUPT::~BoundaryUPT() {
    // Empty
}

void BoundaryUPT::print() {
    Boundary::print();
    std::cout << "> u: " << u[0] << ", " << u[1] << std::endl;
    std::cout << "> p: " << p << std::endl;
    std::cout << "> T: " << T << std::endl;
    std::cout << LOG_SEPARATOR << std::endl;
}

void BoundaryUPT::init(const toml::table& input) {
    auto u_in = input["u"];
    const toml::array* arr = u_in.as_array();
    std::optional<double> p_in = input["p"].value<double>();
    std::optional<double> T_in = input["T"].value<double>();

    if (!u_in) {
        throw std::runtime_error("Missing u for boundary: " + zone->get_name() + ".");
    } else if (arr->size() != 2) {
        throw std::runtime_error("u must be a 2-element array for boundary: " + zone->get_name() + ".");
    }

    if (!p_in.has_value()) {
        throw std::runtime_error("Missing p for boundary: " + zone->get_name() + ".");
    }

    if (!T_in.has_value()) {
        throw std::runtime_error("Missing T for boundary: " + zone->get_name() + ".");
    }

    auto u_x = arr->get_as<double>(0);
    auto u_y = arr->get_as<double>(1);
    u[0] = u_x->as_floating_point()->get();
    u[1] = u_y->as_floating_point()->get();
    p = p_in.value();
    T = T_in.value();

    print();
}
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

#include <toml++/toml.h>

#include "common/common.h"

BoundaryPOut::BoundaryPOut() {
    type = BoundaryType::P_OUT;
}

BoundaryPOut::~BoundaryPOut() {
    // Empty
}

void BoundaryPOut::print() {
    Boundary::print();
    std::cout << "> p: " << p << std::endl;
    std::cout << LOG_SEPARATOR << std::endl;
}

void BoundaryPOut::init(const toml::table & input) {
    std::optional<double> p_in = input["p"].value<double>();

    if (!p_in.has_value()) {
        throw std::runtime_error("Missing p for boundary: " + zone->get_name() + ".");
    }

    p = p_in.value();

    print();
}
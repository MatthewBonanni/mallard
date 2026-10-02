/**
 * @file physics.cpp
 * @author Matthew Bonanni (mbonanni001@gmail.com)
 * @brief Physics model implementation.
 * @version 0.2
 * @date 2023-12-27
 *
 * @copyright Copyright (c) 2023 Matthew Bonanni
 *
 */

#include "physics.h"

#include <iostream>
#include <stdexcept>

Euler Euler::from_reference(rtype gamma, rtype p_ref, rtype T_ref, rtype rho_ref) {
    Euler euler;
    euler.gamma = gamma;
    euler.R = p_ref / (T_ref * rho_ref);
    euler.cp = euler.R * gamma / (gamma - 1.0);
    euler.cv = euler.cp / gamma;
    return euler;
}

Euler Euler::from_input(const toml::value & input) {
    for (const char * key : {"gamma", "p_ref", "T_ref", "rho_ref"}) {
        if (!input.at("physics").contains(key)) {
            throw std::runtime_error(std::string("Missing ") + key + " for physics: euler.");
        }
    }
    Euler euler = from_reference(toml::find<rtype>(input, "physics", "gamma"),
                                 toml::find<rtype>(input, "physics", "p_ref"),
                                 toml::find<rtype>(input, "physics", "T_ref"),
                                 toml::find<rtype>(input, "physics", "rho_ref"));
    euler.print();
    return euler;
}

void Euler::print() const {
    std::cout << LOG_SEPARATOR << std::endl;
    std::cout << "Physics: " << PHYSICS_NAMES.at(get_type()) << std::endl;
    std::cout << "> gamma: " << gamma << std::endl;
    std::cout << "> R: " << R << std::endl;
    std::cout << "> cp: " << cp << std::endl;
    std::cout << "> cv: " << cv << std::endl;
    std::cout << LOG_SEPARATOR << std::endl;
}

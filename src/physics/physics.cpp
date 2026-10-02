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
    const std::string type = toml::find_or<std::string>(input, "physics", "type", "euler");
    if (type == "navier_stokes") {
        if (!input.at("physics").contains("mu")) {
            throw std::runtime_error("Missing mu for physics: navier_stokes.");
        }
        euler.mu_ref = toml::find<rtype>(input, "physics", "mu");
        euler.Pr = toml::find_or<rtype>(input, "physics", "Pr", 0.72);
        const std::string model = toml::find_or<std::string>(input, "physics", "viscosity_model", "constant");
        if (model == "constant") {
            euler.viscosity_model = ViscosityModel::CONSTANT;
        } else if (model == "sutherland") {
            euler.viscosity_model = ViscosityModel::SUTHERLAND;
            euler.T_mu_ref = toml::find_or<rtype>(input, "physics", "T_mu_ref", 273.15);
            euler.S_mu = toml::find_or<rtype>(input, "physics", "sutherland_S", 110.4);
        } else {
            throw std::runtime_error("Unknown viscosity model: " + model + ".");
        }
    } else if (type != "euler") {
        throw std::runtime_error("Unknown physics type: " + type + ".");
    }
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
    if (is_viscous()) {
        std::cout << "> Viscosity model: " << (viscosity_model == ViscosityModel::CONSTANT ? "constant" : "sutherland") << std::endl;
        std::cout << "> mu: " << mu_ref << std::endl;
        std::cout << "> Pr: " << Pr << std::endl;
    }
    std::cout << LOG_SEPARATOR << std::endl;
}

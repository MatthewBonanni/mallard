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

#include "input.h"

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
    Euler euler = from_reference(find_real(input, "physics", "gamma"),
                                 find_real(input, "physics", "p_ref"),
                                 find_real(input, "physics", "T_ref"),
                                 find_real(input, "physics", "rho_ref"));
    const std::string type = toml::find_or<std::string>(input, "physics", "type", "euler");
    if (type == "navier_stokes") {
        if (!input.at("physics").contains("mu")) {
            throw std::runtime_error("Missing mu for physics: navier_stokes.");
        }
        euler.mu_ref = find_real(input, "physics", "mu");
        euler.Pr = find_real_or(input, "physics", "Pr", 0.72);
        const std::string model = toml::find_or<std::string>(input, "physics", "viscosity_model", "constant");
        if (model == "constant") {
            euler.viscosity_model = ViscosityModel::CONSTANT;
        } else if (model == "sutherland") {
            euler.viscosity_model = ViscosityModel::SUTHERLAND;
            euler.T_mu_ref = find_real_or(input, "physics", "T_mu_ref", 273.15);
            euler.S_mu = find_real_or(input, "physics", "sutherland_S", 110.4);
        } else {
            throw InputError("physics.viscosity_model = \"" + model + "\" is not one of: constant, sutherland.");
        }
    } else if (type != "euler") {
        throw unknown_option(PHYSICS_TYPES, "physics.type", type);
    }
    return euler;
}

logging::Items Euler::summary() const {
    using logging::real;
    logging::Items out = {
        {"Model", is_viscous() ? "Navier-Stokes" : "Euler"},
        {"Gas", "gamma " + real(gamma) + ", R " + real(R) + ", cp " + real(cp) + ", cv " + real(cv)},
    };
    if (is_viscous()) {
        if (viscosity_model == ViscosityModel::SUTHERLAND) {
            out.emplace_back("Viscosity", "Sutherland, mu " + real(mu_ref) + " at T " + real(T_mu_ref) + ", S " +
                                              real(S_mu) + ", Pr " + real(Pr));
        } else {
            out.emplace_back("Viscosity", "constant, mu " + real(mu_ref) + ", Pr " + real(Pr));
        }
    }
    return out;
}

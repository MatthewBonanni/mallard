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
    euler.cp = euler.R * gamma / (gamma - 1.0_r);
    euler.cv = euler.cp / gamma;
    return euler;
}

Euler Euler::from_input(const Input & input) {
    for (const char * key : {"gamma", "p_ref", "T_ref", "rho_ref"}) {
        if (!input.at("physics").contains(key)) {
            throw missing_key(input.at("physics"), std::string("physics.") + key,
                              "A perfect gas needs gamma, p_ref, T_ref and rho_ref (e.g. air: gamma = 1.4, "
                              "p_ref = \"1 atm\", T_ref = \"288.15 K\", rho_ref = 1.225)");
        }
    }
    Euler euler = from_reference(find_real(input, "physics", "gamma", dim::none),
                                 find_real(input, "physics", "p_ref", dim::pressure),
                                 find_real(input, "physics", "T_ref", dim::temperature),
                                 find_real(input, "physics", "rho_ref", dim::density));
    const std::string type = find_value_or<std::string>(input, "physics", "type", "euler");
    find_option(PHYSICS_TYPES, "physics.type", type, line_of(input.at("physics"), "type"));
    if (type == "navier_stokes") {
        if (!input.at("physics").contains("mu")) {
            throw std::runtime_error("Missing mu for physics: navier_stokes.");
        }
        euler.mu_ref = find_real(input, "physics", "mu", dim::dynamic_viscosity);
        euler.Pr = find_real_or(input, "physics", "Pr", dim::none, 0.72_r);
        const std::string model = find_value_or<std::string>(input, "physics", "viscosity_model", "constant");
        record_options("physics.viscosity_model", {"constant", "power_law", "sutherland"});
        if (model == "constant") {
            euler.viscosity_model = ViscosityModel::CONSTANT;
        } else if (model == "sutherland") {
            euler.viscosity_model = ViscosityModel::SUTHERLAND;
            euler.T_mu_ref = find_real_or(input, "physics", "T_mu_ref", dim::temperature, 273.15_r);
            euler.S_mu = find_real_or(input, "physics", "sutherland_S", dim::temperature_interval, 110.4_r);
        } else if (model == "power_law") {
            for (const char * key : {"T_mu_ref", "viscosity_exponent"}) {
                if (!input.at("physics").contains(key)) {
                    throw InputError(std::string("Missing physics.") + key + " for viscosity_model = \"power_law\".");
                }
            }
            euler.viscosity_model = ViscosityModel::POWER_LAW;
            euler.T_mu_ref = find_real(input, "physics", "T_mu_ref", dim::temperature);
            euler.n_mu = find_real(input, "physics", "viscosity_exponent", dim::none);
        } else {
            throw InputError("physics.viscosity_model = \"" + model + "\" is not one of: constant, sutherland, power_law.");
        }
    }
    return euler;
}

report::Summary Euler::summary() const {
    using logging::real;
    const std::string specific_heat = units::column("J/(kg K)");
    report::Summary s;
    s.setting("Model", is_viscous() ? "Navier-Stokes" : "Euler");
    s.setting("Gas", "perfect gas");
    s.quantity("gamma", real(double(gamma)), "-", "ratio of specific heats");
    s.quantity("R", real(double(R)), specific_heat, "gas constant");
    s.quantity("cp", real(double(cp)), specific_heat, "specific heat at constant pressure");
    s.quantity("cv", real(double(cv)), specific_heat, "specific heat at constant volume");
    if (!is_viscous()) return s;
    const std::string temperature = units::column("K");
    if (viscosity_model == ViscosityModel::SUTHERLAND) {
        s.setting("Viscosity", "Sutherland's law");
        s.quantity("mu", real(double(mu_ref)), units::column("Pa s"), "dynamic viscosity at T_mu_ref");
        s.quantity("T_mu_ref", real(double(T_mu_ref)), temperature, "reference temperature of the viscosity");
        s.quantity("sutherland_S", real(double(S_mu)), temperature, "Sutherland's constant");
    } else if (viscosity_model == ViscosityModel::POWER_LAW) {
        s.setting("Viscosity", "power law, mu (T / T_mu_ref)^viscosity_exponent");
        s.quantity("mu", real(double(mu_ref)), units::column("Pa s"), "dynamic viscosity at T_mu_ref");
        s.quantity("T_mu_ref", real(double(T_mu_ref)), temperature, "reference temperature of the viscosity");
        s.quantity("viscosity_exponent", real(double(n_mu)), "-", "");
    } else {
        s.setting("Viscosity", "constant");
        s.quantity("mu", real(double(mu_ref)), units::column("Pa s"), "dynamic viscosity");
    }
    s.quantity("Pr", real(double(Pr)), "-", "Prandtl number");
    return s;
}

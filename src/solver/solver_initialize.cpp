/**
 * @file solver_initialize.cpp
 * @author Matthew Bonanni (mbonanni001@gmail.com)
 * @brief Initialization methods for the Solver class.
 * @version 0.1
 * @date 2023-12-31
 * 
 * @copyright Copyright (c) 2023 Matthew Bonanni
 * 
 * 
 * \todo Move this to its own Initialization class.
 */

#include "solver.h"

#include <iostream>
#include <string>
#include <unordered_map>

enum class InitType {
    CONSTANT,
    ANALYTICAL
};

static const std::unordered_map<std::string, InitType> INIT_TYPES = {
    {"constant", InitType::CONSTANT},
    {"analytical", InitType::ANALYTICAL}
};

static const std::unordered_map<InitType, std::string> INIT_NAMES = {
    {InitType::CONSTANT, "constant"},
    {InitType::ANALYTICAL, "analytical"}
};

void Solver::init_solution() {
    std::cout << "Initializing solution..." << std::endl;

    std::string type_str = input["initialize"]["type"].value_or("constant");

    InitType type;
    typename std::unordered_map<std::string, InitType>::const_iterator it = INIT_TYPES.find(type_str);
    if (it == INIT_TYPES.end()) {
        throw std::runtime_error("Unknown initialization type: " + type_str + ".");
    } else {
        type = it->second;
    }

    if (type == InitType::CONSTANT) {
        init_solution_constant();
    } else if (type == InitType::ANALYTICAL) {
        init_solution_analytical();
    } else {
        // Should never get here due to the enum class.
        throw std::runtime_error("Unknown initialization type: " + type_str + ".");
    }
}

void Solver::init_solution_constant() {
    auto u_in = input["initialize"]["u"];
    const toml::array* arr = u_in.as_array();
    std::optional<rtype> p_in = input["initialize"]["p"].value<rtype>();
    std::optional<rtype> T_in = input["initialize"]["T"].value<rtype>();

    if (!u_in) {
        throw std::runtime_error("Missing u for initialization: constant.");
    } else if (arr->size() != 2) {
        throw std::runtime_error("u must be a 2-element array for initialization: constant.");
    }

    if (!p_in.has_value()) {
        throw std::runtime_error("Missing p for initialization: constant.");
    }

    if (!T_in.has_value()) {
        throw std::runtime_error("Missing T for initialization: constant.");
    }

    auto u_x_in = arr->get_as<double>(0);
    auto u_y_in = arr->get_as<double>(1);
    rtype u_x = u_x_in->as_floating_point()->get();
    rtype u_y = u_y_in->as_floating_point()->get();
    rtype p = p_in.value();
    rtype T = T_in.value();

    rtype rho = physics->get_density_from_pressure_temperature(p, T);
    rtype e = physics->get_energy_from_temperature(T);
    rtype E = e + 0.5 * (u_x * u_x +
                         u_y * u_y);
    rtype h = e + p / rho;
    rtype rhou_x = rho * u_x;
    rtype rhou_y = rho * u_y;
    rtype rhoE = rho * E;

    for (int i = 0; i < mesh->n_cells(); ++i) {
        conservatives(i, 0) = rho;
        conservatives(i, 1) = rhou_x;
        conservatives(i, 2) = rhou_y;
        conservatives(i, 3) = rhoE;

        primitives(i, 0) = u_x;
        primitives(i, 1) = u_y;
        primitives(i, 2) = p;
        primitives(i, 3) = T;
        primitives(i, 4) = h;
    }
}

void Solver::init_solution_analytical() {
    throw std::runtime_error("Analytical initialization not implemented.");
}
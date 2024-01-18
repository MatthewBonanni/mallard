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

#include "exprtk.hpp"

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
    auto u_in = input["initialize"]["u"];
    std::optional<std::string> p_in = input["initialize"]["p"].value<std::string>();
    std::optional<std::string> T_in = input["initialize"]["T"].value<std::string>();
    const toml::array* arr = u_in.as_array();

    if (!u_in) {
        throw std::runtime_error("Missing u for initialization: analytical.");
    } else if (arr->size() != 2) {
        throw std::runtime_error("u must be a 2-element array for initialization: analytical.");
    }

    if (!p_in.has_value()) {
        throw std::runtime_error("Missing p for initialization: analytical.");
    }

    if (!T_in.has_value()) {
        throw std::runtime_error("Missing T for initialization: analytical.");
    }

    std::string u_x_str = arr->get_as<std::string>(0)->get();
    std::string u_y_str = arr->get_as<std::string>(1)->get();
    std::string p_str = p_in.value();
    std::string T_str = T_in.value();

    rtype x, y;

    exprtk::symbol_table<rtype> symbol_table;
    symbol_table.add_variable("x", x);
    symbol_table.add_variable("y", y);
    symbol_table.add_constants();

    exprtk::expression<rtype> u_x_expr;
    exprtk::expression<rtype> u_y_expr;
    exprtk::expression<rtype> p_expr;
    exprtk::expression<rtype> T_expr;

    u_x_expr.register_symbol_table(symbol_table);
    u_y_expr.register_symbol_table(symbol_table);
    p_expr.register_symbol_table(symbol_table);
    T_expr.register_symbol_table(symbol_table);

    exprtk::parser<rtype> parser;
    parser.compile(u_x_str, u_x_expr);
    parser.compile(u_y_str, u_y_expr);
    parser.compile(p_str, p_expr);
    parser.compile(T_str, T_expr);

    rtype u_x, u_y, p, T;
    rtype rho, e, E, h, rhou_x, rhou_y, rhoE;

    for (int i = 0; i < mesh->n_cells(); ++i) {
        x = mesh->cell_coords(i)[0];
        y = mesh->cell_coords(i)[1];

        u_x = u_x_expr.value();
        u_y = u_y_expr.value();
        p = p_expr.value();
        T = T_expr.value();

        rho = physics->get_density_from_pressure_temperature(p, T);
        e = physics->get_energy_from_temperature(T);
        E = e + 0.5 * (u_x * u_x +
                       u_y * u_y);
        h = e + p / rho;
        rhou_x = rho * u_x;
        rhou_y = rho * u_y;
        rhoE = rho * E;

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
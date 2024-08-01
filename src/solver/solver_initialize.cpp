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

#include <exprtk.hpp>

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

    std::string type_str = toml::find_or(input, "initialize", "type", "constant");

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

    copy_host_to_device();
}

void Solver::init_solution_constant() {
    if (!input["initialize"].contains("u")) {
        throw std::runtime_error("Missing u for initialization: constant.");
    } else if (input["initialize"]["u"].size() != 2) {
        throw std::runtime_error("u must be a 2-element array for initialization: constant.");
    }

    if (!input["initialize"].contains("p")) {
        throw std::runtime_error("Missing p for initialization: constant.");
    }

    if (!input["initialize"].contains("T")) {
        throw std::runtime_error("Missing T for initialization: constant.");
    }

    std::vector<rtype> u = toml::find<std::vector<rtype>>(input, "initialize", "u");
    rtype p = toml::find<rtype>(input, "initialize", "p"); 
    rtype T = toml::find<rtype>(input, "initialize", "T");

    rtype rho = physics->get_density_from_pressure_temperature(p, T);
    rtype e = physics->get_energy_from_temperature(T);
    rtype E = e + 0.5 * norm_2<N_DIM>(u.data());
    rtype h = e + p / rho;
    rtype rhou_x = rho * u[0];
    rtype rhou_y = rho * u[1];
    rtype rhoE = rho * E;

    for (u_int32_t i = 0; i < mesh->n_cells(); ++i) {
        h_conservatives(i, 0) = rho;
        h_conservatives(i, 1) = rhou_x;
        h_conservatives(i, 2) = rhou_y;
        h_conservatives(i, 3) = rhoE;

        h_primitives(i, 0) = u[0];
        h_primitives(i, 1) = u[1];
        h_primitives(i, 2) = p;
        h_primitives(i, 3) = T;
        h_primitives(i, 4) = h;
    }
}

void Solver::init_solution_analytical() {
    if (!input["initialize"].contains("u")) {
        throw std::runtime_error("Missing u for initialization: analytical.");
    } else if (input["initialize"]["u"].size() != 2) {
        throw std::runtime_error("u must be a 2-element array for initialization: analytical.");
    }

    bool rho_in = input["initialize"].contains("rho");
    bool p_in = input["initialize"].contains("p");
    bool T_in = input["initialize"].contains("T");
    u_int8_t n_specified = rho_in + p_in + T_in;
    if (n_specified != 2) {
        throw std::runtime_error("Exactly two of rho, p, and T must be specified for initialization: analytical.");
    }

    std::vector<std::string> u_str = toml::find<std::vector<std::string>>(input, "initialize", "u");
    std::string u_x_str = u_str[0];
    std::string u_y_str = u_str[1];
    std::string rho_str = toml::find_or(input, "initialize", "rho", "");
    std::string p_str = toml::find_or(input, "initialize", "p", "");
    std::string T_str = toml::find_or(input, "initialize", "T", "");

    rtype x, y;

    exprtk::symbol_table<rtype> symbol_table;
    symbol_table.add_variable("x", x);
    symbol_table.add_variable("y", y);
    symbol_table.add_constants();

    exprtk::expression<rtype> u_x_expr, u_y_expr, rho_expr, p_expr, T_expr;

    u_x_expr.register_symbol_table(symbol_table);
    u_y_expr.register_symbol_table(symbol_table);
    if (rho_in) rho_expr.register_symbol_table(symbol_table);
    if (p_in) p_expr.register_symbol_table(symbol_table);
    if (T_in) T_expr.register_symbol_table(symbol_table);

    exprtk::parser<rtype> parser;
    parser.compile(u_x_str, u_x_expr);
    parser.compile(u_y_str, u_y_expr);
    if (rho_in) parser.compile(rho_str, rho_expr);
    if (p_in) parser.compile(p_str, p_expr);
    if (T_in) parser.compile(T_str, T_expr);

    NVector u;
    rtype p, T;
    rtype rho, e, E, h, rhou_x, rhou_y, rhoE;

    for (u_int32_t i = 0; i < mesh->n_cells(); ++i) {
        x = mesh->h_cell_coords(i, 0);
        y = mesh->h_cell_coords(i, 1);

        u[0] = u_x_expr.value();
        u[1] = u_y_expr.value();
        if (rho_in) rho = rho_expr.value();
        if (p_in) p = p_expr.value();
        if (T_in) T = T_expr.value();

        if (rho_in && p_in) {
            T = physics->get_temperature_from_density_pressure(rho, p);
        } else if (rho_in && T_in) {
            p = physics->get_pressure_from_density_temperature(rho, T);
        } else if (p_in && T_in) {
            rho = physics->get_density_from_pressure_temperature(p, T);
        } else {
            throw std::runtime_error("Exactly two of rho, p, and T must be specified for initialization: analytical.");
        }

        e = physics->get_energy_from_temperature(T);
        E = e + 0.5 * norm_2<N_DIM>(u.data());
        h = e + p / rho;
        rhou_x = rho * u[0];
        rhou_y = rho * u[1];
        rhoE = rho * E;

        h_conservatives(i, 0) = rho;
        h_conservatives(i, 1) = rhou_x;
        h_conservatives(i, 2) = rhou_y;
        h_conservatives(i, 3) = rhoE;

        h_primitives(i, 0) = u[0];
        h_primitives(i, 1) = u[1];
        h_primitives(i, 2) = p;
        h_primitives(i, 3) = T;
        h_primitives(i, 4) = h;
    }
}
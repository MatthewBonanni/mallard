/**
 * @file solver_initialize.cpp
 * @author Matthew Bonanni (mbonanni001@gmail.com)
 * @brief Implementation of solution initialization methods for the Solver class.
 * @version 0.2
 * @date 2024-01-11
 *
 * @copyright Copyright (c) 2024 Matthew Bonanni
 *
 */

#include "solver.h"

#include <iostream>
#include <string>
#include <unordered_map>

#include <exprtk.hpp>

#include "quadrature.h"

enum class InitType {
    CONSTANT,
    ANALYTICAL,
    RESTART
};

static const std::unordered_map<std::string, InitType> INIT_TYPES = {
    {"constant", InitType::CONSTANT},
    {"analytical", InitType::ANALYTICAL},
    {"restart", InitType::RESTART}
};

void Solver::init_solution() {
    std::cout << "Initializing solution..." << std::endl;
    const std::string type_str = toml::find_or<std::string>(input, "initialize", "type", "constant");
    auto it = INIT_TYPES.find(type_str);
    if (it == INIT_TYPES.end()) {
        throw std::runtime_error("Unknown initialization type: " + type_str + ".");
    }
    if (it->second == InitType::CONSTANT) {
        init_solution_constant();
    } else if (it->second == InitType::ANALYTICAL) {
        init_solution_analytical();
    } else {
        init_solution_restart();
    }
    copy_host_to_device();
    update_primitives();
}

void Solver::init_solution_restart() {
    if (!input.at("initialize").contains("file")) {
        throw std::runtime_error("Missing file for initialization: restart.");
    }
    const std::string file = toml::find<std::string>(input, "initialize", "file");
    RestartData restart = read_restart(file);
    if (restart.n_cells != mesh->n_cells || restart.conservatives.size() != N_CONSERVATIVE) {
        throw std::runtime_error("Restart file " + file + " does not match the mesh.");
    }
    for (uint32_t i_cell = 0; i_cell < mesh->n_cells; ++i_cell) {
        FOR_I_CONSERVATIVE h_conservatives(i_cell, i) = restart.conservatives[i][i_cell];
    }
    step = restart.step;
    t = restart.t;
    t_last_check = t;
    for (auto & writer : data_writers) {
        writer->resume(step, t);
    }
    std::cout << "Restarted from " << file << " at step " << step << ", t = " << t << std::endl;
}

void Solver::init_solution_constant() {
    const toml::value & init = input.at("initialize");
    for (const char * key : {"u", "p", "T"}) {
        if (!init.contains(key)) {
            throw std::runtime_error(std::string("Missing ") + key + " for initialization: constant.");
        }
    }
    std::vector<rtype> u = toml::find<std::vector<rtype>>(input, "initialize", "u");
    if (u.size() != N_DIM) {
        throw std::runtime_error("u must be a 2-element array for initialization: constant.");
    }
    const rtype p = toml::find<rtype>(input, "initialize", "p");
    const rtype T = toml::find<rtype>(input, "initialize", "T");
    const rtype W[N_CONSERVATIVE] = {physics.get_density_from_pressure_temperature(p, T), u[0], u[1], p};
    rtype cons[N_CONSERVATIVE];
    physics.compute_conservatives_from_W(cons, W);
    for (uint32_t i_cell = 0; i_cell < mesh->n_cells; ++i_cell) {
        FOR_I_CONSERVATIVE h_conservatives(i_cell, i) = cons[i];
    }
}

/**
 * Cell averages of the conservative variables are computed by splitting each
 * cell into a fan of triangles, subdividing each into n_sub^2 sub-triangles,
 * and applying a degree-5 Dunavant rule on every sub-triangle. This resolves
 * discontinuous initial data that do not align with the mesh and is
 * high-order accurate for smooth data.
 */
void Solver::init_solution_analytical() {
    const toml::value & init = input.at("initialize");
    if (!init.contains("u") || init.at("u").as_array().size() != N_DIM) {
        throw std::runtime_error("u must be a 2-element array for initialization: analytical.");
    }
    const bool rho_in = init.contains("rho");
    const bool p_in = init.contains("p");
    const bool T_in = init.contains("T");
    if (rho_in + p_in + T_in != 2) {
        throw std::runtime_error("Exactly two of rho, p, and T must be specified for initialization: analytical.");
    }
    const uint32_t n_sub = toml::find_or<uint32_t>(input, "initialize", "n_subdivisions", 4);

    std::vector<std::string> u_str = toml::find<std::vector<std::string>>(input, "initialize", "u");

    double x = 0.0, y = 0.0;
    exprtk::symbol_table<double> symbol_table;
    symbol_table.add_variable("x", x);
    symbol_table.add_variable("y", y);
    symbol_table.add_constants();
    exprtk::parser<double> parser;
    auto compile = [&](const std::string & name, const std::string & expr_str) {
        exprtk::expression<double> expr;
        expr.register_symbol_table(symbol_table);
        if (!parser.compile(expr_str, expr)) {
            throw std::runtime_error("Failed to parse initialization expression for " + name + ": " +
                                     parser.error());
        }
        return expr;
    };
    exprtk::expression<double> u_x_expr = compile("u[0]", u_str[0]);
    exprtk::expression<double> u_y_expr = compile("u[1]", u_str[1]);
    exprtk::expression<double> rho_expr, p_expr, T_expr;
    if (rho_in) rho_expr = compile("rho", toml::find<std::string>(input, "initialize", "rho"));
    if (p_in) p_expr = compile("p", toml::find<std::string>(input, "initialize", "p"));
    if (T_in) T_expr = compile("T", toml::find<std::string>(input, "initialize", "T"));

    auto point_conservatives = [&](double px, double py, rtype * cons) {
        x = px;
        y = py;
        rtype rho = rho_in ? rho_expr.value() : 0.0;
        rtype p = p_in ? p_expr.value() : 0.0;
        const rtype T = T_in ? T_expr.value() : 0.0;
        if (!rho_in) rho = physics.get_density_from_pressure_temperature(p, T);
        if (!p_in) p = physics.get_pressure_from_density_temperature(rho, T);
        const rtype W[N_CONSERVATIVE] = {rho, static_cast<rtype>(u_x_expr.value()),
                                         static_cast<rtype>(u_y_expr.value()), p};
        physics.compute_conservatives_from_W(cons, W);
    };

    TriangleDunavant quad(5);
    const uint32_t n_quad = quad.h_weights.extent(0);
    rtype weight_sum = 0.0;
    for (uint32_t q = 0; q < n_quad; q++) weight_sum += quad.h_weights(q);

    for (uint32_t i_cell = 0; i_cell < mesh->n_cells; ++i_cell) {
        const uint32_t n_nodes = mesh->h_n_nodes_of_cell(i_cell);
        rtype sum[N_CONSERVATIVE] = {};
        rtype area_sum = 0.0;
        const uint32_t n0 = mesh->h_node_of_cell(i_cell, 0);
        for (uint32_t k = 1; k + 1 < n_nodes; k++) {
            const uint32_t n1 = mesh->h_node_of_cell(i_cell, k);
            const uint32_t n2 = mesh->h_node_of_cell(i_cell, k + 1);
            const double v0[2] = {mesh->h_node_coords(n0, 0), mesh->h_node_coords(n0, 1)};
            const double e1[2] = {(mesh->h_node_coords(n1, 0) - v0[0]) / n_sub,
                                  (mesh->h_node_coords(n1, 1) - v0[1]) / n_sub};
            const double e2[2] = {(mesh->h_node_coords(n2, 0) - v0[0]) / n_sub,
                                  (mesh->h_node_coords(n2, 1) - v0[1]) / n_sub};
            const double sub_area = 0.5 * std::abs(e1[0] * e2[1] - e1[1] * e2[0]);
            for (uint32_t a = 0; a < n_sub; a++) {
                for (uint32_t b = 0; a + b < n_sub; b++) {
                    // Upward sub-triangle with origin (a, b), and the downward one if it exists
                    for (int orient = 0; orient < 2; orient++) {
                        if (orient == 1 && a + b + 1 >= n_sub) continue;
                        double o[2], d1[2], d2[2];
                        if (orient == 0) {
                            FOR_I_DIM {
                                o[i] = v0[i] + a * e1[i] + b * e2[i];
                                d1[i] = e1[i];
                                d2[i] = e2[i];
                            }
                        } else {
                            FOR_I_DIM {
                                o[i] = v0[i] + (a + 1) * e1[i] + (b + 1) * e2[i];
                                d1[i] = -e1[i];
                                d2[i] = -e2[i];
                            }
                        }
                        for (uint32_t q = 0; q < n_quad; q++) {
                            const double xi = quad.h_points(q, 0);
                            const double eta = quad.h_points(q, 1);
                            rtype cons[N_CONSERVATIVE];
                            point_conservatives(o[0] + xi * d1[0] + eta * d2[0],
                                                o[1] + xi * d1[1] + eta * d2[1],
                                                cons);
                            const rtype w = quad.h_weights(q) / weight_sum * sub_area;
                            FOR_I_CONSERVATIVE sum[i] += w * cons[i];
                        }
                        area_sum += sub_area;
                    }
                }
            }
        }
        FOR_I_CONSERVATIVE h_conservatives(i_cell, i) = sum[i] / area_sum;
    }
}

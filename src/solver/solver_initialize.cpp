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

#include "input.h"

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
    // Restart files hold cells in global order, so any number of ranks can read them
    const bool distributed = mesh->n_global_cells > 0;
    const uint64_t n_expected = distributed ? mesh->n_global_cells : mesh->n_cells;
    if (restart.n_cells != n_expected || restart.conservatives.size() != N_CONSERVATIVE) {
        throw std::runtime_error("Restart file " + file + " does not match the mesh.");
    }
    for (uint32_t i_cell = 0; i_cell < mesh->n_cells; ++i_cell) {
        const uint64_t g = distributed ? mesh->h_global_cell_id[i_cell] : i_cell;
        FOR_I_CONSERVATIVE h_conservatives(i_cell, i) = restart.conservatives[i][g];
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
    std::vector<rtype> u = find_real_vector(input, "initialize", "u");
    if (u.size() != N_DIM) {
        throw std::runtime_error("u must be a " + std::to_string(N_DIM) + "-element array for initialization: constant.");
    }
    const rtype p = find_real(input, "initialize", "p");
    const rtype T = find_real(input, "initialize", "T");
    rtype W[N_CONSERVATIVE];
    W[0] = physics.get_density_from_pressure_temperature(p, T);
    FOR_I_DIM W[1 + i] = u[i];
    W[N_DIM + 1] = p;
    rtype cons[N_CONSERVATIVE];
    physics.compute_conservatives_from_W(cons, W);
    for (uint32_t i_cell = 0; i_cell < mesh->n_cells; ++i_cell) {
        FOR_I_CONSERVATIVE h_conservatives(i_cell, i) = cons[i];
    }
}

/**
 * Cell averages of the conservative variables are computed by splitting each
 * cell into simplices and applying a composite rule on each: in 2D, a fan of
 * triangles, each subdivided into n_sub^2 sub-triangles with a degree-5
 * Dunavant rule; in 3D, the tetrahedra of Mesh::h_cell_tetrahedra, each
 * mapped from the unit cube (Duffy) and integrated with a 4^3-point Gauss
 * rule on each of n_sub^3 sub-cubes (exact to degree 5). This resolves
 * discontinuous initial data that do not align with the mesh and is
 * high-order accurate for smooth data.
 */
void Solver::init_solution_analytical() {
    const toml::value & init = input.at("initialize");
    if (!init.contains("u") || init.at("u").as_array().size() != N_DIM) {
        throw std::runtime_error("u must be a " + std::to_string(N_DIM) + "-element array for initialization: analytical.");
    }
    const bool rho_in = init.contains("rho");
    const bool p_in = init.contains("p");
    const bool T_in = init.contains("T");
    if (rho_in + p_in + T_in != 2) {
        throw std::runtime_error("Exactly two of rho, p, and T must be specified for initialization: analytical.");
    }
    const uint32_t n_sub = toml::find_or<uint32_t>(input, "initialize", "n_subdivisions", (N_DIM == 2) ? 4 : 2);

    std::vector<std::string> u_str = toml::find<std::vector<std::string>>(input, "initialize", "u");

    double x = 0.0, y = 0.0, z = 0.0;
    exprtk::symbol_table<double> symbol_table;
    symbol_table.add_variable("x", x);
    symbol_table.add_variable("y", y);
    symbol_table.add_variable("z", z);
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
    std::vector<exprtk::expression<double>> u_expr;
    FOR_I_DIM u_expr.push_back(compile("u[" + std::to_string(i) + "]", u_str[i]));
    exprtk::expression<double> rho_expr, p_expr, T_expr;
    if (rho_in) rho_expr = compile("rho", toml::find<std::string>(input, "initialize", "rho"));
    if (p_in) p_expr = compile("p", toml::find<std::string>(input, "initialize", "p"));
    if (T_in) T_expr = compile("T", toml::find<std::string>(input, "initialize", "T"));

    auto point_conservatives = [&](double px, double py, double pz, rtype * cons) {
        x = px;
        y = py;
        z = pz;
        rtype rho = rho_in ? rho_expr.value() : 0.0;
        rtype p = p_in ? p_expr.value() : 0.0;
        const rtype T = T_in ? T_expr.value() : 0.0;
        if (!rho_in) rho = physics.get_density_from_pressure_temperature(p, T);
        if (!p_in) p = physics.get_pressure_from_density_temperature(rho, T);
        rtype W[N_CONSERVATIVE];
        W[0] = rho;
        FOR_I_DIM W[1 + i] = static_cast<rtype>(u_expr[i].value());
        W[N_DIM + 1] = p;
        physics.compute_conservatives_from_W(cons, W);
    };

    if constexpr (N_DIM == 3) {
        init_cell_averages_3d(n_sub, [&](double px, double py, double pz, rtype * cons) {
            point_conservatives(px, py, pz, cons);
        });
        return;
    }

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
                                                0.0, cons);
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

void Solver::init_cell_averages_3d(const uint32_t n_sub,
                                   const std::function<void(double, double, double, rtype *)> & f) {
    // 4-point Gauss-Legendre rule on [0, 1]
    const double g[4] = {0.5 - 0.5 * 0.8611363115940526, 0.5 - 0.5 * 0.3399810435848563,
                         0.5 + 0.5 * 0.3399810435848563, 0.5 + 0.5 * 0.8611363115940526};
    const double gw[4] = {0.5 * 0.3478548451374538, 0.5 * 0.6521451548625461,
                          0.5 * 0.6521451548625461, 0.5 * 0.3478548451374538};
    const double h = 1.0 / n_sub;
    std::vector<std::array<std::array<double, 3>, 4>> tets;
    for (uint32_t i_cell = 0; i_cell < mesh->n_cells; ++i_cell) {
        mesh->h_cell_tetrahedra(i_cell, tets);
        double sum[N_CONSERVATIVE] = {};
        double vol_sum = 0.0;
        for (const auto & tet : tets) {
            double e[3][3];
            for (int k = 0; k < 3; k++) {
                for (int d = 0; d < 3; d++) e[k][d] = tet[k + 1][d] - tet[0][d];
            }
            const double det = e[0][0] * (e[1][1] * e[2][2] - e[1][2] * e[2][1]) -
                               e[0][1] * (e[1][0] * e[2][2] - e[1][2] * e[2][0]) +
                               e[0][2] * (e[1][0] * e[2][1] - e[1][1] * e[2][0]);
            const double six_vol = std::abs(det);
            // Duffy map of the unit cube: barycentric (u, v (1 - u), w (1 - u) (1 - v))
            for (uint32_t a = 0; a < n_sub; a++) {
            for (uint32_t b = 0; b < n_sub; b++) {
            for (uint32_t c = 0; c < n_sub; c++) {
                for (int qa = 0; qa < 4; qa++) {
                for (int qb = 0; qb < 4; qb++) {
                for (int qc = 0; qc < 4; qc++) {
                    const double u = (a + g[qa]) * h, v = (b + g[qb]) * h, w = (c + g[qc]) * h;
                    const double l1 = u, l2 = v * (1.0 - u), l3 = w * (1.0 - u) * (1.0 - v);
                    const double weight = gw[qa] * gw[qb] * gw[qc] * h * h * h *
                                          six_vol * (1.0 - u) * (1.0 - u) * (1.0 - v);
                    double p[3];
                    for (int d = 0; d < 3; d++) p[d] = tet[0][d] + l1 * e[0][d] + l2 * e[1][d] + l3 * e[2][d];
                    rtype cons[N_CONSERVATIVE];
                    f(p[0], p[1], p[2], cons);
                    FOR_I_CONSERVATIVE sum[i] += weight * cons[i];
                    vol_sum += weight;
                }
                }
                }
            }
            }
            }
        }
        FOR_I_CONSERVATIVE h_conservatives(i_cell, i) = sum[i] / vol_sum;
    }
}

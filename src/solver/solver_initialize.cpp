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

#include <algorithm>
#include <iomanip>
#include <sstream>
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
    const std::string type_str = toml::find_or<std::string>(input, "initialize", "type", "constant");
    auto it = INIT_TYPES.find(type_str);
    if (it == INIT_TYPES.end()) {
        throw unknown_option(INIT_TYPES, "initialize.type", type_str);
    }
    initial_state = type_str;
    if (it->second == InitType::CONSTANT) {
        init_solution_constant();
    } else if (it->second == InitType::ANALYTICAL) {
        init_solution_analytical();
    } else {
        init_solution_restart();
    }
    copy_host_to_device();
    if (is_mixture() && it->second != InitType::RESTART) init_temperature_seed();
    update_primitives();
}

void Solver::init_solution_restart() {
    if (!input.at("initialize").contains("file")) {
        throw std::runtime_error("Missing file for initialization: restart.");
    }
    const std::string file = toml::find<std::string>(input, "initialize", "file");
    // Restart files hold cells in global order, so any number of ranks can read
    // them; each rank reads only its local cells
    const bool distributed = mesh->n_global_cells > 0;
    RestartData restart = read_restart(file, distributed ? &mesh->h_global_cell_id : nullptr);
    const uint64_t n_expected = distributed ? mesh->n_global_cells : mesh->n_cells;
    if (restart.n_cells != n_expected) {
        throw std::runtime_error("Restart file " + file + " does not match the mesh.");
    }
    // Variables map by name: the flow block, then one RHOY_<name> per species
    // CHEM_H (last chemistry sub-step) only seeds the integrator: a reacting run
    // may start from a non-reacting one, and a non-reacting run ignores it
    std::vector<std::string> expected = restart_variables();
    for (const auto & name : restart.names) {
        if (name == "CHEM_H") continue;
        if (std::find(expected.begin(), expected.end(), name) == expected.end()) {
            throw std::runtime_error("Restart file " + file + " has variable " + name + ", which this run does not " +
                                     (name.rfind("RHOY_", 0) == 0 ? "transport." : "know."));
        }
    }
    for (uint32_t v = 0; v < expected.size(); v++) {
        const std::vector<rtype> * values = restart.find(expected[v]);
        if (expected[v] == "CHEM_H") {
            for (uint32_t i_cell = 0; i_cell < mesh->n_cells; ++i_cell) {
                h_chem_h(i_cell) = values ? (*values)[i_cell] : 0.0_r;
            }
            continue;
        }
        if (values == nullptr) {
            throw std::runtime_error("Restart file " + file + " has no variable " + expected[v] + ".");
        }
        const uint32_t n_species = species_names.size();
        for (uint32_t i_cell = 0; i_cell < mesh->n_cells; ++i_cell) {
            if (v < N_CONSERVATIVE) {
                h_conservatives(i_cell, v) = (*values)[i_cell];
            } else if (v < N_CONSERVATIVE + n_species) {
                h_species(i_cell, v - N_CONSERVATIVE) = (*values)[i_cell];
            } else {
                h_T_seed(i_cell) = (*values)[i_cell];
            }
        }
    }
    step = restart.step;
    t = restart.t;
    for (auto & writer : data_writers) {
        writer->resume(step, t);
    }
    initial_state = "restart from " + file + " (step " + std::to_string(step) + ", t = " + logging::real(double(t)) + ")";
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
    if (is_mixture()) {
        const std::vector<double> Y = mixture_model->mass_fractions(init, "initialize");
        rtype cons[N_CONSERVATIVE];
        std::vector<rtype> rhoY(species_names.size());
        mixture_model->conservatives(static_cast<double>(p), static_cast<double>(T), u.data(), Y, cons, rhoY.data());
        for (uint32_t i_cell = 0; i_cell < mesh->n_cells; ++i_cell) {
            FOR_I_CONSERVATIVE h_conservatives(i_cell, i) = cons[i];
            for (size_t k = 0; k < rhoY.size(); k++) h_species(i_cell, k) = rhoY[k];
        }
        return;
    }
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

    // Mixtures: one expression per listed species of X or Y, and optionally a
    // balance species taking the rest; otherwise the listed values are normalized
    bool mole = false;
    int32_t balance = -1;
    std::vector<int32_t> listed;
    std::vector<exprtk::expression<double>> fraction_expr;
    if (is_mixture()) {
        if (init.contains("X") == init.contains("Y")) {
            throw InputError("initialize: give the composition as exactly one of X and Y.");
        }
        mole = init.contains("X");
        const std::string key = mole ? "X" : "Y";
        if (!init.at(key).is_table()) throw InputError("initialize." + key + " must be a table of species.");
        for (const auto & [name, value] : init.at(key).as_table()) {
            const int32_t k = mixture_model->mechanism().species_index(name);
            if (k < 0) throw InputError("initialize." + key + ": no species " + name + " in the mechanism.");
            listed.push_back(k);
            std::ostringstream text;
            if (value.is_string()) {
                text << value.as_string();
            } else {
                text << std::setprecision(17) << static_cast<double>(as_real(value, name));
            }
            fraction_expr.push_back(compile(key + "." + name, text.str()));
        }
        if (init.contains("balance")) {
            const std::string name = toml::find<std::string>(init, "balance");
            balance = mixture_model->mechanism().species_index(name);
            if (balance < 0) throw InputError("initialize.balance: no species " + name + " in the mechanism.");
        }
    }
    const uint32_t n_species = species_names.size();
    const uint32_t n_vars = N_CONSERVATIVE + n_species;
    std::vector<double> fractions(n_species);

    auto point_conservatives = [&](double px, double py, double pz, rtype * cons) {
        x = px;
        y = py;
        z = pz;
        if (is_mixture()) {
            std::fill(fractions.begin(), fractions.end(), 0.0);
            double sum = 0.0;
            for (size_t i = 0; i < listed.size(); i++) {
                fractions[listed[i]] = std::max(fraction_expr[i].value(), 0.0);
                if (listed[i] != balance) sum += fractions[listed[i]];
            }
            if (balance >= 0) {
                fractions[balance] = std::max(1.0 - sum, 0.0);
                sum += fractions[balance];
            }
            if (!(sum > 0.0)) throw InputError("initialize: the composition is zero at a point.");
            for (double & f : fractions) f /= sum;
            const std::vector<double> Y = mole ? mixture_model->mass_fractions_from_mole(fractions) : fractions;
            const double R = mixture_model->gas_constant(Y);
            const double rho_m = rho_in ? rho_expr.value() : 0.0;
            double p_m = p_in ? p_expr.value() : 0.0;
            double T_m = T_in ? T_expr.value() : 0.0;
            if (!p_in) p_m = rho_m * R * T_m;
            if (!T_in) T_m = p_m / (rho_m * R);
            rtype u[N_DIM];
            FOR_I_DIM u[i] = static_cast<rtype>(u_expr[i].value());
            mixture_model->conservatives(p_m, T_m, u, Y, cons, cons + N_CONSERVATIVE);
            return;
        }
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
        init_cell_averages_3d(n_sub, n_vars, [&](double px, double py, double pz, rtype * cons) {
            point_conservatives(px, py, pz, cons);
        });
        return;
    }

    TriangleDunavant quad(5);
    const uint32_t n_quad = quad.h_weights.extent(0);
    double weight_sum = 0.0;
    for (uint32_t q = 0; q < n_quad; q++) weight_sum += double(quad.h_weights(q));

    std::vector<double> sum(n_vars);
    std::vector<rtype> cons(n_vars);
    for (uint32_t i_cell = 0; i_cell < mesh->n_cells; ++i_cell) {
        const uint32_t n_nodes = mesh->h_n_nodes_of_cell(i_cell);
        std::fill(sum.begin(), sum.end(), 0.0);
        double area_sum = 0.0;
        const uint32_t n0 = mesh->h_node_of_cell(i_cell, 0);
        for (uint32_t k = 1; k + 1 < n_nodes; k++) {
            const uint32_t n1 = mesh->h_node_of_cell(i_cell, k);
            const uint32_t n2 = mesh->h_node_of_cell(i_cell, k + 1);
            const double v0[2] = {double(mesh->h_node_coords(n0, 0)), double(mesh->h_node_coords(n0, 1))};
            const double e1[2] = {(double(mesh->h_node_coords(n1, 0)) - v0[0]) / n_sub,
                                  (double(mesh->h_node_coords(n1, 1)) - v0[1]) / n_sub};
            const double e2[2] = {(double(mesh->h_node_coords(n2, 0)) - v0[0]) / n_sub,
                                  (double(mesh->h_node_coords(n2, 1)) - v0[1]) / n_sub};
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
                            const double xi = double(quad.h_points(q, 0));
                            const double eta = double(quad.h_points(q, 1));
                            point_conservatives(o[0] + xi * d1[0] + eta * d2[0],
                                                o[1] + xi * d1[1] + eta * d2[1],
                                                0.0, cons.data());
                            const double w = double(quad.h_weights(q)) / weight_sum * sub_area;
                            for (uint32_t i = 0; i < n_vars; i++) sum[i] += w * double(cons[i]);
                        }
                        area_sum += sub_area;
                    }
                }
            }
        }
        FOR_I_CONSERVATIVE h_conservatives(i_cell, i) = sum[i] / area_sum;
        for (uint32_t k = 0; k < n_species; k++) {
            h_species(i_cell, k) = static_cast<rtype>(sum[N_CONSERVATIVE + k] / area_sum);
        }
    }
}

void Solver::init_cell_averages_3d(const uint32_t n_sub, const uint32_t n_vars,
                                   const std::function<void(double, double, double, rtype *)> & f) {
    // 4-point Gauss-Legendre rule on [0, 1]
    const double g[4] = {0.5 - 0.5 * 0.8611363115940526, 0.5 - 0.5 * 0.3399810435848563,
                         0.5 + 0.5 * 0.3399810435848563, 0.5 + 0.5 * 0.8611363115940526};
    const double gw[4] = {0.5 * 0.3478548451374538, 0.5 * 0.6521451548625461,
                          0.5 * 0.6521451548625461, 0.5 * 0.3478548451374538};
    const double h = 1.0 / n_sub;
    std::vector<std::array<std::array<double, 3>, 4>> tets;
    std::vector<double> sum(n_vars);
    std::vector<rtype> cons(n_vars);
    for (uint32_t i_cell = 0; i_cell < mesh->n_cells; ++i_cell) {
        mesh->h_cell_tetrahedra(i_cell, tets);
        std::fill(sum.begin(), sum.end(), 0.0);
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
                    f(p[0], p[1], p[2], cons.data());
                    for (uint32_t i = 0; i < n_vars; i++) sum[i] += weight * double(cons[i]);
                    vol_sum += weight;
                }
                }
                }
            }
            }
            }
        }
        FOR_I_CONSERVATIVE h_conservatives(i_cell, i) = sum[i] / vol_sum;
        for (uint32_t k = 0; k + N_CONSERVATIVE < n_vars; k++) {
            h_species(i_cell, k) = static_cast<rtype>(sum[N_CONSERVATIVE + k] / vol_sum);
        }
    }
}

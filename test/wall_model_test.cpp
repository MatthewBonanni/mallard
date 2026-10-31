/**
 * @file wall_model_test.cpp
 * @brief Wall-stress models of wall-modeled LES against independent
 *        solutions of their equations (docs/design/wall_model.md, section 8).
 * @version 0.1
 * @date 2026-10-15
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 *
 */

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <functional>
#include <memory>
#include <utility>
#include <sstream>
#include <string>
#include <vector>

#include "mesh.h"
#include "solver.h"
#include "scratch.h"
#include "test_fixtures.h"
#include "test_utils.h"
#include "wall_exchange.h"
#include "wall_model.h"

namespace {

using wall_model::Gas;
using WallInput = wall_model::Input;
using wall_model::Output;

constexpr double KAPPA = 0.41, A_PLUS = 17.0;

WallModelParameters parameters(const WallModelType type) {
    WallModelParameters P;
    P.type = type;
    return P;
}

Gas constant_gas(const double mu, const double cp = 1004.5, const double R = 287.0, const double Pr = 0.72) {
    Gas g;
    g.gas.viscosity_model = ViscosityModel::CONSTANT;
    g.gas.mu_ref = rtype(mu);
    g.gas.cp = rtype(cp);
    g.gas.R = rtype(R);
    g.gas.Pr = rtype(Pr);
    return g;
}

/** @brief Air-like gas with mu ~ T^0.75, Pr = 0.7, in units with R = 1 and T_w = 1. */
Gas power_law_gas(const double mu_w) {
    Gas g;
    g.gas.viscosity_model = ViscosityModel::POWER_LAW;
    g.gas.mu_ref = rtype(mu_w);
    g.gas.T_mu_ref = 1.0_r;
    g.gas.n_mu = 0.75_r;
    g.gas.R = 1.0_r;
    g.gas.gamma = 1.4_r;
    g.gas.cp = 3.5_r;
    g.gas.Pr = 0.7_r;
    return g;
}

/** @brief Classical fourth-order Runge-Kutta for y' = f(x, y) over n log-spaced steps from 0 to x_end. */
template <size_t M, typename F>
void integrate(const F & f, const double x_end, double * y, const int n = 8000) {
    double x = 0.0;
    const double x_first = 1e-6 * x_end;
    for (int i = 0; i <= n; i++) {
        const double x_next = i == 0 ? x_first : x_first * std::pow(x_end / x_first, double(i) / n);
        const double dx = x_next - x;
        double k1[M], k2[M], k3[M], k4[M], t[M];
        f(x, y, k1);
        for (size_t m = 0; m < M; m++) t[m] = y[m] + 0.5 * dx * k1[m];
        f(x + 0.5 * dx, t, k2);
        for (size_t m = 0; m < M; m++) t[m] = y[m] + 0.5 * dx * k2[m];
        f(x + 0.5 * dx, t, k3);
        for (size_t m = 0; m < M; m++) t[m] = y[m] + dx * k3[m];
        f(x_next, t, k4);
        for (size_t m = 0; m < M; m++) y[m] += dx / 6.0 * (k1[m] + 2.0 * k2[m] + 2.0 * k3[m] + k4[m]);
        x = x_next;
    }
}

/** @brief Root of increasing g by bisection on log x in [lo, hi]. */
double bisect_log(const std::function<double(double)> & g, double lo, double hi) {
    for (int i = 0; i < 60; i++) {
        const double mid = std::sqrt(lo * hi);
        (g(mid) < 0.0 ? lo : hi) = mid;
    }
    return std::sqrt(lo * hi);
}

/** @brief U+ at h+ of the constant-property law (1 + kappa y+ D(y+)) dU+/dy+ = 1. */
double incompressible_U_plus(const double h_plus) {
    double U = 0.0;
    integrate<1>([](double y, const double *, double * d) {
        const double D = std::pow(1.0 - std::exp(-y / A_PLUS), 2);
        d[0] = 1.0 / (1.0 + KAPPA * y * D);
    }, h_plus, &U);
    return U;
}

} // namespace

// At constant properties both models reduce to the incompressible mixing-length
// law; the reference integrates it independently. Fails if the grid is too
// coarse near the wall or far from it, the damping uses another coordinate, or
// either model's transformation is off.
TEST(WallModel, IncompressibleLimit) {
    const double mu = 1e-5, rho = 1.0, T = 300.0, R = 287.0;
    const Gas gas = constant_gas(mu, 1.0e7, R);  // large cp: no viscous heating
    for (const double h_plus : {1.0, 10.0, 100.0, 1000.0, 10000.0}) {
        const double u_tau = 0.05, h = h_plus * mu / (rho * u_tau);
        WallInput in;
        in.h = h;
        in.U = u_tau * incompressible_U_plus(h_plus);
        in.T = in.T_w = in.T_e = T;
        in.U_e = in.U;
        in.p = rho * R * T;
        const double tau = rho * u_tau * u_tau;
        for (const WallModelType type : {WallModelType::EQUILIBRIUM, WallModelType::GFM}) {
            const Output out = wall_model::evaluate(parameters(type), gas, in);
            EXPECT_TRUE(out.converged) << "h+ = " << h_plus;
            EXPECT_NEAR(out.tau_w / tau, 1.0, 1e-4) << "h+ = " << h_plus << ", model " << int(type);
        }
    }
}

// Laminar Couette flow with heat transfer (h+ << 1, mu_t negligible):
// tau = mu U / h and, from the constant total energy flux k dT/dy + mu u du/dy,
// q_w = -(k (T_e - T_w) + mu U^2 / 2) / h. Fails on a sign or factor error in
// the viscous work or the heat flux, and on the heat-flux wall's temperature.
TEST(WallModel, LaminarCouetteWithHeating) {
    const double mu = 1.8e-5, cp = 50.0, Pr = 0.72, k = mu * cp / Pr;
    const Gas gas = constant_gas(mu, cp, 287.0, Pr);
    WallInput in;
    in.h = 1e-8;
    in.U = 1.0;
    in.T_w = 300.0;
    in.T = 290.0;
    in.p = 1e5;
    const Output out = wall_model::equilibrium(parameters(WallModelType::EQUILIBRIUM), gas, in);
    EXPECT_TRUE(out.converged);
    EXPECT_NEAR(out.tau_w / (mu * in.U / in.h), 1.0, 1e-6);
    const double q = -(k * (in.T - in.T_w) + 0.5 * mu * in.U * in.U) / in.h;
    EXPECT_NEAR(out.q_w / q, 1.0, 1e-6);

    WallInput flux = in;
    flux.thermal = WallThermal::HEAT_FLUX;
    flux.q = q;
    const Output out_flux = wall_model::equilibrium(parameters(WallModelType::EQUILIBRIUM), gas, flux);
    EXPECT_NEAR(out_flux.T_w, in.T_w, 1e-6 * in.T_w);
    EXPECT_DOUBLE_EQ(out_flux.q_w, q);
}

// The ODE model with variable properties (a hot, fast layer over a cold wall)
// against the same equations integrated by RK4 on a fine grid and shot on
// tau_w and the heat flux. Fails if the 64-node discretization loses accuracy
// or the properties lag the iteration.
TEST(WallModel, EquilibriumCompressible) {
    const double mu_w = 1.0 / 20000.0;
    const Gas gas = power_law_gas(mu_w);
    const WallModelParameters P = parameters(WallModelType::EQUILIBRIUM);
    WallInput in;
    in.h = 0.3;
    in.U = 0.8;           // M ~ 0.7 at the wall's sound speed sqrt(1.4)
    in.T_w = 1.0;
    in.T = 1.6;
    in.p = 1.0;
    const Output out = wall_model::equilibrium(P, gas, in);
    ASSERT_TRUE(out.converged);

    // u' = tau / (mu + mu_t), T' = (C - tau u) / k_eff; shoot on tau and C
    auto solve = [&](const double tau, const double C, double * uT) {
        uT[0] = 0.0;
        uT[1] = in.T_w;
        integrate<2>([&](double y, const double * s, double * d) {
            const double rho = in.p / (gas.R() * s[1]), mu = gas.mu(s[1]);
            const double y_star = y * std::sqrt(rho * tau) / mu;
            const double mu_t = P.kappa * y * std::sqrt(rho * tau) * std::pow(1.0 - std::exp(-y_star / P.A_plus), 2);
            d[0] = tau / (mu + mu_t);
            d[1] = (C - tau * s[0]) / (gas.cp() * (mu / gas.Pr() + mu_t / P.Pr_t));
        }, in.h, uT, 4000);
    };
    double tau = 0.0, C = 0.0;
    for (int outer = 0; outer < 8; outer++) {
        tau = bisect_log([&](double t) { double s[2]; solve(t, C, s); return s[0] - in.U; }, 1e-8, 1.0);
        double lo = -10.0, hi = 10.0;
        for (int i = 0; i < 50; i++) {
            const double mid = 0.5 * (lo + hi);
            double s[2];
            solve(tau, mid, s);
            (s[1] < in.T ? lo : hi) = mid;
        }
        C = 0.5 * (lo + hi);
    }
    EXPECT_NEAR(out.tau_w / tau, 1.0, 2e-4);
    EXPECT_NEAR(out.q_w / (-C), 1.0, 2e-4);
    EXPECT_LT(out.q_w, 0.0);  // the hot layer heats the wall
}

// Griffin, Fu & Moin's model against their inverse transformation in
// differential form (their eq. 5, with the density and viscosity derivatives
// through the temperature-velocity relation), integrated by RK4 and shot on
// tau_w, for a supersonic layer over a cold wall. Fails if the difference form
// of the transformation, the relation through the exchange point, or the
// shooting is wrong.
TEST(WallModel, InverseTransformationCompressible) {
    const double mu_w = 1.0 / 20000.0;
    const Gas gas = power_law_gas(mu_w);
    const WallModelParameters P = parameters(WallModelType::GFM);
    WallInput in;
    in.h = 0.3;
    in.U = 2.6;   // M_b ~ 3 channel: U_m ~ 0.85 U_c at 0.3 delta, U_c ~ 3 sqrt(1.4 T_w)
    in.T = 2.4;
    in.U_e = 3.1;
    in.T_e = 2.55;
    in.T_w = 1.0;
    in.p = 1.0;
    const Output out = wall_model::gfm(P, gas, in);
    ASSERT_TRUE(out.converged);

    const double cp = gas.cp(), Pr = gas.Pr();
    const double T_r = in.T_e + std::cbrt(Pr) * in.U_e * in.U_e / (2.0 * cp);
    const double a = P.s * Pr * (T_r - in.T_w);
    auto T_of_U = [&](double U) { return in.T_w + a * U / in.U_e * (1.0 - U / in.U) + std::pow(U / in.U, 2) * (in.T - in.T_w); };
    auto dT_dU = [&](double U) { return a / in.U_e * (1.0 - 2.0 * U / in.U) + 2.0 * U / (in.U * in.U) * (in.T - in.T_w); };
    const double rho_w = in.p / in.T_w;
    auto U_at_h = [&](const double tau) {
        const double u_tau = std::sqrt(tau / rho_w), h_plus = in.h * rho_w * u_tau / mu_w;
        double U_plus = 0.0;
        integrate<1>([&](double y_plus, const double * s, double * d) {
            const double U = u_tau * s[0], T = T_of_U(U);
            const double rho_p = in.T_w / T, mu_p = gas.mu(T) / mu_w;
            const double y_star = y_plus * std::sqrt(rho_p) / mu_p;
            const double S_t = 1.0 / (1.0 + KAPPA * y_star * std::pow(1.0 - std::exp(-y_star / A_PLUS), 2));
            // mu+ J = sqrt(rho+) (1 + beta a) with a = dU+/dy+
            const double drho_dT = -rho_p / T, dmu_dT = 0.75 * mu_p / T;
            const double beta = y_plus * u_tau * dT_dU(U) * (0.5 * drho_dT / rho_p - dmu_dT / mu_p);
            const double Af = std::sqrt(rho_p) / mu_p, B = 1.0 / (mu_p * S_t) - 1.0 / mu_p, C = std::sqrt(rho_p);
            // a (B + C (1 + beta a)) = Af (1 + beta a)
            const double qa = C * beta, qb = B + C - Af * beta, qc = -Af;
            d[0] = std::abs(qa) < 1e-14 ? -qc / qb : (-qb + std::sqrt(qb * qb - 4.0 * qa * qc)) / (2.0 * qa);
        }, h_plus, &U_plus);
        return u_tau * U_plus;
    };
    const double tau = bisect_log([&](double t) { return U_at_h(t) - in.U; }, 1e-6, 1.0);
    EXPECT_NEAR(out.tau_w / tau, 1.0, 5e-4);
    // Cold wall: heat flows into it, q_w = s tau cp (T_w - T_r) / U_e
    EXPECT_NEAR(out.q_w / (P.s * tau * cp * (in.T_w - T_r) / in.U_e), 1.0, 5e-4);
}

// An adiabatic wall under the GFM model sits at the recovery temperature and
// takes no heat; a hotter wall heats the fluid. Fails if the recovery factor,
// the sign of q_w or the adiabatic branch changes.
TEST(WallModel, InverseTransformationThermalBranches) {
    const Gas gas = power_law_gas(1.0 / 20000.0);
    const WallModelParameters P = parameters(WallModelType::GFM);
    WallInput in;
    in.h = 0.3;
    in.U = 2.0;
    in.T = 1.5;
    in.U_e = 2.4;
    in.T_e = 1.4;
    in.p = 1.0;
    in.thermal = WallThermal::HEAT_FLUX;
    const Output adiabatic = wall_model::gfm(P, gas, in);
    EXPECT_DOUBLE_EQ(adiabatic.T_w, in.T_e + std::cbrt(gas.Pr()) * in.U_e * in.U_e / (2.0 * gas.cp()));
    EXPECT_EQ(adiabatic.q_w, 0.0);
    EXPECT_GT(adiabatic.tau_w, 0.0);
    in.thermal = WallThermal::ISOTHERMAL;
    in.T_w = adiabatic.T_w + 0.5;
    EXPECT_GT(wall_model::gfm(P, gas, in).q_w, 0.0);
}

#if Mallard_DIM == 3

namespace {

/** @brief Channel between walls at y = 0 (bottom) and y = 1 (top), periodic in x and z. */
std::string channel_input(const std::string & mesh, const uint32_t n[3], const std::string & bottom,
                          const std::string & top, const std::string & init, const std::string & run,
                          const std::string & extra = "") {
    std::ostringstream s;
    s << "[run]\n" << run
      << "[mesh]\ntype = \"" << mesh << "\"\nNx = " << n[0] << "\nNy = " << n[1] << "\nNz = " << n[2]
      << "\nLx = 1.0\nLy = 1.0\nLz = 0.6\n[[periodic]]\nzones = [\"left\", \"right\"]\n[[periodic]]\nzones = [\"back\", \"front\"]\n"
      << "[initialize]\n" << init
      << "[[boundaries]]\nname = \"bottom\"\n" << bottom
      << "[[boundaries]]\nname = \"top\"\n" << top
      << "[numerics]\nriemann_solver = \"HLLC\"\ntime_integrator = \"SSPRK3\"\ncheck_nan = true\n"
      << "[numerics.face_reconstruction]\ntype = \"MUSCL\"\n"
      << "[physics]\ntype = \"navier_stokes\"\ngamma = 1.4\np_ref = 1.0\nT_ref = 1.0\nrho_ref = 1.0\nmu = 0.2\n"
         "Pr = 0.72\n"
      << "[output]\ncheck_interval = 1000000\n" << extra;
    return s.str();
}

const std::string MODELED_GFM = "type = \"wall_isothermal\"\nT = 1.0\n"
                                "wall_model = { model = \"gfm\", exchange_distance = 0.37, edge_distance = 0.8 }\n";
const std::string MODELED_ODE = "type = \"wall_isothermal\"\nT = 1.0\n"
                                "wall_model = { model = \"equilibrium\", exchange_cells = 2.5 }\n";

} // namespace

// Exchange and edge points on every cell type and across periodic seams: the
// cell found and the offset to its centroid place the point at x_f + h n up
// to the periodic lattice, and the point lies within the cell's node bounding
// box. Fails if the walk stops in the wrong cell, loses a periodic shift, or
// measures h along the wrong normal or from the wrong wall-cell height.
TEST(WallModel3D, ExchangePointsOnEveryCellType) {
    for (const std::string type : {"cartesian", "cartesian_tet", "cartesian_prism", "cartesian_pyramid",
                                   "cartesian_mixed"}) {
        auto input = parse_toml("[mesh]\ntype = \"" + type +
                                "\"\nNx = 6\nNy = 7\nNz = 5\nLx = 1.0\nLy = 1.0\nLz = 0.6\n[[periodic]]\nzones = [\"left\", \"right\"]\n[[periodic]]\nzones = [\"back\", \"front\"]\n");
        Mesh mesh;
        mesh.init(input);
        std::vector<int32_t> face_bc(mesh.n_faces, -1);
        std::vector<BoundaryCondition> bcs(2);
        std::vector<WallModelSpec> specs(2);
        for (size_t b = 0; b < 2; b++) {
            bcs[b].type = BoundaryType::WALL_ISOTHERMAL;
            bcs[b].modeled = true;
            specs[b].parameters.type = WallModelType::GFM;
            specs[b].edge_distance = 0.5;
        }
        specs[0].exchange_distance = 0.37;
        specs[1].exchange_cells = 2.3;
        const char * zones[2] = {"bottom", "top"};
        for (int b = 0; b < 2; b++) {
            const FaceZone * zone = mesh.get_face_zone(zones[b]);
            for (uint32_t i = 0; i < zone->n_faces(); i++) face_bc[zone->h_faces(i)] = b;
        }
        WallModelPoints points;
        const WallModelData data = make_wall_model_data(mesh, face_bc, bcs, specs, {}, 0, points);
        auto faces = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), data.faces);
        auto cells = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), data.cells);
        auto offsets = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), data.offsets);
        auto heights = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), data.heights);
        ASSERT_GT(faces.extent(0), 0u);
        const double L[3] = {1.0, 1.0, 0.6};
        for (uint32_t k = 0; k < faces.extent(0); k++) {
            const uint32_t f = faces(k);
            const uint32_t c0 = static_cast<uint32_t>(mesh.h_cells_of_face(f, 0));
            const bool bottom = face_bc[f] == 0;
            double depth = 0.0;
            for (int i = 0; i < 3; i++) {
                depth += double(mesh.h_cell_coords(c0, i) - mesh.h_face_coords(f, i)) *
                         double(-mesh.h_face_normals(f, i) / mesh.h_face_area(f));
            }
            const double tol = precision_tol<double>(1e-12, 1e-5);
            EXPECT_NEAR(double(heights(k, 0)), bottom ? 0.37 : 2.3 * 2.0 * depth, tol) << type;
            for (int s = 0; s < 2; s++) {
                ASSERT_GE(cells(k, s), 0) << type;
                const uint32_t c = static_cast<uint32_t>(cells(k, s));
                double lo[3] = {1e30, 1e30, 1e30}, hi[3] = {-1e30, -1e30, -1e30};
                for (uint32_t m = 0; m < mesh.h_n_nodes_of_cell(c); m++) {
                    for (int i = 0; i < 3; i++) {
                        lo[i] = std::min(lo[i], double(mesh.h_node_coords(mesh.h_node_of_cell(c, m), i)));
                        hi[i] = std::max(hi[i], double(mesh.h_node_coords(mesh.h_node_of_cell(c, m), i)));
                    }
                }
                for (int i = 0; i < 3; i++) {
                    const double x = double(mesh.h_cell_coords(c, i)) + double(offsets(k, s, i));
                    double target = double(mesh.h_face_coords(f, i)) -
                                    double(heights(k, s)) * double(mesh.h_face_normals(f, i) / mesh.h_face_area(f));
                    if (i != 1) target -= L[i] * std::round((target - x) / L[i]);
                    EXPECT_NEAR(x, target, tol) << type << " face " << f << " point " << s;
                    EXPECT_GE(x, lo[i] - tol) << type;
                    EXPECT_LE(x, hi[i] + tol) << type;
                }
            }
        }
    }
}

namespace {

/** @brief Couette flow under a moving no-slip top wall over a modeled bottom wall, from a perturbed profile to steady state. */
std::unique_ptr<Solver> modeled_couette(const std::string & mesh, const std::string & bottom) {
    const uint32_t n[3] = {3, 8u, 3};
    auto solver = std::make_unique<Solver>();
    solver->set_distributed(false);
    solver->init(parse_toml(channel_input(
        mesh, n, bottom, "type = \"wall_isothermal\"\nT = 1.0\nu = [0.1, 0.0, 0.0]\n",
        "type = \"analytical\"\nrho = \"1.0\"\nu = [\"0.1 * y + 0.02 * sin(3.14159265 * y)\", \"0.0\", \"0.0\"]\np = \"1.0\"\n", "t_stop = 4.0\ncfl = 0.4\n")));
    solver->run();
    solver->update_primitives();
    solver->copy_device_to_host();
    return solver;
}

} // namespace

// Laminar Couette flow (h+ < 0.3, where both models are the viscous sublayer)
// with the bottom wall's stress from the model: the steady profile is u = U y
// and the modeled wall carries mu U / H. Fails if the modeled traction has the
// wrong sign or direction, if the model's input is taken at the wrong height,
// or if the modeled wall's flux is not the model's.
TEST(WallModel3D, LaminarCouetteFlowOverAModeledWall) {
    const std::pair<std::string, std::string> cases[] = {{"cartesian", MODELED_ODE}, {"cartesian", MODELED_GFM}};
    for (const auto & [mesh, bottom] : cases) {
        auto solver = modeled_couette(mesh, bottom);
        auto m = solver->get_mesh();
        double max_err = 0.0;
        for (uint32_t i = 0; i < m->n_owned(); i++) {
            max_err = std::max(max_err, std::abs(double(solver->h_primitives(i, 0)) - 0.1 * double(m->h_cell_coords(i, 1))));
        }
        EXPECT_LT(max_err, 1e-4) << mesh << " " << bottom;
        const auto F = solver->calc_force(m->get_face_zone("bottom")->faces);
        // Shear mu U / H = 0.02 on the area 0.6, along +x on the slower wall
        EXPECT_NEAR(F[3], 0.02 * 0.6, 2e-3 * 0.012) << mesh << " " << bottom;
        EXPECT_NEAR(F[4], 0.0, 1e-12);
        EXPECT_NEAR(F[5], 0.0, 1e-6);
    }
}

// A modeled wall whose velocity is given as expressions gets a condition per
// face; each keeps the wall model of its boundary. Fails if the per-face copies
// lose the model's parameters.
TEST(WallModel3D, ExpressionVelocityKeepsTheModel) {
    const uint32_t n[3] = {3, 8u, 3};
    auto run = [&](const std::string & u) {
        Solver solver;
        solver.init(parse_toml(channel_input(
            "cartesian", n, MODELED_GFM + "u = " + u + "\n", "type = \"wall_isothermal\"\nT = 1.0\nu = [0.1, 0.0, 0.0]\n",
            "type = \"analytical\"\nrho = \"1.0\"\nu = [\"0.1 * y\", \"0.0\", \"0.0\"]\np = \"1.0\"\n", "n_steps = 20\ncfl = 0.4\n")));
        solver.run();
        solver.copy_device_to_host();
        return solver.calc_force(solver.get_mesh()->get_face_zone("bottom")->faces);
    };
    const auto numbers = run("[-0.05, 0.0, 0.0]");
    const auto expressions = run("[\"-0.05 + 0 * x\", \"0 * y\", \"0 * z\"]");
    EXPECT_GT(std::abs(double(numbers[3])), 1e-4);
    for (size_t i = 0; i < numbers.size(); i++) {
        EXPECT_NEAR(double(expressions[i]), double(numbers[i]), roundoff(1e-12)) << i;
    }
}

// A restarted wall-modeled run reproduces the uninterrupted one bitwise: the
// models keep no state. Fails if a model or the exchange data came to depend
// on anything but the state (e.g. a filtered input without restart data).
TEST(WallModel3D, RestartIsExact) {
    const std::string dir = (scratch_dir() / "mallard_wall_model_restart").string();
    const uint32_t n[3] = {4, 6, 3};
    const std::string init = "type = \"analytical\"\nrho = \"1.0 + 0.1 * sin(6.283 * x)\"\n"
                             "u = [\"0.5 + 0.1 * sin(6.283 * z)\", \"0.05 * sin(6.283 * x)\", \"0.0\"]\np = \"1.0\"\n";
    auto input = [&](const std::string & sub, const std::string & initialize, const uint32_t steps) {
        return channel_input("cartesian_prism", n, MODELED_GFM, MODELED_ODE, initialize,
                             "n_steps = " + std::to_string(steps) + "\ncfl = 0.4\n",
                             "[[write_data]]\nprefix = \"" + dir + "/" + sub +
                                 "/restart\"\nformat = \"restart\"\ninterval = 5\n");
    };
    Solver straight;
    straight.init(parse_toml(input("a", init, 10)));
    straight.run();
    straight.copy_device_to_host();
    Solver first;
    first.init(parse_toml(input("b", init, 5)));
    first.run();
    Solver second;
    second.init(parse_toml(input("b", "type = \"restart\"\nfile = \"" + dir + "/b/restart_000005.restart\"\n", 10)));
    second.run();
    second.copy_device_to_host();
    ASSERT_EQ(second.get_step(), straight.get_step());
    for (uint32_t i = 0; i < straight.get_mesh()->n_cells; i++) {
        for (uint8_t v = 0; v < N_CONSERVATIVE; v++) {
            ASSERT_EQ(second.h_conservatives(i, v), straight.h_conservatives(i, v)) << "cell " << i;
        }
    }
}

#endif

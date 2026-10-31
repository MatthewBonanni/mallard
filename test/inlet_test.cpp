/**
 * @file inlet_test.cpp
 * @author Matthew Bonanni (mbonanni001@gmail.com)
 * @brief Tests of the stagnation and mass-flow inlets.
 * @version 0.1
 * @date 2026-10-15
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 *
 */

#include <gtest/gtest.h>
#include <Kokkos_Core.hpp>

#include <array>
#include <cmath>
#include <functional>
#include <limits>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "boundary.h"
#include "boundary_fields.h"
#include "data_paths.h"
#include "gmsh_fixtures.h"
#include "inflow.h"
#include "mechanism.h"
#include "solver.h"
#include "test_fixtures.h"
#include "thermo.h"

namespace {

const std::string SOURCE_DIR = source_dir();
const std::string PERFECT_AIR = SOURCE_DIR + "/test/data/chemistry/perfect_air.yaml";
const std::string H2O2 = SOURCE_DIR + "/mechanisms/h2o2.yaml";

constexpr double tol(double in_double, double in_float) {
    return sizeof(rtype) == sizeof(double) ? in_double : in_float;
}

using Vec = std::array<rtype, N_DIM>;

Vec vec(double x, double y, double z = 0.0) {
    Vec v{};
    v[0] = rtype(x);
    v[1] = rtype(y);
    if constexpr (N_DIM == 3) v[2] = rtype(z);
    return v;
}

Vec unit_vec(Vec v) {
    double norm = 0.0;
    for (rtype c : v) norm += double(c) * double(c);
    for (rtype & c : v) c = rtype(double(c) / std::sqrt(norm));
    return v;
}

/** @brief [rho, u, p] of a perfect gas on the isentrope of (T0, p0) at Mach M along d. */
std::array<rtype, N_CONSERVATIVE> isentropic_state(double gamma, double R, double T0, double p0, double M, const Vec & d) {
    const double T = T0 / (1.0 + 0.5 * (gamma - 1.0) * M * M);
    const double p = p0 * std::pow(T / T0, gamma / (gamma - 1.0));
    const double V = M * std::sqrt(gamma * R * T);
    std::array<rtype, N_CONSERVATIVE> W{};
    W[0] = rtype(p / (R * T));
    FOR_I_DIM W[1 + i] = rtype(V * double(d[i]));
    W[N_DIM + 1] = rtype(p);
    return W;
}

std::array<rtype, N_CONSERVATIVE> perfect_ghost(double gamma, double R, double T0, double p0, const Vec & d,
                                                const std::array<rtype, N_CONSERVATIVE> & W_i, const Vec & n) {
    BoundaryCondition bc;
    bc.type = BoundaryType::STAGNATION_INLET;
    bc.data[0] = rtype(T0);
    FOR_I_DIM bc.data[1 + i] = d[i];
    bc.data[N_DIM + 1] = rtype(p0);
    std::array<rtype, N_CONSERVATIVE> W_g{};
    bc.ghost_W(W_i.data(), n.data(), rtype(gamma), rtype(R), false, W_g.data());
    return W_g;
}

double relative_difference(const std::array<rtype, N_CONSERVATIVE> & a, const std::array<rtype, N_CONSERVATIVE> & b) {
    double speed = 0.0;
    FOR_I_DIM speed = std::max(speed, std::abs(double(b[1 + i])));
    double worst = std::abs(double(a[0] - b[0])) / double(b[0]);
    worst = std::max(worst, std::abs(double(a[N_DIM + 1] - b[N_DIM + 1])) / double(b[N_DIM + 1]));
    FOR_I_DIM worst = std::max(worst, std::abs(double(a[1 + i] - b[1 + i])) / std::max(speed, 1e-30));
    return worst;
}

} // namespace

TEST(StagnationInletTest, ReproducesThePerfectGasStateOnItsIsentrope) {
    // An interior already on the isentrope of (T0, p0), flowing along the
    // inlet's direction, is a fixed point: the ghost is that same state
    const double gamma = 1.4, R = 287.0, T0 = 300.0, p0 = 1.0e5;
    const Vec n = unit_vec(vec(-1.0, 0.2, 0.1));
    for (const Vec & d : {Vec{}, unit_vec(vec(0.8, 0.5, -0.2))}) {
        Vec flow = d;
        if (d == Vec{}) FOR_I_DIM flow[i] = -n[i];
        for (const double M : {0.02, 0.3, 0.8}) {
            const auto W_i = isentropic_state(gamma, R, T0, p0, M, flow);
            const auto W_g = perfect_ghost(gamma, R, T0, p0, d, W_i, n);
            EXPECT_LT(relative_difference(W_g, W_i), tol(1e-10, 2e-4)) << "M = " << M;
        }
    }
}

TEST(StagnationInletTest, KeepsTotalConditionsAndTheOutgoingInvariant) {
    const double gamma = 1.4, R = 287.0, T0 = 300.0, p0 = 1.0e5, cp = gamma * R / (gamma - 1.0);
    const Vec n = unit_vec(vec(-1.0, 0.3, 0.0));
    const Vec d = unit_vec(vec(1.0, 0.4, 0.3));
    auto u_n = [&](const auto & W) {
        double s = 0.0;
        FOR_I_DIM s += double(W[1 + i]) * double(n[i]);
        return s;
    };
    auto invariant = [&](const auto & W) {
        return u_n(W) + 2.0 * std::sqrt(gamma * double(W[N_DIM + 1]) / double(W[0])) / (gamma - 1.0);
    };
    // Interior off the isentrope, with a velocity of its own
    const std::array<rtype, N_CONSERVATIVE> W_i = [&] {
        std::array<rtype, N_CONSERVATIVE> W{};
        W[0] = 1.3_r;
        W[1] = 60.0_r;
        W[2] = -25.0_r;
        W[N_DIM + 1] = 9.0e4_r;
        return W;
    }();
    const auto W_g = perfect_ghost(gamma, R, T0, p0, d, W_i, n);
    const double T = double(W_g[N_DIM + 1]) / (double(W_g[0]) * R);
    double V2 = 0.0, along = 0.0;
    FOR_I_DIM {
        V2 += double(W_g[1 + i]) * double(W_g[1 + i]);
        along += double(W_g[1 + i]) * double(d[i]);
    }
    EXPECT_NEAR(cp * T + 0.5 * V2, cp * T0, tol(1e-10, 1e-5) * cp * T0);
    EXPECT_NEAR(double(W_g[N_DIM + 1]) / p0, std::pow(T / T0, gamma / (gamma - 1.0)), tol(1e-12, 1e-5));
    EXPECT_NEAR(along * along, V2, tol(1e-10, 1e-4) * V2);
    EXPECT_GT(along, 0.0);
    EXPECT_NEAR(invariant(W_g), invariant(W_i), tol(1e-10, 1e-5) * invariant(W_i));

    // Above the stagnation pressure the inlet stops the flow, at p0 and T0
    std::array<rtype, N_CONSERVATIVE> W_high = W_i;
    FOR_I_DIM W_high[1 + i] = 0.0_r;
    W_high[N_DIM + 1] = 1.2e5_r;
    const auto W_stop = perfect_ghost(gamma, R, T0, p0, d, W_high, n);
    FOR_I_DIM EXPECT_EQ(W_stop[1 + i], 0.0_r);
    EXPECT_NEAR(double(W_stop[N_DIM + 1]), p0, tol(1e-10, 1e-6) * p0);
    EXPECT_NEAR(double(W_stop[0]), p0 / (R * T0), tol(1e-10, 1e-6) * p0 / (R * T0));

    // Supersonic inflow keeps the interior speed on the isentrope
    const auto W_sup = isentropic_state(gamma, R, T0, p0, 1.0, d);
    std::array<rtype, N_CONSERVATIVE> W_fast = W_sup;
    FOR_I_DIM W_fast[1 + i] = rtype(1.5 * double(W_sup[1 + i]));
    W_fast[N_DIM + 1] = rtype(0.5 * double(W_sup[N_DIM + 1]));
    const auto W_s = perfect_ghost(gamma, R, T0, p0, d, W_fast, n);
    double V2_s = 0.0, V2_fast = 0.0;
    FOR_I_DIM {
        V2_s += double(W_s[1 + i]) * double(W_s[1 + i]);
        V2_fast += double(W_fast[1 + i]) * double(W_fast[1 + i]);
    }
    const double T_s = double(W_s[N_DIM + 1]) / (double(W_s[0]) * R);
    EXPECT_NEAR(V2_s, V2_fast, tol(1e-10, 1e-5) * V2_fast);
    EXPECT_NEAR(cp * T_s + 0.5 * V2_s, cp * T0, tol(1e-10, 1e-5) * cp * T0);
}

namespace {

/**
 * @brief Air between temperatures T0 and T along its isentrope, integrated
 *        from cp(T) alone: h(T0) - h(T) and ln(p / p0).
 */
std::array<double, 2> isentrope_by_quadrature(const chemistry::ThermoTable<Kokkos::HostSpace> & thermo,
                                              const std::vector<double> & Y, double R, double T0, double T) {
    const chemistry::MassFractions y{Y.data()};
    const int n = 2000;
    const double dT = (T0 - T) / n;
    double dh = 0.0, log_p = 0.0;
    for (int k = 0; k <= n; k++) {
        const double w = (k == 0 || k == n) ? 1.0 : (k % 2 ? 4.0 : 2.0);
        const double t = T + k * dT;
        const double cp = thermo.cp_mass(t, y);
        dh += w * cp;
        log_p -= w * cp / (R * t);
    }
    return {dh * dT / 3.0, log_p * dT / 3.0};
}

} // namespace

TEST(StagnationInletTest, ThermallyPerfectStateIsOnTheTrueIsentrope) {
    // Air at T0 = 2500 K, where cp rises by 15% over the expansion: the state
    // of the interior on the isentrope (integrated here from cp alone) is a
    // fixed point, and a constant ratio of specific heats misses it
    const chemistry::Mechanism mech = chemistry::read_mechanism(H2O2);
    const auto thermo = chemistry::make_thermo_table<Kokkos::HostSpace>(mech);
    std::vector<double> Y(mech.n_species(), 0.0);
    Y[size_t(mech.species_index("N2"))] = 0.767;
    Y[size_t(mech.species_index("O2"))] = 0.233;
    const chemistry::MassFractions y{Y.data()};
    const double R = thermo.gas_constant(y), T0 = 2500.0, p0 = 1.0e6;
    const double cp0 = thermo.cp_mass(T0, y), gamma_c = cp0 / (cp0 - R);
    const inflow::ThermallyPerfectGas<chemistry::ThermoTable<Kokkos::HostSpace>, chemistry::MassFractions> gas{thermo, y, R};
    const Vec n = unit_vec(vec(-1.0, 0.0, 0.0));
    const Vec d = unit_vec(vec(1.0, 0.3, 0.0));
    for (const double T : {2450.0, 2250.0, 2200.0}) {
        const auto [dh, log_p] = isentrope_by_quadrature(thermo, Y, R, T0, T);
        const double p = p0 * std::exp(log_p), V = std::sqrt(2.0 * dh);
        std::array<rtype, N_CONSERVATIVE> W_i{};
        W_i[0] = rtype(p / (R * T));
        FOR_I_DIM W_i[1 + i] = rtype(V * double(d[i]));
        W_i[N_DIM + 1] = rtype(p);
        const double a = std::sqrt(gamma_c * p / double(W_i[0]));
        ASSERT_LT(V, a) << "subsonic interior";
        std::array<rtype, N_CONSERVATIVE> W_g{};
        double T_b = 0.0;
        inflow::stagnation_state(gas, gamma_c, T0, p0, d.data(), W_i.data(), n.data(), W_g.data(), T_b);
        EXPECT_NEAR(T_b, T, tol(1e-8, 1e-4) * T);
        EXPECT_LT(relative_difference(W_g, W_i), tol(1e-8, 2e-4)) << "T = " << T;

        const auto W_perfect = perfect_ghost(gamma_c, R, T0, p0, d, W_i, n);
        if (T < 2300.0) {
            EXPECT_GT(relative_difference(W_perfect, W_i), 1e-3) << "T = " << T;
        }
    }
}

namespace {

std::string mesh_block(const std::string & type, uint32_t nx, uint32_t n, double Lx, double L) {
    std::ostringstream s;
    s << "[mesh]\ntype = \"" << type << "\"\nNx = " << nx << "\nNy = " << n << "\nLx = " << Lx << "\nLy = " << L << "\n";
    if (N_DIM == 3) s << "Nz = " << n << "\nLz = " << L << "\n";
    return s.str();
}

/** @brief Conditions for left and right; the sides are symmetry planes. */
std::string duct_boundaries(const std::string & left, const std::string & right) {
    std::ostringstream s;
    s << "[[boundaries]]\nname = \"left\"\n" << left << "[[boundaries]]\nname = \"right\"\n" << right;
    for (const char * side : {"bottom", "top", "back", "front"}) {
        if (N_DIM == 2 && (std::string(side) == "back" || std::string(side) == "front")) continue;
        s << "[[boundaries]]\nname = \"" << side << "\"\ntype = \"symmetry\"\n";
    }
    return s.str();
}

std::string zero_velocity() { return N_DIM == 2 ? "[0.0, 0.0]" : "[0.0, 0.0, 0.0]"; }

} // namespace

TEST(StagnationInletTest, OneSpeciesMixtureMatchesThePerfectGas) {
    // Constant cp = 3.5 R: the thermally perfect path (Newton on the
    // mixture's h and s) must give the perfect gas's flow up to round-off
    const std::string common =
        "[run]\nn_steps = 40\ncfl = 0.4\n" + mesh_block(N_DIM == 2 ? "cartesian_tri" : "cartesian_tet", 16, 4, 1.0, 0.25) +
        "[initialize]\ntype = \"analytical\"\np = \"1.0e5 - 2.0e4 * x\"\nT = \"300.0\"\nu = " +
        (N_DIM == 2 ? std::string("[\"0.0\", \"0.0\"]") : std::string("[\"0.0\", \"0.0\", \"0.0\"]")) + "\n";
    const std::string inlet = "type = \"stagnation_inlet\"\np0 = 1.2e5\nT0 = 320.0\n" +
                              std::string(N_DIM == 2 ? "direction = [1.0, 0.2]\n" : "direction = [1.0, 0.2, 0.1]\n");
    const auto rest = [&](const std::string & composition) {
        return "[numerics]\nriemann_solver = \"HLLC\"\n[numerics.face_reconstruction]\ntype = \"MUSCL\"\n"
               "[output]\ncheck_interval = 1000000\n" +
               duct_boundaries(inlet + composition, "type = \"p_out\"\np = 8.0e4\n");
    };
    Solver perfect;
    perfect.set_distributed(false);
    perfect.init(parse_toml(common + rest("") +
                            "[physics]\ntype = \"euler\"\ngamma = 1.4\np_ref = 101325.0\nT_ref = 300.0\n"
                            "rho_ref = 1.1768292682926829\n"));
    perfect.run();
    perfect.copy_device_to_host();
    Solver mixed;
    mixed.set_distributed(false);
    mixed.init(parse_toml(common + "Y = { AIR = 1.0 }\n" + rest("Y = { AIR = 1.0 }\n") + "[physics]\ntype = \"euler\"\ngas = \"mixture\"\n"
                          "mechanism = \"" + PERFECT_AIR + "\"\nphase = \"air\"\n"));
    mixed.run();
    mixed.copy_device_to_host();
    double worst = 0.0, inflow = 0.0;
    for (uint32_t c = 0; c < perfect.get_mesh()->n_cells; c++) {
        FOR_I_CONSERVATIVE {
            const double scale = (i == 0) ? 1.2 : (i == N_DIM + 1 ? 2.5e5 : 120.0);
            worst = std::max(worst, std::abs(double(mixed.h_conservatives(c, i) - perfect.h_conservatives(c, i))) / scale);
        }
        inflow = std::max(inflow, double(perfect.h_conservatives(c, 1)));
    }
    EXPECT_GT(inflow, 10.0) << "the inlet drives a flow";
    EXPECT_LT(worst, tol(1e-11, 1e-4));
}

namespace {

/** @brief Steady state of a uniform duct of thermally perfect air between a stagnation inlet and p_out. */
void expect_isentropic_duct(const std::string & reconstruction) {
    const double T0 = 2000.0, p0 = 2.0e5, p_e = 1.4e5;
    const std::string input =
        "[run]\nt_stop = 0.04\ncfl = 0.5\n" + mesh_block("cartesian", 12, 1, 1.0, 0.1) +
        "[initialize]\ntype = \"constant\"\np = 1.6e5\nT = 1900.0\nu = " + zero_velocity() +
        "\nX = { N2 = 0.79, O2 = 0.21 }\n"
        "[numerics]\nriemann_solver = \"HLLC\"\n[numerics.face_reconstruction]\n" + reconstruction +
        "[output]\ncheck_interval = 1000000\n"
        "[physics]\ntype = \"euler\"\ngas = \"mixture\"\nmechanism = \"" + H2O2 + "\"\n" +
        duct_boundaries("type = \"stagnation_inlet\"\np0 = 2.0e5\nT0 = 2000.0\nX = { N2 = 0.79, O2 = 0.21 }\n",
                        "type = \"p_out\"\np = 1.4e5\n");
    Solver solver;
    solver.set_distributed(false);
    solver.init(parse_toml(input));
    solver.run();
    solver.update_primitives();
    solver.copy_device_to_host();

    const chemistry::Mechanism mech = chemistry::read_mechanism(H2O2);
    const auto thermo = chemistry::make_thermo_table<Kokkos::HostSpace>(mech);
    std::vector<double> X(mech.n_species(), 0.0), Y(mech.n_species(), 0.0);
    X[size_t(mech.species_index("N2"))] = 0.79;
    X[size_t(mech.species_index("O2"))] = 0.21;
    double W_mix = 0.0;
    for (uint32_t k = 0; k < Y.size(); k++) W_mix += X[k] * mech.species[k].molecular_weight;
    for (uint32_t k = 0; k < Y.size(); k++) Y[k] = X[k] * mech.species[k].molecular_weight / W_mix;
    const double R = thermo.gas_constant(chemistry::MassFractions{Y.data()});
    // Exit temperature on the isentrope at p_e, by bisection
    double lo = 0.5 * T0, hi = T0;
    for (int it = 0; it < 80; it++) {
        const double mid = 0.5 * (lo + hi);
        (std::log(p_e / p0) < isentrope_by_quadrature(thermo, Y, R, T0, mid)[1] ? hi : lo) = mid;
    }
    const double T_e = 0.5 * (lo + hi);
    const double u_e = std::sqrt(2.0 * isentrope_by_quadrature(thermo, Y, R, T0, T_e)[0]);
    for (uint32_t c = 0; c < solver.get_mesh()->n_cells; c++) {
        EXPECT_NEAR(double(solver.h_primitives(c, 0)), u_e, tol(2e-4, 2e-3) * u_e);
        EXPECT_NEAR(double(solver.h_primitives(c, N_DIM + 1)), T_e, tol(2e-5, 2e-4) * T_e);
        EXPECT_NEAR(double(solver.h_primitives(c, N_DIM)), p_e, tol(2e-4, 2e-3) * p_e);
    }
}

} // namespace

TEST(StagnationInletTest, ThermallyPerfectDuctReachesTheIsentropicExitState) {
    expect_isentropic_duct("type = \"MUSCL\"\n");
    // A duct one cell across has no full-rank TENO stencils in 3D
    if constexpr (N_DIM == 2) {
        expect_isentropic_duct("type = \"TENO\"\norder = 3\n");
    }
}

TEST(StagnationInletTest, InvalidInputIsRejected) {
    const std::string head =
        "[run]\nn_steps = 1\ncfl = 0.4\n" + mesh_block("cartesian", 8, 2, 1.0, 0.25) +
        "[initialize]\ntype = \"constant\"\nrho = 1.0\np = 1.0\nu = " + zero_velocity() +
        "\n[physics]\ntype = \"euler\"\ngamma = 1.4\np_ref = 1.0\nT_ref = 1.0\nrho_ref = 1.0\n"
        "[output]\ncheck_interval = 1000000\n";
    const std::string outlet = "type = \"p_out\"\np = 1.0\n";
    const std::string wrong_way = N_DIM == 2 ? "direction = [-1.0, 0.0]\n" : "direction = [-1.0, 0.0, 0.0]\n";
    for (const std::string & left : {std::string("type = \"stagnation_inlet\"\nT0 = 1.0\n"),
                                     std::string("type = \"stagnation_inlet\"\np0 = 1.0\nT0 = -1.0\n"),
                                     "type = \"stagnation_inlet\"\np0 = 1.0\nT0 = 1.0\n" + wrong_way,
                                     std::string("type = \"stagnation_inlet\"\np0 = 1.0\nT0 = 1.0\ndirection = [0.0]\n")}) {
        Solver solver;
        EXPECT_THROW(solver.init(parse_toml(head + duct_boundaries(left, outlet))), std::runtime_error) << left;
    }
}

#if N_DIM == 2
namespace {

/** @brief Half-height of the planar nozzle: throat 1 at x = 2, 1.6 at both ends (x = 0 and 4). */
double nozzle_height(double x) { return 1.0 + 0.6 * (0.5 * x - 1.0) * (0.5 * x - 1.0); }

/** @brief Gmsh 2.2 file of the upper half of the nozzle: zones inlet, outlet, axis and wall. */
std::string nozzle_mesh(uint32_t nx, uint32_t ny) {
    std::ostringstream s;
    s.precision(17);
    s << "$MeshFormat\n2.2 0 8\n$EndMeshFormat\n$PhysicalNames\n4\n"
      << "1 1 \"axis\"\n1 2 \"outlet\"\n1 3 \"wall\"\n1 4 \"inlet\"\n$EndPhysicalNames\n";
    auto id = [&](uint32_t i, uint32_t j) { return j * (nx + 1) + i + 1; };
    s << "$Nodes\n" << (nx + 1) * (ny + 1) << "\n";
    for (uint32_t j = 0; j <= ny; j++) {
        for (uint32_t i = 0; i <= nx; i++) {
            const double x = 4.0 * i / nx;
            s << id(i, j) << " " << x << " " << nozzle_height(x) * j / ny << " 0\n";
        }
    }
    std::vector<std::string> elements;
    for (uint32_t i = 0; i < nx; i++) {
        elements.push_back("1 2 1 1 " + std::to_string(id(i, 0)) + " " + std::to_string(id(i + 1, 0)));
        elements.push_back("1 2 3 3 " + std::to_string(id(i + 1, ny)) + " " + std::to_string(id(i, ny)));
    }
    for (uint32_t j = 0; j < ny; j++) {
        elements.push_back("1 2 4 4 " + std::to_string(id(0, j + 1)) + " " + std::to_string(id(0, j)));
        elements.push_back("1 2 2 2 " + std::to_string(id(nx, j)) + " " + std::to_string(id(nx, j + 1)));
    }
    for (uint32_t j = 0; j < ny; j++) {
        for (uint32_t i = 0; i < nx; i++) {
            elements.push_back("3 2 0 1 " + std::to_string(id(i, j)) + " " + std::to_string(id(i + 1, j)) + " " +
                               std::to_string(id(i + 1, j + 1)) + " " + std::to_string(id(i, j + 1)));
        }
    }
    s << "$EndNodes\n$Elements\n" << elements.size() << "\n";
    for (size_t k = 0; k < elements.size(); k++) s << k + 1 << " " << elements[k] << "\n";
    s << "$EndElements\n";
    return s.str();
}

double bisect(const std::function<double(double)> & f, double lo, double hi) {
    const bool rising = f(hi) > f(lo);
    for (int it = 0; it < 200; it++) {
        const double mid = 0.5 * (lo + hi);
        ((f(mid) > 0.0) == rising ? hi : lo) = mid;
    }
    return 0.5 * (lo + hi);
}

/** @brief A / A* at Mach M. */
double area_ratio(double M, double g) {
    return std::pow(2.0 / (g + 1.0) * (1.0 + 0.5 * (g - 1.0) * M * M), 0.5 * (g + 1.0) / (g - 1.0)) / M;
}

/** @brief Stagnation pressure ratio across a normal shock at Mach M. */
double shock_total_pressure_ratio(double M, double g) {
    const double a = (g + 1.0) * M * M / ((g - 1.0) * M * M + 2.0);
    const double b = (g + 1.0) / (2.0 * g * M * M - (g - 1.0));
    return std::pow(a, g / (g - 1.0)) * std::pow(b, 1.0 / (g - 1.0));
}

double static_ratio(double M, double g) { return std::pow(1.0 + 0.5 * (g - 1.0) * M * M, -g / (g - 1.0)); }

} // namespace

TEST(StagnationInletTest, NozzleMatchesTheQuasiOneDimensionalShockSolution) {
    // Converging-diverging nozzle (exit and inlet areas 1.6 throat areas) fed
    // by p0, T0, with a back pressure that puts a normal shock in the
    // diverging part: the mass flow is the choked one, the shock stands where
    // the quasi-one-dimensional solution puts it, and the stagnation pressure
    // drops across it by the normal-shock ratio
    const double g = 1.4, p0 = 1.0, T0 = 1.0, A_e = 1.6;
    const double p_b = 0.75;
    // Shock area ratio A_s such that the subsonic exit behind it is at p_b
    auto exit_pressure = [&](double A_s) {
        const double M1 = bisect([&](double M) { return area_ratio(M, g) - A_s; }, 1.0, 5.0);
        const double r = shock_total_pressure_ratio(M1, g);
        const double M_e = bisect([&](double M) { return area_ratio(M, g) - A_e * r; }, 1e-6, 1.0);
        return p0 * r * static_ratio(M_e, g);
    };
    const double A_s = bisect([&](double A) { return exit_pressure(A) - p_b; }, 1.0 + 1e-9, A_e);
    const double x_s = 2.0 + 2.0 * std::sqrt((A_s - 1.0) / 0.6);
    const double M1 = bisect([&](double M) { return area_ratio(M, g) - A_s; }, 1.0, 5.0);
    const double r_exact = shock_total_pressure_ratio(M1, g);
    // Choked mass flow per unit depth through the half nozzle (throat height 1)
    const double T_star = 2.0 * T0 / (g + 1.0);
    const double p_star = p0 * std::pow(T_star / T0, g / (g - 1.0));
    const double mdot_exact = p_star / T_star * std::sqrt(g * T_star);

    const uint32_t nx = 100, ny = 6;
    const std::string mesh = write_temp("stagnation_nozzle.msh", nozzle_mesh(nx, ny));
    std::ostringstream s;
    s << "[run]\nt_stop = 40.0\ncfl = 0.8\n"
      << "[mesh]\ntype = \"file\"\nfilename = \"" << mesh << "\"\n"
      << "[initialize]\ntype = \"analytical\"\nrho = \"1.0 - 0.25 * x / 4\"\nu = [\"0.0\", \"0.0\"]\n"
      << "p = \"1.0 - 0.25 * x / 4\"\n"
      << "[[boundaries]]\nname = \"inlet\"\ntype = \"stagnation_inlet\"\np0 = 1.0\nT0 = 1.0\n"
      << "[[boundaries]]\nname = \"outlet\"\ntype = \"p_out\"\np = " << p_b << "\n"
      << "[[boundaries]]\nname = \"axis\"\ntype = \"symmetry\"\n"
      << "[[boundaries]]\nname = \"wall\"\ntype = \"symmetry\"\n"
      << "[numerics]\nriemann_solver = \"HLLC\"\n[numerics.face_reconstruction]\ntype = \"MUSCL\"\n"
      << "[physics]\ntype = \"euler\"\ngamma = 1.4\np_ref = 1.0\nT_ref = 1.0\nrho_ref = 1.0\n"
      << "[output]\ncheck_interval = 1000000\n";
    Solver solver;
    solver.set_distributed(false);
    solver.init(parse_toml(s.str()));
    solver.run();
    solver.update_primitives();
    solver.copy_device_to_host();
    const auto m = solver.get_mesh();

    // The volume integral of rho u_x is the mass flow times the length
    double rhou = 0.0;
    // Column averages of p and of the stagnation pressure
    std::vector<double> p(nx, 0.0), pt(nx, 0.0), area(nx, 0.0);
    for (uint32_t c = 0; c < m->n_cells; c++) {
        const double V = double(m->h_cell_volume(c));
        rhou += double(solver.h_conservatives(c, 1)) * V;
        const uint32_t i = std::min(nx - 1, static_cast<uint32_t>(double(m->h_cell_coords(c, 0)) / 4.0 * nx));
        const double u = double(solver.h_primitives(c, 0)), v = double(solver.h_primitives(c, 1));
        const double pc = double(solver.h_primitives(c, 2)), T = double(solver.h_primitives(c, 3));
        const double M2 = (u * u + v * v) / (g * T);
        p[i] += pc * V;
        pt[i] += pc * std::pow(1.0 + 0.5 * (g - 1.0) * M2, g / (g - 1.0)) * V;
        area[i] += V;
    }
    for (uint32_t i = 0; i < nx; i++) {
        p[i] /= area[i];
        pt[i] /= area[i];
    }
    const double mdot = rhou / 4.0;
    // Shock: where the column pressure crosses the mean of the exact pre- and post-shock pressures
    const double p1 = p0 * static_ratio(M1, g);
    const double M2s = std::sqrt((1.0 + 0.5 * (g - 1.0) * M1 * M1) / (g * M1 * M1 - 0.5 * (g - 1.0)));
    const double p2 = p0 * r_exact * static_ratio(M2s, g);
    double x_num = -1.0;
    for (uint32_t i = nx / 2; i + 1 < nx; i++) {
        const double mid = 0.5 * (p1 + p2);
        if (p[i] < mid && p[i + 1] >= mid) {
            const double xi = 4.0 * (i + 0.5) / nx;
            x_num = xi + (mid - p[i]) / (p[i + 1] - p[i]) * 4.0 / nx;
            break;
        }
    }
    const double dx = 4.0 / nx;
    const uint32_t i_s = static_cast<uint32_t>(x_s / dx);
    double pt_up = 0.0, pt_down = 0.0;
    for (uint32_t i = 0; i + 6 < i_s; i++) pt_up = std::max(pt_up, std::abs(pt[i] - p0));
    for (uint32_t i = i_s + 6; i < nx; i++) pt_down += pt[i] / (nx - i_s - 6);

    RecordProperty("mdot_error", std::to_string(mdot / mdot_exact - 1.0));
    RecordProperty("shock_x", std::to_string(x_num) + " exact " + std::to_string(x_s));
    EXPECT_NEAR(mdot, mdot_exact, 5e-3 * mdot_exact);
    EXPECT_NEAR(x_num, x_s, 2.0 * dx);
    EXPECT_LT(pt_up, 5e-3);
    // Behind the shock where it stands: the normal-shock loss of the Mach number there
    const double M1_num = bisect([&](double M) { return area_ratio(M, g) - nozzle_height(x_num); }, 1.0, 5.0);
    RecordProperty("total_pressure_ratio", std::to_string(pt_down / p0) + " exact " + std::to_string(r_exact) +
                                               " at the shock " + std::to_string(shock_total_pressure_ratio(M1_num, g)));
    EXPECT_NEAR(pt_down / p0, shock_total_pressure_ratio(M1_num, g), 1e-2);
    EXPECT_NEAR(pt_down / p0, r_exact, 2e-2);
}
#endif

TEST(MassFlowInletTest, GhostCarriesTheMassFluxAndTheOutgoingWave) {
    // The ghost has mass flux G through the face, temperature T and the
    // flow direction; its pressure keeps the interior's outgoing acoustic
    // wave p + rho c u_n. A heavier or lighter interior changes the ghost's
    // density, never its mass flux
    const double gamma = 1.4, R = 287.0, T = 400.0, G = 30.0;
    const Vec n = unit_vec(vec(-1.0, 0.2, 0.1));
    for (const Vec & d : {Vec{}, unit_vec(vec(1.0, 0.5, 0.0))}) {
        Vec flow = d;
        if (d == Vec{}) FOR_I_DIM flow[i] = -n[i];
        for (const double rho_i : {0.5, 1.0, 3.0}) {
            std::array<rtype, N_CONSERVATIVE> W_i{};
            W_i[0] = rtype(rho_i);
            W_i[1] = 25.0_r;
            W_i[2] = -4.0_r;
            W_i[N_DIM + 1] = rtype(1.0e5 * rho_i);
            BoundaryCondition bc;
            bc.type = BoundaryType::MASS_FLOW_INLET;
            bc.data[0] = rtype(T);
            FOR_I_DIM bc.data[1 + i] = d[i];
            bc.data[N_DIM + 1] = rtype(G);
            std::array<rtype, N_CONSERVATIVE> W_g{};
            bc.ghost_W(W_i.data(), n.data(), rtype(gamma), rtype(R), false, W_g.data());
            double flux = 0.0, u_n_i = 0.0, u_n_g = 0.0, along = 0.0, V2 = 0.0;
            FOR_I_DIM {
                flux -= double(W_g[0]) * double(W_g[1 + i]) * double(n[i]);
                u_n_i += double(W_i[1 + i]) * double(n[i]);
                u_n_g += double(W_g[1 + i]) * double(n[i]);
                along += double(W_g[1 + i]) * double(flow[i]);
                V2 += double(W_g[1 + i]) * double(W_g[1 + i]);
            }
            const double Z = std::sqrt(gamma * double(W_i[N_DIM + 1]) * double(W_i[0]));
            EXPECT_NEAR(flux, G, tol(1e-13, 1e-6) * G);
            EXPECT_NEAR(double(W_g[N_DIM + 1]) / (double(W_g[0]) * R), T, tol(1e-12, 1e-5) * T);
            EXPECT_NEAR(along * along, V2, tol(1e-12, 1e-5) * V2);
            EXPECT_GT(along, 0.0);
            EXPECT_NEAR(double(W_g[N_DIM + 1]) + Z * u_n_g, double(W_i[N_DIM + 1]) + Z * u_n_i,
                        tol(1e-11, 1e-5) * double(W_i[N_DIM + 1]));
        }
    }
}

namespace {

/** @brief Air through the left end of a duct closed at the right, heated in its middle: mass added against mdot t. */
struct ClosedDuct {
    double added = 0.0;     // M(t) - M(0)
    double expected = 0.0;  // mdot t
};

ClosedDuct closed_heated_duct(const std::string & inlet, const std::string & scheme, const std::string & gas,
                              double mdot) {
    const std::string common =
        mesh_block("cartesian", 40, 2, 1.0, 0.1) + "[source]\nrhoE = \"x > 0.3 and x < 0.7 ? 3.0e7 : 0.0\"\n" + scheme +
        "[output]\ncheck_interval = 1000000\n" + gas +
        duct_boundaries(inlet, "type = \"wall_adiabatic\"\n");
    const std::string init = "[initialize]\ntype = \"constant\"\np = 1.0e5\nT = 300.0\nu = " + zero_velocity() + "\n" +
                             (gas.find("mixture") != std::string::npos ? "X = { N2 = 0.79, O2 = 0.21 }\n" : "");
    auto mass = [](const Solver & s) {
        double m = 0.0;
        for (uint32_t c = 0; c < s.get_mesh()->n_owned(); c++) {
            m += double(s.h_conservatives(c, 0)) * double(s.get_mesh()->h_cell_volume(c));
        }
        return m;
    };
    Solver start;
    start.set_distributed(false);
    start.init(parse_toml("[run]\nn_steps = 0\ncfl = 0.5\n" + init + common));
    start.copy_device_to_host();
    Solver solver;
    solver.set_distributed(false);
    solver.init(parse_toml("[run]\nt_stop = 2.0e-3\ncfl = 0.5\n" + init + common));
    solver.run();
    solver.copy_device_to_host();
    return {mass(solver) - mass(start), mdot * double(solver.get_time())};
}

const std::string PERFECT_GAS_AIR =
    "[physics]\ntype = \"euler\"\ngamma = 1.4\np_ref = 1.0e5\nT_ref = 300.0\nrho_ref = 1.1614401858304297\n";

} // namespace

TEST(MassFlowInletTest, ClosedHeatedDuctGainsExactlyTheInflow) {
    // Heating and the rising pressure change the inlet density all along;
    // the mass in the duct still grows by exactly mdot t
    const double A = N_DIM == 2 ? 0.1 : 0.01, G = 20.0;
    const std::string mixture = "[physics]\ntype = \"euler\"\ngas = \"mixture\"\nmechanism = \"" + H2O2 + "\"\n";
    const std::string X = "X = { N2 = 0.79, O2 = 0.21 }\n";
    const std::string flux = "type = \"mass_flow_inlet\"\nT = 300.0\nmass_flux = 20.0\n";
    const std::string total = "type = \"mass_flow_inlet\"\nT = 300.0\nmdot = " + std::to_string(G * A) +
                              "\nprofile = \"1 + 4 * y\"\n";
    const std::string muscl = "[numerics]\nriemann_solver = \"HLLC\"\n[numerics.face_reconstruction]\ntype = \"MUSCL\"\n";
    // TENO has no full-rank stencils in a 3D duct two cells across
    const std::string teno = std::string("[numerics]\nriemann_solver = \"Roe\"\n[numerics.face_reconstruction]\n") +
                             (N_DIM == 2 ? "type = \"TENO\"\norder = 4\n" : "type = \"MUSCL\"\n");
    const std::string double_flux = "[numerics]\nriemann_solver = \"HLLC\"\ndouble_flux = true\n"
                                    "[numerics.face_reconstruction]\ntype = \"MUSCL\"\n";
    for (const auto & [inlet, scheme, gas] : {std::array<std::string, 3>{flux, muscl, PERFECT_GAS_AIR},
                                             std::array<std::string, 3>{total, teno, PERFECT_GAS_AIR},
                                             std::array<std::string, 3>{flux + X, double_flux, mixture},
                                             std::array<std::string, 3>{total + X, muscl, mixture}}) {
        const ClosedDuct r = closed_heated_duct(inlet, scheme, gas, G * A);
        EXPECT_NEAR(r.added, r.expected, tol(1e-11, 1e-4) * r.expected) << inlet << scheme << gas;
    }
    // upt fixes the velocity: its mass flow follows the inlet density
    const double u = G / 1.1614401858304297;
    const ClosedDuct fixed = closed_heated_duct(
        "type = \"upt\"\nu = [" + std::to_string(u) + (N_DIM == 2 ? ", 0.0]" : ", 0.0, 0.0]") +
            "\np = 1.0e5\nT = 300.0\n",
        muscl, PERFECT_GAS_AIR, G * A);
    EXPECT_GT(std::abs(fixed.added / fixed.expected - 1.0), 1e-2);
}

namespace {

/** @brief Inlet pressure and exit state [p1, u2, T2] of Rayleigh flow: mass flux G at T1, heat Q per unit area, exit at p2. */
std::array<double, 3> rayleigh(double G, double T1, double Q, double p2, double gamma, double R) {
    const double cp = gamma * R / (gamma - 1.0);
    std::array<double, 3> out{};
    auto exit_pressure = [&](double p1) {
        const double u1 = G * R * T1 / p1;
        const double I = p1 + G * u1;
        const double H = cp * T1 + 0.5 * u1 * u1 + Q / G;
        // cp (I - G u2) u2 / (G R) + u2^2 / 2 = H, subsonic root
        const double a = 0.5 - cp / R, b = cp * I / (G * R), c = -H;
        const double u2 = (-b + std::sqrt(b * b - 4.0 * a * c)) / (2.0 * a);
        out = {p1, u2, (I - G * u2) * u2 / (G * R)};
        return I - G * u2;
    };
    double lo = p2, hi = 3.0 * p2;
    for (int it = 0; it < 200; it++) {
        const double mid = 0.5 * (lo + hi);
        (exit_pressure(mid) > p2 ? hi : lo) = mid;
    }
    exit_pressure(0.5 * (lo + hi));
    return out;
}

} // namespace

TEST(MassFlowInletTest, HeatedDuctMatchesRayleighFlow) {
    // Steady flow at mass flux G and 300 K through a duct heated between
    // x = 0.4 and 0.6 to a 1 bar outlet: the inlet pressure the flow needs
    // and the exit state are those of Rayleigh flow
    const double gamma = 1.4, R = 287.0, G = 100.0, T1 = 300.0, q = 1.0e8, p2 = 1.0e5;
    const auto [p1, u2, T2] = rayleigh(G, T1, q * 0.2, p2, gamma, R);
    const std::string input =
        "[run]\nt_stop = 0.06\ncfl = 0.8\n" + mesh_block("cartesian", 50, 1, 1.0, 0.1) +
        "[initialize]\ntype = \"constant\"\np = 1.0e5\nT = 300.0\nu = " + zero_velocity() + "\n" +
        "[source]\nrhoE = \"x > 0.4 and x < 0.6 ? 1.0e8 : 0.0\"\n" +
        "[numerics]\nriemann_solver = \"HLLC\"\n[numerics.face_reconstruction]\ntype = \"MUSCL\"\n"
        "[output]\ncheck_interval = 1000000\n" + PERFECT_GAS_AIR +
        duct_boundaries("type = \"mass_flow_inlet\"\nT = 300.0\nmass_flux = 100.0\n", "type = \"p_out\"\np = 1.0e5\n");
    Solver solver;
    solver.set_distributed(false);
    solver.init(parse_toml(input));
    solver.run();
    solver.update_primitives();
    solver.copy_device_to_host();
    const auto m = solver.get_mesh();
    double p_in = 0.0, u_out = 0.0, T_out = 0.0, n_in = 0.0, n_out = 0.0;
    for (uint32_t c = 0; c < m->n_cells; c++) {
        const double x = double(m->h_cell_coords(c, 0));
        if (x < 0.1) {
            p_in += double(solver.h_primitives(c, N_DIM));
            n_in += 1.0;
        } else if (x > 0.9) {
            u_out += double(solver.h_primitives(c, 0));
            T_out += double(solver.h_primitives(c, N_DIM + 1));
            n_out += 1.0;
        }
    }
    RecordProperty("rayleigh", "p1 " + std::to_string(p_in / n_in) + " exact " + std::to_string(p1) + ", T2 " +
                                   std::to_string(T_out / n_out) + " exact " + std::to_string(T2) + ", u2 " +
                                   std::to_string(u_out / n_out) + " exact " + std::to_string(u2));
    EXPECT_NEAR(p_in / n_in, p1, tol(1e-4, 1e-3) * p1);
    EXPECT_NEAR(u_out / n_out, u2, tol(1e-4, 1e-3) * u2);
    EXPECT_NEAR(T_out / n_out, T2, tol(1e-4, 1e-3) * T2);
}

TEST(MassFlowInletTest, InvalidInputIsRejected) {
    const std::string head =
        "[run]\nn_steps = 1\ncfl = 0.4\n" + mesh_block("cartesian", 8, 2, 1.0, 0.25) +
        "[initialize]\ntype = \"constant\"\nrho = 1.0\np = 1.0\nu = " + zero_velocity() +
        "\n[physics]\ntype = \"euler\"\ngamma = 1.4\np_ref = 1.0\nT_ref = 1.0\nrho_ref = 1.0\n"
        "[output]\ncheck_interval = 1000000\n";
    const std::string outlet = "type = \"p_out\"\np = 1.0\n";
    for (const std::string & left : {std::string("type = \"mass_flow_inlet\"\nT = 1.0\n"),
                                     std::string("type = \"mass_flow_inlet\"\nmass_flux = 1.0\n"),
                                     std::string("type = \"mass_flow_inlet\"\nT = 1.0\nmass_flux = 1.0\nmdot = 1.0\n"),
                                     std::string("type = \"mass_flow_inlet\"\nT = 1.0\nmass_flux = -1.0\n"),
                                     std::string("type = \"mass_flow_inlet\"\nT = 1.0\nmass_flux = \"y - 0.5\"\n"),
                                     std::string("type = \"mass_flow_inlet\"\nT = 1.0\nmass_flux = 1.0\nprofile = \"y\"\n"),
                                     std::string("type = \"mass_flow_inlet\"\nT = 1.0\nmdot = 1.0\nprofile = \"0 * y\"\n"),
                                     std::string("type = \"mass_flow_inlet\"\nT = 1.0\nmass_flux = \"1 + t\"\n"),
                                     std::string("type = \"mass_flow_inlet\"\nT = 1.0\nmass_flux = 1.0\ndirection = ") +
                                         (N_DIM == 2 ? "[-1.0, 0.0]" : "[-1.0, 0.0, 0.0]") + "\n"}) {
        Solver solver;
        EXPECT_THROW(solver.init(parse_toml(head + duct_boundaries(left, outlet))), std::runtime_error) << left;
    }
}

namespace {

/** @brief A time-dependent boundary case: the left (or top) condition and the rest of the input. */
struct ForcedCase {
    std::string name;
    std::string left;
    std::string top;
    std::string physics;
};

/** @brief The input of a forced case to t = 0.3, with `run` the rest of its [run] table. */
std::string forced_input(const ForcedCase & c, const std::string & run) {
    const bool wall = !c.top.empty();
    std::ostringstream s;
    s << "[run]\nt_stop = 0.3\n" << run << mesh_block("cartesian", wall ? 4 : 40, wall ? 8 : 1, 1.0, wall ? 1.0 : 0.1)
      << "[initialize]\ntype = \"constant\"\np = 1.0\nT = 1.0\nu = " << (N_DIM == 2 ? "[0.5, 0.0]" : "[0.5, 0.0, 0.0]")
      << "\n[numerics]\nriemann_solver = \"HLLC\"\n[numerics.face_reconstruction]\ntype = \"MUSCL\"\n"
      << "[output]\ncheck_interval = 1000000\n" << c.physics;
    if (wall) {
        s << "[[boundaries]]\nname = \"left\"\ntype = \"symmetry\"\n[[boundaries]]\nname = \"right\"\ntype = \"symmetry\"\n"
          << "[[boundaries]]\nname = \"bottom\"\ntype = \"wall_adiabatic\"\n[[boundaries]]\nname = \"top\"\n" << c.top;
        if (N_DIM == 3) {
            s << "[[boundaries]]\nname = \"back\"\ntype = \"symmetry\"\n[[boundaries]]\nname = \"front\"\ntype = \"symmetry\"\n";
        }
    } else {
        s << duct_boundaries(c.left, "type = \"p_out\"\np = 1.0\n");
    }
    return s.str();
}

/** @brief Density (or, for a moving wall, x-velocity) of every cell at t = 0.3 with a fixed dt. */
std::vector<double> forced_run(const ForcedCase & c, double dt) {
    const bool wall = !c.top.empty();
    std::ostringstream run;
    run << "dt = " << dt << "\n";
    Solver solver;
    solver.set_distributed(false);
    solver.init(parse_toml(forced_input(c, run.str())));
    solver.run();
    solver.update_primitives();
    solver.copy_device_to_host();
    std::vector<double> out;
    for (uint32_t i = 0; i < solver.get_mesh()->n_cells; i++) {
        out.push_back(wall ? double(solver.h_primitives(i, 0)) : double(solver.h_conservatives(i, 0)));
    }
    return out;
}

double max_difference(const std::vector<double> & a, const std::vector<double> & b) {
    double m = 0.0;
    for (size_t i = 0; i < a.size(); i++) m = std::max(m, std::abs(a[i] - b[i]));
    return m;
}

const std::string UNIT_GAS = "[physics]\ntype = \"euler\"\ngamma = 1.4\np_ref = 1.0\nT_ref = 1.0\nrho_ref = 1.0\n";

} // namespace

TEST(ForcedInletTest, TimeDependentConditionsConvergeAtTheIntegratorsOrder) {
    // Conditions that vary in time are evaluated at every Runge-Kutta stage
    // time: with SSPRK3 the solution converges in dt at third order on a
    // fixed mesh (taking them at the start of each step would give first
    // order). nscbc_inlet advances its boundary point once per step (forward
    // Euler on the LODI relation), so it converges at first order. Each
    // forcing must also move the solution well beyond those errors.
    const std::string u0 = N_DIM == 2 ? "[0.5, 0.0]" : "[0.5, 0.0, 0.0]";
    const std::string u_t = N_DIM == 2 ? "[\"0.5 + 0.1 * sin(40 * t)\", \"0\"]" : "[\"0.5 + 0.1 * sin(40 * t)\", \"0\", \"0\"]";
    // nscbc_inlet, first order, at a lower frequency and smaller steps so that its error stays well below its
    // forcing; the others at steps whose errors stand well above single-precision round-off
    const std::string u_slow = N_DIM == 2 ? "[\"0.5 + 0.1 * sin(20 * t)\", \"0\"]" : "[\"0.5 + 0.1 * sin(20 * t)\", \"0\", \"0\"]";
    const std::string wall_0 = N_DIM == 2 ? "[0.0, 0.0]" : "[0.0, 0.0, 0.0]";
    const std::string wall_u = N_DIM == 2 ? "[\"0.2 * sin(40 * t)\", \"0\"]" : "[\"0.2 * sin(40 * t)\", \"0\", \"0\"]";
    const std::string viscous =
        "[physics]\ntype = \"navier_stokes\"\ngamma = 1.4\np_ref = 1.0\nT_ref = 1.0\nrho_ref = 1.0\nmu = 0.02\nPr = 0.72\n";
    struct Case {
        ForcedCase forced, steady;
        double order;
        double dt = 0.0075;  // coarse; then dt / 2, and dt / 8 as the reference
    };
    const Case cases[] = {
        {{"upt T(t)", "type = \"upt\"\nu = " + u0 + "\np = 1.0\nT = \"1 + 0.2 * sin(40 * t)\"\n", "", UNIT_GAS},
         {"", "type = \"upt\"\nu = " + u0 + "\np = 1.0\nT = 1.0\n", "", UNIT_GAS}, 2.6},
        {{"upt u(t)", "type = \"upt\"\nu = " + u_t + "\np = 1.0\nT = 1.0\n", "", UNIT_GAS},
         {"", "type = \"upt\"\nu = " + u0 + "\np = 1.0\nT = 1.0\n", "", UNIT_GAS}, 2.6},
        {{"nscbc_inlet u(t)", "type = \"nscbc_inlet\"\nu = " + u_slow + "\np = 1.0\nT = 1.0\nL = 1.0\n", "", UNIT_GAS},
         {"", "type = \"nscbc_inlet\"\nu = " + u0 + "\np = 1.0\nT = 1.0\nL = 1.0\n", "", UNIT_GAS}, 0.9, 0.004},
        {{"stagnation_inlet p0(t)", "type = \"stagnation_inlet\"\np0 = \"1.3 + 0.1 * sin(40 * t)\"\nT0 = 1.2\n", "", UNIT_GAS},
         {"", "type = \"stagnation_inlet\"\np0 = 1.3\nT0 = 1.2\n", "", UNIT_GAS}, 2.6},
        {{"mass_flow_inlet mass_flux(t)", "type = \"mass_flow_inlet\"\nmass_flux = \"0.5 + 0.1 * sin(40 * t)\"\nT = 1.0\n", "",
          UNIT_GAS},
         {"", "type = \"mass_flow_inlet\"\nmass_flux = 0.5\nT = 1.0\n", "", UNIT_GAS}, 2.6},
        {{"moving wall u(t)", "", "type = \"wall_adiabatic\"\nu = " + wall_u + "\n", viscous},
         {"", "", "type = \"wall_adiabatic\"\nu = " + wall_0 + "\n", viscous}, 2.6},
    };
    for (const Case & c : cases) {
        const std::vector<double> reference = forced_run(c.forced, c.dt / 8);
        const std::vector<double> coarse = forced_run(c.forced, c.dt);
        const double e1 = max_difference(coarse, reference);
        const double e2 = max_difference(forced_run(c.forced, c.dt / 2), reference);
        const double forcing = max_difference(coarse, forced_run(c.steady, c.dt));
        const double order = std::log2(e1 / e2);
        RecordProperty(c.forced.name, std::to_string(order) + " (errors " + std::to_string(e1) + ", " +
                                          std::to_string(e2) + "; forcing " + std::to_string(forcing) + ")");
        EXPECT_GT(forcing, 20.0 * e1) << c.forced.name;
        // Below the round-off of the accumulated steps, convergence shows as errors that fall to that floor
        // from at most what the order allows (taking the conditions at the start of the step gives far more)
        const double floor = 50.0 * double(std::numeric_limits<rtype>::epsilon());
        if (e2 > floor) {
            EXPECT_GT(order, c.order) << c.forced.name << ": errors " << e1 << ", " << e2;
        } else {
            EXPECT_LT(e1, std::exp2(c.order) * floor) << c.forced.name << ": errors " << e1 << ", " << e2;
        }
    }
}

TEST(ForcedInletTest, QuantitiesWithUnitsAreNumbersNotExpressions) {
    // "1 bar" and "300 K" are the numbers 1e5 and 300, uniform along the inlet;
    // the expression of y varies along it
    const std::string u = N_DIM == 2 ? "u = [\"10 m/s\", 0.0]\n" : "u = [\"10 m/s\", 0.0, 0.0]\n";
    const BoundaryFields with_units(parse_toml("p = \"1 bar\"\nT = \"300 K\"\n" + u), BoundaryType::UPT, "b");
    EXPECT_FALSE(with_units.varies());
    BoundaryCondition bc;
    with_units.apply(bc, Point{}, 0.0, 287.0);
    EXPECT_DOUBLE_EQ(double(bc.data[N_DIM + 1]), double(rtype(1e5)));
    EXPECT_DOUBLE_EQ(double(bc.data[1]), 10.0);
    EXPECT_DOUBLE_EQ(with_units.temperature(Point{}, 0.0), 300.0);
    const BoundaryFields expression(parse_toml("p = 1.0e5\nT = \"300 + y\"\n" + u), BoundaryType::UPT, "b");
    EXPECT_TRUE(expression.varies());
    EXPECT_FALSE(expression.depends_on_time());
}

TEST(ForcedInletTest, NumberOfTThatTurnsNegativeIsAnInputErrorWithFaceAndTime) {
    // T(t) crosses zero at t = 0.1: the run stops at the next check with the
    // key, the value, the time and the face
    try {
        forced_run({"T(t) < 0", "type = \"upt\"\nu = " + std::string(N_DIM == 2 ? "[0.5, 0.0]" : "[0.5, 0.0, 0.0]") +
                                    "\np = 1.0\nT = \"1 - 10 * t\"\n", "", UNIT_GAS},
                   0.004);
        FAIL() << "no error";
    } catch (const InputError & e) {
        const std::string what = e.what();
        EXPECT_NE(what.find("boundaries[name = \"left\"].T must be positive (-"), std::string::npos) << what;
        EXPECT_NE(what.find("at t = 0.1"), std::string::npos) << what;
        EXPECT_NE(what.find(", face (0, "), std::string::npos) << what;
    }
}

TEST(ForcedInletTest, AValueOutOfRangeOnSomeRanksStopsEveryRank) {
    // Distributed, only the ranks holding the inlet see T turn negative; the others throw too
    Solver solver;
    solver.init(parse_toml(forced_input({"T(t) < 0", "type = \"upt\"\nu = " +
                                                         std::string(N_DIM == 2 ? "[0.5, 0.0]" : "[0.5, 0.0, 0.0]") +
                                                         "\np = 1.0\nT = \"1 - 10 * t\"\n", "", UNIT_GAS},
                                        "dt = 0.004\n")));
    EXPECT_THROW(solver.run(), InputError);
}

TEST(ForcedInletTest, ConditionsOfTKeepTheStepGraph) {
    // Single-gas conditions of t (an upt inlet, a moving wall) run as one
    // CUDA graph per step, which replays their device evaluation at each
    // stage time and, for the CFL time step, at the step's start: the same
    // solution as stepping kernel by kernel
#ifndef KOKKOS_ENABLE_CUDA
    GTEST_SKIP() << "CUDA builds only";
#else
    const std::string u = N_DIM == 2 ? "[\"0.5 + 0.1 * sin(20 * t)\", \"0\"]" : "[\"0.5 + 0.1 * sin(20 * t)\", \"0\", \"0\"]";
    const std::string wall_u = N_DIM == 2 ? "[\"0.2 * sin(20 * t)\", \"0\"]" : "[\"0.2 * sin(20 * t)\", \"0\", \"0\"]";
    const std::string viscous =
        "[physics]\ntype = \"navier_stokes\"\ngamma = 1.4\np_ref = 1.0\nT_ref = 1.0\nrho_ref = 1.0\nmu = 0.02\nPr = 0.72\n";
    const ForcedCase cases[] = {
        {"upt u(t), T(t)", "type = \"upt\"\nu = " + u + "\np = 1.0\nT = \"1 + 0.2 * sin(20 * t)\"\n", "", UNIT_GAS},
        {"moving wall u(t)", "", "type = \"wall_adiabatic\"\nu = " + wall_u + "\n", viscous},
    };
    for (const ForcedCase & c : cases) {
        auto run = [&](const bool graphs) {
            auto solver = std::make_unique<Solver>();
            solver->init(parse_toml(forced_input(c, std::string("cfl = 0.5\ncuda_graphs = ") + (graphs ? "true" : "false") + "\n")));
            EXPECT_EQ(solver->step_graph_unsupported(), graphs ? "" : "off ([run] cuda_graphs)") << c.name;
            solver->run();
            solver->copy_device_to_host();
            return solver;
        };
        const auto graph = run(true), kernels = run(false);
        EXPECT_GT(graph->get_graph_steps(), 10u) << c.name;
        EXPECT_EQ(kernels->get_graph_steps(), 0u) << c.name;
        EXPECT_EQ(graph->get_time(), kernels->get_time()) << c.name;
        for (uint32_t i = 0; i < graph->get_mesh()->n_cells; i++) {
            for (uint8_t v = 0; v < N_CONSERVATIVE; v++) {
                ASSERT_EQ(graph->h_conservatives(i, v), kernels->h_conservatives(i, v)) << c.name << ", cell " << i;
            }
        }
    }
#endif
}

TEST(ForcedInletTest, ExpressionThatLoopsForeverIsAnInputError) {
    // Evaluated on the device at the stage times, a loop that never ends past
    // t = 0.01 stops at the iteration limit and the run reports it
    try {
        forced_run({"T(t) loops", "type = \"upt\"\nu = " + std::string(N_DIM == 2 ? "[0.5, 0.0]" : "[0.5, 0.0, 0.0]") +
                                      "\np = 1.0\nT = \"var i := 0; while (t > 0.01 and i >= 0) { i += 1 }; 1 + t\"\n",
                    "", UNIT_GAS},
                   0.004);
        FAIL() << "no error";
    } catch (const InputError & e) {
        EXPECT_NE(std::string(e.what()).find("loop iterations"), std::string::npos) << e.what();
    }
}

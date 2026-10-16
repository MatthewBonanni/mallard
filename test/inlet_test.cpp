/**
 * @file inlet_test.cpp
 * @author Matthew Bonanni (mbonanni001@gmail.com)
 * @brief Tests of the stagnation inlet.
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
#include <sstream>
#include <string>
#include <vector>

#include "boundary.h"
#include "gmsh_fixtures.h"
#include "inflow.h"
#include "mechanism.h"
#include "solver.h"
#include "test_fixtures.h"
#include "thermo.h"

namespace {

const std::string SOURCE_DIR = MALLARD_SOURCE_DIR;
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
    const std::string rest =
        "[numerics]\nriemann_solver = \"HLLC\"\n[numerics.face_reconstruction]\ntype = \"MUSCL\"\n"
        "[output]\ncheck_interval = 1000000\n" +
        duct_boundaries("type = \"stagnation_inlet\"\np0 = 1.2e5\nT0 = 320.0\n" +
                            std::string(N_DIM == 2 ? "direction = [1.0, 0.2]\n" : "direction = [1.0, 0.2, 0.1]\n") +
                            "Y = { AIR = 1.0 }\n",
                        "type = \"p_out\"\np = 8.0e4\n");
    Solver perfect;
    perfect.init(parse_toml(common + rest +
                            "[physics]\ntype = \"euler\"\ngamma = 1.4\np_ref = 101325.0\nT_ref = 300.0\n"
                            "rho_ref = 1.1768292682926829\n"));
    perfect.run();
    perfect.copy_device_to_host();
    Solver mixed;
    mixed.init(parse_toml(common + "Y = { AIR = 1.0 }\n" + rest + "[physics]\ntype = \"euler\"\ngas = \"mixture\"\n"
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

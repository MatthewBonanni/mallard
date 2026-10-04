/**
 * @file source_test.cpp
 * @author Matthew Bonanni (mbonanni001@gmail.com)
 * @brief Tests for source terms.
 * @version 0.1
 * @date 2026-10-02
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 *
 */

#include <gtest/gtest.h>
#include <Kokkos_Core.hpp>

#include <cmath>
#include <sstream>
#include <string>

#include "test_fixtures.h"
#include "solver.h"

namespace {

std::string box_input(const std::string & source, const std::string & init, const std::string & run,
                      const std::string & recon = "MUSCL") {
    std::ostringstream s;
    s << "[run]\n" << run
      << "[mesh]\ntype = \"cartesian_tri\"\nNx = 16\nNy = 16\nLx = 1.0\nLy = 1.0\n"
      << "[initialize]\n" << init
      << "[[boundaries]]\nname = \"left\"\ntype = \"symmetry\"\n"
      << "[[boundaries]]\nname = \"right\"\ntype = \"symmetry\"\n"
      << "[[boundaries]]\nname = \"top\"\ntype = \"symmetry\"\n"
      << "[[boundaries]]\nname = \"bottom\"\ntype = \"symmetry\"\n"
      << "[numerics]\nriemann_solver = \"HLLC\"\ntime_integrator = \"SSPRK3\"\n"
      << "[numerics.face_reconstruction]\ntype = \"" << recon << "\"\n"
      << "[physics]\ntype = \"euler\"\ngamma = 1.4\np_ref = 1.0\nT_ref = 1.0\nrho_ref = 1.0\n"
      << "[output]\ncheck_interval = 1000000\n"
      << "[source]\n" << source;
    return s.str();
}

const std::string REST = "type = \"analytical\"\nrho = \"1.0\"\nu = [\"0.0\", \"0.0\"]\np = \"1.0\"\n";

double hydrostatic_spurious_velocity(uint32_t n, const std::string & recon) {
    std::string input = box_input("gravity = [0.0, -1.0]\n",
                                  "type = \"analytical\"\nrho = \"exp(-y)\"\nu = [\"0.0\", \"0.0\"]\np = \"exp(-y)\"\n",
                                  "t_stop = 0.5\ncfl = 0.5\n", recon);
    const std::string from = "Nx = 16\nNy = 16";
    input.replace(input.find(from), from.size(), "Nx = " + std::to_string(n) + "\nNy = " + std::to_string(n));
    Solver solver;
    solver.init(parse_toml(input));
    solver.run();
    solver.update_primitives();
    solver.copy_device_to_host();
    double v_max = 0.0;
    for (uint32_t i = 0; i < solver.get_mesh()->n_cells; i++) {
        v_max = std::max(v_max, std::hypot(double(solver.h_primitives(i, 0)), double(solver.h_primitives(i, 1))));
    }
    return v_max;
}

} // namespace

TEST(SourceTest, SpatialMassSourceAddsExactMass) {
    // Closed box: total mass grows by integral(S) * t, and the steady source
    // is evaluated once
    Solver solver;
    solver.init(parse_toml(box_input("rho = \"0.2 * (1 + x)\"\n", REST, "t_stop = 0.5\ncfl = 0.5\n")));
    const auto before = solver.integrate_conservatives();
    solver.run();
    const auto after = solver.integrate_conservatives();
    // integral of 0.2 (1 + x) over the unit square = 0.3 (exact for the centroid rule: linear)
    EXPECT_NEAR(after[0] - before[0], 0.3 * 0.5, precision_tol<double>(1e-12, 5e-5));  // single: rounding in ~100 SSPRK3 steps
}

TEST(SourceTest, TimeDependentEnergySourceIsIntegratedInTime) {
    // d(total energy)/dt = sin(t) over a unit area: Delta E = 1 - cos(t)
    Solver solver;
    solver.init(parse_toml(box_input("rhoE = \"sin(t)\"\ntime_dependent = true\n", REST, "t_stop = 1.0\ncfl = 0.5\n")));
    const auto before = solver.integrate_conservatives();
    solver.run();
    const auto after = solver.integrate_conservatives();
    EXPECT_NEAR(after[3] - before[3], 1.0 - std::cos(1.0), precision_tol<double>(1e-7, 1e-4));  // single: rounding in ~100 SSPRK3 steps
}

TEST(SourceTest, GravityPullsGasDown) {
    // Gas initially at rest with uniform pressure falls: total y-momentum
    // decreases at rate (total mass) * g while the box is still balanced by walls
    Solver solver;
    solver.init(parse_toml(box_input("gravity = [0.0, -1.0]\n", REST, "t_stop = 0.05\ncfl = 0.5\n")));
    solver.run();
    const auto total = solver.integrate_conservatives();
    // Walls push back through pressure, so momentum is at most -g * M * t
    EXPECT_LT(total[2], -0.01);
    EXPECT_GT(total[2], -0.05);
}

TEST(SourceTest, IsothermalAtmosphereConvergesToHydrostaticEquilibrium) {
    // rho = p = exp(-y) with R = T = 1 balances gravity g = -1. The scheme is
    // not exactly well balanced, so small spurious velocities remain and
    // vanish under refinement.
    const double v16 = hydrostatic_spurious_velocity(16, "MUSCL"), v32 = hydrostatic_spurious_velocity(32, "MUSCL");
    // Hydrostatic wall ghosts: second-order convergence and small magnitude
    // (mirrored wall pressures gave 0.06 and 0.03). The low-Mach correction
    // damps these acoustic errors with z = 0.1 instead of 1 in gas at rest,
    // which raises them about fourfold (v32 = 5.9e-4, 1.2e-4 without it) but
    // keeps the order
    EXPECT_LT(v32, 0.35 * v16);
    EXPECT_LT(v32, 1e-3);
}

TEST(SourceTest, TENOHydrostaticAtmosphereHasSmallSpuriousVelocity) {
    // TENO mirror cells continue the hydrostatic pressure gradient like the
    // gradient ghosts; mirrored pressures drive a wall-normal jet
    EXPECT_LT(hydrostatic_spurious_velocity(16, "TENO"), 1e-3);
}

namespace {

/**
 * @brief Channel of half-height h = 1 between no-slip adiabatic walls,
 *        periodic in x, with nu = 0.01 and sound speed 1; with hold, the bulk
 *        momentum is held at 0.1 by the mass flow forcing.
 */
std::string channel_input(const std::string & u, const std::string & run, bool hold = true) {
    std::ostringstream s;
    s << "[run]\n" << run
      << "[mesh]\ntype = \"cartesian\"\nNx = 4\nNy = 32\nLx = 0.5\nLy = 2.0\nperiodic = [\"x\"]\n"
      << "[initialize]\ntype = \"analytical\"\nrho = \"1.0\"\nu = [\"" << u << "\", \"0.0\"]\n"
      << "p = \"0.7142857142857143\"\n"
      << "[[boundaries]]\nname = \"top\"\ntype = \"wall_adiabatic\"\n"
      << "[[boundaries]]\nname = \"bottom\"\ntype = \"wall_adiabatic\"\n"
      << "[numerics]\nriemann_solver = \"HLLC\"\ntime_integrator = \"SSPRK3\"\n"
      << "[numerics.face_reconstruction]\ntype = \"MUSCL\"\nlimiter = \"none\"\n"
      << "[physics]\ntype = \"navier_stokes\"\ngamma = 1.4\np_ref = 0.7142857142857143\nT_ref = 1.0\n"
      << "rho_ref = 1.0\nmu = 0.01\n"
      << "[output]\ncheck_interval = 1000000\n";
    if (hold) s << "[source]\nmass_flow = [0.1, 0.0]\n";
    return s.str();
}

} // namespace

TEST(SourceTest, MassFlowForcingHoldsTheBulkMomentumAgainstWallFriction) {
    // Plug flow decelerates at the walls; the force restores the target bulk
    // momentum every step (to rounding), and without it the bulk decays
    const double volume = 0.5 * 2.0;
    Solver held;
    held.init(parse_toml(channel_input("0.1", "n_steps = 200\ncfl = 0.5\n")));
    held.run();
    EXPECT_NEAR(double(held.integrate_conservatives()[1]) / volume, 0.1, precision_tol<double>(1e-12, 1e-6));
    EXPECT_GT(held.get_mass_flow_force(), 0.0);

    Solver free;
    free.init(parse_toml(channel_input("0.1", "n_steps = 200\ncfl = 0.5\n", false)));
    free.run();
    EXPECT_LT(double(free.integrate_conservatives()[1]) / volume, 0.1 - 1e-4);
}

TEST(SourceTest, MassFlowForcingBalancesTheWallShearOfPoiseuilleFlow) {
    // Steady laminar channel at bulk velocity 0.1: the force per unit volume
    // is the wall shear stress over the half height, 3 mu U_b / h^2 = 0.003,
    // and its work f U_b per unit volume heats the gas between adiabatic walls
    Solver solver;
    solver.init(parse_toml(channel_input("0.15 * (1 - (y - 1)^2)", "t_stop = 2.0\ncfl = 0.5\n")));
    const double E0 = double(solver.integrate_conservatives()[N_DIM + 1]);
    solver.run();
    const double f = solver.get_mass_flow_force();
    EXPECT_NEAR(f, 0.003, 0.003 * 0.01);
    // The bulk velocity is 0.1 to within the density changes (Mach 0.15), and
    // f settles from the start-up transient
    const double dE = double(solver.integrate_conservatives()[N_DIM + 1]) - E0;
    EXPECT_NEAR(dE, f * 0.1 * 1.0 * 2.0, f * 0.1 * 2.0 * precision_tol<double>(0.01, 0.1));  // single: rho E rounds at 1e-7 of 1.8
}

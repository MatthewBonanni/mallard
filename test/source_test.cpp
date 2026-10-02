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
    EXPECT_NEAR(after[0] - before[0], 0.3 * 0.5, 1e-12);
}

TEST(SourceTest, TimeDependentEnergySourceIsIntegratedInTime) {
    // d(total energy)/dt = sin(t) over a unit area: Delta E = 1 - cos(t)
    Solver solver;
    solver.init(parse_toml(box_input("rhoE = \"sin(t)\"\ntime_dependent = true\n", REST, "t_stop = 1.0\ncfl = 0.5\n")));
    const auto before = solver.integrate_conservatives();
    solver.run();
    const auto after = solver.integrate_conservatives();
    EXPECT_NEAR(after[3] - before[3], 1.0 - std::cos(1.0), 1e-7);
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
    auto spurious_velocity = [](uint32_t n) {
        std::string input = box_input("gravity = [0.0, -1.0]\n",
                                      "type = \"analytical\"\nrho = \"exp(-y)\"\nu = [\"0.0\", \"0.0\"]\np = \"exp(-y)\"\n",
                                      "t_stop = 0.5\ncfl = 0.5\n", "MUSCL");
        const std::string from = "Nx = 16\nNy = 16";
        input.replace(input.find(from), from.size(), "Nx = " + std::to_string(n) + "\nNy = " + std::to_string(n));
        Solver solver;
        solver.init(parse_toml(input));
        solver.run();
        solver.update_primitives();
        solver.copy_device_to_host();
        double v_max = 0.0;
        for (uint32_t i = 0; i < solver.get_mesh()->n_cells; i++) {
            v_max = std::max(v_max, std::hypot(solver.h_primitives(i, 0), solver.h_primitives(i, 1)));
        }
        return v_max;
    };
    const double v16 = spurious_velocity(16), v32 = spurious_velocity(32);
    // Hydrostatic wall ghosts: second-order convergence and small magnitude
    // (mirrored wall pressures gave 0.06 and 0.03)
    EXPECT_LT(v32, 0.35 * v16);
    EXPECT_LT(v32, 5e-4);
}

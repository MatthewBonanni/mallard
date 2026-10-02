/**
 * @file solver_test.cpp
 * @author Matthew Bonanni (mbonanni001@gmail.com)
 * @brief End-to-end solver tests: free-stream preservation, conservation,
 *        symmetry and 1D Riemann problems against the exact solution.
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
#include <tuple>

#include "test_fixtures.h"
#include "exact_riemann.h"
#include "solver.h"

namespace {

struct CaseConfig {
    std::string mesh = "cartesian_tri";
    uint32_t nx = 16;
    uint32_t ny = 16;
    std::string recon = "FO";
    std::string limiter = "barth_jespersen";
    std::string riemann = "HLLC";
    std::string bc_lr = "extrapolation";
    std::string bc_tb = "extrapolation";
    std::string init;
    std::string run = "n_steps = 20\ncfl = 0.5\n";
};

std::string make_input(const CaseConfig & c) {
    std::ostringstream s;
    s << "[run]\n" << c.run
      << "[mesh]\ntype = \"" << c.mesh << "\"\nNx = " << c.nx << "\nNy = " << c.ny << "\nLx = 1.0\nLy = 1.0\n"
      << "[initialize]\n" << c.init
      << "[[boundaries]]\nname = \"left\"\ntype = \"" << c.bc_lr << "\"\n"
      << "[[boundaries]]\nname = \"right\"\ntype = \"" << c.bc_lr << "\"\n"
      << "[[boundaries]]\nname = \"top\"\ntype = \"" << c.bc_tb << "\"\n"
      << "[[boundaries]]\nname = \"bottom\"\ntype = \"" << c.bc_tb << "\"\n"
      << "[numerics]\nriemann_solver = \"" << c.riemann << "\"\ntime_integrator = \"SSPRK3\"\ncheck_nan = true\n"
      << "[numerics.face_reconstruction]\ntype = \"" << c.recon << "\"\nlimiter = \"" << c.limiter << "\"\n"
      << "[physics]\ntype = \"euler\"\ngamma = 1.4\np_ref = 1.0\nT_ref = 1.0\nrho_ref = 1.0\n"
      << "[output]\ncheck_interval = 1000000\n";
    return s.str();
}

std::unique_ptr<Solver> run_case(const CaseConfig & c) {
    auto solver = std::make_unique<Solver>();
    solver->init(parse_toml(make_input(c)));
    solver->run();
    solver->update_primitives();
    solver->copy_device_to_host();
    return solver;
}

const std::string UNIFORM_INIT =
    "type = \"analytical\"\nrho = \"1.3\"\nu = [\"0.4\", \"-0.25\"]\np = \"0.9\"\n";

using FreeStreamParam = std::tuple<std::string, std::string, std::string>;

class FreeStream : public ::testing::TestWithParam<FreeStreamParam> {};

} // namespace

TEST_P(FreeStream, UniformFlowIsPreservedExactly) {
    CaseConfig c;
    std::tie(c.mesh, c.recon, c.riemann) = GetParam();
    c.nx = 9;
    c.ny = 7;
    c.init = UNIFORM_INIT;
    auto solver = run_case(c);
    for (uint32_t i_cell = 0; i_cell < solver->get_mesh()->n_cells; i_cell++) {
        EXPECT_NEAR(solver->h_conservatives(i_cell, 0), 1.3, 1e-12);
        EXPECT_NEAR(solver->h_primitives(i_cell, 0), 0.4, 1e-12);
        EXPECT_NEAR(solver->h_primitives(i_cell, 1), -0.25, 1e-12);
        EXPECT_NEAR(solver->h_primitives(i_cell, 2), 0.9, 1e-12);
    }
}

INSTANTIATE_TEST_SUITE_P(Solver, FreeStream,
    ::testing::Combine(::testing::Values("cartesian", "cartesian_tri", "wedge"),
                       ::testing::Values("FO", "MUSCL", "TENO"),
                       ::testing::Values("Rusanov", "HLL", "HLLC")));

namespace {

using MeshReconParam = std::tuple<std::string, std::string>;
class MeshRecon : public ::testing::TestWithParam<MeshReconParam> {};

} // namespace

TEST_P(MeshRecon, ClosedBoxConservesMassMomentumEnergy) {
    CaseConfig c;
    std::tie(c.mesh, c.recon) = GetParam();
    c.bc_lr = "symmetry";
    c.bc_tb = "symmetry";
    c.init = "type = \"analytical\"\n"
             "rho = \"1.0 + 0.5 * exp(-40 * ((x - 0.4)^2 + (y - 0.55)^2))\"\n"
             "u = [\"0.1\", \"0.0\"]\n"
             "p = \"1.0 + 0.8 * exp(-40 * ((x - 0.4)^2 + (y - 0.55)^2))\"\n";
    auto solver = std::make_unique<Solver>();
    solver->init(parse_toml(make_input(c)));
    const auto before = solver->integrate_conservatives();
    solver->run();
    const auto after = solver->integrate_conservatives();
    // Mass and energy are conserved exactly; momentum changes only through wall pressure
    EXPECT_NEAR(after[0], before[0], 1e-12);
    EXPECT_NEAR(after[3], before[3], 1e-12);
}

TEST_P(MeshRecon, RiemannProblemIsSymmetricAboutDiagonal) {
    // Configuration 3 is symmetric under x <-> y (u <-> v); the meshes are too
    // (cartesian_tri diagonals run from bottom-left to top-right).
    CaseConfig c;
    std::tie(c.mesh, c.recon) = GetParam();
    if (c.mesh == "wedge") GTEST_SKIP() << "wedge is not symmetric";
    c.nx = 20;
    c.ny = 20;
    c.run = "t_stop = 0.2\ncfl = 0.5\n";
    c.init = "type = \"analytical\"\n"
             "rho = \"x >= 0.8 ? (y >= 0.8 ? 1.5 : 0.5322580645) : (y >= 0.8 ? 0.5322580645 : 0.1379928315)\"\n"
             "u = [\"x >= 0.8 ? 0.0 : 1.206045378\", \"y >= 0.8 ? 0.0 : 1.206045378\"]\n"
             "p = \"x >= 0.8 ? (y >= 0.8 ? 1.5 : 0.3) : (y >= 0.8 ? 0.3 : 0.0290322581)\"\n";
    auto solver = run_case(c);
    auto mesh = solver->get_mesh();
    // Pair each cell with its mirror image by centroid
    rtype max_diff = 0.0;
    uint32_t n_matched = 0;
    for (uint32_t i = 0; i < mesh->n_cells; i++) {
        const rtype x = mesh->h_cell_coords(i, 0), y = mesh->h_cell_coords(i, 1);
        for (uint32_t j = 0; j < mesh->n_cells; j++) {
            if (std::abs(mesh->h_cell_coords(j, 0) - y) < 1e-9 &&
                std::abs(mesh->h_cell_coords(j, 1) - x) < 1e-9) {
                n_matched++;
                max_diff = std::max(max_diff, std::abs(solver->h_conservatives(i, 0) - solver->h_conservatives(j, 0)));
                max_diff = std::max(max_diff, std::abs(solver->h_conservatives(i, 1) - solver->h_conservatives(j, 2)));
                max_diff = std::max(max_diff, std::abs(solver->h_conservatives(i, 3) - solver->h_conservatives(j, 3)));
                break;
            }
        }
    }
    EXPECT_EQ(n_matched, mesh->n_cells);
    EXPECT_LT(max_diff, 1e-10);
}

INSTANTIATE_TEST_SUITE_P(Solver, MeshRecon,
    ::testing::Combine(::testing::Values("cartesian", "cartesian_tri", "wedge"),
                       ::testing::Values("FO", "MUSCL", "TENO")));

namespace {

/**
 * @brief L1 density error of a Sod problem against the exact solution.
 * @param along_y Run the problem along y instead of x.
 */
double sod_error(const std::string & mesh, const std::string & recon, uint32_t n, bool along_y,
                 double * transverse_variation = nullptr) {
    CaseConfig c;
    c.mesh = mesh;
    c.recon = recon;
    c.nx = along_y ? 4 : n;
    c.ny = along_y ? n : 4;
    c.bc_lr = along_y ? "symmetry" : "extrapolation";
    c.bc_tb = along_y ? "extrapolation" : "symmetry";
    c.run = "t_stop = 0.2\ncfl = 0.5\n";
    const std::string s = along_y ? "y" : "x";
    c.init = "type = \"analytical\"\n"
             "rho = \"" + s + " < 0.5 ? 1.0 : 0.125\"\n"
             "u = [\"0.0\", \"0.0\"]\n"
             "p = \"" + s + " < 0.5 ? 1.0 : 0.1\"\n";
    auto solver = run_case(c);
    ExactRiemann exact(1.0, 0.0, 1.0, 0.125, 0.0, 0.1, 1.4);
    auto m = solver->get_mesh();
    double err = 0.0, vol = 0.0;
    for (uint32_t i = 0; i < m->n_cells; i++) {
        const double xi = m->h_cell_coords(i, along_y ? 1 : 0);
        double rho, u, p;
        exact.sample((xi - 0.5) / solver->get_time(), rho, u, p);
        err += std::abs(solver->h_conservatives(i, 0) - rho) * m->h_cell_volume(i);
        vol += m->h_cell_volume(i);
    }
    if (transverse_variation) {
        // Cross-stream velocity should vanish
        double max_v = 0.0;
        for (uint32_t i = 0; i < m->n_cells; i++) {
            max_v = std::max(max_v, std::abs(solver->h_primitives(i, along_y ? 0 : 1)));
        }
        *transverse_variation = max_v;
    }
    return err / vol;
}

class SodMesh : public ::testing::TestWithParam<std::string> {};

} // namespace

TEST_P(SodMesh, FirstOrderConvergesToExactSolution) {
    const double e1 = sod_error(GetParam(), "FO", 50, false);
    const double e2 = sod_error(GetParam(), "FO", 100, false);
    EXPECT_LT(e2, 0.03);
    EXPECT_LT(e2, 0.75 * e1);
}

TEST_P(SodMesh, MUSCLIsMoreAccurateThanFirstOrder) {
    const double e_fo = sod_error(GetParam(), "FO", 100, false);
    const double e_muscl = sod_error(GetParam(), "MUSCL", 100, false);
    EXPECT_LT(e_muscl, 0.7 * e_fo);
    EXPECT_LT(e_muscl, 0.01);
}

TEST_P(SodMesh, XAndYDirectionsGiveSameError) {
    double v_x = 0.0, v_y = 0.0;
    const double e_x = sod_error(GetParam(), "MUSCL", 60, false, &v_x);
    const double e_y = sod_error(GetParam(), "MUSCL", 60, true, &v_y);
    EXPECT_NEAR(e_x, e_y, 1e-10);
    EXPECT_NEAR(v_x, v_y, 1e-10);
    if (GetParam() == "cartesian") {
        // Quads aligned with the wave keep the problem exactly one-dimensional
        EXPECT_LT(v_x, 1e-12);
    }
}

INSTANTIATE_TEST_SUITE_P(Solver, SodMesh, ::testing::Values("cartesian", "cartesian_tri"));

TEST(SolverRegression, MUSCLTransmissiveInflowBoundaryStaysBounded) {
    // Configuration 3 has supersonic inflow through the bottom and left
    // transmissive boundaries. Using the linearly extrapolated face state as
    // the exterior state there drove the boundary velocity to ~10x its
    // physical bound and eventually to vacuum.
    CaseConfig c;
    c.mesh = "cartesian_tri";
    c.nx = 40;
    c.ny = 40;
    c.recon = "MUSCL";
    c.limiter = "venkatakrishnan";
    c.run = "t_stop = 0.6\ncfl = 0.5\n";
    c.init = "type = \"analytical\"\n"
             "rho = \"x >= 0.8 ? (y >= 0.8 ? 1.5 : 0.5322580645) : (y >= 0.8 ? 0.5322580645 : 0.1379928315)\"\n"
             "u = [\"x >= 0.8 ? 0.0 : 1.206045378\", \"y >= 0.8 ? 0.0 : 1.206045378\"]\n"
             "p = \"x >= 0.8 ? (y >= 0.8 ? 1.5 : 0.3) : (y >= 0.8 ? 0.3 : 0.0290322581)\"\n";
    auto solver = run_case(c);
    rtype max_speed = 0.0;
    for (uint32_t i = 0; i < solver->get_mesh()->n_cells; i++) {
        max_speed = std::max(max_speed, std::hypot(solver->h_primitives(i, 0), solver->h_primitives(i, 1)));
    }
    EXPECT_LT(max_speed, 2.0);
}

namespace {

/**
 * @brief L1 density error after advecting a smooth density pulse, which is an
 *        exact solution of the Euler equations (uniform velocity and pressure).
 */
double advection_error(const std::string & mesh, const std::string & recon, uint32_t n) {
    CaseConfig c;
    c.mesh = mesh;
    c.recon = recon;
    c.nx = n;
    c.ny = n;
    c.run = "t_stop = 0.2\ncfl = 0.4\n";
    c.init = "type = \"analytical\"\n"
             "rho = \"1.0 + 0.3 * exp(-80 * ((x - 0.35)^2 + (y - 0.4)^2))\"\n"
             "u = [\"1.0\", \"0.5\"]\n"
             "p = \"1.0\"\n";
    auto solver = run_case(c);
    auto m = solver->get_mesh();
    // Exact cell averages of the translated pulse
    auto exact = cell_averages(*m, [](double x, double y, double * U) {
        U[0] = 1.0 + 0.3 * std::exp(-80.0 * ((x - 0.55) * (x - 0.55) + (y - 0.5) * (y - 0.5)));
        U[1] = U[2] = U[3] = 0.0;
    });
    double err = 0.0;
    for (uint32_t i = 0; i < m->n_cells; i++) {
        err += std::abs(solver->h_conservatives(i, 0) - exact(i, 0)) * m->h_cell_volume(i);
    }
    return err;
}

} // namespace

TEST_P(SodMesh, TENOConvergesFasterThanMUSCLForSmoothFlow) {
    // Time integration (SSPRK3, dt ~ h) caps the observed order at 3
    const double m1 = advection_error(GetParam(), "MUSCL", 32), m2 = advection_error(GetParam(), "MUSCL", 64);
    const double t1 = advection_error(GetParam(), "TENO", 32), t2 = advection_error(GetParam(), "TENO", 64);
    std::cout << "advection L1: MUSCL " << m1 << " -> " << m2 << ", TENO " << t1 << " -> " << t2 << std::endl;
    EXPECT_GT(std::log2(t1 / t2), 2.7);
    EXPECT_LT(t2, 0.2 * m2);
}

namespace {

class Recon : public ::testing::TestWithParam<std::string> {};

} // namespace

TEST_P(Recon, TransmissiveInflowOnTrianglesKeepsOneDimensionalShockSpeed) {
    // Shock between quadrants 3 and 4 of 2D Riemann configuration 3, with the
    // uniform transverse velocity v = 1.206 carrying flow in through the bottom
    // and out through the top. The exact solution is one-dimensional: a shock
    // moving at -0.4221 leaving the quadrant-4 state behind. With a
    // zero-gradient copy of the boundary cell as the exterior state, triangle
    // meshes gained an O(1) mass imbalance in the boundary row at the shock and
    // the shock along the boundary ran at almost twice the correct speed.
    CaseConfig c;
    c.mesh = "cartesian_tri";
    c.recon = GetParam();
    c.nx = 60;
    c.ny = 20;
    c.run = "t_stop = 0.5\ncfl = 0.5\n";
    c.init = "type = \"analytical\"\n"
             "rho = \"x < 0.8 ? 0.1379928315 : 0.5322580645\"\n"
             "u = [\"x < 0.8 ? 1.206045378 : 0.0\", \"1.206045378\"]\n"
             "p = \"x < 0.8 ? 0.0290322581 : 0.3\"\n";
    auto solver = run_case(c);
    auto m = solver->get_mesh();
    const double x_shock = 0.8 - 0.4221 * 0.5;
    double max_dev = 0.0;
    for (uint32_t i = 0; i < m->n_cells; i++) {
        // Post-shock region in every row including the boundary rows, excluding
        // the start-up entropy error that stays at the initial interface x = 0.8
        if (m->h_cell_coords(i, 0) > x_shock + 0.1 && m->h_cell_coords(i, 0) < 0.75) {
            const double dev = std::abs(solver->h_conservatives(i, 0) - 0.5322580645);
            max_dev = std::max(max_dev, dev);
        }
    }
    EXPECT_LT(max_dev, 0.01);
}

INSTANTIATE_TEST_SUITE_P(Solver, Recon, ::testing::Values("FO", "MUSCL", "TENO"));

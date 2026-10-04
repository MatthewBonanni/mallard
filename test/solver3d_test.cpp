/**
 * @file solver3d_test.cpp
 * @author Matthew Bonanni (mbonanni001@gmail.com)
 * @brief Validation of the full 3D solver: free stream, conservation, Sod
 *        shock tube, Couette flow and symmetry.
 * @version 0.1
 * @date 2026-10-02
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 *
 */

#include <gtest/gtest.h>
#include <Kokkos_Core.hpp>

#include <cmath>
#include <filesystem>
#include <map>
#include <memory>
#include <sstream>
#include <string>
#include <tuple>
#include <vector>

#include "test_fixtures.h"
#include "exact_riemann.h"
#include "solver.h"

namespace {

const char * ZONES[6] = {"left", "right", "bottom", "top", "back", "front"};

struct Case3D {
    std::string mesh = "cartesian";
    uint32_t n[3] = {3, 3, 3};
    double L[3] = {1.0, 1.0, 1.0};
    std::string recon = "FO";
    std::string recon_options;
    std::string riemann = "HLLC";
    std::string bc[6] = {"type = \"extrapolation\"\n", "type = \"extrapolation\"\n",
                         "type = \"extrapolation\"\n", "type = \"extrapolation\"\n",
                         "type = \"extrapolation\"\n", "type = \"extrapolation\"\n"};
    std::string init;
    std::string run = "n_steps = 10\ncfl = 0.5\n";
    std::string physics = "type = \"euler\"\ngamma = 1.4\np_ref = 1.0\nT_ref = 1.0\nrho_ref = 1.0\n";
    std::string extra;

    void set_all_bcs(const std::string & b) {
        for (auto & s : bc) s = b;
    }
};

std::unique_ptr<Solver> init_case(const Case3D & c) {
    std::ostringstream s;
    s << "[run]\n" << c.run
      << "[mesh]\ntype = \"" << c.mesh << "\"\nNx = " << c.n[0] << "\nNy = " << c.n[1] << "\nNz = " << c.n[2]
      << "\nLx = " << c.L[0] << "\nLy = " << c.L[1] << "\nLz = " << c.L[2] << "\n"
      << "[initialize]\n" << c.init;
    for (int k = 0; k < 6; k++) s << "[[boundaries]]\nname = \"" << ZONES[k] << "\"\n" << c.bc[k];
    s << "[numerics]\nriemann_solver = \"" << c.riemann << "\"\ntime_integrator = \"SSPRK3\"\ncheck_nan = true\n"
      << "[numerics.face_reconstruction]\ntype = \"" << c.recon << "\"\n" << c.recon_options
      << "[physics]\n" << c.physics
      << "[output]\ncheck_interval = 1000000\n" << c.extra;
    auto solver = std::make_unique<Solver>();
    solver->init(parse_toml(s.str()));
    return solver;
}

std::unique_ptr<Solver> run_case(const Case3D & c) {
    auto solver = init_case(c);
    solver->run();
    solver->update_primitives();
    solver->copy_device_to_host();
    return solver;
}

const std::vector<std::string> MESHES = {"cartesian", "cartesian_tet", "cartesian_prism", "cartesian_pyramid", "cartesian_mixed"};

using FreeStreamParam = std::tuple<std::string, std::string, std::string>;
class FreeStream3D : public ::testing::TestWithParam<FreeStreamParam> {};

} // namespace

TEST_P(FreeStream3D, UniformFlowIsPreservedExactly) {
    // Transmissive in x, characteristic far field on the other boundaries
    Case3D c;
    std::tie(c.mesh, c.recon, c.riemann) = GetParam();
    c.n[0] = 4;
    const std::string farfield = "type = \"farfield\"\nu = [0.4, -0.25, 0.3]\np = 0.9\nT = 0.6923076923076923\n";
    for (int k = 2; k < 6; k++) c.bc[k] = farfield;
    c.init = "type = \"constant\"\nu = [0.4, -0.25, 0.3]\np = 0.9\nT = 0.6923076923076923\n";
    auto solver = run_case(c);
    for (uint32_t i = 0; i < solver->get_mesh()->n_cells; i++) {
        ASSERT_NEAR(solver->h_conservatives(i, 0), 1.3, roundoff(1e-12)) << "cell " << i;
        ASSERT_NEAR(solver->h_primitives(i, 0), 0.4, roundoff(1e-12)) << "cell " << i;
        ASSERT_NEAR(solver->h_primitives(i, 1), -0.25, roundoff(1e-12)) << "cell " << i;
        ASSERT_NEAR(solver->h_primitives(i, 2), 0.3, roundoff(1e-12)) << "cell " << i;
        ASSERT_NEAR(solver->h_primitives(i, 3), 0.9, roundoff(1e-12)) << "cell " << i;
    }
}

INSTANTIATE_TEST_SUITE_P(Solver3D, FreeStream3D,
    ::testing::Combine(::testing::ValuesIn(MESHES),
                       ::testing::Values("FO", "MUSCL"),
                       ::testing::Values("HLLC", "Roe", "RHLL")));

namespace {

using MeshReconParam = std::tuple<std::string, std::string>;
class MeshRecon3D : public ::testing::TestWithParam<MeshReconParam> {};

} // namespace

TEST_P(MeshRecon3D, ClosedBoxConservesMassAndEnergy) {
    Case3D c;
    std::tie(c.mesh, c.recon) = GetParam();
    c.set_all_bcs("type = \"symmetry\"\n");
    const std::string blob = "exp(-30 * ((x - 0.4)^2 + (y - 0.55)^2 + (z - 0.45)^2))";
    c.init = "type = \"analytical\"\nrho = \"1.0 + 0.5 * " + blob + "\"\n"
             "u = [\"0.1\", \"-0.05\", \"0.08\"]\np = \"1.0 + 0.8 * " + blob + "\"\n";
    auto solver = init_case(c);
    const auto before = solver->integrate_conservatives();
    solver->run();
    const auto after = solver->integrate_conservatives();
    EXPECT_NEAR(after[0], before[0], roundoff(1e-12));
    EXPECT_NEAR(after[4], before[4], roundoff(1e-12));
    EXPECT_GT(std::abs(after[1] - before[1]), 1e-6) << "the blob should have hit the walls";
}

INSTANTIATE_TEST_SUITE_P(Solver3D, MeshRecon3D,
    ::testing::Combine(::testing::ValuesIn(MESHES), ::testing::Values("FO", "MUSCL")));

TEST(Solver3DValidation, UnlimitedMUSCLKeepsAnAcousticPulseBoundedOnTetrahedra) {
    // Regression: with gradients fitted over face neighbors only, unlimited
    // MUSCL grew a mode at the walls of this box of tetrahedra until the
    // solution became non-finite at t = 1.9
    Case3D c;
    c.mesh = "cartesian_tet";
    c.n[0] = c.n[1] = c.n[2] = 6;
    c.recon = "MUSCL";
    c.recon_options = "limiter = \"none\"\n";
    c.set_all_bcs("type = \"symmetry\"\n");
    const std::string pulse = "exp(-50 * ((x - 0.5)^2 + (y - 0.5)^2 + (z - 0.5)^2))";
    c.init = "type = \"analytical\"\nrho = \"1.0 + 0.01 * " + pulse + "\"\n"
             "u = [\"0.0\", \"0.0\", \"0.0\"]\np = \"0.7142857142857143 * (1.0 + 0.014 * " + pulse + ")\"\n";
    c.run = "t_stop = 4.0\ncfl = 0.5\n";
    auto solver = run_case(c);
    double u_max = 0.0;
    for (uint32_t i = 0; i < solver->get_mesh()->n_cells; i++) {
        for (int d = 0; d < 3; d++) u_max = std::max(u_max, std::abs(double(solver->h_primitives(i, d))));
    }
    EXPECT_LT(u_max, 0.01);
}

namespace {

/**
 * @brief L1 density error of a Sod problem along axis `axis` against the exact
 *        solution, and the largest cross-stream velocity.
 */
double sod_error(const std::string & mesh, const std::string & recon, uint32_t n, int axis,
                 double * transverse = nullptr) {
    Case3D c;
    c.mesh = mesh;
    c.recon = recon;
    c.riemann = "HLLC";
    for (int d = 0; d < 3; d++) {
        c.n[d] = (d == axis) ? n : 2;
        c.L[d] = (d == axis) ? 1.0 : 2.0 / n;
    }
    c.set_all_bcs("type = \"symmetry\"\n");
    c.bc[2 * axis] = "type = \"extrapolation\"\n";
    c.bc[2 * axis + 1] = "type = \"extrapolation\"\n";
    c.run = "t_stop = 0.2\ncfl = 0.5\n";
    const std::string s = std::string(1, "xyz"[axis]);
    c.init = "type = \"analytical\"\nrho = \"" + s + " < 0.5 ? 1.0 : 0.125\"\n"
             "u = [\"0.0\", \"0.0\", \"0.0\"]\np = \"" + s + " < 0.5 ? 1.0 : 0.1\"\n";
    auto solver = run_case(c);
    ExactRiemann exact(1.0, 0.0, 1.0, 0.125, 0.0, 0.1, 1.4);
    auto m = solver->get_mesh();
    double err = 0.0, vol = 0.0, max_v = 0.0;
    for (uint32_t i = 0; i < m->n_cells; i++) {
        double rho, u, p;
        exact.sample((double(m->h_cell_coords(i, axis)) - 0.5) / double(solver->get_time()), rho, u, p);
        err += std::abs(double(solver->h_conservatives(i, 0)) - rho) * double(m->h_cell_volume(i));
        vol += double(m->h_cell_volume(i));
        for (int d = 0; d < 3; d++) {
            if (d != axis) max_v = std::max(max_v, std::abs(static_cast<double>(solver->h_primitives(i, d))));
        }
    }
    if (transverse) *transverse = max_v;
    return err / vol;
}

class Sod3D : public ::testing::TestWithParam<std::string> {};

} // namespace

TEST_P(Sod3D, FirstOrderConvergesToExactSolution) {
    const double e1 = sod_error(GetParam(), "FO", 50, 0);
    const double e2 = sod_error(GetParam(), "FO", 100, 0);
    EXPECT_LT(e2, 0.03);
    EXPECT_LT(e2, 0.75 * e1);
}

TEST_P(Sod3D, MUSCLIsMoreAccurateThanFirstOrder) {
    const double e_fo = sod_error(GetParam(), "FO", 100, 0);
    const double e_muscl = sod_error(GetParam(), "MUSCL", 100, 0);
    EXPECT_LT(e_muscl, 0.7 * e_fo);
    EXPECT_LT(e_muscl, 0.01);
}

TEST_P(Sod3D, XYAndZDirectionsGiveSameSolution) {
    double v[3];
    double e[3];
    for (int axis = 0; axis < 3; axis++) e[axis] = sod_error(GetParam(), "MUSCL", 60, axis, &v[axis]);
    EXPECT_NEAR(e[1], e[0], precision_tol<double>(1e-10, 1e-5));
    EXPECT_NEAR(e[2], e[0], precision_tol<double>(1e-10, 1e-5));
    EXPECT_NEAR(v[1], v[0], precision_tol<double>(1e-10, 1e-5));
    EXPECT_NEAR(v[2], v[0], precision_tol<double>(1e-10, 1e-5));
    if (GetParam() == "cartesian") {
        // Hexahedra aligned with the wave keep the problem exactly one-dimensional
        EXPECT_LT(v[0], roundoff(1e-12));
    }
}

INSTANTIATE_TEST_SUITE_P(Solver3D, Sod3D, ::testing::Values("cartesian", "cartesian_tet"));

TEST(Solver3DValidation, SphericalBlastIsSymmetricUnderAxisPermutation) {
    // The cube, the blast and both meshes are invariant under (x, y, z) -> (y, z, x)
    for (const char * mesh : {"cartesian", "cartesian_tet"}) {
        Case3D c;
        c.mesh = mesh;
        c.n[0] = c.n[1] = c.n[2] = 6;
        c.recon = "MUSCL";
        c.set_all_bcs("type = \"symmetry\"\n");
        c.run = "t_stop = 0.1\ncfl = 0.5\n";
        const std::string r2 = "((x - 0.5)^2 + (y - 0.5)^2 + (z - 0.5)^2)";
        c.init = "type = \"analytical\"\nrho = \"" + r2 + " < 0.04 ? 2.0 : 1.0\"\n"
                 "u = [\"0.0\", \"0.0\", \"0.0\"]\np = \"" + r2 + " < 0.04 ? 5.0 : 1.0\"\n";
        auto solver = run_case(c);
        auto m = solver->get_mesh();
        double max_diff = 0.0, max_u = 0.0;
        uint32_t matched = 0;
        for (uint32_t i = 0; i < m->n_cells; i++) {
            const double x = double(m->h_cell_coords(i, 0)), y = double(m->h_cell_coords(i, 1)), z = double(m->h_cell_coords(i, 2));
            for (uint32_t j = 0; j < m->n_cells; j++) {
                const double tol = precision_tol<double>(1e-9, 1e-5);
                if (std::abs(double(m->h_cell_coords(j, 0)) - y) < tol && std::abs(double(m->h_cell_coords(j, 1)) - z) < tol &&
                    std::abs(double(m->h_cell_coords(j, 2)) - x) < tol) {
                    matched++;
                    max_diff = std::max(max_diff, std::abs(double(solver->h_conservatives(i, 0)) - double(solver->h_conservatives(j, 0))));
                    max_diff = std::max(max_diff, std::abs(double(solver->h_conservatives(i, 4)) - double(solver->h_conservatives(j, 4))));
                    // Momentum (u, v, w) at (x, y, z) is (w, u, v) at (y, z, x)
                    max_diff = std::max(max_diff, std::abs(double(solver->h_conservatives(i, 1)) - double(solver->h_conservatives(j, 3))));
                    max_diff = std::max(max_diff, std::abs(double(solver->h_conservatives(i, 2)) - double(solver->h_conservatives(j, 1))));
                    max_diff = std::max(max_diff, std::abs(double(solver->h_conservatives(i, 3)) - double(solver->h_conservatives(j, 2))));
                    break;
                }
            }
            max_u = std::max(max_u, std::abs(static_cast<double>(solver->h_primitives(i, 0))));
        }
        EXPECT_EQ(matched, m->n_cells) << mesh;
        EXPECT_LT(max_diff, precision_tol<double>(1e-10, 1e-5)) << mesh;
        EXPECT_GT(max_u, 0.1) << mesh;
    }
}

TEST(Solver3DValidation, AnalyticalInitializationIntegratesPolynomialsExactly) {
    // Degree-5 rule: the domain integral of a cubic is exact on every cell type,
    // and on hexahedra so is each cell average
    for (const std::string & mesh : MESHES) {
        Case3D c;
        c.mesh = mesh;
        c.n[0] = 2;
        c.n[1] = 3;
        c.n[2] = 2;
        c.set_all_bcs("type = \"symmetry\"\n");
        c.init = "type = \"analytical\"\nrho = \"1 + x^2 * y + z^3\"\nu = [\"0\", \"0\", \"0\"]\np = \"1\"\n";
        auto solver = init_case(c);
        solver->copy_device_to_host();
        auto m = solver->get_mesh();
        double total = 0.0;
        for (uint32_t i = 0; i < m->n_cells; i++) total += double(solver->h_conservatives(i, 0)) * double(m->h_cell_volume(i));
        // int_0^1 int_0^1 int_0^1 (1 + x^2 y + z^3) = 1 + 1/6 + 1/4
        EXPECT_NEAR(total, 1.0 + 1.0 / 6.0 + 0.25, roundoff(1e-12)) << mesh;
        if (mesh != "cartesian") continue;
        for (uint32_t i = 0; i < m->n_cells; i++) {
            double lo[3] = {1e30, 1e30, 1e30}, hi[3] = {-1e30, -1e30, -1e30};
            for (uint32_t k = 0; k < m->h_n_nodes_of_cell(i); k++) {
                for (int d = 0; d < 3; d++) {
                    lo[d] = std::min(lo[d], static_cast<double>(m->h_node_coords(m->h_node_of_cell(i, k), d)));
                    hi[d] = std::max(hi[d], static_cast<double>(m->h_node_coords(m->h_node_of_cell(i, k), d)));
                }
            }
            auto mean_pow = [&](int d, int p) {
                return (std::pow(hi[d], p + 1) - std::pow(lo[d], p + 1)) / ((p + 1) * (hi[d] - lo[d]));
            };
            const double exact = 1.0 + mean_pow(0, 2) * mean_pow(1, 1) + mean_pow(2, 3);
            EXPECT_NEAR(solver->h_conservatives(i, 0), exact, roundoff(1e-12)) << "cell " << i;
        }
    }
}

namespace {

/**
 * @brief Steady Couette flow between isothermal plates at z = 0 (at rest) and
 *        z = 1 (moving at (0.1, 0.05, 0)), transmissive in x and y.
 */
std::unique_ptr<Solver> couette(const std::string & mesh) {
    Case3D c;
    c.mesh = mesh;
    c.n[0] = 2;
    c.n[1] = 2;
    c.n[2] = (mesh == "cartesian") ? 16 : 8;
    c.recon = "MUSCL";
    c.bc[4] = "type = \"wall_isothermal\"\nT = 1.0\n";
    c.bc[5] = "type = \"wall_isothermal\"\nT = 1.0\nu = [0.1, 0.05, 0.0]\n";
    c.init = "type = \"analytical\"\nrho = \"1.0\"\nu = [\"0.0\", \"0.0\", \"0.0\"]\np = \"1.0\"\n";
    c.run = "t_stop = 6.0\ncfl = 0.8\n";
    c.physics = "type = \"navier_stokes\"\ngamma = 1.4\np_ref = 1.0\nT_ref = 1.0\nrho_ref = 1.0\nmu = 0.2\nPr = 0.72\n";
    return run_case(c);
}

class Couette3D : public ::testing::TestWithParam<std::string> {};

} // namespace

TEST_P(Couette3D, LinearVelocityAndWallForces) {
    // After 6 diffusion times H^2 / nu = 5 the flow is steady with u = U z
    auto solver = couette(GetParam());
    auto m = solver->get_mesh();
    double max_err = 0.0;
    for (uint32_t i = 0; i < m->n_cells; i++) {
        const double z = double(m->h_cell_coords(i, 2));
        max_err = std::max(max_err, std::abs(double(solver->h_primitives(i, 0)) - 0.1 * z));
        max_err = std::max(max_err, std::abs(double(solver->h_primitives(i, 1)) - 0.05 * z));
        EXPECT_LT(std::abs(solver->h_primitives(i, 2)), precision_tol<double>(1e-6, 1e-4));
    }
    EXPECT_LT(max_err, 1e-3 * 0.1);
    // Shear mu U / H on the unit-area plates opposes the relative motion; p = 1 pushes outward
    const auto front = solver->calc_force(m->get_face_zone("front")->faces);
    const auto back = solver->calc_force(m->get_face_zone("back")->faces);
    EXPECT_NEAR(front[2], 1.0, 1e-3);
    EXPECT_NEAR(back[2], -1.0, 1e-3);
    EXPECT_NEAR(front[0], 0.0, 1e-12);
    EXPECT_NEAR(front[1], 0.0, 1e-12);
    EXPECT_NEAR(front[3], -0.02, 2e-4);
    EXPECT_NEAR(front[4], -0.01, 1e-4);
    EXPECT_NEAR(back[3], 0.02, 2e-4);
    EXPECT_NEAR(back[4], 0.01, 1e-4);
    EXPECT_NEAR(front[5], 0.0, 1e-4);
}

TEST_P(Couette3D, HeatFluxWallSetsTemperatureGradient) {
    // q into the fluid through the back plate, isothermal front plate: dT/dz = -q / kappa
    Case3D c;
    c.mesh = GetParam();
    c.n[0] = 2;
    c.n[1] = 2;
    c.n[2] = 8;
    c.recon = "MUSCL";
    c.bc[4] = "type = \"wall_heat_flux\"\nq = 0.2\n";
    c.bc[5] = "type = \"wall_isothermal\"\nT = 1.0\n";
    c.init = "type = \"analytical\"\nrho = \"1.0\"\nu = [\"0.0\", \"0.0\", \"0.0\"]\np = \"1.0\"\n";
    c.run = "t_stop = 12.0\ncfl = 0.8\n";
    c.physics = "type = \"navier_stokes\"\ngamma = 1.4\np_ref = 1.0\nT_ref = 1.0\nrho_ref = 1.0\nmu = 0.2\nPr = 0.72\n";
    auto solver = run_case(c);
    const double kappa = 0.2 * 3.5 / 0.72;
    auto m = solver->get_mesh();
    for (uint32_t i = 0; i < m->n_cells; i++) {
        const double z = double(m->h_cell_coords(i, 2));
        EXPECT_NEAR(solver->h_primitives(i, 4), 1.0 + 0.2 * (1.0 - z) / kappa, 5e-4);
    }
}

INSTANTIATE_TEST_SUITE_P(Solver3D, Couette3D, ::testing::Values("cartesian", "cartesian_tet"));

TEST(Solver3DValidation, FlowStatisticsOfALinearVelocityField) {
    // u = (x + 2y, 3z, 5x): div u = 1 and omega = (-3, -5, -2), so the
    // integrals over the unit cube are exact for least-squares gradients when
    // the boundary states continue the field
    Case3D c;
    c.mesh = "cartesian_tet";
    c.n[0] = c.n[1] = c.n[2] = 4;
    const std::string u = "u = [\"x + 2 * y\", \"3 * z\", \"5 * x\"]\n";
    c.set_all_bcs("type = \"dirichlet\"\nrho = \"1.0\"\n" + u + "p = \"2.0\"\n");
    c.init = "type = \"analytical\"\nrho = \"1.0\"\n" + u + "p = \"2.0\"\n";
    c.extra = "[integrals]\nfile = \"" +
              (std::filesystem::temp_directory_path() / "mallard_integrals.csv").string() + "\"\n";
    auto solver = init_case(c);
    const auto s = solver->integrate_flow_statistics();
    EXPECT_NEAR(s[1], 0.5 * (9.0 + 25.0 + 4.0), roundoff(1e-10));  // Enstrophy
    EXPECT_NEAR(s[2], 1.0, roundoff(1e-10));                       // (div u)^2
    // Pressure from cell averages carries the same O(h^2) kinetic energy error
    EXPECT_NEAR(s[3], 2.0, 0.05);  // p div u
    // Kinetic energy from the cell averages of rho u misses their variance
    // within the cells: (8/3 + 3 + 25/3) / 2 = 7 minus O(h^2)
    EXPECT_NEAR(s[0], 7.0, 0.1);
}

TEST(Solver3DValidation, QAndVorticityOutputOfALinearVelocityField) {
    // u = (x + 2y - z, 3z - 2x, 5x + y): Q = -(d u_k / d x_i d u_i / d x_k) / 2
    // = 5.5 and omega = (-2, -6, -4) in every cell, exact for least squares
    Case3D c;
    c.mesh = "cartesian_tet";
    c.n[0] = c.n[1] = c.n[2] = 3;
    const std::string u = "u = [\"x + 2 * y - z\", \"3 * z - 2 * x\", \"5 * x + y\"]\n";
    c.set_all_bcs("type = \"dirichlet\"\nrho = \"1.0\"\n" + u + "p = \"2.0\"\n");
    c.init = "type = \"analytical\"\nrho = \"1.0\"\n" + u + "p = \"2.0\"\n";
    c.extra = "[[write_data]]\nprefix = \"" + (std::filesystem::temp_directory_path() / "mallard_vortex").string() +
              "\"\nformat = \"vtu\"\ninterval = 1000000\nvariables = [\"Q\", \"VORTICITY\"]\n";
    auto solver = init_case(c);
    solver->copy_device_to_host();
    const double expected[4] = {5.5, -2.0, -6.0, -4.0};
    for (uint32_t i = 0; i < solver->get_mesh()->n_owned(); i++) {
        for (int k = 0; k < 4; k++) {
            EXPECT_NEAR(solver->h_vortex_fields(i, k), expected[k], roundoff(1e-10)) << "cell " << i << ", field " << k;
        }
    }
}

TEST(Solver3DValidation, TENOEnstrophyOfTheTaylorGreenVortex) {
    // With TENO the integrals take gradients from the reconstruction
    // polynomials (order 5 by default): on 16^3 cells of the octant [0, pi]^3
    // the initial enstrophy of the Taylor-Green vortex, 3/8 per unit volume, is
    // within 0.5% (second-order least squares misses it by 5%)
    Case3D c;
    c.n[0] = c.n[1] = c.n[2] = 16;
    c.L[0] = c.L[1] = c.L[2] = M_PI;
    c.recon = "TENO";
    c.set_all_bcs("type = \"symmetry\"\n");
    c.init = "type = \"analytical\"\nrho = \"1.0\"\n"
             "u = [\"sin(x) * cos(y) * cos(z)\", \"-cos(x) * sin(y) * cos(z)\", \"0.0\"]\np = \"100.0\"\n";
    c.extra = "[integrals]\nfile = \"" +
              (std::filesystem::temp_directory_path() / "mallard_integrals_tgv.csv").string() + "\"\n";
    auto solver = init_case(c);
    const double enstrophy = double(solver->integrate_flow_statistics()[1]) / std::pow(M_PI, 3);
    EXPECT_NEAR(enstrophy, 0.375, 0.005 * 0.375);
}

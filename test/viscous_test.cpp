/**
 * @file viscous_test.cpp
 * @author Matthew Bonanni (mbonanni001@gmail.com)
 * @brief Navier-Stokes diffusive fluxes against exact solutions.
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

// Gas with R = 1, cp = 3.5, Pr = 0.72
struct ViscousCase {
    std::string mesh = "cartesian";
    uint32_t nx = 4;
    uint32_t ny = 16;
    double lx = 1.0;
    double mu = 0.02;
    std::string bottom = "type = \"wall_isothermal\"\nT = 1.0\n";
    std::string top = "type = \"wall_isothermal\"\nT = 1.0\n";
    std::string init = "type = \"analytical\"\nrho = \"1.0\"\nu = [\"0.0\", \"0.0\"]\np = \"1.0\"\n";
    std::string run = "t_stop = 5.0\ncfl = 0.8\n";
    std::string recon = "MUSCL";
    std::string source;
};

std::unique_ptr<Solver> run_viscous(const ViscousCase & c) {
    std::ostringstream s;
    s << "[run]\n" << c.run
      << "[mesh]\ntype = \"" << c.mesh << "\"\nNx = " << c.nx << "\nNy = " << c.ny << "\nLx = " << c.lx << "\nLy = 1.0\n"
      << "[initialize]\n" << c.init
      << "[[boundaries]]\nname = \"left\"\ntype = \"extrapolation\"\n"
      << "[[boundaries]]\nname = \"right\"\ntype = \"extrapolation\"\n"
      << "[[boundaries]]\nname = \"bottom\"\n" << c.bottom
      << "[[boundaries]]\nname = \"top\"\n" << c.top
      << "[numerics]\nriemann_solver = \"HLLC\"\ntime_integrator = \"SSPRK3\"\ncheck_nan = true\n"
      << "[numerics.face_reconstruction]\ntype = \"" << c.recon << "\"\nlimiter = \"venkatakrishnan\"\n"
      << "[physics]\ntype = \"navier_stokes\"\ngamma = 1.4\np_ref = 1.0\nT_ref = 1.0\nrho_ref = 1.0\n"
      << "mu = " << c.mu << "\nPr = 0.72\n"
      << "[output]\ncheck_interval = 1000000\n";
    if (!c.source.empty()) s << "[source]\n" << c.source;
    auto solver = std::make_unique<Solver>();
    solver->init(parse_toml(s.str()));
    solver->run();
    solver->update_primitives();
    solver->copy_device_to_host();
    return solver;
}

class ViscousMesh : public ::testing::TestWithParam<std::string> {};

} // namespace

TEST_P(ViscousMesh, CouetteFlowHasLinearVelocityProfile) {
    // Bottom wall at rest, top wall moving at U; isothermal walls. Low Mach
    // number, so the steady profile is u = U y with negligible heating.
    ViscousCase c;
    c.mesh = GetParam();
    c.mu = 0.2;
    c.top = "type = \"wall_isothermal\"\nT = 1.0\nu = [0.1, 0.0]\n";
    c.run = "t_stop = 15.0\ncfl = 0.8\n";
    auto solver = run_viscous(c);
    auto m = solver->get_mesh();
    double max_err = 0.0;
    for (uint32_t i = 0; i < m->n_cells; i++) {
        const double y = m->h_cell_coords(i, 1);
        max_err = std::max(max_err, std::abs(solver->h_primitives(i, 0) - 0.1 * y));
        EXPECT_LT(std::abs(solver->h_primitives(i, 1)), 1e-6);
    }
    EXPECT_LT(max_err, 1e-3 * 0.1);
}

TEST_P(ViscousMesh, StokesFirstProblemMatchesErfcProfile) {
    // Impulsively started wall: u = U erfc(y / (2 sqrt(nu t))), second order in space
    auto error = [&](uint32_t ny) {
        ViscousCase c;
        c.mesh = GetParam();
        c.ny = ny;
        c.mu = 0.01;
        c.bottom = "type = \"wall_isothermal\"\nT = 1.0\nu = [0.05, 0.0]\n";
        c.top = "type = \"symmetry\"\n";
        c.run = "t_stop = 2.0\ncfl = 0.8\n";
        auto solver = run_viscous(c);
        auto m = solver->get_mesh();
        double err = 0.0;
        for (uint32_t i = 0; i < m->n_cells; i++) {
            const double y = m->h_cell_coords(i, 1);
            const double exact = 0.05 * std::erfc(y / (2.0 * std::sqrt(0.01 * 2.0)));
            err = std::max(err, std::abs(solver->h_primitives(i, 0) - exact));
        }
        return err;
    };
    const double e1 = error(32), e2 = error(64);
    EXPECT_LT(e2, 0.01 * 0.05);
    EXPECT_GT(std::log2(e1 / e2), 1.8);
}

TEST_P(ViscousMesh, ForcedChannelFlowWithTransmissiveEndsConvergesAtSecondOrder) {
    // u = 0.05 (y - y^3) between walls at rest, driven by the body force
    // -mu u'' = 0.06 y, in a strip of width 1/4 with transmissive ends. On
    // triangles, ends that took their viscous flux from the boundary cell's own
    // gradient, rather than from the image face, converged at order 0.5.
    auto error = [&](uint32_t ny) {
        ViscousCase c;
        c.mesh = GetParam();
        c.nx = ny / 4;
        c.ny = ny;
        c.lx = 0.25;
        c.mu = 0.2;
        c.init = "type = \"analytical\"\nrho = \"1.0\"\nu = [\"0.05 * (y - y^3)\", \"0.0\"]\np = \"1.0\"\n";
        c.source = "rhou = [\"0.06 * y\", \"0.0\"]\n";
        c.run = "t_stop = 3.0\ncfl = 0.8\n";
        auto solver = run_viscous(c);
        auto m = solver->get_mesh();
        double err = 0.0;
        for (uint32_t i = 0; i < m->n_cells; i++) {
            const double y = m->h_cell_coords(i, 1);
            err = std::max(err, std::abs(solver->h_primitives(i, 0) - 0.05 * (y - y * y * y)));
        }
        return err;
    };
    const double e1 = error(8), e2 = error(16);
    EXPECT_GT(std::log2(e1 / e2), 1.8);
}

TEST_P(ViscousMesh, ConductionBetweenIsothermalWallsIsLinear) {
    ViscousCase c;
    c.mesh = GetParam();
    c.mu = 0.2;
    c.bottom = "type = \"wall_isothermal\"\nT = 1.2\n";
    c.top = "type = \"wall_isothermal\"\nT = 0.8\n";
    c.run = "t_stop = 20.0\ncfl = 0.8\n";
    auto solver = run_viscous(c);
    auto m = solver->get_mesh();
    for (uint32_t i = 0; i < m->n_cells; i++) {
        const double y = m->h_cell_coords(i, 1);
        EXPECT_NEAR(solver->h_primitives(i, 3), 1.2 - 0.4 * y, 2e-3);
    }
}

TEST_P(ViscousMesh, HeatFluxWallSetsTemperatureGradient) {
    // q into the fluid at the bottom, isothermal top: dT/dy = -q / kappa
    ViscousCase c;
    c.mesh = GetParam();
    c.mu = 0.2;
    const double kappa = 0.2 * 3.5 / 0.72;
    c.bottom = "type = \"wall_heat_flux\"\nq = 0.2\n";
    c.top = "type = \"wall_isothermal\"\nT = 1.0\n";
    c.run = "t_stop = 25.0\ncfl = 0.8\n";
    auto solver = run_viscous(c);
    auto m = solver->get_mesh();
    for (uint32_t i = 0; i < m->n_cells; i++) {
        const double y = m->h_cell_coords(i, 1);
        EXPECT_NEAR(solver->h_primitives(i, 3), 1.0 + 0.2 * (1.0 - y) / kappa, 5e-4);
    }
}

TEST_P(ViscousMesh, UniformFlowIsPreservedWithViscosity) {
    ViscousCase c;
    c.mesh = GetParam();
    c.nx = 8;
    c.ny = 8;
    c.init = "type = \"analytical\"\nrho = \"1.1\"\nu = [\"0.3\", \"-0.2\"]\np = \"0.9\"\n";
    c.bottom = "type = \"extrapolation\"\n";
    c.top = "type = \"extrapolation\"\n";
    c.run = "n_steps = 50\ncfl = 0.8\n";
    auto solver = run_viscous(c);
    for (uint32_t i = 0; i < solver->get_mesh()->n_cells; i++) {
        EXPECT_NEAR(solver->h_primitives(i, 0), 0.3, 1e-12);
        EXPECT_NEAR(solver->h_primitives(i, 1), -0.2, 1e-12);
        EXPECT_NEAR(solver->h_primitives(i, 2), 0.9, 1e-12);
    }
}

INSTANTIATE_TEST_SUITE_P(Viscous, ViscousMesh, ::testing::Values("cartesian", "cartesian_tri"));

TEST(ViscousForces, CouetteWallShearMatchesMuUOverH) {
    // Steady Couette flow: shear stress mu U / H = 0.2 * 0.1 / 1 on both walls.
    // The fluid drags the moving top wall back and the bottom wall forward;
    // pressure (p = 1) pushes both walls outward.
    ViscousCase c;
    c.mu = 0.2;
    c.top = "type = \"wall_isothermal\"\nT = 1.0\nu = [0.1, 0.0]\n";
    c.run = "t_stop = 15.0\ncfl = 0.8\n";
    auto solver = run_viscous(c);
    auto mesh = solver->get_mesh();
    const auto top = solver->calc_force(mesh->get_face_zone("top")->faces);
    const auto bottom = solver->calc_force(mesh->get_face_zone("bottom")->faces);
    EXPECT_NEAR(top[2], -0.02, 2e-4);
    EXPECT_NEAR(bottom[2], 0.02, 2e-4);
    EXPECT_NEAR(top[1], 1.0, 1e-3);
    EXPECT_NEAR(bottom[1], -1.0, 1e-3);
    EXPECT_NEAR(top[0], 0.0, 1e-12);
}

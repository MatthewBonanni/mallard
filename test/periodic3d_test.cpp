/**
 * @file periodic3d_test.cpp
 * @author Matthew Bonanni (mbonanni001@gmail.com)
 * @brief Periodic boundaries in 3D: seam geometry on every cell type,
 *        conservation, design order and translation invariance across seams.
 * @version 0.1
 * @date 2026-10-02
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 *
 */

#include <gtest/gtest.h>
#include <Kokkos_Core.hpp>

#include <cmath>
#include <functional>
#include <sstream>
#include <string>
#include <tuple>

#include "periodic_fixtures.h"
#include "solver.h"
#include "test_fixtures.h"

namespace {

constexpr double GAMMA = 1.4;

std::string periodic_list(const std::string & dirs) {
    std::string s = "[";
    for (size_t k = 0; k < dirs.size(); k++) s += std::string(k ? ", " : "") + "\"" + dirs[k] + "\"";
    return s + "]";
}

struct Case {
    std::string mesh = "cartesian";
    std::array<uint32_t, 3> n = {4, 4, 4};
    std::array<double, 3> L = {1.0, 1.0, 1.0};
    std::string dirs = "xyz";
    std::string recon = "FO";
    std::string extra_recon;
    std::string physics = "type = \"euler\"\n";
    std::string run = "n_steps = 10\ncfl = 0.5\n";
    std::string integrator = "SSPRK3";
    std::string boundaries;
};

std::string make_input(const Case & c) {
    std::ostringstream s;
    s << "[run]\n" << c.run << "[mesh]\ntype = \"" << c.mesh << "\"\n";
    for (int d = 0; d < 3; d++) s << "N" << "xyz"[d] << " = " << c.n[d] << "\nL" << "xyz"[d] << " = " << c.L[d] << "\n";
    s << "periodic = " << periodic_list(c.dirs) << "\n"
      << "[initialize]\ntype = \"constant\"\nu = [0.0, 0.0, 0.0]\np = 1.0\nT = 1.0\n"
      << c.boundaries
      << "[numerics]\nriemann_solver = \"HLLC\"\ntime_integrator = \"" << c.integrator << "\"\n"
      << "[numerics.face_reconstruction]\ntype = \"" << c.recon << "\"\n" << c.extra_recon
      << "[physics]\n" << c.physics << "gamma = 1.4\np_ref = 1.0\nT_ref = 1.0\nrho_ref = 1.0\n"
      << "[output]\ncheck_interval = 1000000\n";
    return s.str();
}

using PrimitiveField = std::function<void(double, double, double, double *)>;  // W = [rho, u, v, w, p]

std::unique_ptr<Solver> start(const Case & c, const PrimitiveField & W) {
    auto solver = std::make_unique<Solver>();
    solver->init(parse_toml(make_input(c)));
    set_cell_averages(*solver, [&](double x, double y, double z, double * U) {
        double w[5];
        W(x, y, z, w);
        U[0] = w[0];
        FOR_I_DIM U[1 + i] = w[0] * w[1 + i];
        U[4] = w[4] / (GAMMA - 1.0) + 0.5 * w[0] * (w[1] * w[1] + w[2] * w[2] + w[3] * w[3]);
    });
    return solver;
}

using MeshParam = std::tuple<std::string, std::string>;  // mesh type, periodic directions
class PeriodicMesh3D : public ::testing::TestWithParam<MeshParam> {};

} // namespace

TEST_P(PeriodicMesh3D, SeamFacesAndVertexNeighborsMeetAfterTheirShift) {
    const auto [type, dirs] = GetParam();
    Case c;
    c.mesh = type;
    c.n = {4, 3, 3};
    c.L = {2.0, 1.0, 1.5};
    c.dirs = dirs;
    Mesh mesh;
    mesh.init(parse_toml(make_input(c)));
    expect_shifts_join_cells(mesh);
}

INSTANTIATE_TEST_SUITE_P(Periodic, PeriodicMesh3D,
    ::testing::Values(MeshParam{"cartesian", "xyz"}, MeshParam{"cartesian_tet", "xyz"},
                      MeshParam{"cartesian_prism", "xz"}, MeshParam{"cartesian_pyramid", "xy"},
                      MeshParam{"cartesian_mixed", "xyz"}));

namespace {

using SchemeParam = std::tuple<std::string, std::string>;  // mesh type, reconstruction
class PeriodicBox3D : public ::testing::TestWithParam<SchemeParam> {};

} // namespace

TEST_P(PeriodicBox3D, PreservesUniformFlowAndConservesMassMomentumAndEnergy) {
    Case c;
    std::tie(c.mesh, c.recon) = GetParam();
    if (c.recon == "TENO") c.extra_recon = "order = 3\n";
    auto uniform = start(c, [](double, double, double, double * W) {
        W[0] = 1.3;
        W[1] = 0.4;
        W[2] = -0.25;
        W[3] = 0.1;
        W[4] = 0.9;
    });
    uniform->run();
    uniform->copy_device_to_host();
    const double U[4] = {1.3, 1.3 * 0.4, -1.3 * 0.25, 1.3 * 0.1};
    for (uint32_t cell = 0; cell < uniform->get_mesh()->n_owned(); cell++) {
        for (int i = 0; i < 4; i++) EXPECT_NEAR(uniform->h_conservatives(cell, i), U[i], 1e-12);
    }
    // With no boundary at all, momentum is conserved too
    auto blob = start(c, [](double x, double y, double z, double * W) {
        const double g = std::exp(-30.0 * ((x - 0.4) * (x - 0.4) + (y - 0.55) * (y - 0.55) + (z - 0.5) * (z - 0.5)));
        W[0] = 1.0 + 0.5 * g;
        W[1] = 0.6;
        W[2] = 0.3;
        W[3] = -0.2;
        W[4] = 1.0 + 0.8 * g;
    });
    const auto before = blob->integrate_conservatives();
    blob->run();
    const auto after = blob->integrate_conservatives();
    FOR_I_CONSERVATIVE EXPECT_NEAR(after[i], before[i], 1e-12) << "variable " << int(i);
}

INSTANTIATE_TEST_SUITE_P(Periodic, PeriodicBox3D,
    ::testing::Combine(::testing::Values("cartesian", "cartesian_tet", "cartesian_mixed"),
                       ::testing::Values("FO", "MUSCL", "TENO")));

namespace {

/**
 * @brief Max density error after advecting a periodic density wave for a
 *        quarter period with RK4, with dt ~ h^(order / 4).
 */
double advection_error(const std::string & recon, int order, uint32_t n) {
    Case c;
    c.n = {n, n, n};
    c.recon = recon;
    c.extra_recon = recon == "TENO" ? "order = " + std::to_string(order) + "\n" : "limiter = \"none\"\n";
    c.integrator = "RK4";
    const double T = 0.25;
    const uint32_t steps = std::ceil(T / (0.25 * std::pow(1.0 / n, std::max(1.0, order / 4.0))));
    c.run = "t_stop = " + std::to_string(T) + "\ndt = " + std::to_string(T / steps) + "\n";
    auto wave = [](double t) {
        return [t](double x, double y, double z, double * W) {
            W[0] = 1.0 + 0.2 * std::sin(2.0 * M_PI * (x - t)) * std::sin(2.0 * M_PI * (y - 0.5 * t)) *
                             std::cos(2.0 * M_PI * (z - 0.25 * t));
            W[1] = 1.0;
            W[2] = 0.5;
            W[3] = 0.25;
            W[4] = 1.0;
        };
    };
    auto solver = start(c, wave(0.0));
    solver->run();
    solver->copy_device_to_host();
    const auto exact = cell_averages_3d(*solver->get_mesh(), [&](double x, double y, double z, double * U) {
        double W[5];
        wave(T)(x, y, z, W);
        U[0] = W[0];
    });
    double err = 0.0;
    for (uint32_t i = 0; i < solver->get_mesh()->n_cells; i++) {
        err = std::max(err, std::abs(solver->h_conservatives(i, 0) - exact(i, 0)));
    }
    return err;
}

class PeriodicAdvection3D : public ::testing::TestWithParam<std::string> {};

} // namespace

TEST_P(PeriodicAdvection3D, SmoothWaveConvergesAtDesignOrderOnHexahedra) {
    const std::string scheme = GetParam();
    const std::string recon = scheme.rfind("TENO", 0) == 0 ? "TENO" : "MUSCL";
    const int order = recon == "TENO" ? std::stoi(scheme.substr(4)) : 2;
    const double e1 = advection_error(recon, order, 8);
    const double e2 = advection_error(recon, order, 16);
    EXPECT_GT(std::log2(e1 / e2), order - 0.25) << e1 << " " << e2;
}

INSTANTIATE_TEST_SUITE_P(Periodic, PeriodicAdvection3D, ::testing::Values("MUSCL", "TENO4"));

namespace {

using InvarianceParam = std::tuple<std::string, std::string, std::string>;  // mesh type, scheme, domain
class PeriodicInvariance3D : public ::testing::TestWithParam<InvarianceParam> {};

} // namespace

TEST_P(PeriodicInvariance3D, PulseCrossingTheSeamMatchesItsInteriorTranslate) {
    // On a uniform periodic mesh every block is equivalent: a pulse that
    // crosses the seams (in a box, through the corner) evolves exactly as the
    // same pulse translated by whole blocks into the interior, up to round-off.
    // A density jump around the core makes TENO select stencils there; the
    // channel has symmetry walls at y = 0 and y = L.
    const auto [mesh, scheme, domain] = GetParam();
    Case c;
    c.mesh = mesh;
    const uint32_t n = mesh == "cartesian_tet" ? 6 : 8;
    c.n = {n, n, n};
    c.L = {4.0, 4.0, 4.0};
    c.run = "t_stop = 0.5\ndt = 0.05\n";
    if (scheme == "MUSCL_NS") {
        c.recon = "MUSCL";
        c.extra_recon = "limiter = \"venkatakrishnan\"\n";
        c.physics = "type = \"navier_stokes\"\nmu = 0.02\n";
    } else {
        c.recon = "TENO";
        c.extra_recon = "order = 4\n";
    }
    const bool channel = domain == "channel";
    if (channel) {
        c.dirs = "xz";
        c.boundaries = "[[boundaries]]\nname = \"bottom\"\ntype = \"symmetry\"\n"
                       "[[boundaries]]\nname = \"top\"\ntype = \"symmetry\"\n";
    }
    const double L = c.L[0];
    const double h = L / c.n[0];
    auto pulse = [&](double xc, double yc, double zc) {
        return [=](double x, double y, double z, double * W) {
            // Periodic stand-ins for x - xc etc., so the field is smooth on the torus
            auto s = [&](double a) { return L / (2.0 * M_PI) * std::sin(2.0 * M_PI * a / L); };
            const double dx = s(x - xc), dy = s(y - yc), dz = s(z - zc);
            const double r2 = dx * dx + dy * dy + dz * dz;
            const double g = std::exp(-r2);
            W[0] = (1.0 + 0.3 * g) * (r2 < 0.8 ? 1.5 : 1.0);
            W[1] = 1.0 - 0.5 * g * dy;
            W[2] = (channel ? 0.0 : 1.0) + 0.5 * g * dx;
            W[3] = 1.0 + 0.3 * g * dx;
            W[4] = 1.0 + 0.2 * g;
        };
    };
    const int k = 3;
    const std::array<double, 3> t = {k * h, channel ? 0.0 : k * h, k * h};
    auto interior = start(c, pulse(2.0, 2.0, 2.0));
    auto across = start(c, pulse(2.0 + t[0], 2.0 + t[1], 2.0 + t[2]));
    interior->run();
    across->run();
    interior->copy_device_to_host();
    across->copy_device_to_host();
    const auto map = translated_cells(*interior->get_mesh(), t);
    double diff = 0.0;
    for (uint32_t cell = 0; cell < map.size(); cell++) {
        FOR_I_CONSERVATIVE diff = std::max(diff, std::abs(across->h_conservatives(map[cell], i) -
                                                          interior->h_conservatives(cell, i)));
    }
    EXPECT_LT(diff, 1e-10);
}

INSTANTIATE_TEST_SUITE_P(Periodic, PeriodicInvariance3D,
    ::testing::Values(InvarianceParam{"cartesian", "TENO", "box"}, InvarianceParam{"cartesian", "TENO", "channel"},
                      InvarianceParam{"cartesian_tet", "TENO", "channel"},
                      InvarianceParam{"cartesian_tet", "MUSCL_NS", "box"}));

/**
 * @file periodic_test.cpp
 * @author Matthew Bonanni (mbonanni001@gmail.com)
 * @brief Periodic boundaries in 2D: seam geometry, conservation, design order
 *        and translation invariance across the seam.
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
#include <map>
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

std::shared_ptr<Mesh> periodic_mesh(const std::string & type, uint32_t n, const std::string & dirs, double L = 1.0) {
    std::ostringstream s;
    s << "[mesh]\ntype = \"" << type << "\"\nNx = " << n << "\nNy = " << n << "\nLx = " << L << "\nLy = " << L
      << "\nperiodic = " << periodic_list(dirs) << "\n";
    auto mesh = std::make_shared<Mesh>();
    mesh->init(parse_toml(s.str()));
    mesh->copy_host_to_device();
    return mesh;
}

struct Case {
    std::string mesh = "cartesian";
    uint32_t n = 12;
    double L = 1.0;
    std::string dirs = "xy";
    std::string recon = "FO";
    std::string extra_recon;
    std::string physics = "type = \"euler\"\n";
    std::string run = "n_steps = 20\ncfl = 0.5\n";
    std::string integrator = "SSPRK3";
    std::string boundaries;
};

std::string make_input(const Case & c) {
    std::ostringstream s;
    s << "[run]\n" << c.run
      << "[mesh]\ntype = \"" << c.mesh << "\"\nNx = " << c.n << "\nNy = " << c.n << "\nLx = " << c.L
      << "\nLy = " << c.L << "\nperiodic = " << periodic_list(c.dirs) << "\n"
      << "[initialize]\ntype = \"constant\"\nu = [0.0, 0.0]\np = 1.0\nT = 1.0\n"
      << c.boundaries
      << "[numerics]\nriemann_solver = \"HLLC\"\ntime_integrator = \"" << c.integrator << "\"\n"
      << "[numerics.face_reconstruction]\ntype = \"" << c.recon << "\"\n" << c.extra_recon
      << "[physics]\n" << c.physics << "gamma = 1.4\np_ref = 1.0\nT_ref = 1.0\nrho_ref = 1.0\n"
      << "[output]\ncheck_interval = 1000000\n";
    return s.str();
}

using PrimitiveField = std::function<void(double, double, double *)>;  // (x, y) -> W = [rho, u, v, p]

/**
 * @brief Solver for a case, started from the exact cell averages of the
 *        conservatives of W(x, y).
 */
std::unique_ptr<Solver> start(const Case & c, const PrimitiveField & W) {
    auto solver = std::make_unique<Solver>();
    solver->init(parse_toml(make_input(c)));
    set_cell_averages(*solver, [&](double x, double y, double, double * U) {
        double w[4];
        W(x, y, w);
        U[0] = w[0];
        U[1] = w[0] * w[1];
        U[2] = w[0] * w[2];
        U[3] = w[3] / (GAMMA - 1.0) + 0.5 * w[0] * (w[1] * w[1] + w[2] * w[2]);
    });
    return solver;
}

using MeshParam = std::tuple<std::string, std::string>;  // mesh type, periodic directions
class PeriodicMesh2D : public ::testing::TestWithParam<MeshParam> {};

} // namespace

TEST_P(PeriodicMesh2D, SeamFacesAndVertexNeighborsMeetAfterTheirShift) {
    const auto [type, dirs] = GetParam();
    auto mesh = periodic_mesh(type, 5, dirs, 2.0);
    expect_shifts_join_cells(*mesh);
    // Only the non-periodic sides keep boundary faces
    const std::map<char, std::array<const char *, 2>> zones = {{'x', {"left", "right"}}, {'y', {"bottom", "top"}}};
    for (const auto & [dir, names] : zones) {
        for (const char * name : names) {
            EXPECT_EQ(mesh->get_face_zone(name) == nullptr, dirs.find(dir) != std::string::npos) << name;
        }
    }
}

INSTANTIATE_TEST_SUITE_P(Periodic, PeriodicMesh2D,
    ::testing::Values(MeshParam{"cartesian", "x"}, MeshParam{"cartesian", "xy"},
                      MeshParam{"cartesian_tri", "y"}, MeshParam{"cartesian_tri", "xy"}));

TEST(PeriodicSetup2D, InvalidSetupsAreRejected) {
    // Two cells across a periodic direction touch both directly and across the seam
    EXPECT_THROW(periodic_mesh("cartesian", 2, "x"), std::runtime_error);
    EXPECT_THROW(periodic_mesh("wedge", 8, "y"), std::runtime_error);
    // A periodic zone takes no boundary condition
    Case c;
    c.dirs = "x";
    c.boundaries = "[[boundaries]]\nname = \"left\"\ntype = \"symmetry\"\n";
    Solver solver;
    EXPECT_THROW(solver.init(parse_toml(make_input(c))), std::runtime_error);
}

namespace {

using SchemeParam = std::tuple<std::string, std::string>;  // mesh type, reconstruction
class PeriodicBox2D : public ::testing::TestWithParam<SchemeParam> {};

} // namespace

TEST_P(PeriodicBox2D, PreservesUniformFlowAndConservesMassMomentumAndEnergy) {
    Case c;
    std::tie(c.mesh, c.recon) = GetParam();
    auto uniform = start(c, [](double, double, double * W) {
        W[0] = 1.3;
        W[1] = 0.4;
        W[2] = -0.25;
        W[3] = 0.9;
    });
    uniform->run();
    uniform->copy_device_to_host();
    for (uint32_t i = 0; i < uniform->get_mesh()->n_cells; i++) {
        EXPECT_NEAR(uniform->h_conservatives(i, 0), 1.3, 1e-12);
        EXPECT_NEAR(uniform->h_conservatives(i, 1), 1.3 * 0.4, 1e-12);
        EXPECT_NEAR(uniform->h_conservatives(i, 2), -1.3 * 0.25, 1e-12);
    }
    // With no boundary at all, momentum is conserved too
    auto blob = start(c, [](double x, double y, double * W) {
        const double g = std::exp(-40.0 * ((x - 0.4) * (x - 0.4) + (y - 0.55) * (y - 0.55)));
        W[0] = 1.0 + 0.5 * g;
        W[1] = 0.6;
        W[2] = 0.3;
        W[3] = 1.0 + 0.8 * g;
    });
    const auto before = blob->integrate_conservatives();
    blob->run();
    const auto after = blob->integrate_conservatives();
    FOR_I_CONSERVATIVE EXPECT_NEAR(after[i], before[i], 1e-12) << "variable " << int(i);
}

INSTANTIATE_TEST_SUITE_P(Periodic, PeriodicBox2D,
    ::testing::Combine(::testing::Values("cartesian", "cartesian_tri"), ::testing::Values("FO", "MUSCL", "TENO")));

namespace {

/**
 * @brief Max density error after advecting a periodic density wave for a
 *        quarter period at (1, 0.5) with RK4, with dt ~ h^(order / 4) so the
 *        time error stays below the spatial one.
 */
double advection_error(const std::string & mesh, const std::string & recon, int order, uint32_t n) {
    Case c;
    c.mesh = mesh;
    c.n = n;
    c.recon = recon;
    c.extra_recon = recon == "TENO" ? "order = " + std::to_string(order) + "\n" : "limiter = \"none\"\n";
    c.integrator = "RK4";
    const double T = 0.25;
    const double h = 1.0 / n;
    const uint32_t steps = std::ceil(T / (0.25 * std::pow(h, std::max(1.0, order / 4.0))));
    c.run = "t_stop = " + std::to_string(T) + "\ndt = " + std::to_string(T / steps) + "\n";
    auto wave = [](double t) {
        return [t](double x, double y, double * W) {
            W[0] = 1.0 + 0.2 * std::sin(2.0 * M_PI * (x - t)) * std::sin(2.0 * M_PI * (y - 0.5 * t));
            W[1] = 1.0;
            W[2] = 0.5;
            W[3] = 1.0;
        };
    };
    auto solver = start(c, wave(0.0));
    solver->run();
    solver->copy_device_to_host();
    const auto exact = cell_averages(*solver->get_mesh(), [&](double x, double y, double * U) {
        double W[4];
        wave(T)(x, y, W);
        U[0] = W[0];
    });
    double err = 0.0;
    for (uint32_t i = 0; i < solver->get_mesh()->n_cells; i++) {
        err = std::max(err, std::abs(solver->h_conservatives(i, 0) - exact(i, 0)));
    }
    return err;
}

using OrderParam = std::tuple<std::string, std::string>;  // mesh type, MUSCL or TENO<order>
class PeriodicAdvection2D : public ::testing::TestWithParam<OrderParam> {};

} // namespace

TEST_P(PeriodicAdvection2D, SmoothWaveConvergesAtDesignOrder) {
    const auto [mesh, scheme] = GetParam();
    const std::string recon = scheme.rfind("TENO", 0) == 0 ? "TENO" : "MUSCL";
    const int order = recon == "TENO" ? std::stoi(scheme.substr(4)) : 2;
    const double e1 = advection_error(mesh, recon, order, 16);
    const double e2 = advection_error(mesh, recon, order, 32);
    EXPECT_GT(std::log2(e1 / e2), order - 0.25) << e1 << " " << e2;
}

INSTANTIATE_TEST_SUITE_P(Periodic, PeriodicAdvection2D,
    ::testing::Combine(::testing::Values("cartesian", "cartesian_tri"),
                       ::testing::Values("MUSCL", "TENO3", "TENO4", "TENO5", "TENO6")));

namespace {

using InvarianceParam = std::tuple<std::string, std::string, std::string>;  // mesh type, scheme, domain
class PeriodicInvariance2D : public ::testing::TestWithParam<InvarianceParam> {};

} // namespace

TEST_P(PeriodicInvariance2D, VortexCrossingTheSeamMatchesItsInteriorTranslate) {
    // On a uniform periodic mesh every cell is equivalent: an isentropic vortex
    // that crosses the seam (in a box, through the corner) evolves exactly as
    // the same vortex translated by whole cells into the interior, up to
    // round-off. The "contact" scheme adds a density jump around the core, so
    // that TENO selects stencils there; the channel has symmetry walls.
    const auto [mesh, scheme, domain] = GetParam();
    Case c;
    c.mesh = mesh;
    c.n = 20;
    c.L = 10.0;
    c.run = "t_stop = 1.0\ndt = 0.04\n";
    if (scheme == "MUSCL_NS") {
        c.recon = "MUSCL";
        c.extra_recon = "limiter = \"venkatakrishnan\"\n";
        c.physics = "type = \"navier_stokes\"\nmu = 0.02\n";
    } else {
        c.recon = "TENO";
        c.extra_recon = "order = 5\n";
    }
    const bool channel = domain == "channel";
    if (channel) {
        c.dirs = "x";
        c.boundaries = "[[boundaries]]\nname = \"bottom\"\ntype = \"symmetry\"\n"
                       "[[boundaries]]\nname = \"top\"\ntype = \"symmetry\"\n";
    }
    const bool contact = scheme == "TENO_contact";
    const double h = c.L / c.n;
    auto vortex = [&](double xc, double yc) {
        return [=](double x, double y, double * W) {
            // Periodic stand-ins for x - xc and y - yc, so the field is smooth on the torus
            const double dx = c.L / (2.0 * M_PI) * std::sin(2.0 * M_PI * (x - xc) / c.L);
            const double dy = c.L / (2.0 * M_PI) * std::sin(2.0 * M_PI * (y - yc) / c.L);
            const double beta = 5.0;
            const double e = std::exp(0.5 * (1.0 - dx * dx - dy * dy));
            const double T = 1.0 - (GAMMA - 1.0) * beta * beta / (8.0 * GAMMA * M_PI * M_PI) * e * e;
            W[0] = std::pow(T, 1.0 / (GAMMA - 1.0));
            W[1] = 1.0 - beta / (2.0 * M_PI) * e * dy;
            W[2] = (channel ? 0.0 : 1.0) + beta / (2.0 * M_PI) * e * dx;
            W[3] = W[0] * T;
            if (contact && dx * dx + dy * dy < 2.25) W[0] *= 1.5;
        };
    };
    // Interior: from (5, 5) to (6, 5 or 6); across: from just before the seam to past it
    const int k = 9;
    const std::array<double, 3> t = {k * h, channel ? 0.0 : k * h, 0.0};
    auto interior = start(c, vortex(5.0, 5.0));
    auto across = start(c, vortex(5.0 + t[0], 5.0 + t[1]));
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

INSTANTIATE_TEST_SUITE_P(Periodic, PeriodicInvariance2D,
    ::testing::Combine(::testing::Values("cartesian", "cartesian_tri"),
                       ::testing::Values("TENO", "TENO_contact", "MUSCL_NS"),
                       ::testing::Values("box", "channel")));

TEST(PeriodicSetup2D, TransmissiveImagesAreFoundAcrossTheSeam) {
    // On a mesh sheared by 45 degrees, the image of a bottom face (inward along
    // the normal) lies in the neighbor to the left, which for the first column
    // is across the periodic seam
    const uint32_t n = 6;
    const double h = 1.0 / n;
    std::vector<std::array<rtype, N_DIM>> nodes;
    for (uint32_t j = 0; j <= n; j++) {
        for (uint32_t i = 0; i <= n; i++) nodes.push_back({i * h + j * h, j * h});
    }
    auto id = [&](uint32_t i, uint32_t j) { return j * (n + 1) + i; };
    std::vector<std::vector<uint32_t>> cells;
    std::vector<Mesh::BoundaryFace> faces;
    for (uint32_t j = 0; j < n; j++) {
        for (uint32_t i = 0; i < n; i++) cells.push_back({id(i, j), id(i + 1, j), id(i + 1, j + 1), id(i, j + 1)});
        faces.push_back({{id(0, j), id(0, j + 1)}, "left"});
        faces.push_back({{id(n, j), id(n, j + 1)}, "right"});
    }
    for (uint32_t i = 0; i < n; i++) {
        faces.push_back({{id(i, 0), id(i + 1, 0)}, "bottom"});
        faces.push_back({{id(i, n), id(i + 1, n)}, "top"});
    }
    Mesh mesh;
    mesh.init_from_connectivity(nodes, cells, faces, "unassigned",
                                std::vector<Mesh::PeriodicPair>{{"left", "right", {1.0, 0.0}}});
    const BoundaryData bd = make_uniform_boundaries(mesh, BoundaryType::EXTRAPOLATION);
    auto h_image_face = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), bd.face_image_face);
    uint32_t n_boundary = 0;
    for (uint32_t f = 0; f < mesh.n_faces; f++) {
        if (mesh.h_cells_of_face(f, 1) >= 0) continue;
        n_boundary++;
        EXPECT_GE(h_image_face(f), 0) << "face " << f;
    }
    EXPECT_EQ(n_boundary, 2 * n);
}

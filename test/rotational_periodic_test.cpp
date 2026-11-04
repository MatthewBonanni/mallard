/**
 * @file rotational_periodic_test.cpp
 * @author Matthew Bonanni (mbonanni001@gmail.com)
 * @brief Rotational periodicity in 2D and 3D: matching through the rotation,
 *        vectors turned across the seam, a sector against the full annulus,
 *        conservation, and distributed runs.
 * @version 0.1
 * @date 2026-10-15
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 *
 */

#include <gtest/gtest.h>
#include <Kokkos_Core.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <map>
#include <numbers>
#include <sstream>
#include <string>
#include <tuple>

#include "gradient.h"
#include "mesh_block.h"
#include "mesh_quality.h"
#include "mpi_compare.h"
#include "partition.h"
#include "periodic.h"
#include "periodic_fixtures.h"
#include "solver.h"
#include "test_fixtures.h"
#include "test_utils.h"

#if N_DIM == 2
#define MPI_SUITE MPITest
#else
#define MPI_SUITE MPI3DTest
#endif

namespace {

constexpr double GAMMA = 1.4;
constexpr double PI = std::numbers::pi;

struct Annulus {
    uint32_t n_theta = 6;  // per sector
    uint32_t n_r = 5;
    uint32_t n_z = 3;
    int sectors = 4;      // the sector is 1 / sectors of the annulus
    bool full = false;    // the whole annulus instead of one sector
    std::string recon = "type = \"MUSCL\"\n";
    std::string physics = "type = \"euler\"\n";
    uint32_t n_steps = 8;
    std::string extra;
};

std::string annulus_input(const Annulus & a) {
    std::ostringstream s;
    const uint32_t n_theta = a.full ? a.n_theta * uint32_t(a.sectors) : a.n_theta;
    s << "[run]\nn_steps = " << a.n_steps << "\ndt = 0.004\n"
      << "[mesh]\ntype = \"annulus\"\nNtheta = " << n_theta << "\nNr = " << a.n_r << "\nNz = " << a.n_z
      << "\nr_inner = 1.0\nr_outer = 2.0\nLz = 0.6\nangle = " << (a.full ? 360.0 : 360.0 / a.sectors) << "\n";
    if (!a.full) s << "[[periodic]]\nzones = [\"start\", \"end\"]\n";
    if (N_DIM == 3) s << "[[periodic]]\nzones = [\"back\", \"front\"]\n";
    s << "[initialize]\ntype = \"constant\"\nu = [" << (N_DIM == 2 ? "0.0, 0.0" : "0.0, 0.0, 0.0") << "]\np = 1.0\nT = 1.0\n"
      << "[[boundaries]]\nname = \"inner\"\ntype = \"symmetry\"\n"
      << "[[boundaries]]\nname = \"outer\"\ntype = \"" << (a.physics.find("navier") != std::string::npos ? "wall_adiabatic" : "symmetry")
      << "\"\n"
      << "[numerics]\nriemann_solver = \"HLLC\"\ntime_integrator = \"SSPRK3\"\n"
      << "[numerics.face_reconstruction]\n" << a.recon
      << "[physics]\n" << a.physics << "gamma = 1.4\np_ref = 1.0\nT_ref = 1.0\nrho_ref = 1.0\n"
      << "[output]\ncheck_interval = 1000000\n" << a.extra;
    return s.str();
}

/**
 * @brief A swirling state with the sector's period: solid-body rotation plus
 *        radial, azimuthal and (3D) axial waves, W = [rho, u, p].
 */
void swirl(const int sectors, const double x, const double y, const double z, double * W) {
    const double r = std::hypot(x, y), theta = std::atan2(y, x);
    const double phase = sectors * theta;
    const double u_r = 0.1 * std::sin(phase) * std::sin(PI * (r - 1.0));
    const double u_t = 0.5 * r + 0.1 * std::cos(phase);
    W[0] = 1.0 + 0.2 * std::sin(phase) + 0.1 * (r - 1.5);
    W[1] = u_r * std::cos(theta) - u_t * std::sin(theta);
    W[2] = u_r * std::sin(theta) + u_t * std::cos(theta);
    if constexpr (N_DIM == 3) W[3] = 0.2 + 0.1 * std::sin(2.0 * PI * z / 0.6) * std::cos(phase);
    W[N_DIM + 1] = 1.0 + 0.1 * r * std::cos(phase);
}

/** @brief The [initialize] table of swirl(). */
std::string swirl_initializer(const int sectors) {
    const std::string r = "sqrt(x^2 + y^2)", phase = "(" + std::to_string(sectors) + " * atan2(y, x))";
    const std::string u_r = "(0.1 * sin" + phase + " * sin(pi * (" + r + " - 1)))";
    const std::string u_t = "(0.5 * " + r + " + 0.1 * cos" + phase + ")";
    std::string s = "[initialize]\ntype = \"analytical\"\n";
    s += "rho = \"1 + 0.2 * sin" + phase + " + 0.1 * (" + r + " - 1.5)\"\n";
    s += "p = \"1 + 0.1 * " + r + " * cos" + phase + "\"\n";
    s += "u = [\"(" + u_r + " * x - " + u_t + " * y) / " + r + "\", \"(" + u_r + " * y + " + u_t + " * x) / " + r + "\"";
    if (N_DIM == 3) s += ", \"0.2 + 0.1 * sin(2 * pi * z / 0.6) * cos" + phase + "\"";
    return s + "]\n";
}

void conservatives(const double * W, double * U) {
    U[0] = W[0];
    double k = 0.0;
    FOR_I_DIM {
        U[1 + i] = W[0] * W[1 + i];
        k += W[1 + i] * W[1 + i];
    }
    U[N_DIM + 1] = W[N_DIM + 1] / (GAMMA - 1.0) + 0.5 * W[0] * k;
}

std::unique_ptr<Solver> start(const Annulus & a) {
    auto solver = std::make_unique<Solver>();
    solver->set_distributed(false);
    solver->init(parse_toml(annulus_input(a)));
    set_cell_averages(*solver, [&](double x, double y, double z, double * U) {
        double W[N_CONSERVATIVE];
        swirl(a.sectors, x, y, z, W);
        conservatives(W, U);
    });
    return solver;
}

/** @brief Point x turned by angle about the z axis. */
std::array<double, 3> turn(const std::array<double, 3> & x, const double angle) {
    return {std::cos(angle) * x[0] - std::sin(angle) * x[1], std::sin(angle) * x[0] + std::cos(angle) * x[1], x[2]};
}

std::array<double, 3> centroid(const Mesh & mesh, const uint32_t c) {
    std::array<double, 3> x = {0.0, 0.0, 0.0};
    FOR_I_DIM x[i] = double(mesh.h_cell_coords(c, i));
    return x;
}

/**
 * @brief For each cell of the full annulus, the sector cell it is a rotated
 *        copy of and the number of sector turns.
 */
std::vector<std::pair<uint32_t, int>> sector_copies(const Mesh & full, const Mesh & sector, const int sectors) {
    const double h = 1e-3 * std::pow(double(sector.h_cell_volume(0)), 1.0 / N_DIM);
    auto key = [&](const std::array<double, 3> & x) {
        return std::array<long long, 3>{std::llround(x[0] / h), std::llround(x[1] / h), std::llround(x[2] / h)};
    };
    std::map<std::array<long long, 3>, uint32_t> cell_at;
    for (uint32_t c = 0; c < sector.n_cells; c++) cell_at[key(centroid(sector, c))] = c;
    const double alpha = 2.0 * PI / sectors;
    std::vector<std::pair<uint32_t, int>> out(full.n_cells);
    for (uint32_t c = 0; c < full.n_cells; c++) {
        const auto x = centroid(full, c);
        double theta = std::atan2(x[1], x[0]);
        if (theta < 0.0) theta += 2.0 * PI;
        const int k = static_cast<int>(std::floor(theta / alpha));
        const auto it = cell_at.find(key(turn(x, -k * alpha)));
        if (it == cell_at.end()) throw std::runtime_error("no sector cell for an annulus cell");
        out[c] = {it->second, k};
    }
    return out;
}

/** @brief Largest difference between the annulus and the turned copies of the sector, over all variables. */
double sector_annulus_difference(const Annulus & a) {
    Annulus s = a, f = a;
    s.full = false;
    f.full = true;
    auto sector = start(s);
    auto full = start(f);
    sector->run();
    full->run();
    sector->copy_device_to_host();
    full->copy_device_to_host();
    const auto copies = sector_copies(*full->get_mesh(), *sector->get_mesh(), a.sectors);
    double diff = 0.0;
    for (uint32_t c = 0; c < copies.size(); c++) {
        const auto [cs, k] = copies[c];
        std::array<double, 3> m = {0.0, 0.0, 0.0};
        FOR_I_DIM m[i] = double(sector->h_conservatives(cs, 1 + i));
        m = turn(m, k * 2.0 * PI / a.sectors);
        diff = std::max(diff, std::abs(double(full->h_conservatives(c, 0)) - double(sector->h_conservatives(cs, 0))));
        diff = std::max(diff, std::abs(double(full->h_conservatives(c, N_DIM + 1)) -
                                       double(sector->h_conservatives(cs, N_DIM + 1))));
        FOR_I_DIM diff = std::max(diff, std::abs(double(full->h_conservatives(c, 1 + i)) - m[i]));
    }
    return diff;
}

using SchemeParam = std::tuple<std::string, std::string>;  // reconstruction, physics
class RotationalSector : public ::testing::TestWithParam<SchemeParam> {};

} // namespace

TEST(RotationalPeriodicSetup, SeamFacesMeetThroughTheRotation) {
    // The nodes of every seam face, moved into the frame of its cell 1, are
    // nodes of that cell, and the face's normal, turned with it, points into it
    Mesh mesh;
    Annulus a;
    mesh.init(parse_toml(annulus_input(a)));
    ASSERT_TRUE(mesh.is_rotational());
    EXPECT_EQ(mesh.get_face_zone("start"), nullptr);
    EXPECT_EQ(mesh.get_face_zone("end"), nullptr);
    uint32_t n_turned = 0;
    for (uint32_t f = 0; f < mesh.n_faces; f++) {
        const int32_t c1 = mesh.h_cells_of_face(f, 1);
        if (c1 < 0 || !mesh.h_rotated_row(mesh.h_face_shift(f))) continue;
        n_turned++;
        const uint32_t c = uint32_t(c1);
        for (uint32_t k = 0; k < mesh.h_n_nodes_of_face(f); k++) {
            rtype x[N_DIM], y[N_DIM];
            FOR_I_DIM x[i] = mesh.h_node_coords(mesh.h_node_of_face(f, k), i);
            mesh.h_backward(mesh.h_face_shift(f), x, y);
            double nearest = 1e300;
            for (uint32_t j = 0; j < mesh.h_n_nodes_of_cell(c); j++) {
                double d = 0.0;
                FOR_I_DIM d = std::max(d, std::abs(double(y[i]) - double(mesh.h_node_coords(mesh.h_node_of_cell(c, j), i))));
                nearest = std::min(nearest, d);
            }
            EXPECT_LT(nearest, roundoff(1e-12)) << "face " << f;
        }
        const auto x_f = mesh.h_face_point(f, c);
        const auto n_f = mesh.h_face_normal(f, c);
        double into = 0.0;
        FOR_I_DIM into += (double(mesh.h_cell_coords(c, i)) - double(x_f[i])) * double(n_f[i]);
        EXPECT_GT(into, 0.0) << "face " << f;
    }
    EXPECT_EQ(n_turned, a.n_r * (N_DIM == 3 ? a.n_z : 1));
}

TEST(RotationalPeriodicSetup, SectorHasTheMeshQualityOfTheFullAnnulus) {
    // Quality metrics across the seam see the neighbor through the rotation:
    // the sector's worst cells are the full annulus's
    auto worst = [](bool full) {
        Annulus a;
        a.full = full;
        Mesh mesh;
        mesh.init(parse_toml(annulus_input(a)));
        const MeshQuality q = mesh_quality(mesh);
        std::array<double, MeshQuality::N_METRICS> m{};
        for (int k = 0; k < MeshQuality::N_METRICS; k++) {
            m[size_t(k)] = *std::max_element(q.cells[size_t(k)].begin(), q.cells[size_t(k)].end());
        }
        return m;
    };
    const auto sector = worst(false), annulus = worst(true);
    // Angles near zero come from acos, which turns rounding into its square root
    const double tol = 10.0 * std::sqrt(double(std::numeric_limits<rtype>::epsilon())) * 180.0 / PI;
    for (int k = 0; k < MeshQuality::N_METRICS; k++) {
        EXPECT_NEAR(sector[size_t(k)], annulus[size_t(k)], tol * (1.0 + annulus[size_t(k)]))
            << MeshQuality::NAMES[size_t(k)];
    }
}

TEST(RotationalPeriodicSetup, NodesMatchWithinTheToleranceOnly) {
    // Nodes of the end zone moved off the rotated start zone: by a tenth of the
    // matching tolerance they still match, by ten times it the mesh is rejected
    const double angle = PI / 3.0;
    const uint32_t n_r = 3;
    const MeshBlock block = annulus_block(5, n_r, 3, 1.0, 2.0, angle, 0.5, 0, 1);
    // The shortest edge of the zones' faces: radial, or in 3D axial
    const double h = N_DIM == 2 ? 1.0 / n_r : std::min(1.0 / n_r, 0.5 / 3.0);
    const double tol = double(periodic_tolerance(rtype(h)));
    std::vector<bool> on_end(block.n_nodes(), false);
    for (uint64_t k = 0; k < block.n_faces(); k++) {
        if (block.zone_names[block.face_zone[k]] != "end") continue;
        for (uint64_t j = block.face_offsets[k]; j < block.face_offsets[k + 1]; j++) on_end[block.face_nodes[j]] = true;
    }
    auto build = [&](const double fraction) {
        MeshBlock b = block;
        for (uint64_t n = 0; n < b.n_nodes(); n++) {
            if (on_end[n]) b.node_coords[n][0] += fraction * tol;
        }
        Mesh::PeriodicPair pair{"start", "end"};
        pair.angle = rtype(angle);
        std::vector<Mesh::PeriodicPair> pairs = {pair};
        if (N_DIM == 3) {
            Mesh::PeriodicPair z{"back", "front"};
            z.translation[N_DIM - 1] = 0.5;
            pairs.push_back(z);
        }
        Mesh mesh;
        mesh.init_from_block(b, pairs);
    };
    EXPECT_NO_THROW(build(0.1));
    EXPECT_THROW(build(10.0), std::runtime_error);
}

TEST(RotationalPeriodicSetup, InvalidPairsAreRejected) {
    auto mesh_with = [](const std::string & periodic) {
        Mesh mesh;
        mesh.init(parse_toml(std::string("[mesh]\ntype = \"annulus\"\nNtheta = 6\nNr = 3\nNz = 3\nangle = 60.0\n") + periodic));
    };
    // The wrong angle leaves the zones unmatched
    EXPECT_THROW(mesh_with("[[periodic]]\nzones = [\"start\", \"end\"]\nangle = 50.0\n"), std::runtime_error);
    // A rotation and a translation that do not commute
    const std::string across = N_DIM == 2 ? "[[periodic]]\nzones = [\"inner\", \"outer\"]\ntranslation = [0.0, 1.0]\n"
                                          : "[[periodic]]\nzones = [\"inner\", \"outer\"]\ntranslation = [0.0, 1.0, 0.0]\n";
    EXPECT_THROW(mesh_with("[[periodic]]\nzones = [\"start\", \"end\"]\nangle = 60.0\n" + across), std::runtime_error);
    // A pair is a translation or a rotation
    EXPECT_THROW(mesh_with(std::string("[[periodic]]\nzones = [\"start\", \"end\"]\nangle = 60.0\ntranslation = ") +
                           (N_DIM == 2 ? "[1.0, 0.0]\n" : "[1.0, 0.0, 0.0]\n")),
                 std::exception);
    // Fewer than 3 cells around
    EXPECT_THROW(Mesh().init(parse_toml("[mesh]\ntype = \"annulus\"\nNtheta = 2\nNr = 3\nNz = 3\nangle = 60.0\n"
                                        "[[periodic]]\nzones = [\"start\", \"end\"]\n")),
                 std::runtime_error);
}

TEST(RotationalPeriodicSetup, LinearSwirlHasExactGradientsAcrossTheSeam) {
    // u = a x + b (z x x) turns with the sector, so the least-squares fits of
    // it are exact in every cell whose neighbors are all interior, including
    // those whose neighbors lie across the seam; unturned neighbors would make
    // them O(1) wrong there
    Mesh mesh;
    Annulus a;
    a.n_theta = 5;
    mesh.init(parse_toml(annulus_input(a)));
    mesh.copy_host_to_device();
    const BoundaryData boundaries = make_uniform_boundaries(mesh, BoundaryType::SYMMETRY);
    const double ga = 0.3, gb = 0.7;
    Kokkos::View<rtype *[N_CONSERVATIVE]> W("W", mesh.n_cells);
    auto h_W = Kokkos::create_mirror_view(W);
    for (uint32_t c = 0; c < mesh.n_cells; c++) {
        const auto x = centroid(mesh, c);
        h_W(c, 0) = 1.0;
        h_W(c, 1) = rtype(ga * x[0] - gb * x[1]);
        h_W(c, 2) = rtype(ga * x[1] + gb * x[0]);
        if constexpr (N_DIM == 3) h_W(c, 3) = 0.25;
        h_W(c, N_DIM + 1) = 1.0;
    }
    Kokkos::deep_copy(W, h_W);
    Kokkos::View<rtype *[N_CONSERVATIVE][N_DIM]> g_face("g_face", mesh.n_cells), g_vertex("g_vertex", mesh.n_cells);
    const LSQGradientFunctor faces = make_gradient(mesh, boundaries, W, g_face);
    Kokkos::parallel_for(mesh.n_cells, faces);
    LSQVertexGradientFunctor vertex = make_vertex_gradient(make_gradient(mesh, boundaries, W, g_vertex), mesh, false);
    Kokkos::parallel_for(mesh.n_cells, vertex);
    auto h_face = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), g_face);
    auto h_vertex = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), g_vertex);
    const double exact[2][2] = {{ga, -gb}, {gb, ga}};
    auto interior = [&](uint32_t c) {
        if (is_boundary_cell(mesh, c)) return false;
        for (uint32_t k = mesh.h_offsets_cells_of_cell(c); k < mesh.h_offsets_cells_of_cell(c + 1); k++) {
            if (is_boundary_cell(mesh, mesh.h_cells_of_cell(k))) return false;
        }
        return true;
    };
    uint32_t n_seam = 0;
    for (uint32_t c = 0; c < mesh.n_cells; c++) {
        bool seam = false;
        for (uint32_t k = mesh.h_offsets_cells_of_cell(c); k < mesh.h_offsets_cells_of_cell(c + 1); k++) {
            seam = seam || mesh.h_rotated_row(mesh.h_cells_of_cell_shift(k));
        }
        if (!interior(c)) continue;
        n_seam += seam;
        for (int i = 0; i < 2; i++) {
            for (int d = 0; d < N_DIM; d++) {
                const double e = d < 2 ? exact[i][d] : 0.0;
                EXPECT_NEAR(h_face(c, 1 + i, d), e, roundoff(1e-11)) << "cell " << c;
                EXPECT_NEAR(h_vertex(c, 1 + i, d), e, roundoff(1e-11)) << "cell " << c;
            }
        }
    }
    EXPECT_GT(n_seam, 0u);
}

TEST_P(RotationalSector, MatchesTheFullAnnulus) {
    // The full annulus is the sector's mesh turned sectors times, so a state
    // with the sector's period evolves on it as on the sector, turned
    const auto [recon, physics] = GetParam();
    Annulus a;
    a.physics = physics == "NS" ? "type = \"navier_stokes\"\nmu = 0.01\n" : "type = \"euler\"\n";
    a.recon = recon == "MUSCL" ? "type = \"MUSCL\"\n" : "type = \"TENO\"\norder = " + recon.substr(4) + "\n";
    if (N_DIM == 3 && recon.rfind("TENO", 0) == 0) {
        a.n_theta = 5;
        a.n_r = 4;
    }
    if (recon.rfind("TENO", 0) == 0) SKIP_IN_SINGLE_PRECISION("round-off in the turned geometry flips TENO stencil selections");
    EXPECT_LT(sector_annulus_difference(a), precision_tol<double>(1e-11, 1e-4));
}

INSTANTIATE_TEST_SUITE_P(Rotational, RotationalSector,
    ::testing::Values(SchemeParam{"MUSCL", "Euler"}, SchemeParam{"MUSCL", "NS"}, SchemeParam{"TENO3", "Euler"},
                      SchemeParam{"TENO5", "Euler"}, SchemeParam{"TENO5", "NS"}));

TEST(RotationalPeriodicSetup, SectorConservesMassAndEnergy) {
    // Slip walls and the seam: no mass or energy leaves the sector
    for (const std::string recon : {"type = \"MUSCL\"\n", "type = \"TENO\"\norder = 5\n"}) {
        Annulus a;
        a.recon = recon;
        a.n_steps = 20;
        auto solver = start(a);
        const auto before = solver->integrate_conservatives();
        solver->run();
        const auto after = solver->integrate_conservatives();
        for (const uint8_t i : {uint8_t(0), uint8_t(N_DIM + 1)}) {
            const double b = double(before[i]), c = double(after[i]);
            EXPECT_NEAR(c, b, roundoff(1e-12) * std::abs(b)) << recon << ", variable " << int(i);
        }
    }
}

TEST(MPI_SUITE, RotationalSectorsMatchSerial) {
    // Halos and stencils across the turned seam, cut by each partitioner
    for (const std::string partitioner : {"hilbert", "multijagged", "graph"}) {
        if (partitioner == "graph" && !have_graph_partitioner()) continue;
        for (const std::string recon : {"type = \"MUSCL\"\n", "type = \"TENO\"\norder = 5\n"}) {
            Annulus a;
            a.n_theta = 8;
            a.n_steps = 5;
            a.recon = recon;
            a.physics = "type = \"navier_stokes\"\nmu = 0.01\n";
            a.extra = "[parallel]\npartitioner = \"" + partitioner + "\"\n";
            std::string input = annulus_input(a);
            const std::string init = "[initialize]\ntype = \"constant\"\nu = [" +
                                     std::string(N_DIM == 2 ? "0.0, 0.0" : "0.0, 0.0, 0.0") + "]\np = 1.0\nT = 1.0\n";
            input.replace(input.find(init), init.size(), swirl_initializer(a.sectors));
            expect_matches_serial(input);
        }
    }
}


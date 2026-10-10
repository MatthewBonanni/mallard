/**
 * @file teno3d_test.cpp
 * @author Matthew Bonanni (mbonanni001@gmail.com)
 * @brief Tests for the TENO-E reconstruction on 3D meshes.
 * @version 0.1
 * @date 2026-10-02
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 *
 */

#include <gtest/gtest.h>
#include <Kokkos_Core.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <sstream>
#include <string>
#include <tuple>

#include "test_fixtures.h"
#include "gmsh_fixtures.h"
#include "face_reconstruction.h"
#include "physics.h"
#include "solver.h"

namespace {

constexpr rtype GAMMA = 1.4;

// Smooth field satisfying symmetry conditions on the unit cube: density,
// pressure and tangential velocities are even across each wall, normal
// velocity odd
void smooth_conservatives(double x, double y, double z, double * U) {
    const double rho = 1.0 + 0.2 * std::cos(2.0 * M_PI * x) * std::cos(M_PI * y) * std::cos(M_PI * z);
    const double u = 0.1 * std::sin(2.0 * M_PI * x) * std::cos(M_PI * y) * std::cos(M_PI * z);
    const double v = -0.1 * std::sin(M_PI * y) * std::cos(M_PI * x) * std::cos(M_PI * z);
    const double w = 0.1 * std::sin(M_PI * z) * std::cos(M_PI * x) * std::cos(2.0 * M_PI * y);
    const double p = 1.0 + 0.1 * std::cos(M_PI * x) * std::cos(2.0 * M_PI * y) * std::cos(M_PI * z);
    U[0] = rho;
    U[1] = rho * u;
    U[2] = rho * v;
    U[3] = rho * w;
    U[4] = p / (double(GAMMA) - 1.0) + 0.5 * rho * (u * u + v * v + w * w);
}

// Quadratic density (velocity and energy constant in conservative form)
void quadratic_conservatives(double x, double y, double z, double * U) {
    U[0] = 1.0 + 0.3 * x * x + 0.2 * x * y - 0.1 * z * z + 0.2 * y * z + 0.1 * x - 0.2 * z;
    U[1] = 0.1;
    U[2] = -0.1;
    U[3] = 0.05;
    U[4] = 10.0;
}

// Quartic density
void quartic_conservatives(double x, double y, double z, double * U) {
    quadratic_conservatives(x, y, z, U);
    U[0] += 0.4 * x * x * y * z - 0.3 * z * z * z * z + 0.2 * x * y * y * y + 0.1 * x * x * z;
}

using Field = void (*)(double, double, double, double *);

std::unique_ptr<TENO> make_teno(std::shared_ptr<Mesh> mesh, const BoundaryData & bd, int order,
                                const std::string & extra = "") {
    auto teno = std::make_unique<TENO>();
    teno->set_mesh(mesh);
    teno->set_boundaries(bd);
    teno->init(parse_toml("type = \"TENO\"\norder = " + std::to_string(order) + "\n" + extra));
    return teno;
}

/**
 * @brief Max density error of the reconstruction at the face quadrature points
 *        (both sides), optionally only on faces at least margin from the walls.
 */
double reconstruction_error(std::shared_ptr<Mesh> mesh, int order, double margin = 0.0,
                            const std::string & extra = "", Field field = smooth_conservatives) {
    BoundaryData bd = make_uniform_boundaries(*mesh, BoundaryType::SYMMETRY, GAMMA);
    auto avg = cell_averages_3d(*mesh, field);
    Euler euler = Euler::from_reference(GAMMA, 1.0, 1.0, 1.0);
    Kokkos::View<rtype *[N_CONSERVATIVE]> W("W", mesh->n_cells);
    auto h_W = Kokkos::create_mirror_view(W);
    for (uint32_t c = 0; c < mesh->n_cells; c++) {
        rtype U[N_CONSERVATIVE], Wc[N_CONSERVATIVE];
        FOR_I_CONSERVATIVE U[i] = avg(c, i);
        euler.compute_W_from_conservatives(Wc, U);
        FOR_I_CONSERVATIVE h_W(c, i) = Wc[i];
    }
    Kokkos::deep_copy(W, h_W);

    auto teno = make_teno(mesh, bd, order, extra);
    const uint8_t n_quad = teno->n_face_quadrature_points();
    Kokkos::View<rtype **[2][N_CONSERVATIVE]> face_W("face_W", mesh->n_faces, n_quad);
    teno->calc_face_values(W, face_W);
    auto h_face_W = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), face_W);
    auto h_points = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), teno->face_quad_points);
    auto h_weights = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), teno->face_quad_weights);

    double err = 0.0;
    for (uint32_t f = 0; f < mesh->n_faces; f++) {
        for (uint8_t q = 0; q < n_quad; q++) {
            if (double(h_weights(f, q)) == 0.0) continue;
            const double x = double(h_points(f, q, 0)), y = double(h_points(f, q, 1)), z = double(h_points(f, q, 2));
            if (std::min({x, y, z}) < margin || std::max({x, y, z}) > 1.0 - margin) continue;
            double U[N_CONSERVATIVE];
            field(x, y, z, U);
            for (uint8_t side = 0; side < 2; side++) {
                if (mesh->h_cells_of_face(f, side) < 0) continue;
                err = std::max(err, std::abs(double(h_face_W(f, q, side, 0)) - U[0]));
            }
        }
    }
    return err;
}

double reconstruction_error(const std::string & mesh_type, uint32_t n, int order, double margin = 0.0,
                            const std::string & extra = "", Field field = smooth_conservatives) {
    return reconstruction_error(make_mesh_3d(mesh_type, n, n, n), order, margin, extra, field);
}

// Smooth: every cell takes the central stencil. Troubled: every cell runs the
// stencil selection, which must keep the central stencil on smooth data.
const char * SMOOTH = "troubled_threshold = 1e9\n";
const char * TROUBLED = "troubled_threshold = 0\n";

using ExactnessParam = std::tuple<std::string, int, uint32_t>;
class TENO3DExactness : public ::testing::TestWithParam<ExactnessParam> {};

} // namespace

TEST_P(TENO3DExactness, ReproducesPolynomialsOfTheReconstructionDegreeAwayFromWalls) {
    // Exactness checks the cell averages of the basis, the stencil
    // least-squares system (including rank detection) and the face quadrature
    // points; the wall mirrors are only exact for symmetric fields
    const auto [mesh_type, order, n] = GetParam();
    const Field field = (order == 3) ? quadratic_conservatives : quartic_conservatives;
    EXPECT_LT(reconstruction_error(mesh_type, n, order, 0.4, SMOOTH, field), precision_tol<double>(1e-10, 1e-5));
}

INSTANTIATE_TEST_SUITE_P(TENO, TENO3DExactness,
    ::testing::Values(ExactnessParam{"cartesian", 3, 8}, ExactnessParam{"cartesian_tet", 3, 8},
                      ExactnessParam{"cartesian_prism", 3, 8}, ExactnessParam{"cartesian_pyramid", 3, 8},
                      ExactnessParam{"cartesian_mixed", 3, 9}, ExactnessParam{"cartesian", 5, 10},
                      ExactnessParam{"cartesian_tet", 5, 10}, ExactnessParam{"cartesian", 6, 10}));

namespace {

using OrderParam = std::tuple<std::string, int, uint32_t, const char *>;
class TENO3DOrder : public ::testing::TestWithParam<OrderParam> {};

} // namespace

TEST_P(TENO3DOrder, SmoothReconstructionConvergesAtDesignOrder) {
    // Max error at face quadrature points over all faces, with mirrored
    // stencils at the symmetry walls
    const auto [mesh_type, order, n, mode] = GetParam();
    const double e1 = reconstruction_error(mesh_type, n, order, 0.0, mode);
    const double e2 = reconstruction_error(mesh_type, 2 * n, order, 0.0, mode);
    const double rate = std::log2(e1 / e2);
    std::cout << mesh_type << " order " << order << ": max errors " << e1 << ", " << e2 << ", rate " << rate
              << std::endl;
    EXPECT_GT(rate, order - 0.5);
}

INSTANTIATE_TEST_SUITE_P(TENO, TENO3DOrder,
    ::testing::Values(OrderParam{"cartesian", 3, 6, SMOOTH}, OrderParam{"cartesian", 4, 6, SMOOTH},
                      OrderParam{"cartesian", 5, 6, SMOOTH}, OrderParam{"cartesian_tet", 3, 4, SMOOTH},
                      OrderParam{"cartesian_tet", 4, 4, SMOOTH}, OrderParam{"cartesian_tet", 5, 4, SMOOTH},
                      OrderParam{"cartesian", 5, 6, TROUBLED},
                      OrderParam{"cartesian_tet", 4, 4, TROUBLED}));

namespace {

// Periodic on the unit cube
void periodic_conservatives(double x, double y, double z, double * U) {
    U[0] = 1.0 + 0.2 * std::sin(2.0 * M_PI * x) * std::cos(2.0 * M_PI * y) * std::sin(2.0 * M_PI * z) +
           0.1 * std::cos(2.0 * M_PI * (x + y - z));
    U[1] = 0.1;
    U[2] = -0.1;
    U[3] = 0.05;
    U[4] = 10.0;
}

} // namespace

TEST(TENO3DJittered, OrderFourBeatsOrderThreeOnJitteredHexahedraAndPrisms) {
    // The nearest 2 x DOFs cells of a jittered hex/prism mesh span about three
    // cell layers per direction, which resolves the cubic terms only through
    // small centroid offsets: full rank, but a reconstruction that amplifies
    // truncation errors tenfold, so order 4 was several times less accurate
    // than order 3 on smooth data
    const std::string file = write_temp("mallard_teno3d_jittered.msh", jittered_periodic_mesh_3d(6, 0.1));
    auto mesh = std::make_shared<Mesh>();
    mesh->init(parse_toml("[mesh]\ntype = \"file\"\nfilename = \"" + file + "\"\n"
                          "[[periodic]]\nzones = [\"left\", \"right\"]\ntranslation = [1.0, 0.0, 0.0]\n"
                          "[[periodic]]\nzones = [\"bottom\", \"top\"]\ntranslation = [0.0, 1.0, 0.0]\n"
                          "[[periodic]]\nzones = [\"back\", \"front\"]\ntranslation = [0.0, 0.0, 1.0]\n"));
    mesh->copy_host_to_device();
    const double e3 = reconstruction_error(mesh, 3, 0.0, SMOOTH, periodic_conservatives);
    const double e4 = reconstruction_error(mesh, 4, 0.0, SMOOTH, periodic_conservatives);
    std::cout << "jittered max errors: order 3 " << e3 << ", order 4 " << e4 << std::endl;
    EXPECT_LT(e4, e3);
}


namespace {

/**
 * @brief L1 density error after advecting a smooth density pulse with the full
 *        solver (uniform velocity and pressure: an exact Euler solution).
 */
double advection_error(const std::string & mesh, const std::string & recon, uint32_t n) {
    std::ostringstream in;
    in << "[run]\nt_stop = 0.1\ncfl = 0.2\n"
       << "[mesh]\ntype = \"" << mesh << "\"\nNx = " << n << "\nNy = " << n << "\nNz = " << n << "\nLx = 2.0\nLy = 2.0\nLz = 2.0\n"
       << "[initialize]\ntype = \"analytical\"\n"
       << "rho = \"1.0 + 0.3 * exp(-8 * ((x - 0.9)^2 + (y - 0.95)^2 + (z - 0.97)^2))\"\n"
       << "u = [\"1.0\", \"0.5\", \"0.3\"]\np = \"1.0\"\n";
    for (const char * zone : {"left", "right", "bottom", "top", "back", "front"}) {
        in << "[[boundaries]]\nname = \"" << zone << "\"\ntype = \"extrapolation\"\n";
    }
    in << "[numerics]\nriemann_solver = \"HLLC\"\ntime_integrator = \"SSPRK3\"\ncheck_nan = true\n"
       << "[numerics.face_reconstruction]\ntype = \"" << recon << "\"\norder = 5\n"
       << "[physics]\ntype = \"euler\"\ngamma = 1.4\np_ref = 1.0\nT_ref = 1.0\nrho_ref = 1.0\n"
       << "[output]\ncheck_interval = 1000000\n";
    Solver solver;
    solver.init(parse_toml(in.str()));
    solver.run();
    solver.copy_device_to_host();
    auto m = solver.get_mesh();
    auto exact = cell_averages_3d(*m, [](double x, double y, double z, double * U) {
        U[0] = 1.0 + 0.3 * std::exp(-8.0 * ((x - 1.0) * (x - 1.0) + (y - 1.0) * (y - 1.0) + (z - 1.0) * (z - 1.0)));
        for (int i = 1; i < N_CONSERVATIVE; i++) U[i] = 0.0;
    });
    double err = 0.0;
    for (uint32_t i = 0; i < m->n_cells; i++) {
        err += std::abs(double(solver.h_conservatives(i, 0)) - double(exact(i, 0))) * double(m->h_cell_volume(i));
    }
    return err;
}

} // namespace

TEST(TENO3DSolver, AdvectedPulseConvergesAndBeatsMUSCL) {
    // TENO5 with SSPRK3 (dt ~ h): third order overall once resolved
    const double t1 = advection_error("cartesian", "TENO", 12), t2 = advection_error("cartesian", "TENO", 24);
    const double m2 = advection_error("cartesian", "MUSCL", 24);
    std::cout << "advection L1: TENO5 " << t1 << " -> " << t2 << " (rate " << std::log2(t1 / t2) << "), MUSCL " << m2
              << std::endl;
    EXPECT_GT(std::log2(t1 / t2), 3.0);
    EXPECT_LT(t2, 0.25 * m2);
}

namespace {

/**
 * @brief Largest flow speed at t = 1 of a small velocity perturbation of a gas
 *        at rest (unit density and pressure), TENO with HLLC: grows only if the
 *        linearized scheme has a growing mode.
 * @param mesh TOML [mesh] table and any [[boundaries]] / [[periodic]] tables.
 */
double largest_speed_at_rest(const std::string & mesh, int order, const std::string & u) {
    std::ostringstream in;
    in << "[run]\nt_stop = 1.0\ncfl = 0.2\n" << mesh
       << "[initialize]\ntype = \"analytical\"\nrho = \"1.0\"\nu = [" << u << "]\np = \"1.0\"\n"
       << "[numerics]\nriemann_solver = \"HLLC\"\ntime_integrator = \"SSPRK3\"\n"
       << "[numerics.face_reconstruction]\ntype = \"TENO\"\norder = " << order << "\n"
       << "[physics]\ntype = \"euler\"\ngamma = 1.4\np_ref = 1.0\nT_ref = 1.0\nrho_ref = 1.0\n"
       << "[output]\ncheck_interval = 1000000\n";
    Solver solver;
    solver.init(parse_toml(in.str()));
    solver.run();
    solver.copy_device_to_host();
    double speed = 0.0;
    for (uint32_t i = 0; i < solver.get_mesh()->n_cells; i++) {
        double q2 = 0.0;
        for (int d = 0; d < N_DIM; d++) {
            q2 += std::pow(double(solver.h_conservatives(i, 1 + d)) / double(solver.h_conservatives(i, 0)), 2);
        }
        speed = std::max(speed, std::sqrt(q2));
    }
    return speed;
}

std::string prism_box(double lz, bool side_walls_only) {
    std::ostringstream in;
    in << "[mesh]\ntype = \"cartesian_prism\"\nNx = 6\nNy = 6\nNz = 4\nLx = 1.0\nLy = 1.0\nLz = " << lz << "\n";
    for (const char * zone : {"left", "right", "bottom", "top", "back", "front"}) {
        if (side_walls_only && std::string(zone) != "left" && std::string(zone) != "right") continue;
        in << "[[boundaries]]\nname = \"" << zone << "\"\ntype = \"symmetry\"\n";
    }
    if (side_walls_only) {
        in << "[[periodic]]\nzones = [\"bottom\", \"top\"]\ntranslation = [0.0, 1.0, 0.0]\n"
           << "[[periodic]]\nzones = [\"back\", \"front\"]\ntranslation = [0.0, 0.0, " << lz << "]\n";
    }
    return in.str();
}

} // namespace

TEST(TENO3DSolver, PerturbationsAtRestStayBoundedOnThinPrisms) {
    // Stencil candidates ranked by physical distance took the whole column
    // across thin cells first and resolved the other directions only through
    // small centroid offsets: full rank and a small Lebesgue constant, but a
    // grid-scale vortical mode grew like exp(13 t) on these prisms of aspect
    // ratio 6 (order 4, walls all around), to Mach 0.85 by t = 1
    const double lz = 4.0 / 6.0 / 6.25;
    const double speed = largest_speed_at_rest(
        prism_box(lz, false), 4,
        "\"1e-3 * sin(7 * x + 3 * y) * cos(5 * z / " + std::to_string(lz) + ")\", \"1e-3 * cos(4 * x - 6 * y)\", \"0.0\"");
    std::cout << "largest speed at t = 1: " << speed << std::endl;
    EXPECT_LT(speed, 1e-2);
}

class TENO3DSideWalls : public ::testing::TestWithParam<int> {};

TEST_P(TENO3DSideWalls, PerturbationsAtRestStayBoundedNextToSideWallsOfMildlyStretchedPrisms) {
    // A symmetry wall mirrors the prism tiling, whose cells are split along
    // the xy diagonal, so wall cells see a seam where the split flips. There
    // the first large stencil within a Lebesgue constant of 10 was a
    // near-degenerate one (8.7, against 3.4 in the interior) whose face values
    // follow the cells across the face: at aspect ratio 2 this grid-scale
    // perturbation reached 0.016 (order 4) and 0.007 (order 6) by t = 1 (#126)
    const double lz = 4.0 / 6.0 / 2.0;
    const double speed = largest_speed_at_rest(prism_box(lz, true), GetParam(),
                                               "\"1e-3 * sin(37 * x + 23 * y)\", \"1e-3 * cos(29 * x - 41 * y)\", \"0.0\"");
    std::cout << "order " << GetParam() << ": largest speed at t = 1: " << speed << std::endl;
    EXPECT_LT(speed, 2e-3);
}

INSTANTIATE_TEST_SUITE_P(EvenOrders, TENO3DSideWalls, ::testing::Values(4, 6));

TEST(TENO3DSolver, PerturbationsAtRestStayBoundedOnJitteredPrismsAtOrderSix) {
    // Without lattice ties the same near-degenerate stencils appear in the
    // interior: on jittered prisms the perturbation reached 0.02 by t = 1 at
    // order 6 (#229)
    const std::string file = write_temp("mallard_teno3d_jittered_prisms.msh", jittered_periodic_mesh_3d(8, 0.1, true));
    const std::string mesh = "[mesh]\ntype = \"file\"\nfilename = \"" + file + "\"\n"
                             "[[periodic]]\nzones = [\"left\", \"right\"]\ntranslation = [1.0, 0.0, 0.0]\n"
                             "[[periodic]]\nzones = [\"bottom\", \"top\"]\ntranslation = [0.0, 1.0, 0.0]\n"
                             "[[periodic]]\nzones = [\"back\", \"front\"]\ntranslation = [0.0, 0.0, 1.0]\n";
    const double speed = largest_speed_at_rest(mesh, 6,
                                               "\"1e-4 * sin(2 * 3.14159265 * (x + 2 * y))\", "
                                               "\"1e-4 * cos(2 * 3.14159265 * (3 * x - z))\", \"0.0\"");
    std::cout << "largest speed at t = 1: " << speed << std::endl;
    EXPECT_LT(speed, 1e-3);
}

TEST(TENO3DStencils, MetricIsTheIdentityOnRegularTilingsOnly) {
    // Regular tilings keep stencils ranked by physical distance, walls
    // included; stretched ones do not
    auto anisotropic_cells = [](const std::string & type, double lz) {
        auto mesh = make_mesh_3d(type, 6, 6, 6, 1.0, 1.0, lz);
        BoundaryData bd = make_uniform_boundaries(*mesh, BoundaryType::SYMMETRY, GAMMA);
        TENO teno;
        teno.set_mesh(mesh);
        teno.set_boundaries(bd);
        const std::array<double, 9> identity = {1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0};
        uint32_t n = 0;
        for (uint32_t i = 0; i < mesh->n_cells; i++) n += (teno.stencil_metric(i) != identity);
        return n;
    };
    for (const char * type : {"cartesian", "cartesian_tet", "cartesian_prism", "cartesian_pyramid", "cartesian_mixed"}) {
        EXPECT_EQ(anisotropic_cells(type, 1.0), 0u) << type;
    }
    for (const char * type : {"cartesian", "cartesian_tet", "cartesian_prism"}) {
        auto mesh = make_mesh_3d(type, 6, 6, 6, 1.0, 1.0, 1.0 / 6.25);
        EXPECT_EQ(anisotropic_cells(type, 1.0 / 6.25), mesh->n_cells) << type;
    }
}

namespace {

/** @brief Every precomputed table of a TENO, as raw bytes, by name. */
std::vector<std::pair<std::string, std::vector<char>>> teno_tables(const TENO & teno) {
    std::vector<std::pair<std::string, std::vector<char>>> out;
    auto add = [&](const std::string & name, const auto & view) {
        auto h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), view);
        const char * p = reinterpret_cast<const char *>(h.data());
        out.emplace_back(name, std::vector<char>(p, p + h.span() * sizeof(*h.data())));
    };
    add("scale", teno.scale);
    add("basis_mean", teno.basis_mean);
    add("moments", teno.moments);
    add("large sizes", teno.stencil_large_size);
    add("small sizes", teno.stencil_small_size);
    add("large slices", teno.stencil_large.slice_start);
    add("large cells", teno.stencil_large.cells);
    add("large faces", teno.stencil_large.faces);
    add("large pinv", teno.stencil_large.pinv);
    add("small slices", teno.stencil_small.slice_start);
    add("small cells", teno.stencil_small.cells);
    add("small faces", teno.stencil_small.faces);
    add("small pinv", teno.stencil_small.pinv);
    out.emplace_back("gather depth", std::vector<char>(teno.gather_depth.begin(), teno.gather_depth.end()));
    return out;
}

void expect_same_tables(const TENO & a, const TENO & b, const std::string & what) {
    const auto ta = teno_tables(a), tb = teno_tables(b);
    for (size_t k = 0; k < ta.size(); k++) EXPECT_TRUE(ta[k].second == tb[k].second) << what << ": " << ta[k].first;
}

/** @brief A TENO whose setup options are set before init(). */
template <typename F>
std::unique_ptr<TENO> make_teno_with(std::shared_ptr<Mesh> mesh, const BoundaryData & bd, int order, F && options) {
    auto teno = std::make_unique<TENO>();
    teno->set_mesh(mesh);
    teno->set_boundaries(bd);
    options(*teno);
    teno->init(parse_toml("type = \"TENO\"\norder = " + std::to_string(order) + "\n"));
    return teno;
}

/**
 * @brief Move the nodes nearest to the given points (x, y, z) by the given
 *        offsets (dx, dy, dz), recompute the geometry; the cells that hold them.
 */
std::vector<uint32_t> move_nodes(Mesh & mesh, const std::vector<std::array<double, 6>> & moves) {
    std::vector<uint32_t> moved;
    for (const auto & m : moves) {
        uint32_t best = 0;
        double best_d2 = 1e300;
        for (uint32_t n = 0; n < mesh.n_nodes; n++) {
            double d2 = 0.0;
            for (int d = 0; d < 3; d++) d2 += std::pow(double(mesh.h_node_coords(n, d)) - m[d], 2);
            if (d2 < best_d2) {
                best_d2 = d2;
                best = n;
            }
        }
        for (int d = 0; d < 3; d++) mesh.h_node_coords(best, d) += rtype(m[3 + d]);
        moved.push_back(best);
    }
    mesh.compute_geometry();
    mesh.copy_host_to_device();
    std::vector<uint32_t> cells;
    for (uint32_t c = 0; c < mesh.n_cells; c++) {
        for (uint32_t k = 0; k < mesh.h_n_nodes_of_cell(c); k++) {
            if (std::find(moved.begin(), moved.end(), mesh.h_node_of_cell(c, k)) != moved.end()) {
                cells.push_back(c);
                break;
            }
        }
    }
    return cells;
}

using IncrementalParam = std::tuple<std::string, int>;
class TENO3DIncremental : public ::testing::TestWithParam<IncrementalParam> {};

} // namespace

TEST_P(TENO3DIncremental, RebuildingTheCellsWithinReachOfMovedNodesGivesTheTablesOfAFullSetup) {
    // An interior node, and a node of the x = 0 symmetry wall moved within the
    // wall, which changes the mirror images of every stencil that reaches it
    const auto [type, order] = GetParam();
    auto mesh = make_mesh_3d(type, 9, 8, 6);
    BoundaryData bd = make_uniform_boundaries(*mesh, BoundaryType::SYMMETRY, GAMMA);
    auto incremental = make_teno(mesh, bd, order);
    auto changed_only = make_teno(mesh, bd, order);
    const auto before = teno_tables(*incremental);
    const double h = 0.125;
    const std::vector<uint32_t> changed =
        move_nodes(*mesh, {{0.22, 0.375, 0.5, 0.06 * h, -0.04 * h, 0.05 * h}, {0.0, 0.25, 0.33, 0.0, 0.05 * h, -0.04 * h}});
    const std::vector<uint32_t> cells = incremental->cells_within_reach(changed);
    ASSERT_GT(cells.size(), changed.size());
    ASSERT_LT(cells.size(), mesh->n_cells);
    incremental->rebuild_cells(cells);
    auto full = make_teno(mesh, bd, order);
    expect_same_tables(*incremental, *full, "rebuilt within reach");
    // The move changed tables beyond the moved cells: rebuilding only those is not enough
    changed_only->rebuild_cells(changed);
    EXPECT_FALSE(teno_tables(*changed_only) == teno_tables(*full));
    EXPECT_FALSE(before == teno_tables(*full));
}

INSTANTIATE_TEST_SUITE_P(TENO, TENO3DIncremental,
    ::testing::Values(IncrementalParam{"cartesian_prism", 4}, IncrementalParam{"cartesian_mixed", 4},
                      IncrementalParam{"cartesian_tet", 3}));

TEST(TENO3DSetup, TablesDoNotDependOnTheBatchOrOnWhetherTheDeviceOrTheHostSetsThemUp) {
    // Mixed cells with walls: pyramids, tetrahedra and mirror images; small
    // batches split the cells differently, and the host setup is the fallback
    // for cells that do not fit the device's scratch
    auto mesh = make_mesh_3d("cartesian_mixed", 4, 4, 4);
    BoundaryData bd = make_uniform_boundaries(*mesh, BoundaryType::SYMMETRY, GAMMA);
    auto reference = make_teno(mesh, bd, 5);
    auto batched = make_teno_with(mesh, bd, 5, [](TENO & t) { t.setup_batch_cells = 64; });
    auto host = make_teno_with(mesh, bd, 5, [](TENO & t) { t.setup_on_host = true; });
    expect_same_tables(*reference, *batched, "batches of 64 cells");
    expect_same_tables(*reference, *host, "host setup");
}


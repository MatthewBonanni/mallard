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
#include <complex>
#include <limits>
#include <sstream>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "test_fixtures.h"
#include "verification.h"
#include "gmsh_fixtures.h"
#include "lattice_symbol.h"
#include "face_reconstruction.h"
#include "physics.h"
#include "solver.h"
#include "teno_sharing.h"
#include "curved.h"

namespace {

constexpr rtype GAMMA = 1.4_r;

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
    const bool smooth = std::string(mode) == SMOOTH;
    verification::expect_order({"reconstruction", "TENO" + std::to_string(order) + (smooth ? "" : ", troubled"),
                                verification::cells_of(mesh_type), "max face-value error",
                                {double(n), double(2 * n)},
                                {reconstruction_error(mesh_type, n, order, 0.0, mode),
                                 reconstruction_error(mesh_type, 2 * n, order, 0.0, mode)},
                                double(order), order - 0.5});
}

INSTANTIATE_TEST_SUITE_P(TENO, TENO3DOrder,
    ::testing::Values(OrderParam{"cartesian", 3, 6, SMOOTH}, OrderParam{"cartesian", 4, 6, SMOOTH},
                      OrderParam{"cartesian", 5, 6, SMOOTH}, OrderParam{"cartesian_tet", 3, 4, SMOOTH},
                      OrderParam{"cartesian_tet", 4, 4, SMOOTH}, OrderParam{"cartesian_tet", 5, 4, SMOOTH},
                      OrderParam{"cartesian", 5, 6, TROUBLED},
                      OrderParam{"cartesian_tet", 4, 4, TROUBLED}));

// The rest of the verification manual's matrix: run by tools/docs/verification.py. TENO6 on
// pyramids starts from 8 blocks: its stencils of about 209 cells span most of a 6^3 box, and 6 -> 12
// observes 5.26
INSTANTIATE_TEST_SUITE_P(DISABLED_Verification, TENO3DOrder,
    ::testing::Values(OrderParam{"cartesian", 6, 6, SMOOTH}, OrderParam{"cartesian_tet", 6, 4, SMOOTH},
                      OrderParam{"cartesian_prism", 3, 6, SMOOTH}, OrderParam{"cartesian_prism", 4, 6, SMOOTH},
                      OrderParam{"cartesian_prism", 5, 6, SMOOTH}, OrderParam{"cartesian_prism", 6, 6, SMOOTH},
                      OrderParam{"cartesian_pyramid", 3, 6, SMOOTH}, OrderParam{"cartesian_pyramid", 4, 6, SMOOTH},
                      OrderParam{"cartesian_pyramid", 5, 6, SMOOTH}, OrderParam{"cartesian_pyramid", 6, 8, SMOOTH},
                      OrderParam{"cartesian_mixed", 3, 6, SMOOTH}, OrderParam{"cartesian_mixed", 4, 6, SMOOTH},
                      OrderParam{"cartesian_mixed", 5, 6, SMOOTH}));

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
        in << "[[boundaries]]\nname = \"" << zone << "\"\ntype = \"outlet_extrapolation\"\n";
    }
    in << "[numerics]\nriemann_solver = \"HLLC\"\ntime_integrator = \"SSPRK3\"\ncheck_nan = true\n"
       << "[numerics.face_reconstruction]\ntype = \"" << recon << "\"\n" << (recon == "TENO" ? "order = 5\n" : "")
       << "[physics]\ntype = \"euler\"\ngamma = 1.4\np_ref = 1.0\nT_ref = 1.0\nrho_ref = 1.0\n"
       << "[output]\ncheck_interval = 1000000\n";
    Solver solver;
    solver.set_distributed(false);
    solver.init(parse_toml(in.str()));
    solver.run();
    solver.copy_device_to_host();
    auto m = solver.get_mesh();
    auto exact = cell_averages_3d(*m, [](double x, double y, double z, double * U) {
        U[0] = 1.0 + 0.3 * std::exp(-8.0 * ((x - 1.0) * (x - 1.0) + (y - 1.0) * (y - 1.0) + (z - 1.0) * (z - 1.0)));
        for (size_t i = 1; i < N_CONSERVATIVE; i++) U[i] = 0.0;
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
    verification::expect_order({"advection", "TENO5, SSPRK3", "hexahedra", "L1 density error, pulse", {12, 24},
                                {t1, t2}, 3.0, 3.0});
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
        for (size_t d = 0; d < N_DIM; d++) {
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

namespace {

/**
 * @brief Largest norm of exp(t J) over the Fourier modes of the linearized
 *        scheme J (lattice_symbol.h) on a periodic box of n^3 blocks of a 3D
 *        cartesian tiling: an eigenvalue of J with a positive real part shows
 *        as exponential growth in t.
 * @param t Time, in block sizes over the sound speed.
 */
double largest_linear_growth(const std::string & type, uint32_t n, int order, double t) {
    const LatticeJacobian jacobian = lattice_jacobian(lattice_mesh(type, n), n, order);
    double growth = 0.0;
    for (uint32_t kx = 0; kx < n; kx++) {
        for (uint32_t ky = 0; ky < n; ky++) {
            for (uint32_t kz = 0; kz < n; kz++) {
                // The mode -k has the complex conjugate symbol, so the same growth
                const uint32_t k = (kx * n + ky) * n + kz;
                if (k > (((n - kx) % n) * n + (n - ky) % n) * n + (n - kz) % n) continue;
                growth = std::max(growth, jacobian.growth({double(kx) / n, double(ky) / n, double(kz) / n}, t));
            }
        }
    }
    return growth;
}

} // namespace

class TENO3DPyramidLattice : public ::testing::TestWithParam<std::pair<int, uint32_t>> {};

TEST_P(TENO3DPyramidLattice, LinearizedSchemeHasNoGrowingMode) {
    // Pyramids at odd degrees took the stencils other cells take, within the
    // Lebesgue bound, and this lattice had growing modes (#236). Order 4 at
    // 77 entries still grows, at 9e-4 sound speeds over the block size, which
    // only boxes of 12^3 blocks resolve: a norm of 100 by t = 5000
    if (sizeof(rtype) == sizeof(float)) {
        GTEST_SKIP() << "finite differences of the right-hand side need double precision";
    }
    const auto [order, n] = GetParam();
    const double growth = largest_linear_growth("cartesian_pyramid", n, order, 5000.0);
    std::cout << "order " << order << ", " << n << "^3: largest |exp(t J)| at t = 5000 h/c: " << growth << std::endl;
    EXPECT_LT(growth, 20.0);
}

INSTANTIATE_TEST_SUITE_P(EvenOrders, TENO3DPyramidLattice,
                         ::testing::Values(std::make_pair(4, 12u), std::make_pair(6, 6u)));

TEST(TENO3DKuhnLattice, LongWavesOfTheLinearizedSchemeGrowAtTheRecordedRateAtOrderFour) {
    // Known failure (#1136): with the default 46-entry stencils the Riemann
    // solver's jump term feeds long acoustic waves on the Kuhn lattice, up to
    // about 0.11 cycles per block (largest at 0.08 along a face diagonal). The
    // box's own wavevectors miss the band, hence the interpolated ones. A fix
    // brings the rate under 1e-7, the level of the stable lattices
    // (TENOLattice in 2D); then lower both bounds to it
    if (sizeof(rtype) == sizeof(float)) {
        GTEST_SKIP() << "finite differences of the right-hand side need double precision";
    }
    const uint32_t n = 6;
    const double rate = lattice_jacobian(lattice_mesh("cartesian_tet", n), n, 4).largest_long_wave_rate(0.12, 0.04, 4e6);
    std::cout << "largest growth rate of the long waves: " << rate << " sound speeds over the block size" << std::endl;
    EXPECT_GT(rate, 0.9e-5);
    EXPECT_LT(rate, 1.5e-5);
}

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
        auto mesh = make_mesh_3d(type, 6, 6, 6, 1.0_r, 1.0_r, static_cast<rtype>(lz));
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
        auto mesh = make_mesh_3d(type, 6, 6, 6, 1.0_r, 1.0_r, 1.0_r / 6.25_r);
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
    add("cell shapes", teno.cell_shape);
    add("basis_mean", teno.basis_mean);
    add("moments", teno.moments);
    add("large sizes", teno.stencil_large_size);
    add("small sizes", teno.stencil_small_size);
    add("large slices", teno.stencil_large.slice_start);
    add("large slots", teno.stencil_large.slots);
    add("large far starts", teno.stencil_large.far_start);
    add("large far entries", teno.stencil_large.far);
    add("large images", teno.stencil_large.images);
    add("large pinv", teno.stencil_large.pinv);
    add("small slices", teno.stencil_small.slice_start);
    add("small slots", teno.stencil_small.slots);
    add("small far starts", teno.stencil_small.far_start);
    add("small far entries", teno.stencil_small.far);
    add("small images", teno.stencil_small.images);
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
            for (size_t d = 0; d < 3; d++) d2 += std::pow(double(mesh.h_node_coords(n, d)) - m[d], 2);
            if (d2 < best_d2) {
                best_d2 = d2;
                best = n;
            }
        }
        for (size_t d = 0; d < 3; d++) mesh.h_node_coords(best, d) += rtype(m[3 + d]);
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
    // An interior node, a node of the x = 0 symmetry wall moved within the
    // wall, which changes the mirror images of every stencil that reaches it,
    // and a node of the first cells, whose rows of basis means and moments
    // their cell shapes hold
    const auto [type, order] = GetParam();
    auto mesh = make_mesh_3d(type, 9, 8, 6);
    BoundaryData bd = make_uniform_boundaries(*mesh, BoundaryType::SYMMETRY, GAMMA);
    auto incremental = make_teno(mesh, bd, order);
    auto changed_only = make_teno(mesh, bd, order);
    const auto before = teno_tables(*incremental);
    const double h = 0.125;
    const std::vector<uint32_t> changed =
        move_nodes(*mesh, {{0.22, 0.375, 0.5, 0.06 * h, -0.04 * h, 0.05 * h}, {0.0, 0.25, 0.33, 0.0, 0.05 * h, -0.04 * h},
                           {1.0 / 9, 0.125, 1.0 / 6, 0.05 * h, 0.04 * h, -0.03 * h}});
    const std::vector<uint32_t> cells = incremental->cells_within_reach(changed);
    ASSERT_GT(cells.size(), changed.size());
    ASSERT_LT(cells.size(), mesh->n_cells);
    incremental->rebuild_cells(cells);
    auto full = make_teno(mesh, bd, order);
    expect_same_tables(*incremental, *full, "rebuilt within reach");
    // The move changed tables beyond the moved cells: rebuilding only those is not enough
    changed_only->rebuild_cells(changed);
    EXPECT_FALSE(teno_tables(*changed_only) == teno_tables(*full));
    // but a cell's scale, basis means and moments depend on its own geometry only
    const auto rows = [](const TENO & teno) {
        const auto all = teno_tables(teno);
        return std::vector<std::pair<std::string, std::vector<char>>>(all.begin(), all.begin() + 4);
    };
    EXPECT_TRUE(rows(*changed_only) == rows(*full));
    EXPECT_FALSE(before == teno_tables(*full));
}

INSTANTIATE_TEST_SUITE_P(TENO, TENO3DIncremental,
    ::testing::Values(IncrementalParam{"cartesian_prism", 4}, IncrementalParam{"cartesian_mixed", 4},
                      IncrementalParam{"cartesian_tet", 3}));

TEST(TENO3DSetup, EveryCellsMomentsAreThoseOfItsOwnBox) {
    // Stretched boxes all differ in size and aspect ratio; host backends compute the moments of
    // several cells at once, and 7 x 5 x 3 cells leave a last group short of them
    auto mesh = std::make_shared<Mesh>();
    mesh->init(parse_toml("[mesh]\ntype = \"cartesian\"\nNx = 7\nNy = 5\nNz = 3\nLx = 1.0\nLy = 0.8\nLz = 0.6\n"
                          "stretching = [1.5, 0.8, 1.2]\n"));
    mesh->copy_host_to_device();
    BoundaryData bd = make_uniform_boundaries(*mesh, BoundaryType::SYMMETRY, GAMMA);
    for (const int order : {3, 5}) {
        auto teno = make_teno_with(mesh, bd, order, [](TENO & t) { t.share_tables = false; });
        auto moments = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), teno->moments);
        ASSERT_EQ(moments.extent(0), mesh->n_cells);
        // Mean of ((x - x_c) / h)^a over the box's extent along x, by a, then b, then c
        const int nm = 2 * order - 3;
        for (uint32_t c = 0; c < mesh->n_cells; c++) {
            double lo[3] = {1e300, 1e300, 1e300}, hi[3] = {-1e300, -1e300, -1e300};
            for (uint32_t k = 0; k < mesh->h_n_nodes_of_cell(c); k++) {
                for (uint32_t d = 0; d < 3; d++) {
                    lo[d] = std::min(lo[d], double(mesh->h_node_coords(mesh->h_node_of_cell(c, k), d)));
                    hi[d] = std::max(hi[d], double(mesh->h_node_coords(mesh->h_node_of_cell(c, k), d)));
                }
            }
            const double h = std::cbrt(double(mesh->h_cell_volume(c)));
            auto mean = [&](const uint32_t d, const int a) {
                return a % 2 != 0 ? 0.0 : std::pow(0.5 * (hi[d] - lo[d]) / h, a) / (a + 1);
            };
            size_t k = 0;
            for (int a = 0; a < nm; a++) {
                for (int b = 0; a + b < nm; b++) {
                    for (int e = 0; a + b + e < nm; e++, k++) {
                        const double exact = mean(0, a) * mean(1, b) * mean(2, e);
                        EXPECT_NEAR(moments(c, k), exact, roundoff(1e-13)) << "order " << order << ", cell " << c
                                                                         << ", moment (" << a << ", " << b << ", " << e << ")";
                    }
                }
            }
            ASSERT_EQ(k, moments.extent(1));
        }
    }
}

TEST(TENO3DSetup, TablesDoNotDependOnTheBatchOrOnWhetherTheDeviceOrTheHostSetsThemUp) {
    // Mixed cells with walls: pyramids, tetrahedra and mirror images; small
    // batches split the cells differently, and the host setup is the fallback
    // for cells that do not fit the device's scratch. Orders 4 and 6 take the
    // pyramids' larger odd-degree stencils
    auto mesh = make_mesh_3d("cartesian_mixed", 4, 4, 4);
    BoundaryData bd = make_uniform_boundaries(*mesh, BoundaryType::SYMMETRY, GAMMA);
    for (const int order : {4, 5, 6}) {
        auto reference = make_teno(mesh, bd, order);
        auto batched = make_teno_with(mesh, bd, order, [](TENO & t) { t.setup_batch_cells = 64; });
        auto host = make_teno_with(mesh, bd, order, [](TENO & t) { t.setup_on_host = true; });
        expect_same_tables(*reference, *batched, "order " + std::to_string(order) + ", batches of 64 cells");
        expect_same_tables(*reference, *host, "order " + std::to_string(order) + ", host setup");
    }
}

TEST(TENO3DSetup, CentralStencilsThatReachACurvedWallFitTheWallDegree) {
    // The wall degree is 3, whose pseudo-inverse has no rows of degrees 4 and
    // 5. Pyramids at order 6 start from 208 entries, other cells from 110. The
    // Lebesgue search grows stencils past their start: here 87 at order 5 and
    // 28 at order 6 reached a curved-wall cell only there, at full degree (#1202).
    // They now stop short of it at full degree or fit the wall degree.
    auto mesh = make_mesh_3d("cartesian_pyramid", 6, 6, 6);
    mesh->apply_curved(curved::surfaces_of_mesh(*mesh, parse_toml("[mesh]\n[[mesh.curved]]\nzone = \"left\"\ntype = \"box\"\n"
                                                                  "filled = false\ncenter = [0.5, 0.5, 0.5]\nsize = [1.0, 1.0, 1.0]\n")));
    mesh->copy_host_to_device();
    BoundaryData bd = make_uniform_boundaries(*mesh, BoundaryType::SYMMETRY, GAMMA);
    std::vector<uint8_t> at_wall(mesh->n_cells, 0);
    for (uint32_t f = 0; f < mesh->n_faces; f++) {
        if (mesh->h_cells_of_face(f, 1) < 0 && mesh->curved_geometry->face_is_curved(f)) at_wall[uint32_t(mesh->h_cells_of_face(f, 0))] = 1;
    }
    const size_t wall_nk = teno::n_dof(3);
    for (const int order : {5, 6}) {
        const size_t nk = teno::n_dof(uint8_t(order - 1));
        uint32_t reaching = 0;
        // Unshared, the central slots are the candidates in order of distance
        auto teno = make_teno_with(mesh, bd, order, [](TENO & t) { t.share_tables = false; });
        teno->for_each_stencil([&](const uint32_t c, const TENO::CellStencils & st) {
            bool reaches = false, full = false;
            for (size_t s = 0; s < st.central; s++) {
                reaches |= st.cells[s] >= 0 && at_wall[uint32_t(st.cells[s])] != 0;
                for (size_t l = wall_nk; l < nk; l++) full |= st.pinv[s * nk + l] != 0.0_r;
            }
            reaching += reaches;
            EXPECT_FALSE(reaches && full) << "order " << order << ": cell " << c << " fits degree " << order - 1
                                          << " with a curved-wall cell in its central stencil of " << st.central;
        }, true);
        EXPECT_GT(reaching, 0u) << "order " << order;
    }
}

TEST(TENO3DSetup, CentralStencilsAloneAreTheFullSetupsBitForBit) {
    // Output of states read from files sets up only what cell_gradients and point_values read, from moments
    // of the central stencils' degree alone. With share_tables, cells share basis means on the means alone,
    // without moments. Orders 4 and 6 take the pyramids' larger odd-degree stencils
    auto mesh = make_mesh_3d("cartesian_mixed", 4, 4, 4);
    BoundaryData bd = make_uniform_boundaries(*mesh, BoundaryType::SYMMETRY, GAMMA);
    for (const int order : {4, 5, 6}) {
        for (const bool share : {true, false}) {
            const auto full = teno_tables(*make_teno_with(mesh, bd, order, [&](TENO & t) { t.share_tables = share; }));
            for (const bool on_host : {false, true}) {
                auto central = make_teno_with(mesh, bd, order, [&](TENO & t) {
                    t.share_tables = share;
                    t.sector_stencils = false;
                    t.setup_on_host = on_host;
                });
                const auto tables = teno_tables(*central);
                for (size_t k = 0; k < full.size(); k++) {
                    const std::string & name = full[k].first;
                    if (name == "scale" || name.starts_with("large") || (!share && (name == "basis_mean" || name == "cell shapes"))) {
                        EXPECT_TRUE(tables[k].second == full[k].second) << name << ", order " << order
                                                                        << (share ? ", shared" : "") << (on_host ? ", host setup" : "");
                    }
                }
                EXPECT_EQ(central->stencil_small.pinv.extent(0), 0u);
                EXPECT_FALSE(central->moments.is_allocated());
                EXPECT_EQ(central->troubled_coeffs.extent(0), 0u);
            }
        }
    }
}

namespace {

// The field along y = z = 0, on slabs and pencils one cell thick
void x_conservatives(double x, double, double, double * U) { smooth_conservatives(x, 0.0, 0.0, U); }

} // namespace

TEST(TENO3DSetup, SlabsAndPencilsOneCellThickConvergeAtDesignOrder) {
    // No neighbors across the thin directions: the fits take the monomials of the others (#706)
    for (const bool pencil : {false, true}) {
        auto mesh_of = [&](const uint32_t n) {
            const rtype h = 1.0_r / rtype(n);
            return pencil ? make_mesh_3d("cartesian", n, 1, 1, 1.0_r, h, h) : make_mesh_3d("cartesian", n, n, 1, 1.0_r, 1.0_r, h);
        };
        const char * what = pencil ? "max face-value error, one cell by one" : "max face-value error, one cell thick";
        verification::expect_order({"boundaries", "TENO5", verification::cells_of("cartesian"), what, {16, 32},
                                    {reconstruction_error(mesh_of(16), 5, 0.0, "", x_conservatives),
                                     reconstruction_error(mesh_of(32), 5, 0.0, "", x_conservatives)},
                                    5.0, 4.6});
        // The device sets them up in teams, the host one cell per thread
        auto mesh = mesh_of(16);
        BoundaryData bd = make_uniform_boundaries(*mesh, BoundaryType::SYMMETRY, GAMMA);
        auto host = make_teno_with(mesh, bd, 5, [](TENO & t) { t.setup_on_host = true; });
        expect_same_tables(*make_teno(mesh, bd, 5), *host, pencil ? "pencil" : "slab");
    }
}

TEST(TENO3DSetup, FacetedSpheresOfTetrahedraMirrorAcrossEachFacet) {
    // Straight-sided spherical walls of tetrahedra: every facet is a mirror
    // plane of its own, so searches near the walls meet dozens of planes. They
    // are set up, the same on the device as on the host
    const Shell s = tetrahedral_shell(4, 2);
    auto mesh = std::make_shared<Mesh>();
    mesh->init_from_connectivity(s.nodes, s.cells, s.faces);
    mesh->copy_host_to_device();
    BoundaryData bd = make_uniform_boundaries(*mesh, BoundaryType::SYMMETRY, GAMMA);
    auto device = make_teno(mesh, bd, 3);
    auto host = make_teno_with(mesh, bd, 3, [](TENO & t) { t.setup_on_host = true; });
    expect_same_tables(*device, *host, "host setup");
}

/**
 * @brief Checks that every mirror image in the central stencils of a mesh with
 *        symmetry walls is the image of a cell in front of its face, across
 *        that face (the cell's foot on the face's plane lies on the face), for
 *        a target cell in front of the face. Returns the number checked.
 */
uint32_t expect_images_across_their_faces(const std::shared_ptr<Mesh> & mesh, int order) {
    const Mesh & m = *mesh;
    BoundaryData bd = make_uniform_boundaries(m, BoundaryType::SYMMETRY, GAMMA);
    const TENO::Stencils stencils = make_teno(mesh, bd, order)->large_stencils();
    auto point = [](const auto & coords, uint32_t k) {
        return std::array<double, 3>{double(coords(k, 0)), double(coords(k, 1)), double(coords(k, 2))};
    };
    auto dot = [](const std::array<double, 3> & a, const std::array<double, 3> & b) {
        return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
    };
    auto sub = [](const std::array<double, 3> & a, const std::array<double, 3> & b) {
        return std::array<double, 3>{a[0] - b[0], a[1] - b[1], a[2] - b[2]};
    };
    uint32_t n_checked = 0;
    for (uint32_t c = 0; c < m.n_cells; c++) {
        for (uint64_t s = stencils.offsets[c]; s < stencils.offsets[c + 1]; s++) {
            const int32_t f = stencils.faces[s];
            if (f < 0) continue;
            const uint32_t uf = static_cast<uint32_t>(f);
            const std::array<double, 3> xf = point(m.h_face_coords, uf);
            std::array<double, 3> n = point(m.h_face_normals, uf);
            const double norm = std::sqrt(dot(n, n));
            for (double & x : n) x /= norm;
            if (dot(n, sub(xf, point(m.h_cell_coords, uint32_t(m.h_cells_of_face(uf, 0))))) < 0) {
                for (double & x : n) x = -x;
            }
            auto in_front = [&](uint32_t cell) { return dot(n, sub(point(m.h_cell_coords, cell), xf)) < 0; };
            const uint32_t image_of = static_cast<uint32_t>(stencils.cells[s]);
            EXPECT_TRUE(in_front(c)) << "cell " << c << " takes an image across face " << f << " from behind it";
            EXPECT_TRUE(in_front(image_of)) << "cell " << c << " takes the image of cell " << image_of << ", behind face " << f;
            const std::array<double, 3> x = point(m.h_cell_coords, image_of);
            const double depth = dot(n, sub(x, xf));
            const std::array<double, 3> foot = {x[0] - depth * n[0], x[1] - depth * n[1], x[2] - depth * n[2]};
            // The foot is on the same side of every edge of the face
            const uint32_t a0 = m.h_offsets_nodes_of_face(uf), nv = m.h_offsets_nodes_of_face(uf + 1) - a0;
            double lowest = std::numeric_limits<double>::max(), highest = std::numeric_limits<double>::lowest();
            for (uint32_t a = 0; a < nv; a++) {
                const auto pa = point(m.h_node_coords, m.h_nodes_of_face(a0 + a));
                const auto pb = point(m.h_node_coords, m.h_nodes_of_face(a0 + (a + 1) % nv));
                const auto e = sub(pb, pa), q = sub(foot, pa);
                const std::array<double, 3> cross = {e[1] * q[2] - e[2] * q[1], e[2] * q[0] - e[0] * q[2],
                                                     e[0] * q[1] - e[1] * q[0]};
                const double side = dot(cross, n) / dot(e, e);
                lowest = std::min(lowest, side);
                highest = std::max(highest, side);
            }
            EXPECT_TRUE(lowest >= -1e-9 || highest <= 1e-9)
                << "cell " << c << " takes the image of cell " << image_of << " across the extension of face " << f;
            n_checked++;
        }
    }
    return n_checked;
}

/** @brief Hexahedra of a quadrilateral mesh of the xy plane extruded over nz layers of height dz. */
std::shared_ptr<Mesh> extruded(const std::vector<std::array<double, 2>> & nodes, const std::vector<std::array<uint32_t, 4>> & quads,
                               uint32_t nz, double dz) {
    std::vector<std::array<rtype, N_DIM>> nodes_3d;
    for (uint32_t k = 0; k <= nz; k++) {
        for (const auto & p : nodes) nodes_3d.push_back({rtype(p[0]), rtype(p[1]), rtype(k * dz)});
    }
    const uint32_t m = static_cast<uint32_t>(nodes.size());
    std::vector<std::vector<uint32_t>> cells;
    for (uint32_t k = 0; k < nz; k++) {
        for (const auto & q : quads) {
            cells.push_back({k * m + q[0], k * m + q[1], k * m + q[2], k * m + q[3],
                             (k + 1) * m + q[0], (k + 1) * m + q[1], (k + 1) * m + q[2], (k + 1) * m + q[3]});
        }
    }
    auto mesh = std::make_shared<Mesh>();
    mesh->init_from_connectivity(nodes_3d, cells, {});
    mesh->copy_host_to_device();
    return mesh;
}

TEST(TENO3DSetup, MirrorImagesOfFacetedCurvedWallsAreAcrossTheFacetUnderTheirCell) {
    // A polygonal annulus of thin hexahedra (aspect ratio 10 at the inner
    // wall), two layers deep: the plane of a facet, extended, passes a cell
    // height inside the next column of cells, so reflections across it
    // misplace the flow there (#382, #443)
    const uint32_t n_theta = 32, n_r = 6;
    const double r0 = 0.5, dr0 = 2.0 * M_PI * r0 / n_theta / 10.0, q = 1.5;
    std::vector<std::array<double, 2>> nodes;
    double r = r0, dr = dr0;
    for (uint32_t j = 0; j <= n_r; j++, r += dr, dr *= q) {
        for (uint32_t i = 0; i < n_theta; i++) {
            const double t = 2.0 * M_PI * i / n_theta;
            nodes.push_back({r * std::cos(t), r * std::sin(t)});
        }
    }
    std::vector<std::array<uint32_t, 4>> quads;
    for (uint32_t j = 0; j < n_r; j++) {
        for (uint32_t i = 0; i < n_theta; i++) {
            const uint32_t i1 = (i + 1) % n_theta;
            quads.push_back({j * n_theta + i, j * n_theta + i1, (j + 1) * n_theta + i1, (j + 1) * n_theta + i});
        }
    }
    const auto annulus = extruded(nodes, quads, 2, 2.0 * M_PI * r0 / n_theta);
    for (int order : {3, 5}) EXPECT_GT(expect_images_across_their_faces(annulus, order), 0u) << "order " << order;
    // Faceted spheres of tetrahedra
    const Shell s = tetrahedral_shell(4, 2);
    auto shell_mesh = std::make_shared<Mesh>();
    shell_mesh->init_from_connectivity(s.nodes, s.cells, s.faces);
    shell_mesh->copy_host_to_device();
    EXPECT_GT(expect_images_across_their_faces(shell_mesh, 3), 0u) << "tetrahedral shell";
}

TEST(TENO3DSetup, MirrorImagesAcrossAThinBodyStayOnTheTargetsSide) {
    // A box with a slot-shaped hole three quarters of a cell thick through it
    // (as at a closed trailing edge): a cell below it mirrored across its lower
    // side lands inside the hole, a quarter cell from its upper side, next to
    // the cells above it (#382, #443)
    const uint32_t n = 8;
    const double h = 1.0 / n;
    std::vector<std::array<double, 2>> nodes;
    for (uint32_t j = 0; j <= n; j++) {
        const double y = j <= 4 ? j * h : (j - 0.25) * h;
        for (uint32_t i = 0; i <= n; i++) nodes.push_back({i * h, y});
    }
    std::vector<std::array<uint32_t, 4>> quads;
    for (uint32_t j = 0; j < n; j++) {
        for (uint32_t i = 0; i < n; i++) {
            if (j == 4 && i >= 2 && i < 6) continue;
            const uint32_t k = j * (n + 1) + i;
            quads.push_back({k, k + 1, k + n + 2, k + n + 1});
        }
    }
    const auto mesh = extruded(nodes, quads, 2, h);
    for (int order : {3, 5}) EXPECT_GT(expect_images_across_their_faces(mesh, order), 0u) << "order " << order;
}

TEST(TENO3DTroubled, FaceValuesDoNotDependOnHowManyTroubledCellsARoundHolds) {
    // A blast (density and pressure jumps on a sphere) on mixed cells with walls:
    // rounds after the first recompute the central coefficients that the smooth
    // pass could not keep, with its arithmetic. On GPUs from order 5 both are
    // team kernels, and only the smooth pass stages its stencil entries in team
    // scratch
    auto mesh = make_mesh_3d("cartesian_mixed", 6, 6, 6);
    BoundaryData bd = make_uniform_boundaries(*mesh, BoundaryType::SYMMETRY, GAMMA);
    Kokkos::View<rtype *[N_CONSERVATIVE]> W("W", mesh->n_cells);
    auto h_W = Kokkos::create_mirror_view(W);
    for (uint32_t c = 0; c < mesh->n_cells; c++) {
        double r2 = 0.0;
        for (int d = 0; d < 3; d++) r2 += std::pow(double(mesh->h_cell_coords(c, d)) - 0.3, 2);
        const bool inside = r2 < 0.16;
        h_W(c, 0) = inside ? 1.0_r : 0.125_r;
        for (int d = 0; d < 3; d++) h_W(c, 1 + d) = static_cast<rtype>(0.1 * double(mesh->h_cell_coords(c, d)));
        h_W(c, 4) = inside ? 1.0_r : 0.1_r;
    }
    Kokkos::deep_copy(W, h_W);
    for (const int order : {3, 5, 6}) {
        auto face_values = [&](const rtype capacity) {
            auto teno = make_teno_with(mesh, bd, order, [&](TENO & t) { t.troubled_capacity = capacity; });
            Kokkos::View<rtype **[2][N_CONSERVATIVE]> face_W("face_W", mesh->n_faces, teno->n_face_quadrature_points());
            teno->calc_face_values(W, face_W);
            auto sigma = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), teno->troubled);
            uint32_t n_troubled = 0;
            for (uint32_t c = 0; c < mesh->n_cells; c++) n_troubled += sigma(c) >= teno->sigma_threshold;
            if (capacity < 1.0_r) {
                EXPECT_GT(n_troubled, 20 * teno->troubled_coeffs.extent(0)) << "order " << order;
            }
            auto h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), face_W);
            const char * p = reinterpret_cast<const char *>(h.data());
            return std::vector<char>(p, p + h.span() * sizeof(rtype));
        };
        const auto rounds = face_values(0.001_r);
        EXPECT_TRUE(rounds == face_values(1.0_r)) << "order " << order;
    }
}

TEST(TENO3DSetup, SinglePrecisionTablesHalveThePseudoInversesAndKeepTheAccuracy) {
    if (sizeof(rtype) == sizeof(float)) {
        GTEST_SKIP() << "the tables are in single precision anyway";
    }
    auto mesh = make_mesh_3d("cartesian_tet", 6, 6, 6);
    BoundaryData bd = make_uniform_boundaries(*mesh, BoundaryType::SYMMETRY, GAMMA);
    const std::string single = "single_precision_tables = true\n";
    auto in_double = make_teno(mesh, bd, 5);
    auto in_single = make_teno(mesh, bd, 5, single);
    EXPECT_EQ(in_single->stencil_large.pinv.span(), (in_double->stencil_large.pinv.span() + 1) / 2);
    EXPECT_EQ(in_single->stencil_small.pinv.span(), (in_double->stencil_small.pinv.span() + 1) / 2);
    for (const char * mode : {SMOOTH, TROUBLED}) {
        const double e_full = reconstruction_error(mesh, 5, 0.0, mode);
        const double e_half = reconstruction_error(mesh, 5, 0.0, std::string(mode) + single);
        EXPECT_NE(e_half, e_full) << mode;
        EXPECT_NEAR(e_half, e_full, 1e-4 * e_full) << mode;
    }
}


TEST(TENO3DSetup, CellsOfAStencilShapeShareOneTableThatChangesFaceValuesOnlyByRoundOff) {
    // A blast on hexahedra with walls (mirror images, smooth and troubled cells):
    // interior cells share their pseudo-inverses, cells along the walls keep
    // their own, and every cell shares one row of basis means and moments
    auto mesh = make_mesh_3d("cartesian", 8, 8, 8);
    BoundaryData bd = make_uniform_boundaries(*mesh, BoundaryType::SYMMETRY, GAMMA);
    Kokkos::View<rtype *[N_CONSERVATIVE]> W("W", mesh->n_cells);
    auto h_W = Kokkos::create_mirror_view(W);
    for (uint32_t c = 0; c < mesh->n_cells; c++) {
        double r2 = 0.0;
        for (int d = 0; d < 3; d++) r2 += std::pow(double(mesh->h_cell_coords(c, d)) - 0.4, 2);
        const bool inside = r2 < 0.09;
        h_W(c, 0) = inside ? 1.0_r : 0.125_r;
        for (int d = 0; d < 3; d++) h_W(c, 1 + d) = static_cast<rtype>(0.1 * std::sin(3.0 * double(mesh->h_cell_coords(c, d))));
        h_W(c, 4) = inside ? 1.0_r : 0.1_r;
    }
    Kokkos::deep_copy(W, h_W);
    for (const int order : {3, 5}) {
        auto face_values = [&](const bool share, const uint8_t shift) {
            auto teno = std::make_unique<TENO>();
            teno->set_mesh(mesh);
            teno->set_boundaries(bd);
            teno->slice_shift = shift;
            teno->share_tables = share;
            teno->init(parse_toml("type = \"TENO\"\norder = " + std::to_string(order) + "\n"));
            Kokkos::View<rtype **[2][N_CONSERVATIVE]> face_W("face_W", mesh->n_faces, teno->n_face_quadrature_points());
            teno->calc_face_values(W, face_W);
            auto h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), face_W);
            return std::make_tuple(std::vector<rtype>(h.data(), h.data() + h.span()),
                                   teno->stencil_large.pinv.span() + teno->stencil_small.pinv.span(),
                                   teno->basis_mean.extent(0));
        };
        const auto [own, own_entries, own_rows] = face_values(false, teno::SLICE_SHIFT);
        const auto [shared, shared_entries, shared_rows] = face_values(true, teno::SLICE_SHIFT);
        EXPECT_LT(4 * shared_entries, own_entries) << "order " << order;
        // Every cell of a lattice has the same basis means and moments
        EXPECT_EQ(own_rows, mesh->n_cells) << "order " << order;
        EXPECT_EQ(shared_rows, 1u) << "order " << order;
        double largest = 0.0, difference = 0.0;
        for (size_t k = 0; k < own.size(); k++) {
            largest = std::max(largest, std::abs(double(own[k])));
            difference = std::max(difference, std::abs(double(shared[k]) - double(own[k])));
        }
        EXPECT_LT(difference, 1e3 * double(std::numeric_limits<rtype>::epsilon()) * largest) << "order " << order;
        // A shape's table is its cell's with the lowest global id, however many cells a slice packs
        for (const uint8_t shift : {uint8_t(0), uint8_t(5)}) {
            EXPECT_TRUE(std::get<0>(face_values(true, shift)) == shared) << "order " << order << ", slices of " << (1 << shift);
        }
    }
}

TEST(TENO3DSetup, BlockCellsReadTheirStencilsByOffsetsAndGiveTheUnstructuredPathsFaceValuesBitwise) {
    // A blast in a box with walls: smooth and troubled cells, cells with mirror images along the walls
    auto mesh = make_mesh_3d("cartesian", 12, 12, 12);
    BoundaryData bd = make_uniform_boundaries(*mesh, BoundaryType::SYMMETRY, GAMMA);
    Kokkos::View<rtype *[N_CONSERVATIVE]> W("W", mesh->n_cells);
    auto h_W = Kokkos::create_mirror_view(W);
    for (uint32_t c = 0; c < mesh->n_cells; c++) {
        double r2 = 0.0;
        for (int d = 0; d < 3; d++) r2 += std::pow(double(mesh->h_cell_coords(c, d)) - 0.45, 2);
        const bool inside = r2 < 0.06;
        h_W(c, 0) = inside ? 1.0_r : 0.125_r;
        for (int d = 0; d < 3; d++) h_W(c, 1 + d) = static_cast<rtype>(0.1 * std::sin(3.0 * double(mesh->h_cell_coords(c, d))));
        h_W(c, 4) = inside ? 1.0_r : 0.1_r;
    }
    Kokkos::deep_copy(W, h_W);
    for (const int order : {3, 5}) {
        for (const uint8_t shift : {uint8_t(0), uint8_t(5)}) {
            auto results = [&](const bool block_path) {
                auto teno = std::make_unique<TENO>();
                teno->set_mesh(mesh);
                teno->set_boundaries(bd);
                teno->slice_shift = shift;
                teno->block_path = block_path;
                teno->init(parse_toml("type = \"TENO\"\norder = " + std::to_string(order) + "\n"));
                teno->smooth_gradients = Kokkos::View<rtype *[N_CONSERVATIVE][N_DIM]>("gradients", mesh->n_cells);
                Kokkos::View<rtype **[2][N_CONSERVATIVE]> face_W("face_W", mesh->n_faces, teno->n_face_quadrature_points());
                teno->calc_face_values(W, face_W);
                auto h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), face_W);
                auto g = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), teno->smooth_gradients);
                std::vector<rtype> out(h.data(), h.data() + h.span());
                out.insert(out.end(), g.data(), g.data() + g.span());
                return std::make_pair(out, teno->n_block_cells);
            };
            const auto [unstructured, none] = results(false);
            const auto [block, n_block] = results(true);
            const std::string at = "order " + std::to_string(order) + ", slices of " + std::to_string(1 << shift);
            EXPECT_EQ(none, 0u) << at;
            // Every cell whose stencil stays clear of the walls; those near them read mirror images
            const uint32_t depth = order == 3 ? 1 : 2;
            EXPECT_GE(n_block, uint64_t(std::pow(12 - 2 * depth, 3))) << at;
            EXPECT_LT(n_block, uint64_t(mesh->n_cells)) << at;
            EXPECT_TRUE(block == unstructured) << at;
        }
    }
}

TEST(TENO3DSetup, CellsAlongPeriodicSeamsTakeTheBlockPathAndGiveTheUnstructuredPathsFaceValuesBitwise) {
    // Periodic along x and z, symmetry walls on y: the cells along the seams read their stencils by offsets
    // that wrap the block's index, and only the cells whose stencils reach a wall read mirror images
    auto mesh = std::make_shared<Mesh>();
    mesh->init(parse_toml("[mesh]\ntype = \"cartesian\"\nNx = 12\nNy = 10\nNz = 11\n"
                          "[[periodic]]\nzones = [\"left\", \"right\"]\n[[periodic]]\nzones = [\"back\", \"front\"]\n"));
    mesh->copy_host_to_device();
    BoundaryData bd = make_uniform_boundaries(*mesh, BoundaryType::SYMMETRY, GAMMA);
    Kokkos::View<rtype *[N_CONSERVATIVE]> W("W", mesh->n_cells);
    auto h_W = Kokkos::create_mirror_view(W);
    for (uint32_t c = 0; c < mesh->n_cells; c++) {
        const double x = double(mesh->h_cell_coords(c, 0)), y = double(mesh->h_cell_coords(c, 1)),
                     z = double(mesh->h_cell_coords(c, 2));
        // A blast across the corner of the seams, so that seam cells are troubled too
        const bool inside = std::pow(x - 0.1, 2) + std::pow(y - 0.5, 2) + std::pow(z - 0.9, 2) < 0.06;
        h_W(c, 0) = inside ? 1.0_r : 0.125_r;
        for (int d = 0; d < 3; d++) h_W(c, 1 + d) = static_cast<rtype>(0.1 * std::sin(6.283185307179586 * (x + 2 * y - z + d)));
        h_W(c, 4) = inside ? 1.0_r : 0.1_r;
    }
    Kokkos::deep_copy(W, h_W);
    for (const int order : {3, 5}) {
        auto results = [&](const bool block_path) {
            auto teno = std::make_unique<TENO>();
            teno->set_mesh(mesh);
            teno->set_boundaries(bd);
            teno->block_path = block_path;
            teno->init(parse_toml("type = \"TENO\"\norder = " + std::to_string(order) + "\n"));
            teno->smooth_gradients = Kokkos::View<rtype *[N_CONSERVATIVE][N_DIM]>("gradients", mesh->n_cells);
            Kokkos::View<rtype **[2][N_CONSERVATIVE]> face_W("face_W", mesh->n_faces, teno->n_face_quadrature_points());
            teno->calc_face_values(W, face_W);
            auto h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), face_W);
            auto g = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), teno->smooth_gradients);
            std::vector<rtype> out(h.data(), h.data() + h.span());
            out.insert(out.end(), g.data(), g.data() + g.span());
            return std::make_pair(out, teno->n_block_cells);
        };
        const auto [unstructured, none] = results(false);
        const auto [block, n_block] = results(true);
        const std::string at = "order " + std::to_string(order);
        EXPECT_EQ(none, 0u) << at;
        const uint32_t depth = order == 3 ? 1 : 2;
        EXPECT_EQ(n_block, uint64_t(12 * (10 - 2 * depth) * 11)) << at;
        EXPECT_TRUE(block == unstructured) << at;
    }
}

TEST(TENO3DSetup, AShapeKeyedByItsCallerTakesTheTableOfItsLowestGlobalIdAndStaysApartFromEqualTables) {
    // Two slots of width 2: two tables, bitwise different, under one key
    const std::vector<rtype> a = {1.0_r, 0.5_r, 0.25_r, 0.125_r};
    const std::vector<rtype> b = {1.0_r + 4 * std::numeric_limits<rtype>::epsilon(), 0.5_r, 0.25_r, 0.125_r};
    const teno::CanonicalTable ta(a.data(), 2, {2}), tb(b.data(), 2, {2});
    teno::TableStore store(2);
    const uint32_t k = store.add_keyed(7, ta, a.data(), 9);
    EXPECT_EQ(store.find_keyed(7), k);
    EXPECT_EQ(store.find_keyed(8), teno::TableStore<>::NONE);
    EXPECT_EQ(store.add_keyed(7, tb, b.data(), 3), k);
    EXPECT_TRUE(store.values(k) == b);
    EXPECT_EQ(store.owner(k), 3u);
    EXPECT_EQ(store.add_keyed(7, ta, a.data(), 5), k);
    EXPECT_TRUE(store.values(k) == b);
    EXPECT_NE(store.add_keyed(8, ta, a.data(), 1), k);
    const uint64_t gid = 2;
    const rtype * values = a.data();
    const auto shape = store.add(1, &ta, [&](uint32_t) { return values; }, &gid);
    EXPECT_EQ(store.size(), 3u);
    EXPECT_NE(shape[0], k);
}

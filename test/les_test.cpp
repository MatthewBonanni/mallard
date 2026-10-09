/**
 * @file les_test.cpp
 * @author Matthew Bonanni (mbonanni001@gmail.com)
 * @brief Subgrid-scale models of large-eddy simulation and their coupling to the solver.
 * @version 0.1
 * @date 2026-10-05
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 *
 */

#include <gtest/gtest.h>
#include <Kokkos_Core.hpp>

#include <array>
#include <cmath>
#include <filesystem>
#include <functional>
#include <iomanip>
#include <sstream>
#include <string>

#include "exact_riemann.h"
#include "input.h"
#include "les.h"
#include "solver.h"
#include "test_fixtures.h"

namespace {

using Gradient = std::array<std::array<double, 3>, 3>;

const SGSModel MODELS[] = {SGSModel::SMAGORINSKY, SGSModel::WALE, SGSModel::VREMAN, SGSModel::SIGMA};

double D(const SGSModel model, const Gradient & g) {
    double a[3][3];
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) a[i][j] = g[i][j];
    }
    return LES::operator_of(model, a);
}

Gradient zero() { return Gradient{}; }

Gradient matmul(const Gradient & a, const Gradient & b) {
    Gradient c{};
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            for (int k = 0; k < 3; k++) c[i][j] += a[i][k] * b[k][j];
        }
    }
    return c;
}

Gradient transpose(const Gradient & a) {
    Gradient t{};
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) t[i][j] = a[j][i];
    }
    return t;
}

const Gradient GENERIC = {{{-0.7819, -0.2572, 0.0081}, {-0.2756, 1.2941, 1.0067}, {-2.7112, -1.8890, -0.1748}}};

/** @brief Eigenvalues of a symmetric 3x3 matrix by cyclic Jacobi rotations, descending. */
std::array<double, 3> jacobi_eigenvalues(Gradient a) {
    for (int sweep = 0; sweep < 50; sweep++) {
        for (int p = 0; p < 2; p++) {
            for (int q = p + 1; q < 3; q++) {
                if (std::abs(a[p][q]) < 1e-300) continue;
                const double theta = 0.5 * std::atan2(2.0 * a[p][q], a[q][q] - a[p][p]);
                const double c = std::cos(theta), s = std::sin(theta);
                Gradient r = {{{1, 0, 0}, {0, 1, 0}, {0, 0, 1}}};
                r[p][p] = c;
                r[q][q] = c;
                r[p][q] = s;
                r[q][p] = -s;
                a = matmul(transpose(r), matmul(a, r));
            }
        }
    }
    std::array<double, 3> e = {a[0][0], a[1][1], a[2][2]};
    std::sort(e.begin(), e.end(), std::greater<double>());
    return e;
}

} // namespace

TEST(LESModels, PureShearHasNoEddyViscosityExceptSmagorinsky) {
    const double s = 3.7;
    Gradient g = zero();
    g[0][1] = s;
    EXPECT_NEAR(D(SGSModel::SMAGORINSKY, g), s, 1e-14 * s);
    for (const SGSModel model : {SGSModel::WALE, SGSModel::VREMAN, SGSModel::SIGMA}) {
        EXPECT_LT(D(model, g), 1e-12 * s) << int(model);
    }
}

TEST(LESModels, SolidRotationHasNoEddyViscosityForSigmaAndSmagorinskyOnly) {
    const double w = 2.3;
    Gradient g = zero();
    g[0][1] = -w;
    g[1][0] = w;
    EXPECT_EQ(D(SGSModel::SMAGORINSKY, g), 0.0);
    EXPECT_LT(D(SGSModel::SIGMA, g), 1e-12 * w);
    // S = 0 and S^d = w^2 diag(-1/3, -1/3, 2/3): (S^d:S^d)^(1/4) = (2/3)^(1/4) w
    EXPECT_NEAR(D(SGSModel::WALE, g), std::pow(2.0 / 3.0, 0.25) * w, 1e-13 * w);
    // b = w^2 diag(1, 1, 0), B = w^4, a:a = 2 w^2
    EXPECT_NEAR(D(SGSModel::VREMAN, g), w / std::sqrt(2.0), 1e-13 * w);
}

TEST(LESModels, SigmaVanishesForExpansionsAndTwoComponentFlows) {
    Gradient isotropic = zero(), axisymmetric = zero(), planar = GENERIC;
    for (int i = 0; i < 3; i++) isotropic[i][i] = 1.5;
    axisymmetric[0][0] = axisymmetric[1][1] = -0.8;
    axisymmetric[2][2] = 1.6;
    for (int i = 0; i < 3; i++) planar[2][i] = planar[i][2] = 0.0;
    const double scale = D(SGSModel::SMAGORINSKY, GENERIC);
    EXPECT_LT(D(SGSModel::SIGMA, isotropic), 1e-12 * scale);
    EXPECT_LT(D(SGSModel::SIGMA, planar), 1e-12 * scale);
    // Eigenvalues of a repeated pair from the characteristic polynomial are accurate to sqrt(epsilon)
    EXPECT_LT(D(SGSModel::SIGMA, axisymmetric), 1e-7 * scale);
    // WALE vanishes only for the isotropic one (g^2 is then isotropic too)
    for (const Gradient & g : {axisymmetric, planar}) EXPECT_GT(D(SGSModel::WALE, g), 1e-3 * scale);
    for (const Gradient & g : {isotropic, axisymmetric, planar}) EXPECT_GT(D(SGSModel::VREMAN, g), 1e-3 * scale);
}

TEST(LESModels, SigmaUsesTheSingularValuesOfTheGradient) {
    Gradient g = GENERIC;
    // Add a nearly repeated pair of singular values too
    Gradient near = zero();
    near[0][0] = 2.0;
    near[1][1] = 2.0 + 1e-4;
    near[2][2] = 0.5;
    near[0][1] = 0.3;
    for (const Gradient & a : {g, near}) {
        const auto lambda = jacobi_eigenvalues(matmul(transpose(a), a));
        const double s1 = std::sqrt(lambda[0]), s2 = std::sqrt(lambda[1]), s3 = std::sqrt(lambda[2]);
        const double expected = s3 * (s1 - s2) * (s2 - s3) / (s1 * s1);
        EXPECT_NEAR(D(SGSModel::SIGMA, a), expected, 1e-10 * s1);
    }
    EXPECT_GT(D(SGSModel::SIGMA, g), 0.0);
}

TEST(LESModels, EddyViscosityIsRotationAndFrameInvariant) {
    // R = exp of a skew matrix, through Rodrigues' formula
    const double axis[3] = {0.48, -0.64, 0.6}, angle = 1.1;
    Gradient K = {{{0, -axis[2], axis[1]}, {axis[2], 0, -axis[0]}, {-axis[1], axis[0], 0}}};
    Gradient R = matmul(K, K);
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) R[i][j] = (i == j) + std::sin(angle) * K[i][j] + (1 - std::cos(angle)) * R[i][j];
    }
    const Gradient rotated = matmul(R, matmul(GENERIC, transpose(R)));
    for (const SGSModel model : MODELS) {
        EXPECT_NEAR(D(model, rotated), D(model, GENERIC), 1e-12 * D(SGSModel::SMAGORINSKY, GENERIC)) << int(model);
    }
}

TEST(LESModels, EddyViscosityScalesWithTheConstantGradientAndFilterWidth) {
    double g[3][3], g2[3][3];
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            g[i][j] = GENERIC[i][j];
            g2[i][j] = 2.0 * GENERIC[i][j];
        }
    }
    for (const SGSModel model : MODELS) {
        LES les;
        les.model = model;
        les.C = 0.3_r;
        const double base = static_cast<double>(les.nu_t(g, 0.1_r));
        const double coefficient = model == SGSModel::VREMAN ? 0.3 : 0.09;
        EXPECT_NEAR(base, coefficient * 0.01 * LES::operator_of(model, g), precision_tol<double>(1e-14, 1e-6) * base);
        EXPECT_NEAR(static_cast<double>(les.nu_t(g2, 0.1_r)), 2.0 * base, precision_tol<double>(1e-12, 1e-5) * base);
        EXPECT_NEAR(static_cast<double>(les.nu_t(g, 0.2_r)), 4.0 * base, precision_tol<double>(1e-12, 1e-5) * base);
    }
}

TEST(LESModels, NearWallEddyViscosityGrowsAsTheCubeOfTheWallDistanceForWALEAndSigma) {
    // Incompressible near-wall expansion: u, w ~ y, v ~ y^2, so du/dy and dw/dy
    // are O(1), dv/dx and dv/dz O(y^2), everything else O(y)
    Gradient G0 = zero(), G1 = {{{0.4, -0.2, 0.3}, {0.0, -0.5, 0.0}, {0.1, 0.6, 0.1}}},
             G2 = {{{0.2, 0.1, -0.3}, {0.5, 0.2, -0.4}, {0.3, -0.2, 0.1}}};
    G0[0][1] = 1.3;
    G0[2][1] = -0.7;
    auto at = [&](double y) {
        Gradient g;
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) g[i][j] = G0[i][j] + y * G1[i][j] + y * y * G2[i][j];
        }
        return g;
    };
    auto exponent = [&](SGSModel m) { return std::log2(D(m, at(2e-3)) / D(m, at(1e-3))); };
    EXPECT_NEAR(exponent(SGSModel::SIGMA), 3.0, 0.01);
    EXPECT_NEAR(exponent(SGSModel::WALE), 3.0, 0.01);
    EXPECT_NEAR(exponent(SGSModel::VREMAN), 1.0, 0.01);
    EXPECT_NEAR(exponent(SGSModel::SMAGORINSKY), 0.0, 0.01);
}

namespace {

const double TWO_PI = 6.283185307179586;

/** @brief Periodic box [0, 1]^d with a smooth vortical field; viscous perfect gas with R = 1. */
std::string periodic_box(const std::string & mesh, uint32_t n, const std::string & les, const std::string & extra = "",
                         const std::string & run = "n_steps = 10\ncfl = 0.5\n", double mu = 1e-3) {
    std::ostringstream s;
    s << "[run]\n" << run << "[mesh]\ntype = \"" << mesh << "\"\nNx = " << n << "\nNy = " << n << "\nLx = 1.0\nLy = 1.0\n";
    if constexpr (N_DIM == 3) {
        s << "Nz = " << n << "\nLz = 1.0\nperiodic = [\"x\", \"y\", \"z\"]\n";
        s << "[initialize]\ntype = \"analytical\"\nrho = \"1.0 + 0.1 * sin(6.283185307179586 * (x + y + z))\"\n"
             "p = \"10.0\"\nu = [\"sin(6.283185307179586 * x) * cos(6.283185307179586 * y) * cos(6.283185307179586 * z)\", "
             "\"-cos(6.283185307179586 * x) * sin(6.283185307179586 * y) * cos(6.283185307179586 * z)\", "
             "\"0.5 * sin(6.283185307179586 * x)\"]\n";
    } else {
        s << "periodic = [\"x\", \"y\"]\n";
        s << "[initialize]\ntype = \"analytical\"\nrho = \"1.0 + 0.1 * sin(6.283185307179586 * (x + y))\"\np = \"10.0\"\n"
             "u = [\"sin(6.283185307179586 * x) * cos(6.283185307179586 * y) + 0.3 * sin(12.566370614359172 * y + 1)\", "
             "\"-cos(6.283185307179586 * x) * sin(6.283185307179586 * y) + 0.2 * cos(12.566370614359172 * x)\"]\n";
    }
    s << "[numerics]\nriemann_solver = \"HLLC\"\n[numerics.face_reconstruction]\ntype = \"MUSCL\"\nlimiter = \"none\"\n"
      << "[physics]\ntype = \"navier_stokes\"\ngamma = 1.4\np_ref = 1.0\nT_ref = 1.0\nrho_ref = 1.0\nmu = " << mu
      << "\nPr = 0.72\n[output]\ncheck_interval = 1000000\n"
      << les << extra;
    return s.str();
}

/** @brief Velocity gradient g[i][j] = d u_i / d x_j of the periodic_box field at x. */
Gradient box_gradient(const double * x) {
    const double k = TWO_PI;
    Gradient g{};
    if constexpr (N_DIM == 3) {
        const double sx = std::sin(k * x[0]), cx = std::cos(k * x[0]), sy = std::sin(k * x[1]), cy = std::cos(k * x[1]),
                     sz = std::sin(k * x[2]), cz = std::cos(k * x[2]);
        g[0] = {k * cx * cy * cz, -k * sx * sy * cz, -k * sx * cy * sz};
        g[1] = {k * sx * sy * cz, -k * cx * cy * cz, k * cx * sy * sz};
        g[2] = {0.5 * k * std::cos(k * x[0]), 0.0, 0.0};
    } else {
        const double sx = std::sin(k * x[0]), cx = std::cos(k * x[0]), sy = std::sin(k * x[1]), cy = std::cos(k * x[1]);
        g[0] = {k * cx * cy, -k * sx * sy + 0.3 * 2 * k * std::cos(2 * k * x[1] + 1), 0.0};
        g[1] = {k * sx * sy - 0.2 * 2 * k * std::sin(2 * k * x[0]), -k * cx * cy, 0.0};
    }
    return g;
}

/** @brief tau : grad u / mu for the deviatoric stress tau = mu (g + g^T - 2/3 div I). */
double dissipation_per_viscosity(const Gradient & g) {
    double div = g[0][0] + g[1][1] + g[2][2], sum = 0.0;
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) sum += (g[i][j] + g[j][i] - (i == j ? 2.0 / 3.0 * div : 0.0)) * g[i][j];
    }
    return sum;
}

const std::string MESH = N_DIM == 2 ? "cartesian_tri" : "cartesian_tet";
const std::string LES_DEFAULT = N_DIM == 2 ? "[les]\nmodel = \"wale\"\n" : "[les]\nmodel = \"sigma\"\n";

std::unique_ptr<Solver> make_solver(const std::string & input) {
    auto solver = std::make_unique<Solver>();
    solver->init(parse_toml(input));
    return solver;
}

double kinetic_energy(const Solver & solver) {
    const auto mesh = solver.get_mesh();
    double sum = 0.0;
    for (uint32_t c = 0; c < mesh->n_owned(); c++) {
        double m2 = 0.0;
        FOR_I_DIM m2 += double(solver.h_conservatives(c, 1 + i)) * double(solver.h_conservatives(c, 1 + i));
        sum += 0.5 * m2 / double(solver.h_conservatives(c, 0)) * double(mesh->h_cell_measure(c));
    }
    return sum;
}

} // namespace

TEST(LESSolver, EddyViscosityUsesTheCellVolumeAsFilterWidth) {
    // A linear velocity field, whose least-squares gradients are exact away
    // from boundaries, on cells whose edges, areas and volumes all differ
    std::ostringstream s;
    const uint32_t nx = 12, ny = 7;
    s << "[run]\nn_steps = 0\ncfl = 0.5\n[mesh]\ntype = \"" << MESH << "\"\nNx = " << nx << "\nNy = " << ny
      << "\nLx = 1.3\nLy = 0.6\n"
      << (N_DIM == 3 ? "Nz = 5\nLz = 0.45\n" : "")
      << "[initialize]\ntype = \"analytical\"\nrho = \"2.0\"\np = \"10.0\"\n"
      << (N_DIM == 3 ? "u = [\"0.3 * x - 0.7 * y + 0.2 * z\", \"0.9 * x + 0.1 * y - 0.4 * z\", \"-0.5 * x + 0.6 * y - 0.4 * z\"]\n"
                     : "u = [\"0.3 * x - 0.7 * y\", \"0.9 * x + 0.1 * y\"]\n")
      << "[numerics]\nriemann_solver = \"HLLC\"\n[numerics.face_reconstruction]\ntype = \"FO\"\n"
      << "[physics]\ntype = \"navier_stokes\"\ngamma = 1.4\np_ref = 1.0\nT_ref = 1.0\nrho_ref = 1.0\nmu = 1e-3\n"
      << "[output]\ncheck_interval = 1000000\n[les]\nmodel = \"wale\"\nC = 0.4\nPr_t = 0.6\nfilter_width = \"volume\"\n";
    for (const char * side : {"left", "right", "bottom", "top", "back", "front"}) {
        if (N_DIM == 2 && (std::string(side) == "back" || std::string(side) == "front")) continue;
        s << "[[boundaries]]\nname = \"" << side << "\"\ntype = \"extrapolation\"\n";
    }
    auto solver = make_solver(s.str());
    solver->copy_device_to_host();
    Gradient g{};
    g[0] = {0.3, -0.7, N_DIM == 3 ? 0.2 : 0.0};
    g[1] = {0.9, 0.1, N_DIM == 3 ? -0.4 : 0.0};
    if (N_DIM == 3) g[2] = {-0.5, 0.6, -0.4};
    const double d = D(SGSModel::WALE, g);
    const auto mesh = solver->get_mesh();
    const auto & sgs = solver->get_les_coefficients();
    const double h = 0.6 / ny;
    uint32_t checked = 0;
    for (uint32_t c = 0; c < mesh->n_cells; c++) {
        const double x = double(mesh->h_cell_coords(c, 0)), y = double(mesh->h_cell_coords(c, 1));
        bool interior = x > 2 * h && x < 1.3 - 2 * h && y > 2 * h && y < 0.6 - 2 * h;
        if constexpr (N_DIM == 3) interior = interior && double(mesh->h_cell_coords(c, 2)) > 2 * h && double(mesh->h_cell_coords(c, 2)) < 0.45 - 2 * h;
        if (!interior) continue;
        const double V = double(mesh->h_cell_volume(c));
        const double mu_t = 2.0 * 0.16 * std::pow(V, 2.0 / N_DIM) * d;
        EXPECT_NEAR(sgs(c, 0), mu_t, precision_tol<double>(1e-9, 1e-4) * mu_t) << "cell " << c;
        EXPECT_NEAR(sgs(c, 1), 3.5 * mu_t / 0.6, precision_tol<double>(1e-9, 1e-4) * 3.5 * mu_t / 0.6);
        checked++;
    }
    EXPECT_GT(checked, 10u);
}

TEST(LESModels, ScottiFactorOfTheAspectRatios) {
    // Scotti, Meneveau & Lilly (1993): 1 for cubes, about 1.2 and 1.4 for aspect ratios 5 and 10
    EXPECT_DOUBLE_EQ(LES::scotti_factor(1.0, 1.0, 1.0), 1.0);
    EXPECT_NEAR(LES::scotti_factor(1.0, 1.0, 5.0), 1.198, 1e-3);
    EXPECT_NEAR(LES::scotti_factor(1.0, 1.0, 10.0), 1.420, 1e-3);
    // Only the ratios count
    EXPECT_DOUBLE_EQ(LES::scotti_factor(0.2, 0.5, 2.0), LES::scotti_factor(1.0, 2.5, 10.0));
}

#if Mallard_DIM == 3

TEST(LESSolver, ScottiWidthUsesTheExtentsOfEachCell) {
    // Boxes of 0.1 x 0.02 x 0.05 and a linear velocity: the eddy viscosity grows by the factor squared of
    // the extents' ratios (0.2, 0.5), against the volume width
    auto input = [](const std::string & width) {
        std::ostringstream s;
        s << "[run]\nn_steps = 0\ncfl = 0.5\n[mesh]\ntype = \"cartesian\"\nNx = 8\nNy = 20\nNz = 10\n"
          << "Lx = 0.8\nLy = 0.4\nLz = 0.5\nperiodic = [\"x\", \"y\", \"z\"]\n"
          << "[initialize]\ntype = \"analytical\"\nrho = \"2.0\"\np = \"10.0\"\n"
          << "u = [\"0.4 * sin(7.853981633974483 * y)\", \"0.3 * sin(12.566370614359172 * z)\", "
             "\"0.2 * sin(7.853981633974483 * x)\"]\n"
          << "[numerics]\nriemann_solver = \"HLLC\"\n[numerics.face_reconstruction]\ntype = \"FO\"\n"
          << "[physics]\ntype = \"navier_stokes\"\ngamma = 1.4\np_ref = 1.0\nT_ref = 1.0\nrho_ref = 1.0\nmu = 1e-3\n"
          << "[output]\ncheck_interval = 1000000\n[les]\nmodel = \"vreman\"\n" << width;
        return make_solver(s.str());
    };
    auto volume = input("filter_width = \"volume\"\n");
    auto scotti = input("filter_width = \"scotti\"\n");
    volume->copy_device_to_host();
    scotti->copy_device_to_host();
    const double f = LES::scotti_factor(0.02, 0.05, 0.1);
    EXPECT_GT(f, 1.1);
    const auto & a = volume->get_les_coefficients();
    const auto & b = scotti->get_les_coefficients();
    uint32_t checked = 0;
    for (uint32_t c = 0; c < volume->get_mesh()->n_owned(); c++) {
        if (a(c, 0) < 1e-12) continue;
        EXPECT_NEAR(b(c, 0), f * f * a(c, 0), 1e-12 * f * f * a(c, 0) + 1e-300) << "cell " << c;
        checked++;
    }
    EXPECT_GT(checked, 100u);
}

#endif

namespace {

/** @brief <L:M> / <M:M> of the dynamic procedure after one step of a periodic box of side length with velocity scale u0 plus u_shift. */
double dynamic_constant(const std::string & model, double length, double u0, double u_shift) {
    std::ostringstream s;
    const double k = 6.283185307179586 / length;
    // Modulated plane waves (sums of plain plane waves give <L:M> = 0 exactly), arbitrary phases per component
    const int waves[4][3] = {{1, 2, 0}, {0, 1, 3}, {2, -1, 1}, {3, 1, -2}};
    auto component = [&](int i) {
        std::ostringstream m;
        m << std::setprecision(17) << u_shift;
        for (int w = 0; w < 4; w++) {
            m << " + " << u0 * (1.0 + 0.3 * i - 0.2 * w) << " * sin(" << k << " * (" << waves[w][0] << " * x + "
              << waves[w][1] << " * y" << (N_DIM == 3 ? " + " + std::to_string(waves[w][2]) + " * z" : std::string())
              << ") + " << 0.7 * i + 1.3 * w << ") * (1.2 + cos(" << k << " * (x - 2 * y) + " << w << "))";
        }
        return m.str();
    };
    s << std::setprecision(17) << "[run]\nn_steps = 1\ncfl = 0.1\n[mesh]\ntype = \"cartesian\"\nNx = 12\nNy = 12\n"
      << "Lx = " << length << "\nLy = " << length << "\n"
      << (N_DIM == 3 ? "Nz = 12\nLz = " + std::to_string(length) + "\nperiodic = [\"x\", \"y\", \"z\"]\n"
                     : std::string("periodic = [\"x\", \"y\"]\n"))
      << "[initialize]\ntype = \"analytical\"\nrho = \"1.0\"\np = \"100.0\"\n";
    if constexpr (N_DIM == 3) {
        s << "u = [\"" << component(0) << "\", \"" << component(1) << "\", \"" << component(2) << "\"]\n";
    } else {
        s << "u = [\"" << component(0) << "\", \"" << component(1) << "\"]\n";
    }
    s << "[numerics]\nriemann_solver = \"HLLC\"\n[numerics.face_reconstruction]\ntype = \"FO\"\n"
      << "[physics]\ntype = \"navier_stokes\"\ngamma = 1.4\np_ref = 1.0\nT_ref = 1.0\nrho_ref = 1.0\nmu = 1e-6\n"
      << "[output]\ncheck_interval = 1000000\n[les]\nmodel = \"" << model << "\"\ndynamic = true\n";
    auto solver = make_solver(s.str());
    solver->run();
    return solver->les_dynamic_ratio();
}

} // namespace

TEST(LESSolver, DynamicConstantIsGalileanAndScaleInvariant) {
    // L, M and so C depend on velocity differences only, and C^2 = <L:M> / <M:M> is dimensionless: a
    // missing Delta^2, a test-filter width in the wrong units or a non-central L would break these
    const std::string model = N_DIM == 3 ? "sigma" : "wale";
    const double c = dynamic_constant(model, 1.0, 1.0, 0.0);
    EXPECT_GT(std::abs(c), 1e-3);
    EXPECT_NEAR(dynamic_constant(model, 1.0, 1.0, 0.7), c, 1e-6 * std::abs(c));
    EXPECT_NEAR(dynamic_constant(model, 10.0, 1.0, 0.0), c, 1e-6 * std::abs(c));
    EXPECT_NEAR(dynamic_constant(model, 1.0, 5.0, 0.0), c, 1e-6 * std::abs(c)) << c;
}

TEST(LESSolver, BudgetSplitsTheKineticEnergyRateOfTheRightHandSide) {
    const uint32_t n = N_DIM == 2 ? 32 : 24;
    auto solver = make_solver(periodic_box("cartesian", n, "[les]\nmodel = \"wale\"\nC = 0.5\n", "", "n_steps = 0\ncfl = 0.5\n"));
    const KineticEnergyBudget b = solver->kinetic_energy_budget();
    const auto mesh = solver->get_mesh();

    State rhs("rhs", mesh->n_cells, 0);
    solver->calc_rhs(solver->state(), rhs, 0.0_r);
    solver->copy_device_to_host();
    auto h_rhs = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), rhs.flow);
    double total = 0.0, viscous = 0.0, sgs = 0.0, scale = 0.0;
    const auto & coefficients = solver->get_les_coefficients();
    for (uint32_t c = 0; c < mesh->n_owned(); c++) {
        const double rho = double(solver->h_conservatives(c, 0)), V = double(mesh->h_cell_measure(c));
        double u[N_DIM], u2 = 0.0, work = 0.0;
        FOR_I_DIM {
            u[i] = double(solver->h_conservatives(c, 1 + i)) / rho;
            u2 += u[i] * u[i];
            work += u[i] * double(h_rhs(c, 1 + i));
        }
        total += (work - 0.5 * u2 * double(h_rhs(c, 0))) * V;
        double x[3] = {};
        FOR_I_DIM x[i] = double(mesh->h_cell_coords(c, i));
        const double phi = dissipation_per_viscosity(box_gradient(x)) * V;
        viscous -= 1e-3 * phi;
        sgs -= double(coefficients(c, 0)) * phi;
        scale += std::abs(work * V);
    }
    // The three parts add up to the whole convective and diffusive rate
    EXPECT_NEAR(b.convective + b.viscous + b.sgs, total, precision_tol<double>(1e-10, 1e-4) * scale);
    // Each diffusive part is the dissipation of its own stress, -int tau : grad u dV
    EXPECT_LT(b.sgs, 0.0);
    EXPECT_NEAR(b.sgs, sgs, 0.08 * std::abs(sgs));
    EXPECT_NEAR(b.viscous, viscous, 0.08 * std::abs(viscous));
}

TEST(LESSolver, ModelDrainsResolvedKineticEnergy) {
    const uint32_t n = N_DIM == 2 ? 24 : 10;
    const std::string run = "t_stop = 0.05\ncfl = 0.5\n";
    auto without = make_solver(periodic_box(MESH, n, "", "", run));
    auto with = make_solver(periodic_box(MESH, n, LES_DEFAULT, "", run));
    without->run();
    with->run();
    without->copy_device_to_host();
    with->copy_device_to_host();
    ASSERT_EQ(with->get_time(), without->get_time());
    const double K0 = kinetic_energy(*without), K1 = kinetic_energy(*with);
    EXPECT_LT(K1, K0 * (1.0 - 1e-4));
}

TEST(LESSolver, BudgetOutputDoesNotChangeTheSolution) {
    const uint32_t n = N_DIM == 2 ? 16 : 8;
    const std::string integrals = "[integrals]\ninterval = 1\nbudget = true\nfile = \"les_budget_test.csv\"\n";
    auto plain = make_solver(periodic_box(MESH, n, LES_DEFAULT));
    auto budgeted = make_solver(periodic_box(MESH, n, LES_DEFAULT, integrals));
    plain->run();
    budgeted->run();
    plain->copy_device_to_host();
    budgeted->copy_device_to_host();
    std::filesystem::remove("les_budget_test.csv");
    for (uint32_t c = 0; c < plain->get_mesh()->n_cells; c++) {
        FOR_I_CONSERVATIVE ASSERT_EQ(plain->h_conservatives(c, i), budgeted->h_conservatives(c, i)) << "cell " << c;
    }
}

TEST(LESSolver, EddyViscosityLimitsTheTimeStep) {
    // A strong Smagorinsky model on a coarse mesh: nu_t far above mu
    const uint32_t n = N_DIM == 2 ? 8 : 6;
    const std::string run = "n_steps = 1\ncfl = 0.5\n";
    auto without = make_solver(periodic_box("cartesian", n, "", "", run));
    auto with = make_solver(periodic_box("cartesian", n, "[les]\nmodel = \"smagorinsky\"\nC = 3.0\n", "", run));
    without->run();
    with->run();
    EXPECT_LT(double(with->get_time()), 0.7 * double(without->get_time()));
}

TEST(LESSolver, InputErrors) {
    std::string euler = periodic_box(MESH, 4, LES_DEFAULT);
    euler.replace(euler.find("type = \"navier_stokes\""), 22, "type = \"euler\"");
    EXPECT_THROW(make_solver(euler), InputError);
    EXPECT_THROW(make_solver(periodic_box(MESH, 4, "[les]\nmodel = \"dynamic\"\n")), InputError);
    EXPECT_THROW(make_solver(periodic_box(MESH, 4, "[les]\nmodel = \"wale\"\nC = -1.0\n")), InputError);
    if constexpr (N_DIM == 2) {
        EXPECT_THROW(make_solver(periodic_box(MESH, 4, "[les]\nmodel = \"sigma\"\n")), InputError);
    }
}

#if Mallard_DIM == 2

namespace {

std::string couette(const std::string & les) {
    return "[run]\nt_stop = 6.0\ncfl = 0.8\n"
           "[mesh]\ntype = \"cartesian\"\nNx = 4\nNy = 12\nLx = 0.5\nLy = 1.0\nperiodic = [\"x\"]\n"
           "[initialize]\ntype = \"analytical\"\nrho = \"1.0\"\nu = [\"0.0\", \"0.0\"]\np = \"1.0\"\n"
           "[[boundaries]]\nname = \"bottom\"\ntype = \"wall_isothermal\"\nT = 1.0\n"
           "[[boundaries]]\nname = \"top\"\ntype = \"wall_isothermal\"\nT = 1.0\nu = [0.1, 0.0]\n"
           "[numerics]\nriemann_solver = \"HLLC\"\n[numerics.face_reconstruction]\ntype = \"MUSCL\"\n"
           "[physics]\ntype = \"navier_stokes\"\ngamma = 1.4\np_ref = 1.0\nT_ref = 1.0\nrho_ref = 1.0\nmu = 0.2\n"
           "[output]\ncheck_interval = 1000000\n" + les;
}

} // namespace

TEST(LESSolver, PureShearIsUnchangedByWALEAndVremanButNotBySmagorinsky) {
    auto reference = make_solver(couette(""));
    reference->run();
    reference->copy_device_to_host();
    for (const char * model : {"wale", "vreman", "smagorinsky"}) {
        auto les = make_solver(couette(std::string("[les]\nmodel = \"") + model + "\"\nC = 1.0\n"));
        les->run();
        les->copy_device_to_host();
        double diff = 0.0;
        for (uint32_t c = 0; c < les->get_mesh()->n_cells; c++) {
            diff = std::max(diff, std::abs(double(les->h_conservatives(c, 1) - reference->h_conservatives(c, 1))));
        }
        if (std::string(model) == "smagorinsky") {
            EXPECT_GT(diff, 1e-7) << model;
        } else {
            EXPECT_LT(diff, precision_tol<double>(1e-11, 1e-5)) << model;
        }
    }
}

#endif

namespace {

std::string mixture_box(const std::string & les, const std::string & composition) {
    std::ostringstream s;
    // A fixed time step: the time step's diffusivity grows with nu_t / Sc_t
    s << "[run]\nn_steps = 8\ndt = 2e-7\n[mesh]\ntype = \"" << MESH << "\"\nNx = 8\nNy = 8\nLx = 0.01\nLy = 0.01\n"
      << (N_DIM == 3 ? "Nz = 4\nLz = 0.005\nperiodic = [\"x\", \"y\", \"z\"]\n" : "periodic = [\"x\", \"y\"]\n")
      << "[initialize]\ntype = \"analytical\"\np = \"1.0e5\"\nT = \"600 + 200 * sin(628.3185307179586 * x)\"\n"
      << "u = [\"30 * sin(628.3185307179586 * y)\", \"-20 * sin(628.3185307179586 * x)\""
      << (N_DIM == 3 ? ", \"10 * cos(628.3185307179586 * (x + y))\"]\n" : "]\n")
      << composition
      << "[numerics]\nriemann_solver = \"HLLC\"\n[numerics.face_reconstruction]\ntype = \"MUSCL\"\n"
      << "[physics]\ntype = \"navier_stokes\"\ngas = \"mixture\"\nmechanism = \"" MALLARD_SOURCE_DIR "/mechanisms/h2o2.yaml\"\n"
      << "[output]\ncheck_interval = 1000000\n" << les;
    return s.str();
}

const std::string UNIFORM = "X = { H2 = 0.2, O2 = 0.1, N2 = 0.7 }\n";
const std::string STRATIFIED = "X = { H2 = \"0.2 + 0.15 * sin(628.3185307179586 * y)\", O2 = 0.1, N2 = 0.5 }\n";

} // namespace

TEST(LESMixture, SchmidtNumberActsOnCompositionGradientsOnly) {
    auto run = [](const std::string & les, const std::string & composition) {
        auto solver = make_solver(mixture_box(les, composition));
        solver->run();
        solver->copy_device_to_host();
        return solver;
    };
    const std::string model = N_DIM == 2 ? "model = \"vreman\"\n" : "model = \"sigma\"\n";
    auto low = run("[les]\n" + model + "Sc_t = 0.3\n", UNIFORM);
    auto high = run("[les]\n" + model + "Sc_t = 3.0\n", UNIFORM);
    auto none = run("", UNIFORM);
    const uint32_t n = low->get_mesh()->n_cells;
    double les_effect = 0.0;
    double sc_effect = 0.0;
    for (uint32_t c = 0; c < n; c++) {
        FOR_I_CONSERVATIVE {
            const double ref = std::abs(double(low->h_conservatives(c, i))) + 1.0;
            sc_effect = std::max(sc_effect, std::abs(double(low->h_conservatives(c, i) - high->h_conservatives(c, i))) / ref);
        }
        les_effect = std::max(les_effect, std::abs(double(low->h_conservatives(c, 1) - none->h_conservatives(c, 1))));
    }
    // Uniform mass fractions leave only round-off gradients for the SGS species fluxes
    EXPECT_LT(sc_effect, precision_tol<double>(1e-11, 1e-5));
    EXPECT_GT(les_effect, 1e-6);

    auto low_s = run("[les]\n" + model + "Sc_t = 0.3\n", STRATIFIED);
    auto high_s = run("[les]\n" + model + "Sc_t = 3.0\n", STRATIFIED);
    double species_effect = 0.0;
    const uint32_t ns = low_s->get_species_names().size();
    for (uint32_t c = 0; c < n; c++) {
        double sum = 0.0;
        for (uint32_t k = 0; k < ns; k++) {
            sum += double(low_s->h_species(c, k));
            species_effect = std::max(species_effect, std::abs(double(low_s->h_species(c, k) - high_s->h_species(c, k))));
        }
        // SGS fluxes sum to zero: the partial densities still add up to rho
        EXPECT_NEAR(sum, low_s->h_conservatives(c, 0), precision_tol<double>(1e-12, 1e-5) * sum);
    }
    EXPECT_GT(species_effect, 1e-9);
}

namespace {

/** @brief Sum over faces of A mean(p) (u_1 - u_0) . n: the pressure work of a two-point flux of Jameson's form. */
double pressure_work(const Solver & solver, const double gamma) {
    const auto m = solver.get_mesh();
    auto state = [&](int32_t c, double * u, double & p) {
        const double rho = double(solver.h_conservatives(c, 0));
        double u2 = 0.0;
        FOR_I_DIM {
            u[i] = double(solver.h_conservatives(c, 1 + i)) / rho;
            u2 += u[i] * u[i];
        }
        p = (gamma - 1.0) * (double(solver.h_conservatives(c, N_DIM + 1)) - 0.5 * rho * u2);
    };
    double sum = 0.0;
    for (uint32_t f = 0; f < m->n_faces; f++) {
        double u0[N_DIM], u1[N_DIM], p0, p1, n2 = 0.0, du_n = 0.0;
        state(m->h_cells_of_face(f, 0), u0, p0);
        state(m->h_cells_of_face(f, 1), u1, p1);
        FOR_I_DIM {
            n2 += double(m->h_face_normals(f, i)) * double(m->h_face_normals(f, i));
            du_n += (u1[i] - u0[i]) * double(m->h_face_normals(f, i));
        }
        sum += double(m->h_face_area(f)) * 0.5 * (p0 + p1) * du_n / std::sqrt(n2);
    }
    return sum;
}

std::string with_flux(std::string input, const std::string & numerics, const std::string & reconstruction) {
    input.replace(input.find("[numerics]\n"), 11, "[numerics]\n" + numerics);
    const std::string muscl = "type = \"MUSCL\"\nlimiter = \"none\"\n";
    input.replace(input.find(muscl), muscl.size(), reconstruction);
    return input;
}

} // namespace

TEST(HybridFlux, CentralFluxChangesKineticEnergyOnlyByThePressureWork) {
    // First-order reconstruction: the face states are the cell values, so
    // KEEP's momentum flux C mean(u) + mean(p) n leaves the pressure work as
    // the whole convective kinetic-energy rate, on any mesh
    const uint32_t n = N_DIM == 2 ? 12 : 6;
    const std::string box = periodic_box(MESH, n, "", "", "n_steps = 0\ncfl = 0.5\n");
    // A threshold of 1 never selects the Riemann solver
    auto central = make_solver(with_flux(box, "convective_flux = \"hybrid\"\n", "type = \"FO\"\n") +
                               "[numerics.hybrid]\nsensor_threshold = 1.0\n");
    auto upwind = make_solver(with_flux(box, "", "type = \"FO\"\n"));
    central->copy_device_to_host();
    const double work = pressure_work(*central, 1.4);
    double scale = 0.0;
    for (uint32_t c = 0; c < central->get_mesh()->n_owned(); c++) {
        scale += double(std::abs(central->h_conservatives(c, 1)) * central->get_mesh()->h_cell_measure(c));
    }
    EXPECT_NEAR(central->kinetic_energy_budget().convective, work, precision_tol<double>(1e-12, 1e-5) * scale);
    // The Riemann solver dissipates on top of it
    EXPECT_LT(upwind->kinetic_energy_budget().convective, work - 1e-3 * scale);
}

#if Mallard_DIM == 2

namespace {

/** @brief L1 density error of Sod's problem against the exact solution, and the largest density. */
std::pair<double, double> sod(const std::string & numerics) {
    std::string input = "[run]\nt_stop = 0.2\ncfl = 0.5\n[mesh]\ntype = \"cartesian_tri\"\nNx = 200\nNy = 2\nLx = 1.0\n"
                        "Ly = 0.01\n[initialize]\ntype = \"analytical\"\nrho = \"x < 0.5 ? 1.0 : 0.125\"\n"
                        "u = [\"0.0\", \"0.0\"]\np = \"x < 0.5 ? 1.0 : 0.1\"\n"
                        "[[boundaries]]\nname = \"left\"\ntype = \"extrapolation\"\n"
                        "[[boundaries]]\nname = \"right\"\ntype = \"extrapolation\"\n"
                        "[[boundaries]]\nname = \"bottom\"\ntype = \"symmetry\"\n"
                        "[[boundaries]]\nname = \"top\"\ntype = \"symmetry\"\n"
                        "[numerics]\nriemann_solver = \"HLLC\"\n" + numerics +
                        "[numerics.face_reconstruction]\ntype = \"MUSCL\"\n"
                        "[physics]\ntype = \"euler\"\ngamma = 1.4\np_ref = 1.0\nT_ref = 1.0\nrho_ref = 1.0\n"
                        "[output]\ncheck_interval = 1000000\n";
    auto solver = make_solver(input);
    solver->run();
    solver->copy_device_to_host();
    ExactRiemann exact(1.0, 0.0, 1.0, 0.125, 0.0, 0.1, 1.4);
    const auto m = solver->get_mesh();
    double err = 0.0, vol = 0.0, rho_max = 0.0;
    for (uint32_t c = 0; c < m->n_cells; c++) {
        double rho, u, p;
        exact.sample((double(m->h_cell_coords(c, 0)) - 0.5) / double(solver->get_time()), rho, u, p);
        err += std::abs(double(solver->h_conservatives(c, 0)) - rho) * double(m->h_cell_volume(c));
        vol += double(m->h_cell_volume(c));
        rho_max = std::max(rho_max, double(solver->h_conservatives(c, 0)));
    }
    return {err / vol, rho_max};
}

} // namespace

TEST(HybridFlux, ShocksKeepTheRiemannSolver) {
    const auto [riemann_error, riemann_max] = sod("");
    const auto [hybrid_error, hybrid_max] = sod("convective_flux = \"hybrid\"\n");
    // Without the sensor, the central flux alone oscillates at the shock
    const auto [central_error, central_max] = sod("convective_flux = \"hybrid\"\n[numerics.hybrid]\nsensor_threshold = 1.0\n");
    // The contact and the expansion stay central: a little less sharp, overshoot under 1%
    EXPECT_LT(hybrid_error, 2.0 * riemann_error);
    EXPECT_LT(hybrid_max, 1.01);
    EXPECT_GT(central_error, 2.0 * hybrid_error) << central_max;
    (void)riemann_max;
}

#endif

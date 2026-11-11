/**
 * @file derived_fields_test.cpp
 * @brief The derived output fields MACH, Q_CRITERION, LAMBDA2 and SCHLIEREN
 *        against analytic flows, and the device kernel against the host.
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 */

#include <gtest/gtest.h>
#include <Kokkos_Core.hpp>

#include <algorithm>
#include <cmath>
#include <memory>
#include <random>
#include <sstream>
#include <string>

#include "data_paths.h"
#include "derived_fields.h"
#include "scratch.h"
#include "solver.h"
#include "test_fixtures.h"
#include "test_utils.h"

namespace {

const char * ZONES[6] = {"left", "right", "bottom", "top", "back", "front"};

/** @brief A TOML array of the velocity components, the third dropped in 2D. */
std::string velocity(const std::string & ux, const std::string & uy, const std::string & uz) {
    return N_DIM == 2 ? "[\"" + ux + "\", \"" + uy + "\"]" : "[\"" + ux + "\", \"" + uy + "\", \"" + uz + "\"]";
}

/**
 * @brief A solver initialized to rho, u, p on 4^2 triangles or 3^3 tetrahedra
 *        of the unit square or cube, the same state on every boundary, writing
 *        the listed variables.
 */
std::unique_ptr<Solver> init_flow(const std::string & rho, const std::string & u, const std::string & p,
                                  const std::string & variables, const std::string & physics = "") {
    const std::string state = "rho = \"" + rho + "\"\nu = " + u + "\np = \"" + p + "\"\n";
    std::ostringstream s;
    s << "[run]\nn_steps = 10\ncfl = 0.25\n[mesh]\n"
      << (N_DIM == 2 ? "type = \"cartesian_tri\"\nNx = 4\nNy = 4\nLx = 1.0\nLy = 1.0\n"
                     : "type = \"cartesian_tet\"\nNx = 3\nNy = 3\nNz = 3\nLx = 1.0\nLy = 1.0\nLz = 1.0\n")
      << "[initialize]\ntype = \"analytical\"\n" << state;
    for (int k = 0; k < 2 * N_DIM; k++) s << "[[boundaries]]\nname = \"" << ZONES[k] << "\"\ntype = \"dirichlet\"\n" << state;
    s << "[numerics]\nriemann_solver = \"HLLC\"\ntime_integrator = \"SSPRK3\"\n"
      << "[numerics.face_reconstruction]\ntype = \"FO\"\n"
      << (physics.empty() ? "[physics]\ntype = \"euler\"\ngamma = 1.4\np_ref = 1.0\nT_ref = 1.0\nrho_ref = 1.0\n" : physics)
      << "[output]\ncheck_interval = 1000000\n[[write_data]]\nprefix = \""
      << (scratch_dir() / "mallard_derived").string() << "\"\nformat = \"vtu\"\ninterval = 1000000\nfields = ["
      << variables << "]\n";
    auto solver = std::make_unique<Solver>();
    solver->set_distributed(false);
    solver->init(parse_toml(s.str()));
    solver->copy_device_to_host();
    return solver;
}

/** @brief Field f of cell c, as the writers read it. */
double field(const Solver & solver, derived::Field f, uint32_t c) {
    const int32_t column = solver.derived_selection.column[f];
    EXPECT_GE(column, 0) << derived::NAMES[f] << " is not computed";
    return double(solver.h_derived_fields(c, static_cast<uint32_t>(std::max(column, 0))));
}

/** @brief Expect field f equal to value, up to round-off, in every owned cell. */
void expect_everywhere(const Solver & solver, derived::Field f, double value, double tol) {
    for (uint32_t c = 0; c < solver.get_mesh()->n_owned(); c++) {
        EXPECT_NEAR(field(solver, f, c), value, tol) << derived::NAMES[f] << ", cell " << c;
    }
}

derived::Tensor tensor(const double (&a)[3][3]) {
    derived::Tensor t{};
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) t.a[i][j] = static_cast<rtype>(a[i][j]);
    }
    return t;
}

/** @brief R a R^T of the rotation R by angle about the unit axis n. */
derived::Tensor rotated(const double (&a)[3][3], const double n[3], double angle) {
    const double c = std::cos(angle), s = std::sin(angle);
    double R[3][3];
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) R[i][j] = (i == j ? c : 0.0) + (1.0 - c) * n[i] * n[j];
    }
    R[0][1] -= s * n[2]; R[1][0] += s * n[2];
    R[0][2] += s * n[1]; R[2][0] -= s * n[1];
    R[1][2] -= s * n[0]; R[2][1] += s * n[0];
    double out[3][3] = {};
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            for (int k = 0; k < 3; k++) {
                for (int l = 0; l < 3; l++) out[i][j] += R[i][k] * a[k][l] * R[j][l];
            }
        }
    }
    return tensor(out);
}

} // namespace

TEST(DerivedFields, SymmetricEigenvaluesOfRotatedSpectra) {
    // R diag(l) R^T for random rotations, with repeated, zero and widely
    // scaled eigenvalues: the Jacobi solver returns l up to round-off
    const double spectra[][3] = {{1.0, 2.0, 3.0},    {1.0, 1.0, -2.0}, {0.0, 0.0, 5.0}, {-4.0, 1e-3, 2.0},
                                 {1e-12, 3e-12, -2e-12}, {7e10, -3e10, 1e10}, {0.0, 0.0, 0.0}};
    std::mt19937 rng(511);
    std::uniform_real_distribution<double> uniform(-1.0, 1.0);
    for (const auto & l : spectra) {
        for (int trial = 0; trial < 20; trial++) {
            double n[3] = {uniform(rng), uniform(rng), uniform(rng)};
            const double norm = std::sqrt(n[0] * n[0] + n[1] * n[1] + n[2] * n[2]);
            for (double & x : n) x /= norm;
            const double d[3][3] = {{l[0], 0.0, 0.0}, {0.0, l[1], 0.0}, {0.0, 0.0, l[2]}};
            derived::Tensor m = rotated(d, n, M_PI * uniform(rng));
            rtype ev[3];
            derived::symmetric_eigenvalues(m.a, ev);
            double got[3] = {double(ev[0]), double(ev[1]), double(ev[2])}, want[3] = {l[0], l[1], l[2]};
            std::sort(got, got + 3);
            std::sort(want, want + 3);
            const double scale = std::max({std::abs(l[0]), std::abs(l[1]), std::abs(l[2])});
            for (int i = 0; i < 3; i++) EXPECT_NEAR(got[i], want[i], roundoff(1e-13) * scale) << l[0] << " " << l[1];
        }
    }
}

TEST(DerivedFields, InvariantsOfRotatedVortexAndStrain) {
    // Rotation w about z with in-plane strain a: S^2 + Omega^2 =
    // diag(a^2 - w^2, a^2 - w^2, 0), the middle of which is lambda2 = a^2 - w^2
    // = -Q (negative in a vortex, w > a), in any frame; and an axisymmetric strain (a, a, -2a)
    // has Q = -3a^2 and lambda2 = a^2
    const double n[3] = {2.0 / 3.0, -1.0 / 3.0, 2.0 / 3.0};
    for (const auto & [w, a] : {std::pair{3.0, 1.0}, std::pair{1.0, 2.0}}) {
        const double g[3][3] = {{a, -w, 0.0}, {w, -a, 0.0}, {0.0, 0.0, 0.0}};
        const derived::Tensor t = rotated(g, n, 0.7);
        EXPECT_NEAR(double(derived::q_criterion(t)), w * w - a * a, roundoff(1e-12) * w * w);
        EXPECT_NEAR(double(derived::lambda2(t)), a * a - w * w, roundoff(1e-12) * w * w);
    }
    const double a = 0.5;
    const double g[3][3] = {{a, 0.0, 0.0}, {0.0, a, 0.0}, {0.0, 0.0, -2.0 * a}};
    const derived::Tensor t = rotated(g, n, 1.1);
    EXPECT_NEAR(double(derived::q_criterion(t)), -3.0 * a * a, roundoff(1e-13));
    EXPECT_NEAR(double(derived::lambda2(t)), a * a, roundoff(1e-13));
}

TEST(DerivedFields, DeviceKernelMatchesHost) {
    // The kernel of derived::compute on the default execution space against
    // the same functions on the host, for random cell states and gradients
    const uint32_t n = 257;
    Kokkos::View<rtype *[N_CONSERVATIVE]> W("W", n);
    Kokkos::View<rtype *[N_CONSERVATIVE][N_DIM]> g("g", n);
    auto h_W = Kokkos::create_mirror_view(W);
    auto h_g = Kokkos::create_mirror_view(g);
    std::mt19937 rng(2);
    std::uniform_real_distribution<double> uniform(-1.0, 1.0);
    for (uint32_t c = 0; c < n; c++) {
        h_W(c, 0) = static_cast<rtype>(1.5 + uniform(rng));
        FOR_I_DIM h_W(c, 1 + i) = static_cast<rtype>(uniform(rng));
        h_W(c, N_DIM + 1) = static_cast<rtype>(2.0 + uniform(rng));
        for (uint8_t k = 0; k < N_CONSERVATIVE; k++) {
            FOR_I_DIM h_g(c, k, i) = static_cast<rtype>(10.0 * uniform(rng));
        }
    }
    Kokkos::deep_copy(W, h_W);
    Kokkos::deep_copy(g, h_g);
    derived::Selection selection;
    for (uint8_t f = 0; f < derived::N_FIELDS; f++) selection.add(static_cast<derived::Field>(f));
    Kokkos::View<rtype **> out("out", n, selection.n_columns);
    derived::compute(selection, derived::Inputs{W, g, 1.3_r, {}, {}}, n, out);
    const auto h_out = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), out);
    for (uint32_t c = 0; c < n; c++) {
        derived::Tensor t{};
        for (uint8_t k = 0; k < N_DIM; k++) {
            FOR_I_DIM t.a[k][i] = h_g(c, 1 + k, i);
        }
        rtype u2 = 0.0, g2 = 0.0;
        FOR_I_DIM {
            u2 += h_W(c, 1 + i) * h_W(c, 1 + i);
            g2 += h_g(c, 0, i) * h_g(c, 0, i);
        }
        const double host[derived::N_FIELDS] = {
            std::sqrt(double(u2) * double(h_W(c, 0)) / (1.3 * double(h_W(c, N_DIM + 1)))),
            double(derived::q_criterion(t)), double(derived::lambda2(t)), std::sqrt(double(g2))};
        for (uint8_t f = 0; f < derived::N_FIELDS; f++) {
            const double scale = f == derived::MACH || f == derived::SCHLIEREN ? std::abs(host[f]) : 100.0;
            EXPECT_NEAR(double(h_out(c, static_cast<uint32_t>(selection.column[f]))), host[f], roundoff(1e-13) * scale)
                << derived::NAMES[f] << ", cell " << c;
        }
    }
}

TEST(DerivedFields, SolidBodyRotation) {
    // u = w (-(y - 1/2), x - 1/2): Q = w^2 and lambda2 = -w^2, exact for least
    // squares
    const double w = 3.0;
    auto solver = init_flow("1.0", velocity("-3.0 * (y - 0.5)", "3.0 * (x - 0.5)", "0.0"), "2.0",
                            "\"Q_CRITERION\", \"LAMBDA2\"");
    expect_everywhere(*solver, derived::Q_CRITERION, w * w, roundoff(1e-11));
    expect_everywhere(*solver, derived::LAMBDA2, -w * w, roundoff(1e-11));
}

TEST(DerivedFields, QIsRenamedQCriterion) {
    try {
        init_flow("1.0", velocity("0.0", "0.0", "0.0"), "1.0", "\"P\", \"Q\"");
        FAIL() << "Q was accepted";
    } catch (const InputError & e) {
        const std::string what = e.what();
        EXPECT_NE(what.find("Q was renamed to Q_CRITERION"), std::string::npos) << what;
        EXPECT_EQ(what.find("unknown variable"), std::string::npos) << what;
    }
}

TEST(DerivedFields, PureStrain) {
    // u = a (x, -y): S = diag(a, -a, 0), Omega = 0, so Q = -a^2 and lambda2 = a^2
    const double a = 2.0;
    auto solver = init_flow("1.0", velocity("2.0 * x", "-2.0 * y", "0.0"), "2.0", "\"Q_CRITERION\", \"LAMBDA2\"");
    expect_everywhere(*solver, derived::Q_CRITERION, -a * a, roundoff(1e-11));
    expect_everywhere(*solver, derived::LAMBDA2, a * a, roundoff(1e-11));
}

TEST(DerivedFields, SchlierenOfALinearDensity) {
    // rho = 1 + x / 2 + y / 4 (+ z / 8) at rest: |grad rho| is exact for least squares
    const std::string rho = N_DIM == 2 ? "1 + 0.5 * x + 0.25 * y" : "1 + 0.5 * x + 0.25 * y + 0.125 * z";
    const double expected = std::sqrt(0.25 + 0.0625 + (N_DIM == 3 ? 0.015625 : 0.0));
    auto solver = init_flow(rho, velocity("0.0", "0.0", "0.0"), "1.0", "\"SCHLIEREN\"");
    expect_everywhere(*solver, derived::SCHLIEREN, expected, roundoff(1e-12));
}

TEST(DerivedFields, MachOfAUniformFlow) {
    // rho = gamma = 1.4 and p = 1 make c = 1, so M = |u|
    auto solver = init_flow("1.4", velocity("0.3", "0.4", "1.2"), "1.0", "\"MACH\"");
    expect_everywhere(*solver, derived::MACH, N_DIM == 2 ? 0.5 : 1.3, roundoff(1e-14));
}

TEST(DerivedFields, MachOfAUniformArgonFlowTakesTheMixtureGamma) {
    // Argon has cp = 5 R / 2, gamma = 5/3: M = |u| / sqrt(5 p / (3 rho)), not
    // the perfect-gas 1.4
    const std::string mixture = "[physics]\ntype = \"euler\"\ngas = \"mixture\"\nmechanism = \"" + source_dir() +
                                "/mechanisms/h2o2.yaml\"\n";
    const std::string u = velocity("100.0", "50.0", "-20.0");
    std::ostringstream s;
    s << "[run]\nn_steps = 10\ncfl = 0.25\n[mesh]\n"
      << (N_DIM == 2 ? "type = \"cartesian\"\nNx = 3\nNy = 3\nLx = 1.0\nLy = 1.0\n"
                     : "type = \"cartesian\"\nNx = 2\nNy = 2\nNz = 2\nLx = 1.0\nLy = 1.0\nLz = 1.0\n")
      << "[initialize]\ntype = \"analytical\"\np = \"1.0e5\"\nT = \"400.0\"\nu = " << u << "\nY = { AR = 1.0 }\n";
    for (int k = 0; k < 2 * N_DIM; k++) {
        s << "[[boundaries]]\nname = \"" << ZONES[k] << "\"\ntype = \"outlet_extrapolation\"\n";
    }
    s << "[numerics]\nriemann_solver = \"HLLC\"\n[numerics.face_reconstruction]\ntype = \"FO\"\n" << mixture
      << "[output]\ncheck_interval = 1000000\n[[write_data]]\nprefix = \""
      << (scratch_dir() / "mallard_derived_argon").string()
      << "\"\nformat = \"vtu\"\ninterval = 1000000\nfields = [\"MACH\"]\n";
    Solver solver;
    solver.set_distributed(false);
    solver.init(parse_toml(s.str()));
    solver.copy_device_to_host();
    const double speed = std::sqrt(100.0 * 100.0 + 50.0 * 50.0 + (N_DIM == 3 ? 400.0 : 0.0));
    for (uint32_t c = 0; c < solver.get_mesh()->n_owned(); c++) {
        const double rho = double(solver.h_conservatives(c, 0)), p = double(solver.h_primitives(c, N_DIM));
        EXPECT_NEAR(field(solver, derived::MACH, c), speed / std::sqrt(5.0 / 3.0 * p / rho),
                    sizeof(rtype) == sizeof(double) ? 1e-9 : 1e-5)
            << "cell " << c;
    }
}

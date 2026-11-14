/**
 * @file derived_fields_test.cpp
 * @brief The derived output fields (derived_catalog.h) against analytic
 *        flows and states, and the device kernel against the host.
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 */

#include <gtest/gtest.h>
#include <Kokkos_Core.hpp>

#include <algorithm>
#include <cmath>
#include <memory>
#include <vector>
#include <random>
#include <sstream>
#include <string>

#include "data_paths.h"
#include "derived_fields.h"
#include "output_reader.h"
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
                                  const std::string & variables, const std::string & physics = "",
                                  const std::string & output = "") {
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
      << "[output]\ncheck_interval = 1000000\n" << output << "[[write_data]]\nprefix = \""
      << (rank_scratch_dir() / "mallard_derived").string() << "\"\nformat = \"vtu\"\ninterval = 1000000\nfields = ["
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
    EXPECT_GE(column, 0) << derived::FIELDS[f].name << " is not computed";
    return double(solver.h_derived_fields(c, static_cast<uint32_t>(std::max(column, 0))));
}

/** @brief Expect field f equal to value, up to round-off, in every owned cell. */
void expect_everywhere(const Solver & solver, derived::Field f, double value, double tol) {
    for (uint32_t c = 0; c < solver.get_mesh()->n_owned(); c++) {
        EXPECT_NEAR(field(solver, f, c), value, tol) << derived::FIELDS[f].name << ", cell " << c;
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
    // the closed forms on the host, for random cell states and gradients
    // (every field of a perfect gas, each in its own column)
    const uint32_t n = 257;
    Kokkos::View<rtype *[N_CONSERVATIVE]> W("W", n);
    Kokkos::View<rtype *[N_CONSERVATIVE][N_DIM]> g("g", n);
    Kokkos::View<rtype *[N_PRIMITIVE]> P("P", n);
    auto h_W = Kokkos::create_mirror_view(W);
    auto h_g = Kokkos::create_mirror_view(g);
    auto h_P = Kokkos::create_mirror_view(P);
    const double gamma = 1.3, R = 2.0, cp = gamma * R / (gamma - 1.0);
    std::mt19937 rng(2);
    std::uniform_real_distribution<double> uniform(-1.0, 1.0);
    for (uint32_t c = 0; c < n; c++) {
        h_W(c, 0) = static_cast<rtype>(1.5 + uniform(rng));
        FOR_I_DIM h_W(c, 1 + i) = static_cast<rtype>(uniform(rng));
        h_W(c, N_DIM + 1) = static_cast<rtype>(2.0 + uniform(rng));
        h_P(c, N_DIM + 1) = static_cast<rtype>(double(h_W(c, N_DIM + 1)) / (double(h_W(c, 0)) * R));
        h_P(c, N_DIM + 2) = static_cast<rtype>(cp * double(h_P(c, N_DIM + 1)));
        for (uint8_t k = 0; k < N_CONSERVATIVE; k++) {
            FOR_I_DIM h_g(c, k, i) = static_cast<rtype>(10.0 * uniform(rng));
        }
    }
    Kokkos::deep_copy(W, h_W);
    Kokkos::deep_copy(g, h_g);
    Kokkos::deep_copy(P, h_P);
    derived::Settings settings;
    settings.cp = static_cast<rtype>(cp);
    settings.R = static_cast<rtype>(R);
    settings.T_ref = 0.5_r;
    settings.p_ref = 3.0_r;
    settings.reference = true;
    settings.p_inf = 1.5_r;
    settings.q_inf = 0.25_r;
    derived::Selection selection;
    // In reverse, so that columns do not follow the enum
    for (int f = derived::CP; f >= 0; f--) {
        if (settings.available(static_cast<derived::Field>(f))) selection.add(static_cast<derived::Field>(f));
    }
    Kokkos::View<rtype **> out("out", n, selection.n_columns);
    derived::compute(selection, settings,
                     derived::Inputs{.W = W, .gradients = g, .gamma = static_cast<rtype>(gamma), .primitives = P}, n, out);
    const auto h_out = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), out);
    for (uint32_t c = 0; c < n; c++) {
        derived::Tensor t{};
        for (uint8_t k = 0; k < N_DIM; k++) {
            FOR_I_DIM t.a[k][i] = h_g(c, 1 + k, i);
        }
        double u2 = 0.0, g2 = 0.0, div = 0.0, ss = 0.0;
        FOR_I_DIM {
            u2 += double(h_W(c, 1 + i)) * double(h_W(c, 1 + i));
            g2 += double(h_g(c, 0, i)) * double(h_g(c, 0, i));
            div += double(t.a[i][i]);
            for (uint8_t j = 0; j < N_DIM; j++) {
                const double sij = 0.5 * (double(t.a[i][j]) + double(t.a[j][i]));
                ss += sij * sij;
            }
        }
        double w[3] = {double(t.a[2][1] - t.a[1][2]), double(t.a[0][2] - t.a[2][0]), double(t.a[1][0] - t.a[0][1])};
        double helicity = 0.0;
        FOR_I_DIM helicity += double(h_W(c, 1 + i)) * w[i];
        const double rho = double(h_W(c, 0)), p = double(h_W(c, N_DIM + 1)), T = double(h_P(c, N_DIM + 1));
        const double M2 = u2 * rho / (gamma * p);
        const double host[derived::N_FIELDS] = {
            std::sqrt(M2), double(derived::q_criterion(t)), double(derived::lambda2(t)), std::sqrt(g2),
            p * std::pow(1.0 + 0.5 * (gamma - 1.0) * M2, gamma / (gamma - 1.0)), T * (1.0 + 0.5 * (gamma - 1.0) * M2),
            cp * T + 0.5 * u2, std::sqrt(gamma * p / rho), cp * std::log(T / 0.5) - R * std::log(p / 3.0), div, helicity,
            0.5 * (w[0] * w[0] + w[1] * w[1] + w[2] * w[2]), std::sqrt(2.0 * ss), (p - 1.5) / 0.25};
        for (uint8_t f = 0; f <= derived::CP; f++) {
            if (selection.column[f] < 0) continue;
            const double scale = std::max(1.0, std::abs(host[f])) * (f == derived::LAMBDA2 ? 100.0 : 1.0);
            EXPECT_NEAR(double(h_out(c, static_cast<uint32_t>(selection.column[f]))), host[f], roundoff(1e-13) * scale)
                << derived::FIELDS[f].name << ", cell " << c;
        }
    }
}

TEST(DerivedFields, SolidBodyRotation) {
    // u = w (-(y - 1/2), x - 1/2, U): Q = w^2, lambda2 = -w^2, vorticity 2 w
    // along z, so enstrophy 2 w^2 and (3D) helicity 2 w U, and no strain nor
    // dilatation; exact for least squares
    const double w = 3.0, U = 0.5;
    const std::string variables = std::string("\"Q_CRITERION\", \"LAMBDA2\", \"ENSTROPHY\", \"STRAIN_RATE\", ") +
                                  "\"DILATATION\"" + (N_DIM == 3 ? ", \"HELICITY\"" : "");
    auto solver = init_flow("1.0", velocity("-3.0 * (y - 0.5)", "3.0 * (x - 0.5)", "0.5"), "2.0", variables);
    expect_everywhere(*solver, derived::Q_CRITERION, w * w, roundoff(1e-11));
    expect_everywhere(*solver, derived::LAMBDA2, -w * w, roundoff(1e-11));
    expect_everywhere(*solver, derived::ENSTROPHY, 2.0 * w * w, roundoff(1e-11));
    expect_everywhere(*solver, derived::STRAIN_RATE, 0.0, roundoff(1e-6));
    expect_everywhere(*solver, derived::DILATATION, 0.0, roundoff(1e-11));
    if (N_DIM == 3) expect_everywhere(*solver, derived::HELICITY, 2.0 * w * U, roundoff(1e-11));
}

TEST(DerivedFields, DilatationAndStrainOfALinearField) {
    // u = (a x, b y, c z): div u = a + b + c and |S| = sqrt(2 (a^2 + b^2 + c^2)),
    // exact for least squares
    const double a = 2.0, b = -0.5, c = N_DIM == 3 ? 1.5 : 0.0;
    auto solver = init_flow("1.0", velocity("2.0 * x", "-0.5 * y", "1.5 * z"), "2.0", "\"DILATATION\", \"STRAIN_RATE\"");
    expect_everywhere(*solver, derived::DILATATION, a + b + c, roundoff(1e-11));
    expect_everywhere(*solver, derived::STRAIN_RATE, std::sqrt(2.0 * (a * a + b * b + c * c)), roundoff(1e-11));
}

TEST(DerivedFields, HelicityIsAnErrorIn2D) {
    if (N_DIM == 3) GTEST_SKIP() << "3D has helicity";
    try {
        init_flow("1.0", velocity("0.0", "0.0", "0.0"), "1.0", "\"HELICITY\"");
        FAIL() << "HELICITY was accepted in 2D";
    } catch (const InputError & e) {
        EXPECT_NE(std::string(e.what()).find("mallard3d"), std::string::npos) << e.what();
    }
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
      << (rank_scratch_dir() / "mallard_derived_argon").string()
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

namespace {

const std::string PERFECT_GAS = "[physics]\ntype = \"euler\"\ngamma = 1.4\np_ref = 2.0\nT_ref = 0.5\nrho_ref = 1.0\n";

} // namespace

TEST(DerivedFields, IsentropicRelationsOfAPerfectGas) {
    // gamma = 1.4, R = p_ref / (rho_ref T_ref) = 4, cp = 14: a sonic flow (rho = 1.4,
    // p = 1, |u| = 1 = c) has the critical ratios p0 / p = 1.2^3.5 and T0 / T = 1.2,
    // h0 = cp T0, and s = cp ln(T / T_ref) - R ln(p / p_ref) with T = p / (rho R)
    const std::string u = N_DIM == 2 ? velocity("0.6", "0.8", "0.0") : velocity("0.6", "0.0", "0.8");
    auto solver = init_flow("1.4", u, "1.0", "\"MACH\", \"P0\", \"T0\", \"H0\", \"SOUND_SPEED\", \"ENTROPY\"",
                            PERFECT_GAS);
    const double T = 1.0 / (1.4 * 4.0);
    expect_everywhere(*solver, derived::MACH, 1.0, roundoff(1e-14));
    expect_everywhere(*solver, derived::SOUND_SPEED, 1.0, roundoff(1e-14));
    expect_everywhere(*solver, derived::P0, std::pow(1.2, 3.5), roundoff(1e-13));
    expect_everywhere(*solver, derived::T0, 1.2 * T, roundoff(1e-13));
    expect_everywhere(*solver, derived::H0, 14.0 * 1.2 * T, roundoff(1e-12));
    expect_everywhere(*solver, derived::ENTROPY, 14.0 * std::log(T / 0.5) - 4.0 * std::log(0.5), roundoff(1e-12));
}

TEST(DerivedFields, EntropyIsConstantAlongAnIsentrope) {
    // p = rho^gamma: s takes one value, as the uniform state rho = p = 1 has
    const std::string variables = "\"ENTROPY\"";
    auto varying = init_flow("1.0 + 0.5 * x", velocity("0.0", "0.0", "0.0"), "(1.0 + 0.5 * x)^1.4", variables,
                             PERFECT_GAS);
    auto uniform = init_flow("1.0", velocity("0.0", "0.0", "0.0"), "1.0", variables, PERFECT_GAS);
    const double s = field(*uniform, derived::ENTROPY, 0);
    // Cell averages of rho and p are not on the isentrope to round-off: of order (h/2)^2 / 12 relative
    expect_everywhere(*varying, derived::ENTROPY, s, 2e-3 * std::abs(s));
    double spread = 0.0;
    for (uint32_t c = 0; c < varying->get_mesh()->n_owned(); c++) {
        spread = std::max(spread, std::abs(field(*varying, derived::ENTROPY, c) - s));
    }
    // A non-isentropic profile of the same pressures (rho uniform) is far off
    auto off = init_flow("1.0", velocity("0.0", "0.0", "0.0"), "(1.0 + 0.5 * x)^1.4", variables, PERFECT_GAS);
    double off_spread = 0.0;
    for (uint32_t c = 0; c < off->get_mesh()->n_owned(); c++) {
        off_spread = std::max(off_spread, std::abs(field(*off, derived::ENTROPY, c) - s));
    }
    EXPECT_GT(off_spread, 100.0 * spread);
}

TEST(DerivedFields, PressureCoefficientOfTheOutputReference) {
    // q_inf = rho |u|^2 / 2 = 2 (rho = 1, u = (2, 0)): CP = (3 - 1) / 2 = 1
    const std::string reference = "reference = {rho = 1.0, u = " +
                                  std::string(N_DIM == 2 ? "[2.0, 0.0]" : "[0.0, 2.0, 0.0]") + ", p = 1.0}\n";
    auto solver = init_flow("1.0", velocity("0.0", "0.0", "0.0"), "3.0", "\"CP\"", PERFECT_GAS, reference);
    expect_everywhere(*solver, derived::CP, 1.0, roundoff(1e-14));
    try {
        init_flow("1.0", velocity("0.0", "0.0", "0.0"), "3.0", "\"CP\"", PERFECT_GAS);
        FAIL() << "CP was accepted without a reference";
    } catch (const InputError & e) {
        EXPECT_NE(std::string(e.what()).find("[output] reference"), std::string::npos) << e.what();
    }
}

namespace {

/** @brief The mixture fields' case: 8 x 2 (x 2) cells of the unit box, composition, T = 400 K and p = 1 bar at rest. */
std::unique_ptr<Solver> init_mixture(const std::string & Y, const std::string & mixture, const std::string & variables,
                                     bool viscous = false, const std::string & prefix = "mallard_derived_mixture") {
    std::ostringstream s;
    s << "[run]\nn_steps = 10\ncfl = 0.25\n[mesh]\n"
      << (N_DIM == 2 ? "type = \"cartesian\"\nNx = 8\nNy = 2\nLx = 1.0\nLy = 1.0\n"
                     : "type = \"cartesian\"\nNx = 8\nNy = 2\nNz = 2\nLx = 1.0\nLy = 1.0\nLz = 1.0\n")
      << "[initialize]\ntype = \"analytical\"\np = \"1.0e5\"\nT = \"400.0\"\nu = " << velocity("0.0", "0.0", "0.0")
      << "\nY = " << Y << "\n";
    for (int k = 0; k < 2 * N_DIM; k++) {
        s << "[[boundaries]]\nname = \"" << ZONES[k] << "\"\ntype = \"outlet_extrapolation\"\n";
    }
    s << "[numerics]\nriemann_solver = \"HLLC\"\n[numerics.face_reconstruction]\ntype = \"FO\"\n"
      << "[physics]\ntype = \"" << (viscous ? "navier_stokes" : "euler") << "\"\ngas = \"mixture\"\nmechanism = \""
      << source_dir() << "/mechanisms/h2o2.yaml\"\n"
      << (viscous ? "transport = \"unity_lewis\"\n" : "") << "[output]\ncheck_interval = 1000000\n"
      << mixture << "[[write_data]]\nprefix = \"" << (rank_scratch_dir() / prefix).string()
      << "\"\nformat = \"vtu\"\ninterval = 1000000\nfields = [" << variables << "]\n";
    auto solver = std::make_unique<Solver>();
    solver->set_distributed(false);
    solver->init(parse_toml(s.str()));
    solver->copy_device_to_host();
    return solver;
}

const std::string H2_AIR = "[output.mixture]\nfuel = {Y = {H2 = 1.0}}\noxidizer = {Y = {O2 = 0.233, N2 = 0.767}}\n";

/** @brief The mass fraction of species name in cell c. */
double mass_fraction(const Solver & solver, const std::string & name, uint32_t c) {
    const auto & names = solver.get_species_names();
    const auto k = static_cast<uint32_t>(std::find(names.begin(), names.end(), name) - names.begin());
    return double(solver.h_species(c, k)) / double(solver.h_conservatives(c, 0));
}

/** @brief x of the centroid of cell c. */
double x_of(const Solver & solver, uint32_t c) { return double(solver.get_mesh()->h_cell_coords(c, 0)); }

/** @brief Whether cell c is off the x boundaries, where least squares sees the zero-gradient ghosts. */
bool interior(const Solver & solver, uint32_t c) { return x_of(solver, c) > 0.125 && x_of(solver, c) < 0.875; }

} // namespace

TEST(DerivedFields, MixtureFractionOfTwoStreamMixing) {
    // Mixtures (1 - Z) Y_O + Z Y_F of pure H2 and air: Bilger's Z is the mass
    // fraction of the fuel stream, here Y_H2 (about 1 - x: the cells average
    // rho Y), and the fuel and oxidizer gradients oppose (flame index -1, non-premixed)
    auto solver = init_mixture("{ H2 = \"1 - x\", O2 = \"0.233 * x\", N2 = \"0.767 * x\" }", H2_AIR,
                               "\"MIXTURE_FRACTION\", \"FLAME_INDEX\"");
    for (uint32_t c = 0; c < solver->get_mesh()->n_owned(); c++) {
        const double Z = field(*solver, derived::MIXTURE_FRACTION, c);
        EXPECT_NEAR(Z, mass_fraction(*solver, "H2", c), roundoff(1e-12)) << c;
        EXPECT_NEAR(Z, 1.0 - x_of(*solver, c), 0.02) << c;
        EXPECT_NEAR(field(*solver, derived::FLAME_INDEX, c), -1.0, roundoff(1e-12)) << c;
    }
}

TEST(DerivedFields, MixtureFractionStreamsFromBoundaries) {
    // The streams named by boundary: the left inlet holds fuel, the right air
    std::ostringstream s;
    const std::string Y = "{ H2 = \"1 - x\", O2 = \"0.233 * x\", N2 = \"0.767 * x\" }";
    s << "[run]\nn_steps = 10\ncfl = 0.25\n[mesh]\n"
      << (N_DIM == 2 ? "type = \"cartesian\"\nNx = 8\nNy = 2\nLx = 1.0\nLy = 1.0\n"
                     : "type = \"cartesian\"\nNx = 8\nNy = 2\nNz = 2\nLx = 1.0\nLy = 1.0\nLz = 1.0\n")
      << "[initialize]\ntype = \"analytical\"\np = \"1.0e5\"\nT = \"400.0\"\nu = " << velocity("0.0", "0.0", "0.0")
      << "\nY = " << Y << "\n"
      << "[[boundaries]]\nname = \"left\"\ntype = \"inlet_state\"\np = 1.0e5\nT = 400.0\nu = "
      << (N_DIM == 2 ? "[0.0, 0.0]" : "[0.0, 0.0, 0.0]") << "\nY = { H2 = 1.0 }\n"
      << "[[boundaries]]\nname = \"right\"\ntype = \"inlet_state\"\np = 1.0e5\nT = 400.0\nu = "
      << (N_DIM == 2 ? "[0.0, 0.0]" : "[0.0, 0.0, 0.0]") << "\nX = { O2 = 1.0, N2 = 3.76 }\n";
    for (int k = 2; k < 2 * N_DIM; k++) s << "[[boundaries]]\nname = \"" << ZONES[k] << "\"\ntype = \"symmetry\"\n";
    s << "[numerics]\nriemann_solver = \"HLLC\"\n[numerics.face_reconstruction]\ntype = \"FO\"\n"
      << "[physics]\ntype = \"euler\"\ngas = \"mixture\"\nmechanism = \"" << source_dir() << "/mechanisms/h2o2.yaml\"\n"
      << "[output]\ncheck_interval = 1000000\n[output.mixture]\nfuel = \"left\"\noxidizer = \"right\"\n"
      << "[[write_data]]\nprefix = \"" << (rank_scratch_dir() / "mallard_derived_streams").string()
      << "\"\nformat = \"vtu\"\ninterval = 1000000\nfields = [\"MIXTURE_FRACTION\"]\n";
    Solver solver;
    solver.set_distributed(false);
    solver.init(parse_toml(s.str()));
    solver.copy_device_to_host();
    // Air of X = {O2 = 1, N2 = 3.76} has Y_O2 = 0.2330, as the field's 0.233 to 4 digits
    for (uint32_t c = 0; c < solver.get_mesh()->n_owned(); c++) {
        EXPECT_NEAR(field(solver, derived::MIXTURE_FRACTION, c), mass_fraction(solver, "H2", c), 1e-4) << c;
    }
}

TEST(DerivedFields, PremixedFlameIndexAndProgressVariable) {
    // Reactants H2 + O2 burning to H2O along x: both reactant gradients point
    // the same way (flame index +1), and c = Y_H2O / 0.26 (about x); T-based c of
    // T = 400 K between 300 and 500 K is 1/2
    const std::string Y = "{ H2 = \"0.03 * (1 - x)\", O2 = \"0.23 * (1 - x)\", H2O = \"0.26 * x\", N2 = 0.74 }";
    auto solver = init_mixture(Y,
                               "[output.mixture]\nfuel_species = \"H2\"\noxidizer_species = \"O2\"\n"
                               "progress = {species = [\"H2O\"], burnt = 0.26}\n",
                               "\"FLAME_INDEX\", \"PROGRESS_VARIABLE\"");
    for (uint32_t c = 0; c < solver->get_mesh()->n_owned(); c++) {
        EXPECT_NEAR(field(*solver, derived::FLAME_INDEX, c), 1.0, roundoff(1e-12)) << c;
        EXPECT_NEAR(field(*solver, derived::PROGRESS_VARIABLE, c), mass_fraction(*solver, "H2O", c) / 0.26,
                    roundoff(1e-12))
            << c;
    }
    auto by_T = init_mixture(Y, "[output.mixture]\nprogress = {T_unburnt = 300.0, T_burnt = 500.0}\n",
                             "\"PROGRESS_VARIABLE\"");
    // T comes from e, about -3.5e6 J/kg of H2O formation against cv T of 4e5 J/kg: its round-off, plus the
    // 1e-10 T of the T_from_e iteration, over the 200 K of T_burnt - T_unburnt
    expect_everywhere(*by_T, derived::PROGRESS_VARIABLE, 0.5, 1e-9 + roundoff(1e-14));
}

TEST(DerivedFields, ScalarDissipationIsTwiceTheDiffusivityForAUnitGradient) {
    // Fuel H2 + AR of the molar mass of the oxidizer O2, so that every blend has
    // the same density and Z = 1 - x is linear, |grad Z| = 1; unity Lewis numbers
    // make every species' D the D = lambda / (rho cp) of chi = 2 D |grad Z|^2
    const double W_H2 = 2.016, W_O2 = 31.998, W_AR = 39.95;
    const double a = (1.0 / W_O2 - 1.0 / W_AR) / (1.0 / W_H2 - 1.0 / W_AR);
    const std::string A = std::to_string(a), B = std::to_string(1.0 - a);
    auto solver = init_mixture("{ H2 = \"" + A + " * (1 - x)\", AR = \"" + B + " * (1 - x)\", O2 = \"x\" }",
                               "[output.mixture]\nfuel = {Y = {H2 = " + A + ", AR = " + B + "}}\noxidizer = {Y = {O2 = 1.0}}\n",
                               "\"SCALAR_DISSIPATION\", \"D_N2\"", true, "mallard_derived_chi");
    solver->write_output_of_state(false);
    const std::string file = (rank_scratch_dir() / "mallard_derived_chi_000000.vtu").string();
    const output::Contents contents = output::open(file);
    const std::vector<double> chi = output::read(contents, "SCALAR_DISSIPATION", 0);
    const std::vector<double> D = output::read(contents, "D_N2", 0);
    ASSERT_EQ(chi.size(), D.size());
    uint32_t checked = 0;
    for (uint32_t c = 0; c < solver->get_mesh()->n_owned(); c++) {
        if (!interior(*solver, c)) continue;
        EXPECT_NEAR(chi[c], 2.0 * D[c], roundoff(1e-11) * D[c]) << c;
        checked++;
    }
    EXPECT_GT(checked, 0u);
}

TEST(DerivedFields, MixtureEntropyOfMixingAndPressure) {
    // s(T, p) of an ideal mixture: the mass-weighted entropies of its pure
    // species at the same T and p plus R ln 2 of the equimolar mixing, and
    // s(2 p) = s(p) - R ln 2, R = p / (rho T) of each gas; argon at 1 atm and
    // 298.15 K has 154.74 J/(mol K) (NIST: 154.846 at 1 bar)
    struct Gas {
        double s, R, Y_AR;
    };
    const auto gas = [](const std::string & X, const std::string & p) {
        std::ostringstream s;
        s << "[run]\nn_steps = 10\ncfl = 0.25\n[mesh]\n"
          << (N_DIM == 2 ? "type = \"cartesian\"\nNx = 2\nNy = 2\nLx = 1.0\nLy = 1.0\n"
                         : "type = \"cartesian\"\nNx = 2\nNy = 2\nNz = 2\nLx = 1.0\nLy = 1.0\nLz = 1.0\n")
          << "[initialize]\ntype = \"analytical\"\np = \"" << p << "\"\nT = \"298.15\"\nu = "
          << velocity("0.0", "0.0", "0.0") << "\nX = " << X << "\n";
        for (int k = 0; k < 2 * N_DIM; k++) s << "[[boundaries]]\nname = \"" << ZONES[k] << "\"\ntype = \"symmetry\"\n";
        s << "[numerics]\nriemann_solver = \"HLLC\"\n[numerics.face_reconstruction]\ntype = \"FO\"\n"
          << "[physics]\ntype = \"euler\"\ngas = \"mixture\"\nmechanism = \"" << source_dir() << "/mechanisms/h2o2.yaml\"\n"
          << "[output]\ncheck_interval = 1000000\n[[write_data]]\nprefix = \""
          << (rank_scratch_dir() / "mallard_derived_entropy").string()
          << "\"\nformat = \"vtu\"\ninterval = 1000000\nfields = [\"ENTROPY\"]\n";
        Solver solver;
        solver.set_distributed(false);
        solver.init(parse_toml(s.str()));
        solver.copy_device_to_host();
        const double rho = double(solver.h_conservatives(0, 0));
        const double R = double(solver.h_primitives(0, N_DIM)) / (rho * double(solver.h_primitives(0, N_DIM + 1)));
        return Gas{field(solver, derived::ENTROPY, 0), R, mass_fraction(solver, "AR", 0)};
    };
    const Gas Ar = gas("{ AR = 1.0 }", "101325.0"), N2 = gas("{ N2 = 1.0 }", "101325.0");
    EXPECT_NEAR(Ar.s * 8.314462618 / Ar.R, 154.74, 0.02);
    const Gas mix = gas("{ AR = 1.0, N2 = 1.0 }", "101325.0");
    EXPECT_NEAR(mix.s, mix.Y_AR * Ar.s + (1.0 - mix.Y_AR) * N2.s + mix.R * std::log(2.0), roundoff(1e-12) * Ar.s);
    EXPECT_NEAR(gas("{ AR = 1.0 }", "202650.0").s, Ar.s - Ar.R * std::log(2.0), roundoff(1e-12) * Ar.s);
}

TEST(DerivedFields, MixtureFieldsNeedTheirInput) {
    // Each names what is missing: the streams, or navier_stokes for the diffusivity
    const std::string Y = "{ H2 = \"1 - x\", O2 = \"0.233 * x\", N2 = \"0.767 * x\" }";
    const auto message = [&](const std::string & mixture, const std::string & variables) {
        try {
            init_mixture(Y, mixture, variables);
        } catch (const InputError & e) {
            return std::string(e.what());
        }
        return std::string("accepted");
    };
    EXPECT_NE(message("", "\"MIXTURE_FRACTION\"").find("[output.mixture] fuel"), std::string::npos);
    EXPECT_NE(message(H2_AIR, "\"SCALAR_DISSIPATION\"").find("navier_stokes"), std::string::npos);
    EXPECT_NE(message("", "\"FLAME_INDEX\"").find("fuel_species"), std::string::npos);
    EXPECT_NE(message("[output.mixture]\nfuel = \"nowhere\"\noxidizer = {X = {O2 = 1.0}}\n", "\"MIXTURE_FRACTION\"")
                  .find("no [[boundaries]] entry is named \"nowhere\""),
              std::string::npos);
}

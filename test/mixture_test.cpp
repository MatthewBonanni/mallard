/**
 * @file mixture_test.cpp
 * @author Matthew Bonanni (mbonanni001@gmail.com)
 * @brief Non-reacting gas mixtures: agreement with the single-gas solver,
 *        species advection, the multicomponent shock tube and contacts.
 * @version 0.3
 * @date 2026-10-02
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 *
 */

#include <gtest/gtest.h>
#include <Kokkos_Core.hpp>

#include <algorithm>
#include <array>
#include <iomanip>
#include <iostream>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <tuple>
#include <vector>

#include "solver.h"
#include "test_fixtures.h"

namespace {

const std::string SOURCE_DIR = MALLARD_SOURCE_DIR;
const std::string PERFECT_AIR = SOURCE_DIR + "/test/data/chemistry/perfect_air.yaml";
const std::string H2O2 = SOURCE_DIR + "/mechanisms/h2o2.yaml";

/** @brief A tolerance for double builds, or its counterpart for float builds. */
constexpr double tol(double in_double, double in_float) {
    return sizeof(rtype) == sizeof(double) ? in_double : in_float;
}

/** @brief Generated box [0, L]^N_DIM with n cells per direction (the first can differ). */
std::string mesh_block(const std::string & type, uint32_t nx, uint32_t n, double Lx, double L,
                       const std::string & periodic = "") {
    std::ostringstream s;
    s << "[mesh]\ntype = \"" << type << "\"\nNx = " << nx << "\nNy = " << n << "\nLx = " << Lx << "\nLy = " << L << "\n";
    if (N_DIM == 3) s << "Nz = " << n << "\nLz = " << L << "\n";
    if (!periodic.empty()) s << "periodic = " << periodic << "\n";
    return s.str();
}

/** @brief Boundary conditions for left, right, bottom, top (and back, front in 3D: symmetry). */
std::string boundaries(const std::string & left, const std::string & right, const std::string & bottom,
                       const std::string & top) {
    std::ostringstream s;
    const std::pair<const char *, std::string> zones[] = {{"left", left},     {"right", right},
                                                          {"bottom", bottom}, {"top", top},
                                                          {"back", "type = \"symmetry\"\n"},
                                                          {"front", "type = \"symmetry\"\n"}};
    for (int k = 0; k < 2 * N_DIM; k++) s << "[[boundaries]]\nname = \"" << zones[k].first << "\"\n" << zones[k].second;
    return s.str();
}

/** @brief A velocity vector of expressions, zero beyond the given components. */
std::string velocity(const std::string & ux, const std::string & uy = "0.0") {
    return N_DIM == 2 ? "[\"" + ux + "\", \"" + uy + "\"]" : "[\"" + ux + "\", \"" + uy + "\", \"0.0\"]";
}

std::string perfect_air() {
    // R = p_ref / (rho_ref T_ref) = 287
    return "[physics]\ntype = \"euler\"\ngamma = 1.4\np_ref = 101325.0\nT_ref = 300.0\nrho_ref = 1.1768292682926829\n";
}

std::string mixture(const std::string & mechanism, const std::string & phase = "",
                    const std::string & type = "euler") {
    return "[physics]\ntype = \"" + type + "\"\ngas = \"mixture\"\nmechanism = \"" + mechanism + "\"\n" +
           (phase.empty() ? "" : "phase = \"" + phase + "\"\n");
}

std::string numerics(const std::string & reconstruction, const std::string & riemann, bool double_flux = false) {
    return "[numerics]\nriemann_solver = \"" + riemann + "\"\n" + (double_flux ? "double_flux = true\n" : "") +
           "[numerics.face_reconstruction]\n" + reconstruction + "[output]\ncheck_interval = 1000000\n";
}

/** @brief Max |a - b| over cells and conservatives, relative to each variable's max |b|. */
double max_relative_difference(const Solver & a, const Solver & b) {
    double scale[N_CONSERVATIVE] = {}, diff[N_CONSERVATIVE] = {};
    FOR_I_CONSERVATIVE {
        for (uint32_t c = 0; c < b.get_mesh()->n_cells; c++) {
            scale[i] = std::max(scale[i], std::abs(static_cast<double>(b.h_conservatives(c, i))));
            diff[i] = std::max(diff[i], std::abs(static_cast<double>(a.h_conservatives(c, i) - b.h_conservatives(c, i))));
        }
    }
    // Momentum components share one scale, so that a component that stays near zero is not magnified
    double momentum = 0.0;
    FOR_I_DIM momentum = std::max(momentum, scale[1 + i]);
    FOR_I_DIM scale[1 + i] = momentum;
    double worst = 0.0;
    FOR_I_CONSERVATIVE worst = std::max(worst, diff[i] / scale[i]);
    return worst;
}

struct EquivalenceCase {
    std::string mesh;
    std::string reconstruction;
    std::string riemann;
    bool double_flux = false;
};

void PrintTo(const EquivalenceCase & c, std::ostream * os) {
    *os << c.mesh << " " << c.riemann << (c.double_flux ? " double flux" : "");
}

class MixtureEquivalence : public ::testing::TestWithParam<EquivalenceCase> {};

} // namespace

TEST_P(MixtureEquivalence, ConstantCpSingleSpeciesReproducesPerfectGas) {
    // A one-species mixture with constant cp = 3.5 R and e = cv T is the
    // perfect gas computed with different arithmetic (T from e by Newton,
    // p = rho R T, gamma = cp / cv, e0 ~ 0): it must give the same flow up to
    // round-off, with every boundary condition mixtures support
    const EquivalenceCase & c = GetParam();
    const uint32_t n = N_DIM == 2 ? 16 : 6;
    const std::string common =
        "[run]\nn_steps = 30\ncfl = 0.5\n" + mesh_block(c.mesh, n, n, 1.0, 1.0) +
        "[initialize]\ntype = \"analytical\"\np = \"x < 0.5 and y < 0.5 ? 1.0e5 : 3.0e4\"\n"
        "T = \"x < 0.5 ? 400.0 : 300.0\"\nu = " + velocity("x > 0.5 ? 50.0 : 0.0", "y > 0.6 ? -30.0 : 0.0") + "\n";
    const std::string scheme = numerics(c.reconstruction, c.riemann);
    const std::string bcs = boundaries(
        "type = \"upt\"\nu = " + std::string(N_DIM == 2 ? "[20.0, 0.0]" : "[20.0, 0.0, 0.0]") +
            "\np = 1.0e5\nT = 400.0\nY = { AIR = 1.0 }\n",
        "type = \"p_out\"\np = 3.0e4\n", "type = \"wall_adiabatic\"\n", "type = \"symmetry\"\n");
    Solver perfect;
    perfect.init(parse_toml(common + scheme + bcs + perfect_air()));
    perfect.run();
    perfect.copy_device_to_host();
    Solver mixed;
    // With one species the frozen gamma and e0 of double flux are the true ones
    mixed.init(parse_toml(common + "Y = { AIR = 1.0 }\n" + numerics(c.reconstruction, c.riemann, c.double_flux) + bcs +
                          mixture(PERFECT_AIR, "air")));
    mixed.run();
    mixed.copy_device_to_host();

    const double t = double(perfect.get_time());
    EXPECT_NEAR(double(mixed.get_time()), t, tol(1e-12, 1e-5) * t);
    EXPECT_LT(max_relative_difference(mixed, perfect), tol(1e-12, 1e-4));
    for (uint32_t cell = 0; cell < perfect.get_mesh()->n_cells; cell++) {
        const double T = double(perfect.h_primitives(cell, N_DIM + 1));
        const double rho = double(mixed.h_conservatives(cell, 0));
        EXPECT_NEAR(double(mixed.h_primitives(cell, N_DIM + 1)), T, tol(1e-10, 1e-4) * T);
        EXPECT_NEAR(double(mixed.h_species(cell, 0)), rho, tol(1e-14, 1e-6) * rho);
    }
}

INSTANTIATE_TEST_SUITE_P(
    Mixture, MixtureEquivalence,
    ::testing::Values(EquivalenceCase{N_DIM == 2 ? "cartesian_tri" : "cartesian_tet", "type = \"MUSCL\"\n", "HLLC"},
                      EquivalenceCase{"cartesian", "type = \"FO\"\n", "Rusanov"},
                      EquivalenceCase{N_DIM == 2 ? "cartesian_tri" : "cartesian_tet",
                                      "type = \"MUSCL\"\nlimiter = \"barth_jespersen\"\n", "HLL"},
                      EquivalenceCase{N_DIM == 2 ? "cartesian_tri" : "cartesian_tet", "type = \"MUSCL\"\n", "HLLC",
                                      true},
                      EquivalenceCase{N_DIM == 2 ? "cartesian_tri" : "cartesian_tet", "type = \"MUSCL\"\n", "Roe"},
                      EquivalenceCase{N_DIM == 2 ? "cartesian_tri" : "cartesian_tet", "type = \"MUSCL\"\n", "RHLL",
                                      true}));

namespace {

/** @brief W = [rho, u, p], the velocity's third component (3D) made from the first two. */
std::array<rtype, N_CONSERVATIVE> primitive(double rho, double ux, double uy, double p) {
    std::array<rtype, N_CONSERVATIVE> W;
    W[0] = rtype(rho);
    W[1] = rtype(ux);
    W[2] = rtype(uy);
    if constexpr (N_DIM == 3) W[3] = rtype(0.5 * uy - 0.2 * ux);
    W[N_DIM + 1] = rtype(p);
    return W;
}

/** @brief The face normal along x and an oblique one. */
std::vector<std::array<rtype, N_DIM>> unit_normals() {
    std::array<rtype, N_DIM> x{}, oblique{};
    x[0] = 1.0;
    const rtype scale = N_DIM == 2 ? 1.0 : 0.8;
    oblique[0] = 0.6 * scale;
    oblique[1] = -0.8 * scale;
    if constexpr (N_DIM == 3) oblique[N_DIM - 1] = 0.6;
    return {x, oblique};
}

/** @brief A unit vector orthogonal to n. */
std::array<rtype, N_DIM> tangent(const std::array<rtype, N_DIM> & n) {
    std::array<rtype, N_DIM> t{};
    const rtype norm = std::sqrt(n[0] * n[0] + n[1] * n[1]);
    t[0] = n[1] / norm;
    t[1] = -n[0] / norm;
    return t;
}

template <typename T_solver>
void expect_flux(const rtype * n, const std::array<rtype, N_CONSERVATIVE> & W_l,
                 const std::array<rtype, N_CONSERVATIVE> & W_r, const riemann::SideThermo & th_l,
                 const riemann::SideThermo & th_r, const rtype * expected, double tolerance, const char * what) {
    rtype f[N_CONSERVATIVE];
    T_solver::calc_flux(f, n, W_l.data(), W_r.data(), th_l, th_r);
    double scale = 0.0;
    FOR_I_CONSERVATIVE scale = std::max(scale, std::abs(double(expected[i])));
    FOR_I_CONSERVATIVE {
        EXPECT_NEAR(double(f[i]), double(expected[i]), tolerance * scale) << what << ", component " << int(i);
    }
}

} // namespace

TEST(MixtureRiemannTest, RoeAndRHLLWithOneGammaAndNoOffsetAreThePerfectGasSolvers) {
    // Shock tube, shear with a contact, and a transonic rarefaction (where
    // the entropy fix acts): the mixture forms average gamma and e0 and
    // take the jump form, the perfect-gas Roe solver its eigenvectors
    const rtype gamma = 1.4;
    const riemann::SideThermo th{gamma, 0.0};
    const std::array<std::array<rtype, N_CONSERVATIVE>, 2> cases[] = {
        {primitive(1.0, 0.0, 0.0, 1.0), primitive(0.125, 0.0, 0.0, 0.1)},
        {primitive(1.0, 0.3, 0.5, 1.0), primitive(0.4, 0.3, -0.2, 0.6)},
        {primitive(1.0, 0.9, 0.1, 1.0), primitive(0.8, 1.15, -0.1, 0.7)}};
    for (const auto & n : unit_normals()) {
        for (const auto & [W_l, W_r] : cases) {
            rtype f[N_CONSERVATIVE];
            riemann::Roe::calc_flux(f, n.data(), W_l.data(), W_r.data(), gamma);
            expect_flux<riemann::Roe>(n.data(), W_l, W_r, th, th, f, tol(1e-13, 1e-5), "Roe");
            riemann::RHLL::calc_flux(f, n.data(), W_l.data(), W_r.data(), gamma);
            expect_flux<riemann::RHLL>(n.data(), W_l, W_r, th, th, f, tol(1e-13, 1e-5), "RHLL");
        }
    }
}

TEST(MixtureRiemannTest, RoeUpwindsAContactBetweenDifferentGasesExactly) {
    // Equal p and u, different gamma and e0: whatever the averages, the flux
    // is the upwind side's; with a slip as well, RHLL is Roe along the face
    // normal
    const riemann::SideThermo th_l{1.4, 0.0}, th_r{1.25, 2.0};
    for (const auto & n : unit_normals()) {
        const std::array<rtype, N_DIM> t = tangent(n);
        for (const double sign : {1.0, -1.0}) {
            const std::array<rtype, N_CONSERVATIVE> W_l = primitive(1.0, sign * 0.7, sign * 0.2, 1.0);
            std::array<rtype, N_CONSERVATIVE> W_r = primitive(0.1, 0.0, 0.0, 1.0);
            FOR_I_DIM W_r[1 + i] = W_l[1 + i];
            rtype U[N_CONSERVATIVE], F[N_CONSERVATIVE];
            if (sign > 0) {
                riemann::physical_flux(W_l.data(), n.data(), th_l, U, F);
            } else {
                riemann::physical_flux(W_r.data(), n.data(), th_r, U, F);
            }
            expect_flux<riemann::Roe>(n.data(), W_l, W_r, th_l, th_r, F, tol(1e-14, 1e-6), "Roe");
            FOR_I_DIM W_r[1 + i] = W_l[1 + i] + 0.5_r * t[i];
            if (sign < 0) riemann::physical_flux(W_r.data(), n.data(), th_r, U, F);
            expect_flux<riemann::Roe>(n.data(), W_l, W_r, th_l, th_r, F, tol(1e-14, 1e-6), "Roe with slip");
            expect_flux<riemann::RHLL>(n.data(), W_l, W_r, th_l, th_r, F, tol(1e-14, 1e-6), "RHLL with slip");
        }
    }
}

TEST(MixtureRiemannTest, RoeAndRHLLAreUpwindOnSupersonicFacesBetweenDifferentGases) {
    // All waves move along n (for RHLL also along both of its directions,
    // the velocity jump being at 45 degrees to n), so the Roe property makes
    // the flux F_l exactly: only if the waves carry the whole jump of U,
    // including the energy of the jump in gamma and e0
    const riemann::SideThermo th_l{1.4, 0.0}, th_r{1.2, 1.5};
    for (const auto & n : unit_normals()) {
        const std::array<rtype, N_DIM> t = tangent(n);
        std::array<rtype, N_CONSERVATIVE> W_l = primitive(1.0, 0.0, 0.0, 1.0), W_r = primitive(0.5, 0.0, 0.0, 0.8);
        FOR_I_DIM {
            W_l[1 + i] = 5.0_r * n[i] + 0.3_r * t[i];
            W_r[1 + i] = 4.6_r * n[i] - 0.1_r * t[i];
        }
        rtype U[N_CONSERVATIVE], F[N_CONSERVATIVE];
        riemann::physical_flux(W_l.data(), n.data(), th_l, U, F);
        expect_flux<riemann::Roe>(n.data(), W_l, W_r, th_l, th_r, F, tol(1e-13, 1e-5), "Roe");
        expect_flux<riemann::RHLL>(n.data(), W_l, W_r, th_l, th_r, F, tol(1e-13, 1e-5), "RHLL");
    }
}

TEST(MixtureTest, IdenticalSpeciesAdvectWithoutDisturbingTheFlow) {
    // Two species with identical thermodynamics: the flow stays exactly
    // uniform, the species are carried by the mass flux, sum to one and
    // stay in [0, 1], and each species' mass is conserved
    const uint32_t n = N_DIM == 2 ? 20 : 8;
    const std::string periodic = N_DIM == 2 ? "[\"x\", \"y\"]" : "[\"x\", \"y\", \"z\"]";
    const std::string input =
        "[run]\nn_steps = " + std::to_string(N_DIM == 2 ? 40 : 100) + "\ncfl = 0.5\n" +
        mesh_block(N_DIM == 2 ? "cartesian_tri" : "cartesian_tet", n, n, 1.0, 1.0, periodic) +
        "[initialize]\ntype = \"analytical\"\np = \"1.0e5\"\nT = \"300.0\"\nu = " + velocity("100.0", "50.0") +
        "\nY = { AIR2 = \"(x - 0.5)^2 + (y - 0.5)^2 < 0.06 ? 1.0 : 0.0\" }\nbalance = \"AIR\"\n" +
        numerics("type = \"MUSCL\"\n", "HLLC") + mixture(PERFECT_AIR, "two-airs");
    Solver solver;
    solver.init(parse_toml(input));
    const std::vector<rtype> before = solver.integrate_species();
    auto centroid = [&](Solver & s) {
        s.copy_device_to_host();
        double m = 0.0, mx = 0.0;
        for (uint32_t c = 0; c < s.get_mesh()->n_cells; c++) {
            const double w = double(s.h_species(c, 1)) * double(s.get_mesh()->h_cell_volume(c));
            m += w;
            mx += w * double(s.get_mesh()->h_cell_coords(c, 0));
        }
        return mx / m;
    };
    const double x_before = centroid(solver);
    solver.run();
    const std::vector<rtype> after = solver.integrate_species();
    const double x_after = centroid(solver);

    for (size_t k = 0; k < before.size(); k++) {
        EXPECT_NEAR(double(after[k]), double(before[k]), tol(1e-13, 1e-5) * double(before[k]));
    }
    EXPECT_NEAR(x_after - x_before, 100.0 * double(solver.get_time()), 0.1 / n);
    EXPECT_GT(x_after - x_before, 0.5 / n);
    for (uint32_t c = 0; c < solver.get_mesh()->n_cells; c++) {
        const double rho = double(solver.h_conservatives(c, 0));
        EXPECT_NEAR(double(solver.h_primitives(c, N_DIM)), 1.0e5, tol(1e-10, 1e-5) * 1.0e5);
        EXPECT_NEAR(double(solver.h_primitives(c, 0)), 100.0, tol(1e-10, 1e-5) * 100.0);
        EXPECT_NEAR(double(solver.h_primitives(c, 1)), 50.0, tol(1e-10, 1e-5) * 100.0);
        double sum = 0.0;
        for (uint32_t k = 0; k < 2; k++) {
            const double Y = double(solver.h_species(c, k)) / rho;
            EXPECT_GE(Y, -tol(1e-15, 1e-7));
            EXPECT_LE(Y, 1.0 + tol(1e-15, 1e-7));
            sum += Y;
        }
        EXPECT_NEAR(sum, 1.0, tol(1e-13, 1e-5));
    }
}

TEST(MixtureTest, UptCompositionExpressionsGiveAStratifiedInflow) {
    // An inlet whose composition varies along it (an expression and a balance
    // species): with two species of identical thermodynamics the flow stays
    // uniform and, once the inflow has crossed the channel, each cell carries
    // the composition of the inlet face at its height
    const uint32_t nx = 12, n = N_DIM == 2 ? 8 : 4;
    const std::string input =
        "[run]\nt_stop = 0.06\ncfl = 0.5\n" + mesh_block("cartesian", nx, n, 1.0, 1.0) +
        "[initialize]\ntype = \"constant\"\np = 1.0e5\nT = 300.0\nu = " +
        (N_DIM == 2 ? "[50.0, 0.0]" : "[50.0, 0.0, 0.0]") + "\nY = { AIR = 1.0 }\n" +
        numerics("type = \"MUSCL\"\n", "HLLC") + mixture(PERFECT_AIR, "two-airs") +
        boundaries("type = \"upt\"\nu = " + std::string(N_DIM == 2 ? "[50.0, 0.0]" : "[50.0, 0.0, 0.0]") +
                       "\np = 1.0e5\nT = 300.0\nY = { AIR2 = \"0.8 * y^2\" }\nbalance = \"AIR\"\n",
                   "type = \"p_out\"\np = 1.0e5\n", "type = \"symmetry\"\n", "type = \"symmetry\"\n");
    Solver solver;
    solver.init(parse_toml(input));
    solver.run();
    solver.copy_device_to_host();
    for (uint32_t c = 0; c < solver.get_mesh()->n_cells; c++) {
        const double rho = double(solver.h_conservatives(c, 0));
        const double y = double(solver.get_mesh()->h_cell_coords(c, 1));
        EXPECT_NEAR(double(solver.h_species(c, 1)) / rho, 0.8 * y * y, tol(1e-6, 1e-5));
        EXPECT_NEAR(double(solver.h_primitives(c, N_DIM)), 1.0e5, tol(1e-8, 1e-3));
    }

    Solver bad;
    EXPECT_THROW(bad.init(parse_toml(input.substr(0, input.find("Y = { AIR2")) + "Y = { AIR2 = \"0.5 *\" }\n" +
                                     input.substr(input.find("balance = \"AIR\"")))),
                 std::runtime_error);
}

namespace {

std::vector<std::vector<double>> read_reference(const std::string & file, std::vector<std::string> & columns) {
    std::ifstream in(SOURCE_DIR + "/test/data/chemistry/" + file);
    std::string line;
    std::vector<std::vector<double>> rows;
    while (std::getline(in, line)) {
        if (line.empty() || line[0] == '#') continue;
        std::stringstream s(line);
        std::string field;
        std::vector<std::string> fields;
        while (std::getline(s, field, ',')) fields.push_back(field);
        if (columns.empty()) {
            columns = fields;
            continue;
        }
        std::vector<double> row;
        for (const auto & f : fields) row.push_back(std::stod(f));
        rows.push_back(row);
    }
    return rows;
}

/**
 * @brief L1 errors (relative to the reference maximum) of rho, u, p, T and
 *        Y_N2 against the exact solution of the multicomponent shock tube
 *        (tools/mixture_riemann.py), and the extreme mass fractions.
 */
std::vector<double> shock_tube_errors(uint32_t n, const std::string & reconstruction, const std::string & riemann,
                                      double & Y_min, double & Y_max) {
    // TENO's stencils need a few rows of cells
    const uint32_t rows = reconstruction.find("TENO") != std::string::npos ? 4 : 1;
    const std::string input =
        "[run]\nt_stop = 2.0e-4\ncfl = 0.5\n" + mesh_block("cartesian", n, rows, 1.0, rows * 1.0 / n) +
        "[initialize]\ntype = \"analytical\"\np = \"x < 0.5 ? 1.0e5 : 1.0e4\"\nT = \"x < 0.5 ? 1000.0 : 300.0\"\n"
        "u = " + velocity("0.0") + "\n"
        "X = { H2 = \"x < 0.5 ? 2 : 0\", O2 = \"x < 0.5 ? 1 : 0\", AR = \"x < 0.5 ? 7 : 0\", N2 = \"x < 0.5 ? 0 : 1\" }\n" +
        boundaries("type = \"extrapolation\"\n", "type = \"extrapolation\"\n", "type = \"symmetry\"\n",
                   "type = \"symmetry\"\n") +
        numerics(reconstruction, riemann) + mixture(H2O2);
    Solver solver;
    solver.init(parse_toml(input));
    solver.run();
    solver.update_primitives();
    solver.copy_device_to_host();
    std::vector<std::string> columns;
    const auto ref = read_reference("shock_tube_" + std::to_string(n) + ".csv", columns);
    EXPECT_EQ(ref.size(), n);
    const uint32_t n_cells = solver.get_mesh()->n_cells;
    const auto & names = solver.get_species_names();
    const size_t k_N2 = std::find(names.begin(), names.end(), "N2") - names.begin();
    std::vector<double> err(5, 0.0), scale(5, 0.0);
    Y_min = 1.0;
    Y_max = 0.0;
    for (uint32_t c = 0; c < n_cells; c++) {
        const uint32_t i = static_cast<uint32_t>(double(solver.get_mesh()->h_cell_coords(c, 0)) * n);
        const double rho = double(solver.h_conservatives(c, 0));
        const double values[5] = {rho, double(solver.h_primitives(c, 0)), double(solver.h_primitives(c, N_DIM)),
                                  double(solver.h_primitives(c, N_DIM + 1)), double(solver.h_species(c, k_N2)) / rho};
        const double exact[5] = {ref[i][1], ref[i][2], ref[i][3], ref[i][4], ref[i][5 + k_N2]};
        for (int j = 0; j < 5; j++) {
            err[j] += std::abs(values[j] - exact[j]) / n_cells;
            scale[j] = std::max(scale[j], std::abs(exact[j]));
        }
        for (size_t k = 0; k < names.size(); k++) {
            Y_min = std::min(Y_min, static_cast<double>(solver.h_species(c, k)) / rho);            Y_max = std::max(Y_max, static_cast<double>(solver.h_species(c, k)) / rho);
        }
    }
    for (int j = 0; j < 5; j++) err[j] /= scale[j];
    return err;
}

} // namespace

using ShockTubeCase = std::pair<std::string, std::string>;  // reconstruction, Riemann solver

class MixtureShockTube : public ::testing::TestWithParam<ShockTubeCase> {};

TEST_P(MixtureShockTube, ConvergesToExactSolution) {
    // H2/O2/Ar at 1000 K and 1 bar against N2 at 300 K and 0.1 bar: shock,
    // contact between different gases, rarefaction, all thermally perfect
    const auto & [reconstruction, riemann] = GetParam();
    double Y_min, Y_max;
    // TENO is slower per cell, and converges as fast on coarser meshes
    const bool teno = reconstruction.find("TENO") != std::string::npos;
    const uint32_t n = teno ? 50 : 100;
    const std::vector<double> coarse = shock_tube_errors(n, reconstruction, riemann, Y_min, Y_max);
    EXPECT_GE(Y_min, -tol(1e-14, 1e-6));
    EXPECT_LE(Y_max, 1.0 + tol(1e-14, 1e-6));
    const std::vector<double> fine = shock_tube_errors(2 * n, reconstruction, riemann, Y_min, Y_max);
    EXPECT_GE(Y_min, -tol(1e-14, 1e-6));
    EXPECT_LE(Y_max, 1.0 + tol(1e-14, 1e-6));
    const char * names[5] = {"rho", "u", "p", "T", "Y_N2"};
    for (int j = 0; j < 5; j++) {
        EXPECT_LT(fine[j], 1.2e-2) << names[j];
        EXPECT_GT(std::log2(coarse[j] / fine[j]), teno ? 0.5 : 0.7) << names[j] << ": " << coarse[j] << " -> " << fine[j];
    }
    // Roe and RHLL resolve this 1D problem as sharply as HLLC (within 1% and 4%)
    if (riemann != "HLLC") {
        const std::vector<double> hllc = shock_tube_errors(2 * n, reconstruction, "HLLC", Y_min, Y_max);
        for (int j = 0; j < 5; j++) EXPECT_LT(fine[j], 1.1 * hllc[j]) << names[j];
    }
}

// TENO5 in 2D only: its 3D stencils make the test too slow for the suite
INSTANTIATE_TEST_SUITE_P(
    Mixture, MixtureShockTube,
    ::testing::ValuesIn(N_DIM == 2 ? std::vector<ShockTubeCase>{{"type = \"MUSCL\"\n", "HLLC"},
                                                                {"type = \"TENO\"\norder = 5\n", "HLLC"},
                                                                {"type = \"MUSCL\"\n", "Roe"},
                                                                {"type = \"MUSCL\"\n", "RHLL"}}
                                   : std::vector<ShockTubeCase>{{"type = \"MUSCL\"\n", "HLLC"},
                                                                {"type = \"MUSCL\"\n", "Roe"},
                                                                {"type = \"MUSCL\"\n", "RHLL"}}));

namespace {

/**
 * @brief L1 errors of rho and rho Y_H2 after advecting a smooth composition
 *        and temperature wave at uniform pressure across a periodic box over
 *        n steps of a time step proportional to the mesh size, against the
 *        exact (translated) cell averages.
 */
std::array<double, 2> advection_errors(const std::string & mesh, uint32_t n, const std::string & reconstruction) {
    constexpr double t_end = 2.0e-4, u = 100.0, v = 50.0;
    const std::string periodic = N_DIM == 2 ? "[\"x\", \"y\"]" : "[\"x\", \"y\", \"z\"]";
    auto input = [&](uint32_t n_steps, double dx, double dy) {
        std::ostringstream x, y;
        x << std::setprecision(17) << "(x - " << dx << ")";
        y << std::setprecision(17) << "(y - " << dy << ")";
        std::ostringstream dt;
        dt << std::setprecision(17) << t_end / n;
        return "[run]\nn_steps = " + std::to_string(n_steps) + "\ndt = " + dt.str() + "\n" +
               mesh_block(mesh, n, n, 1.0, 1.0, periodic) +
               "[initialize]\ntype = \"analytical\"\nn_subdivisions = " + std::string(N_DIM == 2 ? "2" : "1") +
               "\np = \"1.0e5\"\nT = \"300.0 + 50.0 * sin(2 * pi * (" + x.str() +
               " + " + y.str() + "))\"\nu = " + velocity("100.0", "50.0") + "\nX = { H2 = \"0.3 + 0.05 * sin(2 * pi * " +
               x.str() + ") * sin(2 * pi * " + y.str() + ")\" }\nbalance = \"N2\"\n" + numerics(reconstruction, "HLLC") +
               mixture(H2O2);
    };
    Solver solver;
    solver.init(parse_toml(input(n, 0.0, 0.0)));
    solver.run();
    solver.copy_device_to_host();
    Solver exact;
    exact.init(parse_toml(input(1, u * t_end, v * t_end)));
    const auto & names = solver.get_species_names();
    const size_t k = std::find(names.begin(), names.end(), "H2") - names.begin();
    std::array<double, 2> err = {0.0, 0.0};
    const uint32_t n_cells = solver.get_mesh()->n_cells;
    for (uint32_t c = 0; c < n_cells; c++) {
        err[0] += std::abs(double(solver.h_conservatives(c, 0) - exact.h_conservatives(c, 0))) / n_cells;
        err[1] += std::abs(double(solver.h_species(c, k) - exact.h_species(c, k))) / n_cells;
    }
    return err;
}

struct OrderCase {
    std::string mesh;
    std::string reconstruction;
    uint32_t n;
    double order;
};

void PrintTo(const OrderCase & c, std::ostream * os) { *os << c.mesh << " " << c.reconstruction; }

class MixtureAdvectionOrder : public ::testing::TestWithParam<OrderCase> {};

} // namespace

TEST_P(MixtureAdvectionOrder, SmoothCompositionWaveConvergesAtDesignOrder) {
    // The flow block (primitive variables) and the species (the flow block's
    // weights) both reach the scheme's order on a smooth contact wave
    const OrderCase & c = GetParam();
    const std::array<double, 2> coarse = advection_errors(c.mesh, c.n, c.reconstruction);
    const std::array<double, 2> fine = advection_errors(c.mesh, 2 * c.n, c.reconstruction);
    EXPECT_GT(std::log2(coarse[0] / fine[0]), c.order) << "rho: " << coarse[0] << " -> " << fine[0];
    EXPECT_GT(std::log2(coarse[1] / fine[1]), c.order) << "rho Y_H2: " << coarse[1] << " -> " << fine[1];
}

// TENO5 in 2D only: on the coarse 3D meshes a test can afford, its large stencils
// see the wave as under-resolved and flag every cell troubled
INSTANTIATE_TEST_SUITE_P(
    Mixture, MixtureAdvectionOrder,
    ::testing::ValuesIn(N_DIM == 2
                            ? std::vector<OrderCase>{{"cartesian", "type = \"MUSCL\"\n", 16u, 1.8},
                                                     {"cartesian", "type = \"TENO\"\norder = 3\n", 32u, 2.6},
                                                     {"cartesian_tri", "type = \"TENO\"\norder = 5\n", 16u, 4.0}}
                            : std::vector<OrderCase>{{"cartesian", "type = \"MUSCL\"\n", 8u, 1.8},
                                                     {"cartesian", "type = \"TENO\"\norder = 3\n", 8u, 2.4}}));

namespace {

/** @brief Max |p - p0| / p0 after a cold H2 slab crossed a quarter of a periodic hot-air channel. */
double interface_pressure_error(uint32_t n, Solver & solver, const std::string & reconstruction = "type = \"MUSCL\"\n") {
    const uint32_t rows = reconstruction.find("TENO") != std::string::npos ? 4 : 1;
    const std::string input =
        "[run]\nt_stop = 2.5e-4\ncfl = 0.5\n" + mesh_block("cartesian", n, rows, 1.0, rows * 0.02, "[\"x\"]") +
        "[initialize]\ntype = \"analytical\"\np = \"1.0e5\"\nT = \"abs(x - 0.5) < 0.25 ? 300.0 : 1000.0\"\n"
        "u = " + velocity("1000.0") + "\n"
        "X = { H2 = \"abs(x - 0.5) < 0.25 ? 1 : 0\", O2 = \"abs(x - 0.5) < 0.25 ? 0 : 0.21\", "
        "N2 = \"abs(x - 0.5) < 0.25 ? 0 : 0.79\" }\n"
        "[[boundaries]]\nname = \"bottom\"\ntype = \"symmetry\"\n[[boundaries]]\nname = \"top\"\ntype = \"symmetry\"\n" +
        std::string(N_DIM == 3 ? "[[boundaries]]\nname = \"back\"\ntype = \"symmetry\"\n"
                                 "[[boundaries]]\nname = \"front\"\ntype = \"symmetry\"\n"
                               : "") +
        numerics(reconstruction, "HLLC") + mixture(H2O2);
    solver.init(parse_toml(input));
    solver.run();
    solver.update_primitives();
    solver.copy_device_to_host();
    double err = 0.0;
    for (uint32_t c = 0; c < solver.get_mesh()->n_cells; c++) {
        err = std::max(err, std::abs(double(solver.h_primitives(c, N_DIM)) / 1.0e5 - 1.0));
    }
    return err;
}

} // namespace

TEST(MixtureTest, HydrogenAirContactPressureErrorDecreasesUnderRefinement) {
    // The conservative scheme perturbs p at contacts with a jump in gamma and
    // e0 (Abgrall 1996); the perturbation must stay at the percent level and
    // shrink as the mesh is refined, while every species' mass is conserved
    Solver coarse, fine;
    const double e_coarse = interface_pressure_error(50, coarse);
    const double e_fine = interface_pressure_error(100, fine);
    EXPECT_LT(e_coarse, 0.03);
    EXPECT_LT(e_fine, e_coarse);
    EXPECT_GT(e_fine, 1e-4);  // the conservative scheme cannot keep p exactly uniform
    const std::vector<rtype> mass = fine.integrate_species();
    double total = 0.0;
    for (rtype m : mass) total += double(m);
    EXPECT_NEAR(total, double(fine.integrate_conservatives()[0]), tol(1e-12, 1e-5) * total);
    // TENO5 keeps the interface sharper, so the error does not shrink as fast, but stays bounded
    if constexpr (N_DIM == 2) {
        Solver teno;
        EXPECT_LT(interface_pressure_error(50, teno, "type = \"TENO\"\norder = 5\n"), 0.03);
    }
}

TEST(MixtureTest, InvalidMixtureInputsAreRejected) {
    const std::string base = "[run]\nn_steps = 1\ncfl = 0.5\n" + mesh_block("cartesian", 4, 4, 1.0, 1.0) +
                             numerics("type = \"FO\"\n", "HLLC");
    const std::string bcs = boundaries("type = \"extrapolation\"\n", "type = \"extrapolation\"\n",
                                       "type = \"symmetry\"\n", "type = \"symmetry\"\n");
    const std::string init = "[initialize]\ntype = \"constant\"\np = 1.0e5\nT = 300.0\nu = " +
                             std::string(N_DIM == 2 ? "[0.0, 0.0]" : "[0.0, 0.0, 0.0]") + "\n";
    auto expect_error = [&](const std::string & input, const std::string & needle) {
        Solver solver;
        try {
            solver.init(parse_toml(input));
            ADD_FAILURE() << "accepted: " << needle;
        } catch (const std::exception & e) {
            EXPECT_NE(std::string(e.what()).find(needle), std::string::npos) << e.what();
        }
    };
    expect_error(base + init + "X = { H2 = 1.0 }\nY = { H2 = 1.0 }\n" + bcs + mixture(H2O2), "exactly one of X and Y");
    expect_error(base + init + "X = { CH4 = 1.0 }\n" + bcs + mixture(H2O2), "CH4");
    expect_error(base + init + "X = { N2 = 1.0 }\n" + bcs + mixture(PERFECT_AIR, "", "navier_stokes"), "transport data");
    expect_error(base + init + "X = { H2 = 1.0 }\n" +
                     boundaries("type = \"farfield\"\nu = " +
                                    std::string(N_DIM == 2 ? "[0.0, 0.0]" : "[0.0, 0.0, 0.0]") +
                                    "\np = 1.0e5\nT = 300.0\nX = { H2 = 1.0 }\n",
                                "type = \"extrapolation\"\n", "type = \"symmetry\"\n", "type = \"symmetry\"\n") +
                     mixture(H2O2),
                 "farfield");
    expect_error(base + init + "X = { H2 = 1.0 }\n" + bcs + mixture(H2O2) + "[chemistry]\nintegrator = \"bdf\"\n",
                 "chemistry.integrator");
    expect_error(base + init + "X = { N2 = 1.0 }\n" + bcs + mixture(PERFECT_AIR) + "[chemistry]\n", "no reactions");
    expect_error(base + init + "[chemistry]\n" + bcs + perfect_air(), "[chemistry] needs");
}

TEST(MixtureTest, RestartedRunMatchesUninterruptedRunExactly) {
    // The temperature seed is part of the restart: without it, Newton starts
    // elsewhere and the last bits of T, and so of the flow, differ (also with
    // double flux, whose frozen thermodynamics come from the seed); with
    // chemistry so is each cell's last sub-step, which seeds the integrator
    for (const auto & [double_flux, reacting, viscous] :
         {std::tuple{false, false, false}, {true, false, false}, {false, true, false}, {false, true, true}}) {
        const std::string dir = (std::filesystem::temp_directory_path() / "mallard_mixture_restart").string();
        std::filesystem::remove_all(dir);
        auto input = [&](const std::string & init, uint32_t n_steps, const std::string & prefix) {
            return "[run]\nn_steps = " + std::to_string(n_steps) + "\ncfl = 0.5\n" +
                   mesh_block(N_DIM == 2 ? "cartesian_tri" : "cartesian_tet", N_DIM == 2 ? 24 : 8, 4, 1.0, 0.2) +
                   "[initialize]\n" + init +
                   boundaries("type = \"extrapolation\"\n", "type = \"extrapolation\"\n", "type = \"symmetry\"\n",
                              "type = \"wall_adiabatic\"\n") +
                   numerics("type = \"MUSCL\"\n", "HLLC", double_flux) +
                   mixture(H2O2, "", viscous ? "navier_stokes" : "euler") +
                   (reacting ? "[chemistry]\n" : "") + "[[write_data]]\nprefix = \"" +
                   dir + "/" + prefix + "\"\nformat = \"restart\"\ninterval = 15\n";
        };
        const std::string tube =
            "type = \"analytical\"\np = \"x < 0.5 ? 1.0e5 : 1.0e4\"\nT = \"x < 0.5 ? 1000.0 : 300.0\"\nu = " +
            velocity("0.0") + "\nX = { H2 = \"x < 0.5 ? 2 : 0\", O2 = \"x < 0.5 ? 1 : 0\", N2 = \"x < 0.5 ? 0 : 1\" }\n";
        Solver straight;
        straight.init(parse_toml(input(tube, 30, "a")));
        straight.run();
        straight.copy_device_to_host();
        Solver first;
        first.init(parse_toml(input(tube, 15, "b")));
        first.run();
        Solver second;
        second.init(parse_toml(input("type = \"restart\"\nfile = \"" + dir + "/b_000015.restart\"\n", 30, "b")));
        second.run();
        second.copy_device_to_host();
        for (uint32_t c = 0; c < straight.get_mesh()->n_cells; c++) {
            FOR_I_CONSERVATIVE ASSERT_EQ(second.h_conservatives(c, i), straight.h_conservatives(c, i)) << "cell " << c;
            for (uint32_t k = 0; k < straight.get_species_names().size(); k++) {
                ASSERT_EQ(second.h_species(c, k), straight.h_species(c, k)) << "cell " << c;
            }
        }
        std::filesystem::remove_all(dir);
    }
}

namespace {

/**
 * @brief A cold H2 slab crossing a periodic channel of hot air at uniform
 *        pressure and velocity (V4): the largest relative deviations of p
 *        and u after a quarter of a flow-through. The slab's edges lie on
 *        faces, so no cell starts mixed; air at 1200 K keeps clear of the
 *        1000 K breakpoint of the NASA-7 fits, where e(T) jumps.
 */
std::array<double, 2> contact_errors(const std::string & reconstruction, bool double_flux,
                                     const std::string & riemann = "HLLC") {
    const bool teno = reconstruction.find("TENO") != std::string::npos;
    const uint32_t n = 40, rows = teno ? 4 : 1;
    const std::string input =
        "[run]\nt_stop = 2.5e-4\ncfl = 0.5\n" +
        mesh_block("cartesian", n, rows, 1.0, rows * 1.0 / n, "[\"x\"]") +
        "[initialize]\ntype = \"analytical\"\np = \"1.0e5\"\nT = \"abs(x - 0.5) < 0.25 ? 300.0 : 1200.0\"\n"
        "u = " + velocity("1000.0") + "\n"
        "X = { H2 = \"abs(x - 0.5) < 0.25 ? 1 : 0\", O2 = \"abs(x - 0.5) < 0.25 ? 0 : 0.21\", "
        "N2 = \"abs(x - 0.5) < 0.25 ? 0 : 0.79\" }\n"
        "[[boundaries]]\nname = \"bottom\"\ntype = \"symmetry\"\n[[boundaries]]\nname = \"top\"\ntype = \"symmetry\"\n" +
        std::string(N_DIM == 3 ? "[[boundaries]]\nname = \"back\"\ntype = \"symmetry\"\n"
                                 "[[boundaries]]\nname = \"front\"\ntype = \"symmetry\"\n"
                               : "") +
        numerics(reconstruction, riemann, double_flux) + mixture(H2O2);
    Solver solver;
    solver.init(parse_toml(input));
    solver.run();
    solver.update_primitives();
    solver.copy_device_to_host();
    std::array<double, 2> err = {0.0, 0.0};
    for (uint32_t c = 0; c < solver.get_mesh()->n_cells; c++) {
        err[0] = std::max(err[0], std::abs(double(solver.h_primitives(c, N_DIM)) / 1.0e5 - 1.0));
        err[1] = std::max(err[1], std::abs(double(solver.h_primitives(c, 0)) / 1000.0 - 1.0));
    }
    return err;
}

} // namespace

TEST(MixtureTest, DoubleFluxKeepsPressureAndVelocityUniformAtContacts) {
    // Each cell's energy is updated with its own frozen gamma and e0, so a
    // contact between different gases moves without pressure waves; the
    // conservative scheme perturbs p and u by about a percent
    std::vector<std::string> schemes = {"type = \"MUSCL\"\n"};
    if constexpr (N_DIM == 2) schemes.push_back("type = \"TENO\"\norder = 5\n");
    for (const std::string & scheme : schemes) {
        for (const std::string riemann : {"HLLC", "Roe", "RHLL"}) {
            const std::array<double, 2> df = contact_errors(scheme, true, riemann);
            EXPECT_LT(df[0], tol(1e-12, 1e-5)) << scheme << riemann;
            EXPECT_LT(df[1], tol(1e-12, 1e-5)) << scheme << riemann;
        }
    }
    EXPECT_GT(contact_errors("type = \"MUSCL\"\n", false)[0], 1e-3);
}

TEST(MixtureTest, DoubleFluxShockTubeStaysAccurateAndReportsItsEnergyError) {
    // Double flux gives up energy conservation in proportion to the jumps of
    // gamma and e0; across the shock tube's waves (all still inside the
    // domain, so the total energy is constant) it stays small, and the
    // solution remains as close to the exact one as the conservative scheme's
    auto run = [&](bool double_flux, double & energy_error) {
        const uint32_t n = 200;
        const std::string input =
            "[run]\nt_stop = 2.0e-4\ncfl = 0.5\n" + mesh_block("cartesian", n, 1, 1.0, 1.0 / n) +
            "[initialize]\ntype = \"analytical\"\np = \"x < 0.5 ? 1.0e5 : 1.0e4\"\nT = \"x < 0.5 ? 1000.0 : 300.0\"\n"
            "u = " + velocity("0.0") + "\n"
            "X = { H2 = \"x < 0.5 ? 2 : 0\", O2 = \"x < 0.5 ? 1 : 0\", AR = \"x < 0.5 ? 7 : 0\", "
            "N2 = \"x < 0.5 ? 0 : 1\" }\n" +
            boundaries("type = \"extrapolation\"\n", "type = \"extrapolation\"\n", "type = \"symmetry\"\n",
                       "type = \"symmetry\"\n") +
            numerics("type = \"MUSCL\"\n", "HLLC", double_flux) + mixture(H2O2);
        Solver solver;
        solver.init(parse_toml(input));
        const double E0 = double(solver.integrate_conservatives()[N_DIM + 1]);
        solver.run();
        energy_error = std::abs(double(solver.integrate_conservatives()[N_DIM + 1]) / E0 - 1.0);
        solver.update_primitives();
        solver.copy_device_to_host();
        std::vector<std::string> columns;
        const auto ref = read_reference("shock_tube_" + std::to_string(n) + ".csv", columns);
        double err = 0.0, scale = 0.0;
        for (uint32_t c = 0; c < n; c++) {
            const uint32_t i = static_cast<uint32_t>(double(solver.get_mesh()->h_cell_coords(c, 0)) * n);
            err += std::abs(double(solver.h_primitives(c, N_DIM)) - ref[i][3]) / n;
            scale = std::max(scale, ref[i][3]);
        }
        return err / scale;
    };
    double conservative_energy, double_flux_energy;
    const double conservative = run(false, conservative_energy);
    const double double_flux = run(true, double_flux_energy);
    EXPECT_LT(conservative_energy, tol(1e-13, 1e-5));
    EXPECT_GT(double_flux_energy, conservative_energy);
    EXPECT_LT(double_flux_energy, 5e-3);  // 0.16% measured
    EXPECT_LT(double_flux, 1.2 * conservative) << "p: " << conservative << " -> " << double_flux;
    std::cout << "double flux: relative total-energy error " << double_flux_energy << ", L1 error of p "
              << double_flux << " (conservative " << conservative << ")\n";
}

namespace {

/**
 * @brief Largest transverse velocity over the speed of sound upstream after
 *        3000 steps of a standing Mach 6 normal shock in perfect air with a
 *        tiny density bump behind it, as carbuncle_growth of solver_test.cpp.
 */
double mixture_carbuncle_growth(const std::string & riemann) {
    // Rankine-Hugoniot at Mach 6, gamma = 1.4, R = 287
    const double T1 = 300.0, p1 = 1.0e5, a1 = std::sqrt(1.4 * 287.0 * T1), u1 = 6.0 * a1;
    const double rho_ratio = 5.2682926829268295, p2 = 41.833333333333333 * p1, u2 = u1 / rho_ratio;
    const double T2 = T1 * (p2 / p1) / rho_ratio;
    std::ostringstream s;
    s << std::setprecision(17) << "[run]\nn_steps = 3000\ncfl = 0.4\n" << mesh_block("cartesian", 40, 40, 1.0, 1.0)
      << "[initialize]\ntype = \"analytical\"\nY = { AIR = 1.0 }\n"
      << "p = \"x < 0.5 ? " << p1 << " : " << p2 << "\"\n"
      << "T = \"x < 0.5 ? " << T1 << " : " << T2 << " / (1 + (abs(y - 0.5) < 0.026 and abs(x - 0.51) < 0.02 ? 1e-3 : 0))\"\n"
      << "u = " << velocity("x < 0.5 ? " + std::to_string(u1) + " : " + std::to_string(u2)) << "\n"
      << boundaries("type = \"upt\"\nu = [" + std::to_string(u1) + ", 0.0]\np = " + std::to_string(p1) +
                        "\nT = " + std::to_string(T1) + "\nY = { AIR = 1.0 }\n",
                    "type = \"p_out\"\np = " + std::to_string(p2) + "\n", "type = \"symmetry\"\n",
                    "type = \"symmetry\"\n")
      << numerics("type = \"FO\"\n", riemann) << mixture(PERFECT_AIR, "air");
    Solver solver;
    solver.init(parse_toml(s.str()));
    solver.run();
    solver.update_primitives();
    solver.copy_device_to_host();
    double v_max = 0.0;
    for (uint32_t i = 0; i < solver.get_mesh()->n_cells; i++) {
        v_max = std::max(v_max, std::abs(double(solver.h_primitives(i, 1))));
    }
    return v_max / a1;
}

} // namespace

TEST(MixtureTest, RotatedHybridRiemannSolverIsCarbuncleFree) {
    SKIP_IN_SINGLE_PRECISION("Mach 6 amplifies the round-off of p to O(0.1) cross-flow for every solver");
    if constexpr (N_DIM == 2) {
        EXPECT_GT(mixture_carbuncle_growth("Roe"), 0.1);
        EXPECT_LT(mixture_carbuncle_growth("RHLL"), 1e-10);
    } else {
        GTEST_SKIP() << "the 2D setup of solver_test.cpp";
    }
}

namespace {

/** @brief Temperature of every cell of a mixture run from its conservatives (host thermo). */
std::vector<double> cell_temperatures(const Solver & solver, const chemistry::ThermoTable<Kokkos::HostSpace> & thermo) {
    const uint32_t n_cells = solver.get_mesh()->n_cells, ns = thermo.n_species;
    std::vector<double> T(n_cells), Y(ns);
    for (uint32_t c = 0; c < n_cells; c++) {
        const double rho = double(solver.h_conservatives(c, 0));
        double u2 = 0.0;
        FOR_I_DIM u2 += std::pow(double(solver.h_conservatives(c, 1 + i)) / rho, 2);
        for (uint32_t k = 0; k < ns; k++) Y[k] = double(solver.h_species(c, k)) / rho;
        const double e = double(solver.h_conservatives(c, N_DIM + 1)) / rho - 0.5 * u2;
        T[c] = thermo.T_from_e(e, chemistry::MassFractions{Y.data()}, 1000.0);
    }
    return T;
}

std::string reacting_box(uint32_t n_steps, double dt) {
    const std::string periodic = N_DIM == 2 ? "[\"x\", \"y\"]" : "[\"x\", \"y\", \"z\"]";
    std::ostringstream s;
    s << std::setprecision(17) << "[run]\nn_steps = " << n_steps << "\ndt = " << dt << "\n"
      << mesh_block("cartesian", 3, 3, 1.0, 1.0, periodic)
      << "[initialize]\ntype = \"constant\"\np = 101325.0\nT = 1200.0\nu = "
      << (N_DIM == 2 ? "[0.0, 0.0]" : "[0.0, 0.0, 0.0]") << "\nX = { H2 = 2.0, O2 = 1.0, N2 = 3.76 }\n"
      << numerics("type = \"MUSCL\"\n", "HLLC") << mixture(H2O2) << "[chemistry]\n";
    return s.str();
}

} // namespace

TEST(ReactingTest, UniformMixtureIgnitesLikeTheConstantVolumeReactor) {
    // V1 as a solver run: every cell of a uniform, quiescent box is an
    // adiabatic constant-volume reactor; with Strang splitting the chemistry
    // advances by dt per step. Against Cantera (stoichiometric H2/air,
    // 1200 K, 1 atm): T at 0.5 and 2 ignition delays within 1%
    const auto mech = chemistry::read_mechanism(H2O2);
    const auto thermo = chemistry::make_thermo_table<Kokkos::HostSpace>(mech);
    std::vector<std::string> columns;
    const auto ref = read_reference("h2o2_ignition.csv", columns);
    const auto column = [&](const std::string & name) {
        return static_cast<size_t>(std::find(columns.begin(), columns.end(), name) - columns.begin());
    };
    const auto row = std::find_if(ref.begin(), ref.end(), [&](const std::vector<double> & r) {
        return r[column("T0")] == 1200.0 && r[column("phi")] == 1.0 && r[column("p0")] == 101325.0;
    });
    ASSERT_NE(row, ref.end());
    const double tau = (*row)[column("tau")];
    for (const auto & [n_steps, name] : {std::pair<uint32_t, std::string>{25, "T_half_tau"}, {100, "T_2tau"}}) {
        Solver solver;
        solver.init(parse_toml(reacting_box(n_steps, tau / 50.0)));
        solver.run();
        solver.copy_device_to_host();
        const double T_ref = (*row)[column(name)];
        for (double T : cell_temperatures(solver, thermo)) EXPECT_NEAR(T, T_ref, 1e-2 * T_ref) << name;
    }
}

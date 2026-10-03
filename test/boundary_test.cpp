/**
 * @file boundary_test.cpp
 * @author Matthew Bonanni (mbonanni001@gmail.com)
 * @brief Tests for boundary-condition assignment and time-dependent boundaries.
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

std::string strip_input(const std::string & left, const std::string & bottom, const std::string & run,
                        uint32_t nx = 100) {
    std::ostringstream s;
    s << "[run]\n" << run
      << "[mesh]\ntype = \"cartesian\"\nNx = " << nx << "\nNy = 4\nLx = 1.0\nLy = 0.04\n"
      << "[initialize]\ntype = \"analytical\"\nrho = \"1.0\"\nu = [\"1.0\", \"0.0\"]\np = \"1.0\"\n"
      << "[[boundaries]]\nname = \"left\"\n" << left
      << "[[boundaries]]\nname = \"right\"\ntype = \"extrapolation\"\n"
      << "[[boundaries]]\nname = \"top\"\ntype = \"symmetry\"\n"
      << bottom
      << "[numerics]\nriemann_solver = \"HLLC\"\ntime_integrator = \"SSPRK3\"\n"
      << "[numerics.face_reconstruction]\ntype = \"TENO\"\norder = 5\n"
      << "[physics]\ntype = \"euler\"\ngamma = 1.4\np_ref = 1.0\nT_ref = 1.0\nrho_ref = 1.0\n"
      << "[output]\ncheck_interval = 1000000\n";
    return s.str();
}

const std::string BOTTOM_SYMMETRY = "[[boundaries]]\nname = \"bottom\"\ntype = \"symmetry\"\n";

} // namespace

TEST(BoundaryTest, TimeDependentDirichletInflowIsAdvected) {
    // rho_in(t) enters at u = 1 and is carried unchanged: rho(x, t) = rho_in(t - x)
    const std::string left = "type = \"dirichlet\"\nrho = \"1.0 + 0.1 * sin(2 * pi * t)\"\n"
                             "u = [\"1.0\", \"0.0\"]\np = \"1.0\"\n";
    Solver solver;
    solver.init(parse_toml(strip_input(left, BOTTOM_SYMMETRY, "t_stop = 0.8\ncfl = 0.4\n")));
    solver.run();
    solver.update_primitives();
    solver.copy_device_to_host();
    auto m = solver.get_mesh();
    double max_err = 0.0;
    for (uint32_t i = 0; i < m->n_cells; i++) {
        const double x = double(m->h_cell_coords(i, 0));
        if (x > 0.7) continue;  // Beyond the entering front
        const double h = 0.01;
        // Exact cell average of 1 + 0.1 sin(2 pi (t - x)) over [x - h/2, x + h/2]
        const double t = double(solver.get_time());
        const double exact = 1.0 + 0.1 * std::sin(2.0 * M_PI * (t - x)) * std::sin(M_PI * h) / (M_PI * h);
        max_err = std::max(max_err, std::abs(double(solver.h_conservatives(i, 0)) - exact));
    }
    EXPECT_LT(max_err, 5e-4);
}

TEST(BoundaryTest, WhereSplitsAZoneBetweenConditions) {
    const std::string bottom =
        "[[boundaries]]\nname = \"bottom\"\ntype = \"symmetry\"\nwhere = \"x < 0.5\"\n"
        "[[boundaries]]\nname = \"bottom\"\ntype = \"wall_adiabatic\"\nwhere = \"x >= 0.5\"\n";
    Solver solver;
    EXPECT_NO_THROW(solver.init(parse_toml(strip_input("type = \"extrapolation\"\n", bottom, "n_steps = 1\ncfl = 0.4\n", 20))));
}

TEST(BoundaryTest, OverlappingOrMissingAssignmentsAreRejected) {
    const std::string overlap =
        "[[boundaries]]\nname = \"bottom\"\ntype = \"symmetry\"\nwhere = \"x < 0.6\"\n"
        "[[boundaries]]\nname = \"bottom\"\ntype = \"wall_adiabatic\"\nwhere = \"x >= 0.5\"\n";
    const std::string gap =
        "[[boundaries]]\nname = \"bottom\"\ntype = \"symmetry\"\nwhere = \"x < 0.3\"\n"
        "[[boundaries]]\nname = \"bottom\"\ntype = \"wall_adiabatic\"\nwhere = \"x >= 0.5\"\n";
    Solver a, b;
    EXPECT_THROW(a.init(parse_toml(strip_input("type = \"extrapolation\"\n", overlap, "n_steps = 1\ncfl = 0.4\n", 20))),
                 std::runtime_error);
    EXPECT_THROW(b.init(parse_toml(strip_input("type = \"extrapolation\"\n", gap, "n_steps = 1\ncfl = 0.4\n", 20))),
                 std::runtime_error);
}

namespace {

/**
 * @brief Max deviation of the pressure in the outlet column from the
 *        hydrostatic profile exp(-y), for a channel flow under gravity.
 */
double outlet_profile_error(const std::string & outlet) {
    std::ostringstream s;
    s << "[run]\nt_stop = 2.0\ncfl = 0.5\n"
      << "[mesh]\ntype = \"cartesian\"\nNx = 20\nNy = 20\nLx = 1.0\nLy = 1.0\n"
      << "[initialize]\ntype = \"analytical\"\nrho = \"exp(-y)\"\nu = [\"0.3\", \"0.0\"]\np = \"exp(-y)\"\n"
      << "[[boundaries]]\nname = \"left\"\ntype = \"dirichlet\"\nrho = \"exp(-y)\"\nu = [\"0.3\", \"0.0\"]\np = \"exp(-y)\"\n"
      << "[[boundaries]]\nname = \"right\"\n" << outlet
      << "[[boundaries]]\nname = \"top\"\ntype = \"symmetry\"\n"
      << "[[boundaries]]\nname = \"bottom\"\ntype = \"symmetry\"\n"
      << "[numerics]\nriemann_solver = \"HLLC\"\ntime_integrator = \"SSPRK3\"\n"
      << "[numerics.face_reconstruction]\ntype = \"MUSCL\"\n"
      << "[physics]\ntype = \"euler\"\ngamma = 1.4\np_ref = 1.0\nT_ref = 1.0\nrho_ref = 1.0\n"
      << "[output]\ncheck_interval = 1000000\n"
      << "[source]\ngravity = [0.0, -1.0]\n";
    Solver solver;
    solver.init(parse_toml(s.str()));
    solver.run();
    solver.update_primitives();
    solver.copy_device_to_host();
    auto m = solver.get_mesh();
    double err = 0.0;
    for (uint32_t i = 0; i < m->n_cells; i++) {
        if (m->h_cell_coords(i, 0) < 0.95_r) continue;
        err = std::max(err, std::abs(double(solver.h_primitives(i, 2)) - std::exp(-double(m->h_cell_coords(i, 1)))));
    }
    return err;
}

} // namespace

TEST(BoundaryTest, AveragePressureOutletKeepsTransversePressureProfile) {
    // The mean of exp(-y) over [0, 1] is 1 - 1/e. A uniform back pressure
    // fights the hydrostatic profile; imposing only its average does not.
    const std::string target = "p = 0.6321205588285577\n";
    const double uniform = outlet_profile_error("type = \"p_out\"\n" + target);
    const double average = outlet_profile_error("type = \"p_out_average\"\n" + target);
    EXPECT_GT(uniform, 0.1);
    EXPECT_LT(average, 0.03);
}

namespace {

struct FarfieldCase {
    BoundaryCondition bc;
    rtype W_g[N_CONSERVATIVE];
    rtype n[N_DIM] = {0.6, 0.8};
    static constexpr rtype G = 1.4;

    FarfieldCase(const rtype * W_i) {
        bc.type = BoundaryType::FARFIELD;
        const rtype W_inf[N_CONSERVATIVE] = {1.0, 0.3, -0.1, 1.0_r / G};
        FOR_I_CONSERVATIVE bc.data[i] = W_inf[i];
        bc.ghost_W(W_i, n, G, 1.0, false, W_g);
    }
    static rtype u_n(const rtype * W, const rtype * n) { return W[1] * n[0] + W[2] * n[1]; }
    static rtype u_t(const rtype * W, const rtype * n) { return -W[1] * n[1] + W[2] * n[0]; }
    static rtype a(const rtype * W) { return std::sqrt(G * W[3] / W[0]); }
    static rtype r_plus(const rtype * W, const rtype * n) { return u_n(W, n) + 2.0_r * a(W) / (G - 1.0_r); }
    static rtype r_minus(const rtype * W, const rtype * n) { return u_n(W, n) - 2.0_r * a(W) / (G - 1.0_r); }
    static rtype entropy(const rtype * W) { return W[3] / std::pow(W[0], G); }
};

} // namespace

TEST(BoundaryTest, FarfieldReturnsTheFreeStreamForTheFreeStream) {
    const rtype W_inf[N_CONSERVATIVE] = {1.0, 0.3, -0.1, 1.0 / 1.4};
    FarfieldCase c(W_inf);
    FOR_I_CONSERVATIVE EXPECT_NEAR(c.W_g[i], W_inf[i], roundoff(1e-14));
}

TEST(BoundaryTest, FarfieldTakesEachCharacteristicFromItsUpwindSide) {
    using F = FarfieldCase;
    const rtype * W_inf = nullptr;
    // Subsonic outflow (u_n > 0): outgoing invariant, entropy and tangential
    // velocity from the interior; incoming invariant from the free stream
    const rtype W_out[N_CONSERVATIVE] = {1.1, 0.4, 0.2, 0.8};
    F out(W_out);
    W_inf = out.bc.data;
    ASSERT_GT(F::u_n(out.W_g, out.n), 0.0);
    EXPECT_NEAR(F::r_plus(out.W_g, out.n), F::r_plus(W_out, out.n), roundoff(1e-12));
    EXPECT_NEAR(F::r_minus(out.W_g, out.n), F::r_minus(W_inf, out.n), roundoff(1e-12));
    EXPECT_NEAR(F::entropy(out.W_g), F::entropy(W_out), roundoff(1e-12));
    EXPECT_NEAR(F::u_t(out.W_g, out.n), F::u_t(W_out, out.n), roundoff(1e-12));

    // Subsonic inflow (u_n < 0): entropy and tangential velocity from the free stream
    const rtype W_in[N_CONSERVATIVE] = {0.9, -0.5, -0.3, 0.6};
    F in(W_in);
    ASSERT_LT(F::u_n(in.W_g, in.n), 0.0);
    EXPECT_NEAR(F::r_plus(in.W_g, in.n), F::r_plus(W_in, in.n), roundoff(1e-12));
    EXPECT_NEAR(F::r_minus(in.W_g, in.n), F::r_minus(W_inf, in.n), roundoff(1e-12));
    EXPECT_NEAR(F::entropy(in.W_g), F::entropy(W_inf), roundoff(1e-12));
    EXPECT_NEAR(F::u_t(in.W_g, in.n), F::u_t(W_inf, in.n), roundoff(1e-12));

    // Supersonic outflow keeps the interior, supersonic inflow takes the free stream
    const rtype W_sup_out[N_CONSERVATIVE] = {1.0, 1.2, 1.6, 1.0 / 1.4};
    F sup_out(W_sup_out);
    FOR_I_CONSERVATIVE EXPECT_EQ(sup_out.W_g[i], W_sup_out[i]);
    const rtype W_sup_in[N_CONSERVATIVE] = {1.0, -1.2, -1.6, 1.0 / 1.4};
    F sup_in(W_sup_in);
    FOR_I_CONSERVATIVE EXPECT_EQ(sup_in.W_g[i], W_inf[i]);
}

TEST(BoundaryTest, FarfieldLetsAPressurePulseLeave) {
    // An isentropic pressure pulse leaves a box of far-field boundaries without
    // reflection. (Riemann-invariant far fields do reflect entropy waves.)
    std::ostringstream s;
    s << "[run]\nt_stop = 3.0\ncfl = 0.5\n"
      << "[mesh]\ntype = \"cartesian_tri\"\nNx = 24\nNy = 24\nLx = 1.0\nLy = 1.0\n"
      << "[initialize]\ntype = \"analytical\"\nu = [\"0.3\", \"0.0\"]\n"
      << "p = \"1 / 1.4 + 0.1 * exp(-100 * ((x - 0.5)^2 + (y - 0.5)^2))\"\n"
      << "rho = \"(1 + 0.14 * exp(-100 * ((x - 0.5)^2 + (y - 0.5)^2)))^(1 / 1.4)\"\n";
    for (const char * name : {"left", "right", "top", "bottom"}) {
        s << "[[boundaries]]\nname = \"" << name << "\"\ntype = \"farfield\"\n"
          << "u = [0.3, 0.0]\np = 0.7142857142857143\nT = 0.7142857142857143\n";
    }
    s << "[numerics]\nriemann_solver = \"HLLC\"\ntime_integrator = \"SSPRK3\"\n"
      << "[numerics.face_reconstruction]\ntype = \"MUSCL\"\n"
      << "[physics]\ntype = \"euler\"\ngamma = 1.4\np_ref = 1.0\nT_ref = 1.0\nrho_ref = 1.0\n"
      << "[output]\ncheck_interval = 1000000\n";
    Solver solver;
    solver.init(parse_toml(s.str()));
    solver.run();
    solver.update_primitives();
    solver.copy_device_to_host();
    double dp_max = 0.0;
    for (uint32_t i = 0; i < solver.get_mesh()->n_cells; i++) {
        dp_max = std::max(dp_max, std::abs(double(solver.h_primitives(i, 2)) - 1.0 / 1.4));
    }
    // upt leaves 1.8e-4, extrapolation 5e-4; single precision adds ~1e-6 of round-off
    EXPECT_LT(dp_max, precision_tol<double>(1.2e-4, 1.3e-4));
}

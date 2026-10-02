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
        const double x = m->h_cell_coords(i, 0);
        if (x > 0.7) continue;  // Beyond the entering front
        const double h = 0.01;
        // Exact cell average of 1 + 0.1 sin(2 pi (t - x)) over [x - h/2, x + h/2]
        const double t = solver.get_time();
        const double exact = 1.0 + 0.1 * std::sin(2.0 * M_PI * (t - x)) * std::sin(M_PI * h) / (M_PI * h);
        max_err = std::max(max_err, std::abs(solver.h_conservatives(i, 0) - exact));
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
        if (m->h_cell_coords(i, 0) < 0.95) continue;
        err = std::max(err, std::abs(solver.h_primitives(i, 2) - std::exp(-m->h_cell_coords(i, 1))));
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

/**
 * @file axisymmetric_test.cpp
 * @author Matthew Bonanni (mbonanni001@gmail.com)
 * @brief Tests of the axisymmetric (r-z) formulation.
 * @version 0.1
 * @date 2026-10-04
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 *
 */

#include <gtest/gtest.h>
#include <Kokkos_Core.hpp>

#include <array>
#include <cmath>
#include <functional>
#include <sstream>
#include <string>
#include <tuple>
#include <vector>

#include "input.h"
#include "test_fixtures.h"
#include "solver.h"

namespace {

constexpr double PI = 3.14159265358979323846;

struct Case {
    std::string mesh = "cartesian";
    uint32_t n = 16;
    std::string reconstruction = "type = \"FO\"";
    std::string physics = "type = \"euler\"";
    std::string init;
    std::string top = "type = \"wall_adiabatic\"";
    std::string sides = "";  // empty: periodic in x
    std::string source;
    std::string run = "n_steps = 20\ncfl = 0.5\n";
};

/** @brief Unit square (axis at y = 0, symmetry) with a gamma = 1.4 gas. */
std::string input(const Case & c) {
    std::ostringstream s;
    s << "[run]\n" << c.run
      << "[mesh]\ntype = \"" << c.mesh << "\"\nNx = " << c.n << "\nNy = " << c.n << "\nLx = 1.0\nLy = 1.0\n";
    if (c.sides.empty()) s << "periodic = [\"x\"]\n";
    s << "[initialize]\n" << c.init
      << "[[boundaries]]\nname = \"bottom\"\ntype = \"symmetry\"\n"
      << "[[boundaries]]\nname = \"top\"\n" << c.top << "\n";
    if (!c.sides.empty()) {
        s << "[[boundaries]]\nname = \"left\"\n" << c.sides << "\n"
          << "[[boundaries]]\nname = \"right\"\n" << c.sides << "\n";
    }
    s << "[numerics]\nriemann_solver = \"HLLC\"\ntime_integrator = \"SSPRK3\"\n"
      << "[numerics.face_reconstruction]\n" << c.reconstruction << "\n"
      << "[physics]\naxisymmetric = true\ngamma = 1.4\np_ref = 1.0\nT_ref = 1.0\nrho_ref = 1.0\n" << c.physics << "\n"
      << "[output]\ncheck_interval = 1000000\n";
    if (!c.source.empty()) s << "[source]\n" << c.source;
    return s.str();
}

std::vector<std::array<double, N_CONSERVATIVE>> host_rhs(Solver & solver, double t = 0.0) {
    const uint32_t n = solver.get_mesh()->n_cells;
    State rhs("rhs", n, 0);
    solver.calc_rhs(solver.state(), rhs, rtype(t));
    auto h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), rhs.flow);
    std::vector<std::array<double, N_CONSERVATIVE>> out(n);
    for (uint32_t c = 0; c < n; c++) FOR_I_CONSERVATIVE out[c][i] = double(h(c, i));
    return out;
}

double max_speed(Solver & solver) {
    solver.update_primitives();
    solver.copy_device_to_host();
    double v_max = 0.0;
    for (uint32_t i = 0; i < solver.get_mesh()->n_owned(); i++) {
        v_max = std::max(v_max, std::hypot(double(solver.h_primitives(i, 0)), double(solver.h_primitives(i, 1))));
    }
    return v_max;
}

} // namespace

TEST(AxisymmetricTest, RevolvedGeometryOfCellsAndFaces) {
    Solver solver;
    Case c;
    c.mesh = "cartesian_tri";
    c.n = 8;
    c.init = "type = \"constant\"\nu = [0.0, 0.0]\np = 1.0\nT = 1.0\n";
    solver.init(parse_toml(input(c)));
    const Mesh & mesh = *solver.get_mesh();
    double total = 0.0;
    for (uint32_t i = 0; i < mesh.n_cells; i++) {
        total += double(mesh.h_cell_measure(i));
        // Pappus: the revolved volume is the planar area times the radius of its centroid
        double cy = 0.0;
        for (uint32_t k = 0; k < 3; k++) cy += double(mesh.h_node_coords(mesh.h_node_of_cell(i, k), 1)) / 3.0;
        EXPECT_NEAR(double(mesh.h_cell_measure(i)), double(mesh.h_cell_volume(i)) * cy, 1e-15);
        // The face radii close each cell against its planar area: the discrete
        // identity that keeps quiescent gas at rest
        double closure = 0.0;
        for (uint32_t k = 0; k < mesh.h_n_faces_of_cell(i); k++) {
            const uint32_t f = mesh.h_face_of_cell(i, k);
            const double sign = mesh.h_cells_of_face(f, 0) == int32_t(i) ? 1.0 : -1.0;
            closure += sign * double(mesh.h_face_normals(f, 1)) / double(mesh.h_face_area(f)) * double(mesh.h_face_measure(f));
        }
        EXPECT_NEAR(closure, double(mesh.h_cell_volume(i)), precision_tol<double>(1e-15, 1e-7));
    }
    // int_0^1 int_0^1 r dr dx
    EXPECT_NEAR(total, 0.5, precision_tol<double>(1e-14, 1e-6));
}

TEST(AxisymmetricTest, CellCentersAreRWeightedCentroids) {
    // A linear function equals its r-weighted cell average at the cell center
    Solver solver;
    Case c;
    c.mesh = "cartesian_tri";
    c.n = 4;
    c.init = "type = \"analytical\"\nrho = \"1 + 0.3 * x + 0.5 * y\"\nu = [\"0\", \"0\"]\np = \"1\"\n";
    solver.init(parse_toml(input(c)));
    solver.copy_device_to_host();
    const Mesh & mesh = *solver.get_mesh();
    for (uint32_t i = 0; i < mesh.n_cells; i++) {
        const double expected = 1.0 + 0.3 * double(mesh.h_cell_coords(i, 0)) + 0.5 * double(mesh.h_cell_coords(i, 1));
        EXPECT_NEAR(double(solver.h_conservatives(i, 0)), expected, precision_tol<double>(1e-13, 1e-6));
    }
}

TEST(AxisymmetricTest, AxisMustBeSymmetry) {
    Solver solver;
    Case c;
    c.init = "type = \"constant\"\nu = [0.0, 0.0]\np = 1.0\nT = 1.0\n";
    std::string text = input(c);
    text.replace(text.find("name = \"bottom\"\ntype = \"symmetry\""), 33, "name = \"bottom\"\ntype = \"wall_adiabatic\"");
    EXPECT_THROW(solver.init(parse_toml(text)), InputError);
}

using QuiescentParam = std::tuple<std::string, std::string, std::string>;
class AxisymmetricQuiescent : public ::testing::TestWithParam<QuiescentParam> {};

TEST_P(AxisymmetricQuiescent, GasAtUniformPressureStaysAtRest) {
    // Without the geometric pressure source, or with face areas that do not
    // close against it, the gas accelerates away from the axis at O(1)
    const auto [mesh, reconstruction, physics] = GetParam();
    Case c;
    c.mesh = mesh;
    c.reconstruction = reconstruction;
    c.physics = physics;
    c.sides = "type = \"wall_adiabatic\"";
    // Inviscid: a density field in x and r at uniform pressure is also at rest
    const std::string rho = physics == "type = \"euler\"" ? "1 + 0.3 * cos(3 * x) * cos(2 * y)" : "1";
    c.init = "type = \"analytical\"\nrho = \"" + rho + "\"\nu = [\"0\", \"0\"]\np = \"1\"\n";
    Solver solver;
    solver.init(parse_toml(input(c)));
    solver.run();
    EXPECT_LT(max_speed(solver), precision_tol<double>(1e-13, 1e-5));
}

INSTANTIATE_TEST_SUITE_P(
    Axisymmetric, AxisymmetricQuiescent,
    ::testing::Combine(::testing::Values("cartesian", "cartesian_tri"),
                       ::testing::Values("type = \"FO\"", "type = \"MUSCL\"", "type = \"TENO\"\norder = 3",
                                         "type = \"TENO\"\norder = 5"),
                       ::testing::Values("type = \"euler\"",
                                         "type = \"navier_stokes\"\nmu = 0.01\nPr = 0.7")));

TEST(AxisymmetricTest, ConservesMassAxialMomentumAndEnergy) {
    // Periodic pipe with a slip wall: nothing crosses the axis or the wall, and
    // only the radial momentum has a source
    Case c;
    c.top = "type = \"symmetry\"";
    c.mesh = "cartesian_tri";
    c.reconstruction = "type = \"TENO\"\norder = 4";
    c.physics = "type = \"navier_stokes\"\nmu = 0.01\nPr = 0.7";
    c.init = "type = \"analytical\"\nrho = \"1\"\nu = [\"0.2 * (1 - y * y)\", \"0.1 * y * sin(6.283185307179586 * x)\"]\n"
             "p = \"1 + 0.5 * exp(-20 * ((x - 0.5)^2 + y^2))\"\n";
    Solver solver;
    solver.init(parse_toml(input(c)));
    const auto before = solver.integrate_conservatives();
    solver.run();
    const auto after = solver.integrate_conservatives();
    for (int i : {0, 1, 3}) EXPECT_NEAR(after[i], before[i], precision_tol<double>(1e-14, 1e-6) * std::abs(before[3]));
    EXPECT_GT(std::abs(after[2] - before[2]), 1e-6);
}

namespace {

/**
 * @brief Hagen-Poiseuille on quadrilaterals: x-momentum residual at the exact
 *        solution, mean over the cells off the wall (whose one-sided wall
 *        gradient is first order, as in planar runs) and in the axis row.
 */
std::array<double, 2> poiseuille_residual(uint32_t n) {
    // rho g + mu (1/r) d/dr (r du/dr) = 0 with u = U (1 - r^2): g = 4 mu U / rho
    const double mu = 0.01, U = 0.1;
    Case c;
    c.n = n;
    c.physics = "type = \"navier_stokes\"\nmu = 0.01\nPr = 0.7";
    c.reconstruction = "type = \"MUSCL\"";
    c.top = "type = \"wall_isothermal\"\nT = 1.0";
    c.init = "type = \"analytical\"\nrho = \"1\"\nu = [\"0.1 * (1 - y * y)\", \"0\"]\np = \"1\"\n";
    std::ostringstream g;
    g << "gravity = [" << 4.0 * mu * U << ", 0.0]\n";
    c.source = g.str();
    Solver solver;
    solver.init(parse_toml(input(c)));
    const auto rhs = host_rhs(solver);
    const Mesh & mesh = *solver.get_mesh();
    double sum = 0.0, axis = 0.0;
    uint32_t count = 0, count_axis = 0;
    for (uint32_t i = 0; i < rhs.size(); i++) {
        const double r = double(mesh.h_cell_coords(i, 1));
        if (r > 1.0 - 1.0 / n) continue;
        sum += std::abs(rhs[i][1]);
        count++;
        if (r < 1.0 / n) {
            axis += std::abs(rhs[i][1]);
            count_axis++;
        }
    }
    return {sum / count, axis / count_axis};
}

} // namespace

TEST(AxisymmetricTest, PoiseuilleResidualConvergesAtSecondOrderUpToTheAxis) {
    // In planar geometry this profile has an O(1) residual (the 1/r du/dr term);
    // next to the axis, gradients fit the cells' r-weighted averages and the face
    // gradients are interpolated to the faces, or the residual there stays O(1)
    const double forcing = 4.0 * 0.01 * 0.1;
    const auto e1 = poiseuille_residual(8);
    const auto e2 = poiseuille_residual(16);
    EXPECT_GT(std::log2(e1[0] / e2[0]), 1.8);
    EXPECT_LT(e2[0], 1e-3 * forcing);
    EXPECT_LT(e2[1], 1e-3 * forcing);
}

namespace {

/** @brief Smooth manufactured state [rho, u_x, u_r, p], regular on the axis. */
std::array<double, 4> mms_state(double x, double r) {
    const double s = std::sin(2 * PI * x), co = std::cos(2 * PI * x);
    return {1.0 + 0.2 * s * std::cos(PI * r), 0.3 + 0.1 * co * std::cos(PI * r),
            0.1 * std::sin(PI * r) * (1.0 + 0.5 * s), 1.0 + 0.2 * co * std::cos(PI * r)};
}

constexpr double GAMMA = 1.4;

/** @brief Euler flux F . n of the manufactured state. */
std::array<double, 4> mms_flux(double x, double r, double nx, double nr) {
    const auto [rho, u, v, p] = mms_state(x, r);
    const double un = u * nx + v * nr;
    const double E = p / (GAMMA - 1.0) + 0.5 * rho * (u * u + v * v);
    return {rho * un, rho * u * un + p * nx, rho * v * un + p * nr, (E + p) * un};
}

/** @brief dU/dt of each cell's r-weighted average under the exact manufactured state. */
std::vector<std::array<double, 4>> mms_exact_rhs(const Mesh & mesh) {
    const double gx[6] = {-0.9324695142031521, -0.6612093864662645, -0.2386191860831969,
                          0.2386191860831969, 0.6612093864662645, 0.9324695142031521};
    const double gw[6] = {0.1713244923791704, 0.3607615730481386, 0.4679139345726910,
                          0.4679139345726910, 0.3607615730481386, 0.1713244923791704};
    std::vector<std::array<double, 4>> out(mesh.n_cells);
    for (uint32_t c = 0; c < mesh.n_cells; c++) {
        const uint32_t nn = mesh.h_n_nodes_of_cell(c);
        std::vector<double> px(nn), py(nn);
        for (uint32_t k = 0; k < nn; k++) {
            px[k] = double(mesh.h_node_coords(mesh.h_node_of_cell(c, k), 0));
            py[k] = double(mesh.h_node_coords(mesh.h_node_of_cell(c, k), 1));
        }
        double orient = 0.0;
        for (uint32_t k = 0; k < nn; k++) orient += px[k] * py[(k + 1) % nn] - px[(k + 1) % nn] * py[k];
        orient = orient > 0 ? 1.0 : -1.0;
        std::array<double, 4> flux = {};
        // Boundary: int F . n r dl, outward normal of a counterclockwise polygon
        for (uint32_t k = 0; k < nn; k++) {
            const double ax = px[k], ay = py[k], bx = px[(k + 1) % nn], by = py[(k + 1) % nn];
            const double L = std::hypot(bx - ax, by - ay);
            const double nx = orient * (by - ay) / L, nr = -orient * (bx - ax) / L;
            for (int q = 0; q < 6; q++) {
                const double t = 0.5 * (gx[q] + 1.0);
                const double x = ax + t * (bx - ax), r = ay + t * (by - ay);
                const auto F = mms_flux(x, r, nx, nr);
                for (int i = 0; i < 4; i++) flux[i] += 0.5 * gw[q] * L * r * F[i];
            }
        }
        // Interior: int p dA and int r dA by a 6 x 6 collapsed Gauss rule on a fan
        double source = 0.0, volume = 0.0;
        for (uint32_t k = 1; k + 1 < nn; k++) {
            const double ex = px[k] - px[0], ey = py[k] - py[0], fx = px[k + 1] - px[0], fy = py[k + 1] - py[0];
            const double det = std::abs(ex * fy - ey * fx);
            for (int i = 0; i < 6; i++) {
                for (int j = 0; j < 6; j++) {
                    const double s = 0.5 * (gx[i] + 1.0), t = 0.5 * (gx[j] + 1.0) * (1.0 - s);
                    const double w = 0.25 * gw[i] * gw[j] * (1.0 - s) * det;
                    const double x = px[0] + s * ex + t * fx, r = py[0] + s * ey + t * fy;
                    source += w * mms_state(x, r)[3];
                    volume += w * r;
                }
            }
        }
        flux[2] -= source;
        for (int i = 0; i < 4; i++) out[c][i] = -flux[i] / volume;
    }
    return out;
}

/** @brief Mean error of the RHS at the exact manufactured cell averages, over cells with r < 0.6. */
double mms_residual_error(const std::string & mesh, uint32_t n, const std::string & reconstruction) {
    Case c;
    c.mesh = mesh;
    c.n = n;
    c.reconstruction = reconstruction;
    c.init = "type = \"analytical\"\nn_subdivisions = 6\n"
             "rho = \"1 + 0.2 * sin(2 * pi * x) * cos(pi * y)\"\n"
             "u = [\"0.3 + 0.1 * cos(2 * pi * x) * cos(pi * y)\", \"0.1 * sin(pi * y) * (1 + 0.5 * sin(2 * pi * x))\"]\n"
             "p = \"1 + 0.2 * cos(2 * pi * x) * cos(pi * y)\"\n";
    c.top = "type = \"extrapolation\"";
    Solver solver;
    solver.init(parse_toml(input(c)));
    const auto rhs = host_rhs(solver);
    const auto exact = mms_exact_rhs(*solver.get_mesh());
    const Mesh & m = *solver.get_mesh();
    double sum = 0.0;
    uint32_t count = 0;
    for (uint32_t i = 0; i < m.n_cells; i++) {
        if (m.h_cell_coords(i, 1) > 0.6) continue;
        for (int v = 0; v < 4; v++) sum += std::abs(rhs[i][v] - exact[i][v]);
        count++;
    }
    return sum / count;
}

} // namespace

using OrderParam = std::tuple<std::string, int>;
class AxisymmetricOrder : public ::testing::TestWithParam<OrderParam> {};

TEST_P(AxisymmetricOrder, ManufacturedResidualConvergesAtDesignOrder) {
    // The spatial operator at the exact r-weighted averages of a smooth
    // manufactured state, up to and including the axis cells
    const auto [mesh, order] = GetParam();
    const std::string teno = "type = \"TENO\"\norder = " + std::to_string(order);
    const double e1 = mms_residual_error(mesh, 16, teno);
    const double e2 = mms_residual_error(mesh, 32, teno);
    EXPECT_GT(std::log2(e1 / e2), order - 1 - 0.3) << e1 << " " << e2;
}

INSTANTIATE_TEST_SUITE_P(Axisymmetric, AxisymmetricOrder,
                         ::testing::Combine(::testing::Values("cartesian", "cartesian_tri"), ::testing::Values(3, 4, 5)));

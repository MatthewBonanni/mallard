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

#include "data_paths.h"
#include "gmsh_fixtures.h"
#include "input.h"
#include "test_fixtures.h"
#include "test_utils.h"
#include "verification.h"
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
    std::string run = "n_steps = 20\ncfl = 0.25\n";
};

/** @brief Unit square (axis at y = 0, symmetry) with a gamma = 1.4 gas. */
std::string input(const Case & c) {
    std::ostringstream s;
    s << "[run]\n" << c.run
      << "[mesh]\ntype = \"" << c.mesh << "\"\nNx = " << c.n << "\nNy = " << c.n << "\nLx = 1.0\nLy = 1.0\n";
    if (c.sides.empty()) s << "[[periodic]]\nzones = [\"left\", \"right\"]\n";
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
    solver.set_distributed(false);
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
        EXPECT_NEAR(double(mesh.h_cell_measure(i)), double(mesh.h_cell_volume(i)) * cy, precision_tol<double>(1e-15, 1e-7));
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
    solver.set_distributed(false);
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
    solver.set_distributed(false);
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
    solver.set_distributed(false);
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

TEST(AxisymmetricTest, GasMixtureAtRestStaysAtRest) {
    // The mixture flux, its pressure in the geometric source and, viscous, its
    // transport. (Cell averages of a gas whose temperature or composition varies
    // are not at uniform pressure, in planar runs too.)
    const std::string mechanism = "gas = \"mixture\"\nmechanism = \"" + source_dir() + "/mechanisms/h2o2.yaml\"\n";
    for (const std::string type : {"euler", "navier_stokes"}) {
        for (const std::string reconstruction : {"type = \"MUSCL\"", "type = \"TENO\"\norder = 3"}) {
            Case c;
            c.reconstruction = reconstruction;
            c.sides = "type = \"wall_adiabatic\"";
            c.init = "type = \"constant\"\np = 1.0e5\nT = 600.0\nu = [0.0, 0.0]\nX = { H2 = 2.0, O2 = 1.0, N2 = 3.76 }\n";
            c.physics = "type = \"" + type + "\"\n" + mechanism;
            std::string text = input(c);
            // The mixture takes no perfect-gas constants
            const std::string constants = "gamma = 1.4\np_ref = 1.0\nT_ref = 1.0\nrho_ref = 1.0\n";
            text.replace(text.find(constants), constants.size(), "");
            Solver solver;
            solver.set_distributed(false);
            solver.init(parse_toml(text));
            solver.run();
            EXPECT_LT(max_speed(solver), precision_tol<double>(1e-10, 1e-3)) << type << " " << reconstruction;
        }
    }
}

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
    solver.set_distributed(false);
    solver.init(parse_toml(input(c)));
    const auto before = solver.integrate_conservatives();
    solver.run();
    const auto after = solver.integrate_conservatives();
    for (size_t i : {0u, 1u, 3u}) {
        EXPECT_NEAR(double(after[i]), double(before[i]), precision_tol<double>(1e-14, 1e-6) * std::abs(double(before[3])));
    }
    EXPECT_GT(std::abs(double(after[2] - before[2])), 1e-6);
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
    solver.set_distributed(false);
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
    SKIP_IN_SINGLE_PRECISION("the residual on 16 cells is at round-off");
    const double forcing = 4.0 * 0.01 * 0.1;
    const auto e1 = poiseuille_residual(8);
    const auto e2 = poiseuille_residual(16);
    verification::expect_order({"axisymmetric", "Poiseuille pipe flow", "quadrilaterals",
                                "mean axial momentum residual", {8, 16}, {e1[0], e2[0]}, 2.0, 1.8});
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
            for (size_t q = 0; q < 6; q++) {
                const double t = 0.5 * (gx[q] + 1.0);
                const double x = ax + t * (bx - ax), r = ay + t * (by - ay);
                const auto F = mms_flux(x, r, nx, nr);
                for (size_t i = 0; i < 4; i++) flux[i] += 0.5 * gw[q] * L * r * F[i];
            }
        }
        // Interior: int p dA and int r dA by a 6 x 6 collapsed Gauss rule on a fan
        double source = 0.0, volume = 0.0;
        for (uint32_t k = 1; k + 1 < nn; k++) {
            const double ex = px[k] - px[0], ey = py[k] - py[0], fx = px[k + 1] - px[0], fy = py[k + 1] - py[0];
            const double det = std::abs(ex * fy - ey * fx);
            for (size_t i = 0; i < 6; i++) {
                for (size_t j = 0; j < 6; j++) {
                    const double s = 0.5 * (gx[i] + 1.0), t = 0.5 * (gx[j] + 1.0) * (1.0 - s);
                    const double w = 0.25 * gw[i] * gw[j] * (1.0 - s) * det;
                    const double x = px[0] + s * ex + t * fx, r = py[0] + s * ey + t * fy;
                    source += w * mms_state(x, r)[3];
                    volume += w * r;
                }
            }
        }
        flux[2] -= source;
        for (size_t i = 0; i < 4; i++) out[c][i] = -flux[i] / volume;
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
    c.top = "type = \"outlet_extrapolation\"";
    Solver solver;
    solver.set_distributed(false);
    solver.init(parse_toml(input(c)));
    const auto rhs = host_rhs(solver);
    const auto exact = mms_exact_rhs(*solver.get_mesh());
    const Mesh & m = *solver.get_mesh();
    double sum = 0.0;
    uint32_t count = 0;
    for (uint32_t i = 0; i < m.n_cells; i++) {
        if (double(m.h_cell_coords(i, 1)) > 0.6) continue;
        for (size_t v = 0; v < 4; v++) sum += std::abs(rhs[i][v] - exact[i][v]);
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
    verification::expect_order({"axisymmetric", "TENO" + std::to_string(order), verification::cells_of(mesh),
                                "mean residual error, manufactured solution", {16, 32},
                                {mms_residual_error(mesh, 16, teno), mms_residual_error(mesh, 32, teno)},
                                order - 1.0, order - 1 - 0.3});
}

INSTANTIATE_TEST_SUITE_P(Axisymmetric, AxisymmetricOrder,
                         ::testing::Combine(::testing::Values("cartesian", "cartesian_tri"), ::testing::Values(3, 4, 5)));

namespace {

constexpr double SHELL_IN = 1.0, SHELL_OUT = 2.0;

/**
 * @brief The meridian half-plane of the spherical shell SHELL_IN <= |x| <=
 *        SHELL_OUT (half_annulus_gmsh), geometry "curved" (arcs projected
 *        onto the circles), "high_order" (P2 lines from the file) or
 *        "straight" (curved_geometry = false).
 */
std::string shell_input(const std::string & file, const std::string & geometry, const std::string & reconstruction,
                        const std::string & init, const std::string & arcs) {
    std::ostringstream s;
    s << "[run]\nn_steps = 20\ncfl = 0.4\n[mesh]\ntype = \"file\"\nfilename = \"" << file << "\"\n";
    if (geometry == "straight") s << "curved_geometry = false\n";
    if (geometry == "curved") {
        s << "[[mesh.curved]]\nzone = \"inner\"\ntype = \"circle\"\ncenter = [0.0, 0.0]\nradius = " << SHELL_IN << "\n"
          << "[[mesh.curved]]\nzone = \"outer\"\ntype = \"circle\"\ncenter = [0.0, 0.0]\nradius = " << SHELL_OUT << "\n";
    }
    s << "[initialize]\n" << init << "[[boundaries]]\nname = \"axis\"\ntype = \"symmetry\"\n"
      << "[[boundaries]]\nname = \"inner\"\n" << arcs << "\n[[boundaries]]\nname = \"outer\"\n" << arcs << "\n"
      << "[numerics]\nriemann_solver = \"HLLC\"\ntime_integrator = \"SSPRK3\"\n"
      << "[numerics.face_reconstruction]\n" << reconstruction << "\n"
      << "[physics]\ntype = \"euler\"\naxisymmetric = true\ngamma = 1.4\np_ref = 1.0\nT_ref = 1.0\nrho_ref = 1.0\n"
      << "[output]\ncheck_interval = 1000000\n";
    return s.str();
}

std::string shell_file(uint32_t n, int order = 1, bool triangles = false) {
    return write_temp("axisymmetric_shell_" + std::to_string(n) + "_" + std::to_string(order) + (triangles ? "_tri" : "") +
                          ".msh",
                      half_annulus_gmsh(n, 3 * n, SHELL_IN, SHELL_OUT, order, triangles));
}

} // namespace

TEST(AxisymmetricCurved, RevolvedMeasuresOfCurvedCellsAndFacesAreExact) {
    // Projected arcs make the mesh the exact spherical shell: its volume, its
    // first moments and its spheres' areas, per radian. The straight polygons
    // miss each by O(h^2)
    for (const bool triangles : {false, true}) {
        Solver solver;
        solver.set_distributed(false);
        solver.init(parse_toml(shell_input(shell_file(4, 1, triangles), "curved", "type = \"FO\"",
                                           "type = \"constant\"\nu = [0.0, 0.0]\np = 1.0\nT = 1.0\n",
                                           "type = \"symmetry\"")));
        const Mesh & mesh = *solver.get_mesh();
        ASSERT_TRUE(mesh.curved_geometry);
        double volume = 0.0, moment_x = 0.0, moment_r = 0.0;
        for (uint32_t c = 0; c < mesh.n_cells; c++) {
            const double m = double(mesh.h_cell_measure(c));
            volume += m;
            moment_x += m * double(mesh.h_cell_coords(c, 0));
            moment_r += m * double(mesh.h_cell_coords(c, 1));
        }
        const double a = SHELL_IN, b = SHELL_OUT;
        EXPECT_NEAR(volume, 2.0 / 3.0 * (b * b * b - a * a * a), roundoff(1e-13)) << triangles;
        EXPECT_NEAR(moment_x, 0.0, roundoff(1e-13)) << triangles;
        EXPECT_NEAR(moment_r, PI / 8.0 * (b * b * b * b - a * a * a * a), roundoff(1e-13)) << triangles;
        double sphere[2] = {0.0, 0.0};
        for (uint32_t f = 0; f < mesh.n_faces; f++) {
            if (mesh.h_cells_of_face(f, 1) >= 0 || !mesh.curved_geometry->face_is_curved(f)) continue;
            const double r = std::hypot(double(mesh.h_face_coords(f, 0)), double(mesh.h_face_coords(f, 1)));
            sphere[r < 1.5 ? 0 : 1] += double(mesh.h_face_measure(f));
        }
        EXPECT_NEAR(sphere[0], 2.0 * a * a, roundoff(1e-13)) << triangles;
        EXPECT_NEAR(sphere[1], 2.0 * b * b, roundoff(1e-13)) << triangles;
    }
}

using CurvedUniformParam = std::tuple<std::string, std::string, bool>;
class AxisymmetricCurvedUniform : public ::testing::TestWithParam<CurvedUniformParam> {};

TEST_P(AxisymmetricCurvedUniform, UniformFlowAndGasAtRestStayExactlyUniform) {
    // The r-weighted flux points of every curved face sum to its int r n dS,
    // so the faces close each curved cell against the planar area that the
    // pressure source integrates over: without that, curved cells accelerate
    const auto [geometry, reconstruction, triangles] = GetParam();
    const std::string file = shell_file(3, geometry == "high_order" ? 2 : 1, triangles);
    for (const std::string u : {"0.0", "0.5"}) {
        Solver solver;
        solver.set_distributed(false);
        solver.init(parse_toml(shell_input(file, geometry, reconstruction,
                                           "type = \"constant\"\nu = [" + u + ", 0.0]\np = 1.0\nT = 1.0\n",
                                           u == "0.0" ? "type = \"symmetry\"" : "type = \"outlet_extrapolation\"")));
        ASSERT_TRUE(solver.get_mesh()->curved_geometry);
        solver.copy_device_to_host();
        std::vector<std::array<double, N_CONSERVATIVE>> start(solver.get_mesh()->n_owned());
        for (uint32_t c = 0; c < start.size(); c++) FOR_I_CONSERVATIVE start[c][i] = double(solver.h_conservatives(c, i));
        solver.run();
        solver.copy_device_to_host();
        double change = 0.0;
        for (uint32_t c = 0; c < start.size(); c++) {
            FOR_I_CONSERVATIVE change = std::max(change, std::abs(double(solver.h_conservatives(c, i)) - start[c][i]));
        }
        EXPECT_LT(change, roundoff(1e-13)) << "u = " << u;
    }
}

INSTANTIATE_TEST_SUITE_P(
    Axisymmetric, AxisymmetricCurvedUniform,
    ::testing::Combine(::testing::Values("curved", "high_order"),
                       ::testing::Values("type = \"FO\"", "type = \"MUSCL\"", "type = \"TENO\"\norder = 3",
                                         "type = \"TENO\"\norder = 5"),
                       ::testing::Bool()));

namespace {

/** @brief Smooth state [rho, u_x, u_r, p] regular on the axis, its velocity tangent to every sphere |x| = R. */
std::array<double, 4> shell_state(double x, double r) {
    const double h = 0.1 * (1.0 + 0.3 * std::cos(x));
    return {1.0 + 0.2 * std::sin(x) * std::cos(r), -h * r * r, h * r * x, 1.0 + 0.2 * std::cos(x) * std::cos(r)};
}

/** @brief Euler flux F . n of shell_state (n of any length). */
std::array<double, 4> shell_flux(double x, double r, double nx, double nr) {
    const auto [rho, u, v, p] = shell_state(x, r);
    const double un = u * nx + v * nr;
    const double E = p / (GAMMA - 1.0) + 0.5 * rho * (u * u + v * v);
    return {rho * un, rho * u * un + p * nx, rho * v * un + p * nr, (E + p) * un};
}

/**
 * @brief dU/dt of each cell's r-weighted average under shell_state, over the
 *        exact cells (arcs on the circles): int F . n r dS, int p dA = closed
 *        int Q n_x dS with dQ/dx = p, and int r dA = closed int r^2 / 2 n_r dS.
 */
std::vector<std::array<double, 4>> shell_exact_rhs(const Mesh & mesh) {
    const double gx[8] = {0.0198550717512319, 0.1016667612931866, 0.2372337950418355, 0.4082826787521751,
                          0.5917173212478249, 0.7627662049581645, 0.8983332387068134, 0.9801449282487681};
    const double gw[8] = {0.0506142681451881, 0.1111905172266872, 0.1568533229389436, 0.1813418916891810,
                          0.1813418916891810, 0.1568533229389436, 0.1111905172266872, 0.0506142681451881};
    std::vector<std::array<double, 4>> out(mesh.n_cells);
    for (uint32_t c = 0; c < mesh.n_cells; c++) {
        const uint32_t nn = mesh.h_n_nodes_of_cell(c);
        double o[2] = {0.0, 0.0};
        for (uint32_t k = 0; k < nn; k++) {
            for (uint32_t d = 0; d < 2; d++) o[d] += double(mesh.h_node_coords(mesh.h_node_of_cell(c, k), d)) / nn;
        }
        std::array<double, 4> flux = {};
        double source = 0.0, volume = 0.0;
        for (uint32_t k = 0; k < mesh.h_n_faces_of_cell(c); k++) {
            const uint32_t f = mesh.h_face_of_cell(c, k);
            double a[2], b[2];
            for (uint32_t d = 0; d < 2; d++) {
                a[d] = double(mesh.h_node_coords(mesh.h_node_of_face(f, 0), d));
                b[d] = double(mesh.h_node_coords(mesh.h_node_of_face(f, 1), d));
            }
            // Out of the cell: the side of the chord away from the cell's vertex average
            const double cross = (0.5 * (a[0] + b[0]) - o[0]) * (b[1] - a[1]) - (0.5 * (a[1] + b[1]) - o[1]) * (b[0] - a[0]);
            const double side = cross > 0.0 ? 1.0 : -1.0;
            const double R = std::hypot(a[0], a[1]);
            const bool arc = mesh.h_cells_of_face(f, 1) < 0 && std::abs(R - std::hypot(b[0], b[1])) < 1e-12 &&
                             (std::abs(R - SHELL_IN) < 1e-12 || std::abs(R - SHELL_OUT) < 1e-12);
            const double ta = std::atan2(a[1], a[0]), tb = std::atan2(b[1], b[0]);
            for (size_t q = 0; q < 8; q++) {
                double x, r, dx, dr;
                if (arc) {
                    const double t = ta + gx[q] * (tb - ta);
                    x = R * std::cos(t);
                    r = R * std::sin(t);
                    dx = -R * std::sin(t) * (tb - ta);
                    dr = R * std::cos(t) * (tb - ta);
                } else {
                    x = a[0] + gx[q] * (b[0] - a[0]);
                    r = a[1] + gx[q] * (b[1] - a[1]);
                    dx = b[0] - a[0];
                    dr = b[1] - a[1];
                }
                const double nx = side * dr * gw[q], nr = -side * dx * gw[q];
                const auto F = shell_flux(x, r, nx, nr);
                for (size_t i = 0; i < 4; i++) flux[i] += r * F[i];
                source += (x + 0.2 * std::sin(x) * std::cos(r)) * nx;
                volume += 0.5 * r * r * nr;
            }
        }
        flux[2] -= source;
        for (size_t i = 0; i < 4; i++) out[c][i] = -flux[i] / volume;
    }
    return out;
}

/** @brief Mean error of the RHS at the exact averages of shell_state, over every cell of the shell. */
double shell_residual_error(uint32_t n, const std::string & geometry, const std::string & reconstruction) {
    Solver solver;
    solver.set_distributed(false);
    solver.init(parse_toml(shell_input(shell_file(n), geometry, reconstruction,
                                       "type = \"analytical\"\nn_subdivisions = 6\n"
                                       "rho = \"1 + 0.2 * sin(x) * cos(y)\"\n"
                                       "u = [\"-0.1 * (1 + 0.3 * cos(x)) * y^2\", \"0.1 * (1 + 0.3 * cos(x)) * y * x\"]\n"
                                       "p = \"1 + 0.2 * cos(x) * cos(y)\"\n",
                                       "type = \"symmetry\"")));
    const auto rhs = host_rhs(solver);
    const auto exact = shell_exact_rhs(*solver.get_mesh());
    const uint32_t n_cells = solver.get_mesh()->n_owned();
    double sum = 0.0;
    for (uint32_t c = 0; c < n_cells; c++) {
        for (size_t v = 0; v < 4; v++) sum += std::abs(rhs[c][v] - exact[c][v]);
    }
    return sum / n_cells;
}

} // namespace

class AxisymmetricCurvedOrder : public ::testing::TestWithParam<int> {};

TEST_P(AxisymmetricCurvedOrder, ResidualBetweenCurvedSpheresConvergesAtDesignOrder) {
    // The spatial operator at the exact r-weighted averages of a smooth state
    // tangent to the slip walls of a spherical shell, over every cell, those
    // on the curved walls and on the axis included. Straight walls stall
    SKIP_IN_SINGLE_PRECISION("the finest residual errors are near single-precision round-off");
    const int order = GetParam();
    const std::string teno = "type = \"TENO\"\norder = " + std::to_string(order);
    const std::vector<double> errors = {shell_residual_error(8, "curved", teno), shell_residual_error(16, "curved", teno)};
    verification::expect_order({"axisymmetric", "TENO" + std::to_string(order), "quadrilaterals, curved walls",
                                "mean residual error, spherical shell", {8, 16}, errors, order - 1.0,
                                order - 1 - 0.3});
    EXPECT_LT(errors[1], 0.1 * shell_residual_error(16, "straight", teno));
}

INSTANTIATE_TEST_SUITE_P(Axisymmetric, AxisymmetricCurvedOrder, ::testing::Values(3, 4));

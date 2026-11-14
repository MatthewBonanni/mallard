/**
 * @file region_motion_test.cpp
 * @author Matthew Bonanni (mbonanni001@gmail.com)
 * @brief Arrays of copies, regions that move, and regions that wrap across
 *        periodic boundaries.
 * @version 0.1
 * @date 2026-11-10
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 *
 */

#include <gtest/gtest.h>
#include <Kokkos_Core.hpp>

#include <cmath>
#include <limits>
#include <numbers>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "gmsh_fixtures.h"
#include "input.h"
#include "mesh.h"
#include "region_input.h"
#include "region_mesh.h"
#include "region_motion.h"
#include "solver.h"
#include "test_fixtures.h"

namespace {

region::Region read(const std::string & table, const region::MeshRegions * automatic = nullptr) {
    return region::read(parse_toml("r = " + table).at("r"), "r", nullptr, automatic);
}

std::string point(const double x, const double y, const double z = 0.5) {
    std::string s = "[" + std::to_string(x) + ", " + std::to_string(y);
    if (N_DIM == 3) s += ", " + std::to_string(z);
    return s + "]";
}

std::string ball(const std::string & center, const double radius) {
    return std::string("{type = \"") + (N_DIM == 2 ? "disk" : "sphere") + "\", center = " + center +
           ", radius = " + std::to_string(radius);
}

/** @brief The unit square (jittered triangles and quadrilaterals) or cube (tetrahedra). */
std::shared_ptr<Mesh> unit_mesh() {
    auto mesh = std::make_shared<Mesh>();
#if N_DIM == 2
    mesh->init_file(write_temp_shared("mallard_region_motion.msh", jittered_mixed_mesh(10)));
#else
    mesh->init_cart_3d(6, 6, 6, 1.0, 1.0, 1.0, MeshType::CARTESIAN_TET);
#endif
    mesh->copy_host_to_device();
    return mesh;
}

double integral(const Mesh & mesh, const std::vector<double> & f) {
    double sum = 0.0;
    for (uint32_t c = 0; c < mesh.n_owned(); c++) sum += f[c] * double(mesh.h_cell_measure(c));
    return sum;
}

std::vector<double> host(const region::MovingFractions & m) {
    const auto h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), m.fractions());
    return std::vector<double>(h.data(), h.data() + h.extent(0));
}

/** @brief A solver's mesh: gas at rest, with the mesh table given and symmetry on the zones listed. */
std::unique_ptr<Solver> solver_on(const std::string & mesh, const std::vector<std::string> & zones) {
    std::string s = "[run]\nn_steps = 1\ncfl = 0.5\n[mesh]\n" + mesh +
                    "[initialize]\ntype = \"constant\"\nu = " + (N_DIM == 2 ? "[0.0, 0.0]" : "[0.0, 0.0, 0.0]") +
                    "\np = 1.0\nT = 1.0\n[physics]\ntype = \"euler\"\ngamma = 1.4\np_ref = 1.0\nT_ref = 1.0\nrho_ref = 1.0\n"
                    "[numerics]\nriemann_solver = \"HLLC\"\ntime_integrator = \"SSPRK3\"\n"
                    "[numerics.face_reconstruction]\ntype = \"MUSCL\"\n[output]\ncheck_interval = 1000000\n";
    for (const auto & z : zones) s += "[[boundaries]]\nname = \"" + z + "\"\ntype = \"symmetry\"\n";
    auto solver = std::make_unique<Solver>();
    solver->set_distributed(false);
    solver->init(parse_toml(s));
    return solver;
}

/** @brief A unit square of right triangles (unit cube of tetrahedra), n cells per side. */
std::shared_ptr<Mesh> simplex_mesh(const uint32_t n) {
    auto mesh = std::make_shared<Mesh>();
#if N_DIM == 2
    mesh->init_cart_tri(n, n, 1.0, 1.0);
#else
    mesh->init_cart_3d(n, n, n, 1.0, 1.0, 1.0, MeshType::CARTESIAN_TET);
#endif
    mesh->copy_host_to_device();
    return mesh;
}

/**
 * @brief The cell fields of a gas of R = 1, gamma = 1.4 and density 1, with the
 *        pressure p(x) and the velocity u(x, out) at each cell centroid.
 */
template <typename F, typename G>
region::CellFields gas(const Mesh & mesh, F p, G u) {
    region::CellFields fields;
    fields.coords = mesh.cell_coords;
    fields.R = 1.0;
    fields.gamma = 1.4;
    fields.W = Kokkos::View<rtype *[N_CONSERVATIVE]>("W", mesh.n_cells);
    const auto W = Kokkos::create_mirror_view(fields.W);
    const auto x = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), mesh.cell_coords);
    for (uint32_t c = 0; c < mesh.n_cells; c++) {
        const double at[3] = {double(x(c, 0)), double(x(c, 1)), N_DIM == 3 ? double(x(c, N_DIM - 1)) : 0.0};
        double v[3] = {0.0, 0.0, 0.0};
        u(at, v);
        W(c, 0) = 1.0;
        for (int d = 0; d < N_DIM; d++) W(c, 1 + d) = static_cast<rtype>(v[d]);
        W(c, N_DIM + 1) = static_cast<rtype>(p(at));
    }
    Kokkos::deep_copy(fields.W, W);
    return fields;
}

/** @brief The cell fields of a gas at rest, R = 1 and density 1, with the pressure p(x) at each cell centroid. */
template <typename F>
region::CellFields gas_with_pressure(const Mesh & mesh, F p) {
    return gas(mesh, p, [](const double *, double *) {});
}

/** @brief The measure of a solution-defined region on the mesh: the integral of its fractions, or its elements' total. */
double solution_measure(const Mesh & mesh, const std::string & table, const region::CellFields & fields,
                        const std::vector<std::string> & species = {}) {
    const region::Region r = read(table);
    const auto levels = std::make_shared<const region::SolutionLevels>(mesh, r.conditions(), species, fields);
    if (r.dimension() == N_DIM) {
        const region::MovingFractions moving(mesh, r, levels);
        moving.update(0.0);
        return integral(mesh, host(moving));
    }
    double total = 0.0;
    for (const double m : region::elements(mesh, r, region::CrossedLevels(mesh, r, levels).update(0.0)).measure) total += m;
    return total;
}

/** @brief The levels of a region's conditions in every owned cell, on the host. */
region::Levels owned_levels(const Mesh & mesh, const region::SolutionLevels & levels) {
    levels.update(0.0);
    region::Levels out;
    for (uint32_t c = 0; c < mesh.n_owned(); c++) out.cells.push_back(c);
    const auto h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), levels.levels());
    Kokkos::View<double **[N_DIM + 1], Kokkos::LayoutRight, Kokkos::HostSpace> values("levels", h.extent(0), h.extent(1));
    Kokkos::deep_copy(values, h);
    out.values = values;
    return out;
}

} // namespace

TEST(RegionMotionTest, ARepeatIsTheUnionOfItsCopies) {
    const std::string one = ball(point(0.2, 0.3), 0.08);
    const region::Region array = read(one + ", repeat = [{count = 3, translate = " + point(0.25, 0.0, 0.0) +
                                      "}, {count = 2, translate = " + point(0.0, 0.4, 0.0) + "}]}");
    ASSERT_EQ(array.copies().size(), 6u);
    std::string regions;
    for (int j = 0; j < 2; j++) {
        for (int i = 0; i < 3; i++) regions += (regions.empty() ? "" : ", ") + ball(point(0.2 + 0.25 * i, 0.3 + 0.4 * j), 0.08) + "}";
    }
    const region::Region joined = read("{type = \"union\", regions = [" + regions + "]}");
    for (int k = 0; k < 200; k++) {
        const double x[3] = {0.05 * (k % 20), 0.1 * (k / 20), N_DIM == 2 ? 0.0 : 0.5};
        EXPECT_NEAR(array.distance(x), joined.distance(x), 1e-15);
    }
    // Copy k: the first level listed fastest
    for (uint32_t k = 0; k < 6; k++) {
        const double center[3] = {0.2 + 0.25 * (k % 3), 0.3 + 0.4 * (k / 3), N_DIM == 2 ? 0.0 : 0.5};
        const region::Region copy = array.copy(k);
        EXPECT_NEAR(copy.distance(center), -0.08, 1e-15) << k;
        const double other[3] = {0.2 + 0.25 * ((k + 1) % 3), center[1], N_DIM == 2 ? 0.0 : 0.5};
        EXPECT_GT(copy.distance(other), 0.0) << k;
    }
    // Rotated copies about a pivot
    const std::string rotate = N_DIM == 2 ? "90" : "{axis = [0, 0, 1], angle = 90}";
    const region::Region ring = read(ball(point(0.8, 0.5), 0.05) + ", repeat = {count = 4, rotate = " + rotate +
                                     ", pivot = " + point(0.5, 0.5) + "}}");
    const double z = N_DIM == 2 ? 0.0 : 0.5, at_90[3] = {0.5, 0.8, z}, at_270[3] = {0.5, 0.2, z};
    EXPECT_NEAR(ring.copy(1).distance(at_90), -0.05, 1e-14);
    EXPECT_NEAR(ring.copy(3).distance(at_270), -0.05, 1e-14);
}

TEST(RegionMotionTest, DisjointCopiesMeasureTheirSumInClosedForm) {
    const region::Measures single = region::measures(read(ball(point(0.2, 0.3), 0.08) + "}"), false);
    const region::Measures array = region::measures(
        read(ball(point(0.2, 0.3), 0.08) + ", repeat = {count = 3, translate = " + point(0.25, 0.0, 0.0) + "}}"), false);
    EXPECT_EQ(array.tolerance, 0.0);
    EXPECT_NEAR(array.volume, 3.0 * single.volume, 1e-15);
    EXPECT_NEAR(array.surface, 3.0 * single.surface, 1e-15);
    // Overlapping copies are measured by subdivision, below the sum
    const region::Measures overlap = region::measures(
        read(ball(point(0.2, 0.3), 0.08) + ", repeat = {count = 3, translate = " + point(0.1, 0.0, 0.0) + "}}"), false);
    EXPECT_GT(overlap.tolerance, 0.0);
    EXPECT_LT(overlap.volume, 3.0 * single.volume);
}

TEST(RegionMotionTest, RegionsAcrossPeriodicBoundariesWrapIntoTheDomain) {
    // Periodic in x: a ball across x = 1 covers its whole measure in the domain, like one inside it
    const std::string box = N_DIM == 2 ? "type = \"cartesian\"\nNx = 16\nNy = 16\n[[periodic]]\nzones = [\"left\", \"right\"]\n"
                                       : "type = \"cartesian\"\nNx = 8\nNy = 8\nNz = 8\n[[periodic]]\nzones = [\"left\", \"right\"]\n";
    std::vector<std::string> walls = {"bottom", "top"};
    if (N_DIM == 3) walls.insert(walls.end(), {"back", "front"});
    const auto periodic = solver_on(box, walls);
    const Mesh & mesh = *periodic->get_mesh();
    const region::MeshRegions automatic(mesh);
    const region::Region across = read(ball(point(0.95, 0.5), 0.2) + "}", &automatic);
    const region::Region inside = read(ball(point(0.5, 0.5), 0.2) + "}", &automatic);
    EXPECT_NE(across.summary().find("wrapped across periodic boundaries"), std::string::npos) << across.summary();
    const double exact = region::measures(inside, false).volume;
    const region::Measures own = region::measures(across, false);
    EXPECT_EQ(own.tolerance, 0.0);
    EXPECT_NEAR(own.volume, exact, 1e-15);
    const std::vector<double> f = region::cell_fractions(mesh, across);
    EXPECT_NEAR(integral(mesh, f), integral(mesh, region::cell_fractions(mesh, inside)), (N_DIM == 2 ? 1e-5 : 1e-4) * exact);
    EXPECT_NEAR(region::measures_in_domain(mesh, across, f).volume, exact, 1e-3 * exact);
    // Points past x = 0 are inside where their image across x = 1 is
    const double image[3] = {0.05, 0.5, N_DIM == 2 ? 0.0 : 0.5};
    EXPECT_LT(across.distance(image), 0.0);

    // Rotationally periodic: a ball across the seam of a sector of an annulus
    std::string annulus = "type = \"annulus\"\nNtheta = 12\nNr = 8\nr_inner = 1.0\nr_outer = 2.0\nangle = 90\n";
    if (N_DIM == 3) annulus += "Nz = 6\nLz = 1.0\n";
    std::vector<std::string> rims = {"inner", "outer"};
    if (N_DIM == 3) rims.insert(rims.end(), {"back", "front"});
    const auto sector = solver_on(annulus + "[[periodic]]\nzones = [\"start\", \"end\"]\n", rims);
    const Mesh & s_mesh = *sector->get_mesh();
    const region::MeshRegions s_automatic(s_mesh);
    const region::Region seam = read(ball(point(1.5, 0.0), 0.2) + "}", &s_automatic);
    const std::vector<double> g = region::cell_fractions(s_mesh, seam);
    EXPECT_NEAR(integral(s_mesh, g), exact, 1e-2 * exact);
}

TEST(RegionMotionTest, MovingFractionsAreTheFractionsOfTheRegionWhereItIs) {
    const auto mesh = unit_mesh();
    const std::string center = N_DIM == 2 ? "[\"0.3 + 0.2 * t\", 0.5]" : "[\"0.3 + 0.2 * t\", 0.5, \"0.5 - 0.1 * t\"]";
    for (const std::string & table :
         {ball(center, 0.2) + "}",
          ball(point(0.4, 0.5), 0.2) + ", rotate = " + (N_DIM == 2 ? "\"30 * t\"" : "{axis = [0, 0, 1], angle = \"30 * t\"}") +
              ", pivot = " + point(0.5, 0.5) + ", scale = \"1 + 0.5 * t\", edge = 0.05}",
          std::string("{type = \"difference\", a = ") + ball(center, 0.25) + "}, b = {type = \"implicit\", f = \"y - 0.4 - 0.1 * t\"}}",
          ball(center, 0.1) + ", repeat = {count = 3, translate = " + point(0.2, 0.0, 0.0) + "}}"}) {
        const region::Region r = read(table);
        ASSERT_TRUE(r.moving()) << table;
        const region::MovingFractions moving(*mesh, r);
        // Times the device holds exactly: its time is an rtype, as the solver's
        for (const double t : {0.0, static_cast<double>(static_cast<rtype>(0.7))}) {
            moving.update(t);
            const std::vector<double> expected = region::cell_fractions(*mesh, r.at(t)), got = host(moving);
            ASSERT_EQ(got.size(), expected.size());
            for (size_t c = 0; c < got.size(); c++) EXPECT_NEAR(got[c], expected[c], 1e-14) << table << " t " << t << " cell " << c;
        }
    }
    // Where it is at t is the fixed region placed there
    const region::Region r = read(ball(center, 0.2) + "}");
    const region::Region fixed = read(ball(N_DIM == 2 ? point(0.44, 0.5) : point(0.44, 0.5, 0.43), 0.2) + "}");
    EXPECT_TRUE(r.rigid());
    EXPECT_FALSE(read(ball(point(0.4, 0.5), 0.2) + ", scale = \"1 + t\"}").rigid());
    for (int k = 0; k < 50; k++) {
        const double x[3] = {0.02 * k, 0.5 + 0.01 * k, 0.45};
        EXPECT_NEAR(r.at(0.7).distance(x), fixed.distance(x), 1e-15);
    }
}

#if N_DIM == 2
TEST(RegionMotionTest, MovingFractionsFollowCurvedWalls) {
    // A disk crossing the curved outer wall of an annulus: its cones there, through their interpolants
    const std::string file = write_temp_shared("mallard_region_motion_annulus.msh", annulus_gmsh(4, 32));
    const auto solver = solver_on("type = \"file\"\nfilename = \"" + file + "\"\n"
                                  "[[mesh.curved]]\nzone = \"inner\"\ntype = \"circle\"\ncenter = [0.0, 0.0]\nradius = 1.0\n"
                                  "[[mesh.curved]]\nzone = \"outer\"\ntype = \"circle\"\ncenter = [0.0, 0.0]\nradius = 1.384\n",
                                  {"inner", "outer"});
    const Mesh & mesh = *solver->get_mesh();
    ASSERT_TRUE(mesh.curved_geometry);
    const region::Region r = read("{type = \"disk\", center = [\"1.384 * cos(t)\", \"1.384 * sin(t)\"], radius = 0.15}");
    const region::MovingFractions moving(mesh, r);
    for (const double t : {0.3, 2.0}) {
        moving.update(t);
        const std::vector<double> expected = region::cell_fractions(mesh, r.at(t)), got = host(moving);
        double cut = 0.0;
        for (size_t c = 0; c < got.size(); c++) {
            EXPECT_NEAR(got[c], expected[c], 1e-9) << "t " << t << " cell " << c;
            if (mesh.curved_geometry->cell_is_curved(static_cast<uint32_t>(c)) && expected[c] > 0.0 && expected[c] < 1.0) cut++;
        }
        EXPECT_GT(cut, 0.0) << "no curved cell is cut at t " << t;
    }
}
#endif

TEST(RegionMotionTest, ConditionsOfLinearFieldsCoverTheirExactVolume) {
    // Each cell's level is linear and exact for a linear field, and its fraction its exact clip
    const auto mesh = unit_mesh();
    const auto fields = gas_with_pressure(*mesh, [](const double * x) { return 1.0 + x[0] + 0.5 * x[1]; });
    const double tol = std::max(1e-13, 16.0 * double(std::numeric_limits<rtype>::epsilon()));
    EXPECT_NEAR(solution_measure(*mesh, "{type = \"where\", condition = \"P < 1.6\"}", fields), 0.35, tol);
    EXPECT_NEAR(solution_measure(*mesh, "{type = \"where\", condition = \"1.6 < P\"}", fields), 0.65, tol);
    // The same field as a temperature, T = p / rho R
    EXPECT_NEAR(solution_measure(*mesh, "{type = \"where\", condition = \"T < 1.6\"}", fields), 0.35, tol);
    // Booleans with regions of the mesh: here the half-plane is cut at x = 0.4 by a face of no cell
    EXPECT_NEAR(solution_measure(*mesh, "{type = \"intersection\", regions = [{type = \"where\", condition = "
                                        "\"P < 1.6\"}, {type = \"implicit\", f = \"y - 0.6\"}]}", fields),
                0.6 * 0.6 - 0.25 * 0.36, 2e-3);
}

TEST(RegionMotionTest, DerivedFieldsOfLinearFieldsCoverTheirExactVolume) {
    // Each condition below is linear in x on these fields, so its region is exact
    const auto mesh = unit_mesh();
    const double tol = std::max(1e-12, 64.0 * double(std::numeric_limits<rtype>::epsilon()));
    const auto where = [](const std::string & condition) { return "{type = \"where\", condition = \"" + condition + "\"}"; };
    // H = gamma / (gamma - 1) p / rho
    const auto layered = gas_with_pressure(*mesh, [](const double * x) { return 1.0 + x[0] + 0.5 * x[1]; });
    EXPECT_NEAR(solution_measure(*mesh, where("H < 5.6"), layered), 0.35, tol);
    // Its isosurface: x + y / 2 = 0.6
    EXPECT_NEAR(solution_measure(*mesh, "{type = \"isosurface\", field = \"H\", value = 5.6}", layered),
                std::hypot(0.5, 1.0), tol);
    // u = (x, 0): MACH = x / sqrt(gamma), and Q = -(du_x/dx)^2 / 2 = -1/2
    const auto stretched = gas(*mesh, [](const double *) { return 1.0; }, [](const double * x, double * u) { u[0] = x[0]; });
    EXPECT_NEAR(solution_measure(*mesh, where("MACH * sqrt(1.4) < 0.3"), stretched), 0.3, tol);
    EXPECT_NEAR(solution_measure(*mesh, where("Q_CRITERION * x > -0.15"), stretched), 0.3, tol);
    // Rotation at rate 2 about (0.5, 0.5): vorticity 4 about z, Q = 4
    const auto rotating = gas(*mesh, [](const double *) { return 1.0; }, [](const double * x, double * u) {
        u[0] = -2.0 * (x[1] - 0.5);
        u[1] = 2.0 * (x[0] - 0.5);
    });
    EXPECT_NEAR(solution_measure(*mesh, where(N_DIM == 2 ? "VORTICITY * x < 1.6" : "VORTICITY_Z * x < 1.6"), rotating),
                0.4, tol);
    EXPECT_NEAR(solution_measure(*mesh, where("Q_CRITERION * x < 1.6"), rotating), 0.4, tol);
    if constexpr (N_DIM == 3) {
        EXPECT_NEAR(solution_measure(*mesh, where("abs(VORTICITY_X) + abs(VORTICITY_Y) < 1e-6"), rotating), 1.0, tol);
    }
    // Mixtures: X_k = (Y_k / W_k) / sum_j (Y_j / W_j), H = e0 + gamma / (gamma - 1) p / rho
    region::CellFields mixture = layered;
    mixture.R = 0.0;
    mixture.R_universal = 1.0;
    mixture.inv_W = Kokkos::View<double *>("inv_W", 2);
    const auto inv_W = Kokkos::create_mirror_view(mixture.inv_W);
    inv_W(0) = 0.5;
    inv_W(1) = 0.25;
    Kokkos::deep_copy(mixture.inv_W, inv_W);
    mixture.Y = Kokkos::View<rtype **, Kokkos::LayoutRight>("Y", mesh->n_cells, 4);
    const auto Y = Kokkos::create_mirror_view(mixture.Y);
    const auto x = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), mesh->cell_coords);
    for (uint32_t c = 0; c < mesh->n_cells; c++) {
        // X_A = 2 Y_A / (1 + Y_A) = x
        const double X_A = double(x(c, 0));
        Y(c, 0) = static_cast<rtype>(X_A / (2.0 - X_A));
        Y(c, 1) = static_cast<rtype>(1.0 - X_A / (2.0 - X_A));
        Y(c, 2) = 1.4_r;
        Y(c, 3) = 2.0_r;
    }
    Kokkos::deep_copy(mixture.Y, Y);
    const std::vector<std::string> species = {"A", "B-2"};
    EXPECT_NEAR(solution_measure(*mesh, where("X_A < 0.3"), mixture, species), 0.3, tol);
    EXPECT_NEAR(solution_measure(*mesh, where("X_B_2 > 0.7"), mixture, species), 0.3, tol);
    EXPECT_NEAR(solution_measure(*mesh, where("H < 7.6"), mixture, species), 0.35, tol);
}

TEST(RegionMotionTest, ConditionVolumesAndIsosurfaceAreasConvergeAtSecondOrder) {
    // p = 1 + r^2: where p < 1 + R^2 is the disk (ball) of radius R, and its isosurface the circle (sphere)
    const double R = 0.3, pi = std::numbers::pi;
    const double volume = N_DIM == 2 ? pi * R * R : 4.0 / 3.0 * pi * R * R * R;
    const double area = N_DIM == 2 ? 2.0 * pi * R : 4.0 * pi * R * R;
    const std::string value = std::to_string(1.0 + R * R);
    const std::string where = "{type = \"where\", condition = \"P < " + value + "\"}";
    const std::string surface = "{type = \"isosurface\", field = \"P\", value = " + value + "}";
    const std::string inner = "{type = \"boundary\", of = " + where + "}";
    std::vector<double> volume_errors, area_errors;
    for (const uint32_t n : N_DIM == 2 ? std::vector<uint32_t>{10, 20, 40} : std::vector<uint32_t>{6, 12, 24}) {
        const auto mesh = simplex_mesh(n);
        const auto fields = gas_with_pressure(*mesh, [](const double * x) {
            double r2 = 0.0;
            for (int d = 0; d < N_DIM; d++) r2 += (x[d] - 0.5) * (x[d] - 0.5);
            return 1.0 + r2;
        });
        volume_errors.push_back(std::fabs(solution_measure(*mesh, where, fields) - volume));
        const double a = solution_measure(*mesh, surface, fields);
        area_errors.push_back(std::fabs(a - area));
        // The boundary of a where region is its isosurface
        EXPECT_NEAR(solution_measure(*mesh, inner, fields), a, 1e-12 * area) << n;
    }
    for (size_t i = 1; i < volume_errors.size(); i++) {
        EXPECT_GT(std::log2(volume_errors[i - 1] / volume_errors[i]), 1.7) << volume_errors[i - 1] << " " << volume_errors[i];
        EXPECT_GT(std::log2(area_errors[i - 1] / area_errors[i]), 1.7) << area_errors[i - 1] << " " << area_errors[i];
    }
    EXPECT_LT(volume_errors.back(), 0.01 * volume);
    EXPECT_LT(area_errors.back(), 0.01 * area);
}

TEST(RegionMotionTest, SurfacesOfTheSolutionAreCutFromTheLevelsOfTheCellsTheyMayCross) {
    // p = 1 + r^2; the isosurface p = 1.09 (r = 0.3), alone and clipped by a plane, then a plane clipped by a condition
    const auto mesh = simplex_mesh(N_DIM == 2 ? 20 : 8);
    const auto fields = gas_with_pressure(*mesh, [](const double * x) {
        double r2 = 0.0;
        for (int d = 0; d < N_DIM; d++) r2 += (x[d] - 0.5) * (x[d] - 0.5);
        return 1.0 + r2;
    });
    const std::string where = "{type = \"where\", condition = \"P < 1.09\"}";
    const std::string right_half = "{type = \"implicit\", f = \"x - 0.5\"}";
    const std::vector<std::pair<std::string, bool>> cases = {
        {"{type = \"isosurface\", field = \"P\", value = 1.09}", false},
        {"{type = \"boundary\", of = {type = \"intersection\", regions = [" + where + ", " + right_half + "]}}", true},
    };
    for (const auto & [table, every] : cases) {
        const region::Region r = read(table);
        const auto levels = std::make_shared<const region::SolutionLevels>(*mesh, r.conditions(), std::vector<std::string>{}, fields);
        const region::Levels crossed = region::CrossedLevels(*mesh, r, levels).update(0.0);
        if (every) {
            EXPECT_EQ(crossed.cells.size(), mesh->n_owned()) << table;
        } else {
            EXPECT_GT(crossed.cells.size(), 0u) << table;
            EXPECT_LT(crossed.cells.size(), mesh->n_owned() / 2) << table;
        }
        const region::Elements a = region::elements(*mesh, r, crossed), b = region::elements(*mesh, r, owned_levels(*mesh, *levels));
        ASSERT_GT(b.size(), 0u) << table;
        EXPECT_EQ(a.cell, b.cell) << table;
        EXPECT_EQ(a.measure, b.measure) << table;
        EXPECT_EQ(a.quad_points, b.quad_points) << table;
        EXPECT_EQ(a.quad_weights, b.quad_weights) << table;
        EXPECT_EQ(a.vertices, b.vertices) << table;
    }
}

/**
 * @file region_motion_test.cpp
 * @author Matthew Bonanni (mbonanni001@gmail.com)
 * @brief Arrays of copies, regions that move, and regions that wrap across
 *        periodic boundaries; exact sums on the device.
 * @version 0.1
 * @date 2026-11-10
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 *
 */

#include <gtest/gtest.h>
#include <Kokkos_Core.hpp>

#include <array>
#include <cmath>
#include <memory>
#include <string>
#include <vector>

#include "device_comm.h"
#include "device_sums.h"
#include "exact_sums.h"
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
        for (const double t : {0.0, 0.7}) {
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

TEST(RegionMotionTest, DeviceSumsAreTheExactSums) {
    const uint32_t n = 1000;
    Kokkos::View<double *[2]> terms("terms", n);
    Kokkos::View<double **, Kokkos::LayoutLeft> columns("columns", n, 2);
    Kokkos::parallel_for("fill", n, KOKKOS_LAMBDA(const uint32_t c) {
        terms(c, 0) = columns(c, 0) = Kokkos::sin(1.0 + c) * 1e-3;
        terms(c, 1) = columns(c, 1) = (c % 7 == 0 ? 1e8 : 1e-8) * Kokkos::cos(double(c));
    });
    const DeviceSums sums(2, n, false);
    sums.sum(columns, false);
    const auto got = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), sums.totals());
    const std::array<double, 2> expected = exact_sums(terms, n, false);
    EXPECT_EQ(got(0), expected[0]);
    EXPECT_EQ(got(1), expected[1]);
}

TEST(RegionMotionTest, DeviceSumsAcrossRanksAreTheSameWithNcclAndMpi) {
    if (!comm::nccl_available() || comm::size() < 2) GTEST_SKIP() << "needs NCCL and more than one rank";
    const uint32_t n = 100 + 37 * static_cast<uint32_t>(comm::rank());
    Kokkos::View<double **, Kokkos::LayoutLeft> columns("columns", n, 2);
    const double r = comm::rank();
    Kokkos::parallel_for("fill", n, KOKKOS_LAMBDA(const uint32_t c) {
        columns(c, 0) = Kokkos::sin(1.0 + c + 100.0 * r);
        columns(c, 1) = Kokkos::cos(c * r) * 1e-6;
    });
    const DeviceSums sums(2, n, true);
    sums.sum(columns, false);
    const auto mpi = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), sums.totals());
    sums.sum(columns, true);
    const auto nccl = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), sums.totals());
    EXPECT_EQ(mpi(0), nccl(0));
    EXPECT_EQ(mpi(1), nccl(1));
}

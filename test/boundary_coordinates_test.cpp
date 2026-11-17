/**
 * @file boundary_coordinates_test.cpp
 * @author Matthew Bonanni (mbonanni001@gmail.com)
 * @brief Tests of the local coordinates of boundary faces in boundary expressions.
 * @version 0.1
 * @date 2026-11-15
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 *
 */

#include <gtest/gtest.h>

#include <cmath>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "boundary_coordinates.h"
#include "boundary_fields.h"
#include "data_paths.h"
#include "mesh.h"
#include "mpi_compare.h"
#include "solver.h"
#include "test_fixtures.h"

namespace {

namespace bv = boundary_variable;

constexpr double tol(double in_double, double in_float) {
    return sizeof(rtype) == sizeof(double) ? in_double : in_float;
}

std::string mesh_block(uint32_t nx, uint32_t n, double Lx, double L) {
    std::ostringstream s;
    s << "[mesh]\ntype = \"cartesian\"\nNx = " << nx << "\nNy = " << n << "\nLx = " << Lx << "\nLy = " << L << "\n";
    if (N_DIM == 3) s << "Nz = " << n << "\nLz = " << L << "\n";
    return s.str();
}

std::shared_ptr<Mesh> box(uint32_t nx, uint32_t n, double Lx, double L) {
    auto mesh = std::make_shared<Mesh>();
    mesh->init(parse_toml(mesh_block(nx, n, Lx, L)));
    return mesh;
}

std::vector<uint32_t> zone_faces(const Mesh & mesh, const std::vector<std::string> & names) {
    std::vector<uint32_t> faces;
    for (const FaceZone & zone : mesh.h_face_zones()) {
        for (const std::string & name : names) {
            if (zone.get_name() != name) continue;
            for (uint32_t i = 0; i < zone.n_faces(); i++) faces.push_back(zone.h_faces(i));
        }
    }
    return faces;
}

std::vector<uint16_t> all_local() {
    std::vector<uint16_t> used;
    for (uint16_t i = bv::FIRST_LOCAL; i < bv::COUNT; i++) used.push_back(i);
    return used;
}

} // namespace

TEST(BoundaryCoordinatesTest, ExpressionsOfAnEntryListTheCoordinatesTheyRead) {
    const Input entry = parse_toml(
        "name = \"inlet\"\ntype = \"inlet_state\"\nu = [\"10 * n\", 0, 0]\np = \"1 bar\"\nT = 300.0\n"
        "where = \"x < 1\"\nX = { O2 = \"0.21 * u_m\", N2 = 0.79 }\nturbulence = { profile = \"bl.csv\" }\n");
    EXPECT_EQ(local_coordinates_used(entry), (std::vector<uint16_t>{bv::N, bv::U_M}));
    EXPECT_TRUE(local_coordinates_used(parse_toml("name = \"inlet\"\nu = [\"x * t\", 0]\n")).empty());
}

TEST(BoundaryCoordinatesTest, LocalVariablesHideTheLocalCoordinates) {
    // Expressions written before the local coordinates keep their meaning in
    // 2D and 3D alike; x, y, z and t still cannot be hidden
    const Input v = parse_toml("v = \"var u := y; var u_m := 2 * x; var n := 1; u + u_m + n\"\n");
    const BoundaryValue value(v.at("v"), "inlet.T", dim::none);
    Point x{};
    x[0] = 0.25;
    x[1] = 0.5;
    EXPECT_DOUBLE_EQ(value.at(FacePoint(x), 0.0), 2.0);
    EXPECT_TRUE(local_coordinates_used(parse_toml("name = \"inlet\"\nT = \"var u := atan2(y, x); u\"\n")).empty());
    EXPECT_EQ(local_coordinates_used(parse_toml("name = \"inlet\"\nT = \"var q := n; q\"\n")),
              (std::vector<uint16_t>{bv::N}));
    EXPECT_THROW(BoundaryValue(parse_toml("v = \"var y := 1; y\"\n").at("v"), "inlet.T", dim::none),
                 std::exception);
}

TEST(BoundaryCoordinatesTest, PlanarInletBetweenWallsHasExactCoordinates) {
    // The left side of a box: u along +y (and v along +z) from 0 to 1, n_m the
    // distance to the nearer of the bottom and top walls and n that over its
    // largest value, half the height
    const double H = 0.5;
    const auto mesh = box(4, 10, 1.0, H);
    const std::vector<uint32_t> faces = zone_faces(*mesh, {"left"});
    const WallDistance walls(*mesh, zone_faces(*mesh, {"bottom", "top"}), false);
    const Input entry = parse_toml("name = \"left\"\n");
    std::vector<uint16_t> used = all_local();
#if N_DIM == 3
    used.resize(used.size() - 3);  // r, r_m and theta belong to rings and disks
#endif
    std::string summary;
    const std::vector<FacePoint> p = local_coordinates(*mesh, faces, entry, "left", used, walls, false, summary);
    ASSERT_EQ(p.size(), faces.size());
    for (const FacePoint & x : p) {
        const double y = x[1];
        EXPECT_NEAR(x[bv::N_M], std::min(y, H - y), tol(1e-15, 1e-6)) << y;
        EXPECT_NEAR(x[bv::N], std::min(y, H - y) / (0.5 * H), tol(1e-14, 1e-6)) << y;
        EXPECT_NEAR(x[bv::U], y / H, tol(1e-12, 1e-6)) << y;
        EXPECT_NEAR(x[bv::U_M], y, tol(1e-12, 1e-6)) << y;
#if N_DIM == 3
        const double z = x[2];
        EXPECT_NEAR(x[bv::V], z / H, tol(1e-12, 1e-6)) << z;
        EXPECT_NEAR(x[bv::V_M], z, tol(1e-12, 1e-6)) << z;
        EXPECT_TRUE(std::isnan(x[bv::R]));
#endif
    }
    EXPECT_NE(summary.find("n_m 0.025 to 0.225"), std::string::npos) << summary;
    EXPECT_NE(summary.find("u 0.05 to 0.95"), std::string::npos) << summary;
#if N_DIM == 3
    EXPECT_NE(summary.find("four-sided patch, (u, v) = (0, 0) at (0, 0, 0), (1, 0) at (0, 0.5, 0)"), std::string::npos)
        << summary;
    // Corners of its own: from the top, u along -y
    const Input cornered = parse_toml("name = \"left\"\ncorners = [[0, 0.5, 0], [0, 0, 0], [0, 0, 0.5], [0, 0.5, 0.5]]\n");
    const std::vector<FacePoint> q =
        local_coordinates(*mesh, faces, cornered, "left", {bv::U, bv::V}, WallDistance(), false, summary);
    for (const FacePoint & x : q) {
        EXPECT_NEAR(x[bv::U], 1.0 - x[1] / H, tol(1e-12, 1e-6));
        EXPECT_NEAR(x[bv::V], x[2] / H, tol(1e-12, 1e-6));
        EXPECT_TRUE(std::isnan(x[bv::N]));
    }
    try {
        local_coordinates(*mesh, faces, entry, "left", {bv::R}, WallDistance(), false, summary);
        ADD_FAILURE() << "r on a four-sided patch is an error";
    } catch (const InputError & e) {
        EXPECT_NE(std::string(e.what()).find("which rings and disks have"), std::string::npos) << e.what();
    }
#endif
}

#if N_DIM == 3
TEST(BoundaryCoordinatesTest, RingsOfTheAnnulusMeshHaveTheirRadiusHeightAndAngle) {
    // The annulus 1 <= r <= 2 about z, extruded: its end `back` is a flat
    // ring (v logarithmic in r, theta the angle about the axis from +x), its
    // side `inner` a tube (v = z / Lz from the lower rim, u_m the arc length)
    auto mesh = std::make_shared<Mesh>();
    mesh->init(parse_toml("[mesh]\ntype = \"annulus\"\nNtheta = 48\nNr = 12\nNz = 6\nLz = 0.5\n"));
    const Input entry = parse_toml("name = \"zone\"\n");
    std::string summary;
    const std::vector<uint16_t> used{bv::U, bv::U_M, bv::V, bv::V_M, bv::R, bv::R_M, bv::THETA};
    auto angle = [](const FacePoint & x) {
        const double a = std::atan2(x[1], x[0]);
        return a < 0.0 ? a + 2.0 * M_PI : a;
    };
    const std::vector<FacePoint> back =
        local_coordinates(*mesh, zone_faces(*mesh, {"back"}), entry, "zone", used, WallDistance(), false, summary);
    ASSERT_FALSE(back.empty());
    EXPECT_NE(summary.find("ring, v = 0 on the rim around"), std::string::npos) << summary;
    for (const FacePoint & x : back) {
        const double r = std::hypot(x[0], x[1]);
        EXPECT_NEAR(x[bv::V], std::log(r) / std::log(2.0), 5e-3);
        EXPECT_EQ(x[bv::R], x[bv::V]);
        EXPECT_NEAR(x[bv::R_M], r - 1.0, 5e-3);
        EXPECT_NEAR(std::remainder(x[bv::THETA] - angle(x), 2.0 * M_PI), 0.0, tol(1e-9, 1e-4));
        EXPECT_EQ(x[bv::THETA], 2.0 * M_PI * x[bv::U]);
        EXPECT_NEAR(x[bv::U_M], x[bv::U] * 2.0 * M_PI * r, 0.02);
    }
    // A seam of its own: theta from -y
    const Input seamed = parse_toml("name = \"zone\"\nseam = [0.0, -1.5, 0.0]\n");
    double least = 1.0;
    for (const FacePoint & x : local_coordinates(*mesh, zone_faces(*mesh, {"back"}), seamed, "zone", {bv::THETA},
                                                 WallDistance(), false, summary)) {
        const double from_seam = std::remainder(x[bv::THETA] - (angle(x) + 0.5 * M_PI), 2.0 * M_PI);
        EXPECT_NEAR(from_seam, 0.0, 0.5 * 2.0 * M_PI / 48 + 1e-6);
        least = std::min(least, x[bv::THETA]);
    }
    EXPECT_EQ(least, 0.0);
    EXPECT_NE(summary.find("u = 0 at (0."), std::string::npos) << summary;
    const std::vector<FacePoint> inner =
        local_coordinates(*mesh, zone_faces(*mesh, {"inner"}), entry, "zone", used, WallDistance(), false, summary);
    ASSERT_FALSE(inner.empty());
    for (const FacePoint & x : inner) {
        EXPECT_NEAR(x[bv::V], x[2] / 0.5, tol(1e-10, 1e-5));
        EXPECT_NEAR(x[bv::V_M], x[2], tol(1e-10, 1e-5));
        EXPECT_NEAR(std::remainder(x[bv::THETA] - angle(x), 2.0 * M_PI), 0.0, tol(1e-9, 1e-4));
    }
}
#endif

TEST(BoundaryCoordinatesTest, WallDistanceIsExactBesideEdgesAndCorners) {
    // Bottom (y = 0) and left (x = 0) walls of the unit box: off their ends the
    // nearest point is an edge or a corner
    const auto mesh = box(3, 3, 1.0, 1.0);
    const WallDistance walls(*mesh, zone_faces(*mesh, {"bottom", "left"}), false);
    auto at = [](double x, double y, double z) {
        Point p{};
        p[0] = x;
        p[1] = y;
        if constexpr (N_DIM == 3) p[2] = z;
        return p;
    };
    EXPECT_NEAR(walls(at(0.3, 0.2, 0.5)), 0.2, tol(1e-15, 1e-6));
    EXPECT_NEAR(walls(at(0.6, 0.9, 0.5)), 0.6, tol(1e-15, 1e-6));
    EXPECT_NEAR(walls(at(0.5, -0.1, 0.5)), 0.1, tol(1e-15, 1e-6));
    EXPECT_NEAR(walls(at(-0.3, -0.4, 0.5)), 0.5, tol(1e-15, 1e-6));
    EXPECT_NEAR(walls(at(1.3, -0.4, 0.5)), 0.5, tol(1e-15, 1e-6));
#if N_DIM == 3
    EXPECT_NEAR(walls(at(-0.3, -0.4, -1.2)), 1.3, tol(1e-15, 1e-6));
    EXPECT_NEAR(walls(at(0.5, 0.25, 1.4)), std::hypot(0.25, 0.4), tol(1e-15, 1e-6));
#endif
}

#if N_DIM == 2
TEST(BoundaryCoordinatesTest, UNeedsOneOpenCurve) {
    const auto mesh = box(4, 4, 1.0, 1.0);
    const Input entry = parse_toml("name = \"zone\"\n");
    std::string summary;
    // Bottom then right: an L, from (0, 0) along x then up
    const std::vector<uint32_t> corner = zone_faces(*mesh, {"bottom", "right"});
    const std::vector<FacePoint> p =
        local_coordinates(*mesh, corner, entry, "zone", {bv::U, bv::U_M}, WallDistance(), false, summary);
    for (const FacePoint & x : p) {
        const double expected = x[0] < 1.0 ? x[0] : 1.0 + x[1];
        EXPECT_NEAR(x[bv::U_M], expected, tol(1e-15, 1e-6)) << x[0] << ", " << x[1];
        EXPECT_NEAR(x[bv::U], 0.5 * expected, tol(1e-15, 1e-6)) << x[0] << ", " << x[1];
    }
    // The top alone runs along +x, the right side along +y
    for (const FacePoint & x : local_coordinates(*mesh, zone_faces(*mesh, {"top"}), entry, "zone", {bv::U},
                                                 WallDistance(), false, summary)) {
        EXPECT_NEAR(x[bv::U], x[0], tol(1e-15, 1e-6));
    }
    for (const FacePoint & x : local_coordinates(*mesh, zone_faces(*mesh, {"right"}), entry, "zone", {bv::U},
                                                 WallDistance(), false, summary)) {
        EXPECT_NEAR(x[bv::U], x[1], tol(1e-15, 1e-6));
    }
    auto error = [&](const std::vector<std::string> & zones) {
        try {
            local_coordinates(*mesh, zone_faces(*mesh, zones), entry, "zone", {bv::U}, WallDistance(), false, summary);
        } catch (const InputError & e) {
            return std::string(e.what());
        }
        return std::string();
    };
    EXPECT_NE(error({"left", "right"}).find("more than one curve"), std::string::npos);
    EXPECT_NE(error({"left", "right", "bottom", "top"}).find("closed curve"), std::string::npos);
}
#endif

namespace {

std::string vec(const std::string & a, const std::string & b, const std::string & c = "0.0") {
    return "[" + a + ", " + b + (N_DIM == 3 ? ", " + c : std::string()) + "]";
}

std::string euler_state() {
    return "[initialize]\ntype = \"constant\"\np = 1.0\nT = 1.25\nu = " + vec("3.0", "0.0") +
           "\n[physics]\ntype = \"euler\"\ngamma = 1.4\np_ref = 1.0\nT_ref = 1.0\nrho_ref = 1.0\n";
}

std::string ns_state() {
    return "[initialize]\ntype = \"constant\"\np = 1.0\nT = 1.0\nu = " + vec("0.3", "0.0") +
           "\n[physics]\ntype = \"navier_stokes\"\nmu = 0.01\nPr = 0.72\ngamma = 1.4\np_ref = 1.0\nT_ref = 1.0\n"
           "rho_ref = 1.0\n";
}

std::string channel(const std::string & inlet, const std::string & run, const std::string & sides = "symmetry",
                    const std::string & state = euler_state()) {
    std::ostringstream s;
    s << "[run]\n" << run << mesh_block(N_DIM == 2 ? 24 : 12, N_DIM == 2 ? 8 : 4, 1.0, 0.5) << state
      << "[numerics]\nriemann_solver = \"HLLC\"\n[numerics.face_reconstruction]\ntype = \"MUSCL\"\n"
      << "[output]\ncheck_interval = 1000000\n"
      << "[[boundaries]]\nname = \"left\"\n" << inlet
      << "[[boundaries]]\nname = \"right\"\ntype = \"outlet_extrapolation\"\n";
    for (const char * side : {"bottom", "top", "back", "front"}) {
        if (N_DIM == 2 && (std::string(side) == "back" || std::string(side) == "front")) continue;
        s << "[[boundaries]]\nname = \"" << side << "\"\ntype = \"" << sides << "\"\n";
    }
    return s.str();
}

std::string u_inlet() { return N_DIM == 2 ? "u = [3.0, 0.0]\n" : "u = [3.0, 0.0, 0.0]\n"; }

void expect_error(const std::string & input, const std::string & message) {
    Solver solver;
    try {
        solver.init(parse_toml(input));
        ADD_FAILURE() << "no error, expected \"" << message << "\"";
    } catch (const std::exception & e) {
        EXPECT_NE(std::string(e.what()).find(message), std::string::npos) << "expected \"" << message << "\", got\n"
                                                                          << e.what();
    }
}

} // namespace

TEST(BoundaryCoordinatesTest, SupersonicInflowCarriesTheProfileOfTheLocalCoordinates) {
    // Uniform u and p with T a function of n_m and n (from the bottom), u
    // (along +y from 0 to 1) and u_m or v, v_m: the steady flow carries the
    // inlet's profile unchanged, rho = 1 / T
#if N_DIM == 2
    const std::string T = "T = \"1 + n_m + 0.5 * n + 0.2 * u - 0.2 * u_m\"\n";
    auto expected = [](double y, double) { return 1.0 + 2.2 * y; };
#else
    const std::string T = "T = \"1 + n_m + 0.5 * n + 0.2 * u - 0.1 * v + 0.2 * v_m\"\n";
    auto expected = [](double y, double) { return 1.0 + 2.4 * y; };
#endif
    Solver solver;
    solver.set_distributed(false);
    solver.init(parse_toml(channel("type = \"inlet_state\"\n" + u_inlet() + "p = 1.0\n" + T + "walls = [\"bottom\"]\n",
                                   "t_stop = 1.5\ncfl = 0.5\n")));
    solver.run();
    solver.copy_device_to_host();
    const Mesh & mesh = *solver.get_mesh();
    for (uint32_t c = 0; c < mesh.n_owned(); c++) {
        const double y = double(mesh.h_cell_coords(c, 1)), z = N_DIM == 3 ? double(mesh.h_cell_coords(c, N_DIM - 1)) : 0.0;
        EXPECT_NEAR(double(solver.h_conservatives(c, 0)), 1.0 / expected(y, z), tol(1e-10, 1e-5)) << y << ", " << z;
    }
}

TEST(BoundaryCoordinatesTest, ConditionsOfLocalCoordinatesAndTimeMatchSerialAndKeepTheStepGraph) {
    // Evaluated on the device at the stage times from the coordinates fixed at
    // setup: bit for bit on any rank count, with the step graph and without
    const std::string run = "t_stop = 0.05\ncfl = 0.5\n";
    expect_matches_serial_with_and_without_step_graphs(
        channel("type = \"inlet_state\"\n" + u_inlet() + "p = 1.0\nT = \"(1 + n + 0.1 * n_m + 0.1 * u + " +
                    (N_DIM == 2 ? "0.2 * u_m" : "0.2 * v + 0.01 * u_m + 0.1 * v_m") + ") * (1 + 0.1 * sin(60 * t))\"\n",
                run, "wall_adiabatic"));
    expect_matches_serial_with_and_without_step_graphs(
        channel("type = \"inlet_mass_flow\"\nT = 1.0\nmdot = 0.5\nprofile = \"(0.01 + n) * (1 + 0.5 * sin(60 * t))\"\n",
                run, "wall_adiabatic"));
}

#if N_DIM == 3
TEST(BoundaryCoordinatesTest, ConditionsOnARingMatchSerial) {
    // The patch is gathered from every rank and solved in one order, so the
    // harmonic v and its conjugate u are the same bits on any rank count
    expect_matches_serial(
        "[run]\nt_stop = 0.05\ncfl = 0.5\n"
        "[mesh]\ntype = \"annulus\"\nNtheta = 16\nNr = 4\nNz = 6\nLz = 1.0\n"
        "[initialize]\ntype = \"constant\"\np = 1.0\nT = 1.25\nu = [0.0, 0.0, 3.0]\n"
        "[physics]\ntype = \"euler\"\ngamma = 1.4\np_ref = 1.0\nT_ref = 1.0\nrho_ref = 1.0\n"
        "[numerics]\nriemann_solver = \"HLLC\"\n[numerics.face_reconstruction]\ntype = \"MUSCL\"\n"
        "[output]\ncheck_interval = 1000000\n"
        "[[boundaries]]\nname = \"back\"\ntype = \"inlet_state\"\nu = [0.0, 0.0, 3.0]\np = 1.0\n"
        "T = \"(1 + 0.2 * r + 0.1 * sin(theta) + 0.05 * v_m + 0.01 * u_m) * (1 + 0.1 * sin(60 * t))\"\n"
        "[[boundaries]]\nname = \"front\"\ntype = \"outlet_extrapolation\"\n"
        "[[boundaries]]\nname = \"inner\"\ntype = \"symmetry\"\n"
        "[[boundaries]]\nname = \"outer\"\ntype = \"symmetry\"\n");
}
#endif

TEST(BoundaryCoordinatesTest, CoordinatesThatCannotBeComputedAreInputErrors) {
    const std::string run = "n_steps = 1\ncfl = 0.5\n";
    expect_error(channel("type = \"inlet_state\"\n" + u_inlet() + "p = 1.0\nT = \"1 + n\"\n", run),
                 "no wall faces to measure it from");
    expect_error(channel("type = \"inlet_state\"\n" + u_inlet() + "p = 1.0\nT = \"1 + n\"\nwalls = [\"botom\"]\n", run),
                 "no boundary zone \"botom\"");
    expect_error(channel("type = \"inlet_state\"\n" + u_inlet() + "p = 1.0\nT = 1.0\nwalls = [\"botom\"]\n", run),
                 "no expression of the entry reads the wall distance n");
    expect_error(channel("type = \"inlet_state\"\n" + u_inlet() + "p = 1.0\nT = \"1 + n\"\nwalls = []\n", run),
                 "the list of zones is empty");
    expect_error(channel("type = \"inlet_state\"\n" + u_inlet() + "p = 1.0\nT = \"1 + n\"\nwalls = [\"bottom\"]\n"
                         "corners = []\n", run),
                 N_DIM == 2 ? "has two ends and no corners" : "so corners has no effect");
    // A wall reading n measures from itself by default
    expect_error(channel("type = \"inlet_state\"\n" + u_inlet() + "p = 1.0\nT = 1.0\n", run,
                         "wall_isothermal\"\nT = \"1 + n", ns_state()),
                 "is 0 on every face");
    expect_error(channel("type = \"inlet_mass_flow\"\nT = 1.0\nmdot = \"0.5 + n\"\n", run, "wall_adiabatic"),
                 "not on x, y, z or the local coordinates");
#if N_DIM == 2
    expect_error(channel("type = \"inlet_state\"\n" + u_inlet() + "p = 1.0\nT = \"1 + v\"\n", run),
                 "a 2D boundary is a curve");
#else
    expect_error(channel("type = \"inlet_state\"\n" + u_inlet() + "p = 1.0\nT = \"1 + theta\"\n", run),
                 "which rings and disks have");
#endif
}

namespace {

struct LocalCase {
    std::string name, inlet, state = euler_state();
};

std::string with(std::string text, const std::string & coordinate) {
    for (size_t at = text.find('Y'); at != std::string::npos; at = text.find('Y', at + coordinate.size())) {
        text.replace(at, 1, coordinate);
    }
    return text;
}

std::vector<LocalCase> local_cases() {
    const std::string mixture = "[initialize]\ntype = \"constant\"\np = 1.0e5\nT = 300.0\nu = " + vec("100.0", "0.0") +
                                "\nX = { N2 = 0.79, O2 = 0.21 }\n[physics]\ntype = \"euler\"\ngas = \"mixture\"\n"
                                "mechanism = \"" + source_dir() + "/mechanisms/h2o2.yaml\"\n";
    const std::string ns = ns_state();
    const std::string stress = N_DIM == 2 ? "[\"0.004 * (1 + Y)\", 0.002, -0.001]"
                                          : "[\"0.004 * (1 + Y)\", 0.002, 0.002, -0.001, 0.0, 0.0]";
    return {
        {"dirichlet", "type = \"dirichlet\"\nrho = \"1 + 0.2 * Y\"\nu = " + vec("\"3 + Y\"", "0.0") + "\np = 1.0\n"},
        {"direction", "type = \"inlet_state\"\nu = 3.0\ndirection = " + vec("1.0", "\"0.3 * Y\"") +
                          "\np = 1.0\nT = 1.25\n"},
        {"inlet_stagnation", "type = \"inlet_stagnation\"\np0 = \"8 + Y\"\nT0 = \"3 + Y\"\ndirection = " +
                                 vec("1.0", "\"0.2 * Y\"") + "\n"},
        {"nscbc", "type = \"inlet_state\"\nu = " + vec("\"0.3 + 0.1 * Y\"", "0.0") +
                      "\np = 1.0\nT = \"1 + 0.2 * Y\"\nnscbc.L = 1.0\n", ns},
        {"turbulence", "type = \"inlet_state\"\nu = " + vec("\"0.3 + 0.1 * Y\"", "0.0") +
                           "\np = 1.0\nT = 1.0\nnscbc.L = 1.0\nturbulence = { reynolds_stress = " + stress +
                           ", length_scale = 0.2 }\n", ns},
        {"mixture", "type = \"inlet_state\"\nu = " + vec("100.0", "0.0") +
                        "\np = 1.0e5\nT = \"300 + 100 * Y\"\nX = { N2 = \"0.79 - 0.1 * Y\", O2 = \"0.21 + 0.1 * Y\" }\n",
         mixture},
    };
}

class LocalCoordinateConditions : public ::testing::TestWithParam<LocalCase> {};

} // namespace

TEST_P(LocalCoordinateConditions, MatchTheSameConditionWrittenInY) {
    // n_m from the bottom is y: every path that evaluates an entry's expressions
    // on the device must see the faces' local coordinates (not NaN)
    const LocalCase & c = GetParam();
    const std::string run = "n_steps = 3\ncfl = 0.4\n";
    auto solve = [&](const std::string & inlet) {
        auto solver = std::make_unique<Solver>();
        solver->set_distributed(false);
        solver->init(parse_toml(channel(inlet, run, "symmetry", c.state)));
        solver->run();
        solver->copy_device_to_host();
        return solver;
    };
    const auto local = solve(with(c.inlet, "n_m") + "walls = [\"bottom\"]\n");
    const auto plain = solve(with(c.inlet, "y"));
    const Mesh & mesh = *local->get_mesh();
    for (uint32_t i = 0; i < mesh.n_owned(); i++) {
        for (uint32_t k = 0; k < uint32_t(local->h_conservatives.extent(1)); k++) {
            const double a = double(local->h_conservatives(i, k)), b = double(plain->h_conservatives(i, k));
            ASSERT_NEAR(a, b, tol(1e-12, 1e-5) * std::max(1.0, std::abs(b))) << c.name << ", cell " << i << ", " << k;
        }
    }
}

INSTANTIATE_TEST_SUITE_P(BoundaryCoordinates, LocalCoordinateConditions, ::testing::ValuesIn(local_cases()),
                         [](const auto & p) { return p.param.name; });

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
        "where = \"x < 1\"\nX = { O2 = \"0.21 * xi\", N2 = 0.79 }\nturbulence = { profile = \"bl.csv\" }\n");
    EXPECT_EQ(local_coordinates_used(entry), (std::vector<uint16_t>{bv::N, bv::XI}));
    EXPECT_TRUE(local_coordinates_used(parse_toml("name = \"inlet\"\nu = [\"x * t\", 0]\n")).empty());
}

TEST(BoundaryCoordinatesTest, LocalVariablesHideTheLocalCoordinates) {
    // Expressions written before the local coordinates keep their meaning in
    // 2D and 3D alike; x, y, z and t still cannot be hidden
    const Input v = parse_toml("v = \"var s := y; var r := 2 * x; var n := 1; s + r + n\"\n");
    const BoundaryValue value(v.at("v"), "inlet.T", dim::none);
    Point x{};
    x[0] = 0.25;
    x[1] = 0.5;
    EXPECT_DOUBLE_EQ(value.at(FacePoint(x), 0.0), 2.0);
    EXPECT_TRUE(local_coordinates_used(parse_toml("name = \"inlet\"\nT = \"var theta := atan2(y, x); theta\"\n")).empty());
    EXPECT_EQ(local_coordinates_used(parse_toml("name = \"inlet\"\nT = \"var q := n; q\"\n")),
              (std::vector<uint16_t>{bv::N}));
    EXPECT_THROW(BoundaryValue(parse_toml("v = \"var y := 1; y\"\n").at("v"), "inlet.T", dim::none),
                 std::exception);
}

TEST(BoundaryCoordinatesTest, PlanarInletBetweenWallsHasExactCoordinates) {
    // The left side of a box: tangent +y (and span +z) about its center,
    // n the distance to the nearer of the bottom and top walls
    const double H = 0.5;
    const auto mesh = box(4, 10, 1.0, H);
    const std::vector<uint32_t> faces = zone_faces(*mesh, {"left"});
    const WallDistance walls(*mesh, zone_faces(*mesh, {"bottom", "top"}), false);
    const Input entry = parse_toml("name = \"left\"\n");
    std::string summary;
    const std::vector<FacePoint> p =
        local_coordinates(*mesh, faces, entry, "left", all_local(), walls, false, summary);
    ASSERT_EQ(p.size(), faces.size());
    for (const FacePoint & x : p) {
        const double y = x[1];
        EXPECT_NEAR(x[bv::N], std::min(y, H - y), tol(1e-15, 1e-6)) << y;
        EXPECT_NEAR(x[bv::XI], y - 0.5 * H, tol(1e-15, 1e-6)) << y;
#if N_DIM == 2
        EXPECT_NEAR(x[bv::S], y, tol(1e-15, 1e-6)) << y;
#else
        const double z = x[2];
        EXPECT_NEAR(x[bv::ETA], z - 0.5 * H, tol(1e-15, 1e-6)) << z;
        EXPECT_NEAR(x[bv::R], std::hypot(y - 0.5 * H, z - 0.5 * H), tol(1e-15, 1e-6));
        EXPECT_NEAR(x[bv::THETA], std::atan2(z - 0.5 * H, y - 0.5 * H), tol(1e-14, 1e-6));
#endif
    }
    EXPECT_NE(summary.find("tangent (0, 1"), std::string::npos) << summary;
    EXPECT_NE(summary.find("n 0.025 to 0.225"), std::string::npos) << summary;

    // A frame of its own: from the bottom, along -y
    const Input framed = parse_toml(std::string("name = \"left\"\nframe = { origin = ") +
                                    (N_DIM == 2 ? "[0, 0]" : "[0, 0, 0]") + ", tangent = " +
                                    (N_DIM == 2 ? "[0, -1]" : "[0, -1, 0]") + " }\n");
    const std::vector<FacePoint> q =
        local_coordinates(*mesh, faces, framed, "left", {bv::XI}, WallDistance(), false, summary);
    for (const FacePoint & x : q) {
        EXPECT_NEAR(x[bv::XI], -x[1], tol(1e-15, 1e-6));
        EXPECT_TRUE(std::isnan(x[bv::N]));
    }
}

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
TEST(BoundaryCoordinatesTest, ArcLengthNeedsOneOpenCurve) {
    const auto mesh = box(4, 4, 1.0, 1.0);
    const Input entry = parse_toml("name = \"zone\"\n");
    std::string summary;
    // Bottom then right: an L, from (0, 0) along x then up
    const std::vector<uint32_t> corner = zone_faces(*mesh, {"bottom", "right"});
    const std::vector<FacePoint> p =
        local_coordinates(*mesh, corner, entry, "zone", {bv::S}, WallDistance(), false, summary);
    for (const FacePoint & x : p) {
        const double expected = x[0] < 1.0 ? x[0] : 1.0 + x[1];
        EXPECT_NEAR(x[bv::S], expected, tol(1e-15, 1e-6)) << x[0] << ", " << x[1];
    }
    // Opposite sides: their normals cancel, so the frame needs a normal
    const Input framed = parse_toml("name = \"zone\"\nframe.normal = [0.0, 1.0]\n");
    auto error = [&](const std::vector<std::string> & zones) {
        try {
            local_coordinates(*mesh, zone_faces(*mesh, zones), framed, "zone", {bv::S}, WallDistance(), false, summary);
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
    // Uniform u and p with T a function of n (from the bottom), xi (along
    // +y from the inlet's center) and s or eta: the steady flow carries the
    // inlet's profile unchanged, rho = 1 / T
#if N_DIM == 2
    const std::string T = "T = \"1 + n + 0.4 * (xi + 0.25) - 0.2 * s\"\n";
    auto expected = [](double y, double) { return 1.0 + 1.2 * y; };
#else
    const std::string T = "T = \"1 + n + 0.4 * (xi + 0.25) - 0.2 * (eta + 0.25)\"\n";
    auto expected = [](double y, double z) { return 1.0 + 1.4 * y - 0.2 * z; };
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
        channel("type = \"inlet_state\"\n" + u_inlet() + "p = 1.0\nT = \"(1 + n + 0.1 * xi + " +
                    (N_DIM == 2 ? "0.2 * s" : "0.2 * r + 0.01 * theta + 0.1 * eta") + ") * (1 + 0.1 * sin(60 * t))\"\n",
                run, "wall_adiabatic"));
    expect_matches_serial_with_and_without_step_graphs(
        channel("type = \"inlet_mass_flow\"\nT = 1.0\nmdot = 0.5\nprofile = \"(0.01 + n) * (1 + 0.5 * sin(60 * t))\"\n",
                run, "wall_adiabatic"));
}

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
                         "frame.tangent = " + vec("0.0", "1.0") + "\n", run),
                 "so frame has no effect");
    // A wall reading n measures from itself by default
    expect_error(channel("type = \"inlet_state\"\n" + u_inlet() + "p = 1.0\nT = 1.0\n", run,
                         "wall_isothermal\"\nT = \"1 + n", ns_state()),
                 "is 0 on every face");
    expect_error(channel("type = \"inlet_mass_flow\"\nT = 1.0\nmdot = \"0.5 + n\"\n", run, "wall_adiabatic"),
                 "not on x, y, z or the local coordinates");
#if N_DIM == 2
    expect_error(channel("type = \"inlet_state\"\n" + u_inlet() + "p = 1.0\nT = \"1 + eta\"\n", run), "eta");
#else
    expect_error(channel("type = \"inlet_state\"\n" + u_inlet() + "p = 1.0\nT = \"1 + s\"\n", run), "arc length");
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
    // n from the bottom is y: every path that evaluates an entry's expressions
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
    const auto local = solve(with(c.inlet, "n") + "walls = [\"bottom\"]\n");
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

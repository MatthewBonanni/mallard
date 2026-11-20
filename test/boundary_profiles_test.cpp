/**
 * @file boundary_profiles_test.cpp
 * @author Matthew Bonanni (mbonanni001@gmail.com)
 * @brief Tests of profiles from data at boundaries ([[boundaries.data]]).
 * @version 0.1
 * @date 2026-11-17
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 *
 */

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <iomanip>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "boundary_coordinates.h"
#include "boundary_profiles.h"
#include "gmsh_fixtures.h"
#include "mesh.h"
#include "mesh_block.h"
#include "mpi_compare.h"
#include "renumbered_mesh.h"
#include "solver.h"
#include "synthetic_inflow.h"
#include "tabulated.h"
#include "test_fixtures.h"

namespace {

std::string number(double x) {
    std::ostringstream s;
    s << std::setprecision(17) << x;
    return s.str();
}

std::string vec(const std::string & a, const std::string & b, const std::string & c = "0.0") {
    return "[" + a + ", " + b + (N_DIM == 3 ? ", " + c : std::string()) + "]";
}

/** @brief A cubic of the wall distance, which cubic interpolation reproduces exactly between its nodes. */
double profile(double n) { return 1.0 + 2.0 * n * n - 3.0 * n * n * n; }

/** @brief The profile at uneven wall distances from 0 to n_max, given in mm. */
std::string profile_csv(double n_max) {
    std::ostringstream s;
    s << "# wall distance [mm], temperature\nd, T\n";
    for (int k = 0; k <= 12; k++) {
        const double n = n_max * std::pow(k / 12.0, 1.3);
        s << number(1000.0 * n) << ", " << number(profile(n)) << "\n";
    }
    return s.str();
}

std::string euler_physics() {
    return "[physics]\ntype = \"euler\"\ngamma = 1.4\np_ref = 1.0\nT_ref = 1.0\nrho_ref = 1.0\n";
}

std::string sides(const std::string & type) {
    std::string s;
    for (const char * side : {"bottom", "top", "back", "front"}) {
        if (N_DIM == 2 && (std::string(side) == "back" || std::string(side) == "front")) continue;
        s += "[[boundaries]]\nname = \"" + std::string(side) + "\"\ntype = \"" + type + "\"\n";
    }
    return s;
}

} // namespace

TEST(BoundaryProfilesTest, RotatedInletReadsTheProfileAtItsWallDistance) {
    // A box turned by [mesh.transform]: supersonic inflow along the turned x
    // axis, T from a table of the wall distance n_m to the turned bottom wall
    // (in mm, cubic). The steady flow carries the inlet's profile along the
    // rows of cells, so rho = 1 / T(Y) with Y the cell's distance from the
    // bottom, held at the table's last value above its range.
    const double n_max = 0.3;
#if N_DIM == 2
    const std::array<uint32_t, N_DIM> n = {16, 8};
    const std::array<double, N_DIM> L = {1.0, 0.5};
    const std::string transform = "rotate = 30.0\ntranslate = [0.4, -0.2]\n";
#else
    const std::array<uint32_t, N_DIM> n = {8, 6, 2};
    const std::array<double, N_DIM> L = {1.0, 0.5, 0.25};
    const std::string transform = "rotate = {axis = [1.0, 2.0, 3.0], angle = 40.0}\n";
#endif
    const std::string mesh_file = write_temp("box.msh", renumbered_gmsh(jittered_box(MeshType::CARTESIAN, n, L, 0.0)).gmsh);
    const std::string csv = write_temp("profile.csv", profile_csv(n_max));
    const auto Q = mesh_transform(parse_toml("[mesh]\nfilename = \"x\"\n[mesh.transform]\n" + transform)).matrix;
    const std::string u = vec(number(3.0 * Q[0][0]), number(3.0 * Q[1][0]), N_DIM == 3 ? number(3.0 * Q[2][0]) : "");
    std::ostringstream input;
    input << "[run]\nt_stop = 1.5\ncfl = 0.5\n[mesh]\ntype = \"file\"\nfilename = \"" << mesh_file << "\"\n"
          << "[mesh.transform]\n" << transform
          << "[initialize]\ntype = \"constant\"\np = 1.0\nT = 1.25\nu = " << u << "\n" << euler_physics()
          << "[numerics]\nriemann_solver = \"HLLC\"\n[numerics.face_reconstruction]\ntype = \"MUSCL\"\n"
          << "[output]\ncheck_interval = 1000000\n"
          << "[[boundaries]]\nname = \"left\"\ntype = \"inlet_state\"\nu = 3.0\np = 1.0\nT = \"bl_T\"\n"
          << "walls = [\"bottom\"]\n"
          << "[[boundaries.data]]\nname = \"bl\"\nfile = \"" << csv << "\"\ncoordinates = [\"d\"]\nat = [\"n_m\"]\n"
          << "interpolation = \"cubic\"\nunits = {d = \"mm\"}\n"
          << "[[boundaries]]\nname = \"right\"\ntype = \"outlet_extrapolation\"\n" << sides("symmetry");
    Solver solver;
    solver.set_distributed(false);
    solver.init(parse_toml(input.str()));
    solver.run();
    solver.copy_device_to_host();
    const Mesh & mesh = *solver.get_mesh();
    const auto & t = mesh_transform(parse_toml("[mesh]\nfilename = \"x\"\n[mesh.transform]\n" + transform)).translation;
    for (uint32_t c = 0; c < mesh.n_owned(); c++) {
        double Y = 0.0;
        FOR_I_DIM Y += Q[i][1] * (double(mesh.h_cell_coords(c, i)) - t[i]);
        EXPECT_NEAR(double(solver.h_conservatives(c, 0)), 1.0 / profile(std::min(Y, n_max)), roundoff(1e-10))
            << "Y = " << Y;
    }
}

namespace {

std::string channel(const std::string & inlet, const std::string & run) {
    std::ostringstream s;
    s << "[run]\n" << run << "[mesh]\ntype = \"cartesian\"\nNx = " << (N_DIM == 2 ? 24 : 12) << "\nNy = "
      << (N_DIM == 2 ? 8 : 4) << "\nLx = 1.0\nLy = 0.5\n" << (N_DIM == 3 ? "Nz = 4\nLz = 0.5\n" : "")
      << "[initialize]\ntype = \"constant\"\np = 1.0\nT = 1.25\nu = " << vec("3.0", "0.0") << "\n" << euler_physics()
      << "[numerics]\nriemann_solver = \"HLLC\"\n[numerics.face_reconstruction]\ntype = \"MUSCL\"\n"
      << "[output]\ncheck_interval = 1000000\n"
      << "[[boundaries]]\nname = \"left\"\n" << inlet
      << "[[boundaries]]\nname = \"right\"\ntype = \"outlet_extrapolation\"\n" << sides("wall_adiabatic");
    return s.str();
}

std::string error_of(const std::string & input) {
    Solver solver;
    try {
        solver.init(parse_toml(input));
    } catch (const std::exception & e) {
        return e.what();
    }
    return "no error";
}

} // namespace

TEST(BoundaryProfilesTest, DataTimesExpressionsOfTimeMatchSerialAndKeepTheStepGraph) {
    // A table of n_m and a point cloud of u, stored with the faces at setup and
    // combined with t on the device: bit for bit on any rank count, with the
    // step graph and without
    const std::string table = write_temp_shared("bl.csv", profile_csv(0.25));
    std::ostringstream cloud;
    cloud << "u, du\n";
    for (int k = 0; k < 9; k++) cloud << number(0.125 * k) << ", " << number(0.1 * std::sin(k)) << "\n";
    const std::string points = write_temp_shared("jet.csv", cloud.str());
    const std::string inlet = "type = \"inlet_state\"\nu = " + vec("\"3 + jet_du * (1 + 0.5 * cos(40 * t))\"", "0.0") +
                              "\np = 1.0\nT = \"bl_T * (1 + 0.1 * sin(60 * t))\"\n"
                              "[[boundaries.data]]\nname = \"bl\"\nfile = \"" + table + "\"\nat = [\"n_m\"]\n"
                              "units = {d = \"mm\"}\n"
                              "[[boundaries.data]]\nname = \"jet\"\nfile = \"" + points + "\"\ntype = \"points\"\n"
                              "coordinates = [\"u\"]\nneighbors = 2\n";
    expect_matches_serial_with_and_without_step_graphs(channel(inlet, "t_stop = 0.05\ncfl = 0.5\n"));
}

TEST(BoundaryProfilesTest, ProfileOutsideItsTableIsReported) {
    // n_m = y on the left side of a box between walls at y = 0 and 0.5 (10
    // faces across): a table to n = 0.2 leaves the 6 faces above it at its
    // last value, and the run log says so
    auto mesh = std::make_shared<Mesh>();
    mesh->init(parse_toml(std::string("[mesh]\ntype = \"cartesian\"\nNx = 4\nNy = 10\nLx = 1.0\nLy = 0.5\n") +
                          (N_DIM == 3 ? "Nz = 2\nLz = 0.5\n" : "")));
    std::vector<uint32_t> faces, walls;
    for (const FaceZone & zone : mesh->h_face_zones()) {
        for (uint32_t i = 0; i < zone.n_faces(); i++) {
            if (zone.get_name() == "left") faces.push_back(zone.h_faces(i));
            if (zone.get_name() == "bottom") walls.push_back(zone.h_faces(i));
        }
    }
    const std::string csv = write_temp("bl.csv", profile_csv(0.2));
    const Input entry = parse_toml("name = \"left\"\nT = \"bl_T\"\n[[data]]\nname = \"bl\"\nfile = \"" + csv +
                                   "\"\nat = [\"n_m\"]\nunits = {d = \"mm\"}\n");
    const BoundaryProfiles data(entry, "left");
    ASSERT_EQ(data.coordinates_used(), std::vector<uint16_t>{boundary_variable::N_M});
    std::string summary;
    std::vector<FacePoint> points = local_coordinates(*mesh, faces, entry, "left", data.coordinates_used(),
                                                      WallDistance(*mesh, walls, false), false, summary);
    summary = data.evaluate(*mesh, faces, points, false);
    for (const FacePoint & p : points) {
        EXPECT_NEAR(p[boundary_variable::FIRST_DATA], profile(std::min(p[1], 0.2)), 0.01) << p[1];
    }
    const std::string mapped = "data \"bl\" (" + csv + ", table 13, linear): column d <- boundary n_m, ";
    EXPECT_EQ(summary.rfind(mapped, 0), 0u) << summary;
    const std::string outside = std::to_string(N_DIM == 2 ? 6 : 12) + " faces beyond the column's 0 to 0.2 m take";
    EXPECT_NE(summary.find(outside), std::string::npos) << summary;
    EXPECT_NE(summary.find("; values bl_T "), std::string::npos) << summary;
}

TEST(BoundaryProfilesTest, MisplacedDataAreInputErrors) {
    const std::string csv = write_temp_shared("bl.csv", profile_csv(0.25));
    const std::string run = "n_steps = 1\ncfl = 0.5\n";
    auto inlet = [&](const std::string & T, const std::string & data_keys) {
        return channel("type = \"inlet_state\"\nu = " + vec("3.0", "0.0") + "\np = 1.0\nT = " + T +
                           "\n[[boundaries.data]]\nname = \"bl\"\nfile = \"" + csv + "\"\n" + data_keys,
                       run);
    };
    EXPECT_NE(error_of(inlet("1.25", "at = [\"n_m\"]\n")).find("no expression of the boundary reads bl_T, the values"),
              std::string::npos);
    EXPECT_NE(error_of(inlet("\"bl_T\"", "at = [\"n_m + t\"]\n")).find("reads t for column d, but the data are read once"), std::string::npos);
    EXPECT_NE(error_of(inlet("\"bl_T\"", "")).find("the table's coordinate column d is not named like a boundary coordinate"),
              std::string::npos);
    // Column names are the table's: a coordinate column and a value column named like coordinates
    std::ostringstream named;
    named << "n_m, u\n";
    for (int k = 0; k <= 4; k++) named << number(0.0625 * k) << ", " << number(0.25 * k) << "\n";
    const std::string same = write_temp_shared("same.csv", named.str());
    EXPECT_EQ(error_of(channel("type = \"inlet_state\"\nu = " + vec("\"3 + bl_u\"", "0.0") +
                                   "\np = 1.0\nT = 1.25\n[[boundaries.data]]\nname = \"bl\"\nfile = \"" + same + "\"\n",
                               run)),
              "no error");
    EXPECT_NE(error_of(channel("type = \"inlet_state\"\nu = " + vec("\"3 + u_m\"", "0.0") +
                                   "\np = 1.0\nT = 1.25\n[[boundaries.data]]\nname = \"u\"\nfile = \"" +
                                   write_temp_shared("m.csv", "n_m, m\n0, 0\n1, 1\n") + "\"\n",
                               run))
                  .find("makes the variable u_m (name u, column m), which is the boundary coordinate u_m"),
              std::string::npos);
}

namespace {

/** @brief The faces of the left side of a box between walls at y = 0 and 0.5, and those of its wall named wall. */
std::shared_ptr<Mesh> left_side(const std::string & wall, std::vector<uint32_t> & faces, std::vector<uint32_t> & walls) {
    auto mesh = std::make_shared<Mesh>();
    mesh->init(parse_toml(std::string("[mesh]\ntype = \"cartesian\"\nNx = 4\nNy = 10\nLx = 1.0\nLy = 0.5\n") +
                          (N_DIM == 3 ? "Nz = 2\nLz = 0.5\n" : "")));
    for (const FaceZone & zone : mesh->h_face_zones()) {
        for (uint32_t i = 0; i < zone.n_faces(); i++) {
            if (zone.get_name() == "left") faces.push_back(zone.h_faces(i));
            if (zone.get_name() == wall) walls.push_back(zone.h_faces(i));
        }
    }
    return mesh;
}

/** @brief The points of the faces with the data of entry, as the solver sets them up. */
std::vector<FacePoint> profile_points(const Mesh & mesh, const std::vector<uint32_t> & faces,
                                      const std::vector<uint32_t> & walls, const Input & entry, std::string & summary) {
    const BoundaryProfiles data(entry, "left");
    std::vector<FacePoint> points = local_coordinates(mesh, faces, entry, "left", data.coordinates_used(),
                                                      WallDistance(mesh, walls, false), false, summary);
    summary = data.evaluate(mesh, faces, points, false);
    return points;
}

} // namespace

TEST(BoundaryProfilesTest, TurbulenceProfileReadsAPrecursorAtTheWallDistance) {
    // Precursor statistics, linear in the distance n_m from the top wall (y =
    // 0.5), with the shear stress as COV_U_Y_U_X: the inlet's mean velocity and
    // Reynolds stress at each face are the file's at n_m = 0.5 - y, whether
    // turbulence.profile names a data entry or the file itself
    std::vector<uint32_t> faces, walls;
    const auto mesh = left_side("top", faces, walls);
    const auto U = [](double n) { return std::array<double, 3>{1.0 + 2.0 * n, 0.1 * n, 0.0}; };
    const auto R = [](double n) { return std::array<double, 4>{0.01 + 0.02 * n, 0.005, 0.004, -0.003 * n}; };
    std::ostringstream csv;
    csv << "n_m,MEAN_U_X,MEAN_U_Y," << (N_DIM == 3 ? "MEAN_U_Z," : "") << "COV_U_X_U_X,COV_U_Y_U_Y,"
        << (N_DIM == 3 ? "COV_U_Z_U_Z," : "") << "COV_U_Y_U_X\n";
    for (int k = 0; k <= 5; k++) {
        const double n = 0.1 * k;
        csv << number(n) << "," << number(U(n)[0]) << "," << number(U(n)[1]) << ","
            << (N_DIM == 3 ? number(U(n)[2]) + "," : "") << number(R(n)[0]) << "," << number(R(n)[1]) << ","
            << (N_DIM == 3 ? number(R(n)[2]) + "," : "") << number(R(n)[3]) << "\n";
    }
    const std::string file = write_temp("precursor.csv", csv.str());
    const std::string turbulence = "name = \"left\"\n[turbulence]\nlength_scale = 0.1\nprofile = ";
    for (const std::string & entry_text : {turbulence + "\"pre\"\n[[data]]\nname = \"pre\"\nfile = \"" + file + "\"\n",
                                           turbulence + "\"" + file + "\"\n"}) {
        const Input entry = parse_toml(entry_text);
        std::string summary;
        const std::vector<FacePoint> points = profile_points(*mesh, faces, walls, entry, summary);
        const InletProfile profile(entry, "left");
        for (const FacePoint & p : points) {
            const double n = 0.5 - p[1];
            const std::array<double, N_DIM> u = profile.velocity(p, Point{});
            const std::array<double, N_STRESS> s = profile.stress_at(p);
            FOR_I_DIM EXPECT_NEAR(u[i], U(n)[i], 1e-14) << entry_text;
#if N_DIM == 2
            const std::array<double, N_STRESS> expected = {R(n)[0], R(n)[1], R(n)[3]};
#else
            const std::array<double, N_STRESS> expected = {R(n)[0], R(n)[1], R(n)[2], R(n)[3], 0.0, 0.0};
#endif
            for (uint8_t k = 0; k < N_STRESS; k++) EXPECT_NEAR(s[k], expected[k], 1e-15) << entry_text;
        }
    }
}

TEST(BoundaryProfilesTest, MetadataGivesTheEntrysDefaults) {
    // The file's mallard-profile comment gives the coordinate column, where it
    // is read and its unit; columns leaves out a column nothing reads
    std::vector<uint32_t> faces, walls;
    const auto mesh = left_side("bottom", faces, walls);
    std::string body = profile_csv(0.2);
    body = body.substr(body.find('\n') + 1);
    std::ostringstream csv;
    csv << "# mallard-profile {\"coordinates\": [\"d\"], \"at\": [\"n_m\"], \"units\": {\"d\": \"mm\"}, "
           "\"source\": \"a cubic\", \"transforms\": []}\n"
        << "d, T, junk\n";
    std::istringstream rows(body.substr(body.find('\n') + 1));
    for (std::string line; std::getline(rows, line);) csv << line << ", 0\n";
    const std::string file = write_temp("described.csv", csv.str());
    const std::string entry = "name = \"left\"\nT = \"bl_T\"\n[[data]]\nname = \"bl\"\nfile = \"" + file +
                              "\"\ncolumns = [\"T\"]\n";
    std::string summary;
    for (const FacePoint & p : profile_points(*mesh, faces, walls, parse_toml(entry), summary)) {
        EXPECT_NEAR(p[boundary_variable::FIRST_DATA], profile(std::min(p[1], 0.2)), 0.01) << p[1];
    }
    EXPECT_EQ(summary.rfind("data \"bl\" (" + file + ", table 13, linear): column d <- boundary n_m, ", 0), 0u)
        << summary;
    // Beyond the table's range, outside = "error" stops instead of holding its end values
    try {
        profile_points(*mesh, faces, walls, parse_toml(entry + "outside = \"error\"\n"), summary);
        ADD_FAILURE() << "no error";
    } catch (const InputError & e) {
        const std::string beyond = "beyond the column's 0 to 0.2 m at " + std::to_string(N_DIM == 2 ? 6 : 12) + " faces";
        EXPECT_NE(std::string(e.what()).find(beyond), std::string::npos) << e.what();
    }
}

TEST(BoundaryProfilesTest, ScaleRescalesEachColumnByItsDimension) {
    // Length 0.01 m, velocity 2 m/s and density 1 in the metadata, velocity 4 m/s in the
    // entry, rescaled to 0.03 m, 5 m/s and twice the density: lengths x3,
    // velocities x1.25, stresses x1.25^2, pressures (rho U^2) x2 x1.25^2, and
    // temperatures (in degC, converted to K) unchanged
    const std::string file = write_temp(
        "scaled.csv",
        "# mallard-profile {\"units\": {\"d\": \"mm\", \"U\": \"m/s\", \"R\": \"m^2/s^2\", \"T\": \"degC\", "
        "\"p\": \"bar\"}, \"reference\": {\"length\": 0.01, \"velocity\": \"7.2 km/h\", \"density\": 1.0}}\n"
        "d, U, R, T, p\n0, 1, 0.5, 20, 1\n4, 3, 0.25, 30, 2\n10, 4, 0.125, 40, 4\n");
    tabulated::Data data;
    const tabulated::Entry entry = tabulated::read_entry(
        parse_toml("name = \"bl\"\nfile = \"" + file + "\"\nscale = {length = 0.03, velocity = 5.0, density = 2.0}\n"
                   "reference = {velocity = 4.0}\n"),
        "bl", "docs", data);
    EXPECT_NE(entry.kind.find("scaled: length x3, velocity x1.25, density x2"), std::string::npos) << entry.kind;
    const auto host = data.host();
    const double d[3] = {0.0, 4.0, 10.0}, U[3] = {1, 3, 4}, R[3] = {0.5, 0.25, 0.125}, T[3] = {20, 30, 40},
                 p[3] = {1, 2, 4};
    for (int k = 0; k < 3; k++) {
        const double q = 3.0 * 1e-3 * d[k];
        const double expected[4] = {1.25 * U[k], 1.5625 * R[k], 273.15 + T[k], 2.0 * 1.5625 * 1e5 * p[k]};
        for (uint32_t f = 0; f < 4; f++) {
            EXPECT_NEAR(host(entry.set, f, &q), expected[f], 1e-12 * std::abs(expected[f])) << f << " at " << d[k];
        }
    }
}

TEST(BoundaryProfilesTest, MisdescribedProfilesAreInputErrors) {
    const std::string csv = write_temp_shared("plain.csv", profile_csv(0.25));
    const std::string run = "n_steps = 1\ncfl = 0.5\n";
    auto inlet = [&](const std::string & file, const std::string & data_keys) {
        return channel("type = \"inlet_state\"\nu = " + vec("3.0", "0.0") + "\np = 1.0\nT = \"bl_T\"" +
                           "\n[[boundaries.data]]\nname = \"bl\"\nfile = \"" + file + "\"\nat = [\"n_m\"]\n" + data_keys,
                       run);
    };
    EXPECT_NE(error_of(inlet(csv, "units = {d = \"mm\"}\nscale = {length = 0.1}\n"))
                  .find("neither the entry nor the mallard-profile comment of " + csv + " gives its reference length"),
              std::string::npos);
    EXPECT_NE(error_of(inlet(csv, "scale = {length = 0.1}\nreference = {length = 0.25}\n"))
                  .find("needs the unit of every column it rescales, and column d has none"),
              std::string::npos);
    // A value that is not a finite number is an error in a column the entry reads, not in one it leaves out
    const std::string gap = write_temp_shared("gap.csv", "d, T, junk\n0, 1, nan\n100, nan, 0\n250, 2, 0\n");
    EXPECT_NE(error_of(inlet(gap, "units = {d = \"mm\"}\ncolumns = [\"T\"]\n"))
                  .find(gap + ":3: column T is \"nan\", not a finite number"),
              std::string::npos);
    EXPECT_EQ(error_of(inlet(write_temp_shared("gap2.csv", "d, T, junk\n0, 1, nan\n250, 2, 0\n"),
                             "units = {d = \"mm\"}\ncolumns = [\"T\"]\n")),
              "no error");
    const std::string typo = write_temp_shared("typo.csv", "# mallard-profile {\"unit\": {\"d\": \"mm\"}}\n" +
                                                               profile_csv(0.25));
    EXPECT_NE(error_of(inlet(typo, "")).find("its mallard-profile comment has an unknown key \"unit\""),
              std::string::npos);
}

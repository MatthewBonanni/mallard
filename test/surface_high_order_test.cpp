/**
 * @file surface_high_order_test.cpp
 * @brief Snapshots of boundary zones, cut planes and surfaces as VTK Lagrange cells of the reconstruction polynomials.
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 */

#include <gtest/gtest.h>
#include <Kokkos_Core.hpp>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <tuple>
#include <string>
#include <vector>

#include "comm.h"
#include "gmsh_fixtures.h"
#include "lagrange.h"
#include "output_reader.h"
#include "scratch.h"
#include "solver.h"
#include "test_fixtures.h"
#include "test_utils.h"

#ifdef Mallard_HAS_HDF5
#include "hdf5_output.h"
#endif

#if defined(Mallard_HAS_HDF5)

namespace {

/** @brief A Lagrange snapshot as read back: its cells' types and node ranges, nodes and one point field. */
struct Snapshot {
    std::vector<uint8_t> types;
    std::vector<int64_t> offsets;
    std::vector<double> points;
    std::vector<double> values;
};

std::string vtkhdf_type(const std::string & file) {
    const hid_t f = H5Fopen(file.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
    const hid_t top = H5Gopen2(f, "VTKHDF", H5P_DEFAULT);
    const hid_t a = H5Aopen(top, "Type", H5P_DEFAULT);
    const hid_t t = H5Aget_type(a);
    std::string out(H5Tget_size(t), '\0');
    H5Aread(a, t, out.data());
    H5Tclose(t);
    H5Aclose(a);
    H5Gclose(top);
    H5Fclose(f);
    return out.substr(0, out.find('\0'));
}

Snapshot read(const std::string & file, const std::string & field) {
    return {h5_values<uint8_t>(file, "VTKHDF/Types"), h5_values<int64_t>(file, "VTKHDF/Offsets"),
            h5_values<double>(file, "VTKHDF/Points"), h5_values<double>(file, "VTKHDF/PointData/" + field)};
}

} // namespace

#endif

#if Mallard_DIM == 2 && defined(Mallard_HAS_HDF5)

namespace {

// rho quadratic and u linear: rho, rho u and so u are polynomials of TENO4's degree 3
double rho_of(double x, double y) { return 1.0 + 0.3 * x - 0.2 * y + 0.4 * x * y + 0.5 * y * y; }
double u_of(double, double y) { return 0.2 + 0.1 * y; }

const std::string REGIONS = "[regions]\ncut = {type = \"cut_plane\", point = [0.5, 0.5], normal = [1.0, 2.0]}\n"
                            "ring = {type = \"circle\", center = [0.5, 0.52], radius = 0.17}\n";

std::string flow_case(const std::string & mesh_file, const std::string & reconstruction, const std::string & writers,
                      const std::string & regions = "") {
    std::string boundaries;
    for (const char * name : {"left", "right", "top", "bottom"}) {
        boundaries += "[[boundaries]]\nname = \"" + std::string(name) + "\"\ntype = \"outlet_extrapolation\"\n";
    }
    return "[run]\nn_steps = 1\ncfl = 0.2\n"
           "[mesh]\ntype = \"file\"\nfilename = \"" + mesh_file + "\"\n"
           "[initialize]\ntype = \"analytical\"\nrho = \"1.0 + 0.3 * x - 0.2 * y + 0.4 * x * y + 0.5 * y * y\"\n"
           "u = [\"0.2 + 0.1 * y\", \"0.1 - 0.05 * x\"]\np = \"1.0\"\n" + boundaries +
           "[physics]\ntype = \"euler\"\ngamma = 1.4\np_ref = 1.0\nT_ref = 1.0\nrho_ref = 1.0\n"
           "[numerics.face_reconstruction]\n" + reconstruction + regions + writers;
}

const std::string TENO4 = "type = \"TENO\"\ntroubled_threshold = 1.0\norder = 4\n";

bool interior(const std::vector<double> & points, size_t first, size_t n) {
    for (size_t k = first; k < first + n; k++) {
        const double x = points[3 * k], y = points[3 * k + 1];
        if (std::min({x, y, 1.0 - x, 1.0 - y}) < 0.26) return false;
    }
    return true;
}

} // namespace

TEST(SurfaceHighOrderTest, SliceAndCurveNodesHoldThePolynomialsAndTheirDerivatives) {
    // TENO4 reproduces rho, rho u and so u: every node of an element clear of the boundaries holds them, and
    // the gradient fields their derivatives, while one value per element is off by the variation across it
    if (comm::size() > 1) GTEST_SKIP() << "serial";
    const std::string mesh_file = write_temp("surface_high_order.msh", jittered_mixed_mesh(12));
    const std::string dir = (scratch_dir() / "mallard_surface_high_order").string();
    std::filesystem::remove_all(dir);
    const std::string fields = "fields = [\"RHO\", \"U\", \"VORTICITY\", \"Q_CRITERION\", \"SCHLIEREN\"]\n";
    Solver solver;
    solver.set_distributed(false);
    solver.init(parse_toml(flow_case(mesh_file, TENO4,
                                     "[[write_data]]\nprefix = \"" + dir + "/hi\"\ninterval = 1\nregions = [\"cut\", \"ring\"]\n" +
                                         fields + "[[write_data]]\nprefix = \"" + dir +
                                         "/lo\"\ninterval = 1\nregions = [\"cut\", \"ring\"]\nhigh_order = false\n" + fields,
                                     REGIONS)));
    solver.run();
    // grad u = [[0, 0.1], [-0.05, 0]]: vorticity -0.15, Q = -a_01 a_10
    const double vorticity = -0.15, q = 0.005;
    for (const std::string region : {"cut", "ring"}) {
        const std::string hi = dir + "/hi_" + region + "_000000.vtkhdf";
        const Snapshot rho = read(hi, "RHO");
        const auto u = h5_values<double>(hi, "VTKHDF/PointData/U_X");
        const auto w = h5_values<double>(hi, "VTKHDF/PointData/VORTICITY");
        const auto qc = h5_values<double>(hi, "VTKHDF/PointData/Q_CRITERION");
        const auto schlieren = h5_values<double>(hi, "VTKHDF/PointData/SCHLIEREN");
        const auto lo_rho = h5_values<double>(dir + "/lo_" + region + "_000000.vtkhdf", "VTKHDF/CellData/RHO");
        ASSERT_EQ(lo_rho.size(), rho.types.size()) << region;
        size_t n_checked = 0;
        double lo_error = 0.0;
        for (size_t c = 0; c < rho.types.size(); c++) {
            ASSERT_EQ(rho.types[c], 68) << region;
            const size_t first = size_t(rho.offsets[c]), n = size_t(rho.offsets[c + 1]) - first;
            ASSERT_EQ(n, 4u) << region;
            if (!interior(rho.points, first, n)) continue;
            n_checked++;
            for (size_t k = first; k < first + n; k++) {
                const double x = rho.points[3 * k], y = rho.points[3 * k + 1];
                EXPECT_NEAR(rho.values[k], rho_of(x, y), 1e-12) << region << " cell " << c;
                EXPECT_NEAR(u[k], u_of(x, y), 1e-12) << region;
                EXPECT_NEAR(w[k], vorticity, 1e-11) << region;
                EXPECT_NEAR(qc[k], q, 1e-11) << region;
                EXPECT_NEAR(schlieren[k], std::hypot(0.3 + 0.4 * y, -0.2 + 0.4 * x + y), 1e-11) << region;
                lo_error = std::max(lo_error, std::fabs(lo_rho[c] - rho_of(x, y)));
            }
        }
        EXPECT_GT(n_checked, 8u) << region;
        EXPECT_GT(lo_error, 1e-3) << region;
    }
}

TEST(SurfaceHighOrderTest, AWallZoneIsTheFacesAsCurvesOfTheWallSideCells) {
    // Each face of the zone a Lagrange curve of degree 3 from its first end to its second, its nodes on the
    // face; the two curves meeting at a node each take their own cell's polynomial, so they differ there
    if (comm::size() > 1) GTEST_SKIP() << "serial";
    const std::string dir = (scratch_dir() / "mallard_surface_high_order_wall").string();
    std::filesystem::remove_all(dir);
    const std::string mesh_file = write_temp("surface_high_order_wall.msh", jittered_mixed_mesh(10));
    Solver solver;
    solver.set_distributed(false);
    solver.init(parse_toml(flow_case(mesh_file, TENO4,
                                     "[[write_data]]\nprefix = \"" + dir + "/w\"\ninterval = 1\nregion = \"bottom\"\n"
                                     "fields = [\"RHO\", \"P\"]\n")));
    solver.run();
    const Snapshot s = read(dir + "/w_000000.vtkhdf", "RHO");
    ASSERT_EQ(s.types.size(), 10u);
    double jump = 0.0, x_last = -1.0;
    for (size_t c = 0; c < s.types.size(); c++) {
        EXPECT_EQ(s.types[c], 68);
        const size_t first = size_t(s.offsets[c]);
        ASSERT_EQ(size_t(s.offsets[c + 1]) - first, 4u);
        const double x0 = s.points[3 * first], x1 = s.points[3 * (first + 1)];
        for (size_t k = first; k < first + 4; k++) EXPECT_EQ(s.points[3 * k + 1], 0.0);
        // Ends, then the interior nodes at 1/3 and 2/3 of the face
        EXPECT_NEAR(s.points[3 * (first + 2)], x0 + (x1 - x0) / 3.0, 1e-15);
        EXPECT_NEAR(s.points[3 * (first + 3)], x0 + 2.0 * (x1 - x0) / 3.0, 1e-15);
        if (x0 == x_last) jump = std::max(jump, std::fabs(s.values[first] - s.values[first - 3]));
        x_last = x1;
    }
    EXPECT_GT(jump, 0.0);
}

TEST(SurfaceHighOrderTest, ElementsAreLagrangeCellsByDefaultWithPolynomials) {
    if (comm::size() > 1) GTEST_SKIP() << "serial";
    const std::string mesh_file = write_temp("surface_high_order_keys.msh", jittered_mixed_mesh(4));
    const std::string dir = (scratch_dir() / "mallard_surface_high_order_keys").string();
    std::filesystem::remove_all(dir);
    const std::string writer = "[[write_data]]\nprefix = \"" + dir + "/b\"\ninterval = 1\nregion = \"bottom\"\n";
    const std::string muscl = "type = \"MUSCL\"\n";
    // MUSCL has no polynomial beyond the average: one value per element, and the key is an error
    for (const auto & [reconstruction, key, type] :
         {std::tuple{TENO4, std::string(""), std::string("UnstructuredGrid")},
          {muscl, std::string(""), std::string("PolyData")}, {TENO4, std::string("high_order = false\n"), std::string("PolyData")}}) {
        Solver solver;
        solver.set_distributed(false);
        solver.init(parse_toml(flow_case(mesh_file, reconstruction, writer + key + "fields = [\"P\"]\n")));
        solver.run();
        const output::Contents c = output::open(dir + "/b_000000.vtkhdf");
        ASSERT_TRUE(c.field("P")) << key;
        EXPECT_EQ(vtkhdf_type(dir + "/b_000000.vtkhdf"), type) << key;
        if (type != "PolyData") {
            EXPECT_EQ(h5_values<int64_t>(dir + "/b_000000.vtkhdf", "VTKHDF/Offsets")[1], 4) << "degree 3";
        }
        std::filesystem::remove_all(dir);
    }
    for (const auto & [reconstruction, key, message] :
         {std::tuple{muscl, std::string("high_order = true\n"), std::string("TENO")}}) {
        Solver solver;
        solver.set_distributed(false);
        try {
            solver.init(parse_toml(flow_case(mesh_file, reconstruction, writer + key + "fields = [\"P\"]\n")));
            ADD_FAILURE() << "accepted " << key;
        } catch (const std::exception & e) {
            EXPECT_NE(std::string(e.what()).find(message), std::string::npos) << e.what();
        }
    }
}

TEST(SurfaceHighOrderTest, VtuSnapshotsAreUnstructuredGridsOfLagrangeCells) {
    if (comm::size() > 1) GTEST_SKIP() << "serial";
    const std::string mesh_file = write_temp("surface_high_order_vtu.msh", jittered_mixed_mesh(6));
    const std::string dir = (scratch_dir() / "mallard_surface_high_order_vtu").string();
    std::filesystem::remove_all(dir);
    Solver solver;
    solver.set_distributed(false);
    solver.init(parse_toml(flow_case(mesh_file, TENO4,
                                     "[[write_data]]\nprefix = \"" + dir + "/v\"\nformat = \"vtu\"\ninterval = 1\n"
                                     "regions = [\"cut\", \"bottom\"]\nfields = [\"RHO\", \"U\"]\n",
                                     "[regions]\ncut = {type = \"cut_plane\", point = [0.5, 0.5], normal = [1.0, 2.0]}\n")));
    solver.run();
    for (const std::string region : {"cut", "bottom"}) {
        const std::string file = dir + "/v_" + region + "_000001.vtu";
        ASSERT_TRUE(std::filesystem::exists(file)) << file;
        const output::Contents c = output::open(file);
        ASSERT_TRUE(c.field("RHO")) << region;
        EXPECT_EQ(c.field("RHO")->location, "point") << region;
        std::ifstream pvd(dir + "/v_" + region + ".pvd");
        const std::string series((std::istreambuf_iterator<char>(pvd)), std::istreambuf_iterator<char>());
        EXPECT_NE(series.find("v_" + region + "_000001.vtu"), std::string::npos);
    }
}

#endif // Mallard_DIM == 2 && Mallard_HAS_HDF5

#if Mallard_DIM == 3 && defined(Mallard_HAS_HDF5)

TEST(SurfaceHighOrderTest, ACutPlaneOfEveryCellShapeHoldsThePolynomial) {
    // An oblique plane cuts hexahedra, tetrahedra and prisms into polygons of 3 to 6 vertices: quadrilaterals,
    // and triangles from the first vertex of the others, all exact for a quadratic density at TENO3's degree 2
    if (comm::size() > 1) GTEST_SKIP() << "serial";
    auto rho = [](double x, double y, double z) {
        return 1.0 + 0.3 * x - 0.2 * y + 0.1 * z + 0.4 * x * y + 0.5 * z * z - 0.3 * y * z;
    };
    for (const std::string type : {"cartesian", "cartesian_tet", "cartesian_prism"}) {
        const std::string dir = (scratch_dir() / ("mallard_surface_high_order_" + type)).string();
        std::filesystem::remove_all(dir);
        std::string boundaries;
        for (const char * name : {"left", "right", "top", "bottom", "front", "back"}) {
            boundaries += "[[boundaries]]\nname = \"" + std::string(name) + "\"\ntype = \"outlet_extrapolation\"\n";
        }
        Solver solver;
        solver.set_distributed(false);
        solver.init(parse_toml(
            "[run]\nn_steps = 1\ncfl = 0.2\n[mesh]\ntype = \"" + type + "\"\nNx = 8\nNy = 8\nNz = 8\n"
            "Lx = 1.0\nLy = 1.0\nLz = 1.0\n[initialize]\ntype = \"analytical\"\n"
            "rho = \"1.0 + 0.3 * x - 0.2 * y + 0.1 * z + 0.4 * x * y + 0.5 * z * z - 0.3 * y * z\"\n"
            "u = [\"0.2\", \"0.1\", \"0.0\"]\np = \"1.0\"\n" + boundaries +
            "[physics]\ntype = \"euler\"\ngamma = 1.4\np_ref = 1.0\nT_ref = 1.0\nrho_ref = 1.0\n"
            "[numerics.face_reconstruction]\ntype = \"TENO\"\ntroubled_threshold = 1.0\norder = 3\n"
            "[[write_data]]\nprefix = \"" + dir + "/c\"\ninterval = 1\n"
            "region = {type = \"cut_plane\", point = [0.5, 0.5, 0.5], normal = [1.0, 0.7, 0.4]}\n"
            "fields = [\"RHO\"]\n"));
        solver.run();
        const Snapshot s = read(dir + "/c_000000.vtkhdf", "RHO");
        size_t n_checked = 0, n_triangles = 0;
        for (size_t c = 0; c < s.types.size(); c++) {
            ASSERT_TRUE(s.types[c] == 69 || s.types[c] == 70) << type;
            n_triangles += s.types[c] == 69;
            const size_t first = size_t(s.offsets[c]), n = size_t(s.offsets[c + 1]) - first;
            EXPECT_EQ(n, s.types[c] == 69 ? 6u : 9u);
            bool clear = true;
            for (size_t k = first; k < first + n; k++) {
                const double * x = &s.points[3 * k];
                EXPECT_NEAR((x[0] - 0.5) * 1.0 + (x[1] - 0.5) * 0.7 + (x[2] - 0.5) * 0.4, 0.0, 1e-14);
                clear = clear && std::min({x[0], x[1], x[2], 1.0 - x[0], 1.0 - x[1], 1.0 - x[2]}) > 0.26;
            }
            if (!clear) continue;
            n_checked++;
            for (size_t k = first; k < first + n; k++) {
                EXPECT_NEAR(s.values[k], rho(s.points[3 * k], s.points[3 * k + 1], s.points[3 * k + 2]), 1e-11) << type;
            }
        }
        EXPECT_GT(n_checked, 10u) << type;
        EXPECT_GT(n_triangles, 0u) << type;
    }
}

#endif // Mallard_DIM == 3 && Mallard_HAS_HDF5

#ifdef Mallard_HAS_HDF5

TEST(SurfaceHighOrderTest, LagrangeSnapshotsDoNotDependOnTheRanks) {
    // A wall zone, a cut plane and a surface of [regions], their nodes in global order whatever the partition
#if Mallard_DIM == 2
    const std::string mesh = "[mesh]\ntype = \"cartesian\"\nNx = 10\nNy = 8\nLx = 1.0\nLy = 0.8\n";
    const std::string regions = "[regions]\ncut = {type = \"cut_plane\", point = [0.5, 0.4], normal = [1.0, 2.0]}\n"
                                "ring = {type = \"circle\", center = [0.5, 0.4], radius = 0.25}\n";
    std::string boundaries;
    for (const char * name : {"left", "right", "top"}) {
        boundaries += "[[boundaries]]\nname = \"" + std::string(name) + "\"\ntype = \"outlet_extrapolation\"\n";
    }
    const std::string u = "u = [\"0.3\", \"0.1 * sin(6.0 * x)\"]\n";
#else
    const std::string mesh = "[mesh]\ntype = \"cartesian\"\nNx = 6\nNy = 5\nNz = 4\nLx = 1.0\nLy = 0.8\nLz = 0.6\n";
    const std::string regions = "[regions]\ncut = {type = \"cut_plane\", point = [0.5, 0.4, 0.3], normal = [1.0, 2.0, 0.5]}\n"
                                "ring = {type = \"sphere\", center = [0.5, 0.4, 0.3], radius = 0.25, filled = false}\n";
    std::string boundaries;
    for (const char * name : {"left", "right", "top", "front", "back"}) {
        boundaries += "[[boundaries]]\nname = \"" + std::string(name) + "\"\ntype = \"outlet_extrapolation\"\n";
    }
    const std::string u = "u = [\"0.3\", \"0.1 * sin(6.0 * x)\", \"0.0\"]\n";
#endif
    const std::string input = "[run]\nn_steps = 2\ncfl = 0.3\n" + mesh +
                              "[initialize]\ntype = \"analytical\"\nrho = \"1.0 + 0.2 * sin(5.0 * x) * cos(3.0 * y)\"\n" +
                              u + "p = \"1.0\"\n" + boundaries +
                              "[[boundaries]]\nname = \"bottom\"\ntype = \"wall_isothermal\"\nT = 1.0\n"
                              "[physics]\ntype = \"navier_stokes\"\ngamma = 1.4\np_ref = 1.0\nT_ref = 1.0\nrho_ref = 1.0\n"
                              "mu = 0.01\nPr = 0.72\n"
                              "[numerics.face_reconstruction]\ntype = \"TENO\"\norder = 4\n";
    expect_vtkhdf_output_matches_serial(
        input + regions, "interval = 1\nregions = [\"bottom\", \"cut\", \"ring\"]\n"
               "fields = [\"RHO\", \"U\", \"T\", \"Q_CRITERION\", \"SCHLIEREN\"]\n",
        (scratch_dir() / "mallard_surface_high_order_ranks").string(), {"_cut_000002", "_ring_000002"});
    expect_vtkhdf_output_matches_serial(
        input, "interval = 1\nregion = \"bottom\"\nfields = [\"P\", \"TAU_W\", \"Q_W\"]\n",
        (scratch_dir() / "mallard_surface_high_order_ranks_wall").string(), {"_000002"});
    if (comm::rank() == 0) {
        const std::string dir = (scratch_dir() / "mallard_surface_high_order_ranks").string();
        for (const std::string stem : {"/d_cut_000002", "/d_ring_000002"}) {
            const auto types = h5_values<uint8_t>(dir + stem + ".vtkhdf", "VTKHDF/Types");
            EXPECT_FALSE(types.empty()) << stem;
            for (const uint8_t t : types) {
                EXPECT_TRUE(Mallard_DIM == 2 ? t == 68 : t == 69 || t == 70) << stem << " " << int(t);
            }
        }
    }
}

#endif // Mallard_HAS_HDF5

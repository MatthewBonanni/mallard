/**
 * @file high_order_output_test.cpp
 * @brief Tests of the high-order view: VTK Lagrange cells of the reconstruction polynomials.
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 */

#include <gtest/gtest.h>
#include <Kokkos_Core.hpp>

#include <cmath>
#include <filesystem>
#include <fstream>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#include "comm.h"
#include "data_paths.h"
#include "gmsh_fixtures.h"
#include "lagrange.h"
#include "mesh_block.h"
#include "output_reader.h"
#include "scratch.h"
#include "check.h"
#include "solver.h"
#include "test_fixtures.h"
#include "test_utils.h"

#ifdef Mallard_HAS_HDF5
#include "hdf5_output.h"
#endif

using lagrange::Shape;

namespace {

/** @brief A lattice as VTK 9.7.1's vtkLagrange* cells give it: each node's three digits, space separated. */
std::string digits(const std::vector<lagrange::Lattice> & nodes) {
    std::string s;
    for (const auto & n : nodes) {
        if (!s.empty()) s += ' ';
        for (uint32_t i : n) s += char('0' + i);
    }
    return s;
}

} // namespace

TEST(LagrangeTest, NodesAreInVTKOrder) {
    EXPECT_EQ(digits(lagrange::lattice(Shape::TRIANGLE, 6)),
              "000 600 060 100 200 300 400 500 510 420 330 240 150 050 040 030 020 010 110 410 140 210 310 320 230 130 "
              "120 220");
    EXPECT_EQ(digits(lagrange::lattice(Shape::QUADRILATERAL, 3)),
              "000 300 330 030 100 200 310 320 130 230 010 020 110 210 120 220");
    EXPECT_EQ(digits(lagrange::lattice(Shape::TETRAHEDRON, 6)),
              "000 600 060 006 100 200 300 400 500 510 420 330 240 150 050 040 030 020 010 001 002 003 004 005 501 402 "
              "303 204 105 051 042 033 024 015 101 401 104 201 301 302 203 103 102 202 141 114 411 132 123 213 312 321 "
              "231 222 011 014 041 012 013 023 032 031 021 022 110 140 410 120 130 230 320 310 210 220 111 311 131 113 "
              "211 221 121 112 212 122");
    EXPECT_EQ(digits(lagrange::lattice(Shape::HEXAHEDRON, 3)),
              "000 300 330 030 003 303 333 033 100 200 310 320 130 230 010 020 103 203 313 323 133 233 013 023 001 002 "
              "301 302 331 332 031 032 011 021 012 022 311 321 312 322 101 201 102 202 131 231 132 232 110 210 120 220 "
              "113 213 123 223 111 211 121 221 112 212 122 222");
    EXPECT_EQ(digits(lagrange::lattice(Shape::WEDGE, 6)),
              "000 600 060 006 606 066 100 200 300 400 500 510 420 330 240 150 050 040 030 020 010 106 206 306 406 506 "
              "516 426 336 246 156 056 046 036 026 016 001 002 003 004 005 601 602 603 604 605 061 062 063 064 065 110 "
              "210 310 410 120 220 320 130 230 140 116 216 316 416 126 226 326 136 236 146 101 201 301 401 501 102 202 "
              "302 402 502 103 203 303 403 503 104 204 304 404 504 105 205 305 405 505 511 421 331 241 151 512 422 332 "
              "242 152 513 423 333 243 153 514 424 334 244 154 515 425 335 245 155 051 041 031 021 011 052 042 032 022 "
              "012 053 043 033 023 013 054 044 034 024 014 055 045 035 025 015 111 211 311 411 121 221 321 131 231 141 "
              "112 212 312 412 122 222 322 132 232 142 113 213 313 413 123 223 323 133 233 143 114 214 314 414 124 224 "
              "324 134 234 144 115 215 315 415 125 225 325 135 235 145");
    for (uint32_t order = 1; order <= 6; order++) {
        EXPECT_EQ(lagrange::lattice(Shape::TRIANGLE, order).size(), (order + 1) * (order + 2) / 2);
        EXPECT_EQ(lagrange::lattice(Shape::WEDGE, order).size(), (order + 1) * (order + 1) * (order + 2) / 2);
    }
}

TEST(LagrangeTest, BasisInterpolatesTheHighestMonomialsOfItsSpace) {
    // Simplices span P_p, quadrilaterals and hexahedra Q_p, wedges P_p(x, y) Q_p(z)
    std::mt19937 rng(3);
    std::uniform_real_distribution<double> u(0.0, 1.0);
    for (Shape shape : {Shape::TRIANGLE, Shape::QUADRILATERAL, Shape::TETRAHEDRON, Shape::HEXAHEDRON, Shape::WEDGE}) {
        for (uint32_t p = 1; p <= 5; p++) {
            std::vector<lagrange::Point> points(20);
            for (auto & x : points) {
                x = {u(rng), u(rng), u(rng)};
                if (shape == Shape::TRIANGLE || shape == Shape::WEDGE) x[1] *= 1.0 - x[0];
                if (shape == Shape::TETRAHEDRON) {
                    x[1] *= 1.0 - x[0];
                    x[2] *= 1.0 - x[0] - x[1];
                }
            }
            const auto nodes = lagrange::reference_nodes(shape, p);
            const auto phi = lagrange::basis(shape, p, points);
            ASSERT_EQ(phi.size(), points.size() * nodes.size());
            const bool tensor = shape == Shape::QUADRILATERAL || shape == Shape::HEXAHEDRON;
            const bool planar = shape == Shape::TRIANGLE || shape == Shape::QUADRILATERAL;
            for (uint32_t a = 0; a <= p; a++) {
                const uint32_t b = tensor ? p : p - a;
                const uint32_t c = planar ? 0 : shape == Shape::TETRAHEDRON ? 0 : p;
                auto f = [&](const lagrange::Point & x) {
                    return std::pow(x[0], a) * std::pow(x[1], b) * std::pow(x[2], c);
                };
                for (size_t i = 0; i < points.size(); i++) {
                    double v = 0.0;
                    for (size_t k = 0; k < nodes.size(); k++) v += phi[i * nodes.size() + k] * f(nodes[k]);
                    EXPECT_NEAR(v, f(points[i]), 1e-11) << int(shape) << " order " << p << " a " << a;
                }
            }
        }
    }
}

#if Mallard_DIM == 2 && defined(Mallard_HAS_HDF5)

namespace {

std::string file_contents(const std::string & filename) {
    std::ifstream in(filename, std::ios::binary);
    std::stringstream s;
    s << in.rdbuf();
    return s.str();
}

/** @brief A TENO run on triangles and quadrilaterals whose density is a polynomial of the reconstruction's degree. */
std::string polynomial_case(const std::string & mesh_file, int order, const std::string & writers) {
    const std::string rho = order == 3 ? "1.0 + 0.3 * x - 0.2 * y + 0.4 * x * y + 0.5 * y * y"
                                       : "1.0 + 0.3 * x - 0.2 * y + 0.4 * x * x * y + 0.5 * y * y * y";
    return "[run]\nn_steps = 1\ncfl = 0.2\n"
           "[mesh]\ntype = \"file\"\nfilename = \"" + mesh_file + "\"\n"
           "[initialize]\ntype = \"analytical\"\nrho = \"" + rho + "\"\nu = [\"0.2\", \"0.1\"]\np = \"1.0\"\n"
           "[[boundaries]]\nname = \"left\"\ntype = \"outlet_extrapolation\"\n"
           "[[boundaries]]\nname = \"right\"\ntype = \"outlet_extrapolation\"\n"
           "[[boundaries]]\nname = \"top\"\ntype = \"outlet_extrapolation\"\n"
           "[[boundaries]]\nname = \"bottom\"\ntype = \"outlet_extrapolation\"\n"
           "[physics]\ntype = \"euler\"\ngamma = 1.4\np_ref = 1.0\nT_ref = 1.0\nrho_ref = 1.0\n"
           "[numerics.face_reconstruction]\ntype = \"TENO\"\ntroubled_threshold = 1.0\norder = " + std::to_string(order) + "\n" + writers;
}

double polynomial_rho(int order, double x, double y) {
    return order == 3 ? 1.0 + 0.3 * x - 0.2 * y + 0.4 * x * y + 0.5 * y * y
                      : 1.0 + 0.3 * x - 0.2 * y + 0.4 * x * x * y + 0.5 * y * y * y;
}

} // namespace

TEST(HighOrderOutputTest, NodesHoldThePolynomialItReconstructsExactly) {
    // A density the polynomials reproduce: exact at every node of the cells
    // whose stencils the boundaries' ghost states leave alone
    if (comm::size() > 1) GTEST_SKIP() << "serial";
    const std::string mesh_file = write_temp("high_order_mixed.msh", jittered_mixed_mesh(12));
    for (int order : {3, 4}) {
        const std::string dir = (scratch_dir() / ("mallard_high_order_" + std::to_string(order))).string();
        std::filesystem::remove_all(dir);
        Solver solver;
        solver.set_distributed(false);
        solver.init(parse_toml(polynomial_case(mesh_file, order,
                                               "[[write_data]]\nprefix = \"" + dir + "/f\"\nformat = \"vtkhdf\"\n"
                                               "series = false\ninterval = 1\nfields = [\"RHO\", \"U\", \"TENO_SIGMA\"]\n")));
        solver.run();
        const uint32_t p = uint32_t(order) - 1;
        const std::string file = dir + "/f_high_order_000000.vtkhdf";
        const auto points = h5_values<double>(file, "VTKHDF/Points");
        const auto offsets = h5_values<int64_t>(file, "VTKHDF/Offsets");
        const auto types = h5_values<uint8_t>(file, "VTKHDF/Types");
        const auto rho = h5_values<double>(file, "VTKHDF/PointData/RHO");
        const auto u = h5_values<double>(file, "VTKHDF/PointData/U");
        EXPECT_EQ(h5_values<double>(file, "VTKHDF/CellData/TENO_SIGMA").size(), types.size());
        size_t n_checked = 0;
        for (size_t c = 0; c < types.size(); c++) {
            const size_t first = size_t(offsets[c]), n = size_t(offsets[c + 1]) - first;
            ASSERT_TRUE(types[c] == 69 || types[c] == 70) << int(types[c]);
            EXPECT_EQ(n, lagrange::n_nodes(types[c] == 69 ? Shape::TRIANGLE : Shape::QUADRILATERAL, p));
            bool interior = true;
            for (size_t k = first; k < first + n; k++) {
                const double x = points[3 * k], y = points[3 * k + 1];
                interior = interior && std::min({x, y, 1.0 - x, 1.0 - y}) > 0.26;
            }
            if (!interior) continue;
            n_checked++;
            for (size_t k = first; k < first + n; k++) {
                EXPECT_NEAR(rho[k], polynomial_rho(order, points[3 * k], points[3 * k + 1]), roundoff(1e-12))
                    << "order " << order << " cell " << c << " node " << k - first;
                EXPECT_NEAR(u[3 * k], 0.2, roundoff(1e-12));
                EXPECT_EQ(u[3 * k + 2], 0.0);
            }
        }
        EXPECT_GT(n_checked, 20u);
    }
}

TEST(HighOrderOutputTest, TheViewIsOnByDefaultAndLeavesTheSolutionFilesAlone) {
    if (comm::size() > 1) GTEST_SKIP() << "serial";
    const std::string mesh_file = write_temp("high_order_opt_out.msh", jittered_mixed_mesh(6));
    const std::string dir = (scratch_dir() / "mallard_high_order_opt_out").string();
    std::filesystem::remove_all(dir);
    for (const std::string variant : {"on", "off"}) {
        const std::string key = variant == "on" ? "" : "high_order = false\n";
        Solver solver;
        solver.set_distributed(false);
        solver.init(parse_toml(polynomial_case(
            mesh_file, 3,
            "[[write_data]]\nprefix = \"" + dir + "/" + variant + "/v\"\nformat = \"vtu\"\ninterval = 1\n" + key +
                "fields = [\"RHO\", \"P\"]\n[[write_data]]\nprefix = \"" + dir + "/" + variant +
                "/h\"\nformat = \"vtkhdf\"\ninterval = 1\n" + key + "fields = [\"RHO\", \"MACH\"]\n")));
        solver.run();
    }
    for (const std::string stem : {"v_000001.vtu", "v.pvd"}) {
        // But the run's and the snapshot's blocks, which name the run and its input
        const std::string on = without_run_metadata(file_contents(dir + "/on/" + stem));
        EXPECT_GT(on.size(), 200u) << stem;
        EXPECT_TRUE(on == without_run_metadata(file_contents(dir + "/off/" + stem))) << stem;
    }
    for (const std::string stem : {"h_000001.vtkhdf", "h_mesh.vtkhdf"}) {
        EXPECT_EQ(h5_all_datasets(dir + "/on/" + stem), h5_all_datasets(dir + "/off/" + stem)) << stem;
    }
    for (const std::string view : {"v_high_order_000001.vtu", "h_high_order_000001.vtkhdf"}) {
        const output::Contents c = output::open(dir + "/on/" + view);
        ASSERT_TRUE(c.field("RHO")) << view;
        EXPECT_EQ(c.field("RHO")->location, "point") << view;
        ASSERT_TRUE(c.field(view[0] == 'v' ? "P" : "MACH")) << view;
    }
    for (const auto & entry : std::filesystem::directory_iterator(dir + "/off")) {
        EXPECT_EQ(entry.path().filename().string().find("high_order"), std::string::npos) << entry.path();
    }
}

TEST(HighOrderOutputTest, AWriterOfPartOfTheMeshRejectsTheKey) {
    if (comm::size() > 1) GTEST_SKIP() << "serial";
    const std::string mesh_file = write_temp("high_order_subset.msh", jittered_mixed_mesh(4));
    const std::string dir = (scratch_dir() / "mallard_high_order_subset").string();
    Solver solver;
    solver.set_distributed(false);
    try {
        solver.init(parse_toml(polynomial_case(mesh_file, 3,
                                               "[regions]\nbox = {type = \"rectangle\", center = [0.5, 0.5], size = [0.4, 0.4]}\n"
                                               "[[write_data]]\nprefix = \"" + dir + "/b\"\nregion = \"box\"\n"
                                               "interval = 1\nhigh_order = true\nfields = [\"P\"]\n")));
        FAIL() << "accepted";
    } catch (const std::exception & e) {
        EXPECT_NE(std::string(e.what()).find("high_order"), std::string::npos) << e.what();
    }
}

TEST(HighOrderOutputTest, TheViewNeedsTenoAndAFieldAtTheNodes) {
    if (comm::size() > 1) GTEST_SKIP() << "serial";
    const std::string mesh_file = write_temp("high_order_keys.msh", jittered_mixed_mesh(4));
    const std::string dir = (scratch_dir() / "mallard_high_order_keys").string();
    std::string muscl = polynomial_case(mesh_file, 3, "[[write_data]]\nprefix = \"" + dir + "/b\"\ninterval = 1\n"
                                                     "high_order = true\nfields = [\"P\"]\n");
    const std::string teno = "type = \"TENO\"\ntroubled_threshold = 1.0\norder = 3\n";
    muscl.replace(muscl.find(teno), teno.size(), "type = \"MUSCL\"\n");
    for (const auto & [input, key] : {std::pair{muscl, std::string("TENO")},
                                      {polynomial_case(mesh_file, 3, "[[write_data]]\nprefix = \"" + dir + "/b\"\n"
                                                       "interval = 1\nhigh_order = true\nfields = [\"TENO_SIGMA\"]\n"),
                                       std::string("values at the nodes")}}) {
        Solver solver;
        solver.set_distributed(false);
        try {
            solver.init(parse_toml(input));
            ADD_FAILURE() << "accepted " << key;
        } catch (const std::exception & e) {
            EXPECT_NE(std::string(e.what()).find(key), std::string::npos) << e.what();
        }
    }
}

TEST(HighOrderOutputTest, TheDefaultIsOffWhereTheViewDoesNotApply) {
    if (comm::size() > 1) GTEST_SKIP() << "serial";
    const std::string mesh_file = write_temp("high_order_default.msh", jittered_mixed_mesh(4));
    const std::string dir = (scratch_dir() / "mallard_high_order_default").string();
    const auto writer = [&](const std::string & keys) {
        return "[[write_data]]\nprefix = \"" + dir + "/w\"\ninterval = 1\n" + keys;
    };
    std::string muscl = polynomial_case(mesh_file, 3, writer("format = \"vtu\"\nfields = [\"RHO\"]\n"));
    const std::string teno = "type = \"TENO\"\ntroubled_threshold = 1.0\norder = 3\n";
    muscl.replace(muscl.find(teno), teno.size(), "type = \"MUSCL\"\n");
    for (const std::string & input :
         {muscl, polynomial_case(mesh_file, 3, writer("region = \"left\"\nfields = [\"P\"]\n")),
          polynomial_case(mesh_file, 3, writer("format = \"restart\"\n")),
          polynomial_case(mesh_file, 3, writer("format = \"vtu\"\nfields = [\"TENO_SIGMA\"]\n"))}) {
        Solver checked;
        checked.set_distributed(false);
        const CheckReport r = checked.check(parse_toml(input), CheckOptions{});
        EXPECT_TRUE(r.unused.empty()) << input;
        for (const CheckDiagnostic & d : r.diagnostics) EXPECT_NE(d.severity, report::Status::ERROR) << d.message;
        for (const CheckOutput & o : r.outputs) EXPECT_EQ(o.high_order_bytes, 0.0) << input;
    }
}

TEST(HighOrderOutputTest, CheckProjectsTheSizeOfEachSnapshotAndItsView) {
    if (comm::size() > 1) GTEST_SKIP() << "serial";
    // Large enough that the values outweigh each file's metadata, which the projection leaves out
    const std::string mesh_file = write_temp("high_order_check.msh", jittered_mixed_mesh(24));
    const std::string dir = (scratch_dir() / "mallard_high_order_check").string();
    std::filesystem::remove_all(dir);
    const std::string input = polynomial_case(mesh_file, 4,
                                              "[[write_data]]\nprefix = \"" + dir + "/v\"\nformat = \"vtu\"\n"
                                              "interval = 1\nfields = [\"RHO\", \"U\", \"P\", \"TENO_SIGMA\"]\n"
                                              "[[write_data]]\nprefix = \"" + dir + "/c\"\nformat = \"vtu\"\n"
                                              "interval = 1\nhigh_order = false\nfields = [\"RHO\"]\n");
    Solver checked;
    checked.set_distributed(false);
    const CheckReport r = checked.check(parse_toml(input), CheckOptions{});
    ASSERT_EQ(r.outputs.size(), 2u);
    EXPECT_EQ(r.outputs[1].high_order_bytes, 0.0);
    Solver solver;
    solver.set_distributed(false);
    solver.init(parse_toml(input));
    solver.run();
    const double cells = double(std::filesystem::file_size(dir + "/v_000001.vtu"));
    const double view = double(std::filesystem::file_size(dir + "/v_high_order_000001.vtu"));
    EXPECT_NEAR(r.outputs[0].bytes / cells, 1.0, 0.5);  // the run block is a large share of a small file
    EXPECT_NEAR(r.outputs[0].high_order_bytes / view, 1.0, 0.1);
    EXPECT_GT(view, 4.0 * cells);
}

#endif // Mallard_DIM == 2 && Mallard_HAS_HDF5

#if Mallard_DIM == 3 && defined(Mallard_HAS_HDF5)

TEST(HighOrderOutputTest, NodesHoldThePolynomialItReconstructsExactlyOnEveryCellShape) {
    // Hexahedra, tetrahedra, prisms and pyramids (as two tetrahedra each): a
    // quadratic density is exact at every node of cells clear of the walls,
    // and the cells fill the box, positive, without overlap
    if (comm::size() > 1) GTEST_SKIP() << "serial";
    const std::vector<std::pair<std::string, uint8_t>> meshes = {
        {"cartesian", 72}, {"cartesian_tet", 71}, {"cartesian_prism", 73}, {"cartesian_pyramid", 71}};
    auto rho = [](double x, double y, double z) {
        return 1.0 + 0.3 * x - 0.2 * y + 0.1 * z + 0.4 * x * y + 0.5 * z * z - 0.3 * y * z;
    };
    for (const auto & [type, vtk_type] : meshes) {
        const std::string dir = (scratch_dir() / ("mallard_high_order_" + type)).string();
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
            "[[write_data]]\nprefix = \"" + dir + "/f\"\nseries = false\ninterval = 1\nhigh_order = true\n"
            "fields = [\"RHO\", \"TENO_SIGMA\"]\n"));
        solver.run();
        const std::string file = dir + "/f_high_order_000000.vtkhdf";
        const auto points = h5_values<double>(file, "VTKHDF/Points");
        const auto offsets = h5_values<int64_t>(file, "VTKHDF/Offsets");
        const auto types = h5_values<uint8_t>(file, "VTKHDF/Types");
        const auto values = h5_values<double>(file, "VTKHDF/PointData/RHO");
        const size_t n_cells = h5_values<double>(dir + "/f_000000.vtkhdf", "VTKHDF/CellData/TENO_SIGMA").size();
        EXPECT_EQ(types.size(), (type == "cartesian_pyramid" ? 2 : 1) * n_cells) << type;
        EXPECT_EQ(h5_values<double>(file, "VTKHDF/CellData/TENO_SIGMA").size(), types.size()) << type;
        double volume = 0.0;
        const Shape shape = vtk_type == 71 ? Shape::TETRAHEDRON : vtk_type == 73 ? Shape::WEDGE : Shape::HEXAHEDRON;
        size_t n_checked = 0;
        for (size_t c = 0; c < types.size(); c++) {
            const size_t first = size_t(offsets[c]), n = size_t(offsets[c + 1]) - first;
            ASSERT_EQ(types[c], vtk_type) << type;
            ASSERT_EQ(n, lagrange::n_nodes(shape, 2)) << type;
            if (shape == Shape::TETRAHEDRON) {
                double e[3][3];
                for (size_t a = 0; a < 3; a++) {
                    for (size_t i = 0; i < 3; i++) e[a][i] = points[3 * (first + a + 1) + i] - points[3 * first + i];
                }
                const double v = (e[0][0] * (e[1][1] * e[2][2] - e[1][2] * e[2][1]) -
                                  e[0][1] * (e[1][0] * e[2][2] - e[1][2] * e[2][0]) +
                                  e[0][2] * (e[1][0] * e[2][1] - e[1][1] * e[2][0])) / 6.0;
                EXPECT_GT(v, 0.0) << type << " cell " << c;
                volume += v;
            }
            bool interior = true;
            for (size_t k = first; k < first + n; k++) {
                for (size_t i = 0; i < 3; i++) interior = interior && std::min(points[3 * k + i], 1.0 - points[3 * k + i]) > 0.3;
            }
            if (!interior) continue;
            n_checked++;
            for (size_t k = first; k < first + n; k++) {
                EXPECT_NEAR(values[k], rho(points[3 * k], points[3 * k + 1], points[3 * k + 2]), roundoff(1e-11))
                    << type << " cell " << c << " node " << k - first;
            }
        }
        EXPECT_GT(n_checked, 7u) << type;
        if (shape == Shape::TETRAHEDRON) {
            EXPECT_NEAR(volume, 1.0, roundoff(1e-12)) << type;
        }
    }
}

#endif // Mallard_DIM == 3 && Mallard_HAS_HDF5

#ifdef Mallard_HAS_HDF5

TEST(HighOrderOutputTest, CurvedWallsKeepTheirExactShape) {
    // The cylinder's cells written with every node of the wall on the circle,
    // where straight-sided cells would put the edges' inner nodes inside it
    if (comm::size() > 1) GTEST_SKIP() << "serial";
    const std::string dir = (scratch_dir() / "mallard_high_order_curved").string();
    std::filesystem::remove_all(dir);
#if Mallard_DIM == 2
    const std::string mesh = source_dir() + "/test/data/meshes/cyl2d.msh";
    const std::string wall = "type = \"circle\"\ncenter = [0.5, 0.5]\nradius = 0.2\n";
    const std::string u = "u = [\"0.2\", \"0.0\"]\n";
    std::vector<std::string> zones = {"inlet", "outlet", "walls", "cylinder"};
#else
    const std::string mesh = source_dir() + "/test/data/meshes/cyl3d.msh";
    const std::string wall = "type = \"cylinder\"\ncenter = [0.5, 0.5, 0.0]\naxis = [0.0, 0.0, 1.0]\nradius = 0.2\n"
                             "filled = false\n";
    const std::string u = "u = [\"0.2\", \"0.0\", \"0.0\"]\n";
    std::vector<std::string> zones = {"inlet", "outlet", "walls", "cylinder", "front", "back"};
#endif
    std::string boundaries;
    for (const std::string & zone : zones) boundaries += "[[boundaries]]\nname = \"" + zone + "\"\ntype = \"outlet_extrapolation\"\n";
    Solver solver;
    solver.set_distributed(false);
    solver.init(parse_toml("[run]\nn_steps = 1\ncfl = 0.2\n[mesh]\ntype = \"file\"\nfilename = \"" + mesh + "\"\n"
                           "[[mesh.curved]]\nzone = \"cylinder\"\n" + wall +
                           "[initialize]\ntype = \"analytical\"\nrho = \"1.0\"\n" + u + "p = \"1.0\"\n" + boundaries +
                           "[physics]\ntype = \"euler\"\ngamma = 1.4\np_ref = 1.0\nT_ref = 1.0\nrho_ref = 1.0\n"
                           "[numerics.face_reconstruction]\ntype = \"TENO\"\norder = 4\n"
                           "[[write_data]]\nprefix = \"" + dir + "/f\"\nseries = false\ninterval = 1\nhigh_order = true\n"
                           "fields = [\"RHO\"]\n"));
    solver.run();
    const std::string file = dir + "/f_high_order_000000.vtkhdf";
    const auto points = h5_values<double>(file, "VTKHDF/Points");
    const auto offsets = h5_values<int64_t>(file, "VTKHDF/Offsets");
    const auto types = h5_values<uint8_t>(file, "VTKHDF/Types");
    const auto rho = h5_values<double>(file, "VTKHDF/PointData/RHO");
    size_t inner_on_wall = 0;  // nodes but vertices
    const double tol = roundoff(1e-12);  // the nodes' coordinates are written in the build's precision
    for (size_t c = 0; c < types.size(); c++) {
        const int64_t n_vertices = types[c] == 69 ? 3 : types[c] == 70 || types[c] == 71 ? 4 : types[c] == 72 ? 8 : 6;
        for (size_t k = size_t(offsets[c]); k < size_t(offsets[c + 1]); k++) {
            const double r = std::hypot(points[3 * k] - 0.5, points[3 * k + 1] - 0.5);
            EXPECT_GT(r, 0.2 - tol) << "cell " << c << " node " << k;
            inner_on_wall += r < 0.2 + tol && int64_t(k) - offsets[c] >= n_vertices;
            EXPECT_NEAR(rho[k], 1.0, tol);
        }
    }
    EXPECT_GE(inner_on_wall, 16u);  // two per cubic edge on the wall, 8 of them in 2D
}

TEST(HighOrderOutputTest, ViewFilesDoNotDependOnTheRanks) {
    if (comm::size() > 1 && !have_parallel_hdf5()) GTEST_SKIP() << "needs parallel HDF5";
#if Mallard_DIM == 2
    const std::string mesh = "[mesh]\ntype = \"cartesian\"\nNx = 10\nNy = 8\nLx = 1.0\nLy = 0.8\n";
    const std::string boundaries = "[[boundaries]]\nname = \"left\"\ntype = \"outlet_extrapolation\"\n"
                                   "[[boundaries]]\nname = \"right\"\ntype = \"outlet_extrapolation\"\n"
                                   "[[boundaries]]\nname = \"top\"\ntype = \"outlet_extrapolation\"\n"
                                   "[[boundaries]]\nname = \"bottom\"\ntype = \"outlet_extrapolation\"\n";
    const std::string u = "u = [\"0.3\", \"0.1 * sin(6.0 * x)\"]\n";
#else
    const std::string mesh = "[mesh]\ntype = \"cartesian\"\nNx = 6\nNy = 5\nNz = 4\nLx = 1.0\nLy = 0.8\nLz = 0.6\n";
    std::string boundaries;
    for (const char * name : {"left", "right", "top", "bottom", "front", "back"}) {
        boundaries += "[[boundaries]]\nname = \"" + std::string(name) + "\"\ntype = \"outlet_extrapolation\"\n";
    }
    const std::string u = "u = [\"0.3\", \"0.1 * sin(6.0 * x)\", \"0.0\"]\n";
#endif
    const std::string input = "[run]\nn_steps = 2\ncfl = 0.3\n" + mesh +
                              "[initialize]\ntype = \"analytical\"\nrho = \"1.0 + 0.2 * sin(5.0 * x) * cos(3.0 * y)\"\n" +
                              u + "p = \"1.0\"\n" + boundaries +
                              "[physics]\ntype = \"euler\"\ngamma = 1.4\np_ref = 1.0\nT_ref = 1.0\nrho_ref = 1.0\n"
                              "[numerics.face_reconstruction]\ntype = \"TENO\"\norder = 4\n";
    expect_vtkhdf_output_matches_serial(input, "interval = 1\nseries = false\nhigh_order = true\nfields = [\"RHO\", \"U\", \"T\"]\n",
                                        (scratch_dir() / "mallard_high_order_ranks").string(),
                                        {"_high_order_000002", "_000002"});
}

#endif // Mallard_HAS_HDF5

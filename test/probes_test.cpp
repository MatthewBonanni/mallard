/**
 * @file probes_test.cpp
 * @author Matthew Bonanni (mbonanni001@gmail.com)
 * @brief [[probes]]: geometry, point values, reductions and output.
 * @version 0.1
 * @date 2026-10-15
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 *
 */

#include <gtest/gtest.h>
#include <Kokkos_Core.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iomanip>
#include <limits>
#include <map>
#include <numbers>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "comm.h"
#include "data_paths.h"
#include "json.h"
#include "output_reader.h"
#include "gmsh_fixtures.h"
#include "file_format.h"
#include "input.h"
#include "mechanism.h"
#include "probe_geometry.h"
#include "probe_spectra.h"
#include "region_input.h"
#include "region_mesh.h"
#include "scratch.h"
#include "gmsh_fixtures.h"
#include "solver.h"
#include "test_fixtures.h"
#include "test_utils.h"
#include "transport.h"

#ifdef Mallard_HAS_HDF5
#include "hdf5_output.h"
#include "hdf5_util.h"
#endif

namespace {

const std::string PERFECT_GAS = "[physics]\ntype = \"euler\"\ngamma = 1.4\np_ref = 1.0\nT_ref = 1.0\nrho_ref = 1.0\n";

/** @brief A relative tolerance, no tighter than the rounding of single-precision fields and mesh geometry. */
double rounding_or(const double tolerance) {
    return std::max(tolerance, 64.0 * static_cast<double>(std::numeric_limits<rtype>::epsilon()));
}

std::string numbers3(const char * x, const char * y, const char * z) {
    return std::string("[") + x + ", " + y + (N_DIM == 3 ? std::string(", ") + z : "") + "]";
}

std::string vector3(const char * x, const char * y, const char * z) {
    return std::string("[\"") + x + "\", \"" + y + "\"" + (N_DIM == 3 ? std::string(", \"") + z + "\"" : "") + "]";
}

/** @brief An empty directory of the test, the same on every rank. */
std::string fresh_dir(const std::string & name) {
    const std::string dir = (scratch_dir() / ("probes_" + name)).string();
    comm::barrier();
    if (comm::is_root()) std::filesystem::remove_all(dir);
    comm::barrier();
    return dir;
}

/** @brief Runs a solver; afterwards every rank can read the files rank 0 wrote. */
void run(Solver & solver) {
    solver.run();
    comm::barrier();
}

/** @brief A periodic box of n cells per side (n / 2 along z) with the given initial fields and reconstruction. */
std::string box_case(uint32_t n, const std::string & rho, const std::string & recon, const std::string & extra,
                     const std::string & run = "n_steps = 1\ndt = 1.0e-6\n") {
    std::ostringstream s;
    s << "[run]\n" << run << "[mesh]\ntype = \"cartesian\"\nNx = " << n << "\nNy = " << n << "\nLx = 1.0\nLy = 1.0\n";
    if (N_DIM == 3) s << "Nz = " << std::max(n / 2, 3u) << "\nLz = 1.0\n[[periodic]]\nzones = [\"left\", \"right\"]\n[[periodic]]\nzones = [\"bottom\", \"top\"]\n[[periodic]]\nzones = [\"back\", \"front\"]\n";
    else s << "[[periodic]]\nzones = [\"left\", \"right\"]\n[[periodic]]\nzones = [\"bottom\", \"top\"]\n";
    s << "[initialize]\ntype = \"analytical\"\nrho = \"" << rho << "\"\nu = " << vector3("0.3", "-0.1", "0.2")
      << "\np = \"1.0\"\n[numerics]\nriemann_solver = \"HLLC\"\ntime_integrator = \"SSPRK3\"\n"
      << "[numerics.face_reconstruction]\n" << recon << PERFECT_GAS << "[output]\ncheck_interval = 1000000\n" << extra;
    return s.str();
}

/** @brief Rows of a CSV file as numbers, keyed by the header's names. */
std::vector<std::map<std::string, double>> read_csv(const std::string & file) {
    std::ifstream in(file);
    std::string line;
    std::getline(in, line);
    std::vector<std::string> names;
    std::stringstream hs(line);
    std::string cell;
    while (std::getline(hs, cell, ',')) names.push_back(cell);
    std::vector<std::map<std::string, double>> rows;
    while (std::getline(in, line)) {
        if (line.empty() || line[0] == '#') continue;
        std::stringstream ss(line);
        std::map<std::string, double> row;
        for (size_t i = 0; std::getline(ss, cell, ','); i++) row[names.at(i)] = std::stod(cell);
        rows.push_back(row);
    }
    return rows;
}

std::string read_all(const std::string & file) {
    std::ifstream in(file);
    std::stringstream ss;
    ss << in.rdbuf();
    return ss.str();
}

double sine_rho(double x, double y) {
    return 1.0 + 0.2 * std::sin(2.0 * std::numbers::pi * x) * std::sin(2.0 * std::numbers::pi * y);
}
const char * const SINE_RHO = "1.0 + 0.2 * sin(2 * pi * x) * sin(2 * pi * y)";

/** @brief Largest error of RHO and T = p / rho at the step-0 point values of probe points off the cell centroids. */
double point_error(uint32_t n, const std::string & recon, const std::string & sampling) {
    const std::string dir = fresh_dir("order");
    // Points near fixed ones at the same place in their cells on every mesh, so
    // that the error constants do not change with the mesh
    const double near[3][3] = {{0.137, 0.462, 0.31}, {0.713, 0.291, 0.77}, {0.402, 0.858, 0.53}};
    const double within[3] = {0.3, 0.7, 0.6};
    const uint32_t cells[3] = {n, n, std::max(n / 2, 3u)};
    std::ostringstream points;
    points.precision(17);
    for (int k = 0; k < 3; k++) {
        points << (k ? ", [" : "[");
        for (size_t d = 0; d < N_DIM; d++) {
            points << (d ? ", " : "") << (std::floor(near[k][d] * cells[d]) + within[d]) / cells[d];
        }
        points << "]";
    }
    const std::string probe = "[[probes]]\nname = \"p\"\nfile = \"" + dir + "/p.csv\"\nfields = [\"RHO\", \"T\"]\n"
                              "sampling = \"" + sampling + "\"\ntype = \"points\"\npoints = [" + points.str() + "]\n";
    Solver solver;
    solver.init(parse_toml(box_case(n, SINE_RHO, recon, probe)));
    run(solver);
    double error = 0.0;
    for (const auto & row : read_csv(dir + "/p.csv")) {
        if (row.at("step") != 0.0) continue;
        const double rho = sine_rho(row.at("x"), row.at("y"));
        error = std::max({error, std::fabs(row.at("RHO") - rho), std::fabs(row.at("T") - 1.0 / rho)});
    }
    return error;
}

/** @brief Integral of f over [a, b] by Boole's rule, exact for polynomials of degree 5. */
template <typename F>
double boole(F f, double a, double b) {
    const double h = (b - a) / 4.0;
    return 2.0 * h / 45.0 * (7.0 * f(a) + 32.0 * f(a + h) + 12.0 * f(a + 2 * h) + 32.0 * f(a + 3 * h) + 7.0 * f(b));
}

} // namespace

TEST(ProbesTest, CurvesHaveTheirLengthsAndFollowTheirTransforms) {
    const double pi = std::numbers::pi;
    auto length = [](const probe::Geometry & g) {
        double sum = 0.0;
        for (double w : g.quad_weights) sum += w;
        return sum;
    };
    const std::string normal = N_DIM == 3 ? "normal = [0.0, 0.0, 1.0]\n" : "";
    const probe::Geometry circle = probe::make_geometry(
        parse_toml("type = \"circle\"\ncenter = " + numbers3("0.5", "0.5", "0.5") + "\nradius = 0.25\nn_points = 7\n" +
                   normal),
        "circle");
    EXPECT_NEAR(length(circle), 2.0 * pi * 0.25, 1e-14);
    ASSERT_EQ(circle.nodes.size(), 7u);
    ASSERT_EQ(circle.polylines.size(), 1u);
    EXPECT_EQ(circle.polylines[0].size(), 8u);
    EXPECT_EQ(circle.polylines[0].front(), circle.polylines[0].back());

    const probe::Geometry arc = probe::make_geometry(
        parse_toml("type = \"arc\"\ncenter = " + numbers3("0", "0", "0") +
                   "\nradius = 2.0\nangles = [90.0, 0.0]\nn_points = 5\n" + normal),
        "arc");
    EXPECT_NEAR(length(arc), pi, 1e-14);
    EXPECT_NEAR(arc.nodes.front()[1], 2.0, 1e-14);
    EXPECT_NEAR(arc.nodes.back()[0], 2.0, 1e-14);
    EXPECT_NEAR(arc.arc_length.back(), pi, 1e-14);

    // x = s, y = s^2 over [0, 1]: (2 sqrt(5) + asinh(2)) / 4
    const probe::Geometry parabola = probe::make_geometry(
        parse_toml(std::string("type = \"curve\"\nx = \"s\"\ny = \"s^2\"\n") + (N_DIM == 3 ? "z = \"0.5\"\n" : "") +
                   "n_points = 41\n"),
        "parabola");
    EXPECT_NEAR(length(parabola), (2.0 * std::sqrt(5.0) + std::asinh(2.0)) / 4.0, 1e-7);

    // The unit segment along x, scaled by 2, turned a quarter about z, moved by (1, 1)
    const std::string rotate = N_DIM == 3 ? "rotate = {axis = [0.0, 0.0, 1.0], angle = 90.0}\n" : "rotate = 90.0\n";
    const probe::Geometry line = probe::make_geometry(
        parse_toml("start = " + numbers3("0", "0", "0") + "\nend = " + numbers3("1", "0", "0") +
                   "\nn_points = 3\nscale = 2.0\n" + rotate + "translate = " + numbers3("1", "1", "0") + "\n"),
        "line");
    EXPECT_NEAR(line.nodes[0][0], 1.0, 1e-14);
    EXPECT_NEAR(line.nodes[0][1], 1.0, 1e-14);
    EXPECT_NEAR(line.nodes[2][0], 1.0, 1e-14);
    EXPECT_NEAR(line.nodes[2][1], 3.0, 1e-14);
    EXPECT_NEAR(length(line), 2.0, 1e-14);

    // A quarter turn about the pivot (1, 1) takes (2, 1) to (1, 2)
    const probe::Geometry turned = probe::make_geometry(
        parse_toml("point = " + numbers3("2", "1", "0") + "\n" + rotate + "pivot = " + numbers3("1", "1", "0") + "\n"),
        "turned");
    EXPECT_NEAR(turned.nodes[0][0], 1.0, 1e-14);
    EXPECT_NEAR(turned.nodes[0][1], 2.0, 1e-14);
#if N_DIM == 3
    // The same turn as the rotated axes x -> y, y -> -x, z -> z
    const probe::Geometry axes = probe::make_geometry(
        parse_toml("point = [2.0, 1.0, 0.5]\nrotate = {axes = [[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]}\n"
                   "pivot = [1.0, 1.0, 0.0]\n"),
        "axes");
    EXPECT_NEAR(axes.nodes[0][0], 1.0, 1e-14);
    EXPECT_NEAR(axes.nodes[0][1], 2.0, 1e-14);
    EXPECT_NEAR(axes.nodes[0][2], 0.5, 1e-14);
#endif
}

TEST(ProbesTest, ProbeFilesRecordWhereEachPointIsRead) {
    // One point off its cell's centroid and one on the domain's edge, on 8 cells per side
    const std::string dir = fresh_dir("sampling");
    const std::vector<std::array<double, 3>> requested = {{0.3, 0.4, 0.6}, {1.0, 0.4, 0.6}};
    const std::vector<std::array<double, 3>> centroid = {{0.3125, 0.4375, 0.625}, {0.9375, 0.4375, 0.625}};
    std::vector<std::pair<std::string, std::string>> files = {{"csv", "cell"}, {"vtp", "point"}};
    if (have_hdf5()) files.insert(files.end(), {{"hdf5", "point"}, {"vtkhdf", "cell"}});
    std::string extra;
    for (const auto & [format, sampling] : files) {
        extra += "[[probes]]\nname = \"" + format + "\"\nformat = \"" + format + "\"\nsampling = \"" + sampling +
                 "\"\nfile = \"" + dir + "/" + format + "." + (format == "vtp" ? "pvd" : format == "hdf5" ? "h5" : format) +
                 "\"\ntype = \"points\"\npoints = [" + numbers3("0.3", "0.4", "0.6") + ", " + numbers3("1.0", "0.4", "0.6") +
                 "]\nfields = [\"RHO\", \"P\"]\n";
    }
    Solver solver;
    solver.init(parse_toml(box_case(8, SINE_RHO, "type = \"TENO\"\norder = 3\n", extra)));
    run(solver);
    double cell = -1.0;
    for (const auto & [format, sampling] : files) {
        const std::string path = dir + "/" + format + "." + (format == "vtp" ? "pvd" : format == "hdf5" ? "h5" : format);
        SCOPED_TRACE(path);
        const output::Contents file = output::open(path);
        ASSERT_FALSE(file.sampling.empty());
        const json::Value b = json::parse(file.sampling);
        EXPECT_EQ(b["schema"].string, "mallard-probe-sampling");
        EXPECT_EQ(b["sampling"].string, sampling);
        const bool point = sampling == "point";
        if (point) {
            EXPECT_EQ(b["reconstruction"]["method"].string, "TENO");
            EXPECT_EQ(b["reconstruction"]["order"].number, 3);
        } else {
            EXPECT_TRUE(b["reconstruction"].is_null());
        }
        const auto & variables = b["variables"].array;
        ASSERT_EQ(variables.size(), 2u);
        EXPECT_EQ(variables[0]["value"].string, point ? "point_value" : "cell_average");
        EXPECT_EQ(variables[1]["value"].string, point ? "point_value" : "cell_state");
        const json::Value & p = b["points"];
        ASSERT_EQ(p["n"].number, 2);
        EXPECT_TRUE(b["clipped"].is_null());
        for (size_t i = 0; i < 2; i++) {
            double d2 = 0.0;
            for (size_t d = 0; d < N_DIM; d++) {
                EXPECT_DOUBLE_EQ(p["requested"].array[i].array[d].number, requested[i][d]) << i;
                EXPECT_DOUBLE_EQ(p["centroid"].array[i].array[d].number, centroid[i][d]) << i;
                d2 += (requested[i][d] - centroid[i][d]) * (requested[i][d] - centroid[i][d]);
            }
            // Point values are read at the requested point, cell values at the centroid
            const double expected = point ? 0.0 : std::sqrt(d2);
            EXPECT_NEAR(p["distance"].array[i].number, expected, 1e-14) << i;
        }
        if (cell < 0.0) cell = p["cell"].array[0].number;
        EXPECT_EQ(p["cell"].array[0].number, cell);
    }
}

TEST(ProbesTest, PointFilesSkipHeadersAndComments) {
    const std::string file = "cloud_" + std::to_string(comm::rank()) + ".csv";
    std::ofstream(file) << "x, y, z\n# a comment\n0.1, 0.2, 0.3\n\n0.4 0.5 0.6  # trailing\n";
    const auto points = probe::read_points(file, "test");
    ASSERT_EQ(points.size(), 2u);
    EXPECT_EQ(points[1][0], 0.4);
    EXPECT_EQ(points[1][1], 0.5);
    std::ofstream(file) << "0.1, 0.2, 0.3\nx, y\n";
    EXPECT_THROW(probe::read_points(file, "test"), InputError);
}

TEST(ProbesTest, PointValuesConvergeAtTheSchemesOrder) {
    // TENO's central polynomials give point values of the scheme's order, and
    // the temperature of p / rho at the point follows; linear least squares
    // is second order, cell averages first order at points off the centroids
    const uint32_t order = N_DIM == 2 ? 5 : 3;
    // Coarse meshes of a smooth field still read as troubled to the default threshold
    const std::string teno = "type = \"TENO\"\norder = " + std::to_string(order) + "\ntroubled_threshold = 1.0\n";
    // Single precision: meshes whose errors stay above its round-off
    const bool single = sizeof(rtype) == sizeof(float);
    const std::vector<uint32_t> n = N_DIM == 3 ? std::vector<uint32_t>{8, 16}
                                    : single  ? std::vector<uint32_t>{10, 20}
                                              : std::vector<uint32_t>{10, 20, 40};
    std::vector<double> e_teno, e_lsq, e_cell;
    for (uint32_t k : n) {
        e_teno.push_back(point_error(k, teno, "point"));
        e_lsq.push_back(point_error(k, "type = \"MUSCL\"\nlimiter = \"none\"\n", "point"));
        e_cell.push_back(point_error(k, teno, "cell"));
    }
    const size_t last = n.size() - 1;
    const double rate_teno = std::log2(e_teno[last - 1] / e_teno[last]);
    const double rate_lsq = std::log2(e_lsq[last - 1] / e_lsq[last]);
    EXPECT_GT(rate_teno, order - 0.6) << e_teno[last - 1] << " " << e_teno[last];
    EXPECT_GT(rate_lsq, 1.6) << e_lsq[last - 1] << " " << e_lsq[last];
    EXPECT_LT(e_teno[last], 0.25 * e_cell[last]);
}

TEST(ProbesTest, CurveReductionsAreExactForPolynomialFields) {
    // TENO of order 3 reproduces quadratics, and three Gauss points per
    // element integrate their squares along straight elements exactly
    const char * rho = "2.0 + 0.3 * x - 0.2 * y + 0.25 * x^2 + 0.15 * x * y - 0.1 * y^2";
    auto f = [](double x, double y) { return 2.0 + 0.3 * x - 0.2 * y + 0.25 * x * x + 0.15 * x * y - 0.1 * y * y; };
    const std::string dir = fresh_dir("exact");
    const std::string probe = "[[probes]]\nname = \"poly\"\nfile = \"" + dir + "/poly.csv\"\nfields = [\"RHO\", \"P\"]\n"
                              "sampling = \"point\"\ntype = \"polyline\"\nvertices = [" +
                              numbers3("0.35", "0.42", "0.45") + ", " + numbers3("0.62", "0.57", "0.45") + ", " +
                              numbers3("0.48", "0.66", "0.45") + "]\nn_points = 4\n"
                              "reductions = [\"integral\", \"mean\", \"favre\", \"rms\", \"min\", \"max\"]\n"
                              "reductions_file = \"" + dir + "/poly_reductions.csv\"\n";
    Solver solver;
    solver.init(parse_toml(box_case(20, rho, "type = \"TENO\"\norder = 3\ntroubled_threshold = 1.0\n", probe)));
    run(solver);

    const double v[3][2] = {{0.35, 0.42}, {0.62, 0.57}, {0.48, 0.66}};
    double length = 0.0, integral = 0.0, square = 0.0;
    for (int k = 0; k < 2; k++) {
        const double dx = v[k + 1][0] - v[k][0], dy = v[k + 1][1] - v[k][1], L = std::hypot(dx, dy);
        auto along = [&](double s) { return f(v[k][0] + s * dx, v[k][1] + s * dy); };
        length += L;
        integral += L * boole(along, 0.0, 1.0);
        square += L * boole([&](double s) { return std::pow(along(s), 2); }, 0.0, 1.0);
    }
    const auto rows = read_csv(dir + "/poly_reductions.csv");
    ASSERT_FALSE(rows.empty());
    const auto & r = rows.front();
    const double tol = precision_tol<double>(1e-12, 1e-5);
    EXPECT_NEAR(r.at("length"), length, tol * length);
    EXPECT_NEAR(r.at("integral_RHO"), integral, tol * integral);
    EXPECT_NEAR(r.at("mean_RHO"), integral / length, tol * integral / length);
    EXPECT_NEAR(r.at("favre_RHO"), square / integral, tol * square / integral);
    EXPECT_NEAR(r.at("rms_RHO"), std::sqrt(square / length), tol * 2.0);
    EXPECT_NEAR(r.at("integral_P"), length, tol * length);
    EXPECT_NEAR(r.at("min_P"), 1.0, tol);
    EXPECT_NEAR(r.at("max_P"), 1.0, tol);
    // The extrema bound the written samples
    for (const auto & row : read_csv(dir + "/poly.csv")) {
        if (row.at("step") != 0.0) continue;
        EXPECT_GE(rtype(row.at("RHO")), rtype(r.at("min_RHO")));
        EXPECT_LE(rtype(row.at("RHO")), rtype(r.at("max_RHO")));
    }
}

TEST(ProbesTest, RegionReductionsAreFractionWeightedIntegralsOfTheCellAverages) {
    // A box on cell faces covers whole cells, so its reductions are exact for
    // a field whose cell averages are exact (a quadratic); a disk (ball)
    // covers cells in part, and its measure is the sum of their fractions
    const std::string dir = fresh_dir("region");
    const std::string box = N_DIM == 2 ? "{type = \"rectangle\", center = [0.45, 0.45], size = [0.5, 0.3]}"
                                       : "{type = \"box\", center = [0.45, 0.45, 0.5], size = [0.5, 0.3, 0.6]}";
    const std::string ball = "{type = \"sphere\", center = " + numbers3("0.5", "0.5", "0.5") + ", radius = 0.27}";
    const std::string probes =
        "[[probes]]\nname = \"box\"\nfields = [\"RHO\", \"P\"]\nregion = " + box + "\nreductions_file = \"" + dir +
        "/box.csv\"\n[[probes]]\nname = \"ball\"\nfields = [\"P\"]\nregion = " + ball +
        "\nreductions = [\"integral\", \"mean\", \"rms\"]\nreductions_file = \"" + dir + "/ball.csv\"\n";
    Solver solver;
    solver.init(parse_toml(box_case(20, "1 + x + 2 * y + 0.5 * x^2", "type = \"MUSCL\"\n", probes)));
    run(solver);

    const double x0 = 0.2, x1 = 0.7, y0 = 0.3, y1 = 0.6, depth = N_DIM == 3 ? 0.6 : 1.0;
    const double lx = x1 - x0, ly = y1 - y0, volume = lx * ly * depth;
    auto cube = [](double a, double b) { return (b * b * b - a * a * a) / 3.0; };
    const double integral =
        depth * (lx * ly + ly * 0.5 * (x1 * x1 - x0 * x0) + lx * (y1 * y1 - y0 * y0) + 0.5 * ly * cube(x0, x1));
    // Extrema: the averages of the corner cells, where the field is least and largest
    auto cell_average = [&](double xa, double ya) {
        return 1.0 + (xa + 0.025) + 2.0 * (ya + 0.025) + 0.5 * cube(xa, xa + 0.05) / 0.05;
    };
    const std::string measure = N_DIM == 3 ? "volume" : "area";
    const auto rows = read_csv(dir + "/box.csv");
    ASSERT_FALSE(rows.empty());
    const auto & r = rows.front();
    const double tol = precision_tol<double>(1e-12, 1e-5);
    EXPECT_EQ(r.at("step"), 0.0);
    EXPECT_NEAR(r.at(measure), volume, tol * volume);
    EXPECT_NEAR(r.at("integral_RHO"), integral, tol * integral);
    EXPECT_NEAR(r.at("mean_RHO"), integral / volume, tol * integral / volume);
    EXPECT_NEAR(r.at("min_RHO"), cell_average(x0, y0), tol);
    EXPECT_NEAR(r.at("max_RHO"), cell_average(x1 - 0.05, y1 - 0.05), tol);
    EXPECT_NEAR(r.at("integral_P"), volume, tol * volume);
    EXPECT_NEAR(r.at("min_P"), 1.0, tol);

    const double pi = std::numbers::pi, radius = 0.27;
    const double exact = N_DIM == 2 ? pi * radius * radius : 4.0 / 3.0 * pi * radius * radius * radius;
    const auto ball_rows = read_csv(dir + "/ball.csv");
    ASSERT_FALSE(ball_rows.empty());
    const auto & b = ball_rows.front();
    EXPECT_NEAR(b.at(measure), exact, 2e-5 * exact);
    EXPECT_NEAR(b.at("integral_P"), b.at(measure), tol * exact);
    EXPECT_NEAR(b.at("mean_P"), 1.0, tol);
    EXPECT_NEAR(b.at("rms_P"), 1.0, tol);
}

TEST(ProbesTest, RegionReductionsAreTheSameOnAnyNumberOfRanks) {
    // A smooth-edged ball through a sponge region of partly covered cells:
    // the sums run in global cell order, so the files match bit for bit
    const std::string dir = fresh_dir("region_ranks");
    const std::string extra =
        std::string("[regions]\nball = {type = \"sphere\", center = ") + numbers3("0.45", "0.55", "0.5") +
        ", radius = 0.3, edge = 0.05}\n[[sponges]]\nstrength = 20.0\nu = " + numbers3("0.0", "0.0", "0.0") +
        "\np = 1.2\nT = 1.0\nregion = {type = " + std::string(N_DIM == 2 ? "\"rectangle\"" : "\"box\"") + ", center = " + numbers3("0.7", "0.3", "0.5") +
        ", size = " + numbers3("0.33", "0.41", "0.37") + ", rotate = " +
        (N_DIM == 2 ? "25" : "{axis = [1, 2, 3], angle = 25}") + "}\n";
    auto input = [&](const std::string & file) {
        return box_case(12, SINE_RHO, "type = \"MUSCL\"\n",
                        extra + "[[probes]]\nname = \"ball\"\nfields = [\"RHO\", \"U\", \"P\", \"T\"]\n"
                                "region = \"ball\"\nreductions = [\"integral\", \"mean\", \"favre\", \"rms\", \"min\", "
                                "\"max\"]\nreductions_file = \"" + dir + "/" + file + "\"\n",
                        "n_steps = 4\ncfl = 0.4\n");
    };
    Solver distributed;
    distributed.init(parse_toml(input("distributed.csv")));
    run(distributed);
    Solver serial;
    serial.set_distributed(false);
    serial.init(parse_toml(input("serial.csv")));
    run(serial);
    const std::string a = without_run_metadata(read_all(dir + "/distributed.csv"));
    EXPECT_EQ(a, without_run_metadata(read_all(dir + "/serial.csv"))) << "on " << comm::size() << " ranks";
    EXPECT_EQ(read_csv(dir + "/serial.csv").size(), 5u);
}

#if N_DIM == 2
TEST(ProbesTest, RegionsOfTheMeshAreTheSameOnAnyNumberOfRanks) {
    // A cell zone, the domain inside its closed boundary, and a disk cut from the domain
    const std::string dir = fresh_dir("mesh_region_ranks");
    const std::string mesh = write_temp_shared("mallard_probe_cell_zones.msh", jittered_mixed_mesh(16, true));
    auto input = [&](const std::string & suffix) {
        std::string s = box_case(4, SINE_RHO, "type = \"MUSCL\"\n", "", "n_steps = 2\ncfl = 0.4\n");
        s.replace(s.find("type = \"cartesian\""), s.find("[initialize]") - s.find("type = \"cartesian\""),
                  "type = \"file\"\nfilename = \"" + mesh + "\"\n");
        for (const char * zone : {"left", "right", "bottom", "top"}) {
            s += std::string("[[boundaries]]\nname = \"") + zone + "\"\ntype = \"symmetry\"\n";
        }
        const std::vector<std::pair<std::string, std::string>> probes = {
            {"half", "\"left_half\""},
            {"inside", "{type = \"interior\", of = {type = \"union\", regions = [\"left\", \"right\", \"bottom\", "
                       "\"top\"]}}"},
            {"cut", "{type = \"difference\", regions = [\"domain\", {type = \"disk\", center = [0.4, 0.6], "
                    "radius = 0.25}]}"}};
        for (const auto & [name, region] : probes) {
            s += "[[probes]]\nname = \"" + name + "\"\nfields = [\"RHO\", \"P\"]\nregion = " + region +
                 "\nreductions_file = \"" + dir + "/" + name + "_" + suffix + ".csv\"\n";
        }
        return s;
    };
    Solver distributed;
    distributed.init(parse_toml(input("distributed")));
    run(distributed);
    Solver serial;
    serial.set_distributed(false);
    serial.init(parse_toml(input("serial")));
    run(serial);
    for (const char * name : {"half", "inside", "cut"}) {
        const std::string file = dir + "/" + name;
        EXPECT_EQ(without_run_metadata(read_all(file + "_distributed.csv")), without_run_metadata(read_all(file + "_serial.csv")))
            << name << " on " << comm::size() << " ranks";
    }
    const auto & mesh_s = *serial.get_mesh();
    double left = 0.0;
    for (uint32_t c = 0; c < mesh_s.n_cells; c++) left += mesh_s.h_cell_coords(c, 0) < 0.5_r ? double(mesh_s.h_cell_volume(c)) : 0.0;
    EXPECT_NEAR(read_csv(dir + "/half_serial.csv")[0].at("area"), left, precision_tol<double>(1e-14, 1e-5));
    EXPECT_NEAR(read_csv(dir + "/inside_serial.csv")[0].at("area"), 1.0, precision_tol<double>(1e-14, 1e-5));
}
#endif

TEST(ProbesTest, SurfaceAndCurveRegionProbesWeighCellsByTheirMeasure) {
    // A sphere's surface (2D: a circle) and, in 3D, a circle: their area or length, on any number of ranks
    const std::string dir = fresh_dir("surface_region_ranks");
    const std::string shell = std::string("{type = \"sphere\", center = ") + numbers3("0.45", "0.55", "0.5") +
                              ", radius = 0.3, filled = false}";
    const std::string ring = "{type = \"circle\", center = [0.45, 0.55, 0.5], normal = [1, 1, 0], radius = 0.25}";
    auto input = [&](const std::string & suffix) {
        std::string probes;
        for (const auto & [name, region] : std::vector<std::pair<std::string, std::string>>{{"shell", shell}, {"ring", ring}}) {
            if (name == "ring" && N_DIM == 2) continue;
            probes += "[[probes]]\nname = \"" + name + "\"\nfields = [\"RHO\", \"P\"]\nregion = " + region +
                      "\nreductions_file = \"" + dir + "/" + name + "_" + suffix + ".csv\"\n";
        }
        return box_case(10, SINE_RHO, "type = \"MUSCL\"\n", probes, "n_steps = 2\ncfl = 0.4\n");
    };
    Solver distributed;
    distributed.init(parse_toml(input("distributed")));
    run(distributed);
    Solver serial;
    serial.set_distributed(false);
    serial.init(parse_toml(input("serial")));
    run(serial);
    for (const std::string name : {"shell", "ring"}) {
        if (name == "ring" && N_DIM == 2) continue;
        const std::string file = dir + "/" + name;
        EXPECT_EQ(without_run_metadata(read_all(file + "_distributed.csv")), without_run_metadata(read_all(file + "_serial.csv")))
            << name << " on " << comm::size() << " ranks";
        const auto rows = read_csv(file + "_serial.csv");
        const double exact = name == "ring" ? 2.0 * std::numbers::pi * 0.25
                             : N_DIM == 2   ? 2.0 * std::numbers::pi * 0.3
                                            : 4.0 * std::numbers::pi * 0.09;
        const std::string measure = name == "ring" || N_DIM == 2 ? "length" : "area";
        EXPECT_NEAR(rows[0].at(measure), exact, 1e-9 * exact) << name;
        EXPECT_NEAR(rows[0].at("mean_P"), 1.0, 1e-3) << name;
    }
}

TEST(ProbesTest, RejectsRegionProbesWithSampleKeys) {
    const std::string region = "region = {type = \"sphere\", center = " + numbers3("0.5", "0.5", "0.5") +
                               ", radius = 0.2}\n";
    for (const std::string key : {"sampling = \"point\"\n", "format = \"vtp\"\n", "samples = true\n",
                                  "file = \"x.csv\"\n", "type = \"circle\"\n"}) {
        Solver solver;
        EXPECT_ANY_THROW(solver.init(parse_toml(
            box_case(8, "1.0", "type = \"MUSCL\"\n", "[[probes]]\nname = \"r\"\nfields = [\"P\"]\n" + region + key))))
            << key;
    }
}

TEST(ProbesTest, TimeIntervalsLandOnTheirTimes) {
    const std::string dir = fresh_dir("times");
    const std::string probe = "[[probes]]\nname = \"t\"\nfile = \"" + dir + "/t.csv\"\nfields = [\"P\"]\n"
                              "point = " + numbers3("0.5", "0.5", "0.5") + "\ntime_interval = 0.0123\n";
    Solver solver;
    solver.init(parse_toml(box_case(8, SINE_RHO, "type = \"MUSCL\"\n", probe, "t_stop = 0.05\ncfl = 0.5\n")));
    run(solver);
    const auto rows = read_csv(dir + "/t.csv");
    ASSERT_EQ(rows.size(), 5u);
    for (size_t k = 0; k < rows.size(); k++) EXPECT_NEAR(rows[k].at("t"), 0.0123 * double(k), precision_tol<double>(1e-12, 1e-6));
}

TEST(ProbesTest, VtpAndHdf5HoldTheCsvSamplesAndRestartsCutThemBack) {
    const std::string dir = fresh_dir("formats");
    const std::string geometry = "type = \"circle\"\ncenter = " + numbers3("0.5", "0.5", "0.5") +
                                 "\nradius = 0.3\nn_points = 12\n" + (N_DIM == 3 ? "normal = [1.0, 1.0, 1.0]\n" : "") +
                                 "fields = [\"RHO\", \"U\"]\nsampling = \"point\"\ninterval = 2\n";
    auto probes = [&](const std::string & sub) {
        std::string text = "[[probes]]\nname = \"csv\"\nfile = \"" + dir + "/" + sub + "/c.csv\"\n" + geometry +
                           "[[probes]]\nname = \"vtp\"\nformat = \"vtp\"\nfile = \"" + dir + "/" + sub + "/v.pvd\"\n" +
                           geometry;
#ifdef Mallard_HAS_HDF5
        text += "[[probes]]\nname = \"h5\"\nformat = \"hdf5\"\nfile = \"" + dir + "/" + sub + "/h.h5\"\n" + geometry;
        text += "[[probes]]\nname = \"vk\"\nformat = \"vtkhdf\"\nfile = \"" + dir + "/" + sub + "/k.vtkhdf\"\n" +
                geometry;
#endif
        return text + "[[write_data]]\nprefix = \"" + dir + "/" + sub + "/r\"\nformat = \"restart\"\ninterval = 4\n";
    };
    const std::string recon = "type = \"TENO\"\norder = 3\n";
    const std::string run6 = "n_steps = 6\ncfl = 0.3\n";
    Solver whole;
    whole.init(parse_toml(box_case(10, SINE_RHO, recon, probes("a"), run6)));
    run(whole);
    {
        // Steps 0 to 6 of an interrupted run, continued from step 4: the
        // series lose the rows after step 4 before the continuation adds its own
        Solver first;
        first.init(parse_toml(box_case(10, SINE_RHO, recon, probes("b"), run6)));
        run(first);
    }
    std::string input = box_case(10, SINE_RHO, recon, probes("b"), run6);
    const size_t init = input.find("[initialize]\n") + 13;
    input.replace(init, input.find("[numerics]") - init, "type = \"restart\"\nfile = \"" + dir + "/b/r_000004.restart\"\n");
    Solver second;
    second.init(parse_toml(input));
    run(second);

    const auto csv = read_csv(dir + "/a/c.csv");
    ASSERT_EQ(csv.size(), 4u * 12u);
    const std::string pvd = read_all(dir + "/b/v.pvd");
    size_t n_sets = 0;
    for (size_t at = pvd.find("<DataSet"); at != std::string::npos; at = pvd.find("<DataSet", at + 1)) n_sets++;
    EXPECT_EQ(n_sets, 4u);
    EXPECT_EQ(without_run_metadata(pvd), without_run_metadata(read_all(dir + "/a/v.pvd")));
    EXPECT_EQ(without_run_metadata(read_all(dir + "/b/v_000006.vtp")), without_run_metadata(read_all(dir + "/a/v_000006.vtp")));
    const std::string vtp = read_all(dir + "/a/v_000006.vtp");
    EXPECT_NE(vtp.find("NumberOfPoints=\"12\""), std::string::npos);
    EXPECT_NE(vtp.find("NumberOfLines=\"1\""), std::string::npos);
    EXPECT_NE(vtp.find("Name=\"U_X\""), std::string::npos);
    EXPECT_NE(vtp.find("Name=\"MALLARD_FORMAT_VERSION\" NumberOfTuples=\"1\" format=\"ascii\">" +
                       std::to_string(file_format::PROBE_OUTPUT.version) + "<"),
              std::string::npos);
#ifdef Mallard_HAS_HDF5
    // HDF5 files are read by rank 0 alone (the library locks them)
    for (const char * sub : {"a", "b"}) {
        if (!comm::is_root()) break;
        const std::string h5file = dir + "/" + sub + "/h.h5";
        h5::Handle file(H5Fopen(h5file.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT), H5Fclose, "opening");
        EXPECT_EQ(h5::n_rows(file, "step"), 4u);
        if (h5::n_rows(file, "step") != 4u) continue;
        const auto steps = h5::read_rows<uint64_t>(file, "step", 0, 4, 1, H5P_DEFAULT);
        EXPECT_EQ(steps, (std::vector<uint64_t>{0, 2, 4, 6}));
        const auto rho = h5::read_rows<double>(file, "fields/RHO", 0, 4, 12, H5P_DEFAULT);
        for (size_t k = 0; k < 4; k++) {
            for (size_t p = 0; p < 12; p++) EXPECT_EQ(rtype(rho[k * 12 + p]), rtype(csv[k * 12 + p].at("RHO")));
        }
    }
    // VTKHDF: one PolyData of the curve, its point values at each of the 4 snapshots
    if (comm::is_root()) {
        EXPECT_EQ(h5_all_datasets(dir + "/b/k.vtkhdf"), h5_all_datasets(dir + "/a/k.vtkhdf"));
        const std::string k = dir + "/a/k.vtkhdf";
        EXPECT_EQ(h5_values<int64_t>(k, "VTKHDF/Lines/NumberOfCells"), std::vector<int64_t>{1});
        EXPECT_EQ(h5_values<int64_t>(k, "VTKHDF/NumberOfPoints"), std::vector<int64_t>{12});
        EXPECT_EQ(h5_values<double>(k, "VTKHDF/Steps/Values").size(), 4u);
        EXPECT_EQ(h5_values<int64_t>(k, "VTKHDF/Steps/PointDataOffsets/RHO"), (std::vector<int64_t>{0, 12, 24, 36}));
        const auto rho = h5_values<double>(k, "VTKHDF/PointData/RHO");
        ASSERT_EQ(rho.size(), 48u);
        for (size_t i = 0; i < rho.size(); i++) EXPECT_EQ(rtype(rho[i]), rtype(csv[i].at("RHO")));
    }
    if (comm::is_root()) {
        h5::Handle file(H5Fopen((dir + "/b/h.h5").c_str(), H5F_ACC_RDWR, H5P_DEFAULT), H5Fclose, "opening");
        h5::Handle root(H5Gopen2(file, "/", H5P_DEFAULT), H5Gclose, "opening /");
        EXPECT_EQ(h5::read_int_attribute(root, "version"), file_format::PROBE_OUTPUT.version);
        h5::check(H5Adelete(root, "version"), "H5Adelete");
        h5::write_int_attribute(root, "version", static_cast<int>(file_format::PROBE_OUTPUT.version + 1));
    }
    comm::barrier();
    // Rank 0 refuses the file and every rank stops with it
    try {
        Solver newer;
        newer.init(parse_toml(input));
        ADD_FAILURE() << "a probe file of a newer format version was appended to";
    } catch (const std::runtime_error & e) {
        if (comm::is_root()) {
            EXPECT_NE(std::string(e.what()).find("is from a newer Mallard"), std::string::npos) << e.what();
        }
    }
#endif
}

namespace {

// Plane Couette flow between an isothermal wall at rest (y = 0, T0) and one
// moving at U (y = 1, T1), periodic along x (and z): with constant mu and
// kappa, u = U y and T = T0 + (T1 - T0) y + mu U^2 / (2 kappa) y (1 - y) at
// uniform pressure solve the compressible equations exactly
const double COUETTE_U = 0.5, COUETTE_MU = 0.01, COUETTE_T0 = 1.0, COUETTE_T1 = 1.2;
const double COUETTE_KAPPA = COUETTE_MU * 3.5 / 0.72;  // cp = gamma R / (gamma - 1) with R = 1

std::string couette_case(uint32_t n, const std::string & extra, const std::string & run = "n_steps = 1\ndt = 1.0e-9\n") {
    std::ostringstream T;
    T.precision(17);
    T << COUETTE_T0 << " + " << COUETTE_T1 - COUETTE_T0 << " * y + "
      << COUETTE_MU * COUETTE_U * COUETTE_U / (2.0 * COUETTE_KAPPA) << " * y * (1 - y)";
    std::ostringstream s;
    s << "[run]\n" << run << "[mesh]\ntype = \"cartesian\"\nNx = 4\nNy = " << n << "\nLx = 1.0\nLy = 1.0\n";
    if (N_DIM == 3) s << "Nz = 3\nLz = 1.0\n[[periodic]]\nzones = [\"left\", \"right\"]\n[[periodic]]\nzones = [\"back\", \"front\"]\n";
    else s << "[[periodic]]\nzones = [\"left\", \"right\"]\n";
    s << "[initialize]\ntype = \"analytical\"\nrho = \"1 / (" << T.str() << ")\"\nu = " << vector3("0.5 * y", "0", "0")
      << "\np = \"1.0\"\n"
      << "[[boundaries]]\nname = \"bottom\"\ntype = \"wall_isothermal\"\nT = " << COUETTE_T0 << "\n"
      << "[[boundaries]]\nname = \"top\"\ntype = \"wall_isothermal\"\nT = " << COUETTE_T1 << "\nu = "
      << numbers3("0.5", "0.0", "0.0") << "\n"
      << "[numerics]\nriemann_solver = \"HLLC\"\ntime_integrator = \"SSPRK3\"\n"
      << "[numerics.face_reconstruction]\ntype = \"MUSCL\"\n"
      << "[physics]\ntype = \"navier_stokes\"\ngamma = 1.4\np_ref = 1.0\nT_ref = 1.0\nrho_ref = 1.0\nmu = "
      << COUETTE_MU << "\nPr = 0.72\n[output]\ncheck_interval = 1000000\n" << extra;
    return s.str();
}

TEST(ProbesTest, PointsOutsideTheMeshAreErrorsAndCurvesAreClippedToIt) {
    // Walls at y = 0 and 1
    const std::string dir = fresh_dir("outside");
    auto probe = [&](const std::string & name, const std::string & where) {
        return "[[probes]]\nname = \"" + name + "\"\nfile = \"" + dir + "/" + name +
               ".csv\"\nfields = [\"RHO\"]\nsampling = \"point\"\n" + where + "\n";
    };
    auto error_of = [&](const std::string & probes) {
        try {
            Solver solver;
            solver.init(parse_toml(couette_case(4, probes)));
        } catch (const InputError & e) {
            return std::string(e.what());
        }
        return std::string();
    };
    const std::string point = error_of(probe("p", "type = \"points\"\npoints = [" + numbers3("0.3", "0.5", "0.5") + ", " +
                                                       numbers3("0.3", "1.3", "0.5") + "]"));
    EXPECT_NE(point.find("probes[name = \"p\"]"), std::string::npos) << point;
    EXPECT_NE(point.find("1 of 2 points lie outside the mesh: point 1 (0.3, 1.3"), std::string::npos) << point;
    EXPECT_NE(point.find("0.3 m from boundary zone top"), std::string::npos) << point;
    const std::string line = error_of(probe("l", "start = " + numbers3("0.3", "1.2", "0.5") + "\nend = " +
                                                     numbers3("0.3", "1.5", "0.5") + "\nn_points = 4"));
    EXPECT_NE(line.find("lies entirely outside the mesh"), std::string::npos) << line;

    const std::string probes = "run_log = \"" + dir + "/run.jsonl\"\n" + probe("edge", "point = " + numbers3("0.3", "1.0", "0.5")) +
                               probe("half", "start = " + numbers3("0.3", "0.5", "0.5") + "\nend = " +
                                                 numbers3("0.3", "1.5", "0.5") + "\nn_points = 11");
    Solver solver;
    testing::internal::CaptureStderr();
    solver.init(parse_toml(couette_case(4, probes)));
    const std::string warned = testing::internal::GetCapturedStderr();
    run(solver);
    if (!comm::is_root()) return;
    EXPECT_NE(warned.find("5 of 11 samples (50% of its length) lie outside the mesh"), std::string::npos) << warned;
    std::ifstream log(dir + "/run.jsonl");
    std::string first, second;
    std::getline(log, first);
    std::getline(log, second);
    EXPECT_EQ(json::parse(first)["type"].string, "start");
    EXPECT_EQ(json::parse(second)["type"].string, "warning") << "after its run's start record";
    EXPECT_NE(json::parse(second)["message"].string.find("probes[name = \"half\"]"), std::string::npos) << second;
    EXPECT_EQ(read_csv(dir + "/edge.csv").size(), 2u) << "a point on a wall is in the mesh";
    const auto rows = read_csv(dir + "/half.csv");
    ASSERT_EQ(rows.size(), 12u);
    for (const auto & row : rows) EXPECT_LE(row.at("y"), 1.0);
    const json::Value b = json::parse(output::open(dir + "/half.csv").sampling);
    EXPECT_EQ(b["clipped"]["dropped"].number, 5);
    EXPECT_EQ(b["clipped"]["of"].number, 11);
    EXPECT_NEAR(b["clipped"]["length_fraction"].number, 0.5, 1e-12);
    EXPECT_EQ(b["points"]["n"].number, 6);
}

TEST(ProbesTest, RegionsPartlyOutsideTheDomainRecordTheShareOutside) {
    // A disk (sphere) through the top wall at y = 1, its center 0.1 below it
    const double r = 0.2, h = 0.1, a = std::sqrt(r * r - (r - h) * (r - h)), pi = std::numbers::pi;
    const double own = N_DIM == 2 ? pi * r * r : 4.0 / 3.0 * pi * r * r * r;
    const double outside = N_DIM == 2 ? r * r * std::acos((r - h) / r) - (r - h) * a : pi * h * h * (3.0 * r - h) / 3.0;
    // Its surface in the domain: its own below the wall and the wall inside it
    const double surface = N_DIM == 2 ? 2.0 * r * (pi - std::acos((r - h) / r)) + 2.0 * a
                                      : 4.0 * pi * r * r - 2.0 * pi * r * h + pi * a * a;
    const std::string dir = fresh_dir("region_outside");
    const std::string probes = "[regions]\nball = {type = \"" + std::string(N_DIM == 2 ? "disk" : "sphere") +
                               "\", center = " + numbers3("0.5", "0.9", "0.5") + ", radius = 0.2}\n"
                               "[[probes]]\nname = \"ball\"\nregion = \"ball\"\nfields = [\"RHO\"]\n"
                               "reductions_file = \"" + dir + "/ball.csv\"\n";
    Solver solver;
    testing::internal::CaptureStderr();
    solver.init(parse_toml(couette_case(8, probes)));
    const std::string warned = testing::internal::GetCapturedStderr();
    run(solver);
    if (!comm::is_root()) return;
    const char * volume = N_DIM == 2 ? "area" : "volume";
    const char * area = N_DIM == 2 ? "perimeter" : "surface_area";
    const json::Value b = json::parse(output::open(dir + "/ball.csv").sampling);
    EXPECT_EQ(b["measures"]["region"][volume].number, own);
    // The cell fractions' sampling error
    const double tol = N_DIM == 2 ? 1e-4 : 1e-3;
    EXPECT_NEAR(b["measures"]["in_domain"][volume].number, own - outside, tol * own);
    // The surface: exact patches split 30 (3D: 6) times where the domain's boundary crosses them
    EXPECT_NEAR(b["measures"]["in_domain"][area].number, surface, rounding_or(N_DIM == 2 ? 1e-6 : 1e-4) * surface);
    EXPECT_NEAR(b["clipped"][std::string(volume) + "_fraction"].number, outside / own, tol);
    EXPECT_NE(warned.find("probes[name = \"ball\"].region: "), std::string::npos) << warned;
    EXPECT_NE(warned.find("% of its " + std::string(volume) + " lies outside the domain"), std::string::npos) << warned;
}

const char * const MEASURE = N_DIM == 2 ? "length" : "area";

} // namespace

TEST(ProbesTest, WallShearAndHeatFluxMatchCouetteFlow) {
    // The wall values are the walls'; the shear stress and the heat flux,
    // from the one-sided wall gradients of the viscous flux, converge at first
    // order (the shear stress's only error is u = <rho u> / <rho> of the cells),
    // and so does the diffusive energy flux out of the fluid: the heat it loses
    // to the walls less the work the moving wall does on it
    // (u . tau . n + q), whose two walls balance
    const double q_bottom = -COUETTE_KAPPA * (COUETTE_T1 - COUETTE_T0) - 0.5 * COUETTE_MU * COUETTE_U * COUETTE_U;
    const double q_top = COUETTE_KAPPA * (COUETTE_T1 - COUETTE_T0) - 0.5 * COUETTE_MU * COUETTE_U * COUETTE_U;
    const double tau = COUETTE_MU * COUETTE_U, q_ref = 0.5 * COUETTE_U * COUETTE_U;
    const double e_bottom = -q_bottom, e_top = -(COUETTE_MU * COUETTE_U * COUETTE_U + q_top);
    std::vector<double> e_tau, e_q, e_energy;
    for (uint32_t n : {8u, 16u, 32u}) {
        const std::string dir = fresh_dir("couette");
        const std::string probe = "[[probes]]\nname = \"walls\"\nfile = \"" + dir + "/walls.csv\"\nboundary_type = \"wall\"\n"
                                  "fields = [\"CF\", \"TAU_W\", \"TAU_W_X\", \"Q_W\", \"U_TAU\", \"Y_PLUS\", \"T\", \"U_X\", \"RHO\", "
                                  "\"ENERGY_DIFFUSIVE_FLUX\", \"ENERGY_TOTAL_FLUX\"]\n"
                                  "reference = {rho = 1.0, u = " + numbers3("0.5", "0.0", "0.0") + ", p = 1.0}\n";
        Solver solver;
        solver.init(parse_toml(couette_case(n, probe)));
        run(solver);
        double error_tau = 0.0, error_q = 0.0, error_energy = 0.0;
        size_t n_faces = 0;
        const double tol = precision_tol<double>(1e-12, 1e-5);
        for (const auto & row : read_csv(dir + "/walls.csv")) {
            if (row.at("step") != 0.0) continue;
            n_faces++;
            // The fluid drags the wall at rest along the flow, the moving wall against it
            const bool top = row.at("ny") > 0.0;
            const double sign = top ? -1.0 : 1.0;
            EXPECT_NEAR(row.at("T"), top ? COUETTE_T1 : COUETTE_T0, tol);
            EXPECT_NEAR(row.at("U_X"), top ? COUETTE_U : 0.0, tol);
            EXPECT_GT(sign * row.at("TAU_W_X"), 0.0);
            EXPECT_NEAR(row.at("TAU_W"), std::fabs(row.at("TAU_W_X")), tol * tau);
            EXPECT_NEAR(row.at("CF"), row.at("TAU_W_X") / q_ref, tol * tau / q_ref);
            const double rho_w = row.at("RHO");
            EXPECT_NEAR(rho_w * (top ? COUETTE_T1 : COUETTE_T0), 1.0, 1e-3);
            EXPECT_NEAR(row.at("U_TAU"), std::sqrt(row.at("TAU_W") / rho_w), tol);
            EXPECT_NEAR(row.at("Y_PLUS"), rho_w * row.at("U_TAU") * (0.5 / n) / COUETTE_MU, tol);
            error_tau = std::max(error_tau, std::fabs(row.at("TAU_W") - tau) / tau);
            const double q = top ? q_top : q_bottom;
            error_q = std::max(error_q, std::fabs(row.at("Q_W") - q) / std::fabs(q));
            const double e = top ? e_top : e_bottom;
            error_energy = std::max(error_energy, std::fabs(row.at("ENERGY_DIFFUSIVE_FLUX") - e) / std::fabs(e));
            // No flow through the walls: the total is the diffusive flux
            EXPECT_EQ(row.at("ENERGY_TOTAL_FLUX"), row.at("ENERGY_DIFFUSIVE_FLUX"));
            if (!top) {
                EXPECT_NEAR(row.at("ENERGY_DIFFUSIVE_FLUX"), -row.at("Q_W"), precision_tol<double>(1e-12, 1e-4) * std::fabs(q));
            }
        }
        EXPECT_EQ(n_faces, (N_DIM == 2 ? 2u * 4u : 2u * 12u));
        e_tau.push_back(error_tau);
        e_q.push_back(error_q);
        e_energy.push_back(error_energy);
    }
    EXPECT_LT(e_tau[0], 0.01);
    EXPECT_GT(std::log2(e_tau[1] / e_tau[2]), 0.8) << e_tau[1] << " " << e_tau[2];
    EXPECT_LT(e_q.back(), 0.05);
    EXPECT_GT(std::log2(e_q[1] / e_q[2]), 0.8) << e_q[1] << " " << e_q[2];
    EXPECT_LT(e_energy.back(), 0.05);
    EXPECT_GT(std::log2(e_energy[1] / e_energy[2]), 0.8) << e_energy[1] << " " << e_energy[2];
}

TEST(ProbesTest, BoundaryTractionsIntegrateToTheForces) {
    // Cell sampling takes the forces monitor's wall pressure and gradients;
    // axisymmetric runs integrate over a radian of the revolved wall, as
    // [[forces]]. Two layers of cells and a flow that varies along the walls
    // and in time: on several ranks, wall cells read halo cells updated by
    // the last stage
    for (const bool axisymmetric : N_DIM == 2 ? std::vector<bool>{false, true} : std::vector<bool>{false}) {
        SCOPED_TRACE(axisymmetric ? "axisymmetric" : "planar");
        const std::string dir = fresh_dir(axisymmetric ? "forces_axisymmetric" : "forces");
        const std::string extra = "[[forces]]\nzone = \"top\"\nfile = \"" + dir + "/forces.csv\"\n"
                                  "[[probes]]\nname = \"top\"\nboundary = \"top\"\nsamples = false\n"
                                  "fields = [\"TRACTION_PRESSURE\", \"TRACTION_VISCOUS\"]\nreductions = [\"integral\"]\n"
                                  "reductions_file = \"" + dir + "/top.csv\"\n";
        std::string input = couette_case(2, extra, "n_steps = 4\ncfl = 0.5\n");
        const std::string u = "\"0.5 * y\"";
        input.replace(input.find(u), u.size(), "\"0.5 * y + 0.2 * sin(6.283185307179586 * x)\"");
        if (axisymmetric) {
            // The bottom is the axis
            const std::string bottom = "name = \"bottom\"\ntype = \"wall_isothermal\"\nT = ";
            const size_t at = input.find(bottom);
            input.replace(at, input.find('\n', at + bottom.size()) - at, "name = \"bottom\"\ntype = \"symmetry\"");
            const std::string physics = "[physics]\n";
            input.replace(input.find(physics), physics.size(), physics + "axisymmetric = true\n");
        }
        Solver solver;
        solver.init(parse_toml(input));
        run(solver);
        const auto forces = read_csv(dir + "/forces.csv");
        std::map<double, std::map<std::string, double>> probe;
        for (const auto & row : read_csv(dir + "/top.csv")) probe[row.at("step")] = row;
        ASSERT_EQ(forces.size(), 4u);
        const char * const axes[3] = {"x", "y", "z"};
        // [[forces]] writes 12 significant digits
        const double tol = precision_tol<double>(1e-10, 1e-5);
        for (size_t r = 0; r < forces.size(); r++) {
            const auto & row = probe.at(forces[r].at("step"));
            // A radian of the wall at r = 1 has the area of a unit length
            EXPECT_NEAR(row.at(axisymmetric ? "area" : MEASURE), 1.0, tol);
            for (size_t d = 0; d < N_DIM; d++) {
                const double Fp = forces[r].at(std::string("F") + axes[d] + "_pressure");
                const double Fv = forces[r].at(std::string("F") + axes[d] + "_viscous");
                EXPECT_NEAR(row.at(std::string("integral_TRACTION_PRESSURE_") + "XYZ"[d]), Fp, tol * (1.0 + std::fabs(Fp)));
                EXPECT_NEAR(row.at(std::string("integral_TRACTION_VISCOUS_") + "XYZ"[d]), Fv, tol * 1e-2);
            }
        }
    }
}

TEST(ProbesTest, InletAndOutletFluxesHaveTheBoundaryStates) {
    // Uniform flow from an upt inlet to a p_out outlet: the face states are
    // the prescribed ones, and the normals point out of the domain
    const std::string dir = fresh_dir("fluxes");
    std::ostringstream s;
    s << "[run]\nn_steps = 1\ndt = 1.0e-9\n[mesh]\ntype = \"cartesian\"\nNx = 8\nNy = 4\nLx = 2.0\nLy = 1.0\n";
    if (N_DIM == 3) s << "Nz = 3\nLz = 1.0\n[[periodic]]\nzones = [\"back\", \"front\"]\n";
    s << "[initialize]\ntype = \"constant\"\nu = " << numbers3("0.3", "0", "0") << "\np = 1.0\nT = 1.0\n"
      << "[[boundaries]]\nname = \"left\"\ntype = \"inlet_state\"\nu = " << numbers3("0.3", "0", "0") << "\np = 1.0\nT = 1.0\n"
      << "[[boundaries]]\nname = \"right\"\ntype = \"outlet_pressure\"\np = 1.0\n"
      << "[[boundaries]]\nname = \"bottom\"\ntype = \"symmetry\"\n[[boundaries]]\nname = \"top\"\ntype = \"symmetry\"\n"
      << "[numerics]\nriemann_solver = \"HLLC\"\ntime_integrator = \"SSPRK3\"\n"
      << "[numerics.face_reconstruction]\ntype = \"TENO\"\norder = 3\n" << PERFECT_GAS
      << "[output]\ncheck_interval = 1000000\n";
    for (const char * zone : {"left", "right"}) {
        s << "[[probes]]\nname = \"" << zone << "\"\nboundary = \"" << zone << "\"\nsampling = \"point\"\n"
          << "fields = [\"MASS_FLUX\", \"MOMENTUM_FLUX\", \"ENTHALPY_FLUX\"]\nreductions = [\"integral\"]\n"
          << "reductions_file = \"" << dir << "/" << zone << ".csv\"\n";
    }
    Solver solver;
    solver.init(parse_toml(s.str()));
    run(solver);
    const double area = 1.0, mdot = 0.3 * area, h_t = 3.5 + 0.5 * 0.3 * 0.3;
    const double tol = precision_tol<double>(1e-13, 1e-6);
    for (const auto & [zone, sign] : {std::pair{"left", -1.0}, std::pair{"right", 1.0}}) {
        const auto r = read_csv(dir + "/" + zone + ".csv").front();
        EXPECT_NEAR(r.at(MEASURE), area, tol) << zone;
        EXPECT_NEAR(r.at("integral_MASS_FLUX"), sign * mdot, tol) << zone;
        EXPECT_NEAR(r.at("integral_MOMENTUM_FLUX_X"), sign * (0.3 * mdot + area), tol) << zone;
        EXPECT_NEAR(r.at("integral_MOMENTUM_FLUX_Y"), 0.0, tol) << zone;
        EXPECT_NEAR(r.at("integral_ENTHALPY_FLUX"), sign * mdot * h_t, tol) << zone;
    }
}

#if N_DIM == 2
TEST(ProbesTest, WallPolylinesFollowCurvedWallsWithTheFluidOnTheirLeft) {
    // A steady vortex between curved walls (u = M / r, isentropic pressure):
    // each wall is one closed polyline, counterclockwise around the outer wall
    // and clockwise around the inner one; the point values of the wall
    // pressure beat the cell averages
    const double M = 0.8;
    auto p_exact = [&](double r) { return std::pow(1.0 + 0.2 * M * M * (1.0 - 1.0 / (r * r)), 3.5) / 1.4; };
    const std::string file = write_temp_shared("probes_annulus.msh", annulus_gmsh(4, 48));
    const std::string dir = fresh_dir("annulus");
    std::string probes;
    for (const char * zone : {"inner", "outer"}) {
        for (const char * sampling : {"cell", "point"}) {
            probes += std::string("[[probes]]\nname = \"") + zone + "_" + sampling + "\"\nboundary = \"" + zone +
                      "\"\nsampling = \"" + sampling + "\"\nfields = [\"P\", \"MASS_FLUX\"]\nfile = \"" + dir + "/" +
                      zone + "_" + sampling + ".csv\"\n";
        }
    }
    const std::string input =
        "[run]\nn_steps = 1\ndt = 1.0e-9\n[mesh]\ntype = \"file\"\nfilename = \"" + file + "\"\n"
        "[[mesh.curved]]\nzone = \"inner\"\ntype = \"circle\"\ncenter = [0.0, 0.0]\nradius = 1.0\n"
        "[[mesh.curved]]\nzone = \"outer\"\ntype = \"circle\"\ncenter = [0.0, 0.0]\nradius = 1.384\n"
        "[initialize]\ntype = \"analytical\"\nrho = \"(1 + 0.2 * 0.8^2 * (1 - 1 / (x^2 + y^2)))^2.5\"\n"
        "u = [\"-0.8 * y / (x^2 + y^2)\", \"0.8 * x / (x^2 + y^2)\"]\n"
        "p = \"(1 + 0.2 * 0.8^2 * (1 - 1 / (x^2 + y^2)))^3.5 / 1.4\"\n"
        "[[boundaries]]\nname = \"inner\"\ntype = \"symmetry\"\n[[boundaries]]\nname = \"outer\"\ntype = \"wall_adiabatic\"\n"
        "[numerics]\nriemann_solver = \"HLLC\"\ntime_integrator = \"SSPRK3\"\n"
        "[numerics.face_reconstruction]\ntype = \"TENO\"\norder = 4\ntroubled_threshold = 1.0\n" + PERFECT_GAS +
        "[output]\ncheck_interval = 1000000\n" + probes;
    Solver solver;
    solver.init(parse_toml(input));
    run(solver);
    for (const auto & [zone, r, turn] : {std::tuple{"inner", 1.0, -1.0}, std::tuple{"outer", 1.384, 1.0}}) {
        double error[2] = {0.0, 0.0};
        for (int k = 0; k < 2; k++) {
            std::vector<std::map<std::string, double>> rows;
            for (const auto & row : read_csv(dir + "/" + zone + (k ? "_point" : "_cell") + ".csv")) {
                if (row.at("step") == 0.0) rows.push_back(row);
            }
            ASSERT_EQ(rows.size(), 48u) << zone;
            double total = 0.0;
            for (size_t i = 0; i < rows.size(); i++) {
                const auto & a = rows[i];
                const auto & b = rows[(i + 1) % rows.size()];
                double turn_ab = std::atan2(b.at("y"), b.at("x")) - std::atan2(a.at("y"), a.at("x"));
                turn_ab = std::remainder(turn_ab, 2.0 * std::numbers::pi);
                EXPECT_GT(turn * turn_ab, 0.0) << zone << " face " << i;
                if (i + 1 < rows.size()) {
                    EXPECT_GT(b.at("s"), a.at("s"));
                }
                // The normal points out of the fluid: outward at the outer wall, inward at the inner one
                EXPECT_GT(turn * (a.at("nx") * a.at("x") + a.at("ny") * a.at("y")), 0.0);
                EXPECT_NEAR(a.at("MASS_FLUX"), 0.0, roundoff(1e-12));
                total += a.at("area");
                error[k] = std::max(error[k], std::fabs(a.at("P") - p_exact(r)));
            }
            EXPECT_NEAR(total, 2.0 * std::numbers::pi * r, 1e-3 * r) << zone;
        }
        EXPECT_LT(error[1], 0.2 * error[0]) << zone << ": point " << error[1] << ", cell " << error[0];
    }
}
#endif

#if N_DIM == 2
TEST(ProbesTest, RegionsCutByCurvedWallsMeasureAndBalanceTheirPartInTheDomain) {
    // A disk across the outer wall (r = 1.384) of a solid-body rotation
    // between curved slip walls, a steady flow whose momentum the
    // reconstruction reproduces: its part in the domain, its lens with the
    // wall's circle, and the closed surface of that part, through which no
    // net mass flows. The straight cells leave out the slivers between their
    // faces and the wall, whose share of the area and perimeter is 1e-3, and
    // the surface's arc then ends short of its part on the wall
    const double R = 1.384, rho_r = 0.3, d = 1.384, pi = std::numbers::pi;
    const double alpha = std::acos((d * d + R * R - rho_r * rho_r) / (2.0 * d * R));
    const double beta = std::acos((d * d + rho_r * rho_r - R * R) / (2.0 * d * rho_r));
    const double area = R * R * alpha + rho_r * rho_r * beta -
                        0.5 * std::sqrt((-d + R + rho_r) * (d + R - rho_r) * (d - R + rho_r) * (d + R + rho_r));
    const double perimeter = 2.0 * R * alpha + 2.0 * rho_r * beta, own = pi * rho_r * rho_r;
    const std::string file = write_temp_shared("probes_annulus_measures.msh", annulus_gmsh(4, 48));
    const std::string dir = fresh_dir("annulus_measures");
    auto input = [&](const std::string & u, const std::string & p) {
        return
        "[run]\nn_steps = 1\ndt = 1.0e-9\n[mesh]\ntype = \"file\"\nfilename = \"" + file + "\"\n"
        "[[mesh.curved]]\nzone = \"inner\"\ntype = \"circle\"\ncenter = [0.0, 0.0]\nradius = 1.0\n"
        "[[mesh.curved]]\nzone = \"outer\"\ntype = \"circle\"\ncenter = [0.0, 0.0]\nradius = 1.384\n"
        "[initialize]\ntype = \"analytical\"\nrho = \"1.0\"\nu = " + u + "\np = \"" + p + "\"\n"
        "[[boundaries]]\nname = \"inner\"\ntype = \"symmetry\"\n[[boundaries]]\nname = \"outer\"\ntype = \"symmetry\"\n"
        "[numerics]\nriemann_solver = \"HLLC\"\ntime_integrator = \"SSPRK3\"\n"
        "[numerics.face_reconstruction]\ntype = \"TENO\"\norder = 4\ntroubled_threshold = 1.0\n" + PERFECT_GAS +
        "[output]\ncheck_interval = 1000000\nrun_log = \"" + dir + "/run.jsonl\"\n"
        "[regions]\ndisk = {type = \"disk\", center = [1.384, 0.0], radius = 0.3}\n"
        "[[probes]]\nname = \"disk\"\nregion = \"disk\"\nfields = [\"RHO\"]\nreductions = [\"integral\"]\n"
        "reductions_file = \"" + dir + "/disk.csv\"\n"
        "[[probes]]\nname = \"surface\"\nregion = {type = \"boundary\", of = {type = \"intersection\", "
        "regions = [\"domain\", \"disk\"]}}\nfields = [\"MASS_FLUX\"]\nsampling = \"point\"\n"
        "reductions = [\"integral\"]\nreductions_file = \"" + dir + "/surface.csv\"\n";
    };
    Solver solver;
    testing::internal::CaptureStderr();
    solver.init(parse_toml(input("[\"-0.3 * y\", \"0.3 * x\"]", "1 / 1.4 + 0.045 * (x^2 + y^2)")));
    const std::string warned = testing::internal::GetCapturedStderr();
    run(solver);
    const auto steady = comm::is_root() ? read_csv(dir + "/surface.csv").front() : std::map<std::string, double>{};
    // A uniform expansion, div u = 0.2: the net outflow is 0.2 times the area inside, which a surface around the
    // wrong part, closed or not, misses
    Solver expanding;
    testing::internal::CaptureStderr();
    expanding.init(parse_toml(input("[\"0.1 * x\", \"0.1 * y\"]", "1 / 1.4")));
    testing::internal::GetCapturedStderr();
    run(expanding);
    if (!comm::is_root()) return;
    EXPECT_NEAR(read_csv(dir + "/surface.csv").front().at("integral_MASS_FLUX"), 0.2 * area, rounding_or(1e-8) * area);
    const auto disk = read_csv(dir + "/disk.csv").front();
    const auto & surface = steady;
    EXPECT_NEAR(disk.at("area"), area, 1e-5 * area);
    EXPECT_NEAR(surface.at("length"), perimeter, 1e-5 * perimeter);
    // The mass flow in and out across the disk's diameter
    const double through = 0.3 * (1.384 * 1.384 - 1.084 * 1.084) / 2.0;
    EXPECT_NEAR(surface.at("integral_MASS_FLUX"), 0.0, rounding_or(1e-10) * through);
    const json::Value b = json::parse(output::open(dir + "/disk.csv").sampling);
    EXPECT_EQ(b["measures"]["region"]["area"].number, own);
    EXPECT_EQ(b["measures"]["region"]["perimeter"].number, 2.0 * pi * rho_r);
    EXPECT_EQ(b["measures"]["region"]["tolerance"].number, 0.0);
    EXPECT_NEAR(b["measures"]["in_domain"]["area"].number, area, 1e-5 * area);
    EXPECT_NEAR(b["measures"]["in_domain"]["perimeter"].number, perimeter, 1e-5 * perimeter);
    EXPECT_NEAR(b["clipped"]["area_fraction"].number, 1.0 - area / own, 1e-5);
    EXPECT_EQ(b["units"]["measures.in_domain.perimeter"].string, "m");
    char share[16];
    std::snprintf(share, sizeof(share), "%.3g", 100.0 * (1.0 - area / own));
    EXPECT_NE(warned.find("probes[name = \"disk\"].region: " + std::string(share)), std::string::npos) << warned;
    EXPECT_NE(warned.find("% of its area lies outside the domain, and the probe covers only its part inside"),
              std::string::npos)
        << warned;
    std::ifstream log(dir + "/run.jsonl");
    bool recorded = false;
    for (std::string line; std::getline(log, line);) {
        const json::Value r = json::parse(line);
        if (r["type"].string != "probe") continue;
        recorded = true;
        EXPECT_EQ(r["probe"].string, "disk");
        EXPECT_EQ(r["sampling"]["measures"]["in_domain"]["area"].number, b["measures"]["in_domain"]["area"].number);
    }
    EXPECT_TRUE(recorded);
}
#endif

#ifdef Mallard_HAS_HDF5
TEST(ProbesTest, Hdf5ProbesAndAsynchronousHdf5OutputShareTheLibrary) {
    // The probe writes its file on the main thread while the writer's
    // background thread writes snapshots: HDF5 is not thread-safe
    if (comm::size() > 1 && !h5::PARALLEL) GTEST_SKIP() << "HDF5 output on several ranks needs parallel HDF5";
    const std::string dir = fresh_dir("async");
    const uint64_t n_steps = 40;
    const std::string extra = "[[probes]]\nname = \"h5\"\nformat = \"hdf5\"\nfile = \"" + dir + "/h.h5\"\n"
                              "fields = [\"RHO\", \"U\", \"P\"]\ntype = \"circle\"\ncenter = " +
                              numbers3("0.5", "0.5", "0.5") + "\nradius = 0.3\nn_points = 64\n" +
                              (N_DIM == 3 ? "normal = [0.0, 0.0, 1.0]\n" : "") +
                              "[[write_data]]\nprefix = \"" + dir + "/s\"\nformat = \"xdmf\"\ninterval = 1\n"
                              "fields = [\"RHO\", \"P\"]\n";
    std::string input = box_case(12, SINE_RHO, "type = \"MUSCL\"\n", extra,
                                 "n_steps = " + std::to_string(n_steps) + "\ncfl = 0.3\n");
    input.replace(input.find("check_interval"), 0, "async = true\n");
    Solver solver;
    solver.init(parse_toml(input));
    run(solver);
    if (comm::is_root()) {
        h5::Handle file(H5Fopen((dir + "/h.h5").c_str(), H5F_ACC_RDONLY, H5P_DEFAULT), H5Fclose, "opening");
        const uint64_t n = h5::n_rows(file, "step");
        EXPECT_EQ(n, n_steps + 1);
        const auto steps = h5::read_rows<uint64_t>(file, "step", 0, n, 1, H5P_DEFAULT);
        for (uint64_t k = 0; k < n; k++) EXPECT_EQ(steps[k], k);
    }
}
#endif

TEST(ProbesTest, CutPlaneIntegralsAreExactForPolynomialFields) {
    // The cut of the quadratic field's TENO (order 3) reconstruction, within
    // a box away from the periodic seam, integrates as the field itself
    const char * rho = "2.0 + 0.3 * x - 0.2 * y + 0.25 * x^2 + 0.15 * x * y - 0.1 * y^2";
    auto f = [](double x, double y) { return 2.0 + 0.3 * x - 0.2 * y + 0.25 * x * x + 0.15 * x * y - 0.1 * y * y; };
    const std::string dir = fresh_dir("cut_exact");
    const std::string plane = N_DIM == 2 ? "point = [0.5, 0.47]\nnormal = [0.3, 1.0]\nbounds = [[0.3, 0.3], [0.7, 0.7]]\n"
                                         : "point = [0.5, 0.5, 0.47]\nnormal = [0.0, 0.0, 1.0]\n"
                                           "bounds = [[0.3, 0.32, 0.3], [0.7, 0.68, 0.7]]\n";
    const std::string probe = "[[probes]]\nname = \"cut\"\ntype = \"cut_plane\"\n" + plane +
                              "fields = [\"RHO\", \"MASS_FLUX\"]\nsampling = \"point\"\n"
                              "reductions = [\"integral\", \"favre\"]\nreductions_file = \"" + dir + "/cut.csv\"\n";
    Solver solver;
    solver.init(parse_toml(box_case(20, rho, "type = \"TENO\"\norder = 3\ntroubled_threshold = 1.0\n", probe)));
    run(solver);
    double measure = 0.0, integral = 0.0, square = 0.0, flux = 0.0;
    if constexpr (N_DIM == 2) {
        // The line through (0.5, 0.47) normal to (0.3, 1) leaves the box at x = 0.3 and 0.7
        auto y = [](double x) { return 0.47 - 0.3 * (x - 0.5); };
        const double L = std::hypot(0.4, y(0.7) - y(0.3));
        measure = L;
        integral = L / 0.4 * boole([&](double x) { return f(x, y(x)); }, 0.3, 0.7);
        square = L / 0.4 * boole([&](double x) { return std::pow(f(x, y(x)), 2); }, 0.3, 0.7);
        flux = (0.3 * 0.3 - 0.1) / std::hypot(0.3, 1.0) * integral;  // rho u . n with u = (0.3, -0.1)
    } else {
        measure = 0.4 * 0.36;
        integral = boole([&](double x) { return boole([&](double y) { return f(x, y); }, 0.32, 0.68); }, 0.3, 0.7);
        square = boole([&](double x) { return boole([&](double y) { return std::pow(f(x, y), 2); }, 0.32, 0.68); },
                       0.3, 0.7);
        flux = 0.2 * integral;  // u_z = 0.2
    }
    const auto r = read_csv(dir + "/cut.csv").front();
    const double tol = precision_tol<double>(1e-12, 1e-5);
    EXPECT_NEAR(r.at(MEASURE), measure, tol * measure);
    EXPECT_NEAR(r.at("integral_RHO"), integral, tol * integral);
    EXPECT_NEAR(r.at("favre_RHO"), square / integral, tol * square / integral);
    EXPECT_NEAR(r.at("integral_MASS_FLUX"), flux, tol * std::fabs(integral));
}

TEST(ProbesTest, AxisymmetricRunsKeepPerRadianIntegralsOnBoundaries) {
    if (N_DIM != 2) GTEST_SKIP() << "axisymmetric runs are 2D";
    // Only boundary probes integrate over the revolved surface: cut planes and
    // curves sample, but take no fluxes and (cut planes) no reductions
    auto input = [](const std::string & probe) {
        return "[run]\nn_steps = 1\ndt = 1.0e-9\n[mesh]\ntype = \"cartesian\"\nNx = 4\nNy = 4\nLx = 1.0\nLy = 1.0\n"
               "[[periodic]]\nzones = [\"left\", \"right\"]\n[[boundaries]]\nname = \"bottom\"\ntype = \"symmetry\"\n[[boundaries]]\nname = \"top\"\n"
               "type = \"wall_adiabatic\"\n[initialize]\ntype = \"constant\"\nu = [0.3, 0.0]\np = 1.0\nT = 1.0\n"
               "[numerics]\nriemann_solver = \"HLLC\"\n[numerics.face_reconstruction]\ntype = \"FO\"\n" +
               PERFECT_GAS + "axisymmetric = true\n[output]\ncheck_interval = 1000000\n[[probes]]\nname = \"p\"\n" +
               "file = \"" + fresh_dir("axisymmetric") + "/p.csv\"\n" + probe;
    };
    const std::string cut = "type = \"cut_plane\"\npoint = [0.5, 0.5]\nnormal = [1.0, 0.0]\n";
    const std::string circle = "type = \"circle\"\ncenter = [0.5, 0.5]\nradius = 0.2\nn_points = 16\n";
    for (const std::string & rejected : {cut + "fields = [\"MASS_FLUX\"]\n", circle + "fields = [\"MASS_FLUX\"]\n",
                                         cut + "fields = [\"P\"]\nreductions = [\"integral\"]\n"}) {
        Solver solver;
        EXPECT_THROW(solver.init(parse_toml(input(rejected))), InputError) << rejected;
    }
    Solver solver;
    EXPECT_NO_THROW(solver.init(parse_toml(input(cut + "fields = [\"P\"]\n"))));
}

TEST(ProbesTest, CutPlanesTakeFacesOnTheirPlaneOnce) {
    // A plane through mesh faces cuts each column of cells once: the plane's whole area
    const std::string dir = fresh_dir("cut_faces");
    const std::string probe = "[[probes]]\nname = \"cut\"\ntype = \"cut_plane\"\npoint = " + numbers3("0.5", "0", "0") +
                              "\nnormal = " + numbers3("1", "0", "0") + "\nfields = [\"P\"]\nfile = \"" + dir +
                              "/cut.csv\"\nreductions = [\"integral\"]\nreductions_file = \"" + dir + "/red.csv\"\n";
    Solver solver;
    solver.init(parse_toml(box_case(8, SINE_RHO, "type = \"MUSCL\"\n", probe)));
    run(solver);
    size_t n = 0;
    for (const auto & row : read_csv(dir + "/cut.csv")) n += row.at("step") == 0.0;
    EXPECT_EQ(n, N_DIM == 2 ? 8u : 8u * 4u);
    EXPECT_NEAR(read_csv(dir + "/red.csv").front().at(MEASURE), 1.0, 1e-13);
}

TEST(ProbesTest, ClosedSurfaceFluxesFollowTheDivergence) {
    // The outward flux of a linear velocity through a circle (sphere) is the
    // integral of its divergence over the disk (ball)
    const std::string dir = fresh_dir("divergence");
    const std::string shape = N_DIM == 2 ? "type = \"circle\"\ncenter = [0.5, 0.5]\nradius = 0.2\nn_points = 64\n"
                                         : "region = {type = \"sphere\", center = [0.5, 0.5, 0.5], radius = 0.2, filled = false}\n"
                                           "n_points = [24, 48]\n";
    // The same through the region of that surface, sampled at its elements' quadrature points
    const std::string surface = N_DIM == 2 ? "{type = \"circle\", center = [0.5, 0.5], radius = 0.2}"
                                           : "{type = \"sphere\", center = [0.5, 0.5, 0.5], radius = 0.2, filled = false}";
    const std::string probe = "[[probes]]\nname = \"closed\"\n" + shape +
                              "fields = [\"MASS_FLUX\"]\nsampling = \"point\"\n"
                              "reductions = [\"integral\"]\nreductions_file = \"" + dir + "/closed.csv\"\n"
                              "[[probes]]\nname = \"region\"\nregion = " + surface + "\n"
                              "fields = [\"MASS_FLUX\"]\nsampling = \"point\"\nreductions = [\"integral\"]\n"
                              "reductions_file = \"" + dir + "/region.csv\"\n";
    std::string input = box_case(16, "1.0", "type = \"TENO\"\norder = 3\ntroubled_threshold = 1.0\n", probe);
    const std::string u = "u = " + vector3("0.3", "-0.1", "0.2");
    input.replace(input.find(u), u.size(), "u = " + vector3("0.1 * (x - 0.5)", "0.2 * (y - 0.5)", "-0.05 * (z - 0.5)"));
    Solver solver;
    solver.init(parse_toml(input));
    run(solver);
    const double r = 0.2, pi = std::numbers::pi;
    const double expected = N_DIM == 2 ? 0.3 * pi * r * r : 0.25 * 4.0 / 3.0 * pi * r * r * r;
    for (const char * file : {"/closed.csv", "/region.csv"}) {
        // The region's patches are half a cell wide (here coarser than the probe's 24 x 48 grid)
        const double tol = std::string(file) == "/region.csv" ? 1e-8 : 1e-9;
        EXPECT_NEAR(read_csv(dir + file).front().at("integral_MASS_FLUX"), expected,
                    precision_tol<double>(tol, 1e-5) * expected)
            << file;
    }
}

TEST(ProbesTest, FluxesThroughIsosurfacesFollowTheDivergence) {
    // rho = 1 + r^2: its isosurface rho = 1 + R^2 is the circle (sphere) of radius R, with normals toward
    // increasing rho, outward; the flux of rho u through it, with u linear, is 1 + R^2 times the integral
    // of div u over the disk (ball), and its area and flux converge at second order
    const double R = 0.2, pi = std::numbers::pi, div = N_DIM == 2 ? 0.3 : 0.25;
    const double area = N_DIM == 2 ? 2.0 * pi * R : 4.0 * pi * R * R;
    const double flux = (1.0 + R * R) * div * (N_DIM == 2 ? pi * R * R : 4.0 / 3.0 * pi * R * R * R);
    const std::string value = std::to_string(1.0 + R * R);
    std::vector<double> area_errors, flux_errors;
    for (const uint32_t n : N_DIM == 2 ? std::vector<uint32_t>{12, 24, 48} : std::vector<uint32_t>{12, 24}) {
        const std::string dir = fresh_dir("isosurface");
        std::string probes;
        for (const char * flip : {"false", "true"}) {
            probes += std::string("[[probes]]\nname = \"") + (flip[0] == 't' ? "inward" : "outward") +
                      "\"\nregion = {type = \"isosurface\", field = \"RHO\", value = " + value + ", flip = " + flip +
                      "}\nfields = [\"MASS_FLUX\"]\nsampling = \"point\"\nreductions = [\"integral\"]\n"
                      "reductions_file = \"" + dir + "/" + (flip[0] == 't' ? "inward" : "outward") + ".csv\"\n";
        }
        std::string input = box_case(n, N_DIM == 2 ? "1 + (x - 0.5)^2 + (y - 0.5)^2" : "1 + (x - 0.5)^2 + (y - 0.5)^2 + (z - 0.5)^2",
                                     "type = \"TENO\"\norder = 3\ntroubled_threshold = 1.0\n", probes);
        const std::string u = "u = " + vector3("0.3", "-0.1", "0.2");
        input.replace(input.find(u), u.size(), "u = " + vector3("0.1 * (x - 0.5)", "0.2 * (y - 0.5)", "-0.05 * (z - 0.5)"));
        if (N_DIM == 3) input.replace(input.find("Nz = "), 5 + std::to_string(n / 2).size(), "Nz = " + std::to_string(n));
        Solver solver;
        solver.init(parse_toml(input));
        run(solver);
        const auto out = read_csv(dir + "/outward.csv").front(), in = read_csv(dir + "/inward.csv").front();
        const std::string measure = N_DIM == 2 ? "length" : "area";
        area_errors.push_back(std::fabs(out.at(measure) - area) / area);
        flux_errors.push_back(std::fabs(out.at("integral_MASS_FLUX") - flux) / flux);
        EXPECT_EQ(in.at("integral_MASS_FLUX"), -out.at("integral_MASS_FLUX")) << n;
        EXPECT_EQ(in.at(measure), out.at(measure)) << n;
    }
    for (size_t i = 1; i < area_errors.size(); i++) {
        EXPECT_GT(std::log2(area_errors[i - 1] / area_errors[i]), 1.7) << area_errors[i - 1] << " " << area_errors[i];
        // 3D: the flux error is near 1e-3 already on the coarsest mesh, where other errors than the surface's compete
        if (N_DIM == 2) {
            EXPECT_GT(std::log2(flux_errors[i - 1] / flux_errors[i]), 1.7) << flux_errors[i - 1] << " " << flux_errors[i];
        }
    }
    EXPECT_LT(area_errors.back(), 0.01);
    EXPECT_LT(flux_errors.back(), 2e-3);
}

TEST(ProbesTest, RegionsOfTheSolutionAreTheSameOnAnyNumberOfRanks) {
    // Each cell's level and its gradient are its own and its vertex neighbors', across
    // periodic boundaries too; the sums run in global cell order
    const std::string dir = fresh_dir("solution_ranks");
    auto input = [&](const std::string & suffix) {
        std::string probes;
        const std::vector<std::pair<std::string, std::string>> regions = {
            {"dense", "{type = \"where\", condition = \"RHO > 1.1 and not (x < 0.3)\"}"},
            {"level", "{type = \"isosurface\", field = \"RHO\", value = 1.1}"}};
        for (const auto & [name, region] : regions) {
            probes += "[[probes]]\nname = \"" + name + "\"\nfields = [\"RHO\", \"P\"" +
                      (name == "level" ? ", \"MASS_FLUX\"" : "") + "]\nregion = " +
                      region + "\nreductions = [\"integral\", \"mean\", \"max\"]\nreductions_file = \"" + dir +
                      "/" + name + "_" + suffix + ".csv\"\n";
        }
        return box_case(12, SINE_RHO, "type = \"MUSCL\"\n", probes, "n_steps = 4\ncfl = 0.4\n");
    };
    Solver distributed;
    distributed.init(parse_toml(input("distributed")));
    run(distributed);
    Solver serial;
    serial.set_distributed(false);
    serial.init(parse_toml(input("serial")));
    run(serial);
    for (const char * name : {"dense", "level"}) {
        const std::string file = dir + "/" + name;
        EXPECT_EQ(without_run_metadata(read_all(file + "_distributed.csv")), without_run_metadata(read_all(file + "_serial.csv")))
            << name << " on " << comm::size() << " ranks";
        EXPECT_EQ(read_csv(file + "_serial.csv").size(), 5u) << name;
    }
}

TEST(ProbesTest, RegionsOfVelocityGradientFieldsAreTheSameOnAnyNumberOfRanks) {
    // First order and inviscid: the halo is one layer deep unless a condition reads the velocity gradient,
    // which the levels' fit reads in every vertex neighbor
    const std::string dir = fresh_dir("gradient_ranks");
    auto input = [&](const std::string & suffix) {
        std::string probes;
        const std::vector<std::pair<std::string, std::string>> regions = {
            {"vortical", "{type = \"where\", condition = \"Q_CRITERION > 5 and MACH < 2\"}"},
            {"level", "{type = \"isosurface\", field = \"Q_CRITERION\", value = 5}"}};
        for (const auto & [name, region] : regions) {
            probes += "[[probes]]\nname = \"" + name + "\"\nfields = [\"RHO\", \"P\"]\nregion = " + region +
                      "\nreductions = [\"integral\", \"max\"]\nreductions_file = \"" + dir + "/" + name + "_" +
                      suffix + ".csv\"\n";
        }
        std::string text = box_case(12, SINE_RHO, "type = \"FO\"\n", probes, "n_steps = 2\ncfl = 0.4\n");
        const std::string u = "u = " + vector3("0.3", "-0.1", "0.2");
        text.replace(text.find(u), u.size(),
                     "u = " + vector3("sin(2 * pi * x) * cos(2 * pi * y)", "-cos(2 * pi * x) * sin(2 * pi * y)", "0.2"));
        return text;
    };
    Solver distributed;
    distributed.init(parse_toml(input("distributed")));
    run(distributed);
    Solver serial;
    serial.set_distributed(false);
    serial.init(parse_toml(input("serial")));
    run(serial);
    for (const char * name : {"vortical", "level"}) {
        const std::string file = dir + "/" + name;
        EXPECT_EQ(without_run_metadata(read_all(file + "_distributed.csv")), without_run_metadata(read_all(file + "_serial.csv")))
            << name << " on " << comm::size() << " ranks";
        const auto rows = read_csv(file + "_serial.csv");
        ASSERT_EQ(rows.size(), 3u) << name;
        EXPECT_GT(rows.front().at(name == std::string("level") ? (N_DIM == 2 ? "length" : "area")
                                                                : (N_DIM == 2 ? "area" : "volume")),
                  0.0)
            << name;
    }
}

TEST(ProbesTest, ClosedSurfaceFluxesBalanceTheEnclosedMassMomentumAndEnergy) {
    // A density wave convected through a circle (sphere): the rate of change of
    // the mass, momentum and total energy inside, a region probe's integrals of
    // RHO, RHOU_X and RHOE over the disk (ball), balances the outward fluxes
    // through its boundary, up to an error that falls at second order: the
    // region probe weighs the average of each cell the circle cuts by its
    // covered fraction
    const std::string region = "{type = \"sphere\", center = " + numbers3("0.4", "0.55", "0.45") + ", radius = 0.2}";
    const std::string surface = N_DIM == 2 ? "type = \"circle\"\ncenter = [0.4, 0.55]\nradius = 0.2\nn_points = 64\n"
                                           : "region = {type = \"sphere\", center = [0.4, 0.55, 0.45], radius = 0.2, "
                                             "filled = false}\nn_points = [24, 48]\n";
    const std::string rho = std::string(SINE_RHO) + (N_DIM == 3 ? " * cos(2 * pi * z)" : "");
    const std::vector<std::pair<std::string, std::string>> balances = {
        {"RHO", "MASS_FLUX"}, {"RHOU_X", "MOMENTUM_FLUX_X"}, {"RHOE", "ENTHALPY_FLUX"}};
    auto residuals = [&](uint32_t n) {
        const std::string dir = fresh_dir("balance");
        const std::string probes = "[[probes]]\nname = \"content\"\nregion = " + region +
                                   "\nfields = [\"RHO\", \"RHOU_X\", \"RHOE\"]\nreductions = [\"integral\"]\n"
                                   "reductions_file = \"" + dir + "/content.csv\"\n[[probes]]\nname = \"flux\"\n" +
                                   surface + "fields = [\"MASS_FLUX\", \"MOMENTUM_FLUX_X\", \"ENTHALPY_FLUX\"]\n"
                                   "sampling = \"point\"\nsamples = false\nreductions = [\"integral\"]\n"
                                   "reductions_file = \"" + dir + "/flux.csv\"\n";
        const double dt = 0.25 / n;
        std::ostringstream run_keys;
        run_keys.precision(17);
        run_keys << "n_steps = 2\ndt = " << dt << "\n";
        Solver solver;
        solver.init(parse_toml(box_case(n, rho, "type = \"TENO\"\norder = 3\ntroubled_threshold = 1.0\n", probes,
                                        run_keys.str())));
        run(solver);
        const auto content = read_csv(dir + "/content.csv");
        const auto flux = read_csv(dir + "/flux.csv");
        EXPECT_EQ(content.size(), 3u);
        EXPECT_EQ(flux.size(), 3u);
        std::vector<double> errors(balances.size(), 1.0);
        if (content.size() != 3 || flux.size() != 3) return errors;
        for (size_t b = 0; b < balances.size(); b++) {
            auto q = [&](size_t i) { return content[i].at("integral_" + balances[b].first); };
            auto f = [&](size_t i) { return flux[i].at("integral_" + balances[b].second); };
            // Simpson's rule in time, so that the time error is far below the space error
            const double balance = (q(2) - q(0)) / (2.0 * dt) + (f(0) + 4.0 * f(1) + f(2)) / 6.0;
            errors[b] = std::fabs(balance / f(1));
        }
        return errors;
    };
    const std::vector<uint32_t> meshes = {16, 32, 64};
    std::vector<std::vector<double>> errors;
    for (uint32_t n : meshes) errors.push_back(residuals(n));
    for (size_t b = 0; b < balances.size(); b++) {
        for (size_t i = 1; i < errors.size(); i++) {
            EXPECT_GT(std::log2(errors[i - 1][b] / errors[i][b]), 1.8)
                << balances[b].first << " " << errors[i - 1][b] << " " << errors[i][b];
        }
        EXPECT_LT(errors.back()[b], 0.01) << balances[b].first;
    }
}

#if N_DIM == 3
TEST(ProbesTest, SurfacesHaveTheirAreasAndOutwardNormals) {
    const double pi = std::numbers::pi;
    auto area = [](const probe::Geometry & g) {
        double sum = 0.0;
        for (double w : g.quad_weights) sum += w;
        return sum;
    };
    struct Case {
        std::string toml;
        double area;
        bool flat;    // normals along +z
        int radial;   // else: away from the origin in the first `radial` coordinates
    };
    // Surface regions resampled on an n_points grid (cylinders and cones with their ends)
    const std::vector<Case> cases = {
        {"{type = \"rectangle\", center = [0.1, 0.2, 0.3], normal = [0.0, 0.0, 1.0], size = [0.4, 0.3]}", 0.12, true, 0},
        {"{type = \"disk\", center = [0.0, 0.0, 0.0], normal = [0.0, 0.0, 1.0], radius = 0.5, inner_radius = 0.2}",
         pi * (0.25 - 0.04), true, 0},
        {"{type = \"sphere\", center = [0.0, 0.0, 0.0], radius = 0.5, filled = false}", pi, false, 3},
        {"{type = \"cylinder\", center = [0.0, 0.0, 0.0], axis = [0.0, 0.0, 1.0], radius = 0.3, length = 2.0, "
         "filled = false}",
         2.0 * pi * 0.3 * 2.0 + 2.0 * pi * 0.09, false, 2},
        {"{type = \"cone\", base = [0.0, 0.0, 0.0], axis = [0.0, 0.0, 1.0], length = 1.0, base_radius = 0.5, "
         "top_radius = 0.2, filled = false}",
         pi * 0.7 * std::hypot(1.0, 0.3) + pi * (0.25 + 0.04), false, 2},
    };
    for (const auto & c : cases) {
        const region::Region r = region::read(parse_toml("r = " + c.toml).at("r"), "r");
        const probe::Geometry g = probe::grid_geometry(region::sample_grid(region::surface_patches(r, "r"), 16, 32));
        EXPECT_EQ(g.dim, 2);
        EXPECT_NEAR(area(g), c.area, 1e-7 * c.area) << c.toml;
        ASSERT_EQ(g.normals.size(), g.nodes.size());
        for (size_t i = 0; i < g.nodes.size(); i++) {
            if (c.flat) {
                EXPECT_NEAR(g.normals[i][2], 1.0, 1e-14) << c.toml;
                continue;
            }
            double radial = 0.0;
            for (size_t d = 0; d < size_t(c.radial); d++) radial += g.normals[i][d] * g.nodes[i][d];
            EXPECT_GE(radial, -1e-12) << c.toml << " node " << i;
        }
    }
    // The old probe surface types point to the region
    try {
        probe::make_geometry(parse_toml("type = \"sphere\"\ncenter = [0, 0, 0]\nradius = 1\n"), "p");
        ADD_FAILURE() << "a probe sphere was accepted";
    } catch (const InputError & e) {
        EXPECT_NE(std::string(e.what()).find("region = {type = \"sphere\", ...} with filled = false"), std::string::npos)
            << e.what();
    }
}
#endif

namespace {

/** @brief Rows of a CSV file at a step. */
std::vector<std::map<std::string, double>> rows_at(const std::string & file, double step) {
    std::vector<std::map<std::string, double>> rows;
    for (const auto & row : read_csv(file)) {
        if (row.at("step") == step) rows.push_back(row);
    }
    return rows;
}

/**
 * @brief Runs input("ranks") on every rank and input("serial") on each rank
 *        alone; the files of the two (file + "_ranks", file + "_serial") must agree.
 */
void expect_rank_count_independent(const std::function<std::string(const std::string &)> & input,
                                   const std::vector<std::string> & files) {
    for (const bool distributed : {true, false}) {
        Solver solver;
        if (!distributed) solver.set_distributed(false);
        solver.init(parse_toml(input(distributed ? "ranks" : "serial")));
        run(solver);
    }
    if (!comm::is_root()) return;
    for (const auto & file : files) {
        const std::string a = without_run_metadata(read_all(file + "_serial"));
        EXPECT_FALSE(a.empty()) << file;
        EXPECT_EQ(without_run_metadata(read_all(file + "_ranks")), a) << file;
    }
}

/** @brief Couette flow started impulsively: the fluid at rest between the walls, the top one then moving. */
std::string started_couette(const std::string & extra, uint32_t n_steps) {
    std::string s = couette_case(8, extra, "n_steps = " + std::to_string(n_steps) + "\ndt = 2.0e-3\n");
    const std::string u = "u = " + vector3("0.5 * y", "0", "0");
    s.replace(s.find(u), u.size(), "u = " + vector3("0", "0", "0"));
    return s;
}

/**
 * @brief Expects each row of the averaging probe's file to be the time-weighted
 *        mean, over the [statistics] samples so far (every `interval` steps once
 *        t > t_start, weight t - t_prev), of the rows of the same probe written
 *        every step, column by column (MEAN_<column>, or <reduction>_MEAN_<variable>
 *        for <reduction>_<variable>); key names a row's sample.
 */
void expect_time_averages(const std::string & every_step, const std::string & averaged,
                          const std::vector<std::string> & columns, const std::string & key, uint64_t interval,
                          double t_start) {
    auto mean_of = [](const std::string & c) {
        for (const char * r : {"integral_", "mean_", "max_"}) {
            if (c.starts_with(r)) return r + ("MEAN_" + c.substr(std::string(r).size()));
        }
        return "MEAN_" + c;
    };
    const auto steps = read_csv(every_step);
    const auto means = read_csv(averaged);
    ASSERT_FALSE(means.empty());
    for (const auto & mean : means) {
        double weight = 0.0, t_prev = t_start;
        std::map<std::string, double> sum;
        for (const auto & row : steps) {
            if (row.at("step") > mean.at("step")) break;
            const auto step = static_cast<uint64_t>(row.at("step"));
            if (step % interval != 0 || !(row.at("t") > t_start)) continue;
            if (!key.empty() && row.at(key) != mean.at(key)) continue;
            const double w = row.at("t") - t_prev;
            t_prev = row.at("t");
            weight += w;
            for (const auto & c : columns) sum[c] += w * row.at(c);
        }
        ASSERT_GT(weight, 0.0) << "a row before the first sample";
        for (const auto & c : columns) {
            const double expected = sum[c] / weight;
            const double tol = precision_tol<double>(1e-12, 1e-6) * (std::fabs(expected) + 1e-3);
            EXPECT_NEAR(mean.at(mean_of(c)), expected, tol)
                << c << " at step " << mean.at("step");
        }
    }
}

} // namespace

TEST(ProbesTest, ModeledWallsTakeTheWallModelsStressHeatFluxAndExchangeState) {
    // Couette flow over a wall-modeled bottom wall, from its exact linear
    // profile: the model reads the exchange point at h = 0.3 (u = U h, the
    // exact T there) and, with h+ = 2 inside the viscous sublayer, returns the
    // laminar shear mu U. The probe's tractions and heat flux are the model's,
    // so their integrals are [[forces]]'s modeled viscous force and Q
    const double h = 0.3;
    const std::string dir = fresh_dir("wall_model");
    auto input = [&](const std::string & sub) {
        const std::string extra = "[[forces]]\nzone = \"bottom\"\nfile = \"" + dir + "/forces.csv_" + sub + "\"\n"
                                  "[[probes]]\nname = \"bottom\"\nboundary = \"bottom\"\nfile = \"" + dir + "/bottom.csv_" + sub +
                                  "\"\nfields = [\"TRACTION_VISCOUS\", \"Q_W\", \"TAU_W\", \"U_TAU\", \"WALL_MODEL_U\", "
                                  "\"WALL_MODEL_T\", \"WALL_MODEL_H\"]\nreductions = [\"integral\"]\nreductions_file = \"" + dir +
                                  "/bottom_red.csv_" + sub + "\"\n";
        std::string s = couette_case(8, extra, "n_steps = 3\ncfl = 0.4\n");
        const std::string wall = "name = \"bottom\"\ntype = \"wall_isothermal\"\nT = ";
        const size_t end = s.find('\n', s.find(wall) + wall.size());
        s.insert(end + 1, "wall_model = { model = \"equilibrium\", exchange_distance = 0.3 }\n");
        return s;
    };
    expect_rank_count_independent(input, {dir + "/bottom.csv", dir + "/bottom_red.csv"});
    if (!comm::is_root()) return;
    const double tau = COUETTE_MU * COUETTE_U;
    const double T_h = COUETTE_T0 + (COUETTE_T1 - COUETTE_T0) * h +
                       COUETTE_MU * COUETTE_U * COUETTE_U / (2.0 * COUETTE_KAPPA) * h * (1.0 - h);
    const auto start = rows_at(dir + "/bottom.csv_serial", 0.0);
    ASSERT_EQ(start.size(), N_DIM == 2 ? 4u : 12u);
    const double tol = precision_tol<double>(1e-12, 1e-6);
    for (const auto & row : start) {
        EXPECT_NEAR(row.at("WALL_MODEL_H"), h, tol);
        EXPECT_NEAR(row.at("WALL_MODEL_U"), COUETTE_U * h, 1e-3 * COUETTE_U * h);
        EXPECT_NEAR(row.at("WALL_MODEL_T"), T_h, 1e-3);
        EXPECT_NEAR(row.at("TAU_W"), tau, 0.02 * tau);
        // The fluid drags the wall at rest along +x
        EXPECT_NEAR(row.at("TRACTION_VISCOUS_X"), row.at("TAU_W"), tol * tau);
        EXPECT_NEAR(row.at("U_TAU"), std::sqrt(row.at("TAU_W") * COUETTE_T0), 1e-3 * std::sqrt(tau));
    }
    const auto forces = read_csv(dir + "/forces.csv_serial");
    std::map<double, std::map<std::string, double>> reduced;
    for (const auto & row : read_csv(dir + "/bottom_red.csv_serial")) reduced[row.at("step")] = row;
    ASSERT_EQ(forces.size(), 3u);
    const double tol_sum = precision_tol<double>(1e-11, 1e-5);
    for (const auto & f : forces) {
        const auto & r = reduced.at(f.at("step"));
        // [[forces]] writes 12 significant digits
        for (size_t d = 0; d < N_DIM; d++) {
            const double F = f.at(std::string("F") + "xyz"[d] + "_viscous");
            EXPECT_NEAR(r.at(std::string("integral_TRACTION_VISCOUS_") + "XYZ"[d]), F, tol_sum * tau) << d;
        }
        EXPECT_NEAR(r.at("integral_Q_W"), f.at("Q"), tol_sum * std::fabs(f.at("Q")) + 1e-15);
    }
}

TEST(ProbesTest, SpeciesDiffusiveFluxesAreFicksLawThroughTransmissiveBoundaries) {
    // Two gases of equal molar mass and thermodynamics, at uniform T and p
    // and at rest, with a linear composition: their mixture-averaged fluxes
    // with the correction velocity are Fick's law with the binary coefficient,
    // j_1 = -rho D dY_1/dx, the same through each end (transmissive faces take
    // their image face's flux), and carry no energy at uniform T
    SKIP_IN_SINGLE_PRECISION("the binary coefficient is compared to round-off");
    const std::string mechanism = source_dir() + "/test/data/chemistry/test_transport.yaml";
    const double p = 1000.0, T = 300.0, L = 1e-3, a = 0.2, b = 0.5;  // Y_N2 = a + b x / L
    const std::string dir = fresh_dir("species_diffusion");
    auto input = [&](const std::string & sub) {
        std::ostringstream s;
        s << std::setprecision(17) << "[run]\nn_steps = 1\ndt = 1.0e-12\n[mesh]\ntype = \"cartesian\"\nNx = 8\nNy = 3\n"
          << "Lx = " << L << "\nLy = " << 3 * L / 8 << "\n";
        if (N_DIM == 3) s << "Nz = 3\nLz = " << 3 * L / 8 << "\n[[periodic]]\nzones = [\"back\", \"front\"]\n";
        s << "[[periodic]]\nzones = [\"bottom\", \"top\"]\n"
          << "[[boundaries]]\nname = \"left\"\ntype = \"outlet_extrapolation\"\n"
          << "[[boundaries]]\nname = \"right\"\ntype = \"outlet_extrapolation\"\n"
          << "[initialize]\ntype = \"analytical\"\np = \"" << p << "\"\nT = \"" << T << "\"\nu = " << vector3("0", "0", "0")
          << "\nX = { N2 = \"" << a << " + " << b / L << " * x\", N2B = \"" << 1.0 - a << " - " << b / L << " * x\" }\n"
          << "[physics]\ntype = \"navier_stokes\"\ngas = \"mixture\"\nmechanism = \"" << mechanism
          << "\"\ntransport = \"mixture_averaged\"\n[numerics]\nriemann_solver = \"HLLC\"\n"
          << "[numerics.face_reconstruction]\ntype = \"MUSCL\"\n[output]\ncheck_interval = 1000000\n"
          << "[[probes]]\nname = \"ends\"\nboundary = [\"left\", \"right\"]\nfile = \"" << dir << "/ends.csv_" << sub
          << "\"\nfields = [\"SPECIES_DIFFUSIVE_FLUX_*\", \"SPECIES_TOTAL_FLUX_*\", \"SPECIES_FLUX_*\", "
          << "\"ENERGY_DIFFUSIVE_FLUX\"]\nreductions = [\"integral\"]\nreductions_file = \"" << dir << "/ends_red.csv_"
          << sub << "\"\n";
        return s.str();
    };
    expect_rank_count_independent(input, {dir + "/ends.csv", dir + "/ends_red.csv"});
    if (!comm::is_root()) return;
    const auto mech = chemistry::read_mechanism(mechanism);
    const auto table = chemistry::make_transport_table<Kokkos::HostSpace>(mech, chemistry::TransportModel::MIXTURE_AVERAGED);
    const double rho = p * mech.species[0].molecular_weight / (chemistry::GAS_CONSTANT * T);
    const double j = -rho * table.binary_diffusion(0, 1, T, p) * b / L;  // along +x
    const auto rows = rows_at(dir + "/ends.csv_serial", 0.0);
    ASSERT_EQ(rows.size(), N_DIM == 2 ? 6u : 18u);
    for (const auto & row : rows) {
        const double n_x = row.at("nx");
        EXPECT_NEAR(std::fabs(n_x), 1.0, 1e-14);
        EXPECT_NEAR(row.at("SPECIES_DIFFUSIVE_FLUX_N2"), j * n_x, 1e-9 * std::fabs(j));
        EXPECT_NEAR(row.at("SPECIES_DIFFUSIVE_FLUX_N2B"), -j * n_x, 1e-9 * std::fabs(j));
        EXPECT_NEAR(row.at("SPECIES_FLUX_N2"), 0.0, 1e-12 * std::fabs(j));
        EXPECT_EQ(row.at("SPECIES_TOTAL_FLUX_N2"), row.at("SPECIES_FLUX_N2") + row.at("SPECIES_DIFFUSIVE_FLUX_N2"));
        // h_k j_k sums to h sum_k j_k = 0 (h ~ 3e5 J/kg)
        EXPECT_NEAR(row.at("ENERGY_DIFFUSIVE_FLUX"), 0.0, 1e-9 * std::fabs(j) * 3e5);
    }
    // What leaves through the right end enters through the left
    const auto r = rows_at(dir + "/ends_red.csv_serial", 0.0).at(0);
    EXPECT_NEAR(r.at("integral_SPECIES_DIFFUSIVE_FLUX_N2"), 0.0, 1e-9 * std::fabs(j) * r.at(MEASURE));
}

TEST(ProbesTest, MomentsOfTractionsHaveTheirClosedForm) {
    // A gas at rest in a closed unit box under a linear pressure p = 1 + g . x:
    // each wall face carries its cell's pressure, so each pair of opposite
    // walls sums to (1 - h) times its gradient component, and the moment of
    // the pressure traction about c is (1 - h) (x_mean - c) x g: the closed
    // forms of the force and moment of the box's grad p, x_mean its center,
    // at the walls' one-point rule
    const uint32_t n = 8;
    const double h = 1.0 / n;
    const double g[3] = {0.3, -0.2, 0.1}, c[3] = {0.1, 0.7, 0.4};
    const std::string dir = fresh_dir("moments");
    std::ostringstream s;
    s << std::setprecision(17) << "[run]\nn_steps = 1\ndt = 1.0e-9\n[mesh]\ntype = \"cartesian\"\nNx = " << n
      << "\nNy = " << n << "\nLx = 1.0\nLy = 1.0\n" << (N_DIM == 3 ? "Nz = 8\nLz = 1.0\n" : "")
      << "[initialize]\ntype = \"analytical\"\nrho = \"1.0\"\nu = " << vector3("0", "0", "0") << "\np = \"1 + " << g[0]
      << " * x + " << g[1] << " * y + " << g[2] << " * " << (N_DIM == 3 ? "z" : "0") << "\"\n";
    for (const char * zone : {"left", "right", "bottom", "top", "back", "front"}) {
        if (N_DIM == 2 && (std::string(zone) == "back" || std::string(zone) == "front")) continue;
        s << "[[boundaries]]\nname = \"" << zone << "\"\ntype = \"wall_adiabatic\"\n";
    }
    s << "[numerics]\nriemann_solver = \"HLLC\"\ntime_integrator = \"SSPRK3\"\n[numerics.face_reconstruction]\n"
      << "type = \"MUSCL\"\n" << PERFECT_GAS << "[output]\ncheck_interval = 1000000\n"
      << "[[probes]]\nname = \"box\"\nboundary_type = \"wall\"\nsamples = false\n"
      << "fields = [\"TRACTION_PRESSURE\"]\nreductions = [\"integral\", \"moment\"]\nmoment_center = "
      << numbers3("0.1", "0.7", "0.4") << "\nreductions_file = \"" << dir << "/box.csv\"\n";
    Solver solver;
    solver.init(parse_toml(s.str()));
    run(solver);
    const auto r = rows_at(dir + "/box.csv", 0.0).at(0);
    double x[3] = {0.0, 0.0, 0.0}, F[3] = {0.0, 0.0, 0.0};
    for (size_t d = 0; d < N_DIM; d++) {
        x[d] = 0.5 - c[d];
        F[d] = (1.0 - h) * g[d];
    }
    const double M[3] = {x[1] * F[2] - x[2] * F[1], x[2] * F[0] - x[0] * F[2], x[0] * F[1] - x[1] * F[0]};
    const double tol = precision_tol<double>(1e-13, 1e-5);
    for (size_t d = 0; d < N_DIM; d++) {
        EXPECT_NEAR(r.at(std::string("integral_TRACTION_PRESSURE_") + "XYZ"[d]), F[d], tol) << d;
    }
    for (size_t a = N_DIM == 2 ? 2 : 0; a < 3; a++) {
        EXPECT_NEAR(r.at(std::string("moment_TRACTION_PRESSURE_") + "XYZ"[a]), M[a], tol) << a;
    }
}

TEST(ProbesTest, StatisticsAveragesAreProbeVariablesAtTheirSamplesCells) {
    // Each sample of MEAN_ and COV_ variables is its cell's [statistics] average
    const std::string dir = fresh_dir("statistics_variables");
    const std::string ends = N_DIM == 2 ? "start = [0.05, 0.31]\nend = [0.93, 0.77]\n"
                                        : "start = [0.05, 0.31, 0.2]\nend = [0.93, 0.77, 0.6]\n";
    const std::string extra =
        "[statistics]\nfields = [\"P\", \"U_X\", \"U_Y\"" + std::string(N_DIM == 3 ? ", \"U_Z\"" : "") +
        "]\nproducts = [\"RHO*P\"]\n"
        "[[probes]]\nname = \"line\"\nfile = \"" + dir + "/line.csv\"\n" + ends +
        "n_points = 9\nfields = [\"MEAN_P\", \"MEAN_U\", \"COV_RHO_P\", \"P\"]\ninterval = 5\n"
        "reductions = [\"integral\"]\nreductions_file = \"" + dir + "/line_reductions.csv\"\n"
        "[[probes]]\nname = \"first\"\nfile = \"" + dir + "/first.csv\"\n" + ends +
        "n_points = 5\nfields = [\"P\", \"MEAN_P\"]\ninterval = 1\nsamples = false\n"
        "reductions = [\"integral\"]\nreductions_file = \"" + dir + "/first_reductions.csv\"\n";
    Solver solver;
    solver.set_distributed(false);
    solver.init(parse_toml(box_case(8, SINE_RHO, "type = \"MUSCL\"\n", extra, "n_steps = 5\ndt = 2.0e-3\n")));
    run(solver);
    solver.copy_device_to_host();
    const auto & mesh = *solver.get_mesh();
    const auto & mean = solver.get_statistics().host_means();
    const auto & cov = solver.get_statistics().host_covariances();
    // [statistics] keeps P, U_X, U_Y(, U_Z), then RHO for the product
    const auto rows = rows_at(dir + "/line.csv", 5);
    ASSERT_EQ(rows.size(), 9u);
    for (const auto & row : rows) {
        uint32_t cell = 0;
        double best = std::numeric_limits<double>::infinity();
        for (uint32_t c = 0; c < mesh.n_cells; c++) {
            double d = 0.0;
            for (size_t i = 0; i < N_DIM; i++) d += std::pow(double(mesh.h_cell_coords(c, i)) - row.at(std::string(1, "xyz"[i])), 2);
            if (d < best) best = d, cell = c;
        }
        EXPECT_EQ(static_cast<rtype>(row.at("MEAN_P")), mean(cell, 0));
        EXPECT_EQ(static_cast<rtype>(row.at("MEAN_U_X")), mean(cell, 1));
        EXPECT_EQ(static_cast<rtype>(row.at("MEAN_U_Y")), mean(cell, 2));
        EXPECT_EQ(static_cast<rtype>(row.at("COV_RHO_P")), cov(cell, 0));
    }
    // Reductions take them at their own quadrature points: after the first
    // sample the mean is the sample itself
    const auto first = rows_at(dir + "/first_reductions.csv", 1).at(0);
    EXPECT_EQ(first.at("integral_MEAN_P"), first.at("integral_P"));
}

TEST(ProbesTest, AveragedProbesAreTimeWeightedMeansOfTheirSamples) {
    // An impulsively started Couette flow: the wall stress and heat flux
    // change every step; the averaging probe samples on the [statistics]
    // schedule and writes the running means of the samples and reductions
    const std::string dir = fresh_dir("averaged");
    auto probe = [&](const std::string & name, const std::string & keys) {
        return "[[probes]]\nname = \"" + name + "\"\nfile = \"" + dir + "/" + name + ".csv\"\nboundary_type = \"wall\"\n"
               "fields = [\"CF\", \"TAU_W\", \"Q_W\"]\nreference = {rho = 1.0, u = " + numbers3("0.5", "0.0", "0.0") +
               ", p = 1.0}\nreductions = [\"integral\", \"max\"]\nreductions_file = \"" + dir + "/" + name +
               "_reductions.csv\"\n" + keys;
    };
    const std::string extra = "[statistics]\ninterval = 2\nt_start = 0.005\n" + probe("every", "") +
                              probe("averaged", "average = true\ninterval = 4\n");
    Solver solver;
    solver.init(parse_toml(started_couette(extra, 12)));
    run(solver);
    if (comm::is_root()) {
        expect_time_averages(dir + "/every.csv", dir + "/averaged.csv", {"CF", "TAU_W", "Q_W"}, "face", 2, 0.005);
        expect_time_averages(dir + "/every_reductions.csv", dir + "/averaged_reductions.csv",
                             {"integral_CF", "integral_Q_W"}, "", 2, 0.005);
        // Written once they hold a sample (the first at step 4, t = 0.008): steps 4, 8, 12
        EXPECT_EQ(read_csv(dir + "/averaged_reductions.csv").size(), 3u);
        const auto last = rows_at(dir + "/averaged_reductions.csv", 12).at(0);
        // Reductions of the averaged samples: the extrema of the means
        double hi = -std::numeric_limits<double>::infinity();
        for (const auto & row : rows_at(dir + "/averaged.csv", 12)) hi = std::max(hi, row.at("MEAN_TAU_W"));
        EXPECT_NEAR(last.at("max_MEAN_TAU_W"), hi, precision_tol<double>(0.0, 1e-6));
    }
    comm::barrier();
}

TEST(ProbesTest, AveragedProbesOfModeledWallsAreTheMeansOfTheirSamples) {
    // The averages sample before the probes write, on the state the wall model has not yet seen
    const std::string dir = fresh_dir("averaged_wall_model");
    auto probe = [&](const std::string & name, const std::string & keys) {
        return "[[probes]]\nname = \"" + name + "\"\nfile = \"" + dir + "/" + name + ".csv\"\nboundary = \"bottom\"\n"
               "fields = [\"TAU_W\", \"Q_W\", \"WALL_MODEL_U\"]\n" + keys;
    };
    std::string s = started_couette("[statistics]\ninterval = 2\n" + probe("every", "") +
                                        probe("averaged", "average = true\ninterval = 4\n"),
                                    8);
    const std::string wall = "name = \"bottom\"\ntype = \"wall_isothermal\"\nT = ";
    s.insert(s.find('\n', s.find(wall) + wall.size()) + 1,
             "wall_model = { model = \"equilibrium\", exchange_distance = 0.3 }\n");
    Solver solver;
    solver.init(parse_toml(s));
    run(solver);
    if (comm::is_root()) {
        expect_time_averages(dir + "/every.csv", dir + "/averaged.csv", {"TAU_W", "Q_W", "WALL_MODEL_U"}, "face", 2,
                             0.0);
    }
    comm::barrier();
}

TEST(ProbesTest, AveragedProbeOfMassFluxesThroughACutIsTheMeanOfItsSamples) {
    const std::string dir = fresh_dir("averaged_cut");
    const std::string plane = N_DIM == 2 ? "point = [0.5, 0.47]\nnormal = [0.3, 1.0]\n"
                                         : "point = [0.5, 0.5, 0.47]\nnormal = [0.0, 0.3, 1.0]\n";
    auto probe = [&](const std::string & name, const std::string & keys) {
        return "[[probes]]\nname = \"" + name + "\"\nfile = \"" + dir + "/" + name + ".csv\"\ntype = \"cut_plane\"\n" +
               plane + "fields = [\"MASS_FLUX\", \"P\"]\nreductions = [\"integral\"]\nreductions_file = \"" + dir +
               "/" + name + "_reductions.csv\"\n" + keys;
    };
    const std::string extra = probe("every", "") + probe("averaged", "average = true\ninterval = 3\n");
    Solver solver;
    solver.init(parse_toml(box_case(8, SINE_RHO, "type = \"MUSCL\"\n", extra, "n_steps = 6\ndt = 2.0e-3\n")));
    run(solver);
    // Without [statistics]: every step from t = 0
    if (comm::is_root()) {
        expect_time_averages(dir + "/every.csv", dir + "/averaged.csv", {"MASS_FLUX", "P"}, "cell", 1, 0.0);
        expect_time_averages(dir + "/every_reductions.csv", dir + "/averaged_reductions.csv",
                             {"integral_MASS_FLUX"}, "", 1, 0.0);
    }
    comm::barrier();
}

TEST(ProbesTest, RestartedRunsContinueTheTimeAveragesBitwise) {
    const std::string dir = fresh_dir("averaged_restart");
    auto probe = [&](const std::string & file) {
        return "[[probes]]\nname = \"walls\"\nfile = \"" + dir + "/" + file + "\"\nboundary_type = \"wall\"\n"
               "fields = [\"TAU_W\", \"Q_W\", \"MASS_FLUX\"]\naverage = true\ninterval = 3\n"
               "reductions = [\"integral\"]\nreductions_file = \"" + dir + "/red_" + file + "\"\n";
    };
    const std::string stats = "[statistics]\nt_start = 0.003\n";
    Solver straight;
    straight.init(parse_toml(started_couette(stats + probe("walls.csv") + "[[write_data]]\nprefix = \"" + dir +
                                                 "/r\"\nformat = \"restart\"\ninterval = 5\n",
                                             12)));
    run(straight);
    std::map<std::string, std::vector<std::map<std::string, double>>> straight_rows;
    for (const char * file : {"walls.csv", "red_walls.csv"}) straight_rows[file] = rows_at(dir + "/" + file, 12);
    // The restarted run appends to the files, after their rows of steps 6 to 12
    std::string input = started_couette(stats + probe("walls.csv"), 12);
    const size_t from = input.find("[initialize]\n"), to = input.find("[[boundaries]]");
    input.replace(from, to - from, "[initialize]\ntype = \"restart\"\nfile = \"" + dir + "/r_000005.restart\"\n");
    Solver continued;
    continued.init(parse_toml(input));
    run(continued);
    if (comm::is_root()) {
        for (const auto & [file, a] : straight_rows) {
            ASSERT_FALSE(a.empty());
            const auto both = rows_at(dir + "/" + file, 12);
            ASSERT_EQ(both.size(), 2 * a.size()) << file;
            EXPECT_EQ(std::vector(both.begin() + static_cast<long>(a.size()), both.end()), a) << file;
        }
    }
    comm::barrier();
    // The probe's samples or variables changed: its averages are another's
    input.replace(input.find("\"TAU_W\", "), 9, "");
    Solver other;
    EXPECT_THROW(other.init(parse_toml(input)), InputError);
}

TEST(ProbesTest, RejectsAveragesOfAverages) {
    const std::string extra = "[statistics]\nfields = [\"P\"]\n[[probes]]\nname = \"p\"\ntype = \"point\"\npoint = " +
                              numbers3("0.3", "0.3", "0.3") + "\nfields = [\"MEAN_P\"]\naverage = true\n";
    Solver solver;
    try {
        solver.init(parse_toml(box_case(8, "1.0", "type = \"MUSCL\"\n", extra)));
        FAIL() << "averaged MEAN_P accepted";
    } catch (const InputError & e) {
        EXPECT_NE(std::string(e.what()).find("averaged already"), std::string::npos) << e.what();
    }
}

TEST(ProbeSpectraTest, TonesAtBinCentersHaveTheirPowerInTheirBins) {
    // x = c + A sin(2 pi k0 n / M + phase) + B cos(2 pi k1 n / M): the
    // periodic Hann window spreads a tone at a bin center over three bins,
    // A^2 / (3 df) at k0 and A^2 / (12 df) at k0 +- 1, and the PSD integrates
    // to the variance (A^2 + B^2) / 2; without a window each tone is one bin,
    // A^2 / (2 df). Even and odd (prime) lengths: the mixed-radix and the
    // direct transforms
    const double pi = std::numbers::pi, dt = 0.1, a = 0.7, b = 0.3;
    for (const uint32_t m : {48u, 47u}) {
        const uint32_t k0 = 5, k1 = 13;
        const double df = 1.0 / (m * dt);
        std::vector<double> x(m);
        for (uint32_t n = 0; n < m; n++) {
            x[n] = 2.5 + a * std::sin(2 * pi * k0 * n / m + 0.4) + b * std::cos(2 * pi * k1 * n / m);
        }
        const auto hann = probe_spectra::welch(x, 1, probe_spectra::Window::HANN, dt);
        const auto flat = probe_spectra::welch(x, 1, probe_spectra::Window::RECTANGULAR, dt);
        ASSERT_EQ(hann.size(), m / 2 + 1);
        const double tol = 1e-12;
        double total = 0.0;
        for (uint32_t k = 0; k < hann.size(); k++) {
            total += hann[k] * df;
            const int d0 = std::abs(int(k) - int(k0)), d1 = std::abs(int(k) - int(k1));
            const double expected = d0 == 0 ? a * a / (3 * df) : d0 == 1 ? a * a / (12 * df)
                                  : d1 == 0 ? b * b / (3 * df) : d1 == 1 ? b * b / (12 * df) : 0.0;
            EXPECT_NEAR(hann[k], expected, tol) << "M = " << m << ", k = " << k;
            const double one = k == k0 ? a * a / (2 * df) : k == k1 ? b * b / (2 * df) : 0.0;
            EXPECT_NEAR(flat[k], one, tol) << "M = " << m << ", k = " << k;
        }
        EXPECT_NEAR(total, (a * a + b * b) / 2, tol);
        const probe_spectra::Peak peak = probe_spectra::peak(hann, df);
        EXPECT_EQ(peak.bin, k0);
        EXPECT_NEAR(peak.f, k0 * df, 1e-12);
    }
    // The Nyquist frequency is counted once: (-1)^n has mean square 1
    std::vector<double> alternating(50);
    for (size_t n = 0; n < alternating.size(); n++) alternating[n] = n % 2 ? -1.0 : 1.0;
    const auto nyquist = probe_spectra::welch(alternating, 1, probe_spectra::Window::RECTANGULAR, dt);
    EXPECT_NEAR(nyquist.back() / (50 * dt), 1.0, 1e-12);
}

TEST(ProbeSpectraTest, WelchAveragesHalfOverlappingSegmentsEndingAtTheLastSample) {
    // Parseval: each segment's PSD integrates to its windowed variance, so
    // the Welch PSD integrates to the mean of those of the K segments of
    // M = 2 N / (K + 1) samples starting every M / 2, the last ending at the
    // last sample; a peak between bins is located by the parabola through
    // log P to a fraction of the bin width
    const double dt = 0.05;
    std::vector<double> x(53);
    uint64_t state = 12345;
    for (auto & v : x) {
        state = state * 6364136223846793005ull + 1442695040888963407ull;
        v = static_cast<double>(state >> 11) / 9007199254740992.0;
    }
    for (const auto w : {probe_spectra::Window::HANN, probe_spectra::Window::RECTANGULAR}) {
        const uint32_t segments = 4, m = probe_spectra::segment_length(x.size(), segments);
        ASSERT_EQ(m, 21u);
        const auto psd = probe_spectra::welch(x, segments, w, dt);
        ASSERT_EQ(psd.size(), m / 2 + 1);
        double integral = 0.0;
        for (const double p : psd) integral += p / (m * dt);
        const std::vector<double> win = probe_spectra::window(w, m);
        double power = 0.0;
        for (const double v : win) power += v * v;
        double expected = 0.0;
        for (uint32_t s = 0; s < segments; s++) {
            const size_t first = x.size() - (segments - 1) * (m / 2) - m + s * (m / 2);
            double mean = 0.0;
            for (uint32_t i = 0; i < m; i++) mean += x[first + i] / m;
            for (uint32_t i = 0; i < m; i++) expected += std::pow(win[i] * (x[first + i] - mean), 2) / (power * segments);
        }
        EXPECT_NEAR(integral, expected, 1e-13 * expected);
    }
    const double pi = std::numbers::pi;
    std::vector<double> tone(256);
    for (size_t n = 0; n < tone.size(); n++) tone[n] = std::sin(2 * pi * 0.1234 * double(n));
    const auto psd = probe_spectra::welch(tone, 1, probe_spectra::Window::HANN, 1.0);
    EXPECT_NEAR(probe_spectra::peak(psd, 1.0 / 256).f, 0.1234, 0.05 / 256);
}

namespace {

/** @brief The column of rows at a step, in row order (of the rows whose key column has value, if key is given). */
std::vector<double> column_at(const std::vector<std::map<std::string, double>> & rows, double step,
                              const std::string & name, const std::string & key = "", double value = 0.0) {
    std::vector<double> c;
    for (const auto & row : rows) {
        if (row.at("step") == step && (key.empty() || row.at(key) == value)) c.push_back(row.at(name));
    }
    return c;
}

/** @brief The last n values of a column up to time t (of rtype values: those of a samples file). */
std::vector<double> history_of(const std::vector<std::map<std::string, double>> & rows, const std::string & name,
                               double t, size_t n, bool precision, const std::string & key = "", double value = 0.0) {
    std::vector<double> c;
    for (const auto & row : rows) {
        // Samples files hold rtype values, printed to its precision
        if (row.at("t") <= t && (key.empty() || row.at(key) == value)) {
            c.push_back(precision ? static_cast<double>(static_cast<rtype>(row.at(name))) : row.at(name));
        }
    }
    return std::vector<double>(c.end() - static_cast<long>(std::min(n, c.size())), c.end());
}

/** @brief An entropy wave rho = 1 + 0.1 sin(2 pi x) carried by u_x = 0.3: at a point, a sine of frequency 0.3. */
std::string advected_wave(const std::string & extra, const std::string & t_stop) {
    return box_case(16, "1.0 + 0.1 * sin(2 * pi * x)", "type = \"MUSCL\"\n", extra,
                    "t_stop = " + t_stop + "\ndt = 0.015625\n");
}

} // namespace

TEST(ProbesTest, SpectraOfAnAdvectedWavePeakAtItsFrequency) {
    // Three periods (length 10, bin width 0.1, f = 0.3 at bin 3), sampled every
    // 0.25: the Hann PSD there is A^2 / (3 df) for the amplitude A <= 0.1 the
    // scheme leaves, and the spectra are those of the probe's own samples
    const std::string dir = fresh_dir("spectra");
    const std::string extra =
        "[[probes]]\nname = \"p\"\nfile = \"" + dir + "/p.csv\"\ntype = \"points\"\npoints = [" +
        numbers3("0.5", "0.1", "0.3") + ", " + numbers3("0.25", "0.6", "0.7") + "]\nfields = [\"RHO\", \"P\"]\n"
        "time_interval = 0.25\nreductions = [\"max\"]\nreductions_file = \"" + dir + "/p_reductions.csv\"\n"
        "[probes.spectra]\nfields = [\"RHO\", \"max_RHO\"]\nlength = 10.0\nfile = \"" + dir + "/s.csv\"\n"
        "reductions_file = \"" + dir + "/rs.csv\"\nstrouhal = {length = 2.0, velocity = 0.3}\n";
    Solver solver;
    solver.init(parse_toml(advected_wave(extra, "10.0")));
    run(solver);
    if (comm::is_root()) {
        const auto spectra = read_csv(dir + "/s.csv");
        const double step = 640;
        ASSERT_EQ(column_at(spectra, step, "f").size(), 2u * 21u) << "written once, at the end";
        const auto samples = read_csv(dir + "/p.csv");
        for (const double point : {0.0, 1.0}) {
            const auto f = column_at(spectra, step, "f", "point", point);
            const auto st = column_at(spectra, step, "St", "point", point);
            const auto psd = column_at(spectra, step, "PSD_RHO", "point", point);
            ASSERT_EQ(psd.size(), 21u);
            for (size_t k = 0; k < f.size(); k++) {
                EXPECT_NEAR(f[k], 0.1 * double(k), 1e-12);
                EXPECT_NEAR(st[k], f[k] * 2.0 / 0.3, precision_tol<double>(1e-12, 1e-6));
            }
            const auto peak = probe_spectra::peak(psd, 0.1);
            EXPECT_EQ(peak.bin, 3u);
            EXPECT_NEAR(peak.f, 0.3, 0.01);
            const double amplitude = std::sqrt(3 * 0.1 * psd[3]);
            EXPECT_GT(amplitude, 0.09);
            EXPECT_LT(amplitude, 0.1);
            const auto expected = probe_spectra::welch(history_of(samples, "RHO", 10.0, 40, true, "point", point), 1,
                                                       probe_spectra::Window::HANN, 0.25);
            for (size_t k = 0; k < psd.size(); k++) EXPECT_NEAR(psd[k], expected[k], 1e-12 * expected[3]);
        }
        const auto reduced = column_at(read_csv(dir + "/rs.csv"), step, "PSD_max_RHO");
        const auto expected = probe_spectra::welch(history_of(read_csv(dir + "/p_reductions.csv"), "max_RHO", 10.0, 40, false),
                                                   1, probe_spectra::Window::HANN, 0.25);
        ASSERT_EQ(reduced.size(), expected.size());
        for (size_t k = 0; k < reduced.size(); k++) EXPECT_NEAR(reduced[k], expected[k], 1e-12 * expected[3]);
    }
    comm::barrier();
}

TEST(ProbesTest, RestartedRunsContinueTheSpectraHistoriesBitwise) {
    // The restart file at t = 3.75 holds the sample of that time, which fills
    // the history (length 4); spectra at t = 4, 5 and 6
    const std::string dir = fresh_dir("spectra_restart");
    auto probe = [&](const std::string & name) {
        return "[[probes]]\nname = \"p\"\nfile = \"" + dir + "/" + name + ".csv\"\ntype = \"point\"\npoint = " +
               numbers3("0.5", "0.1", "0.3") + "\nfields = [\"RHO\", \"U_X\"]\ntime_interval = 0.25\n"
               "spectra = {fields = [\"RHO\", \"U_X\"], length = 4.0, segments = 3, time_interval = 1.0, file = \"" +
               dir + "/" + name + "_spectra.csv\"}\n";
    };
    Solver straight;
    straight.init(parse_toml(advected_wave(probe("p") + "[[write_data]]\nprefix = \"" + dir +
                                               "/r\"\nformat = \"restart\"\ninterval = 80\n",
                                           "6.0")));
    run(straight);
    std::vector<std::map<std::string, double>> straight_rows;
    for (const auto & row : read_csv(dir + "/p_spectra.csv")) {
        if (row.at("t") > 3.75) straight_rows.push_back(row);
    }
    std::string input = advected_wave(probe("p"), "6.0");
    const size_t from = input.find("[initialize]\n"), to = input.find("[numerics]");
    input.replace(from, to - from, "[initialize]\ntype = \"restart\"\nfile = \"" + dir + "/r_000240.restart\"\n");
    Solver continued;
    continued.init(parse_toml(input));
    run(continued);
    if (comm::is_root()) {
        // Written at t = 4, 5 and 6 (5 frequencies): the rows after the restart time again
        ASSERT_EQ(straight_rows.size(), 3u * 5u);
        const auto all = read_csv(dir + "/p_spectra.csv");
        ASSERT_EQ(all.size(), 2 * straight_rows.size());
        EXPECT_EQ(std::vector(all.end() - static_cast<long>(straight_rows.size()), all.end()), straight_rows);
    }
    comm::barrier();
    // Another history: other variables
    input.replace(input.find("fields = [\"RHO\", \"U_X\"], length"), 31, "fields = [\"RHO\"], length");
    Solver other;
    EXPECT_THROW(other.init(parse_toml(input)), InputError);
}

TEST(ProbesTest, RejectsSpectraOfUnevenSamplesAndOfUnknownVariables) {
    auto message = [](const std::string & keys) {
        const std::string extra = "[[probes]]\nname = \"p\"\ntype = \"point\"\npoint = " + numbers3("0.3", "0.3", "0.3") +
                                  "\nfields = [\"P\"]\n" + keys;
        Solver solver;
        try {
            solver.init(parse_toml(box_case(8, "1.0", "type = \"MUSCL\"\n", extra)));
        } catch (const InputError & e) {
            return std::string(e.what());
        }
        return std::string("accepted");
    };
    EXPECT_NE(message("interval = 2\nspectra = {fields = [\"P\"], length = 1.0}\n").find("time_interval"),
              std::string::npos);
    EXPECT_NE(message("time_interval = 0.1\nspectra = {fields = [\"RHO\"], length = 1.0}\n").find("List one of: P"),
              std::string::npos);
    EXPECT_NE(message("time_interval = 0.1\nspectra = {fields = [\"P\"], length = 0.2}\n").find("at least 4"),
              std::string::npos);
}

TEST(ProbesTest, TimeAveragesOfMovingRegionsAreTheirMeanIntegralsOverTheirMeanMeasures) {
    // A disk (sphere) sweeping x at 1e5 over four steps of 1e-6, where the
    // flow barely changes: rho = 1 + 0.5 x averages to its value at the mean
    // center, x = 0.55, a constant to itself, and the extremes are those of
    // the cells it covered
    const std::string dir = fresh_dir("region_moving_average");
    const std::string disk = std::string("{type = \"") + (N_DIM == 2 ? "disk" : "sphere") + "\", center = " +
                             (N_DIM == 2 ? "[\"0.3 + 1e5 * t\", 0.5]" : "[\"0.3 + 1e5 * t\", 0.5, 0.5]") +
                             ", radius = 0.15}";
    auto probe = [&](const std::string & name) {
        return "[[probes]]\nname = \"" + name + "\"\nregion = " + disk + "\nfields = [\"RHO\", \"P\"]\n"
               "average = true\ninterval = 4\nreductions = [\"integral\", \"mean\", \"max\"]\n"
               "reductions_file = \"" + dir + "/" + name + ".csv\"\n";
    };
    Solver solver;
    solver.init(parse_toml(box_case(32, "1.0 + 0.5 * x", "type = \"MUSCL\"\n",
                                    "[statistics]\ninterval = 1\n" + probe("disk"), "n_steps = 4\ndt = 1.0e-6\n")));
    run(solver);
    if (comm::is_root()) {
        const auto rows = read_csv(dir + "/disk.csv");
        ASSERT_EQ(rows.size(), 1u);
        const auto & row = rows.at(0);
        const char * volume = N_DIM == 2 ? "area" : "volume";
        EXPECT_NEAR(row.at("mean_MEAN_P"), 1.0, rounding_or(1e-9));
        EXPECT_NEAR(row.at("integral_MEAN_P"), row.at(volume), rounding_or(1e-9) * row.at(volume));
        EXPECT_NEAR(row.at("integral_MEAN_RHO"), row.at("mean_MEAN_RHO") * row.at(volume), 1e-12);
        EXPECT_NEAR(row.at("mean_MEAN_RHO"), 1.0 + 0.5 * 0.55, 2e-3);
        EXPECT_LT(row.at("max_MEAN_RHO"), 1.0 + 0.5 * (0.7 + 0.15) + 1e-2);
        EXPECT_GT(row.at("max_MEAN_RHO"), 1.0 + 0.5 * (0.7 + 0.1));
    }
    comm::barrier();
}

TEST(ProbesTest, MovingRegionsMeasureWhereTheyAreAtEachWriteAndCopiesReportEach) {
    // Two disks (spheres) moving across the periodic boundary at x = 1: each
    // copy's probe measures its part in the domain where it is at each write
    const std::string dir = fresh_dir("region_moving");
    const std::string disks = std::string("{type = \"") + (N_DIM == 2 ? "disk" : "sphere") + "\", center = " +
                              (N_DIM == 2 ? "[\"0.7 + 2 * t\", 0.3]" : "[\"0.7 + 2 * t\", 0.3, 0.5]") +
                              ", radius = 0.15, repeat = {count = 2, translate = " +
                              (N_DIM == 2 ? "[0, 0.4]" : "[0, 0.4, 0]") + "}}";
    const std::string probes = "[[probes]]\nname = \"disks\"\nregion = " + disks + "\nfields = [\"RHO\"]\n"
                               "copies = \"each\"\nreductions_file = \"" + dir + "/disks.csv\"\n";
    Solver solver;
    solver.init(parse_toml(box_case(16, "1.0", "type = \"MUSCL\"\n", probes, "n_steps = 4\ndt = 0.05\n")));
    const Mesh & mesh = *solver.get_mesh();
    const region::MeshRegions automatic(mesh);
    const region::Region array = region::read(parse_toml("r = " + disks).at("r"), "r", nullptr, &automatic);
    run(solver);
    const char * volume = N_DIM == 2 ? "area" : "volume";
    for (uint32_t k = 0; k < 2; k++) {
        const std::string file = dir + "/disks_copy" + std::to_string(k) + ".csv";
        const auto rows = read_csv(file);
        ASSERT_EQ(rows.size(), 5u) << file;
        const region::Region copy = array.copy(k);
        for (const auto & row : rows) {
            const std::vector<double> f = region::cell_fractions(mesh, copy.at(row.at("t")));
            double expected = 0.0;
            for (uint32_t c = 0; c < mesh.n_owned(); c++) expected += f[c] * double(mesh.h_cell_measure(c));
            if (mesh.n_global_cells > 0) expected = comm::allreduce(expected, comm::Op::SUM);
            EXPECT_NEAR(row.at(volume), expected, 1e-13 * expected) << file << " t " << row.at("t");
            EXPECT_NEAR(row.at("mean_RHO"), 1.0, 1e-12) << file;
        }
        // It keeps its shape, so its own measures are those of where it starts
        const json::Value b = json::parse(output::open(file).sampling);
        EXPECT_EQ(b["measures"]["region"][volume].number, region::measures(copy.at(0.0), false).volume);
    }
}

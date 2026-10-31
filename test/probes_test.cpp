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
#include <cmath>
#include <filesystem>
#include <fstream>
#include <map>
#include <numbers>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "comm.h"
#include "file_format.h"
#include "input.h"
#include "probe_geometry.h"
#include "scratch.h"
#include "gmsh_fixtures.h"
#include "solver.h"
#include "test_fixtures.h"
#include "test_utils.h"

#ifdef Mallard_HAS_HDF5
#include "hdf5_output.h"
#include "hdf5_util.h"
#endif

namespace {

const std::string PERFECT_GAS = "[physics]\ntype = \"euler\"\ngamma = 1.4\np_ref = 1.0\nT_ref = 1.0\nrho_ref = 1.0\n";

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
    const std::string probe = "[[probes]]\nname = \"p\"\nfile = \"" + dir + "/p.csv\"\nvariables = [\"RHO\", \"T\"]\n"
                              "sampling = \"" + sampling + "\"\nshape = \"points\"\npoints = [" + points.str() + "]\n";
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

/** @brief Gauss-Legendre nodes and weights of n >= 2 points on [-1, 1]. */
std::vector<std::pair<double, double>> gauss_legendre(int n) {
    std::vector<std::pair<double, double>> rule;
    const double m = n;
    for (int i = 0; i < n; i++) {
        double x = std::cos(std::numbers::pi * (i + 0.75) / (m + 0.5)), dp = 1.0;
        for (int it = 0; it < 50; it++) {
            double p0 = 1.0, p1 = x;
            for (double k = 2.0; k <= m; k += 1.0) {
                const double p2 = ((2.0 * k - 1.0) * x * p1 - (k - 1.0) * p0) / k;
                p0 = p1;
                p1 = p2;
            }
            dp = m * (x * p1 - p0) / (x * x - 1.0);
            x -= p1 / dp;
        }
        rule.push_back({x, 2.0 / ((1.0 - x * x) * dp * dp)});
    }
    return rule;
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
        parse_toml("shape = \"circle\"\ncenter = " + numbers3("0.5", "0.5", "0.5") + "\nradius = 0.25\nn_points = 7\n" +
                   normal),
        "circle");
    EXPECT_NEAR(length(circle), 2.0 * pi * 0.25, 1e-14);
    ASSERT_EQ(circle.nodes.size(), 7u);
    ASSERT_EQ(circle.polylines.size(), 1u);
    EXPECT_EQ(circle.polylines[0].size(), 8u);
    EXPECT_EQ(circle.polylines[0].front(), circle.polylines[0].back());

    const probe::Geometry arc = probe::make_geometry(
        parse_toml("shape = \"arc\"\ncenter = " + numbers3("0", "0", "0") +
                   "\nradius = 2.0\nangles = [90.0, 0.0]\nn_points = 5\n" + normal),
        "arc");
    EXPECT_NEAR(length(arc), pi, 1e-14);
    EXPECT_NEAR(arc.nodes.front()[1], 2.0, 1e-14);
    EXPECT_NEAR(arc.nodes.back()[0], 2.0, 1e-14);
    EXPECT_NEAR(arc.arc_length.back(), pi, 1e-14);

    // x = s, y = s^2 over [0, 1]: (2 sqrt(5) + asinh(2)) / 4
    const probe::Geometry parabola = probe::make_geometry(
        parse_toml(std::string("shape = \"curve\"\nx = \"s\"\ny = \"s^2\"\n") + (N_DIM == 3 ? "z = \"0.5\"\n" : "") +
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
    const std::string probe = "[[probes]]\nname = \"poly\"\nfile = \"" + dir + "/poly.csv\"\nvariables = [\"RHO\", \"P\"]\n"
                              "sampling = \"point\"\nshape = \"polyline\"\nvertices = [" +
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
    const std::string box = N_DIM == 2 ? "{type = \"box\", center = [0.45, 0.45], size = [0.5, 0.3]}"
                                       : "{type = \"box\", center = [0.45, 0.45, 0.5], size = [0.5, 0.3, 0.6]}";
    const std::string ball = "{type = \"sphere\", center = " + numbers3("0.5", "0.5", "0.5") + ", radius = 0.27}";
    const std::string probes =
        "[[probes]]\nname = \"box\"\nvariables = [\"RHO\", \"P\"]\nregion = " + box + "\nreductions_file = \"" + dir +
        "/box.csv\"\n[[probes]]\nname = \"ball\"\nvariables = [\"P\"]\nregion = " + ball +
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
        "\np = 1.2\nT = 1.0\nregion = {type = \"box\", center = " + numbers3("0.7", "0.3", "0.5") +
        ", size = " + numbers3("0.33", "0.41", "0.37") + ", rotate = " +
        (N_DIM == 2 ? "25" : "{axis = [1, 2, 3], angle = 25}") + "}\n";
    auto input = [&](const std::string & file) {
        return box_case(12, SINE_RHO, "type = \"MUSCL\"\n",
                        extra + "[[probes]]\nname = \"ball\"\nvariables = [\"RHO\", \"U\", \"P\", \"T\"]\n"
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
    const std::string a = without_provenance(read_all(dir + "/distributed.csv"));
    EXPECT_EQ(a, without_provenance(read_all(dir + "/serial.csv"))) << "on " << comm::size() << " ranks";
    EXPECT_EQ(read_csv(dir + "/serial.csv").size(), 5u);
}

TEST(ProbesTest, RejectsRegionProbesWithSampleKeys) {
    const std::string region = "region = {type = \"sphere\", center = " + numbers3("0.5", "0.5", "0.5") +
                               ", radius = 0.2}\n";
    for (const std::string key : {"sampling = \"point\"\n", "format = \"vtp\"\n", "samples = true\n",
                                  "file = \"x.csv\"\n", "shape = \"circle\"\n"}) {
        Solver solver;
        EXPECT_ANY_THROW(solver.init(parse_toml(
            box_case(8, "1.0", "type = \"MUSCL\"\n", "[[probes]]\nname = \"r\"\nvariables = [\"P\"]\n" + region + key))))
            << key;
    }
}

TEST(ProbesTest, TimeIntervalsLandOnTheirTimes) {
    const std::string dir = fresh_dir("times");
    const std::string probe = "[[probes]]\nname = \"t\"\nfile = \"" + dir + "/t.csv\"\nvariables = [\"P\"]\n"
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
    const std::string geometry = "shape = \"circle\"\ncenter = " + numbers3("0.5", "0.5", "0.5") +
                                 "\nradius = 0.3\nn_points = 12\n" + (N_DIM == 3 ? "normal = [1.0, 1.0, 1.0]\n" : "") +
                                 "variables = [\"RHO\", \"U\"]\nsampling = \"point\"\ninterval = 2\n";
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
    EXPECT_EQ(without_provenance(pvd), without_provenance(read_all(dir + "/a/v.pvd")));
    EXPECT_EQ(without_provenance(read_all(dir + "/b/v_000006.vtp")), without_provenance(read_all(dir + "/a/v_000006.vtp")));
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

const char * const MEASURE = N_DIM == 2 ? "length" : "area";

} // namespace

TEST(ProbesTest, WallShearAndHeatFluxMatchCouetteFlow) {
    // The wall values are the walls'; the shear stress and the heat flux,
    // from the one-sided wall gradients of the viscous flux, converge at first
    // order (the shear stress's only error is u = <rho u> / <rho> of the cells)
    const double q_bottom = -COUETTE_KAPPA * (COUETTE_T1 - COUETTE_T0) - 0.5 * COUETTE_MU * COUETTE_U * COUETTE_U;
    const double q_top = COUETTE_KAPPA * (COUETTE_T1 - COUETTE_T0) - 0.5 * COUETTE_MU * COUETTE_U * COUETTE_U;
    const double tau = COUETTE_MU * COUETTE_U, q_ref = 0.5 * COUETTE_U * COUETTE_U;
    std::vector<double> e_tau, e_q;
    for (uint32_t n : {8u, 16u, 32u}) {
        const std::string dir = fresh_dir("couette");
        const std::string probe = "[[probes]]\nname = \"walls\"\nfile = \"" + dir + "/walls.csv\"\nboundary_type = \"wall\"\n"
                                  "variables = [\"CF\", \"TAU_W\", \"TAU_W_X\", \"Q_W\", \"U_TAU\", \"Y_PLUS\", \"T\", \"U_X\", \"RHO\"]\n"
                                  "reference = {rho = 1.0, u = " + numbers3("0.5", "0.0", "0.0") + ", p = 1.0}\n";
        Solver solver;
        solver.init(parse_toml(couette_case(n, probe)));
        run(solver);
        double error_tau = 0.0, error_q = 0.0;
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
        }
        EXPECT_EQ(n_faces, (N_DIM == 2 ? 2u * 4u : 2u * 12u));
        e_tau.push_back(error_tau);
        e_q.push_back(error_q);
    }
    EXPECT_LT(e_tau[0], 0.01);
    EXPECT_GT(std::log2(e_tau[1] / e_tau[2]), 0.8) << e_tau[1] << " " << e_tau[2];
    EXPECT_LT(e_q.back(), 0.05);
    EXPECT_GT(std::log2(e_q[1] / e_q[2]), 0.8) << e_q[1] << " " << e_q[2];
}

TEST(ProbesTest, BoundaryTractionsIntegrateToTheForces) {
    // Cell sampling takes the forces monitor's wall pressure and gradients;
    // axisymmetric runs integrate over a radian of the revolved wall, as [[forces]]
    for (const bool axisymmetric : N_DIM == 2 ? std::vector<bool>{false, true} : std::vector<bool>{false}) {
        SCOPED_TRACE(axisymmetric ? "axisymmetric" : "planar");
        const std::string dir = fresh_dir(axisymmetric ? "forces_axisymmetric" : "forces");
        const std::string extra = "[[forces]]\nzone = \"top\"\nfile = \"" + dir + "/forces.csv\"\n"
                                  "[[probes]]\nname = \"top\"\nboundary = \"top\"\nsamples = false\n"
                                  "variables = [\"TRACTION_PRESSURE\", \"TRACTION_VISCOUS\"]\nreductions = [\"integral\"]\n"
                                  "reductions_file = \"" + dir + "/top.csv\"\n";
        std::string input = couette_case(8, extra, "n_steps = 4\ncfl = 0.5\n");
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
      << "[[boundaries]]\nname = \"left\"\ntype = \"upt\"\nu = " << numbers3("0.3", "0", "0") << "\np = 1.0\nT = 1.0\n"
      << "[[boundaries]]\nname = \"right\"\ntype = \"p_out\"\np = 1.0\n"
      << "[[boundaries]]\nname = \"bottom\"\ntype = \"symmetry\"\n[[boundaries]]\nname = \"top\"\ntype = \"symmetry\"\n"
      << "[numerics]\nriemann_solver = \"HLLC\"\ntime_integrator = \"SSPRK3\"\n"
      << "[numerics.face_reconstruction]\ntype = \"TENO\"\norder = 3\n" << PERFECT_GAS
      << "[output]\ncheck_interval = 1000000\n";
    for (const char * zone : {"left", "right"}) {
        s << "[[probes]]\nname = \"" << zone << "\"\nboundary = \"" << zone << "\"\nsampling = \"point\"\nsamples = false\n"
          << "variables = [\"MASS_FLUX\", \"MOMENTUM_FLUX\", \"ENTHALPY_FLUX\"]\nreductions = [\"integral\"]\n"
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
                      "\"\nsampling = \"" + sampling + "\"\nvariables = [\"P\", \"MASS_FLUX\"]\nfile = \"" + dir + "/" +
                      zone + "_" + sampling + ".csv\"\n";
        }
    }
    const std::string input =
        "[run]\nn_steps = 1\ndt = 1.0e-9\n[mesh]\ntype = \"file\"\nfilename = \"" + file + "\"\n"
        "[[mesh.curved]]\nzone = \"inner\"\nshape = \"circle\"\ncenter = [0.0, 0.0]\nradius = 1.0\n"
        "[[mesh.curved]]\nzone = \"outer\"\nshape = \"circle\"\ncenter = [0.0, 0.0]\nradius = 1.384\n"
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

#ifdef Mallard_HAS_HDF5
TEST(ProbesTest, Hdf5ProbesAndAsynchronousHdf5OutputShareTheLibrary) {
    // The probe writes its file on the main thread while the writer's
    // background thread writes snapshots: HDF5 is not thread-safe
    if (comm::size() > 1 && !h5::PARALLEL) GTEST_SKIP() << "HDF5 output on several ranks needs parallel HDF5";
    const std::string dir = fresh_dir("async");
    const uint64_t n_steps = 40;
    const std::string extra = "[[probes]]\nname = \"h5\"\nformat = \"hdf5\"\nfile = \"" + dir + "/h.h5\"\n"
                              "variables = [\"RHO\", \"U\", \"P\"]\nshape = \"circle\"\ncenter = " +
                              numbers3("0.5", "0.5", "0.5") + "\nradius = 0.3\nn_points = 64\n" +
                              (N_DIM == 3 ? "normal = [0.0, 0.0, 1.0]\n" : "") +
                              "[[write_data]]\nprefix = \"" + dir + "/s\"\nformat = \"hdf5\"\ninterval = 1\n"
                              "variables = [\"RHO\", \"P\"]\n";
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
    const std::string probe = "[[probes]]\nname = \"cut\"\nshape = \"cut_plane\"\n" + plane +
                              "variables = [\"RHO\", \"MASS_FLUX\"]\nsampling = \"point\"\nsamples = false\n"
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
    const std::string cut = "shape = \"cut_plane\"\npoint = [0.5, 0.5]\nnormal = [1.0, 0.0]\n";
    const std::string circle = "shape = \"circle\"\ncenter = [0.5, 0.5]\nradius = 0.2\nn_points = 16\n";
    for (const std::string & rejected : {cut + "variables = [\"MASS_FLUX\"]\n", circle + "variables = [\"MASS_FLUX\"]\n",
                                         cut + "variables = [\"P\"]\nreductions = [\"integral\"]\n"}) {
        Solver solver;
        EXPECT_THROW(solver.init(parse_toml(input(rejected))), InputError) << rejected;
    }
    Solver solver;
    EXPECT_NO_THROW(solver.init(parse_toml(input(cut + "variables = [\"P\"]\n"))));
}

TEST(ProbesTest, CutPlanesTakeFacesOnTheirPlaneOnce) {
    // A plane through mesh faces cuts each column of cells once: the plane's whole area
    const std::string dir = fresh_dir("cut_faces");
    const std::string probe = "[[probes]]\nname = \"cut\"\nshape = \"cut_plane\"\npoint = " + numbers3("0.5", "0", "0") +
                              "\nnormal = " + numbers3("1", "0", "0") + "\nvariables = [\"P\"]\nfile = \"" + dir +
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
    const std::string shape = N_DIM == 2 ? "shape = \"circle\"\ncenter = [0.5, 0.5]\nradius = 0.2\nn_points = 64\n"
                                         : "shape = \"sphere\"\ncenter = [0.5, 0.5, 0.5]\nradius = 0.2\nn_points = [24, 48]\n";
    const std::string probe = "[[probes]]\nname = \"closed\"\n" + shape +
                              "variables = [\"MASS_FLUX\"]\nsampling = \"point\"\nsamples = false\n"
                              "reductions = [\"integral\"]\nreductions_file = \"" + dir + "/closed.csv\"\n";
    std::string input = box_case(16, "1.0", "type = \"TENO\"\norder = 3\ntroubled_threshold = 1.0\n", probe);
    const std::string u = "u = " + vector3("0.3", "-0.1", "0.2");
    input.replace(input.find(u), u.size(), "u = " + vector3("0.1 * (x - 0.5)", "0.2 * (y - 0.5)", "-0.05 * (z - 0.5)"));
    Solver solver;
    solver.init(parse_toml(input));
    run(solver);
    const double r = 0.2, pi = std::numbers::pi;
    const double expected = N_DIM == 2 ? 0.3 * pi * r * r : 0.25 * 4.0 / 3.0 * pi * r * r * r;
    EXPECT_NEAR(read_csv(dir + "/closed.csv").front().at("integral_MASS_FLUX"), expected,
                precision_tol<double>(1e-9, 1e-5) * expected);
}

TEST(ProbesTest, ClosedSurfaceFluxesBalanceTheEnclosedMass) {
    // A density wave convected through a circle (sphere): the rate of change of
    // the mass inside, from point values at Gauss points of the disk (ball),
    // balances the outward mass flux through its boundary, up to an error that
    // falls at the scheme's third order
    const double R = 0.2, pi = std::numbers::pi;
    const double c[3] = {0.4, 0.55, 0.45};
    std::ostringstream points;
    points.precision(17);
    std::vector<double> weights;
    const auto radial = gauss_legendre(8), polar = gauss_legendre(8);
    const int n_around = 16;
    for (const auto & [xr, wr] : radial) {
        const double r = R * (1.0 + xr) / 2.0;
        for (int k = 0; k < n_around; k++) {
            const double phi = 2.0 * pi * k / n_around, w = R / 2.0 * wr * 2.0 * pi / n_around;
            if constexpr (N_DIM == 2) {
                points << (weights.empty() ? "" : ", ") << "[" << c[0] + r * std::cos(phi) << ", "
                       << c[1] + r * std::sin(phi) << "]";
                weights.push_back(w * r);
            } else {
                for (const auto & [mu, wmu] : polar) {
                    const double s = std::sqrt(1.0 - mu * mu);
                    points << (weights.empty() ? "" : ", ") << "[" << c[0] + r * s * std::cos(phi) << ", "
                           << c[1] + r * s * std::sin(phi) << ", " << c[2] + r * mu << "]";
                    weights.push_back(w * r * r * wmu);
                }
            }
        }
    }
    const std::string surface = N_DIM == 2 ? "shape = \"circle\"\ncenter = [0.4, 0.55]\nradius = 0.2\nn_points = 64\n"
                                           : "shape = \"sphere\"\ncenter = [0.4, 0.55, 0.45]\nradius = 0.2\n"
                                             "n_points = [24, 48]\n";
    const std::string rho = std::string(SINE_RHO) + (N_DIM == 3 ? " * cos(2 * pi * z)" : "");
    auto residual = [&](uint32_t n) {
        const std::string dir = fresh_dir("balance");
        const std::string probes = "[[probes]]\nname = \"mass\"\nshape = \"points\"\npoints = [" + points.str() +
                                   "]\nvariables = [\"RHO\"]\nsampling = \"point\"\nfile = \"" + dir + "/mass.csv\"\n"
                                   "[[probes]]\nname = \"flux\"\n" + surface +
                                   "variables = [\"MASS_FLUX\"]\nsampling = \"point\"\nsamples = false\n"
                                   "reductions = [\"integral\"]\nreductions_file = \"" + dir + "/flux.csv\"\n";
        const double dt = 0.25 / n;
        std::ostringstream run_keys;
        run_keys.precision(17);
        run_keys << "n_steps = 2\ndt = " << dt << "\n";
        Solver solver;
        solver.init(parse_toml(box_case(n, rho, "type = \"TENO\"\norder = 3\ntroubled_threshold = 1.0\n", probes,
                                        run_keys.str())));
        run(solver);
        std::vector<double> mass(3, 0.0), flux;
        for (const auto & row : read_csv(dir + "/mass.csv")) {
            const double w = weights.at(static_cast<size_t>(row.at("point")));
            mass.at(static_cast<size_t>(row.at("step"))) += w * row.at("RHO");
        }
        for (const auto & row : read_csv(dir + "/flux.csv")) flux.push_back(row.at("integral_MASS_FLUX"));
        EXPECT_EQ(flux.size(), 3u);
        if (flux.size() != 3) return 1.0;
        // Simpson's rule in time, so that the time error is far below the space error
        const double balance = (mass[2] - mass[0]) / (2.0 * dt) + (flux[0] + 4.0 * flux[1] + flux[2]) / 6.0;
        return std::fabs(balance / flux[1]);
    };
    // Single precision: meshes whose residuals stay above its round-off
    const bool single = sizeof(rtype) == sizeof(float);
    const std::vector<uint32_t> meshes = N_DIM == 3 || single ? std::vector<uint32_t>{16, 32}
                                                              : std::vector<uint32_t>{16, 32, 64};
    std::vector<double> errors;
    for (uint32_t n : meshes) errors.push_back(residual(n));
    for (size_t i = 1; i < errors.size(); i++) {
        EXPECT_GT(std::log2(errors[i - 1] / errors[i]), 2.5) << errors[i - 1] << " " << errors[i];
    }
    EXPECT_LT(errors.back(), 0.01);
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
    const std::vector<Case> cases = {
        {"shape = \"rectangle\"\ncenter = [0.1, 0.2, 0.3]\nnormal = [0.0, 0.0, 1.0]\nsize = [0.4, 0.3]\nn_points = [5, 4]\n",
         0.12, true, 0},
        {"shape = \"disk\"\ncenter = [0.0, 0.0, 0.0]\nnormal = [0.0, 0.0, 1.0]\nradius = 0.5\ninner_radius = 0.2\n"
         "n_points = [6, 32]\n",
         pi * (0.25 - 0.04), true, 0},
        {"shape = \"sphere\"\ncenter = [0.0, 0.0, 0.0]\nradius = 0.5\nn_points = [16, 32]\n", pi, false, 3},
        {"shape = \"cylinder\"\ncenter = [0.0, 0.0, 0.0]\naxis = [0.0, 0.0, 1.0]\nradius = 0.3\nlength = 2.0\n"
         "n_points = [5, 32]\n",
         2.0 * pi * 0.3 * 2.0, false, 2},
        {"shape = \"cone\"\nbase = [0.0, 0.0, 0.0]\naxis = [0.0, 0.0, 1.0]\nlength = 1.0\nbase_radius = 0.5\n"
         "top_radius = 0.2\nn_points = [5, 32]\n",
         pi * 0.7 * std::hypot(1.0, 0.3), false, 2},
    };
    for (const auto & c : cases) {
        const probe::Geometry g = probe::make_geometry(parse_toml(c.toml), "surface");
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
}
#endif

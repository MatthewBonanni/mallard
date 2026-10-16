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
#include <vector>

#include "input.h"
#include "probe_geometry.h"
#include "solver.h"
#include "gmsh_fixtures.h"
#include "test_fixtures.h"

#ifdef Mallard_HAS_HDF5
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

std::string dir_for(const std::string & name) {
    return (std::filesystem::temp_directory_path() / ("mallard_probes_" + name)).string();
}

/** @brief A periodic box of n cells per side (n / 2 along z) with the given initial fields and reconstruction. */
std::string box_case(uint32_t n, const std::string & rho, const std::string & recon, const std::string & extra,
                     const std::string & run = "n_steps = 1\ndt = 1.0e-6\n") {
    std::ostringstream s;
    s << "[run]\n" << run << "[mesh]\ntype = \"cartesian\"\nNx = " << n << "\nNy = " << n << "\nLx = 1.0\nLy = 1.0\n";
    if (N_DIM == 3) s << "Nz = " << std::max(n / 2, 3u) << "\nLz = 1.0\nperiodic = [\"x\", \"y\", \"z\"]\n";
    else s << "periodic = [\"x\", \"y\"]\n";
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
    const std::string dir = dir_for("order");
    std::filesystem::remove_all(dir);
    const std::string probe = "[[probes]]\nname = \"p\"\nfile = \"" + dir + "/p.csv\"\nvariables = [\"RHO\", \"T\"]\n"
                              "sampling = \"" + sampling + "\"\nshape = \"points\"\npoints = [" +
                              numbers3("0.137", "0.462", "0.31") + ", " + numbers3("0.713", "0.291", "0.77") + ", " +
                              numbers3("0.402", "0.858", "0.53") + "]\n";
    Solver solver;
    solver.init(parse_toml(box_case(n, SINE_RHO, recon, probe)));
    solver.run();
    double error = 0.0;
    for (const auto & row : read_csv(dir + "/p.csv")) {
        if (row.at("step") != 0.0) continue;
        const double rho = sine_rho(row.at("x"), row.at("y"));
        error = std::max({error, std::fabs(row.at("RHO") - rho), std::fabs(row.at("T") - 1.0 / rho)});
    }
    std::filesystem::remove_all(dir);
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
}

TEST(ProbesTest, PointFilesSkipHeadersAndComments) {
    const std::string dir = dir_for("points_file");
    std::filesystem::create_directories(dir);
    const std::string file = dir + "/cloud.csv";
    std::ofstream(file) << "x, y, z\n# a comment\n0.1, 0.2, 0.3\n\n0.4 0.5 0.6  # trailing\n";
    const auto points = probe::read_points(file, "test");
    ASSERT_EQ(points.size(), 2u);
    EXPECT_EQ(points[1][0], 0.4);
    EXPECT_EQ(points[1][1], 0.5);
    std::ofstream(file) << "0.1, 0.2, 0.3\nx, y\n";
    EXPECT_THROW(probe::read_points(file, "test"), InputError);
    std::filesystem::remove_all(dir);
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
    const std::string dir = dir_for("exact");
    std::filesystem::remove_all(dir);
    const std::string probe = "[[probes]]\nname = \"poly\"\nfile = \"" + dir + "/poly.csv\"\nvariables = [\"RHO\", \"P\"]\n"
                              "sampling = \"point\"\nshape = \"polyline\"\nvertices = [" +
                              numbers3("0.35", "0.42", "0.45") + ", " + numbers3("0.62", "0.57", "0.45") + ", " +
                              numbers3("0.48", "0.66", "0.45") + "]\nn_points = 4\n"
                              "reductions = [\"integral\", \"mean\", \"favre\", \"rms\", \"min\", \"max\"]\n"
                              "reductions_file = \"" + dir + "/poly_reductions.csv\"\n";
    Solver solver;
    solver.init(parse_toml(box_case(20, rho, "type = \"TENO\"\norder = 3\ntroubled_threshold = 1.0\n", probe)));
    solver.run();

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
        EXPECT_GE(row.at("RHO"), r.at("min_RHO"));
        EXPECT_LE(row.at("RHO"), r.at("max_RHO"));
    }
    std::filesystem::remove_all(dir);
}

TEST(ProbesTest, TimeIntervalsLandOnTheirTimes) {
    const std::string dir = dir_for("times");
    std::filesystem::remove_all(dir);
    const std::string probe = "[[probes]]\nname = \"t\"\nfile = \"" + dir + "/t.csv\"\nvariables = [\"P\"]\n"
                              "point = " + numbers3("0.5", "0.5", "0.5") + "\ntime_interval = 0.0123\n";
    Solver solver;
    solver.init(parse_toml(box_case(8, SINE_RHO, "type = \"MUSCL\"\n", probe, "t_stop = 0.05\ncfl = 0.5\n")));
    solver.run();
    const auto rows = read_csv(dir + "/t.csv");
    ASSERT_EQ(rows.size(), 5u);
    for (size_t k = 0; k < rows.size(); k++) EXPECT_NEAR(rows[k].at("t"), 0.0123 * double(k), precision_tol<double>(1e-12, 1e-6));
    std::filesystem::remove_all(dir);
}

TEST(ProbesTest, VtpAndHdf5HoldTheCsvSamplesAndRestartsCutThemBack) {
    const std::string dir = dir_for("formats");
    std::filesystem::remove_all(dir);
    const std::string geometry = "shape = \"circle\"\ncenter = " + numbers3("0.5", "0.5", "0.5") +
                                 "\nradius = 0.3\nn_points = 12\n" + (N_DIM == 3 ? "normal = [1.0, 1.0, 1.0]\n" : "") +
                                 "variables = [\"RHO\", \"U\"]\nsampling = \"point\"\ninterval = 2\n";
    auto probes = [&](const std::string & sub) {
        std::string text = "[[probes]]\nname = \"csv\"\nfile = \"" + dir + "/" + sub + "/c.csv\"\n" + geometry +
                           "[[probes]]\nname = \"vtp\"\nformat = \"vtp\"\nfile = \"" + dir + "/" + sub + "/v.pvd\"\n" +
                           geometry;
#ifdef Mallard_HAS_HDF5
        text += "[[probes]]\nname = \"h5\"\nformat = \"hdf5\"\nfile = \"" + dir + "/" + sub + "/h.h5\"\n" + geometry;
#endif
        return text + "[[write_data]]\nprefix = \"" + dir + "/" + sub + "/r\"\nformat = \"restart\"\ninterval = 4\n";
    };
    const std::string recon = "type = \"TENO\"\norder = 3\n";
    const std::string run6 = "n_steps = 6\ncfl = 0.3\n";
    Solver whole;
    whole.init(parse_toml(box_case(10, SINE_RHO, recon, probes("a"), run6)));
    whole.run();
    {
        // Steps 0 to 6 of an interrupted run, continued from step 4: the
        // series lose the rows after step 4 before the continuation adds its own
        Solver first;
        first.init(parse_toml(box_case(10, SINE_RHO, recon, probes("b"), run6)));
        first.run();
    }
    std::string input = box_case(10, SINE_RHO, recon, probes("b"), run6);
    const size_t init = input.find("[initialize]\n") + 13;
    input.replace(init, input.find("[numerics]") - init, "type = \"restart\"\nfile = \"" + dir + "/b/r_000004.restart\"\n");
    Solver second;
    second.init(parse_toml(input));
    second.run();

    const auto csv = read_csv(dir + "/a/c.csv");
    ASSERT_EQ(csv.size(), 4u * 12u);
    const std::string pvd = read_all(dir + "/b/v.pvd");
    size_t n_sets = 0;
    for (size_t at = pvd.find("<DataSet"); at != std::string::npos; at = pvd.find("<DataSet", at + 1)) n_sets++;
    EXPECT_EQ(n_sets, 4u);
    EXPECT_EQ(pvd, read_all(dir + "/a/v.pvd"));
    EXPECT_EQ(read_all(dir + "/b/v_000006.vtp"), read_all(dir + "/a/v_000006.vtp"));
    const std::string vtp = read_all(dir + "/a/v_000006.vtp");
    EXPECT_NE(vtp.find("NumberOfPoints=\"12\""), std::string::npos);
    EXPECT_NE(vtp.find("NumberOfLines=\"1\""), std::string::npos);
    EXPECT_NE(vtp.find("Name=\"U_X\""), std::string::npos);
#ifdef Mallard_HAS_HDF5
    for (const char * sub : {"a", "b"}) {
        const std::string h5file = dir + "/" + sub + "/h.h5";
        h5::Handle file(H5Fopen(h5file.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT), H5Fclose, "opening");
        ASSERT_EQ(h5::n_rows(file, "step"), 4u);
        const auto steps = h5::read_rows<uint64_t>(file, "step", 0, 4, 1, H5P_DEFAULT);
        EXPECT_EQ(steps, (std::vector<uint64_t>{0, 2, 4, 6}));
        const auto rho = h5::read_rows<double>(file, "fields/RHO", 0, 4, 12, H5P_DEFAULT);
        for (size_t k = 0; k < 4; k++) {
            for (size_t p = 0; p < 12; p++) EXPECT_EQ(rtype(rho[k * 12 + p]), rtype(csv[k * 12 + p].at("RHO")));
        }
    }
#endif
    std::filesystem::remove_all(dir);
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
    if (N_DIM == 3) s << "Nz = 3\nLz = 1.0\nperiodic = [\"x\", \"z\"]\n";
    else s << "periodic = [\"x\"]\n";
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
    // The wall shear stress of the linear profile is exact; the heat flux,
    // from the one-sided wall gradient of the viscous flux, converges at
    // first order
    const std::string dir = dir_for("couette");
    const double q_bottom = -COUETTE_KAPPA * (COUETTE_T1 - COUETTE_T0) - 0.5 * COUETTE_MU * COUETTE_U * COUETTE_U;
    const double q_top = COUETTE_KAPPA * (COUETTE_T1 - COUETTE_T0) - 0.5 * COUETTE_MU * COUETTE_U * COUETTE_U;
    const double tau = COUETTE_MU * COUETTE_U, q_ref = 0.5 * COUETTE_U * COUETTE_U;
    std::vector<double> errors;
    for (uint32_t n : {8u, 16u, 32u}) {
        std::filesystem::remove_all(dir);
        const std::string probe = "[[probes]]\nname = \"walls\"\nfile = \"" + dir + "/walls.csv\"\nboundary_type = \"wall\"\n"
                                  "variables = [\"CF\", \"TAU_W\", \"TAU_W_X\", \"Q_W\", \"U_TAU\", \"Y_PLUS\", \"T\", \"U_X\"]\n"
                                  "reference = {rho = 1.0, u = " + numbers3("0.5", "0.0", "0.0") + ", p = 1.0}\n";
        Solver solver;
        solver.init(parse_toml(couette_case(n, probe)));
        solver.run();
        double error = 0.0;
        size_t n_faces = 0;
        for (const auto & row : read_csv(dir + "/walls.csv")) {
            if (row.at("step") != 0.0) continue;
            n_faces++;
            const bool top = row.at("ny") > 0.0;
            const double sign = top ? -1.0 : 1.0;
            const double tol = precision_tol<double>(1e-10, 2e-4);
            EXPECT_NEAR(row.at("TAU_W_X"), sign * tau, tol * tau);
            EXPECT_NEAR(row.at("TAU_W"), tau, tol * tau);
            EXPECT_NEAR(row.at("CF"), sign * tau / q_ref, tol * tau / q_ref);
            EXPECT_NEAR(row.at("T"), top ? COUETTE_T1 : COUETTE_T0, tol);
            EXPECT_NEAR(row.at("U_X"), top ? COUETTE_U : 0.0, tol);
            const double rho_w = 1.0 / (top ? COUETTE_T1 : COUETTE_T0), u_tau = std::sqrt(tau / rho_w);
            EXPECT_NEAR(row.at("U_TAU"), u_tau, tol * u_tau);
            EXPECT_NEAR(row.at("Y_PLUS"), rho_w * u_tau * (0.5 / n) / COUETTE_MU, tol);
            const double q = top ? q_top : q_bottom;
            error = std::max(error, std::fabs(row.at("Q_W") - q) / std::fabs(q));
        }
        EXPECT_EQ(n_faces, (N_DIM == 2 ? 2u * 4u : 2u * 12u));
        errors.push_back(error);
    }
    EXPECT_LT(errors.back(), 0.05);
    EXPECT_GT(std::log2(errors[1] / errors[2]), 0.8) << errors[1] << " " << errors[2];
    std::filesystem::remove_all(dir);
}

TEST(ProbesTest, BoundaryTractionsIntegrateToTheForces) {
    // Cell sampling takes the forces monitor's wall pressure and gradients
    const std::string dir = dir_for("forces");
    std::filesystem::remove_all(dir);
    const std::string extra = "[[forces]]\nzone = \"top\"\nfile = \"" + dir + "/forces.csv\"\n"
                              "[[probes]]\nname = \"top\"\nboundary = \"top\"\nsamples = false\n"
                              "variables = [\"TRACTION_PRESSURE\", \"TRACTION_VISCOUS\"]\nreductions = [\"integral\"]\n"
                              "reductions_file = \"" + dir + "/top.csv\"\n";
    Solver solver;
    solver.init(parse_toml(couette_case(8, extra, "n_steps = 4\ncfl = 0.5\n")));
    solver.run();
    const auto forces = read_csv(dir + "/forces.csv");
    const auto probe = read_csv(dir + "/top.csv");
    ASSERT_EQ(forces.size(), probe.size());
    const char * const axes[3] = {"x", "y", "z"};
    const double tol = precision_tol<double>(1e-12, 1e-5);
    for (size_t r = 0; r < forces.size(); r++) {
        for (size_t d = 0; d < N_DIM; d++) {
            const double Fp = forces[r].at(std::string("F") + axes[d] + "_pressure");
            const double Fv = forces[r].at(std::string("F") + axes[d] + "_viscous");
            EXPECT_NEAR(probe[r].at(std::string("integral_TRACTION_PRESSURE_") + "XYZ"[d]), Fp, tol * (1.0 + std::fabs(Fp)));
            EXPECT_NEAR(probe[r].at(std::string("integral_TRACTION_VISCOUS_") + "XYZ"[d]), Fv, tol * 1e-2);
        }
    }
    std::filesystem::remove_all(dir);
}

TEST(ProbesTest, InletAndOutletFluxesHaveTheBoundaryStates) {
    // Uniform flow from an upt inlet to a p_out outlet: the face states are
    // the prescribed ones, and the normals point out of the domain
    const std::string dir = dir_for("fluxes");
    std::filesystem::remove_all(dir);
    std::ostringstream s;
    s << "[run]\nn_steps = 1\ndt = 1.0e-9\n[mesh]\ntype = \"cartesian\"\nNx = 8\nNy = 4\nLx = 2.0\nLy = 1.0\n";
    if (N_DIM == 3) s << "Nz = 3\nLz = 1.0\nperiodic = [\"z\"]\n";
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
    solver.run();
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
    std::filesystem::remove_all(dir);
}

#if N_DIM == 2
TEST(ProbesTest, WallPolylinesFollowCurvedWallsWithTheFluidOnTheirLeft) {
    // A steady vortex between curved walls (u = M / r, isentropic pressure):
    // each wall is one closed polyline, counterclockwise around the outer wall
    // and clockwise around the inner one; the point values of the wall
    // pressure beat the cell averages
    const double M = 0.8;
    auto p_exact = [&](double r) { return std::pow(1.0 + 0.2 * M * M * (1.0 - 1.0 / (r * r)), 3.5) / 1.4; };
    const std::string file = write_temp("probes_annulus.msh", annulus_gmsh(4, 48));
    const std::string dir = dir_for("annulus");
    std::filesystem::remove_all(dir);
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
        "[numerics.face_reconstruction]\ntype = \"TENO\"\norder = 4\n" + PERFECT_GAS +
        "[output]\ncheck_interval = 1000000\n" + probes;
    Solver solver;
    solver.init(parse_toml(input));
    solver.run();
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
                if (i + 1 < rows.size()) EXPECT_GT(b.at("s"), a.at("s"));
                // The normal points out of the fluid: outward at the outer wall, inward at the inner one
                EXPECT_GT(turn * (a.at("nx") * a.at("x") + a.at("ny") * a.at("y")), 0.0);
                EXPECT_NEAR(a.at("MASS_FLUX"), 0.0, 1e-12);
                total += a.at("area");
                error[k] = std::max(error[k], std::fabs(a.at("P") - p_exact(r)));
            }
            EXPECT_NEAR(total, 2.0 * std::numbers::pi * r, 1e-3 * r) << zone;
        }
        EXPECT_LT(error[1], 0.2 * error[0]) << zone << ": point " << error[1] << ", cell " << error[0];
    }
    std::filesystem::remove_all(dir);
}
#endif

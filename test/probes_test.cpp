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

#include "comm.h"
#include "file_format.h"
#include "input.h"
#include "probe_geometry.h"
#include "scratch.h"
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

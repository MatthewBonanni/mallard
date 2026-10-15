/**
 * @file statistics_test.cpp
 * @author Matthew Bonanni (mbonanni001@gmail.com)
 * @brief Running time averages and probes.
 * @version 0.1
 * @date 2026-10-04
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 *
 */

#include <gtest/gtest.h>
#include <Kokkos_Core.hpp>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <iomanip>
#include <fstream>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

#include "solver.h"
#include "test_fixtures.h"

namespace {

const std::string PERFECT_GAS = "[physics]\ntype = \"euler\"\ngamma = 1.4\np_ref = 1.0\nT_ref = 1.0\nrho_ref = 1.0\n";

std::string box(uint32_t n, bool periodic) {
    std::ostringstream s;
    s << "[mesh]\ntype = \"cartesian\"\nNx = " << n << "\nNy = " << n << "\nLx = 1.0\nLy = 1.0\n";
    if (N_DIM == 3) s << "Nz = " << std::max(n / 2, 3u) << "\nLz = 1.0\n";
    if (periodic) s << (N_DIM == 2 ? "periodic = [\"x\", \"y\"]\n" : "periodic = [\"x\", \"y\", \"z\"]\n");
    return s.str();
}

// Arrays of N_DIM expressions or numbers
std::string vector3(const char * x, const char * y, const char * z) {
    return std::string("[\"") + x + "\", \"" + y + "\"" + (N_DIM == 3 ? std::string(", \"") + z + "\"" : "") + "]";
}

std::string numbers3(const char * x, const char * y, const char * z) {
    return std::string("[") + x + ", " + y + (N_DIM == 3 ? std::string(", ") + z : "") + "]";
}

// A blob advected through a fully periodic box
std::string blob_input(uint32_t n_steps, const std::string & init, const std::string & extra) {
    return "[run]\nn_steps = " + std::to_string(n_steps) + "\ncfl = 0.25\n" + box(10, true) + "[initialize]\n" + init +
           "[numerics]\nriemann_solver = \"HLLC\"\ntime_integrator = \"SSPRK3\"\n"
           "[numerics.face_reconstruction]\ntype = \"MUSCL\"\n" +
           PERFECT_GAS + "[output]\ncheck_interval = 1000000\n" + extra;
}

const std::string BLOB = "type = \"analytical\"\n"
                         "rho = \"1.0 + 0.5 * exp(-30 * ((x - 0.4)^2 + (y - 0.5)^2))\"\n"
                         "u = " + vector3("0.3", "-0.1", "0.2") + "\n"
                         "p = \"1.0 + exp(-30 * ((x - 0.4)^2 + (y - 0.5)^2))\"\n";

std::string read_all(const std::string & file) {
    std::ifstream in(file);
    std::stringstream ss;
    ss << in.rdbuf();
    return ss.str();
}

/** @brief Rows of a CSV file, split at commas, without the header. */
std::vector<std::vector<std::string>> csv_rows(const std::string & file) {
    std::ifstream in(file);
    std::string line;
    std::getline(in, line);
    std::vector<std::vector<std::string>> rows;
    while (std::getline(in, line)) {
        std::vector<std::string> row;
        std::stringstream ss(line);
        std::string cell;
        while (std::getline(ss, cell, ',')) row.push_back(cell);
        rows.push_back(row);
    }
    return rows;
}

} // namespace

TEST(StatisticsTest, AveragesOfAnOscillatingFlowMatchTheAnalyticalOnes) {
    // A uniform body force A cos(w t) on a gas at rest in a periodic box gives
    // u = u0 sin(w t), u0 = A / (rho w), in every cell; SSPRK3 integrates it as
    // Simpson's rule. With dt = 1/128 (exact in binary) and samples every 2
    // steps over t in (1/4, 5/4], the samples cover one period uniformly, so the
    // time-weighted mean of u is 0, its variance u0^2 / 2, and with
    // p = 0.4 (rho E - rho u^2 / 2) the mean pressure p0 - 0.1 u0^2.
    const double u0 = 0.1, w = 2.0 * M_PI;
    std::ostringstream force;
    force << std::setprecision(17) << u0 * w << " * cos(" << w << " * t)";
    const std::string input =
        "[run]\nn_steps = 160\ndt = 0.0078125\n" + box(4, true) +
        "[initialize]\ntype = \"constant\"\nu = " + numbers3("0", "0", "0") + "\np = 1.0\nT = 1.0\n" + PERFECT_GAS +
        "[source]\nrhou = " + vector3(force.str().c_str(), "0", "0") + "\ntime_dependent = true\n"
        "[output]\ncheck_interval = 1000000\n"
        "[statistics]\nt_start = 0.25\ninterval = 2\nfields = [\"RHO\", \"P\"]\nproducts = [\"U_X*U_X\", \"P*U_X\"]\n";
    Solver solver;
    solver.init(parse_toml(input));
    solver.run();
    solver.copy_device_to_host();
    const Statistics & stats = solver.get_statistics();
    // Fields: RHO, P, U_X; products: U_X*U_X, P*U_X
    EXPECT_EQ(stats.samples(), 64u);
    const double tol = precision_tol<double>(1e-7, 1e-5);
    const auto & mean = stats.host_means();
    const auto & cov = stats.host_covariances();
    for (uint32_t c = 0; c < solver.get_mesh()->n_cells; c++) {
        EXPECT_NEAR(double(mean(c, 0)), 1.0, tol);
        EXPECT_NEAR(double(mean(c, 1)), 1.0 - 0.1 * u0 * u0, tol);
        EXPECT_NEAR(double(mean(c, 2)), 0.0, tol * u0);
        EXPECT_NEAR(double(cov(c, 0)), 0.5 * u0 * u0, tol * u0 * u0);
        // p' = 0.1 u0^2 cos(2 w t) is orthogonal to u' over a period
        EXPECT_NEAR(double(cov(c, 1)), 0.0, tol * u0 * u0);
    }
}

TEST(StatisticsTest, StatisticsLeaveTheSolutionUnchangedAndRestartBitwise) {
    // Samples every 3 steps, a restart at step 20 between two of them: the
    // continued averages (their weights and the last sample's time carry over)
    // and the appended probe series equal those of an uninterrupted run
    const std::string dir = (std::filesystem::temp_directory_path() / "mallard_statistics_test").string();
    std::filesystem::remove_all(dir);
    auto extra = [&](const std::string & sub) {
        return "[statistics]\nt_start = 0.01\ninterval = 3\nfields = [\"T\"]\n"
               "products = [\"U_X*U_X\", \"U_X*U_Y\", \"RHO*T\"]\n"
               "[[probes]]\nname = \"line\"\nfile = \"" + dir + "/" + sub + "/probe.csv\"\ninterval = 2\n"
               "variables = [\"RHO\", \"U_Y\"]\nstart = " + numbers3("0.05", "0.05", "0.05") + "\nend = " +
               numbers3("0.95", "0.7", "0.5") + "\nn_points = 7\n"
               "[[write_data]]\nprefix = \"" + dir + "/" + sub + "/r\"\nformat = \"restart\"\ninterval = 20\n";
    };
    auto from = [&](const std::string & file) { return "type = \"restart\"\nfile = \"" + file + "\"\n"; };

    Solver straight;
    straight.init(parse_toml(blob_input(40, BLOB, extra("a"))));
    straight.run();
    straight.copy_device_to_host();

    Solver plain;
    plain.init(parse_toml(blob_input(40, BLOB, "")));
    plain.run();
    plain.copy_device_to_host();

    Solver first;
    first.init(parse_toml(blob_input(20, BLOB, extra("b"))));
    first.run();
    Solver second;
    second.init(parse_toml(blob_input(40, from(dir + "/b/r_000020.restart"), extra("b"))));
    second.run();
    second.copy_device_to_host();

    // A run without statistics reads a restart file holding them
    Solver without;
    without.init(parse_toml(blob_input(40, from(dir + "/b/r_000020.restart"), "")));
    without.run();
    without.copy_device_to_host();

    const uint32_t n_cells = straight.get_mesh()->n_cells;
    for (uint32_t c = 0; c < n_cells; c++) {
        FOR_I_CONSERVATIVE {
            EXPECT_EQ(plain.h_conservatives(c, i), straight.h_conservatives(c, i));
            EXPECT_EQ(without.h_conservatives(c, i), straight.h_conservatives(c, i));
        }
    }
    const Statistics & a = straight.get_statistics();
    const Statistics & b = second.get_statistics();
    EXPECT_EQ(b.samples(), a.samples());
    for (uint32_t c = 0; c < n_cells; c++) {
        for (uint32_t i = 0; i < a.host_means().extent(1); i++) EXPECT_EQ(b.host_means()(c, i), a.host_means()(c, i));
        for (uint32_t p = 0; p < a.host_covariances().extent(1); p++) {
            EXPECT_EQ(b.host_covariances()(c, p), a.host_covariances()(c, p));
        }
    }
    EXPECT_EQ(read_all(dir + "/b/probe.csv"), read_all(dir + "/a/probe.csv"));
    std::filesystem::remove_all(dir);
}

TEST(StatisticsTest, ProbesReadTheCellHoldingEachPoint) {
    // Points on faces between cells take the cell with the lowest id, points
    // outside the domain the nearest centroid; the values are the cell's, bitwise
    const std::string dir = (std::filesystem::temp_directory_path() / "mallard_probe_test").string();
    std::filesystem::remove_all(dir);
    const std::string probes =
        "[[probes]]\nname = \"line\"\nfile = \"" + dir + "/line.csv\"\nvariables = [\"RHO\", \"U_X\", \"P\", \"T\"]\n"
        "start = " + numbers3("0", "0.5", "0.5") + "\nend = " + numbers3("1", "0.5", "0.5") + "\nn_points = 11\n"
        "[[probes]]\nname = \"out\"\nfile = \"" + dir + "/out.csv\"\nvariables = [\"RHO\", \"U_X\", \"P\", \"T\"]\n"
        "point = " + numbers3("1.3", "0.27", "0.62") + "\n";
    Solver solver;
    solver.init(parse_toml(blob_input(6, BLOB, probes)));
    solver.run();
    solver.copy_device_to_host();
    const Mesh & mesh = *solver.get_mesh();

    const double tol = precision_tol<double>(1e-12, 1e-6);
    auto expected_cell = [&](const std::vector<double> & x) {
        for (uint32_t c = 0; c < mesh.n_cells; c++) {
            bool inside = true;
            for (size_t d = 0; d < N_DIM; d++) {
                double lo = std::numeric_limits<double>::max(), hi = std::numeric_limits<double>::lowest();
                for (uint32_t k = 0; k < mesh.h_n_nodes_of_cell(c); k++) {
                    const double v = double(mesh.h_node_coords(mesh.h_node_of_cell(c, k), d));
                    lo = std::min(lo, v);
                    hi = std::max(hi, v);
                }
                inside = inside && x[d] >= lo - tol && x[d] <= hi + tol;
            }
            if (inside) return c;
        }
        uint32_t best = 0;
        double d_best = std::numeric_limits<double>::max();
        for (uint32_t c = 0; c < mesh.n_cells; c++) {
            double d2 = 0.0;
            for (size_t d = 0; d < N_DIM; d++) d2 += std::pow(x[d] - double(mesh.h_cell_coords(c, d)), 2);
            if (d2 < d_best) {
                d_best = d2;
                best = c;
            }
        }
        return best;
    };
    size_t n_checked = 0;
    for (const char * name : {"line", "out"}) {
        for (const auto & row : csv_rows(dir + "/" + name + ".csv")) {
            if (std::stoull(row[0]) != solver.get_step()) continue;
            std::vector<double> x;
            for (size_t d = 0; d < N_DIM; d++) x.push_back(std::stod(row[3 + d]));
            const uint32_t c = expected_cell(x);
            const rtype expected[4] = {solver.h_conservatives(c, 0), solver.h_primitives(c, 0),
                                       solver.h_primitives(c, N_DIM), solver.h_primitives(c, N_DIM + 1)};
            for (size_t v = 0; v < 4; v++) {
                EXPECT_EQ(static_cast<rtype>(std::stod(row[3 + N_DIM + v])), expected[v]) << name << " point " << row[2];
            }
            n_checked++;
        }
    }
    EXPECT_EQ(n_checked, 12u);
    std::filesystem::remove_all(dir);
}

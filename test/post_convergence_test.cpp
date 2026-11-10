/**
 * @file post_convergence_test.cpp
 * @brief Observed order, Richardson extrapolation and GCI: manufactured sequences of known order, the worked
 *        examples of Celik et al. (2008) and of the NPARC Alliance, and mallard post convergence on runs.
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 */

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "cli.h"
#include "comm.h"
#include "commands.h"
#include "convergence.h"
#include "json.h"
#include "scratch.h"
#include "solver.h"
#include "test_fixtures.h"

namespace {

namespace fs = std::filesystem;
using convergence::Behavior;

/** @brief phi(h) = phi0 + c h^p on grids of sizes h, finest first. */
std::array<double, 3> sequence(const std::array<double, 3> & h, double phi0, double c, double p) {
    return {phi0 + c * std::pow(h[0], p), phi0 + c * std::pow(h[1], p), phi0 + c * std::pow(h[2], p)};
}

} // namespace

TEST(ConvergenceTest, ConstantRatioRecoversTheOrderAndTheLimit) {
    const std::array<double, 3> h = {0.01, 0.02, 0.04};
    const auto e = convergence::three_grids(h, sequence(h, 2.0, 30.0, 2.0));
    EXPECT_EQ(e.behavior, Behavior::monotonic);
    EXPECT_NEAR(e.p, 2.0, 1e-9);
    EXPECT_NEAR(e.extrapolated, 2.0, 1e-12);
    EXPECT_NEAR(e.R, 0.25, 1e-12);
    // Exact sequence: the GCI with Fs = 1 would be the true relative error of the fine grid
    const double phi1 = 2.0 + 30.0 * 1e-4;
    EXPECT_NEAR(e.gci, 1.25 * (phi1 - 2.0) / phi1, 1e-12);
}

TEST(ConvergenceTest, VaryingRatioIteratesToTheOrder) {
    const std::array<double, 3> h = {1.0, 1.5, 1.5 * 1.7};
    for (const double p : {0.8, 1.7, 3.0, 5.0}) {
        const auto e = convergence::three_grids(h, sequence(h, -1.0, 0.05, p));
        EXPECT_TRUE(e.iterated) << p;
        EXPECT_NEAR(e.p, p, 1e-9) << p;
        EXPECT_NEAR(e.extrapolated, -1.0, 1e-9) << p;
    }
}

TEST(ConvergenceTest, AlternatingErrorsAreOscillatoryWithTheirOrder) {
    const std::array<double, 3> h = {0.5, 1.0, 2.0};
    const double c = 0.01, p = 2.0;
    const std::array<double, 3> phi = {1.0 + c * std::pow(h[0], p), 1.0 - c * std::pow(h[1], p), 1.0 + c * std::pow(h[2], p)};
    const auto e = convergence::three_grids(h, phi);
    EXPECT_EQ(e.behavior, Behavior::oscillatory);
    EXPECT_LT(e.R, 0.0);
    EXPECT_NEAR(e.p, p, 1e-9);
}

TEST(ConvergenceTest, GrowingChangesDivergeAndEqualValuesAreUnchanged) {
    const std::array<double, 3> h = {1.0, 2.0, 4.0};
    EXPECT_EQ(convergence::three_grids(h, sequence(h, 1.0, 1.0, -1.0)).behavior, Behavior::divergent);
    const auto same = convergence::three_grids(h, {3.0, 3.0, 3.0});
    EXPECT_EQ(same.behavior, Behavior::unchanged);
    EXPECT_EQ(same.gci, 0.0);
    EXPECT_EQ(convergence::three_grids(h, {3.0, 3.0, 3.1}).behavior, Behavior::indeterminate);
    EXPECT_THROW(convergence::three_grids({1.0, 1.0, 2.0}, {1.0, 2.0, 3.0}), std::invalid_argument);
}

TEST(ConvergenceTest, TwoGridsAssumeTheOrderWithRoachesFactorOfSafety) {
    const auto e = convergence::two_grids({0.1, 0.2}, {1.0 + 0.5 * 0.01, 1.0 + 0.5 * 0.04}, 2.0);
    EXPECT_EQ(e.behavior, Behavior::assumed);
    EXPECT_EQ(e.safety, 3.0);
    EXPECT_NEAR(e.extrapolated, 1.0, 1e-12);
    EXPECT_NEAR(e.gci, 3.0 * (0.015 / 1.005) / 3.0, 1e-12);
}

TEST(ConvergenceTest, ErrorsOfKnownOrderGiveItPairwiseAndFitted) {
    const std::vector<double> h = {0.4, 0.2, 0.1, 0.05};
    std::vector<double> e;
    for (const double x : h) e.push_back(3.0 * std::pow(x, 2.5));
    EXPECT_NEAR(convergence::pair_order(h[0], h[1], e[0], e[1]), 2.5, 1e-12);
    EXPECT_NEAR(convergence::fitted_order(h, e), 2.5, 1e-12);
}

// Celik et al. (2008), Table 1: N1, N2, N3 cells of a 2D backward-facing step, h = (1 / N)^(1/2)
TEST(ConvergenceTest, CelikTable1) {
    struct Column {
        std::array<double, 3> n, phi;
        double r21, r32, p, ext, e_a, e_ext, gci;
        Behavior behavior;
    };
    const Column columns[] = {
        {{18000, 8000, 4500}, {6.063, 5.972, 5.863}, 1.5, 1.333, 1.53, 6.1685, 0.015, 0.017, 0.022, Behavior::monotonic},
        {{18000, 4500, 980}, {10.7880, 10.7250, 10.6050}, 2.0, 2.143, 0.75, 10.8801, 0.006, 0.009, 0.011,
         Behavior::monotonic},
        {{18000, 4500, 980}, {6.0042, 5.9624, 6.0909}, 2.0, 2.143, 1.51, 6.0269, 0.007, 0.004, 0.005,
         Behavior::oscillatory},
    };
    for (const Column & c : columns) {
        const std::array<double, 3> h = {std::sqrt(1.0 / c.n[0]), std::sqrt(1.0 / c.n[1]), std::sqrt(1.0 / c.n[2])};
        const auto e = convergence::three_grids(h, c.phi);
        EXPECT_EQ(e.behavior, c.behavior);
        EXPECT_NEAR(e.r21, c.r21, 5e-4);
        EXPECT_NEAR(e.r32, c.r32, 5e-4);
        EXPECT_NEAR(e.p, c.p, 5e-3);
        EXPECT_NEAR(e.extrapolated, c.ext, 5e-5);
        // The table rounds these to 0.1%; its second column's e_ext reads 0.9% where its own data give 0.85%
        EXPECT_NEAR(e.e_a, c.e_a, 1e-3);
        EXPECT_NEAR(e.e_ext, c.e_ext, 1e-3);
        EXPECT_NEAR(e.gci, c.gci, 1e-3);
    }
}

// NPARC Alliance, "Examination of spatial grid convergence": inlet pressure recovery on grids of spacing 1, 2, 4
TEST(ConvergenceTest, NparcPressureRecoveryExample) {
    const auto e = convergence::three_grids({1.0, 2.0, 4.0}, {0.97050, 0.96854, 0.96178});
    EXPECT_EQ(e.behavior, Behavior::monotonic);
    EXPECT_NEAR(e.p, 1.786170, 1e-6);
    EXPECT_NEAR(e.extrapolated, 0.97130, 5e-6);
    EXPECT_NEAR(100.0 * e.gci, 0.103083, 1e-5);
}

namespace {

std::string fresh_dir(const std::string & name) {
    const std::string dir = (scratch_dir() / ("convergence_" + name)).string();
    comm::barrier();
    if (comm::is_root()) fs::remove_all(dir);
    comm::barrier();
    fs::create_directories(dir);
    return dir;
}

constexpr double T_STOP = 0.05;

/** @brief Cells across the 3D box: refined with the others, so that the refinement is uniform. */
uint32_t layers(const uint32_t n) { return std::max(2u, n / 8); }

double cells(const uint32_t n) { return double(n) * n * (N_DIM == 3 ? layers(n) : 1); }

/** @brief A density wave advected across a periodic box: rho(x - u t, y - v t), second order with MUSCL unlimited. */
std::string wave_input(const std::string & dir, const uint32_t n, const std::string & riemann) {
    std::ostringstream s;
    s << "[run]\nt_stop = " << T_STOP << "\ncfl = 0.5\n[mesh]\ntype = \"cartesian\"\nNx = " << n << "\nNy = " << n
      << "\nLx = 1.0\nLy = 1.0\n";
    if (N_DIM == 3) s << "Nz = " << layers(n) << "\nLz = " << double(layers(n)) / n << "\n[[periodic]]\nzones = [\"back\", \"front\"]\n";
    s << "[[periodic]]\nzones = [\"left\", \"right\"]\n[[periodic]]\nzones = [\"bottom\", \"top\"]\n"
      << "[initialize]\ntype = \"analytical\"\nrho = \"1.0 + 0.2 * sin(2 * pi * x) * sin(2 * pi * y)\"\n"
      << "u = " << (N_DIM == 3 ? "[\"1.0\", \"0.5\", \"0.0\"]" : "[\"1.0\", \"0.5\"]") << "\np = \"1.0\"\n"
      << "[numerics]\nriemann_solver = \"" << riemann << "\"\ntime_integrator = \"SSPRK3\"\n"
      << "[numerics.face_reconstruction]\ntype = \"MUSCL\"\nlimiter = \"none\"\n"
      << "[physics]\ntype = \"euler\"\ngamma = 1.4\np_ref = 1.0\nT_ref = 1.0\nrho_ref = 1.0\n"
      << "[output]\ncheck_interval = 1000000\nrun_log = \"" << dir << "/mallard.jsonl\"\n"
      << "[integrals]\ninterval = 1\nfile = \"" << dir << "/integrals.csv\"\n"
      << "[[write_data]]\nprefix = \"" << dir << "/restart\"\nformat = \"restart\"\ntime_interval = " << T_STOP << "\n";
    return s.str();
}

std::string wave_run(const uint32_t n, const std::string & riemann = "HLLC") {
    const std::string dir = fresh_dir("n" + std::to_string(n) + riemann);
    Solver solver;
    solver.init(parse_toml(wave_input(dir, n, riemann)));
    solver.run();
    comm::barrier();
    return dir;
}

struct Result {
    int status = 0;
    std::string out, err;
};

Result convergence_cmd(const std::vector<std::string> & words) {
    cli::Args args;
    std::string error;
    EXPECT_TRUE(cli::parse(*cli::command("post convergence"), words, args, error)) << error;
    std::ostringstream out, err;
    std::streambuf * o = std::cout.rdbuf(out.rdbuf());
    std::streambuf * e = std::cerr.rdbuf(err.rdbuf());
    Result r;
    r.status = post_convergence(args);
    std::cout.rdbuf(o);
    std::cerr.rdbuf(e);
    r.out = out.str();
    r.err = err.str();
    return r;
}

/** @brief The last value of a column of a CSV file. */
double last_value(const std::string & file, const std::string & column) {
    std::ifstream in(file);
    std::string line, last;
    std::vector<std::string> header;
    while (std::getline(in, line)) {
        if (line.empty() || line[0] == '#') continue;
        if (header.empty()) {
            std::stringstream s(line);
            for (std::string c; std::getline(s, c, ',');) header.push_back(c);
        } else {
            last = line;
        }
    }
    std::stringstream s(last);
    std::string cell;
    for (size_t i = 0; std::getline(s, cell, ','); i++) {
        if (header[i] == column) return std::stod(cell);
    }
    return std::nan("");
}

} // namespace

TEST(PostConvergenceTest, RunsInAnyOrderGiveTheirTablesAndTheSchemesOrder) {
    if (comm::size() > 1) GTEST_SKIP() << "runs on one rank";
    const std::string coarse = wave_run(16), medium = wave_run(32), fine = wave_run(64);

    const Result scalars = convergence_cmd({medium, fine, coarse, "--quantity", "integrals:density_squared", "--format",
                                            "json"});
    ASSERT_EQ(scalars.status, 0) << scalars.err;
    const json::Value s = json::parse(scalars.out);
    ASSERT_EQ(s["runs"].array.size(), 3u);
    EXPECT_EQ(s["runs"].array[0]["cells"].number, cells(16));
    EXPECT_EQ(s["runs"].array[2]["cells"].number, cells(64));
    EXPECT_EQ(s["design_order"].number, 2.0);
    ASSERT_EQ(s["quantities"].array.size(), 1u);
    const json::Value & q = s["quantities"].array[0];
    EXPECT_EQ(q["name"].string, "integrals:density_squared");
    const std::string dirs[] = {coarse, medium, fine};
    for (size_t i = 0; i < 3; i++) {
        EXPECT_DOUBLE_EQ(q["values"].array[i].number, last_value(dirs[i] + "/integrals.csv", "density_squared"));
        EXPECT_NEAR(q["t"].array[i].number, T_STOP, 1e-12);
    }
    EXPECT_NE(q["estimate"]["behavior"].string, "divergent");

    const std::string rho = "1.0 + 0.2 * sin(2 * pi * (x - t)) * sin(2 * pi * (y - 0.5 * t))";
    const Result norms = convergence_cmd({fine, coarse, medium, "--field", "RHO", "--exact", rho, "--format", "json"});
    ASSERT_EQ(norms.status, 0) << norms.err;
    const json::Value n = json::parse(norms.out);
    const json::Value & l1 = n["field"]["norms"].array[0];
    EXPECT_EQ(l1["norm"].string, "L1");
    EXPECT_GT(l1["errors"].array[0].number, l1["errors"].array[2].number);
    EXPECT_NEAR(l1["orders"].array[1].number, 2.0, 0.4);

    // The default: every column; those that vanish (no vorticity in this flow) are zero, not divergent noise
    const Result all = convergence_cmd({coarse, medium, fine, "--format", "json"});
    ASSERT_EQ(all.status, 0) << all.out << all.err;
    const json::Value a = json::parse(all.out);
    EXPECT_EQ(a["status"].string, "ok") << all.out;  // their inputs differ only in the mesh and where files go
    bool enstrophy = false;
    for (const json::Value & e : a["quantities"].array) {
        if (e["name"].string != "integrals:enstrophy") continue;
        enstrophy = true;
        EXPECT_EQ(e["behavior"].string, "zero");
        EXPECT_EQ(e["status"].string, "ok");
    }
    EXPECT_TRUE(enstrophy) << all.out;

    const Result text = convergence_cmd({coarse, medium, fine, "--quantity", "density_squared"});
    EXPECT_NE(text.out.find("Grid convergence"), std::string::npos) << text.out;
    EXPECT_NE(text.out.find("integrals:density_squared: "), std::string::npos) << text.out;
}

TEST(PostConvergenceTest, TwoRunsOfDifferentSchemesAreWarnings) {
    if (comm::size() > 1) GTEST_SKIP() << "runs on one rank";
    const Result r = convergence_cmd({wave_run(8), wave_run(16, "Rusanov"), "--quantity", "density_squared",
                                      "--format", "json"});
    ASSERT_EQ(r.status, 0) << r.err;
    const json::Value v = json::parse(r.out);
    EXPECT_EQ(v["status"].string, "warning");
    ASSERT_EQ(v["input_differences"].array.size(), 1u);
    EXPECT_NE(v["input_differences"].array[0].string.find("[numerics]"), std::string::npos);
    const json::Value & q = v["quantities"].array[0];
    EXPECT_EQ(q["behavior"].string, "assumed");
    EXPECT_EQ(q["status"].string, "warning");
    EXPECT_EQ(q["estimate"]["safety_factor"].number, 3.0);
    EXPECT_EQ(q["verdict"].string.rfind("2 grids: order assumed, not observed; Fs = 3", 0), 0u);
}

TEST(PostConvergenceTest, MisuseIsAUsageErrorAndOneRunTwiceAnError) {
    if (comm::size() > 1) GTEST_SKIP() << "runs on one rank";
    const std::string coarse = wave_run(8);
    const Result field = convergence_cmd({coarse, coarse, "--field", "RHO"});
    EXPECT_EQ(field.status, 2);
    EXPECT_NE(field.err.find("planned"), std::string::npos) << field.err;
    const Result twice = convergence_cmd({coarse, coarse});
    EXPECT_EQ(twice.status, 1);
    EXPECT_NE(twice.err.find("same"), std::string::npos) << twice.err;
}

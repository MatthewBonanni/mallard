/**
 * @file conditional_statistics_test.cpp
 * @brief [[statistics.conditional]]: PDFs and conditional moments are those of the sampled cells, weighted in
 *        time, the same bit for bit on any number of ranks, across restarts and in mallard post average.
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 */

#include <gtest/gtest.h>

#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "cli.h"
#include "comm.h"
#include "commands.h"
#include "data_paths.h"
#include "device_comm.h"
#include "mpi_compare.h"
#include "scratch.h"
#include "solver.h"
#include "test_fixtures.h"

namespace {

namespace fs = std::filesystem;

std::string fresh_dir(const std::string & name) {
    const std::string dir = (scratch_dir() / ("conditional_" + name)).string();
    comm::barrier();
    if (comm::is_root()) fs::remove_all(dir);
    comm::barrier();
    return dir;
}

std::string vector_of(const char * x, const char * y, const char * z) {
    return N_DIM == 3 ? std::string("[") + x + ", " + y + ", " + z + "]" : std::string("[") + x + ", " + y + "]";
}

/** @brief A periodic box of a smooth perfect-gas flow, with extra keys (statistics, output). */
std::string flow_case(uint32_t n_steps, const std::string & extra) {
    std::ostringstream s;
    s << "[run]\nn_steps = " << n_steps << "\ndt = 1.0e-2\n"
      << "[mesh]\ntype = \"cartesian\"\nNx = 12\nNy = 10\nLx = 1.0\nLy = 1.0\n";
    if (N_DIM == 3) s << "Nz = 3\nLz = 1.0\n[[periodic]]\nzones = [\"back\", \"front\"]\n";
    s << "[[periodic]]\nzones = [\"left\", \"right\"]\n[[periodic]]\nzones = [\"bottom\", \"top\"]\n"
      << "[initialize]\ntype = \"analytical\"\nrho = \"1.0 + 0.4 * sin(2 * pi * x) * sin(2 * pi * y)\"\n"
      << "p = \"1.0 + 0.2 * cos(2 * pi * y)\"\n"
      << "u = " << vector_of("\"0.3 + 0.1 * sin(2 * pi * y)\"", "\"-0.1 + 0.1 * sin(2 * pi * x)\"", "\"0.0\"") << "\n"
      << "[numerics]\nriemann_solver = \"HLLC\"\ntime_integrator = \"SSPRK3\"\n"
      << "[numerics.face_reconstruction]\ntype = \"MUSCL\"\n"
      << "[physics]\ntype = \"euler\"\ngamma = 1.4\np_ref = 1.0\nT_ref = 1.0\nrho_ref = 1.0\n"
      << "[output]\ncheck_interval = 1000000\nrun_log = \"\"\n"
      << extra;
    return s.str();
}

/** @brief Two entries: a PDF of RHO with moments of P and U_X, and a joint PDF of RHO and P in many bins. */
std::string two_entries(const std::string & dir, const std::string & weight) {
    return "[[statistics.conditional]]\non = \"RHO\"\nbins = 8\nrange = [0.7, 1.3]\nfields = [\"P\", \"U_X\"]\n"
           "weight = \"" + weight + "\"\nfile = \"" + dir + "/rho.csv\"\n"
           "[[statistics.conditional]]\non = [\"RHO\", \"P\"]\nbins = [64, 48]\nrange = [[0.6, 1.4], [0.8, 1.2]]\n"
           "fields = [\"T\", \"U_Y\"]\nweight = \"" + weight + "\"\nfile = \"" + dir + "/joint.csv\"\n";
}

/** @brief The result an entry should hold after one sample of the solver's current state, on the host. */
ConditionalStatistics::Result expected_of_one_sample(const Solver & solver, const ConditionalStatistics::Result & got,
                                                     const std::vector<uint32_t> & on_index,
                                                     const std::vector<uint32_t> & field_index, bool mass) {
    // Values by index: 0 RHO, 1 + i the primitive i
    const Mesh & mesh = *solver.get_mesh();
    const auto value = [&](uint32_t c, uint32_t k) {
        return k == 0 ? double(solver.h_conservatives(c, 0)) : double(solver.h_primitives(c, k - 1));
    };
    const size_t n_grid = got.weight.size(), F = field_index.size();
    std::vector<double> W(n_grid, 0.0), S(n_grid * F, 0.0), S2(n_grid * F, 0.0);
    double total = 0.0, outside = 0.0;
    for (uint32_t c = 0; c < mesh.n_owned(); c++) {
        const double w = double(mesh.h_cell_measure(c)) * (mass ? value(c, 0) : 1.0);
        total += w;
        size_t b = 0;
        bool in = true;
        for (size_t v = 0; v < on_index.size(); v++) {
            const double x = value(c, on_index[v]);
            const double scale = got.bins[v] / (got.range[v][1] - got.range[v][0]);
            const double y = (x - got.range[v][0]) * scale;
            const double i = y == double(got.bins[v]) ? got.bins[v] - 1.0 : std::floor(y);
            in = in && i >= 0.0 && i < double(got.bins[v]);
            b = b * got.bins[v] + (in ? static_cast<size_t>(i) : 0);
        }
        if (!in) {
            outside += w;
            continue;
        }
        W[b] += w;
        for (size_t f = 0; f < F; f++) {
            S[b * F + f] += w * value(c, field_index[f]);
            S2[b * F + f] += w * value(c, field_index[f]) * value(c, field_index[f]);
        }
    }
    ConditionalStatistics::Result r = got;
    const double dt = got.t_last - got.t_first;
    r.total = total * dt;
    r.outside = outside * dt;
    double width = 1.0;
    for (size_t v = 0; v < got.bins.size(); v++) width *= (got.range[v][1] - got.range[v][0]) / got.bins[v];
    for (size_t b = 0; b < n_grid; b++) {
        r.weight[b] = W[b] * dt;
        r.pdf[b] = W[b] / (total * width);
        for (size_t f = 0; f < F; f++) {
            const double m = S[b * F + f] / W[b];
            r.mean[b * F + f] = m;
            r.variance[b * F + f] = S2[b * F + f] / W[b] - m * m;
        }
    }
    return r;
}

void expect_near_result(const ConditionalStatistics::Result & got, const ConditionalStatistics::Result & want,
                        const std::string & what) {
    const double tol = 1e-11;
    ASSERT_EQ(got.weight.size(), want.weight.size());
    EXPECT_NEAR(got.total, want.total, tol * want.total) << what;
    EXPECT_NEAR(got.outside, want.outside, tol * want.total) << what;
    size_t filled = 0;
    const size_t F = got.fields.size();
    for (size_t b = 0; b < got.weight.size(); b++) {
        EXPECT_NEAR(got.weight[b], want.weight[b], tol * want.total) << what << " bin " << b;
        EXPECT_NEAR(got.pdf[b], want.pdf[b], tol * (1.0 + want.pdf[b])) << what << " bin " << b;
        if (!(want.weight[b] > 0.0)) {
            for (size_t f = 0; f < F; f++) EXPECT_TRUE(std::isnan(got.mean[b * F + f])) << what << " bin " << b;
            continue;
        }
        filled++;
        for (size_t f = 0; f < F; f++) {
            const double m = want.mean[b * F + f], v = want.variance[b * F + f];
            EXPECT_NEAR(got.mean[b * F + f], m, tol * (1.0 + std::abs(m))) << what << " bin " << b << " field " << f;
            EXPECT_NEAR(got.variance[b * F + f], v, 1e-9 * (1.0 + m * m)) << what << " bin " << b << " field " << f;
        }
    }
    EXPECT_GT(filled, 3u) << what;
}

bool same_bits(const ConditionalStatistics::Result & a, const ConditionalStatistics::Result & b) {
    const auto same = [](const std::vector<double> & x, const std::vector<double> & y) {
        if (x.size() != y.size()) return false;
        for (size_t i = 0; i < x.size(); i++) {
            if (!(x[i] == y[i] || (std::isnan(x[i]) && std::isnan(y[i])))) return false;
        }
        return true;
    };
    return a.total == b.total && a.outside == b.outside && a.samples == b.samples && same(a.weight, b.weight) &&
           same(a.pdf, b.pdf) && same(a.mean, b.mean) && same(a.variance, b.variance);
}

/** @brief The data rows of a CSV file, as written. */
std::vector<std::string> data_rows(const std::string & path) {
    std::ifstream in(path);
    std::vector<std::string> rows;
    for (std::string line; std::getline(in, line);) {
        if (!line.empty() && line[0] != '#') rows.push_back(line);
    }
    return rows;
}

int average(const std::vector<std::string> & words) {
    cli::Args args;
    std::string error;
    EXPECT_TRUE(cli::parse(*cli::command("post average"), words, args, error)) << error;
    std::ostringstream report, messages;
    std::streambuf * out = std::cout.rdbuf(report.rdbuf());
    std::streambuf * err = std::cerr.rdbuf(messages.rdbuf());
    const int status = post_average(args);
    std::cout.rdbuf(out);
    std::cerr.rdbuf(err);
    EXPECT_EQ(messages.str(), "");
    comm::barrier();
    return status;
}

} // namespace

TEST(ConditionalStatisticsTest, PdfsAndConditionalMomentsAreThoseOfTheSampledCells) {
    // One sample, at the last step: the PDFs, means and variances are those of the final state's cells, in
    // bins of 8 (each team bins in scratch memory) and 64 x 48 (too many for it: global sums), by volume and
    // by mass, the cells outside the ranges in the total only
    const std::string dir = (rank_scratch_dir() / "conditional_one_sample").string();
    const uint32_t P = 1 + N_DIM, T = 2 + N_DIM, U_X = 1, U_Y = 2;
    for (const std::string weight : {"volume", "mass"}) {
        Solver solver;
        solver.set_distributed(false);
        solver.init(parse_toml(flow_case(3, "[statistics]\nt_start = 0.025\n" + two_entries(dir, weight))));
        solver.run();
        solver.copy_device_to_host();
        const auto & conditional = solver.get_statistics().get_conditional();
        ASSERT_EQ(conditional.size(), 2u);
        const bool mass = weight == "mass";
        const auto rho = conditional[0].result();
        EXPECT_EQ(rho.samples, 1u);
        EXPECT_GT(rho.outside, 0.0);
        expect_near_result(rho, expected_of_one_sample(solver, rho, {0}, {P, U_X}, mass), weight + " rho");
        const auto joint = conditional[1].result();
        expect_near_result(joint, expected_of_one_sample(solver, joint, {0, P}, {T, U_Y}, mass), weight + " joint");
    }
}

TEST(ConditionalStatisticsTest, SamplesAreWeightedByTheTimeSinceThePreviousOne) {
    // A uniform state: every sample falls in one bin, so its weight is the domain's volume times the window,
    // t_start to the last sample, and the variance of a constant field is zero
    const std::string extra = "[statistics]\nt_start = 0.015\ninterval = 2\n"
                              "[[statistics.conditional]]\non = \"P\"\nbins = 4\nrange = [0.5, 1.5]\nfields = [\"RHO\"]\n";
    std::string input = flow_case(7, extra);
    const size_t at = input.find("[initialize]");
    input = input.substr(0, at) + "[initialize]\ntype = \"constant\"\nT = 1.0\np = 1.0\nu = " +
            vector_of("0.0", "0.0", "0.0") + "\n" + input.substr(input.find("[numerics]"));
    Solver solver;
    solver.set_distributed(false);
    solver.init(parse_toml(input));
    solver.run();
    const auto r = solver.get_statistics().get_conditional()[0].result();
    EXPECT_EQ(r.samples, 3u);  // steps 2, 4, 6
    EXPECT_DOUBLE_EQ(r.t_first, double(rtype(0.015)));
    EXPECT_NEAR(r.t_last, 0.06, roundoff(1e-16));
    const double volume = 1.0;
    EXPECT_NEAR(r.total, volume * (0.06 - 0.015), roundoff(1e-14));
    EXPECT_NEAR(r.weight[2], r.total, 1e-14);
    EXPECT_NEAR(r.pdf[2], 4.0, 1e-12);
    EXPECT_NEAR(r.mean[2], 1.0, 1e-14);
    EXPECT_NEAR(r.variance[2], 0.0, 1e-14);
}

TEST(ConditionalStatisticsTest, ARestartContinuesThemBitwise) {
    const std::string dir = fresh_dir("restart");
    const auto run = [&](const std::string & sub, uint32_t n_steps, const std::string & init) {
        std::string input = flow_case(n_steps, "[statistics]\nt_start = 0.005\ninterval = 2\n" +
                                                   two_entries(dir + "/" + sub, "mass") +
                                                   "[[write_data]]\nprefix = \"" + dir + "/" + sub +
                                                   "/r\"\nformat = \"restart\"\ninterval = 3\n");
        if (!init.empty()) {
            input = input.substr(0, input.find("[initialize]")) + "[initialize]\n" + init +
                    input.substr(input.find("[numerics]"));
        }
        auto solver = std::make_unique<Solver>();
        solver->init(parse_toml(input));
        solver->run();
        return solver;
    };
    const auto whole = run("whole", 8, "");
    run("first", 3, "");
    const auto rest = run("rest", 8, "type = \"restart\"\nfile = \"" + dir + "/first/r_000003.restart\"\n");
    for (size_t e = 0; e < 2; e++) {
        const auto a = whole->get_statistics().get_conditional()[e].result();
        const auto b = rest->get_statistics().get_conditional()[e].result();
        EXPECT_EQ(a.samples, 4u);
        EXPECT_TRUE(same_bits(a, b)) << "entry " << e;
    }
}

TEST(ConditionalStatisticsTest, TheSameBitForBitOnAnyNumberOfRanks) {
    // Exact fixed-point sums: a distributed run (NCCL builds: with NCCL's reductions and with MPI's) holds the
    // serial run's results bit for bit
    if (comm::size() == 1) GTEST_SKIP() << "needs several ranks";
    const std::string dir = fresh_dir("ranks");
    const std::string input = flow_case(6, "[statistics]\nt_start = 0.005\n" + two_entries(dir, "mass"));
    std::vector<std::string> inputs = {with_halo_exchange(input, "mpi")};
    if (comm::nccl_available()) inputs.push_back(with_halo_exchange(input, "nccl"));
    Solver serial;
    serial.set_distributed(false);
    serial.init(parse_toml(flow_case(6, "[statistics]\nt_start = 0.005\n" +
                                            two_entries((rank_scratch_dir() / "conditional_serial").string(), "mass"))));
    serial.run();
    for (const std::string & text : inputs) {
        Solver distributed;
        distributed.init(parse_toml(text));
        distributed.run();
        for (size_t e = 0; e < 2; e++) {
            EXPECT_TRUE(same_bits(distributed.get_statistics().get_conditional()[e].result(),
                                  serial.get_statistics().get_conditional()[e].result()))
                << "entry " << e << (text.find("\"nccl\"") != std::string::npos ? " (NCCL)" : " (MPI)");
        }
    }
}

TEST(ConditionalStatisticsTest, HeatReleaseOnTemperatureIsTheCellsHeatRelease) {
    // A reacting box with a temperature ramp, sampled once: the conditional means of HRR and OMEGA_H2O on T,
    // and MEAN_HRR, are those of the rates the run writes
    std::ostringstream s;
    s << "[run]\nn_steps = 1\ndt = 1.0e-8\n[mesh]\ntype = \"cartesian\"\nNx = 16\nNy = 4\nLx = 1.0e-3\nLy = 1.0e-3\n";
    if (N_DIM == 3) s << "Nz = 3\nLz = 1.0e-3\n";
    s << "[[periodic]]\nzones = [\"left\", \"right\"]\n[[periodic]]\nzones = [\"bottom\", \"top\"]\n";
    if (N_DIM == 3) s << "[[periodic]]\nzones = [\"back\", \"front\"]\n";
    s << "[initialize]\ntype = \"analytical\"\np = \"101325.0\"\nT = \"1100.0 + 400.0 * sin(pi * x / 1.0e-3)\"\n"
      << "u = " << vector_of("\"0.0\"", "\"0.0\"", "\"0.0\"") << "\nX = { H2 = 2.0, O2 = 1.0, N2 = 3.76 }\n"
      << "[numerics]\nriemann_solver = \"HLLC\"\n[numerics.face_reconstruction]\ntype = \"FO\"\n"
      << "[physics]\ntype = \"euler\"\ngas = \"mixture\"\nmechanism = \"" << source_dir() << "/mechanisms/h2o2.yaml\"\n"
      << "[chemistry]\n[output]\ncheck_interval = 1000000\nrun_log = \"\"\n"
      << "[statistics]\nfields = [\"HRR\"]\n"
      << "[[statistics.conditional]]\non = \"T\"\nbins = 5\nrange = [1100.0, 1500.0]\nfields = [\"HRR\", \"OMEGA_H2O\"]\n";
    Solver solver;
    solver.set_distributed(false);
    std::vector<Data> rates;  // the HRR and OMEGA_H2O the run writes
    solver.set_output_data([&](const Mesh &, std::vector<Data> & data) {
        for (const char * name : {"HRR", "OMEGA_H2O"}) {
            for (const Data & d : data) {
                if (d.name() == name) rates.push_back(d);
            }
        }
    });
    solver.init(parse_toml(s.str()));
    solver.run();
    solver.copy_device_to_host();
    const Mesh & mesh = *solver.get_mesh();
    ASSERT_EQ(rates.size(), 2u);
    const auto r = solver.get_statistics().get_conditional()[0].result();
    std::vector<double> W(5, 0.0), hrr(5, 0.0), omega(5, 0.0);
    for (uint32_t c = 0; c < mesh.n_owned(); c++) {
        const double T = double(solver.h_primitives(c, N_DIM + 1));
        const double y = (T - 1100.0) * 5.0 / 400.0;
        const size_t b = y == 5.0 ? 4 : static_cast<size_t>(std::floor(y));
        const double w = double(mesh.h_cell_measure(c));
        W[b] += w;
        hrr[b] += w * double(rates[0][c]);
        omega[b] += w * double(rates[1][c]);
        EXPECT_EQ(solver.get_statistics().host_means()(c, 0), rates[0][c]);
    }
    double peak = 0.0;
    for (size_t b = 0; b < 5; b++) peak = std::max(peak, std::abs(hrr[b] / W[b]));
    EXPECT_GT(peak, 0.0);
    for (size_t b = 0; b < 5; b++) {
        ASSERT_GT(W[b], 0.0);
        EXPECT_NEAR(r.mean[b * 2], hrr[b] / W[b], 1e-11 * peak) << "bin " << b;
        EXPECT_NEAR(r.mean[b * 2 + 1], omega[b] / W[b], 1e-11 * std::abs(omega[b] / W[b]) + 1e-300) << "bin " << b;
    }
}

TEST(ConditionalStatisticsTest, PostAverageOfRestartFilesGivesTheRunsBitwise) {
    // mallard post average --conditional over the restart files of the run's sampling steps, from its t_start,
    // writes the run's conditional statistics file to the digit
    const std::string dir = fresh_dir("post");
    const std::string entry = "{on = \"RHO\", bins = 8, range = [0.7, 1.3], fields = [\"P\", \"U_X\"], weight = \"mass\"}";
    {
        Solver solver;
        solver.init(parse_toml(flow_case(6, "[statistics]\nt_start = 0.01\ninterval = 2\n"
                                            "[[statistics.conditional]]\non = \"RHO\"\nbins = 8\nrange = [0.7, 1.3]\n"
                                            "fields = [\"P\", \"U_X\"]\nweight = \"mass\"\nfile = \"" + dir +
                                            "/run.csv\"\n[[write_data]]\nprefix = \"" + dir +
                                            "/restart\"\nformat = \"restart\"\ninterval = 2\n")));
        solver.run();
    }
    comm::barrier();
    ASSERT_EQ(average({dir + "/restart_000002.restart", dir + "/restart_000004.restart", dir + "/restart_000006.restart",
                       "--t-start", "0.01", "--conditional", entry, "-o", dir + "/post"}),
              0);
    const std::vector<std::string> run = data_rows(dir + "/run.csv");
    const std::vector<std::string> post = data_rows(dir + "/post/conditional_RHO.csv");
    ASSERT_EQ(run.size(), 1u + 8u);  // header, bins
    EXPECT_EQ(post, run);
}

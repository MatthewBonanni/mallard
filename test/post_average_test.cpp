/**
 * @file post_average_test.cpp
 * @brief mallard post average: statistics over saved snapshots are the run's [statistics] on the same samples.
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 */

#include <gtest/gtest.h>

#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

#include "cli.h"
#include "comm.h"
#include "commands.h"
#include "mesh_block.h"
#include "output_reader.h"
#include "scratch.h"
#include "solver.h"
#include "test_fixtures.h"
#include "test_utils.h"

namespace {

namespace fs = std::filesystem;

std::string fresh_dir(const std::string & name) {
    const std::string dir = (scratch_dir() / ("average_" + name)).string();
    comm::barrier();
    if (comm::is_root()) fs::remove_all(dir);
    comm::barrier();
    return dir;
}

std::string vector_of(const char * x, const char * y, const char * z) {
    return N_DIM == 3 ? std::string("[") + x + ", " + y + ", " + z + "]" : std::string("[") + x + ", " + y + "]";
}

const char * const FIELDS = "RHO,U_X,U_Y,P,MACH";
const char * const PRODUCTS = "U_X*U_X,U_X*U_Y,RHO*U_X,P*MACH";

/**
 * @brief A periodic box with [statistics] from t = 1e-2 every 2 steps, writing restart files and
 *        a VTU snapshot of the averages at those steps; the run of it, in dir. Its cells are
 *        numbered along the Hilbert curve, unlike the mesh --statistics reads the snapshot on.
 */
void run_case(const std::string & dir, bool axisymmetric = false) {
    std::ostringstream s;
    s << "[run]\nn_steps = 6\ndt = 1.0e-2\n[mesh]\ntype = \"cartesian\"\nNx = 8\nNy = 8\nLx = 1.0\nLy = 1.0\n"
      << "cell_order = \"hilbert\"\n";
    if (N_DIM == 3) s << "Nz = 4\nLz = 1.0\n[[periodic]]\nzones = [\"back\", \"front\"]\n";
    s << "[[periodic]]\nzones = [\"left\", \"right\"]\n";
    if (axisymmetric) {
        s << "[[boundaries]]\nname = \"bottom\"\ntype = \"symmetry\"\n[[boundaries]]\nname = \"top\"\ntype = \"symmetry\"\n";
    } else {
        s << "[[periodic]]\nzones = [\"bottom\", \"top\"]\n";
    }
    s
      << "[initialize]\ntype = \"analytical\"\nrho = \"1.0 + 0.2 * sin(2 * pi * x) * sin(2 * pi * y)\"\np = \"1.0\"\n"
      << "u = " << vector_of("\"0.3 + 0.1 * sin(2 * pi * y)\"", "\"-0.1 + 0.1 * sin(2 * pi * x)\"", "\"0.0\"")
      << "\n[numerics]\nriemann_solver = \"HLLC\"\ntime_integrator = \"SSPRK3\"\n"
      << "[numerics.face_reconstruction]\ntype = \"TENO\"\n"
      << "[physics]\ntype = \"euler\"\ngamma = 1.4\np_ref = 1.0\nT_ref = 1.0\nrho_ref = 1.0\n"
      << (axisymmetric ? "axisymmetric = true\n" : "")
      << "[statistics]\nfields = [\"RHO\", \"U_X\", \"U_Y\", \"P\", \"MACH\"]\n"
      << "products = [\"U_X*U_X\", \"U_X*U_Y\", \"RHO*U_X\", \"P*MACH\"]\nt_start = 1.0e-2\ninterval = 2\n"
      << "[output]\ncheck_interval = 1000000\nrun_log = \"\"\n"
      << "[[write_data]]\nprefix = \"" << dir << "/restart\"\nformat = \"restart\"\ninterval = 2\n"
      << "[[write_data]]\nprefix = \"" << dir << "/stats\"\nformat = \"" << (have_hdf5() ? "vtkhdf" : "vtu")
      << "\"\nfields = [\"MEAN_*\", \"COV_*\"]\n"
      << "interval = 2\n"
      << "[[write_data]]\nprefix = \"" << dir << "/flow\"\nformat = \"vtu\"\nfields = [\"MACH\"]\ninterval = 2\n";
    {
        Solver solver;
        solver.init(parse_toml(s.str()));
        solver.run();
    }
    comm::barrier();
}

int post(const char * command, int (*run)(const cli::Args &), const std::vector<std::string> & words,
         std::string * errors = nullptr) {
    cli::Args args;
    std::string error;
    EXPECT_TRUE(cli::parse(*cli::command(command), words, args, error)) << error;
    std::ostringstream report, messages;
    std::streambuf * out = std::cout.rdbuf(report.rdbuf());
    std::streambuf * err = std::cerr.rdbuf(messages.rdbuf());
    const int status = run(args);
    std::cout.rdbuf(out);
    std::cerr.rdbuf(err);
    if (errors) *errors = messages.str();
    comm::barrier();
    return status;
}

int average(const std::vector<std::string> & words, std::string * errors = nullptr) {
    return post("post average", post_average, words, errors);
}

/** @brief The index of the run's snapshots of its [statistics]. */
std::string stats_index(const std::string & dir) { return dir + (have_hdf5() ? "/stats.vtkhdf.series" : "/stats.pvd"); }

std::vector<std::string> restarts(const std::string & dir) {
    return {dir + "/restart_000002.restart", dir + "/restart_000004.restart", dir + "/restart_000006.restart"};
}

/** @brief The columns of a CSV file with a header line, by name. */
std::map<std::string, std::vector<double>> read_csv(const std::string & path) {
    std::ifstream in(path);
    std::string line;
    while (std::getline(in, line) && line.starts_with("#")) {}
    std::vector<std::string> names;
    std::stringstream header(line);
    for (std::string name; std::getline(header, name, ',');) names.push_back(name);
    std::map<std::string, std::vector<double>> columns;
    while (std::getline(in, line)) {
        if (line.starts_with("#")) continue;
        std::stringstream row(line);
        std::string value;
        for (size_t j = 0; j < names.size() && std::getline(row, value, ','); j++) columns[names[j]].push_back(std::stod(value));
    }
    return columns;
}

} // namespace

TEST(PostAverageTest, RestartFilesGiveTheRunsStatisticsBitForBit) {
    // The run samples steps 2, 4 and 6 from t_start: the restart files of those steps, from the same t_start
    const std::string run = fresh_dir("run");
    run_case(run);
    const std::string out = fresh_dir("bitwise");
    std::vector<std::string> words = restarts(run);
    words.insert(words.end(), {"--fields", FIELDS, "--products", PRODUCTS, "--t-start", "1.0e-2", "--snapshot", "vtu",
                               "-o", out});
    ASSERT_EQ(average(words), 0);
    const output::Contents written = output::open(stats_index(run));
    const output::Contents averaged = output::open(out + "/average.pvd");
    ASSERT_EQ(averaged.snapshots.size(), 1u);
    const size_t last = written.snapshots.size() - 1;
    EXPECT_EQ(averaged.snapshots[0].t, written.snapshots[last].t);
    for (const char * name : {"MEAN_RHO", "MEAN_U_X", "MEAN_U_Y", "MEAN_P", "MEAN_MACH", "COV_U_X_U_X", "COV_U_X_U_Y",
                              "COV_RHO_U_X", "COV_P_MACH"}) {
        const std::vector<double> a = output::read(written, name, last), b = output::read(averaged, name, 0);
        ASSERT_EQ(a.size(), b.size()) << name;
        size_t differ = 0;
        for (size_t i = 0; i < a.size(); i++) differ += a[i] != b[i];
        EXPECT_EQ(differ, 0u) << name;
    }
    // MEAN_MACH is the time-weighted mean of the MACH the run writes at the samples
    const output::Contents flow = output::open(run + "/flow.pvd");
    const std::vector<double> mean_mach = output::read(averaged, "MEAN_MACH", 0);
    std::vector<double> expected(mean_mach.size(), 0.0);
    double t_last = 1.0e-2, total = 0.0;
    for (size_t k = 1; k < flow.snapshots.size(); k++) {
        const double w = flow.snapshots[k].t - t_last;
        t_last = flow.snapshots[k].t;
        total += w;
        const std::vector<double> mach = output::read(flow, "MACH", k);
        for (size_t i = 0; i < mach.size(); i++) expected[i] += w * mach[i];
    }
    for (size_t i = 0; i < expected.size(); i++) EXPECT_NEAR(mean_mach[i], expected[i] / total, roundoff(1e-15));
    // RMS_ is the root of the variance
    const std::vector<double> var = output::read(averaged, "COV_U_X_U_X", 0), rms = output::read(averaged, "RMS_U_X", 0);
    for (size_t i = 0; i < var.size(); i++) EXPECT_EQ(rtype(rms[i]), std::sqrt(rtype(var[i])));
}

TEST(PostAverageTest, FavreAveragesWeighTheSamplesByDensityExactly) {
    const std::string run = fresh_dir("favre_run");
    run_case(run);
    const std::string out = fresh_dir("favre");
    std::vector<std::string> words = restarts(run);
    words.insert(words.end(), {"--fields", "U_X,U_Y", "--products", "U_X*U_Y", "--favre", "--t-start", "1.0e-2",
                               "--snapshot", "vtu", "-o", out});
    ASSERT_EQ(average(words), 0);
    // sum w rho a / sum w rho, and the covariance about those means: the triple correlation included
    double t_last = 1.0e-2;
    std::vector<std::vector<double>> rho, ux, uy;
    std::vector<double> w;
    for (const std::string & file : restarts(run)) {
        const output::Contents c = output::open(file);
        rho.push_back(output::read(c, "RHO", 0));
        const std::vector<double> mu = output::read(c, "RHOU_X", 0), mv = output::read(c, "RHOU_Y", 0);
        ux.emplace_back(mu.size());
        uy.emplace_back(mv.size());
        for (size_t i = 0; i < mu.size(); i++) {
            ux.back()[i] = mu[i] / rho.back()[i];
            uy.back()[i] = mv[i] / rho.back()[i];
        }
        w.push_back(c.snapshots[0].t - t_last);
        t_last = c.snapshots[0].t;
    }
    const output::Contents averaged = output::open(out + "/average.pvd");
    const std::vector<double> fx = output::read(averaged, "FAVRE_MEAN_U_X", 0), fy = output::read(averaged, "FAVRE_MEAN_U_Y", 0);
    const std::vector<double> fxy = output::read(averaged, "FAVRE_COV_U_X_U_Y", 0);
    const std::vector<double> reynolds = output::read(averaged, "MEAN_U_X", 0);
    const double mean_tolerance = roundoff(1e-15), cov_tolerance = sizeof(rtype) == sizeof(double) ? 2e-16 : 2.5e-10;
    double largest_difference = 0.0, largest_triple = 0.0;
    for (size_t i = 0; i < fx.size(); i++) {
        double sw = 0.0, sx = 0.0, sy = 0.0, sxy = 0.0;
        for (size_t k = 0; k < w.size(); k++) {
            const double m = w[k] * rho[k][i];
            sw += m;
            sx += m * ux[k][i];
            sy += m * uy[k][i];
            sxy += m * ux[k][i] * uy[k][i];
        }
        const double ex = sx / sw, ey = sy / sw;
        EXPECT_NEAR(fx[i], ex, mean_tolerance);
        EXPECT_NEAR(fy[i], ey, mean_tolerance);
        EXPECT_NEAR(fxy[i], sxy / sw - ex * ey, cov_tolerance);
        largest_difference = std::max(largest_difference, std::abs(fx[i] - reynolds[i]));
        double t = 0.0, r = 0.0, u = 0.0, v = 0.0, triple = 0.0;
        for (size_t k = 0; k < w.size(); k++) {
            t += w[k];
            r += w[k] * rho[k][i];
            u += w[k] * ux[k][i];
            v += w[k] * uy[k][i];
        }
        for (size_t k = 0; k < w.size(); k++) {
            triple += w[k] * (rho[k][i] - r / t) * (ux[k][i] - u / t) * (uy[k][i] - v / t);
        }
        largest_triple = std::max(largest_triple, std::abs(triple / r));
    }
    // The density varies, so Favre and Reynolds means differ, and by more than the tolerance
    // the triple correlation <rho'u'v'> / <rho>, which Favre covariances from Reynolds statistics leave out
    EXPECT_GT(largest_difference, 10 * mean_tolerance);
    EXPECT_GT(largest_triple, 10 * cov_tolerance);
}

TEST(PostAverageTest, HomogeneousAveragesOfStatesAndOfTheRunsStatisticsAgree) {
    if (!have_hdf5()) GTEST_SKIP() << "--statistics reads VTKHDF snapshots, which need HDF5";
    const std::string run = fresh_dir("profile_run");
    run_case(run);
    // Over x: profiles in y, from the restart files and from the run's own statistics snapshot
    const std::string states = fresh_dir("profile_states");
    std::vector<std::string> words = restarts(run);
    words.insert(words.end(), {"--fields", FIELDS, "--products", PRODUCTS, "--t-start", "1.0e-2", "--homogeneous",
                               N_DIM == 3 ? "x,z" : "x", "-o", states});
    ASSERT_EQ(average(words), 0);
    const std::string stored = fresh_dir("profile_stored");
    ASSERT_EQ(average({stats_index(run), "--latest", "--statistics", "--homogeneous", N_DIM == 3 ? "x,z" : "x",
                       "-o", stored}),
              0);
    if (!comm::is_root()) return;
    const auto a = read_csv(states + "/average.csv"), b = read_csv(stored + "/average.csv");
    ASSERT_EQ(a.at("y").size(), 8u);
    for (const double n : a.at("n_cells")) EXPECT_EQ(n, N_DIM == 3 ? 32.0 : 8.0);
    for (const char * name : {"y", "MEAN_U_X", "COV_U_X_U_X", "COV_U_X_U_Y", "COV_P_MACH", "RMS_U_X"}) {
        ASSERT_EQ(b.at(name).size(), a.at(name).size()) << name;
        for (size_t r = 0; r < a.at(name).size(); r++) EXPECT_EQ(a.at(name)[r], b.at(name)[r]) << name << " row " << r;
    }
    // Equal rows of a uniform mesh: the profile's mean is the domain mean, and its variance
    // adds the dispersive part, the variation of the time means over x
    const output::Contents stats = output::open(stats_index(run));
    const size_t last = stats.snapshots.size() - 1;
    const std::vector<double> mean = output::read(stats, "MEAN_U_X", last), cov = output::read(stats, "COV_U_X_U_X", last);
    double domain_mean = 0.0, domain_second = 0.0, profile_mean = 0.0, profile_second = 0.0;
    for (size_t i = 0; i < mean.size(); i++) {
        domain_mean += mean[i] / double(mean.size());
        domain_second += (cov[i] + mean[i] * mean[i]) / double(mean.size());
    }
    const auto & pm = a.at("MEAN_U_X");
    const auto & pc = a.at("COV_U_X_U_X");
    for (size_t r = 0; r < pm.size(); r++) {
        profile_mean += pm[r] / double(pm.size());
        profile_second += (pc[r] + pm[r] * pm[r]) / double(pm.size());
    }
    EXPECT_NEAR(profile_mean, domain_mean, 1e-14);
    EXPECT_NEAR(profile_second, domain_second, 1e-14);
}

TEST(PostAverageTest, AxisymmetricProfilesOfStatesAndOfTheRunsStatisticsAgree) {
    if constexpr (N_DIM == 3) GTEST_SKIP() << "axisymmetric runs are 2D";
    if (!have_hdf5()) GTEST_SKIP() << "--statistics reads VTKHDF snapshots, which need HDF5";
    const std::string run = fresh_dir("axisymmetric_run");
    run_case(run, true);
    // Over x: rows at the centroids of the volumes of revolution; over y (the radius): weighted by those volumes
    for (const char * direction : {"x", "y"}) {
        SCOPED_TRACE(direction);
        const std::string states = fresh_dir(std::string("axisymmetric_states_") + direction);
        std::vector<std::string> words = restarts(run);
        words.insert(words.end(), {"--fields", FIELDS, "--products", PRODUCTS, "--t-start", "1.0e-2", "--homogeneous",
                                   direction, "-o", states});
        ASSERT_EQ(average(words), 0);
        const std::string stored = fresh_dir(std::string("axisymmetric_stored_") + direction);
        ASSERT_EQ(average({stats_index(run), "--latest", "--statistics", "--homogeneous", direction, "-o", stored}), 0);
        if (!comm::is_root()) continue;
        const auto a = read_csv(states + "/average.csv"), b = read_csv(stored + "/average.csv");
        const char * kept = direction[0] == 'x' ? "y" : "x";
        ASSERT_EQ(a.at(kept).size(), 8u);
        for (const char * name : {kept, "MEAN_RHO", "MEAN_U_X", "COV_U_X_U_X", "COV_P_MACH", "RMS_U_X"}) {
            ASSERT_EQ(b.at(name).size(), a.at(name).size()) << name;
            for (size_t r = 0; r < a.at(name).size(); r++) EXPECT_EQ(a.at(name)[r], b.at(name)[r]) << name << " row " << r;
        }
    }
}

TEST(PostAverageTest, ChoicesItCannotAverageAreUsageErrors) {
    const std::string run = fresh_dir("usage_run");
    run_case(run);
    EXPECT_EQ(average({run + "/restart_000006.restart", "--fields", "U_X", "-o", fresh_dir("one")}), 2);
    EXPECT_EQ(average({run + "/restart_000004.restart", run + "/restart_000006.restart", "--fields", "MAHC"}), 2);
    std::string errors;
    EXPECT_EQ(average({run + "/restart_000006.restart", run + "/restart_000004.restart", "--fields", "U_X", "-o",
                       fresh_dir("order")},
                      &errors),
              1);
    if (comm::is_root()) {
        EXPECT_NE(errors.find("increasing time"), std::string::npos) << errors;
    }
}

TEST(PostAverageTest, RegionsOfTheSolutionAverageTheirIntegralsAsTheyAreInEachSnapshot) {
    // The mean over the snapshots, each weighing the time since the previous one, of what
    // post extract reduces over the region on each snapshot's own state
    const std::string run = fresh_dir("solution_run");
    run_case(run);
    const std::string dense = "{type = \"where\", condition = \"RHO > 1.1\"}";
    const std::string front = "{type = \"isosurface\", field = \"RHO\", value = 1.1}";
    for (const auto & [name, region] : {std::pair{"dense", dense}, std::pair{"front", front}}) {
        SCOPED_TRACE(name);
        const std::string extracted = fresh_dir(std::string("solution_extract_") + name);
        std::vector<std::string> words = restarts(run);
        words.insert(words.end(), {"--region", region, "--fields", "RHO,P", "--reductions", "integral,mean", "--name",
                                   name, "-o", extracted});
        ASSERT_EQ(post("post extract", post_extract, words), 0);
        const std::string averaged = fresh_dir(std::string("solution_average_") + name);
        words = restarts(run);
        words.insert(words.end(), {"--region", region, "--fields", "RHO,P", "--t-start", "1.0e-2", "--name", name,
                                   "-o", averaged});
        ASSERT_EQ(average(words), 0);
        if (!comm::is_root()) continue;
        const auto each = read_csv(extracted + "/probe_" + name + "_reductions.csv");
        const auto mean = read_csv(averaged + "/" + name + "_reductions.csv");
        const std::string measure = std::string(name) == "dense" ? (N_DIM == 2 ? "area" : "volume")
                                                                 : (N_DIM == 2 ? "length" : "area");
        ASSERT_EQ(each.at("t").size(), 3u);
        ASSERT_EQ(mean.at("t").size(), 1u);
        EXPECT_EQ(mean.at("t")[0], each.at("t")[2]);
        double t_last = 1.0e-2, total = 0.0, m = 0.0, rho = 0.0, p = 0.0;
        for (size_t k = 0; k < 3; k++) {
            const double w = each.at("t")[k] - t_last;
            t_last = each.at("t")[k];
            total += w;
            m += w * each.at(measure)[k];
            rho += w * each.at("integral_RHO")[k];
            p += w * each.at("integral_P")[k];
        }
        // The region changes between snapshots
        EXPECT_NE(each.at(measure)[0], each.at(measure)[2]);
        const double tolerance = roundoff(1e-13);
        EXPECT_NEAR(mean.at(measure)[0], m / total, tolerance * m / total);
        EXPECT_NEAR(mean.at("integral_MEAN_RHO")[0], rho / total, tolerance * rho / total);
        EXPECT_NEAR(mean.at("integral_MEAN_P")[0], p / total, tolerance * p / total);
        EXPECT_NEAR(mean.at("mean_MEAN_RHO")[0], rho / m, tolerance * rho / m);
    }
    // Their averages are integrals over the region, not fields on cells
    std::string errors;
    std::vector<std::string> words = restarts(run);
    words.insert(words.end(), {"--region", dense, "--snapshot", "vtu", "-o", fresh_dir("solution_snapshot")});
    EXPECT_EQ(average(words, &errors), 2);
    if (comm::is_root()) EXPECT_NE(errors.find("defined by the solution"), std::string::npos) << errors;
}

TEST(PostAverageTest, RegionsOfTheSolutionAverageProductsFavreAndProfilesAsTheyAreInEachSnapshot) {
    // Against the mean over the snapshots, each weighing the time since the previous one, of what post
    // extract reduces over the region on each snapshot's own state: integral of U_X^2 = rms^2 times the
    // measure, of RHO U_X = of RHOU_X, of RHO U_X for Favre = favre times the integral of RHO
    const std::string run = fresh_dir("moments_run");
    run_case(run);
    const std::string measure = N_DIM == 2 ? "area" : "volume";
    const std::string dense = "{type = \"where\", condition = \"RHO > 1.1\"}";
    const auto extracted = [&](const std::string & name, const std::string & region) {
        const std::string dir = fresh_dir("moments_extract_" + name);
        std::vector<std::string> words = restarts(run);
        words.insert(words.end(), {"--region", region, "--fields", "RHO,U_X,RHOU_X", "--reductions",
                                   "integral,favre,rms", "--name", name, "-o", dir});
        EXPECT_EQ(post("post extract", post_extract, words), 0);
        return comm::is_root() ? read_csv(dir + "/probe_" + name + "_reductions.csv")
                               : std::map<std::string, std::vector<double>>{};
    };
    const auto averaged = [&](const std::string & name, const std::string & region, const std::string & homogeneous) {
        const std::string dir = fresh_dir("moments_average_" + name);
        std::vector<std::string> words = restarts(run);
        words.insert(words.end(), {"--region", region, "--fields", "RHO,U_X", "--products", "U_X*U_X,RHO*U_X",
                                   "--favre", "--t-start", "1.0e-2", "--name", name, "-o", dir});
        if (!homogeneous.empty()) words.insert(words.end(), {"--homogeneous", homogeneous});
        EXPECT_EQ(average(words), 0);
        return comm::is_root() ? read_csv(dir + "/" + name + (homogeneous.empty() ? "_reductions.csv" : ".csv"))
                               : std::map<std::string, std::vector<double>>{};
    };
    // The time-weighted mean of f(k) over the snapshots k
    const auto in_time = [](const std::map<std::string, std::vector<double>> & each, const auto & f) {
        double t_last = 1.0e-2, total = 0.0, sum = 0.0;
        for (size_t k = 0; k < each.at("t").size(); k++) {
            const double w = each.at("t")[k] - t_last;
            t_last = each.at("t")[k];
            total += w;
            sum += w * f(k);
        }
        return sum / total;
    };
    const double tolerance = roundoff(1e-12);
    const auto expect_near = [&](double value, double expected, double scale) {
        EXPECT_NEAR(value, expected, tolerance * scale);
    };

    const auto each = extracted("dense", dense);
    const auto mean = averaged("dense", dense, "");
    // A region of the mesh combined with one of the solution: here the same cells
    const auto combined = averaged("combined", "{type = \"intersection\", regions = [\"domain\", " + dense + "]}", "");
    if (comm::is_root()) {
        ASSERT_EQ(each.at("t").size(), 3u);
        const auto e = [&](const char * column) { return [&each, column](size_t k) { return each.at(column)[k]; }; };
        const double m = in_time(each, e(measure.c_str())), rho = in_time(each, e("integral_RHO"));
        const double u = in_time(each, e("integral_U_X")) / m;
        const double uu = in_time(each, [&](size_t k) { return std::pow(each.at("rms_U_X")[k], 2) * each.at(measure)[k]; });
        const double rho_u = in_time(each, e("integral_RHOU_X"));
        const double favre = in_time(each, [&](size_t k) { return each.at("favre_U_X")[k] * each.at("integral_RHO")[k]; });
        expect_near(mean.at("integral_MEAN_U_X_U_X")[0], uu, uu);
        expect_near(mean.at("COV_U_X_U_X")[0], uu / m - u * u, uu / m);
        expect_near(mean.at("RMS_U_X")[0], std::sqrt(uu / m - u * u), std::sqrt(uu / m));
        expect_near(mean.at("integral_MEAN_RHO_U_X")[0], rho_u, std::abs(rho_u));
        expect_near(mean.at("COV_RHO_U_X")[0], rho_u / m - rho / m * u, std::abs(rho_u / m));
        expect_near(mean.at("FAVRE_MEAN_U_X")[0], favre / rho, std::abs(favre / rho));
        EXPECT_TRUE(mean.contains("FAVRE_COV_U_X_U_X"));
        for (const auto & [column, values] : mean) {
            ASSERT_TRUE(combined.contains(column)) << column;
            expect_near(combined.at(column)[0], values[0], std::abs(values[0]) + 1e-300);
        }
    }

    // A profile over y (and z): its row of the cells x in [0.25, 0.375] is the region cut to them
    const std::string strip = "{type = \"intersection\", regions = [" + dense + ", {type = \"half_space\", point = " +
                              vector_of("0.375", "0", "0") + ", normal = " + vector_of("1", "0", "0") +
                              "}, {type = \"half_space\", point = " + vector_of("0.25", "0", "0") + ", normal = " +
                              vector_of("-1", "0", "0") + "}]}";
    const auto cut = extracted("strip", strip);
    const auto profile = averaged("profile", dense, N_DIM == 2 ? "y" : "y,z");
    if (comm::is_root()) {
        size_t row = profile.at("x").size();
        for (size_t r = 0; r < profile.at("x").size(); r++) {
            if (std::abs(profile.at("x")[r] - 0.3125) < 1e-9) row = r;
        }
        ASSERT_LT(row, profile.at("x").size());
        const auto e = [&](const char * column) { return [&cut, column](size_t k) { return cut.at(column)[k]; }; };
        const double m = in_time(cut, e(measure.c_str())), rho = in_time(cut, e("integral_RHO"));
        const double u = in_time(cut, e("integral_U_X")) / m;
        const double uu = in_time(cut, [&](size_t k) { return std::pow(cut.at("rms_U_X")[k], 2) * cut.at(measure)[k]; }) / m;
        const double favre = in_time(cut, [&](size_t k) { return cut.at("favre_U_X")[k] * cut.at("integral_RHO")[k]; });
        const double loose = 1e-10 / tolerance;
        expect_near(profile.at("MEAN_RHO")[row], rho / m, loose * rho / m);
        expect_near(profile.at("MEAN_U_X")[row], u, loose * std::abs(u));
        expect_near(profile.at("COV_U_X_U_X")[row], uu - u * u, loose * uu);
        expect_near(profile.at("FAVRE_MEAN_U_X")[row], favre / rho, loose * std::abs(favre / rho));
    }
    // An isosurface has no cells to profile
    std::string errors;
    std::vector<std::string> words = restarts(run);
    words.insert(words.end(), {"--region", "{type = \"isosurface\", field = \"RHO\", value = 1.1}", "--homogeneous", "y",
                               "-o", fresh_dir("moments_isosurface")});
    EXPECT_EQ(average(words, &errors), 1);
    if (comm::is_root()) EXPECT_NE(errors.find("needs"), std::string::npos) << errors;
}

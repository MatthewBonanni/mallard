/**
 * @file post_signal_test.cpp
 * @brief mallard post signal on synthetic CSV files: the peak of a noisy sine, the autocorrelation of an AR(1)
 *        process, window statistics against [statistics.convergence] and a direct computation, force
 *        coefficients, units, and the errors of a bad column, point or file.
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 */

#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numbers>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#include "cli.h"
#include "comm.h"
#include "commands.h"
#include "json.h"
#include "scratch.h"
#include "statistical_convergence.h"
#include "test_fixtures.h"

namespace {

namespace fs = std::filesystem;

/** @brief Standard normal deviates by Box-Muller from the raw bits of mt19937_64 (the same on every platform). */
class Normal {
    public:
        explicit Normal(uint64_t seed) : engine(seed) {}
        double operator()() {
            const double u1 = (static_cast<double>(engine() >> 11) + 1.0) * 0x1.0p-53;
            const double u2 = static_cast<double>(engine() >> 11) * 0x1.0p-53;
            return std::sqrt(-2.0 * std::log(u1)) * std::cos(2.0 * std::numbers::pi * u2);
        }

    private:
        std::mt19937_64 engine;
};

/** @brief x_i = phi x_{i-1} + e_i, unit innovations, started from the stationary distribution. */
std::vector<double> ar1(double phi, uint64_t n, uint64_t seed) {
    Normal normal(seed);
    std::vector<double> x(n);
    double v = normal() / std::sqrt(1.0 - phi * phi);
    for (auto & xi : x) {
        xi = v;
        v = phi * v + normal();
    }
    return x;
}

/** @brief A CSV table as Mallard writes one: header, units line, a row per time (rows_per_time per time). */
std::string write_table(const std::string & name, const std::vector<std::string> & header,
                        const std::vector<std::string> & units, const std::vector<std::vector<double>> & rows) {
    const fs::path path = scratch_dir() / ("signal_" + name + ".csv");
    std::ofstream out(path);
    for (size_t i = 0; i < header.size(); i++) out << (i ? "," : "") << header[i];
    out << "\n# units: ";
    for (size_t i = 0; i < units.size(); i++) out << (i ? "," : "") << units[i];
    out << "\n" << std::setprecision(std::numeric_limits<double>::max_digits10);
    for (const auto & row : rows) {
        for (size_t i = 0; i < row.size(); i++) out << (i ? "," : "") << row[i];
        out << "\n";
    }
    return path.string();
}

/** @brief A probe file of one point: step, t, then the column, dt apart. */
std::string series_file(const std::string & name, const std::vector<double> & x, double dt,
                        const std::string & units = "m s-1", const std::string & t_units = "s") {
    std::vector<std::vector<double>> rows;
    for (size_t i = 0; i < x.size(); i++) rows.push_back({static_cast<double>(i), dt * static_cast<double>(i), x[i]});
    return write_table(name, {"step", "t", "U_Y"}, {"1", t_units, units}, rows);
}

struct Result {
    int status = 0;
    std::string out, err;
    json::Value json() const { return json::parse(out); }
};

Result signal_cmd(const std::vector<std::string> & words) {
    cli::Args args;
    std::string error;
    EXPECT_TRUE(cli::parse(*cli::command("post signal"), words, args, error)) << error;
    std::ostringstream out, err;
    std::streambuf * o = std::cout.rdbuf(out.rdbuf());
    std::streambuf * e = std::cerr.rdbuf(err.rdbuf());
    Result r;
    r.status = post_signal(args);
    std::cout.rdbuf(o);
    std::cerr.rdbuf(e);
    r.out = out.str();
    r.err = err.str();
    return r;
}

} // namespace

TEST(PostSignalTest, NoisySineGivesItsFrequencyAndAmplitude) {
    if (!comm::is_root()) GTEST_SKIP() << "runs on one rank";
    // 2.37 Hz, between bins of the 0.1 Hz bin width, amplitude 0.8 in noise of standard deviation 0.5, on a mean
    const double dt = 0.01, f0 = 2.37, amplitude = 0.8;
    Normal noise(5);
    std::vector<double> x(40000);
    for (size_t i = 0; i < x.size(); i++) {
        x[i] = 3.0 + amplitude * std::sin(2.0 * std::numbers::pi * f0 * dt * static_cast<double>(i)) + 0.5 * noise();
    }
    const std::string file = series_file("sine", x, dt);
    const Result r = signal_cmd({file, "--column", "U_Y", "--segment", "10", "--strouhal", "2,4", "--format", "json"});
    ASSERT_EQ(r.status, 0) << r.err;
    const json::Value j = r.json();
    EXPECT_EQ(j["schema"].string, "mallard-signal");
    const json::Value & sp = j["spectrum"];
    EXPECT_DOUBLE_EQ(sp["df"].number, 0.1);
    EXPECT_EQ(sp["segment_samples"].number, 1000.0);
    // 40000 samples (none of them a transient), segments of 1000 overlapping by 500
    EXPECT_EQ(sp["segments"].number, 79.0);
    EXPECT_NEAR(sp["peak"]["f"].number, f0, 0.01);
    EXPECT_NEAR(sp["peak"]["St"].number, f0 * 2.0 / 4.0, 0.005);
    // The sine's cross terms with the noise scatter the power of its bins by 2 sqrt(0.0025 / 0.32 / 79) = 1.1%,
    // its amplitude by half that; the noise in them adds 0.0025
    EXPECT_NEAR(sp["peak"]["amplitude"].number, amplitude, 0.02 * amplitude);
    // The PSD integrates to the variance: A^2 / 2 of the sine and 0.25 of the noise
    double power = 0.0;
    for (const auto & p : sp["psd"].array) power += p.number * sp["df"].number;
    EXPECT_NEAR(power, 0.5 * amplitude * amplitude + 0.25, 0.03);
    EXPECT_NEAR(j["statistics"]["mean"].number, 3.0, 0.02);
    EXPECT_NEAR(j["statistics"]["rms"].number, std::sqrt(0.5 * amplitude * amplitude + 0.25), 0.02);
}

TEST(PostSignalTest, AutocorrelationOfAnAR1ProcessIsPhiToTheLag) {
    if (!comm::is_root()) GTEST_SKIP() << "runs on one rank";
    const double phi = 0.8;
    const std::string file = series_file("ar1", ar1(phi, 100000, 3), 0.5);
    const Result r = signal_cmd({file, "--column", "U_Y", "--max-lag", "30", "--format", "json"});
    ASSERT_EQ(r.status, 0) << r.err;
    const json::Value j = r.json();
    const json::Value & ac = j["autocorrelation"];
    ASSERT_EQ(ac["rho"].array.size(), 61u);
    EXPECT_EQ(ac["rho"].array[0].number, 1.0);
    for (size_t k = 1; k <= 20; k++) {
        EXPECT_NEAR(ac["rho"].array[k].number, std::pow(phi, static_cast<double>(k)), 0.02) << k;
        EXPECT_DOUBLE_EQ(ac["t_lag"].array[k].number, 0.5 * static_cast<double>(k));
    }
    // tau = (1 + phi) / (1 - phi) = 9 samples, 4.5 time units
    EXPECT_NEAR(ac["tau"].number, 9.0, 0.6);
    EXPECT_NEAR(j["statistics"]["tau_t"].number, 4.5, 0.3);
    // tau is 1 + 2 sum rho over Sokal's window
    const size_t window = static_cast<size_t>(ac["window"].number);
    ASSERT_GT(window, 0u);
    ASSERT_LE(window, 60u);
    double tau = 1.0;
    for (size_t k = 1; k <= window; k++) tau += 2.0 * ac["rho"].array[k].number;
    EXPECT_DOUBLE_EQ(tau, ac["tau"].number);
}

TEST(PostSignalTest, WindowStatisticsAreThoseOfStatisticsConvergenceAndOfADirectSum) {
    if (!comm::is_root()) GTEST_SKIP() << "runs on one rank";
    // An AR(1) signal on a mean of 10 after a transient decaying from 15 above it
    uint64_t n = 100;  // a sample count at which [statistics.convergence] checks MSER-5: it then checks all of them
    while (n < 20000) n += std::max<uint64_t>(time_series::MSER_BATCH, n / 20);
    std::vector<double> x = ar1(0.8, n, 21);
    for (size_t i = 0; i < x.size(); i++) x[i] += 10.0 + 15.0 * std::exp(-static_cast<double>(i) / 400.0);
    const double dt = 0.01;
    const std::string file = series_file("window", x, dt);
    const Result r = signal_cmd({file, "--column", "U_Y", "--tolerance", "0.01", "--format", "json"});
    ASSERT_EQ(r.status, 0) << r.err;
    const json::Value j = r.json();
    const json::Value & st = j["statistics"];
    ASSERT_TRUE(st["stationary"].boolean);
    const uint64_t i0 = static_cast<uint64_t>(st["transient_samples"].number);
    EXPECT_GE(st["t_stationary"].number, 10.0);
    EXPECT_LE(st["t_stationary"].number, 40.0);
    EXPECT_EQ(st["samples"].number, static_cast<double>(n - i0));

    // The run's [statistics.convergence], fed the same samples, holds the same values bit for bit
    Convergence c;
    c.init(parse_toml("[statistics.convergence]\nsignals = [\"probe_u\"]\ntolerance = 0.01\nfile = \"" +
                      (scratch_dir() / "signal_convergence.csv").string() + "\"\n"),
           {{"probe_u", dim::velocity}}, false);
    for (uint64_t i = 0; i < n; i++) {
        c.add(i, dt * static_cast<double>(i), {{0, x[i]}}, -std::numeric_limits<double>::infinity());
    }
    const auto value = [&](const std::string & name) {
        for (const auto & [key, v] : c.last().values) {
            if (key == name) return v;
        }
        ADD_FAILURE() << name;
        return 0.0;
    };
    EXPECT_EQ(st["t_stationary"].number, value("t_stationary"));
    EXPECT_EQ(st["mean"].number, value("probe_u_mean"));
    EXPECT_EQ(st["mean_ci"].number, value("probe_u_ci"));
    EXPECT_EQ(st["variance"].number, value("probe_u_variance"));
    EXPECT_EQ(st["variance_ci"].number, value("probe_u_variance_ci"));
    EXPECT_EQ(st["converged"].boolean, value("converged") == 1.0);
    EXPECT_NEAR(st["tau"].number, value("probe_u_tau"), 0.05 * value("probe_u_tau"));

    // And a direct computation: mean and variance of the window; batch means of its latest 20 m samples
    const uint64_t w = n - i0, m = w / 20;
    double mean = 0.0, variance = 0.0;
    for (uint64_t i = i0; i < n; i++) mean += x[i] / static_cast<double>(w);
    for (uint64_t i = i0; i < n; i++) variance += (x[i] - mean) * (x[i] - mean) / static_cast<double>(w);
    std::vector<double> batch(20, 0.0);
    for (uint64_t b = 0; b < 20; b++) {
        for (uint64_t i = 0; i < m; i++) batch[b] += x[n - 20 * m + b * m + i] / static_cast<double>(m);
    }
    double bm = 0.0, s2 = 0.0;
    for (const double b : batch) bm += b / 20.0;
    for (const double b : batch) s2 += (b - bm) * (b - bm) / 19.0;
    const double t19 = 2.0930240544083096;  // Student's t, 0.975 quantile, 19 degrees of freedom
    EXPECT_NEAR(st["mean"].number, mean, 1e-12 * std::fabs(mean));
    EXPECT_NEAR(st["variance"].number, variance, 1e-9 * variance);
    EXPECT_NEAR(st["mean_ci"].number, t19 * std::sqrt(s2 / 20.0), 1e-9);
    EXPECT_NEAR(st["rms"].number, std::sqrt(variance), 1e-9);
    EXPECT_EQ(st["batch_size"].number, static_cast<double>(m));
}

TEST(PostSignalTest, UnitsFollowTheFilesUnitsLine) {
    if (!comm::is_root()) GTEST_SKIP() << "runs on one rank";
    std::vector<double> x(400);
    for (size_t i = 0; i < x.size(); i++) x[i] = std::sin(0.3 * static_cast<double>(i)) + 0.01 * static_cast<double>(i % 7);
    const fs::path dir = scratch_dir() / "signal_units";
    {
        const Result r = signal_cmd({series_file("si", x, 0.1, "kg m-1 s-2"), "--column", "U_Y", "-o", dir.string(),
                                     "--format", "json"});
        ASSERT_EQ(r.status, 0) << r.err;
        const json::Value j = r.json();
        EXPECT_EQ(j["units"].string, "kg m-1 s-2");
        EXPECT_EQ(j["statistics"]["units"]["variance"].string, "kg2 m-2 s-4");
        EXPECT_EQ(j["statistics"]["units"]["t"].string, "s");
        EXPECT_EQ(j["spectrum"]["units"]["f"].string, "s-1");
        EXPECT_EQ(j["spectrum"]["units"]["psd"].string, "kg2 m-2 s-3");
        std::ifstream in(dir / "spectrum.csv");
        std::string header, units;
        std::getline(in, header);
        std::getline(in, units);
        EXPECT_EQ(header, "f,PSD_U_Y");
        EXPECT_EQ(units, "# units: s-1,kg2 m-2 s-3");
        std::ifstream stats(dir / "statistics.csv");
        std::getline(stats, header);
        std::getline(stats, units);
        EXPECT_EQ(units, "# units: 1,s,1,kg m-1 s-2,kg m-1 s-2,kg m-1 s-2,kg m-1 s-2,kg2 m-2 s-4,kg2 m-2 s-4,1,1,1");
    }
    {
        // A nondimensional run's units lines say 1: so does everything computed from them
        const Result r = signal_cmd({series_file("nondimensional", x, 0.1, "1", "1"), "--column", "U_Y", "--format", "json"});
        ASSERT_EQ(r.status, 0) << r.err;
        const json::Value j = r.json();
        EXPECT_EQ(j["units"].string, "1");
        EXPECT_EQ(j["statistics"]["units"]["variance"].string, "1");
        EXPECT_EQ(j["spectrum"]["units"]["f"].string, "1");
        EXPECT_EQ(j["spectrum"]["units"]["psd"].string, "1");
    }
}

TEST(PostSignalTest, ForceCoefficientsAlongAndAcrossTheReferenceVelocity) {
    if (!comm::is_root()) GTEST_SKIP() << "runs on one rank";
    // Drag 1.3 and lift 0.3 sin(2 pi 0.1 t) of q L = 0.5 * 1.2 * 5^2 * 0.1 = 1.5 along u = (3, 4)
    const double q = 0.5 * 1.2 * 25.0 * 0.1;
    std::vector<std::vector<double>> rows;
    for (int i = 0; i < 2000; i++) {
        const double t = 0.05 * i, cd = 1.3, cl = 0.3 * std::sin(2.0 * std::numbers::pi * 0.1 * t);
        const double fx = q * (cd * 0.6 - cl * 0.8), fy = q * (cd * 0.8 + cl * 0.6);
        rows.push_back({static_cast<double>(i), t, 0.75 * fx, 0.75 * fy, 0.25 * fx, 0.25 * fy});
    }
    const std::string file = write_table("forces", {"step", "t", "Fx_pressure", "Fy_pressure", "Fx_viscous", "Fy_viscous"},
                                         {"1", "s", "kg s-2", "kg s-2", "kg s-2", "kg s-2"}, rows);
    const std::string reference = "{rho = 1.2, u = [3, 4], length = 0.1}";
    const Result cd = signal_cmd({file, "--column", "CD", "--reference", reference, "--format", "json"});
    ASSERT_EQ(cd.status, 0) << cd.err;
    EXPECT_NEAR(cd.json()["statistics"]["mean"].number, 1.3, 1e-12);
    EXPECT_EQ(cd.json()["units"].string, "1");
    const Result cl = signal_cmd({file, "--column", "CL", "--reference", reference, "--format", "json"});
    ASSERT_EQ(cl.status, 0) << cl.err;
    EXPECT_NEAR(cl.json()["spectrum"]["peak"]["f"].number, 0.1, 1e-3);
    EXPECT_NEAR(cl.json()["spectrum"]["peak"]["amplitude"].number, 0.3, 0.003);

    // 3D forces take an area, and the units must agree with the reference
    const Result wrong = signal_cmd({file, "--column", "CD", "--reference", "{rho = 1.2, u = [3, 4], area = 0.1}"});
    EXPECT_EQ(wrong.status, 1);
    EXPECT_NE(wrong.err.find("--reference.length: is missing"), std::string::npos) << wrong.err;
    const Result typo =
        signal_cmd({file, "--column", "CD", "--reference", "{rho = 1.2, u = [3, 4], length = 0.1, lenght = 1}"});
    EXPECT_EQ(typo.status, 1);
    EXPECT_NE(typo.err.find("--reference.lenght"), std::string::npos) << typo.err;
    const Result no_reference = signal_cmd({file, "--column", "CD"});
    EXPECT_EQ(no_reference.status, 1);
    EXPECT_NE(no_reference.err.find("the coefficients need --reference"), std::string::npos) << no_reference.err;
}

TEST(PostSignalTest, ProbeFilesOfSeveralSamplesTakeAPointAndRestartedRowsReplaceTheirTimes) {
    if (!comm::is_root()) GTEST_SKIP() << "runs on one rank";
    // Three points; point k holds k + sin(t); a restart from t = 10 rewrote the rows from there on
    std::vector<std::vector<double>> rows;
    const auto add = [&](int i, double offset) {
        for (int k = 0; k < 3; k++) rows.push_back({static_cast<double>(i), 0.1 * i, static_cast<double>(k),
                                                    k + offset + std::sin(0.1 * i)});
    };
    for (int i = 0; i < 150; i++) add(i, 100.0 * (i >= 100));
    for (int i = 100; i < 300; i++) add(i, 0.0);
    const std::string file = write_table("points", {"step", "t", "point", "P"}, {"1", "s", "1", "kg m-1 s-2"}, rows);
    const Result none = signal_cmd({file, "--column", "P"});
    EXPECT_EQ(none.status, 1);
    EXPECT_NE(none.err.find("--point: " + file + " has 3 rows per time"), std::string::npos) << none.err;
    const Result r = signal_cmd({file, "--column", "P", "--point", "2", "--format", "json"});
    ASSERT_EQ(r.status, 0) << r.err;
    const json::Value j = r.json();
    EXPECT_EQ(j["samples"].number, 300.0);
    EXPECT_EQ(j["point"].number, 2.0);
    EXPECT_TRUE(j["uniform"].boolean);
    // Without the restart's replaced rows, every value lies in 2 +- 1
    EXPECT_NEAR(j["statistics"]["mean"].number, 2.0, 0.5);
    const Result window = signal_cmd({file, "--column", "P", "--point", "2", "--times", "5:20", "--format", "json"});
    ASSERT_EQ(window.status, 0) << window.err;
    EXPECT_EQ(window.json()["samples"].number, 151.0);
    EXPECT_EQ(signal_cmd({file, "--column", "P", "--point", "3"}).status, 1);
}

TEST(PostSignalTest, BadColumnsAndFilesAreExplained) {
    if (!comm::is_root()) GTEST_SKIP() << "runs on one rank";
    std::vector<double> x(200, 1.0);
    const std::string file = series_file("bad", x, 0.1);
    const Result column = signal_cmd({file, "--column", "U_X"});
    EXPECT_EQ(column.status, 1);
    EXPECT_NE(column.err.find("E-SIGNAL"), std::string::npos) << column.err;
    EXPECT_NE(column.err.find("--column: \"U_X\" is not a column of " + file + " (did you mean \"U_Y\"?)"),
              std::string::npos)
        << column.err;
    EXPECT_NE(column.err.find("Accepted values: U_Y"), std::string::npos) << column.err;
    EXPECT_NE(column.err.find("docs/guide/output.md#"), std::string::npos) << column.err;

    const Result missing = signal_cmd({(scratch_dir() / "signal_no_such.csv").string(), "--column", "U_Y"});
    EXPECT_EQ(missing.status, 1);
    EXPECT_NE(missing.err.find("signal_no_such.csv"), std::string::npos) << missing.err;

    // Not equally spaced: statistics, but no spectrum, and the spectrum's options are an error
    std::vector<std::vector<double>> rows;
    for (int i = 0; i < 200; i++) rows.push_back({static_cast<double>(i), 0.1 * i + 0.01 * (i % 2), std::cos(0.2 * i)});
    const std::string uneven = write_table("uneven", {"step", "t", "Fx_pressure"}, {"1", "s", "kg s-2"}, rows);
    const Result stats = signal_cmd({uneven, "--column", "Fx_pressure", "--format", "json"});
    ASSERT_EQ(stats.status, 0) << stats.err;
    EXPECT_TRUE(stats.json()["spectrum"].is_null());
    const Result spectrum = signal_cmd({uneven, "--column", "Fx_pressure", "--segment", "5"});
    EXPECT_EQ(spectrum.status, 1);
    EXPECT_NE(spectrum.err.find("not equally spaced in time"), std::string::npos) << spectrum.err;

    EXPECT_EQ(signal_cmd({file, "--column", "U_Y", "--times", "50:60"}).status, 1);
    EXPECT_EQ(signal_cmd({file, "--column", "U_Y", "--segment", "1000"}).status, 1);
    EXPECT_EQ(signal_cmd({file, "--column", "U_Y", "--window", "hamming"}).status, 2);
    EXPECT_EQ(signal_cmd({file}).status, 2);
}

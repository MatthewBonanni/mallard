/**
 * @file convergence_test.cpp
 * @author Matthew Bonanni (mbonanni001@gmail.com)
 * @brief [statistics.convergence]: Student's t quantiles against tables, the
 *        integrated autocorrelation time, coverage of the batch-means
 *        intervals and MSER-5 truncation on AR(1) signals with known answers,
 *        and the state across restarts.
 * @version 0.1
 * @date 2026-11-10
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 *
 */

#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <limits>
#include <numbers>
#include <random>
#include <vector>

#include "statistical_convergence.h"
#include "scratch.h"
#include "test_fixtures.h"

namespace {

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

} // namespace

TEST(StatisticalConvergenceTest, StudentTQuantilesMatchTables) {
    struct Case {
        double confidence;
        uint32_t nu;
        double t;
    };
    // Two-sided quantiles, e.g. NIST/SEMATECH e-Handbook table 1.3.6.7.2
    for (const Case c : {Case{0.95, 1, 12.706204736}, Case{0.95, 2, 4.302652730}, Case{0.95, 9, 2.262157163},
                         Case{0.95, 19, 2.093024054}, Case{0.99, 19, 2.860934606}, Case{0.90, 30, 1.697260887},
                         Case{0.95, 1000, 1.962339081}}) {
        EXPECT_NEAR(time_series::student_t(c.confidence, c.nu), c.t, 1e-8 * c.t) << c.confidence << " " << c.nu;
    }
}

TEST(StatisticalConvergenceTest, AutocorrelationTimeOfAR1IsOnePlusPhiOverOneMinusPhi) {
    // tau_int = (1 + phi) / (1 - phi); its statistical error here is about 3%
    for (const double phi : {0.0, 0.5, 0.8}) {
        const std::vector<double> x = ar1(phi, 200000, 7);
        const double tau = (1.0 + phi) / (1.0 - phi);
        EXPECT_NEAR(time_series::autocorrelation_time(x.data(), x.size()), tau, 0.08 * tau) << phi;
    }
    const std::vector<double> constant(100, 3.0);
    EXPECT_EQ(time_series::autocorrelation_time(constant.data(), constant.size()), 1.0);
}

TEST(StatisticalConvergenceTest, BatchMeansIntervalsCoverTheMeanAndVarianceAtTheirConfidence) {
    // 1000 AR(1) series (tau_int = 9): the share of 95% intervals holding the exact mean 0 and
    // variance 1 / (1 - phi^2) is 0.95 within 3 binomial standard deviations (0.021)
    const double phi = 0.8, variance = 1.0 / (1.0 - phi * phi);
    const uint64_t n = 8000, replicates = 1000;
    uint64_t covers_mean = 0, covers_variance = 0;
    double width = 0.0;
    for (uint64_t r = 0; r < replicates; r++) {
        const std::vector<double> x = ar1(phi, n, 1000 + r);
        const time_series::Interval m = time_series::batch_means(x.data(), n, 20, 0.95);
        ASSERT_EQ(m.batch_size, n / 20);
        if (std::fabs(m.mean) <= m.half_width) covers_mean++;
        width += m.half_width;
        double mean = 0.0;
        for (const double v : x) mean += v;
        mean /= static_cast<double>(n);
        std::vector<double> y(n);
        for (uint64_t i = 0; i < n; i++) y[i] = (x[i] - mean) * (x[i] - mean);
        const time_series::Interval v = time_series::batch_means(y.data(), n, 20, 0.95);
        if (std::fabs(v.mean - variance) <= v.half_width) covers_variance++;
    }
    const double share_mean = static_cast<double>(covers_mean) / replicates;
    const double share_variance = static_cast<double>(covers_variance) / replicates;
    EXPECT_NEAR(share_mean, 0.95, 0.021);
    EXPECT_NEAR(share_variance, 0.95, 0.021);
    // The width that the autocorrelation sets: 1.96 sqrt(tau sigma^2 / n), not that of independent samples
    EXPECT_NEAR(width / replicates, 1.96 * std::sqrt(9.0 * variance / n), 0.06 * 1.96 * std::sqrt(9.0 * variance / n));
}

TEST(StatisticalConvergenceTest, MserFindsTheEndOfAKnownTransient) {
    // An offset of 5 standard deviations for the first 1000 samples
    std::vector<double> x = ar1(0.5, 6000, 11);
    for (size_t i = 0; i < 1000; i++) x[i] += 5.0 * std::sqrt(1.0 / 0.75);
    const time_series::Truncation cut = time_series::mser(x.data(), x.size());
    EXPECT_TRUE(cut.accepted);
    EXPECT_GE(cut.samples, 1000u);
    EXPECT_LE(cut.samples, 1050u);

    // Stationary from the start: little or nothing deleted
    const std::vector<double> stationary = ar1(0.5, 6000, 12);
    const time_series::Truncation none = time_series::mser(stationary.data(), stationary.size());
    EXPECT_TRUE(none.accepted);
    EXPECT_LT(none.samples, 600u);

    // A drift that has not ended is never accepted, nor are fewer batches than the minimum
    std::vector<double> drift = ar1(0.5, 4000, 13);
    for (size_t i = 0; i < drift.size(); i++) drift[i] += 0.01 * static_cast<double>(i);
    EXPECT_FALSE(time_series::mser(drift.data(), drift.size()).accepted);
    EXPECT_FALSE(time_series::mser(stationary.data(), 99, 5, 20).accepted);
}

namespace {

const double NO_WINDOW = -std::numeric_limits<double>::infinity();

Input convergence_input(const std::string & extra = "") {
    const std::string file = (scratch_dir() / "convergence_test.csv").string();
    return parse_toml("[statistics.convergence]\nsignals = [\"probe_u\"]\nfile = \"" + file + "\"\n" + extra);
}

void feed(Convergence & c, const std::vector<double> & x, uint64_t begin, uint64_t end, double window_start) {
    for (uint64_t i = begin; i < end; i++) c.add(i, 0.01 * static_cast<double>(i), {{0, x[i]}}, window_start);
}

} // namespace

TEST(StatisticalConvergenceTest, StationaryAfterTheTransientThenConvergedAtTheTolerance) {
    // A transient decaying over 400 samples from 9 standard deviations: the mean of the whole
    // series would be off by 6000 / n, 0.6 at the 10^4 samples the tolerance needs
    std::vector<double> x = ar1(0.8, 40000, 21);
    for (size_t i = 0; i < x.size(); i++) x[i] += 10.0 + 15.0 * std::exp(-static_cast<double>(i) / 400.0);
    Convergence c;
    c.init(convergence_input("tolerance = 0.01\n"), {{"probe_u", dim::velocity}}, false);
    auto value = [&](const std::string & name) {
        for (const auto & [n, v] : c.last().values) {
            if (n == name) return v;
        }
        ADD_FAILURE() << name;
        return 0.0;
    };
    uint64_t stationary_at = 0, converged_at = 0;
    for (uint64_t i = 0; i < x.size() && converged_at == 0; i++) {
        feed(c, x, i, i + 1, NO_WINDOW);
        if (stationary_at == 0 && value("stationary") == 1.0) stationary_at = i;
        if (value("converged") == 1.0) converged_at = i;
    }
    ASSERT_GT(stationary_at, 0u);
    // By then the start has moved past the decay, whose rest biases the mean by 15 * 400 e^-2.5 / n = 0.05
    EXPECT_GE(value("t_stationary"), 10.0);
    EXPECT_LE(value("t_stationary"), 40.0);
    // Relative half width t sqrt(tau var / n) / 10 = 0.01 at n = t^2 tau var / 0.1^2, about 11000 (t = 2.093
    // for 19 degrees of freedom); 20 batches estimate the variance of the mean to about 30%
    ASSERT_GT(converged_at, 0u);
    const double n = value("probe_u_samples"), expected = 2.093 * 2.093 * 9.0 / 0.36 / 1e-2;
    EXPECT_GT(n, 0.5 * expected);
    EXPECT_LT(n, 2.0 * expected);
    EXPECT_NEAR(value("probe_u_mean"), 10.0, 0.15);
    EXPECT_NEAR(value("probe_u_tau"), 9.0, 1.5);
}

TEST(StatisticalConvergenceTest, IntervalsCoverOnlyTheAveragingWindowOfTheStatistics) {
    const std::vector<double> x = ar1(0.5, 3000, 31);
    Convergence c;
    c.init(convergence_input(), {{"probe_u", dim::velocity}}, false);
    // The averages started at t = 20 (sample 2000): samples after it
    feed(c, x, 0, 3000, 20.0);
    double samples = 0.0;
    for (const auto & [n, v] : c.last().values) {
        if (n == "probe_u_samples") samples = v;
    }
    EXPECT_EQ(samples, 999.0);
}

TEST(StatisticalConvergenceTest, RestartedHistoryContinuesBitwise) {
    std::vector<double> x = ar1(0.8, 6000, 41);
    for (size_t i = 0; i < 800; i++) x[i] += 8.0;
    Convergence whole;
    whole.init(convergence_input("tolerance = 0.05\n"), {{"probe_u", dim::velocity}}, false);
    feed(whole, x, 0, 6000, NO_WINDOW);
    for (const uint64_t cut : {uint64_t{300}, uint64_t{3000}}) {
        Convergence a, b;
        a.init(convergence_input("tolerance = 0.05\n"), {{"probe_u", dim::velocity}}, false);
        b.init(convergence_input("tolerance = 0.05\n"), {{"probe_u", dim::velocity}}, false);
        feed(a, x, 0, cut, NO_WINDOW);
        RestartData restart;
        restart.attributes = a.attributes();
        restart.arrays = a.arrays();
        b.restore(restart, "restart", NO_WINDOW);
        ASSERT_EQ(b.last().values.size(), a.last().values.size());
        for (size_t k = 0; k < a.last().values.size(); k++) {
            const double va = a.last().values[k].second, vb = b.last().values[k].second;
            EXPECT_TRUE(va == vb || (std::isnan(va) && std::isnan(vb))) << a.last().values[k].first;
        }
        feed(b, x, cut, 6000, NO_WINDOW);
        for (size_t k = 0; k < whole.last().values.size(); k++) {
            EXPECT_EQ(b.last().values[k].second, whole.last().values[k].second) << whole.last().values[k].first;
        }
    }
}

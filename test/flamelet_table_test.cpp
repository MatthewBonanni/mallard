/**
 * @file flamelet_table_test.cpp
 * @author Matthew Bonanni (mbonanni001@gmail.com)
 * @brief Tabulated chemistry: the table format, the device lookup and the FPV
 *        map, against exact multilinear functions and a Cantera counterflow
 *        flamelet that is not one of the table's.
 * @version 0.1
 * @date 2026-11-15
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 *
 */

#include <gtest/gtest.h>
#include <Kokkos_Core.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iterator>
#include <random>
#include <string>
#include <vector>

#include "data_paths.h"
#include "flamelet_table.h"
#include "scratch.h"
#include "tabulated.h"

using flamelet::Table;

namespace {

const std::string DATA_DIR = source_dir() + "/test/data/flamelet/";

/** @brief Fields fields at each query point q[i] (n_axes coordinates each), looked up in a device kernel. */
std::vector<double> lookup_device(const flamelet::TableView<> & t, const std::vector<std::vector<double>> & q,
                                  const std::vector<uint32_t> & fields) {
    const uint32_t n = static_cast<uint32_t>(q.size()), nf = static_cast<uint32_t>(fields.size()), na = t.n_axes;
    Kokkos::View<double **, Kokkos::LayoutRight> d_q("q", n, na), d_out("out", n, nf);
    Kokkos::View<uint32_t *> d_f("fields", nf);
    auto h_q = Kokkos::create_mirror_view(d_q);
    auto h_f = Kokkos::create_mirror_view(d_f);
    for (uint32_t i = 0; i < n; i++) {
        for (uint32_t a = 0; a < na; a++) h_q(i, a) = q[i][a];
    }
    for (uint32_t k = 0; k < nf; k++) h_f(k) = fields[k];
    Kokkos::deep_copy(d_q, h_q);
    Kokkos::deep_copy(d_f, h_f);
    Kokkos::parallel_for("lookup", n, KOKKOS_LAMBDA(const uint32_t i) {
        const flamelet::Point p = t.locate(&d_q(i, 0));
        t.evaluate(p, d_f.data(), nf, &d_out(i, 0));
    });
    auto h_out = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), d_out);
    std::vector<double> out(static_cast<size_t>(n) * nf);
    for (uint32_t i = 0; i < n; i++) {
        for (uint32_t k = 0; k < nf; k++) out[i * nf + k] = h_out(i, k);
    }
    return out;
}

/** @brief Fields at the FPV points (Z, Zvar, C), looked up in a device kernel. */
std::vector<double> fpv_device(const flamelet::FPV<> & fpv, const std::vector<std::array<double, 3>> & q,
                               const std::vector<uint32_t> & fields) {
    const uint32_t n = static_cast<uint32_t>(q.size()), nf = static_cast<uint32_t>(fields.size());
    Kokkos::View<double *[3]> d_q("q", n);
    Kokkos::View<double **, Kokkos::LayoutRight> d_out("out", n, nf);
    Kokkos::View<uint32_t *> d_f("fields", nf);
    auto h_q = Kokkos::create_mirror_view(d_q);
    auto h_f = Kokkos::create_mirror_view(d_f);
    for (uint32_t i = 0; i < n; i++) {
        for (uint32_t a = 0; a < 3; a++) h_q(i, a) = q[i][a];
    }
    for (uint32_t k = 0; k < nf; k++) h_f(k) = fields[k];
    Kokkos::deep_copy(d_q, h_q);
    Kokkos::deep_copy(d_f, h_f);
    Kokkos::parallel_for("fpv", n, KOKKOS_LAMBDA(const uint32_t i) {
        const flamelet::Point p = fpv.locate(d_q(i, 0), d_q(i, 1), d_q(i, 2));
        fpv.table.evaluate(p, d_f.data(), nf, &d_out(i, 0));
    });
    auto h_out = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), d_out);
    std::vector<double> out(static_cast<size_t>(n) * nf);
    for (uint32_t i = 0; i < n; i++) {
        for (uint32_t k = 0; k < nf; k++) out[i * nf + k] = h_out(i, k);
    }
    return out;
}

uint32_t field(const Table & t, const std::string & name) {
    const int32_t f = t.field(name);
    EXPECT_GE(f, 0) << name;
    return static_cast<uint32_t>(std::max(f, 0));
}

} // namespace

// Multilinear interpolation reproduces every multilinear function exactly, on uneven axes, and
// clamps to the ends outside them: catches wrong strides, corner order or weights.
TEST(FlameletTableTest, LookupIsExactForMultilinearFunctions) {
    const std::vector<Table::Axis> axes = {{"x", {0.0, 0.1, 0.35, 1.0}},
                                           {"y", {-1.0, 2.0, 2.5}},
                                           {"z", {0.0, 0.2, 0.3, 0.7, 1.0}},
                                           {"w", {1.0, 4.0}}};
    auto f0 = [](double x, double y, double z, double w) { return 1.0 + 2.0 * x - 3.0 * y + 0.5 * z + 7.0 * x * y * z * w; };
    auto f1 = [](double x, double y, double z, double w) { return x * y - z * w + 4.0 * x * z * w; };
    std::vector<double> values;
    for (double x : axes[0].nodes) {
        for (double y : axes[1].nodes) {
            for (double z : axes[2].nodes) {
                for (double w : axes[3].nodes) {
                    values.push_back(f0(x, y, z, w));
                    values.push_back(f1(x, y, z, w));
                }
            }
        }
    }
    const Table t("test", axes, {{"f0", "1"}, {"f1", "1"}}, values);
    std::mt19937 rng(631);
    std::uniform_real_distribution<double> u(0.0, 1.0);
    std::vector<std::vector<double>> q;
    for (int i = 0; i < 200; i++) q.push_back({u(rng), -1.0 + 3.5 * u(rng), u(rng), 1.0 + 3.0 * u(rng)});
    q.push_back({-0.5, 3.0, 2.0, 0.0});  // outside every axis: the corner node
    const auto out = lookup_device(t.device(), q, {1, 0});
    for (size_t i = 0; i < q.size(); i++) {
        auto c = [&](size_t a) { return std::clamp(q[i][a], axes[a].nodes.front(), axes[a].nodes.back()); };
        EXPECT_NEAR(out[2 * i], f1(c(0), c(1), c(2), c(3)), 1e-12) << i;
        EXPECT_NEAR(out[2 * i + 1], f0(c(0), c(1), c(2), c(3)), 1e-12) << i;
    }
}

// The FPV map returns the transported C itself wherever C lies between the table's bounds at
// (Z, S), and the bounds outside: catches a normalization of c against the wrong bounds.
TEST(FlameletTableTest, FpvReproducesTheProgressVariable) {
    const Table t = Table::read(DATA_DIR + "h2_fpv.mtab");
    const flamelet::FPV<Kokkos::HostSpace> host = t.fpv_host();
    const uint32_t C = field(t, "C");
    std::mt19937 rng(287);
    std::uniform_real_distribution<double> u(0.0, 1.0);
    std::vector<std::array<double, 3>> q;
    std::vector<double> expected;
    for (int i = 0; i < 300; i++) {
        const double Z = u(rng), S = u(rng) * u(rng);
        flamelet::Point p;
        host.table.locate_axis(0, Z, p.lo[0], p.t[0]);
        host.table.locate_axis(1, S, p.lo[1], p.t[1]);
        p.t[2] = 0.0;
        const double lo = host.table(p, C);
        p.lo[2] = host.table.n_nodes[2] - 2;
        p.t[2] = 1.0;
        const double hi = host.table(p, C);
        const double target = lo + (-0.2 + 1.4 * u(rng)) * (hi - lo);
        q.push_back({Z, S * Z * (1.0 - Z), target});
        expected.push_back(std::clamp(target, lo, hi));
    }
    const auto out = fpv_device(t.fpv(), q, {C});
    for (size_t i = 0; i < q.size(); i++) EXPECT_NEAR(out[i], expected[i], 1e-12 + 1e-10 * expected[i]) << i;
}

// The verification: a Cantera counterflow flamelet at an oxidizer velocity (1.5 m/s) that is not
// one of the table's, looked up from its own Z and C through the device FPV map, against Cantera's
// T, Y_OH and progress-variable source. Catches a wrong mixture fraction or progress variable in
// the generator, a table on the wrong branch of the S-curve, and lookup errors.
TEST(FlameletTableTest, MatchesACanteraCounterflowFlamelet) {
    const Table t = Table::read(DATA_DIR + "h2_fpv.mtab");
    const tabulated::Csv ref = tabulated::read_csv(DATA_DIR + "h2_counterflow_U1.5.csv");
    auto column = [&](const std::string & name) {
        const int32_t j = ref.column(name);
        EXPECT_GE(j, 0) << name;
        return ref.columns[static_cast<size_t>(std::max(j, 0))];
    };
    const auto Z = column("Z"), C = column("C");
    std::vector<std::array<double, 3>> q;
    for (size_t i = 0; i < Z.size(); i++) q.push_back({Z[i], 0.0, C[i]});
    const std::vector<std::string> names = {"T", "Y_OH", "omega_C"};
    std::vector<uint32_t> fields;
    for (const auto & n : names) fields.push_back(field(t, n));
    const auto out = fpv_device(t.fpv(), q, fields);
    // Tolerances: about twice the interpolation error of this coarse table (41 x 21 nodes in Z, c)
    const double tolerance[] = {10.0, 0.15, 0.06};  // K; fractions of the peak
    for (size_t k = 0; k < names.size(); k++) {
        const auto r = column(names[k]);
        double peak = 0.0, peak_table = 0.0, worst = 0.0;
        for (size_t i = 0; i < r.size(); i++) {
            const double v = out[i * names.size() + k];
            peak = std::max(peak, std::abs(r[i]));
            peak_table = std::max(peak_table, std::abs(v));
            worst = std::max(worst, std::abs(v - r[i]));
        }
        const double allowed = k == 0 ? tolerance[0] : tolerance[k] * peak;
        EXPECT_LT(worst, allowed) << names[k] << ": the table differs from Cantera's flamelet by up to " << worst;
        EXPECT_NEAR(peak_table, peak, allowed) << names[k];
    }
}

// Malformed files are refused with the file and the reason, never read as garbage.
TEST(FlameletTableTest, RefusesMalformedTables) {
    const std::string good = DATA_DIR + "h2_fpv.mtab";
    std::ifstream in(good, std::ios::binary);
    const std::string bytes((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    auto expect_refused = [&](const std::string & name, const std::string & content, const std::string & reason) {
        const std::string file = (rank_scratch_dir() / name).string();
        std::ofstream(file, std::ios::binary) << content;
        try {
            Table::read(file);
            ADD_FAILURE() << name << " was read";
        } catch (const std::runtime_error & e) {
            const std::string what = e.what();
            EXPECT_NE(what.find(file), std::string::npos) << what;
            EXPECT_NE(what.find(reason), std::string::npos) << what;
        }
    };
    expect_refused("truncated.mtab", bytes.substr(0, bytes.size() - 8), "values, where its axes and fields need");
    expect_refused("magic.mtab", "mallard-table 2\n" + bytes.substr(bytes.find('\n') + 1), "not a Mallard table");
    std::string nan = bytes;
    const double bad = std::nan("");
    std::memcpy(nan.data() + nan.size() - sizeof bad, &bad, sizeof bad);
    expect_refused("nan.mtab", nan, "is not finite");
    const Table t = Table::read(good);
    const std::vector<double> zeros(t.axes()[0].nodes.size() * t.axes()[2].nodes.size() * t.fields().size(), 0.0);
    const Table two_axes("fpv", {t.axes()[0], t.axes()[2]}, t.fields(), zeros, "two axes");
    EXPECT_THROW(two_axes.fpv(), std::runtime_error);
}

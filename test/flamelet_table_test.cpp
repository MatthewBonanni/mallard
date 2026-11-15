/**
 * @file flamelet_table_test.cpp
 * @author Matthew Bonanni (mbonanni001@gmail.com)
 * @brief Tabulated chemistry: the table format and its reader contract
 *        (docs/design/table_format.md), the device lookup and the FPV map,
 *        against exact multilinear functions and a synthetic fixture table.
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

using flamelet::Table;

namespace {

const std::string FIXTURE = source_dir() + "/test/data/flamelet/fixture_fpv.mtab";

// The fixture's fields, as test/data/flamelet/fixture_fpv.mtab's header (provenance.input) writes them:
// multilinear in (Z, S, c), so multilinear lookups reproduce them anywhere in the table.
const std::vector<std::string> FIXTURE_FIELDS = {"C",     "omega_C", "rho0", "T0",   "e0",   "R",     "gamma0", "a_gamma",
                                                 "mu",    "kappa",   "cp",   "Y_H2", "Y_O2", "Y_H2O", "Y_N2"};
double fixture(const size_t f, const double Z, const double S, const double c) {
    switch (f) {
        case 0: return c * (0.1 + 0.2 * Z) * (1.0 - 0.5 * S);
        case 1: return 40.0 * c * (1.0 + Z) * (1.0 - S);
        case 2: return 1.2 - 0.8 * c + 0.1 * Z;
        case 3: return 300.0 + 1500.0 * c * (1.0 - 0.5 * S) + 20.0 * Z;
        case 4: return -2.0e5 + 1.2e6 * c - 1.0e5 * Z;
        case 5: return 287.0 * (1.0 + Z);
        case 6: return 1.4 - 0.1 * c - 0.05 * Z;
        case 7: return -1.0e-4 * (1.0 + c);
        case 8: return 1.8e-5 * (1.0 + c) * (1.0 + 0.1 * Z);
        case 9: return 0.026 * (1.0 + 2.0 * c) * (1.0 + 0.1 * S);
        case 10: return 1000.0 * (1.0 + 0.4 * Z) * (1.0 + 0.2 * c);
        case 11: return 0.1 * Z * (1.0 - c);
        case 12: return 0.23 * (1.0 - Z) * (1.0 - c);
        case 13: return 0.2 * c * (1.0 - 0.5 * S);
        default: return 0.7 - 0.1 * Z;
    }
}

std::string read_bytes(const std::string & file) {
    std::ifstream in(file, std::ios::binary);
    return std::string((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
}

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
    const std::vector<Table::Axis> axes = {{"x", {0.0, 0.1, 0.35, 1.0}, "1"},
                                           {"y", {-1.0, 2.0, 2.5}, "1"},
                                           {"z", {0.0, 0.2, 0.3, 0.7, 1.0}, "1"},
                                           {"w", {1.0, 4.0}, "1"}};
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
    const Table t = Table::read(FIXTURE);
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

// The fixture, written by an independent writer (Python, its header's provenance.input), reads with
// its tags, species, pressure and provenance, and every field looks up as the writer computed it:
// catches a wrong node order, field order, stride or byte order in the reader.
TEST(FlameletTableTest, ReadsTheFixture) {
    const Table t = Table::read(FIXTURE);
    EXPECT_EQ(t.model(), "fpv");
    EXPECT_EQ(t.thermodynamics(), "saghafian2015");
    EXPECT_EQ(t.species(), (std::vector<std::string>{"H2", "O2", "H2O", "N2"}));
    EXPECT_EQ(t.pressure(), 101325.0);
    EXPECT_EQ(t.generator(), "mallard-test-fixture 1");
    ASSERT_EQ(t.fields().size(), FIXTURE_FIELDS.size());
    std::vector<uint32_t> fields;
    for (const auto & name : FIXTURE_FIELDS) fields.push_back(field(t, name));
    std::mt19937 rng(631);
    std::uniform_real_distribution<double> u(0.0, 1.0);
    std::vector<std::vector<double>> q;
    for (int i = 0; i < 200; i++) q.push_back({u(rng), u(rng), u(rng)});
    for (double Z : t.axes()[0].nodes) q.push_back({Z, 1.0, 0.25});
    const auto out = lookup_device(t.device(), q, fields);
    for (size_t i = 0; i < q.size(); i++) {
        for (size_t k = 0; k < fields.size(); k++) {
            const double expected = fixture(k, q[i][0], q[i][1], q[i][2]);
            EXPECT_NEAR(out[i * fields.size() + k], expected, 1e-12 * (1.0 + std::abs(expected)))
                << FIXTURE_FIELDS[k] << " at query " << i;
        }
    }
}

// Files that break the format or the contract are refused with the file and the reason, never
// read as garbage: catches a check that is dropped or that tests the wrong thing.
TEST(FlameletTableTest, RefusesMalformedTables) {
    const std::string bytes = read_bytes(FIXTURE);
    const size_t data = bytes.find('\n', bytes.find('\n') + 1) + 1;
    const size_t n_fields = FIXTURE_FIELDS.size();
    auto expect_refused = [&](const std::string & name, const std::string & content, const std::string & reason) {
        const std::string file = (rank_scratch_dir() / name).string();
        std::ofstream(file, std::ios::binary) << content;
        try {
            Table::read(file);
            ADD_FAILURE() << name << " was read";
        } catch (const std::runtime_error & e) {
            const std::string what = e.what();
            EXPECT_NE(what.find(file), std::string::npos) << what;
            EXPECT_NE(what.find(reason), std::string::npos) << name << ": " << what;
        }
    };
    auto edit_header = [&](const std::string & from, const std::string & to) {
        std::string b = bytes;
        const size_t at = b.find(from);
        EXPECT_LT(at, data) << from;
        return at < data ? b.replace(at, from.size(), to) : b;
    };
    auto set_value = [&](const size_t node, const std::string & name, const double v) {
        std::string b = bytes;
        const size_t f = static_cast<size_t>(std::find(FIXTURE_FIELDS.begin(), FIXTURE_FIELDS.end(), name) - FIXTURE_FIELDS.begin());
        std::memcpy(b.data() + data + (node * n_fields + f) * sizeof v, &v, sizeof v);
        return b;
    };
    expect_refused("truncated.mtab", bytes.substr(0, bytes.size() - 8), "values, where its axes and fields need");
    expect_refused("magic.mtab", "mallard-table 2\n" + bytes.substr(bytes.find('\n') + 1), "not a Mallard table");
    expect_refused("nan.mtab", set_value(7, "T0", std::nan("")), "is not finite");
    expect_refused("model.mtab", edit_header("\"model\":\"fpv\"", "\"model\":\"fgm\""),
                   "model \"fgm\" is not supported by this version of Mallard (supported: fpv)");
    expect_refused("thermo.mtab", edit_header("\"saghafian2015\"", "\"low-mach\""),
                   "thermodynamics \"low-mach\" is not supported");
    expect_refused("units.mtab", edit_header("\"name\":\"mu\",\"units\":\"Pa s\"", "\"name\":\"mu\",\"units\":\"P\""),
                   "the field \"mu\" is in \"P\", where Mallard reads \"Pa s\"");
    expect_refused("missing.mtab", edit_header("\"name\":\"kappa\"", "\"name\":\"lambda\""), "no field \"kappa\"");
    expect_refused("species.mtab", edit_header("\"H2O\",\"N2\"]", "\"H2O\"]"), "\"Y_N2\" is not of a species");
    expect_refused("axes.mtab", edit_header("{\"name\":\"S\"", "{\"name\":\"chi\""), "axis 1 of a table of model \"fpv\" is \"S\"");
    expect_refused("sha.mtab", edit_header("\"sha256\":\"", "\"sha256\":\"x"), "provenance.mechanism.sha256");
    expect_refused("generator.mtab", edit_header("\"generator\":", "\"writer\":"), "provenance.generator");
    expect_refused("Zst.mtab", edit_header("\"Z_st\":0.3", "\"Z_st\":1.3"), "model_data.Z_st");
    expect_refused("gamma.mtab", set_value(5, "gamma0", 0.9), "the field \"gamma0\" is not above 1 at node 5");
    expect_refused("fraction.mtab", set_value(3, "Y_O2", 1.5), "the field \"Y_O2\" is outside [0, 1] at node 3");
    expect_refused("linear.mtab", set_value(1, "C", 0.5), "C is not linear in c at (Z, S) node 0");
    const Table t = Table::read(FIXTURE);
    const std::vector<double> zeros(t.axes()[0].nodes.size() * t.axes()[2].nodes.size() * t.fields().size(), 0.0);
    const Table two_axes("fpv", {t.axes()[0], t.axes()[2]}, t.fields(), zeros, "two axes");
    EXPECT_THROW(two_axes.fpv(), std::runtime_error);
}

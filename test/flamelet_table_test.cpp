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
#include <filesystem>
#include <fstream>
#include <functional>
#include <random>
#include <string>
#include <vector>

#include "data_paths.h"
#include "flamelet_table.h"
#include "input.h"
#include "scratch.h"

#ifdef Mallard_HAS_HDF5
#include "hdf5_util.h"
#endif

using flamelet::Table;

namespace {

const std::string FIXTURE = source_dir() + "/test/data/flamelet/fixture_fpv.h5";

#ifdef Mallard_HAS_HDF5
const std::string FIXTURE_F32 = source_dir() + "/test/data/flamelet/fixture_fpv_f32.h5";

// The fixture's fields, as the writer in test/data/flamelet/fixture_fpv.h5 (/provenance/input) computes them:
// multilinear in (Z, Sz, Lambda), so multilinear lookups reproduce them anywhere in the table. fixture_fpv_f32.h5
// holds them rounded to float32.
const std::vector<std::string> FIXTURE_FIELDS = {"C",     "omega_C", "rho0", "T0",   "e0",   "R",     "gamma0", "a_gamma",
                                                 "mu",    "kappa",   "cp",   "Y_H2", "Y_O2", "Y_H2O", "Y_N2"};
double fixture(const size_t f, const double Z, const double Sz, const double Lambda) {
    switch (f) {
        case 0: return Lambda * (0.1 + 0.2 * Z) * (1.0 - 0.5 * Sz);
        case 1: return 40.0 * Lambda * (1.0 + Z) * (1.0 - Sz);
        case 2: return 1.2 - 0.8 * Lambda + 0.1 * Z;
        case 3: return 300.0 + 1500.0 * Lambda * (1.0 - 0.5 * Sz) + 20.0 * Z;
        case 4: return -2.0e5 + 1.2e6 * Lambda - 1.0e5 * Z;
        case 5: return 287.0 * (1.0 + Z);
        case 6: return 1.4 - 0.1 * Lambda - 0.05 * Z;
        case 7: return -1.0e-4 * (1.0 + Lambda);
        case 8: return 1.8e-5 * (1.0 + Lambda) * (1.0 + 0.1 * Z);
        case 9: return 0.026 * (1.0 + 2.0 * Lambda) * (1.0 + 0.1 * Sz);
        case 10: return 1000.0 * (1.0 + 0.4 * Z) * (1.0 + 0.2 * Lambda);
        case 11: return 0.1 * Z * (1.0 - Lambda);
        case 12: return 0.23 * (1.0 - Z) * (1.0 - Lambda);
        case 13: return Lambda * (0.1 + 0.2 * Z) * (1.0 - 0.5 * Sz);
        default: return 0.7 - 0.1 * Z;
    }
}
#endif

/** @brief Fields fields at each query point q[i] (D coordinates each), looked up in a device kernel. */
template <uint32_t D, typename Value>
std::vector<double> lookup_device(const flamelet::TableView<D, Value> & t, const std::vector<std::vector<double>> & q,
                                  const std::vector<uint32_t> & fields) {
    const uint32_t n = static_cast<uint32_t>(q.size()), nf = static_cast<uint32_t>(fields.size()), na = D;
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
        const flamelet::Point<D> p = t.locate(&d_q(i, 0));
        t.evaluate(p, d_f.data(), nf, &d_out(i, 0));
    });
    auto h_out = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), d_out);
    std::vector<double> out(static_cast<size_t>(n) * nf);
    for (uint32_t i = 0; i < n; i++) {
        for (uint32_t k = 0; k < nf; k++) out[i * nf + k] = h_out(i, k);
    }
    return out;
}

#ifdef Mallard_HAS_HDF5
/** @brief Fields at the FPV points (Z, Zvar, C), looked up in a device kernel. */
template <typename Value>
std::vector<double> fpv_device(const flamelet::FPV<Value> & fpv, const std::vector<std::array<double, 3>> & q,
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
        const flamelet::Point<3> p = fpv.locate(d_q(i, 0), d_q(i, 1), d_q(i, 2));
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
#endif

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
    const Table t("test", "test", axes, {{"f0", "1"}, {"f1", "1"}}, values);
    std::mt19937 rng(631);
    std::uniform_real_distribution<double> u(0.0, 1.0);
    std::vector<std::vector<double>> q;
    for (int i = 0; i < 200; i++) q.push_back({u(rng), -1.0 + 3.5 * u(rng), u(rng), 1.0 + 3.0 * u(rng)});
    q.push_back({-0.5, 3.0, 2.0, 0.0});  // outside every axis: the corner node
    const auto out = lookup_device(t.device<4>(), q, {1, 0});
    for (size_t i = 0; i < q.size(); i++) {
        auto c = [&](size_t a) { return std::clamp(q[i][a], axes[a].nodes.front(), axes[a].nodes.back()); };
        EXPECT_NEAR(out[2 * i], f1(c(0), c(1), c(2), c(3)), 1e-12) << i;
        EXPECT_NEAR(out[2 * i + 1], f0(c(0), c(1), c(2), c(3)), 1e-12) << i;
    }
}

// The reader and the lookup take any number of axes: six, past any fixed limit, are exact for a
// multilinear function too. Catches an axis limit or a stride that holds only for few axes.
TEST(FlameletTableTest, LookupTakesAnyNumberOfAxes) {
    constexpr uint32_t D = 6;
    std::vector<Table::Axis> axes;
    for (uint32_t a = 0; a < D; a++) {
        Table::Axis axis{"x" + std::to_string(a), {}, "1"};
        for (uint32_t i = 0; i < 2 + a % 3; i++) axis.nodes.push_back(static_cast<double>(i * i) - static_cast<double>(a));
        axes.push_back(axis);
    }
    auto f = [](const std::vector<double> & x) {
        double v = 1.0, product = 1.0;
        for (size_t a = 0; a < x.size(); a++) {
            v += static_cast<double>(a + 1) * x[a];
            product *= 0.5 + 0.1 * x[a];
        }
        return v + product;
    };
    std::vector<double> values;
    std::vector<size_t> i(D, 0);
    while (i[0] < axes[0].nodes.size()) {
        std::vector<double> x(D);
        for (uint32_t a = 0; a < D; a++) x[a] = axes[a].nodes[i[a]];
        values.push_back(f(x));
        for (uint32_t a = D; a-- > 0;) {
            if (++i[a] < axes[a].nodes.size() || a == 0) break;
            i[a] = 0;
        }
    }
    const Table t("test", "test", axes, {{"f", "1"}}, values);
    std::mt19937 rng(6);
    std::uniform_real_distribution<double> u(-0.1, 1.1);
    std::vector<std::vector<double>> q;
    for (int k = 0; k < 200; k++) {
        std::vector<double> x(D);
        for (uint32_t a = 0; a < D; a++) x[a] = axes[a].nodes.front() + u(rng) * (axes[a].nodes.back() - axes[a].nodes.front());
        q.push_back(x);
    }
    const auto out = lookup_device(t.device<D>(), q, {0});
    for (size_t k = 0; k < q.size(); k++) {
        std::vector<double> c(D);
        for (uint32_t a = 0; a < D; a++) c[a] = std::clamp(q[k][a], axes[a].nodes.front(), axes[a].nodes.back());
        EXPECT_NEAR(out[k], f(c), 1e-12 * (1.0 + std::abs(f(c)))) << k;
    }
}


#ifdef Mallard_HAS_HDF5

namespace {

// GPU builds map the values to stage them for the device; host builds read them into memory
constexpr bool MAPPED = !std::is_same_v<Kokkos::DefaultExecutionSpace::memory_space, Kokkos::HostSpace>;

/** @brief The values of /values exactly as HDF5 reads them, as bytes. */
std::vector<char> h5_values(const std::string & file, const size_t value_size) {
    const std::lock_guard<std::mutex> lock(h5::mutex());
    const h5::Handle h5(H5Fopen(file.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT), H5Fclose, "opening " + file);
    const h5::Handle values(H5Dopen2(h5, "values", H5P_DEFAULT), H5Dclose, "values");
    const h5::Handle space(H5Dget_space(values), H5Sclose, "space");
    std::vector<char> bytes(static_cast<size_t>(H5Sget_simple_extent_npoints(space)) * value_size);
    h5::check(H5Dread(values, value_size == 4 ? H5T_NATIVE_FLOAT : H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT,
                      bytes.data()),
              "reading values");
    return bytes;
}

/** @brief Random points of the fixture's (Z, Sz, Lambda) space, a fifth of them outside it. */
std::vector<std::vector<double>> fixture_points(const uint32_t seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> u(-0.1, 1.1);
    std::vector<std::vector<double>> q;
    for (int i = 0; i < 300; i++) q.push_back({u(rng), u(rng), u(rng)});
    return q;
}

std::vector<uint32_t> all_fields(const Table & t) {
    std::vector<uint32_t> f(t.fields().size());
    for (uint32_t k = 0; k < f.size(); k++) f[k] = k;
    return f;
}

} // namespace

// The FPV map returns the transported C itself wherever C lies between the table's bounds at
// (Z, Sz), and the bounds outside, in float64 tables and to float32 rounding in float32 tables:
// catches a normalization of Lambda against the wrong bounds.
TEST(FlameletTableTest, FpvReproducesTheProgressVariable) {
    auto check = [](const Table & t, auto value, const double tolerance) {
        using Value = decltype(value);
        const flamelet::FPV<Value, Kokkos::HostSpace> host = t.fpv_host<Value>();
        const uint32_t C = field(t, "C");
        std::mt19937 rng(287);
        std::uniform_real_distribution<double> u(0.0, 1.0);
        std::vector<std::array<double, 3>> q;
        std::vector<double> expected;
        for (int i = 0; i < 300; i++) {
            const double Z = u(rng), Sz = u(rng) * u(rng);
            flamelet::Point<3> p;
            host.table.locate_axis(0, Z, p.lo[0], p.t[0]);
            host.table.locate_axis(1, Sz, p.lo[1], p.t[1]);
            p.t[2] = 0.0;
            const double lo = host.table(p, C);
            p.lo[2] = host.table.n_nodes[2] - 2;
            p.t[2] = 1.0;
            const double hi = host.table(p, C);
            const double target = lo + (-0.2 + 1.4 * u(rng)) * (hi - lo);
            q.push_back({Z, Sz * Z * (1.0 - Z), target});
            expected.push_back(std::clamp(target, lo, hi));
        }
        const auto out = fpv_device(t.fpv<Value>(), q, {C});
        for (size_t i = 0; i < q.size(); i++) EXPECT_NEAR(out[i], expected[i], 1e-12 + tolerance * expected[i]) << i;
    };
    check(Table::read(FIXTURE), 0.0, 1e-10);
    check(Table::read(FIXTURE_F32), 0.0f, 1e-6);
}

// The fixture, written by an independent writer (h5py, its /provenance/input), reads with its tags,
// species, pressure and provenance, and every field looks up as the writer computed it: catches a
// wrong node order, field order, stride, or the wrong offset of mapped values.
TEST(FlameletTableTest, ReadsTheFixture) {
    const Table t = Table::read(FIXTURE);
    EXPECT_EQ(t.model(), "fpv");
    EXPECT_EQ(t.thermodynamics(), "saghafian2015");
    EXPECT_EQ(t.species(), (std::vector<std::string>{"H2", "O2", "H2O", "N2"}));
    EXPECT_EQ(t.pressure(), 101325.0);
    EXPECT_EQ(t.generator(), "mallard-test-fixture 2");
    EXPECT_EQ(t.mechanism(), "h2o2.yaml sha256:0efc6c52862741a29e0c29b65d979c7d8cb409db5282bca83b9c5437b3d8c8d4");
    EXPECT_FALSE(t.single());
    EXPECT_EQ(t.mapped(), MAPPED);
    ASSERT_EQ(t.fields().size(), FIXTURE_FIELDS.size());
    std::vector<uint32_t> fields;
    for (const auto & name : FIXTURE_FIELDS) fields.push_back(field(t, name));
    std::mt19937 rng(631);
    std::uniform_real_distribution<double> u(0.0, 1.0);
    std::vector<std::vector<double>> q;
    for (int i = 0; i < 200; i++) q.push_back({u(rng), u(rng), u(rng)});
    for (double Z : t.axes()[0].nodes) q.push_back({Z, 1.0, 0.25});
    const auto out = lookup_device(t.device<3>(), q, fields);
    for (size_t i = 0; i < q.size(); i++) {
        for (size_t k = 0; k < fields.size(); k++) {
            const double expected = fixture(k, q[i][0], q[i][1], q[i][2]);
            EXPECT_NEAR(out[i * fields.size() + k], expected, 1e-12 * (1.0 + std::abs(expected)))
                << FIXTURE_FIELDS[k] << " at query " << i;
        }
    }
}

// The loaded values (mapped in GPU builds, read in host builds) are bitwise those HDF5 reads, in
// float64 and float32 tables, and a float32 table looks up as its float64 source to float32 rounding:
// catches a wrong offset or type of mapped values, and float32 values read as float64.
TEST(FlameletTableTest, LoadedValuesAreThoseHdf5Reads) {
    const auto q = fixture_points(1222);
    auto lookups = [&](const std::string & file, auto value) {
        using Value = decltype(value);
        const Table t = Table::read(file);
        EXPECT_EQ(t.mapped(), MAPPED) << file;
        EXPECT_EQ(t.single(), (std::is_same_v<Value, float>)) << file;
        const flamelet::TableView<3, Value, Kokkos::HostSpace> h = t.host<3, Value>();
        const std::vector<char> expected = h5_values(file, sizeof(Value));
        EXPECT_EQ(h.values.size() * sizeof(Value), expected.size()) << file;
        EXPECT_EQ(std::memcmp(h.values.data(), expected.data(), std::min(expected.size(), h.values.size() * sizeof(Value))), 0)
            << file;
        return lookup_device(t.device<3, Value>(), q, all_fields(t));
    };
    const auto f64 = lookups(FIXTURE, 0.0);
    const auto f32 = lookups(FIXTURE_F32, 0.0f);
    const Table t = Table::read(FIXTURE);
    const size_t nf = t.fields().size();
    const flamelet::TableView<3, double, Kokkos::HostSpace> h = t.host<3>();
    for (size_t k = 0; k < nf; k++) {
        double scale = 0.0;
        for (uint64_t i = k; i < h.values.size(); i += nf) scale = std::max(scale, std::abs(h.values(i)));
        for (size_t i = 0; i < q.size(); i++) {
            EXPECT_NEAR(f32[i * nf + k], f64[i * nf + k], 1.2e-7 * scale) << FIXTURE_FIELDS[k] << " at query " << i;
        }
    }
    try {
        Table::read(FIXTURE_F32).device<3>();
        ADD_FAILURE() << "float32 values were looked up as float64";
    } catch (const std::runtime_error & e) {
        EXPECT_NE(std::string(e.what()).find("its values are float32, where this lookup reads float64"), std::string::npos)
            << e.what();
    }
}

// Files that break the format or the contract are refused with the file and the reason, never
// read as garbage: catches a check that is dropped or that tests the wrong thing.
TEST(FlameletTableTest, RefusesMalformedTables) {
    const Table fixture_table = Table::read(FIXTURE);
    std::vector<std::string> names, units;
    for (const auto & f : fixture_table.fields()) {
        names.push_back(f.name);
        units.push_back(f.units);
    }
    auto expect_refused = [&](const std::string & name, const std::function<void(hid_t)> & edit,
                              const std::string & reason) {
        const std::string file = (rank_scratch_dir() / name).string();
        std::filesystem::copy_file(FIXTURE, file, std::filesystem::copy_options::overwrite_existing);
        {
            const std::lock_guard<std::mutex> lock(h5::mutex());
            const h5::Handle h5(H5Fopen(file.c_str(), H5F_ACC_RDWR, H5P_DEFAULT), H5Fclose, "opening " + file);
            edit(h5);
        }
        try {
            Table::read(file);
            ADD_FAILURE() << name << " was read";
        } catch (const std::runtime_error & e) {
            const std::string what = e.what();
            EXPECT_NE(what.find(file), std::string::npos) << what;
            EXPECT_NE(what.find(reason), std::string::npos) << name << ": " << what;
        }
    };
    auto set_text = [](const hid_t object, const char * attribute, const std::string & value) {
        h5::check(H5Adelete(object, attribute), attribute);
        h5::write_string_scalar_attribute(object, attribute, value);
    };
    auto set_texts = [](const hid_t object, const char * attribute, const std::vector<std::string> & values) {
        h5::check(H5Adelete(object, attribute), attribute);
        h5::write_strings_attribute(object, attribute, values);
    };
    auto set_number = [](const hid_t parent, const char * group, const char * attribute, const double value) {
        const h5::Handle g(H5Oopen(parent, group, H5P_DEFAULT), H5Oclose, group);
        h5::set_attribute(g, attribute, value);
    };
    auto replaced = [](std::vector<std::string> v, const std::string & from, const std::string & to) {
        std::replace(v.begin(), v.end(), from, to);
        return v;
    };
    auto set_value = [&](const size_t node, const std::string & name, const double v) {
        return [&names, node, name, v](const hid_t h5) {
            const h5::Handle values(H5Dopen2(h5, "values", H5P_DEFAULT), H5Dclose, "values");
            const h5::Handle space(H5Dget_space(values), H5Sclose, "space");
            hsize_t dims[4] = {};
            H5Sget_simple_extent_dims(space, dims, nullptr);
            const hsize_t at[4] = {node / (dims[1] * dims[2]), node / dims[2] % dims[1], node % dims[2],
                                   static_cast<hsize_t>(std::find(names.begin(), names.end(), name) - names.begin())};
            h5::check(H5Sselect_elements(space, H5S_SELECT_SET, 1, at), "select");
            const hsize_t one = 1;
            const h5::Handle memory(H5Screate_simple(1, &one, nullptr), H5Sclose, "memory");
            h5::check(H5Dwrite(values, H5T_NATIVE_DOUBLE, memory, space, H5P_DEFAULT, &v), "write");
        };
    };
    auto on_values = [](const std::function<void(hid_t)> & edit) {
        return [edit](const hid_t h5) {
            const h5::Handle values(H5Dopen2(h5, "values", H5P_DEFAULT), H5Dclose, "values");
            edit(values);
        };
    };

    const std::string truncated = (rank_scratch_dir() / "truncated.h5").string();
    std::filesystem::copy_file(FIXTURE, truncated, std::filesystem::copy_options::overwrite_existing);
    std::filesystem::resize_file(truncated, std::filesystem::file_size(truncated) - 8);
    const std::string text = (rank_scratch_dir() / "text.h5").string();
    std::ofstream(text) << "mallard-table 1\n";
    for (const std::string & file : {truncated, text}) {
        try {
            Table::read(file);
            ADD_FAILURE() << file << " was read";
        } catch (const std::runtime_error & e) {
            EXPECT_NE(std::string(e.what()).find(file + ": it is not an HDF5 file, or it is damaged"), std::string::npos)
                << e.what();
        }
    }
    expect_refused("format.h5", [&](hid_t h5) { set_text(h5, "format", "mallard-mesh"); }, "it is not a Mallard table");
    expect_refused("version.h5", [&](hid_t h5) { set_number(h5, ".", "version", 3.0); },
                   "a Mallard table of version 3, which this version of Mallard does not read");
    expect_refused("nan.h5", set_value(7, "T0", std::nan("")), "is not finite");
    expect_refused("model.h5", [&](hid_t h5) { set_text(h5, "model", "fgm"); },
                   "model \"fgm\" is not supported by this version of Mallard (supported: fpv)");
    expect_refused("thermo.h5", [&](hid_t h5) { set_text(h5, "thermodynamics", "low-mach"); },
                   "thermodynamics \"low-mach\" is not supported");
    expect_refused("units.h5", on_values([&](hid_t v) {
        std::vector<std::string> u = units;
        u[static_cast<size_t>(fixture_table.field("mu"))] = "P";
        set_texts(v, "units", u);
    }), "the field \"mu\" is in \"P\", where Mallard reads \"Pa s\"");
    expect_refused("missing.h5", on_values([&](hid_t v) { set_texts(v, "fields", replaced(names, "kappa", "lambda")); }),
                   "no field \"kappa\"");
    expect_refused("species.h5", [&](hid_t h5) { set_texts(h5, "species", {"H2", "O2", "H2O"}); },
                   "\"Y_N2\" is not of a species");
    expect_refused("axes.h5", [&](hid_t h5) {
        h5::check(H5Lmove(h5, "axes/Sz", h5, "axes/chi", H5P_DEFAULT, H5P_DEFAULT), "move");
        on_values([&](hid_t v) { set_texts(v, "axes", {"Z", "chi", "Lambda"}); })(h5);
    }, "axis 1 of a table of model \"fpv\" is \"Sz\"");
    expect_refused("axis.h5", [&](hid_t h5) { h5::check(H5Ldelete(h5, "axes/Lambda", H5P_DEFAULT), "delete"); },
                   "no valid dataset /axes/Lambda");
    expect_refused("shape.h5", on_values([&](hid_t v) {
        std::vector<std::string> n = names, u = units;
        n.push_back("extra");
        u.push_back("1");
        set_texts(v, "fields", n);
        set_texts(v, "units", u);
    }), "its values have shape (5, 3, 4, 15), where its axes and fields need (5, 3, 4, 16)");
    expect_refused("sha.h5", [&](hid_t h5) {
        const h5::Handle g(H5Gopen2(h5, "provenance/mechanism", H5P_DEFAULT), H5Gclose, "mechanism");
        set_text(g, "sha256", "x");
    }, "\"sha256\" on /provenance/mechanism");
    expect_refused("generator.h5", [&](hid_t h5) {
        h5::check(H5Lmove(h5, "provenance/generator", h5, "provenance/writer", H5P_DEFAULT, H5P_DEFAULT), "move");
    }, "no valid group /provenance/generator");
    expect_refused("input.h5", [&](hid_t h5) { h5::check(H5Ldelete(h5, "provenance/input", H5P_DEFAULT), "delete"); },
                   "no valid dataset /provenance/input");
    expect_refused("Zst.h5", [&](hid_t h5) { set_number(h5, "model_data", "Z_st", 1.3); },
                   "attribute \"Z_st\" on /model_data");
    expect_refused("gamma.h5", set_value(5, "gamma0", 0.9), "the field \"gamma0\" is not above 1 at node 5");
    expect_refused("fraction.h5", set_value(3, "Y_O2", 1.5), "the field \"Y_O2\" is outside [0, 1] at node 3");
    expect_refused("linear.h5", set_value(1, "C", 0.5), "C is not linear in Lambda at (Z, Sz) node 0");
    expect_refused("name.h5", on_values([&](hid_t v) { set_texts(v, "fields", replaced(names, "Y_N2", "Y N2")); }),
                   "field name \"Y N2\" is not of the form");
    expect_refused("weights.h5", [&](hid_t h5) {
        const h5::Handle g(H5Gopen2(h5, "model_data/progress_variable", H5P_DEFAULT), H5Gclose, "weights");
        set_text(g, "H2O", "1");
    }, "attribute \"H2O\" on /model_data/progress_variable");
    expect_refused("stream.h5", [&](hid_t h5) { set_number(h5, "model_data/streams/fuel/Y", "H2", 1.1); },
                   "the mass fraction of H2 in /model_data/streams/fuel/Y is not in [0, 1]");
    expect_refused("stream_sum.h5", [&](hid_t h5) { set_number(h5, "model_data/streams/fuel/Y", "H2", 0.2); },
                   "the mass fractions of /model_data/streams/fuel/Y sum to 1.1, not 1");
    expect_refused("definition.h5", [&](hid_t h5) {
        const h5::Handle g(H5Gopen2(h5, "model_data", H5P_DEFAULT), H5Gclose, "model_data");
        set_text(g, "mixture_fraction", "elements");
    }, "its attribute \"mixture_fraction\" on /model_data is \"elements\", where Mallard reads \"bilger\"");
    expect_refused("undefined.h5", [&](hid_t h5) {
        const h5::Handle g(H5Gopen2(h5, "model_data", H5P_DEFAULT), H5Gclose, "model_data");
        h5::check(H5Adelete(g, "mixture_fraction"), "delete");
    }, "no valid attribute \"mixture_fraction\" on /model_data");
    expect_refused("weighs.h5", [&](hid_t h5) { set_number(h5, "model_data/progress_variable", "OH", 1.0); },
                   "its progress variable weighs OH, which is not in its species block");
    expect_refused("weighted.h5", set_value(5, "Y_H2O", 0.9),
                   "its field C (0.02) is not the weighted sum of its mass fractions sum_k w_k Y_k (0.9)");

    // Version 1 tables, without the definitions of version 2, still run: their weights and streams are not checked
    const std::string version_1 = (rank_scratch_dir() / "version_1.h5").string();
    std::filesystem::copy_file(FIXTURE, version_1, std::filesystem::copy_options::overwrite_existing);
    {
        const std::lock_guard<std::mutex> lock(h5::mutex());
        const h5::Handle h5(H5Fopen(version_1.c_str(), H5F_ACC_RDWR, H5P_DEFAULT), H5Fclose, "opening");
        set_number(h5, ".", "version", 1.0);
        set_number(h5, "model_data/progress_variable", "OH", 1.0);
        set_number(h5, "model_data/streams/fuel/Y", "H2", 0.2);
        const h5::Handle g(H5Gopen2(h5, "model_data", H5P_DEFAULT), H5Gclose, "model_data");
        h5::check(H5Adelete(g, "mixture_fraction"), "delete");
    }
    const Table old = Table::read(version_1);
    EXPECT_EQ(old.version(), 1u);
    EXPECT_TRUE(old.definitions().mixture_fraction.empty());
    EXPECT_NO_THROW(old.fpv_host());

    const std::vector<double> zeros(fixture_table.axes()[0].nodes.size() * fixture_table.axes()[2].nodes.size() *
                                        fixture_table.fields().size(),
                                    0.0);
    const Table two_axes("fpv", "saghafian2015", {fixture_table.axes()[0], fixture_table.axes()[2]},
                         fixture_table.fields(), zeros, "two axes");
    try {
        two_axes.fpv();
        ADD_FAILURE() << "a table of two axes has an FPV map";
    } catch (const std::runtime_error & e) {
        EXPECT_NE(std::string(e.what()).find("a table of model \"fpv\" has 3 axes"), std::string::npos) << e.what();
    }
}

#else

// A build without HDF5 refuses a table with an input error that says how to get HDF5, rather than
// failing to parse it: catches the reader compiled out silently.
TEST(FlameletTableTest, ReadingNeedsHdf5) {
    try {
        Table::read(FIXTURE);
        ADD_FAILURE() << "a table was read without HDF5";
    } catch (const InputError & e) {
        const std::string what = e.what();
        EXPECT_NE(what.find(FIXTURE), std::string::npos) << what;
        EXPECT_NE(what.find("this build has no HDF5. Rebuild with -DMallard_ENABLE_HDF5=ON"), std::string::npos) << what;
    }
}

#endif

/**
 * @file flamelet_gas_test.cpp
 * @author Matthew Bonanni (mbonanni001@gmail.com)
 * @brief The flamelet gas: saghafian2015 thermodynamics against a quadrature
 *        of cv(T), device lookups of thermodynamics, transport and the progress
 *        source on a synthetic table with exact answers, and the host states.
 * @version 0.1
 * @date 2026-11-20
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 *
 */

#include <gtest/gtest.h>
#include <Kokkos_Core.hpp>

#include <cmath>
#include <memory>
#include <random>
#include <string>
#include <vector>

#include "common.h"
#include "data_paths.h"
#include "flamelet.h"

using flamelet::Table;

namespace {

/** @brief int_T0^T R / (gamma0 + a (T' - T0) - 1) dT' by composite Simpson. */
double cv_integral(const flamelet::Thermo & th, const double T) {
    const int n = 4000;
    const double h = (T - th.T0) / n;
    double sum = th.cv(th.T0) + th.cv(T);
    for (int i = 1; i < n; i++) sum += (i % 2 ? 4.0 : 2.0) * th.cv(th.T0 + i * h);
    return sum * h / 3.0;
}

// A table of the fpv / saghafian2015 contract whose fields are multilinear in (Z, S, c), so that
// lookups reproduce them exactly anywhere in the table.
const std::vector<std::string> FIELDS = {"C",      "omega_C", "rho0", "T0", "e0",   "R",
                                         "gamma0", "a_gamma", "mu",   "kappa", "cp"};
const std::vector<std::string> UNITS = {"1", "kg m-3 s-1", "kg m-3", "K", "J kg-1", "J kg-1 K-1",
                                        "1", "K-1",        "Pa s",   "W m-1 K-1",   "J kg-1 K-1"};

double exact(const size_t f, const double Z, const double S, const double c) {
    switch (f) {
        case 0: return c * (0.05 + 0.25 * Z) * (1.0 - 0.4 * S);
        case 1: return 30.0 * c * (1.0 + Z) * (1.0 - 0.5 * S) + 5.0 * Z;
        case 2: return 1.1 - 0.7 * c + 0.2 * Z;
        case 3: return 320.0 + 1600.0 * c * (1.0 - 0.3 * S) + 40.0 * Z;
        case 4: return -1.0e5 - 2.0e6 * c * (1.0 - 0.2 * S) + 3.0e5 * Z;
        case 5: return 290.0 + 100.0 * Z - 10.0 * c;
        case 6: return 1.38 - 0.08 * c - 0.03 * Z;
        case 7: return -5.0e-5 * (1.0 + c) + 1.0e-5 * Z * S;
        case 8: return 1.7e-5 * (1.0 + 2.0 * c) * (1.0 + 0.2 * Z);
        case 9: return 0.025 * (1.0 + 3.0 * c) * (1.0 + 0.1 * S);
        default: return 1050.0 * (1.0 + 0.3 * Z) * (1.0 + 0.25 * c);
    }
}

std::shared_ptr<const Table> synthetic_table() {
    const std::vector<Table::Axis> axes = {
        {"Z", {0.0, 0.15, 0.4, 1.0}, "1"}, {"S", {0.0, 0.3, 1.0}, "1"}, {"c", {0.0, 0.25, 0.6, 1.0}, "1"}};
    std::vector<Table::Field> fields;
    for (size_t f = 0; f < FIELDS.size(); f++) fields.push_back({FIELDS[f], UNITS[f]});
    std::vector<double> values;
    for (double Z : axes[0].nodes) {
        for (double S : axes[1].nodes) {
            for (double c : axes[2].nodes) {
                for (size_t f = 0; f < FIELDS.size(); f++) values.push_back(exact(f, Z, S, c));
            }
        }
    }
    return std::make_shared<const Table>("fpv", "saghafian2015", axes, fields, values);
}

/** @brief The table's normalized coordinates (S, c) of a point (Z, Zvar, C) inside its bounds. */
void coordinates(const double Z, const double Zvar, const double C, double & S, double & c) {
    S = Zvar / (Z * (1.0 - Z));
    c = C / exact(0, Z, S, 1.0);
}

} // namespace

// The closed-form T(e) and its inverse e(T) are the integral of cv = R / (gamma(T) - 1) with gamma
// linear in T, on both sides of T0, for a_gamma of either sign, at zero and across the switch to
// the series near zero: catches a wrong exponent or logarithm, a sign, or a series that departs
// from the closed form.
TEST(FlameletGasTest, ThermodynamicsIntegrateTheSpecificHeat) {
    for (const double a : {-6.0e-5, 4.0e-5, 0.0, 2.0e-10, -3.0e-10}) {
        const flamelet::Thermo th{1200.0, -1.5e5, 310.0, 1.31, a};
        for (const double T : {300.0, 900.0, 1200.0, 1700.0, 2600.0}) {
            const double e = th.e0 + cv_integral(th, T);
            EXPECT_NEAR(th.energy(T), e, 1e-9 * std::abs(e) + 1e-6) << a << " " << T;
            EXPECT_NEAR(th.temperature(e), T, 1e-9 * T) << a << " " << T;
        }
    }
}

// Device lookups at transported points (Z, Zvar, C) return each field as the table holds it at
// (Z, S, c), and the source omega_C (rho / rho0)^2: catches swapped field indices, a variance
// that does not reach the map, and a source scaled other than to second order in density.
TEST(FlameletGasTest, DeviceLookupsReturnTheTabulatedFields) {
    const flamelet::FlameletModel model(synthetic_table(), true);
    const flamelet::Gas<> gas = model.device();
    ASSERT_TRUE(gas.active());
    std::mt19937 rng(287);
    std::uniform_real_distribution<double> u(0.0, 1.0);
    const uint32_t n = 100;
    Kokkos::View<double *[4]> q("q", n);
    Kokkos::View<double *[10]> out("out", n);
    auto h_q = Kokkos::create_mirror_view(q);
    for (uint32_t i = 0; i < n; i++) {
        const double Z = 0.02 + 0.96 * u(rng), S = 0.9 * u(rng), c = u(rng);
        h_q(i, 0) = Z;
        h_q(i, 1) = S * Z * (1.0 - Z);
        h_q(i, 2) = c * exact(0, Z, S, 1.0);
        h_q(i, 3) = 0.5 + u(rng);  // rho
    }
    Kokkos::deep_copy(q, h_q);
    Kokkos::parallel_for("flamelet_gas", n, KOKKOS_LAMBDA(const uint32_t i) {
        const flamelet::Point<3> p = gas.locate(q(i, 0), q(i, 1), q(i, 2));
        const flamelet::Thermo th = gas.thermo(p);
        const flamelet::Transport tr = gas.transport(p);
        const double v[10] = {th.T0, th.e0, th.R, th.gamma0, th.a_gamma, tr.mu, tr.kappa, tr.cp,
                              gas.source(p, q(i, 3)), gas.field(p, gas.fpv.field_C)};
        for (int k = 0; k < 10; k++) out(i, k) = v[k];
    });
    auto h_out = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), out);
    for (uint32_t i = 0; i < n; i++) {
        double S, c;
        coordinates(h_q(i, 0), h_q(i, 1), h_q(i, 2), S, c);
        auto f = [&](size_t k) { return exact(k, h_q(i, 0), S, c); };
        const double expected[10] = {f(3), f(4), f(5), f(6), f(7), f(8), f(9), f(10),
                                     f(1) * std::pow(h_q(i, 3) / f(2), 2), h_q(i, 2)};
        for (int k = 0; k < 10; k++) {
            EXPECT_NEAR(h_out(i, k), expected[k], 1e-11 * std::abs(expected[k]) + 1e-15) << i << " " << k;
        }
    }
}

// Host states for initial and boundary conditions: rho = p / (R T) with the table's R at (Z, C),
// the energy whose closed-form temperature is T, the scalars' partial densities, and Riemann
// surrogates with which rho e = p / (gamma - 1) + rho e0 holds: catches a state built from the
// wrong table point or an energy inconsistent with the solver's T(e).
TEST(FlameletGasTest, HostStatesAreConsistentWithTheTable) {
    const flamelet::FlameletModel model(synthetic_table(), false);
    const double p = 1.2e5, T = 850.0, u[3] = {12.0, -3.0, 4.0}, Z = 0.3;
    const double C = 0.4 * model.burnt_progress(Z);
    EXPECT_NEAR(model.burnt_progress(Z), exact(0, Z, 0.0, 1.0), 1e-15);
    double U[N_CONSERVATIVE], rhoY[2];
    model.conservatives(p, T, u, Z, C, U, rhoY);
    const double R = exact(5, Z, 0.0, 0.4);
    EXPECT_NEAR(U[0], p / (R * T), 1e-12 * U[0]);
    double kinetic = 0.0;
    FOR_I_DIM {
        EXPECT_NEAR(U[1 + i], U[0] * u[i], 1e-12 * std::abs(U[0] * u[i]));
        kinetic += 0.5 * u[i] * u[i];
    }
    const double e = U[N_DIM + 1] / U[0] - kinetic;
    EXPECT_NEAR(model.thermo(Z, C).temperature(e), T, 1e-9 * T);
    EXPECT_NEAR(rhoY[0], U[0] * Z, 1e-15);
    EXPECT_NEAR(rhoY[1], U[0] * C, 1e-15);
    double gamma, e0;
    model.surrogates(T, Z, C, gamma, e0);
    EXPECT_NEAR(gamma, exact(6, Z, 0.0, 0.4) + exact(7, Z, 0.0, 0.4) * (T - exact(3, Z, 0.0, 0.4)), 1e-12);
    EXPECT_NEAR(U[0] * e, p / (gamma - 1.0) + U[0] * e0, 1e-9 * std::abs(U[0] * e));
}

#ifdef Mallard_HAS_HDF5
// A float32 table, widened for the solver's float64 lookups, looks up as its float64 source to
// float32 rounding: catches a widened copy with the wrong strides or order.
TEST(FlameletGasTest, Float32TablesLookUpAsTheirSource) {
    const std::string dir = source_dir() + "/test/data/flamelet/";
    const flamelet::FlameletModel f64(std::make_shared<const Table>(Table::read(dir + "fixture_fpv.h5")), true);
    const flamelet::FlameletModel f32(std::make_shared<const Table>(Table::read(dir + "fixture_fpv_f32.h5")), true);
    ASSERT_TRUE(f32.table().single());
    std::mt19937 rng(631);
    std::uniform_real_distribution<double> u(0.0, 1.0);
    for (int i = 0; i < 50; i++) {
        const double Z = u(rng), C = u(rng) * f64.burnt_progress(Z);
        const flamelet::Thermo a = f64.thermo(Z, C), b = f32.thermo(Z, C);
        EXPECT_NEAR(b.T0, a.T0, 1e-6 * a.T0) << i;
        EXPECT_NEAR(b.e0, a.e0, 2e-6 * 1.4e6) << i;
        EXPECT_NEAR(b.R, a.R, 1e-6 * a.R) << i;
        EXPECT_NEAR(b.gamma0, a.gamma0, 1e-6) << i;
    }
}
#endif

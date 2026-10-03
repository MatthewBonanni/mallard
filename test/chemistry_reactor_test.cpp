/**
 * @file chemistry_reactor_test.cpp
 * @author Matthew Bonanni (mbonanni001@gmail.com)
 * @brief RODAS order and stiff accuracy, the reactor Jacobian, and
 *        constant-volume ignition and equilibrium against Cantera.
 * @version 0.3
 * @date 2026-10-03
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 *
 */

#include <gtest/gtest.h>
#include <Kokkos_Core.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "kinetics.h"
#include "mechanism.h"
#include "reactor.h"
#include "rosenbrock.h"
#include "thermo.h"

using namespace chemistry;

namespace {

const std::string SOURCE_DIR = MALLARD_SOURCE_DIR;

// Row-major on every backend: kernels take pointers to rows
template <typename T>
using Rows = Kokkos::View<T **, Kokkos::LayoutRight>;

struct Table {
    std::vector<std::string> columns;
    std::vector<std::vector<double>> rows;

    size_t column(const std::string & name) const {
        return static_cast<size_t>(std::find(columns.begin(), columns.end(), name) - columns.begin());
    }
};

Table read_table(const std::string & file) {
    std::ifstream in(SOURCE_DIR + "/test/data/chemistry/" + file);
    std::string line;
    Table table;
    while (std::getline(in, line)) {
        if (line.empty() || line[0] == '#') continue;
        std::stringstream s(line);
        std::string field;
        if (table.columns.empty()) {
            while (std::getline(s, field, ',')) table.columns.push_back(field);
            continue;
        }
        std::vector<double> row;
        while (std::getline(s, field, ',')) row.push_back(std::stod(field));
        table.rows.push_back(row);
    }
    return table;
}

/** @brief Van der Pol oscillator with mu = 1 (not stiff). */
struct VanDerPol {
    KOKKOS_INLINE_FUNCTION uint32_t size() const { return 2; }
    KOKKOS_INLINE_FUNCTION double atol(uint32_t) const { return 0.0; }
    KOKKOS_INLINE_FUNCTION bool admissible(const double *) const { return true; }
    KOKKOS_INLINE_FUNCTION void rhs(const double * y, double * f) const {
        f[0] = y[1];
        f[1] = (1.0 - y[0] * y[0]) * y[1] - y[0];
    }
    KOKKOS_INLINE_FUNCTION void rhs_jacobian(const double * y, double * f, double * J) const {
        rhs(y, f);
        J[0] = 0.0;
        J[1] = 1.0;
        J[2] = -2.0 * y[0] * y[1] - 1.0;
        J[3] = 1.0 - y[0] * y[0];
    }
};

/** @brief Robertson's stiff chemical kinetics problem (Hairer & Wanner II, IV.1). */
struct Robertson {
    KOKKOS_INLINE_FUNCTION uint32_t size() const { return 3; }
    KOKKOS_INLINE_FUNCTION double atol(uint32_t) const { return 1e-14; }
    KOKKOS_INLINE_FUNCTION bool admissible(const double *) const { return true; }
    KOKKOS_INLINE_FUNCTION void rhs(const double * y, double * f) const {
        f[0] = -0.04 * y[0] + 1e4 * y[1] * y[2];
        f[2] = 3e7 * y[1] * y[1];
        f[1] = -f[0] - f[2];
    }
    KOKKOS_INLINE_FUNCTION void rhs_jacobian(const double * y, double * f, double * J) const {
        rhs(y, f);
        J[0] = -0.04;
        J[1] = 1e4 * y[2];
        J[2] = 1e4 * y[1];
        J[6] = 0.0;
        J[7] = 6e7 * y[1];
        J[8] = 0.0;
        for (int j = 0; j < 3; j++) J[3 + j] = -J[j] - J[6 + j];
    }
};

/** @brief Van der Pol at t = 1 from (2, 0) with n fixed RODAS steps; err_1 is the first step's error estimate. */
std::array<double, 2> van_der_pol(const uint32_t n_steps, double & err_1) {
    const VanDerPol system;
    std::array<double, 2> y = {2.0, 0.0};
    const double h = 1.0 / n_steps;
    double J[4], LU[4], f0[2], y_new[2], f_tmp[2], k_data[12];
    uint32_t pivot[2];
    double * k[6];
    for (int s = 0; s < 6; s++) k[s] = k_data + 2 * s;
    for (uint32_t step = 0; step < n_steps; step++) {
        system.rhs_jacobian(y.data(), f0, J);
        for (int a = 0; a < 4; a++) LU[a] = -J[a];
        LU[0] += 1.0 / (Rodas::gamma * h);
        LU[3] += 1.0 / (Rodas::gamma * h);
        lu_factor(2, LU, pivot);
        rodas_step(system, 2, h, LU, pivot, y.data(), f0, k, y_new, f_tmp);
        if (step == 0) err_1 = std::hypot(k[5][0], k[5][1]);
        y = {y_new[0], y_new[1]};
    }
    return y;
}

/** @brief Ignition of every row of a reference table on the device. */
struct IgnitionResults {
    std::vector<double> tau, T_half, T_2tau, T_eq;
    std::vector<std::vector<double>> Y_eq;
    uint32_t steps = 0, failures = 0;
};

IgnitionResults device_ignition(const Mechanism & mech, const Table & ref) {
    const ThermoTable<> thermo = make_thermo_table(mech);
    const KineticsTable<> kinetics = make_kinetics_table(mech);
    const uint32_t ns = mech.n_species(), n_cases = ref.rows.size();
    const size_t c_rho = ref.column("rho"), c_Y0 = ref.column("Y0_" + mech.species[0].name);
    const size_t c_T0 = ref.column("T0"), c_tau = ref.column("tau");
    // Per case: rho, T, tau_ref, Y; out: tau, T(tau/2), T(2 tau), steps, failures, T_eq, Y_eq
    Rows<double> state("state", n_cases, 3 + ns), out("out", n_cases, 6 + ns);
    auto h_state = Kokkos::create_mirror_view(state);
    for (uint32_t c = 0; c < n_cases; c++) {
        h_state(c, 0) = ref.rows[c][c_rho];
        h_state(c, 1) = ref.rows[c][c_T0];
        h_state(c, 2) = ref.rows[c][c_tau];
        for (uint32_t k = 0; k < ns; k++) h_state(c, 3 + k) = ref.rows[c][c_Y0 + k];
    }
    Kokkos::deep_copy(state, h_state);
    Rows<double> work("work", n_cases, reactor_work_size(ns, kinetics.n_reactions));
    Rows<uint32_t> pivot("pivot", n_cases, ns + 1);
    const ReactorOptions options;
    Kokkos::parallel_for("ignition", n_cases, KOKKOS_LAMBDA(const uint32_t c) {
        const double rho = state(c, 0), tau_ref = state(c, 2);
        double T = state(c, 1), h = 0.0;
        double * Y = &state(c, 3);
        IgnitionObserver observer;
        observer.index = ns;
        uint32_t steps = 0, failures = 0;
        // Segments end at tau/2, 2 tau and 1000 tau, like splitting steps
        const double ends[3] = {0.5 * tau_ref, 2.0 * tau_ref, 1000.0 * tau_ref};
        double t = 0.0;
        for (int s = 0; s < 3; s++) {
            observer.t_offset = t;
            const RosenbrockResult r = advance_reactor(thermo, kinetics, rho, ends[s] - t, Y, T, h, options,
                                                       &work(c, 0), &pivot(c, 0), observer);
            steps += r.steps;
            failures += r.status != RosenbrockStatus::SUCCESS;
            t = ends[s];
            if (s < 2) out(c, 1 + s) = T;
        }
        out(c, 0) = observer.t_ignition;
        out(c, 3) = steps;
        out(c, 4) = failures;
        out(c, 5) = T;
        for (uint32_t k = 0; k < ns; k++) out(c, 6 + k) = Y[k];
    });
    auto h_out = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), out);
    IgnitionResults result;
    for (uint32_t c = 0; c < n_cases; c++) {
        result.tau.push_back(h_out(c, 0));
        result.T_half.push_back(h_out(c, 1));
        result.T_2tau.push_back(h_out(c, 2));
        result.steps += static_cast<uint32_t>(h_out(c, 3));
        result.failures += static_cast<uint32_t>(h_out(c, 4));
        result.T_eq.push_back(h_out(c, 5));
        result.Y_eq.emplace_back();
        for (uint32_t k = 0; k < ns; k++) result.Y_eq.back().push_back(h_out(c, 6 + k));
    }
    return result;
}

} // namespace

TEST(ChemistryReactorTest, RodasIsFourthOrderWithThirdOrderEmbeddedSolution) {
    double err_ref;
    const auto reference = van_der_pol(4096, err_ref);
    double error[2], estimate[2];
    for (int i = 0; i < 2; i++) {
        const auto y = van_der_pol(32u << i, estimate[i]);
        error[i] = std::hypot(y[0] - reference[0], y[1] - reference[1]);
    }
    const double order = std::log2(error[0] / error[1]);
    EXPECT_GT(order, 3.8);
    EXPECT_LT(order, 4.3);
    // Local error of the order-3 embedded solution: h^4
    const double estimate_order = std::log2(estimate[0] / estimate[1]);
    EXPECT_GT(estimate_order, 3.8);
    EXPECT_LT(estimate_order, 4.3);
}

TEST(ChemistryReactorTest, RodasSolvesTheStiffRobertsonProblem) {
    const Robertson system;
    double y[3] = {1.0, 0.0, 0.0}, work[rosenbrock_work_size(3)], h = 0.0;
    uint32_t pivot[3];
    RosenbrockOptions options;
    options.rtol = 1e-8;
    const RosenbrockResult r = integrate(system, 0.0, 40.0, y, h, options, work, pivot);
    ASSERT_EQ(r.status, RosenbrockStatus::SUCCESS);
    // Hairer & Wanner, Solving ODEs II, reference solution at t = 40
    const double reference[3] = {0.7158270687193e+00, 0.9185534764529e-05, 0.2841637457462e+00};
    for (int i = 0; i < 3; i++) EXPECT_NEAR(y[i], reference[i], 1e-6 * reference[i]) << i;
    EXPECT_LT(r.steps + r.rejected, 1000u);  // about 460 at this tolerance; an unstable scheme needs millions
}

TEST(ChemistryReactorTest, ReactorJacobianMatchesFiniteDifferences) {
    const std::vector<std::array<std::string, 3>> cases = {
        {"h2o2", SOURCE_DIR + "/mechanisms/h2o2.yaml", "ohmech"},
        {"gri30", SOURCE_DIR + "/mechanisms/gri30.yaml", ""},
        {"test_kinetics", SOURCE_DIR + "/test/data/chemistry/test_kinetics.yaml", "gas"},
    };
    for (const auto & [name, file, phase] : cases) {
        const Mechanism mech = read_mechanism(file, phase);
        const auto thermo = make_thermo_table<Kokkos::HostSpace>(mech);
        const auto kinetics = make_kinetics_table<Kokkos::HostSpace>(mech);
        const uint32_t ns = mech.n_species(), n = ns + 1;
        const Table rates = read_table(name + "_rates.csv");
        std::vector<double> scratch(ConstantVolumeReactor<Kokkos::HostSpace>::scratch_size(ns, kinetics.n_reactions));
        std::vector<double> y(n), f(n), J(n * n), yp(n), fp(n), fm(n), D1(n), D2(n);
        for (size_t s = 0; s < std::min<size_t>(rates.rows.size(), 10); s++) {
            const ConstantVolumeReactor<Kokkos::HostSpace> reactor{thermo, kinetics, rates.rows[s][1], 1e-10,
                                                                   scratch.data()};
            for (uint32_t k = 0; k < ns; k++) y[k] = rates.rows[s][2 + k];
            y[ns] = rates.rows[s][0];
            reactor.rhs_jacobian(y.data(), f.data(), J.data());
            // Centered differences with Richardson extrapolation (one-sided near Y_j = 0)
            std::vector<double> FD(n * n);
            for (uint32_t j = 0; j < n; j++) {
                const double d = j < ns ? 1e-4 * std::max(y[j], 1e-6) : 1e-5 * y[j];
                const bool centered = y[j] > 2.0 * d;
                auto difference = [&](const double dj, std::vector<double> & out) {
                    yp = y;
                    yp[j] += dj;
                    reactor.rhs(yp.data(), fp.data());
                    yp[j] = y[j] - (centered ? dj : 0.0);
                    reactor.rhs(yp.data(), fm.data());
                    for (uint32_t i = 0; i < n; i++) out[i] = (fp[i] - fm[i]) / (centered ? 2.0 * dj : dj);
                };
                difference(d, D1);
                difference(0.5 * d, D2);
                for (uint32_t i = 0; i < n; i++) {
                    FD[i * n + j] = centered ? (4.0 * D2[i] - D1[i]) / 3.0 : 2.0 * D2[i] - D1[i];
                }
            }
            for (uint32_t i = 0; i < n; i++) {
                double row_norm = 0.0;
                for (uint32_t j = 0; j < n; j++) row_norm = std::max(row_norm, std::abs(FD[i * n + j]));
                for (uint32_t j = 0; j < n; j++) {
                    EXPECT_NEAR(J[i * n + j], FD[i * n + j], 1e-6 * row_norm + 1e-300)
                        << name << " state " << s << " d f_" << i << " / d y_" << j;
                }
            }
        }
    }
}

TEST(ChemistryReactorTest, IgnitionDelaysAndEquilibriumMatchCantera) {
    // V1/V2: H2/air (h2o2) and CH4/air (GRI-3.0), T0 1000-1500 K, phi 0.5-2,
    // 1 and 10 atm, default tolerances, integrated in device kernels
    const std::vector<std::array<std::string, 3>> cases = {
        {"h2o2", SOURCE_DIR + "/mechanisms/h2o2.yaml", "ohmech"},
        {"gri30", SOURCE_DIR + "/mechanisms/gri30.yaml", ""},
    };
    for (const auto & [name, file, phase] : cases) {
        const Mechanism mech = read_mechanism(file, phase);
        const uint32_t ns = mech.n_species();
        const Table ref = read_table(name + "_ignition.csv");
        ASSERT_EQ(ref.rows.size(), 18u) << name;
        const IgnitionResults r = device_ignition(mech, ref);
        EXPECT_EQ(r.failures, 0u) << name;
        const size_t c_Y = ref.column("Yeq_" + mech.species[0].name);
        double worst_tau = 0.0;
        for (size_t c = 0; c < ref.rows.size(); c++) {
            const auto & row = ref.rows[c];
            const std::string where = name + " T0 " + std::to_string(row[0]) + " p0 " + std::to_string(row[1]) +
                                      " phi " + std::to_string(row[2]);
            const double tau = row[ref.column("tau")];
            worst_tau = std::max(worst_tau, std::abs(r.tau[c] / tau - 1.0));
            EXPECT_NEAR(r.tau[c], tau, 5e-3 * tau) << where;
            EXPECT_NEAR(r.T_half[c], row[ref.column("T_half_tau")], 1e-2 * row[ref.column("T_half_tau")]) << where;
            EXPECT_NEAR(r.T_2tau[c], row[ref.column("T_2tau")], 1e-2 * row[ref.column("T_2tau")]) << where;
            EXPECT_NEAR(r.T_eq[c], row[ref.column("T_eq")], 0.1) << where;
            for (uint32_t k = 0; k < ns; k++) {
                EXPECT_NEAR(r.Y_eq[c][k], row[c_Y + k], 1e-4) << where << " Y_" << mech.species[k].name;
            }
        }
        std::cout << name << ": max ignition delay error " << worst_tau << ", " << r.steps << " sub-steps\n";
    }
}

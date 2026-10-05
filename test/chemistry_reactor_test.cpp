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
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "kinetics.h"
#include "mechanism.h"
#include "reactor.h"
#include "rosenbrock.h"
#include "sparse_lu.h"
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
    template <typename L> KOKKOS_INLINE_FUNCTION bool admissible(const L &, const double *) const { return true; }
    template <typename L> KOKKOS_INLINE_FUNCTION void rhs(const L &, const double * y, double * f) const {
        f[0] = y[1];
        f[1] = (1.0 - y[0] * y[0]) * y[1] - y[0];
    }
    template <typename L> KOKKOS_INLINE_FUNCTION void rhs_jacobian(const L & lanes, const double * y, double * f,
                                                                    double * J) const {
        rhs(lanes, y, f);
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
    template <typename L> KOKKOS_INLINE_FUNCTION bool admissible(const L &, const double *) const { return true; }
    template <typename L> KOKKOS_INLINE_FUNCTION void rhs(const L &, const double * y, double * f) const {
        f[0] = -0.04 * y[0] + 1e4 * y[1] * y[2];
        f[2] = 3e7 * y[1] * y[1];
        f[1] = -f[0] - f[2];
    }
    template <typename L> KOKKOS_INLINE_FUNCTION void rhs_jacobian(const L & lanes, const double * y, double * f,
                                                                    double * J) const {
        rhs(lanes, y, f);
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
    const SerialLanes lanes;
    const DenseLU solver{2, LU, pivot};
    for (uint32_t step = 0; step < n_steps; step++) {
        system.rhs_jacobian(lanes, y.data(), f0, J);
        solver.factor(lanes, J, 1.0 / (Rodas::gamma * h));
        rodas_step(lanes, system, solver, 2, h, y.data(), f0, k, y_new, f_tmp);
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
    Rows<double> work("work", n_cases, reactor_work_size(kinetics));
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

/**
 * @brief Each case's reactor over [0, 2 tau] by one thread (lanes = 0) or by
 *        the vector lanes of a team: final T, Y and the sub-step count.
 */
struct LanesIgnitionFunctor {
    struct TeamTag {};
    using Member = Kokkos::TeamPolicy<TeamTag>::member_type;
    ThermoTable<> thermo;
    KineticsTable<> kinetics;
    Rows<double> state;  // rho, T, tau, Y
    Rows<double> work;
    Rows<uint32_t> pivot;
    Rows<double> out;    // T, steps, Y
    uint32_t lanes;  // threads x vector lanes of a team

    KOKKOS_INLINE_FUNCTION void operator()(const uint32_t c) const {
        const uint32_t ns = thermo.n_species;
        double * Y = &work(c, 0);
        for (uint32_t k = 0; k < ns; k++) Y[k] = state(c, 3 + k);
        double T = state(c, 1), h = 0.0;
        const RosenbrockResult r = advance_reactor(thermo, kinetics, state(c, 0), 2.0 * state(c, 2), Y, T, h,
                                                   ReactorOptions(), Y + ns, &pivot(c, 0));
        out(c, 0) = T;
        out(c, 1) = r.steps + r.rejected + (r.status == RosenbrockStatus::SUCCESS ? 0.0 : 1e9);
        for (uint32_t k = 0; k < ns; k++) out(c, 2 + k) = Y[k];
    }

    KOKKOS_INLINE_FUNCTION void operator()(TeamTag, const Member & member) const {
        const uint32_t c = static_cast<uint32_t>(member.league_rank()), ns = thermo.n_species;
        const TeamLanes<Member> team(member, lanes);
        double * Y = &work(c, 0);
        team.for_each(ns, [&](const uint32_t k) { Y[k] = state(c, 3 + k); });
        team.sync();
        double T = state(c, 1), h = 0.0;
        const RosenbrockResult r = advance_reactor(team, thermo, kinetics, state(c, 0), 2.0 * state(c, 2), Y, T, h,
                                                   ReactorOptions(), Y + ns, &pivot(c, 0));
        team.for_each(ns, [&](const uint32_t k) { out(c, 2 + k) = Y[k]; });
        team.single([&]() {
            out(c, 0) = T;
            out(c, 1) = r.steps + r.rejected + (r.status == RosenbrockStatus::SUCCESS ? 0.0 : 1e9);
        });
    }
};

Rows<double>::host_mirror_type lanes_ignition(const Mechanism & mech, const Table & ref, const uint32_t lanes,
                                              const uint32_t threads = 1) {
    const uint32_t ns = mech.n_species(), n_cases = ref.rows.size();
    const KineticsTable<> kinetics = make_kinetics_table(mech);
    Rows<double> state("state", n_cases, 3 + ns), out("out", n_cases, 2 + ns);
    auto h_state = Kokkos::create_mirror_view(state);
    const size_t c_Y0 = ref.column("Y0_" + mech.species[0].name);
    for (uint32_t c = 0; c < n_cases; c++) {
        h_state(c, 0) = ref.rows[c][ref.column("rho")];
        h_state(c, 1) = ref.rows[c][ref.column("T0")];
        h_state(c, 2) = ref.rows[c][ref.column("tau")];
        for (uint32_t k = 0; k < ns; k++) h_state(c, 3 + k) = ref.rows[c][c_Y0 + k];
    }
    Kokkos::deep_copy(state, h_state);
    const LanesIgnitionFunctor functor{make_thermo_table(mech), kinetics, state,
                                       Rows<double>("work", n_cases, ns + reactor_work_size(kinetics)),
                                       Rows<uint32_t>("pivot", n_cases, ns + 1), out, threads * lanes};
    if (lanes == 0) {
        Kokkos::parallel_for("ignition_threads", n_cases, functor);
    } else {
        const auto policy =
            Kokkos::TeamPolicy<LanesIgnitionFunctor::TeamTag>(n_cases, static_cast<int>(threads), static_cast<int>(lanes))
                .set_scratch_size(0, Kokkos::PerTeam(TeamLanes<LanesIgnitionFunctor::Member>::scratch_bytes(threads * lanes)));
        Kokkos::parallel_for("ignition_teams", policy, functor);
    }
    return Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), out);
}

} // namespace

TEST(ChemistryReactorTest, VectorLanesIntegrateLikeOneThread) {
    // The team path (rates, Jacobian rows, LU and solves across lanes, reductions
    // in a fixed order) gives the one-thread result to round-off, with the
    // largest vector length of the backend (a warp, 32 or 64 lanes, on GPUs; 1
    // on host backends); on GPUs also with several warps per cell, as for large
    // mechanisms
    constexpr bool gpu = !Kokkos::SpaceAccessibility<Kokkos::DefaultExecutionSpace, Kokkos::HostSpace>::accessible;
    const uint32_t lanes_max = static_cast<uint32_t>(Kokkos::TeamPolicy<>::vector_length_max());
    const uint32_t lanes = gpu ? lanes_max : std::min<uint32_t>(32, lanes_max);
    for (const auto & [name, file, phase] : {std::array<std::string, 3>{"h2o2", SOURCE_DIR + "/mechanisms/h2o2.yaml", "ohmech"},
                                            std::array<std::string, 3>{"gri30", SOURCE_DIR + "/mechanisms/gri30.yaml", ""}}) {
        const Mechanism mech = read_mechanism(file, phase);
        const Table ref = read_table(name + "_ignition.csv");
        const auto one = lanes_ignition(mech, ref, 0);
        const auto team = lanes_ignition(mech, ref, lanes);
        for (uint32_t c = 0; c < ref.rows.size(); c++) {
            EXPECT_LT(one(c, 1), 1e9) << name << " case " << c;
            EXPECT_NEAR(team(c, 0), one(c, 0), 1e-6 * one(c, 0)) << name << " case " << c;
            EXPECT_NEAR(team(c, 1), one(c, 1), 0.1 * one(c, 1) + 2) << name << " case " << c;
            for (uint32_t k = 0; k < mech.n_species(); k++) {
                EXPECT_NEAR(team(c, 2 + k), one(c, 2 + k), 1e-6) << name << " case " << c << " Y_" << k;
            }
        }
        // Several warps per cell (the expensive cells of large mechanisms) give the same bits as one
        if (gpu) {
            const auto wide = lanes_ignition(mech, ref, lanes, 4);
            for (uint32_t c = 0; c < ref.rows.size(); c++) {
                for (uint32_t i = 0; i < 2 + mech.n_species(); i++) EXPECT_EQ(wide(c, i), team(c, i)) << name << " case " << c << " " << i;
            }
        }
    }
}

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
    double y[3] = {1.0, 0.0, 0.0}, J[9], vectors[rosenbrock_vectors_size(3)], LU[9], h = 0.0;
    uint32_t pivot[3];
    RosenbrockOptions options;
    options.rtol = 1e-8;
    const RosenbrockResult r = integrate(SerialLanes(), system, DenseLU{3, LU, pivot}, 0.0, 40.0, y, h, options, J, vectors);
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
        {"propane_2step", SOURCE_DIR + "/test/data/chemistry/propane_2step.yaml", "gas"},
    };
    for (const auto & [name, file, phase] : cases) {
        const Mechanism mech = read_mechanism(file, phase);
        const auto thermo = make_thermo_table<Kokkos::HostSpace>(mech);
        const auto kinetics = make_kinetics_table<Kokkos::HostSpace>(mech);
        const uint32_t ns = mech.n_species(), n = ns + 1;
        const Table rates = read_table(name + "_rates.csv");
        std::vector<double> scratch(ConstantVolumeReactor<Kokkos::HostSpace>::scratch_size(kinetics));
        std::vector<double> y(n), f(n), J(n * n), yp(n), fp(n), fm(n), D1(n), D2(n);
        for (size_t s = 0; s < std::min<size_t>(rates.rows.size(), 10); s++) {
            const ConstantVolumeReactor<Kokkos::HostSpace> reactor{thermo, kinetics, rates.rows[s][1], 1e-10,
                                                                   scratch.data(), nullptr, nullptr};
            for (uint32_t k = 0; k < ns; k++) y[k] = rates.rows[s][2 + k];
            y[ns] = rates.rows[s][0];
            reactor.rhs_jacobian(SerialLanes(), y.data(), f.data(), J.data());
            // Centered differences with Richardson extrapolation (one-sided near Y_j = 0)
            std::vector<double> FD(n * n);
            for (uint32_t j = 0; j < n; j++) {
                const double d = j < ns ? 1e-4 * std::max(y[j], 1e-6) : 1e-5 * y[j];
                const bool centered = y[j] > 2.0 * d;
                auto difference = [&](const double dj, std::vector<double> & out) {
                    yp = y;
                    yp[j] += dj;
                    reactor.rhs(SerialLanes(), yp.data(), fp.data());
                    yp[j] = y[j] - (centered ? dj : 0.0);
                    reactor.rhs(SerialLanes(), yp.data(), fm.data());
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

TEST(ChemistryReactorTest, FractionalOrdersIgniteToDepletionLikeCantera) {
    // Westbrook-Dryer two-step propane/air (orders 0.1, 0.25, 0.5 and 1.65),
    // the cases of V1: a reactant runs out (C3H8 lean, O2 rich, both at phi = 1),
    // and H2O and CO start at zero in the CO oxidation [CO] [H2O]^0.5 [O2]^0.25.
    // The end state is Cantera's reactor at 1000 delays (irreversible mechanism)
    const Mechanism mech = read_mechanism(SOURCE_DIR + "/test/data/chemistry/propane_2step.yaml", "gas");
    const uint32_t ns = mech.n_species();
    const Table ref = read_table("propane_2step_ignition.csv");
    ASSERT_EQ(ref.rows.size(), 18u);
    const IgnitionResults r = device_ignition(mech, ref);
    EXPECT_EQ(r.failures, 0u);
    const size_t c_Y = ref.column("Yeq_" + mech.species[0].name);
    double worst_tau = 0.0;
    for (size_t c = 0; c < ref.rows.size(); c++) {
        const auto & row = ref.rows[c];
        const std::string where = "T0 " + std::to_string(row[0]) + " p0 " + std::to_string(row[1]) + " phi " +
                                  std::to_string(row[2]);
        const double tau = row[ref.column("tau")];
        worst_tau = std::max(worst_tau, std::abs(r.tau[c] / tau - 1.0));
        EXPECT_NEAR(r.tau[c], tau, 5e-3 * tau) << where;
        EXPECT_NEAR(r.T_half[c], row[ref.column("T_half_tau")], 1e-2 * row[ref.column("T_half_tau")]) << where;
        EXPECT_NEAR(r.T_2tau[c], row[ref.column("T_2tau")], 1e-2 * row[ref.column("T_2tau")]) << where;
        EXPECT_NEAR(r.T_eq[c], row[ref.column("T_eq")], 0.1) << where;
        for (uint32_t k = 0; k < ns; k++) {
            EXPECT_NEAR(r.Y_eq[c][k], row[c_Y + k], 1e-6) << where << " Y_" << mech.species[k].name;
        }
    }
    std::cout << "propane_2step: max ignition delay error " << worst_tau << ", " << r.steps << " sub-steps\n";
}

TEST(ChemistryReactorTest, SparseLUSolvesLikeTheDenseOne) {
    // J = J_s + u v^T: J_s on the static pattern (in its compact layout), u v^T
    // the columns shared by every species (third bodies at their default
    // efficiency, PLOG and Chebyshev pressures); the pattern's LU without
    // pivoting plus Sherman-Morrison solves (d I - J) x = b as the dense LU does
    const std::vector<std::array<std::string, 3>> cases = {
        {"gri30", SOURCE_DIR + "/mechanisms/gri30.yaml", ""},
        {"test_kinetics", SOURCE_DIR + "/test/data/chemistry/test_kinetics.yaml", "gas"},
    };
    for (const auto & [name, file, phase] : cases) {
        const Mechanism mech = read_mechanism(file, phase);
        const auto thermo = make_thermo_table<Kokkos::HostSpace>(mech);
        const auto kinetics = make_kinetics_table<Kokkos::HostSpace>(mech);
        const auto pattern = make_sparse_lu_pattern<Kokkos::HostSpace>(mech);
        const uint32_t ns = mech.n_species(), n = ns + 1;
        std::vector<char> in_pattern(n * n, 0);
        for (uint32_t e = 0; e < pattern.nnz; e++) in_pattern[pattern.source(e)] = 1;
        const Table rates = read_table(name + "_rates.csv");
        std::vector<double> scratch(ConstantVolumeReactor<Kokkos::HostSpace>::scratch_size(kinetics));
        std::vector<double> y(n), f(n), J(n * n), Js(n * n), u(n), LU(n * n), values(pattern.nnz), z(n), x(n),
            b(n), b_dense(n), compact(pattern.jacobian_size()), u_compact(n);
        std::vector<uint32_t> pivot(n);
        double beta;
        for (size_t s = 0; s < std::min<size_t>(rates.rows.size(), 5); s++) {
            for (uint32_t k = 0; k < ns; k++) y[k] = rates.rows[s][2 + k];
            y[ns] = rates.rows[s][0];
            const double rho = rates.rows[s][1];
            const ConstantVolumeReactor<Kokkos::HostSpace> full{thermo, kinetics, rho, 1e-10, scratch.data(), nullptr,
                                                                nullptr};
            full.rhs_jacobian(SerialLanes(), y.data(), f.data(), J.data());
            const ConstantVolumeReactor<Kokkos::HostSpace> split{thermo, kinetics, rho, 1e-10, scratch.data(), u.data(),
                                                                 nullptr};
            split.rhs_jacobian(SerialLanes(), y.data(), f.data(), Js.data());
            const ConstantVolumeReactor<Kokkos::HostSpace> sparse_reactor{
                thermo, kinetics, rho, 1e-10, scratch.data(), u_compact.data(), &pattern};
            sparse_reactor.rhs_jacobian(SerialLanes(), y.data(), f.data(), compact.data());
            // The compact layout holds J_s's species entries, T column and T row
            const uint32_t ne = pattern.n_entries;
            for (uint32_t e = 0; e < ne; e++) {
                const uint32_t i = pattern.entry_row(e), j = pattern.entry_column(e);
                EXPECT_NEAR(compact[e], Js[i * n + j], 1e-12 * std::abs(Js[i * n + j]) + 1e-300) << name << " " << i << " " << j;
            }
            for (uint32_t k = 0; k < ns; k++) EXPECT_NEAR(compact[ne + k], Js[k * n + ns], 1e-12 * std::abs(Js[k * n + ns]));
            for (uint32_t j = 0; j < n; j++) {
                EXPECT_NEAR(compact[ne + ns + j], Js[ns * n + j], 1e-12 * std::abs(Js[ns * n + j]) + 1e-300);
            }
            for (uint32_t k = 0; k < n; k++) EXPECT_NEAR(u_compact[k], u[k], 1e-12 * std::abs(u[k]));
            for (uint32_t i = 0; i < n; i++) {
                double row_norm = 0.0;
                for (uint32_t j = 0; j < n; j++) row_norm = std::max(row_norm, std::abs(J[i * n + j]));
                for (uint32_t j = 0; j < n; j++) {
                    EXPECT_NEAR(Js[i * n + j] + u[i] * pattern.v(j), J[i * n + j], 1e-12 * row_norm)
                        << name << " state " << s << " J " << i << " " << j;
                    if (!in_pattern[i * n + j]) {
                        EXPECT_EQ(Js[i * n + j], 0.0) << name << " outside the pattern " << i << " " << j;
                    }
                }
            }
            // A step size where the Newton matrix is far from diagonal
            const double diagonal = 1.0 / (Rodas::gamma * 1e-5);
            const DenseLU dense{n, LU.data(), pivot.data()};
            const SparseLU<Kokkos::HostSpace> sparse{pattern, values.data(), z.data(), u.data(), x.data(), &beta};
            ASSERT_TRUE(dense.factor(SerialLanes(), J.data(), diagonal));
            ASSERT_TRUE(sparse.factor(SerialLanes(), compact.data(), diagonal));
            for (uint32_t i = 0; i < n; i++) b[i] = b_dense[i] = std::sin(1.0 + i);
            dense.solve(SerialLanes(), b_dense.data());
            sparse.solve(SerialLanes(), b.data());
            double largest = 0.0;
            for (uint32_t i = 0; i < n; i++) largest = std::max(largest, std::abs(b_dense[i]));
            for (uint32_t i = 0; i < n; i++) EXPECT_NEAR(b[i], b_dense[i], 1e-9 * largest) << name << " x_" << i;
        }
        std::cout << name << ": " << pattern.nnz << " entries in L + U of " << n * n << ", "
                  << pattern.update.extent(0) << " updates\n";
    }
}

TEST(ChemistryReactorTest, LargeMechanismIgnitesLikeCanteraWithDenseAndSparseLU) {
    // V3 at 100 species (n-dodecane/air, Wang et al., 20 atm, phi 0.5-2,
    // 1000-1400 K): ignition delays and temperatures as V1, with the dense LU
    // and with the static-pattern sparse one
    const Mechanism mech = read_mechanism(SOURCE_DIR + "/mechanisms/nDodecane_Reitz.yaml", "nDodecane_IG");
    const auto thermo = make_thermo_table<Kokkos::HostSpace>(mech);
    const auto kinetics = make_kinetics_table<Kokkos::HostSpace>(mech);
    const auto pattern = make_sparse_lu_pattern<Kokkos::HostSpace>(mech);
    const uint32_t ns = mech.n_species();
    const Table ref = read_table("ndodecane_ignition.csv");
    ASSERT_EQ(ref.rows.size(), 9u);
    std::vector<double> work(reactor_work_size(kinetics, &pattern) + reactor_work_size(kinetics)), Y(ns);
    std::vector<uint32_t> pivot(ns + 1);
    for (const bool sparse : {false, true}) {
        double worst = 0.0;
        for (const auto & row : ref.rows) {
            for (uint32_t k = 0; k < ns; k++) Y[k] = row[ref.column("Y0_" + mech.species[k].name)];
            const double rho = row[ref.column("rho")], tau = row[ref.column("tau")];
            double T = row[ref.column("T0")], h = 0.0, t = 0.0;
            IgnitionObserver observer;
            observer.index = ns;
            double T_at[2];
            // (Irreversible soot-precursor reactions keep this mechanism from its UV equilibrium)
            const double ends[2] = {0.5 * tau, 2.0 * tau};
            for (int s = 0; s < 2; s++) {
                observer.t_offset = t;
                const RosenbrockResult r = advance_reactor(thermo, kinetics, rho, ends[s] - t, Y.data(), T, h,
                                                           ReactorOptions(), work.data(), pivot.data(), observer,
                                                           sparse ? &pattern : nullptr);
                ASSERT_EQ(r.status, RosenbrockStatus::SUCCESS);
                t = ends[s];
                T_at[s] = T;
            }
            const std::string where = (sparse ? "sparse" : "dense") + std::string(" T0 ") +
                                      std::to_string(row[0]) + " phi " + std::to_string(row[2]);
            worst = std::max(worst, std::abs(observer.t_ignition / tau - 1.0));
            EXPECT_NEAR(observer.t_ignition, tau, 5e-3 * tau) << where;
            EXPECT_NEAR(T_at[0], row[ref.column("T_half_tau")], 1e-2 * row[ref.column("T_half_tau")]) << where;
            EXPECT_NEAR(T_at[1], row[ref.column("T_2tau")], 1e-2 * row[ref.column("T_2tau")]) << where;
        }
        std::cout << (sparse ? "sparse" : "dense") << " LU: max ignition delay error " << worst << "\n";
    }
}

TEST(ChemistryReactorTest, LeanIgnitionIsReportedAtTheSteepestTemperatureRise) {
    // H2/air at phi = 0.1, 1070 K and 41 atm (examples/autoignition_2d) heats
    // by less than 400 K. Its delay is the time of the steepest rise of the
    // trajectory sampled every microsecond; a run that ends before ignition,
    // or before dT/dt has fallen to half its peak, has none
    const Mechanism mech = read_mechanism(SOURCE_DIR + "/mechanisms/h2o2.yaml", "ohmech");
    const auto thermo = make_thermo_table<Kokkos::HostSpace>(mech);
    const auto kinetics = make_kinetics_table<Kokkos::HostSpace>(mech);
    const uint32_t ns = mech.n_species();
    std::vector<double> Y0(ns, 0.0);
    double mass = 0.0;
    for (uint32_t k = 0; k < ns; k++) {
        const std::string & name = mech.species[k].name;
        const double X = name == "H2"   ? 0.040322580645161296
                         : name == "O2" ? 0.20161290322580644
                         : name == "N2" ? 0.7580645161290324
                                        : 0.0;
        Y0[k] = X * mech.species[k].molecular_weight;
        mass += Y0[k];
    }
    for (double & y : Y0) y /= mass;
    const double T0 = 1070.0, p0 = 4154332.5;
    const double rho = p0 / (thermo.gas_constant(MassFractions{Y0.data()}) * T0);
    ReactorOptions options;
    options.integrator.rtol = 1e-8;
    options.atol_Y = 1e-12;
    std::vector<double> work(reactor_work_size(kinetics)), Y;
    std::vector<uint32_t> pivot(ns + 1);
    const double interval = 1e-6;
    // The observer at end_time, and T at every interval
    auto run = [&](double end_time, std::vector<double> & T_rows) {
        Y = Y0;
        double T = T0, h = 0.0, t = 0.0;
        IgnitionObserver observer;
        observer.index = ns;
        T_rows.assign(1, T);
        while (t < end_time) {
            const double t_next = std::min(end_time, t + interval);
            observer.t_offset = t;
            const RosenbrockResult r = advance_reactor(thermo, kinetics, rho, t_next - t, Y.data(), T, h, options,
                                                       work.data(), pivot.data(), observer);
            EXPECT_EQ(r.status, RosenbrockStatus::SUCCESS);
            t = t_next;
            T_rows.push_back(T);
        }
        return observer;
    };

    std::vector<double> T_rows;
    const IgnitionObserver lean = run(6e-3, T_rows);
    EXPECT_LT(T_rows.back() - T0, 400.0);
    size_t steepest = 0;
    for (size_t i = 1; i + 1 < T_rows.size(); i++) {
        if (T_rows[i + 1] - T_rows[i] > T_rows[steepest + 1] - T_rows[steepest]) steepest = i;
    }
    const double t_steepest = (steepest + 0.5) * interval;
    ASSERT_TRUE(lean.ignited());
    EXPECT_NEAR(lean.t_ignition, t_steepest, interval);
    EXPECT_NEAR(lean.t_ignition, 3.650e-3, 5e-3 * 3.650e-3);  // Cantera

    // Runs that end before ignition, and past the steepest rise while dT/dt
    // is still above half its peak
    const double steepest_rise = T_rows[steepest + 1] - T_rows[steepest];
    size_t half = steepest;
    while (T_rows[half + 1] - T_rows[half] > 0.5 * steepest_rise) half++;
    for (const double end_time : {0.5 * t_steepest, 0.5 * (t_steepest + half * interval)}) {
        std::vector<double> rows;
        EXPECT_FALSE(run(end_time, rows).ignited()) << "end_time " << end_time;
    }
}

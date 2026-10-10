/**
 * @file chemistry_kinetics_test.cpp
 * @author Matthew Bonanni (mbonanni001@gmail.com)
 * @brief Reaction rates against Cantera and their analytical Jacobian
 *        against finite differences.
 * @version 0.3
 * @date 2026-10-03
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 *
 */

#include <gtest/gtest.h>
#include <Kokkos_Core.hpp>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "kinetics.h"
#include "mechanism.h"
#include "mixture.h"
#include "test_fixtures.h"
#include "thermo.h"

using namespace chemistry;

namespace {

const std::string SOURCE_DIR = MALLARD_SOURCE_DIR;

// Row-major on every backend: kernels take pointers to rows
template <typename T>
using Rows = Kokkos::View<T **, Kokkos::LayoutRight>;

struct Case {
    std::string name, file, phase;
};

const std::vector<Case> CASES = {
    {"h2o2", SOURCE_DIR + "/mechanisms/h2o2.yaml", "ohmech"},
    {"gri30", SOURCE_DIR + "/mechanisms/gri30.yaml", ""},
    {"test_kinetics", SOURCE_DIR + "/test/data/chemistry/test_kinetics.yaml", "gas"},
    {"propane_2step", SOURCE_DIR + "/test/data/chemistry/propane_2step.yaml", "gas"},
};

std::vector<std::vector<double>> read_rows(const std::string & file) {
    std::ifstream in(SOURCE_DIR + "/test/data/chemistry/" + file);
    std::string line;
    std::vector<std::vector<double>> rows;
    bool header = true;
    while (std::getline(in, line)) {
        if (line.empty() || line[0] == '#') continue;
        if (header) {
            header = false;
            continue;
        }
        std::vector<double> row;
        std::stringstream s(line);
        std::string field;
        while (std::getline(s, field, ',')) row.push_back(std::stod(field));
        rows.push_back(row);
    }
    return rows;
}

/**
 * @brief Concentrations, g/RT and h/RT of a state (T, rho, Y), with the
 *        given tables (host or device).
 */
template <typename Thermo>
KOKKOS_INLINE_FUNCTION void species_state(const Thermo & thermo, const double T, const double rho, const double * Y,
                                          double * C, double * g_RT, double * h_RT) {
    const auto p = Thermo::powers(T);
    for (uint32_t k = 0; k < thermo.n_species; k++) {
        C[k] = rho * Y[k] * thermo.inv_W(k);
        h_RT[k] = thermo.h_RT(k, p);
        g_RT[k] = h_RT[k] - thermo.s_R(k, p);
    }
}

/** @brief Net rates of progress and production rates at each state, on the device. */
void device_rates(const ThermoTable<> & thermo, const KineticsTable<> & kinetics, Rows<double> states,
                  Rows<double> q, Rows<double> omega) {
    const uint32_t ns = thermo.n_species;
    Rows<double> work("work", states.extent(0), 3 * ns);
    Kokkos::parallel_for("rates", states.extent(0), KOKKOS_LAMBDA(const uint32_t s) {
        double * C = &work(s, 0);
        double * g_RT = C + ns;
        double * h_RT = g_RT + ns;
        species_state(thermo, states(s, 0), states(s, 1), &states(s, 2), C, g_RT, h_RT);
        kinetics.rates_of_progress(states(s, 0), C, g_RT, &q(s, 0));
        kinetics.production_rates(&q(s, 0), &omega(s, 0));
    });
}

} // namespace

TEST(ChemistryKineticsTest, RatesOfProgressAndProductionRatesMatchCantera) {
    for (const Case & c : CASES) {
        const Mechanism mech = read_mechanism(c.file, c.phase);
        const ThermoTable<> thermo = make_thermo_table(mech);
        const KineticsTable<> kinetics = make_kinetics_table(mech);
        const uint32_t ns = mech.n_species(), nr = mech.reactions.size();
        const auto rows = read_rows(c.name + "_rates.csv");
        ASSERT_EQ(rows.at(0).size(), 2 + 2 * ns + nr) << c.name;
        Rows<double> states("states", rows.size(), 2 + ns);
        auto h_states = Kokkos::create_mirror_view(states);
        for (size_t s = 0; s < rows.size(); s++) {
            for (uint32_t j = 0; j < 2 + ns; j++) h_states(s, j) = rows[s][j];
        }
        Kokkos::deep_copy(states, h_states);
        Rows<double> q("q", rows.size(), nr), omega("omega", rows.size(), ns);
        device_rates(thermo, kinetics, states, q, omega);
        auto h_q = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), q);
        auto h_omega = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), omega);
        for (size_t s = 0; s < rows.size(); s++) {
            // Relative 1e-10 where the value exceeds 1e-12 of the largest of its kind
            auto check = [&](const auto & values, uint32_t offset, uint32_t n, const char * what) {
                double largest = 0.0;
                for (uint32_t j = 0; j < n; j++) largest = std::max(largest, std::abs(rows[s][offset + j]));
                for (uint32_t j = 0; j < n; j++) {
                    const double ref = rows[s][offset + j];
                    const double tol = std::max(1e-10 * std::abs(ref), 1e-12 * largest);
                    EXPECT_NEAR(values(s, j), ref, tol) << c.name << " state " << s << " " << what << " " << j;
                }
            };
            check(h_q, 2 + ns, nr, "q");
            check(h_omega, 2 + ns + nr, ns, "omega");
        }
    }
}

TEST(ChemistryKineticsTest, AnalyticalJacobianMatchesFiniteDifferences) {
    // d omega / dC (all reaction types, third bodies and falloff included) and
    // d q / dT, against centered differences of the rates
    for (const Case & c : CASES) {
        const Mechanism mech = read_mechanism(c.file, c.phase);
        const auto thermo = make_thermo_table<Kokkos::HostSpace>(mech);
        const auto kinetics = make_kinetics_table<Kokkos::HostSpace>(mech);
        const uint32_t ns = mech.n_species(), nr = mech.reactions.size();
        const auto rows = read_rows(c.name + "_rates.csv");
        std::vector<double> C(ns), g(ns), h(ns), q(nr), J(ns * ns), omega(ns), derivatives(kinetics.derivatives_size());
        const ReactionDerivatives deriv = ReactionDerivatives::at(
            derivatives.data(), nr, kinetics.forward_species.extent(0), kinetics.reverse_species.extent(0));
        const double * dq_dT = deriv.dq_dT;
        std::vector<double> Cp(ns), gp(ns), hp(ns), qp(nr), wp(ns), wm(ns);
        for (size_t s = 0; s < std::min<size_t>(rows.size(), 10); s++) {
            const double T = rows[s][0], rho = rows[s][1];
            species_state(thermo, T, rho, &rows[s][2], C.data(), g.data(), h.data());
            double C_total = 0.0;
            for (double x : C) C_total += x;
            kinetics.rates_of_progress(SerialLanes(), T, C.data(), C_total, g.data(), h.data(), q.data(), &deriv);
            kinetics.production_jacobian(SerialLanes(), deriv, J.data(), ns);
            // omega at C with C_j shifted by d
            auto omega_at = [&](uint32_t j, double d, std::vector<double> & w) {
                Cp = C;
                Cp[j] += d;
                kinetics.rates_of_progress(T, Cp.data(), g.data(), qp.data());
                kinetics.production_rates(qp.data(), w.data());
            };
            // Centered differences with Richardson extrapolation (one-sided near C_j = 0)
            auto difference = [&](uint32_t j, double d, std::vector<double> & out) {
                const bool centered = C[j] > 2.0 * d;
                omega_at(j, d, wp);
                omega_at(j, centered ? -d : 0.0, wm);
                for (uint32_t k = 0; k < ns; k++) out[k] = (wp[k] - wm[k]) / (centered ? 2.0 * d : d);
            };
            std::vector<double> FD(ns * ns), D1(ns), D2(ns);
            for (uint32_t j = 0; j < ns; j++) {
                const double d = 1e-4 * std::max(C[j], 1e-6 * C_total);
                difference(j, d, D1);
                difference(j, 0.5 * d, D2);
                const bool centered = C[j] > 2.0 * d;
                for (uint32_t k = 0; k < ns; k++) {
                    FD[k * ns + j] = centered ? (4.0 * D2[k] - D1[k]) / 3.0 : 2.0 * D2[k] - D1[k];
                }
            }
            for (uint32_t k = 0; k < ns; k++) {
                double row_norm = 0.0;
                for (uint32_t j = 0; j < ns; j++) row_norm = std::max(row_norm, std::abs(FD[k * ns + j]));
                for (uint32_t j = 0; j < ns; j++) {
                    EXPECT_NEAR(J[k * ns + j], FD[k * ns + j], 1e-6 * row_norm + 1e-300)
                        << c.name << " state " << s << " d omega_" << k << " / d C_" << j;
                }
            }
            const double dT = 1e-6 * T;
            for (int sign : {1, -1}) {
                std::vector<double> & qq = sign > 0 ? wp : wm;
                qq.resize(nr);
                species_state(thermo, T + sign * dT, rho, &rows[s][2], Cp.data(), gp.data(), hp.data());
                kinetics.rates_of_progress(T + sign * dT, C.data(), gp.data(), qq.data());
            }
            double largest = 0.0;
            for (uint32_t i = 0; i < nr; i++) largest = std::max(largest, std::abs(dq_dT[i]));
            for (uint32_t i = 0; i < nr; i++) {
                const double fd = (wp[i] - wm[i]) / (2.0 * dT);
                EXPECT_NEAR(dq_dT[i], fd, 1e-6 * std::abs(fd) + 1e-9 * largest)
                    << c.name << " state " << s << " d q_" << i << " / dT";
            }
            wp.resize(ns);
            wm.resize(ns);
        }
    }
}

TEST(ChemistryKineticsTest, FractionalOrdersKeepAFiniteJacobianAsConcentrationsVanish) {
    // Westbrook-Dryer two-step propane: [C3H8]^0.1 [O2]^1.65 and [CO] [H2O]^0.5
    // [O2]^0.25. Each species is taken from above C_REG (exact power law) through
    // the regularized range to zero and slightly negative; d omega / dC stays
    // finite and matches finite differences of the rates
    using Kinetics = KineticsTable<Kokkos::HostSpace>;
    const Mechanism mech = read_mechanism(SOURCE_DIR + "/test/data/chemistry/propane_2step.yaml", "gas");
    const auto thermo = make_thermo_table<Kokkos::HostSpace>(mech);
    const auto kinetics = make_kinetics_table<Kokkos::HostSpace>(mech);
    const uint32_t ns = mech.n_species(), nr = mech.reactions.size();
    const double T = 1800.0, C_reg = Kinetics::C_REG;
    std::vector<double> C0(ns), g(ns), h(ns), q(nr), J(ns * ns), derivatives(kinetics.derivatives_size());
    const std::vector<std::pair<std::string, double>> base = {{"C3H8", 1e-4}, {"O2", 2e-3}, {"CO", 1e-3},
                                                              {"H2O", 2e-3}, {"CO2", 1e-3}, {"N2", 2e-2}};
    for (const auto & [name, value] : base) C0[mech.species_index(name)] = value;
    const std::vector<double> Y(ns, 1.0 / ns);
    std::vector<double> unused(ns);
    species_state(thermo, T, 1.0, Y.data(), unused.data(), g.data(), h.data());
    const ReactionDerivatives deriv = ReactionDerivatives::at(derivatives.data(), nr, kinetics.forward_species.extent(0),
                                                              kinetics.reverse_species.extent(0));
    auto omega_at = [&](const std::vector<double> & C, std::vector<double> & w) {
        std::vector<double> qq(nr);
        kinetics.rates_of_progress(T, C.data(), g.data(), qq.data());
        kinetics.production_rates(qq.data(), w.data());
    };
    for (const char * name : {"C3H8", "H2O", "O2"}) {
        const uint32_t j = mech.species_index(name);
        for (const double x : {-0.5, 0.0, 1e-3, 0.25, 0.5, 0.75, 1.0, 3.0, 1e3, 1e6}) {
            std::vector<double> C = C0;
            C[j] = x * C_reg;
            double C_total = 0.0;
            for (const double c : C) C_total += c;
            kinetics.rates_of_progress(SerialLanes(), T, C.data(), C_total, g.data(), h.data(), q.data(), &deriv);
            kinetics.production_jacobian(SerialLanes(), deriv, J.data(), ns);
            for (uint32_t k = 0; k < ns; k++) {
                ASSERT_TRUE(std::isfinite(J[k * ns + j])) << name << " C = " << x << " C_REG, d omega_" << k;
            }
            // Away from the joins at C = 0 and C_REG, centered differences with Richardson extrapolation
            if (x == 0.0 || x == 1.0) continue;
            const double d = 1e-4 * C_reg * std::max(1.0, std::abs(x) * 1e-2);
            std::vector<double> wp(ns), wm(ns), D[2] = {std::vector<double>(ns), std::vector<double>(ns)};
            for (int r = 0; r < 2; r++) {
                std::vector<double> Cs = C;
                Cs[j] = C[j] + d / (1 << r);
                omega_at(Cs, wp);
                Cs[j] = C[j] - d / (1 << r);
                omega_at(Cs, wm);
                for (uint32_t k = 0; k < ns; k++) D[r][k] = (wp[k] - wm[k]) / (2.0 * d / (1 << r));
            }
            double column_norm = 0.0;
            for (uint32_t k = 0; k < ns; k++) column_norm = std::max(column_norm, std::abs(J[k * ns + j]));
            EXPECT_GT(column_norm, 0.0) << name << " C = " << x << " C_REG";
            for (uint32_t k = 0; k < ns; k++) {
                const double fd = (4.0 * D[1][k] - D[0][k]) / 3.0;
                EXPECT_NEAR(J[k * ns + j], fd, 1e-6 * column_norm) << name << " C = " << x << " C_REG, d omega_" << k;
            }
        }
    }
    // The regularized power is C^1 at both joins and the exact power law above C_REG
    for (const double n : {0.1, 0.25, 0.5}) {
        for (const double at : {0.0, C_reg}) {
            const double below = at - 1e-9 * C_reg, above = at + 1e-9 * C_reg;
            const double slope = Kinetics::power_derivative(above, n);
            EXPECT_NEAR(Kinetics::power(below, n), Kinetics::power(above, n), 1e-8 * slope * C_reg) << n << " " << at;
            EXPECT_NEAR(Kinetics::power_derivative(below, n), slope, 1e-7 * slope) << n << " " << at;
        }
        for (const double x : {1.0, 1.5, 1e4}) {
            EXPECT_DOUBLE_EQ(Kinetics::power(x * C_reg, n), std::pow(x * C_reg, n)) << n << " " << x;
            EXPECT_DOUBLE_EQ(Kinetics::power_derivative(x * C_reg, n), n * std::pow(x * C_reg, n - 1.0)) << n << " " << x;
        }
    }
}

TEST(ChemistryKineticsTest, RegularizationConcentrationIsAnOption) {
    // chemistry.C_reg moves the regularization: below it the C3H8 step's rate
    // follows the regularized law in C3H8, above it the power law
    using Kinetics = KineticsTable<Kokkos::HostSpace>;
    const Mechanism mech = read_mechanism(SOURCE_DIR + "/test/data/chemistry/propane_2step.yaml", "gas");
    const double C_reg = 1e-9;
    const toml::value input = parse_toml("[chemistry]\nC_reg = 1.0e-9\n");
    ASSERT_EQ(reactor_options(input).C_reg, C_reg);
    const auto kinetics = make_kinetics_table<Kokkos::HostSpace>(mech, reactor_options(input).C_reg);
    const auto plain = make_kinetics_table<Kokkos::HostSpace>(mech);
    const uint32_t ns = mech.n_species(), nr = mech.reactions.size(), fuel = mech.species_index("C3H8");
    std::vector<double> C(ns, 1e-3), g(ns, 0.0), q(nr), q_plain(nr);
    for (const double x : {0.5, 2.0}) {
        C[fuel] = x * C_reg;
        kinetics.rates_of_progress(1800.0, C.data(), g.data(), q.data());
        plain.rates_of_progress(1800.0, C.data(), g.data(), q_plain.data());
        // Reaction 0 is C3H8 + 3.5 O2 => 3 CO + 4 H2O with [C3H8]^0.1
        const double ratio = q[0] / q_plain[0];
        const double expected = x < 1.0 ? Kinetics::power(C[fuel], 0.1, C_reg) / std::pow(C[fuel], 0.1) : 1.0;
        EXPECT_NEAR(ratio, expected, 1e-12) << x;
        if (x < 1.0) EXPECT_LT(ratio, 0.9) << x;
    }
}

TEST(ChemistryKineticsTest, ReaderKeepsReactionOptionsAndRejectsUnsupportedTypes) {
    const Mechanism mech = read_mechanism(SOURCE_DIR + "/test/data/chemistry/test_kinetics.yaml", "gas");
    ASSERT_EQ(mech.reactions.size(), 10u);  // both sections
    const Reaction & collider = mech.reactions[2];
    EXPECT_EQ(collider.type, ReactionType::FALLOFF);
    EXPECT_EQ(collider.default_efficiency, 0.0);
    ASSERT_EQ(collider.efficiencies.size(), 1u);
    EXPECT_EQ(collider.efficiencies[0].first, mech.species_index("AR"));
    const Reaction & orders = mech.reactions[5];
    EXPECT_FALSE(orders.reversible);
    ASSERT_EQ(orders.orders.size(), 2u);
    EXPECT_EQ(orders.orders[0].second, 0.5);

    const std::string dir = (std::filesystem::temp_directory_path() / "mallard_kinetics_errors").string();
    std::filesystem::create_directories(dir);
    const std::string file = dir + "/mech.yaml";
    std::ofstream(file) << "phases:\n- name: gas\n  thermo: ideal-gas\n  elements: [H]\n  species: [H2, H]\n"
                           "  kinetics: gas\nspecies:\n"
                           "- {name: H2, composition: {H: 2}, thermo: {model: constant-cp}}\n"
                           "- {name: H, composition: {H: 1}, thermo: {model: constant-cp}}\n"
                           "reactions:\n- equation: H2 <=> 2 H\n  type: chemically-activated\n"
                           "  low-P-rate-constant: {A: 1.0, b: 0.0, Ea: 0.0}\n";
    try {
        read_mechanism(file);
        ADD_FAILURE() << "an unsupported reaction type was accepted";
    } catch (const std::runtime_error & e) {
        EXPECT_NE(std::string(e.what()).find("H2 <=> 2 H"), std::string::npos) << e.what();
    }
    std::filesystem::remove_all(dir);
}

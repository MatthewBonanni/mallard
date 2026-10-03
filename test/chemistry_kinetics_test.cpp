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
        kinetics.rates_of_progress(states(s, 0), C, g_RT, h_RT, &q(s, 0), nullptr, nullptr, nullptr);
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
        std::vector<double> C(ns), g(ns), h(ns), q(nr), dq_dT(nr), kf(nr), kr(nr), J(ns * ns), omega(ns);
        std::vector<double> Cp(ns), gp(ns), hp(ns), qp(nr), wp(ns), wm(ns);
        for (size_t s = 0; s < std::min<size_t>(rows.size(), 10); s++) {
            const double T = rows[s][0], rho = rows[s][1];
            species_state(thermo, T, rho, &rows[s][2], C.data(), g.data(), h.data());
            kinetics.rates_of_progress(T, C.data(), g.data(), h.data(), q.data(), dq_dT.data(), kf.data(), kr.data());
            kinetics.production_jacobian(T, C.data(), kf.data(), kr.data(), J.data());
            double C_total = 0.0;
            for (double x : C) C_total += x;
            // omega at C with C_j shifted by d
            auto omega_at = [&](uint32_t j, double d, std::vector<double> & w) {
                Cp = C;
                Cp[j] += d;
                kinetics.rates_of_progress(T, Cp.data(), g.data(), h.data(), qp.data(), nullptr, nullptr, nullptr);
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
                kinetics.rates_of_progress(T + sign * dT, C.data(), gp.data(), hp.data(), qq.data(), nullptr, nullptr,
                                           nullptr);
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

TEST(ChemistryKineticsTest, ReaderKeepsReactionOptionsAndRejectsUnsupportedTypes) {
    const Mechanism mech = read_mechanism(SOURCE_DIR + "/test/data/chemistry/test_kinetics.yaml", "gas");
    ASSERT_EQ(mech.reactions.size(), 8u);  // both sections
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
                           "reactions:\n- equation: H2 <=> 2 H\n  type: pressure-dependent-Arrhenius\n"
                           "  rate-constants: []\n";
    try {
        read_mechanism(file);
        ADD_FAILURE() << "an unsupported reaction type was accepted";
    } catch (const std::runtime_error & e) {
        EXPECT_NE(std::string(e.what()).find("H2 <=> 2 H"), std::string::npos) << e.what();
    }
    std::filesystem::remove_all(dir);
}

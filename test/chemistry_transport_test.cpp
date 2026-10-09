/**
 * @file chemistry_transport_test.cpp
 * @author Matthew Bonanni (mbonanni001@gmail.com)
 * @brief Species transport fits and mixture-averaged properties against Cantera.
 * @version 0.3
 * @date 2026-10-03
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 *
 */

#include <gtest/gtest.h>
#include <Kokkos_Core.hpp>

#include <cmath>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "mechanism.h"
#include "thermo.h"
#include "transport.h"

using namespace chemistry;

namespace {

const std::string SOURCE_DIR = MALLARD_SOURCE_DIR;

struct Case {
    std::string name, file, phase;
};

const std::vector<Case> CASES = {
    {"h2o2", SOURCE_DIR + "/mechanisms/h2o2.yaml", "ohmech"},
    {"gri30", SOURCE_DIR + "/mechanisms/gri30.yaml", ""},
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

/** @brief Mixture properties of each row's state: [mu, lambda, D_1 .. D_Ns]. */
struct MixtureFunctor {
    ThermoTable<> thermo;
    TransportTable<> transport;
    Kokkos::View<double **, Kokkos::LayoutRight> states;  // (row, [T, p, Y])
    Kokkos::View<double **, Kokkos::LayoutRight> out;

    KOKKOS_INLINE_FUNCTION
    void operator()(const uint32_t i) const {
        const double T = states(i, 0), p = states(i, 1);
        const MassFractions y{&states(i, 2)};
        const double rho = p / (thermo.gas_constant(y) * T);
        const double cp = thermo.cp_mass(T, y);
        transport.properties(T, p, rho, cp, y, out(i, 0), out(i, 1), &out(i, 2),
                             transport.thermal_diffusion ? &out(i, 2 + transport.n_species) : nullptr);
    }
};

} // namespace

TEST(TransportTest, SpeciesFitsMatchCantera) {
    // Viscosity, conductivity and binary diffusion coefficients of the fits,
    // which port Cantera's collision integrals, polar corrections, Parker's
    // rotational relaxation and weighted least squares
    for (const Case & c : CASES) {
        const Mechanism mech = read_mechanism(c.file, c.phase);
        const auto table = make_transport_table<Kokkos::HostSpace>(mech, TransportModel::MIXTURE_AVERAGED);
        const uint32_t ns = static_cast<uint32_t>(mech.n_species());
        const auto rows = read_rows(c.name + "_species_transport.csv");
        ASSERT_FALSE(rows.empty()) << c.name;
        double worst = 0.0;
        for (const auto & r : rows) {
            const double T = r[0];
            size_t col = 1;
            auto check = [&](double value, double ref, const std::string & what) {
                const double err = std::abs(value / ref - 1.0);
                worst = std::max(worst, err);
                EXPECT_LT(err, 1e-6) << c.name << " " << what << " at T = " << T;
            };
            for (uint32_t k = 0; k < ns; k++) check(table.species_viscosity(k, T), r[col++], "mu " + mech.species[k].name);
            for (uint32_t k = 0; k < ns; k++) {
                check(table.species_conductivity(k, T), r[col++], "lambda " + mech.species[k].name);
            }
            for (uint32_t k = 0; k < ns; k++) {
                for (uint32_t j = k; j < ns; j++) {
                    check(table.binary_diffusion(k, j, T, ONE_ATM), r[col++],
                          "D " + mech.species[k].name + "-" + mech.species[j].name);
                }
            }
        }
        std::ostringstream text;
        text << std::scientific << worst;
        RecordProperty(c.name + "_max_rel_error", text.str());
    }
}

TEST(TransportTest, MixturePropertiesMatchCantera) {
    // Wilke viscosity, Mathur conductivity and mixture-averaged diffusion
    // coefficients (with Cantera's floor on mole fractions, so that absent
    // species and a pure species are handled alike), the unity-Lewis
    // diffusivity and constant Lewis numbers, in device kernels
    for (const Case & c : CASES) {
        const Mechanism mech = read_mechanism(c.file, c.phase);
        const uint32_t ns = static_cast<uint32_t>(mech.n_species());
        const auto rows = read_rows(c.name + "_mixture_transport.csv");
        ASSERT_FALSE(rows.empty()) << c.name;
        const uint32_t n = static_cast<uint32_t>(rows.size());
        Kokkos::View<double **, Kokkos::LayoutRight> states("states", n, ns + 2);
        auto h_states = Kokkos::create_mirror_view(states);
        for (uint32_t i = 0; i < n; i++) {
            for (uint32_t m = 0; m < ns + 2; m++) h_states(i, m) = rows[i][m];
        }
        Kokkos::deep_copy(states, h_states);
        const ThermoTable<> thermo = make_thermo_table(mech);
        // Constant Lewis numbers 1, 2, 3, ... divide the unity-Lewis diffusivity
        std::vector<double> lewis(ns);
        for (uint32_t k = 0; k < ns; k++) lewis[k] = 1.0 + k;
        for (const TransportModel model :
             {TransportModel::MIXTURE_AVERAGED, TransportModel::UNITY_LEWIS, TransportModel::CONSTANT_LEWIS}) {
            Kokkos::View<double **, Kokkos::LayoutRight> out("out", n, ns + 2);
            Kokkos::parallel_for("mixture_transport", n,
                                 MixtureFunctor{thermo, make_transport_table(mech, model, lewis), states, out});
            auto h_out = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), out);
            for (uint32_t i = 0; i < n; i++) {
                const auto & r = rows[i];
                const size_t ref = ns + 2;
                EXPECT_NEAR(h_out(i, 0) / r[ref] - 1.0, 0.0, 1e-6) << c.name << " mu, row " << i;
                EXPECT_NEAR(h_out(i, 1) / r[ref + 1] - 1.0, 0.0, 1e-6) << c.name << " lambda, row " << i;
                for (uint32_t k = 0; k < ns; k++) {
                    // The mixture-averaged D of a pure species vanishes; Cantera's
                    // formula leaves round-off noise there
                    if (model == TransportModel::MIXTURE_AVERAGED && r[2 + k] == 1.0) {
                        EXPECT_EQ(h_out(i, 2 + k), 0.0) << c.name << ", row " << i;
                        continue;
                    }
                    const double D_ref = model == TransportModel::MIXTURE_AVERAGED ? r[ref + 2 + k]
                                         : model == TransportModel::UNITY_LEWIS   ? r[ref + 2 + ns]
                                                                                  : r[ref + 2 + ns] / lewis[k];
                    EXPECT_NEAR(h_out(i, 2 + k) / D_ref - 1.0, 0.0, 1e-6)
                        << c.name << " D_" << mech.species[k].name << ", row " << i;
                }
            }
        }
    }
}

TEST(TransportTest, ThermalDiffusionMatchesCantera) {
    // Cantera's mixture-averaged thermal diffusion coefficients (C* fits in
    // ln T*, Wilke's mixing operator, the normalization to a zero sum) at
    // random states with absent species and one pure species, in device
    // kernels; the other properties are unchanged by computing them
    for (const Case & c : CASES) {
        const Mechanism mech = read_mechanism(c.file, c.phase);
        const uint32_t ns = static_cast<uint32_t>(mech.n_species());
        const auto rows = read_rows(c.name + "_thermal_diffusion.csv");
        ASSERT_FALSE(rows.empty()) << c.name;
        const uint32_t n = static_cast<uint32_t>(rows.size());
        Kokkos::View<double **, Kokkos::LayoutRight> states("states", n, ns + 2);
        auto h_states = Kokkos::create_mirror_view(states);
        for (uint32_t i = 0; i < n; i++) {
            for (uint32_t m = 0; m < ns + 2; m++) h_states(i, m) = rows[i][m];
        }
        Kokkos::deep_copy(states, h_states);
        const ThermoTable<> thermo = make_thermo_table(mech);
        Kokkos::View<double **, Kokkos::LayoutRight> out("out", n, 2 * ns + 2), plain("plain", n, ns + 2);
        Kokkos::parallel_for("thermal_diffusion", n,
                             MixtureFunctor{thermo, make_transport_table(mech, TransportModel::MIXTURE_AVERAGED, {}, true),
                                            states, out});
        Kokkos::parallel_for("mixture_transport", n,
                             MixtureFunctor{thermo, make_transport_table(mech, TransportModel::MIXTURE_AVERAGED), states,
                                            plain});
        auto h_out = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), out);
        auto h_plain = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), plain);
        double worst = 0.0;
        for (uint32_t i = 0; i < n; i++) {
            const auto & r = rows[i];
            for (uint32_t m = 0; m < ns + 2; m++) EXPECT_EQ(h_out(i, m), h_plain(i, m)) << c.name << ", row " << i;
            double scale = 0.0, sum = 0.0;
            for (uint32_t k = 0; k < ns; k++) scale = std::max(scale, std::abs(r[ns + 2 + k]));
            for (uint32_t k = 0; k < ns; k++) {
                const double DT = h_out(i, ns + 2 + k), ref = r[ns + 2 + k];
                sum += DT;
                if (scale == 0.0) {
                    EXPECT_EQ(DT, 0.0) << c.name << " DT_" << mech.species[k].name << ", row " << i;
                    continue;
                }
                worst = std::max(worst, std::abs(DT - ref) / scale);
                EXPECT_NEAR(DT, ref, 1e-6 * scale) << c.name << " DT_" << mech.species[k].name << ", row " << i;
            }
            EXPECT_NEAR(sum, 0.0, 1e-12 * scale) << c.name << ", row " << i;
        }
        std::ostringstream text;
        text << std::scientific << worst;
        RecordProperty(c.name + "_thermal_diffusion_max_error", text.str());
    }
    // Thermal diffusion is a mixture-averaged option
    const Mechanism mech = read_mechanism(CASES[0].file, CASES[0].phase);
    EXPECT_THROW(make_transport_table<Kokkos::HostSpace>(mech, TransportModel::UNITY_LEWIS, {}, true),
                 std::invalid_argument);
}

TEST(TransportTest, MissingTransportDataIsReported) {
    const Mechanism mech = read_mechanism(SOURCE_DIR + "/test/data/chemistry/test_mechanism.yaml", "mixed");
    try {
        fit_transport(mech);
        ADD_FAILURE() << "no error";
    } catch (const std::runtime_error & e) {
        EXPECT_NE(std::string(e.what()).find("transport data"), std::string::npos) << e.what();
    }
}

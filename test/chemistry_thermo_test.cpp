/**
 * @file chemistry_thermo_test.cpp
 * @author Matthew Bonanni (mbonanni001@gmail.com)
 * @brief Mechanism reader and species/mixture thermodynamics against Cantera.
 * @version 0.3
 * @date 2026-10-02
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 *
 */

#include <gtest/gtest.h>
#include <Kokkos_Core.hpp>

#include <cmath>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "mechanism.h"
#include "thermo.h"

using namespace chemistry;

namespace {

const std::string SOURCE_DIR = MALLARD_SOURCE_DIR;
const std::string DATA_DIR = SOURCE_DIR + "/test/data/chemistry/";

/** @brief Rows of a reference CSV file (comment lines skipped), split at commas, after its header. */
struct Csv {
    std::vector<std::string> columns;
    std::vector<std::vector<std::string>> rows;

    explicit Csv(const std::string & file) {
        std::ifstream in(DATA_DIR + file);
        if (!in.good()) throw std::runtime_error("missing reference file " + file);
        std::string line;
        while (std::getline(in, line)) {
            if (line.empty() || line[0] == '#') continue;
            std::vector<std::string> fields;
            std::stringstream s(line);
            std::string field;
            while (std::getline(s, field, ',')) fields.push_back(field);
            if (columns.empty()) {
                columns = fields;
            } else {
                rows.push_back(fields);
            }
        }
    }
};

std::vector<double> split_numbers(const std::string & text) {
    std::vector<double> values;
    std::stringstream s(text);
    std::string field;
    while (std::getline(s, field, ';')) values.push_back(std::stod(field));
    return values;
}

struct Case {
    std::string name, file, phase;
};

const std::vector<Case> CASES = {
    {"h2o2", SOURCE_DIR + "/mechanisms/h2o2.yaml", "ohmech"},
    {"gri30", SOURCE_DIR + "/mechanisms/gri30.yaml", ""},
    {"airNASA9", SOURCE_DIR + "/mechanisms/airNASA9.yaml", ""},
    {"test_air_cp", DATA_DIR + "test_mechanism.yaml", "air-cp"},
    {"test_mixed", DATA_DIR + "test_mechanism.yaml", "mixed"},
};

void expect_close(double value, double reference, double rtol, const std::string & what) {
    EXPECT_LE(std::abs(value - reference), rtol * std::max(std::abs(reference), 1.0))
        << what << ": " << value << " vs " << reference;
}

// Device kernels live outside the test bodies (nvcc rejects extended lambdas there)

/** @brief cp/R, h/RT, s/R of species k(i) at T(i), evaluated on the device. */
Kokkos::View<double *[3]>::host_mirror_type species_properties(const ThermoTable<> & table,
                                                               const std::vector<uint32_t> & k,
                                                               const std::vector<double> & T) {
    const size_t n = k.size();
    Kokkos::View<uint32_t *> d_k("k", n);
    Kokkos::View<double *> d_T("T", n);
    Kokkos::deep_copy(d_k, Kokkos::View<const uint32_t *, Kokkos::HostSpace>(k.data(), n));
    Kokkos::deep_copy(d_T, Kokkos::View<const double *, Kokkos::HostSpace>(T.data(), n));
    Kokkos::View<double *[3]> out("out", n);
    Kokkos::parallel_for("species_properties", n, KOKKOS_LAMBDA(const size_t i) {
        const ThermoTable<>::Powers p = ThermoTable<>::powers(d_T(i));
        out(i, 0) = table.cp_R(d_k(i), p);
        out(i, 1) = table.h_RT(d_k(i), p);
        out(i, 2) = table.s_R(d_k(i), p);
    });
    return Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), out);
}

/**
 * @brief Per state (T, Y): cp, cv, h, e, R, then T recovered from e from a
 *        nearby and from a distant first guess.
 */
Kokkos::View<double *[7]>::host_mirror_type mixture_properties(const ThermoTable<> & table,
                                                               const std::vector<double> & T,
                                                               const std::vector<double> & Y) {
    const size_t n = T.size();
    const uint32_t n_species = table.n_species;
    Kokkos::View<double *> d_T("T", n);
    Kokkos::View<double *> d_Y("Y", Y.size());
    Kokkos::deep_copy(d_T, Kokkos::View<const double *, Kokkos::HostSpace>(T.data(), n));
    Kokkos::deep_copy(d_Y, Kokkos::View<const double *, Kokkos::HostSpace>(Y.data(), Y.size()));
    Kokkos::View<double *[7]> out("out", n);
    Kokkos::parallel_for("mixture_properties", n, KOKKOS_LAMBDA(const size_t i) {
        const MassFractions y{d_Y.data() + i * n_species};
        double e, cv;
        table.e_cv(d_T(i), y, e, cv);
        out(i, 0) = table.cp_mass(d_T(i), y);
        out(i, 1) = cv;
        out(i, 2) = table.h_mass(d_T(i), y);
        out(i, 3) = e;
        out(i, 4) = table.gas_constant(y);
        out(i, 5) = table.T_from_e(e, y, 1.2 * d_T(i));
        out(i, 6) = table.T_from_e(e, y, table.T_low);
    });
    return Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), out);
}

} // namespace

TEST(ChemistryThermoTest, ReaderMatchesCanteraCoefficientsAndMolecularWeights) {
    for (const Case & c : CASES) {
        const Mechanism mech = read_mechanism(c.file, c.phase);
        const Csv ref(c.name + "_species.csv");
        ASSERT_EQ(mech.n_species(), ref.rows.size()) << c.name;
        for (size_t k = 0; k < ref.rows.size(); k++) {
            const auto & row = ref.rows[k];
            const Species & sp = mech.species[k];
            const std::string what = c.name + " " + row[0];
            ASSERT_EQ(sp.name, row[0]) << c.name;
            expect_close(sp.molecular_weight, std::stod(row[1]), 1e-14, what + " W");
            const std::vector<double> a = split_numbers(row[3]);
            const auto & th = sp.thermo;
            if (row[2] == "NasaPoly2") {
                // Cantera: [T_mid, high range (7), low range (7)]
                ASSERT_EQ(th.model, ThermoModel::NASA7) << what;
                ASSERT_EQ(th.coeffs.size(), 2u) << what;
                EXPECT_EQ(th.T_bounds[1], a[0]) << what;
                for (size_t j = 0; j < 7; j++) {
                    EXPECT_EQ(th.coeffs[1][2 + j], a[1 + j]) << what;
                    EXPECT_EQ(th.coeffs[0][2 + j], a[8 + j]) << what;
                }
            } else if (row[2] == "Nasa9PolyMultiTempRegion") {
                // Cantera: [n, then per region T_min, T_max and 9 coefficients]
                ASSERT_EQ(th.model, ThermoModel::NASA9) << what;
                ASSERT_EQ(th.coeffs.size(), static_cast<size_t>(a[0])) << what;
                for (size_t r = 0; r < th.coeffs.size(); r++) {
                    EXPECT_EQ(th.T_bounds[r], a[1 + 11 * r]) << what;
                    EXPECT_EQ(th.T_bounds[r + 1], a[2 + 11 * r]) << what;
                    for (size_t j = 0; j < 9; j++) EXPECT_EQ(th.coeffs[r][j], a[3 + 11 * r + j]) << what;
                }
            } else {
                // Cantera: [T0, h0, s0, cp0], SI with kmol
                ASSERT_EQ(row[2], "ConstantCp") << what;
                ASSERT_EQ(th.model, ThermoModel::CONSTANT_CP) << what;
                for (size_t j = 0; j < 4; j++) expect_close(th.constant_cp[j], a[j], 1e-14, what + " constant-cp");
            }
        }
    }
}

TEST(ChemistryThermoTest, SpeciesPropertiesMatchCanteraIncludingExtrapolation) {
    for (const Case & c : CASES) {
        const ThermoTable<> table = make_thermo_table(read_mechanism(c.file, c.phase));
        const Csv ref(c.name + "_thermo.csv");
        std::vector<uint32_t> k;
        std::vector<double> T;
        for (const auto & row : ref.rows) {
            k.push_back(static_cast<uint32_t>(std::stoul(row[0])));
            T.push_back(std::stod(row[1]));
        }
        const auto out = species_properties(table, k, T);
        for (size_t i = 0; i < ref.rows.size(); i++) {
            const std::string what = c.name + " species " + ref.rows[i][0] + " T " + ref.rows[i][1];
            expect_close(out(i, 0), std::stod(ref.rows[i][2]), 1e-12, what + " cp/R");
            expect_close(out(i, 1), std::stod(ref.rows[i][3]), 1e-12, what + " h/RT");
            expect_close(out(i, 2), std::stod(ref.rows[i][4]), 1e-12, what + " s/R");
        }
    }
}

TEST(ChemistryThermoTest, MixturePropertiesMatchCanteraAndTemperatureInvertsEnergy) {
    for (const Case & c : CASES) {
        const ThermoTable<> table = make_thermo_table(read_mechanism(c.file, c.phase));
        const Csv ref(c.name + "_mixture.csv");
        ASSERT_EQ(ref.columns.size(), 6 + table.n_species) << c.name;
        std::vector<double> T, Y;
        for (const auto & row : ref.rows) {
            T.push_back(std::stod(row[0]));
            for (uint32_t k = 0; k < table.n_species; k++) Y.push_back(std::stod(row[6 + k]));
        }
        const auto out = mixture_properties(table, T, Y);
        for (size_t i = 0; i < ref.rows.size(); i++) {
            const std::string what = c.name + " state " + std::to_string(i);
            // Energies are compared relative to their scale, h ~ cp T
            const double scale = std::stod(ref.rows[i][1]) * T[i];
            for (size_t j = 0; j < 5; j++) {
                const double r = std::stod(ref.rows[i][1 + j]);
                const double s = (j == 2 || j == 3) ? std::max(std::abs(r), scale) : std::abs(r);
                EXPECT_LE(std::abs(out(i, j) - r), 1e-12 * s) << what << " " << ref.columns[1 + j];
            }
            EXPECT_NEAR(out(i, 5), T[i], 1e-10) << what;
            EXPECT_NEAR(out(i, 6), T[i], 1e-10) << what;
        }
    }
}

TEST(ChemistryThermoTest, UnsupportedOrMalformedInputNamesTheEntry) {
    const std::string dir = (std::filesystem::temp_directory_path() / "mallard_mechanism_errors").string();
    std::filesystem::create_directories(dir);
    auto expect_error = [&](const std::string & yaml, const std::string & phase, const std::string & needle) {
        const std::string file = dir + "/mech.yaml";
        std::ofstream(file) << yaml;
        try {
            read_mechanism(file, phase);
            ADD_FAILURE() << "accepted: " << needle;
        } catch (const std::runtime_error & e) {
            EXPECT_NE(std::string(e.what()).find(needle), std::string::npos) << e.what();
        }
    };
    const std::string phases = "phases:\n- name: gas\n  thermo: ideal-gas\n  elements: [H]\n  species: [H2]\n";
    const std::string h2 = "species:\n- name: H2\n  composition: {H: 2}\n  thermo:\n";
    expect_error(phases + h2 + "    model: Shomate\n", "", "H2");
    expect_error(phases + h2 + "    model: constant-cp\n    cp0: 3 parsecs\n", "", "parsecs");
    expect_error(phases + h2 + "    model: constant-cp\n    cp0: 29 J/mol\n", "", "wrong dimension");
    expect_error(phases + h2 + "    model: constant-cp\n", "liquid", "liquid");
    expect_error("phases:\n- name: gas\n  thermo: ideal-gas\n  elements: [O]\n  species: [H2]\n" + h2 +
                     "    model: constant-cp\n", "", "element H");
    std::filesystem::remove_all(dir);
}

/**
 * @file radiation_test.cpp
 * @author Matthew Bonanni (mbonanni001@gmail.com)
 * @brief Optically thin radiation: the loss against Cantera's, and a
 *        homogeneous gas cooling as the energy equation's ODE.
 * @version 0.3
 * @date 2026-10-05
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 *
 */

#include <gtest/gtest.h>
#include <Kokkos_Core.hpp>

#include <array>
#include <cmath>
#include <iomanip>
#include <sstream>
#include <string>
#include <vector>

#include "input.h"
#include "mechanism.h"
#include "radiation.h"
#include "solver.h"
#include "test_fixtures.h"
#include "thermo.h"

namespace {

const std::string SOURCE_DIR = MALLARD_SOURCE_DIR;
const std::string GRI30 = SOURCE_DIR + "/mechanisms/gri30.yaml";
const std::string H2O2 = SOURCE_DIR + "/mechanisms/h2o2.yaml";

using chemistry::RadiatingSpecies;

chemistry::OpticallyThinRadiation model(const chemistry::Mechanism & mech, const std::vector<RadiatingSpecies> & species,
                                        double T_ambient) {
    const char * names[] = {"H2O", "CO2", "CO", "CH4"};
    chemistry::OpticallyThinRadiation r;
    for (RadiatingSpecies s : species) {
        const int i = static_cast<int>(s);
        r.index[i] = mech.species_index(names[i]);
        r.inv_W[i] = 1.0 / mech.species[r.index[i]].molecular_weight;
    }
    r.T_ambient4 = std::pow(T_ambient, 4);
    return r;
}

std::string box(const std::string & radiation, uint32_t n_steps, double dt) {
    const std::string periodic = N_DIM == 2 ? "[\"x\", \"y\"]" : "[\"x\", \"y\", \"z\"]";
    std::ostringstream s;
    s << std::setprecision(17) << "[run]\nn_steps = " << n_steps << "\ndt = " << dt << "\n"
      << "[mesh]\ntype = \"cartesian\"\nNx = 3\nNy = 3\nLx = 1.0\nLy = 1.0\n"
      << (N_DIM == 3 ? "Nz = 3\nLz = 1.0\n" : "") << "periodic = " << periodic << "\n"
      << "[initialize]\ntype = \"constant\"\np = 101325.0\nT = 2000.0\nu = "
      << (N_DIM == 2 ? "[0.0, 0.0]" : "[0.0, 0.0, 0.0]")
      << "\nX = { H2O = 0.2, CO2 = 0.1, CO = 0.05, CH4 = 0.02, N2 = 0.63 }\n"
      << "[numerics]\nriemann_solver = \"HLLC\"\n[numerics.face_reconstruction]\ntype = \"MUSCL\"\n"
      << "[output]\ncheck_interval = 1000000\n"
      << "[physics]\ntype = \"euler\"\ngas = \"mixture\"\nmechanism = \"" << GRI30 << "\"\n" << radiation;
    return s.str();
}

// Partial densities for OpticallyThinRadiation::loss, a host-device function, from which nvcc
// rejects calling a host lambda
struct SpeciesDensities {
    const double * rho_k;
    KOKKOS_INLINE_FUNCTION double operator()(const uint32_t k) const { return rho_k[k]; }
};

} // namespace

TEST(RadiationTest, LossMatchesCanteraFlameRadiation) {
    // Cantera 3.2's FreeFlame with radiation_enabled (stoichiometric CH4/air,
    // GRI-3.0, 1 atm): q = 4 sigma p (X_H2O a_H2O + X_CO2 a_CO2) T^4, the
    // TNF fits of H2O and CO2 and no background term. Rows: T, p, X_H2O,
    // X_CO2, Cantera's radiative_heat_loss
    const double rows[][5] = {
        {491.51155712139678, 101325.0, 0.021590548570867969, 0.0012949126596820675, 6263.5150010077887},
        {888.39462265354041, 101325.0, 0.062180945267889677, 0.0067482916407113382, 91706.693728477534},
        {1405.1739006593607, 101325.0, 0.1138729932687275, 0.018831394923950143, 520046.17980699654},
        {1902.7443575912089, 101325.0, 0.16460203621234301, 0.058961238324633597, 1802587.1355145751},
        {2175.7565592053088, 101325.0, 0.18389608707073246, 0.084368239631330805, 2759499.9410175886}};
    const auto mech = chemistry::read_mechanism(GRI30);
    const auto r = model(mech, {RadiatingSpecies::H2O, RadiatingSpecies::CO2}, 0.0);
    std::vector<double> rho_k(mech.species.size(), 0.0);
    for (const auto & row : rows) {
        const double T = row[0], p = row[1];
        for (const auto & [name, X] : {std::pair{"H2O", row[2]}, {"CO2", row[3]}}) {
            const int32_t k = mech.species_index(name);
            rho_k[k] = X * p * mech.species[k].molecular_weight / (chemistry::GAS_CONSTANT * T);
        }
        EXPECT_NEAR(r.loss(T, SpeciesDensities{rho_k.data()}), row[4], 1e-9 * row[4]) << "T = " << T;
    }
}

TEST(RadiationTest, CarbonMonoxideFitsMeetAt750K) {
    // The TNF fits of CO below and above 750 K meet within about 1%; a wrong
    // coefficient in either breaks this
    const double below = chemistry::planck_mean_absorption(RadiatingSpecies::CO, 750.0);
    const double above = chemistry::planck_mean_absorption(RadiatingSpecies::CO, std::nextafter(750.0, 1e3));
    EXPECT_NEAR(below, 3.604, 1e-3);
    EXPECT_NEAR(above / below, 1.0, 0.02);
}

TEST(RadiationTest, HomogeneousGasCoolsLikeTheEnergyODE) {
    // A quiescent periodic box of H2O, CO2, CO, CH4 and N2 at 2000 K without
    // chemistry: rho and the composition stay fixed and de/dt = -q(T) / rho,
    // with the background at 1000 K. Against RK4 on that ODE with 100x
    // smaller steps
    const double dt = 2e-4, t_end = 0.2;
    const uint32_t n_steps = static_cast<uint32_t>(std::lround(t_end / dt));
    Solver solver;
    solver.init(parse_toml(box("[radiation]\nT_ambient = 1000.0\n", n_steps, dt)));
    solver.copy_device_to_host();
    const auto mech = chemistry::read_mechanism(GRI30);
    const auto thermo = chemistry::make_thermo_table<Kokkos::HostSpace>(mech);
    const uint32_t ns = thermo.n_species;
    const double rho = double(solver.h_conservatives(0, 0));
    std::vector<double> Y(ns), rho_k(ns);
    for (uint32_t k = 0; k < ns; k++) {
        rho_k[k] = double(solver.h_species(0, k));
        Y[k] = rho_k[k] / rho;
    }
    const chemistry::MassFractions y{Y.data()};
    const auto r = model(mech, {RadiatingSpecies::H2O, RadiatingSpecies::CO2, RadiatingSpecies::CO,
                                RadiatingSpecies::CH4}, 1000.0);
    double T = 2000.0;
    const auto dedt = [&](double e) {
        T = thermo.T_from_e(e, y, T);
        return -r.loss(T, SpeciesDensities{rho_k.data()}) / rho;
    };
    double e = double(solver.h_conservatives(0, N_DIM + 1)) / rho;
    const double h = dt / 100;
    for (uint32_t n = 0; n < 100 * n_steps; n++) {
        const double k1 = dedt(e), k2 = dedt(e + 0.5 * h * k1), k3 = dedt(e + 0.5 * h * k2), k4 = dedt(e + h * k3);
        e += h / 6 * (k1 + 2 * k2 + 2 * k3 + k4);
    }
    const double T_ref = thermo.T_from_e(e, y, T);
    ASSERT_LT(T_ref, 1900.0);  // the case cools by over 100 K

    solver.run();
    solver.copy_device_to_host();
    const bool single = sizeof(rtype) == sizeof(float);
    for (uint32_t c = 0; c < solver.get_mesh()->n_cells; c++) {
        const double e_c = double(solver.h_conservatives(c, N_DIM + 1)) / double(solver.h_conservatives(c, 0));
        EXPECT_NEAR(thermo.T_from_e(e_c, y, T_ref), T_ref, (single ? 1e-4 : 1e-6) * T_ref) << "cell " << c;
        FOR_I_DIM EXPECT_NEAR(double(solver.h_conservatives(c, 1 + i)), 0.0, single ? 1e-4 : 1e-9);
    }
}

TEST(RadiationTest, InvalidInputsAreRejected) {
    const auto rejects = [](const std::string & input, const std::string & what) {
        Solver solver;
        EXPECT_THROW(solver.init(parse_toml(input)), InputError) << what;
    };
    rejects(box("[radiation]\nspecies = [\"OH\"]\n", 1, 1e-6), "not a radiating species");
    rejects(box("[radiation]\nmodel = \"p1\"\n", 1, 1e-6), "unknown model");
    rejects(box("[radiation]\nT_ambient = -1.0\n", 1, 1e-6), "negative T_ambient");
    std::string h2 = box("[radiation]\nspecies = [\"H2O\", \"CO2\"]\n", 1, 1e-6);
    h2.replace(h2.find(GRI30), GRI30.size(), H2O2);
    h2.replace(h2.find("X = {"), h2.find('\n', h2.find("X = {")) - h2.find("X = {"), "X = { H2O = 0.2, N2 = 0.8 }");
    rejects(h2, "CO2 is not in h2o2");
}

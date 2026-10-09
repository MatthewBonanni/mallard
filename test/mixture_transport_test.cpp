/**
 * @file mixture_transport_test.cpp
 * @author Matthew Bonanni (mbonanni001@gmail.com)
 * @brief Viscous, heat-conducting and diffusing gas mixtures against exact
 *        solutions.
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
#include <iomanip>
#include <numbers>
#include <sstream>
#include <string>
#include <vector>

#include "solver.h"
#include "test_fixtures.h"
#include "thermo.h"
#include "transport.h"

namespace {

const std::string SOURCE_DIR = MALLARD_SOURCE_DIR;
const std::string H2O2 = SOURCE_DIR + "/mechanisms/h2o2.yaml";
const std::string TWO_NITROGENS = SOURCE_DIR + "/test/data/chemistry/test_transport.yaml";
constexpr double PI = std::numbers::pi;

/** @brief A tolerance for double builds, or its counterpart for float builds. */
constexpr double tol(double in_double, double in_float) {
    return sizeof(rtype) == sizeof(double) ? in_double : in_float;
}

/**
 * @brief A strip of nx cells over [0, Lx], one cell thick with symmetry
 *        planes around it, or, periodic, three cells thick and periodic in x and
 *        y (symmetry in z), so that the cross-stream velocity is free.
 */
std::string strip(uint32_t nx, double Lx, bool periodic) {
    std::ostringstream s;
    const double h = Lx / nx;
    const int ny = periodic ? 3 : 1;
    s << std::setprecision(17) << "[mesh]\ntype = \"cartesian\"\nNx = " << nx << "\nNy = " << ny << "\nLx = " << Lx
      << "\nLy = " << ny * h << "\n";
    if (N_DIM == 3) s << "Nz = 1\nLz = " << h << "\n";
    if (periodic) s << "periodic = [\"x\", \"y\"]\n";
    const char * zones[] = {"left", "right", "bottom", "top", "back", "front"};
    for (int k = periodic ? 4 : 0; k < 2 * N_DIM; k++) {
        s << "[[boundaries]]\nname = \"" << zones[k] << "\"\ntype = \"symmetry\"\n";
    }
    return s.str();
}

std::string velocity(const std::string & uy) {
    return N_DIM == 2 ? "[\"0.0\", \"" + uy + "\"]" : "[\"0.0\", \"" + uy + "\", \"0.0\"]";
}

std::string viscous_mixture(const std::string & mechanism, const std::string & transport = "mixture_averaged",
                            const bool soret = false) {
    return "[physics]\ntype = \"navier_stokes\"\ngas = \"mixture\"\nmechanism = \"" + mechanism +
           "\"\ntransport = \"" + transport + "\"\n" + (soret ? "soret = true\n" : "") +
           "[numerics]\nriemann_solver = \"HLLC\"\n[numerics.face_reconstruction]\ntype = \"MUSCL\"\n"
           "[output]\ncheck_interval = 1000000\n";
}

/** @brief Cell centroids along x and a field per cell, after a run. */
struct Profile {
    std::vector<double> x, value;
};

/** @brief Amplitude of the sin(2 pi x / L) mode of a periodic profile. */
double sine_amplitude(const Profile & p, double L) {
    double sum = 0.0;
    for (size_t c = 0; c < p.x.size(); c++) sum += p.value[c] * std::sin(2.0 * PI * p.x[c] / L);
    return 2.0 * sum / static_cast<double>(p.x.size());
}

/** @brief Run to completion and refresh the host copies. */
void run(Solver & solver, const std::string & input) {
    solver.init(parse_toml(input));
    solver.run();
    solver.update_primitives();
    solver.copy_device_to_host();
}

/** @brief Owned cells' x and a column of the primitives, or a species' mass fraction. */
Profile column(const Solver & solver, int primitive, int species = -1) {
    Profile p;
    const auto mesh = solver.get_mesh();
    auto coords = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), mesh->cell_coords);
    for (uint32_t c = 0; c < mesh->n_owned(); c++) {
        p.x.push_back(static_cast<double>(coords(c, 0)));
        p.value.push_back(species >= 0 ? static_cast<double>(solver.h_species(c, species) / solver.h_conservatives(c, 0))
                                       : static_cast<double>(solver.h_primitives(c, primitive)));
    }
    return p;
}

} // namespace

TEST(MixtureTransportTest, BinaryInterdiffusionMatchesTheErfSolution) {
    // Two gases with equal molar masses and thermodynamics but different
    // collision parameters, side by side at uniform T and p: the
    // mixture-averaged fluxes with the correction velocity reduce to Fick's
    // law with their binary coefficient, so Y_1 = erfc((x - x0) / (2 sqrt(D t))) / 2
    const double p = 1000.0, T = 300.0, L = 1e-3, x0 = 0.5e-3;
    const auto mech = chemistry::read_mechanism(TWO_NITROGENS);
    const auto table = chemistry::make_transport_table<Kokkos::HostSpace>(mech, chemistry::TransportModel::MIXTURE_AVERAGED);
    const double D = table.binary_diffusion(0, 1, T, p);
    const double t = std::pow(0.1 * L, 2) / D;  // diffusion length 2 sqrt(D t) = L / 5
    std::ostringstream input;
    input << std::setprecision(17) << "[run]\nt_stop = " << t << "\ncfl = 0.25\n" << strip(100, L, false)
          << "[initialize]\ntype = \"analytical\"\np = \"" << p << "\"\nT = \"" << T << "\"\nu = " << velocity("0.0")
          << "\nX = { N2 = \"x < " << x0 << " ? 1 : 0\", N2B = \"x < " << x0 << " ? 0 : 1\" }\n"
          << viscous_mixture(TWO_NITROGENS);
    Solver solver;
    run(solver, input.str());
    const Profile Y = column(solver, 0, 0);
    double err = 0.0;
    for (size_t c = 0; c < Y.x.size(); c++) {
        const double exact = 0.5 * std::erfc((Y.x[c] - x0) / (2.0 * std::sqrt(D * t)));
        err = std::max(err, std::abs(Y.value[c] - exact));
    }
    EXPECT_LT(err, 5e-4);
    // No flow: the species fluxes sum to zero, so rho and p stay uniform
    for (uint32_t c = 0; c < solver.get_mesh()->n_owned(); c++) {
        EXPECT_NEAR(static_cast<double>(solver.h_primitives(c, N_DIM)) / p, 1.0, tol(1e-9, 1e-3));
    }
}

TEST(MixtureTransportTest, EnthalpyDiffusionKeepsTheMixingOfEqualTemperatureGasesIsothermal) {
    // Hydrogen and nitrogen at 1000 K and 1 atm interdiffuse across a smooth
    // layer; hydrogen carries ten times nitrogen's enthalpy per unit mass, so
    // without the enthalpy flux sum_k h_k j_k the temperature in the layer
    // would change by about a hundred kelvin. With it, only the pressure work
    // of the small flow that diffusion induces changes it
    const double T = 1000.0, L = 2e-3, x0 = 1e-3, w = 1e-4;
    std::ostringstream input;
    const std::string f = "0.5 * (1 - tanh((x - " + std::to_string(x0) + ") / " + std::to_string(w) + "))";
    input << std::setprecision(17) << "[run]\nt_stop = 4e-6\ncfl = 0.25\n" << strip(100, L, false)
          << "[initialize]\ntype = \"analytical\"\np = \"101325.0\"\nT = \"" << T << "\"\nu = " << velocity("0.0")
          << "\nX = { H2 = \"" << f << "\", N2 = \"1 - " << f << "\" }\n" << viscous_mixture(H2O2);
    Solver solver;
    run(solver, input.str());
    const Profile temperature = column(solver, N_DIM + 1);
    const Profile Y_H2 = column(solver, 0, 0);
    double T_err = 0.0;
    for (double v : temperature.value) T_err = std::max(T_err, std::abs(v - T));
    EXPECT_LT(T_err, 5.0);
    // and the layer did spread: the H2 mass fraction at x0 + 2 w more than doubled
    EXPECT_GT(Y_H2.value[60], 0.002);
}

TEST(MixtureTransportTest, ShearAndThermalWavesDecayAtTheirDiffusionRates) {
    // Periodic shear wave v = V sin(k x): amplitude V exp(-mu k^2 t / rho);
    // temperature wave at uniform pressure: exp(-lambda k^2 t / (rho cp))
    const double p = 1000.0, T = 300.0, L = 1e-3, k = 2.0 * PI / L;
    const auto mech = chemistry::read_mechanism(TWO_NITROGENS);
    const auto thermo = chemistry::make_thermo_table<Kokkos::HostSpace>(mech);
    const auto table = chemistry::make_transport_table<Kokkos::HostSpace>(mech, chemistry::TransportModel::MIXTURE_AVERAGED);
    const double Y[2] = {1.0, 0.0};
    const chemistry::MassFractions y{Y};
    const double rho = p / (thermo.gas_constant(y) * T), cp = thermo.cp_mass(T, y);
    double mu, lambda, D[2];
    table.properties(T, p, rho, cp, y, mu, lambda, D);
    const double rate[2] = {mu * k * k / rho, lambda * k * k / (rho * cp)};
    for (int wave = 0; wave < 2; wave++) {
        const double t = 0.5 / rate[wave];
        std::ostringstream input;
        input << std::setprecision(17) << "[run]\nt_stop = " << t << "\ncfl = 0.25\n" << strip(32, L, true)
              << "[initialize]\ntype = \"analytical\"\np = \"" << p << "\"\nX = { N2 = 1.0 }\n";
        if (wave == 0) {
            input << "T = \"" << T << "\"\nu = " << velocity("1.0 * sin(2 * pi * x / " + std::to_string(L) + ")") << "\n";
        } else {
            input << "T = \"" << T << " + 3.0 * sin(2 * pi * x / " << L << ")\"\nu = " << velocity("0.0") << "\n";
        }
        input << viscous_mixture(TWO_NITROGENS);
        Solver solver;
        run(solver, input.str());
        const Profile profile = column(solver, wave == 0 ? 1 : N_DIM + 1);
        Profile fluctuation = profile;
        if (wave == 1) for (double & v : fluctuation.value) v -= T;
        const double expected = (wave == 0 ? 1.0 : 3.0) * std::exp(-rate[wave] * t);
        // The thermal mode couples weakly to acoustics, O((lambda k / (rho cp c))^2)
        EXPECT_NEAR(sine_amplitude(fluctuation, L) / expected, 1.0, wave == 0 ? 1e-3 : 5e-3)
            << (wave == 0 ? "shear" : "thermal");
    }
}

TEST(MixtureTransportTest, SoretEffectSeparatesSpeciesInATemperatureWave) {
    // H2/N2 at uniform composition and pressure with a temperature wave
    // T0 + dT sin(k x): only thermal diffusion separates the species, driving
    // the mode of Y_H2 at -D^T k^2 dT(t) / (rho T0) against Fickian relaxation
    // at D_12 k^2 (a binary mixture), so that its amplitude is
    //   -D^T k^2 dT_0 / (rho T0) (exp(-a k^2 t) - exp(-D_12 k^2 t)) / ((D_12 - a) k^2),
    // positive: hydrogen gathers where the gas is hot. The wave decays at
    // a = lambda / (rho cp): the enthalpy the Soret fluxes carry,
    // sum_k h_k j_k, cancels the change of the mixture's enthalpy they cause
    // to first order (without it, the wave ends 2.7% off)
    const double p = 101325.0, T0 = 600.0, dT = 30.0, L = 1e-3, k = 2.0 * PI / L;
    const auto mech = chemistry::read_mechanism(H2O2);
    const auto thermo = chemistry::make_thermo_table<Kokkos::HostSpace>(mech);
    const auto table = chemistry::make_transport_table<Kokkos::HostSpace>(
        mech, chemistry::TransportModel::MIXTURE_AVERAGED, {}, true);
    const uint32_t ns = static_cast<uint32_t>(mech.n_species());
    const int32_t i_H2 = mech.species_index("H2"), i_N2 = mech.species_index("N2");
    std::vector<double> Y(ns, 0.0), D(ns), DT(ns);
    const double W_H2 = mech.species[i_H2].molecular_weight, W_N2 = mech.species[i_N2].molecular_weight;
    Y[i_H2] = W_H2 / (W_H2 + W_N2);
    Y[i_N2] = 1.0 - Y[i_H2];
    const chemistry::MassFractions y{Y.data()};
    const double rho = p / (thermo.gas_constant(y) * T0), cp = thermo.cp_mass(T0, y);
    double mu, lambda;
    table.properties(T0, p, rho, cp, y, mu, lambda, D.data(), DT.data());
    ASSERT_LT(DT[i_H2], 0.0);
    const double D12 = table.binary_diffusion(i_H2, i_N2, T0, p), a = lambda / (rho * cp);
    const double t = 0.3 / (D12 * k * k);
    const double amplitude = -DT[i_H2] * k * k * dT / (rho * T0) *
                             (std::exp(-a * k * k * t) - std::exp(-D12 * k * k * t)) / ((D12 - a) * k * k);
    for (const bool soret : {true, false}) {
        std::ostringstream input;
        input << std::setprecision(17) << "[run]\nt_stop = " << t << "\ncfl = 0.25\n" << strip(32, L, true)
              << "[initialize]\ntype = \"analytical\"\np = \"" << p << "\"\nX = { H2 = 1.0, N2 = 1.0 }\n"
              << "T = \"" << T0 << " + " << dT << " * sin(2 * pi * x / " << L << ")\"\nu = " << velocity("0.0") << "\n"
              << viscous_mixture(H2O2, "mixture_averaged", soret);
        Solver solver;
        run(solver, input.str());
        Profile Y_H2 = column(solver, 0, i_H2);
        for (double & v : Y_H2.value) v -= Y[i_H2];
        Profile temperature = column(solver, N_DIM + 1);
        for (double & v : temperature.value) v -= T0;
        const double A_Y = sine_amplitude(Y_H2, L), A_T = sine_amplitude(temperature, L);
        if (soret) {
            RecordProperty("Y_amplitude_ratio", std::to_string(A_Y / amplitude));
            RecordProperty("T_amplitude_ratio", std::to_string(A_T / (dT * std::exp(-a * k * k * t))));
            EXPECT_NEAR(A_Y / amplitude, 1.0, tol(0.01, 0.02));
            EXPECT_NEAR(A_T / (dT * std::exp(-a * k * k * t)), 1.0, 0.01);
        } else {
            EXPECT_LT(std::abs(A_Y), tol(1e-3, 0.02) * amplitude);
        }
    }
}

TEST(MixtureTransportTest, InvalidTransportInputsAreRejected) {
    const std::string base = "[run]\nn_steps = 1\ncfl = 0.25\n" + strip(4, 1.0, false) +
                             "[initialize]\ntype = \"constant\"\np = 1.0e5\nT = 300.0\nu = " +
                             std::string(N_DIM == 2 ? "[0.0, 0.0]" : "[0.0, 0.0, 0.0]") + "\nX = { N2 = 1.0 }\n" +
                             "[numerics]\n[numerics.face_reconstruction]\ntype = \"FO\"\n";
    auto expect_error = [&](const std::string & physics, const std::string & needle) {
        Solver solver;
        try {
            solver.init(parse_toml(base + "[physics]\ngas = \"mixture\"\nmechanism = \"" + H2O2 + "\"\n" + physics));
            ADD_FAILURE() << "accepted: " << needle;
        } catch (const std::exception & e) {
            EXPECT_NE(std::string(e.what()).find(needle), std::string::npos) << e.what();
        }
    };
    expect_error("type = \"euler\"\ntransport = \"unity_lewis\"\n", "navier_stokes");
    expect_error("type = \"navier_stokes\"\ntransport = \"multicomponent\"\n", "physics.transport");
    expect_error("type = \"navier_stokes\"\nlewis = { H2 = 0.3 }\n", "constant_lewis");
    expect_error("type = \"navier_stokes\"\ntransport = \"constant_lewis\"\nlewis = { CH4 = 1.0 }\n", "CH4");
    expect_error("type = \"navier_stokes\"\ntransport = \"unity_lewis\"\nsoret = true\n", "physics.soret");
    expect_error("type = \"euler\"\nsoret = true\n", "navier_stokes");
}

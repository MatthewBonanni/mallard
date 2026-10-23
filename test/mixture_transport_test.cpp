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
#include <array>
#include <cmath>
#include <iomanip>
#include <numbers>
#include <sstream>
#include <string>
#include <vector>

#include "data_paths.h"
#include "solver.h"
#include "test_fixtures.h"
#include "thermo.h"
#include "transport.h"

namespace {

const std::string SOURCE_DIR = source_dir();
const std::string H2O2 = SOURCE_DIR + "/mechanisms/h2o2.yaml";
const std::string TWO_NITROGENS = SOURCE_DIR + "/test/data/chemistry/test_transport.yaml";
const std::string WALL_GASES = SOURCE_DIR + "/test/data/chemistry/test_wall.yaml";
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

/** @brief Owned cells' coordinate along axis and a column of the primitives, or a species' mass fraction. */
Profile column(const Solver & solver, int primitive, int species = -1, int axis = 0) {
    Profile p;
    const auto mesh = solver.get_mesh();
    auto coords = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), mesh->cell_coords);
    for (uint32_t c = 0; c < mesh->n_owned(); c++) {
        p.x.push_back(static_cast<double>(coords(c, axis)));
        p.value.push_back(species >= 0 ? static_cast<double>(solver.h_species(c, species) / solver.h_conservatives(c, 0))
                                       : static_cast<double>(solver.h_primitives(c, primitive)));
    }
    return p;
}

/**
 * @brief A channel of ny cells across [0, Ly] between the walls bottom and
 *        top (TOML bodies of their [[boundaries]] entries), three cells wide
 *        and periodic in x (one cell between symmetry planes in z).
 */
std::string channel(uint32_t ny, double Ly, const std::string & bottom, const std::string & top) {
    std::ostringstream s;
    const double h = Ly / ny;
    s << std::setprecision(17) << "[mesh]\ntype = \"cartesian\"\nNx = 3\nNy = " << ny << "\nLx = " << 3 * h
      << "\nLy = " << Ly << "\nperiodic = [\"x\"]\n";
    if (N_DIM == 3) s << "Nz = 1\nLz = " << h << "\n";
    s << "[[boundaries]]\nname = \"bottom\"\n" << bottom << "[[boundaries]]\nname = \"top\"\n" << top;
    if (N_DIM == 3) {
        s << "[[boundaries]]\nname = \"back\"\ntype = \"symmetry\"\n[[boundaries]]\nname = \"front\"\ntype = \"symmetry\"\n";
    }
    return s.str();
}

/** @brief Viscosity and conductivity of a mixture of fixed composition, from the host tables. */
struct FixedComposition {
    chemistry::Mechanism mech;
    chemistry::ThermoTable<Kokkos::HostSpace> thermo;
    chemistry::TransportTable<Kokkos::HostSpace> table;
    std::vector<double> Y, D;

    FixedComposition(const std::string & mechanism, const std::vector<std::pair<std::string, double>> & X)
        : mech(chemistry::read_mechanism(mechanism)), thermo(chemistry::make_thermo_table<Kokkos::HostSpace>(mech)),
          table(chemistry::make_transport_table<Kokkos::HostSpace>(mech, chemistry::TransportModel::MIXTURE_AVERAGED)),
          Y(mech.n_species(), 0.0), D(mech.n_species()) {
        double sum = 0.0;
        for (const auto & [name, x] : X) {
            const auto k = static_cast<size_t>(mech.species_index(name));
            Y[k] = x * mech.species[k].molecular_weight;
            sum += Y[k];
        }
        for (double & y : Y) y /= sum;
    }

    /** @brief [mu, lambda] at T (neither depends on p). */
    std::array<double, 2> properties(double T) {
        const chemistry::MassFractions y{Y.data()};
        const double p = 1e5, rho = p / (thermo.gas_constant(y) * T);
        double mu, lambda;
        table.properties(T, p, rho, thermo.cp_mass(T, y), y, mu, lambda, D.data());
        return {mu, lambda};
    }

    /** @brief int_T0^T lambda dT' (Simpson). */
    double conduction_integral(double T0, double T) {
        const int n = 400;
        const double h = (T - T0) / n;
        double sum = properties(T0)[1] + properties(T)[1];
        for (int i = 1; i < n; i++) sum += (i % 2 ? 4.0 : 2.0) * properties(T0 + i * h)[1];
        return sum * h / 3.0;
    }

    /** @brief The T in [T_lo, T_hi] with conduction_integral(T_ref, T) = target (bisection). */
    double invert(double T_ref, double target, double T_lo, double T_hi) {
        for (int it = 0; it < 60; it++) {
            const double T_mid = 0.5 * (T_lo + T_hi);
            (conduction_integral(T_ref, T_mid) < target ? T_lo : T_hi) = T_mid;
        }
        return 0.5 * (T_lo + T_hi);
    }
};

/** @brief An initial T(y) near steady conduction with lambda ~ T^0.75, T0 at y = 0 and T1 at y = L. */
std::string conduction_guess(double T0, double T1, double L) {
    std::ostringstream s;
    s << std::setprecision(17) << "(" << std::pow(T0, 1.75) << " + " << std::pow(T1, 1.75) - std::pow(T0, 1.75)
      << " * y / " << L << ")^(1 / 1.75)";
    return s.str();
}

const std::string AIR = "{ O2 = 1.0, N2 = 3.76 }";

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
    const uint32_t ns = mech.n_species();
    const uint32_t i_H2 = static_cast<uint32_t>(mech.species_index("H2")), i_N2 = static_cast<uint32_t>(mech.species_index("N2"));
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
        Profile Y_H2 = column(solver, 0, static_cast<int>(i_H2));
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

TEST(MixtureTransportTest, ConductionBetweenIsothermalWallsMatchesTheExactProfile) {
    // Air at rest between walls at 300 K and 900 K: the steady temperature
    // has int_T0^T lambda dT' linear in y, whatever the pressure. Started
    // near it; with double flux the same steady state (which needs the
    // diffusive energy flux in both sides' fluxes). The walls' conductivity
    // matters: the boundary cells' own instead of the mean with the wall
    // temperature's gives 1 K errors, 0.005 K with it
    const double T0 = 300.0, T1 = 900.0, L = 1e-3, p = 1000.0;
    FixedComposition air(WALL_GASES, {{"O2", 1.0}, {"N2", 3.76}});
    const double total = air.conduction_integral(T0, T1);
    for (const bool double_flux : {false, true}) {
        std::ostringstream input;
        input << std::setprecision(17) << "[run]\nt_stop = 1e-4\ncfl = 0.4\n"
              << channel(16, L, "type = \"wall_isothermal\"\nT = 300.0\n", "type = \"wall_isothermal\"\nT = 900.0\n")
              << "[initialize]\ntype = \"analytical\"\np = \"" << p << "\"\nT = \"" << conduction_guess(T0, T1, L)
              << "\"\nu = " << velocity("0.0") << "\nX = " << AIR << "\n" << viscous_mixture(WALL_GASES);
        std::string text = input.str();
        if (double_flux) text.insert(text.find("[numerics]\n") + 11, "double_flux = true\n");
        Solver solver;
        run(solver, text);
        const Profile T = column(solver, N_DIM + 1, -1, 1);
        double err = 0.0;
        for (size_t c = 0; c < T.x.size(); c++) {
            const double exact = air.invert(T0, total * T.x[c] / L, T0, T1);
            err = std::max(err, std::abs(T.value[c] - exact));
        }
        RecordProperty(double_flux ? "max_error_K_double_flux" : "max_error_K", std::to_string(err));
        EXPECT_LT(err, tol(0.05, 0.5)) << (double_flux ? "double flux" : "");
    }
}

TEST(MixtureTransportTest, HeatFluxWallSetsTheConductionProfile) {
    // Heat q enters through the bottom wall and leaves through the isothermal
    // top one: int_T(y)^T1 lambda dT' = q (L - y)
    const double T1 = 300.0, L = 1e-3, p = 1000.0, q = 2e4;
    FixedComposition air(WALL_GASES, {{"O2", 1.0}, {"N2", 3.76}});
    const double T_bottom = air.invert(T1, q * L, T1, 2000.0);
    std::ostringstream input;
    input << std::setprecision(17) << "[run]\nt_stop = 2e-4\ncfl = 0.4\n"
          << channel(16, L, "type = \"wall_heat_flux\"\nq = 2e4\n", "type = \"wall_isothermal\"\nT = 300.0\n")
          << "[initialize]\ntype = \"analytical\"\np = \"" << p << "\"\nT = \"" << conduction_guess(T_bottom, T1, L)
          << "\"\nu = " << velocity("0.0") << "\nX = " << AIR << "\n" << viscous_mixture(WALL_GASES);
    Solver solver;
    run(solver, input.str());
    const Profile T = column(solver, N_DIM + 1, -1, 1);
    double err = 0.0;
    for (size_t c = 0; c < T.x.size(); c++) {
        err = std::max(err, std::abs(T.value[c] - air.invert(T1, q * (L - T.x[c]), T1, T_bottom + 1.0)));
    }
    RecordProperty("max_error_K", std::to_string(err));
    EXPECT_LT(err, tol(0.05, 0.5));
}

TEST(MixtureTransportTest, CouetteFlowWithHeatTransferMatchesTheExactSolution) {
    // Air between a wall at rest at 300 K and one moving at U = 300 m/s at
    // 500 K: steady, with uniform pressure, shear stress tau = mu du/dy and
    // energy flux lambda dT/dy + u tau = C. Integrating
    // (u, T)' = (tau / mu, (C - u tau) / lambda) from the bottom wall and
    // shooting on (tau, C) gives the exact profiles; viscous heating bends T
    // above the conduction profile
    const double T0 = 300.0, T1 = 500.0, U = 300.0, L = 1e-3, p = 1000.0;
    FixedComposition air(WALL_GASES, {{"O2", 1.0}, {"N2", 3.76}});
    const int n = 2000;
    auto integrate = [&](double tau, double C, std::vector<std::array<double, 2>> * path) {
        std::array<double, 2> s = {0.0, T0};
        const double h = L / n;
        auto rhs = [&](const std::array<double, 2> & v) {
            const auto [mu, lambda] = air.properties(v[1]);
            return std::array<double, 2>{tau / mu, (C - v[0] * tau) / lambda};
        };
        if (path) path->push_back(s);
        for (int i = 0; i < n; i++) {
            const auto k1 = rhs(s);
            const auto k2 = rhs({s[0] + 0.5 * h * k1[0], s[1] + 0.5 * h * k1[1]});
            const auto k3 = rhs({s[0] + 0.5 * h * k2[0], s[1] + 0.5 * h * k2[1]});
            const auto k4 = rhs({s[0] + h * k3[0], s[1] + h * k3[1]});
            for (size_t j = 0; j < 2; j++) s[j] += h / 6.0 * (k1[j] + 2.0 * k2[j] + 2.0 * k3[j] + k4[j]);
            if (path) path->push_back(s);
        }
        return s;
    };
    double x[2] = {air.properties(400.0)[0] * U / L, air.properties(400.0)[1] * (T1 - T0) / L};
    for (int it = 0; it < 20; it++) {
        const auto f = integrate(x[0], x[1], nullptr);
        const double r[2] = {f[0] - U, f[1] - T1};
        double J[2][2];
        for (int j = 0; j < 2; j++) {
            double xp[2] = {x[0], x[1]};
            xp[j] *= 1.0 + 1e-7;
            const auto fp = integrate(xp[0], xp[1], nullptr);
            J[0][j] = (fp[0] - f[0]) / (xp[j] - x[j]);
            J[1][j] = (fp[1] - f[1]) / (xp[j] - x[j]);
        }
        const double det = J[0][0] * J[1][1] - J[0][1] * J[1][0];
        x[0] -= (J[1][1] * r[0] - J[0][1] * r[1]) / det;
        x[1] -= (-J[1][0] * r[0] + J[0][0] * r[1]) / det;
    }
    std::vector<std::array<double, 2>> exact;
    const auto end = integrate(x[0], x[1], &exact);
    ASSERT_NEAR(end[0], U, 1e-8 * U);
    ASSERT_NEAR(end[1], T1, 1e-8 * T1);
    auto at = [&](double y, size_t j) {
        const double s = y / L * n;
        const size_t i = static_cast<size_t>(std::min(static_cast<int>(s), n - 1));
        return exact[i][j] + (s - static_cast<double>(i)) * (exact[i + 1][j] - exact[i][j]);
    };
    const std::string moving = N_DIM == 2 ? "[300.0, 0.0]" : "[300.0, 0.0, 0.0]";
    std::ostringstream input;
    input << std::setprecision(17) << "[run]\nt_stop = 2e-4\ncfl = 0.4\n"
          << channel(16, L, "type = \"wall_isothermal\"\nT = 300.0\n",
                     "type = \"wall_isothermal\"\nT = 500.0\nu = " + moving + "\n")
          << "[initialize]\ntype = \"analytical\"\np = \"" << p << "\"\nT = \"" << conduction_guess(T0, T1, L)
          << "\"\nu = " << (N_DIM == 2 ? "[\"3e5 * y\", \"0.0\"]" : "[\"3e5 * y\", \"0.0\", \"0.0\"]")
          << "\nX = " << AIR << "\n" << viscous_mixture(WALL_GASES);
    Solver solver;
    run(solver, input.str());
    const Profile u = column(solver, 0, -1, 1), T = column(solver, N_DIM + 1, -1, 1);
    const double total = air.conduction_integral(T0, T1);
    double err_u = 0.0, err_T = 0.0, heating = 0.0;
    for (size_t c = 0; c < T.x.size(); c++) {
        err_u = std::max(err_u, std::abs(u.value[c] - at(u.x[c], 0)));
        err_T = std::max(err_T, std::abs(T.value[c] - at(T.x[c], 1)));
        heating = std::max(heating, at(T.x[c], 1) - air.invert(T0, total * T.x[c] / L, T0, T1));
    }
    RecordProperty("max_error_u", std::to_string(err_u));
    RecordProperty("max_error_T", std::to_string(err_T));
    RecordProperty("viscous_heating_K", std::to_string(heating));
    EXPECT_GT(heating, 5.0);
    EXPECT_LT(err_u, tol(0.3, 1.0));
    EXPECT_LT(err_T, tol(0.1, 0.5));
}

TEST(MixtureTransportTest, WallsAreImpermeableToEverySpecies) {
    // H2 and N2 in a closed channel, at rest at 600 K between walls at 300 K
    // and 900 K, or heated through one wall, or with a temperature gradient
    // between adiabatic walls, diffuse with the Soret effect against the
    // walls: no species crosses them, so each species' total stays to
    // round-off; with adiabatic walls also the total energy, so no enthalpy
    // leaves either
    const std::string walls[][2] = {
        {"type = \"wall_isothermal\"\nT = 300.0\n", "type = \"wall_isothermal\"\nT = 900.0\n"},
        {"type = \"wall_heat_flux\"\nq = 1e4\n", "type = \"wall_isothermal\"\nT = 600.0\n"},
        {"type = \"wall_adiabatic\"\n", "type = \"wall_adiabatic\"\n"}};
    for (const auto & [bottom, top] : walls) {
        const bool adiabatic = bottom.find("adiabatic") != std::string::npos;
        std::ostringstream input;
        input << std::setprecision(17) << "[run]\nn_steps = 300\ncfl = 0.4\n" << channel(16, 1e-3, bottom, top)
              << "[initialize]\ntype = \"analytical\"\np = \"1000.0\"\nT = \"" << (adiabatic ? "300.0 + 6e5 * y" : "600.0")
              << "\"\nu = " << velocity("0.0") << "\nX = { H2 = 1.0, N2 = 1.0 }\n"
              << viscous_mixture(WALL_GASES, "mixture_averaged", true);
        std::string text = input.str();
        text.insert(text.find("[physics]\n") + 10, "phase = \"hydrogen\"\n");
        Solver solver;
        solver.init(parse_toml(text));
        const std::vector<rtype> before = solver.integrate_species();
        const auto energy_before = solver.integrate_conservatives();
        solver.run();
        const std::vector<rtype> after = solver.integrate_species();
        const auto energy_after = solver.integrate_conservatives();
        double total = 0.0;
        for (rtype m : before) total += static_cast<double>(m);
        for (size_t k = 0; k < before.size(); k++) {
            EXPECT_NEAR(static_cast<double>(after[k]), static_cast<double>(before[k]), tol(1e-13, 1e-5) * total)
                << bottom << solver.get_species_names()[k];
        }
        const double E0 = static_cast<double>(energy_before[N_DIM + 1]), E1 = static_cast<double>(energy_after[N_DIM + 1]);
        if (adiabatic) {
            EXPECT_NEAR(E1 / E0, 1.0, tol(1e-13, 1e-5));
        } else {
            EXPECT_GT(std::abs(E1 / E0 - 1.0), 1e-4) << "heat crosses " << bottom;
        }
        // and the light species gathered at the hot end
        solver.update_primitives();
        solver.copy_device_to_host();
        const Profile T = column(solver, N_DIM + 1, -1, 1), Y = column(solver, 0, 0, 1);
        size_t hot = 0, cold = 0;
        for (size_t c = 0; c < T.x.size(); c++) {
            if (T.value[c] > T.value[hot]) hot = c;
            if (T.value[c] < T.value[cold]) cold = c;
        }
        EXPECT_GT(Y.value[hot], Y.value[cold]) << bottom;
    }
}

/**
 * @file liquid_library_test.cpp
 * @author Matthew Bonanni (mbonanni001@gmail.com)
 * @brief Tests of the liquid library: tables fitted to their tolerance and
 *        evaluated on the device against an analytic liquid, their inverses
 *        exact to round-off, and with CoolProp every library fluid's tables
 *        against CoolProp itself and an isolated water droplet's wet-bulb
 *        temperature against CoolProp's water; the input errors of a named
 *        fluid.
 * @version 0.1
 * @date 2026-11-25
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 *
 */

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <functional>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "data_paths.h"
#include "liquid_library.h"
#include "mechanism.h"
#include "particle_evaporation.h"
#include "solver.h"
#include "test_fixtures.h"
#include "thermo.h"
#include "transport.h"

namespace {

/** @brief The liquid of the tables, every property tabulated, its enthalpy from T_ref. */
particles::Liquid liquid_of(const particles::LiquidTables & t) {
    particles::Liquid l;
    Kokkos::View<double *, Kokkos::HostSpace> host("liquid_tables", t.table.size());
    std::copy(t.table.begin(), t.table.end(), host.data());
    l.host_table = host;
    l.table = Kokkos::create_mirror_view_and_copy(Kokkos::DefaultExecutionSpace::memory_space(), host);
    l.density_curve = t.density;
    l.enthalpy_curve = t.enthalpy;
    l.temperature_curve = t.temperature;
    l.latent_curve = t.latent;
    l.pressure_curve = t.pressure;
    l.boiling_curve = t.boiling;
    l.viscosity_curve = t.viscosity;
    l.conductivity_curve = t.conductivity;
    l.tension_curve = t.tension;
    l.boiling_temperature = t.T_ref;
    l.critical_temperature = t.T_max;
    return l;
}

/** @brief What the liquid gives at each temperature, evaluated on the device. */
enum Column : uint32_t { DENSITY, SPECIFIC_HEAT, SENSIBLE, LATENT, PRESSURE, VISCOSITY, CONDUCTIVITY, TENSION,
                         N_COLUMNS };

std::vector<std::array<double, N_COLUMNS>> on_device(const particles::Liquid & liquid, const std::vector<double> & T) {
    Kokkos::View<double *, Kokkos::HostSpace> h_T("T", T.size());
    std::copy(T.begin(), T.end(), h_T.data());
    const auto d_T = Kokkos::create_mirror_view_and_copy(Kokkos::DefaultExecutionSpace::memory_space(), h_T);
    Kokkos::View<double **, Kokkos::LayoutRight> out("out", T.size(), N_COLUMNS);
    const particles::Liquid l = liquid;
    Kokkos::parallel_for("liquid_on_device", T.size(), KOKKOS_LAMBDA(const size_t i) {
        const double t = d_T(i);
        out(i, DENSITY) = l.density_at(t);
        out(i, SPECIFIC_HEAT) = l.specific_heat_at(t);
        out(i, SENSIBLE) = l.sensible(t);
        out(i, LATENT) = l.latent_at(t, 0.0);
        out(i, PRESSURE) = l.saturation_pressure(t);
        out(i, VISCOSITY) = l.viscosity_at(t);
        out(i, CONDUCTIVITY) = l.conductivity_at(t);
        out(i, TENSION) = l.surface_tension_at(t);
    });
    const auto h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), out);
    std::vector<std::array<double, N_COLUMNS>> rows(T.size());
    for (size_t i = 0; i < T.size(); i++) {
        for (uint32_t c = 0; c < N_COLUMNS; c++) rows[i][c] = h(i, c);
    }
    return rows;
}

/** @brief Round trips on the device: temperature_of(sensible(T)) and boiling_temperature_at(p) at p_sat(T). */
std::vector<std::array<double, 2>> round_trips(const particles::Liquid & liquid, const std::vector<double> & T,
                                               const std::vector<double> & p) {
    Kokkos::View<double **, Kokkos::LayoutRight, Kokkos::HostSpace> h_in("in", T.size(), 2);
    for (size_t i = 0; i < T.size(); i++) {
        h_in(i, 0) = T[i];
        h_in(i, 1) = p[i];
    }
    const auto in = Kokkos::create_mirror_view_and_copy(Kokkos::DefaultExecutionSpace::memory_space(), h_in);
    Kokkos::View<double **, Kokkos::LayoutRight> out("out", T.size(), 2);
    const particles::Liquid l = liquid;
    Kokkos::parallel_for("liquid_round_trips", T.size(), KOKKOS_LAMBDA(const size_t i) {
        out(i, 0) = l.temperature_of(l.sensible(in(i, 0)));
        out(i, 1) = l.boiling_temperature_at(in(i, 1));
    });
    const auto h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), out);
    std::vector<std::array<double, 2>> rows(T.size());
    for (size_t i = 0; i < T.size(); i++) rows[i] = {h(i, 0), h(i, 1)};
    return rows;
}

std::vector<double> spaced(const double lo, const double hi, const size_t n) {
    std::vector<double> T(n);
    for (size_t i = 0; i < n; i++) T[i] = lo + (hi - lo) * static_cast<double>(i) / static_cast<double>(n - 1);
    return T;
}

// An analytic liquid with the shapes near the critical point that the tables must resolve
constexpr double TC = 600.0, PC = 2e7, T_TRIPLE = 250.0;
double tau(const double T) { return 1.0 - T / TC; }
particles::LiquidState analytic_state(const double T) {
    particles::LiquidState s;
    s.density = 300.0 + 600.0 * std::pow(tau(T), 0.35);
    s.specific_heat = 2000.0 + 300.0 / std::sqrt(tau(T));
    s.latent_heat = 2e6 * std::pow(tau(T), 0.38);
    s.saturation_pressure = PC * std::exp(7.0 * (1.0 - TC / T));
    s.viscosity = std::exp(-11.0 + 900.0 / T);
    s.conductivity = 0.7 - 4e-4 * T;
    s.surface_tension = 0.06 * std::pow(tau(T), 1.26);
    return s;
}
double analytic_enthalpy(const double T) { return 2000.0 * T - 600.0 * TC * std::sqrt(tau(T)); }

particles::PropertySource analytic_source() {
    particles::PropertySource s;
    s.T_triple = T_TRIPLE;
    s.T_critical = TC;
    s.p_critical = PC;
    s.states = [](const std::vector<double> & T) {
        std::vector<particles::LiquidState> out;
        for (const double t : T) out.push_back(analytic_state(t));
        return out;
    };
    s.saturation_temperature = [](const double p) { return TC / (1.0 - std::log(p / PC) / 7.0); };
    return s;
}

double relative(const double a, const double b) { return std::fabs(a - b) / std::fabs(b); }

} // namespace

TEST(LiquidLibraryTest, TablesOfAnAnalyticLiquidMeetTheirToleranceOnTheDevice) {
    // Between the knots too, not only at the points the fit checks: 3 times the tolerance at most
    for (const double tolerance : {1e-4, 1e-7}) {
        const double T_ref = 400.0;
        const particles::LiquidTables t = particles::tabulate(analytic_source(), T_ref, tolerance);
        EXPECT_DOUBLE_EQ(t.T_max, TC * (1.0 - particles::CRITICAL_MARGIN));
        for (const auto & r : t.report) EXPECT_LE(r.error, tolerance) << r.name;
        const particles::Liquid l = liquid_of(t);
        const std::vector<double> T = spaced(t.T_min, t.T_max, 4001);
        const auto rows = on_device(l, T);
        std::array<double, N_COLUMNS> worst{};
        for (size_t i = 0; i < T.size(); i++) {
            const particles::LiquidState s = analytic_state(T[i]);
            const std::array<double, N_COLUMNS> exact = {
                s.density, s.specific_heat, analytic_enthalpy(T[i]) - analytic_enthalpy(T_ref), s.latent_heat,
                s.saturation_pressure, s.viscosity, s.conductivity, s.surface_tension};
            for (uint32_t c = 0; c < N_COLUMNS; c++) {
                const double scale = c == SENSIBLE ? std::max(std::fabs(exact[c]), analytic_state(T_ref).specific_heat * T_ref)
                                                   : std::fabs(exact[c]);
                worst[c] = std::max(worst[c], std::fabs(rows[i][c] - exact[c]) / scale);
            }
        }
        for (uint32_t c = 0; c < N_COLUMNS; c++) EXPECT_LE(worst[c], 3.0 * tolerance) << "column " << c << ", " << tolerance;
    }
}

TEST(LiquidLibraryTest, InverseTablesAreExactToRoundOffAndTheEnthalpyContinuesPastItsEnds) {
    const particles::LiquidTables t = particles::tabulate(analytic_source(), 400.0, 1e-5);
    const particles::Liquid l = liquid_of(t);
    // Past the ends the enthalpy continues at its end slopes, and temperature_of inverts it there too
    const std::vector<double> T = spaced(t.T_min - 40.0, t.T_max + 40.0, 2001);
    std::vector<double> p;
    for (const double x : T) p.push_back(analytic_state(std::clamp(x, t.T_min, t.T_max)).saturation_pressure);
    const auto rows = round_trips(l, T, p);
    double worst_T = 0.0, worst_boil = 0.0;
    for (size_t i = 0; i < T.size(); i++) {
        worst_T = std::max(worst_T, relative(rows[i][0], T[i]));
        if (T[i] > t.T_min && T[i] < t.T_max) {
            // The boiling temperature inverts the tabulated saturation curve itself
            worst_boil = std::max(worst_boil, std::fabs(std::log(l.saturation_pressure(rows[i][1]) / p[i])));
        }
    }
    EXPECT_LE(worst_T, 1e-13);
    EXPECT_LE(worst_boil, 1e-13);
    // At and above the saturation pressure at the end of the tables the saturation curve has ended
    const auto ended = round_trips(l, {300.0, 300.0}, {analytic_state(t.T_max).saturation_pressure * 1.0001, 10.0 * PC});
    EXPECT_EQ(ended[0][1], t.T_max);
    EXPECT_EQ(ended[1][1], t.T_max);
    // The specific heat is the enthalpy's slope, also past the ends
    double slope;
    l.enthalpy_curve.evaluate(l.values(), t.T_max, slope);
    EXPECT_DOUBLE_EQ(l.specific_heat_at(t.T_max + 40.0), slope);
    EXPECT_NEAR(l.sensible(t.T_max + 40.0) - l.sensible(t.T_max), 40.0 * slope, 1e-9 * l.sensible(t.T_max));
}

namespace {

/** @brief A quiescent droplet of the given liquid in air of mechanisms/h2o2.yaml at 1000 K and 1 atm. */
std::string droplet_case(const std::string & liquid, const std::string & particles_extra = "") {
    std::ostringstream s;
    s << "[run]\nn_steps = 200\ndt = 1e-4\n[mesh]\ntype = \"cartesian\"\nNx = 8\nNy = 8\nLx = 1.0\nLy = 1.0\n";
    if (N_DIM == 3) s << "Nz = 4\nLz = 0.5\n";
    s << periodic_sides(N_DIM == 3 ? "xyz" : "xy")
      << "[initialize]\ntype = \"analytical\"\np = \"101325.0\"\nT = \"1000\"\nu = [\"0.0\", \"0.0\""
      << (N_DIM == 3 ? ", \"0.0\"]" : "]") << "\nX = { O2 = 0.21, N2 = 0.79 }\n"
      << "[numerics]\nriemann_solver = \"HLLC\"\ntime_integrator = \"SSPRK3\"\n"
      << "[numerics.face_reconstruction]\ntype = \"MUSCL\"\n"
      << "[physics]\ntype = \"navier_stokes\"\ngas = \"mixture\"\nmechanism = \"" << source_dir()
      << "/mechanisms/h2o2.yaml\"\ntransport = \"mixture_averaged\"\n[output]\ncheck_interval = 1000000\n"
      << "[particles]\ntype = \"droplet\"\ndiameter = 50e-6\ntemperature = 300.0\nevaporation = \"abramzon_sirignano\"\n"
      << "velocity = [0.0, 0.0" << (N_DIM == 3 ? ", 0.0]" : "]") << "\npoints = [[0.43, 0.56"
      << (N_DIM == 3 ? ", 0.21]]" : "]]") << "\n" << particles_extra << "[particles.liquid]\nspecies = \"H2O\"\n" << liquid;
    return s.str();
}

/** @brief The InputError of setting up input, or "" if there is none. */
std::string input_error(const std::string & input) {
    Solver solver;
    solver.set_distributed(false);
    try {
        solver.init(parse_toml(input));
    } catch (const InputError & e) {
        return e.what();
    }
    return "";
}

} // namespace

TEST(LiquidLibraryTest, NamedFluidsAreCheckedAndNeedCoolProp) {
    const std::string misspelled = input_error(droplet_case("fluid = \"watr\"\n"));
    EXPECT_NE(misspelled.find("particles.liquid.fluid"), std::string::npos) << misspelled;
    EXPECT_NE(misspelled.find("water"), std::string::npos) << misspelled;
    if (particles::coolprop_available()) {
        const std::string tolerance = input_error(droplet_case("fluid = \"water\"\ntolerance = 0.5\n"));
        EXPECT_NE(tolerance.find("particles.liquid.tolerance"), std::string::npos) << tolerance;
    } else {
        // Without CoolProp a named fluid says how to get it, and a liquid given property by property still works
        const std::string error = input_error(droplet_case("fluid = \"water\"\n"));
        EXPECT_NE(error.find("particles.liquid.fluid"), std::string::npos) << error;
        EXPECT_NE(error.find("-DMallard_ENABLE_COOLPROP=ON"), std::string::npos) << error;
    }
    EXPECT_EQ(input_error(droplet_case("density = 958.0\nspecific_heat = 4216.0\nlatent_heat = 2257000.0\n"
                                       "boiling_temperature = 373.15\ncritical_temperature = 647.1\n"
                                       "critical_pressure = 22064000.0\n")), "");
}

#ifdef Mallard_HAS_COOLPROP

TEST(LiquidLibraryTest, EveryFluidsTablesMatchCoolPropOnTheDevice) {
    // At 4001 temperatures from the triple point to the end of the tables, at the default tolerance; prints the
    // largest relative error of each property (the verification of docs/design/liquid_library.md)
    const double tolerance = 1e-6;
    std::cout << std::setprecision(2) << std::scientific;
    for (const particles::LibraryFluid & f : particles::library_fluids()) {
        const particles::PropertySource source = particles::coolprop_source(f.coolprop);
        const double T_ref = source.saturation_temperature(101325.0);
        const particles::LiquidTables t = particles::tabulate(source, T_ref, tolerance);
        const particles::Liquid l = liquid_of(t);
        const std::vector<double> T = spaced(t.T_min, t.T_max, 4001);
        const std::vector<particles::LiquidState> exact = source.states(T);
        const auto rows = on_device(l, T);
        std::vector<double> p;
        for (const auto & s : exact) p.push_back(s.saturation_pressure);
        const auto trips = round_trips(l, T, p);
        std::array<double, N_COLUMNS + 1> worst{};
        for (size_t i = 0; i < T.size(); i++) {
            const particles::LiquidState & s = exact[i];
            const std::array<double, N_COLUMNS> e = {s.density, s.specific_heat, 0.0, s.latent_heat,
                                                     s.saturation_pressure, s.viscosity, s.conductivity, s.surface_tension};
            for (uint32_t c = 0; c < N_COLUMNS; c++) {
                if (c != SENSIBLE) worst[c] = std::max(worst[c], relative(rows[i][c], e[c]));
            }
            // The boiling temperature at CoolProp's saturation pressure is CoolProp's temperature
            worst[N_COLUMNS] = std::max(worst[N_COLUMNS], relative(trips[i][1], T[i]));
        }
        std::cout << f.name << " (" << t.T_min << " to " << t.T_max << " K):";
        const char * names[] = {"density", "c_p", "", "latent", "p_sat", "viscosity", "conductivity", "tension",
                                "T_boil"};
        // A table that reaches the most knots before its tolerance (conductivity's critical enhancement) holds
        // the error it reports; the column of each table in t.report
        const size_t reported[] = {0, 1, 0, 3, 4, 6, 7, 8, 5};
        for (uint32_t c = 0; c <= N_COLUMNS; c++) {
            if (c == SENSIBLE) continue;
            std::cout << " " << names[c] << " " << worst[c];
            const double fit = t.report[reported[c]].error;
            EXPECT_LE(fit, 1e-5) << f.name << " " << names[c];
            EXPECT_LE(worst[c], 3.0 * std::max(tolerance, fit)) << f.name << " " << names[c];
        }
        std::cout << "\n";
        for (const auto & r : t.report) std::cout << "    " << r.name << ": " << r.knots << " knots, " << r.error << "\n";
    }
}

namespace {

/** @brief CoolProp's water droplet at rest in the air of droplet_case(): its Abramzon-Sirignano wet-bulb temperature. */
double coolprop_wet_bulb() {
    const particles::PropertySource water = particles::coolprop_source("Water");
    const chemistry::Mechanism mech = chemistry::read_mechanism(source_dir() + "/mechanisms/h2o2.yaml");
    const auto thermo = chemistry::make_thermo_table<Kokkos::HostSpace>(mech);
    const auto transport = chemistry::make_transport_table<Kokkos::HostSpace>(mech, chemistry::TransportModel::MIXTURE_AVERAGED);
    const auto names = mech.species_names();
    const auto at = [&](const char * name) { return static_cast<uint32_t>(std::find(names.begin(), names.end(), name) - names.begin()); };
    const uint32_t k_v = at("H2O");
    std::vector<double> Y(names.size(), 0.0);
    const double W_O2 = mech.species[at("O2")].molecular_weight, W_N2 = mech.species[at("N2")].molecular_weight;
    const double W_air = 0.21 * W_O2 + 0.79 * W_N2;
    Y[at("O2")] = 0.21 * W_O2 / W_air;
    Y[at("N2")] = 0.79 * W_N2 / W_air;
    const double p = 101325.0, T_gas = 1000.0, W_v = 1.0 / thermo.inv_W(k_v);
    // Zero heat into the liquid: cp_v (T - T_s) / L(T_s) = (1 + B_M)^phi - 1, phi = (cp_v / cp_f) / Le_f
    const auto balance = [&](const double Ts) {
        const particles::LiquidState s = water.states({Ts})[0];
        const double X_s = s.saturation_pressure / p;
        const double Y_s = X_s * W_v / (X_s * W_v + (1.0 - X_s) * W_air);
        const double Yv_f = 2.0 / 3.0 * Y_s;
        std::vector<double> Y_f(Y.size());
        for (uint32_t k = 0; k < Y.size(); k++) Y_f[k] = k == k_v ? Yv_f : Y[k] * (1.0 - Yv_f);
        const chemistry::MassFractions y{Y_f.data()};
        const double T_f = Ts + (T_gas - Ts) / 3.0;
        const double rho = p / (thermo.gas_constant(y) * T_f), cp = thermo.cp_mass(T_f, y);
        std::vector<double> D(Y.size());
        double mu, k;
        transport.properties(T_f, p, rho, cp, y, mu, k, D.data());
        const double cp_v = chemistry::GAS_CONSTANT * thermo.inv_W(k_v) * thermo.cp_R(k_v, decltype(thermo)::powers(T_f));
        const double phi = cp_v / cp / (k / (rho * cp * D[k_v]));
        return cp_v * (T_gas - Ts) / s.latent_heat - (std::pow(1.0 + Y_s / (1.0 - Y_s), phi) - 1.0);
    };
    double a = 280.0, b = 370.0;
    for (int i = 0; i < 200; i++) ((balance(0.5 * (a + b)) > 0.0) == (balance(a) > 0.0) ? a : b) = 0.5 * (a + b);
    return 0.5 * (a + b);
}

} // namespace

TEST(LiquidLibraryTest, WaterDropletHeatsToTheWetBulbTemperatureOfCoolPropsWater) {
    // Its saturation pressure and latent heat at the surface from the tables: the wet bulb of CoolProp's curves
    const double T_wb = coolprop_wet_bulb();
    Solver solver;
    solver.set_distributed(false);
    solver.init(parse_toml(droplet_case("fluid = \"water\"\n")));
    solver.run();
    const auto records = solver.get_particles()->gather();
    ASSERT_EQ(records.size(), 1u);
    EXPECT_NEAR(records[0].T, T_wb, 1e-6 * T_wb);
    EXPECT_LT(records[0].d, 0.9 * 50e-6);
}

#endif

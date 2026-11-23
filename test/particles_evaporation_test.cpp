/**
 * @file particles_evaporation_test.cpp
 * @author Matthew Bonanni (mbonanni001@gmail.com)
 * @brief Tests of evaporating droplets: the liquid's saturation curve, the d^2
 *        law's rate and an isolated droplet's wet-bulb temperature against the
 *        film properties computed here from the mechanism, boiling, closed-box
 *        conservation of mass, species, momentum and energy, removal at the
 *        minimum diameter, input errors, and bitwise independence of the rank
 *        count, step graphs and restarts.
 * @version 0.1
 * @date 2026-11-21
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 *
 */

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <functional>
#include <numbers>
#include <span>
#include <sstream>
#include <string>
#include <vector>

#include "comm.h"
#include "data_paths.h"
#include "mechanism.h"
#include "mpi_compare.h"
#include "particles.h"
#include "scratch.h"
#include "solver.h"
#include "test_fixtures.h"
#include "thermo.h"
#include "transport.h"

namespace {

constexpr double PI = std::numbers::pi;
const std::string MECHANISM = source_dir() + "/mechanisms/h2o2.yaml";

// Water, and air at 1000 K and one atmosphere
constexpr double RHO_L = 958.0, CP_L = 4216.0, L_B = 2.257e6, T_B = 373.15, P_B = 101325.0, T_C = 647.1,
                 P_C = 22.064e6;
constexpr double P_GAS = 101325.0, T_GAS = 1000.0;

std::string vector3(double x, double y, double z) {
    std::ostringstream s;
    s.precision(17);
    s << "[" << x << ", " << y;
    if (N_DIM == 3) s << ", " << z;
    s << "]";
    return s.str();
}

std::string liquid_table(const std::string & extra = "") {
    return "[particles.liquid]\nspecies = \"H2O\"\ndensity = 958.0\nspecific_heat = 4216.0\nlatent_heat = 2257000.0\n"
           "boiling_temperature = 373.15\ncritical_temperature = 647.1\ncritical_pressure = 22064000.0\n" + extra;
}

/**
 * @brief A periodic box of side length, n cells per side (n / 2 along z, half as long), of air at 1000 K and one
 *        atmosphere (or temperature T) with velocity u, and droplets of the given [particles] keys (the liquid table after them).
 */
std::string box(double length, uint32_t n, const std::string & u, const std::string & run, const std::string & particles,
                const std::string & liquid_extra = "", const std::string & T = "1000") {
    std::ostringstream s;
    s.precision(17);
    s << "[run]\n" << run << "[mesh]\ntype = \"cartesian\"\nNx = " << n << "\nNy = " << n << "\nLx = " << length
      << "\nLy = " << length << "\n";
    if (N_DIM == 3) s << "Nz = " << n / 2 << "\nLz = " << 0.5 * length << "\n";
    s << periodic_sides(N_DIM == 3 ? "xyz" : "xy") << "[initialize]\ntype = \"analytical\"\np = \"" << P_GAS
      << "\"\nT = \"" << T << "\"\nu = " << u << "\nX = { O2 = 0.21, N2 = 0.79 }\n"
      << "[numerics]\nriemann_solver = \"HLLC\"\ntime_integrator = \"SSPRK3\"\n"
      << "[numerics.face_reconstruction]\ntype = \"MUSCL\"\n"
      << "[physics]\ntype = \"navier_stokes\"\ngas = \"mixture\"\nmechanism = \"" << MECHANISM
      << "\"\ntransport = \"mixture_averaged\"\n[output]\ncheck_interval = 1000000\n"
      << "[particles]\ntype = \"droplet\"\n" << particles << liquid_table(liquid_extra);
    return s.str();
}

std::string zero_velocity() { return "[\"0.0\", \"0.0\"" + std::string(N_DIM == 3 ? ", \"0.0\"]" : "]"); }

std::string run(uint32_t n_steps, double dt) {
    std::ostringstream s;
    s.precision(17);
    s << "n_steps = " << n_steps << "\ndt = " << dt << "\n";
    return s.str();
}

/** @brief The air of box() and water, on the host: the film properties as the 1/3 rule gives them. */
struct Air {
    chemistry::Mechanism mech = chemistry::read_mechanism(MECHANISM);
    chemistry::ThermoTable<Kokkos::HostSpace> thermo = chemistry::make_thermo_table<Kokkos::HostSpace>(mech);
    chemistry::TransportTable<Kokkos::HostSpace> transport =
        chemistry::make_transport_table<Kokkos::HostSpace>(mech, chemistry::TransportModel::MIXTURE_AVERAGED);
    uint32_t k_v = 0;
    std::vector<double> Y;  // of the air

    Air() {
        const auto names = mech.species_names();
        k_v = static_cast<uint32_t>(std::find(names.begin(), names.end(), "H2O") - names.begin());
        Y.assign(names.size(), 0.0);
        const auto at = [&](const char * name) { return std::find(names.begin(), names.end(), name) - names.begin(); };
        const double W_O2 = mech.species[static_cast<size_t>(at("O2"))].molecular_weight;
        const double W_N2 = mech.species[static_cast<size_t>(at("N2"))].molecular_weight;
        const double W = 0.21 * W_O2 + 0.79 * W_N2;
        Y[static_cast<size_t>(at("O2"))] = 0.21 * W_O2 / W;
        Y[static_cast<size_t>(at("N2"))] = 0.79 * W_N2 / W;
    }

    double h_vapor(double T) const {
        return chemistry::GAS_CONSTANT * T * thermo.inv_W(k_v) * thermo.h_RT(k_v, decltype(thermo)::powers(T));
    }
    double cp_vapor(double T) const {
        return chemistry::GAS_CONSTANT * thermo.inv_W(k_v) * thermo.cp_R(k_v, decltype(thermo)::powers(T));
    }
    /** @brief Kirchhoff's latent heat of the water of box(). */
    double latent(double T) const { return h_vapor(T) - (h_vapor(T_B) - L_B) - CP_L * (T - T_B); }

    /** @brief The film at a surface at Ts in the air of box(). */
    struct Film {
        double rho, D, k, cp, B_M;
    };
    Film film(double Ts) const {
        const double R_v = chemistry::GAS_CONSTANT * thermo.inv_W(k_v);
        const double X_s = P_B * std::exp(L_B / R_v * (1.0 / T_B - 1.0 / Ts)) / P_GAS;
        double n_air = 0.0;
        for (uint32_t k = 0; k < Y.size(); k++) n_air += Y[k] * thermo.inv_W(k);
        const double W_v = 1.0 / thermo.inv_W(k_v), W_air = 1.0 / n_air;
        const double Y_s = X_s * W_v / (X_s * W_v + (1.0 - X_s) * W_air);
        std::vector<double> Y_f(Y.size());
        const double Yv_f = 2.0 / 3.0 * Y_s;
        for (uint32_t k = 0; k < Y.size(); k++) Y_f[k] = k == k_v ? Yv_f : Y[k] * (1.0 - Yv_f);
        const chemistry::MassFractions y{Y_f.data()};
        const double T_f = Ts + (T_GAS - Ts) / 3.0;
        Film f;
        f.rho = P_GAS / (thermo.gas_constant(y) * T_f);
        f.cp = thermo.cp_mass(T_f, y);
        std::vector<double> D(Y.size());
        double mu;
        transport.properties(T_f, P_GAS, f.rho, f.cp, y, mu, f.k, D.data());
        f.D = D[k_v];
        f.B_M = Y_s / (1.0 - Y_s);
        return f;
    }
};

/** @brief The root of f in [a, b] by bisection, f(a) and f(b) of opposite signs. */
double bisect(const std::function<double(double)> & f, double a, double b) {
    const double fa = f(a);
    for (int i = 0; i < 200; i++) {
        const double m = 0.5 * (a + b);
        if ((f(m) > 0.0) == (fa > 0.0)) {
            a = m;
        } else {
            b = m;
        }
    }
    return 0.5 * (a + b);
}

std::string number(double x) {
    std::ostringstream s;
    s.precision(17);
    s << x;
    return s.str();
}

/** @brief One droplet at rest at the center of box(1, 8): the d^2 law keeps a gas at rest at rest. */
std::string quiescent(double d0, double T0, const std::string & model, const std::string & run_keys,
                      const std::string & extra = "") {
    return box(1.0, 8, zero_velocity(), run_keys,
               "diameter = " + number(d0) + "\ntemperature = " + number(T0) + "\nevaporation = \"" + model +
                   "\"\nvelocity = " + vector3(0.0, 0.0, 0.0) + "\npoints = [" + vector3(0.43, 0.56, 0.21) + "]\n" +
                   extra);
}

} // namespace

TEST(DropletTest, SaturationCurvesPassThroughTheBoilingPointAndInvert) {
    particles::Liquid l;
    l.boiling_temperature = T_B;
    l.boiling_pressure = P_B;
    l.latent_heat = L_B;
    l.R_vapor = 8314.46 / 18.015;
    l.critical_temperature = T_C;
    l.critical_pressure = P_C;
    EXPECT_NEAR(l.saturation_pressure(T_B), P_B, 1e-9 * P_B);
    // Clausius-Clapeyron: d ln p / d(1/T) = -L / R_v
    const double slope = (std::log(l.saturation_pressure(350.0)) - std::log(l.saturation_pressure(300.0))) /
                         (1.0 / 350.0 - 1.0 / 300.0);
    EXPECT_NEAR(slope, -L_B / l.R_vapor, 1e-9 * L_B / l.R_vapor);
    for (const double p : {5e3, 1e5, 2e6}) EXPECT_NEAR(l.saturation_pressure(l.boiling_temperature_at(p)), p, 1e-9 * p);
    // Antoine's water (Pa, K): 101.3 kPa at 373.1 K
    l.antoine = true;
    l.A = 10.19621;
    l.B = 1730.63;
    l.C = -39.724;
    EXPECT_NEAR(l.saturation_pressure(373.15), 101325.0, 0.01 * 101325.0);
    for (const double p : {5e3, 1e5, 2e6}) EXPECT_NEAR(l.saturation_pressure(l.boiling_temperature_at(p)), p, 1e-9 * p);
    // Above the critical pressure the saturation curve has ended
    EXPECT_EQ(l.boiling_temperature_at(10.0 * P_C), T_C);
    l.antoine = false;
    EXPECT_EQ(l.boiling_temperature_at(10.0 * P_C), T_C);
}

TEST(DropletTest, FilmCorrectionOfAbramzonAndSirignano) {
    for (const double B : {-0.3, 1e-8, 0.05, 1.0, 7.0}) {
        EXPECT_NEAR(particles::film_correction(B), std::pow(1.0 + B, 0.7) * std::log(1.0 + B) / B, 1e-7) << B;
        EXPECT_NEAR(particles::log_ratio(B), std::log(1.0 + B) / B, 1e-7) << B;
    }
}

TEST(DropletTest, DiameterSquaredFallsLinearlyAtTheD2LawRate) {
    // At its wet-bulb temperature, where the heat from the gas, 2 pi d k (T - T_s), evaporates it,
    // 2 pi d rho D ln(1 + B_M) L, a droplet at rest keeps its temperature, and d^2 falls at
    // K = 8 rho D ln(1 + B_M) / rho_l, film properties by the 1/3 rule
    const Air air;
    const double T_wb = bisect([&](double T) {
        const Air::Film f = air.film(T);
        return f.k * (T_GAS - T) - f.rho * f.D * std::log1p(f.B_M) * air.latent(T);
    }, 280.0, T_B - 1.0);
    const Air::Film f = air.film(T_wb);
    const double K = 8.0 * f.rho * f.D * std::log1p(f.B_M) / RHO_L;
    const double d0 = 50e-6, t_life = d0 * d0 / K, dt = t_life / 400.0;
    const std::string quantities =
        "[[particles.quantities]]\nname = \"T75\"\nfirst_time = \"D_P^2 < " + number(0.75 * d0 * d0) + "\"\n"
        "[[particles.quantities]]\nname = \"T25\"\nfirst_time = \"D_P^2 < " + number(0.25 * d0 * d0) + "\"\n"
        "[[particles.quantities]]\nname = \"T_LO\"\nmin = \"T_P\"\n[[particles.quantities]]\nname = \"T_HI\"\nmax = \"T_P\"\n";
    Solver solver;
    solver.set_distributed(false);
    solver.init(parse_toml(quiescent(d0, T_wb, "d2_law", run(320, dt), quantities)));
    solver.run();
    const auto records = solver.get_particles()->gather();
    ASSERT_EQ(records.size(), 1u);
    const auto & q = records[0].q;
    const double tolerance = dt + roundoff(1e-3) * t_life;
    EXPECT_NEAR(q[0], 0.25 * t_life, tolerance);
    EXPECT_NEAR(q[1], 0.75 * t_life, tolerance);
    EXPECT_NEAR(q[2], T_wb, roundoff(1e-6) * T_wb);
    EXPECT_NEAR(q[3], T_wb, roundoff(1e-6) * T_wb);
    // At 0.8 t_life
    EXPECT_NEAR(records[0].d * records[0].d, 0.2 * d0 * d0, roundoff(1e-3) * d0 * d0);
}

TEST(DropletTest, DropletIsRemovedAtTheMinimumDiameterAfterItsLifetime) {
    const Air air;
    const Air::Film f = air.film(340.0);
    const double K = 8.0 * f.rho * f.D * std::log1p(f.B_M) / RHO_L;
    const double d0 = 20e-6, t_life = d0 * d0 / K;
    for (const auto & [fraction, alive] : {std::pair{0.6, true}, std::pair{1.6, false}}) {
        Solver solver;
        solver.set_distributed(false);
        solver.init(parse_toml(quiescent(d0, 340.0, "d2_law", run(200, fraction * t_life / 200.0))));
        solver.run();
        EXPECT_EQ(solver.get_particles()->count(), alive ? 1u : 0u) << fraction;
        EXPECT_EQ(solver.get_particles()->evaporated(), alive ? 0u : 1u) << fraction;
    }
}

TEST(DropletTest, IsolatedDropletHeatsToTheWetBulbTemperatureOfAbramzonAndSirignano) {
    // At rest (Sh* = Nu* = 2), the droplet stops heating where the heat into the liquid,
    // mdot (cp_v (T - T_s) / B_T - L), vanishes: B_T = cp_v (T - T_s) / L, B_T = (1 + B_M)^phi - 1,
    // phi = (cp_v / cp_f) / Le_f
    const Air air;
    const auto B_T = [&](double T) {
        const Air::Film f = air.film(T);
        const double T_f = T + (T_GAS - T) / 3.0;
        const double Le = f.k / (f.rho * f.cp * f.D);
        return std::pow(1.0 + f.B_M, air.cp_vapor(T_f) / f.cp / Le) - 1.0;
    };
    const double T_wb = bisect([&](double T) {
        const double T_f = T + (T_GAS - T) / 3.0;
        return air.cp_vapor(T_f) * (T_GAS - T) / air.latent(T) - B_T(T);
    }, 280.0, T_B - 1.0);
    Solver solver;
    solver.set_distributed(false);
    solver.init(parse_toml(quiescent(50e-6, 300.0, "abramzon_sirignano", run(200, 1e-4))));
    solver.run();
    const auto records = solver.get_particles()->gather();
    ASSERT_EQ(records.size(), 1u);
    EXPECT_NEAR(records[0].T, T_wb, roundoff(1e-6) * T_wb);
    EXPECT_LT(records[0].d, 0.9 * 50e-6);
}

TEST(DropletTest, SuperheatedDropletBoilsDownTowardTheBoilingTemperature) {
    // Above the boiling temperature the heat from the gas and the droplet's superheat evaporate it: it cools
    // toward the boiling temperature, also in gas far hotter, and loses mass
    for (const char * model : {"abramzon_sirignano", "d2_law"}) {
        Solver solver;
        solver.set_distributed(false);
        solver.init(parse_toml(quiescent(100e-6, 450.0, model, run(100, 2e-4),
                                         "[[particles.quantities]]\nname = \"T_HI\"\nmax = \"T_P\"\n")));
        solver.run();
        const auto records = solver.get_particles()->gather();
        ASSERT_EQ(records.size(), 1u);
        EXPECT_LE(records[0].q[0], 450.0) << model;
        EXPECT_LT(records[0].T, 440.0) << model;
        EXPECT_GT(records[0].T, T_B) << model;
        EXPECT_LT(records[0].d, 0.97 * 100e-6) << model;
    }
}

namespace {

constexpr double WEIGHT = 2e5;

/**
 * @brief Droplets of diameter d0 thrown through a Taylor-Green flow of a periodic box of 10 cm, in a two-way
 *        coupled closed system of air between 600 and 1400 K, with the given extra [particles] keys.
 */
std::string thrown(double d0, uint32_t n_steps, const std::string & extra = "") {
    std::ostringstream keys;
    keys.precision(17);
    keys << "coupling = \"two-way\"\nweight = " << WEIGHT << "\ndiameter = " << d0
         << "\ntemperature = 320.0\nvelocity = " << vector3(9.0, -4.0, 2.0) << "\n" << extra
         << "[particles.lattice]\nlo = " << vector3(0.007, 0.011, 0.013) << "\nhi = " << vector3(0.093, 0.087, 0.041)
         << "\nn = " << (N_DIM == 3 ? "[5, 4, 2]" : "[5, 4]") << "\n";
    return box(0.1, 8,
               "[\"5 * sin(2 * pi * x / 0.1) * cos(2 * pi * y / 0.1)\", \"-5 * cos(2 * pi * x / 0.1) * sin(2 * pi * y / "
               "0.1)\"" + std::string(N_DIM == 3 ? ", \"0.0\"]" : "]"),
               run(n_steps, 5e-6), keys.str(), "", "1000 + 400 * sin(2 * pi * x / 0.1)");
}

/** @brief Mass, vapor mass, momentum and energy of gas and droplets, on all ranks (collective). */
std::array<double, N_DIM + 3> totals(Solver & solver) {
    solver.copy_device_to_host();
    const Mesh & mesh = *solver.get_mesh();
    const Air air;
    std::array<double, N_DIM + 3> t{};
    for (uint32_t c = 0; c < mesh.n_owned(); c++) {
        const double V = static_cast<double>(mesh.h_cell_measure(c));
        t[0] += V * static_cast<double>(solver.h_conservatives(c, 0));
        t[1] += V * static_cast<double>(solver.h_species(c, air.k_v));
        for (uint32_t i = 0; i <= N_DIM; i++) t[2 + i] += V * static_cast<double>(solver.h_conservatives(c, 1 + i));
    }
    if (solver.is_distributed()) comm::allreduce(std::span<double>(t), comm::Op::SUM);
    const double h_ref = air.h_vapor(T_B) - L_B;
    for (const auto & r : solver.get_particles()->gather()) {
        double v2 = 0.0;
        for (size_t d = 0; d < N_DIM; d++) {
            t[2 + d] += WEIGHT * r.m * r.v[d];
            v2 += r.v[d] * r.v[d];
        }
        t[0] += WEIGHT * r.m;
        t[1] += WEIGHT * r.m;
        t[N_DIM + 2] += WEIGHT * r.m * (h_ref + CP_L * (r.T - T_B) + 0.5 * v2);
    }
    return t;
}

} // namespace

TEST(DropletTest, ClosedBoxConservesMassVaporMomentumAndEnergyToRoundOff) {
    // Large droplets evaporate in part, most small ones entirely within the run (their remains go to the gas)
    SKIP_IN_SINGLE_PRECISION("compares totals at double-precision round-off");
    for (const double d0 : {40e-6, 6e-6}) {
        for (const char * width : {"0.0", "0.02"}) {
            const std::string input = thrown(d0, 200, std::string("deposition_width = ") + width + "\n");
            Solver start;
            start.init(parse_toml(input));
            const auto before = totals(start);
            Solver solver;
            solver.init(parse_toml(input));
            solver.run();
            const auto after = totals(solver);
            std::array<double, N_DIM + 3> scale;
            scale.fill(before[0]);
            for (size_t i = 0; i < N_DIM; i++) scale[2 + i] = 10.0 * before[0];
            scale[N_DIM + 2] = std::abs(before[N_DIM + 2]);
            for (size_t i = 0; i < before.size(); i++) {
                EXPECT_NEAR(after[i], before[i], 1e-13 * scale[i]) << "d0 " << d0 << ", width " << width << ", total " << i;
            }
            // The droplets did give the gas vapor, and small ones were removed
            const double vapor_before = before[1] - static_cast<double>(start.get_particles()->count()) * WEIGHT *
                                                        RHO_L * PI / 6.0 * d0 * d0 * d0;
            const double vapor_after = [&] {
                double carried = 0.0;
                for (const auto & r : solver.get_particles()->gather()) carried += WEIGHT * r.m;
                return after[1] - carried;
            }();
            EXPECT_GT(vapor_after - vapor_before, 0.01 * (before[1] - vapor_before)) << "d0 " << d0;
            if (d0 < 1e-5) {
                EXPECT_GT(solver.get_particles()->evaporated(), 0u);
                EXPECT_EQ(solver.get_particles()->count() + solver.get_particles()->evaporated(),
                          start.get_particles()->count());
            }
        }
    }
}

TEST(DropletTest, CoupledDropletsAreBitwiseIndependentOfRanksAndStepGraphs) {
    // Droplets in the hotter air evaporate within the run, some on each rank, the others not
    const std::string input = thrown(8e-6, 140, "deposition_width = 0.02\n");
    Solver serial;
    serial.set_distributed(false);
    serial.init(parse_toml(input));
    serial.run();
    const std::vector<particles::Record> expected = serial.get_particles()->gather();
    const uint64_t evaporated = serial.get_particles()->evaporated();
    ASSERT_GT(evaporated, 0u);
    ASSERT_GT(expected.size(), 0u);
    auto check = [&](const Solver & s) {
        const std::vector<particles::Record> records = s.get_particles()->gather();
        ASSERT_EQ(records.size(), expected.size());
        for (size_t k = 0; k < records.size(); k++) {
            EXPECT_EQ(records[k].x, expected[k].x) << "droplet " << k;
            EXPECT_EQ(records[k].v, expected[k].v) << "droplet " << k;
            EXPECT_EQ(records[k].T, expected[k].T) << "droplet " << k;
            EXPECT_EQ(records[k].m, expected[k].m) << "droplet " << k;
        }
        EXPECT_EQ(s.get_particles()->evaporated(), evaporated);
    };
    if (comm::size() == 1) {
        expect_matches_serial_with_and_without_step_graphs(input, check);
    } else {
        expect_matches_serial(input, check);
    }
}

TEST(DropletTest, RestartedDropletsContinueExactly) {
    // Each rank runs serially and writes its own files
    const std::string dir = (rank_scratch_dir() / "droplet_restart").string();
    const std::string writer = "[[write_data]]\nprefix = \"" + dir + "/restart\"\nformat = \"restart\"\ninterval = 70\n";
    const auto input = [&](uint32_t n_steps, const std::string & init, const std::string & extra) {
        std::string text = thrown(8e-6, n_steps, "deposition_width = 0.02\n") + extra;
        if (!init.empty()) {
            const size_t at = text.find("[initialize]");
            const size_t end = text.find("[numerics]");
            text.replace(at, end - at, "[initialize]\n" + init);
        }
        return text;
    };
    Solver straight;
    straight.set_distributed(false);
    straight.init(parse_toml(input(140, "", "")));
    straight.run();
    const std::vector<particles::Record> expected = straight.get_particles()->gather();
    Solver first;
    first.set_distributed(false);
    first.init(parse_toml(input(70, "", writer)));
    first.run();
    Solver second;
    second.set_distributed(false);
    second.init(parse_toml(input(140, "type = \"restart\"\nfile = \"" + dir + "/restart_000070.restart\"\n", "")));
    second.run();
    const std::vector<particles::Record> records = second.get_particles()->gather();
    ASSERT_EQ(records.size(), expected.size());
    for (size_t k = 0; k < records.size(); k++) {
        EXPECT_EQ(records[k].x, expected[k].x) << "droplet " << k;
        EXPECT_EQ(records[k].v, expected[k].v) << "droplet " << k;
        EXPECT_EQ(records[k].T, expected[k].T) << "droplet " << k;
        EXPECT_EQ(records[k].m, expected[k].m) << "droplet " << k;
    }
    EXPECT_EQ(second.get_particles()->evaporated(), straight.get_particles()->evaporated());
    EXPECT_GT(straight.get_particles()->evaporated(), first.get_particles()->evaporated());
    EXPECT_EQ(totals(second), totals(straight));
}

TEST(DropletTest, UnsupportedInputIsRefusedNamingTheKey) {
    const auto expect_error = [](const std::string & input, const std::string & key) {
        Solver solver;
        solver.set_distributed(false);
        try {
            solver.init(parse_toml(input));
            ADD_FAILURE() << "no error for " << key;
        } catch (const InputError & e) {
            EXPECT_NE(std::string(e.what()).find(key), std::string::npos) << key << ": " << e.what();
        }
    };
    const std::string good = quiescent(50e-6, 300.0, "d2_law", run(1, 1e-4));
    const auto replaced = [&](const std::string & from, const std::string & to) {
        std::string text = good;
        const size_t at = text.find(from);
        EXPECT_NE(at, std::string::npos) << from;
        return text.replace(at, from.size(), to);
    };
    // A single gas has no vapor species
    std::string single = good;
    const auto erase_line = [](std::string & text, const std::string & start) {
        const size_t at = text.find(start);
        text.erase(at, text.find('\n', at) + 1 - at);
    };
    single.replace(single.find("gas = \"mixture\""), 15,
                   "mu = 4e-5\ngamma = 1.4\np_ref = 101325.0\nT_ref = 1000.0\nrho_ref = 0.353");
    erase_line(single, "X = ");
    erase_line(single, "mechanism = ");
    erase_line(single, "transport = ");
    expect_error(single, "particles.type");
    expect_error(replaced("species = \"H2O\"", "species = \"H2O_l\""), "particles.liquid.species");
    expect_error(replaced("evaporation = \"d2_law\"", "evaporation = \"spalding\""), "particles.evaporation");
    expect_error(replaced("temperature = 300", "minimum_diameter = 1.0\ntemperature = 300"), "particles.minimum_diameter");
    expect_error(replaced("temperature = 300", "temperature = 700"), "particles.temperature");
    expect_error(replaced("temperature = 300", "density = 1000.0\ntemperature = 300"), "particles.density");
    expect_error(replaced("specific_heat = 4216.0", "specific_heat = 42160.0"), "particles.liquid.specific_heat");
    expect_error(replaced("critical_temperature = 647.1", "critical_temperature = 350"), "particles.liquid.critical_temperature");
    expect_error(replaced("latent_heat = 2257000.0", "antoine = [1.0, 2.0]\nlatent_heat = 2257000.0"), "particles.liquid.antoine");
    std::string no_liquid = good;
    no_liquid.erase(no_liquid.find("[particles.liquid]"));
    expect_error(no_liquid, "particles.liquid");
}

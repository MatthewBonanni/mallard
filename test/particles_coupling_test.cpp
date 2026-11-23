/**
 * @file particles_coupling_test.cpp
 * @author Matthew Bonanni (mbonanni001@gmail.com)
 * @brief Tests of two-way coupled particles: the exchange rates of a uniform
 *        suspension against their closed forms, conservation of the gas and
 *        particles together, the deposition kernel and the dispersed-phase
 *        fields against the Gaussian, and bitwise independence of the rank count.
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
#include <numbers>
#include <span>
#include <sstream>
#include <string>
#include <vector>

#include "check.h"
#include "comm.h"
#include "diagnostic_code.h"
#include "mpi_compare.h"
#include "particles.h"
#include "scratch.h"
#include "solver.h"
#include "test_fixtures.h"

namespace {

constexpr double MU = 0.01, GAMMA = 1.4, PR = 0.72;
constexpr double PI = std::numbers::pi;

std::string numbers(double x, double y, double z) {
    std::ostringstream s;
    s.precision(17);
    s << "[" << x << ", " << y;
    if (N_DIM == 3) s << ", " << z;
    s << "]";
    return s.str();
}

std::string strings(const std::string & x, const std::string & y, const std::string & z) {
    return "[\"" + x + "\", \"" + y + (N_DIM == 3 ? "\", \"" + z : std::string()) + "\"]";
}

/** @brief Cells per side of the periodic unit box (half as many along z, which is half as long). */
double cell_volume(uint32_t n) { return N_DIM == 3 ? std::pow(1.0 / n, 3) : std::pow(1.0 / n, 2); }

/**
 * @brief A periodic unit box of n cells per side (n / 2 along z, 0.5 long) of viscous gas (mu = MU,
 *        rho = p = T = 1, so R = 1) with velocity u, and solid particles of the given [particles] keys.
 */
std::string box(uint32_t n, const std::string & u, const std::string & run, const std::string & particles,
                const std::string & recon = "type = \"TENO\"\norder = 4\n", const std::string & integrator = "RK4") {
    std::ostringstream s;
    s << "[run]\n" << run << "[mesh]\ntype = \"cartesian\"\nNx = " << n << "\nNy = " << n << "\nLx = 1.0\nLy = 1.0\n";
    if (N_DIM == 3) s << "Nz = " << n / 2 << "\nLz = 0.5\n";
    s << periodic_sides(N_DIM == 3 ? "xyz" : "xy")
      << "[initialize]\ntype = \"analytical\"\nrho = \"1.0\"\nu = " << u << "\np = \"1.0\"\n"
      << "[numerics]\nriemann_solver = \"HLLC\"\ntime_integrator = \"" << integrator << "\"\n"
      << "[numerics.face_reconstruction]\n" << recon
      << "[physics]\ntype = \"navier_stokes\"\nmu = " << MU << "\ngamma = " << GAMMA
      << "\np_ref = 1.0\nT_ref = 1.0\nrho_ref = 1.0\n[output]\ncheck_interval = 1000000\n"
      << "[particles]\ntype = \"solid\"\ncoupling = \"two-way\"\n" << particles;
    return s.str();
}

/** @brief One particle at the center of every cell of box(n). */
std::string every_cell(uint32_t n) {
    const double h = 1.0 / n;
    std::ostringstream s;
    s << "[particles.lattice]\nlo = " << numbers(0.5 * h, 0.5 * h, 0.5 * h) << "\nhi = "
      << numbers(1.0 - 0.5 * h, 1.0 - 0.5 * h, 0.5 - 0.5 * h) << "\nn = "
      << (N_DIM == 3 ? "[" + std::to_string(n) + ", " + std::to_string(n) + ", " + std::to_string(n / 2) + "]"
                     : "[" + std::to_string(n) + ", " + std::to_string(n) + "]")
      << "\n";
    return s.str();
}

/** @brief The diameter of particles of density rho_p whose Stokes relaxation time in the gas of box() is tau. */
double diameter_for(double tau, double density) { return std::sqrt(18.0 * MU * tau / density); }

double sphere_mass(double density, double d) { return density * PI / 6.0 * d * d * d; }

/** @brief [particles] keys of particles of the given density, Stokes relaxation time and weight. */
std::string sphere(double density, double tau, double weight, const std::string & extra = "") {
    std::ostringstream s;
    s.precision(17);
    s << "density = " << density << "\ndiameter = " << diameter_for(tau, density) << "\ndrag = \"stokes\"\nweight = "
      << weight << "\n" << extra;
    return s.str();
}

std::string run(uint32_t n_steps, double dt) {
    std::ostringstream s;
    s.precision(17);
    s << "n_steps = " << n_steps << "\ndt = " << dt << "\n";
    return s.str();
}

/** @brief The momentum and total energy of the gas in the owned cells of a rank, (rho u, rho E) times volume. */
std::array<double, N_DIM + 1> gas_totals(Solver & solver) {
    solver.copy_device_to_host();
    const Mesh & mesh = *solver.get_mesh();
    std::array<double, N_DIM + 1> total{};
    for (uint32_t c = 0; c < mesh.n_owned(); c++) {
        const double V = static_cast<double>(mesh.h_cell_measure(c));
        for (uint32_t i = 0; i <= N_DIM; i++) total[i] += V * static_cast<double>(solver.h_conservatives(c, 1 + i));
    }
    return total;
}

/** @brief The momentum and total energy of the particles, weight m (v, c_p T_p + |v|^2 / 2) (collective). */
std::array<double, N_DIM + 1> particle_totals(const Solver & solver, double mass, double c_p) {
    std::array<double, N_DIM + 1> total{};
    for (const auto & r : solver.get_particles()->gather()) {
        double v2 = 0.0;
        for (size_t d = 0; d < N_DIM; d++) {
            total[d] += mass * r.v[d];
            v2 += r.v[d] * r.v[d];
        }
        total[N_DIM] += mass * ((c_p > 0.0 ? c_p * r.T : 0.0) + 0.5 * v2);
    }
    return total;
}

} // namespace

TEST(ParticleCouplingTest, SlipOfAUniformSuspensionDecaysAtItsRelaxationRate) {
    // One particle per cell keeps the gas uniform. With mass loading phi = w m / (rho V), the slip w = u - v
    // decays as e^{-(1 + phi) t / tau} and the momentum rho u + phi rho v is conserved
    const uint32_t n = 8;
    const double tau = 0.1, density = 1000.0, phi = 1.5, t = 0.2;
    const double weight = phi * cell_volume(n) / sphere_mass(density, diameter_for(tau, density));
    const double decay = std::exp(-(1.0 + phi) * t / tau);
    const double u = phi / (1.0 + phi) * (1.0 - decay), v = u + decay;
    for (const char * width : {"0.0", "0.3"}) {
        Solver solver;
        solver.set_distributed(false);
        solver.init(parse_toml(box(n, strings("0.0", "0.0", "0.0"), run(40, t / 40),
                                   sphere(density, tau, weight, "velocity = " + numbers(1.0, 0.0, 0.0) +
                                                                    "\ndeposition_width = " + width + "\n") +
                                       every_cell(n))));
        solver.run();
        solver.copy_device_to_host();
        for (uint32_t c = 0; c < solver.get_mesh()->n_owned(); c++) {
            const double rho = static_cast<double>(solver.h_conservatives(c, 0));
            EXPECT_NEAR(static_cast<double>(solver.h_conservatives(c, 1)) / rho, u, roundoff(1e-7)) << "width " << width;
            EXPECT_NEAR(static_cast<double>(solver.h_conservatives(c, 2)) / rho, 0.0, roundoff(1e-14));
        }
        for (const auto & r : solver.get_particles()->gather()) {
            EXPECT_NEAR(r.v[0], v, roundoff(1e-7)) << "width " << width;
            EXPECT_NEAR(r.v[1], 0.0, roundoff(1e-14));
        }
    }
}

TEST(ParticleCouplingTest, TemperatureOfAUniformSuspensionEquilibratesAtItsRelaxationRate) {
    // At rest, with heat capacity ratio psi = w m c_s / (rho c_v V), T - T_p decays as e^{-(1 + psi) t / tau_T},
    // tau_T = rho_p c_s d^2 / (6 k Nu), Nu = 2, k = mu c_p / Pr, and rho c_v T V + w m c_s T_p is conserved
    const uint32_t n = 8;
    const double tau = 0.1, density = 1000.0, c_s = 2.0, psi = 0.8, t = 0.1, Tp0 = 2.0;
    const double d = diameter_for(tau, density), m = sphere_mass(density, d);
    const double c_v = 1.0 / (GAMMA - 1.0), c_p = GAMMA * c_v, k = MU * c_p / PR;
    const double weight = psi * c_v * cell_volume(n) / (m * c_s);
    const double tau_T = density * c_s * d * d / (6.0 * k * 2.0);
    const double decay = std::exp(-(1.0 + psi) * t / tau_T);
    const double T_mean = (1.0 + psi * Tp0) / (1.0 + psi);
    const double T = T_mean + (1.0 - T_mean) * decay, Tp = T_mean + (Tp0 - T_mean) * decay;
    std::ostringstream keys;
    keys.precision(17);
    keys << "specific_heat = " << c_s << "\ntemperature = " << Tp0 << "\n";
    Solver solver;
    solver.set_distributed(false);
    solver.init(parse_toml(box(n, strings("0.0", "0.0", "0.0"), run(60, t / 60), sphere(density, tau, weight, keys.str()) +
                                                                                     every_cell(n))));
    solver.run();
    ASSERT_GT(t / tau_T, 1.0);  // the exchange is not negligible
    // Ranz-Marshall's sqrt(Re) turns the reconstruction's round-off slip into changes of Nu of order sqrt(eps)
    const double tolerance = 3e-5 * std::sqrt(roundoff(1.0));
    solver.copy_device_to_host();
    for (uint32_t c = 0; c < solver.get_mesh()->n_owned(); c++) {
        const double rho = static_cast<double>(solver.h_conservatives(c, 0));
        EXPECT_NEAR(static_cast<double>(solver.h_conservatives(c, N_DIM + 1)) / (rho * c_v), T, tolerance);
    }
    for (const auto & r : solver.get_particles()->gather()) EXPECT_NEAR(r.T, Tp, tolerance);
}

namespace {

/**
 * @brief Heated particles thrown through a Taylor-Green flow of the periodic box, faster than it, so that
 *        drag, heat transfer and the deposition all act, with relaxation time tau (explicit below the step,
 *        exponential above it) and the given extra [particles] keys.
 */
std::string thrown(double tau, const std::string & extra, uint32_t n_steps = 30) {
    std::ostringstream keys;
    keys.precision(17);
    keys << "specific_heat = 3.0\ntemperature = 1.5\nvelocity = " << numbers(0.9, -0.4, 0.2) << "\n" << extra
         << "[particles.lattice]\nlo = " << numbers(0.07, 0.11, 0.13) << "\nhi = " << numbers(0.93, 0.87, 0.41)
         << "\nn = " << (N_DIM == 3 ? "[5, 4, 2]" : "[5, 4]") << "\n";
    return box(12, strings("0.5 * sin(2 * pi * x) * cos(2 * pi * y)", "-0.5 * cos(2 * pi * x) * sin(2 * pi * y)", "0.0"),
               run(n_steps, 0.01), sphere(1000.0, tau, 200.0, keys.str()), "type = \"TENO\"\norder = 4\n", "SSPRK3");
}

/** @brief Gas and particle totals, (momentum, energy), of a run of input on this rank count (collective). */
std::array<double, N_DIM + 1> totals(Solver & solver, double tau) {
    const double m = 200.0 * sphere_mass(1000.0, diameter_for(tau, 1000.0));
    std::array<double, N_DIM + 1> gas = gas_totals(solver);
    if (solver.is_distributed()) comm::allreduce(std::span<double>(gas), comm::Op::SUM);
    const std::array<double, N_DIM + 1> carried = particle_totals(solver, m, 3.0);
    for (size_t i = 0; i <= N_DIM; i++) gas[i] += carried[i];
    return gas;
}

} // namespace

TEST(ParticleCouplingTest, GasAndParticlesConserveMomentumAndEnergyToRoundOff) {
    SKIP_IN_SINGLE_PRECISION("compares totals at double-precision round-off");
    for (const double tau : {0.02, 0.005}) {
        for (const char * width : {"0.0", "0.15"}) {
            const std::string input = thrown(tau, std::string("deposition_width = ") + width + "\n");
            Solver start;
            start.init(parse_toml(input));
            const auto before = totals(start, tau);
            Solver solver;
            solver.init(parse_toml(input));
            solver.run();
            const auto after = totals(solver, tau);
            const double scale = std::abs(before[N_DIM]);
            for (size_t i = 0; i <= N_DIM; i++) {
                EXPECT_NEAR(after[i], before[i], 1e-13 * scale) << "tau " << tau << ", width " << width << ", component " << i;
            }
            // The particles did exchange momentum and energy with the gas
            const double m = 200.0 * sphere_mass(1000.0, diameter_for(tau, 1000.0));
            const auto carried_before = particle_totals(start, m, 3.0), carried_after = particle_totals(solver, m, 3.0);
            EXPECT_GT(std::abs(carried_after[0] - carried_before[0]), 1e-4 * scale);
            EXPECT_GT(std::abs(carried_after[N_DIM] - carried_before[N_DIM]), 1e-3 * scale);
        }
    }
}

TEST(ParticleCouplingTest, GravityOnTheParticlesAloneChangesTheTotalMomentumByTheirWeight) {
    SKIP_IN_SINGLE_PRECISION("compares totals at double-precision round-off");
    // Gravity of [particles] acts on the particles only: the gas receives the drag, never the weight
    const double tau = 0.02, g = 3.0, t = 0.3;
    const std::string input = thrown(tau, "gravity = " + numbers(0.0, -g, 0.0) + "\n");
    Solver start;
    start.init(parse_toml(input));
    const auto before = totals(start, tau);
    Solver solver;
    solver.init(parse_toml(input));
    solver.run();
    const auto after = totals(solver, tau);
    const double m = 200.0 * sphere_mass(1000.0, diameter_for(tau, 1000.0));
    const uint64_t count = solver.get_particles()->count();
    EXPECT_NEAR(after[0], before[0], 1e-13 * std::abs(before[N_DIM]));
    EXPECT_NEAR(after[1], before[1] - static_cast<double>(count) * m * g * t, 1e-13 * std::abs(before[N_DIM]));
}

TEST(ParticleCouplingTest, CoupledRunsAreBitwiseIndependentOfRanksAndStepGraphs) {
    // The gas the particles act on, and the particles, match the serial run bit for bit
    const std::string input = thrown(0.01, "deposition_width = 0.15\n", 10);
    Solver serial;
    serial.set_distributed(false);
    serial.init(parse_toml(input));
    serial.run();
    const std::vector<particles::Record> expected = serial.get_particles()->gather();
    auto check = [&](const Solver & s) {
        const std::vector<particles::Record> records = s.get_particles()->gather();
        ASSERT_EQ(records.size(), expected.size());
        for (size_t k = 0; k < records.size(); k++) {
            EXPECT_EQ(records[k].x, expected[k].x) << "particle " << k;
            EXPECT_EQ(records[k].v, expected[k].v) << "particle " << k;
            EXPECT_EQ(records[k].T, expected[k].T) << "particle " << k;
        }
    };
    if (comm::size() == 1) {
        expect_matches_serial_with_and_without_step_graphs(input, check);
    } else {
        expect_matches_serial(input, check);
    }
}

TEST(ParticleCouplingTest, RestartedCoupledRunContinuesExactly) {
    // The particles' energies continue bit for bit (their temperatures alone would round them), and so does the gas
    const std::string dir = (scratch_dir() / "mallard_particle_coupling_restart").string();
    const std::string writer = "[[write_data]]\nprefix = \"" + dir + "/restart\"\nformat = \"restart\"\ninterval = 10\n";
    const auto input = [&](uint32_t n_steps, const std::string & init, const std::string & extra) {
        std::string text = thrown(0.01, "deposition_width = 0.15\n", n_steps) + extra;
        // A specific heat whose temperatures do not round-trip the energies exactly
        text.replace(text.find("specific_heat = 3.0"), 19, "specific_heat = 1004.5");
        if (!init.empty()) {
            const size_t at = text.find("[initialize]");
            const size_t end = text.find("[numerics]");
            text.replace(at, end - at, "[initialize]\n" + init);
        }
        return text;
    };
    Solver straight;
    straight.init(parse_toml(input(20, "", "")));
    straight.run();
    straight.copy_device_to_host();
    const std::vector<particles::Record> expected = straight.get_particles()->gather();
    Solver first;
    first.init(parse_toml(input(10, "", writer)));
    first.run();
    comm::barrier();
    Solver second;
    second.init(parse_toml(input(20, "type = \"restart\"\nfile = \"" + dir + "/restart_000010.restart\"\n", "")));
    second.run();
    second.copy_device_to_host();
    const std::vector<particles::Record> records = second.get_particles()->gather();
    ASSERT_EQ(records.size(), expected.size());
    for (size_t k = 0; k < records.size(); k++) {
        EXPECT_EQ(records[k].x, expected[k].x) << "particle " << k;
        EXPECT_EQ(records[k].v, expected[k].v) << "particle " << k;
        EXPECT_EQ(records[k].T, expected[k].T) << "particle " << k;
    }
    const std::array<double, N_DIM + 1> gas = gas_totals(straight), restarted = gas_totals(second);
    EXPECT_EQ(gas, restarted);
}

TEST(ParticleCouplingTest, CheckWarnsWhereThePointParticleModelBreaks) {
    // Particles of a third of their cell, and a suspension of 3% by volume
    const uint32_t n = 8;
    const double d = 0.3 / n, density = 1000.0;
    const double weight = 0.03 * cell_volume(n) / (PI / 6.0 * d * d * d);
    std::ostringstream keys;
    keys.precision(17);
    keys << "density = " << density << "\ndiameter = " << d << "\nweight = " << weight << "\n" << every_cell(n);
    const auto codes = [&](const std::string & particle_keys) {
        Solver solver;
        const CheckReport r = solver.check(parse_toml(box(n, strings("0.0", "0.0", "0.0"), run(1, 0.01), particle_keys)), {});
        std::vector<std::string> out;
        for (const auto & diagnostic : r.diagnostics) out.push_back(diagnostic.code);
        return out;
    };
    const std::vector<std::string> found = codes(keys.str());
    EXPECT_NE(std::find(found.begin(), found.end(), diagnostic_code::PARTICLE_SIZE), found.end());
    EXPECT_NE(std::find(found.begin(), found.end(), diagnostic_code::PARTICLE_VOLUME_FRACTION), found.end());
    // A tenth of the weight and a quarter of the diameter: neither
    std::ostringstream fine;
    fine.precision(17);
    fine << "density = " << density << "\ndiameter = " << d / 4.0 << "\nweight = " << weight / 10.0 << "\n" << every_cell(n);
    const std::vector<std::string> none = codes(fine.str());
    EXPECT_EQ(std::find(none.begin(), none.end(), diagnostic_code::PARTICLE_SIZE), none.end());
    EXPECT_EQ(std::find(none.begin(), none.end(), diagnostic_code::PARTICLE_VOLUME_FRACTION), none.end());
}

namespace {

/**
 * @brief The L1 error of the volume fraction of one particle at p, spread over a Gaussian of width sigma, on
 *        box(n), against w V_p G_sigma(x - p) at the cell centers.
 */
double spread_error(uint32_t n, double sigma) {
    const double weight = 1e3, density = 1000.0, tau = 0.1;
    const std::array<double, 3> p = {0.53, 0.46, 0.27};
    std::ostringstream keys;
    keys.precision(17);
    keys << "deposition_width = " << sigma << "\npoints = [" << numbers(p[0], p[1], p[2]) << "]\n";
    Solver solver;
    solver.set_distributed(false);
    solver.init(parse_toml(box(n, strings("0.0", "0.0", "0.0"), run(0, 0.01), sphere(density, tau, weight, keys.str()),
                               "type = \"FO\"\n")));
    solver.update_particle_fields();
    const auto fields = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), solver.get_particle_fields());
    const Mesh & mesh = *solver.get_mesh();
    const double volume = weight * PI / 6.0 * std::pow(diameter_for(tau, density), 3);
    double error = 0.0, total = 0.0;
    for (uint32_t c = 0; c < mesh.n_owned(); c++) {
        double r2 = 0.0;
        for (size_t d = 0; d < N_DIM; d++) {
            const double length = d == 2 ? 0.5 : 1.0;
            double dx = static_cast<double>(mesh.h_cell_coords(c, d)) - p[d];
            dx -= length * std::round(dx / length);
            r2 += dx * dx;
        }
        const double exact = volume * std::exp(-0.5 * r2 / (sigma * sigma)) / std::pow(2.0 * PI * sigma * sigma, 0.5 * N_DIM);
        const double V = static_cast<double>(mesh.h_cell_measure(c));
        error += std::abs(static_cast<double>(fields(c, 0)) - exact) * V;
        total += static_cast<double>(fields(c, 0)) * V;
    }
    EXPECT_NEAR(total, volume, roundoff(1e-12) * volume) << "n = " << n;
    return error / volume;
}

} // namespace

TEST(ParticleCouplingTest, DepositionConvergesToTheGaussianUnderRefinement) {
    // At a fixed width in physical units, the deposited field converges to the Gaussian about the particle,
    // at second order once the cell size resolves the width (sigma / h = 1.3, 2.6, 5.1), and keeps its integral
    const double sigma = 0.08;
    const double coarse = spread_error(16, sigma), medium = spread_error(32, sigma), fine = spread_error(64, sigma);
    EXPECT_GT(std::log2(coarse / medium), 1.0);
    EXPECT_GT(std::log2(medium / fine), 1.8);
    EXPECT_LT(fine, 0.01);
}

namespace {

struct Moments {
    double total = 0.0;
    std::array<double, 3> mean{};
    std::array<std::array<double, 3>, 3> covariance{};
};

/** @brief Integral, mean and covariance of the volume fraction of one particle at p spread over sigma on box(n). */
Moments deposit_moments(uint32_t n, double sigma, const std::array<double, 3> & p) {
    const double weight = 1e3, density = 1000.0, tau = 0.1;
    std::ostringstream keys;
    keys.precision(17);
    keys << "deposition_width = " << sigma << "\npoints = [" << numbers(p[0], p[1], p[2]) << "]\n";
    Solver solver;
    solver.set_distributed(false);
    solver.init(parse_toml(box(n, strings("0.0", "0.0", "0.0"), run(0, 0.01), sphere(density, tau, weight, keys.str()),
                               "type = \"FO\"\n")));
    solver.update_particle_fields();
    const auto fields = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), solver.get_particle_fields());
    const Mesh & mesh = *solver.get_mesh();
    Moments m;
    std::vector<std::array<double, 3>> offsets(mesh.n_owned());
    std::vector<double> q(mesh.n_owned());
    for (uint32_t c = 0; c < mesh.n_owned(); c++) {
        q[c] = static_cast<double>(fields(c, 0)) * static_cast<double>(mesh.h_cell_measure(c));
        for (size_t d = 0; d < N_DIM; d++) {
            const double length = d == 2 ? 0.5 : 1.0;
            double dx = static_cast<double>(mesh.h_cell_coords(c, d)) - p[d];
            offsets[c][d] = dx - length * std::round(dx / length);
        }
        m.total += q[c];
        for (size_t a = 0; a < N_DIM; a++) m.mean[a] += q[c] * offsets[c][a];
    }
    for (size_t a = 0; a < N_DIM; a++) m.mean[a] /= m.total;
    for (uint32_t c = 0; c < mesh.n_owned(); c++) {
        for (size_t a = 0; a < N_DIM; a++) {
            for (size_t b = 0; b < N_DIM; b++) {
                m.covariance[a][b] += q[c] * (offsets[c][a] - m.mean[a]) * (offsets[c][b] - m.mean[b]) / m.total;
            }
        }
    }
    return m;
}

} // namespace

TEST(ParticleCouplingTest, DepositedSpreadIsIndependentOfTheParticlesPlaceInItsCell) {
    // The deposit keeps the particle's integral, is centered on it and has the Gaussian's covariance sigma^2 I,
    // whether the particle sits at its cell's center or near a corner: the moment corrections give the
    // pre-spread deposit the particle's first and second moments, and the spread adds sigma^2 I exactly
    const uint32_t n = 32;
    const double h = 1.0 / n, sigma = 0.045;
    const double volume = 1e3 * PI / 6.0 * std::pow(diameter_for(0.1, 1000.0), 3);
    // The Gaussian's tail beyond the nearest periodic half-length, 0.25, changes its moments by under 1e-6
    const double tolerance = std::max(1e-6, roundoff(1e-12)) * sigma * sigma;
    for (const std::array<double, 3> & offset : {std::array<double, 3>{0.0, 0.0, 0.0}, {0.45, 0.45, 0.45},
                                                 {0.45, -0.3, 0.2}, {-0.4, 0.1, -0.45}}) {
        const std::array<double, 3> p = {(12.5 + offset[0]) * h, (14.5 + offset[1]) * h, (5.5 + offset[2]) * h};
        const Moments m = deposit_moments(n, sigma, p);
        EXPECT_NEAR(m.total, volume, roundoff(1e-12) * volume);
        for (size_t a = 0; a < N_DIM; a++) {
            EXPECT_NEAR(m.mean[a], 0.0, roundoff(1e-12)) << "offset " << offset[0] << ", " << offset[1];
            for (size_t b = 0; b < N_DIM; b++) {
                EXPECT_NEAR(m.covariance[a][b], a == b ? sigma * sigma : 0.0, tolerance)
                    << "offset " << offset[0] << ", " << offset[1] << ", component " << a << b;
            }
        }
    }
}

TEST(ParticleCouplingTest, DispersedPhaseFieldsOfAUniformSuspension) {
    // One particle per cell: volume fraction w V_p / V, number density w / V, mean velocity the particles',
    // and the source the opposite of their drag
    const uint32_t n = 8;
    const double tau = 0.1, density = 1000.0, weight = 50.0;
    const double d = diameter_for(tau, density), m = sphere_mass(density, d), V = cell_volume(n);
    const std::string input = box(n, strings("0.2", "0.0", "0.0"), run(0, 0.01),
                                  sphere(density, tau, weight, "velocity = " + numbers(1.0, 0.5, 0.0) + "\n") + every_cell(n));
    Solver solver;
    solver.set_distributed(false);
    solver.init(parse_toml(input));
    solver.update_particle_fields();
    const auto f = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), solver.get_particle_fields());
    const std::vector<std::string> names = Particles::field_names(parse_toml(input));
    ASSERT_EQ(names.size(), f.extent(1));
    ASSERT_EQ(names[0], "PARTICLE_VOLUME_FRACTION");
    ASSERT_EQ(names[N_DIM + 2], "PARTICLE_SOURCE_RHO");
    for (uint32_t c = 0; c < solver.get_mesh()->n_owned(); c++) {
        EXPECT_NEAR(static_cast<double>(f(c, 0)), weight * m / density / V, roundoff(1e-13));
        EXPECT_NEAR(static_cast<double>(f(c, 1)), weight / V, roundoff(1e-12) * weight / V);
        EXPECT_NEAR(static_cast<double>(f(c, 2)), 1.0, roundoff(1e-14));
        EXPECT_NEAR(static_cast<double>(f(c, 3)), 0.5, roundoff(1e-14));
        // Drag per volume on the gas: w m (v - u) / tau / V
        EXPECT_NEAR(static_cast<double>(f(c, N_DIM + 3)), weight * m * (1.0 - 0.2) / tau / V, roundoff(1e-10));
        EXPECT_NEAR(static_cast<double>(f(c, N_DIM + 4)), weight * m * 0.5 / tau / V, roundoff(1e-10));
    }
}

TEST(ParticleCouplingTest, TwoWayCouplingNeedsSolidParticles) {
    std::string input = box(8, strings("0.0", "0.0", "0.0"), run(1, 0.01), "points = [" + numbers(0.5, 0.5, 0.25) + "]\n");
    input.replace(input.find("type = \"solid\""), 14, "type = \"tracer\"");
    Solver solver;
    try {
        solver.init(parse_toml(input));
        FAIL() << "no error";
    } catch (const InputError & e) {
        EXPECT_NE(std::string(e.what()).find("particles.coupling"), std::string::npos) << e.what();
    }
}

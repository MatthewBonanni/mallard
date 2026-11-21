/**
 * @file particles_test.cpp
 * @author Matthew Bonanni (mbonanni001@gmail.com)
 * @brief Tests of the tracer particles: exact pathlines, their order in space
 *        and time, walls and outlets, migration between ranks and restarts.
 * @version 0.1
 * @date 2026-11-15
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 *
 */

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <fstream>
#include <limits>
#include <numbers>
#include <sstream>
#include <string>
#include <vector>

#include "comm.h"
#include "mpi_compare.h"
#include "particles.h"
#include "scratch.h"
#include "solver.h"
#include "test_fixtures.h"

namespace {

const std::string PHYSICS = "[physics]\ntype = \"euler\"\ngamma = 1.4\np_ref = 1.0\nT_ref = 1.0\nrho_ref = 1.0\n";
const std::string TENO4 = "type = \"TENO\"\norder = 4\n";

std::string vec(const std::string & x, const std::string & y, const std::string & z) {
    return "[\"" + x + "\", \"" + y + (N_DIM == 3 ? "\", \"" + z : std::string()) + "\"]";
}

std::string numbers(double x, double y, double z) {
    std::ostringstream s;
    s.precision(17);
    s << "[" << x << ", " << y;
    if (N_DIM == 3) s << ", " << z;
    s << "]";
    return s.str();
}

/**
 * @brief A unit box of n cells per side (n / 2 along z, which is periodic and
 *        half as long), with the given velocity, sides joined periodically,
 *        boundaries, run table and [particles] keys.
 */
std::string box(uint32_t n, const std::string & u, const std::string & periodic, const std::string & boundaries,
                const std::string & run, const std::string & particles, const std::string & recon = TENO4,
                const std::string & integrator = "RK4") {
    std::ostringstream s;
    s << "[run]\n" << run << "[mesh]\ntype = \"cartesian\"\nNx = " << n << "\nNy = " << n << "\nLx = 1.0\nLy = 1.0\n";
    if (N_DIM == 3) s << "Nz = " << n / 2 << "\nLz = 0.5\n";
    s << periodic_sides(N_DIM == 3 ? periodic + "z" : periodic) << boundaries
      << "[initialize]\ntype = \"analytical\"\nrho = \"1.0\"\nu = " << u << "\np = \"1.0\"\n"
      << "[numerics]\nriemann_solver = \"HLLC\"\ntime_integrator = \"" << integrator << "\"\n"
      << "[numerics.face_reconstruction]\n" << recon << PHYSICS << "[output]\ncheck_interval = 1000000\n"
      << "[particles]\ntype = \"tracer\"\n" << particles;
    return s.str();
}

/** @brief A lattice of 4 particles per direction (one along z, at its middle) inside [lo, hi]. */
std::string lattice(double lo, double hi) {
    return "[particles.lattice]\nlo = " + numbers(lo, lo, 0.25) + "\nhi = " + numbers(hi, hi, 0.25) + "\nn = " +
           (N_DIM == 3 ? "[4, 4, 1]" : "[4, 4]") + "\n";
}

/** @brief The seeds of lattice(lo, hi), in id order. */
std::vector<std::array<double, N_DIM>> lattice_points(double lo, double hi) {
    std::vector<std::array<double, N_DIM>> points;
    for (int j = 0; j < 4; j++) {
        for (int i = 0; i < 4; i++) {
            std::array<double, N_DIM> x{};
            x[0] = lo + (hi - lo) * i / 3.0;
            x[1] = lo + (hi - lo) * j / 3.0;
            if constexpr (N_DIM == 3) x[2] = 0.25;
            points.push_back(x);
        }
    }
    return points;
}

double wrap(double x, double length) { return x - length * std::floor(x / length); }

/** @brief Every particle at its exact position, wrapped into the periodic box. */
void expect_pathlines(const Particles & particles, const std::vector<std::array<double, N_DIM>> & exact, double tol) {
    const std::vector<particles::Record> records = particles.gather();
    ASSERT_EQ(records.size(), exact.size());
    for (size_t k = 0; k < records.size(); k++) {
        ASSERT_EQ(records[k].id, k);
        for (size_t d = 0; d < N_DIM; d++) {
            EXPECT_NEAR(records[k].x[d], exact[k][d], tol) << "particle " << k << ", direction " << d;
        }
    }
}

} // namespace

TEST(ParticlesTest, UniformFlowCarriesTracersExactlyAcrossPeriodicSidesAndRanks) {
    // Each particle crosses a periodic side, and on several ranks a partition, at least once
    const double u[3] = {0.3, -0.2, 0.1};
    const double t = 0.6;
    const std::string input = box(12, vec("0.3", "-0.2", "0.1"), "xy", "", "n_steps = 60\ndt = 0.01\n",
                                  lattice(0.07, 0.93));
    std::vector<std::array<double, N_DIM>> exact = lattice_points(0.07, 0.93);
    for (auto & x : exact) {
        for (size_t d = 0; d < N_DIM; d++) x[d] = wrap(x[d] + u[d] * t, d == 2 ? 0.5 : 1.0);
    }
    auto check = [&](const Solver & s) {
        const Particles & p = *s.get_particles();
        EXPECT_EQ(p.count(), exact.size());
        EXPECT_EQ(p.removed(), (std::array<uint64_t, 3>{0, 0, 0}));
        expect_pathlines(p, exact, roundoff(1e-12));
    };
    if (comm::size() == 1) {
        expect_matches_serial_with_and_without_step_graphs(input, check);
    } else {
        // Particles migrate through the host: steps are not graphed
        expect_matches_serial(input, check);
    }
}

namespace {

/**
 * @brief A compact vortex at the middle of the periodic unit box, an exact
 *        steady solution of the Euler equations: the angular velocity
 *        omega(r) = OMEGA (1 - r^2 / R^2)^4 inside r = R and the flow at rest
 *        outside, with the pressure that balances it. Particles move on circles
 *        at omega of their radius. In 3D the box is VORTEX_NZ cells deep, at rest
 *        along z.
 */
constexpr double OMEGA = 4.0, R_VORTEX = 0.45;
constexpr uint32_t VORTEX_NZ = 6;

/** @brief Seeds at several radii, so the particles turn at different rates. */
std::vector<std::array<double, N_DIM>> vortex_seeds(uint32_t n) {
    std::vector<std::array<double, N_DIM>> seeds;
    for (int k = 0; k < 6; k++) {
        const double r = 0.03 + 0.04 * k, a = 0.3 + 0.7 * k;
        std::array<double, N_DIM> x{};
        x[0] = 0.5 + r * std::cos(a);
        x[1] = 0.5 + r * std::sin(a);
        if constexpr (N_DIM == 3) x[2] = 0.5 * VORTEX_NZ / n;
        seeds.push_back(x);
    }
    return seeds;
}

std::string vortex(uint32_t n, double dt, uint32_t n_steps, const std::string & recon, const std::string & integrator,
                   const std::string & init = "", const std::string & extra = "", bool with_particles = true) {
    std::ostringstream s;
    s.precision(17);
    const std::string f = "max(0, 1 - ((x - 0.5)^2 + (y - 0.5)^2) / " + std::to_string(R_VORTEX * R_VORTEX) + ")";
    s << "[run]\nn_steps = " << n_steps << "\ndt = " << dt << "\n"
      << "[mesh]\ntype = \"cartesian\"\nNx = " << n << "\nNy = " << n << "\nLx = 1.0\nLy = 1.0\n";
    if (N_DIM == 3) s << "Nz = " << VORTEX_NZ << "\nLz = " << double(VORTEX_NZ) / n << "\n";
    s << periodic_sides(N_DIM == 3 ? "xyz" : "xy") << "[initialize]\n";
    if (init.empty()) {
        s << "type = \"analytical\"\nrho = \"1.0\"\nu = "
          << vec("-" + std::to_string(OMEGA) + " * (y - 0.5) * " + f + "^4",
                 std::to_string(OMEGA) + " * (x - 0.5) * " + f + "^4", "0.0")
          << "\np = \"1.0 - " << OMEGA * OMEGA * R_VORTEX * R_VORTEX / 18.0 << " * " << f << "^9\"\n";
    } else {
        s << init;
    }
    s << "[numerics]\nriemann_solver = \"HLLC\"\ntime_integrator = \"" << integrator << "\"\n"
      << "[numerics.face_reconstruction]\n" << recon << PHYSICS << "[output]\ncheck_interval = 1000000\n" << extra;
    if (with_particles) {
        s << "[particles]\ntype = \"tracer\"\npoints = [";
        const auto seeds = vortex_seeds(n);
        for (size_t k = 0; k < seeds.size(); k++) {
            s << (k ? ", " : "") << numbers(seeds[k][0], seeds[k][1], N_DIM == 3 ? seeds[k][2] : 0.0);
        }
        s << "]\n";
    }
    return s.str();
}

/** @brief The largest distance of the particles of a vortex run from their exact positions at time t. */
double pathline_error(const Particles & particles, uint32_t n, double t) {
    const auto seeds = vortex_seeds(n);
    const std::vector<particles::Record> records = particles.gather();
    EXPECT_EQ(records.size(), seeds.size());
    double error = 0.0;
    for (const auto & r : records) {
        const double rx = seeds[r.id][0] - 0.5, ry = seeds[r.id][1] - 0.5;
        const double q = 1.0 - (rx * rx + ry * ry) / (R_VORTEX * R_VORTEX);
        const double angle = OMEGA * q * q * q * q * t;
        const double ex = 0.5 + std::cos(angle) * rx - std::sin(angle) * ry;
        const double ey = 0.5 + std::sin(angle) * rx + std::cos(angle) * ry;
        double e2 = (r.x[0] - ex) * (r.x[0] - ex) + (r.x[1] - ey) * (r.x[1] - ey);
        if constexpr (N_DIM == 3) e2 += (r.x[2] - seeds[r.id][2]) * (r.x[2] - seeds[r.id][2]);
        error = std::max(error, std::sqrt(e2));
    }
    return error;
}

double vortex_error(uint32_t n, double dt, double t, const std::string & recon, const std::string & integrator) {
    Solver solver;
    solver.set_distributed(false);
    solver.init(parse_toml(vortex(n, dt, static_cast<uint32_t>(std::lround(t / dt)), recon, integrator)));
    solver.run();
    return pathline_error(*solver.get_particles(), n, t);
}

void expect_same_particles(const Particles & a, const Particles & b) {
    const std::vector<particles::Record> ra = a.gather(), rb = b.gather();
    ASSERT_EQ(ra.size(), rb.size());
    for (size_t k = 0; k < ra.size(); k++) {
        EXPECT_EQ(ra[k].id, rb[k].id);
        for (size_t d = 0; d < N_DIM; d++) {
            EXPECT_EQ(ra[k].x[d], rb[k].x[d]) << "particle " << ra[k].id << ", direction " << d;
            EXPECT_EQ(ra[k].v[d], rb[k].v[d]) << "particle " << ra[k].id << ", direction " << d;
        }
    }
    EXPECT_EQ(a.removed(), b.removed());
}

} // namespace

TEST(ParticlesTest, VortexPathlinesConvergeAtTheReconstructionOrderInSpace) {
    SKIP_IN_SINGLE_PRECISION("the errors on the fine mesh approach single-precision round-off");
    // Steps short enough that the error is that of the reconstruction: the particles read the
    // polynomial of their cell, not its average, at the stage positions
    const double t = 0.5, dt = 0.005;
    const double teno_coarse = vortex_error(16, dt, t, TENO4, "RK4"), teno_fine = vortex_error(32, dt, t, TENO4, "RK4");
    EXPECT_GT(std::log2(teno_coarse / teno_fine), 3.5) << teno_coarse << " -> " << teno_fine;
    const std::string muscl = "type = \"MUSCL\"\n";
    const double muscl_coarse = vortex_error(16, dt, t, muscl, "RK4"), muscl_fine = vortex_error(32, dt, t, muscl, "RK4");
    EXPECT_GT(std::log2(muscl_coarse / muscl_fine), 1.8) << muscl_coarse << " -> " << muscl_fine;
    EXPECT_LT(teno_fine, muscl_fine);
}

namespace {

/**
 * @brief The largest error of the particles of a uniform flow turned by the force rhou = K rho (-u_y, u_x)
 *        after one turn, in the given number of steps. The flow stays uniform, so every reconstruction is
 *        exact and the error is that of time stepping: the velocity turns at the rate K and the particles
 *        move on circles, x(t) = x0 + (u0_x sin Kt + u0_y (cos Kt - 1)) / K, y(t) = y0 + (u0_x (1 - cos Kt) +
 *        u0_y sin Kt) / K. A particle whose stages read the flow of another stage, such as that at the start
 *        of the step, loses the integrator's order.
 */
double turning_error(const std::string & integrator, uint32_t n_steps) {
    const double K = 2.0 * std::numbers::pi, t = 1.0, u0[2] = {0.3, -0.2};
    std::ostringstream run;
    run.precision(17);
    run << "n_steps = " << n_steps << "\ndt = " << t / n_steps << "\n";
    std::string input = box(12, vec("0.3", "-0.2", "0.0"), "xy", "", run.str(), lattice(0.1, 0.9), TENO4, integrator);
    input.insert(input.find("[numerics]"), "[source]\nrhou = " + vec("-2 * pi * rho * u_y", "2 * pi * rho * u_x", "0.0") + "\n");
    Solver solver;
    solver.set_distributed(false);
    solver.init(parse_toml(input));
    solver.run();
    std::vector<std::array<double, N_DIM>> exact = lattice_points(0.1, 0.9);
    const double c = std::cos(K * t), s = std::sin(K * t);
    for (auto & x : exact) {
        x[0] = wrap(x[0] + (u0[0] * s + u0[1] * (c - 1.0)) / K, 1.0);
        x[1] = wrap(x[1] + (u0[0] * (1.0 - c) + u0[1] * s) / K, 1.0);
    }
    const std::vector<particles::Record> records = solver.get_particles()->gather();
    EXPECT_EQ(records.size(), exact.size());
    double error = 0.0;
    for (const auto & r : records) {
        for (size_t d = 0; d < N_DIM; d++) error = std::max(error, std::abs(r.x[d] - exact[r.id][d]));
    }
    return error;
}

} // namespace

TEST(ParticlesTest, PathlinesConvergeAtTheIntegratorOrderInTime) {
    SKIP_IN_SINGLE_PRECISION("RK4's error with 80 steps is near single-precision round-off");
    // At CFL 0.46 and 0.23 (FE: half that)
    const double fe = std::log2(turning_error("FE", 80) / turning_error("FE", 160));
    const double ssprk3 = std::log2(turning_error("SSPRK3", 40) / turning_error("SSPRK3", 80));
    const double rk4 = std::log2(turning_error("RK4", 40) / turning_error("RK4", 80));
    EXPECT_GT(fe, 0.9);
    EXPECT_GT(ssprk3, 2.8);
    EXPECT_GT(rk4, 3.8);
}

TEST(ParticlesTest, TrajectoriesAreBitwiseIndependentOfTheRankCount) {
    // The vortex carries particles across the partitions of 2 and 3 ranks
    const std::string input = vortex(24, 0.01, 40, TENO4, "RK4");
    if (comm::size() == 1) GTEST_SKIP() << "compares distributed runs with serial ones";
    Solver distributed;
    distributed.init(parse_toml(input));
    distributed.run();
    Solver serial;
    serial.set_distributed(false);
    serial.init(parse_toml(input));
    serial.run();
    expect_same_particles(*distributed.get_particles(), *serial.get_particles());
}

TEST(ParticlesTest, RestartedRunContinuesTheTrajectoriesExactly) {
    // On several ranks rank 0 restarts serially: the file restarts on any rank count. Quantities, integrated
    // and extrema, continue too
    const std::string dir = (scratch_dir() / "mallard_particles_restart").string();
    const std::string writer = "[[write_data]]\nprefix = \"" + dir + "/restart\"\nformat = \"restart\"\ninterval = 20\n";
    const std::string q = "[[particles.quantities]]\nname = \"I\"\nintegral = \"U_X * U_Y\"\n"
                          "[[particles.quantities]]\nname = \"top\"\nmax = \"y\"\n";
    Solver straight;
    straight.init(parse_toml(vortex(16, 0.01, 40, TENO4, "SSPRK3") + q));
    straight.run();
    const std::vector<particles::Record> expected = straight.get_particles()->gather();
    const std::array<uint64_t, 3> removed = straight.get_particles()->removed();
    Solver first;
    first.init(parse_toml(vortex(16, 0.01, 20, TENO4, "SSPRK3", "", writer) + q));
    first.run();
    comm::barrier();
    if (!comm::is_root()) return;
    Solver second;
    second.set_distributed(false);
    second.init(parse_toml(vortex(16, 0.01, 40, TENO4, "SSPRK3",
                                  "type = \"restart\"\nfile = \"" + dir + "/restart_000020.restart\"\n") + q));
    second.run();
    const std::vector<particles::Record> records = second.get_particles()->gather();
    ASSERT_EQ(records.size(), expected.size());
    for (size_t k = 0; k < records.size(); k++) {
        EXPECT_EQ(records[k].id, expected[k].id);
        for (size_t d = 0; d < N_DIM; d++) EXPECT_EQ(records[k].x[d], expected[k].x[d]) << "particle " << k;
        EXPECT_EQ(records[k].q, expected[k].q) << "particle " << k;
    }
    EXPECT_EQ(second.get_particles()->removed(), removed);
}

TEST(ParticlesTest, ARestartFileWithoutParticlesSeedsThem) {
    // Tracers added to a developed flow
    const std::string dir = (scratch_dir() / "mallard_particles_restart_seed").string();
    const std::string writer = "[[write_data]]\nprefix = \"" + dir + "/restart\"\nformat = \"restart\"\ninterval = 5\n";
    Solver flow;
    flow.init(parse_toml(vortex(16, 0.01, 5, TENO4, "SSPRK3", "", writer, false)));
    flow.run();
    Solver tracers;
    tracers.init(parse_toml(vortex(16, 0.01, 5, TENO4, "SSPRK3",
                                   "type = \"restart\"\nfile = \"" + dir + "/restart_000005.restart\"\n")));
    const std::vector<particles::Record> records = tracers.get_particles()->gather();
    const auto seeds = vortex_seeds(16);
    ASSERT_EQ(records.size(), seeds.size());
    for (size_t k = 0; k < records.size(); k++) {
        for (size_t d = 0; d < N_DIM; d++) EXPECT_EQ(records[k].x[d], static_cast<double>(static_cast<rtype>(seeds[k][d])));
    }
}

TEST(ParticlesTest, WallsReflectTracers) {
    // A particle that crossed a wall within a step is mirrored in the wall's plane
    const std::string walls = "[[boundaries]]\nname = \"bottom\"\ntype = \"wall_adiabatic\"\n"
                              "[[boundaries]]\nname = \"top\"\ntype = \"symmetry\"\n";
    Solver solver;
    solver.set_distributed(false);
    solver.init(parse_toml(box(12, vec("0.0", "0.0", "0.0"), "x", walls, "n_steps = 1\ndt = 0.01\n",
                               "points = [" + numbers(0.5, 0.02, 0.25) + ", " + numbers(0.3, 0.97, 0.25) + "]\n")));
    Particles & p = *solver.get_particles();
    auto x = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), p.values());
    x(1) = static_cast<rtype>(-0.03);                 // 0.03 below the wall
    x(N_DIM + 1) = static_cast<rtype>(1.04);          // 0.04 above the symmetry plane
    Kokkos::deep_copy(p.values(), x);
    p.relocate();
    const std::vector<particles::Record> records = p.gather();
    ASSERT_EQ(records.size(), 2u);
    EXPECT_NEAR(records[0].x[0], 0.5, roundoff(1e-14));
    EXPECT_NEAR(records[0].x[1], 0.03, roundoff(1e-14));
    EXPECT_NEAR(records[1].x[0], 0.3, roundoff(1e-14));
    EXPECT_NEAR(records[1].x[1], 0.96, roundoff(1e-14));
    EXPECT_EQ(p.removed(), (std::array<uint64_t, 3>{0, 0, 0}));
}

TEST(ParticlesTest, TracersLeaveThroughOutlets) {
    // Supersonic uniform flow at u = 2: in t = 0.25 the particles move by 0.5
    const std::string boundaries = "[[boundaries]]\nname = \"left\"\ntype = \"inlet_state\"\nu = " +
                                   numbers(2.0, 0.0, 0.0) + "\np = 1.0\nT = 1.0\n"
                                   "[[boundaries]]\nname = \"right\"\ntype = \"outlet_extrapolation\"\n";
    Solver solver;
    solver.init(parse_toml(box(12, vec("2.0", "0.0", "0.0"), "y", boundaries, "n_steps = 25\ndt = 0.01\n",
                               lattice(0.1, 0.9))));
    solver.run();
    const Particles & p = *solver.get_particles();
    // Columns at x = 0.1 and 0.3667 stay, those at 0.6333 and 0.9 leave
    EXPECT_EQ(p.count(), 8u);
    EXPECT_EQ(p.removed(), (std::array<uint64_t, 3>{8, 0, 0}));
    for (const auto & r : p.gather()) {
        const double x0 = 0.1 + 0.8 * static_cast<double>(r.id % 4) / 3.0;
        EXPECT_NEAR(r.x[0], x0 + 0.5, roundoff(1e-12)) << "particle " << r.id;
    }
}

TEST(ParticlesTest, SeedsOutsideTheMeshAreErrors) {
    Solver solver;
    const std::string points = "points = [" + numbers(0.5, 0.5, 0.25) + ", " + numbers(1.5, 0.5, 0.25) + "]\n";
    try {
        solver.init(parse_toml(box(8, vec("0.0", "0.0", "0.0"), "xy", "", "n_steps = 1\ndt = 0.01\n", points)));
        FAIL() << "no error";
    } catch (const InputError & e) {
        EXPECT_NE(std::string(e.what()).find("seed point 1"), std::string::npos) << e.what();
    }
}

namespace {

/** @brief The quantity named name of each particle, by id. */
std::vector<double> quantity_of(const Particles & particles, const std::string & name) {
    const auto & names = particles.quantity_names();
    const auto at = std::find(names.begin(), names.end(), name);
    EXPECT_NE(at, names.end()) << name;
    const size_t k = static_cast<size_t>(at - names.begin());
    std::vector<double> out;
    for (const auto & r : particles.gather()) out.push_back(r.q.at(k));
    return out;
}

/** @brief The rows of a CSV file after its comment and header lines, as numbers. */
std::vector<std::vector<double>> csv_rows(const std::string & file) {
    std::ifstream in(file);
    std::vector<std::vector<double>> rows;
    std::string line;
    bool header = true;
    while (std::getline(in, line)) {
        if (line.empty() || line[0] == '#') continue;
        if (header) {
            header = false;
            continue;
        }
        std::vector<double> row;
        std::stringstream s(line);
        std::string cell;
        while (std::getline(s, cell, ',')) row.push_back(std::stod(cell));
        rows.push_back(row);
    }
    return rows;
}

/** @brief The largest error of integral = "x" after one turn of the turning flow of turning_error. */
double turning_integral_error(const std::string & integrator, uint32_t n_steps) {
    const double K = 2.0 * std::numbers::pi, t = 1.0, u0[2] = {0.3, -0.2};
    std::ostringstream run;
    run.precision(17);
    run << "n_steps = " << n_steps << "\ndt = " << t / n_steps << "\n";
    std::string input = box(12, vec("0.3", "-0.2", "0.0"), "xy", "", run.str(),
                            lattice(0.3, 0.7) + "[[particles.quantities]]\nname = \"X\"\nintegral = \"x\"\n", TENO4,
                            integrator);
    input.insert(input.find("[numerics]"), "[source]\nrhou = " + vec("-2 * pi * rho * u_y", "2 * pi * rho * u_x", "0.0") + "\n");
    Solver solver;
    solver.set_distributed(false);
    solver.init(parse_toml(input));
    solver.run();
    const std::vector<double> X = quantity_of(*solver.get_particles(), "X");
    const auto seeds = lattice_points(0.3, 0.7);
    // x(t) = x0 + (u0_x sin Kt + u0_y (cos Kt - 1)) / K, so int_0^t x = x0 t + (u0_x (1 - cos Kt) + u0_y (sin Kt - K t)) / K^2
    double error = 0.0;
    for (size_t k = 0; k < seeds.size(); k++) {
        const double exact = seeds[k][0] * t + (u0[0] * (1.0 - std::cos(K * t)) + u0[1] * (std::sin(K * t) - K * t)) / (K * K);
        error = std::max(error, std::abs(X[k] - exact));
    }
    return error;
}

/** @brief Uniform flow at U_UNIFORM along x through the periodic box, for n_steps of DT_UNIFORM. */
constexpr double U_UNIFORM = 0.3, DT_UNIFORM = 0.01;

std::string uniform(const std::string & particles, uint32_t n_steps, const std::string & extra = "") {
    return box(12, vec("0.3", "0.0", "0.0"), "xy", "", "n_steps = " + std::to_string(n_steps) + "\ndt = 0.01\n",
               particles) + extra;
}

} // namespace

TEST(ParticleQuantitiesTest, IntegralsConvergeAtTheIntegratorOrder) {
    SKIP_IN_SINGLE_PRECISION("RK4's error with 80 steps is near single-precision round-off");
    // The integrand reads each stage's position: integrals take the integrator's stage weights
    const double fe = std::log2(turning_integral_error("FE", 80) / turning_integral_error("FE", 160));
    const double ssprk3 = std::log2(turning_integral_error("SSPRK3", 40) / turning_integral_error("SSPRK3", 80));
    const double rk4 = std::log2(turning_integral_error("RK4", 40) / turning_integral_error("RK4", 80));
    EXPECT_GT(fe, 0.9);
    EXPECT_GT(ssprk3, 2.8);
    EXPECT_GT(rk4, 3.8);
}

TEST(ParticleQuantitiesTest, WindingCountsTheTurnsAboutTheVortexAxis) {
    // Each particle turns at omega(r) about the vortex's axis, t omega(r) / (2 pi) turns, up to the error of its
    // pathline; the winding is its own angle, unwrapped, to the integrator's error. integral = "1" is its age
    const uint32_t n = 32;
    const double dt = 0.005, t = 2.0;
    const std::string axis = N_DIM == 3 ? ", axis = [0.0, 0.0, 2.0]" : "";
    const std::string q = "[[particles.quantities]]\nname = \"turns\"\nwinding = {point = " + numbers(0.5, 0.5, 0.0) +
                          axis + "}\n[[particles.quantities]]\nname = \"age\"\nintegral = \"1\"\n";
    Solver solver;
    solver.set_distributed(false);
    solver.init(parse_toml(vortex(n, dt, static_cast<uint32_t>(std::lround(t / dt)), TENO4, "RK4") + q));
    solver.run();
    const Particles & p = *solver.get_particles();
    const std::vector<double> turns = quantity_of(p, "turns"), age = quantity_of(p, "age");
    const auto seeds = vortex_seeds(n);
    const std::vector<particles::Record> records = p.gather();
    ASSERT_EQ(turns.size(), seeds.size());
    for (size_t k = 0; k < seeds.size(); k++) {
        const double rx = seeds[k][0] - 0.5, ry = seeds[k][1] - 0.5;
        const double s = 1.0 - (rx * rx + ry * ry) / (R_VORTEX * R_VORTEX);
        const double exact = OMEGA * s * s * s * s * t / (2.0 * std::numbers::pi);
        EXPECT_NEAR(turns[k], exact, 3e-3 * exact) << "particle " << k;
        const double swept = (std::atan2(records[k].x[1] - 0.5, records[k].x[0] - 0.5) - std::atan2(ry, rx)) /
                             (2.0 * std::numbers::pi);
        EXPECT_NEAR(turns[k], swept + std::round(turns[k] - swept), roundoff(1e-8)) << "particle " << k;
        EXPECT_NEAR(age[k], t, roundoff(1e-12)) << "particle " << k;
    }
}

TEST(ParticleQuantitiesTest, ExtremaAndFirstTimesAreThoseOfTheStepStarts) {
    // Particles move exactly in uniform flow, so the values at the step starts are known: the extrema of
    // sin(2 pi x) over them, the first with x > 0.6, and the largest x among those with y > 0.5
    const std::string q = "[[particles.quantities]]\nname = \"top\"\nmax = \"sin(2 * pi * x)\"\n"
                          "[[particles.quantities]]\nname = \"bottom\"\nmin = \"sin(2 * pi * x)\"\n"
                          "[[particles.quantities]]\nname = \"arrival\"\nfirst_time = \"x > 0.6\"\n"
                          "[[particles.quantities]]\nname = \"gated\"\nmax = \"x\"\nwhile = \"y > 0.5\"\n";
    const uint32_t n_steps = 150;
    auto check = [&](const Solver & s) {
        const Particles & p = *s.get_particles();
        const std::vector<double> top = quantity_of(p, "top"), bottom = quantity_of(p, "bottom");
        const std::vector<double> arrival = quantity_of(p, "arrival"), gated = quantity_of(p, "gated");
        const auto seeds = lattice_points(0.07, 0.93);
        ASSERT_EQ(top.size(), seeds.size());
        for (size_t k = 0; k < seeds.size(); k++) {
            double hi = -1e300, lo = 1e300, first = std::numeric_limits<double>::quiet_NaN(), x_max = -1e300;
            for (uint32_t i = 0; i < n_steps; i++) {
                const double x = wrap(seeds[k][0] + U_UNIFORM * DT_UNIFORM * i, 1.0);
                hi = std::max(hi, std::sin(2.0 * std::numbers::pi * x));
                lo = std::min(lo, std::sin(2.0 * std::numbers::pi * x));
                if (std::isnan(first) && x > 0.6) first = DT_UNIFORM * i;
                x_max = std::max(x_max, x);
            }
            EXPECT_NEAR(top[k], hi, roundoff(1e-10)) << "particle " << k;
            EXPECT_NEAR(bottom[k], lo, roundoff(1e-10)) << "particle " << k;
            if (std::isnan(first)) {
                EXPECT_TRUE(std::isnan(arrival[k])) << "particle " << k << ": " << arrival[k];
            } else {
                EXPECT_NEAR(arrival[k], first, roundoff(1e-12)) << "particle " << k;
            }
            if (seeds[k][1] > 0.5) {
                EXPECT_NEAR(gated[k], x_max, roundoff(1e-10)) << "particle " << k;
            } else {
                EXPECT_EQ(gated[k], -std::numeric_limits<double>::infinity()) << "particle " << k;
            }
        }
    };
    const std::string input = uniform(lattice(0.07, 0.93) + q, n_steps);
    if (comm::size() == 1) {
        expect_matches_serial_with_and_without_step_graphs(input, check);
    } else {
        expect_matches_serial(input, check);
    }
}

TEST(ParticleQuantitiesTest, ResidenceTimeInARegionIsItsLengthOverTheSpeed) {
    // A particle crosses a slab 0.31 long at U = 0.3: 0.31 / 0.3 inside out of 2.5. A sharp in() is a step
    // function, integrated within a step (here 0.0033 off); a smooth edge makes the integrand smooth and the
    // residence time exact to the integrator's error
    const std::string slab = std::string("type = \"") + (N_DIM == 3 ? "box" : "rectangle") + "\"\ncenter = " + numbers(0.355, 0.5, 0.25) + "\nsize = " + numbers(0.31, 4.0, 4.0) + "\n";
    const std::string regions = "[regions.slab]\n" + slab + "[regions.soft]\n" + slab + "edge = 0.05\n";
    const std::string q = "[[particles.quantities]]\nname = \"sharp\"\nintegral = \"in(slab)\"\n"
                          "[[particles.quantities]]\nname = \"smooth\"\nintegral = \"in(soft)\"\n";
    // Entering at t = 0.4925 and leaving at 1.5258, within steps
    Solver solver;
    solver.set_distributed(false);
    solver.init(parse_toml(uniform("points = [" + numbers(0.05225, 0.5, 0.25) + "]\n" + q, 250, regions)));
    solver.run();
    const Particles & p = *solver.get_particles();
    const double sharp = quantity_of(p, "sharp").at(0), smooth = quantity_of(p, "smooth").at(0);
    const double exact = 0.31 / U_UNIFORM;
    EXPECT_NEAR(sharp, exact, DT_UNIFORM);
    EXPECT_NEAR(smooth, exact, 1e-4);
}

TEST(ParticleProbesTest, StatisticsOfTheParticlesInARegion) {
    // After 0.1 at U = 0.3 the lattice's columns are at x = 0.1, 0.3867, 0.6733, 0.96: the slab
    // 0.3 < x < 0.7 holds the middle two, 8 particles; every particle reads U_X = 0.3
    const std::string file = (scratch_dir() / "mallard_particle_probe_slab.csv").string();
    const std::string all = (scratch_dir() / "mallard_particle_probe_all.csv").string();
    const std::string probes = "[[particles.probes]]\nname = \"slab\"\nregion = \"slab\"\nfields = [\"x\", \"y\", "
                               "\"U_X\"]\ninterval = 5\nfile = \"" + file + "\"\n"
                               "[[particles.probes]]\nname = \"all\"\nfields = [\"x\"]\nfile = \"" + all + "\"\n";
    const std::string regions = std::string("[regions.slab]\ntype = \"") + (N_DIM == 3 ? "box" : "rectangle") +
                                "\"\ncenter = " + numbers(0.5, 0.5, 0.25) + "\nsize = " + numbers(0.4, 4.0, 4.0) + "\n";
    auto check = [&](const Solver &) {
        if (!comm::is_root()) return;
        const auto rows = csv_rows(file);
        ASSERT_EQ(rows.size(), 2u);
        const std::vector<double> & r = rows[1];
        ASSERT_EQ(r.size(), 3u + 12u);
        EXPECT_EQ(r[0], 10.0);
        EXPECT_NEAR(r[1], 0.1, roundoff(1e-15));
        EXPECT_EQ(r[2], 8.0);
        const double a = 0.07 + 0.86 / 3.0 + 0.03, b = 0.07 + 2.0 * 0.86 / 3.0 + 0.03;
        EXPECT_NEAR(r[3], 0.5 * (a + b), roundoff(1e-12));     // x mean
        EXPECT_NEAR(r[4], 0.5 * (b - a), roundoff(1e-7));      // x std
        EXPECT_NEAR(r[5], a, roundoff(1e-12));                 // x min
        EXPECT_NEAR(r[6], b, roundoff(1e-12));                 // x max
        EXPECT_NEAR(r[7], 0.5, roundoff(1e-12));               // y mean
        EXPECT_NEAR(r[9], 0.07, roundoff(1e-12));              // y min
        EXPECT_NEAR(r[10], 0.93, roundoff(1e-12));             // y max
        EXPECT_NEAR(r[11], U_UNIFORM, roundoff(1e-12));        // U_X mean
        EXPECT_NEAR(r[12], 0.0, roundoff(1e-6));               // U_X std
        const auto every = csv_rows(all);
        ASSERT_EQ(every.size(), 10u);
        EXPECT_EQ(every.back()[2], 16.0);
    };
    const std::string input = uniform(lattice(0.07, 0.93) + probes, 10, regions);
    if (comm::size() == 1) {
        expect_matches_serial_with_and_without_step_graphs(input, check);
    } else {
        expect_matches_serial(input, check);
    }
}

TEST(ParticleQuantitiesTest, AQuantityGivesExactlyOneKind) {
    Solver solver;
    try {
        solver.init(parse_toml(uniform(lattice(0.07, 0.93) +
                                           "[[particles.quantities]]\nname = \"q\"\nmax = \"x\"\nmin = \"x\"\n",
                                       1)));
        FAIL() << "no error";
    } catch (const InputError & e) {
        EXPECT_NE(std::string(e.what()).find("gives 2 kinds"), std::string::npos) << e.what();
    }
}

namespace {

constexpr double MU = 0.01;

/**
 * @brief The input with solid particles of the given density, diameter, drag and extra [particles] keys in
 *        place of its tracers, in a viscous gas (mu = MU); gravity in [particles] only, so the gas is unchanged.
 */
std::string solid(const std::string & input, double density, double diameter, const std::string & drag,
                  const std::string & keys) {
    std::string s = input;
    const std::string tracer = "type = \"tracer\"\n";
    std::ostringstream p;
    p.precision(17);
    p << "type = \"solid\"\ndensity = " << density << "\ndiameter = " << diameter << "\ndrag = \"" << drag << "\"\n"
      << keys;
    s.replace(s.find(tracer), tracer.size(), p.str());
    const std::string euler = "type = \"euler\"";
    s.replace(s.find(euler), euler.size(), "type = \"navier_stokes\"\nmu = 0.01");
    return s;
}

/** @brief The diameter of particles of density rho_p whose Stokes relaxation time in the gas of solid() (rho = 1) is tau. */
double diameter_for(double tau, double density) { return std::sqrt(18.0 * MU * tau / density); }

/**
 * @brief The largest errors of position and velocity of particles of relaxation time tau = 0.2 (Stokes drag)
 *        in the turning flow of turning_error, which start at the gas velocity: with w = v_x + i v_y and the
 *        gas U e^{iKt}, w = A e^{iKt} + (U - A) e^{-t/tau} with A = U / (1 + iK tau), and the position moves
 *        by A (e^{iKt} - 1) / (iK) + (U - A) tau (1 - e^{-t/tau}).
 */
std::array<double, 2> inertial_turning_error(const std::string & integrator, uint32_t n_steps) {
    using C = std::complex<double>;
    const double K = 2.0 * std::numbers::pi, t = 1.0, tau = 0.2, density = 1000.0;
    std::ostringstream run;
    run.precision(17);
    run << "n_steps = " << n_steps << "\ndt = " << t / n_steps << "\n";
    std::string input = solid(box(12, vec("0.3", "-0.2", "0.0"), "xy", "", run.str(), lattice(0.3, 0.7), TENO4, integrator),
                              density, diameter_for(tau, density), "stokes", "");
    input.insert(input.find("[numerics]"), "[source]\nrhou = " + vec("-2 * pi * rho * u_y", "2 * pi * rho * u_x", "0.0") + "\n");
    Solver solver;
    solver.set_distributed(false);
    solver.init(parse_toml(input));
    solver.run();
    const C U(0.3, -0.2), I(0.0, 1.0);
    const C A = U / (1.0 + I * K * tau);
    const C w = A * std::exp(I * K * t) + (U - A) * std::exp(-t / tau);
    const C dz = A * (std::exp(I * K * t) - 1.0) / (I * K) + (U - A) * tau * (1.0 - std::exp(-t / tau));
    const auto seeds = lattice_points(0.3, 0.7);
    std::array<double, 2> error{0.0, 0.0};
    const std::vector<particles::Record> records = solver.get_particles()->gather();
    EXPECT_EQ(records.size(), seeds.size());
    for (const auto & r : records) {
        error[0] = std::max({error[0], std::abs(r.x[0] - wrap(seeds[r.id][0] + dz.real(), 1.0)),
                             std::abs(r.x[1] - wrap(seeds[r.id][1] + dz.imag(), 1.0))});
        error[1] = std::max({error[1], std::abs(r.v[0] - w.real()), std::abs(r.v[1] - w.imag())});
    }
    return error;
}

/** @brief Solid particles starting at rest in gas at rest, under gravity (0, -g) in [particles], for n_steps of dt. */
std::string settling(double tau, double g, const std::string & drag, uint32_t n_steps, double dt,
                     const std::string & integrator = "RK4") {
    std::ostringstream run;
    run.precision(17);
    run << "n_steps = " << n_steps << "\ndt = " << dt << "\n";
    const double density = 100.0;
    return solid(box(12, vec("0.0", "0.0", "0.0"), "xy", "", run.str(), lattice(0.1, 0.9), TENO4, integrator), density,
                 diameter_for(tau, density), drag, "gravity = " + numbers(0.0, -g, 0.0) + "\n");
}

/** @brief Solid particles of relaxation time 0.05 in the vortex, which flings them out across the partitions. */
std::string solid_vortex(uint32_t n_steps, const std::string & init = "", const std::string & extra = "") {
    return solid(vortex(16, 0.01, n_steps, TENO4, "SSPRK3", init, extra), 1000.0, diameter_for(0.05, 1000.0),
                 "schiller_naumann", "");
}

} // namespace

TEST(SolidParticlesTest, DragRelaxationConvergesAtTheIntegratorOrder) {
    SKIP_IN_SINGLE_PRECISION("RK4's error with 80 steps is near single-precision round-off");
    // Each stage reads the gas of its own time and the particle's own velocity
    for (const auto & [integrator, steps, order] :
         {std::tuple{"FE", 80u, 0.9}, std::tuple{"SSPRK3", 40u, 2.8}, std::tuple{"RK4", 40u, 3.8}}) {
        const auto coarse = inertial_turning_error(integrator, steps), fine = inertial_turning_error(integrator, 2 * steps);
        EXPECT_GT(std::log2(coarse[0] / fine[0]), order) << integrator << " position";
        EXPECT_GT(std::log2(coarse[1] / fine[1]), order) << integrator << " velocity";
    }
}

TEST(SolidParticlesTest, StokesSettlingIsExact) {
    // v(t) = v_t (1 - e^{-t/tau}) with v_t = -tau g (1 - rho / rho_p), from rest in gas at rest
    const double tau = 0.1, g = 1.0, t = 0.5;
    const double v_t = -tau * g * (1.0 - 1.0 / 100.0);
    const double v = v_t * (1.0 - std::exp(-t / tau)), dy = v_t * (t - tau * (1.0 - std::exp(-t / tau)));
    auto check = [&](const Solver & s) {
        const auto seeds = lattice_points(0.1, 0.9);
        const std::vector<particles::Record> records = s.get_particles()->gather();
        ASSERT_EQ(records.size(), seeds.size());
        for (const auto & r : records) {
            EXPECT_NEAR(r.x[0], seeds[r.id][0], roundoff(1e-14));
            EXPECT_NEAR(r.x[1], wrap(seeds[r.id][1] + dy, 1.0), roundoff(1e-9));
            EXPECT_NEAR(r.v[0], 0.0, roundoff(1e-14));
            EXPECT_NEAR(r.v[1], v, roundoff(1e-8));
        }
    };
    const std::string input = settling(tau, g, "stokes", 100, 0.005);
    if (comm::size() == 1) {
        expect_matches_serial_with_and_without_step_graphs(input, check);
    } else {
        expect_matches_serial(input, check);
    }
}

TEST(SolidParticlesTest, SchillerNaumannTerminalVelocity) {
    // At Re of order 1 the drag is 1 + 0.15 Re^0.687 times Stokes drag: v_t = tau g' / f(Re(v_t))
    const double tau = 0.1, g = 10.0, density = 100.0;
    const double d = diameter_for(tau, density), g_eff = g * (1.0 - 1.0 / density);
    double v_t = tau * g_eff;
    for (int i = 0; i < 200; i++) v_t = tau * g_eff / (1.0 + 0.15 * std::pow(v_t * d / MU, 0.687));
    Solver solver;
    solver.set_distributed(false);
    solver.init(parse_toml(settling(tau, g, "schiller_naumann", 300, 0.01)));
    solver.run();
    const std::vector<particles::Record> records = solver.get_particles()->gather();
    ASSERT_EQ(records.size(), 16u);
    for (const auto & r : records) EXPECT_NEAR(r.v[1], -v_t, roundoff(1e-10) * v_t);
    EXPECT_LT(v_t, 0.9 * tau * g_eff);  // the correction is not negligible
}

TEST(SolidParticlesTest, TinyParticlesTrackTheGasAtAnyStep) {
    // tau = 1e-5 is a thousandth of the step, where explicit integration diverges. The particles move at
    // their equilibrium velocity u + tau g' exactly, and their velocity relaxes to it
    const double tau = 1e-5, g = 1000.0, t = 0.5;
    const double v_t = -tau * g * (1.0 - 1.0 / 100.0);
    Solver solver;
    solver.set_distributed(false);
    solver.init(parse_toml(settling(tau, g, "stokes", 50, t / 50)));
    solver.run();
    const auto seeds = lattice_points(0.1, 0.9);
    const std::vector<particles::Record> records = solver.get_particles()->gather();
    ASSERT_EQ(records.size(), seeds.size());
    for (const auto & r : records) {
        EXPECT_NEAR(r.x[1], seeds[r.id][1] + v_t * t, roundoff(1e-14));
        EXPECT_NEAR(r.v[1], v_t, roundoff(1e-12) * std::abs(v_t));
    }
}

TEST(SolidParticlesTest, WallsReboundWithTheRestitutionCoefficientsOrStick) {
    // In gas at rest Stokes drag slows both components by one factor, so v_y / v_x changes only at the
    // bounce, by -e_n / e_t
    const double e_n = 0.5, e_t = 0.8;
    const auto run = [&](const std::string & wall) {
        const std::string walls = "[[boundaries]]\nname = \"bottom\"\ntype = \"wall_adiabatic\"\n"
                                  "[[boundaries]]\nname = \"top\"\ntype = \"wall_adiabatic\"\n";
        const std::string keys = "velocity = " + numbers(0.3, -0.4, 0.0) + "\npoints = [" + numbers(0.5, 0.06, 0.25) +
                                 "]\n[[particles.walls]]\nzones = [\"bottom\"]\n" + wall;
        auto solver = std::make_unique<Solver>();
        solver->set_distributed(false);
        solver->init(parse_toml(solid(box(12, vec("0.0", "0.0", "0.0"), "x", walls, "n_steps = 50\ndt = 0.01\n", "",
                                          TENO4, "RK4"),
                                      1000.0, diameter_for(5.0, 1000.0), "stokes", keys)));
        solver->run();
        return solver;
    };
    std::ostringstream rebound;
    rebound << "restitution = " << e_n << "\ntangential_restitution = " << e_t << "\n";
    const auto bounced = run(rebound.str());
    const std::vector<particles::Record> records = bounced->get_particles()->gather();
    ASSERT_EQ(records.size(), 1u);
    EXPECT_GT(records[0].v[1], 0.0);
    EXPECT_GT(records[0].x[1], 0.0);
    EXPECT_NEAR(records[0].v[1] / records[0].v[0], (e_n / e_t) * (0.4 / 0.3), roundoff(1e-12));
    const auto stuck = run("interaction = \"stick\"\n");
    EXPECT_EQ(stuck->get_particles()->count(), 0u);
    EXPECT_EQ(stuck->get_particles()->removed(), (std::array<uint64_t, 3>{0, 0, 1}));
}

TEST(SolidParticlesTest, VelocitiesTurnAcrossRotationalPeriodicSides) {
    // In gas at rest a particle moves straight on, slowing by e^{-t/tau}; crossing the side of a quarter
    // annulus at 90 degrees it continues in the sector turned back by 90 degrees, its velocity turned too
    const double tau = 5.0, t = 0.5, s0 = 0.5, theta0 = 85.0 * std::numbers::pi / 180.0;
    std::ostringstream input;
    input.precision(17);
    input << "[run]\nn_steps = 50\ndt = 0.01\n"
          << "[mesh]\ntype = \"annulus\"\nNtheta = 12\nNr = 6\nNz = 3\nr_inner = 1.0\nr_outer = 2.0\nLz = 0.6\n"
          << "angle = 90.0\n[[periodic]]\nzones = [\"start\", \"end\"]\n"
          << (N_DIM == 3 ? "[[periodic]]\nzones = [\"back\", \"front\"]\n" : "")
          << "[initialize]\ntype = \"constant\"\nu = " << numbers(0.0, 0.0, 0.0) << "\np = 1.0\nT = 1.0\n"
          << "[[boundaries]]\nname = \"inner\"\ntype = \"wall_adiabatic\"\n"
          << "[[boundaries]]\nname = \"outer\"\ntype = \"wall_adiabatic\"\n"
          << "[numerics]\nriemann_solver = \"HLLC\"\ntime_integrator = \"RK4\"\n"
          << "[numerics.face_reconstruction]\ntype = \"MUSCL\"\n" << PHYSICS
          << "[output]\ncheck_interval = 1000000\n[particles]\ntype = \"tracer\"\n"
          << "points = [" << numbers(1.5 * std::cos(theta0), 1.5 * std::sin(theta0), 0.3) << "]\n"
          << "velocity = " << numbers(-s0 * std::sin(theta0), s0 * std::cos(theta0), 0.0) << "\n";
    Solver solver;
    solver.set_distributed(false);
    solver.init(parse_toml(solid(input.str(), 1000.0, diameter_for(tau, 1000.0), "stokes", "")));
    solver.run();
    const double L = s0 * tau * (1.0 - std::exp(-t / tau)), s = s0 * std::exp(-t / tau);
    const double x = 1.5 * std::cos(theta0) - L * std::sin(theta0), y = 1.5 * std::sin(theta0) + L * std::cos(theta0);
    ASSERT_GT(std::atan2(y, x), std::numbers::pi / 2.0);  // it crossed
    const std::vector<particles::Record> records = solver.get_particles()->gather();
    ASSERT_EQ(records.size(), 1u);
    // (x, y) turned by -90 degrees is (y, -x)
    EXPECT_NEAR(records[0].x[0], y, roundoff(1e-12));
    EXPECT_NEAR(records[0].x[1], -x, roundoff(1e-12));
    EXPECT_NEAR(records[0].v[0], s * std::cos(theta0), roundoff(1e-12));
    EXPECT_NEAR(records[0].v[1], s * std::sin(theta0), roundoff(1e-12));
}

TEST(SolidParticlesTest, TrajectoriesAreBitwiseIndependentOfTheRankCount) {
    if (comm::size() == 1) GTEST_SKIP() << "compares distributed runs with serial ones";
    const std::string input = solid_vortex(40);
    Solver distributed;
    distributed.init(parse_toml(input));
    distributed.run();
    Solver serial;
    serial.set_distributed(false);
    serial.init(parse_toml(input));
    serial.run();
    expect_same_particles(*distributed.get_particles(), *serial.get_particles());
}

TEST(SolidParticlesTest, RestartedRunContinuesTheVelocitiesExactly) {
    // A restart's particles keep their velocities instead of starting at the gas's
    const std::string dir = (scratch_dir() / "mallard_solid_particles_restart").string();
    const std::string writer = "[[write_data]]\nprefix = \"" + dir + "/restart\"\nformat = \"restart\"\ninterval = 20\n";
    Solver straight;
    straight.init(parse_toml(solid_vortex(40)));
    straight.run();
    Solver first;
    first.init(parse_toml(solid_vortex(20, "", writer)));
    first.run();
    comm::barrier();
    if (!comm::is_root()) return;
    Solver second;
    second.set_distributed(false);
    second.init(parse_toml(solid_vortex(40, "type = \"restart\"\nfile = \"" + dir + "/restart_000020.restart\"\n")));
    second.run();
    expect_same_particles(*second.get_particles(), *straight.get_particles());
}

TEST(SolidParticlesTest, SolidParticlesNeedAViscousGas) {
    Solver solver;
    std::string input = solid(uniform(lattice(0.07, 0.93), 1), 1000.0, 1e-3, "stokes", "");
    input.replace(input.find("type = \"navier_stokes\"\nmu = 0.01"), 32, "type = \"euler\"");
    try {
        solver.init(parse_toml(input));
        FAIL() << "no error";
    } catch (const InputError & e) {
        EXPECT_NE(std::string(e.what()).find("needs the gas viscosity"), std::string::npos) << e.what();
    }
}

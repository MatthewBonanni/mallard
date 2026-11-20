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

#include <array>
#include <cmath>
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
        EXPECT_EQ(p.removed(), (std::array<uint64_t, 2>{0, 0}));
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
        for (size_t d = 0; d < N_DIM; d++) EXPECT_EQ(ra[k].x[d], rb[k].x[d]) << "particle " << ra[k].id << ", direction " << d;
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
    // On several ranks rank 0 restarts serially: the file restarts on any rank count
    const std::string dir = (scratch_dir() / "mallard_particles_restart").string();
    const std::string writer = "[[write_data]]\nprefix = \"" + dir + "/restart\"\nformat = \"restart\"\ninterval = 20\n";
    Solver straight;
    straight.init(parse_toml(vortex(16, 0.01, 40, TENO4, "SSPRK3")));
    straight.run();
    const std::vector<particles::Record> expected = straight.get_particles()->gather();
    const std::array<uint64_t, 2> removed = straight.get_particles()->removed();
    Solver first;
    first.init(parse_toml(vortex(16, 0.01, 20, TENO4, "SSPRK3", "", writer)));
    first.run();
    comm::barrier();
    if (!comm::is_root()) return;
    Solver second;
    second.set_distributed(false);
    second.init(parse_toml(vortex(16, 0.01, 40, TENO4, "SSPRK3",
                                  "type = \"restart\"\nfile = \"" + dir + "/restart_000020.restart\"\n")));
    second.run();
    const std::vector<particles::Record> records = second.get_particles()->gather();
    ASSERT_EQ(records.size(), expected.size());
    for (size_t k = 0; k < records.size(); k++) {
        EXPECT_EQ(records[k].id, expected[k].id);
        for (size_t d = 0; d < N_DIM; d++) EXPECT_EQ(records[k].x[d], expected[k].x[d]) << "particle " << k;
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
    auto x = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), p.positions());
    x(1) = static_cast<rtype>(-0.03);                 // 0.03 below the wall
    x(N_DIM + 1) = static_cast<rtype>(1.04);          // 0.04 above the symmetry plane
    Kokkos::deep_copy(p.positions(), x);
    p.relocate();
    const std::vector<particles::Record> records = p.gather();
    ASSERT_EQ(records.size(), 2u);
    EXPECT_NEAR(records[0].x[0], 0.5, roundoff(1e-14));
    EXPECT_NEAR(records[0].x[1], 0.03, roundoff(1e-14));
    EXPECT_NEAR(records[1].x[0], 0.3, roundoff(1e-14));
    EXPECT_NEAR(records[1].x[1], 0.96, roundoff(1e-14));
    EXPECT_EQ(p.removed(), (std::array<uint64_t, 2>{0, 0}));
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
    EXPECT_EQ(p.removed(), (std::array<uint64_t, 2>{8, 0}));
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

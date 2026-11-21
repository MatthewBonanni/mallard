/**
 * @file particles_injection_test.cpp
 * @author Matthew Bonanni (mbonanni001@gmail.com)
 * @brief Tests of the particle injectors: size distributions against their
 *        PDFs, rates, sites and directions, rank-count and step-graph
 *        independence, restarts and capacity.
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
#include <sstream>
#include <string>
#include <vector>

#include "check.h"
#include "comm.h"
#include "mpi_compare.h"
#include "particle_injection.h"
#include "particles.h"
#include "scratch.h"
#include "solver.h"
#include "test_fixtures.h"

namespace {

constexpr double PI = std::numbers::pi;

std::string numbers(double x, double y, double z) {
    std::ostringstream s;
    s.precision(17);
    s << "[" << x << ", " << y;
    if (N_DIM == 3) s << ", " << z;
    s << "]";
    return s.str();
}

/**
 * @brief A unit box of 16 cells per side (8 along z, 0.5 long) of viscous gas at rho = p = T = 1 moving at u along
 *        x: periodic on every side if u = 0, else with an inlet on the left and an outlet on the right. Solid or
 *        tracer particles with the given [particles] keys.
 */
std::string box(double u, const std::string & run, const std::string & particles, const std::string & type = "tracer") {
    std::ostringstream s;
    s.precision(17);
    s << "[run]\n" << run << "[mesh]\ntype = \"cartesian\"\nNx = 16\nNy = 16\nLx = 1.0\nLy = 1.0\n";
    if (N_DIM == 3) s << "Nz = 8\nLz = 0.5\n";
    if (u == 0.0) {
        s << periodic_sides(N_DIM == 3 ? "xyz" : "xy");
    } else {
        s << periodic_sides(N_DIM == 3 ? "yz" : "y") << "[[boundaries]]\nname = \"left\"\ntype = \"inlet_state\"\nu = "
          << numbers(u, 0.0, 0.0) << "\np = 1.0\nT = 1.0\n[[boundaries]]\nname = \"right\"\ntype = \"outlet_extrapolation\"\n";
    }
    s << "[initialize]\ntype = \"constant\"\nT = 1.0\np = 1.0\nu = " << numbers(u, 0.0, 0.0) << "\n"
      << "[numerics]\nriemann_solver = \"HLLC\"\ntime_integrator = \"RK4\"\n"
      << "[numerics.face_reconstruction]\ntype = \"MUSCL\"\n"
      << "[physics]\ntype = \"navier_stokes\"\nunits = \"nondimensional\"\nmu = 1e-3\ngamma = 1.4\np_ref = 1.0\nT_ref = 1.0\nrho_ref = 1.0\n"
      << "[output]\ncheck_interval = 1000000\n"
      << "[particles]\ntype = \"" << type << "\"\n" << particles;
    return s.str();
}

/** @brief Heavy spheres: their relaxation time, about 55, is far above the runs' durations. */
const std::string HEAVY = "density = 1e6\ndiameter = 1e-3\n";

double norm(const std::array<double, N_DIM> & v) {
    double s = 0.0;
    for (const double x : v) s += x * x;
    return std::sqrt(s);
}

/** @brief The Kolmogorov-Smirnov distance between samples and a CDF. */
double ks_distance(std::vector<double> samples, const std::function<double(double)> & cdf) {
    std::sort(samples.begin(), samples.end());
    const double n = static_cast<double>(samples.size());
    double d = 0.0;
    for (size_t i = 0; i < samples.size(); i++) {
        const double f = cdf(samples[i]);
        d = std::max({d, std::abs(f - static_cast<double>(i) / n), std::abs(static_cast<double>(i + 1) / n - f)});
    }
    return d;
}

/** @brief The 1% critical value of the Kolmogorov-Smirnov distance for n samples. */
double ks_critical(size_t n) { return 1.63 / std::sqrt(static_cast<double>(n)); }

void expect_same_particles(const std::vector<particles::Record> & a, const std::vector<particles::Record> & b) {
    ASSERT_EQ(a.size(), b.size());
    for (size_t k = 0; k < a.size(); k++) {
        ASSERT_EQ(a[k].id, b[k].id);
        for (size_t i = 0; i < N_DIM; i++) {
            EXPECT_EQ(a[k].x[i], b[k].x[i]) << "particle " << a[k].id;
            EXPECT_EQ(a[k].v[i], b[k].v[i]) << "particle " << a[k].id;
        }
        if (!std::isnan(a[k].d) || !std::isnan(b[k].d)) {
            EXPECT_EQ(a[k].d, b[k].d) << "particle " << a[k].id;
            EXPECT_EQ(a[k].w, b[k].w) << "particle " << a[k].id;
        }
    }
}

} // namespace

TEST(ParticleSizeTest, SampledDistributionsMatchTheirPdfs) {
    // Against the CDF of each PDF integrated finely and independently, by number: distributions by mass weigh
    // their PDF by d^-3. Moments within 4 standard errors, and the Kolmogorov-Smirnov distance below 1%.
    struct Case {
        std::string keys;
        std::function<double(double)> number_pdf;
        double lo, hi;
    };
    const double X = 50e-6, q = 3.0, m = 30e-6, s = 0.5;
    const auto rr = [=](double d) { return q / X * std::pow(d / X, q - 1.0) * std::exp(-std::pow(d / X, q)); };
    const auto ln = [=](double d) { return std::exp(-0.5 * std::pow(std::log(d / m) / s, 2)) / d; };
    const auto nt = [=](double d) { return std::pow(d / X, 2.0) * std::exp(-std::pow(d / X, 1.5)); };
    const std::vector<Case> cases = {
        {"distribution = \"rosin_rammler\"\nsize = 50e-6\nspread = 3.0\nbasis = \"number\"\nmin = 5e-6\nmax = 120e-6\n",
         rr, 5e-6, 120e-6},
        {"distribution = \"rosin_rammler\"\nsize = 50e-6\nspread = 3.0\nbasis = \"mass\"\nmin = 5e-6\nmax = 120e-6\n",
         [=](double d) { return rr(d) / (d * d * d); }, 5e-6, 120e-6},
        {"distribution = \"log_normal\"\nmedian = 30e-6\nsigma = 0.5\nbasis = \"number\"\nmin = 8e-6\nmax = 100e-6\n",
         ln, 8e-6, 100e-6},
        {"distribution = \"nukiyama_tanasawa\"\nsize = 50e-6\nspread = 1.5\nexponent = 2.0\nbasis = \"mass\"\nmin = 5e-6\n"
         "max = 300e-6\n",
         [=](double d) { return nt(d) / (d * d * d); }, 5e-6, 300e-6},
        {"distribution = \"uniform\"\nmin = 10e-6\nmax = 20e-6\nbasis = \"number\"\n", [](double) { return 1.0; }, 10e-6,
         20e-6},
        {"distribution = \"table\"\ndiameters = [10e-6, 20e-6, 40e-6]\npdf = [0.0, 1.0, 0.0]\nbasis = \"number\"\n",
         [](double d) { return d < 20e-6 ? (d - 10e-6) / 10e-6 : (40e-6 - d) / 20e-6; }, 10e-6, 40e-6},
    };
    for (const Case & c : cases) {
        SCOPED_TRACE(c.keys);
        Input input = parse_toml("[diameter]\n" + c.keys);
        const particles::SizeDistribution size = particles::read_size_distribution(input, "diameter", "test");
        // Reference: Simpson's rule on 2e5 intervals in d
        const uint32_t n = 200000;
        std::vector<double> d(n + 1), F(n + 1, 0.0);
        double m0 = 0.0, m1 = 0.0, m2 = 0.0, m3 = 0.0;
        for (uint32_t i = 0; i <= n; i++) d[i] = c.lo + (c.hi - c.lo) * i / n;
        for (uint32_t i = 1; i <= n; i++) {
            const double a = d[i - 1], b = d[i], mid = 0.5 * (a + b);
            const double fa = c.number_pdf(a), fm = c.number_pdf(mid), fb = c.number_pdf(b);
            const double h = (b - a) / 6.0;
            F[i] = F[i - 1] + h * (fa + 4.0 * fm + fb);
            m0 += h * (fa + 4.0 * fm + fb);
            m1 += h * (a * fa + 4.0 * mid * fm + b * fb);
            m2 += h * (a * a * fa + 4.0 * mid * mid * fm + b * b * fb);
            m3 += h * (a * a * a * fa + 4.0 * mid * mid * mid * fm + b * b * b * fb);
        }
        const auto cdf = [&](double x) {
            if (x <= c.lo) return 0.0;
            if (x >= c.hi) return 1.0;
            const double at = (x - c.lo) / (c.hi - c.lo) * n;
            const uint32_t i = std::min(static_cast<uint32_t>(at), n - 1);
            return (F[i] + (at - i) * (F[i + 1] - F[i])) / F[n];
        };
        const double mean = m1 / m0, var = m2 / m0 - mean * mean;
        EXPECT_NEAR(size.mean_d, mean, 1e-4 * mean);
        EXPECT_NEAR(size.mean_d3, m3 / m0, 1e-3 * m3 / m0);
        EXPECT_GE(size.lo, c.lo * (1.0 - 1e-12));
        EXPECT_LE(size.hi, c.hi * (1.0 + 1e-12));
        const size_t N = 40000;
        std::vector<double> samples(N);
        double sum = 0.0;
        for (size_t k = 0; k < N; k++) {
            samples[k] = size.sample(particles::Random(7, 0, k).uniform(particles::DIAMETER));
            sum += samples[k];
        }
        EXPECT_NEAR(sum / N, mean, 4.0 * std::sqrt(var / N));
        EXPECT_LT(ks_distance(samples, cdf), ks_critical(N));
    }
    // One diameter
    Input input = parse_toml("diameter = 42e-6\n");
    const particles::SizeDistribution one = particles::read_size_distribution(input, "diameter", "test");
    EXPECT_EQ(one.sample(0.3), 42e-6);
}

TEST(ParticleSizeTest, DefaultBoundsKeepAllButTheFarTails) {
    // Without min and max a log-normal keeps its 1e-9 quantiles: about 6 sigma about the median
    Input input = parse_toml("[diameter]\ndistribution = \"log_normal\"\nmedian = 1e-5\nsigma = 0.4\nbasis = \"number\"\n");
    const particles::SizeDistribution size = particles::read_size_distribution(input, "diameter", "test");
    const double z = std::log(size.hi / 1e-5) / 0.4;
    EXPECT_GT(z, 5.8);
    EXPECT_LT(z, 6.2);
    EXPECT_NEAR(std::log(1e-5 / size.lo) / 0.4, z, 0.1);
    EXPECT_NEAR(size.mean_d, 1e-5 * std::exp(0.5 * 0.4 * 0.4), 1e-4 * 1e-5);
}

TEST(ParticleInjectionTest, APointInjectorEmitsItsRateIntoTheFlow) {
    // 274 parcels/s, 1.37 per step of 0.005: parcel k enters at the end of the first step s with floor(1.37 s) > k,
    // at the point, and moves with the gas at 1 for the rest of the run
    const double dt = 0.005, x0 = 0.2;
    const uint32_t steps = 40;
    const std::string keys = "[[particles.injectors]]\nname = \"nozzle\"\npoint = " + numbers(x0, 0.5, 0.25) +
                             "\nrate = 274.0\n";
    const auto check = [&](const Solver & solver) {
        const std::vector<particles::Record> records = solver.get_particles()->gather();
        ASSERT_EQ(records.size(), 54u);
        for (size_t k = 0; k < records.size(); k++) {
            EXPECT_EQ(records[k].id, k);
            uint32_t s = 1;
            while (std::floor(1.37 * s + 1e-9) <= static_cast<double>(k)) s++;
            const double age = (steps - s) * dt;
            EXPECT_NEAR(records[k].x[0], x0 + age, 1e-12) << "parcel " << k;
            EXPECT_NEAR(records[k].x[1], 0.5, 1e-12);
        }
    };
    const std::string input = box(1.0, "n_steps = 40\ndt = 0.005\n", keys);
    Solver solver;
    solver.init(parse_toml(input));
    solver.run();
    check(solver);
    EXPECT_EQ(solver.get_particles()->removed(), (std::array<uint64_t, 3>{0, 0, 0}));
}

TEST(ParticleInjectionTest, InjectorsEmitOnlyBetweenTheirStartAndEnd) {
    const std::string keys = "[[particles.injectors]]\nname = \"pulse\"\npoint = " + numbers(0.2, 0.5, 0.25) +
                             "\nrate = 200.0\nstart = 0.0525\nend = 0.1525\n";
    Solver solver;
    solver.init(parse_toml(box(1.0, "n_steps = 40\ndt = 0.005\n", keys)));
    solver.run();
    const std::vector<particles::Record> records = solver.get_particles()->gather();
    EXPECT_GE(records.size(), 19u);
    EXPECT_LE(records.size(), 21u);
    // Ages between 0.2 - 0.1525 and 0.2 - 0.0525, to the step
    for (const auto & r : records) {
        EXPECT_GE(r.x[0] - 0.2, 0.2 - 0.155 - 1e-12);
        EXPECT_LE(r.x[0] - 0.2, 0.2 - 0.05 + 1e-12);
    }
}

TEST(ParticleInjectionTest, AZoneInjectorSpreadsByAreaAlongItsInwardNormal) {
    // Heavy spheres launched at 2 from the inlet: uniform over it, moving along +x
    const std::string keys = HEAVY + "[[particles.injectors]]\nname = \"inlet\"\nregion = \"left\"\nrate = 2000.0\n"
                                     "speed = 2.0\n";
    Solver solver;
    solver.init(parse_toml(box(1.0, "n_steps = 40\ndt = 0.005\n", keys, "solid")));
    solver.run();
    const std::vector<particles::Record> records = solver.get_particles()->gather();
    ASSERT_EQ(records.size(), 400u);
    std::vector<double> y, z;
    for (const auto & r : records) {
        EXPECT_NEAR(r.v[0], 2.0, 0.01);
        EXPECT_NEAR(r.v[1], 0.0, 1e-12);
        EXPECT_GE(r.x[0], 0.0);
        EXPECT_LE(r.x[0], 2.0 * 0.2 + 1e-9);
        y.push_back(r.x[1]);
        if (N_DIM == 3) z.push_back(r.x[N_DIM - 1] / 0.5);
        EXPECT_EQ(r.d, 1e-3);
    }
    const auto uniform = [](double x) { return std::clamp(x, 0.0, 1.0); };
    EXPECT_LT(ks_distance(y, uniform), ks_critical(y.size()));
    if (N_DIM == 3) {
        EXPECT_LT(ks_distance(z, uniform), ks_critical(z.size()));
    }
}

TEST(ParticleInjectionTest, VelocitiesFillTheirCone) {
    // Heavy spheres in gas at rest keep their directions: within 30 degrees of the axis, uniformly in angle (2D)
    // or solid angle (3D), and symmetric about it
    const double cone = 30.0 * PI / 180.0;
    std::array<double, N_DIM> axis{};
    axis[0] = 0.6;
    axis[1] = 0.8;
    const std::string keys = HEAVY + "[[particles.injectors]]\nname = \"nozzle\"\npoint = " + numbers(0.5, 0.5, 0.25) +
                             "\nrate = 20000.0\nspeed = 1.0\ndirection = " + numbers(0.6, 0.8, 0.0) +
                             "\ncone_angle = 30.0\n";
    Solver solver;
    solver.init(parse_toml(box(0.0, "n_steps = 10\ndt = 0.005\n", keys, "solid")));
    solver.run();
    const std::vector<particles::Record> records = solver.get_particles()->gather();
    ASSERT_EQ(records.size(), 1000u);
    std::vector<double> u;  // uniform in [0, 1] for a uniform cone
    std::array<double, N_DIM> side{};
    for (const auto & r : records) {
        double along = 0.0;
        for (size_t i = 0; i < N_DIM; i++) along += r.v[i] * axis[i];
        const double c = along / norm(r.v);
        ASSERT_LE(std::acos(std::min(c, 1.0)), cone * (1.0 + 1e-9));
        for (size_t i = 0; i < N_DIM; i++) side[i] += r.v[i] / norm(r.v) - c * axis[i];
        if constexpr (N_DIM == 2) {
            const double cross = (axis[0] * r.v[1] - axis[1] * r.v[0]) / norm(r.v);
            u.push_back((std::atan2(cross, c) + cone) / (2.0 * cone));
        } else {
            u.push_back((1.0 - c) / (1.0 - std::cos(cone)));
        }
    }
    const auto uniform = [](double x) { return std::clamp(x, 0.0, 1.0); };
    EXPECT_LT(ks_distance(u, uniform), ks_critical(u.size()));
    for (size_t i = 0; i < N_DIM; i++) {
        EXPECT_LT(std::abs(side[i]) / static_cast<double>(records.size()), 4.0 * 0.5 / std::sqrt(1000.0));
    }
}

TEST(ParticleInjectionTest, ARegionInjectorFillsItsShapeUniformly) {
    // Tracers in gas at rest stay where they enter: inside the disk (sphere), with the mean and the mean square
    // radius of a uniform one; and inside the mesh, uniform along each axis
    const double R = 0.2, c[3] = {0.5, 0.5, 0.25};
    const std::string shape = N_DIM == 3 ? "sphere" : "disk";
    const std::string keys = "[[particles.injectors]]\nname = \"cloud\"\nregion = {type = \"" + shape +
                             "\", center = " + numbers(c[0], c[1], c[2]) + ", radius = 0.2}\nrate = 20000.0\n";
    Solver solver;
    solver.init(parse_toml(box(0.0, "n_steps = 10\ndt = 0.005\n", keys)));
    solver.run();
    const std::vector<particles::Record> records = solver.get_particles()->gather();
    const size_t N = records.size();
    ASSERT_EQ(N, 1000u);
    double r2 = 0.0;
    std::array<double, N_DIM> mean{};
    for (const auto & r : records) {
        double d2 = 0.0;
        for (size_t i = 0; i < N_DIM; i++) {
            d2 += (r.x[i] - c[i]) * (r.x[i] - c[i]);
            mean[i] += r.x[i] / static_cast<double>(N);
        }
        ASSERT_LE(std::sqrt(d2), R * (1.0 + 1e-12));
        r2 += d2 / (R * R) / static_cast<double>(N);
    }
    // r^2 / R^2 is uniform (2D) or Beta(5/2, 1) (3D)
    const double expected = N_DIM == 3 ? 0.6 : 0.5, sd = N_DIM == 3 ? std::sqrt(12.0 / 175.0) : std::sqrt(1.0 / 12.0);
    EXPECT_NEAR(r2, expected, 4.0 * sd / std::sqrt(static_cast<double>(N)));
    for (size_t i = 0; i < N_DIM; i++) {
        EXPECT_NEAR(mean[i], c[i], 4.0 * R / 2.0 / std::sqrt(static_cast<double>(N)));
    }
    // The whole mesh, cell by cell: in the box, uniformly along each axis
    Solver whole;
    whole.init(parse_toml(box(0.0, "n_steps = 10\ndt = 0.005\n",
                              "[[particles.injectors]]\nname = \"fill\"\nregion = \"domain\"\nrate = 20000.0\n")));
    whole.run();
    const std::vector<particles::Record> filled = whole.get_particles()->gather();
    ASSERT_EQ(filled.size(), 1000u);
    const double length[3] = {1.0, 1.0, 0.5};
    for (size_t i = 0; i < N_DIM; i++) {
        std::vector<double> u;
        for (const auto & r : filled) {
            ASSERT_GE(r.x[i], 0.0);
            ASSERT_LE(r.x[i], length[i]);
            u.push_back(r.x[i] / length[i]);
        }
        // At the 0.1% level: three axes, and the disk's three checks above
        EXPECT_LT(ks_distance(u, [](double x) { return std::clamp(x, 0.0, 1.0); }),
                  1.95 / std::sqrt(static_cast<double>(u.size())));
    }
}

namespace {

/** @brief Two-way coupled spheres of a Rosin-Rammler distribution from the inlet, a disk and a nozzle in a crossflow. */
std::string mixed_injectors(const std::string & run) {
    const std::string keys =
        "coupling = \"two-way\"\ndensity = 1000.0\ndiameter = 2e-3\nweight = 3.0\n"
        "[[particles.injectors]]\nname = \"inlet\"\nregion = \"left\"\nrate = 1530.0\nspeed = 1.5\ncone_angle = 20.0\n"
        "diameter = {distribution = \"rosin_rammler\", size = 1e-3, spread = 2.5, basis = \"mass\", min = 2e-4, "
        "max = 3e-3}\n"
        "[[particles.injectors]]\nname = \"cloud\"\nregion = {type = \"" + std::string(N_DIM == 3 ? "sphere" : "disk") +
        "\", center = " + numbers(0.6, 0.3, 0.25) + ", radius = 0.15}\n"
        "rate = 830.0\nvelocity = " + numbers(0.0, 0.5, 0.0) + "\nweight = 10.0\n"
        "[[particles.injectors]]\nname = \"nozzle\"\npoint = " + numbers(0.3, 0.8, 0.1) + "\nrate = 610.0\nspeed = 3.0\n"
        "direction = " + numbers(1.0, -1.0, 0.5) + "\ncone_angle = 45.0\nstart = 0.02\n";
    return box(1.0, run, keys, "solid");
}

} // namespace

TEST(ParticleInjectionTest, InjectedParticlesAreBitwiseIndependentOfRanksAndStepGraphs) {
    // The gas, which the particles act on, and every particle, on any rank count, with step graphs and without
    const std::string input = mixed_injectors("n_steps = 30\ndt = 0.004\n");
    std::vector<std::vector<particles::Record>> runs;
    const auto keep = [&](const Solver & s) { runs.push_back(s.get_particles()->gather()); };
    if (comm::size() == 1) {
        expect_matches_serial_with_and_without_step_graphs(input, keep);
    } else {
        expect_matches_serial(input, keep);
    }
    Solver serial;
    serial.set_distributed(false);
    serial.init(parse_toml(input));
    serial.run();
    const std::vector<particles::Record> reference = serial.get_particles()->gather();
    EXPECT_GT(reference.size(), 150u);
    for (const auto & run : runs) expect_same_particles(run, reference);
}

TEST(ParticleInjectionTest, RestartedRunContinuesTheInjectionExactly) {
    // The injectors' counts, fractions of a parcel and the next id continue: the same parcels enter
    // (at the restart every injector owes a fraction of a parcel)
    const std::string dir = (scratch_dir() / "mallard_particles_injection_restart").string();
    const std::string writer = "[[write_data]]\nprefix = \"" + dir + "/restart\"\nformat = \"restart\"\ninterval = 15\n";
    Solver straight;
    straight.init(parse_toml(mixed_injectors("n_steps = 30\ndt = 0.004\n")));
    straight.run();
    const std::vector<particles::Record> expected = straight.get_particles()->gather();
    Solver first;
    first.init(parse_toml(mixed_injectors("n_steps = 15\ndt = 0.004\n") + writer));
    first.run();
    comm::barrier();
    if (!comm::is_root()) return;
    Solver second;
    second.set_distributed(false);
    std::string input = mixed_injectors("n_steps = 30\ndt = 0.004\n");
    const size_t at = input.find("type = \"constant\"\nT = 1.0\np = 1.0\nu = ");
    ASSERT_NE(at, std::string::npos);
    input.replace(at, input.find("[numerics]") - at,
                  "type = \"restart\"\nfile = \"" + dir + "/restart_000015.restart\"\n");
    second.init(parse_toml(input));
    second.run();
    expect_same_particles(second.get_particles()->gather(), expected);
}

TEST(ParticleInjectionTest, AMassFlowSetsTheRateFromTheMeanParticleMass) {
    // One diameter: mass_flow / (weight m) parcels per second exactly; a distribution: the mass of the parcels
    // that entered approaches mass_flow t
    const double rho_p = 1000.0, w = 5.0, t = 0.2;
    const double d = 1e-3, m = rho_p * PI / 6.0 * d * d * d;
    // 563.2 parcels/s: 112.64 over the run
    const double mdot = 563.2 * w * m;
    std::ostringstream keys;
    keys.precision(17);
    keys << "density = " << rho_p << "\n[[particles.injectors]]\nname = \"inlet\"\nregion = \"left\"\nspeed = 1.0\n"
         << "weight = " << w << "\nmass_flow = " << mdot << "\ndiameter = " << d << "\n";
    Solver one;
    one.init(parse_toml(box(1.0, "n_steps = 40\ndt = 0.005\n", keys.str(), "solid")));
    one.run();
    EXPECT_EQ(one.get_particles()->count(), 112u);
    std::ostringstream spread;
    spread.precision(17);
    const double mass_flow = 2e-3;
    spread << "density = " << rho_p << "\n[[particles.injectors]]\nname = \"inlet\"\nregion = \"left\"\nspeed = 1.0\n"
           << "weight = " << w << "\nmass_flow = " << mass_flow
           << "\ndiameter = {distribution = \"rosin_rammler\", size = 1e-3, spread = 2.0, basis = \"mass\", min = 2e-4}\n";
    Solver many;
    many.init(parse_toml(box(1.0, "n_steps = 40\ndt = 0.005\n", spread.str(), "solid")));
    many.run();
    // The parcels' masses sum to mass_flow t within 4 standard errors of their spread
    double mass = 0.0, square = 0.0;
    const std::vector<particles::Record> records = many.get_particles()->gather();
    for (const auto & r : records) {
        const double parcel = r.w * rho_p * PI / 6.0 * r.d * r.d * r.d;
        mass += parcel;
        square += parcel * parcel;
    }
    const double N = static_cast<double>(records.size());
    ASSERT_GT(N, 1000.0);
    const double sd = std::sqrt(square / N - (mass / N) * (mass / N));
    EXPECT_NEAR(mass, mass_flow * t, 4.0 * sd * std::sqrt(N));
}

TEST(ParticleInjectionTest, CapacityHoldsTheParcelsOfTheRun) {
    const std::string nozzle = "[[particles.injectors]]\nname = \"nozzle\"\npoint = " + numbers(0.2, 0.5, 0.25) +
                               "\nrate = 274.0\n";
    // To t_stop: 274 * 0.2 = 54.8, rounded up, and one more
    Solver sized;
    sized.init(parse_toml(box(1.0, "t_stop = 0.2\ndt = 0.005\n", nozzle)));
    EXPECT_EQ(sized.get_particles()->get_capacity(), 56u);
    // Too few slots: the run stops, naming capacity
    Solver small;
    small.init(parse_toml(box(1.0, "n_steps = 40\ndt = 0.005\n", "capacity = 20\n" + nozzle)));
    try {
        small.run();
        ADD_FAILURE() << "ran with too few slots";
    } catch (const std::runtime_error & e) {
        EXPECT_NE(std::string(e.what()).find("capacity"), std::string::npos) << e.what();
    }
    // An injector without end in a run without t_stop
    Solver endless;
    try {
        endless.init(parse_toml(box(1.0, "n_steps = 40\ncfl = 0.5\n", nozzle)));
        ADD_FAILURE() << "sized the slots of an endless injection";
    } catch (const InputError & e) {
        EXPECT_NE(std::string(e.what()).find("particles.capacity"), std::string::npos) << e.what();
    }
}

TEST(ParticleInjectionTest, InjectorInputErrors) {
    const auto error = [](const std::string & input) -> std::string {
        try {
            Solver solver;
            solver.init(parse_toml(input));
        } catch (const InputError & e) {
            return e.what();
        }
        return "";
    };
    const std::string point = "point = " + numbers(0.5, 0.5, 0.25) + "\n";
    EXPECT_NE(error(box(1.0, "n_steps = 1\ndt = 0.005\n", "[[particles.injectors]]\nname = \"a\"\nrate = 1.0\n"))
                  .find("exactly one of point and region"),
              std::string::npos);
    EXPECT_NE(error(box(1.0, "n_steps = 1\ndt = 0.005\n", "[[particles.injectors]]\nname = \"a\"\n" + point +
                                                               "rate = 1.0\nspeed = 2.0\n"))
                  .find("tracers move with the gas"),
              std::string::npos);
    EXPECT_NE(error(box(1.0, "n_steps = 1\ndt = 0.005\n",
                        HEAVY + "[[particles.injectors]]\nname = \"a\"\n" + point + "rate = 1.0\nspeed = 2.0\n", "solid"))
                  .find("direction"),
              std::string::npos);
    EXPECT_NE(error(box(1.0, "n_steps = 1\ndt = 0.005\n",
                        HEAVY + "[[particles.injectors]]\nname = \"a\"\npoint = " + numbers(1.5, 0.5, 0.25) +
                            "\nrate = 1.0\nvelocity = " + numbers(1.0, 0.0, 0.0) + "\n",
                        "solid"))
                  .find("outside the mesh"),
              std::string::npos);
    EXPECT_NE(error(box(1.0, "n_steps = 1\ndt = 0.005\n",
                        "[[particles.injectors]]\nname = \"a\"\nregion = {type = \"boundary\", of = {type = \"sphere\", center = " +
                            numbers(0.5, 0.5, 0.25) + ", radius = 0.1}}\nrate = 1.0\n"))
                  .find("boundary zones, or a filled region"),
              std::string::npos);
}

TEST(ParticleInjectionTest, CheckReportsWhereAndHowEachInjectorInjects) {
    const std::string keys = HEAVY +
                             "[[particles.injectors]]\nname = \"inlet\"\nregion = \"left\"\nrate = 100.0\nspeed = 2.0\n"
                             "[[particles.injectors]]\nname = \"nozzle\"\npoint = " + numbers(0.3, 0.4, 0.2) +
                             "\nrate = 50.0\nspeed = 1.0\ndirection = " + numbers(0.0, 1.0, 0.0) +
                             "\ncone_angle = 15.0\nend = 0.1\n";
    Solver solver;
    const CheckReport r = solver.check(parse_toml(box(1.0, "t_stop = 0.2\ndt = 0.005\n", keys, "solid")), {});
    ASSERT_EQ(r.injectors.size(), 2u);
    const CheckInjector & inlet = r.injectors[0];
    EXPECT_EQ(inlet.site, "zone");
    EXPECT_TRUE(inlet.along_normal);
    EXPECT_NEAR(inlet.center[0], 0.0, 1e-12);
    EXPECT_NEAR(inlet.center[1], 0.5, 1e-12);
    EXPECT_NEAR(inlet.measure, N_DIM == 3 ? 0.5 : 1.0, 1e-12);
    EXPECT_NEAR(inlet.parcels, 20.0, 1e-9);
    const CheckInjector & nozzle = r.injectors[1];
    EXPECT_EQ(nozzle.site, "point");
    EXPECT_FALSE(nozzle.along_normal);
    EXPECT_NEAR(nozzle.center[1], 0.4, 1e-12);
    EXPECT_NEAR(nozzle.direction[1], 1.0, 1e-12);
    EXPECT_EQ(nozzle.cone_angle, 15.0);
    EXPECT_NEAR(nozzle.parcels, 5.0, 1e-9);
    EXPECT_NEAR(nozzle.mass_flow, 50.0 * 1e6 * PI / 6.0 * 1e-9, 1e-12);
}

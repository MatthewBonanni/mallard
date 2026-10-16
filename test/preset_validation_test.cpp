/**
 * @file preset_validation_test.cpp
 * @author Matthew Bonanni (mbonanni001@gmail.com)
 * @brief One short validation case per numerics preset, run with the preset
 *        alone: the Noh implosion and Sod's problem (robust), the convergence
 *        of a smooth wave (accurate), the decay of a Taylor-Green vortex
 *        (les) and the speed of a premixed flame (reacting).
 * @version 0.1
 * @date 2026-10-16
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 *
 */

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <memory>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "example_inputs.h"
#include "exact_riemann.h"
#include "solver.h"
#include "test_fixtures.h"

namespace {

std::unique_ptr<Solver> run_input(const Input & input) {
    auto solver = std::make_unique<Solver>();
    solver->init(input);
    solver->run();
    solver->update_primitives();
    solver->copy_device_to_host();
    return solver;
}

std::string numerics(const std::string & preset) {
    return "[numerics]\npreset = \"" + preset + "\"\ncheck_nan = true\n";
}

const std::string PERFECT_GAS = "gamma = 1.4\np_ref = 1.0\nT_ref = 1.0\nrho_ref = 1.0\n";

/** @brief L1 density error of Sod's problem on n x 4 quads at t = 0.2. */
double sod_error(const std::string & preset, uint32_t n) {
    std::ostringstream s;
    s << "[run]\nt_stop = 0.2\n"
      << "[mesh]\ntype = \"cartesian\"\nNx = " << n << "\nNy = 4\nLx = 1.0\nLy = " << 4.0 / n << "\n"
      << "[initialize]\ntype = \"analytical\"\nrho = \"x < 0.5 ? 1.0 : 0.125\"\nu = [\"0.0\", \"0.0\"]\n"
      << "p = \"x < 0.5 ? 1.0 : 0.1\"\n"
      << "[[boundaries]]\nname = \"left\"\ntype = \"extrapolation\"\n"
      << "[[boundaries]]\nname = \"right\"\ntype = \"extrapolation\"\n"
      << "[[boundaries]]\nname = \"top\"\ntype = \"symmetry\"\n"
      << "[[boundaries]]\nname = \"bottom\"\ntype = \"symmetry\"\n"
      << numerics(preset) << "[physics]\ntype = \"euler\"\n" << PERFECT_GAS
      << "[output]\ncheck_interval = 1000000\n";
    auto solver = run_input(parse_toml(s.str()));
    ExactRiemann exact(1.0, 0.0, 1.0, 0.125, 0.0, 0.1, 1.4);
    const auto & m = *solver->get_mesh();
    double err = 0.0, vol = 0.0;
    for (uint32_t i = 0; i < m.n_cells; i++) {
        double rho, u, p;
        exact.sample((double(m.h_cell_coords(i, 0)) - 0.5) / double(solver->get_time()), rho, u, p);
        err += std::abs(double(solver->h_conservatives(i, 0)) - rho) * double(m.h_cell_volume(i));
        vol += double(m.h_cell_volume(i));
    }
    return err / vol;
}

/**
 * @brief Max density error of a density wave advected at (1, 0.5) across a
 *        periodic n x n box for a quarter period.
 */
double wave_error(const std::string & preset, uint32_t n) {
    std::ostringstream s;
    s << "[run]\nt_stop = 0.25\n"
      << "[mesh]\ntype = \"cartesian_tri\"\nNx = " << n << "\nNy = " << n
      << "\nLx = 1.0\nLy = 1.0\nperiodic = [\"x\", \"y\"]\n"
      << "[initialize]\ntype = \"analytical\"\nrho = \"1.0 + 0.2 * sin(2 * pi * x) * sin(2 * pi * y)\"\n"
      << "u = [\"1.0\", \"0.5\"]\np = \"1.0\"\n"
      << numerics(preset) << "[physics]\ntype = \"euler\"\n" << PERFECT_GAS
      << "[output]\ncheck_interval = 1000000\n";
    auto solver = run_input(parse_toml(s.str()));
    const auto exact = cell_averages(*solver->get_mesh(), [](double x, double y, double * U) {
        U[0] = 1.0 + 0.2 * std::sin(2.0 * M_PI * (x - 0.25)) * std::sin(2.0 * M_PI * (y - 0.125));
    });
    double err = 0.0;
    for (uint32_t i = 0; i < solver->get_mesh()->n_cells; i++) {
        err = std::max(err, std::abs(double(solver->h_conservatives(i, 0)) - double(exact(i, 0))));
    }
    return err;
}

/**
 * @brief Kinetic-energy decay of the viscous 2D Taylor-Green vortex at Mach 0.1
 *        and Re = U L / nu = 100 on n x n quads over t = 2, relative to the
 *        exact exp(-4 nu k^2 t): (log of the exact ratio) / (log of the computed
 *        one) - 1, the excess decay caused by the scheme.
 */
double taylor_green_excess_decay(const std::string & preset, uint32_t n) {
    const double nu = 1e-3, k = 2.0 * M_PI, t_end = 2.0;
    std::ostringstream s;
    s << "[run]\nt_stop = " << t_end << "\n"
      << "[mesh]\ntype = \"cartesian\"\nNx = " << n << "\nNy = " << n
      << "\nLx = 1.0\nLy = 1.0\nperiodic = [\"x\", \"y\"]\n"
      << "[initialize]\ntype = \"analytical\"\nrho = \"1.0\"\n"
      << "u = [\"0.1 * sin(2 * pi * x) * cos(2 * pi * y)\", \"-0.1 * cos(2 * pi * x) * sin(2 * pi * y)\"]\n"
      << "p = \"1 / 1.4 + 0.0025 * (cos(4 * pi * x) + cos(4 * pi * y))\"\n"
      << numerics(preset) << "[physics]\ntype = \"navier_stokes\"\nmu = " << nu << "\n" << "gamma = 1.4\n"
      << "p_ref = 0.7142857142857143\nT_ref = 1.0\nrho_ref = 1.0\n"
      << "[output]\ncheck_interval = 1000000\n";
    auto solver = std::make_unique<Solver>();
    solver->init(parse_toml(s.str()));
    const double ke0 = double(solver->integrate_flow_statistics()[0]);
    solver->run();
    const double ke1 = double(solver->integrate_flow_statistics()[0]);
    return std::log(ke1 / ke0) / (-4.0 * nu * k * k * t_end) - 1.0;
}

} // namespace

TEST(PresetValidation, RobustKeepsTheNohImplosionPositive) {
    // examples/noh_axisymmetric at 60 x 60 to t = 0.6: the shock is at r = 0.2
    // with rho = 64 behind it
    Input input = example_under_preset({"noh_axisymmetric/input.toml", "robust", 2, "", {}, ""}, 1, 60);
    input.as_table()["run"] = Input::table_type{{"t_stop", 0.6}};
    auto solver = run_input(input);
    const auto & m = *solver->get_mesh();
    double p_min = 1e30, rho_sum = 0.0, vol = 0.0;
    for (uint32_t i = 0; i < m.n_cells; i++) {
        p_min = std::min(p_min, double(solver->h_primitives(i, 2)));
        const double r = std::hypot(double(m.h_cell_coords(i, 0)), double(m.h_cell_coords(i, 1)));
        if (r > 0.05 && r < 0.15) {
            rho_sum += double(solver->h_conservatives(i, 0)) * double(m.h_cell_volume(i));
            vol += double(m.h_cell_volume(i));
        }
    }
    EXPECT_GT(p_min, 0.0);
    EXPECT_NEAR(rho_sum / vol, 64.0, 0.1 * 64.0);
}

TEST(PresetValidation, RobustResolvesSodsProblem) {
    const double e1 = sod_error("robust", 100);
    const double e2 = sod_error("robust", 200);
    EXPECT_LT(e2, 0.004);
    EXPECT_LT(e2, 0.7 * e1);
}

TEST(PresetValidation, AccurateConvergesAtHighOrderForSmoothFlow) {
    SKIP_IN_SINGLE_PRECISION("the error on the finer mesh is near the single-precision floor");
    const double e1 = wave_error("accurate", 16);
    const double e2 = wave_error("accurate", 32);
    EXPECT_GT(std::log2(e1 / e2), 4.0) << e1 << " " << e2;
    EXPECT_LT(e2, 0.05 * wave_error("robust", 32));
}

TEST(PresetValidation, LesLeavesTheTaylorGreenDecayToViscosity) {
    const double excess_les = taylor_green_excess_decay("les", 16);
    const double excess_robust = taylor_green_excess_decay("robust", 16);
    EXPECT_LT(std::abs(excess_les), 0.01) << excess_les;
    EXPECT_GT(excess_robust, 0.04) << excess_robust;
}

TEST(PresetValidation, ReactingPropagatesAPremixedFlameAtCanterasSpeed) {
    // Stoichiometric H2/air at 700 K and 1 atm (Cantera 3.2 FreeFlame,
    // mechanisms/h2o2.yaml, mixture-averaged: S_L = 9.7473 m/s, thermal
    // thickness 0.4775 mm, T_b = 2574 K), from a tanh profile to Cantera's
    // burnt state, burning toward a closed end through gas at rest: the
    // 1000 K isotherm moves at S_L. 6 cells per thermal thickness.
    const double S_L = 9.7473, delta = 4.775e-4, tau = delta / S_L;
    const int n = 60;
    const double dx = 10 * delta / n;
    const std::string s = "0.5 * (1 + tanh((x - 3.34e-3) / 1.2e-4))";
    std::ostringstream in;
    in << "[run]\nt_stop = " << 2 * tau << "\n"
       << "[mesh]\ntype = \"cartesian\"\nNx = " << n << "\nNy = 1\nLx = " << 10 * delta << "\nLy = " << 10 * dx << "\n"
       << "[initialize]\ntype = \"analytical\"\np = \"101325.0\"\nT = \"700 + 1873.8 * " << s << "\"\n"
       << "u = [\"0.0\", \"0.0\"]\nbalance = \"N2\"\n"
       << "Y = { H2 = \"0.0285224 - 0.0263594 * " << s << "\", O2 = \"0.2263540 - 0.2132528 * " << s << "\", "
       << "H2O = \"0.2276783 * " << s << "\", OH = \"0.0103755 * " << s << "\", O = \"0.0013623 * " << s << "\", "
       << "H = \"0.0002325 * " << s << "\" }\n"
       << "[[boundaries]]\nname = \"left\"\ntype = \"symmetry\"\n"
       << "[[boundaries]]\nname = \"right\"\ntype = \"p_out\"\np = 101325.0\n"
       << "[[boundaries]]\nname = \"bottom\"\ntype = \"symmetry\"\n"
       << "[[boundaries]]\nname = \"top\"\ntype = \"symmetry\"\n"
       << numerics("reacting")
       << "[physics]\ntype = \"navier_stokes\"\ngas = \"mixture\"\nmechanism = \"" << MALLARD_SOURCE_DIR
       << "/mechanisms/h2o2.yaml\"\nphase = \"ohmech\"\ntransport = \"mixture_averaged\"\n"
       << "[chemistry]\nrtol = 1.0e-6\natol = 1.0e-10\n"
       << "[output]\ncheck_interval = 1000000\n";
    Solver solver;
    solver.init(parse_toml(in.str()));
    auto flame_position = [&]() {
        solver.update_primitives();
        solver.copy_device_to_host();
        const auto & m = *solver.get_mesh();
        std::vector<std::pair<double, double>> profile;
        for (uint32_t i = 0; i < m.n_cells; i++) {
            profile.emplace_back(double(m.h_cell_coords(i, 0)), double(solver.h_primitives(i, N_DIM + 1)));
        }
        std::sort(profile.begin(), profile.end());
        for (size_t i = 1; i < profile.size(); i++) {
            const auto [x0, T0] = profile[i - 1];
            const auto [x1, T1] = profile[i];
            if (T0 < 1000.0 && T1 >= 1000.0) return x0 + (1000.0 - T0) * (x1 - x0) / (T1 - T0);
        }
        return -1.0;
    };
    std::vector<double> t, x;
    while (double(solver.get_time()) < 2 * tau * (1 - 1e-9)) {
        solver.calc_dt();
        solver.take_step();
        if (double(solver.get_time()) > 0.5 * tau && solver.get_step() % 50 == 0) {
            t.push_back(double(solver.get_time()));
            x.push_back(flame_position());
        }
    }
    // Least-squares slope of the position over the last 1.5 flame times
    const double n_t = double(t.size());
    double st = 0, sx = 0, stt = 0, stx = 0;
    for (size_t i = 0; i < t.size(); i++) {
        st += t[i];
        sx += x[i];
        stt += t[i] * t[i];
        stx += t[i] * x[i];
    }
    const double speed = -(n_t * stx - st * sx) / (n_t * stt - st * st);
    EXPECT_NEAR(speed, S_L, 0.03 * S_L);
}

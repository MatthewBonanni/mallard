/**
 * @file tfles_test.cpp
 * @author Matthew Bonanni (mbonanni001@gmail.com)
 * @brief Thickened flame model: sensor, thickening, efficiency and the scaled chemistry.
 * @version 0.1
 * @date 2026-10-05
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 *
 */

#include <gtest/gtest.h>
#include <Kokkos_Core.hpp>

#include <cmath>
#include <sstream>
#include <string>

#include "input.h"
#include "solver.h"
#include "test_fixtures.h"
#include "tfles.h"

namespace {

ThickenedFlame flame() {
    ThickenedFlame tf;
    tf.delta_L = 4e-4;
    tf.s_L = 2.0;
    tf.T_u = 300.0;
    tf.T_b = 2300.0;
    return tf;
}

} // namespace

TEST(ThickenedFlame, SensorCoversTheFlameAndVanishesOutside) {
    const ThickenedFlame tf = flame();
    EXPECT_EQ(tf.sensor(300.0), 0.0);
    EXPECT_EQ(tf.sensor(250.0), 0.0);
    EXPECT_EQ(tf.sensor(2300.0), 0.0);
    EXPECT_EQ(tf.sensor(2500.0), 0.0);
    // The whole preheat zone and reaction zone, c = 0.05 to 0.95
    for (const double c : {0.05, 0.2, 0.5, 0.8, 0.95}) EXPECT_DOUBLE_EQ(tf.sensor(300.0 + 2000.0 * c), 1.0) << c;
    EXPECT_GT(tf.sensor(300.0 + 2000.0 * 0.02), 0.0);
    EXPECT_LT(tf.sensor(300.0 + 2000.0 * 0.02), 1.0);
}

TEST(ThickenedFlame, ThickeningResolvesTheFlameOverNResCells) {
    const ThickenedFlame tf = flame();
    EXPECT_DOUBLE_EQ(tf.thickening(1e-3, 1.0), 5.0 * 1e-3 / 4e-4);
    EXPECT_DOUBLE_EQ(tf.thickening(1e-3, 0.0), 1.0);
    EXPECT_DOUBLE_EQ(tf.thickening(1e-3, 0.5), 1.0 + 0.5 * (12.5 - 1.0));
    // Already resolved: no thickening
    EXPECT_DOUBLE_EQ(tf.thickening(5e-5, 1.0), 1.0);
}

TEST(ThickenedFlame, EfficiencyIsOneWithoutSubgridTurbulenceAndSaturatesAtFToTheBeta) {
    ThickenedFlame tf = flame();
    const double nu = 1.6e-5;
    EXPECT_EQ(tf.wrinkling(10.0, 0.0, nu), 1.0);
    EXPECT_EQ(tf.wrinkling(1.0, 5.0, nu), 1.0);
    double last = 1.0;
    for (const double u : {0.2, 1.0, 5.0, 20.0}) {
        const double E = tf.wrinkling(10.0, u, nu);
        EXPECT_GT(E, last) << u;
        EXPECT_LE(E, std::sqrt(10.0) * (1.0 + 1e-12));
        last = E;
    }
    // Strong subgrid turbulence: the wrinkling of the scales between delta_L and F delta_L is complete
    EXPECT_NEAR(tf.wrinkling(10.0, 500.0, nu), std::sqrt(10.0), 1e-9);
    tf.efficiency = false;
    EXPECT_EQ(tf.wrinkling(10.0, 5.0, nu), 1.0);
}

namespace {

/** @brief A uniform reacting H2/air box: no gradients, so only the chemistry changes the state. */
std::string uniform_box(const std::string & run, const std::string & combustion) {
    std::ostringstream s;
    s << "[run]\n" << run << "[mesh]\ntype = \"cartesian\"\nNx = 4\nNy = 4\nLx = 0.04\nLy = 0.04\n"
      << (N_DIM == 3 ? "Nz = 3\nLz = 0.03\nperiodic = [\"x\", \"y\", \"z\"]\n" : "periodic = [\"x\", \"y\"]\n")
      << "[initialize]\ntype = \"analytical\"\np = \"101325.0\"\nT = \"1500.0\"\n"
      << (N_DIM == 3 ? "u = [\"0.0\", \"0.0\", \"0.0\"]\n" : "u = [\"0.0\", \"0.0\"]\n")
      << "X = { H2 = 2.0, O2 = 1.0, N2 = 3.76 }\n"
      << "[numerics]\nriemann_solver = \"HLLC\"\n[numerics.face_reconstruction]\ntype = \"FO\"\n"
      << "[physics]\ntype = \"navier_stokes\"\ngas = \"mixture\"\nmechanism = \"" MALLARD_SOURCE_DIR "/mechanisms/h2o2.yaml\"\n"
      << "[chemistry]\n[output]\ncheck_interval = 1000000\n[les]\n" << (N_DIM == 3 ? "model = \"sigma\"\n" : "model = \"wale\"\n")
      << combustion;
    return s.str();
}

} // namespace

TEST(ThickenedFlame, ReactionRatesAreDividedByTheThickening) {
    // Delta = 1 cm everywhere and n_res Delta / delta_L = 4, the sensor is 1
    // throughout the ignition (T_u = 0, T_b = 4000 K) and there is no
    // subgrid turbulence: the chemistry of each step is that of a step 4
    // times shorter without the model
    const std::string combustion = "[les.combustion]\nmodel = \"tfles\"\ndelta_L = 1.25e-2\ns_L = 2.0\n"
                                   "T_unburnt = 0.0\nT_burnt = 4000.0\nn_res = 5.0\n";
    Solver thick, plain;
    thick.init(parse_toml(uniform_box("n_steps = 60\ndt = 2e-6\n", combustion)));
    plain.init(parse_toml(uniform_box("n_steps = 60\ndt = 5e-7\n", "")));
    thick.run();
    plain.run();
    thick.copy_device_to_host();
    plain.copy_device_to_host();
    const uint32_t ns = thick.get_species_names().size();
    double change = 0.0, diff = 0.0;
    Solver initial;
    initial.init(parse_toml(uniform_box("n_steps = 0\ndt = 1e-6\n", "")));
    initial.copy_device_to_host();
    for (uint32_t k = 0; k < ns; k++) {
        change = std::max(change, std::abs(double(plain.h_species(0, k) - initial.h_species(0, k))));
        diff = std::max(diff, std::abs(double(thick.h_species(0, k) - plain.h_species(0, k))));
    }
    EXPECT_GT(change, 1e-4);  // the mixture reacts
    EXPECT_LT(diff, precision_tol<double>(1e-6, 1e-5) * change);
    EXPECT_NEAR(double(thick.get_time()), 4.0 * double(plain.get_time()), precision_tol<double>(1e-15, 1e-9));
}

TEST(ThickenedFlame, EddyViscosityVelocitySetsTheEfficiency) {
    // u' = C_u nu_t / Delta in Charlette's efficiency, cell by cell (a swirling field so that nu_t > 0; the
    // sensor is 1 throughout: T_u = 0, T_b = 4000 K at 1500 K)
    // One step of 1e-13 s: the fields of the step start and the coefficients of its end agree to ~1e-9
    std::string input = uniform_box("n_steps = 1\ndt = 1e-13\n",
                                    "[les.combustion]\nmodel = \"tfles\"\ndelta_L = 2e-3\ns_L = 2.0\nT_unburnt = 0.0\n"
                                    "T_burnt = 4000.0\nsubgrid_velocity = \"eddy_viscosity\"\nC_u = 26.0\n");
    const std::string still = N_DIM == 3 ? "u = [\"0.0\", \"0.0\", \"0.0\"]\n" : "u = [\"0.0\", \"0.0\"]\n";
    input.replace(input.find(still), still.size(),
                  N_DIM == 3 ? "u = [\"30 * sin(157.08 * y) * cos(78.54 * z)\", \"25 * sin(157.08 * z + 0.4) * cos(157.08 * x)\", "
                               "\"20 * sin(157.08 * x + 1.1)\"]\n"
                             : "u = [\"30 * sin(157.08 * y)\", \"25 * sin(157.08 * x + 0.4) + 10 * cos(157.08 * y)\"]\n");
    Solver solver;
    solver.init(parse_toml(input));
    solver.run();
    solver.copy_device_to_host();
    ThickenedFlame tf = flame();
    tf.delta_L = 2e-3;
    tf.T_u = 0.0;
    tf.T_b = 4000.0;
    const auto & fields = solver.get_tfles_fields();
    const auto & sgs = solver.get_les_coefficients();
    const auto & transport = solver.get_cell_transport();
    const auto m = solver.get_mesh();
    uint32_t turbulent = 0;
    for (uint32_t c = 0; c < m->n_owned(); c++) {
        const double rho = double(solver.h_conservatives(c, 0)), delta = std::pow(double(m->h_cell_volume(c)), 1.0 / N_DIM);
        const double F = double(fields(c, 0));
        const double u_prime = 26.0 * double(sgs(c, 0)) / (rho * delta);
        EXPECT_NEAR(double(fields(c, 1)), tf.wrinkling(F, u_prime, double(transport(c, 0)) / rho), 1e-7) << "cell " << c;
        turbulent += double(fields(c, 1)) > 1.01;
    }
    EXPECT_GT(turbulent, m->n_owned() / 2);
}

TEST(ThickenedFlame, InputErrors) {
    const std::string bad = "[les.combustion]\nmodel = \"tfles\"\ndelta_L = 1e-3\ns_L = 2.0\nT_unburnt = 2000.0\n"
                            "T_burnt = 300.0\n";
    Solver s;
    EXPECT_THROW(s.init(parse_toml(uniform_box("n_steps = 1\ndt = 1e-6\n", bad))), InputError);
    std::string no_chemistry = uniform_box("n_steps = 1\ndt = 1e-6\n",
                                           "[les.combustion]\nmodel = \"tfles\"\ndelta_L = 1e-3\ns_L = 2.0\n"
                                           "T_unburnt = 300.0\nT_burnt = 2000.0\n");
    no_chemistry.replace(no_chemistry.find("[chemistry]\n"), 12, "");
    Solver t;
    EXPECT_THROW(t.init(parse_toml(no_chemistry)), InputError);
}

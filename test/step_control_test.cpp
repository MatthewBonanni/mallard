/**
 * @file step_control_test.cpp
 * @author Matthew Bonanni (mbonanni001@gmail.com)
 * @brief The low-storage integrators and error-controlled time steps in full runs.
 * @version 0.1
 * @date 2026-10-15
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 *
 */

#include <gtest/gtest.h>
#include <Kokkos_Core.hpp>

#include <algorithm>
#include <cmath>
#include <sstream>
#include <string>
#include <vector>

#include "input.h"
#include "solver.h"
#include "test_fixtures.h"
#include "test_utils.h"

namespace {

/** @brief Isentropic vortex convected diagonally across a periodic box of 24 x 24 quads. */
std::string vortex_input(const std::string & integrator, const std::string & stop, const double cfl,
                         const std::string & extra = "") {
    std::ostringstream s;
    s << "[run]\n" << stop << "\ncfl = " << cfl << "\n"
      << "[mesh]\ntype = \"cartesian\"\nNx = 24\nNy = 24\nLx = 10.0\nLy = 10.0\nperiodic = [\"x\", \"y\"]\n"
      << "[initialize]\ntype = \"analytical\"\n"
      << "u = [\"1 - 0.7957747154594767 * exp(0.5 * (1 - (x - 5)^2 - (y - 5)^2)) * (y - 5)\", "
         "\"1 + 0.7957747154594767 * exp(0.5 * (1 - (x - 5)^2 - (y - 5)^2)) * (x - 5)\"]\n"
      << "rho = \"(1 - 0.09021827589374725 * exp(0.5 * (1 - (x - 5)^2 - (y - 5)^2))^2)^2.5\"\n"
      << "p = \"(1 - 0.09021827589374725 * exp(0.5 * (1 - (x - 5)^2 - (y - 5)^2))^2)^3.5\"\n"
      << "[numerics]\nriemann_solver = \"HLLC\"\ntime_integrator = \"" << integrator << "\"\ncheck_nan = true\n"
      << extra
      << "[numerics.face_reconstruction]\ntype = \"MUSCL\"\n"
      << "[physics]\ntype = \"euler\"\ngamma = 1.4\np_ref = 1.0\nT_ref = 1.0\nrho_ref = 1.0\n"
      << "[output]\ncheck_interval = 1000000\n";
    return s.str();
}

const std::string ERROR_CONTROL = "[numerics.error_control]\ntolerance = 1e-4\n";

std::vector<rtype> conservatives(Solver & solver) {
    solver.copy_device_to_host();
    std::vector<rtype> U;
    for (uint32_t c = 0; c < solver.get_mesh()->n_owned(); c++) {
        FOR_I_CONSERVATIVE U.push_back(solver.h_conservatives(c, i));
    }
    return U;
}

double max_rel_diff(const std::vector<rtype> & a, const std::vector<rtype> & b) {
    double m = 0.0;
    for (size_t k = 0; k < a.size(); k++) m = std::max(m, std::abs(double(a[k]) - double(b[k])) / (std::abs(double(b[k])) + 1e-3));
    return m;
}

} // namespace

TEST(TimeStepTest, LowStorageIntegratorsMatchWithAndWithoutStepGraphs) {
    // The captured step reads dt and the controller's decisions from the
    // device; it must take bitwise the steps of the kernel-by-kernel path
    for (const auto & [integrator, extra] : std::vector<std::pair<std::string, std::string>>{
             {"SSPRK54", ""}, {"CK45", ""}, {"RDPK3Sp35", ""}, {"RDPK3Sp35", ERROR_CONTROL}}) {
        std::vector<rtype> U[2];
        rtype t[2];
        for (const int graphs : {0, 1}) {
            Solver solver;
            const std::string stop = std::string("n_steps = 25\ncuda_graphs = ") + (graphs ? "true" : "false");
            solver.init(parse_toml(vortex_input(integrator, stop, 1.0, extra)));
            solver.run();
            U[graphs] = conservatives(solver);
            t[graphs] = solver.get_time();
        }
        EXPECT_EQ(t[0], t[1]) << integrator << extra;
        EXPECT_EQ(max_rel_diff(U[1], U[0]), 0.0) << integrator << extra;
    }
}

TEST(TimeStepTest, ErrorControlRecoversFromAnUnstableFirstStep) {
    // A first step at 20 times the stability limit: rejected and retried with
    // smaller steps until the error is within the tolerance
    Solver controlled;
    controlled.init(parse_toml(vortex_input("RDPK3Sp35", "t_stop = 2.0", 20.0, ERROR_CONTROL)));
    controlled.run();
    EXPECT_GT(controlled.get_rejected_steps(), 0u);
    EXPECT_EQ(controlled.get_time(), 2.0_r);

    Solver reference;
    reference.init(parse_toml(vortex_input("SSPRK54", "t_stop = 2.0", 0.5)));
    reference.run();
    // About the time error of tolerance 1e-4 over these 2 time units (0.7%); a blown-up step would be O(1)
    EXPECT_LT(max_rel_diff(conservatives(controlled), conservatives(reference)), 1e-2);

    Solver uncontrolled;
    uncontrolled.init(parse_toml(vortex_input("RDPK3Sp35", "t_stop = 2.0", 20.0)));
    EXPECT_ANY_THROW(uncontrolled.run());
}

TEST(TimeStepTest, TighterToleranceGivesSmallerTimeError) {
    SKIP_IN_SINGLE_PRECISION("a tolerance of 1e-6 is below single-precision resolution");
    Solver reference;
    reference.init(parse_toml(vortex_input("SSPRK54", "t_stop = 2.0", 0.05)));
    reference.run();
    const std::vector<rtype> U_ref = conservatives(reference);
    double error[2];
    uint64_t steps[2];
    for (const int k : {0, 1}) {
        Solver solver;
        solver.init(parse_toml(vortex_input("RDPK3Sp35", "t_stop = 2.0", 1.0,
                                            std::string("[numerics.error_control]\ntolerance = ") +
                                                (k ? "1e-6" : "1e-4") + "\n")));
        solver.run();
        error[k] = max_rel_diff(conservatives(solver), U_ref);
        steps[k] = solver.get_step();
    }
    EXPECT_GT(steps[1], steps[0]);
    EXPECT_LT(error[1], 0.1 * error[0]);
}

TEST(TimeStepTest, ErrorControlNeedsAnEmbeddedPair) {
    Solver solver;
    EXPECT_THROW(solver.init(parse_toml(vortex_input("SSPRK54", "n_steps = 1", 1.0, ERROR_CONTROL))), InputError);
}

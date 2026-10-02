/**
 * @file restart_test.cpp
 * @author Matthew Bonanni (mbonanni001@gmail.com)
 * @brief Restart files reproduce an uninterrupted run exactly.
 * @version 0.1
 * @date 2026-10-02
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 *
 */

#include <gtest/gtest.h>
#include <Kokkos_Core.hpp>

#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>

#include "test_fixtures.h"
#include "solver.h"

namespace {

std::string restart_input(const std::string & dir, const std::string & init, uint32_t n_steps) {
    std::ostringstream s;
    s << "[run]\nn_steps = " << n_steps << "\ncfl = 0.5\n"
      << "[mesh]\ntype = \"cartesian_tri\"\nNx = 12\nNy = 10\nLx = 1.0\nLy = 1.0\n"
      << "[initialize]\n" << init
      << "[[boundaries]]\nname = \"left\"\ntype = \"extrapolation\"\n"
      << "[[boundaries]]\nname = \"right\"\ntype = \"extrapolation\"\n"
      << "[[boundaries]]\nname = \"top\"\ntype = \"symmetry\"\n"
      << "[[boundaries]]\nname = \"bottom\"\ntype = \"wall_adiabatic\"\n"
      << "[numerics]\nriemann_solver = \"HLLC\"\ntime_integrator = \"SSPRK3\"\n"
      << "[numerics.face_reconstruction]\ntype = \"MUSCL\"\n"
      << "[physics]\ntype = \"euler\"\ngamma = 1.4\np_ref = 1.0\nT_ref = 1.0\nrho_ref = 1.0\n"
      << "[output]\ncheck_interval = 1000000\n"
      << "[[write_data]]\nprefix = \"" << dir << "/restart\"\nformat = \"restart\"\ninterval = 20\n"
      << "[[write_data]]\nprefix = \"" << dir << "/flow\"\nformat = \"vtu\"\ntime_interval = 0.002\n"
      << "variables = [\"RHO\"]\n";
    return s.str();
}

const std::string BLAST =
    "type = \"analytical\"\n"
    "rho = \"1.0 + (x < 0.4 ? 1.0 : 0.0)\"\nu = [\"0.2\", \"0.1\"]\np = \"x < 0.4 ? 2.0 : 1.0\"\n";

} // namespace

TEST(RestartTest, RestartedRunMatchesUninterruptedRunExactly) {
    const std::string dir = (std::filesystem::temp_directory_path() / "mallard_restart_test").string();
    std::filesystem::remove_all(dir);

    Solver straight;
    straight.init(parse_toml(restart_input(dir + "/a", BLAST, 40)));
    straight.run();
    straight.copy_device_to_host();

    Solver first;
    first.init(parse_toml(restart_input(dir + "/b", BLAST, 20)));
    first.run();
    Solver second;
    second.init(parse_toml(restart_input(dir + "/b", "type = \"restart\"\nfile = \"" + dir + "/b/restart_000020.restart\"\n", 40)));
    EXPECT_EQ(second.get_step(), 20u);
    second.run();
    second.copy_device_to_host();

    ASSERT_EQ(second.get_step(), straight.get_step());
    EXPECT_EQ(second.get_time(), straight.get_time());
    for (uint32_t i = 0; i < straight.get_mesh()->n_cells; i++) {
        for (uint8_t v = 0; v < N_CONSERVATIVE; v++) EXPECT_EQ(second.h_conservatives(i, v), straight.h_conservatives(i, v));
    }

    // The time series continues without duplicated or missing snapshots
    auto count_entries = [](const std::string & pvd) {
        std::ifstream in(pvd);
        std::string line;
        int n = 0;
        while (std::getline(in, line)) n += line.find("<DataSet") != std::string::npos;
        return n;
    };
    EXPECT_EQ(count_entries(dir + "/b/flow.pvd"), count_entries(dir + "/a/flow.pvd"));
    std::filesystem::remove_all(dir);
}

TEST(RestartTest, RejectsMismatchedMesh) {
    const std::string dir = (std::filesystem::temp_directory_path() / "mallard_restart_mismatch").string();
    std::filesystem::remove_all(dir);
    Solver first;
    first.init(parse_toml(restart_input(dir, BLAST, 20)));
    first.run();
    std::string input = restart_input(dir, "type = \"restart\"\nfile = \"" + dir + "/restart_000020.restart\"\n", 40);
    input.replace(input.find("Nx = 12"), 7, "Nx = 13");
    Solver second;
    EXPECT_THROW(second.init(parse_toml(input)), std::runtime_error);
    std::filesystem::remove_all(dir);
}

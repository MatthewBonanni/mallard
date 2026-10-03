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
#include <vector>

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
      << "[[write_data]]\nprefix = \"" << dir << "/flow\"\nformat = \"vtu\"\ntime_interval = 0.005\n"
      << "variables = [\"RHO\"]\n"
      << "[[forces]]\nzone = \"bottom\"\ninterval = 5\nfile = \"" << dir << "/forces.csv\"\n";
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
    const double t_stop_first = first.get_time();
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
    // The time series continues: the off-grid snapshot written when the first
    // run stopped is kept, and no file is listed (or written) twice
    std::ifstream pvd(dir + "/b/flow.pvd");
    std::string line;
    std::vector<std::pair<double, std::string>> entries;
    while (std::getline(pvd, line)) {
        const size_t a = line.find("timestep=\""), b = line.find("file=\"");
        if (a == std::string::npos) continue;
        entries.emplace_back(std::stod(line.substr(a + 10)), line.substr(b + 6, line.find('"', b + 6) - b - 6));
    }
    EXPECT_EQ(entries.size(), count_entries(dir + "/a/flow.pvd") + 1u);
    std::string stop_file;
    for (size_t i = 0; i < entries.size(); i++) {
        if (entries[i].first == t_stop_first) stop_file = entries[i].second;
        if (i > 0) {
            EXPECT_LT(entries[i - 1].first, entries[i].first);
            EXPECT_NE(entries[i - 1].second, entries[i].second);
        }
    }
    ASSERT_FALSE(stop_file.empty());
    // ... and still holds the solution at that time
    std::ifstream in(dir + "/b/" + stop_file);
    std::string header(4096, '\0');
    in.read(header.data(), header.size());
    const size_t k = header.find(">", header.find("Name=\"TIME\"")) + 1;
    EXPECT_EQ(std::stod(header.substr(k)), t_stop_first);

    // The force history is appended to, not overwritten
    auto read_all = [](const std::string & file) {
        std::ifstream stream(file);
        std::stringstream ss;
        ss << stream.rdbuf();
        return ss.str();
    };
    EXPECT_EQ(read_all(dir + "/b/forces.csv"), read_all(dir + "/a/forces.csv"));
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

TEST(RestartTest, RejectsZeroForceInterval) {
    std::string input = restart_input("unused", BLAST, 1);
    input.replace(input.find("interval = 5"), 12, "interval = 0");
    Solver solver;
    EXPECT_THROW(solver.init(parse_toml(input)), std::runtime_error);
}

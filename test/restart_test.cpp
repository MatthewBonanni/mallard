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

#include <algorithm>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <sstream>
#include <string>
#include <vector>

#include "test_fixtures.h"
#include "solver.h"

namespace {

const std::string PERFECT_GAS = "[physics]\ntype = \"euler\"\ngamma = 1.4\np_ref = 1.0\nT_ref = 1.0\nrho_ref = 1.0\n";

std::string restart_input(const std::string & dir, const std::string & init, uint32_t n_steps,
                          const std::string & physics = PERFECT_GAS) {
    std::ostringstream s;
    s << "[run]\nn_steps = " << n_steps << "\ncfl = 0.25\n"
      << "[mesh]\ntype = \"cartesian_tri\"\nNx = 12\nNy = 10\nLx = 1.0\nLy = 1.0\n"
      << "[initialize]\n" << init
      << "[[boundaries]]\nname = \"left\"\ntype = \"extrapolation\"\n"
      << "[[boundaries]]\nname = \"right\"\ntype = \"extrapolation\"\n"
      << "[[boundaries]]\nname = \"top\"\ntype = \"symmetry\"\n"
      << "[[boundaries]]\nname = \"bottom\"\ntype = \"wall_adiabatic\"\n"
      << "[numerics]\nriemann_solver = \"HLLC\"\ntime_integrator = \"SSPRK3\"\n"
      << "[numerics.face_reconstruction]\ntype = \"MUSCL\"\n"
      << physics
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

// Hot stoichiometric H2/air igniting next to cold gas
const std::string IGNITION =
    "type = \"analytical\"\np = \"101325.0\"\nT = \"x < 0.5 ? 1200.0 : 300.0\"\nu = [\"0.0\", \"0.0\"]\n"
    "X = { H2 = \"2\", O2 = \"1\", N2 = \"3.76\" }\n";
const std::string REACTING = "[physics]\ntype = \"euler\"\ngas = \"mixture\"\nmechanism = \"" MALLARD_SOURCE_DIR
                             "/mechanisms/h2o2.yaml\"\n[chemistry]\n";

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
    const double t_stop_first = double(first.get_time());
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

TEST(RestartTest, ReactingRestartedRunMatchesUninterruptedRunExactly) {
    // Species, the chemistry's last sub-steps and the temperature seeds carry
    // over, so restarted cells integrate exactly as in one run
    const std::string dir = (std::filesystem::temp_directory_path() / "mallard_reacting_restart_test").string();
    std::filesystem::remove_all(dir);

    Solver straight;
    straight.init(parse_toml(restart_input(dir + "/a", IGNITION, 40, REACTING)));
    straight.run();
    straight.copy_device_to_host();

    Solver first;
    first.init(parse_toml(restart_input(dir + "/b", IGNITION, 20, REACTING)));
    first.run();
    Solver second;
    second.init(parse_toml(restart_input(
        dir + "/b", "type = \"restart\"\nfile = \"" + dir + "/b/restart_000020.restart\"\n", 40, REACTING)));
    second.run();
    second.copy_device_to_host();

    ASSERT_EQ(second.get_step(), straight.get_step());
    EXPECT_EQ(second.get_time(), straight.get_time());
    const uint32_t ns = static_cast<uint32_t>(straight.get_species_names().size());
    ASSERT_GT(ns, 0u);
    for (uint32_t i = 0; i < straight.get_mesh()->n_cells; i++) {
        for (uint8_t v = 0; v < N_CONSERVATIVE; v++) EXPECT_EQ(second.h_conservatives(i, v), straight.h_conservatives(i, v));
        for (uint32_t k = 0; k < ns; k++) EXPECT_EQ(second.h_species(i, k), straight.h_species(i, k));
    }
    std::filesystem::remove_all(dir);
}

TEST(RestartTest, PeakPressureIsARunningMaximumThatSurvivesRestarts) {
    const std::string dir = (std::filesystem::temp_directory_path() / "mallard_restart_p_max").string();
    std::filesystem::remove_all(dir);
    auto input = [&](const std::string & sub, const std::string & init, uint32_t n_steps) {
        std::string s = restart_input(dir + sub, init, n_steps);
        s.replace(s.find("variables = [\"RHO\"]"), 19, "variables = [\"RHO\", \"P_MAX\"]");
        return parse_toml(s);
    };

    Solver straight;
    straight.init(input("/a", BLAST, 40));
    straight.run();
    straight.copy_device_to_host();

    Solver first;
    first.init(input("/b", BLAST, 20));
    first.run();
    Solver second;
    second.init(input("/b", "type = \"restart\"\nfile = \"" + dir + "/b/restart_000020.restart\"\n", 40));
    second.run();
    second.copy_device_to_host();

    const auto & mesh = *straight.get_mesh();
    uint32_t n_decayed = 0;
    for (uint32_t i = 0; i < mesh.n_cells; i++) {
        const rtype p = straight.h_primitives(i, N_DIM);
        EXPECT_GE(straight.h_p_max(i), p);
        if (mesh.h_cell_coords(i, 0) < 0.3_r) {
            EXPECT_GE(straight.h_p_max(i), 2.0_r);
        }
        n_decayed += straight.h_p_max(i) > p + 0.1_r;
        EXPECT_EQ(second.h_p_max(i), straight.h_p_max(i));
    }
    // The blast's rarefaction has lowered the pressure below its peak
    EXPECT_GT(n_decayed, 0u);
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

namespace {

/** @brief A version 2 restart file split into its header fields, names and value blocks. */
struct RestartFile {
    std::string prefix;  // magic through time
    std::vector<std::string> names;
    std::vector<std::string> blocks;

    explicit RestartFile(const std::string & path) {
        std::ifstream in(path, std::ios::binary);
        const std::string bytes((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
        constexpr size_t PREFIX = 16 + 4 + 4 + 8 + 8 + 8 + 8;
        prefix = bytes.substr(0, PREFIX);
        uint64_t n_cells, n_vars;
        std::memcpy(&n_cells, bytes.data() + 24, sizeof(n_cells));
        std::memcpy(&n_vars, bytes.data() + 32, sizeof(n_vars));
        size_t pos = PREFIX;
        for (uint64_t v = 0; v < n_vars; v++) {
            uint32_t length;
            std::memcpy(&length, bytes.data() + pos, sizeof(length));
            names.push_back(bytes.substr(pos + 4, length));
            pos += 4 + length;
        }
        for (uint64_t v = 0; v < n_vars; v++) {
            blocks.push_back(bytes.substr(pos, n_cells * sizeof(rtype)));
            pos += n_cells * sizeof(rtype);
        }
    }

    void write(const std::string & path, uint32_t version) const {
        std::string bytes = prefix;
        std::memcpy(bytes.data() + 16, &version, sizeof(version));
        const uint64_t n_vars = names.size();
        std::memcpy(bytes.data() + 32, &n_vars, sizeof(n_vars));
        if (version >= 2) {
            for (const auto & name : names) {
                const uint32_t length = name.size();
                bytes.append(reinterpret_cast<const char *>(&length), sizeof(length));
                bytes += name;
            }
        }
        for (const auto & block : blocks) bytes += block;
        std::ofstream(path, std::ios::binary) << bytes;
    }
};

} // namespace

TEST(RestartTest, ReadsVersion1FilesAndMapsVersion2VariablesByName) {
    const std::string dir = (std::filesystem::temp_directory_path() / "mallard_restart_versions").string();
    std::filesystem::remove_all(dir);
    Solver straight;
    straight.init(parse_toml(restart_input(dir + "/a", BLAST, 40)));
    straight.run();
    straight.copy_device_to_host();
    Solver first;
    first.init(parse_toml(restart_input(dir + "/b", BLAST, 20)));
    first.run();
    const RestartFile file(dir + "/b/restart_000020.restart");
    ASSERT_EQ(file.names, std::vector<std::string>(CONSERVATIVE_NAMES.begin(), CONSERVATIVE_NAMES.end()));

    // Version 1: the flow block without names
    file.write(dir + "/v1.restart", 1);
    // Version 2 with the variables in reverse order
    RestartFile reversed = file;
    std::reverse(reversed.names.begin(), reversed.names.end());
    std::reverse(reversed.blocks.begin(), reversed.blocks.end());
    reversed.write(dir + "/reversed.restart", 2);
    for (const std::string name : {"v1", "reversed"}) {
        Solver second;
        second.init(parse_toml(restart_input(dir + "/c", "type = \"restart\"\nfile = \"" + dir + "/" + name +
                                                             ".restart\"\n", 40)));
        second.run();
        second.copy_device_to_host();
        for (uint32_t i = 0; i < straight.get_mesh()->n_cells; i++) {
            for (uint8_t v = 0; v < N_CONSERVATIVE; v++) {
                ASSERT_EQ(second.h_conservatives(i, v), straight.h_conservatives(i, v)) << name;
            }
        }
    }

    // A species the run does not transport is an error, not silently dropped
    RestartFile with_species = file;
    with_species.names.push_back("RHOY_H2");
    with_species.blocks.push_back(file.blocks[0]);
    with_species.write(dir + "/species.restart", 2);
    Solver third;
    try {
        third.init(parse_toml(restart_input(dir + "/d", "type = \"restart\"\nfile = \"" + dir + "/species.restart\"\n", 40)));
        ADD_FAILURE() << "a restart file with species was accepted by a single-gas run";
    } catch (const std::runtime_error & e) {
        EXPECT_NE(std::string(e.what()).find("RHOY_H2"), std::string::npos) << e.what();
    }
    std::filesystem::remove_all(dir);
}

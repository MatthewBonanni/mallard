/**
 * @file post_extract_test.cpp
 * @brief mallard post extract: probes of existing output are the run's own probes at those steps.
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 */

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

#include "cli.h"
#include "comm.h"
#include "commands.h"
#include "data_writer.h"
#include "memory_tracker.h"
#include "mesh_block.h"
#include "scratch.h"
#include "solver.h"
#include "test_fixtures.h"

constexpr double ROUNDING = std::max(1e-12, 4.0 * double(std::numeric_limits<rtype>::epsilon()));

namespace {

namespace fs = std::filesystem;

std::string fresh_dir(const std::string & name) {
    const std::string dir = (scratch_dir() / ("extract_" + name)).string();
    comm::barrier();
    if (comm::is_root()) fs::remove_all(dir);
    comm::barrier();
    return dir;
}

bool vtkhdf() { return have_hdf5() && (comm::size() == 1 || have_parallel_hdf5()); }

const std::string BOX = N_DIM == 3 ? "{type = \"box\", center = [0.5, 0.5, 0.5], size = [0.4, 0.2, 0.6]}"
                                   : "{type = \"rectangle\", center = [0.5, 0.5], size = [0.4, 0.2]}";

/** @brief A periodic box with a curve probe of point values, a region probe and restart files and snapshots every 2 steps. */
std::string case_input(const std::string & dir, uint32_t n) {
    std::ostringstream s;
    s << "[run]\nn_steps = 4\ndt = 1.0e-3\n[mesh]\ntype = \"cartesian\"\nNx = " << n << "\nNy = " << n
      << "\nLx = 1.0\nLy = 1.0\n";
    if (N_DIM == 3) s << "Nz = 4\nLz = 1.0\n[[periodic]]\nzones = [\"back\", \"front\"]\n";
    s << "[[periodic]]\nzones = [\"left\", \"right\"]\n[[periodic]]\nzones = [\"bottom\", \"top\"]\n"
      << "[initialize]\ntype = \"analytical\"\nrho = \"1.0 + 0.2 * sin(2 * pi * x) * sin(2 * pi * y)\"\n"
      << "u = " << (N_DIM == 3 ? "[\"0.3\", \"-0.1\", \"0.2\"]" : "[\"0.3\", \"-0.1\"]") << "\np = \"1.0\"\n"
      << "[numerics]\nriemann_solver = \"HLLC\"\ntime_integrator = \"SSPRK3\"\n"
      << "[numerics.face_reconstruction]\ntype = \"MUSCL\"\n"
      << "[physics]\ntype = \"euler\"\ngamma = 1.4\np_ref = 1.0\nT_ref = 1.0\nrho_ref = 1.0\n"
      << "[output]\ncheck_interval = 1000000\nrun_log = \"\"\n"
      << "[[probes]]\nname = \"line\"\ntype = \"line\"\nstart = " << (N_DIM == 3 ? "[0.1, 0.3, 0.5]" : "[0.1, 0.3]")
      << "\nend = " << (N_DIM == 3 ? "[0.9, 0.6, 0.5]" : "[0.9, 0.6]")
      << "\nn_points = 7\nfields = [\"RHO\", \"U\", \"P\"]\nsampling = \"point\"\ninterval = 2\n"
      << "file = \"" << dir << "/probe_line.csv\"\n"
      << "[[probes]]\nname = \"box\"\nregion = " << BOX << "\nfields = [\"RHO\", \"T\", \"P\"]\n"
      << "time_interval = 0.002\nreductions_file = \"" << dir << "/probe_box_reductions.csv\"\n"
      << "[[write_data]]\nprefix = \"" << dir << "/restart\"\nformat = \"restart\"\ninterval = 2\n";
    if (vtkhdf()) {
        s << "[[write_data]]\nprefix = \"" << dir << "/flow\"\nformat = \"vtkhdf\"\nfields = [\"RHO\", \"U\", \"P\"]\n"
          << "interval = 2\n";
    }
    return s.str();
}

/** @brief Runs the case; afterwards every rank can read its files. */
std::string run_dir() {
    const std::string dir = fresh_dir("run");
    Solver solver;
    solver.init(parse_toml(case_input(dir, 16)));
    solver.run();
    comm::barrier();
    return dir;
}

/** @brief The data rows of a CSV file, as text. */
std::vector<std::string> rows(const std::string & file) {
    std::ifstream in(file);
    std::vector<std::string> out;
    std::string line;
    std::getline(in, line);
    while (std::getline(in, line)) {
        if (!line.empty() && line[0] != '#') out.push_back(line);
    }
    return out;
}

/** @brief The rows of the run's probe file from step 2 on: those of the snapshots after the first. */
std::vector<std::string> after_step_0(const std::string & file) {
    std::vector<std::string> out;
    for (const std::string & r : rows(file)) {
        if (!r.starts_with("0,")) out.push_back(r);
    }
    return out;
}

/** @brief Runs mallard post extract; errors, if any, of rank 0 in errors. */
int extract(const std::vector<std::string> & words, std::string * errors = nullptr) {
    cli::Args args;
    std::string error;
    EXPECT_TRUE(cli::parse(*cli::command("post extract"), words, args, error)) << error;
    std::ostringstream report, messages;
    std::streambuf * out = std::cout.rdbuf(report.rdbuf());
    std::streambuf * err = std::cerr.rdbuf(messages.rdbuf());
    const int status = post_extract(args);
    std::cout.rdbuf(out);
    std::cerr.rdbuf(err);
    if (errors) *errors = messages.str();
    comm::barrier();
    return status;
}

std::vector<double> numbers(const std::string & row) {
    std::vector<double> out;
    std::stringstream s(row);
    std::string cell;
    while (std::getline(s, cell, ',')) out.push_back(std::stod(cell));
    return out;
}

} // namespace

TEST(PostExtractTest, RestartFilesGiveTheRunsProbesBitForBit) {
    const std::string run = run_dir();
    const std::string out = fresh_dir("restart");
    ASSERT_EQ(extract({run + "/restart_000000.restart", run + "/restart_000002.restart", run + "/restart_000004.restart",
                       "--steps", "2,4", "-o", out}),
              0);
    EXPECT_EQ(rows(out + "/probe_line.csv"), after_step_0(run + "/probe_line.csv"));
    EXPECT_EQ(rows(out + "/probe_box_reductions.csv"), after_step_0(run + "/probe_box_reductions.csv"));
    EXPECT_FALSE(fs::exists(out + "/restart_000002.restart"));

    // The input's region, given on the command line, is the same probe
    const std::string region = fresh_dir("region");
    ASSERT_EQ(extract({run + "/restart_000004.restart", "--region", BOX, "--fields", "RHO,T,P", "--name", "b", "-o",
                       region}),
              0);
    EXPECT_EQ(rows(region + "/probe_b_reductions.csv"), std::vector<std::string>{rows(run + "/probe_box_reductions.csv").back()});
}

TEST(PostExtractTest, SnapshotsOfPrimitivesGiveTheRunsProbesToRounding) {
    if (!vtkhdf()) GTEST_SKIP() << "VTKHDF snapshots need HDF5, parallel HDF5 on several ranks";
    const std::string run = run_dir();
    const std::string out = fresh_dir("snapshots");
    ASSERT_EQ(extract({run + "/flow.vtkhdf.series", "--times", "0.001:0.005", "--probe", "line", "-o", out}), 0);
    const std::vector<std::string> expected = after_step_0(run + "/probe_line.csv");
    const std::vector<std::string> got = rows(out + "/probe_line.csv");
    ASSERT_EQ(got.size(), expected.size());
    EXPECT_FALSE(fs::exists(out + "/probe_box_reductions.csv"));
    for (size_t i = 0; i < got.size(); i++) {
        const std::vector<double> a = numbers(got[i]), b = numbers(expected[i]);
        ASSERT_EQ(a.size(), b.size());
        // The step each snapshot's block records
        EXPECT_EQ(a[0], b[0]) << got[i];
        for (size_t c = 1; c < a.size(); c++) EXPECT_NEAR(a[c], b[c], ROUNDING * (1.0 + std::abs(b[c]))) << got[i];
    }
}

TEST(PostExtractTest, StepsAndTimesAreThoseTheFilesRecordNotTheirNames) {
    const std::string run = run_dir();
    const std::string renamed = fresh_dir("renamed");
    if (comm::is_root()) fs::create_directories(renamed);
    std::vector<std::string> files = {"restart_000004.restart"};
    if (vtkhdf()) files.push_back("flow_000004.vtkhdf");
    std::string at_step_4;
    for (const std::string & r : rows(run + "/probe_line.csv")) {
        if (r.starts_with("4,")) at_step_4 = r;
    }
    ASSERT_FALSE(at_step_4.empty());
    for (const std::string & file : files) {
        const std::string misnamed = renamed + "/" + file.substr(0, file.find('_')) + "_000099" + fs::path(file).extension().string();
        if (comm::is_root()) fs::copy_file(run + "/" + file, misnamed, fs::copy_options::overwrite_existing);
        comm::barrier();
        const std::string out = fresh_dir("renamed_out_" + fs::path(file).extension().string().substr(1));
        ASSERT_EQ(extract({misnamed, "--probe", "line", "-o", out}), 0) << file;
        const std::vector<std::string> got = rows(out + "/probe_line.csv");
        ASSERT_FALSE(got.empty()) << file;
        const std::vector<double> a = numbers(got[0]), b = numbers(at_step_4);
        EXPECT_EQ(a[0], 4.0) << file;
        EXPECT_EQ(a[1], b[1]) << file;
    }
}

TEST(PostExtractTest, FileOfAnotherMeshIsAnError) {
    const std::string run = run_dir();
    const std::string dir = fresh_dir("coarse");
    if (comm::is_root()) {
        fs::create_directories(dir);
        std::ofstream(dir + "/input.toml") << case_input(dir, 8);
    }
    comm::barrier();
    std::string errors;
    EXPECT_EQ(extract({run + "/restart_000002.restart", "-i", dir + "/input.toml", "-o", dir + "/out"}, &errors), 1);
    if (comm::is_root()) {
        EXPECT_NE(errors.find(" cells, but the case's mesh has "), std::string::npos) << errors;
    }
    EXPECT_TRUE(!fs::exists(dir + "/out/probe_line.csv") || rows(dir + "/out/probe_line.csv").empty());
}

TEST(PostExtractTest, TimeAveragedProbesAreSkippedOrRefused) {
    const std::string run = run_dir();
    const std::string dir = fresh_dir("averaged");
    if (comm::is_root()) {
        fs::create_directories(dir);
        std::ofstream(dir + "/input.toml") << case_input(run, 16) << "[statistics]\n"
                                           << "[[probes]]\nname = \"mean\"\ntype = \"line\"\nstart = "
                                           << (N_DIM == 3 ? "[0.1, 0.3, 0.5]" : "[0.1, 0.3]") << "\nend = "
                                           << (N_DIM == 3 ? "[0.9, 0.6, 0.5]" : "[0.9, 0.6]")
                                           << "\nn_points = 3\nfields = [\"P\"]\naverage = true\n";
    }
    comm::barrier();
    const std::string restart = run + "/restart_000002.restart";
    ASSERT_EQ(extract({restart, "-i", dir + "/input.toml", "-o", dir + "/all"}), 0);
    EXPECT_EQ(rows(dir + "/all/probe_line.csv").size(), 7u);
    EXPECT_FALSE(fs::exists(dir + "/all/probe_mean.csv"));
    std::string errors;
    EXPECT_EQ(extract({restart, "-i", dir + "/input.toml", "--probe", "mean", "-o", dir + "/mean"}, &errors), 2);
    if (comm::is_root()) {
        EXPECT_NE(errors.find("averages in time"), std::string::npos) << errors;
    }
}

TEST(PostExtractTest, OutputOfStatesHoldsNoTimeSteppingBuffers) {
    const std::string run = run_dir();
    const std::string dir = fresh_dir("buffers");
    const std::vector<std::string> prefixes = {"face_solution", "face_flux", "solution", "rhs", "last_good"};
    auto held = [&](const bool from_file) {
        memory::Tracker tracker;
        Solver solver;
        if (from_file) {
            solver.set_state_reader([&](const Mesh & mesh) {
                return read_restart(run + "/restart_000002.restart",
                                    mesh.h_global_cell_id.empty() ? nullptr : &mesh.h_global_cell_id);
            });
        }
        solver.init(parse_toml(case_input(dir, 16)));
        std::vector<std::string> labels;
        for (const bool device : {true, false}) {
            for (const auto & [label, bytes] : tracker.live(device)) {
                for (const std::string & prefix : prefixes) {
                    if (bytes > 0 && label.starts_with(prefix) && memory::category(label) != memory::Category::MESH) {
                        labels.push_back(label);
                    }
                }
            }
        }
        return labels;
    };
    EXPECT_FALSE(held(false).empty());
    EXPECT_EQ(held(true), std::vector<std::string>{});
}

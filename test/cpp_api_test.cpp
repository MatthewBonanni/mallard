/**
 * @file cpp_api_test.cpp
 * @brief The C++ library API (include/mallard/mallard.h): a case run through it matches the run of
 *        `mallard -i` bit for bit, and its input stays the run's record.
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 */

#include <gtest/gtest.h>

#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "mallard/mallard.h"

#include "comm.h"
#include "data_writer.h"
#include "json.h"
#include "solver.h"

namespace {

namespace fs = std::filesystem;

/** @brief A small shock tube with integrals, a region probe and a restart file at the end, its output in dir. */
std::string case_text(const std::string & dir, const std::string & run, int interval = 1) {
    std::ostringstream s;
    s << "[run]\n" << run << "cfl = 0.4\n";
#if Mallard_DIM == 2
    s << "[mesh]\ntype = \"cartesian_tri\"\nNx = 12\nNy = 10\nLx = 1.0\nLy = 1.0\n";
    const char * zones[] = {"left", "right", "top", "bottom"};
    const char * box = "{type = \"rectangle\", center = [0.5, 0.5], size = [0.5, 0.4]}";
#else
    s << "[mesh]\ntype = \"cartesian\"\nNx = 6\nNy = 5\nNz = 4\nLx = 1.0\nLy = 1.0\nLz = 1.0\n";
    const char * zones[] = {"left", "right", "bottom", "top", "back", "front"};
    const char * box = "{type = \"box\", center = [0.5, 0.5, 0.5], size = [0.5, 0.4, 0.6]}";
#endif
    s << "[initialize]\ntype = \"analytical\"\nrho = \"1.0 + (x < 0.4 ? 1.0 : 0.0)\"\n"
      << "u = [" << (Mallard_DIM == 2 ? "\"0.2\", \"0.1\"" : "\"0.2\", \"0.1\", \"0.0\"") << "]\n"
      << "p = \"x < 0.4 ? 2.0 : 1.0\"\n";
    for (const char * zone : zones) s << "[[boundaries]]\nname = \"" << zone << "\"\ntype = \"outlet_extrapolation\"\n";
    s << "[numerics]\nriemann_solver = \"HLLC\"\ntime_integrator = \"SSPRK3\"\n"
      << "[numerics.face_reconstruction]\ntype = \"MUSCL\"\n"
      << "[physics]\ntype = \"euler\"\ngamma = 1.4\np_ref = 1.0\nT_ref = 1.0\nrho_ref = 1.0\n"
      << "[output]\ncheck_interval = 1000000\nrun_log = \"" << dir << "/mallard.jsonl\"\n"
      << "[integrals]\ninterval = " << interval << "\nfile = \"" << dir << "/integrals.csv\"\n"
      << "[[probes]]\nname = \"box\"\nfields = [\"RHO\", \"P\"]\nregion = " << box << "\nreductions = [\"mean\"]\ninterval = " << interval << "\n"
      << "reductions_file = \"" << dir << "/box.csv\"\n"
      << "[[write_data]]\nprefix = \"" << dir << "/restart\"\nformat = \"restart\"\ninterval = 1000000\n";
    return s.str();
}

/** @brief dir, created by rank 0, with the input of case_text written to dir/input.toml. */
std::string write_case(const std::string & dir, const std::string & run, int interval = 1) {
    if (comm::is_root()) {
        fs::create_directories(dir);
        std::ofstream(dir + "/input.toml") << case_text(dir, run, interval);
    }
    comm::barrier();
    return dir + "/input.toml";
}

/** @brief The newest restart file of dir. */
RestartData last_restart(const std::string & dir) {
    std::string newest;
    for (const auto & entry : fs::directory_iterator(dir)) {
        const std::string name = entry.path().string();
        if (name.ends_with(".restart") && name > newest) newest = name;
    }
    return read_restart(newest);
}

/**
 * @brief The run of `mallard -i file` (run_main.cpp): Solver::init of the file, then run(). Its monitored values
 *        as rank 0 has them, which alone has the probes' in a distributed run.
 */
std::vector<double> executable_run(const std::string & file) {
    ::Solver solver;
    solver.init(file);
    EXPECT_EQ(solver.run(), 0);
    const std::vector<double> values = solver.monitored_values();
    std::string bytes(values.size() * sizeof(double), '\0');
    std::memcpy(bytes.data(), values.data(), bytes.size());
    bytes = comm::broadcast(bytes);
    std::vector<double> root(bytes.size() / sizeof(double));
    std::memcpy(root.data(), bytes.data(), bytes.size());
    return root;
}

/** @brief The record of this type in dir's run log. */
json::Value record(const std::string & dir, const std::string & type) {
    json::Value found;
    std::ifstream in(dir + "/mallard.jsonl");
    for (std::string line; std::getline(in, line);) {
        json::Value v = json::parse(line);
        if (v["type"].string == type) found = v;
    }
    return found;
}

void expect_same_values(const std::vector<mallard::Monitor> & monitors, const std::vector<double> & reference) {
    ASSERT_EQ(monitors.size(), reference.size());
    for (size_t i = 0; i < monitors.size(); i++) {
        // Bit for bit: equal and both non-NaN
        EXPECT_EQ(monitors[i].value, reference[i]) << monitors[i].name;
    }
}

} // namespace

TEST(CppApiTest, StepsMatchTheExecutablesRunBitForBit) {
    // Monitors every 5 steps, so that steps run ahead of the host between them
    const std::string run = "n_steps = 12\n";
    const std::vector<double> reference = executable_run(write_case("executable", run, 5));

    mallard::Solver solver(mallard::Case::from_toml_file(write_case("api", run, 5)));
    uint64_t continuing = 0;
    while (solver.step()) continuing++;
    EXPECT_EQ(continuing, 11u);
    EXPECT_TRUE(solver.finished());
    EXPECT_FALSE(solver.step());
    EXPECT_EQ(solver.step_count(), 12u);
    const std::vector<mallard::Monitor> monitors = solver.monitors();
    EXPECT_EQ(solver.finish(), 0);

    expect_same_values(monitors, reference);
    ASSERT_TRUE(solver.monitor("box_MEAN_P"));
    EXPECT_EQ(solver.monitor("box_MEAN_P")->step, 10u);
    ASSERT_TRUE(solver.monitor("integrals_KINETIC_ENERGY"));
    EXPECT_EQ(solver.monitor("integrals_KINETIC_ENERGY")->unit, Mallard_DIM == 2 ? "kg m s-2" : "kg m2 s-2");
    const RestartData a = last_restart("executable"), b = last_restart("api");
    EXPECT_EQ(b.step, a.step);
    EXPECT_EQ(b.t, a.t);
    EXPECT_EQ(b.names, a.names);
    EXPECT_EQ(b.fields, a.fields);
}

TEST(CppApiTest, AdvanceToLandsOnTheTimeAsTStopDoes) {
    const double t_half = 0.05;
    const std::vector<double> reference = executable_run(write_case("executable", "t_stop = 0.05\n"));

    mallard::Solver solver(mallard::Case::from_toml_file(write_case("api", "t_stop = 0.1\n")));
    EXPECT_TRUE(solver.advance_to(t_half));
    // Time is in rtype, as the solver keeps it
    EXPECT_EQ(solver.time(), double(rtype(t_half)));
    expect_same_values(solver.monitors(), reference);
    // Times already reached take no step
    const uint64_t steps = solver.step_count();
    EXPECT_TRUE(solver.advance_to(0.01));
    EXPECT_EQ(solver.step_count(), steps);
    EXPECT_FALSE(solver.advance_to(1.0));
    EXPECT_EQ(solver.time(), double(rtype(0.1)));
    EXPECT_EQ(solver.finish(), 0);
}

TEST(CppApiTest, AnInMemoryCaseIsRecordedAsTheRunsInputAndEndsWhereTheCallerFinishes) {
    // The state the run ends with is that of its last step, though steps run ahead of the host between monitors
    executable_run(write_case("executable", "n_steps = 3\n", 5));
    const std::string text = case_text("memory", "n_steps = 50\n", 5);
    if (comm::is_root()) fs::create_directories("memory");
    comm::barrier();
    const mallard::Case input = mallard::Case::from_toml(text);
    EXPECT_EQ(input.toml(), text);
    mallard::Solver solver(input);
    for (int i = 0; i < 3; i++) EXPECT_TRUE(solver.step());
    EXPECT_FALSE(solver.finished());
    EXPECT_EQ(solver.finish(), 0);
    const RestartData a = last_restart("executable"), b = last_restart("memory");
    EXPECT_EQ(b.step, 3u);
    EXPECT_EQ(b.t, a.t);
    EXPECT_EQ(b.fields, a.fields);
    if (comm::is_root()) {
        EXPECT_EQ(record("memory", "start")["provenance"]["input"]["text"].string, text);
        const json::Value end = record("memory", "end");
        EXPECT_EQ(end["outcome"].string, "stopped");
        EXPECT_EQ(end["cause"].string, "caller");
        EXPECT_EQ(end["step"].number, 3.0);
    }
}

TEST(CppApiTest, InvalidInputsAreErrorsAsForTheExecutable) {
    try {
        mallard::Case::from_toml("[run\n");
        ADD_FAILURE() << "no error";
    } catch (const mallard::Error & e) {
        EXPECT_EQ(e.code(), "E-INPUT");
    }
    // Unused keys too: a case run through the API is checked as its file would be
    std::string text = case_text("unused", "n_steps = 2\n");
    text.insert(text.find("cfl ="), "cfll = 0.5\n");
    mallard::Solver solver(mallard::Case::from_toml(text));
    try {
        solver.setup();
        ADD_FAILURE() << "no error";
    } catch (const mallard::Error & e) {
        EXPECT_EQ(e.code(), "E-KEY-UNUSED");
        EXPECT_NE(std::string(e.what()).find("cfll"), std::string::npos) << e.what();
    }
}

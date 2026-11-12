/**
 * @file post_derive_test.cpp
 * @brief mallard post derive: derived fields of existing output are those the run wrote at those steps.
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 */

#include <gtest/gtest.h>

#include <filesystem>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "cli.h"
#include "comm.h"
#include "commands.h"
#include "data_paths.h"
#include "output_reader.h"
#include "scratch.h"
#include "solver.h"
#include "test_fixtures.h"

namespace {

namespace fs = std::filesystem;

std::string fresh_dir(const std::string & name) {
    const std::string dir = (scratch_dir() / ("derive_" + name)).string();
    comm::barrier();
    if (comm::is_root()) fs::remove_all(dir);
    comm::barrier();
    return dir;
}

std::string vector_of(const char * x, const char * y, const char * z) {
    return N_DIM == 3 ? std::string("[") + x + ", " + y + ", " + z + "]" : std::string("[") + x + ", " + y + "]";
}

/** @brief A periodic box writing restart files and VTU snapshots of the fields every 2 steps. */
std::string case_input(const std::string & dir, const std::string & physics, const std::string & state,
                       const std::string & output, const std::string & fields) {
    std::ostringstream s;
    s << "[run]\nn_steps = 4\ndt = 1.0e-4\n[mesh]\ntype = \"cartesian\"\nNx = 8\nNy = 8\nLx = 1.0\nLy = 1.0\n";
    if (N_DIM == 3) s << "Nz = 4\nLz = 1.0\n[[periodic]]\nzones = [\"back\", \"front\"]\n";
    s << "[[periodic]]\nzones = [\"left\", \"right\"]\n[[periodic]]\nzones = [\"bottom\", \"top\"]\n"
      << "[initialize]\ntype = \"analytical\"\n" << state << "u = "
      << vector_of("\"0.3 + 0.1 * sin(2 * pi * y)\"", "\"-0.1 + 0.1 * sin(2 * pi * x)\"", "\"0.2 * sin(2 * pi * x)\"")
      << "\n[numerics]\nriemann_solver = \"HLLC\"\ntime_integrator = \"SSPRK3\"\n"
      << "[numerics.face_reconstruction]\ntype = \"TENO\"\n" << physics
      << "[output]\ncheck_interval = 1000000\nrun_log = \"\"\n" << output
      << "[[write_data]]\nprefix = \"" << dir << "/restart\"\nformat = \"restart\"\ninterval = 2\n"
      << "[[write_data]]\nprefix = \"" << dir << "/flow\"\nformat = \"vtu\"\nfields = [" << fields
      << "]\ninterval = 2\n";
    return s.str();
}

int derive(const std::vector<std::string> & words, std::string * errors = nullptr, std::string * printed = nullptr) {
    cli::Args args;
    std::string error;
    EXPECT_TRUE(cli::parse(*cli::command("post derive"), words, args, error)) << error;
    std::ostringstream report, messages;
    std::streambuf * out = std::cout.rdbuf(report.rdbuf());
    std::streambuf * err = std::cerr.rdbuf(messages.rdbuf());
    const int status = post_derive(args);
    std::cout.rdbuf(out);
    std::cerr.rdbuf(err);
    if (errors) *errors = messages.str();
    if (printed) *printed = report.str();
    comm::barrier();
    return status;
}

/** @brief Runs the case, derives fields of its restart files, and expects the run's snapshots bit for bit. */
void expect_derived_as_written(const std::string & name, const std::string & physics, const std::string & state,
                               const std::string & output, const std::vector<std::string> & fields) {
    std::string list, quoted;
    for (const std::string & f : fields) {
        list += (list.empty() ? "" : ",") + f;
        quoted += (quoted.empty() ? "\"" : ", \"") + f + "\"";
    }
    const std::string run = fresh_dir(name + "_run");
    {
        Solver solver;
        solver.init(parse_toml(case_input(run, physics, state, output, quoted)));
        solver.run();
    }
    comm::barrier();
    const std::string out = fresh_dir(name);
    ASSERT_EQ(derive({run + "/restart_000000.restart", run + "/restart_000002.restart", run + "/restart_000004.restart",
                      "--fields", list, "--snapshot", "vtu", "-o", out}),
              0);
    const output::Contents written = output::open(run + "/flow.pvd");
    const output::Contents derived = output::open(out + "/derived.pvd");
    ASSERT_EQ(derived.snapshots.size(), 3u);
    ASSERT_EQ(written.snapshots.size(), 3u);
    for (size_t k = 0; k < 3; k++) {
        EXPECT_EQ(derived.snapshots[k].t, written.snapshots[k].t);
        for (const std::string & f : fields) {
            const std::vector<double> a = output::read(written, f, k), b = output::read(derived, f, k);
            ASSERT_EQ(a.size(), b.size()) << f;
            size_t differ = 0;
            for (size_t i = 0; i < a.size(); i++) differ += a[i] != b[i];
            EXPECT_EQ(differ, 0u) << f << " at snapshot " << k;
        }
    }
}

} // namespace

TEST(PostDeriveTest, RestartFilesGiveTheRunsDerivedFieldsBitForBit) {
    // TENO cell gradients of a perfect gas (the solver's vorticity among them), and the [output] reference of CP from the run's input
    std::vector<std::string> fields = {"MACH",    "P0",          "T0",         "H0",        "SOUND_SPEED",
                                       "ENTROPY", "Q_CRITERION", "DILATATION", "ENSTROPHY", "STRAIN_RATE", "CP"};
    if (N_DIM == 3) {
        fields.insert(fields.end(), {"HELICITY", "VORTICITY_X", "VORTICITY_Y", "VORTICITY_Z"});
    } else {
        fields.push_back("VORTICITY");
    }
    expect_derived_as_written("perfect", "[physics]\ntype = \"euler\"\ngamma = 1.4\np_ref = 1.0\nT_ref = 1.0\nrho_ref = 1.0\n",
                              "rho = \"1.0 + 0.2 * sin(2 * pi * x) * sin(2 * pi * y)\"\np = \"1.0\"\n",
                              "reference = {rho = 1.0, u = " + vector_of("0.3", "0.0", "0.0") + ", p = 1.0}\n", fields);
}

TEST(PostDeriveTest, RestartFilesGiveTheRunsMixtureFieldsBitForBit) {
    // Least-squares mass-fraction gradients of the flame index, and the streams of the mixture fraction
    expect_derived_as_written(
        "mixture", "[physics]\ntype = \"euler\"\ngas = \"mixture\"\nmechanism = \"" + source_dir() + "/mechanisms/h2o2.yaml\"\n",
        "p = \"1.0e5\"\nT = \"400.0\"\nY = { H2 = \"0.5 + 0.4 * sin(2 * pi * x)\", O2 = \"0.5 - 0.4 * sin(2 * pi * x)\" }\n",
        "[output.mixture]\nfuel = {Y = {H2 = 1.0}}\noxidizer = {Y = {O2 = 1.0}}\n",
        {"MIXTURE_FRACTION", "FLAME_INDEX", "P0", "ENTROPY"});
}

TEST(PostDeriveTest, FieldsThatNeedInputTheRunLacksSayHowToGiveIt) {
    const std::string run = fresh_dir("lacking_run");
    {
        Solver solver;
        solver.init(parse_toml(case_input(run, "[physics]\ntype = \"euler\"\ngamma = 1.4\np_ref = 1.0\nT_ref = 1.0\nrho_ref = 1.0\n",
                                          "rho = \"1.0\"\np = \"1.0\"\n", "", "\"MACH\"")));
        solver.run();
    }
    comm::barrier();
    std::string errors;
    EXPECT_EQ(derive({run + "/restart_000004.restart", "--fields", "CP", "-o", fresh_dir("lacking")}, &errors), 1);
    if (comm::is_root()) {
        EXPECT_NE(errors.find("--reference"), std::string::npos) << errors;
    }
    EXPECT_EQ(derive({run + "/restart_000004.restart", "--fields", "MACHH"}, &errors), 2);
    // With the reference given, CP = (p - p_inf) / q_inf of the stored state, whose divergence-free flow keeps p = 1
    const std::string out = fresh_dir("given");
    std::string printed;
    ASSERT_EQ(derive({run + "/restart_000004.restart", "--fields", "CP", "--reference",
                      "{rho = 2.0, u = " + vector_of("1.0", "0.0", "0.0") + ", p = 0.5}", "--snapshot", "vtu", "-o", out},
                     nullptr, &printed),
              0);
    if (comm::is_root()) {
        EXPECT_NE(printed.find("== Written"), std::string::npos) << printed;
        EXPECT_NE(printed.find(" 1 file\n"), std::string::npos) << printed;
    }
    const output::Contents derived = output::open(out + "/derived.pvd");
    for (const double cp : output::read(derived, "CP", 0)) EXPECT_NEAR(cp, 0.5, 1e-3);
}

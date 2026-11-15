/**
 * @file post_flame_test.cpp
 * @brief mallard post flame: the position, thickness and speeds of a premixed flame from saved states.
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 */

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "cli.h"
#include "comm.h"
#include "commands.h"
#include "data_paths.h"
#include "json.h"
#include "mesh_block.h"
#include "output_reader.h"
#include "scratch.h"
#include "solver.h"
#include "test_fixtures.h"

namespace {

namespace fs = std::filesystem;

// The stoichiometric H2/air flame at 700 K and 1 atm of preset_validation_test.cpp: a tanh profile of
// half-width W at X_0 burning toward a closed end through gas at rest
const double DELTA = 4.775e-4, TAU = DELTA / 9.7473, LENGTH = 10 * DELTA, X_0 = 3.34e-3, W = 1.2e-4;
const double T_U = 700.0, T_B = 2573.8;

std::string fresh_dir(const std::string & name) {
    const std::string dir = (scratch_dir() / ("flame_" + name)).string();
    comm::barrier();
    if (comm::is_root()) fs::remove_all(dir);
    comm::barrier();
    return dir;
}

struct Case {
    uint32_t cells = 60;      // along the strip: 6 per thermal thickness
    bool along_y = false;     // fresh gas at high y (the mirror image) instead of low x
    std::string stop;         // the [run] line that ends it
    double interval = 0.1 * TAU;  // between restart files
    bool chemistry = true;
    bool snapshots = false;  // also VTKHDF snapshots of what post flame reads, OMEGA_* included
};

std::string flame_input(const std::string & dir, const Case & c) {
    const double h = LENGTH / c.cells;
    const std::string coordinate = c.along_y ? "(" + std::to_string(LENGTH) + " - y)" : "x";
    const std::string s =
        "0.5 * (1 + tanh((" + coordinate + " - " + std::to_string(X_0) + ") / " + std::to_string(W) + "))";
    std::ostringstream in;
    in << "[run]\n" << c.stop << "\n"
       << "[mesh]\ntype = \"cartesian\"\nNx = " << (c.along_y ? 1 : c.cells) << "\nNy = " << (c.along_y ? c.cells : 1)
       << "\nLx = " << (c.along_y ? 10 * h : LENGTH) << "\nLy = " << (c.along_y ? LENGTH : 10 * h) << "\n";
    if (N_DIM == 3) in << "Nz = 1\nLz = " << 10 * h << "\n";
    in << "[initialize]\ntype = \"analytical\"\np = \"101325.0\"\nT = \"" << T_U << " + " << T_B - T_U << " * " << s
       << "\"\nu = [\"0.0\", \"0.0\"" << (N_DIM == 3 ? ", \"0.0\"" : "") << "]\nbalance = \"N2\"\n"
       << "Y = { H2 = \"0.0285224 - 0.0263594 * " << s << "\", O2 = \"0.2263540 - 0.2132528 * " << s << "\", "
       << "H2O = \"0.2276783 * " << s << "\", OH = \"0.0103755 * " << s << "\", O = \"0.0013623 * " << s << "\", "
       << "H = \"0.0002325 * " << s << "\" }\n";
    const std::string open = c.along_y ? "bottom" : "right";
    for (const std::string zone : {"left", "right", "bottom", "top", "back", "front"}) {
        if (N_DIM == 2 && (zone == "back" || zone == "front")) continue;
        in << "[[boundaries]]\nname = \"" << zone << "\"\n"
           << (zone == open ? "type = \"outlet_pressure\"\np = 101325.0\n" : "type = \"symmetry\"\n");
    }
    in << "[numerics]\npreset = \"reacting\"\ncheck_nan = true\n"
       << "[physics]\ntype = \"navier_stokes\"\ngas = \"mixture\"\nmechanism = \"" << source_dir()
       << "/mechanisms/h2o2.yaml\"\nphase = \"ohmech\"\ntransport = \"mixture_averaged\"\n";
    if (c.chemistry) in << "[chemistry]\nrtol = 1.0e-6\natol = 1.0e-10\n";
    in << "[output]\ncheck_interval = 1000000\nrun_log = \"\"\n"
       << "[[write_data]]\nprefix = \"" << dir << "/restart\"\nformat = \"restart\"\ntime_interval = " << c.interval
       << "\n";
    if (c.snapshots) {
        in << "[[write_data]]\nprefix = \"" << dir << "/flame\"\nformat = \"vtkhdf\"\ntime_interval = " << c.interval
           << "\nfields = [\"RHO\", \"U\", \"T\", \"Y_*\", \"OMEGA_*\"]\n";
    }
    return in.str();
}

void run(const std::string & dir, const Case & c) {
    {
        Solver solver;
        solver.init(parse_toml(flame_input(dir, c)));
        solver.run();
    }
    comm::barrier();
}

std::vector<std::string> restart_files(const std::string & dir) {
    std::vector<std::string> out;
    for (const auto & entry : fs::directory_iterator(dir)) {
        if (entry.path().extension() == ".restart") out.push_back(entry.path().string());
    }
    std::sort(out.begin(), out.end());
    return out;
}

int flame(const std::vector<std::string> & words, std::string & report, std::string * messages_out = nullptr) {
    cli::Args args;
    std::string error;
    EXPECT_TRUE(cli::parse(*cli::command("post flame"), words, args, error)) << error;
    std::ostringstream out, messages;
    std::streambuf * const o = std::cout.rdbuf(out.rdbuf());
    std::streambuf * const e = std::cerr.rdbuf(messages.rdbuf());
    const int status = post_flame(args);
    std::cout.rdbuf(o);
    std::cerr.rdbuf(e);
    report = out.str() + (messages_out ? "" : messages.str());
    if (messages_out) *messages_out = messages.str();
    comm::barrier();
    return status;
}

/** @brief The JSON report of post flame, on every rank (rank 0 prints it). */
json::Value measure(std::vector<std::string> words) {
    words.insert(words.end(), {"--format", "json"});
    std::string report, messages;
    EXPECT_EQ(flame(words, report, &messages), 0) << messages;
    return json::parse(comm::broadcast(report));
}

double number(const json::Value & v, const char * key) {
    const json::Value * x = v.find(key);
    EXPECT_NE(x, nullptr) << key;
    return x ? x->number : std::nan("");
}

} // namespace

TEST(PostFlameTest, ConsumptionSpeedBurnsTheFuelTheStripLoses) {
    // Restart files hold no production rates: post flame computes them from each state. Between two of
    // them, the H2 the strip loses is what burns plus what leaves through the outlet, so the time integral
    // of S_c rho_u (Y_u - Y_b) matches the change of its H2 and the outflow
    const std::string dir = fresh_dir("consumption");
    run(dir, {.stop = "t_stop = " + std::to_string(1.5 * TAU), .snapshots = have_hdf5()});
    std::vector<std::string> files = restart_files(dir);
    files.erase(files.begin(), files.begin() + 5);
    std::vector<std::string> words = files;
    words.insert(words.end(), {"--species", "H2", "-o", dir + "/out"});
    const json::Value r = measure(words);
    EXPECT_EQ(r.find("fresh_end")->string, "low");
    const std::vector<json::Value> & rows = r.find("snapshots")->array;
    ASSERT_EQ(rows.size(), files.size());
    const double h = LENGTH / 60;
    std::vector<double> t, burnt, inventory, outflow;
    for (size_t k = 0; k < files.size(); k++) {
        const output::Contents c = output::open(files[k]);
        const std::vector<double> rho = output::read(c, "RHO", 0), rho_u = output::read(c, "RHOU_X", 0);
        const std::vector<double> rho_y = output::read(c, "RHOY_H2", 0);
        double m = 0.0;
        for (const double v : rho_y) m += v * h;
        t.push_back(c.snapshots[0].t);
        inventory.push_back(m);
        // The cartesian mesh numbers its cells along x: the last is at the outlet
        outflow.push_back(rho_u.back() * rho_y.back() / rho.back());
        const json::Value & row = rows[k];
        burnt.push_back(number(row, "S_c") * number(row, "rho_u") * (number(row, "Y_u") - number(row, "Y_b")));
    }
    double burnt_total = 0.0, outflow_total = 0.0;
    for (size_t k = 1; k < t.size(); k++) {
        burnt_total += 0.5 * (burnt[k] + burnt[k - 1]) * (t[k] - t[k - 1]);
        outflow_total += 0.5 * (outflow[k] + outflow[k - 1]) * (t[k] - t[k - 1]);
    }
    const double lost = inventory.front() - inventory.back();
    // Within 1e-4 here; the outflow is 7% of it
    EXPECT_NEAR(burnt_total + outflow_total, lost, 2e-3 * lost);
    EXPECT_TRUE(fs::exists(dir + "/out/flame.csv"));
    if (!have_hdf5()) return;
    // Snapshots of the run's own OMEGA_* are read as they are, without the case's chemistry: the same speeds,
    // but for the chemistry half step that restart files hold pending and snapshots applied (fuse_half_steps)
    const double h_cell = LENGTH / 60;
    const json::Value v = measure({dir + "/flame.vtkhdf", "--species", "H2", "-o", dir + "/stored"});
    EXPECT_EQ(v.find("rates")->string, "stored");
    EXPECT_EQ(r.find("rates")->string, "computed");
    const std::vector<json::Value> & stored = v.find("snapshots")->array;
    size_t compared = 0;
    for (const json::Value & a : rows) {
        for (const json::Value & b : stored) {
            if (number(*b.find("snapshot"), "step") != number(*a.find("snapshot"), "step")) continue;
            EXPECT_NEAR(number(b, "S_c"), number(a, "S_c"), 3e-3 * number(a, "S_c"));
            EXPECT_NEAR(number(b, "x_f"), number(a, "x_f"), 0.1 * h_cell);
            compared++;
        }
    }
    EXPECT_EQ(compared, rows.size());
}

TEST(PostFlameTest, PositionAndThicknessOfAResolvedTanhProfileAreItsCenterAndTwiceItsWidth) {
    // T = T_u + (T_b - T_u) (1 + tanh((x - X_0) / W)) / 2: midway at X_0, (T_b - T_u) / max dT/dx = 2 W
    const std::string dir = fresh_dir("tanh");
    run(dir, {.cells = 600, .stop = "n_steps = 1"});
    const json::Value r = measure({dir + "/restart_000000.restart", "-o", dir + "/out"});
    const json::Value & row = r.find("snapshots")->array.front();
    EXPECT_NEAR(number(row, "T_u"), T_U, 1e-6 * T_U);
    EXPECT_NEAR(number(row, "T_b"), T_B, 1e-6 * T_B);
    EXPECT_NEAR(number(row, "x_f"), X_0, 1e-3 * W);
    // The largest slope between cell centers h apart underestimates the tangent's by about (h / W)^2 / 12
    EXPECT_NEAR(number(row, "delta_T"), 2 * W, 1e-2 * W);
    EXPECT_GT(number(row, "S_c"), 0.0);
}

TEST(PostFlameTest, AMirroredFlameAlongAnotherDirectionMeasuresTheSame) {
    // The fresh gas at the high end of y: the flame moves the other way, at the same speeds
    const std::string x_dir = fresh_dir("along_x"), y_dir = fresh_dir("along_y");
    const std::string stop = "t_stop = " + std::to_string(0.3 * TAU);
    run(x_dir, {.stop = stop});
    run(y_dir, {.along_y = true, .stop = stop});
    const json::Value x = measure({x_dir + "/restart_000000.restart", "-o", x_dir + "/out"});
    const json::Value y = measure({y_dir + "/restart_000000.restart", "--direction", "y", "-o", y_dir + "/out"});
    EXPECT_EQ(x.find("fresh_end")->string, "low");
    EXPECT_EQ(y.find("fresh_end")->string, "high");
    EXPECT_EQ(y.find("species")->string, x.find("species")->string);
    const json::Value & a = x.find("snapshots")->array.front();
    const json::Value & b = y.find("snapshots")->array.front();
    EXPECT_NEAR(number(b, "S_c"), number(a, "S_c"), 1e-9 * std::abs(number(a, "S_c")));
    EXPECT_NEAR(number(b, "delta_T"), number(a, "delta_T"), 1e-9 * DELTA);
    EXPECT_NEAR(number(b, "x_f"), LENGTH - number(a, "x_f"), 1e-9 * LENGTH);
    // The flame burns toward the closed end, into the fresh gas: a positive displacement speed from each end
    std::vector<std::string> x_series = restart_files(x_dir), y_series = restart_files(y_dir);
    x_series.insert(x_series.end(), {"-o", x_dir + "/out"});
    y_series.insert(y_series.end(), {"--direction", "y", "-o", y_dir + "/out"});
    const double s_x = number(*measure(x_series).find("summary"), "S_d");
    const double s_y = number(*measure(y_series).find("summary"), "S_d");
    EXPECT_GT(s_x, 0.0);
    EXPECT_NEAR(s_y, s_x, 1e-6 * std::abs(s_x));
}

TEST(PostFlameTest, RefusesACaseWithoutChemistryAndAnUnknownSpecies) {
    const std::string dir = fresh_dir("errors");
    const Case c{.stop = "n_steps = 1"};
    run(dir, c);
    const std::string restart = dir + "/restart_000000.restart";
    std::string report;
    EXPECT_EQ(flame({restart, "--species", "H3", "-o", dir + "/out"}, report), 2);
    if (comm::is_root()) {
        EXPECT_NE(report.find("did you mean"), std::string::npos) << report;
    }
    const std::string inert = dir + "/inert.toml";
    Case without = c;
    without.chemistry = false;
    if (comm::is_root()) std::ofstream(inert) << flame_input(dir, without);
    comm::barrier();
    EXPECT_EQ(flame({restart, "-i", inert, "-o", dir + "/out"}, report), 1);
    if (comm::is_root()) {
        EXPECT_NE(report.find("no [chemistry]"), std::string::npos) << report;
    }
}

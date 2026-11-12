/**
 * @file step_queue_test.cpp
 * @author Matthew Bonanni (mbonanni001@gmail.com)
 * @brief Steps decided on the device and enqueued ahead of the host
 *        ([run] step_queue): the end of a step as the device takes it, and runs
 *        with every condition that stops the device for the host, against
 *        runs that wait for every step and runs stepped by the host.
 * @version 0.1
 * @date 2026-11-10
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 *
 */

#include <gtest/gtest.h>
#include <Kokkos_Core.hpp>

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

#include "comm.h"
#include "example_inputs.h"
#include "json.h"
#include "solver.h"
#include "step_graph_guard.h"
#include "test_fixtures.h"

const std::filesystem::path & scratch_dir();
std::filesystem::path rank_scratch_dir();

namespace {

namespace fs = std::filesystem;
using step_queue::Loop;
using step_queue::Record;
using step_queue::StepEnd;

Loop loop_at(uint64_t step, uint64_t step_limit, rtype t_limit, uint64_t check_interval = 0) {
    Loop loop = {};
    loop.step = step;
    loop.active = 1;
    loop.check_interval = check_interval;
    loop.horizon.step_limit = step_limit;
    loop.horizon.t_limit = t_limit;
    return loop;
}

StepEnd accepted(rtype t, rtype dt, int32_t reduced = 0) { return {true, false, true, dt, t, reduced}; }

} // namespace

TEST(StepQueueTest, AcceptedStepsAdvanceUntilTheHostsLimits) {
    // The host acts after the first step at its step limit or with t at its time limit, and only then
    Loop loop = loop_at(5, 8, 1.0_r);
    Record record = {};
    for (uint64_t s = 6; s < 8; s++) {
        step_queue::end_step(accepted(0.1_r, 0.1_r), loop, record);
        EXPECT_EQ(record.outcome, step_queue::ACCEPTED);
        EXPECT_EQ(record.reasons, 0u);
        EXPECT_EQ(loop.step, s);
        EXPECT_EQ(loop.active, 1u);
        EXPECT_EQ(loop.advanced, 1u);
    }
    step_queue::end_step(accepted(0.1_r, 0.1_r), loop, record);
    EXPECT_EQ(record.reasons, uint32_t(step_queue::STEP_DUE));
    EXPECT_EQ(loop.step, 8u);
    EXPECT_EQ(loop.active, 0u);

    // t + dt exactly at the time limit is due, as the host's t >= threshold
    loop = loop_at(0, 100, 0.75_r);
    step_queue::end_step(accepted(0.5_r, 0.25_r), loop, record);
    EXPECT_EQ(record.t_end, 0.75_r);
    EXPECT_EQ(record.reasons, uint32_t(step_queue::TIME_DUE));
    loop = loop_at(0, 100, 0.75_r);
    step_queue::end_step(accepted(0.25_r, 0.25_r), loop, record);
    EXPECT_EQ(record.reasons, 0u);
}

TEST(StepQueueTest, RequestsStopTheDeviceAfterTheirStep) {
    Loop loop = loop_at(0, 100, 10.0_r);
    Record record = {};
    step_queue::end_step(accepted(0.0_r, 0.1_r, run_control::STOP), loop, record);
    EXPECT_EQ(record.outcome, step_queue::ACCEPTED);
    EXPECT_EQ(record.reasons, uint32_t(step_queue::REQUEST));
    EXPECT_EQ(loop.step, 1u);
    EXPECT_EQ(loop.active, 0u);
}

TEST(StepQueueTest, RejectedStepsAreRetriedUntilTheLimitOfTries) {
    // Neither t nor the step advances; the run stops for the host at the last try
    Loop loop = loop_at(3, 100, 10.0_r);
    Record record = {};
    const StepEnd rejected{true, true, false, 0.5_r, 1.0_r, run_control::STOP};
    for (uint32_t k = 1; k < step_queue::MAX_TRIES; k++) {
        step_queue::end_step(rejected, loop, record);
        EXPECT_EQ(record.outcome, step_queue::REJECTED);
        EXPECT_EQ(record.reasons, 0u) << "requests count at the accepted try";
        EXPECT_EQ(record.tries, k);
        EXPECT_EQ(record.t_end, 1.0_r);
        EXPECT_EQ(loop.step, 3u);
        EXPECT_EQ(loop.active, 1u);
    }
    step_queue::end_step(rejected, loop, record);
    EXPECT_EQ(record.reasons, uint32_t(step_queue::REJECTIONS));
    EXPECT_EQ(loop.active, 0u);

    // An accepted try clears the count
    loop = loop_at(3, 100, 10.0_r);
    step_queue::end_step(rejected, loop, record);
    step_queue::end_step({true, true, true, 0.25_r, 1.0_r, 0}, loop, record);
    EXPECT_EQ(record.tries, 0u);
    EXPECT_EQ(loop.step, 4u);
    EXPECT_EQ(record.t_end, 1.25_r);
}

TEST(StepQueueTest, AnInvalidStateOrTimeStepStopsWithoutAdvancing) {
    Loop loop = loop_at(3, 100, 10.0_r);
    Record record = {};
    step_queue::end_step({false, true, true, 0.0_r, 1.0_r, 0}, loop, record);
    EXPECT_EQ(record.outcome, step_queue::DIVERGED);
    EXPECT_EQ(record.reasons, uint32_t(step_queue::DIVERGENCE));
    EXPECT_EQ(loop.step, 3u);
    EXPECT_EQ(loop.active, 0u);
    EXPECT_EQ(loop.advanced, 0u);

    loop = loop_at(3, 100, 10.0_r);
    step_queue::end_step(accepted(1.0_r, 0.0_r), loop, record);
    EXPECT_EQ(record.reasons, uint32_t(step_queue::INVALID_DT));
    EXPECT_EQ(loop.step, 3u);
    EXPECT_EQ(loop.advanced, 0u);
}

TEST(StepQueueTest, ProgressRowsOnTheDeviceAtTheirStepsBetweenStops) {
    // A row at a step the host acts on is the host's
    Loop loop = loop_at(0, 6, 10.0_r, 2);
    Record record = {};
    std::vector<uint32_t> rows;
    for (int k = 0; k < 6; k++) {
        step_queue::end_step(accepted(0.0_r, 0.1_r), loop, record);
        rows.push_back(loop.progress);
    }
    EXPECT_EQ(rows, (std::vector<uint32_t>{0, 1, 0, 1, 0, 0}));
}

TEST(StepQueueTest, HorizonTakesTheFirstLimit) {
    step_queue::Horizon h;
    h.at_multiple(10, 4);
    EXPECT_EQ(h.step_limit, 12u);
    h.at_multiple(12, 4);
    EXPECT_EQ(h.step_limit, 12u);
    h.at_multiple(10, 0);
    h.at_multiple(10, 1);
    EXPECT_EQ(h.step_limit, 11u);
    h.at_time(0.5_r);
    h.at_time(0.75_r);
    EXPECT_EQ(h.t_limit, 0.5_r);
}

namespace {

std::string vector_of(const std::string & x, const std::string & y, const std::string & z) {
    return N_DIM == 2 ? "[\"" + x + "\", \"" + y + "\"]" : "[\"" + x + "\", \"" + y + "\", \"" + z + "\"]";
}

std::string point_of(const std::string & x, const std::string & y, const std::string & z) {
    return N_DIM == 2 ? "[" + x + ", " + y + "]" : "[" + x + ", " + y + ", " + z + "]";
}

/** @brief A run of a blast in a box with walls below, outlets elsewhere. */
struct Case {
    std::string run;
    std::string extra;            // after [output]'s first key; {dir} stands for the run's directory
    std::string numerics;         // [numerics] keys
    std::string integrator = "SSPRK3";
    std::string right = "type = \"outlet_extrapolation\"\n";  // the right boundary's keys
    bool stop_file = false;       // a STOP file in the working directory
    bool distributed = false;     // over the ranks, else serial on each
    Solver::CaptureFault capture_fault = Solver::CaptureFault::NONE;
    std::string body;             // the mesh, state, boundaries, numerics and physics of another case than the box
};

std::string box_input(const Case & c, const std::string & dir, int depth) {
    std::ostringstream s;
    s << "[run]\n" << c.run << "step_queue = " << depth << "\n";
    if (!c.body.empty()) {
        s << c.body;
    } else {
        s << "[mesh]\ntype = \"cartesian\"\nNx = 14\nNy = 10\nLx = 1.0\nLy = 1.0\n";
        if (N_DIM == 3) s << "Nz = 4\nLz = 0.4\n";
        s << "[initialize]\ntype = \"analytical\"\n"
          << "rho = \"1.0 + 0.5 * exp(-40 * ((x - 0.4)^2 + (y - 0.5)^2))\"\nu = " << vector_of("0.3", "0.1", "0.05")
          << "\np = \"1.0 + exp(-40 * ((x - 0.4)^2 + (y - 0.5)^2))\"\n"
          << "[[boundaries]]\nname = \"left\"\ntype = \"outlet_extrapolation\"\n"
          << "[[boundaries]]\nname = \"right\"\n" << c.right
          << "[[boundaries]]\nname = \"top\"\ntype = \"outlet_extrapolation\"\n"
          << "[[boundaries]]\nname = \"bottom\"\ntype = \"wall_adiabatic\"\n";
        if (N_DIM == 3) s << "[[boundaries]]\nname = \"back\"\ntype = \"symmetry\"\n[[boundaries]]\nname = \"front\"\ntype = \"symmetry\"\n";
        s << "[numerics]\nriemann_solver = \"HLLC\"\ntime_integrator = \"" << c.integrator << "\"\n" << c.numerics
          << "[numerics.face_reconstruction]\ntype = \"MUSCL\"\n"
          << "[physics]\ntype = \"euler\"\ngamma = 1.4\np_ref = 1.0\nT_ref = 1.0\nrho_ref = 1.0\n";
    }
    s << "[output]\nrun_log = \"" << dir << "/mallard.jsonl\"\n";
    std::string extra = c.extra;
    for (size_t at; (at = extra.find("{dir}")) != std::string::npos;) extra.replace(at, 5, dir);
    s << extra;
    return s.str();
}

std::string read_file(const fs::path & file) {
    std::ifstream in(file, std::ios::binary);
    std::ostringstream s;
    s << in.rdbuf();
    return s.str();
}

/** @brief A file's bytes without its run and snapshot records, which name the run, its input, its directory and the wall time. */
std::string without_records(std::string text) {
    static const std::string START = "{\"schema\":\"mallard-";
    for (size_t at = 0; (at = text.find(START, at)) != std::string::npos;) {
        size_t end = at;
        int depth = 0;
        bool quoted = false;
        for (; end < text.size(); end++) {
            const char c = text[end];
            if (quoted) {
                if (c == '\\') end++;
                else if (c == '"') quoted = false;
            } else if (c == '"') {
                quoted = true;
            } else if (c == '{') {
                depth++;
            } else if (c == '}' && --depth == 0) {
                break;
            }
        }
        text.erase(at, end + 1 - at);
    }
    return text;
}

/** @brief A JSON value without the members that depend on the wall time. */
std::string without_timing(const json::Value & v) {
    static const std::vector<std::string> TIMING = {"wall", "wall_per_step", "cells_per_second", "eta", "progress",
                                                    "memory_host", "memory_host_peak", "memory_device",
                                                    "memory_device_total", "timing", "imbalance"};
    std::ostringstream s;
    switch (v.type) {
        case json::Value::Type::NUMBER: {
            char buffer[32];
            std::snprintf(buffer, sizeof buffer, "%.17g", v.number);
            return buffer;
        }
        case json::Value::Type::STRING: return "\"" + v.string + "\"";
        case json::Value::Type::BOOL: return v.boolean ? "true" : "false";
        case json::Value::Type::NUL: return "null";
        case json::Value::Type::ARRAY:
            s << "[";
            for (const auto & item : v.array) s << without_timing(item) << ",";
            s << "]";
            return s.str();
        case json::Value::Type::OBJECT:
            s << "{";
            for (size_t i = 0; i < v.names.size(); i++) {
                if (std::find(TIMING.begin(), TIMING.end(), v.names[i]) != TIMING.end()) continue;
                s << v.names[i] << ":" << without_timing(v.members[i]) << ",";
            }
            s << "}";
            return s.str();
    }
    return "";
}

/** @brief What a run shows: its state, step, time and rejections, its files, and its run log's rows and events. */
struct Outcome {
    std::vector<rtype> U;
    uint64_t step = 0;
    rtype t = 0.0;
    uint64_t rejected = 0;
    uint64_t queued = 0;
    uint64_t graph_steps = 0;
    std::string error;
    std::map<std::string, std::string> files;
    std::vector<std::string> log;
    std::vector<std::string> warnings;
};

/**
 * @brief Queued steps take the graph's step path on every backend; its capture records no fence, allocation or
 *        host copy.
 */
void expect_capturable(const StepGraphGuard & guard, const Solver & solver) {
    EXPECT_TRUE(solver.step_graph_unsupported().empty()) << solver.step_graph_unsupported();
    if (solver.get_step() > 1) {
        EXPECT_GT(guard.captures(), 0u);
    }
    for (const std::string & v : guard.violations()) ADD_FAILURE() << "in a step graph's capture: " << v;
}

Outcome run_case(const std::string & name, const Case & c, int depth) {
    const fs::path dir = (c.distributed ? scratch_dir() : rank_scratch_dir()) / (name + "_" + std::to_string(depth));
    fs::create_directories(dir);
    if (c.stop_file && comm::is_root()) std::ofstream("STOP") << "";
    comm::barrier();
    Outcome o;
    {
        const StepGraphGuard guard;
        Solver solver;
        solver.set_distributed(c.distributed);
        solver.set_uncaptured_step_graphs(depth > 0);
        solver.set_capture_fault(c.capture_fault);
        solver.init(parse_toml(box_input(c, dir.string(), depth)));
        try {
            solver.run();
        } catch (const std::exception & e) {
            o.error = e.what();
        }
        if (depth > 0 && c.capture_fault == Solver::CaptureFault::NONE) expect_capturable(guard, solver);
        solver.copy_device_to_host();
        const size_t n_species = solver.h_species.span() > 0 ? solver.h_species.extent(1) : 0;
        for (uint32_t i = 0; i < solver.get_mesh()->n_owned(); i++) {
            for (uint8_t v = 0; v < N_CONSERVATIVE; v++) o.U.push_back(solver.h_conservatives(i, v));
            for (size_t k = 0; k < n_species; k++) o.U.push_back(solver.h_species(i, k));
        }
        o.graph_steps = solver.get_graph_steps();
        o.step = solver.get_step();
        o.t = solver.get_time();
        o.rejected = solver.get_rejected_steps();
        o.queued = solver.get_queued_steps();
    }
    comm::barrier();
    if (comm::is_root()) fs::remove("STOP");
    // Messages name the run's files
    const std::string prefix = dir.string() + "/";
    const auto strip = [&](std::string text) {
        for (size_t at; (at = text.find(prefix)) != std::string::npos;) text.erase(at, prefix.size());
        return text;
    };
    o.error = strip(o.error);
    for (const auto & entry : fs::directory_iterator(dir)) {
        const std::string file = entry.path().filename().string();
        if (file == "mallard.jsonl") {
            std::ifstream in(entry.path());
            for (std::string line; std::getline(in, line);) {
                const json::Value v = json::parse(line);
                if (v["type"].string == "progress" || v["type"].string == "event") o.log.push_back(strip(without_timing(v)));
                if (v["type"].string == "warning") o.warnings.push_back(v["message"].string);
            }
        } else if (file.find(".csv") != std::string::npos || file.find(".restart") != std::string::npos) {
            o.files[file] = without_records(read_file(entry.path()));
        } else {
            o.files[file] = "";
        }
    }
    return o;
}

void expect_same(const Outcome & a, const Outcome & b, const std::string & what) {
    EXPECT_EQ(a.error, b.error) << what;
    EXPECT_EQ(a.step, b.step) << what;
    EXPECT_EQ(a.t, b.t) << what;
    EXPECT_EQ(a.rejected, b.rejected) << what;
    ASSERT_EQ(a.U.size(), b.U.size()) << what;
    size_t differ = 0;
    for (size_t k = 0; k < a.U.size(); k++) differ += a.U[k] != b.U[k];
    EXPECT_EQ(differ, 0u) << what;
    ASSERT_EQ(a.files.size(), b.files.size()) << what;
    for (const auto & [file, contents] : a.files) {
        ASSERT_TRUE(b.files.count(file)) << what << ": " << file;
        EXPECT_TRUE(contents == b.files.at(file)) << what << ": " << file << " differs";
    }
    ASSERT_EQ(a.log.size(), b.log.size()) << what;
    for (size_t k = 0; k < a.log.size(); k++) EXPECT_EQ(a.log[k], b.log[k]) << what << ", log line " << k;
}

/**
 * @brief The run stepped by the host (step_queue = 0), one step per launch
 *        (1) and enqueued ahead (3): the same state, files and log. Runs the
 *        host stops at every step (runs_ahead false) take no step ahead.
 */
Outcome expect_queue_invariant(const std::string & name, const Case & c, const bool runs_ahead = true) {
    const Outcome host = run_case(name, c, 0);
    EXPECT_EQ(host.queued, 0u);
    for (const int depth : {1, 3}) {
        const Outcome queued = run_case(name, c, depth);
        expect_same(host, queued, name + ", step_queue = " + std::to_string(depth));
        if (depth > 1 && runs_ahead) {
            EXPECT_GT(queued.queued, 0u) << name << ": no step ran ahead of the host";
        }
    }
    return host;
}

const std::string RESTARTS = "[[write_data]]\nprefix = \"{dir}/restart\"\nformat = \"restart\"\ninterval = 1000\n";

} // namespace

TEST(StepQueueTest, OutputsByStep) {
    Case c;
    c.run = "n_steps = 30\ncfl = 0.5\n";
    c.extra = "check_interval = 3\n"
              "[[write_data]]\nprefix = \"{dir}/restart\"\nformat = \"restart\"\ninterval = 7\n"
              "[[probes]]\nname = \"line\"\nfile = \"{dir}/line.csv\"\ninterval = 5\nfields = [\"RHO\", \"P\"]\n"
              "start = " + point_of("0.1", "0.3", "0.2") + "\nend = " + point_of("0.9", "0.6", "0.2") + "\nn_points = 5\n"
              "[[forces]]\nzone = \"bottom\"\nfile = \"{dir}/forces.csv\"\ninterval = 3\n"
              "[integrals]\nfile = \"{dir}/integrals.csv\"\ninterval = 4\n"
              "[statistics]\ninterval = 2\nt_start = 0.01\nfields = [\"RHO\", \"P\"]\n";
    const Outcome o = expect_queue_invariant("by_step", c);
    EXPECT_EQ(o.step, 30u);
    // Serial runs on several ranks write their files and log on the root rank only
    if (!comm::is_root()) return;
    EXPECT_TRUE(o.files.count("forces.csv"));
    EXPECT_TRUE(o.files.count("restart_000028.restart"));
}

TEST(StepQueueTest, ForcesAndIntegralsAreStagedWithoutStoppingTheDevice) {
    Case c;
    c.run = "n_steps = 20\ncfl = 0.5\n";
    c.extra = "check_interval = 1000\n"
              "[[forces]]\nzone = \"bottom\"\nfile = \"{dir}/forces.csv\"\ninterval = 2\n"
              "[integrals]\nfile = \"{dir}/integrals.csv\"\ninterval = 3\n";
    expect_queue_invariant("monitors", c);
    // As many steps run ahead as without the monitors
    Case bare = c;
    bare.extra = "check_interval = 1000\n";
    EXPECT_EQ(run_case("monitors", c, 3).queued, run_case("bare", bare, 3).queued);
}

TEST(StepQueueTest, KineticEnergyBudgetIsStagedWithoutStoppingTheDevice) {
    // The budget's extra right-hand side runs in the step graph at the monitor's steps, at the device's t
    Case c;
    c.run = "n_steps = 20\ncfl = 0.5\n";
    c.extra = "check_interval = 1000\n[integrals]\nfile = \"{dir}/integrals.csv\"\ninterval = 3\nbudget = true\n";
    c.distributed = comm::size() > 1;
    const Outcome o = expect_queue_invariant("budget", c);
    if (comm::is_root()) {
        EXPECT_NE(o.files.at("integrals.csv").find("eps_numerical"), std::string::npos);
    }
    Case bare = c;
    bare.extra = "check_interval = 1000\n";
    EXPECT_EQ(run_case("budget", c, 3).queued, run_case("bare", bare, 3).queued);
}

TEST(StepQueueTest, MeansAreSampledWithoutStoppingTheDevice) {
    Case c;
    c.run = "n_steps = 20\ncfl = 0.5\n";
    c.extra = "check_interval = 1000\n"
              "[[write_data]]\nprefix = \"{dir}/restart\"\nformat = \"restart\"\ninterval = 20\n"
              "[statistics]\ninterval = 2\nt_start = 0.01\nfields = [\"RHO\", \"P\"]\nproducts = [\"RHO*P\"]\n";
    // The restart carries the means and their clock
    expect_queue_invariant("means", c);
    Case bare = c;
    bare.extra = "check_interval = 1000\n[[write_data]]\nprefix = \"{dir}/restart\"\nformat = \"restart\"\ninterval = 20\n";
    EXPECT_EQ(run_case("means", c, 3).queued, run_case("bare", bare, 3).queued);
}

TEST(StepQueueTest, BatchMeansAreSampledWithoutStoppingTheDevice) {
    // Batches of the means close on the device; the restart carries their averages and clock
    Case c;
    c.run = "n_steps = 30\ncfl = 0.5\n";
    c.extra = "check_interval = 1000\n"
              "[[write_data]]\nprefix = \"{dir}/restart\"\nformat = \"restart\"\ninterval = 30\n"
              "[statistics]\ninterval = 2\nt_start = 0.01\nbatch = 0.05\nfields = [\"RHO\", \"P\"]\nproducts = [\"RHO*P\"]\n";
    expect_queue_invariant("batch_means", c);
    Case bare = c;
    bare.extra = "check_interval = 1000\n[[write_data]]\nprefix = \"{dir}/restart\"\nformat = \"restart\"\ninterval = 30\n";
    EXPECT_EQ(run_case("batch_means", c, 3).queued, run_case("bare", bare, 3).queued);
}

TEST(StepQueueTest, ConditionalStatisticsAreSampledWithoutStoppingTheDevice) {
    // Entries with [statistics]' clock, their own interval and t_start, a joint PDF, and one started by a trigger
    // between restarts; the files and restarts carry their accumulators and clocks
    const std::string restarts = "check_interval = 1000\n"
                                 "[[write_data]]\nprefix = \"{dir}/restart\"\nformat = \"restart\"\ninterval = 10\n";
    const std::string entries =
        "[[statistics.conditional]]\non = \"RHO\"\nbins = 8\nrange = [0.7, 1.3]\nfields = [\"P\", \"U_X\"]\n"
        "file = \"{dir}/conditional_rho.csv\"\n"
        "[[statistics.conditional]]\non = [\"RHO\", \"P\"]\nbins = [6, 4]\nrange = [[0.6, 1.4], [0.8, 1.2]]\n"
        "fields = [\"T\"]\nweight = \"mass\"\ninterval = 3\nt_start = 0.05\nfile = \"{dir}/conditional_joint.csv\"\n";
    Case c;
    c.run = "n_steps = 30\ncfl = 0.5\n";
    c.extra = restarts + "[statistics]\ninterval = 2\nt_start = 0.01\nfields = [\"RHO\"]\n" + entries;
    c.distributed = comm::size() > 1;
    const Outcome o = expect_queue_invariant("conditional", c);
    if (comm::is_root()) {
        for (const std::string file : {"conditional_rho.csv", "conditional_joint.csv"}) {
            ASSERT_TRUE(o.files.count(file)) << file;
            EXPECT_FALSE(o.files.at(file).empty()) << file;
        }
    }
    Case bare = c;
    bare.extra = restarts;
    EXPECT_EQ(run_case("conditional", c, 3).queued, run_case("bare", bare, 3).queued);

    Case alone = c;
    alone.extra = restarts + "[statistics]\ninterval = 2\n" + entries +
                  "[[triggers]]\nname = \"begin\"\nwhen = \"step >= 12\"\ndo = \"start statistics\"\n";
    expect_queue_invariant("conditional_alone", alone);
}

TEST(StepQueueTest, DerivedFieldsAndRatesAreSampledWithoutStoppingTheDevice) {
    // Means and conditional statistics of derived fields, computed under the sampling's conditional node
    const std::string restarts = "check_interval = 1000\n"
                                 "[[write_data]]\nprefix = \"{dir}/restart\"\nformat = \"restart\"\ninterval = 10\n";
    Case c;
    c.run = "n_steps = 20\ncfl = 0.5\n";
    c.extra = restarts + "[statistics]\ninterval = 2\nt_start = 0.01\nfields = [\"RHO\", \"MACH\", \"Q_CRITERION\"]\n"
                         "products = [\"MACH*SCHLIEREN\"]\n"
                         "[[statistics.conditional]]\non = \"MACH\"\nbins = 6\nrange = [0.0, 0.6]\n"
                         "fields = [\"DILATATION\", \"P\"]\ninterval = 3\nfile = \"{dir}/conditional_mach.csv\"\n";
    c.distributed = comm::size() > 1;
    expect_queue_invariant("derived", c);
    Case bare = c;
    bare.extra = restarts;
    EXPECT_EQ(run_case("derived", c, 3).queued, run_case("bare", bare, 3).queued);

    // HRR and OMEGA_<species> of a reacting mixture, with a flame index of its species gradients
    Case reacting;
    reacting.run = "n_steps = 12\ndt = 1.0e-8\n";
    std::ostringstream body;
    body << "[mesh]\ntype = \"cartesian\"\nNx = 16\nNy = 4\nLx = 1.0e-3\nLy = 1.0e-3\n";
    if (N_DIM == 3) body << "Nz = 3\nLz = 1.0e-3\n";
    body << "[[periodic]]\nzones = [\"left\", \"right\"]\n[[periodic]]\nzones = [\"bottom\", \"top\"]\n";
    if (N_DIM == 3) body << "[[periodic]]\nzones = [\"back\", \"front\"]\n";
    body << "[initialize]\ntype = \"analytical\"\np = \"101325.0\"\nT = \"1100.0 + 400.0 * sin(pi * x / 1.0e-3)\"\n"
         << "u = " << vector_of("0.0", "0.0", "0.0")
         << "\nX = { H2 = \"2.0 + sin(pi * x / 1.0e-3)\", O2 = 1.0, N2 = 3.76 }\n"
         << "[numerics]\nriemann_solver = \"HLLC\"\n[numerics.face_reconstruction]\ntype = \"FO\"\n"
         << "[physics]\ntype = \"euler\"\ngas = \"mixture\"\nmechanism = \"" << source_dir() << "/mechanisms/h2o2.yaml\"\n"
         << "[chemistry]\n";
    reacting.body = body.str();
    reacting.extra = restarts + "[output.mixture]\nfuel_species = \"H2\"\noxidizer_species = \"O2\"\n"
                                "[statistics]\ninterval = 2\nfields = [\"HRR\", \"MACH\"]\nproducts = [\"HRR*FLAME_INDEX\"]\n"
                                "[[statistics.conditional]]\non = \"T\"\nbins = 5\nrange = [1100.0, 1500.0]\n"
                                "fields = [\"HRR\", \"OMEGA_H2O\"]\nfile = \"{dir}/conditional_T.csv\"\n";
    reacting.distributed = comm::size() > 1;
    const Outcome o = expect_queue_invariant("rates", reacting);
    if (comm::is_root()) {
        ASSERT_TRUE(o.files.count("conditional_T.csv"));
        EXPECT_FALSE(o.files.at("conditional_T.csv").empty());
    }
    Case reacting_bare = reacting;
    reacting_bare.extra = restarts;
    EXPECT_EQ(run_case("rates", reacting, 3).queued, run_case("rates_bare", reacting_bare, 3).queued);
}

TEST(StepQueueTest, ProbesAreStagedWithoutStoppingTheDevice) {
    Case c;
    c.run = "n_steps = 20\ncfl = 0.5\n";
    const std::string probes =
        "[[probes]]\nname = \"point\"\nfile = \"{dir}/point.csv\"\ninterval = 2\npoint = " + point_of("0.4", "0.5", "0.2") +
        "\nfields = [\"RHO\", \"U\", \"P\", \"T\"]\n"
        "[[probes]]\nname = \"line\"\nfile = \"{dir}/line.csv\"\ninterval = 3\nstart = " + point_of("0.1", "0.3", "0.2") +
        "\nend = " + point_of("0.9", "0.6", "0.2") + "\nn_points = 5\nfields = [\"RHO\", \"P\"]\n"
        "reductions = [\"integral\", \"rms\"]\nreductions_file = \"{dir}/line_reductions.csv\"\n"
        "[[probes]]\nname = \"wall\"\nfile = \"{dir}/wall.csv\"\ninterval = 4\nboundary = \"bottom\"\n"
        "fields = [\"P\", \"MASS_FLUX\", \"TRACTION_PRESSURE\"]\nreductions = [\"integral\", \"mean\"]\n"
        "reductions_file = \"{dir}/wall_reductions.csv\"\n";
    c.extra = "check_interval = 1000\n" + probes;
    expect_queue_invariant("probes", c);
    Case bare = c;
    bare.extra = "check_interval = 1000\n";
    EXPECT_EQ(run_case("probes", c, 3).queued, run_case("bare", bare, 3).queued);
}

TEST(StepQueueTest, RowsOfRunsWithBoundariesOfTRunAhead) {
    // A row every step, with an outlet pressure of t: its range is checked on the device at each row
    Case c;
    c.run = "n_steps = 12\ncfl = 0.5\n";
    c.extra = "check_interval = 1\n";
    c.right = "type = \"outlet_pressure\"\np = \"1.0 + 0.5 * t\"\n";
    expect_queue_invariant("boundary_rows", c);
}

TEST(StepQueueTest, BoundaryOfTOutOfRangeStopsAtTheHostsRow) {
    // The outlet pressure turns negative between rows: every depth stops with its error at the next row
    Case c;
    c.run = "n_steps = 60\ncfl = 0.5\n";
    c.extra = "check_interval = 4\n";
    c.right = "type = \"outlet_pressure\"\np = \"1.0 - 2.0 * t\"\n";
    c.distributed = comm::size() > 1;
    const Outcome o = expect_queue_invariant("boundary_error", c);
    EXPECT_NE(o.error.find("must be positive"), std::string::npos) << o.error;
    EXPECT_EQ(o.step % 4, 0u);
}

TEST(StepQueueTest, TimeAndAveragedProbesAreStagedWithoutStoppingTheDevice) {
    // A probe written by time (the device shortens the steps to its output times) and a line's time averages
    // with reductions; the restart carries their clocks and averages
    Case c;
    c.run = "n_steps = 24\ncfl = 0.5\n";
    const std::string restart = "[[write_data]]\nprefix = \"{dir}/restart\"\nformat = \"restart\"\ninterval = 24\n";
    c.extra = "check_interval = 1000\n" + restart +
              "[[probes]]\nname = \"pt\"\nfile = \"{dir}/pt.csv\"\ntime_interval = 0.013\nfields = [\"RHO\", \"P\"]\n"
              "point = " + point_of("0.5", "0.5", "0.2") + "\n"
              "[[probes]]\nname = \"mean_line\"\nfile = \"{dir}/mean_line.csv\"\ninterval = 5\naverage = true\n"
              "start = " + point_of("0.1", "0.3", "0.2") + "\nend = " + point_of("0.9", "0.6", "0.2") +
              "\nn_points = 5\nfields = [\"RHO\", \"P\"]\nreductions = [\"integral\", \"rms\"]\n"
              "reductions_file = \"{dir}/mean_line_reductions.csv\"\n"
              "[statistics]\ninterval = 2\nt_start = 0.01\n";
    c.distributed = comm::size() > 1;
    const Outcome o = expect_queue_invariant("time_probes", c);
    if (comm::is_root()) {
        EXPECT_GT(std::count(o.files.at("pt.csv").begin(), o.files.at("pt.csv").end(), '\n'), 5);
    }
    Case bare = c;
    bare.extra = "check_interval = 1000\n" + restart;
    EXPECT_EQ(run_case("time_probes", c, 3).queued, run_case("bare", bare, 3).queued);
}

TEST(StepQueueTest, ConvergenceSamplesEveryMonitorRow) {
    // An integral, a force and a probe's value, each sampled at its own monitor's steps while steps run ahead
    Case c;
    c.run = "n_steps = 60\ncfl = 0.5\n";
    c.extra = "[[write_data]]\nprefix = \"{dir}/restart\"\nformat = \"restart\"\ninterval = 25\n"
              "[[probes]]\nname = \"pt\"\nfile = \"{dir}/pt.csv\"\ninterval = 3\nfields = [\"P\"]\n"
              "point = " + point_of("0.6", "0.5", "0.2") + "\n"
              "[[forces]]\nzone = \"bottom\"\nfile = \"{dir}/forces.csv\"\ninterval = 5\n"
              "[integrals]\nfile = \"{dir}/integrals.csv\"\ninterval = 2\n"
              "[statistics.convergence]\nsignals = [\"integrals_kinetic_energy\", \"forces_bottom_Fy_pressure\", "
              "\"pt_P\"]\nmin_samples = 10\nbatches = 4\nfile = \"{dir}/convergence.csv\"\n";
    c.distributed = comm::size() > 1;
    const Outcome o = expect_queue_invariant("convergence", c);
    if (!comm::is_root()) return;
    ASSERT_TRUE(o.files.count("convergence.csv"));
    // A row at each step where one of the three monitors wrote
    size_t rows = 0, expected = 0;
    std::istringstream in(o.files.at("convergence.csv"));
    for (std::string line; std::getline(in, line);) rows += !line.empty() && std::isdigit(static_cast<unsigned char>(line[0]));
    for (uint64_t s = 0; s <= 60; s++) expected += s % 2 == 0 || s % 3 == 0 || s % 5 == 0;
    EXPECT_EQ(rows, expected);
}

TEST(StepQueueTest, TriggersOnConvergenceAreEvaluatedOnTheDevice) {
    // Convergence values change only at their signals' rows (an integral's and a staged probe's): the device
    // evaluates a trigger on them between those rows, where the host updates them. MSER-5 finds both
    // signals stationary at step 57 in 2D, at step 117 in 3D
    Case c;
    c.run = "n_steps = 200\ncfl = 0.5\n";
    c.extra = "check_interval = 1000\n" + RESTARTS +
              "[[probes]]\nname = \"pt\"\nfile = \"{dir}/pt.csv\"\ninterval = 3\nfields = [\"P\"]\n"
              "point = " + point_of("0.6", "0.5", "0.2") + "\n"
              "[integrals]\nfile = \"{dir}/integrals.csv\"\ninterval = 2\n"
              "[statistics.convergence]\nsignals = [\"integrals_kinetic_energy\", \"pt_P\"]\nmin_samples = 10\n"
              "batches = 4\nfile = \"{dir}/convergence.csv\"\n"
              "[[triggers]]\nname = \"enough\"\nwhen = \"convergence_pt_P_samples >= 3\"\ndo = \"stop\"\n";
    c.distributed = comm::size() > 1;
    const Outcome o = expect_queue_invariant("convergence_triggers", c);
    EXPECT_LT(o.step, 200u);
    EXPECT_EQ(o.step % 3, 0u);
}

TEST(StepQueueTest, OutputsByTimeAndTStop) {
    Case c;
    c.run = "t_stop = 0.1\ncfl = 0.3\n";
    c.extra = "check_interval = 2\n"
              "[[write_data]]\nprefix = \"{dir}/restart\"\nformat = \"restart\"\ntime_interval = 0.03\n"
              "[[probes]]\nname = \"pt\"\nfile = \"{dir}/pt.csv\"\ntime_interval = 0.017\nfields = [\"RHO\"]\n"
              "point = " + point_of("0.5", "0.5", "0.2") + "\n";
    const Outcome o = expect_queue_invariant("by_time", c);
    EXPECT_EQ(o.t, 0.1_r);
}

TEST(StepQueueTest, ProgressRowsWithHealthAndNanChecks) {
    // A row every step, its health and ledger evaluated on the device, and check_nan after every step
    Case c;
    c.run = "n_steps = 12\ncfl = 0.5\n";
    c.extra = "check_interval = 1\n";
    c.numerics = "check_nan = true\n";
    const Outcome o = expect_queue_invariant("rows", c);
    if (comm::is_root()) {
        EXPECT_GE(o.log.size(), 12u);
    }
}

TEST(StepQueueTest, RejectedStepsAreRetriedAsTheHostRetriesThem) {
    Case c;
    c.run = "t_stop = 0.3\ncfl = 6.0\n";
    c.extra = "check_interval = 1\n";
    c.integrator = "RDPK3Sp35";
    c.numerics = "check_nan = true\n[numerics.error_control]\ntolerance = 1e-4\n";
    const Outcome o = expect_queue_invariant("rejections", c);
    EXPECT_GT(o.rejected, 0u);
    EXPECT_EQ(o.t, 0.3_r);
}

TEST(StepQueueTest, ControlFilesStopAsRequests) {
    // A STOP file found at step 5 stops the run after step 6 when the host takes every step; with steps queued
    // ahead, at the first step that reads the request, without a stop of the queue at step 5. Either way with a
    // restart, and the state the host's steps reach at that step
    Case c;
    c.run = "n_steps = 40\ncfl = 0.5\ncontrol_file_interval = 5\ncontrol_socket = false\n";
    c.extra = RESTARTS;
    c.stop_file = true;
    c.distributed = comm::size() > 1;
    for (const int depth : {0, 1, 3}) {
        const Outcome o = run_case("control_files", c, depth);
        if (depth == 0) {
            EXPECT_EQ(o.step, 6u);
        }
        EXPECT_GE(o.step, 6u) << depth;
        EXPECT_LE(o.step, 6u + 3) << depth;
        if (comm::is_root()) {
            char name[64];
            std::snprintf(name, sizeof name, "restart_%06llu.restart", static_cast<unsigned long long>(o.step));
            EXPECT_TRUE(o.files.count(name)) << depth << ": " << name;
        }
        Case host = c;
        host.stop_file = false;
        host.run = "n_steps = " + std::to_string(o.step) + "\ncfl = 0.5\ncontrol_socket = false\n";
        const Outcome reached = run_case("control_files_host", host, 0);
        EXPECT_EQ(o.t, reached.t) << depth;
        EXPECT_TRUE(o.U == reached.U) << depth;
    }
}

TEST(StepQueueTest, TriggersAreEvaluatedWithoutStoppingTheDevice) {
    // Conditions of t with a hold, of the integrals' rows, repeated, and of the step: evaluated on the device,
    // which stops for the host only at a step one fires at
    Case c;
    c.run = "n_steps = 40\ncfl = 0.5\n";
    c.extra = RESTARTS + "[integrals]\nfile = \"{dir}/integrals.csv\"\ninterval = 2\n"
              "[[triggers]]\nname = \"held\"\nwhen = \"t > 0.05\"\nfor = 0.04\ndo = \"checkpoint\"\n"
              "[[triggers]]\nname = \"energy\"\nwhen = \"integrals_kinetic_energy > 0 and step % 6 < 3\"\nrepeat = true\n"
              "do = \"checkpoint\"\n"
              "[[triggers]]\nname = \"late\"\nwhen = \"step >= 31\"\ndo = \"stop\"\n";
    c.distributed = comm::size() > 1;
    const Outcome o = expect_queue_invariant("triggers", c);
    EXPECT_EQ(o.step, 31u);
    if (comm::is_root()) {
        size_t firings = 0;
        for (const auto & line : o.log) firings += line.find("trigger") != std::string::npos;
        EXPECT_GT(firings, 3u);
    }
}

TEST(StepQueueTest, TriggersOfTheWallTimeAreEvaluatedOnTheDevice) {
    // The wall time rides along the step's reduction: the device stops for the host only at the step one fires at
    Case c;
    c.run = "n_steps = 20\ncfl = 0.5\n";
    c.extra = RESTARTS + "[[triggers]]\nname = \"held\"\nwhen = \"wall > 0 and t > 0.05\"\nfor = 0.04\ndo = \"checkpoint\"\n"
              "[[triggers]]\nname = \"late\"\nwhen = \"(step >= 15 and wall > 0) or wall > 1e9\"\ndo = \"stop\"\n";
    c.distributed = comm::size() > 1;
    const Outcome o = expect_queue_invariant("wall_triggers", c);
    EXPECT_EQ(o.step, 15u);
}

TEST(StepQueueTest, DivergenceWritesTheLastValidStateAsTheHostDoes) {
    // Far beyond the stable time step: the run diverges, and writes its last valid state and report
    Case c;
    c.run = "n_steps = 60\ncfl = 4.0\n";
    c.extra = RESTARTS;
    const Outcome o = expect_queue_invariant("diverges", c);
    EXPECT_NE(o.error.find("diverged"), std::string::npos) << o.error;
}

TEST(StepQueueTest, DistributedRunsAsTheHostTakesThem) {
    // Over the ranks: every rank stops for the host at the same steps, and the collective outputs agree
    if (comm::size() == 1) GTEST_SKIP() << "MPI runs on several ranks only";
    Case c;
    c.run = "n_steps = 24\ncfl = 0.5\n";
    c.extra = "check_interval = 2\n"
              "[[write_data]]\nprefix = \"{dir}/restart\"\nformat = \"restart\"\ninterval = 9\n"
              "[[forces]]\nzone = \"bottom\"\nfile = \"{dir}/forces.csv\"\ninterval = 5\n"
              "[integrals]\nfile = \"{dir}/integrals.csv\"\ninterval = 4\n"
              "[[probes]]\nname = \"line\"\nfile = \"{dir}/line.csv\"\ninterval = 3\nstart = " + point_of("0.1", "0.3", "0.2") +
              "\nend = " + point_of("0.9", "0.6", "0.2") + "\nn_points = 5\nfields = [\"RHO\", \"P\"]\n"
              "reductions = [\"integral\", \"rms\"]\nreductions_file = \"{dir}/line_reductions.csv\"\n"
              "[statistics]\ninterval = 2\nt_start = 0.01\nfields = [\"RHO\", \"P\"]\n";
    c.numerics = "check_nan = true\n";
    c.distributed = true;
    const Outcome o = expect_queue_invariant("distributed", c);
    EXPECT_EQ(o.step, 24u);
}

TEST(StepQueueTest, AFailedStepGraphCaptureFallsBackToStepsKernelByKernel) {
#ifndef KOKKOS_ENABLE_CUDA
    GTEST_SKIP() << "step graphs are captured in CUDA builds only";
#else
    // A capture that fails inside the step's conditional body, at once (a fence) or only at cudaStreamEndCapture (a
    // copy on the legacy stream), warns, and the run takes its steps kernel by kernel, as the host would
    Case c;
    c.run = "n_steps = 12\ncfl = 0.5\n";
    const Outcome host = run_case("failed_capture", c, 0);
    for (const auto & [fault, reported_by] : {std::pair{Solver::CaptureFault::FENCE, "cudaDeviceSynchronize"},
                                              std::pair{Solver::CaptureFault::LEGACY_STREAM, "cudaStreamEndCapture"}}) {
        c.capture_fault = fault;
        const Outcome fallback = run_case("failed_capture", c, 3);
        expect_same(host, fallback, reported_by);
        EXPECT_EQ(fallback.graph_steps, 0u) << reported_by;
        ASSERT_EQ(fallback.warnings.size(), 1u) << reported_by;
        EXPECT_NE(fallback.warnings[0].find("no CUDA graph of the step"), std::string::npos) << fallback.warnings[0];
        EXPECT_NE(fallback.warnings[0].find(reported_by), std::string::npos) << fallback.warnings[0];
    }
#endif
}

namespace {

class StepQueueExample : public ::testing::TestWithParam<ExampleCase> {};

struct ExampleOutcome {
    std::vector<rtype> U;
    uint64_t step = 0, rejected = 0, queued = 0;
    rtype t = 0.0;
    std::vector<std::string> rows;
    bool rows_on_device = false;
};

ExampleOutcome run_example(const ExampleCase & c, const int depth) {
    Input input = example_under_preset(c, 7, N_DIM == 3 ? 8 : 16);
    input.as_table()["run"].as_table()["step_queue"] = static_cast<Input::integer_type>(depth);
    const fs::path log = rank_scratch_dir() / ("example_rows_" + std::to_string(depth) + ".jsonl");
    fs::remove(log);
    auto & output = input.as_table()["output"].as_table();
    output["check_interval"] = static_cast<Input::integer_type>(1);
    output["run_log"] = log.string();
    ExampleOutcome o;
    const StepGraphGuard guard;
    Solver solver;
    solver.set_uncaptured_step_graphs(depth > 0);
    solver.init(input);
    solver.run();
    if (depth > 0) expect_capturable(guard, solver);
    solver.copy_device_to_host();
    const size_t n_species = solver.h_species.span() > 0 ? solver.h_species.extent(1) : 0;
    for (uint32_t i = 0; i < solver.get_mesh()->n_owned(); i++) {
        for (uint8_t v = 0; v < N_CONSERVATIVE; v++) o.U.push_back(solver.h_conservatives(i, v));
        for (size_t k = 0; k < n_species; k++) o.U.push_back(solver.h_species(i, k));
    }
    o.step = solver.get_step();
    o.t = solver.get_time();
    o.rejected = solver.get_rejected_steps();
    o.queued = solver.get_queued_steps();
    // Rows of runs with fused chemistry follow the host's half step
    const bool fused = input.contains("chemistry") && input.at("chemistry").contains("fuse_half_steps") &&
                       input.at("chemistry").at("fuse_half_steps").as_boolean();
    o.rows_on_device = solver.progress_on_device() && !fused;
    std::ifstream in(log);
    for (std::string line; std::getline(in, line);) {
        const json::Value v = json::parse(line);
        if (v["type"].string == "progress") o.rows.push_back(without_timing(v));
    }
    return o;
}

std::string example_name(const ::testing::TestParamInfo<ExampleCase> & p) {
    std::string name = p.param.file.substr(0, p.param.file.size() - 5);
    std::replace_if(name.begin(), name.end(), [](char ch) { return !std::isalnum(static_cast<unsigned char>(ch)); }, '_');
    return name;
}

} // namespace

TEST_P(StepQueueExample, EnqueuedStepsMatchTheHostsBitForBit) {
    // Every feature's steps (mixtures, chemistry, LES, characteristic boundaries, inflow turbulence) enqueued ahead
    const ExampleOutcome host = run_example(GetParam(), 0);
    const ExampleOutcome queued = run_example(GetParam(), 3);
    EXPECT_EQ(host.queued, 0u);
    // A progress row every step
    if (host.rows_on_device) {
        EXPECT_GT(queued.queued, 0u) << "no step ran ahead of the host";
    }
    EXPECT_EQ(queued.step, host.step);
    EXPECT_EQ(queued.t, host.t);
    EXPECT_EQ(queued.rejected, host.rejected);
    ASSERT_EQ(queued.U.size(), host.U.size());
    size_t differ = 0;
    for (size_t k = 0; k < host.U.size(); k++) differ += !(queued.U[k] == host.U[k]);
    EXPECT_EQ(differ, 0u);
    ASSERT_EQ(queued.rows.size(), host.rows.size());
    for (size_t k = 0; k < host.rows.size(); k++) EXPECT_EQ(queued.rows[k], host.rows[k]) << "row " << k;
}

INSTANTIATE_TEST_SUITE_P(Examples, StepQueueExample, ::testing::ValuesIn(examples_of_this_build()), example_name);

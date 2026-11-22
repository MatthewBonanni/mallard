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
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <map>
#include <sstream>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "comm.h"
#include "example_inputs.h"
#include "json.h"
#include "particles.h"
#include "run_log.h"
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
    std::string cells = "Nx = 14\nNy = 10\n";  // the box's cells in x and y
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
        s << "[mesh]\ntype = \"cartesian\"\n" << c.cells << "Lx = 1.0\nLy = 1.0\n";
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
    s << "[output]\nrun_log = \"" << dir << "/mallard.jsonl\"\ndivergence_report = \"" << dir << "/divergence_report\"\n";
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
    bool gated = false;  // step graphs gated their bodies
    bool conditional_nodes = false;  // step graphs could hold NCCL operations under conditional nodes
    std::string error;
    std::map<std::string, std::string> files;
    std::vector<std::string> log;
    std::vector<std::string> warnings;
};

/** @brief Distributed runs with particles step on the host, which moves the particles between ranks. */
bool migrates_particles(const Solver & solver) { return solver.get_particles() && solver.get_particles()->migrates(); }

/**
 * @brief Queued steps take the graph's step path on every backend; its capture records no fence, allocation or
 *        host copy.
 */
void expect_capturable(const StepGraphGuard & guard, const Solver & solver) {
    if (Solver::are_cuda_graphs_off()) return;
    if (migrates_particles(solver)) {
        EXPECT_EQ(solver.device_steps_unsupported(), "particles migrate between ranks through the host");
        EXPECT_EQ(guard.captures(), 0u);
        return;
    }
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
        const bool captures = c.capture_fault == Solver::CaptureFault::NONE ||
                              c.capture_fault == Solver::CaptureFault::CONDITIONAL_NODES;
        if (depth > 0 && captures) expect_capturable(guard, solver);
        solver.copy_device_to_host();
        const size_t n_species = solver.h_species.span() > 0 ? solver.h_species.extent(1) : 0;
        for (uint32_t i = 0; i < solver.get_mesh()->n_owned(); i++) {
            for (uint8_t v = 0; v < N_CONSERVATIVE; v++) o.U.push_back(solver.h_conservatives(i, v));
            for (size_t k = 0; k < n_species; k++) o.U.push_back(solver.h_species(i, k));
        }
        o.graph_steps = solver.get_graph_steps();
        o.gated = solver.get_gate_steps();
        o.conditional_nodes = o.graph_steps > 0 && solver.conditional_nodes_work();
        o.step = solver.get_step();
        o.t = solver.get_time();
        o.rejected = solver.get_rejected_steps();
        o.queued = solver.get_queued_steps();
    }
    // Rank 0 keeps a distributed run's log open, and NFS shows other nodes an open file's old contents
    run_log::close();
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
        } else if (file.find(".csv") != std::string::npos || file.find(".restart") != std::string::npos ||
                   file.find(".vtu") != std::string::npos) {
            // Restart attributes name the writers by their prefixes
            o.files[file] = strip(without_records(read_file(entry.path())));
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
        if (c.capture_fault == Solver::CaptureFault::CONDITIONAL_NODES && !Solver::are_cuda_graphs_off()) {
            EXPECT_TRUE(queued.gated) << name << ": step graphs did not gate their bodies";
            EXPECT_GT(queued.graph_steps, 0u) << name;
        }
    }
    return host;
}

const std::string RESTARTS = "[[write_data]]\nprefix = \"{dir}/restart\"\nformat = \"restart\"\ninterval = 1000\n";

/**
 * @brief A condition that holds while the probes' values are those reference wrote at step: the columns names of
 *        line's reductions file, and point's value (its file's last column), each as an integer over a power of two,
 *        which conditions read exactly (collective).
 */
std::string written_at(const Outcome & reference, const std::string & line, const std::vector<std::string> & names,
                       const std::string & point, const std::string & point_variable, const uint64_t step) {
    const std::string at = std::to_string(step);
    std::vector<double> values;
    if (comm::is_root()) {
        std::istringstream in(reference.files.at(line + "_reductions.csv"));
        std::vector<std::string> header;
        for (std::string row; std::getline(in, row);) {
            std::vector<std::string> fields;
            std::istringstream cells(row);
            for (std::string f; std::getline(cells, f, ',');) fields.push_back(f);
            if (header.empty()) header = fields;
            if (fields.empty() || fields[0] != at) continue;
            for (const std::string & name : names) {
                const auto column = std::find(header.begin(), header.end(), name);
                EXPECT_NE(column, header.end()) << name;
                if (column != header.end()) values.push_back(std::stod(fields[static_cast<size_t>(column - header.begin())]));
            }
        }
        std::istringstream pt(reference.files.at(point + ".csv"));
        for (std::string row; std::getline(pt, row);) {
            if (row.rfind(at + ",", 0) != 0) continue;
            const std::string p = row.substr(row.rfind(',') + 1);
            values.push_back(std::is_same_v<rtype, float> ? static_cast<double>(std::stof(p)) : std::stod(p));
        }
    }
    values = comm::allgatherv(values);
    EXPECT_EQ(values.size(), names.size() + 1);
    if (values.size() != names.size() + 1) return "false";
    const auto exactly = [](const double x) {
        int e = 0;
        const double m = std::frexp(x, &e);
        return "(" + std::to_string(static_cast<int64_t>(std::ldexp(m, 53))) + " / 2^" + std::to_string(53 - e) + ")";
    };
    std::string when;
    for (size_t i = 0; i < names.size(); i++) when += line + "_" + names[i] + " == " + exactly(values[i]) + " and ";
    return when + point + "_" + point_variable + " == " + exactly(values.back());
}

/** @brief The value of column in reference's file at the row of step (root), NaN if there is none. */
double value_at(const Outcome & reference, const std::string & file, const std::string & column, const uint64_t step) {
    std::istringstream in(reference.files.at(file));
    std::vector<std::string> header;
    for (std::string row; std::getline(in, row);) {
        std::vector<std::string> fields;
        std::istringstream cells(row);
        for (std::string f; std::getline(cells, f, ',');) fields.push_back(f);
        if (header.empty()) {
            header = fields;
            continue;
        }
        if (fields.empty() || !std::isdigit(static_cast<unsigned char>(fields[0][0])) || std::stoull(fields[0]) != step)
            continue;
        const auto k = std::find(header.begin(), header.end(), column) - header.begin();
        if (static_cast<size_t>(k) < fields.size()) return std::stod(fields[static_cast<size_t>(k)]);
    }
    return std::numeric_limits<double>::quiet_NaN();
}

/**
 * @brief The step of the n-th row of reference's convergence file after step `after` at a multiple of 3 steps where
 *        every value is finite, and a condition that holds while the convergence values are that row's, each as an
 *        integer over a power of two (collective). Step 0 if there is none.
 */
std::pair<uint64_t, std::string> convergence_row(const Outcome & reference, const int n, const uint64_t after = 0) {
    std::vector<std::string> names;
    std::vector<double> values;
    if (comm::is_root()) {
        std::istringstream in(reference.files.at("convergence.csv"));
        std::vector<std::string> header;
        int found = 0;
        for (std::string row; std::getline(in, row);) {
            std::vector<std::string> fields;
            std::istringstream cells(row);
            for (std::string f; std::getline(cells, f, ',');) fields.push_back(f);
            if (header.empty()) {
                header = fields;
                continue;
            }
            if (fields.empty() || !std::isdigit(static_cast<unsigned char>(fields[0][0]))) continue;
            std::vector<double> row_values;
            for (size_t k = 2; k < fields.size(); k++) row_values.push_back(std::stod(fields[k]));
            const bool finite = std::all_of(row_values.begin(), row_values.end(), [](double v) { return std::isfinite(v); });
            const uint64_t at = std::stoull(fields[0]);
            if (at <= after || at % 3 != 0 || !finite || ++found < n) continue;
            values.push_back(static_cast<double>(std::stoull(fields[0])));
            values.insert(values.end(), row_values.begin(), row_values.end());
            names.assign(header.begin() + 2, header.end());
            break;
        }
    }
    values = comm::allgatherv(values);
    if (values.empty()) return {0, "false"};
    std::string joined;
    for (const auto & name : names) joined += name + ",";
    joined = comm::broadcast(joined);
    names.clear();
    std::istringstream list(joined);
    for (std::string name; std::getline(list, name, ',');) names.push_back(name);
    EXPECT_EQ(names.size() + 1, values.size());
    const auto exactly = [](const double x) {
        if (x == 0.0) return std::string("0");
        int e = 0;
        const double m = std::frexp(x, &e);
        return "(" + std::to_string(static_cast<int64_t>(std::ldexp(m, 53))) + " / 2^" + std::to_string(53 - e) + ")";
    };
    std::string when;
    for (size_t i = 0; i < names.size() && i + 1 < values.size(); i++) {
        when += (i ? " and " : "") + ("convergence_" + names[i]) + " == " + exactly(values[i + 1]);
    }
    return {static_cast<uint64_t>(values[0]), when};
}

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
#ifdef KOKKOS_ENABLE_CUDA
    // Where conditional nodes fail (NCCL across a network), gated step graphs reach the same results
    Case unconditional = c;
    unconditional.capture_fault = Solver::CaptureFault::CONDITIONAL_NODES;
    expect_queue_invariant("by_step_unconditional", unconditional);
    if (!Solver::are_cuda_graphs_off()) {
        EXPECT_TRUE(run_case("by_step_unconditional", unconditional, 3).warnings.empty());
    }
#endif
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

namespace {

/** @brief The queued run took its steps as step graphs, with no capture fallen back to kernel-by-kernel steps. */
void expect_graphed(const Outcome & o, const std::string & what) {
    if (Solver::are_cuda_graphs_off()) return;
#ifdef KOKKOS_ENABLE_CUDA
    EXPECT_GT(o.graph_steps, 0u) << what;
#endif
    for (const std::string & w : o.warnings) {
        EXPECT_EQ(w.find("no CUDA graph of the step"), std::string::npos) << what << ": " << w;
    }
}

/** @brief A periodic H2/O2/N2 mixture with a temperature and composition wave, reacting; chemistry: [chemistry]'s keys. */
std::string reacting_body(const std::string & chemistry, const std::string & T = "1100.0") {
    std::ostringstream body;
    body << "[mesh]\ntype = \"cartesian\"\nNx = 16\nNy = 4\nLx = 1.0e-3\nLy = 1.0e-3\n";
    if (N_DIM == 3) body << "Nz = 3\nLz = 1.0e-3\n";
    body << "[[periodic]]\nzones = [\"left\", \"right\"]\n[[periodic]]\nzones = [\"bottom\", \"top\"]\n";
    if (N_DIM == 3) body << "[[periodic]]\nzones = [\"back\", \"front\"]\n";
    body << "[initialize]\ntype = \"analytical\"\np = \"101325.0\"\nT = \"" << T << " + 400.0 * sin(pi * x / 1.0e-3)\"\n"
         << "u = " << vector_of("0.0", "0.0", "0.0")
         << "\nX = { H2 = \"2.0 + sin(pi * x / 1.0e-3)\", O2 = 1.0, N2 = 3.76 }\n"
         << "[numerics]\nriemann_solver = \"HLLC\"\n[numerics.face_reconstruction]\ntype = \"FO\"\n"
         << "[physics]\ntype = \"euler\"\ngas = \"mixture\"\nmechanism = \"" << source_dir() << "/mechanisms/h2o2.yaml\"\n"
         << "[chemistry]\n" << chemistry;
    return body.str();
}

} // namespace

TEST(StepQueueTest, DerivedFieldsAndRatesAreSampledWithoutStoppingTheDevice) {
    // Means and conditional statistics of derived fields, computed under the sampling's gate
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
    const Outcome queued = run_case("derived", c, 3);
    expect_graphed(queued, "derived");
    EXPECT_EQ(queued.queued, run_case("bare", bare, 3).queued);

    // HRR and OMEGA_<species> of a reacting mixture, with a flame index of its species gradients
    Case reacting;
    reacting.run = "n_steps = 12\ndt = 1.0e-8\n";
    reacting.body = reacting_body("fuse_half_steps = false\n");
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
    const Outcome reacting_queued = run_case("rates", reacting, 3);
    expect_graphed(reacting_queued, "rates");
    EXPECT_EQ(reacting_queued.queued, run_case("rates_bare", reacting_bare, 3).queued);
}

TEST(StepQueueTest, FusedChemistryIsFlushedOnTheDeviceForWhatReadsTheState) {
    // Rows, monitors, probes, means and conditional statistics of a run with fused half steps read the state with
    // the pending half step applied on the device, as the host's flush applies it; a trigger fires at a monitor's
    // step, which the device flushed for before the host flushes again
    const std::string base = "check_interval = 1000\n"
                             "[[write_data]]\nprefix = \"{dir}/restart\"\nformat = \"restart\"\ninterval = 8\n"
                             "[[triggers]]\nname = \"save\"\nwhen = \"step == 9\"\ndo = \"checkpoint\"\n";
    std::string samples = "[integrals]\nfile = \"{dir}/integrals.csv\"\ninterval = 3\n"
                          "[[probes]]\nname = \"point\"\nfile = \"{dir}/point.csv\"\ninterval = 2\npoint = " +
                          point_of("4.0e-4", "5.0e-4", "5.0e-4") + "\nfields = [\"T\", \"P\", \"RHO\"]\n"
                          "[statistics]\ninterval = 2\nfields = [\"T\", \"HRR\"]\n"
                          "[[statistics.conditional]]\non = \"T\"\nbins = 5\nrange = [1400.0, 2400.0]\n"
                          "fields = [\"OMEGA_H2O\"]\ninterval = 3\nfile = \"{dir}/conditional_T.csv\"\n";
    Case c;
    c.run = "n_steps = 16\ncfl = 0.5\n";
    c.body = reacting_body("", "1800.0");
    c.extra = "check_interval = 2\n" + base.substr(base.find('\n') + 1) + samples;
    c.distributed = comm::size() > 1;
    const Outcome o = expect_queue_invariant("fused", c);
    if (comm::is_root()) {
        for (const std::string file : {"integrals.csv", "point.csv", "conditional_T.csv", "restart_000009.restart"}) {
            EXPECT_TRUE(o.files.count(file)) << file;
        }
    }
    Case bare = c;
    bare.extra = base;
    const Outcome queued = run_case("fused", c, 3), quiet = run_case("fused_bare", bare, 3);
    expect_graphed(queued, "fused");
    EXPECT_EQ(queued.queued, quiet.queued);
    // The flushes leave the run as it was
    ASSERT_EQ(queued.U.size(), quiet.U.size());
    size_t differ = 0;
    for (size_t k = 0; k < queued.U.size(); k++) differ += queued.U[k] != quiet.U[k];
    EXPECT_EQ(differ, 0u);
#ifdef KOKKOS_ENABLE_CUDA
    // Without conditional nodes the flushes and samples happen at the same steps
    c.capture_fault = Solver::CaptureFault::CONDITIONAL_NODES;
    expect_queue_invariant("fused_unconditional", c);
#endif
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

TEST(StepQueueTest, BoundariesOfTSeeTheHostsStageTimes) {
    // Stage times t + c dt with c not a power of two, formed on the device in captured steps
    for (const std::string integrator : {"SSPRK54", "CK45"}) {
        Case c;
        c.run = "n_steps = 8\ncfl = 0.5\n";
        c.extra = "check_interval = 1\n";
        c.integrator = integrator;
        c.right = "type = \"outlet_pressure\"\np = \"1.0 + 0.5 * sin(30 * t)\"\n";
        expect_queue_invariant("stage_times_" + integrator, c);
    }
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

TEST(StepQueueTest, ProbesOfMovingRegionsAreStagedWithoutStoppingTheDevice) {
    // A disk (sphere) leaving the box: its reductions, and its time averages kept per cell, which the restart carries
    const std::string disk = std::string("{type = \"") + (N_DIM == 2 ? "disk" : "sphere") + "\", center = " +
                             vector_of("0.5 + 10 * t", "0.5", "0.2") + ", radius = 0.3}";
    const std::string restart = "[[write_data]]\nprefix = \"{dir}/restart\"\nformat = \"restart\"\ninterval = 24\n";
    Case c;
    c.run = "n_steps = 24\ncfl = 0.5\n";
    c.extra = "check_interval = 1000\n" + restart +
              "[[probes]]\nname = \"disk\"\nregion = " + disk + "\nfields = [\"RHO\", \"P\"]\ninterval = 3\n"
              "reductions = [\"integral\", \"mean\", \"max\"]\nreductions_file = \"{dir}/disk.csv\"\n"
              "[[probes]]\nname = \"mean_disk\"\nregion = " + disk + "\nfields = [\"RHO\", \"P\"]\naverage = true\n"
              "interval = 5\nreductions = [\"integral\", \"mean\"]\nreductions_file = \"{dir}/mean_disk.csv\"\n"
              "[[probes]]\nname = \"disk_t\"\nregion = " + disk + "\nfields = [\"T\", \"U\"]\n"
              "time_interval = 0.013\nreductions = [\"integral\", \"max\"]\nreductions_file = \"{dir}/disk_t.csv\"\n"
              "[statistics]\ninterval = 2\nt_start = 0.01\n";
    c.distributed = comm::size() > 1;
    const Outcome o = expect_queue_invariant("moving_probes", c);
    if (comm::is_root()) {
        EXPECT_GT(std::count(o.files.at("disk.csv").begin(), o.files.at("disk.csv").end(), '\n'), 8);
        EXPECT_GT(std::count(o.files.at("mean_disk.csv").begin(), o.files.at("mean_disk.csv").end(), '\n'), 4);
        EXPECT_GT(std::count(o.files.at("disk_t.csv").begin(), o.files.at("disk_t.csv").end(), '\n'), 8);
        // The last writes find it outside the box: no cell covered on any rank
        const std::string & rows = o.files.at("disk.csv");
        const size_t last = rows.find_last_of('\n', rows.size() - 2) + 1;
        std::istringstream row(rows.substr(last));
        std::string step, t, measure;
        std::getline(row, step, ',');
        std::getline(row, t, ',');
        std::getline(row, measure, ',');
        EXPECT_EQ(std::stod(measure), 0.0) << rows.substr(last);
    }
    Case bare = c;
    bare.extra = "check_interval = 1000\n" + restart;
    EXPECT_EQ(run_case("moving_probes", c, 3).queued, run_case("bare", bare, 3).queued);
#ifdef KOKKOS_ENABLE_CUDA
    c.capture_fault = Solver::CaptureFault::CONDITIONAL_NODES;
    expect_queue_invariant("moving_probes_gated", c);
#endif
}

TEST(StepQueueTest, ProbesOfWhereRegionsAreStagedWithoutStoppingTheDevice) {
    // The blast's high-pressure gas, which spreads and fades: its reductions, by step and by time, and its time
    // averages kept per cell
    const std::string where = "{type = \"where\", condition = \"P > 1.25 and x > 0.2\"}";
    Case c;
    c.run = "n_steps = 24\ncfl = 0.5\n";
    c.extra = "check_interval = 1000\n"
              "[[probes]]\nname = \"hot\"\nregion = " + where + "\nfields = [\"RHO\", \"P\"]\ninterval = 3\n"
              "reductions = [\"integral\", \"mean\", \"max\"]\nreductions_file = \"{dir}/hot.csv\"\n"
              "[[probes]]\nname = \"mean_hot\"\nregion = " + where + "\nfields = [\"P\"]\naverage = true\n"
              "interval = 5\nreductions = [\"integral\", \"mean\"]\nreductions_file = \"{dir}/mean_hot.csv\"\n"
              "[[probes]]\nname = \"hot_t\"\nregion = " + where + "\nfields = [\"T\"]\n"
              "time_interval = 0.013\nreductions = [\"integral\"]\nreductions_file = \"{dir}/hot_t.csv\"\n"
              "[statistics]\ninterval = 2\nt_start = 0.01\n";
    c.distributed = comm::size() > 1;
    const Outcome o = expect_queue_invariant("where_probes", c);
    if (comm::is_root()) {
        // The region shrinks as the blast fades: the measures of the first and last rows differ
        const std::string & rows = o.files.at("hot.csv");
        EXPECT_GT(std::count(rows.begin(), rows.end(), '\n'), 8);
        const auto measure = [&](const size_t at) {
            std::istringstream row(rows.substr(at));
            std::string step, t, m;
            std::getline(row, step, ',');
            std::getline(row, t, ',');
            std::getline(row, m, ',');
            return std::stod(m);
        };
        const size_t first = rows.find("\n3,") + 1, last = rows.find_last_of('\n', rows.size() - 2) + 1;
        ASSERT_NE(first, 0u);
        EXPECT_GT(measure(first), 0.0);
        EXPECT_NE(measure(first), measure(last));
        EXPECT_GT(std::count(o.files.at("mean_hot.csv").begin(), o.files.at("mean_hot.csv").end(), '\n'), 4);
        EXPECT_GT(std::count(o.files.at("hot_t.csv").begin(), o.files.at("hot_t.csv").end(), '\n'), 8);
    }
    Case bare = c;
    bare.extra = "check_interval = 1000\n";
    EXPECT_EQ(run_case("where_probes", c, 3).queued, run_case("bare", bare, 3).queued);
}

TEST(StepQueueTest, ProbesOfIsosurfacesAreStagedWithoutStoppingTheDevice) {
    // The blast's isobar, which spreads and fades: its reductions, by step and by time, and its time averages kept
    // per cell, each extracted by the host from the levels the device staged
    const std::string iso = "{type = \"isosurface\", field = \"P\", value = 1.25}";
    Case c;
    c.run = "n_steps = 24\ncfl = 0.5\n";
    c.extra = "check_interval = 1000\n"
              "[[probes]]\nname = \"front\"\nregion = " + iso + "\nfields = [\"RHO\", \"P\", \"MEAN_P\"]\n"
              "interval = 3\nreductions = [\"integral\", \"mean\", \"max\"]\nreductions_file = \"{dir}/front.csv\"\n"
              "[[probes]]\nname = \"mean_front\"\nregion = " + iso + "\nfields = [\"P\"]\naverage = true\n"
              "interval = 5\nreductions = [\"integral\", \"mean\"]\nreductions_file = \"{dir}/mean_front.csv\"\n"
              "[[probes]]\nname = \"front_t\"\nregion = " + iso + "\nfields = [\"T\"]\n"
              "time_interval = 0.013\nreductions = [\"integral\"]\nreductions_file = \"{dir}/front_t.csv\"\n"
              "[statistics]\ninterval = 2\nt_start = 0.01\nfields = [\"P\"]\n";
    c.distributed = comm::size() > 1;
    const Outcome o = expect_queue_invariant("isosurface_probes", c);
    if (comm::is_root()) {
        // The isobar's length (area) changes as the blast spreads
        const std::string & rows = o.files.at("front.csv");
        EXPECT_GT(std::count(rows.begin(), rows.end(), '\n'), 8);
        const auto measure = [&](const size_t at) {
            std::istringstream row(rows.substr(at));
            std::string step, t, m;
            std::getline(row, step, ',');
            std::getline(row, t, ',');
            std::getline(row, m, ',');
            return std::stod(m);
        };
        const size_t first = rows.find("\n3,") + 1, last = rows.find_last_of('\n', rows.size() - 2) + 1;
        ASSERT_NE(first, 0u);
        EXPECT_GT(measure(first), 0.0);
        EXPECT_NE(measure(first), measure(last));
        EXPECT_GT(std::count(o.files.at("mean_front.csv").begin(), o.files.at("mean_front.csv").end(), '\n'), 4);
        EXPECT_GT(std::count(o.files.at("front_t.csv").begin(), o.files.at("front_t.csv").end(), '\n'), 8);
    }
    Case bare = c;
    bare.extra = "check_interval = 1000\n[statistics]\ninterval = 2\nt_start = 0.01\nfields = [\"P\"]\n";
    EXPECT_EQ(run_case("isosurface_probes", c, 3).queued, run_case("bare", bare, 3).queued);
}

TEST(StepQueueTest, ProbesOfIsosurfacesCutOnTheDeviceAreStagedWithoutStoppingTheDevice) {
    // The isobar's mass flux at each quadrature point (through its normal), the reconstruction there, and their
    // time averages: cut and sampled by the device, then cut again by the host from the levels it staged
    const std::string iso = "{type = \"isosurface\", field = \"P\", value = 1.25}";
    Case c;
    c.run = "n_steps = 24\ncfl = 0.5\n";
    c.extra = "check_interval = 1000\n"
              "[[probes]]\nname = \"flux\"\nregion = " + iso + "\nfields = [\"MASS_FLUX\", \"MEAN_P\"]\n"
              "interval = 3\nreductions = [\"integral\", \"max\"]\nreductions_file = \"{dir}/flux.csv\"\n"
              "[[probes]]\nname = \"points\"\nregion = " + iso + "\nfields = [\"RHO\", \"U\", \"T\"]\n"
              "sampling = \"point\"\ninterval = 4\nreductions = [\"integral\", \"min\", \"max\"]\n"
              "reductions_file = \"{dir}/points.csv\"\n"
              "[[probes]]\nname = \"mean_flux\"\nregion = " + iso + "\nfields = [\"MASS_FLUX\"]\naverage = true\n"
              "interval = 5\nreductions = [\"integral\"]\nreductions_file = \"{dir}/mean_flux.csv\"\n"
              "[[probes]]\nname = \"flux_t\"\nregion = " + iso + "\nfields = [\"ENTHALPY_FLUX\"]\nsampling = \"point\"\n"
              "time_interval = 0.013\nreductions = [\"integral\"]\nreductions_file = \"{dir}/flux_t.csv\"\n"
              "[statistics]\ninterval = 2\nt_start = 0.01\nfields = [\"P\"]\n";
    c.distributed = comm::size() > 1;
    const Outcome o = expect_queue_invariant("isosurface_cuts", c);
    if (comm::is_root()) {
        for (const char * f : {"flux.csv", "points.csv", "mean_flux.csv", "flux_t.csv"}) {
            EXPECT_GT(std::count(o.files.at(f).begin(), o.files.at(f).end(), '\n'), 4) << f;
        }
    }
    Case bare = c;
    bare.extra = "check_interval = 1000\n[statistics]\ninterval = 2\nt_start = 0.01\nfields = [\"P\"]\n";
    EXPECT_EQ(run_case("isosurface_cuts", c, 3).queued, run_case("bare", bare, 3).queued);
}

TEST(StepQueueTest, ProbesOfMovingRegionsOutgrowingTheirStagingStopTheDevice) {
    // A disk (sphere) growing from a corner covers more cells than its staging holds, at first on the ranks that
    // own the corner alone: every rank's host writes those steps and the device stages it anew
    const std::string disk = std::string("{type = \"") + (N_DIM == 2 ? "disk" : "sphere") + "\", center = " +
                             vector_of("0.1", "0.1", "0.2") + ", radius = \"0.05 + 2 * t\"}";
    Case c;
    c.run = "n_steps = 24\ncfl = 0.5\n";
    c.cells = "Nx = 40\nNy = 30\n";
    c.extra = "check_interval = 1000\n"
              "[[probes]]\nname = \"disk\"\nregion = " + disk + "\nfields = [\"RHO\", \"P\"]\ninterval = 2\n"
              "reductions = [\"integral\", \"max\"]\nreductions_file = \"{dir}/disk.csv\"\n";
    c.distributed = comm::size() > 1;
    expect_queue_invariant("growing_probe", c);
    Case bare = c;
    bare.extra = "check_interval = 1000\n";
    const uint64_t queued = run_case("growing_probe", c, 3).queued, unstopped = run_case("bare", bare, 3).queued;
    EXPECT_LT(queued, unstopped);
    // Each stop regrows the staging to twice the cells covered then
    EXPECT_GE(queued + 3, unstopped);
#ifdef KOKKOS_ENABLE_CUDA
    c.capture_fault = Solver::CaptureFault::CONDITIONAL_NODES;
    expect_queue_invariant("growing_probe_gated", c);
#endif
}

TEST(StepQueueTest, ProbesOfStatisticsAreStagedWithoutStoppingTheDevice) {
    // A line of a mean, a covariance, their confidence intervals and a phase average, read from the device's
    // averages at each write
    const std::string statistics = "[statistics]\ninterval = 2\nt_start = 0.01\nbatch = 0.05\nfields = [\"RHO\", \"P\"]\n"
                                   "products = [\"RHO*P\"]\n"
                                   "[statistics.phase]\nfields = [\"P\"]\nbins = 2\nperiod = 0.02\n";
    Case c;
    c.run = "n_steps = 24\ncfl = 0.5\n";
    c.extra = "check_interval = 1000\n" + statistics +
              "[[probes]]\nname = \"line\"\nfile = \"{dir}/line.csv\"\ninterval = 5\n"
              "fields = [\"P\", \"MEAN_P\", \"COV_RHO_P\", \"CI_MEAN_P\", \"CI_COV_RHO_P\", \"PHASE_1_P\"]\n"
              "start = " + point_of("0.1", "0.3", "0.2") + "\nend = " + point_of("0.9", "0.6", "0.2") +
              "\nn_points = 5\nreductions = [\"integral\", \"max\"]\nreductions_file = \"{dir}/line_reductions.csv\"\n";
    c.distributed = comm::size() > 1;
    const Outcome o = expect_queue_invariant("statistics_probes", c);
    if (comm::is_root()) {
        const std::string & line = o.files.at("line.csv");
        EXPECT_GT(std::count(line.begin(), line.end(), '\n'), 20);
        // The last write's intervals are of closed batches
        const size_t last = line.rfind("\n20,");
        ASSERT_NE(last, std::string::npos);
        EXPECT_EQ(line.find("nan", last), std::string::npos);
    }
    Case bare = c;
    bare.extra = "check_interval = 1000\n" + statistics;
    EXPECT_EQ(run_case("statistics_probes", c, 3).queued, run_case("bare", bare, 3).queued);
}

TEST(StepQueueTest, IntervalsOfMoreBatchesThanTheDevicesQuantilesAreStaged) {
    // A batch per sample: the closed batches outgrow the Student's t quantiles the device holds, which the host
    // extends where the device could run out of them
    Case c;
    c.run = "n_steps = 1100\ncfl = 0.5\n";
    c.extra = "check_interval = 2000\n"
              "[statistics]\ninterval = 1\nbatch = 1e-9\nfields = [\"P\"]\n"
              "[[probes]]\nname = \"pt\"\nfile = \"{dir}/pt.csv\"\ninterval = 50\nfields = [\"CI_MEAN_P\"]\n"
              "point = " + point_of("0.6", "0.5", "0.2") + "\n";
    const Outcome o = expect_queue_invariant("many_batches", c);
    if (comm::is_root()) {
        // Every write after the first holds an interval
        const std::string & pt = o.files.at("pt.csv");
        EXPECT_NE(pt.find("\n1100,"), std::string::npos);
        EXPECT_EQ(pt.find(",nan\n"), pt.rfind(",nan\n"));
    }
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
    // An integral's and a staged probe's convergence, updated on the device at their rows: the queue stops only at the
    // row whose every convergence value the condition equals, as a checkpoint at that step does, and goes on from the
    // host's histories. The signals are stationary before the [statistics] averages start, whose start then moves
    // their windows at the step the record says, not at a step queued after it
    Case c;
    c.run = "n_steps = 200\ncfl = 0.5\n";
    const std::string convergence =
        "check_interval = 1000\n" + RESTARTS +
        "[[probes]]\nname = \"pt\"\nfile = \"{dir}/pt.csv\"\ninterval = 3\nfields = [\"P\"]\n"
        "point = " + point_of("0.6", "0.5", "0.2") + "\n"
        "[integrals]\nfile = \"{dir}/integrals.csv\"\ninterval = 2\n"
        "[statistics]\ninterval = 2\nt_start = 1.5\nfields = [\"P\"]\n"
        "[statistics.convergence]\nsignals = [\"integrals_kinetic_energy\", \"pt_P\"]\nmin_samples = 10\n"
        "batches = 4\ntolerance = 0.5\nfile = \"{dir}/convergence.csv\"\n";
    c.extra = convergence;
    c.distributed = comm::size() > 1;
    const Outcome reference = run_case("convergence_triggers_reference", c, 0);
    const auto [at, when] = convergence_row(reference, 3);
    ASSERT_GT(at, 0u);
    c.extra = convergence + "[[triggers]]\nname = \"values\"\nwhen = \"" + when + "\"\ndo = \"checkpoint\"\n";
    const Outcome o = expect_queue_invariant("convergence_triggers", c);
    EXPECT_EQ(o.step, 200u);
    if (comm::is_root()) {
        char restart[64];
        std::snprintf(restart, sizeof restart, "restart_%06llu.restart", static_cast<unsigned long long>(at));
        EXPECT_TRUE(o.files.count(restart)) << restart;
    }
    Case at_step = c;
    at_step.extra = convergence + "[[triggers]]\nname = \"values\"\nwhen = \"step == " + std::to_string(at) +
                    "\"\ndo = \"checkpoint\"\n";
    EXPECT_EQ(run_case("convergence_triggers", c, 3).queued, run_case("convergence_triggers_at_step", at_step, 3).queued);
}

/**
 * @brief The box's mesh, numerics and physics with `initialize`'s [initialize] keys: the left, right and top sides of
 *        type `sides`, the bottom an adiabatic wall.
 */
std::string box_body(const std::string & initialize, const std::string & sides) {
    std::ostringstream box;
    box << "[mesh]\ntype = \"cartesian\"\nNx = 14\nNy = 10\nLx = 1.0\nLy = 1.0\n";
    if (N_DIM == 3) box << "Nz = 4\nLz = 0.4\n";
    box << "[initialize]\n" << initialize;
    for (const char * zone : {"left", "right", "top"}) box << "[[boundaries]]\nname = \"" << zone << "\"\ntype = \"" << sides << "\"\n";
    box << "[[boundaries]]\nname = \"bottom\"\ntype = \"wall_adiabatic\"\n";
    if (N_DIM == 3) box << "[[boundaries]]\nname = \"back\"\ntype = \"symmetry\"\n[[boundaries]]\nname = \"front\"\ntype = \"symmetry\"\n";
    box << "[numerics]\nriemann_solver = \"HLLC\"\ntime_integrator = \"SSPRK3\"\n"
        << "[numerics.face_reconstruction]\ntype = \"MUSCL\"\n"
        << "[physics]\ntype = \"euler\"\ngamma = 1.4\np_ref = 1.0\nT_ref = 1.0\nrho_ref = 1.0\n";
    return box.str();
}

/** @brief The box's initial state. */
std::string box_state() {
    return "type = \"analytical\"\nrho = \"1.0 + 0.5 * exp(-40 * ((x - 0.4)^2 + (y - 0.5)^2))\"\nu = " +
           vector_of("0.3", "0.1", "0.05") + "\np = \"1.0 + exp(-40 * ((x - 0.4)^2 + (y - 0.5)^2))\"\n";
}

/** @brief The rows of a CSV file after step `after` (root). */
std::vector<std::string> rows_after(const std::string & csv, const uint64_t after) {
    std::vector<std::string> rows;
    std::istringstream in(csv);
    for (std::string row; std::getline(in, row);) {
        if (!row.empty() && std::isdigit(static_cast<unsigned char>(row[0])) && std::stoull(row) > after) rows.push_back(row);
    }
    return rows;
}

TEST(StepQueueTest, ConvergenceHistoriesOutgrowingTheirDeviceMemoryStopTheQueueOnce) {
    // An integral sampled every step fills its history's 4096 samples on the device, a probe sampled every 7 steps does
    // not: the device stops there once, and goes on from histories with room for four times as many, on which a
    // condition equal to a later row's values stops the queue at that row, as a checkpoint at that step does. The
    // waves of a closed box keep the kinetic energy correlated over many lags.
    Case c;
    c.run = "n_steps = 4200\ncfl = 0.5\n";
    c.body = box_body(box_state(), "wall_adiabatic");
    const std::string convergence = "check_interval = 100000\n" + RESTARTS +
                                    "[integrals]\nfile = \"{dir}/integrals.csv\"\ninterval = 1\n"
                                    "[[probes]]\nname = \"pt\"\nfile = \"{dir}/pt.csv\"\ninterval = 7\nfields = [\"P\"]\n"
                                    "point = " + point_of("0.6", "0.5", "0.2") + "\n"
                                    "[statistics.convergence]\nsignals = [\"integrals_kinetic_energy\", \"pt_P\"]\n"
                                    "file = \"{dir}/convergence.csv\"\n";
    c.extra = convergence;
    c.distributed = comm::size() > 1;
    const Outcome reference = run_case("convergence_room_reference", c, 0);
    const auto [at, when] = convergence_row(reference, 1, 4100);
    ASSERT_GT(at, 0u);
    // Sokal's window (5 tau) spans more than one device chunk of 256 lags
    if (comm::is_root()) {
        EXPECT_GT(5.0 * value_at(reference, "convergence.csv", "integrals_kinetic_energy_tau", at), 256.0);
    }
    c.extra = convergence + "[[triggers]]\nname = \"values\"\nwhen = \"" + when + "\"\ndo = \"checkpoint\"\n";
    const Outcome o = expect_queue_invariant("convergence_room", c);
    EXPECT_EQ(o.step, 4200u);
    if (comm::is_root()) {
        char restart[64];
        std::snprintf(restart, sizeof restart, "restart_%06llu.restart", static_cast<unsigned long long>(at));
        EXPECT_TRUE(o.files.count(restart)) << restart;
    }
    Case at_step = c;
    at_step.extra = convergence + "[[triggers]]\nname = \"values\"\nwhen = \"step == " + std::to_string(at) +
                    "\"\ndo = \"checkpoint\"\n";
    EXPECT_EQ(run_case("convergence_room", c, 3).queued + 1, run_case("convergence_room_at_step", at_step, 3).queued);
}

TEST(StepQueueTest, ConvergenceHistoryStopsTheQueueAtTheSampleItsDeviceMemoryHasNoRoomFor) {
    // An integral sampled at every step from step 0: the 4096th sample, at step 4095, fills the history's device memory
    // without a stop; the queue stops at step 4096, whose sample it has no room for
    const auto queued = [](const uint64_t n_steps, const bool convergence) {
        Case c;
        c.run = "n_steps = " + std::to_string(n_steps) + "\ncfl = 0.5\n";
        c.extra = "check_interval = 100000\n[integrals]\nfile = \"{dir}/integrals.csv\"\ninterval = 1\n"
                  "[statistics.convergence]\nsignals = [\"integrals_kinetic_energy\"]\nfile = \"{dir}/convergence.csv\"\n"
                  "[[triggers]]\nname = \"never\"\ndo = \"stop\"\nwhen = \"" +
                  std::string(convergence ? "convergence_integrals_kinetic_energy_samples < 0" : "step < 0") + "\"\n";
        c.distributed = comm::size() > 1;
        const std::string name = "convergence_capacity_" + std::to_string(n_steps) + (convergence ? "" : "_host");
        return run_case(name, c, 3).queued;
    };
    EXPECT_EQ(queued(4096, true), queued(4096, false));
    EXPECT_EQ(queued(4097, true) + 1, queued(4097, false));
}

TEST(StepQueueTest, ConvergenceRestartedFromACheckpointGoesOnOnTheDevice) {
    // A run restarted from a checkpoint stages the restart's histories on the device, with a force's, an integral's
    // and a probe's convergence: a condition equal to a row's values after the restart stops the queue at that row, as
    // a checkpoint at that step does, and the restarted run's rows are the uninterrupted run's
    Case c;
    c.run = "n_steps = 200\ncfl = 0.5\n";
    const std::string convergence =
        "check_interval = 1000\n[[write_data]]\nprefix = \"{dir}/restart\"\nformat = \"restart\"\ninterval = 100\n"
        "[[forces]]\nzone = \"bottom\"\nfile = \"{dir}/forces.csv\"\ninterval = 5\n"
        "[[probes]]\nname = \"pt\"\nfile = \"{dir}/pt.csv\"\ninterval = 3\nfields = [\"P\"]\n"
        "point = " + point_of("0.6", "0.5", "0.2") + "\n"
        "[integrals]\nfile = \"{dir}/integrals.csv\"\ninterval = 2\n"
        "[statistics.convergence]\nsignals = [\"integrals_kinetic_energy\", \"forces_bottom_Fy_pressure\", \"pt_P\"]\n"
        "min_samples = 10\nbatches = 4\nfile = \"{dir}/convergence.csv\"\n";
    c.extra = convergence;
    c.distributed = comm::size() > 1;
    const Outcome reference = run_case("convergence_restart_reference", c, 0);
    const auto [at, when] = convergence_row(reference, 2, 100);
    ASSERT_GT(at, 0u);
    const fs::path checkpoint = (c.distributed ? scratch_dir() : rank_scratch_dir()) /
                                "convergence_restart_reference_0" / "restart_000100.restart";
    Case restarted = c;
    restarted.body = box_body("type = \"restart\"\nfile = \"" + checkpoint.string() + "\"\n", "outlet_extrapolation");
    restarted.extra = convergence + "[[triggers]]\nname = \"values\"\nwhen = \"" + when + "\"\ndo = \"checkpoint\"\n";
    const Outcome o = expect_queue_invariant("convergence_restarted", restarted);
    EXPECT_EQ(o.step, 200u);
    if (comm::is_root()) {
        char restart[64];
        std::snprintf(restart, sizeof restart, "restart_%06llu.restart", static_cast<unsigned long long>(at));
        EXPECT_TRUE(o.files.count(restart)) << restart;
        EXPECT_EQ(rows_after(o.files.at("convergence.csv"), 100), rows_after(reference.files.at("convergence.csv"), 100));
    }
    Case at_step = restarted;
    at_step.extra = convergence + "[[triggers]]\nname = \"values\"\nwhen = \"step == " + std::to_string(at) +
                    "\"\ndo = \"checkpoint\"\n";
    EXPECT_EQ(run_case("convergence_restarted", restarted, 3).queued,
              run_case("convergence_restarted_at_step", at_step, 3).queued);
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

TEST(StepQueueTest, SnapshotsAndRestartsAreStagedWithoutStoppingTheDevice) {
    // Restarts by step and snapshots by time (the device shortens the steps to their times), with the means and
    // their batches and the step controller's history in the restarts
    Case c;
    c.run = "n_steps = 30\ncfl = 0.5\n";
    c.integrator = "RDPK3Sp35";
    c.numerics = "[numerics.error_control]\ntolerance = 1e-3\n";
    const std::string outputs = "check_interval = 4\n"
                                "[[write_data]]\nprefix = \"{dir}/restart\"\nformat = \"restart\"\ninterval = 12\n"
                                "[[write_data]]\nprefix = \"{dir}/flow\"\nformat = \"vtu\"\ntime_interval = 0.15\n"
                                "fields = [\"RHO\", \"P\", \"MEAN_P\", \"CI_MEAN_P\", \"CFL\"]\n";
    c.extra = outputs + "[statistics]\ninterval = 2\nt_start = 0.005\nbatch = 0.01\nfields = [\"P\"]\n";
    c.distributed = comm::size() > 1;
    const Outcome o = expect_queue_invariant("staged_outputs", c);
    if (comm::is_root()) {
        EXPECT_TRUE(o.files.count("restart_000012.restart"));
        EXPECT_GT(std::count_if(o.files.begin(), o.files.end(), [](const auto & f) {
                      return f.first.find("flow_") == 0 && f.first.find(".vtu") != std::string::npos;
                  }), 2);
    }
    // Restarts further apart than the queue is deep: as many steps run ahead as without them
    const std::string statistics = "[statistics]\ninterval = 2\nt_start = 0.005\nbatch = 0.01\nfields = [\"P\"]\n";
    Case spaced = c;
    spaced.extra = "check_interval = 4\n[[write_data]]\nprefix = \"{dir}/restart\"\nformat = \"restart\"\ninterval = 7\n" +
                   statistics;
    Case bare = c;
    bare.extra = "check_interval = 4\n" + statistics;
    if (!Solver::are_cuda_graphs_off()) {
        EXPECT_EQ(run_case("spaced_outputs", spaced, 3).queued, run_case("bare", bare, 3).queued);
    }
}

namespace {

/** @brief A periodic channel between isothermal walls, driven by the mass flow forcing. */
std::string channel_body(const std::string & initialize) {
    std::ostringstream body;
    body << "[mesh]\ntype = \"cartesian\"\nNx = 10\nNy = 6\nLx = 2.0\nLy = 1.0\n";
    if (N_DIM == 3) body << "Nz = 3\nLz = 0.6\n[[periodic]]\nzones = [\"back\", \"front\"]\n";
    body << "[[periodic]]\nzones = [\"left\", \"right\"]\n" << initialize
         << "[[boundaries]]\nname = \"bottom\"\ntype = \"wall_isothermal\"\nT = 1.0\n"
         << "[[boundaries]]\nname = \"top\"\ntype = \"wall_isothermal\"\nT = 1.0\n"
         << "[source]\nmass_flow = " << point_of("0.5", "0.0", "0.0") << "\n"
         << "[numerics]\nriemann_solver = \"HLLC\"\ntime_integrator = \"SSPRK3\"\n"
         << "[numerics.face_reconstruction]\ntype = \"MUSCL\"\n"
         << "[physics]\ntype = \"navier_stokes\"\ngamma = 1.4\np_ref = 1.0\nT_ref = 1.0\nrho_ref = 1.0\nmu = 0.01\n"
         << "Pr = 0.72\n";
    return body.str();
}

} // namespace

TEST(StepQueueTest, RestartedRunsCaptureTheStepGraphAgainAndContinueBitForBit) {
    // The run restarted at step 12 takes its steps in a new step graph, the mass flow forcing's domain sums
    // included, and reaches the straight run's state
    Case c;
    c.run = "n_steps = 24\ncfl = 0.4\n";
    c.extra = "[[write_data]]\nprefix = \"{dir}/restart\"\nformat = \"restart\"\ninterval = 12\n";
    c.body = channel_body("[initialize]\ntype = \"analytical\"\nrho = \"1.0\"\np = \"1.0\"\nu = " +
                          vector_of("0.4 * (1 - (2 * y - 1)^2) + 0.05 * sin(3.14159 * x)", "0.02 * sin(6.28318 * x)",
                                    "0.01 * sin(6.28318 * x)") +
                          "\n");
    const Outcome straight = run_case("straight", c, 3);
    expect_graphed(straight, "straight");
    Case resumed = c;
    resumed.body = channel_body("[initialize]\ntype = \"restart\"\nfile = \"" +
                                (rank_scratch_dir() / "straight_3" / "restart_000012.restart").string() + "\"\n");
    const Outcome restarted = run_case("restarted", resumed, 3);
    expect_graphed(restarted, "restarted");
    EXPECT_EQ(restarted.error, straight.error);
    EXPECT_EQ(restarted.step, 24u);
    EXPECT_EQ(restarted.t, straight.t);
    ASSERT_EQ(restarted.U.size(), straight.U.size());
    size_t differ = 0;
    for (size_t k = 0; k < straight.U.size(); k++) differ += restarted.U[k] != straight.U[k];
    EXPECT_EQ(differ, 0u);
}

TEST(StepQueueTest, OutputsDueBeforeTheLastOneIsWrittenStopTheDevice) {
    // A restart every 2 steps with 3 enqueued: the next is due while the host still writes the last one; the
    // restarts carry a characteristic outlet's faces
    Case c;
    c.run = "n_steps = 20\ncfl = 0.5\n";
    c.right = "type = \"outlet_pressure\"\np = 1.0\nnscbc.L = 1.0\n";
    c.extra = "check_interval = 1000\n[[write_data]]\nprefix = \"{dir}/restart\"\nformat = \"restart\"\ninterval = 2\n";
    c.distributed = comm::size() > 1;
    expect_queue_invariant("busy_outputs", c);
}

TEST(StepQueueTest, OutputsOfSourcesAndMovingRegionsAreStagedWithoutStoppingTheDevice) {
    // Sources of t, in a moving region too, evaluated at the device's t; a region that moves and one that follows
    // the solution, their cells taken at the device's t and state. The writers coincide every 6 steps, further
    // apart than the queue is deep
    const std::string moving = std::string("{type = \"") + (N_DIM == 3 ? "box" : "rectangle") +
                               "\", center = [\"0.3 + 0.8 * t\", 0.5" + (N_DIM == 3 ? ", 0.2" : "") +
                               "], size = [0.3, 2.0" + (N_DIM == 3 ? ", 2.0" : "") + "]}";
    const std::string base = "check_interval = 1000\n"
                             "[[write_data]]\nprefix = \"{dir}/restart\"\nformat = \"restart\"\ninterval = 12\n"
                             "[regions]\nmoving = " + moving + "\n"
                             "[[sources]]\nrhoE = \"0.2 * sin(30 * t) * x\"\n"
                             "[[sources]]\ntotal = true\nrhoE = \"0.5 + t\"\nregion = \"moving\"\n";
    Case c;
    c.run = "n_steps = 26\ncfl = 0.5\n";
    c.extra = base +
              "[[write_data]]\nprefix = \"{dir}/flow\"\nformat = \"vtu\"\ninterval = 6\n"
              "fields = [\"RHO\", \"SOURCE_RHOE\"]\n"
              "[[write_data]]\nprefix = \"{dir}/region\"\nformat = \"vtu\"\ninterval = 6\nregions = [\"moving\", "
              "{type = \"where\", condition = \"x < 0.4 + t and RHO > 0\"}]\n"
              "fields = [\"RHO\", \"P\"]\n";
    c.distributed = comm::size() > 1;
    const Outcome o = expect_queue_invariant("sources_regions", c);
    if (comm::is_root()) {
        for (const std::string file : {"flow_000024.vtu", "region_moving_000024.vtu", "region_1_000024.vtu"}) {
            EXPECT_TRUE(o.files.count(file) || o.files.count(file.substr(0, file.size() - 3) + "pvtu")) << file;
        }
    }
    Case bare = c;
    bare.extra = base;
    const Outcome queued = run_case("sources_regions", c, 3);
    expect_graphed(queued, "sources_regions");
    if (!Solver::are_cuda_graphs_off()) {
        EXPECT_EQ(queued.queued, run_case("sources_regions_bare", bare, 3).queued);
    }
}

TEST(StepQueueTest, OutputsOfManyWritersAreStagedWithoutStoppingTheDevice) {
    // More writers than a 64-bit mask holds; the last one, alone due at steps 6 and 18, follows the solution
    std::string writers = "check_interval = 1000\n";
    for (int k = 0; k < 65; k++) {
        writers += "[[write_data]]\nprefix = \"{dir}/w" + std::to_string(k) + "\"\nformat = \"vtu\"\ninterval = 12\n"
                   "fields = [\"RHO\"]\n";
    }
    Case c;
    c.run = "n_steps = 26\ncfl = 0.5\n";
    c.extra = writers + "[[write_data]]\nprefix = \"{dir}/region\"\nformat = \"vtu\"\ninterval = 6\n"
                        "regions = [{type = \"where\", condition = \"x < 0.4 + t and RHO > 0\"}]\nfields = [\"P\"]\n";
    c.distributed = comm::size() > 1;
    const Outcome o = expect_queue_invariant("many_writers", c);
    if (comm::is_root()) {
        for (const std::string file : {"w64_000024.vtu", "region_0_000006.vtu", "region_0_000024.vtu"}) {
            EXPECT_TRUE(o.files.count(file) || o.files.count(file.substr(0, file.size() - 3) + "pvtu")) << file;
        }
    }
    Case bare = c;
    bare.extra = "check_interval = 1000\n";
    const Outcome queued = run_case("many_writers", c, 3);
    expect_graphed(queued, "many_writers");
    if (!Solver::are_cuda_graphs_off()) {
        EXPECT_EQ(queued.queued, run_case("many_writers_bare", bare, 3).queued);
    }
}

TEST(StepQueueTest, OutputsOfFusedChemistryAreStagedWithoutStoppingTheDevice) {
    // Snapshots of the state with the pending half step applied on the device, restarts of the state without it,
    // with the pending time
    Case c;
    c.run = "n_steps = 14\ncfl = 0.5\n";
    c.body = reacting_body("", "1800.0");
    const std::string outputs = "[[write_data]]\nprefix = \"{dir}/restart\"\nformat = \"restart\"\ninterval = 6\n"
                                "[[write_data]]\nprefix = \"{dir}/flow\"\nformat = \"vtu\"\ninterval = 6\n"
                                "fields = [\"T\", \"HRR\", \"Y_H2O\"]\n";
    c.extra = "check_interval = 1000\n" + outputs;
    c.distributed = comm::size() > 1;
    const Outcome o = expect_queue_invariant("fused_outputs", c);
    if (comm::is_root()) {
        ASSERT_TRUE(o.files.count("restart_000012.restart"));
        EXPECT_NE(o.files.at("restart_000012.restart").find("CHEMISTRY_PENDING"), std::string::npos);
    }
    Case bare = c;
    bare.extra = "check_interval = 1000\n";
    const Outcome queued = run_case("fused_outputs", c, 3), quiet = run_case("fused_outputs_bare", bare, 3);
    expect_graphed(queued, "fused_outputs");
    if (!Solver::are_cuda_graphs_off()) {
        EXPECT_EQ(queued.queued, quiet.queued);
    }
    ASSERT_EQ(queued.U.size(), quiet.U.size());
    size_t differ = 0;
    for (size_t k = 0; k < queued.U.size(); k++) differ += queued.U[k] != quiet.U[k];
    EXPECT_EQ(differ, 0u);
}

TEST(StepQueueTest, OutputsOfAveragedSpectraAndPhasesAreStagedWithoutStoppingTheDevice) {
    // Restarts carry the averaged spectra, their clock and the phase averages, snapshots the phase averages, and
    // the spectra file is written with them, all from the device's copies at the staged step
    std::ostringstream body;
    body << "[mesh]\ntype = \"cartesian\"\nNx = 12\nNy = 10\nLx = 6.283185307179586\nLy = 3.0\n";
    if (N_DIM == 3) body << "Nz = 6\nLz = 5.0\n";
    body << periodic_sides(N_DIM == 3 ? "xyz" : "xy")
         << "[initialize]\ntype = \"analytical\"\nrho = \"1.0 + 0.1 * sin(x + 2.1 * y)\"\n"
         << "u = " << vector_of("0.3 * sin(2 * x + 0.5) * cos(1.3 * y)", "0.2 * cos(3 * x - y)", "0.1 * sin(y)")
         << "\np = \"1.0 + 0.05 * cos(x - y)\"\n"
         << "[numerics]\nriemann_solver = \"HLLC\"\n[numerics.face_reconstruction]\ntype = \"MUSCL\"\n"
         << "[physics]\ntype = \"navier_stokes\"\ngamma = 1.4\np_ref = 1.0\nT_ref = 1.0\nrho_ref = 1.0\nmu = 0.01\n"
            "Pr = 0.72\n";
    const std::string restarts = "check_interval = 1000\n"
                                 "[[write_data]]\nprefix = \"{dir}/restart\"\nformat = \"restart\"\ninterval = 6\n";
    const std::string statistics = "[statistics]\ninterval = 2\n[statistics.spectra]\nfields = [\"U_X\", \"P\"]\n"
                                   "products = [\"U_X*U_Y\"]\ninterval = 3\nt_start = 0.004\nfile = \"{dir}/spectra.csv\"\n"
                                   "[statistics.phase]\nfields = [\"U_Y\"]\nbins = 3\ninterval = 2\nperiod = 0.05\n";
    Case c;
    c.run = "n_steps = 20\ncfl = 0.4\n";
    c.body = body.str();
    c.extra = restarts + "[[write_data]]\nprefix = \"{dir}/flow\"\nformat = \"vtu\"\ninterval = 6\n"
                         "fields = [\"RHO\", \"PHASE_0_U_Y\"]\n" + statistics;
    c.distributed = comm::size() > 1;
    const Outcome o = expect_queue_invariant("spectra_phases", c);
    if (comm::is_root()) {
        ASSERT_TRUE(o.files.count("restart_000018.restart"));
        EXPECT_FALSE(o.files.at("spectra.csv").empty());
    }
    Case bare = c;
    bare.extra = "check_interval = 1000\n" + statistics;
    const Outcome queued = run_case("spectra_phases", c, 3);
    expect_graphed(queued, "spectra_phases");
    if (!Solver::are_cuda_graphs_off()) {
        EXPECT_EQ(queued.queued, run_case("spectra_phases_bare", bare, 3).queued);
    }
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

TEST(StepQueueTest, RejectedStepsOfCharacteristicOutletsStartFromTheStep) {
    // A retry advances the faces' incoming waves from the step's start, with its own dt
    Case c;
    c.run = "n_steps = 30\ncfl = 0.5\n";
    c.extra = "check_interval = 1\n";
    c.integrator = "RDPK3Sp35";
    c.numerics = "[numerics.error_control]\ntolerance = 1e-3\n";
    c.right = "type = \"outlet_pressure\"\np = 1.0\nnscbc = { L = 1.0 }\n";
    const Outcome o = expect_queue_invariant("characteristic_rejections", c);
    EXPECT_GT(o.rejected, 0u);
    EXPECT_EQ(o.step, 30u);
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

TEST(StepQueueTest, TriggersOfProbeValuesAreEvaluatedOnTheDevice) {
    // The probe, written every 3 steps, stops the queue at its writes alone; between them the device evaluates
    // the condition on its latest value, which crosses a threshold at a write and fires one step later
    Case c;
    c.run = "n_steps = 60\ncfl = 0.5\n";
    const std::string probe = "check_interval = 1000\n" + RESTARTS +
                              "[[probes]]\nname = \"pt\"\nfile = \"{dir}/pt.csv\"\ninterval = 3\nfields = [\"P\"]\n"
                              "point = " + point_of("0.6", "0.5", "0.2") + "\n";
    c.extra = probe;
    c.distributed = comm::size() > 1;
    const Outcome reference = run_case("probe_triggers_reference", c, 0);
    std::vector<double> p;
    if (comm::is_root()) {
        std::istringstream in(reference.files.at("pt.csv"));
        for (std::string line; std::getline(in, line);) {
            if (line.empty() || !std::isdigit(static_cast<unsigned char>(line[0]))) continue;
            p.push_back(std::stod(line.substr(line.rfind(',') + 1)));
        }
    }
    p = comm::allgatherv(p);
    // The first write past every earlier one, after the second
    size_t k = 2;
    while (k < p.size() && !(p[k] > *std::max_element(p.begin(), p.begin() + static_cast<std::ptrdiff_t>(k)))) k++;
    ASSERT_LT(k, p.size() - 1);
    const double threshold = 0.5 * (p[k] + *std::max_element(p.begin(), p.begin() + static_cast<std::ptrdiff_t>(k)));
    std::ostringstream trigger;
    trigger << std::setprecision(17) << "[[triggers]]\nname = \"rise\"\nwhen = \"pt_P > " << threshold
            << " and step % 3 == 1\"\ndo = \"stop\"\n";
    c.extra = probe + trigger.str();
    const Outcome o = expect_queue_invariant("probe_triggers", c);
    EXPECT_EQ(o.step, 3 * k + 1);
}

TEST(StepQueueTest, TriggersOfProbeReductionsAreReducedOnTheDevice) {
    // A line's reductions and a point's value, written every 3 steps, reduced on the device at its writes: the queue
    // stops only at the step after the write whose every value the condition equals, as a stop at that step does. A
    // moving region's probe, written every 2 steps, has its own flags beside theirs
    Case c;
    c.run = "n_steps = 60\ncfl = 0.5\n";
    const std::string disk = std::string("{type = \"") + (N_DIM == 2 ? "disk" : "sphere") + "\", center = " +
                             vector_of("0.5 + 10 * t", "0.5", "0.2") + ", radius = 0.3}";
    const std::string probes =
        "check_interval = 1000\n" + RESTARTS +
        "[[probes]]\nname = \"disk\"\nregion = " + disk + "\nfields = [\"RHO\"]\ninterval = 2\n"
        "reductions = [\"integral\"]\nreductions_file = \"{dir}/disk.csv\"\n" +
        "[[probes]]\nname = \"line\"\nfile = \"{dir}/line.csv\"\ninterval = 3\nfields = [\"P\", \"RHO\"]\n"
        "start = " + point_of("0.1", "0.3", "0.2") + "\nend = " + point_of("0.9", "0.6", "0.2") + "\nn_points = 7\n"
        "reductions = [\"integral\", \"mean\", \"favre\", \"rms\", \"min\", \"max\"]\n"
        "reductions_file = \"{dir}/line_reductions.csv\"\n"
        "[[probes]]\nname = \"pt\"\nfile = \"{dir}/pt.csv\"\ninterval = 3\nfields = [\"P\"]\n"
        "point = " + point_of("0.6", "0.5", "0.2") + "\n";
    c.extra = probes;
    c.distributed = comm::size() > 1;
    const Outcome reference = run_case("probe_reductions_reference", c, 0);
    const std::string when =
        written_at(reference, "line", {"integral_P", "mean_P", "favre_P", "rms_P", "min_P", "max_RHO"}, "pt", "P", 18) +
        " and step % 3 == 1";
    c.extra = probes + "[[triggers]]\nname = \"values\"\nwhen = \"" + when + "\"\ndo = \"stop\"\n";
    const Outcome o = expect_queue_invariant("probe_reductions", c);
    EXPECT_EQ(o.step, 19u);
    Case at_step = c;
    at_step.extra = probes + "[[triggers]]\nname = \"values\"\nwhen = \"step == 19\"\ndo = \"stop\"\n";
    EXPECT_EQ(run_case("probe_reductions", c, 3).queued, run_case("probe_reductions_at_step", at_step, 3).queued);
}

TEST(StepQueueTest, TriggersOfAveragedAndStatisticsProbesAreReducedOnTheDevice) {
    // A line's time averages, a point of a [statistics] mean and a line of its confidence interval, written every 3
    // steps and sampled every 2: the device keeps the means and samples the statistics before the triggers, so the
    // queue stops only at a sampling step after the write whose every value the condition equals, as a checkpoint at
    // that step does, and goes on from the host's means and the device's statistics. The interval's extrema before
    // two batches close are of NaNs only, +-inf as the host's
    Case c;
    c.run = "n_steps = 60\ncfl = 0.5\n";
    const std::string probes =
        "check_interval = 1000\n" + RESTARTS +
        "[statistics]\ninterval = 2\nt_start = 0.01\nbatch = 0.05\nfields = [\"P\"]\n"
        "[[probes]]\nname = \"line\"\nfile = \"{dir}/line.csv\"\ninterval = 3\naverage = true\nfields = [\"P\", \"RHO\"]\n"
        "start = " + point_of("0.1", "0.3", "0.2") + "\nend = " + point_of("0.9", "0.6", "0.2") + "\nn_points = 7\n"
        "reductions = [\"integral\", \"mean\", \"favre\", \"rms\", \"min\", \"max\"]\n"
        "reductions_file = \"{dir}/line_reductions.csv\"\n"
        "[[probes]]\nname = \"pt\"\nfile = \"{dir}/pt.csv\"\ninterval = 3\nfields = [\"MEAN_P\"]\n"
        "point = " + point_of("0.6", "0.5", "0.2") + "\n"
        "[[probes]]\nname = \"ci\"\nfile = \"{dir}/ci.csv\"\ninterval = 3\nfields = [\"CI_MEAN_P\"]\n"
        "start = " + point_of("0.1", "0.3", "0.2") + "\nend = " + point_of("0.9", "0.6", "0.2") + "\nn_points = 7\n"
        "reductions = [\"min\", \"max\"]\nreductions_file = \"{dir}/ci_reductions.csv\"\n";
    // Beyond +-DBL_MAX: the extrema of no interval
    const std::string no_interval = "[[triggers]]\nname = \"no_interval\"\nwhen = \"ci_min_CI_MEAN_P > "
                                    "1.7976931348623157e308 and ci_max_CI_MEAN_P < -1.7976931348623157e308 and "
                                    "step == 3\"\ndo = \"checkpoint\"\n";
    c.extra = probes;
    c.distributed = comm::size() > 1;
    const Outcome reference = run_case("averaged_reductions_reference", c, 0);
    const std::string when = written_at(reference, "line",
                                        {"integral_MEAN_P", "mean_MEAN_P", "favre_MEAN_P", "rms_MEAN_P", "min_MEAN_P",
                                         "max_MEAN_RHO"},
                                        "pt", "MEAN_P", 18) +
                             " and " + written_at(reference, "ci", {"min_CI_MEAN_P"}, "pt", "MEAN_P", 18) +
                             " and step % 3 == 2";
    c.extra = probes + no_interval + "[[triggers]]\nname = \"values\"\nwhen = \"" + when + "\"\ndo = \"checkpoint\"\n";
    const Outcome o = expect_queue_invariant("averaged_reductions", c);
    EXPECT_EQ(o.step, 60u);
    if (comm::is_root()) {
        EXPECT_TRUE(o.files.count("restart_000003.restart"));
        EXPECT_TRUE(o.files.count("restart_000020.restart"));
    }
    Case at_step = c;
    at_step.extra = probes + "[[triggers]]\nname = \"no_interval\"\nwhen = \"step == 3\"\ndo = \"checkpoint\"\n"
                    "[[triggers]]\nname = \"values\"\nwhen = \"step == 20\"\ndo = \"checkpoint\"\n";
    EXPECT_EQ(run_case("averaged_reductions", c, 3).queued, run_case("averaged_reductions_at_step", at_step, 3).queued);
}

TEST(StepQueueTest, TriggersOfMovingRegionProbesAreReducedOnTheDevice) {
    // A disk's and a where region's reductions and the disk's time averages kept per cell, written every 3 steps and
    // sampled every 2, reduced on the device over the cells covered at each write (or held by the averages): the
    // queue stops only at a sampling step after the write whose every value the condition equals, as a checkpoint at
    // that step does, and goes on from the host's means. The disk leaves the box and comes back: it is reduced
    // again once it covers cells
    Case c;
    c.run = "n_steps = 60\ncfl = 0.5\n";
    const std::string disk = std::string("{type = \"") + (N_DIM == 2 ? "disk" : "sphere") + "\", center = " +
                             vector_of("0.3 + 12 * t * (0.6 - t)", "0.5", "0.2") + ", radius = 0.25}";
    const std::string reductions = "reductions = [\"integral\", \"mean\", \"favre\", \"rms\", \"min\", \"max\"]\n";
    const std::string probes =
        "check_interval = 1000\n" + RESTARTS + "[statistics]\ninterval = 2\nt_start = 0.01\n"
        "[[probes]]\nname = \"disk\"\nregion = " + disk + "\nfields = [\"P\", \"RHO\"]\ninterval = 3\n" + reductions +
        "reductions_file = \"{dir}/disk_reductions.csv\"\n"
        "[[probes]]\nname = \"mean_disk\"\nregion = " + disk + "\nfields = [\"P\", \"RHO\"]\ninterval = 3\n"
        "average = true\n" + reductions + "reductions_file = \"{dir}/mean_disk_reductions.csv\"\n"
        "[[probes]]\nname = \"hot\"\nregion = {type = \"where\", condition = \"P > 1.25 and x > 0.2\"}\n"
        "fields = [\"P\"]\ninterval = 3\nreductions = [\"integral\", \"max\"]\n"
        "reductions_file = \"{dir}/hot_reductions.csv\"\n"
        "[[probes]]\nname = \"pt\"\nfile = \"{dir}/pt.csv\"\ninterval = 3\nfields = [\"P\"]\n"
        "point = " + point_of("0.6", "0.5", "0.2") + "\n";
    c.extra = probes;
    c.distributed = comm::size() > 1;
    const Outcome reference = run_case("moving_reductions_reference", c, 0);
    // The second write after the disk has left the box that finds it back
    std::vector<double> measures;
    if (comm::is_root()) {
        for (uint64_t step = 0; step <= 60; step += 3) {
            measures.push_back(value_at(reference, "disk_reductions.csv", N_DIM == 2 ? "area" : "volume", step));
        }
    }
    measures = comm::allgatherv(measures);
    uint64_t out = 0, back = 0;
    for (size_t k = 3; k < measures.size() && back == 0; k++) {
        if (out == 0 && measures[k] == 0.0) out = 3 * k;
        if (out != 0 && measures[k] > 0.0 && measures[k - 1] > 0.0) back = 3 * k;
    }
    ASSERT_NE(back, 0u) << "the disk does not leave the box and come back";
    const std::vector<std::string> names = {"integral_P", "mean_P", "favre_P", "rms_P", "min_P", "max_RHO"};
    const std::string when = written_at(reference, "disk", names, "pt", "P", 6) + " and " +
                             written_at(reference, "mean_disk",
                                        {"integral_MEAN_P", "mean_MEAN_P", "favre_MEAN_P", "rms_MEAN_P", "min_MEAN_P",
                                         "max_MEAN_RHO"},
                                        "pt", "P", 6) +
                             " and " + written_at(reference, "hot", {"integral_P", "max_P"}, "pt", "P", 6) +
                             " and step % 3 == 2";
    const std::string when_back = written_at(reference, "disk", names, "pt", "P", back) + " and step % 3 == 2";
    const auto triggers = [&](const std::string & values, const std::string & again) {
        return "[[triggers]]\nname = \"values\"\nwhen = \"" + values + "\"\ndo = \"checkpoint\"\n"
               "[[triggers]]\nname = \"back\"\nwhen = \"" + again + "\"\ndo = \"checkpoint\"\n";
    };
    c.extra = probes + triggers(when, when_back);
    const Outcome o = expect_queue_invariant("moving_reductions", c);
    EXPECT_EQ(o.step, 60u);
    if (comm::is_root()) {
        for (const uint64_t step : {uint64_t(8), back + 2}) {
            std::ostringstream file;
            file << "restart_" << std::setw(6) << std::setfill('0') << step << ".restart";
            EXPECT_TRUE(o.files.count(file.str())) << file.str();
        }
        EXPECT_NE(value_at(o, "disk_reductions.csv", "integral_RHO", 3), value_at(o, "disk_reductions.csv", "integral_RHO", 6));
    }
    Case at_step = c;
    at_step.extra = probes + triggers("step == 8", "step == " + std::to_string(back + 2));
    EXPECT_EQ(run_case("moving_reductions", c, 3).queued, run_case("moving_reductions_at_step", at_step, 3).queued);
}

TEST(StepQueueTest, TriggersOfMovingRegionsStagedWhileTheyCoverNoCellAreReducedOnTheDevice) {
    // A disk growing from a corner outgrows its staging and stops the device until it holds every cell: each stop
    // stages the probes anew, among them a small disk still outside the box, which enters it later within its
    // staging; the trigger reading it fires one step after the write whose values the condition equals
    const std::string shape = std::string("{type = \"") + (N_DIM == 2 ? "disk" : "sphere") + "\", center = ";
    const std::string probes =
        "check_interval = 1000\n" + RESTARTS +
        "[[probes]]\nname = \"grow\"\nregion = " + shape + vector_of("0.1", "0.1", "0.2") +
        ", radius = \"0.05 + 6 * t\"}\nfields = [\"RHO\"]\ninterval = 2\nreductions = [\"integral\"]\n"
        "reductions_file = \"{dir}/grow.csv\"\n"
        "[[probes]]\nname = \"dot\"\nregion = " + shape + vector_of("1.3 - t", "0.5", "0.2") +
        ", radius = 0.08}\nfields = [\"P\", \"RHO\"]\ninterval = 2\nreductions = [\"integral\", \"mean\", \"max\"]\n"
        "reductions_file = \"{dir}/dot_reductions.csv\"\n"
        "[[probes]]\nname = \"pt\"\nfile = \"{dir}/pt.csv\"\ninterval = 2\nfields = [\"P\"]\n"
        "point = " + point_of("0.6", "0.5", "0.2") + "\n";
    Case c;
    c.run = "n_steps = 60\ncfl = 0.5\n";
    c.cells = "Nx = 40\nNy = 30\n";
    c.extra = probes;
    c.distributed = comm::size() > 1;
    const Outcome reference = run_case("dot_reductions_reference", c, 0);
    // The second write that finds the dot in the box
    std::vector<double> measures;
    if (comm::is_root()) {
        for (uint64_t step = 0; step <= 60; step += 2) {
            measures.push_back(value_at(reference, "dot_reductions.csv", N_DIM == 2 ? "area" : "volume", step));
        }
    }
    measures = comm::allgatherv(measures);
    uint64_t in = 0;
    for (size_t k = 1; k < measures.size() && in == 0; k++) {
        if (measures[k] > 0.0 && measures[k - 1] > 0.0) in = 2 * k;
    }
    ASSERT_FALSE(measures.empty());
    ASSERT_EQ(measures[0], 0.0) << "the dot is in the box at the start";
    ASSERT_NE(in, 0u) << "the dot does not enter the box";
    const std::string when =
        written_at(reference, "dot", {"integral_P", "mean_P", "max_RHO"}, "pt", "P", in) + " and step % 2 == 1";
    c.extra = probes + "[[triggers]]\nname = \"entered\"\nwhen = \"" + when + "\"\ndo = \"stop\"\n";
    const Outcome o = expect_queue_invariant("dot_reductions", c);
    EXPECT_EQ(o.step, in + 1);
}

TEST(StepQueueTest, SpectraTheDeviceSampledAtAStopOfTheQueueAreWrittenOnce) {
    // The trigger stops the queue at step 20, whose statistics and spectra the device sampled: the host writes the
    // staged sample there, once, and does not sample the statistics again
    std::ostringstream body;
    body << "[mesh]\ntype = \"cartesian\"\nNx = 12\nNy = 10\nLx = 6.283185307179586\nLy = 3.0\n";
    if (N_DIM == 3) body << "Nz = 6\nLz = 5.0\n";
    body << "[[periodic]]\nzones = [\"left\", \"right\"]\n[[periodic]]\nzones = [\"bottom\", \"top\"]\n";
    if (N_DIM == 3) body << "[[periodic]]\nzones = [\"back\", \"front\"]\n";
    body << "[initialize]\ntype = \"analytical\"\nrho = \"1.0 + 0.1 * sin(x + 2.1 * y)\"\n"
         << "u = " << vector_of("0.3 * sin(2 * x + 0.5) * cos(1.3 * y)", "0.2 * cos(3 * x - y)", "0.1 * sin(y)")
         << "\np = \"1.0 + 0.05 * cos(x - y)\"\n"
         << "[numerics]\nriemann_solver = \"HLLC\"\n[numerics.face_reconstruction]\ntype = \"MUSCL\"\n"
         << "[physics]\ntype = \"euler\"\ngamma = 1.4\np_ref = 1.0\nT_ref = 1.0\nrho_ref = 1.0\n";
    Case c;
    c.run = "n_steps = 30\ncfl = 0.4\n";
    c.body = body.str();
    c.extra = "check_interval = 1000\n" + RESTARTS +
              "[statistics]\ninterval = 2\nfields = [\"P\"]\n"
              "[statistics.spectra]\naverage = false\nfields = [\"U_X\", \"P\"]\ninterval = 4\nfile = \"{dir}/spectra.csv\"\n"
              "[[triggers]]\nname = \"at\"\nwhen = \"step == 20\"\ndo = \"checkpoint\"\n";
    c.distributed = comm::size() > 1;
    const Outcome o = expect_queue_invariant("staged_spectra_at_stop", c);
    if (!comm::is_root()) return;
    EXPECT_TRUE(o.files.count("restart_000020.restart"));
    EXPECT_TRUE(o.files.count("spectra_000020.csv"));
    // A file written twice has the same bytes: the run log records each write
    const auto spectra_writes = [&](const int depth) {
        std::vector<uint64_t> steps;
        const fs::path root = c.distributed ? scratch_dir() : rank_scratch_dir();
        std::ifstream in(root / ("staged_spectra_at_stop_" + std::to_string(depth)) / "mallard.jsonl");
        for (std::string line; std::getline(in, line);) {
            const json::Value v = json::parse(line);
            if (v["type"].string == "output" && v["kind"].string == "spectra") steps.push_back(uint64_t(v["step"].number));
        }
        return steps;
    };
    const std::vector<uint64_t> host = spectra_writes(0);
    EXPECT_EQ(std::count(host.begin(), host.end(), 20u), 1);
    for (const int depth : {1, 3}) EXPECT_EQ(spectra_writes(depth), host) << "step_queue = " << depth;
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

TEST(MPITest, GatedStepGraphsRunAsConditionalNodesDo) {
    // Where conditional nodes cannot hold NCCL operations (NCCL over a network), step graphs gate their bodies and
    // launch them also when they are not due: a line's reductions, reduced every 3 steps and read by a trigger that
    // stops at step 19, give the run of conditional nodes bit for bit
#ifndef KOKKOS_ENABLE_CUDA
    GTEST_SKIP() << "off CUDA every step graph gates its bodies";
#else
    if (Solver::are_cuda_graphs_off()) GTEST_SKIP() << "MALLARD_TEST_CUDA_GRAPHS=off";
    if (comm::size() == 1) GTEST_SKIP() << "MPI runs on several ranks only";
    Case c;
    c.run = "n_steps = 30\ncfl = 0.5\n";
    const std::string probes =
        "check_interval = 1000\n" + RESTARTS +
        "[[probes]]\nname = \"line\"\nfile = \"{dir}/line.csv\"\ninterval = 3\nfields = [\"P\", \"RHO\"]\n"
        "start = " + point_of("0.1", "0.3", "0.2") + "\nend = " + point_of("0.9", "0.6", "0.2") + "\nn_points = 7\n"
        "reductions = [\"integral\", \"mean\", \"favre\", \"rms\", \"min\", \"max\"]\n"
        "reductions_file = \"{dir}/line_reductions.csv\"\n"
        "[[probes]]\nname = \"pt\"\nfile = \"{dir}/pt.csv\"\ninterval = 3\nfields = [\"P\"]\n"
        "point = " + point_of("0.6", "0.5", "0.2") + "\n"
        "[integrals]\nfile = \"{dir}/integrals.csv\"\ninterval = 4\n";
    c.extra = probes;
    c.distributed = true;
    const Outcome reference = run_case("gated_reference", c, 0);
    const std::string when =
        written_at(reference, "line", {"integral_P", "mean_P", "favre_P", "rms_P", "min_P", "max_RHO"}, "pt", "P", 18) +
        " and step % 3 == 1";
    c.extra = probes + "[[triggers]]\nname = \"values\"\nwhen = \"" + when + "\"\ndo = \"stop\"\n";
    // Names of one length: restarts carry their run's record, with its length, in binary
    const Outcome nodes = run_case("nodes", c, 3);
    c.capture_fault = Solver::CaptureFault::CONDITIONAL_NODES;
    const Outcome gated = run_case("gated", c, 3);
    expect_same(nodes, gated, "gated step graphs");
    EXPECT_EQ(gated.step, 19u);
    EXPECT_EQ(gated.queued, nodes.queued);
    EXPECT_GT(gated.graph_steps, 0u);
    // Over a network the unfaulted run gates too, as Mallard decides from its trial of a conditional node
    EXPECT_EQ(nodes.gated, !nodes.conditional_nodes);
#ifdef Mallard_HAS_MPI
    MPI_Comm node;
    MPI_Comm_split_type(comm::communicator(), MPI_COMM_TYPE_SHARED, comm::rank(), MPI_INFO_NULL, &node);
    int on_node = 0;
    MPI_Comm_size(node, &on_node);
    MPI_Comm_free(&node);
    if (on_node == comm::size()) {
        EXPECT_TRUE(nodes.conditional_nodes) << "the ranks share a node";
    }
#endif
    EXPECT_TRUE(gated.gated);
#endif
}

TEST(StepQueueTest, AFailedStepGraphCaptureFallsBackToStepsKernelByKernel) {
#ifndef KOKKOS_ENABLE_CUDA
    GTEST_SKIP() << "step graphs are captured in CUDA builds only";
#else
    if (Solver::are_cuda_graphs_off()) GTEST_SKIP() << "MALLARD_TEST_CUDA_GRAPHS=off";
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
    bool host_steps = false;
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
    o.host_steps = migrates_particles(solver);
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
    // A progress row every step, also of runs with fused chemistry
    if (queued.host_steps) {
        EXPECT_EQ(queued.queued, 0u);
    } else {
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

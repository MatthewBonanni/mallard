/**
 * @file cpp_api_test.cpp
 * @brief The C++ library API (include/mallard/mallard.h): a case run through it matches the run of
 *        `mallard -i` bit for bit, and its input stays the run's record.
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 */

#include <gtest/gtest.h>

#include <algorithm>
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
#include "step_graph_guard.h"

namespace {

namespace fs = std::filesystem;

/** @brief A small shock tube with integrals, a region probe and a restart file at the end, its output in dir. */
std::string case_text(const std::string & dir, const std::string & run, int interval = 1, const std::string & extra = "") {
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
      << "[[write_data]]\nprefix = \"" << dir << "/restart\"\nformat = \"restart\"\ninterval = 1000000\n" << extra;
    return s.str();
}

/** @brief dir, created by rank 0, with the input of case_text written to dir/input.toml. */
std::string write_case(const std::string & dir, const std::string & run, int interval = 1, const std::string & extra = "") {
    if (comm::is_root()) {
        fs::create_directories(dir);
        std::ofstream(dir + "/input.toml") << case_text(dir, run, interval, extra);
    }
    comm::barrier();
    return dir + "/input.toml";
}

/** @brief The newest restart file of dir: every cell, or only these global cells. */
RestartData last_restart(const std::string & dir, const std::vector<uint64_t> * cells = nullptr) {
    std::string newest;
    for (const auto & entry : fs::directory_iterator(dir)) {
        const std::string name = entry.path().string();
        if (name.ends_with(".restart") && name > newest) newest = name;
    }
    return read_restart(newest, cells);
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

/** @brief Element (i, j) of a host field. */
template <typename T = rtype>
T at(const mallard::Field & f, int64_t i, int64_t j = 0) {
    return static_cast<const T *>(f.data)[i * f.strides[0] + (f.shape.size() > 1 ? j * f.strides[1] : 0)];
}

/** @brief A host field's values, row by row. */
std::vector<rtype> values(const mallard::Field & f) {
    std::vector<rtype> out;
    const int64_t n_columns = f.shape.size() > 1 ? f.shape[1] : 1;
    for (int64_t i = 0; i < f.shape[0]; i++) {
        for (int64_t j = 0; j < n_columns; j++) out.push_back(at(f, i, j));
    }
    return out;
}

/** @brief The global ids of this rank's owned cells. */
std::vector<uint64_t> global_ids(mallard::Solver & solver) {
    const mallard::Field ids = solver.field("GLOBAL_ID", mallard::Memory::host);
    std::vector<uint64_t> out;
    for (int64_t i = 0; i < ids.shape[0]; i++) out.push_back(at<uint64_t>(ids, i));
    return out;
}

/** @brief The kinds of the run log's event records in dir (rank 0, which writes the log). */
std::vector<std::string> event_kinds(const std::string & dir) {
    std::vector<std::string> kinds;
    std::ifstream in(dir + "/mallard.jsonl");
    for (std::string line; std::getline(in, line);) {
        json::Value v = json::parse(line);
        if (v["type"].string == "event") kinds.push_back(v["kind"].string);
    }
    return kinds;
}

/** @brief The lines of dir's run log (rank 0, which writes it). */
std::vector<std::string> log_lines(const std::string & dir) {
    std::vector<std::string> lines;
    std::ifstream in(dir + "/mallard.jsonl");
    for (std::string line; std::getline(in, line);) lines.push_back(line);
    return lines;
}

/** @brief The case saved as dir/input.toml, dir created by rank 0. */
mallard::Case saved(const std::string & dir, const mallard::Case & c) {
    if (comm::is_root()) fs::create_directories(dir);
    comm::barrier();
    return c.save(dir + "/input.toml");
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
    ASSERT_TRUE(solver.monitor("box_mean_P"));
    EXPECT_EQ(solver.monitor("box_mean_P")->step, 10u);
    ASSERT_TRUE(solver.monitor("integrals_kinetic_energy"));
    EXPECT_EQ(solver.monitor("integrals_kinetic_energy")->unit, Mallard_DIM == 2 ? "kg m s-2" : "kg m2 s-2");
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

TEST(CppApiTest, FieldsViewTheSolversStateAndMatchItsRestartFile) {
    mallard::Solver solver(mallard::Case::from_toml_file(write_case("fields", "n_steps = 4\n", 2)));
    while (solver.step()) {
    }
    const mallard::Field rho = solver.field("RHO"), block = solver.field("conservatives");
    // Views of the solver's memory: a variable is its column of the block, and every view the same memory
    EXPECT_EQ(rho.data, block.data);
    EXPECT_EQ(solver.field("RHO").data, rho.data);
    EXPECT_EQ(static_cast<const char *>(solver.field("RHOE").data) - static_cast<const char *>(rho.data),
              (N_CONSERVATIVE - 1) * block.strides[1] * static_cast<std::ptrdiff_t>(sizeof(rtype)));
    ASSERT_EQ(block.shape.size(), 2u);
    EXPECT_EQ(block.shape[1], N_CONSERVATIVE);
    EXPECT_EQ(rho.shape, std::vector<int64_t>{block.shape[0]});
    EXPECT_FALSE(rho.writable);
    EXPECT_EQ(rho.unit, "kg m-3");
    EXPECT_EQ(rho.type, sizeof(rtype) == 4 ? mallard::ScalarType::float32 : mallard::ScalarType::float64);
    const mallard::Field host = solver.field("conservatives", mallard::Memory::host);
    EXPECT_EQ(host.device, mallard::Device::host);
    // CPU builds: the host's view is the solver's memory, not a copy
    if (block.device == mallard::Device::host) {
        EXPECT_EQ(host.data, block.data);
    }

    const std::vector<rtype> state = values(host);
    const mallard::Field P = solver.field("P", mallard::Memory::host);
    EXPECT_EQ(P.unit, "kg m-1 s-2");
    for (int64_t i = 0; i < host.shape[0]; i++) {
        double kinetic = 0.0;
        for (int d = 0; d < Mallard_DIM; d++) kinetic += 0.5 * std::pow(double(at(host, i, 1 + d)), 2) / double(at(host, i));
        const double p = 0.4 * (double(at(host, i, N_CONSERVATIVE - 1)) - kinetic);
        EXPECT_NEAR(double(at(P, i)), p, (sizeof(rtype) == 4 ? 1e-5 : 1e-12) * p) << "cell " << i;
    }
    const mallard::Field volume = solver.field("VOLUME", mallard::Memory::host);
    double total = 0.0;
    for (int64_t i = 0; i < volume.shape[0]; i++) total += double(at(volume, i));
    EXPECT_NEAR(comm::allreduce(total, comm::Op::SUM), 1.0, 1e-5);
    EXPECT_EQ(solver.field("CENTROID").shape[1], Mallard_DIM);
    const std::vector<uint64_t> ids = global_ids(solver);
    try {
        solver.field("RHOO");
        ADD_FAILURE() << "no error";
    } catch (const mallard::Error & e) {
        EXPECT_NE(std::string(e.what()).find("RHOE"), std::string::npos) << e.what();
    }
    EXPECT_EQ(solver.finish(), 0);

    const RestartData restart = last_restart("fields", &ids);
    ASSERT_EQ(restart.fields[0].size(), ids.size());
    for (size_t v = 0; v < N_CONSERVATIVE; v++) {
        const std::vector<rtype> & expected = *restart.find(CONSERVATIVE_NAMES[v]);
        for (size_t i = 0; i < ids.size(); i++) EXPECT_EQ(state[i * N_CONSERVATIVE + v], expected[i]) << CONSERVATIVE_NAMES[v];
    }
}

TEST(CppApiTest, AStateSetBetweenStepsIsWhereTheNextStepStartsFrom) {
    // Steps 4-6 of one run, taken by another from the state the first reached at step 3 (time does not enter
    // this case): the state set is taken as is, as from a restart
    mallard::Solver first(mallard::Case::from_toml_file(write_case("first", "n_steps = 6\n", 5)));
    for (int i = 0; i < 3; i++) first.step();
    const std::vector<rtype> at_3 = values(first.field("conservatives", mallard::Memory::host));

    mallard::Solver second(mallard::Case::from_toml_file(write_case("second", "n_steps = 3\n", 5)));
    second.setup();
    const mallard::Field state = second.writable_field("conservatives", mallard::Memory::host);
    EXPECT_TRUE(state.writable);
    for (int64_t i = 0; i < state.shape[0]; i++) {
        for (int64_t j = 0; j < N_CONSERVATIVE; j++) {
            static_cast<rtype *>(state.data)[i * state.strides[0] + j * state.strides[1]] =
                at_3[size_t(i * N_CONSERVATIVE + j)];
        }
    }
    // Writes to the solver's memory are seen by its views before they are committed
    if (state.device == second.field("RHO").device) {
        EXPECT_EQ(second.field("RHO").data, state.data);
    }
    while (second.step()) {
    }
    while (first.step()) {
    }
    EXPECT_EQ(values(second.field("conservatives", mallard::Memory::host)),
              values(first.field("conservatives", mallard::Memory::host)));
    second.finish();
    first.finish();
    if (comm::is_root()) {
        const std::vector<std::string> kinds = event_kinds("second");
        EXPECT_EQ(std::count(kinds.begin(), kinds.end(), "state_set"), 1);
    }
}

TEST(CppApiTest, PerStepCallbacksSeeEveryStepAndLeaveTheRunBitForBit) {
    const std::vector<double> reference = executable_run(write_case("executable", "n_steps = 12\n", 5));
    mallard::Solver solver(mallard::Case::from_toml_file(write_case("api", "n_steps = 12\n", 5)));
    std::vector<uint64_t> steps;
    solver.on_step([&](mallard::Solver & s) {
        steps.push_back(s.step_count());
        s.field("P");
        EXPECT_THROW(s.step(), std::logic_error);
    });
    EXPECT_FALSE(solver.advance_to(1.0));
    std::vector<uint64_t> expected(12);
    for (uint64_t i = 0; i < 12; i++) expected[i] = i + 1;
    EXPECT_EQ(steps, expected);
    expect_same_values(solver.monitors(), reference);
    EXPECT_EQ(solver.finish(), 0);
    const RestartData a = last_restart("executable"), b = last_restart("api");
    EXPECT_EQ(b.fields, a.fields);
}

TEST(CppApiTest, PerStageCallbacksSeeEachStageOfTheTimeIntegrator) {
    const std::vector<double> reference = executable_run(write_case("executable", "n_steps = 6\n", 5));
    mallard::Solver solver(mallard::Case::from_toml_file(write_case("api", "n_steps = 6\n", 5)));
    solver.setup();
    const std::vector<rtype> initial = values(solver.field("RHO", mallard::Memory::host));
    std::vector<mallard::Stage> stages;
    std::vector<rtype> first_stage;
    const mallard::CallbackId id = solver.on_stage([&](mallard::Solver & s, const mallard::Stage & stage) {
        stages.push_back(stage);
        if (stages.size() == 1) first_stage = values(s.field("RHO", mallard::Memory::host));
        EXPECT_THROW(s.field("P"), mallard::Error);
        EXPECT_THROW(s.writable_field("RHO"), mallard::Error);
    });
    for (int i = 0; i < 3; i++) solver.step();
    // SSPRK3: stages at t_n, t_n + dt and t_n + dt / 2
    ASSERT_EQ(stages.size(), 9u);
    for (size_t k = 0; k < stages.size(); k++) {
        EXPECT_EQ(stages[k].index, int(k % 3));
        EXPECT_EQ(stages[k].step, k / 3);
    }
    EXPECT_EQ(stages[0].time, 0.0);
    EXPECT_EQ(stages[3].time, stages[1].time);
    EXPECT_EQ(stages[2].time, 0.5 * stages[1].time);
    EXPECT_EQ(first_stage, initial);
    // Then the device queue again
    solver.remove_callback(id);
    while (solver.step()) {
    }
    EXPECT_EQ(stages.size(), 9u);
    expect_same_values(solver.monitors(), reference);
    EXPECT_EQ(solver.finish(), 0);
    const RestartData a = last_restart("executable"), b = last_restart("api");
    EXPECT_EQ(b.fields, a.fields);
}

TEST(CppApiTest, StatisticsAveragesAreFieldsAsTheRestartFileHasThem) {
    const std::string statistics = "[statistics]\nfields = [\"RHO\", \"P\"]\nproducts = [\"RHO*P\"]\ninterval = 1\n";
    mallard::Solver solver(mallard::Case::from_toml_file(write_case("statistics", "n_steps = 4\n", 5, statistics)));
    EXPECT_TRUE(std::isnan(solver.statistics().start));
    while (solver.step()) {
    }
    const std::vector<std::string> names = solver.field_names();
    EXPECT_NE(std::find(names.begin(), names.end(), "COV_RHO_P"), names.end());
    const mallard::Statistics s = solver.statistics();
    EXPECT_EQ(s.samples, 4u);
    EXPECT_EQ(s.end, solver.time());
    const mallard::Field mean = solver.field("MEAN_RHO", mallard::Memory::host);
    EXPECT_EQ(mean.unit, "kg m-3");
    const std::vector<rtype> means = values(mean), covariances = values(solver.field("COV_RHO_P", mallard::Memory::host));
    const std::vector<uint64_t> ids = global_ids(solver);
    EXPECT_EQ(solver.finish(), 0);
    const RestartData restart = last_restart("statistics", &ids);
    EXPECT_EQ(means, *restart.find("MEAN_RHO"));
    EXPECT_EQ(covariances, *restart.find("COV_RHO_P"));
}

TEST(CppApiTest, ACaseOfTheOtherDimensionIsRefused) {
    const mallard::Case same = mallard::Case::from_toml(case_text("dimension", "n_steps = 1\n"));
    EXPECT_EQ(same.dimension(), Mallard_DIM);
#if Mallard_DIM == 2
    const std::string mesh = "[mesh]\ntype = \"cartesian\"\nNx = 2\nNy = 2\nNz = 2\n";
#else
    const std::string mesh = "[mesh]\ntype = \"cartesian_tri\"\nNx = 2\nNy = 2\n";
#endif
    const mallard::Case other = mallard::Case::from_toml("[run]\nn_steps = 1\n" + mesh);
    EXPECT_EQ(other.dimension(), 5 - Mallard_DIM);
    try {
        mallard::Solver solver(other);
        ADD_FAILURE() << "no error";
    } catch (const mallard::Error & e) {
        EXPECT_EQ(e.code(), "E-INPUT");
        EXPECT_NE(std::string(e.what()).find("libmallard" + std::to_string(5 - Mallard_DIM) + "d"), std::string::npos)
            << e.what();
    }
}

TEST(CppApiTest, ASessionOnACommunicatorRunsOnACopyOfIt) {
    char name[] = "mallard";
    char * args[] = {name, nullptr};
    int argc = 1;
    char ** argv = args;
#ifdef Mallard_HAS_MPI
    const int world = comm::size();
    MPI_Comm self = MPI_COMM_SELF;
    {
        mallard::Session session(argc, argv, &self);
        EXPECT_EQ(comm::size(), 1);
        EXPECT_EQ(comm::rank(), 0);
        int same = MPI_UNEQUAL;
        MPI_Comm_compare(comm::communicator(), MPI_COMM_SELF, &same);
        EXPECT_EQ(same, MPI_CONGRUENT);
    }
    EXPECT_EQ(comm::size(), world);
#else
    int comm = 0;
    try {
        mallard::Session session(argc, argv, &comm);
        ADD_FAILURE() << "no error";
    } catch (const mallard::Error & e) {
        EXPECT_EQ(e.code(), "E-SETUP");
        EXPECT_NE(std::string(e.what()).find("Mallard_ENABLE_MPI"), std::string::npos) << e.what();
    }
#endif
}

TEST(CppApiTest, StepNTakesTheStepsOfOneAtATime) {
    const std::vector<double> reference = executable_run(write_case("executable", "n_steps = 10\n", 5));
    mallard::Solver solver(mallard::Case::from_toml_file(write_case("api", "n_steps = 10\n", 5)));
    EXPECT_TRUE(solver.step(0));
    EXPECT_EQ(solver.step_count(), 0u);
    EXPECT_TRUE(solver.step(4));
    EXPECT_EQ(solver.step_count(), 4u);
    // The stop condition ends the steps early
    EXPECT_FALSE(solver.step(100));
    EXPECT_EQ(solver.step_count(), 10u);
    EXPECT_FALSE(solver.step(1));
    expect_same_values(solver.monitors(), reference);
    EXPECT_EQ(solver.finish(), 0);
    const RestartData a = last_restart("executable"), b = last_restart("api");
    EXPECT_EQ(b.fields, a.fields);
}

TEST(CppApiTest, ProgressOutputAndRecordCallbacksGetTheRunLogsRecords) {
    const auto with = [](const std::string & dir) {
        const std::string snapshots =
            "[[write_data]]\nprefix = \"" + dir + "/flow\"\nformat = \"vtu\"\nfields = [\"RHO\"]\ninterval = 4\n";
        return mallard::Case::from_toml(case_text(dir, "n_steps = 8\n", 4, snapshots)).set("output.check_interval", 2);
    };
    executable_run(saved("executable", with("executable")).path());

    mallard::Solver solver(saved("records", with("records")));
    std::vector<mallard::Progress> rows;
    std::vector<mallard::Output> outputs;
    std::vector<mallard::Record> records;
    solver.on_progress([&](const mallard::Progress & p) {
        rows.push_back(p);
        // The record only: the device may be steps ahead of it
        EXPECT_THROW(solver.field("RHO"), std::logic_error);
        EXPECT_THROW(solver.step(), std::logic_error);
    });
    solver.on_output([&](const mallard::Output & o) { outputs.push_back(o); });
    solver.on_record([&](const mallard::Record & r) { records.push_back(r); });
    EXPECT_EQ(solver.run(), 0);

    std::vector<uint64_t> steps;
    for (const mallard::Progress & p : rows) steps.push_back(p.step);
    EXPECT_EQ(steps, (std::vector<uint64_t>{0, 2, 4, 6, 8}));
    EXPECT_EQ(rows.back().fraction, 1.0);
    EXPECT_EQ(rows.back().time, solver.time());
    ASSERT_FALSE(rows.back().diagnostics.empty());
    EXPECT_EQ(rows.back().diagnostics[0].first, "min_rho");
    const RestartData a = last_restart("executable"), b = last_restart("records");
    EXPECT_EQ(b.fields, a.fields);
    if (!comm::is_root()) {
        EXPECT_TRUE(outputs.empty());
        EXPECT_TRUE(records.empty());
        return;
    }
    // Every record of the run log, as written; progress and output records as structs too
    const std::vector<std::string> lines = log_lines("records");
    ASSERT_EQ(records.size(), lines.size());
    std::vector<std::string> progress_lines, output_lines;
    for (size_t i = 0; i < lines.size(); i++) {
        EXPECT_EQ(records[i].json, lines[i]);
        EXPECT_EQ(records[i].type, json::parse(lines[i])["type"].string);
        if (records[i].type == "progress") progress_lines.push_back(lines[i]);
        if (records[i].type == "output") output_lines.push_back(lines[i]);
    }
    ASSERT_EQ(progress_lines.size(), rows.size());
    for (size_t i = 0; i < rows.size(); i++) EXPECT_EQ(rows[i].json, progress_lines[i]);
    ASSERT_EQ(outputs.size(), output_lines.size());
    std::vector<std::string> snapshots;
    for (size_t i = 0; i < outputs.size(); i++) {
        EXPECT_EQ(outputs[i].json, output_lines[i]);
        ASSERT_FALSE(outputs[i].files.empty());
        EXPECT_GT(outputs[i].files[0].bytes, 0u);
        EXPECT_TRUE(fs::exists(outputs[i].files[0].path)) << outputs[i].files[0].path;
        if (outputs[i].kind == "vtu") snapshots.push_back(std::to_string(outputs[i].step) + " " + outputs[i].reason);
    }
    EXPECT_EQ(snapshots, (std::vector<std::string>{"0 start", "4 interval", "8 interval"}));
}

TEST(CppApiTest, ALogSinkTakesTheConsoleOutput) {
    std::vector<mallard::LogMessage> messages;
    {
        mallard::Solver solver(mallard::Case::from_toml_file(write_case("sink", "n_steps = 2\n")));
        solver.set_log([&](const mallard::LogMessage & m) { messages.push_back(m); });
        testing::internal::CaptureStdout();
        EXPECT_EQ(solver.run(), 0);
        EXPECT_EQ(testing::internal::GetCapturedStdout(), "");
    }
    if (comm::is_root()) {
        ASSERT_FALSE(messages.empty());
        bool run_section = false;
        for (const mallard::LogMessage & m : messages) {
            EXPECT_EQ(m.level, mallard::LogLevel::info);
            EXPECT_EQ(m.text.find('\n'), std::string::npos) << m.text;
            EXPECT_EQ(m.text.find('\033'), std::string::npos) << m.text;
            run_section = run_section || m.text.rfind("== Run ", 0) == 0;
        }
        EXPECT_TRUE(run_section);
    } else {
        EXPECT_TRUE(messages.empty());
    }
    // A solver without a sink prints as mallard -i does
    mallard::Solver solver(mallard::Case::from_toml_file(write_case("console", "n_steps = 2\n")));
    testing::internal::CaptureStdout();
    EXPECT_EQ(solver.run(), 0);
    const std::string printed = testing::internal::GetCapturedStdout();
    if (comm::is_root()) {
        EXPECT_NE(printed.find("== Run "), std::string::npos);
    }
}

TEST(CppApiTest, GraphedAndUngraphedRunsWithCallbacksMatchBitForBit) {
    // Progress rows every 4 steps and snapshots every 6, which the host acts on anyway: steps queued on the device
    // (in CUDA graphs where the build has them) take the steps of the host, kernel by kernel, bit for bit
    struct Run {
        std::vector<std::string> rows, outputs;
        RestartData restart;
    };
    const auto with = [](const std::string & dir, const std::string & run) {
        const std::string snapshots =
            "[[write_data]]\nprefix = \"" + dir + "/flow\"\nformat = \"vtu\"\nfields = [\"RHO\"]\ninterval = 6\n";
        return mallard::Case::from_toml(case_text(dir, "n_steps = 12\n" + run, 3, snapshots))
            .set("output.check_interval", 4);
    };
    const auto row = [](uint64_t step, double t, double dt, const std::vector<std::pair<std::string, double>> & d) {
        std::ostringstream s;
        s.precision(17);
        s << step << " " << t << " " << dt;
        for (const auto & [name, value] : d) s << " " << name << "=" << value;
        return s.str();
    };
    const auto api_run = [&](const std::string & dir, const std::string & run) {
        Run r;
        mallard::Solver solver(saved(dir, with(dir, run)));
        solver.set_log([](const mallard::LogMessage &) {});
        solver.on_progress([&](const mallard::Progress & p) {
            r.rows.push_back(row(p.step, p.time, p.dt, p.diagnostics));
        });
        solver.on_output([&](const mallard::Output & o) {
            r.outputs.push_back(o.kind + " " + std::to_string(o.step) + " " + fs::path(o.path).filename().string());
        });
        while (solver.step(5)) {
        }
        EXPECT_EQ(solver.finish(), 0);
        r.restart = last_restart(dir);
        return r;
    };
    const Run host = api_run("host", "step_queue = 0\ncuda_graphs = false\n");
    const Run queued = api_run("queued", "");
    EXPECT_EQ(queued.restart.fields, host.restart.fields);
    EXPECT_EQ(queued.rows, host.rows);
    EXPECT_EQ(queued.outputs, host.outputs);
    EXPECT_EQ(host.rows.size(), 4u);
    if (comm::is_root()) {
        EXPECT_FALSE(host.outputs.empty());
    }

    // The step graphs' path on every backend (uncaptured off CUDA): nothing of the callbacks is in a capture
    std::vector<std::string> rows;
    {
        const StepGraphGuard guard;
        ::Solver solver;
        solver.set_uncaptured_step_graphs(true);
        solver.set_progress_hook([&](const RunStatus & s, const std::string &) {
            std::vector<std::pair<std::string, double>> d = s.monitors;
            rows.push_back(row(s.step, s.t, s.dt, d));
        });
        solver.init(saved("graphed", with("graphed", "")).path());
        EXPECT_EQ(solver.run(), 0);
        for (const std::string & v : guard.violations()) ADD_FAILURE() << "in a step graph's capture: " << v;
        if (!Solver::are_cuda_graphs_off()) {
            EXPECT_GT(guard.captures(), 0u);
        }
    }
    EXPECT_EQ(rows, host.rows);
    EXPECT_EQ(last_restart("graphed").fields, host.restart.fields);
}

TEST(CppApiTest, ACaseSetInCodeEmitsTheTomlThatReproducesTheRun) {
    const auto built = [](const std::string & dir) {
        return mallard::Case::from_toml(case_text(dir, "n_steps = 3\n", 5))
            .set("run.n_steps", 6)
            .set("run.cfl", 0.1 + 0.2)
            .set("numerics.riemann_solver", "HLL")
            .set("probes[0].interval", 2)
            .set("statistics.fields", {"RHO", "P"})
            .set("statistics.interval", 1);
    };
    // Only the lines of the keys set change; a new table goes at the end
    std::string expected = case_text("api", "n_steps = 3\n", 5);
    const auto replace = [&](const std::string & from, const std::string & to) {
        const size_t at = expected.find(from);
        ASSERT_NE(at, std::string::npos) << from;
        expected.replace(at, from.size(), to);
    };
    replace("n_steps = 3\n", "n_steps = 6\n");
    replace("cfl = 0.4\n", "cfl = 0.30000000000000004\n");
    replace("riemann_solver = \"HLLC\"", "riemann_solver = \"HLL\"");
    replace("reductions = [\"mean\"]\ninterval = 5", "reductions = [\"mean\"]\ninterval = 2");
    expected += "\n[statistics]\nfields = [\"RHO\", \"P\"]\ninterval = 1\n";
    const mallard::Case c = built("api");
    EXPECT_EQ(c.toml(), expected);
    EXPECT_TRUE(c.path().empty());

    // mallard -i on the emitted TOML is the run of the case built in code, bit for bit
    const std::string file = write_case("executable", "n_steps = 1\n");
    if (comm::is_root()) std::ofstream(file) << built("executable").toml();
    comm::barrier();
    const std::vector<double> reference = executable_run(file);
    if (comm::is_root()) fs::create_directories("api");
    comm::barrier();
    mallard::Solver solver(c);
    EXPECT_EQ(solver.run(), 0);
    expect_same_values(solver.monitors(), reference);
    const RestartData a = last_restart("executable"), b = last_restart("api");
    EXPECT_EQ(b.step, 6u);
    EXPECT_EQ(b.names, a.names);
    EXPECT_EQ(b.fields, a.fields);
    if (comm::is_root()) {
        EXPECT_EQ(record("api", "start")["provenance"]["input"]["text"].string, c.toml());
    }

    // save: the file a run then builds from
    const mallard::Case on_disk = c.save("api/saved.toml");
    EXPECT_EQ(on_disk.path(), "api/saved.toml");
    EXPECT_EQ(on_disk.toml(), c.toml());

    // A key of an inline table: the text is reformatted, its comments kept, with the value set
    const mallard::Case reformatted =
        mallard::Case::from_toml("# box\n" + c.toml()).set("probes[0].region.center", {0.25, 0.5});
    EXPECT_NE(reformatted.toml().find("# box"), std::string::npos);
    EXPECT_NE(reformatted.toml().find("0.25"), std::string::npos);
    EXPECT_EQ(reformatted.set("probes[0].region.center", {0.25, 0.5}).toml(), reformatted.toml());

    for (const auto & [key, value] : std::vector<std::pair<std::string, mallard::Value>>{
             {"run..cfl", 1}, {"probes[3].interval", 1}, {"run.cfl", mallard::Value::toml("[1,")}, {"run.cfl[0]", 1}}) {
        try {
            c.set(key, value);
            ADD_FAILURE() << key << ": no error";
        } catch (const mallard::Error & e) {
            EXPECT_EQ(e.code(), "E-INPUT") << key;
        }
    }
}

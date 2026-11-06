/**
 * @file check_test.cpp
 * @brief Pre-flight check (mallard check): unused keys, boundary coverage,
 *        mesh quality, time step, memory, cost and warnings.
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 */

#include <gtest/gtest.h>

#include <cmath>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <map>
#include <numeric>
#include <numbers>
#include <sstream>
#include <stdexcept>
#include <string>

#include "hdf5_output.h"
#include "json.h"
#include "mesh_block.h"
#include "data_paths.h"
#include "test_fixtures.h"
#include "check.h"
#include "comm.h"
#include "common_io.h"
#include "memory_tracker.h"
#include "mesh_block.h"
#include "mesh_quality.h"
#include "partition.h"
#include "report.h"
#include "scratch.h"
#include "solver.h"

namespace {

const char * const U_ZERO = N_DIM == 3 ? "[\"0.0\", \"0.0\", \"0.0\"]" : "[\"0.0\", \"0.0\"]";

/** @brief A periodic-free box of Nx x 8 cells and size 2 x 1 with wall boundaries; u_x as an expression of x. */
std::string box(const std::string & u_x = "1.0", const std::string & run = "t_stop = 0.5\ncfl = 0.8\n",
                const std::string & extra = "", const std::vector<std::string> & zones = {}) {
    std::ostringstream s;
    s << "[run]\n" << run
      << "[mesh]\ntype = \"cartesian\"\nNx = 16\nNy = 8\nLx = 2.0\nLy = 1.0\n"
      << (N_DIM == 3 ? "Nz = 2\nLz = 0.25\n" : "")
      << "[initialize]\ntype = \"analytical\"\nrho = \"1.0\"\np = \"1.0\"\nu = [\"" << u_x << "\", \"0.0\""
      << (N_DIM == 3 ? ", \"0.0\"" : "") << "]\n"
      << "[numerics]\nriemann_solver = \"HLLC\"\ntime_integrator = \"SSPRK3\"\n"
      << "[physics]\ntype = \"euler\"\ngamma = 1.4\np_ref = 1.0\nT_ref = 1.0\nrho_ref = 1.0\n";
    std::vector<std::string> names = zones;
    if (names.empty()) {
        names = {"left", "right", "bottom", "top"};
        if (N_DIM == 3) names.insert(names.end(), {"back", "front"});
    }
    for (const auto & name : names) s << "[[boundaries]]\nname = \"" << name << "\"\ntype = \"symmetry\"\n";
    s << extra;
    return s.str();
}

CheckReport check(const std::string & input, const CheckOptions & options = {}) {
    Solver solver;
    return solver.check(parse_input_string(input), options);
}

/** @brief The first error's message, or empty. */
std::string first_error(const CheckReport & r) {
    const std::vector<std::string> errors = r.errors();
    return errors.empty() ? "" : errors.front();
}

bool mentions(const std::vector<std::string> & messages, const std::string & text) {
    for (const auto & m : messages) {
        if (m.find(text) != std::string::npos) return true;
    }
    return false;
}

} // namespace

TEST(CheckKeys, MisspelledKeysAreReportedWithLineAndSuggestion) {
    std::string input = box();
    input.replace(input.find("Ny = 8"), 6, "Nyy = 8");
    input.replace(input.find("[numerics]\nriemann_solver"), 25, "[numerics]\nriemann_solvr");
    input += "[foo]\nbar = 1\n";
    const CheckReport r = check(input);
    ASSERT_EQ(r.unused.size(), 3u);
    EXPECT_EQ(r.unused[0].key, "mesh.Nyy");
    EXPECT_EQ(r.unused[0].suggestion, "mesh.Ny");
    EXPECT_EQ(r.unused[0].line, 7u);
    EXPECT_EQ(r.unused[0].docs, "docs/input.md#mesh");
    EXPECT_EQ(r.unused[1].key, "numerics.riemann_solvr");
    EXPECT_EQ(r.unused[1].suggestion, "numerics.riemann_solver");
    EXPECT_EQ(r.unused[2].key, "foo");  // a table nobody read, not its keys
    EXPECT_EQ(r.unused[2].suggestion, "");
    EXPECT_FALSE(r.ok());
}

TEST(CheckKeys, UnclosedNestedArrayBeforeAKeyIsASyntaxError) {
    // Once an endless loop in toml11's error recovery (test/fuzz/corpus/input/unclosed_nested_array)
    EXPECT_THROW(parse_input_string("s=[\"\",[\"\",\"\"[\n[="), InputError);
}

TEST(CheckKeys, MalformedNestingIsASyntaxErrorNotACrash) {
    // Once a stack overflow and endless loops in toml11's error recovery (test/fuzz/corpus/input/
    // deep_unclosed_array, comment_in_unclosed_array, key_on_the_line_of_a_nested_array)
    EXPECT_THROW(parse_input_string("s=[\"\",[\"\"" + std::string(1000000, '[')), InputError);
    EXPECT_THROW(parse_input_string("s=[\"\",[\"\"[#\n"), InputError);
    EXPECT_THROW(parse_input_string("s=[\n\"\",[\"\"[="), InputError);
}

TEST(CheckKeys, NestingDeeperThan64IsAnInputError) {
    const auto nested = [](size_t depth) { return std::string(depth, '[') + std::string(depth, ']'); };
    const auto error = [](const std::string & text) {
        try {
            parse_input_string(text);
        } catch (const InputError & e) {
            return std::string(e.what());
        }
        return std::string("no error");
    };
    const std::string too_deep = "tables, arrays and inline tables are nested more than 64 deep";
    EXPECT_NO_THROW(parse_input_string("a = " + nested(64)));
    EXPECT_TRUE(mentions({error("x = 1\na = " + nested(65))}, "line 2: a: " + too_deep));
    EXPECT_TRUE(mentions({error("[t]\na = " + nested(64))}, "line 2: t.a: " + too_deep));
    std::string dotted = "a";
    for (int i = 0; i < 64; i++) dotted += ".a";
    EXPECT_TRUE(mentions({error(dotted + " = 1")}, too_deep));
}

TEST(CheckKeys, RemovedKeysNameWhatReplacedThem) {
    std::string input = box();
    input.replace(input.find("rho_ref = 1.0\n"), 14, "rho_ref = 1.0\np_min = 0.0\n");
    const CheckReport r = check(input);
    EXPECT_FALSE(r.ok());
    EXPECT_TRUE(mentions(r.errors(), "physics.p_min was removed in 0.2.0")) << first_error(r);
    EXPECT_TRUE(mentions(r.errors(), "mallard input upgrade"));
}

TEST(CheckKeys, KeyInTheWrongTableSuggestsWhereItIsRead) {
    std::string input = box();
    input.replace(input.find("Lx = 2.0\n"), 9, "Lx = 2.0\ncfl = 0.5\n");
    const CheckReport r = check(input);
    ASSERT_EQ(r.unused.size(), 1u);
    EXPECT_EQ(r.unused[0].key, "mesh.cfl");
    EXPECT_EQ(r.unused[0].suggestion, "run.cfl");
}

TEST(CheckKeys, ShortKeysGetNoFarFetchedSuggestion) {
    std::string input = box();
    input.replace(input.find("p = \"1.0\""), 9, "q = \"1.0\"\np = \"1.0\"");
    const CheckReport r = check(input);
    ASSERT_EQ(r.unused.size(), 1u);
    EXPECT_EQ(r.unused[0].key, "initialize.q");
    EXPECT_EQ(r.unused[0].suggestion, "");
}

TEST(CheckKeys, AValidInputUsesEveryKeyIncludingArraysOfTables) {
    const std::string dir = (scratch_dir() / "mallard_check_valid").string();
    const CheckReport r = check(box("1.0", "t_stop = 0.5\ncfl = 0.8\n",
                                    "[[write_data]]\nprefix = \"" + dir + "/flow\"\nformat = \"vtu\"\ninterval = 5\n"
                                    "variables = [\"RHO\", \"P\"]\n[[forces]]\nzone = \"bottom\"\nfile = \"" + dir +
                                        "/forces.csv\"\n"));
    EXPECT_TRUE(r.unused.empty()) << (r.unused.empty() ? "" : describe(r.unused[0]));
    EXPECT_TRUE(r.ok());
}

TEST(CheckKeys, NormalSetupRejectsUnusedKeys) {
    std::string input = box();
    input.replace(input.find("Ny = 8"), 6, "Nyy = 8");
    Solver solver;
    try {
        solver.init(parse_input_string(input));
        FAIL() << "no error";
    } catch (const InputError & e) {
        EXPECT_NE(std::string(e.what()).find("line 7: mesh.Nyy (did you mean mesh.Ny?)"), std::string::npos)
            << e.what();
    }
}

TEST(CheckKeys, SetupErrorsNameLikelyMisspellings) {
    std::string input = box();
    input.replace(input.find("cfl = 0.8"), 9, "clf = 0.8");
    Solver solver;
    try {
        solver.init(parse_input_string(input));
        FAIL() << "no error";
    } catch (const InputError & e) {
        const std::string text = e.what();
        EXPECT_NE(text.find("specify exactly one of dt and cfl"), std::string::npos) << text;
        EXPECT_NE(text.find("line 3: run.clf (did you mean run.cfl?)"), std::string::npos) << text;
    }
}

TEST(CheckBoundaries, ZonesWithoutAConditionAreNamed) {
    std::vector<std::string> zones = {"left", "right", "bottom"};
    if (N_DIM == 3) zones.insert(zones.end(), {"back", "front"});
    const CheckReport r = check(box("1.0", "t_stop = 0.5\ncfl = 0.8\n", "", zones));
    EXPECT_FALSE(r.setup_done);
    const uint32_t n_top = N_DIM == 3 ? 16 * 2 : 16;
    EXPECT_TRUE(mentions(r.errors(), "no condition for zone top (" + std::to_string(n_top) + " faces)"))
        << (first_error(r));
    EXPECT_FALSE(r.ok());
}

TEST(CheckTimeStep, MatchesTheUniformGridFormulaAndFindsTheFastestRegion) {
    // dt = cfl h / (|u| + |v| + 2a) on a uniform 2D grid (docs/input.md); the
    // flow is 3 times faster for x > 1.5, so that is where dt is set
    const double a = std::sqrt(1.4), h = 2.0 / 16;
    const CheckReport r = check(box("1.0 + 2.0 * (x > 1.5)"));
    ASSERT_TRUE(r.ok()) << (first_error(r));
    if (N_DIM == 2) {
        EXPECT_NEAR(r.dt, 0.8 * h / (3.0 + 2.0 * a), roundoff(1e-12));
    }
    EXPECT_EQ(r.dt_limit, "CFL");
    EXPECT_NEAR(r.cfl, 0.8, roundoff(1e-12));
    EXPECT_GT(r.limiting_x[0], 1.5);
    EXPECT_GT(r.region_lo[0], 1.5);
    EXPECT_GT(r.n_region_cells, 0u);
}

TEST(CheckTimeStep, FixedTimeStepReportsItsCflAndWarnsAboveTheIntegratorsLimit) {
    const CheckReport ok = check(box("1.0", "t_stop = 0.5\ndt = 1e-3\n"));
    EXPECT_NEAR(ok.dt, 1e-3, roundoff(1e-18));
    EXPECT_NEAR(ok.cfl, 1e-3 / ok.dt_cfl1, roundoff(1e-12));
    EXPECT_FALSE(mentions(ok.warnings(), "stability limit"));
    const CheckReport fast = check(box("1.0", "t_stop = 0.5\ncfl = 1.5\n"));
    EXPECT_TRUE(mentions(fast.warnings(), "stability limit of SSPRK3"));
    EXPECT_TRUE(fast.ok());  // a warning, not an error
}

TEST(CheckCost, GivenThroughputSetsWallTimePerUnitTimeAndFlowThrough) {
    CheckOptions options;
    options.throughput = 1e6;
    options.ranks = comm::size() == 1 ? 2 : comm::size();  // estimates for other counts need a serial check
    const CheckReport r = check(box(), options);
    ASSERT_TRUE(r.ok());
    const double n_cells = N_DIM == 3 ? 16 * 8 * 2 : 16 * 8;
    EXPECT_EQ(r.n_cells, uint64_t(n_cells));
    EXPECT_NEAR(r.wall_per_time, n_cells / (options.ranks * 1e6 * r.dt), roundoff(1e-12) * r.wall_per_time);
    EXPECT_NEAR(r.flow_through_time, 2.0, roundoff(1e-12));  // Lx / u
}

TEST(CheckCost, StepsMeasureTheThroughput) {
    CheckOptions options;
    options.steps = 2;
    const CheckReport r = check(box(), options);
    EXPECT_TRUE(r.throughput_measured);
    EXPECT_GT(r.throughput, 0.0);
    EXPECT_GT(r.wall_per_time, 0.0);
}

TEST(CheckMemory, MeasuresAllocationsAndScalesThemToTheTargetRanks) {
    if (comm::size() > 1) GTEST_SKIP() << "serial estimate";
    CheckOptions options;
    options.ranks = 4;
    const CheckReport r = check(box(), options);
    ASSERT_TRUE(r.ok()) << (first_error(r));
    // Conservatives alone are N_CONSERVATIVE reals per cell
    EXPECT_GT(r.bytes, r.n_cells * N_CONSERVATIVE * sizeof(rtype));
    EXPECT_GE(r.bytes_peak, r.bytes);
    EXPECT_EQ(r.target_ranks, 4);
    EXPECT_EQ(r.target_owned, r.n_cells / 4);
    EXPECT_GT(r.target_halo, 0u);
    const double per_cell = double(r.bytes) / double(r.n_cells);
    EXPECT_NEAR(double(r.target_bytes), per_cell * double(r.target_owned + r.target_halo), 1.0);
}

namespace {

/** @brief The box at Nx x Ny cells (x 2 in 3D), with TENO3 or first order. */
std::string sized_box(const int nx, const int ny, const bool teno = true) {
    std::string input =
        box("1.0", "t_stop = 0.5\ncfl = 0.8\n",
            teno ? "[numerics.face_reconstruction]\ntype = \"TENO\"\norder = 3\n" : "");
    const std::string mesh = "Nx = 16\nNy = 8\n";
    input.replace(input.find(mesh), mesh.size(), "Nx = " + std::to_string(nx) + "\nNy = " + std::to_string(ny) + "\n");
    return input;
}

uint64_t sum(const memory::Bytes & b) { return std::accumulate(b.begin(), b.end(), uint64_t(0)); }

} // namespace

TEST(CheckMemory, CategoriesSumToTheTotalsAndNameEveryAllocation) {
    if (comm::size() > 1) GTEST_SKIP() << "one rank's allocations";
    const CheckReport r = check(sized_box(16, 8));
    ASSERT_TRUE(r.ok()) << (first_error(r));
    EXPECT_EQ(sum(r.setup_bytes), r.bytes_peak);
    EXPECT_EQ(sum(r.steady_bytes), r.bytes);
    EXPECT_EQ(r.setup_bytes[size_t(memory::Category::OTHER)], 0u);
    EXPECT_GT(r.steady_bytes[size_t(memory::Category::TABLES)], 0u);
    // The setup's workspace is freed by the time stepping
    EXPECT_GT(r.setup_bytes[size_t(memory::Category::SETUP)], 0u);
    EXPECT_EQ(r.steady_bytes[size_t(memory::Category::SETUP)], 0u);
}

TEST(CheckMemory, PeakPredictedFromACoarseMeshMatchesTheMeasuredOne) {
    if (comm::size() > 1) GTEST_SKIP() << "one rank's allocations";
    // From 16,384 cells, the size of the initial solution's chunks, to four times as many along x (the same
    // share of cells near the walls across y and z, whose TENO stencils are smaller)
    const int ny = N_DIM == 2 ? 128 : 64;
    const CheckReport coarse = check(sized_box(128, ny)), fine = check(sized_box(512, ny));
    ASSERT_TRUE(coarse.ok() && fine.ok());
    const double predicted = predicted_peak(coarse, double(fine.local_cells));
    EXPECT_NEAR(predicted, double(fine.bytes_peak), 0.05 * double(fine.bytes_peak));
}

TEST(CheckSizing, FewestRanksTheMeshFitsOn) {
    if (comm::size() > 1) GTEST_SKIP() << "partitions the whole mesh";
    const std::string input = sized_box(256, 64, false);
    CheckOptions options;
    options.memory = 1ull << 40;
    options.memory_name = "1TiB";
    const CheckReport whole = check(input, options);
    ASSERT_TRUE(whole.ok());
    EXPECT_EQ(whole.min_ranks, 1);
    EXPECT_GT(whole.max_local_cells, whole.n_cells);
    // Room for a third of the cells: three parts with their halos do not fit, four do
    const double third = predicted_peak(whole, double(whole.n_cells) / 3.0);
    options.memory = static_cast<uint64_t>(third / (1.0 - SIZING_MARGIN));
    const CheckReport r = check(input, options);
    ASSERT_TRUE(r.ok());
    EXPECT_EQ(r.min_ranks, 4);
    EXPECT_GT(r.min_halo, 0u);
    EXPECT_LE(double(r.min_peak), third);
    EXPECT_NEAR(double(r.min_peak), predicted_peak(r, double(r.min_owned + r.min_halo)), 1.0);
    EXPECT_NEAR(double(r.max_local_cells), double(whole.n_cells) / 3.0, 1.0);
}

TEST(CheckSizing, MemoryIsASizeWithItsUnitOrAGpu) {
    CheckOptions options;
    EXPECT_EQ(set_memory_target("40GB", options), "");
    EXPECT_EQ(options.memory, 40'000'000'000u);
    EXPECT_EQ(set_memory_target("512MiB", options), "");
    EXPECT_EQ(options.memory, 512u << 20);
    EXPECT_EQ(set_memory_target("A100-80", options), "");
    EXPECT_EQ(options.memory_name, "A100-80GB");
    EXPECT_TRUE(options.memory_gpu);
    EXPECT_NE(set_memory_target("40", options).find("a100-40"), std::string::npos);
    EXPECT_NE(set_memory_target("-3GB", options), "");
    EXPECT_NE(set_memory_target("a100", options), "");
}

TEST(CheckOutput, CheckCreatesNoFilesAndWritesTheQualityFieldsOnRequest) {
    const std::filesystem::path dir = scratch_dir() / "mallard_check_files";
    if (comm::is_root()) std::filesystem::remove_all(dir);
    comm::barrier();
    CheckOptions options;
    options.quality_file = (dir / "quality" / "mesh").string();
    const CheckReport r = check(box("1.0", "t_stop = 0.5\ncfl = 0.8\n",
                                    "[[write_data]]\nprefix = \"" + (dir / "out" / "flow").string() +
                                        "\"\nformat = \"vtu\"\ninterval = 5\nvariables = [\"RHO\"]\n[[forces]]\n"
                                        "zone = \"bottom\"\nfile = \"" + (dir / "forces.csv").string() +
                                        "\"\n[integrals]\nfile = \"" + (dir / "integrals.csv").string() + "\"\n"),
                                options);
    ASSERT_TRUE(r.ok()) << (first_error(r));
    EXPECT_FALSE(std::filesystem::exists(dir / "out"));
    EXPECT_FALSE(std::filesystem::exists(dir / "forces.csv"));
    EXPECT_FALSE(std::filesystem::exists(dir / "integrals.csv"));
    EXPECT_TRUE(output_files_enabled());
    // The default snapshot format: VTKHDF, or VTU pieces without (parallel) HDF5
#ifdef Mallard_HAS_HDF5
    if (comm::size() == 1 || have_parallel_hdf5()) {
        comm::barrier();
        if (!comm::is_root()) return;
        const auto datasets = h5_all_datasets((dir / "quality" / "mesh_000000.vtkhdf").string());
        for (const char * field : MeshQuality::FIELDS) {
            EXPECT_TRUE(datasets.count(std::string("VTKHDF/CellData/") + field)) << field;
        }
        return;
    }
#endif
    std::ostringstream piece;  // each rank's own piece
    piece << "mesh_000000" << (comm::size() > 1 ? "_p" + std::string(3, '0') + std::to_string(comm::rank()) : "")
          << ".vtu";
    std::ifstream vtu(dir / "quality" / piece.str());
    ASSERT_TRUE(vtu.good());
    const std::string text((std::istreambuf_iterator<char>(vtu)), std::istreambuf_iterator<char>());
    for (const char * field : MeshQuality::FIELDS) EXPECT_NE(text.find(field), std::string::npos) << field;
}

TEST(CheckSettings, TflesThickeningAboveFiveIsAWarning) {
    // Delta = 1 cm (4 x 4 cells of a 4 cm box), so F = n_res Delta / delta_L
    auto input = [](double delta_L) {
        std::ostringstream s;
        s << "[run]\nn_steps = 1\ndt = 1e-7\n[mesh]\ntype = \"cartesian\"\nNx = 4\nNy = 4\nLx = 0.04\nLy = 0.04\n"
          << (N_DIM == 3 ? "Nz = 4\nLz = 0.04\n[[periodic]]\nzones = [\"left\", \"right\"]\n[[periodic]]\nzones = [\"bottom\", \"top\"]\n[[periodic]]\nzones = [\"back\", \"front\"]\n" : "[[periodic]]\nzones = [\"left\", \"right\"]\n[[periodic]]\nzones = [\"bottom\", \"top\"]\n")
          << "[initialize]\ntype = \"analytical\"\np = \"101325.0\"\nT = \"300.0\"\nu = " << U_ZERO << "\n"
          << "X = { H2 = 2.0, O2 = 1.0, N2 = 3.76 }\n"
          << "[numerics]\nriemann_solver = \"HLLC\"\n"
          << "[physics]\ntype = \"navier_stokes\"\ngas = \"mixture\"\nmechanism = \"" + source_dir() +
"/mechanisms/h2o2.yaml\"\n[chemistry]\n[les]\nmodel = \""
          << (N_DIM == 3 ? "sigma" : "wale") << "\"\n[les.combustion]\nmodel = \"tfles\"\ndelta_L = " << delta_L
          << "\ns_L = 2.0\nT_unburnt = 300.0\nT_burnt = 2400.0\nn_res = 5.0\n";
        return s.str();
    };
    const double delta = 0.01;
    const CheckReport fine = check(input(5.0 * delta / 4.0));    // F = 4
    ASSERT_TRUE(fine.ok()) << (first_error(fine));
    EXPECT_FALSE(mentions(fine.warnings(), "TFLES"));
    const CheckReport coarse = check(input(5.0 * delta / 10.0));  // F = 10
    EXPECT_TRUE(mentions(coarse.warnings(), "F = 10"));
    EXPECT_NEAR(fine.cells_per_delta_L_min, 1.25, roundoff(1e-12));
    EXPECT_NEAR(coarse.cells_per_delta_L_max, 0.5, roundoff(1e-12));
}

namespace {

/** @brief A periodic box of n^d cells and side 2 pi with viscosity mu, rho = 1; velocity components as expressions. */
std::string periodic_box(const int n, const double mu, const std::vector<std::string> & u) {
    std::ostringstream s;
    s << "[run]\nt_stop = 1.0\ncfl = 0.5\n[mesh]\ntype = \"cartesian\"\nNx = " << n << "\nNy = " << n
      << "\nLx = 6.283185307179586\nLy = 6.283185307179586\n"
      << (N_DIM == 3 ? "Nz = " + std::to_string(n) + "\nLz = 6.283185307179586\n" : "")
      << "[[periodic]]\nzones = [\"left\", \"right\"]\n[[periodic]]\nzones = [\"bottom\", \"top\"]\n"
      << (N_DIM == 3 ? "[[periodic]]\nzones = [\"back\", \"front\"]\n" : "")
      << "[initialize]\ntype = \"analytical\"\nrho = \"1.0\"\np = \"100.0\"\nu = [";
    for (size_t i = 0; i < N_DIM; i++) s << (i ? ", " : "") << "\"" << u[i] << "\"";
    s << "]\n[numerics]\nriemann_solver = \"HLLC\"\n[physics]\ntype = \"navier_stokes\"\ngamma = 1.4\n"
      << "p_ref = 100.0\nT_ref = 1.0\nrho_ref = 1.0\nmu = " << mu << "\nPr = 0.72\n";
    return s.str();
}

} // namespace

TEST(CheckResolution, TaylorGreenDissipationGivesTheKolmogorovLength) {
    // epsilon = 2 nu |k|^2 E: |k|^2 = d, E = 1/4 (2D) or 1/8 (3D); u'^2 = 2 E / d
    const double nu = 0.01;
    const std::vector<std::string> u = N_DIM == 3
        ? std::vector<std::string>{"sin(x) * cos(y) * cos(z)", "-cos(x) * sin(y) * cos(z)", "0.0"}
        : std::vector<std::string>{"sin(x) * cos(y)", "-cos(x) * sin(y)"};
    const double E = N_DIM == 3 ? 0.125 : 0.25, epsilon = 2.0 * nu * N_DIM * E;
    const int n = N_DIM == 3 ? 48 : 64;
    const CheckReport r = check(periodic_box(n, nu, u));
    ASSERT_TRUE(r.ok()) << (first_error(r));
    EXPECT_NEAR(r.epsilon, epsilon, 0.05 * epsilon);
    EXPECT_NEAR(r.eta, std::pow(nu * nu * nu / r.epsilon, 0.25), roundoff(1e-11) * r.eta);
    EXPECT_NEAR(r.taylor, std::sqrt(15.0 * nu * (2.0 * E / N_DIM) / r.epsilon), 0.01 * r.taylor);
    EXPECT_NEAR(r.delta_over_eta, 2.0 * std::numbers::pi / n / r.eta, roundoff(1e-9) * r.delta_over_eta);
    EXPECT_FALSE(mentions(r.warnings(), "DNS"));
    // A hundredth of the viscosity dissipates a hundredth as much: eta ten times smaller
    const CheckReport coarse = check(periodic_box(n, nu / 100.0, u));
    EXPECT_TRUE(mentions(coarse.warnings(), "DNS: cells of"));
}

TEST(CheckResolution, AFlowWithoutGradientsOrForcingHasNoKolmogorovLength) {
    const CheckReport r = check(periodic_box(8, 0.01, std::vector<std::string>(N_DIM, "1.0")));
    ASSERT_TRUE(r.ok()) << (first_error(r));
    EXPECT_EQ(r.eta, 0.0);
    bool said = false;
    for (const CheckEstimate & e : r.resolution) {
        said |= e.id == "eta" && std::isnan(e.value) && e.basis.rfind("not estimated", 0) == 0;
    }
    EXPECT_TRUE(said);
}

TEST(CheckResolution, ChannelWallUnitsFromTheForcingsBulkReynoldsNumber) {
    // Re_b = U_b 2h / nu = 5600, for which Dean's correlation gives Re_tau = u_tau h / nu = 182 (Re_tau 180 cases)
    const double nu = 1.0 / 2800.0;
    std::ostringstream s;
    s << "[run]\nt_stop = 1.0\ncfl = 0.9\n[mesh]\ntype = \"cartesian\"\nNx = 8\nNy = 16\nLx = 4.0\nLy = 2.0\n"
      << (N_DIM == 3 ? "Nz = 4\nLz = 2.0\n[[periodic]]\nzones = [\"back\", \"front\"]\n" : "")
      << "[[periodic]]\nzones = [\"left\", \"right\"]\n"
      << "[initialize]\ntype = \"analytical\"\nrho = \"1.0\"\np = \"17.857142857142858\"\n"
      << "u = [\"1.5 * (1 - (y - 1) * (y - 1))\", \"0.0\"" << (N_DIM == 3 ? ", \"0.0\"" : "") << "]\n"
      << "[[boundaries]]\nname = \"bottom\"\ntype = \"wall_isothermal\"\nT = 1.0\n"
      << "[[boundaries]]\nname = \"top\"\ntype = \"wall_isothermal\"\nT = 1.0\n"
      << "[source]\nmass_flow = [1.0, 0.0" << (N_DIM == 3 ? ", 0.0" : "") << "]\n"
      << "[numerics]\nriemann_solver = \"HLLC\"\n[physics]\ntype = \"navier_stokes\"\ngamma = 1.4\n"
      << "p_ref = 17.857142857142858\nT_ref = 1.0\nrho_ref = 1.0\nmu = " << nu << "\nPr = 0.72\n";
    const CheckReport r = check(s.str());
    ASSERT_TRUE(r.ok()) << (first_error(r));
    EXPECT_NEAR(r.u_tau / nu, 180.0, 0.03 * 180.0);
    const double y1 = 1.0 / 16.0;  // the first cell centers: half of 2 / 16
    EXPECT_NEAR(r.y_plus_max, y1 * r.u_tau / nu, 1e-5 * r.y_plus_max);
    EXPECT_NEAR(r.y_plus_mean, r.y_plus_max, 1e-5 * r.y_plus_max);
    EXPECT_NEAR(r.x_plus_max, 0.5 * r.u_tau / nu, 1e-5 * r.x_plus_max);
    if (N_DIM == 3) {
        EXPECT_NEAR(r.z_plus_max, 0.5 * r.u_tau / nu, 1e-5 * r.z_plus_max);
    }
    EXPECT_NEAR(r.epsilon, r.u_tau * r.u_tau, 1e-5 * r.epsilon);  // u_tau^2 U_b / h with U_b = h = 1
    EXPECT_TRUE(mentions(r.warnings(), "first cells at y+ up to"));
    EXPECT_TRUE(mentions(r.warnings(), "wall cells up to x+"));
}

TEST(MeshQualityMetrics, UniformQuadsAreIdealAndElongatedOnesHaveTheirAspectRatio) {
    if (N_DIM != 2) GTEST_SKIP() << "2D mesh";
    const auto mesh = make_mesh("cartesian", 10, 2, 1.0, 1.0);  // cells 0.1 x 0.5
    const auto s = summarize(*mesh, mesh_quality(*mesh), 3);
    EXPECT_NEAR(s[0].max, 0.0, roundoff(1e-10));
    EXPECT_NEAR(s[1].max, 0.0, roundoff(1e-12));
    EXPECT_NEAR(s[2].max, 5.0, roundoff(1e-12));
    EXPECT_NEAR(s[2].mean, 5.0, roundoff(1e-12));
    EXPECT_NEAR(s[3].max, 1.0, roundoff(1e-12));
    EXPECT_EQ(s[2].worst.size(), 3u);
}

TEST(MeshQualityMetrics, SkewedPairOfCellsAgainstHandComputedValues) {
    if (N_DIM != 2) GTEST_SKIP() << "2D mesh";
    // Unit square (centroid (1/2, 1/2)) beside the triangle (1,0), (2,0), (1,1)
    // (centroid (4/3, 1/3)): the centroid line crosses x = 1 at y = 0.4, 0.1
    // below the shared face's midpoint, at atan(1/5) from its normal
    Mesh mesh;
    mesh.init_from_connectivity({{0.0, 0.0}, {1.0, 0.0}, {1.0, 1.0}, {0.0, 1.0}, {2.0, 0.0}},
                                {{0, 1, 2, 3}, {1, 4, 2}}, {});
    const MeshQuality q = mesh_quality(mesh);
    const uint32_t tri = mesh.h_n_nodes_of_cell(0) == 3 ? 0 : 1, square = 1 - tri;
    const double d = std::sqrt(26.0) / 6.0;
    EXPECT_NEAR(q.cells[0][square], std::atan(0.2) * 180.0 / std::numbers::pi, roundoff(1e-10));
    EXPECT_NEAR(q.cells[1][square], 0.1 / d, roundoff(1e-12));
    EXPECT_NEAR(q.cells[1][tri], 0.1 / d, roundoff(1e-12));
    EXPECT_NEAR(q.cells[3][tri], 2.0, roundoff(1e-12));
    EXPECT_NEAR(q.cells[2][square], 1.0, roundoff(1e-12));
    // Triangle: centroid to edge midpoints sqrt(5)/6 (legs) and sqrt(2)/6 (hypotenuse)
    EXPECT_NEAR(q.cells[2][tri], std::sqrt(5.0) / std::sqrt(2.0), roundoff(1e-12));
    // The triangle's bottom leg: its centroid is 1/6 off the leg's midpoint, 1/3 above it
    EXPECT_NEAR(q.cells[0][tri], std::atan(0.5) * 180.0 / std::numbers::pi, roundoff(1e-10));
}

TEST(MeshQualityMetrics, StretchedMeshVolumeRatioIsTheLargestSpacingRatio) {
    if (N_DIM != 2) GTEST_SKIP() << "2D mesh";
    const auto input = parse_input_string(
        "[mesh]\ntype = \"cartesian\"\nNx = 4\nNy = 20\nLx = 1.0\nLy = 1.0\nstretching = [0.0, 2.0]\n");
    Mesh mesh;
    mesh.init(input);
    double expected = 1.0;
    for (uint32_t j = 0; j + 2 <= 20; j++) {
        const double a = double(stretched_coordinate(j + 1, 20, 1.0, 2.0) - stretched_coordinate(j, 20, 1.0, 2.0));
        const double b =
            double(stretched_coordinate(j + 2, 20, 1.0, 2.0) - stretched_coordinate(j + 1, 20, 1.0, 2.0));
        expected = std::max({expected, a / b, b / a});
    }
    const auto s = summarize(mesh, mesh_quality(mesh), 1);
    EXPECT_NEAR(s[3].max, expected, roundoff(1e-10));
}

TEST(LargestPart, HaloLayersOfNodeNeighbors) {
    // 8 x 8 quads cut at x = 1/2: each half has 32 cells, a first halo layer of
    // the 8 cells across the cut (node neighbors) and a second of 8 more
    if (N_DIM != 2) GTEST_SKIP() << "2D block";
    const MeshBlock block = cartesian_2d_block(8, 8, 1.0, 1.0, MeshType::CARTESIAN, 0, 1);
    std::vector<int> owner(block.n_cells());
    for (uint64_t c = 0; c < block.n_cells(); c++) {
        double x = 0.0;
        for (uint64_t k = block.cell_offsets[c]; k < block.cell_offsets[c + 1]; k++) {
            x += block.node_coords[block.cell_nodes[k] - block.first_node][0];
        }
        owner[c] = x / double(block.cell_offsets[c + 1] - block.cell_offsets[c]) < 0.5 ? 0 : 1;
    }
    EXPECT_EQ(largest_part(block, owner, 2, 1).owned, 32u);
    EXPECT_EQ(largest_part(block, owner, 2, 1).halo, 8u);
    EXPECT_EQ(largest_part(block, owner, 2, 2).halo, 16u);
}

TEST(MemoryTracker, CountsLiveAllocationsAndThePeak) {
    memory::Tracker tracker;
    const bool device = !Kokkos::SpaceAccessibility<Kokkos::HostSpace,
                                                    Kokkos::DefaultExecutionSpace::memory_space>::accessible;
    auto usage = [&] { return device ? tracker.device() : tracker.host(); };
    {
        Kokkos::View<double *> a("a", 1000);
        EXPECT_GE(usage().current, 8000u);
        {
            Kokkos::View<double *> b("b", 3000);
            EXPECT_GE(usage().current, 32000u);
        }
        EXPECT_LT(usage().current, 32000u);
    }
    EXPECT_EQ(usage().current, 0u);
    EXPECT_GE(usage().peak, 32000u);
    EXPECT_EQ(tracker.unclassified(), (std::vector<std::string>{"a", "b"}));
    Kokkos::View<double *> c("conservatives", 500);
    tracker.reset_peak();
    EXPECT_EQ(usage().peak, usage().current);
    EXPECT_GE(usage().at_peak[size_t(memory::Category::STATE)], 4000u);
}

TEST(SetupMemory, NothingOnlyTheSetupReadsSurvivesIt) {
    const bool device = !Kokkos::SpaceAccessibility<Kokkos::HostSpace,
                                                    Kokkos::DefaultExecutionSpace::memory_space>::accessible;
    const std::string ball = N_DIM == 2 ? "{type = \"disk\", center = [1.0, 0.5], radius = 0.3}"
                                        : "{type = \"sphere\", center = [1.0, 0.5, 0.125], radius = 0.3}";
    const std::string source = "[[sources]]\nrhoE = \"1.0 + x\"\nregion = " + ball + "\n";
    // Setup scratch of the initial state, the TENO tables and the mesh: what the run would hold from setup
    const std::vector<std::string> setup = {"init_", "teno_setup_", "teno_batch_", "teno_boxes"};
    const std::vector<std::string> mesh_device = {"node_coords", "nodes_of_cell", "offsets_nodes_of_cell",
                                                  "nodes_of_face", "offsets_nodes_of_face", "zone_"};
    const std::vector<std::string> mesh_host = {"cells_of_cell_mirror", "offsets_cells_of_cell_mirror",
                                                "cells_of_cell_shift_mirror"};
    // Host builds share the mesh's device views with the host copies that output reads
    auto survivors = [&](const std::string & reconstruction, const bool mesh) {
        memory::Tracker tracker;
        Solver solver;
        solver.init(parse_input_string(box("1.0 + 0.1 * x", "n_steps = 1\ncfl = 0.5\n", source + reconstruction)));
        std::vector<std::string> held;
        auto find = [&](const bool space, const std::vector<std::string> & prefixes) {
            for (const auto & [label, bytes] : tracker.live(space)) {
                for (const std::string & prefix : prefixes) {
                    if (bytes > 0 && label.rfind(prefix, 0) == 0) held.push_back(label);
                }
            }
        };
        find(true, setup);
        find(false, setup);
        if (mesh && device) {
            find(true, mesh_device);
            find(false, mesh_host);
        }
        return held;
    };
    EXPECT_EQ(survivors("[numerics.face_reconstruction]\ntype = \"TENO\"\norder = 5\n", false),
              std::vector<std::string>());
    EXPECT_EQ(survivors("[numerics.face_reconstruction]\ntype = \"MUSCL\"\n", true), std::vector<std::string>());
}

TEST(CheckRegions, ListsEveryRegionWithItsDimensionMeasureAndCells) {
    // The box is 2 x 1 (x 0.25): its regions, a probe's ball and the ball's surface
    const std::string ball = N_DIM == 2 ? "{type = \"disk\", center = [0.5, 0.5], radius = 0.3"
                                        : "{type = \"sphere\", center = [0.5, 0.5, 0.125], radius = 0.1";
    const std::string extra = "[regions]\nball = " + ball + "}\n[[probes]]\nname = \"in\"\nvariables = [\"P\"]\n"
                              "region = \"ball\"\n[[probes]]\nname = \"on\"\nvariables = [\"P\"]\nregion = " + ball +
                              ", filled = false}\n";
    auto report = [&](const bool distributed) {
        Solver solver;
        solver.set_distributed(distributed);
        return solver.check(parse_input_string(box("1.0", "t_stop = 0.5\ncfl = 0.8\n", extra)), CheckOptions{});
    };
    const CheckReport r = report(true);
    ASSERT_TRUE(r.ok()) << (first_error(r));
    std::map<std::string, CheckRegion> regions;
    for (const CheckRegion & g : r.regions) regions[g.name] = g;
    const CheckRegion & domain = regions.at("domain");
    EXPECT_TRUE(domain.of_mesh);
    EXPECT_EQ(domain.dimension, N_DIM);
    EXPECT_DOUBLE_EQ(domain.measure, N_DIM == 2 ? 2.0 : 0.5);
    EXPECT_EQ(domain.cells, N_DIM == 2 ? 128u : 256u);
    const CheckRegion & left = regions.at("left");
    EXPECT_EQ(left.dimension, N_DIM - 1);
    EXPECT_DOUBLE_EQ(left.measure, N_DIM == 2 ? 1.0 : 0.25);
    EXPECT_EQ(left.cells, N_DIM == 2 ? 8u : 16u);
    EXPECT_EQ(regions.at("ball").dimension, N_DIM);
    EXPECT_FALSE(regions.at("ball").of_mesh);
    const CheckRegion & on = regions.at("probes[name = \"on\"].region");
    EXPECT_EQ(on.dimension, N_DIM - 1);
    const double sphere = N_DIM == 2 ? 2.0 * std::numbers::pi * 0.3 : 4.0 * std::numbers::pi * 0.01;
    EXPECT_NEAR(on.measure, sphere, 1e-6 * sphere);
    EXPECT_TRUE(on.exact);
    // The same on one rank as on several
    const CheckReport serial = report(false);
    ASSERT_EQ(serial.regions.size(), r.regions.size());
    for (size_t i = 0; i < r.regions.size(); i++) {
        EXPECT_EQ(serial.regions[i].name, r.regions[i].name);
        EXPECT_EQ(serial.regions[i].cells, r.regions[i].cells);
        EXPECT_DOUBLE_EQ(serial.regions[i].measure, r.regions[i].measure);
    }
}

TEST(CheckRegions, ARegionOutsideTheDomainIsAnErrorWithItsExtent) {
    const std::string far = N_DIM == 2 ? "{type = \"rectangle\", center = [5, 5], size = [1, 2]}"
                                       : "{type = \"box\", center = [5, 5, 5], size = [1, 2, 1]}";
    const CheckReport r = check(box("1.0", "t_stop = 0.5\ncfl = 0.8\n",
                                    "[[sponges]]\nstrength = 1.0\nregion = " + far + "\nu = " + U_ZERO +
                                        "\np = 1.0\nT = 1.0\n"));
    EXPECT_TRUE(mentions(r.errors(), "region sponges[0].region lies outside the domain: it spans (4.5, 4"))
        << (first_error(r));
}

#ifdef Mallard_HAS_HDF5
TEST(CheckRegions, TheCaseViewHoldsEveryRegionAndZone) {
    const std::filesystem::path dir = scratch_dir() / "mallard_check_view";
    if (comm::is_root()) std::filesystem::remove_all(dir);
    comm::barrier();
    CheckOptions options;
    options.view_dir = dir.string();
    const std::string surface = N_DIM == 2 ? "{type = \"circle\", center = [1, 0.5], radius = 0.25}"
                                           : "{type = \"disk\", center = [1, 0.5, 0.125], normal = [1, 0, 0], radius = 0.1}";
    const CheckReport r = check(box("1.0", "t_stop = 0.5\ncfl = 0.8\n",
                                    "[[probes]]\nname = \"s\"\nvariables = [\"P\"]\nregion = " + surface + "\n"),
                                options);
    ASSERT_TRUE(r.ok()) << (first_error(r));
    EXPECT_EQ(r.view_file, (dir / "case.vtkhdf").string());
    comm::barrier();
    if (!comm::is_root()) return;
    const auto datasets = h5_all_datasets(r.view_file);
    for (const std::string block : {"left", "right", "bottom", "top", "probes[name = \"s\"].region"}) {
        EXPECT_TRUE(datasets.count("VTKHDF/" + block + "/Connectivity")) << block;
        EXPECT_TRUE(datasets.count("VTKHDF/" + block + "/CellData/normal")) << block;
    }
}
#endif

namespace {

/**
 * @brief The check of a 2D box with an unused key, a CFL above SSPRK3's limit, a sponge outside the domain and
 *        a probe on a circle, with the numbers that depend on the build or the machine (memory) set by hand.
 */
CheckReport report_case() {
    const std::string far = N_DIM == 2 ? "{type = \"rectangle\", center = [5, 5], size = [1, 2]}"
                                       : "{type = \"box\", center = [5, 5, 5], size = [1, 2, 1]}";
    const std::string ring = N_DIM == 2 ? "{type = \"circle\", center = [1, 0.5], radius = 0.25}"
                                        : "{type = \"disk\", center = [1, 0.5, 0.125], normal = [1, 0, 0], radius = 0.1}";
    const std::string extra = "[[sponges]]\nstrength = 1.0\nregion = " + far + "\nu = " + U_ZERO +
                              "\np = 1.0\nT = 1.0\n[[probes]]\nname = \"ring\"\nvariables = [\"P\"]\nregion = " + ring +
                              "\n";
    std::string input = box("1.0", "t_stop = 0.5\ncfl = 1.5\n", extra);
    input.replace(input.find("riemann_solver"), 14, "riemann_solvr");
    CheckOptions options;
    options.input_file = "input.toml";
    options.throughput = 1e6;
    CheckReport r = check(input, options);
    r.device_memory = false;
    r.device_capacity = 0;
    r.outside_kokkos = 0;
    r.sizing_name = "A100-80GB";
    r.sizing_gpu = true;
    r.sizing_memory = 80'000'000'000;
    r.sizing_overhead = 1'600'000'000;
    r.overhead_measured = false;
    r.bytes_per_cell = 547.0;
    r.fixed_bytes = 58'100;
    r.sizing_setup = true;
    r.max_local_cells = 128'683'729;
    r.min_ranks = 1;
    r.min_owned = 128;
    r.min_halo = 0;
    r.min_peak = 1'600'128'100;
    r.min_partitioned = true;
    r.halo_ranks = 0;
    r.setup_bytes.fill(0);
    r.steady_bytes.fill(0);
    r.setup_bytes[size_t(memory::Category::STATE)] = r.steady_bytes[size_t(memory::Category::STATE)] = 27'100;
    r.setup_bytes[size_t(memory::Category::MESH)] = r.steady_bytes[size_t(memory::Category::MESH)] = 42'900;
    r.setup_bytes[size_t(memory::Category::SETUP)] = 58'100;
    r.bytes = r.target_bytes = 70'000;
    r.bytes_peak = r.target_bytes_peak = 128'100;
    return r;
}

} // namespace

// Guards every section, column, unit and status of the text report. After a deliberate change of the report,
// MALLARD_UPDATE_GOLDEN=1 rewrites the expected text (then review its diff).
TEST(CheckReportText, MatchesTheGoldenReport) {
    if (N_DIM != 2 || sizeof(rtype) != 8 || comm::size() > 1) {
        GTEST_SKIP() << "the golden report is of a serial 2D double-precision check";
    }
    const std::string text = check_text(report_case(), false);
    const std::filesystem::path golden =
        std::filesystem::path(source_dir()) / "test" / "data" / "check" / "report_2d.txt";
    if (const char * update = std::getenv("MALLARD_UPDATE_GOLDEN"); update && std::string(update) == "1") {
        std::ofstream(golden) << text;
    }
    std::ifstream file(golden);
    ASSERT_TRUE(file) << golden;
    const std::string expected((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
    EXPECT_EQ(text, expected) << "the report differs from " << golden << "; MALLARD_UPDATE_GOLDEN=1 updates it";
}

TEST(ReportTable, KeepsEveryValueOfAUnitsColumnTableBare) {
    using Kind = report::Column::Kind;
    const std::vector<report::Column> header = {"quantity", {"value", Kind::VALUE}, {"units", Kind::UNITS}, "basis"};
    auto table = [&](const std::string & value, const std::string & units) {
        report::Text text(false);
        text.table(header, {{report::Status::NONE, {"wall time", value, units, "on 4 GPUs"}}});
    };
    for (const char * value : {"21.0", "1,024", "-1.5e-06", "-", "none"}) EXPECT_NO_THROW(table(value, "s")) << value;
    for (const char * value : {"21.0 s", "5%", "1.60 GB", "2h 05m", "(1, 2) m"}) {
        EXPECT_THROW(table(value, "s"), std::logic_error) << value;
    }
    EXPECT_THROW(table("21.0", ""), std::logic_error);
    EXPECT_NO_THROW(table("-", ""));
    report::Text text(false);
    EXPECT_THROW(text.table({"quantity", "value", "units"}, {}), std::logic_error);
    EXPECT_THROW(text.quantities({{report::Status::NONE, {"memory per GPU", "80.0 GB", "", "", "A100-80GB"}}}),
                 std::logic_error);
}

TEST(CheckReportText, ColorChangesOnlyTheStyles) {
    const CheckReport r = report_case();
    std::string colored = check_text(r, true);
    EXPECT_NE(colored.find("\033["), std::string::npos);
    for (size_t at = colored.find("\033["); at != std::string::npos; at = colored.find("\033[")) {
        colored.erase(at, colored.find('m', at) + 1 - at);
    }
    EXPECT_EQ(colored, check_text(r, false));
}

TEST(CheckReportJson, ReadsBackTheReportsNumbersAndDiagnostics) {
    const CheckReport r = report_case();
    const json::Value v = json::parse(check_json(r));
    EXPECT_EQ(v["version"].number, 1.0);
    EXPECT_EQ(v["status"].string, "error");
    EXPECT_EQ(v["errors"].number, double(r.errors().size()));
    EXPECT_EQ(v["warnings"].number, double(r.warnings().size()));
    ASSERT_EQ(v["diagnostics"].array.size(), r.diagnostics.size());
    for (size_t i = 0; i < r.diagnostics.size(); i++) {
        const json::Value & d = v["diagnostics"].array[i];
        EXPECT_EQ(d["code"].string, r.diagnostics[i].code);
        EXPECT_EQ(d["message"].string, r.diagnostics[i].message);
        EXPECT_EQ(d["severity"].string, report::status_name(r.diagnostics[i].severity));
        EXPECT_EQ(d["docs"].string, r.diagnostics[i].docs());
    }
    ASSERT_EQ(v["unused_keys"].array.size(), 1u);
    EXPECT_EQ(v["unused_keys"].array[0]["suggestion"].string, "numerics.riemann_solver");
    EXPECT_EQ(v["time_step"]["dt"].number, r.dt);
    EXPECT_EQ(v["time_step"]["cfl"].number, r.cfl);
    EXPECT_EQ(v["time_step"]["cfl_limit"].number, 1.2);
    EXPECT_EQ(v["time_step"]["limiting_cell"].number, double(r.limiting_cell));
    ASSERT_EQ(v["regions"].array.size(), r.regions.size());
    for (size_t i = 0; i < r.regions.size(); i++) {
        EXPECT_EQ(v["regions"].array[i]["name"].string, r.regions[i].name);
        EXPECT_EQ(v["regions"].array[i]["measure"].number, r.regions[i].measure);
        EXPECT_EQ(v["regions"].array[i]["cells"].number, double(r.regions[i].cells));
    }
    ASSERT_FALSE(r.resolution.empty());
    for (const CheckEstimate & e : r.resolution) {
        const json::Value & j = v["resolution"][e.id];
        if (std::isnan(e.value)) {
            EXPECT_TRUE(j["value"].is_null()) << e.id;
        } else {
            EXPECT_EQ(j["value"].number, e.value) << e.id;
        }
        EXPECT_EQ(j["units"].string, e.units) << e.id;
    }
    EXPECT_EQ(v["memory"]["bytes_peak"].number, double(r.bytes_peak));
    EXPECT_EQ(v["memory"]["at_setup_peak"]["setup"].number, 58'100.0);
    EXPECT_EQ(v["cost"]["wall_per_time"].number, r.wall_per_time);
    EXPECT_EQ(v["mesh_quality"]["ASPECT_RATIO"]["max"].number, r.quality[2].max);
}

TEST(CheckReportJson, ASetupThatStopsStillGivesEveryTopLevelKey) {
    std::string input = box();
    input.replace(input.find("Nx = 16"), 7, "Nx = 16.0");
    const json::Value v = json::parse(check_json(check(input)));
    EXPECT_EQ(v["status"].string, "error");
    EXPECT_EQ(v["diagnostics"].array.at(0)["code"].string, check_code::INPUT);
    for (const char * key : {"time_step", "memory", "sizing", "cost", "mesh_quality"}) {
        EXPECT_TRUE(v[key].is_null()) << key;
    }
}

TEST(CheckDiagnostics, EveryCodeIsExplainedInTheDocs) {
    const std::filesystem::path page = std::filesystem::path(source_dir()) / "docs" / "guide" / "input_files.md";
    if (!std::filesystem::exists(page)) GTEST_SKIP() << "no docs in an install tree";
    std::ifstream file(page);
    const std::string docs((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
    for (const char * code : check_code::ALL) {
        const std::string link = CheckDiagnostic{report::Status::ERROR, code, ""}.docs();
        const std::string anchor = link.substr(link.find('#') + 1);
        EXPECT_NE(docs.find("<a id=\"" + anchor + "\"></a>`" + code + "`"), std::string::npos) << code;
    }
}

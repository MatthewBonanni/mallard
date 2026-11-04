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
#include <string>

#include "hdf5_output.h"
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
    EXPECT_TRUE(mentions(r.errors, "physics.p_min was removed in 0.2.0")) << r.errors.front();
    EXPECT_TRUE(mentions(r.errors, "mallard input upgrade"));
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
    EXPECT_TRUE(mentions(r.errors, "no condition for zone top (" + std::to_string(n_top) + " faces)"))
        << (r.errors.empty() ? "" : r.errors[0]);
    EXPECT_FALSE(r.ok());
}

TEST(CheckTimeStep, MatchesTheUniformGridFormulaAndFindsTheFastestRegion) {
    // dt = cfl h / (|u| + |v| + 2a) on a uniform 2D grid (docs/input.md); the
    // flow is 3 times faster for x > 1.5, so that is where dt is set
    const double a = std::sqrt(1.4), h = 2.0 / 16;
    const CheckReport r = check(box("1.0 + 2.0 * (x > 1.5)"));
    ASSERT_TRUE(r.ok()) << (r.errors.empty() ? "" : r.errors[0]);
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
    EXPECT_FALSE(mentions(ok.warnings, "stability limit"));
    const CheckReport fast = check(box("1.0", "t_stop = 0.5\ncfl = 1.5\n"));
    EXPECT_TRUE(mentions(fast.warnings, "stability limit of SSPRK3"));
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
    ASSERT_TRUE(r.ok()) << (r.errors.empty() ? "" : r.errors[0]);
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
    ASSERT_TRUE(r.ok()) << (r.errors.empty() ? "" : r.errors[0]);
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
    ASSERT_TRUE(r.ok()) << (r.errors.empty() ? "" : r.errors[0]);
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
    ASSERT_TRUE(fine.ok()) << (fine.errors.empty() ? describe(fine.unused[0]) : fine.errors[0]);
    EXPECT_FALSE(mentions(fine.warnings, "TFLES"));
    const CheckReport coarse = check(input(5.0 * delta / 10.0));  // F = 10
    EXPECT_TRUE(mentions(coarse.warnings, "F = 10"));
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
    ASSERT_TRUE(r.ok()) << (r.errors.empty() ? "" : r.errors[0]);
    std::map<std::string, std::string> regions(r.regions.begin(), r.regions.end());
    EXPECT_EQ(regions.at("domain"), std::string(N_DIM == 2 ? "area 2" : "volume 0.5") + ", " +
                                        (N_DIM == 2 ? "128" : "256") + " cells (of the mesh)");
    EXPECT_EQ(regions.at("left"), std::string(N_DIM == 2 ? "curve, length 1, 8" : "surface, area 0.25, 16") +
                                      " cells (of the mesh)");
    EXPECT_NE(regions.at("ball").find(N_DIM == 2 ? "area " : "volume "), std::string::npos);
    const std::string on = regions.at("probes[name = \"on\"].region");
    EXPECT_NE(on.find(std::string(N_DIM == 2 ? "curve, length " : "surface, area ") +
                      logging::real(N_DIM == 2 ? 2.0 * std::numbers::pi * 0.3 : 4.0 * std::numbers::pi * 0.01)),
              std::string::npos)
        << on;
    EXPECT_NE(on.find(", exact"), std::string::npos) << on;
    // The same on one rank as on several
    EXPECT_EQ(report(false).regions, r.regions);
}

TEST(CheckRegions, ARegionOutsideTheDomainIsAnErrorWithItsExtent) {
    const std::string far = N_DIM == 2 ? "{type = \"rectangle\", center = [5, 5], size = [1, 2]}"
                                       : "{type = \"box\", center = [5, 5, 5], size = [1, 2, 1]}";
    const CheckReport r = check(box("1.0", "t_stop = 0.5\ncfl = 0.8\n",
                                    "[[sponges]]\nstrength = 1.0\nregion = " + far + "\nu = " + U_ZERO +
                                        "\np = 1.0\nT = 1.0\n"));
    EXPECT_TRUE(mentions(r.errors, "region sponges[0].region lies outside the domain: it spans (4.5, 4"))
        << (r.errors.empty() ? "" : r.errors[0]);
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
    ASSERT_TRUE(r.ok()) << (r.errors.empty() ? "" : r.errors[0]);
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

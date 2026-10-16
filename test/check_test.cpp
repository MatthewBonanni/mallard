/**
 * @file check_test.cpp
 * @brief Pre-flight check (mallard --check): unused keys, boundary coverage,
 *        mesh quality, time step, memory, cost and warnings.
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 */

#include <gtest/gtest.h>

#include <cmath>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <numbers>
#include <sstream>
#include <string>

#include "test_fixtures.h"
#include "check.h"
#include "comm.h"
#include "common_io.h"
#include "memory_tracker.h"
#include "mesh_block.h"
#include "mesh_quality.h"
#include "partition.h"
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
    const std::string dir = (std::filesystem::temp_directory_path() / "mallard_check_valid").string();
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
    if (N_DIM == 2) EXPECT_NEAR(r.dt, 0.8 * h / (3.0 + 2.0 * a), roundoff(1e-12));
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

TEST(CheckOutput, CheckCreatesNoFilesAndWritesTheQualityFieldsOnRequest) {
    const std::filesystem::path dir = std::filesystem::temp_directory_path() / "mallard_check_files";
    std::filesystem::remove_all(dir);
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
    std::ifstream vtu(dir / "quality" / (comm::size() > 1 ? "mesh_000000_p0000.vtu" : "mesh_000000.vtu"));
    ASSERT_TRUE(vtu.good());
    const std::string text((std::istreambuf_iterator<char>(vtu)), std::istreambuf_iterator<char>());
    for (const char * field : MeshQuality::FIELDS) EXPECT_NE(text.find(field), std::string::npos) << field;
    EXPECT_TRUE(output_files_enabled());
}

TEST(CheckSettings, TflesThickeningAboveFiveIsAWarning) {
    // Delta = 1 cm (4 x 4 cells of a 4 cm box), so F = n_res Delta / delta_L
    auto input = [](double delta_L) {
        std::ostringstream s;
        s << "[run]\nn_steps = 1\ndt = 1e-7\n[mesh]\ntype = \"cartesian\"\nNx = 4\nNy = 4\nLx = 0.04\nLy = 0.04\n"
          << (N_DIM == 3 ? "Nz = 4\nLz = 0.04\nperiodic = [\"x\", \"y\", \"z\"]\n" : "periodic = [\"x\", \"y\"]\n")
          << "[initialize]\ntype = \"analytical\"\np = \"101325.0\"\nT = \"300.0\"\nu = " << U_ZERO << "\n"
          << "X = { H2 = 2.0, O2 = 1.0, N2 = 3.76 }\n"
          << "[numerics]\nriemann_solver = \"HLLC\"\n"
          << "[physics]\ntype = \"navier_stokes\"\ngas = \"mixture\"\nmechanism = \"" MALLARD_SOURCE_DIR
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
}

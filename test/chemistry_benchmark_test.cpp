/**
 * @file chemistry_benchmark_test.cpp
 * @author Matthew Bonanni (mbonanni001@gmail.com)
 * @brief The chemistry benchmark: its timed calls start from the sampled states and take the [chemistry] options.
 * @version 0.3
 * @date 2026-10-03
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 *
 */

#include <gtest/gtest.h>

#include <string>

#include "chemistry_benchmark.h"
#include "data_paths.h"
#include "test_fixtures.h"

namespace {

ChemistryBenchmark igniting_ndodecane_cell(const int repeats) {
    return run_chemistry_benchmark(parse_toml(
        "[physics]\ngas = \"mixture\"\nmechanism = \"" + source_dir() +
        "/mechanisms/nDodecane_Reitz.yaml\"\nphase = \"nDodecane_IG\"\n"
        "[reactor]\nT = 1000.0\np = 2026500.0\nX = { c12h26 = 1.0, o2 = 18.5, n2 = 69.56 }\nend_time = 1.85e-3\n"
        "[benchmark]\ncells = 1\nigniting_fraction = 1.0\nsamples = 64\ndt = [1.0e-6]\nrepeats = " +
        std::to_string(repeats) + "\n"));
}

/** @brief The h2o2 ignition of benchmarks/chemistry/h2o2.toml over 64 cells, with these [chemistry] keys. */
ChemistryBenchmark h2o2_cells(const std::string & chemistry, const std::string & dt) {
    return run_chemistry_benchmark(parse_toml(
        "[physics]\ngas = \"mixture\"\nmechanism = \"" + source_dir() + "/mechanisms/h2o2.yaml\"\n"
        "[reactor]\nT = 1200.0\np = 101325.0\nX = { H2 = 2.0, O2 = 1.0, N2 = 3.76 }\nend_time = 1.0e-4\n"
        "[chemistry]\nrtol = 1.0e-6\natol = 1.0e-10\n" + chemistry +
        "[benchmark]\ncells = 64\nsamples = 64\nrepeats = 1\ndt = [" + dt + "]\n"));
}

} // namespace

// The first igniting state takes about a hundred sub-steps over 1e-6 s, the
// same state advanced by the earlier calls a few: every call must restart
// from the sampled state (on host backends a mirror view of the cells' state
// is that state itself)
TEST(ChemistryBenchmarkTest, EveryCallStartsFromTheSampledStates) {
    const ChemistryBenchmark once = igniting_ndodecane_cell(1);
    const ChemistryBenchmark thrice = igniting_ndodecane_cell(3);
    ASSERT_EQ(once.igniting, 1u);
    ASSERT_EQ(once.steps.size(), 1u);
    ASSERT_EQ(thrice.steps.size(), 1u);
    EXPECT_GT(once.steps[0].mean_substeps, 50.0);
    EXPECT_NEAR(thrice.steps[0].mean_substeps, once.steps[0].mean_substeps, 0.1 * once.steps[0].mean_substeps);
}

// [chemistry] explicit_step reaches the benchmarked kernels: over a short
// step most cells are not stiff and take the explicit step (#1474)
TEST(ChemistryBenchmarkTest, TakesTheExplicitStepsOfTheChemistryTable) {
    const ChemistryBenchmark rodas = h2o2_cells("", "1.0e-8");
    const ChemistryBenchmark explicit_steps = h2o2_cells("explicit_step = true\n", "1.0e-8");
    EXPECT_FALSE(rodas.explicit_step);
    EXPECT_EQ(rodas.steps[0].explicit_cells, 0u);
    EXPECT_TRUE(explicit_steps.explicit_step);
    EXPECT_EQ(explicit_steps.steps[0].active, rodas.steps[0].active);
    EXPECT_GT(explicit_steps.steps[0].explicit_cells, explicit_steps.steps[0].active / 2);
}

// [chemistry] T_frozen reaches the benchmarked kernels: the fresh cells, at
// 1200 K, are frozen below 1250 K
TEST(ChemistryBenchmarkTest, FreezesTheCellsBelowTFrozenOfTheChemistryTable) {
    const ChemistryBenchmark all = h2o2_cells("", "1.0e-6");
    const ChemistryBenchmark warm = h2o2_cells("T_frozen = 1250.0\n", "1.0e-6");
    EXPECT_GT(warm.steps[0].active, 0u);
    EXPECT_LT(warm.steps[0].active, all.steps[0].active);
}

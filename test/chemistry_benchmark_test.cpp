/**
 * @file chemistry_benchmark_test.cpp
 * @author Matthew Bonanni (mbonanni001@gmail.com)
 * @brief The chemistry benchmark's timed calls start from the sampled states.
 * @version 0.3
 * @date 2026-10-03
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 *
 */

#include <gtest/gtest.h>

#include <string>

#include "chemistry_benchmark.h"
#include "test_fixtures.h"

namespace {

ChemistryBenchmark igniting_ndodecane_cell(const int repeats) {
    return run_chemistry_benchmark(parse_toml(
        "[physics]\ngas = \"mixture\"\nmechanism = \"" + std::string(MALLARD_SOURCE_DIR) +
        "/mechanisms/nDodecane_Reitz.yaml\"\nphase = \"nDodecane_IG\"\n"
        "[reactor]\nT = 1000.0\np = 2026500.0\nX = { c12h26 = 1.0, o2 = 18.5, n2 = 69.56 }\nend_time = 1.85e-3\n"
        "[benchmark]\ncells = 1\nigniting_fraction = 1.0\nsamples = 64\ndt = [1.0e-6]\nrepeats = " +
        std::to_string(repeats) + "\n"));
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

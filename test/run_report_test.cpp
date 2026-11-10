/**
 * @file run_report_test.cpp
 * @brief The progress table, the summary and the health section of a run.
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 */

#include <gtest/gtest.h>

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <stdexcept>
#include <string>

#include "common.h"
#include "comm.h"
#include "data_paths.h"
#include "log.h"
#include "run_report.h"
#include "units.h"

namespace {

const ProgressColumns ALL{true, true, true};

ProgressRow typical() {
    return {.step = 1200, .t = 0.0123, .dt = 1.1e-5, .done = 0.4, .wall_per_step = 0.035, .cells_per_second = 4.52e5,
            .eta = 3725.2, .min_rho = 0.0812, .min_p = 9.87e4, .max_mach = 1.879, .troubled = 0.0734, .min_T = 298.4,
            .max_T = 2401.7, .sum_Y_error = 3.1e-15, .reacting = 0.215, .chemistry_steps = 37,
            .written = "7 vtkhdf, 1 restart"};
}

/** @brief The text, or rewrites the golden file with MALLARD_UPDATE_GOLDEN=1. */
void expect_golden(const std::string & text, const std::string & name) {
    const std::filesystem::path golden = std::filesystem::path(source_dir()) / "test" / "data" / "run" / name;
    if (const char * update = std::getenv("MALLARD_UPDATE_GOLDEN"); update && std::string(update) == "1") {
        std::filesystem::create_directories(golden.parent_path());
        std::ofstream(golden) << text;
    }
    std::ifstream file(golden);
    ASSERT_TRUE(file) << golden;
    const std::string expected((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
    EXPECT_EQ(text, expected) << "the text differs from " << golden << "; MALLARD_UPDATE_GOLDEN=1 updates it";
}

std::string progress_text(const units::System system) {
    units::set_system(system);
    const report::Stream table = progress_stream(ALL, false);
    report::Text legend(false);
    legend.note(progress_legend(ALL));
    std::string text = legend.str();
    for (const std::string & line : table.header()) text += line + "\n";
    ProgressRow first = typical();
    first.step = 0;
    first.done = 0.0;
    first.wall_per_step = first.eta = -1.0;
    first.written = "";
    text += table.row(progress_cells(ALL, first)) + "\n";
    text += table.row(progress_cells(ALL, typical())) + "\n";
    units::set_system(units::System::SI);
    return text;
}

} // namespace

// Guards every column, unit and format of the progress table, and the units row of a nondimensional case
TEST(RunReport, ProgressTableMatchesTheGolden) {
    expect_golden(progress_text(units::System::SI) + "\n" + progress_text(units::System::NONDIMENSIONAL),
                  "progress.txt");
}

TEST(RunReport, EveryRowKeepsTheColumnsAligned) {
    const report::Stream table = progress_stream(ALL, false);
    const std::string names = table.header()[0];
    const size_t written = names.find("written");
    ProgressRow wide = typical();
    wide.step = 123'456'789;
    wide.eta = 999.0 * 3600.0;
    wide.min_rho = -1.234e-30;
    wide.min_p = -9.876e+30;
    wide.max_mach = 123.456;
    wide.wall_per_step = 12345.0;
    wide.troubled = wide.reacting = 1.0;
    wide.min_T = wide.max_T = 99999.9;
    wide.sum_Y_error = -1.0e-10;
    wide.chemistry_steps = 99999;
    for (const ProgressRow & row : {typical(), wide}) {
        const std::string line = table.row(progress_cells(ALL, row));
        EXPECT_EQ(line.find(row.written), written) << line;
    }
}

TEST(RunReport, ValuesAreBareAndTheirUnitsInTheUnitsRow) {
    const report::Stream table({{"t", "s", 8}, {"written", "files", 0, report::Column::Kind::TEXT}}, false);
    EXPECT_NO_THROW(table.row({"1.5e-03", "2 vtu"}));
    EXPECT_THROW(table.row({"1.5 ms", "2 vtu"}), std::logic_error);
    EXPECT_THROW(table.row({"1.5e-03"}), std::logic_error);
    EXPECT_THROW(report::Stream({{"t", "", 8}}, false), std::logic_error);
}

TEST(RunReport, WrittenFilesAreCountedByKindUntilTheNextRow) {
    if (!comm::is_root()) GTEST_SKIP() << "rank 0 counts the files";
    logging::take_written();
    logging::written(4, 0.1, "vtu", "a_000004.vtu", "interval", "a");
    logging::written(5, 0.2, "restart", "r_000005.restart", "interval", "r");
    logging::written(8, 0.3, "vtu", "a_000008.vtu", "interval", "a");
    EXPECT_TRUE(logging::written_pending());
    EXPECT_EQ(logging::take_written(), "2 vtu, 1 restart");
    EXPECT_FALSE(logging::written_pending());
    EXPECT_EQ(logging::take_written(), "");
}

// Guards the summary's and the health section's tables, units and what they leave out
TEST(RunReport, SummaryAndHealthMatchTheGolden) {
    if (N_DIM != 2) GTEST_SKIP() << "the golden is of a 2D run (points with two coordinates)";
    RunSummary s{.stop = "t_stop = 0.2 reached",
                 .step = 1822,
                 .steps = 1822,
                 .t = 0.2,
                 .n_cells = 460'800,
                 .total = 125.3,
                 .setup = 4.1,
                 .stepping = 117.6,
                 .diagnostics = 0.0,
                 .output = 3.6,
                 .reacting = true,
                 .chemistry = 61.2,
                 .chemistry_slowest = 61.2,
                 .chemistry_mean = 55.0,
                 .chemistry_sent = 0.123,
                 .rejected_counted = true,
                 .rejected = 3,
                 .writers = true,
                 .bytes = 8.14e9,
                 .writing = 12.5,
                 .hashing = 3.4,
                 .hash_queue = 0.2,
                 .background = true,
                 .files = {{"vtkhdf", 11, "solut/dmr.vtkhdf.series"},
                           {"restart", 10, "last restart_001822.restart"},
                           {"forces", 0, "forces_wall.csv"}}};
    health::Report h;
    h.margins_run = {{"rho", 1.38, {0.84, 0.022, 0.0}, 1800, true}, {"p", 0.999, {0.84, 0.022, 0.0}, 1800, true},
                     {"T", 0.713, {1.19, 0.978, 0.0}, 400, true}};
    h.activity = {{"troubled_cells", 0.0743, 0.0612, 0.0911}};
    h.repairs_run = {{"positivity", 12, 7, {}}};
    h.balance_run = {{"mass", 1.0, 2.2e-14, 0.0, 0.0, 0.0, 1.0, true, ""}};
    h.timing = {{"output_sync", 3.63e-5, 3.1e-5, 3.63e-5}};
    report::Text text(false);
    summary_section(text, s);
    health_section(text, h);
    expect_golden(text.str(), "summary_2d.txt");
}

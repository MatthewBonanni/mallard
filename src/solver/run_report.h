/**
 * @file run_report.h
 * @brief What a run prints as it goes and at its end: the progress table (names over units, a legend,
 *        and a written column), the summary and the health section, rendered with report::.
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 */

#ifndef RUN_REPORT_H
#define RUN_REPORT_H

#include <cstdint>
#include <string>
#include <vector>

#include "report.h"
#include "run_health.h"

/** @brief The optional columns of the progress table. */
struct ProgressColumns {
    bool teno = false;      // troubled
    bool mixture = false;   // min T, max T, sum Y-1
    bool reacting = false;  // reacting, chem/cell
};

/** @brief One progress row; a negative wall_per_step or eta is not known yet. */
struct ProgressRow {
    uint64_t step = 0;
    double t = 0.0, dt = 0.0;
    double done = 0.0;  // fraction of the run
    double wall_per_step = -1.0, cells_per_second = 0.0, eta = -1.0;  // s, cells/s, s
    double min_rho = 0.0, min_p = 0.0, max_mach = 0.0;
    double troubled = 0.0;  // fraction of the cells
    double min_T = 0.0, max_T = 0.0, sum_Y_error = 0.0;
    double reacting = 0.0;  // fraction of the cells
    double chemistry_steps = 0.0;
    std::string written{};  // logging::take_written
};

/** @brief The table's columns, with the case's units (units::column). */
report::Stream progress_stream(const ProgressColumns & columns, bool color);

/** @brief The cells of a row, one per column of progress_stream. */
std::vector<std::string> progress_cells(const ProgressColumns & columns, const ProgressRow & row);

/** @brief What each column is, as a note above the table. */
std::string progress_legend(const ProgressColumns & columns);

/** @brief Wall time and throughput of a run, and what it wrote. */
struct RunSummary {
    std::string stop;
    uint64_t step = 0, steps = 0;  // the last step, and those taken in this run
    double t = 0.0;
    uint64_t n_cells = 0;
    double total = 0.0, setup = 0.0, stepping = 0.0, diagnostics = 0.0, output = 0.0;  // wall seconds
    bool reacting = false;
    double chemistry = 0.0;
    double chemistry_slowest = 0.0, chemistry_mean = 0.0;  // per rank, 0 if not measured
    double chemistry_sent = -1.0;                          // fraction integrated on other ranks, -1 without balancing
    bool rejected_counted = false;
    uint64_t rejected = 0;
    bool writers = false;
    double bytes = 0.0, writing = 0.0;
    double hashing = 0.0;     // seconds hashing the files for the run log, off the main thread
    double hash_queue = 0.0;  // seconds the run waited for room in the hashing queue
    bool background = false;
    struct Files {
        std::string kind;
        uint64_t n = 0;  // 0: a file appended to as the run goes
        std::string where;
    };
    std::vector<Files> files{};
};

/** @brief The Summary section. */
void summary_section(report::Text & text, const RunSummary & s);

/** @brief The Health section, from the last health report. */
void health_section(report::Text & text, const health::Report & h);

#endif // RUN_REPORT_H

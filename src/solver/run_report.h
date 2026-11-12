/**
 * @file run_report.h
 * @brief What a run prints: the setup banner, as it goes the progress table (names over units, a legend,
 *        and a written column), and at its end the summary and the health section, rendered with report::.
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 */

#ifndef RUN_REPORT_H
#define RUN_REPORT_H

#include <array>
#include <cstdint>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "check.h"
#include "common_io.h"
#include "report.h"
#include "run_health.h"

/** @brief Settings described in words, (setting, description) in display order. */
using Settings = std::vector<std::pair<std::string, std::string>>;

/** @brief A section of the setup banner: its settings, then its quantities, then tables of its own. */
struct SetupSection {
    struct Table {
        std::string note;
        std::vector<report::Column> columns;
        std::vector<report::Row> rows;
    };
    std::string title;
    Settings settings{};
    std::vector<report::Row> quantities{};  // quantity, value, units, target (empty), basis
    std::vector<Table> tables{};
};

/** @brief The sections of the setup banner, as report::Text renders them. */
void setup_text(report::Text & text, const std::vector<SetupSection> & sections);

/** @brief The mesh over every rank, and its partition in parallel runs. */
struct MeshSetup {
    Settings settings{};                                // type, transform, stretching, cell order, ...
    std::vector<std::pair<std::string, uint64_t>> cells{};  // by type: "tri", "quad", ...
    uint64_t faces = 0, boundary_faces = 0;
    std::array<double, 3> lo{}, hi{};  // extent of the nodes, m
    double size_min = 0.0, size_max = 0.0;  // cell area (2D) or volume (3D)
    struct Partition {
        std::string partitioner, exchange;  // "multi-jagged", "MPI"
        int ranks = 0, halo_layers = 0;
        uint64_t owned_min = 0, owned_max = 0, halo_max = 0;
        uint64_t cut_faces = 0;  // faces between ranks
        uint64_t layer1_max = 0, layer1_all = 0, received_max = 0, received_all = 0;
    };
    std::optional<Partition> partition{};
};

/** @brief The Mesh section. */
SetupSection mesh_section(const MeshSetup & m);

/** @brief The time step and when the run stops. */
struct TimeSetup {
    bool use_cfl = true;
    double cfl = 0.0, dt = 0.0;  // the fixed time step, or the first one with a controller, s
    bool controlled = false;      // the step controller sets dt after the first step
    uint64_t n_steps = 0;         // 0: no limit, as t_stop and wall_stop
    double t_stop = 0.0, wall_stop = 0.0;  // s
};

/** @brief Quantity rows of the time step and the stops. */
std::vector<report::Row> time_quantities(const TimeSetup & t);

/** @brief The table of the boundary zones: condition, faces over every rank and details. */
SetupSection::Table boundary_table(const std::vector<CheckBoundary> & boundaries);

/** @brief The table of the outputs: files, interval and what they hold. */
SetupSection::Table output_table(const std::vector<OutputSummary> & outputs);

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
    std::string outcome;  // as the end record's
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

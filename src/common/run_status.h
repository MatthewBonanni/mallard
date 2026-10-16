/**
 * @file run_status.h
 * @brief The state of a running solver at a progress report: what the run
 *        log's progress records hold.
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 */

#ifndef RUN_STATUS_H
#define RUN_STATUS_H

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "json.h"

/** @brief Memory in bytes, -1 where unknown. */
struct MemoryUsage {
    double host = -1.0;          // resident set of the process
    double host_peak = -1.0;     // its peak
    double device = -1.0;        // in use on the process's GPU, by all processes
    double device_total = -1.0;  // that GPU's memory
};

/** @brief This process's memory use. */
MemoryUsage memory_usage();

/** @brief Named values of a monitor (e.g. the [integrals]) as of the step it last ran. */
struct MonitorValues {
    std::string name;
    uint64_t step = 0;
    double t = 0.0;
    std::vector<std::pair<std::string, double>> values;
};

struct RunStatus {
    uint64_t step = 0;
    double t = 0.0;
    double dt = 0.0;
    double cfl = 0.0;               // dt over the stable time step at CFL 1
    double progress = 0.0;          // fraction of the run, by the first stop condition to be reached
    double wall = 0.0;              // seconds since the solver started
    double wall_per_step = -1.0;    // time-stepping seconds per step since the previous report, -1: none
    double cells_per_second = -1.0; // likewise
    double eta = -1.0;              // seconds, -1: unknown
    uint64_t n_cells = 0;
    int n_ranks = 1;
    MemoryUsage memory;             // largest of the ranks
    std::vector<std::pair<std::string, double>> monitors;  // the progress row's diagnostics
    std::vector<MonitorValues> monitored;                  // the latest values of the monitors

    /** @brief The run log's progress record. */
    json::Object record() const;
};

#endif // RUN_STATUS_H

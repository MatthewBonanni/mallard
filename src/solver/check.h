/**
 * @file check.h
 * @brief Pre-flight check of an input (mallard check): the setup of a real
 *        run without time stepping, and what it tells about the run.
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 */

#ifndef CHECK_H
#define CHECK_H

#include <array>
#include <cstdint>
#include <string>
#include <vector>

#include "input.h"
#include "log.h"
#include "mesh_quality.h"

struct CheckOptions {
    int ranks = 0;              // ranks to estimate memory and cost for (0: this run's)
    double throughput = 0.0;    // cells/s (cell updates per wall second) per rank; 0: steps or none
    uint64_t steps = 0;         // steps to time for a measured throughput
    size_t n_worst = 5;         // worst cells listed per quality metric
    std::string quality_file;   // VTU of the quality fields, or empty
    std::string catalog_file;   // JSON of the keys setup read (key_catalog), or empty
    std::string input_file;     // named before input errors, as a run names it, or empty
};

struct CheckReport {
    std::vector<std::string> errors;
    std::vector<std::string> warnings;
    std::vector<Diagnostic> unused;
    bool setup_done = false;

    logging::Items boundaries;  // zone, condition and faces

    bool has_quality = false;
    std::array<QualitySummary, MeshQuality::N_METRICS> quality;

    // Time step at the initial state
    double dt = 0.0;
    double dt_cfl1 = 0.0;  // the time step at CFL 1
    double cfl = 0.0;      // that dt's CFL number
    std::string dt_limit;  // what sets dt: "CFL", "sponge" or "fixed"
    uint64_t limiting_cell = 0;
    std::array<double, N_DIM> limiting_x{};
    std::array<double, N_DIM> region_lo{}, region_hi{};  // cells within 10% of the smallest dt
    uint64_t n_region_cells = 0;

    // Memory per rank, from Kokkos allocations
    bool device_memory = false;  // a device with its own memory, else host memory
    uint64_t bytes = 0;          // after setup, largest rank
    uint64_t bytes_peak = 0;     // during setup (and timed steps), largest rank
    uint64_t local_cells = 0;    // owned and halo cells of that rank
    int ranks = 1;               // the run's ranks
    int target_ranks = 1;
    uint64_t target_owned = 0, target_halo = 0;  // largest part at target_ranks
    uint64_t target_bytes = 0, target_bytes_peak = 0;
    uint64_t device_capacity = 0;  // bytes, 0 if unknown

    // Cost at target_ranks
    uint64_t n_cells = 0;
    double throughput = 0.0;  // cells/s per rank
    bool throughput_measured = false;
    uint64_t steps_timed = 0;
    double wall_per_time = 0.0;        // wall seconds per unit simulated time
    double flow_through_time = 0.0;    // domain length along the mean velocity over its magnitude, 0 if none
    double t_stop = 0.0;
    uint64_t n_steps = 0;

    bool ok() const { return errors.empty() && unused.empty(); }
};

/** @brief Print a report on rank 0. */
void print_check(const CheckReport & report);

#endif // CHECK_H

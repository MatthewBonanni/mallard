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
#include "memory_tracker.h"
#include "mesh_quality.h"

struct CheckOptions {
    int ranks = 0;              // ranks to estimate memory and cost for (0: this run's)
    double throughput = 0.0;    // cells/s (cell updates per wall second) per rank; 0: steps or none
    uint64_t steps = 0;         // steps to time for a measured throughput
    size_t n_worst = 5;         // worst cells listed per quality metric
    std::string quality_file;   // prefix of the quality fields' snapshot, or empty
    std::string catalog_file;   // JSON of the keys setup read (key_catalog), or empty
    std::string view_dir;       // directory of the case view (case_view.h), or empty
    std::string input_file;     // named before input errors, as a run names it, or empty
    uint64_t memory = 0;        // memory per rank to size the run for (bytes); 0: the device's, if any
    std::string memory_name;    // what it is ("A100-80GB", "40GB")
    bool memory_gpu = false;    // a GPU's memory: CUDA's context comes on top of Kokkos' allocations
};

struct CheckReport {
    std::vector<std::string> errors;
    std::vector<std::string> warnings;
    std::vector<Diagnostic> unused;
    bool setup_done = false;

    logging::Items boundaries;  // zone, condition and faces
    logging::Items regions;     // name, then dimension, measure, cells and how it is made
    std::string view_file;      // the case view written, or empty

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
    memory::Bytes setup_bytes{};   // largest rank by category: at the setup's peak
    memory::Bytes steady_bytes{};  // after setup, or at the peak of the timed steps
    uint64_t outside_kokkos = 0;   // device memory beyond Kokkos' allocations (CUDA context, libraries), measured

    // Sizing for options.memory (or the device): the largest rank's peak against the memory per rank
    std::string sizing_name;       // empty: no sizing
    bool sizing_gpu = false;
    uint64_t sizing_memory = 0;    // bytes per rank
    uint64_t sizing_overhead = 0;  // outside Kokkos per rank (CUDA context), measured or nominal
    bool overhead_measured = false;
    double bytes_per_cell = 0.0;   // of the peak that limits the cells per rank, cell-proportional categories
    uint64_t fixed_bytes = 0;      // the rest: sized by the device, not the mesh
    bool sizing_setup = false;     // the setup's peak limits the cells per rank, else the steady state
    uint64_t max_local_cells = 0;  // owned and halo cells per rank that fit
    int min_ranks = 0;             // fewest ranks the mesh fits on, 0 if not found
    uint64_t min_owned = 0, min_halo = 0, min_peak = 0;  // largest part there and its predicted peak
    bool min_partitioned = false;  // the part comes from the partitioner (serial checks), else from this halo ratio
    int halo_ranks = 0;            // most ranks (doubling from min_ranks) with halo below 10% of owned, 0 if none

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

/**
 * @brief Set options.memory from --memory: a size with its unit (40GB, 512MiB) or a GPU of the table
 *        (a100-40, a100-80, h100-80, h200, mi250x, mi300x). Returns an error, or empty.
 */
std::string set_memory_target(const std::string & text, CheckOptions & options);

/** @brief Print a report on rank 0. */
void print_check(const CheckReport & report);

#endif // CHECK_H

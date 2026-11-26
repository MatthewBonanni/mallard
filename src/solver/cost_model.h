/**
 * @file cost_model.h
 * @brief The cost estimate of mallard check: a rank's throughput from its device and what the case computes per
 *        cell, and the time of a step of the largest part on N ranks from its owned and halo cells, neighbors and
 *        exchanges between nodes. Coefficients fitted to measured runs (cost_model_data.h, written by
 *        tools/perf/cost_model.py; docs/guide/parallel.md#the-cost-estimate).
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 */

#ifndef COST_MODEL_H
#define COST_MODEL_H

#include <cstdint>
#include <string>

#include "partition.h"

namespace cost_model {

/** @brief What a rank runs on and what the case computes per cell. */
struct Workload {
    std::string device;          // class: "a100", "cpu-apple-m", "cpu" (other CPUs), "gpu" (other GPUs)
    std::string hardware;        // as the device names itself
    int threads = 1;             // per rank, on CPUs
    int dim = 0;
    std::string reconstruction;  // "fo", "muscl", "teno3" ... "teno6"
    std::string physics;         // "euler", "ns", "les", "reacting"
    int species = 0;             // reacting: the mechanism's
};

/** @brief The device class of a device name ("NVIDIA A100-SXM4-80GB (85 GB)", "Apple M4 Pro"), GPU or not. */
std::string device_class(const std::string & name, bool gpu);

/** @brief A rank's throughput alone on its device, without communication. */
struct Throughput {
    double cells_per_second = 0.0;  // 0: no estimate (basis says why)
    double uncertainty = 0.0;       // relative, one standard deviation of its logarithm
    std::string basis;
};

/**
 * @brief The measured throughput of this device class, dimension, reconstruction and physics (the geometric mean
 *        of its measurements; per thread times the threads on CPUs), else the fit over every measurement.
 */
Throughput throughput(const Workload & w);

/** @brief The coefficients of the step time on a device class (cost_model_data.h). */
struct Scaling {
    const char * device;
    double t0;           // s per step, independent of the cells (launches, reductions, synchronization)
    double beta;         // cost of a halo cell over that of an owned cell
    double gamma;        // s per step per neighbor part
    double delta;        // s per step per halo cell owned on another node
    double particle;     // s per step per particle slot
    double particle_mpi; // s per step added by particles on several ranks (no step graphs on GPUs)
    int runs;            // measured step times fitted
    int cases;
    double fit_error;      // root mean square of the relative step-time error over the fitted runs
    double held_out_error; // over the runs left out: every case, each predicted by the fit to the others
    const char * held_out;
    const char * basis;
};

/** @brief The step-time coefficients of a device class (other CPUs: the measured CPU's), or null if none were
 *         fitted for it. */
const Scaling * scaling(const std::string & device);

/** @brief The largest part's share of the work on n ranks. */
struct Part {
    double owned = 0.0, halo = 0.0;
    double neighbors = 0.0;
    double halo_remote = 0.0;
    double particles = 0.0;  // particle slots
    int ranks = 1;
};

Part part(const PartSize & p, int ranks, double particles);

/**
 * @brief Seconds per step of the largest part: s (o + beta h) + t0 + gamma k + delta h_remote + particle P (+
 *        particle_mpi on several ranks with particles), s the seconds per owned cell of one rank alone.
 */
double step_seconds(const Scaling & c, double seconds_per_cell, const Part & p);

/** @brief The seconds per cell that give step_seconds = seconds on this part (a measured step); if the overheads
 *         alone exceed it, seconds over the part's cells. */
double seconds_per_cell(const Scaling & c, const Part & p, double seconds);

} // namespace cost_model

#endif // COST_MODEL_H

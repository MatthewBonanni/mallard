/**
 * @file chemistry_benchmark.h
 * @author Matthew Bonanni (mbonanni001@gmail.com)
 * @brief Chemistry throughput benchmark: states sampled along a reactor's
 *        trajectory, replicated over many cells and advanced by the solver's
 *        chemistry kernels.
 * @version 0.3
 * @date 2026-10-03
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 *
 */

#ifndef CHEMISTRY_BENCHMARK_H
#define CHEMISTRY_BENCHMARK_H

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include <toml.hpp>

/** @brief The timed calls of one splitting step dt (the best one's time, the last one's sub-steps). */
struct ChemistryBenchmarkStep {
    double dt = 0.0;
    double seconds = 0.0;
    uint64_t active = 0;
    double mean_substeps = 0.0;
    double max_substeps = 0.0;
    std::string histogram;  // cells per bin of sub-steps, e.g. "1:960 65-128:64"
};

struct ChemistryBenchmark {
    std::string mechanism;
    uint32_t species = 0, reactions = 0;
    uint32_t threads = 0, lanes = 0;
    size_t shared_bytes = 0;
    uint32_t cells = 0, igniting = 0;
    std::vector<ChemistryBenchmarkStep> steps;
};

/**
 * @brief The [benchmark] mode of MallardReactor: states sampled along the
 *        trajectory of [reactor] (fresh, igniting and burnt), replicated
 *        over the cells and advanced over each splitting step dt, every call
 *        from the sampled states; logs and returns cells per second and the
 *        sub-step distribution.
 */
ChemistryBenchmark run_chemistry_benchmark(const toml::value & input);

#endif // CHEMISTRY_BENCHMARK_H

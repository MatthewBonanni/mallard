/**
 * @file cost_model_data.h
 * @brief The coefficients of mallard check's cost model (cost_model.h), fitted to the measurements of
 *        benchmarks/cost_model/ by tools/perf/cost_model.py fit: do not edit, regenerate.
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 */

#ifndef COST_MODEL_DATA_H
#define COST_MODEL_DATA_H

#include <cstdint>

#include "cost_model.h"

namespace cost_model {

/** @brief A throughput measured on one rank alone (per thread on CPUs). */
struct Measurement {
    const char * device;
    const char * hardware;
    int dim;
    const char * reconstruction;
    const char * physics;
    int species;
    double cells_per_second;
    int threads;
    const char * case_name;
    const char * date;
    const char * source;
};

/** @brief log(cells/s) of MUSCL NS on a device class in a dimension. */
struct Base {
    const char * device;
    int dim;
    double log_value;
};

/** @brief log of the throughput of a reconstruction or physics over MUSCL or NS, on CPUs or GPUs. */
struct Factor {
    const char * kind;
    const char * variable;
    const char * level;
    double log_value;
};

/** @brief d log(cells/s) / d log(species + 1) of reacting cases, from reference species. */
struct SpeciesSlope {
    const char * kind;
    double slope;
    int reference;
};

/** @brief A step time the fit gives, for the test that check's evaluation matches the fit's. */
struct StepCheck {
    const char * device;
    double seconds_per_cell;
    double owned, halo, neighbors, halo_remote, particles;
    int ranks;
    double step;
};

/** @brief Root mean square of the log error of measurements predicted from the others (leave one out). */
inline constexpr double MODEL_ERROR = 0.4439;
/** @brief Root mean square spread of log(cells/s) between the measurements of one key. */
inline constexpr double SAME_KEY_ERROR = 0.1999;

inline constexpr Measurement MEASUREMENTS[] = {
    {"a100", "A100-SXM4-80GB", 2, "muscl", "euler", 0, 2.78e+08, 0, "muscl_2d", "2026-11-09",
     "mallard-gui throughput table: Mallard/benchmarks/perf/baselines.csv, muscl_2d (performance set baseline, best of 3)"},
    {"a100", "A100", 2, "teno3", "euler", 0, 1.389e+08, 0, "riemann_2d_quads", "2026-10-03",
     "mallard-gui throughput table: Mallard/docs/guide/performance.md, first table: A100 (Mallard 0.4.0)"},
    {"a100", "A100-SXM4-80GB", 2, "teno5", "euler", 0, 5.35e+07, 0, "teno5_2d", "2026-11-09",
     "mallard-gui throughput table: Mallard/benchmarks/perf/baselines.csv, teno5_2d (performance set baseline, best of 3)"},
    {"a100", "A100-SXM4-80GB", 3, "muscl", "les", 0, 5.1e+07, 0, "air_3d", "2026-11-09",
     "mallard-gui throughput table: Mallard/benchmarks/perf/baselines.csv, air_3d (performance set baseline, best of 3)"},
    {"a100", "NVIDIA A100-SXM4-80GB", 3, "muscl", "les", 0, 4.23122e+07, 0, "es-air_dyn-n1-plain", "2026-11-22",
     "ovn0110: es-air_dyn-n1-plain-r1-0caeda57, es-air_dyn-n1-plain-r2-f99d40d8 (Mallard 9da4ab1e70)"},
    {"a100", "A100-SXM4-80GB", 3, "muscl", "ns", 0, 2.44e+07, 0, "sphere_3d", "2026-11-09",
     "mallard-gui throughput table: Mallard/benchmarks/perf/baselines.csv, sphere_3d (performance set baseline, best of 3)"},
    {"a100", "A100-SXM4-80GB", 3, "muscl", "reacting", 6, 7.751e+06, 0, "es-flame_dyn-n1-plain", "2026-11-22",
     "mallard-gui throughput table: mallard-devtools/logs/ovn0110/runs/es-flame_dyn-n1-plain-r1-56289962, es-flame_dyn-n1-plain-r2-25151d87 (ovn0110 campaign c1097, median of 2 runs)"},
    {"a100", "A100-SXM4-80GB", 3, "teno5", "euler", 0, 4.073e+06, 0, "t-expl-hex-main", "2026-11-22",
     "mallard-gui throughput table: mallard-devtools/logs/c978-mem/runs/t-expl-hex-main-r1-59d01a19, t-expl-hex-main-r2-123c9ac2, t-expl-hex-main-r3-eff36e22, t-expl-hex-main-r4-42f7b0cb, t-expl-hex-main-r5-ae57818b (c978-mem campaign c978m, median of 5 runs)"},
    {"a100", "A100-SXM4-80GB", 3, "teno5", "ns", 0, 8.515e+06, 0, "st128-main-n1", "2026-11-22",
     "mallard-gui throughput table: mallard-devtools/logs/ovn529/runs/st128-main-n1-r1-9813ba8f, st128-main-n1-r2-98e0fe4e (ovn529 campaign c529, median of 2 runs)"},
    {"a100", "NVIDIA A100-SXM4-80GB", 3, "teno5", "ns", 0, 8.56549e+06, 0, "ws128-main-n1", "2026-11-22",
     "ovn0110b: ws128-main-n1-r1-51b097c3 (Mallard a7a5432e4e)"},
    {"a100", "NVIDIA A100-SXM4-80GB", 3, "teno5", "ns", 0, 8.57131e+06, 0, "ws64-main-n1", "2026-11-22",
     "ovn0110: ws64-main-n1-r1-2f8fa75d, ws64-main-n1-r2-202483f1 (Mallard 9da4ab1e70)"},
    {"a100", "NVIDIA A100-SXM4-80GB", 3, "teno5", "ns", 0, 8.58231e+06, 0, "ws64-main-n1", "2026-11-22",
     "ovn0110b: ws64-main-n1-r1-4dc2bb67, ws64-main-n1-r2-5f24dd9e (Mallard a7a5432e4e)"},
    {"cpu-apple-m", "Apple M4 Pro", 2, "muscl", "euler", 0, 1.832e+06, 6, "riemann_2d_quads", "2026-10-02",
     "mallard-gui throughput table: Mallard/docs/guide/performance.md, first table: CPU, 6 threads (Mallard 0.3.0)"},
    {"cpu-apple-m", "Apple M4 Pro", 2, "muscl", "les", 0, 1.161e+06, 2, "les2d-muscl", "2026-11-23",
     "mallard-gui throughput table: mallard-devtools/logs/throughput-cpu/runs/les2d-muscl (local runs throughput-cpu)"},
    {"cpu-apple-m", "Apple M4 Pro", 2, "muscl", "ns", 0, 1.55134e+06, 0, "cpu-ns2d-muscl-384-n1", "2026-11-23",
     "cost-cpu: cpu-ns2d-muscl-384-n1, cpu-ns2d-muscl-384-n1-r2-e8600486, cpu-ns2d-muscl-384-n1-r3-e8600486 (Mallard 13327c0edd)"},
    {"cpu-apple-m", "Apple M4 Pro", 2, "muscl", "ns", 0, 1.57461e+06, 2, "cpu-ns2d-muscl-384-t2-n1", "2026-11-23",
     "cost-cpu: cpu-ns2d-muscl-384-t2-n1, cpu-ns2d-muscl-384-t2-n1-r2-a9c0af54, cpu-ns2d-muscl-384-t2-n1-r3-a9c0af54 (Mallard 13327c0edd)"},
    {"cpu-apple-m", "Apple M4 Pro", 2, "muscl", "ns", 0, 1.462e+06, 2, "ns2d-muscl", "2026-11-23",
     "mallard-gui throughput table: mallard-devtools/logs/throughput-cpu/runs/ns2d-muscl (local runs throughput-cpu)"},
    {"cpu-apple-m", "Apple M4 Pro", 2, "muscl", "reacting", 6, 123559, 2, "cpu-reacting2d-c3h8_2step-n1", "2026-11-23",
     "cost-cpu: cpu-reacting2d-c3h8_2step-n1, cpu-reacting2d-c3h8_2step-n1-r2-5ad71750, cpu-reacting2d-c3h8_2step-n1-r3-5ad71750 (Mallard 13327c0edd)"},
    {"cpu-apple-m", "Apple M4 Pro", 2, "muscl", "reacting", 6, 206300, 2, "reacting2d-muscl", "2026-11-23",
     "mallard-gui throughput table: mallard-devtools/logs/throughput-cpu/runs/reacting2d-muscl (local runs throughput-cpu)"},
    {"cpu-apple-m", "Apple M4 Pro", 2, "muscl", "reacting", 10, 70181.9, 2, "cpu-reacting2d-h2o2-n1", "2026-11-23",
     "cost-cpu: cpu-reacting2d-h2o2-n1, cpu-reacting2d-h2o2-n1-r2-0bb168e3, cpu-reacting2d-h2o2-n1-r3-0bb168e3 (Mallard 13327c0edd)"},
    {"cpu-apple-m", "Apple M4 Pro", 2, "muscl", "reacting", 53, 6288.78, 2, "cpu-reacting2d-gri30-n1", "2026-11-23",
     "cost-cpu: cpu-reacting2d-gri30-n1, cpu-reacting2d-gri30-n1-r2-653e7509, cpu-reacting2d-gri30-n1-r3-653e7509 (Mallard 13327c0edd)"},
    {"cpu-apple-m", "Apple M4 Pro", 2, "teno3", "euler", 0, 1.333e+06, 6, "riemann_2d_quads", "2026-10-02",
     "mallard-gui throughput table: Mallard/docs/guide/performance.md, first table: CPU, 6 threads (Mallard 0.3.0)"},
    {"cpu-apple-m", "Apple M4 Pro", 2, "teno3", "ns", 0, 1.229e+06, 2, "ns2d-teno3", "2026-11-23",
     "mallard-gui throughput table: mallard-devtools/logs/throughput-cpu/runs/ns2d-teno3 (local runs throughput-cpu)"},
    {"cpu-apple-m", "Apple M4 Pro", 2, "teno5", "euler", 0, 281100, 6, "riemann_2d_quads", "2026-10-02",
     "mallard-gui throughput table: Mallard/docs/guide/performance.md, first table: CPU, 6 threads (Mallard 0.3.0)"},
    {"cpu-apple-m", "Apple M4 Pro", 2, "teno5", "ns", 0, 310967, 0, "cpu-ns2d-teno5-256-n1", "2026-11-23",
     "cost-cpu: cpu-ns2d-teno5-256-n1, cpu-ns2d-teno5-256-n1-r2-c03d524c, cpu-ns2d-teno5-256-n1-r3-c03d524c (Mallard 13327c0edd)"},
    {"cpu-apple-m", "Apple M4 Pro", 2, "teno5", "ns", 0, 493900, 2, "ns2d-teno5", "2026-11-23",
     "mallard-gui throughput table: mallard-devtools/logs/throughput-cpu/runs/ns2d-teno5 (local runs throughput-cpu)"},
    {"cpu-apple-m", "Apple M4 Pro", 2, "teno6", "euler", 0, 187600, 2, "euler2d-teno6", "2026-11-23",
     "mallard-gui throughput table: mallard-devtools/logs/throughput-cpu/runs/euler2d-teno6 (local runs throughput-cpu)"},
    {"cpu-apple-m", "Apple M4 Pro", 2, "teno6", "ns", 0, 264200, 2, "ns2d-teno6", "2026-11-23",
     "mallard-gui throughput table: mallard-devtools/logs/throughput-cpu/runs/ns2d-teno6 (local runs throughput-cpu)"},
    {"cpu-apple-m", "Apple M4 Pro", 3, "muscl", "euler", 0, 717500, 2, "euler3d-muscl", "2026-11-23",
     "mallard-gui throughput table: mallard-devtools/logs/throughput-cpu/runs/euler3d-muscl (local runs throughput-cpu)"},
    {"cpu-apple-m", "Apple M4 Pro", 3, "muscl", "les", 0, 490000, 2, "les3d-muscl", "2026-11-23",
     "mallard-gui throughput table: mallard-devtools/logs/throughput-cpu/runs/les3d-muscl (local runs throughput-cpu)"},
    {"cpu-apple-m", "Apple M4 Pro", 3, "muscl", "ns", 0, 502750, 0, "cpu-ns3d-muscl-48-n1", "2026-11-23",
     "cost-cpu: cpu-ns3d-muscl-48-n1, cpu-ns3d-muscl-48-n1-r2-4cdd6a39, cpu-ns3d-muscl-48-n1-r3-4cdd6a39 (Mallard 13327c0edd)"},
    {"cpu-apple-m", "Apple M4 Pro", 3, "muscl", "ns", 0, 512100, 2, "ns3d-muscl", "2026-11-23",
     "mallard-gui throughput table: mallard-devtools/logs/throughput-cpu/runs/ns3d-muscl (local runs throughput-cpu)"},
    {"cpu-apple-m", "Apple M4 Pro", 3, "muscl", "reacting", 6, 122500, 2, "reacting3d-muscl", "2026-11-23",
     "mallard-gui throughput table: mallard-devtools/logs/throughput-cpu/runs/reacting3d-muscl (local runs throughput-cpu)"},
    {"cpu-apple-m", "Apple M4 Pro", 3, "teno3", "euler", 0, 346300, 2, "euler3d-teno3", "2026-11-23",
     "mallard-gui throughput table: mallard-devtools/logs/throughput-cpu/runs/euler3d-teno3 (local runs throughput-cpu)"},
    {"cpu-apple-m", "Apple M4 Pro", 3, "teno3", "ns", 0, 341000, 2, "ns3d-teno3", "2026-11-23",
     "mallard-gui throughput table: mallard-devtools/logs/throughput-cpu/runs/ns3d-teno3 (local runs throughput-cpu)"},
    {"cpu-apple-m", "Apple M4 Pro", 3, "teno5", "euler", 0, 67380, 2, "euler3d-teno5", "2026-11-23",
     "mallard-gui throughput table: mallard-devtools/logs/throughput-cpu/runs/euler3d-teno5 (local runs throughput-cpu)"},
    {"cpu-apple-m", "Apple M4 Pro", 3, "teno5", "ns", 0, 137248, 0, "cpu-ns3d-teno5-32-n1", "2026-11-23",
     "cost-cpu: cpu-ns3d-teno5-32-n1, cpu-ns3d-teno5-32-n1-r2-2f2d1071, cpu-ns3d-teno5-32-n1-r3-2f2d1071 (Mallard 13327c0edd)"},
    {"cpu-apple-m", "Apple M4 Pro", 3, "teno5", "ns", 0, 80800, 2, "ns3d-teno5", "2026-11-23",
     "mallard-gui throughput table: mallard-devtools/logs/throughput-cpu/runs/ns3d-teno5 (local runs throughput-cpu)"},
    {"cpu-apple-m", "Apple M4 Pro", 3, "teno6", "euler", 0, 38810, 2, "euler3d-teno6", "2026-11-23",
     "mallard-gui throughput table: mallard-devtools/logs/throughput-cpu/runs/euler3d-teno6 (local runs throughput-cpu)"},
    {"cpu-apple-m", "Apple M4 Pro", 3, "teno6", "ns", 0, 45670, 2, "ns3d-teno6", "2026-11-23",
     "mallard-gui throughput table: mallard-devtools/logs/throughput-cpu/runs/ns3d-teno6 (local runs throughput-cpu)"},
};

inline constexpr Base BASES[] = {
    {"a100", 2, 19.954799},
    {"a100", 3, 17.194795},
    {"cpu-apple-m", 2, 14.353782},
    {"cpu-apple-m", 3, 13.208651},
};

inline constexpr Factor FACTORS[] = {
    {"cpu", "physics", "euler", -0.071524},
    {"cpu", "physics", "les", -0.247740},
    {"cpu", "physics", "ns", 0.000000},
    {"cpu", "physics", "reacting", -2.149090},
    {"cpu", "reconstruction", "muscl", 0.000000},
    {"cpu", "reconstruction", "teno3", -0.340616},
    {"cpu", "reconstruction", "teno5", -1.665479},
    {"cpu", "reconstruction", "teno6", -2.264915},
    {"gpu", "physics", "euler", -0.696368},
    {"gpu", "physics", "les", 0.459166},
    {"gpu", "physics", "ns", 0.000000},
    {"gpu", "physics", "reacting", -1.331463},
    {"gpu", "reconstruction", "muscl", 0.000000},
    {"gpu", "reconstruction", "teno3", -0.509165},
    {"gpu", "reconstruction", "teno5", -1.278536},
};

inline constexpr SpeciesSlope SPECIES_SLOPES[] = {
    {"cpu", -1.721618, 6},
};

inline constexpr Scaling SCALINGS[] = {
    {"a100", 0.00114177, 3.3797e-19, 7.59602e-25, 3.76966e-08, 0, 0, 22, 5, 0.0192, 0.2014,
     "es-air_dyn-plain@9da4ab1 41.3%, ws64-main@9da4ab1 1.5%, ws128-main@a7a5432 5.6%, ws64-main@a7a5432 1.2%, st128-main@a7a5432 2.9%",
     "NVIDIA A100-SXM4-80GB, 2026-11-22"},
    {"cpu-apple-m", 0.0236406, 1.51961, 4.46957e-16, 0, 1.16633e-08, 0.00787553, 15, 4, 0.0916, 0.3260,
     "cpu-ns2d-muscl-384@13327c0 30.0%, cpu-ns2d-teno5-256@13327c0 24.2%, cpu-ns3d-muscl-48@13327c0 24.2%, cpu-ns3d-teno5-32@13327c0 49.2%",
     "Apple M4 Pro, 2026-11-23"},
};

inline constexpr StepCheck STEP_CHECKS[] = {
    {"a100", 1.884201635e-08, 221184, 0, 0, 0, 0, 1, 0.005309325291},
    {"a100", 1.123846132e-07, 262144, 0, 0, 0, 0, 1, 0.03060272479},
    {"a100", 1.123846132e-07, 262144, 32768, 2, 16384, 0, 8, 0.03122034656},
    {"a100", 1.125179705e-07, 262144, 32768, 1, 0, 0, 2, 0.0306376836},
    {"a100", 1.13914273e-07, 524288, 139264, 3, 0, 0, 4, 0.06086565909},
    {"cpu-apple-m", 5.586804216e-07, 147456, 0, 0, 0, 0, 1, 0.1060213558},
    {"cpu-apple-m", 5.586804216e-07, 73728, 1536, 1, 0, 0, 2, 0.06613499262},
    {"cpu-apple-m", 2.477838344e-06, 65536, 0, 0, 0, 0, 1, 0.1860281893},
    {"cpu-apple-m", 1.539187366e-06, 110592, 0, 0, 0, 0, 1, 0.1938623847},
    {"cpu-apple-m", 7.227224842e-06, 32768, 0, 0, 0, 0, 1, 0.2604622792},
};

} // namespace cost_model

#endif // COST_MODEL_DATA_H

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
#include <limits>
#include <string>
#include <utility>
#include <vector>

#include "binding.h"
#include "case_view.h"
#include "diagnostic_code.h"
#include "input.h"
#include "json.h"
#include "log.h"
#include "memory_tracker.h"
#include "mesh_quality.h"
#include "report.h"

/** @brief The share of the memory per rank that sizing keeps free (fragmentation, allocations in the run). */
constexpr double SIZING_MARGIN = 0.05;
/** @brief The halo, as a share of the owned cells, that the ranks sizing recommends stay below. */
constexpr double SIZING_HALO_FRACTION = 0.1;

/** @brief Fields of the owned cells by name, written with the mesh-quality fields (--mesh-fields). */
using CheckFields = std::vector<std::pair<std::string, std::vector<double>>>;

struct CheckOptions {
    int ranks = 0;              // ranks to estimate memory and cost for (0: this run's)
    double throughput = 0.0;    // cells/s (cell updates per wall second) per rank; 0: steps or none
    uint64_t steps = 0;         // steps to time for a measured throughput
    size_t n_worst = 5;         // worst cells listed per quality metric
    std::string mesh_fields;    // prefix of the mesh and stencil fields' snapshot (--mesh-fields), or empty
    bool all_stencils = false;  // the stencil file of every cell (--stencils all)
    std::vector<uint64_t> stencil_cells;  // or of these cells, by global id (--cells)
    std::string stencil_region;           // or of the cells of this region: a name or an inline table (--region)
    bool stencil_weights = false;         // with the pseudo-inverse rows (--weights)
    std::string catalog_file;   // JSON of the keys setup read (key_catalog), or empty
    std::string view_dir;       // directory of the case view (case_view.h), or empty
    std::string input_file;     // named before input errors, as a run names it, or empty
    uint64_t memory = 0;        // memory per rank to size the run for (bytes); 0: the device's, if any
    std::string memory_name;    // what it is ("A100-80GB", "40GB")
    bool memory_gpu = false;    // a GPU's memory: CUDA's context comes on top of Kokkos' allocations
};

static_assert(diagnostic_code::QUALITY.size() == MeshQuality::N_METRICS);

/** @brief An error or a warning of the check: its code, and what is wrong, with its numbers and the fix. */
struct CheckDiagnostic {
    report::Status severity = report::Status::ERROR;  // ERROR or WARNING
    std::string code;                                 // one of diagnostic_code::ALL
    std::string message;

    /** @brief The page and anchor that explain the code: "docs/guide/input_files.md#w-cfl-limit". */
    std::string docs() const;
};

/** @brief A boundary zone with its condition. */
struct CheckBoundary {
    std::string zone;
    std::string condition;
    uint64_t faces = 0;      // over every rank
    std::string details;     // how the condition varies or what models it, or empty
    std::string turbulence;  // its synthetic inflow turbulence, or empty
};

/** @brief A region on the mesh: one of the mesh's own (the domain, its zones) or one the case uses. */
struct CheckRegion {
    std::string name;
    bool of_mesh = false;
    int dimension = 0;        // 0 points, 1 curve, 2 surface, N_DIM filled
    double measure = 0.0;     // number of points, length, area or volume (SI units)
    bool per_radian = false;  // a filled region of an axisymmetric case
    uint64_t cells = 0;       // owned cells it touches, over every rank
    bool exact = true;        // a surface or curve of the case: exact, else extracted at resolution
    int resolution = 0;
};

/** @brief An estimate of the resolution: its value and units, what it rests on, and its target if it has one. */
struct CheckEstimate {
    std::string id;        // stable: "y_plus_max"
    std::string quantity;  // as the table names it
    double value = std::numeric_limits<double>::quiet_NaN();  // NaN: not estimated, basis says why
    std::string units;     // SI ("m", "m/s"), "wall units" or "dimensionless"
    std::string target;    // what the regime needs ("<= 1 (DNS)"), or empty
    report::Status status = report::Status::NONE;  // against the target
    std::string basis;     // what it is estimated from
    bool located = false;  // x: where the value is
    std::array<double, N_DIM> x{};
};

/** @brief What a run would run on: the build, where each rank runs, MPI and NCCL (check_environment). */
struct CheckEnvironment {
    bool checked = false;
    std::string build, dependencies;    // the provenance blocks (JSON objects)
    std::vector<binding::Placement> ranks;  // rank 0's report only
    std::vector<std::string> devices;   // of each rank: its GPU and memory, else the CPU model
    int hosts = 1;
    int ranks_per_host = 1;             // the most on one host
    std::vector<int> solvers;           // dimensions of the solvers installed with this one
    int gpu_aware_library = -1;         // the MPI library's own answer for this GPU: 1 yes, 0 no, -1 unknown
    bool gpu_aware_used = false;        // device buffers go to MPI directly (several ranks)
    bool nccl_used = false;             // halo exchanges by NCCL (several ranks, [parallel] halo_exchange)
};

struct CheckReport {
    std::vector<CheckDiagnostic> diagnostics;
    std::vector<Diagnostic> unused;  // each also a KEY_UNUSED diagnostic
    bool setup_done = false;
    std::string resolved;  // the input with every key explicit (resolve_input), once setup is done
    std::string input_file;
    CheckEnvironment environment;

    std::vector<CheckBoundary> boundaries;
    std::vector<CheckRegion> regions;
    std::string view_file;     // the case view's ParaView file written, or empty
    std::vector<std::string> view_images;  // and its images
    std::string mesh_fields_file;  // the mesh and stencil fields written, or empty
    std::string stencil_file;      // the stencils written, or empty

    bool has_quality = false;
    std::array<QualitySummary, MeshQuality::N_METRICS> quality;

    // Time step at the initial state
    double t = 0.0;  // its time
    double dt = 0.0;
    double dt_cfl1 = 0.0;    // the time step at CFL 1
    double cfl = 0.0;        // that dt's CFL number
    double cfl_limit = 0.0;  // the time integrator's for smooth flow, infinity if it has none
    std::string integrator;
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
    uint64_t local_owned = 0;    // its owned cells
    int ranks = 1;               // the run's ranks
    int target_ranks = 1;
    uint64_t target_owned = 0, target_halo = 0;  // largest part at target_ranks
    uint64_t target_bytes = 0, target_bytes_peak = 0;
    uint64_t device_capacity = 0;  // bytes, 0 if unknown
    memory::Bytes setup_bytes{};   // largest rank by category: at the setup's peak
    memory::Bytes steady_bytes{};  // after setup, or at the peak of the timed steps
    std::vector<memory::Bytes> setup_phases;  // at the peak of each phase of the setup
    uint64_t outside_kokkos = 0;   // device memory beyond Kokkos' allocations (CUDA context, libraries), measured

    // Sizing for options.memory (or the device): the largest rank's peak against the memory per rank
    std::string sizing_name;       // empty: no sizing
    bool sizing_gpu = false;
    uint64_t sizing_memory = 0;    // bytes per rank
    uint64_t sizing_overhead = 0;  // outside Kokkos per rank (CUDA context), measured or nominal
    bool overhead_measured = false;
    double bytes_per_cell = 0.0;   // of the peak that limits the cells per rank, cell-proportional categories
    uint64_t fixed_bytes = 0;      // the rest (memory::scales_with_cells)
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

    // Resolution against the physical scales (check_resolution): every estimate in order, and the main ones by
    // name, 0 where not estimated
    std::vector<CheckEstimate> resolution;
    double delta_min = 0.0, delta_max = 0.0;  // cell size V^(1/d)
    double epsilon = 0.0;                     // dissipation per unit mass
    double eta = 0.0;                         // Kolmogorov length (nu^3 / epsilon)^(1/4)
    double delta_over_eta = 0.0;              // volume-weighted mean cell size over eta
    double taylor = 0.0;                      // Taylor microscale (15 nu u'^2 / epsilon)^(1/2)
    double u_tau = 0.0;                       // friction velocity, root mean square over the walls
    double y_plus_max = 0.0, y_plus_mean = 0.0;  // first cell centers, resolved walls (mean: all walls)
    double x_plus_max = 0.0, z_plus_max = 0.0;   // wall cells along the flow and across it
    double y_plus_exchange_min = 0.0;            // the wall model's lowest exchange point
    double cells_per_delta_L_min = 0.0, cells_per_delta_L_max = 0.0;
    double tfles_thickening_max = 0.0;           // TFLES' largest F = n_res Delta / delta_L

    void error(const std::string & code, const std::string & message);
    void warning(const std::string & code, const std::string & message);
    /** @brief The messages of the errors (of the warnings), in order. */
    std::vector<std::string> errors() const;
    std::vector<std::string> warnings() const;
    /** @brief Error if there is an error, else warning if there is a warning, else ok. */
    report::Status status() const;
    bool ok() const { return status() != report::Status::ERROR; }
};

/**
 * @brief Set options.memory from --memory: a size with its unit (40GB, 512MiB) or a GPU of the table
 *        (a100-40, a100-80, h100-80, h200, mi250x, mi300x). Returns an error, or empty.
 */
std::string set_memory_target(const std::string & text, CheckOptions & options);

class Mesh;
namespace region {
struct MeshRegions;
}

/** @brief The case view of a mesh alone, and the check's entries of its own regions (the domain's first). */
struct MeshView {
    std::vector<case_view::Block> blocks;  // the domain's cells, the boundary zones, cell zones and periodic pairs
    std::vector<CheckRegion> regions;
    std::array<std::array<double, 3>, 2> domain{};  // its box
};

/** @brief The case view of the mesh and of its regions (automatic); with outlines, the cell zones' boundaries
 *         for the images. Collective. */
MeshView mesh_view(const Mesh & mesh, const region::MeshRegions & automatic, bool outlines = true);

/**
 * @brief The peak of Kokkos allocations predicted for a rank of local_cells owned and halo cells: the largest
 *        of the peaks of the setup's phases and the steady state, each proportional to the cells but for the categories
 *        that are not (memory::scales_with_cells); for a GPU from a host check, without the output staging.
 */
double predicted_peak(const CheckReport & report, double local_cells);

/**
 * @brief The report as text: the result, a section per part of the check whose rows each have a status, the
 *        diagnostics with their codes, and the result again.
 * @param color ANSI styles
 */
std::string check_text(const CheckReport & report, bool color);

/** @brief The report as one JSON object (docs/guide/input_files.md#the-report-as-json). */
std::string check_json(const CheckReport & report);

/**
 * @brief The parts of the report a run records at setup (docs/guide/output.md#the-check-at-setup): status,
 *        diagnostics, mesh_quality and resolution as check_json writes them, and the time step's integrator,
 *        stability limit and CFL number.
 */
json::Object check_summary(const CheckReport & report);

/** @brief Prints the report on rank 0: as text, colored on terminals, or as JSON. */
void print_check(const CheckReport & report, bool json = false);

#endif // CHECK_H

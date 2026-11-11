/**
 * @file diagnostic_code.h
 * @brief The codes of every warning and error Mallard reports, from mallard check and from runs: stable, errors
 *        "E-" and warnings "W-", each explained in the docs at the anchor docs() gives.
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 */

#ifndef DIAGNOSTIC_CODE_H
#define DIAGNOSTIC_CODE_H

#include <array>
#include <stdexcept>
#include <string>
#include <string_view>

namespace diagnostic_code {

// Setup, from mallard check and runs alike
inline constexpr const char * INPUT = "E-INPUT";  // the setup stopped on an input error
inline constexpr const char * SETUP = "E-SETUP";  // the setup stopped on another error
inline constexpr const char * KEY_UNUSED = "E-KEY-UNUSED";
inline constexpr const char * KEY_DEPRECATED = "W-KEY-DEPRECATED";
inline constexpr const char * CATALOG_WRITE = "E-CATALOG-WRITE";
inline constexpr const char * MESH_FIELDS = "E-MESH-FIELDS";      // --mesh-fields or its stencils not written
inline constexpr const char * RANKS = "E-RANKS";                  // --ranks under mpirun on other ranks
inline constexpr const char * MEMORY_DEVICE = "E-MEMORY-DEVICE";  // the peak exceeds the device's memory
inline constexpr const char * REGION_EMPTY = "E-REGION-EMPTY";
inline constexpr const char * REGION_OUTSIDE = "E-REGION-OUTSIDE";
inline constexpr const char * QUALITY_NON_ORTHOGONALITY = "W-QUALITY-NON-ORTHOGONALITY";
inline constexpr const char * QUALITY_SKEWNESS = "W-QUALITY-SKEWNESS";
inline constexpr const char * QUALITY_ASPECT_RATIO = "W-QUALITY-ASPECT-RATIO";
inline constexpr const char * QUALITY_VOLUME_RATIO = "W-QUALITY-VOLUME-RATIO";
inline constexpr const char * CFL_LIMIT = "W-CFL-LIMIT";
inline constexpr const char * TFLES_THICKENING = "W-TFLES-THICKENING";
inline constexpr const char * WALL_Y_PLUS = "W-WALL-Y-PLUS";
inline constexpr const char * WALL_SPACING = "W-WALL-SPACING";
inline constexpr const char * WALL_MODEL_EXCHANGE = "W-WALL-MODEL-EXCHANGE";
inline constexpr const char * DNS_ETA = "W-DNS-ETA";
inline constexpr const char * SIZING_FIXED = "W-SIZING-FIXED";  // the memory that does not scale with the cells
inline constexpr const char * SIZING_PARTS = "W-SIZING-PARTS";  // no partition fits
inline constexpr const char * ENV_GPU_AWARE_MPI = "E-ENV-GPU-AWARE-MPI";  // built for it, the MPI library has none
inline constexpr const char * ENV_HOST_STAGING = "W-ENV-HOST-STAGING";    // device halos go through host memory
inline constexpr const char * ENV_PLACEMENT = "W-ENV-PLACEMENT";          // binding::check
inline constexpr const char * ENV_THREAD_BINDING = "W-ENV-THREAD-BINDING";  // OpenMP threads free to migrate

// Runs
inline constexpr const char * RUN = "E-RUN";            // the run stopped on an error after its setup
inline constexpr const char * DIVERGED = "E-DIVERGED";  // the divergence report
inline constexpr const char * HEALTH_MARGIN = "W-HEALTH-MARGIN";
inline constexpr const char * RESTART_SKIPPED = "W-RESTART-SKIPPED";  // --restart latest passed over a file
inline constexpr const char * MESH_DIFFERS = "W-MESH-DIFFERS";        // a file written on a mesh of other contents
inline constexpr const char * SNAPSHOT_INDEX = "W-SNAPSHOT-INDEX";
inline constexpr const char * FORMAT_VTU = "W-FORMAT-VTU";
inline constexpr const char * KEY_IGNORED = "W-KEY-IGNORED";
inline constexpr const char * SPECTRA_RESTARTED = "W-SPECTRA-RESTARTED";
inline constexpr const char * CUDA_GRAPH = "W-CUDA-GRAPH";
inline constexpr const char * TENO_CACHE = "W-TENO-CACHE";
inline constexpr const char * QUADRATURE_WEIGHTS = "W-QUADRATURE-WEIGHTS";
inline constexpr const char * SIGNAL_HANDLERS = "W-SIGNAL-HANDLERS";
inline constexpr const char * CONTROL_SOCKET = "W-CONTROL-SOCKET";
inline constexpr const char * RUN_REPORT = "W-RUN-REPORT";
inline constexpr const char * EXTRACT_AVERAGED = "W-EXTRACT-AVERAGED";
inline constexpr const char * PACK_RERUN = "W-PACK-RERUN";

/** @brief A code and the page that explains it, at the anchor of its lowercase name. */
struct Entry {
    const char * code;
    const char * page;
};

inline constexpr const char * CHECK_PAGE = "docs/guide/input_files.md";
inline constexpr const char * RUN_PAGE = "docs/guide/running.md";

/** @brief Every code. */
inline constexpr std::array<Entry, 43> ALL = {{
    {INPUT, CHECK_PAGE}, {SETUP, CHECK_PAGE}, {KEY_UNUSED, CHECK_PAGE}, {KEY_DEPRECATED, CHECK_PAGE},
    {CATALOG_WRITE, CHECK_PAGE}, {MESH_FIELDS, CHECK_PAGE}, {RANKS, CHECK_PAGE}, {MEMORY_DEVICE, CHECK_PAGE}, {REGION_EMPTY, CHECK_PAGE},
    {REGION_OUTSIDE, CHECK_PAGE}, {QUALITY_NON_ORTHOGONALITY, CHECK_PAGE}, {QUALITY_SKEWNESS, CHECK_PAGE},
    {QUALITY_ASPECT_RATIO, CHECK_PAGE}, {QUALITY_VOLUME_RATIO, CHECK_PAGE}, {CFL_LIMIT, CHECK_PAGE},
    {TFLES_THICKENING, CHECK_PAGE}, {WALL_Y_PLUS, CHECK_PAGE}, {WALL_SPACING, CHECK_PAGE},
    {WALL_MODEL_EXCHANGE, CHECK_PAGE}, {DNS_ETA, CHECK_PAGE}, {SIZING_FIXED, CHECK_PAGE}, {SIZING_PARTS, CHECK_PAGE},
    {ENV_GPU_AWARE_MPI, CHECK_PAGE}, {ENV_HOST_STAGING, CHECK_PAGE}, {ENV_PLACEMENT, CHECK_PAGE},
    {ENV_THREAD_BINDING, CHECK_PAGE},
    {RUN, RUN_PAGE}, {DIVERGED, RUN_PAGE}, {HEALTH_MARGIN, RUN_PAGE}, {RESTART_SKIPPED, RUN_PAGE},
    {MESH_DIFFERS, RUN_PAGE}, {SNAPSHOT_INDEX, RUN_PAGE}, {FORMAT_VTU, RUN_PAGE}, {KEY_IGNORED, RUN_PAGE},
    {SPECTRA_RESTARTED, RUN_PAGE}, {CUDA_GRAPH, RUN_PAGE}, {TENO_CACHE, RUN_PAGE}, {QUADRATURE_WEIGHTS, RUN_PAGE},
    {SIGNAL_HANDLERS, RUN_PAGE}, {CONTROL_SOCKET, RUN_PAGE}, {RUN_REPORT, RUN_PAGE}, {EXTRACT_AVERAGED, RUN_PAGE},
    {PACK_RERUN, RUN_PAGE},
}};

/** @brief The warning of each mesh-quality metric, in MeshQuality's order. */
inline constexpr std::array<const char *, 4> QUALITY = {QUALITY_NON_ORTHOGONALITY, QUALITY_SKEWNESS,
                                                        QUALITY_ASPECT_RATIO, QUALITY_VOLUME_RATIO};

/**
 * @brief The page and anchor that explain a code: "docs/guide/input_files.md#w-cfl-limit".
 * @throws std::logic_error for a code not in ALL
 */
std::string docs(std::string_view code);

/** @brief An error that stops a run with its code (one of ALL). */
struct Error : std::runtime_error {
    Error(const char * code_in, const std::string & message) : std::runtime_error(message), code(code_in) {}
    const char * code;
};

} // namespace diagnostic_code

#endif // DIAGNOSTIC_CODE_H

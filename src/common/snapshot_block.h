/**
 * @file snapshot_block.h
 * @brief The snapshot block: which snapshot a file (or a step of a series
 *        file) holds, as one JSON object next to the run block
 *        (provenance.h). The run block is the same in every file of a run;
 *        the snapshot block is this snapshot's: step, time, why and when it
 *        was written, by which run, and its fields (units, long name, stored
 *        precision). Format and placement: docs/guide/output.md#snapshot-metadata.
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 */

#ifndef SNAPSHOT_BLOCK_H
#define SNAPSHOT_BLOCK_H

#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>
#include <string_view>
#include <vector>

namespace snapshot_block {

inline constexpr const char * SCHEMA = "mallard-snapshot";
constexpr int VERSION = 1;

/** @brief The comment name in XML and CSV files ("<!-- mallard-snapshot {...} -->"). */
inline constexpr const char * NAME = "mallard-snapshot";

/** @brief The HDF5 dataset (root group) of fixed-length strings holding one block per snapshot of the file. */
inline constexpr const char * DATASET = "mallard_snapshot";

/** @brief Why a snapshot was written: its interval, the run's start or end, a run-control request, a divergence, a trigger. */
namespace reason {
inline constexpr const char * INTERVAL = "interval";
inline constexpr const char * START = "start";
inline constexpr const char * END = "end";
inline constexpr const char * STOP = "stop";
inline constexpr const char * CHECKPOINT = "checkpoint";
inline constexpr const char * DIVERGENCE = "divergence";
inline constexpr const char * TRIGGER = "trigger";
} // namespace reason

/**
 * @brief What a field's values are: the cell average of the quantity (the
 *        solver's unknowns), computed pointwise from the cell's averaged
 *        state (neither its cell average nor its centroid value at high
 *        order), or the reconstruction's value at a point.
 */
namespace value {
inline constexpr const char * CELL_AVERAGE = "cell_average";
inline constexpr const char * CELL_STATE = "cell_state";
inline constexpr const char * POINT_VALUE = "point_value";
} // namespace value

/** @brief A field of a snapshot. */
struct Field {
    std::string name;
    std::string units;      ///< UDUNITS; "1" for dimensionless
    std::string long_name;
    std::string stored;     ///< "float64", "float32"
    double relative_error = 0.0;  ///< of lossy compression; 0: stored exactly
    std::string value;      ///< value::...; make() fills it from the name where empty
};

/** @brief How point values were evaluated: method, design order, and what replaces it where it cannot be used. */
struct Reconstruction {
    std::string method;                 ///< "TENO", "linear_least_squares"; empty: no point values
    int order = 0;
    std::vector<std::string> fallback;  ///< in order: "linear_least_squares", "cell_average"
};

/** @brief The block of one snapshot. */
struct Block {
    int64_t step = -1;
    double time = std::numeric_limits<double>::quiet_NaN();  ///< s
    std::string run_id;
    bool restarted = false;           ///< whether the run restarted from a file (restart_of)
    std::string restart_of_run_id;    ///< empty where that file recorded none
    std::string restart_of_file;
    int64_t restart_of_step = -1;
    std::string written_at;           ///< UTC, ISO 8601
    std::string reason;
    std::string format;               ///< "vtkhdf", "vtu", "xdmf", "restart", "vtp", "spectra"
    int64_t format_version = -1;
    std::string solver_precision;     ///< "double", "single"
    bool averaged = false;            ///< whether it holds time averages (MEAN_, COV_, PHASE_, spectra)
    double average_start = std::numeric_limits<double>::quiet_NaN();  ///< s
    double average_end = std::numeric_limits<double>::quiet_NaN();    ///< s
    int64_t samples = -1;
    Reconstruction reconstruction;    ///< of the point values among the fields
    std::vector<Field> fields;

    /** @brief The block as one line of JSON. */
    std::string json() const;
    const Field * field(const std::string & name) const;
};

/** @brief A block read back from JSON. @throws std::runtime_error if it is not a snapshot block */
Block parse(std::string_view json);

/** @brief The averaging window of the run's [statistics], for snapshots of averaged fields. */
void set_averages(double start, double end, int64_t samples);

/** @brief Why snapshots forced now are written, while it lives (written for their interval otherwise). */
class Reason {
    public:
        explicit Reason(std::string reason);
        ~Reason();
        Reason(const Reason &) = delete;
        Reason & operator=(const Reason &) = delete;

    private:
        std::string previous;
};

/** @brief The reason of a snapshot written now: its interval, or if forced the current Reason. */
std::string current_reason(bool forced);

/** @brief The value kind of a cell field: cell_average for RHO, RHOU, RHOE, RHOY_*, their means and phase means and
 *         REGION_FRACTION; cell_state for every other. */
std::string value_kind(const std::string & name);

/** @brief The field's CF cell_methods attribute ("volume: mean"), or empty where no CF method fits (cell_state). */
std::string cell_methods(const Field & field);

/** @brief Whether a field holds time averages. */
bool is_averaged(const std::string & name);

/**
 * @brief The block of a snapshot this run writes now, at (step, t), of these
 *        fields: the run's id and restart_of, the time now, and the averaging
 *        window if a field is averaged.
 */
Block make(uint64_t step, double t, const std::string & reason, const std::string & format, int64_t format_version,
           std::vector<Field> fields);

/** @brief The time now, UTC, ISO 8601 ("2026-11-05T18:20:41Z"). */
std::string now_utc();

/** @brief The width of DATASET's strings for a block of this length. */
size_t dataset_width(size_t length);

} // namespace snapshot_block

#endif // SNAPSHOT_BLOCK_H

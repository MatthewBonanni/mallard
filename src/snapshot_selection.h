/**
 * @file snapshot_selection.h
 * @brief The snapshots of output files that the post-processing commands' --steps, --times,
 *        --every and --latest choose, by the steps and times the files record.
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 */

#ifndef SNAPSHOT_SELECTION_H
#define SNAPSHOT_SELECTION_H

#include <cstdint>
#include <string>
#include <vector>

#include "cli.h"
#include "output_reader.h"

namespace snapshots {

/** @brief One snapshot chosen: of which file, its index there, and its step as the file records it
 *         (its position only in files that record none, written before 0.10). */
struct Selected {
    size_t file = 0;
    size_t index = 0;  ///< in the file's snapshots
    output::Snapshot snapshot;
    int64_t step = 0;
    bool step_recorded = false;
};

/** @brief The items of a comma-separated list, trimmed, empty ones dropped. */
std::vector<std::string> split_list(const std::string & text);

/** @brief A finite number. @throws cli::UsageError naming the option otherwise */
double parse_number(const std::string & text, const std::string & option);

/** @brief A non-negative integer. @throws cli::UsageError naming the option otherwise */
uint64_t parse_count(const std::string & text, const std::string & option);

/**
 * @brief The snapshots of the files, in order, then those --steps, --times, --every and --latest choose.
 * @throws cli::UsageError for a malformed option, std::runtime_error if none is left
 */
std::vector<Selected> select(const std::vector<output::Contents> & files, const cli::Args & args);

} // namespace snapshots

#endif // SNAPSHOT_SELECTION_H

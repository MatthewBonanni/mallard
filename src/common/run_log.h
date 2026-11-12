/**
 * @file run_log.h
 * @brief Machine-readable run log: one JSON object per line (JSON Lines),
 *        written by rank 0 next to the human log. docs/guide/output.md
 *        documents the records.
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 */

#ifndef RUN_LOG_H
#define RUN_LOG_H

#include <cstdint>
#include <optional>
#include <string>

#include "json.h"

namespace run_log {

/** @brief Name and version of the record format, in each start record. */
inline constexpr const char * SCHEMA = "mallard-run-log";
constexpr int VERSION = 2;

/**
 * @brief Opens the log on rank 0 (other ranks ignore the log): appended to
 *        when resuming a run, else started afresh. A report (report.html) is
 *        built from the records, from those of the log appended to on, and
 *        written at each restart file, at the first progress record and at
 *        the end. Either file may be empty: none.
 */
void open(const std::string & file, bool append, const std::string & report = "");
void close();
bool is_open();

/** @brief Rank 0: the step of the last record of the log appended to, if it has one. */
std::optional<uint64_t> last_step();

/** @brief Writes a record as one line and flushes it, and adds it to the report; nothing if neither is open. */
void write(const json::Object & record);

} // namespace run_log

#endif // RUN_LOG_H

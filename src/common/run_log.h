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
 *        when resuming a run, else started afresh.
 */
void open(const std::string & file, bool append);
void close();
bool is_open();

/** @brief Rank 0: the step of the last record of the log appended to, if it has one. */
std::optional<uint64_t> last_step();

/**
 * @brief Writes a record as one line and flushes it; nothing if the log is not
 *        open. A held record (a warning) waits while hold() is in effect.
 */
void write(const json::Object & record, bool held = false);

/** @brief Keeps the held records written from now on (a run's setup warnings) until release(). */
void hold();
/** @brief Stops holding: writes the run's start record, then the held records, if the log is open. */
void release(const json::Object & start);

} // namespace run_log

#endif // RUN_LOG_H

/**
 * @file run_endpoint.h
 * @brief The endpoint record of a run, mallard.run.json: where and how its control requests reach it (host, PID,
 *        socket, scheduler job), written by rank 0 at start and at each change of state, and copied to the run
 *        registry so that `mallard ctl RUN_ID` finds the run (docs/guide/running.md#the-endpoint-record).
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 */

#ifndef RUN_ENDPOINT_H
#define RUN_ENDPOINT_H

#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>

#include "interfaces.h"
#include "json.h"

namespace run_endpoint {

inline constexpr const char * SCHEMA = interfaces::RUN_ENDPOINT.name;
inline constexpr int VERSION = interfaces::RUN_ENDPOINT.version;
inline constexpr const char * FILE_NAME = "mallard.run.json";

/** @brief What the run knows of itself at a change of state. */
struct State {
    std::string state;                // setup, running, ended
    std::string outcome;              // when ended: as the run log's end record
    bool socket = true;               // whether the control socket listens (once running)
    uint64_t control_file_interval = 0;
    std::string run_log;              // the run log's path, relative to the run directory, or empty
};

/**
 * @brief Rank 0 of a run: writes the record of the run run_id in the working directory, atomically, and its copy
 *        in the registry. The first call fixes the host, PID, scheduler and start time. Failures are warnings.
 */
void write(const std::string & run_id, int ranks, const State & state);

/** @brief Whether this process wrote a record that is not yet ended. */
bool open();

/** @brief Marks the record of this process ended, with the outcome, if it is not already. */
void end(const std::string & outcome);

/** @brief The registry's directory: $MALLARD_RUN_REGISTRY, else $XDG_STATE_HOME/mallard/runs, else
 *         ~/.local/state/mallard/runs; nothing if MALLARD_RUN_REGISTRY is "off". */
std::optional<std::filesystem::path> registry();

/** @brief The scheduler of this process, from its environment: {"kind": "slurm", "job_id": ...} etc. */
std::string scheduler_json();

/** @brief Reads a record: from DIR/mallard.run.json if run is a directory, else the registry's entry of the run ID.
 *         @throws std::runtime_error saying where it looked */
json::Value find(const std::string & run);

} // namespace run_endpoint

#endif // RUN_ENDPOINT_H

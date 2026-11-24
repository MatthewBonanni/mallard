/**
 * @file restart_changes.h
 * @brief What a run starting from a restart file does with each key of its input
 *        that differs from the input the file was written with: accepts it,
 *        resets some state the file holds, or cannot start from the file at all.
 *
 * Only the solver knows which keys carry state across a restart, so mallard
 * check --restart reports this (check_report.cpp) for front ends to show before
 * a launch. The rules follow what Solver::apply_restart and the restore functions
 * it calls do with the file.
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 */

#ifndef RESTART_CHANGES_H
#define RESTART_CHANGES_H

#include <cstdint>
#include <string>
#include <vector>

#include "input.h"

namespace restart_changes {

enum class Effect { ACCEPTED, RESETS, INCOMPATIBLE };

/** @brief "accepted", "resets" or "incompatible". */
const char * name(Effect effect);

/** @brief What a restart file holds, from its header and run block, without its fields. */
struct Restart {
    std::string file;
    uint64_t step = 0;
    double t = 0.0;
    std::string run_id;                   // of the run that wrote it, or empty
    int dim = 0;                          // that run's dimension, 0 if not recorded
    uint32_t real_size = 0;               // bytes per value
    std::vector<std::string> variables;   // of its cells
    std::vector<std::string> attributes;  // names of its scalar state
    std::vector<std::string> arrays;      // names of its array state
    uint64_t faces = 0;                   // characteristic boundary faces with a state
    std::string resolved;                 // that run's resolved input, empty if not recorded
};

/**
 * @brief Reads a restart file's header and run block.
 * @throws std::runtime_error for a file that is not a readable restart file
 */
Restart read(const std::string & file);

/** @brief One key that differs, its values (TOML; empty where not set) and what a restart does with it. */
struct Change {
    std::string key;            // with entries named: boundaries[name = "inlet"].type
    std::string restart_value;  // in the file's input
    std::string case_value;     // in the case's
    Effect effect = Effect::ACCEPTED;
    std::string detail;         // what is reset, or why the file cannot be used; for accepted, a note or empty
};

/**
 * @brief Each key that differs between the file's resolved input and the case's (leaving out [initialize],
 *        which a restart replaces), and the build's dimension and precision where they differ from the file's.
 * @param case_variables the variables a run of the case reads from a restart file (Solver::restart_variables),
 *                       empty if its setup did not get that far
 * @param case_complete whether case_resolved is the whole resolved input; if not (a setup that stopped), keys it
 *                      lacks are left out rather than reported as unset
 */
std::vector<Change> classify(const Restart & restart, const Input & case_resolved,
                             const std::vector<std::string> & case_variables, bool case_complete = true);

}  // namespace restart_changes

#endif  // RESTART_CHANGES_H

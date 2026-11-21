/**
 * @file interfaces.h
 * @brief Schema names and versions of the machine-readable interfaces whose
 *        producers have no header of their own for them; `mallard version
 *        --format json` lists these with the run log's, the run block's and
 *        the snapshot block's.
 *
 * Additions (fields, record types) keep the version; a field removed or
 * changed in meaning increments it.
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 */

#ifndef INTERFACES_H
#define INTERFACES_H

namespace interfaces {

/** @brief The schema field and integer version field of an interface's JSON. */
struct Schema {
    const char * name;
    int version;
};

inline constexpr Schema CAPABILITIES = {"mallard-capabilities", 1};   // mallard version --format json
inline constexpr Schema KEY_CATALOG = {"mallard-key-catalog", 1};     // mallard list keys --format json
inline constexpr Schema CHECK_REPORT = {"mallard-check", 1};          // mallard check --format json
inline constexpr Schema BOUNDARY_VALUES = {"mallard-boundary-values", 1}; // mallard check --boundary-values DIR
inline constexpr Schema CASE_VIEW = {"mallard-case-view", 1};         // mallard check --view DIR --format json's view
inline constexpr Schema RUN_DIRECTORY = {"mallard-run-directory", 1}; // mallard info RUN_DIR --format json
inline constexpr Schema RUN_CONTROL = {"mallard-run-control", 2};     // mallard ctl, the control socket
inline constexpr Schema RUN_ENDPOINT = {"mallard-run-endpoint", 1};   // mallard.run.json, the run registry
inline constexpr Schema RUN_LAUNCH = {"mallard-run-launch", 1};       // mallard -i INPUT --detach
inline constexpr Schema CONVERGENCE = {"mallard-convergence", 1};     // mallard post convergence --format json
inline constexpr Schema PACK = {"mallard-pack", 1};                   // manifest.json of mallard pack

} // namespace interfaces

#endif // INTERFACES_H

/**
 * @file commands.h
 * @brief Subcommands of the mallard executable that live outside the solver library.
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 */

#ifndef COMMANDS_H
#define COMMANDS_H

#include "cli.h"

#include <string>
#include <vector>

#include "check.h"
#include "json.h"

/** @brief `mallard -i input.toml`: runs a case. */
int run_main(const cli::Args & args);

/** @brief `mallard check input.toml`: sets up a case and reports on it. */
int check_main(const cli::Args & args);

/**
 * @brief The Environment of `mallard check`: the build, every rank's placement and device, MPI, GPU-aware
 *        MPI and NCCL (as the case's resolved input chooses, if any), with what to fix as the report's
 *        diagnostics on rank 0. Collective; after Kokkos is initialized.
 */
void check_environment(CheckReport & report);

/** @brief `mallard input resolve input.toml`: writes the input as setup uses it. */
int resolve_main(const cli::Args & args);

/** @brief `mallard reactor -i input.toml`: a 0D reactor or chemistry benchmark. */
int reactor_main(const cli::Args & args);

/** @brief `mallard info FILE|RUN_DIR`: what a file or a run directory holds. */
int info_main(const cli::Args & args);

/**
 * @brief The JSON of mallard info FILE --format json and of the session's file.info, for an output or restart
 *        file: what it holds (mallard-file-info), and for a restart whether this build continues from it.
 * @throws std::runtime_error for a file that cannot be read
 */
std::string file_info_json(const std::string & path);

/** @brief `mallard list fields [INPUT]`: the output fields there are, or those of a case. */
int list_fields_main(const cli::Args & args);

/** @brief `mallard list species INPUT`: the species of a case. */
int list_species_main(const cli::Args & args);

/** @brief `mallard list units`: the units quantities in input files take. */
int list_units_main(const cli::Args & args);

/** @brief `mallard files INPUT`: the files a case reads, for staging it elsewhere. */
int files_main(const cli::Args & args);

/** @brief `mallard list keys [SECTION] [INPUT]`: the input keys there are, or those of a case. */
int list_keys_main(const cli::Args & args);

/**
 * @brief The key catalog of every example and tutorial (docs/input-keys/, as tools/dev/input_catalog.py merges
 *        it), from Mallard's data directories. @throws std::runtime_error if none has it
 */
json::Value general_key_catalog();

/**
 * @brief A case's catalog with the type and form of each table it has no key of (an optional table not set, as
 *        boundaries.turbulence) from the general catalog, if a data directory has it.
 */
void add_unset_table_forms(json::Value & catalog);

/** @brief A case's side of a key listing: whether its setup read every key, and what stopped it. */
struct KeyCase {
    std::string input;
    bool complete = false;
    std::vector<CheckDiagnostic> diagnostics;
};

/**
 * @brief The catalog's keys in section (a table or key path; empty: all), in order.
 * @throws std::invalid_argument if there is none (and per_case, if given, completed its setup)
 */
std::vector<std::string> keys_in_section(const json::Value & catalog, const std::string & section,
                                         const KeyCase * per_case);

/**
 * @brief The JSON of mallard list keys --format json and of case.keys and keys.catalog: the keys with their
 *        catalog entries; with a case, complete, missing (required keys not set) and its diagnostics.
 */
std::string key_listing_json(const json::Value & catalog, const std::vector<std::string> & keys,
                             const std::string & section, const KeyCase * per_case);

/** @brief SECTION of mallard list keys without brackets: "numerics" of "[numerics]". */
std::string section_argument(std::string section);

/**
 * @brief The options of mallard check (and of the session's case.check, whose parameters are named as them) into
 *        options. @return a usage error, or empty
 */
std::string check_options(const cli::Args & args, CheckOptions & options);

/** @brief What this build supports (mallard-capabilities): `mallard version --format json`, the session's initialize. */
std::string capabilities_json();

/** @brief `mallard session [INPUT]`: answers JSON-RPC requests on stdin and stdout (session.h). */
int session_main(const cli::Args & args);

/** @brief `mallard list templates`: the case templates of mallard new. */
int list_templates_main(const cli::Args & args);

/** @brief `mallard diff A B`: how two solutions differ, field by field. */
int diff_main(const cli::Args & args);

/** @brief `mallard post extract FILE...`: probes, regions and snapshots of existing output. */
int extract_main(const cli::Args & args);

/** @brief The work of `mallard post extract`, with MPI and Kokkos started (tests call it). */
int post_extract(const cli::Args & args);

/** @brief `mallard post derive FILE...`: snapshots of derived fields of existing output. */
int derive_main(const cli::Args & args);

/** @brief The work of `mallard post derive`, with MPI and Kokkos started (tests call it). */
int post_derive(const cli::Args & args);
/** @brief `mallard post convergence RUN...`: observed order, Richardson extrapolation and GCI. */
int convergence_main(const cli::Args & args);

/** @brief The work of `mallard post convergence`, with Kokkos started (tests call it). */
int post_convergence(const cli::Args & args);

/** @brief `mallard post average FILE...`: statistics over saved snapshots. */
int average_main(const cli::Args & args);

/** @brief The work of `mallard post average`, with MPI and Kokkos started (tests call it). */
int post_average(const cli::Args & args);

/** @brief `mallard post flame FILE...`: position, thickness and speeds of a premixed flame in saved snapshots. */
int flame_main(const cli::Args & args);

/** @brief The work of `mallard post flame`, with MPI and Kokkos started (tests call it). */
int post_flame(const cli::Args & args);

/** @brief `mallard pack RUN_DIR`: one archive that reproduces a run; `--verify ARCHIVE` checks one. */
int pack_main(const cli::Args & args);

/** @brief `mallard mesh convert INPUT OUTPUT.h5`. */
int mesh_convert_main(const cli::Args & args);

#endif // COMMANDS_H

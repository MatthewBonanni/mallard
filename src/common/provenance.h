/**
 * @file provenance.h
 * @brief The run block (provenance): the code and build, the input, the
 *        mesh, the resolved input, and where, when and as which run it ran
 *        (its id, and the run it restarted from), as one JSON object that every
 *        output file carries. It describes the run, the same in every file of
 *        it; what describes one snapshot is its snapshot block (snapshot_block.h).
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 */

#ifndef PROVENANCE_H
#define PROVENANCE_H

#include <string>
#include <string_view>
#include <tuple>
#include <vector>


namespace provenance {

/** @brief Name and version of the block's format; version 2 and before were named LEGACY_SCHEMA. */
inline constexpr const char * SCHEMA = "mallard-run";
inline constexpr const char * LEGACY_SCHEMA = "mallard-provenance";
constexpr int VERSION = 3;

/** @brief The root attribute of HDF5 files holding the block; LEGACY_ATTRIBUTE before version 3. */
inline constexpr const char * ATTRIBUTE = "mallard_run";
inline constexpr const char * LEGACY_ATTRIBUTE = "provenance";

/** @brief SHA-256 of data, in hexadecimal. */
std::string sha256(std::string_view data);

/** @brief SHA-256 of a file's bytes, or of a directory's files (path and bytes, by path); "" if unreadable. */
std::string sha256_file(const std::string & path);

/** @brief Starts the block of a new run: a new run id (a random UUID), start time, rank count, host and directory. */
void start_run();

/** @brief The run's id (rank 0's once shared). */
const std::string & run_id();

/**
 * @brief The run this one restarted from: that run's id (empty where its
 *        file records none: written before Mallard 0.10), the restart file
 *        and the step it continues from.
 */
void set_restart_of(const std::string & run_id, const std::string & file, uint64_t step);

/** @brief The block's restart_of: an object, or "null" for a run that did not restart. */
const std::string & restart_of();

/**
 * @brief Where every rank runs (collective, after start_run): binding::Placement of
 *        each rank. The block's run gains the host count; placement() holds the list.
 */
void set_placement();

/** @brief The placement of every rank as a JSON array ("[]" before set_placement). */
const std::string & placement();

/** @brief The input file (empty if given otherwise) and its text. */
void set_input(const std::string & file, const std::string & text);

/** @brief The mesh: its type (e.g. "file", "cartesian"), file and file hash (empty if none). */
void set_mesh(const std::string & type, const std::string & file, const std::string & hash);

/** @brief The input with every key explicit, as setup used it (ResolvedInput::toml). */
void set_resolved(const std::string & toml);

/** @brief The numerics preset (empty name: none) and each key it covers: value, and whether the preset set it. */
void set_preset(const std::string & name, const std::vector<std::tuple<std::string, std::string, bool>> & settings);

/** @brief Every rank takes rank 0's block and run id (collective). */
void share();

/** @brief The block: the build alone until start_run. */
const std::string & json();

/** @brief The block as a comment line of a CSV file ("# mallard-run {...}"), without the newline. */
std::string csv_comment();

/** @brief The block as an XML comment ("<!-- mallard-run {...} -->", no "--" inside it). */
std::string xml_comment();

/**
 * @brief JSON text as an XML comment "<!-- name json -->" or a CSV comment
 *        line "# name json"; parse_comment(text, name) reads either back.
 */
std::string xml_comment(std::string_view name, std::string_view json);
std::string csv_comment(std::string_view name, std::string_view json);

/**
 * @brief The block of an XML comment written by xml_comment, or of a CSV
 *        comment line, or "" if text holds neither; blocks of version 2 and
 *        before, named "provenance", too.
 */
std::string parse_comment(std::string_view text);

/** @brief The JSON of the first comment "name" (XML or CSV) of text, or "". */
std::string parse_comment(std::string_view text, std::string_view name);

/** @brief The JSON of every CSV comment line "# name ..." of text, in order. */
std::vector<std::string> parse_comments(std::string_view text, std::string_view name);

/**
 * @brief The string value of key in a top-level object of a block (e.g.
 *        "mesh", "sha256"), as written (escapes kept), or "" if there is none.
 */
std::string find_string(std::string_view block, std::string_view object, std::string_view key);

} // namespace provenance

#endif // PROVENANCE_H

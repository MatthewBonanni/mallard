/**
 * @file provenance.h
 * @brief Provenance of a run: the code and build, the input, the mesh, the
 *        resolved setup and where and when it ran, as one JSON object that
 *        every output file carries.
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 */

#ifndef PROVENANCE_H
#define PROVENANCE_H

#include <string>
#include <string_view>

#include "log.h"

namespace provenance {

/** @brief Name and version of the block's format. */
inline constexpr const char * SCHEMA = "mallard-provenance";
constexpr int VERSION = 1;

/** @brief SHA-256 of data, in hexadecimal. */
std::string sha256(std::string_view data);

/** @brief SHA-256 of a file's bytes, or of a directory's files (path and bytes, by path); "" if unreadable. */
std::string sha256_file(const std::string & path);

/** @brief Starts the block of a new run: start time, rank count, host and directory. */
void start_run();

/** @brief The input file (empty if given otherwise) and its text. */
void set_input(const std::string & file, const std::string & text);

/** @brief The mesh: its type (e.g. "file", "cartesian"), file and file hash (empty if none). */
void set_mesh(const std::string & type, const std::string & file, const std::string & hash);

/** @brief The resolved setup, as printed to the human log. */
void set_setup(const logging::Record & setup);

/** @brief Every rank takes rank 0's block (collective). */
void share();

/** @brief The block: the build alone until start_run. */
const std::string & json();

/** @brief The block as a comment line of a CSV file, without the newline. */
std::string csv_comment();

/** @brief The block as an XML comment (no "--" inside it). */
std::string xml_comment();

/**
 * @brief The block of an XML comment written by xml_comment, or of a CSV
 *        comment line, or "" if text holds neither.
 */
std::string parse_comment(std::string_view text);

} // namespace provenance

#endif // PROVENANCE_H

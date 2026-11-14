/**
 * @file log.h
 * @brief Console output: rank-0 info lines, aligned sections and tables,
 *        warnings and errors on stderr, optional color on terminals.
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 */

#ifndef LOG_H
#define LOG_H

#include <cstdint>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "common_io.h"

#if defined(__GNUC__) || defined(__clang__)
#define MALLARD_PRINTF(fmt, args) __attribute__((format(printf, fmt, args)))
#else
#define MALLARD_PRINTF(fmt, args)
#endif

namespace logging {

/// Key-value lines of a setup section, in display order.
using Items = std::vector<std::pair<std::string, std::string>>;

/// Width of section rules and the target maximum line width.
constexpr int WIDTH = 92;
/// Column at which the values of key-value lines start.
constexpr int KEY_WIDTH = 18;

enum class Style { BOLD, DIM, RED, YELLOW, GREEN, CYAN };

/**
 * @brief printf-style formatting into a std::string.
 */
std::string format(const char * fmt, ...) MALLARD_PRINTF(1, 2);

/**
 * @brief Wraps text in an ANSI style when stdout gets color, else returns it unchanged.
 */
std::string style(std::string_view text, Style s);
/** @brief As style, with color when on whatever stdout is (text rendered for later, or for a test). */
std::string style(std::string_view text, Style s, bool on);

/**
 * @brief Whether to color stdout (stderr): a terminal, NO_COLOR unset and not under an MPI
 *        launcher, or CLICOLOR_FORCE set.
 */
bool color_stdout();
bool color_stderr();

/**
 * @brief Prints a line on stdout from rank 0.
 */
void line(std::string_view text = "");

/**
 * @brief Section heading, e.g. "== Mesh ===...=== 0.12 s".
 */
void section(std::string_view title, std::string_view right = "");

/**
 * @brief Aligned "  key   value" line under a section.
 */
void item(std::string_view key, std::string_view value);
void items(const Items & list);

/**
 * @brief The cells of a column with its numbers padded so that their decimal points share a column: the point,
 *        or where it would be (after an integer's last digit, before an exponent), at the same offset in every
 *        number, with a slot for a minus sign, and every number as wide. A number may be followed by "%" or by
 *        a space and one word ("27.1 kB"); "-" (no value) aligns as an integer. Other cells are unchanged.
 */
std::vector<std::string> align_decimals(std::vector<std::string> column);

/**
 * @brief Columns under a section: a header row, then one row per entry, each column as wide
 *        as its widest cell, its numbers aligned at the decimal point (align_decimals).
 */
void table(const std::vector<std::string> & header, const std::vector<std::vector<std::string>> & rows);

/**
 * @brief Starts a line naming a setup phase; end_phase() completes it with the elapsed time.
 *        Anything printed in between first ends the open line.
 */
void begin_phase(std::string_view name);
void end_phase(double seconds);

/**
 * @brief An event during the run that is not a file written (a checkpoint or stop requested, a setting
 *        changed, a divergence): a remark line between the progress rows, "  >> step 1200, t = 0.012 s:
 *        stop: STOP file"; also an event record of the run log, with the fields given as (key, JSON text)
        after its own.
 */
void event(uint64_t step, double t, std::string_view kind, std::string_view text,
           const std::vector<std::pair<std::string, std::string>> & fields = {});

/**
 * @brief A file completely written during the run: an output record of the run log, which indexes every
 *        file the run writes (path, kind, step, time, reason, run id, the writer: a [[write_data]] prefix or
 *        a probe's file, and each file's size and SHA-256; a file with no record was not completely
 *        written), counted for the progress table's written column (take_written), and a remark line too
 *        when verbose.
 */
void written(uint64_t step, double t, std::string_view kind, std::string_view path, std::string_view reason,
             std::string_view writer, const std::vector<WrittenFile> & files);
/** @brief written() of the one file at path, written on the main thread: sized and hashed now. */
void written(uint64_t step, double t, std::string_view kind, std::string_view path, std::string_view reason,
             std::string_view writer);

/** @brief The files written since the last call, by kind in the order first written: "16 vtu, 1 restart". */
std::string take_written();
/** @brief Whether files were written since the last take_written. */
bool written_pending();

/** @brief Remark lines for every file written (mallard -i ... --verbose). */
void set_verbose(bool on);

/**
 * @brief Warning on stderr from rank 0, for conditions every rank sees alike: "warning W-CODE: message" and the
 *        docs that explain the code (diagnostic_code::docs); also a record of the run log.
 * @param code one of diagnostic_code::ALL
 */
void warning(std::string_view code, std::string_view message);

/**
 * @brief Error on stderr from the calling rank, as warning, prefixed with the rank in parallel runs; also a record
 *        of the run log (rank 0).
 */
void error(std::string_view code, std::string_view message);

/** @brief The text warning and error print: "warning W-CODE: message" and the docs line, with or without color. */
std::string diagnostic(bool is_error, std::string_view code, std::string_view message, bool color);

/// 1234567 -> "1,234,567".
std::string count(uint64_t n);
/// 1, "file" -> "1 file"; 1234, "file" -> "1,234 files".
std::string count(uint64_t n, std::string_view noun);
/// Wall-clock duration: "850 us" ("850 µs" in a UTF-8 locale), "12.3 ms", "4.56 s", "12m 03s", "2h 05m".
std::string duration(double seconds);
/// Value with an SI suffix and 3 significant digits: "16.3M", "1.20k".
std::string si(double value);
/// Compact real: "%.6g".
std::string real(double value);

} // namespace logging

#endif // LOG_H

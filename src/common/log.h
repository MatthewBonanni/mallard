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
 * @brief Starts a line naming a setup phase; end_phase() completes it with the elapsed time.
 *        Anything printed in between first ends the open line.
 */
void begin_phase(std::string_view name);
void end_phase(double seconds);

/**
 * @brief One-line event during the run (e.g. a file written), led by the step and time
 *        so it lines up with the progress table.
 */
void event(uint64_t step, double t, std::string_view kind, std::string_view text);

/**
 * @brief Warning on stderr from rank 0, for conditions every rank sees alike.
 */
void warning(std::string_view message);

/**
 * @brief Error on stderr from the calling rank, prefixed with the rank in parallel runs.
 */
void error(std::string_view message);

/// 1234567 -> "1,234,567".
std::string count(uint64_t n);
/// Wall-clock duration: "850 us", "12.3 ms", "4.56 s", "12m 03s", "2h 05m".
std::string duration(double seconds);
/// Value with an SI suffix and 3 significant digits: "16.3M", "1.20k".
std::string si(double value);
/// Compact real: "%.6g".
std::string real(double value);

} // namespace logging

#endif // LOG_H

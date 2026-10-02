/**
 * @file log.cpp
 * @brief Console output implementation.
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 */

#include "log.h"

#include <cmath>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <iostream>

#include <unistd.h>

#include "comm.h"

namespace logging {

namespace {

bool phase_open = false;

bool env_set(const char * name) {
    const char * value = std::getenv(name);
    return value != nullptr && value[0] != '\0' && std::string_view(value) != "0";
}

bool want_color(int fd) {
    if (env_set("NO_COLOR")) return false;
    if (env_set("CLICOLOR_FORCE")) return true;
    const char * term = std::getenv("TERM");
    if (term != nullptr && std::string_view(term) == "dumb") return false;
    // MPI launchers give each rank a pseudo-terminal even when their own output goes to a file
    const bool launched = comm::size() > 1 || env_set("OMPI_COMM_WORLD_SIZE") || env_set("PMI_SIZE") ||
                          env_set("PMIX_RANK");
    return !launched && isatty(fd);
}

/// Fixed-point with 3 significant digits for values in [1, 1000): 1.23, 12.3, 123.
std::string sig3(double v) {
    const double a = std::fabs(v);
    return format(a < 9.995 ? "%.2f" : (a < 99.95 ? "%.1f" : "%.0f"), v);
}

const char * ansi(Style s) {
    switch (s) {
        case Style::BOLD: return "\033[1m";
        case Style::DIM: return "\033[2m";
        case Style::RED: return "\033[1;31m";
        case Style::YELLOW: return "\033[1;33m";
        case Style::GREEN: return "\033[32m";
        case Style::CYAN: return "\033[36m";
    }
    return "";
}

std::string paint(std::string_view text, Style s, bool on) {
    if (!on) return std::string(text);
    return ansi(s) + std::string(text) + "\033[0m";
}

void close_phase() {
    if (phase_open) {
        std::cout << "\n";
        phase_open = false;
    }
}

} // namespace

std::string format(const char * fmt, ...) {
    va_list args;
    va_start(args, fmt);
    va_list copy;
    va_copy(copy, args);
    const int n = std::vsnprintf(nullptr, 0, fmt, copy);
    va_end(copy);
    std::string out(n > 0 ? n : 0, '\0');
    if (n > 0) std::vsnprintf(out.data(), n + 1, fmt, args);
    va_end(args);
    return out;
}

bool color_stdout() {
    static const bool on = want_color(STDOUT_FILENO);
    return on;
}

bool color_stderr() {
    static const bool on = want_color(STDERR_FILENO);
    return on;
}

std::string style(std::string_view text, Style s) {
    return paint(text, s, color_stdout());
}

void line(std::string_view text) {
    if (!comm::is_root()) return;
    close_phase();
    std::cout << text << std::endl;
}

void section(std::string_view title, std::string_view right) {
    if (!comm::is_root()) return;
    const int used = 4 + static_cast<int>(title.size()) + (right.empty() ? 0 : 1 + static_cast<int>(right.size()));
    std::string text = style("== ", Style::DIM) + style(title, Style::BOLD) + " ";
    text += style(std::string(std::max(3, WIDTH - used - 1), '='), Style::DIM);
    if (!right.empty()) text += " " + std::string(right);
    line();
    line(text);
}

void item(std::string_view key, std::string_view value) {
    if (!comm::is_root()) return;
    std::string padded(key);
    if (padded.size() < KEY_WIDTH) padded.resize(KEY_WIDTH, ' ');
    line("  " + padded + " " + std::string(value));
}

void items(const Items & list) {
    for (const auto & [key, value] : list) item(key, value);
}

void begin_phase(std::string_view name) {
    if (!comm::is_root()) return;
    close_phase();
    std::string padded(name);
    if (padded.size() < KEY_WIDTH + 8) padded.resize(KEY_WIDTH + 8, ' ');
    std::cout << "  " << padded << std::flush;
    phase_open = true;
}

void end_phase(double seconds) {
    if (!comm::is_root()) return;
    // Something printed since begin_phase moved the time onto its own line
    if (!phase_open) std::cout << "  " << std::string(KEY_WIDTH + 8, ' ');
    std::cout << format("%10s", duration(seconds).c_str()) << std::endl;
    phase_open = false;
}

void event(uint64_t step, double t, std::string_view kind, std::string_view text) {
    if (!comm::is_root()) return;
    std::string padded(kind);
    if (padded.size() < 8) padded.resize(8, ' ');
    line(format("%8llu %10.4e  ", static_cast<unsigned long long>(step), t) + style(padded, Style::CYAN) + " " +
         std::string(text));
}

void warning(std::string_view message) {
    if (!comm::is_root()) return;
    close_phase();
    std::cout << std::flush;
    std::cerr << paint("warning:", Style::YELLOW, color_stderr()) << " " << message << std::endl;
}

void error(std::string_view message) {
    close_phase();
    std::cout << std::flush;
    std::string prefix = comm::size() > 1 ? format("[rank %d] ", comm::rank()) : "";
    std::cerr << prefix << paint("error:", Style::RED, color_stderr()) << " " << message << std::endl;
}

std::string count(uint64_t n) {
    std::string digits = std::to_string(n);
    std::string out;
    const int len = static_cast<int>(digits.size());
    for (int i = 0; i < len; i++) {
        if (i > 0 && (len - i) % 3 == 0) out += ',';
        out += digits[i];
    }
    return out;
}

std::string duration(double s) {
    if (!std::isfinite(s) || s < 0.0) return "--";
    if (s < 0.9995e-3) return sig3(s * 1.0e6) + " us";
    if (s < 0.9995) return sig3(s * 1.0e3) + " ms";
    if (s < 59.95) return sig3(s) + " s";
    const uint64_t total = static_cast<uint64_t>(std::llround(s));
    const uint64_t h = total / 3600, m = (total / 60) % 60, sec = total % 60;
    if (h == 0) return format("%llum %02llus", static_cast<unsigned long long>(m), static_cast<unsigned long long>(sec));
    return format("%lluh %02llum", static_cast<unsigned long long>(h), static_cast<unsigned long long>(m));
}

std::string si(double v) {
    if (!std::isfinite(v)) return "--";
    const char * suffix[] = {"", "k", "M", "G", "T", "P"};
    int k = 0;
    while (std::fabs(v) >= 999.5 && k < 5) {
        v /= 1000.0;
        k++;
    }
    return (k == 0 ? format("%.0f", v) : sig3(v)) + suffix[k];
}

std::string real(double v) {
    return format("%.6g", v);
}

} // namespace logging

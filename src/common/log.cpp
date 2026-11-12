/**
 * @file log.cpp
 * @brief Console output implementation.
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 */

#include "log.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <iostream>

#include <unistd.h>

#include "comm.h"
#include "diagnostic_code.h"
#include "json.h"
#include "provenance.h"
#include "run_log.h"
#include "units.h"

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

/// Columns a UTF-8 string takes: its code points (status marks like ✓ are one column).
size_t columns(std::string_view text) {
    size_t n = 0;
    for (const char c : text) n += (static_cast<unsigned char>(c) & 0xC0) != 0x80;
    return n;
}

/// The micro prefix: the micro sign in a UTF-8 locale (the first of LC_ALL, LC_CTYPE, LANG set), else "u".
const char * micro() {
    static const char * prefix = [] {
        for (const char * name : {"LC_ALL", "LC_CTYPE", "LANG"}) {
            const char * value = std::getenv(name);
            if (value == nullptr || value[0] == '\0') continue;
            std::string v(value);
            std::transform(v.begin(), v.end(), v.begin(), [](unsigned char c) { return char(std::tolower(c)); });
            return v.find("utf-8") != std::string::npos || v.find("utf8") != std::string::npos ? "\u00b5" : "u";
        }
        return "u";
    }();
    return prefix;
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
    std::string out(n > 0 ? static_cast<size_t>(n) : 0, '\0');
    if (n > 0) std::vsnprintf(out.data(), static_cast<size_t>(n) + 1, fmt, args);
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

std::string style(std::string_view text, Style s, bool on) { return paint(text, s, on); }

void line(std::string_view text) {
    if (!comm::is_root()) return;
    close_phase();
    std::cout << text << std::endl;
}

void section(std::string_view title, std::string_view right) {
    if (!comm::is_root()) return;
    const int used = 4 + static_cast<int>(columns(title)) + (right.empty() ? 0 : 1 + static_cast<int>(columns(right)));
    std::string text = style("== ", Style::DIM) + style(title, Style::BOLD) + " ";
    text += style(std::string(static_cast<size_t>(std::max(3, WIDTH - used - 1)), '='), Style::DIM);
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

void table(const std::vector<std::string> & header, const std::vector<std::vector<std::string>> & rows) {
    std::vector<size_t> width(header.size());
    for (size_t c = 0; c < header.size(); c++) width[c] = columns(header[c]);
    for (const auto & row : rows) {
        for (size_t c = 0; c < row.size() && c < width.size(); c++) width[c] = std::max(width[c], columns(row[c]));
    }
    const auto text = [&](const std::vector<std::string> & row) {
        std::string out = " ";
        for (size_t c = 0; c < row.size() && c < width.size(); c++) {
            std::string cell = row[c];
            if (c + 1 < row.size()) cell.append(width[c] - columns(cell), ' ');
            out += " " + cell + (c + 1 < row.size() ? " " : "");
        }
        while (!out.empty() && out.back() == ' ') out.pop_back();
        return out;
    };
    line(style(text(header), Style::DIM));
    for (const auto & row : rows) line(text(row));
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

namespace {

bool verbose = false;
std::vector<std::pair<std::string, uint64_t>> written_since;

void remark(const uint64_t step, const double t, std::string_view kind, std::string_view text) {
    line("  " + style(">>", Style::CYAN) + " step " + count(step) + ", t = " + units::with(format("%.6g", t), "s") +
         ": " + std::string(kind) + ": " + std::string(text));
}

void record(const uint64_t step, const double t, std::string_view kind, std::string_view text,
            const std::vector<std::pair<std::string, std::string>> & fields = {}) {
    json::Object event;
    event.add("type", "event").add("step", step).add("t", t).add("kind", kind).add("text", text);
    for (const auto & [key, value] : fields) event.raw(key, value);
    run_log::write(event);
}

} // namespace

void event(uint64_t step, double t, std::string_view kind, std::string_view text,
           const std::vector<std::pair<std::string, std::string>> & fields) {
    if (!comm::is_root()) return;
    remark(step, t, kind, text);
    record(step, t, kind, text, fields);
}

void written(uint64_t step, double t, std::string_view kind, std::string_view path, std::string_view reason,
             std::string_view writer, const std::vector<WrittenFile> & files) {
    if (!comm::is_root()) return;
    if (verbose) remark(step, t, "wrote " + std::string(kind), path);
    uint64_t bytes = 0;
    std::string list = "[";
    for (const WrittenFile & f : files) {
        if (list.size() > 1) list += ",";
        json::Object o;
        o.add("path", f.path).add("bytes", f.bytes);
        if (!f.sha256.empty()) o.add("sha256", f.sha256);
        list += o.str();
        bytes += f.bytes;
    }
    run_log::write(json::Object()
                       .add("type", "output")
                       .add("step", step)
                       .add("t", t)
                       .add("kind", kind)
                       .add("path", path)
                       .add("reason", reason)
                       .add("writer", writer)
                       .add("run_id", provenance::run_id())
                       .add("complete", true)
                       .add("bytes", bytes)
                       .raw("files", list + "]"));
    auto it = std::find_if(written_since.begin(), written_since.end(), [&](const auto & k) { return k.first == kind; });
    if (it == written_since.end()) {
        written_since.emplace_back(std::string(kind), 1);
    } else {
        it->second++;
    }
}

void written(uint64_t step, double t, std::string_view kind, std::string_view path, std::string_view reason,
             std::string_view writer) {
    if (comm::is_root()) written(step, t, kind, path, reason, writer, {written_file(std::string(path))});
}

std::string take_written() {
    std::string out;
    for (const auto & [kind, n] : written_since) out += (out.empty() ? "" : ", ") + count(n) + " " + kind;
    written_since.clear();
    return out;
}

bool written_pending() { return !written_since.empty(); }

void set_verbose(const bool on) { verbose = on; }

std::string diagnostic(const bool is_error, std::string_view code, std::string_view message, const bool color) {
    return paint(std::string(is_error ? "error " : "warning ") + std::string(code) + ":", is_error ? Style::RED : Style::YELLOW,
                 color) +
           " " + std::string(message) + "\n  " + paint(diagnostic_code::docs(code), Style::DIM, color);
}

void warning(std::string_view code, std::string_view message) {
    if (!comm::is_root()) return;
    close_phase();
    std::cout << std::flush;
    std::cerr << diagnostic(false, code, message, color_stderr()) << std::endl;
    run_log::write(json::Object().add("type", "warning").add("code", std::string(code)).add("message", message).add(
                       "docs", diagnostic_code::docs(code)),
                   true);
}

void error(std::string_view code, std::string_view message) {
    close_phase();
    std::cout << std::flush;
    std::string prefix = comm::size() > 1 ? format("[rank %d] ", comm::rank()) : "";
    std::cerr << prefix << diagnostic(true, code, message, color_stderr()) << std::endl;
    run_log::write(json::Object().add("type", "error").add("code", std::string(code)).add("message", message).add(
        "docs", diagnostic_code::docs(code)));
}

std::string count(uint64_t n) {
    std::string digits = std::to_string(n);
    std::string out;
    const size_t len = digits.size();
    for (size_t i = 0; i < len; i++) {
        if (i > 0 && (len - i) % 3 == 0) out += ',';
        out += digits[i];
    }
    return out;
}

std::string duration(double s) {
    if (!std::isfinite(s) || s < 0.0) return "--";
    if (s < 0.9995e-3) return sig3(s * 1.0e6) + " " + micro() + "s";
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

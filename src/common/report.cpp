/**
 * @file report.cpp
 * @brief Structured console reports: sections, status columns and tables.
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 */

#include "report.h"

#include <algorithm>

#include "log.h"

namespace report {

namespace {

constexpr size_t WIDTH = static_cast<size_t>(logging::WIDTH);
// Tables may run wider than prose, so that a row of many columns keeps its last one on a line
constexpr size_t TABLE_WIDTH = 120;
constexpr size_t INDENT = 2;
constexpr size_t GAP = 2;
// A last column that starts further right than this is not wrapped
constexpr size_t MIN_WRAP = 32;

/** @brief text split at its "\n"s, each part wrapped to width at spaces (a longer word stays whole). */
std::vector<std::string> wrap(std::string_view text, const size_t width) {
    std::vector<std::string> lines;
    size_t start = 0;
    while (start <= text.size()) {
        const size_t end = std::min(text.find('\n', start), text.size());
        std::string_view part = text.substr(start, end - start);
        while (part.size() > width) {
            size_t cut = part.rfind(' ', width);
            if (cut == std::string_view::npos || cut == 0) cut = part.find(' ', width);
            if (cut == std::string_view::npos) break;
            lines.emplace_back(part.substr(0, cut));
            part.remove_prefix(cut + 1);
        }
        lines.emplace_back(part);
        start = end + 1;
    }
    return lines;
}

std::string rtrim(std::string s) {
    while (!s.empty() && s.back() == ' ') s.pop_back();
    return s;
}

} // namespace

Status worse(const Status a, const Status b) { return a > b ? a : b; }

const char * status_name(const Status s) {
    switch (s) {
        case Status::OK: return "ok";
        case Status::WARNING: return "warning";
        case Status::ERROR: return "error";
        case Status::NONE: break;
    }
    return "none";
}

const char * status_word(const Status s) {
    switch (s) {
        case Status::OK: return "ok";
        case Status::WARNING: return "WARN";
        case Status::ERROR: return "ERROR";
        case Status::NONE: break;
    }
    return "";
}

Text::Text(const bool color_in) : color(color_in) {}

void Text::add(std::string_view line) { text += rtrim(std::string(line)) + "\n"; }

std::string Text::styled_status(const Status s, const size_t width) const {
    std::string word = status_word(s);
    const size_t pad = width > word.size() ? width - word.size() : 0;
    const logging::Style style = s == Status::ERROR     ? logging::Style::RED
                                 : s == Status::WARNING ? logging::Style::YELLOW
                                                        : logging::Style::GREEN;
    if (!word.empty()) word = logging::style(word, style, color);
    return word + std::string(pad, ' ');
}

void Text::section(std::string_view title, const Status status) {
    if (!text.empty()) add("");
    const std::string word = status_word(status);
    const size_t used = 3 + title.size() + 1 + (word.empty() ? 0 : 1 + word.size());
    const std::string fill(used + 3 < WIDTH ? WIDTH - used : 3, '=');
    add(logging::style("== ", logging::Style::DIM, color) + logging::style(title, logging::Style::BOLD, color) + " " +
        logging::style(fill, logging::Style::DIM, color) + (word.empty() ? "" : " " + styled_status(status, 0)));
}

void Text::line(std::string_view line_text) {
    const std::vector<std::string> lines = wrap(line_text, WIDTH - INDENT);
    for (const std::string & l : lines) add(std::string(INDENT, ' ') + l);
}

void Text::note(std::string_view line_text) {
    const std::vector<std::string> lines = wrap(line_text, WIDTH - INDENT);
    for (const std::string & l : lines) add(std::string(INDENT, ' ') + logging::style(l, logging::Style::DIM, color));
}

void Text::blank() { add(""); }

void Text::table(const std::vector<std::string> & header, const std::vector<Row> & rows, const bool status_column) {
    std::vector<size_t> width(header.size());
    for (size_t c = 0; c < header.size(); c++) width[c] = header[c].size();
    size_t status_width = std::string_view("status").size();
    for (const Row & row : rows) {
        status_width = std::max(status_width, std::string_view(status_word(row.status)).size());
        for (size_t c = 0; c + 1 < row.cells.size() && c < width.size(); c++) {
            width[c] = std::max(width[c], row.cells[c].size());
        }
    }
    size_t last_start = INDENT + (status_column ? status_width + GAP : 0);
    for (size_t c = 0; c + 1 < width.size(); c++) last_start += width[c] + GAP;
    const size_t last_width = last_start + MIN_WRAP <= TABLE_WIDTH ? TABLE_WIDTH - last_start : std::string::npos;

    auto render = [&](const Status status, const std::vector<std::string> & cells, const bool is_header) {
        std::string first(INDENT, ' ');
        if (status_column) {
            first += is_header ? logging::style("status", logging::Style::DIM, color) +
                                     std::string(status_width - 6 + GAP, ' ')
                               : styled_status(status, status_width + GAP);
        }
        auto dim = [&](const std::string & s) { return is_header ? logging::style(s, logging::Style::DIM, color) : s; };
        // A row with fewer cells than columns ends with its own last cell, unpadded and unwrapped
        const bool short_row = cells.size() < width.size();
        for (size_t c = 0; c + 1 < cells.size() || (short_row && c < cells.size()); c++) {
            std::string cell = cells[c];
            if (c + 1 < cells.size()) cell.resize(width[c] + GAP, ' ');
            first += dim(cell);
        }
        if (short_row || cells.empty()) {
            add(first);
            return;
        }
        const std::vector<std::string> parts = wrap(cells.back(), last_width);
        for (size_t i = 0; i < parts.size(); i++) add((i == 0 ? first : std::string(last_start, ' ')) + dim(parts[i]));
    };
    render(Status::NONE, header, true);
    for (const Row & row : rows) render(row.status, row.cells, false);
}

void Text::print() const {
    size_t start = 0;
    while (start < text.size()) {
        const size_t end = text.find('\n', start);
        logging::line(std::string_view(text).substr(start, end - start));
        start = end + 1;
    }
}

} // namespace report

/**
 * @file report.cpp
 * @brief Structured console reports: sections, status columns and tables.
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 */

#include "report.h"

#include <algorithm>
#include <cctype>
#include <stdexcept>

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

bool digit(const char c) { return std::isdigit(static_cast<unsigned char>(c)) != 0; }

/** @brief Throws unless every VALUE cell is bare, with its row's units when it is a number. */
void validate(const std::vector<Column> & header, const std::vector<Row> & rows) {
    size_t units = header.size();
    bool values = false;
    for (size_t c = 0; c < header.size(); c++) {
        if (header[c].kind == Column::Kind::UNITS) units = c;
        values = values || header[c].kind == Column::Kind::VALUE;
        if (header[c].kind == Column::Kind::TEXT && header[c].name == "units") {
            throw std::logic_error("report: a units column is Column::Kind::UNITS");
        }
    }
    if (values != (units < header.size())) {
        throw std::logic_error("report: a table has VALUE columns exactly when it has a UNITS column");
    }
    for (const Row & row : rows) {
        bool number = false;
        for (size_t c = 0; c < header.size() && c < row.cells.size(); c++) {
            if (header[c].kind != Column::Kind::VALUE) continue;
            const std::string & cell = row.cells[c];
            if (!bare(cell)) {
                throw std::logic_error("report: \"" + cell + "\" in column " + header[c].name +
                                       " is not a bare value: its unit belongs in the units column");
            }
            number = number || (cell != "-" && cell != "none");
        }
        if (number && (units >= row.cells.size() || row.cells[units].empty())) {
            throw std::logic_error("report: row \"" + (row.cells.empty() ? std::string() : row.cells[0]) +
                                   "\" has a value but no units (\"dimensionless\" when it has none)");
        }
    }
}

} // namespace

bool bare(std::string_view s) {
    if (s == "-" || s == "none") return true;
    size_t i = 0;
    if (i < s.size() && (s[i] == '-' || s[i] == '+')) i++;
    if (s.substr(i) == "inf" || s.substr(i) == "nan") return true;
    size_t digits = 0;
    for (; i < s.size() && (digit(s[i]) || (s[i] == ',' && digits > 0)); i++) digits += digit(s[i]);
    if (i < s.size() && s[i] == '.') {
        for (i++; i < s.size() && digit(s[i]); i++) digits++;
    }
    if (digits == 0) return false;
    if (i < s.size() && (s[i] == 'e' || s[i] == 'E')) {
        i++;
        if (i < s.size() && (s[i] == '+' || s[i] == '-')) i++;
        size_t exponent = 0;
        for (; i < s.size() && digit(s[i]); i++) exponent++;
        if (exponent == 0) return false;
    }
    return i == s.size();
}

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
    const std::string fill(used + 4 < WIDTH ? WIDTH - used - 1 : 3, '=');
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

void Text::table(const std::vector<Column> & columns, const std::vector<Row> & rows, const bool status_column) {
    validate(columns, rows);
    std::vector<std::string> header;
    for (const Column & c : columns) header.push_back(c.name);
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

void Text::quantities(const std::vector<Row> & rows, const bool status_column) {
    using Kind = Column::Kind;
    const std::vector<Column> header = {"quantity", {"value", Kind::VALUE}, {"units", Kind::UNITS}, "target", "basis"};
    std::vector<bool> used(header.size(), false);
    for (const Row & row : rows) {
        for (size_t c = 0; c < row.cells.size() && c < used.size(); c++) used[c] = used[c] || !row.cells[c].empty();
    }
    used[2] = used[1];
    std::vector<Column> kept_header;
    for (size_t c = 0; c < header.size(); c++) {
        if (used[c]) kept_header.push_back(header[c]);
    }
    std::vector<Row> kept;
    for (const Row & row : rows) {
        Row k{row.status, {}};
        for (size_t c = 0; c < row.cells.size() && c < used.size(); c++) {
            if (used[c]) k.cells.push_back(row.cells[c]);
        }
        kept.push_back(std::move(k));
    }
    table(kept_header, kept, status_column);
}

Stream::Stream(std::vector<Field> fields_in, const bool color_in) : fields(std::move(fields_in)), color(color_in) {
    for (Field & f : fields) {
        if (f.kind == Column::Kind::VALUE && f.units.empty()) {
            throw std::logic_error("report: column " + f.name + " has values but no units (\"-\" when dimensionless)");
        }
        f.width = std::max({f.width, f.name.size(), f.units.size()});
    }
}

std::string Stream::render(const std::vector<std::string> & cells) const {
    std::string out(INDENT, ' ');
    for (size_t c = 0; c < fields.size(); c++) {
        const std::string & cell = cells[c];
        const size_t pad = fields[c].width > cell.size() ? fields[c].width - cell.size() : 0;
        if (c > 0) out += std::string(GAP, ' ');
        if (fields[c].kind == Column::Kind::TEXT && c + 1 == fields.size()) {
            out += cell;
        } else {
            out += std::string(pad, ' ') + cell;
        }
    }
    return rtrim(out);
}

std::vector<std::string> Stream::header() const {
    std::vector<std::string> names, units;
    for (const Field & f : fields) {
        names.push_back(f.name);
        units.push_back(f.units);
    }
    return {logging::style(render(names), logging::Style::BOLD, color),
            logging::style(render(units), logging::Style::DIM, color)};
}

std::string Stream::row(const std::vector<std::string> & cells) const {
    if (cells.size() != fields.size()) throw std::logic_error("report: a row has one cell per column");
    for (size_t c = 0; c < fields.size(); c++) {
        if (fields[c].kind == Column::Kind::VALUE && !bare(cells[c])) {
            throw std::logic_error("report: \"" + cells[c] + "\" in column " + fields[c].name +
                                   " is not a bare value: its unit belongs in the units row");
        }
    }
    return render(cells);
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

/**
 * @file report.h
 * @brief Structured console reports: sections whose heading ends with their status,
 *        and tables whose first column is a status (ok, WARN, ERROR), rendered to
 *        text with or without color.
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 */

#ifndef REPORT_H
#define REPORT_H

#include <string>
#include <string_view>
#include <vector>

namespace report {

/** @brief The status of a row, a section or a report, from none (information only) to error. */
enum class Status { NONE, OK, WARNING, ERROR };

/** @brief The worse of two statuses. */
Status worse(Status a, Status b);

/** @brief The status as JSON names it: "ok", "warning", "error", or "none". */
const char * status_name(Status s);

/** @brief The status as the status column shows it: "ok", "WARN", "ERROR", or empty. */
const char * status_word(Status s);

/** @brief A row of a table: its status and its cells, after the status column. */
struct Row {
    Status status = Status::NONE;
    std::vector<std::string> cells;
};

/**
 * @brief Text built section by section: headings, indented lines and tables, each line at most
 *        logging::WIDTH wide (tables 120) where it can wrap.
 */
class Text {
    public:
        /** @param color ANSI styles for the status words, headings and table headers. */
        explicit Text(bool color);

        /** @brief "== Title ====...==== STATUS", after a blank line (but for the first). */
        void section(std::string_view title, Status status = Status::NONE);

        /** @brief An indented line, wrapped at the width with its continuation lines indented as far. */
        void line(std::string_view text);

        /** @brief As line, dimmed: what a section's numbers are of. */
        void note(std::string_view text);

        /** @brief An empty line. */
        void blank();

        /**
         * @brief A table: a header row (dimmed), then a row per entry, each column as wide as its widest
         *        cell; the last column wraps at the width, and a cell's "\n" starts a continuation line.
         * @param status_column A "status" column first, holding each row's status.
         */
        void table(const std::vector<std::string> & header, const std::vector<Row> & rows, bool status_column = true);

        /** @brief The text, a "\n" after every line. */
        const std::string & str() const { return text; }

        /** @brief Prints every line on stdout from rank 0 (logging::line). */
        void print() const;

    private:
        void add(std::string_view line);
        std::string styled_status(Status s, size_t width) const;

        bool color;
        std::string text;
};

} // namespace report

#endif // REPORT_H

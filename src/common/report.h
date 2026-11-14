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
#include <utility>
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
 * @brief A column of a table. A table with a UNITS column keeps every value bare: its VALUE cells hold a number
 *        ("21.0", "1,024", "-1.5e-06") or "-" or "none", and a row with a number has its units; Text::table
 *        throws std::logic_error otherwise. A value with its unit ("21.0 s") belongs in a TEXT column of a table
 *        without a UNITS column.
 */
struct Column {
    enum class Kind { TEXT, VALUE, UNITS };
    Column(const char * name_in, Kind kind_in = Kind::TEXT) : name(name_in), kind(kind_in) {}
    Column(std::string name_in, Kind kind_in = Kind::TEXT) : name(std::move(name_in)), kind(kind_in) {}
    std::string name;
    Kind kind;
};

/** @brief Whether a cell is a bare value: a number, "-" or "none". */
bool bare(std::string_view cell);

/**
 * @brief A part of a run as its setup banner shows it: settings in words, then quantities, each a row of
 *        quantity, value, units, target (empty) and basis cells as Text::quantities takes them.
 */
struct Summary {
    std::vector<std::pair<std::string, std::string>> settings{};
    std::vector<Row> quantities{};

    void setting(std::string name, std::string value);
    /** @param units "-" when dimensionless; units::column for dimensional ones */
    void quantity(std::string name, std::string value, std::string units, std::string basis = "");
    void add(const Summary & other);
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
         *        cell, its numbers aligned at the decimal point (logging::align_decimals); the last column wraps
         *        at the width, and a cell's "\n" starts a continuation line.
         * @param status_column A "status" column first, holding each row's status.
         */
        void table(const std::vector<Column> & header, const std::vector<Row> & rows, bool status_column = true);

        /**
         * @brief A table of quantities: status, then quantity, value, units, target and basis cells, the columns
         *        empty in every row left out (units only with value).
         */
        void quantities(const std::vector<Row> & rows, bool status_column = true);

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

/** @brief A column of a Stream: its name, its units ("-" when dimensionless) and the width of its values. */
struct Field {
    std::string name;
    std::string units;
    size_t width = 0;
    Column::Kind kind = Column::Kind::VALUE;
};

/**
 * @brief A table printed a row at a time, as a run goes: the column names over their units, then rows
 *        that keep every column aligned. VALUE cells are bare (Text::table's rule). Cells are right-aligned,
 *        but for a last TEXT column, which runs on unpadded.
 */
class Stream {
    public:
        /** @throws std::logic_error if a VALUE field has no units */
        Stream(std::vector<Field> fields, bool color);

        /** @brief The names line (bold) and the units line (dimmed). */
        std::vector<std::string> header() const;

        /** @throws std::logic_error if the cells are not one per field, or a VALUE cell is not bare */
        std::string row(const std::vector<std::string> & cells) const;

    private:
        std::string render(const std::vector<std::string> & cells) const;

        std::vector<Field> fields;
        bool color;
};

} // namespace report

#endif // REPORT_H

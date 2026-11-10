/**
 * @file run_page.h
 * @brief The run report, report.html: one self-contained page built from a run's records (mallard.jsonl),
 *        record by record. The run feeds it the records it writes; mallard info RUN_DIR --html the records
 *        it reads back, so both build the same page.
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 */

#ifndef RUN_PAGE_H
#define RUN_PAGE_H

#include <cstdint>
#include <map>
#include <string>
#include <utility>
#include <vector>

#include "json.h"

namespace run_page {

/** @brief Buckets of a series: at most this many, merged in pairs when full, so a run of any length fits. */
constexpr size_t BUCKETS = 512;
/** @brief Series plotted; the others are in the table of last values only. */
constexpr size_t MAX_PLOTTED = 48;
/** @brief Warnings and events kept: the first and the last this many, with their count. */
constexpr size_t KEPT = 20;

/** @brief A series of progress rows, downsampled to at most BUCKETS buckets that keep its least and largest values. */
class Series {
    public:
        struct Bucket {
            uint64_t step = 0;  // of the last row
            double t = 0.0, last = 0.0, min = 0.0, max = 0.0;
            uint64_t n = 0;
        };
        std::string group, name, unit;

        void add(uint64_t step, double t, double value);
        /** @brief Drops the rows after the step (a rewound restart). */
        void truncate(uint64_t step);
        const std::vector<Bucket> & buckets() const { return data; }
        /** @brief Rows per full bucket. */
        uint64_t width() const { return per_bucket; }
        double least() const { return lowest; }
        double largest() const { return highest; }
        uint64_t rows() const { return n_rows; }

    private:
        std::vector<Bucket> data;
        uint64_t per_bucket = 1, n_rows = 0;
        double lowest = 0.0, highest = 0.0;
};

/** @brief The first and last KEPT of a list of records, and how many there were. */
struct Kept {
    std::vector<json::Value> first, last;
    uint64_t n = 0;
    void add(const json::Value & v);
};

/** @brief One run of a record continued by restarts. */
struct Part {
    std::string run_id, start_time;
    uint64_t first_step = 0, last_step = 0;
    std::string outcome;  // empty: no end record
};

/** @brief The output index, by writer and kind. */
struct Outputs {
    uint64_t n = 0, bytes = 0, first_step = 0, last_step = 0;
    std::string last_path;
};

class Page {
    public:
        /** @brief Adds the next record of the run log. */
        void add(const json::Value & record);
        /** @brief The page, from the records added so far: the same records give the same page. */
        std::string html() const;
        /** @brief Whether a record has been added. */
        bool empty() const { return n_records == 0; }

    private:
        void progress(const json::Value & r);
        Series & series(const std::string & group, const std::string & name, const std::string & unit);

        uint64_t n_records = 0;
        bool nondimensional = false;
        bool has_resolved = false;
        std::vector<std::pair<std::string, std::string>> configuration, boundaries;  // of the resolved input
        json::Value start, end, last_progress, divergence;
        std::vector<Part> parts;
        std::vector<Series> all;
        std::map<std::string, size_t> by_key;
        Kept warnings, errors, events;
        std::map<std::string, Outputs> outputs;  // by "writer (kind)"
        std::vector<json::Value> recent_outputs;
};

/** @brief The page of a run log file, read line by line (a last line cut short is skipped). */
Page from_file(const std::string & run_log);

/** @brief Writes the page as path.partial, then renames it to path. */
void write(const std::string & path, const std::string & html);

} // namespace run_page

#endif // RUN_PAGE_H

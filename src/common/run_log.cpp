/**
 * @file run_log.cpp
 * @brief Machine-readable run log implementation.
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 */

#include "run_log.h"

#include <filesystem>
#include <fstream>
#include <iostream>
#include <optional>
#include <stdexcept>

#include "comm.h"
#include "common_io.h"
#include "run_page.h"

namespace run_log {

namespace {

std::ofstream out;
std::optional<uint64_t> recorded_step;
std::optional<run_page::Page> page;
std::string report_file;
bool ended = false, reported_progress = false;

void write_report() {
    try {
        run_page::write(report_file, page->html());
    } catch (const std::exception & e) {
        // Not through logging::warning, whose record would come back here
        std::cerr << "warning: no run report from now on: " << e.what() << std::endl;
        page.reset();
    }
}

/** @brief Adds a record to the report, and writes it at a restart file, the first progress row and the end. */
void report(const std::string & line) {
    json::Value v;
    try {
        v = json::parse(line);
    } catch (const std::exception &) {
        return;
    }
    page->add(v);
    const json::Value * type = v.find("type");
    const json::Value * kind = v.find("kind");
    const std::string t = type ? type->string : "";
    bool due = ended || t == "end" || (t == "output" && kind && kind->string == "restart");
    if (t == "progress" && !reported_progress) due = reported_progress = true;
    if (t == "end") ended = true;
    if (due) write_report();
}

/** @brief The step of the last record that has one (its first "step" is the record's own). */
std::optional<uint64_t> last_step_in(const std::string & file) {
    std::ifstream in(file);
    std::optional<uint64_t> found;
    const std::string key = "\"step\":";
    for (std::string line; std::getline(in, line);) {
        const size_t k = line.find(key);
        if (k != std::string::npos) found = std::stoull(line.substr(k + key.size()));
    }
    return found;
}

} // namespace

void open(const std::string & file, bool append, const std::string & report) {
    close();
    recorded_step.reset();
    if (!comm::is_root()) return;
    if (!report.empty()) {
        page = append && !file.empty() ? run_page::from_file(file) : run_page::Page{};
        report_file = report;
        ended = reported_progress = false;
    }
    if (file.empty()) return;
    create_parent_directories("the run record", file);
    if (append) recorded_step = last_step_in(file);
    out.open(file, append ? std::ios::app : std::ios::trunc);
    if (!out.good()) throw cannot_write("the run record", file);
}

void close() {
    if (out.is_open()) out.close();
    page.reset();
}

bool is_open() { return out.is_open(); }

std::optional<uint64_t> last_step() { return recorded_step; }

void write(const json::Object & record) {
    if (!out.is_open() && !page) return;
    const std::string line = record.str();
    if (out.is_open()) {
        out << line << '\n';
        out.flush();
    }
    if (page) report(line);
}

} // namespace run_log

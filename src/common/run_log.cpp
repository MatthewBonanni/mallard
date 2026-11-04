/**
 * @file run_log.cpp
 * @brief Machine-readable run log implementation.
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 */

#include "run_log.h"

#include <filesystem>
#include <fstream>
#include <optional>
#include <stdexcept>

#include "comm.h"
#include "common_io.h"

namespace run_log {

namespace {

std::ofstream out;
std::optional<uint64_t> recorded_step;

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

void open(const std::string & file, bool append) {
    close();
    recorded_step.reset();
    if (!comm::is_root()) return;
    create_parent_directories("the run record", file);
    if (append) recorded_step = last_step_in(file);
    out.open(file, append ? std::ios::app : std::ios::trunc);
    if (!out.good()) throw cannot_write("the run record", file);
}

void close() {
    if (out.is_open()) out.close();
}

bool is_open() { return out.is_open(); }

std::optional<uint64_t> last_step() { return recorded_step; }

void write(const json::Object & record) {
    if (!out.is_open()) return;
    out << record.str() << '\n';
    out.flush();
}

} // namespace run_log

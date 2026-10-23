/**
 * @file run_log.cpp
 * @brief Machine-readable run log implementation.
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 */

#include "run_log.h"

#include <filesystem>
#include <fstream>
#include <stdexcept>

#include "comm.h"

namespace run_log {

namespace {

std::ofstream out;

} // namespace

void open(const std::string & file, bool append) {
    close();
    if (!comm::is_root()) return;
    const std::filesystem::path parent = std::filesystem::path(file).parent_path();
    if (!parent.empty()) std::filesystem::create_directories(parent);
    out.open(file, append ? std::ios::app : std::ios::trunc);
    if (!out.good()) throw std::runtime_error("Could not open the run log " + file + ".");
}

void close() {
    if (out.is_open()) out.close();
}

bool is_open() { return out.is_open(); }

void write(const json::Object & record) {
    if (!out.is_open()) return;
    out << record.str() << '\n';
    out.flush();
}

} // namespace run_log

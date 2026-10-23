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
#include "common_io.h"

namespace run_log {

namespace {

std::ofstream out;

} // namespace

void open(const std::string & file, bool append) {
    close();
    if (!comm::is_root()) return;
    create_parent_directories("the run log", file);
    out.open(file, append ? std::ios::app : std::ios::trunc);
    if (!out.good()) throw cannot_write("the run log", file);
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

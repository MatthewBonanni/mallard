/**
 * @file view_entries.cpp
 * @brief The most entries a multi-dimensional Kokkos View holds.
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 */

#include "view_entries.h"

#include <string>

#include "diagnostic_code.h"

namespace view_entries {

namespace {

uint64_t cells = 0;

} // namespace

void set_cells(const uint64_t n) { cells = n; }

void check(const std::string_view label, const uint64_t entries) {
    if (entries <= MAX) return;
    std::string message = "the Kokkos View \"" + std::string(label) + "\" needs " + std::to_string(entries) +
                          " entries on this rank, more than the " + std::to_string(MAX) +
                          " a multi-dimensional View can address";
    const uint64_t fit = cells * MAX / entries;
    if (fit > 0) {
        message += ": at most " + std::to_string(fit) + " cells fit on a rank (this one holds " +
                   std::to_string(cells) + ", halo included), so run on at least " +
                   std::to_string((cells + fit - 1) / fit) + " times as many ranks";
    } else {
        message += ": run on more ranks";
    }
    throw diagnostic_code::Error(diagnostic_code::SETUP, message);
}

} // namespace view_entries

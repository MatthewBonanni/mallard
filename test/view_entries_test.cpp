/**
 * @file view_entries_test.cpp
 * @brief Tests for the View size check of common/view_entries.h.
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 */

#include <gtest/gtest.h>
#include <Kokkos_Core.hpp>

#include <string>

#include "diagnostic_code.h"
#include "view_entries.h"

using Bytes2 = Kokkos::View<uint8_t **, Kokkos::HostSpace>;
using Bytes1 = Kokkos::View<uint8_t *, Kokkos::HostSpace>;

TEST(ViewEntriesTest, AllocationPastTheLimitThrows) {
    // 2^32 + 2^16 entries: the check runs before the allocation, which would otherwise take 4 GB
    EXPECT_THROW(Bytes2(Kokkos::view_alloc(Kokkos::WithoutInitializing, "too_big"), size_t(1) << 16,
                        (size_t(1) << 16) + 1),
                 diagnostic_code::Error);
}

namespace {

// Offset from v.data() of v's last entry, reached with int indices through v itself (path 0),
// its last row (1) or its last column (2): one of the two subviews is strided in either layout
uint64_t last_entry_offset(const Kokkos::View<uint8_t **> & v, const int path) {
    const int rows = static_cast<int>(v.extent(0));
    const int cols = static_cast<int>(v.extent(1));
    const auto row = Kokkos::subview(v, rows - 1, Kokkos::ALL);
    const auto col = Kokkos::subview(v, Kokkos::ALL, cols - 1);
    uint64_t offset = 0;
    Kokkos::parallel_reduce(
        "last_entry_offset", 1,
        KOKKOS_LAMBDA(int, uint64_t & o) {
            const uint8_t * p = path == 0 ? &v(rows - 1, cols - 1) : (path == 1 ? &row(cols - 1) : &col(rows - 1));
            o = static_cast<uint64_t>(p - v.data());
        },
        offset);
    return offset;
}

} // namespace

TEST(ViewEntriesTest, IntIndicesReachPast2To31Entries) {
#ifndef Mallard_UNSIGNED_VIEW_OFFSETS
    GTEST_SKIP() << "Kokkos_View.hpp is not patched: Views hold at most 2^31 entries";
#endif
    const size_t rows = size_t(1) << 16;
    const size_t cols = (size_t(1) << 15) + 1;
    const Kokkos::View<uint8_t **> v(Kokkos::view_alloc(Kokkos::WithoutInitializing, "past_int"), rows, cols);
    for (int path = 0; path < 3; path++) EXPECT_EQ(last_entry_offset(v, path), rows * cols - 1) << "path " << path;
}

TEST(ViewEntriesTest, RankOneViewsAreNotLimited) {
    const Bytes1 v(Kokkos::view_alloc(Kokkos::WithoutInitializing, "rank_one"), view_entries::MAX + 1);
    EXPECT_EQ(v.extent(0), view_entries::MAX + 1);
}

TEST(ViewEntriesTest, ErrorStatesTheCellsPerRank) {
    EXPECT_NO_THROW(view_entries::check("x", view_entries::MAX));
    view_entries::set_cells(1000);
    std::string message;
    try {
        view_entries::check("x", 4 * view_entries::MAX);
    } catch (const diagnostic_code::Error & e) {
        message = e.what();
    }
    view_entries::set_cells(0);
    EXPECT_NE(message.find("at most 250 cells fit on a rank"), std::string::npos) << message;
    EXPECT_NE(message.find("at least 4 times as many ranks"), std::string::npos) << message;
    EXPECT_THROW(view_entries::check("x", view_entries::MAX + 1), diagnostic_code::Error);
}

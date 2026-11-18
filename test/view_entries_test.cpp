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

namespace {

template <typename Layout>
void expect_load_row_reads_the_row() {
    const Kokkos::View<double * [5], Layout, Kokkos::HostSpace> v("rows", 7);
    for (uint32_t r = 0; r < 7; r++) {
        for (uint32_t i = 0; i < 5; i++) v(r, i) = 10.0 * r + i;
    }
    for (uint32_t r = 0; r < 7; r++) {
        double out[5];
        view_entries::load_row<5>(v, r, out);
        for (uint32_t i = 0; i < 5; i++) EXPECT_EQ(out[i], v(r, i)) << "row " << r << ", entry " << i;
    }
    const Kokkos::View<double * [5][3], Layout, Kokkos::HostSpace> g("rows3", 7);
    for (uint32_t r = 0; r < 7; r++) {
        for (uint32_t k = 0; k < 5; k++) {
            for (uint32_t i = 0; i < 3; i++) g(r, k, i) = 100.0 * r + 10.0 * k + i;
        }
    }
    for (uint32_t r = 0; r < 7; r++) {
        double out[5][3];
        view_entries::load_row(g, r, out);
        for (uint32_t k = 0; k < 5; k++) {
            for (uint32_t i = 0; i < 3; i++) EXPECT_EQ(out[k][i], g(r, k, i)) << "row " << r << ", entry " << k << ", " << i;
        }
    }
}

} // namespace

TEST(ViewEntriesTest, LoadRowReadsTheRowInEitherLayout) {
    expect_load_row_reads_the_row<Kokkos::LayoutLeft>();
    expect_load_row_reads_the_row<Kokkos::LayoutRight>();
}

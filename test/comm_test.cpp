/**
 * @file comm_test.cpp
 * @author Matthew Bonanni (mbonanni001@gmail.com)
 * @brief Communication layer; valid on any number of ranks.
 * @version 0.3
 * @date 2026-10-02
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 *
 */

#include <gtest/gtest.h>

#include <array>
#include <cstdint>

#include "comm.h"

TEST(CommTest, RanksAreNumberedFromZero) {
    EXPECT_GE(comm::rank(), 0);
    EXPECT_LT(comm::rank(), comm::size());
    EXPECT_EQ(comm::is_root(), comm::rank() == 0);
}

TEST(CommTest, AllreduceCombinesEveryRank) {
    const int p = comm::size();
    const double r = comm::rank();
    EXPECT_EQ(comm::allreduce(r + 1.0, comm::Op::SUM), p * (p + 1) / 2.0);
    EXPECT_EQ(comm::allreduce(r, comm::Op::MIN), 0.0);
    EXPECT_EQ(comm::allreduce(r, comm::Op::MAX), p - 1.0);
    EXPECT_EQ(comm::allreduce(uint64_t(1) << 40, comm::Op::SUM), uint64_t(p) << 40);
    const auto v = comm::allreduce(std::array<float, 2>{1.0f, float(r)}, comm::Op::SUM);
    EXPECT_EQ(v[0], float(p));
    EXPECT_EQ(v[1], float(p * (p - 1) / 2));
}

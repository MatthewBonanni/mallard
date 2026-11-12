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
#include <string>
#include <vector>

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

TEST(CommTest, AlltoallvSendsLongMessagesInPieces) {
    const uint64_t p = uint64_t(comm::size()), me = uint64_t(comm::rank());
    const auto length = [](uint64_t from, uint64_t to) { return (from + 2 * to) % 6; };
    const auto value = [](uint64_t from, uint64_t to, uint64_t i) { return 1000 * from + 100 * to + i; };
    std::vector<uint64_t> send_counts(p), send_offsets(p), recv_counts(p), recv_offsets(p);
    std::vector<uint64_t> send, expected;
    for (uint64_t q = 0; q < p; q++) {
        // Gaps between the messages: offsets, not counts, place them
        send.push_back(0);
        send_offsets[q] = send.size();
        send_counts[q] = length(me, q);
        for (uint64_t i = 0; i < send_counts[q]; i++) send.push_back(value(me, q, i));
        expected.push_back(0);
        recv_offsets[q] = expected.size();
        recv_counts[q] = length(q, me);
        for (uint64_t i = 0; i < recv_counts[q]; i++) expected.push_back(value(q, me, i));
    }
    std::vector<uint64_t> recv(expected.size(), 0);
    comm::detail::alltoallv<uint64_t>(send.data(), send_counts, send_offsets, recv.data(), recv_counts, recv_offsets,
                                      2);
    EXPECT_EQ(recv, expected);
}

TEST(CommTest, AllgatherReturnsEveryRanksString) {
    const auto text = [](int r) { return std::string(size_t(r % 3) * 2, char('a' + r)); };
    const std::vector<std::string> all = comm::allgather(text(comm::rank()));
    ASSERT_EQ(all.size(), size_t(comm::size()));
    for (int r = 0; r < comm::size(); r++) EXPECT_EQ(all[size_t(r)], text(r));
}

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
#include <filesystem>
#include <fstream>
#include <string>
#include <unistd.h>
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

#ifdef Mallard_HAS_MPI
TEST(CommTest, FileWritesGoInPieces) {
    const auto count = [](int r) { return r == 1 ? uint64_t(0) : uint64_t(r) + 3; };
    uint64_t first = 0, n_total = 0;
    for (int r = 0; r < comm::size(); r++) {
        if (r < comm::rank()) first += count(r);
        n_total += count(r);
    }
    const uint64_t n = count(comm::rank());
    std::vector<uint64_t> keys(n);
    std::vector<double> values(n);
    std::vector<MPI_Aint> displacements(n);
    for (uint64_t i = 0; i < n; i++) {
        keys[i] = 10 * (first + i) + 1;
        values[i] = 0.5 * double(first + i);
        displacements[i] = MPI_Aint(2 * (first + i) * sizeof(double));
    }
    const std::string header = "head!";
    const std::string filename =
        (std::filesystem::temp_directory_path() / ("mallard_comm_test_" + comm::broadcast(std::to_string(getpid()))))
            .string();
    MPI_File fh;
    ASSERT_EQ(MPI_File_open(comm::world(), filename.c_str(), MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &fh),
              MPI_SUCCESS);
    const MPI_Offset keys_at = MPI_Offset(header.size()), values_at = keys_at + MPI_Offset(n_total * sizeof(uint64_t));
    bool ok = MPI_File_set_size(fh, 0) == MPI_SUCCESS;
    if (comm::is_root()) ok = comm::write_at(fh, 0, header.data(), header.size(), MPI_BYTE, 2) && ok;
    ok = comm::write_at_all(fh, keys_at + MPI_Offset(first * sizeof(uint64_t)), keys.data(), n, MPI_UINT64_T, 2) && ok;
    ok = comm::write_all_hindexed(fh, values_at, values.data(), displacements.data(), n, MPI_DOUBLE, 2) && ok;
    ok = MPI_File_close(&fh) == MPI_SUCCESS && ok;
    EXPECT_TRUE(ok);
    comm::barrier();
    if (comm::is_root()) {
        std::ifstream in(filename, std::ios::binary);
        std::string head(header.size(), '\0');
        std::vector<uint64_t> read_keys(n_total);
        std::vector<double> read_values(2 * n_total, -1.0);
        in.read(head.data(), std::streamsize(head.size()));
        in.read(reinterpret_cast<char *>(read_keys.data()), std::streamsize(n_total * sizeof(uint64_t)));
        in.read(reinterpret_cast<char *>(read_values.data()), std::streamsize((2 * n_total - 1) * sizeof(double)));
        EXPECT_TRUE(in.good());
        EXPECT_EQ(head, header);
        for (uint64_t g = 0; g < n_total; g++) {
            EXPECT_EQ(read_keys[g], 10 * g + 1);
            EXPECT_EQ(read_values[2 * g], 0.5 * double(g));
        }
        std::filesystem::remove(filename);
    }
    comm::barrier();
}
#endif

/**
 * @file comm_test.cpp
 * @author Matthew Bonanni (mbonanni001@gmail.com)
 * @brief Communication layer and exact sums; valid on any number of ranks.
 * @version 0.4
 * @date 2026-11-11
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 *
 */

#include <gtest/gtest.h>
#include <Kokkos_Core.hpp>

#include <array>
#include <cstdint>
#include <string>
#include <vector>

#include "comm.h"
#include "device_comm.h"
#include "exact_sums.h"

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

namespace {

constexpr uint32_t N_GLOBAL_ROWS = 1000;

/** @brief This rank's rows of two columns of terms of the global row g, every size()-th g. */
template <typename Terms>
Terms global_terms(const std::string & label) {
    const uint32_t r = static_cast<uint32_t>(comm::rank()), p = static_cast<uint32_t>(comm::size());
    const uint32_t n = (N_GLOBAL_ROWS - r + p - 1) / p;
    Terms terms(label, n, 2);
    Kokkos::parallel_for("fill", n, KOKKOS_LAMBDA(const uint32_t i) {
        const uint32_t g = r + i * p;
        terms(i, 0) = 1.0 / (g + 1.0);
        terms(i, 1) = (g % 7 == 0 ? 1e8 : 1e-8) / (g + 3.0);
    });
    return terms;
}

// A free function: nvcc rejects device lambdas in TEST bodies, which are private members
void fill_rank_columns(const Kokkos::View<double **, Kokkos::LayoutLeft> & columns, const double r) {
    Kokkos::parallel_for("fill", columns.extent(0), KOKKOS_LAMBDA(const uint32_t c) {
        columns(c, 0) = Kokkos::sin(1.0 + c + 100.0 * r);
        columns(c, 1) = Kokkos::cos(c * r) * 1e-6;
    });
}

}

TEST(ExactSumsTest, SumsAreTheSameBitForBitOnAnyNumberOfRanks) {
    // The fixed-point sums of 1 / (g + 1) and of terms 16 orders apart, not the floating-point ones
    const std::array<double, 2> expected = {0x1.df11f45f4e617p+2, 0x1.8f5a74b891b44p+26};
    const bool distributed = comm::size() > 1;
    const auto fixed = global_terms<Kokkos::View<double *[2]>>("terms");
    const auto columns = global_terms<Kokkos::View<double **, Kokkos::LayoutLeft>>("columns");
    const uint32_t n = static_cast<uint32_t>(fixed.extent(0));
    EXPECT_EQ(exact_sums(fixed, n, distributed), expected);
    for (const bool nccl : {false, comm::nccl_available() && distributed}) {
        const ExactSums sums(2, n, distributed, nccl);
        sums.sum(fixed);
        const auto a = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), sums.sums());
        sums.sum(columns);
        const auto b = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), sums.sums());
        for (size_t k = 0; k < 2; k++) {
            EXPECT_EQ(a(k), expected[k]) << "column " << k << (nccl ? ", NCCL" : "");
            EXPECT_EQ(b(k), expected[k]) << "column " << k << (nccl ? ", NCCL" : "");
        }
    }
}

TEST(ExactSumsTest, AcrossRanksAreTheSameWithNcclAndMpi) {
    if (!comm::nccl_available() || comm::size() < 2) GTEST_SKIP() << "needs NCCL and more than one rank";
    const uint32_t n = 100 + 37 * static_cast<uint32_t>(comm::rank());
    Kokkos::View<double **, Kokkos::LayoutLeft> columns("columns", n, 2);
    const double r = comm::rank();
    fill_rank_columns(columns, r);
    const ExactSums mpi(2, n, true, false), nccl(2, n, true, true);
    mpi.sum(columns);
    nccl.sum(columns);
    const auto a = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), mpi.sums());
    const auto b = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), nccl.sums());
    EXPECT_EQ(a(0), b(0));
    EXPECT_EQ(a(1), b(1));
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

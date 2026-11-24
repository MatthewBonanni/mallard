/**
 * @file comm_test.cpp
 * @author Matthew Bonanni (mbonanni001@gmail.com)
 * @brief Communication layer and exact sums; valid on any number of ranks.
 * @version 0.5
 * @date 2026-11-22
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 *
 */

#include <gtest/gtest.h>
#include <Kokkos_Core.hpp>

#include <array>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <limits>
#include <string>
#include <vector>

#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>

#include "comm.h"
#include "device_comm.h"
#include "exact_sums.h"
#include "scratch.h"
#include "step_graph_guard.h"

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

constexpr uint32_t N_HARD_ROWS = 1100, N_HARD_COLUMNS = 7;

/**
 * @brief Term k of global row g of columns that only an exact sum rounded once gets right: terms from subnormals to
 *        2^1022 that cancel 550 rows apart but for the smallest, a tie broken by a subnormal, a term that is not
 *        finite, sums that overflow only on the way, -0s, and an overflow.
 */
KOKKOS_INLINE_FUNCTION double hard_term(const uint32_t g, const uint32_t k) {
    constexpr double big = 1.7976931348623157e308;
    switch (k) {
        case 0: {
            const uint64_t h = g % 550, e = h * 37 % 2046, m = (h * 0x9E3779B97F4A7C15ull) >> 12;
            const double v = Kokkos::bit_cast<double>(e << 52 | m);
            return g >= 550 && e >= 100 ? -v : v;
        }
        case 1: return g == 0 ? 1.0 : g <= 1024 ? 0x1p-63 : g == 1025 ? 0x1p-1074 : 0.0;
        case 2: return g == 7 ? Kokkos::Experimental::infinity_v<double> : 1.0;
        case 3: return g == 900 ? Kokkos::Experimental::quiet_NaN_v<double> : 1.0;
        case 4: return g < 2 ? big : g == 2 ? -big : -0.0;
        case 5: return -0.0;
        default: return g < 2 ? big : 0.0;
    }
}

/** @brief The exact sums of the columns of hard_term, rounded once (Python's fractions). */
std::array<double, N_HARD_COLUMNS> hard_sums() {
    const double nan = std::numeric_limits<double>::quiet_NaN(), inf = std::numeric_limits<double>::infinity();
    return {0x1.1f5010213c1c1p-925, 0x1.0000000000001p+0, nan, nan, 0x1.fffffffffffffp+1023, 0.0, inf};
}

/** @brief This rank's rows of hard_term, every size()-th global row. */
Kokkos::View<double **, Kokkos::LayoutLeft> hard_terms() {
    const uint32_t r = static_cast<uint32_t>(comm::rank()), p = static_cast<uint32_t>(comm::size());
    const uint32_t n = (N_HARD_ROWS - r + p - 1) / p;
    Kokkos::View<double **, Kokkos::LayoutLeft> terms("hard_terms", n, N_HARD_COLUMNS);
    Kokkos::parallel_for("fill", n, KOKKOS_LAMBDA(const uint32_t i) {
        for (uint32_t k = 0; k < N_HARD_COLUMNS; k++) terms(i, k) = hard_term(r + i * p, k);
    });
    return terms;
}

/** @brief Whether a and b are the same bits, NaN for NaN. */
bool same_bits(const double a, const double b) {
    return std::isnan(a) ? std::isnan(b) : Kokkos::bit_cast<uint64_t>(a) == Kokkos::bit_cast<uint64_t>(b);
}

/** @brief terms, then `extra` rows of terms no sum may read: NaN, and too large for the fixed point of the others. */
Kokkos::View<double *[2]> padded(const Kokkos::View<double *[2]> & terms, const uint32_t extra) {
    const uint32_t n = static_cast<uint32_t>(terms.extent(0));
    Kokkos::View<double *[2]> out("padded", n + extra);
    Kokkos::parallel_for("pad", n + extra, KOKKOS_LAMBDA(const uint32_t i) {
        for (int k = 0; k < 2; k++) out(i, k) = i < n ? terms(i, k) : (i % 2 ? Kokkos::Experimental::quiet_NaN_v<double> : 1e300);
    });
    return out;
}

}

TEST(ExactSumsTest, SumsAllocateNothingOnceConstructed) {
    // As in a step graph's capture, before any other sum: 7 columns take the windows of 4, 2 and 1 columns
    const uint32_t n = 1000;
    Kokkos::View<double **, Kokkos::LayoutLeft> columns("columns", n, 7);
    Kokkos::deep_copy(columns, 1.0);
    const ExactSums sums(7, n, false, false);
    const StepGraphGuard guard;
    Kokkos::Profiling::ProfilingSection section("step_graph_capture");
    section.start();
    sums.sum(columns);
    section.stop();
    for (const std::string & v : guard.violations()) ADD_FAILURE() << v;
}

TEST(ExactSumsTest, UnitLeavesRoomForEveryRow) {
    // 2^(62 - ceil(log2(n + 1))), as the host computed it in floating point before it was computed on the device
    // (exact for these counts; near 2^49 rows, log2 of 2^b + 1 rounds to b)
    for (uint64_t n = 0; n < 4100; n++) {
        const double reference = std::ldexp(1.0, 62 - static_cast<int>(std::ceil(std::log2(static_cast<double>(n) + 1.0))));
        ASSERT_EQ(exact_sums_unit(n), reference) << n;
    }
    for (int b = 12; b <= 40; b++) {
        for (const int64_t d : {-1, 0, 1}) {
            const uint64_t n = static_cast<uint64_t>(static_cast<int64_t>(uint64_t(1) << b) + d);
            const double reference =
                std::ldexp(1.0, 62 - static_cast<int>(std::ceil(std::log2(static_cast<double>(n) + 1.0))));
            EXPECT_EQ(exact_sums_unit(n), reference) << n;
        }
    }
}

TEST(ExactSumsTest, SumsOfRowsCountedOnTheDeviceAreThoseOfTheRowsAlone) {
    // The rows past the count would make every sum NaN
    const std::array<double, 2> expected = {0x1.df11f45f4e61ap+2, 0x1.8f5a74b891b48p+26};
    const bool distributed = comm::size() > 1;
    const auto terms = global_terms<Kokkos::View<double *[2]>>("terms");
    const uint32_t n = static_cast<uint32_t>(terms.extent(0));
    const auto rows = padded(terms, 1100);
    Kokkos::View<uint32_t> count("count");
    Kokkos::deep_copy(count, n);
    for (const bool nccl : {false, comm::nccl_available() && distributed}) {
        const ExactSums sums(2, static_cast<uint32_t>(rows.extent(0)), distributed, nccl);
        sums.sum(rows, count);
        const auto a = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), sums.sums());
        for (size_t k = 0; k < 2; k++) EXPECT_EQ(a(k), expected[k]) << "column " << k << (nccl ? ", NCCL" : "");
    }
}

TEST(ExactSumsTest, SumsAreTheSameBitForBitOnAnyNumberOfRanks) {
    // The exactly rounded sums of 1 / (g + 1) and of terms 16 orders apart, not the floating-point ones
    const std::array<double, 2> expected = {0x1.df11f45f4e61ap+2, 0x1.8f5a74b891b48p+26};
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

TEST(ExactSumsTest, AreTheExactSumRoundedOnce) {
    // Ties to even unless a lower bit is set, on either sign, in the subnormals, and at the overflow
    const auto sum = [](std::vector<double> terms) { return exact_column_sum(terms.data(), terms.size()); };
    const double min = 0x1p-1074, max = 1.7976931348623157e308;
    for (const double s : {1.0, -1.0}) {
        EXPECT_EQ(sum({s, s * 0x1p-53}), s);
        EXPECT_EQ(sum({s, s * 0x1p-53, s * min}), s * (1.0 + 0x1p-52));
        EXPECT_EQ(sum({s * (1.0 + 0x1p-52), s * 0x1p-53}), s * (1.0 + 0x1p-51));
        EXPECT_EQ(sum({s * (1.0 + 0x1p-52), s * 0x1p-53, -s * min}), s * (1.0 + 0x1p-52));
        EXPECT_EQ(sum({s * min, s * min, s * min}), s * 3 * min);
        EXPECT_EQ(sum({s * 0x1p-1022, -s * min}), s * (0x1p-1022 - min));
        EXPECT_EQ(sum({s * max, s * 0x1p969}), s * max);
        EXPECT_EQ(sum({s * max, s * 0x1p970}), s * std::numeric_limits<double>::infinity());
        EXPECT_EQ(sum({s * 0x1p100, s * 1.0, -s * 0x1p100}), s);
    }
    EXPECT_FALSE(std::signbit(sum({-0.0, -0.0})));
    EXPECT_FALSE(std::signbit(sum({-1.0, 1.0})));
    EXPECT_TRUE(std::isnan(sum({1.0, -std::numeric_limits<double>::infinity()})));
}

TEST(ExactSumsTest, AreTheExactSumOnAnyNumberOfRanks) {
    // Each rank holds every size()-th row, so the ranks' partial sums of column 0 are of terms up to 2^1022 that
    // cancel only across the ranks: any rounding before the end, or in any order, changes the sums
    const auto expected = hard_sums();
    std::vector<double> all(size_t(N_HARD_ROWS) * N_HARD_COLUMNS);
    for (uint32_t g = 0; g < N_HARD_ROWS; g++) {
        for (uint32_t k = 0; k < N_HARD_COLUMNS; k++) all[g * N_HARD_COLUMNS + k] = hard_term(g, k);
    }
    for (uint32_t k = 0; k < N_HARD_COLUMNS; k++) {
        EXPECT_TRUE(same_bits(exact_column_sum(all.data() + k, N_HARD_ROWS, N_HARD_COLUMNS), expected[k])) << k;
    }
    const bool distributed = comm::size() > 1;
    const auto terms = hard_terms();
    for (const bool nccl : {false, comm::nccl_available() && distributed}) {
        const ExactSums sums(N_HARD_COLUMNS, static_cast<uint32_t>(terms.extent(0)), distributed, nccl);
        sums.sum(terms);
        const auto a = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), sums.sums());
        for (uint32_t k = 0; k < N_HARD_COLUMNS; k++) {
            EXPECT_TRUE(same_bits(a(k), expected[k])) << "column " << k << ": " << a(k) << (nccl ? ", NCCL" : "");
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
    const std::string filename = (scratch_dir() / "pieces.bin").string();
    MPI_File fh;
    ASSERT_EQ(MPI_File_open(comm::communicator(), filename.c_str(), MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &fh),
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
    }
}
#endif

TEST(CommTest, SharedDirectoriesAreRemovedOrFailAlikeOnEveryRank) {
    const std::filesystem::path dir = scratch_dir() / "shared";
    if (comm::is_root()) {
        std::filesystem::create_directories(dir / "sub");
        std::ofstream(dir / "sub" / "file") << "x";
    }
    EXPECT_EQ(remove_shared_directory(dir), "");
    if (comm::is_root()) {
        EXPECT_FALSE(std::filesystem::exists(dir));
    }
    // A name no file system takes: rank 0's failure reaches every rank, and nothing throws
    std::string failure;
    EXPECT_NO_THROW(failure = remove_shared_directory(scratch_dir() / std::string(300, 'x')));
    EXPECT_NE(failure, "");
    EXPECT_EQ(comm::allgather(failure), std::vector<std::string>(size_t(comm::size()), failure));
}

TEST(CommTest, SharedDirectoryWithAFileStillOpenIsKeptOnEveryRank) {
    // Shared storage cannot remove a file a process has open, so no file system does
    const std::filesystem::path dir = scratch_dir() / "open";
    if (comm::is_root()) std::filesystem::create_directories(dir);
    comm::barrier();
    const std::string name = "held" + std::to_string(comm::rank());
    auto kept = [&](const std::string & failure) {
        EXPECT_NE(failure.find(name), std::string::npos) << failure;
        EXPECT_EQ(comm::allgather(failure), std::vector<std::string>(size_t(comm::size()), failure));
        EXPECT_TRUE(std::filesystem::exists(dir / name));
    };
    {
        std::ofstream out(dir / name);
        out << "x" << std::flush;
        kept(remove_shared_directory(dir));
    }
    // A mapping outlives its descriptor
    const int fd = ::open((dir / name).c_str(), O_RDONLY);
    ASSERT_GE(fd, 0);
    void * mapped = mmap(nullptr, 1, PROT_READ, MAP_PRIVATE, fd, 0);
    ::close(fd);
    ASSERT_NE(mapped, MAP_FAILED);
    kept(remove_shared_directory(dir));
    munmap(mapped, 1);
    EXPECT_EQ(remove_shared_directory(dir), "");
}

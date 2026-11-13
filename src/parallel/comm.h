/**
 * @file comm.h
 * @author Matthew Bonanni (mbonanni001@gmail.com)
 * @brief Process-level communication: MPI when built with Mallard_ENABLE_MPI,
 *        otherwise a single-rank stub with the same interface.
 * @version 0.3
 * @date 2026-10-02
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 *
 */

#ifndef COMM_H
#define COMM_H

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>
#include <type_traits>

#ifdef Mallard_HAS_MPI
#include <mpi.h>
#endif

namespace comm {

/**
 * @brief Initializes MPI on construction (unless it already is) and finalizes it
 *        on destruction if this session initialized it. Create one in main()
 *        before Kokkos is initialized, so Kokkos can map ranks to devices.
 */
class Session {
    public:
        Session(int & argc, char **& argv);
        ~Session();
        Session(const Session &) = delete;
        Session & operator=(const Session &) = delete;

#ifdef Mallard_HAS_MPI
    private:
        bool owns_mpi = false;
#endif
};

/** @brief Rank of this process in the world communicator; 0 before MPI starts and after it ends. */
int rank();

/** @brief Number of processes in the world communicator; 1 before MPI starts and after it ends. */
int size();

/** @brief Whether this is rank 0, which does all logging. */
inline bool is_root() { return rank() == 0; }

/** @brief Block until all ranks arrive. */
void barrier();

/** @brief Ends every rank with the given exit code (MPI_Abort in parallel runs). */
[[noreturn]] void abort(int code);

enum class Op { SUM, MIN, MAX };

/**
 * @brief In-place all-reduce over all ranks of the values in data.
 */
template <typename T>
void allreduce(std::span<T> data, Op op);

template <typename T>
T allreduce(T value, Op op) {
    allreduce(std::span<T>(&value, 1), op);
    return value;
}

template <typename T, std::size_t N>
std::array<T, N> allreduce(std::array<T, N> values, Op op) {
    allreduce(std::span<T>(values), op);
    return values;
}

/**
 * @brief Personalized all-to-all: send[r] goes to rank r; returns what each rank
 *        sent to this one, indexed by source rank.
 */
template <typename T>
std::vector<std::vector<T>> alltoallv(const std::vector<std::vector<T>> & send);

/**
 * @brief What an exchange() received: the part from rank r is
 *        data[offsets[r], offsets[r + 1]).
 */
template <typename T>
struct Received {
    std::vector<T> data;
    std::vector<uint64_t> offsets;

    std::span<const T> from(size_t r) const { return {data.data() + offsets[r], data.data() + offsets[r + 1]}; }
};

/**
 * @brief Personalized all-to-all like alltoallv, for large setup-time
 *        messages: frees each send list once packed, and returns what it
 *        received in one buffer, in rank order.
 */
template <typename T>
Received<T> exchange(std::vector<std::vector<T>> && send);

/**
 * @brief Rank 0's value, on every rank.
 */
std::string broadcast(const std::string & value);

/** @brief Every rank's string, in rank order, on every rank (collective). */
std::vector<std::string> allgather(const std::string & value);

/**
 * @brief The error of the lowest rank that has one (non-empty), on every rank;
 *        empty if none has. For checks of rank-local data, so that every rank
 *        throws and none waits in a later collective.
 */
std::string first_error(const std::string & error);

/**
 * @brief Concatenation of every rank's local vector, in rank order.
 */
template <typename T>
std::vector<T> allgatherv(const std::vector<T> & local);

/**
 * @brief Concatenation of every rank's local vector, in rank order, on rank 0
 *        (empty on the others).
 */
template <typename T>
std::vector<T> gatherv(const std::vector<T> & local);

namespace detail {

/** @brief The most values one MPI message holds (its count is an int). */
inline constexpr uint64_t mpi_max_count = uint64_t(std::numeric_limits<int>::max());

/**
 * @brief Personalized all-to-all of flat buffers with 64-bit counts and
 *        offsets (collective): sends send_counts[q] values from
 *        send + send_offsets[q] to rank q, and receives recv_counts[q] values
 *        from rank q into recv + recv_offsets[q], in messages of at most
 *        max_count values each. The spans hold at least size() entries.
 */
template <typename T>
void alltoallv(const T * send, std::span<const uint64_t> send_counts, std::span<const uint64_t> send_offsets,
               T * recv, std::span<const uint64_t> recv_counts, std::span<const uint64_t> recv_offsets,
               uint64_t max_count = mpi_max_count);

} // namespace detail

#ifdef Mallard_HAS_MPI
/** @brief The world communicator. */
MPI_Comm world();

/**
 * @brief Posts a nonblocking send of n values to rank `to`, in messages of at
 *        most max_count values each, and appends their requests. The matching
 *        irecv splits the same n the same way.
 */
template <typename T>
void isend(const T * data, const uint64_t n, const MPI_Datatype type, const int to, const int tag, const MPI_Comm c,
           std::vector<MPI_Request> & requests, const uint64_t max_count = detail::mpi_max_count) {
    for (uint64_t at = 0; at < n; at += max_count) {
        requests.push_back(MPI_REQUEST_NULL);
        if (MPI_Isend(data + at, static_cast<int>(std::min(max_count, n - at)), type, to, tag, c, &requests.back()) !=
            MPI_SUCCESS) {
            throw std::runtime_error("MPI error in MPI_Isend");
        }
    }
}

/** @brief Posts the nonblocking receive of n values from rank `from` that matches isend. */
template <typename T>
void irecv(T * data, const uint64_t n, const MPI_Datatype type, const int from, const int tag, const MPI_Comm c,
           std::vector<MPI_Request> & requests, const uint64_t max_count = detail::mpi_max_count) {
    for (uint64_t at = 0; at < n; at += max_count) {
        requests.push_back(MPI_REQUEST_NULL);
        if (MPI_Irecv(data + at, static_cast<int>(std::min(max_count, n - at)), type, from, tag, c, &requests.back()) !=
            MPI_SUCCESS) {
            throw std::runtime_error("MPI error in MPI_Irecv");
        }
    }
}

/**
 * @brief Writes n values at byte offset `at` of fh (independent), in pieces of
 *        at most max_count values each. Whether every piece succeeded.
 */
template <typename T>
bool write_at(const MPI_File fh, const MPI_Offset at, const T * data, const uint64_t n, const MPI_Datatype type,
              const uint64_t max_count = detail::mpi_max_count) {
    bool ok = true;
    for (uint64_t i = 0; i < n; i += max_count) {
        const int count = static_cast<int>(std::min(max_count, n - i));
        if (MPI_File_write_at(fh, at + MPI_Offset(i * sizeof(T)), data + i, count, type, MPI_STATUS_IGNORE) !=
            MPI_SUCCESS) {
            ok = false;
        }
    }
    return ok;
}

/**
 * @brief Collective write_at on a file of the world communicator: each rank
 *        writes its own n values, and every rank calls MPI_File_write_at_all
 *        as often as the rank with the most pieces needs.
 */
template <typename T>
bool write_at_all(const MPI_File fh, const MPI_Offset at, const T * data, const uint64_t n, const MPI_Datatype type,
                  const uint64_t max_count = detail::mpi_max_count) {
    const uint64_t n_pieces = allreduce((n + max_count - 1) / max_count, Op::MAX);
    bool ok = true;
    for (uint64_t p = 0; p < n_pieces; p++) {
        const uint64_t i = std::min(n, p * max_count);
        const int count = static_cast<int>(std::min(max_count, n - i));
        if (MPI_File_write_at_all(fh, at + MPI_Offset(i * sizeof(T)), data + i, count, type, MPI_STATUS_IGNORE) !=
            MPI_SUCCESS) {
            ok = false;
        }
    }
    return ok;
}

/**
 * @brief Collective write of n values on a file of the world communicator,
 *        value i at byte displacements[i] (increasing) past `at`, through file
 *        views of at most max_count values each. Leaves the last view on fh.
 */
template <typename T>
bool write_all_hindexed(const MPI_File fh, const MPI_Offset at, const T * data, const MPI_Aint * displacements,
                        const uint64_t n, const MPI_Datatype type, const uint64_t max_count = detail::mpi_max_count) {
    const uint64_t n_pieces = allreduce((n + max_count - 1) / max_count, Op::MAX);
    bool ok = true;
    for (uint64_t p = 0; p < n_pieces; p++) {
        const uint64_t i = std::min(n, p * max_count);
        const int count = static_cast<int>(std::min(max_count, n - i));
        MPI_Datatype file_type;
        MPI_Type_create_hindexed_block(count, 1, displacements + i, type, &file_type);
        MPI_Type_commit(&file_type);
        // Both collectives on every rank, even after a failure
        if (MPI_File_set_view(fh, at, type, file_type, "native", MPI_INFO_NULL) != MPI_SUCCESS) ok = false;
        if (MPI_File_write_all(fh, data + i, count, type, MPI_STATUS_IGNORE) != MPI_SUCCESS) ok = false;
        MPI_Type_free(&file_type);
    }
    return ok;
}
#endif

} // namespace comm

#endif // COMM_H

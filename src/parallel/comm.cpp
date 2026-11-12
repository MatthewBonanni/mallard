/**
 * @file comm.cpp
 * @author Matthew Bonanni (mbonanni001@gmail.com)
 * @brief Process-level communication.
 * @version 0.3
 * @date 2026-10-02
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 *
 */

#include "comm.h"

#ifdef Mallard_HAS_NCCL
#include "device_comm.h"
#endif

#include <algorithm>
#include <cstddef>
#include <cstdlib>
#include <stdexcept>
#include <string>

namespace comm {

#ifdef Mallard_HAS_MPI

namespace {

template <typename T>
MPI_Datatype mpi_type() {
    if constexpr (std::is_same_v<T, double>) return MPI_DOUBLE;
    else if constexpr (std::is_same_v<T, float>) return MPI_FLOAT;
    else if constexpr (std::is_same_v<T, char>) return MPI_CHAR;
    else if constexpr (std::is_same_v<T, int32_t>) return MPI_INT32_T;
    else if constexpr (std::is_same_v<T, int64_t>) return MPI_INT64_T;
    else if constexpr (std::is_same_v<T, uint32_t>) return MPI_UINT32_T;
    else if constexpr (std::is_same_v<T, uint64_t>) return MPI_UINT64_T;
    else static_assert(sizeof(T) == 0, "comm: unsupported type");
}

MPI_Op mpi_op(Op op) {
    switch (op) {
        case Op::SUM: return MPI_SUM;
        case Op::MIN: return MPI_MIN;
        case Op::MAX: return MPI_MAX;
    }
    throw std::logic_error("comm: unknown reduction");
}

void check(int err, const char * what) {
    if (err != MPI_SUCCESS) throw std::runtime_error(std::string("MPI error in ") + what);
}

std::vector<uint64_t> prefix_sum(const std::vector<uint64_t> & counts) {
    std::vector<uint64_t> offsets(counts.size() + 1, 0);
    for (size_t r = 0; r < counts.size(); r++) offsets[r + 1] = offsets[r] + counts[r];
    return offsets;
}

/** @brief Broadcasts a string whose length every rank already knows, in messages of at most mpi_max_count chars. */
void broadcast_chars(std::string & out, const int root) {
    const uint64_t n = out.size();
    for (uint64_t at = 0; at < n; at += detail::mpi_max_count) {
        check(MPI_Bcast(out.data() + at, static_cast<int>(std::min(detail::mpi_max_count, n - at)), MPI_CHAR, root,
                        MPI_COMM_WORLD),
              "MPI_Bcast");
    }
}

// A copy of the world communicator for the collectives built on point-to-point messages, so their messages never
// match receives that others post on the world communicator
MPI_Comm messages_comm = MPI_COMM_NULL;

MPI_Comm messages() {
    if (messages_comm == MPI_COMM_NULL) check(MPI_Comm_dup(MPI_COMM_WORLD, &messages_comm), "MPI_Comm_dup");
    return messages_comm;
}

/** @brief Whether MPI is initialized and not finalized: the front end reports errors before it starts. */
bool started() {
    int initialized = 0, finalized = 0;
    MPI_Initialized(&initialized);
    MPI_Finalized(&finalized);
    return initialized && !finalized;
}

} // namespace

Session::Session(int & argc, char **& argv) {
    int initialized = 0;
    MPI_Initialized(&initialized);
    if (!initialized) {
        check(MPI_Init(&argc, &argv), "MPI_Init");
        owns_mpi = true;
    }
}

Session::~Session() {
#ifdef Mallard_HAS_NCCL
    finalize_nccl();
#endif
    int finalized = 0;
    MPI_Finalized(&finalized);
    if (!finalized && messages_comm != MPI_COMM_NULL) MPI_Comm_free(&messages_comm);
    if (owns_mpi && !finalized) MPI_Finalize();
}

MPI_Comm world() { return MPI_COMM_WORLD; }

int rank() {
    int r = 0;
    if (started()) MPI_Comm_rank(MPI_COMM_WORLD, &r);
    return r;
}

int size() {
    int s = 1;
    if (started()) MPI_Comm_size(MPI_COMM_WORLD, &s);
    return s;
}

void barrier() { check(MPI_Barrier(MPI_COMM_WORLD), "MPI_Barrier"); }

std::string broadcast(const std::string & value) {
    uint64_t n = value.size();
    check(MPI_Bcast(&n, 1, MPI_UINT64_T, 0, MPI_COMM_WORLD), "MPI_Bcast");
    std::string out = is_root() ? value : std::string(n, '\0');
    broadcast_chars(out, 0);
    return out;
}

std::string first_error(const std::string & error) {
    const int n_ranks = size();
    int from = error.empty() ? n_ranks : rank();
    check(MPI_Allreduce(MPI_IN_PLACE, &from, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD), "MPI_Allreduce");
    if (from == n_ranks) return "";
    uint64_t n = error.size();
    check(MPI_Bcast(&n, 1, MPI_UINT64_T, from, MPI_COMM_WORLD), "MPI_Bcast");
    std::string out = rank() == from ? error : std::string(n, '\0');
    broadcast_chars(out, from);
    return out;
}

std::vector<std::string> allgather(const std::string & value) {
    const size_t p = static_cast<size_t>(size());
    uint64_t n_local = value.size();
    std::vector<uint64_t> counts(p);
    check(MPI_Allgather(&n_local, 1, MPI_UINT64_T, counts.data(), 1, MPI_UINT64_T, MPI_COMM_WORLD), "MPI_Allgather");
    const std::vector<uint64_t> offsets = prefix_sum(counts);
    std::string all(offsets[p], '\0');
    detail::alltoallv<char>(value.data(), std::vector<uint64_t>(p, n_local), std::vector<uint64_t>(p, 0), all.data(),
                            counts, offsets);
    std::vector<std::string> out(p);
    for (size_t r = 0; r < p; r++) out[r] = all.substr(offsets[r], counts[r]);
    return out;
}

void abort(int code) {
    int initialized = 0, finalized = 0;
    MPI_Initialized(&initialized);
    MPI_Finalized(&finalized);
    if (initialized && !finalized) MPI_Abort(MPI_COMM_WORLD, code);
    std::exit(code);
}

template <typename T>
void allreduce(std::span<T> data, Op op) {
    check(MPI_Allreduce(MPI_IN_PLACE, data.data(), static_cast<int>(data.size()), mpi_type<T>(), mpi_op(op),
                        MPI_COMM_WORLD),
          "MPI_Allreduce");
}

namespace detail {

template <typename T>
void alltoallv(const T * send, std::span<const uint64_t> send_counts, std::span<const uint64_t> send_offsets,
               T * recv, std::span<const uint64_t> recv_counts, std::span<const uint64_t> recv_offsets,
               const uint64_t max_count) {
    const int p = size(), me = rank();
    const size_t m = static_cast<size_t>(me);
    if (send_counts[m] != recv_counts[m]) throw std::logic_error("comm::alltoallv: unmatched message to self");
    std::copy_n(send + send_offsets[m], send_counts[m], recv + recv_offsets[m]);
    std::vector<MPI_Request> requests;
    for (int q = 0; q < p; q++) {
        const size_t r = static_cast<size_t>(q);
        if (q != me) irecv(recv + recv_offsets[r], recv_counts[r], mpi_type<T>(), q, 0, messages(), requests, max_count);
    }
    for (int q = 0; q < p; q++) {
        const size_t r = static_cast<size_t>(q);
        if (q != me) isend(send + send_offsets[r], send_counts[r], mpi_type<T>(), q, 0, messages(), requests, max_count);
    }
    check(MPI_Waitall(static_cast<int>(requests.size()), requests.data(), MPI_STATUSES_IGNORE), "MPI_Waitall");
}

} // namespace detail

template <typename T>
std::vector<std::vector<T>> alltoallv(const std::vector<std::vector<T>> & send) {
    const size_t p = static_cast<size_t>(size());
    if (send.size() != p) throw std::invalid_argument("comm::alltoallv: one list per rank");
    std::vector<uint64_t> send_counts(p), recv_counts(p);
    for (size_t r = 0; r < p; r++) send_counts[r] = send[r].size();
    check(MPI_Alltoall(send_counts.data(), 1, MPI_UINT64_T, recv_counts.data(), 1, MPI_UINT64_T, MPI_COMM_WORLD),
          "MPI_Alltoall");
    const std::vector<uint64_t> send_offsets = prefix_sum(send_counts), recv_offsets = prefix_sum(recv_counts);
    std::vector<T> send_flat(send_offsets[p]), recv_flat(recv_offsets[p]);
    for (size_t r = 0; r < p; r++) {
        std::copy(send[r].begin(), send[r].end(), send_flat.begin() + static_cast<ptrdiff_t>(send_offsets[r]));
    }
    detail::alltoallv<T>(send_flat.data(), send_counts, send_offsets, recv_flat.data(), recv_counts, recv_offsets);
    std::vector<std::vector<T>> recv(p);
    for (size_t r = 0; r < p; r++) {
        recv[r].assign(recv_flat.begin() + static_cast<ptrdiff_t>(recv_offsets[r]),
                       recv_flat.begin() + static_cast<ptrdiff_t>(recv_offsets[r + 1]));
    }
    return recv;
}

template <typename T>
Received<T> exchange(std::vector<std::vector<T>> && send) {
    const size_t p = static_cast<size_t>(size());
    if (send.size() != p) throw std::invalid_argument("comm::exchange: one list per rank");
    std::vector<uint64_t> send_counts(p), recv_counts(p);
    for (size_t r = 0; r < p; r++) send_counts[r] = send[r].size();
    check(MPI_Alltoall(send_counts.data(), 1, MPI_UINT64_T, recv_counts.data(), 1, MPI_UINT64_T, MPI_COMM_WORLD),
          "MPI_Alltoall");
    const std::vector<uint64_t> send_offsets = prefix_sum(send_counts);
    std::vector<T> send_flat;
    send_flat.reserve(send_offsets[p]);
    for (size_t r = 0; r < p; r++) {
        send_flat.insert(send_flat.end(), send[r].begin(), send[r].end());
        std::vector<T>().swap(send[r]);
    }
    Received<T> received;
    received.offsets = prefix_sum(recv_counts);
    received.data.resize(received.offsets[p]);
    detail::alltoallv<T>(send_flat.data(), send_counts, send_offsets, received.data.data(), recv_counts,
                         received.offsets);
    return received;
}

template <typename T>
std::vector<T> allgatherv(const std::vector<T> & local) {
    const size_t p = static_cast<size_t>(size());
    uint64_t n_local = local.size();
    std::vector<uint64_t> counts(p);
    check(MPI_Allgather(&n_local, 1, MPI_UINT64_T, counts.data(), 1, MPI_UINT64_T, MPI_COMM_WORLD), "MPI_Allgather");
    const std::vector<uint64_t> offsets = prefix_sum(counts);
    std::vector<T> all(offsets[p]);
    detail::alltoallv<T>(local.data(), std::vector<uint64_t>(p, n_local), std::vector<uint64_t>(p, 0), all.data(),
                         counts, offsets);
    return all;
}

template <typename T>
std::vector<T> gatherv(const std::vector<T> & local) {
    const size_t p = static_cast<size_t>(size());
    uint64_t n_local = local.size();
    std::vector<uint64_t> counts(p, 0), send_counts(p, 0);
    check(MPI_Gather(&n_local, 1, MPI_UINT64_T, counts.data(), 1, MPI_UINT64_T, 0, MPI_COMM_WORLD), "MPI_Gather");
    if (!is_root()) counts.assign(p, 0);
    send_counts[0] = n_local;
    const std::vector<uint64_t> offsets = prefix_sum(counts);
    std::vector<T> all(offsets[p]);
    detail::alltoallv<T>(local.data(), send_counts, std::vector<uint64_t>(p, 0), all.data(), counts, offsets);
    return all;
}

#else

Session::Session(int &, char **&) {}
Session::~Session() {}

int rank() { return 0; }
int size() { return 1; }
void barrier() {}
std::string broadcast(const std::string & value) { return value; }
std::string first_error(const std::string & error) { return error; }
std::vector<std::string> allgather(const std::string & value) { return {value}; }

void abort(int code) { std::exit(code); }

template <typename T>
void allreduce(std::span<T>, Op) {}

namespace detail {

template <typename T>
void alltoallv(const T * send, std::span<const uint64_t> send_counts, std::span<const uint64_t> send_offsets,
               T * recv, std::span<const uint64_t> recv_counts, std::span<const uint64_t> recv_offsets, uint64_t) {
    if (send_counts[0] != recv_counts[0]) throw std::logic_error("comm::alltoallv: unmatched message to self");
    std::copy_n(send + send_offsets[0], send_counts[0], recv + recv_offsets[0]);
}

} // namespace detail

template <typename T>
std::vector<T> allgatherv(const std::vector<T> & local) {
    return local;
}

template <typename T>
std::vector<T> gatherv(const std::vector<T> & local) {
    return local;
}

template <typename T>
std::vector<std::vector<T>> alltoallv(const std::vector<std::vector<T>> & send) {
    if (send.size() != 1) throw std::invalid_argument("comm::alltoallv: one list per rank");
    return send;
}

template <typename T>
Received<T> exchange(std::vector<std::vector<T>> && send) {
    if (send.size() != 1) throw std::invalid_argument("comm::exchange: one list per rank");
    Received<T> received;
    received.data = std::move(send[0]);
    received.offsets = {0, received.data.size()};
    return received;
}

#endif

template void allreduce<double>(std::span<double>, Op);
template void allreduce<float>(std::span<float>, Op);
template void allreduce<int32_t>(std::span<int32_t>, Op);
template void allreduce<int64_t>(std::span<int64_t>, Op);
template void allreduce<uint32_t>(std::span<uint32_t>, Op);
template void allreduce<uint64_t>(std::span<uint64_t>, Op);
template std::vector<std::vector<uint64_t>> alltoallv(const std::vector<std::vector<uint64_t>> &);
template std::vector<int32_t> allgatherv(const std::vector<int32_t> &);
template std::vector<uint64_t> allgatherv(const std::vector<uint64_t> &);
template std::vector<double> allgatherv(const std::vector<double> &);
template std::vector<float> allgatherv(const std::vector<float> &);
template std::vector<char> gatherv(const std::vector<char> &);
template std::vector<uint64_t> gatherv(const std::vector<uint64_t> &);
template std::vector<double> gatherv(const std::vector<double> &);
template std::vector<float> gatherv(const std::vector<float> &);
template std::vector<std::vector<double>> alltoallv(const std::vector<std::vector<double>> &);
template std::vector<std::vector<float>> alltoallv(const std::vector<std::vector<float>> &);
template Received<uint64_t> exchange(std::vector<std::vector<uint64_t>> &&);
template Received<double> exchange(std::vector<std::vector<double>> &&);
template Received<float> exchange(std::vector<std::vector<float>> &&);

template void detail::alltoallv<int32_t>(const int32_t *, std::span<const uint64_t>, std::span<const uint64_t>, int32_t *,
                                         std::span<const uint64_t>, std::span<const uint64_t>, uint64_t);
template void detail::alltoallv<uint64_t>(const uint64_t *, std::span<const uint64_t>, std::span<const uint64_t>, uint64_t *,
                                         std::span<const uint64_t>, std::span<const uint64_t>, uint64_t);
template void detail::alltoallv<double>(const double *, std::span<const uint64_t>, std::span<const uint64_t>, double *,
                                         std::span<const uint64_t>, std::span<const uint64_t>, uint64_t);
template void detail::alltoallv<float>(const float *, std::span<const uint64_t>, std::span<const uint64_t>, float *,
                                         std::span<const uint64_t>, std::span<const uint64_t>, uint64_t);

} // namespace comm

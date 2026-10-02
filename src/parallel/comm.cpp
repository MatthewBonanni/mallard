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

#include <algorithm>
#include <cstdlib>
#include <limits>
#include <stdexcept>
#include <string>

namespace comm {

#ifdef Mallard_HAS_MPI

namespace {

template <typename T>
MPI_Datatype mpi_type() {
    if constexpr (std::is_same_v<T, double>) return MPI_DOUBLE;
    else if constexpr (std::is_same_v<T, float>) return MPI_FLOAT;
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
    int finalized = 0;
    MPI_Finalized(&finalized);
    if (owns_mpi && !finalized) MPI_Finalize();
}

MPI_Comm world() { return MPI_COMM_WORLD; }

int rank() {
    int r = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &r);
    return r;
}

int size() {
    int s = 1;
    MPI_Comm_size(MPI_COMM_WORLD, &s);
    return s;
}

void barrier() { check(MPI_Barrier(MPI_COMM_WORLD), "MPI_Barrier"); }

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

template <typename T>
std::vector<std::vector<T>> alltoallv(const std::vector<std::vector<T>> & send) {
    const int p = size();
    if (static_cast<int>(send.size()) != p) throw std::invalid_argument("comm::alltoallv: one list per rank");
    std::vector<int> send_counts(p), recv_counts(p), send_displs(p + 1, 0), recv_displs(p + 1, 0);
    for (int r = 0; r < p; r++) send_counts[r] = static_cast<int>(send[r].size());
    check(MPI_Alltoall(send_counts.data(), 1, MPI_INT, recv_counts.data(), 1, MPI_INT, MPI_COMM_WORLD),
          "MPI_Alltoall");
    for (int r = 0; r < p; r++) {
        send_displs[r + 1] = send_displs[r] + send_counts[r];
        recv_displs[r + 1] = recv_displs[r] + recv_counts[r];
    }
    std::vector<T> send_flat(send_displs[p]), recv_flat(recv_displs[p]);
    for (int r = 0; r < p; r++) std::copy(send[r].begin(), send[r].end(), send_flat.begin() + send_displs[r]);
    check(MPI_Alltoallv(send_flat.data(), send_counts.data(), send_displs.data(), mpi_type<T>(),
                        recv_flat.data(), recv_counts.data(), recv_displs.data(), mpi_type<T>(), MPI_COMM_WORLD),
          "MPI_Alltoallv");
    std::vector<std::vector<T>> recv(p);
    for (int r = 0; r < p; r++) {
        recv[r].assign(recv_flat.begin() + recv_displs[r], recv_flat.begin() + recv_displs[r + 1]);
    }
    return recv;
}

template <typename T>
Received<T> exchange(std::vector<std::vector<T>> && send) {
    const int p = size();
    if (static_cast<int>(send.size()) != p) throw std::invalid_argument("comm::exchange: one list per rank");
    std::vector<int> send_counts(p), recv_counts(p), send_displs(p + 1, 0), recv_displs(p + 1, 0);
    for (int r = 0; r < p; r++) {
        if (send[r].size() > size_t(std::numeric_limits<int>::max())) {
            throw std::length_error("comm::exchange: message too large");
        }
        send_counts[r] = static_cast<int>(send[r].size());
        send_displs[r + 1] = send_displs[r] + send_counts[r];
    }
    check(MPI_Alltoall(send_counts.data(), 1, MPI_INT, recv_counts.data(), 1, MPI_INT, MPI_COMM_WORLD),
          "MPI_Alltoall");
    std::vector<T> send_flat;
    send_flat.reserve(send_displs[p]);
    for (int r = 0; r < p; r++) {
        send_flat.insert(send_flat.end(), send[r].begin(), send[r].end());
        std::vector<T>().swap(send[r]);
    }
    Received<T> received;
    received.offsets.assign(p + 1, 0);
    for (int r = 0; r < p; r++) {
        recv_displs[r + 1] = recv_displs[r] + recv_counts[r];
        received.offsets[r + 1] = recv_displs[r + 1];
    }
    received.data.resize(recv_displs[p]);
    check(MPI_Alltoallv(send_flat.data(), send_counts.data(), send_displs.data(), mpi_type<T>(),
                        received.data.data(), recv_counts.data(), recv_displs.data(), mpi_type<T>(), MPI_COMM_WORLD),
          "MPI_Alltoallv");
    return received;
}

template <typename T>
std::vector<T> allgatherv(const std::vector<T> & local) {
    const int p = size();
    int n_local = static_cast<int>(local.size());
    std::vector<int> counts(p), displs(p + 1, 0);
    check(MPI_Allgather(&n_local, 1, MPI_INT, counts.data(), 1, MPI_INT, MPI_COMM_WORLD), "MPI_Allgather");
    for (int r = 0; r < p; r++) displs[r + 1] = displs[r] + counts[r];
    std::vector<T> all(displs[p]);
    check(MPI_Allgatherv(local.data(), n_local, mpi_type<T>(), all.data(), counts.data(), displs.data(),
                         mpi_type<T>(), MPI_COMM_WORLD),
          "MPI_Allgatherv");
    return all;
}

#else

Session::Session(int &, char **&) {}
Session::~Session() {}

int rank() { return 0; }
int size() { return 1; }
void barrier() {}

void abort(int code) { std::exit(code); }

template <typename T>
void allreduce(std::span<T>, Op) {}

template <typename T>
std::vector<T> allgatherv(const std::vector<T> & local) {
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
template std::vector<std::vector<double>> alltoallv(const std::vector<std::vector<double>> &);
template std::vector<std::vector<float>> alltoallv(const std::vector<std::vector<float>> &);
template Received<uint64_t> exchange(std::vector<std::vector<uint64_t>> &&);
template Received<double> exchange(std::vector<std::vector<double>> &&);

} // namespace comm

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

template <typename T>
void allreduce(std::span<T> data, Op op) {
    check(MPI_Allreduce(MPI_IN_PLACE, data.data(), static_cast<int>(data.size()), mpi_type<T>(), mpi_op(op),
                        MPI_COMM_WORLD),
          "MPI_Allreduce");
}

#else

Session::Session(int &, char **&) {}
Session::~Session() {}

int rank() { return 0; }
int size() { return 1; }
void barrier() {}

template <typename T>
void allreduce(std::span<T>, Op) {}

#endif

template void allreduce<double>(std::span<double>, Op);
template void allreduce<float>(std::span<float>, Op);
template void allreduce<int32_t>(std::span<int32_t>, Op);
template void allreduce<int64_t>(std::span<int64_t>, Op);
template void allreduce<uint32_t>(std::span<uint32_t>, Op);
template void allreduce<uint64_t>(std::span<uint64_t>, Op);

} // namespace comm

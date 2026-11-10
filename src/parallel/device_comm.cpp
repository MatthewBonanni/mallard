/**
 * @file device_comm.cpp
 * @author Matthew Bonanni (mbonanni001@gmail.com)
 * @brief Stream-ordered communication of device buffers (NCCL).
 * @version 0.1
 * @date 2026-10-05
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 *
 */

#include "device_comm.h"

#include <cstdint>
#include <span>
#include <stdexcept>
#include <string>
#include <type_traits>

namespace comm {

#ifdef Mallard_HAS_NCCL

namespace {

ncclComm_t nccl_comm = nullptr;

} // namespace

ncclComm_t nccl() {
    if (nccl_comm) return nccl_comm;
    ncclUniqueId id;
    if (rank() == 0 && ncclGetUniqueId(&id) != ncclSuccess) throw std::runtime_error("NCCL: ncclGetUniqueId failed");
    MPI_Bcast(&id, sizeof(id), MPI_BYTE, 0, world());
    const ncclResult_t err = ncclCommInitRank(&nccl_comm, size(), id, rank());
    if (err != ncclSuccess) {
        nccl_comm = nullptr;
        throw std::runtime_error(std::string("NCCL: ncclCommInitRank failed: ") + ncclGetErrorString(err) +
                                 ". Check that each rank has its own GPU (one rank per GPU; NCCL_DEBUG=INFO tells "
                                 "more), or set parallel.halo_exchange = \"mpi\" to exchange halos without NCCL. "
                                 "See docs/guide/parallel.md#mpi.");
    }
    // NCCL connects at the first operation of each kind: do that now, not in the time loop
    Kokkos::View<double> x("nccl_warm_up");
    ncclAllReduce(x.data(), x.data(), 1, ncclDouble, ncclMin, nccl_comm, Kokkos::DefaultExecutionSpace().cuda_stream());
    Kokkos::fence("nccl_warm_up");
    return nccl_comm;
}

void finalize_nccl() {
    if (!nccl_comm) return;
    ncclCommDestroy(nccl_comm);
    nccl_comm = nullptr;
}

#endif

template <typename T>
void allreduce_device(const Kokkos::View<T *> & values, const Op op, [[maybe_unused]] const bool use_nccl) {
#ifdef Mallard_HAS_NCCL
    if (use_nccl) {
        const ncclDataType_t type =
            std::is_same_v<T, double> ? ncclDouble : (std::is_same_v<T, float> ? ncclFloat : ncclInt64);
        const ncclRedOp_t nccl_op = op == Op::SUM ? ncclSum : (op == Op::MIN ? ncclMin : ncclMax);
        if (ncclAllReduce(values.data(), values.data(), values.extent(0), type, nccl_op, nccl(),
                          Kokkos::DefaultExecutionSpace().cuda_stream()) != ncclSuccess) {
            throw std::runtime_error("NCCL: reduction failed");
        }
        return;
    }
#endif
    const auto host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), values);
    allreduce(std::span<T>(host.data(), host.extent(0)), op);
    Kokkos::deep_copy(values, host);
}

template void allreduce_device<double>(const Kokkos::View<double *> &, Op, bool);
template void allreduce_device<float>(const Kokkos::View<float *> &, Op, bool);
template void allreduce_device<int64_t>(const Kokkos::View<int64_t *> &, Op, bool);

void enqueue_wait(const Kokkos::DefaultExecutionSpace & waiter, const Kokkos::DefaultExecutionSpace & done) {
#ifdef KOKKOS_ENABLE_CUDA
    // A stream waits for the event's state when cudaStreamWaitEvent is called,
    // so one event serves every call
    static cudaEvent_t event = [] {
        cudaEvent_t e;
        cudaEventCreateWithFlags(&e, cudaEventDisableTiming);
        return e;
    }();
    cudaEventRecord(event, done.cuda_stream());
    cudaStreamWaitEvent(waiter.cuda_stream(), event, 0);
#else
    (void)waiter;
    done.fence("comm::enqueue_wait");
#endif
}

} // namespace comm

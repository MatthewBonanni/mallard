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

#include <algorithm>
#include <cstdint>
#include <cstdlib>
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
    MPI_Bcast(&id, sizeof(id), MPI_BYTE, 0, communicator());
    // NCCL's graph mixing support adds event nodes to captured graphs, which the step's conditional node cannot
    // hold. Mallard needs no mixing: its NCCL calls share one stream, and the host drains the step queue before
    // it calls NCCL outside a step graph.
#if NCCL_VERSION_CODE >= NCCL_VERSION(2, 29, 0)
    ncclConfig_t config = NCCL_CONFIG_INITIALIZER;
    config.graphUsageMode = 1;
    const ncclResult_t err = ncclCommInitRankConfig(&nccl_comm, size(), id, rank(), &config);
#else
    setenv("NCCL_GRAPH_MIXING_SUPPORT", "0", 0);
    const ncclResult_t err = ncclCommInitRank(&nccl_comm, size(), id, rank());
#endif
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
    const HostStaged staged;
    const auto host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), values);
    allreduce(std::span<T>(host.data(), host.extent(0)), op);
    Kokkos::deep_copy(values, host);
}

template void allreduce_device<double>(const Kokkos::View<double *> &, Op, bool);
template void allreduce_device<float>(const Kokkos::View<float *> &, Op, bool);
template void allreduce_device<int64_t>(const Kokkos::View<int64_t *> &, Op, bool);

template <typename T>
void allgather_device(const Kokkos::View<T *> & values, const Kokkos::View<T *> & all, [[maybe_unused]] const bool use_nccl) {
#ifdef Mallard_HAS_NCCL
    if (use_nccl) {
        if (ncclAllGather(values.data(), all.data(), values.extent(0), ncclDouble, nccl(),
                          Kokkos::DefaultExecutionSpace().cuda_stream()) != ncclSuccess) {
            throw std::runtime_error("NCCL: gather failed");
        }
        return;
    }
#endif
    const HostStaged staged;
    const auto host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), values);
    const std::vector<T> gathered = allgatherv(std::vector<T>(host.data(), host.data() + host.extent(0)));
    Kokkos::deep_copy(all, Kokkos::View<const T *, Kokkos::HostSpace, Kokkos::MemoryUnmanaged>(gathered.data(),
                                                                                               gathered.size()));
}

template void allgather_device<double>(const Kokkos::View<double *> &, const Kokkos::View<double *> &, bool);

template <typename T>
void alltoallv_device(const Kokkos::View<T *> & send, const std::vector<uint64_t> & send_counts,
                      const Kokkos::View<T *> & recv, const std::vector<uint64_t> & recv_counts, const bool use_nccl) {
    if (use_nccl) {
        std::vector<DeviceMessage> out, in;
        uint64_t s = 0, r = 0;
        for (size_t q = 0; q < send_counts.size(); q++) {
            if (send_counts[q] > 0) out.push_back({static_cast<int>(q), send.data() + s, send_counts[q] * sizeof(T)});
            if (recv_counts[q] > 0) in.push_back({static_cast<int>(q), recv.data() + r, recv_counts[q] * sizeof(T)});
            s += send_counts[q];
            r += recv_counts[q];
        }
        send_receive_device(out, in);
        return;
    }
    const HostStaged staged;
    const auto h_send = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), send);
    std::vector<std::vector<T>> out(send_counts.size());
    uint64_t at = 0;
    for (size_t q = 0; q < out.size(); q++) {
        out[q].assign(h_send.data() + at, h_send.data() + at + send_counts[q]);
        at += send_counts[q];
    }
    const std::vector<std::vector<T>> in = alltoallv(out);
    const auto h_recv = Kokkos::create_mirror_view(recv);
    at = 0;
    for (size_t q = 0; q < in.size(); q++) {
        if (in[q].size() != recv_counts[q]) throw std::logic_error("comm::alltoallv_device: unexpected message size");
        std::copy(in[q].begin(), in[q].end(), h_recv.data() + at);
        at += recv_counts[q];
    }
    Kokkos::deep_copy(recv, h_recv);
}

template void alltoallv_device<double>(const Kokkos::View<double *> &, const std::vector<uint64_t> &,
                                       const Kokkos::View<double *> &, const std::vector<uint64_t> &, bool);
template void alltoallv_device<float>(const Kokkos::View<float *> &, const std::vector<uint64_t> &,
                                      const Kokkos::View<float *> &, const std::vector<uint64_t> &, bool);

template <typename T>
void allgatherv_device(const Kokkos::View<T *> & values, const Kokkos::View<T *> & all,
                       const std::vector<uint64_t> & counts, const bool use_nccl) {
    if (use_nccl) {
        std::vector<DeviceMessage> out, in;
        const size_t me = static_cast<size_t>(rank());
        uint64_t at = 0;
        for (size_t q = 0; q < counts.size(); q++) {
            if (counts[me] > 0) out.push_back({static_cast<int>(q), values.data(), counts[me] * sizeof(T)});
            if (counts[q] > 0) in.push_back({static_cast<int>(q), all.data() + at, counts[q] * sizeof(T)});
            at += counts[q];
        }
        send_receive_device(out, in);
        return;
    }
    const HostStaged staged;
    const auto host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), values);
    const std::vector<T> gathered = allgatherv(std::vector<T>(host.data(), host.data() + host.extent(0)));
    Kokkos::deep_copy(Kokkos::subview(all, Kokkos::make_pair(size_t(0), gathered.size())),
                      Kokkos::View<const T *, Kokkos::HostSpace, Kokkos::MemoryUnmanaged>(gathered.data(), gathered.size()));
}

template void allgatherv_device<double>(const Kokkos::View<double *> &, const Kokkos::View<double *> &,
                                        const std::vector<uint64_t> &, bool);

void send_receive_device([[maybe_unused]] const std::vector<DeviceMessage> & sends,
                         [[maybe_unused]] const std::vector<DeviceMessage> & receives) {
#ifdef Mallard_HAS_NCCL
    const cudaStream_t stream = Kokkos::DefaultExecutionSpace().cuda_stream();
    ncclGroupStart();
    for (const DeviceMessage & m : receives) ncclRecv(m.data, m.bytes, ncclChar, m.rank, nccl(), stream);
    for (const DeviceMessage & m : sends) ncclSend(m.data, m.bytes, ncclChar, m.rank, nccl(), stream);
    if (ncclGroupEnd() != ncclSuccess) throw std::runtime_error("NCCL: send and receive failed");
#else
    throw std::logic_error("comm::send_receive_device: built without NCCL");
#endif
}

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
    const HostStaged staged;
    done.fence("comm::enqueue_wait");
#endif
}

Kokkos::DefaultExecutionSpace high_priority_space() {
#ifdef KOKKOS_ENABLE_CUDA
    int least = 0, greatest = 0;
    cudaDeviceGetStreamPriorityRange(&least, &greatest);
    cudaStream_t stream;
    if (cudaStreamCreateWithPriority(&stream, cudaStreamNonBlocking, greatest) != cudaSuccess) {
        throw std::runtime_error("comm::high_priority_space: cudaStreamCreateWithPriority failed");
    }
    return Kokkos::Cuda(stream, Kokkos::Impl::ManageStream::yes);
#else
    return Kokkos::DefaultExecutionSpace();
#endif
}

} // namespace comm

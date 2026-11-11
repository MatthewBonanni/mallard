/**
 * @file device_comm.h
 * @author Matthew Bonanni (mbonanni001@gmail.com)
 * @brief Stream-ordered communication of device buffers (NCCL), and ordering
 *        between execution space instances without host synchronization.
 * @version 0.1
 * @date 2026-10-05
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 *
 */

#ifndef DEVICE_COMM_H
#define DEVICE_COMM_H

#include <cstdint>
#include <vector>

#include <Kokkos_Core.hpp>
#include <Kokkos_Profiling_ProfileSection.hpp>

#include "comm.h"

#ifdef Mallard_HAS_NCCL
#include <nccl.h>
#endif

namespace comm {

/** @brief Whether this build can exchange device buffers with NCCL. */
constexpr bool nccl_available() {
#ifdef Mallard_HAS_NCCL
    return true;
#else
    return false;
#endif
}

#ifdef Mallard_HAS_NCCL
/** @brief The NCCL communicator of MPI_COMM_WORLD; collective on the first call. */
ncclComm_t nccl();

/** @brief Destroy the communicator, if any (before MPI_Finalize). */
void finalize_nccl();
#endif

/**
 * @brief Reduce values over the ranks in place, after the work enqueued so far
 *        on the default execution space: with NCCL on its stream, without
 *        blocking the host (capturable in a CUDA graph), else through the host
 *        with MPI. Defined for double, float and int64_t.
 */
template <typename T>
void allreduce_device(const Kokkos::View<T *> & values, Op op, bool use_nccl);

/**
 * @brief Gather every rank's values into all, in rank order (all holds size()
 *        times values), after the work enqueued so far on the default
 *        execution space: with NCCL on its stream, else through the host with
 *        MPI. Defined for double.
 */
template <typename T>
void allgather_device(const Kokkos::View<T *> & values, const Kokkos::View<T *> & all, bool use_nccl);

/**
 * @brief Send send_counts[q] values of send, in rank order, to each rank q, and
 *        receive recv_counts[q] values from each into recv, in rank order: with
 *        NCCL on the default execution space's stream, else through the host
 *        with MPI. Defined for double and float.
 */
template <typename T>
void alltoallv_device(const Kokkos::View<T *> & send, const std::vector<uint64_t> & send_counts,
                      const Kokkos::View<T *> & recv, const std::vector<uint64_t> & recv_counts, bool use_nccl);

/**
 * @brief Gather every rank's values into all, in rank order, counts[q] values
 *        from rank q: with NCCL on the default execution space's stream, else
 *        through the host with MPI. Defined for double.
 */
template <typename T>
void allgatherv_device(const Kokkos::View<T *> & values, const Kokkos::View<T *> & all,
                       const std::vector<uint64_t> & counts, bool use_nccl);

/** @brief A message of device memory to or from another rank. */
struct DeviceMessage {
    int rank;
    void * data;
    size_t bytes;
};

/**
 * @brief Send and receive the messages with NCCL on the default execution
 *        space's stream, after the work enqueued so far, without blocking the
 *        host; every rank calls it with messages matching its peers'. NCCL
 *        builds only: callers exchange with MPI otherwise.
 */
void send_receive_device(const std::vector<DeviceMessage> & sends, const std::vector<DeviceMessage> & receives);

/**
 * @brief Make the work enqueued later on waiter wait for the work enqueued so
 *        far on done, without blocking the host where the backend allows it
 *        (CUDA); a fence of done otherwise.
 */
void enqueue_wait(const Kokkos::DefaultExecutionSpace & waiter, const Kokkos::DefaultExecutionSpace & done);

/**
 * @brief While alive, marks for Kokkos Tools (profiling section "host_staged")
 *        host round trips that a CUDA graph's step does not make: MPI's
 *        messages, which it sends with NCCL, and host fences off CUDA.
 */
class HostStaged {
    public:
        HostStaged() : section("host_staged") { section.start(); }
        ~HostStaged() { section.stop(); }
        HostStaged(const HostStaged &) = delete;
        HostStaged & operator=(const HostStaged &) = delete;

    private:
        Kokkos::Profiling::ProfilingSection section;
};

} // namespace comm

#endif // DEVICE_COMM_H

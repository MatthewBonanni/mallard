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

#include <Kokkos_Core.hpp>

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
 * @brief Make the work enqueued later on waiter wait for the work enqueued so
 *        far on done, without blocking the host where the backend allows it
 *        (CUDA); a fence of done otherwise.
 */
void enqueue_wait(const Kokkos::DefaultExecutionSpace & waiter, const Kokkos::DefaultExecutionSpace & done);

} // namespace comm

#endif // DEVICE_COMM_H

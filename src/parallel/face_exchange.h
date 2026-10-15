/**
 * @file face_exchange.h
 * @author Matthew Bonanni (mbonanni001@gmail.com)
 * @brief Sends the reconstructed states of owned cells at partition faces to
 *        the ranks on the other side, so that no rank reconstructs halo cells.
 * @version 0.1
 * @date 2026-10-15
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 *
 */

#ifndef FACE_EXCHANGE_H
#define FACE_EXCHANGE_H

#include <cstdint>
#include <vector>

#include <Kokkos_Core.hpp>

#include "comm.h"
#include "common_typedef.h"
#include "distribution.h"

/** @brief A local face between an owned cell and a halo cell. */
struct PartitionFace {
    uint32_t face;
    uint32_t halo_cell;
    uint8_t owned_side;  // side of the owned cell in cells_of_face
    uint64_t key;        // the same on both ranks: global id of side 0 and the face's index in that cell
};

/** @brief Per neighbor rank, its partition faces, listed in the same order on both ranks. */
struct FacePlan {
    std::vector<int> neighbors;
    std::vector<std::vector<uint32_t>> faces;
    std::vector<std::vector<uint8_t>> owned_side;
};

/**
 * @brief The plan of a FaceExchange, from the partition faces of this rank and
 *        the owners of halo cells in dist's receive lists. Collective: checks
 *        that both sides list the same faces.
 */
FacePlan plan_face_exchange(const Distribution & dist, std::vector<PartitionFace> faces);

/**
 * @brief Point-to-point exchange of face states at partition faces: each rank
 *        sends its owned side and receives the other, as in HaloExchange
 *        (device buffers; GPU-aware MPI, host staging or NCCL).
 */
class FaceExchange {
    public:
        FaceExchange() = default;
        FaceExchange(const FacePlan & plan, uint8_t n_quad, bool nccl);

        /**
         * @brief Pack the owned sides of face_solution on exec and send them.
         *        With MPI the host waits for exec's work so far; with NCCL the
         *        messages are enqueued on exec's stream.
         */
        void start(const Kokkos::DefaultExecutionSpace & exec, Kokkos::View<rtype **[2][N_CONSERVATIVE]> face_solution);

        /** @brief Wait for the messages and write the other sides of face_solution on exec. */
        void finish(const Kokkos::DefaultExecutionSpace & exec, Kokkos::View<rtype **[2][N_CONSERVATIVE]> face_solution);

        bool active() const { return !ranks.empty(); }

    private:
        std::vector<int> ranks;
        bool nccl = false;
        uint32_t n_values = 0;  // per face: quadrature points times states
        std::vector<uint32_t> offsets;  // per neighbor, in faces
        Kokkos::View<uint32_t *> faces;
        Kokkos::View<uint8_t *> sides;
        Kokkos::View<rtype *> send_buffer, recv_buffer;
        Kokkos::View<rtype *>::host_mirror_type h_send_buffer, h_recv_buffer;
#ifdef Mallard_HAS_MPI
        std::vector<MPI_Request> requests;
#endif
};

#endif // FACE_EXCHANGE_H

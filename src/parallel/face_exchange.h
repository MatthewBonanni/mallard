/**
 * @file face_exchange.h
 * @author Matthew Bonanni (mbonanni001@gmail.com)
 * @brief Sends the reconstructed states of owned cells at their faces to the
 *        ranks that read them, so that no rank reconstructs halo cells.
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

class Mesh;

/** @brief A face side this rank reads but does not reconstruct: halo cell `cell`'s side of local face `face`. */
struct RemoteFaceSide {
    uint32_t face;
    uint32_t cell;
};

/** @brief Per neighbor rank, the face sides to send and to receive, as (face, side) pairs in message order. */
struct FacePlan {
    std::vector<int> neighbors;
    std::vector<std::vector<uint32_t>> send_faces, recv_faces;
    std::vector<std::vector<uint8_t>> send_sides, recv_sides;
};

/**
 * @brief The plan of a FaceExchange: each rank asks the owners of the halo
 *        cells in `needed` for those cells' sides of the faces, identified by
 *        the cell's global id and the face's index among the cell's faces
 *        (the same on every rank). Collective.
 */
FacePlan plan_face_exchange(const Distribution & dist, const Mesh & mesh, const std::vector<RemoteFaceSide> & needed);

/**
 * @brief Point-to-point exchange of reconstructed face states, as in
 *        HaloExchange (device buffers; GPU-aware MPI, host staging or NCCL).
 */
class FaceExchange {
    public:
        FaceExchange() = default;
        FaceExchange(const FacePlan & plan, uint8_t n_quad, bool nccl);

        /**
         * @brief Pack the requested sides of face_solution on exec and send them.
         *        With MPI the host waits for exec's work so far; with NCCL the
         *        messages are enqueued on exec's stream.
         */
        void start(const Kokkos::DefaultExecutionSpace & exec, Kokkos::View<rtype **[2][N_CONSERVATIVE]> face_solution);

        /** @brief Wait for the messages and write the received sides of face_solution on exec. */
        void finish(const Kokkos::DefaultExecutionSpace & exec, Kokkos::View<rtype **[2][N_CONSERVATIVE]> face_solution);

        bool active() const { return !ranks.empty(); }

    private:
        std::vector<int> ranks;
        bool nccl = false;
        uint32_t n_values = 0;  // per face side: quadrature points times states
        std::vector<uint32_t> send_offsets, recv_offsets;  // per neighbor, in face sides
        Kokkos::View<uint32_t *> send_faces, recv_faces;
        Kokkos::View<uint8_t *> send_sides, recv_sides;
        Kokkos::View<rtype *> send_buffer, recv_buffer;
        Kokkos::View<rtype *>::host_mirror_type h_send_buffer, h_recv_buffer;
#ifdef Mallard_HAS_MPI
        std::vector<MPI_Request> requests;
#endif
};

#endif // FACE_EXCHANGE_H

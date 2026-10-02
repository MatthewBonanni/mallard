/**
 * @file halo_exchange.h
 * @author Matthew Bonanni (mbonanni001@gmail.com)
 * @brief Fills halo cells with their owners' values.
 * @version 0.3
 * @date 2026-10-02
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 *
 */

#ifndef HALO_EXCHANGE_H
#define HALO_EXCHANGE_H

#include <cstdint>
#include <vector>

#include <Kokkos_Core.hpp>

#include "common_typedef.h"
#include "distribution.h"

/**
 * @brief Point-to-point exchange of per-cell state following a Distribution's
 *        plan. Buffers are packed on the device; they go to MPI directly when
 *        host-accessible or with Mallard_GPU_AWARE_MPI, else through host copies.
 */
class HaloExchange {
    public:
        HaloExchange() = default;
        explicit HaloExchange(const Distribution & dist);

        /** @brief Overwrite the halo cells of U with the owners' values (collective among neighbors). */
        void exchange(Kokkos::View<rtype *[N_CONSERVATIVE]> U) const;

        bool active() const { return !ranks.empty(); }

    private:
        std::vector<int> ranks;
        std::vector<uint32_t> send_offsets, recv_offsets;  // per neighbor, in cells
        Kokkos::View<uint32_t *> send_cells, recv_cells;
        Kokkos::View<rtype *> send_buffer, recv_buffer;
        Kokkos::View<rtype *>::host_mirror_type h_send_buffer, h_recv_buffer;
};

#endif // HALO_EXCHANGE_H

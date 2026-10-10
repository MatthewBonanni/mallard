/**
 * @file chemistry_balance.h
 * @author Matthew Bonanni (mbonanni001@gmail.com)
 * @brief Balancing of the chemistry integration across ranks by moving the
 *        states of queued cells, not cells (docs/design/chemistry.md,
 *        milestone 11).
 * @version 0.1
 * @date 2026-10-09
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 *
 */

#ifndef CHEMISTRY_BALANCE_H
#define CHEMISTRY_BALANCE_H

#include <cstdint>
#include <functional>
#include <vector>

#include <Kokkos_Core.hpp>

#include "cell_chemistry.h"
#include "comm.h"

/**
 * @brief Moves the integration of queued cells from ranks with more chemistry
 *        work than the mean to ranks with less, in every chemistry call.
 *
 * A rank's work is the sum over its queued cells of the sub-steps each took
 * in its last call (at least one). The ranks gather these sums; where the
 * largest exceeds the mean by more than the threshold, each rank above the
 * mean sends the cells at the end of its queue (the cheapest, when the queue
 * is ordered by cost) to ranks below it, pairing them in rank order: every
 * rank computes the same plan from the same numbers. A sent cell's inputs
 * (conservatives, partial densities, temperature seed, last sub-step, last
 * cost, multiplier of dt, and with a forcing its species rates and U + dt
 * rate) travel as raw bits, the receiver integrates it with the same kernel
 * as its own cells, and its partial densities (those of U + dt rate with a
 * forcing), last sub-step and sub-steps come back. A cell's result does not
 * depend on where it is integrated, so results stay bitwise independent of the
 * balancing.
 */
class ChemistryBalance {
    public:
        using Integrate = std::function<uint32_t(const ChemistryCells &, const Kokkos::View<uint32_t *> &,
                                                 const Kokkos::View<float *> &, uint32_t)>;

        ChemistryBalance(uint32_t n_species, double threshold);

        /**
         * @brief Plans the call (collective) and sends the states of the cells
         *        this rank gives away, the end of its queue [0, n_queued).
         * @return The queued cells this rank integrates itself: [0, kept).
         */
        uint32_t send(const ChemistryCells & cells, const Kokkos::View<uint32_t *> & queue, uint32_t n_queued);

        /**
         * @brief Receives the cells sent to this rank, integrates them with
         *        `integrate` (guest cells, their queue, its cost keys, count),
         *        returns their results and takes back the results of the cells
         *        this rank sent. Returns the failures among the received cells.
         */
        uint32_t receive(const ChemistryCells & cells, const Kokkos::View<uint32_t *> & queue, const Integrate & integrate);

    private:
        struct Transfer {
            int rank;
            uint32_t first, count;  // cells: queue slots (sent) or guest slots (received)
        };

        void reserve_guests(uint32_t n);

        uint32_t n_species;
        double threshold;
        uint32_t state_stride = 0, result_stride;  // rtype values per cell in the messages
        uint32_t kept = 0, n_sent = 0, n_guests = 0;
        std::vector<Transfer> sends, receives;
        Kokkos::View<rtype *> send_buffer, guest_buffer;  // states, then results in place
        Kokkos::View<rtype *>::host_mirror_type h_send_buffer, h_guest_buffer;
        ChemistryCells guests;
        Kokkos::View<uint32_t *> guest_queue;
        Kokkos::View<float *> guest_cost;
        bool scaled = false;  // the cells carry a multiplier of dt
        bool forced = false;  // and a forcing
#ifdef Mallard_HAS_MPI
        std::vector<MPI_Request> requests;
#endif
};

#endif // CHEMISTRY_BALANCE_H

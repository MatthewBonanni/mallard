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
#include "device_comm.h"

/**
 * @brief Moves the integration of queued cells from ranks with more chemistry
 *        work than the mean to ranks with less, in every chemistry call.
 *
 * A rank's work is the sum over its queued cells of the sub-steps each took
 * in its last call (at least one). The ranks gather these sums; where the
 * largest exceeds the mean by more than the threshold, each rank above the
 * mean sends the cells at the end of its queue (the cheapest, when the queue
 * is ordered by cost) to ranks below it, pairing them in rank order: every
 * rank computes the same plan from the same numbers, on the device. A sent
 * cell's inputs (conservatives, partial densities, temperature seed, last
 * sub-step, last cost, multiplier of dt, and with a forcing its species rates
 * and U + dt rate) travel as raw bits, the receiver integrates it with the
 * same kernel as its own cells, and its partial densities (those of U + dt
 * rate with a forcing), last sub-step and sub-steps come back. A cell's
 * result does not depend on where it is integrated, so results stay bitwise
 * independent of the balancing.
 *
 * Each pair of ranks has a capacity: the most cells one sends the other in a
 * call. The plan sends no more; the rest stay in the sender's queue. The
 * messages hold the capacity's slots (NCCL) or the planned cells (MPI), so
 * that no call waits for the host: the device records the most cells each
 * pair asked for, and adapt() sets the capacities from them between steps
 * (every rank alike, from the same numbers), growing at once and shrinking
 * after a quiet window. Pairs of capacity 0 exchange nothing.
 */
class ChemistryBalance {
    public:
        using Integrate = std::function<void(const ChemistryCells &, const Kokkos::View<uint32_t *> &,
                                             const Kokkos::View<float *> &, uint32_t, const Kokkos::View<uint32_t *> &)>;

        /** @param use_nccl Whether to gather the loads and move the cells with NCCL (else MPI). */
        ChemistryBalance(uint32_t n_species, double threshold, bool use_nccl);

        /**
         * @brief Plans the call (collective) and sends the states of the cells
         *        this rank gives away, the end of its queue [0, range(2)).
         *        Sets range(2) to the cells it integrates itself, and adds the
         *        cells sent to sent.
         */
        void send(const ChemistryCells & cells, const Kokkos::View<uint32_t *> & queue,
                  const Kokkos::View<uint32_t *> & range, const Kokkos::View<uint64_t *> & tally, int sent);

        /**
         * @brief Receives the cells sent to this rank, integrates them with
         *        `integrate` (guest cells, their queue, its cost keys, slots,
         *        range), returns their results and takes back the results of
         *        the cells this rank sent.
         */
        void receive(const ChemistryCells & cells, const Kokkos::View<uint32_t *> & queue, const Integrate & integrate);

        /** @brief Starts a step: the most cells asked for are those of its calls from now on. */
        void begin_step();

        /**
         * @brief Between steps, after a fence: sets the capacities from the
         *        cells asked for (collective by its inputs alone). Returns
         *        whether they changed, which changes the messages of the next
         *        calls. after_last_call: no calls ran since this step's;
         *        otherwise (a step enqueued ahead, which grew nothing and did
         *        not end the window) it only counts the step.
         */
        bool adapt(bool after_last_call = true);

        /** @brief Steps until a window ends, at whose last step adapt() may shrink capacities. */
        uint32_t steps_to_window_end() const;

        /** @brief Device flag: a pair asked for more cells than its capacity since the last adapt(). */
        const uint32_t * grows_flag() const { return grows.data(); }

        /** @brief Slots of the cells this rank may receive per call. */
        uint32_t guests_capacity() const { return n_in; }

        // Public because nvcc rejects device lambdas in non-public member functions
        void plan(const Kokkos::View<uint32_t *> & range);
        void pack(const ChemistryCells & cells, const Kokkos::View<uint32_t *> & queue);
        void unpack();
        void pack_results();
        void take_results(const ChemistryCells & cells, const Kokkos::View<uint32_t *> & queue);

    private:
        /** @brief Whether MPI messages go through host copies of the buffers. */
        bool staged() const;
        /** @brief Messages of each pair with slots (NCCL) or planned cells (MPI), stride values per cell. */
        std::vector<comm::DeviceMessage> messages(bool out, rtype * data, uint32_t stride) const;
        /** @brief Sends out and receives in with NCCL, or posts them with MPI into requests. */
        void transfer(const std::vector<comm::DeviceMessage> & out, const std::vector<comm::DeviceMessage> & in, int tag);
        void reserve_guests(uint32_t n);
        void resize();

        uint32_t n_species;
        double threshold;
        bool use_nccl;
        int n_ranks = 1, me = 0;
        Kokkos::View<double *> load, loads;  // [load, queued] of this rank, of every rank
        Kokkos::View<double *> excess;       // plan scratch, per rank
        uint32_t state_stride = 0, result_stride;  // rtype values per cell in the messages
        // Capacities (sender * n_ranks + receiver) and this rank's slots: base of each receiver, each sender
        std::vector<uint32_t> cap, out_base, in_base;
        uint32_t n_out = 0, n_in = 0;  // slots
        Kokkos::View<uint32_t *> d_cap, d_out_base, d_in_base, out_rank, in_rank;
        // Plan: [kept, guests, sent to each rank, first queue slot sent to each, received from each, offset of each]
        Kokkos::View<uint32_t *> d_plan;
        Kokkos::View<uint32_t *>::host_mirror_type h_plan;
        // Most cells each pair asked for in this step's calls, on the device and (copied after each call) on the host
        Kokkos::View<uint32_t *> asked;
        Kokkos::View<uint32_t *, Kokkos::SharedHostPinnedSpace> asked_host;
        std::vector<uint32_t> window_max;  // most asked for since the last shrink, per pair
        Kokkos::View<uint32_t *> window_asked;  // window_max, kept by the device
        Kokkos::View<uint32_t> grows;
        uint32_t window_steps = 0;
        Kokkos::View<rtype *> send_buffer, guest_buffer;  // states, then results in place
        Kokkos::View<rtype *>::host_mirror_type h_send_buffer, h_guest_buffer;
        ChemistryCells guests;
        Kokkos::View<uint32_t *> guest_queue, guest_range;
        Kokkos::View<float *> guest_cost;
        bool scaled = false;  // the cells carry a multiplier of dt
        bool forced = false;  // and a forcing
#ifdef Mallard_HAS_MPI
        std::vector<MPI_Request> requests;
#endif
};

#endif // CHEMISTRY_BALANCE_H

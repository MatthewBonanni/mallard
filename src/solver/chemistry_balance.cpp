/**
 * @file chemistry_balance.cpp
 * @author Matthew Bonanni (mbonanni001@gmail.com)
 * @brief Balancing of the chemistry integration across ranks.
 * @version 0.1
 * @date 2026-10-09
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 *
 */

#include "chemistry_balance.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>

namespace {

#ifdef Mallard_HAS_MPI
#ifdef Mallard_GPU_AWARE_MPI
constexpr bool stage_through_host = false;
#else
constexpr bool stage_through_host =
    !Kokkos::SpaceAccessibility<Kokkos::HostSpace, Kokkos::DefaultExecutionSpace::memory_space>::accessible;
#endif
constexpr int STATE_TAG = 1410, RESULT_TAG = 1411;

MPI_Datatype rtype_mpi() { return sizeof(rtype) == sizeof(double) ? MPI_DOUBLE : MPI_FLOAT; }

void wait(std::vector<MPI_Request> & requests) {
    if (MPI_Waitall(static_cast<int>(requests.size()), requests.data(), MPI_STATUSES_IGNORE) != MPI_SUCCESS) {
        throw std::runtime_error("ChemistryBalance: MPI_Waitall failed");
    }
    requests.clear();
}
#endif

template <typename View>
void reserve(View & v, const size_t n, const char * label) {
    if (v.extent(0) < n) v = View(Kokkos::view_alloc(Kokkos::WithoutInitializing, std::string(label)), n + n / 4);
}

} // namespace

ChemistryBalance::ChemistryBalance(const uint32_t n_species_in, const double threshold_in, const bool use_nccl_in)
    : n_species(n_species_in), threshold(threshold_in), use_nccl(use_nccl_in), load("chemistry_balance_load", 2),
      loads("chemistry_balance_loads", 2 * static_cast<size_t>(comm::size())), result_stride(n_species_in + 4) {
#ifndef Mallard_HAS_MPI
    (void)result_stride;
    throw std::logic_error("ChemistryBalance: built without MPI");
#endif
}

void ChemistryBalance::reserve_guests(const uint32_t n) {
    if (guests.U.extent(0) < n) {
        const size_t m = n + n / 4;
        guests.U = StateView(Kokkos::view_alloc(Kokkos::WithoutInitializing, "chem_guest_U"), m);
        guests.rhoY = SpeciesView(Kokkos::view_alloc(Kokkos::WithoutInitializing, "chem_guest_rhoY"), m, n_species);
        guests.T_seed = Kokkos::View<rtype *>(Kokkos::view_alloc(Kokkos::WithoutInitializing, "chem_guest_T_seed"), m);
        guests.chem_h = Kokkos::View<rtype *>(Kokkos::view_alloc(Kokkos::WithoutInitializing, "chem_guest_h"), m);
        guests.chem_cost = Kokkos::View<rtype *>(Kokkos::view_alloc(Kokkos::WithoutInitializing, "chem_guest_cost"), m);
        guests.previous_cost =
            Kokkos::View<float *>(Kokkos::view_alloc(Kokkos::WithoutInitializing, "chem_guest_previous_cost"), m);
        guest_queue = Kokkos::View<uint32_t *>(Kokkos::view_alloc(Kokkos::WithoutInitializing, "chem_guest_queue"), m);
        guest_cost = Kokkos::View<float *>(Kokkos::view_alloc(Kokkos::WithoutInitializing, "chem_guest_queue_cost"), m);
        guests.time_scale = Kokkos::View<rtype *>();
        guests.volume = Kokkos::View<rtype *>(Kokkos::view_alloc(Kokkos::WithoutInitializing, "chem_guest_volume"), m);
        guests.repaired = Kokkos::View<uint8_t *>("chem_guest_repaired", m);
    }
    const size_t m = guests.U.extent(0);
    if (scaled && guests.time_scale.extent(0) < m) {
        guests.time_scale = Kokkos::View<rtype *>(Kokkos::view_alloc(Kokkos::WithoutInitializing, "chem_guest_scale"), m);
    }
    if (!scaled) guests.time_scale = Kokkos::View<rtype *>();
    if (forced && guests.U_end.extent(0) < m) {
        guests.rate = SpeciesView(Kokkos::view_alloc(Kokkos::WithoutInitializing, "chem_guest_rate"), m, n_species);
        guests.U_end = StateView(Kokkos::view_alloc(Kokkos::WithoutInitializing, "chem_guest_U_end"), m);
        guests.rhoY_end = SpeciesView(Kokkos::view_alloc(Kokkos::WithoutInitializing, "chem_guest_rhoY_end"), m, n_species);
    }
    if (!forced) {
        guests.rate = SpeciesView();
        guests.U_end = StateView();
        guests.rhoY_end = SpeciesView();
    }
}

uint32_t ChemistryBalance::send(const ChemistryCells & cells, const Kokkos::View<uint32_t *> & queue,
                                const uint32_t n_queued) {
    sends.clear();
    receives.clear();
    kept = n_queued;
    n_sent = 0;
    n_guests = 0;
    const Kokkos::View<float *> previous = cells.previous_cost;
    using DeviceMemory = Kokkos::DefaultExecutionSpace::memory_space;
    const Kokkos::View<double *> mine = load;
    Kokkos::parallel_reduce("chemistry_balance_load", n_queued, KOKKOS_LAMBDA(const uint32_t i, double & sum) {
        sum += Kokkos::fmax(1.0, static_cast<double>(previous(queue(i))));
    }, Kokkos::Sum<double, DeviceMemory>(Kokkos::View<double, DeviceMemory>(mine.data())));
    Kokkos::parallel_for("chemistry_balance_queued", 1, KOKKOS_LAMBDA(const int) { mine(1) = static_cast<double>(n_queued); });
    comm::allgather_device(load, loads, use_nccl);
    const auto all = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), loads);
    const size_t n_ranks = static_cast<size_t>(comm::size()), me = static_cast<size_t>(comm::rank());
    double total = 0.0, largest = 0.0;
    for (size_t r = 0; r < n_ranks; r++) {
        total += all[2 * r];
        largest = std::max(largest, all[2 * r]);
    }
    const double mean = total / static_cast<double>(n_ranks);
    if (total <= 0.0 || largest <= (1.0 + threshold) * mean) return kept;

    // Ranks above the mean give to ranks below it, both taken in rank order
    std::vector<double> excess(n_ranks);
    std::vector<uint32_t> given(n_ranks, 0);
    for (size_t r = 0; r < n_ranks; r++) excess[r] = all[2 * r] - mean;
    for (size_t s = 0, d = 0;;) {
        while (s < n_ranks && excess[s] <= 0.0) s++;
        while (d < n_ranks && excess[d] >= 0.0) d++;
        if (s == n_ranks || d == n_ranks) break;
        const double amount = std::min(excess[s], -excess[d]);
        const uint32_t queued = static_cast<uint32_t>(all[2 * s + 1]);
        const uint32_t count = std::min(static_cast<uint32_t>(std::floor(amount * queued / all[2 * s])), queued - given[s]);
        if (count > 0) {
            given[s] += count;
            if (s == me) sends.push_back({static_cast<int>(d), queued - given[s], count});
            if (d == me) {
                receives.push_back({static_cast<int>(s), n_guests, count});
                n_guests += count;
            }
        }
        excess[s] -= amount;
        excess[d] += amount;
    }
    kept = n_queued - given[me];
    n_sent = given[me];
    scaled = cells.time_scale.extent(0) > 0;
    forced = cells.forced();
    state_stride = N_CONSERVATIVE + n_species + 5 + (forced ? n_species + N_CONSERVATIVE : 0);

#ifdef Mallard_HAS_MPI
    const uint32_t S = state_stride, ns = n_species, first = kept;
    const bool has_scale = scaled, has_forcing = forced;
    if (n_sent > 0) {
        reserve(send_buffer, static_cast<size_t>(n_sent) * S, "chem_balance_send");
        const Kokkos::View<rtype *> buf = send_buffer;
        const ChemistryCells c = cells;
        Kokkos::parallel_for("chemistry_balance_pack", n_sent, KOKKOS_LAMBDA(const uint32_t k) {
            const uint32_t cell = queue(first + k);
            rtype * b = &buf(static_cast<size_t>(k) * S);
            for (uint32_t i = 0; i < N_CONSERVATIVE; i++) b[i] = c.U(cell, i);
            for (uint32_t j = 0; j < ns; j++) b[N_CONSERVATIVE + j] = c.rhoY(cell, j);
            b[N_CONSERVATIVE + ns] = c.T_seed(cell);
            b[N_CONSERVATIVE + ns + 1] = c.chem_h(cell);
            b[N_CONSERVATIVE + ns + 2] = static_cast<rtype>(c.previous_cost(cell));
            b[N_CONSERVATIVE + ns + 3] = has_scale ? c.time_scale(cell) : rtype(1);
            if (has_forcing) {
                rtype * f = b + N_CONSERVATIVE + ns + 4;
                for (uint32_t j = 0; j < ns; j++) f[j] = c.rate(cell, j);
                for (uint32_t i = 0; i < N_CONSERVATIVE; i++) f[ns + i] = c.U_end(cell, i);
            }
            b[S - 1] = c.volume.extent(0) > 0 ? c.volume(cell) : rtype(0);
        });
        if (!use_nccl) Kokkos::fence("chemistry_balance_pack");
    }
    if (n_guests > 0) reserve(guest_buffer, static_cast<size_t>(n_guests) * S, "chem_balance_guests");
    if (staged()) {
        h_send_buffer = Kokkos::create_mirror_view(send_buffer);
        h_guest_buffer = Kokkos::create_mirror_view(guest_buffer);
        if (n_sent > 0) Kokkos::deep_copy(h_send_buffer, send_buffer);
    }
    transfer(messages(sends, send_ptr(), S, kept), messages(receives, guest_ptr(), S, 0), STATE_TAG);
    // Completed now: large GPU messages progress only inside MPI calls, which the sender would
    // otherwise make only after integrating its own cells, leaving the receivers idle meanwhile
    if (!use_nccl) {
        wait(requests);
        // The results come back in place of the states sent
        transfer({}, messages(sends, send_ptr(), result_stride, kept), RESULT_TAG);
    }
#endif
    return kept;
}

bool ChemistryBalance::staged() const {
#ifdef Mallard_HAS_MPI
    return stage_through_host && !use_nccl;
#else
    return false;
#endif
}

rtype * ChemistryBalance::send_ptr() const { return staged() ? h_send_buffer.data() : send_buffer.data(); }

rtype * ChemistryBalance::guest_ptr() const { return staged() ? h_guest_buffer.data() : guest_buffer.data(); }

std::vector<comm::DeviceMessage> ChemistryBalance::messages(const std::vector<Transfer> & transfers, rtype * data,
                                                            const uint32_t stride, const uint32_t offset) const {
    std::vector<comm::DeviceMessage> out;
    for (const Transfer & t : transfers) {
        out.push_back({t.rank, data + static_cast<size_t>(t.first - offset) * stride,
                       static_cast<size_t>(t.count) * stride * sizeof(rtype)});
    }
    return out;
}

void ChemistryBalance::transfer(const std::vector<comm::DeviceMessage> & out,
                                const std::vector<comm::DeviceMessage> & in, [[maybe_unused]] const int tag) {
    if (use_nccl) {
        comm::send_receive_device(out, in);
        return;
    }
#ifdef Mallard_HAS_MPI
    for (const comm::DeviceMessage & m : in) {
        requests.push_back(MPI_REQUEST_NULL);
        MPI_Irecv(m.data, static_cast<int>(m.bytes / sizeof(rtype)), rtype_mpi(), m.rank, tag, comm::world(),
                  &requests.back());
    }
    for (const comm::DeviceMessage & m : out) {
        requests.push_back(MPI_REQUEST_NULL);
        MPI_Isend(m.data, static_cast<int>(m.bytes / sizeof(rtype)), rtype_mpi(), m.rank, tag, comm::world(),
                  &requests.back());
    }
#endif
}

uint32_t ChemistryBalance::receive(const ChemistryCells & cells, const Kokkos::View<uint32_t *> & queue,
                                   const Integrate & integrate) {
    uint32_t failures = 0;
#ifdef Mallard_HAS_MPI
    if (sends.empty() && receives.empty()) return 0;
    const uint32_t S = state_stride, R = result_stride, ns = n_species;
    if (n_guests > 0) {
        reserve_guests(n_guests);
        if (staged()) Kokkos::deep_copy(guest_buffer, h_guest_buffer);
        const Kokkos::View<rtype *> buf = guest_buffer;
        const ChemistryCells g = guests;
        const Kokkos::View<uint32_t *> q = guest_queue;
        const Kokkos::View<float *> key = guest_cost;
        const bool has_scale = scaled, has_forcing = forced;
        Kokkos::parallel_for("chemistry_balance_unpack", n_guests, KOKKOS_LAMBDA(const uint32_t k) {
            const rtype * b = &buf(static_cast<size_t>(k) * S);
            for (uint32_t i = 0; i < N_CONSERVATIVE; i++) g.U(k, i) = b[i];
            for (uint32_t j = 0; j < ns; j++) g.rhoY(k, j) = b[N_CONSERVATIVE + j];
            g.T_seed(k) = b[N_CONSERVATIVE + ns];
            g.chem_h(k) = b[N_CONSERVATIVE + ns + 1];
            g.previous_cost(k) = static_cast<float>(b[N_CONSERVATIVE + ns + 2]);
            if (has_scale) g.time_scale(k) = b[N_CONSERVATIVE + ns + 3];
            if (has_forcing) {
                const rtype * f = b + N_CONSERVATIVE + ns + 4;
                for (uint32_t j = 0; j < ns; j++) g.rate(k, j) = f[j];
                for (uint32_t i = 0; i < N_CONSERVATIVE; i++) g.U_end(k, i) = f[ns + i];
            }
            g.volume(k) = b[S - 1];
            g.repaired(k) = 0;
            g.chem_cost(k) = rtype(0);
            q(k) = k;
            key(k) = -g.previous_cost(k);
        });
        failures = integrate(guests, guest_queue, guest_cost, n_guests);
        // Results in place of the states (each cell's results fit in its state's slot)
        Kokkos::parallel_for("chemistry_balance_results", n_guests, KOKKOS_LAMBDA(const uint32_t k) {
            rtype * b = &buf(static_cast<size_t>(k) * R);
            for (uint32_t j = 0; j < ns; j++) b[j] = has_forcing ? g.rhoY_end(k, j) : g.rhoY(k, j);
            b[ns] = g.chem_h(k);
            b[ns + 1] = g.chem_cost(k);
            b[ns + 2] = static_cast<rtype>(g.previous_cost(k));
            b[ns + 3] = static_cast<rtype>(g.repaired(k));
        });
        if (!use_nccl) Kokkos::fence("chemistry_balance_results");
        if (staged()) Kokkos::deep_copy(h_guest_buffer, guest_buffer);
    }
    if (use_nccl) {
        transfer(messages(receives, guest_ptr(), R, 0), messages(sends, send_ptr(), R, kept), RESULT_TAG);
    } else {
        transfer(messages(receives, guest_ptr(), R, 0), {}, RESULT_TAG);
        wait(requests);
    }
    if (n_sent > 0) {
        if (staged()) Kokkos::deep_copy(send_buffer, h_send_buffer);
        const Kokkos::View<rtype *> buf = send_buffer;
        const ChemistryCells c = cells;
        const uint32_t first = kept;
        const SpeciesView out = forced ? c.rhoY_end : c.rhoY;
        Kokkos::parallel_for("chemistry_balance_return", n_sent, KOKKOS_LAMBDA(const uint32_t k) {
            const uint32_t cell = queue(first + k);
            const rtype * b = &buf(static_cast<size_t>(k) * R);
            for (uint32_t j = 0; j < ns; j++) out(cell, j) = b[j];
            c.chem_h(cell) = b[ns];
            c.chem_cost(cell) += b[ns + 1];
            c.previous_cost(cell) = static_cast<float>(b[ns + 2]);
            if (b[ns + 3] != rtype(0) && c.repaired.extent(0) > 0) c.repaired(cell) = 1;
        });
    }
#else
    (void)cells;
    (void)queue;
    (void)integrate;
#endif
    return failures;
}

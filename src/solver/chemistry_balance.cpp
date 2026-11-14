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
#include <optional>
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

// Capacity for a pair that asked for n cells: room to grow, in whole blocks
constexpr uint32_t CAPACITY_BLOCK = 64;
uint32_t capacity_for(const uint32_t n) {
    if (n == 0) return 0;
    const uint64_t room = uint64_t(n) + uint64_t(n) / 4;
    return static_cast<uint32_t>((room + CAPACITY_BLOCK - 1) / CAPACITY_BLOCK * CAPACITY_BLOCK);
}
// Steps without a pair needing more than half its capacity before it shrinks
constexpr uint32_t SHRINK_STEPS = 256;

// Plan slots
enum : uint32_t { KEPT = 0, GUESTS = 1, PLAN_HEADER = 2 };

} // namespace

ChemistryBalance::ChemistryBalance(const uint32_t n_species_in, const double threshold_in, const bool use_nccl_in)
    : n_species(n_species_in), threshold(threshold_in), use_nccl(use_nccl_in), result_stride(n_species_in + 4) {
#ifndef Mallard_HAS_MPI
    (void)result_stride;
    throw std::logic_error("ChemistryBalance: built without MPI");
#endif
    n_ranks = comm::size();
    me = comm::rank();
    const size_t P = static_cast<size_t>(n_ranks);
    load = Kokkos::View<double *>("chemistry_balance_load", 2);
    loads = Kokkos::View<double *>("chemistry_balance_loads", 2 * P);
    excess = Kokkos::View<double *>("chemistry_balance_excess", P);
    cap.assign(P * P, 0);
    window_max.assign(P * P, 0);
    d_cap = Kokkos::View<uint32_t *>("chemistry_balance_capacity", P * P);
    d_out_base = Kokkos::View<uint32_t *>("chemistry_balance_out_base", P);
    d_in_base = Kokkos::View<uint32_t *>("chemistry_balance_in_base", P);
    d_plan = Kokkos::View<uint32_t *>("chemistry_balance_plan", PLAN_HEADER + 4 * P);
    h_plan = Kokkos::create_mirror_view(d_plan);
    asked = Kokkos::View<uint32_t *>("chemistry_balance_asked", P * P);
    asked_host = Kokkos::View<uint32_t *, Kokkos::SharedHostPinnedSpace>("chemistry_balance_asked_host", P * P);
    window_asked = Kokkos::View<uint32_t *>("chemistry_balance_window", P * P);
    grows = Kokkos::View<uint32_t>("chemistry_balance_grows");
    guest_range = Kokkos::View<uint32_t *>("chemistry_balance_guest_range", 3);
    resize();
}

void ChemistryBalance::resize() {
    const size_t P = static_cast<size_t>(n_ranks);
    out_base.assign(P, 0);
    in_base.assign(P, 0);
    n_out = n_in = 0;
    for (size_t r = 0; r < P; r++) {
        out_base[r] = n_out;
        n_out += cap[static_cast<size_t>(me) * P + r];
        in_base[r] = n_in;
        n_in += cap[r * P + static_cast<size_t>(me)];
    }
    const auto upload = [](const std::vector<uint32_t> & v, const Kokkos::View<uint32_t *> & d) {
        const Kokkos::View<const uint32_t *, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>> h(v.data(), v.size());
        Kokkos::deep_copy(d, h);
    };
    upload(cap, d_cap);
    upload(out_base, d_out_base);
    upload(in_base, d_in_base);
    // The rank of each slot
    std::vector<uint32_t> h_out(n_out), h_in(n_in);
    for (size_t r = 0; r < P; r++) {
        for (uint32_t k = 0; k < cap[static_cast<size_t>(me) * P + r]; k++) h_out[out_base[r] + k] = static_cast<uint32_t>(r);
        for (uint32_t k = 0; k < cap[r * P + static_cast<size_t>(me)]; k++) h_in[in_base[r] + k] = static_cast<uint32_t>(r);
    }
    out_rank = Kokkos::View<uint32_t *>("chemistry_balance_out_rank", n_out);
    in_rank = Kokkos::View<uint32_t *>("chemistry_balance_in_rank", n_in);
    upload(h_out, out_rank);
    upload(h_in, in_rank);
    const uint32_t S = state_stride > 0 ? state_stride : N_CONSERVATIVE + 2 * n_species + 5 + N_CONSERVATIVE;
    send_buffer = Kokkos::View<rtype *>(Kokkos::view_alloc(Kokkos::WithoutInitializing, std::string("chem_balance_send")),
                                        static_cast<size_t>(n_out) * S);
    guest_buffer = Kokkos::View<rtype *>(Kokkos::view_alloc(Kokkos::WithoutInitializing, std::string("chem_balance_guests")),
                                         static_cast<size_t>(n_in) * S);
    if (staged()) {
        h_send_buffer = Kokkos::create_mirror_view(send_buffer);
        h_guest_buffer = Kokkos::create_mirror_view(guest_buffer);
    }
    reserve_guests(n_in);
}

void ChemistryBalance::reserve_guests(const uint32_t n) {
    const size_t m = n;
    guests.U = StateView(Kokkos::view_alloc(Kokkos::WithoutInitializing, "chem_guest_U"), m);
    guests.rhoY = SpeciesView(Kokkos::view_alloc(Kokkos::WithoutInitializing, "chem_guest_rhoY"), m, n_species);
    guests.T_seed = Kokkos::View<rtype *>(Kokkos::view_alloc(Kokkos::WithoutInitializing, "chem_guest_T_seed"), m);
    guests.chem_h = Kokkos::View<rtype *>(Kokkos::view_alloc(Kokkos::WithoutInitializing, "chem_guest_h"), m);
    guests.chem_cost = Kokkos::View<rtype *>(Kokkos::view_alloc(Kokkos::WithoutInitializing, "chem_guest_cost"), m);
    guests.previous_cost =
        Kokkos::View<float *>(Kokkos::view_alloc(Kokkos::WithoutInitializing, "chem_guest_previous_cost"), m);
    guest_queue = Kokkos::View<uint32_t *>(Kokkos::view_alloc(Kokkos::WithoutInitializing, "chem_guest_queue"), m);
    guest_cost = Kokkos::View<float *>(Kokkos::view_alloc(Kokkos::WithoutInitializing, "chem_guest_queue_cost"), m);
    guests.volume = Kokkos::View<rtype *>(Kokkos::view_alloc(Kokkos::WithoutInitializing, "chem_guest_volume"), m);
    guests.repaired = Kokkos::View<uint8_t *>("chem_guest_repaired", m);
    guests.time_scale =
        scaled ? Kokkos::View<rtype *>(Kokkos::view_alloc(Kokkos::WithoutInitializing, "chem_guest_scale"), m)
               : Kokkos::View<rtype *>();
    if (forced) {
        guests.rate = SpeciesView(Kokkos::view_alloc(Kokkos::WithoutInitializing, "chem_guest_rate"), m, n_species);
        guests.U_end = StateView(Kokkos::view_alloc(Kokkos::WithoutInitializing, "chem_guest_U_end"), m);
        guests.rhoY_end = SpeciesView(Kokkos::view_alloc(Kokkos::WithoutInitializing, "chem_guest_rhoY_end"), m, n_species);
    } else {
        guests.rate = SpeciesView();
        guests.U_end = StateView();
        guests.rhoY_end = SpeciesView();
    }
}

void ChemistryBalance::begin_step() {
    Kokkos::deep_copy(Kokkos::DefaultExecutionSpace(), asked, 0u);
}

uint32_t ChemistryBalance::steps_to_window_end() const { return SHRINK_STEPS - window_steps; }

bool ChemistryBalance::adapt(const bool after_last_call) {
    const size_t P = static_cast<size_t>(n_ranks);
    bool changed = false;
    window_steps++;
    const bool window_over = window_steps >= SHRINK_STEPS;
    if (!after_last_call) {
        if (window_over) throw std::logic_error("chemistry balance: the window ended at a queued step");
        return false;
    }
    const auto window = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), window_asked);
    Kokkos::deep_copy(grows, 0u);
    for (size_t k = 0; k < P * P; k++) {
        const uint32_t n = asked_host(k);
        window_max[k] = window(k);
        if (n > cap[k]) {
            cap[k] = capacity_for(n);
            changed = true;
        } else if (window_over && 2 * uint64_t(window_max[k]) < cap[k]) {
            cap[k] = capacity_for(window_max[k]);
            changed = true;
        }
    }
    if (window_over) {
        window_steps = 0;
        std::fill(window_max.begin(), window_max.end(), 0u);
        Kokkos::deep_copy(window_asked, 0u);
    }
    if (changed) resize();
    return changed;
}

void ChemistryBalance::plan(const Kokkos::View<uint32_t *> & range) {
    const Kokkos::View<double *> all = loads, ex = excess;
    const Kokkos::View<uint32_t *> c = d_cap, p = d_plan, ask = asked, r = range, window = window_asked;
    const Kokkos::View<uint32_t> grow = grows;
    const uint32_t P = static_cast<uint32_t>(n_ranks), self = static_cast<uint32_t>(me);
    const double thr = threshold;
    Kokkos::parallel_for("chemistry_balance_plan", 1, KOKKOS_LAMBDA(const int) {
        const uint32_t n_queued = r(2);
        p(KEPT) = n_queued;
        p(GUESTS) = 0;
        for (uint32_t q = 0; q < 4 * P; q++) p(PLAN_HEADER + q) = 0;
        double total = 0.0, largest = 0.0;
        for (uint32_t q = 0; q < P; q++) {
            total += all(2 * q);
            largest = Kokkos::fmax(largest, all(2 * q));
        }
        const double mean = total / static_cast<double>(P);
        if (total <= 0.0 || largest <= (1.0 + thr) * mean) return;
        // Ranks above the mean give to ranks below it, both taken in rank order
        for (uint32_t q = 0; q < P; q++) ex(q) = all(2 * q) - mean;
        uint32_t given_me = 0, given_s = 0, s_prev = P;
        for (uint32_t s = 0, d = 0;;) {
            while (s < P && ex(s) <= 0.0) s++;
            while (d < P && ex(d) >= 0.0) d++;
            if (s == P || d == P) break;
            if (s != s_prev) {
                given_s = 0;
                s_prev = s;
            }
            const double amount = Kokkos::fmin(ex(s), -ex(d));
            const uint32_t queued = static_cast<uint32_t>(all(2 * s + 1));
            const double share = Kokkos::floor(amount * queued / all(2 * s));
            const uint32_t wanted = Kokkos::min(static_cast<uint32_t>(share), queued - given_s);
            if (wanted > ask(s * P + d)) ask(s * P + d) = wanted;
            if (wanted > window(s * P + d)) window(s * P + d) = wanted;
            if (wanted > c(s * P + d)) grow() = 1;
            const uint32_t count = Kokkos::min(wanted, c(s * P + d));
            if (count > 0) {
                given_s += count;
                if (s == self) {
                    given_me = given_s;
                    p(PLAN_HEADER + d) = count;
                    p(PLAN_HEADER + P + d) = queued - given_s;
                }
                if (d == self) {
                    p(PLAN_HEADER + 2 * P + s) = count;
                    p(PLAN_HEADER + 3 * P + s) = p(GUESTS);
                    p(GUESTS) += count;
                }
            }
            ex(s) -= amount;
            ex(d) += amount;
        }
        p(KEPT) = n_queued - given_me;
    });
}

void ChemistryBalance::send(const ChemistryCells & cells, const Kokkos::View<uint32_t *> & queue,
                            const Kokkos::View<uint32_t *> & range, const Kokkos::View<uint64_t *> & tally,
                            const int sent) {
    scaled = cells.time_scale.extent(0) > 0;
    forced = cells.forced();
    const uint32_t S = N_CONSERVATIVE + n_species + 5 + (forced ? n_species + N_CONSERVATIVE : 0);
    if (S != state_stride) {
        state_stride = S;
        resize();
    }
    const Kokkos::View<float *> previous = cells.previous_cost;
    using DeviceMemory = Kokkos::DefaultExecutionSpace::memory_space;
    const Kokkos::View<double *> mine = load;
    const Kokkos::View<uint32_t *> r = range;
    Kokkos::parallel_reduce("chemistry_balance_load", queue.extent(0), KOKKOS_LAMBDA(const uint32_t i, double & sum) {
        if (i < r(2)) sum += Kokkos::fmax(1.0, static_cast<double>(previous(queue(i))));
    }, Kokkos::Sum<double, DeviceMemory>(Kokkos::View<double, DeviceMemory>(mine.data())));
    Kokkos::parallel_for("chemistry_balance_queued", 1, KOKKOS_LAMBDA(const int) { mine(1) = static_cast<double>(r(2)); });
    comm::allgather_device(load, loads, use_nccl);
    plan(range);
    Kokkos::deep_copy(Kokkos::DefaultExecutionSpace(), asked_host, asked);
    const Kokkos::View<uint32_t *> p = d_plan;
    const Kokkos::View<uint64_t *> t = tally;
    Kokkos::parallel_for("chemistry_balance_kept", 1, KOKKOS_LAMBDA(const int) {
        t(sent) += r(2) - p(KEPT);
        r(2) = p(KEPT);
        if (r(1) > r(2)) r(1) = r(2);
    });
    if (n_out == 0 && n_in == 0) return;
#ifdef Mallard_HAS_MPI
    // MPI's messages need the plan and the states on the host
    std::optional<comm::HostStaged> host_staged;
    if (!use_nccl) {
        host_staged.emplace();
        Kokkos::deep_copy(h_plan, d_plan);
    }
    if (n_out > 0) {
        pack(cells, queue);
        if (!use_nccl) Kokkos::fence("chemistry_balance_pack");
    }
    if (staged() && n_out > 0) Kokkos::deep_copy(h_send_buffer, send_buffer);
    rtype * out = staged() ? h_send_buffer.data() : send_buffer.data();
    rtype * in = staged() ? h_guest_buffer.data() : guest_buffer.data();
    transfer(messages(true, out, S), messages(false, in, S), STATE_TAG);
    // Completed now: large GPU messages progress only inside MPI calls, which the sender would
    // otherwise make only after integrating its own cells, leaving the receivers idle meanwhile
    if (!use_nccl) {
        wait(requests);
        // The results come back in place of the states sent
        transfer({}, messages(true, out, result_stride), RESULT_TAG);
    }
#endif
}

void ChemistryBalance::pack(const ChemistryCells & cells, const Kokkos::View<uint32_t *> & queue) {
    const uint32_t S = state_stride, ns = n_species, P = static_cast<uint32_t>(n_ranks);
    const bool has_scale = scaled, has_forcing = forced;
    const Kokkos::View<rtype *> buf = send_buffer;
    const Kokkos::View<uint32_t *> p = d_plan, rank = out_rank, base = d_out_base;
    const ChemistryCells c = cells;
    Kokkos::parallel_for("chemistry_balance_pack", n_out, KOKKOS_LAMBDA(const uint32_t slot) {
        const uint32_t d = rank(slot), k = slot - base(d);
        if (k >= p(PLAN_HEADER + d)) return;
        const uint32_t cell = queue(p(PLAN_HEADER + P + d) + k);
        rtype * b = &buf(static_cast<size_t>(slot) * S);
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
}

bool ChemistryBalance::staged() const {
#ifdef Mallard_HAS_MPI
    return stage_through_host && !use_nccl;
#else
    return false;
#endif
}

std::vector<comm::DeviceMessage> ChemistryBalance::messages(const bool out, rtype * data, const uint32_t stride) const {
    // out: this rank's slots of each receiver; else those of each sender
    std::vector<comm::DeviceMessage> list;
    const size_t P = static_cast<size_t>(n_ranks), self = static_cast<size_t>(me);
    for (size_t r = 0; r < P; r++) {
        const uint32_t slots = out ? cap[self * P + r] : cap[r * P + self];
        if (slots == 0) continue;
        // MPI sends the planned cells only
        const uint32_t count = use_nccl ? slots : h_plan(PLAN_HEADER + (out ? 0 : 2 * P) + r);
        if (count == 0) continue;
        const uint32_t base = out ? out_base[r] : in_base[r];
        list.push_back({static_cast<int>(r), data + static_cast<size_t>(base) * stride,
                        static_cast<size_t>(count) * stride * sizeof(rtype)});
    }
    return list;
}

void ChemistryBalance::transfer(const std::vector<comm::DeviceMessage> & out,
                                const std::vector<comm::DeviceMessage> & in, [[maybe_unused]] const int tag) {
    if (use_nccl) {
        comm::send_receive_device(out, in);
        return;
    }
#ifdef Mallard_HAS_MPI
    for (const comm::DeviceMessage & m : in) {
        comm::irecv(static_cast<rtype *>(m.data), m.bytes / sizeof(rtype), rtype_mpi(), m.rank, tag, comm::communicator(),
                    requests);
    }
    for (const comm::DeviceMessage & m : out) {
        comm::isend(static_cast<const rtype *>(m.data), m.bytes / sizeof(rtype), rtype_mpi(), m.rank, tag, comm::communicator(),
                    requests);
    }
#endif
}

void ChemistryBalance::unpack() {
    const uint32_t S = state_stride, ns = n_species, P = static_cast<uint32_t>(n_ranks);
    const bool has_scale = scaled, has_forcing = forced;
    const Kokkos::View<rtype *> buf = guest_buffer;
    const ChemistryCells g = guests;
    const Kokkos::View<uint32_t *> q = guest_queue, p = d_plan, rank = in_rank, base = d_in_base, gr = guest_range;
    const Kokkos::View<float *> key = guest_cost;
    Kokkos::parallel_for("chemistry_balance_guest_range", 1, KOKKOS_LAMBDA(const int) {
        gr(0) = 0;
        gr(1) = 0;
        gr(2) = p(GUESTS);
    });
    Kokkos::parallel_for("chemistry_balance_unpack", n_in, KOKKOS_LAMBDA(const uint32_t k) {
        const uint32_t s = rank(k), j = k - base(s);
        if (j >= p(PLAN_HEADER + 2 * P + s)) return;
        const rtype * b = &buf(static_cast<size_t>(k) * S);
        for (uint32_t i = 0; i < N_CONSERVATIVE; i++) g.U(k, i) = b[i];
        for (uint32_t m = 0; m < ns; m++) g.rhoY(k, m) = b[N_CONSERVATIVE + m];
        g.T_seed(k) = b[N_CONSERVATIVE + ns];
        g.chem_h(k) = b[N_CONSERVATIVE + ns + 1];
        g.previous_cost(k) = static_cast<float>(b[N_CONSERVATIVE + ns + 2]);
        if (has_scale) g.time_scale(k) = b[N_CONSERVATIVE + ns + 3];
        if (has_forcing) {
            const rtype * f = b + N_CONSERVATIVE + ns + 4;
            for (uint32_t m = 0; m < ns; m++) g.rate(k, m) = f[m];
            for (uint32_t i = 0; i < N_CONSERVATIVE; i++) g.U_end(k, i) = f[ns + i];
        }
        g.volume(k) = b[S - 1];
        g.repaired(k) = 0;
        g.chem_cost(k) = rtype(0);
        const uint32_t at = p(PLAN_HEADER + 3 * P + s) + j;
        q(at) = k;
        key(at) = -g.previous_cost(k);
    });
}

void ChemistryBalance::pack_results() {
    const uint32_t R = result_stride, ns = n_species, P = static_cast<uint32_t>(n_ranks);
    const bool has_forcing = forced;
    const Kokkos::View<rtype *> buf = guest_buffer;
    const ChemistryCells g = guests;
    const Kokkos::View<uint32_t *> p = d_plan, rank = in_rank, base = d_in_base;
    // Results in place of the states (each cell's results fit in its state's slot)
    Kokkos::parallel_for("chemistry_balance_results", n_in, KOKKOS_LAMBDA(const uint32_t k) {
        const uint32_t s = rank(k);
        if (k - base(s) >= p(PLAN_HEADER + 2 * P + s)) return;
        rtype * b = &buf(static_cast<size_t>(k) * R);
        for (uint32_t j = 0; j < ns; j++) b[j] = has_forcing ? g.rhoY_end(k, j) : g.rhoY(k, j);
        b[ns] = g.chem_h(k);
        b[ns + 1] = g.chem_cost(k);
        b[ns + 2] = static_cast<rtype>(g.previous_cost(k));
        b[ns + 3] = static_cast<rtype>(g.repaired(k));
    });
}

void ChemistryBalance::take_results(const ChemistryCells & cells, const Kokkos::View<uint32_t *> & queue) {
    const uint32_t R = result_stride, ns = n_species, P = static_cast<uint32_t>(n_ranks);
    const Kokkos::View<rtype *> buf = send_buffer;
    const ChemistryCells c = cells;
    const SpeciesView out = forced ? c.rhoY_end : c.rhoY;
    const Kokkos::View<uint32_t *> p = d_plan, rank = out_rank, base = d_out_base;
    Kokkos::parallel_for("chemistry_balance_return", n_out, KOKKOS_LAMBDA(const uint32_t slot) {
        const uint32_t d = rank(slot), k = slot - base(d);
        if (k >= p(PLAN_HEADER + d)) return;
        const uint32_t cell = queue(p(PLAN_HEADER + P + d) + k);
        const rtype * b = &buf(static_cast<size_t>(slot) * R);
        for (uint32_t j = 0; j < ns; j++) out(cell, j) = b[j];
        c.chem_h(cell) = b[ns];
        c.chem_cost(cell) += b[ns + 1];
        c.previous_cost(cell) = static_cast<float>(b[ns + 2]);
        if (b[ns + 3] != rtype(0) && c.repaired.extent(0) > 0) c.repaired(cell) = 1;
    });
}

void ChemistryBalance::receive(const ChemistryCells & cells, const Kokkos::View<uint32_t *> & queue,
                               const Integrate & integrate) {
#ifdef Mallard_HAS_MPI
    if (n_out == 0 && n_in == 0) return;
    const uint32_t R = result_stride;
    rtype * out = staged() ? h_send_buffer.data() : send_buffer.data();
    rtype * in = staged() ? h_guest_buffer.data() : guest_buffer.data();
    if (n_in > 0) {
        if (staged()) {
            const comm::HostStaged host_staged;
            Kokkos::deep_copy(guest_buffer, h_guest_buffer);
        }
        unpack();
        integrate(guests, guest_queue, guest_cost, n_in, guest_range);
        pack_results();
        if (!use_nccl) {
            const comm::HostStaged host_staged;
            Kokkos::fence("chemistry_balance_results");
            if (staged()) Kokkos::deep_copy(h_guest_buffer, guest_buffer);
        }
    }
    if (use_nccl) {
        transfer(messages(false, in, R), messages(true, out, R), RESULT_TAG);
    } else {
        const comm::HostStaged host_staged;
        transfer(messages(false, in, R), {}, RESULT_TAG);
        wait(requests);
        if (staged() && n_out > 0) Kokkos::deep_copy(send_buffer, h_send_buffer);
    }
    if (n_out > 0) take_results(cells, queue);
#else
    (void)cells;
    (void)queue;
    (void)integrate;
#endif
}

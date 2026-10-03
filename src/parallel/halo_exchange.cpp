/**
 * @file halo_exchange.cpp
 * @author Matthew Bonanni (mbonanni001@gmail.com)
 * @brief Fills halo cells with their owners' values.
 * @version 0.3
 * @date 2026-10-02
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 *
 */

#include "halo_exchange.h"

#include <stdexcept>

namespace {

Kokkos::View<uint32_t *> flatten(const std::vector<std::vector<uint32_t>> & lists, std::vector<uint32_t> & offsets,
                                 const char * label) {
    offsets.assign(1, 0);
    for (const auto & l : lists) offsets.push_back(offsets.back() + l.size());
    Kokkos::View<uint32_t *> v(label, offsets.back());
    auto h = Kokkos::create_mirror_view(v);
    size_t k = 0;
    for (const auto & l : lists) {
        for (uint32_t c : l) h(k++) = c;
    }
    Kokkos::deep_copy(v, h);
    return v;
}

#ifdef Mallard_HAS_MPI
constexpr bool device_is_host_accessible =
    Kokkos::SpaceAccessibility<Kokkos::HostSpace, Kokkos::DefaultExecutionSpace::memory_space>::accessible;

#ifdef Mallard_GPU_AWARE_MPI
constexpr bool stage_through_host = false;
#else
constexpr bool stage_through_host = !device_is_host_accessible;
#endif
#endif

} // namespace

HaloExchange::HaloExchange(const Distribution & dist) : ranks(dist.neighbors) {
    send_cells = flatten(dist.send_cells, send_offsets, "halo_send_cells");
    recv_cells = flatten(dist.recv_cells, recv_offsets, "halo_recv_cells");
    allocate_buffers(N_CONSERVATIVE);
}

void HaloExchange::allocate_buffers(const uint32_t n_values_per_cell) {
    n_values = n_values_per_cell;
    send_buffer = Kokkos::View<rtype *>("halo_send_buffer", send_offsets.back() * n_values);
    recv_buffer = Kokkos::View<rtype *>("halo_recv_buffer", recv_offsets.back() * n_values);
    h_send_buffer = Kokkos::create_mirror_view(send_buffer);
    h_recv_buffer = Kokkos::create_mirror_view(recv_buffer);
}

void HaloExchange::start(const State & U) {
    if (!active()) return;
#ifdef Mallard_HAS_MPI
    if (n_values != N_CONSERVATIVE + U.n_species()) allocate_buffers(N_CONSERVATIVE + U.n_species());
    const uint32_t stride = n_values;
    const uint32_t n_species = U.n_species();
    Kokkos::View<uint32_t *> s_cells = send_cells;
    Kokkos::View<rtype *> s_buf = send_buffer;
    StateView flow = U.flow;
    SpeciesView species = U.species;
    Kokkos::parallel_for("halo_pack", s_cells.extent(0), KOKKOS_LAMBDA(const uint32_t k) {
        const uint32_t c = s_cells(k);
        FOR_I_CONSERVATIVE s_buf(k * stride + i) = flow(c, i);
        for (uint32_t j = 0; j < n_species; j++) s_buf(k * stride + N_CONSERVATIVE + j) = species(c, j);
    });
    Kokkos::fence("halo_pack");
    rtype * send_ptr = s_buf.data();
    rtype * recv_ptr = recv_buffer.data();
    if constexpr (stage_through_host) {
        Kokkos::deep_copy(h_send_buffer, s_buf);
        send_ptr = h_send_buffer.data();
        recv_ptr = h_recv_buffer.data();
    }
    const MPI_Datatype type = sizeof(rtype) == sizeof(double) ? MPI_DOUBLE : MPI_FLOAT;
    requests.assign(2 * ranks.size(), MPI_REQUEST_NULL);
    for (size_t n = 0; n < ranks.size(); n++) {
        MPI_Irecv(recv_ptr + recv_offsets[n] * stride, static_cast<int>((recv_offsets[n + 1] - recv_offsets[n]) * stride),
                  type, ranks[n], 0, comm::world(), &requests[n]);
    }
    for (size_t n = 0; n < ranks.size(); n++) {
        MPI_Isend(send_ptr + send_offsets[n] * stride, static_cast<int>((send_offsets[n + 1] - send_offsets[n]) * stride),
                  type, ranks[n], 0, comm::world(), &requests[ranks.size() + n]);
    }
#else
    (void)U;
    throw std::logic_error("HaloExchange: neighbors without MPI");
#endif
}

void HaloExchange::finish(const State & U) {
    if (!active()) return;
#ifdef Mallard_HAS_MPI
    if (MPI_Waitall(static_cast<int>(requests.size()), requests.data(), MPI_STATUSES_IGNORE) != MPI_SUCCESS) {
        throw std::runtime_error("HaloExchange: MPI_Waitall failed");
    }
    const uint32_t stride = n_values;
    const uint32_t n_species = U.n_species();
    Kokkos::View<uint32_t *> r_cells = recv_cells;
    Kokkos::View<rtype *> r_buf = recv_buffer;
    StateView flow = U.flow;
    SpeciesView species = U.species;
    if constexpr (stage_through_host) Kokkos::deep_copy(r_buf, h_recv_buffer);
    Kokkos::parallel_for("halo_unpack", r_cells.extent(0), KOKKOS_LAMBDA(const uint32_t k) {
        const uint32_t c = r_cells(k);
        FOR_I_CONSERVATIVE flow(c, i) = r_buf(k * stride + i);
        for (uint32_t j = 0; j < n_species; j++) species(c, j) = r_buf(k * stride + N_CONSERVATIVE + j);
    });
#else
    (void)U;
    throw std::logic_error("HaloExchange: neighbors without MPI");
#endif
}

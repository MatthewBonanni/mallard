/**
 * @file face_exchange.cpp
 * @author Matthew Bonanni (mbonanni001@gmail.com)
 * @brief Exchange of reconstructed states at partition faces.
 * @version 0.1
 * @date 2026-10-15
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 *
 */

#include "face_exchange.h"

#include <algorithm>
#include <stdexcept>
#include <unordered_map>

#include "device_comm.h"

namespace {

#ifdef Mallard_HAS_MPI
[[maybe_unused]] constexpr bool device_is_host_accessible =
    Kokkos::SpaceAccessibility<Kokkos::HostSpace, Kokkos::DefaultExecutionSpace::memory_space>::accessible;

#ifdef Mallard_GPU_AWARE_MPI
constexpr bool stage_through_host = false;
#else
constexpr bool stage_through_host = !device_is_host_accessible;
#endif

constexpr int FACE_TAG = 1;
#endif

} // namespace

FacePlan plan_face_exchange(const Distribution & dist, std::vector<PartitionFace> faces) {
    std::unordered_map<uint32_t, size_t> neighbor_of;
    for (size_t n = 0; n < dist.neighbors.size(); n++) {
        for (const uint32_t c : dist.recv_cells[n]) neighbor_of.emplace(c, n);
    }
    std::sort(faces.begin(), faces.end(), [](const PartitionFace & a, const PartitionFace & b) { return a.key < b.key; });
    std::vector<std::vector<uint64_t>> keys(dist.neighbors.size());
    std::vector<std::vector<uint32_t>> lists(dist.neighbors.size());
    std::vector<std::vector<uint8_t>> sides(dist.neighbors.size());
    for (const PartitionFace & f : faces) {
        const auto it = neighbor_of.find(f.halo_cell);
        if (it == neighbor_of.end()) throw std::logic_error("plan_face_exchange: a partition face's halo cell is not received");
        keys[it->second].push_back(f.key);
        lists[it->second].push_back(f.face);
        sides[it->second].push_back(f.owned_side);
    }
    std::vector<std::vector<uint64_t>> to_rank(static_cast<size_t>(comm::size()));
    for (size_t n = 0; n < dist.neighbors.size(); n++) to_rank[static_cast<size_t>(dist.neighbors[n])] = keys[n];
    const auto from_rank = comm::alltoallv(to_rank);
    FacePlan plan;
    for (size_t n = 0; n < dist.neighbors.size(); n++) {
        if (from_rank[static_cast<size_t>(dist.neighbors[n])] != keys[n]) {
            throw std::logic_error("plan_face_exchange: ranks disagree on their partition faces");
        }
        if (keys[n].empty()) continue;
        plan.neighbors.push_back(dist.neighbors[n]);
        plan.faces.push_back(std::move(lists[n]));
        plan.owned_side.push_back(std::move(sides[n]));
    }
    return plan;
}

FaceExchange::FaceExchange(const FacePlan & plan, const uint8_t n_quad, const bool nccl_)
    : ranks(plan.neighbors), nccl(nccl_), n_values(uint32_t(n_quad) * N_CONSERVATIVE) {
    if (nccl && !comm::nccl_available()) throw std::logic_error("FaceExchange: built without NCCL");
    offsets.assign(1, 0);
    for (const auto & l : plan.faces) offsets.push_back(offsets.back() + static_cast<uint32_t>(l.size()));
    faces = Kokkos::View<uint32_t *>("face_exchange_faces", offsets.back());
    sides = Kokkos::View<uint8_t *>("face_exchange_sides", offsets.back());
    auto h_faces = Kokkos::create_mirror_view(faces);
    auto h_sides = Kokkos::create_mirror_view(sides);
    size_t k = 0;
    for (size_t n = 0; n < plan.faces.size(); n++) {
        for (size_t i = 0; i < plan.faces[n].size(); i++, k++) {
            h_faces(k) = plan.faces[n][i];
            h_sides(k) = plan.owned_side[n][i];
        }
    }
    Kokkos::deep_copy(faces, h_faces);
    Kokkos::deep_copy(sides, h_sides);
    send_buffer = Kokkos::View<rtype *>("face_exchange_send", offsets.back() * n_values);
    recv_buffer = Kokkos::View<rtype *>("face_exchange_recv", offsets.back() * n_values);
    h_send_buffer = Kokkos::create_mirror_view(send_buffer);
    h_recv_buffer = Kokkos::create_mirror_view(recv_buffer);
}

void FaceExchange::start(const Kokkos::DefaultExecutionSpace & exec,
                         Kokkos::View<rtype **[2][N_CONSERVATIVE]> face_solution) {
    if (!active()) return;
#ifdef Mallard_HAS_MPI
    const uint32_t stride = n_values;
    const uint32_t n_quad = n_values / N_CONSERVATIVE;
    Kokkos::View<uint32_t *> f_list = faces;
    Kokkos::View<uint8_t *> s_list = sides;
    Kokkos::View<rtype *> buf = send_buffer;
    Kokkos::parallel_for("face_pack", Kokkos::RangePolicy<>(exec, 0, f_list.extent(0) * n_quad),
                         KOKKOS_LAMBDA(const uint32_t idx) {
        const uint32_t k = idx / n_quad, q = idx % n_quad;
        const uint32_t f = f_list(k);
        const uint8_t s = s_list(k);
        FOR_I_CONSERVATIVE buf(k * stride + q * N_CONSERVATIVE + i) = face_solution(f, q, s, i);
    });
#ifdef Mallard_HAS_NCCL
    if (nccl) {
        const ncclDataType_t type = sizeof(rtype) == sizeof(double) ? ncclDouble : ncclFloat;
        ncclComm_t c = comm::nccl();
        ncclGroupStart();
        for (size_t n = 0; n < ranks.size(); n++) {
            const size_t count = size_t(offsets[n + 1] - offsets[n]) * stride;
            ncclRecv(recv_buffer.data() + size_t(offsets[n]) * stride, count, type, ranks[n], c, exec.cuda_stream());
            ncclSend(buf.data() + size_t(offsets[n]) * stride, count, type, ranks[n], c, exec.cuda_stream());
        }
        if (ncclGroupEnd() != ncclSuccess) throw std::runtime_error("FaceExchange: NCCL exchange failed");
        return;
    }
#endif
    exec.fence("face_pack");
    rtype * send_ptr = buf.data();
    rtype * recv_ptr = recv_buffer.data();
    if constexpr (stage_through_host) {
        Kokkos::deep_copy(h_send_buffer, buf);
        send_ptr = h_send_buffer.data();
        recv_ptr = h_recv_buffer.data();
    }
    const MPI_Datatype type = sizeof(rtype) == sizeof(double) ? MPI_DOUBLE : MPI_FLOAT;
    requests.assign(2 * ranks.size(), MPI_REQUEST_NULL);
    for (size_t n = 0; n < ranks.size(); n++) {
        MPI_Irecv(recv_ptr + size_t(offsets[n]) * stride, static_cast<int>((offsets[n + 1] - offsets[n]) * stride), type,
                  ranks[n], FACE_TAG, comm::world(), &requests[n]);
    }
    for (size_t n = 0; n < ranks.size(); n++) {
        MPI_Isend(send_ptr + size_t(offsets[n]) * stride, static_cast<int>((offsets[n + 1] - offsets[n]) * stride), type,
                  ranks[n], FACE_TAG, comm::world(), &requests[ranks.size() + n]);
    }
#else
    (void)exec;
    (void)face_solution;
    throw std::logic_error("FaceExchange: neighbors without MPI");
#endif
}

void FaceExchange::finish(const Kokkos::DefaultExecutionSpace & exec,
                          Kokkos::View<rtype **[2][N_CONSERVATIVE]> face_solution) {
    if (!active()) return;
#ifdef Mallard_HAS_MPI
    if (!nccl && MPI_Waitall(static_cast<int>(requests.size()), requests.data(), MPI_STATUSES_IGNORE) != MPI_SUCCESS) {
        throw std::runtime_error("FaceExchange: MPI_Waitall failed");
    }
    if (stage_through_host && !nccl) Kokkos::deep_copy(exec, recv_buffer, h_recv_buffer);
    const uint32_t stride = n_values;
    const uint32_t n_quad = n_values / N_CONSERVATIVE;
    Kokkos::View<uint32_t *> f_list = faces;
    Kokkos::View<uint8_t *> s_list = sides;
    Kokkos::View<rtype *> buf = recv_buffer;
    Kokkos::parallel_for("face_unpack", Kokkos::RangePolicy<>(exec, 0, f_list.extent(0) * n_quad),
                         KOKKOS_LAMBDA(const uint32_t idx) {
        const uint32_t k = idx / n_quad, q = idx % n_quad;
        const uint32_t f = f_list(k);
        const uint8_t s = 1 - s_list(k);
        FOR_I_CONSERVATIVE face_solution(f, q, s, i) = buf(k * stride + q * N_CONSERVATIVE + i);
    });
#else
    (void)exec;
    (void)face_solution;
    throw std::logic_error("FaceExchange: neighbors without MPI");
#endif
}

/**
 * @file face_exchange.cpp
 * @author Matthew Bonanni (mbonanni001@gmail.com)
 * @brief Exchange of reconstructed face states between ranks.
 * @version 0.1
 * @date 2026-10-15
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 *
 */

#include "face_exchange.h"

#include <stdexcept>
#include <unordered_map>

#include "device_comm.h"
#include "mesh.h"

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

// A cell has fewer faces than this
constexpr uint64_t FACE_SLOTS = 64;

uint8_t side_of(const Mesh & mesh, uint32_t face, uint32_t cell) {
    return mesh.h_cells_of_face(face, 0) == static_cast<int32_t>(cell) ? 0 : 1;
}

uint32_t index_in_cell(const Mesh & mesh, uint32_t cell, uint32_t face) {
    uint32_t k = 0;
    while (mesh.h_face_of_cell(cell, k) != face) k++;
    return k;
}

Kokkos::View<uint32_t *> flatten(const std::vector<std::vector<uint32_t>> & lists, std::vector<uint32_t> & offsets,
                                 const char * label) {
    offsets.assign(1, 0);
    for (const auto & l : lists) offsets.push_back(offsets.back() + static_cast<uint32_t>(l.size()));
    Kokkos::View<uint32_t *> v(label, offsets.back());
    auto h = Kokkos::create_mirror_view(v);
    size_t k = 0;
    for (const auto & l : lists) {
        for (const uint32_t x : l) h(k++) = x;
    }
    Kokkos::deep_copy(v, h);
    return v;
}

Kokkos::View<uint8_t *> flatten(const std::vector<std::vector<uint8_t>> & lists, const char * label) {
    size_t n = 0;
    for (const auto & l : lists) n += l.size();
    Kokkos::View<uint8_t *> v(label, n);
    auto h = Kokkos::create_mirror_view(v);
    size_t k = 0;
    for (const auto & l : lists) {
        for (const uint8_t x : l) h(k++) = x;
    }
    Kokkos::deep_copy(v, h);
    return v;
}

} // namespace

FacePlan plan_face_exchange(const Distribution & dist, const Mesh & mesh, const std::vector<RemoteFaceSide> & needed) {
    std::unordered_map<uint32_t, size_t> neighbor_of;
    for (size_t n = 0; n < dist.neighbors.size(); n++) {
        for (const uint32_t c : dist.recv_cells[n]) neighbor_of.emplace(c, n);
    }
    const size_t n_ranks = static_cast<size_t>(comm::size());
    std::vector<std::vector<uint64_t>> requests(n_ranks);
    std::vector<std::vector<uint32_t>> recv_faces(dist.neighbors.size());
    std::vector<std::vector<uint8_t>> recv_sides(dist.neighbors.size());
    for (const RemoteFaceSide & r : needed) {
        const auto it = neighbor_of.find(r.cell);
        if (it == neighbor_of.end()) throw std::logic_error("plan_face_exchange: a face side of an unreceived halo cell");
        requests[static_cast<size_t>(dist.neighbors[it->second])].push_back(
            dist.global_cell[r.cell] * FACE_SLOTS + index_in_cell(mesh, r.cell, r.face));
        recv_faces[it->second].push_back(r.face);
        recv_sides[it->second].push_back(side_of(mesh, r.face, r.cell));
    }
    const auto asked = comm::alltoallv(requests);
    std::unordered_map<uint64_t, uint32_t> owned_of;
    for (uint32_t c = 0; c < dist.n_owned; c++) owned_of.emplace(dist.global_cell[c], c);
    FacePlan plan;
    for (size_t n = 0; n < dist.neighbors.size(); n++) {
        const auto & keys = asked[static_cast<size_t>(dist.neighbors[n])];
        if (keys.empty() && recv_faces[n].empty()) continue;
        std::vector<uint32_t> faces;
        std::vector<uint8_t> sides;
        for (const uint64_t key : keys) {
            const auto it = owned_of.find(key / FACE_SLOTS);
            if (it == owned_of.end()) throw std::logic_error("plan_face_exchange: a face side of a cell not owned here");
            const uint32_t f = mesh.h_face_of_cell(it->second, static_cast<uint32_t>(key % FACE_SLOTS));
            faces.push_back(f);
            sides.push_back(side_of(mesh, f, it->second));
        }
        plan.neighbors.push_back(dist.neighbors[n]);
        plan.send_faces.push_back(std::move(faces));
        plan.send_sides.push_back(std::move(sides));
        plan.recv_faces.push_back(std::move(recv_faces[n]));
        plan.recv_sides.push_back(std::move(recv_sides[n]));
    }
    return plan;
}

FaceExchange::FaceExchange(const FacePlan & plan, const uint8_t n_quad, const bool nccl_)
    : ranks(plan.neighbors), nccl(nccl_), n_values(uint32_t(n_quad) * N_CONSERVATIVE) {
    if (nccl && !comm::nccl_available()) throw std::logic_error("FaceExchange: built without NCCL");
    send_faces = flatten(plan.send_faces, send_offsets, "face_exchange_send_faces");
    recv_faces = flatten(plan.recv_faces, recv_offsets, "face_exchange_recv_faces");
    send_sides = flatten(plan.send_sides, "face_exchange_send_sides");
    recv_sides = flatten(plan.recv_sides, "face_exchange_recv_sides");
    send_buffer = Kokkos::View<rtype *>("face_exchange_send", size_t(send_offsets.back()) * n_values);
    recv_buffer = Kokkos::View<rtype *>("face_exchange_recv", size_t(recv_offsets.back()) * n_values);
    h_send_buffer = Kokkos::create_mirror_view(send_buffer);
    h_recv_buffer = Kokkos::create_mirror_view(recv_buffer);
}

void FaceExchange::start(const Kokkos::DefaultExecutionSpace & exec,
                         Kokkos::View<rtype **[2][N_CONSERVATIVE]> face_solution) {
    if (!active()) return;
#ifdef Mallard_HAS_MPI
    const uint32_t stride = n_values;
    const uint32_t n_quad = n_values / N_CONSERVATIVE;
    Kokkos::View<uint32_t *> f_list = send_faces;
    Kokkos::View<uint8_t *> s_list = send_sides;
    Kokkos::View<rtype *> buf = send_buffer;
    Kokkos::parallel_for("face_pack", Kokkos::RangePolicy<>(exec, 0, f_list.extent(0) * n_quad),
                         KOKKOS_LAMBDA(const uint32_t idx) {
        const uint32_t k = idx / n_quad, q = idx % n_quad;
        FOR_I_CONSERVATIVE buf(k * stride + q * N_CONSERVATIVE + i) = face_solution(f_list(k), q, s_list(k), i);
    });
#ifdef Mallard_HAS_NCCL
    if (nccl) {
        const ncclDataType_t type = sizeof(rtype) == sizeof(double) ? ncclDouble : ncclFloat;
        ncclComm_t c = comm::nccl();
        ncclGroupStart();
        for (size_t n = 0; n < ranks.size(); n++) {
            ncclRecv(recv_buffer.data() + size_t(recv_offsets[n]) * stride,
                     size_t(recv_offsets[n + 1] - recv_offsets[n]) * stride, type, ranks[n], c, exec.cuda_stream());
            ncclSend(buf.data() + size_t(send_offsets[n]) * stride, size_t(send_offsets[n + 1] - send_offsets[n]) * stride,
                     type, ranks[n], c, exec.cuda_stream());
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
        MPI_Irecv(recv_ptr + size_t(recv_offsets[n]) * stride,
                  static_cast<int>((recv_offsets[n + 1] - recv_offsets[n]) * stride), type, ranks[n], FACE_TAG,
                  comm::world(), &requests[n]);
    }
    for (size_t n = 0; n < ranks.size(); n++) {
        MPI_Isend(send_ptr + size_t(send_offsets[n]) * stride,
                  static_cast<int>((send_offsets[n + 1] - send_offsets[n]) * stride), type, ranks[n], FACE_TAG,
                  comm::world(), &requests[ranks.size() + n]);
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
    Kokkos::View<uint32_t *> f_list = recv_faces;
    Kokkos::View<uint8_t *> s_list = recv_sides;
    Kokkos::View<rtype *> buf = recv_buffer;
    Kokkos::parallel_for("face_unpack", Kokkos::RangePolicy<>(exec, 0, f_list.extent(0) * n_quad),
                         KOKKOS_LAMBDA(const uint32_t idx) {
        const uint32_t k = idx / n_quad, q = idx % n_quad;
        FOR_I_CONSERVATIVE face_solution(f_list(k), q, s_list(k), i) = buf(k * stride + q * N_CONSERVATIVE + i);
    });
#else
    (void)exec;
    (void)face_solution;
    throw std::logic_error("FaceExchange: neighbors without MPI");
#endif
}

/**
 * @file distribution.cpp
 * @author Matthew Bonanni (mbonanni001@gmail.com)
 * @brief A rank's share of a distributed mesh.
 * @version 0.3
 * @date 2026-10-02
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 *
 */

#include "distribution.h"

#include <algorithm>
#include <array>
#include <stdexcept>
#include <string>
#include <unordered_map>

#include "comm.h"
#include "mesh.h"

std::shared_ptr<Mesh> build_local_mesh(Mesh & global, const std::vector<int> & owner, int halo_layers,
                                       Distribution & dist) {
    const int me = comm::rank();
    const uint32_t n_global = global.n_cells;
    if (owner.size() != n_global) throw std::invalid_argument("build_local_mesh: one owner per cell");

    // Vertex neighbors on the global mesh
    std::vector<std::vector<uint32_t>> cells_of_node(global.n_nodes);
    for (uint32_t c = 0; c < n_global; c++) {
        for (uint32_t k = 0; k < global.h_n_nodes_of_cell(c); k++) {
            cells_of_node[global.h_node_of_cell(c, k)].push_back(c);
        }
    }

    // Owned cells, then halo layers by breadth-first search over vertex neighbors
    std::vector<int32_t> local_of(n_global, -1);
    dist = Distribution();
    dist.halo_layers = halo_layers;
    for (uint32_t c = 0; c < n_global; c++) {
        if (owner[c] != me) continue;
        local_of[c] = dist.global_cell.size();
        dist.global_cell.push_back(c);
        dist.layer.push_back(0);
    }
    dist.n_owned = dist.global_cell.size();
    size_t layer_begin = 0;
    for (int l = 1; l <= halo_layers; l++) {
        const size_t layer_end = dist.global_cell.size();
        std::vector<uint32_t> next;
        for (size_t i = layer_begin; i < layer_end; i++) {
            const uint32_t c = dist.global_cell[i];
            for (uint32_t k = 0; k < global.h_n_nodes_of_cell(c); k++) {
                for (uint32_t nb : cells_of_node[global.h_node_of_cell(c, k)]) {
                    if (local_of[nb] == -1) {
                        local_of[nb] = -2;
                        next.push_back(nb);
                    }
                }
            }
        }
        std::sort(next.begin(), next.end());
        for (uint32_t c : next) {
            local_of[c] = dist.global_cell.size();
            dist.global_cell.push_back(c);
            dist.layer.push_back(l);
        }
        layer_begin = layer_end;
    }

    // Local nodes and cells
    std::unordered_map<uint32_t, uint32_t> local_node;
    std::vector<std::array<rtype, N_DIM>> nodes;
    std::vector<std::vector<uint32_t>> cells(dist.global_cell.size());
    for (size_t i = 0; i < dist.global_cell.size(); i++) {
        const uint32_t c = dist.global_cell[i];
        for (uint32_t k = 0; k < global.h_n_nodes_of_cell(c); k++) {
            const uint32_t gn = global.h_node_of_cell(c, k);
            auto [it, inserted] = local_node.emplace(gn, nodes.size());
            if (inserted) {
                std::array<rtype, N_DIM> x;
                for (int d = 0; d < N_DIM; d++) x[d] = global.h_node_coords(gn, d);
                nodes.push_back(x);
            }
            cells[i].push_back(it->second);
        }
    }

    // Boundary faces: global boundary zones, and faces towards non-local cells
    std::vector<const std::string *> zone_of_face(global.n_faces, nullptr);
    std::vector<std::string> zone_names;
    zone_names.reserve(global.n_face_zones());
    for (FaceZone & zone : *global.face_zones()) {
        if (zone.get_type() != FaceZoneType::BOUNDARY) continue;
        zone_names.push_back(zone.get_name());
    }
    {
        size_t k = 0;
        for (FaceZone & zone : *global.face_zones()) {
            if (zone.get_type() != FaceZoneType::BOUNDARY) continue;
            for (uint32_t i = 0; i < zone.n_faces(); i++) zone_of_face[zone.h_faces(i)] = &zone_names[k];
            k++;
        }
    }
    std::vector<Mesh::BoundaryFace> boundary_faces;
    for (uint32_t c : dist.global_cell) {
        for (uint32_t k = 0; k < global.h_n_faces_of_cell(c); k++) {
            const uint32_t f = global.h_face_of_cell(c, k);
            const int32_t c0 = global.h_cells_of_face(f, 0), c1 = global.h_cells_of_face(f, 1);
            const int32_t other = (c0 == static_cast<int32_t>(c)) ? c1 : c0;
            std::string zone;
            if (other < 0) {
                zone = zone_of_face[f] ? *zone_of_face[f] : "unassigned";
            } else if (local_of[other] < 0) {
                zone = PARTITION_ZONE;
            } else {
                continue;
            }
            std::vector<uint32_t> face_nodes;
            for (uint32_t k = 0; k < global.h_n_nodes_of_face(f); k++) {
                face_nodes.push_back(local_node.at(global.h_node_of_face(f, k)));
            }
            boundary_faces.push_back({std::move(face_nodes), zone});
        }
    }

    auto local = std::make_shared<Mesh>();
    // Global ids first: they order the faces and neighbor lists like the serial mesh's
    local->h_global_cell_id = dist.global_cell;
    local->n_global_cells = n_global;
    local->init_from_connectivity(nodes, cells, boundary_faces);
    local->n_owned_cells = dist.n_owned;
    local->n_reconstructed_cells = std::count_if(dist.layer.begin(), dist.layer.end(), [](uint8_t l) { return l <= 1; });
    local->n_complete_cells =
        std::count_if(dist.layer.begin(), dist.layer.end(), [&](uint8_t l) { return l < halo_layers; });

    std::vector<int> halo_owner;
    for (size_t i = dist.n_owned; i < dist.global_cell.size(); i++) halo_owner.push_back(owner[dist.global_cell[i]]);
    plan_halo_exchange(dist, halo_owner);
    return local;
}

void plan_halo_exchange(Distribution & dist, const std::vector<int> & halo_owner) {
    const int me = comm::rank();
    const int n_ranks = comm::size();
    std::unordered_map<uint64_t, uint32_t> local_of;
    for (uint32_t i = 0; i < dist.global_cell.size(); i++) local_of.emplace(dist.global_cell[i], i);
    std::vector<std::vector<uint64_t>> wanted(n_ranks);
    for (size_t i = dist.n_owned; i < dist.global_cell.size(); i++) {
        wanted[halo_owner[i - dist.n_owned]].push_back(dist.global_cell[i]);
    }
    const auto requested = comm::alltoallv(wanted);
    dist.neighbors.clear();
    dist.send_cells.clear();
    dist.recv_cells.clear();
    for (int r = 0; r < n_ranks; r++) {
        if (wanted[r].empty() && requested[r].empty()) continue;
        if (r == me) throw std::logic_error("plan_halo_exchange: a rank requested its own cells");
        dist.neighbors.push_back(r);
        std::vector<uint32_t> recv, send;
        for (uint64_t g : wanted[r]) recv.push_back(local_of.at(g));
        for (uint64_t g : requested[r]) {
            const auto it = local_of.find(g);
            if (it == local_of.end() || it->second >= dist.n_owned) {
                throw std::logic_error("plan_halo_exchange: rank " + std::to_string(r) +
                                       " requested a cell not owned here");
            }
            send.push_back(it->second);
        }
        dist.recv_cells.push_back(std::move(recv));
        dist.send_cells.push_back(std::move(send));
    }
}

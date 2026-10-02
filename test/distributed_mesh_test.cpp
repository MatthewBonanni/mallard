/**
 * @file distributed_mesh_test.cpp
 * @author Matthew Bonanni (mbonanni001@gmail.com)
 * @brief The scalable distributed setup reproduces the setup from a global mesh.
 * @version 0.4
 * @date 2026-10-02
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 *
 */

#include <gtest/gtest.h>

#include <algorithm>
#include <map>
#include <memory>
#include <unordered_map>
#include <string>
#include <vector>

#include "comm.h"
#include "distributed_mesh.h"
#include "distribution.h"
#include "mesh.h"
#include "mesh_block.h"
#include "partition.h"
#include "test_fixtures.h"

namespace {

std::vector<std::string> mesh_inputs() {
    if constexpr (N_DIM == 2) {
        return {"type = \"cartesian_tri\"\nNx = 9\nNy = 7\n", "type = \"wedge\"\nNx = 11\nNy = 6\n"};
    }
    return {"type = \"cartesian_mixed\"\nNx = 6\nNy = 3\nNz = 3\n", "type = \"cartesian_tet\"\nNx = 3\nNy = 3\nNz = 2\n"};
}

/**
 * @brief Reference: the local mesh built from the whole mesh, which every
 *        rank holds (the setup before DistributedMesh).
 */
std::shared_ptr<Mesh> build_local_mesh_from_global(Mesh & global, const std::vector<int> & owner, int halo_layers,
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

using Point = std::array<rtype, N_DIM>;

/** @brief Face centroids of every boundary zone, sorted. */
std::map<std::string, std::vector<Point>> zone_faces(Mesh & mesh) {
    std::map<std::string, std::vector<Point>> zones;
    for (FaceZone & zone : *mesh.face_zones()) {
        auto & points = zones[zone.get_name()];
        for (uint32_t j = 0; j < zone.n_faces(); j++) {
            Point x;
            FOR_I_DIM x[i] = mesh.h_face_coords(zone.h_faces(j), i);
            points.push_back(x);
        }
        std::sort(points.begin(), points.end());
    }
    return zones;
}

void expect_same(Mesh & mesh, const Distribution & dist, Mesh & ref, const Distribution & ref_dist) {
    EXPECT_EQ(dist.n_owned, ref_dist.n_owned);
    ASSERT_EQ(dist.global_cell, ref_dist.global_cell);
    EXPECT_EQ(dist.layer, ref_dist.layer);
    EXPECT_EQ(dist.neighbors, ref_dist.neighbors);
    EXPECT_EQ(dist.send_cells, ref_dist.send_cells);
    EXPECT_EQ(dist.recv_cells, ref_dist.recv_cells);
    EXPECT_EQ(mesh.h_global_cell_id, ref.h_global_cell_id);
    EXPECT_EQ(mesh.n_global_cells, ref.n_global_cells);
    EXPECT_EQ(mesh.n_reconstructed(), ref.n_reconstructed());
    EXPECT_EQ(mesh.n_complete(), ref.n_complete());
    ASSERT_EQ(mesh.n_cells, ref.n_cells);
    EXPECT_EQ(mesh.n_nodes, ref.n_nodes);
    EXPECT_EQ(mesh.n_faces, ref.n_faces);
    for (uint32_t c = 0; c < mesh.n_cells; c++) {
        ASSERT_EQ(mesh.h_n_nodes_of_cell(c), ref.h_n_nodes_of_cell(c));
        for (uint32_t k = 0; k < mesh.h_n_nodes_of_cell(c); k++) {
            FOR_I_DIM ASSERT_EQ(mesh.h_node_coords(mesh.h_node_of_cell(c, k), i),
                                ref.h_node_coords(ref.h_node_of_cell(c, k), i));
        }
        EXPECT_EQ(mesh.h_cell_volume(c), ref.h_cell_volume(c));
        FOR_I_DIM EXPECT_EQ(mesh.h_cell_coords(c, i), ref.h_cell_coords(c, i));
    }
    EXPECT_EQ(zone_faces(mesh), zone_faces(ref));
}

} // namespace

/**
 * @brief The distributed setup (face matching by hashing, migration, halo
 *        layers by distributed search, and regrowing them) must build exactly
 *        the local meshes and exchange plans that the global-mesh setup builds
 *        for the same owners, so solutions stay rank-count independent.
 */
TEST(DistributedMeshTest, LocalMeshesMatchTheSetupFromTheGlobalMesh) {
    for (const std::string & mesh_input : mesh_inputs()) {
        SCOPED_TRACE(mesh_input);
        const toml::value input = parse_toml("[mesh]\n" + mesh_input);
        Mesh global;
        global.init(input);
        DistributedMesh distributed(read_mesh_block(input));

        // The dual graph of each block cell: its face neighbors in the global mesh
        for (uint32_t c = 0; c < distributed.n_block_cells(); c++) {
            const uint64_t g = distributed.first_cell() + c;
            std::vector<uint64_t> expected, graph(distributed.graph_neighbors().begin() + distributed.graph_offsets()[c],
                                                  distributed.graph_neighbors().begin() + distributed.graph_offsets()[c + 1]);
            for (uint32_t k = 0; k < global.h_n_faces_of_cell(g); k++) {
                const uint32_t f = global.h_face_of_cell(g, k);
                const int32_t c0 = global.h_cells_of_face(f, 0), c1 = global.h_cells_of_face(f, 1);
                if (c1 >= 0) expected.push_back(c0 == int32_t(g) ? c1 : c0);
            }
            std::sort(expected.begin(), expected.end());
            std::sort(graph.begin(), graph.end());
            ASSERT_EQ(graph, expected) << "cell " << g;
        }

        const std::vector<int> owner = partition_hilbert(distributed, comm::size());
        const std::vector<int32_t> all_owners = comm::allgatherv(std::vector<int32_t>(owner.begin(), owner.end()));
        distributed.distribute(owner);
        // One layer, then grown to three
        for (int layers : {1, 3}) {
            SCOPED_TRACE(layers);
            Distribution dist, ref_dist;
            auto mesh = distributed.build_local_mesh(layers, dist);
            auto ref = build_local_mesh_from_global(global, std::vector<int>(all_owners.begin(), all_owners.end()), layers, ref_dist);
            expect_same(*mesh, dist, *ref, ref_dist);
        }
    }
}

/**
 * @brief The distributed sample sort must give the same order as sorting all
 *        cells along the curve on one rank, and so equal pieces in curve order.
 */
TEST(DistributedMeshTest, HilbertPartitionSplitsTheGlobalCurveOrderEvenly) {
    const toml::value input = parse_toml("[mesh]\n" + mesh_inputs()[0]);
    DistributedMesh distributed(read_mesh_block(input));
    const std::vector<int> owner = partition_hilbert(distributed, comm::size());

    const auto centers = distributed.block_cell_centers();
    std::vector<double> flat;
    for (const auto & x : centers) flat.insert(flat.end(), x.begin(), x.end());
    std::vector<double> all;
    {
        // Gather every center (test-only: a global array)
        std::vector<std::vector<double>> send(comm::size(), flat);
        for (const auto & from : comm::alltoallv(send)) all.insert(all.end(), from.begin(), from.end());
    }
    const uint64_t n = all.size() / N_DIM;
    ASSERT_EQ(n, distributed.n_global_cells());
    std::array<double, N_DIM> lo, hi;
    lo.fill(1e300);
    hi.fill(-1e300);
    for (uint64_t c = 0; c < n; c++) {
        FOR_I_DIM {
            lo[i] = std::min(lo[i], all[c * N_DIM + i]);
            hi[i] = std::max(hi[i], all[c * N_DIM + i]);
        }
    }
    std::vector<std::pair<uint64_t, uint64_t>> order(n);
    for (uint64_t c = 0; c < n; c++) {
        std::array<double, N_DIM> x;
        FOR_I_DIM x[i] = all[c * N_DIM + i];
        order[c] = {hilbert_key(x, lo, hi), c};
    }
    std::sort(order.begin(), order.end());
    std::vector<int> expected(n);
    for (uint64_t k = 0; k < n; k++) expected[order[k].second] = (k * comm::size()) / n;
    for (uint32_t c = 0; c < distributed.n_block_cells(); c++) {
        ASSERT_EQ(owner[c], expected[distributed.first_cell() + c]) << "cell " << distributed.first_cell() + c;
    }
}

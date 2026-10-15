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
#include "gmsh_fixtures.h"
#include "mesh.h"
#include "mesh_block.h"
#include "partition.h"
#include "periodic.h"
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
    dist.halo_layers = static_cast<uint8_t>(halo_layers);
    for (uint32_t c = 0; c < n_global; c++) {
        if (owner[c] != me) continue;
        local_of[c] = static_cast<int32_t>(dist.global_cell.size());
        dist.global_cell.push_back(c);
        dist.layer.push_back(0);
    }
    dist.n_owned = static_cast<uint32_t>(dist.global_cell.size());
    size_t layer_begin = 0;
    for (int l = 1; l <= halo_layers; l++) {
        const size_t layer_end = dist.global_cell.size();
        std::vector<uint32_t> next;
        for (size_t i = layer_begin; i < layer_end; i++) {
            const uint32_t c = static_cast<uint32_t>(dist.global_cell[i]);
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
            local_of[c] = static_cast<int32_t>(dist.global_cell.size());
            dist.global_cell.push_back(c);
            dist.layer.push_back(static_cast<uint8_t>(l));
        }
        layer_begin = layer_end;
    }

    // Local nodes and cells
    std::unordered_map<uint32_t, uint32_t> local_node;
    std::vector<std::array<rtype, N_DIM>> nodes;
    std::vector<std::vector<uint32_t>> cells(dist.global_cell.size());
    for (size_t i = 0; i < dist.global_cell.size(); i++) {
        const uint32_t c = static_cast<uint32_t>(dist.global_cell[i]);
        for (uint32_t k = 0; k < global.h_n_nodes_of_cell(c); k++) {
            const uint32_t gn = global.h_node_of_cell(c, k);
            auto [it, inserted] = local_node.emplace(gn, nodes.size());
            if (inserted) {
                std::array<rtype, N_DIM> x;
                for (size_t d = 0; d < N_DIM; d++) x[d] = global.h_node_coords(gn, d);
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
    for (uint64_t g : dist.global_cell) {
        const uint32_t c = static_cast<uint32_t>(g);
        for (uint32_t k = 0; k < global.h_n_faces_of_cell(c); k++) {
            const uint32_t f = global.h_face_of_cell(c, k);
            const int32_t c0 = global.h_cells_of_face(f, 0), c1 = global.h_cells_of_face(f, 1);
            const int32_t other = (c0 == static_cast<int32_t>(c)) ? c1 : c0;
            std::string zone;
            if (other < 0) {
                zone = zone_of_face[f] ? *zone_of_face[f] : "unassigned";
            } else if (local_of[static_cast<size_t>(other)] < 0) {
                zone = PARTITION_ZONE;
            } else {
                continue;
            }
            std::vector<uint32_t> face_nodes;
            for (uint32_t j = 0; j < global.h_n_nodes_of_face(f); j++) {
                face_nodes.push_back(local_node.at(global.h_node_of_face(f, j)));
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
    local->n_reconstructed_cells =
        static_cast<uint32_t>(std::count_if(dist.layer.begin(), dist.layer.end(), [](uint8_t l) { return l <= 1; }));
    local->n_complete_cells =
        static_cast<uint32_t>(std::count_if(dist.layer.begin(), dist.layer.end(), [&](uint8_t l) { return l < halo_layers; }));

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
            const uint32_t g = static_cast<uint32_t>(distributed.first_cell() + c);
            std::vector<uint64_t> expected, graph(distributed.graph_neighbors().begin() + static_cast<std::ptrdiff_t>(distributed.graph_offsets()[c]),
                                                  distributed.graph_neighbors().begin() + static_cast<std::ptrdiff_t>(distributed.graph_offsets()[c + 1]));
            for (uint32_t k = 0; k < global.h_n_faces_of_cell(g); k++) {
                const uint32_t f = global.h_face_of_cell(g, k);
                const int32_t c0 = global.h_cells_of_face(f, 0), c1 = global.h_cells_of_face(f, 1);
                if (c1 >= 0) expected.push_back(static_cast<uint64_t>(c0 == int32_t(g) ? c1 : c0));
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
        std::vector<std::vector<double>> send(static_cast<size_t>(comm::size()), flat);
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
    for (uint64_t k = 0; k < n; k++) expected[order[k].second] = static_cast<int>((k * static_cast<uint64_t>(comm::size())) / n);
    for (uint32_t c = 0; c < distributed.n_block_cells(); c++) {
        ASSERT_EQ(owner[c], expected[distributed.first_cell() + c]) << "cell " << distributed.first_cell() + c;
    }
}

TEST(DistributedMeshTest, PeriodicHaloLayersFollowVertexNeighborsAcrossSeams) {
    // Each halo layer holds exactly the cells one vertex-neighbor step further,
    // including neighbors across the seams of a fully periodic box
    const std::string input = N_DIM == 2
        ? "[mesh]\ntype = \"cartesian_tri\"\nNx = 9\nNy = 7\nperiodic = [\"x\", \"y\"]\n"
        : "[mesh]\ntype = \"cartesian_tet\"\nNx = 4\nNy = 3\nNz = 3\nperiodic = [\"x\", \"y\", \"z\"]\n";
    const toml::value parsed = parse_toml(input);
    Mesh global;
    global.init(parsed);
    DistributedMesh distributed(read_mesh_block(parsed), Mesh::periodic_pairs(parsed));
    const std::vector<int> block_owner = partition_hilbert(distributed, comm::size());
    std::vector<uint64_t> pairs;
    for (uint32_t c = 0; c < block_owner.size(); c++) {
        pairs.insert(pairs.end(), {distributed.first_cell() + c, uint64_t(block_owner[c])});
    }
    pairs = comm::allgatherv(pairs);
    std::vector<int> owner(global.n_cells);
    for (size_t i = 0; i < pairs.size(); i += 2) owner[pairs[i]] = static_cast<int>(pairs[i + 1]);
    distributed.distribute(block_owner);
    const int layers = 3;
    Distribution dist;
    distributed.build_local_mesh(layers, dist);

    std::vector<int> expected(global.n_cells, -1);
    std::vector<uint32_t> front;
    for (uint32_t c = 0; c < global.n_cells; c++) {
        if (owner[c] == comm::rank()) {
            expected[c] = 0;
            front.push_back(c);
        }
    }
    for (int l = 1; l <= layers; l++) {
        std::vector<uint32_t> next;
        for (uint32_t c : front) {
            for (uint32_t k = global.h_offsets_cells_of_cell(c); k < global.h_offsets_cells_of_cell(c + 1); k++) {
                const uint32_t nb = global.h_cells_of_cell(k);
                if (expected[nb] < 0) {
                    expected[nb] = l;
                    next.push_back(nb);
                }
            }
        }
        front = next;
    }
    const auto n_expected = std::count_if(expected.begin(), expected.end(), [](int l) { return l >= 0; });
    EXPECT_EQ(dist.global_cell.size(), size_t(n_expected));
    for (size_t i = 0; i < dist.global_cell.size(); i++) {
        EXPECT_EQ(int(dist.layer[i]), expected[dist.global_cell[i]]) << "cell " << dist.global_cell[i];
    }
}

namespace {

std::string periodic_pairs_input(int n_dirs) {
    static const char * zones[3][2] = {{"left", "right"}, {"bottom", "top"}, {"back", "front"}};
    std::string s;
    for (size_t d = 0; d < static_cast<size_t>(n_dirs); d++) {
        s += std::string("[[periodic]]\nzones = [\"") + zones[d][0] + "\", \"" + zones[d][1] + "\"]\ntranslation = [";
        for (size_t i = 0; i < N_DIM; i++) s += std::string(i ? ", " : "") + (i == d ? "1.0" : "0.0");
        s += "]\n";
    }
    return s;
}

/** @brief Periodic meshes: generated with one to N_DIM directions, and a Gmsh mesh with jittered seams. */
std::vector<std::string> periodic_inputs() {
    if constexpr (N_DIM == 2) {
        const std::string file = write_temp_shared("mallard_distributed_periodic.msh", jittered_periodic_mesh_2d(7));
        return {"[mesh]\ntype = \"cartesian\"\nNx = 9\nNy = 7\nperiodic = [\"x\", \"y\"]\n",
                "[mesh]\ntype = \"cartesian_tri\"\nNx = 6\nNy = 5\nperiodic = [\"x\"]\n",
                "[mesh]\ntype = \"file\"\nfilename = \"" + file + "\"\n" + periodic_pairs_input(2)};
    }
    const std::string file = write_temp_shared("mallard_distributed_periodic.msh", jittered_periodic_mesh_3d(4));
    return {"[mesh]\ntype = \"cartesian_tet\"\nNx = 4\nNy = 3\nNz = 3\nperiodic = [\"x\", \"y\", \"z\"]\n",
            "[mesh]\ntype = \"cartesian_mixed\"\nNx = 6\nNy = 3\nNz = 4\nperiodic = [\"x\", \"z\"]\n",
            "[mesh]\ntype = \"file\"\nfilename = \"" + file + "\"\n" + periodic_pairs_input(3)};
}

/**
 * @brief The serial matcher on the whole mesh, gathered from the blocks; and
 *        which nodes lie on the periodic zones.
 */
std::pair<PeriodicNodes, std::vector<bool>> serial_classes(const MeshBlock & block,
                                                           const std::vector<Mesh::PeriodicPair> & pairs) {
    std::vector<double> coords;
    for (const auto & x : block.node_coords) coords.insert(coords.end(), x.begin(), x.end());
    coords = comm::allgatherv(coords);
    std::vector<uint64_t> faces;
    for (uint64_t f = 0; f < block.n_faces(); f++) {
        faces.push_back(block.face_zone[f]);
        faces.push_back(block.face_offsets[f + 1] - block.face_offsets[f]);
        faces.insert(faces.end(), block.face_nodes.begin() + static_cast<std::ptrdiff_t>(block.face_offsets[f]),
                     block.face_nodes.begin() + static_cast<std::ptrdiff_t>(block.face_offsets[f + 1]));
    }
    faces = comm::allgatherv(faces);
    std::vector<std::array<rtype, N_DIM>> nodes(coords.size() / N_DIM);
    for (size_t k = 0; k < nodes.size(); k++) FOR_I_DIM nodes[k][i] = static_cast<rtype>(coords[k * N_DIM + i]);
    std::vector<Mesh::BoundaryFace> boundary_faces;
    for (size_t i = 0; i < faces.size(); i += 2 + faces[i + 1]) {
        Mesh::BoundaryFace bf;
        bf.zone = block.zone_names[faces[i]];
        const auto first = faces.begin() + static_cast<std::ptrdiff_t>(i + 2);
        bf.nodes.assign(first, first + static_cast<std::ptrdiff_t>(faces[i + 1]));
        boundary_faces.push_back(std::move(bf));
    }
    PeriodicNodes classes = match_periodic_nodes(nodes, boundary_faces, pairs);
    std::vector<bool> on_zone(nodes.size(), false);
    for (const auto & bf : boundary_faces) {
        if (std::find(classes.zones.begin(), classes.zones.end(), bf.zone) == classes.zones.end()) continue;
        for (uint32_t n : bf.nodes) on_zone[n] = true;
    }
    return {std::move(classes), std::move(on_zone)};
}

} // namespace

/**
 * @brief Distributed periodic matching finds the serial matcher's keys and
 *        lattice offsets (doubly and triply periodic corners included), and no
 *        rank learns the classes of nodes outside its block: the periodic
 *        zones are never gathered.
 */
TEST(DistributedMeshTest, PeriodicClassesMatchTheSerialMatcherWithoutGatheringTheZones) {
    for (const std::string & input : periodic_inputs()) {
        SCOPED_TRACE(input);
        const toml::value parsed = parse_toml(input);
        const MeshBlock block = read_mesh_block(parsed);
        const auto pairs = Mesh::periodic_pairs(parsed);
        const auto [expected, on_zone] = serial_classes(block, pairs);
        const DistributedMesh distributed(block, pairs);
        const auto & held = distributed.block_periodic_classes();

        std::vector<uint64_t> used(block.cell_nodes);
        used.insert(used.end(), block.face_nodes.begin(), block.face_nodes.end());
        std::sort(used.begin(), used.end());
        used.erase(std::unique(used.begin(), used.end()), used.end());
        size_t n_periodic = 0;
        for (uint64_t g : used) {
            const auto it = held.find(g);
            if (!on_zone[g]) {
                EXPECT_TRUE(it == held.end()) << "node " << g;
                continue;
            }
            n_periodic++;
            ASSERT_TRUE(it != held.end()) << "node " << g;
            EXPECT_EQ(it->second.first, expected.key[g]) << "node " << g;
            EXPECT_EQ(it->second.second, expected.lattice[g]) << "node " << g;
        }
        EXPECT_EQ(held.size(), n_periodic);
        if (comm::size() > 1) {
            EXPECT_LT(held.size(), size_t(std::count(on_zone.begin(), on_zone.end(), true)));
        }
    }
}

TEST(DistributedMeshTest, PeriodicZonesThatDoNotMatchAreRejectedOnEveryRank) {
    const std::string file = write_temp_shared("mallard_distributed_mismatch.msh",
                                               N_DIM == 2 ? jittered_periodic_mesh_2d(5) : jittered_periodic_mesh_3d(4));
    std::string pairs = periodic_pairs_input(N_DIM);
    const std::string unit = "translation = [1.0";
    pairs.replace(pairs.find(unit), unit.size(), "translation = [0.9");
    const toml::value parsed = parse_toml("[mesh]\ntype = \"file\"\nfilename = \"" + file + "\"\n" + pairs);
    uint32_t reported = 0;
    try {
        DistributedMesh distributed(read_mesh_block(parsed), Mesh::periodic_pairs(parsed));
        ADD_FAILURE() << "mismatched zones were accepted";
    } catch (const std::runtime_error & e) {
        const std::string what = e.what();
        reported = what.find("Periodic zones left and right: node") != std::string::npos &&
                   what.find("has no match in right") != std::string::npos;
    }
    EXPECT_EQ(comm::allreduce(reported, comm::Op::MAX), 1u);
}

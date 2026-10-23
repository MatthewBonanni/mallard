/**
 * @file distributed_mesh.h
 * @author Matthew Bonanni (mbonanni001@gmail.com)
 * @brief Scalable setup of a distributed mesh: no rank ever holds the whole mesh.
 * @version 0.4
 * @date 2026-10-02
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 *
 */

#ifndef DISTRIBUTED_MESH_H
#define DISTRIBUTED_MESH_H

#include <array>
#include <cstdint>
#include <memory>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "distribution.h"
#include "mesh_block.h"
#include "periodic.h"

class Mesh;

/**
 * @brief The mesh during setup, spread over the ranks. Each rank starts from
 *        its block of the mesh (cells, nodes and boundary faces by global id)
 *        and keeps it as the directory that serves those cells and nodes to
 *        other ranks:
 *
 *        1. Construction matches faces across ranks: each cell face and
 *           boundary face goes to a rank chosen by hashing its sorted node
 *           ids, which pairs them into the dual graph and tags each cell's
 *           boundary faces with their zones. With periodic pairs, nodes are
 *           first replaced by their periodic keys (see docs/design/periodic.md),
 *           so the faces of paired zones join into interior faces.
 *        2. A partitioner assigns an owner to every block cell
 *           (partition_hilbert, partition_graph), and distribute() sends each
 *           cell to its owner.
 *        3. build_local_mesh() adds halo layers by a distributed breadth-first
 *           search over vertex neighbors and builds the rank's Mesh and
 *           Distribution. Asking for more layers later grows the existing
 *           ones.
 *
 *        All memory is O(cells / ranks) per rank, plus the halo.
 */
class DistributedMesh {
    public:
        /** @brief Periodic node -> (key, lattice offset), as in PeriodicNodes but by global id. */
        using PeriodicMap = std::unordered_map<uint64_t, std::pair<uint64_t, std::array<int8_t, 3>>>;

        /**
         * @brief Match faces across ranks, joining the zones of the periodic
         *        pairs (collective).
         */
        explicit DistributedMesh(MeshBlock block, const std::vector<Mesh::PeriodicPair> & periodic = {});

        uint64_t n_global_cells() const { return cell_dist.back(); }
        uint64_t first_cell() const { return block.first_cell; }
        uint32_t n_block_cells() const { return static_cast<uint32_t>(block.n_cells()); }

        /** @brief First block cell of every rank, and the total (ParMETIS vtxdist). */
        const std::vector<uint64_t> & cell_distribution() const { return cell_dist; }

        /** @brief Dual graph of the block cells (cells sharing a face), by global id (CSR); until distribute(). */
        const std::vector<uint64_t> & graph_offsets() const { return graph_offsets_; }
        const std::vector<uint64_t> & graph_neighbors() const { return graph_neighbors_; }

        /** @brief Periodic classes of the periodic nodes of the block's cells and boundary faces, and no others. */
        const PeriodicMap & block_periodic_classes() const { return block_periodic; }

        /** @brief This rank's block of the mesh. */
        const MeshBlock & mesh_block() const { return block; }

        /** @brief Vertex average of every block cell (collective). */
        std::vector<std::array<double, N_DIM>> block_cell_centers() const;

        /** @brief Send every block cell to its owner rank, and drop the dual graph (collective). */
        void distribute(const std::vector<int> & owner);

        /**
         * @brief This rank's mesh: its owned cells in global order, then
         *        halo_layers layers of vertex neighbors, each in global order.
         *        Faces on the global boundary keep their zones; faces towards
         *        cells not on this rank go to PARTITION_ZONE. Fills dist,
         *        including the exchange plan (collective; after distribute()).
         */
        std::shared_ptr<Mesh> build_local_mesh(int halo_layers, Distribution & dist);

    private:
        /** @brief The cells this rank holds: owned cells, then halo layers. */
        struct Cells {
            std::vector<uint64_t> gid;
            std::vector<int> owner;
            std::vector<uint8_t> layer;
            std::vector<uint64_t> node_offsets{0}, nodes;            // global node ids
            std::vector<uint32_t> boundary_offsets{0};
            std::vector<std::array<uint32_t, 2>> boundary;           // (local face, zone)
            PeriodicMap periodic;                                    // classes of their periodic nodes
        };

        /**
         * @brief Match the nodes of the periodic zones without gathering them:
         *        zone nodes meet at ranks chosen by hashing their grid cells,
         *        the matches join into classes at the ranks whose node blocks
         *        hold the nodes, and each rank gets the classes of its block's
         *        nodes (collective).
         */
        void match_periodic(const std::vector<Mesh::PeriodicPair> & pairs);

        /** @brief Periodic key and lattice offset of a node this rank uses; null if it is not periodic. */
        const std::pair<uint64_t, std::array<int8_t, 3>> * periodic_class(uint64_t g) const;

        /** @brief Periodic key of a node (the node itself if it is not periodic). */
        uint64_t node_key(uint64_t g) const;

        size_t rank_of_cell(uint64_t g) const;
        size_t rank_of_node(uint64_t g) const;
        void append_record(std::vector<uint64_t> & out, uint32_t i) const;
        void read_records(const std::vector<uint64_t> & in, uint8_t layer);
        std::vector<std::array<double, N_DIM>> fetch_nodes(const std::vector<uint64_t> & sorted_ids) const;
        void grow_layer();

        MeshBlock block;
        std::vector<uint64_t> cell_dist, node_dist;
        std::vector<std::string> zones;  // the block's zones, then "unassigned"
        PeriodicNodes periodic_classes;  // translations and zones only
        PeriodicMap block_periodic;      // classes of the periodic nodes of the block's cells and faces
        std::vector<uint64_t> graph_offsets_, graph_neighbors_;
        std::vector<uint32_t> boundary_offsets{0};
        std::vector<std::array<uint32_t, 2>> boundary;  // per block cell: (local face, zone)

        // After distribute()
        std::vector<int> owner;                         // per block cell
        std::vector<uint64_t> directory_offsets{0};     // per block node: cells using it (by node key)
        std::vector<std::pair<uint64_t, int>> directory;  // (cell, owner)
        Cells cells;
        std::unordered_map<uint64_t, uint32_t> halo_index;  // halo cell -> index in cells
        std::vector<uint64_t> owned_nodes;                   // keys of the owned cells' nodes, sorted
        std::unordered_set<uint64_t> searched_nodes;         // halo nodes whose cells are all here
        std::vector<std::pair<uint64_t, int>> next_layer;    // layer 1, found by distribute()
        int layers = 0;
};

#endif // DISTRIBUTED_MESH_H

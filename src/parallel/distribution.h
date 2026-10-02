/**
 * @file distribution.h
 * @author Matthew Bonanni (mbonanni001@gmail.com)
 * @brief A rank's share of a distributed mesh: owned cells, halo layers and the
 *        halo exchange plan.
 * @version 0.3
 * @date 2026-10-02
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 *
 */

#ifndef DISTRIBUTION_H
#define DISTRIBUTION_H

#include <cstdint>
#include <memory>
#include <vector>

class Mesh;

/** @brief Boundary zone of local faces that border cells of other ranks. */
inline constexpr const char * PARTITION_ZONE = "__partition__";

/**
 * @brief Local numbering: owned cells first (in global order), then halo
 *        layer 1, 2, ... Every local cell keeps its global id.
 */
struct Distribution {
    uint32_t n_owned = 0;
    uint8_t halo_layers = 0;
    std::vector<uint64_t> global_cell;  // local cell -> global cell
    std::vector<uint8_t> layer;         // 0 for owned cells, k for halo layer k
    std::vector<int> neighbors;         // ranks this rank exchanges with
    std::vector<std::vector<uint32_t>> send_cells;  // per neighbor: owned cells it needs
    std::vector<std::vector<uint32_t>> recv_cells;  // per neighbor: halo cells it owns
};

/**
 * @brief Build this rank's local mesh from the global mesh and the owner of
 *        every cell: the owned cells plus halo_layers layers of vertex
 *        neighbors. Faces on the global boundary keep their zones; faces
 *        between a local and a non-local cell go to PARTITION_ZONE. Fills dist,
 *        including the exchange plan (collective).
 */
std::shared_ptr<Mesh> build_local_mesh(Mesh & global, const std::vector<int> & owner, int halo_layers,
                                       Distribution & dist);

#endif // DISTRIBUTION_H

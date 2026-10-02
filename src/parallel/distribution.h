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
#include <vector>

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
 * @brief Fill the exchange plan of dist, whose cells are numbered, from the
 *        owner of each of its halo cells (collective): each rank asks the
 *        owners for its halo cells, and the requests it receives become its
 *        send lists.
 */
void plan_halo_exchange(Distribution & dist, const std::vector<int> & halo_owner);

#endif // DISTRIBUTION_H

/**
 * @file periodic.h
 * @author Matthew Bonanni (mbonanni001@gmail.com)
 * @brief Identification of the nodes of periodic boundary zones.
 * @version 0.1
 * @date 2026-10-02
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 *
 */

#ifndef PERIODIC_H
#define PERIODIC_H

#include <array>
#include <cstdint>
#include <string>
#include <vector>

#include "mesh.h"

/**
 * @brief Periodic classes of the mesh nodes: the translation of each lattice
 *        direction, and per node its key (the lowest node id of its class) and
 *        its lattice offset L, with x_n = x_key + sum_j L_j translations[j].
 */
struct PeriodicNodes {
    std::vector<std::string> zones;  // the zones joined by the pairs
    std::vector<std::array<rtype, N_DIM>> translations;
    std::vector<uint32_t> key;
    std::vector<std::array<int8_t, 3>> lattice;
};

/**
 * @brief Match the nodes of zone_b of every pair to the nodes of zone_a
 *        translated by the pair's translation (within 1e-6 of the zones'
 *        shortest edge) and join the matches into periodic classes. Pairs with
 *        the same translation share a lattice direction; there are at most
 *        three directions. Throws if a node has no match or several, or if
 *        the matches put a node at two different offsets from its key.
 */
PeriodicNodes match_periodic_nodes(const std::vector<std::array<rtype, N_DIM>> & nodes,
                                   const std::vector<Mesh::BoundaryFace> & boundary_faces,
                                   const std::vector<Mesh::PeriodicPair> & pairs);

#endif // PERIODIC_H

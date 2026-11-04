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
#include <map>
#include <string>
#include <vector>

#include "mesh.h"

/**
 * @brief Periodic classes of the mesh nodes: the translation of each lattice
 *        direction (zero for the rotation's) and the rotation, and per node
 *        its key (the lowest node id of its class) and its lattice offset L,
 *        with x_n = M(L) x_key (see periodic_lattice_map).
 */
struct PeriodicNodes {
    std::vector<std::string> zones;  // the zones joined by the pairs
    std::vector<std::array<rtype, N_DIM>> translations;
    PeriodicRotation rotation;
    std::vector<uint32_t> key;
    std::vector<std::array<int8_t, 3>> lattice;
};

/** @brief Rotation matrix by angle (radians, right-handed) about the unit vector axis. */
std::array<std::array<double, 3>, 3> rotation_matrix(const std::array<double, 3> & axis, double angle);

/**
 * @brief The rigid map x -> Q x + t of lattice offset L: the rotation's
 *        direction turns by L_r angle about its axis, every other direction
 *        translates by L_j translations[j]. The maps of the directions
 *        commute (periodic_directions checks it), so their order does not
 *        matter. Without a rotation, Q is the identity and t is the sum
 *        sum_j L_j translations[j] in double, in order of j.
 */
void periodic_lattice_map(const std::vector<std::array<rtype, N_DIM>> & translations, const PeriodicRotation & rotation,
                          const std::array<int, 3> & L, double Q[3][3], double t[3]);

/** @brief The image in zone_b of a point x of zone_a. */
std::array<rtype, N_DIM> periodic_image(const Mesh::PeriodicPair & pair, const std::array<rtype, N_DIM> & x);

/** @brief "the translation (..)" or "the rotation by .. degrees about (..)", for error messages. */
std::string periodic_map_string(const Mesh::PeriodicPair & pair);

/**
 * @brief Check the pairs (zones used once, each a nonzero translation or
 *        rotation, at most three distinct ones, at most one rotation, and
 *        translations along its axis, so that the maps commute) and fill
 *        result.zones, result.translations and result.rotation. Returns the
 *        lattice direction of every pair: pairs with the same map share one.
 */
std::vector<size_t> periodic_directions(const std::vector<Mesh::PeriodicPair> & pairs, PeriodicNodes & result);

/** @brief Matching tolerance of a pair whose zones' shortest edge is h. */
rtype periodic_tolerance(rtype h);

/**
 * @brief Points on a grid of spacing h. A point within the tolerance of a
 *        query lies in one of the 3^N_DIM grid cells around the query's.
 */
class PeriodicGrid {
    public:
        using Point = std::array<rtype, N_DIM>;
        using Cell = std::array<int64_t, N_DIM>;
        static constexpr int N_NEIGHBORS = N_DIM == 2 ? 9 : 27;

        explicit PeriodicGrid(rtype spacing) : h(spacing), tol(periodic_tolerance(spacing)) {}

        Cell cell_of(const Point & x) const;
        /** @brief Neighbor m of cell c, m in [0, N_NEIGHBORS) (c itself included). */
        static Cell neighbor(const Cell & c, int m);
        void insert(uint32_t id, const Point & x) { grid[cell_of(x)].push_back({id, x}); }
        /** @brief Number of points within the tolerance of y; match is the last one's id. */
        uint32_t find(const Point & y, uint32_t & match) const;

    private:
        rtype h, tol;
        std::map<Cell, std::vector<std::pair<uint32_t, Point>>> grid;
};

/** @brief "(x, y[, z])", for error messages. */
std::string periodic_point_string(const std::array<rtype, N_DIM> & x);

/**
 * @brief Match the nodes of zone_b of every pair to the images of the nodes
 *        of zone_a under the pair's map (within 1e-6 of the zones' shortest
 *        edge) and join the matches into periodic classes. Pairs with
 *        the same translation share a lattice direction; there are at most
 *        three directions. Throws if a node has no match or several, or if
 *        the matches put a node at two different offsets from its key.
 */
PeriodicNodes match_periodic_nodes(const std::vector<std::array<rtype, N_DIM>> & nodes,
                                   const std::vector<Mesh::BoundaryFace> & boundary_faces,
                                   const std::vector<Mesh::PeriodicPair> & pairs);

#endif // PERIODIC_H

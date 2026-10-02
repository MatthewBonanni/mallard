/**
 * @file periodic.cpp
 * @author Matthew Bonanni (mbonanni001@gmail.com)
 * @brief Identification of the nodes of periodic boundary zones.
 * @version 0.1
 * @date 2026-10-02
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 *
 */

#include "periodic.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <set>
#include <stdexcept>
#include <string>

namespace {

using Lattice = std::array<int, 3>;

Lattice add(const Lattice & a, const Lattice & b) { return {a[0] + b[0], a[1] + b[1], a[2] + b[2]}; }
Lattice sub(const Lattice & a, const Lattice & b) { return {a[0] - b[0], a[1] - b[1], a[2] - b[2]}; }

/** @brief Union-find whose links carry lattice offsets: x_n = x_parent + offset. */
struct LatticeUnionFind {
    std::vector<uint32_t> parent;
    std::vector<Lattice> offset;

    explicit LatticeUnionFind(uint32_t n) : parent(n), offset(n, Lattice{0, 0, 0}) {
        for (uint32_t i = 0; i < n; i++) parent[i] = i;
    }

    /** @brief Root of n, with offset[n] made relative to it. */
    uint32_t find(uint32_t n) {
        uint32_t root = n;
        std::vector<uint32_t> path;
        while (parent[root] != root) {
            path.push_back(root);
            root = parent[root];
        }
        // Compress from the node nearest the root outward
        for (auto it = path.rbegin(); it != path.rend(); ++it) {
            const uint32_t p = parent[*it];
            if (p != root) offset[*it] = add(offset[*it], offset[p]);
            parent[*it] = root;
        }
        return root;
    }

    /** @brief Join b and a with x_b = x_a + rel; false if that contradicts earlier joins. */
    bool join(uint32_t b, uint32_t a, const Lattice & rel) {
        const uint32_t rb = find(b), ra = find(a);
        // x_rb = x_b - offset_b = x_a + rel - offset_b = x_ra + offset_a + rel - offset_b
        const Lattice root_offset = sub(add(offset[a], rel), offset[b]);
        if (rb == ra) return root_offset == Lattice{0, 0, 0};
        parent[rb] = ra;
        offset[rb] = root_offset;
        return true;
    }
};

std::string coords_string(const std::array<rtype, N_DIM> & x) {
    std::string s = "(";
    FOR_I_DIM s += (i ? ", " : "") + std::to_string(x[i]);
    return s + ")";
}

} // namespace

PeriodicNodes match_periodic_nodes(const std::vector<std::array<rtype, N_DIM>> & nodes,
                                   const std::vector<Mesh::BoundaryFace> & boundary_faces,
                                   const std::vector<Mesh::PeriodicPair> & pairs) {
    const uint32_t n_nodes = nodes.size();
    PeriodicNodes result;
    std::map<std::string, std::vector<const Mesh::BoundaryFace *>> faces_of_zone;
    for (const auto & bf : boundary_faces) faces_of_zone[bf.zone].push_back(&bf);

    LatticeUnionFind classes(n_nodes);
    std::set<std::string> used_zones;
    for (const auto & pair : pairs) {
        for (const std::string & zone : {pair.zone_a, pair.zone_b}) {
            result.zones.push_back(zone);
            if (!faces_of_zone.count(zone)) {
                throw std::runtime_error("Periodic zone " + zone + " is not a boundary zone of the mesh.");
            }
            if (!used_zones.insert(zone).second) {
                throw std::runtime_error("Periodic zone " + zone + " appears in more than one periodic pair.");
            }
        }
        // Lattice direction of this pair: shared with pairs of the same translation
        rtype length = 0.0;
        FOR_I_DIM length = std::max(length, std::abs(pair.translation[i]));
        if (length == 0.0) {
            throw std::runtime_error("Periodic pair " + pair.zone_a + " / " + pair.zone_b + " has no translation.");
        }
        size_t dir = 0;
        for (; dir < result.translations.size(); dir++) {
            rtype diff = 0.0;
            FOR_I_DIM diff = std::max(diff, std::abs(result.translations[dir][i] - pair.translation[i]));
            if (diff <= 1e-12 * length) break;
        }
        if (dir == result.translations.size()) {
            if (dir == 3) throw std::runtime_error("At most three distinct periodic translations are supported.");
            result.translations.push_back(pair.translation);
        }

        auto zone_nodes = [&](const std::string & zone, rtype & min_edge) {
            std::vector<uint32_t> ids;
            for (const auto * bf : faces_of_zone.at(zone)) {
                const size_t n = bf->nodes.size();
                for (size_t k = 0; k < n; k++) {
                    ids.push_back(bf->nodes[k]);
                    const auto & a = nodes[bf->nodes[k]];
                    const auto & b = nodes[bf->nodes[(k + 1) % n]];
                    rtype d2 = 0.0;
                    FOR_I_DIM d2 += (a[i] - b[i]) * (a[i] - b[i]);
                    if (d2 > 0.0) min_edge = std::min(min_edge, std::sqrt(d2));
                }
            }
            std::sort(ids.begin(), ids.end());
            ids.erase(std::unique(ids.begin(), ids.end()), ids.end());
            return ids;
        };
        rtype h = std::numeric_limits<rtype>::max();
        const std::vector<uint32_t> nodes_a = zone_nodes(pair.zone_a, h);
        const std::vector<uint32_t> nodes_b = zone_nodes(pair.zone_b, h);
        if (nodes_a.size() != nodes_b.size()) {
            throw std::runtime_error("Periodic zones " + pair.zone_a + " and " + pair.zone_b + " have " +
                                     std::to_string(nodes_a.size()) + " and " + std::to_string(nodes_b.size()) +
                                     " nodes.");
        }
        const rtype tol = 1e-6 * h;

        // Nodes of zone_b on a grid of spacing h; a match lies in one of the
        // 3^N_DIM grid cells around the translated node
        auto cell_of = [&](const std::array<rtype, N_DIM> & x) {
            std::array<int64_t, N_DIM> c;
            FOR_I_DIM c[i] = static_cast<int64_t>(std::floor(x[i] / h));
            return c;
        };
        std::map<std::array<int64_t, N_DIM>, std::vector<uint32_t>> grid;
        for (uint32_t b : nodes_b) grid[cell_of(nodes[b])].push_back(b);
        std::vector<bool> matched(n_nodes, false);
        Lattice rel = {0, 0, 0};
        rel[dir] = 1;
        for (uint32_t a : nodes_a) {
            std::array<rtype, N_DIM> y;
            FOR_I_DIM y[i] = nodes[a][i] + pair.translation[i];
            const auto c = cell_of(y);
            int64_t match = -1;
            uint32_t n_found = 0;
            for (int m = 0; m < (N_DIM == 2 ? 9 : 27); m++) {
                std::array<int64_t, N_DIM> q;
                int r = m;
                FOR_I_DIM {
                    q[i] = c[i] + (r % 3) - 1;
                    r /= 3;
                }
                const auto it = grid.find(q);
                if (it == grid.end()) continue;
                for (uint32_t b : it->second) {
                    rtype d2 = 0.0;
                    FOR_I_DIM d2 += (nodes[b][i] - y[i]) * (nodes[b][i] - y[i]);
                    if (d2 <= tol * tol) {
                        match = b;
                        n_found++;
                    }
                }
            }
            if (n_found != 1) {
                throw std::runtime_error("Periodic zones " + pair.zone_a + " and " + pair.zone_b + ": node " +
                                         coords_string(nodes[a]) + " of " + pair.zone_a + " has " +
                                         (n_found ? "several matches" : "no match") + " in " + pair.zone_b +
                                         " after the translation " + coords_string(pair.translation) + ".");
            }
            if (matched[match]) {
                throw std::runtime_error("Periodic zones " + pair.zone_a + " and " + pair.zone_b + ": node " +
                                         coords_string(nodes[match]) + " of " + pair.zone_b +
                                         " matches several nodes.");
            }
            matched[match] = true;
            if (!classes.join(match, a, rel)) {
                throw std::runtime_error("Periodic pairs map node " + coords_string(nodes[a]) +
                                         " onto itself (do the zones of a pair touch?).");
            }
        }
    }

    // Keys are the lowest node id of each class, offsets relative to it
    std::vector<uint32_t> lowest(n_nodes, std::numeric_limits<uint32_t>::max());
    for (uint32_t n = 0; n < n_nodes; n++) {
        const uint32_t root = classes.find(n);
        lowest[root] = std::min(lowest[root], n);
    }
    result.key.resize(n_nodes);
    result.lattice.resize(n_nodes);
    for (uint32_t n = 0; n < n_nodes; n++) {
        const uint32_t key = lowest[classes.find(n)];
        const Lattice L = sub(classes.offset[n], classes.offset[key]);
        result.key[n] = key;
        for (int j = 0; j < 3; j++) {
            if (std::abs(L[j]) > 127) throw std::runtime_error("Periodic lattice offset out of range.");
            result.lattice[n][j] = static_cast<int8_t>(L[j]);
        }
    }
    return result;
}

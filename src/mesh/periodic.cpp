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
#include <numbers>
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

} // namespace

std::array<std::array<double, 3>, 3> rotation_matrix(const std::array<double, 3> & axis, const double angle) {
    const double c = std::cos(angle), s = std::sin(angle);
    const auto & a = axis;
    std::array<std::array<double, 3>, 3> Q;
    for (size_t i = 0; i < 3; i++) {
        for (size_t j = 0; j < 3; j++) Q[i][j] = (1.0 - c) * a[i] * a[j] + (i == j ? c : 0.0);
    }
    Q[0][1] -= s * a[2];
    Q[0][2] += s * a[1];
    Q[1][0] += s * a[2];
    Q[1][2] -= s * a[0];
    Q[2][0] -= s * a[1];
    Q[2][1] += s * a[0];
    return Q;
}

void periodic_lattice_map(const std::vector<std::array<rtype, N_DIM>> & translations, const PeriodicRotation & rotation,
                          const std::array<int, 3> & L, double Q[3][3], double t[3]) {
    for (size_t i = 0; i < 3; i++) {
        t[i] = 0.0;
        for (size_t j = 0; j < 3; j++) Q[i][j] = (i == j) ? 1.0 : 0.0;
    }
    FOR_I_DIM {
        for (size_t j = 0; j < translations.size(); j++) t[i] += L[j] * double(translations[j][i]);
    }
    if (rotation.direction < 0 || L[static_cast<size_t>(rotation.direction)] == 0) return;
    const auto R = rotation_matrix(rotation.axis, L[static_cast<size_t>(rotation.direction)] * rotation.angle);
    for (size_t i = 0; i < 3; i++) {
        double Rc = 0.0;
        for (size_t j = 0; j < 3; j++) {
            Q[i][j] = R[i][j];
            Rc += R[i][j] * rotation.center[j];
        }
        t[i] += rotation.center[i] - Rc;
    }
}

namespace {

/** @brief The pair's rotation, its axis normalized and its center moved to the axis point nearest the origin. */
PeriodicRotation pair_rotation(const Mesh::PeriodicPair & pair) {
    PeriodicRotation r;
    r.angle = double(pair.angle);
    if constexpr (N_DIM == 3) {
        double norm = 0.0;
        for (size_t i = 0; i < 3; i++) norm += double(pair.axis[i]) * double(pair.axis[i]);
        norm = std::sqrt(norm);
        if (norm == 0.0) throw std::runtime_error("Periodic pair " + pair.zone_a + " / " + pair.zone_b + " has no axis.");
        for (size_t i = 0; i < 3; i++) r.axis[i] = double(pair.axis[i]) / norm;
    }
    FOR_I_DIM r.center[i] = double(pair.center[i]);
    double along = 0.0;
    for (size_t i = 0; i < 3; i++) along += r.center[i] * r.axis[i];
    for (size_t i = 0; i < 3; i++) r.center[i] -= along * r.axis[i];
    return r;
}

} // namespace

std::array<rtype, N_DIM> periodic_image(const Mesh::PeriodicPair & pair, const std::array<rtype, N_DIM> & x) {
    std::array<rtype, N_DIM> y;
    if (pair.angle == 0.0_r) {
        FOR_I_DIM y[i] = x[i] + pair.translation[i];
        return y;
    }
    const PeriodicRotation r = pair_rotation(pair);
    const auto Q = rotation_matrix(r.axis, r.angle);
    FOR_I_DIM {
        double v = r.center[i];
        for (size_t j = 0; j < N_DIM; j++) v += Q[i][j] * (double(x[j]) - r.center[j]);
        y[i] = static_cast<rtype>(v);
    }
    return y;
}

std::string periodic_map_string(const Mesh::PeriodicPair & pair) {
    if (pair.angle == 0.0_r) return "the translation " + periodic_point_string(pair.translation);
    std::string s = "the rotation by " + std::to_string(double(pair.angle) * 180.0 / std::numbers::pi) + " degrees about " +
                    periodic_point_string(pair.center);
    if constexpr (N_DIM == 3) {
        s += " along (" + std::to_string(pair.axis[0]) + ", " + std::to_string(pair.axis[1]) + ", " +
             std::to_string(pair.axis[2]) + ")";
    }
    return s;
}

std::vector<size_t> periodic_directions(const std::vector<Mesh::PeriodicPair> & pairs, PeriodicNodes & result) {
    std::set<std::string> used_zones;
    std::vector<size_t> dirs;
    for (const auto & pair : pairs) {
        for (const std::string & zone : {pair.zone_a, pair.zone_b}) {
            result.zones.push_back(zone);
            if (!used_zones.insert(zone).second) {
                throw std::runtime_error("Periodic zone " + zone + " appears in more than one periodic pair.");
            }
        }
        const std::string name = "Periodic pair " + pair.zone_a + " / " + pair.zone_b;
        rtype length = 0.0;
        FOR_I_DIM length = std::max(length, std::abs(pair.translation[i]));
        if (pair.angle != 0.0_r) {
            if (length > 0.0_r) throw std::runtime_error(name + " has both a translation and a rotation.");
            if (std::abs(double(pair.angle)) >= 2.0 * std::numbers::pi) {
                throw std::runtime_error(name + ": the rotation angle must be below 360 degrees.");
            }
            const PeriodicRotation r = pair_rotation(pair);
            if (result.rotation.direction >= 0) {
                const PeriodicRotation & q = result.rotation;
                double diff = std::abs(q.angle - r.angle) / std::abs(r.angle);
                for (size_t i = 0; i < 3; i++) {
                    diff = std::max(diff, std::abs(q.axis[i] - r.axis[i]));
                    diff = std::max(diff, std::abs(q.center[i] - r.center[i]));
                }
                if (diff > 1e-12) throw std::runtime_error("At most one distinct periodic rotation is supported.");
                dirs.push_back(static_cast<size_t>(q.direction));
                continue;
            }
            const size_t dir = result.translations.size();
            if (dir == 3) throw std::runtime_error("At most three distinct periodic maps are supported.");
            result.rotation = r;
            result.rotation.direction = static_cast<int>(dir);
            result.translations.push_back({});
            dirs.push_back(dir);
            continue;
        }
        if (length == 0.0_r) throw std::runtime_error(name + " has no translation.");
        size_t dir = 0;
        for (; dir < result.translations.size(); dir++) {
            if (static_cast<int>(dir) == result.rotation.direction) continue;
            rtype diff = 0.0;
            FOR_I_DIM diff = std::max(diff, std::abs(result.translations[dir][i] - pair.translation[i]));
            if (diff <= 1e-12_r * length) break;
        }
        if (dir == result.translations.size()) {
            if (dir == 3) throw std::runtime_error("At most three distinct periodic maps are supported.");
            result.translations.push_back(pair.translation);
        }
        dirs.push_back(dir);
    }
    if (result.rotation.direction >= 0) {
        // Translations commute with the rotation only along its axis (none in 2D)
        for (size_t j = 0; j < result.translations.size(); j++) {
            if (static_cast<int>(j) == result.rotation.direction) continue;
            const auto & T = result.translations[j];
            double t2 = 0.0, along = 0.0;
            FOR_I_DIM {
                t2 += double(T[i]) * double(T[i]);
                along += double(T[i]) * result.rotation.axis[i];
            }
            if (N_DIM == 2 || t2 - along * along > 1e-20 * t2) {
                throw std::runtime_error(std::string("A periodic translation must be along the axis of the periodic "
                                                     "rotation") + (N_DIM == 2 ? " (in 2D, a rotation admits none)." : "."));
            }
        }
    }
    return dirs;
}

rtype periodic_tolerance(rtype h) { return precision_tol(1e-6, 1e-3) * h; }

PeriodicGrid::Cell PeriodicGrid::cell_of(const Point & x) const {
    Cell c;
    FOR_I_DIM c[i] = static_cast<int64_t>(std::floor(x[i] / h));
    return c;
}

PeriodicGrid::Cell PeriodicGrid::neighbor(const Cell & c, int m) {
    Cell q;
    FOR_I_DIM {
        q[i] = c[i] + (m % 3) - 1;
        m /= 3;
    }
    return q;
}

uint32_t PeriodicGrid::find(const Point & y, uint32_t & match) const {
    const Cell c = cell_of(y);
    uint32_t n_found = 0;
    for (int m = 0; m < N_NEIGHBORS; m++) {
        const auto it = grid.find(neighbor(c, m));
        if (it == grid.end()) continue;
        for (const auto & [id, x] : it->second) {
            rtype d2 = 0.0;
            FOR_I_DIM d2 += (x[i] - y[i]) * (x[i] - y[i]);
            if (d2 <= tol * tol) {
                match = id;
                n_found++;
            }
        }
    }
    return n_found;
}

std::string periodic_point_string(const std::array<rtype, N_DIM> & x) {
    std::string s = "(";
    FOR_I_DIM s += (i ? ", " : "") + std::to_string(x[i]);
    return s + ")";
}

PeriodicNodes match_periodic_nodes(const std::vector<std::array<rtype, N_DIM>> & nodes,
                                   const std::vector<Mesh::BoundaryFace> & boundary_faces,
                                   const std::vector<Mesh::PeriodicPair> & pairs) {
    const uint32_t n_nodes = static_cast<uint32_t>(nodes.size());
    PeriodicNodes result;
    std::map<std::string, std::vector<const Mesh::BoundaryFace *>> faces_of_zone;
    for (const auto & bf : boundary_faces) faces_of_zone[bf.zone].push_back(&bf);
    for (const auto & pair : pairs) {
        for (const std::string & zone : {pair.zone_a, pair.zone_b}) {
            if (!faces_of_zone.count(zone)) {
                throw std::runtime_error("Periodic zone " + zone + " is not a boundary zone of the mesh.");
            }
        }
    }
    const std::vector<size_t> dirs = periodic_directions(pairs, result);

    LatticeUnionFind classes(n_nodes);
    for (size_t p = 0; p < pairs.size(); p++) {
        const auto & pair = pairs[p];
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
                    if (d2 > 0.0_r) min_edge = std::min(min_edge, std::sqrt(d2));
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

        PeriodicGrid grid(h);
        for (uint32_t b : nodes_b) grid.insert(b, nodes[b]);
        std::vector<bool> matched(n_nodes, false);
        Lattice rel = {0, 0, 0};
        rel[dirs[p]] = 1;
        for (uint32_t a : nodes_a) {
            const std::array<rtype, N_DIM> y = periodic_image(pair, nodes[a]);
            uint32_t match = 0;
            const uint32_t n_found = grid.find(y, match);
            if (n_found != 1) {
                throw std::runtime_error("Periodic zones " + pair.zone_a + " and " + pair.zone_b + ": node " +
                                         periodic_point_string(nodes[a]) + " of " + pair.zone_a + " has " +
                                         (n_found ? "several matches" : "no match") + " in " + pair.zone_b +
                                         " after " + periodic_map_string(pair) + ".");
            }
            if (matched[match]) {
                throw std::runtime_error("Periodic zones " + pair.zone_a + " and " + pair.zone_b + ": node " +
                                         periodic_point_string(nodes[match]) + " of " + pair.zone_b +
                                         " matches several nodes.");
            }
            matched[match] = true;
            if (!classes.join(match, a, rel)) {
                throw std::runtime_error("Periodic pairs map node " + periodic_point_string(nodes[a]) +
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
        for (size_t j = 0; j < 3; j++) {
            if (std::abs(L[j]) > 127) throw std::runtime_error("Periodic lattice offset out of range.");
            result.lattice[n][j] = static_cast<int8_t>(L[j]);
        }
    }
    return result;
}

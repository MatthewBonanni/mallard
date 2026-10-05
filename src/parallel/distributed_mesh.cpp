/**
 * @file distributed_mesh.cpp
 * @author Matthew Bonanni (mbonanni001@gmail.com)
 * @brief Scalable setup of a distributed mesh.
 * @version 0.4
 * @date 2026-10-02
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 *
 */

#include "distributed_mesh.h"

#include <algorithm>
#include <bit>
#include <cmath>
#include <limits>
#include <numeric>
#include <span>
#include <stdexcept>
#include <string>

#include "comm.h"
#include "mesh.h"

namespace {

constexpr uint64_t NONE = ~uint64_t(0);
constexpr uint32_t NO_ZONE = ~uint32_t(0);

// A face's sorted node ids, padded with NONE (triangles in 3D)
constexpr int KEY = N_DIM == 2 ? 2 : 4;
using FaceKey = std::array<uint64_t, KEY>;

uint64_t mix(uint64_t x) {
    x += 0x9e3779b97f4a7c15ull;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ull;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebull;
    return x ^ (x >> 31);
}

uint64_t hash(const FaceKey & key) {
    uint64_t h = 0;
    for (uint64_t v : key) h = mix(h ^ v);
    return h;
}

FaceKey face_key(const std::vector<uint64_t> & nodes) {
    FaceKey key;
    if (nodes.size() > key.size()) throw std::logic_error("DistributedMesh: face with too many nodes.");
    key.fill(NONE);
    std::copy(nodes.begin(), nodes.end(), key.begin());
    std::sort(key.begin(), key.end());  // the NONE padding sorts last
    return key;
}

bool valid_cell(uint64_t n_nodes) {
    if constexpr (N_DIM == 2) return n_nodes == 3 || n_nodes == 4;
    return n_nodes == 4 || n_nodes == 5 || n_nodes == 6 || n_nodes == 8;
}

uint32_t n_cell_faces(uint32_t n_nodes) {
    if constexpr (N_DIM == 2) return n_nodes;
    return cell_local_faces(n_nodes).size();
}

/** @brief Nodes of local face k of a cell, ordered as Mesh orders them. */
void cell_face(const uint64_t * cell, uint32_t n_nodes, uint32_t k, std::vector<uint64_t> & face) {
    face.clear();
    if constexpr (N_DIM == 2) {
        face = {cell[k], cell[(k + 1) % n_nodes]};
    } else {
        for (uint8_t i : cell_local_faces(n_nodes)[k]) face.push_back(cell[i]);
    }
}

/** @brief Throw on every rank if any rank found an error (avoids a deadlock in the next collective). */
void check_all(const std::string & error) {
    if (comm::allreduce(uint32_t(!error.empty()), comm::Op::MAX) == 0) return;
    throw std::runtime_error(error.empty() ? "Mesh: invalid mesh (reported by another rank)." : error);
}

/** @brief Prefix sums of every rank's count: rank r holds [dist[r], dist[r + 1]). */
std::vector<uint64_t> distribution(uint64_t count) {
    const std::vector<uint64_t> counts = comm::allgatherv(std::vector<uint64_t>{count});
    std::vector<uint64_t> dist(counts.size() + 1, 0);
    std::partial_sum(counts.begin(), counts.end(), dist.begin() + 1);
    return dist;
}

int rank_in(const std::vector<uint64_t> & dist, uint64_t g) {
    return std::upper_bound(dist.begin(), dist.end(), g) - dist.begin() - 1;
}

PeriodicGrid::Point to_point(const std::array<double, N_DIM> & x) {
    PeriodicGrid::Point y;
    FOR_I_DIM y[i] = x[i];
    return y;
}

// Signs of periodic links
constexpr uint64_t PLUS = 1, MINUS = 0;

uint64_t pack_lattice(const std::array<int8_t, 3> & l) {
    uint64_t x = 0;
    for (int j = 0; j < 3; j++) x |= uint64_t(uint8_t(l[j])) << (8 * j);
    return x;
}

std::array<int8_t, 3> unpack_lattice(uint64_t x) {
    std::array<int8_t, 3> l;
    for (int j = 0; j < 3; j++) l[j] = int8_t(uint8_t(x >> (8 * j)));
    return l;
}

} // namespace

int DistributedMesh::rank_of_cell(uint64_t g) const { return rank_in(cell_dist, g); }

const std::pair<uint64_t, std::array<int8_t, 3>> * DistributedMesh::periodic_class(uint64_t g) const {
    auto it = cells.periodic.find(g);
    if (it != cells.periodic.end()) return &it->second;
    it = block_periodic.find(g);
    return it == block_periodic.end() ? nullptr : &it->second;
}

uint64_t DistributedMesh::node_key(uint64_t g) const {
    const auto * c = periodic_class(g);
    return c ? c->first : g;
}

void DistributedMesh::match_periodic(const std::vector<Mesh::PeriodicPair> & pairs) {
    const int p = comm::size(), me = comm::rank();
    const size_t n_pairs = pairs.size();
    // Role of every periodic zone: 2 * pair + side (0 for zone_a, 1 for zone_b)
    std::vector<uint32_t> role(block.zone_names.size(), NO_ZONE);
    for (size_t k = 0; k < n_pairs; k++) {
        for (uint32_t side = 0; side < 2; side++) {
            const std::string & zone = side ? pairs[k].zone_b : pairs[k].zone_a;
            const auto it = std::find(block.zone_names.begin(), block.zone_names.end(), zone);
            if (it == block.zone_names.end()) {
                throw std::runtime_error("Periodic zone " + zone + " is not a boundary zone of the mesh.");
            }
            role[it - block.zone_names.begin()] = 2 * k + side;
        }
    }
    const std::vector<size_t> dirs = periodic_directions(pairs, periodic_classes);

    // The block's zone nodes; each pair's shortest zone edge h sets its grid
    // and tolerance, as in the serial matching
    std::vector<std::pair<uint64_t, uint64_t>> zone_nodes;  // (role, node)
    std::vector<uint64_t> ids;
    for (uint64_t f = 0; f < block.n_faces(); f++) {
        if (role[block.face_zone[f]] == NO_ZONE) continue;
        for (uint64_t k = block.face_offsets[f]; k < block.face_offsets[f + 1]; k++) {
            zone_nodes.push_back({role[block.face_zone[f]], block.face_nodes[k]});
            ids.push_back(block.face_nodes[k]);
        }
    }
    std::sort(zone_nodes.begin(), zone_nodes.end());
    zone_nodes.erase(std::unique(zone_nodes.begin(), zone_nodes.end()), zone_nodes.end());
    std::sort(ids.begin(), ids.end());
    ids.erase(std::unique(ids.begin(), ids.end()), ids.end());
    const std::vector<std::array<double, N_DIM>> coords = fetch_nodes(ids);
    auto coords_of = [&](uint64_t g) -> const std::array<double, N_DIM> & {
        return coords[std::lower_bound(ids.begin(), ids.end(), g) - ids.begin()];
    };
    std::vector<rtype> min_edge(n_pairs, std::numeric_limits<rtype>::max());
    for (uint64_t f = 0; f < block.n_faces(); f++) {
        if (role[block.face_zone[f]] == NO_ZONE) continue;
        rtype & h = min_edge[role[block.face_zone[f]] / 2];
        const uint64_t * face_nodes = &block.face_nodes[block.face_offsets[f]];
        const uint64_t n = block.face_offsets[f + 1] - block.face_offsets[f];
        for (uint64_t k = 0; k < n; k++) {
            const auto a = to_point(coords_of(face_nodes[k])), b = to_point(coords_of(face_nodes[(k + 1) % n]));
            rtype d2 = 0.0;
            FOR_I_DIM d2 += (a[i] - b[i]) * (a[i] - b[i]);
            if (d2 > 0.0_r) h = std::min(h, std::sqrt(d2));
        }
    }
    std::vector<double> h(n_pairs);
    for (size_t k = 0; k < n_pairs; k++) h[k] = static_cast<double>(min_edge[k]);
    comm::allreduce(std::span<double>(h), comm::Op::MIN);
    std::vector<PeriodicGrid> grids;
    for (size_t k = 0; k < n_pairs; k++) grids.emplace_back(static_cast<rtype>(h[k]));

    // Each node of zone_a, translated, goes to the rank its grid cell hashes
    // to; each node of zone_b to the ranks of every cell around its own, so
    // that rank sees all the candidates the serial matching sees. Records are
    // (role, node, coordinates).
    constexpr int WIDTH = 2 + N_DIM;
    auto rank_of_grid_cell = [&](size_t k, const PeriodicGrid::Cell & c) {
        uint64_t x = mix(k);
        for (int64_t v : c) x = mix(x ^ uint64_t(v));
        return int(x % p);
    };
    auto translated = [&](size_t k, PeriodicGrid::Point x) {
        FOR_I_DIM x[i] += pairs[k].translation[i];
        return x;
    };
    std::vector<std::vector<uint64_t>> send(p);
    std::vector<int> ranks;
    for (const auto & [r, g] : zone_nodes) {
        const size_t k = r / 2;
        const auto & x = coords_of(g);
        ranks.clear();
        if (r % 2 == 0) {
            ranks.push_back(rank_of_grid_cell(k, grids[k].cell_of(translated(k, to_point(x)))));
        } else {
            const PeriodicGrid::Cell c = grids[k].cell_of(to_point(x));
            for (int m = 0; m < PeriodicGrid::N_NEIGHBORS; m++) {
                ranks.push_back(rank_of_grid_cell(k, PeriodicGrid::neighbor(c, m)));
            }
            std::sort(ranks.begin(), ranks.end());
            ranks.erase(std::unique(ranks.begin(), ranks.end()), ranks.end());
        }
        for (int q : ranks) {
            send[q].insert(send[q].end(), {r, g});
            for (double v : x) send[q].push_back(std::bit_cast<uint64_t>(v));
        }
    }
    zone_nodes = {};
    comm::Received<uint64_t> received = comm::exchange(std::move(send));
    const uint64_t n_records = received.data.size() / WIDTH;
    const uint64_t * rec = received.data.data();
    // Sorted by (role, node), each node once
    std::vector<uint64_t> order(n_records);
    std::iota(order.begin(), order.end(), uint64_t(0));
    auto same_node = [&](uint64_t x, uint64_t y) { return std::equal(rec + x * WIDTH, rec + x * WIDTH + 2, rec + y * WIDTH); };
    std::sort(order.begin(), order.end(), [&](uint64_t x, uint64_t y) {
        return std::lexicographical_compare(rec + x * WIDTH, rec + x * WIDTH + 2, rec + y * WIDTH, rec + y * WIDTH + 2);
    });
    order.erase(std::unique(order.begin(), order.end(), same_node), order.end());
    auto record_point = [&](uint64_t j) {
        std::array<double, N_DIM> x;
        FOR_I_DIM x[i] = std::bit_cast<double>(rec[j * WIDTH + 2 + i]);
        return to_point(x);
    };

    // Zone sizes, each node counted by the rank of its own grid cell
    std::vector<uint64_t> b_nodes;
    std::vector<uint64_t> counts(2 * n_pairs, 0);
    for (uint64_t j : order) {
        const uint64_t r = rec[j * WIDTH], k = r / 2;
        if (r % 2 == 0) {
            counts[r]++;
        } else {
            const auto x = record_point(j);
            grids[k].insert(b_nodes.size(), x);
            b_nodes.push_back(rec[j * WIDTH + 1]);
            counts[r] += rank_of_grid_cell(k, grids[k].cell_of(x)) == me;
        }
    }
    comm::allreduce(std::span<uint64_t>(counts), comm::Op::SUM);
    for (size_t k = 0; k < n_pairs; k++) {
        for (size_t side = 0; side < 2; side++) {
            if (counts[2 * k + side] == 0) {
                throw std::runtime_error("Periodic zone " + (side ? pairs[k].zone_b : pairs[k].zone_a) +
                                         " is not a boundary zone of the mesh.");
            }
        }
        if (counts[2 * k] != counts[2 * k + 1]) {
            throw std::runtime_error("Periodic zones " + pairs[k].zone_a + " and " + pairs[k].zone_b + " have " +
                                     std::to_string(counts[2 * k]) + " and " + std::to_string(counts[2 * k + 1]) +
                                     " nodes.");
        }
    }

    // Each match becomes two links (node, other, pair, sign), with x_node =
    // x_other + sign * translation, at the ranks whose blocks hold the nodes
    std::string error;
    send.assign(p, {});
    for (uint64_t j : order) {
        const uint64_t r = rec[j * WIDTH], k = r / 2, a = rec[j * WIDTH + 1];
        if (r % 2) continue;
        const auto x = record_point(j);
        uint32_t match = 0;
        const uint32_t n_found = grids[k].find(translated(k, x), match);
        if (n_found != 1) {
            error = "Periodic zones " + pairs[k].zone_a + " and " + pairs[k].zone_b + ": node " +
                    periodic_point_string(x) + " of " + pairs[k].zone_a + " has " +
                    (n_found ? "several matches" : "no match") + " in " + pairs[k].zone_b +
                    " after the translation " + periodic_point_string(pairs[k].translation) + ".";
            continue;
        }
        const uint64_t b = b_nodes[match];
        send[rank_of_node(b)].insert(send[rank_of_node(b)].end(), {b, a, k, PLUS});
        send[rank_of_node(a)].insert(send[rank_of_node(a)].end(), {a, b, k, MINUS});
    }
    received = {};
    order = {};
    grids = {};
    b_nodes = {};
    check_all(error);
    std::vector<std::array<uint64_t, 4>> links;
    {
        const std::vector<uint64_t> data = comm::exchange(std::move(send)).data;
        for (size_t i = 0; i < data.size(); i += 4) links.push_back({data[i], data[i + 1], data[i + 2], data[i + 3]});
    }
    std::sort(links.begin(), links.end());
    auto block_point = [&](uint64_t g) { return to_point(block.node_coords[g - block.first_node]); };
    for (size_t i = 1; i < links.size(); i++) {
        const auto & [n, m, k, sign] = links[i];
        const auto & prev = links[i - 1];
        if (sign == PLUS && prev[0] == n && prev[2] == k && prev[3] == PLUS) {
            error = "Periodic zones " + pairs[k].zone_a + " and " + pairs[k].zone_b + ": node " +
                    periodic_point_string(block_point(n)) + " of " + pairs[k].zone_b + " matches several nodes.";
        }
    }
    check_all(error);

    // Classes: every node takes the lowest key its links offer, with the
    // offset the link implies, until no key changes. A class has a few nodes
    // (8 at a box corner), so this takes a few rounds; once keys settle, an
    // offer of the same key at another offset is a contradiction.
    using Lattice = std::array<int64_t, 3>;
    struct Class {
        uint64_t key;
        Lattice offset;
    };
    std::unordered_map<uint64_t, Class> classes;
    for (const auto & link : links) classes.emplace(link[0], Class{link[0], {0, 0, 0}});
    while (true) {
        // x_other = x_node - sign * T = x_key + offset - sign * e_dir
        send.assign(p, {});
        for (const auto & [n, other, k, sign] : links) {
            const Class & c = classes.at(n);
            Lattice offset = c.offset;
            offset[dirs[k]] -= sign == PLUS ? 1 : -1;
            auto & out = send[rank_of_node(other)];
            out.insert(out.end(), {other, c.key});
            for (int64_t l : offset) out.push_back(uint64_t(l));
        }
        const std::vector<uint64_t> offers = comm::exchange(std::move(send)).data;
        uint32_t changed = 0;
        for (size_t i = 0; i < offers.size(); i += 5) {
            Class & c = classes.at(offers[i]);
            const Lattice offset = {int64_t(offers[i + 2]), int64_t(offers[i + 3]), int64_t(offers[i + 4])};
            if (offers[i + 1] < c.key) {
                c = {offers[i + 1], offset};
                changed = 1;
            } else if (offers[i + 1] == c.key && offset != c.offset) {
                error = "Periodic pairs map node " + periodic_point_string(block_point(offers[i])) +
                        " onto itself (do the zones of a pair touch?).";
            }
        }
        if (comm::allreduce(changed, comm::Op::MAX) == 0) break;
        error.clear();
    }
    for (const auto & [n, c] : classes) {
        for (int64_t l : c.offset) {
            if (l < -127 || l > 127) error = "Periodic lattice offset out of range.";
        }
    }
    check_all(error);
    links = {};

    // Each rank learns the classes of its block's nodes from the ranks whose
    // node blocks hold them
    ids.assign(block.cell_nodes.begin(), block.cell_nodes.end());
    ids.insert(ids.end(), block.face_nodes.begin(), block.face_nodes.end());
    std::sort(ids.begin(), ids.end());
    ids.erase(std::unique(ids.begin(), ids.end()), ids.end());
    std::vector<std::vector<uint64_t>> wanted(p);
    for (uint64_t g : ids) wanted[rank_of_node(g)].push_back(g);
    const auto asked = comm::exchange(std::move(wanted));
    std::vector<std::vector<uint64_t>> answer(p);
    for (int r = 0; r < p; r++) {
        for (uint64_t g : asked.from(r)) {
            const auto it = classes.find(g);
            if (it == classes.end()) {
                answer[r].insert(answer[r].end(), {NONE, 0});
                continue;
            }
            std::array<int8_t, 3> offset;
            for (int j = 0; j < 3; j++) offset[j] = int8_t(it->second.offset[j]);
            answer[r].insert(answer[r].end(), {it->second.key, pack_lattice(offset)});
        }
    }
    const std::vector<uint64_t> got = comm::exchange(std::move(answer)).data;
    for (size_t i = 0; i < ids.size(); i++) {
        if (got[2 * i] != NONE) block_periodic[ids[i]] = {got[2 * i], unpack_lattice(got[2 * i + 1])};
    }
}

int DistributedMesh::rank_of_node(uint64_t g) const { return rank_in(node_dist, g); }

DistributedMesh::DistributedMesh(MeshBlock b, const std::vector<Mesh::PeriodicPair> & periodic)
    : block(std::move(b)) {
    const int p = comm::size(), me = comm::rank();
    cell_dist = distribution(block.n_cells());
    node_dist = distribution(block.n_nodes());
    std::string error;
    if (cell_dist[me] != block.first_cell || node_dist[me] != block.first_node) {
        error = "Mesh: blocks are not contiguous in rank order.";
    }
    for (uint64_t c = 0; c < block.n_cells(); c++) {
        if (!valid_cell(block.cell_offsets[c + 1] - block.cell_offsets[c])) {
            error = "Mesh: cell " + std::to_string(block.first_cell + c) + " has an unsupported number of nodes.";
        }
    }
    for (uint64_t f = 0; f < block.n_faces(); f++) {
        const uint64_t n = block.face_offsets[f + 1] - block.face_offsets[f];
        if (N_DIM == 2 ? n != 2 : (n != 3 && n != 4)) error = "Mesh: boundary face with " + std::to_string(n) + " nodes.";
        if (block.face_zone[f] >= block.zone_names.size()) error = "Mesh: boundary zone out of range.";
    }
    for (uint64_t node : block.cell_nodes) {
        if (node >= node_dist.back()) error = "Mesh: node id " + std::to_string(node) + " out of range.";
    }
    check_all(error);
    zones = block.zone_names;
    const uint32_t unassigned = zones.size();
    zones.push_back("unassigned");
    if (!periodic.empty()) match_periodic(periodic);
    std::vector<bool> periodic_zone(zones.size(), false);
    for (const std::string & zone : periodic_classes.zones) {
        periodic_zone[std::find(zones.begin(), zones.end(), zone) - zones.begin()] = true;
    }

    // Every cell face and boundary face goes to the rank its node set hashes
    // to, as (key, cell, local face) or (key, NONE, zone); faces of periodic
    // zones are found as interior faces instead
    constexpr int WIDTH = KEY + 2;
    std::vector<std::vector<uint64_t>> send(p);
    std::vector<uint64_t> face;
    // Two passes: count, then fill lists of exactly the right size
    std::vector<uint64_t> count(p, 0);
    auto for_each_face = [&](auto && post) {
        for (uint64_t c = 0; c < block.n_cells(); c++) {
            const uint32_t n = block.cell_offsets[c + 1] - block.cell_offsets[c];
            for (uint32_t k = 0; k < n_cell_faces(n); k++) {
                cell_face(&block.cell_nodes[block.cell_offsets[c]], n, k, face);
                for (uint64_t & node : face) node = node_key(node);
                post(face_key(face), block.first_cell + c, k);
            }
        }
        for (uint64_t f = 0; f < block.n_faces(); f++) {
            if (periodic_zone[block.face_zone[f]]) continue;
            face.assign(block.face_nodes.begin() + block.face_offsets[f],
                        block.face_nodes.begin() + block.face_offsets[f + 1]);
            for (uint64_t & node : face) node = node_key(node);
            post(face_key(face), NONE, block.face_zone[f]);
        }
    };
    for_each_face([&](const FaceKey & key, uint64_t, uint64_t) { count[hash(key) % p] += WIDTH; });
    for (int q = 0; q < p; q++) send[q].reserve(count[q]);
    for_each_face([&](const FaceKey & key, uint64_t cell, uint64_t k_or_zone) {
        auto & out = send[hash(key) % p];
        out.insert(out.end(), key.begin(), key.end());
        out.push_back(cell);
        out.push_back(k_or_zone);
    });
    comm::Received<uint64_t> received = comm::exchange(std::move(send));

    // Sorted by key, then cell: each face's cells come first, then the
    // boundary faces matching it. Two cells make a graph edge; one cell a
    // boundary face in the zone of the first matching boundary face (lowest
    // zone index), else unassigned.
    const uint64_t n_records = received.data.size() / WIDTH;
    const uint64_t * r = received.data.data();
    std::vector<uint32_t> order(n_records);
    std::iota(order.begin(), order.end(), 0u);
    std::sort(order.begin(), order.end(), [&](uint32_t x, uint32_t y) {
        return std::lexicographical_compare(r + x * WIDTH, r + (x + 1) * WIDTH, r + y * WIDTH, r + (y + 1) * WIDTH);
    });
    send.assign(p, {});
    for (uint64_t i = 0; i < n_records;) {
        uint64_t j = i + 1;
        while (j < n_records && std::equal(r + order[i] * WIDTH, r + order[i] * WIDTH + KEY, r + order[j] * WIDTH)) j++;
        uint32_t n_cells = 0;
        while (i + n_cells < j && r[order[i + n_cells] * WIDTH + KEY] != NONE) n_cells++;
        const bool zoned = i + n_cells < j;
        const uint64_t zone = zoned ? r[order[i + n_cells] * WIDTH + KEY + 1] : unassigned;
        if (n_cells == 0) {
            error = "Mesh: boundary face of " + zones[zone] + " is not a cell face.";
        } else if (n_cells > 2) {
            error = "Mesh: a face is shared by more than two cells.";
        } else if (n_cells == 2 && r[order[i] * WIDTH + KEY] == r[order[i + 1] * WIDTH + KEY]) {
            error = "Mesh: a cell touches itself across a periodic boundary; periodic directions need at least 3 cells.";
        } else if (n_cells == 2 && zoned) {
            error = "Mesh: boundary face of " + zones[zone] + " is an interior face.";
        } else {
            for (uint32_t m = 0; m < n_cells; m++) {
                const uint64_t * rec = r + order[i + m] * WIDTH;
                const uint64_t other = n_cells == 2 ? r[order[i + 1 - m] * WIDTH + KEY] : NONE;
                auto & out = send[rank_of_cell(rec[KEY])];
                out.insert(out.end(), {rec[KEY], rec[KEY + 1], other, n_cells == 2 ? uint64_t(NO_ZONE) : zone});
            }
        }
        i = j;
    }
    received = {};
    order = {};
    check_all(error);
    received = comm::exchange(std::move(send));

    // Per block cell, in local face order: neighbors and boundary faces
    std::vector<std::array<uint64_t, 4>> entries(received.data.size() / 4);
    for (size_t i = 0; i < entries.size(); i++) {
        const uint64_t * e = &received.data[4 * i];
        entries[i] = {e[0] - block.first_cell, e[1], e[2], e[3]};
    }
    received = {};
    std::sort(entries.begin(), entries.end());
    graph_offsets_.assign(block.n_cells() + 1, 0);
    boundary_offsets.assign(block.n_cells() + 1, 0);
    for (const auto & [c, k, other, zone] : entries) {
        if (other != NONE) {
            graph_offsets_[c + 1]++;
            graph_neighbors_.push_back(other);
        } else {
            boundary_offsets[c + 1]++;
            boundary.push_back({uint32_t(k), uint32_t(zone)});
        }
    }
    std::partial_sum(graph_offsets_.begin(), graph_offsets_.end(), graph_offsets_.begin());
    std::partial_sum(boundary_offsets.begin(), boundary_offsets.end(), boundary_offsets.begin());
}

std::vector<std::array<double, N_DIM>> DistributedMesh::fetch_nodes(const std::vector<uint64_t> & sorted_ids) const {
    const int p = comm::size();
    std::vector<std::vector<uint64_t>> wanted(p);
    for (uint64_t g : sorted_ids) wanted[rank_of_node(g)].push_back(g);
    const auto asked = comm::exchange(std::move(wanted));
    std::vector<std::vector<double>> answer(p);
    for (int r = 0; r < p; r++) {
        for (uint64_t g : asked.from(r)) {
            const auto & x = block.node_coords[g - block.first_node];
            answer[r].insert(answer[r].end(), x.begin(), x.end());
        }
    }
    const auto got = comm::exchange(std::move(answer));
    // Ids are sorted, so the ranks' answers come back in the same order
    std::vector<std::array<double, N_DIM>> coords(sorted_ids.size());
    for (size_t k = 0; k < coords.size(); k++) FOR_I_DIM coords[k][i] = got.data[k * N_DIM + i];
    return coords;
}

std::vector<std::array<double, N_DIM>> DistributedMesh::block_cell_centers() const {
    std::vector<uint64_t> ids(block.cell_nodes);
    std::sort(ids.begin(), ids.end());
    ids.erase(std::unique(ids.begin(), ids.end()), ids.end());
    const auto coords = fetch_nodes(ids);
    std::vector<std::array<double, N_DIM>> centers(block.n_cells());
    for (uint64_t c = 0; c < block.n_cells(); c++) {
        std::array<double, N_DIM> x{};
        for (uint64_t k = block.cell_offsets[c]; k < block.cell_offsets[c + 1]; k++) {
            const auto & y = coords[std::lower_bound(ids.begin(), ids.end(), block.cell_nodes[k]) - ids.begin()];
            FOR_I_DIM x[i] += y[i];
        }
        const double n = block.cell_offsets[c + 1] - block.cell_offsets[c];
        FOR_I_DIM x[i] /= n;
        centers[c] = x;
    }
    return centers;
}

void DistributedMesh::append_record(std::vector<uint64_t> & out, uint32_t c) const {
    out.push_back(block.first_cell + c);
    out.push_back(owner[c]);
    out.push_back(block.cell_offsets[c + 1] - block.cell_offsets[c]);
    out.insert(out.end(), block.cell_nodes.begin() + block.cell_offsets[c],
               block.cell_nodes.begin() + block.cell_offsets[c + 1]);
    out.push_back(boundary_offsets[c + 1] - boundary_offsets[c]);
    for (uint32_t i = boundary_offsets[c]; i < boundary_offsets[c + 1]; i++) {
        out.push_back(boundary[i][0]);
        out.push_back(boundary[i][1]);
    }
    if (periodic_classes.zones.empty()) return;
    // Periodic classes of the cell's nodes, as (node, key, lattice offset)
    const size_t n_at = out.size();
    out.push_back(0);
    for (uint64_t k = block.cell_offsets[c]; k < block.cell_offsets[c + 1]; k++) {
        const auto it = block_periodic.find(block.cell_nodes[k]);
        if (it == block_periodic.end()) continue;
        out.insert(out.end(), {it->first, it->second.first, pack_lattice(it->second.second)});
        out[n_at]++;
    }
}

void DistributedMesh::read_records(const std::vector<uint64_t> & from, uint8_t layer) {
    const bool periodic = !periodic_classes.zones.empty();
    {
        for (size_t i = 0; i < from.size();) {
            const uint64_t g = from[i++];
            if (layer > 0) halo_index.emplace(g, cells.gid.size());
            cells.gid.push_back(g);
            cells.owner.push_back(from[i++]);
            cells.layer.push_back(layer);
            const uint64_t n = from[i++];
            cells.nodes.insert(cells.nodes.end(), &from[i], &from[i] + n);
            cells.node_offsets.push_back(cells.nodes.size());
            i += n;
            const uint64_t nb = from[i++];
            for (uint64_t k = 0; k < nb; k++, i += 2) cells.boundary.push_back({uint32_t(from[i]), uint32_t(from[i + 1])});
            cells.boundary_offsets.push_back(cells.boundary.size());
            const uint64_t n_periodic = periodic ? from[i++] : 0;
            for (uint64_t k = 0; k < n_periodic; k++, i += 3) {
                cells.periodic.emplace(from[i], std::pair{from[i + 1], unpack_lattice(from[i + 2])});
            }
        }
    }
}

void DistributedMesh::distribute(const std::vector<int> & cell_owner) {
    const int p = comm::size();
    std::string error;
    if (cell_owner.size() != block.n_cells()) error = "DistributedMesh::distribute: one owner per block cell.";
    for (int o : cell_owner) {
        if (o < 0 || o >= p) error = "DistributedMesh::distribute: owner out of range.";
    }
    check_all(error);
    owner = cell_owner;
    // The dual graph only serves the partitioner
    graph_offsets_ = {};
    graph_neighbors_ = {};
    cells = Cells();
    halo_index.clear();
    searched_nodes.clear();
    layers = 0;

    // Every block cell to its owner. Ranks hold increasing ranges of global
    // ids, so the owned cells arrive sorted.
    std::vector<std::vector<uint64_t>> send(p);
    for (uint32_t c = 0; c < block.n_cells(); c++) append_record(send[owner[c]], c);
    read_records(comm::exchange(std::move(send)).data, 0);
    owned_nodes = cells.nodes;
    for (uint64_t & node : owned_nodes) node = node_key(node);
    std::sort(owned_nodes.begin(), owned_nodes.end());
    owned_nodes.erase(std::unique(owned_nodes.begin(), owned_nodes.end()), owned_nodes.end());

    // Node directory: the cells using each block node (all nodes of a
    // periodic class count as its key), and their owners
    send.assign(p, {});
    for (uint32_t c = 0; c < block.n_cells(); c++) {
        for (uint64_t k = block.cell_offsets[c]; k < block.cell_offsets[c + 1]; k++) {
            const uint64_t key = node_key(block.cell_nodes[k]);
            send[rank_of_node(key)].insert(send[rank_of_node(key)].end(), {key, block.first_cell + c, uint64_t(owner[c])});
        }
    }
    const std::vector<uint64_t> uses = comm::exchange(std::move(send)).data;
    directory_offsets.assign(block.n_nodes() + 1, 0);
    for (size_t i = 0; i < uses.size(); i += 3) directory_offsets[uses[i] - block.first_node + 1]++;
    std::partial_sum(directory_offsets.begin(), directory_offsets.end(), directory_offsets.begin());
    directory.assign(directory_offsets.back(), {});
    {
        std::vector<uint64_t> fill(directory_offsets.begin(), directory_offsets.end() - 1);
        for (size_t i = 0; i < uses.size(); i += 3) {
            directory[fill[uses[i] - block.first_node]++] = {uses[i + 1], int(uses[i + 2])};
        }
    }

    // Halo layer 1: at every node shared by several owners, each owner gets
    // the cells there that it does not own
    send.assign(p, {});
    std::vector<int> owners;
    for (uint64_t n = 0; n < block.n_nodes(); n++) {
        owners.clear();
        for (uint64_t i = directory_offsets[n]; i < directory_offsets[n + 1]; i++) owners.push_back(directory[i].second);
        std::sort(owners.begin(), owners.end());
        owners.erase(std::unique(owners.begin(), owners.end()), owners.end());
        if (owners.size() < 2) continue;
        for (int o : owners) {
            for (uint64_t i = directory_offsets[n]; i < directory_offsets[n + 1]; i++) {
                if (directory[i].second != o) send[o].insert(send[o].end(), {directory[i].first, uint64_t(directory[i].second)});
            }
        }
    }
    next_layer.clear();
    const std::vector<uint64_t> pushed = comm::exchange(std::move(send)).data;
    for (size_t i = 0; i < pushed.size(); i += 2) next_layer.push_back({pushed[i], int(pushed[i + 1])});
    std::sort(next_layer.begin(), next_layer.end());
    next_layer.erase(std::unique(next_layer.begin(), next_layer.end()), next_layer.end());
}

void DistributedMesh::grow_layer() {
    const int p = comm::size();
    const uint8_t layer = layers + 1;
    std::vector<std::pair<uint64_t, int>> found;
    if (layer == 1) {
        found.swap(next_layer);
    } else {
        // Ask the directory for the cells around the previous layer's nodes;
        // the cells around nodes of owned cells are all here already
        std::vector<std::vector<uint64_t>> query(p);
        for (uint32_t i = 0; i < cells.gid.size(); i++) {
            if (cells.layer[i] != layer - 1) continue;
            for (uint64_t k = cells.node_offsets[i]; k < cells.node_offsets[i + 1]; k++) {
                const uint64_t node = node_key(cells.nodes[k]);
                if (std::binary_search(owned_nodes.begin(), owned_nodes.end(), node)) continue;
                if (searched_nodes.insert(node).second) query[rank_of_node(node)].push_back(node);
            }
        }
        const auto asked = comm::exchange(std::move(query));
        std::vector<std::vector<uint64_t>> answer(p);
        for (int r = 0; r < p; r++) {
            for (uint64_t node : asked.from(r)) {
                const uint64_t n = node - block.first_node;
                for (uint64_t i = directory_offsets[n]; i < directory_offsets[n + 1]; i++) {
                    answer[r].insert(answer[r].end(), {directory[i].first, uint64_t(directory[i].second)});
                }
            }
        }
        const auto n_owned = cells.gid.begin() + std::count(cells.layer.begin(), cells.layer.end(), 0);
        const std::vector<uint64_t> around = comm::exchange(std::move(answer)).data;
        for (size_t i = 0; i < around.size(); i += 2) {
            const uint64_t g = around[i];
            if (std::binary_search(cells.gid.begin(), n_owned, g) || halo_index.count(g)) continue;
            found.push_back({g, int(around[i + 1])});
        }
        std::sort(found.begin(), found.end());
        found.erase(std::unique(found.begin(), found.end()), found.end());
    }

    // Fetch the new cells from the ranks whose blocks hold them, in global order
    std::vector<std::vector<uint64_t>> wanted(p);
    for (const auto & [g, o] : found) wanted[rank_of_cell(g)].push_back(g);
    const auto asked = comm::exchange(std::move(wanted));
    std::vector<std::vector<uint64_t>> records(p);
    for (int r = 0; r < p; r++) {
        for (uint64_t g : asked.from(r)) append_record(records[r], g - block.first_cell);
    }
    read_records(comm::exchange(std::move(records)).data, layer);
    layers = layer;
}

std::shared_ptr<Mesh> DistributedMesh::build_local_mesh(int halo_layers, Distribution & dist) {
    while (layers < halo_layers) grow_layer();
    const uint32_t n_local = std::count_if(cells.layer.begin(), cells.layer.end(),
                                           [&](uint8_t l) { return l <= halo_layers; });

    // Nodes, numbered in order of first use
    std::unordered_map<uint64_t, uint32_t> local_node;
    std::vector<uint64_t> node_ids;
    std::vector<std::vector<uint32_t>> local_cells(n_local);
    for (uint32_t i = 0; i < n_local; i++) {
        for (uint64_t k = cells.node_offsets[i]; k < cells.node_offsets[i + 1]; k++) {
            const auto [it, inserted] = local_node.emplace(cells.nodes[k], node_ids.size());
            if (inserted) node_ids.push_back(cells.nodes[k]);
            local_cells[i].push_back(it->second);
        }
    }
    std::vector<uint64_t> sorted_ids(node_ids);
    std::sort(sorted_ids.begin(), sorted_ids.end());
    const auto coords = fetch_nodes(sorted_ids);
    std::vector<std::array<rtype, N_DIM>> nodes(node_ids.size());
    for (size_t j = 0; j < node_ids.size(); j++) {
        const auto & x = coords[std::lower_bound(sorted_ids.begin(), sorted_ids.end(), node_ids[j]) - sorted_ids.begin()];
        FOR_I_DIM nodes[j][i] = x[i];
    }

    // Faces on the global boundary; the remaining one-sided faces border other ranks
    std::vector<Mesh::BoundaryFace> boundary_faces;
    std::vector<uint64_t> face;
    for (uint32_t i = 0; i < n_local; i++) {
        const uint32_t n = cells.node_offsets[i + 1] - cells.node_offsets[i];
        for (uint32_t b = cells.boundary_offsets[i]; b < cells.boundary_offsets[i + 1]; b++) {
            cell_face(&cells.nodes[cells.node_offsets[i]], n, cells.boundary[b][0], face);
            std::vector<uint32_t> local;
            for (uint64_t node : face) local.push_back(local_node.at(node));
            boundary_faces.push_back({std::move(local), zones[cells.boundary[b][1]]});
        }
    }
    // Periodic classes of the local nodes, keyed by their first local node
    PeriodicNodes classes = periodic_classes;
    if (!classes.zones.empty()) {
        std::unordered_map<uint64_t, uint32_t> local_key;
        classes.key.resize(node_ids.size());
        classes.lattice.resize(node_ids.size());
        for (uint32_t j = 0; j < node_ids.size(); j++) {
            const auto it = cells.periodic.find(node_ids[j]);
            const bool found = it != cells.periodic.end();
            classes.key[j] = local_key.emplace(found ? it->second.first : node_ids[j], j).first->second;
            classes.lattice[j] = found ? it->second.second : std::array<int8_t, 3>{0, 0, 0};
        }
    }
    dist = Distribution();
    dist.halo_layers = halo_layers;
    dist.global_cell.assign(cells.gid.begin(), cells.gid.begin() + n_local);
    dist.layer.assign(cells.layer.begin(), cells.layer.begin() + n_local);
    dist.n_owned = std::count(dist.layer.begin(), dist.layer.end(), 0);
    plan_halo_exchange(dist, std::vector<int>(cells.owner.begin() + dist.n_owned, cells.owner.begin() + n_local));

    auto mesh = std::make_shared<Mesh>();
    // Global ids first: they order the faces and neighbor lists like the serial mesh's
    mesh->h_global_cell_id = dist.global_cell;
    mesh->n_global_cells = n_global_cells();
    mesh->h_global_node_id = std::move(node_ids);
    mesh->n_global_nodes = node_dist.back();
    mesh->init_from_connectivity(nodes, local_cells, boundary_faces, PARTITION_ZONE, classes);
    mesh->n_owned_cells = dist.n_owned;
    mesh->n_reconstructed_cells = std::count_if(dist.layer.begin(), dist.layer.end(), [](uint8_t l) { return l <= 1; });
    mesh->n_complete_cells =
        std::count_if(dist.layer.begin(), dist.layer.end(), [&](uint8_t l) { return l < halo_layers; });
    return mesh;
}

/**
 * @file structured_blocks.cpp
 * @author Matthew Bonanni (mbonanni001@gmail.com)
 * @brief Structured blocks of quadrilaterals (2D) or hexahedra (3D) in a mesh's connectivity.
 * @version 0.1
 * @date 2026-11-13
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 *
 */

#include "structured_blocks.h"

#include <algorithm>
#include <bit>
#include <functional>
#include <limits>
#include <numeric>
#include <span>
#include <unordered_map>

#include "cell_order.h"

namespace {

constexpr uint32_t N_CORNERS = 1u << N_DIM;
constexpr uint32_t N_FACE_NODES = 1u << (N_DIM - 1);
constexpr uint32_t N_SIDES = 2 * N_DIM;
constexpr uint32_t NONE = std::numeric_limits<uint32_t>::max();

/** Local (i, j, k) bits of node `node` of a quadrilateral or hexahedron. */
constexpr uint32_t corner_bits(uint32_t node) {
    constexpr uint32_t quad[4] = {0b00, 0b01, 0b11, 0b10};
    return quad[node % 4] | ((node / 4) << 2);
}

/** Node with local bits `bits` (inverse of corner_bits). */
constexpr uint32_t corner_of(uint32_t bits) {
    constexpr uint32_t quad[4] = {0, 1, 3, 2};
    return quad[bits & 0b11] + 4 * ((bits >> 2) & 1);
}

using FaceKey = std::array<uint32_t, N_FACE_NODES>;

/** Sorted nodes of local side `side` (2 * axis + bit) of cell `c`. */
FaceKey face_key(std::span<const uint32_t> c, uint32_t side) {
    const uint32_t axis = side / 2, bit = side % 2;
    FaceKey key{};
    uint32_t k = 0;
    for (uint32_t b = 0; b < N_CORNERS; b++) {
        if (((b >> axis) & 1) == bit) key[k++] = c[corner_of(b)];
    }
    std::sort(key.begin(), key.end());
    return key;
}

/** A cell's orientation in its block: the local axis along each block axis, and whether the index grows with the local bit. */
struct Frame {
    std::array<uint32_t, N_DIM> axis{};
    std::array<bool, N_DIM> grows{};
    bool operator==(const Frame &) const = default;
};

using Coord = std::array<int32_t, N_DIM>;

uint64_t pack(const Coord & x) {
    uint64_t key = 0;
    for (size_t d = 0; d < N_DIM; d++) key = (key << 21) | (static_cast<uint64_t>(x[d] + (1 << 20)) & ((1u << 21) - 1));
    return key;
}

using Across = std::array<std::pair<uint32_t, uint32_t>, N_SIDES>;

/** The cell across each local side of every candidate cell, and the neighbor's side there. */
std::vector<Across> neighbors_across(const CellNodes & cells, uint32_t n_cells) {
    struct Entry {
        FaceKey key;
        uint32_t cell, side;
    };
    std::vector<Entry> faces;
    for (uint32_t c = 0; c < n_cells; c++) {
        if (cells[c].size() != N_CORNERS) continue;
        for (uint32_t s = 0; s < N_SIDES; s++) faces.push_back({face_key(cells[c], s), c, s});
    }
    std::sort(faces.begin(), faces.end(), [](const Entry & a, const Entry & b) {
        return a.key < b.key || (a.key == b.key && a.cell < b.cell);
    });
    std::vector<Across> across(n_cells);
    for (auto & a : across) a.fill({NONE, 0});
    for (size_t i = 0; i < faces.size();) {
        size_t j = i + 1;
        while (j < faces.size() && faces[j].key == faces[i].key) j++;
        if (j - i == 2 && faces[i].cell != faces[i + 1].cell) {
            across[faces[i].cell][faces[i].side] = {faces[i + 1].cell, faces[i + 1].side};
            across[faces[i + 1].cell][faces[i + 1].side] = {faces[i].cell, faces[i].side};
        }
        i = j;
    }
    return across;
}

/**
 * Frame of neighbor `o` of cell `c` across c's side in block direction (g, up),
 * o's side there being `o_side`; false if their shared nodes do not match.
 */
bool frame_across(std::span<const uint32_t> c, const Frame & fc, std::span<const uint32_t> o,
                  uint32_t o_side, uint32_t g, bool up, Frame & fo) {
    fo.axis[g] = o_side / 2;
    fo.grows[g] = (o_side % 2 == 0) == up;
    const uint32_t bits_a = (fc.grows[g] == up ? 1u : 0u) << fc.axis[g];
    for (uint32_t h = 0; h < N_DIM; h++) {
        if (h == g) continue;
        // An edge of the shared face along h, from its corner with c's other local bits 0
        const uint32_t node_a = c[corner_of(bits_a)], node_b = c[corner_of(bits_a | (1u << fc.axis[h]))];
        uint32_t oa = NONE, ob = NONE;
        for (uint32_t k = 0; k < N_CORNERS; k++) {
            if (o[k] == node_a) oa = corner_bits(k);
            if (o[k] == node_b) ob = corner_bits(k);
        }
        if (oa == NONE || ob == NONE || std::popcount(oa ^ ob) != 1) return false;
        fo.axis[h] = static_cast<uint32_t>(std::countr_zero(oa ^ ob));
        if (fo.axis[h] == fo.axis[g]) return false;
        fo.grows[h] = ((ob & (oa ^ ob)) != 0) == fc.grows[h];
    }
    return true;
}

} // namespace

std::vector<StructuredBlock> structured_blocks(const CellNodes & cells, const uint32_t n_cells,
                                               const uint32_t min_extent, std::vector<uint32_t> & block_cells) {
    std::vector<std::pair<std::array<uint32_t, N_DIM>, std::vector<uint32_t>>> boxes;
    const std::vector<Across> across = neighbors_across(cells, n_cells);
    std::vector<uint8_t> labeled(n_cells, 0), used(n_cells, 0);
    std::vector<Frame> frame(n_cells);
    std::vector<Coord> coord(n_cells);
    std::vector<uint32_t> member(n_cells, NONE), members;
    // neighbor[member[c]][2 g + up]: the cell one away from c along axis g, where their faces agree
    std::vector<std::array<uint32_t, N_SIDES>> neighbor;
    std::unordered_map<uint64_t, uint32_t> at;
    for (uint32_t seed = 0; seed < n_cells; seed++) {
        if (cells[seed].size() != N_CORNERS || labeled[seed]) continue;
        // Indices from the seed along its local axes, across faces whose nodes agree
        at.clear();
        members.assign(1, seed);
        neighbor.assign(1, {});
        neighbor[0].fill(NONE);
        labeled[seed] = 1;
        member[seed] = 0;
        std::iota(frame[seed].axis.begin(), frame[seed].axis.end(), 0u);
        frame[seed].grows.fill(true);
        coord[seed] = Coord{};
        at[pack(coord[seed])] = seed;
        for (size_t q = 0; q < members.size(); q++) {
            const uint32_t c = members[q];
            for (uint32_t g = 0; g < N_DIM; g++) {
                for (uint32_t up = 0; up < 2; up++) {
                    const uint32_t side = 2 * frame[c].axis[g] + (frame[c].grows[g] == (up == 1) ? 1 : 0);
                    const auto [o, o_side] = across[c][side];
                    if (o == NONE) continue;
                    Frame fo;
                    if (!frame_across(cells[c], frame[c], cells[o], o_side, g, up == 1, fo)) continue;
                    Coord x = coord[c];
                    x[g] += up ? 1 : -1;
                    if (!labeled[o]) {
                        // A second cell at one index (an O-grid's seam) ends the block there
                        if (!at.emplace(pack(x), o).second) continue;
                        labeled[o] = 1;
                        frame[o] = fo;
                        coord[o] = x;
                        member[o] = static_cast<uint32_t>(members.size());
                        members.push_back(o);
                        neighbor.emplace_back().fill(NONE);
                    } else if (member[o] == NONE || coord[o] != x || !(frame[o] == fo)) {
                        continue;
                    }
                    neighbor[q][2 * g + up] = o;
                }
            }
        }
        // Block axes in the order of the seed's nearest neighbor along each in the given numbering, indices growing with it
        std::array<uint32_t, N_DIM> gap, axes;
        std::array<bool, N_DIM> flip{};
        for (uint32_t g = 0; g < N_DIM; g++) {
            gap[g] = NONE;
            for (uint32_t up = 0; up < 2; up++) {
                const uint32_t o = neighbor[0][2 * g + up];
                if (o == NONE || (o > seed ? o - seed : seed - o) >= gap[g]) continue;
                gap[g] = o > seed ? o - seed : seed - o;
                flip[g] = (o > seed) != (up == 1);
            }
        }
        std::iota(axes.begin(), axes.end(), 0u);
        std::stable_sort(axes.begin(), axes.end(), [&](uint32_t a, uint32_t b) { return gap[a] < gap[b]; });
        auto index = [&](uint32_t c) {
            Coord x;
            for (uint32_t a = 0; a < N_DIM; a++) x[a] = flip[axes[a]] ? -coord[c][axes[a]] : coord[c][axes[a]];
            return x;
        };
        auto next = [&](uint32_t c, uint32_t a) { return neighbor[member[c]][2 * axes[a] + (flip[axes[a]] ? 0 : 1)]; };
        std::sort(members.begin(), members.end(), [&](uint32_t a, uint32_t b) {
            const Coord xa = index(a), xb = index(b);
            return std::lexicographical_compare(xa.rbegin(), xa.rend(), xb.rbegin(), xb.rend());
        });
        // Boxes, greedily: from each unused cell in lexicographic order, as long as possible along i, then j, then k
        std::vector<uint32_t> layer, ahead;
        for (const uint32_t start : members) {
            if (used[start]) continue;
            std::array<uint32_t, N_DIM> n;
            n.fill(1);
            std::vector<uint32_t> box = {start};
            used[start] = 2;
            for (uint32_t a = 0; a < N_DIM; a++) {
                layer = box;
                while (true) {
                    // The next layer along a: unused cells, neighbors along the axes below a as in the box
                    ahead.resize(layer.size());
                    bool ok = true;
                    for (size_t k = 0; k < layer.size() && ok; k++) {
                        ahead[k] = next(layer[k], a);
                        ok = ahead[k] != NONE && !used[ahead[k]];
                    }
                    size_t stride = 1;
                    for (uint32_t b = 0; b < a && ok; b++) {
                        for (size_t k = 0; k < ahead.size() && ok; k++) {
                            if ((k / stride) % n[b] + 1 < n[b]) ok = next(ahead[k], b) == ahead[k + stride];
                        }
                        stride *= n[b];
                    }
                    if (!ok) break;
                    for (uint32_t c : ahead) used[c] = 2;
                    box.insert(box.end(), ahead.begin(), ahead.end());
                    layer.swap(ahead);
                    n[a]++;
                }
            }
            // A box too short along an axis leaves its cells unstructured
            for (uint32_t c : box) used[c] = 1;
            if (std::all_of(n.begin(), n.end(), [&](uint32_t e) { return e >= min_extent; })) {
                boxes.emplace_back(n, std::move(box));
            }
        }
        for (uint32_t c : members) member[c] = NONE;
    }
    std::sort(boxes.begin(), boxes.end(), [](const auto & a, const auto & b) {
        return *std::min_element(a.second.begin(), a.second.end()) < *std::min_element(b.second.begin(), b.second.end());
    });
    std::vector<StructuredBlock> blocks;
    block_cells.clear();
    for (const auto & [n, box] : boxes) {
        blocks.push_back({static_cast<uint32_t>(block_cells.size()), n});
        block_cells.insert(block_cells.end(), box.begin(), box.end());
    }
    return blocks;
}

uint32_t structured_block_cells(const std::vector<StructuredBlock> & blocks) {
    if (blocks.empty()) return 0;
    const StructuredBlock & last = blocks.back();
    return last.first + std::accumulate(last.n.begin(), last.n.end(), 1u, std::multiplies<>());
}

/**
 * @file structured_blocks_test.cpp
 * @author Matthew Bonanni (mbonanni001@gmail.com)
 * @brief Structured blocks found in a mesh's connectivity, and the cell numbering that puts them first.
 * @version 0.1
 * @date 2026-11-13
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 *
 */

#include <gtest/gtest.h>

#include <algorithm>
#include <bit>
#include <numeric>
#include <random>
#include <set>
#include <vector>

#include "cell_order.h"
#include "mesh.h"
#include "structured_blocks.h"
#include "test_fixtures.h"

namespace {

constexpr uint32_t N_CORNERS = 1u << N_DIM;
using Cells = std::vector<std::vector<uint32_t>>;

struct Box {
    std::vector<std::array<rtype, N_DIM>> nodes;
    Cells cells;
};

/** Corner `b` (bits i, j, k) of a cell in Mallard's node order. */
uint32_t corner_of(uint32_t b) {
    constexpr uint32_t quad[4] = {0, 1, 3, 2};
    return quad[b & 3] + 4 * (b >> 2);
}

/** A box of n cells, numbered fastest along axes[0], then axes[1]. */
Box make_box(std::array<uint32_t, N_DIM> n, std::array<uint32_t, N_DIM> axes) {
    Box box;
    std::array<uint32_t, N_DIM> m;
    for (size_t d = 0; d < N_DIM; d++) m[d] = n[d] + 1;
    const uint32_t n_nodes = std::accumulate(m.begin(), m.end(), 1u, std::multiplies<>());
    for (uint32_t v = 0; v < n_nodes; v++) {
        std::array<rtype, N_DIM> p{};
        for (uint32_t d = 0, r = v; d < N_DIM; r /= m[d], d++) p[d] = rtype(r % m[d]);
        box.nodes.push_back(p);
    }
    auto node = [&](std::array<uint32_t, N_DIM> x) {
        uint32_t v = 0;
        for (int d = N_DIM - 1; d >= 0; d--) v = v * m[size_t(d)] + x[size_t(d)];
        return v;
    };
    const uint32_t n_cells = std::accumulate(n.begin(), n.end(), 1u, std::multiplies<>());
    for (uint32_t c = 0; c < n_cells; c++) {
        std::array<uint32_t, N_DIM> x;
        for (uint32_t a = 0, r = c; a < N_DIM; r /= n[axes[a]], a++) x[axes[a]] = r % n[axes[a]];
        std::vector<uint32_t> cell(N_CORNERS);
        for (uint32_t b = 0; b < N_CORNERS; b++) {
            std::array<uint32_t, N_DIM> y = x;
            for (size_t d = 0; d < N_DIM; d++) y[d] += (b >> d) & 1;
            cell[corner_of(b)] = node(y);
        }
        box.cells.push_back(cell);
    }
    return box;
}

std::array<uint32_t, N_DIM> natural() {
    std::array<uint32_t, N_DIM> axes;
    std::iota(axes.begin(), axes.end(), 0u);
    return axes;
}

uint32_t shared_nodes(const std::vector<uint32_t> & a, const std::vector<uint32_t> & b) {
    uint32_t n = 0;
    for (uint32_t v : a) n += static_cast<uint32_t>(std::count(b.begin(), b.end(), v));
    return n;
}

/**
 * Every block is a box: the cells one index apart share a face, the cells one
 * index before and after a cell along an axis share no node (lines are
 * straight), no cell is in two blocks, and blocks are at least min_extent long.
 */
void expect_boxes(const Cells & cells, const std::vector<StructuredBlock> & blocks,
                  const std::vector<uint32_t> & block_cells, uint32_t min_extent) {
    std::set<uint32_t> seen(block_cells.begin(), block_cells.end());
    ASSERT_EQ(seen.size(), block_cells.size());
    uint32_t total = 0;
    for (const StructuredBlock & block : blocks) {
        ASSERT_EQ(block.first, total);
        std::array<uint32_t, N_DIM> stride;
        uint32_t size = 1;
        for (size_t a = 0; a < N_DIM; a++) {
            ASSERT_GE(block.n[a], min_extent);
            stride[a] = size;
            size *= block.n[a];
        }
        total += size;
        ASSERT_LE(total, block_cells.size());
        for (uint32_t l = 0; l < size; l++) {
            const auto & c = cells[block_cells[block.first + l]];
            for (size_t a = 0; a < N_DIM; a++) {
                const uint32_t i = (l / stride[a]) % block.n[a];
                if (i + 1 < block.n[a]) {
                    ASSERT_EQ(shared_nodes(c, cells[block_cells[block.first + l + stride[a]]]), N_CORNERS / 2);
                }
                if (i > 0 && i + 1 < block.n[a]) {
                    ASSERT_EQ(shared_nodes(cells[block_cells[block.first + l - stride[a]]],
                                           cells[block_cells[block.first + l + stride[a]]]),
                              0u);
                }
            }
        }
    }
    EXPECT_EQ(total, block_cells.size());
}

uint32_t count(const Cells & cells) { return static_cast<uint32_t>(cells.size()); }

CellNodes nodes_of(const Cells & cells) {
    CellNodes out;
    for (const auto & c : cells) out.add(c);
    return out;
}

} // namespace

/** @brief A box keeps its numbering: its fastest axis becomes i, whichever it is. */
TEST(StructuredBlocks, ABoxIsOneBlockInItsOwnNumbering) {
    std::array<uint32_t, N_DIM> n;
    n[0] = 5;
    n[1] = 7;
    if constexpr (N_DIM == 3) n[N_DIM - 1] = 4;
    std::array<uint32_t, N_DIM> axes = natural();
    do {
        const Box box = make_box(n, axes);
        std::vector<uint32_t> block_cells;
        const auto blocks = structured_blocks(nodes_of(box.cells), count(box.cells), 3, block_cells);
        ASSERT_EQ(blocks.size(), 1u);
        for (size_t a = 0; a < N_DIM; a++) EXPECT_EQ(blocks[0].n[a], n[axes[a]]);
        std::vector<uint32_t> identity(box.cells.size());
        std::iota(identity.begin(), identity.end(), 0u);
        EXPECT_EQ(block_cells, identity);
    } while (std::next_permutation(axes.begin(), axes.end()));
}

/** @brief Shuffled cells, each with its nodes in any of the orientations that keep it right-handed, make one block. */
TEST(StructuredBlocks, AShuffledBoxWithRotatedCellsIsOneBlock) {
    const uint32_t m = N_DIM == 2 ? 12 : 6;
    std::array<uint32_t, N_DIM> n;
    n.fill(m);
    Box box = make_box(n, natural());
    std::mt19937 rng(3);
    std::shuffle(box.cells.begin(), box.cells.end(), rng);
    // Cube symmetries of even parity: an axis permutation and flips
    std::vector<std::vector<uint32_t>> symmetries;
    std::array<uint32_t, N_DIM> perm = natural();
    do {
        int parity = 1;
        for (size_t a = 0; a < N_DIM; a++) {
            for (size_t b = a + 1; b < N_DIM; b++) parity *= perm[a] > perm[b] ? -1 : 1;
        }
        for (uint32_t flips = 0; flips < N_CORNERS; flips++) {
            if (parity * (std::popcount(flips) % 2 ? -1 : 1) < 0) continue;
            std::vector<uint32_t> map(N_CORNERS);
            for (uint32_t b = 0; b < N_CORNERS; b++) {
                uint32_t image = 0;
                for (size_t a = 0; a < N_DIM; a++) image |= (((b >> perm[a]) ^ (flips >> a)) & 1) << a;
                map[corner_of(b)] = corner_of(image);
            }
            symmetries.push_back(map);
        }
    } while (std::next_permutation(perm.begin(), perm.end()));
    ASSERT_EQ(symmetries.size(), N_DIM == 2 ? 4u : 24u);
    for (auto & cell : box.cells) {
        const auto & map = symmetries[rng() % symmetries.size()];
        std::vector<uint32_t> turned(N_CORNERS);
        for (uint32_t k = 0; k < N_CORNERS; k++) turned[map[k]] = cell[k];
        cell = turned;
    }
    std::vector<uint32_t> block_cells;
    const auto blocks = structured_blocks(nodes_of(box.cells), count(box.cells), 3, block_cells);
    ASSERT_EQ(blocks.size(), 1u);
    EXPECT_EQ(block_cells.size(), box.cells.size());
    expect_boxes(box.cells, blocks, block_cells, 3);
}

/** @brief Boxes shorter than min_extent along an axis stay unstructured; only cells [0, n_cells) are considered. */
TEST(StructuredBlocks, ShortBoxesAndCellsPastNCellsStayUnstructured) {
    std::array<uint32_t, N_DIM> n;
    n.fill(8);
    n[1] = 2;
    const Box box = make_box(n, natural());
    std::vector<uint32_t> block_cells;
    EXPECT_TRUE(structured_blocks(nodes_of(box.cells), count(box.cells), 3, block_cells).empty());
    EXPECT_EQ(structured_blocks(nodes_of(box.cells), count(box.cells), 2, block_cells).size(), 1u);
    // The first two rows along i
    EXPECT_EQ(structured_blocks(nodes_of(box.cells), 16, 1, block_cells).size(), 1u);
    EXPECT_EQ(block_cells.size(), 16u);
}

/** @brief An L-shaped region splits into boxes that cover it; triangles or tetrahedra among the cells are skipped. */
TEST(StructuredBlocks, AnLShapeSplitsIntoBoxes) {
    std::array<uint32_t, N_DIM> n;
    n.fill(N_DIM == 2 ? 6 : 3);
    n[0] = n[1] = 6;
    Box box = make_box(n, natural());
    Cells cells;
    for (const auto & c : box.cells) {
        const auto & corner = box.nodes[c[0]];
        if (corner[0] < rtype(3) || corner[1] < rtype(3)) cells.push_back(c);
    }
    const uint32_t n_hex = count(cells);
    cells.push_back(std::vector<uint32_t>(box.cells.back().begin(), box.cells.back().begin() + N_DIM + 1));
    std::vector<uint32_t> block_cells;
    const auto blocks = structured_blocks(nodes_of(cells), count(cells), 3, block_cells);
    EXPECT_EQ(blocks.size(), 2u);
    EXPECT_EQ(block_cells.size(), n_hex);
    expect_boxes(cells, blocks, block_cells, 3);
}

/** @brief A ring of cells closes on itself: it is cut once, and cells around a vertex shared by three blocks stay boxes. */
TEST(StructuredBlocks, ARingIsCutOnceAndThreeBlocksMeetingAtAVertexStayBoxes) {
    const uint32_t around = 16, radial = 4, layers = N_DIM == 2 ? 1 : 3;
    // Ring: node (t, r, z) with t periodic
    auto ring_node = [&](uint32_t t, uint32_t r, uint32_t z) { return (z * (radial + 1) + r) * around + t % around; };
    Cells ring;
    for (uint32_t z = 0; z < layers; z++) {
        for (uint32_t r = 0; r < radial; r++) {
            for (uint32_t t = 0; t < around; t++) {
                std::vector<uint32_t> c = {ring_node(t, r, z), ring_node(t + 1, r, z), ring_node(t + 1, r + 1, z),
                                           ring_node(t, r + 1, z)};
                if constexpr (N_DIM == 3) {
                    for (uint32_t k = 0; k < 4; k++) c.push_back(c[k] + (radial + 1) * around);
                }
                ring.push_back(c);
            }
        }
    }
    std::vector<uint32_t> block_cells;
    auto blocks = structured_blocks(nodes_of(ring), count(ring), 3, block_cells);
    ASSERT_EQ(blocks.size(), 1u);
    EXPECT_EQ(blocks[0].n[0], around);
    EXPECT_EQ(block_cells.size(), ring.size());
    expect_boxes(ring, blocks, block_cells, 3);

    // Three m x m patches around a center vertex, as at a corner of a cube's surface: patch p's (u, v) node is shared
    // with patch p + 1 along u = 0 and with patch p - 1 along v = 0
    const uint32_t m = 6, per = (m + 1) * (m + 1);
    auto patch_node = [&](uint32_t p, uint32_t u, uint32_t v, uint32_t z) {
        if (u == 0) return z * 3 * per + (v == 0 ? 0 : ((p + 1) % 3) * per + v * (m + 1));  // node (p + 1, v, 0)
        return z * 3 * per + p * per + u * (m + 1) + v;
    };
    Cells star;
    for (uint32_t z = 0; z < layers; z++) {
        for (uint32_t p = 0; p < 3; p++) {
            for (uint32_t v = 0; v < m; v++) {
                for (uint32_t u = 0; u < m; u++) {
                    std::vector<uint32_t> c = {patch_node(p, u, v, z), patch_node(p, u + 1, v, z),
                                               patch_node(p, u + 1, v + 1, z), patch_node(p, u, v + 1, z)};
                    if constexpr (N_DIM == 3) {
                        for (uint32_t k = 0; k < 4; k++) c.push_back(c[k] + 3 * per);
                    }
                    star.push_back(c);
                }
            }
        }
    }
    blocks = structured_blocks(nodes_of(star), count(star), 3, block_cells);
    EXPECT_GE(blocks.size(), 2u);
    expect_boxes(star, blocks, block_cells, 3);
}

/** @brief Auto numbers the blocks' cells first and the others after them in their order; input and Hilbert find none. */
TEST(StructuredBlocks, AutoNumbersBlocksFirst) {
    std::array<uint32_t, N_DIM> n;
    n.fill(N_DIM == 2 ? 10 : 6);
    Box box = make_box(n, natural());
    std::mt19937 rng(5);
    std::shuffle(box.cells.begin(), box.cells.end(), rng);
    const uint32_t n_cells = count(box.cells);
    const LocalityOrder numbered = locality_order({CellOrder::AUTO, 2, true}, box.nodes, nodes_of(box.cells), {n_cells});
    ASSERT_EQ(numbered.blocks.size(), 1u);
    EXPECT_EQ(numbered.blocks[0].first, 0u);
    std::vector<uint32_t> sorted = numbered.order;
    std::sort(sorted.begin(), sorted.end());
    for (uint32_t c = 0; c < n_cells; c++) ASSERT_EQ(sorted[c], c);
    Cells renumbered(n_cells);
    for (uint32_t c = 0; c < n_cells; c++) renumbered[c] = box.cells[numbered.order[c]];
    std::vector<uint32_t> identity(n_cells);
    std::iota(identity.begin(), identity.end(), 0u);
    expect_boxes(renumbered, numbered.blocks, identity, 5);
    EXPECT_TRUE(locality_order({CellOrder::AUTO, 2, false}, box.nodes, nodes_of(box.cells), {n_cells}).blocks.empty());
    EXPECT_TRUE(locality_order({CellOrder::HILBERT, 2, false}, box.nodes, nodes_of(box.cells), {n_cells}).blocks.empty());

    // Halo groups keep their place, and blocks are only among the owned cells
    const uint32_t n_owned = n_cells / 2;
    const LocalityOrder split = locality_order({CellOrder::AUTO, 1, true}, box.nodes, nodes_of(box.cells), {n_owned, n_cells});
    for (uint32_t c = 0; c < n_cells; c++) ASSERT_EQ(split.order[c] < n_owned, c < n_owned);
}

/** @brief A generated mesh under the default cell order records its blocks. */
TEST(StructuredBlocks, AGeneratedMeshRecordsItsBlocks) {
    const auto mesh = N_DIM == 2 ? make_mesh("cartesian", 8, 6) : make_mesh_3d("cartesian", 8, 6, 5);
    ASSERT_EQ(mesh->structured_blocks.size(), 1u);
    uint32_t cells = 1;
    for (uint32_t e : mesh->structured_blocks[0].n) cells *= e;
    EXPECT_EQ(cells, mesh->n_cells);
    const auto tets = N_DIM == 2 ? make_mesh("cartesian_tri", 8, 6) : make_mesh_3d("cartesian_tet", 4, 3, 3);
    EXPECT_TRUE(tets->structured_blocks.empty());
}

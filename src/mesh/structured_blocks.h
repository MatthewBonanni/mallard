/**
 * @file structured_blocks.h
 * @author Matthew Bonanni (mbonanni001@gmail.com)
 * @brief Structured blocks of quadrilaterals (2D) or hexahedra (3D) in a mesh's connectivity.
 * @version 0.1
 * @date 2026-11-13
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 *
 */

#ifndef STRUCTURED_BLOCKS_H
#define STRUCTURED_BLOCKS_H

#include <array>
#include <cstdint>
#include <vector>

#include "common_typedef.h"

/**
 * @brief A box of cells numbered lexicographically, i fastest: the cell at
 *        (i, j, k) is first + i + n[0] (j + n[1] k), and the cell across each of
 *        its faces is the one at an index one away (see
 *        docs/design/structured_blocks.md).
 */
struct StructuredBlock {
    uint32_t first = 0;
    std::array<uint32_t, N_DIM> n{};
};

/**
 * @brief Structured blocks among cells [0, n_cells) of `cells` (node lists in
 *        Mallard's order: quadrilaterals counterclockwise, hexahedra as two
 *        quadrilaterals, nodes 4-7 above 0-3). Two cells are neighbors in a
 *        block if they share a whole face; a connected set of such cells takes
 *        (i, j, k) indices along its faces and is cut into boxes, greedily in
 *        lexicographic order, keeping those at least min_extent cells long
 *        along every axis. block_cells receives the blocks' cells, block after
 *        block, each i fastest, and a block's first is its position there.
 *        Blocks come in the order of their lowest cell. A block's i axis is the
 *        one along which the lowest cell of its connected set has its nearest
 *        neighbor in the given numbering, then j, and indices grow with it, so
 *        that a box already numbered i fastest keeps its numbering.
 */
std::vector<StructuredBlock> structured_blocks(const std::vector<std::vector<uint32_t>> & cells, uint32_t n_cells,
                                               uint32_t min_extent, std::vector<uint32_t> & block_cells);

#endif // STRUCTURED_BLOCKS_H

/**
 * @file stencil_shapes.h
 * @author Matthew Bonanni (mbonanni001@gmail.com)
 * @brief Stencil shapes of the cells of structured blocks.
 * @version 0.1
 * @date 2026-11-13
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 *
 */

#ifndef STENCIL_SHAPES_H
#define STENCIL_SHAPES_H

#include <cstdint>
#include <vector>

#include "mesh.h"

/**
 * @brief Cells of the structured blocks grouped by the geometry around them
 *        (see docs/design/structured_blocks.md, Detection).
 */
struct StencilShapes {
    static constexpr uint32_t NONE = UINT32_MAX;
    // For each of the blocks' cells [0, structured_block_cells()): its shape, or NONE
    std::vector<uint32_t> shape;
    // For each shape: its lowest cell
    std::vector<uint32_t> first;
};

/**
 * @brief Stencil shapes of the blocks' cells: two cells have one shape if
 *        everything within `reach` vertex layers around them has the same
 *        geometry relative to the cell, to rounding (1e-9 of the cell's
 *        h = V^(1/N_DIM), and h to 1e-9 relative): the cells, by their nodes,
 *        and the boundary faces of those cells, by their nodes and zone. A cell
 *        whose reach lies in its block is keyed through the block's indices;
 *        the others through their vertex neighbors, across periodic
 *        translations; the two ways never give one shape. Round-off can split
 *        a shape in two; cells whose
 *        surroundings differ by more than the rounding never share one. A cell
 *        has none if its reach leaves the cells on this rank, crosses a
 *        periodic rotation, or reaches a cell across two periodic shifts.
 *        `reach` must cover all the cells the reconstruction's setup reads
 *        (TENO: its gather depth).
 */
StencilShapes stencil_shapes(const Mesh & mesh, uint32_t reach);

#endif // STENCIL_SHAPES_H

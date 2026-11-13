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
    // For each of the blocks' cells [0, structured_block_cells()): its shape, NONE within reach of a face of its block
    std::vector<uint32_t> shape;
    // For each shape: its lowest cell
    std::vector<uint32_t> first;
};

/**
 * @brief Stencil shapes of the blocks' cells at least `reach` cells from every
 *        face of their block: two such cells have one shape if the cells within
 *        reach along every axis around them have the same nodes, relative to
 *        the cell's vertex average and in units of h = V^(1/N_DIM), rounded to
 *        1e-9, and the same h, to 1e-9 of the first block cell's. Round-off can
 *        split a shape in two, never join two.
 */
StencilShapes stencil_shapes(const Mesh & mesh, uint32_t reach);

#endif // STENCIL_SHAPES_H

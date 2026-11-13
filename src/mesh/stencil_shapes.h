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
 *        reach along every axis around them have the same geometry relative to
 *        the cell, to rounding: each cell's nodes relative to its vertex
 *        average and the offset and size of the next cell along each axis, in
 *        units of its h = V^(1/N_DIM), rounded to 1e-9, and the cell's h, to
 *        1e-9 relative. Round-off can split a shape in two; cells whose
 *        geometry within reach differs by more than the rounding never share
 *        one. `reach` must cover all the cells the reconstruction's setup
 *        reads (TENO: its gather depth).
 */
StencilShapes stencil_shapes(const Mesh & mesh, uint32_t reach);

#endif // STENCIL_SHAPES_H

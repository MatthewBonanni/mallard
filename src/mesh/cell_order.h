/**
 * @file cell_order.h
 * @author Matthew Bonanni (mbonanni001@gmail.com)
 * @brief Local cell numbering for memory locality.
 * @version 0.1
 * @date 2026-10-15
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 *
 */

#ifndef CELL_ORDER_H
#define CELL_ORDER_H

#include <array>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

#include "common_typedef.h"
#include "input.h"
#include "structured_blocks.h"

/** @brief How a rank numbers its cells (see docs/design/mpi.md, section 6). */
enum class CellOrder {
    INPUT,    // as given: global order (owned cells, then each halo layer)
    HILBERT,  // along the Hilbert curve of the vertex averages
    AUTO      // whichever of the two touches fewer cache lines (see locality_order)
};

static const std::unordered_map<std::string, CellOrder> CELL_ORDERS = {
    {"auto", CellOrder::AUTO}, {"input", CellOrder::INPUT}, {"hilbert", CellOrder::HILBERT}};

/**
 * @brief A cell order and, for auto, the vertex layers around a cell that its
 *        reconstruction reads, and whether structured blocks of owned cells
 *        come first.
 */
struct CellNumbering {
    CellOrder order = CellOrder::INPUT;
    uint32_t layers = 1;
    bool blocks = false;
};

/** @brief [mesh] cell_order (default auto, with blocks), with the layers of [numerics.face_reconstruction]. */
CellNumbering cell_numbering_of_input(const Input & input);

/** @brief The order a numbering chose, and auto's cache lines per owned cell in each order (0 otherwise). */
struct CellOrderChoice {
    CellOrder order = CellOrder::INPUT;
    double input_lines = 0.0;
    double hilbert_lines = 0.0;
};

struct LocalityOrder {
    std::vector<uint32_t> order;  // new -> old
    CellOrderChoice choice;
    std::vector<StructuredBlock> blocks;  // in the new numbering
};

/**
 * @brief Permutation of the cells that keeps every group [group_ends[k - 1],
 *        group_ends[k]) in place and numbers each by locality; group_ends ends
 *        with cells.size(). Ties keep the given order. Auto counts, for both
 *        orders, the 128-byte lines of a cell field that blocks of 256
 *        consecutive owned cells (group 0) read when each reads the cells
 *        within numbering.layers vertex layers, and takes the Hilbert order
 *        only if it needs a tenth fewer. With numbering.blocks, the structured
 *        blocks of the owned cells at least 2 layers + 1 cells long come first,
 *        each numbered i fastest, then the other owned cells in the chosen
 *        order.
 */
LocalityOrder locality_order(const CellNumbering & numbering, const std::vector<std::array<rtype, N_DIM>> & nodes,
                             const std::vector<std::vector<uint32_t>> & cells, const std::vector<uint32_t> & group_ends);

#endif // CELL_ORDER_H

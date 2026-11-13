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
#include <span>
#include <string>
#include <unordered_map>
#include <vector>

#include "common_typedef.h"
#include "input.h"

/** @brief The nodes of every cell (CSR: cell c holds nodes[offsets[c]] to nodes[offsets[c + 1] - 1]). */
struct CellNodes {
    std::vector<uint32_t> offsets{0};
    std::vector<uint32_t> nodes;

    uint32_t size() const { return static_cast<uint32_t>(offsets.size() - 1); }
    std::span<const uint32_t> operator[](uint32_t c) const { return {nodes.data() + offsets[c], offsets[c + 1] - offsets[c]}; }
    template <typename Range>
    void add(const Range & cell) {
        nodes.insert(nodes.end(), std::begin(cell), std::end(cell));
        offsets.push_back(static_cast<uint32_t>(nodes.size()));
    }
};

/** @brief How a rank numbers its cells (see docs/design/mpi.md, section 6). */
enum class CellOrder {
    INPUT,    // as given: global order (owned cells, then each halo layer)
    HILBERT,  // along the Hilbert curve of the vertex averages
    AUTO      // whichever of the two touches fewer cache lines (see locality_order)
};

static const std::unordered_map<std::string, CellOrder> CELL_ORDERS = {
    {"auto", CellOrder::AUTO}, {"input", CellOrder::INPUT}, {"hilbert", CellOrder::HILBERT}};

/** @brief A cell order and, for auto, the vertex layers around a cell that its reconstruction reads. */
struct CellNumbering {
    CellOrder order = CellOrder::INPUT;
    uint32_t layers = 1;
};

/** @brief [mesh] cell_order (default auto), with the layers of [numerics.face_reconstruction]. */
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
};

/**
 * @brief Permutation of the cells that keeps every group [group_ends[k - 1],
 *        group_ends[k]) in place and numbers each by locality; group_ends ends
 *        with cells.size(). Ties keep the given order. Auto counts, for both
 *        orders, the 128-byte lines of a cell field that blocks of 256
 *        consecutive owned cells (group 0) read when each reads the cells
 *        within numbering.layers vertex layers, and takes the Hilbert order
 *        only if it needs a tenth fewer.
 */
LocalityOrder locality_order(const CellNumbering & numbering, const std::vector<std::array<rtype, N_DIM>> & nodes,
                             const CellNodes & cells, const std::vector<uint32_t> & group_ends);

#endif // CELL_ORDER_H

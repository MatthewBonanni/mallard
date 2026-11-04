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
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include "common_typedef.h"
#include "input.h"

/** @brief How a rank numbers its cells (see docs/design/mpi.md, section 6). */
enum class CellOrder {
    INPUT,   // as given: global order (owned cells, then each halo layer)
    HILBERT  // along the Hilbert curve of the vertex averages
};

// "auto" (nullopt) is hilbert for a mesh file and input for a generated mesh
static const std::unordered_map<std::string, std::optional<CellOrder>> CELL_ORDERS = {
    {"auto", std::nullopt}, {"input", CellOrder::INPUT}, {"hilbert", CellOrder::HILBERT}};

/** @brief [mesh] cell_order (default auto). */
CellOrder cell_order_of_input(const Input & input);

/**
 * @brief Permutation (new -> old) of the cells that keeps every group
 *        [group_ends[k - 1], group_ends[k]) in place and numbers each by
 *        locality; group_ends ends with cells.size(). Ties keep the given
 *        order.
 */
std::vector<uint32_t> locality_order(CellOrder order, const std::vector<std::array<rtype, N_DIM>> & nodes,
                                     const std::vector<std::vector<uint32_t>> & cells,
                                     const std::vector<uint32_t> & group_ends);

#endif // CELL_ORDER_H

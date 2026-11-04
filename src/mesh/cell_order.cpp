/**
 * @file cell_order.cpp
 * @author Matthew Bonanni (mbonanni001@gmail.com)
 * @brief Local cell numbering for memory locality.
 * @version 0.1
 * @date 2026-10-15
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 *
 */

#include "cell_order.h"

#include <algorithm>
#include <limits>

#include "input.h"
#include "partition.h"

CellOrder cell_order_of_input(const Input & input) {
    const std::string name = find_value_or<std::string>(input, "mesh", "cell_order", "auto");
    const std::optional<CellOrder> order =
        find_option(CELL_ORDERS, "mesh.cell_order", name,
                    input.contains("mesh") ? line_of(input.at("mesh"), "cell_order") : size_t(0))
            ->second;
    if (order) return *order;
    // Generated meshes are numbered row by row, which is already local
    return find_value_or<std::string>(input, "mesh", "type", "file") == "file" ? CellOrder::HILBERT : CellOrder::INPUT;
}

std::vector<uint32_t> locality_order(CellOrder kind, const std::vector<std::array<rtype, N_DIM>> & nodes,
                                     const std::vector<std::vector<uint32_t>> & cells,
                                     const std::vector<uint32_t> & group_ends) {
    std::vector<uint32_t> order;
    order.reserve(cells.size());
    std::vector<std::array<double, N_DIM>> centers;
    std::vector<std::pair<uint64_t, uint32_t>> keyed;
    uint32_t begin = 0;
    for (uint32_t end : group_ends) {
        if (kind == CellOrder::INPUT) {
            for (uint32_t c = begin; c < end; c++) order.push_back(c);
            begin = end;
            continue;
        }
        // Vertex averages along the Hilbert curve of the group's bounding box
        centers.assign(end - begin, {});
        std::array<double, N_DIM> lo, hi;
        lo.fill(std::numeric_limits<double>::max());
        hi.fill(std::numeric_limits<double>::lowest());
        for (uint32_t c = begin; c < end; c++) {
            auto & x = centers[c - begin];
            for (uint32_t n : cells[c]) {
                for (size_t d = 0; d < N_DIM; d++) x[d] += double(nodes[n][d]);
            }
            for (size_t d = 0; d < N_DIM; d++) {
                x[d] /= double(cells[c].size());
                lo[d] = std::min(lo[d], x[d]);
                hi[d] = std::max(hi[d], x[d]);
            }
        }
        keyed.resize(end - begin);
        for (uint32_t c = begin; c < end; c++) keyed[c - begin] = {hilbert_key(centers[c - begin], lo, hi), c};
        std::sort(keyed.begin(), keyed.end());
        for (const auto & [key, c] : keyed) order.push_back(c);
        begin = end;
    }
    return order;
}

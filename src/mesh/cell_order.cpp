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
#include <numeric>

#include "input.h"
#include "partition.h"

CellNumbering cell_numbering_of_input(const Input & input) {
    const std::string name = find_value_or<std::string>(input, "mesh", "cell_order", "auto");
    CellNumbering numbering;
    numbering.order = find_option(CELL_ORDERS, "mesh.cell_order", name,
                                  input.contains("mesh") ? line_of(input.at("mesh"), "cell_order") : size_t(0))
                          ->second;
    if (numbering.order == CellOrder::AUTO) {
        // TENO of order p reads about (p - 1) / 2 layers around a cell; MUSCL's gradients and fluxes, one
        const Input reconstruction =
            toml::find_or(input, "numerics", "face_reconstruction", Input(Input::table_type{}));
        if (find_value_or<std::string>(reconstruction, "type", "FO") == "TENO") {
            numbering.layers = static_cast<uint32_t>(std::max(1, (find_value_or<int>(reconstruction, "order", 5) - 1) / 2));
        }
    }
    return numbering;
}

namespace {

/** Cells around each node (CSR). */
struct NodeCells {
    std::vector<uint32_t> offsets, cells;

    NodeCells(const CellNodes & cell_nodes, size_t n_nodes) : offsets(n_nodes + 1, 0) {
        for (uint32_t v : cell_nodes.nodes) offsets[v + 1]++;
        std::partial_sum(offsets.begin(), offsets.end(), offsets.begin());
        cells.resize(offsets.back());
        std::vector<uint32_t> fill(offsets.begin(), offsets.end() - 1);
        for (uint32_t c = 0; c < cell_nodes.size(); c++) {
            for (uint32_t v : cell_nodes[c]) cells[fill[v]++] = c;
        }
    }
};

/**
 * Lines of 128 bytes of a cell field that every 8th block of 256 consecutive
 * cells in [0, n) reads, each cell reading those within `layers` vertex layers,
 * per cell of those blocks; order is new -> old, position old -> new.
 */
double lines_per_cell(const std::vector<uint32_t> & order, const std::vector<uint32_t> & position,
                      const CellNodes & cells, const NodeCells & around, uint32_t n,
                      uint32_t layers) {
    constexpr uint32_t block = 256, stride = 8;
    constexpr uint32_t line = static_cast<uint32_t>(128 / sizeof(rtype));
    std::vector<uint32_t> stamp(cells.size(), 0), frontier, next, lines;
    uint32_t visit = 0;
    uint64_t total = 0, sampled = 0;
    for (uint32_t b = 0; b < n; b += stride * block) {
        lines.clear();
        const uint32_t e = std::min(n, b + block);
        for (uint32_t i = b; i < e; i++) {
            frontier.assign(1, order[i]);
            stamp[order[i]] = ++visit;
            lines.push_back(i / line);
            for (uint32_t l = 0; l < layers; l++) {
                next.clear();
                for (uint32_t c : frontier) {
                    for (uint32_t v : cells[c]) {
                        for (uint32_t k = around.offsets[v]; k < around.offsets[v + 1]; k++) {
                            const uint32_t d = around.cells[k];
                            if (stamp[d] == visit) continue;
                            stamp[d] = visit;
                            next.push_back(d);
                            lines.push_back(position[d] / line);
                        }
                    }
                }
                frontier.swap(next);
            }
        }
        std::sort(lines.begin(), lines.end());
        total += static_cast<uint64_t>(std::unique(lines.begin(), lines.end()) - lines.begin());
        sampled += e - b;
    }
    return sampled ? double(total) / double(sampled) : 0.0;
}

} // namespace

LocalityOrder locality_order(const CellNumbering & numbering, const std::vector<std::array<rtype, N_DIM>> & nodes,
                             const CellNodes & cells, const std::vector<uint32_t> & group_ends) {
    LocalityOrder result;
    std::vector<uint32_t> & order = result.order;
    order.resize(cells.size());
    std::iota(order.begin(), order.end(), 0u);
    if (numbering.order == CellOrder::INPUT) return result;

    std::vector<std::array<double, N_DIM>> centers;
    std::vector<std::pair<uint64_t, uint32_t>> keyed;
    uint32_t begin = 0;
    for (uint32_t end : group_ends) {
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
        for (uint32_t i = begin; i < end; i++) order[i] = keyed[i - begin].second;
        begin = end;
    }
    result.choice.order = CellOrder::HILBERT;
    if (numbering.order == CellOrder::HILBERT) return result;

    const NodeCells around(cells, nodes.size());
    std::vector<uint32_t> identity(cells.size()), position(cells.size());
    std::iota(identity.begin(), identity.end(), 0u);
    for (uint32_t i = 0; i < order.size(); i++) position[order[i]] = i;
    const uint32_t n_owned = group_ends.empty() ? 0 : group_ends[0];
    result.choice.input_lines = lines_per_cell(identity, identity, cells, around, n_owned, numbering.layers);
    result.choice.hilbert_lines = lines_per_cell(order, position, cells, around, n_owned, numbering.layers);
    if (!(result.choice.hilbert_lines < 0.9 * result.choice.input_lines)) {
        order = std::move(identity);
        result.choice.order = CellOrder::INPUT;
    }
    return result;
}

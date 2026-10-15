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
#include <cmath>
#include <cstdlib>
#include <limits>
#include <numeric>
#include <random>

#include "input.h"
#include "partition.h"

CellOrder cell_order_of_input(const toml::value & input) {
    const std::string name = toml::find_or<std::string>(input, "mesh", "cell_order", "hilbert");
    const auto it = CELL_ORDERS.find(name);
    if (it == CELL_ORDERS.end()) throw unknown_option(CELL_ORDERS, "mesh.cell_order", name);
    return it->second;
}

namespace {

void hilbert_order(const std::vector<std::array<rtype, N_DIM>> & nodes, const std::vector<std::vector<uint32_t>> & cells,
                   uint32_t begin, uint32_t end, std::vector<uint32_t> & order) {
    std::vector<std::array<double, N_DIM>> centers(end - begin);
    std::array<double, N_DIM> lo, hi;
    lo.fill(std::numeric_limits<double>::max());
    hi.fill(std::numeric_limits<double>::lowest());
    for (uint32_t c = begin; c < end; c++) {
        auto & x = centers[c - begin];
        x.fill(0.0);
        for (uint32_t n : cells[c]) {
            for (size_t d = 0; d < N_DIM; d++) x[d] += double(nodes[n][d]);
        }
        for (size_t d = 0; d < N_DIM; d++) {
            x[d] /= double(cells[c].size());
            lo[d] = std::min(lo[d], x[d]);
            hi[d] = std::max(hi[d], x[d]);
        }
    }
    std::vector<std::pair<uint64_t, uint32_t>> keyed(end - begin);
    // EXPERIMENT: tiles of about MALLARD_TILE cells along the curve, input order within each
    int shift = 0;
    if (const char * tile = std::getenv("MALLARD_TILE")) {
        const double tiles = std::max(1.0, double(end - begin) / std::atof(tile));
        const int levels = static_cast<int>(std::ceil(std::log2(tiles) / N_DIM));
        shift = std::max(0, (63 / N_DIM) * N_DIM - levels * N_DIM);
    }
    for (uint32_t c = begin; c < end; c++) keyed[c - begin] = {hilbert_key(centers[c - begin], lo, hi) >> shift, c};
    std::sort(keyed.begin(), keyed.end());
    for (const auto & [key, c] : keyed) order.push_back(c);
}

void rcm_order(const std::vector<std::vector<uint32_t>> & cells, uint32_t begin, uint32_t end,
               std::vector<uint32_t> & order) {
    // Cells of the group sharing a node (CSR, local indices)
    const uint32_t n = end - begin;
    std::unordered_map<uint32_t, std::vector<uint32_t>> cells_of_node;
    for (uint32_t c = begin; c < end; c++) {
        for (uint32_t node : cells[c]) cells_of_node[node].push_back(c - begin);
    }
    std::vector<uint32_t> offsets(n + 1, 0), adjacent;
    std::vector<uint32_t> nb;
    for (uint32_t i = 0; i < n; i++) {
        nb.clear();
        for (uint32_t node : cells[begin + i]) {
            for (uint32_t j : cells_of_node[node]) {
                if (j != i) nb.push_back(j);
            }
        }
        std::sort(nb.begin(), nb.end());
        nb.erase(std::unique(nb.begin(), nb.end()), nb.end());
        adjacent.insert(adjacent.end(), nb.begin(), nb.end());
        offsets[i + 1] = static_cast<uint32_t>(adjacent.size());
    }
    auto degree = [&](uint32_t i) { return offsets[i + 1] - offsets[i]; };
    std::vector<uint32_t> level(n, std::numeric_limits<uint32_t>::max());
    // Breadth-first search from root over unvisited cells, neighbors by increasing degree; returns the last cell
    std::vector<uint32_t> queue;
    auto bfs = [&](uint32_t root, std::vector<uint32_t> & visit, uint32_t mark) {
        const size_t start = visit.size();
        visit.push_back(root);
        level[root] = mark;
        for (size_t q = start; q < visit.size(); q++) {
            const uint32_t i = visit[q];
            const size_t first = visit.size();
            for (uint32_t k = offsets[i]; k < offsets[i + 1]; k++) {
                const uint32_t j = adjacent[k];
                if (level[j] == mark) continue;
                level[j] = mark;
                visit.push_back(j);
            }
            std::stable_sort(visit.begin() + static_cast<std::ptrdiff_t>(first), visit.end(),
                             [&](uint32_t a, uint32_t b) { return degree(a) < degree(b); });
        }
    };
    std::vector<uint32_t> visited(n, 0), component;
    std::vector<uint32_t> result;
    uint32_t mark = 0;
    for (uint32_t seed = 0; seed < n; seed++) {
        if (visited[seed]) continue;
        // Pseudo-peripheral root: the last cell reached from the seed, twice
        uint32_t root = seed;
        for (int pass = 0; pass < 2; pass++) {
            component.clear();
            bfs(root, component, ++mark);
            root = component.back();
        }
        component.clear();
        bfs(root, component, ++mark);
        for (uint32_t i : component) visited[i] = 1;
        result.insert(result.end(), component.rbegin(), component.rend());
    }
    for (uint32_t i : result) order.push_back(begin + i);
}

} // namespace

std::vector<uint32_t> locality_order(CellOrder kind, const std::vector<std::array<rtype, N_DIM>> & nodes,
                                     const std::vector<std::vector<uint32_t>> & cells,
                                     const std::vector<uint32_t> & group_ends) {
    std::vector<uint32_t> order;
    order.reserve(cells.size());
    uint32_t begin = 0;
    for (uint32_t end : group_ends) {
        if (kind == CellOrder::HILBERT) {
            hilbert_order(nodes, cells, begin, end, order);
        } else if (kind == CellOrder::RANDOM || kind == CellOrder::ZFIRST) {
            std::vector<uint32_t> group(end - begin);
            std::iota(group.begin(), group.end(), begin);
            if (kind == CellOrder::RANDOM) {
                std::mt19937 rng(12345);
                std::shuffle(group.begin(), group.end(), rng);
            } else {
                auto center = [&](uint32_t c) {
                    std::array<double, N_DIM> x{};
                    for (uint32_t n : cells[c]) for (size_t d = 0; d < N_DIM; d++) x[d] += double(nodes[n][d]);
                    for (size_t d = 0; d < N_DIM; d++) x[d] /= double(cells[c].size());
                    return x;
                };
                std::vector<std::array<double, N_DIM>> xs(end - begin);
                for (uint32_t c = begin; c < end; c++) xs[c - begin] = center(c);
                std::stable_sort(group.begin(), group.end(), [&](uint32_t a, uint32_t b) {
                    return xs[a - begin] < xs[b - begin];
                });
            }
            order.insert(order.end(), group.begin(), group.end());
        } else if (kind == CellOrder::RCM) {
            rcm_order(cells, begin, end, order);
        } else {
            for (uint32_t c = begin; c < end; c++) order.push_back(c);
        }
        begin = end;
    }
    return order;
}

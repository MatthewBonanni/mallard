/**
 * @file stencil_shapes.cpp
 * @author Matthew Bonanni (mbonanni001@gmail.com)
 * @brief Stencil shapes of the cells of structured blocks.
 * @version 0.1
 * @date 2026-11-13
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 *
 */

#include "stencil_shapes.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <unordered_map>

#include <Kokkos_Core.hpp>
#include <utility>

namespace {

constexpr double QUANTUM = 1.0e-9;

int64_t quantize(double x) { return static_cast<int64_t>(std::llround(x / QUANTUM)); }

/** Two independent 64-bit hashes of the rounded geometry (a collision in both is not a concern). */
struct Key {
    uint64_t a = 0x9e3779b97f4a7c15ull, b = 0xc2b2ae3d27d4eb4full;
    void add(int64_t x) {
        const uint64_t v = static_cast<uint64_t>(x);
        a = mix(a ^ v);
        b = mix(b + v * 0xff51afd7ed558ccdull);
    }
    static uint64_t mix(uint64_t z) {
        z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ull;
        z = (z ^ (z >> 27)) * 0x94d049bb133111ebull;
        return z ^ (z >> 31);
    }
    bool operator==(const Key &) const = default;
};

struct KeyHash {
    size_t operator()(const Key & k) const { return static_cast<size_t>(k.a); }
};

} // namespace

StencilShapes stencil_shapes(const Mesh & mesh, const uint32_t reach) {
    using Host = Kokkos::DefaultHostExecutionSpace;
    StencilShapes out;
    const std::vector<StructuredBlock> & blocks = mesh.structured_blocks;
    const uint32_t n = structured_block_cells(blocks);
    out.shape.assign(n, StencilShapes::NONE);
    if (blocks.empty()) return out;
    // Each cell's vertex average, h, and its nodes relative to the average in units of h
    std::vector<std::array<double, N_DIM>> center(n);
    std::vector<double> h(n);
    std::vector<Key> own(n);
    Kokkos::parallel_for("stencil_shapes_cells", Kokkos::RangePolicy<Host>(0, n), [&](const uint32_t c) {
        h[c] = std::pow(double(mesh.h_cell_volume(c)), 1.0 / N_DIM);
        const uint32_t n_nodes = mesh.h_n_nodes_of_cell(c);
        std::array<double, N_DIM> x{};
        for (uint32_t v = 0; v < n_nodes; v++) {
            FOR_I_DIM x[i] += double(mesh.h_node_coords(mesh.h_node_of_cell(c, v), i));
        }
        FOR_I_DIM x[i] /= double(n_nodes);
        center[c] = x;
        std::array<std::array<int64_t, N_DIM>, 1u << N_DIM> nodes;
        for (uint32_t v = 0; v < n_nodes; v++) {
            FOR_I_DIM nodes[v][i] = quantize((double(mesh.h_node_coords(mesh.h_node_of_cell(c, v), i)) - x[i]) / h[c]);
        }
        std::sort(nodes.begin(), nodes.begin() + n_nodes);
        for (uint32_t v = 0; v < n_nodes; v++) {
            for (const int64_t q : nodes[v]) own[c].add(q);
        }
    });
    // Windows of 2 reach + 1 cells along each axis in turn: the cells' own nodes, then the windows of the previous
    // axis with the offset and h of each from the one before, in units of its h. Equal keys give the box of cells
    // around two cells one geometry relative to them, to rounding.
    std::vector<Key> key(n);
    std::vector<uint8_t> has(n, 0);
    const int64_t NO_CELL = std::numeric_limits<int64_t>::min();
    for (const StructuredBlock & block : blocks) {
        const std::array<uint32_t, 3> e = {block.n[0], block.n[1], N_DIM == 3 ? block.n[N_DIM - 1] : 1};
        const std::array<uint32_t, 3> stride = {1, e[0], e[0] * e[1]};
        const uint32_t size = e[0] * e[1] * e[2];
        auto index = [&](uint32_t l) { return std::array<uint32_t, 3>{l % e[0], (l / e[0]) % e[1], l / (e[0] * e[1])}; };
        std::vector<Key> window(own.begin() + block.first, own.begin() + block.first + size), next(size);
        // Windows of 2 reach + 1 descriptors along each axis in turn
        for (uint32_t a = 0; a < N_DIM; a++) {
            Kokkos::parallel_for("stencil_shapes_windows", Kokkos::RangePolicy<Host>(0, size), [&](const uint32_t l) {
                const auto x = index(l);
                Key k;
                if (x[a] < reach || x[a] + reach >= e[a]) {
                    k.add(NO_CELL);
                } else {
                    for (uint32_t m = l - reach * stride[a]; m <= l + reach * stride[a]; m += stride[a]) {
                        k.add(static_cast<int64_t>(window[m].a));
                        k.add(static_cast<int64_t>(window[m].b));
                        if (m == l + reach * stride[a]) break;
                        const uint32_t c = block.first + m, d = c + stride[a];
                        FOR_I_DIM k.add(quantize((center[d][i] - center[c][i]) / h[c]));
                        k.add(quantize(h[d] / h[c]));
                    }
                }
                next[l] = k;
            });
            window.swap(next);
        }
        for (uint32_t l = 0; l < size; l++) {
            const auto x = index(l);
            bool inside = true;
            for (uint32_t a = 0; a < N_DIM; a++) inside = inside && x[a] >= reach && x[a] + reach < e[a];
            if (!inside) continue;
            const uint32_t c = block.first + l;
            key[c] = window[l];
            key[c].add(quantize(std::log(h[c])));
            has[c] = 1;
        }
    }
    std::unordered_map<Key, uint32_t, KeyHash> shapes;
    for (uint32_t c = 0; c < n; c++) {
        if (!has[c]) continue;
        const auto [it, fresh] = shapes.emplace(key[c], static_cast<uint32_t>(out.first.size()));
        if (fresh) out.first.push_back(c);
        out.shape[c] = it->second;
    }
    return out;
}

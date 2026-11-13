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
#include <string>
#include <vector>
#include <unordered_map>

#include <Kokkos_Core.hpp>

#include "zone.h"
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

/** Cells reached from a cell, with the periodic translation they are seen across (open addressing). */
struct Reached {
    std::vector<uint32_t> cells;
    std::vector<std::array<double, N_DIM>> shift;
    std::vector<uint32_t> slots = std::vector<uint32_t>(64, UINT32_MAX);

    /** Index of cell c, inserted with shift t if new. */
    size_t insert(uint32_t c, const std::array<double, N_DIM> & t) {
        if (2 * (cells.size() + 1) > slots.size()) {
            slots.assign(2 * slots.size(), UINT32_MAX);
            for (uint32_t q = 0; q < cells.size(); q++) slots[probe(cells[q])] = q;
        }
        const size_t s = probe(c);
        if (slots[s] == UINT32_MAX) {
            slots[s] = static_cast<uint32_t>(cells.size());
            cells.push_back(c);
            shift.push_back(t);
        }
        return slots[s];
    }

    size_t probe(uint32_t c) const {
        const size_t mask = slots.size() - 1;
        size_t s = (uint64_t(c) * 0x9e3779b97f4a7c15ull >> 32) & mask;
        while (slots[s] != UINT32_MAX && cells[slots[s]] != c) s = (s + 1) & mask;
        return s;
    }
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
    const uint32_t n_all = mesh.n_cells;
    std::vector<std::array<double, N_DIM>> center(n_all);
    std::vector<double> h(n_all);
    std::vector<Key> own(n_all);
    Kokkos::parallel_for("stencil_shapes_cells", Kokkos::RangePolicy<Host>(0, n_all), [&](const uint32_t c) {
        h[c] = std::pow(double(mesh.h_cell_volume(c)), 1.0 / N_DIM);
        const uint32_t n_nodes = mesh.h_n_nodes_of_cell(c);
        std::array<double, N_DIM> x{};
        for (uint32_t v = 0; v < n_nodes; v++) {
            FOR_I_DIM x[i] += double(mesh.h_node_coords(mesh.h_node_of_cell(c, v), i));
        }
        FOR_I_DIM x[i] /= double(n_nodes);
        center[c] = x;
        std::vector<std::array<int64_t, N_DIM>> nodes(n_nodes);
        for (uint32_t v = 0; v < n_nodes; v++) {
            FOR_I_DIM nodes[v][i] = quantize((double(mesh.h_node_coords(mesh.h_node_of_cell(c, v), i)) - x[i]) / h[c]);
        }
        std::sort(nodes.begin(), nodes.end());
        for (const auto & node : nodes) {
            for (const int64_t q : node) own[c].add(q);
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
            // The cells within reach have all their neighbors in the block, so none has a boundary face
            for (uint32_t a = 0; a < N_DIM; a++) inside = inside && x[a] > reach && x[a] + reach + 1 < e[a];
            if (!inside) continue;
            const uint32_t c = block.first + l;
            key[c] = window[l];
            key[c].add(quantize(std::log(h[c])));
            has[c] = 1;
        }
    }
    // The other block cells: every cell within reach (vertex layers) and every boundary face of those cells, by
    // their geometry relative to the cell and their zone, in sorted order
    std::vector<uint64_t> face_zone(mesh.n_faces, 0);
    for (const FaceZone & zone : mesh.h_face_zones()) {
        if (zone.get_type() != FaceZoneType::BOUNDARY) continue;
        const uint64_t name = std::hash<std::string>{}(zone.get_name()) | 1;
        for (size_t k = 0; k < zone.h_faces.extent(0); k++) face_zone[zone.h_faces(k)] = name;
    }
    const bool shifted = mesh.h_cells_of_cell_shift.extent(0) > 0;
    Kokkos::parallel_for("stencil_shapes_near_faces", Kokkos::RangePolicy<Host>(0, n), [&](const uint32_t c) {
        if (has[c]) return;
        Reached at;
        at.insert(c, {});
        size_t begin = 0;
        for (uint32_t layer = 0; layer < reach; layer++) {
            const size_t end = at.cells.size();
            for (size_t q = begin; q < end; q++) {
                const uint32_t d = at.cells[q];
                if (d >= mesh.n_complete()) return;  // its vertex neighbors are not all on this rank
                const std::array<double, N_DIM> t_d = at.shift[q];
                for (uint32_t k = mesh.h_offsets_cells_of_cell(d); k < mesh.h_offsets_cells_of_cell(d + 1); k++) {
                    const uint8_t row = shifted ? mesh.h_cells_of_cell_shift(k) : 0;
                    if (mesh.h_rotated_row(row)) return;
                    std::array<double, N_DIM> t = t_d;
                    if (row != 0) FOR_I_DIM t[i] += double(mesh.h_shifts(row, i));
                    const size_t found = at.insert(mesh.h_cells_of_cell(k), t);
                    // A cell reached across two different periodic shifts: no frame
                    FOR_I_DIM if (std::abs(at.shift[found][i] - t[i]) > QUANTUM * h[c]) return;
                }
            }
            begin = end;
        }
        // Order-independent: a sum of the parts' mixed hashes in each lane
        Key k;
        k.add(NO_CELL);  // apart from the keys of the cells keyed through their block
        k.add(quantize(std::log(h[c])));
        uint64_t sum_a = 0, sum_b = 0;
        auto part = [&](const Key & p) {
            sum_a += Key::mix(p.a);
            sum_b += Key::mix(p.b ^ 0x5851f42d4c957f2dull);
        };
        std::vector<std::array<int64_t, N_DIM>> nodes;
        for (size_t q = 0; q < at.cells.size(); q++) {
            const uint32_t d = at.cells[q];
            const std::array<double, N_DIM> & t = at.shift[q];
            Key p;
            FOR_I_DIM p.add(quantize((center[d][i] + t[i] - center[c][i]) / h[c]));
            p.add(quantize(h[d] / h[c]));
            p.add(static_cast<int64_t>(own[d].a));
            p.add(static_cast<int64_t>(own[d].b));
            part(p);
            for (uint32_t j = 0; j < mesh.h_n_faces_of_cell(d); j++) {
                const uint32_t f = mesh.h_face_of_cell(d, j);
                if (face_zone[f] == 0) continue;
                nodes.resize(mesh.h_n_nodes_of_face(f));
                for (uint32_t v = 0; v < nodes.size(); v++) {
                    FOR_I_DIM nodes[v][i] =
                        quantize((double(mesh.h_node_coords(mesh.h_node_of_face(f, v), i)) + t[i] - center[c][i]) / h[c]);
                }
                std::sort(nodes.begin(), nodes.end());
                Key b;
                b.add(static_cast<int64_t>(face_zone[f]));
                for (const auto & node : nodes) {
                    for (const int64_t x : node) b.add(x);
                }
                part(b);
            }
        }
        k.add(static_cast<int64_t>(sum_a));
        k.add(static_cast<int64_t>(sum_b));
        key[c] = k;
        has[c] = 1;
    });
    std::unordered_map<Key, uint32_t, KeyHash> shapes;
    for (uint32_t c = 0; c < n; c++) {
        if (!has[c]) continue;
        const auto [it, fresh] = shapes.emplace(key[c], static_cast<uint32_t>(out.first.size()));
        if (fresh) out.first.push_back(c);
        out.shape[c] = it->second;
    }
    return out;
}

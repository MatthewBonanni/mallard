/**
 * @file teno_shapes.cpp
 * @author Matthew Bonanni (mbonanni001@gmail.com)
 * @brief Stencil-shape keys of TENO cells.
 * @version 0.1
 * @date 2026-11-14
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 *
 */

#include "teno_shapes.h"

#include <algorithm>
#include <bit>
#include <unordered_map>

#include <Kokkos_BitManipulation.hpp>

#include "comm.h"
#include "gated.h"

namespace teno {

namespace {

constexpr bool DEVICE_IS_HOST = Kokkos::SpaceAccessibility<Kokkos::DefaultExecutionSpace, Kokkos::HostSpace>::accessible;
constexpr uint64_t EMPTY = ~uint64_t(0);
constexpr uint32_t MAX_END = 66;     // the searches' layers: at most 64 central ones, and the cell itself
constexpr int MAX_SHAPE_ITEMS = 40;  // nodes and boundary faces of a cell with a shape

KOKKOS_INLINE_FUNCTION uint64_t mix(uint64_t h) {
    h = (h ^ (h >> 30)) * 0xbf58476d1ce4e5b9ULL;
    h = (h ^ (h >> 27)) * 0x94d049bb133111ebULL;
    return h ^ (h >> 31);
}

/** @brief Two independent hashes of a sequence of words. */
struct Hasher {
    uint64_t a = 0x243f6a8885a308d3ULL, b = 0x13198a2e03707344ULL;
    KOKKOS_INLINE_FUNCTION void add(const uint64_t v) {
        a = mix(a ^ (v + 0x9e3779b97f4a7c15ULL));
        b = mix(b + (v ^ 0xd1b54a32d192ed03ULL) * 0xff51afd7ed558ccdULL);
    }
    KOKKOS_INLINE_FUNCTION void add_signed(const int64_t v) { add(static_cast<uint64_t>(v)); }
};

KOKKOS_INLINE_FUNCTION int lattice_component(const uint32_t lattice, const int a) {
    return int((lattice >> (10 * a)) & 1023u) - 512;
}

constexpr uint32_t LATTICE_ZERO = 512u | (512u << 10) | (512u << 20);

KOKKOS_INLINE_FUNCTION uint64_t hash_slot(const uint32_t c, const uint32_t mask) {
    return ((uint64_t(c) * 0x9E3779B97F4A7C15ULL) >> 32) & mask;
}

template <int N>
KOKKOS_INLINE_FUNCTION bool less(const int64_t * a, const int64_t * b) {
    for (int d = 0; d < N; d++) {
        if (a[d] != b[d]) return a[d] < b[d];
    }
    return false;
}

template <int N>
KOKKOS_INLINE_FUNCTION bool equal(const int64_t * a, const int64_t * b) {
    for (int d = 0; d < N; d++) {
        if (a[d] != b[d]) return false;
    }
    return true;
}

} // namespace

KOKKOS_FUNCTION int64_t ShapeSearch::on_grid(const double v) {
    constexpr double LIMIT = 4.0e18;
    const double x = v * double(uint64_t(1) << ShapeKeys::SHAPE_GRID_BITS);
    return static_cast<int64_t>(Kokkos::round(Kokkos::fmin(Kokkos::fmax(x, -LIMIT), LIMIT)));
}

KOKKOS_FUNCTION int64_t ShapeSearch::position(const ShapeLanes & w, const uint32_t c) const {
    const uint32_t mask = 2 * cap - 1;
    for (uint64_t k = hash_slot(c, mask);; k = (k + 1) & mask) {
        const uint64_t e = w.hash[k];
        if (e == EMPTY) return -1;
        if (uint32_t(e) == c) return int64_t(e >> 32);
    }
}

KOKKOS_FUNCTION void ShapeSearch::clear(const ShapeLanes & w, const uint32_t n) const {
    // In reverse order of insertion, each entry's probe sequence is as it was when it was inserted
    const uint32_t mask = 2 * cap - 1;
    for (uint32_t v = n; v-- > 0;) {
        const uint32_t c = w.cell[v];
        for (uint64_t k = hash_slot(c, mask);; k = (k + 1) & mask) {
            if (uint32_t(w.hash[k]) == c) {
                w.hash[k] = EMPTY;
                break;
            }
        }
    }
}

KOKKOS_FUNCTION ShapeSearch::Result ShapeSearch::search(const uint32_t c, const ShapeLanes & w, uint32_t * end,
                                                         const uint32_t layer_limit, const bool ordered) const {
    Result r;
    const uint32_t mask = 2 * cap - 1;
    w.cell[0] = c;
    w.lattice[0] = LATTICE_ZERO;
    w.hash[hash_slot(c, mask)] = uint64_t(c);
    r.n = 1;
    uint32_t n_end = 1;
    end[0] = 1;
    bool unique = true, exhausted = false, full = false;
    // The search's vertex-neighbor layers (reach_layers in teno_setup.cpp), each cell once
    auto grow = [&]() {
        if (exhausted || full || n_end > layer_limit) return false;
        const uint32_t begin = n_end > 1 ? end[n_end - 2] : 0;
        const uint32_t last = end[n_end - 1];
        for (uint32_t k = begin; k < last; k++) {
            const uint32_t v = w.cell[k];
            const uint32_t l = w.lattice[k];
            for (uint32_t j = offsets_cells_of_cell(v); j < offsets_cells_of_cell(v + 1); j++) {
                const uint32_t nb = cells_of_cell(j);
                const uint8_t sh = cells_of_cell_shift(j);
                const uint32_t nl = uint32_t(lattice_component(l, 0) + shift_lattice(sh, 0) + 512) |
                                    (uint32_t(lattice_component(l, 1) + shift_lattice(sh, 1) + 512) << 10) |
                                    (uint32_t(lattice_component(l, 2) + shift_lattice(sh, 2) + 512) << 20);
                uint64_t slot = hash_slot(nb, mask);
                bool seen = false;
                for (;; slot = (slot + 1) & mask) {
                    const uint64_t e = w.hash[slot];
                    if (e == EMPTY) break;
                    if (uint32_t(e) == nb) {
                        seen = true;
                        if (w.lattice[e >> 32] != nl) unique = false;
                        break;
                    }
                }
                if (seen) continue;
                if (r.n == cap) {
                    full = true;
                    return false;
                }
                w.hash[slot] = uint64_t(nb) | (uint64_t(r.n) << 32);
                w.cell[r.n] = nb;
                w.lattice[r.n] = nl;
                r.n++;
            }
        }
        if (r.n == last) {
            exhausted = true;
            return false;
        }
        end[n_end++] = r.n;
        return true;
    };
    auto reach = [&](const uint32_t n_min, const uint32_t max_layers) {
        uint32_t depth = 0;
        while (depth < max_layers && end[depth] <= n_min) {
            if (depth + 1 == n_end && !grow()) break;
            depth++;
        }
    };
    reach(n_large, 64);
    reach(8 * n_small, 6);
    r.layers = n_end - 1;
    if (full || !unique) return r;
    double x0[N_DIM];
    for (int d = 0; d < N_DIM; d++) x0[d] = double(cell_coords(c, d));
    const double h0 = h(c);
    for (uint32_t k = 0; k < r.n; k++) {
        const uint32_t v = w.cell[k];
        if (v >= n_complete) return r;
        for (int d = 0; d < N_DIM; d++) {
            double x = double(cell_coords(v, d));
            for (uint32_t j = 0; j < n_translations; j++) {
                x = Kokkos::fma(double(lattice_component(w.lattice[k], int(j))), translations(j, d), x);
            }
            w.q[size_t(k) * N_DIM + size_t(d)] = on_grid((x - x0[d]) / h0);
        }
        w.order[k] = k;
    }
    if (!ordered) {
        r.ok = true;
        return r;
    }
    // Each layer by position: a heap sort, whose order is total once ties are excluded
    auto key = [&](const uint32_t k, int64_t * out) {
        for (int d = 0; d < N_DIM; d++) out[d] = w.q[size_t(k) * N_DIM + size_t(d)];
    };
    auto before = [&](const uint32_t a, const uint32_t b) {
        int64_t qa[N_DIM], qb[N_DIM];
        key(a, qa);
        key(b, qb);
        return less<N_DIM>(qa, qb);
    };
    for (uint32_t layer = 1; layer < n_end; layer++) {
        const uint32_t b0 = end[layer - 1], n = end[layer] - b0;
        auto sift = [&](uint32_t root, const uint32_t size) {
            const uint32_t item = w.order[b0 + root];
            while (true) {
                uint32_t child = 2 * root + 1;
                if (child >= size) break;
                if (child + 1 < size && before(w.order[b0 + child], w.order[b0 + child + 1])) child++;
                if (!before(item, w.order[b0 + child])) break;
                w.order[b0 + root] = w.order[b0 + child];
                root = child;
            }
            w.order[b0 + root] = item;
        };
        for (uint32_t k = n / 2; k-- > 0;) sift(k, n);
        for (uint32_t size = n; size > 1; size--) {
            const uint32_t top = w.order[b0];
            w.order[b0] = w.order[b0 + size - 1];
            w.order[b0 + size - 1] = top;
            sift(0, size - 1);
        }
        for (uint32_t k = b0 + 1; k < end[layer]; k++) {
            int64_t qa[N_DIM], qb[N_DIM];
            key(w.order[k - 1], qa);
            key(w.order[k], qb);
            if (equal<N_DIM>(qa, qb)) return r;
        }
    }
    for (uint32_t p = 0; p < r.n; p++) {
        const uint32_t v = w.cell[w.order[p]];
        for (uint64_t k = hash_slot(v, mask);; k = (k + 1) & mask) {
            if (uint32_t(w.hash[k]) == v) {
                w.hash[k] = uint64_t(v) | (uint64_t(p) << 32);
                break;
            }
        }
    }
    r.ok = true;
    return r;
}

KOKKOS_FUNCTION int ShapeSearch::boundary_rank(const uint32_t c, const uint32_t f) const {
    int64_t qf[N_DIM];
    for (int d = 0; d < N_DIM; d++) qf[d] = on_grid((double(face_coords(f, d)) - double(cell_coords(c, d))) / h(c));
    int rank = 0;
    bool found = false;
    for (uint32_t k = offsets_faces_of_cell(c); k < offsets_faces_of_cell(c + 1); k++) {
        const uint32_t g = faces_of_cell(k);
        if (cells_of_face(g, 1) >= 0) continue;
        if (g == f) {
            found = true;
            continue;
        }
        int64_t qg[N_DIM];
        for (int d = 0; d < N_DIM; d++) qg[d] = on_grid((double(face_coords(g, d)) - double(cell_coords(c, d))) / h(c));
        if (equal<N_DIM>(qg, qf)) return -1;
        if (less<N_DIM>(qg, qf)) rank++;
    }
    return found ? rank : -1;
}

KOKKOS_FUNCTION int64_t ShapeSearch::boundary_face(const uint32_t c, const int rank) const {
    for (uint32_t k = offsets_faces_of_cell(c); k < offsets_faces_of_cell(c + 1); k++) {
        const uint32_t g = faces_of_cell(k);
        if (cells_of_face(g, 1) < 0 && boundary_rank(c, g) == rank) return int64_t(g);
    }
    return -1;
}

namespace {

/** @brief Each cell's hash of its nodes and boundary faces with their conditions, sorted, in its own units. */
void compute_shapes(const ShapeSearch & g, const uint32_t n_cells) {
    gated::parallel_for("teno_shape_shapes", Kokkos::RangePolicy<>(0, n_cells), KOKKOS_LAMBDA(const uint32_t c) {
        int64_t items[MAX_SHAPE_ITEMS][N_DIM + 1];
        const uint32_t n0 = g.offsets_nodes_of_cell(c), n1 = g.offsets_nodes_of_cell(c + 1);
        const uint32_t f0 = g.offsets_faces_of_cell(c), f1 = g.offsets_faces_of_cell(c + 1);
        double xc[N_DIM];
        for (int d = 0; d < N_DIM; d++) xc[d] = double(g.cell_coords(c, d));
        const double hc = g.h(c);
        int n = 0;
        bool fits = true;
        for (uint32_t k = n0; k < n1 && fits; k++) {
            fits = n < MAX_SHAPE_ITEMS;
            if (!fits) break;
            for (int d = 0; d < N_DIM; d++) items[n][d] = ShapeSearch::on_grid((double(g.node_coords(g.nodes_of_cell(k), d)) - xc[d]) / hc);
            items[n++][N_DIM] = -1;
        }
        for (uint32_t k = f0; k < f1 && fits; k++) {
            const uint32_t f = g.faces_of_cell(k);
            if (g.cells_of_face(f, 1) >= 0) continue;
            fits = n < MAX_SHAPE_ITEMS;
            if (!fits) break;
            for (int d = 0; d < N_DIM; d++) items[n][d] = ShapeSearch::on_grid((double(g.face_coords(f, d)) - xc[d]) / hc);
            items[n++][N_DIM] = g.face_bc(f);
        }
        if (!fits) {
            g.shape(c) = 0;
            return;
        }
        for (int i = 1; i < n; i++) {
            for (int j = i; j > 0 && less<N_DIM + 1>(items[j], items[j - 1]); j--) {
                for (int d = 0; d <= N_DIM; d++) {
                    const int64_t t = items[j][d];
                    items[j][d] = items[j - 1][d];
                    items[j - 1][d] = t;
                }
            }
        }
        Hasher hs;
        hs.add(f1 - f0);
        for (int i = 0; i < n; i++) {
            for (int d = 0; d <= N_DIM; d++) hs.add_signed(items[i][d]);
        }
        g.shape(c) = hs.a == 0 ? 1 : hs.a;
    });
}

/** @brief f(t, scratch) for t < n, slots searches at a time. */
template <class F>
void each_search(const ShapeSearch & g, const uint32_t slots, const uint32_t n, const F & f) {
    if constexpr (DEVICE_IS_HOST) {
        Kokkos::Experimental::UniqueToken<Kokkos::DefaultExecutionSpace, Kokkos::Experimental::UniqueTokenScope::Global>
            token;
        gated::parallel_for("teno_shape_search",
                             Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace, Kokkos::Schedule<Kokkos::Dynamic>>(0, n),
                             [=](const uint32_t t) {
            const int id = token.acquire();
            f(t, g.lanes(size_t(id)));
            token.release(id);
        });
    } else {
        for (uint32_t t0 = 0; t0 < n; t0 += slots) {
            gated::parallel_for("teno_shape_search", Kokkos::RangePolicy<>(0, std::min(slots, n - t0)),
                                 KOKKOS_LAMBDA(const uint32_t i) { f(t0 + i, g.lanes(i)); });
        }
    }
}

} // namespace

bool ShapeKeys::supported(const Mesh & mesh) {
    return !mesh.curved_geometry && !mesh.is_rotational() && !mesh.axisymmetric;
}

ShapeKeys::ShapeKeys(const Mesh & mesh_in, const BoundaryData & boundaries, const uint32_t n_large,
                     const uint32_t n_small, const uint32_t batch)
    : mesh(mesh_in) {
    const uint32_t n_cells = mesh.n_cells;
    const VertexNeighbors & neighbors = mesh.vertex_neighbors();
    s.cell_coords = mesh.cell_coords;
    s.face_coords = mesh.face_coords;
    s.node_coords = mesh.node_coords;
    s.shifts = mesh.shifts;
    s.offsets_cells_of_cell = neighbors.offsets;
    s.cells_of_cell = neighbors.cells;
    s.offsets_faces_of_cell = mesh.offsets_faces_of_cell;
    s.faces_of_cell = mesh.faces_of_cell;
    s.offsets_nodes_of_cell = mesh.offsets_nodes_of_cell;
    s.nodes_of_cell = mesh.nodes_of_cell;
    s.cells_of_cell_shift = neighbors.shift;
    s.face_shift = mesh.face_shift;
    s.cells_of_face = mesh.cells_of_face;
    s.face_bc = boundaries.face_bc;
    s.n_complete = mesh.n_complete();
    s.n_large = n_large;
    s.n_small = n_small;
    const size_t n_shifts = std::max<size_t>(mesh.shift_lattice.size(), 1);
    s.shift_lattice = Kokkos::View<int8_t *[3]>("teno_shape_shift_lattice", n_shifts);
    auto h_lattice = Kokkos::create_mirror_view(s.shift_lattice);
    for (size_t k = 0; k < mesh.shift_lattice.size(); k++) {
        for (size_t a = 0; a < 3; a++) h_lattice(k, a) = mesh.shift_lattice[k][a];
    }
    Kokkos::deep_copy(s.shift_lattice, h_lattice);
    s.n_translations = static_cast<uint32_t>(mesh.periodic_translations.size());
    s.translations = Kokkos::View<double *[3]>("teno_shape_translations", std::max<uint32_t>(s.n_translations, 1));
    auto h_translations = Kokkos::create_mirror_view(s.translations);
    for (uint32_t j = 0; j < s.n_translations; j++) {
        for (int d = 0; d < N_DIM; d++) h_translations(j, d) = double(mesh.periodic_translations[j][size_t(d)]);
    }
    Kokkos::deep_copy(s.translations, h_translations);
    s.h = Kokkos::View<double *>(Kokkos::view_alloc(Kokkos::WithoutInitializing, "teno_shape_h"), n_cells);
    auto h_h = Kokkos::create_mirror_view(s.h);
    gated::parallel_for("teno_shape_scales", Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(0, n_cells),
                         [&](const uint32_t c) { h_h(c) = teno_setup::cell_scale(mesh, c); });
    Kokkos::deep_copy(s.h, h_h);

    // Room for about four times the cells the searches stop past
    s.cap = std::bit_ceil(std::clamp<uint32_t>(4 * std::max(n_large, 8 * n_small), 256, 4096));
    if constexpr (DEVICE_IS_HOST) {
        slots = static_cast<uint32_t>(Kokkos::Experimental::UniqueToken<Kokkos::DefaultExecutionSpace, Kokkos::Experimental::UniqueTokenScope::Global>().size());
    } else {
        const size_t per_slot = size_t(s.cap) * (3 * sizeof(uint32_t) + N_DIM * sizeof(int64_t) + 2 * sizeof(uint64_t));
        slots = static_cast<uint32_t>(std::clamp<size_t>((size_t(1) << 30) / per_slot, 1024, std::max(batch, 1024u)));
    }
    const bool interleaved = !DEVICE_IS_HOST;
    s.cell = teno_setup::SlotArray<uint32_t>("teno_shape_cells", s.cap, slots, interleaved);
    s.lattice = teno_setup::SlotArray<uint32_t>("teno_shape_lattice", s.cap, slots, interleaved);
    s.order = teno_setup::SlotArray<uint32_t>("teno_shape_order", s.cap, slots, interleaved);
    s.q = teno_setup::SlotArray<int64_t>("teno_shape_q", size_t(s.cap) * N_DIM, slots, interleaved);
    s.hash = teno_setup::SlotArray<uint64_t>("teno_shape_hash", 2 * size_t(s.cap), slots, interleaved);
    Kokkos::deep_copy(s.hash.data, EMPTY);

    s.shape = Kokkos::View<uint64_t *>(Kokkos::view_alloc(Kokkos::WithoutInitializing, "teno_shape_shape"), n_cells);
    compute_shapes(s, n_cells);
#if N_DIM == 2
    const std::vector<uint8_t> behind = teno_setup::boundary_behind(mesh, boundaries);
    s.behind = Kokkos::View<uint8_t *>("teno_shape_behind", n_cells);
    auto h_behind = Kokkos::create_mirror_view(s.behind);
    for (uint32_t f = 0; f < mesh.n_faces; f++) {
        if (behind[f]) h_behind(uint32_t(mesh.h_cells_of_face(f, 0))) = 1;
    }
    Kokkos::deep_copy(s.behind, h_behind);
#endif
}

std::vector<ShapeKey> ShapeKeys::keys(const uint32_t n) const {
    // A cell whose own shape no other owned cell has, over the ranks, is a shape of its own
    std::vector<uint64_t> shape(n);
    Kokkos::deep_copy(Kokkos::View<uint64_t *, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>>(shape.data(), n),
                      Kokkos::subview(s.shape, std::make_pair(0u, n)));
    std::unordered_map<uint64_t, uint64_t> count;
    const uint32_t n_owned = std::min(n, mesh.n_owned());
    for (uint32_t c = 0; c < n_owned; c++) count[shape[c]]++;
    for (uint32_t c = n_owned; c < n; c++) count.try_emplace(shape[c], 0);
    const int n_ranks = comm::size();
    if (n_ranks > 1) {
        std::vector<std::vector<uint64_t>> report(static_cast<size_t>(n_ranks));
        for (const auto & [d, k] : count) report[d % uint64_t(n_ranks)].insert(report[d % uint64_t(n_ranks)].end(), {d, k});
        const auto reports = comm::alltoallv(report);
        std::unordered_map<uint64_t, uint64_t> total;
        for (const auto & from : reports) {
            for (size_t k = 0; k < from.size(); k += 2) total[from[k]] += from[k + 1];
        }
        std::vector<std::vector<uint64_t>> answer(static_cast<size_t>(n_ranks));
        for (int r = 0; r < n_ranks; r++) {
            for (size_t k = 0; k < reports[size_t(r)].size(); k += 2) {
                answer[size_t(r)].insert(answer[size_t(r)].end(), {reports[size_t(r)][k], total.at(reports[size_t(r)][k])});
            }
        }
        count.clear();
        for (const auto & from : comm::alltoallv(answer)) {
            for (size_t k = 0; k < from.size(); k += 2) count[from[k]] = from[k + 1];
        }
    }
    Kokkos::View<uint8_t *> shared("teno_shape_shared", n);
    {
        auto h_shared = Kokkos::create_mirror_view(shared);
        for (uint32_t c = 0; c < n; c++) h_shared(c) = shape[c] != 0 && count.at(shape[c]) >= 2;
        Kokkos::deep_copy(shared, h_shared);
    }
    Kokkos::View<uint64_t *[2]> out("teno_shape_keys", n);
    const ShapeSearch g = s;
    each_search(g, slots, n, KOKKOS_LAMBDA(const uint32_t c, const ShapeLanes & w) {
        if (!shared(c)) return;
        uint32_t end[MAX_END];
        const ShapeSearch::Result r = g.search(c, w, end, ~0u, false);
        // Images that a boundary behind a mirror face drops make a fit of its own
        bool alone = false;
        for (uint32_t p = 0; p < r.n && r.ok && g.behind.extent(0) > 0 && !alone; p++) alone = g.behind(w.cell[p]);
        if (r.ok && !alone) {
            Hasher hs;
            hs.add(r.layers + 1);
            hs.add(r.n);
            // The cell's faces in its own order, in its frame
            double x0[N_DIM];
            for (int d = 0; d < N_DIM; d++) x0[d] = double(g.cell_coords(c, d));
            const double hc = g.h(c);
            for (uint32_t k = g.offsets_faces_of_cell(c); k < g.offsets_faces_of_cell(c + 1); k++) {
                const uint32_t f = g.faces_of_cell(k);
                const bool second = g.cells_of_face(f, 1) == int32_t(c);
                hs.add(second);
                for (int d = 0; d < N_DIM; d++) {
                    const double shift = second ? double(g.shifts(g.face_shift(f), d)) : 0.0;
                    hs.add_signed(ShapeSearch::on_grid((double(g.face_coords(f, d)) - shift - x0[d]) / hc));
                }
            }
            uint64_t sum_a = 0, sum_b = 0;
            uint32_t layer = 0;
            for (uint32_t k = 0; k < r.n; k++) {
                while (k >= end[layer]) layer++;
                const uint32_t v = w.cell[k];
                Hasher visit;
                visit.add(layer);
                for (int d = 0; d < N_DIM; d++) visit.add_signed(w.q[size_t(k) * N_DIM + size_t(d)]);
                visit.add(g.shape(v));
                visit.add_signed(ShapeSearch::on_grid(g.h(v) / hc));
                sum_a += visit.a;
                sum_b += visit.b;
            }
            hs.add(sum_a);
            hs.add(sum_b);
            out(c, 0) = hs.a == 0 ? 1 : hs.a;
            out(c, 1) = hs.b;
        }
        g.clear(w, r.n);
    });
    auto h_out = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), out);
    std::vector<ShapeKey> keys(n);
    for (uint32_t c = 0; c < n; c++) keys[c] = ShapeKey{h_out(c, 0), h_out(c, 1)};
    return keys;
}

void ShapeKeys::encode(const Kokkos::View<uint32_t *> & cells, const uint32_t n, const teno_setup::TableBatch & batch,
                       const Kokkos::View<uint32_t *[2]> & size) const {
    const ShapeSearch g = s;
    each_search(g, slots, n, KOKKOS_LAMBDA(const uint32_t j, const ShapeLanes & w) {
        uint32_t end[MAX_END];
        const uint32_t c = cells(j);
        const ShapeSearch::Result r = g.search(c, w, end);
        bool ok = r.ok && batch.status(j) == teno_setup::OK && batch.failed_large(j) == 0 &&
                  batch.gather_depth(j) == r.layers;
        int64_t last = 0;  // the last position the stencils reach
        auto stencil = [&](const teno_setup::Lane<int32_t> & entries, const teno_setup::Lane<int32_t> & faces,
                           const uint32_t count) {
            for (uint32_t k = 0; k < count && ok; k++) {
                const int64_t p = g.position(w, uint32_t(entries[k]));
                const int32_t f = faces[k];
                int64_t code = -1;
                if (f >= 0) {
                    const uint32_t owner = uint32_t(g.cells_of_face(uint32_t(f), 0));
                    const int64_t po = g.position(w, owner);
                    const int rank = po < 0 ? -1 : g.boundary_rank(owner, uint32_t(f));
                    ok = rank >= 0;
                    code = po * 64 + rank;
                    last = po > last ? po : last;
                }
                ok = ok && p >= 0 && f >= -1;
                last = p > last ? p : last;
                entries[k] = int32_t(p);
                faces[k] = int32_t(code);
            }
        };
        if (ok) stencil(batch.large_cells.lane(j), batch.large_faces.lane(j), batch.large_size(j));
        if (ok) stencil(batch.small_cells.lane(j), batch.small_faces.lane(j), batch.small_total(j));
        uint32_t layers = 0;
        while (end[layers] <= uint64_t(last)) layers++;
        size(j, 0) = ok ? end[layers] : 0;
        size(j, 1) = layers;
        g.clear(w, r.n);
    });
}

std::vector<std::vector<uint64_t>> ShapeKeys::records(const teno_setup::TableBatch & batch, const uint32_t n,
                                                      const Kokkos::View<uint32_t *[2]> & size) {
    const auto used = std::make_pair(0u, n);
    auto h_size = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), Kokkos::subview(size, used, Kokkos::ALL));
    auto h_large = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), Kokkos::subview(batch.large_size, used));
    auto h_small = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), Kokkos::subview(batch.small_total, used));
    const size_t nk = batch.nk;
    const size_t head = 6 + teno::MAX_FACES + batch.basis_mean.cap + batch.si.cap;
    Kokkos::View<uint64_t *> offset(Kokkos::view_alloc(Kokkos::WithoutInitializing, "teno_shape_record_offsets"), n + 1);
    auto h_offset = Kokkos::create_mirror_view(offset);
    h_offset(0) = 0;
    for (uint32_t j = 0; j < n; j++) {
        const size_t length = h_size(j, 0) == 0 ? 1 : head + h_large(j) * (2 + nk) + h_small(j) * (2 + size_t(teno::NK_SMALL));
        h_offset(j + 1) = h_offset(j) + length;
    }
    Kokkos::deep_copy(offset, h_offset);
    Kokkos::View<uint64_t *> data(Kokkos::view_alloc(Kokkos::WithoutInitializing, "teno_shape_record_data"), h_offset(n));
    gated::parallel_for("teno_shape_records", Kokkos::RangePolicy<>(0, n), KOKKOS_LAMBDA(const uint32_t j) {
        uint64_t * r = data.data() + offset(j);
        if (size(j, 0) == 0) {
            r[0] = 0;
            return;
        }
        const uint32_t n_large = batch.large_size(j), n_small = batch.small_total(j);
        size_t i = 0;
        r[i++] = 1;
        r[i++] = size(j, 0);
        r[i++] = size(j, 1);
        r[i++] = batch.gather_depth(j);
        r[i++] = batch.invalid_small(j);
        r[i++] = n_large;
        for (int k = 0; k < teno::MAX_FACES; k++) r[i++] = batch.small_size(j, k);
        auto values = [&](const auto & a, const size_t count) {
            const auto v = a.lane(j);
            for (size_t k = 0; k < count; k++) r[i++] = Kokkos::bit_cast<uint64_t>(double(v[k]));
        };
        auto stencil = [&](const auto & cells, const auto & faces, const size_t count) {
            const auto c = cells.lane(j), f = faces.lane(j);
            for (size_t k = 0; k < count; k++) {
                r[i++] = uint64_t(c[k]);
                r[i++] = uint64_t(int64_t(f[k]));
            }
        };
        values(batch.basis_mean, batch.basis_mean.cap);
        values(batch.si, batch.si.cap);
        stencil(batch.large_cells, batch.large_faces, n_large);
        values(batch.large_pinv, n_large * nk);
        stencil(batch.small_cells, batch.small_faces, n_small);
        values(batch.small_pinv, n_small * size_t(teno::NK_SMALL));
    });
    auto h_data = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), data);
    std::vector<std::vector<uint64_t>> out(n);
    for (uint32_t j = 0; j < n; j++) out[j].assign(h_data.data() + h_offset(j), h_data.data() + h_offset(j + 1));
    return out;
}

ShapeKeys::Records ShapeKeys::upload(const std::vector<std::vector<uint64_t>> & records) {
    Records out;
    out.offset = Kokkos::View<uint64_t *>("teno_shape_record_offsets", records.size() + 1);
    auto h_offset = Kokkos::create_mirror_view(out.offset);
    h_offset(0) = 0;
    for (size_t k = 0; k < records.size(); k++) h_offset(k + 1) = h_offset(k) + records[k].size();
    out.data = Kokkos::View<uint64_t *>(Kokkos::view_alloc(Kokkos::WithoutInitializing, "teno_shape_records"),
                                        h_offset(records.size()));
    auto h_data = Kokkos::create_mirror_view(out.data);
    for (size_t k = 0; k < records.size(); k++) std::copy(records[k].begin(), records[k].end(), h_data.data() + h_offset(k));
    Kokkos::deep_copy(out.offset, h_offset);
    Kokkos::deep_copy(out.data, h_data);
    return out;
}

void ShapeKeys::decode(const uint32_t c0, const uint32_t n, const Kokkos::View<uint32_t *> & group,
                       const Records & records, const teno_setup::TableBatch & out,
                       const Kokkos::View<uint8_t *> & bad) const {
    const ShapeSearch g = s;
    const auto offset = records.offset;
    const auto data = records.data;
    each_search(g, slots, n, KOKKOS_LAMBDA(const uint32_t t, const ShapeLanes & w) {
        const uint32_t c = c0 + t;
        bad(t) = 0;
        if (group(c) == ShapeGroups::NONE) return;
        uint32_t end[MAX_END];
        const uint64_t * rec = data.data() + offset(group(c));
        const ShapeSearch::Result r = g.search(c, w, end, rec[0] == 1 ? uint32_t(rec[2]) : 0u);
        bool ok = r.ok && rec[0] == 1 && rec[1] == r.n && rec[2] == r.layers;
        size_t at = 3;
        const uint8_t depth = static_cast<uint8_t>(rec[at++]);
        const uint8_t invalid = static_cast<uint8_t>(rec[at++]);
        const uint32_t n_large = static_cast<uint32_t>(rec[at++]);
        uint32_t n_small = 0;
        for (int k = 0; k < teno::MAX_FACES; k++) {
            out.small_size(t, k) = static_cast<uint16_t>(rec[at + size_t(k)]);
            n_small += static_cast<uint32_t>(rec[at + size_t(k)]);
        }
        at += teno::MAX_FACES;
        ok = ok && n_large <= out.max_large && n_small <= out.max_small;
        if (ok) {
            auto values = [&](const teno_setup::Lane<rtype> & to, const size_t count) {
                for (size_t k = 0; k < count; k++) to[k] = rtype(Kokkos::bit_cast<double>(rec[at++]));
            };
            auto stencil = [&](const teno_setup::Lane<int32_t> & entries, const teno_setup::Lane<int32_t> & faces,
                               const uint32_t count) {
                for (uint32_t k = 0; k < count; k++) {
                    const uint64_t p = rec[at++];
                    const int64_t code = int64_t(rec[at++]);
                    ok = ok && p < r.n;
                    if (!ok) return;
                    entries[k] = int32_t(w.cell[w.order[p]]);
                    if (code < 0) {
                        faces[k] = -1;
                        continue;
                    }
                    const uint64_t po = uint64_t(code) / 64;
                    ok = po < r.n;
                    if (!ok) return;
                    const int64_t f = g.boundary_face(w.cell[w.order[po]], int(code % 64));
                    ok = f >= 0;
                    faces[k] = int32_t(f);
                }
            };
            values(out.basis_mean.lane(t), out.basis_mean.cap);
            values(out.si.lane(t), out.si.cap);
            stencil(out.large_cells.lane(t), out.large_faces.lane(t), n_large);
            if (ok) values(out.large_pinv.lane(t), size_t(n_large) * out.nk);
            if (ok) stencil(out.small_cells.lane(t), out.small_faces.lane(t), n_small);
            if (ok) values(out.small_pinv.lane(t), size_t(n_small) * teno::NK_SMALL);
        }
        out.status(t) = teno_setup::OK;
        out.gather_depth(t) = depth;
        out.failed_large(t) = 0;
        out.invalid_small(t) = invalid;
        out.large_size(t) = static_cast<uint16_t>(n_large);
        out.small_total(t) = static_cast<uint16_t>(n_small);
        out.scale.lane(t)[0] = rtype(g.h(c));
        bad(t) = !ok;
        g.clear(w, r.n);
    });
}

std::vector<uint32_t> ShapeGroups::first_slots(const uint32_t * slot_group, const uint8_t * refit, const uint32_t n) {
    std::vector<uint32_t> first(n);
    std::unordered_map<uint32_t, uint32_t> held;
    for (uint32_t t = 0; t < n; t++) {
        first[t] = t;
        if (slot_group[t] != NONE && refit[t] == 0) first[t] = held.emplace(slot_group[t], t).first->second;
    }
    return first;
}

ShapeGroups ShapeGroups::build(const std::vector<ShapeKey> & keys, const uint64_t * gid) {
    auto id = [&](const uint32_t c) { return gid != nullptr ? gid[c] : uint64_t(c); };
    ShapeGroups out;
    out.group.assign(keys.size(), NONE);
    std::unordered_map<ShapeKey, uint32_t, ShapeKeyHash> index;
    std::vector<uint64_t> high;  // (group): highest local global id
    for (uint32_t c = 0; c < keys.size(); c++) {
        if (!keys[c]) continue;
        auto [it, added] = index.try_emplace(keys[c], uint32_t(out.groups.size()));
        if (added) {
            Group g;
            g.key = keys[c];
            g.rep_gid = id(c);
            g.rep_cell = c;
            out.groups.push_back(g);
            high.push_back(id(c));
        } else {
            Group & g = out.groups[it->second];
            if (id(c) < g.rep_gid) {
                g.rep_gid = id(c);
                g.rep_cell = c;
            }
            high[it->second] = std::max(high[it->second], id(c));
        }
        out.group[c] = it->second;
    }
    const int p = comm::size();
    if (p == 1) {
        for (size_t k = 0; k < out.groups.size(); k++) {
            out.groups[k].holder = 0;
            out.groups[k].single = out.groups[k].rep_gid == high[k];
        }
        return out;
    }
    // The rank owning a key learns every rank's lowest and highest global id of it
    std::vector<std::vector<uint64_t>> report(static_cast<size_t>(p));
    for (size_t k = 0; k < out.groups.size(); k++) {
        const Group & g = out.groups[k];
        report[g.key.hash % uint64_t(p)].insert(report[g.key.hash % uint64_t(p)].end(),
                                                {g.key.hash, g.key.check, g.rep_gid, high[k]});
    }
    const auto reports = comm::alltoallv(report);
    struct Total {
        uint64_t low, high;
        uint64_t holder;
    };
    std::unordered_map<ShapeKey, Total, ShapeKeyHash> total;
    for (int r = 0; r < p; r++) {
        const auto & from = reports[size_t(r)];
        for (size_t k = 0; k < from.size(); k += 4) {
            const ShapeKey key{from[k], from[k + 1]};
            auto [it, added] = total.try_emplace(key, Total{from[k + 2], from[k + 3], uint64_t(r)});
            if (added) continue;
            if (from[k + 2] < it->second.low) {
                it->second.low = from[k + 2];
                it->second.holder = uint64_t(r);
            }
            it->second.high = std::max(it->second.high, from[k + 3]);
        }
    }
    std::vector<std::vector<uint64_t>> answer(static_cast<size_t>(p));
    for (int r = 0; r < p; r++) {
        const auto & from = reports[size_t(r)];
        for (size_t k = 0; k < from.size(); k += 4) {
            const Total & t = total.at(ShapeKey{from[k], from[k + 1]});
            answer[size_t(r)].insert(answer[size_t(r)].end(), {from[k], from[k + 1], t.low, t.high, t.holder});
        }
    }
    for (const auto & from : comm::alltoallv(answer)) {
        for (size_t k = 0; k < from.size(); k += 5) {
            Group & g = out.groups[index.at(ShapeKey{from[k], from[k + 1]})];
            if (from[k + 2] != g.rep_gid) g.rep_cell = NONE;
            g.rep_gid = from[k + 2];
            g.single = from[k + 2] == from[k + 3];
            g.holder = int(from[k + 4]);
        }
    }
    return out;
}

void ShapeGroups::exchange(std::vector<std::vector<uint64_t>> & records) const {
    const int p = comm::size();
    if (p == 1) return;
    std::vector<std::vector<uint64_t>> request(static_cast<size_t>(p));
    std::vector<std::vector<size_t>> asked(static_cast<size_t>(p));
    for (size_t k = 0; k < groups.size(); k++) {
        const Group & g = groups[k];
        if (g.single || g.rep_cell != NONE) continue;
        request[size_t(g.holder)].insert(request[size_t(g.holder)].end(), {g.key.hash, g.key.check});
        asked[size_t(g.holder)].push_back(k);
    }
    const auto requests = comm::alltoallv(request);
    std::unordered_map<ShapeKey, size_t, ShapeKeyHash> index;
    for (size_t k = 0; k < groups.size(); k++) index.emplace(groups[k].key, k);
    std::vector<std::vector<uint64_t>> reply(static_cast<size_t>(p));
    for (int r = 0; r < p; r++) {
        const auto & from = requests[size_t(r)];
        for (size_t k = 0; k < from.size(); k += 2) {
            const auto & rec = records[index.at(ShapeKey{from[k], from[k + 1]})];
            reply[size_t(r)].push_back(rec.size());
            reply[size_t(r)].insert(reply[size_t(r)].end(), rec.begin(), rec.end());
        }
    }
    const auto replies = comm::alltoallv(reply);
    for (int r = 0; r < p; r++) {
        size_t at = 0;
        for (const size_t k : asked[size_t(r)]) {
            const uint64_t n = replies[size_t(r)][at++];
            records[k].assign(replies[size_t(r)].begin() + long(at), replies[size_t(r)].begin() + long(at + n));
            at += n;
        }
    }
}

} // namespace teno

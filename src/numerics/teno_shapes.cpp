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
#include <array>
#include <cmath>
#include <unordered_map>

#include <Kokkos_Core.hpp>

#include "comm.h"
#include "teno_setup.h"

namespace teno {

namespace {

constexpr uint64_t SEED_HASH = 0x243f6a8885a308d3ULL, SEED_CHECK = 0x13198a2e03707344ULL;

uint64_t mix(uint64_t h) {
    h = (h ^ (h >> 30)) * 0xbf58476d1ce4e5b9ULL;
    h = (h ^ (h >> 27)) * 0x94d049bb133111ebULL;
    return h ^ (h >> 31);
}

/** @brief Two independent hashes of a sequence of words. */
struct Hasher {
    uint64_t a = SEED_HASH, b = SEED_CHECK;
    void add(const uint64_t v) {
        a = mix(a ^ (v + 0x9e3779b97f4a7c15ULL));
        b = mix(b + (v ^ 0xd1b54a32d192ed03ULL) * 0xff51afd7ed558ccdULL);
    }
    void add_signed(const int64_t v) { add(static_cast<uint64_t>(v)); }
};

// Scaling by a power of two is exact, as ldexp is
int64_t on_grid(const double v) { return std::llround(v * double(int64_t(1) << ShapeKeys::SHAPE_GRID_BITS)); }

using Lattice = std::array<int16_t, 3>;

} // namespace

bool ShapeKeys::supported(const Mesh & mesh) { return !mesh.curved_geometry && !mesh.is_rotational(); }

ShapeKeys::ShapeKeys(const Mesh & mesh_in, const BoundaryData & boundaries, const uint32_t n_large_in,
                     const uint32_t n_small_in)
    : mesh(mesh_in), n_large(n_large_in), n_small(n_small_in) {
    auto bc = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), boundaries.face_bc);
    face_bc.assign(bc.data(), bc.data() + bc.extent(0));
    const uint32_t n_cells = mesh.n_cells;
    shape.assign(n_cells, 0);
    h.assign(n_cells, 0.0);
    Kokkos::parallel_for("teno_shape_cells", Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(0, n_cells),
                         [&](const uint32_t c) {
        const double hc = teno_setup::cell_scale(mesh, c);
        h[c] = hc;
        double xc[N_DIM];
        for (int d = 0; d < N_DIM; d++) xc[d] = double(mesh.h_cell_coords(c, d));
        std::vector<std::array<int64_t, N_DIM + 1>> items;
        for (uint32_t k = mesh.h_offsets_nodes_of_cell(c); k < mesh.h_offsets_nodes_of_cell(c + 1); k++) {
            std::array<int64_t, N_DIM + 1> v{};
            for (int d = 0; d < N_DIM; d++) v[size_t(d)] = on_grid((double(mesh.h_node_coords(mesh.h_nodes_of_cell(k), d)) - xc[d]) / hc);
            v[N_DIM] = -1;
            items.push_back(v);
        }
        for (uint32_t k = mesh.h_offsets_faces_of_cell(c); k < mesh.h_offsets_faces_of_cell(c + 1); k++) {
            const uint32_t f = mesh.h_faces_of_cell(k);
            if (mesh.h_cells_of_face(f, 1) >= 0) continue;
            std::array<int64_t, N_DIM + 1> v{};
            for (int d = 0; d < N_DIM; d++) v[size_t(d)] = on_grid((double(mesh.h_face_coords(f, d)) - xc[d]) / hc);
            v[N_DIM] = face_bc[f];
            items.push_back(v);
        }
        std::sort(items.begin(), items.end());
        Hasher hs;
        hs.add(mesh.h_offsets_faces_of_cell(c + 1) - mesh.h_offsets_faces_of_cell(c));
        for (const auto & v : items) {
            for (const int64_t x : v) hs.add_signed(x);
        }
        shape[c] = hs.a;
    });
}

bool ShapeKeys::scan(const uint32_t c, Scan & s, const uint32_t layer_cap) const {
    const uint32_t n_cells = mesh.n_cells;
    if (s.stamp.size() != n_cells) {
        s.stamp.assign(n_cells, 0);
        s.index.assign(n_cells, 0);
        s.generation = 0;
    }
    if (++s.generation == 0) {
        std::fill(s.stamp.begin(), s.stamp.end(), 0u);
        s.generation = 1;
    }
    s.cell.assign(1, c);
    s.lattice.assign(1, Lattice{0, 0, 0});
    s.end.assign(1, 1);
    s.stamp[c] = s.generation;
    s.index[c] = 0;
    const uint32_t n_complete = mesh.n_complete();
    bool unique = true, exhausted = false;
    // The search's vertex-neighbor layers (reach_layers in teno_setup.cpp)
    auto grow = [&]() {
        if (exhausted) return false;
        const uint32_t begin = s.end.size() > 1 ? s.end[s.end.size() - 2] : 0;
        const uint32_t last = s.end.back();
        for (uint32_t k = begin; k < last; k++) {
            const uint32_t v = s.cell[k];
            const Lattice l = s.lattice[k];
            for (uint32_t j = mesh.h_offsets_cells_of_cell(v); j < mesh.h_offsets_cells_of_cell(v + 1); j++) {
                const uint32_t nb = mesh.h_cells_of_cell(j);
                const auto & sl = mesh.shift_lattice[mesh.h_cells_of_cell_shift(j)];
                const Lattice nl{int16_t(l[0] + sl[0]), int16_t(l[1] + sl[1]), int16_t(l[2] + sl[2])};
                if (s.stamp[nb] == s.generation) {
                    if (s.lattice[s.index[nb]] != nl) unique = false;
                    continue;
                }
                s.stamp[nb] = s.generation;
                s.index[nb] = static_cast<uint32_t>(s.cell.size());
                s.cell.push_back(nb);
                s.lattice.push_back(nl);
            }
        }
        if (s.cell.size() == last) {
            exhausted = true;
            return false;
        }
        s.end.push_back(static_cast<uint32_t>(s.cell.size()));
        return true;
    };
    auto reach = [&](const uint32_t n_min, const int max_layers) {
        int depth = 0;
        while (depth < max_layers && s.end[size_t(depth)] <= n_min) {
            if (size_t(depth) + 1 == s.end.size() && !grow()) break;
            depth++;
        }
        return depth;
    };
    if (layer_cap > 0) {
        while (s.end.size() <= layer_cap && grow()) {}
    } else {
        reach(n_large, 64);
        reach(8 * n_small, 6);
    }
    if (!unique) return false;
    double x0[N_DIM];
    for (int d = 0; d < N_DIM; d++) x0[d] = double(mesh.h_cell_coords(c, d));
    const double h0 = h[c];
    s.entries.resize(s.cell.size());
    int32_t layer = 0;
    for (size_t k = 0; k < s.cell.size(); k++) {
        while (k >= s.end[size_t(layer)]) layer++;
        const uint32_t v = s.cell[k];
        if (v >= n_complete) return false;
        Scan::Entry & e = s.entries[k];
        e.layer = layer;
        e.cell = v;
        for (int d = 0; d < N_DIM; d++) {
            double x = double(mesh.h_cell_coords(v, d));
            for (size_t j = 0; j < mesh.periodic_translations.size(); j++) {
                x += double(s.lattice[k][j]) * double(mesh.periodic_translations[j][size_t(d)]);
            }
            e.q[size_t(d)] = on_grid((x - x0[d]) / h0);
        }
    }
    // Entries come in layer order: sorting each layer by position sorts them by (layer, position)
    for (size_t l = 0; l < s.end.size(); l++) {
        const auto first = s.entries.begin() + (l > 0 ? std::ptrdiff_t(s.end[l - 1]) : 0);
        std::sort(first, s.entries.begin() + std::ptrdiff_t(s.end[l]),
                  [](const Scan::Entry & a, const Scan::Entry & b) { return a.q < b.q; });
    }
    for (size_t k = 1; k < s.entries.size(); k++) {
        if (s.entries[k].layer == s.entries[k - 1].layer && s.entries[k].q == s.entries[k - 1].q) return false;
    }
    return true;
}

std::vector<ShapeKey> ShapeKeys::keys(const uint32_t n) const {
    // A cell whose own shape no other owned cell has is a shape of its own
    std::unordered_map<uint64_t, uint64_t> count;
    for (uint32_t c = 0; c < mesh.n_owned(); c++) count[shape[c]]++;
    const int p = comm::size();
    if (p > 1) {
        std::vector<std::vector<uint64_t>> report(static_cast<size_t>(p));
        for (const auto & [d, k] : count) report[d % uint64_t(p)].insert(report[d % uint64_t(p)].end(), {d, k});
        const auto reports = comm::alltoallv(report);
        std::unordered_map<uint64_t, uint64_t> total;
        for (const auto & from : reports) {
            for (size_t k = 0; k < from.size(); k += 2) total[from[k]] += from[k + 1];
        }
        std::vector<std::vector<uint64_t>> answer(static_cast<size_t>(p));
        for (int r = 0; r < p; r++) {
            for (size_t k = 0; k < reports[size_t(r)].size(); k += 2) {
                answer[size_t(r)].insert(answer[size_t(r)].end(), {reports[size_t(r)][k], total.at(reports[size_t(r)][k])});
            }
        }
        count.clear();
        for (const auto & from : comm::alltoallv(answer)) {
            for (size_t k = 0; k < from.size(); k += 2) count[from[k]] = from[k + 1];
        }
    }
    std::vector<ShapeKey> out(n);
    Kokkos::Experimental::UniqueToken<Kokkos::DefaultHostExecutionSpace> token;
    std::vector<Scan> scans(static_cast<size_t>(token.size()));
    Kokkos::parallel_for("teno_shape_keys",
                         Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace, Kokkos::Schedule<Kokkos::Dynamic>>(0, n),
                         [&](const uint32_t c) {
        const auto it = count.find(shape[c]);
        if (it == count.end() || it->second < 2) return;
        const int id = token.acquire();
        Scan & s = scans[size_t(id)];
        if (scan(c, s)) {
            Hasher hs;
            hs.add(s.end.size());
            hs.add(s.entries.size());
            // The cell's faces in its own order, in its frame
            double x0[N_DIM];
            for (int d = 0; d < N_DIM; d++) x0[d] = double(mesh.h_cell_coords(c, d));
            for (uint32_t k = mesh.h_offsets_faces_of_cell(c); k < mesh.h_offsets_faces_of_cell(c + 1); k++) {
                const uint32_t f = mesh.h_faces_of_cell(k);
                const bool second = mesh.h_cells_of_face(f, 1) == int32_t(c);
                hs.add(second);
                for (int d = 0; d < N_DIM; d++) {
                    const double shift = second ? double(mesh.h_shifts(mesh.h_face_shift(f), d)) : 0.0;
                    hs.add_signed(on_grid((double(mesh.h_face_coords(f, d)) - shift - x0[d]) / h[c]));
                }
            }
            for (const Scan::Entry & e : s.entries) {
                hs.add_signed(e.layer);
                for (const int64_t q : e.q) hs.add_signed(q);
                hs.add(shape[e.cell]);
                hs.add_signed(on_grid(h[e.cell] / h[c]));
            }
            out[c] = ShapeKey{hs.a == 0 ? 1 : hs.a, hs.b};
        }
        token.release(id);
    });
    return out;
}

bool ShapeKeys::neighborhood(const uint32_t c, std::vector<uint32_t> & cells, uint32_t & layers, Scan & s,
                             const uint32_t max_layers) const {
    if (!scan(c, s, max_layers)) return false;
    layers = static_cast<uint32_t>(s.end.size() - 1);
    cells.resize(s.entries.size());
    for (size_t k = 0; k < cells.size(); k++) cells[k] = s.entries[k].cell;
    return true;
}

std::vector<uint32_t> ShapeKeys::sorted_boundary_faces(const uint32_t c) const {
    std::vector<std::pair<std::array<int64_t, N_DIM>, uint32_t>> faces;
    for (uint32_t k = mesh.h_offsets_faces_of_cell(c); k < mesh.h_offsets_faces_of_cell(c + 1); k++) {
        const uint32_t f = mesh.h_faces_of_cell(k);
        if (mesh.h_cells_of_face(f, 1) >= 0) continue;
        std::array<int64_t, N_DIM> q{};
        for (int d = 0; d < N_DIM; d++) q[size_t(d)] = on_grid((double(mesh.h_face_coords(f, d)) - double(mesh.h_cell_coords(c, d))) / h[c]);
        faces.emplace_back(q, f);
    }
    std::sort(faces.begin(), faces.end());
    std::vector<uint32_t> out;
    for (const auto & qf : faces) out.push_back(qf.second);
    return out;
}

int ShapeKeys::boundary_rank(const uint32_t c, const uint32_t f) const {
    const std::vector<uint32_t> faces = sorted_boundary_faces(c);
    const auto it = std::find(faces.begin(), faces.end(), f);
    return it == faces.end() ? -1 : int(it - faces.begin());
}

int64_t ShapeKeys::boundary_face(const uint32_t c, const int rank) const {
    const std::vector<uint32_t> faces = sorted_boundary_faces(c);
    return rank >= 0 && size_t(rank) < faces.size() ? int64_t(faces[size_t(rank)]) : -1;
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

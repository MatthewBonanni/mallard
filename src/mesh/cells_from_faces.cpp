/**
 * @file cells_from_faces.cpp
 * @author Matthew Bonanni (mbonanni001@gmail.com)
 * @brief Cell corner lists of face-based meshes (Fluent, OpenFOAM).
 * @version 0.9
 * @date 2026-10-15
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 *
 */

#include "mesh_file.h"

#include <algorithm>
#include <map>
#include <set>
#include <stdexcept>

namespace {

using Face = std::vector<uint32_t>;

[[maybe_unused]] std::vector<uint32_t> polygon_of_edges(const std::vector<const Face *> & edges) {
    std::vector<uint32_t> ring = {(*edges[0])[0], (*edges[0])[1]};
    std::vector<bool> used(edges.size(), false);
    used[0] = true;
    for (size_t step = 1; step + 1 < edges.size(); step++) {
        bool found = false;
        for (size_t e = 0; e < edges.size() && !found; e++) {
            if (used[e]) continue;
            const Face & edge = *edges[e];
            if (edge[0] == ring.back() || edge[1] == ring.back()) {
                ring.push_back(edge[0] == ring.back() ? edge[1] : edge[0]);
                used[e] = found = true;
            }
        }
        if (!found) return {};
    }
    return ring;
}

bool has_edge(const std::set<std::pair<uint32_t, uint32_t>> & edges, uint32_t a, uint32_t b) {
    return edges.count({std::min(a, b), std::max(a, b)}) > 0;
}

/** @brief Corners of a 3D cell from its faces, or empty if its faces are no tetrahedron, pyramid, prism or hexahedron. */
[[maybe_unused]] std::vector<uint32_t> solid_of_faces(const std::vector<const Face *> & faces) {
    std::vector<const Face *> tris, quads;
    std::set<std::pair<uint32_t, uint32_t>> edges;
    for (const Face * f : faces) {
        if (f->size() == 3) tris.push_back(f);
        else if (f->size() == 4) quads.push_back(f);
        else return {};
        for (size_t k = 0; k < f->size(); k++) {
            const uint32_t a = (*f)[k], b = (*f)[(k + 1) % f->size()];
            edges.insert({std::min(a, b), std::max(a, b)});
        }
    }
    auto not_in = [](const Face & f, const Face & g) {
        for (uint32_t n : g) {
            if (std::find(f.begin(), f.end(), n) == f.end()) return n;
        }
        return ~uint32_t(0);
    };
    // The node of `other` joined by an edge to each node of `base`
    auto partners = [&](const Face & base, const std::vector<uint32_t> & other) {
        std::vector<uint32_t> out = base;
        for (uint32_t a : base) {
            uint32_t p = ~uint32_t(0);
            for (uint32_t b : other) {
                if (has_edge(edges, a, b)) p = b;
            }
            out.push_back(p);
        }
        return out;
    };
    if (tris.size() == 4 && quads.empty()) {
        Face c = *tris[0];
        c.push_back(not_in(c, *tris[1]));
        return c;
    }
    if (tris.size() == 4 && quads.size() == 1) {
        Face c = *quads[0];
        c.push_back(not_in(c, *tris[0]));
        return c;
    }
    if (tris.size() == 2 && quads.size() == 3) return partners(*tris[0], *tris[1]);
    if (tris.empty() && quads.size() == 6) {
        const Face & bottom = *quads[0];
        std::vector<uint32_t> top;
        for (const Face * q : quads) {
            if (std::none_of(q->begin(), q->end(), [&](uint32_t n) {
                    return std::find(bottom.begin(), bottom.end(), n) != bottom.end();
                })) {
                top = *q;
            }
        }
        return partners(bottom, top);
    }
    return {};
}

std::vector<std::vector<uint32_t>> sorted_faces(const std::vector<const Face *> & faces) {
    std::vector<std::vector<uint32_t>> out;
    for (const Face * f : faces) {
        out.push_back(*f);
        std::sort(out.back().begin(), out.back().end());
    }
    std::sort(out.begin(), out.end());
    return out;
}

} // namespace

std::vector<std::vector<uint32_t>> cells_from_faces(uint32_t n_cells, const std::vector<std::vector<uint32_t>> & faces,
                                                    const std::vector<std::array<int64_t, 2>> & face_cells,
                                                    const std::string & source) {
    std::vector<std::vector<const Face *>> cell_faces(n_cells);
    for (size_t f = 0; f < faces.size(); f++) {
        for (int64_t c : face_cells[f]) {
            if (c < 0) continue;
            if (c >= int64_t(n_cells)) throw std::runtime_error(source + ": a face refers to cell " + std::to_string(c + 1) + " of " + std::to_string(n_cells) + ".");
            cell_faces[size_t(c)].push_back(&faces[f]);
        }
    }
    std::vector<std::vector<uint32_t>> cells(n_cells);
    for (uint32_t c = 0; c < n_cells; c++) {
        const auto & cf = cell_faces[c];
        std::vector<uint32_t> corners;
        if constexpr (N_DIM == 2) {
            if ((cf.size() == 3 || cf.size() == 4) &&
                std::all_of(cf.begin(), cf.end(), [](const Face * f) { return f->size() == 2; })) {
                corners = polygon_of_edges(cf);
            }
        } else {
            corners = solid_of_faces(cf);
        }
        if (!corners.empty()) {
            std::vector<uint32_t> sorted = corners;
            std::sort(sorted.begin(), sorted.end());
            if (std::adjacent_find(sorted.begin(), sorted.end()) != sorted.end() || sorted.back() == ~uint32_t(0)) {
                corners.clear();
            }
        }
        if (!corners.empty() && N_DIM == 3) {
            // The faces of the reconstructed cell must be exactly the given ones
            std::vector<Face> local;
            std::vector<const Face *> pointers;
            for (const auto & lf : cell_local_faces(uint32_t(corners.size()))) {
                local.emplace_back();
                for (uint8_t k : lf) local.back().push_back(corners[k]);
            }
            for (const Face & f : local) pointers.push_back(&f);
            if (sorted_faces(pointers) != sorted_faces(cf)) corners.clear();
        }
        if (corners.empty()) {
            std::string sizes;
            for (const Face * f : cf) sizes += (sizes.empty() ? "" : ", ") + std::to_string(f->size());
            throw std::runtime_error(source + ": cell " + std::to_string(c + 1) + " has " + std::to_string(cf.size()) +
                                     " faces with " + sizes + " nodes; Mallard supports " +
                                     (N_DIM == 2 ? "triangles and quadrilaterals"
                                                 : "tetrahedra, pyramids, prisms and hexahedra") +
                                     " (no general polygons or polyhedra).");
        }
        cells[c] = std::move(corners);
    }
    return cells;
}

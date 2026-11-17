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
#include <set>
#include <span>
#include <stdexcept>

namespace {

using Face = std::span<const uint64_t>;
using Corners = std::vector<uint64_t>;
constexpr uint64_t MISSING = ~uint64_t(0);

[[maybe_unused]] Corners polygon_of_edges(const std::vector<Face> & edges) {
    Corners ring = {edges[0][0], edges[0][1]};
    std::vector<bool> used(edges.size(), false);
    used[0] = true;
    for (size_t step = 1; step + 1 < edges.size(); step++) {
        bool found = false;
        for (size_t e = 0; e < edges.size() && !found; e++) {
            if (used[e]) continue;
            const Face & edge = edges[e];
            if (edge[0] == ring.back() || edge[1] == ring.back()) {
                ring.push_back(edge[0] == ring.back() ? edge[1] : edge[0]);
                used[e] = found = true;
            }
        }
        if (!found) return {};
    }
    return ring;
}

bool has_edge(const std::set<std::pair<uint64_t, uint64_t>> & edges, uint64_t a, uint64_t b) {
    return edges.count({std::min(a, b), std::max(a, b)}) > 0;
}

/** @brief Corners of a 3D cell from its faces, or empty if its faces are no tetrahedron, pyramid, prism or hexahedron. */
[[maybe_unused]] Corners solid_of_faces(const std::vector<Face> & faces) {
    std::vector<Face> tris, quads;
    std::set<std::pair<uint64_t, uint64_t>> edges;
    for (const Face & f : faces) {
        if (f.size() == 3) tris.push_back(f);
        else if (f.size() == 4) quads.push_back(f);
        else return {};
        for (size_t k = 0; k < f.size(); k++) {
            const uint64_t a = f[k], b = f[(k + 1) % f.size()];
            edges.insert({std::min(a, b), std::max(a, b)});
        }
    }
    auto not_in = [](const Face & f, const Face & g) {
        for (uint64_t n : g) {
            if (std::find(f.begin(), f.end(), n) == f.end()) return n;
        }
        return MISSING;
    };
    // The node of `other` joined by an edge to each node of `base`
    auto partners = [&](const Face & base, const Face & other) {
        Corners out(base.begin(), base.end());
        for (uint64_t a : base) {
            uint64_t p = MISSING;
            for (uint64_t b : other) {
                if (has_edge(edges, a, b)) p = b;
            }
            out.push_back(p);
        }
        return out;
    };
    if (tris.size() == 4 && quads.empty()) {
        Corners c(tris[0].begin(), tris[0].end());
        c.push_back(not_in(tris[0], tris[1]));
        return c;
    }
    if (tris.size() == 4 && quads.size() == 1) {
        Corners c(quads[0].begin(), quads[0].end());
        c.push_back(not_in(quads[0], tris[0]));
        return c;
    }
    if (tris.size() == 2 && quads.size() == 3) return partners(tris[0], tris[1]);
    if (tris.empty() && quads.size() == 6) {
        const Face & bottom = quads[0];
        Face top;
        for (const Face & q : quads) {
            if (std::none_of(q.begin(), q.end(), [&](uint64_t n) {
                    return std::find(bottom.begin(), bottom.end(), n) != bottom.end();
                })) {
                top = q;
            }
        }
        return partners(bottom, top);
    }
    return {};
}

std::vector<Corners> sorted_faces(const std::vector<Face> & faces) {
    std::vector<Corners> out;
    for (const Face & f : faces) {
        out.emplace_back(f.begin(), f.end());
        std::sort(out.back().begin(), out.back().end());
    }
    std::sort(out.begin(), out.end());
    return out;
}

} // namespace

std::vector<uint64_t> cell_corners(const std::vector<std::span<const uint64_t>> & cf, uint64_t c, const std::string & source) {
    Corners corners;
    if constexpr (N_DIM == 2) {
        if ((cf.size() == 3 || cf.size() == 4) && std::all_of(cf.begin(), cf.end(), [](const Face & f) { return f.size() == 2; })) {
            corners = polygon_of_edges(cf);
        }
    } else {
        corners = solid_of_faces(cf);
    }
    if (!corners.empty()) {
        Corners sorted = corners;
        std::sort(sorted.begin(), sorted.end());
        if (std::adjacent_find(sorted.begin(), sorted.end()) != sorted.end() || sorted.back() == MISSING) corners.clear();
    }
    if (!corners.empty() && N_DIM == 3) {
        // The faces of the reconstructed cell must be exactly the given ones
        std::vector<Corners> local;
        for (const auto & lf : cell_local_faces(uint32_t(corners.size()))) {
            local.emplace_back();
            for (uint8_t k : lf) local.back().push_back(corners[k]);
        }
        const std::vector<Face> spans(local.begin(), local.end());
        if (sorted_faces(spans) != sorted_faces(cf)) corners.clear();
    }
    if (corners.empty()) {
        std::string sizes;
        for (const Face & f : cf) sizes += (sizes.empty() ? "" : ", ") + std::to_string(f.size());
        throw std::runtime_error(source + ": cell " + std::to_string(c + 1) + " has " + std::to_string(cf.size()) +
                                 " faces with " + sizes + " nodes; Mallard supports " +
                                 (N_DIM == 2 ? "triangles and quadrilaterals" : "tetrahedra, pyramids, prisms and hexahedra") +
                                 " (no general polygons or polyhedra).");
    }
    return corners;
}

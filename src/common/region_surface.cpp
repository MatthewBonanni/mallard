/**
 * @file region_surface.cpp
 * @author Matthew Bonanni (mbonanni001@gmail.com)
 * @brief Regions from CAD surfaces: reading, checking and laying out STL and
 *        OBJ files.
 * @version 0.1
 * @date 2026-11-19
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 *
 */

#include "region_surface.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iterator>
#include <limits>
#include <numeric>
#include <sstream>
#include <utility>

#include "geometry.h"
#include "input.h"
#include "log.h"
#include "provenance.h"

namespace region::surface {

namespace {

constexpr const char * DOCS = "docs/guide/geometry.md#regions-from-cad-surfaces";

Vec sub(const Vec & a, const Vec & b) { return {a[0] - b[0], a[1] - b[1], a[2] - b[2]}; }
double dot(const Vec & a, const Vec & b) { return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]; }
Vec cross(const Vec & a, const Vec & b) {
    return {a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]};
}
double norm(const Vec & a) { return std::sqrt(dot(a, a)); }

std::string point(const Vec & x) {
    return "(" + logging::real(x[0]) + ", " + logging::real(x[1]) + ", " + logging::real(x[2]) + ") m";
}

[[noreturn]] void fail(const Where & where, const std::string & message, const std::string & fix) {
    throw InputError(Diagnostic{.key = where.key, .line = where.line, .message = message, .fix = fix, .docs = DOCS});
}

/** @brief A number of an ASCII file at its line, or an error. */
double number(const std::string & token, const std::string & file, const std::size_t line, const Where & where) {
    char * end = nullptr;
    const double v = std::strtod(token.c_str(), &end);
    if (token.empty() || end != token.c_str() + token.size()) {
        fail(where, file + ":" + std::to_string(line) + ": \"" + token + "\" is not a number", "");
    }
    return v;
}

std::vector<Triangle> read_stl(const std::string & bytes, const std::string & file, const Where & where) {
    std::vector<Triangle> out;
    if (bytes.size() >= 84) {
        const auto byte = [&](const std::size_t i) { return static_cast<uint32_t>(static_cast<unsigned char>(bytes[i])); };
        const auto u32 = [&](const std::size_t i) { return byte(i) | byte(i + 1) << 8 | byte(i + 2) << 16 | byte(i + 3) << 24; };
        const uint64_t n = u32(80);
        if (84 + 50 * n == bytes.size()) {
            out.resize(n);
            for (uint64_t t = 0; t < n; t++) {
                for (std::size_t k = 0; k < 3; k++) {
                    for (std::size_t i = 0; i < 3; i++) {
                        const uint32_t u = u32(84 + 50 * t + 12 + 12 * k + 4 * i);
                        float f = 0.0f;
                        std::memcpy(&f, &u, sizeof f);
                        out[t][k][i] = static_cast<double>(f);
                    }
                }
            }
            return out;
        }
    }
    std::istringstream in(bytes);
    std::string line, word;
    std::size_t line_number = 0;
    bool solid = false;
    int in_loop = -1;  // corners of the open loop, -1 outside one
    Triangle t{};
    while (std::getline(in, line)) {
        line_number++;
        std::istringstream words(line);
        if (!(words >> word)) continue;
        if (!solid) {
            if (word != "solid") break;
            solid = true;
        } else if (word == "outer") {
            in_loop = 0;
        } else if (word == "vertex") {
            if (in_loop < 0 || in_loop >= 3) {
                fail(where, file + ":" + std::to_string(line_number) + ": a vertex outside a loop of 3",
                     "Each facet of an ASCII STL file holds one outer loop of 3 vertices");
            }
            for (std::size_t i = 0; i < 3; i++) {
                std::string token;
                words >> token;
                t[static_cast<std::size_t>(in_loop)][i] = number(token, file, line_number, where);
            }
            in_loop++;
        } else if (word == "endloop") {
            if (in_loop != 3) {
                fail(where, file + ":" + std::to_string(line_number) + ": a loop of " + std::to_string(std::max(in_loop, 0)) + " vertices",
                     "Each facet of an ASCII STL file holds one outer loop of 3 vertices");
            }
            out.push_back(t);
            in_loop = -1;
        }
    }
    if (!solid) {
        fail(where, file + " is not an STL file: its size is not that of a binary one (84 + 50 n bytes for n triangles) "
                    "and it does not start with \"solid\" as ASCII ones do",
             "Export the surface as STL (binary or ASCII), or give an OBJ file with type = \"obj\"");
    }
    return out;
}

std::vector<Triangle> read_obj(const std::string & bytes, const std::string & file, const Where & where) {
    std::vector<Triangle> out;
    std::vector<Vec> points;
    std::istringstream in(bytes);
    std::string line, word;
    std::size_t line_number = 0;
    while (std::getline(in, line)) {
        line_number++;
        std::istringstream words(line);
        if (!(words >> word)) continue;
        const std::string at = file + ":" + std::to_string(line_number);
        if (word == "v") {
            Vec x{};
            for (std::size_t i = 0; i < 3; i++) {
                std::string token;
                words >> token;
                x[i] = number(token, file, line_number, where);
            }
            points.push_back(x);
        } else if (word == "f") {
            std::vector<Vec> face;
            std::string token;
            while (words >> token) {
                const std::string index = token.substr(0, token.find('/'));
                char * end = nullptr;
                const long long i = std::strtoll(index.c_str(), &end, 10);
                const long long n = static_cast<long long>(points.size());
                const long long k = i < 0 ? n + i : i - 1;
                if (index.empty() || end != index.c_str() + index.size() || i == 0 || k < 0 || k >= n) {
                    fail(where, at + ": \"" + token + "\" is not the index of a vertex before it (1 to " +
                                    std::to_string(n) + ", or -1 to -" + std::to_string(n) + ")",
                         "");
                }
                face.push_back(points[static_cast<std::size_t>(k)]);
            }
            if (face.size() < 3) fail(where, at + ": a face of " + std::to_string(face.size()) + " vertices", "");
            // A fan from the first corner, which covers the face only if it is convex
            Vec normal = {0.0, 0.0, 0.0};
            for (std::size_t k = 0; k < face.size(); k++) {
                const Vec w = cross(face[k], face[(k + 1) % face.size()]);
                for (std::size_t i = 0; i < 3; i++) normal[i] += w[i];
            }
            for (std::size_t k = 1; k + 1 < face.size(); k++) {
                if (face.size() > 3 && !(dot(cross(sub(face[k], face[0]), sub(face[k + 1], face[0])), normal) > 0.0)) {
                    fail(where, at + ": a face of " + std::to_string(face.size()) + " vertices that is not convex",
                         "Export the surface as triangles");
                }
                out.push_back({face[0], face[k], face[k + 1]});
            }
        }
    }
    return out;
}

/** @brief Union-find with path halving. */
struct Sets {
    std::vector<uint32_t> parent;
    explicit Sets(const std::size_t n) : parent(n) { std::iota(parent.begin(), parent.end(), 0u); }
    uint32_t find(uint32_t i) {
        while (parent[i] != i) i = parent[i] = parent[parent[i]];
        return i;
    }
    void join(const uint32_t a, const uint32_t b) {
        const uint32_t x = find(a), y = find(b);
        if (x != y) parent[std::max(x, y)] = std::min(x, y);
    }
};

/** @brief A triangle's edge k (from corner k to corner k + 1), by its vertices in ascending order. */
struct Edge {
    uint32_t lo = 0, hi = 0;
    uint32_t triangle = 0;
    uint8_t k = 0;
    bool forward = true;  // corner k is lo

    bool operator<(const Edge & o) const {
        return lo != o.lo ? lo < o.lo : hi != o.hi ? hi < o.hi : triangle != o.triangle ? triangle < o.triangle : k < o.k;
    }
};

using Corners = std::array<uint32_t, 3>;

std::vector<Edge> edges_of(const std::vector<Corners> & tris) {
    std::vector<Edge> edges;
    edges.reserve(3 * tris.size());
    for (uint32_t t = 0; t < tris.size(); t++) {
        for (std::size_t k = 0; k < 3; k++) {
            const uint32_t a = tris[t][k], b = tris[t][(k + 1) % 3];
            edges.push_back({std::min(a, b), std::max(a, b), t, static_cast<uint8_t>(k), a < b});
        }
    }
    std::sort(edges.begin(), edges.end());
    return edges;
}

/** @brief The triangles laid out for Kind::TRIANGLES (closed and oriented). */
std::vector<double> layout(const std::vector<Vec> & points, const std::vector<Corners> & tris, const Where & where) {
    const std::size_t n = tris.size();
    std::vector<Vec> normal(n);
    for (std::size_t t = 0; t < n; t++) {
        const Vec w = cross(sub(points[tris[t][1]], points[tris[t][0]]), sub(points[tris[t][2]], points[tris[t][0]]));
        const double l = norm(w);
        normal[t] = l > 0.0 ? Vec{w[0] / l, w[1] / l, w[2] / l} : Vec{0.0, 0.0, 0.0};
    }
    // Pseudonormals: of an edge the sum of its two faces' normals, of a corner their sum weighted by its angles
    std::vector<Vec> corner(points.size(), Vec{0.0, 0.0, 0.0});
    for (std::size_t t = 0; t < n; t++) {
        for (std::size_t k = 0; k < 3; k++) {
            const Vec & a = points[tris[t][k]];
            const Vec u = sub(points[tris[t][(k + 1) % 3]], a), v = sub(points[tris[t][(k + 2) % 3]], a);
            const double angle = std::atan2(norm(cross(u, v)), dot(u, v));
            for (std::size_t i = 0; i < 3; i++) corner[tris[t][k]][i] += angle * normal[t][i];
        }
    }
    const std::vector<Edge> edges = edges_of(tris);
    std::vector<std::array<Vec, 3>> edge_normal(n);
    for (std::size_t e = 0; e + 1 < edges.size(); e += 2) {
        const Edge & a = edges[e];
        const Edge & b = edges[e + 1];
        Vec s;
        for (std::size_t i = 0; i < 3; i++) s[i] = normal[a.triangle][i] + normal[b.triangle][i];
        edge_normal[a.triangle][std::size_t{a.k}] = s;
        edge_normal[b.triangle][std::size_t{b.k}] = s;
    }
    // The hierarchy: boxes split at the median of the triangles' centroids along the longest side
    std::vector<uint32_t> order(n);
    std::iota(order.begin(), order.end(), 0u);
    std::vector<Vec> centroid(n);
    for (std::size_t t = 0; t < n; t++) {
        for (std::size_t i = 0; i < 3; i++) {
            centroid[t][i] = (points[tris[t][0]][i] + points[tris[t][1]][i] + points[tris[t][2]][i]) / 3.0;
        }
    }
    std::vector<std::array<double, triangles::NODE>> nodes;
    int deepest = 0;
    const auto build = [&](auto && self, const std::size_t begin, const std::size_t end, const int depth) -> uint32_t {
        deepest = std::max(deepest, depth);
        const uint32_t index = static_cast<uint32_t>(nodes.size());
        nodes.emplace_back();
        std::array<double, triangles::NODE> node;
        Vec lo, hi, clo, chi;
        lo.fill(std::numeric_limits<double>::infinity());
        hi.fill(-std::numeric_limits<double>::infinity());
        clo = lo;
        chi = hi;
        for (std::size_t j = begin; j < end; j++) {
            for (std::size_t i = 0; i < 3; i++) {
                for (std::size_t k = 0; k < 3; k++) {
                    lo[i] = std::min(lo[i], points[tris[order[j]][k]][i]);
                    hi[i] = std::max(hi[i], points[tris[order[j]][k]][i]);
                }
                clo[i] = std::min(clo[i], centroid[order[j]][i]);
                chi[i] = std::max(chi[i], centroid[order[j]][i]);
            }
        }
        for (std::size_t i = 0; i < 3; i++) {
            node[i] = lo[i];
            node[3 + i] = hi[i];
        }
        if (end - begin <= triangles::LEAF) {
            node[6] = static_cast<double>(begin);
            node[7] = -static_cast<double>(end - begin);
        } else {
            std::size_t axis = 0;
            for (std::size_t i = 1; i < 3; i++) {
                if (chi[i] - clo[i] > chi[axis] - clo[axis]) axis = i;
            }
            // Sorted, ties by index, so that the hierarchy is the same with every standard library
            std::sort(order.begin() + static_cast<std::ptrdiff_t>(begin), order.begin() + static_cast<std::ptrdiff_t>(end),
                      [&](const uint32_t a, const uint32_t b) {
                          return centroid[a][axis] != centroid[b][axis] ? centroid[a][axis] < centroid[b][axis] : a < b;
                      });
            const std::size_t mid = begin + (end - begin) / 2;
            node[6] = static_cast<double>(self(self, begin, mid, depth + 1));
            node[7] = static_cast<double>(self(self, mid, end, depth + 1));
        }
        nodes[index] = node;
        return index;
    };
    build(build, 0, n, 0);
    if (deepest >= triangles::MAX_DEPTH) fail(where, "has too many triangles (" + logging::count(n) + ")", "");
    std::vector<double> out;
    out.reserve(triangles::HEADER + triangles::NODE * nodes.size() + triangles::TRIANGLE * n + 1);
    out.push_back(static_cast<double>(n));
    out.push_back(static_cast<double>(nodes.size()));
    for (const auto & node : nodes) out.insert(out.end(), node.begin(), node.end());
    for (const uint32_t t : order) {
        for (std::size_t k = 0; k < 3; k++) out.insert(out.end(), points[tris[t][k]].begin(), points[tris[t][k]].end());
        for (std::size_t k = 0; k < 3; k++) out.insert(out.end(), edge_normal[t][k].begin(), edge_normal[t][k].end());
        for (std::size_t k = 0; k < 3; k++) out.insert(out.end(), corner[tris[t][k]].begin(), corner[tris[t][k]].end());
    }
    if (out.size() % 2 != 0) out.push_back(0.0);
    return out;
}

} // namespace

std::vector<Triangle> read(const std::string & file, const std::string & format, const double factor,
                           const Where & where) {
    std::ifstream in(file, std::ios::binary);
    if (!in) {
        fail(where, "cannot open the surface file " + file,
             "Give its path relative to the directory the run starts in, or an absolute one");
    }
    provenance::read_file("surface", file);
    const std::string bytes((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    std::vector<Triangle> out = format == "stl" ? read_stl(bytes, file, where) : read_obj(bytes, file, where);
    if (out.empty()) fail(where, file + " holds no triangles", "");
    for (Triangle & t : out) {
        for (Vec & x : t) {
            for (double & v : x) {
                if (!std::isfinite(v)) fail(where, file + " has a vertex that is not a finite number", "");
                v *= factor;
            }
        }
    }
    return out;
}

Packed pack(const std::vector<Triangle> & soup, const std::string & file, const Where & where) {
    // One vertex per distinct position
    std::vector<Vec> corners;
    corners.reserve(3 * soup.size());
    for (const Triangle & t : soup) corners.insert(corners.end(), t.begin(), t.end());
    std::vector<uint32_t> by_position(corners.size());
    std::iota(by_position.begin(), by_position.end(), 0u);
    std::sort(by_position.begin(), by_position.end(), [&](const uint32_t a, const uint32_t b) {
        return corners[a] != corners[b] ? corners[a] < corners[b] : a < b;
    });
    std::vector<Vec> points;
    std::vector<uint32_t> vertex(corners.size());
    for (std::size_t j = 0; j < by_position.size(); j++) {
        if (j == 0 || corners[by_position[j]] != corners[by_position[j - 1]]) points.push_back(corners[by_position[j]]);
        vertex[by_position[j]] = static_cast<uint32_t>(points.size() - 1);
    }
    std::vector<Corners> tris;
    for (std::size_t t = 0; t < soup.size(); t++) {
        const Corners c = {vertex[3 * t], vertex[3 * t + 1], vertex[3 * t + 2]};
        if (c[0] != c[1] && c[1] != c[2] && c[2] != c[0]) tris.push_back(c);
    }
    if (tris.empty()) fail(where, file + " has no triangle of three distinct vertices", "");
    Vec lo = points[0], hi = points[0];
    for (const Vec & x : points) {
        for (std::size_t i = 0; i < 3; i++) {
            lo[i] = std::min(lo[i], x[i]);
            hi[i] = std::max(hi[i], x[i]);
        }
    }
    const double diagonal = norm(sub(hi, lo));

    // Closed: every edge in exactly two triangles
    const std::vector<Edge> edges = edges_of(tris);
    std::size_t open = 0, shared = 0;
    const Edge * open_example = nullptr;
    const Edge * shared_example = nullptr;
    for (std::size_t e = 0; e < edges.size();) {
        std::size_t f = e;
        while (f < edges.size() && edges[f].lo == edges[e].lo && edges[f].hi == edges[e].hi) f++;
        if (f - e == 1) {
            open++;
            if (!open_example) open_example = &edges[e];
        } else if (f - e > 2) {
            shared++;
            if (!shared_example) shared_example = &edges[e];
        }
        e = f;
    }
    const std::string surface_regions = "Open surfaces are not regions yet";
    if (open > 0) {
        fail(where,
             file + " is not closed: it has " + logging::count(open, "edge") + " of one triangle only (holes or gaps), "
                 "e.g. the edge from " + point(points[open_example->lo]) + " to " + point(points[open_example->hi]) +
                 ", and a filled region needs a closed (watertight) surface",
             "Close the holes in the CAD tool or a mesh repair tool, and export the surface again; vertices must match "
             "exactly across neighboring triangles. " + surface_regions);
    }
    if (shared > 0) {
        fail(where,
             file + " is not a manifold surface: it has " + logging::count(shared, "edge") + " of more than two "
                 "triangles, e.g. the edge from " + point(points[shared_example->lo]) + " to " +
                 point(points[shared_example->hi]) + " (bodies that touch along an edge, or triangles given twice)",
             "Remove duplicated triangles, or export each body to its own file and combine them with "
             "{type = \"union\", regions = [...]}");
    }
    std::vector<std::array<uint32_t, 3>> neighbor(tris.size());
    std::vector<std::array<bool, 3>> same(tris.size());  // the neighbor runs along the edge the same way
    Sets fans(3 * tris.size());
    const auto corner_of = [&](const uint32_t t, const uint32_t v) {
        for (uint32_t k = 0; k < 3; k++) {
            if (tris[t][k] == v) return 3 * t + k;
        }
        return 3 * t;
    };
    for (std::size_t e = 0; e < edges.size(); e += 2) {
        const Edge & a = edges[e];
        const Edge & b = edges[e + 1];
        neighbor[a.triangle][std::size_t{a.k}] = b.triangle;
        neighbor[b.triangle][std::size_t{b.k}] = a.triangle;
        same[a.triangle][std::size_t{a.k}] = same[b.triangle][std::size_t{b.k}] = a.forward == b.forward;
        fans.join(corner_of(a.triangle, a.lo), corner_of(b.triangle, a.lo));
        fans.join(corner_of(a.triangle, a.hi), corner_of(b.triangle, a.hi));
    }
    // Manifold at the corners too: each vertex's triangles one fan
    std::vector<uint32_t> fan_of(points.size(), std::numeric_limits<uint32_t>::max());
    for (uint32_t c = 0; c < 3 * tris.size(); c++) {
        const uint32_t v = tris[c / 3][c % 3], f = fans.find(c);
        if (fan_of[v] == std::numeric_limits<uint32_t>::max()) {
            fan_of[v] = f;
        } else if (fan_of[v] != f) {
            fail(where,
                 file + " is not a manifold surface: its triangles at the vertex " + point(points[v]) +
                     " do not form one fan (bodies, or parts of one, that touch at a point)",
                 "Separate the parts in the CAD tool, or export each body to its own file and combine them with "
                 "{type = \"union\", regions = [...]}");
        }
    }

    // One orientation per closed surface, spread across its edges
    const uint32_t none = std::numeric_limits<uint32_t>::max();
    std::vector<uint32_t> component(tris.size(), none);
    std::vector<bool> flip(tris.size(), false);
    uint32_t n_components = 0;
    for (uint32_t seed = 0; seed < tris.size(); seed++) {
        if (component[seed] != none) continue;
        std::vector<uint32_t> queue = {seed};
        component[seed] = n_components;
        for (std::size_t q = 0; q < queue.size(); q++) {
            const uint32_t t = queue[q];
            for (std::size_t k = 0; k < 3; k++) {
                const uint32_t m = neighbor[t][k];
                const bool want = flip[t] != same[t][k];
                if (component[m] == none) {
                    component[m] = n_components;
                    flip[m] = want;
                    queue.push_back(m);
                } else if (flip[m] != want) {
                    fail(where,
                         file + " is not orientable: its triangles near " + point(points[tris[t][k]]) +
                             " cannot all face one side (a one-sided surface, like a Moebius strip)",
                         "Repair the surface in the CAD tool or a mesh repair tool");
                }
            }
        }
        n_components++;
    }
    for (std::size_t t = 0; t < tris.size(); t++) {
        if (flip[t]) std::swap(tris[t][1], tris[t][2]);
    }
    // Each closed surface facing out of the volume it encloses
    std::vector<double> volume(n_components, 0.0);
    std::vector<std::array<Vec, 2>> boxes(n_components, std::array<Vec, 2>{points[tris[0][0]], points[tris[0][0]]});
    std::vector<bool> seen(n_components, false);
    for (std::size_t t = 0; t < tris.size(); t++) {
        const uint32_t c = component[t];
        volume[c] += dot(points[tris[t][0]], cross(points[tris[t][1]], points[tris[t][2]])) / 6.0;
        for (const uint32_t v : tris[t]) {
            if (!seen[c]) boxes[c] = std::array<Vec, 2>{points[v], points[v]};
            seen[c] = true;
            for (std::size_t i = 0; i < 3; i++) {
                boxes[c][0][i] = std::min(boxes[c][0][i], points[v][i]);
                boxes[c][1][i] = std::max(boxes[c][1][i], points[v][i]);
            }
        }
    }
    for (uint32_t c = 0; c < n_components; c++) {
        const double size = norm(sub(boxes[c][1], boxes[c][0]));
        if (!(std::abs(volume[c]) > 1e-12 * size * size * size)) {
            fail(where,
                 file + " has a closed surface that encloses no volume, around " + point(boxes[c][0]) +
                     " (a flat surface given on both sides)",
                 "Remove it from the file");
        }
    }
    for (std::size_t t = 0; t < tris.size(); t++) {
        if (volume[component[t]] < 0.0) std::swap(tris[t][1], tris[t][2]);
    }
    // Nesting: a closed surface inside an odd number of others bounds a cavity, and faces into it
    std::vector<uint32_t> depth(n_components, 0);
    if (n_components > 1) {
        std::vector<std::vector<Corners>> of(n_components);
        std::vector<std::vector<uint32_t>> vertices_of(n_components);
        for (std::size_t t = 0; t < tris.size(); t++) {
            of[component[t]].push_back(tris[t]);
            for (const uint32_t v : tris[t]) vertices_of[component[t]].push_back(v);
        }
        std::vector<std::vector<double>> laid(n_components);
        for (uint32_t c = 0; c < n_components; c++) {
            std::sort(vertices_of[c].begin(), vertices_of[c].end());
            vertices_of[c].erase(std::unique(vertices_of[c].begin(), vertices_of[c].end()), vertices_of[c].end());
            laid[c] = layout(points, of[c], where);
        }
        const double tolerance = 1e-10 * diagonal;
        for (uint32_t c = 0; c < n_components; c++) {
            for (uint32_t d = 0; d < n_components; d++) {
                bool overlap = c != d;
                for (std::size_t i = 0; i < 3 && overlap; i++) {
                    overlap = boxes[c][0][i] <= boxes[d][1][i] && boxes[d][0][i] <= boxes[c][1][i];
                }
                if (!overlap) continue;
                std::size_t inside = 0, outside = 0;
                uint32_t example = vertices_of[c][0];
                for (const uint32_t v : vertices_of[c]) {
                    const double phi = detail::triangulated(laid[d].data(), points[v].data());
                    if (phi < -tolerance) {
                        inside++;
                        example = v;
                    } else if (phi > tolerance) {
                        outside++;
                    } else {
                        inside = outside = 1;
                        example = v;
                        break;
                    }
                }
                if (inside > 0 && outside > 0) {
                    fail(where,
                         file + " has closed surfaces that cross or touch each other, at " + point(points[example]) +
                             ", so its inside is not defined",
                         "Merge the bodies into one in the CAD tool, or export each body to its own file and combine "
                         "them with {type = \"union\", regions = [...]}");
                }
                if (inside > 0) depth[c]++;
            }
        }
    }
    Packed out;
    for (uint32_t c = 0; c < n_components; c++) (depth[c] % 2 == 0 ? out.n_bodies : out.n_cavities)++;
    for (std::size_t t = 0; t < tris.size(); t++) {
        if (depth[component[t]] % 2 == 1) std::swap(tris[t][1], tris[t][2]);
    }
    out.n_triangles = static_cast<uint32_t>(tris.size());
    out.data = layout(points, tris, where);
    return out;
}

std::vector<Triangle> triangles(const double * s) {
    const uint32_t n = static_cast<uint32_t>(s[0]), n_nodes = static_cast<uint32_t>(s[1]);
    const double * t = s + triangles::HEADER + triangles::NODE * n_nodes;
    std::vector<Triangle> out(n);
    for (uint32_t k = 0; k < n; k++) {
        for (std::size_t c = 0; c < 3; c++) {
            for (std::size_t i = 0; i < 3; i++) out[k][c][i] = t[triangles::TRIANGLE * k + 3 * c + i];
        }
    }
    return out;
}

std::array<Vec, 2> box(const double * s) {
    const double * root = s + triangles::HEADER;
    return {Vec{root[0], root[1], root[2]}, Vec{root[3], root[4], root[5]}};
}

std::vector<Segment> section(const double * s, const double z0) {
    std::vector<Segment> out;
    for (const Triangle & t : triangles(s)) {
        Vec cut[2];
        int n = 0;
        for (std::size_t k = 0; k < 3; k++) {
            const Vec & a = t[k];
            const Vec & b = t[(k + 1) % 3];
            const bool a_above = a[2] >= z0, b_above = b[2] >= z0;
            if (a_above == b_above) continue;
            const double w = (z0 - a[2]) / (b[2] - a[2]);
            cut[n++] = {a[0] + w * (b[0] - a[0]), a[1] + w * (b[1] - a[1]), z0};
        }
        if (n != 2 || cut[0] == cut[1]) continue;
        // The solid on the left: along z x the outward normal
        const Vec normal = cross(sub(t[1], t[0]), sub(t[2], t[0]));
        if ((cut[1][0] - cut[0][0]) * -normal[1] + (cut[1][1] - cut[0][1]) * normal[0] < 0.0) std::swap(cut[0], cut[1]);
        out.push_back({cut[0], cut[1]});
    }
    return out;
}

} // namespace region::surface

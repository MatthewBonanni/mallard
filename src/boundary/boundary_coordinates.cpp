/**
 * @file boundary_coordinates.cpp
 * @author Matthew Bonanni (mbonanni001@gmail.com)
 * @brief Local coordinates of boundary faces.
 * @version 0.1
 * @date 2026-11-15
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 *
 */

#include "boundary_coordinates.h"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <limits>
#include <map>
#include <numeric>
#include <sstream>

#include "comm.h"
#include "kexpr.h"
#include "log.h"
#include "mesh.h"
#include "patch_coordinates.h"
#include "units.h"

namespace {

using Vertex = std::array<double, N_DIM>;

std::string real(const double v) {
    std::ostringstream s;
    s << std::setprecision(6) << v;
    return s.str();
}

/** @brief The local coordinates read by the strings of v, a value of a [[boundaries]] entry or below. */
void collect_used(const Input & v, std::vector<bool> & used) {
    if (v.is_string()) {
        try {
            const kexpr::Program p("", v.as_string(), BoundaryValue::variables(), nullptr, boundary_variable::FIRST_LOCAL);
            for (uint16_t i = boundary_variable::FIRST_LOCAL; i < boundary_variable::COUNT; i++) {
                if (p.uses(i)) used[i] = true;
            }
        } catch (const std::exception &) {
            // Not an expression: a quantity with a unit, a name or a file
        }
    } else if (v.is_array()) {
        for (const Input & item : v.as_array()) collect_used(item, used);
    } else if (v.is_table()) {
        for (const auto & [key, item] : v.as_table().entries()) {
            if (key != "corners" && key != "walls" && key != "where" && key != "name" && key != "type") {
                collect_used(item, used);
            }
        }
    }
}

#if N_DIM == 2
double segment_distance2(const Vertex & x, const Vertex & a, const Vertex & b) {
    double ab2 = 0.0, t = 0.0;
    for (size_t d = 0; d < N_DIM; d++) {
        ab2 += (b[d] - a[d]) * (b[d] - a[d]);
        t += (x[d] - a[d]) * (b[d] - a[d]);
    }
    t = ab2 > 0.0 ? std::clamp(t / ab2, 0.0, 1.0) : 0.0;
    double s = 0.0;
    for (size_t d = 0; d < N_DIM; d++) {
        const double e = x[d] - (a[d] + t * (b[d] - a[d]));
        s += e * e;
    }
    return s;
}
#else
/** @brief Squared distance from x to the triangle abc (Ericson, Real-Time Collision Detection, 5.1.5). */
double triangle_distance2(const Vertex & x, const Vertex & a, const Vertex & b, const Vertex & c) {
    auto sub = [](const Vertex & p, const Vertex & q) { return Vertex{p[0] - q[0], p[1] - q[1], p[2] - q[2]}; };
    auto dot3 = [](const Vertex & p, const Vertex & q) { return p[0] * q[0] + p[1] * q[1] + p[2] * q[2]; };
    auto distance2 = [&](const Vertex & q) {
        const Vertex e = sub(x, q);
        return dot3(e, e);
    };
    auto along = [&](const Vertex & p, const Vertex & dir, const double t) {
        return Vertex{p[0] + t * dir[0], p[1] + t * dir[1], p[2] + t * dir[2]};
    };
    const Vertex ab = sub(b, a), ac = sub(c, a), ap = sub(x, a);
    const double d1 = dot3(ab, ap), d2 = dot3(ac, ap);
    if (d1 <= 0.0 && d2 <= 0.0) return distance2(a);
    const Vertex bp = sub(x, b);
    const double d3 = dot3(ab, bp), d4 = dot3(ac, bp);
    if (d3 >= 0.0 && d4 <= d3) return distance2(b);
    const double vc = d1 * d4 - d3 * d2;
    if (vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0) return distance2(along(a, ab, d1 / (d1 - d3)));
    const Vertex cp = sub(x, c);
    const double d5 = dot3(ab, cp), d6 = dot3(ac, cp);
    if (d6 >= 0.0 && d5 <= d6) return distance2(c);
    const double vb = d5 * d2 - d1 * d6;
    if (vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0) return distance2(along(a, ac, d2 / (d2 - d6)));
    const double va = d3 * d6 - d5 * d4;
    if (va <= 0.0 && (d4 - d3) >= 0.0 && (d5 - d6) >= 0.0) {
        return distance2(along(b, sub(c, b), (d4 - d3) / ((d4 - d3) + (d5 - d6))));
    }
    const double denom = 1.0 / (va + vb + vc);
    const double v = vb * denom, w = vc * denom;
    const Vertex q{a[0] + ab[0] * v + ac[0] * w, a[1] + ab[1] * v + ac[1] * w, a[2] + ab[2] * v + ac[2] * w};
    return distance2(q);
}
#endif

Vertex node(const Mesh & mesh, const uint32_t n) {
    Vertex v{};
    for (size_t d = 0; d < N_DIM; d++) v[d] = static_cast<double>(mesh.h_node_coords(n, d));
    return v;
}

/** @brief Position of face f among the faces of its cell, keyed by the cell's global id: an order every partition agrees on. */
uint64_t face_key(const Mesh & mesh, const uint32_t f) {
    const uint32_t c = static_cast<uint32_t>(mesh.h_cells_of_face(f, 0));
    uint32_t j = 0;
    while (mesh.h_face_of_cell(c, j) != f) j++;
    return mesh.h_global_cell(c) * 8 + j;  // No cell has 8 faces
}

bool owned(const Mesh & mesh, const uint32_t f) {
    return static_cast<uint32_t>(mesh.h_cells_of_face(f, 0)) < mesh.n_owned();
}

/** @brief Every rank's values, per face of `keys`, sorted by key: values[k * width, (k + 1) * width) of face k. */
std::vector<double> gather_sorted(std::vector<uint64_t> keys, std::vector<double> values, const size_t width,
                                  const bool distributed) {
    if (distributed) {
        keys = comm::allgatherv(keys);
        values = comm::allgatherv(values);
    }
    std::vector<size_t> order(keys.size());
    std::iota(order.begin(), order.end(), size_t(0));
    std::sort(order.begin(), order.end(), [&](const size_t a, const size_t b) { return keys[a] < keys[b]; });
    std::vector<double> sorted;
    sorted.reserve(values.size());
    for (const size_t i : order) sorted.insert(sorted.end(), values.begin() + long(i * width), values.begin() + long((i + 1) * width));
    return sorted;
}

#if N_DIM == 2
std::string vector_text(const Point & v) {
    std::string text;
    for (size_t d = 0; d < N_DIM; d++) text += (d ? ", " : "(") + real(v[d]);
    return text + ")";
}

/** @brief Index of the largest |v[d]|, ties to the lower d. */
size_t dominant(const Point & v) {
    size_t k = 0;
    for (size_t d = 1; d < N_DIM; d++) {
        if (std::abs(v[d]) > std::abs(v[k])) k = d;
    }
    return k;
}

/**
 * @brief Arc length at the center of each of this rank's faces along the curve that all faces of the entry form,
 *        from its end with the lower coordinate along the dominant axis of the line between its ends.
 * @param length Set to the length of the curve.
 */
std::vector<double> arc_length(const Mesh & mesh, const std::vector<uint32_t> & faces, const Input & entry,
                               const std::string & where, const bool distributed, double & length) {
    using Ends = std::array<Vertex, 2>;
    auto ends_of = [&](const uint32_t f) {
        Ends e{node(mesh, mesh.h_node_of_face(f, 0)), node(mesh, mesh.h_node_of_face(f, 1))};
        if (e[1] < e[0]) std::swap(e[0], e[1]);
        return e;
    };
    std::vector<uint64_t> keys;
    std::vector<double> values;
    for (const uint32_t f : faces) {
        if (!owned(mesh, f)) continue;
        keys.push_back(face_key(mesh, f));
        for (const Vertex & v : ends_of(f)) values.insert(values.end(), v.begin(), v.end());
    }
    const std::vector<double> all = gather_sorted(keys, values, 4, distributed);
    const size_t n = all.size() / 4;
    std::vector<Ends> ends(n);
    std::map<Vertex, std::vector<size_t>> at_node;
    for (size_t k = 0; k < n; k++) {
        ends[k] = {Vertex{all[4 * k], all[4 * k + 1]}, Vertex{all[4 * k + 2], all[4 * k + 3]}};
        for (const Vertex & v : ends[k]) at_node[v].push_back(k);
    }
    auto fail = [&](const std::string & message) {
        return InputError({.key = where, .line = line_of(entry), .message = "u: " + message,
                           .fix = "u runs along one open curve: split the zone into entries with where = \"...\", or "
                                  "use n",
                           .docs = "docs/guide/boundary_conditions.md#local-coordinates"});
    };
    std::vector<Vertex> tips;
    for (const auto & [v, list] : at_node) {
        if (list.size() > 2) throw fail("more than two faces meet at the node " + vector_text(Point{v[0], v[1]}));
        if (list.size() == 1) tips.push_back(v);
    }
    if (tips.empty()) throw fail("the faces form a closed curve, which has no start");
    if (tips.size() > 2) throw fail("the faces form more than one curve");
    const size_t axis = dominant(Point{tips[1][0] - tips[0][0], tips[1][1] - tips[0][1]});
    Vertex at = tips[1][axis] < tips[0][axis] ? tips[1] : tips[0];
    std::map<Ends, double> s_of;
    std::vector<bool> visited(n, false);
    length = 0.0;
    for (size_t step = 0; step < n; step++) {
        size_t next = n;
        for (const size_t k : at_node[at]) {
            if (!visited[k]) next = k;
        }
        if (next == n) break;
        visited[next] = true;
        const Vertex other = ends[next][0] == at ? ends[next][1] : ends[next][0];
        const double l = std::hypot(other[0] - at[0], other[1] - at[1]);
        s_of[ends[next]] = length + 0.5 * l;
        length += l;
        at = other;
    }
    if (s_of.size() != n) throw fail("the faces form more than one curve");
    std::vector<double> s(faces.size());
    for (size_t k = 0; k < faces.size(); k++) s[k] = s_of.at(ends_of(faces[k]));
    return s;
}
#else
/** @brief The patch coordinates of each of this rank's faces, from all faces of the entry (patch_coordinates.h). */
patch::Coordinates patch_coordinates(const Mesh & mesh, const std::vector<uint32_t> & faces, const Input & entry,
                                     const std::string & where, const bool distributed) {
    constexpr size_t MAX_NODES = 4, W = 1 + 3 * MAX_NODES + 3;  // node count, nodes, inward area vector
    auto nodes_of = [&](const uint32_t f) {
        std::vector<Vertex> nodes(mesh.h_n_nodes_of_face(f));
        for (uint32_t j = 0; j < nodes.size(); j++) nodes[j] = node(mesh, mesh.h_node_of_face(f, j));
        return nodes;
    };
    std::vector<uint64_t> keys;
    std::vector<double> values;
    for (const uint32_t f : faces) {
        if (!owned(mesh, f)) continue;
        const std::vector<Vertex> nodes = nodes_of(f);
        if (nodes.size() > MAX_NODES) throw std::logic_error("patch coordinates: a boundary face has more than 4 nodes");
        keys.push_back(face_key(mesh, f));
        values.push_back(static_cast<double>(nodes.size()));
        for (size_t j = 0; j < MAX_NODES; j++) {
            for (size_t d = 0; d < 3; d++) values.push_back(j < nodes.size() ? nodes[j][d] : 0.0);
        }
        for (size_t d = 0; d < 3; d++) values.push_back(-static_cast<double>(mesh.h_face_normals(f, d)));
    }
    const std::vector<double> all = gather_sorted(keys, values, W, distributed);
    patch::Surface surface;
    std::map<std::vector<Vertex>, size_t> index;
    for (size_t k = 0; k < all.size(); k += W) {
        std::vector<Vertex> nodes(static_cast<size_t>(all[k]));
        for (size_t j = 0; j < nodes.size(); j++) nodes[j] = {all[k + 1 + 3 * j], all[k + 2 + 3 * j], all[k + 3 + 3 * j]};
        surface.faces.push_back(nodes);
        surface.inward.push_back({all[k + W - 3], all[k + W - 2], all[k + W - 1]});
        std::sort(nodes.begin(), nodes.end());
        index[nodes] = k / W;
    }
    patch::Corners corners{.points = std::nullopt, .seam = std::nullopt, .key = where, .line = line_of(entry)};
    auto point = [&](const Input & item, const std::string & key) {
        const std::vector<rtype> p = as_real_vector(item, key, dim::length);
        if (p.size() != 3) {
            throw InputError({.key = key, .line = line_of(entry, key.substr(where.size() + 1)),
                              .message = "a point has 3 components",
                              .docs = "docs/guide/boundary_conditions.md#local-coordinates"});
        }
        return patch::Vec3{static_cast<double>(p[0]), static_cast<double>(p[1]), static_cast<double>(p[2])};
    };
    if (entry.contains("seam")) corners.seam = point(entry.at("seam"), where + ".seam");
    if (entry.contains("corners")) {
        const std::string key = where + ".corners";
        corners.key = key;
        corners.line = line_of(entry, "corners");
        const Input & list = entry.at("corners");
        if (!list.is_array()) {
            throw InputError({.key = key, .line = corners.line, .message = "must be a list of points",
                              .fix = "Write corners = [[x, y, z], [x, y, z], [x, y, z], [x, y, z]]",
                              .docs = "docs/guide/boundary_conditions.md#local-coordinates"});
        }
        corners.points.emplace();
        for (const Input & item : list.as_array()) corners.points->push_back(point(item, key));
    }
    const patch::Coordinates global = patch::coordinates(surface, corners);
    patch::Coordinates local{global.shape, std::vector<patch::Values>(faces.size()), global.summary};
    for (size_t k = 0; k < faces.size(); k++) {
        std::vector<Vertex> nodes = nodes_of(faces[k]);
        std::sort(nodes.begin(), nodes.end());
        local.faces[k] = global.faces[index.at(nodes)];
    }
    return local;
}
#endif

} // namespace

std::vector<uint16_t> local_coordinates_used(const Input & entry) {
    std::vector<bool> used(boundary_variable::COUNT, false);
    collect_used(entry, used);
    std::vector<uint16_t> list;
    for (uint16_t i = boundary_variable::FIRST_LOCAL; i < boundary_variable::COUNT; i++) {
        if (used[i]) list.push_back(i);
    }
    return list;
}

WallDistance::WallDistance(const Mesh & mesh, const std::vector<uint32_t> & faces, const bool distributed) {
    // Per face: its node count, then its nodes
    std::vector<double> local;
    for (const uint32_t f : faces) {
        if (!owned(mesh, f)) continue;
        const uint32_t m = mesh.h_n_nodes_of_face(f);
        local.push_back(static_cast<double>(m));
        for (uint32_t j = 0; j < m; j++) {
            const Vertex v = node(mesh, mesh.h_node_of_face(f, j));
            local.insert(local.end(), v.begin(), v.end());
        }
    }
    const std::vector<double> all = distributed ? comm::allgatherv(local) : local;
    std::vector<Box> boxes;
    for (size_t i = 0; i < all.size();) {
        const size_t m = static_cast<size_t>(all[i++]);
        std::vector<Vertex> v(m);
        for (size_t j = 0; j < m; j++) {
            for (size_t d = 0; d < N_DIM; d++) v[j][d] = all[i++];
        }
#if N_DIM == 2
        std::array<Vertex, 2> segment{v[0], v[1]};
        std::sort(segment.begin(), segment.end());
        pieces.push_back(segment);
#else
        // Fan from the lowest node, each triangle's vertices sorted: the same pieces from any node order
        const size_t lowest = static_cast<size_t>(std::min_element(v.begin(), v.end()) - v.begin());
        std::rotate(v.begin(), v.begin() + long(lowest), v.end());
        for (size_t j = 1; j + 1 < m; j++) {
            std::array<Vertex, 3> triangle{v[0], v[j], v[j + 1]};
            std::sort(triangle.begin(), triangle.end());
            pieces.push_back(triangle);
        }
#endif
    }
    // The same pieces in the same order on every rank count, so that ties pick the same piece
    std::sort(pieces.begin(), pieces.end());
    for (const auto & piece : pieces) {
        Box b;
        for (const Vertex & v : piece) b.add(v.data());
        boxes.push_back(b);
    }
    tree = BoxTree(boxes);
}

double WallDistance::operator()(const Point & x) const {
    double d2 = 0.0;
    tree.nearest(x.data(), [&](const uint32_t i) {
        const auto & p = pieces[i];
#if N_DIM == 2
        return segment_distance2(x, p[0], p[1]);
#else
        return triangle_distance2(x, p[0], p[1], p[2]);
#endif
    }, d2);
    return std::sqrt(d2);
}

std::vector<FacePoint> local_coordinates(const Mesh & mesh, const std::vector<uint32_t> & faces, const Input & entry,
                                         const std::string & where, const std::vector<uint16_t> & used,
                                         const WallDistance & walls, const bool distributed, std::string & summary) {
    namespace bv = boundary_variable;
    std::vector<FacePoint> points(faces.size());
    for (size_t k = 0; k < faces.size(); k++) {
        Point x{};
        for (size_t d = 0; d < N_DIM; d++) x[d] = static_cast<double>(mesh.h_face_coords(faces[k], d));
        points[k] = FacePoint(x);
    }
    summary.clear();
    if (used.empty()) return points;
    auto uses = [&](const uint16_t i) { return std::find(used.begin(), used.end(), i) != used.end(); };
    if (uses(bv::N) || uses(bv::N_M)) {
        if (walls.empty()) {
            throw InputError({.key = where, .line = line_of(entry),
                              .message = "an expression reads n, the distance to the nearest wall, but the mesh has "
                                         "no wall faces to measure it from",
                              .fix = "Give the zones to measure from in walls = [\"...\"], or give some zone a wall "
                                     "condition",
                              .docs = "docs/guide/boundary_conditions.md#local-coordinates"});
        }
        // n is n_m over the largest distance on the boundary, at its face centers and nodes
        double farthest = 0.0, scale = 0.0;
        for (size_t k = 0; k < faces.size(); k++) {
            points[k][bv::N_M] = walls(points[k].point());
            farthest = std::max(farthest, points[k][bv::N_M]);
            for (uint32_t j = 0; j < mesh.h_n_nodes_of_face(faces[k]); j++) {
                const Vertex v = node(mesh, mesh.h_node_of_face(faces[k], j));
                Point x{};
                for (size_t d = 0; d < N_DIM; d++) x[d] = v[d];
                farthest = std::max(farthest, walls(x));
            }
            for (size_t d = 0; d < N_DIM; d++) scale = std::max(scale, std::abs(points[k][d]));
        }
        if (distributed) {
            farthest = comm::allreduce(farthest, comm::Op::MAX);
            scale = comm::allreduce(scale, comm::Op::MAX);
        }
        // Faces on the walls come out at round-off, not exactly 0
        const double round_off = (sizeof(rtype) == sizeof(double) ? 1e-10 : 1e-5) * std::max(scale, 1e-300);
        if (!(farthest > round_off)) {
            throw InputError({.key = where, .line = line_of(entry),
                              .message = "n, the distance to the nearest wall, is 0 on every face: the faces lie on "
                                         "the walls it measures from (by default every wall entry, this one too if "
                                         "it is a wall)",
                              .fix = "Name the zones to measure from in walls = [\"...\"]",
                              .docs = "docs/guide/boundary_conditions.md#local-coordinates"});
        }
        for (FacePoint & p : points) p[bv::N] = p[bv::N_M] / farthest;
    }
    bool along = false;
    for (const uint16_t i : used) along = along || (i != bv::N && i != bv::N_M);
    if (along) {
#if N_DIM == 2
        double length = 0.0;
        const std::vector<double> s = arc_length(mesh, faces, entry, where, distributed, length);
        for (size_t k = 0; k < faces.size(); k++) {
            points[k][bv::U_M] = s[k];
            points[k][bv::U] = s[k] / length;
        }
#else
        const patch::Coordinates c = patch_coordinates(mesh, faces, entry, where, distributed);
        summary = c.summary;
        const bool round = c.shape != patch::Shape::FOUR_SIDED;
        for (const uint16_t i : {bv::R, bv::R_M, bv::THETA}) {
            if (!round && uses(i)) {
                throw InputError({.key = where, .line = line_of(entry),
                                  .message = "an expression reads " + BoundaryValue::variables()[i] + ", which rings "
                                             "and disks have, but this is a " + c.summary,
                                  .fix = "Write the condition in u and v, or in the wall distance n",
                                  .docs = "docs/guide/boundary_conditions.md#local-coordinates"});
            }
        }
        for (size_t k = 0; k < faces.size(); k++) {
            const patch::Values & f = c.faces[k];
            FacePoint & p = points[k];
            p[bv::U] = f.u;
            p[bv::U_M] = f.u_m;
            p[bv::V] = f.v;
            p[bv::V_M] = f.v_m;
            if (round) {
                p[bv::R] = f.v;
                p[bv::R_M] = f.v_m;
                p[bv::THETA] = 2.0 * 3.14159265358979323846 * f.u;
            }
        }
#endif
    }
    for (const uint16_t i : used) {
        double lo = std::numeric_limits<double>::infinity(), hi = -lo;
        for (const FacePoint & p : points) {
            lo = std::min(lo, p[i]);
            hi = std::max(hi, p[i]);
        }
        if (distributed) {
            lo = comm::allreduce(lo, comm::Op::MIN);
            hi = comm::allreduce(hi, comm::Op::MAX);
        }
        summary += (summary.empty() ? "" : "; ") + BoundaryValue::variables()[i] + " " + real(lo) + " to " + real(hi);
    }
    return points;
}

/**
 * @file patch_coordinates.cpp
 * @author Matthew Bonanni (mbonanni001@gmail.com)
 * @brief Coordinates u, v of a boundary patch that follow its shape.
 * @version 0.1
 * @date 2026-11-17
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 *
 */

#include "patch_coordinates.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <limits>
#include <numeric>
#include <queue>
#include <sstream>
#include <utility>

#include "input.h"

namespace patch {

namespace {

constexpr double TWO_PI = 6.283185307179586476925286766559;
constexpr double CORNER_ANGLE = TWO_PI / 12.0;  // an outline turning by more than 30 degrees at a node has a corner there
constexpr uint32_t NONE = std::numeric_limits<uint32_t>::max();
const char * const DOCS = "docs/guide/boundary_conditions.md#local-coordinates";
const char * const SPLIT = "Split the entry into patches with where = \"...\", one entry each";

Vec3 sub(const Vec3 & a, const Vec3 & b) { return {a[0] - b[0], a[1] - b[1], a[2] - b[2]}; }
Vec3 axpy(const Vec3 & a, const double s, const Vec3 & b) { return {a[0] + s * b[0], a[1] + s * b[1], a[2] + s * b[2]}; }
double dot(const Vec3 & a, const Vec3 & b) { return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]; }
double length(const Vec3 & a) { return std::sqrt(dot(a, a)); }
Vec3 cross(const Vec3 & a, const Vec3 & b) {
    return {a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]};
}
Vec3 unit(const Vec3 & a) {
    const double l = length(a);
    return l > 0.0 ? Vec3{a[0] / l, a[1] / l, a[2] / l} : Vec3{};
}

/** @brief A point or a direction for messages, without the round-off of components that are zero. */
std::string text(const Vec3 & v) {
    const double small = 1e-9 * std::max({std::abs(v[0]), std::abs(v[1]), std::abs(v[2])});
    std::ostringstream s;
    s << std::setprecision(6);
    for (size_t d = 0; d < 3; d++) s << (d ? ", " : "(") << (std::abs(v[d]) < small ? 0.0 : v[d]);
    return s.str() + ")";
}

[[noreturn]] void fail(const Corners & at, const std::string & message, const std::string & fix) {
    throw InputError({.key = at.key, .line = at.line, .message = "u, v: " + message, .fix = fix, .docs = DOCS});
}

/** @brief Index of the largest |v[d]|, ties to the lower d. */
size_t dominant(const Vec3 & v) {
    size_t k = 0;
    for (size_t d = 1; d < 3; d++) {
        if (std::abs(v[d]) > std::abs(v[k])) k = d;
    }
    return k;
}

/**
 * @brief The directions u and v grow along where the patch leaves a choice,
 *        across a unit normal: the coordinate axis after the normal's
 *        dominant one (x to y, y to z, z to x), and the one after that.
 */
struct Reference {
    Vec3 first, second;

    explicit Reference(const Vec3 & normal) {
        Vec3 axis{};
        axis[(dominant(normal) + 1) % 3] = 1.0;
        first = unit(axpy(axis, -dot(axis, normal), normal));
        second = cross(normal, first);
        if (second[dominant(second)] < 0.0) second = {-second[0], -second[1], -second[2]};
    }
};

/** @brief The patch as triangles that turn counterclockwise seen from the domain, and the loops of its outline. */
struct Triangulation {
    std::vector<Vec3> x;                       // nodes, sorted
    std::vector<std::array<uint32_t, 3>> tri;
    std::vector<uint32_t> first;               // face f has the triangles [first[f], first[f + 1])
    std::vector<double> area;                  // of each triangle
    std::vector<Vec3> normal;                  // of each triangle, unit
    std::vector<std::array<uint32_t, 3>> edge; // of each triangle, opposite each of its nodes
    std::vector<std::array<uint32_t, 2>> edge_tri;
    std::vector<uint32_t> node_first, node_tri;
    std::vector<std::vector<uint32_t>> loops;  // nodes around each, the patch on the left seen from the domain
    double total_area = 0.0;
    Vec3 center{}, mean_normal{};

    size_t n_faces() const { return first.size() - 1; }

    Vec3 centroid(const uint32_t t) const {
        Vec3 c{};
        for (const uint32_t i : tri[t]) c = axpy(c, 1.0 / 3.0, x[i]);
        return c;
    }

    Vec3 face_center(const size_t f) const {
        Vec3 c{};
        double a = 0.0;
        for (uint32_t t = first[f]; t < first[f + 1]; t++) {
            c = axpy(c, area[t], centroid(t));
            a += area[t];
        }
        return {c[0] / a, c[1] / a, c[2] / a};
    }

    /** @brief A nodal field at the center of face f. */
    double at_center(const size_t f, const std::vector<double> & g) const {
        double sum = 0.0, a = 0.0;
        for (uint32_t t = first[f]; t < first[f + 1]; t++) {
            sum += area[t] * (g[tri[t][0]] + g[tri[t][1]] + g[tri[t][2]]) / 3.0;
            a += area[t];
        }
        return sum / a;
    }

    double loop_length(const std::vector<uint32_t> & loop) const {
        double l = 0.0;
        for (size_t k = 0; k < loop.size(); k++) l += length(sub(x[loop[(k + 1) % loop.size()]], x[loop[k]]));
        return l;
    }

    /** @brief The center of a chain of nodes by length (closed: the last node joins the first). */
    Vec3 chain_center(const std::vector<uint32_t> & chain, const bool closed) const {
        Vec3 c{};
        double l = 0.0;
        for (size_t k = 0; k + (closed ? 0 : 1) < chain.size(); k++) {
            const Vec3 & a = x[chain[k]], & b = x[chain[(k + 1) % chain.size()]];
            const double e = length(sub(b, a));
            c = axpy(c, 0.5 * e, a);
            c = axpy(c, 0.5 * e, b);
            l += e;
        }
        return l > 0.0 ? Vec3{c[0] / l, c[1] / l, c[2] / l} : x[chain[0]];
    }
};

Triangulation triangulate(const Surface & surface, const Corners & at) {
    Triangulation m;
    for (const auto & face : surface.faces) m.x.insert(m.x.end(), face.begin(), face.end());
    std::sort(m.x.begin(), m.x.end());
    m.x.erase(std::unique(m.x.begin(), m.x.end()), m.x.end());
    if (surface.faces.empty()) fail(at, "the entry has no faces", "");
    auto id = [&](const Vec3 & v) {
        return static_cast<uint32_t>(std::lower_bound(m.x.begin(), m.x.end(), v) - m.x.begin());
    };

    struct Side {
        uint32_t a, b;  // a < b
        bool forward;   // the face runs a to b
        bool operator<(const Side & o) const { return a != o.a ? a < o.a : b < o.b; }
    };
    std::vector<Side> sides;
    m.first.push_back(0);
    for (size_t f = 0; f < surface.faces.size(); f++) {
        std::vector<uint32_t> p;
        for (const Vec3 & v : surface.faces[f]) p.push_back(id(v));
        Vec3 newell{};
        for (size_t k = 0; k < p.size(); k++) newell = axpy(newell, 1.0, cross(m.x[p[k]], m.x[p[(k + 1) % p.size()]]));
        if (dot(newell, surface.inward[f]) < 0.0) std::reverse(p.begin(), p.end());
        std::rotate(p.begin(), std::min_element(p.begin(), p.end()), p.end());
        for (size_t k = 0; k < p.size(); k++) {
            const uint32_t a = p[k], b = p[(k + 1) % p.size()];
            sides.push_back({std::min(a, b), std::max(a, b), a < b});
        }
        for (size_t k = 1; k + 1 < p.size(); k++) {
            const std::array<uint32_t, 3> t{p[0], p[k], p[k + 1]};
            const Vec3 c = cross(sub(m.x[t[1]], m.x[t[0]]), sub(m.x[t[2]], m.x[t[0]]));
            m.tri.push_back(t);
            m.area.push_back(0.5 * length(c));
            m.normal.push_back(unit(c));
            m.total_area += m.area.back();
            m.center = axpy(m.center, m.area.back(), m.centroid(static_cast<uint32_t>(m.tri.size() - 1)));
            m.mean_normal = axpy(m.mean_normal, 0.5, c);
        }
        m.first.push_back(static_cast<uint32_t>(m.tri.size()));
    }
    if (!(m.total_area > 0.0)) fail(at, "the entry's faces have no area", "");
    for (double & c : m.center) c /= m.total_area;

    // The outline: sides that one face has
    std::sort(sides.begin(), sides.end());
    std::vector<uint32_t> next(m.x.size(), NONE), group(m.x.size());
    std::iota(group.begin(), group.end(), uint32_t(0));
    auto root = [&](uint32_t i) {
        while (group[i] != i) i = group[i] = group[group[i]];
        return i;
    };
    size_t n_sides = 0;
    for (size_t k = 0; k < sides.size();) {
        size_t end = k;
        while (end < sides.size() && !(sides[k] < sides[end])) end++;
        const Side & s = sides[k];
        if (end - k > 2) {
            fail(at, "more than two faces share the edge at " + text(m.x[s.a]) + ", so the faces are not one surface",
                 SPLIT);
        }
        if (end - k == 1) {
            const uint32_t from = s.forward ? s.a : s.b, to = s.forward ? s.b : s.a;
            if (next[from] != NONE) {
                fail(at, "the patch touches itself at the node " + text(m.x[from]), SPLIT);
            }
            next[from] = to;
        }
        group[root(s.a)] = root(s.b);
        n_sides++;
        k = end;
    }
    size_t pieces = 0;
    for (uint32_t i = 0; i < m.x.size(); i++) pieces += root(i) == i;
    if (pieces > 1) {
        fail(at, "the entry's faces form " + std::to_string(pieces) + " separate pieces, and u, v belong to one patch",
             SPLIT);
    }
    std::vector<bool> seen(m.x.size(), false);
    for (uint32_t i = 0; i < m.x.size(); i++) {
        if (next[i] == NONE || seen[i]) continue;
        std::vector<uint32_t> loop;
        for (uint32_t j = i; !seen[j]; j = next[j]) {
            seen[j] = true;
            loop.push_back(j);
            if (next[j] == NONE) fail(at, "the patch touches itself at the node " + text(m.x[j]), SPLIT);
        }
        m.loops.push_back(loop);
    }
    if (m.loops.empty()) {
        fail(at, "the entry's faces form a closed surface, which has no edge to measure u, v from", SPLIT);
    }
    const long euler = static_cast<long>(m.x.size()) - static_cast<long>(n_sides) + static_cast<long>(m.n_faces());
    if (m.loops.size() > 2 || euler != 2 - static_cast<long>(m.loops.size())) {
        fail(at, "the patch's outline has " + std::to_string(m.loops.size()) + " loops" +
                     (m.loops.size() > 2 ? "" : " and the patch has handles") +
                     "; u, v exist on a patch with one loop (four-sided, or a disk) or two (a ring)",
             SPLIT);
    }

    // Edges of the triangles, and the triangles around each node
    struct Edge {
        uint32_t a, b, t, k;
        bool operator<(const Edge & o) const { return a != o.a ? a < o.a : b != o.b ? b < o.b : t < o.t; }
    };
    std::vector<Edge> edges;
    m.node_first.assign(m.x.size() + 1, 0);
    for (uint32_t t = 0; t < m.tri.size(); t++) {
        for (uint32_t k = 0; k < 3; k++) {
            const uint32_t a = m.tri[t][(k + 1) % 3], b = m.tri[t][(k + 2) % 3];
            edges.push_back({std::min(a, b), std::max(a, b), t, k});
            m.node_first[m.tri[t][k] + 1]++;
        }
    }
    std::partial_sum(m.node_first.begin(), m.node_first.end(), m.node_first.begin());
    m.node_tri.resize(3 * m.tri.size());
    std::vector<uint32_t> fill(m.node_first.begin(), m.node_first.end() - 1);
    for (uint32_t t = 0; t < m.tri.size(); t++) {
        for (const uint32_t i : m.tri[t]) m.node_tri[fill[i]++] = t;
    }
    std::sort(edges.begin(), edges.end());
    m.edge.resize(m.tri.size());
    for (size_t k = 0; k < edges.size(); k++) {
        const bool same = k > 0 && edges[k].a == edges[k - 1].a && edges[k].b == edges[k - 1].b;
        if (!same) m.edge_tri.push_back({edges[k].t, NONE});
        else m.edge_tri.back()[1] = edges[k].t;
        m.edge[edges[k].t][edges[k].k] = static_cast<uint32_t>(m.edge_tri.size() - 1);
    }
    return m;
}

/** @brief The Laplacian of the surface on piecewise linear functions (cotangent weights). */
struct Laplacian {
    std::vector<uint32_t> first, column;
    std::vector<double> value, diagonal;

    explicit Laplacian(const Triangulation & m) : diagonal(m.x.size(), 0.0) {
        std::vector<std::vector<std::pair<uint32_t, double>>> rows(m.x.size());
        for (uint32_t t = 0; t < m.tri.size(); t++) {
            if (!(m.area[t] > 0.0)) continue;
            for (size_t k = 0; k < 3; k++) {
                const uint32_t i = m.tri[t][k], j = m.tri[t][(k + 1) % 3], l = m.tri[t][(k + 2) % 3];
                const double w = 0.25 * dot(sub(m.x[j], m.x[i]), sub(m.x[l], m.x[i])) / m.area[t];
                rows[j].emplace_back(l, -w);
                rows[l].emplace_back(j, -w);
                diagonal[j] += w;
                diagonal[l] += w;
            }
        }
        first.push_back(0);
        for (auto & row : rows) {
            std::sort(row.begin(), row.end());
            for (size_t k = 0; k < row.size(); k++) {
                if (k > 0 && row[k].first == row[k - 1].first) {
                    value.back() += row[k].second;
                } else {
                    column.push_back(row[k].first);
                    value.push_back(row[k].second);
                }
            }
            first.push_back(static_cast<uint32_t>(column.size()));
        }
    }

    double row(const size_t i, const std::vector<double> & x) const {
        double s = diagonal[i] * x[i];
        for (uint32_t k = first[i]; k < first[i + 1]; k++) s += value[k] * x[column[k]];
        return s;
    }

    /**
     * @brief The function with the values of x at the fixed nodes and K x = load at the others
     *        (conjugate gradients, the same operations in the same order on every rank).
     */
    std::vector<double> solve(const std::vector<bool> & fixed, std::vector<double> x, const std::vector<double> & load,
                              const Corners & at) const {
        const size_t n = x.size();
        std::vector<double> r(n, 0.0), z(n, 0.0), p(n, 0.0), q(n, 0.0);
        std::vector<uint32_t> free;
        for (uint32_t i = 0; i < n; i++) {
            if (!fixed[i] && diagonal[i] > 0.0) free.push_back(i);
        }
        double b2 = 0.0, rz = 0.0;
        for (const uint32_t i : free) {
            r[i] = load[i] - row(i, x);
            b2 += r[i] * r[i];
            p[i] = z[i] = r[i] / diagonal[i];
            rz += r[i] * z[i];
        }
        double r2 = b2;
        for (size_t it = 0; it < 10 * n + 1000 && r2 > 1e-28 * b2; it++) {
            double pq = 0.0;
            for (const uint32_t i : free) {
                q[i] = row(i, p);
                pq += p[i] * q[i];
            }
            if (!(pq > 0.0)) break;
            const double alpha = rz / pq;
            double rz_new = 0.0;
            r2 = 0.0;
            for (const uint32_t i : free) {
                x[i] += alpha * p[i];
                r[i] -= alpha * q[i];
                r2 += r[i] * r[i];
                z[i] = r[i] / diagonal[i];
                rz_new += r[i] * z[i];
            }
            const double beta = rz_new / rz;
            rz = rz_new;
            for (const uint32_t i : free) p[i] = z[i] + beta * p[i];
        }
        if (!(r2 <= 1e-16 * b2)) {
            fail(at, "Laplace's equation on the patch did not converge; its faces may be degenerate",
                 "Check the mesh of this zone (mallard check), or write the condition in x, y, z");
        }
        return x;
    }
};

/** @brief The distance of C along a front that passes A at da and B at db, or along the triangle's edges. */
double front(const Vec3 & A, const Vec3 & B, const Vec3 & C, const double da, const double db) {
    const Vec3 ca = sub(A, C), cb = sub(B, C);
    const double edge = std::min(da + length(ca), db + length(cb));
    if (!std::isfinite(da) || !std::isfinite(db)) return edge;
    const double g00 = dot(ca, ca), g01 = dot(ca, cb), g11 = dot(cb, cb);
    const double det = g00 * g11 - g01 * g01;
    if (!(det > 1e-28 * g00 * g11)) return edge;
    const double q00 = g11 / det, q01 = -g01 / det, q11 = g00 / det;
    const double s11 = q00 + 2.0 * q01 + q11, s1d = (q00 + q01) * da + (q01 + q11) * db;
    const double sdd = q00 * da * da + 2.0 * q01 * da * db + q11 * db * db;
    const double disc = s1d * s1d - s11 * (sdd - 1.0);
    if (!(disc >= 0.0) || !(s11 > 0.0)) return edge;
    const double t = (s1d + std::sqrt(disc)) / s11;
    // The front reaches C from inside the triangle
    const double w0 = q00 * (da - t) + q01 * (db - t), w1 = q01 * (da - t) + q11 * (db - t);
    return w0 <= 0.0 && w1 <= 0.0 && t >= std::max(da, db) ? std::min(edge, t) : edge;
}

/** @brief The distance along the surface from the nodes where d is finite (fast marching). */
std::vector<double> distance(const Triangulation & m, std::vector<double> d) {
    using Item = std::pair<double, uint32_t>;
    std::priority_queue<Item, std::vector<Item>, std::greater<Item>> queue;
    for (uint32_t i = 0; i < d.size(); i++) {
        if (std::isfinite(d[i])) queue.emplace(d[i], i);
    }
    while (!queue.empty()) {
        const auto [da, a] = queue.top();
        queue.pop();
        if (da > d[a]) continue;
        for (uint32_t k = m.node_first[a]; k < m.node_first[a + 1]; k++) {
            const auto & t = m.tri[m.node_tri[k]];
            for (size_t j = 0; j < 3; j++) {
                const uint32_t c = t[j], b = t[(j + 1) % 3] == a ? t[(j + 2) % 3] : t[(j + 1) % 3];
                if (c == a) continue;
                const double candidate = front(m.x[a], m.x[b], m.x[c], da, d[b]);
                if (candidate < d[c] - 1e-13 * candidate) {
                    d[c] = candidate;
                    queue.emplace(candidate, c);
                }
            }
        }
    }
    return d;
}

/**
 * @brief The harmonic conjugate of the nodal function g at the centroid of
 *        each triangle, up to multiples of its period around a hole of the
 *        surface (or around the triangle `pole`, where g has its source).
 */
std::vector<double> conjugate(const Triangulation & m, const std::vector<double> & g, const uint32_t pole) {
    const double unset = std::numeric_limits<double>::quiet_NaN();
    std::vector<double> at_edge(m.edge_tri.size(), unset), at_tri(m.tri.size(), unset);
    // The conjugate is linear on a triangle, continuous at the middle of its edges
    auto place = [&](const uint32_t t, const bool spread) {
        const auto & n = m.tri[t];
        Vec3 gradient{};
        for (size_t k = 0; k < 3; k++) {
            const Vec3 opposite = sub(m.x[n[(k + 2) % 3]], m.x[n[(k + 1) % 3]]);
            if (m.area[t] > 0.0) gradient = axpy(gradient, g[n[k]] / (2.0 * m.area[t]), cross(m.normal[t], opposite));
        }
        const Vec3 turned = cross(m.normal[t], gradient), c = m.centroid(t);
        std::array<double, 3> offset{};
        double center = 0.0;
        bool known = false;
        for (size_t k = 0; k < 3; k++) {
            offset[k] = 0.5 * dot(turned, sub(c, m.x[n[k]]));
            if (!known && !std::isnan(at_edge[m.edge[t][k]])) {
                center = at_edge[m.edge[t][k]] - offset[k];
                known = true;
            }
        }
        at_tri[t] = center;
        for (size_t k = 0; spread && k < 3; k++) {
            if (std::isnan(at_edge[m.edge[t][k]])) at_edge[m.edge[t][k]] = center + offset[k];
        }
    };
    std::vector<uint32_t> queue{pole == 0 && m.tri.size() > 1 ? 1u : 0u};
    std::vector<bool> seen(m.tri.size(), false);
    seen[queue[0]] = true;
    place(queue[0], true);
    for (size_t head = 0; head < queue.size(); head++) {
        const uint32_t t = queue[head];
        for (const uint32_t e : m.edge[t]) {
            for (const uint32_t other : m.edge_tri[e]) {
                if (other == NONE || other == pole || seen[other]) continue;
                seen[other] = true;
                place(other, true);
                queue.push_back(other);
            }
        }
    }
    if (pole != NONE && !seen[pole]) place(pole, false);
    return at_tri;
}

/** @brief A triangle's point nearest to p (Ericson, Real-Time Collision Detection, 5.1.5): its weights of the nodes. */
std::array<double, 3> nearest_weights(const Vec3 & p, const Vec3 & a, const Vec3 & b, const Vec3 & c) {
    const Vec3 ab = sub(b, a), ac = sub(c, a), ap = sub(p, a);
    const double d1 = dot(ab, ap), d2 = dot(ac, ap);
    if (d1 <= 0.0 && d2 <= 0.0) return {1.0, 0.0, 0.0};
    const Vec3 bp = sub(p, b);
    const double d3 = dot(ab, bp), d4 = dot(ac, bp);
    if (d3 >= 0.0 && d4 <= d3) return {0.0, 1.0, 0.0};
    const double vc = d1 * d4 - d3 * d2;
    if (vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0) {
        const double v = d1 / (d1 - d3);
        return {1.0 - v, v, 0.0};
    }
    const Vec3 cp = sub(p, c);
    const double d5 = dot(ab, cp), d6 = dot(ac, cp);
    if (d6 >= 0.0 && d5 <= d6) return {0.0, 0.0, 1.0};
    const double vb = d5 * d2 - d1 * d6;
    if (vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0) {
        const double w = d2 / (d2 - d6);
        return {1.0 - w, 0.0, w};
    }
    const double va = d3 * d6 - d5 * d4;
    if (va <= 0.0 && (d4 - d3) >= 0.0 && (d5 - d6) >= 0.0) {
        const double w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
        return {0.0, 1.0 - w, w};
    }
    const double v = vb / (va + vb + vc), w = vc / (va + vb + vc);
    return {1.0 - v - w, v, w};
}

/**
 * @brief u of the faces of a ring or a disk from the conjugate of its v: in
 *        [0, 1) around the patch, counterclockwise from `first` toward
 *        `second` about the patch's center on average.
 */
std::vector<double> around(const Triangulation & m, const std::vector<double> & at_tri, const double period,
                           const Reference & reference) {
    std::vector<double> raw(m.n_faces());
    std::array<double, 2> re{}, im{};
    for (size_t f = 0; f < m.n_faces(); f++) {
        const double base = at_tri[m.first[f]];
        double sum = 0.0, a = 0.0;
        for (uint32_t t = m.first[f]; t < m.first[f + 1]; t++) {
            const double d = at_tri[t] - base;
            sum += m.area[t] * (d - period * std::round(d / period));
            a += m.area[t];
        }
        raw[f] = (base + sum / a) / period;
        const Vec3 offset = sub(m.face_center(f), m.center);
        const double theta = std::atan2(dot(offset, reference.second), dot(offset, reference.first));
        for (size_t s = 0; s < 2; s++) {
            const double phase = (s == 0 ? TWO_PI : -TWO_PI) * raw[f] - theta;
            re[s] += a * std::cos(phase);
            im[s] += a * std::sin(phase);
        }
    }
    const size_t s = std::hypot(re[0], im[0]) >= std::hypot(re[1], im[1]) ? 0 : 1;
    const double shift = std::atan2(im[s], re[s]) / TWO_PI;
    for (double & u : raw) {
        u = (s == 0 ? u : -u) - shift;
        u -= std::floor(u);
        if (!(u < 1.0)) u = 0.0;
    }
    return raw;
}

/** @brief Position on the loop of its node nearest to p. */
size_t nearest_on_loop(const Triangulation & m, const std::vector<uint32_t> & loop, const Vec3 & p) {
    size_t best = 0;
    for (size_t k = 1; k < loop.size(); k++) {
        if (length(sub(m.x[loop[k]], p)) < length(sub(m.x[loop[best]], p))) best = k;
    }
    return best;
}

/** @brief The nodes of the loop from position a to b, forward (step 1) or backward (step n - 1). */
std::vector<uint32_t> side(const std::vector<uint32_t> & loop, const size_t a, const size_t b, const size_t step) {
    std::vector<uint32_t> nodes{loop[a]};
    for (size_t k = a; k != b;) {
        k = (k + step) % loop.size();
        nodes.push_back(loop[k]);
    }
    return nodes;
}

Coordinates four_sided(const Triangulation & m, const Laplacian & K, const Corners & at,
                       const std::vector<size_t> & found) {
    const std::vector<uint32_t> & loop = m.loops[0];
    const size_t n = loop.size();
    // sides[0] has v = 0 and u from 0 to 1; sides[1] u = 1; sides[2] v = 1; sides[3] u = 0
    std::array<std::vector<uint32_t>, 4> sides;
    if (at.points) {
        std::array<size_t, 4> q{};
        for (size_t i = 0; i < 4; i++) q[i] = nearest_on_loop(m, loop, (*at.points)[i]);
        std::array<size_t, 4> ahead{};
        for (size_t i = 0; i < 4; i++) ahead[i] = (q[i] + n - q[0]) % n;
        const bool forward = 0 < ahead[1] && ahead[1] < ahead[2] && ahead[2] < ahead[3];
        const bool backward = ahead[1] > ahead[2] && ahead[2] > ahead[3] && ahead[3] > 0;
        if (!forward && !backward) {
            fail(at, "corners must be four different points in order around the patch's outline; they are nearest "
                     "to its nodes " + text(m.x[loop[q[0]]]) + ", " + text(m.x[loop[q[1]]]) + ", " +
                     text(m.x[loop[q[2]]]) + " and " + text(m.x[loop[q[3]]]),
                 "List the corners where (u, v) is (0, 0), (1, 0), (1, 1) and (0, 1), in that order");
        }
        for (size_t i = 0; i < 4; i++) sides[i] = side(loop, q[i], q[(i + 1) % 4], forward ? 1 : n - 1);
    } else {
        // u grows along the reference's first direction and v along its second, as nearly as the sides allow
        const Reference reference(unit(m.mean_normal));
        std::array<std::vector<uint32_t>, 4> segment;
        std::array<Vec3, 4> center;
        for (size_t i = 0; i < 4; i++) {
            segment[i] = side(loop, found[i], found[(i + 1) % 4], 1);
            center[i] = m.chain_center(segment[i], false);
        }
        double best = -std::numeric_limits<double>::infinity();
        for (size_t start = 0; start < 4; start++) {
            for (const bool forward : {true, false}) {
                std::array<size_t, 4> order{};
                for (size_t i = 0; i < 4; i++) order[i] = forward ? (start + i) % 4 : (start + 7 - i) % 4;
                const double score = dot(unit(sub(center[order[1]], center[order[3]])), reference.first) +
                                     dot(unit(sub(center[order[2]], center[order[0]])), reference.second);
                if (!(score > best + 1e-12)) continue;
                best = score;
                for (size_t i = 0; i < 4; i++) {
                    sides[i] = segment[order[i]];
                    if (!forward) std::reverse(sides[i].begin(), sides[i].end());
                }
            }
        }
    }

    std::vector<bool> fixed(m.x.size(), false);
    std::vector<double> u(m.x.size(), 0.0), v(m.x.size(), 0.0);
    const double inf = std::numeric_limits<double>::infinity();
    std::vector<double> from_u(m.x.size(), inf), from_v(m.x.size(), inf);
    for (size_t i = 0; i < 4; i++) {
        std::vector<double> along(sides[i].size(), 0.0);
        for (size_t k = 1; k < along.size(); k++) {
            along[k] = along[k - 1] + length(sub(m.x[sides[i][k]], m.x[sides[i][k - 1]]));
        }
        for (size_t k = 0; k < along.size(); k++) {
            const uint32_t node = sides[i][k];
            const double s = along[k] / along.back();
            fixed[node] = true;
            u[node] = i == 0 ? s : i == 1 ? 1.0 : i == 2 ? 1.0 - s : 0.0;
            v[node] = i == 0 ? 0.0 : i == 1 ? s : i == 2 ? 1.0 : 1.0 - s;
            if (i == 3) from_u[node] = 0.0;
            if (i == 0) from_v[node] = 0.0;
        }
    }
    const std::vector<double> none(m.x.size(), 0.0);
    u = K.solve(fixed, u, none, at);
    v = K.solve(fixed, v, none, at);
    from_u = distance(m, from_u);
    from_v = distance(m, from_v);

    Coordinates out{Shape::FOUR_SIDED, std::vector<Values>(m.n_faces()), ""};
    for (size_t f = 0; f < m.n_faces(); f++) {
        out.faces[f] = {std::clamp(m.at_center(f, u), 0.0, 1.0), std::clamp(m.at_center(f, v), 0.0, 1.0),
                        m.at_center(f, from_u), m.at_center(f, from_v)};
    }
    out.summary = "four-sided patch, (u, v) = (0, 0) at " + text(m.x[sides[0].front()]) + ", (1, 0) at " +
                  text(m.x[sides[1].front()]) + ", (1, 1) at " + text(m.x[sides[2].front()]) + ", (0, 1) at " +
                  text(m.x[sides[3].front()]);
    return out;
}

/** @brief u_m of a ring or a disk: the arc length around it at each face, between the lengths of its v = 0 and v = 1 ends. */
void set_arc_lengths(Coordinates & out, const std::vector<double> & from_v,
                     const std::vector<uint32_t> & last, const double length_first, const double length_last) {
    double across = 0.0;
    for (const uint32_t i : last) across += from_v[i];
    across /= static_cast<double>(last.size());
    for (Values & f : out.faces) {
        const double part = across > 0.0 ? std::min(f.v_m / across, 1.0) : 0.0;
        f.u_m = f.u * (length_first + (length_last - length_first) * part);
    }
}

Coordinates ring(const Triangulation & m, const Laplacian & K, const Corners & at) {
    if (at.points) {
        fail(at, "corners are given, but the patch is a ring (its outline has two loops), which has none",
             "Remove corners");
    }
    // v = 0 on the shorter rim; of two as long, on the lower one along the axis' dominant coordinate
    const std::array<double, 2> rim{m.loop_length(m.loops[0]), m.loop_length(m.loops[1])};
    const std::array<Vec3, 2> center{m.chain_center(m.loops[0], true), m.chain_center(m.loops[1], true)};
    Vec3 axis = unit(m.mean_normal);
    if (!(length(m.mean_normal) > 0.1 * m.total_area)) axis = unit(sub(center[1], center[0]));
    if (dot(axis, axis) == 0.0) axis = {0.0, 0.0, 1.0};
    const size_t k = dominant(axis);
    size_t inner = center[0][k] <= center[1][k] ? 0 : 1;
    if (std::abs(rim[0] - rim[1]) > 1e-6 * (rim[0] + rim[1])) inner = rim[0] < rim[1] ? 0 : 1;
    const std::vector<uint32_t> & first = m.loops[inner], & last = m.loops[1 - inner];

    std::vector<bool> fixed(m.x.size(), false);
    const double inf = std::numeric_limits<double>::infinity();
    std::vector<double> v(m.x.size(), 0.0), from_v(m.x.size(), inf);
    for (const uint32_t i : first) {
        fixed[i] = true;
        from_v[i] = 0.0;
    }
    for (const uint32_t i : last) {
        fixed[i] = true;
        v[i] = 1.0;
    }
    v = K.solve(fixed, v, std::vector<double>(m.x.size(), 0.0), at);
    from_v = distance(m, from_v);
    double period = 0.0;
    for (const uint32_t i : last) period += K.row(i, v);
    if (!(period > 0.0)) fail(at, "Laplace's equation on the ring has no flux between its rims", SPLIT);
    const std::vector<double> u = around(m, conjugate(m, v, NONE), period, Reference(axis));

    Coordinates out{Shape::RING, std::vector<Values>(m.n_faces()), ""};
    for (size_t f = 0; f < m.n_faces(); f++) {
        out.faces[f] = {u[f], std::clamp(m.at_center(f, v), 0.0, 1.0), 0.0, m.at_center(f, from_v)};
    }
    set_arc_lengths(out, from_v, last, rim[inner], rim[1 - inner]);
    out.summary = "ring, v = 0 on the rim around " + text(center[inner]) + ", 1 on the rim around " +
                  text(center[1 - inner]) + ", u = 0 toward " + text(Reference(axis).first);
    return out;
}

Coordinates disk(const Triangulation & m, const Laplacian & K, const Corners & at) {
    // The source of the Green's function: the surface's point nearest to its center
    uint32_t pole = 0;
    std::array<double, 3> weights{};
    double nearest = std::numeric_limits<double>::infinity();
    Vec3 center{};
    for (uint32_t t = 0; t < m.tri.size(); t++) {
        const auto & n = m.tri[t];
        const std::array<double, 3> w = nearest_weights(m.center, m.x[n[0]], m.x[n[1]], m.x[n[2]]);
        Vec3 p{};
        for (size_t k = 0; k < 3; k++) p = axpy(p, w[k], m.x[n[k]]);
        const double d = length(sub(p, m.center));
        if (d < nearest) {
            nearest = d;
            pole = t;
            weights = w;
            center = p;
        }
    }
    std::vector<bool> fixed(m.x.size(), false);
    for (const uint32_t i : m.loops[0]) fixed[i] = true;
    const double inf = std::numeric_limits<double>::infinity();
    std::vector<double> load(m.x.size(), 0.0), from_v(m.x.size(), inf);
    for (size_t k = 0; k < 3; k++) {
        const uint32_t i = m.tri[pole][k];
        if (fixed[i]) {
            fail(at, "the patch is too coarse to place its center: the face at its center touches its outline",
                 "Refine the zone, or give four corners = [[x, y, z], ...] to make it a four-sided patch");
        }
        load[i] = weights[k];
        from_v[i] = length(sub(m.x[i], center));
    }
    std::vector<double> v = K.solve(fixed, std::vector<double>(m.x.size(), 0.0), load, at);
    const std::vector<double> u = around(m, conjugate(m, v, pole), 1.0, Reference(unit(m.mean_normal)));
    for (double & value : v) value = std::exp(-TWO_PI * value);
    from_v = distance(m, from_v);

    Coordinates out{Shape::DISK, std::vector<Values>(m.n_faces()), ""};
    for (size_t f = 0; f < m.n_faces(); f++) {
        out.faces[f] = {u[f], std::clamp(m.at_center(f, v), 0.0, 1.0), 0.0, m.at_center(f, from_v)};
    }
    set_arc_lengths(out, from_v, m.loops[0], 0.0, m.loop_length(m.loops[0]));
    out.summary = "disk, v = 0 at " + text(center) + ", u = 0 toward " + text(Reference(unit(m.mean_normal)).first);
    return out;
}

} // namespace

Coordinates coordinates(const Surface & surface, const Corners & at) {
    if (at.points && !at.points->empty() && at.points->size() != 4) {
        fail(at, "corners has " + std::to_string(at.points->size()) + " points",
             "Give the four corners of the patch, or corners = [] for a patch without corners (a disk)");
    }
    const Triangulation m = triangulate(surface, at);
    const Laplacian K(m);
    if (m.loops.size() == 2) return ring(m, K, at);
    if (at.points) return at.points->empty() ? disk(m, K, at) : four_sided(m, K, at, {});

    const std::vector<uint32_t> & loop = m.loops[0];
    std::vector<size_t> found;
    for (size_t k = 0; k < loop.size(); k++) {
        const Vec3 in = sub(m.x[loop[k]], m.x[loop[(k + loop.size() - 1) % loop.size()]]);
        const Vec3 out = sub(m.x[loop[(k + 1) % loop.size()]], m.x[loop[k]]);
        if (std::atan2(length(cross(in, out)), dot(in, out)) > CORNER_ANGLE) found.push_back(k);
    }
    if (found.empty()) return disk(m, K, at);
    if (found.size() == 4) return four_sided(m, K, at, found);
    std::string list;
    for (size_t k = 0; k < std::min<size_t>(found.size(), 6); k++) list += (k ? ", " : "") + text(m.x[loop[found[k]]]);
    fail(at, "the patch's outline has " + std::to_string(found.size()) + " corners (where it turns by more than 30 "
             "degrees: " + list + (found.size() > 6 ? ", ..." : "") + "), and u, v need four, or none",
         "Give the four corners to use as corners = [[x, y, z], ...], in the order (u, v) = (0, 0), (1, 0), (1, 1), "
         "(0, 1), or corners = [] to treat the patch as a disk");
}

} // namespace patch

/**
 * @file geometry.h
 * @author Matthew Bonanni (mbonanni001@gmail.com)
 * @brief Constructive solid geometry: primitives, transforms and booleans,
 *        evaluated with the same code on the host and in device kernels.
 *        Implicit shapes come from the caller as a function of the local
 *        point, so the geometry knows neither inputs nor expressions.
 * @version 0.1
 * @date 2026-10-15
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 *
 */

#ifndef GEOMETRY_H
#define GEOMETRY_H

#include <cstdint>
#include <string>
#include <vector>

#include <Kokkos_Core.hpp>
#include "common_typedef.h"

namespace region {

enum class Kind : uint8_t {
    SPHERE, ELLIPSOID, CYLINDER, CONE, BOX, TORUS, HALF_SPACE, POLYGON, IMPLICIT, TRIANGLES,
    UNION, INTERSECTION, DIFFERENCE,
};

/** @brief Subdivisions of the cells' simplices for extracted boundaries: the default and the most. */
constexpr int DEFAULT_RESOLUTION = 3;
constexpr int MAX_RESOLUTION = N_DIM == 2 ? 8 : 5;

/** @brief Deepest evaluation stack of a region (nesting of booleans). */
constexpr int MAX_STACK = 16;

/**
 * @brief A primitive, in postfix order with the booleans that combine the
 *        ones before it. A primitive's signed distance phi (negative inside,
 *        1-Lipschitz, so |phi| never exceeds the distance to the surface) is
 *        scale * phi_local(q) at the local point q = m x + d.
 */
struct Node {
    Kind kind = Kind::SPHERE;
    uint32_t first = 0, count = 0;  // POLYGON: its vertices; IMPLICIT: its implicit shape; TRIANGLES: its surface
                                    // (vertices from 2 first, count pairs; see triangles::)
    double m[9] = {1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0};
    double d[3] = {0.0, 0.0, 0.0};
    double scale = 1.0;
    double p[4] = {0.0, 0.0, 0.0, 0.0};  // primitive parameters (see region.cpp); TRIANGLES: p[0] = 1 if open
};

/**
 * @brief One step of a region's frame, local' = L (local - o), distances
 *        times s, whose values may be expressions of t (programs of a
 *        kexpr::ProgramTable of t, -1 for the constant). AFFINE: L and s
 *        fixed, o_i = program[i](t) or o[i]; ROTATE: by -(program[0](t) or
 *        c[0]) degrees about axis, o fixed; SCALE: L = diag(1 / v), s the
 *        least v_i of the run's axes, v_i = program[i](t) or c[i], o fixed.
 */
struct Step {
    enum class Kind : uint8_t { AFFINE, ROTATE, SCALE };
    Kind kind = Kind::AFFINE;
    double L[9] = {1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0};
    double o[3] = {0.0, 0.0, 0.0};
    double s = 1.0;
    double axis[3] = {0.0, 0.0, 1.0};
    double c[3] = {0.0, 0.0, 0.0};
    int32_t program[3] = {-1, -1, -1};

    bool fixed() const { return program[0] < 0 && program[1] < 0 && program[2] < 0; }
};

/** @brief Rows of the rotation by -angle degrees about the unit axis (world to rotated coordinates). */
KOKKOS_INLINE_FUNCTION void inverse_rotation(const double axis[3], const double angle, double L[9]) {
    const double a = -angle * 3.14159265358979323846 / 180.0, c = Kokkos::cos(a), s = Kokkos::sin(a), C = 1.0 - c;
    const double x = axis[0], y = axis[1], z = axis[2];
    L[0] = c + x * x * C;
    L[1] = x * y * C - z * s;
    L[2] = x * z * C + y * s;
    L[3] = y * x * C + z * s;
    L[4] = c + y * y * C;
    L[5] = y * z * C - x * s;
    L[6] = z * x * C - y * s;
    L[7] = z * y * C + x * s;
    L[8] = c + z * z * C;
}

/** @brief The frame local = M x + d (distances times k) followed by local' = L (local - o), distances times s. */
KOKKOS_INLINE_FUNCTION void compose(const double L[9], const double o[3], const double s, double M[9], double d[3],
                                    double & k) {
    double M2[9], d2[3];
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            double v = 0.0;
            for (int l = 0; l < 3; l++) v += L[3 * i + l] * M[3 * l + j];
            M2[3 * i + j] = v;
        }
        double v = 0.0;
        for (int l = 0; l < 3; l++) v += L[3 * i + l] * (d[l] - o[l]);
        d2[i] = v;
    }
    for (int i = 0; i < 9; i++) M[i] = M2[i];
    for (int i = 0; i < 3; i++) d[i] = d2[i];
    k *= s;
}

/** @brief The frame (M, d, k) followed by step, with value(p) the value of program p at the time. */
template <typename Value>
KOKKOS_INLINE_FUNCTION void apply(const Step & step, const Value & value, double M[9], double d[3], double & k) {
    double v[3];
    for (int i = 0; i < 3; i++) {
        const double fixed = step.kind == Step::Kind::AFFINE ? step.o[i] : step.c[i];
        v[i] = step.program[i] >= 0 ? value(static_cast<uint32_t>(step.program[i])) : fixed;
    }
    if (step.kind == Step::Kind::AFFINE) {
        compose(step.L, v, step.s, M, d, k);
    } else if (step.kind == Step::Kind::ROTATE) {
        double L[9];
        inverse_rotation(step.axis, v[0], L);
        compose(L, step.o, 1.0, M, d, k);
    } else {
        const double L[9] = {1.0 / v[0], 0.0, 0.0, 0.0, 1.0 / v[1], 0.0, 0.0, 0.0, 1.0 / v[2]};
        double least = v[0] < v[1] ? v[0] : v[1];
        if (N_DIM == 3 && v[2] < least) least = v[2];
        compose(L, step.o, least, M, d, k);
    }
}

/**
 * @brief How a moving primitive's node follows t: its frame, the steps
 *        [first, first + count) from the world's, and parameters p[i] =
 *        factor[i] * program[i](t) where program[i] >= 0.
 */
struct Motion {
    uint32_t node = 0;
    uint32_t first = 0, count = 0;
    int32_t program[4] = {-1, -1, -1, -1};
    double factor[4] = {1.0, 1.0, 1.0, 1.0};
};

/** @brief Node n of a moving region at the time whose program values value(p) gives. */
template <typename Value>
KOKKOS_INLINE_FUNCTION void move(const Motion & motion, const Step * steps, const Value & value, Node & n) {
    double M[9] = {1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0}, d[3] = {0.0, 0.0, 0.0}, k = 1.0;
    for (uint32_t i = 0; i < motion.count; i++) apply(steps[motion.first + i], value, M, d, k);
    for (int i = 0; i < 9; i++) n.m[i] = M[i];
    for (int i = 0; i < 3; i++) n.d[i] = d[i];
    n.scale = k;
    for (int i = 0; i < 4; i++) {
        if (motion.program[i] >= 0) n.p[i] = motion.factor[i] * value(static_cast<uint32_t>(motion.program[i]));
    }
}

/**
 * @brief The layout of a triangulated surface (Kind::TRIANGLES) in a region's vertices, from
 *        vertices + 2 first: the counts of triangles and of nodes of its bounding volume hierarchy, the nodes
 *        (each its box lo, hi, then a leaf's first triangle and -count, or an inner node's two children), and
 *        the triangles in the leaves' order (corners a, b, c counterclockwise seen from outside, then the
 *        pseudonormals of the edges ab, bc, ca and of the corners a, b, c, angle weighted). An open surface
 *        (Node::p[0] = 1) bounds no solid: its sign is not an inside, and it is only ever a surface region.
 */
namespace triangles {
constexpr uint32_t HEADER = 2;
constexpr uint32_t NODE = 8;
constexpr uint32_t TRIANGLE = 27;
constexpr uint32_t LEAF = 4;    // most triangles in a leaf
constexpr int MAX_DEPTH = 40;   // of the hierarchy, and its traversal stack
} // namespace triangles

namespace detail {

KOKKOS_INLINE_FUNCTION double max(const double a, const double b) { return a < b ? b : a; }
KOKKOS_INLINE_FUNCTION double min(const double a, const double b) { return b < a ? b : a; }
KOKKOS_INLINE_FUNCTION double clamp(const double v, const double lo, const double hi) { return min(max(v, lo), hi); }
KOKKOS_INLINE_FUNCTION double length2(const double a, const double b) { return Kokkos::sqrt(a * a + b * b); }

/** @brief Signed distance to the rectangle |u| <= hu, |v| <= hv (infinite half-widths allowed). */
KOKKOS_INLINE_FUNCTION double rectangle(const double du, const double dv) {
    return min(max(du, dv), 0.0) + length2(max(du, 0.0), max(dv, 0.0));
}

/** @brief Signed distance of a capped cone (frustum) along z: radius r1 at z = -h, r2 at z = h. */
KOKKOS_INLINE_FUNCTION double cone(const double rho, const double z, const double h, const double r1, const double r2) {
    const double k2x = r2 - r1, k2y = 2.0 * h;
    const double cax = rho - min(rho, z < 0.0 ? r1 : r2), cay = Kokkos::fabs(z) - h;
    const double s = clamp(((r2 - rho) * k2x + (h - z) * k2y) / (k2x * k2x + k2y * k2y), 0.0, 1.0);
    const double cbx = rho - r2 + k2x * s, cby = z - h + k2y * s;
    const double sign = (cbx < 0.0 && cay < 0.0) ? -1.0 : 1.0;
    return sign * Kokkos::sqrt(min(cax * cax + cay * cay, cbx * cbx + cby * cby));
}

/** @brief Signed distance to the polygon of vertices v[0, n) in the xy plane. */
KOKKOS_INLINE_FUNCTION double polygon(const double * v, const uint32_t n, const double x, const double y) {
    double d = (x - v[0]) * (x - v[0]) + (y - v[1]) * (y - v[1]);
    bool inside = false;
    for (uint32_t i = 0, j = n - 1; i < n; j = i, i++) {
        const double ex = v[2 * j] - v[2 * i], ey = v[2 * j + 1] - v[2 * i + 1];
        const double wx = x - v[2 * i], wy = y - v[2 * i + 1];
        const double s = clamp((wx * ex + wy * ey) / (ex * ex + ey * ey), 0.0, 1.0);
        const double bx = wx - ex * s, by = wy - ey * s;
        d = min(d, bx * bx + by * by);
        const bool c1 = y >= v[2 * i + 1], c2 = y < v[2 * j + 1], c3 = ex * wy > ey * wx;
        if ((c1 && c2 && c3) || (!c1 && !c2 && !c3)) inside = !inside;
    }
    return (inside ? -1.0 : 1.0) * Kokkos::sqrt(d);
}

KOKKOS_INLINE_FUNCTION double dot3(const double * a, const double * b) { return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]; }

/** @brief The squared distance from p to x, with c set to x. */
KOKKOS_INLINE_FUNCTION double to_point(const double p[3], const double * x, double c[3]) {
    double d2 = 0.0;
    for (int i = 0; i < 3; i++) {
        c[i] = x[i];
        d2 += (p[i] - x[i]) * (p[i] - x[i]);
    }
    return d2;
}

/** @brief The squared distance from p to the point a + s (b - a), with c set to it. */
KOKKOS_INLINE_FUNCTION double to_lerp(const double p[3], const double * a, const double * b, const double s, double c[3]) {
    double d2 = 0.0;
    for (int i = 0; i < 3; i++) {
        c[i] = a[i] + s * (b[i] - a[i]);
        d2 += (p[i] - c[i]) * (p[i] - c[i]);
    }
    return d2;
}

/**
 * @brief The squared distance from p to the triangle t (triangles:: layout), the closest point c, and the
 *        feature it lies on: 0 the face, 1, 2, 3 the edges ab, bc, ca, 4, 5, 6 the corners a, b, c (Ericson,
 *        Real-Time Collision Detection, 5.1.5). A triangle of no area gives the closest of its edges.
 */
KOKKOS_INLINE_FUNCTION double to_triangle(const double * t, const double p[3], double c[3], int & feature) {
    const double *a = t, *b = t + 3, *v = t + 6;
    double ab[3], ac[3], ap[3], bp[3], cp[3];
    for (int i = 0; i < 3; i++) {
        ab[i] = b[i] - a[i];
        ac[i] = v[i] - a[i];
        ap[i] = p[i] - a[i];
        bp[i] = p[i] - b[i];
        cp[i] = p[i] - v[i];
    }
    const double d1 = dot3(ab, ap), d2 = dot3(ac, ap);
    if (d1 <= 0.0 && d2 <= 0.0) {
        feature = 4;
        return to_point(p, a, c);
    }
    const double d3 = dot3(ab, bp), d4 = dot3(ac, bp);
    if (d3 >= 0.0 && d4 <= d3) {
        feature = 5;
        return to_point(p, b, c);
    }
    const double vc = d1 * d4 - d3 * d2;
    if (vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0) {
        feature = 1;
        return to_lerp(p, a, b, d1 / (d1 - d3), c);
    }
    const double d5 = dot3(ab, cp), d6 = dot3(ac, cp);
    if (d6 >= 0.0 && d5 <= d6) {
        feature = 6;
        return to_point(p, v, c);
    }
    const double vb = d5 * d2 - d1 * d6;
    if (vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0) {
        feature = 3;
        return to_lerp(p, a, v, d2 / (d2 - d6), c);
    }
    const double va = d3 * d6 - d5 * d4;
    if (va <= 0.0 && d4 - d3 >= 0.0 && d5 - d6 >= 0.0) {
        feature = 2;
        return to_lerp(p, b, v, (d4 - d3) / ((d4 - d3) + (d5 - d6)), c);
    }
    if (!(va + vb + vc > 0.0)) {
        // No area: the closest edge
        const double * ends[3][2] = {{a, b}, {b, v}, {v, a}};
        double best = 0.0;
        for (int e = 0; e < 3; e++) {
            double l = 0.0, s = 0.0;
            for (int i = 0; i < 3; i++) {
                l += (ends[e][1][i] - ends[e][0][i]) * (ends[e][1][i] - ends[e][0][i]);
                s += (p[i] - ends[e][0][i]) * (ends[e][1][i] - ends[e][0][i]);
            }
            s = l > 0.0 ? clamp(s / l, 0.0, 1.0) : 0.0;
            double q[3];
            const double d = to_lerp(p, ends[e][0], ends[e][1], s, q);
            if (e == 0 || d < best) {
                best = d;
                feature = 1 + e;
                for (int i = 0; i < 3; i++) c[i] = q[i];
            }
        }
        return best;
    }
    const double w = 1.0 / (va + vb + vc), sb = vb * w, sc = vc * w;
    feature = 0;
    double squared = 0.0;
    for (int i = 0; i < 3; i++) {
        c[i] = a[i] + ab[i] * sb + ac[i] * sc;
        squared += (p[i] - c[i]) * (p[i] - c[i]);
    }
    return squared;
}

/** @brief The squared distance from p to a box lo, hi. */
KOKKOS_INLINE_FUNCTION double to_box(const double * box, const double p[3]) {
    double d2 = 0.0;
    for (int i = 0; i < 3; i++) {
        const double e = max(max(box[i] - p[i], p[i] - box[3 + i]), 0.0);
        d2 += e * e;
    }
    return d2;
}

/**
 * @brief Signed distance to a closed triangulated surface s (triangles:: layout) at q: the distance to the
 *        closest triangle, found through the hierarchy, negative where q is behind the pseudonormal of the
 *        feature the closest point lies on (Baerentzen and Aanaes 2005), which is exact for a closed,
 *        consistently oriented surface.
 */
KOKKOS_INLINE_FUNCTION double triangulated(const double * s, const double q[3]) {
    const uint32_t n_nodes = static_cast<uint32_t>(s[1]);
    const double * bvh = s + triangles::HEADER;
    const double * tris = bvh + triangles::NODE * n_nodes;
    double best = 1.0e300, closest[3] = {0.0, 0.0, 0.0};
    uint32_t best_triangle = 0;
    int best_feature = 0;
    uint32_t stack[triangles::MAX_DEPTH + 1];
    int top = 0;
    stack[top++] = 0;
    while (top > 0) {
        const double * node = bvh + triangles::NODE * stack[--top];
        if (to_box(node, q) >= best) continue;
        if (node[7] < 0.0) {
            const uint32_t first = static_cast<uint32_t>(node[6]), n = static_cast<uint32_t>(-node[7]);
            for (uint32_t k = first; k < first + n; k++) {
                double c[3];
                int feature = 0;
                const double d2 = to_triangle(tris + triangles::TRIANGLE * k, q, c, feature);
                if (d2 < best) {
                    best = d2;
                    best_triangle = k;
                    best_feature = feature;
                    for (int i = 0; i < 3; i++) closest[i] = c[i];
                }
            }
            continue;
        }
        const uint32_t l = static_cast<uint32_t>(node[6]), r = static_cast<uint32_t>(node[7]);
        const double dl = to_box(bvh + triangles::NODE * l, q), dr = to_box(bvh + triangles::NODE * r, q);
        // The nearer child on top, so that it is searched first
        const bool left_first = dl <= dr;
        const double d_far = left_first ? dr : dl, d_near = left_first ? dl : dr;
        if (d_far < best) stack[top++] = left_first ? r : l;
        if (d_near < best) stack[top++] = left_first ? l : r;
    }
    const double * t = tris + triangles::TRIANGLE * best_triangle;
    double n[3];
    if (best_feature == 0) {
        const double u[3] = {t[3] - t[0], t[4] - t[1], t[5] - t[2]}, v[3] = {t[6] - t[0], t[7] - t[1], t[8] - t[2]};
        n[0] = u[1] * v[2] - u[2] * v[1];
        n[1] = u[2] * v[0] - u[0] * v[2];
        n[2] = u[0] * v[1] - u[1] * v[0];
    } else {
        for (int i = 0; i < 3; i++) n[i] = t[9 + 3 * (best_feature - 1) + i];
    }
    const double side = (q[0] - closest[0]) * n[0] + (q[1] - closest[1]) * n[1] + (q[2] - closest[2]) * n[2];
    const double d = Kokkos::sqrt(best);
    return side < 0.0 ? -d : d;
}

/** @brief implicit(i, q): the function of implicit shape i at the local point q (inside: negative). */
template <typename Implicit>
KOKKOS_INLINE_FUNCTION double primitive(const Node & n, const double * vertices, const Implicit & implicit,
                                        const double x[3]) {
    double q[3];
    for (int i = 0; i < 3; i++) {
        q[i] = n.m[3 * i] * x[0] + n.m[3 * i + 1] * x[1] + n.m[3 * i + 2] * x[2] + n.d[i];
    }
    double phi = 0.0;
    switch (n.kind) {
        case Kind::SPHERE:
            phi = Kokkos::sqrt(q[0] * q[0] + q[1] * q[1] + q[2] * q[2]) - n.p[0];
            break;
        case Kind::ELLIPSOID: {
            // p: semi-axes and the smallest one, which makes |q / a| - 1 1-Lipschitz
            const double a = q[0] / n.p[0], b = q[1] / n.p[1], c = q[2] / n.p[2];
            phi = (Kokkos::sqrt(a * a + b * b + c * c) - 1.0) * n.p[3];
            break;
        }
        case Kind::CYLINDER: {
            // p: outer radius, inner radius, half length
            const double rho = length2(q[0], q[1]);
            const double dr = n.p[1] > 0.0 ? max(rho - n.p[0], n.p[1] - rho) : rho - n.p[0];
            phi = rectangle(dr, Kokkos::fabs(q[2]) - n.p[2]);
            break;
        }
        case Kind::CONE:
            // p: half length, radius at -h, radius at +h
            phi = cone(length2(q[0], q[1]), q[2], n.p[0], n.p[1], n.p[2]);
            break;
        case Kind::BOX: {
            // p: half sizes
            const double dx = Kokkos::fabs(q[0]) - n.p[0], dy = Kokkos::fabs(q[1]) - n.p[1];
            const double dz = Kokkos::fabs(q[2]) - n.p[2];
            const double ox = max(dx, 0.0), oy = max(dy, 0.0), oz = max(dz, 0.0);
            phi = Kokkos::sqrt(ox * ox + oy * oy + oz * oz) + min(max(dx, max(dy, dz)), 0.0);
            break;
        }
        case Kind::TORUS:
            // p: major and minor radii
            phi = length2(length2(q[0], q[1]) - n.p[0], q[2]) - n.p[1];
            break;
        case Kind::HALF_SPACE:
            phi = q[2];
            break;
        case Kind::POLYGON:
            phi = polygon(vertices + 2 * n.first, n.count, q[0], q[1]);
            break;
        case Kind::IMPLICIT:
            phi = implicit(n.first, q);
            break;
        case Kind::TRIANGLES:
            phi = triangulated(vertices + 2 * n.first, q);
            break;
        default:
            break;
    }
    return n.scale * phi;
}

} // namespace detail

/**
 * @brief Signed distance of the region nodes[0, n) at x (a lower bound on the
 *        distance to its surface, negative inside; implicit shapes give
 *        their function instead), with implicit as in detail::primitive.
 */
template <typename Implicit>
KOKKOS_INLINE_FUNCTION double distance(const Node * nodes, const uint32_t n, const double * vertices,
                                       const Implicit & implicit, const double x[3]) {
    if (n == 1) return detail::primitive(nodes[0], vertices, implicit, x);
    double s[MAX_STACK] = {};
    int top = -1;
    for (uint32_t i = 0; i < n; i++) {
        const Node & node = nodes[i];
        switch (node.kind) {
            case Kind::UNION:
                top--;
                s[top] = detail::min(s[top], s[top + 1]);
                break;
            case Kind::INTERSECTION:
                top--;
                s[top] = detail::max(s[top], s[top + 1]);
                break;
            case Kind::DIFFERENCE:
                top--;
                s[top] = detail::max(s[top], -s[top + 1]);
                break;
            default:
                s[++top] = detail::primitive(node, vertices, implicit, x);
                break;
        }
    }
    return s[0];
}

/**
 * @brief Indicator of a region with signed distance phi: 1 inside, 0 outside,
 *        and with a smooth edge of width edge, the smoothstep from 1 at
 *        phi = -edge / 2 to 0 at phi = edge / 2.
 */
KOKKOS_INLINE_FUNCTION double indicator(const double phi, const double edge) {
    if (!(edge > 0.0)) return phi < 0.0 ? 1.0 : 0.0;
    const double s = detail::clamp(0.5 - phi / edge, 0.0, 1.0);
    return s * s * (3.0 - 2.0 * s);
}

/**
 * @brief A filled region (a volume; an area in 2D) of a Region: shapes with a
 *        signed distance, the mesh's cells, where a condition on the solution
 *        holds, or a boolean of filled regions.
 */
struct Filled {
    enum class Kind : uint8_t { SHAPES, ALL, CELL_ZONE, INSIDE, CONDITION, UNION, INTERSECTION, DIFFERENCE };
    Kind kind = Kind::SHAPES;
    uint32_t first = 0, count = 0;   // SHAPES: its nodes, in postfix order; CONDITION: its condition
    std::vector<std::string> zones;  // CELL_ZONE: its name; INSIDE: the boundary zones that enclose it
    std::vector<uint32_t> parts;     // booleans: filled regions (DIFFERENCE: the first minus the others)
    bool cells = false;              // every cell entirely inside or outside (no shapes)
};

/**
 * @brief Part of a region of lower dimension (a surface, curve or points):
 *        where the boundaries of the carriers meet (each one lowers the
 *        dimension by one), or a set of mesh faces or their edges, in every
 *        inside filled region and outside every outside one.
 */
struct Piece {
    std::vector<uint32_t> carriers;  // filled regions of SHAPES (or a CONDITION) whose boundaries meet here
    std::vector<std::string> zones;  // or: the faces of these boundary zones
    int32_t cells = -1;              // or: the faces between a cells-only filled region and the rest
    bool edges = false;              // faces: their edges (2D: end points) that only one face of the set has
    std::vector<uint32_t> inside, outside;
    bool flip = false;               // facing into the filled region it bounds (the inner side of a difference)
};

} // namespace region

#endif // GEOMETRY_H

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
    SPHERE, ELLIPSOID, CYLINDER, CONE, BOX, TORUS, HALF_SPACE, POLYGON, IMPLICIT,
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
    uint32_t first = 0, count = 0;  // POLYGON: its vertices; IMPLICIT: its implicit shape
    double m[9] = {1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0};
    double d[3] = {0.0, 0.0, 0.0};
    double scale = 1.0;
    double p[4] = {0.0, 0.0, 0.0, 0.0};  // primitive parameters (see region.cpp)
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

/** @brief implicit(i, q): the function of implicit shape i at the local point q (inside: negative). */
template <typename Implicit>
KOKKOS_INLINE_FUNCTION double primitive(const Node & n, const double * vertices, const Implicit & implicit,
                                        const double x[3]) {
    double q[3];
    for (int i = 0; i < 3; i++) {
        q[i] = n.m[3 * i] * x[0] + n.m[3 * i + 1] * x[1] + n.m[3 * i + 2] * x[2] + n.d[i];
    }
    const double rho = length2(q[0], q[1]);
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
            const double dr = n.p[1] > 0.0 ? max(rho - n.p[0], n.p[1] - rho) : rho - n.p[0];
            phi = rectangle(dr, Kokkos::fabs(q[2]) - n.p[2]);
            break;
        }
        case Kind::CONE:
            // p: half length, radius at -h, radius at +h
            phi = cone(rho, q[2], n.p[0], n.p[1], n.p[2]);
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
            phi = length2(rho - n.p[0], q[2]) - n.p[1];
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
 *        signed distance, the mesh's cells, or a boolean of filled regions.
 */
struct Filled {
    enum class Kind : uint8_t { SHAPES, ALL, CELL_ZONE, INSIDE, UNION, INTERSECTION, DIFFERENCE };
    Kind kind = Kind::SHAPES;
    uint32_t first = 0, count = 0;   // SHAPES: its nodes, in postfix order
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
    std::vector<uint32_t> carriers;  // filled regions of SHAPES whose boundaries meet here
    std::vector<std::string> zones;  // or: the faces of these boundary zones
    int32_t cells = -1;              // or: the faces between a cells-only filled region and the rest
    bool edges = false;              // faces: their edges (2D: end points) that only one face of the set has
    std::vector<uint32_t> inside, outside;
    bool flip = false;               // facing into the filled region it bounds (the inner side of a difference)
};

} // namespace region

#endif // GEOMETRY_H

/**
 * @file region_clip.h
 * @author Matthew Bonanni (mbonanni001@gmail.com)
 * @brief The fraction of a straight cell inside a region with planar faces
 *        (boxes, rectangles, half-spaces, polygons): its simplices cut along
 *        those planes exactly, and only what a curved surface crosses
 *        sampled. The same code on the host and in device kernels.
 * @version 0.1
 * @date 2026-11-15
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 *
 */

#ifndef REGION_CLIP_H
#define REGION_CLIP_H

#include <cstdint>

#include <Kokkos_Core.hpp>
#include "common_typedef.h"
#include "geometry.h"
#include "region_sampler.h"

namespace region::sampler {

/** @brief Most planes a simplex is cut along before the rest of it is sampled. */
constexpr int CLIP_DEPTH = N_DIM == 2 ? 12 : 6;

/** @brief Most simplices of the two sides of a cut: 1 and 2 triangles, or two prisms of 3 tetrahedra. */
constexpr int CLIP_PIECES = N_DIM == 2 ? 3 : 6;

/** @brief Most pending simplices of a depth-first clip. */
constexpr int CLIP_STACK = CLIP_DEPTH * (CLIP_PIECES - 1) + 1;

/** @brief Most parts of a simplex classified before it is sampled instead: past it, sampling costs less. */
constexpr int CLIP_PARTS = N_DIM == 2 ? 256 : 512;

/** @brief The primitives of a region's shapes, in postfix order, and the vertices of its polygons. */
struct Planes {
    const Node * nodes = nullptr;
    uint32_t count = 0;
    const double * vertices = nullptr;
};

namespace detail {

/** @brief Planes that bound a primitive: a box's 6 faces, a half-space's plane, a polygon's edges, else none. */
KOKKOS_INLINE_FUNCTION uint32_t plane_count(const Node & n) {
    switch (n.kind) {
        case Kind::BOX: return 6;
        case Kind::HALF_SPACE: return 1;
        case Kind::POLYGON: return n.count;
        default: return 0;
    }
}

/**
 * @brief The values g[k] at the vertices of simplex v of the affine function
 *        that vanishes on plane j of primitive n, and whether the plane cuts
 *        the simplex: g changes sign over it, and the face (of a box or a
 *        polygon) may meet it, not lying wholly beside it.
 */
KOKKOS_INLINE_FUNCTION bool cuts(const Node & n, const double * vertices, const uint32_t j, const double v[4][3],
                                 double g[4]) {
    constexpr int n_vertices = N_DIM + 1;
    double q[4][3];
    for (int k = 0; k < n_vertices; k++) {
        for (int i = 0; i < 3; i++) {
            q[k][i] = n.m[3 * i] * v[k][0] + n.m[3 * i + 1] * v[k][1] + n.m[3 * i + 2] * v[k][2] + n.d[i];
        }
    }
    auto beside = [&](const int i, const double lo, const double hi) {
        bool below = true, above = true;
        for (int k = 0; k < n_vertices; k++) {
            below = below && q[k][i] < lo;
            above = above && q[k][i] > hi;
        }
        return below || above;
    };
    if (n.kind == Kind::BOX) {
        const int i = static_cast<int>(j / 2);
        const double side = j % 2 == 0 ? 1.0 : -1.0;
        if (!Kokkos::isfinite(n.p[i])) return false;
        for (int l = 0; l < 3; l++) {
            if (l != i && Kokkos::isfinite(n.p[l]) && beside(l, -n.p[l], n.p[l])) return false;
        }
        for (int k = 0; k < n_vertices; k++) g[k] = side * q[k][i] - n.p[i];
    } else if (n.kind == Kind::HALF_SPACE) {
        for (int k = 0; k < n_vertices; k++) g[k] = q[k][2];
    } else {
        const double * a = vertices + 2 * (n.first + j);
        const double * b = vertices + 2 * (n.first + (j + 1 == n.count ? 0 : j + 1));
        const double ex = b[0] - a[0], ey = b[1] - a[1];
        bool before = true, after = true;
        for (int k = 0; k < n_vertices; k++) {
            const double u = ex * (q[k][0] - a[0]) + ey * (q[k][1] - a[1]);
            before = before && u < 0.0;
            after = after && u > ex * ex + ey * ey;
            g[k] = ey * (q[k][0] - a[0]) - ex * (q[k][1] - a[1]);
        }
        if (before || after) return false;
    }
    bool negative = false, positive = false;
    for (int k = 0; k < n_vertices; k++) {
        negative = negative || g[k] < 0.0;
        positive = positive || g[k] > 0.0;
    }
    return negative && positive;
}

/** @brief The measure of simplex v: a triangle's area or a tetrahedron's volume. */
KOKKOS_INLINE_FUNCTION double measure(const double v[4][3]) {
    if constexpr (N_DIM == 2) {
        return 0.5 * Kokkos::fabs((v[1][0] - v[0][0]) * (v[2][1] - v[0][1]) - (v[1][1] - v[0][1]) * (v[2][0] - v[0][0]));
    } else {
        double e[3][3];
        for (int k = 0; k < 3; k++) {
            for (int i = 0; i < 3; i++) e[k][i] = v[k + 1][i] - v[0][i];
        }
        return Kokkos::fabs(e[0][0] * (e[1][1] * e[2][2] - e[1][2] * e[2][1]) -
                            e[0][1] * (e[1][0] * e[2][2] - e[1][2] * e[2][0]) +
                            e[0][2] * (e[1][0] * e[2][1] - e[1][1] * e[2][0])) / 6.0;
    }
}

/**
 * @brief The simplices of the two sides of simplex v cut by the plane where
 *        g, of values g[k] at its vertices, vanishes (g changes sign over v),
 *        into out; returns their count.
 */
KOKKOS_INLINE_FUNCTION int split(const double v[4][3], const double g[4], double out[CLIP_PIECES][4][3]) {
    auto cut = [&](const int a, const int b, double * x) {
        const double s = g[a] / (g[a] - g[b]);
        for (int i = 0; i < 3; i++) x[i] = v[a][i] + s * (v[b][i] - v[a][i]);
    };
    auto set = [&](double t[4][3], const double * p0, const double * p1, const double * p2, const double * p3) {
        for (int i = 0; i < 3; i++) {
            t[0][i] = p0[i];
            t[1][i] = p1[i];
            t[2][i] = p2[i];
            t[3][i] = p3 ? p3[i] : 0.0;
        }
    };
    if constexpr (N_DIM == 2) {
        // The vertex alone on its side: a triangle there, and the rest a quadrilateral of 2
        const int positive = (g[0] > 0.0) + (g[1] > 0.0) + (g[2] > 0.0);
        int a = 0;
        while ((g[a] > 0.0) != (positive == 1)) a++;
        const int b = (a + 1) % 3, c = (a + 2) % 3;
        double pb[3], pc[3];
        cut(a, b, pb);
        cut(a, c, pc);
        set(out[0], v[a], pb, pc, nullptr);
        set(out[1], v[b], v[c], pc, nullptr);
        set(out[2], v[b], pc, pb, nullptr);
        return 3;
    } else {
        // Prism a0 a1 a2 / b0 b1 b2 (edges a_i b_i): 3 tetrahedra
        auto prism = [&](double (*t)[4][3], const double * a0, const double * a1, const double * a2, const double * b0,
                         const double * b1, const double * b2) {
            set(t[0], a0, a1, a2, b2);
            set(t[1], a0, a1, b1, b2);
            set(t[2], a0, b0, b1, b2);
        };
        const int positive = (g[0] > 0.0) + (g[1] > 0.0) + (g[2] > 0.0) + (g[3] > 0.0);
        if (positive != 2) {
            // The vertex alone on its side: a tetrahedron there, and the rest a prism
            int a = 0;
            while ((g[a] > 0.0) != (positive == 1)) a++;
            const int b = (a + 1) % 4, c = (a + 2) % 4, d = (a + 3) % 4;
            double pb[3], pc[3], pd[3];
            cut(a, b, pb);
            cut(a, c, pc);
            cut(a, d, pd);
            set(out[0], v[a], pb, pc, pd);
            prism(out + 1, v[b], v[c], v[d], pb, pc, pd);
            return 4;
        }
        // Two on each side: a prism on each, from the edge of its pair to the cut
        int lo[2], hi[2], n_lo = 0, n_hi = 0;
        for (int k = 0; k < 4; k++) {
            if (g[k] > 0.0) {
                hi[n_hi++] = k;
            } else {
                lo[n_lo++] = k;
            }
        }
        double p[2][2][3];  // p[i][j]: on the edge from lo[i] to hi[j]
        for (int i = 0; i < 2; i++) {
            for (int j = 0; j < 2; j++) cut(lo[i], hi[j], p[i][j]);
        }
        prism(out, v[lo[0]], p[0][0], p[0][1], v[lo[1]], p[1][0], p[1][1]);
        prism(out + 3, v[hi[0]], p[0][0], p[1][0], v[hi[1]], p[0][1], p[1][1]);
        return 6;
    }
}


/** @brief Where a part of a simplex lies: wholly inside, wholly outside, or crossed by a surface. */
enum : uint8_t { INSIDE = 0, OUTSIDE = 1, CROSSED = 2 };

/** @brief A plane to cut along: plane of primitive node, and its values at the part's vertices. */
struct Cut {
    uint32_t node = 0, plane = 0;
    double g[4] = {0.0, 0.0, 0.0, 0.0};
};

/**
 * @brief Where simplex v, of centroid c and radius (its largest distance from
 *        c), lies in the region of planes: each planar primitive is crossed
 *        where a plane of it at or after plane0 of node0 cuts v, the first one
 *        into cut (cut.node = planes.count if none), and otherwise lies on the
 *        side of c (the earlier planes cut an ancestor of v, which lies on one
 *        side of them); each curved one is crossed unless its distance at c
 *        exceeds the radius; then the booleans, inside or outside wherever
 *        their parts decide it.
 */
KOKKOS_INLINE_FUNCTION uint8_t classify(const Planes & planes, const double v[4][3], const double c[3],
                                        const double radius, const uint32_t node0, const uint32_t plane0, Cut & cut) {
    uint8_t s[MAX_STACK] = {};
    int top = -1;
    cut.node = planes.count;
    const auto no_implicit = [](const uint32_t, const double *) { return 0.0; };
    for (uint32_t i = 0; i < planes.count; i++) {
        const Node & n = planes.nodes[i];
        if (n.kind == Kind::UNION || n.kind == Kind::INTERSECTION || n.kind == Kind::DIFFERENCE) {
            top--;
            const uint8_t a = s[top], b = s[top + 1];
            if (n.kind == Kind::UNION) {
                s[top] = a == INSIDE || b == INSIDE ? INSIDE : a == OUTSIDE && b == OUTSIDE ? OUTSIDE : CROSSED;
            } else if (n.kind == Kind::INTERSECTION) {
                s[top] = a == OUTSIDE || b == OUTSIDE ? OUTSIDE : a == INSIDE && b == INSIDE ? INSIDE : CROSSED;
            } else {
                s[top] = a == OUTSIDE || b == INSIDE ? OUTSIDE : a == INSIDE && b == OUTSIDE ? INSIDE : CROSSED;
            }
            continue;
        }
        const double phi = region::detail::primitive(n, planes.vertices, no_implicit, c);
        uint8_t state = phi < 0.0 ? INSIDE : OUTSIDE;
        const uint32_t n_planes = plane_count(n);
        if (n_planes == 0) {
            if (Kokkos::fabs(phi) < radius) state = CROSSED;
        } else if (i >= node0) {
            double g[4];
            for (uint32_t j = i == node0 ? plane0 : 0; j < n_planes; j++) {
                if (!cuts(n, planes.vertices, j, v, g)) continue;
                state = CROSSED;
                if (cut.node == planes.count) {
                    cut.node = i;
                    cut.plane = j;
                    for (int k = 0; k < 4; k++) cut.g[k] = g[k];
                }
                break;
            }
        }
        s[++top] = state;
    }
    return s[0];
}

} // namespace detail

/**
 * @brief Whether clipped_cell cuts along planes: a region with a sharp edge
 *        and a distance bound, of which some primitive has planar faces.
 */
template <typename Distance>
KOKKOS_INLINE_FUNCTION bool clips(const Shape<Distance> & shape, const Planes & planes) {
    if (!shape.bounded || shape.edge > 0.0) return false;
    for (uint32_t i = 0; i < planes.count; i++) {
        if (detail::plane_count(planes.nodes[i]) > 0) return true;
    }
    return false;
}

/**
 * @brief Adds the integrals of the indicator (times r in axisymmetric runs) and
 *        of 1 over a simplex of N_DIM + 1 vertices v, for a region that clips:
 *        a part wholly inside or outside (detail::classify) is added whole,
 *        otherwise it is cut along the first plane that crosses it, depth
 *        first. The simplex is sampled instead (simplex), as by straight_cell,
 *        where a curved surface crosses a part, a part is still crossed after
 *        CLIP_DEPTH cuts, or it takes more than CLIP_PARTS parts.
 */
template <typename Distance>
KOKKOS_INLINE_FUNCTION void clipped_simplex(const Shape<Distance> & shape, const Planes & planes,
                                            const bool axisymmetric, const double v0[4][3], const double measure0,
                                            double & inside, double & total) {
    struct Item {
        double v[4][3];
        double measure;
        uint32_t node, plane;  // the first plane that may still cut it
        int depth;
    };
    constexpr int n_vertices = N_DIM + 1;
    auto weight = [&](const double v[4][3], const double measure) {
        if (!axisymmetric) return measure;
        double r = 0.0;
        for (int k = 0; k < n_vertices; k++) r += v[k][1] / double(n_vertices);
        return measure * r;
    };
    Item stack[CLIP_STACK];
    int top = 0;
    for (int k = 0; k < 4; k++) {
        for (int i = 0; i < 3; i++) stack[0].v[k][i] = v0[k][i];
    }
    stack[0].measure = measure0;
    stack[0].node = stack[0].plane = 0;
    stack[0].depth = 0;
    const double w0 = weight(v0, measure0);
    double in = 0.0;
    bool some_in = false, some_out = false, sample = false;
    for (int parts = 0; top >= 0; parts++) {
        if (parts == CLIP_PARTS) {
            sample = true;
            break;
        }
        const Item item = stack[top--];
        const double w = weight(item.v, item.measure);
        double c[3] = {0.0, 0.0, 0.0};
        for (int k = 0; k < n_vertices; k++) {
            for (int i = 0; i < 3; i++) c[i] += item.v[k][i] / double(n_vertices);
        }
        double radius = 0.0;
        for (int k = 0; k < n_vertices; k++) {
            double d[3];
            for (int i = 0; i < 3; i++) d[i] = item.v[k][i] - c[i];
            radius = region::detail::max(radius, detail::norm(d));
        }
        const double phi = shape.distance(c);
        detail::Cut cut;
        const uint8_t state = Kokkos::fabs(phi) >= radius ? (phi < 0.0 ? detail::INSIDE : detail::OUTSIDE)
                                                           : detail::classify(planes, item.v, c, radius, item.node,
                                                                              item.plane, cut);
        if (state != detail::CROSSED) {
            some_in = some_in || state == detail::INSIDE;
            some_out = some_out || state == detail::OUTSIDE;
            in += state == detail::INSIDE ? w : 0.0;
            continue;
        }
        if (cut.node == planes.count || item.depth >= CLIP_DEPTH) {
            sample = true;
            break;
        }
        double pieces[CLIP_PIECES][4][3];
        const int n_pieces = detail::split(item.v, cut.g, pieces);
        for (int p = n_pieces - 1; p >= 0; p--) {
            Item & next = stack[++top];
            for (int k = 0; k < 4; k++) {
                for (int i = 0; i < 3; i++) next.v[k][i] = pieces[p][k][i];
            }
            next.measure = detail::measure(pieces[p]);
            next.node = cut.node;
            next.plane = cut.plane + 1;
            next.depth = item.depth + 1;
        }
    }
    if (sample) {
        simplex(shape, axisymmetric, v0, measure0, inside, total);
        return;
    }
    // Wholly inside or outside, it adds exactly what sampling adds
    inside += !some_out ? w0 : some_in ? region::detail::min(in, w0) : 0.0;
    total += w0;
}

/**
 * @brief Adds the integrals of the indicator and of 1 over a straight cell,
 *        whose nodes node(k, x) gives (n of them), over its simplices
 *        (straight_simplices): cut along the planes of planes, the primitives
 *        of shape, where it clips (clipped_simplex), otherwise sampled
 *        (straight_cell).
 */
template <typename Distance, typename Nodes>
KOKKOS_INLINE_FUNCTION void clipped_cell(const Shape<Distance> & shape, const Planes & planes, const bool axisymmetric,
                                         const uint32_t n, const Nodes & node, double & inside, double & total) {
    if (!clips(shape, planes)) {
        straight_cell(shape, axisymmetric, n, node, inside, total);
        return;
    }
    straight_simplices(n, node, [&](const double s[4][3], const double measure) {
        clipped_simplex(shape, planes, axisymmetric, s, measure, inside, total);
    });
}

} // namespace region::sampler

#endif // REGION_CLIP_H

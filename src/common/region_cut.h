/**
 * @file region_cut.h
 * @author Matthew Bonanni (mbonanni001@gmail.com)
 * @brief A surface (2D: curve) cut from one condition's linear level in a
 *        cell, on the host or the device: its quadrature points as
 *        region::elements gives them, bit for bit (the same double
 *        arithmetic in the same order, without contraction).
 * @version 0.1
 * @date 2026-11-18
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 *
 */

#ifndef REGION_CUT_H
#define REGION_CUT_H

#include <cstdint>

#include <Kokkos_Core.hpp>
#include "common_typedef.h"

namespace region::cut {

constexpr uint32_t MAX_CELL_NODES = 8;

/**
 * @brief Whether the point (x, y) comes before (u, v) in the order of atan2: angles in (-pi, pi], -pi only for
 *        y = -0 and x < 0. Exact arithmetic, so that the host and the device order alike.
 */
KOKKOS_INLINE_FUNCTION bool angle_before(const double y, const double x, const double v, const double u) {
    const auto group = [](const double b, const double a) {
        if (b == 0.0 && a < 0.0) return Kokkos::signbit(b) ? 0 : 4;
        return b < 0.0 ? 1 : b == 0.0 ? 2 : 3;
    };
    const int g = group(y, x), h = group(v, u);
    if (g != h) return g < h;
    return (g == 1 || g == 3) && x * v - y * u > 0.0;
}

/**
 * @brief Sorts the first n items with less(a, b), as libstdc++'s std::sort does a range this short (an insertion
 *        sort that first tries the front): the same order on the host and the device.
 */
template <typename T, typename Less>
KOKKOS_INLINE_FUNCTION void insertion_sort(T * items, const uint32_t n, const Less & less) {
    for (uint32_t i = 1; i < n; i++) {
        T v = items[i];
        if (less(v, items[0])) {
            for (uint32_t j = i; j > 0; j--) items[j] = items[j - 1];
            items[0] = v;
            continue;
        }
        uint32_t j = i;
        while (less(v, items[j - 1])) {
            items[j] = items[j - 1];
            j--;
        }
        items[j] = v;
    }
}

struct Vec {
    double x[3] = {0.0, 0.0, 0.0};
};

KOKKOS_INLINE_FUNCTION Vec add(const Vec & a, const Vec & b) { return {{a.x[0] + b.x[0], a.x[1] + b.x[1], a.x[2] + b.x[2]}}; }
KOKKOS_INLINE_FUNCTION Vec sub(const Vec & a, const Vec & b) { return {{a.x[0] - b.x[0], a.x[1] - b.x[1], a.x[2] - b.x[2]}}; }
KOKKOS_INLINE_FUNCTION Vec mul(const double s, const Vec & a) { return {{s * a.x[0], s * a.x[1], s * a.x[2]}}; }
KOKKOS_INLINE_FUNCTION double dot(const Vec & a, const Vec & b) { return a.x[0] * b.x[0] + a.x[1] * b.x[1] + a.x[2] * b.x[2]; }
KOKKOS_INLINE_FUNCTION Vec cross(const Vec & a, const Vec & b) {
    return {{a.x[1] * b.x[2] - a.x[2] * b.x[1], a.x[2] * b.x[0] - a.x[0] * b.x[2], a.x[0] * b.x[1] - a.x[1] * b.x[0]}};
}
KOKKOS_INLINE_FUNCTION double norm(const Vec & a) { return Kokkos::sqrt(dot(a, a)); }
KOKKOS_INLINE_FUNCTION Vec unit(const Vec & a) {
    const double n = norm(a);
    return n > 0.0 ? mul(1.0 / n, a) : Vec{};
}
KOKKOS_INLINE_FUNCTION Vec lerp(const Vec & a, const Vec & b, const double t) { return add(a, mul(t, sub(b, a))); }
KOKKOS_INLINE_FUNCTION double max(const double a, const double b) { return a < b ? b : a; }

/**
 * @brief Orders the n vertices of a planar polygon by index, counterclockwise about its normal n, from their
 *        angles about their mean (region_mesh.cpp: order_about).
 */
struct AngleAbout {
    const Vec * v;
    Vec c, e1, e2;
    KOKKOS_INLINE_FUNCTION AngleAbout(const Vec * vertices, const uint32_t count, const Vec & n) : v(vertices) {
        for (uint32_t i = 0; i < count; i++) c = add(c, mul(1.0 / double(count), v[i]));
        e1 = unit(sub(v[0], c));
        e2 = cross(n, e1);
    }
    KOKKOS_INLINE_FUNCTION bool operator()(const uint32_t a, const uint32_t b) const {
        const Vec p = sub(v[a], c), q = sub(v[b], c);
        return angle_before(dot(p, e2), dot(p, e1), dot(q, e2), dot(q, e1));
    }
};

/** @brief The local nodes of each face of a 3D cell of n nodes, as cell_local_faces lists them (-1 ends a face). */
KOKKOS_INLINE_FUNCTION int8_t local_face_node(const uint32_t n, const uint32_t f, const uint32_t k) {
    constexpr int8_t tet[4][4] = {{0, 2, 1, -1}, {0, 1, 3, -1}, {0, 3, 2, -1}, {1, 2, 3, -1}};
    constexpr int8_t pyramid[5][4] = {{0, 3, 2, 1}, {0, 1, 4, -1}, {1, 2, 4, -1}, {2, 3, 4, -1}, {3, 0, 4, -1}};
    constexpr int8_t prism[5][4] = {{0, 2, 1, -1}, {3, 4, 5, -1}, {0, 1, 4, 3}, {1, 2, 5, 4}, {2, 0, 3, 5}};
    constexpr int8_t hex[6][4] = {{0, 3, 2, 1}, {4, 5, 6, 7}, {0, 1, 5, 4}, {1, 2, 6, 5}, {2, 3, 7, 6}, {3, 0, 4, 7}};
    if (k >= 4) return -1;
    switch (n) {
        case 4: return f < 4 ? tet[f][k] : -1;
        case 5: return f < 5 ? pyramid[f][k] : -1;
        case 6: return f < 5 ? prism[f][k] : -1;
        case 8: return f < 6 ? hex[f][k] : -1;
        default: return -1;
    }
}

/**
 * @brief Emits the quadrature points (x, w, n) of a condition's zero set in a straight cell, as
 *        region::elements cuts a piece cut from that condition alone (unclipped, facing out of it): nodes (z = 0
 *        in 2D) and centroid xc of the cell, and the condition's level L = [value, gradient] there. Returns how
 *        many it emitted.
 */
template <typename Emit>
KOKKOS_INLINE_FUNCTION uint32_t level_cut(const Vec * nodes, const uint32_t n_nodes, const Vec & xc,
                                          const double * L, const Emit & emit) {
    // A level that keeps its sign over a sphere around the cell (region_mesh.cpp: off_conditions)
    double radius = 0.0;
    for (uint32_t k = 0; k < n_nodes; k++) radius = max(radius, norm(sub(nodes[k], xc)));
    double slope = 0.0;
    for (int d = 0; d < N_DIM; d++) slope += L[1 + d] * L[1 + d];
    if (Kokkos::fabs(L[0]) > 1.01 * Kokkos::sqrt(slope) * radius) return 0;
    uint32_t n_out = 0;
    const auto simplex = [&](const Vec * s) {
        constexpr int S = N_DIM + 1;
        Vec center;
        for (int i = 0; i < S; i++) center = add(center, mul(1.0 / double(S), s[i]));
        double r = 0.0;
        for (int i = 0; i < S; i++) r = max(r, norm(sub(s[i], center)));
        double phi[S];
        for (int i = 0; i < S; i++) {
            double g = L[0];
            for (int d = 0; d < N_DIM; d++) g += L[1 + d] * (s[i].x[d] - xc.x[d]);
            phi[i] = Kokkos::fabs(g) < 1e-12 * r ? 0.0 : g;
        }
        Vec poly[4];
        uint32_t n = 0;
        for (int i = 0; i < S; i++) {
            for (int j = i + 1; j < S; j++) {
                const double a = phi[i], b = phi[j];
                if ((a < 0.0) != (b < 0.0)) poly[n++] = lerp(s[i], s[j], a / (a - b));
            }
        }
        if (n == 0) return;
        Vec g;
        if constexpr (N_DIM == 2) {
            const Vec e1 = sub(s[1], s[0]), e2 = sub(s[2], s[0]);
            const double det = e1.x[0] * e2.x[1] - e1.x[1] * e2.x[0];
            const double d1 = phi[1] - phi[0], d2 = phi[2] - phi[0];
            g = {{(d1 * e2.x[1] - d2 * e1.x[1]) / det, (d2 * e1.x[0] - d1 * e2.x[0]) / det, 0.0}};
        } else {
            const Vec e1 = sub(s[1], s[0]), e2 = sub(s[2], s[0]), e3 = sub(s[3], s[0]);
            const double det = dot(e1, cross(e2, e3));
            const double d1 = phi[1] - phi[0], d2 = phi[2] - phi[0], d3 = phi[3] - phi[0];
            g = mul(1.0 / det, add(add(mul(d1, cross(e2, e3)), mul(d2, cross(e3, e1))), mul(d3, cross(e1, e2))));
        }
        const Vec direction = unit(g);
        if constexpr (N_DIM == 3) {
            if (n >= 3) {
                uint32_t order[4] = {0, 1, 2, 3};
                insertion_sort(order, n, AngleAbout(poly, n, direction));
                const Vec sorted[4] = {poly[order[0]], poly[order[1]], poly[order[2]], poly[order[3]]};
                for (uint32_t i = 0; i < n; i++) poly[i] = sorted[i];
            }
        } else if (n == 2) {
            const Vec t = sub(poly[1], poly[0]);
            if (t.x[1] * direction.x[0] - t.x[0] * direction.x[1] < 0.0) {
                const Vec swap = poly[0];
                poly[0] = poly[1];
                poly[1] = swap;
            }
        }
        // region_mesh.cpp: emit (measure_of, then default_quadrature)
        double measure = 1.0;
        if (n == 2) {
            measure = norm(sub(poly[1], poly[0]));
        } else if (n > 2) {
            Vec area;
            for (uint32_t i = 1; i + 1 < n; i++) area = add(area, cross(sub(poly[i], poly[0]), sub(poly[i + 1], poly[0])));
            measure = 0.5 * norm(area);
        }
        if (!(measure > 0.0)) return;
        if (n == 1) {
            emit(poly[0], 1.0, direction);
            n_out++;
        } else if (n == 2) {
            const double gauss_x[3] = {-0.7745966692414834, 0.0, 0.7745966692414834};
            const double gauss_w[3] = {5.0 / 9.0, 8.0 / 9.0, 5.0 / 9.0};
            const double length = norm(sub(poly[1], poly[0]));
            for (int q = 0; q < 3; q++) {
                emit(lerp(poly[0], poly[1], 0.5 * (1.0 + gauss_x[q])), 0.5 * gauss_w[q] * length, direction);
                n_out++;
            }
        } else {
            const double rt = Kokkos::sqrt(15.0);
            const double a1 = (6.0 - rt) / 21.0, a2 = (6.0 + rt) / 21.0;
            const double w0 = 9.0 / 40.0, w1 = (155.0 - rt) / 1200.0, w2 = (155.0 + rt) / 1200.0;
            const double rule[7][4] = {{1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0, w0},
                                       {a1, a1, 1.0 - 2.0 * a1, w1},
                                       {a1, 1.0 - 2.0 * a1, a1, w1},
                                       {1.0 - 2.0 * a1, a1, a1, w1},
                                       {a2, a2, 1.0 - 2.0 * a2, w2},
                                       {a2, 1.0 - 2.0 * a2, a2, w2},
                                       {1.0 - 2.0 * a2, a2, a2, w2}};
            for (uint32_t i = 1; i + 1 < n; i++) {
                const double area = 0.5 * norm(cross(sub(poly[i], poly[0]), sub(poly[i + 1], poly[0])));
                for (int q = 0; q < 7; q++) {
                    const double * b = rule[q];
                    emit(add(add(mul(b[0], poly[0]), mul(b[1], poly[i])), mul(b[2], poly[i + 1])), b[3] * area, direction);
                    n_out++;
                }
            }
        }
    };
    // The cell's simplices (region_mesh.cpp: simplices; mesh_3d.cpp: cell_tetrahedra)
    if constexpr (N_DIM == 2) {
        for (uint32_t k = 1; k + 1 < n_nodes; k++) {
            const Vec s[3] = {nodes[0], nodes[k], nodes[k + 1]};
            simplex(s);
        }
    } else {
        Vec o;
        for (uint32_t k = 0; k < n_nodes; k++) {
            for (int i = 0; i < 3; i++) o.x[i] += nodes[k].x[i];
        }
        for (int i = 0; i < 3; i++) o.x[i] /= static_cast<double>(n_nodes);
        for (uint32_t f = 0; local_face_node(n_nodes, f, 0) >= 0; f++) {
            uint32_t n = 0;
            while (n < 4 && local_face_node(n_nodes, f, n) >= 0) n++;
            const auto node = [&](const uint32_t k) { return nodes[static_cast<uint32_t>(local_face_node(n_nodes, f, k))]; };
            if (n == 3) {
                const Vec s[4] = {o, node(0), node(1), node(2)};
                simplex(s);
                continue;
            }
            Vec c;
            for (uint32_t k = 0; k < n; k++) {
                for (int i = 0; i < 3; i++) c.x[i] += node(k).x[i];
            }
            for (int i = 0; i < 3; i++) c.x[i] /= static_cast<double>(n);
            for (uint32_t k = 0; k < n; k++) {
                const Vec s[4] = {o, c, node(k), node((k + 1) % n)};
                simplex(s);
            }
        }
    }
    return n_out;
}

} // namespace region::cut

#endif // REGION_CUT_H

/**
 * @file region_sampler.h
 * @author Matthew Bonanni (mbonanni001@gmail.com)
 * @brief The fraction of a cell inside a region: its simplices, or a curved
 *        cell's cones, subdivided where the region's surface may cross
 *        them. The same code on the host and in device kernels.
 * @version 0.1
 * @date 2026-11-10
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 *
 */

#ifndef REGION_SAMPLER_H
#define REGION_SAMPLER_H

#include <cstdint>

#include <Kokkos_Core.hpp>
#include "common_typedef.h"
#include "geometry.h"

namespace region::sampler {

/** @brief Deepest subdivision of a simplex or cone box. */
constexpr int MAX_LEVEL = N_DIM == 2 ? 8 : 5;

/** @brief Children of a subdivision: 4 triangles, 8 tetrahedra, or 2^N_DIM boxes. */
constexpr int CHILDREN = N_DIM == 2 ? 4 : 8;

/** @brief Most pending simplices or boxes of a depth-first subdivision. */
constexpr int STACK = MAX_LEVEL * (CHILDREN - 1) + 1;

namespace detail {

KOKKOS_INLINE_FUNCTION double dot(const double * a, const double * b) { return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]; }
KOKKOS_INLINE_FUNCTION double norm(const double * a) { return Kokkos::sqrt(dot(a, a)); }

} // namespace detail

/**
 * @brief A region to sample: distance(x) its signed distance (or implicit
 *        function), with the width of its smooth edge, and whether |distance|
 *        bounds the distance to its surface.
 */
template <typename Distance>
struct Shape {
    Distance distance;
    double edge = 0.0;
    bool bounded = true;
};

/**
 * @brief Adds the integral of the indicator (times r in axisymmetric runs) and
 *        of 1 over a simplex of N_DIM + 1 vertices v to inside and total,
 *        subdivided into CHILDREN children where the surface or edge may
 *        cross it, depth first in the children's order.
 */
template <typename Distance>
KOKKOS_INLINE_FUNCTION void simplex(const Shape<Distance> & shape, const bool axisymmetric, const double v0[4][3],
                                    const double measure0, double & inside, double & total) {
    struct Item {
        double v[4][3];
        double measure;
        int level;
    };
    Item stack[STACK];
    int top = 0;
    for (int k = 0; k < 4; k++) {
        for (int i = 0; i < 3; i++) stack[0].v[k][i] = v0[k][i];
    }
    stack[0].measure = measure0;
    stack[0].level = 0;
    constexpr int n_vertices = N_DIM + 1;
    const double half_edge = 0.5 * shape.edge;
    while (top >= 0) {
        const Item item = stack[top--];
        const auto & v = item.v;
        double c[3] = {0.0, 0.0, 0.0};
        for (int k = 0; k < n_vertices; k++) {
            for (int i = 0; i < 3; i++) c[i] += v[k][i] / double(n_vertices);
        }
        double radius = 0.0;
        for (int k = 0; k < n_vertices; k++) {
            double d[3];
            for (int i = 0; i < 3; i++) d[i] = v[k][i] - c[i];
            radius = region::detail::max(radius, detail::norm(d));
        }
        const double phi = shape.distance(c);
        bool uniform = item.level >= MAX_LEVEL;
        if (shape.bounded) {
            if (Kokkos::fabs(phi) >= radius + half_edge) uniform = true;
            if (half_edge > 0.0 && radius < 0.125 * shape.edge && Kokkos::fabs(phi) < half_edge - radius) uniform = true;
        } else if (!uniform) {
            // Implicit regions: subdivide unless f keeps its sign over the simplex by twice its slope there
            double slope = 0.0;
            for (int k = 0; k < n_vertices; k++) {
                double d[3];
                for (int i = 0; i < 3; i++) d[i] = v[k][i] - c[i];
                slope = region::detail::max(slope, Kokkos::fabs(shape.distance(v[k]) - phi) / detail::norm(d));
            }
            uniform = Kokkos::fabs(phi) > 2.0 * slope * radius && half_edge == 0.0;
        }
        if (uniform) {
            const double w = item.measure * (axisymmetric ? c[1] : 1.0);
            inside += w * indicator(phi, shape.edge);
            total += w;
            continue;
        }
        // Midpoint subdivision: 4 triangles, or 8 tetrahedra (4 corners and an octahedron split along a diagonal)
        double m[4][4][3];
        for (int a = 0; a < n_vertices; a++) {
            for (int b = 0; b < n_vertices; b++) {
                for (int i = 0; i < 3; i++) m[a][b][i] = a == b ? v[a][i] : 0.5 * (v[a][i] + v[b][i]);
            }
        }
        constexpr int TRIANGLES[4][3][2] = {{{0, 0}, {0, 1}, {0, 2}},
                                            {{0, 1}, {1, 1}, {1, 2}},
                                            {{0, 2}, {1, 2}, {2, 2}},
                                            {{0, 1}, {1, 2}, {0, 2}}};
        constexpr int TETRAHEDRA[8][4][2] = {{{0, 0}, {0, 1}, {0, 2}, {0, 3}}, {{0, 1}, {1, 1}, {1, 2}, {1, 3}},
                                             {{0, 2}, {1, 2}, {2, 2}, {2, 3}}, {{0, 3}, {1, 3}, {2, 3}, {3, 3}},
                                             {{0, 1}, {2, 3}, {0, 2}, {0, 3}}, {{0, 1}, {2, 3}, {0, 3}, {1, 3}},
                                             {{0, 1}, {2, 3}, {1, 3}, {1, 2}}, {{0, 1}, {2, 3}, {1, 2}, {0, 2}}};
        // Pushed last child first, so that they are summed in order
        for (int child = CHILDREN - 1; child >= 0; child--) {
            Item & next = stack[++top];
            for (int k = 0; k < 4; k++) {
                for (int i = 0; i < 3; i++) {
                    if (N_DIM == 2) {
                        next.v[k][i] = k < 3 ? m[TRIANGLES[child][k][0]][TRIANGLES[child][k][1]][i] : 0.0;
                    } else {
                        next.v[k][i] = m[TETRAHEDRA[child][k][0]][TETRAHEDRA[child][k][1]][i];
                    }
                }
            }
            next.measure = item.measure / double(CHILDREN);
            next.level = item.level + 1;
        }
    }
}

/**
 * @brief Calls add(s, measure) for each simplex of a straight cell, whose
 *        nodes node(k, x) gives (n of them): 2D its triangles fanned from
 *        node 0; 3D the tetrahedra of cell_tetrahedra (mesh.h).
 */
template <typename Nodes, typename Add>
KOKKOS_INLINE_FUNCTION void straight_simplices(const uint32_t n, const Nodes & node, const Add & add_simplex) {
    double s[4][3] = {};
    if constexpr (N_DIM == 2) {
        node(0, s[0]);
        for (uint32_t k = 1; k + 1 < n; k++) {
            node(k, s[1]);
            node(k + 1, s[2]);
            const double area = 0.5 * Kokkos::fabs((s[1][0] - s[0][0]) * (s[2][1] - s[0][1]) -
                                                   (s[1][1] - s[0][1]) * (s[2][0] - s[0][0]));
            add_simplex(s, area);
        }
    } else {
        // Local faces of tetrahedra, pyramids, prisms and hexahedra (cell_local_faces), -1 past a face's end
        constexpr int TET[4][4] = {{0, 2, 1, -1}, {0, 1, 3, -1}, {0, 3, 2, -1}, {1, 2, 3, -1}};
        constexpr int PYRAMID[5][4] = {{0, 3, 2, 1}, {0, 1, 4, -1}, {1, 2, 4, -1}, {2, 3, 4, -1}, {3, 0, 4, -1}};
        constexpr int PRISM[5][4] = {{0, 2, 1, -1}, {3, 4, 5, -1}, {0, 1, 4, 3}, {1, 2, 5, 4}, {2, 0, 3, 5}};
        constexpr int HEX[6][4] = {{0, 3, 2, 1}, {4, 5, 6, 7}, {0, 1, 5, 4}, {1, 2, 6, 5}, {2, 3, 7, 6}, {3, 0, 4, 7}};
        const int n_faces = n == 4 ? 4 : n == 5 ? 5 : n == 6 ? 5 : 6;
        auto local = [&](const int f, const int k) {
            return n == 4 ? TET[f][k] : n == 5 ? PYRAMID[f][k] : n == 6 ? PRISM[f][k] : HEX[f][k];
        };
        double o[3] = {0.0, 0.0, 0.0}, x[3];
        for (uint32_t k = 0; k < n; k++) {
            node(k, x);
            for (int i = 0; i < 3; i++) o[i] += x[i];
        }
        for (int i = 0; i < 3; i++) o[i] /= static_cast<double>(n);
        auto add = [&](double t[4][3]) {
            double e[3][3];
            for (int k = 0; k < 3; k++) {
                for (int i = 0; i < 3; i++) e[k][i] = t[k + 1][i] - t[0][i];
            }
            const double cx = e[1][1] * e[2][2] - e[1][2] * e[2][1], cy = e[1][2] * e[2][0] - e[1][0] * e[2][2],
                         cz = e[1][0] * e[2][1] - e[1][1] * e[2][0];
            const double volume = Kokkos::fabs(e[0][0] * cx + e[0][1] * cy + e[0][2] * cz) / 6.0;
            add_simplex(t, volume);
        };
        for (int f = 0; f < n_faces; f++) {
            const int m = local(f, 3) < 0 ? 3 : 4;
            if (m == 3) {
                for (int i = 0; i < 3; i++) s[0][i] = o[i];
                for (int k = 0; k < 3; k++) node(static_cast<uint32_t>(local(f, k)), s[k + 1]);
                add(s);
                continue;
            }
            double c[3] = {0.0, 0.0, 0.0};
            for (int k = 0; k < m; k++) {
                node(static_cast<uint32_t>(local(f, k)), x);
                for (int i = 0; i < 3; i++) c[i] += x[i];
            }
            for (int i = 0; i < 3; i++) c[i] /= static_cast<double>(m);
            for (int k = 0; k < m; k++) {
                for (int i = 0; i < 3; i++) {
                    s[0][i] = o[i];
                    s[1][i] = c[i];
                }
                node(static_cast<uint32_t>(local(f, k)), s[2]);
                node(static_cast<uint32_t>(local(f, (k + 1) % m)), s[3]);
                add(s);
            }
        }
    }
}

/**
 * @brief Adds the integrals of the indicator and of 1 over a straight cell,
 *        whose nodes node(k, x) gives (n of them), over its simplices
 *        (straight_simplices).
 */
template <typename Distance, typename Nodes>
KOKKOS_INLINE_FUNCTION void straight_cell(const Shape<Distance> & shape, const bool axisymmetric, const uint32_t n,
                                          const Nodes & node, double & inside, double & total) {
    straight_simplices(n, node, [&](const double s[4][3], const double measure) {
        simplex(shape, axisymmetric, s, measure, inside, total);
    });
}

/**
 * @brief Adds the integrals of the indicator of {g < 0} (times r in
 *        axisymmetric runs) and of 1 over a simplex v of that measure, exactly
 *        for the linear function g with values g[k] at its vertices.
 */
KOKKOS_INLINE_FUNCTION void linear_simplex(const double v[4][3], const double g[4], const double measure,
                                           const bool axisymmetric, double & inside, double & total) {
    constexpr int n = N_DIM + 1;
    int negative = 0;
    for (int k = 0; k < n; k++) negative += g[k] < 0.0;
    double weight = measure;
    if (axisymmetric) {
        double c = 0.0;
        for (int k = 0; k < n; k++) c += v[k][1] / double(n);
        weight *= c;
    }
    total += weight;
    if (negative == 0) return;
    if (negative == n) {
        inside += weight;
        return;
    }
    if constexpr (N_DIM == 2) {
        // The polygon where g < 0: the triangle clipped along its edges
        double p[4][2];
        int m = 0;
        for (int k = 0; k < 3; k++) {
            const int j = (k + 1) % 3;
            if (g[k] < 0.0) {
                p[m][0] = v[k][0];
                p[m][1] = v[k][1];
                m++;
            }
            if ((g[k] < 0.0) != (g[j] < 0.0)) {
                const double s = g[k] / (g[k] - g[j]);
                p[m][0] = v[k][0] + s * (v[j][0] - v[k][0]);
                p[m][1] = v[k][1] + s * (v[j][1] - v[k][1]);
                m++;
            }
        }
        double area = 0.0, cy = 0.0;
        for (int k = 0; k < m; k++) {
            const int j = (k + 1) % m;
            const double cross = p[k][0] * p[j][1] - p[j][0] * p[k][1];
            area += cross;
            cy += (p[k][1] + p[j][1]) * cross;
        }
        area *= 0.5;
        const double a = Kokkos::fabs(area);
        inside += axisymmetric ? (area != 0.0 ? a * cy / (6.0 * area) : 0.0) : a;
    } else {
        auto volume = [](const double * a, const double * b, const double * c, const double * d) {
            const double e[3][3] = {{b[0] - a[0], b[1] - a[1], b[2] - a[2]},
                                    {c[0] - a[0], c[1] - a[1], c[2] - a[2]},
                                    {d[0] - a[0], d[1] - a[1], d[2] - a[2]}};
            return Kokkos::fabs(e[0][0] * (e[1][1] * e[2][2] - e[1][2] * e[2][1]) -
                                e[0][1] * (e[1][0] * e[2][2] - e[1][2] * e[2][0]) +
                                e[0][2] * (e[1][0] * e[2][1] - e[1][1] * e[2][0])) / 6.0;
        };
        if (negative == 1 || negative == 3) {
            // The corner on its own side, a tetrahedron cut off at the edges' roots
            const bool alone_negative = negative == 1;
            int a = 0;
            while ((g[a] < 0.0) != alone_negative) a++;
            double corner = 1.0;
            for (int k = 0; k < 4; k++) {
                if (k != a) corner *= g[a] / (g[a] - g[k]);
            }
            inside += weight * (alone_negative ? corner : 1.0 - corner);
            return;
        }
        // Two on each side: the prism between the negative edge and the cut
        int neg[2], pos[2], i_n = 0, i_p = 0;
        for (int k = 0; k < 4; k++) {
            if (g[k] < 0.0) {
                neg[i_n++] = k;
            } else {
                pos[i_p++] = k;
            }
        }
        double q[2][2][3];  // q[i][j]: the root on the edge from neg[i] to pos[j]
        for (int i = 0; i < 2; i++) {
            for (int j = 0; j < 2; j++) {
                const double s = g[neg[i]] / (g[neg[i]] - g[pos[j]]);
                for (int d = 0; d < 3; d++) q[i][j][d] = v[neg[i]][d] + s * (v[pos[j]][d] - v[neg[i]][d]);
            }
        }
        const double * a0 = v[neg[0]];
        const double * b0 = v[neg[1]];
        inside += volume(a0, q[0][0], q[0][1], q[1][1]) + volume(a0, q[0][0], q[1][0], q[1][1]) +
                  volume(a0, b0, q[1][0], q[1][1]);
    }
}

/**
 * @brief Adds the integrals of the indicator and of 1 over cone k of a curved
 *        cell, at(k, p, x) giving its point x at box coordinates p (face
 *        coordinates, then t last) and the volume per unit p, over the box
 *        [lo, hi] subdivided into 2^N_DIM children where the surface or edge
 *        may cross it, with 3 Gauss points per direction in each box kept.
 */
template <typename Distance, typename Cone>
KOKKOS_INLINE_FUNCTION void cone(const Shape<Distance> & shape, const bool axisymmetric, const Cone & at,
                                 const uint32_t k, const double lo0[3], const double hi0[3], double & inside,
                                 double & total) {
    struct Item {
        double lo[3], hi[3];
        int level;
    };
    Item stack[STACK];
    int top = 0;
    for (int i = 0; i < 3; i++) {
        stack[0].lo[i] = lo0[i];
        stack[0].hi[i] = hi0[i];
    }
    stack[0].level = 0;
    const double half_edge = 0.5 * shape.edge;
    while (top >= 0) {
        const Item item = stack[top--];
        double mid[3], x_mid[3];
        for (int i = 0; i < 3; i++) mid[i] = 0.5 * (item.lo[i] + item.hi[i]);
        at(k, mid, x_mid);
        double radius = 0.0;
        for (int corner = 0; corner < (1 << N_DIM); corner++) {
            double p[3] = {mid[0], mid[1], mid[2]}, x[3], d[3];
            for (int i = 0; i < N_DIM; i++) p[i] = (corner >> i) & 1 ? item.hi[i] : item.lo[i];
            at(k, p, x);
            for (int i = 0; i < 3; i++) d[i] = x[i] - x_mid[i];
            radius = region::detail::max(radius, detail::norm(d));
        }
        radius *= 1.25;  // the box's image is curved
        const double phi = shape.distance(x_mid);
        bool uniform = item.level >= MAX_LEVEL;
        if (shape.bounded) {
            if (Kokkos::fabs(phi) >= radius + half_edge) uniform = true;
        } else if (!uniform) {
            double slope = 0.0;
            for (int corner = 0; corner < (1 << N_DIM); corner++) {
                double p[3] = {mid[0], mid[1], mid[2]}, x[3], d[3];
                for (int i = 0; i < N_DIM; i++) p[i] = (corner >> i) & 1 ? item.hi[i] : item.lo[i];
                at(k, p, x);
                for (int i = 0; i < 3; i++) d[i] = x[i] - x_mid[i];
                if (detail::norm(d) > 0.0) {
                    slope = region::detail::max(slope, Kokkos::fabs(shape.distance(x) - phi) / detail::norm(d));
                }
            }
            uniform = Kokkos::fabs(phi) > 2.0 * slope * radius && half_edge == 0.0;
        }
        if (uniform) {
            // Gauss points: 3 per direction, exact for the cones' radial factor and smooth faces
            constexpr double G[3] = {-0.7745966692414834, 0.0, 0.7745966692414834};
            constexpr double W[3] = {5.0 / 9.0, 8.0 / 9.0, 5.0 / 9.0};
            constexpr int n = N_DIM == 2 ? 9 : 27;
            for (int q = 0; q < n; q++) {
                double p[3] = {mid[0], mid[1], mid[2]}, x[3] = {0.0, 0.0, 0.0};
                double w = 1.0;
                int rest = q;
                for (int i = 0; i < N_DIM; i++) {
                    const int g = rest % 3;
                    rest /= 3;
                    p[i] = mid[i] + 0.5 * (item.hi[i] - item.lo[i]) * G[g];
                    w *= 0.5 * (item.hi[i] - item.lo[i]) * W[g];
                }
                w *= at(k, p, x) * (axisymmetric ? x[1] : 1.0);
                inside += w * indicator(shape.distance(x), shape.edge);
                total += w;
            }
            continue;
        }
        for (int child = (1 << N_DIM) - 1; child >= 0; child--) {
            Item & next = stack[++top];
            for (int i = 0; i < 3; i++) {
                next.lo[i] = item.lo[i];
                next.hi[i] = item.hi[i];
            }
            for (int i = 0; i < N_DIM; i++) {
                if ((child >> i) & 1) {
                    next.lo[i] = mid[i];
                } else {
                    next.hi[i] = mid[i];
                }
            }
            next.level = item.level + 1;
        }
    }
}

} // namespace region::sampler

#endif // REGION_SAMPLER_H

/**
 * @file lagrange.cpp
 * @brief VTK Lagrange cells: node orders, basis and cell geometry.
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 */

#include "lagrange.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>

#include "common_io.h"
#include "curved.h"
#include "mesh.h"

namespace lagrange {

namespace {

using Vec = std::array<int64_t, 3>;

Vec operator+(const Vec & a, const Vec & b) { return {a[0] + b[0], a[1] + b[1], a[2] + b[2]}; }
Vec operator-(const Vec & a, const Vec & b) { return {a[0] - b[0], a[1] - b[1], a[2] - b[2]}; }
Vec operator*(int64_t s, const Vec & a) { return {s * a[0], s * a[1], s * a[2]}; }
Vec over(const Vec & a, int64_t n) { return {a[0] / n, a[1] / n, a[2] / n}; }

struct Nodes {
    std::vector<Lattice> out;
    void push(const Vec & v) { out.push_back({uint32_t(v[0]), uint32_t(v[1]), uint32_t(v[2])}); }
    void edge(const Vec & a, const Vec & b, int64_t n) {
        for (int64_t i = 1; i < n; i++) push(a + i * over(b - a, n));
    }
};

void triangle(Nodes & nodes, const Vec & a, const Vec & b, const Vec & c, int64_t n);

/** @brief The nodes strictly inside the triangle (a, b, c) of order n, in VTK's order. */
void triangle_interior(Nodes & nodes, const Vec & a, const Vec & b, const Vec & c, int64_t n) {
    if (n < 3) return;
    const Vec db = over(b - a, n), dc = over(c - a, n);
    const Vec a1 = a + db + dc;
    triangle(nodes, a1, a1 + (n - 3) * db, a1 + (n - 3) * dc, n - 3);
}

void triangle(Nodes & nodes, const Vec & a, const Vec & b, const Vec & c, int64_t n) {
    nodes.push(a);
    if (n == 0) return;
    nodes.push(b);
    nodes.push(c);
    nodes.edge(a, b, n);
    nodes.edge(b, c, n);
    nodes.edge(c, a, n);
    triangle_interior(nodes, a, b, c, n);
}

void tetrahedron(Nodes & nodes, const Vec & a, const Vec & b, const Vec & c, const Vec & d, int64_t n) {
    nodes.push(a);
    if (n == 0) return;
    nodes.push(b);
    nodes.push(c);
    nodes.push(d);
    nodes.edge(a, b, n);
    nodes.edge(b, c, n);
    nodes.edge(c, a, n);
    nodes.edge(a, d, n);
    nodes.edge(b, d, n);
    nodes.edge(c, d, n);
    triangle_interior(nodes, a, b, d, n);
    triangle_interior(nodes, c, d, b, n);
    triangle_interior(nodes, a, d, c, n);
    triangle_interior(nodes, a, c, b, n);
    if (n < 4) return;
    const Vec db = over(b - a, n), dc = over(c - a, n), dd = over(d - a, n);
    const Vec a1 = a + db + dc + dd;
    tetrahedron(nodes, a1, a1 + (n - 4) * db, a1 + (n - 4) * dc, a1 + (n - 4) * dd, n - 4);
}

std::vector<Lattice> make_lattice(Shape shape, int64_t n) {
    Nodes nodes;
    const Vec o{0, 0, 0}, x{n, 0, 0}, y{0, n, 0}, z{0, 0, n};
    switch (shape) {
        case Shape::TRIANGLE:
            triangle(nodes, o, x, y, n);
            break;
        case Shape::TETRAHEDRON:
            tetrahedron(nodes, o, x, y, z, n);
            break;
        case Shape::QUADRILATERAL:
        case Shape::HEXAHEDRON: {
            const bool solid = shape == Shape::HEXAHEDRON;
            const Vec v[8] = {o, x, x + y, y, z, x + z, x + y + z, y + z};
            for (int k = 0; k < (solid ? 8 : 4); k++) nodes.push(v[k]);
            for (int top = 0; top < (solid ? 2 : 1); top++) {
                const int b = 4 * top;
                nodes.edge(v[b], v[b + 1], n);
                nodes.edge(v[b + 1], v[b + 2], n);
                nodes.edge(v[b + 3], v[b + 2], n);
                nodes.edge(v[b], v[b + 3], n);
            }
            if (solid) {
                for (int k = 0; k < 4; k++) nodes.edge(v[k], v[k + 4], n);
            }
            // Faces (hexahedra: x = 0, x = n, y = 0, y = n, z = 0, z = n), each with its first free axis fastest
            const auto face = [&](int normal, int64_t at) {
                const int u = normal == 0 ? 1 : 0, w = normal == 2 ? 1 : 2;
                for (int64_t j = 1; j < n; j++) {
                    for (int64_t i = 1; i < n; i++) {
                        Vec p{0, 0, 0};
                        p[size_t(normal)] = at;
                        p[size_t(u)] = i;
                        p[size_t(w)] = j;
                        nodes.push(p);
                    }
                }
            };
            if (solid) {
                for (int normal = 0; normal < 3; normal++) {
                    face(normal, 0);
                    face(normal, n);
                }
                for (int64_t k = 1; k < n; k++) {
                    for (int64_t j = 1; j < n; j++) {
                        for (int64_t i = 1; i < n; i++) nodes.push({i, j, k});
                    }
                }
            } else {
                face(2, 0);
            }
            break;
        }
        case Shape::WEDGE: {
            const Vec v[6] = {o, x, y, z, x + z, y + z};
            for (const Vec & p : v) nodes.push(p);
            nodes.edge(v[0], v[1], n);
            nodes.edge(v[1], v[2], n);
            nodes.edge(v[2], v[0], n);
            nodes.edge(v[3], v[4], n);
            nodes.edge(v[4], v[5], n);
            nodes.edge(v[5], v[3], n);
            nodes.edge(v[0], v[3], n);
            nodes.edge(v[1], v[4], n);
            nodes.edge(v[2], v[5], n);
            // Triangle faces and body layers: their interior nodes row by row, unlike triangles
            const auto layer = [&](int64_t k) {
                for (int64_t j = 1; j < n; j++) {
                    for (int64_t i = 1; i + j < n; i++) nodes.push({i, j, k});
                }
            };
            layer(0);
            layer(n);
            // Quadrilateral faces along the bottom edges 0-1, 1-2, 2-0: the edge's direction fastest
            const std::array<std::array<int, 2>, 3> edges = {{{0, 1}, {1, 2}, {2, 0}}};
            for (const auto & e : edges) {
                for (int64_t k = 1; k < n; k++) {
                    for (int64_t i = 1; i < n; i++) nodes.push(v[e[0]] + i * over(v[e[1]] - v[e[0]], n) + Vec{0, 0, k});
                }
            }
            for (int64_t k = 1; k < n; k++) layer(k);
            break;
        }
    }
    return nodes.out;
}

/** @brief The exponents of the monomials spanning the cell's polynomial space of this order. */
std::vector<std::array<uint32_t, 3>> exponents(Shape shape, uint32_t order) {
    std::vector<std::array<uint32_t, 3>> e;
    const bool solid = shape == Shape::TETRAHEDRON || shape == Shape::HEXAHEDRON || shape == Shape::WEDGE;
    for (uint32_t c = 0; c <= (solid ? order : 0); c++) {
        for (uint32_t b = 0; b <= order; b++) {
            for (uint32_t a = 0; a <= order; a++) {
                const bool fits = shape == Shape::TRIANGLE || shape == Shape::WEDGE ? a + b <= order
                                  : shape == Shape::TETRAHEDRON                     ? a + b + c <= order
                                                                                    : true;
                if (fits) e.push_back({a, b, c});
            }
        }
    }
    return e;
}

double monomial(const std::array<uint32_t, 3> & e, const Point & x) {
    return std::pow(x[0], double(e[0])) * std::pow(x[1], double(e[1])) * std::pow(x[2], double(e[2]));
}

} // namespace

uint8_t vtk_type(Shape shape) {
    switch (shape) {
        case Shape::TRIANGLE: return 69;
        case Shape::QUADRILATERAL: return 70;
        case Shape::TETRAHEDRON: return 71;
        case Shape::HEXAHEDRON: return 72;
        case Shape::WEDGE: return 73;
    }
    return 0;
}

uint32_t n_nodes(Shape shape, uint32_t order) {
    const uint32_t p = order;
    switch (shape) {
        case Shape::TRIANGLE: return (p + 1) * (p + 2) / 2;
        case Shape::QUADRILATERAL: return (p + 1) * (p + 1);
        case Shape::TETRAHEDRON: return (p + 1) * (p + 2) * (p + 3) / 6;
        case Shape::HEXAHEDRON: return (p + 1) * (p + 1) * (p + 1);
        case Shape::WEDGE: return (p + 1) * (p + 1) * (p + 2) / 2;
    }
    return 0;
}

std::vector<Lattice> lattice(Shape shape, uint32_t order) {
    if (order == 0) throw std::invalid_argument("lagrange::lattice: order 0");
    return make_lattice(shape, int64_t(order));
}

std::vector<Point> reference_nodes(Shape shape, uint32_t order) {
    std::vector<Point> out;
    for (const Lattice & l : lattice(shape, order)) {
        out.push_back({double(l[0]) / double(order), double(l[1]) / double(order), double(l[2]) / double(order)});
    }
    return out;
}

std::vector<double> basis(Shape shape, uint32_t order, std::span<const Point> points) {
    const std::vector<Point> nodes = reference_nodes(shape, order);
    const auto e = exponents(shape, order);
    const size_t n = nodes.size();
    // phi(x) solves V^T phi = m(x), V(i, k) = monomial k at node i: an LU of V^T with partial pivoting
    std::vector<double> a(n * n);
    for (size_t i = 0; i < n; i++) {
        for (size_t k = 0; k < n; k++) a[k * n + i] = monomial(e[k], nodes[i]);
    }
    std::vector<size_t> pivot(n);
    for (size_t j = 0; j < n; j++) {
        size_t best = j;
        for (size_t i = j + 1; i < n; i++) {
            if (std::fabs(a[i * n + j]) > std::fabs(a[best * n + j])) best = i;
        }
        pivot[j] = best;
        if (best != j) {
            for (size_t k = 0; k < n; k++) std::swap(a[j * n + k], a[best * n + k]);
        }
        for (size_t i = j + 1; i < n; i++) {
            a[i * n + j] /= a[j * n + j];
            for (size_t k = j + 1; k < n; k++) a[i * n + k] -= a[i * n + j] * a[j * n + k];
        }
    }
    std::vector<double> out(points.size() * n);
    std::vector<double> rhs(n);
    for (size_t p = 0; p < points.size(); p++) {
        for (size_t k = 0; k < n; k++) rhs[k] = monomial(e[k], points[p]);
        for (size_t j = 0; j < n; j++) std::swap(rhs[j], rhs[pivot[j]]);
        for (size_t i = 0; i < n; i++) {
            for (size_t k = 0; k < i; k++) rhs[i] -= a[i * n + k] * rhs[k];
        }
        for (size_t i = n; i-- > 0;) {
            for (size_t k = i + 1; k < n; k++) rhs[i] -= a[i * n + k] * rhs[k];
            rhs[i] /= a[i * n + i];
        }
        std::copy(rhs.begin(), rhs.end(), out.begin() + std::ptrdiff_t(p * n));
    }
    return out;
}

Shape shape_of(uint32_t dim, uint32_t n_vertices) {
    if (dim == 2) {
        if (n_vertices == 3) return Shape::TRIANGLE;
        if (n_vertices == 4) return Shape::QUADRILATERAL;
    } else {
        switch (n_vertices) {
            case 4: return Shape::TETRAHEDRON;
            case 5:
            case 8: return Shape::HEXAHEDRON;
            case 6: return Shape::WEDGE;
            default: break;
        }
    }
    throw std::runtime_error("no Lagrange cell for a " + std::to_string(dim) + "D cell with " +
                             std::to_string(n_vertices) + " vertices.");
}

namespace {

constexpr uint8_t VTK_QUADRATIC_PYRAMID = 27;

using Pairs = std::vector<std::array<uint8_t, 2>>;
using Quads = std::vector<std::array<uint8_t, 4>>;

/** @brief A curved cell as a quadratic VTK cell: its edges and quadrilateral faces with a middle node. */
struct Quadratic {
    uint8_t type;
    Pairs edges;
    Quads faces;
    bool body;
};

const Quadratic & quadratic(uint32_t n_corners) {
    static const Quadratic triangle{vtk_type(Shape::TRIANGLE), {{0, 1}, {1, 2}, {2, 0}}, {}, false};
    static const Quadratic quad{vtk_type(Shape::QUADRILATERAL), {{0, 1}, {1, 2}, {3, 2}, {0, 3}}, {}, true};
    static const Quadratic tetrahedron{
        vtk_type(Shape::TETRAHEDRON), {{0, 1}, {1, 2}, {0, 2}, {0, 3}, {1, 3}, {2, 3}}, {}, false};
    static const Quadratic pyramid{
        VTK_QUADRATIC_PYRAMID, {{0, 1}, {1, 2}, {2, 3}, {3, 0}, {0, 4}, {1, 4}, {2, 4}, {3, 4}}, {}, false};
    static const Quadratic wedge{vtk_type(Shape::WEDGE),
                                 {{0, 1}, {1, 2}, {2, 0}, {3, 4}, {4, 5}, {5, 3}, {0, 3}, {1, 4}, {2, 5}},
                                 {{0, 1, 4, 3}, {1, 2, 5, 4}, {2, 0, 3, 5}},
                                 false};
    static const Quadratic hexahedron{
        vtk_type(Shape::HEXAHEDRON),
        {{0, 1}, {1, 2}, {3, 2}, {0, 3}, {4, 5}, {5, 6}, {7, 6}, {4, 7}, {0, 4}, {1, 5}, {2, 6}, {3, 7}},
        {{0, 3, 7, 4}, {1, 2, 6, 5}, {0, 1, 5, 4}, {3, 2, 6, 7}, {0, 1, 2, 3}, {4, 5, 6, 7}},
        true};
    if (N_DIM == 2) return n_corners == 3 ? triangle : quad;
    switch (n_corners) {
        case 4: return tetrahedron;
        case 5: return pyramid;
        case 6: return wedge;
        default: return hexahedron;
    }
}

/** @brief Reference coordinates of node k of a face with n nodes (curved::Geometry::eval). */
std::array<double, 2> face_reference(uint32_t n, uint32_t k) {
    static constexpr double TRIANGLE[3][2] = {{0.0, 0.0}, {1.0, 0.0}, {0.0, 1.0}};
    static constexpr double QUAD[4][2] = {{-1.0, -1.0}, {1.0, -1.0}, {1.0, 1.0}, {-1.0, 1.0}};
    if (n == 2) return {k == 0 ? -1.0 : 1.0, 0.0};
    if (n == 3) return {TRIANGLE[k][0], TRIANGLE[k][1]};
    return {QUAD[k][0], QUAD[k][1]};
}

/**
 * @brief The point of cell c's curved geometry at the centroid of these
 *        nodes (an edge's two or a quadrilateral face's four): on a face of the
 *        cell holding them all (a curved one where there is one), else their
 *        average.
 */
Point face_point(const Mesh & mesh, const curved::Geometry & geometry, uint32_t c, const std::vector<uint32_t> & nodes,
                 const Point & average) {
    int64_t chosen = -1;
    std::array<double, 2> xi = {0.0, 0.0};
    for (uint32_t j = 0; j < mesh.h_n_faces_of_cell(c); j++) {
        const uint32_t f = mesh.h_face_of_cell(c, j);
        if (chosen >= 0 && !geometry.face_is_curved(f)) continue;
        const uint32_t n = mesh.h_n_nodes_of_face(f);
        std::array<double, 2> sum = {0.0, 0.0};
        uint32_t found = 0;
        for (uint32_t k = 0; k < n; k++) {
            if (std::find(nodes.begin(), nodes.end(), mesh.h_node_of_face(f, k)) == nodes.end()) continue;
            const auto r = face_reference(n, k);
            sum = {sum[0] + r[0], sum[1] + r[1]};
            found++;
        }
        if (found != nodes.size()) continue;
        chosen = f;
        xi = {sum[0] / double(found), sum[1] / double(found)};
        if (geometry.face_is_curved(f)) break;
    }
    if (chosen < 0) return average;
    curved::Vec3 x;
    geometry.eval(static_cast<uint32_t>(chosen), xi.data(), x, nullptr);
    return x;
}

} // namespace

uint8_t quadratic_vtk_type(uint32_t n_corners) { return quadratic(n_corners).type; }

std::vector<Point> quadratic_nodes(const Mesh & mesh, const curved::Geometry & geometry, uint32_t c,
                                   const std::vector<uint32_t> & corners) {
    const Quadratic & q = quadratic(uint32_t(corners.size()));
    std::vector<Point> x(corners.size());
    for (size_t k = 0; k < corners.size(); k++) {
        x[k] = {0.0, 0.0, 0.0};
        FOR_I_DIM x[k][i] = double(mesh.h_node_coords(corners[k], i));
    }
    auto average = [&](const auto & local) {
        Point sum = {0.0, 0.0, 0.0};
        for (uint8_t k : local) {
            for (size_t i = 0; i < 3; i++) sum[i] += x[k][i];
        }
        for (double & v : sum) v /= double(local.size());
        return sum;
    };
    auto nodes_of = [&](const auto & local) {
        std::vector<uint32_t> nodes;
        for (uint8_t k : local) nodes.push_back(corners[k]);
        return nodes;
    };
    std::vector<Point> out;
    Point body = {0.0, 0.0, 0.0};
    auto accumulate = [&](const Point & p, double w) {
        for (size_t i = 0; i < 3; i++) body[i] += w * p[i];
    };
    const bool solid = N_DIM == 3;
    for (const auto & e : q.edges) {
        out.push_back(face_point(mesh, geometry, c, nodes_of(e), average(e)));
        accumulate(out.back(), solid ? -0.25 : 0.5);
    }
    for (const auto & f : q.faces) {
        out.push_back(face_point(mesh, geometry, c, nodes_of(f), average(f)));
        accumulate(out.back(), 0.5);
    }
    if (q.body) {
        for (const Point & corner : x) accumulate(corner, solid ? 0.125 : -0.25);
        out.push_back(body);
    }
    return out;
}

uint32_t cell_geometry(const Mesh & mesh, const curved::Geometry * curves, uint32_t c, std::vector<Point> & nodes) {
    const uint32_t n = mesh.h_n_nodes_of_cell(c);
    std::vector<uint32_t> corners(n);
    for (uint32_t k = 0; k < n; k++) corners[k] = mesh.h_node_of_cell(c, vtk_local_node(n, k));
    const bool pyramid = N_DIM == 3 && n == 5;
    nodes.clear();
    for (uint32_t k = 0; k < (pyramid ? 8u : n); k++) {
        Point x = {0.0, 0.0, 0.0};
        FOR_I_DIM x[i] = double(mesh.h_node_coords(corners[std::min(k, n - 1)], i));
        nodes.push_back(x);
    }
    if (pyramid || !curves || !curves->cell_is_curved(c)) return 1;
    for (const Point & x : quadratic_nodes(mesh, *curves, c, corners)) nodes.push_back(x);
    return 2;
}

} // namespace lagrange

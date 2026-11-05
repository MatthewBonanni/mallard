/**
 * @file region.cpp
 * @author Matthew Bonanni (mbonanni001@gmail.com)
 * @brief Regions from their definitions, and the fraction of each cell inside them.
 * @version 0.1
 * @date 2026-10-15
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 *
 */

#include "region.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <map>
#include <utility>

#include "comm.h"
#include "input.h"
#include "mesh.h"

namespace region {

namespace {

using Vec = std::array<double, 3>;
constexpr double INF = std::numeric_limits<double>::infinity();
constexpr double PI = 3.14159265358979323846;

double dot(const Vec & a, const Vec & b) { return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]; }
[[maybe_unused]] Vec cross(const Vec & a, const Vec & b) {
    return {a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]};
}
double norm(const Vec & a) { return std::sqrt(dot(a, a)); }

} // namespace

std::vector<uint32_t> owned_zone_faces(const Mesh & mesh, const std::vector<std::string> & zones) {
    std::vector<uint32_t> faces;
    for (const FaceZone & z : mesh.h_face_zones()) {
        if (z.get_type() != FaceZoneType::BOUNDARY) continue;
        if (std::find(zones.begin(), zones.end(), z.get_name()) == zones.end()) continue;
        for (uint32_t i = 0; i < z.n_faces(); i++) {
            const uint32_t f = z.h_faces(i);
            if (static_cast<uint32_t>(mesh.h_cells_of_face(f, 0)) < mesh.n_owned()) faces.push_back(f);
        }
    }
    return faces;
}

Region::Region(Definition d) :
    m_nodes(std::move(d.nodes)), m_vertices(std::move(d.vertices)), m_programs(std::move(d.programs)),
    m_filled(std::move(d.filled)), m_pieces(std::move(d.pieces)), m_root(d.root), m_bounded(d.bounded),
    m_closed(d.closed), m_dimension(d.dimension), m_resolution(d.resolution), m_name(std::move(d.name)),
    m_summary(std::move(d.summary)) {
    if (analytic()) {
        m_host = host(m_filled[m_root]);
        m_host.edge = d.edge;
    }
}

RegionView<Kokkos::HostSpace> Region::host(const Filled & shapes) const {
    RegionView<Kokkos::HostSpace> r;
    r.nodes = Kokkos::View<const Node *, Kokkos::HostSpace>(m_nodes.data() + shapes.first, shapes.count);
    r.vertices = Kokkos::View<const double *, Kokkos::HostSpace>(m_vertices.data(), m_vertices.size());
    std::vector<const kexpr::Program *> programs;
    for (const auto & p : m_programs) programs.push_back(&p);
    r.programs = kexpr::ProgramTable<Kokkos::HostSpace>(programs);
    return r;
}

RegionView<> Region::device() const {
    using Space = Kokkos::DefaultExecutionSpace::memory_space;
    RegionView<> r;
    const Filled & root = m_filled[m_root];
    Kokkos::View<Node *, Space> nodes("region_nodes", root.count);
    Kokkos::deep_copy(nodes, Kokkos::View<const Node *, Kokkos::HostSpace, Kokkos::MemoryUnmanaged>(
                                 m_nodes.data() + root.first, root.count));
    Kokkos::View<double *, Space> vertices("region_vertices", m_vertices.size());
    Kokkos::deep_copy(vertices, Kokkos::View<const double *, Kokkos::HostSpace, Kokkos::MemoryUnmanaged>(
                                    m_vertices.data(), m_vertices.size()));
    std::vector<const kexpr::Program *> programs;
    for (const auto & p : m_programs) programs.push_back(&p);
    r.nodes = nodes;
    r.vertices = vertices;
    r.programs = kexpr::ProgramTable<>(programs);
    r.edge = m_host.edge;
    return r;
}

namespace {

/**
 * @brief Integral of the indicator (times r in axisymmetric runs) and of 1
 *        over a simplex of N_DIM + 1 vertices, subdivided into 2^N_DIM
 *        children where the surface or edge may cross it.
 */
struct Sampler {
    const RegionView<Kokkos::HostSpace> & region;
    bool bounded;
    bool axisymmetric;
    int max_level;

    void integrate(const std::array<Vec, 4> & v, const double measure, const int level, double & inside,
                   double & total) const {
        constexpr size_t n_vertices = N_DIM + 1;
        Vec c = {0.0, 0.0, 0.0};
        for (size_t k = 0; k < n_vertices; k++) {
            for (size_t i = 0; i < 3; i++) c[i] += v[k][i] / double(n_vertices);
        }
        double radius = 0.0;
        for (size_t k = 0; k < n_vertices; k++) {
            Vec d;
            for (size_t i = 0; i < 3; i++) d[i] = v[k][i] - c[i];
            radius = std::max(radius, norm(d));
        }
        const double phi = region.distance(c.data());
        const double half_edge = 0.5 * region.edge;
        bool uniform = level >= max_level;
        if (bounded) {
            if (std::abs(phi) >= radius + half_edge) uniform = true;
            if (half_edge > 0.0 && radius < 0.125 * region.edge && std::abs(phi) < half_edge - radius) uniform = true;
        } else if (!uniform) {
            // Implicit regions: subdivide unless f keeps its sign over the simplex by twice its slope there
            double slope = 0.0;
            for (size_t k = 0; k < n_vertices; k++) {
                Vec d;
                for (size_t i = 0; i < 3; i++) d[i] = v[k][i] - c[i];
                slope = std::max(slope, std::abs(region.distance(v[k].data()) - phi) / norm(d));
            }
            uniform = std::abs(phi) > 2.0 * slope * radius && half_edge == 0.0;
        }
        if (uniform) {
            const double w = measure * (axisymmetric ? c[1] : 1.0);
            inside += w * indicator(phi, region.edge);
            total += w;
            return;
        }
        // Midpoint subdivision: 4 triangles, or 8 tetrahedra (4 corners and an octahedron split along a diagonal)
        auto mid = [&](const size_t a, const size_t b) {
            Vec m;
            for (size_t i = 0; i < 3; i++) m[i] = 0.5 * (v[a][i] + v[b][i]);
            return m;
        };
        if constexpr (N_DIM == 2) {
            const Vec m01 = mid(0, 1), m12 = mid(1, 2), m02 = mid(0, 2);
            const std::array<std::array<Vec, 4>, 4> children = {{{v[0], m01, m02, {}}, {m01, v[1], m12, {}},
                                                                  {m02, m12, v[2], {}}, {m01, m12, m02, {}}}};
            for (const auto & child : children) integrate(child, 0.25 * measure, level + 1, inside, total);
        } else {
            const Vec m01 = mid(0, 1), m02 = mid(0, 2), m03 = mid(0, 3), m12 = mid(1, 2), m13 = mid(1, 3),
                      m23 = mid(2, 3);
            const std::array<std::array<Vec, 4>, 8> children = {{{v[0], m01, m02, m03}, {m01, v[1], m12, m13},
                                                                 {m02, m12, v[2], m23}, {m03, m13, m23, v[3]},
                                                                 {m01, m23, m02, m03}, {m01, m23, m03, m13},
                                                                 {m01, m23, m13, m12}, {m01, m23, m12, m02}}};
            for (const auto & child : children) integrate(child, 0.125 * measure, level + 1, inside, total);
        }
    }
};

/** @brief The fractions of a filled region of shapes. */
std::vector<double> shape_fractions(const Mesh & mesh, const Region & region, const RegionView<Kokkos::HostSpace> & view) {
    const Sampler sampler{view, region.bounded(), mesh.axisymmetric, N_DIM == 2 ? 8 : 5};
    std::vector<double> fractions(mesh.n_cells, 0.0);
    using HostSpace = Kokkos::DefaultHostExecutionSpace;
    Kokkos::parallel_for("region_cell_fractions",
                         Kokkos::RangePolicy<HostSpace, Kokkos::Schedule<Kokkos::Dynamic>>(0, mesh.n_cells),
                         [&](const uint32_t i_cell) {
        double inside = 0.0, total = 0.0;
        if constexpr (N_DIM == 2) {
            const uint32_t n_nodes = mesh.h_n_nodes_of_cell(i_cell);
            auto node = [&](const uint32_t k) {
                const uint32_t n = mesh.h_node_of_cell(i_cell, k);
                return Vec{double(mesh.h_node_coords(n, 0)), double(mesh.h_node_coords(n, 1)), 0.0};
            };
            const Vec v0 = node(0);
            for (uint32_t k = 1; k + 1 < n_nodes; k++) {
                const Vec v1 = node(k), v2 = node(k + 1);
                const double area = 0.5 * std::abs((v1[0] - v0[0]) * (v2[1] - v0[1]) - (v1[1] - v0[1]) * (v2[0] - v0[0]));
                sampler.integrate({v0, v1, v2, Vec{}}, area, 0, inside, total);
            }
        } else {
            std::vector<std::array<std::array<double, 3>, 4>> tets;
            mesh.h_cell_tetrahedra(i_cell, tets);
            for (const auto & t : tets) {
                Vec e[3];
                for (size_t k = 0; k < 3; k++) {
                    for (size_t i = 0; i < 3; i++) e[k][i] = t[k + 1][i] - t[0][i];
                }
                const double volume = std::abs(dot(e[0], cross(e[1], e[2]))) / 6.0;
                sampler.integrate({t[0], t[1], t[2], t[3]}, volume, 0, inside, total);
            }
        }
        fractions[i_cell] = total > 0.0 ? inside / total : 0.0;
    });
    if (view.programs.loops) {
        const uint32_t stopped = comm::allreduce(view.programs.stopped(), comm::Op::MAX);
        if (stopped > 0) throw InputError(region.programs()[stopped - 1].runaway_error());
    }
    return fractions;
}

/** @brief Winding number about x of closed segments (2D: a, b) or triangles (3D: a, b, c), packed by coordinates. */
double winding(const std::vector<double> & elements, const Vec & x) {
    double w = 0.0;
    if constexpr (N_DIM == 2) {
        for (size_t i = 0; i < elements.size(); i += 4) {
            const double ax = elements[i] - x[0], ay = elements[i + 1] - x[1];
            const double bx = elements[i + 2] - x[0], by = elements[i + 3] - x[1];
            w += std::atan2(ax * by - ay * bx, ax * bx + ay * by);
        }
        return w / (2.0 * PI);
    } else {
        for (size_t i = 0; i < elements.size(); i += 9) {
            Vec v[3];
            for (size_t k = 0; k < 3; k++) {
                for (size_t j = 0; j < 3; j++) v[k][j] = elements[i + 3 * k + j] - x[j];
            }
            const double a = norm(v[0]), b = norm(v[1]), c = norm(v[2]);
            const double num = dot(v[0], cross(v[1], v[2]));
            const double den = a * b * c + dot(v[0], v[1]) * c + dot(v[0], v[2]) * b + dot(v[1], v[2]) * a;
            w += 2.0 * std::atan2(num, den);
        }
        return w / (4.0 * PI);
    }
}

/**
 * @brief 1 in the cells inside closed boundary zones, else 0. The zones lie on
 *        the mesh's boundary, so each connected part of the mesh is entirely
 *        inside or outside: one winding number per part.
 */
std::vector<double> inside_fractions(const Mesh & mesh, const std::vector<std::string> & zones) {
    std::vector<double> elements;
    for (const uint32_t f : owned_zone_faces(mesh, zones)) {
        const uint32_t a = mesh.h_offsets_nodes_of_face(f), b = mesh.h_offsets_nodes_of_face(f + 1);
        auto push = [&](const uint32_t k) {
            for (size_t i = 0; i < N_DIM; i++) elements.push_back(double(mesh.h_node_coords(mesh.h_nodes_of_face(k), i)));
        };
        if (N_DIM == 2) {
            push(a);
            push(a + 1);
        } else {
            for (uint32_t k = a + 1; k + 1 < b; k++) {
                push(a);
                push(k);
                push(k + 1);
            }
        }
    }
    if (!mesh.h_global_cell_id.empty()) elements = comm::allgatherv(elements);
    std::vector<uint32_t> part(mesh.n_cells);
    for (uint32_t c = 0; c < mesh.n_cells; c++) part[c] = c;
    auto find = [&](uint32_t c) {
        while (part[c] != c) c = part[c] = part[part[c]];
        return c;
    };
    for (uint32_t f = 0; f < mesh.n_faces; f++) {
        const int32_t a = mesh.h_cells_of_face(f, 0), b = mesh.h_cells_of_face(f, 1);
        if (a >= 0 && b >= 0) part[find(uint32_t(a))] = find(uint32_t(b));
    }
    std::vector<double> fractions(mesh.n_cells, 0.0);
    std::map<uint32_t, double> inside;
    for (uint32_t c = 0; c < mesh.n_cells; c++) {
        const uint32_t root = find(c);
        auto it = inside.find(root);
        if (it == inside.end()) {
            Vec x = {0.0, 0.0, 0.0};
            for (size_t i = 0; i < N_DIM; i++) x[i] = double(mesh.h_cell_coords(c, i));
            it = inside.emplace(root, std::abs(winding(elements, x)) > 0.5 ? 1.0 : 0.0).first;
        }
        fractions[c] = it->second;
    }
    return fractions;
}

std::vector<double> fractions_of(const Mesh & mesh, const Region & region, const uint32_t i) {
    const Filled & f = region.filled()[i];
    switch (f.kind) {
        case Filled::Kind::SHAPES: {
            RegionView<Kokkos::HostSpace> view = region.host(f);
            if (i == region.root()) view.edge = region.edge();
            return shape_fractions(mesh, region, view);
        }
        case Filled::Kind::ALL:
            return std::vector<double>(mesh.n_cells, 1.0);
        case Filled::Kind::CELL_ZONE: {
            const auto & names = mesh.cell_zone_names;
            const auto zone = static_cast<uint32_t>(std::find(names.begin(), names.end(), f.zones[0]) - names.begin());
            std::vector<double> out(mesh.n_cells);
            for (uint32_t c = 0; c < mesh.n_cells; c++) out[c] = mesh.h_cell_zone[c] == zone ? 1.0 : 0.0;
            return out;
        }
        case Filled::Kind::INSIDE:
            return inside_fractions(mesh, f.zones);
        default:
            break;
    }
    // Booleans with regions of the mesh, whose fractions are 0 or 1: exact with at most one part of shapes
    std::vector<double> out = fractions_of(mesh, region, f.parts[0]);
    for (size_t k = 1; k < f.parts.size(); k++) {
        const std::vector<double> g = fractions_of(mesh, region, f.parts[k]);
        for (uint32_t c = 0; c < mesh.n_cells; c++) {
            if (f.kind == Filled::Kind::UNION) {
                out[c] = std::max(out[c], g[c]);
            } else if (f.kind == Filled::Kind::INTERSECTION) {
                out[c] *= g[c];
            } else {
                out[c] *= 1.0 - g[c];
            }
        }
    }
    return out;
}

} // namespace

std::vector<double> cell_fractions(const Mesh & mesh, const Region & region) {
    if (region.dimension() != N_DIM) throw std::logic_error("cell_fractions: the region is not filled.");
    return fractions_of(mesh, region, region.root());
}

std::vector<double> cell_fractions(const Mesh & mesh, const Region & region, const uint32_t filled) {
    return fractions_of(mesh, region, filled);
}

namespace {

using Box = std::array<Vec, 2>;
constexpr Box EVERYWHERE = {Vec{-INF, -INF, -INF}, Vec{INF, INF, INF}};

/** @brief A box around a primitive: its local box, through its frame. */
Box primitive_bounds(const Node & n, const std::vector<double> & vertices) {
    Vec lo, hi;
    const double * p = n.p;
    switch (n.kind) {
        case Kind::SPHERE: lo = {-p[0], -p[0], -p[0]}; break;
        case Kind::ELLIPSOID: lo = {-p[0], -p[1], -p[2]}; break;
        case Kind::CYLINDER: lo = {-p[0], -p[0], -p[2]}; break;
        case Kind::CONE: lo = {-std::max(p[1], p[2]), -std::max(p[1], p[2]), -p[0]}; break;
        case Kind::BOX: lo = {-p[0], -p[1], -p[2]}; break;
        case Kind::TORUS: lo = {-(p[0] + p[1]), -(p[0] + p[1]), -p[1]}; break;
        case Kind::POLYGON: {
            lo = {INF, INF, -INF};
            hi = {-INF, -INF, INF};
            for (uint32_t k = 0; k < n.count; k++) {
                for (size_t i = 0; i < 2; i++) {
                    lo[i] = std::min(lo[i], vertices[2 * (n.first + k) + i]);
                    hi[i] = std::max(hi[i], vertices[2 * (n.first + k) + i]);
                }
            }
            break;
        }
        default: return EVERYWHERE;
    }
    if (n.kind != Kind::POLYGON) hi = {-lo[0], -lo[1], -lo[2]};
    // 2D: a local z axis along the world's leaves x and y bounded however far it extends
    const bool planar = n.m[2] == 0.0 && n.m[5] == 0.0 && n.m[6] == 0.0 && n.m[7] == 0.0;
    if (N_DIM == 2 && planar) lo[2] = hi[2] = 0.0;
    for (size_t i = 0; i < 3; i++) {
        if (!std::isfinite(lo[i]) || !std::isfinite(hi[i])) return EVERYWHERE;
    }
    // x = M^-1 (q - d) over the corners
    const double * m = n.m;
    const double det = m[0] * (m[4] * m[8] - m[5] * m[7]) - m[1] * (m[3] * m[8] - m[5] * m[6]) +
                       m[2] * (m[3] * m[7] - m[4] * m[6]);
    const std::array<double, 9> inv = {(m[4] * m[8] - m[5] * m[7]) / det, (m[2] * m[7] - m[1] * m[8]) / det,
                                       (m[1] * m[5] - m[2] * m[4]) / det, (m[5] * m[6] - m[3] * m[8]) / det,
                                       (m[0] * m[8] - m[2] * m[6]) / det, (m[2] * m[3] - m[0] * m[5]) / det,
                                       (m[3] * m[7] - m[4] * m[6]) / det, (m[1] * m[6] - m[0] * m[7]) / det,
                                       (m[0] * m[4] - m[1] * m[3]) / det};
    Box out = {Vec{INF, INF, INF}, Vec{-INF, -INF, -INF}};
    for (int c = 0; c < 8; c++) {
        const Vec q = {((c & 1) ? hi[0] : lo[0]) - n.d[0], ((c & 2) ? hi[1] : lo[1]) - n.d[1],
                       ((c & 4) ? hi[2] : lo[2]) - n.d[2]};
        for (size_t i = 0; i < 3; i++) {
            const double x = inv[3 * i] * q[0] + inv[3 * i + 1] * q[1] + inv[3 * i + 2] * q[2];
            out[0][i] = std::min(out[0][i], x);
            out[1][i] = std::max(out[1][i], x);
        }
    }
    return out;
}

Box filled_bounds(const Region & region, const uint32_t i) {
    const Filled & f = region.filled()[i];
    auto join = [](const Box & a, const Box & b, const bool both) {
        Box out;
        for (size_t k = 0; k < 3; k++) {
            out[0][k] = both ? std::max(a[0][k], b[0][k]) : std::min(a[0][k], b[0][k]);
            out[1][k] = both ? std::min(a[1][k], b[1][k]) : std::max(a[1][k], b[1][k]);
        }
        return out;
    };
    if (f.kind == Filled::Kind::SHAPES) {
        const RegionView<Kokkos::HostSpace> view = region.host(f);
        const std::vector<double> vertices(view.vertices.data(), view.vertices.data() + view.vertices.extent(0));
        std::vector<Box> stack;
        for (uint32_t k = 0; k < f.count; k++) {
            const Node & n = view.nodes(k);
            if (n.kind == Kind::UNION || n.kind == Kind::INTERSECTION || n.kind == Kind::DIFFERENCE) {
                const Box b = stack.back();
                stack.pop_back();
                if (n.kind != Kind::DIFFERENCE) stack.back() = join(stack.back(), b, n.kind == Kind::INTERSECTION);
            } else {
                stack.push_back(primitive_bounds(n, vertices));
            }
        }
        return stack.back();
    }
    if (f.parts.empty()) return EVERYWHERE;
    Box out = filled_bounds(region, f.parts[0]);
    for (size_t k = 1; k < f.parts.size() && f.kind != Filled::Kind::DIFFERENCE; k++) {
        out = join(out, filled_bounds(region, f.parts[k]), f.kind == Filled::Kind::INTERSECTION);
    }
    return out;
}

} // namespace

std::array<std::array<double, 3>, 2> bounds(const Region & region) {
    if (region.dimension() == N_DIM) return filled_bounds(region, region.root());
    Box out = {Vec{INF, INF, INF}, Vec{-INF, -INF, -INF}};
    for (const Piece & p : region.pieces()) {
        Box b = EVERYWHERE;
        for (const uint32_t i : p.carriers) {
            const Box c = filled_bounds(region, i);
            for (size_t k = 0; k < 3; k++) {
                b[0][k] = std::max(b[0][k], c[0][k]);
                b[1][k] = std::min(b[1][k], c[1][k]);
            }
        }
        for (const uint32_t i : p.inside) {
            const Box c = filled_bounds(region, i);
            for (size_t k = 0; k < 3; k++) {
                b[0][k] = std::max(b[0][k], c[0][k]);
                b[1][k] = std::min(b[1][k], c[1][k]);
            }
        }
        for (size_t k = 0; k < 3; k++) {
            out[0][k] = std::min(out[0][k], b[0][k]);
            out[1][k] = std::max(out[1][k], b[1][k]);
        }
    }
    return out;
}

} // namespace region

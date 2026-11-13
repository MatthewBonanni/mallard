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
#include "region_sampler.h"

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

/** @brief The values of programs of t for move: values[p]. */
struct ProgramValues {
    const double * values;
    KOKKOS_INLINE_FUNCTION double operator()(const uint32_t p) const { return values[p]; }
};

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
    m_steps = std::move(d.steps);
    m_motions = std::move(d.motions);
    m_times = std::move(d.times);
    m_copies = std::move(d.copies);
    m_own = d.dimension == N_DIM ? d.own : 0;
    m_moving = !m_motions.empty() ||
               std::any_of(m_programs.begin(), m_programs.end(), [](const kexpr::Program & p) { return p.uses(3); });
    m_host.edge = d.edge;
    if (analytic()) {
        m_host = host(m_filled[m_root]);
        m_host.edge = d.edge;
    }
}

Region Region::with_root(const uint32_t i, std::vector<Copy> cs) const {
    Definition d;
    d.nodes = m_nodes;
    d.vertices = m_vertices;
    d.programs = m_programs;
    d.filled = m_filled;
    d.pieces = m_pieces;
    d.steps = m_steps;
    d.motions = m_motions;
    d.times = m_times;
    d.copies = std::move(cs);
    d.root = d.own = i;
    d.dimension = m_dimension;
    d.bounded = m_bounded;
    d.closed = m_closed;
    d.resolution = m_resolution;
    d.edge = m_host.edge;
    d.name = m_name;
    d.summary = m_summary;
    return Region(std::move(d));
}

bool Region::rigid() const {
    if (std::any_of(m_programs.begin(), m_programs.end(), [](const kexpr::Program & p) { return p.uses(3); })) return false;
    for (const Motion & m : m_motions) {
        for (const int32_t p : m.program) {
            if (p >= 0) return false;
        }
        for (uint32_t i = 0; i < m.count; i++) {
            if (m_steps[m.first + i].kind == Step::Kind::SCALE && !m_steps[m.first + i].fixed()) return false;
        }
    }
    return true;
}

Region Region::at(const double t) const {
    Region r = with_root(m_root, m_copies);
    r.m_own = m_own;
    std::vector<double> values(m_times.size());
    for (size_t p = 0; p < m_times.size(); p++) values[p] = m_times[p](&t);
    for (const Motion & m : m_motions) move(m, m_steps.data(), ProgramValues{values.data()}, r.m_nodes[m.node]);
    r.m_motions.clear();
    r.m_steps.clear();
    r.m_moving = false;
    r.m_time = t;
    if (r.analytic()) {
        r.m_host = r.host(r.m_filled[r.m_root]);
        r.m_host.edge = m_host.edge;
    }
    return r;
}

Region Region::copy(const uint32_t k) const {
    const Copy & c = m_copies.at(k);
    const std::string summary = m_summary + ", copy " + std::to_string(k);
    if (m_dimension == N_DIM) {
        Region r = with_root(c.filled);
        r.m_own = c.own;
        r.m_summary = summary;
        return r;
    }
    Region r = with_root(0);
    r.m_pieces.assign(m_pieces.begin() + c.first, m_pieces.begin() + c.first + c.count);
    r.m_summary = summary;
    return r;
}

std::vector<Piece> boundary_pieces(const std::vector<Filled> & filled, const uint32_t i, const bool flip) {
    const Filled f = filled[i];
    auto with = [](std::vector<Piece> pieces, const std::vector<uint32_t> & inside,
                   const std::vector<uint32_t> & outside) {
        for (Piece & p : pieces) {
            p.inside.insert(p.inside.end(), inside.begin(), inside.end());
            p.outside.insert(p.outside.end(), outside.begin(), outside.end());
        }
        return pieces;
    };
    Piece piece;
    piece.flip = flip;
    if (f.kind == Filled::Kind::SHAPES) {
        piece.carriers = {i};
        return {piece};
    }
    if (f.kind == Filled::Kind::INSIDE) {
        piece.zones = f.zones;
        return {piece};
    }
    if (f.cells) {
        piece.cells = static_cast<int32_t>(i);
        return {piece};
    }
    // A boolean of shapes and regions of the mesh: each part's boundary where it bounds the whole
    std::vector<Piece> out;
    for (size_t k = 0; k < f.parts.size(); k++) {
        std::vector<uint32_t> others;
        for (size_t j = 0; j < f.parts.size(); j++) {
            if (j != k && !(f.kind == Filled::Kind::DIFFERENCE && j == 0)) others.push_back(f.parts[j]);
        }
        std::vector<Piece> sub;
        if (f.kind == Filled::Kind::UNION) {
            sub = with(boundary_pieces(filled, f.parts[k], flip), {}, others);
        } else if (f.kind == Filled::Kind::INTERSECTION) {
            sub = with(boundary_pieces(filled, f.parts[k], flip), others, {});
        } else if (k == 0) {
            sub = with(boundary_pieces(filled, f.parts[0], flip), {}, others);
        } else {
            sub = with(boundary_pieces(filled, f.parts[k], !flip), {f.parts[0]}, others);
        }
        out.insert(out.end(), sub.begin(), sub.end());
    }
    return out;
}

Region Region::surface_in_domain() const {
    Definition d;
    d.nodes = m_nodes;
    d.vertices = m_vertices;
    d.programs = m_programs;
    d.filled = m_filled;
    Filled domain;
    domain.kind = Filled::Kind::ALL;
    domain.cells = true;
    d.filled.push_back(domain);
    // The boundary of its intersection with the domain: its own in the cells, and the domain's inside it, but
    // outside its parts of whole cells, whose faces on the domain's boundary are already their own
    d.pieces = boundary_pieces(m_filled, m_root, false);
    Piece faces;
    faces.cells = static_cast<int32_t>(d.filled.size() - 1);
    faces.inside = {m_root};
    std::vector<uint32_t> stack = {m_root};
    while (!stack.empty()) {
        const uint32_t i = stack.back();
        stack.pop_back();
        if (m_filled[i].cells) {
            faces.outside.push_back(i);
        } else {
            stack.insert(stack.end(), m_filled[i].parts.begin(), m_filled[i].parts.end());
        }
    }
    d.pieces.push_back(faces);
    d.dimension = N_DIM - 1;
    d.bounded = m_bounded;
    d.closed = true;
    d.resolution = m_resolution;
    d.summary = "boundary(intersection(domain, " + m_summary + "))";
    return Region(std::move(d));
}

RegionView<Kokkos::HostSpace> Region::host(const Filled & shapes) const {
    RegionView<Kokkos::HostSpace> r;
    r.nodes = Kokkos::View<const Node *, Kokkos::HostSpace>(m_nodes.data() + shapes.first, shapes.count);
    r.vertices = Kokkos::View<const double *, Kokkos::HostSpace>(m_vertices.data(), m_vertices.size());
    std::vector<const kexpr::Program *> programs;
    for (const auto & p : m_programs) programs.push_back(&p);
    r.programs = kexpr::ProgramTable<Kokkos::HostSpace>(programs);
    r.time = m_time;
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
    r.time = m_time;
    return r;
}

namespace {

/** @brief Cone k of a curved cell on the host: its point at box coordinates p and the volume per unit p. */
struct HostCone {
    const Mesh & mesh;
    uint32_t c;

    KOKKOS_INLINE_FUNCTION double operator()(const uint32_t k, const double * p, double * x) const {
        KOKKOS_IF_ON_HOST((
            const curved::Geometry & geometry = *mesh.curved_geometry;
            const uint32_t f = mesh.h_face_of_cell(c, k);
            double xi[2] = {0.0, 0.0};
            const double j = geometry.face_coordinates(f, p, xi);
            curved::Vec3 y, dx[3];
            const double dv = geometry.cone_eval(c, k, xi, p[N_DIM - 1], y, dx);
            for (size_t i = 0; i < 3; i++) x[i] = y[i];
            return dv * j;
        ))
        KOKKOS_IF_ON_DEVICE(((void)k; (void)p; (void)x; Kokkos::abort("HostCone is host-only"); return 0.0;))
    }
};

/** @brief The signed distance of a region on the host. */
struct HostDistance {
    const RegionView<Kokkos::HostSpace> & view;

    KOKKOS_INLINE_FUNCTION double operator()(const double * x) const { return view.distance(x); }
};

/** @brief Node k of a straight cell on the host. */
struct HostNodes {
    const Mesh & mesh;
    uint32_t c;

    KOKKOS_INLINE_FUNCTION void operator()(const uint32_t k, double * x) const {
        const uint32_t n = mesh.h_nodes_of_cell(mesh.h_offsets_nodes_of_cell(c) + k);
        for (size_t i = 0; i < 3; i++) x[i] = i < N_DIM ? double(mesh.h_node_coords(n, i)) : 0.0;
    }
};

/** @brief The fractions of a filled region of shapes. */
std::vector<double> shape_fractions(const Mesh & mesh, const Region & region, const RegionView<Kokkos::HostSpace> & view) {
    const sampler::Shape<HostDistance> shape{HostDistance{view}, view.edge, region.bounded()};
    std::vector<double> fractions(mesh.n_cells, 0.0);
    using HostSpace = Kokkos::DefaultHostExecutionSpace;
    Kokkos::parallel_for("region_cell_fractions",
                         Kokkos::RangePolicy<HostSpace, Kokkos::Schedule<Kokkos::Dynamic>>(0, mesh.n_cells),
                         [&](const uint32_t i_cell) {
        double inside = 0.0, total = 0.0;
        if (mesh.curved_geometry && mesh.curved_geometry->cell_is_curved(i_cell)) {
            const HostCone cone{mesh, i_cell};
            for (uint32_t k = 0; k < mesh.h_n_faces_of_cell(i_cell); k++) {
                std::array<double, 2> lo, hi;
                mesh.curved_geometry->box_of(mesh.h_face_of_cell(i_cell, k), lo, hi);
                const double a[3] = {lo[0], N_DIM == 2 ? 0.0 : lo[1], 0.0}, b[3] = {hi[0], N_DIM == 2 ? 1.0 : hi[1], 1.0};
                sampler::cone(shape, mesh.axisymmetric, cone, k, a, b, inside, total);
            }
        } else {
            sampler::straight_cell(shape, mesh.axisymmetric, mesh.h_n_nodes_of_cell(i_cell), HostNodes{mesh, i_cell}, inside,
                                   total);
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
    if (mesh.n_global_cells > 0) elements = comm::allgatherv(elements);
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


namespace {

/** @brief The measures of one primitive in closed form, or false: a similarity transform of a bounded primitive. */
bool primitive_measures(const Node & n, const std::vector<double> & vertices, const bool axisymmetric, Measures & out) {
    const double k = n.scale;
    const double * m = n.m;
    for (size_t i = 0; i < 3; i++) {
        for (size_t j = 0; j < 3; j++) {
            const double g = m[3 * i] * m[3 * j] + m[3 * i + 1] * m[3 * j + 1] + m[3 * i + 2] * m[3 * j + 2];
            if (std::abs(g - (i == j ? 1.0 / (k * k) : 0.0)) > 1e-12 / (k * k)) return false;
        }
    }
    // World points x = k^2 M^T (q - d) of local ones q
    auto world = [&](const Vec & q) {
        Vec x = {0.0, 0.0, 0.0};
        for (size_t i = 0; i < 3; i++) {
            for (size_t j = 0; j < 3; j++) x[i] += k * k * m[3 * j + i] * (q[j] - n.d[j]);
        }
        return x;
    };
    const double * p = n.p;
    double volume = 0.0, surface = 0.0;
    double r_volume = world({0.0, 0.0, 0.0})[1], r_surface = r_volume;  // axisymmetric: the centroids' radii
    if constexpr (N_DIM == 2) {
        const bool planar = m[2] == 0.0 && m[5] == 0.0 && m[6] == 0.0 && m[7] == 0.0;
        if (!planar) return false;
        switch (n.kind) {
            case Kind::SPHERE:
                volume = PI * p[0] * p[0];
                surface = 2.0 * PI * p[0];
                break;
            case Kind::ELLIPSOID:
                if (p[0] != p[1]) return false;
                volume = PI * p[0] * p[0];
                surface = 2.0 * PI * p[0];
                break;
            case Kind::CYLINDER:
                if (!(p[2] > 0.0)) return false;
                volume = PI * (p[0] * p[0] - p[1] * p[1]);
                surface = 2.0 * PI * (p[0] + p[1]);
                break;
            case Kind::BOX:
                if (!std::isfinite(p[0]) || !std::isfinite(p[1]) || !(p[2] > 0.0)) return false;
                volume = 4.0 * p[0] * p[1];
                surface = 4.0 * (p[0] + p[1]);
                break;
            case Kind::POLYGON: {
                const uint32_t c = n.count;
                const double * v = vertices.data() + 2 * n.first;
                auto at = [&](const uint32_t i) { return std::array<double, 2>{v[2 * (i % c)], v[2 * (i % c) + 1]}; };
                // Simple polygons only: the even-odd rule of a self-crossing one is not the shoelace's area
                auto side = [](const std::array<double, 2> & a, const std::array<double, 2> & b,
                               const std::array<double, 2> & q) {
                    return (b[0] - a[0]) * (q[1] - a[1]) - (b[1] - a[1]) * (q[0] - a[0]);
                };
                if (c > 1000) return false;
                for (uint32_t i = 0; i < c; i++) {
                    for (uint32_t j = i + 2; j < c; j++) {
                        if (i == 0 && j == c - 1) continue;
                        const auto a = at(i), b = at(i + 1), e = at(j), f = at(j + 1);
                        const double s1 = side(a, b, e), s2 = side(a, b, f), s3 = side(e, f, a), s4 = side(e, f, b);
                        if (s1 * s2 <= 0.0 && s3 * s4 <= 0.0) return false;
                    }
                }
                double signed_area = 0.0;
                Vec area_centroid = {0.0, 0.0, 0.0}, length_centroid = {0.0, 0.0, 0.0};
                for (uint32_t i = 0; i < c; i++) {
                    const auto a = at(i), b = at(i + 1);
                    const double cr = a[0] * b[1] - b[0] * a[1], length = std::hypot(b[0] - a[0], b[1] - a[1]);
                    signed_area += 0.5 * cr;
                    surface += length;
                    for (size_t d = 0; d < 2; d++) {
                        area_centroid[d] += (a[d] + b[d]) * cr / 6.0;
                        length_centroid[d] += 0.5 * (a[d] + b[d]) * length;
                    }
                }
                volume = std::abs(signed_area);
                for (size_t d = 0; d < 2; d++) {
                    area_centroid[d] /= signed_area;
                    length_centroid[d] /= surface;
                }
                r_volume = world(area_centroid)[1];
                r_surface = world(length_centroid)[1];
                break;
            }
            default: return false;
        }
        volume *= k * k;
        surface *= k;
    } else {
        switch (n.kind) {
            case Kind::SPHERE:
                volume = 4.0 / 3.0 * PI * p[0] * p[0] * p[0];
                surface = 4.0 * PI * p[0] * p[0];
                break;
            case Kind::ELLIPSOID:
                if (p[0] != p[1] || p[1] != p[2]) return false;
                volume = 4.0 / 3.0 * PI * p[0] * p[0] * p[0];
                surface = 4.0 * PI * p[0] * p[0];
                break;
            case Kind::CYLINDER:
                if (!std::isfinite(p[2])) return false;
                volume = 2.0 * p[2] * PI * (p[0] * p[0] - p[1] * p[1]);
                surface = 2.0 * PI * (p[0] * p[0] - p[1] * p[1]) + 4.0 * p[2] * PI * (p[0] + p[1]);
                break;
            case Kind::CONE: {
                const double h = 2.0 * p[0], r1 = p[1], r2 = p[2];
                volume = PI * h / 3.0 * (r1 * r1 + r1 * r2 + r2 * r2);
                surface = PI * (r1 + r2) * std::hypot(r1 - r2, h) + PI * (r1 * r1 + r2 * r2);
                break;
            }
            case Kind::BOX:
                if (!std::isfinite(p[0]) || !std::isfinite(p[1]) || !std::isfinite(p[2])) return false;
                volume = 8.0 * p[0] * p[1] * p[2];
                surface = 8.0 * (p[0] * p[1] + p[1] * p[2] + p[2] * p[0]);
                break;
            case Kind::TORUS:
                if (!(p[0] > p[1])) return false;
                volume = 2.0 * PI * PI * p[0] * p[1] * p[1];
                surface = 4.0 * PI * PI * p[0] * p[1];
                break;
            default: return false;
        }
        volume *= k * k * k;
        surface *= k * k;
    }
    if (axisymmetric) {
        volume *= r_volume;
        surface *= r_surface;
    }
    out = {volume, surface, 0.0};
    return true;
}

/**
 * @brief The measures of a filled region of shapes by adaptive subdivision of
 *        a box around it into cubes (squares in 2D), down to max_level
 *        halvings of the first ones, where the volume (area) of the
 *        linear interpolant of the distance's negative part and the measure
 *        of its zero set are summed on each cube's simplices.
 */
struct Subdivision {
    const RegionView<Kokkos::HostSpace> & region;
    bool bounded;
    bool axisymmetric;
    int max_level;
    double volume = 0.0, surface = 0.0;

    double phi(const Vec & x) const { return region.distance(x.data()); }

    /** @brief The weight of a piece with centroid c: r in axisymmetric runs (the revolved part at r >= 0). */
    double weight(const Vec & c) const { return axisymmetric ? std::max(c[1], 0.0) : 1.0; }

    void cube(const Vec & lo, const Vec & h, const int level) {
        Vec c = lo;
        double radius = 0.0;
        for (size_t i = 0; i < N_DIM; i++) {
            c[i] += 0.5 * h[i];
            radius += 0.25 * h[i] * h[i];
        }
        radius = std::sqrt(radius);
        const double f = phi(c), half_edge = 0.5 * region.edge;
        bool uniform = false;
        if (bounded) {
            uniform = std::abs(f) >= radius + half_edge;
            // Well within the smooth edge and away from the surface: the indicator is smooth there
            if (!uniform && half_edge > 0.0 && radius < 0.125 * region.edge && std::abs(f) >= radius &&
                std::abs(f) < half_edge - radius) {
                gauss(c, h);
                return;
            }
        } else if (level > 1) {
            double slope = 0.0;
            for (int corner = 0; corner < (1 << N_DIM); corner++) {
                Vec x = lo;
                for (size_t i = 0; i < N_DIM; i++) x[i] += (corner >> i & 1) ? h[i] : 0.0;
                slope = std::max(slope, std::abs(phi(x) - f) / radius);
            }
            uniform = std::abs(f) > 2.0 * slope * radius && half_edge == 0.0;
        }
        if (uniform) {
            if (f < 0.0) {
                double measure = 1.0;
                for (size_t i = 0; i < N_DIM; i++) measure *= h[i];
                if (axisymmetric) {
                    // The integral of max(r, 0) across the cube
                    const double a = std::max(lo[1], 0.0), b = std::max(lo[1] + h[1], 0.0);
                    measure *= (b * b - a * a) / (2.0 * h[1]);
                }
                volume += measure;
            }
            return;
        }
        if (level < max_level) {
            Vec half;
            for (size_t i = 0; i < 3; i++) half[i] = 0.5 * h[i];
            for (int child = 0; child < (1 << N_DIM); child++) {
                Vec l = lo;
                for (size_t i = 0; i < N_DIM; i++) l[i] += (child >> i & 1) ? half[i] : 0.0;
                cube(l, half, level + 1);
            }
            return;
        }
        std::array<Vec, 1 << N_DIM> x;
        std::array<double, 1 << N_DIM> v;
        for (size_t corner = 0; corner < (size_t(1) << N_DIM); corner++) {
            x[corner] = lo;
            for (size_t i = 0; i < N_DIM; i++) x[corner][i] += (corner >> i & 1) ? h[i] : 0.0;
            v[corner] = phi(x[corner]);
        }
        const double volume_before = volume;
        if constexpr (N_DIM == 2) {
            triangle({x[0], x[1], x[3]}, {v[0], v[1], v[3]});
            triangle({x[0], x[3], x[2]}, {v[0], v[3], v[2]});
        } else {
            // The six tetrahedra of the cube about its diagonal from corner 0 to corner 7
            constexpr size_t ORDER[6][2] = {{0, 1}, {0, 2}, {1, 0}, {1, 2}, {2, 0}, {2, 1}};
            for (const auto & o : ORDER) {
                const size_t a = size_t(1) << o[0], b = a | size_t(1) << o[1];
                tetrahedron({x[0], x[a], x[b], x[7]}, {v[0], v[a], v[b], v[7]});
            }
        }
        if (half_edge > 0.0) {
            volume = volume_before;
            gauss(c, h);
        }
    }

    /** @brief A smooth edge: the indicator's integral over the cube by 3-point Gauss rules. */
    void gauss(const Vec & c, const Vec & h) {
        constexpr double G[3] = {-0.7745966692414834, 0.0, 0.7745966692414834};
        constexpr double W[3] = {5.0 / 9.0, 8.0 / 9.0, 5.0 / 9.0};
        for (int a = 0; a < 3; a++) {
            for (int b = 0; b < 3; b++) {
                for (int e = 0; e < (N_DIM == 3 ? 3 : 1); e++) {
                    Vec q = c;
                    const int g[3] = {a, b, e};
                    double w = 1.0;
                    for (size_t i = 0; i < N_DIM; i++) {
                        q[i] += 0.5 * h[i] * G[g[i]];
                        w *= 0.5 * h[i] * W[g[i]];
                    }
                    volume += w * weight(q) * indicator(phi(q), region.edge);
                }
            }
        }
    }

    static Vec lerp(const Vec & a, const Vec & b, const double fa, const double fb) {
        const double t = fa / (fa - fb);
        return {a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1]), a[2] + t * (b[2] - a[2])};
    }

    void triangle(const std::array<Vec, 3> & x, const std::array<double, 3> & f) {
        std::vector<Vec> negative, zeros;
        for (size_t i = 0; i < 3; i++) {
            const size_t j = (i + 1) % 3;
            if (f[i] < 0.0) negative.push_back(x[i]);
            if ((f[i] < 0.0) != (f[j] < 0.0)) {
                const Vec z = lerp(x[i], x[j], f[i], f[j]);
                negative.push_back(z);
                zeros.push_back(z);
            }
        }
        double area = 0.0;
        Vec centroid = {0.0, 0.0, 0.0};
        for (size_t i = 0; i < negative.size(); i++) {
            const Vec & a = negative[i];
            const Vec & b = negative[(i + 1) % negative.size()];
            const double cr = a[0] * b[1] - b[0] * a[1];
            area += 0.5 * cr;
            for (size_t d = 0; d < 2; d++) centroid[d] += (a[d] + b[d]) * cr / 6.0;
        }
        if (area != 0.0) {
            for (size_t d = 0; d < 2; d++) centroid[d] /= area;
            volume += std::abs(area) * weight(centroid);
        }
        if (zeros.size() == 2) {
            const Vec mid = {0.5 * (zeros[0][0] + zeros[1][0]), 0.5 * (zeros[0][1] + zeros[1][1]), 0.0};
            surface += std::hypot(zeros[1][0] - zeros[0][0], zeros[1][1] - zeros[0][1]) * weight(mid);
        }
    }

    static double tet_volume(const Vec & a, const Vec & b, const Vec & c, const Vec & d) {
        const Vec u = {b[0] - a[0], b[1] - a[1], b[2] - a[2]}, v = {c[0] - a[0], c[1] - a[1], c[2] - a[2]};
        const Vec w = {d[0] - a[0], d[1] - a[1], d[2] - a[2]};
        return std::abs(dot(u, cross(v, w))) / 6.0;
    }

    static double triangle_area(const Vec & a, const Vec & b, const Vec & c) {
        return 0.5 * norm(cross({b[0] - a[0], b[1] - a[1], b[2] - a[2]}, {c[0] - a[0], c[1] - a[1], c[2] - a[2]}));
    }

    void tetrahedron(const std::array<Vec, 4> & x, const std::array<double, 4> & f) {
        std::vector<size_t> in, out;
        for (size_t i = 0; i < 4; i++) (f[i] < 0.0 ? in : out).push_back(i);
        const double whole = tet_volume(x[0], x[1], x[2], x[3]);
        if (in.empty()) return;
        if (out.empty()) {
            volume += whole;
            return;
        }
        auto z = [&](const size_t i, const size_t j) { return lerp(x[i], x[j], f[i], f[j]); };
        if (in.size() == 1 || in.size() == 3) {
            // A corner cut off: inside it (one vertex in) or outside (three)
            const size_t a = in.size() == 1 ? in[0] : out[0];
            const std::vector<size_t> & rest = in.size() == 1 ? out : in;
            const Vec p = z(a, rest[0]), q = z(a, rest[1]), r = z(a, rest[2]);
            const double corner = tet_volume(x[a], p, q, r);
            volume += in.size() == 1 ? corner : whole - corner;
            surface += triangle_area(p, q, r);
            return;
        }
        // Two in, two out: a prism between the two inside vertices' triangles
        const size_t a = in[0], b = in[1], c = out[0], d = out[1];
        const Vec ac = z(a, c), ad = z(a, d), bc = z(b, c), bd = z(b, d);
        volume += tet_volume(x[a], ac, ad, bd) + tet_volume(x[a], ac, bd, bc) + tet_volume(x[a], x[b], bc, bd);
        surface += triangle_area(ac, ad, bd) + triangle_area(ac, bd, bc);
    }
};

} // namespace

Measures measures(const Region & region, const bool axisymmetric) {
    if (region.dimension() == N_DIM && region.own() != region.root()) {
        // Wrapped across periodic boundaries: the region itself, before its images
        std::vector<Copy> copies = region.copies();
        for (Copy & c : copies) c.filled = c.own;
        return measures(region.with_root(region.own(), copies), axisymmetric);
    }
    const std::vector<Copy> & copies = region.copies();
    if (region.analytic() && copies.size() > 1) {
        // Copies apart from each other: the sum of theirs
        std::vector<Box> boxes;
        for (const Copy & c : copies) boxes.push_back(filled_bounds(region, c.filled));
        bool apart = true;
        for (size_t a = 0; a < boxes.size() && apart; a++) {
            for (size_t b = a + 1; b < boxes.size() && apart; b++) {
                bool overlap = true;
                for (size_t i = 0; i < N_DIM; i++) {
                    overlap = overlap && boxes[a][0][i] < boxes[b][1][i] && boxes[b][0][i] < boxes[a][1][i];
                }
                apart = !overlap;
            }
        }
        if (apart) {
            Measures sum{0.0, 0.0, 0.0};
            for (uint32_t k = 0; k < copies.size(); k++) {
                const Measures m = measures(region.copy(k), axisymmetric);
                sum.volume += m.volume;
                sum.surface += m.surface;
                sum.tolerance = std::max(sum.tolerance, m.tolerance);
            }
            return sum;
        }
    }
    Measures out;
    if (!region.analytic()) return out;
    const Box b = bounds(region);
    for (size_t i = 0; i < N_DIM; i++) {
        if (!std::isfinite(b[0][i]) || !std::isfinite(b[1][i])) return out;
    }
    const RegionView<Kokkos::HostSpace> & view = region.host();
    if (region.edge() == 0.0 && view.nodes.extent(0) == 1) {
        const std::vector<double> vertices(view.vertices.data(), view.vertices.data() + view.vertices.extent(0));
        if (!axisymmetric || b[0][1] >= 0.0) {
            if (primitive_measures(view.nodes(0), vertices, axisymmetric, out)) return out;
        }
    }
    // Cubes of about a quarter of the box's longest side, padded by the smooth edge and a little more
    double longest = 0.0;
    for (size_t i = 0; i < N_DIM; i++) longest = std::max(longest, b[1][i] - b[0][i]);
    const double pad = 0.5 * region.edge() + 1e-6 * longest;
    Vec lo = {0.0, 0.0, 0.0}, h = {0.0, 0.0, 0.0};
    std::array<int, 3> n = {1, 1, 1};
    for (size_t i = 0; i < N_DIM; i++) {
        lo[i] = b[0][i] - pad;
        const double extent = b[1][i] - b[0][i] + 2.0 * pad;
        n[i] = std::max(1, static_cast<int>(std::ceil(4.0 * extent / (longest + 2.0 * pad))));
        h[i] = extent / n[i];
    }
    constexpr int LEVELS = N_DIM == 2 ? 10 : 5;
    std::array<Measures, 3> m;
    for (size_t k = 0; k < 3; k++) {
        Subdivision s{view, region.bounded(), axisymmetric, LEVELS - 2 + static_cast<int>(k)};
        for (int i = 0; i < n[0]; i++) {
            for (int j = 0; j < n[1]; j++) {
                for (int l = 0; l < n[2]; l++) {
                    s.cube({lo[0] + i * h[0], lo[1] + j * h[1], lo[2] + l * h[2]}, h, 0);
                }
            }
        }
        m[k] = {s.volume, s.surface, 0.0};
    }
    // Extrapolated in the cubes' size at the order the three levels show: 2 for smooth surfaces, down to 1 for the
    // surface area of sharp edges, which the interpolant's zero set cuts; the correction is the tolerance
    auto extrapolate = [](const double m0, const double m1, const double m2, double & tolerance) {
        const double d1 = m1 - m0, d2 = m2 - m1;
        const double order = d1 * d2 > 0.0 && std::abs(d2) < std::abs(d1) ? std::clamp(std::log2(d1 / d2), 1.0, 2.0) : 1.0;
        const double correction = d2 / (std::pow(2.0, order) - 1.0), value = m2 + correction;
        if (value != 0.0) tolerance = std::max(tolerance, std::abs(correction / value));
        return value;
    };
    out.volume = extrapolate(m[0].volume, m[1].volume, m[2].volume, out.tolerance);
    out.surface = extrapolate(m[0].surface, m[1].surface, m[2].surface, out.tolerance);
    return out;
}

} // namespace region

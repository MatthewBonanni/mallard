/**
 * @file region_mesh.cpp
 * @author Matthew Bonanni (mbonanni001@gmail.com)
 * @brief Surfaces, curves and points of regions on a mesh.
 * @version 0.1
 * @date 2026-10-30
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 *
 */

#include "region_mesh.h"

#include <algorithm>
#include <cmath>
#include <functional>
#include <limits>
#include <map>
#include <optional>
#include <span>

#include "comm.h"
#include "mesh.h"
#include "statistics.h"

namespace region {

namespace {

using Vec = std::array<double, 3>;
constexpr double PI = 3.14159265358979323846;

Vec add(const Vec & a, const Vec & b) { return {a[0] + b[0], a[1] + b[1], a[2] + b[2]}; }
Vec sub(const Vec & a, const Vec & b) { return {a[0] - b[0], a[1] - b[1], a[2] - b[2]}; }
Vec mul(const double s, const Vec & a) { return {s * a[0], s * a[1], s * a[2]}; }
double dot(const Vec & a, const Vec & b) { return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]; }
Vec cross(const Vec & a, const Vec & b) {
    return {a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]};
}
double norm(const Vec & a) { return std::sqrt(dot(a, a)); }
Vec unit(const Vec & a) {
    const double n = norm(a);
    return n > 0.0 ? mul(1.0 / n, a) : Vec{0.0, 0.0, 0.0};
}
Vec lerp(const Vec & a, const Vec & b, const double t) { return add(a, mul(t, sub(b, a))); }

/** @brief A vertex of a cut: its position and the values of the level functions there. */
struct Vertex {
    Vec x;
    std::vector<double> phi;
};

Vertex lerp(const Vertex & a, const Vertex & b, const double t) {
    Vertex v{lerp(a.x, b.x, t), a.phi};
    for (size_t k = 0; k < v.phi.size(); k++) v.phi[k] = a.phi[k] + t * (b.phi[k] - a.phi[k]);
    return v;
}

/** @brief The part of a polygon (or segment, or point) where the linear function of values s is negative. */
std::vector<Vertex> clip(const std::vector<Vertex> & poly, const std::function<double(const Vertex &)> & s) {
    if (poly.size() <= 2) {
        std::vector<Vertex> out;
        if (poly.size() == 1) {
            if (s(poly[0]) < 0.0) out = poly;
            return out;
        }
        const double a = s(poly[0]), b = s(poly[1]);
        if (a >= 0.0 && b >= 0.0) return out;
        if (a < 0.0 && b < 0.0) return poly;
        const Vertex m = lerp(poly[0], poly[1], a / (a - b));
        return a < 0.0 ? std::vector<Vertex>{poly[0], m} : std::vector<Vertex>{m, poly[1]};
    }
    std::vector<Vertex> out;
    for (size_t i = 0; i < poly.size(); i++) {
        const Vertex & p = poly[i];
        const Vertex & q = poly[(i + 1) % poly.size()];
        const double a = s(p), b = s(q);
        if (a < 0.0) out.push_back(p);
        if ((a < 0.0) != (b < 0.0)) out.push_back(lerp(p, q, a / (a - b)));
    }
    return out;
}

/** @brief The zero set of level k on a simplex (2D triangle, 3D tetrahedron): a segment or a polygon. */
std::vector<Vertex> cut(const std::vector<Vertex> & simplex, const size_t k) {
    std::vector<Vertex> out;
    for (size_t i = 0; i < simplex.size(); i++) {
        for (size_t j = i + 1; j < simplex.size(); j++) {
            const double a = simplex[i].phi[k], b = simplex[j].phi[k];
            if ((a < 0.0) != (b < 0.0)) out.push_back(lerp(simplex[i], simplex[j], a / (a - b)));
        }
    }
    return out;
}

/** @brief The zero set of level k on a polygon (or segment): a segment (or point). */
std::vector<Vertex> cut_polygon(const std::vector<Vertex> & poly, const size_t k) {
    std::vector<Vertex> out;
    const size_t n = poly.size() == 2 ? 1 : poly.size();
    for (size_t i = 0; i < n; i++) {
        const Vertex & p = poly[i];
        const Vertex & q = poly[(i + 1) % poly.size()];
        const double a = p.phi[k], b = q.phi[k];
        if ((a < 0.0) != (b < 0.0)) out.push_back(lerp(p, q, a / (a - b)));
    }
    return out;
}

/** @brief Gradient of the linear function with values phi at the simplex's vertices. */
Vec gradient(const std::vector<Vertex> & s, const size_t k) {
    if constexpr (N_DIM == 2) {
        const Vec e1 = sub(s[1].x, s[0].x), e2 = sub(s[2].x, s[0].x);
        const double det = e1[0] * e2[1] - e1[1] * e2[0];
        const double d1 = s[1].phi[k] - s[0].phi[k], d2 = s[2].phi[k] - s[0].phi[k];
        return {(d1 * e2[1] - d2 * e1[1]) / det, (d2 * e1[0] - d1 * e2[0]) / det, 0.0};
    } else {
        const Vec e1 = sub(s[1].x, s[0].x), e2 = sub(s[2].x, s[0].x), e3 = sub(s[3].x, s[0].x);
        const double det = dot(e1, cross(e2, e3));
        const double d1 = s[1].phi[k] - s[0].phi[k], d2 = s[2].phi[k] - s[0].phi[k], d3 = s[3].phi[k] - s[0].phi[k];
        return mul(1.0 / det, add(add(mul(d1, cross(e2, e3)), mul(d2, cross(e3, e1))), mul(d3, cross(e1, e2))));
    }
}

/** @brief A polygon's vertices ordered counterclockwise about n. */
void order_about(std::vector<Vertex> & poly, const Vec & n) {
    if (poly.size() < 3) return;
    Vec c = {0.0, 0.0, 0.0};
    for (const Vertex & v : poly) c = add(c, mul(1.0 / double(poly.size()), v.x));
    const Vec e1 = unit(sub(poly[0].x, c)), e2 = cross(n, e1);
    std::sort(poly.begin(), poly.end(), [&](const Vertex & a, const Vertex & b) {
        return std::atan2(dot(sub(a.x, c), e2), dot(sub(a.x, c), e1)) <
               std::atan2(dot(sub(b.x, c), e2), dot(sub(b.x, c), e1));
    });
}

/** @brief Area and centroid of a planar polygon, or length and midpoint of a segment; 1 and the point itself. */
std::pair<double, Vec> measure_of(const std::vector<Vec> & v) {
    if (v.size() == 1) return {1.0, v[0]};
    if (v.size() == 2) return {norm(sub(v[1], v[0])), mul(0.5, add(v[0], v[1]))};
    Vec area = {0.0, 0.0, 0.0}, c = {0.0, 0.0, 0.0};
    double total = 0.0;
    for (size_t i = 1; i + 1 < v.size(); i++) {
        const Vec a = cross(sub(v[i], v[0]), sub(v[i + 1], v[0]));
        area = add(area, a);
        const double w = norm(a);
        total += w;
        c = add(c, mul(w / 3.0, add(add(v[0], v[i]), v[i + 1])));
    }
    return {0.5 * norm(area), total > 0.0 ? mul(1.0 / total, c) : v[0]};
}

/** @brief Inverse of a 3 x 3 matrix (rows). */
std::array<double, 9> inverse(const double * m) {
    const double det = m[0] * (m[4] * m[8] - m[5] * m[7]) - m[1] * (m[3] * m[8] - m[5] * m[6]) +
                       m[2] * (m[3] * m[7] - m[4] * m[6]);
    return {(m[4] * m[8] - m[5] * m[7]) / det, (m[2] * m[7] - m[1] * m[8]) / det, (m[1] * m[5] - m[2] * m[4]) / det,
            (m[5] * m[6] - m[3] * m[8]) / det, (m[0] * m[8] - m[2] * m[6]) / det, (m[2] * m[3] - m[0] * m[5]) / det,
            (m[3] * m[7] - m[4] * m[6]) / det, (m[1] * m[6] - m[0] * m[7]) / det, (m[0] * m[4] - m[1] * m[3]) / det};
}

/** @brief A primitive's local coordinates and the world: x = M^-1 (q - d). */
struct Local {
    std::array<double, 9> inv;
    Vec d;
    explicit Local(const Node & n) : inv(inverse(n.m)), d{n.d[0], n.d[1], n.d[2]} {}
    Vec vector(const Vec & q) const {
        return {inv[0] * q[0] + inv[1] * q[1] + inv[2] * q[2], inv[3] * q[0] + inv[4] * q[1] + inv[5] * q[2],
                inv[6] * q[0] + inv[7] * q[1] + inv[8] * q[2]};
    }
    Vec point(const Vec & q) const { return vector(sub(q, d)); }
};

/** @brief Whether a node's frame keeps the z = 0 plane and the z axis (2D runs: its slice is its own z = 0 section). */
bool planar(const Node & n) {
    return n.m[2] == 0.0 && n.m[5] == 0.0 && n.m[6] == 0.0 && n.m[7] == 0.0 && n.m[8] > 0.0 && n.d[2] == 0.0;
}

/**
 * @brief A surface patch (or curve, with v unused) of a primitive in its local
 *        coordinates over [u0, u1] x [v0, v1], split into nu x nv patches:
 *        position and derivatives, oriented so that q_u x q_v (2D curves:
 *        the right of q_u) faces out.
 */
struct Patch {
    std::function<void(double, double, Vec &, Vec &, Vec &)> at;
    double u0, u1, v0 = 0.0, v1 = 0.0;
};

const double GAUSS_X[3] = {-0.7745966692414834, 0.0, 0.7745966692414834};
const double GAUSS_W[3] = {5.0 / 9.0, 8.0 / 9.0, 5.0 / 9.0};

class Builder {
    public:
        Builder(const Mesh & mesh, const Region & region) : m_mesh(mesh), m_region(region) {
            const bool distributed = !mesh.h_global_cell_id.empty();
            double volume = 0.0;
            for (uint32_t c = 0; c < mesh.n_owned(); c++) volume += double(mesh.h_cell_volume(c));
            for (size_t i = 0; i < 3; i++) {
                m_lo[i] = i < N_DIM ? std::numeric_limits<double>::max() : 0.0;
                m_hi[i] = i < N_DIM ? std::numeric_limits<double>::lowest() : 0.0;
            }
            for (uint32_t n = 0; n < mesh.n_nodes; n++) {
                for (size_t i = 0; i < N_DIM; i++) {
                    m_lo[i] = std::min(m_lo[i], double(mesh.h_node_coords(n, i)));
                    m_hi[i] = std::max(m_hi[i], double(mesh.h_node_coords(n, i)));
                }
            }
            double n_cells = double(mesh.n_owned());
            if (distributed) {
                volume = comm::allreduce(volume, comm::Op::SUM);
                n_cells = double(mesh.n_global_cells);
                m_lo = comm::allreduce(m_lo, comm::Op::MIN);
                m_hi = comm::allreduce(m_hi, comm::Op::MAX);
            }
            m_h = std::pow(volume / std::max(n_cells, 1.0), 1.0 / N_DIM);
            for (const auto & f : region.filled()) m_views.push_back(region.host(f));
            // Clipping regions of whole cells: their cells, computed collectively up front
            m_cells.resize(region.filled().size());
            for (const Piece & p : region.pieces()) {
                for (const auto * masks : {&p.inside, &p.outside}) {
                    for (const uint32_t i : *masks) need_cells(i);
                }
                if (p.cells >= 0) need_cells(static_cast<uint32_t>(p.cells));
            }
        }

        Elements build() {
            m_out.dimension = m_region.dimension();
            for (const Piece & p : m_region.pieces()) {
                if (!p.zones.empty() || p.cells >= 0) {
                    faces(p);
                } else if (!exact_piece(p)) {
                    m_out.exact = m_out.exact && affine(p);
                    extract(p);
                }
            }
            return m_out;
        }

    private:
        const Mesh & m_mesh;
        const Region & m_region;
        std::vector<RegionView<Kokkos::HostSpace>> m_views;
        std::vector<std::vector<double>> m_cells;
        Vec m_lo, m_hi;
        double m_h = 1.0;
        Elements m_out;

        void need_cells(const uint32_t i) {
            const Filled & f = m_region.filled()[i];
            if (f.kind == Filled::Kind::SHAPES || !m_cells[i].empty()) return;
            if (f.cells) {
                m_cells[i] = cell_fractions(m_mesh, m_region, i);
            } else {
                for (const uint32_t part : f.parts) need_cells(part);
            }
        }

        /** @brief Whether x (in local cell c, or -1) is inside filled region i. */
        bool inside(const uint32_t i, const Vec & x, const int32_t c) const {
            const Filled & f = m_region.filled()[i];
            if (f.kind == Filled::Kind::SHAPES) return m_views[i].distance(x.data()) < 0.0;
            if (f.cells) return c >= 0 && m_cells[i][static_cast<size_t>(c)] > 0.5;
            bool in = f.kind != Filled::Kind::UNION;
            for (size_t k = 0; k < f.parts.size(); k++) {
                const bool part = inside(f.parts[k], x, c);
                if (f.kind == Filled::Kind::UNION) in = in || part;
                if (f.kind == Filled::Kind::INTERSECTION) in = in && part;
                if (f.kind == Filled::Kind::DIFFERENCE) in = k == 0 ? part : in && !part;
            }
            return in;
        }

        /** @brief A level of filled region i, negative inside: its signed distance, or -1 / 1 at x in cell c. */
        double level(const uint32_t i, const Vec & x, const int32_t c) const {
            if (m_region.filled()[i].kind == Filled::Kind::SHAPES) return m_views[i].distance(x.data());
            return inside(i, x, c) ? -1.0 : 1.0;
        }

        bool kept(const Piece & p, const Vec & x, const int32_t c) const {
            for (const uint32_t i : p.inside) {
                if (!inside(i, x, c)) return false;
            }
            for (const uint32_t i : p.outside) {
                if (inside(i, x, c)) return false;
            }
            return true;
        }

        /** @brief Clip a polygon in cell c by the piece's clipping regions: along their levels, linear between vertices. */
        std::vector<Vertex> clipped(const Piece & p, std::vector<Vertex> poly, const int32_t c) const {
            for (const bool in : {true, false}) {
                for (const uint32_t i : in ? p.inside : p.outside) {
                    if (poly.empty()) return poly;
                    poly = clip(poly, [&](const Vertex & v) { return (in ? 1.0 : -1.0) * level(i, v.x, c); });
                }
            }
            return poly;
        }

        void emit(const uint32_t c, std::vector<Vec> v, Vec direction, const bool flip, double measure = -1.0,
                  const Vec * center = nullptr) {
            const auto [m, centroid] = measure_of(v);
            if (flip) {
                direction = mul(-1.0, direction);
                std::reverse(v.begin(), v.end());
            }
            if (measure < 0.0) measure = m;
            if (!(measure > 0.0)) return;
            m_out.cell.push_back(c);
            m_out.center.push_back(center ? *center : centroid);
            m_out.direction.push_back(direction);
            m_out.measure.push_back(measure);
            m_out.vertices.insert(m_out.vertices.end(), v.begin(), v.end());
            m_out.offsets.push_back(static_cast<uint32_t>(m_out.vertices.size()));
        }

        Vec node(const uint32_t n) const {
            Vec x = {0.0, 0.0, 0.0};
            for (size_t i = 0; i < N_DIM; i++) x[i] = double(m_mesh.h_node_coords(n, i));
            return x;
        }

        // ------------------------------------------------------------ mesh faces

        uint64_t key(const uint32_t n) const {
            return m_mesh.h_global_node_id.empty() ? uint64_t(n) : m_mesh.h_global_node_id[n];
        }

        struct Face {
            uint32_t f, cell;
            bool reversed;  // facing into its cell
        };

        /** @brief Zone faces, the faces around a region of whole cells, or the edges of zone faces. */
        void faces(const Piece & p) {
            std::vector<Face> list;
            if (!p.zones.empty()) {
                for (const FaceZone & z : m_mesh.h_face_zones()) {
                    if (z.get_type() != FaceZoneType::BOUNDARY) continue;
                    if (std::find(p.zones.begin(), p.zones.end(), z.get_name()) == p.zones.end()) continue;
                    for (uint32_t i = 0; i < z.n_faces(); i++) {
                        const uint32_t f = z.h_faces(i);
                        const auto c = static_cast<uint32_t>(m_mesh.h_cells_of_face(f, 0));
                        if (c < m_mesh.n_owned()) list.push_back({f, c, false});
                    }
                }
            } else {
                const auto & in = m_cells[static_cast<size_t>(p.cells)];
                for (uint32_t f = 0; f < m_mesh.n_faces; f++) {
                    const int32_t a = m_mesh.h_cells_of_face(f, 0), b = m_mesh.h_cells_of_face(f, 1);
                    const bool in_a = in[static_cast<size_t>(a)] > 0.5, in_b = b >= 0 && in[static_cast<size_t>(b)] > 0.5;
                    if (in_a == in_b) continue;
                    // Periodic neighbors sit elsewhere: their faces are not this region's surface
                    if (b >= 0 && m_mesh.h_face_shift.extent(0) > f && m_mesh.h_face_shift(f) != 0) continue;
                    const auto c = static_cast<uint32_t>(in_a ? a : b);
                    if (c < m_mesh.n_owned()) list.push_back({f, c, !in_a});
                }
            }
            if (p.edges) {
                edges(p, list);
                return;
            }
            for (const Face & face : list) {
                std::vector<Vertex> poly;
                for (uint32_t k = m_mesh.h_offsets_nodes_of_face(face.f); k < m_mesh.h_offsets_nodes_of_face(face.f + 1); k++) {
                    poly.push_back({node(m_mesh.h_nodes_of_face(k)), {}});
                }
                Vec n = {0.0, 0.0, 0.0};
                for (size_t i = 0; i < N_DIM; i++) n[i] = double(m_mesh.h_face_normals(face.f, i));
                n = unit(n);  // the mesh's normals carry the face area
                if (face.reversed) {
                    n = mul(-1.0, n);
                    std::reverse(poly.begin(), poly.end());
                }
                const bool whole = p.inside.empty() && p.outside.empty();
                poly = clipped(p, poly, static_cast<int32_t>(face.cell));
                if (poly.empty()) continue;
                std::vector<Vec> v;
                for (const Vertex & x : poly) v.push_back(x.x);
                emit(face.cell, v, n, p.flip, whole ? double(m_mesh.h_face_area(face.f)) : -1.0);
            }
        }

        /** @brief The edges (2D: end points) of a set of faces: those of one face only, across every rank. */
        void edges(const Piece & p, const std::vector<Face> & list) {
            std::vector<uint64_t> keys;
            auto edges_of = [&](const Face & face, auto && use) {
                const uint32_t a = m_mesh.h_offsets_nodes_of_face(face.f), b = m_mesh.h_offsets_nodes_of_face(face.f + 1);
                for (uint32_t k = a; k < b; k++) {
                    if (N_DIM == 2) {
                        use(m_mesh.h_nodes_of_face(k), m_mesh.h_nodes_of_face(k));
                    } else {
                        uint32_t u = m_mesh.h_nodes_of_face(k), v = m_mesh.h_nodes_of_face(k + 1 < b ? k + 1 : a);
                        if (face.reversed) std::swap(u, v);
                        use(u, v);
                    }
                }
            };
            for (const auto & face : list) {
                edges_of(face, [&](const uint32_t u, const uint32_t v) {
                    keys.push_back(std::min(key(u), key(v)));
                    keys.push_back(std::max(key(u), key(v)));
                });
            }
            if (!m_mesh.h_global_cell_id.empty()) keys = comm::allgatherv(keys);
            std::map<std::pair<uint64_t, uint64_t>, int> count;
            for (size_t i = 0; i < keys.size(); i += 2) count[{keys[i], keys[i + 1]}]++;
            for (const auto & face : list) {
                edges_of(face, [&](const uint32_t u, const uint32_t v) {
                    if (count[{std::min(key(u), key(v)), std::max(key(u), key(v))}] != 1) return;
                    std::vector<Vertex> seg = {{node(u), {}}};
                    if (N_DIM == 3) seg.push_back({node(v), {}});
                    seg = clipped(p, seg, static_cast<int32_t>(face.cell));
                    if (seg.empty()) return;
                    std::vector<Vec> x;
                    for (const Vertex & s : seg) x.push_back(s.x);
                    emit(face.cell, x, N_DIM == 3 ? unit(sub(node(v), node(u))) : Vec{0.0, 0.0, 0.0}, p.flip);
                });
            }
        }

        // ------------------------------------------------------------ extracted

        bool affine(const Piece & p) const {
            for (const uint32_t i : p.carriers) {
                const Filled & f = m_region.filled()[i];
                if (f.count != 1 || m_region.host(f).nodes(0).kind != Kind::HALF_SPACE) return false;
            }
            return true;
        }

        /** @brief The simplices of an owned cell, as vertex positions. */
        std::vector<std::vector<Vec>> simplices(const uint32_t c) const {
            std::vector<std::vector<Vec>> out;
            if constexpr (N_DIM == 2) {
                const uint32_t n = m_mesh.h_n_nodes_of_cell(c);
                const Vec v0 = node(m_mesh.h_node_of_cell(c, 0));
                for (uint32_t k = 1; k + 1 < n; k++) {
                    out.push_back({v0, node(m_mesh.h_node_of_cell(c, k)), node(m_mesh.h_node_of_cell(c, k + 1))});
                }
            } else {
                std::vector<std::array<std::array<double, 3>, 4>> tets;
                m_mesh.h_cell_tetrahedra(c, tets);
                for (const auto & t : tets) out.push_back({t[0], t[1], t[2], t[3]});
            }
            return out;
        }

        void extract(const Piece & p) {
            const int levels = m_region.resolution();
            for (uint32_t c = 0; c < m_mesh.n_owned(); c++) {
                for (const auto & s : simplices(c)) extract(p, c, s, 0, levels);
            }
        }

        void extract(const Piece & p, const uint32_t c, const std::vector<Vec> & s, const int level, const int levels) {
            std::vector<Vertex> simplex;
            Vec center = {0.0, 0.0, 0.0};
            for (const Vec & x : s) {
                Vertex v{x, {}};
                for (const uint32_t i : p.carriers) v.phi.push_back(m_views[i].distance(x.data()));
                simplex.push_back(v);
                center = add(center, mul(1.0 / double(s.size()), x));
            }
            double radius = 0.0;
            for (const Vec & x : s) radius = std::max(radius, norm(sub(x, center)));
            for (size_t k = 0; k < p.carriers.size(); k++) {
                // No zero set here: bounded distances farther than the simplex's radius, or implicit
                // functions of one sign with a margin of twice their slope
                const double phi = m_views[p.carriers[k]].distance(center.data());
                if (m_region.bounded()) {
                    if (std::abs(phi) >= radius) return;
                    continue;
                }
                bool positive = false, negative = false;
                double slope = 0.0;
                for (const Vertex & v : simplex) {
                    (v.phi[k] < 0.0 ? negative : positive) = true;
                    slope = std::max(slope, std::abs(v.phi[k] - phi) / std::max(norm(sub(v.x, center)), 1e-300));
                }
                if (positive != negative && std::abs(phi) > 2.0 * slope * radius) return;
            }
            if (level < levels) {
                for (const auto & child : subdivide(s)) extract(p, c, child, level + 1, levels);
                return;
            }
            std::vector<Vertex> poly = cut(simplex, 0);
            if (poly.empty()) return;
            Vec direction = unit(gradient(simplex, 0));
            if (p.carriers.size() == 2) {
                poly = cut_polygon(N_DIM == 3 ? ordered(poly, direction) : poly, 1);
                direction = N_DIM == 3 ? unit(cross(direction, gradient(simplex, 1))) : Vec{0.0, 0.0, 0.0};
                if (N_DIM == 3 && poly.size() == 2 && dot(sub(poly[1].x, poly[0].x), direction) < 0.0) {
                    std::swap(poly[0], poly[1]);
                }
            } else if (N_DIM == 3) {
                poly = ordered(poly, direction);
            } else if (poly.size() == 2) {
                // 2D: the normal to the right of travel
                const Vec t = sub(poly[1].x, poly[0].x);
                if (t[1] * direction[0] - t[0] * direction[1] < 0.0) std::swap(poly[0], poly[1]);
            }
            poly = clipped(p, poly, static_cast<int32_t>(c));
            if (poly.empty()) return;
            std::vector<Vec> v;
            for (const Vertex & x : poly) v.push_back(x.x);
            emit(c, v, direction, p.flip);
        }

        static std::vector<Vertex> ordered(std::vector<Vertex> poly, const Vec & n) {
            order_about(poly, n);
            return poly;
        }

        static std::vector<std::vector<Vec>> subdivide(const std::vector<Vec> & v) {
            auto mid = [&](const size_t a, const size_t b) { return mul(0.5, add(v[a], v[b])); };
            if constexpr (N_DIM == 2) {
                const Vec m01 = mid(0, 1), m12 = mid(1, 2), m02 = mid(0, 2);
                return {{v[0], m01, m02}, {m01, v[1], m12}, {m02, m12, v[2]}, {m01, m12, m02}};
            } else {
                const Vec m01 = mid(0, 1), m02 = mid(0, 2), m03 = mid(0, 3), m12 = mid(1, 2), m13 = mid(1, 3),
                          m23 = mid(2, 3);
                return {{v[0], m01, m02, m03}, {m01, v[1], m12, m13}, {m02, m12, v[2], m23}, {m03, m13, m23, v[3]},
                        {m01, m23, m02, m03},  {m01, m23, m03, m13}, {m01, m23, m13, m12}, {m01, m23, m12, m02}};
            }
        }

        // ------------------------------------------------------------ exact

        const Node & only_node(const uint32_t i) const { return m_views[i].nodes(0); }
        bool single(const uint32_t i) const { return m_region.filled()[i].count == 1; }

        /** @brief The range of local coordinate k over the domain's bounding box. */
        std::pair<double, double> local_range(const Node & n, const size_t k) const {
            double lo = std::numeric_limits<double>::max(), hi = std::numeric_limits<double>::lowest();
            for (int corner = 0; corner < 8; corner++) {
                const Vec x = {(corner & 1) ? m_hi[0] : m_lo[0], (corner & 2) ? m_hi[1] : m_lo[1],
                               (corner & 4) ? m_hi[2] : m_lo[2]};
                const double q = n.m[3 * k] * x[0] + n.m[3 * k + 1] * x[1] + n.m[3 * k + 2] * x[2] + n.d[k];
                lo = std::min(lo, q);
                hi = std::max(hi, q);
            }
            return {lo, hi};
        }

        static Patch ring(const double r, const double z, const bool ccw) {
            return {[r, z, ccw](double u, double, Vec & q, Vec & qu, Vec &) {
                        const double a = ccw ? u : -u;
                        q = {r * std::cos(a), r * std::sin(a), z};
                        qu = {-r * std::sin(a) * (ccw ? 1.0 : -1.0), r * std::cos(a) * (ccw ? 1.0 : -1.0), 0.0};
                    },
                    0.0, 2.0 * PI};
        }

        static Patch segment(const Vec & a, const Vec & b) {
            return {[a, b](double u, double, Vec & q, Vec & qu, Vec &) {
                        q = lerp(a, b, u);
                        qu = sub(b, a);
                    },
                    0.0, 1.0};
        }

        /** @brief The flat region z = 0 inside a prism along z (disk, ring, ellipse, rectangle, polygon): surface patches. */
        std::vector<Patch> flat_patches(const Node & m) const {
            std::vector<Patch> out;
            if (m.kind == Kind::CYLINDER || m.kind == Kind::ELLIPSOID) {
                const double a = m.p[0], b = m.kind == Kind::ELLIPSOID ? m.p[1] : m.p[0];
                const double r0 = m.kind == Kind::CYLINDER ? m.p[1] / m.p[0] : 0.0;
                out.push_back({[a, b](double u, double v, Vec & q, Vec & qu, Vec & qv) {
                                   q = {a * u * std::cos(v), b * u * std::sin(v), 0.0};
                                   qu = {a * std::cos(v), b * std::sin(v), 0.0};
                                   qv = {-a * u * std::sin(v), b * u * std::cos(v), 0.0};
                               },
                               r0, 1.0, 0.0, 2.0 * PI});
            } else if (m.kind == Kind::BOX) {
                const double a = m.p[0], b = m.p[1];
                out.push_back({[](double u, double v, Vec & q, Vec & qu, Vec & qv) {
                                   q = {u, v, 0.0};
                                   qu = {1.0, 0.0, 0.0};
                                   qv = {0.0, 1.0, 0.0};
                               },
                               -a, a, -b, b});
            } else {
                for (const auto & t : triangles(m)) {
                    out.push_back({[t](double u, double v, Vec & q, Vec & qu, Vec & qv) {
                                       // The triangle as a quadrilateral collapsed at its third corner
                                       q = add(t[0], add(mul(u, sub(t[1], t[0])), mul(u * v, sub(t[2], t[1]))));
                                       qu = add(sub(t[1], t[0]), mul(v, sub(t[2], t[1])));
                                       qv = mul(u, sub(t[2], t[1]));
                                   },
                                   0.0, 1.0, 0.0, 1.0});
                }
            }
            return out;
        }

        /** @brief The closed curves of the prism's section z = 0, counterclockwise about z (holes clockwise). */
        std::vector<Patch> outline(const Node & m) const {
            std::vector<Patch> out;
            if (m.kind == Kind::CYLINDER) {
                out.push_back(ring(m.p[0], 0.0, true));
                if (m.p[1] > 0.0) out.push_back(ring(m.p[1], 0.0, false));
            } else if (m.kind == Kind::ELLIPSOID) {
                const double a = m.p[0], b = m.p[1];
                out.push_back({[a, b](double u, double, Vec & q, Vec & qu, Vec &) {
                                   q = {a * std::cos(u), b * std::sin(u), 0.0};
                                   qu = {-a * std::sin(u), b * std::cos(u), 0.0};
                               },
                               0.0, 2.0 * PI});
            } else {
                std::vector<Vec> v;
                if (m.kind == Kind::BOX) {
                    v = {{-m.p[0], -m.p[1], 0.0}, {m.p[0], -m.p[1], 0.0}, {m.p[0], m.p[1], 0.0}, {-m.p[0], m.p[1], 0.0}};
                } else {
                    v = polygon(m);
                }
                for (size_t i = 0; i < v.size(); i++) out.push_back(segment(v[i], v[(i + 1) % v.size()]));
            }
            return out;
        }

        /** @brief A polygon node's vertices, counterclockwise. */
        std::vector<Vec> polygon(const Node & m) const {
            std::vector<Vec> v;
            const auto & all = m_views[0].vertices;
            for (uint32_t k = 0; k < m.count; k++) v.push_back({all(2 * (m.first + k)), all(2 * (m.first + k) + 1), 0.0});
            double area = 0.0;
            for (size_t i = 0; i < v.size(); i++) area += cross(v[i], v[(i + 1) % v.size()])[2];
            if (area < 0.0) std::reverse(v.begin(), v.end());
            return v;
        }

        /** @brief A simple polygon cut into triangles (ear clipping), counterclockwise. */
        std::vector<std::array<Vec, 3>> triangles(const Node & m) const {
            std::vector<Vec> v = polygon(m);
            std::vector<std::array<Vec, 3>> out;
            while (v.size() > 3) {
                bool cut = false;
                for (size_t i = 0; i < v.size() && !cut; i++) {
                    const Vec & a = v[(i + v.size() - 1) % v.size()];
                    const Vec & b = v[i];
                    const Vec & c = v[(i + 1) % v.size()];
                    if (cross(sub(b, a), sub(c, b))[2] <= 0.0) continue;
                    bool empty = true;
                    for (const Vec & x : v) {
                        if (&x == &a || &x == &b || &x == &c) continue;
                        if (cross(sub(b, a), sub(x, a))[2] > 0.0 && cross(sub(c, b), sub(x, b))[2] > 0.0 &&
                            cross(sub(a, c), sub(x, c))[2] > 0.0) {
                            empty = false;
                        }
                    }
                    if (!empty) continue;
                    out.push_back({a, b, c});
                    v.erase(v.begin() + static_cast<std::ptrdiff_t>(i));
                    cut = true;
                }
                if (!cut) break;
            }
            if (v.size() == 3) out.push_back({v[0], v[1], v[2]});
            return out;
        }

        /** @brief The surface (2D: curve) of a primitive, in patches, or none if it has no exact form here. */
        std::optional<std::vector<Patch>> surface(const Node & n) const {
            std::vector<Patch> out;
            if (N_DIM == 2) {
                if (!planar(n)) return std::nullopt;
                if (n.kind == Kind::SPHERE) {
                    out.push_back(ring(n.p[0], 0.0, true));
                } else if (n.kind == Kind::ELLIPSOID || n.kind == Kind::BOX || n.kind == Kind::POLYGON ||
                           n.kind == Kind::CYLINDER) {
                    return outline(n);
                } else {
                    return std::nullopt;
                }
                return out;
            }
            const double inf = std::numeric_limits<double>::infinity();
            if (n.kind == Kind::SPHERE || n.kind == Kind::ELLIPSOID) {
                const double a = n.p[0], b = n.kind == Kind::SPHERE ? n.p[0] : n.p[1],
                             c = n.kind == Kind::SPHERE ? n.p[0] : n.p[2];
                out.push_back({[a, b, c](double u, double v, Vec & q, Vec & qu, Vec & qv) {
                                   q = {a * std::sin(u) * std::cos(v), b * std::sin(u) * std::sin(v), c * std::cos(u)};
                                   qu = {a * std::cos(u) * std::cos(v), b * std::cos(u) * std::sin(v), -c * std::sin(u)};
                                   qv = {-a * std::sin(u) * std::sin(v), b * std::sin(u) * std::cos(v), 0.0};
                               },
                               0.0, PI, 0.0, 2.0 * PI});
            } else if (n.kind == Kind::CYLINDER || n.kind == Kind::CONE) {
                // Sides along z with radius r(z), outer and inner, and the caps
                double h, r_lo, r_hi, inner;
                if (n.kind == Kind::CYLINDER) {
                    h = n.p[2];
                    r_lo = r_hi = n.p[0];
                    inner = n.p[1];
                } else {
                    h = n.p[0];
                    r_lo = n.p[1];
                    r_hi = n.p[2];
                    inner = 0.0;
                }
                auto [z0, z1] = h == inf ? local_range(n, 2) : std::pair<double, double>{-h, h};
                for (const double sign : {1.0, -1.0}) {
                    if (sign < 0.0 && !(inner > 0.0)) continue;
                    const double a = sign > 0.0 ? r_lo : inner, b = sign > 0.0 ? r_hi : inner;
                    const double H = h;
                    out.push_back({[a, b, H, sign](double u, double v, Vec & q, Vec & qu, Vec & qv) {
                                       // u around, v along z (inner side: v around, u along)
                                       const double t = sign > 0.0 ? u : v, z = sign > 0.0 ? v : u;
                                       const double s = H == std::numeric_limits<double>::infinity() ? 0.0 : (z + H) / (2.0 * H);
                                       const double r = a + (b - a) * s;
                                       const double dr = H == std::numeric_limits<double>::infinity() ? 0.0 : (b - a) / (2.0 * H);
                                       q = {r * std::cos(t), r * std::sin(t), z};
                                       const Vec dt = {-r * std::sin(t), r * std::cos(t), 0.0}, dz = {dr * std::cos(t), dr * std::sin(t), 1.0};
                                       qu = sign > 0.0 ? dt : dz;
                                       qv = sign > 0.0 ? dz : dt;
                                   },
                                   sign > 0.0 ? 0.0 : z0, sign > 0.0 ? 2.0 * PI : z1, sign > 0.0 ? z0 : 0.0,
                                   sign > 0.0 ? z1 : 2.0 * PI});
                }
                if (h != inf) {
                    for (const double z : {-h, h}) {
                        const double r = z > 0.0 ? r_hi : r_lo;
                        if (!(r > 0.0)) continue;
                        const double r0 = inner / r;
                        out.push_back({[r, z](double u, double v, Vec & q, Vec & qu, Vec & qv) {
                                           // Top: (radius, angle) faces +z; bottom: (angle, radius) faces -z
                                           const double rho = z > 0.0 ? u : v, t = z > 0.0 ? v : u;
                                           q = {r * rho * std::cos(t), r * rho * std::sin(t), z};
                                           const Vec dr = {r * std::cos(t), r * std::sin(t), 0.0},
                                                     dt = {-r * rho * std::sin(t), r * rho * std::cos(t), 0.0};
                                           qu = z > 0.0 ? dr : dt;
                                           qv = z > 0.0 ? dt : dr;
                                       },
                                       z > 0.0 ? r0 : 0.0, z > 0.0 ? 1.0 : 2.0 * PI, z > 0.0 ? 0.0 : r0,
                                       z > 0.0 ? 2.0 * PI : 1.0});
                    }
                }
            } else if (n.kind == Kind::BOX) {
                const Vec hs = {n.p[0], n.p[1], n.p[2]};
                for (size_t k = 0; k < 3; k++) {
                    for (const double sign : {1.0, -1.0}) {
                        // Face of axis k at sign * h_k, spanned by the next two axes (in this order it faces out)
                        const size_t i = (k + 1) % 3, j = (k + 2) % 3;
                        const size_t a = sign > 0.0 ? i : j, b = sign > 0.0 ? j : i;
                        out.push_back({[k, a, b, sign, hs](double u, double v, Vec & q, Vec & qu, Vec & qv) {
                                           q = {0.0, 0.0, 0.0};
                                           q[k] = sign * hs[k];
                                           q[a] = u;
                                           q[b] = v;
                                           qu = {0.0, 0.0, 0.0};
                                           qv = {0.0, 0.0, 0.0};
                                           qu[a] = 1.0;
                                           qv[b] = 1.0;
                                       },
                                       -hs[a], hs[a], -hs[b], hs[b]});
                    }
                }
            } else if (n.kind == Kind::TORUS) {
                const double R = n.p[0], r = n.p[1];
                out.push_back({[R, r](double u, double v, Vec & q, Vec & qu, Vec & qv) {
                                   const double A = R + r * std::cos(v);
                                   q = {A * std::cos(u), A * std::sin(u), r * std::sin(v)};
                                   qu = {-A * std::sin(u), A * std::cos(u), 0.0};
                                   qv = {-r * std::sin(v) * std::cos(u), -r * std::sin(v) * std::sin(u), r * std::cos(v)};
                               },
                               0.0, 2.0 * PI, 0.0, 2.0 * PI});
            } else {
                return std::nullopt;
            }
            return out;
        }

        bool same_frame(const Node & a, const Node & b) const {
            return std::equal(a.m, a.m + 9, b.m) && std::equal(a.d, a.d + 3, b.d);
        }

        bool prism(const Node & m) const {
            const double inf = std::numeric_limits<double>::infinity();
            return m.kind == Kind::POLYGON || ((m.kind == Kind::CYLINDER || m.kind == Kind::ELLIPSOID || m.kind == Kind::BOX) &&
                                               m.p[2] == inf);
        }

        /** @brief Sample a piece of exact form on its patches, if it has one. */
        bool exact_piece(const Piece & p) {
            std::vector<Patch> patches;
            const Node * frame = nullptr;
            Piece rest = p;
            if (p.carriers.size() == 1 && single(p.carriers[0])) {
                const Node & n = only_node(p.carriers[0]);
                if (N_DIM == 3 && n.kind == Kind::HALF_SPACE && !p.inside.empty() && single(p.inside[0]) &&
                    prism(only_node(p.inside[0])) && same_frame(n, only_node(p.inside[0]))) {
                    // A flat surface
                    patches = flat_patches(only_node(p.inside[0]));
                    rest.inside.erase(rest.inside.begin());
                } else if (const auto s = surface(n)) {
                    patches = *s;
                }
                frame = &n;
            } else if (N_DIM == 3 && p.carriers.size() == 2 && single(p.carriers[0]) && single(p.carriers[1])) {
                const Node & plane = only_node(p.carriers[0]);
                const Node & m = only_node(p.carriers[1]);
                if (plane.kind == Kind::HALF_SPACE && prism(m) && same_frame(plane, m)) {
                    patches = outline(m);
                    frame = &plane;
                }
            }
            if (patches.empty()) return false;
            sample(rest, Local(*frame), patches);
            return true;
        }

        /** @brief Patches of a curve (one parameter) or surface (two), split near the cell size, sampled at Gauss points. */
        void sample(const Piece & p, const Local & map, const std::vector<Patch> & patches) {
            const int dim = m_region.dimension();
            const bool curve = dim == 1;
            struct Cell {
                double u0, u1, v0, v1;
                const Patch * patch;
            };
            std::vector<Cell> cells;
            for (const Patch & patch : patches) {
                // Splits from the patch's extent along each parameter
                auto extent = [&](const bool along_u) {
                    double longest = 0.0;
                    for (int k = 0; k <= 4; k++) {
                        double length = 0.0;
                        for (int i = 0; i < 64; i++) {
                            const double t = (i + 0.5) / 64.0, w = k / 4.0;
                            const double u = along_u ? patch.u0 + t * (patch.u1 - patch.u0) : patch.u0 + w * (patch.u1 - patch.u0);
                            const double v = along_u ? patch.v0 + w * (patch.v1 - patch.v0) : patch.v0 + t * (patch.v1 - patch.v0);
                            Vec q, qu, qv;
                            patch.at(u, v, q, qu, qv);
                            length += norm(map.vector(along_u ? qu : qv)) * (along_u ? patch.u1 - patch.u0 : patch.v1 - patch.v0) / 64.0;
                        }
                        longest = std::max(longest, length);
                    }
                    return static_cast<int>(std::clamp(std::ceil(2.0 * longest / m_h), 2.0, 4096.0));
                };
                const int nu = extent(true), nv = curve ? 1 : extent(false);
                if (double(cells.size()) + double(nu) * nv > 1e7) {
                    throw InputError("region " + m_region.summary() + ": its surface is too large for the cells to "
                                     "sample (more than 10^7 pieces).");
                }
                for (int i = 0; i < nu; i++) {
                    for (int j = 0; j < nv; j++) {
                        cells.push_back({patch.u0 + (patch.u1 - patch.u0) * i / nu, patch.u0 + (patch.u1 - patch.u0) * (i + 1) / nu,
                                         patch.v0 + (patch.v1 - patch.v0) * j / nv, patch.v0 + (patch.v1 - patch.v0) * (j + 1) / nv,
                                         &patch});
                    }
                }
            }
            // Gauss points of every patch (then its center), located on the mesh
            const int n_gauss = curve ? 3 : 9;
            std::vector<std::array<double, N_DIM>> points;
            std::vector<double> weights;
            std::vector<Vec> world, normals;
            for (const Cell & c : cells) {
                for (int g = 0; g <= n_gauss; g++) {
                    const bool center = g == n_gauss;
                    const double xu = center ? 0.0 : GAUSS_X[curve ? g : g / 3], xv = center || curve ? 0.0 : GAUSS_X[g % 3];
                    const double u = c.u0 + 0.5 * (1.0 + xu) * (c.u1 - c.u0), v = c.v0 + 0.5 * (1.0 + xv) * (c.v1 - c.v0);
                    Vec q, qu, qv;
                    c.patch->at(u, v, q, qu, qv);
                    const Vec x = map.point(q), xu_w = map.vector(qu), xv_w = map.vector(qv);
                    Vec n = curve ? unit(xu_w) : cross(xu_w, xv_w);
                    double w = center ? 0.0 : (curve ? GAUSS_W[g] * 0.5 * (c.u1 - c.u0) * norm(xu_w)
                                                     : GAUSS_W[g / 3] * GAUSS_W[g % 3] * 0.25 * (c.u1 - c.u0) * (c.v1 - c.v0) * norm(n));
                    if (!curve) n = unit(n);
                    if (curve && N_DIM == 2) n = {n[1], -n[0], 0.0};
                    std::array<double, N_DIM> point;
                    for (size_t i = 0; i < N_DIM; i++) point[i] = x[i];
                    points.push_back(point);
                    weights.push_back(w);
                    world.push_back(x);
                    normals.push_back(n);
                }
            }
            std::vector<uint8_t> contained;
            const std::vector<int32_t> located = locate_points(m_mesh, points, &contained);
            std::vector<uint32_t> keep(points.size(), 0);
            for (size_t k = 0; k < points.size(); k++) {
                keep[k] = located[k] >= 0 && contained[k] && kept(p, world[k], located[k]) ? 1 : 0;
            }
            if (!m_mesh.h_global_cell_id.empty()) comm::allreduce(std::span<uint32_t>(keep), comm::Op::MAX);
            const size_t stride = static_cast<size_t>(n_gauss) + 1;
            for (size_t c = 0; c < cells.size(); c++) {
                const size_t first = c * stride, center = first + static_cast<size_t>(n_gauss);
                double measure = 0.0;
                size_t owner = keep[center] ? center : first + stride;
                for (size_t k = first; k < center; k++) {
                    if (!keep[k]) continue;
                    measure += weights[k];
                    if (owner == first + stride) owner = k;
                }
                if (!(measure > 0.0) || located[owner] < 0) continue;  // another rank's
                // Vertices: the patch's corners in order (a segment's ends)
                std::vector<Vec> corners;
                const auto & cell = cells[c];
                const std::vector<std::pair<double, double>> uv =
                    curve ? std::vector<std::pair<double, double>>{{cell.u0, 0.0}, {cell.u1, 0.0}}
                          : std::vector<std::pair<double, double>>{{cell.u0, cell.v0}, {cell.u1, cell.v0}, {cell.u1, cell.v1}, {cell.u0, cell.v1}};
                for (const auto & [u, v] : uv) {
                    Vec q, qu, qv;
                    cell.patch->at(u, v, q, qu, qv);
                    const Vec x = map.point(q);
                    if (corners.empty() || norm(sub(x, corners.back())) > 1e-14 * m_h) corners.push_back(x);
                }
                if (corners.size() > 2 && norm(sub(corners.front(), corners.back())) <= 1e-14 * m_h) corners.pop_back();
                Vec n = normals[center];
                if (norm(n) == 0.0) {
                    for (size_t k = first; k < center; k++) n = add(n, mul(weights[k], normals[k]));
                    n = unit(n);
                }
                emit(static_cast<uint32_t>(located[owner]), corners, n, p.flip, measure, &world[owner]);
            }
        }
};

} // namespace

Elements elements(const Mesh & mesh, const Region & region) {
    if (region.dimension() == N_DIM) throw std::logic_error("region::elements: the region is filled.");
    return Builder(mesh, region).build();
}

} // namespace region

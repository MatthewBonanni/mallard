/**
 * @file curved.cpp
 * @author Matthew Bonanni (mbonanni001@gmail.com)
 * @brief Curved (high-order) boundary geometry.
 * @version 0.1
 * @date 2026-10-09
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 *
 */

#include "curved.h"

#include <algorithm>
#include <cmath>
#include <map>
#include <stdexcept>

#include "comm.h"
#include "input.h"
#include "mesh.h"
#include "mesh_block.h"
#include "zone.h"

namespace curved {

namespace {

Vec3 sub(const Vec3 & a, const Vec3 & b) { return {a[0] - b[0], a[1] - b[1], a[2] - b[2]}; }
Vec3 add(const Vec3 & a, const Vec3 & b) { return {a[0] + b[0], a[1] + b[1], a[2] + b[2]}; }
Vec3 scale(double s, const Vec3 & a) { return {s * a[0], s * a[1], s * a[2]}; }
double dot(const Vec3 & a, const Vec3 & b) { return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]; }
Vec3 cross(const Vec3 & a, const Vec3 & b) {
    return {a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]};
}
// 2D area vector of a tangent t: (t_y, -t_x), pointing out of the cell the tangent runs counterclockwise around
Vec3 rot(const Vec3 & t) { return {t[1], -t[0], 0.0}; }
Vec3 matvec(const double * J, const Vec3 & a) {
    return {J[0] * a[0] + J[1] * a[1] + J[2] * a[2], J[3] * a[0] + J[4] * a[1] + J[5] * a[2],
            J[6] * a[0] + J[7] * a[1] + J[8] * a[2]};
}

void gauss_legendre(size_t n, std::vector<double> & x, std::vector<double> & w) {
    x.resize(n);
    w.resize(n);
    const double nd = double(n);
    for (size_t i = 0; i < n; i++) {
        double z = std::cos(M_PI * (double(i) + 0.75) / (nd + 0.5));
        double dp = 0.0;
        for (int it = 0; it < 100; it++) {
            double p0 = 1.0, p1 = z;
            for (size_t k = 2; k <= n; k++) {
                const double kd = double(k);
                const double p2 = ((2.0 * kd - 1.0) * z * p1 - (kd - 1.0) * p0) / kd;
                p0 = p1;
                p1 = p2;
            }
            if (n == 1) p0 = 1.0;
            dp = nd * (z * p1 - p0) / (z * z - 1.0);
            const double dz = p1 / dp;
            z -= dz;
            if (std::abs(dz) < 1e-16) break;
        }
        x[n - 1 - i] = z;
        w[n - 1 - i] = 2.0 / ((1.0 - z * z) * dp * dp);
    }
}

/** @brief Gauss-Legendre rule on [0, 1]. */
void gauss_unit(size_t n, std::vector<double> & x, std::vector<double> & w) {
    gauss_legendre(n, x, w);
    for (size_t i = 0; i < n; i++) {
        x[i] = 0.5 * (x[i] + 1.0);
        w[i] *= 0.5;
    }
}

/** @brief Reference rule with n points per direction: 2D [-1, 1]; triangles (collapsed); quadrilaterals [-1, 1]^2. */
void reference_rule(uint32_t n_nodes, size_t n, std::vector<std::array<double, 2>> & xi, std::vector<double> & w) {
    std::vector<double> g, gw;
    xi.clear();
    w.clear();
    if (N_DIM == 2) {
        gauss_legendre(n, g, gw);
        for (size_t a = 0; a < n; a++) {
            xi.push_back({g[a], 0.0});
            w.push_back(gw[a]);
        }
    } else if (n_nodes == 3) {
        gauss_unit(n, g, gw);
        for (size_t a = 0; a < n; a++) {
            for (size_t b = 0; b < n; b++) {
                xi.push_back({g[a], g[b] * (1.0 - g[a])});
                w.push_back(gw[a] * gw[b] * (1.0 - g[a]));
            }
        }
    } else {
        gauss_legendre(n, g, gw);
        for (size_t a = 0; a < n; a++) {
            for (size_t b = 0; b < n; b++) {
                xi.push_back({g[a], g[b]});
                w.push_back(gw[a] * gw[b]);
            }
        }
    }
}

/** @brief Lagrange basis on g + 1 equispaced nodes of [0, 1], and its derivative. */
void lagrange(size_t g, double t, double * l, double * dl) {
    const double gd = double(g);
    for (size_t i = 0; i <= g; i++) {
        double num = 1.0, den = 1.0, d = 0.0;
        const double ti = double(i) / gd;
        for (size_t j = 0; j <= g; j++) {
            if (j == i) continue;
            const double tj = double(j) / gd;
            den *= ti - tj;
            // d/dt prod_j (t - tj) by the product rule
            double prod = 1.0;
            for (size_t k = 0; k <= g; k++) {
                if (k == i || k == j) continue;
                prod *= t - double(k) / gd;
            }
            d += prod;
            num *= t - tj;
        }
        l[i] = num / den;
        dl[i] = d / den;
    }
}

/**
 * @brief Indices into a Gmsh element's nodes of edge k (corner k to corner
 *        k + 1), corners included, for an element of order g with n corners.
 */
std::vector<uint32_t> element_edge(uint32_t n_corners, uint32_t g, uint32_t k) {
    std::vector<uint32_t> idx = {k};
    if (n_corners == 2) {
        // line3: [a, b, m]; line4: [a, b, 1/3, 2/3]
        for (uint32_t i = 0; i + 1 < g; i++) idx.push_back(2 + i);
        idx.push_back(1);
        return idx;
    }
    for (uint32_t i = 0; i + 1 < g; i++) idx.push_back(n_corners + k * (g - 1) + i);
    idx.push_back((k + 1) % n_corners);
    return idx;
}

using Key = std::vector<uint64_t>;

Key sorted_key(const std::vector<uint64_t> & corners) {
    Key k = corners;
    std::sort(k.begin(), k.end());
    return k;
}

/** @brief Faces sorted by key, the first of equal keys kept. */
void sort_unique(std::vector<SurfaceFace> & faces) {
    std::stable_sort(faces.begin(), faces.end(), [](const SurfaceFace & a, const SurfaceFace & b) {
        return sorted_key(a.corners) < sorted_key(b.corners);
    });
    std::vector<SurfaceFace> out;
    for (auto & f : faces) {
        if (!out.empty() && sorted_key(out.back().corners) == sorted_key(f.corners)) continue;
        out.push_back(std::move(f));
    }
    faces = std::move(out);
}

/** @brief Throws for high-order faces of unsupported element types. */
void check_high_order(const std::vector<SurfaceFace> & faces) {
    for (const auto & f : faces) {
        if (f.shape < 0 && element_order(uint32_t(f.corners.size()), uint32_t(f.nodes.size())) < 2) {
            throw std::runtime_error("curved boundaries: unsupported high-order face with " +
                                     std::to_string(f.corners.size()) + " corners and " +
                                     std::to_string(f.nodes.size()) + " nodes.");
        }
    }
}

int32_t shape_of_zone(const std::vector<Shape> & shapes, const std::string & zone) {
    for (size_t s = 0; s < shapes.size(); s++) {
        if (shapes[s].zone == zone) return int32_t(s);
    }
    return -1;
}

}  // namespace

void Shape::project(const Vec3 & x, Vec3 & p, double * J) const {
    Vec3 d = sub(x, center);
    Vec3 along = {0.0, 0.0, 0.0};
    if (kind == Kind::CIRCLE) {
        d[2] = 0.0;
    } else if (kind == Kind::CYLINDER) {
        along = scale(dot(d, axis), axis);
        d = sub(d, along);
    }
    const double r = std::sqrt(dot(d, d));
    if (!(r > 0.0)) throw std::runtime_error("curved boundaries: a point lies on the center of its shape.");
    const Vec3 u = scale(1.0 / r, d);
    p = add(add(center, along), scale(radius, u));
    if (kind == Kind::CIRCLE) p[2] = x[2];
    for (size_t a = 0; a < 3; a++) {
        for (size_t b = 0; b < 3; b++) {
            const double I = (a == b) ? 1.0 : 0.0;
            if (kind == Kind::CYLINDER) {
                J[3 * a + b] = axis[a] * axis[b] + radius / r * (I - axis[a] * axis[b] - u[a] * u[b]);
            } else {
                J[3 * a + b] = radius / r * (I - u[a] * u[b]);
            }
        }
    }
    if (kind == Kind::CIRCLE) {
        for (size_t a = 0; a < 3; a++) J[3 * a + 2] = J[3 * 2 + a] = 0.0;
        J[8] = 1.0;
    }
}

uint32_t element_order(uint32_t n_corners, uint32_t n_nodes) {
    switch (n_corners) {
        case 2:
            return (n_nodes >= 2 && n_nodes <= 4) ? n_nodes - 1 : 0;
        case 3:
            return n_nodes == 3 ? 1 : n_nodes == 6 ? 2 : (n_nodes == 9 || n_nodes == 10) ? 3 : 0;
        case 4:
            return n_nodes == 4 ? 1 : (n_nodes == 8 || n_nodes == 9) ? 2 : n_nodes == 16 ? 3 : 0;
        default:
            return 0;
    }
}

bool enabled(const toml::value & input) {
    if (!input.contains("mesh")) return true;
    return toml::find_or<bool>(input, "mesh", "curved_geometry", true);
}

std::vector<Shape> read_shapes(const toml::value & input) {
    std::vector<Shape> shapes;
    if (!enabled(input) || !input.contains("mesh") || !input.at("mesh").contains("curved")) return shapes;
    for (const auto & entry : toml::find<std::vector<toml::value>>(input, "mesh", "curved")) {
        Shape s;
        s.zone = toml::find<std::string>(entry, "zone");
        const std::string kind = toml::find<std::string>(entry, "shape");
        if (kind == "circle" && N_DIM == 2) {
            s.kind = Shape::Kind::CIRCLE;
        } else if (kind == "sphere" && N_DIM == 3) {
            s.kind = Shape::Kind::SPHERE;
        } else if (kind == "cylinder" && N_DIM == 3) {
            s.kind = Shape::Kind::CYLINDER;
        } else {
            throw InputError("mesh.curved: shape = \"" + kind + "\" is not one of: " +
                             (N_DIM == 2 ? "circle" : "sphere, cylinder") + ".");
        }
        const auto center = toml::find<std::vector<toml::value>>(entry, "center");
        if (center.size() != N_DIM) throw InputError("mesh.curved: center needs " + std::to_string(N_DIM) + " components.");
        for (size_t i = 0; i < N_DIM; i++) s.center[i] = double(as_real(center[i], "mesh.curved.center"));
        s.radius = double(as_real(toml::find(entry, "radius"), "mesh.curved.radius"));
        if (!(s.radius > 0.0)) throw InputError("mesh.curved: radius must be positive.");
        if (s.kind == Shape::Kind::CYLINDER) {
            const auto axis = toml::find<std::vector<toml::value>>(entry, "axis");
            if (axis.size() != 3) throw InputError("mesh.curved: axis needs 3 components.");
            for (size_t i = 0; i < 3; i++) s.axis[i] = double(as_real(axis[i], "mesh.curved.axis"));
            const double norm = std::sqrt(dot(s.axis, s.axis));
            if (!(norm > 0.0)) throw InputError("mesh.curved: axis must be nonzero.");
            s.axis = scale(1.0 / norm, s.axis);
        }
        if (shape_of_zone(shapes, s.zone) >= 0) throw InputError("mesh.curved: zone " + s.zone + " is listed twice.");
        shapes.push_back(s);
    }
    return shapes;
}

Surfaces surfaces_of_mesh(const Mesh & mesh, const toml::value & input) {
    Surfaces out;
    if (!enabled(input)) return out;
    out.shapes = read_shapes(input);
    for (size_t s = 0; s < out.shapes.size(); s++) {
        const FaceZone * zone = nullptr;
        for (const FaceZone & z : mesh.h_face_zones()) {
            if (z.get_name() == out.shapes[s].zone && z.get_type() == FaceZoneType::BOUNDARY) zone = &z;
        }
        if (zone == nullptr) throw InputError("mesh.curved: no boundary zone named " + out.shapes[s].zone + ".");
        for (uint32_t k = 0; k < zone->h_faces.extent(0); k++) {
            const uint32_t f = zone->h_faces(k);
            SurfaceFace face;
            face.shape = int32_t(s);
            for (uint32_t j = 0; j < mesh.h_n_nodes_of_face(f); j++) {
                const uint32_t node = mesh.h_node_of_face(f, j);
                face.corners.push_back(mesh.h_global_node_id.empty() ? node : mesh.h_global_node_id[node]);
            }
            out.faces.push_back(std::move(face));
        }
    }
    check_high_order(mesh.high_order_faces);
    out.faces.insert(out.faces.end(), mesh.high_order_faces.begin(), mesh.high_order_faces.end());
    sort_unique(out.faces);
    return out;
}

Surfaces surfaces_of_block(const MeshBlock & block, const toml::value & input) {
    Surfaces out;
    if (!enabled(input)) return out;
    out.shapes = read_shapes(input);
    for (const Shape & s : out.shapes) {
        if (std::find(block.zone_names.begin(), block.zone_names.end(), s.zone) == block.zone_names.end()) {
            throw InputError("mesh.curved: no boundary zone named " + s.zone + ".");
        }
    }
    std::vector<int32_t> zone_shape(block.zone_names.size(), -1);
    for (size_t z = 0; z < block.zone_names.size(); z++) zone_shape[z] = shape_of_zone(out.shapes, block.zone_names[z]);
    // Records [n_corners, shape + 1, n_nodes, corners...] and the nodes' coordinates
    std::vector<uint64_t> header;
    std::vector<double> coords;
    const bool high_order = block.face_high_order_offsets.size() == block.n_faces() + 1;
    for (uint64_t f = 0; f < block.n_faces(); f++) {
        const int32_t shape = zone_shape[block.face_zone[f]];
        const uint64_t h0 = high_order ? block.face_high_order_offsets[f] : 0;
        const uint64_t h1 = high_order ? block.face_high_order_offsets[f + 1] : 0;
        if (shape < 0 && h1 == h0) continue;
        const uint64_t n_nodes = shape >= 0 ? 0 : h1 - h0;
        header.push_back(block.face_offsets[f + 1] - block.face_offsets[f]);
        header.push_back(uint64_t(shape + 1));
        header.push_back(n_nodes);
        for (uint64_t k = block.face_offsets[f]; k < block.face_offsets[f + 1]; k++) header.push_back(block.face_nodes[k]);
        for (uint64_t k = 0; k < n_nodes; k++) {
            for (size_t i = 0; i < 3; i++) coords.push_back(i < N_DIM ? block.face_high_order_nodes[h0 + k][i] : 0.0);
        }
    }
    const std::vector<uint64_t> all_header = comm::allgatherv(header);
    const std::vector<double> all_coords = comm::allgatherv(coords);
    size_t h = 0, c = 0;
    while (h < all_header.size()) {
        SurfaceFace face;
        const uint64_t n_corners = all_header[h++];
        face.shape = int32_t(all_header[h++]) - 1;
        const uint64_t n_nodes = all_header[h++];
        face.corners.assign(all_header.begin() + std::ptrdiff_t(h), all_header.begin() + std::ptrdiff_t(h + n_corners));
        h += n_corners;
        for (uint64_t k = 0; k < n_nodes; k++, c += 3) face.nodes.push_back({all_coords[c], all_coords[c + 1], all_coords[c + 2]});
        out.faces.push_back(std::move(face));
    }
    check_high_order(out.faces);
    sort_unique(out.faces);
    return out;
}

Geometry::Geometry(const Mesh & mesh_in, const Surfaces & surfaces) : mesh(mesh_in), shapes(surfaces.shapes) {
    // Curved edges, from the first surface face that has them
    std::map<std::pair<uint64_t, uint64_t>, int32_t> edge_of;
    std::map<Key, size_t> surface_of;
    for (size_t r = 0; r < surfaces.faces.size(); r++) {
        const SurfaceFace & sf = surfaces.faces[r];
        surface_of.emplace(sorted_key(sf.corners), r);
        const uint32_t nc = uint32_t(sf.corners.size());
        const uint32_t g = sf.shape >= 0 ? 1u : element_order(nc, uint32_t(sf.nodes.size()));
        for (uint32_t k = 0; k < (nc == 2 ? 1u : nc); k++) {
            const uint64_t a = sf.corners[k], b = sf.corners[(k + 1) % nc];
            const auto key = std::make_pair(std::min(a, b), std::max(a, b));
            if (edge_of.count(key)) continue;
            Edge e;
            e.shape = sf.shape;
            if (sf.shape < 0) {
                for (uint32_t i : element_edge(nc, g, k)) e.nodes.push_back(sf.nodes[i]);
                if (a > b) std::reverse(e.nodes.begin(), e.nodes.end());
                // Straight high-order edges (Gmsh writes every boundary face at the mesh's
                // order, planes included) stay straight
                const Vec3 & p = e.nodes.front();
                const Vec3 d = sub(e.nodes.back(), p);
                const double length2 = dot(d, d);
                bool straight = true;
                for (size_t i = 1; i + 1 < e.nodes.size(); i++) {
                    const Vec3 lin = add(p, scale(double(i) / double(e.nodes.size() - 1), d));
                    const Vec3 dev = sub(e.nodes[i], lin);
                    straight &= dot(dev, dev) <= 1e-20 * length2;
                }
                if (straight) continue;
            }
            edge_of[key] = int32_t(edges.size());
            edges.push_back(std::move(e));
        }
    }
    auto edges_of = [&](const std::vector<uint64_t> & g, Face & face) {
        const uint32_t n = uint32_t(g.size());
        bool any = false;
        for (uint32_t k = 0; k < (n == 2 ? 1u : n); k++) {
            const uint64_t a = g[k], b = g[(k + 1) % n];
            const auto it = edge_of.find(std::make_pair(std::min(a, b), std::max(a, b)));
            if (it == edge_of.end()) continue;
            face.edge[k] = it->second;
            face.flip[k] = a > b;
            any = true;
        }
        return any;
    };
    // Interior corrections of high-order elements, from their interior nodes
    auto bubbles = [&](const SurfaceFace & sf, const std::vector<uint64_t> & local) {
        const uint32_t nc = uint32_t(sf.corners.size()), nn = uint32_t(sf.nodes.size());
        std::vector<std::array<double, 2>> xi;
        double factor = 1.0;
        if (nc == 3 && nn == 10) {
            xi = {{1.0 / 3.0, 1.0 / 3.0}};
        } else if (nc == 4 && nn == 9) {
            xi = {{0.0, 0.0}};
        } else if (nc == 4 && nn == 16) {
            const double a = 1.0 / 3.0;
            xi = {{-a, -a}, {a, -a}, {a, a}, {-a, a}};
            factor = 64.0 / 81.0;
        } else {
            return std::vector<Vec3>();
        }
        Face element;
        edges_of(sf.corners, element);
        const std::vector<Vec3> v(sf.nodes.begin(), sf.nodes.begin() + nc);
        std::vector<Vec3> c(xi.size());
        for (size_t j = 0; j < xi.size(); j++) {
            Vec3 x;
            face_eval(element, v, xi[j].data(), x, nullptr);
            c[j] = scale(1.0 / factor, sub(sf.nodes[nn - xi.size() + j], x));
        }
        if (c.size() == 1) return c;
        // Interior node j sits next to corner j: reorder by the local corners
        std::vector<Vec3> out(c.size());
        for (uint32_t m = 0; m < nc; m++) {
            out[m] = c[size_t(std::find(sf.corners.begin(), sf.corners.end(), local[m]) - sf.corners.begin())];
        }
        return out;
    };

    auto gid = [&](uint32_t node) -> uint64_t {
        return mesh.h_global_node_id.empty() ? node : mesh.h_global_node_id[node];
    };
    face_index.assign(mesh.n_faces, -1);
    curved_cell.assign(mesh.n_cells, 0);
    std::vector<uint64_t> g;
    for (uint32_t f = 0; f < mesh.n_faces; f++) {
        const uint32_t n = mesh.h_n_nodes_of_face(f);
        g.resize(n);
        for (uint32_t k = 0; k < n; k++) g[k] = gid(mesh.h_node_of_face(f, k));
        Face face;
        if (!edges_of(g, face)) continue;
        if (N_DIM == 3) {
            const auto it = surface_of.find(sorted_key(g));
            if (it != surface_of.end()) {
                const SurfaceFace & sf = surfaces.faces[it->second];
                if (sf.shape >= 0) {
                    face.shape = sf.shape;
                } else {
                    face.bubble = bubbles(sf, g);
                }
            }
        }
        face_index[f] = int32_t(faces.size());
        faces.push_back(std::move(face));
        curved_cell[uint32_t(mesh.h_cells_of_face(f, 0))] = 1;
        if (mesh.h_cells_of_face(f, 1) >= 0) curved_cell[uint32_t(mesh.h_cells_of_face(f, 1))] = 1;
    }
}

std::vector<Vec3> Geometry::face_nodes(uint32_t f) const {
    std::vector<Vec3> v(mesh.h_n_nodes_of_face(f), Vec3{0.0, 0.0, 0.0});
    for (size_t k = 0; k < v.size(); k++) {
        FOR_I_DIM v[k][i] = double(mesh.h_node_coords(mesh.h_node_of_face(f, uint8_t(k)), i));
    }
    return v;
}

void Geometry::edge_eval(const Face & face, size_t k, const Vec3 & a, const Vec3 & b, double t, Vec3 & x,
                         Vec3 & dx) const {
    const Edge & e = edges[size_t(face.edge[k])];
    const bool flip = face.flip[k];
    const double tc = flip ? 1.0 - t : t;
    const Vec3 & ca = flip ? b : a;
    const Vec3 & cb = flip ? a : b;
    Vec3 dxc;
    if (e.shape >= 0) {
        const Shape & s = shapes[size_t(e.shape)];
        double J[9], Ja[9], Jb[9];
        Vec3 p, pa, pb;
        const Vec3 lin = add(ca, scale(tc, sub(cb, ca)));
        s.project(lin, p, J);
        s.project(ca, pa, Ja);
        s.project(cb, pb, Jb);
        const Vec3 oa = sub(pa, ca), ob = sub(pb, cb);
        x = sub(sub(p, scale(1.0 - tc, oa)), scale(tc, ob));
        dxc = sub(matvec(J, sub(cb, ca)), sub(ob, oa));
    } else {
        const size_t g = e.nodes.size() - 1;
        double l[8], dl[8];
        lagrange(g, tc, l, dl);
        x = {0.0, 0.0, 0.0};
        dxc = {0.0, 0.0, 0.0};
        for (size_t i = 0; i <= g; i++) {
            x = add(x, scale(l[i], e.nodes[i]));
            dxc = add(dxc, scale(dl[i], e.nodes[i]));
        }
    }
    dx = flip ? scale(-1.0, dxc) : dxc;
}

void Geometry::face_eval(const Face & face, const std::vector<Vec3> & v, const double * xi, Vec3 & x,
                         Vec3 * dx) const {
    const uint32_t n = uint32_t(v.size());
    Vec3 d0 = {0.0, 0.0, 0.0}, d1 = {0.0, 0.0, 0.0};
    if (n == 2) {
        const double t = 0.5 * (xi[0] + 1.0);
        if (face.edge[0] >= 0) {
            edge_eval(face, 0, v[0], v[1], t, x, d0);
        } else {
            x = add(v[0], scale(t, sub(v[1], v[0])));
            d0 = sub(v[1], v[0]);
        }
        if (dx) dx[0] = scale(0.5, d0);
        return;
    }
    // Shape functions of the flat element and their derivatives
    double N[4], dN[4][2];
    if (n == 3) {
        const double u = xi[0], w = xi[1];
        N[0] = 1.0 - u - w;
        N[1] = u;
        N[2] = w;
        dN[0][0] = -1.0, dN[0][1] = -1.0;
        dN[1][0] = 1.0, dN[1][1] = 0.0;
        dN[2][0] = 0.0, dN[2][1] = 1.0;
    } else {
        const double s = xi[0], t = xi[1];
        const double sg[4] = {-1.0, 1.0, 1.0, -1.0}, tg[4] = {-1.0, -1.0, 1.0, 1.0};
        for (size_t k = 0; k < 4; k++) {
            N[k] = 0.25 * (1.0 + sg[k] * s) * (1.0 + tg[k] * t);
            dN[k][0] = 0.25 * sg[k] * (1.0 + tg[k] * t);
            dN[k][1] = 0.25 * tg[k] * (1.0 + sg[k] * s);
        }
    }
    x = {0.0, 0.0, 0.0};
    for (uint32_t k = 0; k < n; k++) {
        x = add(x, scale(N[k], v[k]));
        d0 = add(d0, scale(dN[k][0], v[k]));
        d1 = add(d1, scale(dN[k][1], v[k]));
    }
    if (face.shape >= 0) {
        // Projection of the flat point, less the corners' own projection offsets
        const Shape & s = shapes[size_t(face.shape)];
        Vec3 p;
        double J[9], Jk[9];
        s.project(x, p, J);
        d0 = matvec(J, d0);
        d1 = matvec(J, d1);
        x = p;
        for (uint32_t k = 0; k < n; k++) {
            Vec3 pk;
            s.project(v[k], pk, Jk);
            const Vec3 o = sub(pk, v[k]);
            x = sub(x, scale(N[k], o));
            d0 = sub(d0, scale(dN[k][0], o));
            d1 = sub(d1, scale(dN[k][1], o));
        }
    } else if (n == 3) {
        // Szabo-Babuska blending: lambda_i lambda_j q(t), q = d(t) / (t (1 - t)), t = (1 + lambda_j - lambda_i) / 2
        const double lam[3] = {N[0], N[1], N[2]};
        for (size_t k = 0; k < 3; k++) {
            if (face.edge[k] < 0) continue;
            const size_t i = k, j = (k + 1) % 3;
            const double t = 0.5 * (1.0 + lam[j] - lam[i]);
            const double den = t * (1.0 - t);
            if (!(den > 1e-300)) continue;
            Vec3 gx, gd;
            edge_eval(face, k, v[i], v[j], t, gx, gd);
            const Vec3 d = sub(gx, add(scale(1.0 - t, v[i]), scale(t, v[j])));
            const Vec3 dd = sub(gd, sub(v[j], v[i]));
            const Vec3 q = scale(1.0 / den, d);
            const Vec3 dq = scale(1.0 / den, sub(dd, scale(1.0 - 2.0 * t, q)));
            const double ll = lam[i] * lam[j];
            x = add(x, scale(ll, q));
            for (size_t m = 0; m < 2; m++) {
                const double dll = dN[i][m] * lam[j] + lam[i] * dN[j][m];
                const double dt = 0.5 * (dN[j][m] - dN[i][m]);
                const Vec3 term = add(scale(dll, q), scale(ll * dt, dq));
                if (m == 0) {
                    d0 = add(d0, term);
                } else {
                    d1 = add(d1, term);
                }
            }
        }
        if (!face.bubble.empty()) {
            const double b = 27.0 * lam[0] * lam[1] * lam[2];
            x = add(x, scale(b, face.bubble[0]));
            for (size_t m = 0; m < 2; m++) {
                const double db =
                    27.0 * (dN[0][m] * lam[1] * lam[2] + lam[0] * dN[1][m] * lam[2] + lam[0] * lam[1] * dN[2][m]);
                if (m == 0) {
                    d0 = add(d0, scale(db, face.bubble[0]));
                } else {
                    d1 = add(d1, scale(db, face.bubble[0]));
                }
            }
        }
    } else {
        // Coons patch of the edge deviations D_k (edge k from node k to node k + 1)
        const double s = xi[0], t = xi[1];
        const double a = 0.5 * (s + 1.0), b = 0.5 * (t + 1.0);
        auto deviation = [&](size_t k, double tau, Vec3 & D, Vec3 & dD) {
            D = {0.0, 0.0, 0.0};
            dD = {0.0, 0.0, 0.0};
            if (face.edge[k] < 0) return;
            const Vec3 & p = v[k];
            const Vec3 & q = v[(k + 1) % 4];
            Vec3 gx, gd;
            edge_eval(face, k, p, q, tau, gx, gd);
            D = sub(gx, add(scale(1.0 - tau, p), scale(tau, q)));
            dD = sub(gd, sub(q, p));
        };
        Vec3 D0, dD0, D1, dD1, D2, dD2, D3, dD3;
        deviation(0, a, D0, dD0);
        deviation(1, b, D1, dD1);
        deviation(2, 1.0 - a, D2, dD2);
        deviation(3, 1.0 - b, D3, dD3);
        x = add(x, add(add(scale(1.0 - b, D0), scale(b, D2)), add(scale(1.0 - a, D3), scale(a, D1))));
        const Vec3 da = add(add(scale(1.0 - b, dD0), scale(-b, dD2)), sub(D1, D3));
        const Vec3 db = add(sub(D2, D0), add(scale(-(1.0 - a), dD3), scale(a, dD1)));
        d0 = add(d0, scale(0.5, da));
        d1 = add(d1, scale(0.5, db));
        if (!face.bubble.empty()) {
            const double B = (1.0 - s * s) * (1.0 - t * t);
            const double dBs = -2.0 * s * (1.0 - t * t), dBt = -2.0 * t * (1.0 - s * s);
            Vec3 c = {0.0, 0.0, 0.0}, dcs = {0.0, 0.0, 0.0}, dct = {0.0, 0.0, 0.0};
            if (face.bubble.size() == 1) {
                c = face.bubble[0];
            } else {
                const double sg[4] = {-1.0, 1.0, 1.0, -1.0}, tg[4] = {-1.0, -1.0, 1.0, 1.0};
                for (size_t k = 0; k < 4; k++) {
                    const double Ls = 0.5 + 1.5 * sg[k] * s, Lt = 0.5 + 1.5 * tg[k] * t;
                    c = add(c, scale(Ls * Lt, face.bubble[k]));
                    dcs = add(dcs, scale(1.5 * sg[k] * Lt, face.bubble[k]));
                    dct = add(dct, scale(1.5 * tg[k] * Ls, face.bubble[k]));
                }
            }
            x = add(x, scale(B, c));
            d0 = add(d0, add(scale(dBs, c), scale(B, dcs)));
            d1 = add(d1, add(scale(dBt, c), scale(B, dct)));
        }
    }
    if (dx) {
        dx[0] = d0;
        dx[1] = d1;
    }
}

void Geometry::eval(uint32_t f, const double * xi, Vec3 & x, Vec3 * dx) const {
    static const Face flat;
    face_eval(face_index[f] >= 0 ? faces[size_t(face_index[f])] : flat, face_nodes(f), xi, x, dx);
}

Vec3 Geometry::area_vector(uint32_t f) const {
    const std::vector<Vec3> v = face_nodes(f);
    if (v.size() == 2) return rot(sub(v[1], v[0]));
    std::vector<double> g, gw;
    gauss_unit(12, g, gw);
    Vec3 A = {0.0, 0.0, 0.0};
    const int32_t idx = face_index[f];
    for (size_t k = 0; k < v.size(); k++) {
        const Vec3 & a = v[k];
        const Vec3 & b = v[(k + 1) % v.size()];
        if (idx < 0 || faces[size_t(idx)].edge[k] < 0) {
            A = add(A, scale(0.5, cross(a, b)));
            continue;
        }
        for (size_t q = 0; q < g.size(); q++) {
            Vec3 x, dx;
            edge_eval(faces[size_t(idx)], k, a, b, g[q], x, dx);
            A = add(A, scale(0.5 * gw[q], cross(x, dx)));
        }
    }
    return A;
}

void Geometry::face_rule(uint32_t f, const std::vector<std::array<double, 2>> & ref, const std::vector<double> & ref_w,
                         std::vector<Vec3> & x, std::vector<Vec3> & n, std::vector<double> & w) const {
    const size_t nq = ref.size();
    x.resize(nq);
    n.resize(nq);
    w.resize(nq);
    std::vector<Vec3> V(nq);
    Vec3 sum = {0.0, 0.0, 0.0};
    double w_sum = 0.0;
    for (size_t q = 0; q < nq; q++) {
        Vec3 dx[2];
        eval(f, ref[q].data(), x[q], dx);
        V[q] = scale(ref_w[q], N_DIM == 2 ? rot(dx[0]) : cross(dx[0], dx[1]));
        sum = add(sum, V[q]);
        w_sum += ref_w[q];
    }
    // A common shift of the area vectors, by the quadrature error of their sum
    const Vec3 shift = scale(1.0 / w_sum, sub(area_vector(f), sum));
    for (size_t q = 0; q < nq; q++) {
        V[q] = add(V[q], scale(ref_w[q], shift));
        w[q] = std::sqrt(dot(V[q], V[q]));
        n[q] = scale(1.0 / w[q], V[q]);
    }
}

void Geometry::cell_rule(uint32_t c, int degree, std::vector<Vec3> & x, std::vector<double> & w) const {
    x.clear();
    w.clear();
    const uint32_t n_nodes = mesh.h_n_nodes_of_cell(c);
    Vec3 o = {0.0, 0.0, 0.0};
    for (uint32_t k = 0; k < n_nodes; k++) {
        FOR_I_DIM o[i] += double(mesh.h_node_coords(mesh.h_node_of_cell(c, k), i));
    }
    o = scale(1.0 / n_nodes, o);
    std::vector<double> gt, gtw;
    gauss_unit(size_t(degree + N_DIM) / 2 + 1, gt, gtw);
    // Curved faces: enough points for maps of order 3, and for analytic ones to round-off
    const size_t n_curved = size_t(N_DIM == 2 ? std::max((3 * degree + 7) / 2, 12) : std::max(degree + 3, 8));
    const size_t n_straight = size_t(degree / 2 + 2);
    std::vector<std::array<double, 2>> ref;
    std::vector<double> ref_w;
    // Cone from o over a point F of a face with area vector N (out of the cell) per unit reference measure
    auto cone = [&](const Vec3 & F, const Vec3 & N, const double weight) {
        const Vec3 d = sub(F, o);
        const double base = weight * dot(d, N);
        for (size_t k = 0; k < gt.size(); k++) {
            x.push_back(add(o, scale(gt[k], d)));
            w.push_back(base * gtw[k] * (N_DIM == 2 ? gt[k] : gt[k] * gt[k]));
        }
    };
    for (uint32_t k = 0; k < mesh.h_n_faces_of_cell(c); k++) {
        const uint32_t f = mesh.h_face_of_cell(c, k);
        const double sign = (mesh.h_cells_of_face(f, 0) == int32_t(c)) ? 1.0 : -1.0;
        Vec3 offset = {0.0, 0.0, 0.0};
        FOR_I_DIM offset[i] = double(mesh.h_face_offset(f, c, i));
        const std::vector<Vec3> v = face_nodes(f);
        if (N_DIM == 2 || face_is_curved(f)) {
            reference_rule(uint32_t(v.size()), face_is_curved(f) ? n_curved : n_straight, ref, ref_w);
            for (size_t q = 0; q < ref.size(); q++) {
                Vec3 F, dF[2];
                eval(f, ref[q].data(), F, dF);
                cone(sub(F, offset), N_DIM == 2 ? rot(dF[0]) : cross(dF[0], dF[1]), sign * ref_w[q]);
            }
            continue;
        }
        // Straight faces: the triangles of the cell's tetrahedra (cell_tetrahedra)
        std::vector<std::array<Vec3, 3>> tris;
        if (v.size() == 3) {
            tris.push_back({v[0], v[1], v[2]});
        } else {
            Vec3 m = {0.0, 0.0, 0.0};
            for (const Vec3 & p : v) m = add(m, scale(1.0 / double(v.size()), p));
            for (size_t j = 0; j < v.size(); j++) tris.push_back({m, v[j], v[(j + 1) % v.size()]});
        }
        reference_rule(3, n_straight, ref, ref_w);
        for (const auto & tri : tris) {
            const Vec3 e1 = sub(tri[1], tri[0]), e2 = sub(tri[2], tri[0]);
            const Vec3 N = cross(e1, e2);
            for (size_t q = 0; q < ref.size(); q++) {
                const Vec3 F = add(tri[0], add(scale(ref[q][0], e1), scale(ref[q][1], e2)));
                cone(sub(F, offset), N, sign * ref_w[q]);
            }
        }
    }
}


double Geometry::max_shape_offset() const {
    double out = 0.0;
    for (uint32_t f = 0; f < mesh.n_faces; f++) {
        if (face_index[f] < 0) continue;
        const Face & face = faces[size_t(face_index[f])];
        const std::vector<Vec3> v = face_nodes(f);
        for (size_t k = 0; k < (v.size() == 2 ? 1 : v.size()); k++) {
            if (face.edge[k] < 0 || edges[size_t(face.edge[k])].shape < 0) continue;
            const Shape & s = shapes[size_t(edges[size_t(face.edge[k])].shape)];
            const Vec3 & a = v[k];
            const Vec3 & b = v[(k + 1) % v.size()];
            const double length = std::sqrt(dot(sub(b, a), sub(b, a)));
            for (const Vec3 * p : {&a, &b}) {
                Vec3 q;
                double J[9];
                s.project(*p, q, J);
                out = std::max(out, std::sqrt(dot(sub(q, *p), sub(q, *p))) / length);
            }
        }
    }
    return out;
}

}  // namespace curved

void Mesh::apply_curved(const curved::Surfaces & surfaces) {
    curved_geometry = nullptr;
    if (surfaces.empty()) return;
    auto geometry = std::make_shared<curved::Geometry>(*this, surfaces);
    const double offset = comm::allreduce(geometry->max_shape_offset(), comm::Op::MAX);
    if (offset > 0.05) {
        throw InputError("mesh.curved: the zone's nodes are " + std::to_string(offset) +
                         " edge lengths away from its shape (center or radius?).");
    }
    for (uint32_t f = 0; f < n_faces; f++) {
        if (!geometry->face_is_curved(f) || N_DIM == 2) continue;
        const curved::Vec3 A = geometry->area_vector(f);
        FOR_I_DIM h_face_normals(f, i) = rtype(A[i]);
        h_face_area(f) = rtype(std::sqrt(A[0] * A[0] + A[1] * A[1] + A[2] * A[2]));
    }
    std::vector<curved::Vec3> x;
    std::vector<double> w;
    for (uint32_t c = 0; c < n_cells; c++) {
        if (!geometry->cell_is_curved(c)) continue;
        geometry->cell_rule(c, 1, x, w);
        double V = 0.0;
        curved::Vec3 m = {0.0, 0.0, 0.0};
        for (size_t q = 0; q < w.size(); q++) {
            V += w[q];
            for (size_t i = 0; i < 3; i++) m[i] += w[q] * x[q][i];
        }
        if (!(V > 0.0)) {
            throw std::runtime_error("Mesh: curved cell " + std::to_string(c) + " has non-positive volume.");
        }
        h_cell_volume(c) = rtype(V);
        FOR_I_DIM h_cell_coords(c, i) = rtype(m[i] / V);
    }
    curved_geometry = geometry;
}

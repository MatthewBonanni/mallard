/**
 * @file mesh_3d.cpp
 * @author Matthew Bonanni (mbonanni001@gmail.com)
 * @brief 3D mesh construction and geometry.
 * @version 0.1
 * @date 2026-10-02
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 *
 */

#include "mesh.h"
#include "mesh_block.h"

#include <algorithm>
#include <cmath>
#include <map>
#include <stdexcept>

namespace {

using Vec3 = std::array<double, 3>;

Vec3 sub(const Vec3 & a, const Vec3 & b) { return {a[0] - b[0], a[1] - b[1], a[2] - b[2]}; }

Vec3 cross(const Vec3 & a, const Vec3 & b) {
    return {a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]};
}

double dot3(const Vec3 & a, const Vec3 & b) { return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]; }

Vec3 average(const std::vector<Vec3> & p) {
    Vec3 c = {0.0, 0.0, 0.0};
    for (const Vec3 & q : p) {
        for (size_t i = 0; i < 3; i++) c[i] += q[i];
    }
    for (size_t i = 0; i < 3; i++) c[i] /= static_cast<double>(p.size());
    return c;
}

double signed_volume(const std::vector<Vec3> & nodes, std::vector<std::array<Vec3, 4>> & tets) {
    cell_tetrahedra(nodes, tets);
    double v = 0.0;
    for (const auto & t : tets) v += dot3(sub(t[1], t[0]), cross(sub(t[2], t[0]), sub(t[3], t[0]))) / 6.0;
    return v;
}

/**
 * @brief Node order of the mirror image of a cell (reverses its orientation).
 */
std::vector<uint32_t> reflected(const std::vector<uint32_t> & c) {
    switch (c.size()) {
        case 4: return {c[0], c[2], c[1], c[3]};
        case 5: return {c[0], c[3], c[2], c[1], c[4]};
        case 6: return {c[0], c[2], c[1], c[3], c[5], c[4]};
        case 8: return {c[0], c[3], c[2], c[1], c[4], c[7], c[6], c[5]};
        default: throw std::runtime_error("Mesh: unsupported 3D cell with " + std::to_string(c.size()) + " nodes.");
    }
}

} // namespace

const std::vector<std::vector<uint8_t>> & cell_local_faces(uint32_t n_nodes) {
    static const std::vector<std::vector<uint8_t>> tet = {{0, 2, 1}, {0, 1, 3}, {0, 3, 2}, {1, 2, 3}};
    static const std::vector<std::vector<uint8_t>> pyramid = {{0, 3, 2, 1}, {0, 1, 4}, {1, 2, 4}, {2, 3, 4}, {3, 0, 4}};
    static const std::vector<std::vector<uint8_t>> prism = {{0, 2, 1}, {3, 4, 5}, {0, 1, 4, 3}, {1, 2, 5, 4}, {2, 0, 3, 5}};
    static const std::vector<std::vector<uint8_t>> hex = {{0, 3, 2, 1}, {4, 5, 6, 7}, {0, 1, 5, 4},
                                                          {1, 2, 6, 5}, {2, 3, 7, 6}, {3, 0, 4, 7}};
    switch (n_nodes) {
        case 4: return tet;
        case 5: return pyramid;
        case 6: return prism;
        case 8: return hex;
        default: throw std::runtime_error("Mesh: unsupported 3D cell with " + std::to_string(n_nodes) + " nodes.");
    }
}

void cell_tetrahedra(const std::vector<std::array<double, 3>> & nodes,
                     std::vector<std::array<std::array<double, 3>, 4>> & tets) {
    tets.clear();
    const Vec3 o = average(nodes);
    for (const auto & face : cell_local_faces(static_cast<uint32_t>(nodes.size()))) {
        const size_t n = face.size();
        if (n == 3) {
            tets.push_back({o, nodes[face[0]], nodes[face[1]], nodes[face[2]]});
            continue;
        }
        Vec3 c = {0.0, 0.0, 0.0};
        for (uint8_t k : face) {
            for (size_t i = 0; i < 3; i++) c[i] += nodes[k][i];
        }
        for (size_t i = 0; i < 3; i++) c[i] /= static_cast<double>(n);
        for (size_t k = 0; k < n; k++) tets.push_back({o, c, nodes[face[k]], nodes[face[(k + 1) % n]]});
    }
}

CellGeometry cell_geometry(const std::vector<std::array<rtype, N_DIM>> & cell_nodes, bool axisymmetric) {
    CellGeometry g{};
    if constexpr (N_DIM == 2) {
        // Counterclockwise, as Mesh::init_from_connectivity stores it
        rtype area2 = 0.0;
        for (size_t k = 0; k < cell_nodes.size(); k++) {
            const auto & a = cell_nodes[k];
            const auto & b = cell_nodes[(k + 1) % cell_nodes.size()];
            area2 += a[0] * b[1] - b[0] * a[1];
        }
        thread_local std::vector<std::array<rtype, N_DIM>> reversed;
        if (area2 < 0.0_r) reversed.assign(cell_nodes.rbegin(), cell_nodes.rend());
        const auto & nodes = area2 < 0.0_r ? reversed : cell_nodes;
        if (nodes.size() == 3) {
            g.volume = triangle_area<2>(nodes[0].data(), nodes[1].data(), nodes[2].data());
        } else if (nodes.size() == 4) {
            const rtype a1 = triangle_area<2>(nodes[0].data(), nodes[1].data(), nodes[2].data());
            const rtype a2 = triangle_area<2>(nodes[0].data(), nodes[2].data(), nodes[3].data());
            g.volume = a1 + a2;
        } else {
            throw std::runtime_error("Unknown cell type.");
        }
        // Area centroid of the polygon (the vertex average is only correct for
        // triangles and parallelograms), relative to the cell's first node: in
        // absolute coordinates the cross products cancel, leaving errors of
        // eps |x|^2 / h, which break the equidistance of mirror-equivalent
        // stencil entries on fine meshes away from the origin
        const size_t n = nodes.size();
        const rtype x0 = nodes[0][0], y0 = nodes[0][1];
        rtype A = 0.0, Cx = 0.0, Cy = 0.0;
        for (size_t k = 0; k < n; ++k) {
            const auto & a = nodes[k];
            const auto & b = nodes[(k + 1) % n];
            const rtype xa = a[0] - x0, ya = a[1] - y0;
            const rtype xb = b[0] - x0, yb = b[1] - y0;
            const rtype cross = xa * yb - xb * ya;
            A += 0.5_r * cross;
            Cx += (xa + xb) * cross;
            Cy += (ya + yb) * cross;
        }
        g.centroid[0] = x0 + Cx / (6.0_r * A);
        g.centroid[1] = y0 + Cy / (6.0_r * A);
        g.measure = g.volume;
        if (axisymmetric) {
            const AxisymmetricCell a = axisymmetric_cell(nodes);
            g.measure = rtype(a.measure);
            g.centroid[0] = rtype(a.centroid[0]);
            g.centroid[1] = rtype(a.centroid[1]);
        }
    } else {
        if (axisymmetric) throw std::runtime_error("Axisymmetric flows need the 2D solver.");
        thread_local std::vector<Vec3> p;
        thread_local std::vector<std::array<Vec3, 4>> tets;
        p.resize(cell_nodes.size());
        for (size_t k = 0; k < p.size(); k++) {
            for (size_t i = 0; i < 3; i++) p[k][i] = (i < N_DIM) ? double(cell_nodes[k][i]) : 0.0;
        }
        double V = 0.0;
        Vec3 x = {0.0, 0.0, 0.0};
        // Twice for a negatively oriented cell: reflected, as Mesh::orient_cells_3d stores it
        for (int pass = 0; pass < 2; pass++) {
            cell_tetrahedra(p, tets);
            V = 0.0;
            x = {0.0, 0.0, 0.0};
            for (const auto & t : tets) {
                const double v = dot3(sub(t[1], t[0]), cross(sub(t[2], t[0]), sub(t[3], t[0]))) / 6.0;
                V += v;
                for (size_t i = 0; i < 3; i++) x[i] += v * (t[0][i] + t[1][i] + t[2][i] + t[3][i]) / 4.0;
            }
            if (V >= 0.0 || pass == 1) break;
            std::vector<uint32_t> order(p.size());
            for (uint32_t k = 0; k < order.size(); k++) order[k] = k;
            const std::vector<Vec3> q = p;
            const std::vector<uint32_t> r = reflected(order);
            for (size_t k = 0; k < p.size(); k++) p[k] = q[r[k]];
        }
        g.volume = static_cast<rtype>(V);
        g.measure = g.volume;
        FOR_I_DIM g.centroid[i] = static_cast<rtype>(x[i] / V);
    }
    return g;
}

void Mesh::h_cell_tetrahedra(uint32_t i_cell, std::vector<std::array<std::array<double, 3>, 4>> & tets) const {
    thread_local std::vector<Vec3> p;
    p.resize(h_n_nodes_of_cell(i_cell));
    for (uint32_t k = 0; k < p.size(); k++) {
        const uint32_t node = h_node_of_cell(i_cell, k);
        for (size_t i = 0; i < 3; i++) p[k][i] = (i < N_DIM) ? double(h_node_coords(node, i)) : 0.0;
    }
    cell_tetrahedra(p, tets);
}

void Mesh::compute_geometry() {
    if constexpr (N_DIM == 2) {
        compute_face_areas();
        compute_cell_volumes_and_centroids();
        compute_face_normals();
        compute_face_centroids();
        compute_cell_neighbors();
        return;
    }
    auto coords = [&](uint32_t node) {
        Vec3 p = {0.0, 0.0, 0.0};
        FOR_I_DIM p[i] = double(h_node_coords(node, i));
        return p;
    };
    // Faces: area vector and centroid from a triangle fan around the vertex
    // average, which stays exact for warped quadrilaterals and closes every cell
    std::vector<Vec3> p, a;
    for (uint32_t f = 0; f < n_faces; f++) {
        p.resize(h_n_nodes_of_face(f));
        for (uint32_t k = 0; k < p.size(); k++) p[k] = coords(h_node_of_face(f, k));
        const Vec3 c = average(p);
        Vec3 A = {0.0, 0.0, 0.0};
        a.resize(p.size());
        for (size_t k = 0; k < p.size(); k++) {
            a[k] = cross(sub(p[k], c), sub(p[(k + 1) % p.size()], c));
            for (size_t i = 0; i < 3; i++) A[i] += 0.5 * a[k][i];
        }
        const double area = std::sqrt(dot3(A, A));
        Vec3 x = {0.0, 0.0, 0.0};
        double w_sum = 0.0;
        for (size_t k = 0; k < p.size(); k++) {
            const double w = 0.5 * dot3(a[k], A) / area;
            const Vec3 & q = p[(k + 1) % p.size()];
            for (size_t i = 0; i < 3; i++) x[i] += w * (c[i] + p[k][i] + q[i]) / 3.0;
            w_sum += w;
        }
        h_face_area(f) = static_cast<rtype>(area);
        FOR_I_DIM {
            h_face_normals(f, i) = static_cast<rtype>(A[i]);
            h_face_coords(f, i) = static_cast<rtype>(x[i] / w_sum);
        }
    }
    compute_cell_volumes_and_centroids();
    compute_cell_neighbors();
}

void Mesh::compute_cell_volumes_and_centroids() {
    std::vector<std::array<rtype, N_DIM>> nodes;
    for (uint32_t c = 0; c < n_cells; c++) {
        nodes.resize(h_n_nodes_of_cell(c));
        for (uint32_t k = 0; k < nodes.size(); k++) FOR_I_DIM nodes[k][i] = h_node_coords(h_node_of_cell(c, k), i);
        const CellGeometry g = cell_geometry(nodes);
        if (N_DIM == 3 && g.volume <= 0.0_r) {
            throw std::runtime_error("Mesh: cell " + std::to_string(c) + " has non-positive volume.");
        }
        h_cell_volume(c) = g.volume;
        FOR_I_DIM h_cell_coords(c, i) = g.centroid[i];
    }
}

void Mesh::orient_cells_3d(const std::vector<std::array<rtype, N_DIM>> & nodes,
                           const std::vector<std::vector<uint32_t>> & cells, std::vector<uint32_t> & offsets,
                           std::vector<uint32_t> & cell_nodes) {
    offsets.assign(1, 0);
    cell_nodes.clear();
    std::vector<Vec3> p;
    std::vector<std::array<Vec3, 4>> tets;
    for (const auto & c : cells) {
        cell_local_faces(static_cast<uint32_t>(c.size()));
        p.clear();
        for (uint32_t node : c) {
            Vec3 x = {0.0, 0.0, 0.0};
            FOR_I_DIM x[i] = double(nodes[node][i]);
            p.push_back(x);
        }
        if (signed_volume(p, tets) < 0.0) {
            const auto r = reflected(c);
            cell_nodes.insert(cell_nodes.end(), r.begin(), r.end());
        } else {
            cell_nodes.insert(cell_nodes.end(), c.begin(), c.end());
        }
        offsets.push_back(static_cast<uint32_t>(cell_nodes.size()));
    }
}

void Mesh::init_cart_3d(uint32_t nx, uint32_t ny, uint32_t nz, rtype Lx, rtype Ly, rtype Lz, MeshType kind,
                        const std::vector<PeriodicPair> & periodic, const Stretching & stretching) {
    const auto [cells, nodes] = cartesian_3d_size(nx, ny, nz, kind);
    init_from_block(cartesian_3d_block(nx, ny, nz, Lx, Ly, Lz, kind, 0, cells, 0, nodes, stretching), periodic);
}

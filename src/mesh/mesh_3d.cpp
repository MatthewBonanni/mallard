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
        for (int i = 0; i < 3; i++) c[i] += q[i];
    }
    for (int i = 0; i < 3; i++) c[i] /= p.size();
    return c;
}

double signed_volume(const std::vector<Vec3> & nodes) {
    std::vector<std::array<Vec3, 4>> tets;
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
    for (const auto & face : cell_local_faces(nodes.size())) {
        std::vector<Vec3> p;
        for (uint8_t k : face) p.push_back(nodes[k]);
        if (p.size() == 3) {
            tets.push_back({o, p[0], p[1], p[2]});
            continue;
        }
        const Vec3 c = average(p);
        for (size_t k = 0; k < p.size(); k++) tets.push_back({o, c, p[k], p[(k + 1) % p.size()]});
    }
}

void Mesh::h_cell_tetrahedra(uint32_t i_cell, std::vector<std::array<std::array<double, 3>, 4>> & tets) const {
    std::vector<Vec3> p(h_n_nodes_of_cell(i_cell));
    for (size_t k = 0; k < p.size(); k++) {
        const uint32_t node = h_node_of_cell(i_cell, k);
        for (int i = 0; i < 3; i++) p[k][i] = (i < N_DIM) ? h_node_coords(node, i) : 0.0;
    }
    cell_tetrahedra(p, tets);
}

void Mesh::compute_geometry() {
    if constexpr (N_DIM == 2) {
        compute_face_areas();
        compute_cell_volumes();
        compute_cell_centroids();
        compute_face_normals();
        compute_face_centroids();
        compute_cell_neighbors();
        return;
    }
    auto coords = [&](uint32_t node) {
        Vec3 p = {0.0, 0.0, 0.0};
        FOR_I_DIM p[i] = h_node_coords(node, i);
        return p;
    };
    // Faces: area vector and centroid from a triangle fan around the vertex
    // average, which stays exact for warped quadrilaterals and closes every cell
    for (uint32_t f = 0; f < n_faces; f++) {
        std::vector<Vec3> p(h_n_nodes_of_face(f));
        for (size_t k = 0; k < p.size(); k++) p[k] = coords(h_node_of_face(f, k));
        const Vec3 c = average(p);
        Vec3 A = {0.0, 0.0, 0.0};
        std::vector<Vec3> a(p.size());
        for (size_t k = 0; k < p.size(); k++) {
            a[k] = cross(sub(p[k], c), sub(p[(k + 1) % p.size()], c));
            for (int i = 0; i < 3; i++) A[i] += 0.5 * a[k][i];
        }
        const double area = std::sqrt(dot3(A, A));
        Vec3 x = {0.0, 0.0, 0.0};
        double w_sum = 0.0;
        for (size_t k = 0; k < p.size(); k++) {
            const double w = 0.5 * dot3(a[k], A) / area;
            const Vec3 & q = p[(k + 1) % p.size()];
            for (int i = 0; i < 3; i++) x[i] += w * (c[i] + p[k][i] + q[i]) / 3.0;
            w_sum += w;
        }
        h_face_area(f) = area;
        FOR_I_DIM {
            h_face_normals(f, i) = A[i];
            h_face_coords(f, i) = x[i] / w_sum;
        }
    }
    // Cells: volume and centroid by tetrahedral decomposition
    std::vector<std::array<Vec3, 4>> tets;
    for (uint32_t c = 0; c < n_cells; c++) {
        h_cell_tetrahedra(c, tets);
        double V = 0.0;
        Vec3 x = {0.0, 0.0, 0.0};
        for (const auto & t : tets) {
            const double v = dot3(sub(t[1], t[0]), cross(sub(t[2], t[0]), sub(t[3], t[0]))) / 6.0;
            V += v;
            for (int i = 0; i < 3; i++) x[i] += v * (t[0][i] + t[1][i] + t[2][i] + t[3][i]) / 4.0;
        }
        if (V <= 0.0) {
            throw std::runtime_error("Mesh: cell " + std::to_string(c) + " has non-positive volume.");
        }
        h_cell_volume(c) = V;
        FOR_I_DIM h_cell_coords(c, i) = x[i] / V;
    }
    compute_cell_neighbors();
}

void Mesh::init_from_connectivity_3d(const std::vector<std::array<rtype, N_DIM>> & nodes,
                                     const std::vector<std::vector<uint32_t>> & cells,
                                     const std::vector<BoundaryFace> & boundary_faces) {
    n_nodes = nodes.size();
    n_cells = cells.size();
    auto coords = [&](uint32_t node) {
        Vec3 p = {0.0, 0.0, 0.0};
        FOR_I_DIM p[i] = nodes[node][i];
        return p;
    };

    // Orient every cell positively (Gmsh/VTK convention)
    std::vector<std::vector<uint32_t>> cell_nodes = cells;
    for (auto & c : cell_nodes) {
        cell_local_faces(c.size());
        std::vector<Vec3> p;
        for (uint32_t node : c) p.push_back(coords(node));
        if (signed_volume(p) < 0.0) c = reflected(c);
    }

    // Faces keyed by their sorted node set, ordered outward from their first cell
    std::map<std::vector<uint32_t>, uint32_t> face_of_key;
    std::vector<std::vector<uint32_t>> face_nodes;
    std::vector<std::array<int32_t, 2>> face_cells;
    std::vector<std::vector<uint32_t>> cell_faces(n_cells);
    for (uint32_t c = 0; c < n_cells; c++) {
        for (const auto & local : cell_local_faces(cell_nodes[c].size())) {
            std::vector<uint32_t> fn;
            for (uint8_t k : local) fn.push_back(cell_nodes[c][k]);
            std::vector<uint32_t> key = fn;
            std::sort(key.begin(), key.end());
            auto it = face_of_key.find(key);
            if (it == face_of_key.end()) {
                face_of_key.emplace(key, face_nodes.size());
                cell_faces[c].push_back(face_nodes.size());
                face_nodes.push_back(fn);
                face_cells.push_back({(int32_t)c, -1});
            } else {
                if (face_cells[it->second][1] != -1) {
                    throw std::runtime_error("Mesh: a face is shared by more than two cells.");
                }
                face_cells[it->second][1] = c;
                cell_faces[c].push_back(it->second);
            }
        }
    }
    n_faces = face_nodes.size();

    std::map<std::string, std::vector<uint32_t>> zone_faces;
    std::vector<uint32_t> interior;
    std::vector<bool> zoned(n_faces, false);
    for (const auto & bf : boundary_faces) {
        std::vector<uint32_t> key = bf.nodes;
        std::sort(key.begin(), key.end());
        auto it = face_of_key.find(key);
        if (it == face_of_key.end()) {
            throw std::runtime_error("Mesh: boundary face of " + bf.zone + " is not a cell face.");
        }
        if (face_cells[it->second][1] != -1) {
            throw std::runtime_error("Mesh: boundary face of " + bf.zone + " is an interior face.");
        }
        if (!zoned[it->second]) {
            zone_faces[bf.zone].push_back(it->second);
            zoned[it->second] = true;
        }
    }
    for (uint32_t f = 0; f < n_faces; f++) {
        if (face_cells[f][1] >= 0) {
            interior.push_back(f);
        } else if (!zoned[f]) {
            zone_faces["unassigned"].push_back(f);
        }
    }

    allocate_and_fill(nodes, cell_nodes, cell_faces, face_nodes, face_cells, interior, zone_faces);
    compute_geometry();
}

void Mesh::init_cart_3d(uint32_t nx, uint32_t ny, uint32_t nz, rtype Lx, rtype Ly, rtype Lz, MeshType kind) {
    const auto [n_cells, n_nodes] = cartesian_3d_size(nx, ny, nz, kind);
    init_from_block(cartesian_3d_block(nx, ny, nz, Lx, Ly, Lz, kind, 0, n_cells, 0, n_nodes));
}

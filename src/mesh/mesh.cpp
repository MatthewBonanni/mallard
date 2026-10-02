/**
 * @file mesh.cpp
 * @author Matthew Bonanni (mbonanni001@gmail.com)
 * @brief Mesh class implementation.
 * @version 0.1
 * @date 2023-12-17
 *
 * @copyright Copyright (c) 2023 Matthew Bonanni
 *
 */

#include "mesh.h"
#include "mesh_block.h"

#include "input.h"

#include <iostream>
#include <string>
#include <cmath>
#include <algorithm>
#include <numeric>

#include <Kokkos_Core.hpp>

#include "common.h"
#include "boundary.h"

Mesh::Mesh() {
    // Empty
}

Mesh::~Mesh() {
    std::cout << "Destroying mesh: " << MESH_NAMES.at(type) << std::endl;
}

void Mesh::init(const toml::value & input) {
    std::string type_str = toml::find_or<std::string>(input, "mesh", "type", "file");
    typename std::unordered_map<std::string, MeshType>::const_iterator it = MESH_TYPES.find(type_str);
    if (it == MESH_TYPES.end()) {
        throw std::runtime_error("Unknown mesh type: " + type_str + ".");
    } else {
        set_type(it->second);
    }

    if (get_type() == MeshType::FILE) {
        std::string filename = toml::find_or<std::string>(input, "mesh", "filename", "mesh.msh");
        this->init_file(filename);
        return;
    }
    uint32_t Nx = toml::find_or<uint32_t>(input, "mesh", "Nx", 100);
    uint32_t Ny = toml::find_or<uint32_t>(input, "mesh", "Ny", 100);
    rtype Lx = find_real_or(input, "mesh", "Lx", 1.0);
    rtype Ly = find_real_or(input, "mesh", "Ly", 1.0);
    if constexpr (N_DIM == 3) {
        const uint32_t Nz = toml::find_or<uint32_t>(input, "mesh", "Nz", 100);
        const rtype Lz = find_real_or(input, "mesh", "Lz", 1.0);
        if (get_type() == MeshType::CARTESIAN_TRI || get_type() == MeshType::WEDGE) {
            throw std::runtime_error("Mesh type " + type_str + " is 2D only.");
        }
        this->init_cart_3d(Nx, Ny, Nz, Lx, Ly, Lz, get_type());
        return;
    }
    if (get_type() == MeshType::CARTESIAN) {
        this->init_cart(Nx, Ny, Lx, Ly);
    } else if (get_type() == MeshType::CARTESIAN_TRI) {
        this->init_cart_tri(Nx, Ny, Lx, Ly);
    } else if (get_type() == MeshType::WEDGE) {
        this->init_wedge(Nx, Ny, Lx, Ly);
    } else {
        throw std::runtime_error("Mesh type " + type_str + " is 3D only.");
    }
}

MeshType Mesh::get_type() const {
    return type;
}

void Mesh::set_type(MeshType type) {
    this->type = type;
}

uint32_t Mesh::n_face_zones() const {
    return m_face_zones.size();
}

std::vector<FaceZone> * Mesh::face_zones() {
    return &m_face_zones;
}

FaceZone * Mesh::get_face_zone(const std::string& name) {
    for (uint32_t i = 0; i < n_face_zones(); ++i) {
        if (m_face_zones[i].get_name() == name) {
            return &(m_face_zones[i]);
        }
    }
    return nullptr;
}

CellType Mesh::h_cell_type(uint32_t i_cell) const {
    if constexpr (N_DIM == 3) {
        switch (h_n_nodes_of_cell(i_cell)) {
            case 4: return CellType::TETRAHEDRON;
            case 5: return CellType::PYRAMID;
            case 6: return CellType::PRISM;
            case 8: return CellType::HEXAHEDRON;
            default: throw std::runtime_error("Unknown cell type.");
        }
    }
    if (h_n_nodes_of_cell(i_cell) == 4) {
        return CellType::QUAD;
    } else if (h_n_nodes_of_cell(i_cell) == 3) {
        return CellType::TRIANGLE;
    } else {
        throw std::runtime_error("Unknown cell type.");
    }
}

uint32_t Mesh::h_n_nodes_of_cell(uint32_t i_cell) const {
    return h_offsets_nodes_of_cell(i_cell + 1) - h_offsets_nodes_of_cell(i_cell);
}

uint32_t Mesh::h_n_faces_of_cell(uint32_t i_cell) const {
    return h_offsets_faces_of_cell(i_cell + 1) - h_offsets_faces_of_cell(i_cell);
}

uint32_t Mesh::h_n_nodes_of_face(uint32_t i_face) const {
    return h_offsets_nodes_of_face(i_face + 1) - h_offsets_nodes_of_face(i_face);
}

uint32_t Mesh::h_node_of_cell(uint32_t i_cell, uint8_t i_node_local) const {
    return h_nodes_of_cell(h_offsets_nodes_of_cell(i_cell) + i_node_local);
}

uint32_t Mesh::h_face_of_cell(uint32_t i_cell, uint8_t i_face_local) const {
    return h_faces_of_cell(h_offsets_faces_of_cell(i_cell) + i_face_local);
}

uint32_t Mesh::h_node_of_face(uint32_t i_face, uint8_t i_node_local) const {
    return h_nodes_of_face(h_offsets_nodes_of_face(i_face) + i_node_local);
}

void Mesh::h_neighbors_of_cell_helper(uint32_t i_cell, uint8_t n_order, std::vector<uint32_t> & neighbors) const {
    // Warning: results are not sorted and may contain duplicates
    // List will contain the current cv and its neighbors up to n_neighbors graph distance

    // Add the current cv to the list
    neighbors.push_back(i_cell);

    if (n_order == 0) {
        // Base case - no more neighbors to add
        return;
    } else {
        // Iterate over the neighbors of the current cv, and call the function recursively
        for (uint8_t i_face_local = 0; i_face_local < h_n_faces_of_cell(i_cell); ++i_face_local) {
            uint32_t i_face = h_face_of_cell(i_cell, i_face_local);
            int32_t i_cell_0 = h_cells_of_face(i_face, 0);
            int32_t i_cell_1 = h_cells_of_face(i_face, 1);
            if (i_cell_1 == -1) {
                // This is a boundary face, so skip it
                continue;
            } else {
                // Recursively call the function for the neighbor cv
                if (i_cell_0 == (int32_t)i_cell) {
                    h_neighbors_of_cell_helper(i_cell_1, n_order - 1, neighbors);
                } else {
                    h_neighbors_of_cell_helper(i_cell_0, n_order - 1, neighbors);
                }
            }
        }
    }
}

void Mesh::h_neighbors_of_cell(uint32_t i_cell, uint8_t n_order, std::vector<uint32_t> & neighbors) const {
    // Get the neighbors, unsorted and with duplicates
    h_neighbors_of_cell_helper(i_cell, n_order, neighbors);
    
    // Sort and remove duplicates
    std::sort(neighbors.begin(), neighbors.end());
    neighbors.erase(std::unique(neighbors.begin(), neighbors.end()), neighbors.end());
}

void Mesh::compute_cell_centroids() {
    // Area centroid of the polygon (the vertex average is only correct for
    // triangles and parallelograms)
    for (uint32_t i_cell = 0; i_cell < n_cells; ++i_cell) {
        const uint32_t n_nodes = h_n_nodes_of_cell(i_cell);
        rtype A = 0.0, Cx = 0.0, Cy = 0.0;
        for (uint32_t k = 0; k < n_nodes; ++k) {
            const uint32_t a = h_node_of_cell(i_cell, k);
            const uint32_t b = h_node_of_cell(i_cell, (k + 1) % n_nodes);
            const rtype xa = h_node_coords(a, 0), ya = h_node_coords(a, 1);
            const rtype xb = h_node_coords(b, 0), yb = h_node_coords(b, 1);
            const rtype cross = xa * yb - xb * ya;
            A += 0.5 * cross;
            Cx += (xa + xb) * cross;
            Cy += (ya + yb) * cross;
        }
        h_cell_coords(i_cell, 0) = Cx / (6.0 * A);
        h_cell_coords(i_cell, 1) = Cy / (6.0 * A);
    }
}

void Mesh::compute_cell_volumes() {
    for (uint32_t i_cell = 0; i_cell < n_cells; ++i_cell) {
        switch (h_cell_type(i_cell)) {
            case CellType::TRIANGLE: {
                uint32_t i_node_0 = h_node_of_cell(i_cell, 0);
                uint32_t i_node_1 = h_node_of_cell(i_cell, 1);
                uint32_t i_node_2 = h_node_of_cell(i_cell, 2);

                const NVector coords_0 = {h_node_coords(i_node_0, 0), h_node_coords(i_node_0, 1)};
                const NVector coords_1 = {h_node_coords(i_node_1, 0), h_node_coords(i_node_1, 1)};
                const NVector coords_2 = {h_node_coords(i_node_2, 0), h_node_coords(i_node_2, 1)};

                h_cell_volume(i_cell) = triangle_area<2>(coords_0.data(), coords_1.data(), coords_2.data());
                break;
            }
            case CellType::QUAD: {
                uint32_t i_node_0 = h_node_of_cell(i_cell, 0);
                uint32_t i_node_1 = h_node_of_cell(i_cell, 1);
                uint32_t i_node_2 = h_node_of_cell(i_cell, 2);
                uint32_t i_node_3 = h_node_of_cell(i_cell, 3);

                const NVector coords_0 = {h_node_coords(i_node_0, 0), h_node_coords(i_node_0, 1)};
                const NVector coords_1 = {h_node_coords(i_node_1, 0), h_node_coords(i_node_1, 1)};
                const NVector coords_2 = {h_node_coords(i_node_2, 0), h_node_coords(i_node_2, 1)};
                const NVector coords_3 = {h_node_coords(i_node_3, 0), h_node_coords(i_node_3, 1)};

                rtype a1 = triangle_area<2>(coords_0.data(), coords_1.data(), coords_2.data());
                rtype a2 = triangle_area<2>(coords_0.data(), coords_2.data(), coords_3.data());
                h_cell_volume(i_cell) = a1 + a2;
                break;
            }
            default:
                throw std::runtime_error("Unknown cell type.");
        }
    }
}

void Mesh::compute_face_areas() {
    for (uint32_t i_face = 0; i_face < n_faces; ++i_face) {
        uint32_t i_node_0 = h_node_of_face(i_face, 0);
        uint32_t i_node_1 = h_node_of_face(i_face, 1);
        h_face_area(i_face) = std::sqrt(std::pow(h_node_coords(i_node_1, 0) -
                                                 h_node_coords(i_node_0, 0), 2) +
                                        std::pow(h_node_coords(i_node_1, 1) -
                                                 h_node_coords(i_node_0, 1), 2));
    }
}

void Mesh::compute_face_normals() {
    for (uint32_t i_face = 0; i_face < n_faces; ++i_face) {
        // Compute normal with area magnitude
        uint32_t i_node_0 = h_node_of_face(i_face, 0);
        uint32_t i_node_1 = h_node_of_face(i_face, 1);
        rtype x0 = h_node_coords(i_node_0, 0);
        rtype y0 = h_node_coords(i_node_0, 1);
        rtype x1 = h_node_coords(i_node_1, 0);
        rtype y1 = h_node_coords(i_node_1, 1);
        rtype dx = x1 - x0;
        rtype dy = y1 - y0;
        rtype mag = std::sqrt(dx * dx + dy * dy);
        h_face_normals(i_face, 0) =  dy / mag * h_face_area(i_face);
        h_face_normals(i_face, 1) = -dx / mag * h_face_area(i_face);

        // Flip normal if it points into cell 0
        // (shouln't be necessary for meshes generated by this class,
        // but just in case, for example if the mesh is read from a file)
        int32_t i_cell_0 = h_cells_of_face(i_face, 0);
        rtype x_cell_0 = h_cell_coords(i_cell_0, 0);
        rtype y_cell_0 = h_cell_coords(i_cell_0, 1);
        rtype x_face_centroid = 0.5 * (x0 + x1);
        rtype y_face_centroid = 0.5 * (y0 + y1);
        rtype dx_cell_0 = x_face_centroid - x_cell_0;
        rtype dy_cell_0 = y_face_centroid - y_cell_0;
        rtype dot = dx_cell_0 * h_face_normals(i_face, 0) +
                    dy_cell_0 * h_face_normals(i_face, 1);
        if (dot < 0) {
            h_face_normals(i_face, 0) *= -1;
            h_face_normals(i_face, 1) *= -1;
        }
    }
}

void Mesh::compute_face_centroids() {
    for (uint32_t i_face = 0; i_face < n_faces; ++i_face) {
        uint32_t i_node_0 = h_node_of_face(i_face, 0);
        uint32_t i_node_1 = h_node_of_face(i_face, 1);
        FOR_I_DIM {
            h_face_coords(i_face, i) = 0.5 * (h_node_coords(i_node_0, i) + h_node_coords(i_node_1, i));
        }
    }
}

std::vector<uint32_t> Mesh::cells_by_global_id() const {
    std::vector<uint32_t> order(n_cells);
    std::iota(order.begin(), order.end(), 0u);
    std::sort(order.begin(), order.end(), [&](uint32_t a, uint32_t b) { return h_global_cell(a) < h_global_cell(b); });
    return order;
}

void Mesh::compute_cell_neighbors() {
    // Cells of every node (CSR, cells in increasing order)
    std::vector<uint32_t> node_offsets(n_nodes + 1, 0), node_cells;
    for (uint32_t c = 0; c < n_cells; c++) {
        for (uint32_t k = 0; k < h_n_nodes_of_cell(c); k++) node_offsets[h_node_of_cell(c, k) + 1]++;
    }
    for (uint32_t n = 0; n < n_nodes; n++) node_offsets[n + 1] += node_offsets[n];
    node_cells.resize(node_offsets[n_nodes]);
    {
        std::vector<uint32_t> fill(node_offsets.begin(), node_offsets.end() - 1);
        for (uint32_t c = 0; c < n_cells; c++) {
            for (uint32_t k = 0; k < h_n_nodes_of_cell(c); k++) node_cells[fill[h_node_of_cell(c, k)]++] = c;
        }
    }
    std::vector<uint32_t> offsets(n_cells + 1, 0), flat, nb;
    for (uint32_t c = 0; c < n_cells; c++) {
        nb.clear();
        for (uint32_t k = 0; k < h_n_nodes_of_cell(c); k++) {
            const uint32_t n = h_node_of_cell(c, k);
            for (uint32_t i = node_offsets[n]; i < node_offsets[n + 1]; i++) {
                if (node_cells[i] != c) nb.push_back(node_cells[i]);
            }
        }
        std::sort(nb.begin(), nb.end(), [&](uint32_t a, uint32_t b) { return h_global_cell(a) < h_global_cell(b); });
        nb.erase(std::unique(nb.begin(), nb.end()), nb.end());
        flat.insert(flat.end(), nb.begin(), nb.end());
        offsets[c + 1] = flat.size();
    }
    offsets_cells_of_cell = Kokkos::View<uint32_t *>("offsets_cells_of_cell", n_cells + 1);
    cells_of_cell = Kokkos::View<uint32_t *>("cells_of_cell", flat.size());
    auto h_offsets = Kokkos::create_mirror_view(offsets_cells_of_cell);
    auto h_cells = Kokkos::create_mirror_view(cells_of_cell);
    for (uint32_t i = 0; i <= n_cells; i++) h_offsets(i) = offsets[i];
    for (size_t i = 0; i < flat.size(); i++) h_cells(i) = flat[i];
    Kokkos::deep_copy(offsets_cells_of_cell, h_offsets);
    Kokkos::deep_copy(cells_of_cell, h_cells);
}

void Mesh::copy_host_to_device() {
    Kokkos::deep_copy(node_coords, h_node_coords);
    Kokkos::deep_copy(cell_coords, h_cell_coords);
    Kokkos::deep_copy(cell_volume, h_cell_volume);
    Kokkos::deep_copy(face_area, h_face_area);
    Kokkos::deep_copy(face_normals, h_face_normals);
    Kokkos::deep_copy(face_coords, h_face_coords);
    Kokkos::deep_copy(nodes_of_cell, h_nodes_of_cell);
    Kokkos::deep_copy(offsets_nodes_of_cell, h_offsets_nodes_of_cell);
    Kokkos::deep_copy(faces_of_cell, h_faces_of_cell);
    Kokkos::deep_copy(offsets_faces_of_cell, h_offsets_faces_of_cell);
    Kokkos::deep_copy(nodes_of_face, h_nodes_of_face);
    Kokkos::deep_copy(offsets_nodes_of_face, h_offsets_nodes_of_face);
    Kokkos::deep_copy(cells_of_face, h_cells_of_face);
    for (auto & zone : m_face_zones) {
        zone.copy_host_to_device();
    }
    for (auto & zone : m_cell_zones) {
        zone.copy_host_to_device();
    }
}

void Mesh::copy_device_to_host() {
    Kokkos::deep_copy(h_node_coords, node_coords);
    Kokkos::deep_copy(h_cell_coords, cell_coords);
    Kokkos::deep_copy(h_cell_volume, cell_volume);
    Kokkos::deep_copy(h_face_area, face_area);
    Kokkos::deep_copy(h_face_normals, face_normals);
    Kokkos::deep_copy(h_face_coords, face_coords);
    Kokkos::deep_copy(h_nodes_of_cell, nodes_of_cell);
    Kokkos::deep_copy(h_offsets_nodes_of_cell, offsets_nodes_of_cell);
    Kokkos::deep_copy(h_faces_of_cell, faces_of_cell);
    Kokkos::deep_copy(h_offsets_faces_of_cell, offsets_faces_of_cell);
    Kokkos::deep_copy(h_nodes_of_face, nodes_of_face);
    Kokkos::deep_copy(h_offsets_nodes_of_face, offsets_nodes_of_face);
    Kokkos::deep_copy(h_cells_of_face, cells_of_face);
    for (auto & zone : m_face_zones) {
        zone.copy_device_to_host();
    }
    for (auto & zone : m_cell_zones) {
        zone.copy_device_to_host();
    }
}

void Mesh::init_box(uint32_t nx, uint32_t ny, rtype Lx, rtype Ly, bool triangles, bool wedge) {
    const rtype dx = Lx / nx;
    const rtype dy = Ly / ny;
    std::vector<std::array<rtype, N_DIM>> nodes;
    for (uint32_t i = 0; i < nx + 1; ++i) {
        for (uint32_t j = 0; j < ny + 1; ++j) {
            std::array<rtype, 2> x = {i * dx, j * dy};
            if (wedge) x = wedge_node(x[0], x[1], Ly);
            std::array<rtype, N_DIM> p{};
            p[0] = x[0];
            p[1] = x[1];
            nodes.push_back(p);
        }
    }
    auto node = [&](uint32_t i, uint32_t j) { return i * (ny + 1) + j; };
    std::vector<std::vector<uint32_t>> cells;
    for (uint32_t ic = 0; ic < nx; ++ic) {
        for (uint32_t jc = 0; jc < ny; ++jc) {
            const uint32_t tr = node(ic + 1, jc + 1), tl = node(ic, jc + 1);
            const uint32_t bl = node(ic, jc), br = node(ic + 1, jc);
            if (triangles) {
                cells.push_back({br, tr, bl});
                cells.push_back({tl, bl, tr});
            } else {
                cells.push_back({tr, tl, bl, br});
            }
        }
    }
    std::vector<BoundaryFace> boundary_faces;
    for (uint32_t i = 0; i < nx; ++i) {
        boundary_faces.push_back({{node(i, 0), node(i + 1, 0)}, "bottom"});
        boundary_faces.push_back({{node(i + 1, ny), node(i, ny)}, "top"});
    }
    for (uint32_t j = 0; j < ny; ++j) {
        boundary_faces.push_back({{node(nx, j), node(nx, j + 1)}, "right"});
        boundary_faces.push_back({{node(0, j + 1), node(0, j)}, "left"});
    }
    init_from_connectivity(nodes, cells, boundary_faces);
}

void Mesh::init_cart(uint32_t nx, uint32_t ny, rtype Lx, rtype Ly) {
    init_box(nx, ny, Lx, Ly, false, false);
}

void Mesh::init_cart_tri(uint32_t nx, uint32_t ny, rtype Lx, rtype Ly) {
    init_box(nx, ny, Lx, Ly, true, false);
}

void Mesh::init_wedge(uint32_t nx, uint32_t ny, rtype Lx, rtype Ly) {
    init_box(nx, ny, Lx, Ly, false, true);
}

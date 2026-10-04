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
    // Empty
}

void Mesh::init(const toml::value & input) {
    std::string type_str = toml::find_or<std::string>(input, "mesh", "type", "file");
    typename std::unordered_map<std::string, MeshType>::const_iterator it = MESH_TYPES.find(type_str);
    if (it == MESH_TYPES.end()) {
        throw unknown_option(MESH_TYPES, "mesh.type", type_str);
    } else {
        set_type(it->second);
    }

    const std::vector<PeriodicPair> periodic = periodic_pairs(input);
    if (get_type() == MeshType::FROM_FILE) {
        std::string filename = toml::find_or<std::string>(input, "mesh", "filename", "mesh.msh");
        this->init_file(filename, periodic);
        return;
    }
    uint32_t Nx = toml::find_or<uint32_t>(input, "mesh", "Nx", 100);
    uint32_t Ny = toml::find_or<uint32_t>(input, "mesh", "Ny", 100);
    rtype Lx = find_real_or(input, "mesh", "Lx", 1.0);
    rtype Ly = find_real_or(input, "mesh", "Ly", 1.0);
    const Stretching stretching = mesh_stretching(input);
    if constexpr (N_DIM == 3) {
        const uint32_t Nz = toml::find_or<uint32_t>(input, "mesh", "Nz", 100);
        const rtype Lz = find_real_or(input, "mesh", "Lz", 1.0);
        if (get_type() == MeshType::CARTESIAN_TRI || get_type() == MeshType::WEDGE) {
            throw std::runtime_error("Mesh type " + type_str + " is 2D only.");
        }
        this->init_cart_3d(Nx, Ny, Nz, Lx, Ly, Lz, get_type(), periodic, stretching);
        return;
    }
    if (!periodic.empty() || stretching != Stretching{}) {
        if (get_type() == MeshType::WEDGE && !periodic.empty()) {
            throw std::runtime_error("The wedge mesh cannot be periodic.");
        }
        if (get_type() != MeshType::CARTESIAN && get_type() != MeshType::CARTESIAN_TRI &&
            get_type() != MeshType::WEDGE) {
            throw std::runtime_error("Mesh type " + type_str + " is 3D only.");
        }
        init_from_block(cartesian_2d_block(Nx, Ny, Lx, Ly, get_type(), 0, 1, stretching), periodic);
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

std::vector<Mesh::PeriodicPair> Mesh::periodic_pairs(const toml::value & input) {
    std::vector<PeriodicPair> pairs;
    if (input.contains("periodic")) {
        const auto entries = toml::find<std::vector<toml::value>>(input, "periodic");
        for (const auto & entry : entries) {
            const auto zones = toml::find<std::vector<std::string>>(entry, "zones");
            const std::vector<rtype> t = find_real_vector(entry, "translation");
            if (zones.size() != 2 || t.size() != N_DIM) {
                throw InputError("[[periodic]] needs zones = [A, B] and a translation with " +
                                 std::to_string(N_DIM) + " components.");
            }
            PeriodicPair pair{zones[0], zones[1], {}};
            FOR_I_DIM pair.translation[i] = t[i];
            pairs.push_back(pair);
        }
    }
    if (!input.contains("mesh") || !input.at("mesh").contains("periodic")) return pairs;
    if (toml::find_or<std::string>(input, "mesh", "type", "file") == "file") {
        throw InputError("[mesh] periodic applies to generated meshes; pair the zones of a mesh file with "
                         "[[periodic]].");
    }
    const rtype L[3] = {find_real_or(input, "mesh", "Lx", 1.0), find_real_or(input, "mesh", "Ly", 1.0),
                        find_real_or(input, "mesh", "Lz", 1.0)};
    const char * zones[3][2] = {{"left", "right"}, {"bottom", "top"}, {"back", "front"}};
    for (const std::string & dir : toml::find<std::vector<std::string>>(input, "mesh", "periodic")) {
        const int d = (dir == "x") ? 0 : (dir == "y") ? 1 : (dir == "z") ? 2 : -1;
        if (d < 0 || d >= N_DIM) {
            throw std::runtime_error("[mesh] periodic: unknown direction " + dir + " (one of x, y" +
                                     (N_DIM == 3 ? ", z" : "") + ").");
        }
        for (const auto & pair : pairs) {
            if (pair.zone_a == zones[d][0]) throw std::runtime_error("[mesh] periodic lists " + dir + " twice.");
        }
        PeriodicPair pair{zones[d][0], zones[d][1], {}};
        pair.translation[d] = L[d];
        pairs.push_back(pair);
    }
    return pairs;
}

MeshType Mesh::get_type() const {
    return type;
}

void Mesh::set_type(MeshType type_in) {
    this->type = type_in;
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
                if (i_cell_0 == static_cast<int32_t>(i_cell)) {
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
    // triangles and parallelograms), relative to the cell's first node: in
    // absolute coordinates the cross products cancel, leaving errors of
    // eps |x|^2 / h, which break the equidistance of mirror-equivalent
    // stencil entries on fine meshes away from the origin
    for (uint32_t i_cell = 0; i_cell < n_cells; ++i_cell) {
        const uint32_t n = h_n_nodes_of_cell(i_cell);
        const uint32_t o = h_node_of_cell(i_cell, 0);
        const rtype x0 = h_node_coords(o, 0), y0 = h_node_coords(o, 1);
        rtype A = 0.0, Cx = 0.0, Cy = 0.0;
        for (uint32_t k = 0; k < n; ++k) {
            const uint32_t a = h_node_of_cell(i_cell, k);
            const uint32_t b = h_node_of_cell(i_cell, (k + 1) % n);
            const rtype xa = h_node_coords(a, 0) - x0, ya = h_node_coords(a, 1) - y0;
            const rtype xb = h_node_coords(b, 0) - x0, yb = h_node_coords(b, 1) - y0;
            const rtype cross = xa * yb - xb * ya;
            A += 0.5_r * cross;
            Cx += (xa + xb) * cross;
            Cy += (ya + yb) * cross;
        }
        h_cell_coords(i_cell, 0) = x0 + Cx / (6.0_r * A);
        h_cell_coords(i_cell, 1) = y0 + Cy / (6.0_r * A);
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
        rtype x_face_centroid = 0.5_r * (x0 + x1);
        rtype y_face_centroid = 0.5_r * (y0 + y1);
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
            h_face_coords(i_face, i) = 0.5_r * (h_node_coords(i_node_0, i) + h_node_coords(i_node_1, i));
        }
    }
}

std::vector<uint32_t> Mesh::cells_by_global_id() const {
    std::vector<uint32_t> order(n_cells);
    std::iota(order.begin(), order.end(), 0u);
    std::sort(order.begin(), order.end(), [&](uint32_t a, uint32_t b) { return h_global_cell(a) < h_global_cell(b); });
    return order;
}

uint8_t Mesh::shift_index(const std::array<int8_t, 3> & lattice) {
    for (size_t i = 0; i < shift_lattice.size(); i++) {
        if (shift_lattice[i] == lattice) return i;
    }
    if (shift_lattice.size() == 256) throw std::runtime_error("Mesh: too many periodic translations.");
    shift_lattice.push_back(lattice);
    return shift_lattice.size() - 1;
}

void Mesh::compute_cell_neighbors() {
    // Meshes built without periodic pairs
    if (shift_lattice.empty()) shift_lattice.assign(1, {0, 0, 0});
    if (h_face_shift.extent(0) != n_faces) {
        face_shift = Kokkos::View<uint8_t *>("face_shift", n_faces);
        h_face_shift = Kokkos::create_mirror_view(face_shift);
        Kokkos::deep_copy(h_face_shift, 0);
    }
    // Cells sharing a node key with each cell, with the translation that brings
    // them next to it: L(a) - L(b) for the cell's node a and the other's node b
    const bool periodic = !h_node_key.empty();
    auto key = [&](uint32_t n) { return periodic ? h_node_key[n] : n; };
    // Cells of every key and their node of that key (CSR, cells in increasing order)
    std::vector<uint32_t> key_offsets(n_nodes + 1, 0);
    std::vector<std::array<uint32_t, 2>> key_cells;
    for (uint32_t c = 0; c < n_cells; c++) {
        for (uint32_t k = 0; k < h_n_nodes_of_cell(c); k++) key_offsets[key(h_node_of_cell(c, k)) + 1]++;
    }
    for (uint32_t n = 0; n < n_nodes; n++) key_offsets[n + 1] += key_offsets[n];
    key_cells.resize(key_offsets[n_nodes]);
    {
        std::vector<uint32_t> fill(key_offsets.begin(), key_offsets.end() - 1);
        for (uint32_t c = 0; c < n_cells; c++) {
            for (uint32_t k = 0; k < h_n_nodes_of_cell(c); k++) {
                const uint32_t n = h_node_of_cell(c, k);
                key_cells[fill[key(n)]++] = {c, n};
            }
        }
    }
    // Neighbors as (cell << 8 | shift row)
    std::vector<uint32_t> offsets(n_cells + 1, 0);
    std::vector<uint64_t> flat, nb;
    for (uint32_t c = 0; c < n_cells; c++) {
        nb.clear();
        for (uint32_t k = 0; k < h_n_nodes_of_cell(c); k++) {
            const uint32_t a = h_node_of_cell(c, k);
            for (uint32_t i = key_offsets[key(a)]; i < key_offsets[key(a) + 1]; i++) {
                const auto [other, b] = key_cells[i];
                uint8_t shift = 0;
                if (periodic) {
                    std::array<int8_t, 3> lattice;
                    for (int j = 0; j < 3; j++) lattice[j] = h_node_lattice[a][j] - h_node_lattice[b][j];
                    shift = shift_index(lattice);
                }
                if (other == c && shift == 0) continue;
                if (other == c) {
                    throw std::runtime_error("Mesh: a cell touches itself across a periodic boundary; periodic "
                                             "directions need at least 3 cells.");
                }
                nb.push_back(uint64_t(other) << 8 | shift);
            }
        }
        // By global id, so that neighbor order does not depend on the local numbering
        std::sort(nb.begin(), nb.end(), [&](uint64_t a, uint64_t b) {
            return std::make_pair(h_global_cell(a >> 8), a & 0xff) < std::make_pair(h_global_cell(b >> 8), b & 0xff);
        });
        nb.erase(std::unique(nb.begin(), nb.end()), nb.end());
        for (size_t i = 1; i < nb.size(); i++) {
            if (nb[i] >> 8 == nb[i - 1] >> 8) {
                throw std::runtime_error("Mesh: two cells touch both directly and across a periodic boundary; "
                                         "periodic directions need at least 3 cells.");
            }
        }
        flat.insert(flat.end(), nb.begin(), nb.end());
        offsets[c + 1] = flat.size();
    }
    offsets_cells_of_cell = Kokkos::View<uint32_t *>("offsets_cells_of_cell", n_cells + 1);
    cells_of_cell = Kokkos::View<uint32_t *>("cells_of_cell", flat.size());
    cells_of_cell_shift = Kokkos::View<uint8_t *>("cells_of_cell_shift", flat.size());
    h_offsets_cells_of_cell = Kokkos::create_mirror_view(offsets_cells_of_cell);
    h_cells_of_cell = Kokkos::create_mirror_view(cells_of_cell);
    h_cells_of_cell_shift = Kokkos::create_mirror_view(cells_of_cell_shift);
    for (uint32_t i = 0; i <= n_cells; i++) h_offsets_cells_of_cell(i) = offsets[i];
    for (size_t i = 0; i < flat.size(); i++) {
        h_cells_of_cell(i) = flat[i] >> 8;
        h_cells_of_cell_shift(i) = flat[i] & 0xff;
    }

    shifts = Kokkos::View<rtype *[N_DIM]>("shifts", shift_lattice.size());
    h_shifts = Kokkos::create_mirror_view(shifts);
    for (size_t r = 0; r < shift_lattice.size(); r++) {
        FOR_I_DIM {
            double x = 0.0;
            for (size_t j = 0; j < periodic_translations.size(); j++) {
                x += shift_lattice[r][j] * double(periodic_translations[j][i]);
            }
            h_shifts(r, i) = x;
        }
    }
    Kokkos::deep_copy(offsets_cells_of_cell, h_offsets_cells_of_cell);
    Kokkos::deep_copy(cells_of_cell, h_cells_of_cell);
    Kokkos::deep_copy(cells_of_cell_shift, h_cells_of_cell_shift);
    Kokkos::deep_copy(shifts, h_shifts);
    Kokkos::deep_copy(face_shift, h_face_shift);
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
    Kokkos::deep_copy(shifts, h_shifts);
    Kokkos::deep_copy(face_shift, h_face_shift);
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
    Kokkos::deep_copy(h_shifts, shifts);
    Kokkos::deep_copy(h_face_shift, face_shift);
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

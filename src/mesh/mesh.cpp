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

#include <iostream>
#include <cmath>

#include <Kokkos_Core.hpp>

#include "common.h"
#include "boundary.h"

Mesh::Mesh(MeshType type) {
    this->type = type;
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
        throw std::runtime_error("MeshType::FILE not implemented.");
    } else if (get_type() == MeshType::CARTESIAN) {
        u_int32_t Nx = toml::find_or<u_int32_t>(input, "mesh", "Nx", 100);
        u_int32_t Ny = toml::find_or<u_int32_t>(input, "mesh", "Ny", 100);
        rtype Lx = toml::find_or<rtype>(input, "mesh", "Lx", 1.0);
        rtype Ly = toml::find_or<rtype>(input, "mesh", "Ly", 1.0);
        this->init_cart(Nx, Ny, Lx, Ly);
    } else if (get_type() == MeshType::WEDGE) {
        u_int32_t Nx = toml::find_or<u_int32_t>(input, "mesh", "Nx", 100);
        u_int32_t Ny = toml::find_or<u_int32_t>(input, "mesh", "Ny", 100);
        rtype Lx = toml::find_or<rtype>(input, "mesh", "Lx", 1.0);
        rtype Ly = toml::find_or<rtype>(input, "mesh", "Ly", 1.0);
        this->init_wedge(Nx, Ny, Lx, Ly);
    } else {
        // Should never get here due to the enum class.
        throw std::runtime_error("Unknown mesh type.");
    }
}

MeshType Mesh::get_type() const {
    return type;
}

void Mesh::set_type(MeshType type) {
    this->type = type;
}

u_int32_t Mesh::n_cells() const {
    return nx * ny;
}

u_int32_t Mesh::n_cells_x() const {
    return nx;
}

u_int32_t Mesh::n_cells_y() const {
    return ny;
}

u_int32_t Mesh::n_nodes() const {
    return (nx + 1) * (ny + 1);
}

u_int32_t Mesh::n_faces() const {
    return 2 * n_cells() + nx + ny;
}

u_int32_t Mesh::n_face_zones() const {
    return m_face_zones.size();
}

std::vector<FaceZone> * Mesh::face_zones() {
    return &m_face_zones;
}

FaceZone * Mesh::get_face_zone(const std::string& name) {
    for (u_int32_t i = 0; i < n_face_zones(); ++i) {
        if (m_face_zones[i].get_name() == name) {
            return &(m_face_zones[i]);
        }
    }
    return nullptr;
}

void Mesh::compute_cell_centroids() {
    for (u_int32_t i_cell = 0; i_cell < n_cells(); ++i_cell) {
        h_cell_coords(i_cell, 0) = 0.25 * (h_node_coords(node_of_cell(i_cell, 0), 0) +
                                           h_node_coords(node_of_cell(i_cell, 1), 0) +
                                           h_node_coords(node_of_cell(i_cell, 2), 0) +
                                           h_node_coords(node_of_cell(i_cell, 3), 0));
        h_cell_coords(i_cell, 1) = 0.25 * (h_node_coords(node_of_cell(i_cell, 0), 1) +
                                           h_node_coords(node_of_cell(i_cell, 1), 1) +
                                           h_node_coords(node_of_cell(i_cell, 2), 1) +
                                           h_node_coords(node_of_cell(i_cell, 3), 1));
    }
}

void Mesh::compute_face_centroids() {
    for (u_int32_t i_face = 0; i_face < n_faces(); ++i_face) {
        h_face_coords(i_face, 0) = 0.5 * (h_node_coords(node_of_face(i_face, 0), 0) +
                                          h_node_coords(node_of_face(i_face, 1), 0));
        h_face_coords(i_face, 1) = 0.5 * (h_node_coords(node_of_face(i_face, 0), 1) +
                                          h_node_coords(node_of_face(i_face, 1), 1));
    }
}

void Mesh::compute_cell_volumes() {
    for (u_int32_t i_cell = 0; i_cell < n_cells(); ++i_cell) {
        u_int32_t i_node_0 = node_of_cell(i_cell, 0);
        u_int32_t i_node_1 = node_of_cell(i_cell, 1);
        u_int32_t i_node_2 = node_of_cell(i_cell, 2);
        u_int32_t i_node_3 = node_of_cell(i_cell, 3);

        const NVector coords_0 = {h_node_coords(i_node_0, 0), h_node_coords(i_node_0, 1)};
        const NVector coords_1 = {h_node_coords(i_node_1, 0), h_node_coords(i_node_1, 1)};
        const NVector coords_2 = {h_node_coords(i_node_2, 0), h_node_coords(i_node_2, 1)};
        const NVector coords_3 = {h_node_coords(i_node_3, 0), h_node_coords(i_node_3, 1)};

        rtype a1 = triangle_area_2(coords_0, coords_1, coords_2);
        rtype a2 = triangle_area_2(coords_0, coords_2, coords_3);
        h_cell_volume(i_cell) = a1 + a2;
    }
}

void Mesh::compute_face_areas() {
    for (u_int32_t i_face = 0; i_face < n_faces(); ++i_face) {
        u_int32_t i_node_0 = node_of_face(i_face, 0);
        u_int32_t i_node_1 = node_of_face(i_face, 1);
        h_face_area(i_face) = sqrt(pow(h_node_coords(i_node_1, 0) -
                                       h_node_coords(i_node_0, 0), 2) +
                                   pow(h_node_coords(i_node_1, 1) -
                                       h_node_coords(i_node_0, 1), 2));
    }
}

void Mesh::compute_face_normals() {
    for (u_int32_t i_face = 0; i_face < n_faces(); ++i_face) {
        // Compute normal with area magnitude
        u_int32_t i_node_0 = node_of_face(i_face, 0);
        u_int32_t i_node_1 = node_of_face(i_face, 1);
        rtype x0 = h_node_coords(i_node_0, 0);
        rtype y0 = h_node_coords(i_node_0, 1);
        rtype x1 = h_node_coords(i_node_1, 0);
        rtype y1 = h_node_coords(i_node_1, 1);
        rtype dx = x1 - x0;
        rtype dy = y1 - y0;
        rtype mag = sqrt(dx * dx + dy * dy);
        h_face_normals(i_face, 0) =  dy / mag * face_area(i_face);
        h_face_normals(i_face, 1) = -dx / mag * face_area(i_face);

        // Flip normal if it points into cell 0
        // (shouln't be necessary for meshes generated by this class,
        // but just in case, for example if the mesh is read from a file)
        int32_t i_cell_0 = h_cells_of_face(i_face, 0);
        rtype x_cell_0 = h_cell_coords(i_cell_0, 0);
        rtype y_cell_0 = h_cell_coords(i_cell_0, 1);
        rtype x_face = h_face_coords(i_face, 0);
        rtype y_face = h_face_coords(i_face, 1);
        rtype dx_cell_0 = x_face - x_cell_0;
        rtype dy_cell_0 = y_face - y_cell_0;
        rtype dot = dx_cell_0 * h_face_normals(i_face, 0) +
                    dy_cell_0 * h_face_normals(i_face, 1);
        if (dot < 0) {
            h_face_normals(i_face, 0) *= -1;
            h_face_normals(i_face, 1) *= -1;
        }
    }
}

void Mesh::copy_host_to_device() {
    Kokkos::deep_copy(node_coords, h_node_coords);
    Kokkos::deep_copy(cell_coords, h_cell_coords);
    Kokkos::deep_copy(face_coords, h_face_coords);
    Kokkos::deep_copy(cell_volume, h_cell_volume);
    Kokkos::deep_copy(face_area, h_face_area);
    Kokkos::deep_copy(face_normals, h_face_normals);
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
    Kokkos::deep_copy(h_face_coords, face_coords);
    Kokkos::deep_copy(h_cell_volume, cell_volume);
    Kokkos::deep_copy(h_face_area, face_area);
    Kokkos::deep_copy(h_face_normals, face_normals);
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

void Mesh::init_cart(u_int32_t nx, u_int32_t ny, rtype Lx, rtype Ly) {
    /** \todo These two lines are a hack for WENO, remove this */
    this->nx = nx;
    this->ny = ny;

    node_coords = Kokkos::View<rtype *[N_DIM]>("node_coords", n_nodes());
    cell_coords = Kokkos::View<rtype *[N_DIM]>("cell_coords", n_cells());
    face_coords = Kokkos::View<rtype *[N_DIM]>("face_coords", n_faces());
    cell_volume = Kokkos::View<rtype *>("cell_volume", n_cells());
    face_area = Kokkos::View<rtype *>("face_area", n_faces());
    face_normals = Kokkos::View<rtype *[N_DIM]>("face_normals", n_faces());
    cells_of_face = Kokkos::View<int32_t *[2]>("cells_of_face", n_faces());

    h_node_coords = Kokkos::create_mirror_view(node_coords);
    h_cell_coords = Kokkos::create_mirror_view(cell_coords);
    h_face_coords = Kokkos::create_mirror_view(face_coords);
    h_cell_volume = Kokkos::create_mirror_view(cell_volume);
    h_face_area = Kokkos::create_mirror_view(face_area);
    h_face_normals = Kokkos::create_mirror_view(face_normals);
    h_cells_of_face = Kokkos::create_mirror_view(cells_of_face);

    // Temporary connectivity arrays
    std::vector<std::vector<u_int32_t>> _nodes_of_cell;
    std::vector<std::vector<u_int32_t>> _faces_of_cell;
    std::vector<std::vector<u_int32_t>> _nodes_of_face;

    // In this case, we know the sizes of the connectivity arrays a priori
    _nodes_of_cell.resize(n_cells());
    _faces_of_cell.resize(n_cells());
    _nodes_of_face.resize(n_faces());
    for (u_int32_t i_cell = 0; i_cell < n_cells(); ++i_cell) {
        _nodes_of_cell[i_cell].resize(4);
        _faces_of_cell[i_cell].resize(4);
    }
    for (u_int32_t i_face = 0; i_face < n_faces(); ++i_face) {
        _nodes_of_face[i_face].resize(2);
    }

    // Compute node coordinates
    rtype dx = Lx / nx;
    rtype dy = Ly / ny;

    for (u_int32_t i = 0; i < nx + 1; ++i) {
        for (u_int32_t j = 0; j < ny + 1; ++j) {
            u_int32_t i_node = i * (ny + 1) + j;
            h_node_coords(i_node, 0) = i * dx;
            h_node_coords(i_node, 1) = j * dy;
        }
    }

    // Build associations between nodes, cells, and faces
    for (u_int32_t i_cell = 0; i_cell < n_cells(); ++i_cell) {
        u_int32_t ic = i_cell / ny;
        u_int32_t jc = i_cell % ny;
        _nodes_of_cell[i_cell][0] = (ic + 1) * (ny + 1) + jc + 1; // Top right
        _nodes_of_cell[i_cell][1] = (ic)     * (ny + 1) + jc + 1; // Top left
        _nodes_of_cell[i_cell][2] = (ic)     * (ny + 1) + jc    ; // Bottom left
        _nodes_of_cell[i_cell][3] = (ic + 1) * (ny + 1) + jc    ; // Bottom right

        _faces_of_cell[i_cell][0] = (2 * ny + 1) * (ic + 1) + jc         ; // Right
        _faces_of_cell[i_cell][1] = (2 * ny + 1) * (ic)     + jc + ny + 1; // Top
        _faces_of_cell[i_cell][2] = (2 * ny + 1) * (ic)     + jc         ; // Left
        _faces_of_cell[i_cell][3] = (2 * ny + 1) * (ic)     + jc + ny    ; // Bottom

        // NOTE: cells_of_face and nodes_of_face will be overwritten
        // by future loop iterations, but this is okay because
        // the values will be the same for all cells.

        h_cells_of_face(_faces_of_cell[i_cell][0], 0) = i_cell;      // Right face - center cell
        h_cells_of_face(_faces_of_cell[i_cell][0], 1) = i_cell + ny; // Right face - right cell
        h_cells_of_face(_faces_of_cell[i_cell][1], 0) = i_cell;      // Top face - center cell
        h_cells_of_face(_faces_of_cell[i_cell][1], 1) = i_cell + 1;  // Top face - top cell
        h_cells_of_face(_faces_of_cell[i_cell][2], 0) = i_cell;      // Left face - center cell
        h_cells_of_face(_faces_of_cell[i_cell][2], 1) = i_cell - ny; // Left face - left cell
        h_cells_of_face(_faces_of_cell[i_cell][3], 0) = i_cell;      // Bottom face - center cell
        h_cells_of_face(_faces_of_cell[i_cell][3], 1) = i_cell - 1;  // Bottom face - bottom cell

        _nodes_of_face[_faces_of_cell[i_cell][0]][0] = _nodes_of_cell[i_cell][3]; // Right face - bottom right node
        _nodes_of_face[_faces_of_cell[i_cell][0]][1] = _nodes_of_cell[i_cell][0]; // Right face - top right node
        _nodes_of_face[_faces_of_cell[i_cell][1]][0] = _nodes_of_cell[i_cell][0]; // Top face - top right node
        _nodes_of_face[_faces_of_cell[i_cell][1]][1] = _nodes_of_cell[i_cell][1]; // Top face - top left node
        _nodes_of_face[_faces_of_cell[i_cell][2]][0] = _nodes_of_cell[i_cell][1]; // Left face - top left node
        _nodes_of_face[_faces_of_cell[i_cell][2]][1] = _nodes_of_cell[i_cell][2]; // Left face - bottom left node
        _nodes_of_face[_faces_of_cell[i_cell][3]][0] = _nodes_of_cell[i_cell][2]; // Bottom face - bottom left node
        _nodes_of_face[_faces_of_cell[i_cell][3]][1] = _nodes_of_cell[i_cell][3]; // Bottom face - bottom right node
    }

    // Handle boundary faces
    FaceZone zone_r = FaceZone();
    FaceZone zone_t = FaceZone();
    FaceZone zone_l = FaceZone();
    FaceZone zone_b = FaceZone();
    zone_r.set_name("right");
    zone_t.set_name("top");
    zone_l.set_name("left");
    zone_b.set_name("bottom");
    zone_r.set_type(FaceZoneType::BOUNDARY);
    zone_t.set_type(FaceZoneType::BOUNDARY);
    zone_l.set_type(FaceZoneType::BOUNDARY);
    zone_b.set_type(FaceZoneType::BOUNDARY);

    std::vector<u_int32_t> faces_r, faces_t, faces_l, faces_b;
    for (u_int32_t i_cell = 0; i_cell < n_cells(); i_cell++) {
        u_int32_t ic = i_cell / ny;
        u_int32_t jc = i_cell % ny;
        u_int32_t i_face;

        // Right boundary
        if (ic == nx - 1) {
            i_face = _faces_of_cell[i_cell][0];
            h_cells_of_face(i_face, 1) = -1;
            faces_r.push_back(i_face);
        }
        // Top boundary
        if (jc == ny - 1) {
            i_face = _faces_of_cell[i_cell][1];
            h_cells_of_face(i_face, 1) = -1;
            faces_t.push_back(i_face);
        }
        // Left boundary
        if (ic == 0) {
            i_face = _faces_of_cell[i_cell][2];
            h_cells_of_face(i_face, 1) = -1;
            faces_l.push_back(i_face);
        }
        // Bottom boundary
        if (jc == 0) {
            i_face = _faces_of_cell[i_cell][3];
            h_cells_of_face(i_face, 1) = -1;
            faces_b.push_back(i_face);
        }
    }

    zone_r.faces = Kokkos::View<u_int32_t *>("zone_r_faces", faces_r.size());
    zone_t.faces = Kokkos::View<u_int32_t *>("zone_t_faces", faces_t.size());
    zone_l.faces = Kokkos::View<u_int32_t *>("zone_l_faces", faces_l.size());
    zone_b.faces = Kokkos::View<u_int32_t *>("zone_b_faces", faces_b.size());

    zone_r.h_faces = Kokkos::create_mirror_view(zone_r.faces);
    zone_t.h_faces = Kokkos::create_mirror_view(zone_t.faces);
    zone_l.h_faces = Kokkos::create_mirror_view(zone_l.faces);
    zone_b.h_faces = Kokkos::create_mirror_view(zone_b.faces);

    for (u_int32_t i = 0; i < faces_r.size(); ++i) {
        zone_r.h_faces(i) = faces_r[i];
    }
    for (u_int32_t i = 0; i < faces_t.size(); ++i) {
        zone_t.h_faces(i) = faces_t[i];
    }
    for (u_int32_t i = 0; i < faces_l.size(); ++i) {
        zone_l.h_faces(i) = faces_l[i];
    }
    for (u_int32_t i = 0; i < faces_b.size(); ++i) {
        zone_b.h_faces(i) = faces_b[i];
    }

    m_face_zones.push_back(zone_r);
    m_face_zones.push_back(zone_t);
    m_face_zones.push_back(zone_l);
    m_face_zones.push_back(zone_b);

    // Compute offsets and assign connectivity views
    u_int32_t nodes_of_cell_size = 0;
    u_int32_t faces_of_cell_size = 0;
    u_int32_t nodes_of_face_size = 0;
    for (u_int32_t i_cell = 0; i_cell < n_cells(); ++i_cell) {
        nodes_of_cell_size += _nodes_of_cell[i_cell].size();
        faces_of_cell_size += _faces_of_cell[i_cell].size();
    }
    for (u_int32_t i_face = 0; i_face < n_faces(); ++i_face) {
        nodes_of_face_size += _nodes_of_face[i_face].size();
    }

    nodes_of_cell = Kokkos::View<u_int32_t *>("nodes_of_cell", nodes_of_cell_size);
    offsets_nodes_of_cell = Kokkos::View<u_int32_t *>("offsets_nodes_of_cell", n_cells() + 1);
    faces_of_cell = Kokkos::View<u_int32_t *>("faces_of_cell", faces_of_cell_size);
    offsets_faces_of_cell = Kokkos::View<u_int32_t *>("offsets_faces_of_cell", n_cells() + 1);
    nodes_of_face = Kokkos::View<u_int32_t *>("nodes_of_face", nodes_of_face_size);
    offsets_nodes_of_face = Kokkos::View<u_int32_t *>("offsets_nodes_of_face", n_faces() + 1);

    h_nodes_of_cell = Kokkos::create_mirror_view(nodes_of_cell);
    h_offsets_nodes_of_cell = Kokkos::create_mirror_view(offsets_nodes_of_cell);
    h_faces_of_cell = Kokkos::create_mirror_view(faces_of_cell);
    h_offsets_faces_of_cell = Kokkos::create_mirror_view(offsets_faces_of_cell);
    h_nodes_of_face = Kokkos::create_mirror_view(nodes_of_face);
    h_offsets_nodes_of_face = Kokkos::create_mirror_view(offsets_nodes_of_face);

    u_int32_t _n_nodes_of_cell;
    u_int32_t _n_faces_of_cell;
    u_int32_t _n_nodes_of_face;
    h_offsets_nodes_of_cell(0) = 0;
    h_offsets_faces_of_cell(0) = 0;
    h_offsets_nodes_of_face(0) = 0;
    for (u_int32_t i_cell = 0; i_cell < n_cells(); ++i_cell) {
        _n_nodes_of_cell = _nodes_of_cell[i_cell].size();
        _n_faces_of_cell = _faces_of_cell[i_cell].size();
        h_offsets_nodes_of_cell(i_cell + 1) = h_offsets_nodes_of_cell(i_cell) + _n_nodes_of_cell;
        h_offsets_faces_of_cell(i_cell + 1) = h_offsets_faces_of_cell(i_cell) + _n_faces_of_cell;
        for (u_int32_t i_node_local = 0; i_node_local < _n_nodes_of_cell; ++i_node_local) {
            h_nodes_of_cell(h_offsets_nodes_of_cell(i_cell) + i_node_local) = _nodes_of_cell[i_cell][i_node_local];
        }
        for (u_int32_t i_face_local = 0; i_face_local < _n_faces_of_cell; ++i_face_local) {
            h_faces_of_cell(h_offsets_faces_of_cell(i_cell) + i_face_local) = _faces_of_cell[i_cell][i_face_local];
        }
    }
    for (u_int32_t i_face = 0; i_face < n_faces(); ++i_face) {
        _n_nodes_of_face = _nodes_of_face[i_face].size();
        h_offsets_nodes_of_face(i_face + 1) = h_offsets_nodes_of_face(i_face) + _n_nodes_of_face;
        for (u_int32_t i_node_local = 0; i_node_local < _n_nodes_of_face; ++i_node_local) {
            h_nodes_of_face(h_offsets_nodes_of_face(i_face) + i_node_local) = _nodes_of_face[i_face][i_node_local];
        }
    }

    // Compute derived quantities
    compute_face_areas();
    compute_cell_volumes();
    compute_cell_centroids();
    compute_face_centroids();
    compute_face_normals();
}

void Mesh::init_wedge(u_int32_t nx, u_int32_t ny, rtype Lx, rtype Ly) {
    init_cart(nx, ny, Lx, Ly);

    rtype wedge_theta = 8 * Kokkos::numbers::pi / 180.0;
    rtype wedge_x = 0.5;

    // Adjust node coordinates
    rtype dx = Lx / nx;
    rtype dy = Ly / ny;

    for (u_int32_t i = 0; i < nx + 1; ++i) {
        for (u_int32_t j = 0; j < ny + 1; ++j) {
            u_int32_t i_node = i * (ny + 1) + j;
            rtype x = i * dx;
            rtype y;
            if (x > wedge_x) {
                rtype y_bottom = (x - wedge_x) * tan(wedge_theta);
                y = j * (Ly - y_bottom) / ny + y_bottom;
            } else {
                y = j * dy;
            }
            h_node_coords(i_node, 0) = x;
            h_node_coords(i_node, 1) = y;
        }
    }

    // Recompute derived quantities
    compute_face_areas();
    compute_cell_volumes();
    compute_cell_centroids();
    compute_face_centroids();
    compute_face_normals();
}
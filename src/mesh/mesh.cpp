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

std::array<u_int32_t, 4> Mesh::nodes_of_cell(int32_t i_cell) const {
    return m_nodes_of_cell[i_cell];
}

std::array<u_int32_t, 4> Mesh::faces_of_cell(int32_t i_cell) const {
    return m_faces_of_cell[i_cell];
}

std::array<int32_t, 2> Mesh::cells_of_face(u_int32_t i_face) const {
    return m_cells_of_face[i_face];
}

std::array<u_int32_t, 2> Mesh::nodes_of_face(u_int32_t i_face) const {
    return m_nodes_of_face[i_face];
}

void Mesh::compute_cell_centroids() {
    for (u_int32_t i_cell = 0; i_cell < n_cells(); ++i_cell) {
        h_cell_coords(i_cell, 0) = 0.25 * (h_node_coords(m_nodes_of_cell[i_cell][0], 0) +
                                           h_node_coords(m_nodes_of_cell[i_cell][1], 0) +
                                           h_node_coords(m_nodes_of_cell[i_cell][2], 0) +
                                           h_node_coords(m_nodes_of_cell[i_cell][3], 0));
        h_cell_coords(i_cell, 1) = 0.25 * (h_node_coords(m_nodes_of_cell[i_cell][0], 1) +
                                           h_node_coords(m_nodes_of_cell[i_cell][1], 1) +
                                           h_node_coords(m_nodes_of_cell[i_cell][2], 1) +
                                           h_node_coords(m_nodes_of_cell[i_cell][3], 1));
    }
}

void Mesh::compute_face_centroids() {
    for (u_int32_t i_face = 0; i_face < n_faces(); ++i_face) {
        h_face_coords(i_face, 0) = 0.5 * (h_node_coords(m_nodes_of_face[i_face][0], 0) +
                                          h_node_coords(m_nodes_of_face[i_face][1], 0));
        h_face_coords(i_face, 1) = 0.5 * (h_node_coords(m_nodes_of_face[i_face][0], 1) +
                                          h_node_coords(m_nodes_of_face[i_face][1], 1));
    }
}

void Mesh::compute_cell_volumes() {
    for (u_int32_t i_cell = 0; i_cell < n_cells(); ++i_cell) {
        u_int32_t i_node_0 = m_nodes_of_cell[i_cell][0];
        u_int32_t i_node_1 = m_nodes_of_cell[i_cell][1];
        u_int32_t i_node_2 = m_nodes_of_cell[i_cell][2];
        u_int32_t i_node_3 = m_nodes_of_cell[i_cell][3];

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
        u_int32_t i_node_0 = m_nodes_of_face[i_face][0];
        u_int32_t i_node_1 = m_nodes_of_face[i_face][1];
        h_face_area(i_face) = sqrt(pow(h_node_coords(i_node_1, 0) -
                                       h_node_coords(i_node_0, 0), 2) +
                                   pow(h_node_coords(i_node_1, 1) -
                                       h_node_coords(i_node_0, 1), 2));
    }
}

void Mesh::compute_face_normals() {
    for (u_int32_t i_face = 0; i_face < n_faces(); ++i_face) {
        // Compute normal with area magnitude
        u_int32_t i_node_0 = m_nodes_of_face[i_face][0];
        u_int32_t i_node_1 = m_nodes_of_face[i_face][1];
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
        int32_t i_cell_0 = m_cells_of_face[i_face][0];
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
}

void Mesh::copy_device_to_host() {
    Kokkos::deep_copy(h_node_coords, node_coords);
    Kokkos::deep_copy(h_cell_coords, cell_coords);
    Kokkos::deep_copy(h_face_coords, face_coords);
    Kokkos::deep_copy(h_cell_volume, cell_volume);
    Kokkos::deep_copy(h_face_area, face_area);
    Kokkos::deep_copy(h_face_normals, face_normals);
}

void Mesh::init_cart(u_int32_t nx, u_int32_t ny, rtype Lx, rtype Ly) {
    /** \todo This is a hack for WENO, remove this */
    this->nx = nx;
    this->ny = ny;

    node_coords = Kokkos::View<rtype *[N_DIM]>("node_coords", n_nodes());
    cell_coords = Kokkos::View<rtype *[N_DIM]>("cell_coords", n_cells());
    face_coords = Kokkos::View<rtype *[N_DIM]>("face_coords", n_faces());
    cell_volume = Kokkos::View<rtype *>("cell_volume", n_cells());
    face_area = Kokkos::View<rtype *>("face_area", n_faces());
    face_normals = Kokkos::View<rtype *[N_DIM]>("face_normals", n_faces());

    h_node_coords = Kokkos::create_mirror_view(node_coords);
    h_cell_coords = Kokkos::create_mirror_view(cell_coords);
    h_face_coords = Kokkos::create_mirror_view(face_coords);
    h_cell_volume = Kokkos::create_mirror_view(cell_volume);
    h_face_area = Kokkos::create_mirror_view(face_area);
    h_face_normals = Kokkos::create_mirror_view(face_normals);

    m_nodes_of_cell.resize(n_cells());
    m_faces_of_cell.resize(n_cells());
    m_cells_of_face.resize(n_faces());
    m_nodes_of_face.resize(n_faces());

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
        m_nodes_of_cell[i_cell][0] = (ic + 1) * (ny + 1) + jc + 1; // Top right
        m_nodes_of_cell[i_cell][1] = (ic)     * (ny + 1) + jc + 1; // Top left
        m_nodes_of_cell[i_cell][2] = (ic)     * (ny + 1) + jc;     // Bottom left
        m_nodes_of_cell[i_cell][3] = (ic + 1) * (ny + 1) + jc;     // Bottom right

        m_faces_of_cell[i_cell][0] = (2 * ny + 1) * (ic + 1) + jc;          // Right
        m_faces_of_cell[i_cell][1] = (2 * ny + 1) * (ic)     + jc + ny + 1; // Top
        m_faces_of_cell[i_cell][2] = (2 * ny + 1) * (ic)     + jc;          // Left
        m_faces_of_cell[i_cell][3] = (2 * ny + 1) * (ic)     + jc + ny;     // Bottom

        // NOTE: m_cells_of_face and m_nodes_of_face will be overwritten
        // by future loop iterations, but this is okay because
        // the values will be the same for all cells.

        m_cells_of_face[m_faces_of_cell[i_cell][0]][0] = i_cell;      // Right face - center cell
        m_cells_of_face[m_faces_of_cell[i_cell][0]][1] = i_cell + ny; // Right face - right cell
        m_cells_of_face[m_faces_of_cell[i_cell][1]][0] = i_cell;      // Top face - center cell
        m_cells_of_face[m_faces_of_cell[i_cell][1]][1] = i_cell + 1;  // Top face - top cell
        m_cells_of_face[m_faces_of_cell[i_cell][2]][0] = i_cell;      // Left face - center cell
        m_cells_of_face[m_faces_of_cell[i_cell][2]][1] = i_cell - ny; // Left face - left cell
        m_cells_of_face[m_faces_of_cell[i_cell][3]][0] = i_cell;      // Bottom face - center cell
        m_cells_of_face[m_faces_of_cell[i_cell][3]][1] = i_cell - 1;  // Bottom face - bottom cell

        m_nodes_of_face[m_faces_of_cell[i_cell][0]][0] = m_nodes_of_cell[i_cell][3]; // Right face - bottom right node
        m_nodes_of_face[m_faces_of_cell[i_cell][0]][1] = m_nodes_of_cell[i_cell][0]; // Right face - top right node
        m_nodes_of_face[m_faces_of_cell[i_cell][1]][0] = m_nodes_of_cell[i_cell][0]; // Top face - top right node
        m_nodes_of_face[m_faces_of_cell[i_cell][1]][1] = m_nodes_of_cell[i_cell][1]; // Top face - top left node
        m_nodes_of_face[m_faces_of_cell[i_cell][2]][0] = m_nodes_of_cell[i_cell][1]; // Left face - top left node
        m_nodes_of_face[m_faces_of_cell[i_cell][2]][1] = m_nodes_of_cell[i_cell][2]; // Left face - bottom left node
        m_nodes_of_face[m_faces_of_cell[i_cell][3]][0] = m_nodes_of_cell[i_cell][2]; // Bottom face - bottom left node
        m_nodes_of_face[m_faces_of_cell[i_cell][3]][1] = m_nodes_of_cell[i_cell][3]; // Bottom face - bottom right node
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
    for (u_int32_t i_cell = 0; i_cell < n_cells(); i_cell++) {
        u_int32_t ic = i_cell / ny;
        u_int32_t jc = i_cell % ny;
        u_int32_t i_face;

        // Right boundary
        if (ic == nx - 1) {
            i_face = m_faces_of_cell[i_cell][0];
            m_cells_of_face[i_face][1] = -1;
            zone_r.faces()->push_back(i_face);
        }
        // Top boundary
        if (jc == ny - 1) {
            i_face = m_faces_of_cell[i_cell][1];
            m_cells_of_face[i_face][1] = -1;
            zone_t.faces()->push_back(i_face);
        }
        // Left boundary
        if (ic == 0) {
            i_face = m_faces_of_cell[i_cell][2];
            m_cells_of_face[i_face][1] = -1;
            zone_l.faces()->push_back(i_face);
        }
        // Bottom boundary
        if (jc == 0) {
            i_face = m_faces_of_cell[i_cell][3];
            m_cells_of_face[i_face][1] = -1;
            zone_b.faces()->push_back(i_face);
        }
    }
    m_face_zones.push_back(zone_r);
    m_face_zones.push_back(zone_t);
    m_face_zones.push_back(zone_l);
    m_face_zones.push_back(zone_b);

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
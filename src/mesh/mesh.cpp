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

#include <cmath>

#include "common/common.h"
#include "boundary/boundary.h"

Mesh::Mesh(MeshType type) {
    this->type = type;
}

Mesh::~Mesh() {
    // Empty
}

MeshType Mesh::get_type() const {
    return type;
}

void Mesh::set_type(MeshType type) {
    this->type = type;
}

int Mesh::n_cells() const {
    return nx * ny;
}

int Mesh::n_nodes() const {
    return (nx + 1) * (ny + 1);
}

int Mesh::n_faces() const {
    return 2 * n_cells() + nx + ny;
}

int Mesh::n_face_zones() const {
    return m_face_zones.size();
}

std::vector<FaceZone> * Mesh::face_zones() {
    return &m_face_zones;
}

FaceZone * Mesh::get_face_zone(const std::string& name) {
    for (int i = 0; i < n_face_zones(); ++i) {
        if (m_face_zones[i].get_name() == name) {
            return &(m_face_zones[i]);
        }
    }
    return nullptr;
}

std::array<double, 2> Mesh::cell_coords(int i_cell) const {
    return m_cell_coords[i_cell];
}

std::array<double, 2> Mesh::node_coords(int i_node) const {
    return m_node_coords[i_node];
}

double Mesh::cell_volume(int i_cell) const {
    return m_cell_volume[i_cell];
}

double Mesh::face_area(int i_face) const {
    return m_face_area[i_face];
}

std::array<double, 2> Mesh::face_normal(int i_face) const {
    return m_face_normals[i_face];
}

std::array<int, 4> Mesh::nodes_of_cell(int i_cell) const {
    return m_nodes_of_cell[i_cell];
}

std::array<int, 4> Mesh::faces_of_cell(int i_cell) const {
    return m_faces_of_cell[i_cell];
}

std::array<int, 2> Mesh::cells_of_face(int i_face) const {
    return m_cells_of_face[i_face];
}

std::array<int, 2> Mesh::nodes_of_face(int i_face) const {
    return m_nodes_of_face[i_face];
}

void Mesh::compute_cell_centroids() {
    for (int i_cell = 0; i_cell < n_cells(); ++i_cell) {
        m_cell_coords[i_cell][0] = 0.25 * (m_node_coords[m_nodes_of_cell[i_cell][0]][0] +
                                           m_node_coords[m_nodes_of_cell[i_cell][1]][0] +
                                           m_node_coords[m_nodes_of_cell[i_cell][2]][0] +
                                           m_node_coords[m_nodes_of_cell[i_cell][3]][0]);
        m_cell_coords[i_cell][1] = 0.25 * (m_node_coords[m_nodes_of_cell[i_cell][0]][1] +
                                           m_node_coords[m_nodes_of_cell[i_cell][1]][1] +
                                           m_node_coords[m_nodes_of_cell[i_cell][2]][1] +
                                           m_node_coords[m_nodes_of_cell[i_cell][3]][1]);
    }
}

void Mesh::compute_face_centroids() {
    for (int i_face = 0; i_face < n_faces(); ++i_face) {
        m_face_coords[i_face][0] = 0.5 * (m_node_coords[m_nodes_of_face[i_face][0]][0] +
                                          m_node_coords[m_nodes_of_face[i_face][1]][0]);
        m_face_coords[i_face][1] = 0.5 * (m_node_coords[m_nodes_of_face[i_face][0]][1] +
                                          m_node_coords[m_nodes_of_face[i_face][1]][1]);
    }
}

void Mesh::compute_cell_volumes() {
    for (int i = 0; i < n_cells(); ++i) {
        int i_node0 = m_nodes_of_cell[i][0];
        int i_node1 = m_nodes_of_cell[i][1];
        int i_node2 = m_nodes_of_cell[i][2];
        int i_node3 = m_nodes_of_cell[i][3];

        double a1 = triangle_area_2(m_node_coords[i_node0],
                                    m_node_coords[i_node1],
                                    m_node_coords[i_node2]);
        double a2 = triangle_area_2(m_node_coords[i_node0],
                                    m_node_coords[i_node2],
                                    m_node_coords[i_node3]);
        m_cell_volume[i] = a1 + a2;
    }
}

void Mesh::compute_face_areas() {
    for (int i = 0; i < n_faces(); ++i) {
        int i_node0 = m_nodes_of_face[i][0];
        int i_node1 = m_nodes_of_face[i][1];
        m_face_area[i] = sqrt(pow(m_node_coords[i_node1][0] -
                                  m_node_coords[i_node0][0], 2) +
                              pow(m_node_coords[i_node1][1] -
                                  m_node_coords[i_node0][1], 2));
    }
}

void Mesh::compute_face_normals() {
    for (int i_face = 0; i_face < n_faces(); ++i_face) {
        // Compute normal with area magnitude
        int i_node0 = m_nodes_of_face[i_face][0];
        int i_node1 = m_nodes_of_face[i_face][1];
        double x0 = m_node_coords[i_node0][0];
        double y0 = m_node_coords[i_node0][1];
        double x1 = m_node_coords[i_node1][0];
        double y1 = m_node_coords[i_node1][1];
        double dx = x1 - x0;
        double dy = y1 - y0;
        double mag = sqrt(dx * dx + dy * dy);
        m_face_normals[i_face][0] =  dy / mag * face_area(i_face);
        m_face_normals[i_face][1] = -dx / mag * face_area(i_face);

        // Flip normal if it points into its left cell
        int i_cell0 = m_cells_of_face[i_face][0];
        double x_cell = m_cell_coords[i_cell0][0];
        double y_cell = m_cell_coords[i_cell0][1];
        double x_face = m_face_coords[i_face][0];
        double y_face = m_face_coords[i_face][1];
        double dot = (x_cell - x_face) * m_face_normals[i_face][0] +
                     (y_cell - y_face) * m_face_normals[i_face][1];
        if (dot < 0.0) {
            m_face_normals[i_face][0] *= -1.0;
            m_face_normals[i_face][1] *= -1.0;
        }
    }
}

void Mesh::init_wedge(int nx, int ny, double Lx, double Ly) {
    double wedge_theta = 8 * M_PI / 180.0;
    double wedge_x = 0.5;

    this->nx = nx;
    this->ny = ny;

    m_node_coords.resize(n_nodes());
    m_cell_coords.resize(n_cells());
    m_face_coords.resize(n_faces());
    m_nodes_of_cell.resize(n_cells());
    m_faces_of_cell.resize(n_cells());
    m_cell_volume.resize(n_cells());
    m_cells_of_face.resize(n_faces());
    m_nodes_of_face.resize(n_faces());
    m_face_area.resize(n_faces());
    m_face_normals.resize(n_faces());

    // Compute node coordinates
    double dx = Lx / nx;
    double dy = Ly / ny;

    for (int i = 0; i < nx + 1; ++i) {
        for (int j = 0; j < ny + 1; ++j) {
            int i_node = i * (ny + 1) + j;
            double x = i * dx;
            double y;
            if (x / Lx > wedge_x) {
                double y_bottom = (x - wedge_x * Lx) * tan(wedge_theta);
                y = j * (Ly - y_bottom) / ny + y_bottom;
            } else {
                y = j * dy;
            }
            m_node_coords[i_node][0] = x;
            m_node_coords[i_node][1] = y;
        }
    }

    // Build associations between nodes, cells, and faces
    for (int i_cell = 0; i_cell < n_cells(); ++i_cell) {
        int ic = i_cell / ny;
        int jc = i_cell % ny;
        m_nodes_of_cell[i_cell][0] = (ic + 1) * (ny + 1) + jc + 1; // Top right
        m_nodes_of_cell[i_cell][1] = ic * (ny + 1) + jc + 1;       // Top left
        m_nodes_of_cell[i_cell][2] = ic * (ny + 1) + jc;           // Bottom left
        m_nodes_of_cell[i_cell][3] = (ic + 1) * (ny + 1) + jc;     // Bottom right

        m_faces_of_cell[i_cell][0] = (2 * ny + 1) * (ic + 1);    // Right
        m_faces_of_cell[i_cell][1] = (2 * ny + 1) * ic + ny + 1; // Top
        m_faces_of_cell[i_cell][2] = (2 * ny + 1) * ic;          // Left
        m_faces_of_cell[i_cell][3] = (2 * ny + 1) * ic + ny;     // Bottom

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
    for (int i_cell = 0; i_cell < n_cells(); i_cell++) {
        int ic = i_cell / ny;
        int jc = i_cell % ny;

        if (ic == nx - 1) {
            m_cells_of_face[m_faces_of_cell[i_cell][0]][1] = -1;
            zone_r.faces()->push_back(m_faces_of_cell[i_cell][0]);
        }

        if (jc == ny - 1) {
            m_cells_of_face[m_faces_of_cell[i_cell][1]][1] = -1;
            zone_t.faces()->push_back(m_faces_of_cell[i_cell][1]);
        }

        if (ic == 0) {
            m_cells_of_face[m_faces_of_cell[i_cell][2]][1] = -1;
            zone_l.faces()->push_back(m_faces_of_cell[i_cell][2]);
        }

        if (jc == 0) {
            m_cells_of_face[m_faces_of_cell[i_cell][3]][1] = -1;
            zone_b.faces()->push_back(m_faces_of_cell[i_cell][3]);
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
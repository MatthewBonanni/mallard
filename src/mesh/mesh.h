/**
 * @file mesh.h
 * @author Matthew Bonanni (mbonanni001@gmail.com)
 * @brief Mesh class declaration.
 * @version 0.1
 * @date 2023-12-17
 * 
 * @copyright Copyright (c) 2023 Matthew Bonanni
 * 
 */

#ifndef MESH_H
#define MESH_H

#include <vector>
#include <unordered_map>

#include <Kokkos_Core.hpp>
#include <toml.hpp>

#include "zone.h"

enum class MeshType {
    UNKNOWN = -1,
    FILE = 1,
    CARTESIAN = 2,
    WEDGE = 3
};

static const std::unordered_map<std::string, MeshType> MESH_TYPES = {
    {"unknown", MeshType::UNKNOWN},
    {"file", MeshType::FILE},
    {"cartesian", MeshType::CARTESIAN},
    {"wedge", MeshType::WEDGE}
};

static const std::unordered_map<MeshType, std::string> MESH_NAMES = {
    {MeshType::UNKNOWN, "unknown"},
    {MeshType::FILE, "file"},
    {MeshType::CARTESIAN, "cart"},
    {MeshType::WEDGE, "wedge"}
};


class Mesh {
    public:
        /**
         * @brief Construct a new Mesh object
         */
        Mesh(MeshType type = MeshType::UNKNOWN);

        /**
         * @brief Destroy the Mesh object
         */
        ~Mesh();

        /**
         * @brief Initialize the mesh.
         * @param input TOML input data.
         */
        void init(const toml::value & input);

        /**
         * @brief Get the type of mesh.
         */
        MeshType get_type() const;

        /**
         * @brief Set the type of mesh.
         */
        void set_type(MeshType type);

        /**
         * @brief Get the number of cells.
         * @return Number of cells.
         */
        u_int32_t n_cells() const;

        /**
         * @brief Get the number of cells in the x-direction.
         * @return Number of cells in the x-direction.
         * \todo This is a hack for WENO, remove this
         */
        u_int32_t n_cells_x() const;

        /**
         * @brief Get the number of cells in the y-direction.
         * @return Number of cells in the y-direction.
         * \todo This is a hack for WENO, remove this
         */
        u_int32_t n_cells_y() const;

        /**
         * @brief Get the number of nodes.
         * @return Number of nodes.
         */
        u_int32_t n_nodes() const;

        /**
         * @brief Get the number of faces.
         * @return Number of faces.
         */
        u_int32_t n_faces() const;

        /**
         * @brief Get the number of face zones.
         * @return Number of face zones.
         */
        u_int32_t n_face_zones() const;

        /**
         * @brief Get the face zones.
         * @return Pointer to the vector of face zones.
         */
        std::vector<FaceZone> * face_zones();

        /**
         * @brief Get face zone by name.
         * @return Pointer to the face zone.
         */
        FaceZone * get_face_zone(const std::string& name);

        /**
         * @brief Get the nodes comprising a cell.
         * 
         * @param i_cell Index of the cell.
         * @return Array of node ids comprising the cell.
         */
        std::array<u_int32_t, 4> nodes_of_cell(int32_t i_cell) const;

        /**
         * @brief Get the faces comprising a cell.
         * 
         * @param i_cell Index of the cell.
         * @return Array of face ids comprising the cell.
         */
        std::array<u_int32_t, 4> faces_of_cell(int32_t i_cell) const;

        /**
         * @brief Get the cells bounding a face.
         * 
         * @param i_face Index of the face.
         * @return Array of cell ids bounding the face.
         */
        std::array<int32_t, 2> cells_of_face(u_int32_t i_face) const;

        /**
         * @brief Get the nodes comprising a face.
         * 
         * @param i_face Index of the face.
         * @return Array of node ids comprising the face.
         */
        std::array<u_int32_t, 2> nodes_of_face(u_int32_t i_face) const;

        /**
         * @brief Compute cell centroids.
         */
        void compute_cell_centroids();

        /**
         * @brief Compute face centroids.
         */
        void compute_face_centroids();

        /**
         * @brief Compute cell volumes.
         */
        void compute_cell_volumes();

        /**
         * @brief Compute face areas.
         */
        void compute_face_areas();

        /**
         * @brief Compute face normals.
         */
        void compute_face_normals();

        /**
         * @brief Copy mesh data from host to device.
         */
        void copy_host_to_device();

        /**
         * @brief Copy mesh data from device to host.
         */
        void copy_device_to_host();

        /**
         * @brief Initialize the mesh as a cartesian grid.
         * 
         * @param nx Number of cells in the x-direction.
         * @param ny Number of cells in the y-direction.
         * @param Lx Length of the domain in the x-direction.
         * @param Ly Length of the domain in the y-direction.
         */
        void init_cart(u_int32_t nx, u_int32_t ny, rtype Lx, rtype Ly);

        /**
         * @brief Initialize the supersonic wedge mesh.
         * 
         * @param nx Number of cells in the x-direction.
         * @param ny Number of cells in the y-direction.
         * @param Lx Length of the domain in the x-direction.
         * @param Ly Length of the domain in the y-direction.
         */
        void init_wedge(u_int32_t nx, u_int32_t ny, rtype Lx, rtype Ly);

        Kokkos::View<rtype *[N_DIM]> node_coords;
        Kokkos::View<rtype *[N_DIM]> cell_coords;
        Kokkos::View<rtype *[N_DIM]> face_coords;
        Kokkos::View<rtype *> cell_volume;
        Kokkos::View<rtype *> face_area;
        Kokkos::View<rtype *[N_DIM]> face_normals;

        Kokkos::View<rtype *[N_DIM]>::HostMirror h_node_coords;
        Kokkos::View<rtype *[N_DIM]>::HostMirror h_cell_coords;
        Kokkos::View<rtype *[N_DIM]>::HostMirror h_face_coords;
        Kokkos::View<rtype *>::HostMirror h_cell_volume;
        Kokkos::View<rtype *>::HostMirror h_face_area;
        Kokkos::View<rtype *[N_DIM]>::HostMirror h_face_normals;
    protected:
    private:
        u_int32_t nx, ny; /** \todo This is a hack for WENO, remove this */
        MeshType type;
        std::vector<std::array<u_int32_t, 4>> m_nodes_of_cell;
        std::vector<std::array<u_int32_t, 4>> m_faces_of_cell;
        std::vector<std::array<int32_t, 2>> m_cells_of_face;
        std::vector<std::array<u_int32_t, 2>> m_nodes_of_face;
        std::vector<CellZone> m_cell_zones;
        std::vector<FaceZone> m_face_zones;
};

#endif // MESH_H
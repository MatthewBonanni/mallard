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

#include <array>
#include <map>
#include <string>
#include <vector>
#include <unordered_map>

#include <Kokkos_Core.hpp>
#include <toml.hpp>

#include "zone.h"

enum class MeshType {
    FILE,
    CARTESIAN,
    CARTESIAN_TRI,
    WEDGE,
    CARTESIAN_TET,
    CARTESIAN_PRISM,
    CARTESIAN_PYRAMID,
    CARTESIAN_MIXED
};

static const std::unordered_map<std::string, MeshType> MESH_TYPES = {
    {"file", MeshType::FILE},
    {"cartesian", MeshType::CARTESIAN},
    {"cartesian_tri", MeshType::CARTESIAN_TRI},
    {"wedge", MeshType::WEDGE},
    {"cartesian_tet", MeshType::CARTESIAN_TET},
    {"cartesian_prism", MeshType::CARTESIAN_PRISM},
    {"cartesian_pyramid", MeshType::CARTESIAN_PYRAMID},
    {"cartesian_mixed", MeshType::CARTESIAN_MIXED}
};

static const std::unordered_map<MeshType, std::string> MESH_NAMES = {
    {MeshType::FILE, "file"},
    {MeshType::CARTESIAN, "cartesian"},
    {MeshType::CARTESIAN_TRI, "cartesian_tri"},
    {MeshType::WEDGE, "wedge"},
    {MeshType::CARTESIAN_TET, "cartesian_tet"},
    {MeshType::CARTESIAN_PRISM, "cartesian_prism"},
    {MeshType::CARTESIAN_PYRAMID, "cartesian_pyramid"},
    {MeshType::CARTESIAN_MIXED, "cartesian_mixed"}
};

enum class CellType {
    TRIANGLE,
    QUAD,
    TETRAHEDRON,
    PYRAMID,
    PRISM,
    HEXAHEDRON
};

static const std::unordered_map<std::string, CellType> CELL_TYPES = {
    {"triangle", CellType::TRIANGLE},
    {"quad", CellType::QUAD},
    {"tetrahedron", CellType::TETRAHEDRON},
    {"pyramid", CellType::PYRAMID},
    {"prism", CellType::PRISM},
    {"hexahedron", CellType::HEXAHEDRON}
};

static const std::unordered_map<CellType, std::string> CELL_NAMES = {
    {CellType::TRIANGLE, "triangle"},
    {CellType::QUAD, "quad"},
    {CellType::TETRAHEDRON, "tetrahedron"},
    {CellType::PYRAMID, "pyramid"},
    {CellType::PRISM, "prism"},
    {CellType::HEXAHEDRON, "hexahedron"}
};

/**
 * @brief Local faces of a positively oriented 3D cell with n_nodes nodes
 *        (4 tetrahedron, 5 pyramid, 6 prism, 8 hexahedron; Gmsh/VTK node
 *        order), each ordered so its right-hand normal points out of the cell.
 */
const std::vector<std::vector<uint8_t>> & cell_local_faces(uint32_t n_nodes);

/**
 * @brief Decompose a 3D cell (node coordinates in Gmsh/VTK order, positively
 *        oriented) into tetrahedra: each face is fanned around its vertex
 *        average and joined to the cell's vertex average. This is the
 *        decomposition that defines the mesh's cell volumes and centroids.
 */
void cell_tetrahedra(const std::vector<std::array<double, 3>> & nodes,
                     std::vector<std::array<std::array<double, 3>, 4>> & tets);


struct MeshBlock;

class Mesh {
    public:
        /**
         * @brief Construct a new Mesh object
         */
        Mesh();

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
         * @brief Get the number of face zones.
         * @return Number of face zones.
         */
        uint32_t n_face_zones() const;

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
         * @brief Get the number of nodes comprising a cell - host version.
         * @param i_cell Index of the cell.
         * @return Number of nodes comprising the cell.
         */
        uint32_t h_n_nodes_of_cell(uint32_t i_cell) const;

        /**
         * @brief Get the number of faces comprising a cell - host version.
         * @param i_cell Index of the cell.
         * @return Number of faces comprising the cell.
         */
        uint32_t h_n_faces_of_cell(uint32_t i_cell) const;

        /**
         * @brief Get the number of nodes comprising a face - host version.
         * @param i_face Index of the face.
         * @return Number of nodes comprising the face.
         */
        uint32_t h_n_nodes_of_face(uint32_t i_face) const;

        /**
         * @brief Get the id of the i-th node of a cell - host version.
         * @param i_cell Index of the cell.
         * @param i_node_local Index of the node.
         * @return Node id.
         */
        uint32_t h_node_of_cell(uint32_t i_cell, uint8_t i_node_local) const;

        /**
         * @brief Get the id of the i-th face of a cell - host version.
         * @param i_cell Index of the cell.
         * @param i_face_local Index of the face.
         * @return Face id.
         */
        uint32_t h_face_of_cell(uint32_t i_cell, uint8_t i_face_local) const;

        /**
         * @brief Get the id of the i-th node of a face - host version.
         * @param i_face Index of the face.
         * @param i_node_local Index of the node.
         * @return Node id.
         */
        uint32_t h_node_of_face(uint32_t i_face, uint8_t i_node_local) const;

        /**
         * @brief Get the type of a cell.
         * @param i_cell Index of the cell.
         */
        CellType h_cell_type(uint32_t i_cell) const;

        /**
         * @brief Get neighbors of a cell up to n_order graph distance.
         * @param i_cell Index of the cell.
         * @param n_order Maximum graph distance.
         * @param neighbors Vector to store the neighbors.
         */
        void h_neighbors_of_cell(uint32_t i_cell,
                                 uint8_t n_order,
                                 std::vector<uint32_t> & neighbors) const;

        /**
         * @brief Compute cell centroids.
         */
        void compute_cell_centroids();

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
         * @brief Compute the face centroids.
         */
        void compute_face_centroids();

        /**
         * @brief Compute all derived geometry (face areas and normals, cell
         *        volumes and centroids, face centroids) and the vertex-neighbor
         *        adjacency from the node coordinates and connectivity.
         */
        void compute_geometry();

        /**
         * @brief Tetrahedra of a 3D cell (see cell_tetrahedra).
         */
        void h_cell_tetrahedra(uint32_t i_cell, std::vector<std::array<std::array<double, 3>, 4>> & tets) const;

        /**
         * @brief Build the vertex-neighbor (cells sharing a node) adjacency on
         *        the device.
         */
        void compute_cell_neighbors();

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
        void init_cart(uint32_t nx, uint32_t ny, rtype Lx, rtype Ly);

        /**
         * @brief Initialize the mesh as a cartesian grid,
         * with each cell split into two triangles.
         * 
         * @param nx Number of cells in the x-direction.
         * @param ny Number of cells in the y-direction.
         * @param Lx Length of the domain in the x-direction.
         * @param Ly Length of the domain in the y-direction.
         */
        void init_cart_tri(uint32_t nx, uint32_t ny, rtype Lx, rtype Ly);

        /**
         * @brief Initialize a 3D box [0, Lx] x [0, Ly] x [0, Lz] of nx x ny x nz
         *        blocks, each a hexahedron or split into tetrahedra (6, Kuhn),
         *        prisms (2) or pyramids (6, apex at the block center). The
         *        mixed type uses hexahedra, pyramids and prisms in thirds of x.
         *        Boundary zones: left/right (x), bottom/top (y), back/front (z).
         */
        void init_cart_3d(uint32_t nx, uint32_t ny, uint32_t nz, rtype Lx, rtype Ly, rtype Lz, MeshType kind);

        /**
         * @brief A boundary face (2 nodes in 2D; 3 or 4 nodes in 3D) and its
         *        zone name.
         */
        struct BoundaryFace {
            std::vector<uint32_t> nodes;
            std::string zone;
        };

        /**
         * @brief Build the mesh from nodes, cells (any orientation) and named
         *        boundary faces. Boundary faces without a named face go to the
         *        zone unlisted_zone. Cells are triangles and quadrilaterals in 2D;
         *        tetrahedra, pyramids, prisms and hexahedra (Gmsh/VTK node
         *        order) in 3D.
         */
        void init_from_connectivity(const std::vector<std::array<rtype, N_DIM>> & nodes,
                                    const std::vector<std::vector<uint32_t>> & cells,
                                    const std::vector<BoundaryFace> & boundary_faces,
                                    const std::string & unlisted_zone = "unassigned");

        /**
         * @brief Build the mesh from a block holding the whole mesh.
         */
        void init_from_block(const MeshBlock & block);

        /**
         * @brief Read a Mallard HDF5 mesh file (.h5, .hdf5), or an ASCII Gmsh
         *        mesh (format 2.2 or 4.1) whose named physical curves (surfaces
         *        in 3D) become boundary zones.
         */
        void init_file(const std::string & filename);

        /**
         * @brief Initialize the supersonic wedge mesh.
         * 
         * @param nx Number of cells in the x-direction.
         * @param ny Number of cells in the y-direction.
         * @param Lx Length of the domain in the x-direction.
         * @param Ly Length of the domain in the y-direction.
         */
        void init_wedge(uint32_t nx, uint32_t ny, rtype Lx, rtype Ly);

        /**
         * @brief Global id of local cell i_cell (i_cell itself unless distributed).
         *        Connectivity is ordered by global ids, so that every rank count
         *        builds bitwise identical faces and stencils.
         */
        uint64_t h_global_cell(uint32_t i_cell) const {
            return h_global_cell_id.empty() ? i_cell : h_global_cell_id[i_cell];
        }

        uint32_t n_cells, n_nodes, n_faces;
        // Cells [0, n_owned()) are owned by this rank; the rest are halo cells
        uint32_t n_owned_cells = 0;
        uint32_t n_owned() const { return n_owned_cells ? n_owned_cells : n_cells; }
        // Cells [0, n_reconstructed()) need face values: owned cells and halo layer 1
        uint32_t n_reconstructed_cells = 0;
        uint32_t n_reconstructed() const { return n_reconstructed_cells ? n_reconstructed_cells : n_cells; }
        // Cells [0, n_complete()) have all their vertex neighbors: all but the outermost halo layer
        uint32_t n_complete_cells = 0;
        uint32_t n_complete() const { return n_complete_cells ? n_complete_cells : n_cells; }
        // Distributed runs: global id of every local cell and the global cell count (0 otherwise)
        std::vector<uint64_t> h_global_cell_id;
        uint64_t n_global_cells = 0;
        Kokkos::View<rtype *[N_DIM]> node_coords;
        Kokkos::View<rtype *[N_DIM]> cell_coords;
        Kokkos::View<rtype *> cell_volume;
        Kokkos::View<rtype *> face_area;
        Kokkos::View<rtype *[N_DIM]> face_normals;
        Kokkos::View<rtype *[N_DIM]> face_coords;
        Kokkos::View<uint32_t *> nodes_of_cell;
        Kokkos::View<uint32_t *> offsets_nodes_of_cell;
        Kokkos::View<uint32_t *> faces_of_cell;
        Kokkos::View<uint32_t *> offsets_faces_of_cell;
        Kokkos::View<uint32_t *> nodes_of_face;
        Kokkos::View<uint32_t *> offsets_nodes_of_face;
        Kokkos::View<int32_t *[2]> cells_of_face;
        Kokkos::View<uint32_t *> cells_of_cell;            // Vertex neighbors (CSR)
        Kokkos::View<uint32_t *> offsets_cells_of_cell;

        Kokkos::View<rtype *[N_DIM]>::host_mirror_type h_node_coords;
        Kokkos::View<rtype *[N_DIM]>::host_mirror_type h_cell_coords;
        Kokkos::View<rtype *>::host_mirror_type h_cell_volume;
        Kokkos::View<rtype *>::host_mirror_type h_face_area;
        Kokkos::View<rtype *[N_DIM]>::host_mirror_type h_face_normals;
        Kokkos::View<rtype *[N_DIM]>::host_mirror_type h_face_coords;
        Kokkos::View<uint32_t *>::host_mirror_type h_nodes_of_cell;
        Kokkos::View<uint32_t *>::host_mirror_type h_offsets_nodes_of_cell;
        Kokkos::View<uint32_t *>::host_mirror_type h_faces_of_cell;
        Kokkos::View<uint32_t *>::host_mirror_type h_offsets_faces_of_cell;
        Kokkos::View<uint32_t *>::host_mirror_type h_nodes_of_face;
        Kokkos::View<uint32_t *>::host_mirror_type h_offsets_nodes_of_face;
        Kokkos::View<int32_t *[2]>::host_mirror_type h_cells_of_face;
    protected:
    private:
        void init_box(uint32_t nx, uint32_t ny, rtype Lx, rtype Ly, bool triangles, bool wedge);
        // Local cells in increasing global id: the first cell to visit a face becomes its cell 0
        std::vector<uint32_t> cells_by_global_id() const;
        /**
         * @brief Positively oriented copy of 3D cells (Gmsh/VTK convention), as
         *        CSR offsets and node lists.
         */
        static void orient_cells_3d(const std::vector<std::array<rtype, N_DIM>> & nodes,
                                    const std::vector<std::vector<uint32_t>> & cells,
                                    std::vector<uint32_t> & offsets, std::vector<uint32_t> & cell_nodes);

        void allocate_and_fill(const std::vector<std::array<rtype, N_DIM>> & nodes,
                               const std::vector<uint32_t> & cell_node_offsets,
                               const std::vector<uint32_t> & cell_nodes,
                               const std::vector<uint32_t> & cell_face_offsets,
                               const std::vector<uint32_t> & cell_faces,
                               const std::vector<uint32_t> & face_node_offsets,
                               const std::vector<uint32_t> & face_nodes,
                               const std::vector<std::array<int32_t, 2>> & face_cells,
                               const std::vector<uint32_t> & interior,
                               const std::map<std::string, std::vector<uint32_t>> & zone_faces);

        void h_neighbors_of_cell_helper(uint32_t i_cell,
                                        uint8_t n_order,
                                        std::vector<uint32_t> & neighbors) const;

        MeshType type = MeshType::FILE;
        std::vector<CellZone> m_cell_zones;
        std::vector<FaceZone> m_face_zones;
};

#endif // MESH_H
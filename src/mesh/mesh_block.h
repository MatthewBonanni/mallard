/**
 * @file mesh_block.h
 * @author Matthew Bonanni (mbonanni001@gmail.com)
 * @brief A rank's contiguous block of a mesh, by global id, and the sources
 *        that produce one: HDF5 mesh files, Gmsh files and generated meshes.
 * @version 0.4
 * @date 2026-10-02
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 *
 */

#ifndef MESH_BLOCK_H
#define MESH_BLOCK_H

#include <array>
#include <cstdint>
#include <string>
#include <vector>

#include <toml.hpp>

#include "common_typedef.h"
#include "mesh.h"

/**
 * @brief Contiguous ranges of a mesh's cells and nodes and a set of its
 *        boundary faces. Global ids are positions in the whole mesh: cell
 *        first_cell + i is local cell i, node first_node + i local node i.
 *        Cells and boundary faces list global node ids; cell types follow from
 *        the node count (triangle 3, quadrilateral 4; tetrahedron 4, pyramid 5,
 *        prism 6, hexahedron 8; Gmsh/VTK node order). Concatenating the blocks
 *        of ranks 0, 1, ... gives the whole mesh.
 */
struct MeshBlock {
    uint64_t first_cell = 0;
    uint64_t first_node = 0;
    std::vector<uint64_t> cell_offsets{0};  // CSR into cell_nodes
    std::vector<uint64_t> cell_nodes;
    std::vector<std::array<double, N_DIM>> node_coords;
    std::vector<uint64_t> face_offsets{0};  // boundary faces, CSR into face_nodes
    std::vector<uint64_t> face_nodes;
    std::vector<uint32_t> face_zone;        // index into zone_names
    std::vector<std::string> zone_names;    // every zone, on every rank

    uint64_t n_cells() const { return cell_offsets.size() - 1; }
    uint64_t n_nodes() const { return node_coords.size(); }
    uint64_t n_faces() const { return face_offsets.size() - 1; }

    template <typename Range>
    void add_cell(const Range & nodes) {
        cell_nodes.insert(cell_nodes.end(), std::begin(nodes), std::end(nodes));
        cell_offsets.push_back(cell_nodes.size());
    }
    void add_cell(std::initializer_list<uint64_t> nodes) { add_cell<std::initializer_list<uint64_t>>(nodes); }

    template <typename Range>
    void add_face(const Range & nodes, uint32_t zone) {
        face_nodes.insert(face_nodes.end(), std::begin(nodes), std::end(nodes));
        face_offsets.push_back(face_nodes.size());
        face_zone.push_back(zone);
    }
};

/**
 * @brief Node i of n along a side of length L of a generated mesh, clustered
 *        toward both ends by the tanh stretching factor beta > 0:
 *        L / 2 (1 + tanh(beta (2 i / n - 1)) / tanh(beta)).
 */
rtype stretched_coordinate(uint32_t i, uint32_t n, rtype L, rtype beta);

/** @brief [mesh] stretching: the tanh factor of each direction, 0 (uniform) if not given. */
Stretching mesh_stretching(const toml::value & input);

/**
 * @brief Part r of p of a generated 2D mesh (cartesian, cartesian_tri or
 *        wedge): near-equal blocks of its cells and nodes, numbered as
 *        Mesh::init_cart, init_cart_tri and init_wedge number them, and the
 *        boundary faces of those cells.
 */
MeshBlock cartesian_2d_block(uint32_t nx, uint32_t ny, rtype Lx, rtype Ly, MeshType kind, int r, int p,
                             const Stretching & stretching = {});

/**
 * @brief A generated 3D box (see Mesh::init_cart_3d): the cells
 *        [first_cell, end_cell) and nodes [first_node, end_node) in the global
 *        numbering of the whole box, and the boundary faces of those cells.
 */
MeshBlock cartesian_3d_block(uint32_t nx, uint32_t ny, uint32_t nz, rtype Lx, rtype Ly, rtype Lz,
                             MeshType kind, uint64_t first_cell, uint64_t end_cell,
                             uint64_t first_node, uint64_t end_node, const Stretching & stretching = {});

/** @brief Total cell and node counts of a generated 3D box. */
std::array<uint64_t, 2> cartesian_3d_size(uint32_t nx, uint32_t ny, uint32_t nz, MeshType kind);

/** @brief Node (x, y) of the supersonic wedge mesh, moved from the box node (x, y). */
std::array<rtype, 2> wedge_node(rtype x, rtype y, rtype Ly);

/** @brief First element of block r of n elements split into p near-equal blocks. */
inline uint64_t block_begin(uint64_t n, int r, int p) { return n * uint64_t(r) / uint64_t(p); }

/**
 * @brief This rank's block of the mesh described by the [mesh] table
 *        (collective): an HDF5 mesh file (.h5), a Gmsh file, or a generated
 *        mesh. Generated meshes number cells and nodes exactly as Mesh::init.
 *        A Gmsh file is read whole by every rank, so large meshes should be
 *        converted to HDF5 first.
 */
MeshBlock read_mesh_block(const toml::value & input);

/** @brief This rank's block of a Gmsh file (every rank reads the whole file). */
MeshBlock read_gmsh_block(const std::string & filename);

/** @brief Whether a mesh file name denotes an HDF5 mesh file (.h5 or .hdf5). */
bool is_hdf5_mesh(const std::string & filename);

/** @brief Whether this build reads and writes HDF5 mesh files. */
bool have_hdf5();

/** @brief Whether several ranks can write one HDF5 mesh file together. */
bool have_parallel_hdf5();

/**
 * @brief Read this rank's block of an HDF5 mesh file (collective): cells,
 *        nodes and boundary faces split evenly in global order. With whole,
 *        this rank reads the whole file on its own (not collective).
 */
MeshBlock read_mesh_h5(const std::string & filename, bool whole = false);

/**
 * @brief Write the mesh whose blocks the ranks hold, in rank order, to an HDF5
 *        mesh file (collective; several ranks need parallel HDF5).
 */
void write_mesh_h5(const std::string & filename, const MeshBlock & block);

#endif // MESH_BLOCK_H

/**
 * @file mesh_file.h
 * @author Matthew Bonanni (mbonanni001@gmail.com)
 * @brief Readers of third-party mesh files (Gmsh, CGNS, Fluent, OpenFOAM)
 *        into a common whole-mesh representation.
 * @version 0.9
 * @date 2026-10-15
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 *
 */

#ifndef MESH_FILE_H
#define MESH_FILE_H

#include <array>
#include <cstdint>
#include <span>
#include <string>
#include <vector>

#include "curved.h"
#include "mesh.h"
#include "mesh_block.h"

/**
 * @brief A whole mesh read from a file: cells by their corners (Gmsh/VTK
 *        order, any orientation), named boundary faces, and for high-order
 *        boundary faces the coordinates of all their nodes (corners first,
 *        Gmsh order; empty for linear faces). Nodes that are no cell's or
 *        boundary face's corner are dropped.
 */
struct MeshFileData {
    /** @brief The cells' corners (CSR). */
    struct Cells {
        std::vector<uint64_t> offsets{0};
        std::vector<uint32_t> nodes;

        size_t size() const { return offsets.size() - 1; }
        bool empty() const { return size() == 0; }
        std::span<const uint32_t> operator[](size_t c) const { return {nodes.data() + offsets[c], offsets[c + 1] - offsets[c]}; }
        template <typename Range>
        void add(const Range & cell) {
            nodes.insert(nodes.end(), std::begin(cell), std::end(cell));
            offsets.push_back(nodes.size());
        }
    };

    std::vector<std::array<rtype, N_DIM>> nodes;
    Cells cells;
    std::vector<Mesh::BoundaryFace> boundary_faces;
    std::vector<std::vector<curved::Vec3>> boundary_high_order;
    // Cell zones: the zone of each cell (Mesh::NO_CELL_ZONE: none), empty if the file has none
    std::vector<uint32_t> cell_zone;
    std::vector<std::string> cell_zone_names;

    /** @brief Put cell c (cells numbered in order) in the zone of that name, created if new. */
    void set_cell_zone(size_t c, const std::string & name);
};

/**
 * @brief Read a mesh file by its extension: Gmsh or Fluent (.msh, told apart
 *        by their first character), Fluent case (.cas), CGNS (.cgns), or a
 *        directory: OpenFOAM.
 */
MeshFileData read_mesh_file(const std::string & filename);

/** @brief An ASCII Gmsh file, format 2.2 or 4.1. */
MeshFileData read_gmsh(const std::string & filename);

/**
 * @brief This rank's block of an ASCII Gmsh file (collective): each rank
 *        parses a near-equal share of the file's nodes and elements and sends
 *        them to the ranks whose blocks hold them, so that no rank holds the
 *        whole mesh. The blocks are those of read_gmsh's contents. Each rank
 *        indexes the lines of its share of the bytes in reads of chunk_bytes,
 *        keeping at least every index_spacing-th line; tests make both small.
 */
MeshBlock read_gmsh_block(const std::string & filename, uint64_t chunk_bytes = uint64_t(1) << 22,
                          uint64_t index_spacing = 1024);

/** @brief An unstructured CGNS file (needs a build with Mallard_ENABLE_CGNS). */
MeshFileData read_cgns(const std::string & filename);

/**
 * @brief An ANSYS Fluent mesh or case file (.msh, .cas), ASCII or binary:
 *        boundary zones are the face zones other than interior ones, named
 *        as the file's zone sections name them.
 */
MeshFileData read_fluent(const std::string & filename);

/**
 * @brief An OpenFOAM polyMesh, given as its directory, its parent, or its
 *        case (constant/polyMesh), ASCII or binary: each patch becomes a
 *        boundary zone. 2D builds read meshes one cell thick in z between
 *        empty patches, each cell as its face at the lowest z.
 */
MeshFileData read_openfoam(const std::string & path);

/**
 * @brief Corners (Gmsh/VTK order, either orientation) of the cells of a
 *        face-based mesh: faces by their nodes in cyclic order, and the one or
 *        two cells (0-based, -1 for none) of each face. Throws, naming source,
 *        for a cell that is not a triangle or quadrilateral (2D; faces of two
 *        nodes) or a tetrahedron, pyramid, prism or hexahedron (3D).
 */
MeshFileData::Cells cells_from_faces(uint32_t n_cells, const std::vector<std::vector<uint32_t>> & faces,
                                     const std::vector<std::array<int64_t, 2>> & face_cells, const std::string & source);

/** @brief The cell dimension of a CGNS file's base. */
int cgns_dimension(const std::string & filename);

/** @brief Whether this build reads CGNS files. */
bool have_cgns();

/** @brief Drop the nodes no cell or boundary face has as a corner, renumbering the rest in order. */
void drop_unused_nodes(MeshFileData & data);

#endif // MESH_FILE_H

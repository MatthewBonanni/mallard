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
#include <string>
#include <vector>

#include "curved.h"
#include "mesh.h"

/**
 * @brief A whole mesh read from a file: cells by their corners (Gmsh/VTK
 *        order, any orientation), named boundary faces, and for high-order
 *        boundary faces the coordinates of all their nodes (corners first,
 *        Gmsh order; empty for linear faces). Nodes that are no cell's or
 *        boundary face's corner are dropped.
 */
struct MeshFileData {
    std::vector<std::array<rtype, N_DIM>> nodes;
    std::vector<std::vector<uint32_t>> cells;
    std::vector<Mesh::BoundaryFace> boundary_faces;
    std::vector<std::vector<curved::Vec3>> boundary_high_order;
};

/** @brief Read a Gmsh (.msh, ASCII 2.2 or 4.1), CGNS (.cgns) or other supported mesh file. */
MeshFileData read_mesh_file(const std::string & filename);

/** @brief An ASCII Gmsh file, format 2.2 or 4.1. */
MeshFileData read_gmsh(const std::string & filename);

/** @brief An unstructured CGNS file (needs a build with Mallard_ENABLE_CGNS). */
MeshFileData read_cgns(const std::string & filename);

/** @brief Whether this build reads CGNS files. */
bool have_cgns();

/** @brief Drop the nodes no cell or boundary face has as a corner, renumbering the rest in order. */
void drop_unused_nodes(MeshFileData & data);

#endif // MESH_FILE_H

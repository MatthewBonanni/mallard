/**
 * @file mesh_file.h
 * @author Matthew Bonanni (mbonanni001@gmail.com)
 * @brief Readers of third-party mesh files (Gmsh, CGNS, Fluent, OpenFOAM),
 *        each rank reading its own part of the file into its block.
 * @version 0.10
 * @date 2026-11-15
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 *
 */

#ifndef MESH_FILE_H
#define MESH_FILE_H

#include <cstdint>
#include <span>
#include <string>
#include <vector>

#include "mesh_block.h"

// Each reader returns this rank's block of the file's mesh (collective): the
// ranks read near-equal parts of the file and send each other the parts of
// their blocks, so that no rank holds the whole mesh. Cells, boundary faces
// and nodes are numbered in file order; nodes that are no cell's or boundary
// face's corner are dropped. Blocks do not depend on the number of ranks.

/**
 * @brief An ASCII Gmsh file, format 2.2 or 4.1. Each rank indexes the lines of
 *        its share of the bytes in reads of chunk_bytes, keeping at least every
 *        index_spacing-th line; tests make both small.
 */
MeshBlock read_gmsh_block(const std::string & filename, uint64_t chunk_bytes = uint64_t(1) << 22,
                          uint64_t index_spacing = 1024);

/** @brief An unstructured CGNS file (needs a build with Mallard_ENABLE_CGNS). */
MeshBlock read_cgns_block(const std::string & filename);

/**
 * @brief An ANSYS Fluent mesh or case file (.msh, .cas), ASCII or binary:
 *        boundary zones are the face zones other than interior ones, named
 *        as the file's zone sections name them. ASCII sections hold one node
 *        or face per line, as Fluent writes them.
 */
MeshBlock read_fluent_block(const std::string & filename);

/**
 * @brief An OpenFOAM polyMesh, given as its directory, its parent, or its
 *        case (constant/polyMesh), ASCII or binary: each patch becomes a
 *        boundary zone. 2D builds read meshes one cell thick in z between
 *        empty patches, each cell as its face at the lowest z. ASCII lists
 *        hold one item per line, as OpenFOAM writes them.
 */
MeshBlock read_openfoam_block(const std::string & path);

/**
 * @brief Corners (Gmsh/VTK order, either orientation) of cell c of a
 *        face-based mesh from its faces, each by its nodes in cyclic order.
 *        Throws, naming source, for a cell that is not a triangle or
 *        quadrilateral (2D; faces of two nodes) or a tetrahedron, pyramid,
 *        prism or hexahedron (3D).
 */
std::vector<uint64_t> cell_corners(const std::vector<std::span<const uint64_t>> & faces, uint64_t c,
                                   const std::string & source);

/** @brief The cell dimension of a CGNS file's base. */
int cgns_dimension(const std::string & filename);

/** @brief Whether this build reads CGNS files. */
bool have_cgns();

#endif // MESH_FILE_H

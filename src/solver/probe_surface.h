/**
 * @file probe_surface.h
 * @author Matthew Bonanni (mbonanni001@gmail.com)
 * @brief Probes on mesh faces: boundary zones.
 * @version 0.1
 * @date 2026-10-15
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 *
 */

#ifndef PROBE_SURFACE_H
#define PROBE_SURFACE_H

#include <cstdint>
#include <string>
#include <vector>

#include "mesh.h"
#include "probe_geometry.h"

namespace probe {

/**
 * @brief Faces sampled by a probe. Each rank holds its owned faces; rank 0
 *        holds the geometry in the global sample order and, for each global
 *        sample, its row among the faces gathered rank by rank.
 */
struct FaceSet {
    std::vector<uint32_t> faces;
    uint64_t n_global = 0;
    std::vector<uint64_t> gathered_row;  // rank 0 (every rank if not distributed)
    Geometry geometry;                   // rank 0 (every rank if not distributed)
};

/**
 * @brief The boundary faces of the given zones. 2D: chained into polylines
 *        through shared nodes, each traversed with the fluid on its left,
 *        samples at the face centroids with their arc length. 3D: polygons,
 *        ordered by their cells' global ids. Normals point out of the fluid;
 *        weights are the face measures (per radian in axisymmetric runs).
 */
FaceSet boundary_faces(const Mesh & mesh, const std::vector<std::string> & zones, bool distributed);

/** @brief Samples at points of known owned cells (this rank's), and rank 0's row of each global sample. */
struct CellPoints {
    std::vector<uint32_t> cells;
    std::vector<Point> points;
    uint64_t n_global = 0;
    std::vector<uint64_t> gathered_row;  // rank 0 (every rank if not distributed)
};

/**
 * @brief The exact intersection of a plane (3D) or line (2D) with the mesh:
 *        one sample per cut cell, at the centroid of its cut, weighing its
 *        area (length), and quadrature points over each cut (seven-point
 *        degree-5 rules on the triangles where the plane cuts the cell's
 *        tetrahedra; three Gauss points on 2D segments), ordered by the
 *        cells' global ids. A plane through mesh faces takes the faces once,
 *        from the cells on the side the normal points away from. With
 *        bounds = {lo, hi}, only the part inside that box.
 */
struct CutSet {
    CellPoints nodes, quadrature;
    Geometry geometry;  // rank 0: polygons (segments in 2D) with node and quadrature weights and normals
};

CutSet cut_plane(const Mesh & mesh, const Vec3 & point, const Vec3 & normal, bool distributed,
                 const std::array<Vec3, 2> * bounds = nullptr);

} // namespace probe

#endif // PROBE_SURFACE_H

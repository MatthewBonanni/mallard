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

} // namespace probe

#endif // PROBE_SURFACE_H

/**
 * @file region_mesh.h
 * @author Matthew Bonanni (mbonanni001@gmail.com)
 * @brief Surfaces, curves and points of regions on a mesh: their elements in
 *        the owned cells, oriented, exact where the region is a primitive's
 *        or the mesh's own, else extracted from the cells.
 * @version 0.1
 * @date 2026-10-30
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 *
 */

#ifndef REGION_MESH_H
#define REGION_MESH_H

#include <array>
#include <cstdint>
#include <vector>

#include "region.h"

class Mesh;

namespace region {

/**
 * @brief A surface, curve or set of points of a region in this rank's owned
 *        cells: polygons (3D surfaces), segments (curves) or points, each in
 *        one cell.
 */
struct Elements {
    int dimension = 0;
    std::vector<uint32_t> cell;                    // owned cell of each element
    std::vector<std::array<double, 3>> center;     // its centroid (exact ones: the point of its parameters' middle)
    std::vector<std::array<double, 3>> direction;  // unit normal (surfaces, 2D curves) or tangent (3D curves); 0 for points
    std::vector<double> measure;                   // area, length, or 1 for a point
    std::vector<uint32_t> offsets{0};              // its vertices: polygons counterclockwise about the normal,
    std::vector<std::array<double, 3>> vertices;   // segments along the tangent (2D: normal to the right)
    bool exact = true;                             // primitives' own surfaces and curves, or mesh faces, throughout

    size_t size() const { return cell.size(); }
};

/**
 * @brief The elements of a region of lower dimension than the mesh.
 *        Primitives' surfaces and curves are sampled on their parameters with
 *        Gauss points (measures exact up to round-off where nothing clips
 *        them); boundary zones and the faces of the mesh's regions are their
 *        faces; the rest is cut from each cell's simplices, subdivided
 *        region.resolution() times, as the zero set of the linear interpolant.
 *        Clipping regions keep the parts inside them (outside subtracted
 *        ones): cut exactly along planes, at Gauss points on primitives' own
 *        surfaces. Collective.
 */
Elements elements(const Mesh & mesh, const Region & region);

} // namespace region

#endif // REGION_MESH_H

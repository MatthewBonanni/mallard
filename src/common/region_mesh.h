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
#include <functional>
#include <string>
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
    // Quadrature of each element: points with weights summing to its measure, and normals (tangents);
    // primitives' surfaces at Gauss points of their parameters, flat elements degree 5 (3 Gauss points per segment)
    std::vector<uint32_t> quad_offsets{0};
    std::vector<std::array<double, 3>> quad_points;
    std::vector<double> quad_weights;
    std::vector<std::array<double, 3>> quad_normals;

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
 *        ones): on primitives' own surfaces at the roots of their distance
 *        along the parameters (high order), elsewhere linearly between
 *        vertices (exact for planes). Collective.
 */
Elements elements(const Mesh & mesh, const Region & region);

/**
 * @brief A patch of a surface in world coordinates: x(u, v) and its
 *        derivatives over [u0, u1] x [v0, v1], with x_u x x_v facing as the
 *        surface does.
 */
struct SurfacePatch {
    std::function<void(double, double, std::array<double, 3> &, std::array<double, 3> &, std::array<double, 3> &)> at;
    double u0 = 0.0, u1 = 1.0, v0 = 0.0, v1 = 1.0;
    bool cap = false;  // an end of a cylinder or cone
};

/**
 * @brief The patches of a surface that is one primitive's (of a filled
 *        region, its surface) or a flat surface, in the parameters region_mesh
 *        samples them on (docs/guide/geometry.md lists them per type). An
 *        error at where otherwise.
 */
std::vector<SurfacePatch> surface_patches(const Region & region, const std::string & where);

/** @brief A surface resampled on a grid: nodes with unit normals, quadrilaterals, and 3 x 3 Gauss points per quadrilateral. */
struct Grid {
    std::vector<std::array<double, 3>> nodes, normals;
    std::vector<std::array<uint32_t, 4>> quads;
    std::vector<std::array<double, 3>> quad_points, quad_normals;
    std::vector<double> quad_weights;
};

/**
 * @brief Patches on nu x nv nodes each, evenly in their parameters (a
 *        parameter spanning a closed turn does not repeat its first nodes),
 *        with Gauss points of degree 5 on each quadrilateral of parameters.
 */
Grid sample_grid(const std::vector<SurfacePatch> & patches, uint32_t nu, uint32_t nv);

} // namespace region

#endif // REGION_MESH_H

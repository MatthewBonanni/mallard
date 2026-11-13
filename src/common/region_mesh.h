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
 *        along the parameters (high order), elsewhere at the roots of
 *        their distance along the elements' edges. Collective.
 */
Elements elements(const Mesh & mesh, const Region & region);

/**
 * @brief The levels of a region's conditions (SolutionLevels::levels()) on
 *        the host in some owned cells, ascending: row i is cells[i]'s.
 */
struct Levels {
    std::vector<uint32_t> cells;
    Kokkos::View<const double **[N_DIM + 1], Kokkos::LayoutRight, Kokkos::HostSpace> values;
};

/**
 * @brief The elements of a region defined by the solution, with its
 *        conditions' levels: where a condition bounds or clips it, each cell's
 *        linear level is cut on the cell's simplices (exact for it, without
 *        subdividing). Pieces cut from conditions' levels are cut in the cells
 *        of levels; any other piece reads the levels of the cells it is in,
 *        which levels must hold (level_cells). Collective.
 */
Elements elements(const Mesh & mesh, const Region & region, const Levels & levels);

/**
 * @brief The owned cells elements(mesh, region, levels) reads levels in. For
 *        each piece cut from conditions' levels, its carriers' conditions: it
 *        is cut in the cells where none of their linear levels keeps its sign
 *        within radii of the cell's centroid (a bound of its distance to the
 *        cell's points). every: another piece reads levels in any owned cell.
 */
struct LevelCells {
    std::vector<std::vector<uint32_t>> conditions;
    std::vector<double> radii;
    bool every = false;
};
LevelCells level_cells(const Mesh & mesh, const Region & region);

/**
 * @brief The measures of a filled region's intersection with the domain:
 *        the sum of f V over the cells, with fractions f (cell_fractions),
 *        and the measure of its surface_in_domain(); per radian in
 *        axisymmetric runs. Measures::tolerance is 0: the mesh's own
 *        accuracy. Collective.
 */
Measures measures_in_domain(const Mesh & mesh, const Region & region, const std::vector<double> & fractions);

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

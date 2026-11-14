/**
 * @file probe_cells.h
 * @author Matthew Bonanni (mbonanni001@gmail.com)
 * @brief Probes on the cells they cover: curves cut by the cells they cross,
 *        filled regions integrated over the cells they cut, and grids of
 *        samples in filled regions.
 * @version 0.1
 * @date 2026-11-11
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 *
 */

#ifndef PROBE_CELLS_H
#define PROBE_CELLS_H

#include <array>
#include <cstdint>
#include <vector>

#include "mesh.h"
#include "probe_geometry.h"
#include "probe_surface.h"
#include "region.h"

namespace probe {

/**
 * @brief The curves of a geometry (Geometry::curves) cut by the owned cells:
 *        one sample per cell and crossing, at the middle of its length, with
 *        that length as its measure, and three Gauss points on each of its
 *        pieces (exact for polynomials of degree 5 along straight pieces), in
 *        the order along the curves. A curve on a face between two cells
 *        belongs to one of them (the mesh's boundary keeps it); curved cells
 *        are cut by their curved shapes. Rank 0 holds the geometry: the samples' segments (cell
 *        data), arc lengths, normals (2D) and quadrature. Collective.
 */
struct CurveCells {
    CellPoints nodes, quadrature;
    std::vector<Point> node_normals, quad_normals;  // 2D: of this rank's samples
    Geometry geometry;
    double length = 0.0;  // of the curves, in the mesh or not
};
CurveCells curve_cells(const Mesh & mesh, const Geometry & g, bool distributed);

/**
 * @brief Quadrature of a filled region over the owned cells, for point
 *        values: a cell entirely inside is one sample of its cell's value
 *        weighing its volume, whose integral is that of any reconstruction
 *        polynomial of mean equal to that value; a cell the region's surface
 *        cuts has points over its part inside (cut_quadrature), reduced by
 *        moment fitting to points that integrate polynomials of the given
 *        degree as the full quadrature does. Volumes are per radian in
 *        axisymmetric runs. Rank 0 holds the weights in the samples' global
 *        order (cells by global id). Static regions only. Collective.
 */
struct VolumeQuadrature {
    CellPoints points;
    std::vector<double> weights;
};
VolumeQuadrature volume_quadrature(const Mesh & mesh, const region::Region & region, uint32_t degree, bool distributed);

/**
 * @brief The nodes of an n[0] x n[1] (x n[2]) grid over the box around a
 *        filled region and the mesh, inside both, in grid order (x fastest),
 *        the same on every rank. Collective.
 */
std::vector<Point> volume_grid(const Mesh & mesh, const region::Region & region, const std::array<uint32_t, N_DIM> & n,
                               bool distributed);

} // namespace probe

#endif // PROBE_CELLS_H

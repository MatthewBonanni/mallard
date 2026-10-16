/**
 * @file overlap.h
 * @author Matthew Bonanni (mbonanni001@gmail.com)
 * @brief Exact intersection of mesh cells (2D polygons, 3D polyhedra) as sets
 *        of simplices, and quadrature on simplices of any degree.
 * @version 0.1
 * @date 2026-10-15
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 *
 */

#ifndef OVERLAP_H
#define OVERLAP_H

#include <array>
#include <cstdint>
#include <vector>

#include "common_typedef.h"

class Mesh;

namespace overlap {

using Point = std::array<double, N_DIM>;
/** @brief A triangle (2D) or tetrahedron (3D). */
using Simplex = std::array<Point, N_DIM + 1>;

/** @brief The half-space n . x <= d. */
struct HalfSpace {
    Point n;
    double d;
};

/** @brief Volume (area in 2D) of a simplex, of either orientation. */
double volume(const Simplex & s);

/**
 * @brief A cell as simplices that tile it (to integrate over, or to clip), and
 *        as convex pieces, each the intersection of half-spaces (to clip
 *        others with). A convex cell with planar faces is one piece bounded
 *        by its faces; any other cell is the mesh's own decomposition, a
 *        triangle fan around the vertex average in 2D, the tetrahedra of
 *        cell_tetrahedra in 3D, each a piece.
 */
struct Region {
    std::vector<Simplex> simplices;
    std::vector<std::vector<HalfSpace>> pieces;
    double volume = 0.0;
};

/**
 * @brief Region of a cell from its nodes: a polygon in either orientation in
 *        2D, a positively oriented cell in Gmsh/VTK node order in 3D.
 */
Region region(const std::vector<Point> & nodes);

/** @brief Nodes of cell c of a mesh, translated by shift if given. */
std::vector<Point> cell_nodes(const Mesh & mesh, uint32_t c, const double * shift = nullptr);

/** @brief Append the parts of simplex s inside h to out (at most N_DIM simplices). */
void clip(const Simplex & s, const HalfSpace & h, std::vector<Simplex> & out);

/**
 * @brief Append to out simplices tiling the intersection of the region the
 *        subject simplices tile with the region other. Exact up to rounding:
 *        the volumes of a subject's intersections with cells that tile a
 *        domain sum to the subject's volume.
 */
void intersect(const std::vector<Simplex> & subject, const Region & other, std::vector<Simplex> & out);

/**
 * @brief Quadrature on the reference simplex exact for polynomials of degree
 *        <= degree: conical products of Gauss-Legendre rules. Points are
 *        reference coordinates (xi_1..xi_N_DIM, x = x_0 + sum_i xi_i (x_i - x_0));
 *        weights sum to 1, so a simplex of volume V integrates f as
 *        V sum_q w_q f(x_q).
 */
struct Rule {
    std::vector<std::array<double, N_DIM>> points;
    std::vector<double> weights;
};
const Rule & simplex_rule(uint32_t degree);

/** @brief Physical point of reference coordinates xi in simplex s. */
Point map(const Simplex & s, const std::array<double, N_DIM> & xi);

} // namespace overlap

#endif // OVERLAP_H

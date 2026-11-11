/**
 * @file cut_quadrature.h
 * @author Matthew Bonanni (mbonanni001@gmail.com)
 * @brief High-order quadrature on the part of a cell where a level function
 *        is negative, and the moment fitting that reduces it to few points.
 * @version 0.1
 * @date 2026-11-11
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 *
 */

#ifndef CUT_QUADRATURE_H
#define CUT_QUADRATURE_H

#include <array>
#include <cstdint>
#include <functional>
#include <vector>

#include "common_typedef.h"
#include "overlap.h"

namespace cut_quadrature {

using Vec = std::array<double, 3>;
using Coordinates = std::array<double, N_DIM>;

/** @brief A map of the unit box [0, 1]^N_DIM into space: writes x(u), returns the volume per unit u. */
using Map = std::function<double(const Coordinates & u, Vec & x)>;

/** @brief A level function: negative inside. */
using Level = std::function<double(const Vec & x)>;

/** @brief A quadrature point: its position and weight. */
struct Point {
    Vec x;
    double w;
};

/**
 * @brief The multilinear map of the unit box whose corner b (bit k: u_k = 1)
 *        is corners[b]; it returns the signed determinant of its Jacobian.
 */
Map box_map(const std::array<Vec, 1u << N_DIM> & corners);

/**
 * @brief A simplex as the images of the unit box by N_DIM + 1 multilinear
 *        maps, one per vertex: onto the quadrilateral (hexahedron) of the
 *        vertex, the midpoints of its edges, the centroids of its faces and
 *        the simplex's centroid.
 */
std::vector<Map> simplex_maps(const overlap::Simplex & s);

/**
 * @brief Appends to out[i] the quadrature of {u in [0, 1]^N_DIM : level(x(u))
 *        < 0} with points[i] Gauss points per interval (Saye's algorithm, SIAM
 *        J. Sci. Comput. 37(2), 2015): the box is split until the zero set is a
 *        graph over a face of each piece along one coordinate, and so on for
 *        the zero sets on that face's edges; the intervals between the roots of
 *        those graphs, found to round-off, get Gauss points. Spectrally
 *        accurate where the zero set is smooth. distance: |level| bounds the
 *        distance to the zero set (a signed distance), which tells which
 *        pieces it cannot cross; otherwise a piece is taken not to cross it
 *        when level has one sign at its sample points.
 */
void integrate(const Map & map, const Level & level, bool distance, const std::vector<uint32_t> & points,
               std::vector<std::vector<Point>> & out);

/**
 * @brief Reduces points to as many as there are monomials of degree <=
 *        degree in (x - center) / scale, those of them whose monomials are
 *        the most independent, with the weights that integrate these
 *        monomials as reference does. False, with points unchanged, where no
 *        such weights match reference to round-off.
 */
bool fit(std::vector<Point> & points, const std::vector<Point> & reference, uint32_t degree, const Vec & center,
         double scale);

} // namespace cut_quadrature

#endif // CUT_QUADRATURE_H

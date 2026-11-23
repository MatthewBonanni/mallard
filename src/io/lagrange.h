/**
 * @file lagrange.h
 * @brief VTK Lagrange cells: their nodes in VTK's order, their basis, and the
 *        geometry of a mesh cell at those nodes (docs/design/high_order_output.md).
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 */

#ifndef LAGRANGE_H
#define LAGRANGE_H

#include <array>
#include <cstdint>
#include <span>
#include <vector>

class Mesh;
namespace curved {
class Geometry;
}

namespace lagrange {

/** @brief The shapes of VTK Lagrange cells that Mallard writes (pyramids as collapsed hexahedra). */
enum class Shape : uint8_t {
    TRIANGLE,
    QUADRILATERAL,
    TETRAHEDRON,
    HEXAHEDRON,
    WEDGE,
};

/** @brief A node's reference coordinates times the cell's order: integers in [0, order]. */
using Lattice = std::array<uint32_t, 3>;
using Point = std::array<double, 3>;

/** @brief VTK_LAGRANGE_TRIANGLE (69) to VTK_LAGRANGE_WEDGE (73). */
uint8_t vtk_type(Shape shape);

/** @brief Nodes of a cell of this shape and order. */
uint32_t n_nodes(Shape shape, uint32_t order);

/**
 * @brief The nodes of a cell of this shape and order (at least 1), in VTK's
 *        order: vertices, edge nodes edge by edge, face nodes face by face,
 *        then body nodes, as vtkLagrangeTriangle ... vtkLagrangeWedge number
 *        them; reference coordinates are lattice / order, on VTK's reference
 *        cells (simplices with a vertex at 0, unit squares and cubes).
 */
std::vector<Lattice> lattice(Shape shape, uint32_t order);

/** @brief lattice(shape, order) / order. */
std::vector<Point> reference_nodes(Shape shape, uint32_t order);

/**
 * @brief The Lagrange basis of the cell of this shape and order on its
 *        reference_nodes, at each point: row p holds the n_nodes(shape, order)
 *        values of the basis functions at points[p].
 */
std::vector<double> basis(Shape shape, uint32_t order, std::span<const Point> points);

/** @brief The shape of a mesh cell of n_vertices vertices: a pyramid is a hexahedron with its top face at the apex. */
Shape shape_of(uint32_t dim, uint32_t n_vertices);

/**
 * @brief The geometry of mesh cell c as a Lagrange cell of shape_of its
 *        vertices: its nodes, the vertices in VTK order (a pyramid's apex four
 *        times), and on curved cells (curves not null) but pyramids the
 *        nodes of a quadratic cell on the curved geometry.
 * @return The geometry's order: 1, or 2 for curved cells.
 */
uint32_t cell_geometry(const Mesh & mesh, const curved::Geometry * curves, uint32_t c, std::vector<Point> & nodes);

/**
 * @brief The nodes of curved cell c after its corners (mesh nodes in VTK
 *        order), as a quadratic VTK cell: the middles of the edges, of the
 *        quadrilateral faces and of the body (by transfinite interpolation of
 *        its boundary). Pyramids: those of VTK_QUADRATIC_PYRAMID.
 */
std::vector<Point> quadratic_nodes(const Mesh & mesh, const curved::Geometry & geometry, uint32_t c,
                                   const std::vector<uint32_t> & corners);

/** @brief The VTK type of a curved cell of n_corners corners written as a quadratic cell. */
uint8_t quadratic_vtk_type(uint32_t n_corners);

} // namespace lagrange

#endif // LAGRANGE_H

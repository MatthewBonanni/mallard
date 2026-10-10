/**
 * @file curved.h
 * @author Matthew Bonanni (mbonanni001@gmail.com)
 * @brief Curved (high-order) boundary geometry: face maps, face quadrature
 *        and cell integration rules (docs/design/curved_boundaries.md).
 * @version 0.1
 * @date 2026-10-09
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 *
 */

#ifndef CURVED_H
#define CURVED_H

#include <array>
#include <cstdint>
#include <map>
#include <string>
#include <utility>
#include <vector>

#include <toml.hpp>

#include "common_typedef.h"

class Mesh;
struct MeshBlock;

namespace curved {

using Vec3 = std::array<double, 3>;

/**
 * @brief An analytic surface onto which the faces of a boundary zone are
 *        projected: a circle (2D) or a sphere or circular cylinder (3D).
 */
struct Shape {
    enum class Kind : uint8_t { CIRCLE, SPHERE, CYLINDER };
    Kind kind = Kind::CIRCLE;
    std::string zone;
    Vec3 center = {0.0, 0.0, 0.0};
    Vec3 axis = {0.0, 0.0, 1.0};  // cylinder: unit vector
    double radius = 1.0;

    /** @brief Radial projection p of x onto the shape, and dp/dx (row-major 3 x 3). */
    void project(const Vec3 & x, Vec3 & p, double * J) const;
};

/**
 * @brief A curved boundary face of the whole mesh: its corners (global node
 *        ids, in the element's order) and either an analytic shape or the
 *        nodes of a high-order Gmsh element (corners first, Gmsh order).
 */
struct SurfaceFace {
    std::vector<uint64_t> corners;
    int32_t shape = -1;
    std::vector<Vec3> nodes;
};

/**
 * @brief The curved boundary of a mesh, identical on every rank: the analytic
 *        shapes and every curved boundary face, sorted by their sorted corners.
 */
struct Surfaces {
    std::vector<Shape> shapes;
    std::vector<SurfaceFace> faces;

    bool empty() const { return faces.empty(); }
};

/** @brief Analytic shapes of the [[mesh.curved]] entries (none if [mesh] curved = false). */
std::vector<Shape> read_shapes(const toml::value & input);

/** @brief [mesh] curved: whether curved geometry is used at all (default true). */
bool enabled(const toml::value & input);

/**
 * @brief Order of a Gmsh boundary element with n_corners corners and n_nodes
 *        nodes (1 linear, 2 or 3; triangle9 and quadrangle8 are the incomplete
 *        cubic and quadratic ones), or 0 if unsupported.
 */
uint32_t element_order(uint32_t n_corners, uint32_t n_nodes);

/**
 * @brief The curved surfaces of a whole mesh held by this rank: the high-order
 *        faces it was read with (Mesh::high_order_faces) and the faces of the
 *        zones of the analytic shapes.
 */
Surfaces surfaces_of_mesh(const Mesh & mesh, const toml::value & input);

/**
 * @brief The curved surfaces of a mesh spread over the ranks as blocks
 *        (collective): every rank gets the faces of every block.
 */
Surfaces surfaces_of_block(const MeshBlock & block, const toml::value & input);

/**
 * @brief Curved geometry of one rank's mesh. Faces are curved where they lie
 *        on a curved surface, and in 3D where they share an edge with one
 *        (blended from their edges). Reference coordinates of a face, in its
 *        node order: 2D s in [-1, 1] from node 0 to node 1; triangles (u, v)
 *        with x = v0 + u (v1 - v0) + v (v2 - v0) when flat; quadrilaterals
 *        (s, t) in [-1, 1]^2, nodes 0..3 at (-1, -1), (1, -1), (1, 1), (-1, 1).
 */
class Geometry {
    public:
        Geometry(const Mesh & mesh, const Surfaces & surfaces);

        bool face_is_curved(uint32_t f) const { return face_index[f] >= 0; }
        bool cell_is_curved(uint32_t c) const { return curved_cell[c] != 0; }
        uint32_t n_curved_faces() const { return uint32_t(faces.size()); }

        /**
         * @brief Largest distance of a node of a projected edge from its
         *        analytic shape, relative to the edge's length (0 if none).
         */
        double max_shape_offset() const;

        /**
         * @brief Point x of face f at reference coordinates xi, and its
         *        derivatives dx[k] = dx / dxi_k (k < N_DIM - 1), in the frame
         *        of the face's cell 0.
         */
        void eval(uint32_t f, const double * xi, Vec3 & x, Vec3 * dx) const;

        /** @brief The area vector int n dS of face f (out of its cell 0), exact for its edges. */
        Vec3 area_vector(uint32_t f) const;

        /**
         * @brief Quadrature of face f from a rule on its flat reference element
         *        (weights summing to the reference measure: 2 on edges and
         *        quadrilaterals, 1 on triangles, as the flat rules): points,
         *        unit normals out of cell 0, and area elements, whose area
         *        vectors sum exactly to area_vector(f).
         */
        void face_rule(uint32_t f, const std::vector<std::array<double, 2>> & ref, const std::vector<double> & ref_w,
                       std::vector<Vec3> & x, std::vector<Vec3> & n, std::vector<double> & w) const;

        /**
         * @brief Points and signed weights integrating polynomials of the given
         *        degree over cell c (curved or not), in the cell's frame: cones
         *        from its vertex average over its faces.
         */
        void cell_rule(uint32_t c, int degree, std::vector<Vec3> & x, std::vector<double> & w) const;

    private:
        struct Edge {
            int32_t shape = -1;
            std::vector<Vec3> nodes;  // Lagrange nodes, equispaced from the lower global node id to the higher
        };
        struct Face {
            int32_t shape = -1;                        // analytic surface face: its own projection
            std::array<int32_t, 4> edge = {-1, -1, -1, -1};  // local edge k (node k to k + 1), or -1 if straight
            std::array<uint8_t, 4> flip = {0, 0, 0, 0};      // edge k runs from its higher global node id
            std::vector<Vec3> bubble;                  // interior corrections of high-order elements
        };

        void edge_eval(const Face & face, size_t k, const Vec3 & a, const Vec3 & b, double t, Vec3 & x, Vec3 & dx) const;
        void face_eval(const Face & face, const std::vector<Vec3> & v, const double * xi, Vec3 & x, Vec3 * dx) const;
        std::vector<Vec3> face_nodes(uint32_t f) const;

        const Mesh & mesh;
        std::vector<Shape> shapes;
        std::vector<Edge> edges;
        std::vector<Face> faces;
        std::vector<int32_t> face_index;  // per local face: index into faces, or -1
        std::vector<uint8_t> curved_cell;
};

}  // namespace curved

#endif  // CURVED_H

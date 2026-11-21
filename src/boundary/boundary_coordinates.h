/**
 * @file boundary_coordinates.h
 * @author Matthew Bonanni (mbonanni001@gmail.com)
 * @brief Local coordinates of boundary faces for boundary expressions: u and
 *        v, which follow the shape of the boundary's patch, the distance to
 *        the nearest wall, and each one's twin in meters.
 * @version 0.1
 * @date 2026-11-15
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 *
 */

#ifndef BOUNDARY_COORDINATES_H
#define BOUNDARY_COORDINATES_H

#include <array>
#include <optional>
#include <string>
#include <vector>

#include "boundary_fields.h"
#include "box_tree.h"
#include "input.h"

class Mesh;

/**
 * @brief Which variables (BoundaryValue::variables(entry)) some expression
 *        of a [[boundaries]] entry reads, outside its [[boundaries.data]].
 */
std::vector<bool> variables_read(const Input & entry);

/**
 * @brief The local coordinates (boundary_variable indices) that some
 *        expression of a [[boundaries]] entry reads, in increasing order,
 *        outside its [[boundaries.data]].
 */
std::vector<uint16_t> local_coordinates_used(const Input & entry);

/** @brief The points of a [[boundaries]] entry that place u and v: those of origin, toward and corners it gives. */
struct PatchPoints {
    std::optional<Point> origin, toward;
    std::optional<std::vector<Point>> corners;
};

/** @brief Read and check the entry's origin, toward and corners; errors name `where`. */
PatchPoints read_patch_points(const Input & entry, const std::string & where);

/**
 * @brief Distance to the nearest of a set of wall faces, gathered from every
 *        rank: exact distance to segments (2D) or to triangles fanned from
 *        each face's lowest node (3D), so that it does not depend on the
 *        rank count or the order of the faces.
 */
class WallDistance {
    public:
        WallDistance() = default;
        /** @param faces This rank's wall faces (faces of halo cells are skipped: their owners add them). */
        WallDistance(const Mesh & mesh, const std::vector<uint32_t> & faces, bool distributed);

        bool empty() const { return pieces.empty(); }

        /** @brief Distance from x to the nearest wall face. */
        double operator()(const Point & x) const;

    private:
        using Vertex = std::array<double, N_DIM>;
        std::vector<std::array<Vertex, N_DIM>> pieces;  // segments (2D) or triangles (3D)
        BoxTree tree;
};

/**
 * @brief Local coordinates of this rank's faces of a [[boundaries]] entry.
 *        u, v (patch_coordinates.h; in 2D u is the arc length over the
 *        boundary's length) come from every face of the entry on every rank,
 *        gathered in the order of global cells, so each face's values are
 *        the same bits on any rank count or partition.
 * @param faces This rank's faces of the entry.
 * @param used The local coordinates to compute (local_coordinates_used); the others stay NaN.
 * @param walls Faces n measures from; may be empty if `used` has no n or n_m.
 * @param summary Set to the patch and the range of each coordinate computed, for the run log.
 * @return The faces' points: x, y, z and their local coordinates.
 */
std::vector<FacePoint> local_coordinates(const Mesh & mesh, const std::vector<uint32_t> & faces, const Input & entry,
                                         const std::string & where, const std::vector<uint16_t> & used,
                                         const WallDistance & walls, bool distributed, std::string & summary);

#endif // BOUNDARY_COORDINATES_H

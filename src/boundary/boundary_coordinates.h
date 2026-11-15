/**
 * @file boundary_coordinates.h
 * @author Matthew Bonanni (mbonanni001@gmail.com)
 * @brief Local coordinates of boundary faces for boundary expressions: the
 *        distance to the nearest wall, coordinates in the frame of the
 *        boundary's faces, and (2D) the arc length along them.
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
 * @brief The local coordinates (boundary_variable indices) that some
 *        expression of a [[boundaries]] entry reads, in increasing order.
 */
std::vector<uint16_t> local_coordinates_used(const Input & entry);

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

/** @brief The frame of a [[boundaries]] entry's faces: an origin, the inward normal and the tangent axes. */
struct LocalFrame {
    Point origin{}, normal{}, tangent{};
    Point span{};  // 3D: the second tangent axis
};

/** @brief The `frame` of a [[boundaries]] entry: each part given, or derived from the faces. */
struct FrameKey {
    std::optional<Point> origin, normal, tangent;

    /** @brief Read entry.frame (if present), recording its keys. */
    FrameKey(const Input & entry, const std::string & where);
};

/**
 * @brief Local coordinates of this rank's faces of a [[boundaries]] entry.
 *        The frame and the arc length come from every face of the entry on
 *        every rank, summed in the order of global cells, so each face's
 *        values are the same bits on any rank count or partition.
 * @param faces This rank's faces of the entry.
 * @param used The local coordinates to compute (local_coordinates_used); the others stay NaN.
 * @param walls Faces n measures from; may be empty if `used` has no n.
 * @param summary Set to the frame and the range of each coordinate computed, for the run log.
 * @return The faces' points: x, y, z and their local coordinates.
 */
std::vector<FacePoint> local_coordinates(const Mesh & mesh, const std::vector<uint32_t> & faces, const Input & entry,
                                         const std::string & where, const std::vector<uint16_t> & used,
                                         const WallDistance & walls, bool distributed, std::string & summary);

/** @brief The frame of the faces: frame from their gathered centers and normals (exposed for tests). */
LocalFrame local_frame(const Mesh & mesh, const std::vector<uint32_t> & faces, const FrameKey & key,
                       const std::string & where, bool distributed);

#endif // BOUNDARY_COORDINATES_H

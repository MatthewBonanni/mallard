/**
 * @file patch_coordinates.h
 * @author Matthew Bonanni (mbonanni001@gmail.com)
 * @brief Coordinates u, v of a boundary patch that follow its shape
 *        (docs/design/patch_coordinates.md): harmonic functions of the
 *        patch's own surface, and their twins in meters.
 * @version 0.1
 * @date 2026-11-17
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 *
 */

#ifndef PATCH_COORDINATES_H
#define PATCH_COORDINATES_H

#include <array>
#include <cstddef>
#include <optional>
#include <string>
#include <vector>

namespace patch {

using Vec3 = std::array<double, 3>;

/** @brief The faces of a patch: polygons that share nodes with identical coordinates. */
struct Surface {
    std::vector<std::vector<Vec3>> faces;  // each face's nodes, in order around it
    std::vector<Vec3> inward;              // each face's area vector, into the domain
};

/** @brief What a patch is, from the loops of its outline. */
enum class Shape {
    FOUR_SIDED,  // one loop with four corners
    RING,        // two loops: an annulus, the side of a tube or of a cone
    DISK         // one loop without corners
};

/** @brief A face's coordinates: u, v in [0, 1] and their twins in meters. */
struct Values {
    double u, v, u_m, v_m;
};

struct Coordinates {
    Shape shape;
    std::vector<Values> faces;  // per face of the surface, at its center
    std::string summary;        // the shape and what fixes its coordinates, for the run log
};

/** @brief Where the patch's corners are, where u starts and which way it grows: found from the patch, or given. */
struct Corners {
    std::optional<std::vector<Vec3>> points;  // none (a disk) or four, in the order (u, v) = (0, 0), (1, 0), (1, 1), (0, 1)
    // u (and v) is 0 at the corner (four-sided) or the face (ring, disk) nearest to `origin`, and u grows toward
    // the adjacent corner nearest to `toward`, or around the patch the way that reaches `toward` sooner
    std::optional<Vec3> origin, toward;
    std::string key;                          // of the [[boundaries]] entry, for errors
    size_t line = 0;
};

/**
 * @brief u, v of every face of a patch, the same bits for the same faces in
 *        the same order.
 *
 *        Four-sided: u and v solve Laplace's equation on the surface, u from
 *        0 on one side to 1 on the opposite one and linear in arc length
 *        along the other two, v likewise. Ring: v is harmonic from 0 on one
 *        rim to 1 on the other and u, in [0, 1), is its harmonic conjugate
 *        around the ring. Disk: v and 2 pi u are the radius and the angle of
 *        the conformal map of the patch to the unit disk that sends its
 *        center to 0, f = exp(-2 pi (G + i G*)) with G the surface's Green's
 *        function of the center.
 *
 *        v_m is the distance along the surface from v = 0 and u_m that from
 *        u = 0 (four-sided) or the arc length around the patch (ring, disk).
 *
 * @throws InputError if the faces are not one patch of those shapes.
 */
Coordinates coordinates(const Surface & surface, const Corners & corners);

} // namespace patch

#endif // PATCH_COORDINATES_H

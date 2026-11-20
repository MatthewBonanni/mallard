/**
 * @file region_surface.h
 * @author Matthew Bonanni (mbonanni001@gmail.com)
 * @brief Regions from CAD surfaces: STL and OBJ files read, checked to be
 *        closed, oriented, and laid out as Kind::TRIANGLES for the signed
 *        distance of geometry.h.
 * @version 0.1
 * @date 2026-11-19
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 *
 */

#ifndef REGION_SURFACE_H
#define REGION_SURFACE_H

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace region::surface {

using Vec = std::array<double, 3>;
using Triangle = std::array<Vec, 3>;
using Segment = std::array<Vec, 2>;

/** @brief Where a surface file is named in the input, for its errors: the key of the region and its line. */
struct Where {
    std::string key;
    std::size_t line = 0;
};

/** @brief A closed surface in the triangles:: layout of geometry.h (an even number of values), and what it holds. */
struct Packed {
    std::vector<double> data;
    uint32_t n_triangles = 0;
    uint32_t n_bodies = 0;    // closed surfaces that bound solids
    uint32_t n_cavities = 0;  // closed surfaces that bound holes inside them
};

/**
 * @brief A surface file read as triangles, in meters: format "stl" (binary or ASCII) or "obj" (convex faces of
 *        any number of corners), its coordinates times factor. Corners at equal coordinates are one vertex.
 *        An InputError at where if it cannot be read or holds no triangle.
 */
std::vector<Triangle> read(const std::string & file, const std::string & format, double factor, const Where & where);

/**
 * @brief The closed surface of the triangles, laid out for Kind::TRIANGLES: every edge shared by two
 *        triangles and every corner's triangles one fan, the triangles of no area dropped, each closed surface
 *        oriented to face out of the solid it bounds (into a cavity inside another), whatever the file's
 *        order of corners. An InputError at where (naming file) if the surface is open, not manifold, encloses
 *        no volume, or its closed surfaces cross each other.
 */
Packed pack(const std::vector<Triangle> & triangles, const std::string & file, const Where & where);

/** @brief The triangles of a laid out surface s, counterclockwise seen from outside. */
std::vector<Triangle> triangles(const double * s);

/** @brief The bounding box lo, hi of a laid out surface s. */
std::array<Vec, 2> box(const double * s);

/**
 * @brief The section of a laid out surface s by the plane z = z0: segments with the solid to their left seen
 *        from +z (closed curves counterclockwise around solids, clockwise around holes). A corner on the plane
 *        counts as above it, so every closed curve closes.
 */
std::vector<Segment> section(const double * s, double z0);

} // namespace region::surface

#endif // REGION_SURFACE_H

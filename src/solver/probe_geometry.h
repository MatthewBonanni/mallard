/**
 * @file probe_geometry.h
 * @author Matthew Bonanni (mbonanni001@gmail.com)
 * @brief Geometry of [[probes]]: point sets and curves, with transforms.
 * @version 0.1
 * @date 2026-10-15
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 *
 */

#ifndef PROBE_GEOMETRY_H
#define PROBE_GEOMETRY_H

#include <array>
#include <cstdint>
#include <string>
#include <vector>

#include <toml.hpp>

#include "common_typedef.h"

namespace probe {

using Vec3 = std::array<double, 3>;
using Point = std::array<double, N_DIM>;

/**
 * @brief Affine map x -> A x + t with A = R S, as the regions of #317 take
 *        it: scale (a number or one factor per axis), then rotate (2D: an
 *        angle in degrees; 3D: {axis, angle} or {axes}), both about `pivot`,
 *        then translate.
 */
struct Transform {
    std::array<Vec3, 3> A = {Vec3{1.0, 0.0, 0.0}, Vec3{0.0, 1.0, 0.0}, Vec3{0.0, 0.0, 1.0}};
    Vec3 t = {0.0, 0.0, 0.0};

    Vec3 point(const Vec3 & x) const;
    Vec3 vector(const Vec3 & v) const;

    /** @brief The transform of a [[probes]] entry (identity without the keys). */
    static Transform read(const toml::value & entry, const std::string & key);
};

/**
 * @brief Where a probe samples. nodes are the points written out; quadrature
 *        points with weights (length elements of curves, 1 for point sets)
 *        are those the reductions sum over.
 */
struct Geometry {
    uint8_t dim = 0;  // 0: point set, 1: curves, 2: surfaces
    std::string shape;
    std::vector<Point> nodes;
    std::vector<double> arc_length;                // curves: per node, from the start of its curve
    std::vector<std::vector<uint32_t>> polylines;  // curves: node indices in order, closed ones end at their start
    std::vector<Point> quad_points;
    std::vector<double> quad_weights;
    std::vector<Point> quad_normals;  // surfaces and 2D curves: unit normals of the quadrature points
    // Samples that are their own quadrature points (no quad_points): the
    // measure of each one's element (empty: unit weights); unit normals of
    // the samples of surfaces and 2D curves
    std::vector<double> node_weights;
    std::vector<Point> normals;
    // Surfaces: polygons over the vertices, each a sample (cell data), or
    // without vertices over the nodes (point data)
    std::vector<Point> vertices;
    std::vector<uint64_t> polygon_offsets;  // polygons + 1
    std::vector<uint32_t> polygon_vertices;
};

/** @brief The geometry of a [[probes]] entry (key names it in errors). */
Geometry make_geometry(const toml::value & entry, const std::string & key);

/** @brief Points of a text file: rows of N_DIM coordinates split by commas or blanks; '#' starts a comment. */
std::vector<Point> read_points(const std::string & file, const std::string & key);

} // namespace probe

#endif // PROBE_GEOMETRY_H

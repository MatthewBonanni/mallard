/**
 * @file region_mesh_test.cpp
 * @author Matthew Bonanni (mbonanni001@gmail.com)
 * @brief Surfaces, curves and points of regions on meshes: exact measures of
 *        primitives' surfaces, orientation, extraction, clipping.
 * @version 0.1
 * @date 2026-10-30
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 *
 */

#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <memory>
#include <string>
#include <vector>

#include "gmsh_fixtures.h"
#include "input.h"
#include "mesh.h"
#include "region.h"
#include "region_mesh.h"
#include "test_fixtures.h"

namespace {

using Vec = std::array<double, 3>;
constexpr double PI = 3.14159265358979323846;

/** @brief The unit square (jittered triangles and quadrilaterals) or cube (tetrahedra), zones left, right, ... */
std::shared_ptr<Mesh> unit_mesh() {
    auto mesh = std::make_shared<Mesh>();
#if N_DIM == 2
    mesh->init_file(write_temp_shared("mallard_region_mesh.msh", jittered_mixed_mesh(10)));
#else
    mesh->init_cart_3d(6, 6, 6, 1.0, 1.0, 1.0, MeshType::CARTESIAN_TET);
#endif
    return mesh;
}

struct Fixture {
    std::shared_ptr<Mesh> mesh = unit_mesh();
    region::MeshRegions automatic{*mesh};

    region::Region region(const std::string & table) const {
        return region::Region(parse_toml("r = " + table).at("r"), "r", nullptr, &automatic);
    }
    region::Elements elements(const std::string & table) const { return region::elements(*mesh, region(table)); }
};

double total(const region::Elements & e) {
    double sum = 0.0;
    for (const double m : e.measure) sum += m;
    return sum;
}

/** @brief The integrals of n and of (x - c) . n over the elements: 0 and N_DIM times the enclosed measure. */
std::pair<double, double> divergence(const region::Elements & e, const Vec & c) {
    Vec n = {0.0, 0.0, 0.0};
    double flux = 0.0;
    for (size_t k = 0; k < e.size(); k++) {
        for (size_t i = 0; i < 3; i++) {
            n[i] += e.direction[k][i] * e.measure[k];
            flux += (e.center[k][i] - c[i]) * e.direction[k][i] * e.measure[k];
        }
    }
    return {std::sqrt(n[0] * n[0] + n[1] * n[1] + n[2] * n[2]), flux};
}

/** @brief Perimeter of an ellipse, or area of an ellipsoid, by fine midpoint sums of the parameters. */
double ellipse_perimeter(const double a, const double b) {
    const int n = 200000;
    double s = 0.0;
    for (int i = 0; i < n; i++) {
        const double t = 2.0 * PI * (i + 0.5) / n;
        s += std::hypot(a * std::sin(t), b * std::cos(t));
    }
    return s * 2.0 * PI / n;
}

[[maybe_unused]] double ellipsoid_area(const double a, const double b, const double c) {
    const int n = 2000;
    double s = 0.0;
    for (int i = 0; i < n; i++) {
        const double u = PI * (i + 0.5) / n;
        for (int j = 0; j < 2 * n; j++) {
            const double v = PI * (j + 0.5) / n;
            const Vec xu = {a * std::cos(u) * std::cos(v), b * std::cos(u) * std::sin(v), -c * std::sin(u)};
            const Vec xv = {-a * std::sin(u) * std::sin(v), b * std::sin(u) * std::cos(v), 0.0};
            s += std::hypot(xu[1] * xv[2] - xu[2] * xv[1], xu[2] * xv[0] - xu[0] * xv[2], xu[0] * xv[1] - xu[1] * xv[0]);
        }
    }
    return s * (PI / n) * (PI / n);
}

} // namespace

TEST(RegionMeshTest, PrimitivesHaveExactMeasures) {
    const Fixture f;
    std::vector<std::pair<std::string, double>> cases;
    if constexpr (N_DIM == 2) {
        cases = {{"{type = \"circle\", center = [0.5, 0.5], radius = 0.3}", 2.0 * PI * 0.3},
                 {"{type = \"disk\", center = [0.45, 0.5], radius = 0.3, filled = false}", 2.0 * PI * 0.3},
                 {"{type = \"ellipse\", center = [0.5, 0.5], semi_axes = [0.3, 0.15], rotate = 20, pivot = [0.5, 0.5]}",
                  ellipse_perimeter(0.3, 0.15)},
                 {"{type = \"rectangle\", center = [0.5, 0.5], size = [0.4, 0.2], rotate = 30, pivot = [0.5, 0.5], "
                  "filled = false}",
                  1.2},
                 {"{type = \"polygon\", vertices = [[0.2, 0.2], [0.8, 0.3], [0.4, 0.7]], filled = false}",
                  std::hypot(0.6, 0.1) + std::hypot(0.4, 0.4) + std::hypot(0.2, 0.5)},
                 {"{type = \"annulus\", center = [0.5, 0.5], radius = 0.3, inner_radius = 0.15, filled = false}",
                  2.0 * PI * 0.45}};
    } else {
        const double slant = std::hypot(0.4, 0.2);
        cases = {
            {"{type = \"sphere\", center = [0.5, 0.5, 0.5], radius = 0.3, filled = false}", 4.0 * PI * 0.09},
            {"{type = \"box\", center = [0.5, 0.5, 0.5], size = [0.4, 0.3, 0.2], rotate = {axis = [1, 2, 3], angle = 25}, "
             "pivot = [0.5, 0.5, 0.5], filled = false}",
             2.0 * (0.12 + 0.06 + 0.08)},
            {"{type = \"cylinder\", center = [0.5, 0.5, 0.5], axis = [1, 1, 0], radius = 0.2, length = 0.5, filled = false}",
             2.0 * PI * 0.2 * 0.5 + 2.0 * PI * 0.04},
            {"{type = \"cylinder\", center = [0.5, 0.5, 0.5], axis = [0, 0, 1], radius = 0.25, inner_radius = 0.1, "
             "length = 0.4, filled = false}",
             2.0 * PI * 0.35 * 0.4 + 2.0 * PI * (0.0625 - 0.01)},
            {"{type = \"cone\", base = [0.5, 0.5, 0.3], axis = [0, 1, 1], length = 0.4, base_radius = 0.3, "
             "top_radius = 0.1, filled = false}",
             PI * 0.4 * slant + PI * (0.09 + 0.01)},
            {"{type = \"torus\", center = [0.5, 0.5, 0.5], axis = [1, 0, 0], major_radius = 0.25, minor_radius = 0.1, "
             "filled = false}",
             4.0 * PI * PI * 0.25 * 0.1},
            {"{type = \"ellipsoid\", center = [0.5, 0.5, 0.5], semi_axes = [0.3, 0.2, 0.15], filled = false}",
             ellipsoid_area(0.3, 0.2, 0.15)},
            {"{type = \"disk\", center = [0.5, 0.5, 0.5], normal = [1, 2, 2], radius = 0.3}", PI * 0.09},
            {"{type = \"annulus\", center = [0.5, 0.5, 0.5], normal = [0, 1, 0], radius = 0.3, inner_radius = 0.2}",
             PI * (0.09 - 0.04)},
            {"{type = \"rectangle\", center = [0.5, 0.5, 0.5], normal = [1, 1, 1], size = [0.4, 0.2]}", 0.08},
            {"{type = \"polygon\", vertices = [[0.2, 0.2, 0.56], [0.8, 0.3, 0.69], [0.6, 0.7, 0.69], [0.3, 0.6, 0.62]]}",
             -1.0},
            {"{type = \"circle\", center = [0.5, 0.5, 0.5], normal = [1, 2, 2], radius = 0.3}", 2.0 * PI * 0.3},
            {"{type = \"ellipse\", center = [0.5, 0.5, 0.5], normal = [0, 0, 1], reference = [1, 1, 0], "
             "semi_axes = [0.3, 0.15]}",
             ellipse_perimeter(0.3, 0.15)},
            {"{type = \"rectangle\", center = [0.5, 0.5, 0.5], normal = [1, 1, 1], size = [0.4, 0.2], filled = false}", 1.2},
            {"{type = \"boundary\", of = {type = \"annulus\", center = [0.5, 0.5, 0.5], normal = [0, 1, 0], "
             "radius = 0.3, inner_radius = 0.2}}",
             2.0 * PI * 0.5}};
        // The polygon lies in z = 0.5 + 0.2 x + 0.1 y: its area in the xy plane times the slope factor
        const double xy[4][2] = {{0.2, 0.2}, {0.8, 0.3}, {0.6, 0.7}, {0.3, 0.6}};
        double area = 0.0;
        for (int i = 0; i < 4; i++) area += 0.5 * (xy[i][0] * xy[(i + 1) % 4][1] - xy[(i + 1) % 4][0] * xy[i][1]);
        cases[10].second = area * std::sqrt(1.0 + 0.04 + 0.01);
    }
    for (const auto & [table, exact] : cases) {
        const region::Elements e = f.elements(table);
        EXPECT_TRUE(e.exact) << table;
        // The ellipsoid's reference is a midpoint sum, good to 1e-7
        const double tol = table.find("ellipsoid") != std::string::npos ? 1e-6 : 1e-9;
        EXPECT_NEAR(total(e), exact, tol * exact) << table;
    }
}

TEST(RegionMeshTest, ClosedSurfacesFaceOutAndFlipFacesIn) {
    // The integrals of n and of x . n over a closed surface: 0 and N_DIM times what it encloses
    const Fixture f;
    const Vec c = {0.5, 0.5, N_DIM == 3 ? 0.5 : 0.0};
    const std::string ball = N_DIM == 2 ? "{type = \"disk\", center = [0.5, 0.5], radius = 0.3"
                                        : "{type = \"sphere\", center = [0.5, 0.5, 0.5], radius = 0.3";
    const double enclosed = N_DIM == 2 ? PI * 0.09 : 4.0 / 3.0 * PI * 0.027;
    for (const bool flip : {false, true}) {
        const region::Elements e = f.elements(ball + ", filled = false" + (flip ? ", flip = true}" : "}"));
        const auto [n, flux] = divergence(e, c);
        EXPECT_LT(n, 1e-3 * total(e));
        EXPECT_NEAR(flux, (flip ? -1.0 : 1.0) * N_DIM * enclosed, 2e-3 * N_DIM * enclosed);
        for (size_t k = 0; k < e.size(); k++) {
            double out = 0.0;
            for (size_t i = 0; i < 3; i++) out += (e.center[k][i] - c[i]) * e.direction[k][i];
            EXPECT_EQ(out > 0.0, !flip) << k;
        }
    }
    // Extracted from the cells: an implicit ball, and a box minus the ball (whose ball part faces in)
    const std::string implicit = std::string("{type = \"implicit\", f = \"sqrt((x - 0.5)^2 + (y - 0.5)^2") +
                                 (N_DIM == 3 ? " + (z - 0.5)^2" : "") + ") - 0.3\", filled = false}";
    const region::Elements e = f.elements(implicit);
    EXPECT_FALSE(e.exact);
    const double area = N_DIM == 2 ? 2.0 * PI * 0.3 : 4.0 * PI * 0.09;
    EXPECT_NEAR(total(e), area, 1e-2 * area);
    EXPECT_NEAR(divergence(e, c).second, N_DIM * enclosed, 2e-2 * N_DIM * enclosed);
    const std::string box = N_DIM == 2 ? "{type = \"rectangle\", center = [0.5, 0.5], size = [0.9, 0.9]}"
                                       : "{type = \"box\", center = [0.5, 0.5, 0.5], size = [0.9, 0.9, 0.9]}";
    const double box_measure = N_DIM == 2 ? 0.81 : 0.729;
    const region::Elements hollow = f.elements("{type = \"difference\", regions = [" + box + ", " + ball + "}], filled = false}");
    EXPECT_FALSE(hollow.exact);
    EXPECT_NEAR(divergence(hollow, c).second, N_DIM * (box_measure - enclosed), 2e-2 * N_DIM * box_measure);
}

TEST(RegionMeshTest, OpenSurfacesAndCurvesFollowTheirRules) {
    const Fixture f;
    if constexpr (N_DIM == 3) {
        // A disk faces along its normal; its edge runs counterclockwise about it, and a flipped one clockwise
        const Vec n = {1.0 / 3.0, 2.0 / 3.0, 2.0 / 3.0};
        const std::string disk = "{type = \"disk\", center = [0.5, 0.5, 0.5], normal = [1, 2, 2], radius = 0.3";
        for (const Vec & d : f.elements(disk + "}").direction) {
            for (size_t i = 0; i < 3; i++) EXPECT_NEAR(d[i], n[i], 1e-12);
        }
        for (const bool flip : {false, true}) {
            const region::Elements edge = f.elements(disk + ", filled = false" + (flip ? ", flip = true}" : "}"));
            ASSERT_GT(edge.size(), 0u);
            for (size_t k = 0; k < edge.size(); k++) {
                const Vec r = {edge.center[k][0] - 0.5, edge.center[k][1] - 0.5, edge.center[k][2] - 0.5};
                const Vec & t = edge.direction[k];
                const double turn = (r[1] * t[2] - r[2] * t[1]) * n[0] + (r[2] * t[0] - r[0] * t[2]) * n[1] +
                                    (r[0] * t[1] - r[1] * t[0]) * n[2];
                EXPECT_EQ(turn > 0.0, !flip) << k;
            }
        }
    } else {
        // 2D curves: the normal to the right of travel, outward for a circle
        const region::Elements circle = f.elements("{type = \"circle\", center = [0.5, 0.5], radius = 0.3}");
        for (size_t k = 0; k < circle.size(); k++) {
            const auto & v = circle.vertices;
            const Vec t = {v[circle.offsets[k] + 1][0] - v[circle.offsets[k]][0],
                           v[circle.offsets[k] + 1][1] - v[circle.offsets[k]][1], 0.0};
            EXPECT_GT(t[1] * circle.direction[k][0] - t[0] * circle.direction[k][1], 0.0) << k;
        }
    }
    // A plane through the domain is exact: the affine level is cut exactly
    const region::Elements plane =
        f.elements(std::string("{type = \"half_space\", point = [0.3, 0, 0], normal = [1, 0, 0], filled = false}"));
    EXPECT_TRUE(plane.exact);
    EXPECT_NEAR(total(plane), 1.0, 1e-12);
    for (const Vec & d : plane.direction) EXPECT_NEAR(d[0], 1.0, 1e-12);
}

TEST(RegionMeshTest, BoundaryZonesAreTheirFaces) {
    const Fixture f;
    const region::Elements left = f.elements("\"left\"");
    EXPECT_TRUE(left.exact);
    EXPECT_NEAR(total(left), 1.0, 1e-12);
    for (const Vec & d : left.direction) EXPECT_NEAR(d[0], -1.0, 1e-12);
    EXPECT_NEAR(total(f.elements("{type = \"boundary\", of = \"domain\"}")), 2.0 * N_DIM, 1e-12);
    // The edges of a side: its two end points (2D), or the square around it (3D)
    const region::Elements edges = f.elements("{type = \"boundary\", of = \"left\"}");
    EXPECT_EQ(edges.dimension, N_DIM - 2);
    EXPECT_NEAR(total(edges), N_DIM == 2 ? 2.0 : 4.0, 1e-12);
    // The faces around a filled region of whole cells face out of it: the domain minus a ball's cells
    const std::string ball = N_DIM == 2 ? "{type = \"disk\", center = [0.5, 0.5], radius = 0.3}"
                                        : "{type = \"sphere\", center = [0.5, 0.5, 0.5], radius = 0.3}";
    const region::Elements around = f.elements("{type = \"boundary\", of = {type = \"difference\", regions = "
                                               "[\"domain\", " + ball + "]}}");
    const auto [n, flux] = divergence(around, {0.5, 0.5, N_DIM == 3 ? 0.5 : 0.0});
    EXPECT_LT(n, 1e-2);
    EXPECT_GT(flux, 0.0);
}

TEST(RegionMeshTest, FilledRegionsClipSurfacesAndCurves) {
    const Fixture f;
    // A side cut along mesh nodes by a box: exactly the part inside, or outside
    const std::string box = N_DIM == 2 ? "{type = \"rectangle\", center = [0, 0.4], size = [1, 0.4]}"
                                       : "{type = \"box\", center = [0, 0.5, 0.5], size = [1, 0.3333333333333333, 2]}";
    const double in = N_DIM == 2 ? 0.4 : 1.0 / 3.0;
    EXPECT_NEAR(total(f.elements("{type = \"intersection\", regions = [\"left\", " + box + "]}")), in, 1e-12);
    EXPECT_NEAR(total(f.elements("{type = \"difference\", regions = [\"left\", " + box + "]}")), 1.0 - in, 1e-12);
    // A sphere's (circle's) half below a plane through its center: half the measure
    const std::string ball = N_DIM == 2 ? "{type = \"disk\", center = [0.5, 0.5], radius = 0.3, filled = false}"
                                        : "{type = \"sphere\", center = [0.5, 0.5, 0.5], radius = 0.3, filled = false}";
    const std::string below = "{type = \"half_space\", point = [0.5, 0.5, 0.5], normal = [0, 1, 0]}";
    const double half = N_DIM == 2 ? PI * 0.3 : 2.0 * PI * 0.09;
    const region::Region cap = f.region("{type = \"intersection\", regions = [" + ball + ", " + below + "]}");
    EXPECT_EQ(cap.dimension(), N_DIM - 1);
    EXPECT_FALSE(cap.closed());
    EXPECT_NEAR(total(region::elements(*f.mesh, cap)), half, 2e-2 * half);
    // Its boundary: the equator (2D: two points), running opposite ways for the two halves
    const region::Elements upper = f.elements("{type = \"boundary\", of = {type = \"difference\", regions = [" + ball +
                                              ", " + below + "]}}");
    const region::Elements lower = f.elements("{type = \"boundary\", of = {type = \"intersection\", regions = [" + ball +
                                              ", " + below + "]}}");
    if constexpr (N_DIM == 2) {
        EXPECT_NEAR(total(lower), 2.0, 1e-12);
    } else {
        EXPECT_NEAR(total(lower), 2.0 * PI * 0.3, 2e-2);
        ASSERT_GT(upper.size(), 0u);
        ASSERT_GT(lower.size(), 0u);
        auto turn = [](const region::Elements & e) {
            double s = 0.0;
            for (size_t k = 0; k < e.size(); k++) {
                const double rx = e.center[k][0] - 0.5, rz = e.center[k][2] - 0.5;
                s += (rz * e.direction[k][0] - rx * e.direction[k][2]) * e.measure[k];  // about +y
            }
            return s;
        };
        EXPECT_LT(turn(lower) * turn(upper), 0.0);
    }
}

TEST(RegionMeshTest, RejectsMixedDimensionsThatAreNotClipping) {
    const Fixture f;
    const std::string c = N_DIM == 2 ? "center = [0.5, 0.5]" : "center = [0.5, 0.5, 0.5], normal = [0, 0, 1]";
    const std::string circle = "{type = \"circle\", " + c + ", radius = 0.2}";
    const std::string ball = "{type = \"sphere\", center = [0.5, 0.5, 0.5], radius = 0.3}";
    auto error = [&](const std::string & table) {
        try {
            f.region(table);
        } catch (const InputError & e) {
            return std::string(e.what());
        }
        return std::string("accepted");
    };
    EXPECT_NE(error("{type = \"union\", regions = [" + circle + ", " + ball + "]}").find("one dimension"), std::string::npos);
    EXPECT_NE(error("{type = \"intersection\", regions = [" + circle + ", " + circle + ", " + ball + "]}")
                  .find("combine by union"),
              std::string::npos);
    EXPECT_NE(error("{type = \"difference\", regions = [" + ball + ", " + circle + "]}").find("takes nothing"),
              std::string::npos);
    EXPECT_NE(error(ball.substr(0, ball.size() - 1) + ", flip = true}").find("flip"), std::string::npos);
    EXPECT_NE(error(ball.substr(0, ball.size() - 1) + ", resolution = 2}").find("resolution"), std::string::npos);
    EXPECT_NE(error(circle.substr(0, circle.size() - 1) + ", resolution = 99}").find("resolution"), std::string::npos);
    EXPECT_NE(error("{type = \"boundary\", of = {type = \"intersection\", regions = [\"left\", " + ball + "]}}")
                  .find("not defined"),
              std::string::npos);
    EXPECT_EQ(error("{type = \"intersection\", regions = [" + ball + ", " + circle + "]}"), "accepted");
}

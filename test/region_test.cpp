/**
 * @file region_test.cpp
 * @author Matthew Bonanni (mbonanni001@gmail.com)
 * @brief Regions: membership and distances of primitives, transforms and
 *        booleans; device against host; volume fractions on unstructured meshes.
 * @version 0.1
 * @date 2026-10-15
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 *
 */

#include <gtest/gtest.h>
#include <Kokkos_Core.hpp>

#include <array>
#include <cmath>
#include <functional>
#include <random>
#include <string>
#include <vector>

#include "gmsh_fixtures.h"
#include "input.h"
#include "region.h"
#include "test_fixtures.h"

namespace {

using Point = std::array<double, 3>;
constexpr double PI = 3.14159265358979323846;

region::Region make_region(const std::string & table) {
    return region::Region(parse_toml("r = " + table).at("r"), "r");
}

std::vector<Point> random_points(const int n, const double lo, const double hi, const uint32_t seed = 3) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> u(lo, hi);
    std::vector<Point> points;
    for (int i = 0; i < n; i++) points.push_back({u(rng), u(rng), N_DIM == 3 ? u(rng) : 0.0});
    return points;
}

double dot(const Point & a, const Point & b) { return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]; }

Point normalized(Point a) {
    const double n = std::sqrt(dot(a, a));
    for (double & v : a) v /= n;
    return a;
}

/** @brief Rotation of v about the unit axis a by angle (radians). */
Point rotate(const Point & v, const Point & a, const double angle) {
    const double c = std::cos(angle), s = std::sin(angle);
    const Point axv = {a[1] * v[2] - a[2] * v[1], a[2] * v[0] - a[0] * v[2], a[0] * v[1] - a[1] * v[0]};
    Point r;
    for (size_t i = 0; i < 3; i++) r[i] = v[i] * c + axv[i] * s + a[i] * dot(a, v) * (1.0 - c);
    return r;
}

struct Case {
    std::string table;
    std::function<bool(const Point &)> inside;
    double volume;  // in the unit square or cube
};

const double TILT_ANGLE = 0.7;
const Point TILT_AXIS = normalized({1.0, 2.0, 3.0});

/** @brief Regions inside the unit square (2D) or cube (3D), with membership and measure. */
std::vector<Case> cases() {
    std::vector<Case> c;
    const Point center = {0.5, 0.5, N_DIM == 3 ? 0.5 : 0.0};
    auto rel = [center](const Point & x) { return Point{x[0] - center[0], x[1] - center[1], x[2] - center[2]}; };
    // Box of size (0.5, 0.3, 0.2) rotated by TILT_ANGLE (degrees in the input), then moved to the center
    auto in_box = [rel](const Point & x) {
        const Point a = N_DIM == 3 ? TILT_AXIS : Point{0.0, 0.0, 1.0};
        const Point q = rotate(rel(x), a, -TILT_ANGLE);
        return std::abs(q[0]) < 0.25 && std::abs(q[1]) < 0.15 && (N_DIM == 2 || std::abs(q[2]) < 0.1);
    };
    const std::string deg = std::to_string(TILT_ANGLE * 180.0 / PI);
    if constexpr (N_DIM == 2) {
        c.push_back({"{type = \"disk\", center = [0.5, 0.5], radius = 0.35}",
                     [rel](const Point & x) { return dot(rel(x), rel(x)) < 0.35 * 0.35; }, PI * 0.35 * 0.35});
        c.push_back({"{type = \"annulus\", center = [0.5, 0.5], radius = 0.35, inner_radius = 0.15}",
                     [rel](const Point & x) {
                         const double r2 = dot(rel(x), rel(x));
                         return r2 < 0.35 * 0.35 && r2 > 0.15 * 0.15;
                     },
                     PI * (0.35 * 0.35 - 0.15 * 0.15)});
        c.push_back({"{type = \"rectangle\", center = [0, 0], size = [0.5, 0.3], rotate = " + deg +
                         ", translate = [0.5, 0.5]}",
                     in_box, 0.15});
        c.push_back({"{type = \"ellipse\", center = [0.5, 0.5], semi_axes = [0.4, 0.2]}",
                     [rel](const Point & x) {
                         const Point q = rel(x);
                         return q[0] * q[0] / 0.16 + q[1] * q[1] / 0.04 < 1.0;
                     },
                     PI * 0.4 * 0.2});
        c.push_back({"{type = \"polygon\", vertices = [[0.1, 0.1], [0.9, 0.2], [0.5, 0.5], [0.6, 0.9]]}",
                     [](const Point & x) {
                         // Two triangles: (0.1,0.1)-(0.9,0.2)-(0.5,0.5) and (0.1,0.1)-(0.5,0.5)-(0.6,0.9)
                         auto in_tri = [&x](Point a, Point b, Point p) {
                             auto side = [](const Point & u, const Point & v, const Point & w) {
                                 return (v[0] - u[0]) * (w[1] - u[1]) - (v[1] - u[1]) * (w[0] - u[0]);
                             };
                             const double s1 = side(a, b, x), s2 = side(b, p, x), s3 = side(p, a, x);
                             return (s1 > 0 && s2 > 0 && s3 > 0) || (s1 < 0 && s2 < 0 && s3 < 0);
                         };
                         return in_tri({0.1, 0.1, 0}, {0.9, 0.2, 0}, {0.5, 0.5, 0}) ||
                                in_tri({0.1, 0.1, 0}, {0.5, 0.5, 0}, {0.6, 0.9, 0});
                     },
                     0.5 * std::abs(0.8 * 0.4 - 0.1 * 0.4) + 0.5 * std::abs(0.4 * 0.8 - 0.4 * 0.5)});
        c.push_back({"{type = \"difference\", a = {type = \"disk\", center = [0.5, 0.5], radius = 0.4}, "
                     "b = {type = \"rectangle\", center = [0.5, 0.5], size = [0.3, 0.2], rotate = 45, pivot = [0.5, 0.5]}}",
                     [rel](const Point & x) {
                         const Point q = rotate(rel(x), {0.0, 0.0, 1.0}, -PI / 4.0);
                         return dot(rel(x), rel(x)) < 0.16 && !(std::abs(q[0]) < 0.15 && std::abs(q[1]) < 0.1);
                     },
                     PI * 0.16 - 0.06});
        c.push_back({"{type = \"intersection\", regions = [{type = \"half_space\", point = [0.5, 0.5], normal = [1, 1]}, "
                     "{type = \"disk\", center = [0.5, 0.5], radius = 0.4}]}",
                     [rel](const Point & x) { return dot(rel(x), rel(x)) < 0.16 && rel(x)[0] + rel(x)[1] < 0.0; },
                     0.5 * PI * 0.16});
    } else {
        c.push_back({"{type = \"sphere\", center = [0.5, 0.5, 0.5], radius = 0.35}",
                     [rel](const Point & x) { return dot(rel(x), rel(x)) < 0.35 * 0.35; },
                     4.0 / 3.0 * PI * 0.35 * 0.35 * 0.35});
        const Point axis = normalized({1.0, 1.0, 0.5});
        c.push_back({"{type = \"cylinder\", center = [0.5, 0.5, 0.5], axis = [1, 1, 0.5], radius = 0.3, "
                     "inner_radius = 0.15, length = 0.4}",
                     [rel, axis](const Point & x) {
                         const Point q = rel(x);
                         const double a = dot(q, axis), r2 = dot(q, q) - a * a;
                         return std::abs(a) < 0.2 && r2 < 0.09 && r2 > 0.0225;
                     },
                     PI * (0.09 - 0.0225) * 0.4});
        c.push_back({"{type = \"box\", center = [0, 0, 0], size = [0.5, 0.3, 0.2], "
                     "rotate = {axis = [1, 2, 3], angle = " + deg + "}, translate = [0.5, 0.5, 0.5]}",
                     in_box, 0.03});
        c.push_back({"{type = \"cone\", base = [0.3, 0.5, 0.4], axis = [2, 0, 1], length = 0.5, base_radius = 0.25, "
                     "top_radius = 0.1}",
                     [](const Point & x) {
                         const Point a = normalized({2.0, 0.0, 1.0});
                         const Point q = {x[0] - 0.3, x[1] - 0.5, x[2] - 0.4};
                         const double h = dot(q, a), r2 = dot(q, q) - h * h;
                         const double r = 0.25 + (0.1 - 0.25) * h / 0.5;
                         return h > 0.0 && h < 0.5 && r2 < r * r;
                     },
                     PI * 0.5 / 3.0 * (0.0625 + 0.025 + 0.01)});
        c.push_back({"{type = \"torus\", center = [0.5, 0.5, 0.5], axis = [0, 1, 1], major_radius = 0.3, "
                     "minor_radius = 0.1}",
                     [rel](const Point & x) {
                         const Point a = normalized({0.0, 1.0, 1.0});
                         const Point q = rel(x);
                         const double h = dot(q, a), rho = std::sqrt(std::max(dot(q, q) - h * h, 0.0));
                         return (rho - 0.3) * (rho - 0.3) + h * h < 0.01;
                     },
                     2.0 * PI * PI * 0.3 * 0.01});
        c.push_back({"{type = \"ellipsoid\", center = [0, 0, 0], semi_axes = [0.4, 0.25, 0.15], "
                     "rotate = {axes = [[0, 1, 0], [0, 0, 1], [1, 0, 0]]}, translate = [0.5, 0.5, 0.5]}",
                     [rel](const Point & x) {
                         const Point q = rel(x);
                         // Local axes are the rows: local = (y, z, x)
                         return q[1] * q[1] / 0.16 + q[2] * q[2] / 0.0625 + q[0] * q[0] / 0.0225 < 1.0;
                     },
                     4.0 / 3.0 * PI * 0.4 * 0.25 * 0.15});
        c.push_back({"{type = \"difference\", a = {type = \"sphere\", center = [0.5, 0.5, 0.5], radius = 0.4}, "
                     "b = {type = \"union\", regions = [{type = \"cylinder\", center = [0.5, 0.5, 0.5], axis = [0, 0, 1], "
                     "radius = 0.1}, {type = \"box\", center = [1.4, 1, 1], size = [0.2, 0.2, 0.2], scale = 0.5}]}}",
                     [rel](const Point & x) {
                         const Point q = rel(x);
                         const bool cyl = q[0] * q[0] + q[1] * q[1] < 0.01;
                         // The box scales about the origin: center (0.7, 0.5, 0.5), size 0.1
                         const bool box = std::abs(x[0] - 0.7) < 0.05 && std::abs(x[1] - 0.5) < 0.05 &&
                                          std::abs(x[2] - 0.5) < 0.05;
                         return dot(q, q) < 0.16 && !cyl && !box;
                     },
                     // Sphere minus the cylinder through it (with its caps) minus the box, which lies inside the sphere
                     4.0 / 3.0 * PI * 0.064 - (PI * 0.01 * 2.0 * std::sqrt(0.15) +
                                               2.0 * PI / 3.0 * (0.4 - std::sqrt(0.15)) * (0.4 - std::sqrt(0.15)) *
                                                   (3.0 * 0.4 - (0.4 - std::sqrt(0.15)))) -
                         0.001});
        c.push_back({"{type = \"implicit\", f = \"(x - 0.5)^2 + (y - 0.5)^2 + (z - 0.5)^2 - 0.09\"}",
                     [rel](const Point & x) { return dot(rel(x), rel(x)) < 0.09; }, 4.0 / 3.0 * PI * 0.027});
    }
    return c;
}

std::shared_ptr<Mesh> coarse_unstructured_mesh() {
    auto mesh = std::make_shared<Mesh>();
    if constexpr (N_DIM == 2) {
        mesh->init_file(write_temp("mallard_region.msh", jittered_mixed_mesh(8)));
    } else {
        mesh->init_file(write_temp("mallard_region.msh", jittered_periodic_mesh_3d(5, 0.2)));
    }
    return mesh;
}

} // namespace

TEST(RegionTest, MembershipMatchesTheShapes) {
    for (const Case & c : cases()) {
        const region::Region r = make_region(c.table);
        int n_inside = 0;
        for (const Point & x : random_points(4000, 0.0, 1.0)) {
            const bool inside = r.distance(x.data()) < 0.0;
            n_inside += inside;
            EXPECT_EQ(inside, c.inside(x)) << c.table << " at (" << x[0] << ", " << x[1] << ", " << x[2] << ")";
        }
        EXPECT_GT(n_inside, 0) << c.table;
    }
}

TEST(RegionTest, DistancesAreOneLipschitzLowerBounds) {
    // |phi(a) - phi(b)| <= |a - b|: what makes cells farther than their size from the surface safe to skip
    const auto a = random_points(3000, -0.2, 1.2, 5), b = random_points(3000, -0.2, 1.2, 6);
    for (const Case & c : cases()) {
        const region::Region r = make_region(c.table);
        if (!r.bounded()) continue;
        for (size_t i = 0; i < a.size(); i++) {
            Point d;
            for (size_t k = 0; k < 3; k++) d[k] = a[i][k] - b[i][k];
            EXPECT_LE(std::abs(r.distance(a[i].data()) - r.distance(b[i].data())), std::sqrt(dot(d, d)) * (1.0 + 1e-12))
                << c.table;
        }
    }
    // Exact where the surface is simple: a sphere scaled by 2 is a sphere of twice the radius
    const region::Region s = make_region("{type = \"sphere\", center = [0.1, 0, 0], radius = 0.5, scale = 2}");
    const Point x = {1.2, 0.0, 0.0};
    EXPECT_NEAR(s.distance(x.data()), 1.2 - 0.2 - 1.0 + 0.0 * x[1], 1e-15);
}

TEST(RegionTest, DeviceMatchesHost) {
    const auto points = random_points(5000, -0.1, 1.1);
    const uint32_t n = static_cast<uint32_t>(points.size());
    Kokkos::View<double *[3]> d_points("points", n);
    auto h_points = Kokkos::create_mirror_view(d_points);
    for (uint32_t i = 0; i < n; i++) {
        for (size_t k = 0; k < 3; k++) h_points(i, k) = points[i][k];
    }
    Kokkos::deep_copy(d_points, h_points);
    for (const Case & c : cases()) {
        const region::Region r = make_region(c.table);
        const region::RegionView<> device = r.device();
        Kokkos::View<double *> d_phi("phi", n);
        Kokkos::parallel_for("region_distance", n, KOKKOS_LAMBDA(const uint32_t i) {
            const double x[3] = {d_points(i, 0), d_points(i, 1), d_points(i, 2)};
            d_phi(i) = device.distance(x);
        });
        auto h_phi = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), d_phi);
        for (uint32_t i = 0; i < n; i++) {
#ifdef KOKKOS_ENABLE_CUDA
            // Contracted multiply-adds and the device's square roots differ by rounding
            EXPECT_NEAR(h_phi(i), r.distance(points[i].data()), 1e-14) << c.table;
#else
            EXPECT_EQ(h_phi(i), r.distance(points[i].data())) << c.table;
#endif
        }
    }
}

TEST(RegionTest, VolumeFractionsIntegrateToTheVolumeOnCoarseUnstructuredMeshes) {
    const auto mesh = coarse_unstructured_mesh();
    for (const Case & c : cases()) {
        const region::Region r = make_region(c.table);
        const std::vector<double> f = region::volume_fractions(*mesh, r);
        double volume = 0.0;
        for (uint32_t i = 0; i < mesh->n_cells; i++) {
            ASSERT_GE(f[i], 0.0);
            ASSERT_LE(f[i], 1.0);
            volume += f[i] * double(mesh->h_cell_volume(i));
        }
        EXPECT_NEAR(volume, c.volume, (N_DIM == 2 ? 2e-5 : 2e-4) * c.volume) << c.table;
    }
}

TEST(RegionTest, FractionsAreExactlyOneInsideAndZeroOutside) {
    const auto mesh = coarse_unstructured_mesh();
    const region::Region r = make_region(N_DIM == 2 ? "{type = \"disk\", center = [0.5, 0.5], radius = 0.4}"
                                                    : "{type = \"sphere\", center = [0.5, 0.5, 0.5], radius = 0.45}");
    const std::vector<double> f = region::volume_fractions(*mesh, r);
    int n_one = 0, n_zero = 0, n_cut = 0;
    for (uint32_t c = 0; c < mesh->n_cells; c++) {
        bool all_in = true, all_out = true;
        for (uint32_t k = 0; k < mesh->h_n_nodes_of_cell(c); k++) {
            const uint32_t node = mesh->h_node_of_cell(c, k);
            double x[3] = {0.0, 0.0, 0.0};
            FOR_I_DIM x[i] = double(mesh->h_node_coords(node, i));
            const bool in = r.distance(x) < 0.0;
            all_in = all_in && in;
            all_out = all_out && !in;
        }
        // Convex cells with every node inside a convex region lie inside it
        if (all_in) {
            EXPECT_EQ(f[c], 1.0);
            n_one++;
        } else if (all_out && f[c] == 0.0) {
            n_zero++;
        } else {
            n_cut++;
        }
    }
    EXPECT_GT(n_one, 0);
    EXPECT_GT(n_zero, 0);
    EXPECT_GT(n_cut, 0);
}

TEST(RegionTest, SmoothEdgesFollowTheSmoothstepAndKeepTheVolume) {
    const region::Region r =
        make_region(N_DIM == 2 ? "{type = \"disk\", center = [0.5, 0.5], radius = 0.3, edge = 0.1}"
                               : "{type = \"sphere\", center = [0.5, 0.5, 0.5], radius = 0.3, edge = 0.1}");
    // Smoothstep across [r - edge / 2, r + edge / 2]
    for (const double d : {-0.06, -0.05, -0.025, 0.0, 0.025, 0.05, 0.06}) {
        const double x[3] = {0.8 + d, 0.5, N_DIM == 3 ? 0.5 : 0.0};
        const double s = std::min(std::max(0.5 - d / 0.1, 0.0), 1.0);
        EXPECT_NEAR(r.weight(x), s * s * (3.0 - 2.0 * s), 1e-14) << d;
    }
    // The odd smoothstep moves as much volume out as in, up to the curvature of the surface
    const auto mesh = coarse_unstructured_mesh();
    const std::vector<double> f = region::volume_fractions(*mesh, r);
    double volume = 0.0;
    for (uint32_t i = 0; i < mesh->n_cells; i++) volume += f[i] * double(mesh->h_cell_volume(i));
    // Disk: pi (R^2 + e^2 / 20); sphere: 4/3 pi R^3 + pi R e^2 / 5 (from the smoothstep's second moment)
    const double exact = N_DIM == 2 ? PI * (0.09 + 0.01 / 20.0) : 4.0 / 3.0 * PI * 0.027 + PI * 0.3 * 0.01 / 5.0;
    EXPECT_NEAR(volume, exact, 2e-4 * exact);
}

TEST(RegionTest, RejectsInvalidRegions) {
    for (const std::string t : {"{type = \"blob\"}", "{type = \"sphere\", radius = 1}",
                                "{type = \"sphere\", center = [0, 0, 0], radius = -1}",
                                "{type = \"sphere\", center = [0, 0, 0], radius = 1, raduis = 2}",
                                "{type = \"union\", regions = []}", "{type = \"difference\", a = {type = \"sphere\", "
                                                                    "center = [0, 0, 0], radius = 1}}",
                                "{type = \"cylinder\", center = [0, 0, 0], axis = [0, 0, 1], radius = 1, "
                                "inner_radius = 2}",
                                "{type = \"implicit\", f = \"x + q\"}",
                                "{type = \"sphere\", center = [0, 0, 0], radius = 1, rotate = {axes = [[1, 0, 0], "
                                "[1, 0, 0], [0, 0, 1]]}}",
                                "{type = \"sphere\", center = [0, 0, 0], radius = 1, edge = -1}"}) {
        EXPECT_ANY_THROW(make_region(t)) << t;
    }
}

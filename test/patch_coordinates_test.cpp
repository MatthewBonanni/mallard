/**
 * @file patch_coordinates_test.cpp
 * @author Matthew Bonanni (mbonanni001@gmail.com)
 * @brief Tests of the coordinates u, v of boundary patches against patches with known coordinates.
 * @version 0.1
 * @date 2026-11-17
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 *
 */

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <functional>
#include <optional>
#include <string>
#include <vector>

#include "input.h"
#include "patch_coordinates.h"

namespace {

using patch::Vec3;

constexpr double PI = 3.14159265358979323846;

Vec3 center(const std::vector<Vec3> & face) {
    Vec3 c{};
    for (const Vec3 & p : face) {
        for (size_t d = 0; d < 3; d++) c[d] += p[d] / static_cast<double>(face.size());
    }
    return c;
}

/** @brief The centroid of a flat face's area. */
Vec3 centroid(const std::vector<Vec3> & face) {
    Vec3 c{};
    double area = 0.0;
    for (size_t k = 1; k + 1 < face.size(); k++) {
        const Vec3 a{face[k][0] - face[0][0], face[k][1] - face[0][1], face[k][2] - face[0][2]};
        const Vec3 b{face[k + 1][0] - face[0][0], face[k + 1][1] - face[0][1], face[k + 1][2] - face[0][2]};
        const double w = std::sqrt(std::pow(a[1] * b[2] - a[2] * b[1], 2) + std::pow(a[2] * b[0] - a[0] * b[2], 2) +
                                   std::pow(a[0] * b[1] - a[1] * b[0], 2));
        for (size_t d = 0; d < 3; d++) c[d] += w * (face[0][d] + face[k][d] + face[k + 1][d]) / 3.0;
        area += w;
    }
    for (double & x : c) x /= area;
    return c;
}

/**
 * @brief na x nb quadrilaterals (or their halves) over the unit square mapped
 *        by X, joined around a if `wrap`; `inward` gives the side of the domain.
 */
patch::Surface grid(int na, int nb, bool wrap, const std::function<Vec3(double, double)> & X,
                    const std::function<Vec3(const Vec3 &)> & inward, bool triangles = false) {
    patch::Surface s;
    for (int i = 0; i < na; i++) {
        for (int j = 0; j < nb; j++) {
            auto P = [&](int a, int b) { return X(double(wrap ? a % na : a) / na, double(b) / nb); };
            const std::vector<Vec3> q{P(i, j), P(i + 1, j), P(i + 1, j + 1), P(i, j + 1)};
            const Vec3 n = inward(center(q));
            if (triangles) {
                s.faces.push_back({q[0], q[1], q[2]});
                s.faces.push_back({q[2], q[3], q[0]});
                s.inward.push_back(n);
            } else {
                s.faces.push_back(q);
            }
            s.inward.push_back(n);
        }
    }
    return s;
}

Vec3 up(const Vec3 &) { return {0.0, 0.0, 1.0}; }

/** @brief The ring r(q) e_r(2 pi p) + z(q) e_z about the z axis, the domain inside it. */
patch::Surface ring(int n, double r0, double r1, double height) {
    return grid(4 * n, n, true,
                [=](double p, double q) {
                    const double r = r0 + (r1 - r0) * q;
                    return Vec3{r * std::cos(2 * PI * p), r * std::sin(2 * PI * p), height * q};
                },
                [=](const Vec3 & c) { return height > 0.0 ? Vec3{-c[0], -c[1], 0.0} : Vec3{0.0, 0.0, 1.0}; });
}

double azimuth(const Vec3 & x) {
    const double turn = std::atan2(x[1], x[0]) / (2 * PI);
    return turn - std::floor(turn);
}

/** @brief |a - b| of two values in [0, 1) that wrap. */
double around(double a, double b) {
    const double d = a - b;
    return std::abs(d - std::round(d));
}

std::string error_of(const patch::Surface & s, const patch::Corners & at = {}) {
    try {
        patch::coordinates(s, at);
    } catch (const InputError & e) {
        return e.what();
    }
    return "";
}

} // namespace

TEST(PatchCoordinatesTest, ParallelogramHasItsAffineCoordinates) {
    // Affine coordinates are harmonic with boundary values linear in arc
    // length, and piecewise linear elements hold them exactly: any error in
    // the weights, the boundary values or the face averages shows here
    const Vec3 o{0.3, -0.2, 0.1}, a{2.0, 0.3, 0.0}, b{0.7, 1.1, 0.0};
    for (const bool triangles : {false, true}) {
        const patch::Surface s = grid(7, 5, false, [&](double p, double q) {
            return Vec3{o[0] + p * a[0] + q * b[0], o[1] + p * a[1] + q * b[1], o[2]};
        }, up, triangles);
        const patch::Coordinates c = patch::coordinates(s, {});
        EXPECT_EQ(c.shape, patch::Shape::FOUR_SIDED);
        const double det = a[0] * b[1] - a[1] * b[0];
        for (size_t f = 0; f < s.faces.size(); f++) {
            const Vec3 x = center(s.faces[f]);
            const double p = ((x[0] - o[0]) * b[1] - (x[1] - o[1]) * b[0]) / det;
            const double q = (a[0] * (x[1] - o[1]) - a[1] * (x[0] - o[0])) / det;
            ASSERT_NEAR(c.faces[f].u, p, 1e-12) << f;
            ASSERT_NEAR(c.faces[f].v, q, 1e-12) << f;
        }
    }
}

TEST(PatchCoordinatesTest, RectangleRunsAlongTheAxesAfterItsNormalInMeters) {
    // Across x: u along +y and v along +z, whichever way the domain lies;
    // across y: u along +z, v along +x. u_m and v_m are the distances from
    // the sides u = 0 and v = 0
    for (const double side : {1.0, -1.0}) {
        const patch::Surface yz = grid(8, 6, false, [](double p, double q) { return Vec3{0.4, 0.2 * p, 1.0 + 0.1 * q}; },
                                       [=](const Vec3 &) { return Vec3{side, 0.0, 0.0}; });
        const patch::Coordinates c = patch::coordinates(yz, {});
        for (size_t f = 0; f < yz.faces.size(); f++) {
            const Vec3 x = center(yz.faces[f]);
            ASSERT_NEAR(c.faces[f].u, x[1] / 0.2, 1e-12);
            ASSERT_NEAR(c.faces[f].v, (x[2] - 1.0) / 0.1, 1e-12);
            ASSERT_NEAR(c.faces[f].u_m, x[1], 1e-12);
            ASSERT_NEAR(c.faces[f].v_m, x[2] - 1.0, 1e-12);
        }
        const patch::Surface zx = grid(8, 6, false, [](double p, double q) { return Vec3{0.2 * p, -1.0, 0.1 * q}; },
                                       [=](const Vec3 &) { return Vec3{0.0, side, 0.0}; });
        const patch::Coordinates d = patch::coordinates(zx, {});
        for (size_t f = 0; f < zx.faces.size(); f++) {
            const Vec3 x = center(zx.faces[f]);
            ASSERT_NEAR(d.faces[f].u, x[2] / 0.1, 1e-12);
            ASSERT_NEAR(d.faces[f].v, x[0] / 0.2, 1e-12);
        }
    }
}

TEST(PatchCoordinatesTest, CornersGivenSetWhereUAndVStart) {
    const patch::Surface s = grid(8, 6, false, [](double p, double q) { return Vec3{2.0 * p, 1.0 * q, 0.0}; }, up);
    // From the corner (2, 1): u along -y, v along -x
    patch::Corners at;
    at.points = std::vector<Vec3>{{2.01, 0.98, 0.0}, {2.0, 0.0, 0.0}, {0.0, 0.0, 0.0}, {0.0, 1.0, 0.0}};
    const patch::Coordinates c = patch::coordinates(s, at);
    for (size_t f = 0; f < s.faces.size(); f++) {
        const Vec3 x = center(s.faces[f]);
        ASSERT_NEAR(c.faces[f].u, 1.0 - x[1], 1e-12);
        ASSERT_NEAR(c.faces[f].v, 1.0 - x[0] / 2.0, 1e-12);
    }
    at.points = std::vector<Vec3>{{0.0, 0.0, 0.0}, {2.0, 1.0, 0.0}, {2.0, 0.0, 0.0}, {0.0, 1.0, 0.0}};
    EXPECT_NE(error_of(s, at).find("in order around the patch's outline"), std::string::npos);
    at.points = std::vector<Vec3>{{0.0, 0.0, 0.0}, {2.0, 1.0, 0.0}};
    EXPECT_NE(error_of(s, at).find("corners has 2 points"), std::string::npos);
}

TEST(PatchCoordinatesTest, WarpedPatchWithCurvedSidesIsCoveredMonotonically) {
    // A saddle with bulging sides: u grows along one family of mesh lines
    // and v along the other, both inside [0, 1]
    const int na = 12, nb = 9;
    const patch::Surface s = grid(na, nb, false, [](double p, double q) {
        const double x = 2.0 * p + 0.25 * std::sin(PI * q), y = q + 0.2 * std::sin(PI * p) * (2.0 * q - 1.0);
        return Vec3{x, y, 0.4 * (p - 0.5) * (q - 0.5) + 0.2 * p * p};
    }, up);
    const patch::Coordinates c = patch::coordinates(s, {});
    ASSERT_EQ(c.shape, patch::Shape::FOUR_SIDED);
    auto at = [&](int i, int j) { return c.faces[size_t(i * nb + j)]; };
    for (int i = 0; i < na; i++) {
        for (int j = 0; j < nb; j++) {
            ASSERT_GT(at(i, j).u, 0.0);
            ASSERT_LT(at(i, j).u, 1.0);
            ASSERT_GT(at(i, j).v, 0.0);
            ASSERT_LT(at(i, j).v, 1.0);
            if (i > 0) {
                ASSERT_GT(at(i, j).u, at(i - 1, j).u) << i << ", " << j;
            }
            if (j > 0) {
                ASSERT_GT(at(i, j).v, at(i, j - 1).v) << i << ", " << j;
            }
            if (i > 0) {
                ASSERT_GT(at(i, j).u_m, at(i - 1, j).u_m) << i << ", " << j;
            }
        }
    }
}

TEST(PatchCoordinatesTest, AnnulusIsLogarithmicInTheRadiusAndLinearInTheAngle) {
    // The harmonic function between two circles is log(r / r0) / log(r1 / r0),
    // and its conjugate the angle; second order in the mesh
    double coarse = 0.0;
    for (const int n : {8, 16}) {
        const patch::Surface s = ring(n, 0.5, 2.0, 0.0);
        const patch::Coordinates c = patch::coordinates(s, {});
        ASSERT_EQ(c.shape, patch::Shape::RING);
        double worst = 0.0;
        for (size_t f = 0; f < s.faces.size(); f++) {
            const Vec3 x = center(s.faces[f]);
            const double r = std::hypot(x[0], x[1]);
            worst = std::max(worst, std::abs(c.faces[f].v - std::log(r / 0.5) / std::log(4.0)));
            ASSERT_LT(around(c.faces[f].u, azimuth(x)), 1e-10);
            ASSERT_GE(c.faces[f].u, 0.0);
            ASSERT_LT(c.faces[f].u, 1.0);
            ASSERT_NEAR(c.faces[f].v_m, r - 0.5, 0.02);
            ASSERT_NEAR(c.faces[f].u_m, c.faces[f].u * 2 * PI * r, 0.06);
        }
        if (n == 8) coarse = worst;
        else EXPECT_LT(worst, 0.3 * coarse);
        EXPECT_LT(worst, 4e-3);
    }
}

TEST(PatchCoordinatesTest, SidesOfATubeAndOfAConeFollowTheirSlantAndAzimuth) {
    // A cone unrolls into a sector of an annulus about its apex: v is
    // logarithmic in the distance from the apex; on a tube it is linear.
    // v = 0 on the shorter rim, or the lower one of a tube
    const int n = 16;
    for (const double r1 : {2.0, 1.0, 0.5}) {
        const patch::Surface s = ring(n, 1.0, r1, 1.5);
        const patch::Coordinates c = patch::coordinates(s, {});
        ASSERT_EQ(c.shape, patch::Shape::RING);
        const double slant = std::hypot(r1 - 1.0, 1.5), circumference = 8.0 * n * std::sin(PI / (4.0 * n));
        for (size_t f = 0; f < s.faces.size(); f++) {
            const Vec3 x = center(s.faces[f]);
            const double q = x[2] / 1.5;
            const double from_small = r1 < 1.0 ? 1.0 - q : q, ratio = r1 < 1.0 ? 1.0 / r1 : r1;
            const double r = (r1 < 1.0 ? r1 : 1.0) * (1.0 + (ratio - 1.0) * from_small);
            const double exact = r1 == 1.0 ? q : std::log(r / std::min(r1, 1.0)) / std::log(ratio);
            ASSERT_NEAR(c.faces[f].v, exact, r1 == 1.0 ? 1e-10 : 5e-4) << r1;
            ASSERT_LT(around(c.faces[f].u, azimuth(x)), 1e-10) << r1;
            ASSERT_NEAR(c.faces[f].v_m, from_small * slant, r1 == 1.0 ? 1e-10 : 2e-3) << r1;
            if (r1 == 1.0) {
                ASSERT_NEAR(c.faces[f].u_m, c.faces[f].u * circumference, 1e-10);
            }
        }
    }
}

TEST(PatchCoordinatesTest, DiskHasTheRadiusAndTheAngleAboutItsCenter) {
    // The conformal map of a circle to the unit disk is (x - center) / R,
    // which linear elements hold: v = r / R and u the angle to round-off,
    // through the center too, wherever the circle lies. Quadrilaterals
    // around a fan
    const Vec3 o{1.0, -1.0, 0.5};
    const double R = 2.0;
    const int n = 16, na = 4 * n;
    patch::Surface s;
    auto P = [&](int i, int j) {
        const double r = R * j / n, t = 2 * PI * (i % na) / na;
        return Vec3{o[0] + r * std::cos(t), o[1] + r * std::sin(t), o[2]};
    };
    for (int i = 0; i < na; i++) {
        s.faces.push_back({o, P(i, 1), P(i + 1, 1)});
        s.inward.push_back({0.0, 0.0, -1.0});
        for (int j = 1; j < n; j++) {
            s.faces.push_back({P(i, j), P(i + 1, j), P(i + 1, j + 1), P(i, j + 1)});
            s.inward.push_back({0.0, 0.0, -1.0});
        }
    }
    const patch::Coordinates c = patch::coordinates(s, {});
    ASSERT_EQ(c.shape, patch::Shape::DISK);
    for (size_t f = 0; f < s.faces.size(); f++) {
        const Vec3 x = centroid(s.faces[f]);
        const Vec3 offset{x[0] - o[0], x[1] - o[1], 0.0};
        const double r = std::hypot(offset[0], offset[1]);
        ASSERT_NEAR(c.faces[f].v, r / R, 1e-10) << r;
        ASSERT_LT(around(c.faces[f].u, azimuth(offset)), 1e-10) << r;
        ASSERT_NEAR(c.faces[f].v_m, r, 0.01 * R);
        ASSERT_NEAR(c.faces[f].u_m, c.faces[f].u * 2 * PI * r, 0.03 * R);
    }
    // Four corners make the same disk a four-sided patch
    patch::Corners at;
    at.points = std::vector<Vec3>{P(0, n), P(n, n), P(2 * n, n), P(3 * n, n)};
    EXPECT_EQ(patch::coordinates(s, at).shape, patch::Shape::FOUR_SIDED);
}

TEST(PatchCoordinatesTest, OriginAndTowardSetWhereUStartsAndWhichWayItGrows) {
    // Four-sided: (u, v) = (0, 0) at the corner nearest to origin, u along
    // the side to the adjacent corner nearest to toward (or, without it,
    // along the default direction: +x across z), v along the other side
    const patch::Surface s = grid(8, 6, false, [](double p, double q) { return Vec3{2.0 * p, q, 0.0}; }, up);
    struct Case {
        Vec3 origin;
        std::optional<Vec3> toward;
        std::function<double(const Vec3 &)> u, v;
    };
    const std::vector<Case> cases{
        {{2.2, 0.9, 0.1}, std::nullopt, [](const Vec3 & x) { return 1.0 - 0.5 * x[0]; }, [](const Vec3 & x) { return 1.0 - x[1]; }},
        {{2.2, 0.9, 0.1}, Vec3{2.0, -3.0, 0.0}, [](const Vec3 & x) { return 1.0 - x[1]; }, [](const Vec3 & x) { return 1.0 - 0.5 * x[0]; }},
        {{-1.0, -1.0, 0.0}, Vec3{0.0, 0.4, 0.0}, [](const Vec3 & x) { return x[1]; }, [](const Vec3 & x) { return 0.5 * x[0]; }},
        {{-1.0, 2.0, 0.0}, Vec3{5.0, 5.0, 0.0}, [](const Vec3 & x) { return 0.5 * x[0]; }, [](const Vec3 & x) { return 1.0 - x[1]; }},
    };
    for (const Case & c : cases) {
        patch::Corners at;
        at.origin = c.origin;
        at.toward = c.toward;
        const patch::Coordinates got = patch::coordinates(s, at);
        for (size_t f = 0; f < s.faces.size(); f++) {
            const Vec3 x = center(s.faces[f]);
            ASSERT_NEAR(got.faces[f].u, c.u(x), 1e-12) << c.origin[0] << ", " << c.origin[1];
            ASSERT_NEAR(got.faces[f].v, c.v(x), 1e-12) << c.origin[0] << ", " << c.origin[1];
        }
    }

    // Rings and disks: u = 0 at the face nearest to origin, growing the default way (from +x toward +y around
    // z) or the way that reaches toward sooner
    const Vec3 point{-0.1, 1.3, 0.7};
    for (const patch::Surface & r : {ring(8, 1.0, 1.0, 1.5), ring(8, 0.5, 2.0, 0.0)}) {
        size_t nearest = 0;
        auto distance = [&](size_t f) {
            const Vec3 x = centroid(r.faces[f]);
            return std::hypot(x[0] - point[0], x[1] - point[1], x[2] - point[2]);
        };
        for (size_t f = 1; f < r.faces.size(); f++) {
            if (distance(f) < distance(nearest)) nearest = f;
        }
        const double start = azimuth(center(r.faces[nearest]));
        for (const double way : {1.0, -1.0}) {
            patch::Corners at;
            at.origin = point;
            if (way < 0.0) at.toward = Vec3{1.5, 0.2, 0.7};  // clockwise from the origin, near +y
            const patch::Coordinates c = patch::coordinates(r, at);
            EXPECT_EQ(c.faces[nearest].u, 0.0);
            for (size_t f = 0; f < r.faces.size(); f++) {
                ASSERT_LT(around(c.faces[f].u, way * (azimuth(center(r.faces[f])) - start)), 1e-10) << way;
                ASSERT_GE(c.faces[f].u, 0.0);
                ASSERT_LT(c.faces[f].u, 1.0);
            }
        }
    }

    // Points that do not decide are errors naming the candidates: within 1% of the patch's size of two corners,
    // on a ring's axis, at a disk's center, or half way around from the origin
    patch::Corners tied;
    tied.origin = Vec3{1.0, -4.0, 0.0};
    EXPECT_NE(error_of(s, tied).find("is as near to the corner (2, 0, 0) as to (0, 0, 0)"), std::string::npos)
        << error_of(s, tied);
    tied.origin = Vec3{1.1, -4.0, 0.0};  // far off, and nearer to (2, 0, 0): u from there along -x
    EXPECT_NEAR(patch::coordinates(s, tied).faces.back().u, patch::coordinates(s, {}).faces.front().u, 1e-12);
    tied.origin = Vec3{0.0, 0.0, 0.0};
    tied.toward = Vec3{1.5, 1.5, 0.0};
    EXPECT_NE(error_of(s, tied).find("toward (1.5, 1.5, 0) is as near to the corner"), std::string::npos)
        << error_of(s, tied);
    const patch::Surface tube = ring(8, 1.0, 1.0, 1.5);
    tied.toward.reset();
    tied.origin = Vec3{0.0, 0.0, 0.3};
    EXPECT_NE(error_of(tube, tied).find("on another side of the patch"), std::string::npos) << error_of(tube, tied);
    tied.origin = Vec3{3.0, 0.3, 0.3};
    tied.toward = Vec3{-3.0, -0.3, 0.3};
    EXPECT_NE(error_of(tube, tied).find("is opposite the origin"), std::string::npos) << error_of(tube, tied);
    tied.toward = Vec3{0.1, 2.0, 0.3};
    EXPECT_NE(patch::coordinates(tube, tied).summary.find(", growing toward ("), std::string::npos);

    patch::Corners at;
    at.toward = point;
    EXPECT_NE(error_of(s, at).find("toward is given without origin"), std::string::npos);
    at.origin = point;
    at.points = std::vector<Vec3>{{0.0, 0.0, 0.0}, {2.0, 0.0, 0.0}, {2.0, 1.0, 0.0}, {0.0, 1.0, 0.0}};
    EXPECT_NE(error_of(s, at).find("corners already says where u and v start"), std::string::npos);
}

TEST(PatchCoordinatesTest, FacesThatAreNotOnePatchAreInputErrorsWithAFix) {
    auto square = [](double x0, double y0, int n) {
        return grid(n, n, false, [=](double p, double q) { return Vec3{x0 + p, y0 + q, 0.0}; }, up);
    };
    // An L: six corners
    patch::Surface l = square(0.0, 0.0, 4);
    for (size_t f = l.faces.size(); f-- > 0;) {
        const Vec3 x = center(l.faces[f]);
        if (x[0] > 0.5 && x[1] > 0.5) {
            l.faces.erase(l.faces.begin() + long(f));
            l.inward.erase(l.inward.begin() + long(f));
        }
    }
    const std::string corners = error_of(l);
    EXPECT_NE(corners.find("has 6 corners"), std::string::npos) << corners;
    EXPECT_NE(corners.find("corners = [[x, y, z], ...]"), std::string::npos) << corners;
    patch::Corners at;
    at.points = std::vector<Vec3>{{0.0, 0.0, 0.0}, {1.0, 0.0, 0.0}, {0.5, 0.5, 0.0}, {0.0, 1.0, 0.0}};
    EXPECT_EQ(patch::coordinates(l, at).shape, patch::Shape::FOUR_SIDED);
    at.points = std::vector<Vec3>{};
    EXPECT_EQ(patch::coordinates(square(0.0, 0.0, 6), at).shape, patch::Shape::DISK);

    // Two squares apart
    patch::Surface two = square(0.0, 0.0, 3);
    const patch::Surface far = square(5.0, 0.0, 3);
    two.faces.insert(two.faces.end(), far.faces.begin(), far.faces.end());
    two.inward.insert(two.inward.end(), far.inward.begin(), far.inward.end());
    EXPECT_NE(error_of(two).find("2 separate pieces"), std::string::npos) << error_of(two);

    // A square with two holes
    patch::Surface holes = square(0.0, 0.0, 7);
    for (size_t f = holes.faces.size(); f-- > 0;) {
        const Vec3 x = center(holes.faces[f]);
        const bool hole = std::abs(x[1] - 0.5) < 0.05 && (std::abs(x[0] - 1.5 / 7) < 0.05 || std::abs(x[0] - 5.5 / 7) < 0.05);
        if (hole) {
            holes.faces.erase(holes.faces.begin() + long(f));
            holes.inward.erase(holes.inward.begin() + long(f));
        }
    }
    EXPECT_NE(error_of(holes).find("has 3 loops"), std::string::npos) << error_of(holes);

    // The whole surface of a tetrahedron
    const std::vector<Vec3> p{{0.0, 0.0, 0.0}, {1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}, {0.0, 0.0, 1.0}};
    patch::Surface closed;
    closed.faces = {{p[0], p[1], p[2]}, {p[0], p[1], p[3]}, {p[1], p[2], p[3]}, {p[0], p[2], p[3]}};
    closed.inward = {{0.0, 0.0, 1.0}, {0.0, 1.0, 0.0}, {-1.0, -1.0, -1.0}, {1.0, 0.0, 0.0}};
    EXPECT_NE(error_of(closed).find("closed surface"), std::string::npos) << error_of(closed);

    at.points = std::vector<Vec3>{p[0], p[1], p[2], p[3]};
    EXPECT_NE(error_of(ring(4, 1.0, 2.0, 0.0), at).find("the patch is a ring"), std::string::npos);
}

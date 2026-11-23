/**
 * @file region_surface_test.cpp
 * @author Matthew Bonanni (mbonanni001@gmail.com)
 * @brief Regions from CAD surfaces (STL, OBJ): signed distances against exact
 *        ones, the readers, orientation and nesting, units and transforms,
 *        measures, fractions, exact surfaces, and the errors of broken files.
 * @version 0.1
 * @date 2026-11-19
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 *
 */

#include <gtest/gtest.h>
#include <Kokkos_Core.hpp>

#include <array>
#include <cmath>
#include <cstring>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#include "gmsh_fixtures.h"
#include "input.h"
#include "region_input.h"
#include "region_mesh.h"
#include "region_sampler.h"
#include "test_fixtures.h"

namespace {

using Vec = std::array<double, 3>;
using Tri = std::array<Vec, 3>;

/** @brief The 12 triangles of the box [lo, hi], counterclockwise seen from outside. */
std::vector<Tri> cube(const Vec & lo, const Vec & hi) {
    std::vector<Tri> out;
    const auto corner = [&](const int c) {
        return Vec{(c & 1) ? hi[0] : lo[0], (c & 2) ? hi[1] : lo[1], (c & 4) ? hi[2] : lo[2]};
    };
    // Faces as corner indices, counterclockwise seen from outside
    const int faces[6][4] = {{0, 2, 3, 1}, {4, 5, 7, 6}, {0, 1, 5, 4}, {2, 6, 7, 3}, {0, 4, 6, 2}, {1, 3, 7, 5}};
    for (const auto & f : faces) {
        out.push_back({corner(f[0]), corner(f[1]), corner(f[2])});
        out.push_back({corner(f[0]), corner(f[2]), corner(f[3])});
    }
    return out;
}

std::vector<Tri> joined(std::vector<Tri> a, const std::vector<Tri> & b) {
    a.insert(a.end(), b.begin(), b.end());
    return a;
}

std::string stl_ascii(const std::vector<Tri> & tris) {
    std::ostringstream s;
    s.precision(17);
    s << "solid test\n";
    for (const Tri & t : tris) {
        s << "  facet normal 0 0 0\n    outer loop\n";
        for (const Vec & x : t) s << "      vertex " << x[0] << " " << x[1] << " " << x[2] << "\n";
        s << "    endloop\n  endfacet\n";
    }
    s << "endsolid test\n";
    return s.str();
}

std::string stl_binary(const std::vector<Tri> & tris) {
    std::string s(80, ' ');
    const auto u32 = [&](const uint32_t v) {
        for (int k = 0; k < 4; k++) s.push_back(static_cast<char>((v >> (8 * k)) & 0xFF));
    };
    u32(static_cast<uint32_t>(tris.size()));
    for (const Tri & t : tris) {
        for (int k = 0; k < 3; k++) u32(0);
        for (const Vec & x : t) {
            for (const double v : x) {
                const float f = static_cast<float>(v);
                uint32_t u = 0;
                std::memcpy(&u, &f, sizeof u);
                u32(u);
            }
        }
        s.push_back('\0');
        s.push_back('\0');
    }
    return s;
}

/** @brief The box [lo, hi] as an OBJ file of quadrilaterals. */
std::string obj_cube(const Vec & lo, const Vec & hi) {
    std::ostringstream s;
    s.precision(17);
    s << "# a box\no box\n";
    for (int c = 0; c < 8; c++) {
        s << "v " << ((c & 1) ? hi[0] : lo[0]) << " " << ((c & 2) ? hi[1] : lo[1]) << " " << ((c & 4) ? hi[2] : lo[2]) << "\n";
    }
    s << "vn 0 0 1\n";
    const int faces[6][4] = {{0, 2, 3, 1}, {4, 5, 7, 6}, {0, 1, 5, 4}, {2, 6, 7, 3}, {0, 4, 6, 2}, {1, 3, 7, 5}};
    for (const auto & f : faces) s << "f " << f[0] + 1 << "//1 " << f[1] + 1 << "//1 " << f[2] + 1 << "//1 " << f[3] - 8 << "\n";
    return s.str();
}

/** @brief A sphere of radius r at c: an icosahedron subdivided n times, its vertices on the sphere. */
std::vector<Tri> icosphere(const int n, const double r, const Vec & c) {
    const double g = (1.0 + std::sqrt(5.0)) / 2.0;
    std::vector<Vec> v = {{-1, g, 0}, {1, g, 0}, {-1, -g, 0}, {1, -g, 0}, {0, -1, g}, {0, 1, g},
                          {0, -1, -g}, {0, 1, -g}, {g, 0, -1}, {g, 0, 1}, {-g, 0, -1}, {-g, 0, 1}};
    const int f[20][3] = {{0, 11, 5}, {0, 5, 1}, {0, 1, 7}, {0, 7, 10}, {0, 10, 11}, {1, 5, 9}, {5, 11, 4},
                          {11, 10, 2}, {10, 7, 6}, {7, 1, 8}, {3, 9, 4}, {3, 4, 2}, {3, 2, 6}, {3, 6, 8},
                          {3, 8, 9}, {4, 9, 5}, {2, 4, 11}, {6, 2, 10}, {8, 6, 7}, {9, 8, 1}};
    const auto on_sphere = [&](Vec x) {
        const double l = std::sqrt(x[0] * x[0] + x[1] * x[1] + x[2] * x[2]);
        for (size_t i = 0; i < 3; i++) x[i] = c[i] + r * x[i] / l;
        return x;
    };
    std::vector<Tri> tris;
    for (const auto & t : f) tris.push_back({on_sphere(v[size_t(t[0])]), on_sphere(v[size_t(t[1])]), on_sphere(v[size_t(t[2])])});
    for (int level = 0; level < n; level++) {
        std::vector<Tri> next;
        for (const Tri & t : tris) {
            Vec m[3];
            for (size_t k = 0; k < 3; k++) {
                Vec x;
                for (size_t i = 0; i < 3; i++) x[i] = 0.5 * (t[k][i] + t[(k + 1) % 3][i]) - c[i];
                m[k] = on_sphere(x);
            }
            next.push_back({t[0], m[0], m[2]});
            next.push_back({m[0], t[1], m[1]});
            next.push_back({m[2], m[1], t[2]});
            next.push_back({m[0], m[1], m[2]});
        }
        tris = std::move(next);
    }
    return tris;
}

/** @brief A file in the scratch directory, written by rank 0. */
std::string file(const std::string & name, const std::string & content) { return write_temp_shared(name, content); }

region::Region read(const std::string & table) { return region::read(parse_toml("r = " + table).at("r"), "r"); }

std::string stl_region(const std::string & path, const std::string & extra = "") {
    return "{type = \"stl\", file = \"" + path + "\", units = \"m\"" + extra + "}";
}

std::string error_of(const std::string & table) {
    try {
        read(table);
    } catch (const InputError & e) {
        return e.what();
    }
    return "";
}

std::vector<Vec> random_points(const int n, const double lo, const double hi, const uint32_t seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> u(lo, hi);
    std::vector<Vec> points;
    for (int i = 0; i < n; i++) points.push_back({u(rng), u(rng), N_DIM == 3 ? u(rng) : 0.0});
    return points;
}

std::vector<double> device_distances(const region::Region & r, const std::vector<Vec> & points) {
    const uint32_t n = static_cast<uint32_t>(points.size());
    Kokkos::View<double *[3]> d_points("points", n);
    auto h_points = Kokkos::create_mirror_view(d_points);
    for (uint32_t i = 0; i < n; i++) {
        for (size_t k = 0; k < 3; k++) h_points(i, k) = points[i][k];
    }
    Kokkos::deep_copy(d_points, h_points);
    const region::RegionView<> device = r.device();
    Kokkos::View<double *> d_phi("phi", n);
    Kokkos::parallel_for("surface_distance", n, KOKKOS_LAMBDA(const uint32_t i) {
        const double x[3] = {d_points(i, 0), d_points(i, 1), d_points(i, 2)};
        d_phi(i) = device.distance(x);
    });
    auto h_phi = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), d_phi);
    return std::vector<double>(h_phi.data(), h_phi.data() + n);
}

// The test box: in 2D thick enough along z that its section's distances are those of a rectangle
const Vec LO = {0.23, 0.31, N_DIM == 3 ? 0.27 : -10.0}, HI = {0.71, 0.62, N_DIM == 3 ? 0.58 : 10.0};

std::string box_region(const std::string & extra = "") {
    const std::string center = "[" + std::to_string(0.5 * (LO[0] + HI[0])) + ", " + std::to_string(0.5 * (LO[1] + HI[1])) +
                               (N_DIM == 3 ? ", " + std::to_string(0.5 * (LO[2] + HI[2])) : "") + "]";
    const std::string size = "[" + std::to_string(HI[0] - LO[0]) + ", " + std::to_string(HI[1] - LO[1]) +
                             (N_DIM == 3 ? ", " + std::to_string(HI[2] - LO[2]) : "") + "]";
    return std::string("{type = \"") + (N_DIM == 3 ? "box" : "rectangle") + "\", center = " + center +
           ", size = " + size + extra + "}";
}

/** @brief The signed distance of a region on the host, callable from the sampler's KOKKOS_INLINE_FUNCTION templates. */
struct HostDistance {
    const region::RegionView<Kokkos::HostSpace> & view;

    KOKKOS_INLINE_FUNCTION double operator()(const double * x) const { return view.distance(x); }
};

/** @brief Node k of straight cell c on the host. */
struct HostNodes {
    const Mesh & mesh;
    uint32_t c;

    KOKKOS_INLINE_FUNCTION void operator()(const uint32_t k, double * x) const {
        const uint32_t n = mesh.h_nodes_of_cell(mesh.h_offsets_nodes_of_cell(c) + k);
        for (size_t d = 0; d < 3; d++) x[d] = d < N_DIM ? double(mesh.h_node_coords(n, d)) : 0.0;
    }
};

} // namespace

TEST(RegionSurfaceTest, DistancesAreThoseOfTheBoxTheFileHolds) {
    // A box's signed distance is the exact distance to its surface, which the surface's must be, on the device too
    const region::Region stl = read(stl_region(file("box.stl", stl_ascii(cube(LO, HI)))));
    const region::Region box = read(box_region());
    EXPECT_TRUE(stl.bounded());
    const auto points = random_points(4000, -0.2, 1.2, 1);
    const std::vector<double> device = device_distances(stl, points);
    for (size_t i = 0; i < points.size(); i++) {
        const double phi = stl.distance(points[i].data());
        EXPECT_NEAR(phi, box.distance(points[i].data()), 1e-14);
#ifdef KOKKOS_ENABLE_CUDA
        EXPECT_NEAR(device[i], phi, 1e-14);
#else
        EXPECT_EQ(device[i], phi);
#endif
    }
}

TEST(RegionSurfaceTest, FormatsAndOrdersOfCornersGiveTheSameSurface) {
    // Coordinates exact in single precision, as binary STL files hold them
    const Vec lo = {0.25, 0.125, N_DIM == 3 ? 0.375 : -0.5}, hi = {0.75, 0.625, N_DIM == 3 ? 0.875 : 0.5};
    std::vector<Tri> mixed = cube(lo, hi), inverted = cube(lo, hi);
    for (size_t t = 0; t < mixed.size(); t += 3) std::swap(mixed[t][1], mixed[t][2]);
    for (Tri & t : inverted) std::swap(t[1], t[2]);
    const region::Region reference = read(stl_region(file("ref.stl", stl_ascii(cube(lo, hi)))));
    const std::vector<std::string> others = {
        stl_region(file("binary.stl", stl_binary(cube(lo, hi)))),
        stl_region(file("mixed.stl", stl_ascii(mixed))),
        stl_region(file("inverted.stl", stl_ascii(inverted))),
        "{type = \"obj\", file = \"" + file("quads.obj", obj_cube(lo, hi)) + "\", units = \"m\"}"};
    const auto points = random_points(2000, 0.0, 1.0, 2);
    for (const std::string & table : others) {
        const region::Region r = read(table);
        for (const Vec & x : points) {
            const double phi = reference.distance(x.data());
            if (std::abs(phi) < 1e-12) continue;
            EXPECT_NEAR(r.distance(x.data()), phi, 1e-15) << table;
        }
    }
}

TEST(RegionSurfaceTest, UnitsAndTransformsPlaceTheSurface) {
    // The box in millimeters, scaled, rotated and moved, against the box primitive given the same transforms in m
    std::vector<Tri> mm = cube(LO, HI);
    for (Tri & t : mm) {
        for (Vec & x : t) {
            for (double & v : x) v *= 1000.0;
        }
    }
    const std::string transforms = std::string(", scale = 1.5, pivot = [0.4, 0.5") + (N_DIM == 3 ? ", 0.4]" : "]") +
                                   (N_DIM == 3 ? ", rotate = {axis = [1, 2, 3], angle = 35}" : ", rotate = 35") +
                                   ", translate = [0.05, -0.1" + (N_DIM == 3 ? ", 0.02]" : "]");
    const region::Region stl = read("{type = \"stl\", file = \"" + file("mm.stl", stl_ascii(mm)) +
                                    "\", units = \"mm\"" + transforms + "}");
    const region::Region box = read(box_region(transforms));
    for (const Vec & x : random_points(3000, -0.3, 1.3, 3)) {
        EXPECT_NEAR(stl.distance(x.data()), box.distance(x.data()), 1e-12);
    }
}

TEST(RegionSurfaceTest, CavitiesAndSeparateBodiesAreTheirsWhateverTheFilesOrientation) {
    // A box with a box-shaped cavity, both facing out in the file, and a second body beside it
    const double z0 = N_DIM == 3 ? 0.1 : -1.0, z1 = N_DIM == 3 ? 0.6 : 1.0;
    const auto tris = joined(joined(cube({0.1, 0.1, z0}, {0.6, 0.6, z1}), cube({0.25, 0.3, N_DIM == 3 ? 0.25 : -0.5},
                                                                              {0.45, 0.4, N_DIM == 3 ? 0.4 : 0.5})),
                             cube({0.7, 0.2, z0}, {0.9, 0.8, z1}));
    const region::Region stl = read(stl_region(file("cavity.stl", stl_ascii(tris))));
    EXPECT_NE(stl.summary().find("2 bodies, 1 cavity"), std::string::npos) << stl.summary();
    const auto inside = [&](const Vec & x) {
        const auto in = [&](const Vec & lo, const Vec & hi) {
            for (size_t i = 0; i < size_t(N_DIM); i++) {
                if (x[i] <= lo[i] || x[i] >= hi[i]) return false;
            }
            return true;
        };
        return (in({0.1, 0.1, 0.1}, {0.6, 0.6, 0.6}) && !in({0.25, 0.3, 0.25}, {0.45, 0.4, 0.4})) ||
               in({0.7, 0.2, 0.1}, {0.9, 0.8, 0.6});
    };
    for (const Vec & x : random_points(4000, 0.0, 1.0, 4)) {
        const double phi = stl.distance(x.data());
        if (std::abs(phi) < 1e-12) continue;
        EXPECT_EQ(phi < 0.0, inside(x)) << x[0] << " " << x[1] << " " << x[2];
    }
    const double volume = N_DIM == 3 ? 0.125 - 0.003 + 0.06 : 0.25 - 0.02 + 0.12;
    const double surface = N_DIM == 3 ? 1.5 + 0.13 + 1.04 : 2.0 + 0.6 + 1.6;
    // The cavity's surface faces into it, so that it counts against the volume
    const region::Measures m = region::measures(stl, false);
    EXPECT_EQ(m.tolerance, 0.0);
    EXPECT_NEAR(m.volume, volume, 1e-14);
    EXPECT_NEAR(m.surface, surface, 1e-13);
}

TEST(RegionSurfaceTest, TheHierarchyFindsTheClosestTriangle) {
    // A convex polyhedron: inside behind every face's plane, and the distance the least over all triangles
    const Vec c = {0.48, 0.52, N_DIM == 3 ? 0.47 : 0.0};
    const std::vector<Tri> tris = icosphere(3, 0.37, c);
    ASSERT_EQ(tris.size(), 1280u);
    const region::Region stl = read(stl_region(file("sphere.stl", stl_ascii(tris))));
    for (const Vec & x : random_points(2000, -0.1, 1.1, 5)) {
        double least = 1e300;
        bool inside = true;
        for (const Tri & t : tris) {
            double flat[27] = {};
            for (size_t k = 0; k < 3; k++) {
                for (size_t i = 0; i < 3; i++) flat[3 * k + i] = t[k][i];
            }
            double closest[3];
            int feature = 0;
            least = std::min(least, region::detail::to_triangle(flat, x.data(), closest, feature));
            const Vec u = {t[1][0] - t[0][0], t[1][1] - t[0][1], t[1][2] - t[0][2]};
            const Vec v = {t[2][0] - t[0][0], t[2][1] - t[0][1], t[2][2] - t[0][2]};
            const Vec n = {u[1] * v[2] - u[2] * v[1], u[2] * v[0] - u[0] * v[2], u[0] * v[1] - u[1] * v[0]};
            inside = inside && n[0] * (x[0] - t[0][0]) + n[1] * (x[1] - t[0][1]) + n[2] * (x[2] - t[0][2]) < 0.0;
        }
        const double phi = stl.distance(x.data());
        EXPECT_NEAR(std::abs(phi), std::sqrt(least), 1e-15);
        if (std::sqrt(least) > 1e-12) {
            EXPECT_EQ(phi < 0.0, inside);
        }
    }
}

TEST(RegionSurfaceTest, MeasuresAreExact) {
    const region::Region stl = read(stl_region(file("measured.stl", stl_ascii(cube(LO, HI))), ", scale = 2"));
    const double a = 2.0 * (HI[0] - LO[0]), b = 2.0 * (HI[1] - LO[1]), c = 2.0 * (HI[2] - LO[2]);
    const region::Measures m = region::measures(stl, false);
    EXPECT_EQ(m.tolerance, 0.0);
    // 2D: the section z = 0, a rectangle
    EXPECT_NEAR(m.volume, N_DIM == 3 ? a * b * c : a * b, 1e-14);
    EXPECT_NEAR(m.surface, N_DIM == 3 ? 2.0 * (a * b + b * c + c * a) : 2.0 * (a + b), 1e-13);
}

TEST(RegionSurfaceTest, CellFractionsAreTheBoxes) {
    auto mesh = std::make_shared<Mesh>();
    if constexpr (N_DIM == 2) {
        mesh->init_file(write_temp_shared("mallard_region_surface.msh", jittered_mixed_mesh(8)));
    } else {
        mesh->init_file(write_temp_shared("mallard_region_surface.msh", jittered_periodic_mesh_3d(5, 0.2)));
    }
    const std::vector<double> stl = region::cell_fractions(*mesh, read(stl_region(file("cells.stl", stl_ascii(cube(LO, HI))))));
    // Surfaces are sampled, not clipped as the box's planes are: the box's distance sampled the same way
    const region::Region box = read(box_region());
    const region::sampler::Shape<HostDistance> shape{HostDistance{box.host()}, 0.0, true};
    double total = 0.0;
    for (uint32_t i = 0; i < mesh->n_cells; i++) {
        double inside = 0.0, measure = 0.0;
        region::sampler::straight_cell(shape, false, mesh->h_n_nodes_of_cell(i), HostNodes{*mesh, i}, inside, measure);
        EXPECT_NEAR(stl[i], inside / measure, 1e-12) << i;
        total += stl[i];
    }
    EXPECT_GT(total, 1.0);
}

TEST(RegionSurfaceTest, ItsSurfaceIsExactAndFacesOut) {
    std::vector<std::shared_ptr<Mesh>> meshes = {std::make_shared<Mesh>()};
#if N_DIM == 2
    meshes[0]->init_file(write_temp_shared("mallard_region_surface_elements.msh", jittered_mixed_mesh(10)));
#else
    meshes[0]->init_cart_3d(6, 6, 6, 1.0, 1.0, 1.0, MeshType::CARTESIAN_TET);
    // Hexahedra with warped faces, whose cells must still hold every point near them
    meshes.push_back(std::make_shared<Mesh>());
    meshes[1]->init_file(write_temp_shared("mallard_region_surface_hex.msh", jittered_periodic_mesh_3d(5, 0.2)));
#endif
    const region::Region r = read(stl_region(file("elements.stl", stl_ascii(cube(LO, HI))), ", filled = false"));
    const double a = HI[0] - LO[0], b = HI[1] - LO[1], c = HI[2] - LO[2];
    for (const auto & mesh : meshes) {
        const region::Elements e = region::elements(*mesh, r);
        EXPECT_TRUE(e.exact);
        double measure = 0.0, flux = 0.0;
        for (size_t k = 0; k < e.size(); k++) {
            measure += e.measure[k];
            for (size_t i = 0; i < 3; i++) flux += (e.center[k][i] - 0.5) * e.direction[k][i] * e.measure[k];
        }
        // Facing out: the flux of x - x0 is N_DIM times the enclosed measure
        EXPECT_NEAR(measure, N_DIM == 3 ? 2.0 * (a * b + b * c + c * a) : 2.0 * (a + b), 1e-12);
        EXPECT_NEAR(flux, N_DIM == 3 ? 3.0 * a * b * c : 2.0 * a * b, 1e-12);
    }
}

TEST(RegionSurfaceTest, BrokenFilesAndKeysAreErrorsThatSayWhatToFix) {
    const Vec lo = {0.2, 0.2, -0.5}, hi = {0.4, 0.4, 0.5};
    const std::string good = file("good.stl", stl_ascii(cube(lo, hi)));
    struct Bad {
        std::string table, says;
    };
    std::vector<Tri> open = cube(lo, hi);
    open.pop_back();
    // Sharing an edge, a corner, or overlapping
    const auto edge = joined(cube(lo, hi), cube({0.4, 0.4, -0.5}, {0.6, 0.6, 0.5}));
    const auto corner = joined(cube({0.2, 0.2, -0.5}, {0.4, 0.4, 0.0}), cube({0.4, 0.4, 0.0}, {0.6, 0.6, 0.5}));
    const auto overlap = joined(cube(lo, hi), cube({0.3, 0.3, -0.4}, {0.5, 0.5, 0.4}));
    const Tri flat = {Vec{0.2, 0.2, 0.0}, Vec{0.4, 0.2, 0.1}, Vec{0.2, 0.4, -0.1}};
    const std::vector<Tri> sheet = {flat, {flat[0], flat[2], flat[1]}};
    const std::vector<Bad> bad = {
        {"{type = \"stl\", file = \"" + good + "\"}", "needs units, the unit of length of the file's coordinates"},
        {"{type = \"stl\", file = \"" + good + "\", units = \"kg\"}", "\"kg\" is not a unit of length"},
        {"{type = \"stl\", units = \"m\"}", "needs file"},
        {stl_region(good + ".missing"), "cannot open the surface file"},
        {stl_region(file("open.stl", stl_ascii(open))), "is not closed: it has 3 edges of one triangle only"},
        {stl_region(file("edge.stl", stl_ascii(edge))), "it has 1 edge of more than two triangles"},
        {stl_region(file("corner.stl", stl_ascii(corner))), "do not form one fan"},
        {stl_region(file("overlap.stl", stl_ascii(overlap))), "closed surfaces that cross or touch each other"},
        {stl_region(file("sheet.stl", stl_ascii(sheet))), "encloses no volume"},
        {stl_region(file("garbage.stl", "not a surface\n")), "is not an STL file"},
        {"{type = \"obj\", units = \"m\", file = \"" +
             file("concave.obj", "v 0 0 0\nv 2 0 0\nv 2 1 0\nv 1 0.2 0\nv 0 1 0\nf 1 2 3 4 5\n") + "\"}",
         "that is not convex"},
        {"{type = \"obj\", units = \"m\", file = \"" + file("index.obj", "v 0 0 0\nv 1 0 0\nf 1 2 3\n") + "\"}",
         "\"3\" is not the index of a vertex before it"},
    };
    for (const Bad & b : bad) {
        const std::string message = error_of(b.table);
        EXPECT_NE(message.find(b.says), std::string::npos) << b.table << "\n" << message;
        EXPECT_NE(message.find("geometry.md#regions-from-cad-surfaces"), std::string::npos) << message;
    }
    if constexpr (N_DIM == 2) {
        // The section z = 0 must cut the surface: a body extruded from z = 0 does not
        const std::string above = file("above.stl", stl_ascii(cube({0.2, 0.2, 0.0}, {0.4, 0.4, 1.0})));
        EXPECT_NE(error_of(stl_region(above)).find("has faces in the plane z = 0"), std::string::npos);
        EXPECT_NE(error_of(stl_region(above, ", translate = [0, 0, 0.5]")).find("spans z = 0.5 to 1.5 m"), std::string::npos);
        EXPECT_EQ(error_of(stl_region(above, ", translate = [0, 0, -0.5]")), "");
    }
}

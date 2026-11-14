/**
 * @file remap_test.cpp
 * @author Matthew Bonanni (mbonanni001@gmail.com)
 * @brief Remap between meshes: polynomial exactness, conservation, round trips
 *        and convergence, in 2D and 3D.
 * @version 0.1
 * @date 2026-10-15
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 *
 */

#include <gtest/gtest.h>

#include <cmath>
#include <functional>
#include <random>
#include <string>
#include <vector>

#include "box_tree.h"
#include "mesh.h"
#include "overlap.h"
#include "remap.h"
#include "test_fixtures.h"
#include "verification.h"

namespace {

using Field = std::function<double(const overlap::Point &)>;

std::string mesh_table(const std::string & type, uint32_t n, const std::string & extra = "") {
    std::string s = "[mesh]\ntype = \"" + type + "\"\nNx = " + std::to_string(n) + "\nNy = " +
                    std::to_string(n + 1) + "\nLx = 1.0\nLy = 0.8\n";
    if (N_DIM == 3) s += "Nz = " + std::to_string(n) + "\nLz = 0.6\n";
    return s + extra;
}

std::shared_ptr<Mesh> serial_mesh(const std::string & table) {
    auto mesh = std::make_shared<Mesh>();
    mesh->init(parse_toml(table));
    return mesh;
}

/** @brief Averages of f over the first n cells of a mesh. */
std::vector<double> averages(const Mesh & mesh, uint32_t n, const Field & f) {
    const overlap::Rule & rule = overlap::simplex_rule(N_DIM == 2 ? 10 : 6);
    std::vector<double> out(n);
    for (uint32_t c = 0; c < n; c++) {
        const overlap::Region r = overlap::region(overlap::cell_nodes(mesh, c));
        double sum = 0.0;
        for (const auto & s : r.simplices) {
            for (size_t k = 0; k < rule.weights.size(); k++) sum += overlap::volume(s) * rule.weights[k] * f(overlap::map(s, rule.points[k]));
        }
        out[c] = sum / r.volume;
    }
    return out;
}

/** @brief A polynomial of total degree p with random coefficients. */
Field random_polynomial(uint32_t p, uint32_t seed) {
    std::mt19937 gen(seed);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    std::vector<std::array<uint32_t, 3>> e;
    std::vector<double> a;
    for (uint32_t i = 0; i <= p; i++) {
        for (uint32_t j = 0; i + j <= p; j++) {
            for (uint32_t k = 0; i + j + k <= p && (N_DIM == 3 || k == 0); k++) {
                e.push_back({i, j, k});
                a.push_back(dist(gen));
            }
        }
    }
    return [e, a](const overlap::Point & x) {
        double s = 0.0;
        for (size_t l = 0; l < e.size(); l++) {
            s += a[l] * std::pow(x[0], e[l][0]) * std::pow(x[1], e[l][1]) * (N_DIM == 3 ? std::pow(x[N_DIM - 1], e[l][2]) : 1.0);
        }
        return s;
    };
}

/** @brief The source mesh of table with fields fs, remapped to the first n cells of target. */
std::vector<std::vector<double>> remap_fields(const std::string & table, const std::vector<Field> & fs,
                                              const Mesh & target, const remap::Options & options,
                                              remap::Report & report) {
    const std::shared_ptr<Mesh> source = remap::source_mesh(parse_toml(table), options.degree);
    std::vector<std::vector<double>> values;
    for (const auto & f : fs) values.push_back(averages(*source, source->n_cells, f));
    const remap::Source field(source, values, std::vector<int>(fs.size(), -1), options);
    return field.to(target, target.n_cells, report);
}

double max_error(const std::vector<double> & a, const std::vector<double> & b) {
    double m = 0.0;
    for (size_t i = 0; i < a.size(); i++) m = std::max(m, std::abs(a[i] - b[i]));
    return m;
}

const char * SOURCE_TYPE = "cartesian";
const char * TARGET_TYPE = N_DIM == 2 ? "cartesian_tri" : "cartesian_tet";

} // namespace

TEST(RemapTest, PolynomialsUpToTheDegreeAreReproducedExactly) {
    // Quads to triangles (hexahedra to tetrahedra) of unrelated sizes
    const auto target = serial_mesh(mesh_table(TARGET_TYPE, N_DIM == 2 ? 13 : 5));
    for (uint8_t p = 1; p <= (N_DIM == 2 ? 4 : 3); p++) {
        remap::Options options;
        options.degree = p;
        options.limit = false;
        remap::Report report;
        const Field exact = random_polynomial(p, 7u + p), higher = random_polynomial(p + 1u, 11u + p);
        const auto out = remap_fields(mesh_table(SOURCE_TYPE, N_DIM == 2 ? 9 : 4), {exact, higher}, *target,
                                      options, report);
        // Single-precision meshes put the nodes of the two meshes' common boundary apart by rounding
        EXPECT_LT(max_error(out[0], averages(*target, target->n_cells, exact)), precision_tol<double>(1e-11, 1e-5))
            << "degree " << int(p);
        // One degree more is not: the test can see a lower order
        EXPECT_GT(max_error(out[1], averages(*target, target->n_cells, higher)), 1e-6) << "degree " << int(p);
        EXPECT_EQ(report.n_reduced, 0u);
    }
}

TEST(RemapTest, IntegralsAreConservedToRounding) {
    // A discontinuous field, limited, onto a mesh of another topology and size
    const Field step = [](const overlap::Point & x) {
        return (x[0] + 0.3 * x[1] < 0.55 ? 1.0 : 0.125) + 0.1 * std::sin(9.0 * x[1]);
    };
    for (bool limit : {false, true}) {
        remap::Options options;
        options.degree = 3;
        options.limit = limit;
        remap::Report report;
        const auto target = serial_mesh(mesh_table(TARGET_TYPE, N_DIM == 2 ? 23 : 6));
        const auto out = remap_fields(mesh_table(N_DIM == 2 ? "cartesian" : "cartesian_prism", N_DIM == 2 ? 16 : 5),
                                      {step}, *target, options, report);
        EXPECT_NEAR(report.target_total[0], report.source_total[0],
                    precision_tol<double>(1e-14, 1e-5) * report.source_magnitude[0]);
        EXPECT_EQ(report.n_partial + report.n_nearest, 0u);
        if (!limit) continue;
        // Bounded by the source averages
        EXPECT_GT(report.n_limited, 0u);
        for (double u : out[0]) {
            EXPECT_LE(u, 1.1 + precision_tol<double>(1e-12, 1e-5));
            EXPECT_GE(u, 0.025 - precision_tol<double>(1e-12, 1e-5));
        }
    }
}

TEST(RemapTest, RefiningThenCoarseningRecoversTheCoarseAverages) {
    const Field f = [](const overlap::Point & x) { return std::exp(x[0]) * std::cos(3.0 * x[1]) + (x[0] > 0.5); };
    const uint32_t n = N_DIM == 2 ? 8 : 3;
    const std::string coarse = mesh_table(SOURCE_TYPE, n);
    const std::string fine = "[mesh]\ntype = \"" + std::string(SOURCE_TYPE) + "\"\nNx = " + std::to_string(2 * n) +
                             "\nNy = " + std::to_string(2 * (n + 1)) + "\nLx = 1.0\nLy = 0.8\n" +
                             (N_DIM == 3 ? "Nz = " + std::to_string(2 * n) + "\nLz = 0.6\n" : "");
    remap::Options options;
    options.degree = 2;
    remap::Report report;
    const auto fine_mesh = remap::source_mesh(parse_toml(fine), options.degree);
    const auto coarse_mesh = serial_mesh(coarse);
    const auto on_fine = remap_fields(coarse, {f}, *fine_mesh, options, report);
    const remap::Source back(fine_mesh, on_fine, {-1}, options);
    const auto round_trip = back.to(*coarse_mesh, coarse_mesh->n_cells, report);
    EXPECT_LT(max_error(round_trip[0], averages(*coarse_mesh, coarse_mesh->n_cells, f)), precision_tol<double>(1e-13, 1e-5));
}

TEST(RemapTest, SmoothPeriodicFieldsConvergeAtTheDegreePlusOne) {
    // The stencils of cells on the seams wrap around
    const Field f = [](const overlap::Point & x) {
        return std::sin(2.0 * M_PI * x[0]) * std::cos(2.0 * M_PI * x[1] / 0.8) +
               (N_DIM == 3 ? 0.5 * std::sin(2.0 * M_PI * x[N_DIM - 1] / 0.6) : 0.0);
    };
    const std::string periodic = N_DIM == 2 ? "[[periodic]]\nzones = [\"left\", \"right\"]\n[[periodic]]\nzones = [\"bottom\", \"top\"]\n" : "[[periodic]]\nzones = [\"left\", \"right\"]\n[[periodic]]\nzones = [\"bottom\", \"top\"]\n[[periodic]]\nzones = [\"back\", \"front\"]\n";
    const auto target = serial_mesh(mesh_table(TARGET_TYPE, N_DIM == 2 ? 37 : 7));
    const auto exact = averages(*target, target->n_cells, f);
    for (uint8_t p = 1; p <= (N_DIM == 2 ? 3 : 2); p++) {
        std::vector<double> errors(2);
        for (uint32_t k = 0; k < 2; k++) {
            remap::Options options;
            options.degree = p;
            options.limit = false;
            remap::Report report;
            const uint32_t n = uint32_t(N_DIM == 2 ? 10 : 5) << k;
            errors[k] = max_error(remap_fields(mesh_table(SOURCE_TYPE, n, periodic), {f}, *target, options, report)[0],
                                  exact);
        }
        const double n = N_DIM == 2 ? 10 : 5;
        verification::expect_order({"remap", "degree " + std::to_string(p), verification::cells_of(SOURCE_TYPE),
                                    "max error of the target averages", {n, 2 * n}, errors, p + 1.0, p + 0.6});
    }
}

TEST(RemapTest, TargetCellsOutsideTheSourceMeshAreReported) {
    const auto target = serial_mesh(mesh_table(TARGET_TYPE, N_DIM == 2 ? 10 : 4) + "");
    // A source box shorter in x: the cells beyond it take the nearest source cells
    std::string table = mesh_table(SOURCE_TYPE, N_DIM == 2 ? 8 : 4);
    table.replace(table.find("Lx = 1.0"), 8, "Lx = 0.73");
    remap::Options options;
    remap::Report report;
    const auto out = remap_fields(table, {[](const overlap::Point &) { return 2.5; }}, *target, options, report);
    EXPECT_GT(report.n_partial, 0u);
    EXPECT_GT(report.n_nearest, 0u);
    for (double u : out[0]) EXPECT_NEAR(u, 2.5, precision_tol<double>(1e-12, 1e-5));
}

TEST(RemapTest, BoxTreeFindsEveryOverlapAndTheNearestItem) {
    std::mt19937 gen(3);
    std::uniform_real_distribution<double> u(0.0, 1.0);
    std::vector<Box> boxes(500);
    for (auto & b : boxes) {
        double x[N_DIM], y[N_DIM];
        for (size_t d = 0; d < N_DIM; d++) {
            x[d] = u(gen);
            y[d] = x[d] + 0.05 * u(gen);
        }
        b.add(x);
        b.add(y);
    }
    const BoxTree tree(boxes);
    for (int q = 0; q < 50; q++) {
        Box query;
        double x[N_DIM], y[N_DIM];
        for (size_t d = 0; d < N_DIM; d++) {
            x[d] = u(gen);
            y[d] = x[d] + 0.1 * u(gen);
        }
        query.add(x);
        query.add(y);
        std::vector<uint32_t> found, expected;
        tree.overlapping(query, found);
        for (uint32_t i = 0; i < boxes.size(); i++) {
            if (boxes[i].overlaps(query)) expected.push_back(i);
        }
        EXPECT_EQ(found, expected);
        double d2 = 0.0;
        const uint32_t near = tree.nearest(x, [&](uint32_t i) { return boxes[i].distance2(x); }, d2);
        double best = 1e300;
        for (const auto & b : boxes) best = std::min(best, b.distance2(x));
        EXPECT_EQ(boxes[near].distance2(x), best);
    }
}

#if Mallard_DIM == 3
TEST(RemapTest, WarpedHexahedraAreTheMeshsTetrahedra) {
    // A hexahedron with a twisted top face: not convex, so it is clipped as the
    // tetrahedra that define its volume
    std::vector<overlap::Point> nodes = {{0, 0, 0}, {1, 0, 0}, {1, 1, 0}, {0, 1, 0},
                                         {0, 0, 1}, {1, 0, 1.3}, {1, 1, 1}, {0, 1, 0.8}};
    const overlap::Region r = overlap::region(nodes);
    EXPECT_EQ(r.pieces.size(), r.simplices.size());
    std::vector<std::array<double, 3>> p(nodes.begin(), nodes.end());
    std::vector<std::array<std::array<double, 3>, 4>> tets;
    cell_tetrahedra(p, tets);
    double volume = 0.0;
    for (const auto & t : tets) volume += overlap::volume({t[0], t[1], t[2], t[3]});
    EXPECT_NEAR(r.volume, volume, 1e-15);
    // Cut by the planes x = 0.4 and x = 0.4: the halves sum to the whole
    const overlap::Region left = overlap::region({{-1, -1, -1}, {0.4, -1, -1}, {0.4, 2, -1}, {-1, 2, -1},
                                                  {-1, -1, 2}, {0.4, -1, 2}, {0.4, 2, 2}, {-1, 2, 2}});
    const overlap::Region right = overlap::region({{0.4, -1, -1}, {2, -1, -1}, {2, 2, -1}, {0.4, 2, -1},
                                                   {0.4, -1, 2}, {2, -1, 2}, {2, 2, 2}, {0.4, 2, 2}});
    std::vector<overlap::Simplex> a, b;
    overlap::intersect(r.simplices, left, a);
    overlap::intersect(r.simplices, right, b);
    double va = 0.0, vb = 0.0;
    for (const auto & s : a) va += overlap::volume(s);
    for (const auto & s : b) vb += overlap::volume(s);
    EXPECT_NEAR(va + vb, volume, 1e-15);
    EXPECT_GT(va, 0.35);
    EXPECT_GT(vb, 0.5);
}
#endif

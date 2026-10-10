/**
 * @file curved_test.cpp
 * @author Matthew Bonanni (mbonanni001@gmail.com)
 * @brief Tests of curved (high-order) boundaries in 2D: geometry against exact
 *        regions, closure, high-order Gmsh lines, and TENO reconstruction order.
 * @version 0.1
 * @date 2026-10-09
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 *
 */

#include <gtest/gtest.h>
#include <Kokkos_Core.hpp>

#include <cmath>
#include <filesystem>
#include <functional>
#include <memory>
#include <sstream>
#include <string>
#include <tuple>
#include <vector>

#include "curved.h"
#include "face_reconstruction.h"
#include "gmsh_fixtures.h"
#include "input.h"
#include "mesh.h"
#include "physics.h"
#include "solver.h"
#include "test_fixtures.h"
#include "test_utils.h"

namespace {

constexpr double R_IN = 1.0, R_OUT = 1.384;
constexpr rtype GAMMA = 1.4;

const char * SHAPES = R"(
[[mesh.curved]]
zone = "inner"
shape = "circle"
center = [0.0, 0.0]
radius = 1.0

[[mesh.curved]]
zone = "outer"
shape = "circle"
center = [0.0, 0.0]
radius = 1.384
)";

/** @brief Nodes and cells of an O-grid of the annulus (n_r rings of n_t cells), nodes on the circles. */
struct Annulus {
    std::vector<std::array<rtype, N_DIM>> nodes;
    std::vector<std::vector<uint32_t>> cells;
    std::vector<Mesh::BoundaryFace> faces;
    std::vector<curved::SurfaceFace> high_order;  // with order > 1: the walls as Lagrange lines
};

Annulus annulus(uint32_t n_r, uint32_t n_t, bool triangles = false, int order = 1) {
    Annulus a;
    auto id = [&](uint32_t i, uint32_t j) { return j * n_t + (i % n_t); };
    for (uint32_t j = 0; j <= n_r; j++) {
        const double r = R_IN + (R_OUT - R_IN) * j / n_r;
        for (uint32_t i = 0; i < n_t; i++) {
            const double t = 2.0 * M_PI * i / n_t;
            a.nodes.push_back({rtype(r * std::cos(t)), rtype(r * std::sin(t))});
        }
    }
    for (uint32_t j = 0; j < n_r; j++) {
        for (uint32_t i = 0; i < n_t; i++) {
            const uint32_t q[4] = {id(i, j), id(i + 1, j), id(i + 1, j + 1), id(i, j + 1)};
            if (!triangles) {
                a.cells.push_back({q[0], q[1], q[2], q[3]});
            } else if ((i + j) % 2 == 0) {
                a.cells.push_back({q[0], q[1], q[2]});
                a.cells.push_back({q[0], q[2], q[3]});
            } else {
                a.cells.push_back({q[0], q[1], q[3]});
                a.cells.push_back({q[1], q[2], q[3]});
            }
        }
    }
    for (uint32_t i = 0; i < n_t; i++) {
        for (const auto & [j, zone, r] : {std::tuple{0u, "inner", R_IN}, std::tuple{n_r, "outer", R_OUT}}) {
            a.faces.push_back({{id(i, j), id(i + 1, j)}, zone});
            if (order < 2) continue;
            curved::SurfaceFace face;
            face.corners = {id(i, j), id(i + 1, j)};
            for (uint32_t k : {id(i, j), id(i + 1, j)}) {
                face.nodes.push_back({double(a.nodes[k][0]), double(a.nodes[k][1]), 0.0});
            }
            for (int k = 1; k < order; k++) {
                const double t = 2.0 * M_PI * (i + double(k) / order) / n_t;
                face.nodes.push_back({r * std::cos(t), r * std::sin(t), 0.0});
            }
            a.high_order.push_back(face);
        }
    }
    return a;
}

/**
 * @brief The annulus mesh with straight walls ("straight"), walls projected
 *        onto the circles ("curved"), or high-order walls of the given order.
 */
std::shared_ptr<Mesh> annulus_mesh(uint32_t n_r, uint32_t n_t, const std::string & geometry, bool triangles = false,
                                   int order = 2) {
    const Annulus a = annulus(n_r, n_t, triangles, geometry == "high_order" ? order : 1);
    auto mesh = std::make_shared<Mesh>();
    mesh->init_from_connectivity(a.nodes, a.cells, a.faces);
    mesh->high_order_faces = a.high_order;
    const std::string input = "[mesh]\n" + std::string(geometry == "curved" ? SHAPES : "");
    mesh->apply_curved(curved::surfaces_of_mesh(*mesh, parse_toml(input)));
    mesh->copy_host_to_device();
    return mesh;
}

/** @brief Whether node n lies on a wall circle (of radius R). */
bool on_circle(const Mesh & mesh, uint32_t n, double & R) {
    const double r = std::hypot(double(mesh.h_node_coords(n, 0)), double(mesh.h_node_coords(n, 1)));
    for (double c : {R_IN, R_OUT}) {
        if (std::abs(r - c) < precision_tol<double>(1e-12, 1e-6)) {
            R = c;
            return true;
        }
    }
    return false;
}

/**
 * @brief int_cell f dA = closed-integral F dy (Green), F = int f dx, over the
 *        exact region of a cell: its wall faces are circular arcs.
 */
double exact_integral(const Mesh & mesh, uint32_t c, const std::function<double(double, double)> & F) {
    const double g[8] = {-0.9602898564975363, -0.7966664774136267, -0.5255324099163290, -0.1834346424956498,
                         0.1834346424956498, 0.5255324099163290, 0.7966664774136267, 0.9602898564975363};
    const double w[8] = {0.1012285362903763, 0.2223810344533745, 0.3137066458778873, 0.3626837833783620,
                         0.3626837833783620, 0.3137066458778873, 0.2223810344533745, 0.1012285362903763};
    double sum = 0.0;
    for (uint32_t k = 0; k < mesh.h_n_faces_of_cell(c); k++) {
        const uint32_t f = mesh.h_face_of_cell(c, k);
        uint32_t a = mesh.h_node_of_face(f, 0), b = mesh.h_node_of_face(f, 1);
        if (mesh.h_cells_of_face(f, 0) != int32_t(c)) std::swap(a, b);
        const double xa = double(mesh.h_node_coords(a, 0)), ya = double(mesh.h_node_coords(a, 1));
        const double xb = double(mesh.h_node_coords(b, 0)), yb = double(mesh.h_node_coords(b, 1));
        double R;
        const bool arc = mesh.h_cells_of_face(f, 1) < 0 && on_circle(mesh, a, R);
        // Arcs in 8 pieces of Gauss-8: the integrals are exact to round-off
        const int pieces = arc ? 8 : 1;
        double ta = std::atan2(ya, xa), tb = std::atan2(yb, xb);
        if (tb - ta > M_PI) tb -= 2.0 * M_PI;
        if (ta - tb > M_PI) tb += 2.0 * M_PI;
        for (int p = 0; p < pieces; p++) {
            for (int q = 0; q < 8; q++) {
                const double s = (p + 0.5 * (g[q] + 1.0)) / pieces;
                double x, y, dy;
                if (arc) {
                    const double t = ta + s * (tb - ta);
                    x = R * std::cos(t);
                    y = R * std::sin(t);
                    dy = R * std::cos(t) * (tb - ta);
                } else {
                    x = xa + s * (xb - xa);
                    y = ya + s * (yb - ya);
                    dy = yb - ya;
                }
                sum += 0.5 * w[q] / pieces * F(x, y) * dy;
            }
        }
    }
    return sum;
}

// A smooth state: conservatives U_k = c_k + a_k sin(al_k x + b_k) cos(ga_k y + d_k), with x-antiderivatives
struct Field {
    double c[4] = {1.0, 0.1, -0.05, 2.6};
    double a[4] = {0.2, 0.1, 0.1, 0.1};
    double al[4] = {2.0, 1.5, 2.5, 1.0};
    double b[4] = {1.0, 0.3, -0.7, 0.2};
    double ga[4] = {3.0, 2.0, 1.0, 2.5};
    double d[4] = {-0.5, 0.4, 0.9, -1.1};

    double value(int k, double x, double y) const { return c[k] + a[k] * std::sin(al[k] * x + b[k]) * std::cos(ga[k] * y + d[k]); }
    double antiderivative(int k, double x, double y) const {
        return c[k] * x - a[k] / al[k] * std::cos(al[k] * x + b[k]) * std::cos(ga[k] * y + d[k]);
    }
};

/** @brief Exact cell averages of the field over the exact (curved) cells. */
Kokkos::View<rtype *[N_CONSERVATIVE]>::host_mirror_type exact_averages(const Mesh & mesh, const Field & field) {
    Kokkos::View<rtype *[N_CONSERVATIVE]>::host_mirror_type avg("avg", mesh.n_cells);
    for (uint32_t c = 0; c < mesh.n_cells; c++) {
        const double area = exact_integral(mesh, c, [](double x, double) { return x; });
        for (int k = 0; k < 4; k++) {
            avg(c, k) = rtype(exact_integral(mesh, c, [&](double x, double y) { return field.antiderivative(k, x, y); }) / area);
        }
    }
    return avg;
}

/**
 * @brief Max error of the reconstructed face density at the faces'
 *        quadrature points, from exact averages over the exact curved cells.
 */
double curved_reconstruction_error(uint32_t n_r, const std::string & geometry, int order, bool triangles = false,
                                   const std::string & extra = "") {
    auto mesh = annulus_mesh(n_r, 16 * n_r, geometry, triangles);
    const Field field;
    // Exact averages over the true cells, whatever geometry the scheme sees
    auto truth = annulus_mesh(n_r, 16 * n_r, "curved", triangles);
    auto avg = exact_averages(*truth, field);
    BoundaryData bd = make_uniform_boundaries(*mesh, BoundaryType::SYMMETRY, GAMMA);
    Euler euler = Euler::from_reference(GAMMA, 1.0, 1.0, 1.0);
    Kokkos::View<rtype *[N_CONSERVATIVE]> W("W", mesh->n_cells);
    auto h_W = Kokkos::create_mirror_view(W);
    for (uint32_t c = 0; c < mesh->n_cells; c++) {
        rtype U[N_CONSERVATIVE], Wc[N_CONSERVATIVE];
        FOR_I_CONSERVATIVE U[i] = avg(c, i);
        euler.compute_W_from_conservatives(Wc, U);
        FOR_I_CONSERVATIVE h_W(c, i) = Wc[i];
    }
    Kokkos::deep_copy(W, h_W);
    auto teno = std::make_unique<TENO>();
    teno->set_mesh(mesh);
    teno->set_boundaries(bd);
    teno->init(parse_toml("type = \"TENO\"\norder = " + std::to_string(order) + "\ncurved_mirrors = false\n" +
                          (extra.empty() ? "curved_wall_degree = " + std::to_string(order - 1) + "\n" : extra)));
    const uint8_t n_quad = teno->n_face_quadrature_points();
    Kokkos::View<rtype **[2][N_CONSERVATIVE]> face_W("face_W", mesh->n_faces, n_quad);
    teno->calc_face_values(W, face_W);
    auto h_face_W = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), face_W);
    auto h_qp = teno->quadrature_face.h_points;
    // The points the scheme reconstructed at: on the curved faces, or on the chords
    Kokkos::View<rtype ***>::host_mirror_type h_points;
    if (mesh->curved_geometry) h_points = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), teno->face_quad_points);
    double err = 0.0;
    for (uint32_t f = 0; f < mesh->n_faces; f++) {
        if (mesh->h_cells_of_face(f, 1) >= 0) continue;
        const uint32_t a = mesh->h_node_of_face(f, 0), b = mesh->h_node_of_face(f, 1);
        for (uint8_t q = 0; q < n_quad; q++) {
            double x, y;
            if (mesh->curved_geometry) {
                x = double(h_points(f, q, 0));
                y = double(h_points(f, q, 1));
            } else {
                const double s = 0.5 * double(h_qp(q, 0));
                x = double(mesh->h_face_coords(f, 0)) + s * (double(mesh->h_node_coords(b, 0)) - double(mesh->h_node_coords(a, 0)));
                y = double(mesh->h_face_coords(f, 1)) + s * (double(mesh->h_node_coords(b, 1)) - double(mesh->h_node_coords(a, 1)));
            }
            err = std::max(err, std::abs(double(h_face_W(f, q, 0, 0)) - field.value(0, x, y)));
        }
    }
    return err;
}

}  // namespace

TEST(CurvedGeometry, CellAreasAndCentroidsMatchTheExactRegions) {
    // Projected walls give the exact annulus: every cell's area and centroid
    // match those of its exact region (circular arcs on the walls)
    for (bool triangles : {false, true}) {
        auto mesh = annulus_mesh(3, 24, "curved", triangles);
        ASSERT_TRUE(mesh->curved_geometry);
        double total = 0.0;
        for (uint32_t c = 0; c < mesh->n_cells; c++) {
            const double area = exact_integral(*mesh, c, [](double x, double) { return x; });
            const double xc = exact_integral(*mesh, c, [](double x, double) { return 0.5 * x * x; }) / area;
            const double yc = exact_integral(*mesh, c, [](double x, double y) { return x * y; }) / area;
            EXPECT_NEAR(double(mesh->h_cell_volume(c)), area, roundoff(1e-14) * 10 * area) << c;
            EXPECT_NEAR(double(mesh->h_cell_coords(c, 0)), xc, roundoff(1e-13)) << c;
            EXPECT_NEAR(double(mesh->h_cell_coords(c, 1)), yc, roundoff(1e-13)) << c;
            total += double(mesh->h_cell_volume(c));
        }
        EXPECT_NEAR(total, M_PI * (R_OUT * R_OUT - R_IN * R_IN), roundoff(1e-12));
    }
}

TEST(CurvedGeometry, StraightWallsAreSecondOrderAndHighOrderLinesConverge) {
    // Total area error: the polygon's falls as h^2; Lagrange lines on the
    // circle (P2, P3) as h^4 (even orders superconverge)
    const double exact = M_PI * (R_OUT * R_OUT - R_IN * R_IN);
    auto area_error = [&](uint32_t n_t, const std::string & geometry, int order) {
        auto mesh = annulus_mesh(2, n_t, geometry, false, order);
        double total = 0.0;
        for (uint32_t c = 0; c < mesh->n_cells; c++) total += double(mesh->h_cell_volume(c));
        return std::abs(total - exact);
    };
    EXPECT_NEAR(std::log2(area_error(32, "straight", 1) / area_error(64, "straight", 1)), 2.0, 0.05);
    for (int order : {2, 3}) {
        const double e1 = area_error(16, "high_order", order), e2 = area_error(32, "high_order", order);
        EXPECT_GT(std::log2(e1 / e2), 3.8) << order;
        EXPECT_LT(e2, 1e-3 * area_error(32, "straight", 1)) << order;
    }
}

TEST(CurvedGeometry, CurvedFaceRulesCloseEveryCell) {
    // The area vectors of every cell's face quadrature sum to zero (free
    // streams are preserved), and curved faces' points lie on the circles
    auto mesh = annulus_mesh(3, 24, "curved", true);
    const curved::Geometry & g = *mesh->curved_geometry;
    const std::vector<std::array<double, 2>> ref = {{-0.7745966692414834, 0.0}, {0.0, 0.0}, {0.7745966692414834, 0.0}};
    const std::vector<double> ref_w = {5.0 / 9.0, 8.0 / 9.0, 5.0 / 9.0};
    std::vector<double> sum(2 * mesh->n_cells, 0.0);
    std::vector<curved::Vec3> x, n;
    std::vector<double> w;
    uint32_t n_curved = 0;
    for (uint32_t f = 0; f < mesh->n_faces; f++) {
        g.face_rule(f, ref, ref_w, x, n, w);
        for (size_t q = 0; q < w.size(); q++) {
            for (int side = 0; side < 2; side++) {
                const int32_t c = mesh->h_cells_of_face(f, side);
                if (c < 0) continue;
                for (int i = 0; i < 2; i++) sum[2 * c + i] += (side == 0 ? 1.0 : -1.0) * w[q] * n[q][i];
            }
            if (g.face_is_curved(f)) {
                const double r = std::hypot(x[q][0], x[q][1]);
                EXPECT_NEAR(std::min(std::abs(r - R_IN), std::abs(r - R_OUT)), 0.0, precision_tol<double>(1e-14, 1e-6));
                // The normal of a circle is radial
                // (up to the shift that closes the cell, by the quadrature error of the area vector)
                EXPECT_NEAR(std::abs(n[q][0] * x[q][1] - n[q][1] * x[q][0]) / r, 0.0, 1e-6);
            }
        }
        n_curved += g.face_is_curved(f);
    }
    EXPECT_EQ(n_curved, 2u * 24u);
    for (double s : sum) EXPECT_NEAR(s, 0.0, 1e-14);
}

TEST(CurvedGeometry, MeshesWithoutCurvedDataAreUnchanged) {
    auto plain = std::make_shared<Mesh>();
    const Annulus a = annulus(3, 24);
    plain->init_from_connectivity(a.nodes, a.cells, a.faces);
    auto off = annulus_mesh(3, 24, "straight");
    EXPECT_FALSE(off->curved_geometry);
    for (uint32_t c = 0; c < plain->n_cells; c++) {
        EXPECT_EQ(plain->h_cell_volume(c), off->h_cell_volume(c));
        EXPECT_EQ(plain->h_cell_coords(c, 0), off->h_cell_coords(c, 0));
    }
    // A zone without a matching shape is an input error
    auto bad = std::make_shared<Mesh>();
    bad->init_from_connectivity(a.nodes, a.cells, a.faces);
    EXPECT_THROW(curved::surfaces_of_mesh(*bad, parse_toml("[mesh]\n[[mesh.curved]]\nzone = \"wall\"\nshape = \"circle\"\n"
                                                           "center = [0.0, 0.0]\nradius = 1.0\n")),
                 InputError);
    // So is a shape that misses the zone's nodes
    EXPECT_THROW(bad->apply_curved(curved::surfaces_of_mesh(*bad, parse_toml("[mesh]\n[[mesh.curved]]\nzone = \"inner\"\n"
                                                                              "shape = \"circle\"\ncenter = [0.3, 0.0]\n"
                                                                              "radius = 1.0\n"))),
                 InputError);
}

TEST(CurvedGeometry, GmshHighOrderLinesCurveTheWalls) {
    // A Gmsh file with quadratic boundary lines (line3) and quadratic cells
    // (quad9): only the corners stay nodes, and the walls follow the lines
    const uint32_t n_r = 2, n_t = 16;
    std::ostringstream s;
    s.precision(17);
    s << "$MeshFormat\n2.2 0 8\n$EndMeshFormat\n$PhysicalNames\n2\n1 1 \"inner\"\n1 2 \"outer\"\n$EndPhysicalNames\n";
    // Corner nodes, then edge midpoints (radial and angular) and cell centers on the polar grid
    std::vector<std::array<double, 2>> nodes;
    auto polar = [&](double r, double t) {
        nodes.push_back({r * std::cos(t), r * std::sin(t)});
        return nodes.size();
    };
    auto r_of = [&](double j) { return R_IN + (R_OUT - R_IN) * j / n_r; };
    auto t_of = [&](double i) { return 2.0 * M_PI * i / n_t; };
    std::vector<size_t> corner((n_r + 1) * n_t);
    for (uint32_t j = 0; j <= n_r; j++) {
        for (uint32_t i = 0; i < n_t; i++) corner[j * n_t + i] = polar(r_of(j), t_of(i));
    }
    auto c = [&](uint32_t i, uint32_t j) { return corner[j * n_t + (i % n_t)]; };
    std::vector<std::string> elements;
    for (uint32_t i = 0; i < n_t; i++) {
        for (const auto & [j, tag] : {std::pair{0u, 1}, std::pair{n_r, 2}}) {
            const size_t m = polar(r_of(j), t_of(i + 0.5));
            elements.push_back("8 2 " + std::to_string(tag) + " " + std::to_string(tag) + " " + std::to_string(c(i, j)) +
                               " " + std::to_string(c(i + 1, j)) + " " + std::to_string(m));
        }
    }
    for (uint32_t j = 0; j < n_r; j++) {
        for (uint32_t i = 0; i < n_t; i++) {
            std::string e = "10 2 0 1 " + std::to_string(c(i, j)) + " " + std::to_string(c(i + 1, j)) + " " +
                            std::to_string(c(i + 1, j + 1)) + " " + std::to_string(c(i, j + 1));
            for (const auto & [jj, ii] : {std::pair{double(j), i + 0.5}, std::pair{j + 0.5, i + 1.0},
                                           std::pair{j + 1.0, i + 0.5}, std::pair{j + 0.5, double(i)},
                                           std::pair{j + 0.5, i + 0.5}}) {
                e += " " + std::to_string(polar(r_of(jj), t_of(ii)));
            }
            elements.push_back(e);
        }
    }
    s << "$Nodes\n" << nodes.size() << "\n";
    for (size_t k = 0; k < nodes.size(); k++) s << k + 1 << " " << nodes[k][0] << " " << nodes[k][1] << " 0\n";
    s << "$EndNodes\n$Elements\n" << elements.size() << "\n";
    for (size_t k = 0; k < elements.size(); k++) s << k + 1 << " " << elements[k] << "\n";
    s << "$EndElements\n";
    const std::string path = write_temp("curved_annulus_p2.msh", s.str());

    auto mesh = std::make_shared<Mesh>();
    mesh->init_file(path);
    EXPECT_EQ(mesh->n_nodes, (n_r + 1) * n_t);
    EXPECT_EQ(mesh->n_cells, n_r * n_t);
    EXPECT_EQ(mesh->high_order_faces.size(), 2 * n_t);
    mesh->apply_curved(curved::surfaces_of_mesh(*mesh, parse_toml("[mesh]\n")));
    ASSERT_TRUE(mesh->curved_geometry);
    auto quadratic = annulus_mesh(n_r, n_t, "high_order", false, 2);
    double total = 0.0;
    for (uint32_t k = 0; k < mesh->n_cells; k++) total += double(mesh->h_cell_volume(k));
    double total_quadratic = 0.0;
    for (uint32_t k = 0; k < quadratic->n_cells; k++) total_quadratic += double(quadratic->h_cell_volume(k));
    EXPECT_NEAR(total, total_quadratic, roundoff(1e-13));
    // Without curved geometry the file gives the polygonal annulus
    auto straight = std::make_shared<Mesh>();
    straight->init_file(path);
    straight->apply_curved(curved::surfaces_of_mesh(*straight, parse_toml("[mesh]\ncurved_geometry = false\n")));
    EXPECT_FALSE(straight->curved_geometry);
    double total_straight = 0.0;
    for (uint32_t k = 0; k < straight->n_cells; k++) total_straight += double(straight->h_cell_volume(k));
    const double exact = M_PI * (R_OUT * R_OUT - R_IN * R_IN);
    EXPECT_LT(std::abs(total - exact), 0.01 * std::abs(total_straight - exact));
}

using CurvedOrderParam = std::tuple<bool, int>;
class CurvedTENOOrder : public ::testing::TestWithParam<CurvedOrderParam> {};

TEST_P(CurvedTENOOrder, ReconstructionAtCurvedWallsConvergesAtDesignOrder) {
    // From exact averages over the curved cells, wall values converge at the
    // design order on curved geometry (fits of full degree at the walls); seen
    // as straight-sided cells, the same averages give second order at best
    SKIP_IN_SINGLE_PRECISION("errors reach single-precision round-off");
    const auto [triangles, order] = GetParam();
    const double e1 = curved_reconstruction_error(4, "curved", order, triangles);
    const double e2 = curved_reconstruction_error(8, "curved", order, triangles);
    EXPECT_GT(std::log2(e1 / e2), order - 0.5) << e1 << " " << e2;
    const double s1 = curved_reconstruction_error(4, "straight", order, triangles);
    const double s2 = curved_reconstruction_error(8, "straight", order, triangles);
    EXPECT_LT(std::log2(s1 / s2), 2.3) << s1 << " " << s2;
    EXPECT_LT(e2, 0.1 * s2);
}

INSTANTIATE_TEST_SUITE_P(Curved, CurvedTENOOrder,
                         ::testing::Combine(::testing::Values(false, true), ::testing::Values(3, 4, 5, 6)));

namespace {

/**
 * @brief The isentropic vortex u_theta = M / r between the walls (an exact
 *        steady Euler solution), on an annulus Gmsh file; geometry "curved"
 *        projects the walls onto the circles.
 */
std::string vortex_input(const std::string & mesh_file, const std::string & geometry, const std::string & recon,
                         const std::string & run, const std::string & output = "") {
    const std::string M = "0.8";
    return "[run]\n" + run + "cfl = 0.4\n[mesh]\ntype = \"file\"\nfilename = \"" + mesh_file + "\"\n" +
           (geometry == "straight" ? "curved_geometry = false\n" : "") + (geometry == "curved" ? SHAPES : "") +
           "[initialize]\ntype = \"analytical\"\n"
           "rho = \"(1 + 0.2 * " + M + "^2 * (1 - 1 / (x^2 + y^2)))^2.5\"\n"
           "u = [\"-" + M + " * y / (x^2 + y^2)\", \"" + M + " * x / (x^2 + y^2)\"]\n"
           "p = \"(1 + 0.2 * " + M + "^2 * (1 - 1 / (x^2 + y^2)))^3.5 / 1.4\"\n"
           "[[boundaries]]\nname = \"inner\"\ntype = \"symmetry\"\n"
           "[[boundaries]]\nname = \"outer\"\ntype = \"wall_adiabatic\"\n"
           "[numerics]\nriemann_solver = \"HLLC\"\ntime_integrator = \"SSPRK3\"\n"
           "[numerics.face_reconstruction]\n" + recon +
           "[physics]\ntype = \"euler\"\ngamma = 1.4\np_ref = 1.0\nT_ref = 1.0\nrho_ref = 1.0\n"
           "[output]\ncheck_interval = 1000000\n" + output;
}

std::vector<double> density(Solver & solver) {
    solver.copy_device_to_host();
    std::vector<double> rho;
    for (uint32_t c = 0; c < solver.get_mesh()->n_owned(); c++) rho.push_back(double(solver.h_conservatives(c, 0)));
    return rho;
}

}  // namespace

TEST(CurvedSolver, SteadyVortexKeepsItsExactAveragesOnlyBetweenCurvedWalls) {
    // Started from its exact cell averages, the steady vortex drifts by the
    // scheme's error: at the wall it is the geometry's, O(h^2) for straight
    // walls, which push the flow off the circles
    SKIP_IN_SINGLE_PRECISION("the curved drift is near single-precision round-off");
    const std::string file = write_temp("curved_vortex.msh", annulus_gmsh(6, 96));
    auto drift = [&](const std::string & geometry, const std::string & recon) {
        Solver solver;
        solver.init(parse_toml(vortex_input(file, geometry, recon, "t_stop = 0.5\n")));
        const std::vector<double> start = density(solver);
        solver.run();
        const std::vector<double> end = density(solver);
        double err = 0.0;
        for (size_t c = 0; c < start.size(); c++) err = std::max(err, std::abs(end[c] - start[c]));
        return err;
    };
    for (const std::string order : {"3", "5"}) {
        const std::string recon = "type = \"TENO\"\norder = " + order + "\ncurved_mirrors = false\n";
        const double straight = drift("straight", recon), curved = drift("curved", recon);
        EXPECT_LT(curved, 0.1 * straight) << "order " << order << ": " << curved << " vs " << straight;
    }
}

TEST(CurvedSolver, GasAtRestStaysExactlyAtRest) {
    // Uniform gas at rest between curved walls: the face quadrature of every
    // curved cell closes, so the wall pressures balance to round-off
    const std::string file = write_temp("curved_rest.msh", annulus_gmsh(3, 40, 3, true));
    for (const std::string geometry : {"curved", "high_order"}) {
        for (const std::string recon : {"type = \"MUSCL\"\n", "type = \"TENO\"\norder = 4\n"}) {
            std::string input = vortex_input(file, geometry, recon, "n_steps = 20\n");
            const size_t a = input.find("rho = "), b = input.find("[[boundaries]]");
            input.replace(a, b - a, "rho = \"1.0\"\nu = [\"0.0\", \"0.0\"]\np = \"0.7142857142857143\"\n");
            Solver solver;
            solver.init(parse_toml(input));
            ASSERT_TRUE(solver.get_mesh()->curved_geometry);
            solver.run();
            solver.copy_device_to_host();
            double u_max = 0.0;
            for (uint32_t c = 0; c < solver.get_mesh()->n_cells; c++) {
                u_max = std::max({u_max, std::abs(double(solver.h_conservatives(c, 1))),
                                  std::abs(double(solver.h_conservatives(c, 2)))});
            }
            EXPECT_LT(u_max, roundoff(1e-13)) << geometry << " " << recon;
        }
    }
}

TEST(CurvedSolver, RestartedRunMatchesUninterruptedRunExactly) {
    const std::string file = write_temp("curved_restart.msh", annulus_gmsh(3, 40, 2));
    const std::string dir = (std::filesystem::temp_directory_path() / "mallard_curved_restart").string();
    std::filesystem::remove_all(dir);
    const std::string recon = "type = \"TENO\"\norder = 4\n";
    const std::string output = "[[write_data]]\nprefix = \"" + dir + "/restart\"\nformat = \"restart\"\ninterval = 10\n";
    Solver straight;
    straight.init(parse_toml(vortex_input(file, "high_order", recon, "n_steps = 20\n")));
    straight.run();
    Solver first;
    first.init(parse_toml(vortex_input(file, "high_order", recon, "n_steps = 10\n", output)));
    first.run();
    Solver second;
    std::string input = vortex_input(file, "high_order", recon, "n_steps = 20\n");
    const size_t a = input.find("type = \"analytical\""), b = input.find("[[boundaries]]");
    input.replace(a, b - a, "type = \"restart\"\nfile = \"" + dir + "/restart_000010.restart\"\n");
    second.init(parse_toml(input));
    second.run();
    EXPECT_EQ(density(second), density(straight));
    std::filesystem::remove_all(dir);
}

TEST(CurvedTENO, CellsReachingCurvedWallsFitAtMostTheWallDegree) {
    // By default orders 5 and 6 fit degree 3 in cells whose stencils reach a
    // curved wall (one-sided fits of higher degree grow in time there): the
    // wall values then converge at fourth order, and at fifth with
    // curved_wall_degree = 4
    SKIP_IN_SINGLE_PRECISION("errors reach single-precision round-off");
    const double c1 = curved_reconstruction_error(4, "curved", 5, false, "\n");
    const double c2 = curved_reconstruction_error(8, "curved", 5, false, "\n");
    const double f1 = curved_reconstruction_error(4, "curved", 5, false, "curved_wall_degree = 4\n");
    const double f2 = curved_reconstruction_error(8, "curved", 5, false, "curved_wall_degree = 4\n");
    EXPECT_NEAR(std::log2(c1 / c2), 4.0, 0.5) << c1 << " " << c2;
    EXPECT_GT(std::log2(f1 / f2), 4.5) << f1 << " " << f2;
    EXPECT_THROW(curved_reconstruction_error(4, "curved", 5, false, "curved_wall_degree = 1\n"), std::runtime_error);
}

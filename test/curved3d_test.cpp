/**
 * @file curved3d_test.cpp
 * @author Matthew Bonanni (mbonanni001@gmail.com)
 * @brief Tests of curved (high-order) boundaries in 3D: spherical shells of
 *        hexahedra and prisms, with projected or quadratic walls.
 * @version 0.1
 * @date 2026-10-09
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 *
 */

#include <gtest/gtest.h>
#include <Kokkos_Core.hpp>

#include <array>
#include <cmath>
#include <map>
#include <sstream>
#include <tuple>
#include <memory>
#include <string>
#include <vector>

#include "curved.h"
#include "face_reconstruction.h"
#include "gmsh_fixtures.h"
#include "mesh.h"
#include "physics.h"
#include "solver.h"
#include "test_fixtures.h"
#include "test_utils.h"

namespace {

constexpr double R_IN = 1.0, R_OUT = 1.5;
constexpr rtype GAMMA = 1.4;

const char * SHAPES = R"(
[[mesh.curved]]
zone = "inner"
shape = "sphere"
center = [0.0, 0.0, 0.0]
radius = 1.0

[[mesh.curved]]
zone = "outer"
shape = "sphere"
center = [0.0, 0.0, 0.0]
radius = 1.5
)";

std::shared_ptr<Mesh> shell_mesh(int n, int n_r, bool prisms, const std::string & geometry) {
    const Shell s = shell(n, n_r, prisms, geometry == "quadratic");
    auto mesh = std::make_shared<Mesh>();
    mesh->init_from_connectivity(s.nodes, s.cells, s.faces);
    mesh->high_order_faces = s.high_order;
    mesh->apply_curved(curved::surfaces_of_mesh(*mesh, parse_toml("[mesh]\n" + std::string(geometry == "curved" ? SHAPES : ""))));
    mesh->copy_host_to_device();
    return mesh;
}

/** @brief int f dV over every cell by its cell rule. */
template <typename F>
std::vector<double> cell_integrals(const Mesh & mesh, int degree, F && f) {
    std::vector<double> out(mesh.n_cells, 0.0);
    std::vector<curved::Vec3> x;
    std::vector<double> w;
    for (uint32_t c = 0; c < mesh.n_cells; c++) {
        mesh.curved_geometry->cell_rule(c, degree, x, w);
        for (size_t q = 0; q < w.size(); q++) out[c] += w[q] * f(x[q]);
    }
    return out;
}

const std::string REST = "rho = \"1.0\"\nu = [\"0.0\", \"0.0\", \"0.0\"]\np = \"0.7142857142857143\"\n";

std::string shell_input(const std::string & file, const std::string & shapes, const std::string & recon,
                        const std::string & init, uint32_t n_steps) {
    return "[run]\nn_steps = " + std::to_string(n_steps) + "\ncfl = 0.3\n[mesh]\ntype = \"file\"\nfilename = \"" + file +
           "\"\n" + shapes + "[initialize]\ntype = \"analytical\"\n" + init +
           "[[boundaries]]\nname = \"inner\"\ntype = \"wall_adiabatic\"\n[[boundaries]]\nname = \"outer\"\ntype = \"symmetry\"\n"
           "[numerics]\nriemann_solver = \"HLLC\"\ntime_integrator = \"SSPRK3\"\n[numerics.face_reconstruction]\n" + recon +
           "[physics]\ntype = \"euler\"\ngamma = 1.4\np_ref = 1.0\nT_ref = 1.0\nrho_ref = 1.0\n[output]\ncheck_interval = 1000000\n";
}

}  // namespace

TEST(Curved3DGeometry, ShellVolumeAndMomentsAreExactWithProjectedWalls) {
    // The walls lie on the spheres: the shell's volume and its second moment
    // int r^2 dV are exact, whatever the blended side faces in between
    for (bool prisms : {false, true}) {
        auto mesh = shell_mesh(4, 2, prisms, "curved");
        ASSERT_TRUE(mesh->curved_geometry);
        double V = 0.0;
        for (uint32_t c = 0; c < mesh->n_cells; c++) V += double(mesh->h_cell_volume(c));
        const double exact_V = 4.0 / 3.0 * M_PI * (std::pow(R_OUT, 3) - std::pow(R_IN, 3));
        EXPECT_NEAR(V, exact_V, roundoff(1e-11) * exact_V) << prisms;
        double I = 0.0;
        for (double v : cell_integrals(*mesh, 4, [](const curved::Vec3 & x) { return x[0] * x[0] + x[1] * x[1] + x[2] * x[2]; })) I += v;
        EXPECT_NEAR(I, 4.0 / 5.0 * M_PI * (std::pow(R_OUT, 5) - std::pow(R_IN, 5)), roundoff(1e-10)) << prisms;
        // Straight walls lose O(h^2) of it
        auto straight = shell_mesh(4, 2, prisms, "straight");
        double V_straight = 0.0;
        for (uint32_t c = 0; c < straight->n_cells; c++) V_straight += double(straight->h_cell_volume(c));
        EXPECT_GT(std::abs(V_straight - exact_V), 1e-3 * exact_V);
    }
}

TEST(Curved3DGeometry, QuadraticWallsConvergeAtFourthOrder) {
    const double exact_V = 4.0 / 3.0 * M_PI * (std::pow(R_OUT, 3) - std::pow(R_IN, 3));
    for (bool prisms : {false, true}) {
        auto error = [&](int n, const std::string & geometry) {
            auto mesh = shell_mesh(n, 1, prisms, geometry);
            double V = 0.0;
            for (uint32_t c = 0; c < mesh->n_cells; c++) V += double(mesh->h_cell_volume(c));
            return std::abs(V - exact_V);
        };
        EXPECT_NEAR(std::log2(error(4, "straight") / error(8, "straight")), 2.0, 0.15) << prisms;
        EXPECT_GT(std::log2(error(4, "quadratic") / error(8, "quadratic")), 3.7) << prisms;
    }
}

TEST(Curved3DGeometry, FaceQuadratureClosesEveryCell) {
    // Area vectors of every cell's face quadrature (curved walls, blended
    // side faces, straight faces) sum to zero, so free streams are preserved
    for (bool prisms : {false, true}) {
        for (const std::string geometry : {"curved", "quadratic"}) {
            auto mesh = shell_mesh(4, 3, prisms, geometry);
            Kokkos::View<rtype *[N_CONSERVATIVE]> W("W", mesh->n_cells);
            BoundaryData bd = make_uniform_boundaries(*mesh, BoundaryType::SYMMETRY, GAMMA);
            auto teno = std::make_unique<TENO>();
            teno->set_mesh(mesh);
            teno->set_boundaries(bd);
            teno->init(parse_toml("type = \"TENO\"\norder = 3\n"));
            auto w = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), teno->face_quad_weights);
            auto n = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), teno->face_quad_normals);
            ASSERT_GT(n.extent(0), 0u);
            std::vector<double> sum(3 * mesh->n_cells, 0.0);
            double scale = 0.0;
            for (uint32_t f = 0; f < mesh->n_faces; f++) {
                for (uint32_t q = 0; q < w.extent(1); q++) {
                    const double a = 0.5 * double(mesh->h_face_area(f)) * double(w(f, q));
                    for (int side = 0; side < 2; side++) {
                        const int32_t c = mesh->h_cells_of_face(f, side);
                        if (c < 0) continue;
                        for (int i = 0; i < 3; i++) sum[3 * c + i] += (side == 0 ? a : -a) * double(n(f, q, i));
                    }
                }
                scale = std::max(scale, double(mesh->h_face_area(f)));
            }
            for (double s : sum) EXPECT_NEAR(s, 0.0, roundoff(1e-13) * scale) << prisms << " " << geometry;
        }
    }
}

TEST(Curved3DSolver, GasAtRestStaysExactlyAtRest) {
    // Uniform gas at rest between curved walls stays at rest to round-off,
    // and runs on any number of ranks match (MPI3DTest covers more ranks)
    for (bool prisms : {false, true}) {
        for (const std::string geometry : {"curved", "quadratic"}) {
            const std::string file = write_temp("curved_shell.msh", shell_gmsh(4, 3, prisms, geometry == "quadratic"));
            Solver solver;
            solver.init(parse_toml(shell_input(file, geometry == "curved" ? SHAPES : "", "type = \"TENO\"\norder = 3\n",
                                               REST, 10)));
            ASSERT_TRUE(solver.get_mesh()->curved_geometry);
            solver.run();
            solver.copy_device_to_host();
            double u_max = 0.0;
            for (uint32_t c = 0; c < solver.get_mesh()->n_cells; c++) {
                for (int i = 1; i <= 3; i++) u_max = std::max(u_max, std::abs(double(solver.h_conservatives(c, i))));
            }
            EXPECT_LT(u_max, roundoff(1e-13)) << prisms << " " << geometry;
        }
    }
}

namespace {

/** @brief Conservative k of a polynomial state of the given degree. */
double polynomial(int k, const curved::Vec3 & x, int degree) {
    const double base[5] = {1.0, 0.1, -0.05, 0.08, 2.6};
    double v = base[k] + 0.05 * (x[0] - 0.3 * x[1] + 0.2 * x[2]);
    if (degree >= 2) v += 0.05 * (x[0] * x[0] - 0.3 * x[1] * x[2] + 0.1 * x[0] * x[1]);
    if (degree >= 3) v += 0.02 * (x[0] * x[1] * x[2] - 0.5 * x[2] * x[2] * x[2] + 0.3 * x[0] * x[0] * x[1]);
    if (degree >= 4) v += 0.01 * (x[0] * x[0] * x[1] * x[1] - 0.4 * x[2] * x[2] * x[2] * x[0]);
    return v;
}

/**
 * @brief Max error of reconstructed wall densities at the faces' quadrature
 *        points, for exact averages over the curved cells (their cell rules at
 *        high degree) of a polynomial of the reconstruction's degree, seen by
 *        the scheme on the given geometry.
 */
double shell_polynomial_error(int n, const std::string & geometry, int order, bool prisms, int degree = -1,
                              const std::string & extra = "") {
    if (degree < 0) degree = order - 1;
    auto truth = shell_mesh(n, n / 2, prisms, "curved");
    auto mesh = shell_mesh(n, n / 2, prisms, geometry);
    std::vector<double> avg(5 * mesh->n_cells, 0.0);
    std::vector<curved::Vec3> x;
    std::vector<double> w;
    for (uint32_t c = 0; c < truth->n_cells; c++) {
        truth->curved_geometry->cell_rule(c, 8, x, w);
        double vol = 0.0;
        for (size_t q = 0; q < w.size(); q++) {
            vol += w[q];
            for (int k = 0; k < 5; k++) avg[5 * c + k] += w[q] * polynomial(k, x[q], degree);
        }
        for (int k = 0; k < 5; k++) avg[5 * c + k] /= vol;
    }
    Euler euler = Euler::from_reference(GAMMA, 1.0, 1.0, 1.0);
    Kokkos::View<rtype *[N_CONSERVATIVE]> W("W", mesh->n_cells);
    auto h_W = Kokkos::create_mirror_view(W);
    for (uint32_t c = 0; c < mesh->n_cells; c++) {
        rtype U[N_CONSERVATIVE], Wc[N_CONSERVATIVE];
        FOR_I_CONSERVATIVE U[i] = rtype(avg[5 * c + i]);
        euler.compute_W_from_conservatives(Wc, U);
        FOR_I_CONSERVATIVE h_W(c, i) = Wc[i];
    }
    Kokkos::deep_copy(W, h_W);
    // Walls take no mirror images: curved ones never do, and straight walls of a sphere
    // (each a mirror plane of its own) become partition faces
    const BoundaryType walls = geometry == "straight" ? BoundaryType::PARTITION : BoundaryType::SYMMETRY;
    BoundaryData bd = make_uniform_boundaries(*mesh, walls, GAMMA);
    auto teno = std::make_unique<TENO>();
    teno->set_mesh(mesh);
    teno->set_boundaries(bd);
    teno->init(parse_toml("type = \"TENO\"\norder = " + std::to_string(order) + "\n" + extra));
    const uint8_t n_quad = teno->n_face_quadrature_points();
    Kokkos::View<rtype **[2][N_CONSERVATIVE]> face_W("face_W", mesh->n_faces, n_quad);
    teno->calc_face_values(W, face_W);
    auto h_face_W = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), face_W);
    auto h_points = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), teno->face_quad_points);
    auto h_weights = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), teno->face_quad_weights);
    double err = 0.0;
    for (uint32_t f = 0; f < mesh->n_faces; f++) {
        if (mesh->h_cells_of_face(f, 1) >= 0) continue;
        for (uint8_t q = 0; q < n_quad; q++) {
            if (h_weights(f, q) == 0.0_r) continue;
            const curved::Vec3 p = {double(h_points(f, q, 0)), double(h_points(f, q, 1)), double(h_points(f, q, 2))};
            err = std::max(err, std::abs(double(h_face_W(f, q, 0, 0)) - polynomial(0, p, degree)));
        }
    }
    return err;
}

}  // namespace

TEST(Curved3DTENO, WallValuesAreExactForPolynomialsOnCurvedCells) {
    // A polynomial of the reconstruction's degree, averaged over the curved
    // cells, is reproduced at the curved walls to round-off: the curved cells'
    // moments, centroids and face points agree. On straight-sided cells the
    // same averages miss by the geometry's error
    SKIP_IN_SINGLE_PRECISION("round-off");
    for (bool prisms : {false, true}) {
        for (int order : {3, 4}) {
            EXPECT_LT(shell_polynomial_error(4, "curved", order, prisms), 1e-11) << prisms << " " << order;
            EXPECT_GT(shell_polynomial_error(4, "straight", order, prisms), 1e-4) << prisms << " " << order;
        }
    }
}

TEST(Curved3DGeometry, CellRulesMatchTheDivergenceOfTheirFaces) {
    // int_cell x^2 y dV = closed-integral (x^3 y / 3) n_x dS over the cell's
    // faces (curved walls, blended side faces): the cone rules and the face
    // maps agree cell by cell. On prisms every straight face is planar; the
    // straight quadrilaterals of hexahedral shells are not, and their volumes
    // follow the triangles of the cell's tetrahedra rather than the bilinear face
    for (bool prisms : {true}) {
        auto mesh = shell_mesh(4, 2, prisms, "curved");
        const curved::Geometry & g = *mesh->curved_geometry;
        std::vector<double> surface(mesh->n_cells, 0.0);
        std::vector<curved::Vec3> x, n;
        std::vector<double> w;
        for (uint32_t f = 0; f < mesh->n_faces; f++) {
            std::vector<std::array<double, 2>> ref;
            std::vector<double> ref_w;
            const int m = 10;
            std::vector<double> gx = {-0.9739065285171717, -0.8650633666889845, -0.6794095682990244, -0.4333953941292472,
                                      -0.1488743389816312, 0.1488743389816312, 0.4333953941292472, 0.6794095682990244,
                                      0.8650633666889845, 0.9739065285171717};
            std::vector<double> gw = {0.0666713443086881, 0.1494513491505806, 0.2190863625159820, 0.2692667193099963,
                                      0.2955242247147529, 0.2955242247147529, 0.2692667193099963, 0.2190863625159820,
                                      0.1494513491505806, 0.0666713443086881};
            for (int a = 0; a < m; a++) {
                for (int b = 0; b < m; b++) {
                    if (mesh->h_n_nodes_of_face(f) == 3) {
                        const double u = 0.5 * (gx[a] + 1.0), v = 0.5 * (gx[b] + 1.0);
                        ref.push_back({u, v * (1.0 - u)});
                        ref_w.push_back(0.25 * gw[a] * gw[b] * (1.0 - u));
                    } else {
                        ref.push_back({gx[a], gx[b]});
                        ref_w.push_back(gw[a] * gw[b]);
                    }
                }
            }
            g.face_rule(f, ref, ref_w, x, n, w);
            double flux = 0.0;
            for (size_t q = 0; q < w.size(); q++) flux += w[q] * n[q][0] * std::pow(x[q][0], 3) * x[q][1] / 3.0;
            surface[mesh->h_cells_of_face(f, 0)] += flux;
            if (mesh->h_cells_of_face(f, 1) >= 0) surface[mesh->h_cells_of_face(f, 1)] -= flux;
        }
        const auto volume = cell_integrals(*mesh, 6, [](const curved::Vec3 & p) { return p[0] * p[0] * p[1]; });
        double worst = 0.0;
        for (uint32_t c = 0; c < mesh->n_cells; c++) worst = std::max(worst, std::abs(volume[c] - surface[c]));
        EXPECT_LT(worst, 1e-9) << prisms;
    }
}

TEST(Curved3DTENO, CellsReachingCurvedWallsFitAtMostTheWallDegree) {
    // Order 5 fits degree 3 next to curved walls by default: cubics are still
    // exact at the walls, quartics only with curved_wall_degree = 4
    SKIP_IN_SINGLE_PRECISION("round-off");
    EXPECT_LT(shell_polynomial_error(4, "curved", 5, true, 3), 1e-11);
    EXPECT_GT(shell_polynomial_error(4, "curved", 5, true, 4), 1e-6);
    EXPECT_LT(shell_polynomial_error(4, "curved", 5, true, 4, "curved_wall_degree = 4\n"), 1e-11);
}

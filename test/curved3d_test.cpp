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
#include <functional>
#include <map>
#include <random>
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
#include "verification.h"

namespace {

constexpr double R_IN = 1.0, R_OUT = 1.5;
constexpr rtype GAMMA = 1.4_r;

const char * SHAPES = R"(
[[mesh.curved]]
zone = "inner"
type = "sphere"
filled = false
center = [0.0, 0.0, 0.0]
radius = 1.0

[[mesh.curved]]
zone = "outer"
type = "sphere"
filled = false
center = [0.0, 0.0, 0.0]
radius = 1.5
)";

/** @brief Walls of the shell mapped by A (row-major 3 x 3): the spheres |x| = R become |A^-1 x| = R. */
struct Walls {
    std::string name;
    std::array<double, 9> A;
    std::string shapes;

    double det() const {
        return A[0] * (A[4] * A[8] - A[5] * A[7]) - A[1] * (A[3] * A[8] - A[5] * A[6]) + A[2] * (A[3] * A[7] - A[4] * A[6]);
    }
};

const Walls SPHERES = {"spheres", {1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0}, SHAPES};

// Ellipsoids (semi-axes 1, 0.8, 0.6 and 1.5 times those) turned 40 degrees about [1, 1, 1]:
// a primitive, and a sphere scaled along its axes
const Walls ELLIPSOIDS = {"ellipsoids",
                          {0.84402962874598542, -0.23450273108581779, 0.26945927106677214, 0.44909878511128692,
                           0.67522370299678836, -0.17587704831436332, -0.29312841385727223, 0.35927902808902956,
                           0.50641777724759118},
                          R"(
[[mesh.curved]]
zone = "inner"
type = "ellipsoid"
filled = false
center = [0.0, 0.0, 0.0]
semi_axes = [1.0, 0.8, 0.6]
rotate = {axis = [1, 1, 1], angle = 40}

[[mesh.curved]]
zone = "outer"
type = "sphere"
filled = false
center = [0.0, 0.0, 0.0]
radius = 1.5
scale = [1.0, 0.8, 0.6]
rotate = {axis = [1, 1, 1], angle = 40}
)"};

std::shared_ptr<Mesh> shell_mesh(int n, int n_r, bool prisms, const std::string & geometry, const Walls & walls = SPHERES) {
    Shell s = shell(n, n_r, prisms, geometry == "quadratic");
    for (auto & x : s.nodes) {
        const std::array<double, 3> y = {double(x[0]), double(x[1]), double(x[2])};
        for (size_t i = 0; i < 3; i++) x[i] = rtype(walls.A[3 * i] * y[0] + walls.A[3 * i + 1] * y[1] + walls.A[3 * i + 2] * y[2]);
    }
    auto mesh = std::make_shared<Mesh>();
    mesh->init_from_connectivity(s.nodes, s.cells, s.faces);
    mesh->high_order_faces = s.high_order;
    mesh->apply_curved(curved::surfaces_of_mesh(*mesh, parse_toml("[mesh]\n" + (geometry == "curved" ? walls.shapes : ""))));
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
    // The walls lie on the spheres, or on the ellipsoids by closest-point
    // projection: the shell's volume and its second moment int |A^-1 x|^2 dV
    // are exact, whatever the blended side faces in between
    for (const Walls * walls : {&SPHERES, &ELLIPSOIDS}) {
        for (bool prisms : {false, true}) {
            // The cell rules integrate the ellipsoids' tighter curves to round-off from 8 cells per cube edge
            const int n = walls == &SPHERES ? 4 : 8;
            auto mesh = shell_mesh(n, 2, prisms, "curved", *walls);
            ASSERT_TRUE(mesh->curved_geometry);
            double V = 0.0;
            for (uint32_t c = 0; c < mesh->n_cells; c++) V += double(mesh->h_cell_volume(c));
            const double exact_V = walls->det() * 4.0 / 3.0 * M_PI * (std::pow(R_OUT, 3) - std::pow(R_IN, 3));
            EXPECT_NEAR(V, exact_V, roundoff(1e-11) * exact_V) << walls->name << prisms;
            // |A^-1 x|^2 = x^T (A A^T)^-1 x
            const auto & A = walls->A;
            double AAt[9], B[9];
            for (size_t i = 0; i < 3; i++) {
                for (size_t j = 0; j < 3; j++) AAt[3 * i + j] = A[3 * i] * A[3 * j] + A[3 * i + 1] * A[3 * j + 1] + A[3 * i + 2] * A[3 * j + 2];
            }
            const double d = walls->det() * walls->det();
            for (size_t i = 0; i < 3; i++) {
                for (size_t j = 0; j < 3; j++) {
                    const size_t i1 = (j + 1) % 3, i2 = (j + 2) % 3, j1 = (i + 1) % 3, j2 = (i + 2) % 3;
                    B[3 * i + j] = (AAt[3 * i1 + j1] * AAt[3 * i2 + j2] - AAt[3 * i1 + j2] * AAt[3 * i2 + j1]) / d;
                }
            }
            double I = 0.0;
            for (double v : cell_integrals(*mesh, 4, [&](const curved::Vec3 & x) {
                     double q = 0.0;
                     for (size_t i = 0; i < 3; i++) {
                         for (size_t j = 0; j < 3; j++) q += x[i] * B[3 * i + j] * x[j];
                     }
                     return q;
                 })) {
                I += v;
            }
            EXPECT_NEAR(I, walls->det() * 4.0 / 5.0 * M_PI * (std::pow(R_OUT, 5) - std::pow(R_IN, 5)), roundoff(1e-10))
                << walls->name << prisms;
            // Straight walls lose O(h^2) of it
            auto straight = shell_mesh(n, 2, prisms, "straight", *walls);
            double V_straight = 0.0;
            for (uint32_t c = 0; c < straight->n_cells; c++) V_straight += double(straight->h_cell_volume(c));
            EXPECT_GT(std::abs(V_straight - exact_V), 1e-3 * exact_V);
        }
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
        const std::string cells = prisms ? "prisms" : "hexahedra";
        verification::expect_order({"curved", "straight walls", cells, "spherical shell volume error", {4, 8},
                                    {error(4, "straight"), error(8, "straight")}, 2.0, 1.85, 2.15});
        verification::expect_order({"curved", "quadratic walls", cells, "spherical shell volume error", {4, 8},
                                    {error(4, "quadratic"), error(8, "quadratic")}, 4.0, 3.7});
    }
}

TEST(Curved3DGeometry, FaceQuadratureClosesEveryCell) {
    // Area vectors of every cell's face quadrature (curved walls, blended
    // side faces, straight faces) sum to zero, so free streams are preserved
    for (bool prisms : {false, true}) {
        for (const auto & [geometry, walls] : {std::pair{"curved", &SPHERES}, std::pair{"quadratic", &SPHERES},
                                               std::pair{"curved", &ELLIPSOIDS}}) {
            auto mesh = shell_mesh(4, 3, prisms, geometry, *walls);
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
                        for (size_t i = 0; i < 3; i++) sum[3 * static_cast<size_t>(c) + i] += (side == 0 ? a : -a) * double(n(f, q, i));
                    }
                }
                scale = std::max(scale, double(mesh->h_face_area(f)));
            }
            // Closest-point projections are smooth to round-off only, so the two faces of an edge, which
            // evaluate it from opposite ends, agree to a few ulps of their points
            const double tol = walls == &SPHERES ? 1e-13 : 1e-12;
            for (double s : sum) EXPECT_NEAR(s, 0.0, roundoff(tol) * scale) << prisms << " " << geometry << " " << walls->name;
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
double polynomial(size_t k, const curved::Vec3 & x, int degree) {
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
/**
 * @brief Max and mean (by area) errors of reconstructed wall densities at the
 *        faces' quadrature points, for exact averages over the curved cells
 *        (their cell rules at high degree) of the conservatives field(k, x),
 *        seen by the scheme on the given geometry.
 */
std::pair<double, double> shell_wall_errors(int n, const std::string & geometry, int order, bool prisms,
                        const std::function<double(size_t, const curved::Vec3 &)> & field, const std::string & extra,
                        const Walls & walls) {
    auto truth = shell_mesh(n, n / 2, prisms, "curved", walls);
    auto mesh = shell_mesh(n, n / 2, prisms, geometry, walls);
    std::vector<double> avg(5 * mesh->n_cells, 0.0);
    std::vector<curved::Vec3> x;
    std::vector<double> w;
    for (uint32_t c = 0; c < truth->n_cells; c++) {
        truth->curved_geometry->cell_rule(c, 8, x, w);
        double vol = 0.0;
        for (size_t q = 0; q < w.size(); q++) {
            vol += w[q];
            for (uint32_t k = 0; k < 5; k++) avg[5 * c + k] += w[q] * field(k, x[q]);
        }
        for (uint32_t k = 0; k < 5; k++) avg[5 * c + k] /= vol;
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
    const BoundaryType wall_type = geometry == "straight" ? BoundaryType::PARTITION : BoundaryType::SYMMETRY;
    BoundaryData bd = make_uniform_boundaries(*mesh, wall_type, GAMMA);
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
    double err = 0.0, sum = 0.0, area = 0.0;
    for (uint32_t f = 0; f < mesh->n_faces; f++) {
        if (mesh->h_cells_of_face(f, 1) >= 0) continue;
        for (uint8_t q = 0; q < n_quad; q++) {
            if (h_weights(f, q) == 0.0_r) continue;
            const curved::Vec3 p = {double(h_points(f, q, 0)), double(h_points(f, q, 1)), double(h_points(f, q, 2))};
            const double e = std::abs(double(h_face_W(f, q, 0, 0)) - field(0, p));
            const double a = double(h_weights(f, q)) * double(mesh->h_face_area(f));
            err = std::max(err, e);
            sum += a * e;
            area += a;
        }
    }
    return {err, sum / area};
}

/**
 * @brief shell_wall_error of a polynomial of the given degree (by default the
 *        reconstruction's).
 */
double shell_polynomial_error(int n, const std::string & geometry, int order, bool prisms, int degree = -1,
                              const std::string & extra = "", const Walls & walls = SPHERES) {
    if (degree < 0) degree = order - 1;
    return shell_wall_errors(n, geometry, order, prisms, [degree](size_t k, const curved::Vec3 & x) {
        return polynomial(k, x, degree);
    }, extra, walls).first;
}

/** @brief A smooth state: conservatives c_k + a_k sin(x + b_k) cos(1.5 y + d_k) cos(z - b_k). */
double smooth(size_t k, const curved::Vec3 & x) {
    const double c[5] = {1.0, 0.1, -0.05, 0.08, 2.6}, a[5] = {0.2, 0.1, 0.1, 0.1, 0.1};
    const double b[5] = {1.0, 0.3, -0.7, 0.2, 0.5}, d[5] = {-0.5, 0.4, 0.9, -1.1, 0.3};
    return c[k] + a[k] * std::sin(x[0] + b[k]) * std::cos(1.5 * x[1] + d[k]) * std::cos(x[2] - b[k]);
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
    // Ellipsoids, projected by closest points (whose tighter curves the cell rules integrate to
    // round-off from 8 cells per cube edge)
    EXPECT_LT(shell_polynomial_error(8, "curved", 4, true, -1, "", ELLIPSOIDS), 1e-11);
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
            const size_t m = 10;
            std::vector<double> gx = {-0.9739065285171717, -0.8650633666889845, -0.6794095682990244, -0.4333953941292472,
                                      -0.1488743389816312, 0.1488743389816312, 0.4333953941292472, 0.6794095682990244,
                                      0.8650633666889845, 0.9739065285171717};
            std::vector<double> gw = {0.0666713443086881, 0.1494513491505806, 0.2190863625159820, 0.2692667193099963,
                                      0.2955242247147529, 0.2955242247147529, 0.2692667193099963, 0.2190863625159820,
                                      0.1494513491505806, 0.0666713443086881};
            for (size_t a = 0; a < m; a++) {
                for (size_t b = 0; b < m; b++) {
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
            surface[static_cast<uint32_t>(mesh->h_cells_of_face(f, 0))] += flux;
            if (mesh->h_cells_of_face(f, 1) >= 0) surface[static_cast<uint32_t>(mesh->h_cells_of_face(f, 1))] -= flux;
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

TEST(Curved3DTENO, WallValuesConvergeAtDesignOrderOnSpheresAndEllipsoids) {
    // From exact averages of a smooth state over the curved cells, wall values
    // converge at the design order on spherical walls (projected in closed
    // form) and ellipsoidal ones (by closest-point projection). Their mean
    // error: the max wall error of these shells is not yet asymptotic at 8
    // cells per cube edge, on spheres as on ellipsoids
    SKIP_IN_SINGLE_PRECISION("errors reach single-precision round-off");
    for (const Walls * walls : {&SPHERES, &ELLIPSOIDS}) {
        for (int order : {3, 4}) {
            const double e1 = shell_wall_errors(4, "curved", order, true, smooth, "", *walls).second;
            const double e2 = shell_wall_errors(8, "curved", order, true, smooth, "", *walls).second;
            verification::expect_order({"curved", "TENO" + std::to_string(order) + ", curved walls (" + walls->name + ")",
                                        "prisms", "mean wall-value error", {4, 8}, {e1, e2}, double(order), order - 0.5});
        }
    }
}



namespace {

/** @brief The shape of a region table, as [[mesh.curved]] reads it. */
curved::Shape shape_of(const std::string & table) {
    const Input input = parse_toml("shape = " + table + "\n");
    return curved::shape_of(input.at("shape"), "shape");
}

/** @brief An exact surface f(x) = 0 with grad f, a point of it per parameter pair, and its region table. */
struct ExactSurface {
    std::string name, table;
    std::function<std::array<double, 4>(const curved::Vec3 &)> f;  // f, grad f
    std::function<curved::Vec3(double, double)> point;              // u, v in [0, 1)
};

}  // namespace

TEST(Curved3DShapes, ProjectionsAreClosestPointsWithTheirDerivatives) {
    // Points near each surface go to their closest points on it (on the exact
    // surface, along its normal), and dp/dx matches fourth-order differences of
    // the projection. The union's points, away from where the cylinder meets
    // the sphere, go to the sphere
    auto polar = [](double u, double v) {
        const double t = M_PI * (0.1 + 0.8 * v), p = 2.0 * M_PI * u;
        return curved::Vec3{std::sin(t) * std::cos(p), std::sin(t) * std::sin(p), std::cos(t)};
    };
    const std::vector<ExactSurface> surfaces = {
        {"ellipsoid", "{type = \"ellipsoid\", center = [0.1, 0.0, -0.2], semi_axes = [1.0, 0.7, 0.5], filled = false}",
         [](const curved::Vec3 & x) {
             const double a = x[0] - 0.1, b = x[1] / 0.7, c = (x[2] + 0.2) / 0.5;
             return std::array<double, 4>{a * a + b * b + c * c - 1.0, 2.0 * a, 2.0 * b / 0.7, 2.0 * c / 0.5};
         },
         [&](double u, double v) {
             const curved::Vec3 d = polar(u, v);
             return curved::Vec3{0.1 + d[0], 0.7 * d[1], -0.2 + 0.5 * d[2]};
         }},
        {"torus", "{type = \"torus\", center = [0.0, 0.0, 0.0], axis = [0.0, 0.0, 1.0], major_radius = 1.0, minor_radius = 0.3, filled = false}",
         [](const curved::Vec3 & x) {
             const double rho = std::hypot(x[0], x[1]), d = rho - 1.0;
             return std::array<double, 4>{d * d + x[2] * x[2] - 0.09, 2.0 * d * x[0] / rho, 2.0 * d * x[1] / rho, 2.0 * x[2]};
         },
         [](double u, double v) {
             const double a = 2.0 * M_PI * u, b = 2.0 * M_PI * v, rho = 1.0 + 0.3 * std::cos(b);
             return curved::Vec3{rho * std::cos(a), rho * std::sin(a), 0.3 * std::sin(b)};
         }},
        {"cone", "{type = \"cone\", base = [0.0, 0.0, 0.0], axis = [0.0, 0.0, 1.0], length = 2.0, base_radius = 1.0, filled = false}",
         [](const curved::Vec3 & x) {
             const double rho = std::hypot(x[0], x[1]);
             return std::array<double, 4>{rho - 1.0 + 0.5 * x[2], x[0] / rho, x[1] / rho, 0.5};
         },
         [](double u, double v) {
             const double a = 2.0 * M_PI * u, z = 0.4 + v, rho = 1.0 - 0.5 * z;
             return curved::Vec3{rho * std::cos(a), rho * std::sin(a), z};
         }},
        {"union", "{type = \"union\", regions = [{type = \"sphere\", center = [0.0, 0.0, 0.0], radius = 1.0}, "
                  "{type = \"cylinder\", center = [0.0, 0.0, 0.0], axis = [1.0, 0.0, 0.0], radius = 0.4, length = 4.0}], filled = false}",
         [](const curved::Vec3 & x) {
             return std::array<double, 4>{x[0] * x[0] + x[1] * x[1] + x[2] * x[2] - 1.0, 2.0 * x[0], 2.0 * x[1], 2.0 * x[2]};
         },
         [&](double u, double v) {
             const curved::Vec3 d = polar(u, v);
             return curved::Vec3{d[2], d[0], d[1]};
         }},
        {"implicit", "{type = \"implicit\", f = \"x^4 + y^4 + z^4 - 1\", filled = false}",
         [](const curved::Vec3 & x) {
             return std::array<double, 4>{std::pow(x[0], 4) + std::pow(x[1], 4) + std::pow(x[2], 4) - 1.0,
                                          4.0 * std::pow(x[0], 3), 4.0 * std::pow(x[1], 3), 4.0 * std::pow(x[2], 3)};
         },
         [&](double u, double v) {
             const curved::Vec3 d = polar(u, v);
             const double r = std::pow(std::pow(d[0], 4) + std::pow(d[1], 4) + std::pow(d[2], 4), -0.25);
             return curved::Vec3{r * d[0], r * d[1], r * d[2]};
         }},
    };
    std::mt19937 rng(11);
    std::uniform_real_distribution<double> offset(-0.03, 0.03);
    const double length = 0.1, k[4] = {-2.0, -1.0, 1.0, 2.0}, wk[4] = {1.0 / 12.0, -8.0 / 12.0, 8.0 / 12.0, -1.0 / 12.0};
    for (const ExactSurface & c : surfaces) {
        const curved::Shape s = shape_of(c.table);
        EXPECT_EQ(s.kind, curved::Shape::Kind::SURFACE) << c.name;
        double worst_f = 0.0, worst_normal = 0.0, worst_J = 0.0;
        for (int i = 0; i < 8; i++) {
            for (int j = 0; j < 8; j++) {
                const curved::Vec3 on = c.point((i + 0.5) / 8.0, (j + 0.5) / 8.0);
                const curved::Vec3 x = {on[0] + offset(rng), on[1] + offset(rng), on[2] + offset(rng)};
                curved::Vec3 p;
                double J[9], unused[9];
                s.project(x, length, p, J);
                const auto f = c.f(p);
                const double gn = std::sqrt(f[1] * f[1] + f[2] * f[2] + f[3] * f[3]);
                const curved::Vec3 d = {x[0] - p[0], x[1] - p[1], x[2] - p[2]};
                const curved::Vec3 cross = {d[1] * f[3] - d[2] * f[2], d[2] * f[1] - d[0] * f[3], d[0] * f[2] - d[1] * f[1]};
                worst_f = std::max(worst_f, std::abs(f[0]) / gn);
                worst_normal = std::max(worst_normal, std::sqrt(cross[0] * cross[0] + cross[1] * cross[1] + cross[2] * cross[2]) / gn);
                for (size_t b = 0; b < 3; b++) {
                    curved::Vec3 D = {0.0, 0.0, 0.0};
                    for (size_t m = 0; m < 4; m++) {
                        curved::Vec3 y = x, q;
                        y[b] += k[m] * 1e-3;
                        s.project(y, length, q, unused);
                        for (size_t a = 0; a < 3; a++) D[a] += wk[m] * q[a] / 1e-3;
                    }
                    for (size_t a = 0; a < 3; a++) worst_J = std::max(worst_J, std::abs(J[3 * a + b] - D[a]));
                }
            }
        }
        EXPECT_LT(worst_f, 1e-14) << c.name;
        EXPECT_LT(worst_normal, 1e-13) << c.name;
        EXPECT_LT(worst_J, 1e-8) << c.name;
    }
    // Zones lie on surfaces: a volume needs filled = false
    EXPECT_THROW(shape_of("{type = \"sphere\", center = [0.0, 0.5, 0.0], radius = 1.0}"), InputError);
    // Spheres and infinite cylinders without transforms keep their closed forms; an implicit
    // sphere goes where the closed form does, with its Jacobian
    EXPECT_EQ(shape_of("{type = \"sphere\", center = [0.0, 0.5, 0.0], radius = 1.0, filled = false}").kind, curved::Shape::Kind::SPHERE);
    EXPECT_EQ(shape_of("{type = \"sphere\", center = [0.0, 0.5, 0.0], radius = 1.0, scale = 2, filled = false}").kind,
              curved::Shape::Kind::SURFACE);
    const curved::Shape cylinder = shape_of("{type = \"cylinder\", center = [1.0, 0.0, 0.0], axis = [1.0, 1.0, 0.0], radius = 0.5, filled = false}");
    ASSERT_EQ(cylinder.kind, curved::Shape::Kind::CYLINDER);
    EXPECT_NEAR(cylinder.axis[0], std::sqrt(0.5), 1e-15);
    EXPECT_NEAR(cylinder.axis[1], std::sqrt(0.5), 1e-15);
    EXPECT_EQ(cylinder.center[0], 1.0);
    EXPECT_EQ(shape_of("{type = \"cylinder\", center = [1.0, 0.0, 0.0], axis = [1.0, 1.0, 0.0], radius = 0.5, length = 3.0, filled = false}").kind,
              curved::Shape::Kind::SURFACE);
    const curved::Shape exact = shape_of("{type = \"sphere\", center = [0.3, 0.1, -0.2], radius = 0.8, filled = false}");
    const curved::Shape implicit = shape_of("{type = \"implicit\", f = \"(x - 0.3)^2 + (y - 0.1)^2 + (z + 0.2)^2 - 0.64\", filled = false}");
    double worst_p = 0.0, worst_J = 0.0;
    for (int i = 0; i < 8; i++) {
        for (int j = 0; j < 8; j++) {
            const curved::Vec3 d = polar((i + 0.5) / 8.0, (j + 0.5) / 8.0);
            const curved::Vec3 x = {0.3 + 0.8 * d[0] + offset(rng), 0.1 + 0.8 * d[1] + offset(rng), -0.2 + 0.8 * d[2] + offset(rng)};
            curved::Vec3 p, q;
            double J[9], K[9];
            exact.project(x, length, p, J);
            implicit.project(x, length, q, K);
            for (size_t a = 0; a < 3; a++) worst_p = std::max(worst_p, std::abs(p[a] - q[a]));
            for (size_t a = 0; a < 9; a++) worst_J = std::max(worst_J, std::abs(J[a] - K[a]));
        }
    }
    EXPECT_LT(worst_p, roundoff(1e-14));
    EXPECT_LT(worst_J, roundoff(1e-11));
}


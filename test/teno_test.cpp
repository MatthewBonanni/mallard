/**
 * @file teno_test.cpp
 * @author Matthew Bonanni (mbonanni001@gmail.com)
 * @brief Tests for the TENO-E reconstruction.
 * @version 0.1
 * @date 2026-10-02
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 *
 */

#include <gtest/gtest.h>
#include <Kokkos_Core.hpp>

#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <string>
#include <tuple>
#include <vector>

#include "test_fixtures.h"
#include "verification.h"
#include "gmsh_fixtures.h"
#include "lattice_symbol.h"
#include "face_reconstruction.h"
#include "physics.h"
#include "scratch.h"

namespace {

constexpr rtype GAMMA = 1.4_r;

// Smooth field satisfying symmetry conditions on the unit square: density,
// pressure and tangential velocity are even across each wall, normal velocity odd
void smooth_conservatives(double x, double y, double * U) {
    const double rho = 1.0 + 0.2 * std::cos(2.0 * M_PI * x) * std::cos(M_PI * y);
    const double u = 0.1 * std::sin(2.0 * M_PI * x) * std::cos(M_PI * y);
    const double v = -0.1 * std::sin(M_PI * y) * std::cos(M_PI * x);
    const double p = 1.0 + 0.1 * std::cos(M_PI * x) * std::cos(2.0 * M_PI * y);
    U[0] = rho;
    U[1] = rho * u;
    U[2] = rho * v;
    U[3] = p / (double(GAMMA) - 1.0) + 0.5 * rho * (u * u + v * v);
}

std::unique_ptr<TENO> make_teno(std::shared_ptr<Mesh> mesh, const BoundaryData & bd, int order,
                                const std::string & extra = "") {
    auto teno = std::make_unique<TENO>();
    teno->set_mesh(mesh);
    teno->set_boundaries(bd);
    teno->init(parse_toml("type = \"TENO\"\norder = " + std::to_string(order) + "\n" + extra));
    return teno;
}

// The field along y = 0, on a strip one cell thick
void strip_conservatives(double x, double, double * U) { smooth_conservatives(x, 0.0, U); }

using Field = void (*)(double, double, double *);

/**
 * @brief Max error of reconstructed face-point conservatives against the exact field.
 */
double reconstruction_error(std::shared_ptr<Mesh> mesh, int order, const std::string & extra = "", double margin = 0.0,
                            Field field = smooth_conservatives) {
    BoundaryData bd = make_uniform_boundaries(*mesh, BoundaryType::SYMMETRY, GAMMA);
    auto avg = cell_averages(*mesh, field);
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

    auto teno = make_teno(mesh, bd, order, extra);
    const uint8_t n_quad = teno->n_face_quadrature_points();
    Kokkos::View<rtype **[2][N_CONSERVATIVE]> face_W("face_W", mesh->n_faces, n_quad);
    teno->calc_face_values(W, face_W);
    auto h_face_W = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), face_W);
    auto h_qp = teno->quadrature_face.h_points;

    double err = 0.0;
    for (uint32_t f = 0; f < mesh->n_faces; f++) {
        const uint32_t a = mesh->h_node_of_face(f, 0), b = mesh->h_node_of_face(f, 1);
        for (uint8_t q = 0; q < n_quad; q++) {
            const double s = 0.5 * double(h_qp(q, 0));
            const double x = double(mesh->h_face_coords(f, 0)) + s * (double(mesh->h_node_coords(b, 0)) - double(mesh->h_node_coords(a, 0)));
            const double y = double(mesh->h_face_coords(f, 1)) + s * (double(mesh->h_node_coords(b, 1)) - double(mesh->h_node_coords(a, 1)));
            if (x < margin || x > 1.0 - margin || y < margin || y > 1.0 - margin) continue;
            double U[N_CONSERVATIVE];
            field(x, y, U);
            for (uint8_t side = 0; side < 2; side++) {
                if (mesh->h_cells_of_face(f, side) < 0) continue;
                err = std::max(err, std::abs(double(h_face_W(f, q, side, 0)) - U[0]));
            }
        }
    }
    return err;
}

double reconstruction_error(const std::string & mesh_type, uint32_t n, int order, const std::string & extra = "",
                            double margin = 0.0) {
    return reconstruction_error(make_mesh(mesh_type, n, n), order, extra, margin);
}

/** @brief Max face-value error on a strip of n square cells, one cell thick, with symmetry walls all around. */
double strip_error(uint32_t n, int order) {
    return reconstruction_error(make_mesh("cartesian", n, 1, 1.0_r, 1.0_r / rtype(n)), order, "", 0.0,
                                strip_conservatives);
}

using OrderParam = std::tuple<std::string, int>;
class TENOOrder : public ::testing::TestWithParam<OrderParam> {};

} // namespace

TEST_P(TENOOrder, SmoothReconstructionConvergesAtDesignOrderInInterior) {
    const auto [mesh_type, order] = GetParam();
    verification::expect_order({"reconstruction", "TENO" + std::to_string(order), verification::cells_of(mesh_type),
                                "max face-value error, interior", {16, 32},
                                {reconstruction_error(mesh_type, 16, order, "", 0.25),
                                 reconstruction_error(mesh_type, 32, order, "", 0.25)},
                                double(order), order - 0.2});
}

TEST_P(TENOOrder, SmoothReconstructionConvergesAtDesignOrderWithMirroredBoundaries) {
    // Mirror ghost cells at symmetry walls keep boundary stencils centered
    const auto [mesh_type, order] = GetParam();
    if (order == 5) SKIP_IN_SINGLE_PRECISION("the order-5 error on the 64-cell mesh is at round-off");
    verification::expect_order({"boundaries", "TENO" + std::to_string(order), verification::cells_of(mesh_type),
                                "max face-value error, symmetry walls", {32, 64},
                                {reconstruction_error(mesh_type, 32, order), reconstruction_error(mesh_type, 64, order)},
                                double(order), order - 0.4});
}

INSTANTIATE_TEST_SUITE_P(TENO, TENOOrder,
    ::testing::Combine(::testing::Values("cartesian", "cartesian_tri"),
                       ::testing::Values(3, 4, 5)));

// The rest of the verification manual's matrix: run by tools/docs/verification.py
INSTANTIATE_TEST_SUITE_P(DISABLED_Verification, TENOOrder,
    ::testing::Combine(::testing::Values("cartesian", "cartesian_tri"), ::testing::Values(6)));

TEST(TENOTest, StripsOneCellThickConvergeAtDesignOrder) {
    // No neighbors across the strip: the fits take the monomials of x alone (#706)
    SKIP_IN_SINGLE_PRECISION("the errors on the 64-cell strip are at round-off");
    for (int order : {5, 6}) {
        verification::expect_order({"boundaries", "TENO" + std::to_string(order), verification::cells_of("cartesian"),
                                    "max face-value error, one cell thick", {32, 64},
                                    {strip_error(32, order), strip_error(64, order)}, double(order), order - 0.4});
    }
}

TEST(TENOTest, EigenvectorsAreInverse) {
    const rtype W[N_CONSERVATIVE] = {1.3_r, 0.4_r, -0.7_r, 2.1_r};
    const rtype n[N_DIM] = {0.6_r, -0.8_r};
    rtype L[N_CONSERVATIVE][N_CONSERVATIVE], R[N_CONSERVATIVE][N_CONSERVATIVE];
    teno::eigenvectors(W, n, GAMMA, L, R);
    FOR_I_CONSERVATIVE {
        for (uint8_t j = 0; j < N_CONSERVATIVE; j++) {
            rtype s = 0.0;
            for (uint8_t k = 0; k < N_CONSERVATIVE; k++) s += L[i][k] * R[k][j];
            EXPECT_NEAR(s, i == j ? 1.0 : 0.0, roundoff(1e-13));
        }
    }
}

TEST(TENOTest, MonomialOrderingByTotalDegree) {
    const uint8_t expected[][2] = {{1, 0}, {0, 1}, {2, 0}, {1, 1}, {0, 2}, {3, 0}, {2, 1}, {1, 2}, {0, 3}};
    for (uint8_t l = 0; l < 9; l++) {
        uint8_t a, b;
        teno::exponents(l, a, b);
        EXPECT_EQ(a, expected[l][0]);
        EXPECT_EQ(b, expected[l][1]);
    }
}

TEST(TENOTest, AdaptiveCutoffSpansDesignRange) {
    EXPECT_RTYPE_EQ(teno::adaptive_CT(1e-3_r, 1e-3_r, 1e-2_r), 1e-10_r);
    EXPECT_RTYPE_EQ(teno::adaptive_CT(5e-2_r, 1e-3_r, 1e-2_r), 1e-6_r);
    // Troubled cells at or beyond the upper bound get the largest cutoff, also
    // when the threshold is not below the upper bound
    EXPECT_RTYPE_EQ(teno::adaptive_CT(1e-2_r, 1e-2_r, 1e-2_r), 1e-6_r);
    EXPECT_RTYPE_EQ(teno::adaptive_CT(5e-2_r, 5e-2_r, 1e-2_r), 1e-6_r);
}

namespace {

/**
 * @brief Largest overshoot of reconstructed face densities beyond the range
 *        of the data, for a density step along a slanted line.
 */
double step_overshoot(const std::string & mesh_type, const std::string & extra) {
    auto mesh = make_mesh(mesh_type, 24, 24);
    BoundaryData bd = make_uniform_boundaries(*mesh, BoundaryType::SYMMETRY, GAMMA);
    auto avg = cell_averages(*mesh, [](double x, double y, double * U) {
        const double rho = (x + 0.3 * y < 0.55) ? 1.0 : 0.125;
        const double p = (x + 0.3 * y < 0.55) ? 1.0 : 0.1;
        U[0] = rho;
        U[1] = 0.0;
        U[2] = 0.0;
        U[3] = p / (double(GAMMA) - 1.0);
    });
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
    auto teno = make_teno(mesh, bd, 5, extra);
    Kokkos::View<rtype **[2][N_CONSERVATIVE]> face_W("face_W", mesh->n_faces, teno->n_face_quadrature_points());
    teno->calc_face_values(W, face_W);
    auto h_face_W = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), face_W);
    double overshoot = 0.0;
    for (uint32_t f = 0; f < mesh->n_faces; f++) {
        // Away from the walls, where the mirrored step forms a corner
        if (double(mesh->h_face_coords(f, 1)) < 0.2 || double(mesh->h_face_coords(f, 1)) > 0.8) continue;
        for (uint8_t q = 0; q < teno->n_face_quadrature_points(); q++) {
            for (uint8_t side = 0; side < 2; side++) {
                if (mesh->h_cells_of_face(f, side) < 0) continue;
                const double rho = double(h_face_W(f, q, side, 0));
                overshoot = std::max({overshoot, rho - 1.0, 0.125 - rho});
            }
        }
    }
    return overshoot;
}

class TENOMesh : public ::testing::TestWithParam<std::string> {};

} // namespace

TEST(TENOTest, StencilSelectionSuppressesOscillationsOnQuads) {
    // Pure TENO selection, no safeguard. (On triangles every small sector
    // stencil next to a slanted step contains cut cells, so selection alone
    // cannot remove the overshoot there; see the bound-preserving test.)
    const double linear = step_overshoot("cartesian", "troubled_threshold = 1e9\n");
    const double teno = step_overshoot("cartesian", "bound_preserving = false\n");
    EXPECT_GT(linear, 0.05);
    EXPECT_LT(teno, 0.01);
}

TEST_P(TENOMesh, BoundPreservingScalingLimitsOvershoot) {
    // Residual overshoot comes only from smooth-flagged cells whose large
    // stencil grazes the step
    const double linear = step_overshoot(GetParam(), "troubled_threshold = 1e9\n");
    const double teno = step_overshoot(GetParam(), "bound_preserving = true\n");
    EXPECT_LT(teno, 0.02);
    // Single-precision node coordinates perturb the degree-4 least-squares fits, which raises it by ~30%
    EXPECT_LT(teno, precision_tol<double>(0.05, 0.07) * linear);
}

INSTANTIATE_TEST_SUITE_P(TENO, TENOMesh, ::testing::Values("cartesian", "cartesian_tri"));

TEST(TENOTest, ConditionLimitIsEnforced) {
    // Equilibrated least-squares systems on these meshes are well conditioned,
    // so only an impossible limit (below 1) rejects every stencil
    auto mesh = make_mesh("wedge", 16, 12);
    BoundaryData bd = make_uniform_boundaries(*mesh, BoundaryType::SYMMETRY, GAMMA);
    EXPECT_NO_THROW(make_teno(mesh, bd, 4, "max_condition = 1e3\n"));
    EXPECT_THROW(make_teno(mesh, bd, 4, "max_condition = 0.5\n"), std::runtime_error);
}

TEST(TENOTest, OrderTwoIsRejected) {
    auto mesh = make_mesh("cartesian", 8, 8);
    BoundaryData bd = make_uniform_boundaries(*mesh, BoundaryType::SYMMETRY, GAMMA);
    EXPECT_THROW(make_teno(mesh, bd, 2), std::runtime_error);
}

TEST(TENOTest, MirrorImagesTakeTheConditionOfTheNearestBoundaryFace) {
    // Bottom boundary: Dirichlet for x < 0.25, symmetry elsewhere. Cells far
    // from the junction must mirror across symmetry faces only.
    auto mesh = make_mesh("cartesian", 16, 8);
    std::vector<int32_t> face_bc(mesh->n_faces, -1);
    for (uint32_t f = 0; f < mesh->n_faces; f++) {
        if (mesh->h_cells_of_face(f, 1) >= 0) continue;
        const bool bottom = double(mesh->h_face_coords(f, 1)) < 1e-12;
        face_bc[f] = (bottom && double(mesh->h_face_coords(f, 0)) < 0.25) ? 1 : 0;
    }
    BoundaryCondition sym, dir;
    sym.type = BoundaryType::SYMMETRY;
    dir.type = BoundaryType::DIRICHLET;
    BoundaryData bd = make_boundary_data(*mesh, face_bc, {sym, dir}, GAMMA);
    auto teno = make_teno(mesh, bd, 4);
    const TENO::Stencils stencils = teno->large_stencils();
    uint32_t n_checked = 0;
    for (uint32_t c = 0; c < mesh->n_cells; c++) {
        for (uint64_t s = stencils.offsets[c]; s < stencils.offsets[c + 1]; s++) {
            const int32_t f = stencils.faces[s];
            if (f < 0 || double(mesh->h_face_coords(f, 1)) > 1e-12) continue;
            // The image of a cell across the bottom takes the state of the face
            // directly beneath that cell
            EXPECT_NEAR(mesh->h_face_coords(f, 0), mesh->h_cell_coords(stencils.cells[s], 0), 1e-12) << "cell " << c;
            if (double(mesh->h_cell_coords(c, 0)) > 0.6) {
                EXPECT_EQ(face_bc[static_cast<size_t>(f)], 0) << "cell " << c;
            }
            n_checked++;
        }
    }
    EXPECT_GT(n_checked, 0u);
}

TEST(TENOTest, MirrorImagesNeverLandInsideNonConvexDomains) {
    // L-shaped domain: the unit square minus its upper-right quarter
    Mesh mesh;
    const uint32_t n = 8;
    std::vector<std::array<rtype, N_DIM>> nodes;
    for (uint32_t j = 0; j <= n; j++) {
        for (uint32_t i = 0; i <= n; i++) nodes.push_back({rtype(i) / n, rtype(j) / n});
    }
    auto id = [&](uint32_t i, uint32_t j) { return j * (n + 1) + i; };
    std::vector<std::vector<uint32_t>> cells;
    for (uint32_t j = 0; j < n; j++) {
        for (uint32_t i = 0; i < n; i++) {
            if (i >= n / 2 && j >= n / 2) continue;
            cells.push_back({id(i, j), id(i + 1, j), id(i + 1, j + 1), id(i, j + 1)});
        }
    }
    mesh.init_from_connectivity(nodes, cells, {});
    auto mesh_ptr = std::make_shared<Mesh>(std::move(mesh));
    mesh_ptr->copy_host_to_device();
    std::vector<int32_t> face_bc(mesh_ptr->n_faces, -1);
    for (uint32_t f = 0; f < mesh_ptr->n_faces; f++) face_bc[f] = mesh_ptr->h_cells_of_face(f, 1) < 0 ? 0 : -1;
    BoundaryCondition sym;
    sym.type = BoundaryType::SYMMETRY;
    BoundaryData bd = make_boundary_data(*mesh_ptr, face_bc, {sym}, GAMMA);
    auto teno = make_teno(mesh_ptr, bd, 3);
    const TENO::Stencils stencils = teno->large_stencils();
    const auto & m = *mesh_ptr;
    for (uint32_t c = 0; c < m.n_cells; c++) {
        for (uint64_t s = stencils.offsets[c]; s < stencils.offsets[c + 1]; s++) {
            const int32_t f = stencils.faces[s];
            if (f < 0) continue;
            // Mirror the stencil cell's centroid across the face's line
            const rtype nx = m.h_face_normals(f, 0) / m.h_face_area(f), ny = m.h_face_normals(f, 1) / m.h_face_area(f);
            const rtype px = m.h_cell_coords(stencils.cells[s], 0), py = m.h_cell_coords(stencils.cells[s], 1);
            const rtype d = (px - m.h_face_coords(f, 0)) * nx + (py - m.h_face_coords(f, 1)) * ny;
            const rtype qx = px - 2 * d * nx, qy = py - 2 * d * ny;
            const bool in_domain = (qx > 0 && qx < 1 && qy > 0 && qy < 1) && !(qx > 0.5_r && qy > 0.5_r);
            EXPECT_FALSE(in_domain) << "cell " << c << " image at " << qx << ", " << qy;
        }
    }
}

std::shared_ptr<Mesh> mesh_from(const std::vector<std::array<rtype, N_DIM>> & nodes,
                                const std::vector<std::vector<uint32_t>> & cells) {
    Mesh mesh;
    mesh.init_from_connectivity(nodes, cells, {});
    auto ptr = std::make_shared<Mesh>(std::move(mesh));
    ptr->copy_host_to_device();
    return ptr;
}

/**
 * @brief Checks that every mirror image in the central stencils is the image
 *        of a cell in front of its face, across that face (the foot of the
 *        cell on the face's line lies on the face), for a target cell in front
 *        of the face. Returns the number of images checked.
 */
uint32_t expect_images_across_their_faces(const std::shared_ptr<Mesh> & mesh_ptr, int order) {
    const Mesh & m = *mesh_ptr;
    std::vector<int32_t> face_bc(m.n_faces, -1);
    for (uint32_t f = 0; f < m.n_faces; f++) face_bc[f] = m.h_cells_of_face(f, 1) < 0 ? 0 : -1;
    BoundaryCondition sym;
    sym.type = BoundaryType::SYMMETRY;
    BoundaryData bd = make_boundary_data(m, face_bc, {sym}, GAMMA);
    auto teno = make_teno(mesh_ptr, bd, order);
    const TENO::Stencils stencils = teno->large_stencils();
    uint32_t n_checked = 0;
    for (uint32_t c = 0; c < m.n_cells; c++) {
        for (uint64_t s = stencils.offsets[c]; s < stencils.offsets[c + 1]; s++) {
            const int32_t f = stencils.faces[s];
            if (f < 0) continue;
            const double fx = double(m.h_face_coords(f, 0)), fy = double(m.h_face_coords(f, 1));
            double nx = double(m.h_face_normals(f, 0)), ny = double(m.h_face_normals(f, 1));
            const uint32_t inner = static_cast<uint32_t>(m.h_cells_of_face(f, 0));
            if ((double(m.h_cell_coords(inner, 0)) - fx) * nx + (double(m.h_cell_coords(inner, 1)) - fy) * ny < 0) {
                nx = -nx;
                ny = -ny;
            }
            auto in_front = [&](uint32_t cell) {
                return (double(m.h_cell_coords(cell, 0)) - fx) * nx + (double(m.h_cell_coords(cell, 1)) - fy) * ny > 0;
            };
            const uint32_t image_of = static_cast<uint32_t>(stencils.cells[s]);
            EXPECT_TRUE(in_front(c)) << "cell " << c << " takes an image across face " << f << " from behind it";
            EXPECT_TRUE(in_front(image_of)) << "cell " << c << " takes the image of cell " << image_of << ", behind face " << f;
            // Foot on the face's line, as a fraction of the way from node a to node b
            const uint32_t a = m.h_node_of_face(static_cast<uint32_t>(f), 0), b = m.h_node_of_face(static_cast<uint32_t>(f), 1);
            const double ax = double(m.h_node_coords(a, 0)), ay = double(m.h_node_coords(a, 1));
            const double tx = double(m.h_node_coords(b, 0)) - ax, ty = double(m.h_node_coords(b, 1)) - ay;
            const double foot = ((double(m.h_cell_coords(image_of, 0)) - ax) * tx + (double(m.h_cell_coords(image_of, 1)) - ay) * ty) /
                                (tx * tx + ty * ty);
            EXPECT_GE(foot, -1e-9) << "cell " << c << " takes the image of cell " << image_of << " across the extension of face " << f;
            EXPECT_LE(foot, 1 + 1e-9) << "cell " << c << " takes the image of cell " << image_of << " across the extension of face " << f;
            n_checked++;
        }
    }
    return n_checked;
}

TEST(TENOTest, MirrorImagesOfAFacetedCurvedWallAreAcrossTheFacetUnderTheirCell) {
    // Annulus of thin quadrilaterals (aspect ratio 10 at the inner circle) with
    // straight faces: the line of a facet, extended, passes a cell height
    // inside the next column of cells, so reflections across it misplace the
    // flow there (#382)
    const uint32_t n_theta = 32, n_r = 8;
    const double r0 = 0.5, dr0 = 2.0 * M_PI * r0 / n_theta / 10.0, q = 1.5;
    std::vector<std::array<rtype, N_DIM>> nodes;
    double r = r0, dr = dr0;
    for (uint32_t j = 0; j <= n_r; j++, r += dr, dr *= q) {
        for (uint32_t i = 0; i < n_theta; i++) {
            const double t = 2.0 * M_PI * i / n_theta;
            nodes.push_back({rtype(r * std::cos(t)), rtype(r * std::sin(t))});
        }
    }
    std::vector<std::vector<uint32_t>> cells;
    for (uint32_t j = 0; j < n_r; j++) {
        for (uint32_t i = 0; i < n_theta; i++) {
            const uint32_t i1 = (i + 1) % n_theta;
            cells.push_back({j * n_theta + i, j * n_theta + i1, (j + 1) * n_theta + i1, (j + 1) * n_theta + i});
        }
    }
    const auto mesh = mesh_from(nodes, cells);
    for (int order : {3, 5}) EXPECT_GT(expect_images_across_their_faces(mesh, order), 0u) << "order " << order;
}

TEST(TENOTest, MirrorImagesAcrossAThinBodyStayOnTheTargetsSide) {
    // Square with a slot-shaped hole three quarters of a cell thick (as at a
    // closed trailing edge): a cell below it mirrored across its lower side
    // lands inside the hole, a quarter cell from its upper side, next to the
    // cells above it (#382)
    const uint32_t n = 8;
    const double h = 1.0 / n;
    std::vector<std::array<rtype, N_DIM>> nodes;
    for (uint32_t j = 0; j <= n; j++) {
        const double y = j <= 4 ? j * h : (j - 0.25) * h;
        for (uint32_t i = 0; i <= n; i++) nodes.push_back({rtype(i * h), rtype(y)});
    }
    std::vector<std::vector<uint32_t>> cells;
    for (uint32_t j = 0; j < n; j++) {
        for (uint32_t i = 0; i < n; i++) {
            if (j == 4 && i >= 2 && i < 6) continue;
            const uint32_t k = j * (n + 1) + i;
            cells.push_back({k, k + 1, k + n + 2, k + n + 1});
        }
    }
    const auto mesh = mesh_from(nodes, cells);
    for (int order : {3, 5}) EXPECT_GT(expect_images_across_their_faces(mesh, order), 0u) << "order " << order;
}

/**
 * @brief Checks that no mirror image in the central stencils lies past the
 *        far side of the body behind its face: the path from its foot on the
 *        face's line to the image crosses no boundary face. Returns the number
 *        of images checked.
 */
uint32_t expect_images_inside_the_body(const std::shared_ptr<Mesh> & mesh_ptr, int order) {
    const Mesh & m = *mesh_ptr;
    std::vector<int32_t> face_bc(m.n_faces, -1);
    std::vector<uint32_t> boundary;
    for (uint32_t f = 0; f < m.n_faces; f++) {
        if (m.h_cells_of_face(f, 1) >= 0) continue;
        face_bc[f] = 0;
        boundary.push_back(f);
    }
    BoundaryCondition sym;
    sym.type = BoundaryType::SYMMETRY;
    BoundaryData bd = make_boundary_data(m, face_bc, {sym}, GAMMA);
    auto teno = make_teno(mesh_ptr, bd, order);
    const TENO::Stencils stencils = teno->large_stencils();
    using P = std::array<double, 2>;
    auto node = [&](uint32_t f, uint32_t k) {
        const uint32_t n = m.h_node_of_face(f, k);
        return P{double(m.h_node_coords(n, 0)), double(m.h_node_coords(n, 1))};
    };
    auto side = [](const P & a, const P & b, const P & q) {
        const double cross = (b[0] - a[0]) * (q[1] - a[1]) - (b[1] - a[1]) * (q[0] - a[0]);
        return (cross > 1e-6) - (cross < -1e-6);
    };
    auto hidden = [&](const P & a, const P & b) {
        for (uint32_t f : boundary) {
            const P p = node(f, 0u), q = node(f, 1u);
            if (side(a, b, p) * side(a, b, q) < 0 && side(p, q, a) * side(p, q, b) < 0) return true;
        }
        return false;
    };
    uint32_t n_checked = 0;
    for (uint32_t c = 0; c < m.n_cells; c++) {
        for (uint64_t s = stencils.offsets[c]; s < stencils.offsets[c + 1]; s++) {
            const int32_t f = stencils.faces[s];
            if (f < 0) continue;
            const uint32_t of = static_cast<uint32_t>(stencils.cells[s]);
            const P y{double(m.h_cell_coords(of, 0)), double(m.h_cell_coords(of, 1))};
            const P a = node(static_cast<uint32_t>(f), 0u), b = node(static_cast<uint32_t>(f), 1u);
            const double tx = b[0] - a[0], ty = b[1] - a[1];
            const double t = ((y[0] - a[0]) * tx + (y[1] - a[1]) * ty) / (tx * tx + ty * ty);
            const P foot{a[0] + t * tx, a[1] + t * ty};
            const P image{2 * foot[0] - y[0], 2 * foot[1] - y[1]};
            EXPECT_FALSE(hidden(foot, image)) << "cell " << c << " takes the image of cell " << of << " across face " << f
                                              << " at " << image[0] << ", " << image[1] << ", past another boundary";
            n_checked++;
        }
    }
    return n_checked;
}

TEST(TENOTest, MirrorImagesAtAClosedTrailingEdgeStayInsideTheBody) {
    // Square with a lens-shaped hole, closed at both ends and at most three
    // quarters of a cell thick, as at the closed trailing edge of an airfoil:
    // images of the cells on one side, mirrored across it, would land past
    // the other side, among the cells there (#976)
    const uint32_t n = 8;
    const double h = 1.0 / n;
    std::vector<std::array<rtype, N_DIM>> nodes;
    for (uint32_t j = 0; j <= n; j++) {
        for (uint32_t i = 0; i <= n; i++) nodes.push_back({rtype(i * h), rtype(j * h)});
    }
    // The upper side of the hole, between the shared nodes at its ends (i = 2 and 6)
    const double thickness[] = {0.5 * h, 0.75 * h, 0.5 * h};
    const uint32_t upper0 = static_cast<uint32_t>(nodes.size());
    for (uint32_t i = 3; i <= 5; i++) nodes.push_back({rtype(i * h), rtype(0.5 + thickness[i - 3])});
    auto id = [&](uint32_t i, uint32_t j) { return j * (n + 1) + i; };
    auto lower_of_row4 = [&](uint32_t i) { return i >= 3 && i <= 5 ? upper0 + i - 3 : id(i, 4); };
    std::vector<std::vector<uint32_t>> cells;
    for (uint32_t j = 0; j < n; j++) {
        for (uint32_t i = 0; i < n; i++) {
            if (j == 4) {
                cells.push_back({lower_of_row4(i), lower_of_row4(i + 1), id(i + 1, 5), id(i, 5)});
            } else {
                cells.push_back({id(i, j), id(i + 1, j), id(i + 1, j + 1), id(i, j + 1)});
            }
        }
    }
    const auto mesh = mesh_from(nodes, cells);
    for (int order : {3, 4, 5}) EXPECT_GT(expect_images_inside_the_body(mesh, order), 0u) << "order " << order;
}

TEST(TENOTest, StencilsOfThinCellsSpanTheirLongDirection) {
    // Cells ten times wider than tall: by physical distance the nearest cells
    // are whole columns above and below, and the fit resolves x only from the
    // next columns, which amplifies grid-scale modes (#382). In the metric of
    // the mesh spacing the stencils span two columns either side.
    const uint32_t n = 16;
    auto mesh = make_mesh("cartesian", n, n, 1.0_r, 0.1_r);
    BoundaryData bd = make_uniform_boundaries(*mesh, BoundaryType::SYMMETRY, GAMMA);
    auto teno = make_teno(mesh, bd, 4);
    const TENO::Stencils stencils = teno->large_stencils();
    const double dx = 1.0 / n;
    for (uint32_t c = 0; c < mesh->n_cells; c++) {
        const double x = double(mesh->h_cell_coords(c, 0));
        if (x < 3 * dx || x > 1 - 3 * dx) continue;
        double left = 0.0, right = 0.0;
        for (uint64_t s = stencils.offsets[c]; s < stencils.offsets[c + 1]; s++) {
            const double d = double(mesh->h_cell_coords(stencils.cells[s], 0)) - x;
            left = std::max(left, -d);
            right = std::max(right, d);
        }
        EXPECT_GT(std::min(left, right), 1.5 * dx) << "cell " << c;
    }
}

/**
 * @brief Face values (every cell troubled, so the large, sector and smoothness
 *        tables all take part) and centroid gradients of a TENO on a smooth field.
 */
std::vector<double> teno_outputs(std::shared_ptr<Mesh> mesh, const BoundaryData & bd, int order,
                                 const std::string & cache, uint8_t slice_shift = teno::SLICE_SHIFT,
                                 bool save = false, const std::string & options = "") {
    auto teno = std::make_unique<TENO>();
    teno->set_mesh(mesh);
    teno->set_boundaries(bd);
    teno->slice_shift = slice_shift;
    teno->init(parse_toml("type = \"TENO\"\norder = " + std::to_string(order) +
                          "\ntroubled_threshold = 0\ncache_file = \"" + cache + "\"\n" + options));
    if (save) teno->save_cache();
    auto avg = cell_averages(*mesh, smooth_conservatives);
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
    Kokkos::View<rtype **[2][N_CONSERVATIVE]> face_W("face_W", mesh->n_faces, teno->n_face_quadrature_points());
    teno->calc_face_values(W, face_W);
    Kokkos::View<rtype *[N_CONSERVATIVE][N_DIM]> gradients("gradients", mesh->n_cells);
    teno->cell_gradients(W, gradients, mesh->n_cells);
    auto h_face_W = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), face_W);
    auto h_gradients = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), gradients);
    std::vector<double> out(h_face_W.data(), h_face_W.data() + h_face_W.span());
    out.insert(out.end(), h_gradients.data(), h_gradients.data() + h_gradients.span());
    return out;
}

void expect_bitwise_equal(const std::vector<double> & a, const std::vector<double> & b, const std::string & what) {
    ASSERT_EQ(a.size(), b.size()) << what;
    size_t n_diff = 0;
    for (size_t i = 0; i < a.size(); i++) n_diff += std::memcmp(&a[i], &b[i], sizeof(double)) != 0;
    EXPECT_EQ(n_diff, 0u) << what;
}

std::string temp_cache(const std::string & name) {
    const std::string path = (scratch_dir() / name).string();
    std::filesystem::remove(path);
    return path;
}

/** @brief The file of cache_file on this rank: each rank of several has its own. */
std::string file_of(const std::string & cache) {
    return comm::size() > 1 ? cache + ".r" + std::to_string(comm::rank()) + "-of-" + std::to_string(comm::size()) : cache;
}

TEST(TENOTest, PackedStencilsAndTheirCacheDoNotDependOnTheSliceWidth) {
    // Stencils of different sizes, and a cell count that is not a multiple of
    // the GPU slice width, so slices are both ragged and partly empty
    auto mesh = make_mesh("cartesian_tri", 13, 11);
    BoundaryData bd = make_uniform_boundaries(*mesh, BoundaryType::SYMMETRY, GAMMA);
    {
        auto teno = make_teno(mesh, bd, 4);
        const TENO::Stencils stencils = teno->large_stencils();
        uint64_t shortest = UINT64_MAX, longest = 0;
        for (uint32_t c = 0; c < mesh->n_cells; c++) {
            shortest = std::min(shortest, stencils.offsets[c + 1] - stencils.offsets[c]);
            longest = std::max(longest, stencils.offsets[c + 1] - stencils.offsets[c]);
        }
        ASSERT_LT(shortest, longest);
        ASSERT_NE(mesh->n_cells % 32, 0u);
    }
    const std::string cache = temp_cache("mallard_teno_cache_slices.bin");
    const auto host = teno_outputs(mesh, bd, 4, "", 0);
    const auto gpu = teno_outputs(mesh, bd, 4, cache, 5, true);
    ASSERT_TRUE(std::filesystem::exists(file_of(cache)));
    const auto written = std::filesystem::last_write_time(file_of(cache));
    // A cache written with GPU slices loads into host rows, and back
    const auto loaded_host = teno_outputs(mesh, bd, 4, cache, 0, true);
    const auto loaded_gpu = teno_outputs(mesh, bd, 4, cache, 5, true);
    EXPECT_EQ(std::filesystem::last_write_time(file_of(cache)), written);  // Loaded, never rewritten
    expect_bitwise_equal(host, gpu, "32-cell slices");
    expect_bitwise_equal(host, loaded_host, "loaded into host rows");
    expect_bitwise_equal(host, loaded_gpu, "loaded into 32-cell slices");
    std::filesystem::remove(file_of(cache));
}

TEST(TENOTest, SinglePrecisionCacheHalvesTheFileAndRunsThatWriteAndReadItAgree) {
    if constexpr (sizeof(rtype) == sizeof(float)) GTEST_SKIP() << "tables are single precision already";
    auto mesh = make_mesh("cartesian_tri", 10, 8);
    BoundaryData bd = make_uniform_boundaries(*mesh, BoundaryType::SYMMETRY, GAMMA);
    const std::string single = temp_cache("mallard_teno_cache_single.bin");
    const std::string full = temp_cache("mallard_teno_cache_double.bin");
    const std::string option = "cache_single_precision = true\n";
    const auto exact = teno_outputs(mesh, bd, 4, full, teno::SLICE_SHIFT, true);
    const auto written = teno_outputs(mesh, bd, 4, single, teno::SLICE_SHIFT, true, option);
    const auto loaded = teno_outputs(mesh, bd, 4, single, teno::SLICE_SHIFT, true, option);
    expect_bitwise_equal(written, loaded, "single-precision cache written, then read");
    double max_diff = 0.0, max_value = 0.0;
    for (size_t k = 0; k < exact.size(); k++) {
        max_diff = std::max(max_diff, std::abs(written[k] - exact[k]));
        max_value = std::max(max_value, std::abs(exact[k]));
    }
    EXPECT_GT(max_diff, 0.0);
    EXPECT_LT(max_diff, 1e-5 * max_value);
    EXPECT_LT(static_cast<double>(std::filesystem::file_size(file_of(single))), 0.6 * static_cast<double>(std::filesystem::file_size(file_of(full))));
    // A cache in the other precision is recomputed, never reinterpreted
    const auto exact_again = teno_outputs(mesh, bd, 4, single, teno::SLICE_SHIFT, true);
    expect_bitwise_equal(exact, exact_again, "double run given a single-precision cache");
    std::filesystem::remove(file_of(single));
    std::filesystem::remove(file_of(full));
}

TEST(TENOTest, StencilCacheIsRecomputedForOtherMeshesOrOptionsAndWhenTruncated) {
    auto mesh = make_mesh("cartesian_tri", 10, 8);
    BoundaryData bd = make_uniform_boundaries(*mesh, BoundaryType::SYMMETRY, GAMMA);
    auto other = make_mesh("cartesian_tri", 10, 8, 1.0, 1.5);
    BoundaryData bd_other = make_uniform_boundaries(*other, BoundaryType::SYMMETRY, GAMMA);
    const std::string cache = temp_cache("mallard_teno_cache_mismatch.bin");
    auto fresh = [&](std::shared_ptr<Mesh> m, const BoundaryData & b, int order) {
        return teno_outputs(m, b, order, "");
    };
    auto write_cache = [&]() {
        std::filesystem::remove(file_of(cache));
        teno_outputs(mesh, bd, 4, cache, teno::SLICE_SHIFT, true);
        ASSERT_TRUE(std::filesystem::exists(file_of(cache)));
    };

    write_cache();
    expect_bitwise_equal(teno_outputs(other, bd_other, 4, cache), fresh(other, bd_other, 4), "other mesh");
    write_cache();
    expect_bitwise_equal(teno_outputs(mesh, bd, 3, cache), fresh(mesh, bd, 3), "other order");
    write_cache();
    std::filesystem::resize_file(file_of(cache), std::filesystem::file_size(file_of(cache)) / 2);
    expect_bitwise_equal(teno_outputs(mesh, bd, 4, cache), fresh(mesh, bd, 4), "truncated cache");
    std::filesystem::remove(file_of(cache));
}

namespace {

/** @brief Every precomputed table of a TENO, as raw bytes. */
std::vector<std::vector<char>> teno_tables(const TENO & teno) {
    std::vector<std::vector<char>> out;
    auto add = [&](const auto & view) {
        auto h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), view);
        const char * p = reinterpret_cast<const char *>(h.data());
        out.emplace_back(p, p + h.span() * sizeof(*h.data()));
    };
    add(teno.scale);
    add(teno.cell_shape);
    add(teno.basis_mean);
    add(teno.si_matrix);
    add(teno.stencil_large_size);
    add(teno.stencil_small_size);
    for (const teno::PackedStencils * s : {&teno.stencil_large, &teno.stencil_small}) {
        add(s->slice_start);
        add(s->cells);
        add(s->images);
        add(s->pinv);
    }
    out.emplace_back(teno.gather_depth.begin(), teno.gather_depth.end());
    return out;
}

/** @brief The tables of a TENO whose setup options are set before init(). */
template <typename F>
std::vector<std::vector<char>> tables_with(std::shared_ptr<Mesh> mesh, const BoundaryData & bd, int order, F && options) {
    TENO teno;
    teno.set_mesh(mesh);
    teno.set_boundaries(bd);
    options(teno);
    teno.init(parse_toml("type = \"TENO\"\norder = " + std::to_string(order) + "\n"));
    return teno_tables(teno);
}

} // namespace

TEST(TENOTest, TablesDoNotDependOnTheBatchOrOnWhetherTheDeviceOrTheHostSetsThemUp) {
    // Triangles with mirror images at the walls: small batches split the cells
    // differently, and the host setup is the fallback for cells whose search
    // outgrows the device's scratch; order 6 on quadrilaterals too, the largest fits and smoothness matrix
    struct Case {
        const char * type;
        uint32_t nx, ny;
        int order;
    };
    for (const Case c : {Case{"cartesian_tri", 12, 10, 5}, Case{"cartesian_tri", 12, 10, 6}, Case{"cartesian", 14, 12, 6}}) {
        auto mesh = make_mesh(c.type, c.nx, c.ny);
        BoundaryData bd = make_uniform_boundaries(*mesh, BoundaryType::SYMMETRY, GAMMA);
        const auto reference = teno_tables(*make_teno(mesh, bd, c.order));
        const std::string label = std::string(c.type) + ", order " + std::to_string(c.order);
        EXPECT_TRUE(tables_with(mesh, bd, c.order, [](TENO & t) { t.setup_batch_cells = 64; }) == reference)
            << label << ", batches of 64 cells";
        EXPECT_TRUE(tables_with(mesh, bd, c.order, [](TENO & t) { t.setup_on_host = true; }) == reference)
            << label << ", host setup";
    }
}

TEST(TENOTest, CentralStencilsAloneAreTheFullSetupsBitForBit) {
    // Output of states read from files sets up only what cell_gradients and point_values read: scale,
    // basis means, and the central stencils' sizes, slices, cells, images and pseudo-inverses. With
    // share_tables, cells share basis means on the means alone, without smoothness matrices
    auto mesh = make_mesh("cartesian_tri", 12, 10);
    BoundaryData bd = make_uniform_boundaries(*mesh, BoundaryType::SYMMETRY, GAMMA);
    for (const bool share : {true, false}) {
        const auto full = tables_with(mesh, bd, 5, [&](TENO & t) { t.share_tables = share; });
        const auto central = tables_with(mesh, bd, 5, [&](TENO & t) {
            t.share_tables = share;
            t.sector_stencils = false;
        });
        std::vector<size_t> same = {0u, 4u, 6u, 7u, 8u, 9u};
        if (!share) same.insert(same.end(), {1u, 2u});
        for (const size_t k : same) EXPECT_TRUE(central[k] == full[k]) << "table " << k << (share ? ", shared" : "");
        EXPECT_TRUE(central[3].empty()) << "smoothness matrices";
        EXPECT_TRUE(central[13].empty()) << "sector pseudo-inverses";
    }
}

TEST(TENOTest, CellsWhoseSearchOutgrowsTheScratchGetTheHostSetupsTables) {
    // A thin annulus with symmetry walls: each search reaches more straight wall faces than the scratch
    // holds mirror lines, so every cell is redone on the host with a larger scratch
    Input input = parse_toml("[mesh]\ntype = \"annulus\"\nNtheta = 64\nNr = 4\nr_outer = 1.1\n");
    auto mesh = std::make_shared<Mesh>();
    mesh->init(input);
    mesh->copy_host_to_device();
    BoundaryData bd = make_uniform_boundaries(*mesh, BoundaryType::SYMMETRY, GAMMA);
    auto teno = make_teno(mesh, bd, 3);
    std::string status;
    for (const auto & [name, value] : teno->summary().settings) {
        if (name == "TENO setup") status = value;
    }
    EXPECT_NE(status.find("256 cells redone on the host"), std::string::npos) << status;
    EXPECT_TRUE(teno_tables(*teno) == tables_with(mesh, bd, 3, [](TENO & t) { t.setup_on_host = true; }));
}

TEST(TENOTest, RebuildingTheCellsWithinReachOfMovedNodesGivesTheTablesOfAFullSetup) {
    // An interior node, and a node of the bottom symmetry wall moved along it
    auto mesh = make_mesh("cartesian_tri", 14, 12);
    BoundaryData bd = make_uniform_boundaries(*mesh, BoundaryType::SYMMETRY, GAMMA);
    auto incremental = make_teno(mesh, bd, 4);
    auto changed_only = make_teno(mesh, bd, 4);
    const auto before = teno_tables(*incremental);
    auto nearest = [&](double x, double y) {
        uint32_t best = 0;
        double best_d2 = 1e300;
        for (uint32_t n = 0; n < mesh->n_nodes; n++) {
            const double d2 = std::pow(double(mesh->h_node_coords(n, 0)) - x, 2) + std::pow(double(mesh->h_node_coords(n, 1)) - y, 2);
            if (d2 < best_d2) {
                best_d2 = d2;
                best = n;
            }
        }
        return best;
    };
    const uint32_t a = nearest(0.21, 0.42), b = nearest(0.29, 0.0);
    mesh->h_node_coords(a, 0) += rtype(0.004);
    mesh->h_node_coords(a, 1) -= rtype(0.003);
    mesh->h_node_coords(b, 0) += rtype(0.005);
    mesh->compute_geometry();
    mesh->copy_host_to_device();
    std::vector<uint32_t> changed;
    for (uint32_t c = 0; c < mesh->n_cells; c++) {
        for (uint32_t k = 0; k < mesh->h_n_nodes_of_cell(c); k++) {
            const uint32_t node = mesh->h_node_of_cell(c, k);
            if (node == a || node == b) {
                changed.push_back(c);
                break;
            }
        }
    }
    const std::vector<uint32_t> cells = incremental->cells_within_reach(changed);
    ASSERT_GT(cells.size(), changed.size());
    ASSERT_LT(cells.size(), mesh->n_cells);
    incremental->rebuild_cells(cells);
    auto full = make_teno(mesh, bd, 4);
    const auto tables = teno_tables(*full);
    const auto rebuilt = teno_tables(*incremental);
    for (size_t k = 0; k < tables.size(); k++) EXPECT_TRUE(rebuilt[k] == tables[k]) << "table " << k;
    // The move changed tables beyond the moved cells: rebuilding only those is not enough
    changed_only->rebuild_cells(changed);
    EXPECT_FALSE(teno_tables(*changed_only) == tables);
    EXPECT_FALSE(before == tables);
}

namespace {

struct LatticeCase {
    const char * lattice;
    int order;
    double lowest, highest; // growth rate, sound speeds over the block size
};

void PrintTo(const LatticeCase & c, std::ostream * out) { *out << c.lattice << ", order " << c.order; }

} // namespace

class TENOLattice : public ::testing::TestWithParam<LatticeCase> {};

TEST_P(TENOLattice, LongWavesOfTheLinearizedSchemeGrowAtTheRecordedRate) {
    // Largest real part of the Bloch symbol of the scheme linearized at rest
    // (HLLC), over wavevectors up to 0.12 cycles per block. Orders 3 and 5 are
    // neutral to the noise of the finite differences. Order 4 is a known
    // failure on triangles (#1136): with the default 18-entry stencils the
    // Riemann solver's jump term feeds long acoustic waves. A fix brings its
    // rates under the bound of orders 3 and 5; then give order 4 that bound
    if (sizeof(rtype) == sizeof(float)) {
        GTEST_SKIP() << "finite differences of the right-hand side need double precision";
    }
    const LatticeCase c = GetParam();
    const uint32_t n = 12;
    const std::string mesh = std::string(c.lattice) == "crossed_squares"
                                 ? "[mesh]\ntype = \"file\"\nfilename = \"" +
                                       write_temp("mallard_teno_crossed_squares.msh", crossed_squares_mesh_2d(n)) + "\"\n"
                                 : lattice_mesh(c.lattice, n);
    const double rate = lattice_jacobian(mesh, n, c.order).largest_long_wave_rate(0.12, 0.02, 4e6);
    std::cout << c.lattice << ", order " << c.order << ": largest growth rate of the long waves: " << rate
              << " sound speeds over the block size" << std::endl;
    EXPECT_GT(rate, c.lowest);
    EXPECT_LT(rate, c.highest);
}

INSTANTIATE_TEST_SUITE_P(Triangles, TENOLattice,
                         ::testing::Values(LatticeCase{"cartesian_tri", 3, -1.0, 1e-7}, LatticeCase{"cartesian_tri", 4, 1.0e-5, 1.5e-5},
                                           LatticeCase{"cartesian_tri", 5, -1.0, 1e-7}, LatticeCase{"crossed_squares", 3, -1.0, 1e-7},
                                           LatticeCase{"crossed_squares", 4, 0.8e-6, 1.3e-6},
                                           LatticeCase{"crossed_squares", 5, -1.0, 1e-7}));

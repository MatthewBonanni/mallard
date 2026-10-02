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
#include <string>
#include <tuple>

#include "test_fixtures.h"
#include "face_reconstruction.h"
#include "physics.h"

namespace {

constexpr rtype GAMMA = 1.4;

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
    U[3] = p / (GAMMA - 1.0) + 0.5 * rho * (u * u + v * v);
}

std::unique_ptr<TENO> make_teno(std::shared_ptr<Mesh> mesh, const BoundaryData & bd, int order,
                                const std::string & extra = "") {
    auto teno = std::make_unique<TENO>();
    teno->set_mesh(mesh);
    teno->set_boundaries(bd);
    teno->init(parse_toml("type = \"TENO\"\norder = " + std::to_string(order) + "\n" + extra));
    return teno;
}

/**
 * @brief Max error of reconstructed face-point conservatives against the exact field.
 */
double reconstruction_error(const std::string & mesh_type, uint32_t n, int order, const std::string & extra = "",
                            double margin = 0.0) {
    auto mesh = make_mesh(mesh_type, n, n);
    BoundaryData bd = make_uniform_boundaries(*mesh, BoundaryType::SYMMETRY, GAMMA);
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
            const double s = 0.5 * h_qp(q, 0);
            const double x = mesh->h_face_coords(f, 0) + s * (mesh->h_node_coords(b, 0) - mesh->h_node_coords(a, 0));
            const double y = mesh->h_face_coords(f, 1) + s * (mesh->h_node_coords(b, 1) - mesh->h_node_coords(a, 1));
            if (x < margin || x > 1.0 - margin || y < margin || y > 1.0 - margin) continue;
            double U[N_CONSERVATIVE];
            smooth_conservatives(x, y, U);
            for (uint8_t side = 0; side < 2; side++) {
                if (mesh->h_cells_of_face(f, side) < 0) continue;
                err = std::max(err, std::abs(h_face_W(f, q, side, 0) - U[0]));
            }
        }
    }
    return err;
}

using OrderParam = std::tuple<std::string, int>;
class TENOOrder : public ::testing::TestWithParam<OrderParam> {};

} // namespace

TEST_P(TENOOrder, SmoothReconstructionConvergesAtDesignOrderInInterior) {
    const auto [mesh_type, order] = GetParam();
    const double e1 = reconstruction_error(mesh_type, 16, order, "", 0.25);
    const double e2 = reconstruction_error(mesh_type, 32, order, "", 0.25);
    EXPECT_GT(std::log2(e1 / e2), order - 0.2);
}

TEST_P(TENOOrder, SmoothReconstructionConvergesAtDesignOrderWithMirroredBoundaries) {
    // Mirror ghost cells at symmetry walls keep boundary stencils centered
    const auto [mesh_type, order] = GetParam();
    const double e1 = reconstruction_error(mesh_type, 32, order);
    const double e2 = reconstruction_error(mesh_type, 64, order);
    EXPECT_GT(std::log2(e1 / e2), order - 0.4);
}

INSTANTIATE_TEST_SUITE_P(TENO, TENOOrder,
    ::testing::Combine(::testing::Values("cartesian", "cartesian_tri"),
                       ::testing::Values(3, 4, 5)));

TEST(TENOTest, EigenvectorsAreInverse) {
    const rtype W[N_CONSERVATIVE] = {1.3, 0.4, -0.7, 2.1};
    const rtype n[N_DIM] = {0.6, -0.8};
    rtype L[N_CONSERVATIVE][N_CONSERVATIVE], R[N_CONSERVATIVE][N_CONSERVATIVE];
    teno::eigenvectors(W, n, GAMMA, L, R);
    FOR_I_CONSERVATIVE {
        for (uint8_t j = 0; j < N_CONSERVATIVE; j++) {
            rtype s = 0.0;
            for (uint8_t k = 0; k < N_CONSERVATIVE; k++) s += L[i][k] * R[k][j];
            EXPECT_NEAR(s, i == j ? 1.0 : 0.0, 1e-13);
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
    EXPECT_DOUBLE_EQ(teno::adaptive_CT(1e-3, 1e-3, 1e-2), 1e-10);
    EXPECT_DOUBLE_EQ(teno::adaptive_CT(5e-2, 1e-3, 1e-2), 1e-6);
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
        U[3] = p / (GAMMA - 1.0);
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
        if (mesh->h_face_coords(f, 1) < 0.2 || mesh->h_face_coords(f, 1) > 0.8) continue;
        for (uint8_t q = 0; q < teno->n_face_quadrature_points(); q++) {
            for (uint8_t side = 0; side < 2; side++) {
                if (mesh->h_cells_of_face(f, side) < 0) continue;
                const double rho = h_face_W(f, q, side, 0);
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
    EXPECT_LT(teno, 0.05 * linear);
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

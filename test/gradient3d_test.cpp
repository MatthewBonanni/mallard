/**
 * @file gradient3d_test.cpp
 * @author Matthew Bonanni (mbonanni001@gmail.com)
 * @brief Tests for 3D gradients, MUSCL reconstruction and viscous stresses.
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

#include "test_fixtures.h"
#include "face_reconstruction.h"
#include "gradient.h"
#include "viscous_flux.h"

namespace {

// Linear field W = c + g . x, positive in the unit cube
constexpr rtype C[N_CONSERVATIVE] = {2.0, 0.3, -0.2, 0.1, 3.0};
constexpr rtype G[N_CONSERVATIVE][3] = {
    {0.5, -0.25, 0.3},
    {-1.0, 0.5, 0.2},
    {0.25, 1.5, -0.7},
    {0.4, -0.3, 1.1},
    {0.75, -0.5, 0.6},
};

rtype linear(uint8_t i, const rtype * x) {
    return C[i] + G[i][0] * x[0] + G[i][1] * x[1] + G[i][2] * x[2];
}

template <typename T>
rtype linear_at(uint8_t i, const T & coords, uint32_t k) {
    const rtype x[3] = {coords(k, 0), coords(k, 1), coords(k, 2)};
    return linear(i, x);
}

Kokkos::View<rtype *[N_CONSERVATIVE]> linear_cell_field(const Mesh & mesh) {
    Kokkos::View<rtype *[N_CONSERVATIVE]> W("W", mesh.n_cells);
    auto h_W = Kokkos::create_mirror_view(W);
    for (uint32_t c = 0; c < mesh.n_cells; c++) {
        FOR_I_CONSERVATIVE h_W(c, i) = linear_at(i, mesh.h_cell_coords, c);
    }
    Kokkos::deep_copy(W, h_W);
    return W;
}

BoundaryData linear_dirichlet_boundaries(const Mesh & mesh) {
    std::vector<int32_t> face_bc(mesh.n_faces, -1);
    for (uint32_t f = 0; f < mesh.n_faces; f++) face_bc[f] = mesh.h_cells_of_face(f, 1) < 0 ? 0 : -1;
    BoundaryCondition dir;
    dir.type = BoundaryType::DIRICHLET;
    BoundaryData bd = make_boundary_data(mesh, face_bc, {dir}, 1.4);
    auto h_index = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), bd.face_state_index);
    auto h_state = Kokkos::create_mirror_view(bd.face_state);
    for (uint32_t f = 0; f < mesh.n_faces; f++) {
        if (h_index(f) < 0) continue;
        FOR_I_CONSERVATIVE h_state(h_index(f), i) = linear_at(i, mesh.h_face_coords, f);
    }
    Kokkos::deep_copy(bd.face_state, h_state);
    return bd;
}

void expect_exact_gradient(const Mesh & mesh, Kokkos::View<rtype *[N_CONSERVATIVE][N_DIM]> grad,
                           bool interior_only) {
    auto h_grad = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), grad);
    for (uint32_t c = 0; c < mesh.n_cells; c++) {
        if (interior_only && is_boundary_cell(mesh, c)) continue;
        FOR_I_CONSERVATIVE {
            for (int d = 0; d < 3; d++) {
                ASSERT_NEAR(h_grad(c, i, d), G[i][d], 1e-10) << "cell " << c << " var " << (int)i << " dir " << d;
            }
        }
    }
}

class MeshTypes3D : public ::testing::TestWithParam<std::string> {};

} // namespace

TEST_P(MeshTypes3D, LSQGradientExactForLinearFieldInInterior) {
    auto mesh = make_mesh_3d(GetParam(), 4, 3, 4);
    BoundaryData bd = make_uniform_boundaries(*mesh, BoundaryType::EXTRAPOLATION);
    auto W = linear_cell_field(*mesh);
    Kokkos::View<rtype *[N_CONSERVATIVE][N_DIM]> grad("grad", mesh->n_cells);
    LSQGradientFunctor functor{mesh->offsets_faces_of_cell, mesh->faces_of_cell, mesh->cells_of_face,
                               mesh->cell_coords, mesh->face_coords, mesh->face_normals, bd, W, grad};
    Kokkos::parallel_for(mesh->n_cells, functor);
    expect_exact_gradient(*mesh, grad, true);
}

TEST_P(MeshTypes3D, LSQGradientsExactForLinearFieldWithDirichletBoundaries) {
    // A Dirichlet state is the value at the face, so boundary cells are exact too
    auto mesh = make_mesh_3d(GetParam(), 4, 3, 4);
    BoundaryData bd = linear_dirichlet_boundaries(*mesh);
    auto W = linear_cell_field(*mesh);
    Kokkos::View<rtype *[N_CONSERVATIVE][N_DIM]> grad("grad", mesh->n_cells);
    LSQGradientFunctor functor{mesh->offsets_faces_of_cell, mesh->faces_of_cell, mesh->cells_of_face,
                               mesh->cell_coords, mesh->face_coords, mesh->face_normals, bd, W, grad};
    Kokkos::parallel_for(mesh->n_cells, functor);
    expect_exact_gradient(*mesh, grad, false);

    Kokkos::View<rtype *[N_CONSERVATIVE][N_DIM]> grad_v("grad_v", mesh->n_cells);
    functor.gradients = grad_v;
    LSQVertexGradientFunctor vertex{functor, mesh->offsets_cells_of_cell, mesh->cells_of_cell};
    Kokkos::parallel_for(mesh->n_cells, vertex);
    expect_exact_gradient(*mesh, grad_v, false);
}

TEST_P(MeshTypes3D, UnlimitedMUSCLReproducesLinearFieldAtInteriorFaces) {
    auto mesh = make_mesh_3d(GetParam(), 4, 3, 4);
    BoundaryData bd = make_uniform_boundaries(*mesh, BoundaryType::EXTRAPOLATION);
    auto W = linear_cell_field(*mesh);
    auto muscl = std::make_unique<MUSCL>();
    muscl->set_mesh(mesh);
    muscl->set_boundaries(bd);
    muscl->init(parse_toml("type = \"MUSCL\"\nlimiter = \"none\"\n"));
    Kokkos::View<rtype **[2][N_CONSERVATIVE]> face_W("face_W", mesh->n_faces, 1);
    muscl->calc_face_values(W, face_W);
    auto h_face_W = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), face_W);
    for (uint32_t f = 0; f < mesh->n_faces; f++) {
        for (uint8_t side = 0; side < 2; side++) {
            const int32_t c = mesh->h_cells_of_face(f, side);
            if (c < 0 || is_boundary_cell(*mesh, c)) continue;
            FOR_I_CONSERVATIVE {
                ASSERT_NEAR(h_face_W(f, 0, side, i), linear_at(i, mesh->h_face_coords, f), 1e-10)
                    << "face " << f << " side " << (int)side;
            }
        }
    }
}

INSTANTIATE_TEST_SUITE_P(Gradient3D, MeshTypes3D,
                         ::testing::Values("cartesian", "cartesian_tet", "cartesian_prism",
                                           "cartesian_pyramid", "cartesian_mixed"));

TEST(ViscousTraction3DTest, RigidRotationAndIsotropicExpansionAreStressFree) {
    // Stokes' hypothesis: no deviatoric stress for isotropic dilation in 3D
    const rtype mu = 0.7;
    const rtype n[3] = {2.0 / 7.0, 3.0 / 7.0, 6.0 / 7.0};
    const rtype rotation[3][3] = {{0.0, -0.3, 0.5}, {0.3, 0.0, -0.2}, {-0.5, 0.2, 0.0}};
    const rtype dilation[3][3] = {{0.4, 0.0, 0.0}, {0.0, 0.4, 0.0}, {0.0, 0.0, 0.4}};
    rtype tau_n[3];
    viscous_traction(mu, rotation, n, tau_n);
    for (int d = 0; d < 3; d++) EXPECT_NEAR(tau_n[d], 0.0, 1e-15);
    viscous_traction(mu, dilation, n, tau_n);
    for (int d = 0; d < 3; d++) EXPECT_NEAR(tau_n[d], 0.0, 1e-15);
}

TEST(ViscousTraction3DTest, SimpleShearGivesTangentialTraction) {
    // u_x = s z: the traction on a z-plane is mu s along x, on an x-plane mu s along z
    const rtype mu = 0.7, s = 1.3;
    const rtype g[3][3] = {{0.0, 0.0, s}, {0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}};
    const rtype nz[3] = {0.0, 0.0, 1.0}, nx[3] = {1.0, 0.0, 0.0};
    rtype tau_n[3];
    viscous_traction(mu, g, nz, tau_n);
    EXPECT_NEAR(tau_n[0], mu * s, 1e-15);
    EXPECT_NEAR(tau_n[1], 0.0, 1e-15);
    EXPECT_NEAR(tau_n[2], 0.0, 1e-15);
    viscous_traction(mu, g, nx, tau_n);
    EXPECT_NEAR(tau_n[0], 0.0, 1e-15);
    EXPECT_NEAR(tau_n[2], mu * s, 1e-15);
}

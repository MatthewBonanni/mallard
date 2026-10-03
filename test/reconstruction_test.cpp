/**
 * @file reconstruction_test.cpp
 * @author Matthew Bonanni (mbonanni001@gmail.com)
 * @brief Tests for gradients, limiters and face reconstruction.
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

namespace {

// Linear field W(x, y) = c + gx * x + gy * y, positive for x, y in [0, 1]
constexpr rtype C[N_CONSERVATIVE] = {1.0, 0.3, -0.2, 2.0};
constexpr rtype GX[N_CONSERVATIVE] = {0.5, -1.0, 0.25, 0.75};
constexpr rtype GY[N_CONSERVATIVE] = {-0.25, 0.5, 1.5, -0.5};

rtype linear(uint8_t i, rtype x, rtype y) {
    return C[i] + GX[i] * x + GY[i] * y;
}

Kokkos::View<rtype *[N_CONSERVATIVE]> linear_cell_field(const Mesh & mesh) {
    Kokkos::View<rtype *[N_CONSERVATIVE]> W("W", mesh.n_cells);
    auto h_W = Kokkos::create_mirror_view(W);
    for (uint32_t i_cell = 0; i_cell < mesh.n_cells; i_cell++) {
        FOR_I_CONSERVATIVE h_W(i_cell, i) = linear(i, mesh.h_cell_coords(i_cell, 0),
                                                       mesh.h_cell_coords(i_cell, 1));
    }
    Kokkos::deep_copy(W, h_W);
    return W;
}

std::unique_ptr<MUSCL> make_muscl(std::shared_ptr<Mesh> mesh, const BoundaryData & bd,
                                  const std::string & limiter) {
    auto muscl = std::make_unique<MUSCL>();
    muscl->set_mesh(mesh);
    muscl->set_boundaries(bd);
    muscl->init(parse_toml("type = \"MUSCL\"\nlimiter = \"" + limiter + "\"\n"));
    return muscl;
}

class MeshTypes : public ::testing::TestWithParam<std::string> {};

} // namespace

TEST_P(MeshTypes, LSQGradientExactForLinearFieldInInterior) {
    auto mesh = make_mesh(GetParam(), 8, 7);
    BoundaryData bd = make_uniform_boundaries(*mesh, BoundaryType::EXTRAPOLATION);
    auto W = linear_cell_field(*mesh);
    Kokkos::View<rtype *[N_CONSERVATIVE][N_DIM]> grad("grad", mesh->n_cells);
    LSQGradientFunctor functor = make_gradient(*mesh, bd, W, grad);
    Kokkos::parallel_for(mesh->n_cells, functor);
    auto h_grad = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), grad);
    for (uint32_t i_cell = 0; i_cell < mesh->n_cells; i_cell++) {
        if (is_boundary_cell(*mesh, i_cell)) continue;
        FOR_I_CONSERVATIVE {
            EXPECT_NEAR(h_grad(i_cell, i, 0), GX[i], 1e-10) << "cell " << i_cell << " var " << static_cast<int>(i);
            EXPECT_NEAR(h_grad(i_cell, i, 1), GY[i], 1e-10) << "cell " << i_cell << " var " << static_cast<int>(i);
        }
    }
}

TEST_P(MeshTypes, LSQGradientExactForLinearFieldWithDirichletBoundaries) {
    // A Dirichlet state is the value at the face, so boundary cells are exact too
    auto mesh = make_mesh(GetParam(), 8, 7);
    std::vector<int32_t> face_bc(mesh->n_faces, -1);
    for (uint32_t f = 0; f < mesh->n_faces; f++) face_bc[f] = mesh->h_cells_of_face(f, 1) < 0 ? 0 : -1;
    BoundaryCondition dir;
    dir.type = BoundaryType::DIRICHLET;
    BoundaryData bd = make_boundary_data(*mesh, face_bc, {dir}, 1.4);
    auto h_index = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), bd.face_state_index);
    auto h_state = Kokkos::create_mirror_view(bd.face_state);
    for (uint32_t f = 0; f < mesh->n_faces; f++) {
        if (h_index(f) < 0) continue;
        FOR_I_CONSERVATIVE h_state(h_index(f), i) = linear(i, mesh->h_face_coords(f, 0), mesh->h_face_coords(f, 1));
    }
    Kokkos::deep_copy(bd.face_state, h_state);
    auto W = linear_cell_field(*mesh);
    Kokkos::View<rtype *[N_CONSERVATIVE][N_DIM]> grad("grad", mesh->n_cells);
    LSQGradientFunctor functor = make_gradient(*mesh, bd, W, grad);
    Kokkos::parallel_for(mesh->n_cells, functor);
    auto h_grad = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), grad);
    for (uint32_t i_cell = 0; i_cell < mesh->n_cells; i_cell++) {
        FOR_I_CONSERVATIVE {
            EXPECT_NEAR(h_grad(i_cell, i, 0), GX[i], 1e-10) << "cell " << i_cell << " var " << static_cast<int>(i);
            EXPECT_NEAR(h_grad(i_cell, i, 1), GY[i], 1e-10) << "cell " << i_cell << " var " << static_cast<int>(i);
        }
    }
}

TEST_P(MeshTypes, VertexLSQGradientExactForQuadraticFieldWithDirichletBoundaries) {
    // The viscous gradients fit a quadratic, so they are exact for quadratic
    // fields in every cell, including boundary cells with one-sided stencils
    // (a linear fit is off by O(h) there)
    auto quadratic = [](uint8_t i, rtype x, rtype y) {
        return linear(i, x, y) + (0.3 + 0.1 * i) * x * x - 0.4 * x * y + (0.2 - 0.1 * i) * y * y;
    };
    auto mesh = make_mesh(GetParam(), 8, 7);
    std::vector<int32_t> face_bc(mesh->n_faces, -1);
    for (uint32_t f = 0; f < mesh->n_faces; f++) face_bc[f] = mesh->h_cells_of_face(f, 1) < 0 ? 0 : -1;
    BoundaryCondition dir;
    dir.type = BoundaryType::DIRICHLET;
    BoundaryData bd = make_boundary_data(*mesh, face_bc, {dir}, 1.4);
    auto h_index = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), bd.face_state_index);
    auto h_state = Kokkos::create_mirror_view(bd.face_state);
    for (uint32_t f = 0; f < mesh->n_faces; f++) {
        if (h_index(f) < 0) continue;
        FOR_I_CONSERVATIVE h_state(h_index(f), i) = quadratic(i, mesh->h_face_coords(f, 0), mesh->h_face_coords(f, 1));
    }
    Kokkos::deep_copy(bd.face_state, h_state);
    Kokkos::View<rtype *[N_CONSERVATIVE]> W("W", mesh->n_cells);
    auto h_W = Kokkos::create_mirror_view(W);
    for (uint32_t c = 0; c < mesh->n_cells; c++) {
        FOR_I_CONSERVATIVE h_W(c, i) = quadratic(i, mesh->h_cell_coords(c, 0), mesh->h_cell_coords(c, 1));
    }
    Kokkos::deep_copy(W, h_W);
    Kokkos::View<rtype *[N_CONSERVATIVE][N_DIM]> grad("grad", mesh->n_cells);
    LSQGradientFunctor faces = make_gradient(*mesh, bd, W, grad);
    Kokkos::parallel_for(mesh->n_cells, make_vertex_gradient(faces, *mesh));
    auto h_grad = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), grad);
    for (uint32_t c = 0; c < mesh->n_cells; c++) {
        const rtype x = mesh->h_cell_coords(c, 0), y = mesh->h_cell_coords(c, 1);
        FOR_I_CONSERVATIVE {
            EXPECT_NEAR(h_grad(c, i, 0), GX[i] + 2.0 * (0.3 + 0.1 * i) * x - 0.4 * y, 1e-9) << "cell " << c;
            EXPECT_NEAR(h_grad(c, i, 1), GY[i] - 0.4 * x + 2.0 * (0.2 - 0.1 * i) * y, 1e-9) << "cell " << c;
        }
    }
}

TEST_P(MeshTypes, UnlimitedMUSCLReproducesLinearFieldAtInteriorFaces) {
    auto mesh = make_mesh(GetParam(), 8, 7);
    BoundaryData bd = make_uniform_boundaries(*mesh, BoundaryType::EXTRAPOLATION);
    auto W = linear_cell_field(*mesh);
    auto muscl = make_muscl(mesh, bd, "none");
    Kokkos::View<rtype **[2][N_CONSERVATIVE]> face_W("face_W", mesh->n_faces, 1);
    muscl->calc_face_values(W, face_W);
    auto h_face_W = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), face_W);
    for (uint32_t i_face = 0; i_face < mesh->n_faces; i_face++) {
        for (uint8_t side = 0; side < 2; side++) {
            const int32_t c = mesh->h_cells_of_face(i_face, side);
            if (c < 0 || is_boundary_cell(*mesh, c)) continue;
            FOR_I_CONSERVATIVE {
                EXPECT_NEAR(h_face_W(i_face, 0, side, i),
                            linear(i, mesh->h_face_coords(i_face, 0), mesh->h_face_coords(i_face, 1)),
                            1e-10) << "face " << i_face << " side " << static_cast<int>(side);
            }
        }
    }
}

TEST_P(MeshTypes, BarthJespersenFaceValuesStayWithinNeighborBounds) {
    auto mesh = make_mesh(GetParam(), 10, 10);
    BoundaryData bd = make_uniform_boundaries(*mesh, BoundaryType::EXTRAPOLATION);
    // Discontinuous field: a step in x plus a smooth bump
    Kokkos::View<rtype *[N_CONSERVATIVE]> W("W", mesh->n_cells);
    auto h_W = Kokkos::create_mirror_view(W);
    for (uint32_t i_cell = 0; i_cell < mesh->n_cells; i_cell++) {
        const rtype x = mesh->h_cell_coords(i_cell, 0);
        const rtype y = mesh->h_cell_coords(i_cell, 1);
        FOR_I_CONSERVATIVE h_W(i_cell, i) = 1.0 + (x > 0.45 ? 1.0 : 0.0) + 0.3 * std::sin(6.0 * y + i);
    }
    Kokkos::deep_copy(W, h_W);
    auto muscl = make_muscl(mesh, bd, "barth_jespersen");
    Kokkos::View<rtype **[2][N_CONSERVATIVE]> face_W("face_W", mesh->n_faces, 1);
    muscl->calc_face_values(W, face_W);
    auto h_face_W = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), face_W);
    for (uint32_t i_face = 0; i_face < mesh->n_faces; i_face++) {
        for (uint8_t side = 0; side < 2; side++) {
            const int32_t c = mesh->h_cells_of_face(i_face, side);
            if (c < 0) continue;
            FOR_I_CONSERVATIVE {
                rtype lo = h_W(c, i), hi = h_W(c, i);
                for (uint32_t k = 0; k < mesh->h_n_faces_of_cell(c); k++) {
                    const uint32_t f = mesh->h_face_of_cell(c, k);
                    const int32_t c0 = mesh->h_cells_of_face(f, 0);
                    const int32_t c1 = mesh->h_cells_of_face(f, 1);
                    const int32_t nb = (c0 == c) ? c1 : c0;
                    if (nb < 0) continue;
                    lo = std::min(lo, h_W(nb, i));
                    hi = std::max(hi, h_W(nb, i));
                }
                EXPECT_GE(h_face_W(i_face, 0, side, i), lo - 1e-12);
                EXPECT_LE(h_face_W(i_face, 0, side, i), hi + 1e-12);
            }
        }
    }
}

TEST_P(MeshTypes, FirstOrderCopiesCellValuesAndSkipsGhostSide) {
    auto mesh = make_mesh(GetParam(), 4, 3);
    BoundaryData bd = make_uniform_boundaries(*mesh, BoundaryType::EXTRAPOLATION);
    auto W = linear_cell_field(*mesh);
    FirstOrder fo;
    fo.set_mesh(mesh);
    fo.set_boundaries(bd);
    fo.init(parse_toml("type = \"FO\"\n"));
    Kokkos::View<rtype **[2][N_CONSERVATIVE]> face_W("face_W", mesh->n_faces, 1);
    Kokkos::deep_copy(face_W, -7.0);
    fo.calc_face_values(W, face_W);
    auto h_face_W = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), face_W);
    auto h_W = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), W);
    for (uint32_t i_face = 0; i_face < mesh->n_faces; i_face++) {
        for (uint8_t side = 0; side < 2; side++) {
            const int32_t c = mesh->h_cells_of_face(i_face, side);
            FOR_I_CONSERVATIVE {
                EXPECT_EQ(h_face_W(i_face, 0, side, i), c < 0 ? -7.0 : h_W(c, i));
            }
        }
    }
}

INSTANTIATE_TEST_SUITE_P(Reconstruction, MeshTypes,
                         ::testing::Values("cartesian", "cartesian_tri", "wedge"));

/**
 * @file boundary3d_test.cpp
 * @author Matthew Bonanni (mbonanni001@gmail.com)
 * @brief Tests for boundary ghost states and transmissive image faces in 3D.
 * @version 0.1
 * @date 2026-10-02
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 *
 */

#include <gtest/gtest.h>

#include <array>
#include <Kokkos_Core.hpp>

#include <cmath>
#include <string>

#include "test_fixtures.h"
#include "boundary.h"

namespace {

constexpr rtype GAMMA = 1.4;
constexpr rtype R_GAS = 1.0 / 1.4;

const rtype N[3] = {2.0 / 7.0, 3.0 / 7.0, 6.0 / 7.0};
const rtype W_I[N_CONSERVATIVE] = {1.2, 0.3, -0.5, 0.4, 0.9};

Euler gas() {
    return Euler::from_reference(GAMMA, 1.0_r, 1.0_r / R_GAS, 1.0_r);
}

BoundaryCondition parse(const std::string & body) {
    return BoundaryCondition::from_input(parse_toml("name = \"b\"\n" + body), gas());
}

rtype normal_velocity(const rtype * W) {
    return dot<3>(W + 1, N);
}

} // namespace

TEST(Boundary3DTest, SymmetryReflectsOnlyTheNormalVelocity) {
    BoundaryCondition bc = parse("type = \"symmetry\"\n");
    rtype W_g[N_CONSERVATIVE];
    bc.ghost_W(W_I, N, GAMMA, R_GAS, true, W_g);
    EXPECT_RTYPE_EQ(W_g[0], W_I[0]);
    EXPECT_RTYPE_EQ(W_g[4], W_I[4]);
    EXPECT_NEAR(normal_velocity(W_g), -normal_velocity(W_I), roundoff(1e-14));
    rtype t_g[3], t_i[3];
    for (int d = 0; d < 3; d++) {
        t_g[d] = W_g[1 + d] - normal_velocity(W_g) * N[d];
        t_i[d] = W_I[1 + d] - normal_velocity(W_I) * N[d];
        EXPECT_NEAR(t_g[d], t_i[d], roundoff(1e-14));
    }
}

TEST(Boundary3DTest, InviscidWallIsASlipWall) {
    BoundaryCondition bc = parse("type = \"wall_adiabatic\"\nu = [1.0, 2.0, 3.0]\n");
    rtype W_g[N_CONSERVATIVE], W_s[N_CONSERVATIVE];
    bc.ghost_W(W_I, N, GAMMA, R_GAS, false, W_g);
    parse("type = \"symmetry\"\n").ghost_W(W_I, N, GAMMA, R_GAS, false, W_s);
    FOR_I_CONSERVATIVE EXPECT_RTYPE_EQ(W_g[i], W_s[i]);
}

TEST(Boundary3DTest, NoSlipMovingWallAveragesToTheWallVelocity) {
    BoundaryCondition bc = parse("type = \"wall_adiabatic\"\nu = [0.2, -0.1, 0.7]\n");
    const rtype u_wall[3] = {0.2, -0.1, 0.7};
    rtype W_g[N_CONSERVATIVE];
    bc.ghost_W(W_I, N, GAMMA, R_GAS, true, W_g);
    FOR_I_DIM EXPECT_NEAR(0.5_r * (W_g[1 + i] + W_I[1 + i]), u_wall[i], roundoff(1e-14));
    EXPECT_RTYPE_EQ(W_g[0], W_I[0]);
    EXPECT_RTYPE_EQ(W_g[4], W_I[4]);
}

TEST(Boundary3DTest, IsothermalWallAveragesToTheWallTemperature) {
    BoundaryCondition bc = parse("type = \"wall_isothermal\"\nT = 1.3\n");
    rtype W_g[N_CONSERVATIVE];
    bc.ghost_W(W_I, N, GAMMA, R_GAS, true, W_g);
    const rtype T_i = W_I[4] / (W_I[0] * R_GAS);
    const rtype T_g = W_g[4] / (W_g[0] * R_GAS);
    EXPECT_NEAR(0.5_r * (T_i + T_g), 1.3_r, roundoff(1e-12));
    EXPECT_RTYPE_EQ(W_g[4], W_I[4]);
    FOR_I_DIM EXPECT_NEAR(W_g[1 + i], -W_I[1 + i], 1e-14);
}

TEST(Boundary3DTest, HeatFluxIsStoredAfterTheVelocity) {
    BoundaryCondition bc = parse("type = \"wall_heat_flux\"\nq = 0.25\nu = [0.0, 0.0, 1.5]\n");
    EXPECT_RTYPE_EQ(bc.data[N_DIM + 1], 0.25);
    EXPECT_RTYPE_EQ(bc.data[3], 1.5);
}

TEST(Boundary3DTest, TwoComponentVelocityIsRejected) {
    EXPECT_THROW(parse("type = \"upt\"\nu = [1.0, 0.0]\np = 1.0\nT = 1.0\n"), std::runtime_error);
}

TEST(Boundary3DTest, UPTImposesTheInflowState) {
    BoundaryCondition bc = parse("type = \"upt\"\nu = [0.5, 0.25, -0.125]\np = 2.0\nT = 1.5\n");
    rtype W_g[N_CONSERVATIVE];
    bc.ghost_W(W_I, N, GAMMA, R_GAS, false, W_g);
    EXPECT_NEAR(W_g[0], 2.0_r / (R_GAS * 1.5_r), roundoff(1e-12));
    EXPECT_RTYPE_EQ(W_g[1], 0.5);
    EXPECT_RTYPE_EQ(W_g[2], 0.25);
    EXPECT_RTYPE_EQ(W_g[3], -0.125);
    EXPECT_RTYPE_EQ(W_g[4], 2.0);
}

TEST(Boundary3DTest, PressureOutletImposesSubsonicBackPressureAtFixedTemperature) {
    BoundaryCondition bc = parse("type = \"p_out\"\np = 0.6\n");
    rtype W_g[N_CONSERVATIVE];
    bc.ghost_W(W_I, N, GAMMA, R_GAS, false, W_g);
    EXPECT_RTYPE_EQ(W_g[4], 0.6);
    EXPECT_NEAR(W_g[4] / W_g[0], W_I[4] / W_I[0], 1e-14);
    FOR_I_DIM EXPECT_RTYPE_EQ(W_g[1 + i], W_I[1 + i]);
}

TEST(Boundary3DTest, FarfieldSupersonicInflowAndOutflow) {
    BoundaryCondition bc = parse("type = \"farfield\"\nu = [-3.0, 0.5, 0.2]\np = 1.0\nT = 1.0\n");
    rtype W_g[N_CONSERVATIVE];
    // Supersonic inflow (u . n < -a): the free stream
    const rtype W_in[N_CONSERVATIVE] = {1.0_r, -6.0_r * N[0], -6.0_r * N[1], -6.0_r * N[2], 1.0_r};
    bc.ghost_W(W_in, N, GAMMA, R_GAS, false, W_g);
    FOR_I_CONSERVATIVE EXPECT_RTYPE_EQ(W_g[i], bc.data[i]);
    // Supersonic outflow: the interior
    const rtype W_out[N_CONSERVATIVE] = {1.0_r, 6.0_r * N[0] + 0.1_r, 6.0_r * N[1], 6.0_r * N[2], 1.0_r};
    bc.ghost_W(W_out, N, GAMMA, R_GAS, false, W_g);
    FOR_I_CONSERVATIVE EXPECT_RTYPE_EQ(W_g[i], W_out[i]);
}

TEST(Boundary3DTest, FarfieldSubsonicStateKeepsRiemannInvariantsAndUpwindTangentialVelocity) {
    BoundaryCondition bc = parse("type = \"farfield\"\nu = [0.3, -0.2, 0.1]\np = 1.0\nT = 1.0\n");
    const rtype * W_inf = bc.data;
    for (const rtype sign : {1.0, -1.0}) {
        // Subsonic outflow (sign = 1) and inflow (sign = -1) through n
        rtype W_i[N_CONSERVATIVE] = {1.3, 0.0, 0.0, 0.0, 1.05};
        const rtype tang[3] = {3.0 / 7.0, -6.0 / 7.0, 2.0 / 7.0};
        for (int d = 0; d < 3; d++) W_i[1 + d] = sign * 0.6_r * N[d] + 0.25_r * tang[d];
        rtype W_g[N_CONSERVATIVE];
        bc.ghost_W(W_i, N, GAMMA, R_GAS, false, W_g);
        auto a = [](const rtype * W) { return std::sqrt(GAMMA * W[4] / W[0]); };
        const rtype r_out = normal_velocity(W_i) + 2.0_r * a(W_i) / (GAMMA - 1.0_r);
        const rtype r_in = normal_velocity(W_inf) - 2.0_r * a(W_inf) / (GAMMA - 1.0_r);
        EXPECT_NEAR(normal_velocity(W_g) + 2.0_r * a(W_g) / (GAMMA - 1.0_r), r_out, roundoff(1e-12));
        EXPECT_NEAR(normal_velocity(W_g) - 2.0_r * a(W_g) / (GAMMA - 1.0_r), r_in, roundoff(1e-12));
        EXPECT_GT(sign * normal_velocity(W_g), 0.0);
        const rtype * W_up = (sign > 0.0_r) ? W_i : W_inf;
        EXPECT_NEAR(W_g[4] / std::pow(W_g[0], GAMMA), W_up[4] / std::pow(W_up[0], GAMMA), roundoff(1e-12));
        for (int d = 0; d < 3; d++) {
            EXPECT_NEAR(W_g[1 + d] - normal_velocity(W_g) * N[d], W_up[1 + d] - normal_velocity(W_up) * N[d], roundoff(1e-12));
        }
    }
}

namespace {

struct GhostArgs {
    rtype W[N_CONSERVATIVE];
    rtype n[N_DIM];
};

// BoundaryData holds device views, so it is evaluated in a kernel
std::array<rtype, N_CONSERVATIVE> ghost_at_on_device(const BoundaryData & bd, uint32_t f, const rtype * W,
                                                     const rtype * n, rtype dist) {
    GhostArgs args;
    FOR_I_CONSERVATIVE args.W[i] = W[i];
    FOR_I_DIM args.n[i] = n[i];
    Kokkos::View<rtype[N_CONSERVATIVE]> out("ghost");
    Kokkos::parallel_for(1, KOKKOS_LAMBDA(const int) {
        rtype g[N_CONSERVATIVE];
        bd.ghost_W_at(f, args.W, args.n, dist, g);
        FOR_I_CONSERVATIVE out(i) = g[i];
    });
    auto h_out = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), out);
    std::array<rtype, N_CONSERVATIVE> result;
    FOR_I_CONSERVATIVE result[i] = h_out(i);
    return result;
}

} // namespace

TEST(Boundary3DTest, HydrostaticGhostPressureFollowsGravityAlongTheNormal) {
    auto mesh = make_mesh_3d("cartesian", 2, 2, 2);
    BoundaryData bd = make_uniform_boundaries(*mesh, BoundaryType::SYMMETRY);
    bd.gravity[0] = 0.0;
    bd.gravity[1] = 0.0;
    bd.gravity[2] = -2.0;
    uint32_t f_back = 0;
    for (uint32_t f = 0; f < mesh->n_faces; f++) {
        if (mesh->h_cells_of_face(f, 1) < 0 && mesh->h_face_coords(f, 2) < 1e-12_r) f_back = f;
    }
    const rtype n_out[3] = {0.0, 0.0, -1.0};
    // The ghost point 0.5 below the plane sits deeper in the hydrostatic column
    const auto W_g = ghost_at_on_device(bd, f_back, W_I, n_out, 0.5);
    EXPECT_NEAR(W_g[4], W_I[4] + W_I[0] * 2.0_r * 0.5_r, roundoff(1e-14));
}

TEST(Boundary3DTest, TransmissiveImageFaceIsTheOppositeFaceOfAHexCell) {
    auto mesh = make_mesh_3d("cartesian", 3, 4, 5, 1.5, 1.0, 2.0);
    BoundaryData bd = make_uniform_boundaries(*mesh, BoundaryType::EXTRAPOLATION);
    auto h_image = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), bd.face_image);
    auto h_image_face = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), bd.face_image_face);
    auto h_side = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), bd.face_image_side);
    const rtype h[3] = {0.5, 0.25, 0.4};
    uint32_t n_boundary = 0;
    for (uint32_t f = 0; f < mesh->n_faces; f++) {
        if (mesh->h_cells_of_face(f, 1) >= 0) continue;
        n_boundary++;
        const int32_t c = mesh->h_cells_of_face(f, 0);
        EXPECT_EQ(h_image(f), c);
        const int32_t g = h_image_face(f);
        ASSERT_GE(g, 0) << "face " << f;
        EXPECT_EQ(mesh->h_cells_of_face(g, h_side(f)), c);
        for (int d = 0; d < 3; d++) {
            const rtype n_in = -mesh->h_face_normals(f, d) / mesh->h_face_area(f);
            EXPECT_NEAR(mesh->h_face_coords(g, d), mesh->h_face_coords(f, d) + h[d] * n_in, 1e-12);
        }
    }
    EXPECT_EQ(n_boundary, 2u * (3 * 4 + 4 * 5 + 3 * 5));
}

TEST(Boundary3DTest, TransmissiveImageCellContainsTheImagePoint) {
    for (const char * type : {"cartesian_tet", "cartesian_prism", "cartesian_pyramid", "cartesian_mixed"}) {
        auto mesh = make_mesh_3d(type, 3, 3, 3);
        BoundaryData bd = make_uniform_boundaries(*mesh, BoundaryType::EXTRAPOLATION);
        auto h_image = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), bd.face_image);
        uint32_t n_moved = 0;
        for (uint32_t f = 0; f < mesh->n_faces; f++) {
            if (mesh->h_cells_of_face(f, 1) >= 0) continue;
            const int32_t c = mesh->h_cells_of_face(f, 0);
            rtype n_in[3], depth = 0.0;
            for (int k = 0; k < 3; k++) n_in[k] = -mesh->h_face_normals(f, k) / mesh->h_face_area(f);
            for (uint32_t j = 0; j < mesh->h_n_nodes_of_cell(c); j++) {
                const uint32_t node = mesh->h_node_of_cell(c, j);
                rtype d = 0.0;
                for (int k = 0; k < 3; k++) d += (mesh->h_node_coords(node, k) - mesh->h_face_coords(f, k)) * n_in[k];
                depth = std::max(depth, d);
            }
            const int32_t img = h_image(f);
            ASSERT_GE(img, 0) << type;
            if (img != c) n_moved++;
            // The image point, 3/4 of the boundary cell's depth inward, is inside the image cell
            for (int k = 0; k < 3; k++) {
                const rtype p = mesh->h_face_coords(f, k) + 0.75_r * depth * n_in[k];
                rtype lo = 1e30, hi = -1e30;
                for (uint32_t j = 0; j < mesh->h_n_nodes_of_cell(img); j++) {
                    lo = std::min(lo, mesh->h_node_coords(mesh->h_node_of_cell(img, j), k));
                    hi = std::max(hi, mesh->h_node_coords(mesh->h_node_of_cell(img, j), k));
                }
                EXPECT_GE(p, double(lo) - roundoff(1e-12)) << type << " face " << f;
                EXPECT_LE(p, double(hi) + roundoff(1e-12)) << type << " face " << f;
            }
        }
        // Boundary tetrahedra taper away from the face, so some image points leave them
        if (std::string(type) == "cartesian_tet") {
            EXPECT_GT(n_moved, 0u);
        }
    }
}

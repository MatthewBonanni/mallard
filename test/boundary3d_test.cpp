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
#include "solver.h"

namespace {

constexpr rtype GAMMA = 1.4_r;
constexpr rtype R_GAS = 1.0_r / 1.4_r;

const rtype N[3] = {2.0_r / 7.0_r, 3.0_r / 7.0_r, 6.0_r / 7.0_r};
const rtype W_I[N_CONSERVATIVE] = {1.2_r, 0.3_r, -0.5_r, 0.4_r, 0.9_r};

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
    const rtype u_wall[3] = {0.2_r, -0.1_r, 0.7_r};
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
    EXPECT_RTYPE_EQ(W_g[4], 0.6_r);
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
    for (const rtype sign : {1.0_r, -1.0_r}) {
        // Subsonic outflow (sign = 1) and inflow (sign = -1) through n
        rtype W_i[N_CONSERVATIVE] = {1.3_r, 0.0_r, 0.0_r, 0.0_r, 1.05_r};
        const rtype tang[3] = {3.0_r / 7.0_r, -6.0_r / 7.0_r, 2.0_r / 7.0_r};
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
    const rtype h[3] = {0.5_r, 0.25_r, 0.4_r};
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

TEST(Boundary3DTest, CharacteristicNeighborsContinueAcrossPeriodicSeams) {
    // The transverse terms fit the faces around each characteristic face; at a
    // periodic seam, a one-sided fit drove an outlet's pressure away with
    // turbulence leaving through it
    auto mesh = std::make_shared<Mesh>();
    mesh->init(parse_toml("[mesh]\ntype = \"cartesian\"\nNx = 3\nNy = 4\nNz = 5\nLx = 1.5\nLy = 1.0\nLz = 2.0\n"
                          "periodic = [\"y\", \"z\"]\n"));
    mesh->copy_host_to_device();
    BoundaryData bd = make_uniform_boundaries(*mesh, BoundaryType::NSCBC_OUTLET);
    auto h_faces = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), bd.char_faces);
    auto h_offsets = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), bd.char_offsets);
    auto h_dx = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), bd.char_neighbor_dx);
    ASSERT_EQ(h_faces.extent(0), 2u * 4 * 5);
    for (uint32_t k = 0; k < h_faces.extent(0); k++) {
        // Every face of the x planes has the 4 neighbors sharing an edge in the periodic 4 x 5 lattice, one
        // spacing away along y or z
        EXPECT_EQ(h_offsets(k + 1) - h_offsets(k), 4u) << "face " << h_faces(k);
        for (uint32_t j = h_offsets(k); j < h_offsets(k + 1); j++) {
            const double dy = std::abs(double(h_dx(j, 1))), dz = std::abs(double(h_dx(j, 2)));
            const double tol = double(precision_tol(1e-12, 1e-5));
            EXPECT_TRUE((std::abs(dy - 0.25) < tol && dz < tol) || (dy < tol && std::abs(dz - 0.4) < tol))
                << "face " << h_faces(k) << ": " << dy << ", " << dz;
            EXPECT_NEAR(double(h_dx(j, 0)), 0.0, tol);
        }
    }
}

TEST(Boundary3DTest, TransmissiveImageCellContainsTheImagePoint) {
    for (const char * type : {"cartesian_tet", "cartesian_prism", "cartesian_pyramid", "cartesian_mixed"}) {
        auto mesh = make_mesh_3d(type, 3, 3, 3);
        BoundaryData bd = make_uniform_boundaries(*mesh, BoundaryType::EXTRAPOLATION);
        auto h_image = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), bd.face_image);
        uint32_t n_moved = 0;
        for (uint32_t f = 0; f < mesh->n_faces; f++) {
            if (mesh->h_cells_of_face(f, 1) >= 0) continue;
            const uint32_t c = static_cast<uint32_t>(mesh->h_cells_of_face(f, 0));
            rtype n_in[3], depth = 0.0;
            for (int k = 0; k < 3; k++) n_in[k] = -mesh->h_face_normals(f, k) / mesh->h_face_area(f);
            for (uint32_t j = 0; j < mesh->h_n_nodes_of_cell(c); j++) {
                const uint32_t node = mesh->h_node_of_cell(c, j);
                rtype d = 0.0;
                for (int k = 0; k < 3; k++) d += (mesh->h_node_coords(node, k) - mesh->h_face_coords(f, k)) * n_in[k];
                depth = std::max(depth, d);
            }
            const int32_t img_index = h_image(f);
            ASSERT_GE(img_index, 0) << type;
            const uint32_t img = static_cast<uint32_t>(img_index);
            if (img != c) n_moved++;
            // The image point, 3/4 of the boundary cell's depth inward, is inside the image cell
            for (int k = 0; k < 3; k++) {
                const rtype p = mesh->h_face_coords(f, k) + 0.75_r * depth * n_in[k];
                rtype lo = 1e30_r, hi = -1e30_r;
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

namespace {

/**
 * @brief Largest |p' - u'| (rho = c = 1) of the wave reflected by the right
 *        boundary at t = 1, from a right-running pulse of 1e-3 that reaches it
 *        at t = 0.5, in a box of 200 x 2 x 2 blocks with symmetry sides.
 */
double reflection_3d(const std::string & mesh, const std::string & right) {
    const std::string pulse = "1e-3 * exp(-((x - 0.5) / 0.05)^2)";
    const std::string p0 = "0.7142857142857143";
    std::string input = "[run]\nt_stop = 1.0\ncfl = 0.2\n[mesh]\ntype = \"" + mesh +
                        "\"\nNx = 200\nNy = 2\nNz = 2\nLx = 1.0\nLy = 0.01\nLz = 0.01\n"
                        "[initialize]\ntype = \"analytical\"\nrho = \"1.0 + " + pulse + "\"\nu = [\"" + pulse +
                        "\", \"0.0\", \"0.0\"]\np = \"" + p0 + " + " + pulse + "\"\n"
                        "[[boundaries]]\nname = \"left\"\ntype = \"extrapolation\"\n"
                        "[[boundaries]]\nname = \"right\"\n" + right;
    for (const char * side : {"bottom", "top", "back", "front"}) {
        input += std::string("[[boundaries]]\nname = \"") + side + "\"\ntype = \"symmetry\"\n";
    }
    input += "[numerics]\nriemann_solver = \"HLLC\"\n[numerics.face_reconstruction]\ntype = \"MUSCL\"\n"
             "[physics]\ntype = \"euler\"\ngamma = 1.4\np_ref = " + p0 + "\nT_ref = " + p0 +
             "\nrho_ref = 1.0\n[output]\ncheck_interval = 1000000\n";
    Solver solver;
    solver.init(parse_toml(input));
    solver.run();
    solver.update_primitives();
    solver.copy_device_to_host();
    double reflected = 0.0;
    for (uint32_t c = 0; c < solver.get_mesh()->n_owned(); c++) {
        const double x = double(solver.get_mesh()->h_cell_coords(c, 0));
        if (x < 0.2 || x > 0.8) continue;
        const double p = double(solver.h_primitives(c, N_DIM)) - 1.0 / 1.4;
        reflected = std::max(reflected, std::abs(p - double(solver.h_primitives(c, 0))));
    }
    return reflected / 2e-3;
}

} // namespace

TEST(Boundary3DTest, CharacteristicOutletLetsAPlaneWaveLeave) {
    // With the transverse terms, on hexahedra and on tetrahedra
    for (const std::string mesh : {"cartesian", "cartesian_tet"}) {
        SCOPED_TRACE(mesh);
        EXPECT_LT(reflection_3d(mesh, "type = \"nscbc_outlet\"\np = 0.7142857142857143\nL = 1.0\nsigma = 0.0\n"),
                  0.03);
        EXPECT_GT(reflection_3d(mesh, "type = \"p_out\"\np = 0.7142857142857143\n"), 0.9);
    }
}

TEST(Boundary3DTest, CharacteristicOutletSurvivesEddiesThatReverseTheFlow) {
    // Eddies four times faster than the mean flow cross the outlet with local
    // backflow, in a box with symmetry sides. The transverse terms, fitted over
    // corner neighbors and applied at edges and on reversed faces, drove the
    // outlet's incoming waves away until the run diverged (t = 9.8).
    const std::string p0 = "0.7142857142857143";
    const std::string k = "6.283185307179586", k2 = "12.566370614359172";
    const std::string u = "0.2 + 0.8 * sin(" + k + " * x) * cos(" + k + " * y + 0.3) * cos(" + k +
                          " * z + 0.7) + 0.4 * sin(" + k2 + " * y + 1.1) * cos(" + k2 + " * z)";
    const std::string v = "-0.8 * cos(" + k + " * x) * sin(" + k + " * y + 0.3) * cos(" + k +
                          " * z + 0.7) + 0.4 * sin(" + k2 + " * z + 0.4) * cos(" + k2 + " * x)";
    const std::string w = "0.4 * sin(" + k2 + " * x + 2.0) * cos(" + k2 + " * y + 1.1)";
    std::string input = "[run]\nt_stop = 12.0\ncfl = 0.4\n[mesh]\ntype = \"cartesian\"\nNx = 24\nNy = 12\nNz = 12\n"
                        "Lx = 2.0\nLy = 1.0\nLz = 1.0\n[initialize]\ntype = \"analytical\"\nrho = \"1.0\"\n"
                        "u = [\"" + u + "\", \"" + v + "\", \"" + w + "\"]\np = \"" + p0 + "\"\n"
                        "[[boundaries]]\nname = \"left\"\ntype = \"nscbc_inlet\"\nu = [0.2, 0.0, 0.0]\np = " + p0 +
                        "\nT = " + p0 + "\nL = 1.0\n"
                        "[[boundaries]]\nname = \"right\"\ntype = \"nscbc_outlet\"\np = " + p0 + "\nL = 12.0\n";
    for (const char * side : {"bottom", "top", "back", "front"}) {
        input += std::string("[[boundaries]]\nname = \"") + side + "\"\ntype = \"symmetry\"\n";
    }
    input += "[numerics]\nriemann_solver = \"HLLC\"\ntime_integrator = \"SSPRK3\"\ncheck_nan = true\n"
             "[numerics.face_reconstruction]\ntype = \"MUSCL\"\nlimiter = \"none\"\n"
             "[physics]\ntype = \"navier_stokes\"\ngamma = 1.4\np_ref = " + p0 + "\nT_ref = " + p0 +
             "\nrho_ref = 1.0\nmu = 2e-4\nPr = 0.72\n[output]\ncheck_interval = 1000000\n";
    Solver solver;
    solver.init(parse_toml(input));
    ASSERT_NO_THROW(solver.run());
    solver.update_primitives();
    solver.copy_device_to_host();
    double max_speed = 0.0, max_dp = 0.0;
    for (uint32_t c = 0; c < solver.get_mesh()->n_owned(); c++) {
        double speed2 = 0.0;
        for (int i = 0; i < N_DIM; i++) speed2 += std::pow(double(solver.h_primitives(c, i)), 2);
        max_speed = std::max(max_speed, std::sqrt(speed2));
        max_dp = std::max(max_dp, std::abs(double(solver.h_primitives(c, N_DIM)) - 1.0 / 1.4));
    }
    // Bounded: the inflow is 0.2, and the weakly held mean pressure (L = 12) is still settling
    EXPECT_LT(max_speed, 0.6);
    EXPECT_LT(max_dp, 0.15);
}

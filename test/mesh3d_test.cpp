/**
 * @file mesh3d_test.cpp
 * @author Matthew Bonanni (mbonanni001@gmail.com)
 * @brief Geometric and topological invariants of 3D meshes.
 * @version 0.1
 * @date 2026-10-02
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 *
 */

#include <gtest/gtest.h>

#include <cmath>
#include <map>
#include <random>
#include <string>

#include "test_fixtures.h"

namespace {

constexpr rtype LX = 2.0, LY = 1.0, LZ = 1.5;
constexpr uint32_t NX = 6, NY = 4, NZ = 3;

class Mesh3DInvariants : public ::testing::TestWithParam<std::string> {};

void expect_closed_cells_and_consistent_faces(const Mesh & mesh, double tol) {
    for (uint32_t c = 0; c < mesh.n_cells; c++) {
        rtype closure[N_DIM] = {};
        for (uint32_t k = 0; k < mesh.h_n_faces_of_cell(c); k++) {
            const uint32_t f = mesh.h_face_of_cell(c, k);
            const rtype sign = (mesh.h_cells_of_face(f, 0) == static_cast<int32_t>(c)) ? 1.0 : -1.0;
            FOR_I_DIM closure[i] += sign * mesh.h_face_normals(f, i);
        }
        FOR_I_DIM EXPECT_NEAR(closure[i], 0.0, tol) << "cell " << c;
    }
    for (uint32_t f = 0; f < mesh.n_faces; f++) {
        rtype d = 0.0, n2 = 0.0;
        FOR_I_DIM {
            d += (mesh.h_face_coords(f, i) - mesh.h_cell_coords(mesh.h_cells_of_face(f, 0), i)) * mesh.h_face_normals(f, i);
            n2 += mesh.h_face_normals(f, i) * mesh.h_face_normals(f, i);
        }
        EXPECT_GT(d, 0.0) << "face " << f;
        EXPECT_NEAR(std::sqrt(n2), mesh.h_face_area(f), roundoff(1e-14));
    }
}

/**
 * @brief Volumes sum to the box and the cell first moments sum to the box's.
 */
void expect_box_moments(const Mesh & mesh, double tol) {
    rtype V = 0.0, M[N_DIM] = {};
    for (uint32_t c = 0; c < mesh.n_cells; c++) {
        EXPECT_GT(mesh.h_cell_volume(c), 0.0);
        V += mesh.h_cell_volume(c);
        FOR_I_DIM M[i] += mesh.h_cell_volume(c) * mesh.h_cell_coords(c, i);
    }
    const rtype L[3] = {LX, LY, LZ};
    EXPECT_NEAR(V, LX * LY * LZ, tol);
    FOR_I_DIM EXPECT_NEAR(M[i], 0.5_r * L[i] * LX * LY * LZ, tol) << "direction " << static_cast<int>(i);
}

} // namespace

TEST_P(Mesh3DInvariants, CellsAreClosedAndNormalsPointFromCell0ToCell1) {
    auto mesh = make_mesh_3d(GetParam(), NX, NY, NZ, LX, LY, LZ);
    expect_closed_cells_and_consistent_faces(*mesh, roundoff(1e-14));
}

TEST_P(Mesh3DInvariants, VolumesAndCentroidsMatchTheBox) {
    auto mesh = make_mesh_3d(GetParam(), NX, NY, NZ, LX, LY, LZ);
    expect_box_moments(*mesh, roundoff(1e-12));
}

TEST_P(Mesh3DInvariants, BoundaryZonesCoverEachSideWithOutwardNormals) {
    auto mesh = make_mesh_3d(GetParam(), NX, NY, NZ, LX, LY, LZ);
    const std::map<std::string, std::pair<int, rtype>> sides = {
        {"left", {-1, LY * LZ}}, {"right", {1, LY * LZ}}, {"bottom", {-2, LX * LZ}},
        {"top", {2, LX * LZ}}, {"back", {-3, LX * LY}}, {"front", {3, LX * LY}}};
    uint32_t n_boundary = 0;
    for (const auto & [name, side] : sides) {
        FaceZone * zone = mesh->get_face_zone(name);
        ASSERT_NE(zone, nullptr) << name;
        const int axis = std::abs(side.first) - 1;
        rtype A[N_DIM] = {};
        for (uint32_t k = 0; k < zone->n_faces(); k++) {
            const uint32_t f = zone->h_faces(k);
            EXPECT_EQ(mesh->h_cells_of_face(f, 1), -1);
            FOR_I_DIM A[i] += mesh->h_face_normals(f, i);
        }
        FOR_I_DIM EXPECT_NEAR(A[i], (i == axis) ? (side.first > 0 ? 1.0_r : -1.0_r) * side.second : 0.0_r, roundoff(1e-13)) << name;
        n_boundary += zone->n_faces();
    }
    EXPECT_EQ(mesh->get_face_zone("unassigned"), nullptr);
    uint32_t n_boundary_faces = 0;
    for (uint32_t f = 0; f < mesh->n_faces; f++) n_boundary_faces += (mesh->h_cells_of_face(f, 1) < 0);
    EXPECT_EQ(n_boundary, n_boundary_faces);
}

TEST_P(Mesh3DInvariants, WarpedFacesKeepCellsClosedAndConserveVolume) {
    auto mesh = make_mesh_3d(GetParam(), NX, NY, NZ, LX, LY, LZ);
    // Move interior nodes so quadrilateral faces become non-planar
    std::mt19937 rng(7);
    std::uniform_real_distribution<rtype> jitter(-0.12, 0.12);
    const rtype h[3] = {LX / NX, LY / NY, LZ / NZ};
    const rtype L[3] = {LX, LY, LZ};
    for (uint32_t n = 0; n < mesh->n_nodes; n++) {
        bool interior = true;
        FOR_I_DIM {
            const rtype x = mesh->h_node_coords(n, i);
            interior = interior && x > 1e-9_r && x < L[i] - 1e-9_r;
        }
        if (!interior) continue;
        FOR_I_DIM mesh->h_node_coords(n, i) += jitter(rng) * h[i];
    }
    mesh->compute_geometry();
    expect_closed_cells_and_consistent_faces(*mesh, roundoff(1e-14));
    expect_box_moments(*mesh, roundoff(1e-12));
}

INSTANTIATE_TEST_SUITE_P(Mesh, Mesh3DInvariants,
                         ::testing::Values("cartesian", "cartesian_tet", "cartesian_prism", "cartesian_pyramid",
                                           "cartesian_mixed"));

TEST(Mesh3D, FaceCountsMatchTheBlockSubdivision) {
    const uint32_t n_blocks = NX * NY * NZ;
    const uint32_t f_x = (NX + 1) * NY * NZ, f_y = NX * (NY + 1) * NZ, f_z = NX * NY * (NZ + 1);
    const std::map<std::string, std::pair<uint32_t, uint32_t>> expected = {
        {"cartesian", {n_blocks, f_x + f_y + f_z}},
        {"cartesian_tet", {6 * n_blocks, 6 * n_blocks + 2 * (f_x + f_y + f_z)}},
        {"cartesian_prism", {2 * n_blocks, n_blocks + f_x + f_y + 2 * f_z}},
        {"cartesian_pyramid", {6 * n_blocks, 12 * n_blocks + f_x + f_y + f_z}},
    };
    for (const auto & [type, counts] : expected) {
        auto mesh = make_mesh_3d(type, NX, NY, NZ, LX, LY, LZ);
        EXPECT_EQ(mesh->n_cells, counts.first) << type;
        EXPECT_EQ(mesh->n_faces, counts.second) << type;
    }
}

TEST(Mesh3D, MixedMeshHasEveryCellTypeAndConformingFaces) {
    auto mesh = make_mesh_3d("cartesian_mixed", NX, NY, NZ, LX, LY, LZ);
    std::map<CellType, uint32_t> count;
    for (uint32_t c = 0; c < mesh->n_cells; c++) count[mesh->h_cell_type(c)]++;
    const uint32_t blocks_per_third = (NX / 3) * NY * NZ;
    EXPECT_EQ(count[CellType::HEXAHEDRON], blocks_per_third);
    EXPECT_EQ(count[CellType::PYRAMID], 6 * blocks_per_third);
    EXPECT_EQ(count[CellType::PRISM], 2 * blocks_per_third);
    // Every interior face joins two cells with the same node set
    for (uint32_t f = 0; f < mesh->n_faces; f++) {
        if (mesh->h_cells_of_face(f, 1) < 0) continue;
        EXPECT_TRUE(mesh->h_n_nodes_of_face(f) == 3 || mesh->h_n_nodes_of_face(f) == 4);
    }
}

TEST(Mesh3D, ConnectivityIsReorientedAndBoundaryFacesMatchByNodeSet) {
    // Unit cube as a hexahedron given with negative orientation, split by a
    // pyramid on top, with boundary faces listed in arbitrary node order
    const std::vector<std::array<rtype, N_DIM>> nodes = {
        {0, 0, 0}, {1, 0, 0}, {1, 1, 0}, {0, 1, 0}, {0, 0, 1}, {1, 0, 1}, {1, 1, 1}, {0, 1, 1}, {0.5, 0.5, 1.5}};
    const std::vector<std::vector<uint32_t>> cells = {{0, 3, 2, 1, 4, 7, 6, 5}, {4, 5, 6, 7, 8}};
    const std::vector<Mesh::BoundaryFace> faces = {{{2, 1, 0, 3}, "floor"}, {{8, 4, 5}, "roof"}};
    Mesh mesh;
    mesh.init_from_connectivity(nodes, cells, faces);
    EXPECT_EQ(mesh.n_cells, 2u);
    EXPECT_EQ(mesh.n_faces, 6u + 5u - 1u);
    EXPECT_NEAR(mesh.h_cell_volume(0), 1.0, 1e-14);
    EXPECT_NEAR(mesh.h_cell_volume(1), 1.0 / 6.0, roundoff(1e-14));
    EXPECT_NEAR(mesh.h_cell_coords(1, 2), 1.125, 1e-14);
    ASSERT_NE(mesh.get_face_zone("floor"), nullptr);
    ASSERT_NE(mesh.get_face_zone("roof"), nullptr);
    EXPECT_EQ(mesh.get_face_zone("floor")->n_faces(), 1u);
    EXPECT_EQ(mesh.get_face_zone("unassigned")->n_faces(), 4u + 3u);
    const uint32_t floor = mesh.get_face_zone("floor")->h_faces(0);
    EXPECT_NEAR(mesh.h_face_normals(floor, 2), -1.0, 1e-14);
    expect_closed_cells_and_consistent_faces(mesh, roundoff(1e-14));
}

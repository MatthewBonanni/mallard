/**
 * @file mesh_test.cpp
 * @author Matthew Bonanni (mbonanni001@gmail.com)
 * @brief Geometric and topological invariants of the generated meshes.
 * @version 0.1
 * @date 2026-10-02
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 *
 */

#include <gtest/gtest.h>

#include <cmath>
#include <set>
#include <string>

#include "test_fixtures.h"

namespace {

class MeshInvariants : public ::testing::TestWithParam<std::string> {};

} // namespace

TEST_P(MeshInvariants, CellsAreClosedAndNormalsPointFromCell0ToCell1) {
    auto mesh = make_mesh(GetParam(), 7, 5, 2.0, 1.0);
    for (uint32_t i_cell = 0; i_cell < mesh->n_cells; i_cell++) {
        rtype closure[N_DIM] = {0.0, 0.0};
        for (uint32_t k = 0; k < mesh->h_n_faces_of_cell(i_cell); k++) {
            const uint32_t f = mesh->h_face_of_cell(i_cell, k);
            const rtype sign = (mesh->h_cells_of_face(f, 0) == static_cast<int32_t>(i_cell)) ? 1.0 : -1.0;
            FOR_I_DIM closure[i] += sign * mesh->h_face_normals(f, i);
        }
        FOR_I_DIM EXPECT_NEAR(closure[i], 0.0, 1e-14) << "cell " << i_cell;
    }
    for (uint32_t f = 0; f < mesh->n_faces; f++) {
        const int32_t c0 = mesh->h_cells_of_face(f, 0);
        rtype d = 0.0;
        FOR_I_DIM d += (mesh->h_face_coords(f, i) - mesh->h_cell_coords(c0, i)) * mesh->h_face_normals(f, i);
        EXPECT_GT(d, 0.0) << "face " << f;
        const rtype n_mag = std::hypot(mesh->h_face_normals(f, 0), mesh->h_face_normals(f, 1));
        EXPECT_NEAR(n_mag, mesh->h_face_area(f), 1e-14);
    }
}

TEST_P(MeshInvariants, VolumesSumToDomainAreaAndCentroidsAreExact) {
    auto mesh = make_mesh(GetParam(), 7, 5, 2.0, 1.0);
    rtype total = 0.0;
    for (uint32_t i = 0; i < mesh->n_cells; i++) {
        EXPECT_GT(mesh->h_cell_volume(i), 0.0);
        total += mesh->h_cell_volume(i);
    }
    if (GetParam() != "wedge") {
        EXPECT_NEAR(total, 2.0, 1e-13);
    }
    // Centroid check: integral of x over each cell via its nodes (shoelace moments)
    for (uint32_t i = 0; i < mesh->n_cells; i++) {
        const uint32_t n = mesh->h_n_nodes_of_cell(i);
        rtype A = 0.0, Cx = 0.0, Cy = 0.0;
        for (uint32_t k = 0; k < n; k++) {
            const uint32_t a = mesh->h_node_of_cell(i, k), b = mesh->h_node_of_cell(i, (k + 1) % n);
            const rtype xa = mesh->h_node_coords(a, 0), ya = mesh->h_node_coords(a, 1);
            const rtype xb = mesh->h_node_coords(b, 0), yb = mesh->h_node_coords(b, 1);
            const rtype cross = xa * yb - xb * ya;
            A += 0.5 * cross;
            Cx += (xa + xb) * cross / 6.0;
            Cy += (ya + yb) * cross / 6.0;
        }
        EXPECT_NEAR(std::abs(A), mesh->h_cell_volume(i), 1e-14);
        EXPECT_NEAR(Cx / A, mesh->h_cell_coords(i, 0), 1e-13) << "cell " << i;
        EXPECT_NEAR(Cy / A, mesh->h_cell_coords(i, 1), 1e-13) << "cell " << i;
    }
}

TEST_P(MeshInvariants, BoundaryZonesPartitionBoundaryFaces) {
    auto mesh = make_mesh(GetParam(), 6, 4);
    std::multiset<uint32_t> zone_faces;
    for (auto & zone : *mesh->face_zones()) {
        if (zone.get_type() != FaceZoneType::BOUNDARY) continue;
        for (uint32_t i = 0; i < zone.n_faces(); i++) zone_faces.insert(zone.h_faces(i));
    }
    uint32_t n_boundary = 0;
    for (uint32_t f = 0; f < mesh->n_faces; f++) {
        const bool is_boundary = mesh->h_cells_of_face(f, 1) < 0;
        n_boundary += is_boundary;
        EXPECT_EQ(zone_faces.count(f), is_boundary ? 1u : 0u) << "face " << f;
    }
    EXPECT_EQ(n_boundary, 2u * (6 + 4));
}

TEST_P(MeshInvariants, InteriorZoneHoldsEveryInteriorFaceOnce) {
    auto mesh = make_mesh(GetParam(), 6, 4);
    FaceZone * interior = mesh->get_face_zone("interior");
    ASSERT_NE(interior, nullptr);
    std::set<uint32_t> faces;
    for (uint32_t i = 0; i < interior->n_faces(); i++) faces.insert(interior->h_faces(i));
    EXPECT_EQ(faces.size(), interior->n_faces());
    for (uint32_t f = 0; f < mesh->n_faces; f++) {
        EXPECT_EQ(faces.count(f), mesh->h_cells_of_face(f, 1) >= 0 ? 1u : 0u);
    }
}

TEST(MeshZones, BoxZonesLieOnTheirEdges) {
    for (const char * type : {"cartesian", "cartesian_tri"}) {
        auto mesh = make_mesh(type, 6, 4, 2.0, 1.0);
        const std::pair<const char *, std::pair<int, rtype>> edges[] = {
            {"left", {0, 0.0}}, {"right", {0, 2.0}}, {"bottom", {1, 0.0}}, {"top", {1, 1.0}}};
        for (const auto & [name, edge] : edges) {
            FaceZone * zone = mesh->get_face_zone(name);
            ASSERT_NE(zone, nullptr) << type << " " << name;
            EXPECT_EQ(zone->n_faces(), edge.first == 0 ? 4u : 6u) << type << " " << name;
            for (uint32_t i = 0; i < zone->n_faces(); i++) {
                EXPECT_NEAR(mesh->h_face_coords(zone->h_faces(i), edge.first), edge.second, 1e-14)
                    << type << " " << name;
            }
        }
    }
}

INSTANTIATE_TEST_SUITE_P(Mesh, MeshInvariants,
                         ::testing::Values("cartesian", "cartesian_tri", "wedge"));

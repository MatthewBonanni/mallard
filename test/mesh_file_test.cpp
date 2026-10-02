/**
 * @file mesh_file_test.cpp
 * @author Matthew Bonanni (mbonanni001@gmail.com)
 * @brief Gmsh reader and solver runs on unstructured mixed meshes.
 * @version 0.1
 * @date 2026-10-02
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 *
 */

#include <gtest/gtest.h>
#include <Kokkos_Core.hpp>

#include <cmath>
#include <filesystem>
#include <fstream>
#include <random>
#include <sstream>
#include <string>

#include "test_fixtures.h"
#include "solver.h"

namespace {

// [0, 2] x [0, 1]: two triangles on the left, one quad on the right
const char * MSH22 = R"($MeshFormat
2.2 0 8
$EndMeshFormat
$PhysicalNames
4
1 1 "bottom"
1 2 "right"
1 3 "top"
1 4 "left"
$EndPhysicalNames
$Nodes
6
1 0 0 0
2 1 0 0
3 2 0 0
4 2 1 0
5 1 1 0
6 0 1 0
$EndNodes
$Elements
9
1 1 2 1 1 1 2
2 1 2 1 1 2 3
3 1 2 2 2 3 4
4 1 2 3 3 4 5
5 1 2 3 3 5 6
6 1 2 4 4 6 1
7 2 2 0 1 1 2 5
8 2 2 0 1 1 5 6
9 3 2 0 2 2 3 4 5
$EndElements
)";

const char * MSH41 = R"($MeshFormat
4.1 0 8
$EndMeshFormat
$PhysicalNames
4
1 1 "bottom"
1 2 "right"
1 3 "top"
1 4 "left"
$EndPhysicalNames
$Entities
0 4 1 0
1 0 0 0 2 0 0 1 1 0
2 2 0 0 2 1 0 1 2 0
3 0 1 0 2 1 0 1 3 0
4 0 0 0 0 1 0 1 4 0
1 0 0 0 2 1 0 0 0
$EndEntities
$Nodes
1 6 1 6
2 1 0 6
1
2
3
4
5
6
0 0 0
1 0 0
2 0 0
2 1 0
1 1 0
0 1 0
$EndNodes
$Elements
6 9 1 9
1 1 1 2
1 1 2
2 2 3
1 2 1 1
3 3 4
1 3 1 2
4 4 5
5 5 6
1 4 1 1
6 6 1
2 1 2 2
7 1 2 5
8 1 5 6
2 1 3 1
9 2 3 4 5
$EndElements
)";

std::string write_temp(const std::string & name, const std::string & content) {
    const auto path = std::filesystem::temp_directory_path() / name;
    std::ofstream(path) << content;
    return path.string();
}

void check_small_mesh(const std::string & file) {
    Mesh mesh;
    mesh.init_file(file);
    EXPECT_EQ(mesh.n_cells, 3u);
    EXPECT_EQ(mesh.n_faces, 8u);
    rtype total = 0.0;
    for (uint32_t c = 0; c < mesh.n_cells; c++) total += mesh.h_cell_volume(c);
    EXPECT_NEAR(total, 2.0, 1e-14);
    const std::pair<const char *, uint32_t> zones[] = {{"bottom", 2}, {"right", 1}, {"top", 2}, {"left", 1}};
    for (const auto & [name, n] : zones) {
        FaceZone * zone = mesh.get_face_zone(name);
        ASSERT_NE(zone, nullptr) << name;
        EXPECT_EQ(zone->n_faces(), n) << name;
    }
    EXPECT_EQ(mesh.get_face_zone("unassigned"), nullptr);
    EXPECT_EQ(mesh.get_face_zone("interior")->n_faces(), 2u);
}

/**
 * @brief Gmsh 2.2 file of the unit square: columns alternate between quads and
 *        pairs of triangles, and interior nodes are randomly displaced.
 */
std::string jittered_mixed_mesh(uint32_t n) {
    std::mt19937 rng(42);
    std::uniform_real_distribution<double> jitter(-0.2, 0.2);
    std::ostringstream s;
    s << "$MeshFormat\n2.2 0 8\n$EndMeshFormat\n$PhysicalNames\n4\n"
      << "1 1 \"bottom\"\n1 2 \"right\"\n1 3 \"top\"\n1 4 \"left\"\n$EndPhysicalNames\n";
    auto id = [&](uint32_t i, uint32_t j) { return j * (n + 1) + i + 1; };
    s << "$Nodes\n" << (n + 1) * (n + 1) << "\n";
    for (uint32_t j = 0; j <= n; j++) {
        for (uint32_t i = 0; i <= n; i++) {
            const bool interior = i > 0 && i < n && j > 0 && j < n;
            const double x = (i + (interior ? jitter(rng) : 0.0)) / n;
            const double y = (j + (interior ? jitter(rng) : 0.0)) / n;
            s << id(i, j) << " " << x << " " << y << " 0\n";
        }
    }
    std::vector<std::string> elements;
    for (uint32_t i = 0; i < n; i++) {
        elements.push_back("1 2 1 1 " + std::to_string(id(i, 0)) + " " + std::to_string(id(i + 1, 0)));
        elements.push_back("1 2 3 3 " + std::to_string(id(i + 1, n)) + " " + std::to_string(id(i, n)));
        elements.push_back("1 2 4 4 " + std::to_string(id(0, i + 1)) + " " + std::to_string(id(0, i)));
        elements.push_back("1 2 2 2 " + std::to_string(id(n, i)) + " " + std::to_string(id(n, i + 1)));
    }
    for (uint32_t j = 0; j < n; j++) {
        for (uint32_t i = 0; i < n; i++) {
            const auto a = std::to_string(id(i, j)), b = std::to_string(id(i + 1, j));
            const auto c = std::to_string(id(i + 1, j + 1)), d = std::to_string(id(i, j + 1));
            if (i % 2 == 0) {
                elements.push_back("3 2 0 1 " + a + " " + b + " " + c + " " + d);
            } else {
                elements.push_back("2 2 0 1 " + a + " " + b + " " + c);
                elements.push_back("2 2 0 1 " + a + " " + c + " " + d);
            }
        }
    }
    s << "$EndNodes\n$Elements\n" << elements.size() << "\n";
    for (size_t k = 0; k < elements.size(); k++) s << k + 1 << " " << elements[k] << "\n";
    s << "$EndElements\n";
    return s.str();
}

std::string file_mesh_input(const std::string & file, const std::string & recon, const std::string & init,
                            const std::string & bc, const std::string & run) {
    std::ostringstream s;
    s << "[run]\n" << run
      << "[mesh]\ntype = \"file\"\nfilename = \"" << file << "\"\n"
      << "[initialize]\n" << init;
    for (const char * name : {"left", "right", "top", "bottom"}) {
        s << "[[boundaries]]\nname = \"" << name << "\"\ntype = \"" << bc << "\"\n";
    }
    s << "[numerics]\nriemann_solver = \"HLLC\"\ntime_integrator = \"SSPRK3\"\ncheck_nan = true\n"
      << "[numerics.face_reconstruction]\ntype = \"" << recon << "\"\n"
      << "[physics]\ntype = \"euler\"\ngamma = 1.4\np_ref = 1.0\nT_ref = 1.0\nrho_ref = 1.0\n"
      << "[output]\ncheck_interval = 1000000\n";
    return s.str();
}

class MixedMesh : public ::testing::TestWithParam<std::string> {};

} // namespace

TEST(MeshFileTest, ReadsGmsh22MixedMesh) {
    check_small_mesh(write_temp("mallard_small_22.msh", MSH22));
}

TEST(MeshFileTest, Gmsh22IgnoresUntaggedCurves) {
    // An untagged line (physical 0) along the interior edge 2-5
    std::string msh = MSH22;
    msh.replace(msh.find("$Elements\n9\n"), 12, "$Elements\n10\n");
    msh.replace(msh.find("$EndElements"), 12, "10 1 2 0 0 2 5\n$EndElements");
    check_small_mesh(write_temp("mallard_untagged_22.msh", msh));
}

TEST(MeshFileTest, ReadsGmsh41MixedMesh) {
    check_small_mesh(write_temp("mallard_small_41.msh", MSH41));
}

TEST(MeshFileTest, JitteredMixedMeshSatisfiesInvariants) {
    Mesh mesh;
    mesh.init_file(write_temp("mallard_jitter.msh", jittered_mixed_mesh(12)));
    EXPECT_EQ(mesh.n_cells, 6u * 12 + 6u * 12 * 2);
    rtype total = 0.0;
    for (uint32_t c = 0; c < mesh.n_cells; c++) {
        EXPECT_GT(mesh.h_cell_volume(c), 0.0);
        total += mesh.h_cell_volume(c);
        rtype closure[N_DIM] = {0.0, 0.0};
        for (uint32_t k = 0; k < mesh.h_n_faces_of_cell(c); k++) {
            const uint32_t f = mesh.h_face_of_cell(c, k);
            const rtype sign = (mesh.h_cells_of_face(f, 0) == (int32_t)c) ? 1.0 : -1.0;
            FOR_I_DIM closure[i] += sign * mesh.h_face_normals(f, i);
        }
        FOR_I_DIM EXPECT_NEAR(closure[i], 0.0, 1e-14);
    }
    EXPECT_NEAR(total, 1.0, 1e-13);
}

TEST_P(MixedMesh, UniformFlowIsPreserved) {
    const std::string file = write_temp("mallard_jitter_fs.msh", jittered_mixed_mesh(12));
    Solver solver;
    solver.init(parse_toml(file_mesh_input(file, GetParam(),
        "type = \"analytical\"\nrho = \"1.2\"\nu = [\"0.4\", \"-0.3\"]\np = \"0.8\"\n",
        "extrapolation", "n_steps = 20\ncfl = 0.5\n")));
    solver.run();
    solver.update_primitives();
    solver.copy_device_to_host();
    for (uint32_t i = 0; i < solver.get_mesh()->n_cells; i++) {
        EXPECT_NEAR(solver.h_conservatives(i, 0), 1.2, 1e-12);
        EXPECT_NEAR(solver.h_primitives(i, 0), 0.4, 1e-12);
        EXPECT_NEAR(solver.h_primitives(i, 2), 0.8, 1e-12);
    }
}

TEST_P(MixedMesh, BlastInClosedBoxConservesMassAndEnergy) {
    const std::string file = write_temp("mallard_jitter_blast.msh", jittered_mixed_mesh(16));
    Solver solver;
    solver.init(parse_toml(file_mesh_input(file, GetParam(),
        "type = \"analytical\"\nrho = \"1.0\"\nu = [\"0.0\", \"0.0\"]\n"
        "p = \"(x - 0.5)^2 + (y - 0.5)^2 < 0.04 ? 10.0 : 0.1\"\n",
        "symmetry", "t_stop = 0.1\ncfl = 0.5\n")));
    const auto before = solver.integrate_conservatives();
    solver.run();
    const auto after = solver.integrate_conservatives();
    EXPECT_NEAR(after[0], before[0], 1e-12);
    EXPECT_NEAR(after[3], before[3], 1e-11);
}

INSTANTIATE_TEST_SUITE_P(MeshFile, MixedMesh, ::testing::Values("FO", "MUSCL", "TENO"));

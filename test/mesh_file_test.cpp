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
#include <sstream>
#include <string>

#include "comm.h"
#include "gmsh_fixtures.h"
#include "mesh_block.h"
#include "periodic_fixtures.h"
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

TEST(MeshFileTest, HDF5ConversionKeepsCellsNodesAndZones) {
    if (!have_hdf5()) GTEST_SKIP() << "built without HDF5";
    if (comm::size() > 1) GTEST_SKIP() << "converts on one rank";
    const std::string msh = write_temp("mallard_convert.msh", jittered_mixed_mesh(6));
    const std::string h5 = (std::filesystem::temp_directory_path() / "mallard_convert.h5").string();
    write_mesh_h5(h5, read_gmsh_block(msh));
    Mesh gmsh, hdf5;
    gmsh.init_file(msh);
    hdf5.init_file(h5);
    ASSERT_EQ(hdf5.n_cells, gmsh.n_cells);
    ASSERT_EQ(hdf5.n_faces, gmsh.n_faces);
    for (uint32_t c = 0; c < gmsh.n_cells; c++) {
        ASSERT_EQ(hdf5.h_n_nodes_of_cell(c), gmsh.h_n_nodes_of_cell(c));
        for (uint32_t k = 0; k < gmsh.h_n_nodes_of_cell(c); k++) {
            FOR_I_DIM EXPECT_EQ(hdf5.h_node_coords(hdf5.h_node_of_cell(c, k), i),
                                gmsh.h_node_coords(gmsh.h_node_of_cell(c, k), i));
        }
    }
    for (FaceZone & zone : *gmsh.face_zones()) {
        FaceZone * other = hdf5.get_face_zone(zone.get_name());
        ASSERT_NE(other, nullptr) << zone.get_name();
        ASSERT_EQ(other->n_faces(), zone.n_faces()) << zone.get_name();
        for (uint32_t i = 0; i < zone.n_faces(); i++) EXPECT_EQ(other->h_faces(i), zone.h_faces(i));
    }
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
            const rtype sign = (mesh.h_cells_of_face(f, 0) == static_cast<int32_t>(c)) ? 1.0 : -1.0;
            FOR_I_DIM closure[i] += sign * mesh.h_face_normals(f, i);
        }
        FOR_I_DIM EXPECT_NEAR(closure[i], 0.0, roundoff(1e-14));
    }
    EXPECT_NEAR(total, 1.0, roundoff(1e-13));
}

TEST_P(MixedMesh, UniformFlowIsPreserved) {
    const std::string file = write_temp("mallard_jitter_fs.msh", jittered_mixed_mesh(12));
    Solver solver;
    solver.init(parse_toml(file_mesh_input(file, GetParam(),
        "type = \"analytical\"\nrho = \"1.2\"\nu = [\"0.4\", \"-0.3\"]\np = \"0.8\"\n",
        "extrapolation", "n_steps = 20\ncfl = 0.25\n")));
    solver.run();
    solver.update_primitives();
    solver.copy_device_to_host();
    for (uint32_t i = 0; i < solver.get_mesh()->n_cells; i++) {
        EXPECT_NEAR(solver.h_conservatives(i, 0), 1.2, roundoff(1e-12));
        EXPECT_NEAR(solver.h_primitives(i, 0), 0.4, roundoff(1e-12));
        EXPECT_NEAR(solver.h_primitives(i, 2), 0.8, roundoff(1e-12));
    }
}

TEST_P(MixedMesh, BlastInClosedBoxConservesMassAndEnergy) {
    const std::string file = write_temp("mallard_jitter_blast.msh", jittered_mixed_mesh(16));
    Solver solver;
    solver.init(parse_toml(file_mesh_input(file, GetParam(),
        "type = \"analytical\"\nrho = \"1.0\"\nu = [\"0.0\", \"0.0\"]\n"
        "p = \"(x - 0.5)^2 + (y - 0.5)^2 < 0.04 ? 10.0 : 0.1\"\n",
        "symmetry", "t_stop = 0.1\ncfl = 0.25\n")));
    const auto before = solver.integrate_conservatives();
    solver.run();
    const auto after = solver.integrate_conservatives();
    EXPECT_NEAR(after[0], before[0], roundoff(1e-12));
    EXPECT_NEAR(after[3], before[3], roundoff(1e-11));
}

INSTANTIATE_TEST_SUITE_P(MeshFile, MixedMesh, ::testing::Values("FO", "MUSCL", "TENO"));

namespace {

const char * PERIODIC_PAIRS = "[[periodic]]\nzones = [\"left\", \"right\"]\ntranslation = [1.0, 0.0]\n"
                              "[[periodic]]\nzones = [\"bottom\", \"top\"]\ntranslation = [0.0, 1.0]\n";

std::string periodic_file_input(const std::string & file) {
    std::string input = file_mesh_input(file, "TENO",
        "type = \"analytical\"\nrho = \"1.0 + 0.3 * exp(-30 * ((x - 0.3)^2 + (y - 0.6)^2))\"\n"
        "u = [\"0.7\", \"-0.4\"]\np = \"1.0\"\n", "symmetry", "n_steps = 15\ncfl = 0.25\n");
    return input.substr(0, input.find("[[boundaries]]")) + input.substr(input.find("[numerics]")) + PERIODIC_PAIRS;
}

std::string periodic_mesh_error(const std::string & file, const std::string & pairs) {
    try {
        Mesh().init(parse_toml("[mesh]\ntype = \"file\"\nfilename = \"" + file + "\"\n" + pairs));
    } catch (const std::runtime_error & e) {
        return e.what();
    }
    return "";
}

} // namespace

TEST(MeshFileTest, PeriodicZonePairsJoinAJitteredGmshMesh) {
    // The fully periodic box has no boundary faces and conserves momentum too
    const std::string file = write_temp("mallard_jitter_periodic.msh", jittered_mixed_mesh(12));
    Solver solver;
    solver.init(parse_toml(periodic_file_input(file)));
    const Mesh & mesh = *solver.get_mesh();
    for (uint32_t f = 0; f < mesh.n_faces; f++) EXPECT_GE(mesh.h_cells_of_face(f, 1), 0);
    expect_shifts_join_cells(mesh);
    const auto before = solver.integrate_conservatives();
    solver.run();
    const auto after = solver.integrate_conservatives();
    FOR_I_CONSERVATIVE EXPECT_NEAR(after[i], before[i], roundoff(1e-12)) << "variable " << int(i);

    if (!have_hdf5() || comm::size() > 1) return;
    const std::string h5 = (std::filesystem::temp_directory_path() / "mallard_jitter_periodic.h5").string();
    write_mesh_h5(h5, read_gmsh_block(file));
    Solver hdf5;
    hdf5.init(parse_toml(periodic_file_input(h5)));
    hdf5.run();
    solver.copy_device_to_host();
    hdf5.copy_device_to_host();
    for (uint32_t c = 0; c < mesh.n_cells; c++) {
        FOR_I_CONSERVATIVE ASSERT_EQ(hdf5.h_conservatives(c, i), solver.h_conservatives(c, i)) << "cell " << c;
    }
}

TEST(MeshFileTest, PeriodicZonesThatDoNotMatchAreRejected) {
    std::string msh = jittered_mixed_mesh(6);
    const std::string node = "\n28 1 0.5 0\n";  // node (i, j) = (6, 3), on zone right
    ASSERT_NE(msh.find(node), std::string::npos);
    msh.replace(msh.find(node), node.size(), "\n28 1 0.52 0\n");
    const std::string moved = write_temp("mallard_periodic_mismatch.msh", msh);
    EXPECT_NE(periodic_mesh_error(moved, PERIODIC_PAIRS).find("Periodic zones left and right: node (0.000000, "
                                                                "0.500000) of left has no match in right"),
              std::string::npos);

    const std::string file = write_temp("mallard_periodic_pairs.msh", jittered_mixed_mesh(6));
    EXPECT_NE(periodic_mesh_error(file, "[[periodic]]\nzones = [\"left\", \"right\"]\ntranslation = [0.9, 0.0]\n")
                  .find("has no match in right"),
              std::string::npos);
    EXPECT_NE(periodic_mesh_error(file, "[[periodic]]\nzones = [\"left\", \"rigth\"]\ntranslation = [1.0, 0.0]\n")
                  .find("Periodic zone rigth is not a boundary zone"),
              std::string::npos);
    EXPECT_NE(periodic_mesh_error(file, "[[periodic]]\nzones = [\"left\", \"right\"]\ntranslation = [1.0]\n")
                  .find("[[periodic]] needs zones = [A, B]"),
              std::string::npos);
}

/**
 * @file front_end_test.cpp
 * @brief The dimension `mallard` finds for a case, to pick mallard2d or mallard3d.
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 */

#include <gtest/gtest.h>

#include <fstream>
#include <string>

#include "data_paths.h"
#include "front_end.h"
#include "mesh_block.h"
#include "mesh_file.h"

namespace {

std::string mesh(const std::string & name) { return source_dir() + "/test/data/meshes/" + name; }

int input_dim(const std::string & mesh_table) {
    std::ofstream("input.toml") << "[mesh]\n" << mesh_table;
    return front_end::input_dimension("input.toml").dim;
}

} // namespace

TEST(FrontEndTest, MeshFilesOfEveryFormatHaveTheirDimension) {
    for (const int dim : {2, 3}) {
        const std::string cyl = "cyl" + std::to_string(dim) + "d";
        for (const std::string & name : {cyl + ".msh", cyl + "_fluent.msh", cyl + "_fluent_bin.msh", cyl + "_foam",
                                         cyl + "_foam_bin"}) {
            const front_end::Dimension d = front_end::mesh_file_dimension(mesh(name));
            EXPECT_EQ(d.dim, dim) << name << ": " << d.why;
        }
        if (have_cgns()) {
            EXPECT_EQ(front_end::mesh_file_dimension(mesh(cyl + ".cgns")).dim, dim);
        }
    }
    EXPECT_EQ(front_end::mesh_file_dimension(mesh("cyl3d_tet.msh")).dim, 3);
    if (have_hdf5()) {
        EXPECT_EQ(front_end::mesh_file_dimension(source_dir() + "/test/data/formats/v0.8.0/mesh.h5").dim, 2);
    }
}

TEST(FrontEndTest, Gmsh22DimensionIsThatOfItsElements) {
    EXPECT_EQ(front_end::mesh_file_dimension(source_dir() + "/test/data/formats/annulus.msh").dim, 2);
    std::ofstream("tet.msh") << "$MeshFormat\n2.2 0 8\n$EndMeshFormat\n$Nodes\n4\n1 0 0 0\n2 1 0 0\n3 0 1 0\n"
                                "4 0 0 1\n$EndNodes\n$Elements\n2\n1 2 2 1 1 1 2 3\n2 4 2 2 2 1 2 3 4\n$EndElements\n";
    EXPECT_EQ(front_end::mesh_file_dimension("tet.msh").dim, 3);
}

TEST(FrontEndTest, GeneratedMeshIs3DWithAThirdDirection) {
    EXPECT_EQ(input_dim("type = \"cartesian\"\nNx = 4\nNy = 4\n"), 2);
    EXPECT_EQ(input_dim("type = \"cartesian\"\nNz = 4\n"), 3);
    EXPECT_EQ(input_dim("type = \"cartesian\"\nLz = 2.0\n"), 3);
    EXPECT_EQ(input_dim("type = \"cartesian\"\nperiodic = [\"x\", \"z\"]\n"), 3);
    EXPECT_EQ(input_dim("type = \"annulus\"\nNz = 4\n"), 3);
    EXPECT_EQ(input_dim("type = \"cartesian_tet\"\n"), 3);
    EXPECT_EQ(input_dim("type = \"wedge\"\n"), 2);
}

TEST(FrontEndTest, MeshFileGivesTheInputItsDimension) {
    EXPECT_EQ(input_dim("type = \"file\"\nfilename = \"" + mesh("cyl3d.msh") + "\"\n"), 3);
    EXPECT_EQ(input_dim("filename = \"" + mesh("cyl2d_foam") + "\"\n"), 2);
    std::ofstream("input.toml") << "[mesh]\nfilename = \"missing.msh\"\n";
    const front_end::Dimension d = front_end::input_dimension("input.toml");
    EXPECT_EQ(d.dim, 0);
    EXPECT_NE(d.why.find("missing.msh"), std::string::npos) << d.why;
    EXPECT_TRUE(d.unreadable);
}

TEST(FrontEndTest, InvalidInputIsLeftToTheSolverToReport) {
    std::ofstream("input.toml") << "[mesh]\ntype = cartesian\n";
    EXPECT_TRUE(front_end::input_dimension("input.toml").unreadable);
    EXPECT_TRUE(front_end::input_dimension("no_such_input.toml").unreadable);
}

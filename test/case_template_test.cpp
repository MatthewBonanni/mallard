/**
 * @file case_template_test.cpp
 * @brief Tests for the input edits of `Mallard new --quick` and the path rewriting of templates.
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 */

#include <stdexcept>
#include <string>

#include <gtest/gtest.h>

#include "case_template.h"

using case_template::relocate;
using case_template::set_key;

TEST(CaseTemplateTest, SetKeyReplacesOnlyInItsTable) {
    const std::string in = "[run]\nt_stop = 0.2  # end\ncfl = 1.0\n\n[mesh]\nNx = 200\n";
    EXPECT_EQ(set_key(in, "run.t_stop", "0.05"), "[run]\nt_stop = 0.05\ncfl = 1.0\n\n[mesh]\nNx = 200\n");
    EXPECT_EQ(set_key(in, "mesh.Nx", "50"), "[run]\nt_stop = 0.2  # end\ncfl = 1.0\n\n[mesh]\nNx = 50\n");
}

TEST(CaseTemplateTest, SetKeyReplacesMultiLineValues) {
    const std::string in = "[mesh]\nperiodic = [\n  \"x\",\n  \"y\",\n]\nNx = 4\n[a]\nk = \"\"\"\nx\n\"\"\"\nz = 1\n";
    EXPECT_EQ(set_key(in, "mesh.periodic", "[\"x\"]"),
              "[mesh]\nperiodic = [\"x\"]\nNx = 4\n[a]\nk = \"\"\"\nx\n\"\"\"\nz = 1\n");
    EXPECT_EQ(set_key(in, "a.k", "\"y\""), "[mesh]\nperiodic = [\n  \"x\",\n  \"y\",\n]\nNx = 4\n[a]\nk = \"y\"\nz = 1\n");
}

TEST(CaseTemplateTest, SetKeyInArraysOfTables) {
    // Only the entries that define the key change (a restart writer keeps its step interval)
    const std::string in = "[[write_data]]\ntime_interval = 1.0\n[[write_data]]\ninterval = 100\n";
    EXPECT_EQ(set_key(in, "write_data.time_interval", "0.1"),
              "[[write_data]]\ntime_interval = 0.1\n[[write_data]]\ninterval = 100\n");
    // A key no entry defines is added to each
    EXPECT_EQ(set_key("[[b]]\nname = \"l\"\n[[b]]\nname = \"r\"\n", "b.p", "1"),
              "[[b]]\np = 1\nname = \"l\"\n[[b]]\np = 1\nname = \"r\"\n");
}

TEST(CaseTemplateTest, SetKeyNeedsTheTable) {
    EXPECT_THROW(set_key("[run]\ncfl = 1\n", "mesh.Nx", "4"), std::runtime_error);
    EXPECT_EQ(set_key("[numerics.face_reconstruction]\norder = 5\n", "numerics.face_reconstruction.order", "3"),
              "[numerics.face_reconstruction]\norder = 3\n");
}

TEST(CaseTemplateTest, RelocateRepositoryPaths) {
    EXPECT_EQ(relocate("mechanism = \"../../mechanisms/h2o2.yaml\"\n"
                       "#   python ../../tools/make_cylinder_mesh.py c.msh\n"
                       "#   mpirun -n 4 ../../build3d/src/Mallard -i input.toml\n"),
              "mechanism = \"mechanisms/h2o2.yaml\"\n"
              "#   python tools/make_cylinder_mesh.py c.msh\n"
              "#   mpirun -n 4 Mallard -i input.toml\n");
}

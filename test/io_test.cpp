/**
 * @file io_test.cpp
 * @author Matthew Bonanni (mbonanni001@gmail.com)
 * @brief Tests for output files.
 * @version 0.1
 * @date 2026-10-02
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 *
 */

#include <gtest/gtest.h>
#include <Kokkos_Core.hpp>

#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "test_fixtures.h"
#include "solver.h"

TEST(IOTest, BoundaryZoneOutputCarriesAdjacentCellValues) {
    const std::string dir = (std::filesystem::temp_directory_path() / "mallard_surface_test").string();
    std::filesystem::remove_all(dir);
    std::ostringstream s;
    s << "[run]\nn_steps = 4\ncfl = 0.5\n"
      << "[mesh]\ntype = \"wedge\"\nNx = 12\nNy = 6\nLx = 2.0\nLy = 1.5\n"
      << "[initialize]\ntype = \"analytical\"\nrho = \"1.0 + 0.1 * x\"\nu = [\"1.5\", \"0.0\"]\np = \"1.0\"\n"
      << "[[boundaries]]\nname = \"left\"\ntype = \"extrapolation\"\n"
      << "[[boundaries]]\nname = \"right\"\ntype = \"extrapolation\"\n"
      << "[[boundaries]]\nname = \"top\"\ntype = \"symmetry\"\n"
      << "[[boundaries]]\nname = \"bottom\"\ntype = \"symmetry\"\n"
      << "[numerics]\nriemann_solver = \"HLLC\"\n"
      << "[numerics.face_reconstruction]\ntype = \"FO\"\n"
      << "[physics]\ntype = \"euler\"\ngamma = 1.4\np_ref = 1.0\nT_ref = 1.0\nrho_ref = 1.0\n"
      << "[output]\ncheck_interval = 1000000\n"
      << "[[write_data]]\nprefix = \"" << dir << "/wall\"\nformat = \"vtu\"\ngeometry = \"bottom\"\n"
      << "interval = 4\nvariables = [\"P\"]\n";
    Solver solver;
    solver.init(parse_toml(s.str()));
    solver.run();
    solver.update_primitives();
    solver.copy_device_to_host();

    std::ifstream in(dir + "/wall_000004.vtu");
    ASSERT_TRUE(in.good());
    std::stringstream buffer;
    buffer << in.rdbuf();
    const std::string text = buffer.str();
    EXPECT_NE(text.find("NumberOfCells=\"12\""), std::string::npos);
    EXPECT_NE(text.find("NumberOfPoints=\"13\""), std::string::npos);

    // P values in the file, in zone order, equal the adjacent cells' pressure
    const size_t start = text.find(">", text.find("Name=\"P\"")) + 1;
    std::istringstream values(text.substr(start, text.find("</DataArray>", start) - start));
    FaceZone * zone = solver.get_mesh()->get_face_zone("bottom");
    for (uint32_t i = 0; i < zone->n_faces(); i++) {
        double p;
        values >> p;
        const int32_t c = solver.get_mesh()->h_cells_of_face(zone->h_faces(i), 0);
        EXPECT_DOUBLE_EQ(p, solver.h_primitives(c, 2));
    }
    std::filesystem::remove_all(dir);
}

TEST(IOTest, IntegerValuedRealInputsAreAccepted) {
    // TOML integers for real parameters must not be silently replaced by defaults
    const std::string input =
        "[run]\nn_steps = 1\ncfl = 1\n"
        "[mesh]\ntype = \"cartesian\"\nNx = 4\nNy = 2\nLx = 2\nLy = 1\n"
        "[initialize]\ntype = \"constant\"\nu = [1, 0]\np = 1\nT = 1\n"
        "[[boundaries]]\nname = \"left\"\ntype = \"extrapolation\"\n"
        "[[boundaries]]\nname = \"right\"\ntype = \"extrapolation\"\n"
        "[[boundaries]]\nname = \"top\"\ntype = \"symmetry\"\n"
        "[[boundaries]]\nname = \"bottom\"\ntype = \"symmetry\"\n"
        "[numerics.face_reconstruction]\ntype = \"FO\"\n"
        "[physics]\ntype = \"euler\"\ngamma = 1.4\np_ref = 1\nT_ref = 1\nrho_ref = 1\n";
    Solver solver;
    solver.init(parse_toml(input));
    rtype x_max = 0.0;
    for (uint32_t i = 0; i < solver.get_mesh()->n_nodes; i++) x_max = std::max(x_max, solver.get_mesh()->h_node_coords(i, 0));
    EXPECT_DOUBLE_EQ(x_max, 2.0);
    solver.copy_device_to_host();
    EXPECT_DOUBLE_EQ(solver.h_conservatives(0, 1), 1.0);
    EXPECT_THROW(Solver().init(parse_toml(input + "[source]\ngravity = [\"down\", 0]\n")), std::runtime_error);
}

TEST(IOTest, FixedTimeStepIsOnlyShortenedToLandOnOutputs) {
    // dt = 0.01 with outputs every 0.015: every other step is clipped to land
    // on an output time, the others take the full dt
    const std::string dir = (std::filesystem::temp_directory_path() / "mallard_fixed_dt").string();
    const std::string input =
        "[run]\nn_steps = 10\ndt = 0.01\n"
        "[mesh]\ntype = \"cartesian\"\nNx = 4\nNy = 2\nLx = 1.0\nLy = 1.0\n"
        "[initialize]\ntype = \"constant\"\nu = [0.1, 0.0]\np = 1.0\nT = 1.0\n"
        "[[boundaries]]\nname = \"left\"\ntype = \"extrapolation\"\n"
        "[[boundaries]]\nname = \"right\"\ntype = \"extrapolation\"\n"
        "[[boundaries]]\nname = \"top\"\ntype = \"symmetry\"\n"
        "[[boundaries]]\nname = \"bottom\"\ntype = \"symmetry\"\n"
        "[numerics.face_reconstruction]\ntype = \"FO\"\n"
        "[physics]\ntype = \"euler\"\ngamma = 1.4\np_ref = 1.0\nT_ref = 1.0\nrho_ref = 1.0\n"
        "[output]\ncheck_interval = 1000000\n"
        "[[write_data]]\nprefix = \"" + dir + "/f\"\nformat = \"vtu\"\ntime_interval = 0.015\nvariables = [\"RHO\"]\n";
    Solver solver;
    solver.init(parse_toml(input));
    solver.run();
    EXPECT_NEAR(solver.get_time(), 0.075, 1e-12);
    std::filesystem::remove_all(dir);
}

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

#include "data_writer.h"
#include "mesh_block.h"
#include "test_fixtures.h"
#include "solver.h"

TEST(IOTest, BoundaryZoneOutputCarriesAdjacentCellValues) {
    const std::string dir = (std::filesystem::temp_directory_path() / "mallard_surface_test").string();
    std::filesystem::remove_all(dir);
    std::ostringstream s;
    s << "[run]\nn_steps = 4\ncfl = 0.25\n"
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
      << "interval = 4\nvariables = [\"P\", \"U\"]\n";
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
        EXPECT_RTYPE_EQ(p, solver.h_primitives(c, 2));
    }

    // U is a 3-component vector, zero-padded in 2D
    const size_t u_tag = text.find("Name=\"U\"");
    const size_t u_start = text.find(">", u_tag) + 1;
    EXPECT_NE(text.substr(u_tag, u_start - u_tag).find("NumberOfComponents=\"3\""), std::string::npos);
    std::istringstream u_values(text.substr(u_start, text.find("</DataArray>", u_start) - u_start));
    for (uint32_t i = 0; i < zone->n_faces(); i++) {
        double u[3];
        u_values >> u[0] >> u[1] >> u[2];
        const int32_t c = solver.get_mesh()->h_cells_of_face(zone->h_faces(i), 0);
        EXPECT_RTYPE_EQ(u[0], solver.h_primitives(c, 0));
        EXPECT_RTYPE_EQ(u[1], solver.h_primitives(c, 1));
        EXPECT_EQ(u[2], 0.0);
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
    EXPECT_RTYPE_EQ(x_max, 2.0);
    solver.copy_device_to_host();
    EXPECT_RTYPE_EQ(solver.h_conservatives(0, 1), 1.0);
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
    EXPECT_NEAR(solver.get_time(), 0.075, roundoff(1e-12));
    std::filesystem::remove_all(dir);
}

TEST(IOTest, LocalCFLIsWrittenWithFixedTimeStepAndAtStepZero) {
    // Uniform flow u = 0.1, a = sqrt(1.4) on 0.25 x 0.5 cells: the CFL = 1 time
    // step is 2 V / sum_f (|u_n| + a) A_f = 0.25 / (0.1 + 1.5 a)
    const std::string dir = (std::filesystem::temp_directory_path() / "mallard_cfl_output").string();
    std::filesystem::remove_all(dir);
    const std::string input =
        "[run]\nn_steps = 1\ndt = 0.01\n"
        "[mesh]\ntype = \"cartesian\"\nNx = 4\nNy = 2\nLx = 1.0\nLy = 1.0\n"
        "[initialize]\ntype = \"constant\"\nu = [0.1, 0.0]\np = 1.0\nT = 1.0\n"
        "[[boundaries]]\nname = \"left\"\ntype = \"extrapolation\"\n"
        "[[boundaries]]\nname = \"right\"\ntype = \"extrapolation\"\n"
        "[[boundaries]]\nname = \"top\"\ntype = \"symmetry\"\n"
        "[[boundaries]]\nname = \"bottom\"\ntype = \"symmetry\"\n"
        "[physics]\ntype = \"euler\"\ngamma = 1.4\np_ref = 1.0\nT_ref = 1.0\nrho_ref = 1.0\n"
        "[output]\ncheck_interval = 1000000\n"
        "[[write_data]]\nprefix = \"" + dir + "/wall\"\nformat = \"vtu\"\ngeometry = \"bottom\"\n"
        "interval = 1\nvariables = [\"CFL\"]\n";
    Solver solver;
    solver.init(parse_toml(input));
    solver.run();
    const double expected = 0.01 * (0.1 + 1.5 * std::sqrt(1.4)) / 0.25;
    for (const char * file : {"/wall_000000.vtu", "/wall_000001.vtu"}) {
        std::ifstream in(dir + file);
        ASSERT_TRUE(in.good()) << file;
        std::stringstream buffer;
        buffer << in.rdbuf();
        const std::string text = buffer.str();
        const size_t start = text.find(">", text.find("Name=\"CFL\"")) + 1;
        std::istringstream values(text.substr(start, text.find("</DataArray>", start) - start));
        double cfl;
        int n = 0;
        while (values >> cfl) {
            EXPECT_NEAR(cfl, expected, 1e-6) << file;
            n++;
        }
        EXPECT_EQ(n, 4) << file;
    }
    std::filesystem::remove_all(dir);
}

TEST(IOTest, ZeroCheckIntervalIsRejected) {
    const std::string input =
        "[run]\nn_steps = 1\ncfl = 0.25\n"
        "[mesh]\ntype = \"cartesian\"\nNx = 4\nNy = 2\nLx = 1.0\nLy = 1.0\n"
        "[initialize]\ntype = \"constant\"\nu = [0.1, 0.0]\np = 1.0\nT = 1.0\n"
        "[[boundaries]]\nname = \"left\"\ntype = \"extrapolation\"\n"
        "[[boundaries]]\nname = \"right\"\ntype = \"extrapolation\"\n"
        "[[boundaries]]\nname = \"top\"\ntype = \"symmetry\"\n"
        "[[boundaries]]\nname = \"bottom\"\ntype = \"symmetry\"\n"
        "[physics]\ntype = \"euler\"\ngamma = 1.4\np_ref = 1.0\nT_ref = 1.0\nrho_ref = 1.0\n";
    EXPECT_NO_THROW(Solver().init(parse_toml(input)));
    EXPECT_THROW(Solver().init(parse_toml(input + "[output]\ncheck_interval = 0\n")), std::runtime_error);
}

TEST(IOTest, ResumedHDF5SeriesKeepsTheSnapshotsUpToTheRestartTime) {
    // A run restarted at t = 0.1 rewrites the later snapshots: the XDMF time
    // series lists each time once, and numbering continues from the restart
    if (!have_hdf5()) GTEST_SKIP() << "built without HDF5";
    const std::string dir = (std::filesystem::temp_directory_path() / "mallard_h5_resume").string();
    std::filesystem::remove_all(dir);
    auto mesh = make_mesh("cartesian_tri", 3, 2);
    Kokkos::View<rtype **>::host_mirror_type values("values", mesh->n_cells, 1);
    std::vector<Data> data = {Data("RHO", Kokkos::subview(values, Kokkos::ALL(), 0))};
    const std::string input = "prefix = \"" + dir + "/f\"\nformat = \"hdf5\"\ntime_interval = 0.1\n"
                              "variables = [\"RHO\"]\n";
    {
        DataWriter first;
        first.init(parse_toml(input), data, mesh);
        for (uint64_t step = 0; step < 4; step++) first.write(10 * step, static_cast<rtype>(0.1 * static_cast<double>(step)));
    }
    DataWriter second;
    second.init(parse_toml(input), data, mesh);
    second.resume(10, 0.1_r);
    second.write(25, 0.2_r);

    std::ifstream in(dir + "/f.xmf");
    std::stringstream buffer;
    buffer << in.rdbuf();
    const std::string text = buffer.str();
    std::vector<std::string> grids;
    for (size_t at = 0; (at = text.find("<Grid Name=\"f_", at)) != std::string::npos; at++) {
        grids.push_back(text.substr(at + 12, 8));
    }
    EXPECT_EQ(grids, (std::vector<std::string>{"f_000000", "f_000001", "f_000002"}));
    std::vector<double> times;
    for (size_t at = 0; (at = text.find("<Time Value=\"", at)) != std::string::npos; at++) {
        times.push_back(std::stod(text.substr(at + 13)));
    }
    EXPECT_EQ(times, (std::vector<double>{0.0, double(rtype(0.1)), double(rtype(0.2))}));
    EXPECT_TRUE(std::filesystem::exists(dir + "/f_mesh.h5"));
    std::filesystem::remove_all(dir);
}

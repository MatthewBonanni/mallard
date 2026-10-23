/**
 * @file mpi3d_test.cpp
 * @author Matthew Bonanni (mbonanni001@gmail.com)
 * @brief Distributed 3D runs reproduce the serial solution, on any number of ranks.
 * @version 0.4
 * @date 2026-10-02
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 *
 */

#include <gtest/gtest.h>

#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <string>

#include "data_paths.h"
#include "gmsh_fixtures.h"
#include "hdf5_output.h"
#include "mesh_block.h"
#include "mpi_compare.h"
#include "partition.h"
#include "renumbered_mesh.h"
#include "scratch.h"

namespace {

const char * ZONES[6] = {"left", "right", "bottom", "top", "back", "front"};

std::string blast_input(const std::string & mesh, const std::string & recon) {
    const std::string blob = "exp(-20 * ((x - 0.4)^2 + (y - 0.55)^2 + (z - 0.45)^2))";
    std::ostringstream s;
    s << "[run]\nn_steps = 3\ncfl = 0.25\n"
      << "[mesh]\ntype = \"" << mesh << "\"\nNx = 12\nNy = 12\nNz = 12\nLx = 1.0\nLy = 1.0\nLz = 1.0\n"
      << "[initialize]\ntype = \"analytical\"\nrho = \"1.0 + 0.5 * " << blob << "\"\n"
      << "u = [\"0.3\", \"-0.1\", \"0.2\"]\np = \"1.0 + 0.8 * " << blob << "\"\n";
    for (int k = 0; k < 6; k++) {
        s << "[[boundaries]]\nname = \"" << ZONES[k] << "\"\ntype = \""
          << (k % 2 ? "extrapolation" : "symmetry") << "\"\n";
    }
    s << "[numerics]\nriemann_solver = \"HLLC\"\ntime_integrator = \"SSPRK3\"\n"
      << "[numerics.face_reconstruction]\n" << recon
      << "[physics]\ntype = \"euler\"\ngamma = 1.4\np_ref = 1.0\nT_ref = 1.0\nrho_ref = 1.0\n"
      << "[output]\ncheck_interval = 1000000\n";
    return s.str();
}

std::string box_input(const std::string & mesh, const std::string & recon, const std::string & physics) {
    const char * types[6] = {"type = \"extrapolation\"\n", "type = \"symmetry\"\n", "type = \"wall_adiabatic\"\n",
                             "type = \"extrapolation\"\n", "type = \"symmetry\"\n", "type = \"extrapolation\"\n"};
    std::ostringstream s;
    s << "[run]\nn_steps = 8\ncfl = 0.25\n"
      << "[mesh]\ntype = \"" << mesh << "\"\nNx = 6\nNy = 4\nNz = 3\nLx = 1.0\nLy = 0.8\nLz = 0.6\n"
      << "[initialize]\ntype = \"analytical\"\n"
      << "rho = \"1.0 + 0.8 * exp(-20 * ((x - 0.35)^2 + (y - 0.45)^2 + (z - 0.3)^2))\"\n"
      << "u = [\"0.3\", \"-0.1\", \"0.2\"]\n"
      << "p = \"1.0 + 2.0 * exp(-20 * ((x - 0.35)^2 + (y - 0.45)^2 + (z - 0.3)^2))\"\n";
    for (int k = 0; k < 6; k++) s << "[[boundaries]]\nname = \"" << ZONES[k] << "\"\n" << types[k];
    s << "[numerics]\nriemann_solver = \"HLLC\"\ntime_integrator = \"SSPRK3\"\n"
      << "[numerics.face_reconstruction]\n" << recon
      << "[physics]\n" << physics << "gamma = 1.4\np_ref = 1.0\nT_ref = 1.0\nrho_ref = 1.0\n"
      << "[output]\ncheck_interval = 1000000\n";
    return s.str();
}

} // namespace

TEST(MPI3DTest, TENOOnHexahedraMatchesSerial) {
    // Order 5 needs five distinct cell planes across a planar partition
    // interface; a halo that cuts the stencil search short must be deepened
    // rather than reported as a rank-deficient stencil
    expect_matches_serial(blast_input("cartesian", "type = \"TENO\"\norder = 5\n"));
}

TEST(MPI3DTest, MUSCLOnMixedCellsMatchesSerial) {
    expect_matches_serial(box_input("cartesian_mixed", "type = \"MUSCL\"\n", "type = \"euler\"\n"));
}

TEST(MPI3DTest, NavierStokesOnTetrahedraMatchesSerial) {
    expect_matches_serial(box_input("cartesian_tet", "type = \"MUSCL\"\n",
                                    "type = \"navier_stokes\"\nmu = 0.01\nPr = 0.72\n"));
}

TEST(MPI3DTest, RenumberedMeshesMatchSerialWithEveryPartitioner) {
    // Global ids in random order carry no locality: partitions, halos and the
    // global orders of faces and stencils must not lean on them
    const RenumberedMesh mesh = renumbered_gmsh(jittered_box(MeshType::CARTESIAN_MIXED, {6, 4, 3}, {1.0, 0.8, 0.6}), 2026);
    const std::string file = write_temp_shared("mallard_mpi3d_renumbered.msh", mesh.gmsh);
    std::string input = box_input("cartesian_mixed", "type = \"TENO\"\norder = 3\n",
                                  "type = \"navier_stokes\"\nmu = 0.01\nPr = 0.72\n");
    const std::string generated = "type = \"cartesian_mixed\"\nNx = 6\nNy = 4\nNz = 3\nLx = 1.0\nLy = 0.8\nLz = 0.6\n";
    input.replace(input.find(generated), generated.size(), "type = \"file\"\nfilename = \"" + file + "\"\n");
    for (const std::string partitioner : {"hilbert", "multijagged", "graph"}) {
        if (partitioner == "graph" && !have_graph_partitioner()) continue;
        expect_matches_serial(input + "[parallel]\npartitioner = \"" + partitioner + "\"\n");
    }
    comm::barrier();
}

TEST(MPI3DTest, LargeEddySimulationOnTetrahedraMatchesSerial) {
    std::string input =
        box_input("cartesian_tet", "type = \"MUSCL\"\n", "type = \"navier_stokes\"\nmu = 0.01\nPr = 0.72\n");
    const std::string u = "u = [\"0.3\", \"-0.1\", \"0.2\"]\n";
    input.replace(input.find(u), u.size(),
                  "u = [\"0.3 + 0.4 * sin(9 * y)\", \"-0.1 + 0.3 * sin(7 * z)\", \"0.2 + 0.3 * sin(8 * x + 3 * y)\"]\n");
    expect_matches_serial(input + "[les]\nmodel = \"sigma\"\nC = 3.0\n");
    // The dynamic constant is a domain sum: exact in fixed point, so the same on any rank count
    expect_matches_serial(input + "[les]\nmodel = \"sigma\"\ndynamic = true\n");
}

TEST(MPI3DTest, WallModelsMatchSerial) {
    // Exchange and edge points several cells off the walls, in cells that a
    // partition may leave beyond the default halo, which must then deepen.
    // Without LES or the hybrid flux, only the wall model reads the gradients
    // of the deep halo cells holding its points. The mass flow forcing's
    // domain sums are exact, so the same on any rank count
    for (const char * partitioner : {"multijagged", "hilbert"}) for (const bool les : {true, false}) {
        std::ostringstream s;
        s << "[run]\nn_steps = 6\ncfl = 0.4\n"
          << "[mesh]\ntype = \"cartesian_tet\"\nNx = 6\nNy = 8\nNz = 4\nLx = 1.0\nLy = 1.0\nLz = 0.6\n"
          << "periodic = [\"x\", \"z\"]\n"
          << "[parallel]\npartitioner = \"" << partitioner << "\"\n"
          << "[initialize]\ntype = \"analytical\"\nrho = \"1.0 + 0.1 * sin(6.283 * x)\"\n"
          << "u = [\"0.5 + 0.1 * sin(6.283 * z)\", \"0.05 * sin(6.283 * x)\", \"0.02\"]\np = \"1.0\"\n"
          << "[[boundaries]]\nname = \"bottom\"\ntype = \"wall_isothermal\"\nT = 1.0\n"
          << "wall_model = { model = \"gfm\", exchange_distance = 0.45, edge_distance = 0.9 }\n"
          << "[[boundaries]]\nname = \"top\"\ntype = \"wall_adiabatic\"\n"
          << "wall_model = { model = \"equilibrium\", exchange_cells = 3.0 }\n"
          << "[numerics]\nriemann_solver = \"HLLC\"\ntime_integrator = \"SSPRK3\"\n"
          << (les ? "convective_flux = \"hybrid\"\n" : "")
          << "[numerics.face_reconstruction]\ntype = \"MUSCL\"\nlimiter = \"none\"\n"
          << "[physics]\ntype = \"navier_stokes\"\ngamma = 1.4\np_ref = 1.0\nT_ref = 1.0\nrho_ref = 1.0\n"
          << "mu = 1e-3\nPr = 0.72\n"
          << (les ? "[les]\nmodel = \"sigma\"\n" : "")
          << "[output]\ncheck_interval = 1000000\n"
          << "[source]\nmass_flow = [0.5, 0.0, 0.0]\n";
        expect_matches_serial(s.str());
    }
}

TEST(MPI3DTest, DynamicFlameEfficiencyOnTetrahedraMatchesSerial) {
    // The test filter passes exchange the filtered progress variable; beta is an exact domain sum
    std::ostringstream s;
    s << "[run]\nn_steps = 3\ncfl = 0.25\n"
      << "[mesh]\ntype = \"cartesian_tet\"\nNx = 8\nNy = 5\nNz = 4\nLx = 0.008\nLy = 0.005\nLz = 0.004\n"
      << "[initialize]\ntype = \"analytical\"\np = \"101325.0\"\n"
      << "T = \"1300 + 1000 * tanh((x - 0.004 - 0.001 * sin(1200 * y) * cos(900 * z)) / 0.0008)\"\n"
      << "u = [\"3 * sin(1500 * y)\", \"2 * sin(1300 * z)\", \"0.0\"]\nX = { H2 = 2.0, O2 = 1.0, N2 = 3.76 }\n";
    for (int k = 0; k < 6; k++) {
        s << "[[boundaries]]\nname = \"" << ZONES[k] << "\"\ntype = \"" << (k == 0 ? "p_out\"\np = 101325.0" : "symmetry\"")
          << "\n";
    }
    s << "[numerics]\nriemann_solver = \"HLLC\"\n[numerics.face_reconstruction]\ntype = \"MUSCL\"\n"
      << "[physics]\ntype = \"navier_stokes\"\ngas = \"mixture\"\nmechanism = \"" + source_dir() +
"/mechanisms/h2o2.yaml\"\n"
      << "[chemistry]\n[output]\ncheck_interval = 1000000\n[les]\nmodel = \"sigma\"\n"
      << "[les.combustion]\nmodel = \"tfles\"\ndelta_L = 3e-4\ns_L = 2.3\nT_unburnt = 300.0\nT_burnt = 2300.0\n"
      << "efficiency = \"dynamic\"\n";
    expect_matches_serial(s.str(), [](const Solver & solver) { EXPECT_GT(solver.tfles_beta(), 0.0); });
}

TEST(MPI3DTest, CharacteristicBoundariesMatchSerial) {
    std::string input = box_input("cartesian_tet", "type = \"MUSCL\"\n", "type = \"euler\"\n");
    const std::string from = "name = \"right\"\ntype = \"symmetry\"\n";
    input.replace(input.find(from), from.size(), "name = \"right\"\ntype = \"nscbc_outlet\"\np = 1.0\nL = 1.0\n");
    const std::string left = "name = \"left\"\ntype = \"extrapolation\"\n";
    input.replace(input.find(left), left.size(),
                  "name = \"left\"\ntype = \"nscbc_inlet\"\nu = [0.3, 0.0, 0.0]\np = 1.0\nT = 1.0\nL = 1.0\n");
    expect_matches_serial(input);
}

TEST(MPI3DTest, SyntheticTurbulenceInletMatchesSerial) {
    // The inflow field is a function of the time and the inlet's geometry alone,
    // and its zero-net-flux correction sums over the whole inlet in a fixed order
    std::string input = box_input("cartesian_tet", "type = \"MUSCL\"\n", "type = \"euler\"\n");
    const std::string from = "name = \"right\"\ntype = \"symmetry\"\n";
    input.replace(input.find(from), from.size(), "name = \"right\"\ntype = \"nscbc_outlet\"\np = 1.0\nL = 1.0\n");
    const std::string left = "name = \"left\"\ntype = \"extrapolation\"\n";
    input.replace(input.find(left), left.size(),
                  "name = \"left\"\ntype = \"nscbc_inlet\"\nu = [\"0.3 + 0.1 * y\", 0.0, 0.0]\np = 1.0\nT = 1.0\nL = 1.0\n"
                  "[boundaries.turbulence]\nreynolds_stress = [0.004, 0.002, 0.002, -0.001, 0.0, 0.0]\n"
                  "length_scale = [0.3, 0.15, 0.1]\n");
    expect_matches_serial(input);
}

TEST(MPI3DTest, PeriodicZonePairsOfAGmshMeshMatchSerial) {
    // Fully periodic: every node class of the box corners spans eight nodes
    const std::string file = write_temp_shared("mallard_mpi3d_periodic.msh", jittered_periodic_mesh_3d(6));
    std::string input = blast_input("cartesian", "type = \"TENO\"\norder = 3\n");
    input = input.substr(0, input.find("[[boundaries]]")) + input.substr(input.find("[numerics]"));
    const std::string generated = "type = \"cartesian\"\nNx = 12\nNy = 12\nNz = 12\nLx = 1.0\nLy = 1.0\nLz = 1.0\n";
    input.replace(input.find(generated), generated.size(), "type = \"file\"\nfilename = \"" + file + "\"\n");
    for (int d = 0; d < 3; d++) {
        input += std::string("[[periodic]]\nzones = [\"") + ZONES[2 * d] + "\", \"" + ZONES[2 * d + 1] +
                 "\"]\ntranslation = [" + (d == 0 ? "1.0" : "0.0") + ", " + (d == 1 ? "1.0" : "0.0") + ", " +
                 (d == 2 ? "1.0" : "0.0") + "]\n";
    }
    expect_matches_serial(input + "[parallel]\npartitioner = \"hilbert\"\n");
    comm::barrier();
}

TEST(MPI3DTest, SurfaceOutputOfAZoneMissingFromSomeRanks) {
    // Ranks whose part of the mesh does not touch the zone write empty pieces
    const std::string dir = (scratch_dir() / "mallard_mpi3d_faces").string();
    if (comm::is_root()) std::filesystem::remove_all(dir);
    comm::barrier();
    Solver solver;
    solver.init(parse_toml(blast_input("cartesian", "type = \"FO\"\n") +
                           "[[write_data]]\nprefix = \"" + dir + "/left\"\nformat = \"vtu\"\ngeometry = \"left\"\n"
                           "interval = 3\nvariables = [\"RHO\"]\n"));
    solver.run();
    comm::barrier();
    uint64_t total = 0;
    for (int r = 0; r < comm::size(); r++) {
        std::ostringstream piece;
        piece << dir << "/left_000003";
        if (comm::size() > 1) piece << "_p" << std::setw(4) << std::setfill('0') << r;
        std::ifstream in(piece.str() + ".vtu");
        std::stringstream ss;
        ss << in.rdbuf();
        const std::string text = ss.str();
        const size_t k = text.find("NumberOfCells=\"");
        ASSERT_NE(k, std::string::npos) << piece.str();
        total += std::stoull(text.substr(k + 15));
    }
    EXPECT_EQ(total, 12u * 12u);
}

TEST(MPI3DTest, HDF5OutputDoesNotDependOnTheRankCount) {
    // Tetrahedra, pyramids, prisms and hexahedra at their global ids
    if (!have_hdf5()) GTEST_SKIP() << "built without HDF5";
    if (comm::size() > 1 && !have_parallel_hdf5()) GTEST_SKIP() << "needs parallel HDF5";
#ifdef Mallard_HAS_HDF5
    const std::string dir = (scratch_dir() / "mallard_mpi3d_hdf5").string();
    expect_hdf5_output_matches_serial(box_input("cartesian_mixed", "type = \"MUSCL\"\n", "type = \"euler\"\n"),
                                      "interval = 4\nvariables = [\"RHO\", \"U\", \"P\"]\n", dir, {0, 4, 8});
#endif
}

TEST(MPI3DTest, CurvedWallsMatchSerial) {
    // A spherical shell of prisms: walls projected onto the spheres, or from
    // triangle6 faces of the file
    const std::string shapes = "[[mesh.curved]]\nzone = \"inner\"\nshape = \"sphere\"\ncenter = [0.0, 0.0, 0.0]\n"
                               "radius = 1.0\n[[mesh.curved]]\nzone = \"outer\"\nshape = \"sphere\"\n"
                               "center = [0.0, 0.0, 0.0]\nradius = 1.5\n";
    for (bool quadratic : {false, true}) {
        const std::string file = write_temp_shared(std::string("mallard_mpi_shell_") + (quadratic ? "p2" : "p1") + ".msh",
                                                   shell_gmsh(4, 3, true, quadratic));
        const std::string input =
            "[run]\nn_steps = 6\ncfl = 0.3\n[mesh]\ntype = \"file\"\nfilename = \"" + file + "\"\n" +
            (quadratic ? "" : shapes) +
            "[initialize]\ntype = \"analytical\"\nrho = \"1.0 + 0.2 * exp(-4 * ((x - 1.2)^2 + y^2 + z^2))\"\n"
            "u = [\"0.0\", \"0.3 * z\", \"-0.3 * y\"]\np = \"1.0\"\n"
            "[[boundaries]]\nname = \"inner\"\ntype = \"wall_adiabatic\"\n"
            "[[boundaries]]\nname = \"outer\"\ntype = \"symmetry\"\n"
            "[numerics]\nriemann_solver = \"HLLC\"\ntime_integrator = \"SSPRK3\"\n"
            "[numerics.face_reconstruction]\ntype = \"TENO\"\norder = 3\n"
            "[physics]\ntype = \"euler\"\ngamma = 1.4\np_ref = 1.0\nT_ref = 1.0\nrho_ref = 1.0\n"
            "[output]\ncheck_interval = 1000000\n";
        expect_matches_serial(input, [](const Solver & solver) { EXPECT_TRUE(solver.get_mesh()->curved_geometry); });
    }
    comm::barrier();
}

TEST(MPI3DTest, RemapDoesNotDependOnTheRankCount) {
    // A prism run's restart file remapped to tetrahedra of another size
    const std::string dir = (scratch_dir() / "mallard_mpi3d_remap").string();
    if (comm::is_root()) std::filesystem::remove_all(dir);
    comm::barrier();
    std::string coarse = box_input("cartesian_prism", "type = \"MUSCL\"\n", "type = \"euler\"\n");
    coarse.replace(coarse.find("n_steps = 8"), 11, "n_steps = 4");
    {
        Solver solver;
        solver.init(parse_toml(coarse + "[[write_data]]\nprefix = \"" + dir +
                               "/c\"\nformat = \"restart\"\ninterval = 4\n"));
        solver.run();
    }
    comm::barrier();
    std::string input = box_input("cartesian_tet", "type = \"MUSCL\"\n", "type = \"euler\"\n");
    input.replace(input.find("Nx = 6\nNy = 4\nNz = 3"), 20, "Nx = 5\nNy = 5\nNz = 4");
    input.replace(input.find("n_steps = 8"), 11, "n_steps = 6");
    const size_t at = input.find("[initialize]\n");
    const size_t end = input.find("[[boundaries]]", at);
    input.replace(at, end - at,
                  "[initialize]\ntype = \"remap\"\nfile = \"" + dir + "/c_000004.restart\"\ndegree = 3\n"
                  "[initialize.mesh]\ntype = \"cartesian_prism\"\nNx = 6\nNy = 4\nNz = 3\nLx = 1.0\nLy = 0.8\nLz = 0.6\n");
    expect_matches_serial(input);
    comm::barrier();
}

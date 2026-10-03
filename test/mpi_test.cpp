/**
 * @file mpi_test.cpp
 * @author Matthew Bonanni (mbonanni001@gmail.com)
 * @brief Distributed runs reproduce the serial solution, on any number of ranks.
 * @version 0.3
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
#include <iomanip>
#include <set>
#include <sstream>
#include <string>
#include <vector>

#include "comm.h"
#include "gmsh_fixtures.h"
#include "mesh_block.h"
#include "mpi_compare.h"
#include "partition.h"
#include "test_fixtures.h"
#include "solver.h"

namespace {

const char * BLAST =
    "type = \"analytical\"\n"
    "rho = \"1.0 + 0.8 * exp(-40 * ((x - 0.35)^2 + (y - 0.45)^2))\"\n"
    "u = [\"0.3\", \"-0.1\"]\n"
    "p = \"1.0 + 2.0 * exp(-40 * ((x - 0.35)^2 + (y - 0.45)^2))\"\n";

std::string box_input(const std::string & mesh, const std::string & recon, const std::string & physics,
                      const std::string & boundaries, uint32_t n_steps) {
    std::ostringstream s;
    s << "[run]\nn_steps = " << n_steps << "\ncfl = 0.5\n"
      << "[mesh]\ntype = \"" << mesh << "\"\nNx = 24\nNy = 18\nLx = 1.0\nLy = 0.8\n"
      << "[initialize]\n" << BLAST << boundaries
      << "[numerics]\nriemann_solver = \"HLLC\"\ntime_integrator = \"SSPRK3\"\n"
      << "[numerics.face_reconstruction]\n" << recon
      << "[physics]\n" << physics << "gamma = 1.4\np_ref = 1.0\nT_ref = 1.0\nrho_ref = 1.0\n"
      << "[output]\ncheck_interval = 1000000\n";
    return s.str();
}

std::string bcs(const char * left, const char * right, const char * top, const char * bottom) {
    std::ostringstream s;
    for (auto [name, type] : {std::pair{"left", left}, {"right", right}, {"top", top}, {"bottom", bottom}}) {
        s << "[[boundaries]]\nname = \"" << name << "\"\n" << type;
    }
    return s.str();
}

const std::string EULER = "type = \"euler\"\n";
const std::string NS = "type = \"navier_stokes\"\nmu = 0.01\nPr = 0.72\n";

} // namespace

TEST(MPITest, HaloExchangeFillsEveryHaloCellFromItsOwner) {
    Solver solver;
    solver.init(parse_toml(box_input("cartesian_tri", "type = \"MUSCL\"\n", EULER,
                                     bcs("type = \"extrapolation\"\n", "type = \"extrapolation\"\n",
                                         "type = \"symmetry\"\n", "type = \"symmetry\"\n"), 0)));
    if (!solver.is_distributed()) GTEST_SKIP() << "needs more than one rank";
    const auto & dist = solver.get_distribution();
    const uint32_t n_local = solver.get_mesh()->n_cells;
    HaloExchange halo(dist);
    // Flow block alone, then with species, through the same exchange object
    for (const uint32_t n_species : {0u, 3u}) {
        State U("U", n_local, n_species);
        auto h_flow = Kokkos::create_mirror_view(U.flow);
        auto h_species = Kokkos::create_mirror_view(U.species);
        for (uint32_t c = 0; c < n_local; c++) {
            const bool owned = c < dist.n_owned;
            FOR_I_CONSERVATIVE h_flow(c, i) = owned ? dist.global_cell[c] + 0.25 * i : -1.0;
            for (uint32_t k = 0; k < n_species; k++) h_species(c, k) = owned ? dist.global_cell[c] + 0.125 * k : -1.0;
        }
        Kokkos::deep_copy(U.flow, h_flow);
        Kokkos::deep_copy(U.species, h_species);
        halo.exchange(U);
        Kokkos::deep_copy(h_flow, U.flow);
        Kokkos::deep_copy(h_species, U.species);
        for (uint32_t c = 0; c < n_local; c++) {
            FOR_I_CONSERVATIVE EXPECT_EQ(h_flow(c, i), dist.global_cell[c] + 0.25 * i) << "local cell " << c;
            for (uint32_t k = 0; k < n_species; k++) {
                EXPECT_EQ(h_species(c, k), dist.global_cell[c] + 0.125 * k) << "local cell " << c;
            }
        }
    }
    std::set<uint64_t> ids(dist.global_cell.begin(), dist.global_cell.end());
    EXPECT_EQ(ids.size(), dist.global_cell.size());
}

TEST(MPITest, FirstOrderMatchesSerial) {
    expect_matches_serial(box_input("cartesian_tri", "type = \"FO\"\n", EULER,
                                    bcs("type = \"extrapolation\"\n", "type = \"symmetry\"\n",
                                        "type = \"wall_adiabatic\"\n", "type = \"extrapolation\"\n"), 30));
}

TEST(MPITest, MUSCLMatchesSerial) {
    expect_matches_serial(box_input("cartesian_tri", "type = \"MUSCL\"\n", EULER,
                                    bcs("type = \"extrapolation\"\n", "type = \"symmetry\"\n",
                                        "type = \"wall_adiabatic\"\n", "type = \"extrapolation\"\n"), 30));
}

TEST(MPITest, TENOOnQuadsMatchesSerial) {
    // Stencils reach several layers into neighboring ranks and mirror across
    // physical boundaries, never across partition faces
    expect_matches_serial(box_input("cartesian", "type = \"TENO\"\norder = 5\n", EULER,
                                    bcs("type = \"extrapolation\"\n", "type = \"symmetry\"\n",
                                        "type = \"symmetry\"\n", "type = \"extrapolation\"\n"), 20));
}

TEST(MPITest, TENOOnTrianglesMatchesSerial) {
    expect_matches_serial(box_input("cartesian_tri", "type = \"TENO\"\norder = 4\n", EULER,
                                    bcs("type = \"extrapolation\"\n", "type = \"symmetry\"\n",
                                        "type = \"wall_adiabatic\"\n", "type = \"extrapolation\"\n"), 15));
}

namespace {

/**
 * @brief Periodic faces whose cells are on different ranks, summed over the
 *        ranks: halo cells reached across a seam.
 */
uint64_t seam_faces_cut_by_partition(const std::string & input) {
    Solver solver;
    solver.init(parse_toml(input));
    const auto mesh = solver.get_mesh();
    uint64_t n = 0;
    for (uint32_t f = 0; f < mesh->n_faces; f++) {
        const int32_t c0 = mesh->h_cells_of_face(f, 0), c1 = mesh->h_cells_of_face(f, 1);
        if (c1 < 0 || mesh->h_face_shift(f) == 0) continue;
        n += (uint32_t(c0) < mesh->n_owned()) != (uint32_t(c1) < mesh->n_owned());
    }
    return comm::allreduce(n, comm::Op::SUM);
}

std::string periodic_box(const std::string & mesh, const std::string & recon, const std::string & physics,
                         const std::string & dirs, const std::string & boundaries, uint32_t n_steps) {
    std::string input = box_input(mesh, recon, physics, boundaries, n_steps);
    // A Hilbert partition always cuts the seams; a graph partition of a torus need not
    const std::string anchor = "Ly = 0.8\n";
    return input.replace(input.find(anchor), anchor.size(), anchor + "periodic = " + dirs + "\n") +
           "[parallel]\npartitioner = \"hilbert\"\n";
}

} // namespace

TEST(MPITest, PeriodicRunsMatchSerialAcrossSeamsCutByThePartition) {
    // Stencils and halos wrap across the seams, and the partition cuts them
    const std::string none = "";
    const std::string walls = "[[boundaries]]\nname = \"top\"\ntype = \"symmetry\"\n"
                              "[[boundaries]]\nname = \"bottom\"\ntype = \"wall_adiabatic\"\n";
    const std::string teno = periodic_box("cartesian", "type = \"TENO\"\norder = 5\n", EULER, "[\"x\", \"y\"]", none, 15);
    if (comm::size() > 1) {
        EXPECT_GT(seam_faces_cut_by_partition(teno), 0u);
    }
    expect_matches_serial(teno);
    expect_matches_serial(periodic_box("cartesian_tri", "type = \"TENO\"\norder = 4\n", EULER, "[\"x\"]", walls, 10));
    expect_matches_serial(periodic_box("cartesian_tri", "type = \"MUSCL\"\n", NS, "[\"x\", \"y\"]", none, 20));
}

TEST(MPITest, PeriodicZonePairsOfAGmshMeshMatchSerial) {
    const std::string file = write_temp_shared("mallard_mpi_periodic.msh", jittered_mixed_mesh(16));
    std::string input = box_input("cartesian", "type = \"TENO\"\norder = 4\n", EULER, "", 12);
    const std::string generated = "type = \"cartesian\"\n";
    input.replace(input.find(generated), generated.size(), "type = \"file\"\nfilename = \"" + file + "\"\n");
    input += "[[periodic]]\nzones = [\"left\", \"right\"]\ntranslation = [1.0, 0.0]\n"
             "[[periodic]]\nzones = [\"bottom\", \"top\"]\ntranslation = [0.0, 1.0]\n"
             "[parallel]\npartitioner = \"hilbert\"\n";
    if (comm::size() > 1) {
        EXPECT_GT(seam_faces_cut_by_partition(input), 0u);
    }
    expect_matches_serial(input);
    comm::barrier();
}

TEST(MPITest, NavierStokesWithBoundaryConditionsMatchesSerial) {
    expect_matches_serial(box_input(
        "cartesian", "type = \"MUSCL\"\n", NS,
        bcs("type = \"dirichlet\"\nrho = \"1.0\"\nu = [\"0.3\", \"0.0\"]\np = \"1.0 + 0.1 * sin(6 * y) * sin(20 * t)\"\n",
            "type = \"p_out_average\"\np = 1.0\n",
            "type = \"wall_isothermal\"\nT = 1.2\nu = [0.1, 0.0]\n",
            "type = \"wall_adiabatic\"\n"),
        25));
}

TEST(MPITest, BoundaryConditionsSurviveTheHaloRebuild) {
    // TENO stencils need a deeper halo than the first one, so the local mesh is
    // built twice. Dirichlet face lists of the first mesh used to survive, and
    // their faces, renumbered, could have no Dirichlet state in the second.
    expect_matches_serial(box_input(
        "cartesian", "type = \"TENO\"\norder = 3\n", EULER,
        bcs("type = \"dirichlet\"\nrho = \"1.0\"\nu = [\"0.3\", \"0.0\"]\np = \"1.0 + 0.1 * sin(6 * y) * sin(20 * t)\"\n",
            "type = \"p_out_average\"\np = 1.0\n",
            "type = \"dirichlet\"\nrho = \"1.0\"\nu = [\"0.0\", \"0.0\"]\np = \"1.0\"\n",
            "type = \"dirichlet\"\nrho = \"1.0\"\nu = [\"0.0\", \"0.0\"]\np = \"1.0\"\n"),
        10));
}

namespace {

std::string io_dir() {
    // Same path on every rank (they share a file system)
    return (std::filesystem::temp_directory_path() / "mallard_mpi_io").string();
}

std::string restart_case(uint32_t n_steps, const std::string & output) {
    return box_input("cartesian_tri", "type = \"MUSCL\"\n", EULER,
                     bcs("type = \"extrapolation\"\n", "type = \"symmetry\"\n", "type = \"wall_adiabatic\"\n",
                         "type = \"extrapolation\"\n"),
                     n_steps) +
           output;
}

std::vector<double> gather(Solver & solver) {
    solver.copy_device_to_host();
    const auto mesh = solver.get_mesh();
    const uint64_t n_global = mesh->n_global_cells ? mesh->n_global_cells : mesh->n_cells;
    std::vector<double> U(n_global * N_CONSERVATIVE, 0.0);
    for (uint32_t c = 0; c < mesh->n_owned(); c++) {
        const uint64_t g = mesh->n_global_cells ? mesh->h_global_cell_id[c] : c;
        FOR_I_CONSERVATIVE U[g * N_CONSERVATIVE + i] = double(solver.h_conservatives(c, i));
    }
    // A serial run holds every cell on every rank already
    if (mesh->n_global_cells > 0) comm::allreduce(std::span<double>(U), comm::Op::SUM);
    return U;
}

double max_rel_diff(const std::vector<double> & a, const std::vector<double> & b) {
    double m = 0.0;
    for (size_t k = 0; k < a.size(); k++) m = std::max(m, std::abs(a[k] - b[k]) / (std::abs(b[k]) + 1e-3));
    return m;
}

} // namespace

TEST(MPITest, RestartFilesDoNotDependOnTheRankCount) {
    const std::string dir = io_dir();
    if (comm::is_root()) std::filesystem::remove_all(dir);
    comm::barrier();
    const std::string writer = "[[write_data]]\nprefix = \"" + dir + "/r\"\nformat = \"restart\"\ninterval = 10\n";
    auto from = [&](const std::string & file) { return "type = \"restart\"\nfile = \"" + file + "\"\n"; };
    std::string init = BLAST;

    // Uninterrupted serial reference
    Solver reference;
    reference.set_distributed(false);
    reference.init(parse_toml(restart_case(20, "")));
    reference.run();
    const auto U_ref = gather(reference);

    // Written by all ranks at step 10, continued by all ranks
    {
        Solver first;
        first.init(parse_toml(restart_case(10, writer)));
        first.run();
    }
    comm::barrier();
    std::string input = restart_case(20, "");
    input.replace(input.find("[initialize]\n") + 13, init.size(), from(dir + "/r_000010.restart"));
    Solver second;
    second.init(parse_toml(input));
    EXPECT_EQ(second.get_step(), 10u);
    second.run();
    EXPECT_EQ(max_rel_diff(gather(second), U_ref), 0.0);

    // The same file read by a single rank
    Solver serial;
    serial.set_distributed(false);
    serial.init(parse_toml(input));
    serial.run();
    EXPECT_EQ(max_rel_diff(gather(serial), U_ref), 0.0);
    comm::barrier();
}

TEST(MPITest, EveryCellIsInExactlyOneOutputPiece) {
    const std::string dir = io_dir() + "_vtu";
    if (comm::is_root()) std::filesystem::remove_all(dir);
    comm::barrier();
    Solver solver;
    solver.init(parse_toml(restart_case(2, "[[write_data]]\nprefix = \"" + dir + "/f\"\nformat = \"vtu\"\n"
                                                  "interval = 2\nvariables = [\"RHO\"]\n")));
    solver.run();
    comm::barrier();
    const uint64_t n_global = solver.get_mesh()->n_global_cells ? solver.get_mesh()->n_global_cells
                                                                : solver.get_mesh()->n_cells;
    auto read = [](const std::string & path) {
        std::ifstream in(path);
        std::stringstream ss;
        ss << in.rdbuf();
        return ss.str();
    };
    if (comm::size() == 1) {
        EXPECT_TRUE(std::filesystem::exists(dir + "/f_000002.vtu"));
        return;
    }
    const std::string index = read(dir + "/f_000002.pvtu");
    uint64_t total = 0;
    for (int r = 0; r < comm::size(); r++) {
        std::ostringstream piece;
        piece << "f_000002_p" << std::setw(4) << std::setfill('0') << r << ".vtu";
        EXPECT_NE(index.find(piece.str()), std::string::npos) << piece.str();
        const std::string text = read(dir + "/" + piece.str());
        const size_t k = text.find("NumberOfCells=\"");
        ASSERT_NE(k, std::string::npos) << piece.str();
        total += std::stoull(text.substr(k + 15));
    }
    EXPECT_EQ(total, n_global);
    EXPECT_NE(read(dir + "/f.pvd").find("f_000002.pvtu"), std::string::npos);
}

TEST(MPITest, GraphPartitionIsBalancedAndMatchesSerial) {
    if (!have_graph_partitioner()) GTEST_SKIP() << "built without a graph partitioner";
    const std::string input =
        box_input("cartesian_tri", "type = \"TENO\"\norder = 4\n", EULER,
                  bcs("type = \"extrapolation\"\n", "type = \"symmetry\"\n", "type = \"wall_adiabatic\"\n",
                      "type = \"extrapolation\"\n"),
                  10) +
        "[parallel]\npartitioner = \"graph\"\n";
    expect_matches_serial(input);
    Solver solver;
    solver.init(parse_toml(input));
    if (!solver.is_distributed()) return;
    const uint64_t n_owned = solver.get_distribution().n_owned;
    const uint64_t n_global = solver.get_mesh()->n_global_cells;
    EXPECT_LE(comm::allreduce(n_owned, comm::Op::MAX), 1.03 * n_global / comm::size() + 1);
}

TEST(MPITest, RunFromHDF5MeshFileMatchesSerial) {
    if (!have_hdf5()) GTEST_SKIP() << "built without HDF5";
    if (comm::size() > 1 && !have_parallel_hdf5()) GTEST_SKIP() << "needs parallel HDF5 to write the mesh";
    const std::string file = (std::filesystem::temp_directory_path() / "mallard_mpi_mesh.h5").string();
    write_mesh_h5(file, read_mesh_block(parse_toml("[mesh]\ntype = \"cartesian_tri\"\nNx = 24\nNy = 18\n"
                                                   "Lx = 1.0\nLy = 0.8\n")));
    comm::barrier();
    std::string input = box_input("cartesian_tri", "type = \"MUSCL\"\n", EULER,
                                  bcs("type = \"extrapolation\"\n", "type = \"symmetry\"\n",
                                      "type = \"wall_adiabatic\"\n", "type = \"extrapolation\"\n"), 10);
    const std::string generated = "type = \"cartesian_tri\"\n";
    input.replace(input.find(generated), generated.size(), "type = \"file\"\nfilename = \"" + file + "\"\n");
    expect_matches_serial(input);
    comm::barrier();
}

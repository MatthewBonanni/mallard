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

/**
 * @brief Run the input distributed and serially; every rank compares the
 *        gathered distributed solution with its serial one.
 */
void expect_matches_serial(const std::string & input) {
    Solver distributed;
    distributed.init(parse_toml(input));
    distributed.run();
    distributed.copy_device_to_host();

    Solver serial;
    serial.set_distributed(false);
    serial.init(parse_toml(input));
    serial.run();
    serial.copy_device_to_host();

    const uint32_t n_global = serial.get_mesh()->n_cells;
    const auto & dist = distributed.get_distribution();
    std::vector<double> gathered(n_global * N_CONSERVATIVE, 0.0);
    std::vector<double> count(n_global, 0.0);
    for (uint32_t c = 0; c < distributed.get_mesh()->n_owned(); c++) {
        const uint64_t g = distributed.is_distributed() ? dist.global_cell[c] : c;
        FOR_I_CONSERVATIVE gathered[g * N_CONSERVATIVE + i] = distributed.h_conservatives(c, i);
        count[g] += 1.0;
    }
    comm::allreduce(std::span<double>(gathered), comm::Op::SUM);
    comm::allreduce(std::span<double>(count), comm::Op::SUM);

    EXPECT_EQ(distributed.get_step(), serial.get_step());
    EXPECT_NEAR(distributed.get_time(), serial.get_time(), 1e-12 * serial.get_time());
    double max_rel = 0.0;
    for (uint32_t g = 0; g < n_global; g++) {
        ASSERT_EQ(count[g], 1.0) << "cell " << g << " owned " << count[g] << " times";
        FOR_I_CONSERVATIVE {
            const double ref = serial.h_conservatives(g, i);
            max_rel = std::max(max_rel, std::abs(gathered[g * N_CONSERVATIVE + i] - ref) / (std::abs(ref) + 1e-3));
        }
    }
    // Ranks sum the same face fluxes in a different order: round-off only
    EXPECT_LT(max_rel, 1e-11) << "on " << comm::size() << " ranks";
}

} // namespace

TEST(MPITest, HaloExchangeFillsEveryHaloCellFromItsOwner) {
    Solver solver;
    solver.init(parse_toml(box_input("cartesian_tri", "type = \"MUSCL\"\n", EULER,
                                     bcs("type = \"extrapolation\"\n", "type = \"extrapolation\"\n",
                                         "type = \"symmetry\"\n", "type = \"symmetry\"\n"), 0)));
    if (!solver.is_distributed()) GTEST_SKIP() << "needs more than one rank";
    const auto & dist = solver.get_distribution();
    const uint32_t n_local = solver.get_mesh()->n_cells;
    Kokkos::View<rtype *[N_CONSERVATIVE]> U("U", n_local);
    auto h_U = Kokkos::create_mirror_view(U);
    for (uint32_t c = 0; c < n_local; c++) {
        FOR_I_CONSERVATIVE h_U(c, i) = c < dist.n_owned ? dist.global_cell[c] + 0.25 * i : -1.0;
    }
    Kokkos::deep_copy(U, h_U);
    HaloExchange(dist).exchange(U);
    Kokkos::deep_copy(h_U, U);
    for (uint32_t c = 0; c < n_local; c++) {
        FOR_I_CONSERVATIVE EXPECT_EQ(h_U(c, i), dist.global_cell[c] + 0.25 * i) << "local cell " << c;
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

TEST(MPITest, NavierStokesWithBoundaryConditionsMatchesSerial) {
    expect_matches_serial(box_input(
        "cartesian", "type = \"MUSCL\"\n", NS,
        bcs("type = \"dirichlet\"\nrho = \"1.0\"\nu = [\"0.3\", \"0.0\"]\np = \"1.0 + 0.1 * sin(6 * y) * sin(20 * t)\"\n",
            "type = \"p_out_average\"\np = 1.0\n",
            "type = \"wall_isothermal\"\nT = 1.2\nu = [0.1, 0.0]\n",
            "type = \"wall_adiabatic\"\n"),
        25));
}

namespace {

std::string io_dir() {
    // Same path on every rank (they share a file system)
    return (std::filesystem::temp_directory_path() / "mallard_mpi_io").string();
}

std::string restart_case(const std::string & init, uint32_t n_steps, const std::string & output) {
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
        FOR_I_CONSERVATIVE U[g * N_CONSERVATIVE + i] = solver.h_conservatives(c, i);
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
    reference.init(parse_toml(restart_case(init, 20, "")));
    reference.run();
    const auto U_ref = gather(reference);

    // Written by all ranks at step 10, continued by all ranks
    {
        Solver first;
        first.init(parse_toml(restart_case(init, 10, writer)));
        first.run();
    }
    comm::barrier();
    std::string input = restart_case(init, 20, "");
    input.replace(input.find("[initialize]\n") + 13, init.size(), from(dir + "/r_000010.restart"));
    Solver second;
    second.init(parse_toml(input));
    EXPECT_EQ(second.get_step(), 10u);
    second.run();
    EXPECT_LT(max_rel_diff(gather(second), U_ref), 1e-11);

    // The same file read by a single rank
    Solver serial;
    serial.set_distributed(false);
    serial.init(parse_toml(input));
    serial.run();
    EXPECT_LT(max_rel_diff(gather(serial), U_ref), 1e-11);
    comm::barrier();
}

TEST(MPITest, EveryCellIsInExactlyOneOutputPiece) {
    const std::string dir = io_dir() + "_vtu";
    if (comm::is_root()) std::filesystem::remove_all(dir);
    comm::barrier();
    Solver solver;
    solver.init(parse_toml(restart_case(BLAST, 2, "[[write_data]]\nprefix = \"" + dir + "/f\"\nformat = \"vtu\"\n"
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

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
#include <set>
#include <sstream>
#include <string>
#include <vector>

#include "comm.h"
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

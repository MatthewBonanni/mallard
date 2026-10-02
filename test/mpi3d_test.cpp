/**
 * @file mpi3d_test.cpp
 * @author Matthew Bonanni (mbonanni001@gmail.com)
 * @brief Distributed 3D runs reproduce the serial solution, on any number of ranks.
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
#include <iomanip>
#include <sstream>
#include <string>
#include <vector>

#include "comm.h"
#include "test_fixtures.h"
#include "solver.h"

namespace {

std::string blast_input(const std::string & mesh, const std::string & recon) {
    const std::string blob = "exp(-20 * ((x - 0.4)^2 + (y - 0.55)^2 + (z - 0.45)^2))";
    std::ostringstream s;
    s << "[run]\nn_steps = 3\ncfl = 0.5\n"
      << "[mesh]\ntype = \"" << mesh << "\"\nNx = 12\nNy = 12\nNz = 12\nLx = 1.0\nLy = 1.0\nLz = 1.0\n"
      << "[initialize]\ntype = \"analytical\"\nrho = \"1.0 + 0.5 * " << blob << "\"\n"
      << "u = [\"0.3\", \"-0.1\", \"0.2\"]\np = \"1.0 + 0.8 * " << blob << "\"\n";
    const char * zones[6] = {"left", "right", "bottom", "top", "back", "front"};
    for (int k = 0; k < 6; k++) {
        s << "[[boundaries]]\nname = \"" << zones[k] << "\"\ntype = \""
          << (k % 2 ? "extrapolation" : "symmetry") << "\"\n";
    }
    s << "[numerics]\nriemann_solver = \"HLLC\"\ntime_integrator = \"SSPRK3\"\n"
      << "[numerics.face_reconstruction]\n" << recon
      << "[physics]\ntype = \"euler\"\ngamma = 1.4\np_ref = 1.0\nT_ref = 1.0\nrho_ref = 1.0\n"
      << "[output]\ncheck_interval = 1000000\n";
    return s.str();
}

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
    const auto mesh = distributed.get_mesh();
    std::vector<double> gathered(n_global * N_CONSERVATIVE, 0.0);
    for (uint32_t c = 0; c < mesh->n_owned(); c++) {
        const uint64_t g = distributed.is_distributed() ? mesh->h_global_cell_id[c] : c;
        FOR_I_CONSERVATIVE gathered[g * N_CONSERVATIVE + i] = distributed.h_conservatives(c, i);
    }
    if (distributed.is_distributed()) comm::allreduce(std::span<double>(gathered), comm::Op::SUM);
    double max_rel = 0.0;
    for (uint32_t g = 0; g < n_global; g++) {
        FOR_I_CONSERVATIVE {
            const double ref = serial.h_conservatives(g, i);
            max_rel = std::max(max_rel, std::abs(gathered[g * N_CONSERVATIVE + i] - ref) / (std::abs(ref) + 1e-3));
        }
    }
    EXPECT_EQ(max_rel, 0.0) << "on " << comm::size() << " ranks";
}

} // namespace

TEST(MPI3DTest, TENOOnHexahedraMatchesSerial) {
    // Order 5 needs five distinct cell planes across a planar partition
    // interface; a halo that cuts the stencil search short must be deepened
    // rather than reported as a rank-deficient stencil
    expect_matches_serial(blast_input("cartesian", "type = \"TENO\"\norder = 5\n"));
}


TEST(MPI3DTest, SurfaceOutputOfAZoneMissingFromSomeRanks) {
    // Ranks whose part of the mesh does not touch the zone write empty pieces
    const std::string dir = (std::filesystem::temp_directory_path() / "mallard_mpi3d_faces").string();
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

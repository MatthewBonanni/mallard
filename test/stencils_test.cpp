/**
 * @file stencils_test.cpp
 * @brief Mesh and stencil fields: every cell's stencils as built for the run,
 *        their roles against the geometry, the summary fields, and the stencil
 *        file of a distributed run against the serial one.
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 */

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <filesystem>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

#include "check.h"
#include "comm.h"
#include "hdf5_output.h"
#include "scratch.h"
#include "solver.h"
#include "stencils.h"
#include "test_fixtures.h"

#ifdef Mallard_HAS_HDF5

namespace {

constexpr double LX = 1.5, LY = 1.0, LZ = 0.75;
constexpr int NX = 12, NY = 8;
// The fits round in the build's precision, and the condition number's iterations stop at 1e-10
const double FIT_TOLERANCE = std::max(1e-9, 100.0 * double(std::numeric_limits<rtype>::epsilon()));

/** @brief A box periodic in x (and z), with a wall at the bottom and a symmetry plane on top. */
std::string box(const std::string & recon, const std::string & extra = "",
                const std::string & top = "type = \"symmetry\"\n") {
    std::ostringstream s;
    s << "[run]\nn_steps = 1\ncfl = 0.5\n"
      << "[mesh]\ntype = \"cartesian\"\nNx = " << NX << "\nNy = " << NY << "\nLx = " << LX << "\nLy = " << LY << "\n"
      << (N_DIM == 3 ? "Nz = 6\nLz = 0.75\n[[periodic]]\nzones = [\"back\", \"front\"]\n" : "")
      << "[[periodic]]\nzones = [\"left\", \"right\"]\n"
      << "[initialize]\ntype = \"analytical\"\nrho = \"1.0\"\np = \"1.0\"\nu = [\"1.0\", \"0.0\""
      << (N_DIM == 3 ? ", \"0.0\"" : "") << "]\n"
      << "[numerics]\nriemann_solver = \"HLLC\"\n[numerics.face_reconstruction]\n" << recon
      << "[physics]\ntype = \"euler\"\ngamma = 1.4\np_ref = 1.0\nT_ref = 1.0\nrho_ref = 1.0\n"
      << "[[boundaries]]\nname = \"bottom\"\ntype = \"wall_adiabatic\"\n"
      << "[[boundaries]]\nname = \"top\"\n" << top;
    s << extra;
    return s.str();
}

/** @brief mallard check --mesh-fields PREFIX of the input, undistributed; the report. */
CheckReport check_fields(const std::string & input, CheckOptions options, const std::string & prefix) {
    options.mesh_fields = prefix;
    Solver solver;
    solver.set_distributed(false);
    return solver.check(parse_input_string(input), options);
}

/** @brief Cell centroids of a VTKHDF file, by row (the global id of a serial run). */
std::vector<std::array<double, 3>> centroids(const std::string & file) {
    const auto points = h5_values<double>(file, "VTKHDF/Points");
    const auto offsets = h5_values<int64_t>(file, "VTKHDF/Offsets");
    const auto connectivity = h5_values<int64_t>(file, "VTKHDF/Connectivity");
    std::vector<std::array<double, 3>> out;
    for (size_t c = 0; c + 1 < offsets.size(); c++) {
        std::array<double, 3> x{};
        for (int64_t k = offsets[c]; k < offsets[c + 1]; k++) {
            for (size_t d = 0; d < 3; d++) x[d] += points[size_t(connectivity[size_t(k)]) * 3 + d];
        }
        for (double & v : x) v /= double(offsets[c + 1] - offsets[c]);
        out.push_back(x);
    }
    return out;
}

std::string scratch(const std::string & name) {
    const std::filesystem::path dir = scratch_dir() / name;
    if (comm::is_root()) std::filesystem::remove_all(dir);
    comm::barrier();
    return (dir / "mesh").string();
}

}  // namespace

TEST(Stencils, RolesAndReachFollowTheSeamTheWallAndTheMirrors) {
    const std::string prefix = scratch("mallard_stencils_roles");
    CheckOptions options;
    options.all_stencils = true;
    const CheckReport r = check_fields(box("type = \"TENO\"\norder = 3\n"), options, prefix);
    ASSERT_TRUE(r.ok()) << r.errors().front();
    ASSERT_EQ(r.stencil_file, prefix + "_stencils.h5");
    if (!comm::is_root()) return;
    const std::string file = r.stencil_file;
    const auto x = centroids(r.mesh_fields_file);
    const auto cells = h5_values<uint64_t>(file, "stencils/cells");
    const auto offsets = h5_values<uint64_t>(file, "stencils/offsets");
    const auto sizes = h5_values<uint32_t>(file, "stencils/sizes");
    const auto degree = h5_values<uint8_t>(file, "stencils/degree");
    const auto reach = h5_values<uint8_t>(file, "stencils/reach");
    const auto condition = h5_values<double>(file, "stencils/condition");
    const auto member = h5_values<uint64_t>(file, "stencils/members/cell");
    const auto stencil = h5_values<uint8_t>(file, "stencils/members/stencil");
    const auto role = h5_values<uint8_t>(file, "stencils/members/role");
    const auto face_cell = h5_values<int64_t>(file, "stencils/members/face_cell");
    ASSERT_EQ(cells.size(), x.size());
    const double dy = LY / NY;
    size_t n_periodic = 0, n_mirror = 0;
    for (size_t t = 0; t < cells.size(); t++) {
        ASSERT_EQ(cells[t], t);
        EXPECT_EQ(degree[t], 2u);
        EXPECT_GE(condition[t], 1.0);
        EXPECT_TRUE(std::isfinite(condition[t]));
        uint64_t n_central = 0, total = 0;
        for (size_t k = 0; k < stencils::N_STENCILS; k++) total += sizes[t * stencils::N_STENCILS + k];
        EXPECT_EQ(offsets[t + 1] - offsets[t], total);
        bool wall = false, seam = false;
        for (uint64_t m = offsets[t]; m < offsets[t + 1]; m++) {
            const auto & xm = x[member[m]];
            n_central += stencil[m] == 0;
            // A periodic image is the copy across the seam: on the far side of the box
            const bool across = std::abs(xm[0] - x[t][0]) > 0.5 * LX || std::abs(xm[2] - x[t][2]) > 0.5 * LZ;
            EXPECT_EQ((role[m] & stencils::PERIODIC) != 0, across) << "cell " << t << " member " << member[m];
            n_periodic += across;
            seam = seam || across;
            const bool mirror = (role[m] & stencils::MIRROR) != 0;
            EXPECT_EQ(mirror, face_cell[m] >= 0);
            if (mirror) {
                // Mirrored across the wall or the symmetry plane: a face of a cell of the first or last row
                const double y = x[size_t(face_cell[m])][1];
                EXPECT_TRUE(y < dy || y > LY - dy) << "cell " << t;
                wall = wall || y < dy;
                n_mirror++;
            }
            wall = wall || xm[1] < dy;
        }
        EXPECT_EQ(n_central, sizes[t * stencils::N_STENCILS]);
        EXPECT_EQ((reach[t] & stencils::WALL) != 0, wall) << "cell " << t;
        EXPECT_EQ((reach[t] & stencils::SEAM) != 0, seam) << "cell " << t;
    }
    EXPECT_GT(n_periodic, 0u);
    EXPECT_GT(n_mirror, 0u);
    // The summary fields of the snapshot are the file's
    const auto size_field = h5_values<double>(r.mesh_fields_file, "VTKHDF/CellData/STENCIL_SIZE");
    const auto seam_field = h5_values<double>(r.mesh_fields_file, "VTKHDF/CellData/STENCIL_SEAM");
    for (size_t t = 0; t < cells.size(); t++) {
        EXPECT_EQ(size_field[t], double(sizes[t * stencils::N_STENCILS]));
        EXPECT_EQ(seam_field[t], (reach[t] & stencils::SEAM) ? 1.0 : 0.0);
    }
}

TEST(Stencils, SecondAndFirstOrderStencilsAreTheFaceNeighbors) {
    for (const std::string recon : {"type = \"MUSCL\"\n", "type = \"FO\"\n"}) {
        const std::string prefix = scratch("mallard_stencils_muscl");
        const CheckReport r = check_fields(box(recon), {}, prefix);
        ASSERT_TRUE(r.ok()) << r.errors().front();
        if (!comm::is_root()) continue;
        const auto x = centroids(r.mesh_fields_file);
        const auto size = h5_values<double>(r.mesh_fields_file, "VTKHDF/CellData/STENCIL_SIZE");
        const auto degree = h5_values<double>(r.mesh_fields_file, "VTKHDF/CellData/STENCIL_DEGREE");
        const auto condition = h5_values<double>(r.mesh_fields_file, "VTKHDF/CellData/STENCIL_CONDITION");
        const bool muscl = recon.find("MUSCL") != std::string::npos;
        for (size_t c = 0; c < x.size(); c++) {
            // The cell and one neighbor or ghost per face (MUSCL); the cell alone (first order)
            EXPECT_EQ(size[c], muscl ? 1.0 + 2.0 * N_DIM : 1.0) << recon << " cell " << c;
            EXPECT_EQ(degree[c], muscl ? 1.0 : 0.0);
            // MUSCL: the face fit on cubes of equal sides, ghosts mirrored, weighs every direction alike
            if (muscl) {
                EXPECT_NEAR(condition[c], 1.0, FIT_TOLERANCE) << "cell " << c;
            } else {
                EXPECT_EQ(condition[c], 0.0);
            }
        }
    }
}

TEST(Stencils, MUSCLWeightsDifferentiateLinearFieldsExactly) {
    // The face fit everywhere, then the vertex fit of the cells on a characteristic outlet
    const std::string outlet = "type = \"outlet_pressure\"\np = 1.0\nnscbc.L = 1.0\n";
    for (const std::string & top : {std::string("type = \"symmetry\"\n"), outlet}) {
        const std::string prefix = scratch("mallard_stencils_muscl_weights");
        CheckOptions options;
        options.all_stencils = true;
        options.stencil_weights = true;
        const CheckReport r = check_fields(box("type = \"MUSCL\"\n", "", top), options, prefix);
        ASSERT_TRUE(r.ok()) << r.errors().front();
        if (!comm::is_root()) continue;
        const auto x = centroids(r.mesh_fields_file);
        const auto condition = h5_values<double>(r.mesh_fields_file, "VTKHDF/CellData/STENCIL_CONDITION");
        const auto cells = h5_values<uint64_t>(r.stencil_file, "stencils/cells");
        const auto offsets = h5_values<uint64_t>(r.stencil_file, "stencils/offsets");
        const auto member = h5_values<uint64_t>(r.stencil_file, "stencils/members/cell");
        const auto role = h5_values<uint8_t>(r.stencil_file, "stencils/members/role");
        const auto weights = h5_values<double>(r.stencil_file, "stencils/members/weights");
        ASSERT_EQ(weights.size(), member.size() * N_DIM);
        size_t n_vertex_fit = 0;
        for (size_t t = 0; t < cells.size(); t++) {
            const auto & xt = x[cells[t]];
            // sum_s w_s (x_s - x_t) is the identity: x_s where the fit places member s
            double J[N_DIM][N_DIM] = {};
            for (uint64_t m = offsets[t]; m < offsets[t + 1]; m++) {
                std::array<double, 3> dx{};
                for (size_t d = 0; d < 3; d++) dx[d] = x[member[m]][d] - xt[d];
                if (role[m] & stencils::MIRROR) dx[1] = (xt[1] < 0.5 * LY ? -2.0 : 2.0) * (xt[1] < 0.5 * LY ? xt[1] : LY - xt[1]);
                const double L[3] = {LX, 0.0, LZ};
                for (const size_t d : {size_t(0), size_t(2)}) {
                    if (L[d] > 0.0 && std::abs(dx[d]) > 0.5 * L[d]) dx[d] -= std::copysign(L[d], dx[d]);
                }
                for (size_t a = 0; a < N_DIM; a++) {
                    for (size_t b = 0; b < N_DIM; b++) J[a][b] += weights[m * N_DIM + a] * dx[b];
                }
            }
            for (size_t a = 0; a < N_DIM; a++) {
                for (size_t b = 0; b < N_DIM; b++) EXPECT_NEAR(J[a][b], a == b ? 1.0 : 0.0, FIT_TOLERANCE) << "cell " << cells[t];
            }
            EXPECT_GE(condition[cells[t]], 1.0 - FIT_TOLERANCE);
            n_vertex_fit += offsets[t + 1] - offsets[t] > 1 + 2 * N_DIM;
        }
        // The outlet's cells take the vertex fit, whose stencils are wider
        EXPECT_EQ(n_vertex_fit > 0, top == outlet);
    }
}

TEST(Stencils, SectorWeightRowsHoldTheQuadraticCoefficientsOnly) {
    // Fifth order: the central fit's coefficients outnumber a sector fit's quadratic ones
    const std::string prefix = scratch("mallard_stencils_sector_weights");
    CheckOptions options;
    options.all_stencils = true;
    options.stencil_weights = true;
    const CheckReport r = check_fields(box("type = \"TENO\"\norder = 5\n"), options, prefix);
    ASSERT_TRUE(r.ok()) << r.errors().front();
    if (!comm::is_root()) return;
    const auto stencil = h5_values<uint8_t>(r.stencil_file, "stencils/members/stencil");
    const auto weights = h5_values<double>(r.stencil_file, "stencils/members/weights");
    ASSERT_FALSE(stencil.empty());
    const size_t width = weights.size() / stencil.size();
    ASSERT_GT(width, size_t(teno::NK_SMALL));
    size_t n_sector = 0;
    double largest = 0.0;
    for (size_t m = 0; m < stencil.size(); m++) {
        if (stencil[m] == 0) continue;
        n_sector++;
        for (size_t l = 0; l < teno::NK_SMALL; l++) largest = std::max(largest, std::abs(weights[m * width + l]));
        for (size_t l = teno::NK_SMALL; l < width; l++) ASSERT_EQ(weights[m * width + l], 0.0) << "member " << m;
    }
    EXPECT_GT(n_sector, 0u);
    EXPECT_GT(largest, 0.0);
}

TEST(Stencils, CellsAndRegionsSelectTheStencilsWrittenAndViewed) {
    const std::string prefix = scratch("mallard_stencils_cells");
    const std::string input = box("type = \"TENO\"\norder = 3\n");
    const CheckReport plain = check_fields(input, {}, prefix);
    ASSERT_TRUE(plain.ok()) << plain.errors().front();
    // The cells nearest the middle of the box and a corner, by global id
    uint64_t middle = 0, corner = 0;
    if (comm::is_root()) {
        const auto x = centroids(plain.mesh_fields_file);
        auto nearest = [&](double px, double py) {
            uint64_t best = 0;
            for (uint64_t c = 0; c < x.size(); c++) {
                if (std::hypot(x[c][0] - px, x[c][1] - py) < std::hypot(x[best][0] - px, x[best][1] - py)) best = c;
            }
            return best;
        };
        middle = nearest(0.5 * LX, 0.5 * LY);
        corner = nearest(0.0, 0.0);
    }
    middle = comm::allreduce(middle, comm::Op::MAX);
    corner = comm::allreduce(corner, comm::Op::MAX);
    CheckOptions options;
    options.stencil_cells = {std::max(middle, corner), std::min(middle, corner)};
    options.stencil_weights = true;
    const CheckReport r = check_fields(input, options, prefix);
    ASSERT_TRUE(r.ok()) << r.errors().front();
    if (comm::is_root()) {
        const auto x = centroids(r.mesh_fields_file);
        const auto cells = h5_values<uint64_t>(r.stencil_file, "stencils/cells");
        ASSERT_EQ(cells, (std::vector<uint64_t>{std::min(middle, corner), std::max(middle, corner)}));
        const size_t t = cells[0] == middle ? 0 : 1;
        const auto offsets = h5_values<uint64_t>(r.stencil_file, "stencils/offsets");
        const auto member = h5_values<uint64_t>(r.stencil_file, "stencils/members/cell");
        const auto stencil = h5_values<uint8_t>(r.stencil_file, "stencils/members/stencil");
        const auto role = h5_values<uint8_t>(r.stencil_file, "stencils/members/role");
        const auto weights = h5_values<double>(r.stencil_file, "stencils/members/weights");
        const size_t nk = N_DIM == 2 ? 5 : 9;  // the degree-2 monomials
        ASSERT_EQ(weights.size(), member.size() * nk);
        // The middle cell's central fit of u = x and u = y: the coefficients of the
        // monomials (x - x_c) / h and (y - y_c) / h are h, and the cross terms vanish
        double a[2][2] = {};
        for (uint64_t m = offsets[t]; m < offsets[t + 1]; m++) {
            if (stencil[m] != 0) continue;
            ASSERT_EQ(role[m] & stencils::MIRROR, 0) << "the middle cell's central stencil reaches no boundary";
            for (int l = 0; l < 2; l++) {
                for (int d = 0; d < 2; d++) a[l][d] += weights[m * nk + size_t(l)] * (x[member[m]][size_t(d)] - x[middle][size_t(d)]);
            }
        }
        EXPECT_GT(a[0][0], 0.0);
        EXPECT_NEAR(a[0][0], a[1][1], 1e-6 * a[0][0]);
        EXPECT_NEAR(a[0][1], 0.0, 1e-6 * a[0][0]);
        EXPECT_NEAR(a[1][0], 0.0, 1e-6 * a[0][0]);
        // STENCIL_<id>: 4 at the cell, 1 to 3 at its members, 0 elsewhere
        const auto field = h5_values<double>(r.mesh_fields_file, "VTKHDF/CellData/STENCIL_" + std::to_string(middle));
        std::vector<double> expected(field.size(), 0.0);
        for (uint64_t m = offsets[t]; m < offsets[t + 1]; m++) {
            double & e = expected[member[m]];
            e = double(int(e) | (stencil[m] == 0 ? 1 : 2));
        }
        expected[middle] = 4.0;
        EXPECT_EQ(field, expected);
    }

    options = {};
    options.stencil_region = N_DIM == 2 ? "{type = \"rectangle\", center = [0.1, 0.1], size = [0.2, 0.2]}"
                                        : "{type = \"box\", center = [0.1, 0.1, 0.0], size = [0.2, 0.2, 10.0]}";
    const CheckReport region = check_fields(input, options, prefix);
    ASSERT_TRUE(region.ok()) << region.errors().front();
    if (comm::is_root()) {
        const auto x = centroids(region.mesh_fields_file);
        const auto cells = h5_values<uint64_t>(region.stencil_file, "stencils/cells");
        EXPECT_EQ(cells.size(), size_t(4 * (N_DIM == 3 ? 6 : 1)));
        for (const uint64_t c : cells) {
            EXPECT_LT(x[c][0], 0.2);
            EXPECT_LT(x[c][1], 0.2);
        }
    }
    options = {};
    options.stencil_cells = {100000};
    EXPECT_NE(check_fields(box("type = \"FO\"\n"), options, prefix).errors().front().find("--cells: no cell 100000"),
              std::string::npos);
}

TEST(Stencils, WriteDataWritesTheMeshAndStencilFieldsOfTheCheck) {
    const std::string prefix = scratch("mallard_stencils_write_data");
    const std::string dir = std::filesystem::path(prefix).parent_path().string();
    const std::string input = box("type = \"TENO\"\norder = 3\n",
                                  "[[write_data]]\nprefix = \"" + dir + "/run\"\nformat = \"vtkhdf\"\nseries = false\ninterval = 1\n"
                                  "fields = [\"NON_ORTHOGONALITY\", \"STENCIL_SIZE\", \"STENCIL_WALL\"]\n");
    const CheckReport r = check_fields(input, {}, prefix);
    ASSERT_TRUE(r.ok()) << r.errors().front();
    Solver solver;
    solver.set_distributed(false);
    solver.init(parse_input_string(input));
    solver.run();
    comm::barrier();
    if (!comm::is_root()) return;
    const std::string run = dir + "/run_000000.vtkhdf";
    for (const char * field : {"NON_ORTHOGONALITY", "STENCIL_SIZE", "STENCIL_WALL"}) {
        EXPECT_EQ(h5_values<double>(run, std::string("VTKHDF/CellData/") + field),
                  h5_values<double>(r.mesh_fields_file, std::string("VTKHDF/CellData/") + field))
            << field;
    }
}

TEST(Stencils, FileOfADistributedRunMatchesTheSerialOne) {
    const std::string dir = std::filesystem::path(scratch("mallard_stencils_ranks")).parent_path().string();
    auto input = [&](const std::string & name) {
        return box("type = \"TENO\"\norder = 3\n", "[output]\nwrite_stencils = \"" + dir + "/" + name + ".h5\"\n");
    };
    Solver distributed;
    distributed.init(parse_input_string(input("d")));
    Solver serial;
    serial.set_distributed(false);
    serial.init(parse_input_string(input("s")));
    comm::barrier();
    if (!comm::is_root()) return;
    const auto a = h5_all_datasets(dir + "/d.h5"), b = h5_all_datasets(dir + "/s.h5");
    size_t compared = 0;
    for (const auto & [name, bytes] : b) {
        if (name.rfind("stencils/", 0) != 0) continue;
        EXPECT_TRUE(a.count(name) && a.at(name) == bytes) << name << " on " << comm::size() << " ranks";
        compared++;
    }
    EXPECT_EQ(compared, 11u);
    if (comm::size() > 1) {
        const auto ranks = h5_values<uint32_t>(dir + "/d.h5", "partition/rank");
        EXPECT_NE(std::count(ranks.begin(), ranks.end(), 0u), int64_t(ranks.size()));
    }
}

#endif

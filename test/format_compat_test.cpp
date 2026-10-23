/**
 * @file format_compat_test.cpp
 * @author Matthew Bonanni (mbonanni001@gmail.com)
 * @brief Restart and mesh files written by past releases (test/data/formats)
 *        are read, and files of newer format versions are rejected clearly.
 * @version 0.9
 * @date 2026-10-16
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 *
 */

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <type_traits>
#include <vector>

#include <toml.hpp>

#include "data_paths.h"
#include "data_writer.h"
#include "mesh_block.h"
#include "scratch.h"
#include "solver.h"
#include "test_fixtures.h"
#ifdef Mallard_HAS_HDF5
#include "hdf5_util.h"
#endif

namespace {

namespace fs = std::filesystem;

const fs::path FORMATS = fs::path(source_dir()) / "test" / "data" / "formats";

/** @brief Releases with fixtures, and the restart format version each case was written in. */
struct Fixture {
    std::string release, name;
    uint32_t version;
};

[[maybe_unused]] const std::vector<Fixture> RESTART_FIXTURES = {
    {"v0.6.0", "gas", 2},   {"v0.6.0", "statistics", 3}, {"v0.6.0", "mixture", 2},
    {"v0.7.0", "gas", 2},   {"v0.7.0", "statistics", 3}, {"v0.7.0", "mixture", 2}, {"v0.7.0", "nscbc", 4},
    {"v0.8.0", "gas", 2},   {"v0.8.0", "statistics", 3}, {"v0.8.0", "mixture", 2}, {"v0.8.0", "nscbc", 4},
};

/** @brief The fixture's input, without output; from a restart file if one is given. */
[[maybe_unused]] Input fixture_input(const std::string & name, const fs::path & restart = {}) {
    Input input = parse_input((FORMATS / (name + ".toml")).string());
    input.as_table().erase("write_data");
    auto & physics = input.at("physics");
    if (physics.contains("mechanism")) {
        physics["mechanism"] = (FORMATS / physics.at("mechanism").as_string()).lexically_normal().string();
    }
    if (!restart.empty()) {
        Input init(Input::table_type{});
        init["type"] = "restart";
        init["file"] = restart.string();
        input["initialize"] = init;
    }
    return input;
}

/** @brief Each conserved variable and species density of every cell, after init (and the run's steps if run). */
[[maybe_unused]] std::vector<std::vector<double>> solver_state(const Input & input, bool run) {
    Solver solver;
    solver.init(input);
    if (run) {
        solver.run();
        solver.copy_device_to_host();
    }
    const uint32_t n_cells = solver.get_mesh()->n_cells;
    const size_t n_species = solver.h_species.is_allocated() ? solver.h_species.extent(1) : 0;
    std::vector<std::vector<double>> state(N_CONSERVATIVE + n_species, std::vector<double>(n_cells));
    for (uint32_t i = 0; i < n_cells; i++) {
        for (uint32_t v = 0; v < N_CONSERVATIVE; v++) state[v][i] = double(solver.h_conservatives(i, v));
        for (size_t k = 0; k < n_species; k++) state[N_CONSERVATIVE + k][i] = double(solver.h_species(i, k));
    }
    return state;
}

[[maybe_unused]] uint32_t restart_version(const fs::path & file) {
    std::ifstream in(file, std::ios::binary);
    char header[20];
    in.read(header, sizeof(header));
    uint32_t version;
    std::memcpy(&version, header + 16, sizeof(version));
    return version;
}

template <typename F>
std::string error_of(F && f) {
    try {
        f();
    } catch (const std::exception & e) {
        return e.what();
    }
    return "";
}

} // namespace

TEST(FormatCompatTest, RestartFilesOfPastReleasesStartARun) {
    if constexpr (N_DIM != 2 || !std::is_same_v<rtype, double>) {
        GTEST_SKIP() << "2D double-precision fixtures";
    } else {
        for (const Fixture & f : RESTART_FIXTURES) {
            SCOPED_TRACE(f.release + "/" + f.name);
            const fs::path first = FORMATS / f.release / (f.name + "_000000.restart");
            const fs::path last = FORMATS / f.release / (f.name + "_000002.restart");
            ASSERT_EQ(restart_version(last), f.version);

            // Step 0 holds the analytical initial state, which 2D builds compute the same since 0.5;
            // step 2 the two steps this build takes from it, up to the round-off of scheme changes
            const auto analytical = solver_state(fixture_input(f.name), false);
            const auto from_first = solver_state(fixture_input(f.name, first), false);
            const auto from_last = solver_state(fixture_input(f.name, last), false);
            const auto rerun = solver_state(fixture_input(f.name, first), true);
            ASSERT_EQ(from_first.size(), analytical.size());
            auto size_of = [](const std::vector<double> & x) {
                double m = 0;
                for (double value : x) m = std::max(m, std::abs(value));
                return m;
            };
            bool moved = false;
            for (size_t v = 0; v < analytical.size(); v++) {
                // Species densities, some zero everywhere, compare with the density
                const double size = std::max(size_of(analytical[v]), v < N_CONSERVATIVE ? 0.0 : size_of(analytical[0]));
                for (size_t i = 0; i < analytical[v].size(); i++) {
                    ASSERT_NEAR(from_first[v][i], analytical[v][i], 1e-12 * size) << "variable " << v << ", cell " << i;
                    ASSERT_NEAR(from_last[v][i], rerun[v][i], 1e-10 * size) << "variable " << v << ", cell " << i;
                    moved = moved || from_last[v][i] != from_first[v][i];
                }
            }
            EXPECT_TRUE(moved);

            const RestartData data = read_restart(last.string());
            EXPECT_EQ(data.step, 2u);
            EXPECT_GT(data.t, 0.0);
            if (f.name == "statistics") {
                EXPECT_NE(data.find("MEAN_RHO"), nullptr);
                EXPECT_NE(data.find("COV_P_P"), nullptr);
                EXPECT_FALSE(data.attributes.empty());
            }
            if (f.name == "mixture") {
                EXPECT_NE(data.find("RHOY_H2"), nullptr);
            }
            // One face per boundary edge of the inlet and the outlet (Ny = 5 each)
            if (f.name == "nscbc") {
                EXPECT_EQ(data.faces.keys.size(), 10u);
            }
        }
    }
}

TEST(FormatCompatTest, RestartFilesOfANewerFormatAreRejectedClearly) {
    if constexpr (N_DIM != 2) {
        GTEST_SKIP() << "2D fixtures";
    } else {
        const fs::path file = scratch_dir() / "mallard_newer.restart";
        fs::copy_file(FORMATS / "v0.8.0" / "gas_000000.restart", file, fs::copy_options::overwrite_existing);
        {
            std::fstream out(file, std::ios::in | std::ios::out | std::ios::binary);
            const uint32_t version = 99;
            out.seekp(16);
            out.write(reinterpret_cast<const char *>(&version), sizeof(version));
        }
        const std::string error = error_of([&] { read_restart(file.string()); });
        EXPECT_NE(error.find("format version 99 is from a newer Mallard"), std::string::npos) << error;
        EXPECT_NE(error.find("reads restart files of versions 1 to 6"), std::string::npos) << error;
        fs::remove(file);
    }
}

TEST(FormatCompatTest, RestartFilesOfAnotherBuildNameTheDifference) {
    const std::string error = error_of([] { read_restart((FORMATS / "v0.8.0" / "gas_000000.restart").string()); });
    // The precision is checked first
    if constexpr (!std::is_same_v<rtype, double>) {
        EXPECT_NE(error.find("different floating-point precision"), std::string::npos) << error;
    } else if constexpr (N_DIM != 2) {
        EXPECT_NE(error.find("holds a 2D run"), std::string::npos) << error;
    } else {
        EXPECT_EQ(error, "");
    }
}

#ifdef Mallard_HAS_HDF5

namespace {

[[maybe_unused]] void expect_same_mesh(const MeshBlock & a, const MeshBlock & b) {
    EXPECT_EQ(a.cell_offsets, b.cell_offsets);
    EXPECT_EQ(a.cell_nodes, b.cell_nodes);
    EXPECT_EQ(a.node_coords, b.node_coords);
    EXPECT_EQ(a.face_offsets, b.face_offsets);
    EXPECT_EQ(a.face_nodes, b.face_nodes);
    EXPECT_EQ(a.face_zone, b.face_zone);
    EXPECT_EQ(a.zone_names, b.zone_names);
    EXPECT_EQ(a.face_high_order_offsets, b.face_high_order_offsets);
    EXPECT_EQ(a.face_high_order_nodes, b.face_high_order_nodes);
}

} // namespace

TEST(FormatCompatTest, MeshFilesOfPastReleasesAreRead) {
    if constexpr (N_DIM != 2 || !std::is_same_v<rtype, double>) {
        GTEST_SKIP() << "2D fixtures, compared with double-precision meshes";
    } else {
        const MeshBlock generated = read_mesh_block(fixture_input("gas"));
        for (const std::string release : {"v0.6.0", "v0.7.0", "v0.8.0"}) {
            SCOPED_TRACE(release);
            expect_same_mesh(read_mesh_h5((FORMATS / release / "mesh.h5").string(), true), generated);
        }
        // Curved boundaries: the second-order nodes of the walls
        const MeshBlock curved = read_mesh_h5((FORMATS / "v0.8.0" / "curved_mesh.h5").string(), true);
        ASSERT_FALSE(curved.face_high_order_nodes.empty());
        expect_same_mesh(curved, read_file_block((FORMATS / "annulus.msh").string()));
    }
}

TEST(FormatCompatTest, MeshFilesOfANewerFormatAreRejectedClearly) {
    if constexpr (N_DIM != 2) {
        GTEST_SKIP() << "2D fixtures";
    } else {
        const fs::path file = scratch_dir() / "mallard_newer_mesh.h5";
        fs::copy_file(FORMATS / "v0.8.0" / "mesh.h5", file, fs::copy_options::overwrite_existing);
        {
            h5::Handle h(H5Fopen(file.string().c_str(), H5F_ACC_RDWR, H5P_DEFAULT), H5Fclose, "H5Fopen");
            h5::Handle root(H5Gopen2(h, "/", H5P_DEFAULT), H5Gclose, "H5Gopen2");
            h5::check(H5Adelete(root, "version"), "H5Adelete");
            h5::write_int_attribute(root, "version", 99);
        }
        const std::string error = error_of([&] { read_mesh_h5(file.string(), true); });
        EXPECT_NE(error.find("format version 99 is from a newer Mallard"), std::string::npos) << error;
        EXPECT_NE(error.find("reads HDF5 mesh files of version 1."), std::string::npos) << error;
        fs::remove(file);
    }
}

#endif

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

#include <array>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <vector>

#include "common_io.h"
#include "compression.h"
#include "data_writer.h"
#include "gmsh_fixtures.h"
#include "json_reader.h"
#include "hdf5_output.h"
#include "mesh_block.h"
#include "provenance.h"
#include "test_fixtures.h"
#include "solver.h"
#include "scratch.h"

TEST(IOTest, ProvenanceHashesTheMeshFile) {
    const std::string file = write_temp("mallard_provenance_mesh.msh", jittered_mixed_mesh(6));
    auto input = [&](const std::string & output) {
        std::ostringstream s;
        s << "[run]\nn_steps = 1\ncfl = 0.25\n[mesh]\ntype = \"file\"\nfilename = \"" << file << "\"\n"
          << "[initialize]\ntype = \"analytical\"\nrho = \"1.0\"\nu = [\"0.1\", \"0.0\"]\np = \"1.0\"\n";
        for (const char * name : {"left", "right", "top", "bottom"}) {
            s << "[[boundaries]]\nname = \"" << name << "\"\ntype = \"extrapolation\"\n";
        }
        s << "[numerics]\nriemann_solver = \"HLLC\"\n[numerics.face_reconstruction]\ntype = \"FO\"\n"
          << "[physics]\ntype = \"euler\"\ngamma = 1.4\np_ref = 1.0\nT_ref = 1.0\nrho_ref = 1.0\n[output]\n" << output;
        return s.str();
    };
    Solver hashed;
    hashed.init(parse_toml(input("")));
    const auto p = json_reader::parse(provenance::json());
    EXPECT_EQ(p["mesh"]["type"].string, "file");
    EXPECT_EQ(p["mesh"]["file"].string, file);
    std::ifstream in(file, std::ios::binary);
    EXPECT_EQ(p["mesh"]["sha256"].string, provenance::sha256(std::string(std::istreambuf_iterator<char>(in), {})));

    Solver unhashed;
    unhashed.init(parse_toml(input("hash_mesh = false\n")));
    const auto q = json_reader::parse(provenance::json());
    EXPECT_EQ(q["mesh"]["file"].string, file);
    EXPECT_FALSE(q["mesh"].has("sha256"));
}

TEST(IOTest, BoundaryZoneOutputCarriesAdjacentCellValues) {
    const std::string dir = (scratch_dir() / "mallard_surface_test").string();
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
        EXPECT_RTYPE_EQ(static_cast<rtype>(p), solver.h_primitives(c, 2));
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
        EXPECT_RTYPE_EQ(static_cast<rtype>(u[0]), solver.h_primitives(c, 0));
        EXPECT_RTYPE_EQ(static_cast<rtype>(u[1]), solver.h_primitives(c, 1));
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
    const std::string dir = (scratch_dir() / "mallard_fixed_dt").string();
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
    const std::string dir = (scratch_dir() / "mallard_cfl_output").string();
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
    const std::string dir = (scratch_dir() / "mallard_h5_resume").string();
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

namespace {

std::string file_contents(const std::string & filename) {
    std::ifstream in(filename, std::ios::binary);
    std::stringstream buffer;
    buffer << in.rdbuf();
    return buffer.str();
}

/** @brief Every file under dir, by path relative to it. */
std::map<std::string, std::string> files_under(const std::string & dir) {
    std::map<std::string, std::string> files;
    for (const auto & entry : std::filesystem::recursive_directory_iterator(dir)) {
        if (entry.is_regular_file()) {
            files[std::filesystem::relative(entry.path(), dir).string()] = file_contents(entry.path().string());
        }
    }
    return files;
}

/**
 * @brief A short run with statistics that writes volume, surface, HDF5 and
 *        restart files under dir; snapshot_keys and restart_keys are added to
 *        the snapshot and restart writers.
 */
std::string output_case(const std::string & dir, const std::string & output_keys, const std::string & snapshot_keys,
                        const std::string & restart_keys) {
    std::ostringstream s;
    s << "[run]\nn_steps = 12\ncfl = 0.25\n"
      << "[mesh]\ntype = \"cartesian_tri\"\nNx = 24\nNy = 20\nLx = 1.0\nLy = 1.0\n"
      << "[initialize]\ntype = \"analytical\"\nrho = \"1.0 + (x < 0.4 ? 1.0 : 0.0)\"\nu = [\"0.2\", \"0.1\"]\n"
      << "p = \"x < 0.4 ? 2.0 : 1.0\"\n"
      << "[[boundaries]]\nname = \"left\"\ntype = \"extrapolation\"\n"
      << "[[boundaries]]\nname = \"right\"\ntype = \"extrapolation\"\n"
      << "[[boundaries]]\nname = \"top\"\ntype = \"symmetry\"\n"
      << "[[boundaries]]\nname = \"bottom\"\ntype = \"wall_adiabatic\"\n"
      << "[numerics]\nriemann_solver = \"HLLC\"\ntime_integrator = \"SSPRK3\"\n"
      << "[numerics.face_reconstruction]\ntype = \"MUSCL\"\n"
      << "[physics]\ntype = \"euler\"\ngamma = 1.4\np_ref = 1.0\nT_ref = 1.0\nrho_ref = 1.0\n"
      << "[statistics]\nfields = [\"P\"]\n"
      << "[output]\ncheck_interval = 1000000\n" << output_keys
      << "[[write_data]]\nprefix = \"" << dir << "/flow\"\nformat = \"vtu\"\ninterval = 4\n"
      << "variables = [\"RHO\", \"U\", \"P\", \"MEAN_P\"]\n" << snapshot_keys
      << "[[write_data]]\nprefix = \"" << dir << "/wall\"\nformat = \"vtu\"\ngeometry = \"bottom\"\ninterval = 4\n"
      << "variables = [\"P\"]\n"
      << "[[write_data]]\nprefix = \"" << dir << "/restart\"\nformat = \"restart\"\ninterval = 6\n" << restart_keys;
    if (have_hdf5()) {
        s << "[[write_data]]\nprefix = \"" << dir << "/h5/flow\"\nformat = \"hdf5\"\ninterval = 4\n"
          << "variables = [\"RHO\", \"U\", \"P\", \"MEAN_P\"]\n" << snapshot_keys
          << "[[write_data]]\nprefix = \"" << dir << "/vtkhdf/flow\"\nformat = \"vtkhdf\"\ninterval = 4\n"
          << "variables = [\"RHO\", \"U\", \"P\", \"MEAN_P\"]\n" << snapshot_keys
          << "[[write_data]]\nprefix = \"" << dir << "/vtkhdf/series\"\nseries = true\ninterval = 4\n"
          << "variables = [\"RHO\", \"U\"]\n" << snapshot_keys
          << "[[write_data]]\nprefix = \"" << dir << "/vtkhdf/wall\"\ngeometry = \"bottom\"\ninterval = 4\n"
          << "variables = [\"P\"]\n" << snapshot_keys;
    }
    return s.str();
}

void run_case(const std::string & input) {
    Solver solver;
    solver.init(parse_toml(input));
    solver.run();
}

/**
 * @brief The arrays of a VTU file with appended raw data, in file order:
 *        each as its bytes, decompressed if the file is compressed.
 */
std::vector<std::string> vtu_arrays(const std::string & filename) {
    const std::string text = file_contents(filename);
    const std::string marker = "<AppendedData encoding=\"raw\">\n_";
    const size_t base = text.find(marker) + marker.size();
    const bool zlib = text.find("compressor=\"vtkZLibDataCompressor\"") < base;
    std::vector<std::string> arrays;
    for (size_t at = 0; (at = text.find("offset=\"", at)) < base; at++) {
        size_t p = base + std::stoull(text.substr(at + 8));
        auto next = [&]() {
            uint64_t value = 0;
            std::memcpy(&value, text.data() + p, sizeof(value));
            p += sizeof(value);
            return value;
        };
        if (!zlib) {
            const uint64_t n = next();
            arrays.push_back(text.substr(p, n));
            continue;
        }
        const uint64_t n_blocks = next(), block = next(), last = next();
        std::vector<uint64_t> sizes(n_blocks);
        for (auto & size : sizes) size = next();
        std::string bytes;
        for (uint64_t b = 0; b < n_blocks; b++) {
            std::string out(b + 1 == n_blocks && last > 0 ? last : block, '\0');
            compression::inflate(text.data() + p, sizes[b], out.data(), out.size());
            p += sizes[b];
            bytes += out;
        }
        arrays.push_back(bytes);
    }
    return arrays;
}

std::vector<rtype> as_reals(const std::string & bytes) {
    std::vector<rtype> values(bytes.size() / sizeof(rtype));
    std::memcpy(values.data(), bytes.data(), values.size() * sizeof(rtype));
    return values;
}

void expect_within_relative_error(const std::vector<rtype> & rounded, const std::vector<rtype> & exact, double bound,
                                  const std::string & what) {
    ASSERT_EQ(rounded.size(), exact.size()) << what;
    size_t n_changed = 0;
    for (size_t i = 0; i < exact.size(); i++) {
        EXPECT_LE(std::abs(double(rounded[i]) - double(exact[i])), bound * std::abs(double(exact[i]))) << what << " " << i;
        n_changed += rounded[i] != exact[i];
    }
    EXPECT_GT(n_changed, 0u) << what;
}

} // namespace

TEST(IOTest, AsynchronousOutputWritesTheSameFiles) {
    // The snapshot taken when a file is due is what lands in it, although the
    // run has moved on while it is written
    const std::string dir = (scratch_dir() / "mallard_async_output").string();
    std::vector<std::pair<std::string, std::string>> variants = {{"", ""}};
    if (compression::have_zlib()) {
        variants.emplace_back("compression = \"zlib\"\n", "compression = \"zlib\"\n");
        variants.emplace_back("compression = \"zlib\"\nrelative_error = 1e-4\n",
                              "compression = \"zlib\"\nchunk_cells = 100\n");
    }
    for (const auto & [snapshot_keys, restart_keys] : variants) {
        std::filesystem::remove_all(dir);
        run_case(output_case(dir + "/sync", "async = false\n", snapshot_keys, restart_keys));
        run_case(output_case(dir + "/async", "io_threads = 3\n", snapshot_keys, restart_keys));
        const auto sync = files_under(dir + "/sync"), async = files_under(dir + "/async");
        EXPECT_EQ(sync.size(), have_hdf5() ? 34u : 13u) << snapshot_keys;
        ASSERT_EQ(async.size(), sync.size()) << snapshot_keys;
        for (const auto & [name, bytes] : sync) {
            ASSERT_TRUE(async.count(name)) << name;
#ifdef Mallard_HAS_HDF5
            if (name.ends_with(".h5")) {
                EXPECT_EQ(h5_datasets(dir + "/async/" + name), h5_datasets(dir + "/sync/" + name)) << name;
                continue;
            }
            if (name.ends_with(".vtkhdf")) {
                EXPECT_EQ(h5_all_datasets(dir + "/async/" + name), h5_all_datasets(dir + "/sync/" + name)) << name;
                continue;
            }
#endif
            EXPECT_TRUE(without_provenance(async.at(name)) == without_provenance(bytes)) << name << " " << snapshot_keys;
        }
    }
    std::filesystem::remove_all(dir);
}

TEST(IOTest, AWriterGoingAwayFinishesItsFileFirst) {
    // A run that stops on an error destroys its writers without finish():
    // the file being written in the background is still written whole
    const std::string dir = (scratch_dir() / "mallard_async_unwind").string();
    std::filesystem::remove_all(dir);
    auto mesh = make_mesh("cartesian_tri", 300, 200);
    Kokkos::View<rtype **>::host_mirror_type values("values", mesh->n_cells, 1);
    for (uint32_t i = 0; i < mesh->n_cells; i++) values(i, 0) = rtype(i) / rtype(7);
    std::vector<Data> data = {Data("RHO", Kokkos::subview(values, Kokkos::ALL(), 0))};
    for (bool async : {false, true}) {
        DataWriter writer;
        writer.init(parse_toml("prefix = \"" + dir + (async ? "/a" : "/s") + "\"\nformat = \"vtu\"\ninterval = 1\n"
                               "variables = [\"RHO\"]\n"),
                    data, mesh);
        writer.set_async(async);
        writer.write(0, 0.0_r);
    }
    const std::string sync = file_contents(dir + "/s_000000.vtu");
    EXPECT_GT(sync.size(), 120000u * sizeof(rtype));
    EXPECT_TRUE(file_contents(dir + "/a_000000.vtu") == sync);
    std::filesystem::remove_all(dir);
}

TEST(IOTest, CompressedFilesHoldTheSameValuesOrValuesWithinTheRelativeError) {
    if (!compression::have_zlib()) GTEST_SKIP() << "built without zlib";
    const std::string dir = (scratch_dir() / "mallard_compressed_output").string();
    std::filesystem::remove_all(dir);
    run_case(output_case(dir + "/raw", "", "", ""));
    run_case(output_case(dir + "/zlib", "", "compression = \"zlib\"\n", "compression = \"zlib\"\nchunk_cells = 100\n"));
    run_case(output_case(dir + "/lossy", "", "compression = \"zlib\"\nrelative_error = 1e-3\n", ""));

    // Restart files: the same contents, whole or for some cells
    for (const char * file : {"/restart_000006.restart", "/restart_000012.restart"}) {
        const RestartData raw = read_restart(dir + "/raw" + file), zlib = read_restart(dir + "/zlib" + file);
        EXPECT_LT(std::filesystem::file_size(dir + "/zlib" + file), std::filesystem::file_size(dir + "/raw" + file));
        EXPECT_EQ(zlib.step, raw.step);
        EXPECT_EQ(zlib.t, raw.t);
        EXPECT_EQ(zlib.names, raw.names);
        EXPECT_EQ(zlib.fields, raw.fields);
        EXPECT_EQ(zlib.attributes, raw.attributes);
        EXPECT_FALSE(zlib.attributes.empty());
        EXPECT_EQ(zlib.faces.keys, raw.faces.keys);
        const std::vector<uint64_t> cells = {907, 3, 4, 5, 958, 0, 480};
        const RestartData some = read_restart(dir + "/zlib" + file, &cells);
        for (size_t v = 0; v < raw.names.size(); v++) {
            for (size_t k = 0; k < cells.size(); k++) EXPECT_EQ(some.fields[v][k], raw.fields[v][cells[k]]);
        }
    }

    // VTU: the cell arrays, then points, connectivity, offsets and types
    const std::vector<std::string> raw = vtu_arrays(dir + "/raw/flow_000008.vtu");
    ASSERT_EQ(raw.size(), 8u);
    EXPECT_EQ(raw[0].size(), 960 * sizeof(rtype));
    EXPECT_EQ(vtu_arrays(dir + "/zlib/flow_000008.vtu"), raw);
    EXPECT_LT(std::filesystem::file_size(dir + "/zlib/flow_000008.vtu"),
              std::filesystem::file_size(dir + "/raw/flow_000008.vtu"));
    const std::vector<std::string> lossy = vtu_arrays(dir + "/lossy/flow_000008.vtu");
    ASSERT_EQ(lossy.size(), raw.size());
    for (size_t a = 0; a < 4; a++) expect_within_relative_error(as_reals(lossy[a]), as_reals(raw[a]), 1e-3, "vtu");
    for (size_t a = 4; a < raw.size(); a++) EXPECT_EQ(lossy[a], raw[a]) << "mesh array " << a;

#ifdef Mallard_HAS_HDF5
    const auto h5_raw = h5_datasets(dir + "/raw/h5/flow_000008.h5");
    EXPECT_EQ(h5_datasets(dir + "/zlib/h5/flow_000008.h5"), h5_raw);
    EXPECT_EQ(h5_datasets(dir + "/lossy/h5/flow_mesh.h5"), h5_datasets(dir + "/raw/h5/flow_mesh.h5"));
    const auto h5_lossy = h5_datasets(dir + "/lossy/h5/flow_000008.h5");
    ASSERT_EQ(h5_lossy.size(), h5_raw.size());
    for (const auto & [name, bytes] : h5_raw) {
        expect_within_relative_error(as_reals(std::string(h5_lossy.at(name).begin(), h5_lossy.at(name).end())),
                                     as_reals(std::string(bytes.begin(), bytes.end())), 1e-3, name);
    }

    // VTKHDF: deflated datasets with the same values; rounding changes only the cell values
    for (const char * file : {"/vtkhdf/flow_000008.vtkhdf", "/vtkhdf/series.vtkhdf", "/vtkhdf/wall_000008.vtkhdf"}) {
        const auto raw_file = h5_all_datasets(dir + "/raw" + file);
        EXPECT_EQ(h5_all_datasets(dir + "/zlib" + file), raw_file) << file;
        if (!std::string(file).starts_with("/vtkhdf/wall")) {
            EXPECT_LT(std::filesystem::file_size(dir + "/zlib" + file),
                      std::filesystem::file_size(dir + "/raw" + file))
                << file;
        }
        const auto lossy_file = h5_all_datasets(dir + "/lossy" + file);
        ASSERT_EQ(lossy_file.size(), raw_file.size()) << file;
        for (const auto & [name, bytes] : raw_file) {
            if (!name.starts_with("VTKHDF/CellData/")) {
                EXPECT_EQ(lossy_file.at(name), bytes) << file << " " << name;
                continue;
            }
            expect_within_relative_error(
                as_reals(std::string(lossy_file.at(name).begin(), lossy_file.at(name).end())),
                as_reals(std::string(bytes.begin(), bytes.end())), 1e-3, std::string(file) + " " + name);
        }
    }
#endif
    std::filesystem::remove_all(dir);
}

TEST(IOTest, RoundingKeepsEveryValueWithinTheRelativeError) {
    std::vector<double> values = {1.0, -1.0, 1.5, 0.1, -3.7e-200, 6.02e23, std::numeric_limits<double>::max(),
                                  -std::numeric_limits<double>::max(), std::numeric_limits<double>::min(),
                                  std::numeric_limits<double>::denorm_min(), 0.0, -0.0,
                                  std::numeric_limits<double>::infinity(), std::numeric_limits<double>::quiet_NaN()};
    // Random signs, 52 random mantissa bits and exponents within 2^±60
    uint64_t state = 12345;
    for (int i = 0; i < 10000; i++) {
        state = state * 6364136223846793005ULL + 1442695040888963407ULL;
        const uint64_t mantissa = (state >> 12) & ((uint64_t(1) << 52) - 1);
        const uint64_t bits = (state & 1) << 63 | (uint64_t(1023 - 60) + (state >> 1) % 121) << 52 | mantissa;
        double x;
        std::memcpy(&x, &bits, sizeof(x));
        values.push_back(x);
    }
    for (double bound : {0.4, 1e-3, 1e-7, 1e-15, 1e-17}) {
        std::vector<double> rounded = values;
        compression::round_to_relative_error(rounded, bound);
        size_t n_changed = 0;
        for (size_t i = 0; i < values.size(); i++) {
            if (!std::isfinite(values[i])) {
                EXPECT_EQ(std::isnan(rounded[i]), std::isnan(values[i]));
                if (!std::isnan(values[i])) {
                    EXPECT_EQ(rounded[i], values[i]);
                }
                continue;
            }
            EXPECT_LE(std::abs(rounded[i] - values[i]), bound * std::abs(values[i])) << values[i] << " " << bound;
            EXPECT_EQ(std::signbit(rounded[i]), std::signbit(values[i]));
            n_changed += rounded[i] != values[i];
        }
        // Rounded values need fewer bits: rounding them again leaves them as they are
        std::vector<double> twice = rounded;
        compression::round_to_relative_error(twice, bound);
        EXPECT_EQ(std::memcmp(twice.data(), rounded.data(), twice.size() * sizeof(double)), 0) << bound;
        // Dropping at least 3 random bits changes 7 values in 8
        if (bound >= 1e-15) {
            EXPECT_GT(n_changed, values.size() * 3 / 4) << bound;
        }
    }
    std::vector<float> floats = {1.0f, 0.3f, -2.5e-30f, 3.0e30f, std::numeric_limits<float>::max()};
    const std::vector<float> exact = floats;
    compression::round_to_relative_error(floats, 1e-3);
    for (size_t i = 0; i < floats.size(); i++) {
        EXPECT_LE(double(std::abs(floats[i] - exact[i])), 1e-3 * double(std::abs(exact[i])));
    }
    EXPECT_NE(floats[1], exact[1]);
}

TEST(IOTest, VTKHDFIsTheDefaultFormatWithHDF5) {
    auto mesh = make_mesh("cartesian_tri", 3, 2);
    Kokkos::View<rtype **>::host_mirror_type values("values", mesh->n_cells, 1);
    std::vector<Data> data = {Data("RHO", Kokkos::subview(values, Kokkos::ALL(), 0))};
    DataWriter writer;
    writer.init(parse_toml("prefix = \"" + (scratch_dir() / "default_format").string() +
                           "\"\ninterval = 1\nvariables = [\"RHO\"]\n"),
                data, mesh);
    EXPECT_EQ(writer.get_format(), have_hdf5() ? DataFormat::VTKHDF : DataFormat::VTU);
    DataWriter vtu;
    EXPECT_THROW(vtu.init(parse_toml("prefix = \"x\"\nformat = \"vtu\"\nseries = true\ninterval = 1\n"
                                     "variables = [\"RHO\"]\n"),
                          data, mesh),
                 InputError);
}

#ifdef Mallard_HAS_HDF5

namespace {

/** @brief A snapshot's mesh and cells as VTKHDF stores them. */
struct VtkhdfGrid {
    std::vector<double> points;  // 3 per point
    std::vector<int64_t> connectivity, offsets;
    std::vector<uint8_t> types;

    explicit VtkhdfGrid(const std::string & file)
        : points(h5_values<double>(file, "VTKHDF/Points")),
          connectivity(h5_values<int64_t>(file, "VTKHDF/Connectivity")),
          offsets(h5_values<int64_t>(file, "VTKHDF/Offsets")),
          types(h5_values<uint8_t>(file, "VTKHDF/Types")) {}

    std::array<double, 3> point(size_t c, size_t k) const {
        const size_t p = static_cast<size_t>(connectivity[static_cast<size_t>(offsets[c]) + k]);
        return {points[3 * p], points[3 * p + 1], points[3 * p + 2]};
    }
    size_t size(size_t c) const { return static_cast<size_t>(offsets[c + 1] - offsets[c]); }
};

std::string vtkhdf_type(const std::string & file) {
    h5::Handle f(H5Fopen(file.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT), H5Fclose, file);
    h5::Handle top(H5Gopen2(f, "VTKHDF", H5P_DEFAULT), H5Gclose, "VTKHDF");
    h5::Handle attr(H5Aopen(top, "Type", H5P_DEFAULT), H5Aclose, "Type");
    h5::Handle type(H5Aget_type(attr), H5Tclose, "H5Aget_type");
    std::string text(H5Tget_size(type), '\0');
    h5::check(H5Aread(attr, type, text.data()), "Type");
    return text.substr(0, text.find('\0'));
}

} // namespace

TEST(IOTest, VTKHDFHoldsTheCellsAndTheirValues) {
    // Triangles and quadrilaterals of a Gmsh file: each cell's nodes, type
    // and values round trip, in serial cell order
    const std::string dir = (scratch_dir() / "mallard_vtkhdf_cells").string();
    std::filesystem::remove_all(dir);
    const std::string file = write_temp("vtkhdf_mixed.msh", jittered_mixed_mesh(6));
    auto mesh = std::make_shared<Mesh>();
    mesh->init(parse_toml("[mesh]\ntype = \"file\"\nfilename = \"" + file + "\"\n"));
    mesh->copy_host_to_device();
    Kokkos::View<rtype **>::host_mirror_type values("values", mesh->n_cells, 3);
    for (uint32_t i = 0; i < mesh->n_cells; i++) {
        values(i, 0) = rtype(1) + rtype(i) / rtype(7);
        values(i, 1) = rtype(i);
        values(i, 2) = -rtype(2 * i);
    }
    std::vector<Data> data = {Data("RHO", Kokkos::subview(values, Kokkos::ALL(), 0)),
                              Data("U_X", Kokkos::subview(values, Kokkos::ALL(), 1)),
                              Data("U_Y", Kokkos::subview(values, Kokkos::ALL(), 2))};
    DataWriter writer;
    writer.init(parse_toml("prefix = \"" + dir + "/f\"\nformat = \"vtkhdf\"\ninterval = 5\n"
                           "variables = [\"RHO\", \"U\"]\n"),
                data, mesh);
    writer.write(10, 0.25_r);
    writer.finish();
    const std::string out = dir + "/f_000010.vtkhdf";
    EXPECT_EQ(vtkhdf_type(out), "UnstructuredGrid");
    EXPECT_EQ(h5_values<int64_t>(out, "VTKHDF/NumberOfCells"), std::vector<int64_t>{int64_t(mesh->n_cells)});
    EXPECT_EQ(h5_values<int64_t>(out, "VTKHDF/NumberOfPoints"), std::vector<int64_t>{int64_t(mesh->n_nodes)});
    EXPECT_EQ(h5_values<double>(out, "VTKHDF/FieldData/TIME"), std::vector<double>{double(0.25_r)});
    const VtkhdfGrid grid(out);
    ASSERT_EQ(grid.types.size(), mesh->n_cells);
    std::set<uint8_t> kinds(grid.types.begin(), grid.types.end());
    EXPECT_EQ(kinds, (std::set<uint8_t>{5, 9}));
    const std::vector<double> rho = h5_values<double>(out, "VTKHDF/CellData/RHO");
    const std::vector<double> u = h5_values<double>(out, "VTKHDF/CellData/U");
    for (uint32_t i = 0; i < mesh->n_cells; i++) {
        const uint32_t n = mesh->h_n_nodes_of_cell(i);
        ASSERT_EQ(grid.size(i), n);
        EXPECT_EQ(grid.types[i], vtk_cell_type(2, n));
        for (uint32_t k = 0; k < n; k++) {
            const auto x = grid.point(i, k);
            EXPECT_EQ(x[0], double(mesh->h_node_coords(mesh->h_node_of_cell(i, k), 0)));
            EXPECT_EQ(x[1], double(mesh->h_node_coords(mesh->h_node_of_cell(i, k), 1)));
            EXPECT_EQ(x[2], 0.0);
        }
        EXPECT_EQ(rho[i], double(values(i, 0)));
        EXPECT_EQ(u[3 * i], double(values(i, 1)));
        EXPECT_EQ(u[3 * i + 1], double(values(i, 2)));
        EXPECT_EQ(u[3 * i + 2], 0.0);
    }
    std::ifstream index(dir + "/f.vtkhdf.series");
    std::stringstream text;
    text << index.rdbuf();
    EXPECT_NE(text.str().find("{\"name\": \"f_000010.vtkhdf\", \"time\": 0.25}"), std::string::npos) << text.str();
    std::filesystem::remove_all(dir);
}

TEST(IOTest, VTKHDFSeriesAppendsAcrossARestart) {
    // A run restarted at t = 0.1 continues the series of one file after its
    // second snapshot, and a series of files after its second file
    const std::string dir = (scratch_dir() / "mallard_vtkhdf_resume").string();
    std::filesystem::remove_all(dir);
    auto mesh = make_mesh("cartesian_tri", 3, 2);
    const uint32_t n = mesh->n_cells;
    Kokkos::View<rtype **>::host_mirror_type values("values", n, 1);
    std::vector<Data> data = {Data("RHO", Kokkos::subview(values, Kokkos::ALL(), 0))};
    for (bool one_file : {true, false}) {
        const std::string input = "prefix = \"" + dir + "/f\"\nformat = \"vtkhdf\"\ntime_interval = 0.1\n"
                                  "variables = [\"RHO\"]\nseries = " + (one_file ? "true" : "false") + "\n";
        {
            DataWriter first;
            first.init(parse_toml(input), data, mesh);
            for (uint64_t step = 0; step < 4; step++) {
                for (uint32_t i = 0; i < n; i++) values(i, 0) = rtype(step);
                first.write(10 * step, static_cast<rtype>(0.1 * static_cast<double>(step)));
            }
        }
        DataWriter second;
        second.init(parse_toml(input), data, mesh);
        second.resume(10, 0.1_r);
        for (uint32_t i = 0; i < n; i++) values(i, 0) = rtype(9);
        second.write(25, 0.2_r);
        second.finish();
        if (!one_file) {
            std::ifstream index(dir + "/f.vtkhdf.series");
            std::stringstream text;
            text << index.rdbuf();
            for (const char * name : {"f_000000", "f_000001", "f_000002"}) {
                EXPECT_NE(text.str().find(name), std::string::npos) << name;
            }
            EXPECT_EQ(text.str().find("f_000003"), std::string::npos) << text.str();
            EXPECT_EQ(h5_values<double>(dir + "/f_000002.vtkhdf", "VTKHDF/CellData/RHO"), std::vector<double>(n, 9.0));
            continue;
        }
        const std::string file = dir + "/f.vtkhdf";
        EXPECT_EQ(h5_values<double>(file, "VTKHDF/Steps/Values"),
                  (std::vector<double>{0.0, double(rtype(0.1)), double(rtype(0.2))}));
        EXPECT_EQ(h5_values<int64_t>(file, "VTKHDF/Steps/CellDataOffsets/RHO"),
                  (std::vector<int64_t>{0, int64_t(n), int64_t(2 * n)}));
        for (const char * offsets : {"PartOffsets", "PointOffsets", "CellOffsets", "ConnectivityIdOffsets"}) {
            EXPECT_EQ(h5_values<int64_t>(file, std::string("VTKHDF/Steps/") + offsets), std::vector<int64_t>(3, 0));
        }
        std::vector<double> expected(3 * n, 0.0);
        std::fill(expected.begin() + n, expected.begin() + 2 * n, 1.0);
        std::fill(expected.begin() + 2 * n, expected.end(), 9.0);
        EXPECT_EQ(h5_values<double>(file, "VTKHDF/CellData/RHO"), expected);
        h5::Handle f(H5Fopen(file.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT), H5Fclose, file);
        h5::Handle steps(H5Gopen2(f, "VTKHDF/Steps", H5P_DEFAULT), H5Gclose, "Steps");
        h5::Handle attr(H5Aopen(steps, "NSteps", H5P_DEFAULT), H5Aclose, "NSteps");
        int64_t n_steps = 0;
        h5::check(H5Aread(attr, H5T_NATIVE_INT64, &n_steps), "NSteps");
        EXPECT_EQ(n_steps, 3);
    }
    // A series of another mesh is not appended to
    auto other = make_mesh("cartesian_tri", 4, 2);
    Kokkos::View<rtype **>::host_mirror_type other_values("values", other->n_cells, 1);
    std::vector<Data> other_data = {Data("RHO", Kokkos::subview(other_values, Kokkos::ALL(), 0))};
    DataWriter third;
    third.init(parse_toml("prefix = \"" + dir + "/f\"\nformat = \"vtkhdf\"\ntime_interval = 0.1\n"
                          "variables = [\"RHO\"]\nseries = true\n"),
               other_data, other);
    third.resume(10, 0.1_r);
    EXPECT_THROW(third.write(25, 0.2_r), std::runtime_error);
    std::filesystem::remove_all(dir);
}

TEST(IOTest, VTKHDFBoundaryZoneIsAPolyDataOfItsFaces) {
    const std::string dir = (scratch_dir() / "mallard_vtkhdf_surface").string();
    std::filesystem::remove_all(dir);
    std::ostringstream s;
    s << "[run]\nn_steps = 4\ncfl = 0.25\n"
      << "[mesh]\ntype = \"wedge\"\nNx = 12\nNy = 6\nLx = 2.0\nLy = 1.5\n"
      << "[initialize]\ntype = \"analytical\"\nrho = \"1.0 + 0.1 * x\"\nu = [\"1.5\", \"0.0\"]\np = \"1.0 + x * x\"\n"
      << "[[boundaries]]\nname = \"left\"\ntype = \"extrapolation\"\n"
      << "[[boundaries]]\nname = \"right\"\ntype = \"extrapolation\"\n"
      << "[[boundaries]]\nname = \"top\"\ntype = \"symmetry\"\n"
      << "[[boundaries]]\nname = \"bottom\"\ntype = \"symmetry\"\n"
      << "[numerics]\nriemann_solver = \"HLLC\"\n"
      << "[numerics.face_reconstruction]\ntype = \"FO\"\n"
      << "[physics]\ntype = \"euler\"\ngamma = 1.4\np_ref = 1.0\nT_ref = 1.0\nrho_ref = 1.0\n"
      << "[output]\ncheck_interval = 1000000\n"
      << "[[write_data]]\nprefix = \"" << dir << "/wall\"\nformat = \"vtkhdf\"\ngeometry = \"bottom\"\n"
      << "interval = 4\nvariables = [\"P\"]\n";
    Solver solver;
    solver.init(parse_toml(s.str()));
    solver.run();
    solver.update_primitives();
    solver.copy_device_to_host();
    const std::string file = dir + "/wall_000004.vtkhdf";
    EXPECT_EQ(vtkhdf_type(file), "PolyData");
    for (const char * empty : {"Vertices", "Polygons", "Strips"}) {
        EXPECT_EQ(h5_values<int64_t>(file, std::string("VTKHDF/") + empty + "/NumberOfCells"), std::vector<int64_t>{0});
        EXPECT_EQ(h5_values<int64_t>(file, std::string("VTKHDF/") + empty + "/Offsets"), std::vector<int64_t>{0});
    }
    EXPECT_EQ(h5_values<int64_t>(file, "VTKHDF/Lines/NumberOfCells"), std::vector<int64_t>{12});
    EXPECT_EQ(h5_values<int64_t>(file, "VTKHDF/NumberOfPoints"), std::vector<int64_t>{13});
    const std::vector<double> points = h5_values<double>(file, "VTKHDF/Points");
    const std::vector<int64_t> connectivity = h5_values<int64_t>(file, "VTKHDF/Lines/Connectivity");
    const std::vector<double> p = h5_values<double>(file, "VTKHDF/CellData/P");
    ASSERT_EQ(connectivity.size(), 24u);
    ASSERT_EQ(p.size(), 12u);
    // Each segment carries the pressure of the cell above it
    const auto & mesh = *solver.get_mesh();
    FaceZone * zone = solver.get_mesh()->get_face_zone("bottom");
    for (size_t k = 0; k < 12; k++) {
        const double x = 0.5 * (points[3 * size_t(connectivity[2 * k])] + points[3 * size_t(connectivity[2 * k + 1])]);
        int matches = 0;
        for (uint32_t i = 0; i < zone->n_faces(); i++) {
            const uint32_t f = zone->h_faces(i);
            const double xf = 0.5 * (double(mesh.h_node_coords(mesh.h_node_of_face(f, 0), 0)) +
                                     double(mesh.h_node_coords(mesh.h_node_of_face(f, 1), 0)));
            if (std::abs(xf - x) > 1e-6) continue;
            matches++;
            EXPECT_EQ(p[k], double(solver.h_primitives(mesh.h_cells_of_face(f, 0), 2))) << k;
        }
        EXPECT_EQ(matches, 1) << k;
    }
    std::filesystem::remove_all(dir);
}

TEST(IOTest, VTKHDFWritesCurvedCellsAsQuadraticLagrangeCells) {
    // Annulus cells at the walls (circles) are quadratic: their edge nodes on
    // the walls lie on the circles; the ring in between stays linear
    const std::string dir = (scratch_dir() / "mallard_vtkhdf_curved").string();
    std::filesystem::remove_all(dir);
    constexpr uint32_t N_T = 24;
    const double tol = 64 * std::numeric_limits<rtype>::epsilon();
    for (bool triangles : {false, true}) {
        const std::string file = write_temp("vtkhdf_annulus.msh", annulus_gmsh(3, N_T, 1, triangles));
        Solver solver;
        solver.init(parse_toml(
            "[run]\nn_steps = 1\ncfl = 0.3\n[mesh]\ntype = \"file\"\nfilename = \"" + file + "\"\n"
            "[[mesh.curved]]\nzone = \"inner\"\nshape = \"circle\"\ncenter = [0.0, 0.0]\nradius = 1.0\n"
            "[[mesh.curved]]\nzone = \"outer\"\nshape = \"circle\"\ncenter = [0.0, 0.0]\nradius = 1.384\n"
            "[initialize]\ntype = \"analytical\"\nrho = \"1.0\"\nu = [\"0.0\", \"0.0\"]\np = \"1.0\"\n"
            "[[boundaries]]\nname = \"inner\"\ntype = \"wall_adiabatic\"\n"
            "[[boundaries]]\nname = \"outer\"\ntype = \"symmetry\"\n"
            "[numerics]\nriemann_solver = \"HLLC\"\n[numerics.face_reconstruction]\ntype = \"MUSCL\"\n"
            "[physics]\ntype = \"euler\"\ngamma = 1.4\np_ref = 1.0\nT_ref = 1.0\nrho_ref = 1.0\n"
            "[output]\ncheck_interval = 1000000\n"
            "[[write_data]]\nprefix = \"" + dir + "/f\"\nformat = \"vtkhdf\"\ninterval = 1\nvariables = [\"RHO\"]\n"));
        solver.run();
        const VtkhdfGrid grid(dir + "/f_000000.vtkhdf");
        const uint8_t quadratic = triangles ? 69 : 70, linear = triangles ? 5 : 9;
        const std::vector<std::array<size_t, 2>> edges =
            triangles ? std::vector<std::array<size_t, 2>>{{0, 1}, {1, 2}, {2, 0}}
                      : std::vector<std::array<size_t, 2>>{{0, 1}, {1, 2}, {3, 2}, {0, 3}};
        auto radius = [](const std::array<double, 3> & x) { return std::hypot(x[0], x[1]); };
        size_t n_quadratic = 0, n_wall_edges = 0;
        for (size_t c = 0; c < grid.types.size(); c++) {
            if (grid.types[c] == linear) continue;
            ASSERT_EQ(grid.types[c], quadratic) << c;
            ASSERT_EQ(grid.size(c), triangles ? 6u : 9u);
            n_quadratic++;
            for (size_t e = 0; e < edges.size(); e++) {
                const auto a = grid.point(c, edges[e][0]), b = grid.point(c, edges[e][1]);
                for (double wall : {1.0, 1.384}) {
                    if (std::abs(radius(a) - wall) > tol || std::abs(radius(b) - wall) > tol) continue;
                    n_wall_edges++;
                    EXPECT_NEAR(radius(grid.point(c, edges.size() == 3 ? 3 + e : 4 + e)), wall, tol) << c;
                    const std::array<double, 3> chord = {0.5 * (a[0] + b[0]), 0.5 * (a[1] + b[1]), 0.0};
                    EXPECT_GT(wall - radius(chord), 1e-3) << c;
                }
            }
        }
        EXPECT_EQ(n_quadratic, 2u * N_T);
        EXPECT_EQ(n_wall_edges, 2u * N_T);
    }
    std::filesystem::remove_all(dir);
}

#endif

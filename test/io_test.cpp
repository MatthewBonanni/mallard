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

#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <map>
#include <sstream>
#include <string>
#include <vector>

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
    solver.set_distributed(false);
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
    solver.set_distributed(false);
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
    solver.set_distributed(false);
    solver.init(parse_toml(input));
    solver.run();
    EXPECT_NEAR(solver.get_time(), 0.075, roundoff(1e-12));
}

TEST(IOTest, LocalCFLIsWrittenWithFixedTimeStepAndAtStepZero) {
    // Uniform flow u = 0.1, a = sqrt(1.4) on 0.25 x 0.5 cells: the CFL = 1 time
    // step is 2 V / sum_f (|u_n| + a) A_f = 0.25 / (0.1 + 1.5 a)
    const std::string dir = (scratch_dir() / "mallard_cfl_output").string();
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
    solver.set_distributed(false);
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
          << "variables = [\"RHO\", \"U\", \"P\", \"MEAN_P\"]\n" << snapshot_keys;
    }
    return s.str();
}

void run_case(const std::string & input) {
    Solver solver;
    solver.set_distributed(false);
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
    std::vector<std::pair<std::string, std::string>> variants = {{"", ""}};
    if (compression::have_zlib()) {
        variants.emplace_back("compression = \"zlib\"\n", "compression = \"zlib\"\n");
        variants.emplace_back("compression = \"zlib\"\nrelative_error = 1e-4\n",
                              "compression = \"zlib\"\nchunk_cells = 100\n");
    }
    int variant = 0;
    for (const auto & [snapshot_keys, restart_keys] : variants) {
        const std::string dir = (scratch_dir() / ("mallard_async_output_" + std::to_string(variant++))).string();
        run_case(output_case(dir + "/sync", "async = false\n", snapshot_keys, restart_keys));
        run_case(output_case(dir + "/async", "io_threads = 3\n", snapshot_keys, restart_keys));
        const auto sync = files_under(dir + "/sync"), async = files_under(dir + "/async");
        EXPECT_EQ(sync.size(), have_hdf5() ? 23u : 13u) << snapshot_keys;
        ASSERT_EQ(async.size(), sync.size()) << snapshot_keys;
        for (const auto & [name, bytes] : sync) {
            ASSERT_TRUE(async.count(name)) << name;
#ifdef Mallard_HAS_HDF5
            if (name.ends_with(".h5")) {
                EXPECT_EQ(h5_datasets(dir + "/async/" + name), h5_datasets(dir + "/sync/" + name)) << name;
                continue;
            }
#endif
            EXPECT_TRUE(without_provenance(async.at(name)) == without_provenance(bytes)) << name << " " << snapshot_keys;
        }
    }
}

TEST(IOTest, AWriterGoingAwayFinishesItsFileFirst) {
    // A run that stops on an error destroys its writers without finish():
    // the file being written in the background is still written whole
    const std::string dir = (scratch_dir() / "mallard_async_unwind").string();
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
    comm::barrier();  // Rank 0 writes the files
    const std::string sync = file_contents(dir + "/s_000000.vtu");
    EXPECT_GT(sync.size(), 120000u * sizeof(rtype));
    EXPECT_TRUE(file_contents(dir + "/a_000000.vtu") == sync);
}

TEST(IOTest, CompressedFilesHoldTheSameValuesOrValuesWithinTheRelativeError) {
    if (!compression::have_zlib()) GTEST_SKIP() << "built without zlib";
    const std::string dir = (scratch_dir() / "mallard_compressed_output").string();
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
#endif
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

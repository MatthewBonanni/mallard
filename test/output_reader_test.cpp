/**
 * @file output_reader_test.cpp
 * @brief The output reader reads back what every writer wrote: the solver's
 *        values, times and units, the same from every format.
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 */

#include <gtest/gtest.h>
#include <Kokkos_Core.hpp>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <map>
#include <sstream>
#include <string>
#include <vector>

#include "comm.h"
#include "compression.h"
#include "data_paths.h"
#include "mesh_block.h"
#include "output_reader.h"
#include "scratch.h"
#include "solver.h"
#include "test_fixtures.h"

namespace {

namespace fs = std::filesystem;

std::string coords(const char * x, const char * y, const char * z) {
    return std::string("[") + x + ", " + y + (N_DIM == 3 ? std::string(", ") + z : "") + "]";
}

const std::string REGION = N_DIM == 2 ? "{type = \"rectangle\", center = [0.45, 0.45], size = [0.5, 0.3]}"
                                      : "{type = \"box\", center = [0.45, 0.45, 0.5], size = [0.5, 0.3, 0.6]}";

/** @brief Four steps on a periodic box, every writer every two steps (each probe format every step). */
std::string every_format(const std::string & dir, bool hdf5 = have_hdf5()) {
    std::ostringstream s;
    s << "[run]\nn_steps = 4\ndt = 2.0e-3\n[mesh]\ntype = \"cartesian\"\nNx = 10\nNy = 10\nLx = 1.0\nLy = 1.0\n";
    if (N_DIM == 3) s << "Nz = 5\nLz = 1.0\n[[periodic]]\nzones = [\"back\", \"front\"]\n";
    s << "[[periodic]]\nzones = [\"left\", \"right\"]\n[[periodic]]\nzones = [\"bottom\", \"top\"]\n"
      << "[initialize]\ntype = \"analytical\"\nrho = \"1.0 + 0.3 * x + 0.2 * y * y\"\n"
      << "u = " << (N_DIM == 2 ? "[\"0.3\", \"-0.1 * x\"]" : "[\"0.3\", \"-0.1 * x\", \"0.2 * y\"]") << "\n"
      << "p = \"1.0 + 0.1 * x * y\"\n[numerics]\nriemann_solver = \"HLLC\"\ntime_integrator = \"SSPRK3\"\n"
      << "[numerics.face_reconstruction]\ntype = \"MUSCL\"\n"
      << "[physics]\ntype = \"euler\"\ngamma = 1.4\np_ref = 1.0\nT_ref = 1.0\nrho_ref = 1.0\n"
      << "[output]\ncheck_interval = 1000000\nrun_log = \"" << dir << "/mallard.jsonl\"\n";
    auto writer = [&](const std::string & prefix, const std::string & keys) {
        s << "[[write_data]]\nprefix = \"" << dir << "/" << prefix << "\"\ninterval = 2\n" << keys;
    };
    const std::string fields = "variables = [\"RHO\", \"U\", \"P\"]\n";
    writer("restart", "format = \"restart\"\n");
    writer("flow", "format = \"vtu\"\n" + fields);
    writer("near", "format = \"vtu\"\nregion = " + REGION + "\nvariables = [\"RHO\"]\n");
    if (compression::have_zlib()) {
        writer("zlib/restart", "format = \"restart\"\ncompression = \"zlib\"\nchunk_cells = 64\n");
        writer("zlib/flow", "format = \"vtu\"\ncompression = \"zlib\"\n" + fields);
    }
    std::vector<std::string> probe_formats = {"csv", "vtp"};
    if (hdf5) {
        writer("flow", "format = \"vtkhdf\"\n" + fields);
        writer("flow", "format = \"xdmf\"\n" + fields);
        writer("series", "series = true\n" + fields);
        writer("region", "region = " + REGION + "\nvariables = [\"RHO\"]\n");
        if (compression::have_zlib()) writer("zlib/flow", "compression = \"zlib\"\n" + fields);
        probe_formats.insert(probe_formats.end(), {"hdf5", "vtkhdf"});
    }
    for (const std::string & format : probe_formats) {
        s << "[[probes]]\nname = \"" << format << "\"\nformat = \"" << format << "\"\nstart = "
          << coords("0.13", "0.27", "0.41") << "\nend = " << coords("0.88", "0.62", "0.55")
          << "\nn_points = 6\nvariables = [\"RHO\", \"U\"]\n";
        if (format == "csv") s << "file = \"" << dir << "/probe_csv.csv\"\n";
        if (format == "vtp") s << "file = \"" << dir << "/probe_vtp.pvd\"\n";
        if (format == "hdf5") s << "file = \"" << dir << "/probe_hdf5.h5\"\n";
        if (format == "vtkhdf") s << "file = \"" << dir << "/probe_vtkhdf.vtkhdf\"\n";
    }
    return s.str();
}

/** @brief The case of every_format, run; the solver keeps the final state. */
struct CaseRun {
    std::string dir;
    Solver solver;
    std::vector<double> rho;          ///< final density, in local cell order
    std::vector<uint64_t> global;     ///< global cell of each local cell

    CaseRun() : dir((scratch_dir() / "output_reader_every_format").string()) {
        fs::remove_all(dir);
        solver.set_distributed(false);
        solver.init(parse_toml(every_format(dir)));
        solver.run();
        const auto state = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), solver.conservatives);
        const auto mesh = solver.get_mesh();
        for (uint32_t c = 0; c < mesh->n_owned(); c++) {
            rho.push_back(static_cast<double>(state(c, 0)));
            global.push_back(static_cast<uint64_t>(mesh->h_global_cell(c)));
        }
    }

    /** @brief Values of a file's cells in the global order. */
    std::vector<double> in_global_order(const output::Contents & file, const std::vector<double> & values,
                                        uint32_t components = 1) const {
        if (file.global_order) return values;
        std::vector<double> out(values.size());
        for (size_t c = 0; c < global.size(); c++) {
            for (uint32_t i = 0; i < components; i++) out[global[c] * components + i] = values[c * components + i];
        }
        return out;
    }
};

#define SKIP_UNDER_MPI() \
    if (comm::size() > 1) GTEST_SKIP() << "a run on one rank"

/** @brief The snapshot files of the run, by what wrote them. */
std::vector<std::string> snapshot_files(const std::string & dir, const std::string & step) {
    std::vector<std::string> files = {dir + "/flow_" + step + ".vtu"};
    if (compression::have_zlib()) files.push_back(dir + "/zlib/flow_" + step + ".vtu");
    if (have_hdf5()) {
        files.push_back(dir + "/flow_" + step + ".vtkhdf");
        files.push_back(dir + "/flow_" + step + ".h5");
        if (compression::have_zlib()) files.push_back(dir + "/zlib/flow_" + step + ".vtkhdf");
    }
    return files;
}

} // namespace

TEST(OutputReaderTest, RestartFilesHoldTheSolversState) {
    SKIP_UNDER_MPI();
    const CaseRun run;
    std::vector<std::string> files = {run.dir + "/restart_000004.restart"};
    if (compression::have_zlib()) files.push_back(run.dir + "/zlib/restart_000004.restart");
    for (const std::string & name : files) {
        const output::Contents file = output::open(name);
        EXPECT_EQ(file.kind, output::Kind::restart) << name;
        EXPECT_EQ(file.dim, N_DIM) << name;
        EXPECT_EQ(file.real_size, sizeof(rtype)) << name;
        EXPECT_EQ(file.compressed, name.find("zlib") != std::string::npos) << name;
        EXPECT_EQ(file.mesh.n_cells, run.rho.size()) << name;
        ASSERT_EQ(file.snapshots.size(), 1u) << name;
        EXPECT_EQ(file.snapshots[0].step, 4) << name;
        EXPECT_EQ(file.restart.n_cells, run.rho.size()) << name;
        EXPECT_FALSE(file.provenance.empty()) << name;
        ASSERT_TRUE(file.field("RHOE")) << name;
        EXPECT_EQ(file.field("RHO")->units, "kg m-3") << name;
        EXPECT_EQ(output::read(file, "RHO", 0), run.in_global_order(file, run.rho)) << name;
    }
}

TEST(OutputReaderTest, EverySnapshotFormatHoldsTheSameValuesAndUnits) {
    SKIP_UNDER_MPI();
    const CaseRun run;
    const std::vector<std::string> files = snapshot_files(run.dir, "000004");
    std::vector<double> velocity;
    for (const std::string & name : files) {
        const output::Contents file = output::open(name);
        EXPECT_EQ(file.kind, output::Kind::snapshot) << name;
        EXPECT_EQ(file.mesh.n_cells, run.rho.size()) << name;
        ASSERT_EQ(file.snapshots.size(), 1u) << name;
        EXPECT_DOUBLE_EQ(file.snapshots[0].t, 8.0e-3) << name;
        std::vector<std::string> names;
        for (const output::Field & f : file.fields) names.push_back(f.name);
        std::sort(names.begin(), names.end());
        EXPECT_EQ(names, (std::vector<std::string>{"P", "RHO", "U"})) << name;
        ASSERT_TRUE(file.field("U")) << name;
        EXPECT_EQ(file.field("U")->components, 3u) << name;
        EXPECT_EQ(file.field("U")->units, "m s-1") << name;
        EXPECT_EQ(file.field("U")->long_name, "velocity") << name;
        EXPECT_EQ(file.field("RHO")->units, "kg m-3") << name;
        EXPECT_EQ(output::read(file, "RHO", 0), run.in_global_order(file, run.rho)) << name;
        const std::vector<double> u = run.in_global_order(file, output::read(file, "U", 0), 3);
        if (velocity.empty()) velocity = u;
        EXPECT_EQ(u, velocity) << name;
    }
    ASSERT_EQ(velocity.size(), 3 * run.rho.size());
    if (N_DIM == 2) {
        for (size_t c = 0; c < run.rho.size(); c++) EXPECT_EQ(velocity[3 * c + 2], 0.0);
    }
}

TEST(OutputReaderTest, SeriesIndexesListEverySnapshotOfTheirFiles) {
    SKIP_UNDER_MPI();
    const CaseRun run;
    std::vector<std::string> series = {run.dir + "/flow.pvd"};
    if (have_hdf5()) series.insert(series.end(), {run.dir + "/flow.vtkhdf.series", run.dir + "/flow.xmf",
                                                  run.dir + "/series.vtkhdf"});
    const std::vector<std::string> steps = {"000000", "000002", "000004"};
    for (const std::string & name : series) {
        const output::Contents file = output::open(name);
        ASSERT_EQ(file.snapshots.size(), 3u) << name;
        for (size_t i = 0; i < steps.size(); i++) {
            EXPECT_NEAR(file.snapshots[i].t, 4.0e-3 * double(i), 1e-15) << name << " " << i;
            const output::Contents restart = output::open(run.dir + "/restart_" + steps[i] + ".restart");
            EXPECT_EQ(run.in_global_order(file, output::read(file, "RHO", i)), output::read(restart, "RHO", 0))
                << name << " " << i;
        }
    }
}

TEST(OutputReaderTest, PiecesOfARunOnSeveralRanksHoldEveryCellOnce) {
    if (comm::size() == 1) GTEST_SKIP() << "a run on several ranks";
    const std::string dir = (scratch_dir() / "output_reader_pieces").string();
    Solver solver;
    solver.init(parse_toml(every_format(dir, false)));
    solver.run();
    comm::barrier();
    if (!comm::is_root()) return;
    const output::Contents restart = output::open(dir + "/restart_000004.restart");
    std::vector<double> expected = output::read(restart, "RHO", 0);
    std::sort(expected.begin(), expected.end());
    for (const std::string & name : {dir + "/flow_000004.pvtu", dir + "/flow.pvd"}) {
        const output::Contents file = output::open(name);
        EXPECT_EQ(file.mesh.n_cells, expected.size()) << name;
        std::vector<double> rho = output::read(file, "RHO", file.snapshots.size() - 1);
        std::sort(rho.begin(), rho.end());
        EXPECT_EQ(rho, expected) << name;
    }
}

TEST(OutputReaderTest, RegionSnapshotsHoldTheCellsInTheRegion) {
    SKIP_UNDER_MPI();
    const CaseRun run;
    const auto mesh = run.solver.get_mesh();
    const std::array<double, 3> center = {0.45, 0.45, 0.5}, size = {0.5, 0.3, 0.6};
    std::vector<double> inside;
    for (uint32_t c = 0; c < mesh->n_owned(); c++) {
        bool in = true;
        for (size_t i = 0; i < N_DIM; i++) {
            in = in && std::abs(double(mesh->h_cell_coords(c, i)) - center[i]) < size[i] / 2;
        }
        if (in) inside.push_back(run.rho[c]);
    }
    std::sort(inside.begin(), inside.end());
    ASSERT_GT(inside.size(), 0u);
    std::vector<std::string> files = {run.dir + "/near_000004.vtu"};
    if (have_hdf5()) files.push_back(run.dir + "/region_000004.vtkhdf");
    for (const std::string & name : files) {
        const output::Contents file = output::open(name);
        EXPECT_EQ(file.mesh.n_cells, inside.size()) << name;
        EXPECT_FALSE(file.global_order) << name;
        std::vector<double> rho = output::read(file, "RHO", 0);
        std::sort(rho.begin(), rho.end());
        EXPECT_EQ(rho, inside) << name;
    }
}

TEST(OutputReaderTest, EveryProbeFormatHoldsTheSameSamples) {
    SKIP_UNDER_MPI();
    const CaseRun run;
    std::vector<std::string> files = {run.dir + "/probe_csv.csv", run.dir + "/probe_vtp.pvd"};
    if (have_hdf5()) files.insert(files.end(), {run.dir + "/probe_hdf5.h5", run.dir + "/probe_vtkhdf.vtkhdf"});
    const output::Contents reference = output::open(files[0]);
    ASSERT_EQ(reference.snapshots.size(), 5u);
    for (const std::string & name : files) {
        const output::Contents file = output::open(name);
        EXPECT_EQ(file.kind, name.ends_with(".csv") ? output::Kind::table : output::Kind::probe) << name;
        ASSERT_EQ(file.snapshots.size(), reference.snapshots.size()) << name;
        ASSERT_TRUE(file.field("RHO")) << name;
        EXPECT_EQ(file.field("RHO")->units, "kg m-3") << name;
        for (size_t i = 0; i < file.snapshots.size(); i++) {
            EXPECT_NEAR(file.snapshots[i].t, 2.0e-3 * double(i), 1e-12) << name << " " << i;
            const std::vector<double> rho = output::read(file, "RHO", i), expected = output::read(reference, "RHO", i);
            ASSERT_EQ(rho.size(), 6u) << name << " " << i;
            for (size_t p = 0; p < rho.size(); p++) {
                EXPECT_NEAR(rho[p], expected[p], 1e-9 * std::abs(expected[p])) << name << " " << i << " " << p;
            }
        }
    }
}

TEST(OutputReaderTest, RunDirectoryListsEveryOutputOnce) {
    SKIP_UNDER_MPI();
    const CaseRun run;
    std::vector<std::string> listed;
    for (const std::string & f : output::run_files(run.dir)) {
        listed.push_back(fs::relative(f, run.dir).string());
    }
    for (const char * name : {"flow.pvd", "near.pvd", "restart_000000.restart", "restart_000004.restart",
                              "probe_csv.csv", "probe_vtp.pvd"}) {
        EXPECT_EQ(std::count(listed.begin(), listed.end(), name), 1) << name;
    }
    for (const char * name : {"flow_000000.vtu", "probe_vtp_000000.vtp", "flow_mesh.h5", "mallard.jsonl"}) {
        EXPECT_EQ(std::count(listed.begin(), listed.end(), name), 0) << name;
    }
}

TEST(OutputReaderTest, RestartFilesOfPastReleasesAreRead) {
    // Each release's files of the same case hold the same state
    const std::string formats = source_dir() + "/test/data/formats/";
    const std::map<std::string, int64_t> versions = {{"gas", 2}, {"statistics", 3}, {"mixture", 2}};
    for (const auto & [name, version] : versions) {
        std::vector<double> first;
        for (const char * release : {"v0.6.0", "v0.7.0", "v0.8.0"}) {
            const std::string path = formats + release + "/" + name + "_000002.restart";
            const output::Contents file = output::open(path);
            EXPECT_EQ(file.version, version) << path;
            EXPECT_EQ(file.dim, 2) << path;
            EXPECT_EQ(file.snapshots.at(0).step, 2) << path;
            EXPECT_TRUE(file.provenance.empty()) << path;
            const std::vector<double> rho = output::read(file, "RHO", 0);
            EXPECT_EQ(rho.size(), file.mesh.n_cells) << path;
            if (first.empty()) first = rho;
            for (size_t c = 0; c < rho.size(); c++) EXPECT_NEAR(rho[c], first[c], 1e-12) << path << " " << c;
        }
    }
    const output::Contents mixture = output::open(formats + "v0.8.0/mixture_000002.restart");
    const std::vector<std::string> species = mixture.species();
    EXPECT_NE(std::find(species.begin(), species.end(), "H2"), species.end());
    EXPECT_NE(std::find(species.begin(), species.end(), "N2"), species.end());
}

TEST(OutputReaderTest, MeshFilesAreReadWithThisBuildsDimension) {
    const std::string mesh = source_dir() + (N_DIM == 2 ? "/test/data/meshes/cyl2d.msh" : "/test/data/meshes/cyl3d.msh");
    ASSERT_TRUE(output::is_mesh_file(mesh));
    const MeshBlock block = output::read_mesh(mesh);
    const output::Contents file = output::mesh_contents(mesh, block);
    EXPECT_EQ(file.kind, output::Kind::mesh);
    EXPECT_EQ(file.dim, N_DIM);
    EXPECT_EQ(file.mesh.n_cells, block.n_cells());
    uint64_t typed = 0;
    for (const auto & [type, n] : file.mesh.cell_types) typed += n;
    EXPECT_EQ(typed, file.mesh.n_cells);
    uint64_t zoned = 0;
    for (const auto & [zone, n] : file.mesh.boundary_zones) zoned += n;
    EXPECT_GT(zoned, 0u);
    EXPECT_FALSE(output::is_mesh_file(source_dir() + "/test/data/formats/v0.8.0/gas_000002.restart"));
    if (have_hdf5()) {
        EXPECT_TRUE(output::is_mesh_file(source_dir() + "/test/data/formats/v0.8.0/mesh.h5"));
    }
}

TEST(OutputReaderTest, FilesItCannotReadAreNamedInTheError) {
    const std::string missing = (scratch_dir() / "output_reader_missing.vtu").string();
    try {
        output::open(missing);
        FAIL() << "opened a missing file";
    } catch (const std::runtime_error & e) {
        EXPECT_NE(std::string(e.what()).find(missing), std::string::npos) << e.what();
    }
}

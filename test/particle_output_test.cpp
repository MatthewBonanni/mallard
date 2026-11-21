/**
 * @file particle_output_test.cpp
 * @author Matthew Bonanni (mbonanni001@gmail.com)
 * @brief Tests of particle snapshots ([[write_data]] particle_fields): the
 *        particles of each snapshot with their values, region filters,
 *        restarts continuing the series, and the run log's records.
 * @version 0.1
 * @date 2026-11-21
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 *
 */

#include <gtest/gtest.h>

#include <cmath>
#include <filesystem>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

#include "comm.h"
#include "data_writer.h"
#include "json.h"
#include "mesh_block.h"
#include "mpi_compare.h"
#include "output_reader.h"
#include "particles.h"
#include "scratch.h"
#include "solver.h"
#include "test_fixtures.h"

#ifdef Mallard_HAS_HDF5
#include "hdf5_util.h"

namespace {

std::string numbers(double x, double y, double z) {
    std::ostringstream s;
    s.precision(17);
    s << "[" << x << ", " << y;
    if (N_DIM == 3) s << ", " << z;
    s << "]";
    return s.str();
}

/** @brief A periodic unit box of 12 cells per side (z: 6, length 0.5) in uniform flow u, n x n tracers in [lo, hi]. */
std::string box(const double u[3], uint32_t n_steps, uint32_t n, double lo, double hi, const std::string & extra,
                const std::string & init = "") {
    std::ostringstream s;
    s.precision(17);
    s << "[run]\nn_steps = " << n_steps << "\ndt = 0.01\n[mesh]\ntype = \"cartesian\"\nNx = 12\nNy = 12\nLx = 1.0\nLy = 1.0\n";
    if (N_DIM == 3) s << "Nz = 6\nLz = 0.5\n";
    s << periodic_sides(N_DIM == 3 ? "xyz" : "xy");
    if (init.empty()) {
        s << "[initialize]\ntype = \"analytical\"\nrho = \"1.0\"\nu = [\"" << u[0] << "\", \"" << u[1]
          << (N_DIM == 3 ? "\", \"" + std::to_string(u[2]) : std::string()) << "\"]\np = \"1.0\"\n";
    } else {
        s << "[initialize]\n" << init;
    }
    s << "[numerics]\nriemann_solver = \"HLLC\"\ntime_integrator = \"RK4\"\n[numerics.face_reconstruction]\n"
      << "type = \"TENO\"\norder = 4\n[physics]\ntype = \"euler\"\ngamma = 1.4\np_ref = 1.0\nT_ref = 1.0\n"
      << "rho_ref = 1.0\n[output]\ncheck_interval = 1000000\n[particles]\ntype = \"tracer\"\n[particles.lattice]\nlo = "
      << numbers(lo, lo, 0.25) << "\nhi = " << numbers(hi, hi, 0.25) << "\nn = [" << n << ", "
      << n << (N_DIM == 3 ? ", 1]" : "]") << "\n"
      << extra;
    return s.str();
}

/** @brief The seeds of box(), in id order. */
std::vector<std::array<double, 3>> seeds(uint32_t n, double lo, double hi) {
    std::vector<std::array<double, 3>> out;
    for (uint32_t j = 0; j < n; j++) {
        for (uint32_t i = 0; i < n; i++) {
            out.push_back({lo + (hi - lo) * i / (n - 1), lo + (hi - lo) * j / (n - 1), N_DIM == 3 ? 0.25 : 0.0});
        }
    }
    return out;
}

double wrap(double x, double length) { return x - length * std::floor(x / length); }

bool can_write_vtkhdf() { return have_hdf5() && (comm::size() == 1 || have_parallel_hdf5()); }

/** @brief One snapshot of a particle file: its time and step, and per particle id its position then fields. */
struct Snapshot {
    double t = 0.0;
    int64_t step = -1;
    std::map<uint64_t, std::vector<double>> particles;
};

/** @brief Every snapshot of a particle VTKHDF file, with the named point fields (vectors as 3 columns). */
std::vector<Snapshot> read_particles(const std::string & filename, const std::vector<std::string> & fields,
                                     const std::string & group = "VTKHDF") {
    const std::lock_guard<std::mutex> lock(h5::mutex());
    h5::Handle file(H5Fopen(filename.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT), H5Fclose, "opening " + filename);
    h5::Handle top(H5Gopen2(file, group.c_str(), H5P_DEFAULT), H5Gclose, group);
    EXPECT_EQ(h5::read_string_attribute(top, "Type"), "PolyData");
    h5::Handle steps(H5Gopen2(top, "Steps", H5P_DEFAULT), H5Gclose, "Steps");
    h5::Handle data(H5Gopen2(top, "PointData", H5P_DEFAULT), H5Gclose, "PointData");
    h5::Handle field_data(H5Gopen2(top, "FieldData", H5P_DEFAULT), H5Gclose, "FieldData");
    const uint64_t n_steps = h5::n_rows(steps, "Values");
    const auto times = h5::read_rows<double>(steps, "Values", 0, n_steps, 1, H5P_DEFAULT);
    const auto step_numbers = h5::read_rows<int64_t>(field_data, "STEP", 0, n_steps, 1, H5P_DEFAULT);
    const auto counts = h5::read_rows<int64_t>(top, "NumberOfPoints", 0, n_steps, 1, H5P_DEFAULT);
    const auto offsets = h5::read_rows<int64_t>(steps, "PointOffsets", 0, n_steps, 1, H5P_DEFAULT);
    const uint64_t n_rows = h5::n_rows(top, "Points");
    const auto points = h5::read_rows<double>(top, "Points", 0, n_rows, 3, H5P_DEFAULT);
    const auto ids = h5::read_rows<int64_t>(data, "ID", 0, n_rows, 1, H5P_DEFAULT);
    std::vector<std::vector<double>> columns;
    std::vector<uint32_t> widths;
    for (const auto & f : fields) {
        h5::Handle dset(H5Dopen2(data, f.c_str(), H5P_DEFAULT), H5Dclose, f);
        h5::Handle space(H5Dget_space(dset), H5Sclose, "space");
        const uint32_t cols = H5Sget_simple_extent_ndims(space) == 2 ? 3 : 1;
        widths.push_back(cols);
        columns.push_back(h5::read_rows<double>(data, f.c_str(), 0, n_rows, cols, H5P_DEFAULT));
    }
    std::vector<Snapshot> out(n_steps);
    for (uint64_t k = 0; k < n_steps; k++) {
        out[k].t = times[k];
        out[k].step = step_numbers[k];
        for (int64_t p = offsets[k]; p < offsets[k] + counts[k]; p++) {
            const size_t r = static_cast<size_t>(p);
            std::vector<double> v = {points[3 * r], points[3 * r + 1], points[3 * r + 2]};
            for (size_t f = 0; f < fields.size(); f++) {
                for (uint32_t c = 0; c < widths[f]; c++) v.push_back(columns[f][widths[f] * r + c]);
            }
            EXPECT_TRUE(out[k].particles.emplace(static_cast<uint64_t>(ids[r]), v).second)
                << "particle " << ids[r] << " twice in snapshot " << k;
        }
    }
    return out;
}

/** @brief Every dataset of an HDF5 file but its provenance and snapshot blocks, by path. */
std::map<std::string, std::vector<double>> datasets(const std::string & filename) {
    std::map<std::string, std::vector<double>> out;
    const std::lock_guard<std::mutex> lock(h5::mutex());
    h5::Handle file(H5Fopen(filename.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT), H5Fclose, "opening " + filename);
    struct Visit {
        hid_t file;
        std::map<std::string, std::vector<double>> * out;
    } visit{file, &out};
    H5Ovisit3(file, H5_INDEX_NAME, H5_ITER_INC, [](hid_t, const char * name, const H5O_info2_t * info, void * op) -> herr_t {
        auto & v = *static_cast<Visit *>(op);
        if (info->type != H5O_TYPE_DATASET || std::string(name).starts_with("mallard_")) return 0;
        h5::Handle dset(H5Dopen2(v.file, name, H5P_DEFAULT), H5Dclose, name);
        h5::Handle space(H5Dget_space(dset), H5Sclose, "space");
        const hssize_t n = H5Sget_simple_extent_npoints(space);
        std::vector<double> values(static_cast<size_t>(n));
        h5::check(H5Dread(dset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, values.data()), name);
        (*v.out)[name] = values;
        return 0;
    }, &visit, H5O_INFO_BASIC);
    return out;
}

/** @brief The output records of a run log. */
std::vector<json::Value> output_records(const std::string & log) {
    std::vector<json::Value> out;
    std::ifstream in(log);
    std::string line;
    while (std::getline(in, line)) {
        const json::Value r = json::parse(line);
        if (r["type"].string == "output") out.push_back(r);
    }
    return out;
}

const double U[3] = {0.3, -0.2, 0.1};

} // namespace

TEST(ParticleSnapshotsTest, EverySnapshotHoldsEveryParticleAtItsStepWithItsFields) {
    // Steps 0 (start), 10 and 20 (interval) and 25 (end); with and without step graphs, which stage the snapshots on
    // the device. The tracers cross the periodic sides, and on several ranks the partitions.
    if (!can_write_vtkhdf()) GTEST_SKIP() << "needs HDF5, parallel on several ranks";
    const std::string prefix = (scratch_dir() / "mallard_particle_snapshots" / "cloud").string();
    const std::string writer = "[[write_data]]\nprefix = \"" + prefix +
                               "\"\nparticle_fields = [\"U\", \"RHO\", \"V_X\"]\ninterval = 10\n";
    const std::string input = box(U, 25, 4, 0.07, 0.93, writer);
    const auto x0 = seeds(4, 0.07, 0.93);
    auto check = [&](const Solver & solver) {
        comm::barrier();
        const auto snapshots = read_particles(prefix + "_particles.vtkhdf", {"U", "RHO", "V_X"});
        ASSERT_EQ(snapshots.size(), 4u);
        const int64_t steps[4] = {0, 10, 20, 25};
        for (size_t k = 0; k < snapshots.size(); k++) {
            const Snapshot & s = snapshots[k];
            EXPECT_EQ(s.step, steps[k]);
            EXPECT_NEAR(s.t, 0.01 * static_cast<double>(steps[k]), 1e-14);
            ASSERT_EQ(s.particles.size(), x0.size()) << "snapshot " << k;
            for (const auto & [id, v] : s.particles) {
                ASSERT_LT(id, x0.size());
                for (size_t d = 0; d < N_DIM; d++) {
                    EXPECT_NEAR(v[d], wrap(x0[id][d] + U[d] * s.t, d == 2 ? 0.5 : 1.0), roundoff(1e-12))
                        << "particle " << id << " at step " << s.step;
                    EXPECT_NEAR(v[3 + d], U[d], roundoff(1e-12)) << "U of particle " << id;
                }
                EXPECT_NEAR(v[6], 1.0, roundoff(1e-12)) << "RHO of particle " << id;
                EXPECT_EQ(v[7], v[3]) << "V_X of tracer " << id << " is the gas velocity";
            }
        }
        // The last snapshot is the solver's final state, bit for bit
        for (const auto & r : solver.get_particles()->gather()) {
            for (size_t d = 0; d < N_DIM; d++) EXPECT_EQ(snapshots.back().particles.at(r.id)[d], r.x[d]);
        }
    };
    if (comm::size() == 1) {
        expect_matches_serial_with_and_without_step_graphs(input, check);
    } else {
        Solver solver;
        solver.init(parse_toml(input));
        solver.run();
        check(solver);
    }
}

TEST(ParticleSnapshotsTest, ARegionKeepsExactlyTheParticlesInsideIt) {
    if (!can_write_vtkhdf()) GTEST_SKIP() << "needs HDF5, parallel on several ranks";
    const std::string prefix = (scratch_dir() / "mallard_particle_snapshots_region" / "disk").string();
    const double still[3] = {0.0, 0.0, 0.0};
    const double r = 0.3;
    const std::string region = N_DIM == 3 ? "{type = \"sphere\", center = [0.5, 0.5, 0.25], radius = 0.3}"
                                          : "{type = \"disk\", center = [0.5, 0.5], radius = 0.3}";
    const std::string writer = "[[write_data]]\nprefix = \"" + prefix +
                               "\"\nparticle_fields = [\"RHO\"]\nregion = " + region + "\ninterval = 100\n";
    Solver solver;
    solver.init(parse_toml(box(still, 1, 9, 0.05, 0.95, writer)));
    solver.run();
    comm::barrier();
    const auto snapshots = read_particles(prefix + "_particles.vtkhdf", {"RHO"});
    ASSERT_EQ(snapshots.size(), 2u);
    const auto x0 = seeds(9, 0.05, 0.95);
    std::vector<uint64_t> inside;
    for (uint64_t id = 0; id < x0.size(); id++) {
        if (std::hypot(x0[id][0] - 0.5, x0[id][1] - 0.5) < r) inside.push_back(id);
    }
    ASSERT_GT(inside.size(), 10u);
    ASSERT_LT(inside.size(), x0.size() / 2);
    for (const Snapshot & s : snapshots) {
        std::vector<uint64_t> ids;
        for (const auto & [id, v] : s.particles) {
            ids.push_back(id);
            for (size_t d = 0; d < N_DIM; d++) EXPECT_NEAR(v[d], x0[id][d], roundoff(1e-14));
            EXPECT_NEAR(v[3], 1.0, roundoff(1e-12));
        }
        EXPECT_EQ(ids, inside) << "at step " << s.step;
    }
}

TEST(ParticleSnapshotsTest, OneEntryWritesWhatAMeshEntryAndAParticleEntryWrite) {
    if (!can_write_vtkhdf()) GTEST_SKIP() << "needs HDF5, parallel on several ranks";
    const std::filesystem::path dir = scratch_dir() / "mallard_particle_snapshots_entries";
    const std::string both = (dir / "both").string(), apart = (dir / "apart").string();
    const std::string tail = "interval = 5\n";
    const std::string one = "[[write_data]]\nprefix = \"" + both +
                            "\"\nfields = [\"RHO\", \"U\"]\nparticle_fields = [\"U\"]\nparticles = \"separate\"\n" + tail;
    const std::string two = "[[write_data]]\nprefix = \"" + apart + "\"\nfields = [\"RHO\", \"U\"]\n" + tail +
                            "[[write_data]]\nprefix = \"" + apart + "\"\nparticle_fields = [\"U\"]\n" + tail;
    for (const std::string & writers : {one, two}) {
        Solver solver;
        solver.init(parse_toml(box(U, 12, 4, 0.07, 0.93, writers)));
        solver.run();
    }
    comm::barrier();
    for (const char * suffix : {".vtkhdf", "_particles.vtkhdf"}) {
        const auto a = datasets(both + suffix), b = datasets(apart + suffix);
        EXPECT_FALSE(a.empty());
        EXPECT_EQ(a, b) << suffix;
    }
}

TEST(ParticleSnapshotsTest, ACombinedFileHoldsTheSeparateFilesAsBlocks) {
    // A PartitionedDataSetCollection of the mesh file's datasets and the particle file's, each block with its steps
    if (!can_write_vtkhdf()) GTEST_SKIP() << "needs HDF5, parallel on several ranks";
    const std::filesystem::path dir = scratch_dir() / "mallard_particle_snapshots_combined";
    for (const bool series : {true, false}) {
        const std::string both = (dir / (series ? "both" : "both_files")).string();
        const std::string apart = (dir / (series ? "apart" : "apart_files")).string();
        const std::string keys = "fields = [\"RHO\", \"U\"]\nparticle_fields = [\"U\", \"RHO\"]\ninterval = 5\n" +
                                 std::string(series ? "" : "series = false\n");
        Solver combined, separate;
        combined.init(parse_toml(box(U, 12, 4, 0.07, 0.93, "[[write_data]]\nprefix = \"" + both + "\"\n" + keys)));
        combined.run();
        separate.init(parse_toml(box(U, 12, 4, 0.07, 0.93,
                                     "[[write_data]]\nprefix = \"" + apart + "\"\nparticles = \"separate\"\n" + keys)));
        separate.run();
        comm::barrier();
        const std::string last = series ? "" : "_000012";
        std::map<std::string, std::vector<double>> expected;
        for (const auto & [name, values] : datasets(apart + last + ".vtkhdf")) {
            expected["VTKHDF/mesh" + name.substr(6)] = values;
        }
        const std::string particle_file = series ? apart + "_particles.vtkhdf" : apart + "_particles_000012.vtkhdf";
        for (const auto & [name, values] : datasets(particle_file)) expected["VTKHDF/particles" + name.substr(6)] = values;
        EXPECT_EQ(datasets(both + last + ".vtkhdf"), expected) << (series ? "series" : "one file per snapshot");
        if (!comm::is_root()) continue;
        {
            const std::lock_guard<std::mutex> lock(h5::mutex());
            const std::string filename = both + last + ".vtkhdf";
            h5::Handle file(H5Fopen(filename.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT), H5Fclose, filename);
            h5::Handle top(H5Gopen2(file, "VTKHDF", H5P_DEFAULT), H5Gclose, "VTKHDF");
            EXPECT_EQ(h5::read_string_attribute(top, "Type"), "PartitionedDataSetCollection");
            for (const char * block : {"VTKHDF/Assembly/mesh", "VTKHDF/Assembly/particles"}) {
                EXPECT_GT(H5Oexists_by_name(file, block, H5P_DEFAULT), 0) << block;
            }
        }
        // Mallard's reader reads the mesh block as it reads a mesh file
        const output::Contents a = output::open(both + last + ".vtkhdf"), b = output::open(apart + last + ".vtkhdf");
        ASSERT_EQ(a.snapshots.size(), b.snapshots.size());
        EXPECT_EQ(output::read(a, "RHO", a.snapshots.size() - 1), output::read(b, "RHO", b.snapshots.size() - 1));
        const auto particles = read_particles(both + last + ".vtkhdf", {"U", "RHO"}, "VTKHDF/particles");
        EXPECT_EQ(particles.back().particles.size(), 16u);
    }
}

TEST(ParticleSnapshotsTest, ARestartContinuesTheSeriesAsTheUninterruptedRun) {
    // The continued run keeps the snapshots up to its restart file's step, drops the later ones of the run it
    // continues, and appends its own
    if (!can_write_vtkhdf()) GTEST_SKIP() << "needs HDF5, parallel on several ranks";
    const std::filesystem::path dir = scratch_dir() / "mallard_particle_snapshots_restart";
    for (const bool combined : {false, true}) {
        const std::string mode = combined ? "combined_" : "";
        auto writers = [&](const std::string & name) {
            return "[[write_data]]\nprefix = \"" + (dir / name).string() + "\"\nparticle_fields = [\"U\"]\n" +
                   (combined ? "fields = [\"RHO\"]\n" : "") + "interval = 10\n[[write_data]]\nprefix = \"" +
                   (dir / name).string() + "_restart\"\nformat = \"restart\"\ninterval = 20\n";
        };
        Solver straight;
        straight.init(parse_toml(box(U, 40, 4, 0.07, 0.93, writers(mode + "straight"))));
        straight.run();
        Solver first;
        first.init(parse_toml(box(U, 25, 4, 0.07, 0.93, writers(mode + "continued"))));
        first.run();
        Solver second;
        second.init(parse_toml(box(U, 40, 4, 0.07, 0.93, writers(mode + "continued"),
                                   "type = \"restart\"\nfile = \"" +
                                       (dir / (mode + "continued_restart_000020.restart")).string() + "\"\n")));
        second.run();
        comm::barrier();
        auto file = [&](const std::string & name) {
            return (dir / (mode + name + (combined ? ".vtkhdf" : "_particles.vtkhdf"))).string();
        };
        const std::string group = combined ? "VTKHDF/particles" : "VTKHDF";
        const auto a = read_particles(file("straight"), {"U"}, group);
        const auto b = read_particles(file("continued"), {"U"}, group);
        ASSERT_EQ(a.size(), 5u);
        ASSERT_EQ(b.size(), a.size()) << mode;
        for (size_t k = 0; k < a.size(); k++) {
            EXPECT_EQ(b[k].step, a[k].step);
            EXPECT_EQ(b[k].particles, a[k].particles) << mode << "snapshot " << k;
        }
        if (combined) {
            EXPECT_EQ(datasets(file("continued")), datasets(file("straight")));
        }
    }
}

TEST(ParticleSnapshotsTest, TheRunLogRecordsEachParticleFile) {
    if (!can_write_vtkhdf()) GTEST_SKIP() << "needs HDF5, parallel on several ranks";
    const std::filesystem::path dir = scratch_dir() / "mallard_particle_snapshots_log";
    const std::string log = (dir / "mallard.jsonl").string(), prefix = (dir / "flow").string();
    const std::string writer = "[[write_data]]\nprefix = \"" + prefix +
                               "\"\nfields = [\"RHO\"]\nparticle_fields = [\"U\"]\nparticles = \"separate\"\n"
                               "series = false\ninterval = 5\n";
    Solver solver;
    std::string input = box(U, 10, 4, 0.07, 0.93, writer);
    input.replace(input.find("[output]\n"), 9, "[output]\nrun_log = \"" + log + "\"\n");
    solver.init(parse_toml(input));
    solver.run();
    comm::barrier();
    if (!comm::is_root()) return;
    std::vector<std::string> particle_files, mesh_files;
    for (const json::Value & r : output_records(log)) {
        const std::string path = r["path"].string;
        (r["kind"].string == "particles" ? particle_files : mesh_files).push_back(path);
        ASSERT_EQ(r["files"].array.size(), 1u) << path;
        EXPECT_EQ(r["files"].array[0]["path"].string, path);
        EXPECT_EQ(r["files"].array[0]["bytes"].number, static_cast<double>(std::filesystem::file_size(path)));
        EXPECT_FALSE(r["files"].array[0]["sha256"].string.empty()) << path;
    }
    const std::vector<std::string> expected = {prefix + "_particles_000000.vtkhdf", prefix + "_particles_000005.vtkhdf",
                                               prefix + "_particles_000010.vtkhdf"};
    EXPECT_EQ(particle_files, expected);
    EXPECT_EQ(mesh_files.size(), 3u);
    // The series index lists the particle files, and the run's last snapshot is the mesh's
    std::ifstream in(prefix + "_particles.vtkhdf.series");
    const json::Value index = json::parse(std::string(std::istreambuf_iterator<char>(in), {}));
    ASSERT_EQ(index["files"].array.size(), 3u);
    EXPECT_EQ(index["files"].array[2]["name"].string, "flow_particles_000010.vtkhdf");
    std::ifstream run_log(log);
    std::string line, last_snapshot;
    while (std::getline(run_log, line)) {
        const json::Value r = json::parse(line);
        if (r["type"].string == "end") last_snapshot = r["last_snapshot"]["path"].string;
    }
    EXPECT_EQ(last_snapshot, prefix + "_000010.vtkhdf");
}

TEST(ParticleSnapshotsTest, InputErrorsNameTheKey) {
    auto error = [](const std::string & writer) {
        try {
            Solver solver;
            solver.set_distributed(false);
            solver.init(parse_toml(box(U, 1, 2, 0.2, 0.8, writer)));
        } catch (const InputError & e) {
            return std::string(e.what());
        }
        return std::string();
    };
    const std::string prefix = "[[write_data]]\nprefix = \"" + (rank_scratch_dir() / "errors").string() + "\"\n";
    const std::string unknown = error(prefix + "particle_fields = [\"U_Q\"]\ninterval = 1\n");
    EXPECT_NE(unknown.find("\"U_Q\" is no particle variable or quantity"), std::string::npos) << unknown;
    EXPECT_NE(unknown.find("particle_fields"), std::string::npos) << unknown;
    const std::string position = error(prefix + "particle_fields = [\"x\"]\ninterval = 1\n");
    EXPECT_NE(position.find("positions are the snapshot's points"), std::string::npos) << position;
    const std::string ball = N_DIM == 3 ? "{type = \"sphere\", center = [0.5, 0.5, 0.25], radius = 0.3}"
                                        : "{type = \"disk\", center = [0.5, 0.5], radius = 0.3}";
    const std::string surface = error(prefix + "particle_fields = [\"U\"]\nregion = {type = \"boundary\", of = " + ball +
                                      "}\ninterval = 1\n");
    EXPECT_NE(surface.find("[[particles.crossings]]"), std::string::npos) << surface;
}

#endif

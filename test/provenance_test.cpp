/**
 * @file provenance_test.cpp
 * @brief The run log's records and the provenance every output file carries.
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 */

#include <gtest/gtest.h>

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <set>
#include <sstream>
#include <string>
#include <vector>

#include "build_info.h"
#include "compression.h"
#include "comm.h"
#include "data_writer.h"
#include "json.h"
#include "mesh_block.h"
#include "output_reader.h"
#include "provenance.h"
#include "run_log.h"
#include "scratch.h"
#include "sha256_hw.h"
#include "solver.h"
#include "test_fixtures.h"
#ifdef Mallard_HAS_HDF5
#include "hdf5_util.h"
#endif

namespace {

using json::Value;

std::string read_file(const std::string & path) {
    std::ifstream in(path, std::ios::binary);
    std::stringstream ss;
    ss << in.rdbuf();
    return ss.str();
}

std::vector<std::string> lines(const std::string & path) {
    std::ifstream in(path);
    std::vector<std::string> out;
    for (std::string line; std::getline(in, line);) out.push_back(line);
    return out;
}

/** @brief A periodic box of uniform flow, with every kind of output under dir. */
std::string box(const std::string & dir, uint32_t n_steps, const std::string & initialize, bool hdf5) {
    std::ostringstream s;
    s << "[run]\nn_steps = " << n_steps << "\ncfl = 0.25\n"
      << "[mesh]\ntype = \"cartesian\"\nNx = 4\nNy = 4\nLx = 1.0\nLy = 1.0\n"
      << (N_DIM == 3 ? "Nz = 4\nLz = 1.0\n[[periodic]]\nzones = [\"left\", \"right\"]\n[[periodic]]\nzones = [\"bottom\", \"top\"]\n[[periodic]]\nzones = [\"back\", \"front\"]\n" : "[[periodic]]\nzones = [\"left\", \"right\"]\n[[periodic]]\nzones = [\"bottom\", \"top\"]\n")
      << "[initialize]\n" << initialize
      << "[numerics]\nriemann_solver = \"HLLC\"\n[numerics.face_reconstruction]\ntype = \"FO\"\n"
      << "[physics]\ntype = \"euler\"\ngamma = 1.4\np_ref = 1.0\nT_ref = 1.0\nrho_ref = 1.0\n"
      << "[output]\ncheck_interval = 2\nrun_log = \"" << dir << "/run.jsonl\"\n"
      << "[integrals]\ninterval = 2\nfile = \"" << dir << "/integrals.csv\"\n"
      << "[[probes]]\nname = \"p\"\nfile = \"" << dir << "/probe.csv\"\ninterval = 2\n"
      << "point = " << (N_DIM == 3 ? "[0.5, 0.5, 0.5]" : "[0.5, 0.5]") << "\nfields = [\"RHO\"]\n"
      << "[[probes]]\nname = \"line\"\nformat = \"vtp\"\nfile = \"" << dir << "/line.pvd\"\ninterval = 2\n"
      << "start = " << (N_DIM == 3 ? "[0.1, 0.5, 0.5]" : "[0.1, 0.5]") << "\nend = "
      << (N_DIM == 3 ? "[0.9, 0.5, 0.5]" : "[0.9, 0.5]") << "\nn_points = 5\nfields = [\"RHO\"]\n"
      << "[[write_data]]\nprefix = \"" << dir << "/f\"\nformat = \"vtu\"\ninterval = 4\nfields = [\"RHO\"]\n"
      << "[[write_data]]\nprefix = \"" << dir << "/r\"\nformat = \"restart\"\ninterval = 4\n";
    if (hdf5) s << "[[write_data]]\nprefix = \"" << dir << "/h\"\nformat = \"xdmf\"\ninterval = 4\nfields = [\"RHO\"]\n";
    return s.str();
}

const std::string UNIFORM = "type = \"analytical\"\nrho = \"1.0\"\np = \"1.0\"\nu = " +
                            std::string(N_DIM == 3 ? "[\"0.5\", \"0.0\", \"0.0\"]" : "[\"0.5\", \"0.0\"]") + "\n";

/** @brief Whether these runs can write HDF5: built with it, and in parallel runs with parallel HDF5. */
bool writes_hdf5() { return have_hdf5() && (comm::size() == 1 || have_parallel_hdf5()); }

/** @brief Runs the box from an input file in dir, as the executable does; returns the input file's path. */
std::string run_box(const std::string & dir, uint32_t n_steps, const std::string & initialize,
                    const std::string & extra = "") {
    const std::string input = dir + "/input_" + std::to_string(n_steps) + ".toml";
    if (comm::is_root()) {
        std::filesystem::create_directories(dir);
        std::ofstream(input) << box(dir, n_steps, initialize, writes_hdf5()) << extra;
    }
    comm::barrier();
    Solver solver;
    solver.init(input);
    solver.run();
    comm::barrier();
    return input;
}

std::set<std::string> key_set(const Value & v) {
    const auto keys = v.keys();
    return {keys.begin(), keys.end()};
}

} // namespace

TEST(ProvenanceTest, Sha256MatchesTheStandardDigests) {
    // FIPS 180-4 examples, including messages that fill the last block past its length field, on
    // the CPU's SHA instructions where it has them and on the portable compression
    const bool hardware = sha256_hw::enabled();
    for (const bool on : {true, false}) {
        sha256_hw::set_enabled(on);
        SCOPED_TRACE(sha256_hw::enabled() ? "SHA instructions" : "portable");
        EXPECT_EQ(provenance::sha256(""), "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855");
        EXPECT_EQ(provenance::sha256("abc"), "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
        EXPECT_EQ(provenance::sha256("abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq"),
                  "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1");
        EXPECT_EQ(provenance::sha256("abcdefghbcdefghicdefghijdefghijkefghijklfghijklmghijklmnhijklmno"
                                     "ijklmnopjklmnopqklmnopqrlmnopqrsmnopqrstnopqrstu"),
                  "cf5b16a778af8380036ce59e7b0492370b249b11e8f07a51afac45037afee9d1");
        EXPECT_EQ(provenance::sha256(std::string(1000000, 'a')),
                  "cdc76e5c9914fb9281a1c7e284d73e67f1809a48a497200e046d39ccc7112cd0");
    }
    sha256_hw::set_enabled(hardware);
}


TEST(ProvenanceTest, FileHashesCoverContentsAndDirectoryLayout) {
    if (!comm::is_root()) return;
    const auto dir = scratch_dir() / "mallard_provenance_hash";
    std::filesystem::remove_all(dir);
    std::filesystem::create_directories(dir / "mesh");
    const std::string content(3 << 20, 'x');  // several read buffers
    std::ofstream(dir / "mesh" / "points", std::ios::binary) << content;
    std::ofstream(dir / "mesh" / "faces") << "faces";
    EXPECT_EQ(provenance::sha256_file((dir / "mesh" / "points").string()), provenance::sha256(content));
    EXPECT_EQ(provenance::sha256_file((dir / "missing").string()), "");
    const std::string tree = provenance::sha256_file((dir / "mesh").string());
    EXPECT_EQ(tree, provenance::sha256(std::string("faces") + '\0' + "faces" + "points" + '\0' + content));
    // The same bytes under another name are another mesh
    std::filesystem::rename(dir / "mesh" / "faces", dir / "mesh" / "owner");
    EXPECT_NE(provenance::sha256_file((dir / "mesh").string()), tree);
    std::filesystem::remove_all(dir);
}

TEST(ProvenanceTest, XmlCommentsHoldNoDoubleDashAndReadBack) {
    provenance::start_run();
    provenance::set_input("a--b.toml", "# x -- y ---\n");
    const std::string comment = provenance::xml_comment();
    const std::string inner = comment.substr(4, comment.size() - 7);  // between <!-- and -->
    EXPECT_EQ(inner.find("--"), std::string::npos);
    EXPECT_EQ(provenance::parse_comment("<?xml?>\n" + comment + "\n<VTKFile>"), provenance::json());
    EXPECT_EQ(provenance::parse_comment("step,t\n" + provenance::csv_comment() + "\n1,0\n"), provenance::json());
    EXPECT_EQ(json::parse(provenance::json())["input"]["text"].string, "# x -- y ---\n");
}

TEST(RunLogTest, RecordsFollowTheSchema) {
    const std::string dir = (scratch_dir() / "mallard_run_log_schema").string();
    if (comm::is_root()) std::filesystem::remove_all(dir);
    run_box(dir, 4, UNIFORM);
    if (!comm::is_root()) return;

    std::vector<Value> records;
    for (const auto & line : lines(dir + "/run.jsonl")) records.push_back(json::parse(line));
    ASSERT_GE(records.size(), 4u);

    // The first record names the format, the run and what it restarted from, and holds the run block
    const Value & start = records.front();
    EXPECT_EQ(key_set(start), (std::set<std::string>{"type", "schema", "version", "run_id", "restart_of", "provenance",
                                                     "placement", "setup_phases"}));
    EXPECT_EQ(start["run_id"].string, provenance::run_id());
    EXPECT_TRUE(start["restart_of"].is_null());
    EXPECT_EQ(start["placement"].array.size(), static_cast<size_t>(comm::size()));
    EXPECT_EQ(start["placement"].array[0]["rank"].number, 0.0);
    EXPECT_TRUE(start["setup_phases"].has("mesh"));
    EXPECT_TRUE(start["provenance"]["dependencies"].has("kokkos"));
    EXPECT_EQ(start["provenance"]["run"]["hosts"].number, 1.0);
    EXPECT_EQ(start["type"].string, "start");
    EXPECT_EQ(start["schema"].string, run_log::SCHEMA);
    EXPECT_EQ(start["version"].number, run_log::VERSION);
    EXPECT_EQ(start["provenance"]["schema"].string, provenance::SCHEMA);

    const std::set<std::string> progress_keys = {
        "type", "step", "t", "dt", "cfl", "progress", "wall", "wall_per_step", "cells_per_second", "eta", "cells",
        "ranks", "memory_host", "memory_host_peak", "memory_device", "memory_device_total", "monitors", "integrals",
        "health"};
    std::vector<double> steps;
    double last_progress = 0.0;
    std::set<std::pair<std::string, double>> outputs;
    for (const auto & r : records) {
        const std::string type = r["type"].string;
        if (type == "progress") {
            EXPECT_EQ(key_set(r), progress_keys);
            steps.push_back(r["step"].number);
            last_progress = r["progress"].number;
            // A CFL time step: dt over the CFL-1 step is the input's CFL
            EXPECT_NEAR(r["cfl"].number, 0.25, 1e-6);
            EXPECT_DOUBLE_EQ(r["cells"].number, N_DIM == 3 ? 64.0 : 16.0);
            EXPECT_EQ(r["ranks"].number, comm::size());
            EXPECT_GT(r["memory_host"].number, 0.0);
            EXPECT_EQ(key_set(r["monitors"]), (std::set<std::string>{"min_rho", "min_p", "max_mach"}));
            EXPECT_NEAR(r["monitors"]["min_rho"].number, 1.0, 1e-6);
            EXPECT_EQ(r["integrals"]["step"].number, r["step"].number);
            EXPECT_TRUE(r["integrals"].has("kinetic_energy"));
            // Throughput and ETA from the second report on, once steps were timed
            const bool timed = r["step"].number > 0;
            EXPECT_EQ(r["wall_per_step"].type == Value::Type::NUMBER, timed);
            EXPECT_EQ(r["cells_per_second"].type == Value::Type::NUMBER, timed);
            EXPECT_EQ(r["eta"].type == Value::Type::NUMBER, timed);
        } else if (type == "output") {
            // The output index: every file completely written, by step, time, writer and reason, with its size and hash
            EXPECT_EQ(key_set(r), (std::set<std::string>{"type", "step", "t", "kind", "path", "reason", "writer",
                                                         "run_id", "complete", "bytes", "files"}));
            EXPECT_EQ(r["run_id"].string, provenance::run_id());
            EXPECT_EQ(r["reason"].string, r["step"].number == 0 ? "start" : "interval") << r["path"].string;
            EXPECT_TRUE(std::filesystem::exists(r["path"].string)) << r["path"].string;
            EXPECT_TRUE(r["complete"].boolean);
            ASSERT_FALSE(r["files"].array.empty()) << r["path"].string;
            double bytes = 0.0;
            for (const Value & f : r["files"].array) {
                const std::string file = f["path"].string;
                EXPECT_FALSE(std::filesystem::exists(file + ".partial")) << file;
                EXPECT_EQ(f["bytes"].number, double(std::filesystem::file_size(file))) << file;
                EXPECT_EQ(f["sha256"].string, provenance::sha256_file(file)) << file;
                bytes += f["bytes"].number;
            }
            EXPECT_EQ(r["bytes"].number, bytes);
            outputs.insert({r["kind"].string, r["step"].number});
        }
    }
    EXPECT_EQ(steps, (std::vector<double>{0, 2, 4}));
    for (const char * kind : {"vtu", "restart"}) {
        EXPECT_TRUE(outputs.count({kind, 0.0}) && outputs.count({kind, 4.0})) << kind;
    }
    EXPECT_EQ(last_progress, 1.0);

    const Value & end = records.back();
    EXPECT_EQ(end["type"].string, "end");
    EXPECT_EQ(key_set(end), (std::set<std::string>{"type", "outcome", "cause", "reason", "step", "t", "last_snapshot",
                                                   "last_restart", "steps", "wall", "wall_setup", "wall_stepping",
                                                   "wall_diagnostics", "wall_output", "cells_per_second"}));
    EXPECT_EQ(end["step"].number, 4);
    EXPECT_EQ(end["outcome"].string, "finished");
    EXPECT_EQ(end["cause"].string, "n_steps");
    EXPECT_NE(end["reason"].string.find("n_steps"), std::string::npos);
    EXPECT_EQ(end["last_restart"]["path"].string, dir + "/r_000004.restart");
    for (const char * last : {"last_snapshot", "last_restart"}) {
        EXPECT_EQ(end[last]["step"].number, 4) << last;
        EXPECT_EQ(end[last]["t"].number, end["t"].number) << last;
        EXPECT_TRUE(std::filesystem::exists(end[last]["path"].string)) << last;
    }
    std::filesystem::remove_all(dir);
}

#ifdef Mallard_HAS_HDF5
TEST(ProvenanceTest, Hdf5ProbeFilesCarryTheRunsProvenance) {
    if (!writes_hdf5()) GTEST_SKIP() << "needs HDF5 output on this rank count";
    const std::string dir = (scratch_dir() / "mallard_provenance_probe_h5").string();
    const std::string point = N_DIM == 3 ? "[0.5, 0.5, 0.5]" : "[0.5, 0.5]";
    std::string input = box(dir, 2, UNIFORM, false);
    input = input.substr(0, input.find("[[write_data]]")) + "[[probes]]\nname = \"ph\"\nformat = \"hdf5\"\nfile = \"" +
            dir + "/probe.h5\"\ninterval = 2\npoint = " + point + "\nfields = [\"RHO\"]\n";
    Solver solver;
    solver.init(parse_toml(input));
    solver.run();
    comm::barrier();
    if (!comm::is_root()) return;
    h5::Handle file(H5Fopen((dir + "/probe.h5").c_str(), H5F_ACC_RDONLY, H5P_DEFAULT), H5Fclose, "probe.h5");
    h5::Handle root(H5Gopen2(file, "/", H5P_DEFAULT), H5Gclose, "H5Gopen2");
    EXPECT_EQ(h5::read_strings_attribute(root, provenance::ATTRIBUTE).at(0), provenance::json());
}
#endif

TEST(ProvenanceTest, EveryOutputFileCarriesTheRunsProvenance) {
    const std::string dir = (scratch_dir() / "mallard_provenance_files").string();
    if (comm::is_root()) std::filesystem::remove_all(dir);
    const std::string input = run_box(dir, 4, UNIFORM);
    const std::string expected = provenance::json();
    if (!comm::is_root()) return;

    // What the block says about this run
    const Value p = json::parse(expected);
    EXPECT_EQ(p["schema"].string, provenance::SCHEMA);
    EXPECT_EQ(p["version"].number, provenance::VERSION);
    EXPECT_EQ(p["mallard"]["version"].string, mallard_version());
    EXPECT_EQ(p["build"]["dim"].number, N_DIM);
    EXPECT_EQ(p["build"]["precision"].string, sizeof(rtype) == 8 ? "double" : "single");
    EXPECT_EQ(p["build"]["hdf5"].boolean, have_hdf5());
    EXPECT_EQ(p["run"]["ranks"].number, comm::size());
    EXPECT_EQ(p["input"]["file"].string, input);
    EXPECT_EQ(p["input"]["text"].string, read_file(input));
    EXPECT_EQ(p["input"]["sha256"].string, provenance::sha256(read_file(input)));
    EXPECT_EQ(p["mesh"]["type"].string, "cartesian");
    EXPECT_FALSE(p["mesh"].has("sha256")) << "a generated mesh is given by the input";
    EXPECT_EQ(p["preset"].type, Value::Type::NUL);
    EXPECT_NE(p["input"]["resolved"].string.find("\ntime_integrator = \"SSPRK3\"\n"), std::string::npos)
        << "the resolved input holds what the input leaves to defaults";

    EXPECT_EQ(read_restart(dir + "/r_000004.restart").provenance, expected);
    for (const std::string & vtu : comm::size() > 1 ? std::vector<std::string>{"f_000004.pvtu", "f_000004_p0000.vtu"}
                                                    : std::vector<std::string>{"f_000004.vtu"}) {
        EXPECT_EQ(provenance::parse_comment(read_file(dir + "/" + vtu)), expected) << vtu;
    }
    EXPECT_EQ(provenance::parse_comment(read_file(dir + "/f.pvd")), expected);
    for (const char * probe : {"/line.pvd", "/line_000004.vtp"}) {
        EXPECT_EQ(provenance::parse_comment(read_file(dir + probe)), expected) << probe;
    }
    for (const char * csv : {"/integrals.csv", "/probe.csv"}) {
        const auto rows = lines(dir + csv);
        ASSERT_GE(rows.size(), 4u) << csv;
        EXPECT_EQ(rows[0].rfind("step,t", 0), 0u) << csv;
        EXPECT_EQ(rows[1].rfind("# units: ", 0), 0u) << csv;
        EXPECT_EQ(provenance::parse_comment(rows[2]), expected) << csv;
    }
    EXPECT_EQ(json::parse(lines(dir + "/run.jsonl").front())["provenance"].keys(), p.keys());
    EXPECT_NE(lines(dir + "/run.jsonl").front().find(expected), std::string::npos);
#ifdef Mallard_HAS_HDF5
    for (const char * h5 : {"/h_000004.h5", "/h_mesh.h5"}) {
        if (!writes_hdf5()) break;
        h5::Handle file(H5Fopen((dir + h5).c_str(), H5F_ACC_RDONLY, H5P_DEFAULT), H5Fclose, h5);
        h5::Handle root(H5Gopen2(file, "/", H5P_DEFAULT), H5Gclose, "H5Gopen2");
        EXPECT_EQ(h5::read_strings_attribute(root, provenance::ATTRIBUTE).at(0), expected) << h5;
    }
    if (writes_hdf5()) {
        EXPECT_EQ(provenance::parse_comment(read_file(dir + "/h.xmf")), expected);
    }
#endif
    std::filesystem::remove_all(dir);
}

TEST(ProvenanceTest, ContinuedRunsAppendTheirProvenance) {
    const std::string dir = (scratch_dir() / "mallard_provenance_restart").string();
    if (comm::is_root()) std::filesystem::remove_all(dir);
    run_box(dir, 4, UNIFORM);
    const std::string first = provenance::json();
    const std::string first_id = provenance::run_id();
    const std::string restart = dir + "/r_000004.restart";
    run_box(dir, 8, "type = \"restart\"\nfile = \"" + restart + "\"\n");
    const std::string second = provenance::json();
    const std::string second_id = provenance::run_id();
    if (!comm::is_root()) return;
    EXPECT_NE(first, second);
    EXPECT_NE(first_id, second_id);
    // The second run names the run and the snapshot it continued, in its run block and in each snapshot's
    const Value restart_of = json::parse(second)["run"]["restart_of"];
    EXPECT_EQ(restart_of["run_id"].string, first_id);
    EXPECT_EQ(restart_of["file"].string, restart);
    EXPECT_EQ(restart_of["step"].number, 4);
    EXPECT_TRUE(json::parse(first)["run"]["restart_of"].is_null());
    std::vector<std::string> files = {dir + "/f.pvd", dir + "/line.pvd", dir + "/integrals.csv", dir + "/probe.csv"};
    if (writes_hdf5()) files.push_back(dir + "/h.xmf");
    for (const std::string & name : files) {
        const output::Contents c = output::open(name);
        ASSERT_FALSE(c.snapshots.empty()) << name;
        for (const output::Snapshot & s : c.snapshots) {
            EXPECT_EQ(s.run_id, s.step <= 4 ? first_id : second_id) << name << " step " << s.step;
            if (!s.has_block) continue;
            EXPECT_EQ(s.block.restarted, s.step > 4) << name << " step " << s.step;
            if (s.step > 4) {
                EXPECT_EQ(s.block.restart_of_run_id, first_id) << name;
                EXPECT_EQ(s.block.restart_of_step, 4) << name;
            }
        }
        EXPECT_EQ(c.snapshots.back().step, 8) << name;
    }
    const output::Contents last = output::open(dir + "/r_000008.restart");
    ASSERT_TRUE(last.snapshots.at(0).has_block);
    EXPECT_EQ(last.snapshots[0].block.run_id, second_id);
    EXPECT_EQ(last.snapshots[0].block.restart_of_file, restart);
    // Each part of a monitor file follows the provenance of the run that wrote it
    std::vector<std::string> blocks;
    for (const auto & row : lines(dir + "/integrals.csv")) {
        if (row.rfind("# mallard-run ", 0) == 0) blocks.push_back(provenance::parse_comment(row));
    }
    EXPECT_EQ(blocks, (std::vector<std::string>{first, second}));
    std::vector<std::string> starts;
    for (const auto & line : lines(dir + "/run.jsonl")) {
        const Value r = json::parse(line);
        if (r["type"].string == "start") {
            starts.push_back(r["provenance"]["input"]["file"].string);
            EXPECT_EQ(r["run_id"].string, starts.size() == 1 ? first_id : second_id);
            EXPECT_EQ(r["restart_of"].is_null(), starts.size() == 1);
        }
    }
    EXPECT_EQ(starts, (std::vector<std::string>{dir + "/input_4.toml", dir + "/input_8.toml"}));
    EXPECT_EQ(read_restart(dir + "/r_000008.restart").provenance, second);
    std::filesystem::remove_all(dir);
}

TEST(ProvenanceTest, RecordsTheFilesTheRunRead) {
    const std::string dir = (scratch_dir() / "mallard_provenance_read_files").string();
    const std::string points = dir + "/points.txt";
    if (comm::is_root()) {
        std::filesystem::remove_all(dir);
        std::filesystem::create_directories(dir);
        std::ofstream(points) << (N_DIM == 3 ? "0.25 0.25 0.5\n0.75 0.75 0.5\n" : "0.25 0.25\n0.75 0.75\n");
    }
    const std::string probe = "[[probes]]\nname = \"pts\"\ntype = \"points\"\npoints_file = \"" + points +
                              "\"\nfile = \"" + dir + "/pts.csv\"\ninterval = 2\nvariables = [\"RHO\"]\n";
    run_box(dir, 4, UNIFORM, probe);
    const std::string first_id = provenance::run_id();
    const Value first = json::parse(provenance::json());
    const std::string restart = dir + "/r_000004.restart";
    run_box(dir, 8, "type = \"restart\"\nfile = \"" + restart + "\"\n");
    const Value second = json::parse(provenance::json());
    const std::string copy = dir + "/copy.restart";
    if (comm::is_root()) std::filesystem::copy_file(restart, copy);
    run_box(dir, 12, "type = \"restart\"\nfile = \"" + copy + "\"\n");
    const Value third = json::parse(provenance::json());
    if (!comm::is_root()) return;

    // A small input file is hashed, once whichever ranks read it
    ASSERT_EQ(first["files"].array.size(), 1u);
    const Value & read = first["files"].array[0];
    EXPECT_EQ(read["kind"].string, "probe_points");
    EXPECT_EQ(read["path"].string, points);
    EXPECT_EQ(read["sha256"].string, provenance::sha256_file(points));
    // A restart file is not hashed again: its run, step and the SHA-256 of that run's output record
    ASSERT_EQ(second["files"].array.size(), 1u);
    const Value & start = second["files"].array[0];
    EXPECT_EQ(start["kind"].string, "restart");
    EXPECT_EQ(start["run_id"].string, first_id);
    EXPECT_EQ(start["step"].number, 4);
    EXPECT_EQ(start["sha256"].string, provenance::sha256_file(restart));
    EXPECT_EQ(start["bytes"].number, double(std::filesystem::file_size(restart)));
    // A copy has no output record of its own: its size and time instead
    const Value & copied = third["files"].array.at(0);
    EXPECT_EQ(copied["run_id"].string, first_id);
    EXPECT_TRUE(copied["sha256"].is_null());
    EXPECT_EQ(copied["note"].string, "not hashed (no output record)");
    EXPECT_FALSE(copied["modified"].string.empty());
    std::filesystem::remove_all(dir);
}

TEST(RunLogTest, ResumedRunsAppendAndMarkRewinds) {
    const std::string dir = (scratch_dir() / "mallard_run_log_resume").string();
    if (comm::is_root()) std::filesystem::remove_all(dir);
    const std::string restart = dir + "/r_000004.restart";
    run_box(dir, 4, UNIFORM);
    run_box(dir, 8, "type = \"restart\"\nfile = \"" + restart + "\"\n");
    run_box(dir, 8, "type = \"restart\"\nfile = \"" + restart + "\"\n");
    if (!comm::is_root()) return;
    std::vector<Value> starts;
    std::vector<std::vector<double>> steps;
    for (const auto & line : lines(dir + "/run.jsonl")) {
        const Value r = json::parse(line);
        if (r["type"].string == "start") {
            starts.push_back(r);
            steps.emplace_back();
        } else if (r.has("step") && r["type"].string != "output") {
            // Output records follow the completion of their write, not the step
            ASSERT_FALSE(steps.empty());
            steps.back().push_back(r["step"].number);
        }
    }
    ASSERT_EQ(starts.size(), 3u);
    EXPECT_FALSE(starts[0].has("restart"));
    for (size_t i : {1u, 2u}) {
        EXPECT_EQ(starts[i]["restart"]["file"].string, restart);
        EXPECT_EQ(starts[i]["restart"]["step"].number, 4.0);
    }
    EXPECT_FALSE(starts[1]["restart"].has("rewound_from"));
    EXPECT_EQ(starts[2]["restart"]["rewound_from"].number, 8.0);
    const std::vector<std::pair<double, double>> spans = {{0.0, 4.0}, {4.0, 8.0}, {4.0, 8.0}};
    for (size_t i = 0; i < spans.size(); i++) {
        ASSERT_FALSE(steps[i].empty());
        EXPECT_TRUE(std::ranges::is_sorted(steps[i]));
        EXPECT_EQ(steps[i].front(), spans[i].first);
        EXPECT_EQ(steps[i].back(), spans[i].second);
    }
    std::filesystem::remove_all(dir);
}

TEST(ProvenanceTest, RecordsTheResolvedNumericsPreset) {
    const std::string dir = (scratch_dir() / "mallard_provenance_preset").string();
    std::string input = box(dir, 1, UNIFORM, false);
    input.replace(input.find("[numerics]\n"), 11, "[numerics]\npreset = \"second_order\"\n");
    Solver solver;
    solver.init(parse_toml(input));
    const Value p = json::parse(provenance::json());
    const auto & resolved = solver.get_numerics_preset();
    const ResolvedInput records = resolve_input(solver.resolved_input());
    EXPECT_EQ(p["preset"]["name"].string, "second_order");
    EXPECT_FALSE(p["preset"]["default"].boolean);
    ASSERT_EQ(p["preset"]["settings"].array.size(), resolved.settings.size());
    bool overridden = false;
    for (size_t i = 0; i < resolved.settings.size(); i++) {
        const Value & s = p["preset"]["settings"].array[i];
        EXPECT_EQ(s["key"].string, resolved.settings[i].key);
        ASSERT_NE(records.find(s["key"].string), nullptr) << s["key"].string;
        EXPECT_EQ(s["value"].string, records.find(s["key"].string)->value) << "a view of the key's record";
        EXPECT_EQ(s["from_preset"].boolean, resolved.settings[i].from_preset);
        if (s["key"].string == "numerics.riemann_solver") {
            overridden = true;
            EXPECT_EQ(s["value"].string, "\"HLLC\"");
            EXPECT_FALSE(s["from_preset"].boolean) << "the input sets it";
        }
    }
    EXPECT_TRUE(overridden);

    // Without a preset or a reconstruction, the default, marked as such
    std::string bare = box(dir, 1, UNIFORM, false);
    const size_t recon = bare.find("[numerics.face_reconstruction]");
    ASSERT_NE(recon, std::string::npos);
    bare.erase(recon, bare.find("[", recon + 1) - recon);
    Solver by_default;
    by_default.init(parse_toml(bare));
    const Value q = json::parse(provenance::json());
    EXPECT_EQ(q["preset"]["name"].string, "high_order");
    EXPECT_TRUE(q["preset"]["default"].boolean);
    if (comm::is_root()) std::filesystem::remove_all(dir);
}

// The provenance holds the input as setup used it, the text mallard input resolve
// writes: a quantity with a unit in the key's units, a default written out, and
// the same text again when run. Fails if check and a run read different keys, or a
// key written out changes what setup reads.
TEST(ProvenanceTest, HoldsTheResolvedInput) {
    const std::string dir = (scratch_dir() / "mallard_provenance_resolved").string();
    std::string input = box(dir, 2, UNIFORM, false);
    input.replace(input.find("cfl = 0.25\n"), 11, "cfl = 0.25\nt_stop = \"1 min\"\n");
    Solver solver;
    solver.init(parse_toml(input));
    const std::string resolved = json::parse(provenance::json())["input"]["resolved"].string;
    EXPECT_NE(resolved.find("\nt_stop = 60.0\n"), std::string::npos) << resolved;
    EXPECT_NE(resolved.find("\nconvective_flux = \"riemann\"\n"), std::string::npos) << "a default";
    Solver checked;
    EXPECT_EQ(checked.check(parse_toml(input), CheckOptions{}).resolved, resolved);
    Solver again;
    again.init(parse_toml(resolved));
    EXPECT_EQ(json::parse(provenance::json())["input"]["resolved"].string, resolved);
    if (comm::is_root()) std::filesystem::remove_all(dir);
}

TEST(ProvenanceTest, CompressedFilesCarryTheProvenance) {
    if (!compression::have_zlib()) GTEST_SKIP() << "no zlib";
    const std::string dir = (scratch_dir() / "mallard_provenance_zlib").string();
    if (comm::is_root()) std::filesystem::remove_all(dir);
    std::string input = box(dir, 4, UNIFORM, false);
    for (size_t at = 0; (at = input.find("interval = 4\n", at)) != std::string::npos; at += 13) {
        input.insert(at, "compression = \"zlib\"\n");
        at += 21;
    }
    Solver solver;
    solver.init(parse_toml(input));
    solver.run();
    comm::barrier();
    const std::string expected = provenance::json();
    if (!comm::is_root()) return;
    EXPECT_EQ(read_restart(dir + "/r_000004.restart").provenance, expected);
    const std::string vtu = comm::size() > 1 ? "/f_000004_p0000.vtu" : "/f_000004.vtu";
    EXPECT_NE(read_file(dir + vtu).find("compressor"), std::string::npos) << "compressed";
    EXPECT_EQ(provenance::parse_comment(read_file(dir + vtu)), expected);
    std::filesystem::remove_all(dir);
}

TEST(ProvenanceTest, LargeProvenanceFitsInHdf5Attributes) {
#ifndef Mallard_HAS_HDF5
    GTEST_SKIP() << "no HDF5";
#else
    if (!writes_hdf5()) GTEST_SKIP() << "no parallel HDF5";
    // An input past HDF5's 64 KiB limit of compact attributes
    const std::string dir = (scratch_dir() / "mallard_provenance_large").string();
    if (comm::is_root()) std::filesystem::remove_all(dir);
    const std::string input = dir + "/input.toml";
    if (comm::is_root()) {
        std::filesystem::create_directories(dir);
        std::ofstream(input) << box(dir, 4, UNIFORM, true) << "# " << std::string(100000, 'x') << "\n";
    }
    comm::barrier();
    Solver solver;
    solver.init(input);
    solver.run();
    comm::barrier();
    if (!comm::is_root()) return;
    h5::Handle file(H5Fopen((dir + "/h_000004.h5").c_str(), H5F_ACC_RDONLY, H5P_DEFAULT), H5Fclose, "h5");
    h5::Handle root(H5Gopen2(file, "/", H5P_DEFAULT), H5Gclose, "H5Gopen2");
    EXPECT_EQ(h5::read_strings_attribute(root, provenance::ATTRIBUTE).at(0), provenance::json());
    EXPECT_GT(provenance::json().size(), 100000u);
    std::filesystem::remove_all(dir);
#endif
}

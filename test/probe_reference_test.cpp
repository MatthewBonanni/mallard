/**
 * @file probe_reference_test.cpp
 * @author Matthew Bonanni (mbonanni001@gmail.com)
 * @brief Probe output of every backend against a committed CPU reference
 *        (test/data/probes), so that GPU runs are checked against the CPU.
 * @version 0.1
 * @date 2026-10-30
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 *
 */

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

#include "comm.h"
#include "data_paths.h"
#include "scratch.h"
#include "solver.h"
#include "test_fixtures.h"
#include "test_utils.h"

namespace {

std::string vector3(const char * x, const char * y, const char * z) {
    return std::string("[\"") + x + "\", \"" + y + "\"" + (N_DIM == 3 ? std::string(", \"") + z + "\"" : "") + "]";
}

std::string numbers3(const char * x, const char * y, const char * z) {
    return std::string("[") + x + ", " + y + (N_DIM == 3 ? std::string(", ") + z : "") + "]";
}

const std::vector<std::string> PROBES = {"points", "line", "circle", "cut", "walls", "open"
#if N_DIM == 3
                                         , "sphere"
#endif
};

/**
 * @brief A smooth viscous flow on simplices between an isothermal and an
 *        adiabatic wall, with a fixed step, and every probe kind with point
 *        values and every reduction.
 */
std::string reference_case(const std::string & dir) {
    std::ostringstream s;
    s << "[run]\nn_steps = 6\ndt = 4.0e-3\n[mesh]\n";
    if (N_DIM == 2) s << "type = \"cartesian_tri\"\nNx = 16\nNy = 12\nLx = 1.0\nLy = 0.8\n";
    else s << "type = \"cartesian_tet\"\nNx = 6\nNy = 5\nNz = 4\nLx = 1.0\nLy = 0.8\nLz = 0.6\nperiodic = [\"z\"]\n";
    s << "[initialize]\ntype = \"analytical\"\nrho = \"1.0 + 0.2 * sin(2 * pi * x) * cos(pi * y)\"\nu = "
      << vector3("0.3 + 0.1 * cos(2 * pi * y)", "0.05 * sin(2 * pi * x)", "0.04 * cos(2 * pi * x)")
      << "\np = \"1.0 + 0.1 * cos(2 * pi * x) * sin(pi * y)\"\n"
      << "[[boundaries]]\nname = \"left\"\ntype = \"extrapolation\"\n"
      << "[[boundaries]]\nname = \"right\"\ntype = \"extrapolation\"\n"
      << "[[boundaries]]\nname = \"bottom\"\ntype = \"wall_isothermal\"\nT = 1.1\n"
      << "[[boundaries]]\nname = \"top\"\ntype = \"wall_adiabatic\"\n"
      << "[numerics]\nriemann_solver = \"HLLC\"\ntime_integrator = \"SSPRK3\"\n"
      << "[numerics.face_reconstruction]\ntype = \"TENO\"\norder = 3\n"
      << "[physics]\ntype = \"navier_stokes\"\nmu = 0.01\nPr = 0.72\ngamma = 1.4\np_ref = 1.0\nT_ref = 1.0\n"
      << "rho_ref = 1.0\n[output]\ncheck_interval = 1000000\n";
    const std::map<std::string, std::string> geometry = {
        {"points", "shape = \"points\"\npoints = [" + numbers3("0.21", "0.33", "0.17") + ", " +
                       numbers3("0.67", "0.52", "0.41") + ", " + numbers3("0.45", "0.71", "0.29") +
                       "]\nvariables = [\"RHO\", \"U\", \"P\", \"T\", \"H\"]\n"},
        {"line", "start = " + numbers3("0.07", "0.11", "0.11") + "\nend = " + numbers3("0.92", "0.66", "0.47") +
                     "\nn_points = 17\nvariables = [\"RHO\", \"U\", \"P\"]\n"},
        {"circle", "shape = \"circle\"\ncenter = " + numbers3("0.48", "0.41", "0.3") +
                       "\nradius = 0.23\nn_points = 19\n" + (N_DIM == 3 ? "normal = [0.0, 0.0, 1.0]\n" : "") +
                       (N_DIM == 2 ? "variables = [\"P\", \"MASS_FLUX\"]\n" : "variables = [\"P\", \"U\"]\n")},
        {"cut", "shape = \"cut_plane\"\npoint = " + numbers3("0.47", "0.39", "0.3") + "\nnormal = " +
                    numbers3("1.0", "0.7", "0.3") + "\nvariables = [\"RHO\", \"P\", \"MASS_FLUX\"]\n"},
        {"walls", "boundary_type = \"wall\"\nvariables = [\"P\", \"T\", \"CP\", \"CF\", \"TAU_W\", \"Q_W\", \"U_TAU\", "
                  "\"Y_PLUS\"]\nreference = {rho = 1.0, u = " + numbers3("0.3", "0.0", "0.0") + ", p = 1.0}\n"},
        {"open", "boundary = [\"left\", \"right\"]\nvariables = [\"P\", \"MASS_FLUX\", \"MOMENTUM_FLUX\", "
                 "\"ENTHALPY_FLUX\"]\n"},
        {"sphere", "shape = \"sphere\"\ncenter = [0.5, 0.4, 0.3]\nradius = 0.2\nn_points = [6, 12]\n"
                   "variables = [\"RHO\", \"P\", \"MASS_FLUX\"]\n"},
    };
    for (const auto & name : PROBES) {
        s << "[[probes]]\nname = \"" << name << "\"\nfile = \"" << dir << "/" << name << ".csv\"\nreductions_file = \""
          << dir << "/" << name << "_reductions.csv\"\nsampling = \"point\"\ninterval = 3\n"
          << "reductions = [\"integral\", \"mean\", \"favre\", \"min\", \"max\", \"rms\"]\n" << geometry.at(name);
    }
    return s.str();
}

struct Table {
    std::vector<std::string> names;
    std::vector<std::vector<std::string>> rows;
};

/** @brief A CSV file's header and rows, without its comment lines (provenance, units). */
Table read_table(const std::string & file) {
    std::ifstream in(file);
    Table table;
    auto split = [](const std::string & line) {
        std::vector<std::string> cells;
        std::stringstream ss(line);
        for (std::string cell; std::getline(ss, cell, ',');) cells.push_back(cell);
        return cells;
    };
    std::string line;
    if (std::getline(in, line)) table.names = split(line);
    while (std::getline(in, line)) {
        if (!line.empty() && line[0] != '#') table.rows.push_back(split(line));
    }
    return table;
}

enum class Kind { EXACT, GEOMETRY, VALUE };

Kind kind_of(const std::string & column) {
    for (const char * name : {"step", "t", "point", "cell", "face"}) {
        if (column == name) return Kind::EXACT;
    }
    for (const char * name : {"x", "y", "z", "s", "nx", "ny", "nz", "area", "count", "length", "volume"}) {
        if (column == name) return Kind::GEOMETRY;
    }
    return Kind::VALUE;
}

// Relative to the largest magnitude of the column in the reference
constexpr double GEOMETRY_TOL = 1e-14;
constexpr double VALUE_TOL = 1e-10;

void expect_matches_reference(const std::string & file, const std::string & reference) {
    const Table a = read_table(file), ref = read_table(reference);
    ASSERT_FALSE(ref.rows.empty()) << reference;
    ASSERT_EQ(a.names, ref.names) << file;
    ASSERT_EQ(a.rows.size(), ref.rows.size()) << file;
    for (size_t c = 0; c < ref.names.size(); c++) {
        const Kind kind = kind_of(ref.names[c]);
        double scale = 0.0, worst = 0.0;
        size_t worst_row = 0;
        for (const auto & row : ref.rows) scale = std::max(scale, std::fabs(std::stod(row.at(c))));
        for (size_t r = 0; r < ref.rows.size(); r++) {
            ASSERT_EQ(a.rows[r].size(), ref.names.size()) << file << " row " << r;
            if (kind == Kind::EXACT) {
                EXPECT_EQ(a.rows[r][c], ref.rows[r][c]) << file << " row " << r << " " << ref.names[c];
                continue;
            }
            const double d = std::fabs(std::stod(a.rows[r][c]) - std::stod(ref.rows[r][c]));
            if (d > worst) {
                worst = d;
                worst_row = r;
            }
        }
        if (kind == Kind::EXACT) continue;
        const double tol = (kind == Kind::GEOMETRY ? GEOMETRY_TOL : VALUE_TOL) * scale;
        EXPECT_LE(worst, tol) << file << " " << ref.names[c] << " row " << worst_row << ": "
                              << a.rows[worst_row][c] << " against " << ref.rows[worst_row][c];
    }
}

} // namespace

TEST(ProbesTest, EveryBackendReproducesTheCpuReference) {
    // The geometry is set up on the host and the reductions are taken on the
    // host from the gathered samples: only the solution and the point values,
    // computed on the device, may differ from the CPU in their rounding.
    // MALLARD_UPDATE_PROBE_REFERENCE=1 rewrites the reference from a CPU build
    SKIP_IN_SINGLE_PRECISION("the reference is double precision");
    const std::string dir = scratch_dir().string();
    Solver solver;
    solver.init(parse_toml(reference_case(dir)));
    solver.run();
    comm::barrier();
    if (!comm::is_root()) return;
    const std::filesystem::path reference =
        std::filesystem::path(source_dir()) / "test" / "data" / "probes" / (std::to_string(N_DIM) + "d");
    const bool update = std::getenv("MALLARD_UPDATE_PROBE_REFERENCE") != nullptr;
    if (update) std::filesystem::create_directories(reference);
    for (const auto & name : PROBES) {
        for (const std::string & file : {name + ".csv", name + "_reductions.csv"}) {
            if (update) {
                std::ifstream in(dir + "/" + file);
                std::ofstream out(reference / file);
                for (std::string line; std::getline(in, line);) {
                    if (line.empty() || line[0] != '#') out << line << "\n";
                }
                continue;
            }
            expect_matches_reference(dir + "/" + file, (reference / file).string());
        }
    }
}

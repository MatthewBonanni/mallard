/**
 * @file particle_trajectories_test.cpp
 * @author Matthew Bonanni (mbonanni001@gmail.com)
 * @brief [[particles.trajectories]]: the particles of a subset at their
 *        output steps, with their fields, on any rank count and across
 *        restarts.
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
#include <set>
#include <sstream>
#include <string>
#include <vector>

#include "comm.h"
#include "particles.h"
#include "scratch.h"
#include "solver.h"
#include "test_fixtures.h"

namespace {

std::string numbers(double x, double y, double z) {
    std::ostringstream s;
    s.precision(17);
    s << "[" << x << ", " << y;
    if (N_DIM == 3) s << ", " << z;
    s << "]";
    return s.str();
}

/** @brief Uniform flow u in the periodic unit box (z: 0.5) of 12^2 cells, n x n tracers in [lo, hi]. */
std::string box(const double u[3], uint64_t n_steps, uint32_t n, double lo, double hi, const std::string & extra,
                const std::string & init = "") {
    std::ostringstream s;
    s.precision(17);
    s << "[run]\nn_steps = " << n_steps << "\ndt = 0.01\n[mesh]\ntype = \"cartesian\"\nNx = 12\nNy = 12\nLx = 1.0\n"
      << "Ly = 1.0\n";
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
      << numbers(lo, lo, 0.25) << "\nhi = " << numbers(hi, hi, 0.25) << "\nn = [" << n << ", " << n
      << (N_DIM == 3 ? ", 1]" : "]") << "\n" << extra;
    return s.str();
}

/** @brief The seed of box(n, lo, hi) with this id. */
std::array<double, 2> seed(uint64_t id, uint32_t n, double lo, double hi) {
    const double step = (hi - lo) / static_cast<double>(n - 1);
    return {lo + step * static_cast<double>(id % n), lo + step * static_cast<double>(id / n % n)};
}

/** @brief The rows of a CSV file after its header, as text, without the comment lines. */
std::vector<std::string> lines(const std::string & file) {
    std::ifstream in(file);
    std::vector<std::string> out;
    std::string line;
    bool header = true;
    while (std::getline(in, line)) {
        if (line.empty() || line[0] == '#') continue;
        if (!header) out.push_back(line);
        header = false;
    }
    return out;
}

/** @brief The header of a CSV file. */
std::string header(const std::string & file) {
    std::ifstream in(file);
    std::string line;
    while (std::getline(in, line)) {
        if (!line.empty() && line[0] != '#') return line;
    }
    return "";
}

std::vector<std::vector<double>> rows(const std::string & file) {
    std::vector<std::vector<double>> out;
    for (const std::string & line : lines(file)) {
        std::vector<double> row;
        std::stringstream s(line);
        std::string cell;
        while (std::getline(s, cell, ',')) row.push_back(std::stod(cell));
        out.push_back(row);
    }
    return out;
}

/** @brief The particle ids of a trajectory file at one step. */
std::set<uint64_t> ids_at(const std::vector<std::vector<double>> & records, double step) {
    std::set<uint64_t> out;
    for (const auto & r : records) {
        if (r[0] == step) out.insert(static_cast<uint64_t>(r[2]));
    }
    return out;
}

double wrap(double x) { return x - std::floor(x); }

} // namespace

TEST(ParticleTrajectoriesTest, EveryRecordIsAParticleAtItsOutputWithItsFields) {
    // Tracers in uniform flow: at t a particle is at its seed plus u t (wrapped), has the gas velocity u and an age
    // of t; every 5 steps and every 0.07 of time from the start, every particle once, by id
    const std::filesystem::path dir = scratch_dir() / "mallard_particle_trajectories_uniform";
    const double u[3] = {0.3, 0.2, 0.0};
    for (const bool graphs : {true, false}) {
        const std::string out = (dir / (graphs ? "graphs" : "plain")).string();
        const std::string entries =
            "[[particles.quantities]]\nname = \"age\"\nintegral = \"1\"\n"
            "[[particles.trajectories]]\nname = \"steps\"\ninterval = 5\nfields = [\"U\", \"age\"]\nfile = \"" + out +
            "_steps.csv\"\n[[particles.trajectories]]\nname = \"time\"\ntime_interval = 0.07\nfile = \"" + out +
            "_time.csv\"\n";
        std::string input = box(u, 20, 4, 0.1, 0.9, entries);
        if (!graphs) input.replace(input.find("[run]\n"), 6, "[run]\ncuda_graphs = false\n");
        Solver solver;
        solver.init(parse_toml(input));
        solver.run();
        comm::barrier();
    }
    const std::string out = (dir / "graphs").string();
    const double tol = roundoff(1e-12);
    EXPECT_EQ(header(out + "_steps.csv"),
              std::string("step,t,id,x,y") + (N_DIM == 3 ? ",z,U_X,U_Y,U_Z" : ",U_X,U_Y") + ",age");
    const auto steps = rows(out + "_steps.csv");
    ASSERT_EQ(steps.size(), 5u * 16u);
    for (size_t k = 0; k < steps.size(); k++) {
        const auto & r = steps[k];
        const uint64_t id = static_cast<uint64_t>(r[2]);
        EXPECT_EQ(r[0], 5.0 * static_cast<double>(k / 16));
        EXPECT_EQ(id, k % 16) << "by id";
        const double t = 0.01 * r[0];
        EXPECT_NEAR(r[1], t, tol);
        const auto x0 = seed(id, 4, 0.1, 0.9);
        EXPECT_NEAR(r[3], wrap(x0[0] + u[0] * t), tol) << "particle " << id << " at step " << r[0];
        EXPECT_NEAR(r[4], wrap(x0[1] + u[1] * t), tol) << "particle " << id << " at step " << r[0];
        for (size_t i = 0; i < N_DIM; i++) EXPECT_NEAR(r[3 + N_DIM + i], u[i], tol) << "U";
        EXPECT_NEAR(r[3 + 2 * N_DIM], t, tol) << "age";
    }
    const auto time = rows(out + "_time.csv");
    ASSERT_EQ(time.size(), 3u * 16u);
    for (size_t k = 0; k < time.size(); k++) {
        EXPECT_EQ(time[k][0], 7.0 * static_cast<double>(k / 16));
        EXPECT_EQ(time[k].size(), 3u + N_DIM);
    }
    for (const char * kind : {"_steps.csv", "_time.csv"}) {
        EXPECT_EQ(lines((dir / ("plain" + std::string(kind))).string()), lines(out + kind)) << kind;
    }
}

TEST(ParticleTrajectoriesTest, ASubsetIsAFractionOfTheIdsAndTheParticlesSeededInARegion) {
    // 400 tracers moving right: a fraction keeps about that share of them, the same ones at every output, and the
    // particles of a smaller fraction are among a larger one's; a region keeps those seeded in it wherever they go
    const std::filesystem::path dir = scratch_dir() / "mallard_particle_trajectories_subset";
    const double u[3] = {0.5, 0.0, 0.0};
    auto entry = [&](const std::string & name, const std::string & subset) {
        return "[[particles.trajectories]]\nname = \"" + name + "\"\ninterval = 10\n" + subset + "file = \"" +
               (dir / name).string() + ".csv\"\n";
    };
    const std::string left = "region = {type = \"half_space\", point = " + numbers(0.5, 0.0, 0.0) + ", normal = " +
                             numbers(1.0, 0.0, 0.0) + "}\n";
    Solver solver;
    solver.init(parse_toml(box(u, 40, 20, 0.025, 0.975,
                               entry("tenth", "fraction = 0.1\n") + entry("half", "fraction = 0.5\n") +
                                   entry("left", left) + entry("both", "fraction = 0.5\n" + left))));
    solver.run();
    comm::barrier();
    std::map<std::string, std::set<uint64_t>> kept;
    for (const char * name : {"tenth", "half", "left", "both"}) {
        const auto records = rows((dir / (std::string(name) + ".csv")).string());
        kept[name] = ids_at(records, 0.0);
        // The same particles at every output, though at t = 0.4 a fifth of the left half has moved right of 0.5
        for (const double step : {10.0, 20.0, 30.0, 40.0}) EXPECT_EQ(ids_at(records, step), kept[name]) << name;
    }
    // Binomial: 4 standard deviations
    EXPECT_NEAR(static_cast<double>(kept["tenth"].size()), 40.0, 4.0 * 6.0);
    EXPECT_NEAR(static_cast<double>(kept["half"].size()), 200.0, 4.0 * 10.0);
    for (const uint64_t id : kept["tenth"]) EXPECT_EQ(kept["half"].count(id), 1u) << "particle " << id;
    EXPECT_EQ(kept["left"].size(), 200u);
    for (const uint64_t id : kept["left"]) EXPECT_LT(seed(id, 20, 0.025, 0.975)[0], 0.5) << "particle " << id;
    std::set<uint64_t> both;
    for (const uint64_t id : kept["half"]) {
        if (kept["left"].count(id)) both.insert(id);
    }
    EXPECT_EQ(kept["both"], both);
}

TEST(ParticleTrajectoriesTest, RecordsAreBitwiseTheSameOnAnyRankCount) {
    if (comm::size() == 1) GTEST_SKIP() << "compares distributed runs with serial ones";
    const std::filesystem::path dir = scratch_dir() / "mallard_particle_trajectories_ranks";
    const double u[3] = {0.7, -0.45, 0.2};
    auto input = [&](const std::string & name) {
        return box(u, 60, 8, 0.07, 0.93,
                   "[[particles.trajectories]]\nname = \"some\"\ninterval = 4\nfraction = 0.6\nfields = [\"U\"]\n"
                   "file = \"" + (dir / name).string() + "\"\n");
    };
    Solver distributed;
    distributed.init(parse_toml(input("distributed.csv")));
    distributed.run();
    comm::barrier();
    if (!comm::is_root()) return;
    Solver serial;
    serial.set_distributed(false);
    serial.init(parse_toml(input("serial.csv")));
    serial.run();
    const auto a = lines((dir / "distributed.csv").string());
    EXPECT_GE(a.size(), 16u * 20u);
    EXPECT_EQ(a, lines((dir / "serial.csv").string()));
}

TEST(ParticleTrajectoriesTest, ARestartContinuesTheFileAsTheUninterruptedRun) {
    const std::filesystem::path dir = scratch_dir() / "mallard_particle_trajectories_restart";
    const double u[3] = {0.3, 0.2, 0.0};
    auto input = [&](const std::string & name, uint64_t n_steps, const std::string & init) {
        const std::string file = (dir / name).string();
        return box(u, n_steps, 4, 0.1, 0.9,
                   "[[particles.trajectories]]\nname = \"steps\"\ninterval = 3\nfile = \"" + file + "_steps.csv\"\n"
                   "[[particles.trajectories]]\nname = \"time\"\ntime_interval = 0.04\nfile = \"" + file +
                       "_time.csv\"\n[[write_data]]\nprefix = \"" + file + "\"\nformat = \"restart\"\ninterval = 6\n",
                   init);
    };
    Solver straight;
    straight.init(parse_toml(input("straight", 20, "")));
    straight.run();
    Solver first;
    first.init(parse_toml(input("continued", 6, "")));
    first.run();
    comm::barrier();
    Solver second;
    second.init(parse_toml(
        input("continued", 20, "type = \"restart\"\nfile = \"" + (dir / "continued_000006.restart").string() + "\"\n")));
    second.run();
    comm::barrier();
    const auto steps = lines((dir / "straight_steps.csv").string());
    EXPECT_EQ(steps.size(), 7u * 16u);
    EXPECT_EQ(lines((dir / "continued_steps.csv").string()), steps);
    const auto time = lines((dir / "straight_time.csv").string());
    EXPECT_EQ(time.size(), 6u * 16u);
    EXPECT_EQ(lines((dir / "continued_time.csv").string()), time);
}

TEST(ParticleTrajectoriesTest, InputErrorsNameTheKey) {
    auto error = [](const std::string & entry) {
        try {
            Solver solver;
            solver.set_distributed(false);
            const double u[3] = {0.3, 0.2, 0.0};
            solver.init(parse_toml(box(u, 1, 4, 0.2, 0.8, entry)));
        } catch (const InputError & e) {
            return std::string(e.what());
        }
        return std::string();
    };
    const std::string file = "file = \"" + (rank_scratch_dir() / "errors.csv").string() + "\"\n";
    const std::string start = "[[particles.trajectories]]\nname = \"a\"\n" + file;
    const std::string both = error(start + "interval = 1\ntime_interval = 0.1\n");
    EXPECT_NE(both.find("sets both interval and time_interval"), std::string::npos) << both;
    const std::string neither = error(start);
    EXPECT_NE(neither.find("sets neither"), std::string::npos) << neither;
    const std::string fraction = error(start + "interval = 1\nfraction = 0.0\n");
    EXPECT_NE(fraction.find("particles.trajectories[0].fraction"), std::string::npos) << fraction;
    const std::string empty = error(start + "interval = 1\nregion = {type = \"half_space\", point = " +
                                    numbers(0.1, 0.0, 0.0) + ", normal = " + numbers(1.0, 0.0, 0.0) + "}\n");
    EXPECT_NE(empty.find("selects none of the 16 particles"), std::string::npos) << empty;
    const std::string zone = error(start + "interval = 1\nregion = \"domain\"\n");
    EXPECT_NE(zone.find("must be a filled region of shapes"), std::string::npos) << zone;
    const std::string field = error(start + "interval = 1\nfields = [\"x\"]\n");
    EXPECT_NE(field.find("particles.trajectories[0].fields"), std::string::npos) << field;
    const std::string twice = error(start + "interval = 1\n" + start + "interval = 1\n");
    EXPECT_NE(twice.find("names two entries"), std::string::npos) << twice;
}

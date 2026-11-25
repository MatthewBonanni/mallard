/**
 * @file particle_crossings_test.cpp
 * @author Matthew Bonanni (mbonanni001@gmail.com)
 * @brief [[particles.crossings]]: particles leaving through a boundary zone
 *        and crossing surfaces, where and when, once each, on any rank count
 *        and across restarts; [[particles.fluxes]]: their counts per interval.
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
#include <numbers>
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

/**
 * @brief Uniform flow u in the unit box (z: 0.5, periodic) of n^2 cells, 4 x 4 tracers in [lo, hi], periodic
 *        along the sides of periodic, the others given by boundaries.
 */
std::string box(const double u[3], const std::string & periodic, const std::string & boundaries, uint64_t n_steps,
                double lo, double hi, const std::string & extra, const std::string & init = "") {
    std::ostringstream s;
    s.precision(17);
    s << "[run]\nn_steps = " << n_steps << "\ndt = 0.01\n[mesh]\ntype = \"cartesian\"\nNx = 12\nNy = 12\nLx = 1.0\n"
      << "Ly = 1.0\n";
    if (N_DIM == 3) s << "Nz = 6\nLz = 0.5\n";
    s << periodic_sides(N_DIM == 3 ? periodic + "z" : periodic) << boundaries;
    if (init.empty()) {
        s << "[initialize]\ntype = \"analytical\"\nrho = \"1.0\"\nu = [\"" << u[0] << "\", \"" << u[1]
          << (N_DIM == 3 ? "\", \"" + std::to_string(u[2]) : std::string()) << "\"]\np = \"1.0\"\n";
    } else {
        s << "[initialize]\n" << init;
    }
    s << "[numerics]\nriemann_solver = \"HLLC\"\ntime_integrator = \"RK4\"\n[numerics.face_reconstruction]\n"
      << "type = \"TENO\"\norder = 4\n[physics]\ntype = \"euler\"\ngamma = 1.4\np_ref = 1.0\nT_ref = 1.0\n"
      << "rho_ref = 1.0\n[output]\ncheck_interval = 1000000\n[particles]\ntype = \"tracer\"\n[particles.lattice]\nlo = "
      << numbers(lo, lo, 0.25) << "\nhi = " << numbers(hi, hi, 0.25) << "\nn = " << (N_DIM == 3 ? "[4, 4, 1]" : "[4, 4]")
      << "\n" << extra;
    return s.str();
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

/** @brief The rows of a CSV file as numbers. */
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

/** @brief The seeds of box(lo, hi), by id. */
std::array<double, 2> seed(uint64_t id, double lo, double hi) {
    return {lo + (hi - lo) * static_cast<double>(id % 4) / 3.0, lo + (hi - lo) * static_cast<double>(id / 4 % 4) / 3.0};
}

/** @brief The columns of a CSV file by name. */
std::map<std::string, std::vector<double>> columns(const std::string & file) {
    std::ifstream in(file);
    std::string line;
    while (std::getline(in, line) && (line.empty() || line[0] == '#')) {
    }
    std::vector<std::string> names;
    std::stringstream header(line);
    for (std::string name; std::getline(header, name, ',');) names.push_back(name);
    std::map<std::string, std::vector<double>> out;
    for (const auto & r : rows(file)) {
        for (size_t k = 0; k < names.size() && k < r.size(); k++) out[names[k]].push_back(r[k]);
    }
    return out;
}

/** @brief An n x n lattice in place of box()'s 4 x 4. */
std::string with_lattice(std::string input, int n) {
    const std::string from = N_DIM == 3 ? "n = [4, 4, 1]" : "n = [4, 4]";
    const std::string to = "n = [" + std::to_string(n) + ", " + std::to_string(n) + (N_DIM == 3 ? ", 1]" : "]");
    return input.replace(input.find(from), from.size(), to);
}

// Supersonic: the inlet and outlet impose nothing the flow does not already have
const std::string CHANNEL = "[[boundaries]]\nname = \"left\"\ntype = \"inlet_state\"\nu = " + numbers(2.0, 0.0, 0.0) +
                            "\np = 1.0\nT = 1.0\n[[boundaries]]\nname = \"right\"\ntype = \"outlet_extrapolation\"\n";
const double CHANNEL_U[3] = {2.0, 0.0, 0.0};

} // namespace

TEST(ParticleCrossingsTest, ExitsAndCrossingsOfAUniformFlowAtTheirTimesAndPlaces) {
    // At u = 2 the column at x0 crosses x = 0.5 at t = (0.5 - x0) / 2 and leaves at x = 1 at t = (1 - x0) / 2;
    // in 25 steps of 0.01 the columns at 0.1 and 0.3667 cross and stay, those at 0.6333 and 0.9 leave
    const std::filesystem::path dir = scratch_dir() / "mallard_particle_crossings_uniform";
    for (const bool graphs : {true, false}) {
        const std::string out = (dir / (graphs ? "graphs" : "plain")).string();
        const std::string entries =
            "[[particles.quantities]]\nname = \"AGE\"\nintegral = \"1\"\n"
            "[[particles.crossings]]\nname = \"outlet\"\nregion = \"right\"\nfields = [\"U\", \"AGE\"]\nfile = \"" +
            out + "_outlet.csv\"\n[[particles.crossings]]\nname = \"middle\"\nregion = {type = \"boundary\", of = "
            "{type = \"half_space\", point = " + numbers(0.5, 0.0, 0.0) + ", normal = " + numbers(-1.0, 0.0, 0.0) +
            "}}\nfile = \"" + out + "_middle.csv\"\n";
        Solver solver;
        std::string input = box(CHANNEL_U, "y", CHANNEL, 25, 0.1, 0.9, entries);
        if (!graphs) input.replace(input.find("[run]\n"), 6, "[run]\ncuda_graphs = false\n");
        solver.init(parse_toml(input));
        solver.run();
        comm::barrier();
        const double tol = roundoff(1e-12);
        std::map<uint64_t, int> left;
        for (const auto & r : rows(out + "_outlet.csv")) {
            const uint64_t id = static_cast<uint64_t>(r[2]);
            const auto x0 = seed(id, 0.1, 0.9);
            left[id]++;
            EXPECT_NEAR(r[1], (1.0 - x0[0]) / 2.0, tol) << "particle " << id;
            EXPECT_NEAR(r[0], std::ceil(100.0 * (1.0 - x0[0]) / 2.0 - 1e-6), 1.0) << "step of particle " << id;
            EXPECT_NEAR(r[3], 1.0, tol);
            EXPECT_NEAR(r[4], x0[1], tol);
            EXPECT_NEAR(r[5 + N_DIM - 2], 2.0, tol) << "U_X";
            // The quantity at the end of the step it left in
            EXPECT_NEAR(r.back(), 0.01 * r[0], tol);
        }
        EXPECT_EQ(left.size(), 8u);
        for (const auto & [id, n] : left) {
            EXPECT_EQ(n, 1) << "particle " << id << " left " << n << " times";
            EXPECT_GE(id % 4, 2u) << "particle " << id;
        }
        std::map<uint64_t, int> crossed;
        for (const auto & r : rows(out + "_middle.csv")) {
            const uint64_t id = static_cast<uint64_t>(r[2]);
            crossed[id]++;
            EXPECT_NEAR(r[1], (0.5 - seed(id, 0.1, 0.9)[0]) / 2.0, tol) << "particle " << id;
            EXPECT_NEAR(r[3], 0.5, tol);
            EXPECT_EQ(r[3 + N_DIM], -1.0) << "into the half space x > 0.5";
        }
        EXPECT_EQ(crossed.size(), 8u);
        for (const auto & [id, n] : crossed) {
            EXPECT_EQ(n, 1);
            EXPECT_LE(id % 4, 1u) << "particle " << id;
        }
    }
    EXPECT_EQ(lines((dir / "graphs_outlet.csv").string()), lines((dir / "plain_outlet.csv").string()));
    EXPECT_EQ(lines((dir / "graphs_middle.csv").string()), lines((dir / "plain_middle.csv").string()));
}

TEST(ParticleCrossingsTest, ParticlesCrossACircleInThenOutAndAClippedPieceOfIt) {
    // Across the periodic box at u = (0.4, 0) for t = 2: a particle from (0, 0.5) enters the disk of radius 0.2
    // about (0.5, 0.5) at x = 0.3 and leaves it at 0.7; the rows of seeds at y = 0.35 and 0.65 cross the circle
    // at x = 0.5 -+ sqrt(0.2^2 - 0.15^2), those at y = 0.05 and 0.95 miss it
    const std::string file = (scratch_dir() / "mallard_particle_crossings_circle.csv").string();
    const double u[3] = {0.4, 0.0, 0.0};
    const std::string disk = N_DIM == 3 ? "{type = \"cylinder\", center = [0.5, 0.5, 0.25], axis = [0.0, 0.0, 1.0], "
                                          "radius = 0.2}"
                                        : "{type = \"disk\", center = [0.5, 0.5], radius = 0.2}";
    // The circle clipped to x < 0.6: its left arc
    const std::string entry = "[[particles.crossings]]\nname = \"circle\"\nregion = {type = \"boundary\", of = " +
                              disk + "}\nfile = \"" + file + "\"\n[[particles.crossings]]\nname = \"arc\"\nregion = "
                              "{type = \"intersection\", regions = [{type = \"boundary\", of = " + disk +
                              "}, {type = \"half_space\", point = " + numbers(0.6, 0.0, 0.0) + ", normal = " +
                              numbers(1.0, 0.0, 0.0) + "}]}\nfile = \"" + file + ".arc\"\n";
    Solver solver;
    std::string input = box(u, "xy", "", 200, 0.05, 0.95, entry);
    input.replace(input.find("type = \"tracer\"\n"), 16,
                  "type = \"tracer\"\npoints = [" + numbers(0.0, 0.5, 0.25) + "]\n");
    solver.init(parse_toml(input));
    solver.run();
    comm::barrier();
    const double tol = roundoff(1e-12);
    const double half_chord = std::sqrt(0.04 - 0.15 * 0.15);
    // Per row: the seeds at x = 0.05 and 0.35 cross in and out, 0.65 wraps around and enters, 0.95 enters and
    // leaves after wrapping
    std::map<uint64_t, std::vector<std::vector<double>>> by_id;
    const auto circle = rows(file);
    EXPECT_EQ(circle.size(), 2u + 2u * 7u);
    for (const auto & r : circle) by_id[static_cast<uint64_t>(r[2])].push_back(r);
    ASSERT_EQ(by_id[0].size(), 2u);
    EXPECT_NEAR(by_id[0][0][1], 0.3 / 0.4, tol);
    EXPECT_NEAR(by_id[0][0][3], 0.3, tol);
    EXPECT_EQ(by_id[0][0][3 + N_DIM], -1.0) << "into the disk";
    EXPECT_NEAR(by_id[0][1][1], 0.7 / 0.4, tol);
    EXPECT_NEAR(by_id[0][1][3], 0.7, tol);
    EXPECT_EQ(by_id[0][1][3 + N_DIM], 1.0) << "out of it";
    for (const auto & [id, list] : by_id) {
        if (id == 0) continue;
        const double y0 = 0.05 + 0.3 * static_cast<double>((id - 1) / 4 % 4);
        EXPECT_NEAR(std::abs(y0 - 0.5), 0.15, 1e-12) << "particle " << id;
        for (const auto & r : list) {
            EXPECT_NEAR(std::abs(r[3] - 0.5), half_chord, tol);
            EXPECT_EQ(r[3 + N_DIM], r[3] < 0.5 ? -1.0 : 1.0);
        }
    }
    // The left arc: only the entries into the disk, one per pass
    const auto arc = rows(file + ".arc");
    EXPECT_EQ(arc.size(), 1u + 2u * 4u);
    for (const auto & r : arc) {
        EXPECT_LT(r[3], 0.5) << "particle " << r[2];
        EXPECT_EQ(r[3 + N_DIM], -1.0);
    }
}

TEST(ParticleCrossingsTest, RecordsAreBitwiseTheSameOnAnyRankCount) {
    // Particles moving across the partitions record each crossing once, on the rank that held them
    if (comm::size() == 1) GTEST_SKIP() << "compares distributed runs with serial ones";
    const std::filesystem::path dir = scratch_dir() / "mallard_particle_crossings_ranks";
    const double u[3] = {0.7, -0.45, 0.2};
    auto input = [&](const std::string & name) {
        return box(u, "xy", "", 240, 0.07, 0.93,
                   "[[particles.crossings]]\nname = \"circle\"\nregion = {type = \"boundary\", of = " +
                       std::string(N_DIM == 3 ? "{type = \"sphere\", center = [0.5, 0.5, 0.25], radius = 0.3}"
                                              : "{type = \"disk\", center = [0.5, 0.5], radius = 0.3}") +
                       "}\nfields = [\"U\"]\nfile = \"" + (dir / name).string() + "\"\n");
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
    EXPECT_GE(a.size(), 30u);
    EXPECT_EQ(a, lines((dir / "serial.csv").string()));
}

TEST(ParticleCrossingsTest, ARestartAppendsTheRecordsOfTheContinuedRun) {
    const std::filesystem::path dir = scratch_dir() / "mallard_particle_crossings_restart";
    auto input = [&](const std::string & name, uint64_t n_steps, const std::string & init) {
        return box(CHANNEL_U, "y", CHANNEL, n_steps, 0.1, 0.9,
                   "[[particles.crossings]]\nname = \"outlet\"\nregion = \"right\"\nfile = \"" + (dir / name).string() +
                       "_outlet.csv\"\n[[particles.crossings]]\nname = \"middle\"\nregion = {type = \"boundary\", of = "
                       "{type = \"half_space\", point = " + numbers(0.5, 0.0, 0.0) + ", normal = " +
                       numbers(-1.0, 0.0, 0.0) + "}}\nfile = \"" + (dir / name).string() + "_middle.csv\"\n"
                       "[[write_data]]\nprefix = \"" + (dir / name).string() + "\"\nformat = \"restart\"\ninterval = 6\n",
                   init);
    };
    Solver straight;
    straight.init(parse_toml(input("straight", 25, "")));
    straight.run();
    Solver first;
    // The column at x = 0.3667 crosses x = 0.5 in step 7, the first after the restart
    first.init(parse_toml(input("continued", 6, "")));
    first.run();
    comm::barrier();
    Solver second;
    second.init(parse_toml(input("continued", 25, "type = \"restart\"\nfile = \"" + (dir / "continued_000006.restart").string() +
                                                       "\"\n")));
    second.run();
    comm::barrier();
    for (const char * kind : {"_outlet.csv", "_middle.csv"}) {
        const auto a = lines((dir / ("straight" + std::string(kind))).string());
        EXPECT_EQ(a.size(), 8u);
        EXPECT_EQ(lines((dir / ("continued" + std::string(kind))).string()), a) << kind;
    }
}

TEST(ParticleCrossingsTest, InputErrorsNameTheKey) {
    auto error = [](const std::string & entry) {
        try {
            Solver solver;
            solver.set_distributed(false);
            solver.init(parse_toml(box(CHANNEL_U, "y", CHANNEL, 1, 0.2, 0.8, entry)));
        } catch (const InputError & e) {
            return std::string(e.what());
        }
        return std::string();
    };
    const std::string file = "\nfile = \"" + (rank_scratch_dir() / "errors.csv").string() + "\"\n";
    const std::string wall = error("[[particles.crossings]]\nname = \"w\"\nregion = \"top\"" + file);
    // top is periodic here: a boundary zone no particle leaves through
    EXPECT_NE(wall.find("particles.crossings[0].region"), std::string::npos) << wall;
    const std::string volume = error("[[particles.crossings]]\nname = \"v\"\nregion = {type = \"half_space\", normal = " + numbers(1.0, 0.0, 0.0) + "}" + file);
    EXPECT_NE(volume.find("must be a boundary zone or a surface"), std::string::npos) << volume;
    const std::string field = error("[[particles.crossings]]\nname = \"f\"\nregion = \"right\"\nfields = [\"Q\"]" + file);
    EXPECT_NE(field.find("\"Q\" is no particle variable or quantity"), std::string::npos) << field;
    EXPECT_NE(field.find("particles.crossings[0].fields"), std::string::npos) << field;
    const std::string twice = error("[[particles.crossings]]\nname = \"a\"\nregion = \"right\"" + file +
                                    "[[particles.crossings]]\nname = \"a\"\nregion = \"left\"" + file);
    EXPECT_NE(twice.find("names two entries"), std::string::npos) << twice;
    const std::string interval = error("[[particles.fluxes]]\nname = \"i\"\nregion = \"right\"\ninterval = 0" + file);
    EXPECT_NE(interval.find("particles.fluxes[0].interval"), std::string::npos) << interval;
    const std::string flux_wall = error("[[particles.fluxes]]\nname = \"w\"\nregion = \"top\"" + file);
    EXPECT_NE(flux_wall.find("particles.fluxes[0].region"), std::string::npos) << flux_wall;
}

TEST(ParticleFluxesTest, TheNetFluxOutOfACircleIsWhatItsProbeLoses) {
    // Particles drifting across the periodic box enter and leave the circle: at every row, those inside at the
    // start less the integrated net flux out are the probe's count at the same step, with or without step graphs
    const std::filesystem::path dir = scratch_dir() / "mallard_particle_fluxes_closed";
    const double u[3] = {0.7, -0.45, 0.0};
    const double cx = 0.5, cy = 0.5, r = 0.3;
    const std::string disk = N_DIM == 3 ? "{type = \"cylinder\", center = [0.5, 0.5, 0.25], axis = [0.0, 0.0, 1.0], "
                                          "radius = 0.3}"
                                        : "{type = \"disk\", center = [0.5, 0.5], radius = 0.3}";
    const int n = 10;
    const double lo = 0.03, hi = 0.97;
    uint64_t inside0 = 0;
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            const double x = lo + (hi - lo) * i / (n - 1), y = lo + (hi - lo) * j / (n - 1);
            inside0 += std::hypot(x - cx, y - cy) < r ? 1u : 0u;
        }
    }
    std::vector<std::string> flux_rows[2];
    for (const bool graphs : {true, false}) {
        const std::string out = (dir / (graphs ? "graphs" : "plain")).string();
        std::string input = with_lattice(
            box(u, "xy", "", 120, lo, hi,
                "[[particles.fluxes]]\nname = \"circle\"\nregion = {type = \"boundary\", of = " + disk +
                    "}\ninterval = 8\nfile = \"" + out + "_fluxes.csv\"\n[[particles.probes]]\nname = \"inside\"\n"
                    "region = " + disk + "\ninterval = 8\nfile = \"" + out + "_probe.csv\"\n"),
            n);
        if (!graphs) input.replace(input.find("[run]\n"), 6, "[run]\ncuda_graphs = false\n");
        Solver solver;
        solver.init(parse_toml(input));
        solver.run();
        comm::barrier();
        auto fluxes = columns(out + "_fluxes.csv");
        auto probe = columns(out + "_probe.csv");
        const auto & count = probe.count("COUNT") ? probe["COUNT"] : probe["count"];
        ASSERT_EQ(fluxes["step"].size(), 15u);
        ASSERT_EQ(count.size(), 15u);
        double net = 0.0, in = 0.0, previous = 0.0;
        for (size_t k = 0; k < count.size(); k++) {
            EXPECT_EQ(fluxes["step"][k], 8.0 * static_cast<double>(k + 1));
            EXPECT_EQ(fluxes["t_start"][k], previous) << "row " << k;
            previous = fluxes["t"][k];
            EXPECT_EQ(fluxes["parcels_net"][k], fluxes["parcels_out"][k] - fluxes["parcels_in"][k]);
            net += fluxes["parcels_net"][k];
            in += fluxes["parcels_in"][k];
            EXPECT_EQ(static_cast<double>(inside0) - net, count[k]) << "step " << fluxes["step"][k];
        }
        EXPECT_GE(in, 10.0) << "particles enter as well as leave";
        flux_rows[graphs ? 0 : 1] = lines(out + "_fluxes.csv");
    }
    EXPECT_EQ(flux_rows[0], flux_rows[1]);
}

TEST(ParticleFluxesTest, ParcelsParticlesAndMassLeavingThroughAZoneByInterval) {
    // Solid particles moving with the gas at u = 2: the column at x0 leaves at x = 1 in the step holding
    // t = (1 - x0) / 2, so in 50 steps of 0.01 the 4 columns of 4 leave in steps 5, 19, 32 and 45
    const std::string file = (scratch_dir() / "mallard_particle_fluxes_outlet.csv").string();
    const double weight = 3.0, diameter = 0.01, density = 50.0;
    std::string input = box(CHANNEL_U, "y", CHANNEL, 50, 0.1, 0.9,
                            "[[particles.fluxes]]\nname = \"outlet\"\nregion = \"right\"\ninterval = 10\nfile = \"" +
                                file + "\"\n");
    std::ostringstream solid;
    solid.precision(17);
    solid << "type = \"solid\"\ndiameter = " << diameter << "\ndensity = " << density << "\nweight = " << weight
          << "\nvelocity = " << numbers(2.0, 0.0, 0.0) << "\n";
    input.replace(input.find("type = \"tracer\"\n"), 16, solid.str());
    input.replace(input.find("type = \"euler\"\n"), 15, "type = \"navier_stokes\"\nmu = 1e-4\n");
    Solver solver;
    solver.init(parse_toml(input));
    solver.run();
    comm::barrier();
    auto c = columns(file);
    const double mass = weight * density * std::numbers::pi / 6.0 * diameter * diameter * diameter;
    const std::vector<double> expected = {4.0, 4.0, 0.0, 4.0, 4.0};
    ASSERT_EQ(c["step"].size(), expected.size());
    for (size_t k = 0; k < expected.size(); k++) {
        EXPECT_EQ(c["parcels_out"][k], expected[k]) << "interval ending at step " << c["step"][k];
        EXPECT_EQ(c["parcels_in"][k], 0.0);
        EXPECT_EQ(c["particles_out"][k], weight * expected[k]);
        EXPECT_NEAR(c["mass_out"][k], mass * expected[k], roundoff(1e-14));
        EXPECT_NEAR(c["mass_rate"][k], mass * expected[k] / (c["t"][k] - c["t_start"][k]), roundoff(1e-10));
    }
}

TEST(ParticleFluxesTest, RowsAreBitwiseTheSameOnAnyRankCount) {
    if (comm::size() == 1) GTEST_SKIP() << "compares distributed runs with serial ones";
    const std::filesystem::path dir = scratch_dir() / "mallard_particle_fluxes_ranks";
    const double u[3] = {0.7, -0.45, 0.2};
    auto input = [&](const std::string & name) {
        return with_lattice(
            box(u, "xy", "", 240, 0.07, 0.93,
                "[[particles.fluxes]]\nname = \"circle\"\nregion = {type = \"boundary\", of = " +
                    std::string(N_DIM == 3 ? "{type = \"sphere\", center = [0.5, 0.5, 0.25], radius = 0.3}"
                                           : "{type = \"disk\", center = [0.5, 0.5], radius = 0.3}") +
                    "}\ninterval = 12\nfile = \"" + (dir / name).string() + "\"\n"),
            8);
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
    ASSERT_EQ(a.size(), 20u);
    double crossings = 0.0;
    for (const auto & r : rows((dir / "distributed.csv").string())) crossings += r[3] + r[4];
    EXPECT_GE(crossings, 40.0);
    EXPECT_EQ(a, lines((dir / "serial.csv").string()));
}

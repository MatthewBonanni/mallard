/**
 * @file boundary_types_test.cpp
 * @brief Boundary types and their options: the guide's support matrix is the
 *        parser's table, vectors given either way agree, and numbers given as
 *        expressions of x, y, z, t act as the conditions they stand for.
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 */

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <vector>

#include "boundary.h"
#include "data_paths.h"
#include "solver.h"
#include "test_fixtures.h"

namespace {

std::vector<std::string> cells_of_row(const std::string & line) {
    std::vector<std::string> cells;
    std::stringstream s(line);
    std::string cell;
    std::getline(s, cell, '|');
    while (std::getline(s, cell, '|')) {
        cell.erase(0, cell.find_first_not_of(' '));
        cell.erase(cell.find_last_not_of(' ') + 1);
        cells.push_back(cell);
    }
    if (!cells.empty() && cells.back().empty()) cells.pop_back();
    return cells;
}

/** @brief The rows of the first table after `heading` in the guide, by their first cell without backticks or links. */
std::map<std::string, std::vector<std::string>> guide_table(const std::string & heading,
                                                           std::vector<std::string> & header) {
    std::ifstream in(std::string(source_dir()) + "/docs/guide/boundary_conditions.md");
    std::string line;
    while (std::getline(in, line) && line != heading) {}
    std::map<std::string, std::vector<std::string>> rows;
    bool table = false;
    while (std::getline(in, line)) {
        if (line.rfind("|", 0) != 0) {
            if (table) break;
            continue;
        }
        std::vector<std::string> cells = cells_of_row(line);
        if (!table) {
            header = cells;
            table = true;
            continue;
        }
        if (cells[0].rfind("---", 0) == 0) continue;
        std::string name = cells[0];
        const size_t a = name.find('`'), b = name.find('`', a + 1);
        if (a != std::string::npos && b != std::string::npos) name = name.substr(a + 1, b - a - 1);
        rows[name] = cells;
    }
    return rows;
}

std::string vec(const std::string & x, const std::string & y, const std::string & z = "0.0") {
    return "[" + x + ", " + y + (N_DIM == 3 ? ", " + z : "") + "]";
}

/** @brief A 12 x 6 channel along x: entries for left, right, bottom, top (and back/front symmetry in 3D). */
std::string channel(const std::string & left, const std::string & right, const std::string & bottom,
                    const std::string & top, const std::string & physics = "euler", const std::string & run = "",
                    const std::string & init = "") {
    std::ostringstream s;
    s.precision(17);
    s << "[run]\n" << (run.empty() ? "n_steps = 12\ncfl = 0.5\n" : run)
      << "[mesh]\ntype = \"cartesian\"\nNx = 12\nNy = 6\nLx = 2.0\nLy = 1.0\n"
      << (N_DIM == 3 ? "Nz = 2\nLz = 0.3\nperiodic = [\"z\"]\n" : "")
      << "[initialize]\n"
      << (init.empty() ? "type = \"analytical\"\nrho = \"1.0 + 0.05 * sin(3 * x) * cos(2 * y)\"\nu = " +
                             vec("\"0.3\"", "\"0.05 * sin(2 * x)\"", "\"0.0\"") + "\np = \"1.0 + 0.02 * cos(x + y)\"\n"
                       : init)
      << "[[boundaries]]\nname = \"left\"\n" << left << "[[boundaries]]\nname = \"right\"\n" << right
      << "[[boundaries]]\nname = \"bottom\"\n" << bottom << "[[boundaries]]\nname = \"top\"\n" << top
      << "[numerics]\nriemann_solver = \"HLLC\"\ntime_integrator = \"SSPRK3\"\n"
      << "[numerics.face_reconstruction]\ntype = \"MUSCL\"\n"
      << "[physics]\ntype = \"" << physics << "\"\ngamma = 1.4\np_ref = 1.0\nT_ref = 1.0\nrho_ref = 1.0\n"
      << (physics == "navier_stokes" ? "mu = 0.01\nPr = 0.72\n" : "") << "[output]\ncheck_interval = 1000000\n";
    return s.str();
}

/** @brief The conservative state of every owned cell after the run. */
std::vector<double> final_state(const std::string & input) {
    Solver solver;
    solver.init(parse_toml(input));
    solver.run();
    solver.copy_device_to_host();
    std::vector<double> U;
    for (uint32_t c = 0; c < solver.get_mesh()->n_owned(); c++) {
        for (uint8_t i = 0; i < N_CONSERVATIVE; i++) U.push_back(double(solver.h_conservatives(c, i)));
    }
    return U;
}

double max_difference(const std::vector<double> & a, const std::vector<double> & b) {
    double d = 0.0;
    for (size_t i = 0; i < a.size(); i++) d = std::max(d, std::abs(a[i] - b[i]));
    return d;
}

const std::string WALL = "type = \"wall_adiabatic\"\n";
const std::string OUTLET = "type = \"outlet_pressure\"\np = 1.0\n";
const std::string STATE = "p = 1.0\nT = 1.0\n";

} // namespace

// The guide's support matrix and its table of types are the parser's: a
// subtype that gains or loses an option, or a required key, without the docs
// (or the docs without the parser) turns this red.
TEST(BoundaryTypesTest, GuideTablesAreTheParsersTables) {
    std::vector<std::string> header;
    const auto matrix = guide_table("### Support matrix", header);
    ASSERT_GE(header.size(), 7u);
    std::vector<std::string> available_header;
    const auto available = guide_table("## Available boundary conditions", available_header);
    for (const BoundarySubtype & s : boundary_subtypes()) {
        const std::string name = BOUNDARY_NAMES.at(s.type);
        ASSERT_TRUE(matrix.count(name)) << name << " has no row in the support matrix";
        const auto & row = matrix.at(name);
        for (size_t col = 1; col + 1 < header.size(); col++) {
            std::string option = header[col];
            option.erase(std::remove(option.begin(), option.end(), '`'), option.end());
            if (option == "u field") option = "u-field";
            const bool parser = std::find(s.options.begin(), s.options.end(), option) != s.options.end();
            EXPECT_EQ(row[col].rfind("✓", 0) == 0, parser) << name << ", " << option;
        }
        ASSERT_TRUE(available.count(name)) << name << " has no row in Available boundary conditions";
        std::string required;
        for (const std::string & r : s.required) {
            std::string text = "`" + r + "`";
            if (const size_t bar = r.find('|'); bar != std::string::npos) {
                text = "`" + r.substr(0, bar) + "` or `" + r.substr(bar + 1) + "`";
            }
            required += (required.empty() ? "" : ", ") + text;
        }
        EXPECT_EQ(available.at(name)[2], required.empty() ? "—" : required) << name;
    }
    EXPECT_EQ(matrix.size(), boundary_subtypes().size());
}

// A speed along a direction is the vector it stands for, and without a
// direction it is along each face's inward normal. Fails if the speed form
// misses the direction, takes the outward normal, or is applied to the
// condition before it is split by face.
TEST(BoundaryTypesTest, SpeedAlongADirectionIsTheVector) {
    const auto vector = final_state(channel("type = \"inlet_state\"\nu = " + vec("0.6", "0.8") + "\n" + STATE, OUTLET,
                                            WALL, WALL));
    const auto speed = final_state(channel("type = \"inlet_state\"\nu = 1.0\ndirection = " + vec("3.0", "4.0") + "\n" +
                                           STATE, OUTLET, WALL, WALL));
    EXPECT_LT(max_difference(vector, speed), roundoff(1e-14));
    const auto along_x = final_state(channel("type = \"inlet_state\"\nu = " + vec("0.4", "0.0") + "\n" + STATE, OUTLET,
                                             WALL, WALL));
    const auto normal = final_state(channel("type = \"inlet_state\"\nu = 0.4\n" + STATE, OUTLET, WALL, WALL));
    EXPECT_EQ(along_x, normal);
    EXPECT_GT(max_difference(vector, along_x), 1e-3);
    // With nscbc the speed sets the target velocity of each face
    const std::string nscbc = "nscbc = { L = 2.0 }\n";
    EXPECT_EQ(final_state(channel("type = \"inlet_state\"\nu = " + vec("0.4", "0.0") + "\n" + STATE + nscbc, OUTLET, WALL,
                                  WALL)),
              final_state(channel("type = \"inlet_state\"\nu = \"0.4 + 0 * t\"\n" + STATE + nscbc, OUTLET, WALL, WALL)));
}

TEST(BoundaryTypesTest, WallSpeedAlongADirectionIsTheVector) {
    const std::string ns = "navier_stokes";
    const auto vector =
        final_state(channel(WALL, WALL, WALL, "type = \"wall_isothermal\"\nT = 1.0\nu = " + vec("0.2", "0.0") + "\n", ns));
    const auto speed = final_state(channel(WALL, WALL, WALL, "type = \"wall_isothermal\"\nT = 1.0\nu = 0.2\ndirection = " +
                                                                 vec("1.0", "0.0") + "\n", ns));
    EXPECT_EQ(vector, speed);
    EXPECT_GT(max_difference(vector, final_state(channel(WALL, WALL, WALL, "type = \"wall_isothermal\"\nT = 1.0\n", ns))),
              1e-6);
}

// A mass flux vector m flows along m / |m| with m . n_in at each face: along
// the normal it is the scalar mass flux. Fails if the vector's normal
// component, its direction or its per-face split is lost.
TEST(BoundaryTypesTest, MassFluxVectorIsTheFluxAndItsDirection) {
    const auto scalar = final_state(channel("type = \"inlet_mass_flow\"\nmass_flux = 0.3\nT = 1.0\ndirection = " +
                                            vec("1.0", "0.25") + "\n", OUTLET, WALL, WALL));
    const auto vector = final_state(channel("type = \"inlet_mass_flow\"\nmass_flux = " + vec("0.3", "0.075") +
                                            "\nT = 1.0\n", OUTLET, WALL, WALL));
    EXPECT_LT(max_difference(scalar, vector), roundoff(1e-13));
    const auto timed = final_state(channel("type = \"inlet_mass_flow\"\nmass_flux = " + vec("\"0.3 + 0 * t\"", "0.075") +
                                           "\nT = 1.0\n", OUTLET, WALL, WALL));
    EXPECT_LT(max_difference(scalar, timed), roundoff(1e-13));
}

// A wall temperature, a heat flux, an outlet pressure and a free stream that
// vary along the boundary are the conditions of the faces split by `where`.
// Fails if one of these numbers is read only as a number again, or evaluated
// at the wrong face.
TEST(BoundaryTypesTest, ExpressionsOfXAreTheConditionsSplitByWhere) {
    const std::string ns = "navier_stokes";
    struct Case {
        std::string zone, expression, low, high, physics;
    };
    const std::vector<Case> cases = {
        {"top", "type = \"wall_isothermal\"\nT = \"x < 1 ? 1.0 : 1.3\"\n", "type = \"wall_isothermal\"\nT = 1.0\n",
         "type = \"wall_isothermal\"\nT = 1.3\n", ns},
        {"top", "type = \"wall_heat_flux\"\nq = \"x < 1 ? 0.0 : 0.05\"\n", "type = \"wall_heat_flux\"\nq = 0.0\n",
         "type = \"wall_heat_flux\"\nq = 0.05\n", ns},
        {"top", "type = \"farfield\"\nu = " + vec("0.3", "0.0") + "\np = 1.0\nT = \"x < 1 ? 1.0 : 1.2\"\n",
         "type = \"farfield\"\nu = " + vec("0.3", "0.0") + "\np = 1.0\nT = 1.0\n",
         "type = \"farfield\"\nu = " + vec("0.3", "0.0") + "\np = 1.0\nT = 1.2\n", "euler"},
    };
    for (const Case & c : cases) {
        const std::string split = c.low + "where = \"x < 1\"\n[[boundaries]]\nname = \"top\"\n" + c.high + "where = \"x >= 1\"\n";
        const auto expression = final_state(channel(WALL, OUTLET, WALL, c.expression, c.physics));
        const auto where = final_state(channel(WALL, OUTLET, WALL, split, c.physics));
        const auto uniform = final_state(channel(WALL, OUTLET, WALL, c.low, c.physics));
        EXPECT_EQ(expression, where) << c.expression;
        EXPECT_GT(max_difference(expression, uniform), 1e-8) << c.expression;
    }
    const std::string inflow = "type = \"inlet_state\"\nu = " + vec("0.3", "0.0") + "\n" + STATE;
    const auto expression = final_state(channel(inflow, "type = \"outlet_pressure\"\np = \"y < 0.5 ? 1.0 : 0.9\"\n", WALL, WALL));
    const auto where = final_state(channel(inflow, "type = \"outlet_pressure\"\np = 1.0\nwhere = \"y < 0.5\"\n"
                                                   "[[boundaries]]\nname = \"right\"\ntype = \"outlet_pressure\"\np = 0.9\n"
                                                   "where = \"y >= 0.5\"\n", WALL, WALL));
    EXPECT_EQ(expression, where);
}

// The target of an area-averaged outlet may follow t: on an outlet uniform
// across, the run follows a local outlet at the same p(t) (up to the face
// states' departure from the cells' mean, which the average does not see),
// far from one at a fixed p. Fails if the target of t is not evaluated (it
// would be zero) or only once.
TEST(BoundaryTypesTest, AveragePressureOfTFollowsTheTarget) {
    const std::string inflow = "type = \"inlet_state\"\nu = " + vec("0.3", "0.0") + "\np = 1.0\nT = 1.0\n";
    const std::string p = "p = \"1.0 + 0.5 * t\"\n";
    const std::string run = "n_steps = 30\ncfl = 0.5\n";
    const std::string init = "type = \"analytical\"\nrho = \"1.0 + 0.05 * sin(3 * x)\"\nu = " +
                             vec("\"0.3\"", "\"0.0\"", "\"0.0\"") + "\np = \"1.0\"\n";
    const auto averaged = final_state(channel(inflow, "type = \"outlet_pressure\"\naverage = true\n" + p,
                                              "type = \"symmetry\"\n", "type = \"symmetry\"\n", "euler", run, init));
    const auto local = final_state(channel(inflow, "type = \"outlet_pressure\"\n" + p, "type = \"symmetry\"\n",
                                           "type = \"symmetry\"\n", "euler", run, init));
    const auto fixed = final_state(channel(inflow, "type = \"outlet_pressure\"\naverage = true\np = 1.0\n",
                                           "type = \"symmetry\"\n", "type = \"symmetry\"\n", "euler", run, init));
    EXPECT_LT(max_difference(averaged, local), 0.15 * max_difference(averaged, fixed));
}

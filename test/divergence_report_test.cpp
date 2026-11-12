/**
 * @file divergence_report_test.cpp
 * @brief The divergence report: runs made to diverge in different ways each
 *        name their own cause first, with the evidence and the files.
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 */

#include <gtest/gtest.h>

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>

#include "comm.h"
#include "data_paths.h"
#include "diagnostic_code.h"
#include "divergence_report.h"
#include "gmsh_fixtures.h"
#include "json.h"
#include "provenance.h"
#include "solver.h"
#include "test_fixtures.h"
#include "units.h"

namespace {

namespace fs = std::filesystem;

std::string fresh_dir(const std::string & name) {
    if (comm::is_root()) {
        fs::remove_all(name);
        fs::create_directories(name);
    }
    comm::barrier();
    return name;
}

std::string velocity(const std::string & u) {
    std::string s = "u = [\"" + u + "\"";
    for (int d = 1; d < N_DIM; d++) s += ", \"0.0\"";
    return s + "]\n";
}

std::string physics() {
    return "[numerics.face_reconstruction]\ntype = \"FO\"\n"
           "[physics]\ntype = \"euler\"\ngamma = 1.4\np_ref = 1.0\nT_ref = 1.0\nrho_ref = 1.0\n"
           "[output]\ncheck_interval = 5\n";
}

/** @brief A periodic box with an acoustic pulse, at rest. */
std::string periodic_box(const std::string & run, const std::string & rho, const std::string & p) {
    std::ostringstream s;
    s << "[run]\nn_steps = 200\n" << run << "[mesh]\ntype = \"cartesian\"\nNx = 12\nNy = 12\n"
      << (N_DIM == 3 ? "Nz = 3\nLz = 0.25\n[[periodic]]\nzones = [\"back\", \"front\"]\n" : "")
      << "[[periodic]]\nzones = [\"left\", \"right\"]\n[[periodic]]\nzones = [\"bottom\", \"top\"]\n"
      << "[initialize]\ntype = \"analytical\"\nrho = \"" << rho << "\"\n" << velocity("0.0") << "p = \"" << p << "\"\n"
      << physics();
    return s.str();
}

const char * PULSE = "1.0 + 0.01 * exp(-50 * ((x - 0.5)^2 + (y - 0.5)^2))";

struct Outcome {
    std::string message;
    std::string code;
    json::Value report;  // divergence_report.json
    std::string text;           // divergence_report.txt
};

/** @brief Runs the input, which must diverge, with its report in dir. */
Outcome diverge(const std::string & dir, std::string input) {
    input += "divergence_report = \"" + dir + "/report\"\n[[write_data]]\nprefix = \"" + dir +
             "/restart\"\nformat = \"restart\"\ninterval = 1000\n";
    Solver solver;
    solver.init(parse_toml(input));
    Outcome o;
    try {
        solver.run();
    } catch (const diagnostic_code::Error & e) {
        o.message = e.what();
        o.code = e.code;
    }
    comm::barrier();
    std::ifstream json(dir + "/report.json");
    std::stringstream text;
    text << std::ifstream(dir + "/report.txt").rdbuf();
    o.text = text.str();
    const std::string line((std::istreambuf_iterator<char>(json)), {});
    if (!line.empty()) o.report = json::parse(line);
    return o;
}

/** @brief The evidence of a failure at a boundary of a reacting case, every part of the report filled. */
divergence::Evidence evidence() {
    using divergence::Cell;
    divergence::Evidence e;
    e.step = 120, e.t = 2.5e-4, e.step_start = 100, e.has_valid = true, e.step_valid = 119, e.t_valid = 2.48e-4;
    e.mixture = e.reacting = true;
    e.n_cells = 4096, e.n_invalid = 3;
    e.failed[divergence::TEMPERATURE] = 2, e.failed[divergence::PRESSURE] = 1;
    e.lo = {0.01, 0.02, 0.0}, e.hi = {0.015, 0.025, 0.0};
    const auto cell = [](uint64_t id, double x, int failed, double T) {
        Cell c;
        c.id = id, c.x = {x, 0.02, 0.0}, c.failed = failed;
        c.rho = 1.17, c.speed = 35.2, c.p = 101325.0, c.T = T, c.cfl = 0.92;
        c.quality = {12.0, 0.1, 1.5, 1.2};
        c.zone = 0, c.layers = failed >= 0 ? 0 : 1;
        c.chemistry_cost = failed >= 0 ? 480.0 : 6.0;
        return c;
    };
    e.focus = cell(812, 0.01, divergence::TEMPERATURE, 2850.0);
    e.focus.repaired = true;
    e.neighbors = {cell(811, 0.0095, -1, 2100.0), cell(813, 0.0105, divergence::PRESSURE, 2600.0)};
    e.integrator = "SSPRK3", e.limit = 1.2, e.cfl_input = 0.95, e.dt = 2.0e-8, e.dt_cfl1 = 2.1e-8, e.n_over_limit = 0;
    e.rho_mean = 1.05, e.p_mean = 101000.0;
    e.mesh_max = {35.0, 0.4, 4.0, 2.0};
    divergence::Zone inlet;
    inlet.name = "inlet", inlet.type = "inlet_mass_flow", inlet.near = 3, inlet.prescribed = true;
    inlet.ratio = 1.4, inlet.jump = 0.3, inlet.p_ghost = 101400.0, inlet.p_cell = 101325.0, inlet.rho_ghost = 1.2,
    inlet.rho_cell = 1.17;
    e.zones = {inlet};
    e.chemistry_cost_mean = 8.0, e.rejected_fraction = 0.25;
    e.history = {{100, 2.1e-4, 2.0e-8, 0.95, 1.1, 1.0e5, 300.0}, {110, 2.3e-4, 2.0e-8, 0.95, 1.09, 1.0e5, 298.0}};
    e.restart_file = "restart_000119.restart", e.snapshot_file = "divergence_report_neighborhood_000120.vtkhdf";
    return e;
}

/** @brief The report's text with its run id replaced, so that it is the same in every run. */
std::string report_text(const divergence::Evidence & e) {
    std::string text = divergence::text(e, divergence::rank_causes(e));
    const std::string id = provenance::run_id();
    const size_t at = text.find("Run " + id + ".\n");
    if (at != std::string::npos) text.replace(at + 4, id.size(), "RUN-ID");
    return text;
}

void expect_golden(const std::string & text, const std::string & name) {
    const fs::path golden = fs::path(source_dir()) / "test" / "data" / "run" / name;
    if (const char * update = std::getenv("MALLARD_UPDATE_GOLDEN"); update && std::string(update) == "1") {
        std::ofstream(golden) << text;
    }
    std::ifstream file(golden);
    ASSERT_TRUE(file) << golden;
    const std::string expected((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
    EXPECT_EQ(text, expected) << "the text differs from " << golden << "; MALLARD_UPDATE_GOLDEN=1 updates it";
}

std::string top_cause(const Outcome & o) {
    if (o.report.type != json::Value::Type::OBJECT || o.report["causes"].array.empty()) return "";
    return o.report["causes"].array[0]["kind"].string;
}

} // namespace

TEST(DivergenceReportTest, TooLargeACflNumberIsTheTimeStep) {
    const std::string dir = fresh_dir("divergence_cfl");
    const Outcome o = diverge(dir, periodic_box("cfl = 3.0\n", PULSE, PULSE));
    ASSERT_NE(o.message.find("diverged at step"), std::string::npos) << o.message;
    EXPECT_EQ(o.code, diagnostic_code::DIVERGED);
    EXPECT_NE(o.message.find("Most likely cause: time step too large"), std::string::npos) << o.message;
    EXPECT_NE(o.message.find("lower run.cfl to 1"), std::string::npos) << o.message;
    ASSERT_EQ(top_cause(o), "time_step") << o.text;
    const auto & step = o.report["time_step"];
    EXPECT_NEAR(step["cfl"].number, 3.0, 0.05);
    EXPECT_EQ(step["limit"].number, 1.2);
    EXPECT_GT(step["cells_over_limit"].number, 0.0);
    // The last valid state is the step before, kept as a restart and in the snapshot of the neighborhood
    const auto & valid = o.report["last_valid"];
    EXPECT_EQ(valid["step"].number + 1, o.report["step"].number);
    EXPECT_TRUE(fs::exists(valid["file"].string)) << valid["file"].string;
    EXPECT_TRUE(fs::exists(o.report["snapshot"].string)) << o.report["snapshot"].string;
    EXPECT_NE(o.text.find("--restart " + valid["file"].string), std::string::npos) << o.text;
    // On several ranks, every cell named with its rank
    if (comm::size() > 1) {
        EXPECT_NE(o.text.find(" on rank "), std::string::npos) << o.text;
    }
    EXPECT_EQ(o.report["cell"]["rank"].type, json::Value::Type::NUMBER);
}

TEST(DivergenceReportTest, AnInflowTheFlowCannotTakeIsTheBoundary) {
    const std::string dir = fresh_dir("divergence_inflow");
    std::ostringstream s;
    s << "[run]\nn_steps = 100\ncfl = 0.8\n[mesh]\ntype = \"cartesian\"\nNx = 24\nNy = 6\nLx = 4.0\nLy = 1.0\n"
      << (N_DIM == 3 ? "Nz = 1\nLz = 0.1667\n[[boundaries]]\nname = \"back\"\ntype = \"symmetry\"\n"
                       "[[boundaries]]\nname = \"front\"\ntype = \"symmetry\"\n" : "")
      << "[initialize]\ntype = \"analytical\"\nrho = \"1.0\"\n" << velocity("0.3") << "p = \"1.0\"\n"
      << "[[boundaries]]\nname = \"left\"\ntype = \"inlet_mass_flow\"\nmass_flux = 100.0\nT = 1.0\n"
      << "[[boundaries]]\nname = \"right\"\ntype = \"outlet_extrapolation\"\n"
      << "[[boundaries]]\nname = \"bottom\"\ntype = \"symmetry\"\n[[boundaries]]\nname = \"top\"\ntype = \"symmetry\"\n"
      << physics();
    const Outcome o = diverge(dir, s.str());
    ASSERT_EQ(top_cause(o), "boundary") << o.message << "\n" << o.text;
    EXPECT_NE(o.message.find("Most likely cause: boundary condition of left (inlet_mass_flow)"), std::string::npos)
        << o.message;
    for (const auto & zone : o.report["zones"].array) {
        if (zone["name"].string == "left") {
            EXPECT_GT(zone["ratio"].number, 10.0);
        }
    }
}

TEST(DivergenceReportTest, ANegativeInitialDensityIsTheInitialState) {
    const std::string dir = fresh_dir("divergence_initial");
    const Outcome o = diverge(dir, periodic_box("cfl = 0.8\n", "(x - 0.5)^2 + (y - 0.5)^2 < 0.02 ? -1.0 : 1.0", "1.0"));
    ASSERT_EQ(top_cause(o), "initial_state") << o.message << "\n" << o.text;
    EXPECT_NE(o.message.find("check [initialize]"), std::string::npos) << o.message;
    EXPECT_NE(o.text.find("the initial state already has a non-positive density or pressure"), std::string::npos)
        << o.text;
}

TEST(DivergenceReportTest, ASliverCellIsTheMesh) {
    if (N_DIM != 2) GTEST_SKIP() << "a 2D mesh file";
    // 16 x 16 quadrilaterals; the right side of cell (7, 8) moved almost onto its left side
    constexpr uint32_t n = 16;
    auto id = [&](uint32_t i, uint32_t j) { return std::to_string(j * (n + 1) + i + 1); };
    std::ostringstream m;
    m << "$MeshFormat\n2.2 0 8\n$EndMeshFormat\n$PhysicalNames\n4\n"
      << "1 1 \"bottom\"\n1 2 \"right\"\n1 3 \"top\"\n1 4 \"left\"\n$EndPhysicalNames\n$Nodes\n" << (n + 1) * (n + 1) << "\n";
    m.precision(17);
    for (uint32_t j = 0; j <= n; j++) {
        for (uint32_t i = 0; i <= n; i++) {
            const double x = (i - (i == 8 && (j == 8 || j == 9) ? 0.95 : 0.0)) / double(n);
            m << id(i, j) << " " << x << " " << double(j) / n << " 0\n";
        }
    }
    std::vector<std::string> elements;
    for (uint32_t i = 0; i < n; i++) {
        elements.push_back("1 2 1 1 " + id(i, 0) + " " + id(i + 1, 0));
        elements.push_back("1 2 3 3 " + id(i + 1, n) + " " + id(i, n));
        elements.push_back("1 2 4 4 " + id(0, i + 1) + " " + id(0, i));
        elements.push_back("1 2 2 2 " + id(n, i) + " " + id(n, i + 1));
    }
    for (uint32_t j = 0; j < n; j++) {
        for (uint32_t i = 0; i < n; i++) {
            elements.push_back("3 2 0 1 " + id(i, j) + " " + id(i + 1, j) + " " + id(i + 1, j + 1) + " " + id(i, j + 1));
        }
    }
    m << "$EndNodes\n$Elements\n" << elements.size() << "\n";
    for (size_t k = 0; k < elements.size(); k++) m << k + 1 << " " << elements[k] << "\n";
    m << "$EndElements\n";
    const std::string mesh = write_temp_shared("sliver.msh", m.str());

    const std::string dir = fresh_dir("divergence_sliver");
    std::ostringstream s;
    // A time step at CFL 0.4 for the regular cells
    s << "[run]\nn_steps = 300\ndt = 0.011\n[mesh]\nfilename = \"" << mesh << "\"\n"
      << "[initialize]\ntype = \"analytical\"\nrho = \"" << PULSE << "\"\n" << velocity("0.0") << "p = \"" << PULSE << "\"\n";
    for (const char * zone : {"left", "right", "bottom", "top"}) {
        s << "[[boundaries]]\nname = \"" << zone << "\"\ntype = \"symmetry\"\n";
    }
    s << physics();
    const Outcome o = diverge(dir, s.str());
    ASSERT_EQ(top_cause(o), "mesh") << o.message << "\n" << o.text;
    // The sliver is cell 7 + 16 * 8 = 135 (cells in file order)
    EXPECT_NE(o.message.find("improve cells"), std::string::npos) << o.message;
    EXPECT_NE(o.message.find("135"), std::string::npos) << o.message;
}

// Guards every section, table, column and unit of the report, and that a nondimensional case's has no units
TEST(DivergenceReportTest, TheReportMatchesTheGolden) {
    if (N_DIM != 2) GTEST_SKIP() << "the golden is of a 2D case (points with two coordinates)";
    std::string text = report_text(evidence());
    units::set_system(units::System::NONDIMENSIONAL);
    text += "\n" + report_text(evidence());
    units::set_system(units::System::SI);
    expect_golden(text, "divergence_report_2d.txt");
}

TEST(DivergenceReportTest, TheRecordGivesTheUnitsOfItsValues) {
    const divergence::Evidence e = evidence();
    const json::Value si = json::parse(divergence::record(e, divergence::rank_causes(e)).str());
    EXPECT_EQ(si["units"]["t"].string, "s");
    EXPECT_EQ(si["units"]["rho"].string, "kg m-3");
    EXPECT_EQ(si["units"]["x"].string, "m");
    units::set_system(units::System::NONDIMENSIONAL);
    const json::Value none = json::parse(divergence::record(e, divergence::rank_causes(e)).str());
    units::set_system(units::System::SI);
    for (const std::string & key : none["units"].keys()) EXPECT_EQ(none["units"][key].string, "1") << key;
    EXPECT_EQ(none["units"].keys(), si["units"].keys());
}

TEST(DivergenceReportTest, AHealthyRunWritesNoReport) {
    const std::string dir = fresh_dir("divergence_healthy");
    Solver solver;
    solver.init(parse_toml(periodic_box("cfl = 0.8\n", PULSE, PULSE) + "divergence_report = \"" + dir + "/report\"\n"));
    solver.run();
    comm::barrier();
    EXPECT_FALSE(fs::exists(dir + "/report.txt"));
    EXPECT_TRUE(fs::is_empty(dir));
}

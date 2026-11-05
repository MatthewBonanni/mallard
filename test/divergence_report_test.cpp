/**
 * @file divergence_report_test.cpp
 * @brief The divergence report: runs made to diverge in different ways each
 *        name their own cause first, with the evidence and the files.
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 */

#include <gtest/gtest.h>

#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>

#include "comm.h"
#include "gmsh_fixtures.h"
#include "json_reader.h"
#include "solver.h"
#include "test_fixtures.h"

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
    return "[physics]\ntype = \"euler\"\ngamma = 1.4\np_ref = 1.0\nT_ref = 1.0\nrho_ref = 1.0\n"
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
    json_reader::Value report;  // divergence_report.json
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
    } catch (const std::runtime_error & e) {
        o.message = e.what();
    }
    comm::barrier();
    std::ifstream json(dir + "/report.json");
    std::stringstream text;
    text << std::ifstream(dir + "/report.txt").rdbuf();
    o.text = text.str();
    const std::string line((std::istreambuf_iterator<char>(json)), {});
    if (!line.empty()) o.report = json_reader::parse(line);
    return o;
}

std::string top_cause(const Outcome & o) {
    if (o.report.type != json_reader::Value::Type::OBJECT || o.report["causes"].array.empty()) return "";
    return o.report["causes"].array[0]["kind"].string;
}

} // namespace

TEST(DivergenceReportTest, TooLargeACflNumberIsTheTimeStep) {
    const std::string dir = fresh_dir("divergence_cfl");
    const Outcome o = diverge(dir, periodic_box("cfl = 3.0\n", PULSE, PULSE));
    ASSERT_NE(o.message.find("diverged at step"), std::string::npos) << o.message;
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
    EXPECT_EQ(o.report["cell"]["rank"].type, json_reader::Value::Type::NUMBER);
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

TEST(DivergenceReportTest, AHealthyRunWritesNoReport) {
    const std::string dir = fresh_dir("divergence_healthy");
    Solver solver;
    solver.init(parse_toml(periodic_box("cfl = 0.8\n", PULSE, PULSE) + "divergence_report = \"" + dir + "/report\"\n"));
    solver.run();
    comm::barrier();
    EXPECT_FALSE(fs::exists(dir + "/report.txt"));
    EXPECT_TRUE(fs::is_empty(dir));
}

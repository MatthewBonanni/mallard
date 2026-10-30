/**
 * @file error_messages_test.cpp
 * @brief Errors a user is likely to hit say what is wrong, where (key and line),
 *        what to do and where the docs explain it, in a run and in mallard check.
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 */

#include <gtest/gtest.h>

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <initializer_list>
#include <sstream>
#include <string>

#include "check.h"
#include "data_paths.h"
#include "input.h"
#include "mixture.h"
#include "scratch.h"
#include "solver.h"

namespace {

/** @brief A box of 16 x 8 cells with symmetry boundaries (the zones given, or all four or six). */
std::string box(const std::string & run = "t_stop = 0.5\ncfl = 0.8\n", const std::string & mesh_extra = "",
                const std::vector<std::string> & zones = {}) {
    std::ostringstream s;
    s << "[run]\n" << run
      << "[mesh]\ntype = \"cartesian\"\nNx = 16\nNy = 8\nLx = 2.0\nLy = 1.0\n" << (N_DIM == 3 ? "Nz = 2\nLz = 0.25\n" : "")
      << mesh_extra
      << "[initialize]\nu = [1.0, 0.0" << (N_DIM == 3 ? ", 0.0" : "") << "]\np = 1.0\nT = 1.0\n"
      << "[numerics]\nriemann_solver = \"HLLC\"\n"
      << "[physics]\ntype = \"euler\"\ngamma = 1.4\np_ref = 1.0\nT_ref = 1.0\nrho_ref = 1.0\n";
    std::vector<std::string> names = zones;
    if (names.empty()) {
        names = {"left", "right", "bottom", "top"};
        if (N_DIM == 3) names.insert(names.end(), {"back", "front"});
    }
    for (const auto & name : names) s << "[[boundaries]]\nname = \"" << name << "\"\ntype = \"symmetry\"\n";
    return s.str();
}

std::string replaced(std::string text, const std::string & from, const std::string & to) {
    const size_t at = text.find(from);
    EXPECT_NE(at, std::string::npos) << from;
    if (at != std::string::npos) text.replace(at, from.size(), to);
    return text;
}

/** @brief The number of the first line of text that contains needle. */
std::ptrdiff_t line_number(const std::string & text, const std::string & needle) {
    return 1 + std::count(text.begin(), text.begin() + std::ptrdiff_t(text.find(needle)), '\n');
}

/** @brief "line N: " of the first line of text that contains needle. */
std::string line_prefix(const std::string & text, const std::string & needle) {
    return "line " + std::to_string(line_number(text, needle)) + ": ";
}

/** @brief The error of setting up a run of the input, or "" if there is none. */
std::string run_error(const std::string & input) {
    try {
        Solver solver;
        solver.init(parse_input_string(input));
    } catch (const std::exception & e) {
        return e.what();
    }
    return "";
}

void expect_parts(const std::string & text, std::initializer_list<std::string> parts) {
    ASSERT_FALSE(text.empty()) << "no error";
    for (const auto & part : parts) EXPECT_NE(text.find(part), std::string::npos) << "missing \"" << part << "\" in:\n" << text;
}

} // namespace

TEST(ErrorMessages, ValueOfTheWrongTypeIsAnErrorWithTheFix) {
    // toml::find_or would fall back to the default (100 cells) without a word
    const std::string input = replaced(box(), "Nx = 16", "Nx = 16.0");
    expect_parts(run_error(input), {line_prefix(input, "Nx = ") + "mesh.Nx", "16 (a float), but must be a non-negative integer",
                                    "Write it without a decimal point: 16", "See docs/input.md#mesh"});
    expect_parts(run_error(replaced(box(), "Nx = 16", "Nx = \"16\"")), {"mesh.Nx", "Remove the quotes: 16"});
    expect_parts(run_error(replaced(box(), "Nx = 16", "Nx = -4")), {"mesh.Nx: -4 is out of range"});
    expect_parts(run_error(box("t_stop = 0.5\ncfl = 0.8\nkeep_last_good = \"false\"\n")),
                 {"run.keep_last_good", "must be true or false", "Remove the quotes: false"});
}

TEST(ErrorMessages, ArrayOfTablesWrittenAsATableSaysToDoubleTheBrackets) {
    const Input input = parse_input_string("[boundaries]\nname = \"left\"\ntype = \"symmetry\"\n");
    try {
        find_value<std::vector<Input>>(input, "boundaries");
        FAIL() << "no error";
    } catch (const InputError & e) {
        expect_parts(e.what(), {"line 1: boundaries: is a table, but must be an array", "double brackets: [[boundaries]]"});
    }
}

TEST(ErrorMessages, UnacceptedValuesListTheAcceptedOnesAndTheClosest) {
    const std::string input = replaced(box(), "riemann_solver = \"HLLC\"", "riemann_solver = \"hllcc\"");
    expect_parts(run_error(input), {line_prefix(input, "riemann_solver") + "numerics.riemann_solver", "did you mean \"HLLC\"?",
                                    "Accepted values: ", "See docs/input.md#numerics"});
    // No close name: the ones that contain it
    expect_parts(run_error(replaced(box(), "type = \"symmetry\"", "type = \"wall\"")),
                 {"\"wall\" is not accepted (did you mean \"wall_adiabatic\", \"wall_heat_flux\" or \"wall_isothermal\"?)"});
}

TEST(ErrorMessages, BoundaryZonesNotInTheMeshListTheMeshsZones) {
    const std::string input = replaced(box(), "name = \"left\"", "name = \"inlet\"");
    expect_parts(run_error(input), {line_prefix(input, "\"inlet\"") + "boundaries[name = \"inlet\"].name",
                                    "the mesh has no boundary zone \"inlet\"", "left, right, top",
                                    "docs/guide/boundary_conditions.md"});
}

TEST(ErrorMessages, ZonesWithoutAConditionSayHowToAddOne) {
    std::vector<std::string> zones = {"left", "right", "bottom"};
    if (N_DIM == 3) zones.insert(zones.end(), {"back", "front"});
    expect_parts(run_error(box("t_stop = 0.5\ncfl = 0.8\n", "", zones)),
                 {"no condition for zone top", "Add a [[boundaries]] entry", "docs/guide/boundary_conditions.md"});
}

TEST(ErrorMessages, FacesWithTwoConditionsNameBothEntries) {
    std::vector<std::string> zones = {"left", "right", "bottom", "top", "left"};
    if (N_DIM == 3) zones.insert(zones.end(), {"back", "front"});
    const std::string input = box("t_stop = 0.5\ncfl = 0.8\n", "", zones);
    // The entry's [[boundaries]] header, the line before its name
    const std::string first = std::to_string(line_number(input, "name = \"left\"") - 1);
    expect_parts(run_error(input), {"already has a condition, from the entry on line " + first + " (name = \"left\")",
                                    "where = "});
}

TEST(ErrorMessages, MissingTablesAndKeysSayWhatToAdd) {
    const std::string no_run = replaced(box(), "[run]\nt_stop = 0.5\ncfl = 0.8\n", "");
    expect_parts(run_error(no_run), {"the [run] table is missing", "cfl = 0.5", "See docs/input.md#run"});
    expect_parts(run_error(box("cfl = 0.8\n")), {"nothing stops the run", "t_stop"});
    expect_parts(run_error(box("t_stop = 0.5\ncfl = 0.8\ndt = 0.1\n")), {"sets both dt and cfl"});
    expect_parts(run_error(box("t_stop = 0.5\ncfl = -0.8\n")), {"run.cfl: must be positive"});
    expect_parts(run_error(replaced(box(), "gamma = 1.4\n", "")), {"physics.gamma: missing", "A perfect gas needs gamma"});
}

TEST(ErrorMessages, MissingMeshFileNamesTheKeyAndWhereRelativePathsStart) {
    const std::string input = replaced(box(), "type = \"cartesian\"", "type = \"file\"\nfilename = \"no_such_mesh.msh\"");
    expect_parts(run_error(input), {line_prefix(input, "filename") + "mesh.filename", "no mesh file no_such_mesh.msh",
                                    std::filesystem::current_path().string(), "docs/guide/meshes.md"});
    expect_parts(run_error(replaced(box(), "type = \"cartesian\"\n", "")),
                 {"no mesh file mesh.msh", "mesh.msh is the default", "type = \"cartesian\""});
}

TEST(ErrorMessages, GeneratedMeshOfTheOtherDimensionNamesTheSolverToRun) {
    const std::string other = N_DIM == 2 ? "cartesian_tet" : "wedge";
    expect_parts(run_error(replaced(box(), "\"cartesian\"", "\"" + other + "\"")),
                 {"mesh.type", "is a " + std::to_string(5 - N_DIM) + "D mesh, but this is the " + std::to_string(N_DIM) +
                                   "D solver",
                  "-DMallard_DIM=" + std::to_string(5 - N_DIM)});
}

TEST(ErrorMessages, OutputThatCannotBeWrittenSaysWhatToCheck) {
    const std::filesystem::path blocker = scratch_dir() / "a_file_not_a_directory";
    std::ofstream(blocker) << "x";
    const std::string input = box() + "[[write_data]]\nprefix = \"" + (blocker / "flow").string() +
                              "\"\nformat = \"vtu\"\ninterval = 5\nvariables = [\"P\"]\n";
    expect_parts(run_error(input), {"cannot write the output " + (blocker / "flow").string(), "directory exists"});
}

TEST(ErrorMessages, InputFileProblemsSayWhatIsWrong) {
    try {
        parse_input((scratch_dir() / "no_such_input.toml").string());
        FAIL() << "no error";
    } catch (const std::exception & e) {
        expect_parts(e.what(), {"cannot open the input file", "no such file"});
    }
    try {
        parse_input_string("[run]\ncfl = 0.8\n[physics]\ntype = euler\n");
        FAIL() << "no error";
    } catch (const InputError & e) {
        expect_parts(e.what(), {"not valid TOML", "type = \"euler\"", "docs/guide/input_files.md"});
    }
}

TEST(ErrorMessages, UnknownSpeciesSuggestTheMechanismsName) {
    const std::string mechanism = source_dir() + "/mechanisms/h2o2.yaml";
    const MixtureModel model = MixtureModel::from_input(
        parse_input_string("[physics]\ngas = \"mixture\"\nmechanism = \"" + mechanism + "\"\n"));
    try {
        model.mass_fractions(parse_input_string("X = { o2 = 1.0, N2 = 3.76 }\n"), "initialize");
        FAIL() << "no error";
    } catch (const InputError & e) {
        expect_parts(e.what(), {"initialize.X.o2", "the mechanism has no species o2", "did you mean \"O2\"?",
                                "case-sensitive"});
    }
}

TEST(ErrorMessages, MechanismProblemsNameTheKeyThatReadsTheFile) {
    const std::string mechanism = source_dir() + "/mechanisms/h2o2.yaml";
    try {
        MixtureModel::from_input(parse_input_string("[physics]\ngas = \"mixture\"\nmechanism = \"" + mechanism +
                                                    "\"\nphase = \"air\"\n"));
        FAIL() << "no error";
    } catch (const InputError & e) {
        expect_parts(e.what(), {"line 4: physics.phase", "has no phase \"air\"", "Set phase to one of the file's phases",
                                "docs/guide/thermochemistry.md#gas-mixtures"});
    }
    try {
        MixtureModel::from_input(parse_input_string("[physics]\ngas = \"mixture\"\nmechanism = \"no_such.yaml\"\n"));
        FAIL() << "no error";
    } catch (const InputError & e) {
        expect_parts(e.what(), {"line 3: physics.mechanism", "cannot open the mechanism file no_such.yaml"});
    }
}

TEST(ErrorMessages, MassFlowInletTakesExactlyOneOfMdotAndMassFlux) {
    const std::string inlet = "type = \"inlet_mass_flow\"\nT = 1.0\n";
    expect_parts(run_error(replaced(box(), "type = \"symmetry\"", inlet)),
                 {"boundaries[name = \"left\"].mass_flux: missing", "needs one of `mass_flux` or `mdot`",
                  "Give exactly one", "docs/guide/boundary_conditions.md#inlet_mass_flow"});
    const std::string both = replaced(box(), "type = \"symmetry\"", inlet + "mdot = 1.0\nmass_flux = 1.0\n");
    expect_parts(run_error(both), {line_prefix(both, "mdot") + "boundaries[name = \"left\"].mdot", "given together"});
}

TEST(ErrorMessages, TextErrorsThatStartWithAKeyLinkItsDocs) {
    EXPECT_EQ(std::string(InputError("mesh.transform must be a table.").what()),
              "mesh.transform must be a table. See docs/input.md#mesh.");
    EXPECT_EQ(std::string(InputError("[[periodic]] angle must be nonzero.").what()),
              "[[periodic]] angle must be nonzero. See docs/input.md#periodic.");
    EXPECT_EQ(std::string(InputError("cannot read the file.").what()), "cannot read the file.");
}

TEST(ErrorMessages, CheckReportsInputErrorsAsARunDoes) {
    const std::string input = replaced(box(), "Nx = 16", "Nx = 16.0");
    CheckOptions options;
    options.input_file = "case.toml";
    Solver solver;
    const CheckReport report = solver.check(parse_input_string(input), options);
    ASSERT_EQ(report.errors.size(), 1u);
    EXPECT_EQ(report.errors[0], "case.toml: " + run_error(input));
}

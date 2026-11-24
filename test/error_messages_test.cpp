/**
 * @file error_messages_test.cpp
 * @brief Errors a user is likely to hit say what is wrong, where (key and line),
 *        what to do and where the docs explain it, in a run and in mallard check.
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 */

#include <gtest/gtest.h>

#include <algorithm>
#include <cctype>
#include <filesystem>
#include <fstream>
#include <functional>
#include <initializer_list>
#include <iterator>
#include <map>
#include <regex>
#include <set>
#include <sstream>
#include <string>
#include <vector>

#include "check.h"
#include "data_paths.h"
#include "gmsh_fixtures.h"
#include "input.h"
#include "input_upgrade.h"
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

TEST(ErrorMessages, EveryAccessorReportsAValueOfTheWrongTypeWithItsKeyAndLine) {
    using Read = std::function<void(const Input &)>;
    const auto first = [](const Input & in) { return find_value<std::vector<Input>>(in, "boundaries")[0]; };
    const std::string element = "x = 0\nboundaries = [1]\n";
    const std::string not_in_a_table = "is looked up in 1 (an integer), which is not a table";
    const std::string mesh_number = "x = 0\nmesh = 1\n";
    const std::string not_a_table = "line 2: mesh: is 1 (an integer), but must be a table";
    struct Case {
        std::string name, input;
        Read read;
        std::vector<std::string> parts;
    };
    const std::vector<Case> cases = {
        {"find_value in a non-table", element, [&](const Input & in) { find_value<std::string>(first(in), "name"); },
         {"line 2: name: " + not_in_a_table, "{ name = ... }"}},
        {"find_value_or in a non-table", element,
         [&](const Input & in) { find_value_or<std::string>(first(in), "name", ""); }, {"line 2: name: " + not_in_a_table}},
        {"find_real in a non-table", element, [&](const Input & in) { find_real(first(in), "p", dim::pressure); },
         {"line 2: p: " + not_in_a_table}},
        {"find_real_or in a non-table", element,
         [&](const Input & in) { find_real_or(first(in), "p", dim::pressure, 1.0_r); }, {"line 2: p: " + not_in_a_table}},
        {"find_double in a non-table", element, [&](const Input & in) { find_double(first(in), "T", dim::temperature); },
         {"line 2: T: " + not_in_a_table}},
        {"find_double_or in a non-table", element,
         [&](const Input & in) { find_double_or(first(in), "T", dim::temperature, 1.0); }, {"line 2: T: " + not_in_a_table}},
        {"find_real_vector in a non-table", element,
         [&](const Input & in) { find_real_vector(first(in), "axis", dim::none); }, {"line 2: axis: " + not_in_a_table}},
        {"find_value of a table that is a number", mesh_number,
         [](const Input & in) { find_value<int>(in, "mesh", "Nx"); }, {not_a_table, "mesh = { ... }"}},
        {"find_value_or of a table that is a number", mesh_number,
         [](const Input & in) { find_value_or<int>(in, "mesh", "Nx", 4); }, {not_a_table}},
        {"find_real of a table that is a number", mesh_number,
         [](const Input & in) { find_real(in, "mesh", "Lx", dim::length); }, {not_a_table}},
        {"find_real_or of an array of tables", "x = 0\n[[mesh]]\nLx = 1.0\n",
         [](const Input & in) { find_real_or(in, "mesh", "Lx", dim::length, 1.0_r); },
         {"mesh: is an array, but must be a table", "single brackets: [mesh]", "See docs/input.md#mesh"}},
        {"find_real_vector of a table that is a number", mesh_number,
         [](const Input & in) { find_real_vector(in, "mesh", "stretching", dim::none); }, {not_a_table}},
        {"find_real_vector of a number", "x = 0\ntranslation = 2.5\n",
         [](const Input & in) { find_real_vector(in, "translation", dim::length); },
         {"line 2: translation: is 2.5 (a float), but must be an array of numbers", "square brackets"}},
        {"find_real_vector in a table, of a number", "[mesh]\nstretching = 2.5\n",
         [](const Input & in) { find_real_vector(in, "mesh", "stretching", dim::none); },
         {"line 2: mesh.stretching: is 2.5 (a float), but must be an array of numbers"}},
        {"as_real_vector of a string", "x = 0\naxis = \"z\"\n",
         [](const Input & in) { as_real_vector(in.at("axis"), "axis", dim::none); },
         {"line 2: axis: is \"z\" (a string), but must be an array of numbers"}},
        {"find_real of an array", "x = 0\np = [1.0]\n", [](const Input & in) { find_real(in, "p", dim::pressure); },
         {"line 2: p: is an array, but must be a number, or a string of a number and a unit"}},
        {"find_double_or of a boolean", "x = 0\nT = true\n",
         [](const Input & in) { find_double_or(in, "T", dim::none, 1.0); }, {"line 2: T: is true, but must be a number."}},
        {"find_value of an integer out of range", "x = 0\nn = 5000000000\n",
         [](const Input & in) { find_value<int>(in, "n"); }, {"line 2: n: 5000000000 is out of range"}},
    };
    for (const Case & c : cases) {
        SCOPED_TRACE(c.name);
        const Input input = parse_input_string(c.input);
        try {
            c.read(input);
            ADD_FAILURE() << "no error";
        } catch (const InputError & e) {
            for (const auto & part : c.parts) EXPECT_NE(std::string(e.what()).find(part), std::string::npos) << e.what();
        } catch (const std::exception & e) {
            ADD_FAILURE() << "not an InputError: " << e.what();
        }
    }
}

TEST(ErrorMessages, ARemovedKeyWhoseValueCannotBeConvertedNamesTheKey) {
    for (const std::string value : {"\"two\"", "2.5", "true"}) {
        SCOPED_TRACE(value);
        Input input = parse_input_string("[numerics.face_reconstruction]\nbasis_order = " + value + "\n");
        try {
            input_upgrade::apply(input);
            ADD_FAILURE() << "no error";
        } catch (const InputError & e) {
            expect_parts(e.what(), {"line 2: numerics.face_reconstruction.basis_order: is " + value +
                                    ", but must be an integer"});
        } catch (const std::exception & e) {
            ADD_FAILURE() << "not an InputError: " << e.what();
        }
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
    // First order: the default preset would supply [run] cfl
    const std::string no_run =
        replaced(box(), "[run]\nt_stop = 0.5\ncfl = 0.8\n", "") + "[numerics.face_reconstruction]\ntype = \"FO\"\n";
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
    const std::filesystem::path blocker = write_temp("a_file_not_a_directory", "x");
    const std::string input = box() + "[[write_data]]\nprefix = \"" + (blocker / "flow").string() +
                              "\"\nformat = \"vtu\"\ninterval = 5\nfields = [\"P\"]\n";
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

// Each boundary type requires its keys and rejects those of other types, naming
// the type that takes them. Fails if a type silently ignores a key of another
// (a wall_isothermal with q runs as if q were not there) or infers its subtype
// from the keys present.
TEST(ErrorMessages, BoundaryKeysOfAnotherTypeNameTheTypeThatTakesThem) {
    const std::string q = replaced(box(), "type = \"symmetry\"", "type = \"wall_isothermal\"\nT = 300.0\nq = 1.0\n");
    expect_parts(run_error(q), {line_prefix(q, "q = 1.0") + "boundaries[name = \"left\"].q",
                                "`q` doesn't apply to `wall_isothermal`", "Use `wall_heat_flux`",
                                "docs/guide/boundary_conditions.md#wall_isothermal"});
    const std::string missing = replaced(box(), "type = \"symmetry\"", "type = \"wall_isothermal\"\n");
    expect_parts(run_error(missing), {"boundaries[name = \"left\"].T: missing; `wall_isothermal` needs `T`"});
    const std::string nscbc =
        replaced(box(), "type = \"symmetry\"", "type = \"inlet_stagnation\"\np0 = 2.0\nT0 = 1.2\nnscbc = { L = 1.0 }\n");
    expect_parts(run_error(nscbc), {"`nscbc` doesn't apply to `inlet_stagnation` (not supported yet: planned in "
                                    "https://github.com/MatthewBonanni/mallard/issues/565)",
                                    "Use `inlet_state` or `outlet_pressure`"});
}

// Options that do not compose (yet) are errors naming both, and a vector is
// given one way only. Fails if one of them is accepted and silently dropped.
TEST(ErrorMessages, BoundaryOptionsThatDontComposeNameBoth) {
    const std::string u = N_DIM == 3 ? "[1.0, 0.0, 0.0]" : "[1.0, 0.0]";
    auto error = [&](const std::string & left) { return run_error(replaced(box(), "type = \"symmetry\"", left)); };
    expect_parts(error("type = \"outlet_pressure\"\np = 1.0\naverage = true\nnscbc = { L = 1.0 }\n"),
                 {".average: `average` and `nscbc` don't combine on `outlet_pressure`", "issues/565", "Remove one"});
    expect_parts(error("type = \"inlet_state\"\nu = " + u + "\np = 1.0\nT = 1.0\n"
                       "turbulence = { reynolds_stress = " + std::string(N_DIM == 3 ? "[0.01, 0.01, 0.01, 0, 0, 0]" : "[0.01, 0.01, 0]") +
                       ", length_scale = 0.2 }\n"),
                 {".turbulence: `turbulence` needs `nscbc` on `inlet_state`", "Add nscbc = { L = ... }"});
    expect_parts(error("type = \"inlet_state\"\nu = " + u + "\ndirection = " + u + "\np = 1.0\nT = 1.0\n"),
                 {"u is a vector, so it has a direction already"});
    expect_parts(error("type = \"wall_adiabatic\"\nu = 1.0\n"),
                 {".u: is a speed, so the wall needs a direction", "Add direction = [...], or give u as a vector"});
    expect_parts(error("type = \"outlet_pressure\"\np = \"1 + 0.1 * y\"\naverage = true\n"),
                 {".p: with average = true is one area-averaged pressure: it may depend on t, not on x, y, z or the local coordinates"});
    expect_parts(error("type = \"outlet_pressure\"\np = 1.0\nnscbc = { L = 1.0, sigma_T = 1.0 }\n"),
                 {".nscbc.sigma_T: doesn't apply to an outlet's nscbc"});
}

TEST(ErrorMessages, InitialRegionsAndDataNameTheEntryAndTheColumns) {
    const std::string region = N_DIM == 2 ? "{type = \"disk\", center = [0.5, 0.5], radius = 0.2}"
                                          : "{type = \"sphere\", center = [0.5, 0.5, 0.1], radius = 0.2}";
    const std::string cells = box() + "[[initialize.regions]]\nT = 2.0\ncells = \"centre\"\nregion = " + region + "\n";
    expect_parts(run_error(cells), {line_prefix(cells, "cells =") + "initialize.regions[0].cells",
                                    "did you mean \"center\"?", "volume_fraction"});
    const std::string empty = box() + "[[initialize.regions]]\nregion = " + region + "\n";
    expect_parts(run_error(empty), {"initialize.regions[0]", "sets none of u, rho, p, T, X, Y", "T = 2000",
                                    "docs/guide/initialization.md#regions"});
    const std::string file = write_temp("profile.csv", "r,T\n0,2\n1,1\n");
    const std::string data = box() + "[[initialize.data]]\nname = \"flame\"\nfile = \"" + file +
                             "\"\ncoordinates = [\"radius\"]\n";
    expect_parts(run_error(data), {line_prefix(data, "coordinates =") + "initialize.data[0].coordinates",
                                   "has no column radius", "Accepted values: r, T",
                                   "docs/guide/initialization.md#tabulated-data-and-point-clouds"});
}

TEST(ErrorMessages, TextErrorsThatStartWithAKeyLinkItsDocs) {
    EXPECT_EQ(std::string(InputError("mesh.transform must be a table.").what()),
              "mesh.transform must be a table. See docs/input.md#mesh.");
    EXPECT_EQ(std::string(InputError("[[periodic]] angle must be nonzero.").what()),
              "[[periodic]] angle must be nonzero. See docs/input.md#periodic.");
    EXPECT_EQ(std::string(InputError("cannot read the file.").what()), "cannot read the file.");
}

TEST(ErrorMessages, ExpressionsThatDoNotCompileNameTheKeyLineAndGuide) {
    const std::string input =
        replaced(replaced(box(), "[initialize]\n", "[initialize]\ntype = \"analytical\"\n"), "T = 1.0", "T = \"(1 + x\"");
    expect_parts(run_error(input), {line_prefix(input, "T = \"") + "initialize.T: \"(1 + x\": expected ')'",
                                    "at position 7", "See docs/guide/expressions.md."});
    CheckOptions options;
    options.input_file = "case.toml";
    Solver solver;
    const CheckReport report = solver.check(parse_input_string(input), options);
    ASSERT_EQ(report.errors().size(), 1u);
    EXPECT_EQ(report.errors()[0], "case.toml: " + run_error(input));
    const std::string dirichlet =
        replaced(box(), "name = \"left\"\ntype = \"symmetry\"\n",
                 std::string("name = \"left\"\ntype = \"dirichlet\"\nrho = \"(1 + y\"\np = 1.0\nu = [1.0, 0.0") +
                     (N_DIM == 3 ? ", 0.0" : "") + "]\n");
    expect_parts(run_error(dirichlet), {line_prefix(dirichlet, "rho = \"") + "left.rho: \"(1 + y\": expected ')'"});
}

TEST(ErrorMessages, DataFileErrorsLinkTheDocsOfTheKeyThatReadsThem) {
    const std::filesystem::path previous = std::filesystem::current_path();
    std::filesystem::current_path(scratch_dir());
    std::ofstream("inflow.csv") << "r,T\n0,2\n1\n";
    const std::string error = run_error(box() + "[[initialize.data]]\nname = \"flame\"\nfile = \"inflow.csv\"\n");
    std::filesystem::current_path(previous);
    EXPECT_EQ(error, "inflow.csv:3: 1 value for 2 columns. See docs/guide/initialization.md#tabulated-data-and-point-clouds.");
}

TEST(ErrorMessages, UnusedKeysOfMallardReactorLinkItsDocs) {
    for (const std::string & section : {std::string("reactor"), std::string("benchmark")}) {
        const Input input = parse_input_string("[" + section + "]\nT = 1000.0\nTT = 1.0\n");
        find_value<double>(input, section, "T");
        const std::vector<Diagnostic> unused = unused_keys(input);
        ASSERT_EQ(unused.size(), 1u) << section;
        EXPECT_EQ(unused[0].docs, "docs/input.md#mallard-reactor") << section;
    }
}

namespace {

/** @brief The anchors of a Markdown page: its headings as GitHub links them, and its <a id="...">. */
std::set<std::string> anchors_of(const std::filesystem::path & page) {
    std::ifstream file(page);
    std::set<std::string> anchors;
    std::map<std::string, int> seen;
    const std::regex id("<a id=\"([^\"]+)\">");
    bool fenced = false;
    for (std::string line; std::getline(file, line);) {
        for (std::sregex_iterator it(line.begin(), line.end(), id), end; it != end; ++it) anchors.insert((*it)[1]);
        if (line.rfind("```", 0) == 0 || line.rfind("~~~", 0) == 0) fenced = !fenced;
        if (fenced || line.empty() || line[0] != '#') continue;
        const size_t text = line.find_first_not_of('#');
        if (text == std::string::npos || line[text] != ' ') continue;
        std::string slug;
        for (const char c : line.substr(text + 1)) {
            const auto u = static_cast<unsigned char>(c);
            if (std::isalnum(u) || c == '_' || c == '-') slug += static_cast<char>(std::tolower(u));
            else if (c == ' ') slug += '-';
        }
        const int n = seen[slug]++;
        anchors.insert(n == 0 ? slug : slug + "-" + std::to_string(n));
    }
    return anchors;
}

} // namespace

TEST(ErrorMessages, EveryDocsLinkTheCodeWritesExists) {
    const std::filesystem::path root = source_dir();
    if (!std::filesystem::exists(root / "src") || !std::filesystem::exists(root / "docs" / "input.md")) {
        GTEST_SKIP() << "no sources or docs in an install tree";
    }
    std::map<std::string, std::set<std::string>> anchors;
    const auto exists = [&](const std::string & link) {
        const size_t hash = link.find('#');
        const std::string page = link.substr(0, hash);
        if (!std::filesystem::exists(root / page)) return false;
        if (!anchors.count(page)) anchors[page] = anchors_of(root / page);
        return hash == std::string::npos || anchors[page].count(link.substr(hash + 1)) > 0;
    };
    std::set<std::string> sections = {"reactor", "benchmark"};
    for (const auto & entry : std::filesystem::directory_iterator(root / "docs" / "input-keys")) {
        const std::string name = entry.path().filename().string();
        sections.insert(name.substr(0, name.find_first_of(".")));
    }
    for (const auto & section : sections) {
        EXPECT_TRUE(exists(docs_of(section + ".key"))) << docs_of(section + ".key") << " (docs_of(\"" << section << ".key\"))";
    }
    const std::regex link("docs/[A-Za-z0-9_./-]+\\.md#[A-Za-z0-9_-]+");
    // The code's links, and the theory sections of the models the key catalog lists (its "models")
    for (const char * dir : {"src", "docs/input-keys"}) {
        for (const auto & entry : std::filesystem::recursive_directory_iterator(root / dir)) {
            const std::string path = entry.path().generic_string();
            if (!entry.is_regular_file() || path.find("/src/external/") != std::string::npos) continue;
            std::ifstream file(entry.path());
            const std::string text((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
            for (std::sregex_iterator it(text.begin(), text.end(), link), end; it != end; ++it) {
                EXPECT_TRUE(exists(it->str()))
                    << it->str() << " in " << std::filesystem::relative(entry.path(), root).string();
            }
        }
    }
}

TEST(ErrorMessages, CheckReportsInputErrorsAsARunDoes) {
    const std::string input = replaced(box(), "Nx = 16", "Nx = 16.0");
    CheckOptions options;
    options.input_file = "case.toml";
    Solver solver;
    const CheckReport report = solver.check(parse_input_string(input), options);
    ASSERT_EQ(report.errors().size(), 1u);
    EXPECT_EQ(report.errors()[0], "case.toml: " + run_error(input));
}

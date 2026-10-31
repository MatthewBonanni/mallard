/**
 * @file input_upgrade_test.cpp
 * @author Matthew Bonanni (mbonanni001@gmail.com)
 * @brief Deprecated input keys: warnings, errors once removed, and the
 *        rewrite of `mallard input upgrade`, which keeps comments and layout.
 * @version 0.9
 * @date 2026-10-16
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 *
 */

#include <gtest/gtest.h>

#include <string>
#include <vector>

#include <toml.hpp>

#include "input.h"
#include "input_upgrade.h"

namespace {

using input_upgrade::Finding;
using input_upgrade::Rules;

std::string twice(const std::string & text) { return std::to_string(2 * std::stoll(text)); }

// Deprecated in 0.9.0, removed in 0.10.0
const Rules RULES = {
    {
        {"run.old_cfl", "run.cfl", "0.9.0", "0.10.0", "renamed"},
        {"run.steps", "numerics.n_steps", "0.9.0", "0.10.0", "moved"},
        {"run.flag", "solver.flag", "0.9.0", "0.10.0", "moved to a new table"},
        {"boundaries[].u_in", "boundaries[].u", "0.9.0", "0.10.0", "renamed"},
        {"run.half", "run.full", "0.9.0", "0.10.0", "doubled", twice},
        {"run.gone", "", "0.9.0", "0.10.0", "no longer needed"},
    },
    {
        {"boundaries[].type", "outflow", "extrapolation", "0.9.0", "0.10.0", "renamed"},
    },
    {},
};

std::string upgrade(const std::string & text, bool & reformatted, std::vector<Finding> & findings) {
    return input_upgrade::upgrade_text(text, findings, reformatted, RULES);
}

} // namespace

TEST(InputUpgradeTest, RewritesOnlyTheLinesOfChangedKeys) {
    const std::string text =
        "# A case\n"
        "[run]\n"
        "old_cfl = 0.5   # CFL number\n"
        "steps = [\n"
        "    10,  # first\n"
        "]\n"
        "# comment of half\n"
        "half = 3 # doubled\n"
        "gone = true\n"
        "flag = 1\n"
        "\n"
        "[numerics]\n"
        "riemann_solver = \"HLLC\"  # keep me\n"
        "\n"
        "# the boundaries\n"
        "[[boundaries]]\n"
        "name = \"left\"\n"
        "type = \"outflow\"\n"
        "\n"
        "[[boundaries]]\n"
        "name = \"right\"\n"
        "u_in = [1.0, 0.0]  # inflow\n";
    const std::string expected =
        "# A case\n"
        "[run]\n"
        "cfl = 0.5   # CFL number\n"
        "# comment of half\n"
        "full = 6 # doubled\n"
        "# Removed in Mallard 0.10.0 (no longer needed):\n"
        "# gone = true\n"
        "\n"
        "[numerics]\n"
        "riemann_solver = \"HLLC\"  # keep me\n"
        "n_steps = [\n"
        "    10,  # first\n"
        "]\n"
        "\n"
        "# the boundaries\n"
        "[[boundaries]]\n"
        "name = \"left\"\n"
        "type = \"extrapolation\"\n"
        "\n"
        "[[boundaries]]\n"
        "name = \"right\"\n"
        "u = [1.0, 0.0]  # inflow\n"
        "\n"
        "[solver]\n"
        "flag = 1\n";
    bool reformatted = true;
    std::vector<Finding> findings;
    EXPECT_EQ(upgrade(text, reformatted, findings), expected);
    EXPECT_FALSE(reformatted);
    ASSERT_EQ(findings.size(), 7u);
    EXPECT_EQ(findings[0].key, "run.old_cfl");
    EXPECT_EQ(findings[0].line, 3u);
    EXPECT_EQ(findings[0].suggestion, "run.cfl");
    EXPECT_EQ(findings[3].key, "boundaries[1].u_in");
    EXPECT_EQ(findings[3].suggestion, "boundaries[1].u");
}

TEST(InputUpgradeTest, AnUpToDateFileComesBackUnchanged) {
    const std::string text = "[run]   # comment\ncfl=0.5\n\n[[boundaries]]\nname='a'\n";
    bool reformatted = true;
    std::vector<Finding> findings;
    EXPECT_EQ(upgrade(text, reformatted, findings), text);
    EXPECT_TRUE(findings.empty());
}

TEST(InputUpgradeTest, DottedAndInlineKeysAreReformattedWithTheSameMeaning) {
    const std::string text = "run.old_cfl = 0.5\nnumerics = { steps = 1 }\n[[boundaries]]\nname = 'a'\n";
    bool reformatted = false;
    std::vector<Finding> findings;
    const Input out = parse_input_string(upgrade(text, reformatted, findings));
    EXPECT_TRUE(reformatted);
    EXPECT_EQ(out.at("run").at("cfl").as_floating(), 0.5);
    EXPECT_FALSE(out.at("run").contains("old_cfl"));
}

TEST(InputUpgradeTest, OldAndNewNameTogetherAreAnError) {
    bool reformatted = false;
    std::vector<Finding> findings;
    EXPECT_THROW(upgrade("[run]\nold_cfl = 0.5\ncfl = 0.4\n", reformatted, findings), InputError);
}

TEST(InputUpgradeTest, DeprecatedKeysAreReadWithAWarningUntilTheyAreRemoved) {
    const std::string text = "[run]\nold_cfl = 0.5\nhalf = 2\n[[boundaries]]\ntype = 'outflow'\n";
    Input input = parse_input_string(text);
    const std::vector<Finding> findings = input_upgrade::apply(input, "0.9.3", RULES);
    ASSERT_EQ(findings.size(), 3u);
    EXPECT_FALSE(findings[0].removed);
    EXPECT_NE(input_upgrade::describe(findings[0]).find("line 2: run.old_cfl is deprecated since 0.9.0"),
              std::string::npos);
    // The setup reads the new keys, and the old ones are gone
    EXPECT_EQ(input.at("run").at("cfl").as_floating(), 0.5);
    EXPECT_EQ(input.at("run").at("full").as_integer(), 4);
    EXPECT_FALSE(input.at("run").contains("old_cfl"));
    EXPECT_EQ(input.at("boundaries").at(0).at("type").as_string(), "extrapolation");

    Input later = parse_input_string(text);
    try {
        input_upgrade::apply(later, "0.10.0", RULES);
        ADD_FAILURE() << "a removed key was accepted";
    } catch (const InputError & e) {
        const std::string what = e.what();
        EXPECT_NE(what.find("run.old_cfl was removed in 0.10.0: use run.cfl"), std::string::npos) << what;
        EXPECT_NE(what.find("mallard input upgrade"), std::string::npos) << what;
    }
}

TEST(InputUpgradeTest, KeysAreDeprecatedForAtLeastOneMinorReleaseBeforeRemoval) {
    auto minor = [](const std::string & v) { return std::stoi(v.substr(v.find('.') + 1)); };
    auto major = [](const std::string & v) { return std::stoi(v.substr(0, v.find('.'))); };
    auto check = [&](const std::string & what, const std::string & since, const std::string & removed) {
        // Removals before the policy (0.9.0) had no deprecation period
        if (major(since) == 0 && minor(since) < 9) return;
        EXPECT_TRUE(major(removed) > major(since) || minor(removed) > minor(since)) << what;
    };
    for (const auto & rule : input_upgrade::rules().keys) check(rule.from, rule.since, rule.removed);
    for (const auto & rule : input_upgrade::rules().options) check(rule.key, rule.since, rule.removed);
    for (const auto & rule : input_upgrade::rules().arrays) check(rule.from, rule.since, rule.removed);
}

TEST(InputUpgradeTest, KeptOptionsWorkAsBeforeButAreRewritten) {
    // format = "hdf5" still writes XDMF and HDF5 while deprecated; the
    // upgraded file asks for VTKHDF
    const std::string text = "[[write_data]]\nprefix = 'f'\nformat = \"hdf5\"  # ParaView\n";
    Input input = parse_input_string(text);
    const std::vector<Finding> findings = input_upgrade::apply(input, "0.9.0");
    ASSERT_EQ(findings.size(), 1u);
    EXPECT_FALSE(findings[0].removed);
    EXPECT_NE(findings[0].message.find("use \"vtkhdf\""), std::string::npos) << findings[0].message;
    EXPECT_EQ(input.at("write_data").at(0).at("format").as_string(), "hdf5");
    bool reformatted = true;
    std::vector<Finding> rewritten;
    EXPECT_EQ(input_upgrade::upgrade_text(text, rewritten, reformatted),
              "[[write_data]]\nprefix = 'f'\nformat = \"vtkhdf\"  # ParaView\n");
    EXPECT_FALSE(reformatted);
    Input later = parse_input_string(text);
    EXPECT_THROW(input_upgrade::apply(later, "0.10.0"), InputError);
}

// The boundary types renamed in 0.9.0 (#564): a characteristic boundary's
// parameters move into its nscbc table, written after the entry's own keys
// with their comments, p_out_average gains average = true, and the rest is
// untouched. Fails if a rule moves keys of another entry, drops a comment,
// or rewrites the file instead of its lines.
TEST(InputUpgradeTest, OldBoundaryTypesBecomeTypesWithOptions) {
    const std::string text =
        "[[boundaries]]\n"
        "name = \"left\"\n"
        "type = \"nscbc_inlet\"   # inflow\n"
        "u = [1.0, 0.0]\n"
        "p = 1.0\n"
        "T = 1.0\n"
        "L = 2.0\n"
        "sigma = 0.5  # stiffer\n"
        "X = { O2 = 1.0 }\n"
        "\n"
        "[boundaries.turbulence]\n"
        "length_scale = 0.1\n"
        "\n"
        "[[boundaries]]\n"
        "name = \"right\"\n"
        "type = \"p_out_average\"\n"
        "p = 1.0\n"
        "\n"
        "[[boundaries]]\n"
        "name = \"top\"\n"
        "type = \"upt\"\n";
    const std::string expected =
        "[[boundaries]]\n"
        "name = \"left\"\n"
        "type = \"inlet_state\"   # inflow\n"
        "u = [1.0, 0.0]\n"
        "p = 1.0\n"
        "T = 1.0\n"
        "X = { O2 = 1.0 }\n"
        "\n"
        "[boundaries.nscbc]\n"
        "L = 2.0\n"
        "sigma = 0.5  # stiffer\n"
        "\n"
        "[boundaries.turbulence]\n"
        "length_scale = 0.1\n"
        "\n"
        "[[boundaries]]\n"
        "name = \"right\"\n"
        "type = \"outlet_pressure\"\n"
        "average = true\n"
        "p = 1.0\n"
        "\n"
        "[[boundaries]]\n"
        "name = \"top\"\n"
        "type = \"inlet_state\"\n";
    bool reformatted = true;
    std::vector<Finding> findings;
    EXPECT_EQ(input_upgrade::upgrade_text(text, findings, reformatted), expected);
    EXPECT_FALSE(reformatted);
    EXPECT_EQ(findings.size(), 5u);
    // The setup reads the old form as the new one
    Input old_form = parse_input_string(text);
    input_upgrade::apply(old_form);
    const Input & left = old_form.at("boundaries").at(0);
    EXPECT_EQ(left.at("type").as_string(), "inlet_state");
    EXPECT_FALSE(left.contains("L"));
    EXPECT_EQ(left.at("nscbc").at("L").as_floating(), 2.0);
    EXPECT_EQ(left.at("nscbc").at("sigma").as_floating(), 0.5);
    EXPECT_TRUE(old_form.at("boundaries").at(1).at("average").as_boolean());
    // An nscbc type without its parameters still gets an nscbc table, so that the missing L is reported
    Input bare = parse_input_string("[[boundaries]]\nname = \"a\"\ntype = \"nscbc_outlet\"\np = 1.0\n");
    input_upgrade::apply(bare);
    EXPECT_TRUE(bare.at("boundaries").at(0).at("nscbc").is_table());
}

TEST(InputUpgradeTest, MeshPeriodicBecomesPeriodicZonePairsAfterTheMesh) {
    const std::string text =
        "[run]\n"
        "n_steps = 1\n"
        "\n"
        "[mesh]\n"
        "type = \"cartesian\"\n"
        "periodic = [\"x\", \"z\"]  # walls in y\n"
        "Lx = 2.0\n"
        "\n"
        "[output]\n"
        "check_interval = 10\n";
    const std::string expected =
        "[run]\n"
        "n_steps = 1\n"
        "\n"
        "[mesh]\n"
        "type = \"cartesian\"\n"
        "Lx = 2.0\n"
        "\n"
        "[[periodic]]\n"
        "zones = [\"left\", \"right\"]  # walls in y\n"
        "\n"
        "[[periodic]]\n"
        "zones = [\"back\", \"front\"]\n"
        "\n"
        "[output]\n"
        "check_interval = 10\n";
    bool reformatted = true;
    std::vector<Finding> findings;
    EXPECT_EQ(input_upgrade::upgrade_text(text, findings, reformatted), expected);
    EXPECT_FALSE(reformatted);
    ASSERT_EQ(findings.size(), 1u);
    EXPECT_EQ(findings[0].line, 6u);
    EXPECT_EQ(findings[0].suggestion,
              "[[periodic]] zones = [\"left\", \"right\"], [[periodic]] zones = [\"back\", \"front\"]");

    // The setup reads the pairs the rewritten file has
    Input input = parse_input_string(text);
    ASSERT_EQ(input_upgrade::apply(input, "0.9.0").size(), 1u);
    EXPECT_FALSE(input.at("mesh").contains("periodic"));
    ASSERT_EQ(input.at("periodic").size(), 2u);
    EXPECT_EQ(input.at("periodic").at(1).at("zones").at(0).as_string(), "back");
    Input later = parse_input_string(text);
    EXPECT_THROW(input_upgrade::apply(later, "0.10.0"), InputError);
}

TEST(InputUpgradeTest, MeshPeriodicFollowsTheExistingPairsAndTheRemapSourceGetsItsOwn) {
    const std::string text =
        "[mesh]\n"
        "type = \"annulus\"\n"
        "periodic = [\"theta\"]\n"
        "\n"
        "[[periodic]]\n"
        "zones = [\"inner\", \"outer\"]\n"
        "translation = [0.0, 1.0]\n"
        "\n"
        "[initialize]\n"
        "type = \"remap\"\n"
        "\n"
        "[initialize.mesh]\n"
        "type = \"cartesian\"\n"
        "periodic = [\n"
        "    \"y\",\n"
        "]\n";
    const std::string expected =
        "[mesh]\n"
        "type = \"annulus\"\n"
        "\n"
        "[[periodic]]\n"
        "zones = [\"inner\", \"outer\"]\n"
        "translation = [0.0, 1.0]\n"
        "\n"
        "[[periodic]]\n"
        "zones = [\"start\", \"end\"]\n"
        "\n"
        "[initialize]\n"
        "type = \"remap\"\n"
        "\n"
        "[initialize.mesh]\n"
        "type = \"cartesian\"\n"
        "\n"
        "[[initialize.periodic]]\n"
        "zones = [\"bottom\", \"top\"]\n";
    bool reformatted = true;
    std::vector<Finding> findings;
    EXPECT_EQ(input_upgrade::upgrade_text(text, findings, reformatted), expected);
    EXPECT_FALSE(reformatted);
    ASSERT_EQ(findings.size(), 2u);
    EXPECT_EQ(findings[1].key, "initialize.mesh.periodic");
    EXPECT_EQ(findings[1].suggestion, "[[initialize.periodic]] zones = [\"bottom\", \"top\"]");
}

TEST(InputUpgradeTest, MeshPeriodicOnlyTakesTheDirectionsOfAGeneratedMesh) {
    for (const auto & [text, error] : std::vector<std::pair<std::string, std::string>>{
             {"[mesh]\nfilename = \"a.msh\"\nperiodic = [\"x\"]\n", "line 3: mesh.periodic: applies to generated meshes"},
             {"[mesh]\ntype = \"cartesian\"\nperiodic = [\"theta\"]\n", "line 3: mesh.periodic: \"theta\" is not accepted"},
             {"[mesh]\ntype = \"annulus\"\nperiodic = [\"x\"]\n", "the annulus is periodic in \"theta\""}}) {
        Input input = parse_input_string(text);
        try {
            input_upgrade::apply(input, "0.9.0");
            ADD_FAILURE() << text;
        } catch (const InputError & e) {
            EXPECT_NE(std::string(e.what()).find(error), std::string::npos) << e.what();
        }
    }
}

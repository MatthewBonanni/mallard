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
}

TEST(InputUpgradeTest, RegionTypesAreRenamedInPlaceAtAnyDepthAndByDimension) {
    const std::string region = "[[sources]]\n"
                               "rhoE = 1\n"
                               "region = {type = \"union\", regions = [{type = \"circle\", center = [0, 0], radius = 1},\n"
                               "          {type = \"ellipse\", center = [0, 0], semi_axes = [1, 2]},\n"
                               "          {type = \"box\", center = [0, 0], size = [1, 1]},\n"
                               "          {type = \"annulus\", center = [0, 0], radius = 2, inner_radius = 1}]}\n";
    const std::string probe = "[[probes]]\nname = \"p\"\nshape = \"circle\"   # sampled\n";
    std::vector<Finding> findings;
    bool reformatted = false;
    const std::string out = input_upgrade::upgrade_text(region + probe, findings, reformatted);
    EXPECT_FALSE(reformatted);
    EXPECT_EQ(out, "[[sources]]\n"
                   "rhoE = 1\n"
                   "region = {type = \"union\", regions = [{type = \"" + std::string(N_DIM == 2 ? "disk" : "sphere") +
                       "\", center = [0, 0], radius = 1},\n"
                       "          {type = \"ellipsoid\", center = [0, 0], semi_axes = [1, 2]},\n"
                       "          {type = \"" + (N_DIM == 2 ? "rectangle" : "box") + "\", center = [0, 0], size = [1, 1]},\n"
                       "          {type = \"" + (N_DIM == 2 ? "annulus\"" : "cylinder\", axis = [0, 0, 1]") +
                       ", center = [0, 0], radius = 2, inner_radius = 1}]}\n"
                       "[[probes]]\nname = \"p\"\ntype = \"circle\"   # sampled\n");
    EXPECT_EQ(findings.size(), 4u);

    // Without the case's dimension, a rename that depends on it is an error
    EXPECT_THROW(input_upgrade::upgrade_text(region, findings, reformatted, input_upgrade::rules(), 0), InputError);
    EXPECT_NO_THROW(input_upgrade::upgrade_text(probe, findings, reformatted, input_upgrade::rules(), 0));
}

TEST(InputUpgradeTest, ARenamedOptionCanAddAKeyBesideIt) {
    const Rules rules = {{}, {{"s[].region.**.type", "annulus", "cylinder", "0.9.0", "0.10.0", "hollow", false,
                               0, "axis = [0, 0, 1]"}}};
    const std::string text = "[[s]]\n"
                             "region = {type = \"union\", regions = [{type = \"annulus\", radius = 2}]}\n"
                             "[[s]]\n"
                             "[s.region]\n"
                             "  type = \"annulus\"  # a ring\n"
                             "  radius = 1\n"
                             "[[s]]\n"
                             "region = {type = \"annulus\", axis = [1, 0, 0]}\n";
    std::vector<Finding> findings;
    bool reformatted = false;
    const std::string out = input_upgrade::upgrade_text(text, findings, reformatted, rules);
    EXPECT_FALSE(reformatted);
    EXPECT_EQ(out, "[[s]]\n"
                   "region = {type = \"union\", regions = [{type = \"cylinder\", axis = [0, 0, 1], radius = 2}]}\n"
                   "[[s]]\n"
                   "[s.region]\n"
                   "  type = \"cylinder\"  # a ring\n"
                   "  axis = [0, 0, 1]\n"
                   "  radius = 1\n"
                   "[[s]]\n"
                   "region = {type = \"cylinder\", axis = [1, 0, 0]}\n");

    Input input = parse_input_string(text);
    input_upgrade::apply(input, "0.9.0", rules);
    EXPECT_EQ(input.at("s").at(1).at("region").at("axis").as_array().size(), 3u);
    EXPECT_EQ(input.at("s").at(2).at("region").at("axis").at(0).as_integer(), 1);
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

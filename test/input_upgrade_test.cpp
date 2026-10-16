/**
 * @file input_upgrade_test.cpp
 * @author Matthew Bonanni (mbonanni001@gmail.com)
 * @brief Deprecated input keys: warnings, errors once removed, and the
 *        rewrite of `Mallard upgrade-input`, which keeps comments and layout.
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
    const toml::value out = toml::parse_str(upgrade(text, reformatted, findings));
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
    toml::value input = toml::parse_str(text);
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

    toml::value later = toml::parse_str(text);
    try {
        input_upgrade::apply(later, "0.10.0", RULES);
        ADD_FAILURE() << "a removed key was accepted";
    } catch (const InputError & e) {
        const std::string what = e.what();
        EXPECT_NE(what.find("run.old_cfl was removed in 0.10.0: use run.cfl"), std::string::npos) << what;
        EXPECT_NE(what.find("Mallard upgrade-input"), std::string::npos) << what;
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

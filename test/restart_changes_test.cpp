/**
 * @file restart_changes_test.cpp
 * @brief The classification of the keys a run from a restart file changes: what it accepts, resets or cannot
 *        take (restart_changes.h), and mallard check's report of it.
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 */

#include <gtest/gtest.h>

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <set>
#include <sstream>
#include <string>
#include <vector>

#include "check.h"
#include "data_paths.h"
#include "json.h"
#include "restart_changes.h"

using restart_changes::Change;
using restart_changes::Effect;

namespace {

const std::string BASE = "[run]\nt_stop = 1.0\ncfl = 0.5\n"
                         "[mesh]\ntype = \"box\"\nNx = 8\ncell_order = \"auto\"\n"
                         "[physics]\nunits = \"SI\"\ngamma = 1.4\n"
                         "[initialize]\ntype = \"constant\"\nrho = 1.0\n"
                         "[statistics]\nfields = [\"RHO\"]\ninterval = 1\n"
                         "[particles]\ntype = \"tracer\"\n"
                         "[[boundaries]]\nname = \"inlet\"\ntype = \"farfield\"\n"
                         "[[probes]]\nname = \"avg\"\naverage = true\npoint = [0.1, 0.1]\n"
                         "[[probes]]\nname = \"plain\"\npoint = [0.2, 0.2]\n"
                         "[[triggers]]\nname = \"stop\"\nwhen = \"t > 1\"\n";

/** @brief A file of the BASE input holding every kind of state. */
restart_changes::Restart stateful() {
    restart_changes::Restart r;
    r.file = "solut/restart_000010.restart";
    r.step = 10;
    r.t = 0.5;
    r.dim = N_DIM;
    r.real_size = sizeof(rtype);
    r.variables = {"RHO", "MEAN_RHO"};
    r.attributes = {"STATISTICS_WEIGHT"};
    r.arrays = {"PARTICLE_ID"};
    r.faces = 4;
    r.resolved = BASE;
    return r;
}

std::string replaced(std::string text, const std::string & from, const std::string & to) {
    const size_t at = text.find(from);
    EXPECT_NE(at, std::string::npos) << from;
    return at == std::string::npos ? text : text.replace(at, from.size(), to);
}

std::vector<Change> changes(const std::string & case_text, const restart_changes::Restart & r = stateful()) {
    return restart_changes::classify(r, parse_input_string(case_text), {});
}

const Change * find(const std::vector<Change> & all, const std::string & key) {
    for (const Change & c : all) {
        if (c.key == key) return &c;
    }
    return nullptr;
}

Effect effect(const std::vector<Change> & all, const std::string & key) {
    const Change * c = find(all, key);
    EXPECT_NE(c, nullptr) << key << " not reported";
    return c ? c->effect : Effect::ACCEPTED;
}

}  // namespace

TEST(RestartChanges, TheSameInputChangesNothing) { EXPECT_TRUE(changes(BASE).empty()); }

TEST(RestartChanges, RunAndOutputSettingsAreAccepted) {
    const auto all = changes(replaced(replaced(BASE, "t_stop = 1.0", "t_stop = 2.0"), "cfl = 0.5", "cfl = 0.8"));
    ASSERT_EQ(all.size(), 2u);
    const Change * stop = find(all, "run.t_stop");
    ASSERT_NE(stop, nullptr);
    EXPECT_EQ(stop->restart_value, "1.0");
    EXPECT_EQ(stop->case_value, "2.0");
    EXPECT_EQ(stop->effect, Effect::ACCEPTED);
}

TEST(RestartChanges, TheInitialStateARestartReplacesIsLeftOut) {
    EXPECT_TRUE(changes(replaced(BASE, "rho = 1.0", "rho = 2.0")).empty());
}

TEST(RestartChanges, KeysASetupThatStoppedDidNotGetToAreLeftOut) {
    const std::string partial = "[run]\nt_stop = 2.0\ncfl = 0.5\n[mesh]\ntype = \"box\"\nNx = 16\n";
    const auto all = restart_changes::classify(stateful(), parse_input_string(partial), {}, false);
    ASSERT_EQ(all.size(), 2u);
    EXPECT_EQ(effect(all, "run.t_stop"), Effect::ACCEPTED);
    EXPECT_EQ(effect(all, "mesh.Nx"), Effect::INCOMPATIBLE);
}

TEST(RestartChanges, AnotherMeshOrUnitsCannotStartFromTheFile) {
    EXPECT_EQ(effect(changes(replaced(BASE, "Nx = 8", "Nx = 16")), "mesh.Nx"), Effect::INCOMPATIBLE);
    EXPECT_EQ(effect(changes(replaced(BASE, "units = \"SI\"", "units = \"nondimensional\"")), "physics.units"),
              Effect::INCOMPATIBLE);
    // The cell order is not the file's: it holds cells in the mesh file's order
    EXPECT_EQ(effect(changes(replaced(BASE, "cell_order = \"auto\"", "cell_order = \"input\"")), "mesh.cell_order"),
              Effect::ACCEPTED);
}

TEST(RestartChanges, OtherAveragesNeedAReset) {
    const std::string other = replaced(BASE, "fields = [\"RHO\"]", "fields = [\"RHO\", \"P\"]");
    const auto all = changes(other);
    const Change * c = find(all, "statistics.fields");
    ASSERT_NE(c, nullptr);
    EXPECT_EQ(c->effect, Effect::INCOMPATIBLE);
    EXPECT_NE(c->detail.find("statistics.reset = true"), std::string::npos);
    // With the reset, every averaged quantity starts afresh
    const auto reset = changes(replaced(other, "interval = 1\n", "interval = 1\nreset = true\n"));
    EXPECT_EQ(effect(reset, "statistics.fields"), Effect::RESETS);
    EXPECT_EQ(effect(reset, "statistics.reset"), Effect::RESETS);
    // A file without averages has nothing to continue
    restart_changes::Restart none = stateful();
    none.variables = {"RHO"};
    none.attributes = {};
    EXPECT_EQ(effect(changes(other, none), "statistics.fields"), Effect::ACCEPTED);
}

TEST(RestartChanges, AResetSetInBothInputsIsReported) {
    restart_changes::Restart r = stateful();
    r.resolved = replaced(BASE, "interval = 1\n", "interval = 1\nreset = true\n");
    const auto all = changes(r.resolved, r);
    ASSERT_EQ(all.size(), 1u);
    EXPECT_EQ(all[0].key, "statistics.reset");
    EXPECT_EQ(all[0].effect, Effect::RESETS);
}

TEST(RestartChanges, ParticlesProbesBoundariesAndTriggersResetTheirState) {
    EXPECT_EQ(effect(changes(replaced(BASE, "type = \"tracer\"", "type = \"solid\"")), "particles.type"),
              Effect::RESETS);
    EXPECT_EQ(effect(changes(replaced(BASE, "point = [0.1, 0.1]", "point = [0.3, 0.1]")), "probes[name = \"avg\"].point"),
              Effect::RESETS);
    EXPECT_EQ(effect(changes(replaced(BASE, "point = [0.2, 0.2]", "point = [0.3, 0.2]")),
                     "probes[name = \"plain\"].point"),
              Effect::ACCEPTED);
    EXPECT_EQ(effect(changes(replaced(BASE, "type = \"farfield\"", "type = \"outflow\"")),
                     "boundaries[name = \"inlet\"].type"),
              Effect::RESETS);
    EXPECT_EQ(effect(changes(replaced(BASE, "name = \"stop\"", "name = \"end\"")), "triggers[name = \"end\"].name"),
              Effect::RESETS);
}

TEST(RestartChanges, OtherVariablesOrBuildsCannotStartFromTheFile) {
    restart_changes::Restart r = stateful();
    r.variables = {"RHO", "RHOY_H2"};
    const auto all = restart_changes::classify(r, parse_input_string(BASE), {"RHO", "RHOY_O2"});
    const Change * v = find(all, "variables");
    ASSERT_NE(v, nullptr);
    EXPECT_EQ(v->effect, Effect::INCOMPATIBLE);
    EXPECT_NE(v->detail.find("RHOY_O2"), std::string::npos);
    EXPECT_NE(v->detail.find("RHOY_H2"), std::string::npos);
    r = stateful();
    r.dim = N_DIM == 2 ? 3 : 2;
    r.real_size = sizeof(rtype) == 8 ? 4 : 8;
    const auto build = changes(BASE, r);
    EXPECT_EQ(effect(build, "build.dimension"), Effect::INCOMPATIBLE);
    EXPECT_EQ(effect(build, "build.precision"), Effect::INCOMPATIBLE);
}

namespace {

/** @brief Every key of the catalog (docs/input-keys) and its type. */
std::map<std::string, std::string> catalog() {
    std::map<std::string, std::string> keys;
    for (const auto & entry : std::filesystem::directory_iterator(std::filesystem::path(source_dir()) / "docs" /
                                                                  "input-keys")) {
        if (entry.path().extension() != ".json") continue;
        const std::string stem = entry.path().stem().string();
        std::string key;
        for (size_t i = 0; i < stem.size(); i++) {
            if (stem[i] == '%' && i + 2 < stem.size()) {
                key += static_cast<char>(std::stoi(stem.substr(i + 1, 2), nullptr, 16));
                i += 2;
            } else {
                key += stem[i];
            }
        }
        std::ifstream in(entry.path());
        std::stringstream text;
        text << in.rdbuf();
        const json::Value v = json::parse(text.str());
        keys[key] = v.find("type") ? v.find("type")->string : "";
    }
    return keys;
}

bool under(const std::string & key, const std::string & table) {
    return key == table || key.starts_with(table + ".");
}

}  // namespace

// Ground truth, from what Solver::apply_restart and the restore functions do with the file: a change of any of these
// keys never goes on as if nothing happened. A key added to these tables later is stateful until it is listed here.
TEST(RestartChanges, NoKeyThatCarriesStateIsAccepted) {
    const std::map<std::string, std::string> keys = catalog();
    ASSERT_GT(keys.size(), 300u);
    const std::set<std::string> stateless = {"mesh.cell_order", "mesh.curved", "mesh.curved_geometry",
                                             "statistics.t_start", "statistics.interval", "statistics.confidence"};
    auto stateful_key = [&](const std::string & k) {
        if (stateless.contains(k) || under(k, "statistics.convergence")) return false;
        return under(k, "mesh") || under(k, "statistics") || under(k, "particles") || k == "physics.units" ||
               k == "boundaries.type" || under(k, "probes") || k == "triggers.name";
    };
    size_t checked = 0;
    for (const auto & [key, type] : keys) {
        if (!stateful_key(key) || key.find('<') != std::string::npos) continue;
        const bool table = std::any_of(keys.begin(), keys.end(), [&](const auto & k) { return k.first.starts_with(key + "."); });
        if (table) continue;
        // The innermost array of tables holding the key, as one entry
        std::string array, leaf = key;
        for (size_t dot = key.find('.'); dot != std::string::npos; dot = key.find('.', dot + 1)) {
            const auto t = keys.find(key.substr(0, dot));
            if (t != keys.end() && t->second == "array") array = key.substr(0, dot);
        }
        if (!array.empty()) leaf = key.substr(array.size() + 1);
        if (leaf == "name") continue;  // an entry's name: its entry is another, of keys of their own
        const std::string value = type == "bool" ? "true" : type == "string" ? "\"x\"" : type == "array" ? "[\"x\"]" : "1";
        std::string entry = array.empty() ? "" : "[[" + array + "]]\nname = \"e\"\n";
        if (array == "probes" && leaf != "average") entry += "average = true\n";
        const std::string top = array.empty() ? key + " = " + value + "\n" : "";
        restart_changes::Restart r = stateful();
        r.resolved = "[run]\nt_stop = 1.0\n" + entry;
        const std::string now = top + r.resolved + (array.empty() ? "" : leaf + " = " + value + "\n");
        const auto all = restart_changes::classify(r, parse_input_string(now), {});
        ASSERT_EQ(all.size(), 1u) << key;
        EXPECT_NE(all[0].effect, Effect::ACCEPTED) << key << " carries state: " << all[0].detail;
        checked++;
    }
    EXPECT_GT(checked, 40u);
}

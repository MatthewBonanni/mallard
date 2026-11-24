/**
 * @file gas_models_test.cpp
 * @brief The gas models of mallard list models (gas_models.h): their keys against the key catalog, the model an
 *        input chooses, switching between them, and the solver refusing what a model removes.
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 */

#include <gtest/gtest.h>

#include <filesystem>
#include <fstream>
#include <map>
#include <set>
#include <sstream>
#include <string>

#include "check.h"
#include "data_paths.h"
#include "gas_models.h"
#include "json.h"
#include "solver/solver.h"

namespace {

std::filesystem::path catalog_file(const std::string & key) {
    return std::filesystem::path(source_dir()) / "docs" / "input-keys" / (key + ".json");
}

json::Value catalog_entry(const std::string & key) {
    std::ifstream in(catalog_file(key));
    std::stringstream text;
    text << in.rdbuf();
    return json::parse(text.str());
}

std::string model_of_file(const std::string & relative) {
    return gas_models::of(parse_input((std::filesystem::path(source_dir()) / relative).string()));
}

// An input as its tables' keys and values (TOML), for switching between models as a front end would
using Doc = std::map<std::string, std::map<std::string, std::string>>;

std::string text_of(const Doc & doc) {
    std::string text;
    for (const auto & [table, keys] : doc) {
        text += "[" + table + "]\n";
        for (const auto & [key, value] : keys) text += key + " = " + value + "\n";
    }
    return text;
}

/** @brief doc with the model's keys set, those it requires given and those it removes taken out. */
Doc switched(Doc doc, const gas_models::GasModel & m) {
    const auto split = [](const std::string & key) {
        const size_t dot = key.find('.');
        return std::pair{key.substr(0, dot), dot == std::string::npos ? std::string() : key.substr(dot + 1)};
    };
    for (const std::string & key : m.removes) {
        const auto [table, leaf] = split(key);
        if (leaf.empty()) {
            doc.erase(table);
        } else if (doc.contains(table)) {
            doc[table].erase(leaf);
        }
    }
    for (const auto & [key, value] : m.sets) {
        const auto [table, leaf] = split(key);
        doc[table][leaf] = value;
    }
    for (const std::string & key : m.requires_keys) {
        const auto [table, leaf] = split(key);
        if (!doc[table].contains(leaf)) doc[table][leaf] = "\"file\"";
    }
    return doc;
}

}  // namespace

TEST(GasModels, EveryGasTheSolverTakesIsAModelsAndEveryKeyIsInTheCatalog) {
    std::set<std::string> gases;
    const json::Value gas = catalog_entry("physics.gas");
    for (const json::Value & v : gas["values"].array) gases.insert(v.string);
    std::set<std::string> set_by_models;
    for (const gas_models::GasModel & m : gas_models::all()) {
        for (const auto & [key, value] : m.sets) {
            EXPECT_TRUE(std::filesystem::exists(catalog_file(key))) << m.name << ": " << key;
            if (key == "physics.gas") set_by_models.insert(json::parse(value).string);
        }
        for (const std::string & key : m.requires_keys) {
            ASSERT_TRUE(std::filesystem::exists(catalog_file(key))) << m.name << ": " << key;
            const json::Value entry = catalog_entry(key);
            const json::Value * required = entry.find("required");
            EXPECT_TRUE(required && required->boolean) << m.name << ": " << key << " has a default";
        }
        for (const std::string & key : m.removes) {
            EXPECT_TRUE(std::filesystem::exists(catalog_file(key))) << m.name << ": " << key;
        }
    }
    EXPECT_EQ(set_by_models, gases);
}

TEST(GasModels, TheExamplesChooseTheirModels) {
    EXPECT_EQ(model_of_file("examples/cylinder/input.toml"), "perfect");
    EXPECT_EQ(model_of_file("examples/shock_bubble_3d/input.toml"), "frozen");
    EXPECT_EQ(model_of_file("examples/flame_2d/input.toml"), "finite_rate");
    EXPECT_EQ(model_of_file("tools/dev/catalog_inputs/2d/flamelet.toml"), "flamelet");
    EXPECT_EQ(gas_models::of(parse_input_string("[physics]\ngas = \"mixture\"\n[chemistry]\nenabled = false\n")),
              "frozen");
    EXPECT_EQ(gas_models::of(parse_input_string("[physics]\ngas = \"ideal\"\n")), "");
}

TEST(GasModels, SwitchingAnInputGivesTheModel) {
    const Doc start = {{"physics", {{"gas", "\"mixture\""}, {"mechanism", "\"h2o2.yaml\""}, {"units", "\"SI\""}}},
                       {"chemistry", {{"rtol", "1e-6"}}},
                       {"radiation", {{"model", "\"optically_thin\""}}}};
    for (const gas_models::GasModel & from : gas_models::all()) {
        const Doc before = switched(start, from);
        ASSERT_EQ(gas_models::of(parse_input_string(text_of(before))), from.name) << text_of(before);
        for (const gas_models::GasModel & to : gas_models::all()) {
            const std::string after = text_of(switched(before, to));
            EXPECT_EQ(gas_models::of(parse_input_string(after)), to.name) << from.name << " -> " << to.name << "\n"
                                                                          << after;
        }
    }
}

// What a model removes, the solver refuses alongside it
TEST(GasModels, APerfectGasRefusesTheTablesItRemoves) {
    std::string perfect = "[run]\nt_stop = 0.5\ncfl = 0.8\n[mesh]\ntype = \"cartesian\"\nNx = 16\nNy = 8\nLx = 2.0\n"
                          "Ly = 1.0\n";
    perfect += N_DIM == 3 ? "Nz = 2\nLz = 0.25\n" : "";
    perfect += "[initialize]\ntype = \"analytical\"\nrho = \"1.0\"\np = \"1.0\"\nu = [\"1.0\", \"0.0\"";
    perfect += N_DIM == 3 ? ", \"0.0\"]\n" : "]\n";
    perfect += "[numerics]\nriemann_solver = \"HLLC\"\ntime_integrator = \"SSPRK3\"\n"
               "[numerics.face_reconstruction]\ntype = \"FO\"\n"
               "[physics]\ntype = \"euler\"\ngamma = 1.4\np_ref = 1.0\nT_ref = 1.0\nrho_ref = 1.0\n";
    for (const char * name : {"left", "right", "bottom", "top", "back", "front"}) {
        if (N_DIM == 2 && (std::string(name) == "back" || std::string(name) == "front")) continue;
        perfect += std::string("[[boundaries]]\nname = \"") + name + "\"\ntype = \"symmetry\"\n";
    }
    Solver ok;
    EXPECT_TRUE(ok.check(parse_input_string(perfect), {}).setup_done);
    for (const std::string & table : {"chemistry", "radiation"}) {
        Solver solver;
        const CheckReport r = solver.check(parse_input_string(perfect + "[" + table + "]\n"), {});
        EXPECT_FALSE(r.errors().empty()) << table;
    }
}

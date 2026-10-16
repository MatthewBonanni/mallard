/**
 * @file numerics_preset_test.cpp
 * @author Matthew Bonanni (mbonanni001@gmail.com)
 * @brief Resolution of `[numerics] preset` against the keys an input sets.
 * @version 0.1
 * @date 2026-10-16
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 *
 */

#include <gtest/gtest.h>

#include <algorithm>
#include <cctype>
#include <filesystem>
#include <map>
#include <string>
#include <vector>

#include "example_inputs.h"
#include "input.h"
#include "numerics_preset.h"
#include "solver.h"
#include "test_fixtures.h"

namespace {

using Keys = std::map<std::string, std::string>;

Keys resolved(const numerics_preset::Resolution & r, bool from_preset) {
    Keys keys;
    for (const auto & s : r.settings) {
        if (s.from_preset == from_preset) keys[s.key] = s.value;
    }
    return keys;
}

numerics_preset::Resolution resolve(const std::string & text, toml::value * out = nullptr) {
    toml::value input = parse_toml(text);
    auto r = numerics_preset::apply(input);
    if (out) *out = input;
    return r;
}

} // namespace

// The table of docs/guide/numerics.md#presets
TEST(NumericsPreset, EachPresetResolvesToItsDocumentedKeys) {
    const std::map<std::string, Keys> documented = {
        {"robust", {{"numerics.face_reconstruction.type", "MUSCL"},
                    {"numerics.face_reconstruction.limiter", "venkatakrishnan"},
                    {"numerics.riemann_solver", "RHLL"},
                    {"numerics.convective_flux", "riemann"},
                    {"numerics.time_integrator", "SSPRK3"},
                    {"run.cfl", "0.5"}}},
        {"accurate", {{"numerics.face_reconstruction.type", "TENO"},
                      {"numerics.face_reconstruction.order", "5"},
                      {"numerics.riemann_solver", "HLLC"},
                      {"numerics.convective_flux", "riemann"},
                      {"numerics.time_integrator", "SSPRK3"},
                      {"run.cfl", "1"}}},
        {"les", {{"numerics.face_reconstruction.type", "MUSCL"},
                 {"numerics.face_reconstruction.limiter", "none"},
                 {"numerics.riemann_solver", "HLLC"},
                 {"numerics.convective_flux", "hybrid"},
                 {"numerics.time_integrator", "SSPRK3"},
                 {"run.cfl", "0.8"}}},
        {"reacting", {{"numerics.face_reconstruction.type", "MUSCL"},
                      {"numerics.face_reconstruction.limiter", "venkatakrishnan"},
                      {"numerics.riemann_solver", "HLLC"},
                      {"numerics.convective_flux", "riemann"},
                      {"numerics.time_integrator", "SSPRK3"},
                      {"run.cfl", "0.35"}}},
    };
    EXPECT_EQ(numerics_preset::names(), (std::vector<std::string>{"robust", "accurate", "les", "reacting"}));
    for (const auto & name : numerics_preset::names()) {
        SCOPED_TRACE(name);
        toml::value input;
        const auto r = resolve("[run]\nt_stop = 1.0\n[numerics]\npreset = \"" + name + "\"\n", &input);
        EXPECT_EQ(r.name, name);
        EXPECT_EQ(resolved(r, true), documented.at(name));
        EXPECT_TRUE(resolved(r, false).empty());
        EXPECT_EQ(r.summary(), name);
        // The keys land in the input the rest of the setup reads
        const toml::value & recon = input.at("numerics").at("face_reconstruction");
        EXPECT_EQ(recon.at("type").as_string(), documented.at(name).at("numerics.face_reconstruction.type"));
        EXPECT_EQ(input.at("numerics").at("riemann_solver").as_string(),
                  documented.at(name).at("numerics.riemann_solver"));
        EXPECT_TRUE(input.at("run").at("cfl").is_floating());
    }
}

TEST(NumericsPreset, ExplicitKeysOverrideThePreset) {
    toml::value input;
    const auto r = resolve("[run]\nt_stop = 1.0\ndt = 1e-3\n"
                         "[numerics]\npreset = \"accurate\"\nriemann_solver = \"RHLL\"\n"
                         "[numerics.face_reconstruction]\norder = 3\n", &input);
    EXPECT_EQ(resolved(r, false), (Keys{{"numerics.riemann_solver", "RHLL"},
                                        {"numerics.face_reconstruction.order", "3"},
                                        {"run.dt", "0.001"}}));
    EXPECT_EQ(resolved(r, true), (Keys{{"numerics.face_reconstruction.type", "TENO"},
                                       {"numerics.convective_flux", "riemann"},
                                       {"numerics.time_integrator", "SSPRK3"}}));
    EXPECT_EQ(input.at("numerics").at("riemann_solver").as_string(), "RHLL");
    EXPECT_EQ(input.at("numerics").at("face_reconstruction").at("order").as_integer(), 3);
    EXPECT_FALSE(input.at("run").contains("cfl"));
    EXPECT_EQ(r.summary(), "accurate (the input sets numerics.face_reconstruction.order = 3, "
                           "numerics.riemann_solver = RHLL, run.dt = 0.001)");
}

TEST(NumericsPreset, AnotherReconstructionTypeGetsNoneOfThePresetsReconstructionOptions) {
    toml::value input;
    const auto r = resolve("[run]\nt_stop = 1.0\n[numerics]\npreset = \"accurate\"\n"
                         "[numerics.face_reconstruction]\ntype = \"MUSCL\"\n", &input);
    EXPECT_FALSE(input.at("numerics").at("face_reconstruction").contains("order"));
    EXPECT_EQ(resolved(r, false), (Keys{{"numerics.face_reconstruction.type", "MUSCL"}}));
    EXPECT_EQ(resolved(r, true).count("numerics.face_reconstruction.order"), 0u);
}

TEST(NumericsPreset, WithoutAPresetTheInputIsUnchanged) {
    const std::string text = "[run]\nt_stop = 1.0\ncfl = 0.7\n[numerics]\nriemann_solver = \"Roe\"\n";
    toml::value input;
    const auto r = resolve(text, &input);
    EXPECT_TRUE(r.name.empty());
    EXPECT_TRUE(r.settings.empty());
    EXPECT_EQ(input, parse_toml(text));
}

TEST(NumericsPreset, InvalidPresetsAreInputErrors) {
    try {
        resolve("[numerics]\npreset = \"fast\"\n");
        FAIL() << "no error for an unknown preset";
    } catch (const InputError & e) {
        EXPECT_EQ(std::string(e.what()),
                  "numerics.preset = \"fast\" is not one of: accurate, les, reacting, robust.");
    }
    EXPECT_THROW(resolve("[numerics]\npreset = 1\n"), InputError);
}

namespace {

const std::string DETONATION_INIT =
    "type = \"constant\"\np = 6670.0\nT = 298.0\nX = { H2 = 2.0, O2 = 1.0, AR = 7.0 }\n";
const std::string H2_AIR_INIT = "type = \"constant\"\np = 101325.0\nT = 300.0\nu = [0.0, 0.0, 0.0]\n"
                                "X = { H2 = 0.2959, O2 = 0.1479, N2 = 0.5562 }\n";

/** @brief Every solver example, with the preset closest to its numerics and purpose. */
const std::vector<ExampleCase> EXAMPLES = {
    {"autoignition_2d/input.toml", "reacting", 2,
     "type = \"constant\"\np = 4.154e6\nT = 1070.0\nu = [0.0, 0.0]\nX = { H2 = 0.04, O2 = 0.2, N2 = 0.76 }\n", {}, ""},
    {"cylinder/input.toml", "accurate", 2, "",
     {{"inlet", "farfield"}, {"outlet", "farfield"}, {"walls", "farfield"}, {"cylinder", "cylinder"}}, ""},
    {"detonation_1d/input.toml", "reacting", 2, DETONATION_INIT + "u = [0.0, 0.0]\n", {}, ""},
    {"detonation_2d/input.toml", "reacting", 2, DETONATION_INIT + "u = [0.0, 0.0]\n", {}, ""},
    {"detonation_3d/input_2d.toml", "reacting", 2, DETONATION_INIT + "u = [0.0, 0.0]\n", {}, ""},
    {"double_mach/input.toml", "robust", 2, "", {}, ""},
    {"flame_2d/input.toml", "reacting", 2, "", {}, ""},
    {"flame_2d/unity/input.toml", "reacting", 2, "", {}, ""},
    {"flame_outflow/input.toml", "reacting", 2, "", {}, ""},
    {"flame_vortex/input.toml", "reacting", 2, "", {}, ""},
    {"flame_vortex_quenching/input.toml", "reacting", 2, "", {}, ""},
    {"mallard/input.toml", "robust", 2, "",
     {{"inlet", "inflow"}, {"outlet", "outflow"}, {"walls", "top"}, {"cylinder", "duck"}}, ""},
    {"noh_axisymmetric/input.toml", "robust", 2, "", {}, ""},
    {"poiseuille_pipe/input.toml", "accurate", 2, "", {}, ""},
    {"premixed_flame/input.toml", "reacting", 2, "", {}, ""},
    {"premixed_flame/ch4/input.toml", "reacting", 2, "", {}, ""},
    {"reactive_shock_tube/input.toml", "reacting", 2, "", {}, ""},
    {"riemann_2d/input.toml", "accurate", 2, "", {}, ""},
    {"riemann_2d_quads/input.toml", "accurate", 2, "", {}, ""},
    {"sedov_axisymmetric/input.toml", "robust", 2, "", {}, ""},
    {"shu_osher/input.toml", "accurate", 2, "", {}, ""},
    {"sod/input.toml", "accurate", 2, "", {}, ""},
    {"sphere_axisymmetric/input.toml", "accurate", 2, "",
     {{"inlet", "farfield"}, {"outlet", "farfield"}, {"walls", "axis"}, {"cylinder", "cylinder"}}, ""},
    {"triple_flame/input.toml", "reacting", 2, "", {}, ""},
    {"triple_flame/thick/input.toml", "reacting", 2, "", {}, ""},
    {"viscous_shock_tube/input.toml", "robust", 2, "", {}, ""},
    {"wedge/input.toml", "robust", 2, "", {}, ""},
    {"cbc_les/input.toml", "les", 3, "", {}, ""},
    {"channel_inflow_retau180/input.toml", "les", 3, "", {}, ""},
    {"channel_les/input.toml", "les", 3, "", {}, ""},
    {"channel_les/input_590.toml", "les", 3, "", {}, ""},
    {"channel_retau180/input.toml", "les", 3, "", {}, ""},
    {"counterflow_diffusion/input.toml", "reacting", 3, "",
     {{"inlet", "fuel"}, {"outlet", "oxidizer"}, {"walls", "side_y"}, {"cylinder", "sym_y"}, {"back", "sym_z"},
      {"front", "side_z"}}, ""},
    {"detonation_3d/input.toml", "reacting", 3, DETONATION_INIT + "u = [0.0, 0.0, 0.0]\n", {}, ""},
    {"explosion_3d/input.toml", "accurate", 3, "", {}, ""},
    {"flame_turbulence/hit.toml", "les", 3, "", {}, ""},
    {"flame_turbulence/input.toml", "les", 3, H2_AIR_INIT, {}, ""},
    {"flame_turbulence/les.toml", "les", 3, H2_AIR_INIT, {}, ""},
    {"isotropic_turbulence/input.toml", "accurate", 3, "", {}, ""},
    {"sedov_3d/input.toml", "robust", 3, "", {}, ""},
    {"shock_bubble_3d/input.toml", "robust", 3, "", {}, ""},
    {"sphere_mach3/input.toml", "robust", 3, "",
     {{"inlet", "inflow"}, {"outlet", "outflow"}, {"walls", "lateral"}, {"cylinder", "sphere"},
      {"back", "symmetry_z"}, {"front", "symmetry_y"}}, ""},
    {"sphere_re300/input.toml", "accurate", 3, "",
     {{"left", "inflow"}, {"right", "outflow"}, {"bottom", "lateral"}, {"top", "lateral"}, {"back", "lateral"},
      {"front", "lateral"}},
     "type = \"cartesian\"\nNx = 8\nNy = 8\nNz = 8\nLx = 2.0\nLy = 2.0\nLz = 2.0\n"},
    {"taylor_green_3d/input.toml", "accurate", 3, "", {}, ""},
    {"taylor_green_3d/input_periodic.toml", "accurate", 3, "", {}, ""},
    {"turbulent_inflow_duct/input.toml", "les", 3, "", {}, ""},
    {"volvo_bluff_body/cold.toml", "les", 3, "",
     {{"inlet", "inlet"}, {"outlet", "outlet"}, {"walls", "top"}, {"cylinder", "bluffbody"}, {"back", "bottom"},
      {"front", "bottom"}}, ""},
    {"volvo_bluff_body/reacting.toml", "les", 3, "",
     {{"inlet", "inlet"}, {"outlet", "outlet"}, {"walls", "top"}, {"cylinder", "bluffbody"}, {"back", "bottom"},
      {"front", "bottom"}}, ""},
};

std::vector<ExampleCase> examples_of_this_build() {
    std::vector<ExampleCase> list;
    for (const auto & c : EXAMPLES) {
        if (c.dim == N_DIM) list.push_back(c);
    }
    return list;
}

class ExampleUnderPreset : public ::testing::TestWithParam<ExampleCase> {};

/** @brief A fresh working directory for the files a run writes, left on destruction. */
struct ScratchDirectory {
    std::filesystem::path previous = std::filesystem::current_path();
    std::filesystem::path path;
    explicit ScratchDirectory(const std::string & name)
        : path(std::filesystem::temp_directory_path() / ("mallard_preset_" + name)) {
        std::filesystem::create_directories(path);
        std::filesystem::current_path(path);
    }
    ~ScratchDirectory() {
        std::filesystem::current_path(previous);
        std::filesystem::remove_all(path);
    }
};

std::string case_name(const std::string & file) {
    std::string name = file.substr(0, file.size() - 5);
    auto other = [](char ch) { return !std::isalnum(static_cast<unsigned char>(ch)); };
    std::replace_if(name.begin(), name.end(), other, '_');
    return name;
}

std::string param_name(const ::testing::TestParamInfo<ExampleCase> & p) { return case_name(p.param.file); }

} // namespace

TEST(ExampleUnderPreset, EveryExampleHasAPreset) {
    size_t n_inputs = 0;
    for (const auto & entry : std::filesystem::recursive_directory_iterator(example_path(""))) {
        if (entry.path().extension() != ".toml") continue;
        const toml::value input = toml::parse(entry.path().string());
        if (!input.contains("numerics")) continue;  // reactor inputs
        n_inputs++;
        const std::string file = std::filesystem::relative(entry.path(), example_path("")).generic_string();
        auto listed = [&](const ExampleCase & c) { return c.file == file; };
        EXPECT_TRUE(std::any_of(EXAMPLES.begin(), EXAMPLES.end(), listed))
            << file << " has no entry in EXAMPLES";
    }
    EXPECT_EQ(n_inputs, EXAMPLES.size());
}

TEST_P(ExampleUnderPreset, RunsAFewSteps) {
    const ExampleCase & c = GetParam();
    const toml::value input = example_under_preset(c, 3, N_DIM == 3 ? 8 : 16);
    ScratchDirectory scratch(case_name(c.file));
    Solver solver;
    solver.init(input);
    EXPECT_EQ(solver.get_numerics_preset().name, c.preset);
    solver.run();
    EXPECT_EQ(solver.get_step(), 3u);
}

INSTANTIATE_TEST_SUITE_P(Examples, ExampleUnderPreset, ::testing::ValuesIn(examples_of_this_build()), param_name);

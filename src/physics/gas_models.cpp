/**
 * @file gas_models.cpp
 * @brief The gas models and the keys that select each one (gas_models.h). The rules follow Solver::init_physics,
 *        Solver::check_flamelet_input and Solver::init_chemistry.
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 */

#include "gas_models.h"

namespace gas_models {

const std::vector<GasModel> & all() {
    static const std::vector<GasModel> models = {
        {.name = "perfect",
         .title = "Perfect gas",
         .description = "One calorically perfect gas with a constant ratio of specific heats.",
         .docs = "docs/guide/physics.md",
         .sets = {{"physics.gas", "\"perfect\""}},
         .requires_keys = {},
         .removes = {"physics.mechanism", "physics.table", "chemistry", "radiation"}},
        {.name = "frozen",
         .title = "Frozen mixture",
         .description = "A mixture of species from a mechanism, transported without reactions.",
         .docs = "docs/guide/thermochemistry.md#gas-mixtures",
         .sets = {{"physics.gas", "\"mixture\""}},
         .requires_keys = {"physics.mechanism"},
         .removes = {"physics.table", "physics.units", "chemistry"}},
        {.name = "finite_rate",
         .title = "Finite-rate chemistry",
         .description = "A reacting mixture: the mechanism's reactions integrated in every cell.",
         .docs = "docs/guide/thermochemistry.md#finite-rate-chemistry",
         .sets = {{"physics.gas", "\"mixture\""}, {"chemistry.enabled", "true"}},
         .requires_keys = {"physics.mechanism"},
         .removes = {"physics.table", "physics.units"}},
        {.name = "flamelet",
         .title = "Flamelet table (FPV)",
         .description = "Flamelet/progress-variable combustion: the thermochemistry read from a table.",
         .docs = "docs/guide/flamelet.md",
         .sets = {{"physics.gas", "\"flamelet\""}},
         .requires_keys = {"physics.table"},
         .removes = {"physics.mechanism", "physics.units", "chemistry", "radiation"}},
    };
    return models;
}

std::string of(const Input & input) {
    const std::string gas = find_value_or<std::string>(input, "physics", "gas", "perfect");
    if (gas == "perfect" || gas == "flamelet") return gas;
    if (gas != "mixture") return "";
    const bool reacting = input.contains("chemistry") && find_value_or<bool>(input.at("chemistry"), "enabled", true);
    return reacting ? "finite_rate" : "frozen";
}

}  // namespace gas_models

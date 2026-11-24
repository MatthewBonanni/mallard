/**
 * @file gas_models.h
 * @brief The gas models a case chooses between, and the keys that select each one: what a front end sets and
 *        removes to switch a case's model (mallard list models).
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 */

#ifndef GAS_MODELS_H
#define GAS_MODELS_H

#include <string>
#include <utility>
#include <vector>

#include "input.h"

namespace gas_models {

struct GasModel {
    std::string name;         // perfect, frozen, finite_rate, flamelet
    std::string title;        // "Finite-rate chemistry"
    std::string description;  // one sentence
    std::string docs;         // the guide page and section that explains it
    std::vector<std::pair<std::string, std::string>> sets;  // key and its value as TOML
    std::vector<std::string> requires_keys;                 // keys the model needs, without a default
    std::vector<std::string> removes;                       // keys and tables the model does not read
};

/** @brief Every model, in the order a front end offers them. */
const std::vector<GasModel> & all();

/** @brief The model an input chooses (by its own keys, without a setup), or empty if its keys match none. */
std::string of(const Input & input);

}  // namespace gas_models

#endif  // GAS_MODELS_H

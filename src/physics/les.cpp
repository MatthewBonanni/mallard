/**
 * @file les.cpp
 * @author Matthew Bonanni (mbonanni001@gmail.com)
 * @brief Input of the subgrid-scale models.
 * @version 0.1
 * @date 2026-10-05
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 *
 */

#include "les.h"

#include <cmath>

#include "input.h"

namespace {

const std::unordered_map<SGSModel, std::string> SGS_MODEL_NAMES = {
    {SGSModel::SMAGORINSKY, "Smagorinsky"},
    {SGSModel::WALE, "WALE"},
    {SGSModel::VREMAN, "Vreman"},
    {SGSModel::SIGMA, "Sigma"},
};

} // namespace

rtype LES::default_constant(const SGSModel model) {
    switch (model) {
        case SGSModel::SMAGORINSKY:
            return 0.17_r;
        case SGSModel::WALE:
            return 0.5_r;
        case SGSModel::VREMAN:
            return 0.07_r;
        default:
            return 1.35_r;
    }
}

LES LES::from_input(const toml::value & input) {
    const toml::value & table = input.at("les");
    if (!table.is_table()) throw InputError("[les] must be a table.");
    if (toml::find_or<std::string>(input, "physics", "type", "euler") != "navier_stokes") {
        throw InputError("[les] needs physics.type = \"navier_stokes\".");
    }
    LES les;
    const std::string name = toml::find_or<std::string>(table, "model", N_DIM == 3 ? "sigma" : "wale");
    const auto it = SGS_MODELS.find(name);
    if (it == SGS_MODELS.end()) throw unknown_option(SGS_MODELS, "les.model", name);
    les.model = it->second;
    if (N_DIM == 2 && les.model == SGSModel::SIGMA) {
        throw InputError("les.model = \"sigma\" vanishes for every two-dimensional flow; use \"wale\" or \"vreman\" in "
                         "2D builds.");
    }
    les.C = find_real_or(table, "C", default_constant(les.model));
    les.Pr_t = find_real_or(table, "Pr_t", 0.9_r);
    les.Sc_t = find_real_or(table, "Sc_t", 0.9_r);
    const std::string width = toml::find_or<std::string>(table, "filter_width", N_DIM == 3 ? "scotti" : "volume");
    if (width != "volume" && width != "scotti") {
        throw InputError("les.filter_width = \"" + width + "\" is not one of: volume, scotti.");
    }
    les.scotti = width == "scotti";
    les.dynamic = toml::find_or<bool>(table, "dynamic", false);
    if (N_DIM == 2 && les.scotti) throw InputError("les.filter_width = \"scotti\" is for 3D builds.");
    for (const auto & [key, value] : std::initializer_list<std::pair<const char *, rtype>>{
             {"C", les.C}, {"Pr_t", les.Pr_t}, {"Sc_t", les.Sc_t}}) {
        if (!(value > 0.0_r)) throw InputError(std::string("les.") + key + " must be positive.");
    }
    return les;
}

double LES::scotti_factor(const double h1, const double h2, const double h3) {
    const double l1 = std::log(h1 / h3), l2 = std::log(h2 / h3);
    return std::cosh(std::sqrt(4.0 / 27.0 * (l1 * l1 - l1 * l2 + l2 * l2)));
}

logging::Items LES::summary() const {
    using logging::real;
    return {{"LES", SGS_MODEL_NAMES.at(model) + " (" + (model == SGSModel::VREMAN ? "c " : "C ") +
                        real(double(C)) + "), Pr_t " + real(double(Pr_t)) + ", Sc_t " + real(double(Sc_t)) +
                        ", filter width V^(1/" + std::to_string(N_DIM) + ")" +
                        (scotti ? " times Scotti's f(a_1, a_2)" : "") +
                        (dynamic ? "; C from the global dynamic procedure" : "")}};
}

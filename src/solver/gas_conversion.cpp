/**
 * @file gas_conversion.cpp
 * @brief Conversion of a restart's state to another gas.
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 */

#include "gas_conversion.h"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <memory>
#include <optional>
#include <sstream>

#include "comm.h"
#include "common_typedef.h"
#include "composition_map.h"
#include "flamelet.h"
#include "flamelet_table.h"
#include "json.h"
#include "log.h"
#include "mixture.h"
#include "provenance.h"

namespace gas_conversion {

namespace {

namespace fs = std::filesystem;
using chemistry::CompositionMap;
using chemistry::Species;
using Definitions = flamelet::Table::Definitions;

const char * const DOCS = "docs/guide/flamelet.md#converting-between-finite-rate-and-fpv";
const char * const KEY = "initialize.convert";

std::string real(const double v) {
    std::ostringstream o;
    o.precision(6);
    o << v;
    return o.str();
}

std::string list(const std::vector<std::string> & items) {
    std::string text;
    for (const auto & item : items) text += (text.empty() ? "" : ", ") + item;
    return text;
}

/** @brief What the restart's run block says of the gas that wrote it. */
struct Source {
    std::string gas;          // "mixture", "flamelet", "perfect", or empty if the file has no run block
    std::string model;        // the mechanism or table as the run named it
    std::string phase;
    std::string directory;    // where that run ran
};

Source source_of(const RestartData & restart) {
    Source s;
    if (restart.provenance.empty()) return s;
    try {
        const json::Value block = json::parse(restart.provenance);
        if (const json::Value * run = block.find("run"); run && run->find("directory")) {
            s.directory = run->find("directory")->string;
        }
        const json::Value * input = block.find("input");
        if (!input || !input->find("resolved")) return s;
        const Input resolved = parse_input_string(input->find("resolved")->string, "the resolved input");
        if (!resolved.contains("physics")) return s;
        const Input & physics = resolved.at("physics");
        s.gas = find_value_or<std::string>(physics, "gas", "perfect");
        s.model = find_value_or<std::string>(physics, s.gas == "flamelet" ? "table" : "mechanism", "");
        s.phase = find_value_or<std::string>(physics, "phase", "");
    } catch (const std::exception &) {
        // A block this build cannot read: the variables and the options decide
    }
    return s;
}

/** @brief The file the source's run named, as the options give it or as it resolves here or where that run ran. */
std::string resolve(const std::string & option, const char * key, const char * flag, const std::string & named,
                    const std::string & directory, const std::string & what) {
    if (!option.empty()) {
        if (!fs::exists(option)) {
            throw InputError(Diagnostic{.key = std::string(KEY) + "." + key,
                                        .message = "names " + option + ", which does not exist",
                                        .docs = DOCS});
        }
        return option;
    }
    if (!named.empty()) {
        if (fs::exists(named)) return named;
        if (!directory.empty() && fs::path(named).is_relative() && fs::exists(fs::path(directory) / named)) {
            return (fs::path(directory) / named).string();
        }
    }
    throw InputError(Diagnostic{
        .key = std::string(KEY) + "." + key,
        .message = named.empty() ? "is needed: the restart file does not say which " + what + " wrote it"
                                 : "is needed: the " + what + " " + named + " of the restart's run block does not "
                                   "exist here" + (directory.empty() ? "" : " nor in " + directory),
        .fix = std::string("Give [initialize.convert] ") + key + " = \"...\" (mallard restart convert " + flag + ")",
        .docs = DOCS});
}

const Species * find_species(const std::string & name, const std::vector<const std::vector<Species> *> & catalogs) {
    for (const auto * catalog : catalogs) {
        for (const Species & s : *catalog) {
            if (s.name == name) return &s;
        }
    }
    return nullptr;
}

/** @brief Bilger's mixture fraction Z = (beta - beta_O) / (beta_F - beta_O) of the table's streams. */
struct Bilger {
    double beta_F = 0.0, beta_O = 0.0;

    Bilger(const Definitions & d, const std::vector<const std::vector<Species> *> & catalogs) {
        auto beta = [&](const std::vector<std::pair<std::string, double>> & stream, const char * which) {
            double b = 0.0;
            for (const auto & [name, Y] : stream) {
                const Species * s = find_species(name, catalogs);
                if (!s) {
                    throw std::runtime_error(std::string("the table's ") + which + " stream holds " + name +
                                             ", which no mechanism of the conversion has: give the mechanism the "
                                             "table was generated with as [initialize.convert] mechanism");
                }
                b += chemistry::bilger_coefficient(*s) * Y;
            }
            return b;
        };
        beta_F = beta(d.fuel, "fuel");
        beta_O = beta(d.oxidizer, "oxidizer");
        if (!(std::abs(beta_F - beta_O) > 1e-12)) {
            throw std::runtime_error("the table's fuel and oxidizer streams have the same Bilger coupling function, "
                                     "so no mixture fraction separates them");
        }
    }

    /** @brief Z's weights z_k over species, Z = offset + sum_k z_k Y_k. */
    std::vector<double> weights(const std::vector<Species> & species) const {
        std::vector<double> z(species.size());
        for (size_t k = 0; k < species.size(); k++) z[k] = chemistry::bilger_coefficient(species[k]) / (beta_F - beta_O);
        return z;
    }
    double offset() const { return -beta_O / (beta_F - beta_O); }
};

/** @brief The weights of C over species, by the table's names after the aliases; missing collects the names none has. */
std::vector<double> progress_weights(const Definitions & d, const std::vector<Species> & species,
                                     const std::vector<std::pair<std::string, std::string>> & aliases,
                                     const bool species_are_source, std::vector<std::string> & missing) {
    std::vector<double> w(species.size(), 0.0);
    for (const auto & [name, weight] : d.progress_variable) {
        bool found = false;
        for (size_t k = 0; k < species.size(); k++) {
            std::string as = species[k].name;
            for (const auto & [from, to] : aliases) {
                if (species_are_source && from == as) as = to;
            }
            std::string want = name;
            for (const auto & [from, to] : aliases) {
                if (!species_are_source && from == want) want = to;
            }
            if (as == want) {
                w[k] += weight;
                found = true;
            }
        }
        if (!found && weight != 0.0) missing.push_back(name);
    }
    return w;
}

} // namespace

Options Options::from_input(const Input & initialize) {
    Options o;
    if (!contains_optional(initialize, "convert")) return o;
    const Input & c = initialize.at("convert");
    o.mechanism = find_value_or<std::string>(c, "mechanism", "");
    o.table = find_value_or<std::string>(c, "table", "");
    o.accept = find_value_or<std::string>(c, "accept", "");
    if (!o.accept.empty() && o.accept != CompositionMap::RULE) {
        throw not_accepted(c, std::string(KEY) + ".accept", o.accept, {CompositionMap::RULE});
    }
    if (contains_optional(c, "aliases")) {
        const Input & t = c.at("aliases");
        if (!t.is_table()) {
            throw InputError(Diagnostic{.key = std::string(KEY) + ".aliases",
                                        .line = line_of(c, "aliases"),
                                        .message = "is a table of source species = target species, e.g. "
                                                   "{\"CH2*\" = \"CH2(S)\"}",
                                        .docs = DOCS});
        }
        for (const auto & [name, value] : t.as_table()) {
            if (!value.is_string()) {
                throw InputError(Diagnostic{.key = std::string(KEY) + ".aliases." + name,
                                            .line = line_of(t, name),
                                            .message = "is the name of a species of the target",
                                            .docs = DOCS});
            }
            o.aliases.emplace_back(name, value.as_string());
        }
    }
    return o;
}

bool needed(const RestartData & restart, const std::vector<std::string> & target_species) {
    std::vector<std::string> have, want;
    for (const auto & name : restart.names) {
        if (name.starts_with("RHOY_")) have.push_back(name.substr(5));
    }
    if (have.empty()) return false;  // a perfect gas: the run reports the variables it lacks
    want = target_species;
    std::sort(have.begin(), have.end());
    std::sort(want.begin(), want.end());
    return have != want;
}

Summary convert(RestartData & restart, const std::string & file, const Options & options, const MixtureModel & target,
                const uint32_t n_owned, const std::vector<double> & volumes) {
    Summary s;
    s.file = file;
    s.rule = CompositionMap::RULE;
    s.aliased = {};
    const Source source = source_of(restart);
    const bool has_Z = restart.find("RHOY_Z") && restart.find("RHOY_C");
    s.source_gas = !source.gas.empty() ? source.gas : (has_Z ? "flamelet" : "mixture");
    s.target_gas = target.tabulated() ? "flamelet" : "mixture";
    if (s.source_gas != "mixture" && s.source_gas != "flamelet") {
        throw std::runtime_error("restart file " + file + " holds a " + s.source_gas + " gas: conversion is between "
                                 "gas = \"mixture\" and gas = \"flamelet\"");
    }
    const bool from_table = s.source_gas == "flamelet", to_table = target.tabulated();
    if (from_table && to_table) {
        throw std::runtime_error("restart file " + file + " holds another flamelet gas: conversion is between "
                                 "gas = \"mixture\" runs, or between gas = \"mixture\" and gas = \"flamelet\"");
    }

    // The source's gas, and the catalogs of species compositions
    std::optional<MixtureModel> source_mixture;
    std::shared_ptr<const flamelet::Table> source_table;
    std::optional<flamelet::FlameletModel> source_flamelet;
    std::optional<chemistry::Mechanism> table_mechanism;
    if (from_table) {
        s.source_model = resolve(options.table, "table", "--table", source.model, source.directory, "table");
        try {
            source_table = std::make_shared<const flamelet::Table>(flamelet::Table::read(s.source_model));
        } catch (const std::runtime_error & e) {
            throw InputError(Diagnostic{.key = std::string(KEY) + ".table", .message = e.what(), .docs = DOCS});
        }
        source_flamelet.emplace(source_table, false);
        s.table_mechanism = source_table->mechanism();
    } else {
        s.source_model = resolve(options.mechanism, "mechanism", "--mechanism", source.model, source.directory,
                                 "mechanism");
        std::string text = "[physics]\nmechanism = " + json::quote(s.source_model) + "\n";
        if (!source.phase.empty()) text += "phase = " + json::quote(source.phase) + "\n";
        source_mixture.emplace(MixtureModel::from_input(parse_input_string(text, "the source's gas")));
    }
    if (from_table && !options.mechanism.empty()) {
        try {
            table_mechanism = chemistry::read_mechanism(options.mechanism);
        } catch (const std::runtime_error & e) {
            throw InputError(Diagnostic{.key = std::string(KEY) + ".mechanism", .message = e.what(), .docs = DOCS});
        }
    }
    if (to_table) {
        s.target_model = target.flamelet().file();
        s.table_mechanism = target.flamelet().table().mechanism();
    } else {
        s.target_model = target.mechanism().file;
    }
    if (comm::is_root()) {
        s.source_hash = provenance::sha256_file(s.source_model);
        s.target_hash = provenance::sha256_file(s.target_model);
    }

    // The source's species variables
    std::vector<std::string> source_block;
    if (from_table) {
        source_block = {"Z", "C"};
        if (restart.find("RHOY_Z2")) source_block.emplace_back("Z2");
    } else {
        source_block = source_mixture->mechanism().species_names();
    }
    std::vector<const std::vector<rtype> *> rhoY_source;
    for (const auto & name : source_block) {
        rhoY_source.push_back(restart.find("RHOY_" + name));
        if (!rhoY_source.back()) {
            throw std::runtime_error("restart file " + file + " has no variable RHOY_" + name + " of the source's " +
                                     (from_table ? "flamelet scalars" : "mechanism " + s.source_model));
        }
    }
    for (const auto & name : restart.names) {
        if (name.starts_with("RHOY_") &&
            std::find(source_block.begin(), source_block.end(), name.substr(5)) == source_block.end()) {
            throw std::runtime_error("restart file " + file + " has variable " + name + ", which the source's " +
                                     (from_table ? "flamelet gas" : "mechanism " + s.source_model) +
                                     " does not have: give the mechanism that wrote it as [initialize.convert] "
                                     "mechanism");
        }
    }
    const std::vector<rtype> * U_source[N_CONSERVATIVE];
    FOR_I_CONSERVATIVE {
        U_source[i] = restart.find(CONSERVATIVE_NAMES[i]);
        if (!U_source[i]) throw std::runtime_error("restart file " + file + " has no variable " + CONSERVATIVE_NAMES[i]);
    }
    const std::vector<rtype> * T_seed = restart.find("T_SEED");

    // The composition: over the source mechanism's species, or the table's
    std::vector<Species> composition;
    std::vector<uint32_t> table_fields;
    const chemistry::Mechanism * target_mechanism = to_table ? nullptr : &target.mechanism();
    std::vector<const std::vector<Species> *> catalogs;
    if (from_table) {
        if (table_mechanism) catalogs.push_back(&table_mechanism->species);
        if (target_mechanism) catalogs.push_back(&target_mechanism->species);
        if (!to_table) {
            std::vector<std::string> unknown;
            for (const auto & name : source_table->species()) {
                const Species * sp = find_species(name, catalogs);
                if (!sp) {
                    unknown.push_back(name);
                    continue;
                }
                composition.push_back(*sp);
                table_fields.push_back(static_cast<uint32_t>(source_table->field("Y_" + name)));
            }
            if (!unknown.empty()) {
                throw InputError(Diagnostic{
                    .key = std::string(KEY) + ".mechanism",
                    .message = "is needed: the table's species " + list(unknown) + " are not species of " +
                               (table_mechanism ? options.mechanism : "the target's mechanism") +
                               ", so their elements are unknown",
                    .fix = "Give the mechanism the table was generated with as [initialize.convert] mechanism "
                           "(mallard restart convert --mechanism)",
                    .docs = DOCS});
            }
            if (composition.empty()) {
                throw std::runtime_error("the table " + s.source_model + " has no species block: a conversion to a "
                                         "mixture takes the composition from it");
            }
        }
    } else {
        composition = source_mixture->mechanism().species;
        catalogs.push_back(&source_mixture->mechanism().species);
        if (target_mechanism) catalogs.push_back(&target_mechanism->species);
    }
    std::optional<CompositionMap> map;
    if (!to_table) {
        try {
            map.emplace(composition, target_mechanism->species, options.aliases);
        } catch (const std::runtime_error & e) {
            throw InputError(Diagnostic{.key = std::string(KEY) + ".aliases", .message = e.what(), .docs = DOCS});
        }
        s.aliased = map->aliased();
        for (const uint32_t k : map->added()) s.added.push_back(target_mechanism->species[k].name);
    } else if (!options.aliases.empty() && !from_table) {
        for (const auto & [from, to] : options.aliases) {
            if (!find_species(from, {&composition})) {
                throw InputError(Diagnostic{.key = std::string(KEY) + ".aliases." + from,
                                            .message = "names a species the source's mechanism does not have",
                                            .docs = DOCS});
            }
            s.aliased.emplace_back(from, to);
        }
    }

    // Z and C: the target table's definitions, or the source table's for the round trip
    const flamelet::Table * table = to_table ? &target.flamelet().table() : source_table.get();
    std::optional<Definitions> defs;
    std::optional<Bilger> bilger;
    std::vector<double> z_weights, c_weights;
    double z_offset = 0.0;
    std::vector<std::string> missing_weights;
    if (table) {
        defs = table->definitions();
        try {
            bilger.emplace(*defs, catalogs);
        } catch (const std::runtime_error & e) {
            throw InputError(Diagnostic{.key = std::string(KEY) + ".mechanism", .message = e.what(), .docs = DOCS});
        }
        z_offset = bilger->offset();
        if (to_table) {
            z_weights = bilger->weights(composition);
            c_weights = progress_weights(*defs, composition, options.aliases, true, missing_weights);
            if (!missing_weights.empty()) {
                throw InputError(Diagnostic{
                    .key = std::string(KEY) + ".aliases",
                    .message = "the table's progress variable C = sum_k w_k Y_k weighs " + list(missing_weights) +
                               ", which the source's species lack",
                    .fix = "Map a species of the source onto each with aliases = {\"<source>\" = \"<table>\"}",
                    .docs = DOCS});
            }
        } else {
            z_weights = bilger->weights(target_mechanism->species);
            c_weights = progress_weights(*defs, target_mechanism->species, options.aliases, false, missing_weights);
            if (!missing_weights.empty()) {
                s.reasons.push_back("the target lacks " + list(missing_weights) +
                                    ", which the table's progress variable weighs");
            }
        }
    }

    // The cells
    const size_t n_cells = U_source[0]->size();
    const uint32_t n_target = target.n_species();
    const std::vector<std::string> target_block = target.species_names();
    std::vector<std::vector<rtype>> U_out(N_CONSERVATIVE, std::vector<rtype>(n_cells));
    std::vector<std::vector<rtype>> rhoY_out(n_target, std::vector<rtype>(n_cells));
    std::vector<rtype> T_out(n_cells);
    const size_t n_e = map ? map->elements().size() : 0;
    std::vector<double> moved_max(n_e, 0.0), moved_mass(n_e, 0.0), dropped_max(map ? map->dropped().size() : 0, 0.0);
    double mass = 0.0, max_shared = 0.0, e_source = 0.0, e_target = 0.0, e_scale = 0.0, Z_trip = 0.0, C_trip = 0.0;
    double sum_error = 0.0;
    uint64_t projected = 0, Z_clamped = 0, C_clamped = 0;
    const double tol = table && table->single() ? 1e-6 : 1e-9;
    std::string error;
    try {
        std::vector<double> Y_source(source_block.size()), Y_comp(composition.size()), Y_target(map ? n_target : 0);
        const auto & host = source_flamelet ? source_flamelet->host() : flamelet::Gas<Kokkos::HostSpace>{};
        for (size_t c = 0; c < n_cells; c++) {
            const bool owned = c < n_owned;
            const double rho = static_cast<double>((*U_source[0])[c]), V = c < volumes.size() ? volumes[c] : 0.0;
            rtype u[N_DIM];
            double u2 = 0.0;
            FOR_I_DIM {
                u[i] = static_cast<rtype>(static_cast<double>((*U_source[1 + i])[c]) / rho);
                u2 += static_cast<double>(u[i]) * static_cast<double>(u[i]);
            }
            const double E = static_cast<double>((*U_source[N_DIM + 1])[c]), e = E / rho - 0.5 * u2;
            for (size_t k = 0; k < source_block.size(); k++) Y_source[k] = static_cast<double>((*rhoY_source[k])[c]) / rho;
            double T = 0.0, p = 0.0, Z = 0.0, C = 0.0, Zvar = 0.0;
            if (from_table) {
                Z = Y_source[0];
                C = Y_source[1];
                if (source_block.size() > 2) Zvar = std::max(Y_source[2] - Z * Z, 0.0);
                const flamelet::Thermo th = source_flamelet->thermo(Z, C, Zvar);
                T = th.temperature(e);
                p = rho * th.R * T;
                if (!to_table) {
                    const flamelet::Point<3> point = host.locate(Z, Zvar, C);
                    double sum = 0.0;
                    for (size_t k = 0; k < composition.size(); k++) {
                        Y_comp[k] = host.field(point, table_fields[k]);
                        sum += Y_comp[k];
                    }
                    if (std::abs(sum - 1.0) > 1e-3) {
                        throw std::runtime_error("the mass fractions of the table's species block sum to " + real(sum) +
                                                 " at Z = " + real(Z) + ", C = " + real(C) + ": a conversion to a "
                                                 "mixture needs every species of the table's mechanism in its "
                                                 "species block");
                    }
                    if (owned) sum_error = std::max(sum_error, std::abs(sum - 1.0));
                    for (double & y : Y_comp) y /= sum;
                    // The C a lookup reads: clamped to the table's range at (Z, Sz)
                    const double Lambda = host.lambda(Z, Zvar, C);
                    const flamelet::Point<3> at = host.locate(Z, Zvar, 0.0);
                    const double C_min = host.fpv.table.at_last_node(at, 0, host.fpv.field_C);
                    const double C_max = host.fpv.table.at_last_node(at, host.fpv.table.n_nodes[2] - 1, host.fpv.field_C);
                    const double C_lookup = C_min + Lambda * (C_max - C_min);
                    if (owned && std::abs(C_lookup - C) > tol * std::max(1.0, std::abs(C_max))) C_clamped++;
                    C = C_lookup;
                }
            } else {
                Y_comp = Y_source;
                T = source_mixture->temperature(e, Y_comp, T_seed ? static_cast<double>((*T_seed)[c]) : 300.0);
                p = rho * source_mixture->gas_constant(Y_comp) * T;
            }
            rtype U[N_CONSERVATIVE];
            std::vector<rtype> rhoY(n_target);
            if (to_table) {
                const flamelet::FlameletModel & fl = target.flamelet();
                if (!from_table) {
                    Z = z_offset;
                    C = 0.0;
                    for (size_t k = 0; k < composition.size(); k++) {
                        Z += z_weights[k] * Y_comp[k];
                        C += c_weights[k] * Y_comp[k];
                    }
                    const double Z_in = std::clamp(Z, 0.0, 1.0);
                    if (owned && std::abs(Z_in - Z) > tol) Z_clamped++;
                    Z = Z_in;
                    const double C_min = fl.host().fpv.table.at_last_node(fl.host().locate(Z, 0.0, 0.0), 0,
                                                                          fl.host().fpv.field_C);
                    const double C_max = fl.burnt_progress(Z);
                    const double C_in = std::clamp(C, C_min, C_max);
                    if (owned && std::abs(C_in - C) > tol * std::max(1.0, std::abs(C_max))) C_clamped++;
                    C = C_in;
                }
                double Ud[N_CONSERVATIVE], uu[N_DIM], rhoYd[3] = {};
                FOR_I_DIM uu[i] = static_cast<double>(u[i]);
                fl.conservatives(p, T, uu, Z, C, Ud, rhoYd);
                FOR_I_CONSERVATIVE U[i] = static_cast<rtype>(Ud[i]);
                for (uint32_t k = 0; k < n_target; k++) rhoY[k] = static_cast<rtype>(rhoYd[k]);
            } else {
                const CompositionMap::Result r = map->map(Y_comp.data(), Y_target.data());
                if (owned) {
                    if (r.projected) projected++;
                    max_shared = std::max(max_shared, r.max_shared_change);
                    for (size_t e2 = 0; e2 < n_e; e2++) {
                        moved_max[e2] = std::max(moved_max[e2], r.moved[e2]);
                        moved_mass[e2] += rho * V * r.moved[e2];
                    }
                    for (size_t d = 0; d < dropped_max.size(); d++) {
                        dropped_max[d] = std::max(dropped_max[d], Y_comp[map->dropped()[d]]);
                    }
                }
                target.conservatives(p, T, u, Y_target, U, rhoY.data());
                if (from_table && owned && bilger) {
                    double Z2 = z_offset, C2 = 0.0;
                    for (uint32_t k = 0; k < n_target; k++) {
                        Z2 += z_weights[k] * Y_target[k];
                        C2 += c_weights[k] * Y_target[k];
                    }
                    Z_trip = std::max(Z_trip, std::abs(Z2 - Z));
                    C_trip = std::max(C_trip, std::abs(C2 - C));
                }
            }
            FOR_I_CONSERVATIVE U_out[i][c] = U[i];
            for (uint32_t k = 0; k < n_target; k++) rhoY_out[k][c] = rhoY[k];
            T_out[c] = static_cast<rtype>(T);
            if (owned) {
                mass += rho * V;
                e_source += E * V;
                e_scale += std::abs(E) * V;
                e_target += static_cast<double>(U[N_DIM + 1]) * V;
            }
        }
    } catch (const std::exception & e) {
        error = e.what();
    }
    for (const std::string & message : comm::allgather(error)) {
        if (!message.empty()) throw std::runtime_error("converting restart file " + file + ": " + message);
    }

    // Every rank's cells
    std::vector<double> sums = {mass, e_source, e_target, e_scale};
    sums.insert(sums.end(), moved_mass.begin(), moved_mass.end());
    comm::allreduce(std::span<double>(sums), comm::Op::SUM);
    std::vector<double> maxima = {max_shared, Z_trip, C_trip, sum_error};
    maxima.insert(maxima.end(), moved_max.begin(), moved_max.end());
    maxima.insert(maxima.end(), dropped_max.begin(), dropped_max.end());
    comm::allreduce(std::span<double>(maxima), comm::Op::MAX);
    std::array<uint64_t, 3> counts = comm::allreduce(std::array<uint64_t, 3>{projected, Z_clamped, C_clamped}, comm::Op::SUM);
    s.energy_source = sums[1];
    s.energy_target = sums[2];
    s.energy_relative = sums[3] > 0.0 ? (sums[2] - sums[1]) / sums[3] : 0.0;
    s.max_shared_change = maxima[0];
    s.Z_round_trip = maxima[1];
    s.C_round_trip = maxima[2];
    s.table_sum_error = maxima[3];
    s.cells_projected = counts[0];
    s.Z_clamped = counts[1];
    s.C_clamped = counts[2];
    if (map) {
        for (size_t e2 = 0; e2 < n_e; e2++) {
            if (maxima[4 + e2] <= 0.0) continue;
            s.moved_max.emplace_back(map->elements()[e2], maxima[4 + e2]);
            s.moved_mean.emplace_back(map->elements()[e2], sums[0] > 0.0 ? sums[4 + e2] / sums[0] : 0.0);
        }
        for (size_t d = 0; d < map->dropped().size(); d++) {
            s.dropped.emplace_back(composition[map->dropped()[d]].name, maxima[4 + n_e + d]);
        }
    }

    // What makes it inexact, which the user must accept
    if (s.cells_projected > 0) {
        s.reasons.push_back("species the target lacks hold mass, moved onto the target's species in " +
                            std::to_string(s.cells_projected) + " cells");
    }
    if (!s.aliased.empty()) {
        std::string pairs;
        for (const auto & [from, to] : s.aliased) pairs += (pairs.empty() ? "" : ", ") + from + " = " + to;
        s.reasons.push_back("species mapped by alias: " + pairs);
    }
    if (!from_table && !to_table && std::abs(s.energy_relative) > 1e-12) {
        s.reasons.push_back("the target mechanism's thermodynamics change the total energy by " +
                            real(100.0 * s.energy_relative) + "% at the kept p and T");
    }
    if (s.Z_clamped > 0) s.reasons.push_back("Z clamped to [0, 1] in " + std::to_string(s.Z_clamped) + " cells");
    if (s.C_clamped > 0) {
        s.reasons.push_back("C clamped to the table's range at Z in " + std::to_string(s.C_clamped) + " cells");
    }
    if (from_table && !to_table) {
        if (s.Z_round_trip > tol || s.C_round_trip > tol) {
            s.reasons.push_back("the table's species do not reproduce Z and C (max error " + real(s.Z_round_trip) +
                                " in Z, " + real(s.C_round_trip) + " in C)");
        }
        if (s.table_sum_error > tol) {
            s.reasons.push_back("the table's mass fractions sum to 1 within " + real(s.table_sum_error) +
                                ", normalized");
        }
    }
    s.exact = s.reasons.empty();

    // Notes on what else changed
    if (restart.find("CHEM_H")) s.notes.emplace_back("dropped CHEM_H, the chemistry integrator's step seed");
    if (from_table) {
        s.notes.emplace_back(source_block.size() > 2 ? "the source's subgrid variance of Z from its Z2"
                                                     : "the source's table read at zero subgrid variance of Z");
        s.notes.emplace_back("the temperature kept is the flamelet run's T, of its energy equation, not T_TABLE");
    }
    if (to_table) {
        s.notes.emplace_back("Z (Bilger, the table's streams) and C (the table's weights) from the composition, at "
                             "zero subgrid variance" +
                             std::string(target.flamelet().variance() == flamelet::FlameletModel::Variance::TRANSPORTED
                                             ? ": Z2 = Z^2"
                                             : ""));
    }
    s.notes.emplace_back("T_SEED set to the kept temperature");
    s.notes.emplace_back("averages of fields the target does not sample are not read; the target refuses averages "
                         "it samples and the file lacks (statistics.reset = true starts them afresh)");

    // The converted variables replace the source's
    RestartData out;
    out.step = restart.step;
    out.t = restart.t;
    out.n_cells = restart.n_cells;
    for (size_t v = 0; v < restart.names.size(); v++) {
        const std::string & name = restart.names[v];
        const bool replaced = name.starts_with("RHOY_") || name == "T_SEED" || name == "CHEM_H" ||
                              std::find(CONSERVATIVE_NAMES.begin(), CONSERVATIVE_NAMES.end(), name) !=
                                  CONSERVATIVE_NAMES.end();
        if (replaced) continue;
        out.names.push_back(name);
        out.fields.push_back(std::move(restart.fields[v]));
    }
    FOR_I_CONSERVATIVE {
        out.names.push_back(CONSERVATIVE_NAMES[i]);
        out.fields.push_back(std::move(U_out[i]));
    }
    for (uint32_t k = 0; k < n_target; k++) {
        out.names.push_back("RHOY_" + target_block[k]);
        out.fields.push_back(std::move(rhoY_out[k]));
    }
    out.names.emplace_back("T_SEED");
    out.fields.push_back(std::move(T_out));
    out.attributes = std::move(restart.attributes);
    out.arrays = std::move(restart.arrays);
    out.faces = std::move(restart.faces);
    out.provenance = std::move(restart.provenance);
    out.snapshot = std::move(restart.snapshot);
    out.complete = restart.complete;
    restart = std::move(out);
    return s;
}

std::string Summary::json() const {
    auto pairs = [](const std::vector<std::pair<std::string, double>> & values) {
        json::Object o;
        for (const auto & [k, v] : values) o.add(k, v);
        return o.str();
    };
    auto strings = [](const std::vector<std::string> & values) {
        std::string a = "[";
        for (const auto & v : values) a += (a.size() > 1 ? "," : "") + json::quote(v);
        return a + "]";
    };
    std::string aliases = "[";
    for (const auto & [from, to] : aliased) {
        aliases += (aliases.size() > 1 ? "," : "") + json::Object().add("source", from).add("target", to).str();
    }
    aliases += "]";
    json::Object source, target, energy, flamelet;
    source.add("file", file).add("gas", source_gas).add("model", source_model).add("sha256", source_hash);
    target.add("gas", target_gas).add("model", target_model).add("sha256", target_hash);
    energy.add("source", energy_source).add("target", energy_target).add("relative_change", energy_relative);
    flamelet.add("table_mechanism", table_mechanism)
        .add("Z_clamped_cells", Z_clamped)
        .add("C_clamped_cells", C_clamped)
        .add("Z_round_trip", Z_round_trip)
        .add("C_round_trip", C_round_trip)
        .add("table_sum_error", table_sum_error);
    return json::Object()
        .add("schema", std::string(SCHEMA))
        .add("version", VERSION)
        .raw("source", source.str())
        .raw("target", target.str())
        .add("rule", rule)
        .add("exact", exact)
        .raw("reasons", strings(reasons))
        .raw("dropped_species", pairs(dropped))
        .raw("added_species", strings(added))
        .raw("aliases", aliases)
        .raw("moved_max", pairs(moved_max))
        .raw("moved_mean", pairs(moved_mean))
        .add("cells_projected", cells_projected)
        .add("max_shared_change", max_shared_change)
        .raw("energy", energy.str())
        .raw("flamelet", flamelet.str())
        .raw("notes", strings(notes))
        .str();
}

std::string Summary::text() const {
    std::ostringstream o;
    o << "Conversion of " << file << ": " << source_gas << " (" << source_model << ") to " << target_gas << " ("
      << target_model << ")\n";
    if (!source_hash.empty()) o << "  source sha256: " << source_hash << "\n";
    if (!target_hash.empty()) o << "  target sha256: " << target_hash << "\n";
    if (!table_mechanism.empty()) o << "  table's mechanism: " << table_mechanism << "\n";
    o << "  rule: " << rule << (exact ? " (exact: nothing to accept)" : "") << "\n";
    for (const auto & reason : reasons) o << "  not exact: " << reason << "\n";
    if (!dropped.empty()) {
        o << "  species dropped (max Y):";
        for (const auto & [name, y] : dropped) o << " " << name << " (" << real(y) << ")";
        o << "\n";
    }
    if (!added.empty()) o << "  species added: " << list(added) << "\n";
    if (!moved_max.empty()) {
        o << "  element mass moved (max, mass-weighted mean):";
        for (size_t i = 0; i < moved_max.size(); i++) {
            o << " " << moved_max[i].first << " " << real(moved_max[i].second) << ", " << real(moved_mean[i].second) << ";";
        }
        o << "\n  max change of a shared species' Y: " << real(max_shared_change) << "\n";
    }
    o << "  total energy: " << real(energy_source) << " -> " << real(energy_target) << " J"
      << (N_DIM == 2 ? "/m" : "") << " (" << real(100.0 * energy_relative) << "%)\n";
    if (source_gas == "flamelet" && target_gas == "mixture") {
        o << "  round trip: max |Z(Y) - Z| " << real(Z_round_trip) << ", max |C(Y) - C| " << real(C_round_trip) << "\n";
    }
    for (const auto & note : notes) o << "  note: " << note << "\n";
    return o.str();
}

} // namespace gas_conversion

/**
 * @file gas_conversion.h
 * @brief Conversion of a restart's state to another gas: finite-rate chemistry of one mechanism to another, and
 *        finite-rate chemistry to and from a flamelet table (docs/guide/flamelet.md#converting-between-finite-rate-and-fpv,
 *        docs/guide/thermochemistry.md#converting-between-species-sets). Pressure, temperature and velocity are kept;
 *        the composition is mapped by species name, with an element-conserving projection of the mass of species
 *        the target lacks (chemistry::CompositionMap); density and energy are those of the target's thermodynamics.
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 */

#ifndef GAS_CONVERSION_H
#define GAS_CONVERSION_H

#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "data_writer.h"
#include "input.h"
#include "interfaces.h"

class MixtureModel;

namespace gas_conversion {

/** @brief Name and version of the summary's JSON (mallard restart convert --format json). */
inline constexpr const char * SCHEMA = interfaces::RESTART_CONVERT.name;
constexpr int VERSION = interfaces::RESTART_CONVERT.version;

/** @brief What a conversion did, the same on every rank. */
struct Summary {
    std::string file;                   ///< the source restart file
    std::string source_gas, target_gas;  ///< "mixture" or "flamelet"
    std::string source_model, target_model;  ///< the mechanism or table file of each
    std::string source_hash, target_hash;    ///< their SHA-256, for provenance (never compared)
    std::string table_mechanism;        ///< the flamelet table's mechanism, "<name> sha256:<hash>", if a table takes part
    std::string rule;                   ///< chemistry::CompositionMap::RULE
    bool exact = true;                  ///< nothing moved, aliased, clamped or changed beyond round-off
    std::vector<std::string> reasons;   ///< why not exact
    std::vector<std::pair<std::string, double>> dropped;  ///< source species the target lacks, with their max mass fraction
    std::vector<std::string> added;     ///< target species the source lacks, set to zero where nothing moves onto them
    std::vector<std::pair<std::string, std::string>> aliased;  ///< (source, target) names
    std::vector<std::pair<std::string, double>> moved_max;   ///< per element: max mass fraction moved over the cells
    std::vector<std::pair<std::string, double>> moved_mean;  ///< per element: mass-weighted mean
    uint64_t cells_projected = 0;
    double max_shared_change = 0.0;    ///< max |delta Y| of a species both sets have
    double energy_source = 0.0, energy_target = 0.0;  ///< sum of rho E V over the cells
    double energy_relative = 0.0;      ///< (target - source) / sum |rho E V| of the source
    uint64_t Z_clamped = 0, C_clamped = 0;  ///< flamelet: cells whose Z or C was clamped to the table's range
    double Z_round_trip = 0.0, C_round_trip = 0.0;  ///< flamelet to mixture: max |Z(Y) - Z|, |C(Y) - C|
    double table_sum_error = 0.0;      ///< flamelet to mixture: max |sum_k Y_k - 1| of the table's species
    std::vector<std::string> notes;    ///< what else was dropped or chosen

    /** @brief As JSON (schema SCHEMA, version VERSION). */
    std::string json() const;
    /** @brief As text for the terminal and the run log. */
    std::string text() const;
};

/** @brief A conversion its approval declined (mallard restart convert --dry-run, or no acceptance). */
struct Declined : std::runtime_error {
    using std::runtime_error::runtime_error;
};

/** @brief The keys of [initialize.convert], or the flags of mallard restart convert. */
struct Options {
    std::string mechanism;  ///< the source's species' mechanism, if its run block's does not resolve
    std::string table;      ///< the source's table, if its run block's does not resolve
    std::vector<std::pair<std::string, std::string>> aliases;  ///< (source name, target name)
    std::string accept;     ///< the rule the user accepted, or empty
    std::vector<std::pair<std::string, double>> progress_variable;  ///< version 1 tables: C = sum_k w_k Y_k
    std::vector<std::pair<std::string, double>> fuel, oxidizer;     ///< version 1 tables: the streams of Z
    bool fuel_moles = false, oxidizer_moles = false;  ///< the streams are given as mole fractions

    /** @brief From the optional table [initialize.convert]. */
    static Options from_input(const Input & initialize);
};

/**
 * @brief Whether a restart holds another gas than the target's: its species variables (RHOY_*) differ from
 *        the target's species block.
 */
bool needed(const RestartData & restart, const std::vector<std::string> & target_species);

/**
 * @brief Converts the restart's state to the target gas in place (collective): its RHOY_* and T_SEED become the
 *        target's, CHEM_H is dropped, everything else is kept. The source gas comes from the restart's run block
 *        (resolved input), with the options' files where given.
 * @param n_owned The cells of this rank that count in the summary
 * @param volumes The volume of each cell
 * @throws InputError for what the user can fix (a file, a definition, an alias)
 * @throws std::runtime_error for a conversion that cannot be done
 */
Summary convert(RestartData & restart, const std::string & file, const Options & options, const MixtureModel & target,
                uint32_t n_owned, const std::vector<double> & volumes);

} // namespace gas_conversion

#endif // GAS_CONVERSION_H

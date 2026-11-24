/**
 * @file composition_map.h
 * @brief Maps mass fractions between two species sets (two mechanisms, or a flamelet table's species and a
 *        mechanism): species matched by name and elemental composition, and the mass of source species the
 *        target lacks moved onto the target's species by an element-conserving projection
 *        (docs/guide/thermochemistry.md#converting-between-species-sets).
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 */

#ifndef CHEMISTRY_COMPOSITION_MAP_H
#define CHEMISTRY_COMPOSITION_MAP_H

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "mechanism.h"

namespace chemistry {

/**
 * @brief Bilger's coupling function per unit mass fraction of species s, (2 C + H / 2 - O) / W [kmol/kg]:
 *        beta = sum_k b_k Y_k is conserved by reactions (Bilger, Starner & Kee 1990), so
 *        Z = (beta - beta_O) / (beta_F - beta_O).
 */
double bilger_coefficient(const Species & s);

/** @brief Atoms of element e in species s. */
double atoms(const Species & s, const std::string & e);

/**
 * @brief The map of mass fractions from one species set to another. Species match by name, after the aliases
 *        (source name, target name); a matched pair must have the same elemental composition. Where the source
 *        species the target lacks hold mass, the target composition is the element-conserving projection of the
 *        source's: of the compositions Y >= 0 with the source's element mass fractions, the one closest to the
 *        source on the shared species in the chi-square distance sum_k (Y_k - Y_k^s)^2 / Y_k^s (a second-order
 *        approximation of the relative entropy), so the moved mass goes to the abundant carriers of each element
 *        in proportion to their abundance. Elsewhere the shared species keep their mass fractions exactly.
 */
class CompositionMap {
    public:
        /** @brief The name of the rule, which a conversion that moves mass must be accepted by. */
        static constexpr const char * RULE = "element-projection";

        /**
         * @throws std::runtime_error for a name shared by species of different compositions, an alias naming a
         *         species neither set has, or two source species mapped onto one target species
         */
        CompositionMap(std::vector<Species> from, std::vector<Species> to,
                       const std::vector<std::pair<std::string, std::string>> & aliases = {});

        /** @brief The source species of each target species, or -1. */
        const std::vector<int32_t> & source_of() const { return m_source_of; }
        /** @brief The source species the target lacks. */
        const std::vector<uint32_t> & dropped() const { return m_dropped; }
        /** @brief The target species the source lacks. */
        const std::vector<uint32_t> & added() const { return m_added; }
        /** @brief The aliases the map applies: (source name, target name). */
        const std::vector<std::pair<std::string, std::string>> & aliased() const { return m_aliased; }
        /** @brief The elements of both sets. */
        const std::vector<std::string> & elements() const { return m_elements; }
        const std::vector<Species> & from() const { return m_from; }
        const std::vector<Species> & to() const { return m_to; }

        /** @brief What one map did. */
        struct Result {
            std::vector<double> moved;      ///< per element, its mass fraction in dropped species
            double max_shared_change = 0.0;  ///< max |Y_target - Y_source| over the shared species
            bool projected = false;          ///< dropped species held mass, so the projection ran
        };

        /**
         * @brief The target mass fractions Y_to of the source mass fractions Y_from.
         * @throws std::runtime_error when no composition of the target species has the source's element mass
         *         fractions (an element only dropped species carry)
         */
        Result map(const double * Y_from, double * Y_to) const;

        /** @brief Element mass fractions of mass fractions Y of the source (from) or target species. */
        std::vector<double> element_fractions(const double * Y, bool target) const;

    private:
        std::vector<Species> m_from, m_to;
        std::vector<std::string> m_elements;
        std::vector<std::vector<double>> m_mu_from, m_mu_to;  // [element][species]: element mass per species mass
        std::vector<int32_t> m_source_of;
        std::vector<uint32_t> m_dropped, m_added;
        std::vector<std::pair<std::string, std::string>> m_aliased;
};

} // namespace chemistry

#endif // CHEMISTRY_COMPOSITION_MAP_H

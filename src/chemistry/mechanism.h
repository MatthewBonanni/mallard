/**
 * @file mechanism.h
 * @author Matthew Bonanni (mbonanni001@gmail.com)
 * @brief Gas-phase mechanisms read from Cantera YAML files.
 * @version 0.3
 * @date 2026-10-02
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 *
 */

#ifndef CHEMISTRY_MECHANISM_H
#define CHEMISTRY_MECHANISM_H

#include <array>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace chemistry {

// Physical constants (CODATA 2018 exact values, as in Cantera); quantities in kmol
constexpr double AVOGADRO = 6.02214076e26;           // 1/kmol
constexpr double BOLTZMANN = 1.380649e-23;           // J/K
constexpr double GAS_CONSTANT = AVOGADRO * BOLTZMANN; // J/(kmol K)
constexpr double ELEMENTARY_CHARGE = 1.602176634e-19; // C
constexpr double ONE_ATM = 101325.0;                 // Pa

/**
 * @brief Atomic weight in kg/kmol of an element symbol (Cantera's table;
 *        case-insensitive), or a negative value if unknown.
 */
double atomic_weight(const std::string & symbol);

enum class ThermoModel {
    NASA7,
    NASA9,
    CONSTANT_CP,
};

/**
 * @brief Species thermodynamics as piecewise polynomials in NASA-9 form, one
 *        per temperature range:
 *        cp/R = a0/T^2 + a1/T + a2 + a3 T + a4 T^2 + a5 T^3 + a6 T^4,
 *        h/(RT) = -a0/T^2 + a1 ln(T)/T + a2 + a3 T/2 + a4 T^2/3 + a5 T^3/4 + a6 T^4/5 + a7/T,
 *        s/R = -a0/(2 T^2) - a1/T + a2 ln(T) + a3 T + a4 T^2/2 + a5 T^3/3 + a6 T^4/4 + a8.
 *
 * NASA-7 coefficients (b0..b6) are stored as (0, 0, b0, ..., b6); constant
 * cp as a2 = cp0/R, a7 = (h0 - cp0 T0)/R, a8 = (s0 - cp0 ln T0)/R. Outside
 * the bounds the outermost polynomial is extrapolated.
 */
struct SpeciesThermo {
    ThermoModel model = ThermoModel::NASA7;
    std::vector<double> T_bounds;               // n_ranges + 1, increasing
    std::vector<std::array<double, 9>> coeffs;  // per range
    std::array<double, 4> constant_cp = {};     // CONSTANT_CP: T0 [K], h0 [J/kmol], s0, cp0 [J/(kmol K)]

    /**
     * @brief Index of the range used at T. As in Cantera, a NASA-7 bound
     *        belongs to the range below it, a NASA-9 bound to the range above.
     */
    size_t range(double T) const;
};

struct Species {
    std::string name;
    std::vector<std::pair<std::string, double>> composition;  // element, atoms
    double molecular_weight = 0.0;                            // kg/kmol
    SpeciesThermo thermo;
};

/** @brief Modified Arrhenius rate k = A T^b exp(-Ea / (R T)), SI units with kmol. */
struct Arrhenius {
    double A = 0.0;
    double b = 0.0;
    double Ea_R = 0.0;  // activation energy over the gas constant [K]
};

enum class ReactionType {
    ELEMENTARY,
    THREE_BODY,
    FALLOFF,
    PLOG,       // pressure-dependent Arrhenius: ln k interpolated linearly in ln p
    CHEBYSHEV,  // log10 k as a Chebyshev series in 1/T and log10 p
};

enum class FalloffType {
    LINDEMANN,
    TROE,
    SRI,
};

/**
 * @brief A gas-phase reaction: elementary, three-body or falloff
 *        (Lindemann, Troe, SRI), reversible (reverse rate from the
 *        equilibrium constant) or not, with optional non-integer orders.
 */
struct Reaction {
    std::string equation;
    ReactionType type = ReactionType::ELEMENTARY;
    std::vector<std::pair<int32_t, double>> reactants;  // species, stoichiometric coefficient
    std::vector<std::pair<int32_t, double>> products;
    std::vector<std::pair<int32_t, double>> orders;     // forward reaction orders of the reactants
    bool reversible = true;
    bool duplicate = false;
    Arrhenius rate;  // falloff: the high-pressure limit
    Arrhenius low;   // falloff: the low-pressure limit
    FalloffType falloff = FalloffType::LINDEMANN;
    std::array<double, 5> falloff_params = {};  // Troe A, T3, T1, T2; SRI a, b, c, d, e
    std::vector<std::pair<int32_t, double>> efficiencies;  // third-body efficiencies other than the default
    double default_efficiency = 1.0;
    std::vector<std::pair<double, Arrhenius>> plog;  // PLOG: (pressure [Pa], rate), by increasing pressure
    std::array<double, 4> chebyshev_range = {};     // Chebyshev: T_min, T_max [K], p_min, p_max [Pa]
    uint32_t chebyshev_n_T = 0, chebyshev_n_p = 0;
    std::vector<double> chebyshev;                  // (n_T x n_p) coefficients of log10 k, k in SI with kmol
};

/**
 * @brief One ideal-gas phase of a Cantera YAML file: its elements, species and reactions.
 */
struct Mechanism {
    std::string file;
    std::string phase;
    std::vector<std::string> elements;
    std::vector<Species> species;
    std::vector<Reaction> reactions;

    size_t n_species() const { return species.size(); }

    /** @brief Index of a species by name, or -1 if the mechanism has none. */
    int32_t species_index(const std::string & name) const;

    std::vector<std::string> species_names() const;
};

/**
 * @brief Read a phase of a Cantera YAML file.
 * @param file Path of the YAML file.
 * @param phase Name of the phase; empty for the first one.
 * @throws std::runtime_error naming the offending entry for unsupported or
 *         malformed input.
 */
Mechanism read_mechanism(const std::string & file, const std::string & phase = "");

} // namespace chemistry

#endif // CHEMISTRY_MECHANISM_H

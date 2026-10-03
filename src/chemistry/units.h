/**
 * @file units.h
 * @author Matthew Bonanni (mbonanni001@gmail.com)
 * @brief Unit conversion for Cantera YAML mechanism files.
 * @version 0.3
 * @date 2026-10-02
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 *
 */

#ifndef CHEMISTRY_UNITS_H
#define CHEMISTRY_UNITS_H

#include <array>
#include <string>

#include <yaml-cpp/yaml.h>

namespace chemistry {

/**
 * @brief Exponents of a quantity's dimension in the basis in which a YAML
 *        file sets its default units: mass, length, time, temperature,
 *        quantity, energy, pressure (e.g. a molar enthalpy is energy^1 quantity^-1).
 */
struct Dimension {
    double mass = 0.0;
    double length = 0.0;
    double time = 0.0;
    double temperature = 0.0;
    double quantity = 0.0;
    double energy = 0.0;
    double pressure = 0.0;
};

/**
 * @brief Default units of a YAML file (its `units:` map; Cantera's defaults
 *        otherwise) and conversion of values to Mallard's units: SI with
 *        quantities in kmol, as Cantera uses internally.
 *
 * A value is a number in the default units or a string "<number> <units>",
 * with units a product of known units and their SI prefixes, each optionally
 * raised to a power, joined by '*' and '/' (e.g. "cm^3/mol/s", "kJ/mol").
 */
class UnitSystem {
    public:
        UnitSystem() = default;

        /** @brief Defaults from a `units:` map; keys not given keep Cantera's defaults. */
        explicit UnitSystem(const YAML::Node & units);

        /**
         * @brief Value of a node in SI (kmol) units.
         * @param value Number or "<number> <units>" string.
         * @param dimension Dimension of the value, for numbers in default units
         *        and to check the units of strings.
         * @param what Description of the value for error messages.
         */
        double convert(const YAML::Node & value, const Dimension & dimension, const std::string & what) const;

        /**
         * @brief Activation energy divided by the gas constant, in K: a
         *        temperature, a molar energy or an energy per molecule.
         */
        double convert_activation_energy(const YAML::Node & value, const std::string & what) const;

    private:
        // SI factors of the default units, in the order of Dimension
        std::array<double, 7> defaults = {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0};
        std::string activation_energy_units = "J/kmol";
};

} // namespace chemistry

#endif // CHEMISTRY_UNITS_H

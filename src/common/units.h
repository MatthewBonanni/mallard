/**
 * @file units.h
 * @brief Physical dimensions and quantities with units in input files
 *        ("1 atm", "300 K", "5 mm", "20 1/s"), converted to SI.
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 */

#ifndef UNITS_H
#define UNITS_H

#include <array>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>

/**
 * @brief The dimension of an input key: exponents of mass, length, time,
 *        temperature, amount of substance and angle, and how a plain number
 *        given for the key is read.
 */
struct Dimension {
    enum Base { MASS, LENGTH, TIME, TEMPERATURE, AMOUNT, ANGLE, N_BASE };
    std::array<int8_t, N_BASE> exponent{};
    double plain = 1.0;               // SI value of a plain number: 1 (SI) except for keys documented otherwise
    const char * plain_unit = "SI";   // the unit of a plain number
    bool interval = false;            // a temperature difference, for which degC and degF are meaningless

    bool same_as(const Dimension & o) const { return exponent == o.exponent; }
    Dimension times(const Dimension & o, int power = 1) const;

    /** @brief "pressure (kg m^-1 s^-2)", or the SI base units alone if the dimension has no name. */
    std::string describe() const;
    /** @brief Units of this dimension to suggest, e.g. "Pa, kPa, bar, atm, psi", or empty. */
    std::string examples() const;
};

namespace dim {

constexpr Dimension of(int M, int L, int T, int Theta = 0, int N = 0, int A = 0) {
    Dimension d;
    d.exponent = {static_cast<int8_t>(M), static_cast<int8_t>(L), static_cast<int8_t>(T),
                  static_cast<int8_t>(Theta), static_cast<int8_t>(N), static_cast<int8_t>(A)};
    return d;
}

constexpr Dimension with_plain(Dimension d, double plain, const char * unit) {
    d.plain = plain;
    d.plain_unit = unit;
    return d;
}

inline constexpr Dimension none = of(0, 0, 0);
inline constexpr Dimension length = of(0, 1, 0);
inline constexpr Dimension time = of(0, 0, 1);
inline constexpr Dimension temperature = of(0, 0, 0, 1);
inline constexpr Dimension temperature_interval = [] {
    Dimension d = of(0, 0, 0, 1);
    d.interval = true;
    return d;
}();
inline constexpr Dimension velocity = of(0, 1, -1);
inline constexpr Dimension velocity_squared = of(0, 2, -2);
inline constexpr Dimension acceleration = of(0, 1, -2);
inline constexpr Dimension rate = of(0, 0, -1);
inline constexpr Dimension pressure = of(1, -1, -2);
inline constexpr Dimension density = of(1, -3, 0);
inline constexpr Dimension dynamic_viscosity = of(1, -1, -1);
inline constexpr Dimension heat_flux = of(1, 0, -3);
inline constexpr Dimension mass_flux = of(1, -2, -1);          // kg/(m^2 s), e.g. rho u
inline constexpr Dimension mass_flow = of(1, 0, -1);           // kg/s
inline constexpr Dimension mass_source = of(1, -3, -1);        // kg/(m^3 s)
inline constexpr Dimension momentum_source = of(1, -2, -2);    // N/m^3
inline constexpr Dimension power_density = of(1, -1, -3);      // W/m^3
inline constexpr Dimension angle_degrees = with_plain(of(0, 0, 0, 0, 0, 1), 3.14159265358979323846 / 180.0, "deg");
inline constexpr Dimension concentration_kmol = with_plain(of(0, -3, 0, 0, 1), 1000.0, "kmol/m^3");

} // namespace dim

/** @brief A malformed quantity or unknown unit; suggestion is a likely intended unit, or empty. */
struct UnitError : std::runtime_error {
    UnitError(const std::string & message, std::string suggestion_ = "") :
        std::runtime_error(message), suggestion(std::move(suggestion_)) {}
    std::string suggestion;
};

/** @brief A number in SI units. */
struct Quantity {
    double value = 0.0;
    Dimension dimension;
    bool has_unit = false;  // false for a plain number, whose dimension is unknown
    bool absolute = false;  // given in degC or degF (offset applied)
};

/**
 * @brief Parse "<number> [unit]", e.g. "1 atm", "300 K", "-20 degC", "5mm",
 *        "9.81 m/s^2", "1.8e-5 Pa s", "8.314 J/(mol K)", converted to SI.
 *        Units are SI symbols with prefixes (Y to y; u or µ for micro), and
 *        atm, bar, psi, Torr, in, ft, min, h, lb, lbf, cal, L, degC, degF,
 *        degR, deg, rad. Products are written "a b" or "a*b", powers "a^n",
 *        quotients "a/b" (one following unit, or a parenthesized group).
 * @throws UnitError on malformed text or an unknown unit
 */
Quantity parse_quantity(const std::string & text);

/**
 * @brief Whether a string is meant as a quantity with a unit rather than an
 *        expression: a number followed by unit symbols and unit operators only,
 *        none of them the expression variables x, y, z and t or a function call.
 *        A plain number is not (it stays an expression where one is accepted).
 */
bool has_unit(const std::string & text);

/**
 * @brief For an expression that failed to compile: a note that expressions
 *        are in SI units, if a unit symbol appears in it, else empty.
 */
std::string expression_unit_hint(const std::string & expression);

#endif // UNITS_H

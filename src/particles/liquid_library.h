/**
 * @file liquid_library.h
 * @author Matthew Bonanni (mbonanni001@gmail.com)
 * @brief The library of named liquids for droplets: the fluids, and the
 *        tables of their saturated-liquid properties against temperature,
 *        fitted on the host at setup to a tolerance from a property source
 *        (CoolProp) for the device. Design: docs/design/liquid_library.md.
 * @version 0.1
 * @date 2026-11-25
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 *
 */

#ifndef LIQUID_LIBRARY_H
#define LIQUID_LIBRARY_H

#include <cstdint>
#include <functional>
#include <string>
#include <vector>

#include "liquid_curve.h"

namespace particles {

/** @brief A fluid of the library: its name in input files and in CoolProp, and the sources of its equations. */
struct LibraryFluid {
    const char * name;
    const char * coolprop;
    const char * sources;  // equation of state; viscosity; conductivity; surface tension
};

/** @brief The fluids of the library, by name. */
const std::vector<LibraryFluid> & library_fluids();

/** @brief The library fluid of this name, or nullptr. */
const LibraryFluid * find_library_fluid(const std::string & name);

/** @brief Whether this build has CoolProp (Mallard_ENABLE_COOLPROP), and its version ("" without). */
bool coolprop_available();
std::string coolprop_version();

/** @brief The saturated liquid at a temperature. */
struct LiquidState {
    double density = 0.0;
    double specific_heat = 0.0;   // c_p
    double latent_heat = 0.0;     // h of the saturated vapor less h of the saturated liquid
    double saturation_pressure = 0.0;
    double viscosity = 0.0;
    double conductivity = 0.0;
    double surface_tension = 0.0;
};

/** @brief A source of a fluid's saturated-liquid properties between its triple and critical points. */
struct PropertySource {
    double T_triple = 0.0;
    double T_critical = 0.0;
    double p_critical = 0.0;
    double T_limit = 0.0;  // the highest temperature at which every property exists; 0: T_critical
    std::function<std::vector<LiquidState>(const std::vector<double> & T)> states;
    std::function<double(double p)> saturation_temperature;
};

/** @brief CoolProp's properties of a fluid by its CoolProp name; throws std::runtime_error without CoolProp. */
PropertySource coolprop_source(const std::string & coolprop_name);

/** @brief Samples y at points s, and dy/ds if slopes (else slope may be left empty). */
using Sampler = std::function<void(const std::vector<double> & s, bool slopes, std::vector<double> & y,
                                   std::vector<double> & slope)>;

/** @brief How a curve is fitted and how its error is measured. */
struct FitOptions {
    double a = 0.05;           // Curve::a
    double tolerance = 1e-6;   // on the error between the knots (at the quarters of their intervals)
    bool relative = true;      // error |dy| / max(|y|, floor); else |dy| (y a logarithm: the relative error of e^y)
    double floor = 0.0;
    bool check_slope = false;  // the slope's relative error too
    uint32_t max_knots = 8193;
};

/** @brief A curve fitted on the host: its knots in table (offset 0), and its largest error between them. */
struct CurveFit {
    Curve curve;
    std::vector<double> table;
    double error = 0.0;
    double slope_error = 0.0;

    double value(const double s) const { return curve.value(table.data(), s); }
    double evaluate(const double s, double & slope) const { return curve.evaluate(table.data(), s, slope); }
};

/**
 * @brief The curve of f on [s_min, s_max] with the fewest knots, 2^m + 1 from 9 up to options.max_knots,
 *        whose error at the quarters of its intervals is at most options.tolerance.
 */
CurveFit fit_curve(const Sampler & f, double s_min, double s_max, const FitOptions & options);

/** @brief The curve of the inverse of an increasing fitted curve over its range of values. */
CurveFit fit_inverse(const CurveFit & forward, const FitOptions & options);

/** @brief A tabulated property: its name, knots and largest relative error. */
struct TableReport {
    std::string name;
    uint32_t knots = 0;
    double error = 0.0;
};

/**
 * @brief The tables of a liquid between T_min (its triple point) and T_max (just below its critical point),
 *        in one table of values: density, enthalpy above T_ref (slope: c_p) and its inverse, latent heat,
 *        ln of the saturation pressure and its inverse, ln of the viscosity, conductivity, surface tension.
 */
struct LiquidTables {
    std::vector<double> table;
    Curve density, enthalpy, temperature, latent, pressure, boiling, viscosity, conductivity, tension;
    double T_min = 0.0, T_max = 0.0, T_critical = 0.0, p_critical = 0.0;
    double T_ref = 0.0;
    std::vector<TableReport> report;
};

/** @brief How far below the critical temperature the tables end, relative to it. */
inline constexpr double CRITICAL_MARGIN = 1e-3;

/** @brief Where the tables of source end: CRITICAL_MARGIN below T_limit, or T_critical without one. */
inline double table_end(const PropertySource & source) {
    return (source.T_limit > 0.0 ? source.T_limit : source.T_critical) * (1.0 - CRITICAL_MARGIN);
}

/** @brief The tables of the liquid of source, its enthalpy from T_ref, each fitted to tolerance. */
LiquidTables tabulate(const PropertySource & source, double T_ref, double tolerance);

} // namespace particles

#endif // LIQUID_LIBRARY_H

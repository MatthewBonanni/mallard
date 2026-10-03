/**
 * @file units.cpp
 * @author Matthew Bonanni (mbonanni001@gmail.com)
 * @brief Unit conversion for Cantera YAML mechanism files.
 * @version 0.3
 * @date 2026-10-02
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 *
 */

#include "units.h"

#include <cmath>
#include <cstdlib>
#include <stdexcept>
#include <tuple>
#include <unordered_map>
#include <utility>

#include "mechanism.h"

namespace chemistry {

namespace {

// Exponents of kg, m, s, K, kmol
using BaseDims = std::array<double, 5>;

struct Unit {
    double factor;
    BaseDims dims;
};

constexpr BaseDims MASS = {1, 0, 0, 0, 0};
constexpr BaseDims LENGTH = {0, 1, 0, 0, 0};
constexpr BaseDims TIME = {0, 0, 1, 0, 0};
constexpr BaseDims TEMPERATURE = {0, 0, 0, 1, 0};
constexpr BaseDims QUANTITY = {0, 0, 0, 0, 1};
constexpr BaseDims ENERGY = {1, 2, -2, 0, 0};
constexpr BaseDims FORCE = {1, 1, -2, 0, 0};
constexpr BaseDims PRESSURE = {1, -1, -2, 0, 0};
constexpr BaseDims NONE = {0, 0, 0, 0, 0};

const std::unordered_map<std::string, Unit> & known_units() {
    static const std::unordered_map<std::string, Unit> units = {
        {"g", {1e-3, MASS}},
        {"m", {1.0, LENGTH}},
        {"Angstrom", {1e-10, LENGTH}},
        {"angstrom", {1e-10, LENGTH}},
        {"s", {1.0, TIME}},
        {"min", {60.0, TIME}},
        {"hr", {3600.0, TIME}},
        {"K", {1.0, TEMPERATURE}},
        {"mol", {1e-3, QUANTITY}},
        {"gmol", {1e-3, QUANTITY}},
        {"molec", {1.0 / AVOGADRO, QUANTITY}},
        {"J", {1.0, ENERGY}},
        {"cal", {4.184, ENERGY}},
        {"erg", {1e-7, ENERGY}},
        {"eV", {ELEMENTARY_CHARGE, ENERGY}},
        {"N", {1.0, FORCE}},
        {"dyn", {1e-5, FORCE}},
        {"Pa", {1.0, PRESSURE}},
        {"atm", {ONE_ATM, PRESSURE}},
        {"bar", {1e5, PRESSURE}},
        {"torr", {ONE_ATM / 760.0, PRESSURE}},
    };
    return units;
}

const std::unordered_map<std::string, double> & prefixes() {
    static const std::unordered_map<std::string, double> p = {
        {"Y", 1e24}, {"Z", 1e21}, {"E", 1e18}, {"P", 1e15}, {"T", 1e12}, {"G", 1e9},  {"M", 1e6},
        {"k", 1e3},  {"h", 1e2},  {"da", 1e1}, {"d", 1e-1}, {"c", 1e-2}, {"m", 1e-3}, {"u", 1e-6},
        {"n", 1e-9}, {"p", 1e-12}, {"f", 1e-15}, {"a", 1e-18},
    };
    return p;
}

Unit named_unit(const std::string & name) {
    if (name == "1") return {1.0, NONE};
    const auto & units = known_units();
    if (auto it = units.find(name); it != units.end()) return it->second;
    for (size_t n : {size_t(2), size_t(1)}) {
        if (name.size() <= n) continue;
        const auto p = prefixes().find(name.substr(0, n));
        const auto u = units.find(name.substr(n));
        if (p != prefixes().end() && u != units.end()) return {p->second * u->second.factor, u->second.dims};
    }
    throw std::runtime_error("unknown unit \"" + name + "\"");
}

/** @brief SI factor and base dimensions of a unit expression such as "cm^3/mol/s". */
Unit parse_units(const std::string & text) {
    Unit result{1.0, NONE};
    size_t pos = 0;
    double sign = 1.0;
    std::string expression;
    for (char c : text) {
        if (c != ' ') expression += c;
    }
    if (expression.empty()) throw std::runtime_error("empty unit expression");
    while (pos <= expression.size()) {
        const size_t end = expression.find_first_of("*/", pos);
        const std::string factor = expression.substr(pos, end == std::string::npos ? std::string::npos : end - pos);
        const size_t caret = factor.find('^');
        const std::string name = factor.substr(0, caret);
        double power = 1.0;
        if (caret != std::string::npos) {
            char * stop = nullptr;
            const std::string exponent = factor.substr(caret + 1);
            power = std::strtod(exponent.c_str(), &stop);
            if (exponent.empty() || *stop != '\0') throw std::runtime_error("bad exponent in \"" + text + "\"");
        }
        const Unit u = named_unit(name);
        result.factor *= std::pow(u.factor, sign * power);
        for (size_t i = 0; i < result.dims.size(); i++) result.dims[i] += sign * power * u.dims[i];
        if (end == std::string::npos) break;
        sign = expression[end] == '/' ? -1.0 : 1.0;
        pos = end + 1;
    }
    return result;
}

BaseDims base_dims(const Dimension & d) {
    BaseDims b{};
    const std::pair<double, BaseDims> parts[] = {{d.mass, MASS},         {d.length, LENGTH},     {d.time, TIME},
                                                 {d.temperature, TEMPERATURE}, {d.quantity, QUANTITY},
                                                 {d.energy, ENERGY},     {d.pressure, PRESSURE}};
    for (const auto & [power, dims] : parts) {
        for (size_t i = 0; i < b.size(); i++) b[i] += power * dims[i];
    }
    return b;
}

bool same_dims(const BaseDims & a, const BaseDims & b) {
    for (size_t i = 0; i < a.size(); i++) {
        if (std::abs(a[i] - b[i]) > 1e-12) return false;
    }
    return true;
}

/** @brief Whether text is a complete number, which is then stored in value. */
bool parse_number(const std::string & text, double & value) {
    char * stop = nullptr;
    value = std::strtod(text.c_str(), &stop);
    return stop != text.c_str() && *stop == '\0';
}

/** @brief Split "<number> <units>" into the number and the units. */
std::pair<double, std::string> split_value(const std::string & text, const std::string & what) {
    char * stop = nullptr;
    const double value = std::strtod(text.c_str(), &stop);
    if (stop == text.c_str()) throw std::runtime_error(what + ": \"" + text + "\" is not a number with units");
    std::string units(stop);
    units.erase(0, units.find_first_not_of(' '));
    return {value, units};
}

} // namespace

UnitSystem::UnitSystem(const YAML::Node & units) {
    if (!units) return;
    if (!units.IsMap()) throw std::runtime_error("units: expected a map");
    const std::pair<const char *, size_t> keys[] = {{"mass", 0},     {"length", 1},   {"time", 2},     {"temperature", 3},
                                                    {"quantity", 4}, {"energy", 5},   {"pressure", 6}};
    for (const auto & [key, index] : keys) {
        if (!units[key]) continue;
        const std::string text = units[key].as<std::string>();
        try {
            defaults[index] = parse_units(text).factor;
        } catch (const std::runtime_error & e) {
            throw std::runtime_error(std::string("units.") + key + ": " + e.what());
        }
    }
    if (units["activation-energy"]) {
        activation_energy_units = units["activation-energy"].as<std::string>();
    } else {
        activation_energy_units.clear();
    }
}

double UnitSystem::convert(const YAML::Node & value, const Dimension & dimension, const std::string & what) const {
    if (!value || !value.IsScalar()) throw std::runtime_error(what + ": expected a value");
    const std::string text = value.Scalar();
    double number;
    if (parse_number(text, number)) {
        const double powers[7] = {dimension.mass,     dimension.length, dimension.time,    dimension.temperature,
                                  dimension.quantity, dimension.energy, dimension.pressure};
        double factor = 1.0;
        for (size_t i = 0; i < defaults.size(); i++) {
            if (powers[i] != 0.0) factor *= std::pow(defaults[i], powers[i]);
        }
        return number * factor;
    }
    const auto [x, units] = split_value(text, what);
    Unit u{1.0, NONE};
    try {
        u = parse_units(units);
    } catch (const std::runtime_error & e) {
        throw std::runtime_error(what + ": " + e.what());
    }
    if (!same_dims(u.dims, base_dims(dimension))) {
        throw std::runtime_error(what + ": units \"" + units + "\" have the wrong dimension");
    }
    return x * u.factor;
}

double UnitSystem::convert_activation_energy(const YAML::Node & value, const std::string & what) const {
    if (!value || !value.IsScalar()) throw std::runtime_error(what + ": expected a value");
    const std::string text = value.Scalar();
    double x;
    Unit u{1.0, NONE};
    if (parse_number(text, x)) {
        if (activation_energy_units.empty()) {
            // Cantera's default: the file's energy per quantity
            return x * defaults[5] / defaults[4] / GAS_CONSTANT;
        }
        u = parse_units(activation_energy_units);
    } else {
        std::string units;
        std::tie(x, units) = split_value(text, what);
        try {
            u = parse_units(units);
        } catch (const std::runtime_error & e) {
            throw std::runtime_error(what + ": " + e.what());
        }
    }
    if (same_dims(u.dims, TEMPERATURE)) return x * u.factor;
    BaseDims molar = ENERGY;
    molar[4] = -1.0;
    if (same_dims(u.dims, molar)) return x * u.factor / GAS_CONSTANT;
    if (same_dims(u.dims, ENERGY)) return x * u.factor / BOLTZMANN;
    throw std::runtime_error(what + ": activation energy units must be a temperature, energy per quantity or energy");
}

} // namespace chemistry

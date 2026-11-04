/**
 * @file units.cpp
 * @brief Parsing of quantities with units, and names of dimensions.
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 */

#include "units.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdlib>
#include <numbers>
#include <string_view>
#include <vector>

Dimension Dimension::times(const Dimension & o, const int power) const {
    Dimension d;
    for (size_t i = 0; i < N_BASE; i++) d.exponent[i] = static_cast<int8_t>(exponent[i] + power * o.exponent[i]);
    return d;
}

namespace {

struct NamedDimension {
    const char * name;
    Dimension dimension;
    const char * examples;
};

const NamedDimension NAMED[] = {
    {"dimensionless", dim::none, ""},
    {"length", dim::length, "m, mm, cm, km, in, ft"},
    {"time", dim::time, "s, ms, us, min, h"},
    {"mass", dim::of(1, 0, 0), "kg, g, lb"},
    {"temperature", dim::temperature, "K, degC, degF"},
    {"amount of substance", dim::of(0, 0, 0, 0, 1), "mol, kmol"},
    {"angle", dim::of(0, 0, 0, 0, 0, 1), "deg, rad"},
    {"velocity", dim::velocity, "m/s, km/h, ft/s"},
    {"squared velocity", dim::velocity_squared, "m^2/s^2, J/kg"},
    {"acceleration", dim::acceleration, "m/s^2, ft/s^2"},
    {"rate", dim::rate, "1/s, Hz, kHz"},
    {"pressure", dim::pressure, "Pa, kPa, MPa, bar, atm, psi"},
    {"density", dim::density, "kg/m^3, g/cm^3"},
    {"dynamic viscosity", dim::dynamic_viscosity, "Pa s, kg/(m s)"},
    {"heat flux", dim::heat_flux, "W/m^2, kW/m^2"},
    {"mass flux", dim::mass_flux, "kg/(m^2 s)"},
    {"mass flow", dim::mass_flow, "kg/s, g/s"},
    {"mass source", dim::mass_source, "kg/(m^3 s)"},
    {"momentum source", dim::momentum_source, "N/m^3"},
    {"power density", dim::power_density, "W/m^3"},
    {"concentration", dim::concentration_kmol, "mol/m^3, kmol/m^3"},
    {"area", dim::of(0, 2, 0), "m^2, mm^2"},
    {"volume", dim::of(0, 3, 0), "m^3, L"},
    {"energy", dim::of(1, 2, -2), "J, kJ, cal"},
    {"power", dim::of(1, 2, -3), "W, kW"},
    {"force", dim::of(1, 1, -2), "N, kN, lbf"},
    {"kinematic viscosity", dim::of(0, 2, -1), "m^2/s"},
    {"specific heat", dim::of(0, 2, -2, -1), "J/(kg K)"},
    {"molar mass", dim::of(1, 0, 0, 0, -1), "kg/mol, g/mol"},
};

const NamedDimension * named(const Dimension & d) {
    for (const auto & n : NAMED) {
        if (n.dimension.same_as(d)) return &n;
    }
    return nullptr;
}

std::string base_units(const Dimension & d, const char * power = "^") {
    static const char * symbols[Dimension::N_BASE] = {"kg", "m", "s", "K", "mol", "rad"};
    std::string text;
    for (size_t i = 0; i < Dimension::N_BASE; i++) {
        if (d.exponent[i] == 0) continue;
        text += (text.empty() ? "" : " ") + std::string(symbols[i]);
        if (d.exponent[i] != 1) text += power + std::to_string(d.exponent[i]);
    }
    return text.empty() ? "1" : text;
}

struct UnitDef {
    std::string_view symbol;
    double factor;  // SI value of one unit
    Dimension dimension;
    bool prefixable;
    double offset = 0.0;  // absolute temperatures: SI = (value + offset) * factor
    bool affine = false;
};

constexpr double PI = std::numbers::pi;

const UnitDef UNITS[] = {
    {"m", 1.0, dim::length, true},
    {"g", 1.0e-3, dim::of(1, 0, 0), true},
    {"s", 1.0, dim::time, true},
    {"K", 1.0, dim::temperature, true},
    {"mol", 1.0, dim::of(0, 0, 0, 0, 1), true},
    {"rad", 1.0, dim::of(0, 0, 0, 0, 0, 1), true},
    {"Pa", 1.0, dim::pressure, true},
    {"J", 1.0, dim::of(1, 2, -2), true},
    {"W", 1.0, dim::of(1, 2, -3), true},
    {"N", 1.0, dim::of(1, 1, -2), true},
    {"Hz", 1.0, dim::rate, true},
    {"bar", 1.0e5, dim::pressure, true},
    {"L", 1.0e-3, dim::of(0, 3, 0), true},
    {"cal", 4.184, dim::of(1, 2, -2), true},
    {"atm", 101325.0, dim::pressure, false},
    {"psi", 6894.757293168361, dim::pressure, false},
    {"Torr", 101325.0 / 760.0, dim::pressure, false},
    {"mmHg", 133.322387415, dim::pressure, false},
    {"in", 0.0254, dim::length, false},
    {"ft", 0.3048, dim::length, false},
    {"min", 60.0, dim::time, false},
    {"h", 3600.0, dim::time, false},
    {"lb", 0.45359237, dim::of(1, 0, 0), false},
    {"lbf", 4.4482216152605, dim::of(1, 1, -2), false},
    {"deg", PI / 180.0, dim::of(0, 0, 0, 0, 0, 1), false},
    {"°", PI / 180.0, dim::of(0, 0, 0, 0, 0, 1), false},
    {"degR", 5.0 / 9.0, dim::temperature, false},
    {"degC", 1.0, dim::temperature, false, 273.15, true},
    {"°C", 1.0, dim::temperature, false, 273.15, true},
    {"degF", 5.0 / 9.0, dim::temperature, false, 459.67, true},
    {"°F", 5.0 / 9.0, dim::temperature, false, 459.67, true},
};

struct Prefix {
    std::string_view symbol;
    double factor;
};

const Prefix PREFIXES[] = {
    {"da", 1e1}, {"Y", 1e24}, {"Z", 1e21}, {"E", 1e18}, {"P", 1e15}, {"T", 1e12}, {"G", 1e9},
    {"M", 1e6}, {"k", 1e3}, {"h", 1e2}, {"d", 1e-1}, {"c", 1e-2}, {"m", 1e-3}, {"u", 1e-6},
    {"µ", 1e-6}, {"μ", 1e-6}, {"n", 1e-9}, {"p", 1e-12}, {"f", 1e-15}, {"a", 1e-18},
    {"z", 1e-21}, {"y", 1e-24},
};

// Common misnomers, suggested when a name is not a unit
const std::pair<std::string_view, std::string_view> ALIASES[] = {
    {"sec", "s"}, {"secs", "s"}, {"second", "s"}, {"seconds", "s"}, {"hr", "h"}, {"hrs", "h"},
    {"hour", "h"}, {"hours", "h"}, {"mins", "min"}, {"minute", "min"}, {"minutes", "min"},
    {"C", "degC"}, {"F", "degF"}, {"R", "degR"}, {"celsius", "degC"}, {"fahrenheit", "degF"},
    {"kelvin", "K"}, {"meter", "m"}, {"meters", "m"}, {"metre", "m"}, {"metres", "m"},
    {"micron", "um"}, {"microns", "um"}, {"inch", "in"}, {"inches", "in"}, {"foot", "ft"},
    {"feet", "ft"}, {"lbm", "lb"}, {"psia", "psi"}, {"pascal", "Pa"}, {"degree", "deg"},
    {"degrees", "deg"}, {"liter", "L"}, {"litre", "L"}, {"l", "L"}, {"gram", "g"}, {"grams", "g"},
};

// Prefixed forms worth suggesting for a misspelled unit
const std::string_view COMMON_PREFIXED[] = {"kPa", "MPa", "GPa", "hPa", "mbar", "mm", "cm", "km", "um", "nm",
                                            "ms", "us", "ns", "kg", "mg", "kmol", "kJ", "MJ", "kW", "MW",
                                            "kHz", "MHz", "kN", "mL", "kcal"};

struct Found {
    double factor;
    Dimension dimension;
    const UnitDef * def;
};

bool lookup(const std::string_view name, Found & out) {
    for (const auto & u : UNITS) {
        if (u.symbol == name) {
            out = {u.factor, u.dimension, &u};
            return true;
        }
    }
    for (const auto & p : PREFIXES) {
        if (name.size() <= p.symbol.size() || name.substr(0, p.symbol.size()) != p.symbol) continue;
        const std::string_view rest = name.substr(p.symbol.size());
        for (const auto & u : UNITS) {
            if (u.prefixable && u.symbol == rest) {
                out = {p.factor * u.factor, u.dimension, &u};
                return true;
            }
        }
    }
    return false;
}

std::string lower(std::string_view s) {
    std::string out(s);
    for (char & c : out) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    return out;
}

size_t edit_distance(const std::string & a, const std::string & b) {
    std::vector<size_t> row(b.size() + 1);
    for (size_t j = 0; j <= b.size(); j++) row[j] = j;
    for (size_t i = 1; i <= a.size(); i++) {
        size_t diagonal = row[0];
        row[0] = i;
        for (size_t j = 1; j <= b.size(); j++) {
            const size_t above = row[j];
            row[j] = std::min({row[j] + 1, row[j - 1] + 1, diagonal + (a[i - 1] == b[j - 1] ? 0 : 1)});
            diagonal = above;
        }
    }
    return row[b.size()];
}

std::string suggest_unit(const std::string & name) {
    for (const auto & [alias, unit] : ALIASES) {
        if (alias == name) return std::string(unit);
    }
    // "m2" for "m^2"
    const size_t digits = name.find_first_of("0123456789");
    Found found;
    if (digits != std::string::npos && digits > 0 && lookup(std::string_view(name).substr(0, digits), found)) {
        return name.substr(0, digits) + "^" + name.substr(digits);
    }
    std::vector<std::string_view> candidates;
    for (const auto & u : UNITS) candidates.push_back(u.symbol);
    for (const auto & p : COMMON_PREFIXED) candidates.push_back(p);
    for (const auto & [alias, unit] : ALIASES) {
        if (alias.size() > 2 && lower(alias) == lower(name)) return std::string(unit);
    }
    std::string best;
    size_t best_distance = std::max<size_t>(1, name.size() / 3) + 1;
    for (const auto & c : candidates) {
        // A case slip ("kpa", "PA") is the likeliest mistake
        const size_t distance = lower(c) == lower(name) ? 0 : edit_distance(lower(c), lower(name)) + 1;
        if (distance < best_distance) {
            best_distance = distance;
            best = std::string(c);
        }
    }
    return best;
}

bool starts_with(const std::string & s, size_t pos, std::string_view what) {
    return s.compare(pos, what.size(), what) == 0;
}

// Bytes that continue a unit symbol: letters, and UTF-8 characters other than
// the operators and superscripts below
bool symbol_char(const std::string & s, size_t pos) {
    const unsigned char c = static_cast<unsigned char>(s[pos]);
    if (std::isalpha(c)) return true;
    if (c < 0x80) return false;
    return !starts_with(s, pos, "·") && !starts_with(s, pos, "²") && !starts_with(s, pos, "³");
}

/** @brief Length of the number at the start of s, or 0. */
size_t number_length(const std::string & s) {
    size_t i = 0;
    if (i < s.size() && (s[i] == '+' || s[i] == '-')) i++;
    const size_t start = i;
    while (i < s.size() && std::isdigit(static_cast<unsigned char>(s[i]))) i++;
    if (i < s.size() && s[i] == '.') i++;
    while (i < s.size() && std::isdigit(static_cast<unsigned char>(s[i]))) i++;
    if (i == start || (i == start + 1 && s[start] == '.')) return 0;
    if (i < s.size() && (s[i] == 'e' || s[i] == 'E')) {
        size_t j = i + 1;
        if (j < s.size() && (s[j] == '+' || s[j] == '-')) j++;
        if (j < s.size() && std::isdigit(static_cast<unsigned char>(s[j]))) {
            while (j < s.size() && std::isdigit(static_cast<unsigned char>(s[j]))) j++;
            i = j;
        }
    }
    return i;
}

struct Unit {
    double factor = 1.0;
    Dimension dimension;
};

class UnitParser {
    public:
        UnitParser(const std::string & text_, const std::string & quantity_) : text(text_), quantity(quantity_) {}

        Unit parse() {
            Unit u = product();
            skip_spaces();
            if (pos < text.size()) fail("unexpected \"" + text.substr(pos) + "\"");
            return u;
        }

    private:
        const std::string & text;
        const std::string & quantity;
        size_t pos = 0;

        [[noreturn]] void fail(const std::string & what, const std::string & suggestion = "") const {
            throw UnitError("\"" + quantity + "\": " + what + ".", suggestion);
        }

        bool skip_spaces() {
            const size_t start = pos;
            while (pos < text.size() && text[pos] == ' ') pos++;
            return pos > start;
        }

        Unit product() {
            Unit result;
            bool first = true, after_division = false;
            while (true) {
                const bool space = skip_spaces();
                if (pos >= text.size() || text[pos] == ')') break;
                bool divide = false;
                if (!first) {
                    if (text[pos] == '/') {
                        divide = true;
                        pos++;
                    } else if (text[pos] == '*') {
                        pos++;
                    } else if (starts_with(text, pos, "·")) {
                        pos += 2;
                    } else if (!space) {
                        fail("unexpected \"" + text.substr(pos) + "\"");
                    }
                    if (!divide && after_division) {
                        fail("ambiguous after \"/\"; group the denominator, e.g. J/(kg K), or write J K^-1 kg^-1");
                    }
                    skip_spaces();
                }
                const Unit f = factor();
                result.factor = divide ? result.factor / f.factor : result.factor * f.factor;
                result.dimension = result.dimension.times(f.dimension, divide ? -1 : 1);
                after_division = after_division || divide;
                first = false;
            }
            if (first) fail("expected a unit");
            return result;
        }

        Unit factor() {
            Unit base;
            if (pos < text.size() && text[pos] == '(') {
                pos++;
                base = product();
                if (pos >= text.size() || text[pos] != ')') fail("missing \")\"");
                pos++;
            } else if (pos < text.size() && text[pos] == '1' &&
                       (pos + 1 == text.size() || !std::isdigit(static_cast<unsigned char>(text[pos + 1])))) {
                pos++;
            } else {
                const size_t start = pos;
                while (pos < text.size() && symbol_char(text, pos)) pos++;
                while (pos < text.size() && std::isdigit(static_cast<unsigned char>(text[pos]))) pos++;
                const std::string name = text.substr(start, pos - start);
                if (name.empty()) fail("expected a unit at \"" + text.substr(start) + "\"");
                Found found;
                if (!lookup(name, found)) {
                    const std::string suggestion = suggest_unit(name);
                    fail("unknown unit \"" + name + "\"" + (suggestion.empty() ? "" : " (did you mean " + suggestion + "?)"),
                         suggestion);
                }
                if (found.def->affine) {
                    fail(name + " is an absolute temperature and stands alone (\"20 " + name +
                         "\"); in compound units use K, or degR for Fahrenheit-sized degrees");
                }
                base = {found.factor, found.dimension};
            }
            const int power = exponent();
            Unit u;
            u.factor = std::pow(base.factor, power);
            u.dimension = Dimension().times(base.dimension, power);
            return u;
        }

        int exponent() {
            if (starts_with(text, pos, "²")) {
                pos += 2;
                return 2;
            }
            if (starts_with(text, pos, "³")) {
                pos += 2;
                return 3;
            }
            if (starts_with(text, pos, "**")) {
                pos += 2;
            } else if (pos < text.size() && text[pos] == '^') {
                pos++;
            } else {
                return 1;
            }
            const bool parenthesized = pos < text.size() && text[pos] == '(';
            if (parenthesized) pos++;
            int sign = 1;
            if (pos < text.size() && (text[pos] == '-' || text[pos] == '+')) sign = text[pos++] == '-' ? -1 : 1;
            const size_t start = pos;
            while (pos < text.size() && std::isdigit(static_cast<unsigned char>(text[pos]))) pos++;
            if (pos == start || pos - start > 2) fail("expected an integer power after \"^\"");
            if (parenthesized) {
                if (pos >= text.size() || text[pos] != ')') fail("missing \")\"");
                pos++;
            }
            return sign * std::stoi(text.substr(start, pos - start));
        }
};

std::string trim(const std::string & s) {
    const size_t first = s.find_first_not_of(" \t");
    if (first == std::string::npos) return "";
    return s.substr(first, s.find_last_not_of(" \t") - first + 1);
}

} // namespace

std::string Dimension::describe() const {
    const NamedDimension * n = named(*this);
    if (n && same_as(dim::none)) return "dimensionless";
    if (n) return std::string(n->name) + " (" + base_units(*this) + ")";
    return base_units(*this);
}

std::string Dimension::udunits() const { return base_units(*this, ""); }

std::string Dimension::examples() const {
    const NamedDimension * n = named(*this);
    return n ? n->examples : "";
}

Quantity parse_quantity(const std::string & text_in) {
    const std::string text = trim(text_in);
    const size_t n = number_length(text);
    if (n == 0) throw UnitError("\"" + text + "\": expected a number, then optionally a unit (e.g. \"300 K\").");
    Quantity q;
    q.value = std::strtod(text.substr(0, n).c_str(), nullptr);
    const std::string unit = trim(text.substr(n));
    if (unit.empty()) return q;
    q.has_unit = true;
    for (const auto & u : UNITS) {
        if (u.affine && u.symbol == unit) {
            q.value = (q.value + u.offset) * u.factor;
            q.dimension = u.dimension;
            q.absolute = true;
            return q;
        }
    }
    const Unit parsed = UnitParser(unit, text).parse();
    q.value *= parsed.factor;
    q.dimension = parsed.dimension;
    return q;
}

bool has_unit(const std::string & text_in) {
    const std::string text = trim(text_in);
    const size_t n = number_length(text);
    if (n == 0) return false;
    const size_t first = text.find_first_not_of(' ', n);
    if (first == std::string::npos) return false;
    // "2 * 3" and "2^x" are expressions; "1/s" and "(kg m)/s" units
    if (!symbol_char(text, first) && text[first] != '(' && !starts_with(text, first, "1/")) return false;
    bool after_power = false, symbol = false;
    for (size_t i = first; i < text.size();) {
        const char c = text[i];
        if (symbol_char(text, i)) {
            const size_t start = i;
            while (i < text.size() && (symbol_char(text, i) || std::isdigit(static_cast<unsigned char>(text[i])))) i++;
            const std::string name = text.substr(start, i - start);
            for (const char * reserved : {"x", "y", "z", "t", "pi", "epsilon", "inf"}) {
                if (name == reserved) return false;
            }
            if (i < text.size() && text[i] == '(') return false;  // a function call
            after_power = false;
            symbol = true;
            continue;
        }
        const bool sign = (c == '-' || c == '+') && after_power;
        if (!(std::isdigit(static_cast<unsigned char>(c)) || c == ' ' || c == '^' || c == '*' || c == '/' ||
              c == '(' || c == ')' || sign || static_cast<unsigned char>(c) >= 0x80)) {
            return false;
        }
        after_power = c == '^' || (c == '(' && after_power);
        i++;
    }
    return symbol;
}

std::string expression_unit_hint(const std::string & expression) {
    for (size_t i = 0; i < expression.size();) {
        if (!symbol_char(expression, i)) {
            i++;
            continue;
        }
        const size_t start = i;
        while (i < expression.size() && (symbol_char(expression, i) || std::isdigit(static_cast<unsigned char>(expression[i])))) i++;
        const std::string name = expression.substr(start, i - start);
        Found found;
        const bool call = i < expression.size() && expression[i] == '(';
        if (!call && name != "x" && name != "y" && name != "z" && name != "t" && lookup(name, found)) {
            return " (expressions are in SI units and take no unit symbols such as \"" + name +
                   "\"; a unit is accepted after a single number, e.g. \"300 K\")";
        }
    }
    return "";
}

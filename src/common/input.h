/**
 * @file input.h
 * @author Matthew Bonanni (mbonanni001@gmail.com)
 * @brief Lookup of real-valued input parameters.
 * @version 0.1
 * @date 2026-10-02
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 *
 */

#ifndef INPUT_H
#define INPUT_H

#include <algorithm>
#include <cstddef>
#include <memory>
#include <set>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include <toml.hpp>

#include "common_typedef.h"
#include "units.h"

/**
 * @brief A TOML table that records which keys the program looked up, present
 *        or not. Copies share the record, so a key read through a copy counts
 *        as read. Iterating the table counts every key as read. Mallard code
 *        holds the input only as Input: a conversion to toml::value would
 *        iterate it (src/CMakeLists.txt rejects toml::value).
 */
template <typename K, typename V>
class InputTable : public std::unordered_map<K, V> {
        using Base = std::unordered_map<K, V>;

    public:
        struct Lookups {
            std::set<K> keys;
            bool all = false;
        };

        using Base::Base;
        InputTable() = default;

        auto find(const K & k) { note(k); return Base::find(k); }
        auto find(const K & k) const { note(k); return Base::find(k); }
        V & at(const K & k) { note(k); return Base::at(k); }
        const V & at(const K & k) const { note(k); return Base::at(k); }
        std::size_t count(const K & k) const { note(k); return Base::count(k); }
        bool contains(const K & k) const { note(k); return Base::contains(k); }
        V & operator[](const K & k) { note(k); return Base::operator[](k); }

        auto begin() { lookups->all = true; return Base::begin(); }
        auto begin() const { lookups->all = true; return Base::begin(); }
        auto cbegin() const { lookups->all = true; return Base::cbegin(); }

        /** @brief The entries, without counting them as read. */
        const Base & entries() const { return *this; }
        Base & entries() { return *this; }
        const Lookups & lookups_made() const { return *lookups; }
        void forget_lookups() { lookups = std::make_shared<Lookups>(); }

    private:
        void note(const K & k) const { lookups->keys.insert(k); }

        std::shared_ptr<Lookups> lookups = std::make_shared<Lookups>();
};

struct InputConfig : toml::type_config {
    template <typename K, typename V>
    using table_type = InputTable<K, V>;
};

/**
 * @brief A parsed input file. Every read goes through InputTable, so the keys
 *        no setup code read are known afterwards (see unused_keys).
 */
using Input = toml::basic_value<InputConfig>;

/** @brief Parse an input file, with no key counted as read. */
Input parse_input(const std::string & filename);
/** @brief Parse TOML text, with no key counted as read. */
Input parse_input_string(const std::string & text, const std::string & name = "input");

/** @brief A problem with one key of the input, as normal runs and --check report it. */
struct Diagnostic {
    std::string key;                  // e.g. "mesh.Nxx" or "boundaries[2].tpye"
    std::size_t line = 0;             // 0 if unknown
    std::string message;              // what is wrong
    std::vector<std::string> valid;   // accepted values, if a closed set
    std::string suggestion;           // full path of the likely intended key (or the unit), or empty
    std::string expected;             // what the key takes, e.g. "pressure (kg m^-1 s^-2): Pa, kPa, bar, atm, psi"
    std::string docs;                 // page and anchor of the input reference
};

/**
 * @brief Keys of the input that were never read, outermost first (a table
 *        nobody read is reported, not its contents), each with the closest key
 *        the setup looked up, at the same level or (exact name) elsewhere.
 */
std::vector<Diagnostic> unused_keys(const Input & input);

/** @brief "line 12: mesh.Nxx (did you mean mesh.Nx?)". */
std::string describe(const Diagnostic & key);

/**
 * @brief The message of an invalid value: "line 12: boundaries[0].p = \"300 K\":
 *        a temperature where a pressure is expected (expected: ...; see ...)".
 */
std::string explain(const Diagnostic & d);

/**
 * @brief An invalid or missing input value. The message names the key and the
 *        offending value; the caller adds the input file name.
 */
struct InputError : std::runtime_error {
    using std::runtime_error::runtime_error;
    explicit InputError(const Diagnostic & d) : std::runtime_error(explain(d)), diagnostic(d) {}
    Diagnostic diagnostic;  // empty key for errors given as text only
};

/**
 * @brief Error for a value not among the accepted options, listing them.
 */
template <typename T>
InputError unknown_option(const std::unordered_map<std::string, T> & options, const std::string & key,
                          const std::string & value) {
    std::vector<std::string> names;
    for (const auto & entry : options) names.push_back(entry.first);
    std::sort(names.begin(), names.end());
    std::string list;
    for (const auto & name : names) list += (list.empty() ? "" : ", ") + name;
    return InputError(key + " = \"" + value + "\" is not one of: " + list + ".");
}

/**
 * Real-valued lookups. Each names the key's Dimension: a value is a plain
 * number (SI, or the unit the key documents: Dimension::plain) or a string
 * with a unit ("1 atm", "300 K", "5 mm"), converted on read; a unit of the
 * wrong dimension is an error naming the key and the expected dimension.
 * Integers count as numbers: TOML tells them from floats, and
 * toml::find_or<double> would silently fall back to its default for `cfl = 1`.
 */

/** @brief The value of v in the units the key reads (SI unless the dimension says otherwise). */
double quantity(const Input & v, const std::string & key, const Dimension & dimension);

/**
 * @brief The text of an expression in x, y, z (and t) for a value that is a
 *        number or a quantity with a unit (converted as quantity() does), or
 *        else an expression, which is in SI units.
 */
std::string quantity_or_expression(const Input & v, const std::string & key, const Dimension & dimension);

inline rtype as_real(const Input & v, const std::string & key, const Dimension & dimension) {
    return static_cast<rtype>(quantity(v, key, dimension));
}

/** @brief Double-precision lookups, for inputs of the chemistry (double in every build). */
inline double as_double(const Input & v, const std::string & key, const Dimension & dimension) {
    return quantity(v, key, dimension);
}

inline double find_double(const Input & v, const std::string & key, const Dimension & dimension) {
    if (!v.contains(key)) throw InputError("missing " + key + ".");
    return as_double(v.at(key), key, dimension);
}

inline double find_double_or(const Input & v, const std::string & key, const Dimension & dimension,
                             const double fallback) {
    return v.contains(key) ? as_double(v.at(key), key, dimension) : fallback;
}

inline rtype find_real(const Input & v, const std::string & key, const Dimension & dimension) {
    if (!v.contains(key)) throw InputError("missing " + key + ".");
    return as_real(v.at(key), key, dimension);
}

inline rtype find_real(const Input & v, const std::string & table, const std::string & key,
                       const Dimension & dimension) {
    if (!v.contains(table)) throw InputError("missing [" + table + "].");
    if (!v.at(table).contains(key)) throw InputError("missing " + table + "." + key + ".");
    return as_real(v.at(table).at(key), table + "." + key, dimension);
}

inline rtype find_real_or(const Input & v, const std::string & key, const Dimension & dimension,
                          const rtype fallback) {
    return v.contains(key) ? as_real(v.at(key), key, dimension) : fallback;
}

inline rtype find_real_or(const Input & v, const std::string & table, const std::string & key,
                          const Dimension & dimension, const rtype fallback) {
    if (!v.contains(table) || !v.at(table).contains(key)) return fallback;
    return as_real(v.at(table).at(key), table + "." + key, dimension);
}

inline std::vector<rtype> as_real_vector(const Input & v, const std::string & key, const Dimension & dimension) {
    std::vector<rtype> out;
    for (const auto & item : v.as_array()) out.push_back(as_real(item, key, dimension));
    return out;
}

inline std::vector<rtype> find_real_vector(const Input & v, const std::string & key, const Dimension & dimension) {
    if (!v.contains(key)) throw InputError("missing " + key + ".");
    return as_real_vector(v.at(key), key, dimension);
}

inline std::vector<rtype> find_real_vector(const Input & v, const std::string & table, const std::string & key,
                                           const Dimension & dimension) {
    if (!v.contains(table)) throw InputError("missing [" + table + "].");
    if (!v.at(table).contains(key)) throw InputError("missing " + table + "." + key + ".");
    return as_real_vector(v.at(table).at(key), table + "." + key, dimension);
}

#endif // INPUT_H

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

/** @brief A key of the input that no setup code read. */
struct UnusedKey {
    std::string path;        // e.g. "mesh.Nxx" or "boundaries[2].tpye"
    std::size_t line = 0;    // 0 if unknown
    std::string suggestion;  // full path of a key read or looked up by setup, or empty
};

/**
 * @brief Keys of the input that were never read, outermost first (a table
 *        nobody read is reported, not its contents), each with the closest key
 *        the setup looked up, at the same level or (exact name) elsewhere.
 */
std::vector<UnusedKey> unused_keys(const Input & input);

/** @brief "line 12: mesh.Nxx (did you mean mesh.Nx?)". */
std::string describe(const UnusedKey & key);

/**
 * @brief An invalid or missing input value. The message names the key and the
 *        offending value; the caller adds the input file name.
 */
struct InputError : std::runtime_error {
    using std::runtime_error::runtime_error;
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
 * TOML distinguishes integers from floats, so `cfl = 1` is not a float and
 * toml::find_or<double> silently falls back to its default for it. These
 * helpers accept either and reject anything else.
 */

inline rtype as_real(const Input & v, const std::string & key) {
    if (v.is_floating()) return static_cast<rtype>(v.as_floating());
    if (v.is_integer()) return static_cast<rtype>(v.as_integer());
    throw InputError(toml::format_error(key + " must be a number", v, "here"));
}

/** @brief Double-precision lookups, for inputs of the chemistry (double in every build). */
inline double as_double(const Input & v, const std::string & key) {
    if (v.is_floating()) return v.as_floating();
    if (v.is_integer()) return static_cast<double>(v.as_integer());
    throw InputError(toml::format_error(key + " must be a number", v, "here"));
}

inline double find_double(const Input & v, const std::string & key) {
    if (!v.contains(key)) throw InputError("missing " + key + ".");
    return as_double(v.at(key), key);
}

inline double find_double_or(const Input & v, const std::string & key, const double fallback) {
    return v.contains(key) ? as_double(v.at(key), key) : fallback;
}

inline rtype find_real(const Input & v, const std::string & key) {
    if (!v.contains(key)) throw InputError("missing " + key + ".");
    return as_real(v.at(key), key);
}

inline rtype find_real(const Input & v, const std::string & table, const std::string & key) {
    if (!v.contains(table)) throw InputError("missing [" + table + "].");
    if (!v.at(table).contains(key)) throw InputError("missing " + table + "." + key + ".");
    return as_real(v.at(table).at(key), table + "." + key);
}

inline rtype find_real_or(const Input & v, const std::string & key, const rtype fallback) {
    return v.contains(key) ? as_real(v.at(key), key) : fallback;
}

inline rtype find_real_or(const Input & v, const std::string & table, const std::string & key,
                          const rtype fallback) {
    if (!v.contains(table) || !v.at(table).contains(key)) return fallback;
    return as_real(v.at(table).at(key), table + "." + key);
}

inline std::vector<rtype> find_real_vector(const Input & v, const std::string & key) {
    if (!v.contains(key)) throw InputError("missing " + key + ".");
    std::vector<rtype> out;
    for (const auto & item : v.at(key).as_array()) out.push_back(as_real(item, key));
    return out;
}

inline std::vector<rtype> find_real_vector(const Input & v, const std::string & table, const std::string & key) {
    if (!v.contains(table)) throw InputError("missing [" + table + "].");
    if (!v.at(table).contains(key)) throw InputError("missing " + table + "." + key + ".");
    std::vector<rtype> out;
    for (const auto & item : v.at(table).at(key).as_array()) out.push_back(as_real(item, table + "." + key));
    return out;
}

#endif // INPUT_H

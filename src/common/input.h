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
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include <toml.hpp>

#include "common_typedef.h"

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

inline rtype as_real(const toml::value & v, const std::string & key) {
    if (v.is_floating()) return static_cast<rtype>(v.as_floating());
    if (v.is_integer()) return static_cast<rtype>(v.as_integer());
    throw InputError(toml::format_error(key + " must be a number", v, "here"));
}

/** @brief Double-precision lookups, for inputs of the chemistry (double in every build). */
inline double as_double(const toml::value & v, const std::string & key) {
    if (v.is_floating()) return v.as_floating();
    if (v.is_integer()) return static_cast<double>(v.as_integer());
    throw InputError(toml::format_error(key + " must be a number", v, "here"));
}

inline double find_double(const toml::value & v, const std::string & key) {
    if (!v.contains(key)) throw InputError("missing " + key + ".");
    return as_double(v.at(key), key);
}

inline double find_double_or(const toml::value & v, const std::string & key, const double fallback) {
    return v.contains(key) ? as_double(v.at(key), key) : fallback;
}

inline rtype find_real(const toml::value & v, const std::string & key) {
    if (!v.contains(key)) throw InputError("missing " + key + ".");
    return as_real(v.at(key), key);
}

inline rtype find_real(const toml::value & v, const std::string & table, const std::string & key) {
    if (!v.contains(table)) throw InputError("missing [" + table + "].");
    if (!v.at(table).contains(key)) throw InputError("missing " + table + "." + key + ".");
    return as_real(v.at(table).at(key), table + "." + key);
}

inline rtype find_real_or(const toml::value & v, const std::string & key, const rtype fallback) {
    return v.contains(key) ? as_real(v.at(key), key) : fallback;
}

inline rtype find_real_or(const toml::value & v, const std::string & table, const std::string & key,
                          const rtype fallback) {
    if (!v.contains(table) || !v.at(table).contains(key)) return fallback;
    return as_real(v.at(table).at(key), table + "." + key);
}

inline std::vector<rtype> find_real_vector(const toml::value & v, const std::string & key) {
    if (!v.contains(key)) throw InputError("missing " + key + ".");
    std::vector<rtype> out;
    for (const auto & item : v.at(key).as_array()) out.push_back(as_real(item, key));
    return out;
}

inline std::vector<rtype> find_real_vector(const toml::value & v, const std::string & table, const std::string & key) {
    if (!v.contains(table)) throw InputError("missing [" + table + "].");
    if (!v.at(table).contains(key)) throw InputError("missing " + table + "." + key + ".");
    std::vector<rtype> out;
    for (const auto & item : v.at(table).at(key).as_array()) out.push_back(as_real(item, table + "." + key));
    return out;
}

#endif // INPUT_H

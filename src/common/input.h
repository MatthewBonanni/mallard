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

#include <stdexcept>
#include <string>
#include <vector>

#include <toml.hpp>

#include "common_typedef.h"

/**
 * TOML distinguishes integers from floats, so `cfl = 1` is not a float and
 * toml::find_or<double> silently falls back to its default for it. These
 * helpers accept either and reject anything else.
 */

inline rtype as_real(const toml::value & v, const std::string & key) {
    if (v.is_floating()) return static_cast<rtype>(v.as_floating());
    if (v.is_integer()) return static_cast<rtype>(v.as_integer());
    throw std::runtime_error("Input: " + key + " must be a number.");
}

inline rtype find_real(const toml::value & v, const std::string & key) {
    if (!v.contains(key)) throw std::runtime_error("Input: missing " + key + ".");
    return as_real(v.at(key), key);
}

inline rtype find_real(const toml::value & v, const std::string & table, const std::string & key) {
    if (!v.contains(table)) throw std::runtime_error("Input: missing [" + table + "].");
    return find_real(v.at(table), key);
}

inline rtype find_real_or(const toml::value & v, const std::string & key, const rtype fallback) {
    return v.contains(key) ? as_real(v.at(key), key) : fallback;
}

inline rtype find_real_or(const toml::value & v, const std::string & table, const std::string & key,
                          const rtype fallback) {
    return v.contains(table) ? find_real_or(v.at(table), key, fallback) : fallback;
}

inline std::vector<rtype> find_real_vector(const toml::value & v, const std::string & key) {
    if (!v.contains(key)) throw std::runtime_error("Input: missing " + key + ".");
    std::vector<rtype> out;
    for (const auto & item : v.at(key).as_array()) out.push_back(as_real(item, key));
    return out;
}

inline std::vector<rtype> find_real_vector(const toml::value & v, const std::string & table, const std::string & key) {
    if (!v.contains(table)) throw std::runtime_error("Input: missing [" + table + "].");
    return find_real_vector(v.at(table), key);
}

#endif // INPUT_H

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
#include <iomanip>
#include <map>
#include <memory>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <vector>

#include <toml.hpp>

#include "common_typedef.h"

/**
 * @brief How setup reads a key, as the accessors below record it: the type it
 *        asks for, and the value it uses when the key is absent, or that it is
 *        required. The key catalog (mallard --check --catalog) is made of these.
 */
struct KeyUse {
    std::string type;      // bool, integer, number, string, array, or empty if not recorded
    std::string fallback;  // the value used when absent, as TOML would write it
    bool required = false;
};

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
            std::map<K, KeyUse> uses;
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
        void record_use(const K & k, KeyUse use) const {
            note(k);
            lookups->uses[k] = std::move(use);
        }
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
    std::string suggestion;           // full path of the likely intended key, or empty
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
 * @brief An invalid or missing input value. The message names the key and the
 *        offending value; the caller adds the input file name.
 */
struct InputError : std::runtime_error {
    using std::runtime_error::runtime_error;
};

/** @brief Record the values a key accepts, by its path ("numerics.riemann_solver"; indices are dropped). */
void record_options(const std::string & key, std::vector<std::string> names);

/**
 * @brief The key catalog of an input after setup read it: every key setup looked
 *        up (by path, without array indices), whether the input set it, and its
 *        KeyUse and accepted values where the accessors recorded them, as JSON.
 */
std::string key_catalog(const Input & input);

/**
 * @brief The entry of options named by value, or an error listing the accepted
 *        values; records them for the key catalog.
 */
template <typename T>
auto find_option(const std::unordered_map<std::string, T> & options, const std::string & key,
                 const std::string & value) {
    std::vector<std::string> names;
    for (const auto & entry : options) names.push_back(entry.first);
    std::sort(names.begin(), names.end());
    record_options(key, names);
    const auto it = options.find(value);
    if (it == options.end()) {
        std::string list;
        for (const auto & name : names) list += (list.empty() ? "" : ", ") + name;
        throw InputError(key + " = \"" + value + "\" is not one of: " + list + ".");
    }
    return it;
}

namespace input_detail {

template <typename T> struct is_vector : std::false_type {};
template <typename T, typename A> struct is_vector<std::vector<T, A>> : std::true_type {};

template <typename T> std::string type_name() {
    if constexpr (std::is_same_v<T, bool>) return "bool";
    else if constexpr (std::is_integral_v<T>) return "integer";
    else if constexpr (std::is_floating_point_v<T>) return "number";
    else if constexpr (std::is_convertible_v<T, std::string>) return "string";
    else if constexpr (is_vector<T>::value) return "array";
    else return "";
}

template <typename T> std::string toml_text(const T & x) {
    if constexpr (std::is_same_v<T, bool>) {
        return x ? "true" : "false";
    } else if constexpr (std::is_integral_v<T>) {
        return std::to_string(x);
    } else if constexpr (std::is_floating_point_v<T>) {
        std::ostringstream os;
        os << std::setprecision(15) << x;
        std::string t = os.str();
        if (t.find_first_of(".eEn") == std::string::npos) t += ".0";
        return t;
    } else if constexpr (std::is_convertible_v<T, std::string>) {
        return "\"" + std::string(x) + "\"";
    } else if constexpr (is_vector<T>::value) {
        std::string t;
        for (const auto & item : x) t += (t.empty() ? "" : ", ") + toml_text(item);
        return "[" + t + "]";
    } else {
        return "";
    }
}

/** @brief The table holding the key, if every enclosing table is in the input. */
inline const InputTable<std::string, Input> * table_of(const Input & v) {
    return v.is_table() ? &v.as_table() : nullptr;
}
inline const InputTable<std::string, Input> * table_of(const Input & v, const std::string & table) {
    return v.is_table() && v.contains(table) && v.at(table).is_table() ? &v.at(table).as_table() : nullptr;
}

} // namespace input_detail

/**
 * Lookups of typed values, the way setup reads every key that is not a real
 * number: like toml::find and toml::find_or (whose semantics they keep), and
 * recording the type and the fallback (or that the key is required) for the
 * key catalog. src/CMakeLists.txt rejects toml::find and toml::find_or.
 */
template <typename T, typename... Tables>
T find_value(const Input & v, const Tables &... keys) {
    static_assert(sizeof...(Tables) == 1 || sizeof...(Tables) == 2, "find_value(v, [table,] key)");
    const std::string path[] = {keys...};
    const auto * table = sizeof...(Tables) == 1 ? input_detail::table_of(v) : input_detail::table_of(v, path[0]);
    if (table) table->record_use(path[sizeof...(Tables) - 1], {input_detail::type_name<T>(), "", true});
    return toml::find<T>(v, keys...);
}

template <typename T>
T find_value_or(const Input & v, const std::string & key, const T & fallback) {
    if (const auto * table = input_detail::table_of(v)) {
        table->record_use(key, {input_detail::type_name<T>(), input_detail::toml_text(fallback), false});
    }
    return toml::find_or<T>(v, key, fallback);
}

template <typename T>
T find_value_or(const Input & v, const std::string & table_name, const std::string & key, const T & fallback) {
    if (const auto * table = input_detail::table_of(v, table_name)) {
        table->record_use(key, {input_detail::type_name<T>(), input_detail::toml_text(fallback), false});
    }
    return toml::find_or<T>(v, table_name, key, fallback);
}

/**
 * TOML distinguishes integers from floats, so `cfl = 1` is not a float and
 * find_value_or<double> silently falls back to its default for it. These
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
    if (const auto * t = input_detail::table_of(v)) t->record_use(key, {"number", "", true});
    if (!v.contains(key)) throw InputError("missing " + key + ".");
    return as_double(v.at(key), key);
}

inline double find_double_or(const Input & v, const std::string & key, const double fallback) {
    if (const auto * t = input_detail::table_of(v)) {
        t->record_use(key, {"number", input_detail::toml_text(fallback), false});
    }
    return v.contains(key) ? as_double(v.at(key), key) : fallback;
}

inline rtype find_real(const Input & v, const std::string & key) {
    if (const auto * t = input_detail::table_of(v)) t->record_use(key, {"number", "", true});
    if (!v.contains(key)) throw InputError("missing " + key + ".");
    return as_real(v.at(key), key);
}

inline rtype find_real(const Input & v, const std::string & table, const std::string & key) {
    if (const auto * t = input_detail::table_of(v, table)) t->record_use(key, {"number", "", true});
    if (!v.contains(table)) throw InputError("missing [" + table + "].");
    if (!v.at(table).contains(key)) throw InputError("missing " + table + "." + key + ".");
    return as_real(v.at(table).at(key), table + "." + key);
}

inline rtype find_real_or(const Input & v, const std::string & key, const rtype fallback) {
    if (const auto * t = input_detail::table_of(v)) {
        t->record_use(key, {"number", input_detail::toml_text(fallback), false});
    }
    return v.contains(key) ? as_real(v.at(key), key) : fallback;
}

inline rtype find_real_or(const Input & v, const std::string & table, const std::string & key,
                          const rtype fallback) {
    if (const auto * t = input_detail::table_of(v, table)) {
        t->record_use(key, {"number", input_detail::toml_text(fallback), false});
    }
    if (!v.contains(table) || !v.at(table).contains(key)) return fallback;
    return as_real(v.at(table).at(key), table + "." + key);
}

inline std::vector<rtype> find_real_vector(const Input & v, const std::string & key) {
    if (const auto * t = input_detail::table_of(v)) t->record_use(key, {"array", "", true});
    if (!v.contains(key)) throw InputError("missing " + key + ".");
    std::vector<rtype> out;
    for (const auto & item : v.at(key).as_array()) out.push_back(as_real(item, key));
    return out;
}

inline std::vector<rtype> find_real_vector(const Input & v, const std::string & table, const std::string & key) {
    if (const auto * t = input_detail::table_of(v, table)) t->record_use(key, {"array", "", true});
    if (!v.contains(table)) throw InputError("missing [" + table + "].");
    if (!v.at(table).contains(key)) throw InputError("missing " + table + "." + key + ".");
    std::vector<rtype> out;
    for (const auto & item : v.at(table).at(key).as_array()) out.push_back(as_real(item, table + "." + key));
    return out;
}

#endif // INPUT_H

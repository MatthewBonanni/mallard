/**
 * @file input.cpp
 * @brief Parsing of input files, and the keys no setup code read.
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 */

#include "input.h"

#include <cctype>
#include <map>
#include <sstream>

namespace {

void forget_lookups(Input & v) {
    if (v.is_table()) {
        auto & table = v.as_table();
        table.forget_lookups();
        for (auto & entry : table.entries()) forget_lookups(entry.second);
    } else if (v.is_array()) {
        for (auto & item : v.as_array()) forget_lookups(item);
    }
}

std::string lower(std::string s) {
    for (char & c : s) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    return s;
}

/** @brief Optimal string alignment distance, ignoring case. */
size_t edit_distance(const std::string & a_in, const std::string & b_in) {
    const std::string a = lower(a_in), b = lower(b_in);
    std::vector<std::vector<size_t>> d(a.size() + 1, std::vector<size_t>(b.size() + 1));
    for (size_t i = 0; i <= a.size(); i++) d[i][0] = i;
    for (size_t j = 0; j <= b.size(); j++) d[0][j] = j;
    for (size_t i = 1; i <= a.size(); i++) {
        for (size_t j = 1; j <= b.size(); j++) {
            const size_t cost = a[i - 1] == b[j - 1] ? 0 : 1;
            d[i][j] = std::min({d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + cost});
            if (i > 1 && j > 1 && a[i - 1] == b[j - 2] && a[i - 2] == b[j - 1]) {
                d[i][j] = std::min(d[i][j], d[i - 2][j - 2] + 1);
            }
        }
    }
    return d[a.size()][b.size()];
}

std::string without_indices(const std::string & path) {
    std::string out;
    bool in_index = false;
    for (char c : path) {
        if (c == '[') in_index = true;
        if (!in_index) out += c;
        if (c == ']') in_index = false;
    }
    return out;
}

std::string join(const std::string & table, const std::string & key) { return table.empty() ? key : table + "." + key; }

using Lookups = std::multimap<std::string, std::string>;  // key -> path of the table it was looked up in

void collect_lookups(const Input & v, const std::string & path, Lookups & out) {
    if (v.is_table()) {
        const auto & table = v.as_table();
        for (const auto & key : table.lookups_made().keys) out.emplace(key, path);
        for (const auto & [key, value] : table.entries()) collect_lookups(value, join(path, key), out);
    } else if (v.is_array()) {
        const auto & array = v.as_array();
        for (size_t i = 0; i < array.size(); i++) collect_lookups(array[i], path + "[" + std::to_string(i) + "]", out);
    }
}

std::string suggest(const InputTable<std::string, Input> & table, const std::string & path, const std::string & key,
                    const Lookups & lookups) {
    std::string best;
    size_t best_distance = (key.size() <= 2 ? 0 : std::max<size_t>(1, key.size() / 3)) + 1;
    for (const auto & candidate : table.lookups_made().keys) {
        if (table.entries().count(candidate)) continue;
        const size_t distance = edit_distance(key, candidate);
        if (distance < best_distance) {
            best_distance = distance;
            best = join(path, candidate);
        }
    }
    if (!best.empty()) return best;
    const auto [first, last] = lookups.equal_range(key);
    for (auto it = first; it != last; ++it) {
        if (without_indices(it->second) != without_indices(path)) return join(it->second, key);
    }
    return "";
}

void collect_unused(const Input & v, const std::string & path, const Lookups & lookups, std::vector<Diagnostic> & out) {
    if (v.is_table()) {
        const auto & table = v.as_table();
        const auto & made = table.lookups_made();
        for (const auto & [key, value] : table.entries()) {
            if (made.all || made.keys.count(key)) {
                collect_unused(value, join(path, key), lookups, out);
            } else {
                const auto location = value.location();
                Diagnostic d;
                d.key = join(path, key);
                d.line = location.is_ok() ? location.first_line_number() : 0;
                d.message = "unknown, or unused with these settings";
                d.suggestion = suggest(table, path, key, lookups);
                const std::string section = d.key.substr(0, d.key.find_first_of(".["));
                d.docs = "docs/input.md" + (section == d.key ? std::string() : "#" + lower(section));
                out.push_back(std::move(d));
            }
        }
    } else if (v.is_array()) {
        const auto & array = v.as_array();
        for (size_t i = 0; i < array.size(); i++) {
            collect_unused(array[i], path + "[" + std::to_string(i) + "]", lookups, out);
        }
    }
}

} // namespace

Input parse_input(const std::string & filename) {
    Input input = toml::parse<InputConfig>(filename);
    forget_lookups(input);
    return input;
}

Input parse_input_string(const std::string & text, const std::string & name) {
    std::istringstream stream(text);
    Input input = toml::parse<InputConfig>(stream, name);
    forget_lookups(input);
    return input;
}

std::vector<Diagnostic> unused_keys(const Input & input) {
    Lookups lookups;
    collect_lookups(input, "", lookups);
    std::vector<Diagnostic> out;
    collect_unused(input, "", lookups, out);
    std::stable_sort(out.begin(), out.end(), [](const Diagnostic & a, const Diagnostic & b) { return a.line < b.line; });
    return out;
}

std::string describe(const Diagnostic & key) {
    std::string text = key.line > 0 ? "line " + std::to_string(key.line) + ": " : "";
    text += key.key;
    if (!key.suggestion.empty()) text += " (did you mean " + key.suggestion + "?)";
    if (!key.valid.empty()) {
        text += ": one of";
        for (const auto & v : key.valid) text += " " + v;
    }
    return text;
}

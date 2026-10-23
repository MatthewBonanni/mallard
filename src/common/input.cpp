/**
 * @file input.cpp
 * @brief Parsing of input files, and the keys no setup code read.
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 */

#include "input.h"

#include <cctype>
#include <iomanip>
#include <map>
#include <mutex>
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
                if (!location.is_ok()) continue;  // set by the program (e.g. a numerics preset), not the file
                Diagnostic d;
                d.key = join(path, key);
                d.line = location.first_line_number();
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

std::map<std::string, std::vector<std::string>> & recorded_options() {
    static std::map<std::string, std::vector<std::string>> options;
    return options;
}

std::map<std::string, std::pair<std::string, std::string>> & recorded_dimensions() {
    static std::map<std::string, std::pair<std::string, std::string>> dimensions;
    return dimensions;
}

std::mutex & options_mutex() {
    static std::mutex m;
    return m;
}

std::string json_string(const std::string & s) {
    std::string out = "\"";
    for (const char c : s) {
        if (c == '"' || c == '\\') out += '\\';
        if (static_cast<unsigned char>(c) < 0x20) continue;
        out += c;
    }
    return out + "\"";
}

struct CatalogEntry {
    KeyUse use;
    bool set = false;
};

void collect_catalog(const Input & v, const std::string & path, std::map<std::string, CatalogEntry> & out) {
    if (v.is_table()) {
        const auto & table = v.as_table();
        for (const auto & key : table.lookups_made().keys) {
            auto & entry = out[join(path, key)];
            const auto use = table.lookups_made().uses.find(key);
            if (use != table.lookups_made().uses.end() && !use->second.type.empty()) entry.use = use->second;
            if (table.entries().count(key)) entry.set = true;
        }
        for (const auto & [key, value] : table.entries()) collect_catalog(value, join(path, key), out);
    } else if (v.is_array()) {
        for (const auto & item : v.as_array()) collect_catalog(item, path, out);
    }
}

} // namespace

/** @brief The dimension of a real value read under this key path (values read directly, not through find_). */
static void record_dimension(const std::string & key, const Dimension & dimension) {
    if (dimension.same_as(dim::none)) return;
    std::lock_guard<std::mutex> lock(options_mutex());
    recorded_dimensions()[without_indices(key)] = {dimension.describe(),
                                                  std::string(dimension.plain_unit) == "SI" ? "" : dimension.plain_unit};
}

void record_options(const std::string & key, std::vector<std::string> names) {
    std::lock_guard<std::mutex> lock(options_mutex());
    recorded_options()[without_indices(key)] = std::move(names);
}

std::string key_catalog(const Input & input) {
    std::map<std::string, CatalogEntry> keys;
    collect_catalog(input, "", keys);
    std::map<std::string, std::vector<std::string>> options;
    {
        std::lock_guard<std::mutex> lock(options_mutex());
        options = recorded_options();
        for (auto & [path, entry] : keys) {
            const auto d = recorded_dimensions().find(path);
            if (d == recorded_dimensions().end() || !entry.use.dimension.empty()) continue;
            entry.use.dimension = d->second.first;
            entry.use.unit = d->second.second;
        }
    }
    std::string out = "{\n";
    bool first = true;
    for (const auto & [path, entry] : keys) {
        out += first ? "" : ",\n";
        first = false;
        out += "  " + json_string(path) + ": {\"set\": " + (entry.set ? "true" : "false");
        if (!entry.use.type.empty()) {
            out += ", \"type\": " + json_string(entry.use.type);
            if (!entry.use.dimension.empty()) out += ", \"dimension\": " + json_string(entry.use.dimension);
            if (!entry.use.unit.empty()) out += ", \"unit\": " + json_string(entry.use.unit);
            out += entry.use.required ? ", \"required\": true" : ", \"default\": " + json_string(entry.use.fallback);
        }
        const auto allowed = options.find(path);
        if (allowed != options.end()) {
            out += ", \"values\": [";
            for (size_t i = 0; i < allowed->second.size(); i++) {
                out += (i ? ", " : "") + json_string(allowed->second[i]);
            }
            out += "]";
        }
        out += "}";
    }
    return out + "\n}\n";
}

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

std::string explain(const Diagnostic & d) {
    std::string text = d.line > 0 ? "line " + std::to_string(d.line) + ": " : "";
    text += d.key.empty() ? d.message : d.key + ": " + d.message;
    if (!d.suggestion.empty() && d.message.find("did you mean") == std::string::npos) {
        text += " (did you mean " + d.suggestion + "?)";
    }
    if (!text.empty() && text.back() != '.') text += ".";
    if (!d.expected.empty()) text += " Expected " + d.expected + ".";
    if (!d.docs.empty()) text += " See " + d.docs + ".";
    return text;
}

namespace {

std::string expected_text(const Dimension & d) {
    if (d.same_as(dim::none)) return "a number";
    std::string text = d.describe();
    const std::string examples = d.examples();
    if (!examples.empty()) text += ", e.g. in " + examples;
    return text + "; a number without a unit is in " + d.plain_unit;
}

[[noreturn]] void value_error(const Input & v, const std::string & key, const Dimension & d, const std::string & message,
                              const std::string & suggestion = "") {
    Diagnostic diagnostic;
    diagnostic.key = key;
    const auto location = v.location();
    diagnostic.line = location.is_ok() ? location.first_line_number() : 0;
    diagnostic.message = message;
    diagnostic.suggestion = suggestion;
    diagnostic.expected = expected_text(d);
    diagnostic.docs = "docs/guide/input_files.md#units";
    throw InputError(diagnostic);
}

std::string number_text(const double x) {
    std::ostringstream text;
    text << std::setprecision(17) << x;
    return text.str();
}

} // namespace

double quantity(const Input & v, const std::string & key, const Dimension & dimension) {
    record_dimension(key, dimension);
    if (v.is_floating()) return v.as_floating();
    if (v.is_integer()) return static_cast<double>(v.as_integer());
    if (!v.is_string()) {
        value_error(v, key, dimension,
                    dimension.same_as(dim::none) ? "must be a number" : "must be a number, or a string of a number and a unit");
    }
    const std::string & text = v.as_string();
    Quantity q;
    try {
        q = parse_quantity(text);
    } catch (const UnitError & e) {
        value_error(v, key, dimension, e.what(), e.suggestion);
    }
    if (!q.has_unit) return q.value;
    if (!q.dimension.same_as(dimension)) {
        const std::string what = q.dimension.same_as(dim::none) ? "is dimensionless"
                                                                 : "has the dimension of " + q.dimension.describe();
        value_error(v, key, dimension, "\"" + text + "\" " + what);
    }
    if (q.absolute && dimension.interval) {
        value_error(v, key, dimension,
                    "\"" + text + "\" is an absolute temperature, but this key is a temperature difference: use K or degR");
    }
    return q.value / dimension.plain;
}

std::string quantity_or_expression(const Input & v, const std::string & key, const Dimension & dimension) {
    record_dimension(key, dimension);
    // Numbers as the solver reads them (in its precision), as expressions always have
    if (!v.is_string()) return number_text(static_cast<double>(as_real(v, key, dimension)));
    return has_unit(v.as_string()) ? number_text(quantity(v, key, dimension)) : v.as_string();
}

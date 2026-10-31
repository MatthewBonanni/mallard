/**
 * @file input_upgrade.cpp
 * @author Matthew Bonanni (mbonanni001@gmail.com)
 * @brief Renamed, moved and removed input keys and options, and `mallard input upgrade`.
 * @version 0.9
 * @date 2026-10-16
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 *
 */

#include "input_upgrade.h"

#include <array>
#include <fstream>
#include <iostream>
#include <map>
#include <optional>
#include <regex>
#include <sstream>
#include <stdexcept>

#include "input.h"
#include "version.h"

namespace input_upgrade {

namespace {

std::string plus_one(const std::string & text) { return std::to_string(std::stoll(text) + 1); }

/** @brief The zones of a generated mesh that a direction of `[mesh] periodic` joined. */
std::vector<std::string> periodic_zones(const std::string & direction, const std::string & mesh_type) {
    if (mesh_type.empty() || mesh_type == "file") {
        throw std::invalid_argument("applies to generated meshes; pair the zones of a mesh file with [[periodic]] "
                                    "zones and a translation or an angle");
    }
    if (mesh_type == "annulus") {
        if (direction == "theta") return {"start", "end"};
        if (direction == "z") return {"back", "front"};
        throw std::invalid_argument("\"" + direction + "\" is not accepted: the annulus is periodic in \"theta\" "
                                    "(and \"z\" in 3D)");
    }
    if (direction == "x") return {"left", "right"};
    if (direction == "y") return {"bottom", "top"};
    if (direction == "z") return {"back", "front"};
    throw std::invalid_argument("\"" + direction + "\" is not accepted: a " + mesh_type +
                                " mesh is periodic in \"x\", \"y\" (and \"z\" in 3D)");
}

} // namespace

const Rules & rules() {
    static const std::string TYPES = "boundary types name their kind first, and options are keys";
    static const std::string NSCBC = "the characteristic parameters are the nscbc table";
    // Deprecate before removing: `removed` is at least one minor release after `since`.
    // Keys removed before this policy (0.9.0) have since == removed.
    static const Rules table = {
        {
            {"numerics.face_reconstruction.basis_order", "numerics.face_reconstruction.order", "0.2.0", "0.2.0",
             "the polynomial degree became the order of accuracy, degree + 1", plus_one},
            {"numerics.face_reconstruction.basis_type", "", "0.2.0", "0.2.0", "TENO-E chooses its basis"},
            {"numerics.face_reconstruction.max_stencil_size_factor", "", "0.2.0", "0.2.0",
             "TENO-E sizes its stencils"},
            {"numerics.face_reconstruction.quadrature_type_cell", "", "0.2.0", "0.2.0", "quadrature follows the order"},
            {"numerics.face_reconstruction.quadrature_order_cell", "", "0.2.0", "0.2.0", "quadrature follows the order"},
            {"numerics.face_reconstruction.quadrature_type_face", "", "0.2.0", "0.2.0", "quadrature follows the order"},
            {"numerics.face_reconstruction.quadrature_order_face", "", "0.2.0", "0.2.0", "quadrature follows the order"},
            {"numerics.face_reconstruction.quadrature_type", "", "0.2.0", "0.2.0", "never read"},
            {"numerics.face_reconstruction.quadrature_order", "", "0.2.0", "0.2.0", "never read"},
            {"physics.p_min", "", "0.2.0", "0.2.0", "pressures are not clipped"},
            {"physics.p_max", "", "0.2.0", "0.2.0", "pressures are not clipped"},
            {"boundaries[].L", "boundaries[].nscbc.L", "0.9.0", "0.10.0", NSCBC},
            {"boundaries[].sigma", "boundaries[].nscbc.sigma", "0.9.0", "0.10.0", NSCBC},
            {"boundaries[].beta", "boundaries[].nscbc.beta", "0.9.0", "0.10.0", NSCBC},
            {"boundaries[].sigma_T", "boundaries[].nscbc.sigma_T", "0.9.0", "0.10.0", NSCBC},
            {"boundaries[].sigma_t", "boundaries[].nscbc.sigma_t", "0.9.0", "0.10.0", NSCBC},
        },
        {
            {"write_data[].format", "tecplot", "vtu", "0.2.0", "0.2.0", "Tecplot output was removed"},
            {"write_data[].format", "hdf5", "vtkhdf", "0.9.0", "0.10.0",
             "VTKHDF replaces the XDMF and HDF5 files; ParaView 5.12 and later read it", true},
            {"boundaries[].type", "upt", "inlet_state", "0.9.0", "0.10.0", TYPES},
            {"boundaries[].type", "stagnation_inlet", "inlet_stagnation", "0.9.0", "0.10.0", TYPES},
            {"boundaries[].type", "mass_flow_inlet", "inlet_mass_flow", "0.9.0", "0.10.0", TYPES},
            {"boundaries[].type", "p_out", "outlet_pressure", "0.9.0", "0.10.0", TYPES},
            {"boundaries[].type", "p_out_average", "outlet_pressure", "0.9.0", "0.10.0", TYPES, false,
             {{"average", "true"}}},
            {"boundaries[].type", "extrapolation", "outlet_extrapolation", "0.9.0", "0.10.0", TYPES},
            {"boundaries[].type", "nscbc_inlet", "inlet_state", "0.9.0", "0.10.0", TYPES + "; " + NSCBC, false,
             {{"nscbc", "{}"}}},
            {"boundaries[].type", "nscbc_outlet", "outlet_pressure", "0.9.0", "0.10.0", TYPES + "; " + NSCBC, false,
             {{"nscbc", "{}"}}},
        },
        {
            {"mesh.periodic", "periodic", "zones", "type", "0.9.0", "0.10.0",
             "one syntax for the periodic zones of generated meshes and mesh files", periodic_zones},
            {"initialize.mesh.periodic", "initialize.periodic", "zones", "type", "0.9.0", "0.10.0",
             "one syntax for the periodic zones of generated meshes and mesh files", periodic_zones},
        },
    };
    return table;
}

namespace {

// ---------------------------------------------------------------- paths

struct Step {
    std::string key;
    bool array = false;
};

using Path = std::vector<Step>;

/** @brief A key of one input: each step's key and, in an array of tables, the element (else -1). */
using Concrete = std::vector<std::pair<std::string, long>>;

Path parse_path(const std::string & dotted) {
    Path path;
    std::stringstream ss(dotted);
    std::string part;
    while (std::getline(ss, part, '.')) {
        const bool array = part.size() > 2 && part.compare(part.size() - 2, 2, "[]") == 0;
        path.push_back({array ? part.substr(0, part.size() - 2) : part, array});
    }
    return path;
}

std::string to_string(const Concrete & key) {
    std::string s;
    for (const auto & [name, index] : key) {
        s += (s.empty() ? "" : ".") + name;
        if (index >= 0) s += "[" + std::to_string(index) + "]";
    }
    return s;
}

std::string dotted(const Concrete & key) {
    std::string s;
    for (const auto & step : key) s += (s.empty() ? "" : ".") + step.first;
    return s;
}

/** @brief A table's entries; for Input, without counting them as read, so that only setup reads count. */
template <typename V>
auto & entries_of(V & node) {
    if constexpr (requires { node.as_table().entries(); }) {
        return node.as_table().entries();
    } else {
        return node.as_table();
    }
}

template <typename V>
void find_keys(V & node, const Path & path, size_t k, Concrete & at, std::vector<Concrete> & out) {
    if (!node.is_table() || !entries_of(node).contains(path[k].key)) return;
    if (k + 1 == path.size()) {
        at.emplace_back(path[k].key, -1);
        out.push_back(at);
        at.pop_back();
        return;
    }
    V & child = entries_of(node).at(path[k].key);
    if (path[k].array) {
        if (!child.is_array()) return;
        for (size_t i = 0; i < child.size(); i++) {
            at.emplace_back(path[k].key, static_cast<long>(i));
            find_keys(child.at(i), path, k + 1, at, out);
            at.pop_back();
        }
    } else {
        at.emplace_back(path[k].key, -1);
        find_keys(child, path, k + 1, at, out);
        at.pop_back();
    }
}

template <typename V>
std::vector<Concrete> matches(V & root, const std::string & dotted_path) {
    std::vector<Concrete> out;
    Concrete at;
    find_keys(root, parse_path(dotted_path), 0, at, out);
    return out;
}

/** @brief The node at `key`, or nullptr; with `create`, missing tables on the way are made. */
template <typename V>
V * at(V & root, const Concrete & key, bool create = false) {
    V * node = &root;
    for (const auto & [name, index] : key) {
        if (!node->is_table()) return nullptr;
        if (!entries_of(*node).contains(name)) {
            if (!create || index >= 0) return nullptr;
            entries_of(*node)[name] = V(typename V::table_type{});
        }
        node = &entries_of(*node).at(name);
        if (index >= 0) {
            if (!node->is_array() || static_cast<size_t>(index) >= node->size()) return nullptr;
            node = &node->at(static_cast<size_t>(index));
        }
    }
    return node;
}

/** @brief `to` for the element of `from` that `key` matched. */
Concrete target(const KeyRule & rule, const Concrete & key) {
    std::vector<long> elements;
    for (const auto & step : key) {
        if (step.second >= 0) elements.push_back(step.second);
    }
    Concrete out;
    size_t next = 0;
    for (const Step & step : parse_path(rule.to)) {
        if (step.array && next >= elements.size()) {
            throw std::logic_error("input_upgrade: " + rule.to + " has more [] than " + rule.from);
        }
        out.emplace_back(step.key, step.array ? elements[next++] : -1);
    }
    return out;
}

Concrete parent(Concrete key) {
    key.pop_back();
    return key;
}

// ---------------------------------------------------------------- versions and messages

std::array<int, 3> parse_version(const std::string & v) {
    std::array<int, 3> out{};
    std::stringstream ss(v);
    std::string part;
    for (int & x : out) {
        if (!std::getline(ss, part, '.')) break;
        x = std::stoi(part);
    }
    return out;
}

bool is_removed(const std::string & removed, const std::string & version) {
    return parse_version(version) >= parse_version(removed);
}

Finding key_finding(const KeyRule & rule, const Concrete & key, std::size_t line, const std::string & version) {
    Finding f;
    f.key = to_string(key);
    f.line = line;
    f.removed = is_removed(rule.removed, version);
    f.suggestion = rule.to.empty() ? "" : to_string(target(rule, key));
    f.message = f.key + (f.removed ? " was removed in " + rule.removed
                                   : " is deprecated since " + rule.since + " and will be removed in " + rule.removed);
    f.message += f.suggestion.empty() ? " (" + rule.note + ")" : ": use " + f.suggestion + " (" + rule.note + ")";
    return f;
}

Finding option_finding(const OptionRule & rule, const Concrete & key, std::size_t line, const std::string & version) {
    Finding f;
    f.key = to_string(key);
    f.line = line;
    f.removed = is_removed(rule.removed, version);
    f.suggestion = f.key + " = \"" + rule.to + "\"";
    f.message = f.key + " = \"" + rule.from + "\"" +
                (f.removed ? " was removed in " + rule.removed
                           : " is deprecated since " + rule.since + " and will be removed in " + rule.removed) +
                ": use \"" + rule.to + "\" (" + rule.note + ")";
    return f;
}

Finding array_finding(const ArrayRule & rule, const Concrete & key, std::size_t line, const std::string & version,
                      const std::vector<std::vector<std::string>> & elements) {
    Finding f;
    f.key = to_string(key);
    f.line = line;
    f.removed = is_removed(rule.removed, version);
    for (const auto & values : elements) {
        f.suggestion += (f.suggestion.empty() ? "" : ", ") + ("[[" + rule.to + "]] " + rule.key + " = ") +
                        input_detail::toml_text(values);
    }
    f.message = f.key + (f.removed ? " was removed in " + rule.removed
                                   : " is deprecated since " + rule.since + " and will be removed in " + rule.removed);
    f.message += ": use " + (f.suggestion.empty() ? "[[" + rule.to + "]]" : f.suggestion) + " (" + rule.note + ")";
    return f;
}

/** @brief A value's TOML text, without the comments attached to it. */
template <typename V>
std::string bare_text(V value) {
    value.comments().clear();
    return toml::format(value);
}

template <typename V>
std::size_t line_of(const V & value) {
    return value.location().is_ok() ? value.location().first_line_number() : 0;
}

// ---------------------------------------------------------------- the rewrite of a parsed input

/** @brief The new elements' values for the `from` key matched at `key`, or an InputError for one it does not take. */
template <typename V>
std::vector<std::vector<std::string>> array_elements(V & root, const ArrayRule & rule, const Concrete & key) {
    const auto & table = entries_of(*at(root, parent(key)));
    const V & value = table.at(key.back().first);
    std::string context;
    if (!rule.context.empty() && table.contains(rule.context) && table.at(rule.context).is_string()) {
        context = table.at(rule.context).as_string();
    }
    auto error = [&](const std::string & message) {
        Diagnostic d;
        d.key = to_string(key);
        d.line = line_of(value);
        d.message = message;
        return InputError(d);
    };
    if (!value.is_array()) throw error("takes an array of strings");
    std::vector<std::vector<std::string>> elements;
    for (size_t i = 0; i < value.size(); i++) {
        if (!value.at(i).is_string()) throw error("takes an array of strings");
        try {
            elements.push_back(rule.convert(value.at(i).as_string(), context));
        } catch (const std::invalid_argument & e) {
            throw error(e.what());
        }
    }
    return elements;
}

Concrete plain_path(const std::string & dotted_path) {
    Concrete key;
    for (const Step & step : parse_path(dotted_path)) key.emplace_back(step.key, -1);
    return key;
}

/**
 * @brief Applies every rule to the parsed input; returns what changed. Throws
 *        if a key is set under both its old and its new name.
 */
template <typename V>
std::vector<Finding> transform(V & root, const Rules & table, const std::string & version, bool setup = false) {
    std::vector<Finding> findings;
    for (const KeyRule & rule : table.keys) {
        for (const Concrete & key : matches(root, rule.from)) {
            V value = *at(root, key);
            findings.push_back(key_finding(rule, key, line_of(value), version));
            if (!rule.to.empty()) {
                const Concrete to = target(rule, key);
                if (at(root, to) != nullptr) {
                    throw InputError(to_string(key) + " and " + to_string(to) + " are both set; " +
                                     to_string(key) + " is the old name of " + to_string(to) + ": remove it.");
                }
                if (rule.convert) {
                    const std::string text = rule.convert(bare_text(value));
                    value = toml::parse_str<typename V::config_type>("x = " + text).at("x");
                }
                V * new_parent = at(root, parent(to), true);
                if (new_parent == nullptr || !new_parent->is_table()) {
                    throw InputError("cannot move " + to_string(key) + " to " + to_string(to) + ".");
                }
                entries_of(*new_parent)[to.back().first] = value;
            }
            entries_of(*at(root, parent(key))).erase(key.back().first);
        }
    }
    for (const OptionRule & rule : table.options) {
        for (const Concrete & key : matches(root, rule.key)) {
            V & value = *at(root, key);
            if (!value.is_string() || value.as_string() != rule.from) continue;
            findings.push_back(option_finding(rule, key, line_of(value), version));
            if (setup && rule.kept && !findings.back().removed) continue;
            value = rule.to;
            V & siblings = *at(root, parent(key));
            for (const auto & [name, text] : rule.adds) {
                if (entries_of(siblings).contains(name)) continue;
                entries_of(siblings)[name] = toml::parse_str<typename V::config_type>("x = " + text).at("x");
            }
        }
    }
    for (const ArrayRule & rule : table.arrays) {
        for (const Concrete & key : matches(root, rule.from)) {
            const auto elements = array_elements(root, rule, key);
            findings.push_back(array_finding(rule, key, line_of(*at(root, key)), version, elements));
            const Concrete to = plain_path(rule.to);
            V * holder = at(root, parent(to), true);
            if (holder == nullptr || !holder->is_table()) {
                throw InputError("cannot move " + to_string(key) + " to [[" + rule.to + "]].");
            }
            auto & entries = entries_of(*holder);
            if (!entries.contains(to.back().first)) entries[to.back().first] = V(typename V::array_type{});
            V & array = entries.at(to.back().first);
            if (!array.is_array()) throw InputError(rule.to + " must be an array of tables.");
            for (const auto & values : elements) {
                typename V::array_type names;
                for (const auto & v : values) names.push_back(V(v));
                typename V::table_type element;
                element[rule.key] = V(names);
                array.as_array().push_back(V(element));
            }
            entries_of(*at(root, parent(key))).erase(key.back().first);
        }
    }
    return findings;
}

// ---------------------------------------------------------------- the rewrite of the text

/** @brief Equal values and keys; comments and layout do not matter. */
template <typename V>
bool same(const V & a, const V & b) {
    if (a.type() != b.type()) return false;
    if (a.is_table()) {
        if (a.size() != b.size()) return false;
        for (const auto & [k, v] : entries_of(a)) {
            if (!entries_of(b).contains(k) || !same(v, entries_of(b).at(k))) return false;
        }
        return true;
    }
    if (a.is_array()) {
        if (a.size() != b.size()) return false;
        for (size_t i = 0; i < a.size(); i++) {
            if (!same(a.at(i), b.at(i))) return false;
        }
        return true;
    }
    return bare_text(a) == bare_text(b);
}

std::vector<std::string> split_lines(const std::string & text) {
    std::vector<std::string> lines;
    std::stringstream ss(text);
    std::string line;
    while (std::getline(ss, line)) lines.push_back(line);
    return lines;
}

std::string join_lines(const std::vector<std::string> & lines) {
    std::string out;
    for (const auto & line : lines) out += line + "\n";
    return out;
}

/** @brief A table header line: its dotted path, resolved to elements of arrays of tables. */
struct Header {
    size_t line;  // 0-based
    Concrete key;
};

std::vector<Header> headers(const std::vector<std::string> & lines) {
    static const std::regex pattern(R"(^\s*(\[\[?)\s*([^\[\]#]+?)\s*\]\]?\s*(#.*)?$)");
    std::vector<Header> out;
    std::map<std::string, long> elements;  // per array of tables, its element count so far
    for (size_t i = 0; i < lines.size(); i++) {
        std::smatch m;
        if (!std::regex_match(lines[i], m, pattern)) continue;
        const bool array = m[1].str() == "[[";
        std::vector<std::string> names;
        std::stringstream ss(m[2].str());
        std::string part;
        while (std::getline(ss, part, '.')) {
            part.erase(0, part.find_first_not_of(" \t\"'"));
            part.erase(part.find_last_not_of(" \t\"'") + 1);
            names.push_back(part);
        }
        Concrete key;
        std::string prefix;
        for (size_t k = 0; k < names.size(); k++) {
            prefix += (k ? "." : "") + names[k];
            const bool last = k + 1 == names.size();
            if (array && last) {
                key.emplace_back(names[k], elements[prefix]++);
                // A new element starts the arrays of tables inside it afresh
                for (auto it = elements.begin(); it != elements.end();) {
                    it = it->first.rfind(prefix + ".", 0) == 0 ? elements.erase(it) : std::next(it);
                }
            } else {
                const auto it = elements.find(prefix);
                key.emplace_back(names[k], it == elements.end() ? -1 : it->second - 1);
            }
        }
        out.push_back({i, key});
    }
    return out;
}

/** @brief The line after the last key of the table `key` (0-based), or nullopt if it has no header. */
std::optional<size_t> end_of_table(const std::vector<std::string> & lines, const Concrete & key) {
    const std::vector<Header> all = headers(lines);
    size_t begin = 0, end = all.empty() ? lines.size() : all.front().line;
    if (!key.empty()) {
        size_t h = 0;
        while (h < all.size() && all[h].key != key) h++;
        if (h == all.size()) return std::nullopt;
        begin = all[h].line + 1;
        end = h + 1 < all.size() ? all[h + 1].line : lines.size();
    }
    size_t last = begin;
    for (size_t i = begin; i < end; i++) {
        const size_t first = lines[i].find_first_not_of(" \t");
        if (first != std::string::npos && lines[i][first] != '#') last = i + 1;
    }
    return last;
}

/** @brief `key = value` on its own line (not a dotted key or an inline table): the column of the key. */
std::optional<size_t> key_column(const std::string & line, const std::string & name) {
    const std::regex pattern("^(\\s*)(" + name + "|\"" + name + "\"|'" + name + "')\\s*=");
    std::smatch m;
    if (!std::regex_search(line, m, pattern)) return std::nullopt;
    return m[1].length();
}

using OrderedValue = toml::ordered_value;

/** @brief One textual change for the first match of a rule, or false if the rule has none left. */
bool rewrite_one_key(std::vector<std::string> & lines, const KeyRule & rule, bool & impossible) {
    OrderedValue root = toml::parse_str<toml::ordered_type_config>(join_lines(lines));
    const std::vector<Concrete> found = matches(root, rule.from);
    if (found.empty()) return false;
    const Concrete & key = found.front();
    const OrderedValue & value = *at(root, key);
    const size_t first = value.location().first_line_number() - 1, last = value.location().last_line_number() - 1;
    const std::string name = key.back().first;
    const auto column = key_column(lines[first], name);
    if (!column) {
        impossible = true;
        return false;
    }
    const std::string indent = lines[first].substr(0, *column);
    if (rule.to.empty()) {
        for (size_t i = first; i <= last; i++) lines[i] = indent + "# " + lines[i].substr(std::min(*column, lines[i].size()));
        lines.insert(lines.begin() + static_cast<long>(first),
                     indent + "# Removed in Mallard " + rule.removed + " (" + rule.note + "):");
        return true;
    }
    const Concrete to = target(rule, key);
    std::vector<std::string> moved(lines.begin() + static_cast<long>(first), lines.begin() + static_cast<long>(last) + 1);
    const size_t equals = moved[0].find('=', *column);
    std::string rest = moved[0].substr(equals);  // "= value # comment"
    if (rule.convert) {
        const std::string old_text = bare_text(value);
        const size_t v = rest.find_first_not_of(" \t", 1);
        if (last != first || v == std::string::npos || rest.compare(v, old_text.size(), old_text) != 0) {
            impossible = true;
            return false;
        }
        rest = rest.substr(0, v) + rule.convert(old_text) + rest.substr(v + old_text.size());
    }
    if (parent(to) == parent(key)) {
        lines[first] = indent + to.back().first + " " + rest;
        return true;
    }
    // A new table in an element of an array of tables: a header of it after the element's own keys
    const Concrete table = parent(to);
    const bool in_element = !table.empty() && table.back().second < 0 && table.size() >= 2 &&
                            table[table.size() - 2].second >= 0 && at(root, parent(table)) != nullptr;
    if (at(root, table) == nullptr && !in_element) {
        for (const auto & step : to) {
            if (step.second >= 0) {
                impossible = true;
                return false;
            }
        }
    }
    moved[0] = to.back().first + " " + rest;
    lines.erase(lines.begin() + static_cast<long>(first), lines.begin() + static_cast<long>(last) + 1);
    const auto end = end_of_table(lines, table);
    if (end) {
        lines.insert(lines.begin() + static_cast<long>(*end), moved.begin(), moved.end());
    } else if (in_element) {
        const auto element_end = end_of_table(lines, parent(table));
        if (!element_end) {
            impossible = true;
            return false;
        }
        moved.insert(moved.begin(), {"", "[" + dotted(table) + "]"});
        lines.insert(lines.begin() + static_cast<long>(*element_end), moved.begin(), moved.end());
    } else {
        lines.push_back("");
        lines.push_back("[" + dotted(parent(to)) + "]");
        lines.insert(lines.end(), moved.begin(), moved.end());
    }
    return true;
}

bool rewrite_one_option(std::vector<std::string> & lines, const OptionRule & rule, bool & impossible) {
    OrderedValue root = toml::parse_str<toml::ordered_type_config>(join_lines(lines));
    for (const Concrete & key : matches(root, rule.key)) {
        const OrderedValue & value = *at(root, key);
        if (!value.is_string() || value.as_string() != rule.from) continue;
        const size_t line = value.location().first_line_number() - 1;
        const auto column = key_column(lines[line], key.back().first);
        const std::string old_text = bare_text(value);
        const size_t v = column ? lines[line].find(old_text, *column) : std::string::npos;
        if (v == std::string::npos) {
            impossible = true;
            return false;
        }
        lines[line].replace(v, old_text.size(), toml::format(OrderedValue(rule.to)));
        const OrderedValue & siblings = *at(root, parent(key));
        size_t after = line + 1;
        for (const auto & [name, text] : rule.adds) {
            if (entries_of(siblings).contains(name)) continue;
            lines.insert(lines.begin() + static_cast<long>(after++), lines[line].substr(0, *column) + name + " = " + text);
        }
        return true;
    }
    return false;
}

/** @brief The line of the first match of an array rule removed and its elements inserted, or false if none is left. */
bool rewrite_one_array(std::vector<std::string> & lines, const ArrayRule & rule, bool & impossible) {
    OrderedValue root = toml::parse_str<toml::ordered_type_config>(join_lines(lines));
    const std::vector<Concrete> found = matches(root, rule.from);
    if (found.empty()) return false;
    const Concrete & key = found.front();
    const OrderedValue & value = *at(root, key);
    const size_t first = value.location().first_line_number() - 1, last = value.location().last_line_number() - 1;
    const Concrete to = plain_path(rule.to);
    const OrderedValue * existing = at(root, to);
    if (!key_column(lines[first], key.back().first) || (existing != nullptr && !existing->is_array())) {
        impossible = true;
        return false;
    }
    std::string comment;
    if (const size_t hash = lines[last].find('#', last == first ? lines[first].find('=') : 0); hash != std::string::npos) {
        comment = "  " + lines[last].substr(hash);
    }
    std::vector<std::string> block;
    for (const auto & values : array_elements(root, rule, key)) {
        block.push_back("");
        block.push_back("[[" + rule.to + "]]");
        block.push_back(rule.key + " = " + input_detail::toml_text(values) + comment);
        comment.clear();
    }
    lines.erase(lines.begin() + static_cast<long>(first), lines.begin() + static_cast<long>(last) + 1);
    Concrete after = parent(key);
    if (existing != nullptr && existing->size() > 0) {
        after = to;
        after.back().second = static_cast<long>(existing->size()) - 1;
    }
    const auto end = end_of_table(lines, after);
    if (!end || (existing != nullptr && existing->size() == 0)) {
        impossible = true;
        return false;
    }
    lines.insert(lines.begin() + static_cast<long>(*end), block.begin(), block.end());
    return true;
}

} // namespace

std::string advice(const Finding & finding) {
    return describe(finding) + "; `mallard input upgrade FILE > NEW_FILE` rewrites the file";
}

std::string describe(const Finding & finding) {
    return (finding.line ? "line " + std::to_string(finding.line) + ": " : std::string()) + finding.message;
}

std::vector<Finding> apply(Input & input, const std::string & version, const Rules & table) {
    std::vector<Finding> findings = transform(input, table, version.empty() ? MALLARD_VERSION : version, true);
    std::string removed;
    for (const Finding & f : findings) {
        if (f.removed) removed += "\n  " + describe(f);
    }
    if (!removed.empty()) {
        throw InputError("the input uses keys or options that this release no longer reads:" + removed +
                         "\n`mallard input upgrade FILE > NEW_FILE` rewrites them.");
    }
    return findings;
}

std::string upgrade_text(const std::string & text, std::vector<Finding> & findings,
                         bool & reformatted, const Rules & table) {
    reformatted = false;
    Input expected = parse_input_string(text);
    findings = transform(expected, table, MALLARD_VERSION);
    if (findings.empty()) return text;

    std::vector<std::string> lines = split_lines(text);
    bool impossible = false;
    for (const KeyRule & rule : table.keys) {
        for (int guard = 0; guard < 100000 && !impossible && rewrite_one_key(lines, rule, impossible); guard++) {}
    }
    for (const OptionRule & rule : table.options) {
        for (int guard = 0; guard < 100000 && !impossible && rewrite_one_option(lines, rule, impossible); guard++) {}
    }
    for (const ArrayRule & rule : table.arrays) {
        for (int guard = 0; guard < 100000 && !impossible && rewrite_one_array(lines, rule, impossible); guard++) {}
    }
    if (!impossible) {
        try {
            if (same(parse_input_string(join_lines(lines)), expected)) return join_lines(lines);
        } catch (const std::exception &) {
        }
    }
    reformatted = true;
    OrderedValue ordered = toml::parse_str<toml::ordered_type_config>(text);
    transform(ordered, table, MALLARD_VERSION);
    return toml::format(ordered);
}

int upgrade_file(const std::string & file, bool check) {
    std::ifstream in(file, std::ios::binary);
    if (!in) {
        std::cerr << "mallard input upgrade: cannot open " << file << "\n";
        return 1;
    }
    const std::string text((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    try {
        parse_input(file);  // syntax errors with the file's name
        std::vector<Finding> findings;
        bool reformatted = false;
        const std::string upgraded = upgrade_text(text, findings, reformatted);
        for (const Finding & f : findings) std::cerr << file << ": " << describe(f) << "\n";
        if (reformatted) {
            std::cerr << file << ": a changed key is a dotted key or in an inline table, so the whole file was "
                      << "reformatted (comments are kept, layout is not)\n";
        }
        if (check) return findings.empty() ? 0 : 1;
        std::cout << upgraded;
    } catch (const std::exception & e) {
        std::cerr << "mallard input upgrade: " << file << ": " << e.what() << "\n";
        return 1;
    }
    return 0;
}

} // namespace input_upgrade

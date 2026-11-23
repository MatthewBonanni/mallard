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

#include <algorithm>
#include <array>
#include <cctype>
#include <fstream>
#include <functional>
#include <iostream>
#include <limits>
#include <map>
#include <optional>
#include <regex>
#include <sstream>
#include <stdexcept>

#include "input.h"
#include "version.h"
#include "diagnostic_code.h"
#include "log.h"

namespace input_upgrade {

namespace {

std::string plus_one(const std::string & text) {
    const Input value = parse_input_string("x = " + text).at("x");
    if (!value.is_integer()) throw std::invalid_argument("is " + text + ", but must be an integer");
    if (value.as_integer() == std::numeric_limits<int64_t>::max()) throw std::invalid_argument(text + " is out of range");
    return std::to_string(value.as_integer() + 1);
}

// geometry = "all" wrote the whole mesh, which region calls "domain"
std::string all_to_domain(const std::string & text) { return text == "\"all\"" || text == "'all'" ? "\"domain\"" : text; }

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
    static const std::string FIELDS = "[[write_data]], [[probes]] and [statistics] all name their output fields";
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
            {"write_data[].geometry", "write_data[].region", "0.9.0", "0.10.0",
             "region takes any region: a boundary zone by name, a named or inline region, or \"domain\"",
             all_to_domain},
            {"mesh.curved[].shape", "mesh.curved[].type", "0.9.0", "0.10.0",
             "type selects the kind of every region; a curved zone lies on a surface (2D: a curve)"},
            {"write_data[].variables", "write_data[].fields", "0.10.0", "0.11.0", FIELDS},
            {"probes[].variables", "probes[].fields", "0.10.0", "0.11.0", FIELDS},
        },
        {
            {"write_data[].format", "tecplot", "vtu", "0.2.0", "0.2.0", "Tecplot output was removed"},
            {"write_data[].format", "hdf5", "vtkhdf", "0.9.0", "0.10.0",
             "the default, for ParaView 5.12 and later; or \"xdmf\", the XDMF indexes over HDF5 files, for VisIt, "
             "meshio and older ParaView", true},
            {"boundaries[].type", "upt", "inlet_state", "0.9.0", "0.10.0", TYPES},
            {"boundaries[].type", "p_out", "outlet_pressure", "0.9.0", "0.10.0", TYPES},
            {"boundaries[].type", "p_out_average", "outlet_pressure", "0.9.0", "0.10.0", TYPES, false, 0,
             "average = true"},
            {"boundaries[].type", "extrapolation", "outlet_extrapolation", "0.9.0", "0.10.0", TYPES},
            {"boundaries[].type", "nscbc_inlet", "inlet_state", "0.9.0", "0.10.0", TYPES + "; " + NSCBC, false, 0,
             "nscbc = {}"},
            {"boundaries[].type", "nscbc_outlet", "outlet_pressure", "0.9.0", "0.10.0", TYPES + "; " + NSCBC, false, 0,
             "nscbc = {}"},
            {"mesh.curved[].shape", "sphere", "sphere", "0.9.0", "0.10.0",
             "a sphere is a volume: filled = false, added, is its surface", false, 3, "filled = false"},
            {"mesh.curved[].shape", "cylinder", "cylinder", "0.9.0", "0.10.0",
             "a cylinder is a volume: filled = false, added, is its surface", false, 3, "filled = false"},
            {"numerics.preset", "robust", "second_order", "0.10.0", "0.11.0", "presets are named by their order"},
            {"numerics.preset", "accurate", "high_order", "0.10.0", "0.11.0", "presets are named by their order"},
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

/**
 * @brief Parses TOML text. Not toml::parse_str: its default std::source_location
 *        argument points at another object when GCC builds for macOS (#692).
 */
template <typename TC>
toml::basic_value<TC> parse_toml(const std::string & text) {
    std::istringstream stream(text);
    return toml::parse<TC>(stream, "input");
}

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

// The release whose removals this source has made, which rejects them before its version bump
constexpr const char * REMOVALS = "0.11.0";

std::string this_release() { return is_removed(REMOVALS, MALLARD_VERSION) ? MALLARD_VERSION : REMOVALS; }

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
    const bool kept = rule.from == rule.to;
    f.suggestion = kept ? rule.add : f.key + " = \"" + rule.to + "\"";
    f.message = f.key + " = \"" + rule.from + "\"" +
                (f.removed ? " was removed in " + rule.removed
                           : " is deprecated since " + rule.since + " and will be removed in " + rule.removed) +
                (kept ? ": add " + rule.add
                      : ": use \"" + rule.to + "\"" + (rule.add.empty() ? "" : " with " + rule.add)) +
                " (" + rule.note + ")";
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
 * @brief Whether the rule applies to a run of dimension dim (0: unknown). Throws
 *        if it depends on the dimension, which is unknown, and the input has its option.
 */
template <typename V>
bool applies(V & root, const OptionRule & rule, const int dim, const std::string & unknown_dimension) {
    if (rule.dim == 0 || rule.dim == dim) return true;
    if (dim != 0) return false;
    for (const Concrete & key : matches(root, rule.key)) {
        const V & value = *at(root, key);
        if (value.is_string() && value.as_string() == rule.from) {
            throw InputError(to_string(key) + " = \"" + rule.from + "\" is renamed differently in 2D and 3D runs, and "
                             "Mallard can't tell 2D or 3D" +
                             (unknown_dimension.empty() ? std::string() : ": " + unknown_dimension) +
                             "; run from the directory you run the case in, or pass --dim 2|3.");
        }
    }
    return false;
}

/** @brief Whether a rule that keeps its option and only adds a key finds that key set already. */
template <typename V>
bool added_already(V & root, const OptionRule & rule, const Concrete & key) {
    return rule.from == rule.to && entries_of(*at(root, parent(key))).contains(rule.add.substr(0, rule.add.find(' ')));
}

/**
 * @brief Applies every rule to the parsed input; returns what changed. Throws
 *        if a key is set under both its old and its new name. Option rules
 *        come first, so that they can match a key by the old name that a key
 *        rule then renames.
 *        `setup`: as the run reads it (kept options keep their value).
 */
template <typename V>
std::vector<Finding> transform(V & root, const Rules & table, const std::string & version, const int dim,
                               bool setup = false, const std::string & unknown_dimension = "") {
    std::vector<Finding> findings, option_findings;
    for (const OptionRule & rule : table.options) {
        if (!applies(root, rule, dim, unknown_dimension)) continue;
        for (const Concrete & key : matches(root, rule.key)) {
            V & value = *at(root, key);
            if (!value.is_string() || value.as_string() != rule.from || added_already(root, rule, key)) continue;
            option_findings.push_back(option_finding(rule, key, line_of(value), version));
            if (rule.from != rule.to && !(setup && rule.kept && !option_findings.back().removed)) value = rule.to;
            if (!rule.add.empty()) {
                const auto added = parse_toml<typename V::config_type>(rule.add);
                auto & beside = entries_of(*at(root, parent(key)));
                for (const auto & [name, v] : entries_of(added)) {
                    if (!beside.contains(name)) beside[name] = v;
                }
            }
        }
    }
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
                    std::string text;
                    try {
                        text = rule.convert(bare_text(value));
                    } catch (const std::invalid_argument & e) {
                        throw InputError(Diagnostic{.key = to_string(key), .line = line_of(value), .message = e.what()});
                    }
                    value = parse_toml<typename V::config_type>("x = " + text).at("x");
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
    findings.insert(findings.end(), option_findings.begin(), option_findings.end());
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
    OrderedValue root = parse_toml<toml::ordered_type_config>(join_lines(lines));
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
    // An empty inline table there (an option rule's `add`, e.g. nscbc = {}) gives way to the header
    if (const OrderedValue * existing = at(root, table);
        existing && existing->is_table() && existing->size() == 0 && existing->location().is_ok()) {
        const size_t line = existing->location().first_line_number() - 1;
        if (line == existing->location().last_line_number() - 1 && key_column(lines[line], table.back().first)) {
            lines.erase(lines.begin() + static_cast<long>(line));
            return true;
        }
    }
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
    OrderedValue root = parse_toml<toml::ordered_type_config>(join_lines(lines));
    for (const Concrete & key : matches(root, rule.key)) {
        const OrderedValue & value = *at(root, key);
        if (!value.is_string() || value.as_string() != rule.from || added_already(root, rule, key)) continue;
        const size_t line = value.location().first_line_number() - 1;
        const size_t v = value.location().first_column_number() - 1;
        const std::string old_text = bare_text(value);
        if (lines[line].compare(v, old_text.size(), old_text) != 0) {
            impossible = true;
            return false;
        }
        const std::string new_text = toml::format(OrderedValue(rule.to));
        lines[line].replace(v, old_text.size(), new_text);
        if (!rule.add.empty() && !entries_of(*at(root, parent(key))).contains(rule.add.substr(0, rule.add.find(' ')))) {
            const auto column = key_column(lines[line], key.back().first);
            const size_t after = v + new_text.size(), rest = lines[line].find_first_not_of(" \t", after);
            if (column && (rest == std::string::npos || lines[line][rest] == '#')) {
                lines.insert(lines.begin() + static_cast<long>(line) + 1, lines[line].substr(0, *column) + rule.add);
            } else {
                lines[line].insert(after, ", " + rule.add);
            }
        }
        return true;
    }
    return false;
}

/** @brief The line of the first match of an array rule removed and its elements inserted, or false if none is left. */
bool rewrite_one_array(std::vector<std::string> & lines, const ArrayRule & rule, bool & impossible) {
    OrderedValue root = parse_toml<toml::ordered_type_config>(join_lines(lines));
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

// ---------------------------------------------------------------- uppercase particle quantity names

const std::regex UPPER_NAME("[A-Z][A-Z0-9_]*");

std::string upper(std::string s) {
    for (char & c : s) c = static_cast<char>(std::toupper(static_cast<unsigned char>(c)));
    return s;
}

/** @brief The particle variables, uppercased: a quantity cannot take their names. */
const std::vector<std::string> & particle_variables() {
    static const std::vector<std::string> v = {"X", "Y", "Z", "T", "RHO", "U_X", "U_Y", "U_Z",
                                               "P", "V_X", "V_Y", "V_Z", "T_P"};
    return v;
}

/**
 * @brief text with the identifiers of `renames` renamed, but not those that
 *        call a function (`name(`) or name a region (`in(name)`).
 */
std::string rename_identifiers(const std::string & text, const std::map<std::string, std::string> & renames) {
    static const std::regex region_call(R"(\bin\s*\(\s*$)");
    std::string out;
    size_t i = 0;
    while (i < text.size()) {
        const auto c = static_cast<unsigned char>(text[i]);
        const bool starts = (std::isalpha(c) || c == '_') &&
                            (i == 0 || !(std::isalnum(static_cast<unsigned char>(text[i - 1])) || text[i - 1] == '_' ||
                                         text[i - 1] == '.'));
        if (!starts) {
            out += text[i++];
            continue;
        }
        size_t j = i;
        while (j < text.size() && (std::isalnum(static_cast<unsigned char>(text[j])) || text[j] == '_')) j++;
        const std::string word = text.substr(i, j - i);
        size_t k = j;
        while (k < text.size() && std::isspace(static_cast<unsigned char>(text[k]))) k++;
        const bool call = k < text.size() && text[k] == '(';
        const auto it = renames.find(word);
        out += it == renames.end() || call || std::regex_search(text.substr(0, i), region_call) ? word : it->second;
        i = j;
    }
    return out;
}

constexpr const char * QUANTITY_EXPRESSIONS[] = {"integral", "max", "min", "first_time", "while"};

/** @brief The renames of the lowercase quantity names of root, and the findings, also for names it cannot rename. */
std::map<std::string, std::string> quantity_renames(OrderedValue & root, std::vector<Finding> & findings) {
    std::map<std::string, std::string> renames;
    const std::vector<Concrete> keys = matches(root, "particles.quantities[].name");
    std::vector<std::string> taken;
    for (const Concrete & key : keys) {
        const OrderedValue & v = *at(root, key);
        if (v.is_string()) taken.push_back(v.as_string());
    }
    for (const Concrete & key : keys) {
        const OrderedValue & v = *at(root, key);
        if (!v.is_string() || std::regex_match(v.as_string(), UPPER_NAME)) continue;
        const std::string from = v.as_string(), to = upper(from);
        const auto & vars = particle_variables();
        const bool clash = std::find(vars.begin(), vars.end(), to) != vars.end() ||
                           std::count_if(taken.begin(), taken.end(), [&](const std::string & t) { return upper(t) == to; }) > 1;
        Finding f{.key = to_string(key), .line = line_of(v), .removed = true};
        if (!std::regex_match(to, UPPER_NAME) || clash) {
            f.message = to_string(key) + " \"" + from + "\": particle quantity names are uppercase, like field names, " +
                        "and \"" + to + "\" " + (clash ? "is taken" : "is not one") + "; rename it and its references by hand";
        } else {
            f.message = to_string(key) + " \"" + from + "\" is now \"" + to + "\" with its references: particle " +
                        "quantity names are uppercase, like field names";
            f.suggestion = to;
            renames[from] = to;
        }
        findings.push_back(f);
    }
    return renames;
}

/** @brief Renames the quantities and their references in the expressions of quantities and particle probes. */
void rename_quantities(OrderedValue & root, const std::map<std::string, std::string> & renames) {
    const auto rename = [&](OrderedValue & v) {
        if (v.is_string()) v.as_string() = rename_identifiers(v.as_string(), renames);
    };
    for (const char * key : {"name", "integral", "max", "min", "first_time", "while"}) {
        for (const Concrete & c : matches(root, std::string("particles.quantities[].") + key)) rename(*at(root, c));
    }
    for (const Concrete & c : matches(root, "particles.probes[].fields")) {
        OrderedValue & fields = *at(root, c);
        if (!fields.is_array()) continue;
        for (size_t i = 0; i < fields.size(); i++) rename(fields.at(i));
    }
}

/** @brief The strings of a line of TOML with rename applied to their contents. */
std::string rename_in_strings(const std::string & line, const std::map<std::string, std::string> & renames) {
    std::string out;
    size_t i = 0;
    while (i < line.size()) {
        if (line[i] == '#') return out + line.substr(i);
        if (line[i] != '"' && line[i] != '\'') {
            out += line[i++];
            continue;
        }
        const char quote = line[i];
        size_t j = i + 1;
        while (j < line.size() && line[j] != quote) j += quote == '"' && line[j] == '\\' ? 2 : 1;
        if (j >= line.size()) return out + line.substr(i);
        out += quote + rename_identifiers(line.substr(i + 1, j - i - 1), renames) + quote;
        i = j + 1;
    }
    return out;
}

/** @brief The renames that apply to the strings of a key of a table (dotted, of its header), or nullptr. */
using RenamesOf = std::function<const std::map<std::string, std::string> *(const std::string & table,
                                                                           const std::string & key)>;

/** @brief The lines with renames_of's renames applied to the strings of the keys it names. */
std::vector<std::string> rename_lines(std::vector<std::string> lines, const RenamesOf & renames_of) {
    static const std::regex header(R"(^\s*\[\[?\s*([^\[\]#]+?)\s*\]\]?\s*(#.*)?$)");
    static const std::regex assignment(R"(^\s*([A-Za-z0-9_]+)\s*=)");
    std::string table, key;
    for (std::string & line : lines) {
        std::smatch m;
        if (std::regex_match(line, m, header)) {
            table = m[1].str();
            key.clear();
            continue;
        }
        if (std::regex_search(line, m, assignment)) key = m[1].str();
        if (const auto * renames = renames_of(table, key)) line = rename_in_strings(line, *renames);
    }
    return lines;
}

/**
 * @brief The lines of the tables [[particles.quantities]] and
 *        [[particles.probes]] (and of [particles], for inline arrays) with
 *        rename applied to the strings of the keys that name or read quantities.
 */
std::vector<std::string> rename_quantity_lines(const std::vector<std::string> & lines,
                                               const std::map<std::string, std::string> & renames) {
    return rename_lines(lines, [&](const std::string & table, const std::string & key) {
        const bool quantity = table == "particles.quantities" &&
                              (key == "name" || std::find(std::begin(QUANTITY_EXPRESSIONS), std::end(QUANTITY_EXPRESSIONS),
                                                          key) != std::end(QUANTITY_EXPRESSIONS));
        const bool probe = table == "particles.probes" && key == "fields";
        const bool inline_array = table == "particles" && (key == "quantities" || key == "probes");
        return quantity || probe || inline_array ? &renames : nullptr;
    });
}

/** @brief upgrade_text's step for quantity names: text with them uppercase, or reformatted if that cannot be kept in place. */
std::string uppercase_quantities(const std::string & text, std::vector<Finding> & findings, bool & reformatted) {
    OrderedValue expected = parse_toml<toml::ordered_type_config>(text);
    const std::map<std::string, std::string> renames = quantity_renames(expected, findings);
    if (renames.empty()) return text;
    rename_quantities(expected, renames);
    const std::string rewritten = join_lines(rename_quantity_lines(split_lines(text), renames));
    try {
        if (same(parse_toml<toml::ordered_type_config>(rewritten), expected)) return rewritten;
    } catch (const std::exception &) {
    }
    reformatted = true;
    return toml::format(expected);
}

// ---------------------------------------------------------------- uppercase monitored values and columns

const std::vector<std::string> INTEGRAL_COLUMNS = {
    "kinetic_energy",      "enstrophy",       "dilatation_squared", "pressure_dilatation", "velocity_squared",
    "vorticity_squared",   "density_squared", "temperature",        "temperature_squared", "ke_rate_convective",
    "ke_rate_viscous",     "ke_rate_sgs",     "eps_numerical",      "pressure_work",       "les_C",
    "tf_beta"};

/** @brief The reduction column a lowercase one became (mean_P: MEAN_P), or empty. */
std::string uppercase_column(const std::string & name) {
    static const std::regex reduction("(integral|mean|favre|min|max|rms|moment)_(\\w+)");
    std::smatch m;
    return std::regex_match(name, m, reduction) ? upper(m[1].str()) + "_" + m[2].str() : "";
}

/**
 * @brief The monitored value a lowercase one became, or empty: integrals_<column>,
 *        forces_<zone>_F<axis>_<kind>, convergence_<column> and the reductions of
 *        the probes named.
 */
std::string uppercase_monitor(const std::string & name, const std::vector<std::string> & probes) {
    static const std::regex integral("integrals_(\\w+)"), force("forces_(\\w+)_F([xyz])_(pressure|viscous)"),
        convergence("convergence_(stationary|t_stationary|converged)"), copy("(copy[0-9]+_)?(\\w+)");
    std::smatch m;
    if (std::regex_match(name, m, integral) &&
        std::find(INTEGRAL_COLUMNS.begin(), INTEGRAL_COLUMNS.end(), m[1].str()) != INTEGRAL_COLUMNS.end()) {
        return "integrals_" + upper(m[1].str());
    }
    if (std::regex_match(name, m, force)) {
        return "forces_" + m[1].str() + "_F_" + upper(m[2].str()) + "_" + upper(m[3].str());
    }
    if (std::regex_match(name, m, convergence)) return "convergence_" + upper(m[1].str());
    const std::string prefix = "convergence_";
    if (name.starts_with(prefix)) {
        for (const std::string suffix : {"variance_ci", "mean", "ci", "variance", "tau", "n_eff", "samples"}) {
            if (!name.ends_with("_" + suffix) || name.size() <= prefix.size() + suffix.size() + 1) continue;
            const std::string signal = name.substr(prefix.size(), name.size() - prefix.size() - suffix.size() - 1);
            const std::string renamed = uppercase_monitor(signal, probes);
            return prefix + (renamed.empty() ? signal : renamed) + "_" + upper(suffix);
        }
    }
    for (const std::string & probe : probes) {
        if (!name.starts_with(probe + "_")) continue;
        const std::string rest = name.substr(probe.size() + 1);
        if (!std::regex_match(rest, m, copy)) continue;
        const std::string column = uppercase_column(m[2].str());
        if (!column.empty()) return probe + "_" + m[1].str() + column;
    }
    return "";
}

/** @brief The identifiers of an expression or message. */
std::vector<std::string> identifiers(const std::string & text) {
    static const std::regex word("[A-Za-z_][A-Za-z0-9_]*");
    std::vector<std::string> out;
    for (auto it = std::sregex_iterator(text.begin(), text.end(), word); it != std::sregex_iterator(); ++it) {
        out.push_back(it->str());
    }
    return out;
}

/** @brief The strings of a value: itself, or the strings of an array. */
std::vector<OrderedValue *> strings_of(OrderedValue & v) {
    std::vector<OrderedValue *> out;
    if (v.is_string()) out.push_back(&v);
    if (v.is_array()) {
        for (size_t i = 0; i < v.size(); i++) {
            if (v.at(i).is_string()) out.push_back(&v.at(i));
        }
    }
    return out;
}

/**
 * @brief upgrade_text's step for the names of monitored values and of probe
 *        reduction columns that became uppercase (integrals_kinetic_energy:
 *        integrals_KINETIC_ENERGY) in the keys that read them: text with them
 *        renamed, or reformatted if that cannot be kept in place.
 */
std::string uppercase_monitors(const std::string & text, std::vector<Finding> & findings, bool & reformatted) {
    OrderedValue expected = parse_toml<toml::ordered_type_config>(text);
    std::vector<std::string> probes;
    for (const Concrete & c : matches(expected, "probes[].name")) {
        if (at(expected, c)->is_string()) probes.push_back(at(expected, c)->as_string());
    }
    // Longest first, so that a probe whose name extends another's takes its own columns
    std::sort(probes.begin(), probes.end(),
              [](const std::string & a, const std::string & b) { return a.size() > b.size(); });
    std::map<std::string, std::string> monitors, columns;
    bool renamed = false;
    const auto rename = [&](const std::string & path, std::map<std::string, std::string> & renames,
                            const std::function<std::string(const std::string &)> & uppercase, const bool expressions) {
        for (const Concrete & c : matches(expected, path)) {
            std::vector<std::string> changes;
            for (OrderedValue * v : strings_of(*at(expected, c))) {
                const std::string before = v->as_string();
                std::vector<std::string> found;
                for (const std::string & word : expressions ? identifiers(before) : std::vector<std::string>{before}) {
                    const std::string to = renames.contains(word) ? renames[word] : uppercase(word);
                    if (to.empty()) continue;
                    renames[word] = to;
                    found.push_back("\"" + word + "\" is now \"" + to + "\"");
                }
                v->as_string() = rename_identifiers(before, renames);
                if (v->as_string() == before) continue;
                for (const auto & f : found) {
                    if (std::find(changes.begin(), changes.end(), f) == changes.end()) changes.push_back(f);
                }
            }
            if (changes.empty()) continue;
            renamed = true;
            std::string list;
            for (size_t i = 0; i < changes.size(); i++) list += (i == 0 ? "" : ", ") + changes[i];
            findings.push_back({.key = to_string(c),
                                .line = line_of(*at(expected, c)),
                                .message = to_string(c) + ": " + list + ": monitored values and output columns are " +
                                           "uppercase, like field names",
                                .removed = true});
        }
    };
    const auto monitor = [&](const std::string & name) { return uppercase_monitor(name, probes); };
    rename("triggers[].when", monitors, monitor, true);
    rename("triggers[].do", monitors, monitor, true);
    rename("statistics.convergence.signals", monitors, monitor, false);
    rename("probes[].spectra.fields", columns, uppercase_column, false);
    if (!renamed) return text;
    const auto renames_of = [&](const std::string & table,
                                const std::string & key) -> const std::map<std::string, std::string> * {
        if ((table == "triggers" && (key == "when" || key == "do")) ||
            (table == "statistics.convergence" && key == "signals") || (table == "statistics" && key == "convergence")) {
            return &monitors;
        }
        if ((table == "probes" && key == "spectra") || (table == "probes.spectra" && key == "fields")) return &columns;
        return nullptr;
    };
    const std::string rewritten = join_lines(rename_lines(split_lines(text), renames_of));
    try {
        if (same(parse_toml<toml::ordered_type_config>(rewritten), expected)) return rewritten;
    } catch (const std::exception &) {
    }
    reformatted = true;
    return toml::format(expected);
}

} // namespace

std::vector<std::string> notices(const std::string & text) {
    const Input input = parse_input_string(text);
    const auto has = [&](const char * table, const char * key) {
        return input.contains(table) && input.at(table).is_table() && input.at(table).contains(key);
    };
    std::vector<std::string> out;
    if (input.contains("reactor")) {
        out.push_back("[reactor]: mallard reactor writes the pressure column as P, not p, as field names are uppercase; "
                      "scripts reading its output by column name need the new name");
    }
    if (input.contains("run") && !has("numerics", "preset") && !has("numerics", "face_reconstruction")) {
        out.push_back("no preset and no [numerics.face_reconstruction]: since 0.10.0 this runs the high_order "
                      "preset, not first order; add [numerics.face_reconstruction] type = \"FO\" to reproduce "
                      "older first-order results");
    }
    if (input.contains("chemistry") && input.at("chemistry").is_table() && !has("chemistry", "fuse_half_steps")) {
        const Input & chemistry = input.at("chemistry");
        const bool off = chemistry.contains("enabled") && chemistry.at("enabled").is_boolean() &&
                         !chemistry.at("enabled").as_boolean();
        const bool simpler = chemistry.contains("coupling") && chemistry.at("coupling").is_string() &&
                             chemistry.at("coupling").as_string() == "simpler";
        if (!off && !simpler) {
            out.push_back("no chemistry.fuse_half_steps: since 0.10.0 Strang splitting fuses each step's closing "
                          "chemistry half step with the next step's opening one (results change within the "
                          "chemistry's tolerances); add fuse_half_steps = false to [chemistry] to reproduce older "
                          "results");
        }
    }
    if (input.contains("write_data") && input.at("write_data").is_array()) {
        for (const auto & entry : input.at("write_data").as_array()) {
            if (!entry.is_table() || entry.contains("series") || entry.contains("region") || entry.contains("regions")) {
                continue;
            }
            if (entry.contains("format") && !(entry.at("format").is_string() && entry.at("format").as_string() == "vtkhdf")) {
                continue;
            }
            const std::string prefix =
                entry.contains("prefix") && entry.at("prefix").is_string() ? entry.at("prefix").as_string() : "PREFIX";
            out.push_back("[[write_data]] \"" + prefix + "\" has no series: VTKHDF output writes the series file " + prefix +
                          ".vtkhdf over a file per snapshot by default; add series = false for each snapshot whole, " +
                          prefix + "_<n>.vtkhdf, indexed by " + prefix + ".vtkhdf.series");
        }
    }
    if (has("numerics", "preset") && input.at("numerics").at("preset").is_string() &&
        input.at("numerics").at("preset").as_string() == "les" && !has("numerics", "time_integrator")) {
        const bool sets_step = has("run", "cfl") || has("run", "dt");
        std::string note = "preset = \"les\": since 0.10.0 it runs SSPRK54 at cfl 2.0, not SSPRK3 at 0.8; add "
                           "[numerics] time_integrator = \"SSPRK3\"" +
                           std::string(sets_step ? "" : " and [run] cfl = 0.8") + " to reproduce older results";
        if (has("run", "cfl")) note += "; the cfl this input sets now applies to SSPRK54, which is cheapest near 2.0";
        out.push_back(note);
    }
    return out;
}

std::string advice(const Finding & finding) {
    return describe(finding) + "; `mallard input upgrade old.toml > new.toml` rewrites the file";
}

std::string describe(const Finding & finding) {
    return (finding.line ? "line " + std::to_string(finding.line) + ": " : std::string()) + finding.message;
}

std::vector<Finding> apply(Input & input, const std::string & version, const Rules & table) {
    std::vector<Finding> findings = transform(input, table, version.empty() ? this_release() : version, N_DIM, true);
    std::string removed;
    for (const Finding & f : findings) {
        if (f.removed) removed += "\n  " + describe(f);
    }
    if (!removed.empty()) {
        throw InputError("the input uses keys or options that this release no longer reads:" + removed +
                         "\n`mallard input upgrade old.toml > new.toml` rewrites them.");
    }
    return findings;
}

namespace {

std::string upgrade_rules(const std::string & text, std::vector<Finding> & findings, bool & reformatted,
                          const Rules & table, const int dim, const std::string & unknown_dimension) {
    reformatted = false;
    Input expected = parse_input_string(text);
    findings = transform(expected, table, this_release(), dim, false, unknown_dimension);
    if (findings.empty()) return text;

    std::vector<std::string> lines = split_lines(text);
    bool impossible = false;
    for (const OptionRule & rule : table.options) {
        if (rule.dim != 0 && rule.dim != dim) continue;
        for (int guard = 0; guard < 100000 && !impossible && rewrite_one_option(lines, rule, impossible); guard++) {}
    }
    for (const KeyRule & rule : table.keys) {
        for (int guard = 0; guard < 100000 && !impossible && rewrite_one_key(lines, rule, impossible); guard++) {}
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
    OrderedValue ordered = parse_toml<toml::ordered_type_config>(text);
    transform(ordered, table, this_release(), dim);
    return toml::format(ordered);
}

} // namespace

std::string upgrade_text(const std::string & text, std::vector<Finding> & findings,
                         bool & reformatted, const Rules & table, const int dim,
                         const std::string & unknown_dimension) {
    const std::string upgraded = upgrade_rules(text, findings, reformatted, table, dim, unknown_dimension);
    return uppercase_monitors(uppercase_quantities(upgraded, findings, reformatted), findings, reformatted);
}

int upgrade_file(const std::string & file, bool check, int dim, const std::string & unknown_dimension) {
    std::ifstream in(file, std::ios::binary);
    if (!in) {
        logging::error(diagnostic_code::INPUT_UPGRADE, "mallard input upgrade: cannot open " + file);
        return 1;
    }
    const std::string text((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    try {
        parse_input(file);  // syntax errors with the file's name
        std::vector<Finding> findings;
        bool reformatted = false;
        const std::string upgraded = upgrade_text(text, findings, reformatted, rules(), dim, unknown_dimension);
        for (const Finding & f : findings) std::cerr << file << ": " << describe(f) << "\n";
        for (const std::string & note : notices(text)) std::cerr << file << ": note: " << note << "\n";
        if (reformatted) {
            std::cerr << file << ": a changed key is a dotted key or in an inline table, so the whole file was "
                      << "reformatted (comments are kept, layout is not)\n";
        }
        if (check) return findings.empty() ? 0 : 1;
        std::cout << upgraded;
    } catch (const std::exception & e) {
        logging::error(diagnostic_code::INPUT_UPGRADE, "mallard input upgrade: " + file + ": " + e.what());
        return 1;
    }
    return 0;
}

} // namespace input_upgrade

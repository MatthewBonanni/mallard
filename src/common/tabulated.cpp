/**
 * @file tabulated.cpp
 * @author Matthew Bonanni (mbonanni001@gmail.com)
 * @brief Tabulated data: reading CSV files, building tables and point clouds.
 * @version 0.1
 * @date 2026-10-23
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 *
 */

#include "tabulated.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <limits>

#include "box_tree.h"
#include "common_io.h"
#include "input.h"
#include "json.h"
#include "log.h"
#include "provenance.h"
#include "units.h"

namespace tabulated {

namespace {

std::string trim(const std::string & s) {
    const size_t a = s.find_first_not_of(" \t\r\"'");
    if (a == std::string::npos) return "";
    const size_t b = s.find_last_not_of(" \t\r\"'");
    return s.substr(a, b - a + 1);
}

std::vector<std::string> split(const std::string & line) {
    std::vector<std::string> out;
    const bool commas = line.find(',') != std::string::npos;
    std::string item;
    for (const char c : line) {
        const bool separator = commas ? c == ',' : (c == ' ' || c == '\t');
        if (!separator) {
            item += c;
            continue;
        }
        if (commas || !trim(item).empty()) out.push_back(trim(item));
        item.clear();
    }
    if (commas || !trim(item).empty()) out.push_back(trim(item));
    return out;
}

template <typename T>
Kokkos::View<const T *, Kokkos::HostSpace> host_view(const std::vector<T> & v) {
    return Kokkos::View<const T *, Kokkos::HostSpace>(v.data(), v.size());
}

template <typename T>
Kokkos::View<const T *> device_view(const std::vector<T> & v, const std::string & label) {
    Kokkos::View<T *> d(label, v.size());
    Kokkos::deep_copy(d, Kokkos::View<const T *, Kokkos::HostSpace, Kokkos::MemoryUnmanaged>(v.data(), v.size()));
    return d;
}

void check_columns(const std::vector<std::vector<double>> & coordinates,
                   const std::vector<std::vector<double>> & fields, const std::string & where) {
    if (coordinates.empty() || coordinates.size() > MAX_DIMS) {
        throw InputError(where + ": give 1 to " + std::to_string(MAX_DIMS) + " coordinates.");
    }
    const size_t n = coordinates[0].size();
    if (n == 0) throw InputError(where + " has no rows.");
    for (const auto & c : coordinates) {
        if (c.size() != n) throw InputError(where + ": the coordinate columns differ in length.");
        for (const double v : c) {
            if (!std::isfinite(v)) throw InputError(where + " has a coordinate that is not a finite number.");
        }
    }
    for (const auto & f : fields) {
        if (f.size() != n) throw InputError(where + ": the columns differ in length.");
    }
}

std::string identifier(const std::string & name) {
    std::string out = name;
    for (char & c : out) {
        if (!std::isalnum(static_cast<unsigned char>(c)) && c != '_') c = '_';
    }
    return out;
}

/** @brief An invalid value of table.leaf (key: its full path). */
InputError invalid(const Input & table, const std::string & key, const std::string & leaf, const std::string & message,
                   const std::string & fix, const std::string & docs) {
    return InputError({.key = key, .line = line_of(table, leaf), .message = message, .fix = fix, .docs = docs});
}

/**
 * @brief The coordinate columns of entry e of a file with these columns: its
 *        coordinates, else the metadata's, else the default.
 */
std::vector<std::string> entry_coordinates(const Input & e, const std::vector<std::string> & columns, const bool points,
                                           const std::vector<std::string> & metadata) {
    if (e.contains("coordinates")) return find_value<std::vector<std::string>>(e, "coordinates");
    if (!metadata.empty()) return metadata;
    if (points) return N_DIM == 2 ? std::vector<std::string>{"x", "y"} : std::vector<std::string>{"x", "y", "z"};
    return {columns.empty() ? std::string() : columns[0]};
}

/** @brief The value columns of entry e: its columns, else every column but the coordinates. */
std::vector<std::string> entry_values(const Input & e, const std::vector<std::string> & columns,
                                      const std::vector<std::string> & coordinates) {
    if (contains_optional(e, "columns")) return find_value<std::vector<std::string>>(e, "columns");
    std::vector<std::string> values;
    for (const std::string & c : columns) {
        if (std::find(coordinates.begin(), coordinates.end(), c) == coordinates.end()) values.push_back(c);
    }
    return values;
}

const std::string METADATA = "mallard-profile";

/** @brief The metadata JSON of a comment line, or "" if it is not one. */
std::string metadata_of(const std::string & line) {
    const std::string open = "# " + METADATA + " ";
    const size_t a = line.find_first_not_of(" \t");
    return a != std::string::npos && line.compare(a, open.size(), open) == 0 ? line.substr(a + open.size()) : "";
}

/** @brief The scales a data entry rescales by, in the order of Metadata::reference's names. */
struct Scale {
    const char * name;
    Dimension dimension;
};
const std::array<Scale, 4> SCALES = {{{"length", dim::length},
                                      {"velocity", dim::velocity},
                                      {"density", dim::density},
                                      {"temperature", dim::temperature}}};

std::string scale_names() {
    std::string names;
    for (const Scale & s : SCALES) names += (names.empty() ? "" : ", ") + std::string(s.name);
    return names;
}

} // namespace

int32_t Csv::column(const std::string & name) const {
    const auto it = std::find(names.begin(), names.end(), name);
    return it == names.end() ? -1 : static_cast<int32_t>(it - names.begin());
}

Csv read_csv(const std::string & file, const std::string & docs) {
    const auto error = [&](const std::string & what) { return InputError(Diagnostic{.message = what, .docs = docs}); };
    std::ifstream in(file);
    if (!in) throw cannot_read("data file", file);
    provenance::read_file("table", file);
    Csv csv;
    std::string line;
    size_t line_number = 0;
    while (std::getline(in, line)) {
        line_number++;
        const std::string t = trim(line);
        if (!t.empty() && t[0] == '#' && csv.names.empty() && csv.metadata.empty()) csv.metadata = metadata_of(line);
        if (t.empty() || t[0] == '#') continue;
        const std::vector<std::string> items = split(line);
        if (csv.names.empty()) {
            csv.names = items;
            for (const auto & name : csv.names) {
                if (name.empty()) throw error(file + ":" + std::to_string(line_number) + ": a column has no name");
                if (std::count(csv.names.begin(), csv.names.end(), name) > 1) {
                    throw error(file + ":" + std::to_string(line_number) + ": column " + name + " appears twice");
                }
            }
            csv.columns.resize(csv.names.size());
            continue;
        }
        if (items.size() != csv.names.size()) {
            throw error(file + ":" + std::to_string(line_number) + ": " + logging::count(items.size(), "value") +
                        " for " + logging::count(csv.names.size(), "column"));
        }
        for (size_t i = 0; i < items.size(); i++) {
            char * end = nullptr;
            const double v = std::strtod(items[i].c_str(), &end);
            if (items[i].empty() || end != items[i].c_str() + items[i].size()) {
                throw error(file + ":" + std::to_string(line_number) + ": \"" + items[i] + "\" is not a number");
            }
            csv.columns[i].push_back(v);
        }
    }
    if (csv.names.empty() || csv.columns[0].empty()) throw error("Data file " + file + " has no rows");
    return csv;
}

std::vector<std::string> read_header(const std::string & file, std::string * metadata) {
    std::ifstream in(file);
    std::string line;
    while (std::getline(in, line)) {
        const std::string t = trim(line);
        if (metadata && metadata->empty() && !t.empty() && t[0] == '#') *metadata = metadata_of(line);
        if (!t.empty() && t[0] != '#') return split(line);
    }
    return {};
}

Metadata parse_metadata(const std::string & text, const std::string & file, const std::string & docs) {
    Metadata m;
    if (text.empty()) return m;
    const auto error = [&](const std::string & what) {
        return InputError(Diagnostic{.message = file + ": its " + METADATA + " comment " + what,
                                     .fix = "Correct the comment line \"# " + METADATA + " {...}\" before the "
                                            "file's header, or remove it",
                                     .docs = docs});
    };
    json::Value v;
    try {
        v = json::parse(text);
    } catch (const std::exception & err) {
        throw error(std::string("is not JSON (") + err.what() + ")");
    }
    if (v.type != json::Value::Type::OBJECT) throw error("is not a JSON object");
    const std::vector<std::string> keys = {"coordinates", "at", "units", "reference", "source", "transforms"};
    for (const std::string & key : v.names) {
        if (std::find(keys.begin(), keys.end(), key) == keys.end()) {
            throw error("has an unknown key \"" + key + "\" (keys: " + one_of(keys) + ")");
        }
    }
    const auto strings = [&](const std::string & key) {
        std::vector<std::string> out;
        const json::Value * a = v.find(key);
        if (!a) return out;
        if (a->type != json::Value::Type::ARRAY) throw error("gives " + key + " that is not a list of strings");
        for (const json::Value & item : a->array) {
            if (item.type != json::Value::Type::STRING) throw error("gives " + key + " that is not a list of strings");
            out.push_back(item.string);
        }
        return out;
    };
    m.coordinates = strings("coordinates");
    m.at = strings("at");
    m.transforms = strings("transforms");
    if (const json::Value * s = v.find("source")) {
        if (s->type != json::Value::Type::STRING) throw error("gives a source that is not a string");
        m.source = s->string;
    }
    if (const json::Value * u = v.find("units")) {
        if (u->type != json::Value::Type::OBJECT) throw error("gives units that are not an object of columns");
        for (size_t i = 0; i < u->names.size(); i++) {
            if (u->members[i].type != json::Value::Type::STRING) {
                throw error("gives the unit of column " + u->names[i] + " as something other than a string");
            }
            m.units.emplace_back(u->names[i], u->members[i].string);
        }
    }
    if (const json::Value * r = v.find("reference")) {
        if (r->type != json::Value::Type::OBJECT) throw error("gives a reference that is not an object of scales");
        for (size_t i = 0; i < r->names.size(); i++) {
            const std::string & name = r->names[i];
            const auto scale = std::find_if(SCALES.begin(), SCALES.end(), [&](const Scale & s) { return name == s.name; });
            if (scale == SCALES.end()) throw error("gives a reference " + name + " (scales: " + scale_names() + ")");
            const json::Value & value = r->members[i];
            double si = value.number;
            if (value.type == json::Value::Type::STRING) {
                try {
                    const Quantity q = parse_quantity(value.string);
                    if (q.has_unit && !q.dimension.same_as(scale->dimension)) {
                        throw error("gives the reference " + name + " \"" + value.string + "\", not a " +
                                    scale->dimension.describe());
                    }
                    si = q.value;
                } catch (const UnitError & err) {
                    throw error("gives the reference " + name + " " + err.what());
                }
            } else if (value.type != json::Value::Type::NUMBER) {
                throw error("gives the reference " + name + " as something other than a number or a quantity");
            }
            if (!(si > 0.0) || !std::isfinite(si)) throw error("gives a reference " + name + " that is not positive");
            m.reference.emplace_back(name, si);
        }
    }
    return m;
}

Entry read_entry(const Input & e, const std::string & where, const std::string & docs, Data & data) {
    if (!e.is_table()) {
        throw InputError({.key = where, .line = line_of(e), .message = "must be a table",
                          .fix = "Give the entry name and file", .docs = docs});
    }
    if (!e.contains("name")) throw missing_key(e, where + ".name", "Add name = \"...\": the prefix of the variables");
    if (!e.contains("file")) throw missing_key(e, where + ".file", "Add file = \"...\": a CSV file");
    Entry entry;
    entry.name = find_value<std::string>(e, "name");
    const std::string & name = entry.name;
    if (name.empty() || identifier(name) != name || std::isdigit(static_cast<unsigned char>(name[0]))) {
        throw invalid(e, where + ".name", "name", "\"" + name + "\" is not an identifier",
                      "Use letters, digits and _, starting with a letter: the variables are <name>_<column>", docs);
    }
    entry.file = find_value<std::string>(e, "file");
    const std::string & file = entry.file;
    const std::string type = find_value_or<std::string>(e, "type", "table");
    record_options(where + ".type", {"table", "points"});
    if (type != "table" && type != "points") throw not_accepted(e, where + ".type", type, {"points", "table"});
    const bool points = type == "points";
    const Csv csv = read_csv(file, docs);
    const Metadata metadata = parse_metadata(csv.metadata, file, docs);
    if (!e.contains("coordinates")) entry.at = metadata.at;  // where the metadata's coordinates are read
    entry.coordinates = entry_coordinates(e, csv.names, points, metadata.coordinates);
    const std::vector<std::string> & coordinates = entry.coordinates;
    if (coordinates.empty() || coordinates.size() > MAX_DIMS) {
        throw invalid(e, where + ".coordinates", "coordinates", "names " + std::to_string(coordinates.size()) + " columns",
                      "Name 1 to 3 columns of " + file, docs);
    }
    const auto no_column = [&](const std::string & key, const Input & table, const std::string & leaf,
                               const std::string & c, const std::string & fix) {
        return InputError({.key = key,
                           .line = line_of(table, leaf),
                           .message = file + " has no column " + c,
                           .valid = csv.names,
                           .suggestion = suggest_value(c, csv.names),
                           .fix = fix,
                           .docs = docs});
    };
    std::vector<int32_t> coordinate_columns;
    for (const auto & c : coordinates) {
        const int32_t i = csv.column(c);
        if (i < 0) {
            throw no_column(where + ".coordinates", e, e.contains("coordinates") ? "coordinates" : "file", c,
                            "Name the coordinate columns with coordinates = [...]");
        }
        coordinate_columns.push_back(i);
    }
    std::vector<int32_t> value_columns;
    for (const std::string & c : entry_values(e, csv.names, coordinates)) {
        const int32_t i = csv.column(c);
        if (i < 0) throw no_column(where + ".columns", e, "columns", c, "Name value columns of the file");
        if (std::find(coordinate_columns.begin(), coordinate_columns.end(), i) != coordinate_columns.end() ||
            std::find(value_columns.begin(), value_columns.end(), i) != value_columns.end()) {
            throw invalid(e, where + ".columns", "columns", "names column " + c + " twice, or a coordinate",
                          "Name each value column once, and no coordinate column", docs);
        }
        value_columns.push_back(i);
    }
    if (value_columns.empty()) {
        throw invalid(e, where + ".file", "file", file + " has no columns besides the coordinates",
                      "Add value columns to the file, or name fewer coordinates", docs);
    }
    // Units of columns: the metadata's, then the entry's
    std::vector<std::string> unit(csv.names.size()), unit_key(csv.names.size());
    for (const auto & [column, text] : metadata.units) {
        const int32_t i = csv.column(column);
        if (i < 0) {
            throw InputError(Diagnostic{.message = file + ": its " + METADATA + " comment gives the unit of column " +
                                                   column + ", which the file does not have",
                                        .valid = csv.names,
                                        .docs = docs});
        }
        unit[size_t(i)] = text;
        unit_key[size_t(i)] = "the " + METADATA + " comment of " + file;
    }
    if (e.contains("units")) {
        const Input & units = e.at("units");
        if (!units.is_table()) {
            throw invalid(e, where + ".units", "units", "must be a table of columns and their units",
                          "Write units = {x = \"mm\", T = \"degC\"}", docs);
        }
        for (const auto & [column, value] : units.as_table()) {
            const int32_t i = csv.column(column);
            if (i < 0) throw no_column(where + ".units." + column, units, column, column, "");
            unit[size_t(i)] = find_value<std::string>(units, column);
            unit_key[size_t(i)] = where + ".units." + column;
        }
    }
    // As affine maps to SI (a + b v), and the dimension of each column with a unit
    std::vector<double> offset(csv.names.size(), 0.0), factor(csv.names.size(), 1.0);
    std::vector<Dimension> dimension(csv.names.size());
    for (size_t i = 0; i < csv.names.size(); i++) {
        if (unit[i].empty() || unit[i] == "1") continue;
        try {
            const Quantity q0 = parse_quantity("0 " + unit[i]), q1 = parse_quantity("1 " + unit[i]);
            offset[i] = q0.value;
            factor[i] = q1.value - q0.value;
            dimension[i] = q1.dimension;
        } catch (const UnitError & err) {
            throw InputError(Diagnostic{.key = unit_key[i],
                                        .line = e.contains("units") ? line_of(e.at("units"), csv.names[i]) : 0,
                                        .message = "column " + csv.names[i] + ": " + err.what(),
                                        .fix = "Give a unit such as \"mm\", \"bar\" or \"degC\" (\"1\" if dimensionless)",
                                        .docs = docs});
        }
    }
    // Rescaling: each scale over the reference's, the entry's or else the metadata's
    std::vector<double> scale(csv.names.size(), 1.0);
    std::string scaled_text;
    const bool scaled = contains_optional(e, "scale");
    if (contains_optional(e, "reference") && !scaled) {
        throw invalid(e, where + ".reference", "reference", "is given without scale",
                      "Add scale = {...}, the scales to rescale the data to, or remove reference", docs);
    }
    if (scaled) {
        const Input & to = e.at("scale");
        if (!to.is_table() || to.as_table().entries().empty()) {
            throw invalid(e, where + ".scale", "scale", "must be a table of scales (" + scale_names() + ")",
                          "Write scale = {length = 0.02, velocity = 15}", docs);
        }
        const Input * from = contains_optional(e, "reference") ? &e.at("reference") : nullptr;
        if (from && !from->is_table()) {
            throw invalid(e, where + ".reference", "reference", "must be a table of scales (" + scale_names() + ")",
                          "Write reference = {length = 0.01, velocity = 30}, the scales of the data", docs);
        }
        std::array<double, 4> f = {1.0, 1.0, 1.0, 1.0};
        std::string text;
        for (size_t k = 0; k < SCALES.size(); k++) {
            const Scale & sc = SCALES[k];
            if (!contains_optional(to, sc.name)) continue;
            const double target = find_double(to, sc.name, sc.dimension);
            double ref = 0.0;
            if (from && contains_optional(*from, sc.name)) {
                ref = find_double(*from, sc.name, sc.dimension);
            } else {
                for (const auto & [scale_name, value] : metadata.reference) {
                    if (scale_name == sc.name) ref = value;
                }
            }
            if (!(ref > 0.0)) {
                throw invalid(e, where + ".scale", "scale",
                              "rescales the " + std::string(sc.name) + ", but neither the entry nor the " + METADATA +
                                  " comment of " + file + " gives its reference " + sc.name,
                              "Add reference = {" + std::string(sc.name) + " = ...}, the " + sc.name +
                                  " the data are scaled by",
                              docs);
            }
            if (!(target > 0.0) || !std::isfinite(target)) {
                throw invalid(to, where + ".scale." + sc.name, sc.name, "must be positive", "", docs);
            }
            f[k] = target / ref;
            text += ", " + std::string(sc.name) + " x" + logging::real(f[k]);
        }
        const double f_mass = f[2] * f[0] * f[0] * f[0], f_time = f[0] / f[1];
        std::vector<int32_t> read = coordinate_columns;
        read.insert(read.end(), value_columns.begin(), value_columns.end());
        for (const int32_t c : read) {
            const size_t i = size_t(c);
            if (unit[i].empty()) {
                throw invalid(e, where + ".scale", "scale",
                              "needs the unit of every column it rescales, and column " + csv.names[i] + " has none",
                              "Give units = {" + csv.names[i] + " = \"m/s\"} (\"1\" if dimensionless), here or in "
                              "the file's " + METADATA + " comment",
                              docs);
            }
            const auto & x = dimension[i].exponent;
            scale[i] = std::pow(f_mass, x[Dimension::MASS] + x[Dimension::AMOUNT]) * std::pow(f[0], x[Dimension::LENGTH]) *
                       std::pow(f_time, x[Dimension::TIME]) * std::pow(f[3], x[Dimension::TEMPERATURE]);
        }
        scaled_text = ", scaled:" + text.substr(1);
    }
    auto column = [&](const size_t i) {
        std::vector<double> c = csv.columns[i];
        for (double & v : c) v = offset[i] + factor[i] * v;
        if (scaled) {
            for (double & v : c) v *= scale[i];
        }
        return c;
    };
    std::vector<std::vector<double>> coords, fields;
    for (const int32_t i : coordinate_columns) coords.push_back(column(size_t(i)));
    for (const int32_t i : value_columns) {
        fields.push_back(column(size_t(i)));
        entry.variables.push_back(name + "_" + identifier(csv.names[size_t(i)]));
    }
    if (points) {
        if (e.contains("interpolation")) {
            throw invalid(e, where + ".interpolation", "interpolation", "is for tables (type = \"table\")",
                          "Remove it; point clouds take neighbors = k", docs);
        }
        const uint32_t neighbors = find_value_or<uint32_t>(e, "neighbors", 1);
        entry.set = data.add_points(coords, fields, neighbors, where);
        entry.kind = logging::count(coords[0].size(), "point") + ", " +
                     (neighbors == 1 ? "nearest" : std::to_string(neighbors) + " nearest by inverse distance") + scaled_text;
    } else {
        if (e.contains("neighbors")) {
            throw invalid(e, where + ".neighbors", "neighbors", "is for point clouds (type = \"points\")",
                          "Remove it; tables take interpolation = \"linear\" or \"cubic\"", docs);
        }
        const std::string interpolation = find_value_or<std::string>(e, "interpolation", "linear");
        record_options(where + ".interpolation", {"linear", "cubic"});
        if (interpolation != "linear" && interpolation != "cubic") {
            throw not_accepted(e, where + ".interpolation", interpolation, {"cubic", "linear"});
        }
        entry.set = data.add_table(coords, fields, interpolation == "cubic", where);
        const Set & s = data.set(entry.set);
        std::string shape;
        for (uint32_t d = 0; d < s.dims; d++) shape += (d ? " x " : "") + std::to_string(s.n_axis[d]);
        entry.kind = "table " + shape + ", " + interpolation + scaled_text;
    }
    return entry;
}

std::vector<std::string> entry_variables(const Input & e) {
    if (!e.is_table() || !e.contains("name") || !e.at("name").is_string() || !e.contains("file") ||
        !e.at("file").is_string()) {
        return {};
    }
    std::string metadata;
    const std::vector<std::string> columns = read_header(e.at("file").as_string(), &metadata);
    const bool points = e.contains("type") && e.at("type").is_string() && e.at("type").as_string() == "points";
    std::vector<std::string> variables;
    try {
        const Metadata m = parse_metadata(metadata, "", "");
        for (const std::string & c : entry_values(e, columns, entry_coordinates(e, columns, points, m.coordinates))) {
            variables.push_back(e.at("name").as_string() + "_" + identifier(c));
        }
    } catch (const std::exception &) {
        return {};
    }
    return variables;
}

uint32_t Data::add_table(const std::vector<std::vector<double>> & coordinates,
                         const std::vector<std::vector<double>> & fields, const bool cubic, const std::string & where) {
    check_columns(coordinates, fields, where);
    Set set;
    set.cubic = cubic ? 1 : 0;
    set.dims = static_cast<uint32_t>(coordinates.size());
    set.n_fields = static_cast<uint32_t>(fields.size());
    const size_t n_rows = coordinates[0].size();
    std::vector<std::vector<double>> axes(set.dims);
    size_t n_nodes = 1;
    for (uint32_t d = 0; d < set.dims; d++) {
        axes[d] = coordinates[d];
        std::sort(axes[d].begin(), axes[d].end());
        axes[d].erase(std::unique(axes[d].begin(), axes[d].end()), axes[d].end());
        set.axes[d] = static_cast<uint32_t>(m_axes.size());
        set.n_axis[d] = static_cast<uint32_t>(axes[d].size());
        m_axes.insert(m_axes.end(), axes[d].begin(), axes[d].end());
        n_nodes *= axes[d].size();
    }
    if (n_nodes != n_rows) {
        std::string shape;
        for (uint32_t d = 0; d < set.dims; d++) shape += (d ? " x " : "") + std::to_string(axes[d].size());
        throw InputError(where + ": " + std::to_string(n_rows) + " rows do not form a full grid of the distinct " +
                         "coordinate values (" + shape + " = " + std::to_string(n_nodes) + "), or a row repeats.");
    }
    set.values = static_cast<uint32_t>(m_values.size());
    m_values.resize(m_values.size() + n_nodes * set.n_fields, std::numeric_limits<double>::quiet_NaN());
    std::vector<bool> filled(n_nodes, false);
    for (size_t r = 0; r < n_rows; r++) {
        size_t node = 0;
        for (uint32_t d = 0; d < set.dims; d++) {
            const auto it = std::lower_bound(axes[d].begin(), axes[d].end(), coordinates[d][r]);
            node = node * axes[d].size() + static_cast<size_t>(it - axes[d].begin());
        }
        if (filled[node]) throw InputError(where + ": row " + std::to_string(r + 1) + " repeats a grid node.");
        filled[node] = true;
        for (uint32_t f = 0; f < set.n_fields; f++) m_values[set.values + node * set.n_fields + f] = fields[f][r];
    }
    m_sets.push_back(set);
    return static_cast<uint32_t>(m_sets.size() - 1);
}

uint32_t Data::add_points(const std::vector<std::vector<double>> & coordinates,
                          const std::vector<std::vector<double>> & fields, const uint32_t neighbors,
                          const std::string & where) {
    check_columns(coordinates, fields, where);
    if (coordinates.size() > N_DIM) {
        throw InputError(where + ": a point cloud has at most " + std::to_string(N_DIM) + " coordinates in " +
                         std::to_string(N_DIM) + "D.");
    }
    if (neighbors < 1 || neighbors > MAX_NEIGHBORS) {
        throw InputError(where + ".neighbors must be 1 to " + std::to_string(MAX_NEIGHBORS) + ".");
    }
    Set set;
    set.points = 1;
    set.dims = static_cast<uint32_t>(coordinates.size());
    set.n_fields = static_cast<uint32_t>(fields.size());
    const size_t n = coordinates[0].size();
    set.neighbors = static_cast<uint32_t>(std::min<size_t>(neighbors, n));
    std::vector<Box> boxes(n);
    set.coords = static_cast<uint32_t>(m_coords.size() / 3);
    for (size_t i = 0; i < n; i++) {
        double x[3] = {0.0, 0.0, 0.0};
        for (uint32_t d = 0; d < set.dims; d++) x[d] = coordinates[d][i];
        boxes[i].add(x);
        m_coords.insert(m_coords.end(), x, x + 3);
    }
    const BoxTree tree(boxes);
    set.nodes = static_cast<uint32_t>(m_nodes.size());
    for (const auto & node : tree.tree()) {
        TreeNode t;
        for (size_t d = 0; d < N_DIM; d++) {
            t.lo[d] = node.box.lo[d];
            t.hi[d] = node.box.hi[d];
        }
        t.first = node.first;
        t.count = node.count;
        m_nodes.push_back(t);
    }
    set.items = static_cast<uint32_t>(m_order.size());
    m_order.insert(m_order.end(), tree.item_order().begin(), tree.item_order().end());
    set.values = static_cast<uint32_t>(m_values.size());
    for (size_t i = 0; i < n; i++) {
        for (uint32_t f = 0; f < set.n_fields; f++) m_values.push_back(fields[f][i]);
    }
    m_sets.push_back(set);
    return static_cast<uint32_t>(m_sets.size() - 1);
}

DataView<Kokkos::HostSpace> Data::host() const {
    DataView<Kokkos::HostSpace> v;
    v.sets = host_view(m_sets);
    v.axes = host_view(m_axes);
    v.values = host_view(m_values);
    v.coords = host_view(m_coords);
    v.nodes = host_view(m_nodes);
    v.order = host_view(m_order);
    return v;
}

DataView<> Data::device() const {
    DataView<> v;
    v.sets = device_view(m_sets, "data_sets");
    v.axes = device_view(m_axes, "data_axes");
    v.values = device_view(m_values, "data_values");
    v.coords = device_view(m_coords, "data_coords");
    v.nodes = device_view(m_nodes, "data_nodes");
    v.order = device_view(m_order, "data_order");
    return v;
}

std::string column_of(const Entry & entry, const std::string & variable) {
    return variable.substr(entry.name.size() + 1);
}

InputError taken(const Input & e, const std::string & where, const std::string & docs, const Entry & entry,
                 const std::string & variable, const bool coordinate, const std::string & side) {
    const std::string what = "makes the variable " + variable + " (name " + entry.name + ", column " +
                             column_of(entry, variable) + "), which ";
    return InputError({.key = where + ".name",
                       .line = line_of(e, "name"),
                       .message = what + (coordinate ? "is the " + side + " coordinate " + variable
                                                     : "another data entry also makes"),
                       .fix = "Give the entry another name",
                       .docs = docs});
}

std::string one_of(const std::vector<std::string> & names) {
    std::string text;
    for (size_t i = 0; i < names.size(); i++) {
        text += (i == 0 ? "" : i + 1 == names.size() ? " or " : ", ") + names[i];
    }
    return text;
}

std::string at_text(const std::string & at, const std::vector<std::string> & names) {
    if (std::find(names.begin(), names.end(), at) != names.end()) return at;
    return "\"" + at + "\"";
}

} // namespace tabulated

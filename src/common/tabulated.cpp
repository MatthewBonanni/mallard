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
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <limits>

#include "box_tree.h"
#include "input.h"

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

} // namespace

int32_t Csv::column(const std::string & name) const {
    const auto it = std::find(names.begin(), names.end(), name);
    return it == names.end() ? -1 : static_cast<int32_t>(it - names.begin());
}

Csv read_csv(const std::string & file) {
    std::ifstream in(file);
    if (!in) throw InputError("Cannot open data file " + file + ".");
    Csv csv;
    std::string line;
    size_t line_number = 0;
    while (std::getline(in, line)) {
        line_number++;
        const std::string t = trim(line);
        if (t.empty() || t[0] == '#') continue;
        const std::vector<std::string> items = split(line);
        if (csv.names.empty()) {
            csv.names = items;
            for (const auto & name : csv.names) {
                if (name.empty()) throw InputError(file + ":" + std::to_string(line_number) + ": a column has no name.");
                if (std::count(csv.names.begin(), csv.names.end(), name) > 1) {
                    throw InputError(file + ":" + std::to_string(line_number) + ": column " + name + " appears twice.");
                }
            }
            csv.columns.resize(csv.names.size());
            continue;
        }
        if (items.size() != csv.names.size()) {
            throw InputError(file + ":" + std::to_string(line_number) + ": " + std::to_string(items.size()) +
                             " values for " + std::to_string(csv.names.size()) + " columns.");
        }
        for (size_t i = 0; i < items.size(); i++) {
            char * end = nullptr;
            const double v = std::strtod(items[i].c_str(), &end);
            if (items[i].empty() || end != items[i].c_str() + items[i].size()) {
                throw InputError(file + ":" + std::to_string(line_number) + ": \"" + items[i] + "\" is not a number.");
            }
            csv.columns[i].push_back(v);
        }
    }
    if (csv.names.empty() || csv.columns[0].empty()) throw InputError("Data file " + file + " has no rows.");
    return csv;
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

} // namespace tabulated

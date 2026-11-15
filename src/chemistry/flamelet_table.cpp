/**
 * @file flamelet_table.cpp
 * @author Matthew Bonanni (mbonanni001@gmail.com)
 * @brief Reading and checking tables of tabulated chemistry.
 * @version 0.1
 * @date 2026-11-15
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 *
 */

#include "flamelet_table.h"

#include <bit>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iterator>
#include <stdexcept>
#include <unordered_set>
#include <utility>

#include "common_io.h"
#include "json.h"

namespace flamelet {

namespace {

constexpr const char * MAGIC = "mallard-table 1";

[[noreturn]] void fail(const std::string & where, const std::string & what) {
    throw std::runtime_error(where + ": " + what + " (Mallard's table format: docs/design/flamelet.md#table-format)");
}

const json::Value & member(const json::Value & v, const std::string & key, const json::Value::Type type,
                           const std::string & where) {
    const json::Value * m = v.find(key);
    if (m == nullptr || m->type != type) fail(where, "the header has no valid \"" + key + "\"");
    return *m;
}

double swap_bytes(double x) {
    uint64_t u;
    std::memcpy(&u, &x, sizeof u);
    u = ((u & 0x00000000000000ffULL) << 56) | ((u & 0x000000000000ff00ULL) << 40) |
        ((u & 0x0000000000ff0000ULL) << 24) | ((u & 0x00000000ff000000ULL) << 8) |
        ((u & 0x000000ff00000000ULL) >> 8) | ((u & 0x0000ff0000000000ULL) >> 24) |
        ((u & 0x00ff000000000000ULL) >> 40) | ((u & 0xff00000000000000ULL) >> 56);
    std::memcpy(&x, &u, sizeof x);
    return x;
}

template <typename MemorySpace>
TableView<MemorySpace> make_view(const std::vector<Table::Axis> & axes, const uint32_t n_fields,
                                 Kokkos::View<const double *, MemorySpace> axis_values,
                                 Kokkos::View<const double *, MemorySpace> values) {
    TableView<MemorySpace> t;
    t.n_axes = static_cast<uint32_t>(axes.size());
    t.n_fields = n_fields;
    uint32_t first = 0;
    for (uint32_t a = 0; a < t.n_axes; a++) {
        t.n_nodes[a] = static_cast<uint32_t>(axes[a].nodes.size());
        t.axis_first[a] = first;
        first += t.n_nodes[a];
    }
    t.axes = axis_values;
    t.values = values;
    return t;
}

} // namespace

Table::Table(std::string model, std::vector<Axis> axes, std::vector<Field> fields, std::vector<double> values,
             const std::string & where)
    : m_model(std::move(model)), m_axes(std::move(axes)), m_fields(std::move(fields)), m_values(std::move(values)) {
    finish(where);
}

void Table::finish(const std::string & where) {
    if (m_axes.empty() || m_axes.size() > MAX_AXES) {
        fail(where, std::to_string(m_axes.size()) + " axes, where 1 to " + std::to_string(MAX_AXES) + " are supported");
    }
    if (m_fields.empty()) fail(where, "no fields");
    std::unordered_set<std::string> names;
    for (const auto & f : m_fields) {
        if (!names.insert(f.name).second) fail(where, "the field \"" + f.name + "\" appears twice");
    }
    uint64_t nodes = 1;
    for (const auto & a : m_axes) {
        if (a.nodes.size() < 2) fail(where, "the axis \"" + a.name + "\" has fewer than 2 nodes");
        for (size_t i = 0; i < a.nodes.size(); i++) {
            if (!std::isfinite(a.nodes[i]) || (i > 0 && !(a.nodes[i] > a.nodes[i - 1]))) {
                fail(where, "the nodes of the axis \"" + a.name + "\" are not finite and strictly increasing");
            }
        }
        nodes *= a.nodes.size();
        m_axis_nodes.insert(m_axis_nodes.end(), a.nodes.begin(), a.nodes.end());
    }
    if (m_values.size() != nodes * m_fields.size()) {
        fail(where, "it holds " + std::to_string(m_values.size()) + " values, where its axes and fields need " +
                        std::to_string(nodes * m_fields.size()));
    }
    for (size_t i = 0; i < m_values.size(); i++) {
        if (!std::isfinite(m_values[i])) {
            fail(where, "the value of field \"" + m_fields[i % m_fields.size()].name + "\" at node " +
                            std::to_string(i / m_fields.size()) + " is not finite");
        }
    }
}

Table Table::read(const std::string & file) {
    std::ifstream in(file, std::ios::binary);
    if (!in) throw cannot_read("table file", file);
    std::string magic, header;
    if (!std::getline(in, magic) || magic != MAGIC) {
        fail(file, "it does not start with \"" + std::string(MAGIC) + "\": not a Mallard table, or a newer version");
    }
    if (!std::getline(in, header)) fail(file, "it ends before its header");
    json::Value h;
    try {
        h = json::parse(header);
    } catch (const std::exception & e) {
        fail(file, std::string("its header is not valid JSON: ") + e.what());
    }
    if (h.type != json::Value::Type::OBJECT) fail(file, "its header is not a JSON object");
    if (member(h, "format", json::Value::Type::STRING, file).string != "mallard-table" ||
        member(h, "version", json::Value::Type::NUMBER, file).number != 1.0) {
        fail(file, "its header is not that of a Mallard table of version 1");
    }
    Table t;
    t.m_header = header;
    t.m_model = member(h, "model", json::Value::Type::STRING, file).string;
    for (const auto & a : member(h, "axes", json::Value::Type::ARRAY, file).array) {
        Axis axis;
        axis.name = member(a, "name", json::Value::Type::STRING, file).string;
        for (const auto & v : member(a, "values", json::Value::Type::ARRAY, file).array) {
            if (v.type != json::Value::Type::NUMBER) fail(file, "the axis \"" + axis.name + "\" has a node that is not a number");
            axis.nodes.push_back(v.number);
        }
        t.m_axes.push_back(std::move(axis));
    }
    for (const auto & f : member(h, "fields", json::Value::Type::ARRAY, file).array) {
        t.m_fields.push_back({member(f, "name", json::Value::Type::STRING, file).string,
                              member(f, "units", json::Value::Type::STRING, file).string});
    }
    const std::string bytes((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    if (bytes.size() % sizeof(double) != 0) fail(file, "its data is not a whole number of float64 values");
    t.m_values.resize(bytes.size() / sizeof(double));
    std::memcpy(t.m_values.data(), bytes.data(), bytes.size());
    if constexpr (std::endian::native == std::endian::big) {
        for (auto & v : t.m_values) v = swap_bytes(v);
    }
    t.finish(file);
    return t;
}

int32_t Table::field(const std::string & name) const {
    for (size_t i = 0; i < m_fields.size(); i++) {
        if (m_fields[i].name == name) return static_cast<int32_t>(i);
    }
    return -1;
}

TableView<Kokkos::HostSpace> Table::host() const {
    using Unmanaged = Kokkos::View<const double *, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
    return make_view<Kokkos::HostSpace>(m_axes, static_cast<uint32_t>(m_fields.size()),
                                        Unmanaged(m_axis_nodes.data(), m_axis_nodes.size()),
                                        Unmanaged(m_values.data(), m_values.size()));
}

TableView<> Table::device() const {
    using Space = Kokkos::DefaultExecutionSpace::memory_space;
    const TableView<Kokkos::HostSpace> h = host();
    Kokkos::View<double *, Space> axes("flamelet_axes", h.axes.size()), values("flamelet_values", h.values.size());
    Kokkos::deep_copy(axes, h.axes);
    Kokkos::deep_copy(values, h.values);
    return make_view<Space>(m_axes, h.n_fields, axes, values);
}

uint32_t Table::fpv_field(const std::string & where) const {
    if (m_model != "fpv") fail(where, "its model is \"" + m_model + "\", not \"fpv\"");
    if (m_axes.size() != 3 || m_axes[0].name != "Z" || m_axes[1].name != "S" || m_axes[2].name != "c") {
        fail(where, "an FPV table has the axes Z, S and c");
    }
    if (m_axes[1].nodes.front() != 0.0 || m_axes[1].nodes.back() != 1.0 || m_axes[2].nodes.front() != 0.0 ||
        m_axes[2].nodes.back() != 1.0) {
        fail(where, "the axes S and c of an FPV table run from 0 to 1");
    }
    const int32_t C = field("C");
    if (C < 0) fail(where, "an FPV table has the field C");
    return static_cast<uint32_t>(C);
}

FPV<> Table::fpv() const {
    const uint32_t C = fpv_field("table");
    return FPV<>{device(), C};
}

FPV<Kokkos::HostSpace> Table::fpv_host() const {
    const uint32_t C = fpv_field("table");
    return FPV<Kokkos::HostSpace>{host(), C};
}

} // namespace flamelet

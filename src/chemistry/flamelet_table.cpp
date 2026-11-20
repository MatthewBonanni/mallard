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

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <unordered_set>
#include <utility>

#include "common_io.h"
#include "json.h"

namespace flamelet {

namespace {

constexpr const char * MAGIC = "mallard-table 1";

[[noreturn]] void fail(const std::string & where, const std::string & what) {
    throw std::runtime_error(where + ": " + what + " (Mallard's table format: docs/design/table_format.md)");
}

const json::Value & member(const json::Value & v, const std::string & key, const json::Value::Type type,
                           const std::string & where, const std::string & path = "") {
    const json::Value * m = v.find(key);
    if (m == nullptr || m->type != type) fail(where, "the header has no valid \"" + path + key + "\"");
    return *m;
}

enum class Bound { ANY, NON_NEGATIVE, POSITIVE, ABOVE_ONE, FRACTION };

struct Need {
    const char * name;
    const char * units;
    Bound bound;
};

struct Contract {
    const char * tag;
    std::vector<const char *> axes;  // in order, each from 0 to 1, units "1"; empty: any
    std::vector<Need> fields;
};

const Contract & fpv_contract() {
    static const Contract c{"fpv",
                            {"Z", "S", "c"},
                            {{"C", "1", Bound::NON_NEGATIVE},
                             {"omega_C", "kg m-3 s-1", Bound::ANY},
                             {"rho0", "kg m-3", Bound::POSITIVE}}};
    return c;
}

const std::vector<const Contract *> & model_contracts() {
    static const std::vector<const Contract *> m = {&fpv_contract()};
    return m;
}

const std::vector<const Contract *> & thermo_contracts() {
    static const Contract saghafian{"saghafian2015",
                                    {},
                                    {{"T0", "K", Bound::POSITIVE},
                                     {"e0", "J kg-1", Bound::ANY},
                                     {"R", "J kg-1 K-1", Bound::POSITIVE},
                                     {"gamma0", "1", Bound::ABOVE_ONE},
                                     {"a_gamma", "K-1", Bound::ANY}}};
    static const std::vector<const Contract *> t = {&saghafian};
    return t;
}

const std::vector<Need> & transport() {
    static const std::vector<Need> t = {{"mu", "Pa s", Bound::POSITIVE},
                                        {"kappa", "W m-1 K-1", Bound::POSITIVE},
                                        {"cp", "J kg-1 K-1", Bound::POSITIVE}};
    return t;
}

const Contract * find_contract(const std::vector<const Contract *> & list, const std::string & tag) {
    for (const Contract * c : list) {
        if (tag == c->tag) return c;
    }
    return nullptr;
}

const Contract * supported(const std::vector<const Contract *> & list, const std::string & kind, const std::string & tag,
                           const std::string & where) {
    const Contract * c = find_contract(list, tag);
    if (c == nullptr) {
        std::string names;
        for (const Contract * s : list) names += (names.empty() ? "" : ", ") + std::string(s->tag);
        fail(where, "its " + kind + " \"" + tag + "\" is not supported by this version of Mallard (supported: " + names +
                        ")");
    }
    return c;
}

bool within(const double v, const Bound b) {
    switch (b) {
        case Bound::NON_NEGATIVE: return v >= 0.0;
        case Bound::POSITIVE: return v > 0.0;
        case Bound::ABOVE_ONE: return v > 1.0;
        case Bound::FRACTION: return v >= 0.0 && v <= 1.0;
        case Bound::ANY: break;
    }
    return true;
}

const char * bound_text(const Bound b) {
    switch (b) {
        case Bound::NON_NEGATIVE: return "negative";
        case Bound::POSITIVE: return "not positive";
        case Bound::ABOVE_ONE: return "not above 1";
        case Bound::FRACTION: return "outside [0, 1]";
        case Bound::ANY: break;
    }
    return "";
}

bool is_sha256(const std::string & s) {
    if (s.size() != 64) return false;
    for (const char ch : s) {
        if (!((ch >= '0' && ch <= '9') || (ch >= 'a' && ch <= 'f'))) return false;
    }
    return true;
}

bool is_name(const std::string & s) {
    auto letter = [](const char ch) { return (ch >= 'A' && ch <= 'Z') || (ch >= 'a' && ch <= 'z'); };
    if (s.empty() || !letter(s[0])) return false;
    for (const char ch : s) {
        if (!(letter(ch) || (ch >= '0' && ch <= '9') || ch == '_')) return false;
    }
    return true;
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

Table::Table(std::string model, std::string thermodynamics, std::vector<Axis> axes, std::vector<Field> fields,
             std::vector<double> values, const std::string & where)
    : m_where(where), m_model(std::move(model)), m_thermodynamics(std::move(thermodynamics)), m_axes(std::move(axes)),
      m_fields(std::move(fields)), m_values(std::move(values)) {
    finish();
}

void Table::finish() {
    const std::string & where = m_where;
    if (m_axes.empty() || m_axes.size() > MAX_AXES) {
        fail(where, std::to_string(m_axes.size()) + " axes, where this version of Mallard reads 1 to " +
                        std::to_string(MAX_AXES));
    }
    if (m_fields.empty()) fail(where, "no fields");
    std::unordered_set<std::string> names;
    for (const auto & f : m_fields) {
        if (!is_name(f.name)) fail(where, "the field name \"" + f.name + "\" is not of the form [A-Za-z][A-Za-z0-9_]*");
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

void Table::check_contract() const {
    const std::string & where = m_where;
    const Contract & model = *supported(model_contracts(), "model", m_model, where);
    const Contract & thermo = *supported(thermo_contracts(), "thermodynamics", m_thermodynamics, where);
    if (m_axes.size() != model.axes.size()) {
        fail(where, "a table of model \"" + m_model + "\" has " + std::to_string(model.axes.size()) + " axes");
    }
    for (size_t a = 0; a < model.axes.size(); a++) {
        const Axis & axis = m_axes[a];
        if (axis.name != model.axes[a] || axis.units != "1" || axis.nodes.front() != 0.0 || axis.nodes.back() != 1.0) {
            fail(where, "axis " + std::to_string(a) + " of a table of model \"" + m_model + "\" is \"" + model.axes[a] +
                            "\", in units \"1\", from 0 to 1");
        }
    }
    std::vector<Need> needs = model.fields;
    needs.insert(needs.end(), thermo.fields.begin(), thermo.fields.end());
    needs.insert(needs.end(), transport().begin(), transport().end());
    std::vector<std::string> y_names;
    for (const auto & s : m_species) y_names.push_back("Y_" + s);
    for (const auto & y : y_names) needs.push_back({y.c_str(), "1", Bound::FRACTION});
    const size_t n_fields = m_fields.size();
    for (const Need & need : needs) {
        const int32_t f = field(need.name);
        if (f < 0) fail(where, "it has no field \"" + std::string(need.name) + "\"");
        if (m_fields[static_cast<size_t>(f)].units != need.units) {
            fail(where, "the field \"" + std::string(need.name) + "\" is in \"" + m_fields[static_cast<size_t>(f)].units +
                            "\", where Mallard reads \"" + need.units + "\"");
        }
        for (size_t i = static_cast<size_t>(f); i < m_values.size(); i += n_fields) {
            if (!within(m_values[i], need.bound)) {
                fail(where, "the field \"" + std::string(need.name) + "\" is " + bound_text(need.bound) + " at node " +
                                std::to_string(i / n_fields));
            }
        }
    }
    if (&model == &fpv_contract()) {
        // C linear in c at each (Z, S), so that the FPV map returns the transported C exactly
        const size_t C = static_cast<size_t>(field("C"));
        const std::vector<double> & c = m_axes[2].nodes;
        const size_t n_c = c.size(), n_zs = m_axes[0].nodes.size() * m_axes[1].nodes.size();
        double scale = 0.0;
        for (size_t i = C; i < m_values.size(); i += n_fields) scale = std::max(scale, std::abs(m_values[i]));
        for (size_t zs = 0; zs < n_zs; zs++) {
            const double * row = m_values.data() + zs * n_c * n_fields + C;
            const double lo = row[0], hi = row[(n_c - 1) * n_fields];
            if (hi < lo) fail(where, "C decreases from c = 0 to c = 1 at (Z, S) node " + std::to_string(zs));
            for (size_t k = 0; k < n_c; k++) {
                if (std::abs(row[k * n_fields] - (lo + c[k] * (hi - lo))) > 1e-9 * scale) {
                    fail(where, "C is not linear in c at (Z, S) node " + std::to_string(zs));
                }
            }
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
    using T = json::Value::Type;
    if (member(h, "format", T::STRING, file).string != "mallard-table" ||
        member(h, "version", T::NUMBER, file).number != 1.0) {
        fail(file, "its header is not that of a Mallard table of version 1");
    }
    Table t;
    t.m_where = file;
    t.m_header = header;
    t.m_model = member(h, "model", T::STRING, file).string;
    t.m_thermodynamics = member(h, "thermodynamics", T::STRING, file).string;
    supported(model_contracts(), "model", t.m_model, file);
    supported(thermo_contracts(), "thermodynamics", t.m_thermodynamics, file);
    for (const auto & a : member(h, "axes", T::ARRAY, file).array) {
        Axis axis;
        axis.name = member(a, "name", T::STRING, file, "axes[].").string;
        axis.units = member(a, "units", T::STRING, file, "axes[].").string;
        for (const auto & v : member(a, "values", T::ARRAY, file, "axes[].").array) {
            if (v.type != T::NUMBER) fail(file, "the axis \"" + axis.name + "\" has a node that is not a number");
            axis.nodes.push_back(v.number);
        }
        t.m_axes.push_back(std::move(axis));
    }
    for (const auto & f : member(h, "fields", T::ARRAY, file).array) {
        t.m_fields.push_back(
            {member(f, "name", T::STRING, file, "fields[].").string, member(f, "units", T::STRING, file, "fields[].").string});
    }
    for (const auto & s : member(h, "species", T::ARRAY, file).array) {
        if (s.type != T::STRING) fail(file, "the header has no valid \"species\"");
        t.m_species.push_back(s.string);
    }
    for (const auto & f : t.m_fields) {
        if (f.name.starts_with("Y_") &&
            std::find(t.m_species.begin(), t.m_species.end(), f.name.substr(2)) == t.m_species.end()) {
            fail(file, "the field \"" + f.name + "\" is not of a species in \"species\"");
        }
    }
    t.m_pressure = member(h, "pressure", T::NUMBER, file).number;
    if (!(t.m_pressure > 0.0)) fail(file, "the header has no valid \"pressure\"");
    const json::Value model_data = member(h, "model_data", T::OBJECT, file);
    if (t.m_model == "fpv") {
        const double Z_st = member(model_data, "Z_st", T::NUMBER, file, "model_data.").number;
        if (!(Z_st > 0.0 && Z_st < 1.0)) fail(file, "the header has no valid \"model_data.Z_st\"");
        const json::Value weights = member(model_data, "progress_variable", T::OBJECT, file, "model_data.");
        if (weights.members.empty()) fail(file, "the header has no valid \"model_data.progress_variable\"");
        for (size_t k = 0; k < weights.members.size(); k++) {
            if (weights.members[k].type != T::NUMBER) {
                fail(file, "the weight of " + weights.names[k] + " in \"model_data.progress_variable\" is not a number");
            }
        }
        const json::Value streams = member(model_data, "streams", T::OBJECT, file, "model_data.");
        for (const char * s : {"fuel", "oxidizer"}) {
            const std::string path = "model_data.streams." + std::string(s) + ".";
            const json::Value stream = member(streams, s, T::OBJECT, file, "model_data.streams.");
            if (!(member(stream, "T", T::NUMBER, file, path).number > 0.0)) {
                fail(file, "the header has no valid \"" + path + "T\"");
            }
            const json::Value Y = member(stream, "Y", T::OBJECT, file, path);
            for (size_t k = 0; k < Y.members.size(); k++) {
                if (Y.members[k].type != T::NUMBER || !within(Y.members[k].number, Bound::FRACTION)) {
                    fail(file, "the mass fraction of " + Y.names[k] + " in \"" + path + "Y\" is not in [0, 1]");
                }
            }
        }
    }
    const json::Value provenance = member(h, "provenance", T::OBJECT, file);
    const json::Value generator = member(provenance, "generator", T::OBJECT, file, "provenance.");
    t.m_generator = member(generator, "name", T::STRING, file, "provenance.generator.").string + " " +
                    member(generator, "version", T::STRING, file, "provenance.generator.").string;
    member(provenance, "created", T::STRING, file, "provenance.");
    member(provenance, "input", T::STRING, file, "provenance.");
    const json::Value mechanism = member(provenance, "mechanism", T::OBJECT, file, "provenance.");
    member(mechanism, "name", T::STRING, file, "provenance.mechanism.");
    if (!is_sha256(member(mechanism, "sha256", T::STRING, file, "provenance.mechanism.").string)) {
        fail(file, "the header has no valid \"provenance.mechanism.sha256\" (64 lowercase hexadecimal digits)");
    }
    const std::streamoff start = in.tellg();
    in.seekg(0, std::ios::end);
    const std::streamoff size = in.tellg() - start;
    in.seekg(start);
    if (start < 0 || size < 0 || !in) fail(file, "its data cannot be read");
    if (size % static_cast<std::streamoff>(sizeof(double)) != 0) fail(file, "its data is not a whole number of float64 values");
    t.m_values.resize(static_cast<size_t>(size) / sizeof(double));
    if (!in.read(reinterpret_cast<char *>(t.m_values.data()), size)) fail(file, "its data cannot be read");
    if constexpr (std::endian::native == std::endian::big) {
        for (auto & v : t.m_values) v = swap_bytes(v);
    }
    t.finish();
    t.check_contract();
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

uint32_t Table::fpv_field() const {
    check_contract();
    if (m_model != "fpv") fail(m_where, "its model is \"" + m_model + "\", not \"fpv\"");
    return static_cast<uint32_t>(field("C"));
}

FPV<> Table::fpv() const {
    const uint32_t C = fpv_field();
    return FPV<>{device(), C};
}

FPV<Kokkos::HostSpace> Table::fpv_host() const {
    const uint32_t C = fpv_field();
    return FPV<Kokkos::HostSpace>{host(), C};
}

} // namespace flamelet

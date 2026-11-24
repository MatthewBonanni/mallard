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
#include <cmath>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <unordered_set>
#include <utility>

#include "common_io.h"
#include "input.h"

#ifdef Mallard_HAS_HDF5
#include "hdf5_util.h"

// GPU builds map the values: they only stage them for the copy to the device. Host builds read them
// into memory, where lookups run on them: lookups on page-cache pages, which get no huge pages, ran
// about 20% slower (docs/design/table_format.md, section 2)
#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP) || defined(KOKKOS_ENABLE_SYCL)
#define FLAMELET_MAP_VALUES
#include <cstring>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif
#endif

namespace flamelet {

namespace {

[[noreturn]] void fail(const std::string & where, const std::string & what) {
    throw std::runtime_error(where + ": " + what + " (Mallard's table format: docs/design/table_format.md)");
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
                            {"Z", "Sz", "Lambda"},
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

#ifdef Mallard_HAS_HDF5
bool is_sha256(const std::string & s) {
    if (s.size() != 64) return false;
    for (const char ch : s) {
        if (!((ch >= '0' && ch <= '9') || (ch >= 'a' && ch <= 'f'))) return false;
    }
    return true;
}
#endif

bool is_name(const std::string & s) {
    auto letter = [](const char ch) { return (ch >= 'A' && ch <= 'Z') || (ch >= 'a' && ch <= 'z'); };
    if (s.empty() || !letter(s[0])) return false;
    for (const char ch : s) {
        if (!(letter(ch) || (ch >= '0' && ch <= '9') || ch == '_')) return false;
    }
    return true;
}

} // namespace

Table::Table(std::string model, std::string thermodynamics, std::vector<Axis> axes, std::vector<Field> fields,
             std::vector<double> values, const std::string & where)
    : m_where(where), m_model(std::move(model)), m_thermodynamics(std::move(thermodynamics)), m_axes(std::move(axes)),
      m_fields(std::move(fields)) {
    auto v = std::make_shared<std::vector<double>>(std::move(values));
    m_values = v->data();
    m_n_values = v->size();
    m_storage = std::move(v);
    finish();
}

void Table::finish() {
    const std::string & where = m_where;
    if (m_axes.empty()) fail(where, "no axes");
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
    if (m_n_values != nodes * m_fields.size()) {
        fail(where, "it holds " + std::to_string(m_n_values) + " values, where its axes and fields need " +
                        std::to_string(nodes * m_fields.size()));
    }
    with_values([&](const auto * v) {
        for (uint64_t i = 0; i < m_n_values; i++) {
            if (!std::isfinite(v[i])) {
                fail(where, "the value of field \"" + m_fields[i % m_fields.size()].name + "\" at node " +
                                std::to_string(i / m_fields.size()) + " is not finite");
            }
        }
    });
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
    // Optional fields, checked where present
    for (const Need & need : {Need{"HRR", "W m-3", Bound::ANY}}) {
        if (field(need.name) >= 0) needs.push_back(need);
    }
    const size_t n_fields = m_fields.size();
    for (const Need & need : needs) {
        const int32_t f = field(need.name);
        if (f < 0) fail(where, "it has no field \"" + std::string(need.name) + "\"");
        if (m_fields[static_cast<size_t>(f)].units != need.units) {
            fail(where, "the field \"" + std::string(need.name) + "\" is in \"" + m_fields[static_cast<size_t>(f)].units +
                            "\", where Mallard reads \"" + need.units + "\"");
        }
        with_values([&](const auto * v) {
            for (uint64_t i = static_cast<uint64_t>(f); i < m_n_values; i += n_fields) {
                if (!within(static_cast<double>(v[i]), need.bound)) {
                    fail(where, "the field \"" + std::string(need.name) + "\" is " + bound_text(need.bound) +
                                    " at node " + std::to_string(i / n_fields));
                }
            }
        });
    }
    if (&model == &fpv_contract()) {
        // C linear in Lambda at each (Z, Sz), so that the FPV map returns the transported C exactly; float32
        // tables to their rounding
        const size_t C = static_cast<size_t>(field("C"));
        const std::vector<double> & Lambda = m_axes[2].nodes;
        const size_t n_Lambda = Lambda.size(), n_zs = m_axes[0].nodes.size() * m_axes[1].nodes.size();
        const double tolerance = m_single ? 1e-6 : 1e-9;
        with_values([&](const auto * v) {
            double scale = 0.0;
            for (uint64_t i = C; i < m_n_values; i += n_fields) scale = std::max(scale, std::abs(static_cast<double>(v[i])));
            for (size_t zs = 0; zs < n_zs; zs++) {
                const auto * row = v + zs * n_Lambda * n_fields + C;
                const double lo = static_cast<double>(row[0]), hi = static_cast<double>(row[(n_Lambda - 1) * n_fields]);
                if (hi < lo) fail(where, "C decreases from Lambda = 0 to Lambda = 1 at (Z, Sz) node " + std::to_string(zs));
                for (size_t k = 0; k < n_Lambda; k++) {
                    if (std::abs(static_cast<double>(row[k * n_fields]) - (lo + Lambda[k] * (hi - lo))) > tolerance * scale) {
                        fail(where, "C is not linear in Lambda at (Z, Sz) node " + std::to_string(zs));
                    }
                }
            }
        });
    }
}

#ifdef Mallard_HAS_HDF5

namespace {

using h5::Handle;

/** HDF5's own error printing off while the reader probes for objects and reports what is missing itself. */
class Quiet {
    public:
        Quiet() {
            H5Eget_auto2(H5E_DEFAULT, &func, &data);
            H5Eset_auto2(H5E_DEFAULT, nullptr, nullptr);
        }
        ~Quiet() { H5Eset_auto2(H5E_DEFAULT, func, data); }
        Quiet(const Quiet &) = delete;
        Quiet & operator=(const Quiet &) = delete;

    private:
        H5E_auto2_t func = nullptr;
        void * data = nullptr;
};

struct Reader {
    const std::string & file;

    [[noreturn]] void missing(const std::string & what) const { fail(file, "it has no valid " + what); }
    [[noreturn]] void missing(const std::string & name, const std::string & on) const {
        missing("attribute \"" + name + "\" on " + on);
    }

    Handle group(const hid_t parent, const std::string & path, const std::string & full) const {
        const hid_t id = H5Gopen2(parent, path.c_str(), H5P_DEFAULT);
        if (id < 0) missing("group " + full);
        return Handle(id, H5Gclose, full);
    }

    Handle dataset(const hid_t parent, const std::string & path, const std::string & full) const {
        const hid_t id = H5Dopen2(parent, path.c_str(), H5P_DEFAULT);
        if (id < 0) missing("dataset " + full);
        return Handle(id, H5Dclose, full);
    }

    Handle attribute(const hid_t object, const std::string & name, const std::string & on) const {
        if (H5Aexists(object, name.c_str()) <= 0) missing(name, on);
        return Handle(H5Aopen(object, name.c_str(), H5P_DEFAULT), H5Aclose, name);
    }

    /** A number: a scalar integer or floating-point attribute. */
    double number(const hid_t object, const std::string & name, const std::string & on) const {
        const Handle a = attribute(object, name, on);
        const Handle type(H5Aget_type(a), H5Tclose, "H5Aget_type");
        const Handle space(H5Aget_space(a), H5Sclose, "H5Aget_space");
        const H5T_class_t c = H5Tget_class(type);
        if ((c != H5T_INTEGER && c != H5T_FLOAT) || H5Sget_simple_extent_npoints(space) != 1) missing(name, on);
        double v = 0.0;
        if (H5Aread(a, H5T_NATIVE_DOUBLE, &v) < 0) missing(name, on);
        return v;
    }

    /** The strings of an attribute or dataset of strings, variable or fixed length (h5py writes the first, C often the second). */
    std::vector<std::string> strings(const hid_t object, const bool is_attribute, const std::string & what) const {
        const Handle type(is_attribute ? H5Aget_type(object) : H5Dget_type(object), H5Tclose, "H5Aget_type");
        const Handle space(is_attribute ? H5Aget_space(object) : H5Dget_space(object), H5Sclose, "H5Aget_space");
        if (H5Tget_class(type) != H5T_STRING || H5Sget_simple_extent_ndims(space) > 1) missing(what);
        const size_t n = static_cast<size_t>(H5Sget_simple_extent_npoints(space));
        auto read = [&](const hid_t memory, void * buffer) {
            const herr_t status = is_attribute ? H5Aread(object, memory, buffer)
                                               : H5Dread(object, memory, H5S_ALL, H5S_ALL, H5P_DEFAULT, buffer);
            if (status < 0) missing(what);
        };
        std::vector<std::string> out;
        if (n == 0) return out;
        if (H5Tis_variable_str(type) > 0) {
            const Handle memory(H5Tcopy(H5T_C_S1), H5Tclose, "H5Tcopy");
            h5::check(H5Tset_size(memory, H5T_VARIABLE), "H5Tset_size");
            h5::check(H5Tset_cset(memory, H5Tget_cset(type)), "H5Tset_cset");
            std::vector<char *> buffer(n, nullptr);
            read(memory, buffer.data());
            for (const char * p : buffer) out.emplace_back(p ? p : "");
#if H5_VERSION_GE(1, 12, 0)
            H5Treclaim(memory, space, H5P_DEFAULT, buffer.data());
#else
            H5Dvlen_reclaim(memory, space, H5P_DEFAULT, buffer.data());
#endif
        } else {
            const size_t width = H5Tget_size(type);
            const Handle memory(H5Tcopy(type), H5Tclose, "H5Tcopy");
            std::vector<char> buffer(n * width, '\0');
            read(memory, buffer.data());
            const bool spaces = H5Tget_strpad(type) == H5T_STR_SPACEPAD;
            for (size_t i = 0; i < n; i++) {
                const char * p = buffer.data() + i * width;
                std::string s(p, std::find(p, p + width, '\0'));
                while (spaces && !s.empty() && s.back() == ' ') s.pop_back();
                out.push_back(std::move(s));
            }
        }
        return out;
    }

    std::vector<std::string> texts(const hid_t object, const std::string & name, const std::string & on) const {
        const Handle a = attribute(object, name, on);
        return strings(a, true, "attribute \"" + name + "\" on " + on);
    }

    std::string text(const hid_t object, const std::string & name, const std::string & on) const {
        const std::vector<std::string> s = texts(object, name, on);
        if (s.size() != 1) missing(name, on);
        return s[0];
    }

    /** The names of an object's attributes, in name order. */
    static std::vector<std::string> attribute_names(const hid_t object) {
        std::vector<std::string> names;
        auto add = [](hid_t, const char * name, const H5A_info_t *, void * out) -> herr_t {
            static_cast<std::vector<std::string> *>(out)->emplace_back(name);
            return 0;
        };
        H5Aiterate2(object, H5_INDEX_NAME, H5_ITER_INC, nullptr, add, &names);
        return names;
    }
};

#ifdef FLAMELET_MAP_VALUES
/**
 * The values mapped from the file, with data pointing at the first; null when the dataset's layout
 * does not allow it. The first and last values must read the same through HDF5, which guards the offset.
 */
std::shared_ptr<const void> map_values(const std::string & file, const hid_t dset, const hid_t file_type,
                                       const hid_t native, const size_t size, const uint64_t n, const void *& data) {
    const Handle dcpl(H5Dget_create_plist(dset), H5Pclose, "H5Dget_create_plist");
    if (H5Pget_layout(dcpl) != H5D_CONTIGUOUS || H5Pget_nfilters(dcpl) != 0 || H5Tequal(file_type, native) <= 0) {
        return nullptr;
    }
    const haddr_t offset = H5Dget_offset(dset);
    const uint64_t bytes = n * size;
    if (offset == HADDR_UNDEF || offset % size != 0) return nullptr;
    const int fd = open(file.c_str(), O_RDONLY);
    if (fd < 0) return nullptr;
    struct stat st {};
    if (fstat(fd, &st) != 0 || static_cast<uint64_t>(st.st_size) < offset + bytes) {
        close(fd);
        return nullptr;
    }
    const uint64_t page = static_cast<uint64_t>(sysconf(_SC_PAGESIZE));
    const uint64_t start = offset / page * page, length = offset - start + bytes;
    void * p = mmap(nullptr, static_cast<size_t>(length), PROT_READ, MAP_PRIVATE, fd, static_cast<off_t>(start));
    close(fd);
    if (p == MAP_FAILED) return nullptr;
    std::shared_ptr<const void> mapping(p, [length](void * q) { munmap(q, static_cast<size_t>(length)); });
    const char * first = static_cast<const char *>(p) + (offset - start);

    const Handle space(H5Dget_space(dset), H5Sclose, "H5Dget_space");
    const int rank = H5Sget_simple_extent_ndims(space);
    std::vector<hsize_t> dims(static_cast<size_t>(rank)), points(2 * dims.size(), 0);
    H5Sget_simple_extent_dims(space, dims.data(), nullptr);
    for (size_t a = 0; a < dims.size(); a++) points[dims.size() + a] = dims[a] - 1;
    h5::check(H5Sselect_elements(space, H5S_SELECT_SET, 2, points.data()), "H5Sselect_elements");
    const hsize_t two = 2;
    const Handle memory(H5Screate_simple(1, &two, nullptr), H5Sclose, "H5Screate_simple");
    char ends[16] = {};
    if (H5Dread(dset, native, memory, space, H5P_DEFAULT, ends) < 0 || std::memcmp(ends, first, size) != 0 ||
        std::memcmp(ends + size, first + bytes - size, size) != 0) {
        return nullptr;
    }
    data = first;
    return mapping;
}
#endif

} // namespace

#endif // Mallard_HAS_HDF5

Table Table::read(const std::string & file) {
#ifndef Mallard_HAS_HDF5
    throw InputError(file + ": tabulated chemistry reads tables from HDF5 files, and this build has no HDF5. Rebuild "
                            "with -DMallard_ENABLE_HDF5=ON (add -DUSE_SYSTEM_HDF5=OFF to build the bundled HDF5). See "
                            "docs/design/table_format.md.");
#else
    if (!std::ifstream(file, std::ios::binary)) throw cannot_read("table file", file);
    Table t;
    t.m_where = file;
    {
        const std::lock_guard<std::mutex> lock(h5::mutex());
        const Quiet quiet;
        const hid_t id = H5Fopen(file.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
        if (id < 0) fail(file, "it is not an HDF5 file, or it is damaged");
        const Handle h5(id, H5Fclose, "opening " + file);
        const Reader r{file};
        if (H5Aexists(h5, "format") <= 0 || r.text(h5, "format", "/") != "mallard-table") {
            fail(file, "it is not a Mallard table: its attribute \"format\" on / is not \"mallard-table\"");
        }
        const double version = r.number(h5, "version", "/");
        if (version != 1.0) {
            std::ostringstream v;
            v << version;
            fail(file, "it is a Mallard table of version " + v.str() +
                           ", which this version of Mallard does not read (it reads version 1)");
        }
        t.m_model = r.text(h5, "model", "/");
        t.m_thermodynamics = r.text(h5, "thermodynamics", "/");
        supported(model_contracts(), "model", t.m_model, file);
        supported(thermo_contracts(), "thermodynamics", t.m_thermodynamics, file);

        const Handle values = r.dataset(h5, "values", "/values");
        for (const std::string & name : r.texts(values, "axes", "/values")) {
            if (!is_name(name)) fail(file, "the axis name \"" + name + "\" is not of the form [A-Za-z][A-Za-z0-9_]*");
            Axis axis;
            axis.name = name;
            const std::string path = "/axes/" + name;
            const Handle nodes = r.dataset(h5, "axes/" + name, path);
            const Handle space(H5Dget_space(nodes), H5Sclose, "H5Dget_space");
            const Handle type(H5Dget_type(nodes), H5Tclose, "H5Dget_type");
            if (H5Tget_class(type) != H5T_FLOAT || H5Sget_simple_extent_ndims(space) != 1) {
                fail(file, "its dataset " + path + " is not one-dimensional floating point");
            }
            axis.nodes.resize(static_cast<size_t>(H5Sget_simple_extent_npoints(space)));
            if (!axis.nodes.empty() &&
                H5Dread(nodes, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, axis.nodes.data()) < 0) {
                fail(file, "its dataset " + path + " cannot be read");
            }
            axis.units = r.text(nodes, "units", path);
            t.m_axes.push_back(std::move(axis));
        }
        const std::vector<std::string> names = r.texts(values, "fields", "/values");
        const std::vector<std::string> units = r.texts(values, "units", "/values");
        if (names.size() != units.size()) {
            fail(file, "it has " + std::to_string(names.size()) + " field names and " + std::to_string(units.size()) +
                           " units (attributes \"fields\" and \"units\" on /values)");
        }
        for (size_t i = 0; i < names.size(); i++) t.m_fields.push_back({names[i], units[i]});
        t.m_species = r.texts(h5, "species", "/");
        for (const auto & f : t.m_fields) {
            if (f.name.starts_with("Y_") &&
                std::find(t.m_species.begin(), t.m_species.end(), f.name.substr(2)) == t.m_species.end()) {
                fail(file, "the field \"" + f.name + "\" is not of a species in the attribute \"species\" on /");
            }
        }
        t.m_pressure = r.number(h5, "pressure", "/");
        if (!(t.m_pressure > 0.0)) r.missing("pressure", "/");

        const Handle model_data = r.group(h5, "model_data", "/model_data");
        if (t.m_model == "fpv") {
            const double Z_st = r.number(model_data, "Z_st", "/model_data");
            if (!(Z_st > 0.0 && Z_st < 1.0)) r.missing("Z_st", "/model_data");
            const Handle weights = r.group(model_data, "progress_variable", "/model_data/progress_variable");
            const std::vector<std::string> species = Reader::attribute_names(weights);
            if (species.empty()) fail(file, "its group /model_data/progress_variable has no weights");
            for (const auto & s : species) {
                t.m_definitions.progress_variable.emplace_back(s, r.number(weights, s, "/model_data/progress_variable"));
            }
            t.m_definitions.mixture_fraction = r.text(model_data, "mixture_fraction", "/model_data");
            if (t.m_definitions.mixture_fraction != "bilger") {
                fail(file, "its attribute \"mixture_fraction\" on /model_data is \"" +
                               t.m_definitions.mixture_fraction + "\", where Mallard reads \"bilger\"");
            }
            for (const char * s : {"fuel", "oxidizer"}) {
                const std::string path = "/model_data/streams/" + std::string(s);
                const Handle stream = r.group(model_data, "streams/" + std::string(s), path);
                if (!(r.number(stream, "T", path) > 0.0)) r.missing("T", path);
                const Handle Y = r.group(stream, "Y", path + "/Y");
                auto & composition = std::string(s) == "fuel" ? t.m_definitions.fuel : t.m_definitions.oxidizer;
                double sum = 0.0;
                for (const auto & k : Reader::attribute_names(Y)) {
                    const double y = r.number(Y, k, path + "/Y");
                    if (!within(y, Bound::FRACTION)) {
                        fail(file, "the mass fraction of " + k + " in " + path + "/Y is not in [0, 1]");
                    }
                    composition.emplace_back(k, y);
                    sum += y;
                }
                if (!(std::abs(sum - 1.0) <= 1e-6)) {
                    std::ostringstream o;
                    o << sum;
                    fail(file, "the mass fractions of " + path + "/Y sum to " + o.str() + ", not 1");
                }
            }
        }
        const Handle provenance = r.group(h5, "provenance", "/provenance");
        const Handle generator = r.group(provenance, "generator", "/provenance/generator");
        t.m_generator = r.text(generator, "name", "/provenance/generator") + " " +
                        r.text(generator, "version", "/provenance/generator");
        r.text(provenance, "created", "/provenance");
        const Handle input = r.dataset(provenance, "input", "/provenance/input");
        if (r.strings(input, false, "dataset /provenance/input").size() != 1) r.missing("dataset /provenance/input");
        const Handle mechanism = r.group(provenance, "mechanism", "/provenance/mechanism");
        const std::string sha = r.text(mechanism, "sha256", "/provenance/mechanism");
        if (!is_sha256(sha)) {
            fail(file, "its attribute \"sha256\" on /provenance/mechanism is not 64 lowercase hexadecimal digits");
        }
        t.m_mechanism = r.text(mechanism, "name", "/provenance/mechanism") + " sha256:" + sha;

        const Handle space(H5Dget_space(values), H5Sclose, "H5Dget_space");
        std::vector<hsize_t> dims(static_cast<size_t>(std::max(H5Sget_simple_extent_ndims(space), 0)));
        H5Sget_simple_extent_dims(space, dims.data(), nullptr);
        std::vector<hsize_t> need;
        for (const Axis & a : t.m_axes) need.push_back(a.nodes.size());
        need.push_back(t.m_fields.size());
        if (dims != need) {
            auto shape = [](const std::vector<hsize_t> & d) {
                std::string s;
                for (const hsize_t n : d) s += (s.empty() ? "" : ", ") + std::to_string(n);
                return "(" + s + ")";
            };
            fail(file, "its values have shape " + shape(dims) + ", where its axes and fields need " + shape(need));
        }
        const Handle type(H5Dget_type(values), H5Tclose, "H5Dget_type");
        const size_t size = H5Tget_size(type);
        if (H5Tget_class(type) != H5T_FLOAT || (size != 4 && size != 8)) fail(file, "its values are not float64 or float32");
        t.m_single = size == 4;
        t.m_n_values = 1;
        for (const hsize_t n : dims) t.m_n_values *= n;
        const hid_t native = t.m_single ? H5T_NATIVE_FLOAT : H5T_NATIVE_DOUBLE;
#ifdef FLAMELET_MAP_VALUES
        t.m_storage = map_values(file, values, type, native, size, t.m_n_values, t.m_values);
#endif
        t.m_mapped = t.m_storage != nullptr;
        if (!t.m_mapped) {
            auto read = [&](auto storage) {
                if (H5Dread(values, native, H5S_ALL, H5S_ALL, H5P_DEFAULT, storage->data()) < 0) {
                    fail(file, "its values cannot be read");
                }
                t.m_values = storage->data();
                t.m_storage = std::move(storage);
            };
            if (t.m_single) read(std::make_shared<std::vector<float>>(t.m_n_values));
            else read(std::make_shared<std::vector<double>>(t.m_n_values));
        }
    }
    t.finish();
    t.check_contract();
    if (t.m_model == "fpv") t.check_definitions();
    return t;
#endif
}

void Table::check_definitions() const {
    const std::string & where = m_where;
    const Definitions & d = m_definitions;
    std::vector<uint32_t> fields;
    std::vector<double> weights;
    for (const auto & [species, w] : d.progress_variable) {
        const int32_t f = field("Y_" + species);
        if (f < 0) {
            fail(where, "its progress variable weighs " + species +
                            ", which is not in its species block: C = sum_k w_k Y_k needs every Y_k it weighs");
        }
        fields.push_back(static_cast<uint32_t>(f));
        weights.push_back(w);
    }
    // The stored C is the weighted sum of the stored mass fractions, to the rounding of the values
    const size_t n_fields = m_fields.size(), C = static_cast<size_t>(field("C"));
    const double tolerance = m_single ? 1e-6 : 1e-9;
    with_values([&](const auto * v) {
        double scale = 0.0;
        for (uint64_t i = C; i < m_n_values; i += n_fields) scale = std::max(scale, std::abs(static_cast<double>(v[i])));
        for (uint64_t node = 0; node < m_n_values / n_fields; node++) {
            const auto * row = v + node * n_fields;
            double sum = 0.0, magnitude = 0.0;
            for (size_t i = 0; i < fields.size(); i++) {
                sum += weights[i] * static_cast<double>(row[fields[i]]);
                magnitude += std::abs(weights[i] * static_cast<double>(row[fields[i]]));
            }
            if (std::abs(static_cast<double>(row[C]) - sum) > tolerance * std::max(scale, magnitude)) {
                std::ostringstream o;
                o << "its field C (" << static_cast<double>(row[C]) << ") is not the weighted sum of its mass "
                  << "fractions sum_k w_k Y_k (" << sum << ") of /model_data/progress_variable at node " << node;
                fail(where, o.str());
            }
        }
    });
}

int32_t Table::field(const std::string & name) const {
    for (size_t i = 0; i < m_fields.size(); i++) {
        if (m_fields[i].name == name) return static_cast<int32_t>(i);
    }
    return -1;
}

void Table::require_axes(const uint32_t d) const {
    if (m_axes.size() != d) {
        fail(m_where, "it has " + std::to_string(m_axes.size()) + " axes, where this lookup reads " + std::to_string(d));
    }
}

void Table::require_value(const bool single) const {
    if (single != m_single) {
        fail(m_where, std::string("its values are ") + (m_single ? "float32" : "float64") + ", where this lookup reads " +
                          (single ? "float32" : "float64"));
    }
}

uint32_t Table::fpv_field() const {
    check_contract();
    if (m_model != "fpv") fail(m_where, "its model is \"" + m_model + "\", not \"fpv\"");
    return static_cast<uint32_t>(field("C"));
}

} // namespace flamelet

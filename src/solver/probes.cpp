/**
 * @file probes.cpp
 * @author Matthew Bonanni (mbonanni001@gmail.com)
 * @brief [[probes]]: samples of the solution on points and curves.
 * @version 0.2
 * @date 2026-10-15
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 *
 */

#include "probes.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <exception>
#include <filesystem>
#include <iomanip>
#include <limits>
#include <mutex>
#include <regex>
#include <sstream>
#include <unordered_map>

#include "comm.h"
#include "common_io.h"
#include "field_units.h"
#include "file_format.h"
#include "input.h"
#include "log.h"
#include "provenance.h"
#include "region.h"
#include "viscous_flux.h"

#ifdef Mallard_HAS_HDF5
#include "hdf5_util.h"
#endif

namespace {

using Reduction = Probes::Reduction;

template <typename T>
Kokkos::View<T *> to_device(const std::vector<T> & v, const char * label) {
    Kokkos::View<T *> d(label, v.size());
    Kokkos::deep_copy(d, Kokkos::View<const T *, Kokkos::HostSpace>(v.data(), v.size()));
    return d;
}

void make_parent(const std::string & file) { create_parent_directories("the probe file", file); }

/** @brief A CSV file with a header of columns and their units, or the end of an existing one (resume). */
std::shared_ptr<std::ofstream> open_csv(const std::string & file, bool resume,
                                        const std::vector<std::pair<std::string, Dimension>> & columns) {
    if (!output_files_enabled()) return std::make_shared<std::ofstream>();
    make_parent(file);
    auto out = std::make_shared<std::ofstream>(file, resume ? std::ios::app : std::ios::trunc);
    if (!out->good()) throw cannot_write("the probe file", file);
    if (!resume) {
        std::vector<Dimension> units;
        for (size_t i = 0; i < columns.size(); i++) {
            *out << (i ? "," : "") << columns[i].first;
            units.push_back(columns[i].second);
        }
        *out << "\n" << field_units::csv_line(units) << "\n";
    }
    return out;
}

/**
 * @brief Runs f, the file work of rank 0, so that every rank throws if it
 *        throws and none waits in a later collective. Ranks that each run
 *        alone (not distributed) do not communicate.
 */
template <typename F>
void on_root(bool distributed, F && f) {
    std::exception_ptr error;
    if (comm::is_root()) {
        try {
            f();
        } catch (...) {
            error = std::current_exception();
        }
    }
    const uint32_t failed = distributed ? comm::allreduce(uint32_t(error != nullptr), comm::Op::MAX) : 0;
    if (error) std::rethrow_exception(error);
    if (failed) throw std::runtime_error("[[probes]]: rank 0 could not set up the probe files.");
}

const std::unordered_map<std::string, Reduction> REDUCTIONS = {
    {"integral", Reduction::INTEGRAL}, {"mean", Reduction::MEAN}, {"favre", Reduction::FAVRE},
    {"min", Reduction::MIN},           {"max", Reduction::MAX},   {"rms", Reduction::RMS}};

/** @brief The name of the measure of g's elements: a revolved boundary's per radian is an area. */
const char * measure_name(const probe::Geometry & g) {
    static const char * const NAMES[] = {"count", "length", "area", "volume"};
    return NAMES[g.dim + (g.revolved ? 1 : 0)];
}

/** @brief The description of a sample array: a geometry one (arc_length, normal, measure) or a variable's. */
FieldDescription geometry_field(const std::string & name, const probe::Geometry & g) {
    if (name == "arc_length") return {dim::length, "arc length", "", ""};
    if (name == "normal") return {dim::none, "unit normal", "", ""};
    if (name == "measure") {
        const Dimension measure = g.revolved ? dim::of(0, g.dim + 1, 0, 0, 0, -1) : dim::none.times(dim::length, g.dim);
        return {measure, measure_name(g), "", ""};
    }
    return field_units::describe(name);
}

std::string reduction_name(Reduction r) {
    for (const auto & [name, value] : REDUCTIONS) {
        if (value == r) return name;
    }
    return "";
}

/**
 * @brief Codes of the variables of samples on faces, after CellSampler's:
 *        convective flux densities through the face along its normal, and
 *        wall quantities. Vector quantities take one code per component.
 */
namespace code {
constexpr int32_t FIRST = 1 << 20;
constexpr int32_t MASS_FLUX = FIRST;
constexpr int32_t MOMENTUM_FLUX = FIRST + 1;  // + component
constexpr int32_t ENTHALPY_FLUX = FIRST + 4;
constexpr int32_t TRACTION_PRESSURE = FIRST + 5;  // + component
constexpr int32_t TRACTION_VISCOUS = FIRST + 8;   // + component
constexpr int32_t TAU_W_VECTOR = FIRST + 11;      // + component
constexpr int32_t TAU_W = FIRST + 14;
constexpr int32_t CF = FIRST + 15;
constexpr int32_t CP = FIRST + 16;
constexpr int32_t Q_W = FIRST + 17;
constexpr int32_t U_TAU = FIRST + 18;
constexpr int32_t Y_PLUS = FIRST + 19;
constexpr int32_t SPECIES_FLUX = FIRST + 32;  // + species

constexpr bool on_faces(int32_t k) { return k >= TRACTION_PRESSURE && k < SPECIES_FLUX; }
constexpr bool normals(int32_t k) { return k >= FIRST; }
constexpr bool gradients(int32_t k) { return (k >= TRACTION_VISCOUS && k <= Q_W && k != CP) || k == U_TAU || k == Y_PLUS; }
constexpr bool reference(int32_t k) { return k == CF || k == CP; }

const std::vector<std::pair<std::string, int32_t>> SCALARS = {
    {"MASS_FLUX", MASS_FLUX}, {"ENTHALPY_FLUX", ENTHALPY_FLUX}, {"TAU_W", TAU_W}, {"CF", CF}, {"CP", CP},
    {"Q_W", Q_W},             {"U_TAU", U_TAU},                 {"Y_PLUS", Y_PLUS}};
const std::vector<std::pair<std::string, int32_t>> VECTORS = {{"MOMENTUM_FLUX", MOMENTUM_FLUX},
                                                              {"TRACTION_PRESSURE", TRACTION_PRESSURE},
                                                              {"TRACTION_VISCOUS", TRACTION_VISCOUS},
                                                              {"TAU_W", TAU_W_VECTOR}};
} // namespace code

const char * const AXES = "XYZ";

int32_t variable_code(const std::string & name, const std::vector<std::string> & species_names,
                      const std::string & key) {
    for (const auto & [base, k] : code::SCALARS) {
        if (name == base) return k;
    }
    for (const auto & [base, k] : code::VECTORS) {
        for (size_t d = 0; d < N_DIM; d++) {
            if (name == base + "_" + AXES[d]) return k + static_cast<int32_t>(d);
        }
    }
    for (size_t k = 0; k < species_names.size(); k++) {
        if (name == "SPECIES_FLUX_" + species_names[k]) return code::SPECIES_FLUX + static_cast<int32_t>(k);
    }
    try {
        return CellSampler::code(name, species_names, key);
    } catch (const InputError &) {
        throw InputError(key + ": unknown variable \"" + name + "\" (the [statistics] fields, or on boundaries " +
                         "MASS_FLUX, MOMENTUM_FLUX, ENTHALPY_FLUX, SPECIES_FLUX_<species>, TRACTION_PRESSURE, " +
                         "TRACTION_VISCOUS, TAU_W, TAU_W_<axis>, CF, CP, Q_W, U_TAU, Y_PLUS).");
    }
}

/**
 * @brief The variables a [[probes]] entry lists, with U, MOMENTUM_FLUX,
 *        TRACTION_PRESSURE and TRACTION_VISCOUS for their components, Y_* and
 *        SPECIES_FLUX_* for every species.
 */
std::vector<std::string> expand_variables(const std::vector<std::string> & listed,
                                          const std::vector<std::string> & species_names, const std::string & key) {
    std::vector<std::string> names;
    for (const auto & v : listed) {
        if (v == "U" || v == "MOMENTUM_FLUX" || v == "TRACTION_PRESSURE" || v == "TRACTION_VISCOUS") {
            for (size_t d = 0; d < N_DIM; d++) names.push_back(v + "_" + AXES[d]);
        } else if (v == "Y_*" || v == "SPECIES_FLUX_*") {
            if (species_names.empty()) throw InputError(key + ".variables: " + v + " needs a gas mixture.");
            for (const auto & s : species_names) names.push_back(v.substr(0, v.size() - 1) + s);
        } else {
            names.push_back(v);
        }
    }
    for (size_t i = 0; i < names.size(); i++) {
        if (std::find(names.begin(), names.begin() + static_cast<long>(i), names[i]) != names.begin() + static_cast<long>(i)) {
            throw InputError(key + ".variables: \"" + names[i] + "\" is listed twice.");
        }
    }
    return names;
}

/** @brief A probe's VTK PolyData: its points, cells and whether the samples are the cells. */
struct PolyLayout {
    std::vector<double> xyz;  // 3 per point
    std::vector<int64_t> connectivity, offsets;  // offsets from 0
    const char * cells;  // Verts, Lines or Polys
    bool cell_data;
    size_t n_points() const { return xyz.size() / 3; }
};

/**
 * @brief Point sets as vertices, curves as polylines over the samples; polygons
 *        (segments in 2D) over vertices each a sample (faces, cut planes), or
 *        over the samples (resampled surfaces).
 */
PolyLayout poly_layout(const probe::Geometry & g) {
    PolyLayout l;
    const bool polygons = !g.polygon_offsets.empty();
    l.cells = polygons ? (g.dim == 1 ? "Lines" : "Polys") : g.dim == 0 ? "Verts" : "Lines";
    l.cell_data = polygons && !g.vertices.empty();
    const std::vector<probe::Point> & points = l.cell_data ? g.vertices : g.nodes;
    l.xyz.assign(3 * points.size(), 0.0);
    for (size_t p = 0; p < points.size(); p++) {
        for (size_t d = 0; d < N_DIM; d++) l.xyz[3 * p + d] = points[p][d];
    }
    l.offsets = {0};
    if (polygons) {
        l.connectivity.assign(g.polygon_vertices.begin(), g.polygon_vertices.end());
        l.offsets.assign(g.polygon_offsets.begin(), g.polygon_offsets.end());
    } else if (g.dim == 0) {
        for (size_t p = 0; p < g.nodes.size(); p++) {
            l.connectivity.push_back(static_cast<int64_t>(p));
            l.offsets.push_back(static_cast<int64_t>(p + 1));
        }
    } else {
        for (const auto & line : g.polylines) {
            for (uint32_t p : line) l.connectivity.push_back(p);
            l.offsets.push_back(static_cast<int64_t>(l.connectivity.size()));
        }
    }
    return l;
}

struct GeometryArray {
    std::string name;
    int components;
    std::vector<double> values;
};

/** @brief The per-sample geometry a probe writes beside its variables: arc_length, normal, measure. */
std::vector<GeometryArray> geometry_arrays(const probe::Geometry & g) {
    std::vector<GeometryArray> arrays;
    if (!g.arc_length.empty()) arrays.push_back({"arc_length", 1, g.arc_length});
    if (!g.normals.empty()) {
        std::vector<double> normal(3 * g.normals.size(), 0.0);
        for (size_t p = 0; p < g.normals.size(); p++) {
            for (size_t d = 0; d < N_DIM; d++) normal[3 * p + d] = g.normals[p][d];
        }
        arrays.push_back({"normal", 3, normal});
    }
    if (!g.node_weights.empty()) arrays.push_back({"measure", 1, g.node_weights});
    return arrays;
}

/**
 * @brief A VTK XML PolyData file with its arrays appended raw: the samples as
 *        vertices (points), polyline points (curves) or polygons (3D
 *        surfaces of mesh faces), with one array per variable.
 */
void write_vtp_file(const std::string & file, const probe::Geometry & g, const std::vector<std::string> & names,
                    const std::vector<double> & values, size_t n_columns) {
    struct Array {
        std::string name, type;
        int components;
        std::vector<char> bytes;
    };
    auto doubles = [](const std::string & name, int components, const std::vector<double> & v) {
        Array a{name, "Float64", components, std::vector<char>(v.size() * sizeof(double))};
        std::memcpy(a.bytes.data(), v.data(), a.bytes.size());
        return a;
    };
    auto integers = [](const std::string & name, const std::vector<int64_t> & v) {
        Array a{name, "Int64", 1, std::vector<char>(v.size() * sizeof(int64_t))};
        std::memcpy(a.bytes.data(), v.data(), a.bytes.size());
        return a;
    };
    const PolyLayout layout = poly_layout(g);
    const size_t n = g.nodes.size();
    const std::vector<int64_t> offsets(layout.offsets.begin() + 1, layout.offsets.end());
    std::vector<Array> sample_data;
    for (size_t v = 0; v < names.size(); v++) {
        std::vector<double> column(n);
        for (size_t p = 0; p < n; p++) column[p] = values[p * n_columns + v];
        sample_data.push_back(doubles(names[v], 1, column));
    }
    for (const auto & a : geometry_arrays(g)) sample_data.push_back(doubles(a.name, a.components, a.values));
    std::vector<Array> arrays = {doubles("Points", 3, layout.xyz), integers("connectivity", layout.connectivity),
                                 integers("offsets", offsets)};
    for (auto & a : sample_data) arrays.push_back(std::move(a));

    make_parent(file);
    std::ofstream out(file, std::ios::binary | std::ios::trunc);
    if (!out.good()) throw cannot_write("the probe file", file);
    uint64_t offset = 0;
    auto data_array = [&](const Array & a) {
        out << "        <DataArray type=\"" << a.type << "\" Name=\"" << a.name << "\" NumberOfComponents=\""
            << a.components << "\" format=\"appended\" offset=\"" << offset << "\"/>\n";
        offset += sizeof(uint64_t) + a.bytes.size();
    };
    const std::string cells = layout.cells;
    const char * data = layout.cell_data ? "CellData" : "PointData";
    const size_t n_cells = offsets.size();
    std::vector<std::string> sample_names;
    std::vector<FieldDescription> descriptions;
    for (size_t a = 3; a < arrays.size(); a++) {
        sample_names.push_back(arrays[a].name);
        descriptions.push_back(geometry_field(arrays[a].name, g));
    }
    // Field data belongs to the data set, before the piece, for VTK to read it
    out << "<?xml version=\"1.0\"?>\n" << provenance::xml_comment() << "\n<VTKFile type=\"PolyData\" version=\"1.0\" byte_order=\"LittleEndian\" "
           "header_type=\"UInt64\">\n  <PolyData>\n";
    out << "    <FieldData>\n      <DataArray type=\"Int32\" Name=\"MALLARD_FORMAT_VERSION\" NumberOfTuples=\"1\" "
           "format=\"ascii\">"
        << file_format::PROBE_OUTPUT.version << "</DataArray>\n"
        << field_units::vtk_field_data(sample_names, descriptions, "      ") << "    </FieldData>\n";
    out << "    <Piece NumberOfPoints=\""
        << layout.n_points() << "\" NumberOfVerts=\"" << (cells == "Verts" ? n_cells : 0) << "\" NumberOfLines=\""
        << (cells == "Lines" ? n_cells : 0) << "\" NumberOfStrips=\"0\" NumberOfPolys=\""
        << (cells == "Polys" ? n_cells : 0) << "\">\n";
    out << "      <Points>\n";
    data_array(arrays[0]);
    out << "      </Points>\n      <" << cells << ">\n";
    data_array(arrays[1]);
    data_array(arrays[2]);
    out << "      </" << cells << ">\n      <" << data << ">\n";
    for (size_t a = 3; a < arrays.size(); a++) data_array(arrays[a]);
    out << "      </" << data << ">\n    </Piece>\n  </PolyData>\n  <AppendedData encoding=\"raw\">\n   _";
    for (const auto & a : arrays) {
        const uint64_t size = a.bytes.size();
        out.write(reinterpret_cast<const char *>(&size), sizeof(size));
        out.write(a.bytes.data(), static_cast<std::streamsize>(a.bytes.size()));
    }
    out << "\n  </AppendedData>\n</VTKFile>\n";
}

void write_pvd_file(const std::string & file, const std::vector<std::pair<double, std::string>> & series) {
    make_parent(file);
    std::ofstream out(file, std::ios::trunc);
    out << "<?xml version=\"1.0\"?>\n" << provenance::xml_comment() << "\n<VTKFile type=\"Collection\" version=\"1.0\">\n  <Collection>\n";
    out << std::setprecision(std::numeric_limits<double>::max_digits10);
    for (const auto & [t, name] : series) {
        out << "    <DataSet timestep=\"" << t << "\" file=\"" << name << "\"/>\n";
    }
    out << "  </Collection>\n</VTKFile>\n";
}

} // namespace

void Probes::select_region(Probe & p, const Mesh & mesh, const region::Region & r,
                           const std::vector<double> & fractions, uint32_t n_variables) {
    std::vector<uint32_t> cells;
    std::vector<uint64_t> global;
    std::vector<double> weights;
    for (uint32_t c = 0; c < mesh.n_owned(); c++) {
        if (fractions[c] == 0.0) continue;
        cells.push_back(c);
        global.push_back(mesh.h_global_cell(c));
        weights.push_back(fractions[c] * double(mesh.h_cell_measure(c)));
    }
    p.quadrature.select(mesh, cells, n_variables);
    if (mesh.n_global_cells > 0) {
        global = comm::allgatherv(global);
        weights = comm::allgatherv(weights);
    }
    std::vector<size_t> by_id(global.size());
    for (size_t i = 0; i < by_id.size(); i++) by_id[i] = i;
    std::sort(by_id.begin(), by_id.end(), [&global](const size_t a, const size_t b) { return global[a] < global[b]; });
    p.geometry.dim = static_cast<uint8_t>(r.dimension());
    p.geometry.revolved = mesh.axisymmetric;  // per radian
    p.geometry.type = "region " + r.summary();
    for (const size_t i : by_id) p.geometry.quad_weights.push_back(weights[i]);
}

namespace {

/** @brief Device views of samples in cells (inside: index into cells of those with an offset). */
void allocate_samples(Probes::Samples & s, const std::vector<uint32_t> & h_cells, const std::vector<uint32_t> & h_inside,
                      const std::vector<rtype> & h_offsets, uint32_t n_variables) {
    std::vector<uint32_t> h_inside_cells;
    for (uint32_t i : h_inside) h_inside_cells.push_back(h_cells[i]);
    s.cells = to_device(h_cells, "probe_cells");
    s.inside = to_device(h_inside, "probe_inside");
    s.inside_cells = to_device(h_inside_cells, "probe_inside_cells");
    s.inside_offsets = Kokkos::View<rtype *[N_DIM]>("probe_offsets", h_inside.size());
    auto h_off = Kokkos::create_mirror_view(s.inside_offsets);
    for (size_t i = 0; i < h_inside.size(); i++) {
        for (size_t d = 0; d < N_DIM; d++) h_off(i, d) = h_offsets[i * N_DIM + d];
    }
    Kokkos::deep_copy(s.inside_offsets, h_off);
    std::vector<uint32_t> unique = h_cells;
    std::sort(unique.begin(), unique.end());
    unique.erase(std::unique(unique.begin(), unique.end()), unique.end());
    s.unique_cells = to_device(unique, "probe_unique_cells");
    s.inside_W = Kokkos::View<rtype *[N_CONSERVATIVE]>("probe_inside_W", h_inside.size());
    s.W = Kokkos::View<rtype *[N_CONSERVATIVE]>("probe_W", h_cells.size());
    s.values = Kokkos::View<rtype **, Kokkos::LayoutRight>("probe_values", h_cells.size(), n_variables);
}

} // namespace

void Probes::Samples::locate(const Mesh & mesh, const std::vector<probe::Point> & points, uint32_t n_variables,
                             const std::vector<probe::Point> & point_normals) {
    n_global = points.size();
    std::vector<uint8_t> contained;
    const std::vector<int32_t> located = locate_points(mesh, points, &contained);
    ids.clear();
    std::vector<uint32_t> h_cells, h_inside;
    std::vector<rtype> h_offsets;
    for (uint64_t p = 0; p < points.size(); p++) {
        if (located[p] < 0) continue;
        const uint32_t c = static_cast<uint32_t>(located[p]);
        if (contained[p]) {
            h_inside.push_back(static_cast<uint32_t>(ids.size()));
            for (size_t d = 0; d < N_DIM; d++) {
                h_offsets.push_back(static_cast<rtype>(points[p][d] - static_cast<double>(mesh.h_cell_coords(c, d))));
            }
        }
        ids.push_back(p);
        h_cells.push_back(c);
    }
    allocate_samples(*this, h_cells, h_inside, h_offsets, n_variables);
    if (!point_normals.empty()) {
        std::vector<probe::Point> local;
        for (uint64_t id : ids) local.push_back(point_normals[id]);
        set_normals(local);
    }
    order(mesh);
}

void Probes::Samples::select(const Mesh & mesh, const std::vector<uint32_t> & owned_cells, uint32_t n_variables) {
    std::vector<uint64_t> global(owned_cells.size());
    for (size_t i = 0; i < owned_cells.size(); i++) global[i] = mesh.h_global_cell(owned_cells[i]);
    std::vector<uint64_t> all = mesh.n_global_cells > 0 ? comm::allgatherv(global) : global;
    std::sort(all.begin(), all.end());
    n_global = all.size();
    for (const uint64_t g : global) {
        ids.push_back(static_cast<uint64_t>(std::lower_bound(all.begin(), all.end(), g) - all.begin()));
    }
    cells = to_device(owned_cells, "probe_cells");
    W = Kokkos::View<rtype *[N_CONSERVATIVE]>("probe_W", owned_cells.size());
    values = Kokkos::View<rtype **, Kokkos::LayoutRight>("probe_values", owned_cells.size(), n_variables);
    order(mesh);
}

void Probes::Samples::order(const Mesh & mesh) {
    distributed = mesh.n_global_cells > 0;
    const std::vector<uint64_t> all = distributed ? comm::gatherv(ids) : ids;
    if (comm::is_root() || !distributed) {
        if (all.size() != n_global) throw std::logic_error("probe samples held by no rank or by several");
        gathered_row.assign(n_global, 0);
        for (uint64_t r = 0; r < all.size(); r++) gathered_row[all[r]] = r;
    }
}

void Probes::Samples::on_faces(const Mesh & mesh, const probe::FaceSet & set, uint32_t n_variables) {
    n_global = set.n_global;
    distributed = mesh.n_global_cells > 0;
    gathered_row = set.gathered_row;
    const size_t n = set.faces.size();
    std::vector<uint32_t> h_cells, h_inside;
    std::vector<rtype> h_offsets;
    faces = to_device(set.faces, "probe_faces");
    normals = Kokkos::View<rtype *[N_DIM]>("probe_normals", n);
    distance = Kokkos::View<rtype *>("probe_distance", n);
    auto h_normals = Kokkos::create_mirror_view(normals);
    auto h_distance = Kokkos::create_mirror_view(distance);
    for (size_t i = 0; i < n; i++) {
        const uint32_t f = set.faces[i];
        const uint32_t c = static_cast<uint32_t>(mesh.h_cells_of_face(f, 0));
        h_cells.push_back(c);
        h_inside.push_back(static_cast<uint32_t>(i));
        double area = 0.0, dn = 0.0;
        for (size_t d = 0; d < N_DIM; d++) area += std::pow(static_cast<double>(mesh.h_face_normals(f, d)), 2);
        area = std::sqrt(area);
        for (size_t d = 0; d < N_DIM; d++) {
            const double r = static_cast<double>(mesh.h_face_coords(f, d)) - static_cast<double>(mesh.h_cell_coords(c, d));
            const double nd = static_cast<double>(mesh.h_face_normals(f, d)) / area;
            h_offsets.push_back(static_cast<rtype>(r));
            h_normals(i, d) = static_cast<rtype>(nd);
            dn += r * nd;
        }
        h_distance(i) = static_cast<rtype>(dn);
    }
    Kokkos::deep_copy(normals, h_normals);
    Kokkos::deep_copy(distance, h_distance);
    allocate_samples(*this, h_cells, h_inside, h_offsets, n_variables);
}

void Probes::Samples::in_cells(const Mesh & mesh, const probe::CellPoints & set, const std::vector<probe::Point> & n,
                               uint32_t n_variables) {
    n_global = set.n_global;
    distributed = mesh.n_global_cells > 0;
    gathered_row = set.gathered_row;
    std::vector<uint32_t> h_inside;
    std::vector<rtype> h_offsets;
    for (size_t i = 0; i < set.cells.size(); i++) {
        h_inside.push_back(static_cast<uint32_t>(i));
        for (size_t d = 0; d < N_DIM; d++) {
            h_offsets.push_back(static_cast<rtype>(set.points[i][d] - static_cast<double>(mesh.h_cell_coords(set.cells[i], d))));
        }
    }
    allocate_samples(*this, set.cells, h_inside, h_offsets, n_variables);
    set_normals(std::vector<probe::Point>(set.cells.size(), n.empty() ? probe::Point{} : n[0]));
}

void Probes::Samples::set_normals(const std::vector<probe::Point> & n) {
    normals = Kokkos::View<rtype *[N_DIM]>("probe_normals", n.size());
    auto h = Kokkos::create_mirror_view(normals);
    for (size_t i = 0; i < n.size(); i++) {
        for (size_t d = 0; d < N_DIM; d++) h(i, d) = static_cast<rtype>(n[i][d]);
    }
    Kokkos::deep_copy(normals, h);
}

void Probes::Samples::evaluate(Sampling sampling, Kokkos::View<int32_t *> codes, const ProbeFields & fields,
                               const ProbeReference & reference, bool need_gradients) {
    const uint32_t n = static_cast<uint32_t>(cells.extent(0));
    const uint32_t n_vars = static_cast<uint32_t>(codes.extent(0));
    const auto cell = cells;
    const auto code = codes;
    const auto out = values;
    const CellSampler sampler = fields.cells;
    const bool on_faces = faces.extent(0) > 0;
    const bool has_normals = normals.extent(0) > 0;
    if (sampling == Sampling::CELL && !has_normals) {
        Kokkos::parallel_for("probe_cell_values", n, KOKKOS_LAMBDA(const uint32_t i) {
            for (uint32_t v = 0; v < n_vars; v++) out(i, v) = sampler(cell(i), code(v));
        });
        return;
    }
    constexpr uint8_t E = N_DIM + 1;
    const auto G = fields.gradients;
    if (sampling == Sampling::POINT || need_gradients) {
        const auto gradient = fields.gradient;
        const auto unique = unique_cells;
        Kokkos::parallel_for("probe_gradients", unique.extent(0), KOKKOS_LAMBDA(const uint32_t i) { gradient(unique(i)); });
    }
    // The trace: the cell average (from the current conservatives), or the reconstruction at the sample
    const auto Ws = W;
    Kokkos::parallel_for("probe_cell_W", n, KOKKOS_LAMBDA(const uint32_t s) {
        const uint32_t c = cell(s);
        Ws(s, 0) = sampler.U(c, 0);
        for (uint8_t i = 0; i <= N_DIM; i++) Ws(s, 1 + i) = sampler.P(c, i);
    });
    if (sampling == Sampling::POINT) {
        const auto Wc = fields.W;
        const auto in_cells = inside_cells;
        const auto in_offsets = inside_offsets;
        const auto in_W = inside_W;
        // Linear least squares, kept where the reconstruction has no admissible polynomial value
        Kokkos::parallel_for("probe_linear_values", in_cells.extent(0), KOKKOS_LAMBDA(const uint32_t j) {
            const uint32_t c = in_cells(j);
            rtype Wq[N_CONSERVATIVE];
            FOR_I_CONSERVATIVE {
                Wq[i] = Wc(c, i);
                for (uint8_t d = 0; d < N_DIM; d++) Wq[i] += G(c, i, d) * in_offsets(j, d);
            }
            const bool admissible = Wq[0] > 0.0_r && Wq[E] > 0.0_r;
            FOR_I_CONSERVATIVE in_W(j, i) = admissible ? Wq[i] : Wc(c, i);
        });
        if (fields.reconstruction) fields.reconstruction->point_values(Wc, in_cells, in_offsets, in_W);
        const auto in = inside;
        Kokkos::parallel_for("probe_point_W", in.extent(0), KOKKOS_LAMBDA(const uint32_t j) {
            FOR_I_CONSERVATIVE Ws(in(j), i) = in_W(j, i);
        });
    }

    const Euler physics = fields.physics;
    const auto cell_gamma = fields.cell_gamma;
    const bool mixture = cell_gamma.extent(0) > 0;
    const auto face = faces;
    const auto normal = normals;
    const auto dist = distance;
    const BoundaryData boundaries = fields.boundaries;
    const bool viscous = fields.viscous;
    if (on_faces) {
        // The state the boundary condition sets on the face
        Kokkos::parallel_for("probe_face_state", n, KOKKOS_LAMBDA(const uint32_t s) {
            const uint32_t c = cell(s), f = face(s);
            const BoundaryCondition & bc = boundaries.bcs(boundaries.face_bc(f));
            rtype Wf[N_CONSERVATIVE], nv[N_DIM];
            FOR_I_CONSERVATIVE Wf[i] = Ws(s, i);
            FOR_I_DIM nv[i] = normal(s, i);
            const rtype R_c = mixture ? sampler.P(c, N_DIM) / (sampler.U(c, 0) * sampler.P(c, N_DIM + 1)) : physics.R;
            const rtype g = mixture ? cell_gamma(c) : physics.gamma;
            const rtype u_n = dot<N_DIM>(Wf + 1, nv);
            if (bc.is_wall() && viscous) {
                FOR_I_DIM Wf[1 + i] = bc.data[1 + i];
                if (bc.type == BoundaryType::WALL_ISOTHERMAL) Wf[0] = Wf[E] / (R_c * bc.data[0]);
            } else if (bc.is_wall() || bc.type == BoundaryType::SYMMETRY) {
                FOR_I_DIM Wf[1 + i] -= u_n * nv[i];
            } else if (bc.type == BoundaryType::UPT || bc.type == BoundaryType::FARFIELD ||
                       bc.type == BoundaryType::P_OUT || bc.type == BoundaryType::P_OUT_AVERAGE) {
                rtype Wg[N_CONSERVATIVE];
                bc.ghost_W(Wf, nv, g, R_c, viscous, Wg);
                FOR_I_CONSERVATIVE Wf[i] = Wg[i];
            } else if (bc.type == BoundaryType::DIRICHLET) {
                const int32_t k = boundaries.face_state_index(f);
                if (k >= 0) FOR_I_CONSERVATIVE Wf[i] = boundaries.face_state(k, i);
            }
            FOR_I_CONSERVATIVE Ws(s, i) = Wf[i];
        });
    }

    const auto transport = fields.transport;
    const auto face_coords = fields.face_coords;
    const bool axisymmetric = fields.axisymmetric;
    const ProbeReference ref = reference;
    Kokkos::parallel_for("probe_values", n, KOKKOS_LAMBDA(const uint32_t s) {
        const uint32_t c = cell(s);
        const rtype g = mixture ? cell_gamma(c) : physics.gamma;
        const rtype rho_c = sampler.U(c, 0), p_c = sampler.P(c, N_DIM), T_c = sampler.P(c, N_DIM + 1);
        rtype Wf[N_CONSERVATIVE];
        FOR_I_CONSERVATIVE Wf[i] = Ws(s, i);
        // Temperature and enthalpy follow p / rho at the cell's composition
        const rtype pv = Wf[E] / Wf[0], pv_c = p_c / rho_c;
        const rtype T = T_c * (pv / pv_c);
        const rtype h = sampler.P(c, N_DIM + 2) + g / (g - 1.0_r) * (pv - pv_c);
        rtype nv[N_DIM] = {}, u_n = 0.0_r;
        int32_t ibc = -1;
        if (has_normals) {
            FOR_I_DIM nv[i] = normal(s, i);
            u_n = dot<N_DIM>(Wf + 1, nv);
        }
        if (on_faces) ibc = boundaries.face_bc(face(s));
        // Viscous traction tau . n and heat flux into the fluid, with the
        // viscous flux's wall treatment: the gradients normal to walls from
        // the wall values
        rtype tau_n[N_DIM] = {}, q_n = 0.0_r, mu = 0.0_r;
        if (need_gradients && on_faces) {
            const BoundaryCondition & bc = boundaries.bcs(ibc);
            const rtype R_c = mixture ? p_c / (rho_c * T_c) : physics.R;
            rtype gu[N_DIM][N_DIM], gT[N_DIM];
            FOR_I_DIM {
                for (uint8_t k = 0; k < N_DIM; k++) gu[k][i] = G(c, 1 + k, i);
                gT[i] = (G(c, E, i) - R_c * T_c * G(c, 0, i)) / (rho_c * R_c);
            }
            if (bc.is_wall()) {
                const rtype dn = dist(s);
                for (uint8_t k = 0; k < N_DIM; k++) {
                    const rtype correction = (Wf[1 + k] - sampler.P(c, k)) / dn - dot<N_DIM>(gu[k], nv);
                    FOR_I_DIM gu[k][i] += correction * nv[i];
                }
                if (bc.type == BoundaryType::WALL_ISOTHERMAL) {
                    const rtype correction = (bc.data[0] - T_c) / dn - dot<N_DIM>(gT, nv);
                    FOR_I_DIM gT[i] += correction * nv[i];
                }
            }
            mu = transport.extent(0) > 0 ? transport(c, 0) : physics.viscosity(T);
            const rtype kappa = transport.extent(0) > 0 ? transport(c, 1) : physics.conductivity(mu);
            viscous_traction(mu, gu, nv, tau_n,
                             axisymmetric ? hoop_divergence(face_coords(face(s), 1), Wf + 1, gu) : 0.0_r);
            q_n = kappa * dot<N_DIM>(gT, nv);
            if (bc.type == BoundaryType::WALL_ADIABATIC) q_n = 0.0_r;
            if (bc.type == BoundaryType::WALL_HEAT_FLUX) q_n = bc.data[N_DIM + 1];
        }
        // Wall shear stress: the tangential part of the fluid's viscous traction on the wall
        rtype tau_w[N_DIM];
        FOR_I_DIM tau_w[i] = -tau_n[i];
        const rtype tau_nn = dot<N_DIM>(tau_w, nv);
        FOR_I_DIM tau_w[i] -= tau_nn * nv[i];
        const rtype tau_mag = Kokkos::sqrt(dot<N_DIM>(tau_w, tau_w));
        const rtype u_tau = Kokkos::sqrt(tau_mag / Wf[0]);
        for (uint32_t v = 0; v < n_vars; v++) {
            const int32_t k = code(v);
            rtype value = 0.0_r;
            if (k == CellSampler::RHO) {
                value = Wf[0];
            } else if (k < CellSampler::FIRST_PRIMITIVE + N_DIM + 1) {
                value = Wf[1 + (k - CellSampler::FIRST_PRIMITIVE)];
            } else if (k == CellSampler::FIRST_PRIMITIVE + N_DIM + 1) {
                value = T;
            } else if (k == CellSampler::FIRST_PRIMITIVE + N_DIM + 2) {
                value = h;
            } else if (k < code::FIRST) {
                value = sampler(c, k);
            } else if (k == code::MASS_FLUX) {
                value = Wf[0] * u_n;
            } else if (k < code::MOMENTUM_FLUX + 3) {
                const int32_t d = k - code::MOMENTUM_FLUX;
                value = Wf[0] * Wf[1 + d] * u_n + Wf[E] * nv[d];
            } else if (k == code::ENTHALPY_FLUX) {
                value = Wf[0] * u_n * (h + 0.5_r * dot<N_DIM>(Wf + 1, Wf + 1));
            } else if (k < code::TRACTION_PRESSURE + 3) {
                value = Wf[E] * nv[k - code::TRACTION_PRESSURE];
            } else if (k < code::TRACTION_VISCOUS + 3) {
                value = -tau_n[k - code::TRACTION_VISCOUS];
            } else if (k < code::TAU_W_VECTOR + 3) {
                value = tau_w[k - code::TAU_W_VECTOR];
            } else if (k == code::TAU_W) {
                value = tau_mag;
            } else if (k == code::CF) {
                value = dot<N_DIM>(tau_w, ref.e) / ref.q;
            } else if (k == code::CP) {
                value = (Wf[E] - ref.p) / ref.q;
            } else if (k == code::Q_W) {
                value = q_n;
            } else if (k == code::U_TAU) {
                value = u_tau;
            } else if (k == code::Y_PLUS) {
                value = Wf[0] * u_tau * dist(s) / mu;
            } else {
                const int32_t species = k - code::SPECIES_FLUX;
                const bool prescribed = ibc >= 0 && boundaries.bcs(ibc).type == BoundaryType::UPT &&
                                        boundaries.bc_Y.extent(0) > 0;
                const rtype Y = prescribed ? boundaries.bc_Y(ibc, species)
                                           : sampler(c, CellSampler::FIRST_SPECIES + species);
                value = Wf[0] * Y * u_n;
            }
            out(s, v) = value;
        }
    });
}

std::vector<double> Probes::Samples::gather() const {
    const auto h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), values);
    const size_t n_vars = values.extent(1);
    std::vector<double> local(h.size());
    for (size_t i = 0; i < local.size(); i++) local[i] = static_cast<double>(h.data()[i]);
    const std::vector<double> all = distributed ? comm::gatherv(local) : local;
    if (!comm::is_root() && distributed) return {};
    std::vector<double> ordered(all.size());
    for (uint64_t p = 0; p < n_global; p++) {
        std::copy_n(all.begin() + static_cast<long>(gathered_row[p] * n_vars), n_vars,
                    ordered.begin() + static_cast<long>(p * n_vars));
    }
    return ordered;
}

void Probes::init(const Input & input, const Mesh & mesh, region::Regions & regions,
                  const std::vector<std::string> & species_names, bool resume, bool viscous) {
    resumed = resume;
    distributed = mesh.n_global_cells > 0;
    if (!input.contains("probes")) return;
    const auto entries = find_value<std::vector<Input>>(input, "probes");
    for (const auto & entry : entries) {
        auto probe_ptr = std::make_unique<Probe>();
        Probe & p = *probe_ptr;
        if (!entry.contains("name")) throw InputError("probes: every probe needs a name.");
        p.name = find_value<std::string>(entry, "name");
        const std::string key = "probes[name = \"" + p.name + "\"]";
        for (const auto & other : probes) {
            if (other->name == p.name) throw InputError(key + ": name is used twice.");
        }

        const bool steps = entry.contains("interval"), times = entry.contains("time_interval");
        if (steps && times) throw InputError(key + ": give interval or time_interval, not both.");
        if (times) {
            p.time_interval = find_real(entry, "time_interval", dim::time);
            if (!(p.time_interval > 0.0_r)) throw InputError(key + ".time_interval must be positive.");
        } else {
            p.interval = find_value_or<uint64_t>(entry, "interval", 1);
            if (p.interval == 0) throw InputError(key + ".interval must be positive.");
        }

        p.from_region = entry.contains("region");
        for (const char * k : {"type", "boundary", "boundary_type", "sampling", "format", "file", "samples"}) {
            if (p.from_region && entry.contains(k)) {
                throw InputError(Diagnostic{.key = key + "." + k,
                                            .line = line_of(entry, k),
                                            .message = std::string("applies to probes of points and curves, but a "
                                                                   "region probe writes only its reductions, to "
                                                                   "reductions_file"),
                                            .fix = std::string("Remove ") + k + ", or give the probe a type instead "
                                                   "of a region",
                                            .docs = "docs/guide/output.md#region-probes"});
            }
        }
        const std::string sampling = find_value_or<std::string>(entry, "sampling", "cell");
        record_options(key + ".sampling", {"cell", "point"});
        if (sampling == "point") {
            p.sampling = Sampling::POINT;
        } else if (sampling != "cell") {
            throw InputError(key + ".sampling = \"" + sampling + "\" is not one of: cell, point.");
        }
        const std::string format = find_value_or<std::string>(entry, "format", "csv");
        record_options(key + ".format", {"csv", "hdf5", "vtkhdf", "vtp"});
        if (format == "hdf5" || format == "vtkhdf") {
#ifndef Mallard_HAS_HDF5
            throw InputError(key + ".format = \"" + format + "\" needs a build with -DMallard_ENABLE_HDF5=ON.");
#endif
            p.format = format == "hdf5" ? Format::HDF5 : Format::VTKHDF;
        } else if (format == "vtp") {
            p.format = Format::VTP;
        } else if (format != "csv") {
            throw not_accepted(entry, key + ".format", format, {"csv", "hdf5", "vtkhdf", "vtp"});
        }
        const char * extension = p.format == Format::CSV      ? ".csv"
                                 : p.format == Format::HDF5   ? ".h5"
                                 : p.format == Format::VTKHDF ? ".vtkhdf"
                                                              : ".pvd";
        p.file = find_value_or<std::string>(entry, "file", "probe_" + p.name + extension);
        p.write_samples = !p.from_region && find_value_or<bool>(entry, "samples", true);
        std::vector<std::string> reductions = find_value_or<std::vector<std::string>>(entry, "reductions", {});
        if (p.from_region && reductions.empty()) reductions = {"integral", "mean", "min", "max"};
        for (const auto & r : reductions) {
            const auto it = find_option(REDUCTIONS, key + ".reductions", r, line_of(entry, "reductions"));
            if (std::find(p.reductions.begin(), p.reductions.end(), it->second) != p.reductions.end()) {
                throw InputError(key + ".reductions: \"" + r + "\" is listed twice.");
            }
            p.reductions.push_back(it->second);
        }
        if (!p.write_samples && p.reductions.empty()) throw InputError(key + ": samples = false needs reductions.");
        p.reductions_file = find_value_or<std::string>(entry, "reductions_file", "probe_" + p.name + "_reductions.csv");

        if (!entry.contains("variables")) throw InputError(key + ": variables not specified.");
        p.variables = expand_variables(find_value<std::vector<std::string>>(entry, "variables"), species_names, key);
        if (p.variables.empty()) throw InputError(key + ".variables is empty.");
        std::vector<int32_t> h_codes;
        for (const auto & v : p.variables) h_codes.push_back(variable_code(v, species_names, key + ".variables"));
        const auto rho = std::find(h_codes.begin(), h_codes.end(), CellSampler::RHO);
        p.rho_column = static_cast<uint32_t>(rho - h_codes.begin());
        if (rho == h_codes.end() &&
            std::find(p.reductions.begin(), p.reductions.end(), Reduction::FAVRE) != p.reductions.end()) {
            h_codes.push_back(CellSampler::RHO);
        }
        p.codes = to_device(h_codes, "probe_codes");

        const bool boundary = entry.contains("boundary") || entry.contains("boundary_type");
        const std::string type = find_value_or<std::string>(entry, "type", "");
        const bool cut = type == "cut_plane";
        const bool oriented = !p.from_region &&
                              (cut || (N_DIM == 2 && entry.contains("type") && type != "point" && type != "points") ||
                               (N_DIM == 2 && !entry.contains("type") && !entry.contains("point")) ||
                               (N_DIM == 3 && (type == "rectangle" || type == "disk" || type == "sphere" ||
                                               type == "cylinder" || type == "cone" || type == "surface")));
        if (cut && mesh.axisymmetric && !p.reductions.empty()) {
            throw InputError(key + ".reductions: cut planes take no reductions with physics.axisymmetric.");
        }
        for (size_t v = 0; v < p.variables.size(); v++) {
            const int32_t k = h_codes[v];
            if (code::on_faces(k) && !boundary) {
                throw InputError(key + ".variables: " + p.variables[v] + " is for boundary probes.");
            }
            if (code::normals(k) && !boundary && !oriented) {
                throw InputError(key + ".variables: " + p.variables[v] + " needs a surface (" +
                                 (N_DIM == 2 ? "a curve, boundary or cut plane)." : "a boundary, cut plane or surface)."));
            }
            if (code::normals(k) && !boundary && mesh.axisymmetric) {
                throw InputError(key + ".variables: " + p.variables[v] +
                                 " through a cut plane or curve is not supported with physics.axisymmetric; boundary "
                                 "probes give fluxes per radian.");
            }
            if (code::gradients(k) && !viscous) {
                throw InputError(key + ".variables: " + p.variables[v] + " needs a viscous run.");
            }
            p.gradients = p.gradients || code::gradients(k);
            if (code::reference(k) && !entry.contains("reference")) {
                throw InputError(key + ".variables: " + p.variables[v] + " needs a reference = {rho, u, p}.");
            }
        }
        p.transport = p.gradients && !species_names.empty();
        if (entry.contains("reference")) {
            const Input & ref = entry.at("reference");
            const std::string rkey = key + ".reference";
            if (!ref.is_table() || !ref.contains("rho") || !ref.contains("u") || !ref.contains("p")) {
                throw InputError(rkey + " must be a table {rho, u = [...], p}.");
            }
            const std::vector<rtype> u = find_real_vector(ref, "u", dim::velocity);
            if (u.size() != N_DIM) throw InputError(rkey + ".u must have " + std::to_string(N_DIM) + " components.");
            double speed = 0.0;
            for (rtype x : u) speed += static_cast<double>(x) * static_cast<double>(x);
            speed = std::sqrt(speed);
            if (!(speed > 0.0)) throw InputError(rkey + ".u must not be zero.");
            for (size_t d = 0; d < N_DIM; d++) p.reference.e[d] = static_cast<rtype>(static_cast<double>(u[d]) / speed);
            p.reference.q = static_cast<rtype>(0.5 * static_cast<double>(find_real(ref, "rho", dim::density)) * speed * speed);
            p.reference.p = find_real(ref, "p", dim::pressure);
        }

        if (p.from_region) {
            const region::Region & r = regions.get(entry.at("region"), key + ".region");
            region::require_full_dimension(r, entry.at("region"), key + ".region");
            select_region(p, mesh, r, regions.fractions(entry.at("region"), key + ".region"),
                          static_cast<uint32_t>(h_codes.size()));
            if (p.quadrature.n_global == 0) {
                throw InputError(Diagnostic{.key = key + ".region",
                                            .line = line_of(entry, "region"),
                                            .message = "covers no part of any cell of the mesh",
                                            .fix = "Move or enlarge the region so that it overlaps the domain",
                                            .docs = "docs/guide/output.md#region-probes"});
            }
        } else if (boundary) {
            for (const char * k : {"type", "point", "start", "end"}) {
                if (entry.contains(k)) throw InputError(key + ": boundary probes take no " + k + ".");
            }
            std::vector<std::string> zones;
            if (entry.contains("boundary")) {
                const Input & b = entry.at("boundary");
                if (b.is_string()) {
                    zones.push_back(b.as_string());
                } else {
                    zones = find_value<std::vector<std::string>>(entry, "boundary");
                }
            }
            if (entry.contains("boundary_type")) {
                const std::string zone_type = find_value<std::string>(entry, "boundary_type");
                if (zone_type != "wall" && BOUNDARY_TYPES.find(zone_type) == BOUNDARY_TYPES.end()) {
                    throw InputError(key + ".boundary_type = \"" + zone_type + "\" is not wall or a boundary type.");
                }
                for (const auto & b : find_value_or<std::vector<Input>>(input, "boundaries", {})) {
                    const std::string t = find_value_or<std::string>(b, "type", "");
                    const auto it = BOUNDARY_TYPES.find(t);
                    const bool wall = it != BOUNDARY_TYPES.end() && BoundaryCondition{it->second}.is_wall();
                    if (t == zone_type || (zone_type == "wall" && wall)) zones.push_back(find_value<std::string>(b, "name"));
                }
            }
            if (zones.empty()) throw InputError(key + ": no boundary zone selected.");
            for (const auto & z : zones) {
                const auto & all = mesh.h_face_zones();
                const bool here = std::any_of(all.begin(), all.end(), [&](const FaceZone & fz) {
                    return fz.get_name() == z && fz.get_type() == FaceZoneType::BOUNDARY;
                });
                if ((distributed ? comm::allreduce(uint32_t(here), comm::Op::SUM) : uint32_t(here)) == 0) {
                    throw InputError(key + ": unknown boundary zone " + z + ".");
                }
            }
            probe::FaceSet set = probe::boundary_faces(mesh, zones, distributed);
            p.nodes.on_faces(mesh, set, static_cast<uint32_t>(h_codes.size()));
            p.geometry = std::move(set.geometry);
            p.geometry.type = zones.size() == 1 ? "boundary " + zones[0] : "boundaries";
        } else if (cut) {
            probe::Vec3 x = {0.0, 0.0, 0.0}, n = {0.0, 0.0, 0.0};
            const std::vector<rtype> xv = find_real_vector(entry, "point", dim::length),
                                     nv = find_real_vector(entry, "normal", dim::none);
            if (xv.size() != N_DIM || nv.size() != N_DIM) {
                throw InputError(key + ": point and normal must have " + std::to_string(N_DIM) + " components.");
            }
            for (size_t d = 0; d < N_DIM; d++) {
                x[d] = static_cast<double>(xv[d]);
                n[d] = static_cast<double>(nv[d]);
            }
            if (!(n[0] * n[0] + n[1] * n[1] + n[2] * n[2] > 0.0)) throw InputError(key + ".normal must not be zero.");
            std::array<probe::Vec3, 2> bounds{};
            const bool bounded = entry.contains("bounds");
            if (bounded) {
                const Input & b = entry.at("bounds");
                if (!b.is_array() || b.as_array().size() != 2) throw InputError(key + ".bounds must be [[lo...], [hi...]].");
                for (size_t k = 0; k < 2; k++) {
                    const Input & corner = b.as_array()[k];
                    if (!corner.is_array() || corner.as_array().size() != N_DIM) {
                        throw InputError(key + ".bounds corners must have " + std::to_string(N_DIM) + " components.");
                    }
                    for (size_t d = 0; d < N_DIM; d++) bounds[k][d] = as_double(corner.as_array()[d], key + ".bounds", dim::length);
                }
            }
            probe::CutSet set = probe::cut_plane(mesh, x, n, mesh.n_global_cells > 0, bounded ? &bounds : nullptr);
            const double l = std::sqrt(n[0] * n[0] + n[1] * n[1] + n[2] * n[2]);
            probe::Point unit_n;
            for (size_t d = 0; d < N_DIM; d++) unit_n[d] = n[d] / l;
            p.nodes.in_cells(mesh, set.nodes, {unit_n}, static_cast<uint32_t>(h_codes.size()));
            if (!p.reductions.empty()) p.quadrature.in_cells(mesh, set.quadrature, {unit_n}, static_cast<uint32_t>(h_codes.size()));
            p.geometry = std::move(set.geometry);
            if (p.nodes.n_global == 0) throw InputError(key + ": the plane misses the mesh.");
        } else {
            p.geometry = probe::make_geometry(entry, key);
            p.nodes.locate(mesh, p.geometry.nodes, static_cast<uint32_t>(h_codes.size()), p.geometry.normals);
            if (!p.reductions.empty() && p.geometry.dim > 0) {
                p.quadrature.locate(mesh, p.geometry.quad_points, static_cast<uint32_t>(h_codes.size()),
                                    p.geometry.quad_normals);
            }
        }

        on_root(distributed, [&] {
            if (p.write_samples && p.format == Format::CSV) {
                std::vector<std::pair<std::string, Dimension>> columns = {
                    {"step", dim::none}, {"t", dim::time}, {boundary ? "face" : cut ? "cell" : "point", dim::none}};
                for (size_t d = 0; d < N_DIM; d++) columns.emplace_back(std::string(1, "xyz"[d]), dim::length);
                if (!p.geometry.node_weights.empty()) {
                    if (!p.geometry.arc_length.empty()) columns.emplace_back("s", dim::length);
                    for (size_t d = 0; d < N_DIM; d++) columns.emplace_back(std::string("n") + "xyz"[d], dim::none);
                    columns.emplace_back("area", geometry_field("measure", p.geometry).dimension);
                }
                for (const auto & v : p.variables) columns.emplace_back(v, field_units::describe(v).dimension);
                p.out = open_csv(p.file, resume, columns);
            }
            if (!p.reductions.empty()) {
                const Dimension measure = geometry_field("measure", p.geometry).dimension;
                std::vector<std::pair<std::string, Dimension>> columns = {
                    {"step", dim::none}, {"t", dim::time}, {measure_name(p.geometry), measure}};
                for (const Reduction r : p.reductions) {
                    for (const auto & v : p.variables) {
                        const Dimension d = field_units::describe(v).dimension;
                        // Integrals take the units of the measure (sums over points)
                        columns.emplace_back(reduction_name(r) + "_" + v,
                                             r == Reduction::INTEGRAL ? d.times(measure) : d);
                    }
                }
                p.out_reductions = open_csv(p.reductions_file, resume, columns);
            }
#ifdef Mallard_HAS_HDF5
            if (p.write_samples && p.format == Format::HDF5 && output_files_enabled() &&
                !(resume && std::filesystem::exists(p.file))) {
                make_parent(p.file);
                const std::lock_guard<std::mutex> lock(h5::mutex());
                // Room for a provenance attribute of any size (annotate)
                h5::Handle fapl(H5Pcreate(H5P_FILE_ACCESS), H5Pclose, "H5Pcreate");
                h5::Handle fcpl(h5::file_creation(fapl, std::numeric_limits<size_t>::max()), H5Pclose, "H5Pcreate");
                h5::Handle file(H5Fcreate(p.file.c_str(), H5F_ACC_TRUNC, fcpl, fapl), H5Fclose, "creating " + p.file);
                {
                    h5::Handle root(H5Gopen2(file, "/", H5P_DEFAULT), H5Gclose, "opening /");
                    h5::write_int_attribute(root, "version", static_cast<int>(file_format::PROBE_OUTPUT.version));
                }
                const uint64_t n = p.geometry.nodes.size();
                std::vector<double> xyz;
                for (const auto & x : p.geometry.nodes) xyz.insert(xyz.end(), x.begin(), x.end());
                h5::write_rows(file, "coordinates", n, N_DIM, 0, n, xyz.data(), H5P_DEFAULT);
                h5::annotate(file, "coordinates", FieldDescription{dim::length, "sample coordinates", "", ""});
                if (!p.geometry.arc_length.empty()) {
                    h5::write_rows(file, "arc_length", n, 1, 0, n, p.geometry.arc_length.data(), H5P_DEFAULT);
                    h5::annotate(file, "arc_length", FieldDescription{dim::length, "arc length", "", ""});
                    std::vector<uint64_t> nodes, offsets = {0};
                    for (const auto & line : p.geometry.polylines) {
                        nodes.insert(nodes.end(), line.begin(), line.end());
                        offsets.push_back(nodes.size());
                    }
                    h5::Handle group(H5Gcreate2(file, "polylines", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT), H5Gclose,
                                     "creating polylines");
                    h5::write_rows(file, "polylines/nodes", nodes.size(), 1, 0, nodes.size(), nodes.data(),
                                   H5P_DEFAULT);
                    h5::write_rows(file, "polylines/offsets", offsets.size(), 1, 0, offsets.size(), offsets.data(),
                                   H5P_DEFAULT);
                }
                if (!p.geometry.normals.empty()) {
                    std::vector<double> normals;
                    for (const auto & x : p.geometry.normals) normals.insert(normals.end(), x.begin(), x.end());
                    h5::write_rows(file, "normals", n, N_DIM, 0, n, normals.data(), H5P_DEFAULT);
                    h5::annotate(file, "normals", geometry_field("normal", p.geometry));
                    if (!p.geometry.node_weights.empty()) {
                        h5::write_rows(file, "area", n, 1, 0, n, p.geometry.node_weights.data(), H5P_DEFAULT);
                        h5::annotate(file, "area", geometry_field("measure", p.geometry));
                    }
                }
                if (!p.geometry.polygon_offsets.empty()) {
                    const auto & g = p.geometry;
                    std::vector<double> xyz_v;
                    for (const auto & x : g.vertices) xyz_v.insert(xyz_v.end(), x.begin(), x.end());
                    std::vector<uint64_t> vertices(g.polygon_vertices.begin(), g.polygon_vertices.end());
                    h5::Handle group(H5Gcreate2(file, "polygons", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT), H5Gclose,
                                     "creating polygons");
                    if (!g.vertices.empty()) {
                        h5::write_rows(file, "vertices", g.vertices.size(), N_DIM, 0, g.vertices.size(), xyz_v.data(),
                                       H5P_DEFAULT);
                        h5::annotate(file, "vertices", FieldDescription{dim::length, "polygon vertex coordinates", "", ""});
                    }
                    h5::write_rows(file, "polygons/vertices", vertices.size(), 1, 0, vertices.size(), vertices.data(),
                                   H5P_DEFAULT);
                    h5::write_rows(file, "polygons/offsets", g.polygon_offsets.size(), 1, 0, g.polygon_offsets.size(),
                                   g.polygon_offsets.data(), H5P_DEFAULT);
                }
                auto extensible = [&](const std::string & name, hid_t dtype, uint64_t cols) {
                    const int rank = cols > 0 ? 2 : 1;
                    const hsize_t dims[2] = {0, cols}, max[2] = {H5S_UNLIMITED, cols};
                    const hsize_t chunk[2] = {cols > 0 ? hsize_t(1) : hsize_t(256), std::max<hsize_t>(cols, 1)};
                    h5::Handle space(H5Screate_simple(rank, dims, max), H5Sclose, "H5Screate_simple");
                    h5::Handle plist(H5Pcreate(H5P_DATASET_CREATE), H5Pclose, "H5Pcreate");
                    h5::check(H5Pset_chunk(plist, rank, chunk), "H5Pset_chunk");
                    h5::Handle lcpl(H5Pcreate(H5P_LINK_CREATE), H5Pclose, "H5Pcreate");
                    h5::check(H5Pset_create_intermediate_group(lcpl, 1), "H5Pset_create_intermediate_group");
                    h5::Handle dset(H5Dcreate2(file, name.c_str(), dtype, space, lcpl, plist, H5P_DEFAULT), H5Dclose,
                                    "creating " + name);
                };
                extensible("step", H5T_STD_U64LE, 0);
                extensible("time", H5T_IEEE_F64LE, 0);
                h5::annotate(file, "step", FieldDescription{dim::none, "time step", "", ""});
                h5::annotate(file, "time", FieldDescription{dim::time, "time", "", ""});
                for (const auto & v : p.variables) {
                    extensible("fields/" + v, H5T_IEEE_F64LE, n);
                    h5::annotate(file, "fields/" + v, field_units::describe(v));
                }
            }
#endif
        });
        probes.push_back(std::move(probe_ptr));
    }
}

std::string Probes::set_interval(const std::string & name, uint64_t steps) {
    for (auto & p : probes) {
        if (p->name != name) continue;
        if (p->interval == 0) return "is written every time_interval; set that instead";
        if (steps == 0) return "interval must be positive";
        p->interval = steps;
        return "";
    }
    return "no [[probes]] named " + name;
}

std::string Probes::set_time_interval(const std::string & name, rtype value, rtype t) {
    for (auto & p : probes) {
        if (p->name != name) continue;
        if (p->interval > 0) return "is written every interval steps; set that instead";
        if (!(value > 0.0_r)) return "time_interval must be positive";
        p->time_interval = value;
        p->n_written = static_cast<uint64_t>(std::floor(t / value + precision_tol(1e-9, 1e-5))) + 1;
        return "";
    }
    return "no [[probes]] named " + name;
}

void Probes::resume([[maybe_unused]] uint64_t step, rtype t) {
    const rtype tol = precision_tol(1e-9, 1e-5);
    for (auto & pp : probes) {
        Probe & p = *pp;
        if (p.time_interval > 0.0_r) {
            p.n_written = static_cast<uint64_t>(std::floor(t / p.time_interval + tol)) + 1;
        }
    }
    if (!output_files_enabled()) return;
    on_root(distributed, [&] {
        for (auto & pp : probes) {
            Probe & p = *pp;
            if (!p.write_samples) continue;
            if (p.format == Format::VTP && std::filesystem::exists(p.file)) {
                // Keep the series up to the restart time
                std::ifstream in(p.file);
                std::stringstream ss;
                ss << in.rdbuf();
                const std::string text = ss.str();
                const std::regex entry_re("timestep=\"([^\"]+)\" file=\"([^\"]+)\"");
                for (auto it = std::sregex_iterator(text.begin(), text.end(), entry_re); it != std::sregex_iterator();
                     ++it) {
                    const double t_entry = std::stod((*it)[1].str());
                    if (t_entry <= static_cast<double>(t) * (1.0 + 1e-12)) {
                        p.series.emplace_back(t_entry, (*it)[2].str());
                    }
                }
            }
#ifdef Mallard_HAS_HDF5
            if (p.format == Format::VTKHDF && std::filesystem::exists(p.file)) {
                // Continue after the snapshots up to the restart time
                const std::lock_guard<std::mutex> lock(h5::mutex());
                h5::Handle file(H5Fopen(p.file.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT), H5Fclose, "opening " + p.file);
                h5::Handle steps(H5Gopen2(file, "VTKHDF/Steps", H5P_DEFAULT), H5Gclose, "opening VTKHDF/Steps");
                const std::vector<double> times =
                    h5::read_rows<double>(steps, "Values", 0, h5::n_rows(steps, "Values"), 1, H5P_DEFAULT);
                while (p.vtkhdf_steps < times.size() && times[p.vtkhdf_steps] <= static_cast<double>(t) * (1.0 + 1e-12)) {
                    p.vtkhdf_steps++;
                }
            }
            if (p.format == Format::HDF5 && std::filesystem::exists(p.file)) {
                // Drop the rows after the restart step
                const std::lock_guard<std::mutex> lock(h5::mutex());
                h5::Handle file(H5Fopen(p.file.c_str(), H5F_ACC_RDWR, H5P_DEFAULT), H5Fclose, "opening " + p.file);
                {
                    h5::Handle root(H5Gopen2(file, "/", H5P_DEFAULT), H5Gclose, "opening /");
                    file_format::check_version(file_format::PROBE_OUTPUT, p.file,
                                               h5::read_int_attribute(root, "version"));
                }
                const uint64_t n_rows = h5::n_rows(file, "step");
                const std::vector<uint64_t> steps = h5::read_rows<uint64_t>(file, "step", 0, n_rows, 1, H5P_DEFAULT);
                uint64_t keep = 0;
                while (keep < steps.size() && steps[keep] <= step) keep++;
                std::vector<std::string> names = {"step", "time"};
                for (const auto & v : p.variables) names.push_back("fields/" + v);
                for (const auto & name : names) {
                    h5::Handle dset(H5Dopen2(file, name.c_str(), H5P_DEFAULT), H5Dclose, "opening " + name);
                    hsize_t dims[2] = {keep, p.geometry.nodes.size()};
                    h5::check(H5Dset_extent(dset, dims), "H5Dset_extent");
                }
            }
#endif
        }
    });
}

bool Probes::Probe::due(uint64_t step, rtype t) const {
    if (step == step_last) return false;
    if (interval > 0) return step % interval == 0;
    return t >= static_cast<rtype>(n_written) * time_interval - precision_tol(1e-9, 1e-5) * time_interval;
}

bool Probes::due(uint64_t step, rtype t) const {
    return std::any_of(probes.begin(), probes.end(), [&](const auto & p) { return p->due(step, t); });
}

bool Probes::point_values_due(uint64_t step, rtype t) const {
    return std::any_of(probes.begin(), probes.end(),
                       [&](const auto & p) { return p->sampling == Sampling::POINT && p->due(step, t); });
}

bool Probes::gradients_due(uint64_t step, rtype t) const {
    return std::any_of(probes.begin(), probes.end(), [&](const auto & p) { return p->gradients && p->due(step, t); });
}

bool Probes::transport_due(uint64_t step, rtype t) const {
    return std::any_of(probes.begin(), probes.end(), [&](const auto & p) { return p->transport && p->due(step, t); });
}

bool Probes::any_gradients() const {
    return std::any_of(probes.begin(), probes.end(),
                       [](const auto & p) { return p->gradients || p->sampling == Sampling::POINT; });
}

bool Probes::any_point_values() const {
    return std::any_of(probes.begin(), probes.end(), [](const auto & p) { return p->sampling == Sampling::POINT; });
}

rtype Probes::next_time() const {
    rtype t = std::numeric_limits<rtype>::infinity();
    for (const auto & p : probes) {
        if (p->time_interval > 0.0_r) t = std::min(t, static_cast<rtype>(p->n_written) * p->time_interval);
    }
    return t;
}

void Probes::write(uint64_t step, rtype t, const ProbeFields & fields) {
    for (auto & pp : probes) {
        Probe & p = *pp;
        if (!p.due(step, t)) continue;
        p.step_last = step;
        if (p.time_interval > 0.0_r) {
            while (static_cast<rtype>(p.n_written) * p.time_interval <=
                   t + precision_tol(1e-9, 1e-5) * p.time_interval) {
                p.n_written++;
            }
        }
        const bool extrema = std::any_of(p.reductions.begin(), p.reductions.end(),
                                         [](Reduction r) { return r == Reduction::MIN || r == Reduction::MAX; });
        std::vector<double> nodes, quad;
        // Collective on every rank (the geometry lives on rank 0 for some shapes)
        const bool own_quadrature = p.quadrature.n_global == 0;
        if (!p.from_region && (p.write_samples || extrema || own_quadrature)) {
            p.nodes.evaluate(p.sampling, p.codes, fields, p.reference, p.gradients);
            nodes = p.nodes.gather();
        }
        if (!p.reductions.empty() && !own_quadrature) {
            p.quadrature.evaluate(p.sampling, p.codes, fields, p.reference, p.gradients);
            quad = p.quadrature.gather();
        }
        if (!comm::is_root()) continue;
        if (p.write_samples) {
            if (p.format == Format::CSV) {
                std::ostream & out = *p.out;
                const size_t n_cols = p.n_evaluated();
                for (size_t i = 0; i < p.geometry.nodes.size(); i++) {
                    out << step << "," << std::setprecision(std::numeric_limits<double>::max_digits10) << double(t)
                        << "," << i;
                    for (size_t d = 0; d < N_DIM; d++) out << "," << p.geometry.nodes[i][d];
                    if (!p.geometry.node_weights.empty()) {
                        if (!p.geometry.arc_length.empty()) out << "," << p.geometry.arc_length[i];
                        for (size_t d = 0; d < N_DIM; d++) out << "," << p.geometry.normals[i][d];
                        out << "," << p.geometry.node_weights[i];
                    }
                    out << std::setprecision(std::numeric_limits<rtype>::max_digits10);
                    for (size_t v = 0; v < p.variables.size(); v++) {
                        out << "," << static_cast<rtype>(nodes[i * n_cols + v]);
                    }
                    out << "\n";
                }
                out.flush();
            } else if (p.format == Format::HDF5) {
                p.write_hdf5(step, t, nodes);
            } else if (p.format == Format::VTKHDF) {
                p.write_vtkhdf(t, nodes);
            } else {
                p.write_vtp(step, t, nodes);
            }
        }
        if (!p.reductions.empty()) p.write_reductions(step, t, nodes, quad);
    }
}

void Probes::Probe::write_hdf5([[maybe_unused]] uint64_t step, [[maybe_unused]] rtype t,
                               [[maybe_unused]] const std::vector<double> & values) const {
#ifdef Mallard_HAS_HDF5
    const std::lock_guard<std::mutex> lock(h5::mutex());
    h5::Handle h5file(H5Fopen(file.c_str(), H5F_ACC_RDWR, H5P_DEFAULT), H5Fclose, "opening " + file);
    const uint64_t n = geometry.nodes.size();
    const size_t n_cols = n_evaluated();
    auto append = [&](const std::string & dataset, hid_t type, const void * data, uint64_t cols) {
        h5::Handle dset(H5Dopen2(h5file, dataset.c_str(), H5P_DEFAULT), H5Dclose, "opening " + dataset);
        hsize_t dims[2] = {0, 0};
        {
            h5::Handle space(H5Dget_space(dset), H5Sclose, "H5Dget_space");
            H5Sget_simple_extent_dims(space, dims, nullptr);
        }
        const hsize_t row = dims[0];
        dims[0] = row + 1;
        h5::check(H5Dset_extent(dset, dims), "H5Dset_extent");
        h5::Handle space(H5Dget_space(dset), H5Sclose, "H5Dget_space");
        const int rank = cols > 0 ? 2 : 1;
        const hsize_t start[2] = {row, 0}, count[2] = {1, std::max<hsize_t>(cols, 1)};
        h5::check(H5Sselect_hyperslab(space, H5S_SELECT_SET, start, nullptr, count, nullptr), "H5Sselect_hyperslab");
        h5::Handle memory(H5Screate_simple(rank, count, nullptr), H5Sclose, "H5Screate_simple");
        h5::check(H5Dwrite(dset, type, memory, space, H5P_DEFAULT, data), "writing " + dataset);
    };
    const double time = static_cast<double>(t);
    append("step", H5T_NATIVE_UINT64, &step, 0);
    append("time", H5T_NATIVE_DOUBLE, &time, 0);
    std::vector<double> column(n);
    for (size_t v = 0; v < variables.size(); v++) {
        for (uint64_t i = 0; i < n; i++) column[i] = values[i * n_cols + v];
        append("fields/" + variables[v], H5T_NATIVE_DOUBLE, column.data(), n);
    }
#endif
}

void Probes::Probe::write_vtkhdf([[maybe_unused]] rtype t, [[maybe_unused]] const std::vector<double> & values) {
#ifdef Mallard_HAS_HDF5
    const std::lock_guard<std::mutex> lock(h5::mutex());
    const uint64_t k = vtkhdf_steps, n = geometry.nodes.size();
    const size_t n_cols = n_evaluated();
    const PolyLayout layout = poly_layout(geometry);
    const std::string data = layout.cell_data ? "CellData" : "PointData";
    const std::vector<GeometryArray> fixed = geometry_arrays(geometry);
    if (k == 0) {
        make_parent(file);
        const std::string & provenance = provenance::json();
        h5::Handle fapl(H5Pcreate(H5P_FILE_ACCESS), H5Pclose, "H5Pcreate");
        h5::Handle fcpl(h5::file_creation(fapl, provenance.size()), H5Pclose, "H5Pcreate");
        h5::Handle created(H5Fcreate(file.c_str(), H5F_ACC_TRUNC, fcpl, fapl), H5Fclose, "creating " + file);
        {
            h5::Handle root(H5Gopen2(created, "/", H5P_DEFAULT), H5Gclose, "opening /");
            h5::write_strings_attribute(root, "provenance", {provenance});
            h5::write_int_attribute(root, "version", static_cast<int>(file_format::PROBE_OUTPUT.version));
        }
        h5::Handle top(H5Gcreate2(created, "VTKHDF", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT), H5Gclose, "VTKHDF");
        h5::write_array_attribute(top, "Version", std::vector<int64_t>{2, 0});
        h5::write_string_scalar_attribute(top, "Type", "PolyData");
        const uint64_t n_points = layout.n_points();
        const int64_t n_points_i = static_cast<int64_t>(n_points);
        h5::write_rows<int64_t>(top, "NumberOfPoints", 1, 1, 0, 1, &n_points_i, H5P_DEFAULT);
        h5::write_rows(top, "Points", n_points, 3, 0, n_points, layout.xyz.data(), H5P_DEFAULT);
        const std::string mine = layout.cells == std::string("Verts")   ? "Vertices"
                                 : layout.cells == std::string("Lines") ? "Lines"
                                                                        : "Polygons";
        for (const char * kind : {"Vertices", "Lines", "Polygons", "Strips"}) {
            const std::vector<int64_t> none, zero = {0};
            const auto & o = kind == mine ? layout.offsets : zero;
            const auto & c = kind == mine ? layout.connectivity : none;
            const int64_t counts[2] = {static_cast<int64_t>(o.size() - 1), static_cast<int64_t>(c.size())};
            h5::Handle group(H5Gcreate2(top, kind, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT), H5Gclose, kind);
            h5::write_rows<int64_t>(group, "NumberOfCells", 1, 1, 0, 1, &counts[0], H5P_DEFAULT);
            h5::write_rows<int64_t>(group, "NumberOfConnectivityIds", 1, 1, 0, 1, &counts[1], H5P_DEFAULT);
            h5::write_rows(group, "Offsets", o.size(), 1, 0, o.size(), o.data(), H5P_DEFAULT);
            h5::write_rows(group, "Connectivity", c.size(), 1, 0, c.size(), c.data(), H5P_DEFAULT);
        }
        for (const char * group_name : {"CellData", "PointData", "FieldData"}) {
            h5::Handle group(H5Gcreate2(top, group_name, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT), H5Gclose, group_name);
        }
        h5::Handle samples(H5Gopen2(top, data.c_str(), H5P_DEFAULT), H5Gclose, "opening " + data);
        for (const auto & a : fixed) {
            const uint64_t rows = a.values.size() / static_cast<size_t>(a.components);
            h5::write_rows(samples, a.name.c_str(), rows, a.components, 0, rows, a.values.data(),
                           H5P_DEFAULT);
            h5::annotate(samples, a.name, geometry_field(a.name, geometry));
        }
        h5::annotate(top, "Points", FieldDescription{dim::length, "sample coordinates", "", ""});
        h5::Handle steps(H5Gcreate2(top, "Steps", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT), H5Gclose, "Steps");
        for (const char * group_name : {"CellDataOffsets", "PointDataOffsets", "FieldDataOffsets"}) {
            h5::Handle group(H5Gcreate2(steps, group_name, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT), H5Gclose, group_name);
        }
    }
    h5::Handle h5file(H5Fopen(file.c_str(), H5F_ACC_RDWR, H5P_DEFAULT), H5Fclose, "opening " + file);
    h5::Handle samples(H5Gopen2(h5file, ("VTKHDF/" + data).c_str(), H5P_DEFAULT), H5Gclose, "opening " + data);
    std::vector<double> column(n);
    for (size_t v = 0; v < variables.size(); v++) {
        for (uint64_t i = 0; i < n; i++) column[i] = values[i * n_cols + v];
        h5::put_rows(samples, variables[v].c_str(), (k + 1) * n, 1, k * n, n, column.data(), H5P_DEFAULT);
        if (k == 0) h5::annotate(samples, variables[v], field_units::describe(variables[v]));
    }
    // The geometry is the same at every snapshot: its offsets all stay 0
    h5::Handle steps(H5Gopen2(h5file, "VTKHDF/Steps", H5P_DEFAULT), H5Gclose, "opening Steps");
    h5::set_attribute<int64_t>(steps, "NSteps", static_cast<int64_t>(k + 1));
    const double time = static_cast<double>(t);
    const int64_t zeros[4] = {0, 0, 0, 0}, one = 1, offset = static_cast<int64_t>(k * n);
    h5::put_rows(steps, "Values", k + 1, 1, k, 1, &time, H5P_DEFAULT);
    if (k == 0) h5::annotate(steps, "Values", FieldDescription{dim::time, "time", "time", ""});
    h5::put_rows(steps, "PartOffsets", k + 1, 1, k, 1, zeros, H5P_DEFAULT);
    h5::put_rows(steps, "NumberOfParts", k + 1, 1, k, 1, &one, H5P_DEFAULT);
    h5::put_rows(steps, "PointOffsets", k + 1, 1, k, 1, zeros, H5P_DEFAULT);
    h5::put_rows(steps, "CellOffsets", k + 1, 4, k, 1, zeros, H5P_DEFAULT);
    h5::put_rows(steps, "ConnectivityIdOffsets", k + 1, 4, k, 1, zeros, H5P_DEFAULT);
    h5::Handle data_offsets(H5Gopen2(steps, (data + "Offsets").c_str(), H5P_DEFAULT), H5Gclose,
                            "opening " + data + "Offsets");
    for (const auto & v : variables) h5::put_rows(data_offsets, v.c_str(), k + 1, 1, k, 1, &offset, H5P_DEFAULT);
    for (const auto & a : fixed) h5::put_rows(data_offsets, a.name.c_str(), k + 1, 1, k, 1, zeros, H5P_DEFAULT);
    vtkhdf_steps = k + 1;
#endif
}

void Probes::Probe::write_vtp(uint64_t step, rtype t, const std::vector<double> & values) {
    const std::filesystem::path pvd(file);
    std::ostringstream piece;
    piece << pvd.stem().string() << "_" << std::setw(6) << std::setfill('0') << step << ".vtp";
    write_vtp_file((pvd.parent_path() / piece.str()).string(), geometry, variables, values, n_evaluated());
    series.emplace_back(static_cast<double>(t), piece.str());
    write_pvd_file(file, series);
}

void Probes::Probe::write_reductions(uint64_t step, rtype t, const std::vector<double> & node_values,
                                     const std::vector<double> & quad) const {
    // Point sets and boundaries reduce their samples; curves their quadrature points
    const bool points = quadrature.n_global == 0;
    const std::vector<double> & q = points ? node_values : quad;
    const size_t n_cols = n_evaluated();
    const size_t n_q = points ? geometry.nodes.size() : geometry.quad_weights.size();
    auto weight = [&](size_t i) {
        return points ? (geometry.node_weights.empty() ? 1.0 : geometry.node_weights[i]) : geometry.quad_weights[i];
    };
    double measure = 0.0;
    for (size_t i = 0; i < n_q; i++) measure += weight(i);
    std::ostream & row = *out_reductions;
    row << step << "," << std::setprecision(std::numeric_limits<double>::max_digits10) << double(t) << "," << measure;
    for (const Reduction r : reductions) {
        for (size_t v = 0; v < variables.size(); v++) {
            double sum = 0.0, sum_rho = 0.0;
            double lo = std::numeric_limits<double>::infinity(), hi = -lo;
            for (size_t i = 0; i < n_q; i++) {
                const double w = weight(i);
                const double value = q[i * n_cols + v];
                switch (r) {
                    case Reduction::INTEGRAL:
                    case Reduction::MEAN: sum += w * value; break;
                    case Reduction::FAVRE:
                        sum += w * q[i * n_cols + rho_column] * value;
                        sum_rho += w * q[i * n_cols + rho_column];
                        break;
                    case Reduction::RMS: sum += w * value * value; break;
                    case Reduction::MIN: lo = std::min(lo, value); break;
                    case Reduction::MAX: hi = std::max(hi, value); break;
                }
            }
            if (r == Reduction::MIN || r == Reduction::MAX) {
                for (size_t i = 0; i < geometry.nodes.size(); i++) {
                    lo = std::min(lo, node_values[i * n_cols + v]);
                    hi = std::max(hi, node_values[i * n_cols + v]);
                }
            }
            double result = 0.0;
            switch (r) {
                case Reduction::INTEGRAL: result = sum; break;
                case Reduction::MEAN: result = sum / measure; break;
                case Reduction::FAVRE: result = sum / sum_rho; break;
                case Reduction::RMS: result = std::sqrt(sum / measure); break;
                case Reduction::MIN: result = lo; break;
                case Reduction::MAX: result = hi; break;
            }
            row << "," << result;
        }
    }
    row << "\n";
    row.flush();
}

std::vector<std::pair<std::string, std::string>> Probes::summary() const {
    std::vector<std::pair<std::string, std::string>> items;
    for (const auto & pp : probes) {
        const Probe & p = *pp;
        std::string vars;
        for (const auto & v : p.variables) vars += (vars.empty() ? "" : ", ") + v;
        const size_t n = p.from_region ? p.quadrature.n_global : p.geometry.nodes.size();
        const char * unit = p.from_region ? (n == 1 ? " cell" : " cells") : (n == 1 ? " point" : " points");
        std::string text = (p.write_samples ? p.file : p.reductions_file) + ", " + p.geometry.type + (p.from_region ? " over " : " of ") +
                           logging::count(n) + unit +
                           (p.interval > 0 ? " every " + logging::count(p.interval) + " steps"
                                           : " every t = " + logging::real(double(p.time_interval))) +
                           (p.sampling == Sampling::POINT ? " (point values): " : ": ") + vars;
        if (!p.reductions.empty()) {
            std::string reds;
            for (const Reduction r : p.reductions) reds += (reds.empty() ? "" : ", ") + reduction_name(r);
            text += "; " + reds + (p.write_samples ? " to " + p.reductions_file : "");
        }
        items.emplace_back("probe " + p.name, text);
    }
    return items;
}

void Probes::annotate(const std::string & csv_line) {
    if (!comm::is_root() || !output_files_enabled()) return;
    for (auto & p : probes) {
        if (p->out) *p->out << csv_line << std::flush;
        if (p->out_reductions) *p->out_reductions << csv_line << std::flush;
#ifdef Mallard_HAS_HDF5
        if (p->write_samples && p->format == Format::HDF5 && std::filesystem::exists(p->file)) {
            const std::lock_guard<std::mutex> lock(h5::mutex());
            h5::Handle file(H5Fopen(p->file.c_str(), H5F_ACC_RDWR, H5P_DEFAULT), H5Fclose, "opening " + p->file);
            h5::Handle root(H5Gopen2(file, "/", H5P_DEFAULT), H5Gclose, "opening /");
            if (H5Aexists(root, "provenance") > 0) h5::check(H5Adelete(root, "provenance"), "H5Adelete");
            h5::write_strings_attribute(root, "provenance", {provenance::json()});
        }
#endif
    }
}

std::vector<std::string> Probes::files() const {
    std::vector<std::string> files;
    for (const auto & p : probes) {
        if (p->write_samples) files.push_back(p->file);
        if (!p->reductions.empty()) files.push_back(p->reductions_file);
    }
    return files;
}

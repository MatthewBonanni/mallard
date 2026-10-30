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

std::string reduction_name(Reduction r) {
    for (const auto & [name, value] : REDUCTIONS) {
        if (value == r) return name;
    }
    return "";
}

/** @brief The variables a [[probes]] entry lists, with U for its components and Y_* for every species. */
std::vector<std::string> expand_variables(const std::vector<std::string> & listed,
                                          const std::vector<std::string> & species_names, const std::string & key) {
    std::vector<std::string> names;
    for (const auto & v : listed) {
        if (v == "U") {
            for (size_t d = 0; d < N_DIM; d++) names.push_back(std::string("U_") + "XYZ"[d]);
        } else if (v == "Y_*") {
            if (species_names.empty()) throw InputError(key + ".variables: Y_* needs a gas mixture.");
            for (const auto & s : species_names) names.push_back("Y_" + s);
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

/**
 * @brief A VTK XML PolyData file with its arrays appended raw: the points
 *        of the samples, as vertices (points) or polylines (curves), and
 *        one point-data array per variable.
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
    const size_t n = g.nodes.size();
    std::vector<double> xyz(3 * n, 0.0);
    for (size_t p = 0; p < n; p++) {
        for (size_t d = 0; d < N_DIM; d++) xyz[3 * p + d] = g.nodes[p][d];
    }
    std::vector<int64_t> connectivity, offsets;
    if (g.dim == 0) {
        for (size_t p = 0; p < n; p++) {
            connectivity.push_back(static_cast<int64_t>(p));
            offsets.push_back(static_cast<int64_t>(p + 1));
        }
    } else {
        for (const auto & line : g.polylines) {
            for (uint32_t p : line) connectivity.push_back(p);
            offsets.push_back(static_cast<int64_t>(connectivity.size()));
        }
    }
    std::vector<Array> point_data;
    for (size_t v = 0; v < names.size(); v++) {
        std::vector<double> column(n);
        for (size_t p = 0; p < n; p++) column[p] = values[p * n_columns + v];
        point_data.push_back(doubles(names[v], 1, column));
    }
    if (!g.arc_length.empty()) point_data.push_back(doubles("arc_length", 1, g.arc_length));
    std::vector<Array> arrays = {doubles("Points", 3, xyz), integers("connectivity", connectivity),
                                 integers("offsets", offsets)};
    for (auto & a : point_data) arrays.push_back(std::move(a));

    make_parent(file);
    std::ofstream out(file, std::ios::binary | std::ios::trunc);
    if (!out.good()) throw cannot_write("the probe file", file);
    uint64_t offset = 0;
    auto data_array = [&](const Array & a) {
        out << "        <DataArray type=\"" << a.type << "\" Name=\"" << a.name << "\" NumberOfComponents=\""
            << a.components << "\" format=\"appended\" offset=\"" << offset << "\"/>\n";
        offset += sizeof(uint64_t) + a.bytes.size();
    };
    const char * cells = g.dim == 0 ? "Verts" : "Lines";
    std::vector<std::string> point_names;
    std::vector<FieldDescription> descriptions;
    for (size_t a = 3; a < arrays.size(); a++) {
        point_names.push_back(arrays[a].name);
        descriptions.push_back(arrays[a].name == "arc_length" ? FieldDescription{dim::length, "arc length", "", ""}
                                                              : field_units::describe(arrays[a].name));
    }
    // Field data belongs to the data set, before the piece, for VTK to read it
    out << "<?xml version=\"1.0\"?>\n" << provenance::xml_comment() << "\n<VTKFile type=\"PolyData\" version=\"1.0\" byte_order=\"LittleEndian\" "
           "header_type=\"UInt64\">\n  <PolyData>\n";
    out << "    <FieldData>\n      <DataArray type=\"Int32\" Name=\"MALLARD_FORMAT_VERSION\" NumberOfTuples=\"1\" "
           "format=\"ascii\">"
        << file_format::PROBE_OUTPUT.version << "</DataArray>\n"
        << field_units::vtk_field_data(point_names, descriptions, "      ") << "    </FieldData>\n";
    out << "    <Piece NumberOfPoints=\""
        << n << "\" NumberOfVerts=\"" << (g.dim == 0 ? offsets.size() : 0) << "\" NumberOfLines=\""
        << (g.dim == 0 ? 0 : offsets.size()) << "\" NumberOfStrips=\"0\" NumberOfPolys=\"0\">\n";
    out << "      <Points>\n";
    data_array(arrays[0]);
    out << "      </Points>\n      <" << cells << ">\n";
    data_array(arrays[1]);
    data_array(arrays[2]);
    out << "      </" << cells << ">\n      <PointData>\n";
    for (size_t a = 3; a < arrays.size(); a++) data_array(arrays[a]);
    out << "      </PointData>\n    </Piece>\n  </PolyData>\n  <AppendedData encoding=\"raw\">\n   _";
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
    p.geometry.type = "region " + r.summary();
    for (const size_t i : by_id) p.geometry.quad_weights.push_back(weights[i]);
}

void Probes::Samples::locate(const Mesh & mesh, const std::vector<probe::Point> & points, uint32_t n_variables) {
    n_global = points.size();
    std::vector<uint8_t> contained;
    const std::vector<int32_t> located = locate_points(mesh, points, &contained);
    std::vector<uint32_t> h_cells, h_inside, h_inside_cells;
    std::vector<rtype> h_offsets;
    for (uint64_t p = 0; p < points.size(); p++) {
        if (located[p] < 0) continue;
        const uint32_t c = static_cast<uint32_t>(located[p]);
        if (contained[p]) {
            h_inside.push_back(static_cast<uint32_t>(ids.size()));
            h_inside_cells.push_back(c);
            for (size_t d = 0; d < N_DIM; d++) {
                h_offsets.push_back(static_cast<rtype>(points[p][d] - static_cast<double>(mesh.h_cell_coords(c, d))));
            }
        }
        ids.push_back(p);
        h_cells.push_back(c);
    }
    cells = to_device(h_cells, "probe_cells");
    inside = to_device(h_inside, "probe_inside");
    inside_cells = to_device(h_inside_cells, "probe_inside_cells");
    inside_offsets = Kokkos::View<rtype *[N_DIM]>("probe_offsets", h_inside.size());
    auto h_off = Kokkos::create_mirror_view(inside_offsets);
    for (size_t i = 0; i < h_inside.size(); i++) {
        for (size_t d = 0; d < N_DIM; d++) h_off(i, d) = h_offsets[i * N_DIM + d];
    }
    Kokkos::deep_copy(inside_offsets, h_off);
    std::vector<uint32_t> unique = h_inside_cells;
    std::sort(unique.begin(), unique.end());
    unique.erase(std::unique(unique.begin(), unique.end()), unique.end());
    unique_cells = to_device(unique, "probe_unique_cells");
    inside_W = Kokkos::View<rtype *[N_CONSERVATIVE]>("probe_inside_W", h_inside.size());
    order(mesh, n_variables, h_cells.size());
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
    order(mesh, n_variables, owned_cells.size());
}

void Probes::Samples::order(const Mesh & mesh, uint32_t n_variables, size_t n_cells) {
    W = Kokkos::View<rtype *[N_CONSERVATIVE]>("probe_W", n_cells);
    values = Kokkos::View<rtype **, Kokkos::LayoutRight>("probe_values", n_cells, n_variables);
    distributed = mesh.n_global_cells > 0;
    const std::vector<uint64_t> all = distributed ? comm::gatherv(ids) : ids;
    if (comm::is_root() || !distributed) {
        if (all.size() != n_global) throw std::logic_error("probe samples held by no rank or by several");
        gathered_row.assign(n_global, 0);
        for (uint64_t r = 0; r < all.size(); r++) gathered_row[all[r]] = r;
    }
}

void Probes::Samples::evaluate(Sampling sampling, Kokkos::View<int32_t *> codes, const ProbeFields & fields) {
    const uint32_t n = static_cast<uint32_t>(cells.extent(0));
    const uint32_t n_vars = static_cast<uint32_t>(codes.extent(0));
    const auto cell = cells;
    const auto code = codes;
    const auto out = values;
    const CellSampler sampler = fields.cells;
    if (sampling == Sampling::CELL) {
        Kokkos::parallel_for("probe_cell_values", n, KOKKOS_LAMBDA(const uint32_t i) {
            for (uint32_t v = 0; v < n_vars; v++) out(i, v) = sampler(cell(i), code(v));
        });
        return;
    }
    constexpr uint8_t E = N_DIM + 1;
    const auto gradient = fields.gradient;
    const auto unique = unique_cells;
    Kokkos::parallel_for("probe_gradients", unique.extent(0), KOKKOS_LAMBDA(const uint32_t i) { gradient(unique(i)); });
    const auto Wc = fields.W;
    const auto G = fields.gradients;
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
    const auto Ws = W;
    const auto in = inside;
    Kokkos::parallel_for("probe_cell_W", n, KOKKOS_LAMBDA(const uint32_t s) {
        FOR_I_CONSERVATIVE Ws(s, i) = Wc(cell(s), i);
    });
    Kokkos::parallel_for("probe_point_W", in.extent(0), KOKKOS_LAMBDA(const uint32_t j) {
        FOR_I_CONSERVATIVE Ws(in(j), i) = in_W(j, i);
    });
    const rtype gamma = fields.gamma;
    const auto cell_gamma = fields.cell_gamma;
    const bool mixture = cell_gamma.extent(0) > 0;
    Kokkos::parallel_for("probe_point_values", n, KOKKOS_LAMBDA(const uint32_t i) {
        const uint32_t c = cell(i);
        const rtype g = mixture ? cell_gamma(c) : gamma;
        // Temperature and enthalpy follow p / rho at the cell's composition
        const rtype pv = Ws(i, E) / Ws(i, 0), pv_c = Wc(c, E) / Wc(c, 0);
        for (uint32_t v = 0; v < n_vars; v++) {
            const int32_t k = code(v);
            rtype value;
            if (k == CellSampler::RHO) {
                value = Ws(i, 0);
            } else if (k < CellSampler::FIRST_PRIMITIVE + N_DIM + 1) {
                value = Ws(i, 1 + (k - CellSampler::FIRST_PRIMITIVE));
            } else if (k == CellSampler::FIRST_PRIMITIVE + N_DIM + 1) {
                value = sampler.P(c, N_DIM + 1) * (pv / pv_c);
            } else if (k == CellSampler::FIRST_PRIMITIVE + N_DIM + 2) {
                value = sampler.P(c, N_DIM + 2) + g / (g - 1.0_r) * (pv - pv_c);
            } else {
                value = sampler(c, k);
            }
            out(i, v) = value;
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
                  const std::vector<std::string> & species_names, bool resume) {
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
        for (const char * k : {"type", "shape", "sampling", "format", "file", "samples"}) {
            if (p.from_region && entry.contains(k)) {
                throw InputError(key + "." + k + ": a region probe writes only its reductions, to reductions_file; " +
                                 k + " applies to points and curves.");
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
        for (const auto & v : p.variables) h_codes.push_back(CellSampler::code(v, species_names, key + ".variables"));
        const auto rho = std::find(h_codes.begin(), h_codes.end(), CellSampler::RHO);
        p.rho_column = static_cast<uint32_t>(rho - h_codes.begin());
        if (rho == h_codes.end() &&
            std::find(p.reductions.begin(), p.reductions.end(), Reduction::FAVRE) != p.reductions.end()) {
            h_codes.push_back(CellSampler::RHO);
        }
        p.codes = to_device(h_codes, "probe_codes");

        const uint32_t n_evaluated = static_cast<uint32_t>(h_codes.size());
        if (p.from_region) {
            const region::Region & r = regions.get(entry.at("region"), key + ".region");
            region::require_full_dimension(r, key + ".region");
            select_region(p, mesh, r, regions.fractions(entry.at("region"), key + ".region"), n_evaluated);
            if (p.quadrature.n_global == 0) throw InputError(key + ".region contains no part of any cell.");
        } else {
            p.geometry = probe::make_geometry(entry, key);
            p.nodes.locate(mesh, p.geometry.nodes, n_evaluated);
            if (!p.reductions.empty() && p.geometry.dim > 0) {
                p.quadrature.locate(mesh, p.geometry.quad_points, n_evaluated);
            }
        }

        on_root(distributed, [&] {
            if (p.write_samples && p.format == Format::CSV) {
                std::vector<std::pair<std::string, Dimension>> columns = {
                    {"step", dim::none}, {"t", dim::time}, {"point", dim::none}};
                for (size_t d = 0; d < N_DIM; d++) columns.emplace_back(std::string(1, "xyz"[d]), dim::length);
                for (const auto & v : p.variables) columns.emplace_back(v, field_units::describe(v).dimension);
                p.out = open_csv(p.file, resume, columns);
            }
            if (!p.reductions.empty()) {
                const bool curve = p.geometry.dim > 0;
                // Regions: sum f V, per unit depth in planar 2D runs, per radian in axisymmetric ones
                const bool volume = N_DIM == 3 || mesh.axisymmetric;
                const Dimension cell_measure = mesh.axisymmetric ? dim::of(0, 3, 0, 0, 0, -1) : dim::of(0, N_DIM, 0);
                const Dimension measure = p.from_region ? cell_measure : curve ? dim::length : dim::none;
                std::vector<std::pair<std::string, Dimension>> columns = {
                    {"step", dim::none},
                    {"t", dim::time},
                    {p.from_region ? (volume ? "volume" : "area") : curve ? "length" : "count", measure}};
                for (const Reduction r : p.reductions) {
                    for (const auto & v : p.variables) {
                        const Dimension d = field_units::describe(v).dimension;
                        // Integrals carry the measure; over points, sums
                        columns.emplace_back(reduction_name(r) + "_" + v, r == Reduction::INTEGRAL ? d.times(measure) : d);
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
                auto extensible = [&](const std::string & name, hid_t type, uint64_t cols) {
                    const int rank = cols > 0 ? 2 : 1;
                    const hsize_t dims[2] = {0, cols}, max[2] = {H5S_UNLIMITED, cols};
                    const hsize_t chunk[2] = {cols > 0 ? hsize_t(1) : hsize_t(256), std::max<hsize_t>(cols, 1)};
                    h5::Handle space(H5Screate_simple(rank, dims, max), H5Sclose, "H5Screate_simple");
                    h5::Handle plist(H5Pcreate(H5P_DATASET_CREATE), H5Pclose, "H5Pcreate");
                    h5::check(H5Pset_chunk(plist, rank, chunk), "H5Pset_chunk");
                    h5::Handle lcpl(H5Pcreate(H5P_LINK_CREATE), H5Pclose, "H5Pcreate");
                    h5::check(H5Pset_create_intermediate_group(lcpl, 1), "H5Pset_create_intermediate_group");
                    h5::Handle dset(H5Dcreate2(file, name.c_str(), type, space, lcpl, plist, H5P_DEFAULT), H5Dclose,
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
        if (!p.from_region && (p.write_samples || extrema || p.geometry.dim == 0)) {
            p.nodes.evaluate(p.sampling, p.codes, fields);
            nodes = p.nodes.gather();
        }
        if (!p.reductions.empty() && p.geometry.dim > 0) {
            p.quadrature.evaluate(p.sampling, p.codes, fields);
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
    const bool curves = geometry.dim > 0;
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
        const int64_t n_points = static_cast<int64_t>(n);
        h5::write_rows<int64_t>(top, "NumberOfPoints", 1, 1, 0, 1, &n_points, H5P_DEFAULT);
        std::vector<double> xyz(3 * n, 0.0);
        for (size_t i = 0; i < n; i++) {
            for (size_t d = 0; d < N_DIM; d++) xyz[3 * i + d] = geometry.nodes[i][d];
        }
        h5::write_rows(top, "Points", n, 3, 0, n, xyz.data(), H5P_DEFAULT);
        // Point sets as vertices, curves as polylines
        std::vector<int64_t> connectivity, offsets = {0};
        if (!curves) {
            for (size_t i = 0; i < n; i++) {
                connectivity.push_back(static_cast<int64_t>(i));
                offsets.push_back(static_cast<int64_t>(i + 1));
            }
        } else {
            for (const auto & line : geometry.polylines) {
                for (uint32_t i : line) connectivity.push_back(i);
                offsets.push_back(static_cast<int64_t>(connectivity.size()));
            }
        }
        for (const char * kind : {"Vertices", "Lines", "Polygons", "Strips"}) {
            const bool mine = std::string(kind) == (curves ? "Lines" : "Vertices");
            const std::vector<int64_t> none, zero = {0};
            const auto & o = mine ? offsets : zero;
            const auto & c = mine ? connectivity : none;
            const int64_t counts[2] = {static_cast<int64_t>(o.size() - 1), static_cast<int64_t>(c.size())};
            h5::Handle group(H5Gcreate2(top, kind, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT), H5Gclose, kind);
            h5::write_rows<int64_t>(group, "NumberOfCells", 1, 1, 0, 1, &counts[0], H5P_DEFAULT);
            h5::write_rows<int64_t>(group, "NumberOfConnectivityIds", 1, 1, 0, 1, &counts[1], H5P_DEFAULT);
            h5::write_rows(group, "Offsets", o.size(), 1, 0, o.size(), o.data(), H5P_DEFAULT);
            h5::write_rows(group, "Connectivity", c.size(), 1, 0, c.size(), c.data(), H5P_DEFAULT);
        }
        for (const char * group_name : {"CellData", "FieldData"}) {
            h5::Handle group(H5Gcreate2(top, group_name, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT), H5Gclose, group_name);
        }
        h5::Handle point_data(H5Gcreate2(top, "PointData", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT), H5Gclose,
                              "PointData");
        if (curves) {
            h5::write_rows(point_data, "arc_length", n, 1, 0, n, geometry.arc_length.data(), H5P_DEFAULT);
            h5::annotate(point_data, "arc_length", FieldDescription{dim::length, "arc length", "", ""});
        }
        h5::annotate(top, "Points", FieldDescription{dim::length, "sample coordinates", "", ""});
        h5::Handle steps(H5Gcreate2(top, "Steps", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT), H5Gclose, "Steps");
        for (const char * group_name : {"CellDataOffsets", "PointDataOffsets", "FieldDataOffsets"}) {
            h5::Handle group(H5Gcreate2(steps, group_name, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT), H5Gclose, group_name);
        }
    }
    h5::Handle h5file(H5Fopen(file.c_str(), H5F_ACC_RDWR, H5P_DEFAULT), H5Fclose, "opening " + file);
    h5::Handle point_data(H5Gopen2(h5file, "VTKHDF/PointData", H5P_DEFAULT), H5Gclose, "opening PointData");
    std::vector<double> column(n);
    for (size_t v = 0; v < variables.size(); v++) {
        for (uint64_t i = 0; i < n; i++) column[i] = values[i * n_cols + v];
        h5::put_rows(point_data, variables[v].c_str(), (k + 1) * n, 1, k * n, n, column.data(), H5P_DEFAULT);
        if (k == 0) h5::annotate(point_data, variables[v], field_units::describe(variables[v]));
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
    h5::Handle data_offsets(H5Gopen2(steps, "PointDataOffsets", H5P_DEFAULT), H5Gclose, "opening PointDataOffsets");
    for (const auto & v : variables) h5::put_rows(data_offsets, v.c_str(), k + 1, 1, k, 1, &offset, H5P_DEFAULT);
    if (curves) h5::put_rows(data_offsets, "arc_length", k + 1, 1, k, 1, zeros, H5P_DEFAULT);
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
    const bool points = geometry.dim == 0;
    const std::vector<double> & q = points ? node_values : quad;
    const size_t n_cols = n_evaluated();
    const size_t n_q = points ? geometry.nodes.size() : geometry.quad_weights.size();
    double measure = 0.0;
    for (size_t i = 0; i < n_q; i++) measure += points ? 1.0 : geometry.quad_weights[i];
    std::ostream & row = *out_reductions;
    row << step << "," << std::setprecision(std::numeric_limits<double>::max_digits10) << double(t) << "," << measure;
    for (const Reduction r : reductions) {
        for (size_t v = 0; v < variables.size(); v++) {
            double sum = 0.0, sum_rho = 0.0;
            double lo = std::numeric_limits<double>::infinity(), hi = -lo;
            for (size_t i = 0; i < n_q; i++) {
                const double w = points ? 1.0 : geometry.quad_weights[i];
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

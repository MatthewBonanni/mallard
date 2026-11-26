/**
 * @file data_writer_vtkhdf.cpp
 * @brief VTKHDF solution output (VTK's HDF5 format, read by ParaView 5.12 and later).
 * @version 0.9
 * @date 2026-10-29
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 *
 */

#include "data_writer.h"

#include <algorithm>
#include <array>
#include <bit>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <mutex>
#include <regex>
#include <sstream>

#include "comm.h"
#include "common_io.h"
#include "compression.h"
#include "file_format.h"
#include "hdf5_util.h"
#include "json.h"
#include "lagrange.h"
#include "provenance.h"

namespace {

bool kept_on_resume(double t_entry, rtype t) {
    return t_entry <= static_cast<double>(t) * (1.0 + 1.0e-12) + 1.0e-300;
}

} // namespace

void DataWriter::write_vtkhdf_series(const Index & list) const {
    std::ofstream out(partial_name(prefix + ".vtkhdf.series"));
    if (!out.good()) throw std::runtime_error("DataWriter: Could not open file: " + prefix + ".vtkhdf.series.");
    out << std::setprecision(std::numeric_limits<double>::max_digits10);
    out << "{\n  \"file-series-version\": \"1.0\",\n  \"files\": [\n";
    for (size_t i = 0; i < list.size(); i++) {
        out << "    {\"name\": " << json::quote(std::filesystem::path(list[i].file).filename().string())
            << ", \"time\": " << list[i].t;
        if (list[i].step >= 0) out << ", \"step\": " << list[i].step;
        out << "}" << (i + 1 < list.size() ? "," : "") << "\n";
    }
    out << "  ]\n}\n";
    out.close();
    if (!out) throw cannot_write("the VTKHDF series index", prefix + ".vtkhdf.series");
    publish_file("the VTKHDF series index", prefix + ".vtkhdf.series");
}

void DataWriter::resume_vtkhdf(rtype t) {
    if (!one_file) {
        // Keep the snapshots of the previous run's index up to the restart time
        std::ifstream in(prefix + ".vtkhdf.series");
        if (!in.good()) return;
        const std::filesystem::path dir = std::filesystem::path(prefix).parent_path();
        try {
            const json::Value index = json::parse(std::string(std::istreambuf_iterator<char>(in), {}));
            for (const auto & e : index["files"].array) {
                const double t_entry = e["time"].number;
                const json::Value * step = e.find("step");
                if (kept_on_resume(t_entry, t)) {
                    history.push_back({t_entry, step ? static_cast<int64_t>(step->number) : -1,
                                       (dir / e["name"].string).string()});
                }
            }
        } catch (const std::exception & error) {
            throw std::runtime_error("DataWriter: " + prefix + ".vtkhdf.series is not a series index: " + error.what());
        }
        return;
    }
    // Rank 0 keeps the series' snapshots up to the restart time (renaming a series in one file of Mallard 0.11
    // and earlier, which it continues), and every rank numbers on from their count
    series_steps.clear();
    vtkhdf_mesh_ready = false;
    vtkhdf_reuse_mesh = std::filesystem::exists(vtkhdf_mesh_file());
    std::string kept, error;
    if (comm::is_root()) {
        try {
            series_steps = vtkhdf_series::resume(prefix + ".vtkhdf", double(t), prefix + "_legacy.vtkhdf");
        } catch (const std::exception & e) {
            error = e.what();
        }
        std::ostringstream out;
        out << std::setprecision(std::numeric_limits<double>::max_digits10);
        for (const auto & s : series_steps) {
            const auto step = s.values.find("FieldData/STEP");
            out << s.t << " " << (step == s.values.end() ? int64_t(-1) : int64_t(step->second)) << " " << s.file << "\n";
        }
        kept = out.str();
    }
    const bool shared = mesh->n_global_cells > 0 && comm::size() > 1;
    if (shared) {
        error = comm::broadcast(error);
        kept = comm::broadcast(kept);
    }
    if (!error.empty()) throw std::runtime_error("DataWriter: continuing " + prefix + ".vtkhdf: " + error);
    const std::filesystem::path dir = std::filesystem::path(prefix).parent_path();
    std::istringstream in(kept);
    IndexEntry e;
    while (in >> e.t >> e.step && in.get() && std::getline(in, e.file)) {
        e.file = (dir / e.file).string();
        history.push_back(e);
    }
}

#ifdef Mallard_HAS_HDF5

using namespace h5;

namespace {

using curved::Vec3;

// The VTKHDF version whose features the files use (temporal data): ParaView 5.12 reads it
const std::vector<int64_t> VTKHDF_VERSION = {2, 0};

/** @brief This rank's offset in a sum over the ranks (in rank order), and the sum. */
std::pair<uint64_t, uint64_t> scan(uint64_t value, bool distributed) {
    if (!distributed) return {0, value};
    const std::vector<uint64_t> all = comm::allgatherv(std::vector<uint64_t>{value});
    uint64_t first = 0, total = 0;
    for (int r = 0; r < comm::size(); r++) {
        if (r < comm::rank()) first += all[static_cast<size_t>(r)];
        total += all[static_cast<size_t>(r)];
    }
    return {first, total};
}

bool last_rank(bool distributed) { return !distributed || comm::rank() == comm::size() - 1; }

bool first_rank(bool distributed) { return !distributed || comm::rank() == 0; }

std::vector<uint64_t> sorted_unique(std::vector<uint64_t> values) {
    std::sort(values.begin(), values.end());
    values.erase(std::unique(values.begin(), values.end()), values.end());
    return values;
}

uint64_t index_in(const std::vector<uint64_t> & sorted, uint64_t value) {
    return static_cast<uint64_t>(std::lower_bound(sorted.begin(), sorted.end(), value) - sorted.begin());
}

/** @brief A one-value dataset, from the first rank. */
void write_count(hid_t parent, const char * name, uint64_t value, hid_t dxpl, bool distributed) {
    const int64_t v = static_cast<int64_t>(value);
    write_rows<int64_t>(parent, name, 1, 1, 0, first_rank(distributed) ? 1 : 0, &v, dxpl);
}

} // namespace

void DataWriter::build_vtkhdf_cells() {
    const bool distributed = mesh->n_global_cells > 0;
    const uint32_t n_out = subset ? static_cast<uint32_t>(subset_cells.size()) : mesh->n_owned();
    const uint64_t n_cells = subset ? n_subset : distributed ? mesh->n_global_cells : mesh->n_cells;
    // A subset's points are the nodes of its cells, numbered in global order
    std::vector<uint64_t> subset_nodes;
    auto mesh_node = [&](uint32_t n) { return distributed ? mesh->h_global_node_id[n] : uint64_t(n); };
    if (subset) {
        for (uint32_t c : subset_cells) {
            for (uint32_t k = 0; k < mesh->h_n_nodes_of_cell(c); k++) subset_nodes.push_back(mesh_node(mesh->h_node_of_cell(c, k)));
        }
        subset_nodes = sorted_unique(subset_nodes);
        subset_nodes = sorted_unique(distributed ? comm::allgatherv(subset_nodes) : subset_nodes);
    }
    const uint64_t n_nodes = subset ? subset_nodes.size() : distributed ? mesh->n_global_nodes : mesh->n_nodes;
    auto global_node = [&](uint32_t n) { return subset ? index_in(subset_nodes, mesh_node(n)) : mesh_node(n); };
    const curved::Geometry * curves = mesh->curved_geometry.get();

    // Owned cells as records: type, corners, their global nodes, the count of
    // further (quadratic) nodes and those nodes' coordinates as raw bits
    std::vector<uint64_t> ids(n_out), offsets{0}, records;
    std::vector<bool> used(mesh->n_nodes, false);
    std::vector<uint32_t> corners;
    for (uint32_t j = 0; j < n_out; j++) {
        const uint32_t i = subset ? subset_cells[j] : j;
        ids[j] = subset ? subset_ids[j] : mesh->h_global_cell(i);
        const uint32_t n = mesh->h_n_nodes_of_cell(i);
        corners.resize(n);
        for (uint32_t k = 0; k < n; k++) {
            corners[k] = mesh->h_node_of_cell(i, vtk_local_node(n, k));
            used[corners[k]] = true;
        }
        const bool curved = curves && curves->cell_is_curved(i);
        const std::vector<Vec3> extra = curved ? lagrange::quadratic_nodes(*mesh, *curves, i, corners) : std::vector<Vec3>{};
        records.push_back(curved ? lagrange::quadratic_vtk_type(n) : vtk_cell_type(N_DIM, n));
        records.push_back(n);
        for (uint32_t node : corners) records.push_back(global_node(node));
        records.push_back(extra.size());
        for (const Vec3 & x : extra) {
            for (double v : x) records.push_back(std::bit_cast<uint64_t>(v));
        }
        offsets.push_back(records.size());
    }
    vtkhdf.order = GlobalOrder(ids, n_cells, distributed);
    const auto [block_offsets, block] = vtkhdf.order.gather_csr(offsets, records);
    const uint64_t n_block = vtkhdf.order.size();

    uint64_t n_extra = 0, n_ids = 0;
    for (uint64_t c = 0; c < n_block; c++) {
        const uint64_t * r = &block[block_offsets[c]];
        n_extra += r[2 + r[1]];
        n_ids += r[1] + r[2 + r[1]];
    }
    const auto [extra_first, extra_total] = scan(n_extra, distributed);
    const auto [id_first, id_total] = scan(n_ids, distributed);

    // Points: the mesh's nodes, then the quadratic nodes in the order of their cells
    std::vector<uint64_t> point_ids;
    std::vector<rtype> coords;
    for (uint32_t n = 0; n < mesh->n_nodes; n++) {
        if (!used[n]) continue;
        point_ids.push_back(global_node(n));
        for (uint32_t i = 0; i < 3; i++) coords.push_back(i < N_DIM ? mesh->h_node_coords(n, i) : rtype(0));
    }
    VtkhdfMesh & m = vtkhdf;
    m.connectivity.clear();
    m.offsets.clear();
    m.types.clear();
    uint64_t next = n_nodes + extra_first;
    int64_t at = static_cast<int64_t>(id_first);
    for (uint64_t c = 0; c < n_block; c++) {
        const uint64_t * r = &block[block_offsets[c]];
        const uint64_t n = r[1], e = r[2 + n];
        m.types.push_back(static_cast<uint8_t>(r[0]));
        m.offsets.push_back(at);
        for (uint64_t k = 0; k < n; k++) m.connectivity.push_back(static_cast<int64_t>(r[2 + k]));
        for (uint64_t k = 0; k < e; k++) {
            m.connectivity.push_back(static_cast<int64_t>(next));
            point_ids.push_back(next++);
            for (uint64_t i = 0; i < 3; i++) coords.push_back(static_cast<rtype>(std::bit_cast<double>(r[3 + n + 3 * k + i])));
        }
        at += static_cast<int64_t>(n + e);
    }
    if (last_rank(distributed)) m.offsets.push_back(static_cast<int64_t>(id_total));
    m.n_cells = n_cells;
    m.n_ids = id_total;
    m.n_points = n_nodes + extra_total;
    m.first_cell = vtkhdf.order.first();
    m.first_id = id_first;
    const GlobalOrder point_order(point_ids, m.n_points, distributed);
    m.points = point_order.gather<rtype>(coords, 3);
    m.first_point = point_order.first();
    m.built = true;
}

namespace {

/** @brief Values as the raw bits of doubles, for GlobalOrder::gather_csr. */
std::vector<uint64_t> as_bits(const rtype * values, size_t n) {
    std::vector<uint64_t> out(n);
    for (size_t i = 0; i < n; i++) out[i] = std::bit_cast<uint64_t>(static_cast<double>(values[i]));
    return out;
}

} // namespace

void DataWriter::build_vtkhdf_view() {
    const bool distributed = mesh->n_global_cells > 0;
    const uint32_t n_out = mesh->n_owned();
    // Each cell's record: its Lagrange cells' count, their types and node counts, then their nodes' coordinates as raw bits
    std::vector<uint64_t> ids(n_out), offsets{0}, records;
    for (uint32_t k = 0; k < n_out; k++) {
        ids[k] = mesh->h_global_cell(owned_by_id[k]);
        const uint64_t e0 = view->first_entry[k], e1 = view->first_entry[k + 1];
        records.push_back(e1 - e0);
        for (uint64_t e = e0; e < e1; e++) {
            records.push_back(view->types[e]);
            records.push_back(view->first_node[e + 1] - view->first_node[e]);
        }
        const std::vector<uint64_t> bits = as_bits(view->points.data() + 3 * view->first_node[e0],
                                                   3 * (view->first_node[e1] - view->first_node[e0]));
        records.insert(records.end(), bits.begin(), bits.end());
        offsets.push_back(records.size());
    }
    VtkhdfMesh & m = vtkhdf;
    m.order = GlobalOrder(ids, distributed ? mesh->n_global_cells : mesh->n_cells, distributed);
    const auto [block_offsets, block] = m.order.gather_csr(offsets, records);
    const uint64_t n_block = m.order.size();
    m.connectivity.clear();
    m.offsets.clear();
    m.types.clear();
    m.points.clear();
    m.repeat.clear();
    uint64_t n_points = 0;
    for (uint64_t c = 0; c < n_block; c++) {
        const uint64_t * r = &block[block_offsets[c]];
        for (uint64_t e = 0; e < r[0]; e++) n_points += r[2 + 2 * e];
    }
    const auto [first, total] = scan(n_points, distributed);
    int64_t at = static_cast<int64_t>(first);
    for (uint64_t c = 0; c < n_block; c++) {
        const uint64_t * r = &block[block_offsets[c]];
        const uint64_t n_sub = r[0];
        const uint64_t * coords = r + 1 + 2 * n_sub;
        m.repeat.push_back(static_cast<uint32_t>(n_sub));
        uint64_t n_cell_points = 0;
        for (uint64_t e = 0; e < n_sub; e++) {
            const uint64_t n = r[2 + 2 * e];
            m.types.push_back(static_cast<uint8_t>(r[1 + 2 * e]));
            m.offsets.push_back(at);
            for (uint64_t k = 0; k < n; k++) m.connectivity.push_back(at + static_cast<int64_t>(k));
            at += static_cast<int64_t>(n);
            n_cell_points += n;
        }
        for (uint64_t i = 0; i < 3 * n_cell_points; i++) m.points.push_back(static_cast<rtype>(std::bit_cast<double>(coords[i])));
    }
    const auto [first_cell, n_cells] = scan(m.types.size(), distributed);
    if (last_rank(distributed)) m.offsets.push_back(static_cast<int64_t>(total));
    m.n_cells = n_cells;
    m.n_ids = m.n_points = total;
    m.first_cell = first_cell;
    m.first_id = m.first_point = first;
    m.built = true;
}

void DataWriter::write_vtkhdf(const std::string & filename, const Block & snapshot, const Values & values) {
    const std::lock_guard<std::mutex> lock(h5::mutex());
    const bool distributed = mesh->n_global_cells > 0;
    if (!vtkhdf.built) {
        if (view) {
            build_vtkhdf_view();
        } else {
            build_vtkhdf_cells();
        }
    }
    const VtkhdfMesh & m = vtkhdf;
    const bool deflate = compression == Compression::ZLIB;
    const bool serial = !distributed;
    const uint64_t n_block = m.order.size();
    // A series shares one mesh file, linked from each snapshot's, unless its region changes
    const bool linked = one_file && !subset_update;

    Handle fapl(file_access(distributed), H5Pclose, "H5Pcreate");
    Handle dxpl(transfer(distributed), H5Pclose, "H5Pcreate");
    const std::string & provenance = provenance::json();
    // A file under its temporary name with the run block and the VTKHDF group
    const auto create = [&](const std::string & name) {
        Handle fcpl(file_creation(fapl, provenance.size()), H5Pclose, "H5Pcreate");
        const hid_t file = H5Fcreate(partial_name(name).c_str(), H5F_ACC_TRUNC, fcpl, fapl);
        if (file < 0) throw std::runtime_error("HDF5: creating " + name + " failed.");
        try {
            Handle root(H5Gopen2(file, "/", H5P_DEFAULT), H5Gclose, "H5Gopen2");
            write_strings_attribute(root, provenance::ATTRIBUTE, {provenance});
            write_int_attribute(root, "version", static_cast<int>(file_format::VTKHDF_OUTPUT.version));
            Handle top(H5Gcreate2(file, "VTKHDF", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT), H5Gclose, "creating VTKHDF");
            write_array_attribute(top, "Version", VTKHDF_VERSION);
            write_string_scalar_attribute(top, "Type", "UnstructuredGrid");
        } catch (...) {
            H5Fclose(file);
            throw;
        }
        return file;
    };
    // The mesh's counts and arrays, or its arrays as links to the series' mesh file
    const std::string target = std::filesystem::path(vtkhdf_mesh_file()).filename().string();
    const auto write_mesh = [&](hid_t top, bool link) {
        const auto array = [&](const char * name, uint64_t n_rows, int cols, uint64_t first, const auto & data) {
            if (link) {
                check(H5Lcreate_external(target.c_str(), (std::string("/VTKHDF/") + name).c_str(), top, name,
                                         H5P_DEFAULT, H5P_DEFAULT),
                      std::string("linking ") + name);
            } else {
                write_dataset(top, name, n_rows, cols, first, data.size() / size_t(cols), data.data(), dxpl, deflate,
                              serial, io_threads);
            }
        };
        write_count(top, "NumberOfPoints", m.n_points, dxpl, distributed);
        array("Points", m.n_points, 3, m.first_point, m.points);
        if (!link) annotate(top, "Points", FieldDescription{dim::length, "point coordinates", "", ""});
        write_count(top, "NumberOfCells", m.n_cells, dxpl, distributed);
        write_count(top, "NumberOfConnectivityIds", m.n_ids, dxpl, distributed);
        array("Offsets", m.n_cells + 1, 1, m.first_cell, m.offsets);
        array("Connectivity", m.n_ids, 1, m.first_id, m.connectivity);
        array("Types", m.n_cells, 1, m.first_cell, m.types);
        for (const char * name : {"CellData", "PointData", "FieldData"}) {
            Handle group(H5Gcreate2(top, name, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT), H5Gclose, name);
        }
    };
    if (linked && !vtkhdf_mesh_ready) {
        const std::string mesh_file = vtkhdf_mesh_file();
        if (vtkhdf_reuse_mesh) {
            // A continued series: the mesh file is the run's mesh
            int32_t same = 1;
            if (comm::is_root()) {
                Handle file(H5Fopen(mesh_file.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT), H5Fclose, "opening " + mesh_file);
                Handle top(H5Gopen2(file, "VTKHDF", H5P_DEFAULT), H5Gclose, "opening VTKHDF");
                const auto count = [&](const char * name) {
                    return static_cast<uint64_t>(read_rows<int64_t>(top, name, 0, 1, 1, H5P_DEFAULT).at(0));
                };
                same = count("NumberOfPoints") == m.n_points && count("NumberOfCells") == m.n_cells &&
                       count("NumberOfConnectivityIds") == m.n_ids;
            }
            if (distributed && comm::size() > 1) same = comm::allreduce(same, comm::Op::MIN);
            if (!same) {
                throw std::runtime_error("DataWriter: " + mesh_file +
                                         " holds another mesh than the series continued; move the series away to "
                                         "start a new one.");
            }
        } else {
            {
                Handle file(create(mesh_file), H5Fclose, "creating " + mesh_file);
                Handle top(H5Gopen2(file, "VTKHDF", H5P_DEFAULT), H5Gclose, "opening VTKHDF");
                write_mesh(top, false);
            }
            if (comm::is_root()) publish_file("the VTKHDF mesh file", mesh_file);
        }
        vtkhdf_mesh_ready = true;
    }

    Handle file(create(filename), H5Fclose, "creating " + filename);
    Handle top(H5Gopen2(file, "VTKHDF", H5P_DEFAULT), H5Gclose, "opening VTKHDF");
    write_mesh(top, linked);
    {
        // The snapshot's time and step as field data (ParaView shows them), and its snapshot block
        const uint64_t mine = first_rank(distributed) ? 1 : 0;
        Handle field_data(H5Gopen2(top, "FieldData", H5P_DEFAULT), H5Gclose, "opening FieldData");
        const double time = snapshot.time;
        put_rows(field_data, "TIME", 1, 1, 0, mine, &time, dxpl);
        annotate(field_data, "TIME", FieldDescription{dim::time, "time", "time", ""});
        if (snapshot_blocks) {
            const int64_t step = snapshot.step;
            put_rows(field_data, "STEP", 1, 1, 0, mine, &step, dxpl);
            annotate(field_data, "STEP", FieldDescription{dim::none, "time step number", "", ""});
            const std::string text = snapshot.json();
            put_string_row(file, snapshot_block::DATASET, 1, 0, mine, text, snapshot_block::dataset_width(text.size()),
                           dxpl);
        }
    }
    Handle cell_data(H5Gopen2(top, "CellData", H5P_DEFAULT), H5Gclose, "opening CellData");
    std::vector<rtype> local;
    for (size_t f = 0; f < fields.size(); f++) {
        const uint32_t n_comp = fields[f].n_vtk_components();
        local = values[f];
        if (relative_error > 0.0) compression::round_to_relative_error(local, relative_error);
        std::vector<rtype> block = m.order.gather<rtype>(local, n_comp);
        if (!m.repeat.empty() && m.types.size() != n_block) {
            // A cell's value on each of its Lagrange cells
            std::vector<rtype> each;
            each.reserve(m.types.size() * n_comp);
            for (uint64_t c = 0; c < n_block; c++) {
                for (uint32_t e = 0; e < m.repeat[c]; e++) {
                    each.insert(each.end(), block.begin() + std::ptrdiff_t(c * n_comp),
                                block.begin() + std::ptrdiff_t((c + 1) * n_comp));
                }
            }
            block = std::move(each);
        }
        write_dataset(cell_data, fields[f].name.c_str(), m.n_cells, int(n_comp), m.first_cell, block.size() / n_comp,
                      block.data(), dxpl, deflate, serial, io_threads);
        annotate(cell_data, fields[f].name, fields[f].description);
        h5::cell_methods(cell_data, fields[f].name, cell_methods(fields[f].name));
    }
    if (node_fields.empty()) return;
    Handle point_data(H5Gopen2(top, "PointData", H5P_DEFAULT), H5Gclose, "opening PointData");
    for (size_t f = 0; f < node_fields.size(); f++) {
        const Field & field = node_fields[f];
        const uint32_t n_comp = LagrangeView::components(field.name);
        local = values[fields.size() + f];
        if (relative_error > 0.0) compression::round_to_relative_error(local, relative_error);
        std::vector<uint64_t> offsets(view->first_entry.size());
        for (size_t k = 0; k < offsets.size(); k++) offsets[k] = view->first_node[view->first_entry[k]] * n_comp;
        const auto [block_offsets, bits] = m.order.gather_csr(offsets, as_bits(local.data(), local.size()));
        std::vector<rtype> block(bits.size());
        for (size_t i = 0; i < bits.size(); i++) block[i] = static_cast<rtype>(std::bit_cast<double>(bits[i]));
        write_dataset(point_data, field.name.c_str(), m.n_points, int(n_comp), m.first_point, block.size() / n_comp,
                      block.data(), dxpl, deflate, serial, io_threads);
        annotate(point_data, field.name, field.description);
        snapshot_block::Field point{field.name, "", "", "", 0.0, snapshot_block::value::POINT_VALUE, {}};
        if (snapshot_blocks) h5::cell_methods(point_data, field.name, snapshot_block::cell_methods(point));
    }
}

#else

void DataWriter::write_vtkhdf(const std::string &, const Block &, const Values &) {
    throw std::logic_error("DataWriter: VTKHDF output without HDF5");
}

void DataWriter::build_vtkhdf_cells() {}

void DataWriter::build_vtkhdf_view() {}

#endif

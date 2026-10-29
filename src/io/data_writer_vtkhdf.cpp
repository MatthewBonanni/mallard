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
#include "provenance.h"

namespace {

bool kept_on_resume(double t_entry, rtype t) {
    return t_entry <= static_cast<double>(t) * (1.0 + 1.0e-12) + 1.0e-300;
}

} // namespace

void DataWriter::write_vtkhdf_series(const std::vector<std::pair<rtype, std::string>> & list) const {
    std::ofstream out(prefix + ".vtkhdf.series");
    if (!out.good()) throw std::runtime_error("DataWriter: Could not open file: " + prefix + ".vtkhdf.series.");
    out << std::setprecision(std::numeric_limits<double>::max_digits10);
    out << "{\n  \"file-series-version\": \"1.0\",\n  \"files\": [\n";
    for (size_t i = 0; i < list.size(); i++) {
        out << "    {\"name\": " << json::quote(std::filesystem::path(list[i].second).filename().string())
            << ", \"time\": " << static_cast<double>(list[i].first) << "}" << (i + 1 < list.size() ? "," : "") << "\n";
    }
    out << "  ]\n}\n";
}

void DataWriter::resume_vtkhdf(rtype t) {
    if (!one_file) {
        // Keep the snapshots of the previous run's index up to the restart time
        std::ifstream in(prefix + ".vtkhdf.series");
        const std::filesystem::path dir = std::filesystem::path(prefix).parent_path();
        const std::regex entry(R"re("name":\s*"((?:[^"\\]|\\.)*)",\s*"time":\s*([^\s}]+))re");
        std::string line;
        while (std::getline(in, line)) {
            std::smatch match;
            if (!std::regex_search(line, match, entry)) continue;
            const double t_entry = std::stod(match[2].str());
            if (kept_on_resume(t_entry, t)) history.emplace_back(t_entry, (dir / match[1].str()).string());
        }
        return;
    }
    vtkhdf_steps = 0;
#ifdef Mallard_HAS_HDF5
    const std::string filename = prefix + ".vtkhdf";
    if (!std::filesystem::exists(filename)) return;
    const std::lock_guard<std::mutex> lock(h5::mutex());
    h5::Handle file(H5Fopen(filename.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT), H5Fclose, "opening " + filename);
    if (H5Lexists(file, "VTKHDF", H5P_DEFAULT) <= 0 || H5Lexists(file, "VTKHDF/Steps", H5P_DEFAULT) <= 0) return;
    h5::Handle steps(H5Gopen2(file, "VTKHDF/Steps", H5P_DEFAULT), H5Gclose, "opening VTKHDF/Steps");
    const std::vector<double> times =
        h5::read_rows<double>(steps, "Values", 0, h5::n_rows(steps, "Values"), 1, H5P_DEFAULT);
    while (vtkhdf_steps < times.size() && kept_on_resume(times[vtkhdf_steps], t)) vtkhdf_steps++;
#endif
}

#ifdef Mallard_HAS_HDF5

using namespace h5;

namespace {

using curved::Vec3;

constexpr const char * VTKHDF_FORMAT = "mallard-vtkhdf";
// The VTKHDF version whose features the files use (temporal data): ParaView 5.12 reads it
const std::vector<int64_t> VTKHDF_VERSION = {2, 0};

constexpr uint8_t VTK_LAGRANGE_TRIANGLE = 69, VTK_LAGRANGE_QUADRILATERAL = 70, VTK_LAGRANGE_TETRAHEDRON = 71,
                  VTK_LAGRANGE_HEXAHEDRON = 72, VTK_LAGRANGE_WEDGE = 73, VTK_QUADRATIC_PYRAMID = 27;

using Pairs = std::vector<std::array<uint8_t, 2>>;
using Quads = std::vector<std::array<uint8_t, 4>>;

/** @brief A curved cell as a quadratic VTK cell: its edges and quadrilateral faces with a middle node. */
struct Quadratic {
    uint8_t type;
    Pairs edges;
    Quads faces;
    bool body;
};

const Quadratic & quadratic(uint32_t n_corners) {
    static const Quadratic triangle{VTK_LAGRANGE_TRIANGLE, {{0, 1}, {1, 2}, {2, 0}}, {}, false};
    static const Quadratic quad{VTK_LAGRANGE_QUADRILATERAL, {{0, 1}, {1, 2}, {3, 2}, {0, 3}}, {}, true};
    static const Quadratic tetrahedron{
        VTK_LAGRANGE_TETRAHEDRON, {{0, 1}, {1, 2}, {0, 2}, {0, 3}, {1, 3}, {2, 3}}, {}, false};
    static const Quadratic pyramid{
        VTK_QUADRATIC_PYRAMID, {{0, 1}, {1, 2}, {2, 3}, {3, 0}, {0, 4}, {1, 4}, {2, 4}, {3, 4}}, {}, false};
    static const Quadratic wedge{VTK_LAGRANGE_WEDGE,
                                 {{0, 1}, {1, 2}, {2, 0}, {3, 4}, {4, 5}, {5, 3}, {0, 3}, {1, 4}, {2, 5}},
                                 {{0, 1, 4, 3}, {1, 2, 5, 4}, {2, 0, 3, 5}},
                                 false};
    static const Quadratic hexahedron{
        VTK_LAGRANGE_HEXAHEDRON,
        {{0, 1}, {1, 2}, {3, 2}, {0, 3}, {4, 5}, {5, 6}, {7, 6}, {4, 7}, {0, 4}, {1, 5}, {3, 7}, {2, 6}},
        {{0, 3, 7, 4}, {1, 2, 6, 5}, {0, 1, 5, 4}, {3, 2, 6, 7}, {0, 1, 2, 3}, {4, 5, 6, 7}},
        true};
    if (N_DIM == 2) return n_corners == 3 ? triangle : quad;
    switch (n_corners) {
        case 4: return tetrahedron;
        case 5: return pyramid;
        case 6: return wedge;
        default: return hexahedron;
    }
}

/** @brief Reference coordinates of node k of a face with n nodes (curved::Geometry::eval). */
std::array<double, 2> face_reference(uint32_t n, uint32_t k) {
    static constexpr double TRIANGLE[3][2] = {{0.0, 0.0}, {1.0, 0.0}, {0.0, 1.0}};
    static constexpr double QUAD[4][2] = {{-1.0, -1.0}, {1.0, -1.0}, {1.0, 1.0}, {-1.0, 1.0}};
    if (n == 2) return {k == 0 ? -1.0 : 1.0, 0.0};
    if (n == 3) return {TRIANGLE[k][0], TRIANGLE[k][1]};
    return {QUAD[k][0], QUAD[k][1]};
}

/**
 * @brief The point of cell c's curved geometry at the centroid of these
 *        nodes (an edge's two or a quadrilateral face's four): on a face of the
 *        cell holding them all (a curved one where there is one), else their
 *        average.
 */
Vec3 face_point(const Mesh & mesh, const curved::Geometry & geometry, uint32_t c, const std::vector<uint32_t> & nodes,
                const Vec3 & average) {
    int64_t chosen = -1;
    std::array<double, 2> xi = {0.0, 0.0};
    for (uint32_t j = 0; j < mesh.h_n_faces_of_cell(c); j++) {
        const uint32_t f = mesh.h_face_of_cell(c, j);
        if (chosen >= 0 && !geometry.face_is_curved(f)) continue;
        const uint32_t n = mesh.h_n_nodes_of_face(f);
        std::array<double, 2> sum = {0.0, 0.0};
        uint32_t found = 0;
        for (uint32_t k = 0; k < n; k++) {
            if (std::find(nodes.begin(), nodes.end(), mesh.h_node_of_face(f, k)) == nodes.end()) continue;
            const auto r = face_reference(n, k);
            sum = {sum[0] + r[0], sum[1] + r[1]};
            found++;
        }
        if (found != nodes.size()) continue;
        chosen = f;
        xi = {sum[0] / double(found), sum[1] / double(found)};
        if (geometry.face_is_curved(f)) break;
    }
    if (chosen < 0) return average;
    Vec3 x;
    geometry.eval(static_cast<uint32_t>(chosen), xi.data(), x, nullptr);
    return x;
}

/**
 * @brief The nodes of curved cell c after its corners (in VTK order) as a
 *        quadratic VTK cell: middles of the edges, of the quadrilateral faces
 *        and of the body (by transfinite interpolation of its boundary).
 */
std::vector<Vec3> quadratic_nodes(const Mesh & mesh, const curved::Geometry & geometry, uint32_t c,
                                  const std::vector<uint32_t> & corners) {
    const Quadratic & q = quadratic(uint32_t(corners.size()));
    std::vector<Vec3> x(corners.size());
    for (size_t k = 0; k < corners.size(); k++) {
        x[k] = {0.0, 0.0, 0.0};
        FOR_I_DIM x[k][i] = double(mesh.h_node_coords(corners[k], i));
    }
    auto average = [&](const auto & local) {
        Vec3 sum = {0.0, 0.0, 0.0};
        for (uint8_t k : local) {
            for (size_t i = 0; i < 3; i++) sum[i] += x[k][i];
        }
        for (double & v : sum) v /= double(local.size());
        return sum;
    };
    auto nodes_of = [&](const auto & local) {
        std::vector<uint32_t> nodes;
        for (uint8_t k : local) nodes.push_back(corners[k]);
        return nodes;
    };
    std::vector<Vec3> out;
    Vec3 body = {0.0, 0.0, 0.0};
    auto accumulate = [&](const Vec3 & p, double w) {
        for (size_t i = 0; i < 3; i++) body[i] += w * p[i];
    };
    const bool solid = N_DIM == 3;
    for (const auto & e : q.edges) {
        out.push_back(face_point(mesh, geometry, c, nodes_of(e), average(e)));
        accumulate(out.back(), solid ? -0.25 : 0.5);
    }
    for (const auto & f : q.faces) {
        out.push_back(face_point(mesh, geometry, c, nodes_of(f), average(f)));
        accumulate(out.back(), 0.5);
    }
    if (q.body) {
        for (const Vec3 & corner : x) accumulate(corner, solid ? 0.125 : -0.25);
        out.push_back(body);
    }
    return out;
}

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

Handle open_or_create_group(hid_t parent, const char * name) {
    if (H5Lexists(parent, name, H5P_DEFAULT) > 0) {
        return Handle(H5Gopen2(parent, name, H5P_DEFAULT), H5Gclose, std::string("opening ") + name);
    }
    return Handle(H5Gcreate2(parent, name, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT), H5Gclose,
                  std::string("creating ") + name);
}

} // namespace

void DataWriter::build_vtkhdf_cells() {
    const bool distributed = mesh->n_global_cells > 0;
    const uint32_t n_owned = mesh->n_owned();
    const uint64_t n_cells = distributed ? mesh->n_global_cells : mesh->n_cells;
    const uint64_t n_nodes = distributed ? mesh->n_global_nodes : mesh->n_nodes;
    auto global_node = [&](uint32_t n) { return distributed ? mesh->h_global_node_id[n] : uint64_t(n); };
    const curved::Geometry * curves = mesh->curved_geometry.get();

    // Owned cells as records: type, corners, their global nodes, the count of
    // further (quadratic) nodes and those nodes' coordinates as raw bits
    std::vector<uint64_t> ids(n_owned), offsets{0}, records;
    std::vector<bool> used(mesh->n_nodes, false);
    std::vector<uint32_t> corners;
    for (uint32_t i = 0; i < n_owned; i++) {
        ids[i] = mesh->h_global_cell(i);
        const uint32_t n = mesh->h_n_nodes_of_cell(i);
        corners.resize(n);
        for (uint32_t k = 0; k < n; k++) {
            corners[k] = mesh->h_node_of_cell(i, vtk_local_node(n, k));
            used[corners[k]] = true;
        }
        const bool curved = curves && curves->cell_is_curved(i);
        const std::vector<Vec3> extra = curved ? quadratic_nodes(*mesh, *curves, i, corners) : std::vector<Vec3>{};
        records.push_back(curved ? quadratic(n).type : vtk_cell_type(N_DIM, n));
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

void DataWriter::build_vtkhdf_faces() {
    const bool distributed = mesh->n_global_cells > 0;
    auto global_node = [&](uint32_t n) { return distributed ? mesh->h_global_node_id[n] : uint64_t(n); };

    // Faces in the order of their key (their cell's global id and place in it), which no rank count changes
    std::vector<uint64_t> keys, offsets{0}, nodes;
    for (uint32_t f : geometry_faces) {
        const uint32_t cell = static_cast<uint32_t>(mesh->h_cells_of_face(f, 0));
        uint32_t j = 0;
        while (mesh->h_face_of_cell(cell, j) != f) j++;
        keys.push_back(mesh->h_global_cell(cell) * RESTART_FACES_PER_CELL + j);
        for (uint32_t k = 0; k < mesh->h_n_nodes_of_face(f); k++) nodes.push_back(global_node(mesh->h_node_of_face(f, k)));
        offsets.push_back(nodes.size());
    }
    const std::vector<uint64_t> all_keys = sorted_unique(distributed ? comm::allgatherv(keys) : keys);
    std::vector<uint64_t> ids;
    for (uint64_t key : keys) ids.push_back(index_in(all_keys, key));
    vtkhdf.order = GlobalOrder(ids, all_keys.size(), distributed);
    const auto [block_offsets, block] = vtkhdf.order.gather_csr(offsets, nodes);

    // Points: the zone's nodes in global order
    const std::vector<uint64_t> used = sorted_unique(nodes);
    const std::vector<uint64_t> all_nodes = sorted_unique(distributed ? comm::allgatherv(used) : used);
    std::vector<uint64_t> point_ids;
    std::vector<rtype> coords;
    for (uint32_t f : geometry_faces) {
        for (uint32_t k = 0; k < mesh->h_n_nodes_of_face(f); k++) {
            const uint32_t node = mesh->h_node_of_face(f, k);
            point_ids.push_back(index_in(all_nodes, global_node(node)));
            for (uint32_t i = 0; i < 3; i++) coords.push_back(i < N_DIM ? mesh->h_node_coords(node, i) : rtype(0));
        }
    }
    VtkhdfMesh & m = vtkhdf;
    const auto [id_first, id_total] = scan(block.size(), distributed);
    m.connectivity.clear();
    m.offsets.clear();
    for (uint64_t c = 0; c + 1 < block_offsets.size(); c++) {
        m.offsets.push_back(static_cast<int64_t>(id_first + block_offsets[c]));
        for (uint64_t k = block_offsets[c]; k < block_offsets[c + 1]; k++) {
            m.connectivity.push_back(static_cast<int64_t>(index_in(all_nodes, block[k])));
        }
    }
    if (last_rank(distributed)) m.offsets.push_back(static_cast<int64_t>(id_total));
    m.n_cells = all_keys.size();
    m.n_ids = id_total;
    m.n_points = all_nodes.size();
    m.first_cell = vtkhdf.order.first();
    m.first_id = id_first;
    const GlobalOrder point_order(point_ids, m.n_points, distributed);
    m.points = point_order.gather<rtype>(coords, 3);
    m.first_point = point_order.first();
    m.built = true;
}

void DataWriter::write_vtkhdf(const std::string & filename, uint64_t step, rtype t, const Values & values) {
    const std::lock_guard<std::mutex> lock(h5::mutex());
    const bool distributed = mesh->n_global_cells > 0;
    if (!vtkhdf.built) {
        if (surface) {
            build_vtkhdf_faces();
        } else {
            build_vtkhdf_cells();
        }
    }
    const VtkhdfMesh & m = vtkhdf;
    const bool deflate = compression == Compression::ZLIB;
    const bool serial = !distributed;
    const bool append = one_file && vtkhdf_steps > 0;
    const uint64_t k = one_file ? vtkhdf_steps : 0;
    const uint64_t n_block = m.order.size();

    Handle fapl(file_access(distributed), H5Pclose, "H5Pcreate");
    Handle dxpl(transfer(distributed), H5Pclose, "H5Pcreate");
    const std::string & provenance = provenance::json();
    Handle fcpl(file_creation(fapl, provenance.size()), H5Pclose, "H5Pcreate");
    Handle file(append ? H5Fopen(filename.c_str(), H5F_ACC_RDWR, fapl)
                       : H5Fcreate(filename.c_str(), H5F_ACC_TRUNC, fcpl, fapl),
                H5Fclose, (append ? "opening " : "creating ") + filename);
    if (!append) {
        Handle root(H5Gopen2(file, "/", H5P_DEFAULT), H5Gclose, "H5Gopen2");
        write_strings_attribute(root, "provenance", {provenance});
        write_strings_attribute(root, "format", {VTKHDF_FORMAT});
        write_int_attribute(root, "version", static_cast<int>(file_format::VTKHDF_OUTPUT.version));
        write_int_attribute(root, "dimension", N_DIM);
        if (!one_file) {
            write_attribute(root, "step", step);
            write_attribute(root, "time", double(t));
        }
        Handle top(H5Gcreate2(file, "VTKHDF", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT), H5Gclose, "creating VTKHDF");
        write_array_attribute(top, "Version", VTKHDF_VERSION);
        write_string_scalar_attribute(top, "Type", surface ? "PolyData" : "UnstructuredGrid");
        write_count(top, "NumberOfPoints", m.n_points, dxpl, distributed);
        write_dataset(top, "Points", m.n_points, 3, m.first_point, m.points.size() / 3, m.points.data(), dxpl, deflate,
                      serial, io_threads);
        annotate(top, "Points", FieldDescription{dim::length, "point coordinates", "", ""});
        auto topology = [&](hid_t group, uint64_t n_cells, uint64_t n_ids, bool mine) {
            const std::vector<int64_t> none;
            const std::vector<int64_t> zero = {0};
            const auto & offsets = mine ? m.offsets : last_rank(distributed) ? zero : none;
            const auto & connectivity = mine ? m.connectivity : none;
            write_count(group, "NumberOfCells", n_cells, dxpl, distributed);
            write_count(group, "NumberOfConnectivityIds", n_ids, dxpl, distributed);
            write_dataset(group, "Offsets", n_cells + 1, 1, mine ? m.first_cell : n_cells, offsets.size(),
                          offsets.data(), dxpl, deflate, serial, io_threads);
            write_dataset(group, "Connectivity", n_ids, 1, mine ? m.first_id : n_ids, connectivity.size(),
                          connectivity.data(), dxpl, deflate, serial, io_threads);
        };
        if (surface) {
            for (const char * name : {"Vertices", "Lines", "Polygons", "Strips"}) {
                const bool mine = std::string(name) == (N_DIM == 2 ? "Lines" : "Polygons");
                Handle group(H5Gcreate2(top, name, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT), H5Gclose, name);
                topology(group, mine ? m.n_cells : 0, mine ? m.n_ids : 0, mine);
            }
        } else {
            topology(top, m.n_cells, m.n_ids, true);
            write_dataset(top, "Types", m.n_cells, 1, m.first_cell, m.types.size(), m.types.data(), dxpl, deflate,
                          serial, io_threads);
        }
        for (const char * name : {"CellData", "PointData", "FieldData"}) {
            Handle group(H5Gcreate2(top, name, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT), H5Gclose, name);
            if (!one_file && std::string(name) == "FieldData") {
                const double time = double(t);
                write_rows<double>(group, "TIME", 1, 1, 0, first_rank(distributed) ? 1 : 0, &time, dxpl);
                annotate(group, "TIME", FieldDescription{dim::time, "time", "time", ""});
            }
        }
    }
    Handle top(H5Gopen2(file, "VTKHDF", H5P_DEFAULT), H5Gclose, "opening VTKHDF");
    if (append) {
        Handle cells(H5Gopen2(top, surface ? (N_DIM == 2 ? "Lines" : "Polygons") : ".", H5P_DEFAULT), H5Gclose,
                     "opening the cells");
        bool same = read_rows<int64_t>(cells, "NumberOfCells", 0, 1, 1, H5P_DEFAULT)[0] == static_cast<int64_t>(m.n_cells);
        Handle cell_data(H5Gopen2(top, "CellData", H5P_DEFAULT), H5Gclose, "opening CellData");
        H5G_info_t info;
        check(H5Gget_info(cell_data, &info), "H5Gget_info");
        same = same && info.nlinks == fields.size();
        for (const auto & field : fields) same = same && H5Lexists(cell_data, field.name.c_str(), H5P_DEFAULT) > 0;
        if (!same) {
            throw std::runtime_error("DataWriter: " + filename +
                                     " holds a series of another mesh or other variables; move it away to start a "
                                     "new series.");
        }
    }

    Handle cell_data(H5Gopen2(top, "CellData", H5P_DEFAULT), H5Gclose, "opening CellData");
    std::vector<rtype> local;
    for (size_t f = 0; f < fields.size(); f++) {
        const uint32_t n_comp = fields[f].n_vtk_components();
        local = values[f];
        if (relative_error > 0.0) compression::round_to_relative_error(local, relative_error);
        const std::vector<rtype> block = m.order.gather<rtype>(local, n_comp);
        const char * name = fields[f].name.c_str();
        if (one_file) {
            put_rows(cell_data, name, (k + 1) * m.n_cells, int(n_comp), k * m.n_cells + m.first_cell, n_block,
                     block.data(), dxpl, deflate);
        } else {
            write_dataset(cell_data, name, m.n_cells, int(n_comp), m.first_cell, n_block, block.data(), dxpl, deflate,
                          serial, io_threads);
        }
        if (!append) annotate(cell_data, fields[f].name, fields[f].description);
    }
    if (!one_file) return;

    // Every snapshot reuses the mesh: its parts, points and cells all start at 0
    Handle steps = open_or_create_group(top, "Steps");
    set_attribute<int64_t>(steps, "NSteps", static_cast<int64_t>(k + 1));
    const uint64_t mine = first_rank(distributed) ? 1 : 0;
    const double time = double(t);
    const std::array<int64_t, 4> zeros = {0, 0, 0, 0};
    const int64_t one = 1;
    put_rows(steps, "Values", k + 1, 1, k, mine, &time, dxpl);
    if (!append) annotate(steps, "Values", FieldDescription{dim::time, "time", "time", ""});
    put_rows(steps, "PartOffsets", k + 1, 1, k, mine, zeros.data(), dxpl);
    put_rows(steps, "NumberOfParts", k + 1, 1, k, mine, &one, dxpl);
    put_rows(steps, "PointOffsets", k + 1, 1, k, mine, zeros.data(), dxpl);
    const int topologies = surface ? 4 : 1;
    put_rows(steps, "CellOffsets", k + 1, topologies, k, mine, zeros.data(), dxpl);
    put_rows(steps, "ConnectivityIdOffsets", k + 1, topologies, k, mine, zeros.data(), dxpl);
    for (const char * name : {"PointDataOffsets", "FieldDataOffsets"}) open_or_create_group(steps, name);
    Handle data_offsets = open_or_create_group(steps, "CellDataOffsets");
    const int64_t offset = static_cast<int64_t>(k * m.n_cells);
    for (const auto & field : fields) put_rows(data_offsets, field.name.c_str(), k + 1, 1, k, mine, &offset, dxpl);
    vtkhdf_steps = k + 1;
}

#else

void DataWriter::write_vtkhdf(const std::string &, uint64_t, rtype, const Values &) {
    throw std::logic_error("DataWriter: VTKHDF output without HDF5");
}

void DataWriter::build_vtkhdf_cells() {}

void DataWriter::build_vtkhdf_faces() {}

#endif

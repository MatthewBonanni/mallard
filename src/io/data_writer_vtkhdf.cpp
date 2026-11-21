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
#include <numeric>
#include <regex>
#include <sstream>
#include <tuple>

#include "comm.h"
#include "common_io.h"
#include "compression.h"
#include "file_format.h"
#include "hdf5_util.h"
#include "json.h"
#include "provenance.h"

#define MESH_BLOCK "VTKHDF/mesh"
#define PARTICLE_BLOCK "VTKHDF/particles"

namespace {

bool kept_on_resume(double t_entry, rtype t) {
    return t_entry <= static_cast<double>(t) * (1.0 + 1.0e-12) + 1.0e-300;
}

} // namespace

void DataWriter::write_vtkhdf_series(const Index & list) const { write_series_index(prefix, list); }

void DataWriter::write_series_index(const std::string & file_prefix, const Index & list) {
    const std::string index = file_prefix + ".vtkhdf.series";
    std::ofstream out(partial_name(index));
    if (!out.good()) throw std::runtime_error("DataWriter: Could not open file: " + index + ".");
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
    if (!out) throw cannot_write("the VTKHDF series index", index);
    publish_file("the VTKHDF series index", index);
}

namespace {

/** @brief The snapshots of a .vtkhdf.series index kept by a run restarted at t. */
std::vector<std::tuple<double, int64_t, std::string>> kept_series(const std::string & file_prefix, rtype t) {
    std::vector<std::tuple<double, int64_t, std::string>> kept;
    std::ifstream in(file_prefix + ".vtkhdf.series");
    if (!in.good()) return kept;
    const std::filesystem::path dir = std::filesystem::path(file_prefix).parent_path();
    try {
        const json::Value index = json::parse(std::string(std::istreambuf_iterator<char>(in), {}));
        for (const auto & e : index["files"].array) {
            const double t_entry = e["time"].number;
            const json::Value * step = e.find("step");
            if (kept_on_resume(t_entry, t)) {
                kept.emplace_back(t_entry, step ? static_cast<int64_t>(step->number) : -1,
                                  (dir / e["name"].string).string());
            }
        }
    } catch (const std::exception & error) {
        throw std::runtime_error("DataWriter: " + file_prefix + ".vtkhdf.series is not a series index: " + error.what());
    }
    return kept;
}

/** @brief The steps of a VTKHDF series file kept by a run restarted at t. */
uint64_t kept_steps([[maybe_unused]] const std::string & filename, [[maybe_unused]] const std::string & group,
                    [[maybe_unused]] rtype t) {
    uint64_t n = 0;
#ifdef Mallard_HAS_HDF5
    if (!std::filesystem::exists(filename)) return 0;
    const std::lock_guard<std::mutex> lock(h5::mutex());
    h5::Handle file(H5Fopen(filename.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT), H5Fclose, "opening " + filename);
    const std::string path = group + "/Steps";
    if (H5Lexists(file, "VTKHDF", H5P_DEFAULT) <= 0 || H5Lexists(file, group.c_str(), H5P_DEFAULT) <= 0 ||
        H5Lexists(file, path.c_str(), H5P_DEFAULT) <= 0) {
        return 0;
    }
    h5::Handle steps(H5Gopen2(file, path.c_str(), H5P_DEFAULT), H5Gclose, "opening " + path);
    const std::vector<double> times =
        h5::read_rows<double>(steps, "Values", 0, h5::n_rows(steps, "Values"), 1, H5P_DEFAULT);
    while (n < times.size() && kept_on_resume(times[n], t)) n++;
#endif
    return n;
}

} // namespace

std::optional<FieldDescription> particle_field_description(const std::string & name) {
    if (name == "V" || name == "V_X" || name == "V_Y" || name == "V_Z") {
        return FieldDescription{dim::velocity, "particle velocity", "", ""};
    }
    if (name == "T_P") return FieldDescription{dim::temperature, "particle temperature", "", ""};
    if (name == "ID") return FieldDescription{dim::none, "particle id", "", ""};
    for (const char * gas : {"RHO", "U", "U_X", "U_Y", "U_Z", "P", "T"}) {
        if (name == gas) return field_units::find(name);
    }
    return std::nullopt;
}

void DataWriter::resume_particles(rtype t) {
    particles->run_written = false;
    particles->history.clear();
    particles->steps = 0;
    if (one_file) {
        particles->steps = combined ? kept_steps(prefix + ".vtkhdf", PARTICLE_BLOCK, t)
                                    : kept_steps(particles->prefix + ".vtkhdf", "VTKHDF", t);
        return;
    }
    if (combined) return;
    for (const auto & [t_entry, step, file] : kept_series(particles->prefix, t)) {
        particles->history.push_back({t_entry, step, file});
    }
}

void DataWriter::resume_vtkhdf(rtype t) {
    vtkhdf_run_written = false;
    if (!one_file) {
        // Keep the snapshots of the previous run's index up to the restart time
        for (const auto & [t_entry, step, file] : kept_series(prefix, t)) history.push_back({t_entry, step, file});
        return;
    }
    vtkhdf_steps = kept_steps(prefix + ".vtkhdf", combined ? MESH_BLOCK : "VTKHDF", t);
}

namespace {

// The snapshot blocks of the particles of a file of the mesh and its particles
constexpr const char * PARTICLE_SNAPSHOT_DATASET = "mallard_particle_snapshot";

} // namespace

#ifdef Mallard_HAS_HDF5

using namespace h5;

namespace {

using curved::Vec3;

// The VTKHDF version whose features the files use (temporal data): ParaView 5.12 reads it
const std::vector<int64_t> VTKHDF_VERSION = {2, 0};
// That of the files of a mesh and its particles (a temporal PartitionedDataSetCollection): ParaView 6.0 reads it
const std::vector<int64_t> VTKHDF_COLLECTION_VERSION = {2, 1};

/**
 * @brief A VTKHDF PartitionedDataSetCollection of the mesh (index 0) and the particles (index 1), each a
 *        block of its own with its own steps; the mesh's block is created empty.
 */
void create_collection(hid_t file) {
    Handle gcpl(H5Pcreate(H5P_GROUP_CREATE), H5Pclose, "H5Pcreate");
    check(H5Pset_link_creation_order(gcpl, H5P_CRT_ORDER_TRACKED | H5P_CRT_ORDER_INDEXED), "H5Pset_link_creation_order");
    Handle top(H5Gcreate2(file, "VTKHDF", H5P_DEFAULT, gcpl, H5P_DEFAULT), H5Gclose, "creating VTKHDF");
    write_array_attribute(top, "Version", VTKHDF_COLLECTION_VERSION);
    write_string_scalar_attribute(top, "Type", "PartitionedDataSetCollection");
    Handle assembly(H5Gcreate2(top, "Assembly", H5P_DEFAULT, gcpl, H5P_DEFAULT), H5Gclose, "creating Assembly");
    check(H5Lcreate_soft("/" MESH_BLOCK, assembly, "mesh", H5P_DEFAULT, H5P_DEFAULT), "linking the mesh");
    check(H5Lcreate_soft("/" PARTICLE_BLOCK, assembly, "particles", H5P_DEFAULT, H5P_DEFAULT), "linking the particles");
    Handle mesh(H5Gcreate2(top, "mesh", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT), H5Gclose, "creating the mesh block");
    write_attribute<int64_t>(mesh, "Index", 0);
}

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
        {{0, 1}, {1, 2}, {3, 2}, {0, 3}, {4, 5}, {5, 6}, {7, 6}, {4, 7}, {0, 4}, {1, 5}, {2, 6}, {3, 7}},
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

void DataWriter::write_vtkhdf(const std::string & filename, const Block & snapshot, const Values & values) {
    const std::lock_guard<std::mutex> lock(h5::mutex());
    const bool distributed = mesh->n_global_cells > 0;
    if (!vtkhdf.built) build_vtkhdf_cells();
    const VtkhdfMesh & m = vtkhdf;
    const bool deflate = compression == Compression::ZLIB;
    const bool serial = !distributed;
    const bool append = one_file && vtkhdf_steps > 0;
    const uint64_t k = one_file ? vtkhdf_steps : 0;
    const uint64_t n_block = m.order.size();
    // A region that changes between snapshots: each step's mesh follows the previous ones' in the series
    const bool changing = one_file && subset_update;

    Handle fapl(file_access(distributed), H5Pclose, "H5Pcreate");
    Handle dxpl(transfer(distributed), H5Pclose, "H5Pcreate");
    const std::string & provenance = provenance::json();
    Handle fcpl(file_creation(fapl, provenance.size()), H5Pclose, "H5Pcreate");
    Handle file(append ? H5Fopen(filename.c_str(), H5F_ACC_RDWR, fapl)
                       : H5Fcreate(partial_name(filename).c_str(), H5F_ACC_TRUNC, fcpl, fapl),
                H5Fclose, (append ? "opening " : "creating ") + filename);
    if (append && !vtkhdf_run_written) {
        // A series continued by another run carries that run's block; each step's snapshot block names its run
        Handle root(H5Gopen2(file, "/", H5P_DEFAULT), H5Gclose, "H5Gopen2");
        for (const char * name : {provenance::ATTRIBUTE, provenance::LEGACY_ATTRIBUTE}) {
            if (H5Aexists(root, name) > 0) check(H5Adelete(root, name), "H5Adelete");
        }
        write_strings_attribute(root, provenance::ATTRIBUTE, {provenance});
    }
    vtkhdf_run_written = true;
    if (!append) {
        Handle root(H5Gopen2(file, "/", H5P_DEFAULT), H5Gclose, "H5Gopen2");
        write_strings_attribute(root, provenance::ATTRIBUTE, {provenance});
        write_int_attribute(root, "version", static_cast<int>(vtkhdf_version()));
        if (combined) create_collection(file);
        Handle top(combined ? H5Gopen2(file, MESH_BLOCK, H5P_DEFAULT)
                            : H5Gcreate2(file, "VTKHDF", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT),
                   H5Gclose, "creating VTKHDF");
        write_array_attribute(top, "Version", combined ? VTKHDF_COLLECTION_VERSION : VTKHDF_VERSION);
        write_string_scalar_attribute(top, "Type", "UnstructuredGrid");
        if (!changing) {
            write_count(top, "NumberOfPoints", m.n_points, dxpl, distributed);
            write_dataset(top, "Points", m.n_points, 3, m.first_point, m.points.size() / 3, m.points.data(), dxpl,
                          deflate, serial, io_threads);
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
            topology(top, m.n_cells, m.n_ids, true);
            write_dataset(top, "Types", m.n_cells, 1, m.first_cell, m.types.size(), m.types.data(), dxpl, deflate,
                          serial, io_threads);
        }
        for (const char * name : {"CellData", "PointData", "FieldData"}) {
            Handle group(H5Gcreate2(top, name, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT), H5Gclose, name);
        }
    }
    Handle top(H5Gopen2(file, combined ? MESH_BLOCK : "VTKHDF", H5P_DEFAULT), H5Gclose, "opening the mesh");
    {
        // The snapshot's time and step as field data (ParaView shows them), and its snapshot block
        const uint64_t mine = first_rank(distributed) ? 1 : 0;
        Handle field_data(H5Gopen2(top, "FieldData", H5P_DEFAULT), H5Gclose, "opening FieldData");
        const double time = snapshot.time;
        put_rows(field_data, "TIME", k + 1, 1, k, mine, &time, dxpl);
        if (!append) annotate(field_data, "TIME", FieldDescription{dim::time, "time", "time", ""});
        if (snapshot_blocks) {
            const int64_t step = snapshot.step;
            put_rows(field_data, "STEP", k + 1, 1, k, mine, &step, dxpl);
            if (!append) annotate(field_data, "STEP", FieldDescription{dim::none, "time step number", "", ""});
            const std::string text = snapshot.json();
            put_string_row(file, snapshot_block::DATASET, k + 1, k, mine, text,
                           snapshot_block::dataset_width(text.size()), dxpl);
        }
    }
    if (append) {
        Handle cells(H5Gopen2(top, ".", H5P_DEFAULT), H5Gclose, "opening the cells");
        bool same = changing || read_rows<int64_t>(cells, "NumberOfCells", 0, 1, 1, H5P_DEFAULT)[0] ==
                                    static_cast<int64_t>(m.n_cells);
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

    // Points, cells and connectivity ids of the steps before this one
    uint64_t points_before = 0, cells_before = 0, ids_before = 0;
    if (changing && k > 0) {
        auto sum = [&](const char * name) {
            uint64_t total = 0;
            for (const int64_t n : read_rows<int64_t>(top, name, 0, k, 1, H5P_DEFAULT)) total += static_cast<uint64_t>(n);
            return total;
        };
        points_before = sum("NumberOfPoints");
        cells_before = sum("NumberOfCells");
        ids_before = sum("NumberOfConnectivityIds");
    }
    const uint64_t data_before = changing ? cells_before : k * m.n_cells;
    if (changing) {
        const uint64_t mine = first_rank(distributed) ? 1 : 0;
        const int64_t counts[3] = {static_cast<int64_t>(m.n_points), static_cast<int64_t>(m.n_cells),
                                   static_cast<int64_t>(m.n_ids)};
        put_rows(top, "NumberOfPoints", k + 1, 1, k, mine, &counts[0], dxpl);
        put_rows(top, "NumberOfCells", k + 1, 1, k, mine, &counts[1], dxpl);
        put_rows(top, "NumberOfConnectivityIds", k + 1, 1, k, mine, &counts[2], dxpl);
        put_rows(top, "Points", points_before + m.n_points, 3, points_before + m.first_point, m.points.size() / 3,
                 m.points.data(), dxpl, deflate);
        // Each step's offsets have one more row than its cells
        put_rows(top, "Offsets", cells_before + k + m.n_cells + 1, 1, cells_before + k + m.first_cell,
                 m.offsets.size(), m.offsets.data(), dxpl, deflate);
        put_rows(top, "Connectivity", ids_before + m.n_ids, 1, ids_before + m.first_id, m.connectivity.size(),
                 m.connectivity.data(), dxpl, deflate);
        put_rows(top, "Types", cells_before + m.n_cells, 1, cells_before + m.first_cell, m.types.size(),
                 m.types.data(), dxpl, deflate);
        if (!append) annotate(top, "Points", FieldDescription{dim::length, "point coordinates", "", ""});
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
            put_rows(cell_data, name, data_before + m.n_cells, int(n_comp), data_before + m.first_cell, n_block,
                     block.data(), dxpl, deflate);
        } else {
            write_dataset(cell_data, name, m.n_cells, int(n_comp), m.first_cell, n_block, block.data(), dxpl, deflate,
                          serial, io_threads);
        }
        if (!append) {
            annotate(cell_data, fields[f].name, fields[f].description);
            h5::cell_methods(cell_data, fields[f].name, cell_methods(fields[f].name));
        }
    }
    if (!one_file) return;

    // Every snapshot reuses the mesh, its parts, points and cells all starting at 0, unless it changes
    Handle steps = open_or_create_group(top, "Steps");
    set_attribute<int64_t>(steps, "NSteps", static_cast<int64_t>(k + 1));
    const uint64_t mine = first_rank(distributed) ? 1 : 0;
    const double time = snapshot.time;
    const int64_t one = 1;
    put_rows(steps, "Values", k + 1, 1, k, mine, &time, dxpl);
    if (!append) annotate(steps, "Values", FieldDescription{dim::time, "time", "time", ""});
    const int64_t part = changing ? static_cast<int64_t>(k) : 0, point_offset = static_cast<int64_t>(points_before),
                  cell_offset = static_cast<int64_t>(cells_before), id_offset = static_cast<int64_t>(ids_before);
    put_rows(steps, "PartOffsets", k + 1, 1, k, mine, &part, dxpl);
    put_rows(steps, "NumberOfParts", k + 1, 1, k, mine, &one, dxpl);
    put_rows(steps, "PointOffsets", k + 1, 1, k, mine, &point_offset, dxpl);
    put_rows(steps, "CellOffsets", k + 1, 1, k, mine, &cell_offset, dxpl);
    put_rows(steps, "ConnectivityIdOffsets", k + 1, 1, k, mine, &id_offset, dxpl);
    open_or_create_group(steps, "PointDataOffsets");
    Handle field_offsets = open_or_create_group(steps, "FieldDataOffsets");
    const int64_t field_offset = static_cast<int64_t>(k);
    put_rows(field_offsets, "TIME", k + 1, 1, k, mine, &field_offset, dxpl);
    if (snapshot_blocks) put_rows(field_offsets, "STEP", k + 1, 1, k, mine, &field_offset, dxpl);
    Handle data_offsets = open_or_create_group(steps, "CellDataOffsets");
    const int64_t offset = static_cast<int64_t>(data_before);
    for (const auto & field : fields) put_rows(data_offsets, field.name.c_str(), k + 1, 1, k, mine, &offset, dxpl);
    vtkhdf_steps = k + 1;
}

void DataWriter::write_particles(const std::string & filename, const uint64_t k, const Block & block,
                                 const ParticleCloud & cloud) {
    const std::lock_guard<std::mutex> lock(h5::mutex());
    const bool distributed = mesh->n_global_cells > 0;
    const uint64_t mine = first_rank(distributed) ? 1 : 0;
    const uint64_t n = cloud.id.size();
    const auto [first, total] = scan(n, distributed);
    const char * const kinds[4] = {"Vertices", "Lines", "Polygons", "Strips"};
    const std::string & provenance = provenance::json();
    Handle fapl(file_access(distributed), H5Pclose, "H5Pcreate");
    Handle dxpl(transfer(distributed), H5Pclose, "H5Pcreate");
    if (k == 0 && !combined) {
        Handle fcpl(file_creation(fapl, provenance.size()), H5Pclose, "H5Pcreate");
        Handle created(H5Fcreate(partial_name(filename).c_str(), H5F_ACC_TRUNC, fcpl, fapl), H5Fclose,
                       "creating " + filename);
        Handle root(H5Gopen2(created, "/", H5P_DEFAULT), H5Gclose, "opening /");
        write_strings_attribute(root, provenance::ATTRIBUTE, {provenance});
        write_int_attribute(root, "version", static_cast<int>(file_format::PARTICLE_OUTPUT.version));
        write_string_scalar_attribute(root, "content", "particles");
        Handle top(H5Gcreate2(created, "VTKHDF", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT), H5Gclose, "VTKHDF");
        write_array_attribute(top, "Version", VTKHDF_VERSION);
        write_string_scalar_attribute(top, "Type", "PolyData");
        for (const char * name : {"Vertices", "Lines", "Polygons", "Strips", "CellData", "PointData", "FieldData"}) {
            Handle group(H5Gcreate2(top, name, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT), H5Gclose, name);
        }
        Handle steps(H5Gcreate2(top, "Steps", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT), H5Gclose, "Steps");
        for (const char * name : {"CellDataOffsets", "PointDataOffsets", "FieldDataOffsets"}) {
            Handle group(H5Gcreate2(steps, name, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT), H5Gclose, name);
        }
    }
    Handle file(H5Fopen((k == 0 ? partial_name(filename) : filename).c_str(), H5F_ACC_RDWR, fapl), H5Fclose,
                "opening " + filename);
    if (k == 0 && combined) {
        // The mesh's snapshot created the file and the collection
        Handle part(H5Gcreate2(file, PARTICLE_BLOCK, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT), H5Gclose,
                     "creating the particle block");
        write_attribute<int64_t>(part, "Index", 1);
        write_array_attribute(part, "Version", VTKHDF_COLLECTION_VERSION);
        write_string_scalar_attribute(part, "Type", "PolyData");
        for (const char * name : {"Vertices", "Lines", "Polygons", "Strips", "CellData", "PointData", "FieldData"}) {
            Handle group(H5Gcreate2(part, name, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT), H5Gclose, name);
        }
        Handle steps(H5Gcreate2(part, "Steps", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT), H5Gclose, "Steps");
        for (const char * name : {"CellDataOffsets", "PointDataOffsets", "FieldDataOffsets"}) {
            Handle group(H5Gcreate2(steps, name, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT), H5Gclose, name);
        }
    }
    if (k > 0 && !particles->run_written && !combined) {
        // A series continued by another run carries that run's block; each step's snapshot block names its run
        Handle root(H5Gopen2(file, "/", H5P_DEFAULT), H5Gclose, "opening /");
        for (const char * name : {provenance::ATTRIBUTE, provenance::LEGACY_ATTRIBUTE}) {
            if (H5Aexists(root, name) > 0) check(H5Adelete(root, name), "H5Adelete");
        }
        write_strings_attribute(root, provenance::ATTRIBUTE, {provenance});
    }
    particles->run_written = true;
    Handle top(H5Gopen2(file, combined ? PARTICLE_BLOCK : "VTKHDF", H5P_DEFAULT), H5Gclose, "opening the particles");
    {
        Handle field_data(H5Gopen2(top, "FieldData", H5P_DEFAULT), H5Gclose, "opening FieldData");
        put_rows(field_data, "TIME", k + 1, 1, k, mine, &block.time, dxpl);
        put_rows(field_data, "STEP", k + 1, 1, k, mine, &block.step, dxpl);
        if (k == 0) {
            annotate(field_data, "TIME", FieldDescription{dim::time, "time", "time", ""});
            annotate(field_data, "STEP", FieldDescription{dim::none, "time step number", "", ""});
        }
        const std::string text = block.json();
        put_string_row(file, combined ? PARTICLE_SNAPSHOT_DATASET : snapshot_block::DATASET, k + 1, k, mine, text,
                       snapshot_block::dataset_width(text.size()), dxpl);
    }
    // Every step has its own points, one vertex each: its rows follow the previous steps', and each rank's the
    // lower ranks'
    auto sum = [&](hid_t group, const char * name) {
        int64_t s = 0;
        if (k > 0) {
            for (const int64_t c : read_rows<int64_t>(group, name, 0, k, 1, H5P_DEFAULT)) s += c;
        }
        return s;
    };
    const int64_t points_before = sum(top, "NumberOfPoints");
    const uint64_t at = static_cast<uint64_t>(points_before);
    const int64_t n_points = static_cast<int64_t>(total);
    put_rows<int64_t>(top, "NumberOfPoints", k + 1, 1, k, mine, &n_points, dxpl);
    put_rows(top, "Points", at + total, 3, at + first, n, cloud.x.data(), dxpl);
    if (k == 0) annotate(top, "Points", FieldDescription{dim::length, "particle positions", "", ""});
    std::array<int64_t, 4> cells_before = {0, 0, 0, 0}, ids_before = {0, 0, 0, 0};
    for (size_t i = 0; i < 4; i++) {
        Handle group(H5Gopen2(top, kinds[i], H5P_DEFAULT), H5Gclose, kinds[i]);
        cells_before[i] = sum(group, "NumberOfCells");
        ids_before[i] = sum(group, "NumberOfConnectivityIds");
        const uint64_t n_cells = i == 0 ? total : 0;
        const int64_t counts[2] = {static_cast<int64_t>(n_cells), static_cast<int64_t>(n_cells)};
        put_rows<int64_t>(group, "NumberOfCells", k + 1, 1, k, mine, &counts[0], dxpl);
        put_rows<int64_t>(group, "NumberOfConnectivityIds", k + 1, 1, k, mine, &counts[1], dxpl);
        // Each step's offsets have one more row than its cells: the last rank writes it
        std::vector<int64_t> offsets, connectivity;
        if (i == 0) {
            for (uint64_t p = 0; p < n; p++) offsets.push_back(static_cast<int64_t>(first + p));
            connectivity = offsets;
        }
        if (last_rank(distributed)) offsets.push_back(static_cast<int64_t>(n_cells));
        const uint64_t base = static_cast<uint64_t>(cells_before[i]) + k;
        put_rows(group, "Offsets", base + n_cells + 1, 1, base + (i == 0 ? first : 0), offsets.size(), offsets.data(),
                 dxpl);
        const uint64_t ids = static_cast<uint64_t>(ids_before[i]);
        put_rows(group, "Connectivity", ids + n_cells, 1, ids + (i == 0 ? first : 0), connectivity.size(),
                 connectivity.data(), dxpl);
    }
    Handle point_data(H5Gopen2(top, "PointData", H5P_DEFAULT), H5Gclose, "opening PointData");
    const std::vector<int64_t> ids(cloud.id.begin(), cloud.id.end());
    put_rows(point_data, "ID", at + total, 1, at + first, n, ids.data(), dxpl);
    if (k == 0) annotate(point_data, "ID", *particle_field_description("ID"));
    const size_t n_columns = n > 0 ? cloud.values.size() / n : 0;
    std::vector<double> column;
    for (const auto & field : particles->fields) {
        const uint32_t n_comp = field.n_components == 1 ? 1 : 3;
        column.assign(n * n_comp, 0.0);
        for (uint64_t p = 0; p < n; p++) {
            for (uint32_t c = 0; c < field.n_components; c++) {
                column[p * n_comp + c] = cloud.values[p * n_columns + field.first + c];
            }
        }
        put_rows(point_data, field.name.c_str(), at + total, static_cast<int>(n_comp), at + first, n, column.data(),
                 dxpl);
        if (k == 0) {
            if (const auto d = particle_field_description(field.name)) annotate(point_data, field.name, *d);
            h5::cell_methods(point_data, field.name, N_DIM == 2 ? "area: point" : "volume: point");
        }
    }
    Handle steps(H5Gopen2(top, "Steps", H5P_DEFAULT), H5Gclose, "opening Steps");
    set_attribute<int64_t>(steps, "NSteps", static_cast<int64_t>(k + 1));
    const int64_t part = static_cast<int64_t>(k), one = 1;
    put_rows(steps, "Values", k + 1, 1, k, mine, &block.time, dxpl);
    if (k == 0) annotate(steps, "Values", FieldDescription{dim::time, "time", "time", ""});
    put_rows(steps, "PartOffsets", k + 1, 1, k, mine, &part, dxpl);
    put_rows(steps, "NumberOfParts", k + 1, 1, k, mine, &one, dxpl);
    put_rows(steps, "PointOffsets", k + 1, 1, k, mine, &points_before, dxpl);
    put_rows(steps, "CellOffsets", k + 1, 4, k, mine, cells_before.data(), dxpl);
    put_rows(steps, "ConnectivityIdOffsets", k + 1, 4, k, mine, ids_before.data(), dxpl);
    Handle field_offsets(H5Gopen2(steps, "FieldDataOffsets", H5P_DEFAULT), H5Gclose, "opening FieldDataOffsets");
    const int64_t field_offset = static_cast<int64_t>(k);
    for (const char * name : {"TIME", "STEP"}) put_rows(field_offsets, name, k + 1, 1, k, mine, &field_offset, dxpl);
    Handle point_offsets(H5Gopen2(steps, "PointDataOffsets", H5P_DEFAULT), H5Gclose, "opening PointDataOffsets");
    put_rows(point_offsets, "ID", k + 1, 1, k, mine, &points_before, dxpl);
    for (const auto & field : particles->fields) {
        put_rows(point_offsets, field.name.c_str(), k + 1, 1, k, mine, &points_before, dxpl);
    }
}

#else

void DataWriter::write_vtkhdf(const std::string &, const Block &, const Values &) {
    throw std::logic_error("DataWriter: VTKHDF output without HDF5");
}

void DataWriter::build_vtkhdf_cells() {}

void DataWriter::write_particles(const std::string &, uint64_t, const Block &, const ParticleCloud &) {
    throw std::logic_error("DataWriter: VTKHDF output without HDF5");
}

#endif

/**
 * @file data_writer_hdf5.cpp
 * @author Matthew Bonanni (mbonanni001@gmail.com)
 * @brief HDF5 solution output with XDMF indexes for ParaView.
 * @version 0.5
 * @date 2026-10-04
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 *
 */

#include "data_writer.h"

#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <mutex>
#include <sstream>

#include "comm.h"
#include "common_io.h"
#include "compression.h"
#include "file_format.h"
#include "hdf5_util.h"

namespace {

std::string mesh_file(const std::string & prefix) { return prefix + "_mesh.h5"; }

std::string file_name(const std::string & path) { return std::filesystem::path(path).filename().string(); }

void write_xdmf_file(const std::string & filename, const std::string & grids) {
    std::ofstream out(filename);
    if (!out.good()) throw std::runtime_error("DataWriter: Could not open file: " + filename + ".");
    out << "<?xml version=\"1.0\" ?>\n"
        << "<!DOCTYPE Xdmf SYSTEM \"Xdmf.dtd\" []>\n"
        << "<Xdmf Version=\"3.0\">\n"
        << "  <Domain>\n"
        << grids << "  </Domain>\n"
        << "</Xdmf>\n";
}

} // namespace

std::string DataWriter::xdmf_grid(const std::string & stem, double t, const std::string & indent) const {
    const std::string mesh_h5 = file_name(mesh_file(prefix));
    const std::string data_h5 = file_name(stem) + ".h5";
    const std::string real = "NumberType=\"Float\" Precision=\"" + std::to_string(sizeof(rtype)) + "\" Format=\"HDF\"";
    const uint64_t n_cells = cell_order.n_global();
    std::ostringstream out;
    out << std::setprecision(std::numeric_limits<double>::max_digits10);
    out << indent << "<Grid Name=\"" << file_name(stem) << "\" GridType=\"Uniform\">\n"
        << indent << "  <Time Value=\"" << t << "\"/>\n"
        << indent << "  <Topology TopologyType=\"Mixed\" NumberOfElements=\"" << n_cells << "\">\n"
        << indent << "    <DataItem Dimensions=\"" << hdf5_topology_size
        << "\" NumberType=\"Int\" Precision=\"8\" Format=\"HDF\">" << mesh_h5 << ":/cells/topology</DataItem>\n"
        << indent << "  </Topology>\n"
        << indent << "  <Geometry GeometryType=\"" << (N_DIM == 2 ? "XY" : "XYZ") << "\">\n"
        << indent << "    <DataItem Dimensions=\"" << hdf5_n_nodes << " " << N_DIM << "\" " << real << ">" << mesh_h5
        << ":/nodes/coordinates</DataItem>\n"
        << indent << "  </Geometry>\n";
    for (const auto & field : fields) {
        const bool is_vector = field.n_vtk_components() > 1;
        out << indent << "  <Attribute Name=\"" << field.name << "\" AttributeType=\""
            << (is_vector ? "Vector" : "Scalar") << "\" Center=\"Cell\">\n"
            << indent << "    <DataItem Dimensions=\"" << n_cells << (is_vector ? " 3" : "") << "\" " << real << ">"
            << data_h5 << ":/fields/" << field.name << "</DataItem>\n"
            << indent << "  </Attribute>\n";
    }
    out << indent << "</Grid>\n";
    return out.str();
}

void DataWriter::write_xdmf(const std::string & stem, const std::vector<std::pair<rtype, std::string>> & series) const {
    write_xdmf_file(stem + ".xmf", xdmf_grid(stem, double(series.back().first), "    "));
    std::string grids =
        "    <Grid Name=\"" + file_name(prefix) + "\" GridType=\"Collection\" CollectionType=\"Temporal\">\n";
    for (const auto & [t, file] : series) {
        grids += xdmf_grid(std::filesystem::path(file).replace_extension().string(), double(t), "      ");
    }
    grids += "    </Grid>\n";
    write_xdmf_file(prefix + ".xmf", grids);
}

void DataWriter::resume_xdmf(rtype t) {
    // Keep the snapshots of the previous run's series up to the restart time
    std::ifstream in(prefix + ".xmf");
    const std::filesystem::path dir = std::filesystem::path(prefix).parent_path();
    std::string line, stem;
    while (std::getline(in, line)) {
        const size_t g = line.find("<Grid Name=\"");
        if (g != std::string::npos && line.find("GridType=\"Uniform\"") != std::string::npos) {
            const size_t g0 = g + 12;
            stem = line.substr(g0, line.find('"', g0) - g0);
            continue;
        }
        const size_t a = line.find("<Time Value=\"");
        if (a == std::string::npos || stem.empty()) continue;
        const double t_entry = std::stod(line.substr(a + 13));
        if (t_entry <= static_cast<double>(t) * (1.0 + 1.0e-12) + 1.0e-300) {
            history.emplace_back(t_entry, (dir / (stem + ".h5")).string());
        }
        stem.clear();
    }
}

#ifdef Mallard_HAS_HDF5

using namespace h5;

namespace {

constexpr const char * MESH_FORMAT = "mallard-solution-mesh";
constexpr const char * SOLUTION_FORMAT = "mallard-solution";

// The HDF5 library is not thread-safe, and asynchronous writers call it from their threads
std::mutex hdf5_mutex;

/** @brief XDMF code of a cell in a Mixed topology, by node count. */
uint64_t xdmf_type(uint32_t n_nodes) {
    if (N_DIM == 2 && n_nodes == 3) return 4;  // Triangle
    if (N_DIM == 2 && n_nodes == 4) return 5;  // Quadrilateral
    if (N_DIM == 3 && n_nodes == 4) return 6;  // Tetrahedron
    if (N_DIM == 3 && n_nodes == 5) return 7;  // Pyramid
    if (N_DIM == 3 && n_nodes == 6) return 8;  // Wedge
    if (N_DIM == 3 && n_nodes == 8) return 9;  // Hexahedron
    throw std::runtime_error("DataWriter: no XDMF cell type for a " + std::to_string(N_DIM) + "D cell with " +
                             std::to_string(n_nodes) + " nodes.");
}

/**
 * @brief write_rows: compressed with HDF5's shuffle and deflate filters, which
 *        compress on one thread, or, in a file of this rank alone (serial),
 *        with the same chunks compressed on n_threads threads and written
 *        directly.
 */
template <typename T>
void write_dataset(hid_t parent, const char * name, uint64_t n_rows, int cols, uint64_t first, uint64_t n,
                   const T * data, hid_t dxpl, bool deflate, bool serial, unsigned n_threads) {
    if (!(deflate && serial && compression::have_zlib() && n_rows > 0 && first == 0 && n == n_rows)) {
        write_rows(parent, name, n_rows, cols, first, n, data, dxpl, deflate);
        return;
    }
    const int rank = cols > 1 ? 2 : 1;
    const hsize_t dims[2] = {n_rows, hsize_t(cols)};
    const hsize_t chunk_rows = std::min<hsize_t>(n_rows, CHUNK_ROWS);
    const hsize_t chunk[2] = {chunk_rows, hsize_t(cols)};
    Handle space(H5Screate_simple(rank, dims, nullptr), H5Sclose, "H5Screate_simple");
    Handle dcpl(H5Pcreate(H5P_DATASET_CREATE), H5Pclose, "H5Pcreate");
    check(H5Pset_chunk(dcpl, rank, chunk), "H5Pset_chunk");
    check(H5Pset_shuffle(dcpl), "H5Pset_shuffle");
    check(H5Pset_deflate(dcpl, 1), "H5Pset_deflate");
    Handle dset(H5Dcreate2(parent, name, file_type<T>(), space, H5P_DEFAULT, dcpl, H5P_DEFAULT), H5Dclose,
                std::string("creating ") + name);
    // Chunks are stored whole: the last one is padded with zeros (HDF5's fill value)
    const size_t chunk_values = size_t(chunk_rows) * size_t(cols);
    std::vector<std::vector<char>> chunks((n_rows + chunk_rows - 1) / chunk_rows);
    compression::parallel_for(chunks.size(), n_threads, [&](size_t c) {
        std::vector<T> values(chunk_values, T(0));
        const size_t at = c * chunk_values;
        std::copy(data + at, data + std::min(at + chunk_values, size_t(n_rows) * size_t(cols)), values.begin());
        chunks[c] = compression::deflate(values.data(), values.size() * sizeof(T), sizeof(T));
    });
    for (size_t c = 0; c < chunks.size(); c++) {
        const hsize_t offset[2] = {c * chunk_rows, 0};
        check(H5Dwrite_chunk(dset, dxpl, 0, offset, chunks[c].size(), chunks[c].data()), std::string("writing ") + name);
    }
}

} // namespace

bool have_hdf5_deflate() { return H5Zfilter_avail(H5Z_FILTER_DEFLATE) > 0 && H5Zfilter_avail(H5Z_FILTER_SHUFFLE) > 0; }

// Serial runs on several ranks (tests) hold the whole mesh on every rank: the root writes alone
void DataWriter::write_hdf5_mesh() {
    const std::lock_guard<std::mutex> lock(hdf5_mutex);
    hdf5_mesh_written = true;
    const bool distributed = mesh->n_global_cells > 0;
    const uint32_t n_owned = mesh->n_owned();
    const uint64_t n_cells = distributed ? mesh->n_global_cells : mesh->n_cells;
    const uint64_t n_nodes = distributed ? mesh->n_global_nodes : mesh->n_nodes;
    auto global_node = [&](uint32_t n) { return distributed ? mesh->h_global_node_id[n] : uint64_t(n); };

    // Owned cells as XDMF Mixed records (type, then nodes in VTK order), and the nodes they use
    std::vector<uint64_t> cell_ids(n_owned), offsets{0}, topology;
    std::vector<bool> used(mesh->n_nodes, false);
    for (uint32_t i = 0; i < n_owned; i++) {
        cell_ids[i] = mesh->h_global_cell(i);
        const uint32_t n = mesh->h_n_nodes_of_cell(i);
        topology.push_back(xdmf_type(n));
        for (uint32_t k = 0; k < n; k++) {
            const uint32_t node = mesh->h_node_of_cell(i, vtk_local_node(n, k));
            topology.push_back(global_node(node));
            used[node] = true;
        }
        offsets.push_back(topology.size());
    }
    cell_order = GlobalOrder(cell_ids, n_cells, distributed);
    const auto [block_offsets, block_topology] = cell_order.gather_csr(offsets, topology);

    std::vector<uint64_t> node_ids;
    std::vector<rtype> coords;
    for (uint32_t n = 0; n < mesh->n_nodes; n++) {
        if (!used[n]) continue;
        node_ids.push_back(global_node(n));
        FOR_I_DIM coords.push_back(mesh->h_node_coords(n, i));
    }
    const GlobalOrder node_order(node_ids, n_nodes, distributed);
    const std::vector<rtype> block_coords = node_order.gather<rtype>(coords, N_DIM);

    // This rank's records follow those of the lower ranks; the last rank closes the offsets
    uint64_t first = 0, total = block_topology.size();
    bool last = true;
    if (distributed) {
        const std::vector<uint64_t> sizes = comm::allgatherv(std::vector<uint64_t>{total});
        total = 0;
        for (int r = 0; r < comm::size(); r++) {
            if (r < comm::rank()) first += sizes[static_cast<size_t>(r)];
            total += sizes[static_cast<size_t>(r)];
        }
        last = comm::rank() == comm::size() - 1;
    }
    hdf5_n_nodes = n_nodes;
    hdf5_topology_size = total;
    std::vector<int64_t> starts(cell_order.size() + (last ? 1 : 0));
    for (size_t i = 0; i < starts.size(); i++) starts[i] = int64_t(first + block_offsets[i]);
    const std::vector<int64_t> records(block_topology.begin(), block_topology.end());

    const bool deflate = compression == Compression::ZLIB;
    const std::string filename = mesh_file(prefix);
    Handle fapl(file_access(distributed), H5Pclose, "H5Pcreate");
    Handle dxpl(transfer(distributed), H5Pclose, "H5Pcreate");
    Handle file(H5Fcreate(filename.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, fapl), H5Fclose, "creating " + filename);
    {
        Handle root(H5Gopen2(file, "/", H5P_DEFAULT), H5Gclose, "H5Gopen2");
        write_strings_attribute(root, "format", {MESH_FORMAT});
        write_int_attribute(root, "version", static_cast<int>(file_format::HDF5_OUTPUT.version));
        write_int_attribute(root, "dimension", N_DIM);
    }
    {
        Handle group(H5Gcreate2(file, "nodes", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT), H5Gclose, "H5Gcreate2");
        write_dataset(group, "coordinates", n_nodes, N_DIM, node_order.first(), node_order.size(), block_coords.data(),
                      dxpl, deflate, !distributed, io_threads);
    }
    {
        Handle group(H5Gcreate2(file, "cells", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT), H5Gclose, "H5Gcreate2");
        write_dataset(group, "offsets", n_cells + 1, 1, cell_order.first(), starts.size(), starts.data(), dxpl, deflate,
                      !distributed, io_threads);
        write_dataset(group, "topology", total, 1, first, records.size(), records.data(), dxpl, deflate, !distributed,
                      io_threads);
    }
}

void DataWriter::write_hdf5(const std::string & stem, uint64_t step, rtype t, const Values & snapshot) {
    const std::lock_guard<std::mutex> lock(hdf5_mutex);
    const bool distributed = mesh->n_global_cells > 0;
    const uint32_t n_owned = mesh->n_owned();
    uint32_t width = 0;
    for (const auto & field : fields) width += field.n_vtk_components();
    std::vector<rtype> values(size_t(n_owned) * width);
    size_t at = 0;
    for (uint32_t i = 0; i < n_owned; i++) {
        for (size_t f = 0; f < fields.size(); f++) {
            const uint32_t n_comp = fields[f].n_vtk_components();
            for (uint32_t k = 0; k < n_comp; k++) values[at++] = snapshot[f][size_t(i) * n_comp + k];
        }
    }
    if (relative_error > 0.0) compression::round_to_relative_error(values, relative_error);
    const std::vector<rtype> block = cell_order.gather<rtype>(values, width);
    const uint64_t n = cell_order.size();
    const bool deflate = compression == Compression::ZLIB;
    const std::string filename = stem + ".h5";
    Handle fapl(file_access(distributed), H5Pclose, "H5Pcreate");
    Handle dxpl(transfer(distributed), H5Pclose, "H5Pcreate");
    Handle file(H5Fcreate(filename.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, fapl), H5Fclose, "creating " + filename);
    {
        Handle root(H5Gopen2(file, "/", H5P_DEFAULT), H5Gclose, "H5Gopen2");
        write_strings_attribute(root, "format", {SOLUTION_FORMAT});
        write_int_attribute(root, "version", static_cast<int>(file_format::HDF5_OUTPUT.version));
        write_int_attribute(root, "dimension", N_DIM);
        write_attribute(root, "step", step);
        write_attribute(root, "time", double(t));
        write_strings_attribute(root, "mesh", {file_name(mesh_file(prefix))});
    }
    Handle group(H5Gcreate2(file, "fields", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT), H5Gclose, "H5Gcreate2");
    std::vector<rtype> column;
    uint32_t c0 = 0;
    for (const auto & field : fields) {
        const uint32_t n_comp = field.n_vtk_components();
        column.resize(n * n_comp);
        for (uint64_t row = 0; row < n; row++) {
            for (uint32_t k = 0; k < n_comp; k++) column[row * n_comp + k] = block[row * width + c0 + k];
        }
        write_dataset(group, field.name.c_str(), cell_order.n_global(), int(n_comp), cell_order.first(), n,
                      column.data(), dxpl, deflate, !distributed, io_threads);
        c0 += n_comp;
    }
}

#else

bool have_hdf5_deflate() { return false; }

void DataWriter::write_hdf5_mesh() { throw std::logic_error("DataWriter: HDF5 output without HDF5"); }

void DataWriter::write_hdf5(const std::string &, uint64_t, rtype, const Values &) {
    throw std::logic_error("DataWriter: HDF5 output without HDF5");
}

#endif

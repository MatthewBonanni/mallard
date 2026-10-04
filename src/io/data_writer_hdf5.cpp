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
#include <sstream>

#include "comm.h"
#include "common_io.h"
#include "hdf5_util.h"

namespace {

constexpr const char * MESH_FORMAT = "mallard-solution-mesh";
constexpr const char * SOLUTION_FORMAT = "mallard-solution";
constexpr int OUTPUT_VERSION = 1;

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
        const bool vector = field.n_vtk_components() > 1;
        out << indent << "  <Attribute Name=\"" << field.name << "\" AttributeType=\"" << (vector ? "Vector" : "Scalar")
            << "\" Center=\"Cell\">\n"
            << indent << "    <DataItem Dimensions=\"" << n_cells << (vector ? " 3" : "") << "\" " << real << ">"
            << data_h5 << ":/fields/" << field.name << "</DataItem>\n"
            << indent << "  </Attribute>\n";
    }
    out << indent << "</Grid>\n";
    return out.str();
}

void DataWriter::write_xdmf(const std::string & stem) const {
    write_xdmf_file(stem + ".xmf", xdmf_grid(stem, double(history.back().first), "    "));
    std::string grids = "    <Grid Name=\"" + file_name(prefix) + "\" GridType=\"Collection\" CollectionType=\"Temporal\">\n";
    for (const auto & [t, file] : history) {
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

void DataWriter::write_hdf5_mesh() {
    hdf5_mesh_written = true;
    const bool distributed = mesh->n_global_cells > 0;
    // Serial runs on several ranks (tests) hold the whole mesh on every rank
    if (!distributed && !comm::is_root()) return;
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
            if (r < comm::rank()) first += sizes[r];
            total += sizes[r];
        }
        last = comm::rank() == comm::size() - 1;
    }
    hdf5_n_nodes = n_nodes;
    hdf5_topology_size = total;
    std::vector<int64_t> starts(cell_order.size() + (last ? 1 : 0));
    for (size_t i = 0; i < starts.size(); i++) starts[i] = int64_t(first + block_offsets[i]);
    const std::vector<int64_t> records(block_topology.begin(), block_topology.end());

    Handle fapl(file_access(distributed), H5Pclose, "H5Pcreate");
    Handle dxpl(transfer(distributed), H5Pclose, "H5Pcreate");
    const std::string filename = mesh_file(prefix);
    Handle file(H5Fcreate(filename.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, fapl), H5Fclose, "creating " + filename);
    {
        Handle root(H5Gopen2(file, "/", H5P_DEFAULT), H5Gclose, "H5Gopen2");
        write_strings_attribute(root, "format", {MESH_FORMAT});
        write_int_attribute(root, "version", OUTPUT_VERSION);
        write_int_attribute(root, "dimension", N_DIM);
    }
    {
        Handle group(H5Gcreate2(file, "nodes", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT), H5Gclose, "H5Gcreate2");
        write_rows(group, "coordinates", n_nodes, N_DIM, node_order.first(), node_order.size(), block_coords.data(),
                   dxpl);
    }
    {
        Handle group(H5Gcreate2(file, "cells", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT), H5Gclose, "H5Gcreate2");
        write_rows(group, "offsets", n_cells + 1, 1, cell_order.first(), starts.size(), starts.data(), dxpl);
        write_rows(group, "topology", total, 1, first, records.size(), records.data(), dxpl);
    }
}

void DataWriter::write_hdf5(const std::string & stem, uint64_t step, rtype t) const {
    const bool distributed = mesh->n_global_cells > 0;
    if (!distributed && !comm::is_root()) return;
    const uint32_t n_owned = mesh->n_owned();
    uint32_t width = 0;
    for (const auto & field : fields) width += field.n_vtk_components();
    std::vector<rtype> values(size_t(n_owned) * width);
    for (uint32_t i = 0, at = 0; i < n_owned; i++) {
        for (const auto & field : fields) {
            for (uint32_t k = 0; k < field.n_vtk_components(); k++) values[at++] = field.value(i, k);
        }
    }
    const std::vector<rtype> block = cell_order.gather<rtype>(values, width);
    const uint64_t n = cell_order.size();

    Handle fapl(file_access(distributed), H5Pclose, "H5Pcreate");
    Handle dxpl(transfer(distributed), H5Pclose, "H5Pcreate");
    const std::string filename = stem + ".h5";
    Handle file(H5Fcreate(filename.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, fapl), H5Fclose, "creating " + filename);
    {
        Handle root(H5Gopen2(file, "/", H5P_DEFAULT), H5Gclose, "H5Gopen2");
        write_strings_attribute(root, "format", {SOLUTION_FORMAT});
        write_int_attribute(root, "version", OUTPUT_VERSION);
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
        write_rows(group, field.name.c_str(), cell_order.n_global(), int(n_comp), cell_order.first(), n,
                   column.data(), dxpl);
        c0 += n_comp;
    }
}

#else

void DataWriter::write_hdf5_mesh() { throw std::logic_error("DataWriter: HDF5 output without HDF5"); }

void DataWriter::write_hdf5(const std::string &, uint64_t, rtype) const {
    throw std::logic_error("DataWriter: HDF5 output without HDF5");
}

#endif

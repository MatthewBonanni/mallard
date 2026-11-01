/**
 * @file case_view.cpp
 * @author Matthew Bonanni (mbonanni001@gmail.com)
 * @brief The case view's ParaView file.
 * @version 0.1
 * @date 2026-10-30
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 *
 */

#include "case_view.h"

#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <stdexcept>

#include "comm.h"
#include "common_io.h"
#ifdef Mallard_HAS_HDF5
#include "hdf5_util.h"
#include "provenance.h"
#endif

namespace case_view {

void Block::add_cell(const uint8_t type, const std::vector<std::array<double, 3>> & vertices) {
    for (const auto & x : vertices) {
        connectivity.push_back(static_cast<int64_t>(points.size()));
        points.push_back(x);
    }
    offsets.push_back(static_cast<int64_t>(connectivity.size()));
    types.push_back(type);
}

void Block::add_cell(const uint8_t type, const std::vector<int64_t> & ids) {
    connectivity.insert(connectivity.end(), ids.begin(), ids.end());
    offsets.push_back(static_cast<int64_t>(connectivity.size()));
    types.push_back(type);
}

Block::Field & Block::field(const std::string & field_name, const int components) {
    for (Field & f : fields) {
        if (f.name == field_name) return f;
    }
    fields.push_back({field_name, components, {}});
    return fields.back();
}

namespace {

/** @brief Every rank's part of a block, in rank order (on every rank). */
Block gather(const Block & local) {
    if (comm::size() == 1) return local;
    Block all;
    all.name = local.name;
    std::vector<double> xyz;
    for (const auto & x : local.points) xyz.insert(xyz.end(), x.begin(), x.end());
    std::vector<double> sizes = {double(local.points.size()), double(local.n_cells()), double(local.connectivity.size())};
    const std::vector<double> all_sizes = comm::allgatherv(sizes);
    const std::vector<double> all_xyz = comm::allgatherv(xyz);
    for (size_t i = 0; i < all_xyz.size(); i += 3) all.points.push_back({all_xyz[i], all_xyz[i + 1], all_xyz[i + 2]});
    std::vector<uint64_t> ids(local.connectivity.begin(), local.connectivity.end());
    std::vector<uint64_t> types(local.types.begin(), local.types.end());
    const std::vector<uint64_t> all_ids = comm::allgatherv(ids), all_types = comm::allgatherv(types);
    size_t id = 0, point = 0;
    for (size_t r = 0; r < all_sizes.size() / 3; r++) {
        const auto n_points = static_cast<size_t>(all_sizes[3 * r]), n_ids = static_cast<size_t>(all_sizes[3 * r + 2]);
        for (size_t k = 0; k < n_ids; k++) all.connectivity.push_back(static_cast<int64_t>(all_ids[id + k] + point));
        id += n_ids;
        point += n_points;
    }
    // Offsets from the gathered cell sizes
    std::vector<uint64_t> counts;
    for (size_t c = 0; c < local.n_cells(); c++) counts.push_back(uint64_t(local.offsets[c + 1] - local.offsets[c]));
    for (const uint64_t n : comm::allgatherv(counts)) all.offsets.push_back(all.offsets.back() + int64_t(n));
    for (const uint64_t t : all_types) all.types.push_back(static_cast<uint8_t>(t));
    for (const Block::Field & f : local.fields) all.fields.push_back({f.name, f.components, comm::allgatherv(f.values)});
    return all;
}

#ifndef Mallard_HAS_HDF5
void write_vtu(const std::string & path, const Block & b) {
    std::ofstream out(path);
    out << std::setprecision(std::numeric_limits<double>::max_digits10);
    out << "<?xml version=\"1.0\"?>\n<VTKFile type=\"UnstructuredGrid\" version=\"1.0\" byte_order=\"LittleEndian\" "
           "header_type=\"UInt64\">\n<UnstructuredGrid>\n<Piece NumberOfPoints=\""
        << b.points.size() << "\" NumberOfCells=\"" << b.n_cells() << "\">\n<Points>\n"
        << "<DataArray type=\"Float64\" NumberOfComponents=\"3\" format=\"ascii\">\n";
    for (const auto & x : b.points) out << x[0] << " " << x[1] << " " << x[2] << "\n";
    out << "</DataArray>\n</Points>\n<Cells>\n<DataArray type=\"Int64\" Name=\"connectivity\" format=\"ascii\">\n";
    for (const int64_t i : b.connectivity) out << i << " ";
    out << "\n</DataArray>\n<DataArray type=\"Int64\" Name=\"offsets\" format=\"ascii\">\n";
    for (size_t c = 1; c < b.offsets.size(); c++) out << b.offsets[c] << " ";
    out << "\n</DataArray>\n<DataArray type=\"UInt8\" Name=\"types\" format=\"ascii\">\n";
    for (const uint8_t t : b.types) out << int(t) << " ";
    out << "\n</DataArray>\n</Cells>\n<CellData>\n";
    for (const Block::Field & f : b.fields) {
        out << "<DataArray type=\"Float64\" Name=\"" << f.name << "\" NumberOfComponents=\"" << f.components
            << "\" format=\"ascii\">\n";
        for (const double v : f.values) out << v << " ";
        out << "\n</DataArray>\n";
    }
    out << "</CellData>\n</Piece>\n</UnstructuredGrid>\n</VTKFile>\n";
    if (!out) throw cannot_write("the case view", path);
}
#endif

} // namespace

std::string write(const std::string & path, const std::vector<Block> & local) {
    std::vector<Block> blocks;
    for (const Block & b : local) blocks.push_back(gather(b));
    if (!comm::is_root()) return path;
    create_parent_directories("the case view", path);
#ifdef Mallard_HAS_HDF5
    using namespace h5;
    const std::lock_guard<std::mutex> lock(h5::mutex());
    Handle fapl(file_access(false), H5Pclose, "H5Pcreate");
    Handle dxpl(transfer(false), H5Pclose, "H5Pcreate");
    const std::string & provenance = provenance::json();
    Handle fcpl(file_creation(fapl, provenance.size()), H5Pclose, "H5Pcreate");
    Handle file(H5Fcreate(path.c_str(), H5F_ACC_TRUNC, fcpl, fapl), H5Fclose, "creating " + path);
    {
        Handle root(H5Gopen2(file, "/", H5P_DEFAULT), H5Gclose, "H5Gopen2");
        write_strings_attribute(root, "provenance", {provenance});
    }
    // VTK reads the blocks and the assembly in the order of creation
    Handle gcpl(H5Pcreate(H5P_GROUP_CREATE), H5Pclose, "H5Pcreate");
    check(H5Pset_link_creation_order(gcpl, H5P_CRT_ORDER_TRACKED | H5P_CRT_ORDER_INDEXED), "H5Pset_link_creation_order");
    Handle top(H5Gcreate2(file, "VTKHDF", H5P_DEFAULT, gcpl, H5P_DEFAULT), H5Gclose, "creating VTKHDF");
    write_array_attribute(top, "Version", std::vector<int64_t>{2, 2});
    write_string_scalar_attribute(top, "Type", "PartitionedDataSetCollection");
    Handle assembly(H5Gcreate2(top, "Assembly", H5P_DEFAULT, gcpl, H5P_DEFAULT), H5Gclose, "creating Assembly");
    for (size_t i = 0; i < blocks.size(); i++) {
        const Block & b = blocks[i];
        Handle group(H5Gcreate2(top, b.name.c_str(), H5P_DEFAULT, gcpl, H5P_DEFAULT), H5Gclose, "creating " + b.name);
        write_array_attribute(group, "Version", std::vector<int64_t>{2, 2});
        write_string_scalar_attribute(group, "Type", "UnstructuredGrid");
        write_attribute<int64_t>(group, "Index", static_cast<int64_t>(i));
        const uint64_t n_points = b.points.size(), n_cells = b.n_cells(), n_ids = b.connectivity.size();
        const int64_t counts[3] = {int64_t(n_points), int64_t(n_cells), int64_t(n_ids)};
        write_rows<int64_t>(group, "NumberOfPoints", 1, 1, 0, 1, &counts[0], dxpl);
        write_rows<int64_t>(group, "NumberOfCells", 1, 1, 0, 1, &counts[1], dxpl);
        write_rows<int64_t>(group, "NumberOfConnectivityIds", 1, 1, 0, 1, &counts[2], dxpl);
        write_rows<double>(group, "Points", n_points, 3, 0, n_points, n_points ? b.points[0].data() : nullptr, dxpl);
        write_rows<int64_t>(group, "Offsets", n_cells + 1, 1, 0, n_cells + 1, b.offsets.data(), dxpl);
        write_rows<int64_t>(group, "Connectivity", n_ids, 1, 0, n_ids, b.connectivity.data(), dxpl);
        write_rows<uint8_t>(group, "Types", n_cells, 1, 0, n_cells, b.types.data(), dxpl);
        Handle cell_data(H5Gcreate2(group, "CellData", H5P_DEFAULT, gcpl, H5P_DEFAULT), H5Gclose, "CellData");
        for (const Block::Field & f : b.fields) {
            write_rows<double>(cell_data, f.name.c_str(), n_cells, f.components, 0, n_cells, f.values.data(), dxpl);
        }
        for (const char * name : {"PointData", "FieldData"}) {
            Handle empty(H5Gcreate2(group, name, H5P_DEFAULT, gcpl, H5P_DEFAULT), H5Gclose, name);
        }
        Handle node(H5Gcreate2(assembly, b.name.c_str(), H5P_DEFAULT, gcpl, H5P_DEFAULT), H5Gclose, b.name);
        check(H5Lcreate_soft(("/VTKHDF/" + b.name).c_str(), node, b.name.c_str(), H5P_DEFAULT, H5P_DEFAULT),
              "linking " + b.name);
    }
    return path;
#else
    const std::filesystem::path vtm = std::filesystem::path(path).replace_extension(".vtm");
    const std::filesystem::path dir = vtm.parent_path() / (vtm.stem().string() + "_blocks");
    std::filesystem::create_directories(dir);
    std::ofstream out(vtm);
    out << "<?xml version=\"1.0\"?>\n<VTKFile type=\"vtkMultiBlockDataSet\" version=\"1.0\">\n<vtkMultiBlockDataSet>\n";
    for (size_t i = 0; i < blocks.size(); i++) {
        const std::string piece = blocks[i].name + ".vtu";
        write_vtu((dir / piece).string(), blocks[i]);
        out << "<DataSet index=\"" << i << "\" name=\"" << blocks[i].name << "\" file=\"" << dir.filename().string() << "/"
            << piece << "\"/>\n";
    }
    out << "</vtkMultiBlockDataSet>\n</VTKFile>\n";
    if (!out) throw cannot_write("the case view", vtm.string());
    return vtm.string();
#endif
}

} // namespace case_view

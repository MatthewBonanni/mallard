/**
 * @file data_writer.cpp
 * @author Matthew Bonanni (mbonanni001@gmail.com)
 * @brief DataWriter class implementation.
 * @version 0.1
 * @date 2024-01-01
 * 
 * @copyright Copyright (c) 2024 Matthew Bonanni
 * 
 */

#include "data_writer.h"

#include <iostream>
#include <string>
#include <iomanip>
#include <optional>

#include "common_io.h"

DataWriter::DataWriter() {
    // Empty
}

DataWriter::~DataWriter() {
    // Empty
}

void DataWriter::init(const toml::value & input,
                      std::vector<Data> & data,
                      std::shared_ptr<Mesh> mesh) {
    if (!input.contains("prefix")) {
        throw std::runtime_error("DataWriter: prefix not specified.");
    }
    if (!input.contains("format")) {
        throw std::runtime_error("DataWriter: format not specified.");
    }
    if (!input.contains("interval")) {
        throw std::runtime_error("DataWriter: interval not specified.");
    }
    if (!input.contains("variables")) {
        throw std::runtime_error("DataWriter: variables not specified.");
    }

    this->prefix = toml::find<std::string>(input, "prefix");
    this->interval = toml::find<u_int32_t>(input, "interval");
    std::string format_str = toml::find<std::string>(input, "format");
    std::vector<std::string> variables = toml::find<std::vector<std::string>>(input, "variables");

    if (variables.empty()) {
        throw std::runtime_error("DataWriter: No variables specified.");
    }

    typename std::unordered_map<std::string, DataFormat>::const_iterator it = FORMAT_TYPES.find(format_str);
    if (it == FORMAT_TYPES.end()) {
        throw std::runtime_error("DataWriter: Unknown format type: " + format_str + ".");
    } else {
        format = it->second;
    }

    for (const auto & var : variables) {
        bool found = false;
        for (const auto & data_var : data) {
            if (data_var.name() == var) {
                data_ptrs.push_back(&data_var);
                found = true;
                break;
            }
        }
        if (!found) {
            throw std::runtime_error("DataWriter: Unknown variable: " + var + ".");
        }
    }

    this->mesh = mesh;
}

void DataWriter::write(u_int32_t step, bool force) const {
    bool write_now = (step % interval == 0) || force;
    if (!write_now) {
        return;
    }

    if (format == DataFormat::VTU) {
        write_vtu(step);
    } else if (format == DataFormat::TECPLOT) {
        write_tecplot(step);
    } else {
        throw std::runtime_error("DataWriter::write not implemented.");
    }
}

void DataWriter::write_vtu(u_int32_t step) const {
    std::string filename;
    std::ostringstream stream;
    stream << std::setw(LEN_STEP) << std::setfill('0') << step;
    filename = prefix + "_" + stream.str() + ".vtu";

    std::ofstream out(filename);
    if (!out.is_open()) {
        throw std::runtime_error("DataWriter::write_vtu: Could not open file: " + filename + ".");
    }
    if (!out.good()) {
        throw std::runtime_error("DataWriter::write_vtu: Could not write to file: " + filename + ".");
    }

    std::cout << "Writing data to file: " << filename << std::endl;

    u_int64_t offset_data = 0;
    u_int64_t len_connectivity = 0;

    // ---------------------------------------------------------------------------------------------
    // Write header
    out << "<?xml version=\"1.0\"?>\n";
    out << "<VTKFile type=\"UnstructuredGrid\" version=\"0.1\" byte_order=\"" << endianness() << "\">\n";
    out << "  <UnstructuredGrid>\n";
    out << "    <Piece NumberOfPoints=\"" << mesh->n_nodes << "\" NumberOfCells=\"" << mesh->n_cells << "\">\n";

    // Write PointData
    out << "      <PointData>\n";
    out << "      </PointData>\n";

    // Write CellData
    out << "      <CellData>\n";
    for (const auto & data_ptr : data_ptrs) {
        out << "        <DataArray type=\"" << vtk_float_type() << "\" ";
        out << "Name=\"" << data_ptr->name() << "\" ";
        out << "format=\"appended\" ";
        out << "offset=\"" << offset_data << "\">\n";
        out << "        </DataArray>\n";
        offset_data += sizeof(int) + mesh->n_cells * sizeof(rtype);
    }
    out << "      </CellData>\n";

    // Write Points
    out << "      <Points>\n";
    out << "        <DataArray type=\"" << vtk_float_type() << "\" ";
    out << "NumberOfComponents=\"" << 3 << "\" ";
    out << "format=\"appended\" ";
    out << "offset=\"" << offset_data << "\">\n";
    out << "        </DataArray>\n";
    offset_data += sizeof(int) + mesh->n_nodes * 3 * sizeof(rtype);
    out << "      </Points>\n";

    // Write Cells
    out << "      <Cells>\n";

    // connectivity
    out << "        <DataArray type=\"Int32\" Name=\"connectivity\" ";
    out << "format=\"appended\" ";
    out << "offset=\"" << offset_data << "\">\n";
    out << "        </DataArray>\n";
    for (u_int32_t i = 0; i < mesh->n_cells; i++) {
        len_connectivity += mesh->h_n_nodes_of_cell(i);
    }
    offset_data += sizeof(int) + len_connectivity * sizeof(int);

    // offsets
    out << "        <DataArray type=\"Int32\" Name=\"offsets\" ";
    out << "format=\"appended\" ";
    out << "offset=\"" << offset_data << "\">\n";
    out << "        </DataArray>\n";
    offset_data += sizeof(int) + mesh->n_cells * sizeof(int);

    // types
    out << "        <DataArray type=\"Int32\" Name=\"types\" ";
    out << "format=\"appended\" ";
    out << "offset=\"" << offset_data << "\">\n";
    out << "        </DataArray>\n";
    offset_data += sizeof(int) + mesh->n_cells * sizeof(int);

    out << "      </Cells>\n";
    out << "    </Piece>\n";
    out << "  </UnstructuredGrid>\n";

    // ---------------------------------------------------------------------------------------------
    // Write appended data
    out << "<AppendedData encoding=\"raw\">\n_";

    u_int64_t n_bytes;

    // Write PointData
    // Do nothing, no point data

    // Write CellData
    n_bytes = sizeof(rtype) * mesh->n_cells;
    for (const auto & data_ptr : data_ptrs) {
        out.write(reinterpret_cast<const char *>(&n_bytes), sizeof(int));
        for (u_int32_t i = 0; i < mesh->n_cells; i++) {
            out.write(reinterpret_cast<const char *>(&(*data_ptr)[i]), sizeof(rtype));
        }
    }

    // Write Points
    n_bytes = sizeof(rtype) * mesh->n_nodes * 3;
    out.write(reinterpret_cast<const char *>(&n_bytes), sizeof(int));
    rtype coord;
    for (u_int32_t i_node = 0; i_node < mesh->n_nodes; i_node++) {
        FOR_I_DIM {
            coord = mesh->h_node_coords(i_node, i);
            out.write(reinterpret_cast<const char *>(&coord), sizeof(rtype));
        }
        if (N_DIM == 2) {
            rtype zero = 0.0;
            out.write(reinterpret_cast<const char *>(&zero), sizeof(rtype));
        }
    }

    // Write Cells

    // connectivity
    n_bytes = sizeof(int) * len_connectivity;
    out.write(reinterpret_cast<const char *>(&n_bytes), sizeof(int));
    u_int32_t i_node;
    for (u_int32_t i = 0; i < mesh->n_cells; i++) {
        for (u_int32_t j = 0; j < mesh->h_n_nodes_of_cell(i); j++) {
            i_node = mesh->h_node_of_cell(i, j);
            out.write(reinterpret_cast<const char *>(&i_node), sizeof(int));
        }
    }

    // offsets
    n_bytes = sizeof(int) * mesh->n_cells;
    out.write(reinterpret_cast<const char *>(&n_bytes), sizeof(int));
    u_int64_t offset = 0;
    for (u_int32_t i = 0; i < mesh->n_cells; i++) {
        offset += mesh->h_n_nodes_of_cell(i);
        out.write(reinterpret_cast<const char *>(&offset), sizeof(int));
    }

    // types
    n_bytes = sizeof(int) * mesh->n_cells;
    out.write(reinterpret_cast<const char *>(&n_bytes), sizeof(int));
    u_int8_t cell_type = 7;
    for (u_int32_t i = 0; i < mesh->n_cells; i++) {
        out.write(reinterpret_cast<const char *>(&cell_type), sizeof(int));
    }

    out << "\n";
    out << "</AppendedData>\n";
    out << "</VTKFile>\n";

    out.close();
}

void DataWriter::write_tecplot(u_int32_t step) const {
    std::string filename = prefix + "_" + std::to_string(step) + ".dat";

    throw std::runtime_error("DataWriter::write_tecplot not implemented.");
}
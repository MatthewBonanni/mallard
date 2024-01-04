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

#include <string>
#include <iomanip>
#include <optional>

#include "common/common_io.h"

DataWriter::DataWriter() {
    // Empty
}

DataWriter::~DataWriter() {
    // Empty
}

void DataWriter::init(const toml::table & input,
                      std::vector<Data> & data,
                      std::shared_ptr<Mesh> mesh) {
    std::optional<std::string> prefix = input["prefix"].value<std::string>();
    std::optional<std::string> format_str = input["format"].value<std::string>();
    std::optional<int> interval = input["interval"].value<int>();
    auto variables = input["variables"];
    const toml::array* arr = variables.as_array();

    if (!prefix.has_value()) {
        throw std::runtime_error("DataWriter: prefix not specified.");
    }
    if (!format_str.has_value()) {
        throw std::runtime_error("DataWriter: format not specified.");
    }
    if (!interval.has_value()) {
        throw std::runtime_error("DataWriter: interval not specified.");
    }
    if (!variables) {
        throw std::runtime_error("DataWriter: variables not specified.");
    } else if (arr->size() == 0) {
        throw std::runtime_error("DataWriter: variables must be a non-empty array.");
    }

    this->prefix = prefix.value();
    this->interval = interval.value();

    DataFormat format;
    typename std::unordered_map<std::string, DataFormat>::const_iterator it = FORMAT_TYPES.find(*format_str);
    if (it == FORMAT_TYPES.end()) {
        throw std::runtime_error("DataWriter: Unknown format type: " + *format_str + ".");
    } else {
        format = it->second;
    }

    for (const auto & var : *arr) {
        std::optional<std::string> var_str = var.value<std::string>();
        if (!var_str.has_value()) {
            throw std::runtime_error("DataWriter: variable must be a string.");
        }

        bool found = false;
        for (const auto & data_var : data) {
            if (data_var.name() == var_str.value()) {
                data_ptrs.push_back(&data_var);
                found = true;
                break;
            }
        }
        if (!found) {
            throw std::runtime_error("DataWriter: Unknown variable: " + var_str.value() + ".");
        }
    }

    this->mesh = mesh;
}

void DataWriter::write(int step, bool force) const {
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

void DataWriter::write_vtu(int step) const {
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

    // \todo binary format

    // Write header
    out << "<?xml version=\"1.0\"?>\n";
    out << "<VTKFile type=\"UnstructuredGrid\" version=\"0.1\" byte_order=\"LittleEndian\">\n";
    out << "  <UnstructuredGrid>\n";
    out << "    <Piece NumberOfPoints=\"" << mesh->n_nodes() << "\" NumberOfCells=\"" << mesh->n_cells() << "\">\n";

    // Write PointData
    out << "      <PointData>\n";
    out << "      </PointData>\n";

    // Write CellData
    out << "      <CellData>\n";
    for (const auto & data_ptr : data_ptrs) {
        out << "        <DataArray type=\"Float64\" Name=\"" << data_ptr->name() << "\" format=\"ascii\">\n";
        for (int i = 0; i < mesh->n_cells(); i++) {
            out << "          " << (*data_ptr)[i] << "\n";
        }
        out << "        </DataArray>\n";
    }
    out << "      </CellData>\n";

    // Write Points
    out << "      <Points>\n";
    out << "        <DataArray type=\"Float64\" NumberOfComponents=\"" << 3 << "\" format=\"ascii\">\n";
    for (int i = 0; i < mesh->n_nodes(); i++) {
        out << "          ";
        for (int j = 0; j < N_DIM; j++) {
            out << mesh->node_coords(i)[j] << " ";
        }
        if (N_DIM == 2) {
            out << "0.0";
        }
        out << "\n";
        // \todo Add support for 3D meshes
    }
    out << "        </DataArray>\n";
    out << "      </Points>\n";

    // Write Cells
    out << "      <Cells>\n";
    out << "        <DataArray type=\"Int32\" Name=\"connectivity\" format=\"ascii\">\n";
    for (int i = 0; i < mesh->n_cells(); i++) {
        out << "          ";
        for (int j = 0; j < mesh->nodes_of_cell(i).size(); j++) {
            out << mesh->nodes_of_cell(i)[j] << " ";
        }
        out << "\n";
    }
    out << "        </DataArray>\n";
    out << "        <DataArray type=\"Int32\" Name=\"offsets\" format=\"ascii\">\n";
    int offset = 0;
    for (int i = 0; i < mesh->n_cells(); i++) {
        offset += mesh->nodes_of_cell(i).size();
        out << "          " << offset << "\n";
    }
    out << "        </DataArray>\n";
    out << "        <DataArray type=\"Int32\" Name=\"types\" format=\"ascii\">\n";
    for (int i = 0; i < mesh->n_cells(); i++) {
        out << "          " << 7 << "\n";
    }
    out << "        </DataArray>\n";
    out << "      </Cells>\n";

    // Write footer
    out << "    </Piece>\n";
    out << "  </UnstructuredGrid>\n";
    out << "</VTKFile>\n";

    out.close();
}

void DataWriter::write_tecplot(int step) const {
    std::string filename = prefix + "_" + std::to_string(step) + ".dat";

    throw std::runtime_error("DataWriter::write_tecplot not implemented.");
}
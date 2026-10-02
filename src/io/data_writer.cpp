/**
 * @file data_writer.cpp
 * @author Matthew Bonanni (mbonanni001@gmail.com)
 * @brief Data writer class implementation.
 * @version 0.2
 * @date 2024-01-11
 *
 * @copyright Copyright (c) 2024 Matthew Bonanni
 *
 */

#include "data_writer.h"

#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>

#include "common_io.h"

void DataWriter::init(const toml::value & input,
                      std::vector<Data> & data,
                      std::shared_ptr<Mesh> mesh) {
    for (const char * key : {"prefix", "format"}) {
        if (!input.contains(key)) {
            throw std::runtime_error(std::string("DataWriter: ") + key + " not specified.");
        }
    }
    const bool has_interval = input.contains("interval");
    const bool has_time_interval = input.contains("time_interval");
    if (has_interval == has_time_interval) {
        throw std::runtime_error("DataWriter: specify exactly one of interval or time_interval.");
    }
    prefix = toml::find<std::string>(input, "prefix");
    if (has_interval) {
        interval = toml::find<uint64_t>(input, "interval");
        if (interval == 0) {
            throw std::runtime_error("DataWriter: interval must be positive.");
        }
    } else {
        time_interval = toml::find<rtype>(input, "time_interval");
        if (!(time_interval > 0.0)) {
            throw std::runtime_error("DataWriter: time_interval must be positive.");
        }
    }

    const std::string format_str = toml::find<std::string>(input, "format");
    auto it = FORMAT_TYPES.find(format_str);
    if (it == FORMAT_TYPES.end()) {
        throw std::runtime_error("DataWriter: Unknown format type: " + format_str + ".");
    }
    format = it->second;

    std::vector<std::string> variables;
    if (format == DataFormat::RESTART) {
        variables = {"RHO", "RHOU_X", "RHOU_Y", "RHOE"};
    } else {
        if (!input.contains("variables")) {
            throw std::runtime_error("DataWriter: variables not specified.");
        }
        variables = toml::find<std::vector<std::string>>(input, "variables");
    }
    if (variables.empty()) {
        throw std::runtime_error("DataWriter: No variables specified.");
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

    const std::filesystem::path parent = std::filesystem::path(prefix).parent_path();
    if (!parent.empty()) {
        std::filesystem::create_directories(parent);
    }
}

bool DataWriter::due(uint64_t step, rtype t) const {
    if (interval > 0) {
        return step % interval == 0;
    }
    // Relative tolerance absorbs round-off in accumulated time
    return t >= next_time() - 1.0e-9 * time_interval;
}

rtype DataWriter::next_time() const {
    if (interval > 0) {
        return std::numeric_limits<rtype>::infinity();
    }
    return n_written * time_interval;
}

void DataWriter::write(uint64_t step, rtype t, bool force) {
    if (!(due(step, t) || force) || step == step_last) {
        return;
    }
    std::ostringstream stream;
    stream << prefix << "_" << std::setw(LEN_STEP) << std::setfill('0')
           << (interval > 0 ? step : n_written);
    if (format == DataFormat::RESTART) {
        write_restart(stream.str() + ".restart", step, t);
    } else {
        const std::string filename = stream.str() + ".vtu";
        write_vtu(filename, t);
        history.emplace_back(t, filename);
        write_pvd();
    }
    if (interval == 0) {
        // Skip any output times already passed (e.g. if dt exceeded time_interval)
        while (next_time() <= t + 1.0e-9 * time_interval) {
            n_written++;
        }
    } else {
        n_written++;
    }
    step_last = step;
    t_last = t;
}

void DataWriter::resume(uint64_t step, rtype t) {
    step_last = step;
    t_last = t;
    if (interval > 0) {
        n_written = step / interval + 1;
    } else {
        n_written = static_cast<uint64_t>(std::floor(t / time_interval + 1.0e-9)) + 1;
    }
    // Keep the entries of the previous run's .pvd up to the restart time
    history.clear();
    std::ifstream in(prefix + ".pvd");
    std::string line;
    const std::filesystem::path dir = std::filesystem::path(prefix).parent_path();
    while (std::getline(in, line)) {
        const size_t a = line.find("timestep=\"");
        const size_t b = line.find("file=\"");
        if (a == std::string::npos || b == std::string::npos) continue;
        const double t_entry = std::stod(line.substr(a + 10));
        const size_t b0 = b + 6;
        const std::string file = line.substr(b0, line.find('"', b0) - b0);
        if (t_entry <= t * (1.0 + 1.0e-12) + 1.0e-300) {
            history.emplace_back(t_entry, (dir / file).string());
        }
    }
}

void DataWriter::write_restart(const std::string & filename, uint64_t step, rtype t) const {
    std::ofstream out(filename, std::ios::binary);
    if (!out.good()) {
        throw std::runtime_error("DataWriter::write_restart: Could not open file: " + filename + ".");
    }
    std::cout << "Writing restart file: " << filename << std::endl;
    const char magic[16] = "MALLARD-RESTART";
    const uint32_t version = 1;
    const uint32_t real_size = sizeof(rtype);
    const uint64_t n_cells = mesh->n_cells;
    const uint64_t n_vars = data_ptrs.size();
    const double time = t;
    out.write(magic, sizeof(magic));
    out.write(reinterpret_cast<const char *>(&version), sizeof(version));
    out.write(reinterpret_cast<const char *>(&real_size), sizeof(real_size));
    out.write(reinterpret_cast<const char *>(&n_cells), sizeof(n_cells));
    out.write(reinterpret_cast<const char *>(&n_vars), sizeof(n_vars));
    out.write(reinterpret_cast<const char *>(&step), sizeof(step));
    out.write(reinterpret_cast<const char *>(&time), sizeof(time));
    for (const auto & data_ptr : data_ptrs) {
        for (uint64_t i = 0; i < n_cells; i++) {
            const rtype value = (*data_ptr)[i];
            out.write(reinterpret_cast<const char *>(&value), sizeof(rtype));
        }
    }
}

RestartData read_restart(const std::string & filename) {
    std::ifstream in(filename, std::ios::binary);
    if (!in.good()) {
        throw std::runtime_error("Could not open restart file: " + filename + ".");
    }
    char magic[16];
    uint32_t version, real_size;
    uint64_t n_vars;
    RestartData data;
    in.read(magic, sizeof(magic));
    in.read(reinterpret_cast<char *>(&version), sizeof(version));
    in.read(reinterpret_cast<char *>(&real_size), sizeof(real_size));
    in.read(reinterpret_cast<char *>(&data.n_cells), sizeof(data.n_cells));
    in.read(reinterpret_cast<char *>(&n_vars), sizeof(n_vars));
    in.read(reinterpret_cast<char *>(&data.step), sizeof(data.step));
    in.read(reinterpret_cast<char *>(&data.t), sizeof(data.t));
    if (!in.good() || std::string(magic) != "MALLARD-RESTART" || version != 1) {
        throw std::runtime_error("Not a Mallard restart file: " + filename + ".");
    }
    if (real_size != sizeof(rtype)) {
        throw std::runtime_error("Restart file " + filename + " was written with a different floating-point precision.");
    }
    data.conservatives.assign(n_vars, std::vector<rtype>(data.n_cells));
    for (auto & var : data.conservatives) {
        in.read(reinterpret_cast<char *>(var.data()), data.n_cells * sizeof(rtype));
    }
    if (!in.good()) {
        throw std::runtime_error("Restart file " + filename + " is truncated.");
    }
    return data;
}

void DataWriter::write_pvd() const {
    std::ofstream out(prefix + ".pvd");
    out << "<?xml version=\"1.0\"?>\n";
    out << "<VTKFile type=\"Collection\" version=\"0.1\" byte_order=\"" << endianness() << "\">\n";
    out << "  <Collection>\n";
    for (const auto & [t, filename] : history) {
        out << "    <DataSet timestep=\"" << std::setprecision(10) << t << "\" file=\""
            << std::filesystem::path(filename).filename().string() << "\"/>\n";
    }
    out << "  </Collection>\n";
    out << "</VTKFile>\n";
}

void DataWriter::write_vtu(const std::string & filename, rtype t) const {
    std::ofstream out(filename, std::ios::binary);
    if (!out.good()) {
        throw std::runtime_error("DataWriter::write_vtu: Could not open file: " + filename + ".");
    }
    std::cout << "Writing data to file: " << filename << std::endl;

    using header_t = uint64_t;
    uint64_t len_connectivity = 0;
    for (uint32_t i = 0; i < mesh->n_cells; i++) {
        len_connectivity += mesh->h_n_nodes_of_cell(i);
    }

    // Header with appended-data offsets
    uint64_t offset = 0;
    auto data_array = [&](const std::string & type, const std::string & name,
                          uint32_t n_comp, uint64_t n_bytes) {
        out << "        <DataArray type=\"" << type << "\" ";
        if (!name.empty()) out << "Name=\"" << name << "\" ";
        if (n_comp > 1) out << "NumberOfComponents=\"" << n_comp << "\" ";
        out << "format=\"appended\" offset=\"" << offset << "\"/>\n";
        offset += sizeof(header_t) + n_bytes;
    };

    out << "<?xml version=\"1.0\"?>\n";
    out << "<VTKFile type=\"UnstructuredGrid\" version=\"1.0\" byte_order=\"" << endianness()
        << "\" header_type=\"UInt64\">\n";
    out << "  <UnstructuredGrid>\n";
    out << "    <FieldData>\n";
    out << "      <DataArray type=\"Float64\" Name=\"TIME\" NumberOfTuples=\"1\" format=\"ascii\">"
        << std::setprecision(17) << static_cast<double>(t) << "</DataArray>\n";
    out << "    </FieldData>\n";
    out << "    <Piece NumberOfPoints=\"" << mesh->n_nodes << "\" NumberOfCells=\"" << mesh->n_cells << "\">\n";
    out << "      <CellData>\n";
    for (const auto & data_ptr : data_ptrs) {
        data_array(vtk_float_type(), data_ptr->name(), 1, mesh->n_cells * sizeof(rtype));
    }
    out << "      </CellData>\n";
    out << "      <Points>\n";
    data_array(vtk_float_type(), "", 3, mesh->n_nodes * 3 * sizeof(rtype));
    out << "      </Points>\n";
    out << "      <Cells>\n";
    data_array("Int64", "connectivity", 1, len_connectivity * sizeof(int64_t));
    data_array("Int64", "offsets", 1, mesh->n_cells * sizeof(int64_t));
    data_array("UInt8", "types", 1, mesh->n_cells * sizeof(uint8_t));
    out << "      </Cells>\n";
    out << "    </Piece>\n";
    out << "  </UnstructuredGrid>\n";
    out << "  <AppendedData encoding=\"raw\">\n_";

    auto write_header = [&](uint64_t n_bytes) {
        header_t h = n_bytes;
        out.write(reinterpret_cast<const char *>(&h), sizeof(header_t));
    };
    auto write_value = [&](auto value) {
        out.write(reinterpret_cast<const char *>(&value), sizeof(value));
    };

    for (const auto & data_ptr : data_ptrs) {
        write_header(mesh->n_cells * sizeof(rtype));
        for (uint32_t i = 0; i < mesh->n_cells; i++) {
            write_value(static_cast<rtype>((*data_ptr)[i]));
        }
    }

    write_header(mesh->n_nodes * 3 * sizeof(rtype));
    for (uint32_t i_node = 0; i_node < mesh->n_nodes; i_node++) {
        FOR_I_DIM write_value(static_cast<rtype>(mesh->h_node_coords(i_node, i)));
        for (uint8_t i = N_DIM; i < 3; i++) write_value(static_cast<rtype>(0.0));
    }

    write_header(len_connectivity * sizeof(int64_t));
    for (uint32_t i = 0; i < mesh->n_cells; i++) {
        for (uint32_t j = 0; j < mesh->h_n_nodes_of_cell(i); j++) {
            write_value(static_cast<int64_t>(mesh->h_node_of_cell(i, j)));
        }
    }

    write_header(mesh->n_cells * sizeof(int64_t));
    int64_t cell_offset = 0;
    for (uint32_t i = 0; i < mesh->n_cells; i++) {
        cell_offset += mesh->h_n_nodes_of_cell(i);
        write_value(cell_offset);
    }

    write_header(mesh->n_cells * sizeof(uint8_t));
    for (uint32_t i = 0; i < mesh->n_cells; i++) {
        const uint32_t n_nodes = mesh->h_n_nodes_of_cell(i);
        const uint8_t vtk_type = (n_nodes == 3) ? 5 : (n_nodes == 4) ? 9 : 7;
        write_value(vtk_type);
    }

    out << "\n  </AppendedData>\n";
    out << "</VTKFile>\n";
}

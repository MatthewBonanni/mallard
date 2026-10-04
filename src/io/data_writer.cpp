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

#include "comm.h"
#include "input.h"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <numeric>
#include <sstream>
#include <unordered_map>

#include "common_io.h"
#include "log.h"

namespace {

/**
 * @brief VTK cell type of a cell or boundary face, by dimension and node count.
 */
uint8_t vtk_type(uint32_t dim, uint32_t n_nodes) {
    if (dim == 1 && n_nodes == 2) return 3;   // VTK_LINE
    if (dim == 2 && n_nodes == 3) return 5;   // VTK_TRIANGLE
    if (dim == 2 && n_nodes == 4) return 9;   // VTK_QUAD
    if (dim == 3 && n_nodes == 4) return 10;  // VTK_TETRA
    if (dim == 3 && n_nodes == 8) return 12;  // VTK_HEXAHEDRON
    if (dim == 3 && n_nodes == 6) return 13;  // VTK_WEDGE
    if (dim == 3 && n_nodes == 5) return 14;  // VTK_PYRAMID
    throw std::runtime_error("DataWriter: no VTK cell type for a " + std::to_string(dim) + "D element with " +
                             std::to_string(n_nodes) + " nodes.");
}

/**
 * @brief Local node of a cell at VTK position k. Mallard prisms have the
 *        (0, 1, 2) normal pointing toward (3, 4, 5); VTK wedges point it away.
 */
uint32_t vtk_local_node(uint32_t n_nodes, uint32_t k) {
    constexpr uint32_t WEDGE[6] = {0, 2, 1, 3, 5, 4};
    return (N_DIM == 3 && n_nodes == 6) ? WEDGE[k] : k;
}

} // namespace

void DataWriter::init(const toml::value & input,
                      std::vector<Data> & data,
                      std::shared_ptr<Mesh> mesh_in,
                      const std::vector<std::string> & restart_variables) {
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
        time_interval = find_real(input, "time_interval");
        if (!(time_interval > 0.0_r)) {
            throw std::runtime_error("DataWriter: time_interval must be positive.");
        }
    }

    const std::string format_str = toml::find<std::string>(input, "format");
    auto it = FORMAT_TYPES.find(format_str);
    if (it == FORMAT_TYPES.end()) {
        throw unknown_option(FORMAT_TYPES, "write_data.format", format_str);
    }
    format = it->second;

    std::vector<std::string> variables;
    if (format == DataFormat::RESTART) {
        if (restart_variables.empty()) {
            variables.assign(CONSERVATIVE_NAMES.begin(), CONSERVATIVE_NAMES.end());
        } else {
            variables = restart_variables;
        }
    } else {
        if (!input.contains("variables")) {
            throw std::runtime_error("DataWriter: variables not specified.");
        }
        variables = toml::find<std::vector<std::string>>(input, "variables");
    }
    if (variables.empty()) {
        throw std::runtime_error("DataWriter: No variables specified.");
    }
    auto find_data = [&](const std::string & name) -> const Data * {
        for (const auto & data_var : data) {
            if (data_var.name() == name) return &data_var;
        }
        return nullptr;
    };
    // A trailing * selects every variable with that prefix (e.g. Y_* for all mass fractions)
    std::vector<std::string> expanded;
    for (const auto & var : variables) {
        if (var.empty() || var.back() != '*') {
            expanded.push_back(var);
            continue;
        }
        const std::string stem = var.substr(0, var.size() - 1);
        const size_t n_before = expanded.size();
        for (const auto & data_var : data) {
            if (data_var.name().rfind(stem, 0) == 0) expanded.push_back(data_var.name());
        }
        if (expanded.size() == n_before) {
            throw InputError("write_data.variables: no variable matches \"" + var + "\".");
        }
    }
    variables = expanded;
    for (const auto & var : variables) {
        Field field{var, {}};
        if (const Data * scalar = find_data(var)) {
            field.components.push_back(scalar);
        } else if (format == DataFormat::VTU) {
            // A vector name (e.g. U) collects its components U_X, U_Y(, U_Z)
            const char * suffixes[3] = {"_X", "_Y", "_Z"};
            FOR_I_DIM {
                if (const Data * component = find_data(var + suffixes[i])) field.components.push_back(component);
            }
            if (field.components.size() != N_DIM) field.components.clear();
        }
        if (field.components.empty()) {
            throw InputError("write_data.variables: unknown variable \"" + var + "\".");
        }
        fields.push_back(field);
    }
    this->mesh = mesh_in;

    geometry = toml::find_or<std::string>(input, "geometry", "all");
    if (geometry != "all") {
        if (format != DataFormat::VTU) {
            throw std::runtime_error("DataWriter: geometry can only be set for vtu output.");
        }
        FaceZone * zone = mesh->get_face_zone(geometry);
        if (comm::allreduce(uint32_t(zone != nullptr), comm::Op::SUM) == 0) {
            throw std::runtime_error("DataWriter: unknown geometry: " + geometry + ".");
        }
        surface = true;
        // Each rank writes the faces of its owned cells
        for (uint32_t i = 0; zone && i < zone->n_faces(); i++) {
            const uint32_t f = zone->h_faces(i);
            if (static_cast<uint32_t>(mesh->h_cells_of_face(f, 0)) < mesh->n_owned()) geometry_faces.push_back(f);
        }
    }

    const std::filesystem::path parent = std::filesystem::path(prefix).parent_path();
    if (!parent.empty()) {
        std::filesystem::create_directories(parent);
    }
}

std::pair<std::string, std::string> DataWriter::summary() const {
    std::string text = prefix + (format == DataFormat::RESTART ? "_*.restart" : "_*.vtu");
    if (geometry != "all") text += " (" + geometry + ")";
    text += interval > 0 ? " every " + logging::count(interval) + " steps" : " every t = " + logging::real(double(time_interval));
    if (format == DataFormat::VTU) {
        std::string names;
        for (const auto & field : fields) names += (names.empty() ? "" : ", ") + field.name;
        text += ": " + names;
    }
    return {FORMAT_NAMES.at(format), text};
}

bool DataWriter::due(uint64_t step, rtype t) const {
    if (interval > 0) {
        return step % interval == 0;
    }
    // Relative tolerance absorbs round-off in accumulated time
    return t >= next_time() - precision_tol(1e-9, 1e-5) * time_interval;
}

rtype DataWriter::next_time() const {
    if (interval > 0) {
        return std::numeric_limits<rtype>::infinity();
    }
    return n_written * time_interval;
}

void DataWriter::write(uint64_t step, rtype t, bool force, const RestartAttributes & attributes) {
    if (!(due(step, t) || force) || step == step_last) {
        return;
    }
    std::ostringstream stream;
    stream << prefix << "_" << std::setw(LEN_STEP) << std::setfill('0')
           << (interval > 0 || format == DataFormat::RESTART ? step : history.size());
    if (format == DataFormat::RESTART) {
        write_restart(stream.str() + ".restart", step, t, attributes);
        logging::event(step, double(t), "restart", stream.str() + ".restart");
    } else {
        // Distributed runs: one piece per rank and a .pvtu index
        const bool pieces = mesh->n_global_cells > 0;
        std::ostringstream piece;
        piece << stream.str();
        if (pieces) piece << "_p" << std::setw(4) << std::setfill('0') << comm::rank();
        if (!surface) {
            write_vtu(piece.str() + ".vtu", t);
        } else {
            write_vtu_faces(piece.str() + ".vtu", t);
        }
        const std::string filename = stream.str() + (pieces ? ".pvtu" : ".vtu");
        if (pieces && comm::is_root()) write_pvtu(filename, std::filesystem::path(stream.str()).filename().string());
        history.emplace_back(t, filename);
        if (comm::is_root()) write_pvd();
        logging::event(step, double(t), surface ? "surface" : "vtu", filename);
    }
    if (interval == 0) {
        // Skip any output times already passed (e.g. if dt exceeded time_interval)
        while (next_time() <= t + precision_tol(1e-9, 1e-5) * time_interval) {
            n_written++;
        }
    } else {
        n_written++;
    }
    step_last = step;
    t_last = t;
    n_files++;
}

void DataWriter::resume(uint64_t step, rtype t) {
    step_last = step;
    t_last = t;
    if (interval > 0) {
        n_written = step / interval + 1;
    } else {
        n_written = static_cast<uint64_t>(std::floor(t / time_interval + precision_tol(1e-9, 1e-5))) + 1;
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
        if (t_entry <= static_cast<double>(t) * (1.0 + 1.0e-12) + 1.0e-300) {
            history.emplace_back(t_entry, (dir / file).string());
        }
    }
}

namespace {

constexpr char RESTART_MAGIC[16] = "MALLARD-RESTART";
constexpr uint32_t RESTART_VERSION = 3;

/**
 * @brief Restart header: magic, version, real size, cell count, variable
 *        count, step, time, then (version 2) each variable name as a uint32
 *        length and its characters, then (version 3) the attribute count as a
 *        uint64 and each attribute's name, likewise, and double value. The
 *        values follow, one block of n_cells per variable. Files without
 *        attributes are written as version 2, which older builds read.
 */
std::vector<char> restart_header(uint64_t n_cells, uint64_t step, double t, const std::vector<std::string> & names,
                                 const RestartAttributes & attributes) {
    std::vector<char> header;
    auto put = [&](const void * p, size_t n) {
        header.insert(header.end(), static_cast<const char *>(p), static_cast<const char *>(p) + n);
    };
    const uint32_t real_size = sizeof(rtype);
    const uint64_t n_vars = names.size();
    const uint32_t version = attributes.empty() ? 2 : RESTART_VERSION;
    put(RESTART_MAGIC, sizeof(RESTART_MAGIC));
    put(&version, sizeof(version));
    put(&real_size, sizeof(real_size));
    put(&n_cells, sizeof(n_cells));
    put(&n_vars, sizeof(n_vars));
    put(&step, sizeof(step));
    put(&t, sizeof(t));
    for (const auto & name : names) {
        const uint32_t length = static_cast<uint32_t>(name.size());
        put(&length, sizeof(length));
        put(name.data(), name.size());
    }
    if (attributes.empty()) return header;
    const uint64_t n_attributes = attributes.size();
    put(&n_attributes, sizeof(n_attributes));
    for (const auto & [name, value] : attributes) {
        const uint32_t length = static_cast<uint32_t>(name.size());
        put(&length, sizeof(length));
        put(name.data(), name.size());
        put(&value, sizeof(value));
    }
    return header;
}

} // namespace

const std::vector<rtype> * RestartData::find(const std::string & name) const {
    for (size_t v = 0; v < names.size(); v++) {
        if (names[v] == name) return &fields[v];
    }
    return nullptr;
}

const double * RestartData::attribute(const std::string & name) const {
    for (const auto & [key, value] : attributes) {
        if (key == name) return &value;
    }
    return nullptr;
}

void DataWriter::write_restart(const std::string & filename, uint64_t step, rtype t,
                               const RestartAttributes & attributes) const {
    if (mesh->n_global_cells > 0) {
        write_restart_distributed(filename, step, t, attributes);
        return;
    }
    std::ofstream out(filename, std::ios::binary);
    if (!out.good()) {
        throw std::runtime_error("DataWriter::write_restart: Could not open file: " + filename + ".");
    }
    std::vector<std::string> names;
    for (const auto & field : fields) names.push_back(field.name);
    const std::vector<char> header = restart_header(mesh->n_cells, step, double(t), names, attributes);
    out.write(header.data(), static_cast<std::streamsize>(header.size()));
    for (const auto & field : fields) {
        for (uint64_t i = 0; i < mesh->n_cells; i++) {
            const rtype value = field.value(i, 0);
            out.write(reinterpret_cast<const char *>(&value), sizeof(rtype));
        }
    }
}

void DataWriter::write_restart_distributed(const std::string & filename, uint64_t step, rtype t,
                                           const RestartAttributes & attributes) const {
#ifdef Mallard_HAS_MPI
    // Same layout as a serial restart, cells in global order: each rank writes
    // its owned cells at their global offsets, so any rank count can read it
    MPI_File fh;
    if (MPI_File_open(comm::world(), filename.c_str(), MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &fh) !=
        MPI_SUCCESS) {
        throw std::runtime_error("DataWriter::write_restart: Could not open file: " + filename + ".");
    }
    MPI_File_set_size(fh, 0);
    const uint64_t n_global = mesh->n_global_cells;
    std::vector<std::string> names;
    for (const auto & field : fields) names.push_back(field.name);
    const std::vector<char> header = restart_header(n_global, step, double(t), names, attributes);
    if (comm::is_root()) {
        MPI_File_write_at(fh, 0, header.data(), static_cast<int>(header.size()), MPI_BYTE, MPI_STATUS_IGNORE);
    }
    const MPI_Offset header_size = static_cast<MPI_Offset>(header.size());
    const uint32_t n_owned = mesh->n_owned();
    const MPI_Datatype real_type = sizeof(rtype) == sizeof(double) ? MPI_DOUBLE : MPI_FLOAT;
    std::vector<MPI_Aint> displacements(n_owned);
    for (uint32_t i = 0; i < n_owned; i++) displacements[i] = mesh->h_global_cell_id[i] * sizeof(rtype);
    MPI_Datatype file_type;
    MPI_Type_create_hindexed_block(static_cast<int>(n_owned), 1, displacements.data(), real_type, &file_type);
    MPI_Type_commit(&file_type);
    std::vector<rtype> values(n_owned);
    for (size_t v = 0; v < fields.size(); v++) {
        for (uint32_t i = 0; i < n_owned; i++) values[i] = fields[v].value(i, 0);
        MPI_File_set_view(fh, header_size + MPI_Offset(v * n_global * sizeof(rtype)), real_type, file_type, "native",
                          MPI_INFO_NULL);
        MPI_File_write_all(fh, values.data(), static_cast<int>(n_owned), real_type, MPI_STATUS_IGNORE);
    }
    MPI_Type_free(&file_type);
    MPI_File_close(&fh);
#else
    (void)filename;
    (void)step;
    (void)t;
    (void)attributes;
    throw std::logic_error("DataWriter: distributed restart without MPI");
#endif
}

RestartData read_restart(const std::string & filename, const std::vector<uint64_t> * cells) {
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
    magic[sizeof(magic) - 1] = '\0';
    if (!in.good() || std::string(magic) != RESTART_MAGIC || version < 1 || version > RESTART_VERSION) {
        throw std::runtime_error("Not a Mallard restart file: " + filename + ".");
    }
    if (real_size != sizeof(rtype)) {
        throw std::runtime_error("Restart file " + filename + " was written with a different floating-point precision.");
    }
    if (version == 1) {
        // Version 1 holds the flow block only; its size gives the dimension
        if (n_vars != N_CONSERVATIVE) {
            throw std::runtime_error("Restart file " + filename + " has " + std::to_string(n_vars) +
                                     " variables per cell (a " + std::to_string(n_vars - 2) +
                                     "D run), but Mallard was built with Mallard_DIM = " + std::to_string(N_DIM) +
                                     " (" + std::to_string(N_CONSERVATIVE) + " variables).");
        }
        data.names.assign(CONSERVATIVE_NAMES.begin(), CONSERVATIVE_NAMES.end());
    } else {
        constexpr uint32_t MAX_NAME = 256;
        auto read_name = [&]() {
            uint32_t length = 0;
            in.read(reinterpret_cast<char *>(&length), sizeof(length));
            if (!in.good() || length > MAX_NAME) {
                throw std::runtime_error("Restart file " + filename + " has a corrupt variable list.");
            }
            std::string name(length, '\0');
            in.read(name.data(), length);
            return name;
        };
        for (uint64_t v = 0; v < n_vars && in.good(); v++) data.names.push_back(read_name());
        if (version >= 3) {
            uint64_t n_attributes = 0;
            in.read(reinterpret_cast<char *>(&n_attributes), sizeof(n_attributes));
            for (uint64_t a = 0; a < n_attributes && in.good(); a++) {
                std::string name = read_name();
                double value = 0.0;
                in.read(reinterpret_cast<char *>(&value), sizeof(value));
                data.attributes.emplace_back(std::move(name), value);
            }
        }
        const bool has_z = std::find(data.names.begin(), data.names.end(), "RHOU_Z") != data.names.end();
        if (has_z != (N_DIM == 3)) {
            throw std::runtime_error("Restart file " + filename + " holds a " + std::string(has_z ? "3" : "2") +
                                     "D run, but Mallard was built with Mallard_DIM = " + std::to_string(N_DIM) + ".");
        }
        for (const auto & name : CONSERVATIVE_NAMES) {
            if (std::find(data.names.begin(), data.names.end(), name) == data.names.end()) {
                throw std::runtime_error("Restart file " + filename + " has no variable " + name + ".");
            }
        }
    }
    if (!in.good()) {
        throw std::runtime_error("Restart file " + filename + " is truncated.");
    }
    if (!cells) {
        data.fields.assign(n_vars, std::vector<rtype>(data.n_cells));
        for (auto & var : data.fields) {
            in.read(reinterpret_cast<char *>(var.data()), static_cast<std::streamsize>(data.n_cells * sizeof(rtype)));
        }
    } else {
        // Runs of consecutive global ids, read with one seek each
        const std::streamoff header = in.tellg();
        std::vector<uint32_t> order(cells->size());
        std::iota(order.begin(), order.end(), 0u);
        std::sort(order.begin(), order.end(), [&](uint32_t a, uint32_t b) { return (*cells)[a] < (*cells)[b]; });
        if (!order.empty() && (*cells)[order.back()] >= data.n_cells) {
            throw std::runtime_error("Restart file " + filename + " does not match the mesh.");
        }
        data.fields.assign(n_vars, std::vector<rtype>(cells->size()));
        std::vector<rtype> run;
        for (uint64_t v = 0; v < n_vars; v++) {
            for (size_t a = 0; a < order.size();) {
                size_t b = a + 1;
                while (b < order.size() && (*cells)[order[b]] == (*cells)[order[b - 1]] + 1) b++;
                run.resize(b - a);
                in.seekg(header + static_cast<std::streamoff>((v * data.n_cells + (*cells)[order[a]]) * sizeof(rtype)));
                in.read(reinterpret_cast<char *>(run.data()), static_cast<std::streamsize>(run.size() * sizeof(rtype)));
                for (size_t k = a; k < b; k++) data.fields[v][order[k]] = run[k - a];
                a = b;
            }
        }
    }
    if (!in.good()) {
        throw std::runtime_error("Restart file " + filename + " is truncated.");
    }
    return data;
}

void DataWriter::write_pvtu(const std::string & filename, const std::string & stem) const {
    std::ofstream out(filename);
    out << "<?xml version=\"1.0\"?>\n";
    out << "<VTKFile type=\"PUnstructuredGrid\" version=\"1.0\" byte_order=\"" << endianness()
        << "\" header_type=\"UInt64\">\n";
    out << "  <PUnstructuredGrid GhostLevel=\"0\">\n";
    out << "    <PCellData>\n";
    for (const auto & field : fields) {
        out << "      <PDataArray type=\"" << vtk_float_type() << "\" Name=\"" << field.name << "\"";
        if (field.n_vtk_components() > 1) out << " NumberOfComponents=\"" << field.n_vtk_components() << "\"";
        out << "/>\n";
    }
    out << "    </PCellData>\n";
    out << "    <PPoints>\n      <PDataArray type=\"" << vtk_float_type() << "\" NumberOfComponents=\"3\"/>\n    </PPoints>\n";
    for (int r = 0; r < comm::size(); r++) {
        out << "    <Piece Source=\"" << stem << "_p" << std::setw(4) << std::setfill('0') << r << ".vtu\"/>\n";
    }
    out << "  </PUnstructuredGrid>\n";
    out << "</VTKFile>\n";
}

void DataWriter::write_pvd() const {
    std::ofstream out(prefix + ".pvd");
    out << "<?xml version=\"1.0\"?>\n";
    out << "<VTKFile type=\"Collection\" version=\"0.1\" byte_order=\"" << endianness() << "\">\n";
    out << "  <Collection>\n";
    for (const auto & [t, filename] : history) {
        out << "    <DataSet timestep=\"" << std::setprecision(std::numeric_limits<double>::max_digits10) << t << "\" file=\""
            << std::filesystem::path(filename).filename().string() << "\"/>\n";
    }
    out << "  </Collection>\n";
    out << "</VTKFile>\n";
}

void DataWriter::write_vtu_faces(const std::string & filename, rtype t) const {
    std::ofstream out(filename);
    if (!out.good()) {
        throw std::runtime_error("DataWriter::write_vtu_faces: Could not open file: " + filename + ".");
    }

    // Renumber the nodes of the selected faces
    std::unordered_map<uint32_t, uint32_t> local;
    std::vector<uint32_t> nodes;
    for (uint32_t f : geometry_faces) {
        for (uint32_t k = 0; k < mesh->h_n_nodes_of_face(f); k++) {
            const uint32_t node = mesh->h_node_of_face(f, k);
            if (local.emplace(node, nodes.size()).second) nodes.push_back(node);
        }
    }

    out << std::setprecision(std::numeric_limits<rtype>::max_digits10);
    out << "<?xml version=\"1.0\"?>\n";
    out << "<VTKFile type=\"UnstructuredGrid\" version=\"1.0\" byte_order=\"" << endianness() << "\">\n";
    out << "  <UnstructuredGrid>\n";
    out << "    <FieldData>\n";
    out << "      <DataArray type=\"Float64\" Name=\"TIME\" NumberOfTuples=\"1\" format=\"ascii\">"
        << static_cast<double>(t) << "</DataArray>\n";
    out << "    </FieldData>\n";
    out << "    <Piece NumberOfPoints=\"" << nodes.size() << "\" NumberOfCells=\"" << geometry_faces.size() << "\">\n";
    out << "      <CellData>\n";
    for (const auto & field : fields) {
        out << "        <DataArray type=\"Float64\" Name=\"" << field.name << "\" ";
        if (field.n_vtk_components() > 1) out << "NumberOfComponents=\"" << field.n_vtk_components() << "\" ";
        out << "format=\"ascii\">\n";
        for (uint32_t f : geometry_faces) {
            for (uint32_t k = 0; k < field.n_vtk_components(); k++) {
                out << static_cast<double>(field.value(mesh->h_cells_of_face(f, 0), k)) << " ";
            }
        }
        out << "\n        </DataArray>\n";
    }
    out << "      </CellData>\n";
    out << "      <Points>\n";
    out << "        <DataArray type=\"Float64\" NumberOfComponents=\"3\" format=\"ascii\">\n";
    for (uint32_t node : nodes) {
        FOR_I_DIM out << static_cast<double>(mesh->h_node_coords(node, i)) << " ";
        for (uint8_t i = N_DIM; i < 3; i++) out << "0 ";
    }
    out << "\n        </DataArray>\n";
    out << "      </Points>\n";
    out << "      <Cells>\n";
    out << "        <DataArray type=\"Int64\" Name=\"connectivity\" format=\"ascii\">\n";
    for (uint32_t f : geometry_faces) {
        for (uint32_t k = 0; k < mesh->h_n_nodes_of_face(f); k++) out << local.at(mesh->h_node_of_face(f, k)) << " ";
    }
    out << "\n        </DataArray>\n";
    out << "        <DataArray type=\"Int64\" Name=\"offsets\" format=\"ascii\">\n";
    uint64_t offset = 0;
    for (uint32_t f : geometry_faces) {
        offset += mesh->h_n_nodes_of_face(f);
        out << offset << " ";
    }
    out << "\n        </DataArray>\n";
    out << "        <DataArray type=\"UInt8\" Name=\"types\" format=\"ascii\">\n";
    for (uint32_t f : geometry_faces) out << static_cast<int>(vtk_type(N_DIM - 1, mesh->h_n_nodes_of_face(f))) << " ";
    out << "\n        </DataArray>\n";
    out << "      </Cells>\n";
    out << "    </Piece>\n";
    out << "  </UnstructuredGrid>\n";
    out << "</VTKFile>\n";
}

void DataWriter::write_vtu(const std::string & filename, rtype t) const {
    std::ofstream out(filename, std::ios::binary);
    if (!out.good()) {
        throw std::runtime_error("DataWriter::write_vtu: Could not open file: " + filename + ".");
    }

    using header_t = uint64_t;
    // Distributed runs write their owned cells; unused halo nodes are harmless
    const uint32_t n_out = mesh->n_owned();
    uint64_t len_connectivity = 0;
    for (uint32_t i = 0; i < n_out; i++) {
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
    out << "    <Piece NumberOfPoints=\"" << mesh->n_nodes << "\" NumberOfCells=\"" << n_out << "\">\n";
    out << "      <CellData>\n";
    for (const auto & field : fields) {
        data_array(vtk_float_type(), field.name, field.n_vtk_components(),
                   n_out * field.n_vtk_components() * sizeof(rtype));
    }
    out << "      </CellData>\n";
    out << "      <Points>\n";
    data_array(vtk_float_type(), "", 3, mesh->n_nodes * 3 * sizeof(rtype));
    out << "      </Points>\n";
    out << "      <Cells>\n";
    data_array("Int64", "connectivity", 1, len_connectivity * sizeof(int64_t));
    data_array("Int64", "offsets", 1, n_out * sizeof(int64_t));
    data_array("UInt8", "types", 1, n_out * sizeof(uint8_t));
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

    for (const auto & field : fields) {
        write_header(n_out * field.n_vtk_components() * sizeof(rtype));
        for (uint32_t i = 0; i < n_out; i++) {
            for (uint32_t k = 0; k < field.n_vtk_components(); k++) write_value(field.value(i, k));
        }
    }

    write_header(mesh->n_nodes * 3 * sizeof(rtype));
    for (uint32_t i_node = 0; i_node < mesh->n_nodes; i_node++) {
        FOR_I_DIM write_value(static_cast<rtype>(mesh->h_node_coords(i_node, i)));
        for (uint8_t i = N_DIM; i < 3; i++) write_value(static_cast<rtype>(0.0));
    }

    write_header(len_connectivity * sizeof(int64_t));
    for (uint32_t i = 0; i < n_out; i++) {
        const uint32_t n_nodes = mesh->h_n_nodes_of_cell(i);
        for (uint32_t j = 0; j < n_nodes; j++) {
            write_value(static_cast<int64_t>(mesh->h_node_of_cell(i, vtk_local_node(n_nodes, j))));
        }
    }

    write_header(n_out * sizeof(int64_t));
    int64_t cell_offset = 0;
    for (uint32_t i = 0; i < n_out; i++) {
        cell_offset += mesh->h_n_nodes_of_cell(i);
        write_value(cell_offset);
    }

    write_header(n_out * sizeof(uint8_t));
    for (uint32_t i = 0; i < n_out; i++) {
        write_value(vtk_type(N_DIM, mesh->h_n_nodes_of_cell(i)));
    }

    out << "\n  </AppendedData>\n";
    out << "</VTKFile>\n";
}

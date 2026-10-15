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
#include <chrono>
#include <cerrno>
#include <cmath>
#include <cstring>
#include <fcntl.h>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <numeric>
#include <sstream>
#include <unistd.h>
#include <unordered_map>

#include "common_io.h"
#include "compression.h"
#include "log.h"
#include "mesh_block.h"

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

const std::unordered_map<std::string, Compression> COMPRESSION_TYPES = {
    {"none", Compression::NONE},
    {"zlib", Compression::ZLIB},
};

double seconds_since(std::chrono::steady_clock::time_point start) {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
}

uint64_t file_bytes(const std::string & filename) {
    std::error_code error;
    const auto size = std::filesystem::file_size(filename, error);
    return error ? 0 : uint64_t(size);
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
        } else if (format != DataFormat::RESTART) {
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

    if (format == DataFormat::HDF5) {
        if (!have_hdf5()) {
            throw std::runtime_error(
                "DataWriter: format = \"hdf5\" needs a build with HDF5 (configure with Mallard_ENABLE_HDF5=ON).");
        }
        if (mesh->n_global_cells > 0 && comm::size() > 1 && !have_parallel_hdf5()) {
            throw std::runtime_error("DataWriter: format = \"hdf5\" on several ranks needs parallel HDF5.");
        }
    }

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

    const std::string compression_str = toml::find_or<std::string>(input, "compression", "none");
    const auto compression_it = COMPRESSION_TYPES.find(compression_str);
    if (compression_it == COMPRESSION_TYPES.end()) {
        throw unknown_option(COMPRESSION_TYPES, "write_data.compression", compression_str);
    }
    compression = compression_it->second;
    if (compression != Compression::NONE) {
        if (surface) throw InputError("write_data.compression: boundary-zone output is ASCII, which is not compressed.");
        if (format == DataFormat::HDF5 ? !have_hdf5_deflate() : !compression::have_zlib()) {
            throw std::runtime_error(format == DataFormat::HDF5
                                         ? "DataWriter: compression = \"zlib\" needs an HDF5 library with the deflate filter."
                                         : "DataWriter: compression = \"zlib\" needs a build with zlib (configure with "
                                           "Mallard_ENABLE_ZLIB=ON).");
        }
    }
    if (input.contains("relative_error")) {
        if (format == DataFormat::RESTART) {
            throw InputError("write_data.relative_error: restart files are lossless.");
        }
        if (compression == Compression::NONE) {
            throw InputError("write_data.relative_error needs compression = \"zlib\".");
        }
        relative_error = double(find_real(input, "relative_error"));
        if (!(relative_error > 0.0 && relative_error < 1.0)) {
            throw InputError("write_data.relative_error must be between 0 and 1.");
        }
    }

    const std::filesystem::path parent = std::filesystem::path(prefix).parent_path();
    if (!parent.empty()) {
        std::filesystem::create_directories(parent);
    }
}

std::pair<std::string, std::string> DataWriter::summary() const {
    std::string text = prefix + (format == DataFormat::RESTART ? "_*.restart"
                                 : format == DataFormat::HDF5  ? "_*.h5"
                                                               : "_*.vtu");
    if (geometry != "all") text += " (" + geometry + ")";
    if (compression == Compression::ZLIB) {
        text += relative_error > 0.0 ? ", zlib to relative error " + logging::real(relative_error) : ", zlib";
    }
    text += interval > 0 ? " every " + logging::count(interval) + " steps" : " every t = " + logging::real(double(time_interval));
    if (format != DataFormat::RESTART) {
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
    return static_cast<rtype>(n_written) * time_interval;
}

DataWriter::~DataWriter() {
    if (!pending.valid()) return;
    try {
        pending.get();
    } catch (...) {
        // Unwinding from another error, or the run ended without finish()
    }
}

void DataWriter::run(std::function<void()> job, bool allow_async) {
    auto timed = [this, job = std::move(job)]() {
        const auto start = std::chrono::steady_clock::now();
        job();
        t_writing += seconds_since(start);
    };
    if (async && allow_async) {
        pending = std::async(std::launch::async, std::move(timed));
    } else {
        timed();
    }
}

void DataWriter::finish() {
    const auto start = std::chrono::steady_clock::now();
    if (pending.valid()) pending.get();
    // Every rank's part of the files is written before any rank reads them
    if (comm::size() > 1) comm::barrier();
    t_blocking += seconds_since(start);
}

DataWriter::Values DataWriter::capture(uint32_t n_cells) const {
    Values values(fields.size());
    for (size_t f = 0; f < fields.size(); f++) {
        const Field & field = fields[f];
        const uint32_t n_comp = field.n_vtk_components();
        values[f].resize(size_t(n_cells) * n_comp);
        rtype * out = values[f].data();
        if (n_comp == 1) {
            const Data & data = *field.components[0];
            for (uint32_t i = 0; i < n_cells; i++) out[i] = data[i];
            continue;
        }
        for (uint32_t i = 0; i < n_cells; i++) {
            for (uint32_t k = 0; k < n_comp; k++) out[size_t(i) * n_comp + k] = field.value(i, k);
        }
    }
    return values;
}

DataWriter::Values DataWriter::capture_faces() const {
    Values values(fields.size());
    for (size_t f = 0; f < fields.size(); f++) {
        for (uint32_t face : geometry_faces) {
            const uint32_t cell = static_cast<uint32_t>(mesh->h_cells_of_face(face, 0));
            for (uint32_t k = 0; k < fields[f].n_vtk_components(); k++) values[f].push_back(fields[f].value(cell, k));
        }
    }
    return values;
}

void DataWriter::write(uint64_t step, rtype t, bool force, const RestartAttributes & attributes,
                       const RestartFaces & faces) {
    if (!(due(step, t) || force) || step == step_last) {
        return;
    }
    const auto start = std::chrono::steady_clock::now();
    // One snapshot at a time: the previous one is written before the next is taken
    if (pending.valid()) pending.get();
    const bool distributed = mesh->n_global_cells > 0;
    const bool root = comm::is_root();
    std::ostringstream stream;
    stream << prefix << "_" << std::setw(LEN_STEP) << std::setfill('0')
           << (interval > 0 || format == DataFormat::RESTART ? step : history.size());
    const std::string stem = stream.str();
    if (format == DataFormat::RESTART) {
        write_restart(stem + ".restart", step, t, attributes, faces);
        logging::event(step, double(t), "restart", stem + ".restart");
    } else if (format == DataFormat::HDF5) {
        history.emplace_back(t, stem + ".h5");
        // Parallel HDF5 writes collectively through MPI, which only the main thread calls
        const bool collective = distributed && comm::size() > 1;
        if (distributed || root) {
            auto job = [this, stem, step, t, root, values = capture(mesh->n_owned()), series = history]() {
                if (!hdf5_mesh_written) {
                    write_hdf5_mesh();
                    if (root) n_bytes += file_bytes(prefix + "_mesh.h5");
                }
                write_hdf5(stem, step, t, values);
                if (root) {
                    n_bytes += file_bytes(stem + ".h5");
                    write_xdmf(stem, series);
                }
            };
            run(std::move(job), !collective);
        }
        logging::event(step, double(t), "hdf5", stem + ".h5");
    } else {
        // Distributed runs: one piece per rank and a .pvtu index
        std::ostringstream piece;
        piece << stem;
        if (distributed) piece << "_p" << std::setw(4) << std::setfill('0') << comm::rank();
        const std::string filename = stem + (distributed ? ".pvtu" : ".vtu");
        history.emplace_back(t, filename);
        const int n_ranks = comm::size();
        if (distributed || root) {
            auto job = [this, piece = piece.str() + ".vtu", filename, stem, t, root, distributed, n_ranks,
                        values = surface ? capture_faces() : capture(mesh->n_owned()), series = history]() mutable {
                if (!surface) {
                    write_vtu(piece, t, values);
                } else {
                    write_vtu_faces(piece, t, values);
                }
                n_bytes += file_bytes(piece);
                if (distributed && root) {
                    write_pvtu(filename, std::filesystem::path(stem).filename().string(), n_ranks);
                }
                if (root) write_pvd(series);
            };
            run(std::move(job));
        }
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
    t_blocking += seconds_since(start);
}

void DataWriter::resume(uint64_t step, rtype t) {
    step_last = step;
    t_last = t;
    if (interval > 0) {
        n_written = step / interval + 1;
    } else {
        n_written = static_cast<uint64_t>(std::floor(t / time_interval + precision_tol(1e-9, 1e-5))) + 1;
    }
    history.clear();
    if (format == DataFormat::HDF5) {
        resume_xdmf(t);
        return;
    }
    // Keep the entries of the previous run's .pvd up to the restart time
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
constexpr uint32_t RESTART_VERSION = 5;
// Compressed restart files (version 5): cells per chunk, and the codec of the chunks
constexpr uint64_t RESTART_CHUNK = 65536;
constexpr uint32_t RESTART_CODEC_ZLIB = 1;

/**
 * @brief Restart header: magic, version, real size, cell count, variable
 *        count, step, time, then (version 2) each variable name as a uint32
 *        length and its characters, then (version 3) the attribute count as a
 *        uint64 and each attribute's name, likewise, and double value, then
 *        (version 4) the face count as a uint64 and the values per face as a
 *        uint32, then (version 5) the codec as a uint32 and the cells per
 *        chunk as a uint64. The values follow, one block of n_cells per
 *        variable, then the face keys in increasing order and the faces'
 *        values. In version 5, the blocks are compressed in chunks of
 *        chunk_cells cells, each byte-shuffled and deflated, and the header is
 *        followed by the file offset of every chunk (variable by variable) and
 *        of the faces. Files are written with the lowest version that holds
 *        their contents, which older builds read.
 */
std::vector<char> restart_header(uint64_t n_cells, uint64_t step, double t, const std::vector<std::string> & names,
                                 const RestartAttributes & attributes, uint64_t n_faces, uint64_t chunk_cells = 0) {
    std::vector<char> header;
    auto put = [&](const void * p, size_t n) {
        header.insert(header.end(), static_cast<const char *>(p), static_cast<const char *>(p) + n);
    };
    const uint32_t real_size = sizeof(rtype);
    const uint64_t n_vars = names.size();
    const uint32_t version = chunk_cells > 0 ? 5 : n_faces > 0 ? 4 : attributes.empty() ? 2 : 3;
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
    if (version < 3) return header;
    const uint64_t n_attributes = attributes.size();
    put(&n_attributes, sizeof(n_attributes));
    for (const auto & [name, value] : attributes) {
        const uint32_t length = static_cast<uint32_t>(name.size());
        put(&length, sizeof(length));
        put(name.data(), name.size());
        put(&value, sizeof(value));
    }
    if (version < 4) return header;
    const uint32_t width = RestartFaces::WIDTH;
    put(&n_faces, sizeof(n_faces));
    put(&width, sizeof(width));
    if (version < 5) return header;
    put(&RESTART_CODEC_ZLIB, sizeof(RESTART_CODEC_ZLIB));
    put(&chunk_cells, sizeof(chunk_cells));
    return header;
}

/** @brief The faces of every rank in key order: this rank's contiguous part, and where it starts. */
struct OrderedFaces {
    RestartFaces part;
    uint64_t first = 0, total = 0;
};

/** @brief Orders the faces of all ranks by key (collective if distributed). */
OrderedFaces order_faces(const RestartFaces & faces, uint64_t n_cells, bool distributed) {
    constexpr uint32_t W = RestartFaces::WIDTH;
    constexpr uint32_t RECORD = 1 + W;
    static_assert(sizeof(rtype) <= sizeof(uint64_t));
    // Per cell, each of its faces as its key and the bits of its values
    std::vector<uint32_t> order(faces.keys.size());
    std::iota(order.begin(), order.end(), 0u);
    std::sort(order.begin(), order.end(), [&](uint32_t a, uint32_t b) { return faces.keys[a] < faces.keys[b]; });
    std::vector<uint64_t> cells, offsets{0}, records;
    for (uint32_t i : order) {
        const uint64_t cell = faces.keys[i] / RESTART_FACES_PER_CELL;
        if (cells.empty() || cells.back() != cell) {
            cells.push_back(cell);
            offsets.push_back(offsets.back());
        }
        records.push_back(faces.keys[i]);
        for (uint32_t w = 0; w < W; w++) {
            uint64_t bits = 0;
            std::memcpy(&bits, &faces.values[size_t(i) * W + w], sizeof(rtype));
            records.push_back(bits);
        }
        offsets.back() += RECORD;
    }
    const GlobalOrder by_cell(cells, n_cells, distributed);
    const std::vector<uint64_t> block = by_cell.gather_csr(offsets, records).second;
    OrderedFaces out;
    const uint64_t n = block.size() / RECORD;
    out.part.keys.resize(n);
    out.part.values.resize(n * W);
    for (uint64_t k = 0; k < n; k++) {
        out.part.keys[k] = block[k * RECORD];
        for (uint32_t w = 0; w < W; w++) {
            std::memcpy(&out.part.values[k * W + w], &block[k * RECORD + 1 + w], sizeof(rtype));
        }
    }
    const std::vector<uint64_t> counts =
        distributed ? comm::allgatherv(std::vector<uint64_t>{n}) : std::vector<uint64_t>{n};
    out.first = std::accumulate(counts.begin(), counts.begin() + (distributed ? comm::rank() : 0), uint64_t(0));
    out.total = std::accumulate(counts.begin(), counts.end(), uint64_t(0));
    return out;
}

/**
 * @brief The compressed chunks of the cells [first, first + n) of a restart
 *        file, first a multiple of RESTART_CHUNK: chunk c of variable v is
 *        data[v * n_chunks + c].
 */
struct RestartChunks {
    uint64_t n_chunks = 0;
    std::vector<std::vector<char>> data;

    RestartChunks(const std::vector<std::vector<rtype>> & values, uint64_t n, unsigned n_threads)
        : n_chunks((n + RESTART_CHUNK - 1) / RESTART_CHUNK), data(values.size() * n_chunks) {
        compression::parallel_for(data.size(), n_threads, [&](size_t k) {
            const uint64_t c = k % n_chunks, first = c * RESTART_CHUNK;
            const uint64_t n_cells = std::min(RESTART_CHUNK, n - first);
            data[k] = compression::deflate(values[k / n_chunks].data() + first, n_cells * sizeof(rtype), sizeof(rtype));
        });
    }

    /** @brief The chunk count, then each chunk's size, for chunk_offsets. */
    std::vector<uint64_t> sizes() const {
        std::vector<uint64_t> out{n_chunks};
        for (const auto & chunk : data) out.push_back(chunk.size());
        return out;
    }
};

/**
 * @brief File offsets of every chunk of a restart file (variable by variable)
 *        and of its faces, from each rank's RestartChunks::sizes in rank order.
 */
std::vector<uint64_t> chunk_offsets(const std::vector<uint64_t> & all_sizes, uint64_t n_vars, uint64_t n_chunks,
                                    uint64_t data_start) {
    std::vector<uint64_t> size(n_vars * n_chunks, 0);
    uint64_t c0 = 0;
    for (size_t at = 0; at < all_sizes.size();) {
        const uint64_t n = all_sizes[at++];
        for (uint64_t v = 0; v < n_vars; v++) {
            for (uint64_t c = 0; c < n; c++) size[v * n_chunks + c0 + c] = all_sizes[at++];
        }
        c0 += n;
    }
    if (c0 != n_chunks) throw std::logic_error("DataWriter: restart chunks do not cover the cells.");
    std::vector<uint64_t> offsets{data_start};
    for (uint64_t s : size) offsets.push_back(offsets.back() + s);
    return offsets;
}

void write_at(int fd, uint64_t offset, const void * data, size_t n, const std::string & filename) {
    const char * p = static_cast<const char *>(data);
    while (n > 0) {
        const ssize_t written = ::pwrite(fd, p, std::min<size_t>(n, size_t(1) << 30), static_cast<off_t>(offset));
        if (written < 0 && errno == EINTR) continue;
        if (written <= 0) throw std::runtime_error("DataWriter: could not write " + filename + ".");
        p += written;
        n -= static_cast<size_t>(written);
        offset += static_cast<uint64_t>(written);
    }
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

const rtype * RestartData::face(uint64_t key) const {
    const auto it = std::lower_bound(faces.keys.begin(), faces.keys.end(), key);
    if (it == faces.keys.end() || *it != key) return nullptr;
    return &faces.values[size_t(it - faces.keys.begin()) * RestartFaces::WIDTH];
}

void DataWriter::write_restart(const std::string & filename, uint64_t step, rtype t,
                               const RestartAttributes & attributes, const RestartFaces & faces) {
    if (mesh->n_global_cells > 0) {
        write_restart_distributed(filename, step, t, attributes, faces);
        return;
    }
    // Serial runs on several ranks (tests) hold the whole mesh on every rank
    if (!comm::is_root()) return;
    auto job = [this, filename, step, t, attributes, ordered = order_faces(faces, mesh->n_cells, false),
                values = capture(mesh->n_cells)]() {
        std::ofstream out(filename, std::ios::binary);
        if (!out.good()) {
            throw std::runtime_error("DataWriter::write_restart: Could not open file: " + filename + ".");
        }
        std::vector<std::string> names;
        for (const auto & field : fields) names.push_back(field.name);
        auto put = [&](const void * data, size_t n) { out.write(static_cast<const char *>(data), std::streamsize(n)); };
        const uint64_t n_cells = mesh->n_cells;
        if (compression == Compression::NONE) {
            const std::vector<char> header = restart_header(n_cells, step, double(t), names, attributes, ordered.total);
            put(header.data(), header.size());
            for (const auto & var : values) put(var.data(), var.size() * sizeof(rtype));
        } else {
            const RestartChunks chunks(values, n_cells, io_threads);
            const std::vector<char> header =
                restart_header(n_cells, step, double(t), names, attributes, ordered.total, RESTART_CHUNK);
            const std::vector<uint64_t> offsets = chunk_offsets(
                chunks.sizes(), names.size(), chunks.n_chunks,
                header.size() + (names.size() * chunks.n_chunks + 1) * sizeof(uint64_t));
            put(header.data(), header.size());
            put(offsets.data(), offsets.size() * sizeof(uint64_t));
            for (const auto & chunk : chunks.data) put(chunk.data(), chunk.size());
        }
        put(ordered.part.keys.data(), ordered.part.keys.size() * sizeof(uint64_t));
        put(ordered.part.values.data(), ordered.part.values.size() * sizeof(rtype));
        out.close();
        if (!out) throw std::runtime_error("DataWriter::write_restart: Could not write file: " + filename + ".");
        n_bytes += file_bytes(filename);
    };
    run(std::move(job));
}

void DataWriter::write_restart_distributed(const std::string & filename, uint64_t step, rtype t,
                                           const RestartAttributes & attributes, const RestartFaces & faces) {
#ifdef Mallard_HAS_MPI
    const OrderedFaces ordered = order_faces(faces, mesh->n_global_cells, true);
    const uint64_t n_global = mesh->n_global_cells;
    std::vector<std::string> names;
    for (const auto & field : fields) names.push_back(field.name);
    const uint32_t n_owned = mesh->n_owned();
    constexpr uint32_t W = RestartFaces::WIDTH;
    if (async || compression != Compression::NONE) {
        // Each rank gathers a run of whole chunks of cells in global order, and
        // writes it (and its faces) at offsets that all ranks agree on here,
        // so that writing needs no MPI
        if (!restart_order_built) {
            const std::vector<uint64_t> ids(mesh->h_global_cell_id.begin(), mesh->h_global_cell_id.begin() + n_owned);
            restart_order = GlobalOrder(ids, n_global, true, RESTART_CHUNK);
            restart_order_built = true;
        }
        std::vector<std::vector<rtype>> block(fields.size());
        std::vector<rtype> owned(n_owned);
        for (size_t v = 0; v < fields.size(); v++) {
            for (uint32_t i = 0; i < n_owned; i++) owned[i] = fields[v].value(i, 0);
            block[v] = restart_order.gather<rtype>(owned, 1);
        }
        const uint64_t first = restart_order.first(), n = restart_order.size();
        const uint64_t n_vars = fields.size();
        std::shared_ptr<RestartChunks> chunks;
        std::vector<char> header;
        std::vector<uint64_t> offsets;  // file offset of each variable's block of this rank, then of the faces
        std::vector<uint64_t> table;
        if (compression == Compression::NONE) {
            header = restart_header(n_global, step, double(t), names, attributes, ordered.total);
            for (uint64_t v = 0; v < n_vars; v++) offsets.push_back(header.size() + (v * n_global + first) * sizeof(rtype));
            offsets.push_back(header.size() + n_vars * n_global * sizeof(rtype));
        } else {
            chunks = std::make_shared<RestartChunks>(block, n, io_threads);
            const uint64_t n_chunks = (n_global + RESTART_CHUNK - 1) / RESTART_CHUNK;
            header = restart_header(n_global, step, double(t), names, attributes, ordered.total, RESTART_CHUNK);
            table = chunk_offsets(comm::allgatherv(chunks->sizes()), n_vars, n_chunks,
                                  header.size() + (n_vars * n_chunks + 1) * sizeof(uint64_t));
            for (uint64_t v = 0; v < n_vars; v++) offsets.push_back(table[v * n_chunks + first / RESTART_CHUNK]);
            offsets.push_back(table.back());
            block.clear();
        }
        const bool root = comm::is_root();
        if (root) {
            const int fd = ::open(filename.c_str(), O_CREAT | O_TRUNC | O_WRONLY, 0644);
            if (fd < 0) throw std::runtime_error("DataWriter::write_restart: Could not open file: " + filename + ".");
            ::close(fd);
        }
        comm::barrier();
        auto job = [this, filename, root, header = std::move(header), table = std::move(table),
                    offsets = std::move(offsets), block = std::move(block), chunks, ordered]() {
            const int fd = ::open(filename.c_str(), O_WRONLY);
            if (fd < 0) throw std::runtime_error("DataWriter::write_restart: Could not open file: " + filename + ".");
            uint64_t written = 0;
            auto put = [&](uint64_t at, const void * data, size_t size) {
                write_at(fd, at, data, size, filename);
                written += size;
            };
            try {
                if (root) {
                    put(0, header.data(), header.size());
                    put(header.size(), table.data(), table.size() * sizeof(uint64_t));
                }
                for (size_t v = 0; v + 1 < offsets.size(); v++) {
                    if (!chunks) {
                        put(offsets[v], block[v].data(), block[v].size() * sizeof(rtype));
                        continue;
                    }
                    uint64_t at = offsets[v];
                    for (uint64_t c = 0; c < chunks->n_chunks; c++) {
                        const auto & chunk = chunks->data[v * chunks->n_chunks + c];
                        put(at, chunk.data(), chunk.size());
                        at += chunk.size();
                    }
                }
                const uint64_t faces_at = offsets.back();
                put(faces_at + ordered.first * sizeof(uint64_t), ordered.part.keys.data(),
                    ordered.part.keys.size() * sizeof(uint64_t));
                put(faces_at + ordered.total * sizeof(uint64_t) + ordered.first * W * sizeof(rtype),
                    ordered.part.values.data(), ordered.part.values.size() * sizeof(rtype));
            } catch (...) {
                ::close(fd);
                throw;
            }
            if (::close(fd) != 0) throw std::runtime_error("DataWriter::write_restart: Could not write " + filename + ".");
            n_bytes += written;
        };
        run(std::move(job));
        return;
    }
    // Same layout as a serial restart, cells in global order: each rank writes
    // its owned cells at their global offsets, so any rank count can read it
    const auto start = std::chrono::steady_clock::now();
    MPI_File fh;
    if (MPI_File_open(comm::world(), filename.c_str(), MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &fh) !=
        MPI_SUCCESS) {
        throw std::runtime_error("DataWriter::write_restart: Could not open file: " + filename + ".");
    }
    MPI_File_set_size(fh, 0);
    const std::vector<char> header = restart_header(n_global, step, double(t), names, attributes, ordered.total);
    if (comm::is_root()) {
        MPI_File_write_at(fh, 0, header.data(), static_cast<int>(header.size()), MPI_BYTE, MPI_STATUS_IGNORE);
    }
    const MPI_Offset header_size = static_cast<MPI_Offset>(header.size());
    const MPI_Datatype real_type = sizeof(rtype) == sizeof(double) ? MPI_DOUBLE : MPI_FLOAT;
    std::vector<MPI_Aint> displacements(n_owned);
    for (uint32_t i = 0; i < n_owned; i++) displacements[i] = static_cast<MPI_Aint>(mesh->h_global_cell_id[i] * sizeof(rtype));
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
    // Each rank's faces are a contiguous run of the keys and of the values
    const MPI_Offset faces_at = header_size + MPI_Offset(fields.size() * n_global * sizeof(rtype));
    MPI_File_set_view(fh, 0, MPI_BYTE, MPI_BYTE, "native", MPI_INFO_NULL);
    MPI_File_write_at_all(fh, faces_at + MPI_Offset(ordered.first * sizeof(uint64_t)), ordered.part.keys.data(),
                          static_cast<int>(ordered.part.keys.size()), MPI_UINT64_T, MPI_STATUS_IGNORE);
    MPI_File_write_at_all(fh,
                          faces_at + MPI_Offset(ordered.total * sizeof(uint64_t) + ordered.first * W * sizeof(rtype)),
                          ordered.part.values.data(), static_cast<int>(ordered.part.values.size()), real_type,
                          MPI_STATUS_IGNORE);
    MPI_File_close(&fh);
    if (comm::is_root()) n_bytes += file_bytes(filename);
    t_writing += seconds_since(start);
#else
    (void)filename;
    (void)step;
    (void)t;
    (void)attributes;
    (void)faces;
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
    uint64_t n_vars, n_faces = 0, chunk_cells = 0;
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
        if (version >= 4) {
            uint32_t width = 0;
            in.read(reinterpret_cast<char *>(&n_faces), sizeof(n_faces));
            in.read(reinterpret_cast<char *>(&width), sizeof(width));
            if (in.good() && width != RestartFaces::WIDTH) {
                throw std::runtime_error("Restart file " + filename + " has " + std::to_string(width) +
                                         " values per face, not " + std::to_string(RestartFaces::WIDTH) + ".");
            }
        }
        if (version >= 5) {
            uint32_t codec = 0;
            in.read(reinterpret_cast<char *>(&codec), sizeof(codec));
            in.read(reinterpret_cast<char *>(&chunk_cells), sizeof(chunk_cells));
            if (in.good() && (codec != RESTART_CODEC_ZLIB || chunk_cells == 0)) {
                throw std::runtime_error("Restart file " + filename + " has an unknown compression.");
            }
            if (!compression::have_zlib()) {
                throw std::runtime_error("Restart file " + filename +
                                         " is compressed, which needs a build with zlib (Mallard_ENABLE_ZLIB=ON).");
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
    const std::streamoff header = in.tellg();
    std::vector<uint64_t> chunk_at;  // version 5: file offset of each chunk, then of the faces
    if (chunk_cells > 0) {
        const uint64_t n_chunks = (data.n_cells + chunk_cells - 1) / chunk_cells;
        chunk_at.resize(n_vars * n_chunks + 1);
        in.read(reinterpret_cast<char *>(chunk_at.data()), static_cast<std::streamsize>(chunk_at.size() * sizeof(uint64_t)));
        if (!in.good() || !std::is_sorted(chunk_at.begin(), chunk_at.end())) {
            throw std::runtime_error("Restart file " + filename + " has a corrupt chunk table.");
        }
    }
    // Chunk c of variable v of a version 5 file, decompressed
    std::vector<char> compressed;
    auto read_chunk = [&](uint64_t v, uint64_t c, std::vector<rtype> & out) {
        const uint64_t n_chunks = (chunk_at.size() - 1) / n_vars;
        const uint64_t k = v * n_chunks + c;
        out.resize(std::min(chunk_cells, data.n_cells - c * chunk_cells));
        compressed.resize(chunk_at[k + 1] - chunk_at[k]);
        in.seekg(static_cast<std::streamoff>(chunk_at[k]));
        in.read(compressed.data(), static_cast<std::streamsize>(compressed.size()));
        if (!in.good()) throw std::runtime_error("Restart file " + filename + " is truncated.");
        compression::inflate(compressed.data(), compressed.size(), out.data(), out.size() * sizeof(rtype),
                             sizeof(rtype));
    };
    if (!cells) {
        data.fields.assign(n_vars, std::vector<rtype>(data.n_cells));
        if (chunk_cells > 0) {
            std::vector<rtype> chunk;
            for (uint64_t v = 0; v < n_vars; v++) {
                for (uint64_t c = 0; c * chunk_cells < data.n_cells; c++) {
                    read_chunk(v, c, chunk);
                    std::copy(chunk.begin(), chunk.end(), data.fields[v].begin() + std::ptrdiff_t(c * chunk_cells));
                }
            }
        } else {
            for (auto & var : data.fields) {
                in.read(reinterpret_cast<char *>(var.data()), static_cast<std::streamsize>(data.n_cells * sizeof(rtype)));
            }
        }
    } else {
        // Runs of consecutive global ids, read with one seek each (or one chunk each)
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
                if (chunk_cells > 0) {
                    const uint64_t c = (*cells)[order[a]] / chunk_cells;
                    while (b < order.size() && (*cells)[order[b]] / chunk_cells == c) b++;
                    read_chunk(v, c, run);
                    for (size_t k = a; k < b; k++) data.fields[v][order[k]] = run[(*cells)[order[k]] - c * chunk_cells];
                    a = b;
                    continue;
                }
                while (b < order.size() && (*cells)[order[b]] == (*cells)[order[b - 1]] + 1) b++;
                run.resize(b - a);
                in.seekg(header + static_cast<std::streamoff>((v * data.n_cells + (*cells)[order[a]]) * sizeof(rtype)));
                in.read(reinterpret_cast<char *>(run.data()), static_cast<std::streamsize>(run.size() * sizeof(rtype)));
                for (size_t k = a; k < b; k++) data.fields[v][order[k]] = run[k - a];
                a = b;
            }
        }
    }
    // Every rank reads all faces: they are few, and any cell may have one
    in.seekg(chunk_cells > 0 ? static_cast<std::streamoff>(chunk_at.back())
                             : header + static_cast<std::streamoff>(n_vars * data.n_cells * sizeof(rtype)));
    data.faces.keys.resize(n_faces);
    data.faces.values.resize(n_faces * RestartFaces::WIDTH);
    in.read(reinterpret_cast<char *>(data.faces.keys.data()), static_cast<std::streamsize>(n_faces * sizeof(uint64_t)));
    in.read(reinterpret_cast<char *>(data.faces.values.data()),
            static_cast<std::streamsize>(data.faces.values.size() * sizeof(rtype)));
    if (!std::is_sorted(data.faces.keys.begin(), data.faces.keys.end())) {
        throw std::runtime_error("Restart file " + filename + " has unordered faces.");
    }
    if (!in.good()) {
        throw std::runtime_error("Restart file " + filename + " is truncated.");
    }
    return data;
}

void DataWriter::write_pvtu(const std::string & filename, const std::string & stem, int n_ranks) const {
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
    for (int r = 0; r < n_ranks; r++) {
        out << "    <Piece Source=\"" << stem << "_p" << std::setw(4) << std::setfill('0') << r << ".vtu\"/>\n";
    }
    out << "  </PUnstructuredGrid>\n";
    out << "</VTKFile>\n";
}

void DataWriter::write_pvd(const std::vector<std::pair<rtype, std::string>> & series) const {
    std::ofstream out(prefix + ".pvd");
    out << "<?xml version=\"1.0\"?>\n";
    out << "<VTKFile type=\"Collection\" version=\"0.1\" byte_order=\"" << endianness() << "\">\n";
    out << "  <Collection>\n";
    for (const auto & [t, filename] : series) {
        out << "    <DataSet timestep=\"" << std::setprecision(std::numeric_limits<double>::max_digits10) << t << "\" file=\""
            << std::filesystem::path(filename).filename().string() << "\"/>\n";
    }
    out << "  </Collection>\n";
    out << "</VTKFile>\n";
}

void DataWriter::write_vtu_faces(const std::string & filename, rtype t, const Values & values) const {
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
    for (size_t i_field = 0; i_field < fields.size(); i_field++) {
        const Field & field = fields[i_field];
        out << "        <DataArray type=\"Float64\" Name=\"" << field.name << "\" ";
        if (field.n_vtk_components() > 1) out << "NumberOfComponents=\"" << field.n_vtk_components() << "\" ";
        out << "format=\"ascii\">\n";
        for (rtype value : values[i_field]) out << static_cast<double>(value) << " ";
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

namespace {

/** @brief Bytes of a vector. */
template <typename T>
std::vector<char> as_bytes(const std::vector<T> & values) {
    std::vector<char> bytes(values.size() * sizeof(T));
    std::memcpy(bytes.data(), values.data(), bytes.size());
    return bytes;
}

/**
 * @brief An appended VTU array compressed as vtkZLibDataCompressor reads it:
 *        the UInt64 block count, block size and size of the last, partial
 *        block (0: full), each block's compressed size, then the blocks.
 */
std::vector<char> vtu_zlib_array(const void * data, size_t n_bytes, unsigned n_threads) {
    constexpr size_t BLOCK = size_t(1) << 20;
    const size_t n_blocks = (n_bytes + BLOCK - 1) / BLOCK;
    std::vector<std::vector<char>> blocks(n_blocks);
    compression::parallel_for(n_blocks, n_threads, [&](size_t b) {
        blocks[b] = compression::deflate(static_cast<const char *>(data) + b * BLOCK, std::min(BLOCK, n_bytes - b * BLOCK));
    });
    std::vector<uint64_t> header{n_blocks, BLOCK, n_bytes % BLOCK};
    for (const auto & block : blocks) header.push_back(block.size());
    std::vector<char> out = as_bytes(header);
    for (const auto & block : blocks) out.insert(out.end(), block.begin(), block.end());
    return out;
}

} // namespace

void DataWriter::write_vtu(const std::string & filename, rtype t, Values & values) {
    std::ofstream out(filename, std::ios::binary);
    if (!out.good()) {
        throw std::runtime_error("DataWriter::write_vtu: Could not open file: " + filename + ".");
    }

    using header_t = uint64_t;
    // Distributed runs write their owned cells; unused halo nodes are harmless
    const uint32_t n_out = mesh->n_owned();
    const bool zlib = compression == Compression::ZLIB;

    // The mesh's arrays (points, connectivity, offsets, types): the same in
    // every snapshot, so compressed ones are kept
    std::vector<std::vector<char>> mesh_arrays;
    std::vector<std::vector<char>> & arrays = zlib ? vtu_mesh : mesh_arrays;
    if (arrays.empty()) {
        std::vector<rtype> points;
        points.reserve(size_t(mesh->n_nodes) * 3);
        for (uint32_t i_node = 0; i_node < mesh->n_nodes; i_node++) {
            FOR_I_DIM points.push_back(static_cast<rtype>(mesh->h_node_coords(i_node, i)));
            for (uint8_t i = N_DIM; i < 3; i++) points.push_back(static_cast<rtype>(0.0));
        }
        std::vector<int64_t> connectivity, offsets;
        std::vector<uint8_t> types;
        int64_t cell_offset = 0;
        for (uint32_t i = 0; i < n_out; i++) {
            const uint32_t n_nodes = mesh->h_n_nodes_of_cell(i);
            for (uint32_t j = 0; j < n_nodes; j++) {
                connectivity.push_back(static_cast<int64_t>(mesh->h_node_of_cell(i, vtk_local_node(n_nodes, j))));
            }
            cell_offset += n_nodes;
            offsets.push_back(cell_offset);
            types.push_back(vtk_type(N_DIM, n_nodes));
        }
        arrays = {as_bytes(points), as_bytes(connectivity), as_bytes(offsets), as_bytes(types)};
        if (zlib) {
            for (auto & array : arrays) array = vtu_zlib_array(array.data(), array.size(), io_threads);
        }
    }
    if (relative_error > 0.0) {
        for (auto & field : values) compression::round_to_relative_error(field, relative_error);
    }
    std::vector<std::vector<char>> field_arrays;
    if (zlib) {
        for (const auto & field : values) {
            field_arrays.push_back(vtu_zlib_array(field.data(), field.size() * sizeof(rtype), io_threads));
        }
    }

    // Header with appended-data offsets
    uint64_t offset = 0;
    auto data_array = [&](const std::string & type, const std::string & name, uint32_t n_comp, uint64_t n_bytes_in) {
        out << "        <DataArray type=\"" << type << "\" ";
        if (!name.empty()) out << "Name=\"" << name << "\" ";
        if (n_comp > 1) out << "NumberOfComponents=\"" << n_comp << "\" ";
        out << "format=\"appended\" offset=\"" << offset << "\"/>\n";
        offset += n_bytes_in;
    };
    // An array's appended bytes: its size and its values, or its compressed blocks
    auto appended_size = [&](size_t i_field, const std::vector<char> * mesh_array) -> uint64_t {
        if (mesh_array) return zlib ? mesh_array->size() : sizeof(header_t) + mesh_array->size();
        return zlib ? field_arrays[i_field].size() : sizeof(header_t) + values[i_field].size() * sizeof(rtype);
    };

    out << "<?xml version=\"1.0\"?>\n";
    out << "<VTKFile type=\"UnstructuredGrid\" version=\"1.0\" byte_order=\"" << endianness()
        << "\" header_type=\"UInt64\"" << (zlib ? " compressor=\"vtkZLibDataCompressor\"" : "") << ">\n";
    out << "  <UnstructuredGrid>\n";
    out << "    <FieldData>\n";
    out << "      <DataArray type=\"Float64\" Name=\"TIME\" NumberOfTuples=\"1\" format=\"ascii\">"
        << std::setprecision(17) << static_cast<double>(t) << "</DataArray>\n";
    out << "    </FieldData>\n";
    out << "    <Piece NumberOfPoints=\"" << mesh->n_nodes << "\" NumberOfCells=\"" << n_out << "\">\n";
    out << "      <CellData>\n";
    for (size_t f = 0; f < fields.size(); f++) {
        data_array(vtk_float_type(), fields[f].name, fields[f].n_vtk_components(), appended_size(f, nullptr));
    }
    out << "      </CellData>\n";
    out << "      <Points>\n";
    data_array(vtk_float_type(), "", 3, appended_size(0, &arrays[0]));
    out << "      </Points>\n";
    out << "      <Cells>\n";
    data_array("Int64", "connectivity", 1, appended_size(0, &arrays[1]));
    data_array("Int64", "offsets", 1, appended_size(0, &arrays[2]));
    data_array("UInt8", "types", 1, appended_size(0, &arrays[3]));
    out << "      </Cells>\n";
    out << "    </Piece>\n";
    out << "  </UnstructuredGrid>\n";
    out << "  <AppendedData encoding=\"raw\">\n_";

    auto append = [&](const void * data, uint64_t n_bytes_in) {
        if (!zlib) out.write(reinterpret_cast<const char *>(&n_bytes_in), sizeof(header_t));
        out.write(static_cast<const char *>(data), static_cast<std::streamsize>(n_bytes_in));
    };
    for (size_t f = 0; f < fields.size(); f++) {
        if (zlib) {
            append(field_arrays[f].data(), field_arrays[f].size());
        } else {
            append(values[f].data(), values[f].size() * sizeof(rtype));
        }
    }
    for (const auto & array : arrays) append(array.data(), array.size());

    out << "\n  </AppendedData>\n";
    out << "</VTKFile>\n";
    out.close();
    if (!out) throw std::runtime_error("DataWriter::write_vtu: Could not write file: " + filename + ".");
}

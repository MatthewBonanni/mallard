/**
 * @file output_reader.cpp
 * @brief Reader of every file format Mallard writes or reads.
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 */

#include "output_reader.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <mutex>
#include <optional>
#include <regex>
#include <set>
#include <sstream>
#include <stdexcept>

#include "common_io.h"
#include "compression.h"
#include "field_units.h"
#include "file_format.h"
#include "json.h"
#include "log.h"
#include "mesh_block.h"
#include "provenance.h"
#ifdef Mallard_HAS_HDF5
#include "hdf5_util.h"
#endif

namespace fs = std::filesystem;

namespace output {

namespace {

[[noreturn]] void fail(const std::string & path, const std::string & what) {
    throw std::runtime_error(path + ": " + what);
}

bool ends_with(const std::string & s, const std::string & suffix) {
    return s.size() >= suffix.size() && s.compare(s.size() - suffix.size(), suffix.size(), suffix) == 0;
}

std::string extension(const std::string & path) { return fs::path(path).extension().string(); }

std::string read_bytes(const std::string & path) {
    std::ifstream in(path, std::ios::binary);
    if (!in.good()) throw cannot_read("the file", path);
    std::stringstream text;
    text << in.rdbuf();
    return text.str();
}

std::string sibling(const std::string & path, const std::string & name) {
    return (fs::path(path).parent_path() / name).string();
}

/** @brief Name of a VTK cell type, as a mesh summary lists it. */
std::string vtk_type_name(int type) {
    switch (type) {
        case 1: return "vertex";
        case 3: return "line";
        case 4: return "polyline";
        case 5: return "triangle";
        case 7: return "polygon";
        case 9: return "quadrilateral";
        case 10: return "tetrahedron";
        case 12: return "hexahedron";
        case 13: return "prism";
        case 14: return "pyramid";
        case 27: return "curved pyramid";
        case 69: return "curved triangle";
        case 70: return "curved quadrilateral";
        case 71: return "curved tetrahedron";
        case 72: return "curved hexahedron";
        case 73: return "curved prism";
        default: return "VTK type " + std::to_string(type);
    }
}

/** @brief Name of a cell of a mesh file by its corner count. */
std::string corner_type_name(int dim, uint64_t corners) {
    if (dim == 2) return corners == 3 ? "triangle" : corners == 4 ? "quadrilateral" : std::to_string(corners) + "-gon";
    switch (corners) {
        case 4: return "tetrahedron";
        case 5: return "pyramid";
        case 6: return "prism";
        case 8: return "hexahedron";
        default: return std::to_string(corners) + "-node cell";
    }
}

void count_types(MeshSummary & m, const std::map<std::string, uint64_t> & counts) {
    m.cell_types.assign(counts.begin(), counts.end());
}

/** @brief Bounds of points stored with `width` coordinates each. */
void bound(MeshSummary & m, const std::vector<double> & xyz, size_t width) {
    if (xyz.empty() || width == 0) return;
    m.has_bounds = true;
    m.lo.fill(0.0);
    m.hi.fill(0.0);
    for (size_t i = 0; i < width && i < 3; i++) {
        m.lo[i] = m.hi[i] = xyz[i];
        for (size_t p = i; p < xyz.size(); p += width) {
            m.lo[i] = std::min(m.lo[i], xyz[p]);
            m.hi[i] = std::max(m.hi[i], xyz[p]);
        }
    }
}

/** @brief A field's description from Mallard's units table, for files that record none (restart files). */
void describe_from_table(Field & f) {
    const auto d = field_units::find(f.name);
    if (!d) return;
    f.units = d->units();
    f.long_name = d->long_name;
}

/** @brief A snapshot at (t, step) of file, with its snapshot block if json holds one: the block's step, time and run then hold. */
Snapshot snapshot_of(double t, int64_t step, const std::string & file, uint64_t index, const std::string & json_text) {
    Snapshot s{t, step, file, index, false, {}, ""};
    if (json_text.empty()) return s;
    try {
        s.block = snapshot_block::parse(json_text);
        s.has_block = true;
        if (s.block.step >= 0) s.step = s.block.step;
        if (std::isfinite(s.block.time)) s.t = s.block.time;
        s.run_id = s.block.run_id;
    } catch (const std::exception &) {
        // Reported by note_snapshots: a block that does not parse is no block
    }
    return s;
}

void note_provenance(Contents & c) {
    if (c.run.empty()) {
        c.notes.push_back("no run block: written before Mallard 0.9, or not by Mallard");
        return;
    }
    try {
        const json::Value p = json::parse(c.run);
        const json::Value * v = p.find("version");
        const json::Value * schema = p.find("schema");
        if (v && v->number > provenance::VERSION) {
            c.notes.push_back("run block version " + std::to_string(int(v->number)) + " is newer than this build's (" +
                              std::to_string(provenance::VERSION) + "): shown as recorded");
        }
        if (schema && schema->string == provenance::LEGACY_SCHEMA) {
            c.notes.push_back("run block of version " + std::to_string(v ? int(v->number) : 0) +
                              ", under its earlier name \"provenance\" (written before Mallard 0.10): it records no "
                              "run id");
        }
        if (c.dim == 0) {
            const json::Value * build = p.find("build");
            const json::Value * d = build ? build->find("dim") : nullptr;
            if (d && d->type == json::Value::Type::NUMBER) c.dim = int(d->number);
        }
    } catch (const std::exception & e) {
        c.notes.push_back(std::string("the run block is not valid JSON (") + e.what() + ")");
    }
}

/** @brief The run id of a run block, or "". */
std::string run_id_of(const std::string & run) { return provenance::find_string(run, "run", "run_id"); }

/** @brief Notes on snapshots that record less than the snapshot block (older files), and the run of each. */
void note_snapshots(Contents & c) {
    if (c.kind == Kind::mesh || c.snapshots.empty()) return;
    const std::string run = run_id_of(c.run);
    bool all_blocks = true, any_step = false, all_steps = true;
    for (Snapshot & s : c.snapshots) {
        if (s.run_id.empty() && c.kind != Kind::table) s.run_id = run;
        all_blocks = all_blocks && s.has_block;
        any_step = any_step || s.step >= 0;
        all_steps = all_steps && s.step >= 0;
    }
    if (c.kind == Kind::table) return;
    if (!all_steps) {
        c.notes.push_back(std::string(any_step ? "some snapshots record" : "the file records") +
                          " no step (written before Mallard 0.10): those snapshots are numbered by their position "
                          "in the file, from 0, and their step is shown as -");
    }
    if (!all_blocks && c.format != "probe HDF5") {
        c.notes.push_back("no snapshot block (written before Mallard 0.10): step and time are the format's own "
                          "records; reason, run and stored precision are unknown");
    }
}

// ---------------------------------------------------------------------------
// Restart files

Contents open_restart(const std::string & path) {
    std::ifstream in(path, std::ios::binary);
    if (!in.good()) throw cannot_read("the restart file", path);
    Contents c;
    c.path = path;
    c.kind = Kind::restart;
    c.format = "restart";
    c.restart = read_restart_header(in, path);
    const RestartHeader & h = c.restart;
    c.version = h.version;
    c.real_size = h.real_size;
    c.compressed = h.chunk_cells > 0;
    c.dim = std::find(h.names.begin(), h.names.end(), "RHOU_Z") != h.names.end() ? 3 : 2;
    c.mesh.n_cells = h.n_cells;
    c.global_order = true;
    c.snapshots.push_back(snapshot_of(h.t, static_cast<int64_t>(h.step), path, 0, h.snapshot));
    for (const auto & name : h.names) {
        Field f;
        f.name = name;
        f.location = "cell";
        describe_from_table(f);
        c.fields.push_back(f);
    }
    c.run = h.provenance;
    return c;
}

std::vector<double> read_restart_field(const Contents & c, size_t v) {
    const RestartHeader & h = c.restart;
    std::ifstream in(c.path, std::ios::binary);
    if (!in.good()) throw cannot_read("the restart file", c.path);
    const size_t width = h.real_size;
    std::vector<char> bytes(h.n_cells * width);
    if (h.chunk_cells == 0) {
        in.seekg(static_cast<std::streamoff>(h.data_offset + v * h.n_cells * width));
        in.read(bytes.data(), static_cast<std::streamsize>(bytes.size()));
    } else {
        if (!compression::have_zlib()) {
            fail(c.path, "is compressed, which needs a build with zlib (Mallard_ENABLE_ZLIB=ON)");
        }
        const uint64_t n_chunks = (h.n_cells + h.chunk_cells - 1) / h.chunk_cells;
        std::vector<char> packed;
        for (uint64_t k = 0; k < n_chunks; k++) {
            const uint64_t at = v * n_chunks + k;
            packed.resize(h.chunk_at[at + 1] - h.chunk_at[at]);
            in.seekg(static_cast<std::streamoff>(h.chunk_at[at]));
            in.read(packed.data(), static_cast<std::streamsize>(packed.size()));
            const uint64_t first = k * h.chunk_cells, n = std::min(h.chunk_cells, h.n_cells - first);
            compression::inflate(packed.data(), packed.size(), bytes.data() + first * width, n * width, width);
        }
    }
    if (!in.good()) fail(c.path, "is truncated");
    std::vector<double> out(h.n_cells);
    for (uint64_t i = 0; i < h.n_cells; i++) {
        if (width == 8) {
            std::memcpy(&out[i], bytes.data() + i * 8, 8);
        } else {
            float x;
            std::memcpy(&x, bytes.data() + i * 4, 4);
            out[i] = static_cast<double>(x);
        }
    }
    return out;
}

// ---------------------------------------------------------------------------
// VTK XML: VTU, PVTU, VTP, PVD

/** @brief The attributes of an XML start tag's text (name="value" ...). */
std::map<std::string, std::string> attributes(const std::string & tag) {
    std::map<std::string, std::string> out;
    static const std::regex attribute(R"re(([A-Za-z_][\w.-]*)="([^"]*)")re");
    for (auto it = std::sregex_iterator(tag.begin(), tag.end(), attribute); it != std::sregex_iterator(); ++it) {
        out[(*it)[1].str()] = (*it)[2].str();
    }
    return out;
}

/** @brief Each start tag <name ...> of text, as its attributes. */
std::vector<std::map<std::string, std::string>> tags(const std::string & text, const std::string & name) {
    std::vector<std::map<std::string, std::string>> out;
    const std::string open = "<" + name;
    for (size_t at = text.find(open); at != std::string::npos; at = text.find(open, at + 1)) {
        const char next = at + open.size() < text.size() ? text[at + open.size()] : '>';
        if (next != ' ' && next != '>' && next != '/' && next != '\n' && next != '\t') continue;
        const size_t end = text.find('>', at);
        if (end == std::string::npos) break;
        out.push_back(attributes(text.substr(at, end - at)));
    }
    return out;
}

/** @brief The text of the section <name>...</name> of text, or "". */
std::string section(const std::string & text, const std::string & name) {
    const size_t a = text.find("<" + name + ">");
    if (a == std::string::npos) return "";
    const size_t b = text.find("</" + name + ">", a);
    return b == std::string::npos ? "" : text.substr(a, b - a);
}

/** @brief The ASCII content of the inline DataArray (or Array) named name of text, or nullopt. */
std::optional<std::string> inline_array(const std::string & text, const std::string & name) {
    const size_t at = text.find("Name=\"" + name + "\"");
    if (at == std::string::npos) return std::nullopt;
    const size_t start = text.find('>', at);
    const size_t end = text.find('<', start);
    if (start == std::string::npos || end == std::string::npos) return std::nullopt;
    return text.substr(start + 1, end - start - 1);
}

/** @brief The strings of a VTK ASCII String array: each string's bytes, ended by a 0. */
std::vector<std::string> vtk_strings(const std::string & content) {
    std::vector<std::string> out(1);
    std::istringstream in(content);
    int byte = 0;
    while (in >> byte) {
        if (byte == 0) {
            out.emplace_back();
        } else {
            out.back() += static_cast<char>(byte);
        }
    }
    out.pop_back();
    return out;
}

struct XmlArray {
    std::string name, type, section;
    uint32_t components = 1;
    uint64_t offset = 0;
};

/** @brief A VTK XML file: its header (up to the appended data) and arrays. */
struct VtkXml {
    std::string path, header, type;
    bool uint64_header = false, zlib = false;
    uint64_t appended = 0;  // byte where the appended data starts
    std::vector<XmlArray> arrays;
    std::map<std::string, std::string> piece;

    explicit VtkXml(const std::string & file) : path(file) {
        std::ifstream in(file, std::ios::binary);
        if (!in.good()) throw cannot_read("the VTK file", file);
        std::string chunk(1 << 16, '\0');
        while (in) {
            in.read(chunk.data(), static_cast<std::streamsize>(chunk.size()));
            header.append(chunk.data(), static_cast<size_t>(in.gcount()));
            const size_t a = header.find("<AppendedData");
            if (a != std::string::npos && header.find('_', a) != std::string::npos) break;
        }
        const size_t a = header.find("<AppendedData");
        if (a != std::string::npos) {
            appended = header.find('_', a) + 1;
            header.resize(appended);
        }
        const auto file_tags = tags(header, "VTKFile");
        if (file_tags.empty()) fail(file, "is not a VTK XML file");
        const auto & attrs = file_tags.front();
        type = attrs.count("type") ? attrs.at("type") : "";
        uint64_header = attrs.count("header_type") && attrs.at("header_type") == "UInt64";
        zlib = attrs.count("compressor") && attrs.at("compressor") == "vtkZLibDataCompressor";
        const auto pieces = tags(header, "Piece");
        if (!pieces.empty()) piece = pieces.front();
        for (const char * s : {"CellData", "PointData", "Points", "Cells", "Verts", "Lines", "Polys"}) {
            const std::string text = section(header, s);
            for (const auto & t : tags(text, "DataArray")) {
                if (!t.count("offset")) continue;
                XmlArray x;
                x.section = s;
                x.name = t.count("Name") ? t.at("Name") : "";
                x.type = t.count("type") ? t.at("type") : "";
                x.components = t.count("NumberOfComponents") ? uint32_t(std::stoul(t.at("NumberOfComponents"))) : 1;
                x.offset = std::stoull(t.at("offset"));
                arrays.push_back(x);
            }
        }
    }

    const XmlArray * find(const std::string & section_name, const std::string & name) const {
        for (const auto & x : arrays) {
            if (x.section == section_name && (name.empty() || x.name == name)) return &x;
        }
        return nullptr;
    }

    uint64_t piece_count(const char * key) const {
        return piece.count(key) ? std::stoull(piece.at(key)) : 0;
    }

    /** @brief The bytes of an appended array. */
    std::vector<char> bytes(const XmlArray & x) const {
        if (appended == 0) fail(path, "has no appended data (Mallard writes appended raw data only)");
        std::ifstream in(path, std::ios::binary);
        in.seekg(static_cast<std::streamoff>(appended + x.offset));
        auto word = [&]() {
            uint64_t v = 0;
            if (uint64_header) {
                in.read(reinterpret_cast<char *>(&v), 8);
            } else {
                uint32_t w = 0;
                in.read(reinterpret_cast<char *>(&w), 4);
                v = w;
            }
            return v;
        };
        std::vector<char> out;
        if (!zlib) {
            out.resize(word());
            in.read(out.data(), static_cast<std::streamsize>(out.size()));
        } else {
            if (!compression::have_zlib()) fail(path, "is compressed, which needs a build with zlib (Mallard_ENABLE_ZLIB=ON)");
            const uint64_t n_blocks = word(), block = word(), last = word();
            std::vector<uint64_t> sizes(n_blocks);
            for (auto & s : sizes) s = word();
            std::vector<char> packed;
            for (uint64_t b = 0; b < n_blocks; b++) {
                const uint64_t n = (b + 1 == n_blocks && last > 0) ? last : block;
                packed.resize(sizes[b]);
                in.read(packed.data(), static_cast<std::streamsize>(packed.size()));
                const size_t at = out.size();
                out.resize(at + n);
                compression::inflate(packed.data(), packed.size(), out.data() + at, n);
            }
        }
        if (!in.good()) fail(path, "is truncated");
        return out;
    }

    std::vector<double> values(const XmlArray & x) const {
        const std::vector<char> raw = bytes(x);
        std::vector<double> out;
        auto convert = [&](auto sample) {
            using T = decltype(sample);
            out.resize(raw.size() / sizeof(T));
            for (size_t i = 0; i < out.size(); i++) {
                T v;
                std::memcpy(&v, raw.data() + i * sizeof(T), sizeof(T));
                out[i] = static_cast<double>(v);
            }
        };
        if (x.type == "Float64") {
            convert(double{});
        } else if (x.type == "Float32") {
            convert(float{});
        } else if (x.type == "Int64") {
            convert(int64_t{});
        } else if (x.type == "Int32") {
            convert(int32_t{});
        } else if (x.type == "UInt8") {
            convert(uint8_t{});
        } else {
            fail(path, "array " + x.name + " has type " + x.type + ", which Mallard does not write");
        }
        return out;
    }
};

/** @brief The snapshot of a VTU or VTP piece's header: TIME, STEP and the snapshot block. */
std::optional<Snapshot> vtk_snapshot(const VtkXml & x, const std::string & path) {
    const std::string field_data = section(x.header, "FieldData");
    const auto t = inline_array(field_data, "TIME");
    if (!t) return std::nullopt;
    const auto step = inline_array(field_data, "STEP");
    return snapshot_of(std::stod(*t), step ? std::stoll(*step) : -1, path, 0,
                       provenance::parse_comment(x.header, snapshot_block::NAME));
}

/** @brief Fields, time and version from a VTU or VTP piece's header. */
void read_vtk_metadata(Contents & c, const VtkXml & x, const char * data_section, const file_format::Format & format) {
    const std::string field_data = section(x.header, "FieldData");
    if (const auto s = vtk_snapshot(x, c.path)) c.snapshots.push_back(*s);
    if (const auto v = inline_array(field_data, "MALLARD_FORMAT_VERSION")) {
        c.version = std::stoll(*v);
        file_format::check_version(format, c.path, c.version);
    } else {
        c.notes.push_back("no format version or units: written before Mallard 0.9, or not by Mallard");
    }
    std::vector<std::string> names, units, long_names;
    if (const auto a = inline_array(field_data, "MALLARD_FIELD_NAMES")) names = vtk_strings(*a);
    if (const auto a = inline_array(field_data, "MALLARD_FIELD_UNITS")) units = vtk_strings(*a);
    if (const auto a = inline_array(field_data, "MALLARD_FIELD_LONG_NAMES")) long_names = vtk_strings(*a);
    for (const auto & a : x.arrays) {
        if (a.section != data_section) continue;
        Field f;
        f.name = a.name;
        f.components = a.components;
        f.location = std::string(data_section) == "CellData" ? "cell" : "point";
        for (size_t i = 0; i < names.size(); i++) {
            if (names[i] != a.name) continue;
            if (i < units.size()) f.units = units[i];
            if (i < long_names.size()) f.long_name = long_names[i];
        }
        c.fields.push_back(f);
        if (c.real_size == 0 && (a.type == "Float64" || a.type == "Float32")) c.real_size = a.type == "Float64" ? 8 : 4;
    }
    c.compressed = x.zlib;
    c.run = provenance::parse_comment(x.header);
}

void read_vtu_mesh(MeshSummary & m, const VtkXml & x) {
    m.n_points += x.piece_count("NumberOfPoints");
    m.n_cells += x.piece_count("NumberOfCells");
    if (const XmlArray * types = x.find("Cells", "types")) {
        std::map<std::string, uint64_t> counts(m.cell_types.begin(), m.cell_types.end());
        for (const double t : x.values(*types)) counts[vtk_type_name(static_cast<int>(t))]++;
        count_types(m, counts);
    }
    if (const XmlArray * points = x.find("Points", "")) {
        MeshSummary piece;
        bound(piece, x.values(*points), 3);
        if (piece.has_bounds && !m.has_bounds) {
            m.has_bounds = true;
            m.lo = piece.lo;
            m.hi = piece.hi;
        } else if (piece.has_bounds) {
            for (size_t i = 0; i < 3; i++) {
                m.lo[i] = std::min(m.lo[i], piece.lo[i]);
                m.hi[i] = std::max(m.hi[i], piece.hi[i]);
            }
        }
    }
}

/** @brief The pieces of a PVTU file. */
std::vector<std::string> pvtu_pieces(const std::string & path, const std::string & text) {
    std::vector<std::string> out;
    for (const auto & t : tags(text, "Piece")) {
        if (t.count("Source")) out.push_back(sibling(path, t.at("Source")));
    }
    return out;
}

Contents open_vtu(const std::string & path) {
    const VtkXml x(path);
    Contents c;
    c.path = path;
    if (x.type == "UnstructuredGrid") {
        c.kind = Kind::snapshot;
        c.format = "VTU";
        read_vtk_metadata(c, x, "CellData", file_format::VTU_OUTPUT);
        read_vtu_mesh(c.mesh, x);
    } else if (x.type == "PolyData") {
        c.kind = Kind::probe;
        c.format = "VTP";
        const bool cell_data = x.find("CellData", "") != nullptr;
        read_vtk_metadata(c, x, cell_data ? "CellData" : "PointData", file_format::PROBE_OUTPUT);
        c.mesh.n_points = x.piece_count("NumberOfPoints");
        c.mesh.n_cells = x.piece_count("NumberOfVerts") + x.piece_count("NumberOfLines") + x.piece_count("NumberOfPolys");
        if (const XmlArray * points = x.find("Points", "")) bound(c.mesh, x.values(*points), 3);
    } else {
        fail(path, "is a VTK " + x.type + " file, which Mallard does not write");
    }
    if (c.snapshots.empty()) c.snapshots.push_back(snapshot_of(0.0, -1, path, 0, ""));
    return c;
}

Contents open_pvtu(const std::string & path) {
    const std::string text = read_bytes(path);
    const std::vector<std::string> pieces = pvtu_pieces(path, text);
    if (pieces.empty()) fail(path, "lists no pieces");
    Contents c = open_vtu(pieces.front());
    c.path = path;
    c.format = "PVTU";
    c.mesh = MeshSummary{};
    for (const auto & piece : pieces) read_vtu_mesh(c.mesh, VtkXml(piece));
    for (auto & s : c.snapshots) s.file = path;
    const std::string own = provenance::parse_comment(text);
    if (!own.empty()) c.run = own;
    c.notes.push_back(std::to_string(pieces.size()) + " pieces, one per rank of the run");
    return c;
}

std::vector<double> read_vtu_field(const std::string & file, const std::string & field) {
    if (ends_with(file, ".pvtu")) {
        std::vector<double> out;
        for (const auto & piece : pvtu_pieces(file, read_bytes(file))) {
            const std::vector<double> v = read_vtu_field(piece, field);
            out.insert(out.end(), v.begin(), v.end());
        }
        return out;
    }
    const VtkXml x(file);
    const XmlArray * a = x.find("CellData", field);
    if (!a) a = x.find("PointData", field);
    if (!a) fail(file, "has no field " + field);
    return x.values(*a);
}

// ---------------------------------------------------------------------------
// Series indexes: .vtkhdf.series, .pvd, .xmf

struct Entry {
    double t;
    int64_t step;  // -1 where the index records none
    std::string file;
    std::string mesh;  // XDMF: the grid's mesh file
};

/** @brief The text of each element <name ...>...</name> of text. */
std::vector<std::string> elements(const std::string & text, const std::string & name) {
    std::vector<std::string> out;
    const std::string open = "<" + name + " ", close = "</" + name + ">";
    for (size_t a = text.find(open); a != std::string::npos; a = text.find(open, a + 1)) {
        const size_t b = text.find(close, a);
        if (b == std::string::npos) break;
        out.push_back(text.substr(a, b - a));
    }
    return out;
}

/**
 * @brief The snapshots an index lists: time, step where it records one, and
 *        file. XDMF grids name their file in their data items, read from there.
 */
std::vector<Entry> series_entries(const std::string & path, std::string & own_provenance) {
    std::vector<Entry> out;
    const std::string text = read_bytes(path);
    if (ends_with(path, ".vtkhdf.series")) {
        const json::Value index = json::parse(text);
        for (const auto & e : index["files"].array) {
            const json::Value * step = e.find("step");
            out.push_back({e["time"].number, step ? static_cast<int64_t>(step->number) : -1,
                           sibling(path, e["name"].string), ""});
        }
        return out;
    }
    own_provenance = provenance::parse_comment(text);
    if (ends_with(path, ".pvd")) {
        for (const auto & t : tags(text, "DataSet")) {
            if (!t.count("file")) continue;
            out.push_back({std::stod(t.count("timestep") ? t.at("timestep") : "0"),
                           t.count("step") ? std::stoll(t.at("step")) : -1, sibling(path, t.at("file")), ""});
        }
        return out;
    }
    // XDMF: the uniform grids of the temporal collection (or the one grid)
    static const std::regex time(R"re(<Time Value="([^"]+)"/>)re");
    static const std::regex step(R"re(<Information Name="step" Value="([^"]+)"/>)re");
    static const std::regex fields(R"re(>([^<>:]+):/fields/)re");
    static const std::regex mesh(R"re(>([^<>:]+):/nodes/coordinates)re");
    for (const std::string & grid : elements(text, "Grid")) {
        if (grid.substr(0, grid.find('>')).find("GridType=\"Uniform\"") == std::string::npos) continue;
        std::smatch t, s, f, m;
        if (!std::regex_search(grid, f, fields)) continue;
        const bool has_time = std::regex_search(grid, t, time);
        const bool has_step = std::regex_search(grid, s, step);
        std::regex_search(grid, m, mesh);
        out.push_back({has_time ? std::stod(t[1].str()) : 0.0, has_step ? std::stoll(s[1].str()) : -1,
                       sibling(path, f[1].str()), m.empty() ? "" : sibling(path, m[1].str())});
    }
    return out;
}

std::vector<Snapshot> member_snapshots(const std::string & path);

Contents open_series(const std::string & path, const std::string & format) {
    std::string own;
    const std::vector<Entry> entries = series_entries(path, own);
    if (entries.empty()) fail(path, "lists no snapshot");
    const std::string & last = entries.back().file;
    if (!fs::exists(last)) fail(path, "lists " + last + ", which does not exist");
    Contents c = open(last);
    c.notes.erase(std::remove_if(c.notes.begin(), c.notes.end(),
                                 [](const std::string & n) {
                                     return n.starts_with("no run block") || n.starts_with("run block") ||
                                            n.starts_with("no snapshot block") || n.starts_with("the file records no step") ||
                                            n.starts_with("some snapshots record");
                                 }),
                  c.notes.end());
    const std::string member_format = c.format;
    c.path = path;
    c.format = format + " (" + member_format + " snapshots)";
    c.snapshots.clear();
    for (const Entry & e : entries) {
        // The index's time and step, and the member's own block
        Snapshot s = snapshot_of(e.t, e.step, e.file, 0, "");
        if (fs::exists(e.file)) {
            const std::vector<Snapshot> own_snapshots = member_snapshots(e.file);
            if (!own_snapshots.empty()) {
                const Snapshot & m = own_snapshots.front();
                s.has_block = m.has_block;
                s.block = m.block;
                s.run_id = m.run_id;
                if (s.step < 0) s.step = m.step;
            }
        }
        c.snapshots.push_back(s);
    }
    if (!own.empty()) c.run = own;
    return c;
}

// ---------------------------------------------------------------------------
// CSV tables

struct Table {
    std::vector<std::string> header, units;
    std::vector<std::vector<double>> rows;
    std::string provenance;                                // the last run block
    std::vector<std::pair<size_t, std::string>> runs;      // each run block, with the row it precedes
    std::string snapshot;                                  // a file of one snapshot (spectra): its block
    std::string legacy_snapshot;                           // spectra before 0.10: "# step = ..., t = ..."
};

std::vector<std::string> split(const std::string & line, char sep) {
    std::vector<std::string> out;
    std::string item;
    std::istringstream in(line);
    while (std::getline(in, item, sep)) out.push_back(item);
    if (!line.empty() && line.back() == sep) out.emplace_back();
    return out;
}

Table read_table(const std::string & path) {
    std::ifstream in(path);
    if (!in.good()) throw cannot_read("the CSV file", path);
    Table t;
    std::string line;
    const std::string snapshot_open = "# " + std::string(snapshot_block::NAME) + " ";
    while (std::getline(in, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (line.starts_with("# units: ")) {
            t.units = split(line.substr(9), ',');
        } else if (line.starts_with(snapshot_open)) {
            t.snapshot = line.substr(snapshot_open.size());
        } else if (line.starts_with("# step = ")) {
            t.legacy_snapshot = line;
        } else if (line.starts_with("#")) {
            const std::string p = provenance::parse_comment(line);
            if (!p.empty()) {
                t.provenance = p;
                t.runs.emplace_back(t.rows.size(), p);
            }
        } else if (t.header.empty()) {
            t.header = split(line, ',');
        } else if (!line.empty()) {
            std::vector<double> row;
            for (const auto & cell : split(line, ',')) {
                char * end = nullptr;
                const double v = std::strtod(cell.c_str(), &end);
                row.push_back(end != cell.c_str() ? v : std::numeric_limits<double>::quiet_NaN());
            }
            row.resize(t.header.size(), std::numeric_limits<double>::quiet_NaN());
            t.rows.push_back(std::move(row));
        }
    }
    return t;
}

Contents open_csv(const std::string & path) {
    const Table t = read_table(path);
    if (t.header.empty()) fail(path, "is empty");
    Contents c;
    c.path = path;
    c.kind = Kind::table;
    c.format = "CSV";
    c.run = t.provenance;
    if (t.units.empty()) c.notes.push_back("no units line: written before Mallard 0.9, or not by Mallard");
    size_t time_column = t.header.size(), step_column = t.header.size();
    for (size_t i = 0; i < t.header.size(); i++) {
        Field f;
        f.name = t.header[i];
        f.location = "column";
        if (i < t.units.size()) f.units = t.units[i];
        c.fields.push_back(f);
        if (t.header[i] == "t") time_column = i;
        if (t.header[i] == "step") step_column = i;
    }
    // A snapshot per run of rows of one time (probes write one row per sample), of the run whose block precedes it
    size_t segment = 0;
    for (size_t r = 0; r < t.rows.size(); r++) {
        while (segment + 1 < t.runs.size() && t.runs[segment + 1].first <= r) segment++;
        const double time = time_column < t.header.size() ? t.rows[r][time_column] : 0.0;
        if (r > 0 && time == c.snapshots.back().t && time_column < t.header.size()) continue;
        const double step = step_column < t.header.size() ? t.rows[r][step_column] : -1.0;
        Snapshot s = snapshot_of(time, std::isfinite(step) ? static_cast<int64_t>(step) : -1, path, r, "");
        if (!t.runs.empty()) s.run_id = run_id_of(t.runs[segment].second);
        c.snapshots.push_back(s);
    }
    c.mesh.n_points = c.snapshots.size() > 1 ? c.snapshots[1].index : t.rows.size();
    if (t.runs.size() > 1) {
        c.notes.push_back("rows of " + std::to_string(t.runs.size()) +
                          " runs: each run's block precedes the rows it appended");
    }
    if (!t.snapshot.empty() || !t.legacy_snapshot.empty()) {
        // One snapshot of the whole table (spectra)
        Snapshot s = snapshot_of(0.0, -1, path, 0, t.snapshot);
        if (!s.has_block) {
            static const std::regex legacy(R"re(# step = (\d+), t = ([^,]+),)re");
            std::smatch m;
            if (std::regex_search(t.legacy_snapshot, m, legacy)) {
                s.step = std::stoll(m[1].str());
                s.t = std::stod(m[2].str());
            }
            c.notes.push_back("no snapshot block (written before Mallard 0.10): step and time from its \"# step = ...\" "
                              "comment");
        }
        c.snapshots = {s};
    }
    return c;
}

std::vector<double> read_csv_field(const Contents & c, const std::string & field, size_t snapshot) {
    const Table t = read_table(c.path);
    const auto it = std::find(t.header.begin(), t.header.end(), field);
    if (it == t.header.end()) fail(c.path, "has no column " + field);
    const size_t column = static_cast<size_t>(it - t.header.begin());
    const size_t first = c.snapshots[snapshot].index;
    const size_t end = snapshot + 1 < c.snapshots.size() ? c.snapshots[snapshot + 1].index : t.rows.size();
    std::vector<double> out;
    for (size_t r = first; r < end && r < t.rows.size(); r++) out.push_back(t.rows[r][column]);
    return out;
}

// ---------------------------------------------------------------------------
// Mesh files

Contents mesh_block_contents(const std::string & path, const std::string & format, const MeshBlock & block) {
    Contents c;
    c.path = path;
    c.kind = Kind::mesh;
    c.format = format;
    c.notes.push_back("mesh files carry no run block");
    c.dim = N_DIM;
    c.global_order = true;
    MeshSummary & m = c.mesh;
    m.n_cells = block.n_cells();
    m.n_points = block.n_nodes();
    m.n_boundary_faces = block.n_faces();
    std::map<std::string, uint64_t> types;
    for (uint64_t i = 0; i < block.n_cells(); i++) {
        types[corner_type_name(N_DIM, block.cell_offsets[i + 1] - block.cell_offsets[i])]++;
    }
    count_types(m, types);
    std::vector<double> xyz;
    for (const auto & x : block.node_coords) xyz.insert(xyz.end(), x.begin(), x.end());
    bound(m, xyz, N_DIM);
    std::vector<uint64_t> faces(block.zone_names.size(), 0);
    for (const uint32_t z : block.face_zone) faces[z]++;
    for (size_t z = 0; z < faces.size(); z++) m.boundary_zones.emplace_back(block.zone_names[z], faces[z]);
    std::vector<uint64_t> cells(block.cell_zone_names.size(), 0);
    for (const uint32_t z : block.cell_zone) {
        if (z < cells.size()) cells[z]++;
    }
    for (size_t z = 0; z < cells.size(); z++) m.cell_zones.emplace_back(block.cell_zone_names[z], cells[z]);
    c.snapshots.clear();
    return c;
}

std::string mesh_format(const std::string & path) {
    if (fs::is_directory(path)) return "OpenFOAM";
    if (is_hdf5_mesh(path)) return "Mallard HDF5 mesh";
    const std::string ext = extension(path);
    if (ext == ".cgns") return "CGNS";
    if (ext == ".cas") return "Fluent";
    std::ifstream in(path);
    char first = 0;
    in >> first;
    return first == '(' ? "Fluent" : "Gmsh";
}

// ---------------------------------------------------------------------------
// HDF5: VTKHDF, HDF5 output and its mesh, probe files, HDF5 meshes

#ifdef Mallard_HAS_HDF5

using h5::Handle;

bool exists(hid_t parent, const std::string & name) { return H5Lexists(parent, name.c_str(), H5P_DEFAULT) > 0; }

std::vector<hsize_t> dims(hid_t parent, const std::string & name) {
    Handle dset(H5Dopen2(parent, name.c_str(), H5P_DEFAULT), H5Dclose, "opening " + name);
    Handle space(H5Dget_space(dset), H5Sclose, "H5Dget_space");
    const int rank = H5Sget_simple_extent_ndims(space);
    std::vector<hsize_t> out(static_cast<size_t>(std::max(rank, 0)));
    H5Sget_simple_extent_dims(space, out.data(), nullptr);
    return out;
}

/** @brief Rows [first, first + n) of a dataset, every column, as T. */
template <typename T>
std::vector<T> rows(hid_t parent, const std::string & name, uint64_t first, uint64_t n) {
    Handle dset(H5Dopen2(parent, name.c_str(), H5P_DEFAULT), H5Dclose, "opening " + name);
    Handle space(H5Dget_space(dset), H5Sclose, "H5Dget_space");
    const int rank = H5Sget_simple_extent_ndims(space);
    hsize_t d[2] = {0, 1};
    H5Sget_simple_extent_dims(space, d, nullptr);
    if (rank < 1 || rank > 2 || first + n > d[0]) throw std::runtime_error("reading " + name + ": rows out of range");
    const hsize_t cols = rank == 2 ? d[1] : 1;
    const hsize_t start[2] = {first, 0}, count[2] = {n, cols};
    h5::check(H5Sselect_hyperslab(space, H5S_SELECT_SET, start, nullptr, count, nullptr), "H5Sselect_hyperslab");
    Handle memory(H5Screate_simple(rank, count, nullptr), H5Sclose, "H5Screate_simple");
    std::vector<T> out(std::max<hsize_t>(n * cols, 1));
    h5::check(H5Dread(dset, h5::memory_type<T>(), memory, space, H5P_DEFAULT, out.data()), "reading " + name);
    out.resize(n * cols);
    return out;
}

template <typename T>
std::vector<T> all(hid_t parent, const std::string & name) {
    const std::vector<hsize_t> d = dims(parent, name);
    return rows<T>(parent, name, 0, d.empty() ? 0 : d[0]);
}

std::string string_attribute(hid_t object, const char * name) {
    if (H5Aexists(object, name) <= 0) return "";
    return h5::read_string_attribute(object, name);
}

/** @brief The datasets of a group, in name order. */
std::vector<std::string> datasets(hid_t group) {
    std::vector<std::string> out;
    H5G_info_t info;
    h5::check(H5Gget_info(group, &info), "H5Gget_info");
    for (hsize_t i = 0; i < info.nlinks; i++) {
        const ssize_t n = H5Lget_name_by_idx(group, ".", H5_INDEX_NAME, H5_ITER_INC, i, nullptr, 0, H5P_DEFAULT);
        std::string name(static_cast<size_t>(n), '\0');
        H5Lget_name_by_idx(group, ".", H5_INDEX_NAME, H5_ITER_INC, i, name.data(), name.size() + 1, H5P_DEFAULT);
        const hid_t object = H5Oopen(group, name.c_str(), H5P_DEFAULT);
        if (object < 0) continue;
        if (H5Iget_type(object) == H5I_DATASET) out.push_back(name);
        H5Oclose(object);
    }
    return out;
}

/** @brief The fields of a group's datasets: components from their shape (columns per row, or a fixed count). */
void group_fields(Contents & c, hid_t group, const std::string & location, bool rows_are_steps = false) {
    for (const auto & name : datasets(group)) {
        Field f;
        f.name = name;
        f.location = location;
        const std::vector<hsize_t> d = dims(group, name);
        f.components = !rows_are_steps && d.size() == 2 ? static_cast<uint32_t>(d[1]) : 1;
        Handle dset(H5Dopen2(group, name.c_str(), H5P_DEFAULT), H5Dclose, "opening " + name);
        f.units = string_attribute(dset, "units");
        f.long_name = string_attribute(dset, "long_name");
        if (c.real_size == 0) {
            Handle type(H5Dget_type(dset), H5Tclose, "H5Dget_type");
            if (H5Tget_class(type) == H5T_FLOAT) c.real_size = static_cast<uint32_t>(H5Tget_size(type));
        }
        if (!c.compressed) {
            Handle plist(H5Dget_create_plist(dset), H5Pclose, "H5Dget_create_plist");
            c.compressed = H5Pget_nfilters(plist) > 0;
        }
        c.fields.push_back(f);
    }
}

Handle open_file(const std::string & path) {
    if (!fs::exists(path)) throw cannot_read("the file", path);
    return Handle(H5Fopen(path.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT), H5Fclose, "opening " + path);
}

std::string root_provenance(hid_t file) {
    Handle root(H5Gopen2(file, "/", H5P_DEFAULT), H5Gclose, "opening /");
    for (const char * name : {provenance::ATTRIBUTE, provenance::LEGACY_ATTRIBUTE}) {
        if (H5Aexists(root, name) <= 0) continue;
        const auto values = h5::read_strings_attribute(root, name);
        return values.empty() ? "" : values[0];
    }
    return "";
}

/** @brief The snapshot blocks of an HDF5 file (snapshot_block::DATASET), one per snapshot; none in older files. */
std::vector<std::string> root_blocks(hid_t file) {
    if (H5Lexists(file, snapshot_block::DATASET, H5P_DEFAULT) <= 0) return {};
    return h5::read_string_rows(file, snapshot_block::DATASET);
}

int64_t root_int(hid_t file, const char * name, int64_t fallback) {
    Handle root(H5Gopen2(file, "/", H5P_DEFAULT), H5Gclose, "opening /");
    return H5Aexists(root, name) > 0 ? h5::read_int_attribute(root, name) : fallback;
}

template <typename T>
T root_scalar(hid_t file, const char * name, T fallback) {
    Handle root(H5Gopen2(file, "/", H5P_DEFAULT), H5Gclose, "opening /");
    if (H5Aexists(root, name) <= 0) return fallback;
    Handle attr(H5Aopen(root, name, H5P_DEFAULT), H5Aclose, name);
    T value = fallback;
    h5::check(H5Aread(attr, h5::memory_type<T>(), &value), name);
    return value;
}

std::vector<std::string> root_strings(hid_t file, const char * name) {
    Handle root(H5Gopen2(file, "/", H5P_DEFAULT), H5Gclose, "opening /");
    return H5Aexists(root, name) > 0 ? h5::read_strings_attribute(root, name) : std::vector<std::string>{};
}

uint64_t first_count(hid_t parent, const std::string & name) {
    return exists(parent, name) ? static_cast<uint64_t>(all<int64_t>(parent, name).at(0)) : 0;
}

Contents open_vtkhdf(const std::string & path) {
    const std::lock_guard<std::mutex> lock(h5::mutex());
    Handle file = open_file(path);
    if (!exists(file, "VTKHDF")) fail(path, "is an HDF5 file without a VTKHDF group");
    Handle top(H5Gopen2(file, "VTKHDF", H5P_DEFAULT), H5Gclose, "opening VTKHDF");
    const std::string type = string_attribute(top, "Type");
    Contents c;
    c.path = path;
    c.run = root_provenance(file);
    c.version = root_int(file, "version", -1);
    MeshSummary & m = c.mesh;
    m.n_points = first_count(top, "NumberOfPoints");
    if (exists(top, "Points")) bound(m, all<double>(top, "Points"), 3);
    std::string data = "CellData";
    if (type == "UnstructuredGrid") {
        c.kind = Kind::snapshot;
        c.format = "VTKHDF";
        if (c.version >= 0) file_format::check_version(file_format::VTKHDF_OUTPUT, path, c.version);
        m.n_cells = first_count(top, "NumberOfCells");
        std::map<std::string, uint64_t> counts;
        for (const uint8_t t : all<uint8_t>(top, "Types")) counts[vtk_type_name(t)]++;
        count_types(m, counts);
    } else if (type == "PolyData") {
        c.kind = Kind::probe;
        c.format = "VTKHDF";
        if (c.version >= 0) file_format::check_version(file_format::PROBE_OUTPUT, path, c.version);
        for (const char * kind : {"Vertices", "Lines", "Polygons"}) {
            const std::string group = std::string(kind) + "/NumberOfCells";
            m.n_cells += exists(top, kind) ? first_count(top, group) : 0;
        }
        Handle cell_data(H5Gopen2(top, "CellData", H5P_DEFAULT), H5Gclose, "opening CellData");
        if (datasets(cell_data).empty()) data = "PointData";
    } else {
        fail(path, "is a VTKHDF " + type + ", which Mallard writes only for mallard check --view");
    }
    if (c.version < 0) c.notes.push_back("no format version: not written by Mallard");
    Handle group(H5Gopen2(top, data.c_str(), H5P_DEFAULT), H5Gclose, "opening " + data);
    const bool series = exists(top, "Steps");
    // A series stores every snapshot's rows one after the other: a field's columns are its shape's second dimension
    group_fields(c, group, data == "CellData" ? "cell" : "point");
    Handle field_data(H5Gopen2(top, "FieldData", H5P_DEFAULT), H5Gclose, "opening FieldData");
    const std::vector<int64_t> recorded_steps =
        exists(field_data, "STEP") ? all<int64_t>(field_data, "STEP") : std::vector<int64_t>{};
    const std::vector<std::string> blocks = root_blocks(file);
    std::vector<double> times;
    if (series) {
        Handle steps(H5Gopen2(top, "Steps", H5P_DEFAULT), H5Gclose, "opening Steps");
        times = all<double>(steps, "Values");
    } else {
        times = {exists(field_data, "TIME") ? all<double>(field_data, "TIME").at(0) : 0.0};
    }
    for (size_t k = 0; k < times.size(); k++) {
        c.snapshots.push_back(snapshot_of(times[k], k < recorded_steps.size() ? recorded_steps[k] : -1, path, k,
                                          k < blocks.size() ? blocks[k] : ""));
    }
    c.global_order = c.kind == Kind::snapshot && !c.field("REGION_FRACTION");
    if (c.field("REGION_FRACTION")) c.notes.push_back("the cells of a region, in their own order");
    return c;
}

std::vector<double> read_vtkhdf_field(const std::string & path, const std::string & field, uint64_t k) {
    const std::lock_guard<std::mutex> lock(h5::mutex());
    Handle file = open_file(path);
    Handle top(H5Gopen2(file, "VTKHDF", H5P_DEFAULT), H5Gclose, "opening VTKHDF");
    std::string data;
    for (const char * g : {"CellData", "PointData"}) {
        if (exists(top, std::string(g) + "/" + field)) data = g;
    }
    if (data.empty()) fail(path, "has no field " + field);
    Handle group(H5Gopen2(top, data.c_str(), H5P_DEFAULT), H5Gclose, "opening " + data);
    const std::string offsets = "Steps/" + data + "Offsets/" + field;
    if (!exists(top, "Steps") || !exists(top, offsets)) return all<double>(group, field);
    const uint64_t at = static_cast<uint64_t>(rows<int64_t>(top, offsets, k, 1).at(0));
    uint64_t n = 0;
    if (data == "PointData") {
        n = first_count(top, "NumberOfPoints");
    } else if (exists(top, "Types")) {
        n = first_count(top, "NumberOfCells");
    } else {
        for (const char * kind : {"Vertices", "Lines", "Polygons"}) {
            if (exists(top, kind)) n += first_count(top, std::string(kind) + "/NumberOfCells");
        }
    }
    const uint64_t total = dims(group, field).at(0);
    return rows<double>(group, field, at, std::min(n, total - std::min(at, total)));
}

/** @brief The mesh summary of an HDF5 output mesh file (XDMF Mixed topology). */
void read_output_mesh(MeshSummary & m, hid_t file) {
    Handle nodes(H5Gopen2(file, "nodes", H5P_DEFAULT), H5Gclose, "opening nodes");
    const std::vector<hsize_t> d = dims(nodes, "coordinates");
    m.n_points = d.at(0);
    bound(m, all<double>(nodes, "coordinates"), d.size() == 2 ? d[1] : 1);
    Handle cells(H5Gopen2(file, "cells", H5P_DEFAULT), H5Gclose, "opening cells");
    const std::vector<int64_t> offsets = all<int64_t>(cells, "offsets");
    const std::vector<int64_t> topology = all<int64_t>(cells, "topology");
    m.n_cells = offsets.empty() ? 0 : offsets.size() - 1;
    static const std::map<int64_t, int> XDMF_TO_VTK = {{4, 5}, {5, 9}, {6, 10}, {7, 14}, {8, 13}, {9, 12}};
    std::map<std::string, uint64_t> counts;
    for (size_t i = 0; i + 1 < offsets.size(); i++) {
        const int64_t code = topology.at(static_cast<size_t>(offsets[i]));
        const auto it = XDMF_TO_VTK.find(code);
        counts[it == XDMF_TO_VTK.end() ? "XDMF type " + std::to_string(code) : vtk_type_name(it->second)]++;
    }
    count_types(m, counts);
}

Contents open_h5(const std::string & path) {
    std::unique_lock<std::mutex> lock(h5::mutex());
    Handle file = open_file(path);
    if (exists(file, "VTKHDF")) {
        lock.unlock();
        return open_vtkhdf(path);
    }
    const std::vector<std::string> format = root_strings(file, "format");
    const std::string kind = format.empty() ? "" : format[0];
    Contents c;
    c.path = path;
    c.run = root_provenance(file);
    c.version = root_int(file, "version", -1);
    c.dim = static_cast<int>(root_int(file, "dimension", 0));
    if (kind == "mallard-mesh") {
        lock.unlock();
        file_format::check_version(file_format::MESH, path, c.version);
        if (c.dim != N_DIM) {
            fail(path, "is a " + std::to_string(c.dim) + "D mesh; read it with mallard" + std::to_string(c.dim) +
                           "d (mallard info runs it by itself)");
        }
        Contents mesh = mesh_block_contents(path, "Mallard HDF5 mesh", read_mesh_h5(path, true));
        mesh.version = c.version;
        return mesh;
    }
    if (kind == "mallard-solution-mesh") {
        file_format::check_version(file_format::HDF5_OUTPUT, path, c.version);
        c.kind = Kind::mesh;
        c.format = "XDMF output mesh";
        c.global_order = true;
        read_output_mesh(c.mesh, file);
        return c;
    }
    if (kind == "mallard-solution") {
        file_format::check_version(file_format::HDF5_OUTPUT, path, c.version);
        c.kind = Kind::snapshot;
        c.format = "XDMF output";
        c.global_order = true;
        const auto mesh = root_strings(file, "mesh");
        if (!mesh.empty()) c.mesh_file = sibling(path, mesh[0]);
        const std::vector<std::string> blocks = root_blocks(file);
        c.snapshots.push_back(snapshot_of(root_scalar<double>(file, "time", 0.0),
                                          static_cast<int64_t>(root_scalar<uint64_t>(file, "step", 0)), path, 0,
                                          blocks.empty() ? "" : blocks[0]));
        Handle fields(H5Gopen2(file, "fields", H5P_DEFAULT), H5Gclose, "opening fields");
        group_fields(c, fields, "cell");
        if (!c.mesh_file.empty() && fs::exists(c.mesh_file)) {
            Handle mesh_h5 = open_file(c.mesh_file);
            read_output_mesh(c.mesh, mesh_h5);
        } else {
            c.notes.push_back("its mesh file " + c.mesh_file + " is missing");
        }
        return c;
    }
    if (exists(file, "fields") && exists(file, "time")) {
        if (c.version >= 0) file_format::check_version(file_format::PROBE_OUTPUT, path, c.version);
        c.kind = Kind::probe;
        c.format = "probe HDF5";
        const std::vector<hsize_t> d = dims(file, "coordinates");
        c.mesh.n_points = d.at(0);
        c.dim = d.size() == 2 ? static_cast<int>(d[1]) : 1;
        bound(c.mesh, all<double>(file, "coordinates"), static_cast<size_t>(c.dim));
        const std::vector<double> times = all<double>(file, "time");
        const std::vector<uint64_t> steps = exists(file, "step") ? all<uint64_t>(file, "step") : std::vector<uint64_t>{};
        // Each run that wrote the file, from its first row
        std::vector<std::pair<uint64_t, std::string>> runs;
        for (const std::string & r : root_strings(file, "mallard_runs")) {
            try {
                const json::Value v = json::parse(r);
                runs.emplace_back(static_cast<uint64_t>(v["first_row"].number), v["run_id"].string);
            } catch (const std::exception &) {
                c.notes.push_back("an entry of mallard_runs is not valid JSON");
            }
        }
        if (runs.size() > 1) c.notes.push_back("samples of " + std::to_string(runs.size()) + " runs (mallard_runs)");
        if (runs.empty()) {
            c.notes.push_back("no mallard_runs (written before Mallard 0.10): the run of each sample is that of the run "
                              "block, the last run that wrote the file");
        }
        for (size_t k = 0; k < times.size(); k++) {
            Snapshot s = snapshot_of(times[k], k < steps.size() ? static_cast<int64_t>(steps[k]) : -1, path, k, "");
            for (const auto & [first, id] : runs) {
                if (first <= k) s.run_id = id;
            }
            c.snapshots.push_back(s);
        }
        Handle fields(H5Gopen2(file, "fields", H5P_DEFAULT), H5Gclose, "opening fields");
        group_fields(c, fields, "point", true);
        return c;
    }
    fail(path, "is an HDF5 file Mallard did not write (no VTKHDF group, no Mallard format attribute)");
}

std::vector<double> read_h5_field(const Contents & c, const std::string & field, const Snapshot & s) {
    if (c.format == "VTKHDF") return read_vtkhdf_field(s.file, field, s.index);
    const std::lock_guard<std::mutex> lock(h5::mutex());
    Handle file = open_file(s.file);
    Handle fields(H5Gopen2(file, "fields", H5P_DEFAULT), H5Gclose, "opening fields");
    if (!exists(fields, field)) fail(s.file, "has no field " + field);
    if (c.kind == Kind::probe) return rows<double>(fields, field, s.index, 1);
    return all<double>(fields, field);
}

#else

[[noreturn]] void no_hdf5(const std::string & path) {
    fail(path, "is an HDF5 file, which needs a build with HDF5 (-DMallard_ENABLE_HDF5=ON)");
}

Contents open_h5(const std::string & path) { no_hdf5(path); }


Contents open_vtkhdf(const std::string & path) { no_hdf5(path); }

std::vector<double> read_h5_field(const Contents &, const std::string &, const Snapshot & s) { no_hdf5(s.file); }

#endif

#ifdef Mallard_HAS_HDF5
/** @brief The snapshots of a series member that is an HDF5 file: time, step and block, without the mesh. */
std::vector<Snapshot> h5_member_snapshots(const std::string & path) {
    const std::lock_guard<std::mutex> lock(h5::mutex());
    Handle file = open_file(path);
    const std::vector<std::string> blocks = root_blocks(file);
    std::vector<Snapshot> out;
    if (exists(file, "VTKHDF")) {
        Handle top(H5Gopen2(file, "VTKHDF", H5P_DEFAULT), H5Gclose, "opening VTKHDF");
        Handle field_data(H5Gopen2(top, "FieldData", H5P_DEFAULT), H5Gclose, "opening FieldData");
        const double t = exists(field_data, "TIME") ? all<double>(field_data, "TIME").at(0) : 0.0;
        const int64_t step = exists(field_data, "STEP") ? all<int64_t>(field_data, "STEP").at(0) : -1;
        out.push_back(snapshot_of(t, step, path, 0, blocks.empty() ? "" : blocks[0]));
    } else {
        const bool has_step = [&] {
            Handle root(H5Gopen2(file, "/", H5P_DEFAULT), H5Gclose, "opening /");
            return H5Aexists(root, "step") > 0;
        }();
        out.push_back(snapshot_of(root_scalar<double>(file, "time", 0.0),
                                  has_step ? static_cast<int64_t>(root_scalar<uint64_t>(file, "step", 0)) : -1, path,
                                  0, blocks.empty() ? "" : blocks[0]));
    }
    return out;
}
#endif

/** @brief The snapshots of a series member, without reading its mesh. */
std::vector<Snapshot> member_snapshots(const std::string & path) {
    const std::string ext = extension(path);
    if (ext == ".vtu" || ext == ".vtp") {
        const auto s = vtk_snapshot(VtkXml(path), path);
        return s ? std::vector<Snapshot>{*s} : std::vector<Snapshot>{};
    }
    if (ext == ".pvtu") {
        const std::vector<std::string> pieces = pvtu_pieces(path, read_bytes(path));
        if (pieces.empty()) return {};
        auto s = vtk_snapshot(VtkXml(pieces.front()), path);
        return s ? std::vector<Snapshot>{*s} : std::vector<Snapshot>{};
    }
#ifdef Mallard_HAS_HDF5
    return h5_member_snapshots(path);
#else
    return {};
#endif
}

bool has_restart_magic(const std::string & path) {
    std::ifstream in(path, std::ios::binary);
    char magic[16] = {};
    in.read(magic, sizeof(magic));
    return in.good() && std::string(magic, 15) == "MALLARD-RESTART";
}

bool has_hdf5_signature(const std::string & path) {
    std::ifstream in(path, std::ios::binary);
    char signature[8] = {};
    in.read(signature, sizeof(signature));
    return in.good() && std::memcmp(signature, "\x89HDF\r\n\x1a\n", 8) == 0;
}

bool is_openfoam(const fs::path & path) {
    for (const fs::path & dir : {path, path / "polyMesh", path / "constant" / "polyMesh"}) {
        if (fs::exists(dir / "boundary")) return true;
    }
    return false;
}

} // namespace

const char * kind_name(Kind kind) {
    switch (kind) {
        case Kind::mesh: return "mesh";
        case Kind::snapshot: return "snapshot";
        case Kind::restart: return "restart";
        case Kind::probe: return "probe";
        case Kind::table: return "table";
    }
    return "";
}

const Field * Contents::field(const std::string & name) const {
    for (const auto & f : fields) {
        if (f.name == name) return &f;
    }
    return nullptr;
}

std::vector<std::string> Contents::species() const {
    std::vector<std::string> out;
    for (const auto & f : fields) {
        for (const char * prefix : {"Y_", "X_", "RHOY_"}) {
            if (!f.name.starts_with(prefix) || f.name.size() <= std::strlen(prefix)) continue;
            const std::string s = f.name.substr(std::strlen(prefix));
            if (std::find(out.begin(), out.end(), s) == out.end()) out.push_back(s);
        }
    }
    return out;
}

MeshBlock read_mesh(const std::string & path) {
    if (is_hdf5_mesh(path)) return read_mesh_h5(path, true);
    return read_file_block(path);
}

Contents mesh_contents(const std::string & path, const MeshBlock & block) {
    Contents c = mesh_block_contents(path, mesh_format(path), block);
    if (is_hdf5_mesh(path)) c.version = file_format::MESH.version;
    return c;
}

bool is_mesh_file(const std::string & path) {
    std::error_code error;
    if (fs::is_directory(path, error)) return is_openfoam(path);
    const std::string ext = extension(path);
    if (ext == ".msh" || ext == ".cas" || ext == ".cgns") return true;
    if (!is_hdf5_mesh(path) || !fs::exists(path, error)) return false;
#ifdef Mallard_HAS_HDF5
    try {
        const std::lock_guard<std::mutex> lock(h5::mutex());
        Handle file = open_file(path);
        const std::vector<std::string> format = root_strings(file, "format");
        return !format.empty() && format[0] == "mallard-mesh";
    } catch (const std::exception &) {
        return false;
    }
#else
    return true;
#endif
}

Contents open(const std::string & path) {
    std::error_code error;
    if (fs::is_directory(path, error)) {
        if (!is_openfoam(path)) fail(path, "is a directory, not an OpenFOAM case");
        return mesh_contents(path, read_mesh(path));
    }
    if (!fs::exists(path, error)) throw cannot_read("the file", path);
    if (ends_with(path, PARTIAL)) {
        fail(path, "its write did not finish: the run was stopped while writing it, or is still writing it. The file "
                   "gets its name " + path.substr(0, path.size() - std::string(PARTIAL).size()) + " once complete.");
    }
    Contents c;
    const std::string ext = extension(path);
    if (ends_with(path, ".vtkhdf.series")) {
        c = open_series(path, "VTKHDF series index");
    } else if (ext == ".pvd") {
        c = open_series(path, "PVD series index");
    } else if (ext == ".xmf") {
        const bool series = read_bytes(path).find("CollectionType=\"Temporal\"") != std::string::npos;
        c = open_series(path, series ? "XDMF series index" : "XDMF index");
    } else if (ext == ".pvtu") {
        c = open_pvtu(path);
    } else if (ext == ".vtu" || ext == ".vtp") {
        c = open_vtu(path);
    } else if (ext == ".csv") {
        c = open_csv(path);
    } else if (has_restart_magic(path)) {
        c = open_restart(path);
    } else if (ext == ".vtkhdf") {
        c = open_vtkhdf(path);
    } else if (has_hdf5_signature(path)) {
        c = open_h5(path);
    } else if (ext == ".msh" || ext == ".cas" || ext == ".cgns") {
        return mesh_contents(path, read_mesh(path));
    } else {
        fail(path, "is not a file Mallard writes or reads: snapshots (.vtkhdf, .vtu, .pvtu, .h5 and the series "
                   "indexes .vtkhdf.series, .pvd, .xmf), restart files, probes (.csv, .h5, .vtkhdf, .vtp), CSV "
                   "tables, or meshes (.msh, .cas, .cgns, .h5, an OpenFOAM case)");
    }
    if (c.format != "Mallard HDF5 mesh") note_provenance(c);
    note_snapshots(c);
    return c;
}

std::vector<double> read(const Contents & file, const std::string & field, size_t snapshot) {
    if (!file.field(field)) fail(file.path, "has no field " + field);
    if (snapshot >= file.snapshots.size()) {
        fail(file.path,
             "has " + logging::count(file.snapshots.size(), "snapshot") + ", not " + std::to_string(snapshot + 1));
    }
    const Snapshot & s = file.snapshots[snapshot];
    switch (file.kind) {
        case Kind::restart: {
            const auto & names = file.restart.names;
            return read_restart_field(file, static_cast<size_t>(std::find(names.begin(), names.end(), field) - names.begin()));
        }
        case Kind::table: return read_csv_field(file, field, snapshot);
        case Kind::mesh: fail(file.path, "holds no field values");
        default: break;
    }
    if (s.file != file.path) return read(open(s.file), field, 0);
    const std::string ext = extension(s.file);
    if (ext == ".vtu" || ext == ".vtp" || ext == ".pvtu") return read_vtu_field(s.file, field);
    return read_h5_field(file, field, s);
}

void read_ranges(Contents & file, size_t snapshot) {
    for (auto & f : file.fields) {
        const std::vector<double> v = read(file, f.name, snapshot);
        double lo = std::numeric_limits<double>::infinity(), hi = -lo;
        const size_t n = f.components;
        for (size_t i = 0; i + n <= v.size(); i += n) {
            double x = v[i];
            if (n > 1) {
                x = 0.0;
                for (size_t k = 0; k < n; k++) x += v[i + k] * v[i + k];
                x = std::sqrt(x);
            }
            if (std::isnan(x)) continue;
            lo = std::min(lo, x);
            hi = std::max(hi, x);
        }
        if (lo <= hi) {
            f.min = lo;
            f.max = hi;
        }
    }
}

std::vector<std::string> run_files(const std::string & dir) {
    std::vector<std::string> files;
    std::error_code error;
    for (auto it = fs::recursive_directory_iterator(dir, error); !error && it != fs::recursive_directory_iterator();
         it.increment(error)) {
        if (it.depth() > 1) it.disable_recursion_pending();
        if (it->is_regular_file(error)) files.push_back(it->path().string());
    }
    std::sort(files.begin(), files.end());
    // Members of an index (and the pieces of a PVTU), and the mesh files XDMF grids name, are reported through it
    const auto normal = [](const std::string & p) { return fs::path(p).lexically_normal().string(); };
    std::set<std::string> members, collection_members;
    std::vector<std::pair<std::string, std::vector<std::string>>> single_grids;  // XDMF files of one grid: their data
    for (const auto & f : files) {
        try {
            const std::string ext = extension(f);
            std::string own;
            if (ends_with(f, ".vtkhdf.series") || ext == ".pvd" || ext == ".xmf") {
                const auto entries = series_entries(f, own);
                const bool single = ext == ".xmf" && read_bytes(f).find("GridType=\"Collection\"") == std::string::npos;
                if (single) single_grids.push_back({normal(f), {}});
                for (const auto & e : entries) {
                    members.insert(normal(e.file));
                    if (!e.mesh.empty()) members.insert(normal(e.mesh));
                    if (single) single_grids.back().second.push_back(normal(e.file));
                    if (ext == ".xmf" && !single) collection_members.insert(normal(e.file));
                    if (ext == ".pvd" && ends_with(e.file, ".pvtu")) {
                        for (const auto & p : pvtu_pieces(e.file, read_bytes(e.file))) members.insert(fs::path(p).lexically_normal().string());
                    }
                }
            } else if (ext == ".pvtu") {
                for (const auto & p : pvtu_pieces(f, read_bytes(f))) members.insert(fs::path(p).lexically_normal().string());
            }
        } catch (const std::exception &) {
            // An unreadable index lists nothing; info reports its error
        }
    }
    // An XDMF file of one grid that a collection also lists is reported through the collection
    for (const auto & [file, data] : single_grids) {
        if (!data.empty() &&
            std::all_of(data.begin(), data.end(), [&](const std::string & d) { return collection_members.count(d) > 0; })) {
            members.insert(file);
        }
    }
    std::vector<std::string> out;
    static const std::set<std::string> OUTPUT = {".vtkhdf", ".series", ".pvd", ".xmf", ".vtu", ".pvtu", ".vtp",
                                                 ".h5", ".restart", ".csv"};
    for (const auto & f : files) {
        // A file whose write did not finish is listed, and reported as such
        const std::string name = ends_with(f, PARTIAL) ? f.substr(0, f.size() - std::string(PARTIAL).size()) : f;
        if (!OUTPUT.count(extension(name)) || members.count(fs::path(f).lexically_normal().string())) continue;
        if (ends_with(name, ".series") && !ends_with(name, ".vtkhdf.series")) continue;
        out.push_back(f);
    }
    return out;
}

} // namespace output

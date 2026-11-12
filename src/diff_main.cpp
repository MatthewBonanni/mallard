/**
 * @file diff_main.cpp
 * @brief `mallard diff`: how two solutions differ, field by field, and
 *        whether restart files hold bitwise the same state.
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 */

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <exception>
#include <filesystem>
#include <fstream>
#include <initializer_list>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <Kokkos_Core.hpp>

#include "comm.h"
#include "commands.h"
#include "input.h"
#include "json.h"
#include "log.h"
#include "mesh.h"
#include "output_reader.h"
#include "provenance.h"
#include "remap.h"
#include "report.h"
#include "diagnostic_code.h"

namespace fs = std::filesystem;

namespace {

using logging::count;
using logging::format;

constexpr int DIFFERENT = 1, TROUBLE = 3;
constexpr uint8_t REMAP_DEGREE = 2;
constexpr double INF = std::numeric_limits<double>::infinity();

/** @brief Files that cannot be compared: exit status 3. */
struct Trouble : std::runtime_error {
    using std::runtime_error::runtime_error;
};

/** @brief Identical, different within the tolerance (or not compared), or beyond it. */
enum class Status { same, within, beyond };

Status worst(Status a, Status b) { return static_cast<int>(a) > static_cast<int>(b) ? a : b; }

struct Tolerance {
    double all = 0.0;
    std::map<std::string, double> field;
    double of(const std::string & name) const {
        const auto it = field.find(name);
        return it == field.end() ? all : it->second;
    }
};

bool parse_number(const std::string & text, double & out) {
    try {
        size_t used = 0;
        out = std::stod(text, &used);
        return used == text.size() && std::isfinite(out) && out >= 0.0;
    } catch (const std::exception &) {
        return false;
    }
}

std::vector<std::string> split(const std::string & text) {
    std::vector<std::string> out;
    std::istringstream in(text);
    for (std::string part; std::getline(in, part, ',');) {
        if (!part.empty()) out.push_back(part);
    }
    return out;
}

/** @brief "1e-6" or "1e-6,P=1e-3": a default and per-field tolerances; false if it is neither. */
bool parse_tolerance(const std::string & text, Tolerance & t) {
    for (const std::string & part : split(text)) {
        const size_t eq = part.find('=');
        double value = 0.0;
        if (!parse_number(eq == std::string::npos ? part : part.substr(eq + 1), value) || eq == 0) return false;
        if (eq == std::string::npos) {
            t.all = value;
        } else {
            t.field[part.substr(0, eq)] = value;
        }
    }
    return true;
}

/** @brief |y - x|, with a NaN in one value only as an infinite difference and in both as none. */
double difference(double x, double y) {
    if (x == y || (std::isnan(x) && std::isnan(y))) return 0.0;
    if (std::isnan(x) || std::isnan(y)) return INF;
    return std::abs(y - x);
}

/** @brief A norm of B - A over the same norm of A; 0 / 0 is 0. */
double relative(double d, double a) {
    if (a > 0.0) return d / a;
    return d == 0.0 ? 0.0 : INF;
}

struct Norms {
    double l1 = 0.0, l2 = 0.0, linf = 0.0;
};

/** @brief How two arrays of values differ, entry by entry (cells, points, rows; vectors by magnitude). */
struct Stats {
    uint64_t n = 0;
    uint64_t differing = 0;  ///< entries whose stored values differ bitwise
    Norms d, a;              ///< of |B - A| and of |A|: mean, root mean square, max
    uint64_t at = 0;         ///< the entry of the largest |B - A|
};

Stats compare(const std::vector<double> & a, const std::vector<double> & b, size_t components) {
    Stats s;
    s.n = a.size() / components;
    for (size_t i = 0; i < s.n; i++) {
        double d2 = 0.0, a2 = 0.0;
        bool bits = false;
        for (size_t k = 0; k < components; k++) {
            const double x = a[i * components + k], y = b[i * components + k];
            bits = bits || std::memcmp(&x, &y, sizeof x) != 0;
            const double d = difference(x, y);
            d2 += d * d;
            if (std::isfinite(x)) a2 += x * x;
        }
        const double d = std::sqrt(d2), m = std::sqrt(a2);
        s.differing += bits;
        s.d.l1 += d;
        s.d.l2 += d2;
        s.a.l1 += m;
        s.a.l2 += a2;
        s.a.linf = std::max(s.a.linf, m);
        if (d > s.d.linf) {
            s.d.linf = d;
            s.at = i;
        }
    }
    if (s.n > 0) {
        const double n = static_cast<double>(s.n);
        s.d.l1 /= n;
        s.d.l2 = std::sqrt(s.d.l2 / n);
        s.a.l1 /= n;
        s.a.l2 = std::sqrt(s.a.l2 / n);
    }
    return s;
}

double relative_max(const Stats & s) { return relative(s.d.linf, s.a.linf); }

Status status(const Stats & s, double tol) {
    if (s.differing == 0) return Status::same;
    return relative_max(s) <= tol ? Status::within : Status::beyond;
}

/** @brief A file of the comparison, at the snapshot compared. */
struct Side {
    output::Contents c;
    size_t snapshot = 0;
    json::Value run;  ///< its run block
    std::shared_ptr<Mesh> mesh;   ///< its case's mesh, where needed and found
    std::vector<uint64_t> local;  ///< [global cell] local cell of mesh
    std::string mesh_from;        ///< where the mesh came from
    std::string no_mesh;          ///< why there is none
    bool mesh_tried = false;

    double t() const { return c.snapshots[snapshot].t; }
    int64_t step() const { return c.snapshots[snapshot].step; }
};

/** @brief A field compared. */
struct FieldResult {
    std::string name, units, location;
    uint32_t components = 1;
    std::string problem;  ///< why it is not compared (units or components differ), else empty
    Stats stats;
    double tol = 0.0;
    Status result = Status::same;
    std::vector<double> a_at, b_at;  ///< the values at the largest difference
    bool has_x = false;
    std::array<double, 3> x{};
};

/** @brief A row of the restart comparison, values as text. */
struct Row {
    Status s = Status::same;
    std::string item, a, b;
};

/** @brief An array of restart state (an attribute, a named array, the face records) compared. */
struct ArrayResult {
    std::string name;
    Status s = Status::same;
    std::string problem;
    Stats stats;
    double a = 0.0, b = 0.0;  ///< attributes: the values
};

struct RestartResult {
    std::vector<Row> header;
    std::vector<ArrayResult> attributes, arrays;
    ArrayResult faces;
    bool has_faces = false;
};

struct RemapResult {
    remap::Report report;
    std::vector<std::string> names;
    double kept = 0.0;  ///< largest |integral after - before| over the integral of the magnitude
};

struct Comparison {
    std::string how;  ///< "same_mesh", "file_order", "remap"
    Status time = Status::same, step = Status::same;
    Tolerance tolerance;
    std::vector<FieldResult> fields;
    std::vector<std::string> only_a, only_b;
    bool restart = false;
    RestartResult state;
    bool remapped = false;
    RemapResult remap;
    std::vector<std::string> notes;
    Status result = Status::same;
};

std::string units_text(const std::string & units) {
    if (units.empty()) return "not recorded";
    return units == "1" ? "dimensionless" : units;
}

std::string real(double v) { return format("%.6g", v); }
std::string seconds(double t) { return format("%.9g", t); }

std::string list_text(const std::vector<std::string> & items) {
    std::string out;
    for (const auto & s : items) out += (out.empty() ? "" : ", ") + s;
    return out;
}

/** @brief Keeps std::cout quiet while it lives: the mesh setup's progress lines are not part of the report. */
class Quiet {
    public:
        Quiet() : saved(std::cout.rdbuf(sink.rdbuf())) {}
        ~Quiet() { std::cout.rdbuf(saved); }
        Quiet(const Quiet &) = delete;
        Quiet & operator=(const Quiet &) = delete;

    private:
        std::ostringstream sink;
        std::streambuf * saved;
};

json::Value parsed_run(const output::Contents & c) {
    if (c.run.empty()) return {};
    try {
        return json::parse(c.run);
    } catch (const std::exception &) {
        return {};
    }
}

const json::Value * at(const json::Value & v, std::initializer_list<const char *> path) {
    const json::Value * p = &v;
    for (const char * key : path) {
        if (p->type != json::Value::Type::OBJECT) return nullptr;
        p = p->find(key);
        if (!p) return nullptr;
    }
    return p;
}

std::string string_at(const json::Value & v, std::initializer_list<const char *> path) {
    const json::Value * p = at(v, path);
    return p && p->type == json::Value::Type::STRING ? p->string : "";
}

/** @brief The block's files by kind and path as given, with their SHA-256 ("" if not hashed). */
std::map<std::string, std::string> read_files(const json::Value & run) {
    std::map<std::string, std::string> out;
    const json::Value * files = at(run, {"files"});
    if (!files || files->type != json::Value::Type::ARRAY) return out;
    for (const json::Value & f : files->array) {
        out[string_at(f, {"kind"}) + " " + string_at(f, {"path"})] = string_at(f, {"sha256"});
    }
    return out;
}

/** @brief Notes on the files the runs of A and B read that differ (SHA-256), or that only one of them read. */
void read_files_notes(const json::Value & a, const json::Value & b, std::vector<std::string> & notes) {
    if (!at(a, {"files"}) || !at(b, {"files"})) return;
    const auto files_a = read_files(a), files_b = read_files(b);
    for (const auto & [file, sha] : files_a) {
        const auto other = files_b.find(file);
        if (other == files_b.end()) {
            notes.push_back("only the run of A read " + file);
        } else if (!sha.empty() && !other->second.empty() && sha != other->second) {
            notes.push_back("the runs of A and B read different " + file + " (SHA-256)");
        }
    }
    for (const auto & [file, sha] : files_b) {
        if (!files_a.contains(file)) notes.push_back("only the run of B read " + file);
    }
}

/**
 * @brief The mesh of a file's case (collective): mesh_file, else the [mesh]
 *        table of the input its run block records, with its [[periodic]]
 *        entries; mesh files named relative to the run's directory.
 */
void find_mesh(Side & s, const std::string & mesh_file) {
    if (s.mesh_tried) return;
    s.mesh_tried = true;
    if (s.c.dim != 0 && s.c.dim != N_DIM) {
        s.no_mesh = "its case is " + std::to_string(s.c.dim) + "D and this is the " + std::to_string(N_DIM) +
                    "D solver; mallard --dim " + std::to_string(s.c.dim) + " diff ... runs the other";
        return;
    }
    std::string text = string_at(s.run, {"input", "resolved"});
    if (text.empty()) text = string_at(s.run, {"input", "text"});
    Input recorded(Input::table_type{});
    if (!text.empty()) recorded = parse_input_string(text, s.c.path + " (its run block)");
    Input setup(Input::table_type{});
    const std::string directory = string_at(s.run, {"run", "directory"});
    std::string recorded_file;
    if (!mesh_file.empty()) {
        Input mesh(Input::table_type{});
        mesh.as_table()["type"] = Input(std::string("file"));
        mesh.as_table()["filename"] = Input(mesh_file);
        setup.as_table()["mesh"] = mesh;
        s.mesh_from = mesh_file;
    } else if (recorded.contains("mesh")) {
        Input mesh = recorded.at("mesh");
        const std::string type = find_value_or<std::string>(recorded, "mesh", "type", "file");
        if (type == "file") {
            fs::path file = find_value_or<std::string>(recorded, "mesh", "filename", "mesh.msh");
            if (file.is_relative() && !directory.empty()) file = fs::path(directory) / file;
            recorded_file = file.string();
            mesh.as_table()["filename"] = Input(recorded_file);
            s.mesh_from = recorded_file + ", as its run block records";
        } else {
            s.mesh_from = "the " + type + " mesh its run block's input generates";
        }
        setup.as_table()["mesh"] = mesh;
    } else {
        s.no_mesh = s.run.is_null() ? "it has no run block (files before Mallard 0.9), so no record of its mesh"
                                           : "its run block records no input";
        return;
    }
    if (recorded.contains("periodic")) setup.as_table()["periodic"] = recorded.at("periodic");
    if (!recorded_file.empty()) {
        std::error_code error;
        if (!fs::exists(recorded_file, error)) {
            s.no_mesh = "its mesh file " + recorded_file + " is not there";
            s.mesh_from.clear();
            return;
        }
        const std::string sha = string_at(s.run, {"mesh", "sha256"});
        if (!sha.empty() && provenance::sha256_file(recorded_file) != sha) {
            s.no_mesh = "its mesh file " + recorded_file + " has changed since the run (its SHA-256 differs)";
            s.mesh_from.clear();
            return;
        }
    }
    try {
        Quiet quiet;
        s.mesh = remap::source_mesh(setup, REMAP_DEGREE);
    } catch (const std::exception & e) {
        std::string why = e.what();
        if (why.ends_with('.')) why.pop_back();
        s.no_mesh = "its mesh cannot be set up: " + why;
        s.mesh_from.clear();
        return;
    }
    const uint64_t n = s.mesh->n_global_cells > 0 ? s.mesh->n_global_cells : s.mesh->n_cells;
    if (n != s.c.mesh.n_cells) {
        s.no_mesh = "its mesh (" + s.mesh_from + ") has " + count(n) + " cells, the file " + count(s.c.mesh.n_cells);
        s.mesh.reset();
        s.mesh_from.clear();
        return;
    }
    s.local.assign(n, 0);
    for (uint32_t c = 0; c < s.mesh->n_owned(); c++) s.local[s.mesh->h_global_cell(c)] = c;
}

std::string mesh_needed(const Side & s, const cli::Args & args, const char * option) {
    return s.c.path + ": " + s.no_mesh + (args.has(option) ? "" : std::string("; ") + option + " names its mesh file");
}

/** @brief Whether a file's values are per cell, in a mesh's cell order: what remap and coordinates need. */
bool cell_values(const output::Contents & c) {
    return (c.kind == output::Kind::snapshot || c.kind == output::Kind::restart) && c.global_order;
}

/** @brief B's cell fields remapped onto A's mesh, in A's global cell order, by name. */
std::map<std::string, std::vector<double>> remap_fields(Side & a, Side & b, const std::vector<FieldResult> & fields,
                                                        RemapResult & out) {
    std::vector<std::vector<double>> values;
    std::vector<int> groups;
    std::vector<std::pair<std::string, uint32_t>> layout;
    for (const FieldResult & f : fields) {
        if (!f.problem.empty()) continue;
        const std::vector<double> v = output::read(b.c, f.name, b.snapshot);
        for (uint32_t k = 0; k < f.components; k++) {
            std::vector<double> local(b.mesh->n_cells);
            for (uint32_t c = 0; c < b.mesh->n_cells; c++) local[c] = v[b.mesh->h_global_cell(c) * f.components + k];
            values.push_back(std::move(local));
            groups.push_back(f.name == "RHO" || f.name.starts_with("RHOY_") ? 0 : -1);
            out.names.push_back(f.components > 1 ? f.name + "[" + std::to_string(k) + "]" : f.name);
        }
        layout.emplace_back(f.name, f.components);
    }
    remap::Options options;
    options.degree = REMAP_DEGREE;
    options.axisymmetric = a.mesh->axisymmetric;
    const remap::Source source(b.mesh, values, groups, options);
    const std::vector<std::vector<double>> on_a = source.to(*a.mesh, a.mesh->n_owned(), out.report);
    for (size_t v = 0; v < out.names.size() && v < out.report.source_total.size(); v++) {
        const double scale = std::max(out.report.source_magnitude[v], std::numeric_limits<double>::min());
        out.kept = std::max(out.kept, std::abs(out.report.target_total[v] - out.report.source_total[v]) / scale);
    }
    std::map<std::string, std::vector<double>> remapped;
    size_t v = 0;
    for (const auto & [name, n] : layout) {
        std::vector<double> & r = remapped[name];
        r.assign(a.local.size() * n, 0.0);
        for (uint32_t k = 0; k < n; k++, v++) {
            for (uint32_t c = 0; c < a.mesh->n_owned(); c++) r[a.mesh->h_global_cell(c) * n + k] = on_a[v][c];
        }
    }
    return remapped;
}

std::string entry_name(const std::string & location) {
    return location == "point" ? "point" : location == "column" ? "row" : "cell";
}

/** @brief The face records of a restart file, as stored: keys and values. */
std::pair<std::vector<uint64_t>, std::vector<double>> read_faces(const output::Contents & c) {
    const RestartHeader & h = c.restart;
    std::pair<std::vector<uint64_t>, std::vector<double>> out;
    if (h.n_faces == 0) return out;
    std::ifstream in(c.path, std::ios::binary);
    in.seekg(static_cast<std::streamoff>(h.faces_offset));
    out.first.resize(h.n_faces);
    in.read(reinterpret_cast<char *>(out.first.data()), static_cast<std::streamsize>(h.n_faces * sizeof(uint64_t)));
    const size_t n = h.n_faces * RestartFaces::WIDTH;
    std::vector<char> bytes(n * h.real_size);
    in.read(bytes.data(), static_cast<std::streamsize>(bytes.size()));
    if (!in.good()) throw Trouble(c.path + " is truncated in its face records");
    out.second.resize(n);
    for (size_t i = 0; i < n; i++) {
        if (h.real_size == 8) {
            std::memcpy(&out.second[i], bytes.data() + i * 8, 8);
        } else {
            float x;
            std::memcpy(&x, bytes.data() + i * 4, 4);
            out.second[i] = static_cast<double>(x);
        }
    }
    return out;
}

ArrayResult compare_array(const std::string & name, const std::vector<double> & a, const std::vector<double> & b,
                          double tol) {
    ArrayResult r;
    r.name = name;
    if (a.size() != b.size()) {
        r.s = Status::beyond;
        r.problem = count(a.size()) + " values in A, " + count(b.size()) + " in B";
        return r;
    }
    r.stats = compare(a, b, 1);
    r.s = status(r.stats, tol);
    return r;
}

std::string precision(uint32_t bytes) { return bytes == 8 ? "double" : bytes == 4 ? "single" : "-"; }

RestartResult compare_restart(const Side & a, const Side & b, const Tolerance & tol) {
    const RestartHeader & x = a.c.restart;
    const RestartHeader & y = b.c.restart;
    RestartResult r;
    const auto row = [&](const std::string & name, const std::string & va, const std::string & vb, Status differ) {
        r.header.push_back({va == vb ? Status::same : differ, name, va, vb});
    };
    row("format version", std::to_string(x.version), std::to_string(y.version), Status::within);
    row("precision", precision(x.real_size), precision(y.real_size), Status::within);
    row("compression", x.chunk_cells ? "zlib" : "none", y.chunk_cells ? "zlib" : "none", Status::within);
    row("cells", count(x.n_cells), count(y.n_cells), Status::beyond);
    row("variables", count(x.names.size()), count(y.names.size()), Status::beyond);
    if (x.names.size() == y.names.size() && x.names != y.names) r.header.back().s = Status::beyond;

    std::set<std::string> names;
    for (const auto & [name, v] : x.attributes) names.insert(name);
    for (const auto & [name, v] : y.attributes) names.insert(name);
    for (const std::string & name : names) {
        const auto find = [&](const RestartAttributes & list) -> const double * {
            for (const auto & [n, v] : list) {
                if (n == name) return &v;
            }
            return nullptr;
        };
        const double * va = find(x.attributes);
        const double * vb = find(y.attributes);
        ArrayResult e;
        if (!va || !vb) {
            e.name = name;
            e.s = Status::beyond;
            e.problem = va ? "only in A" : "only in B";
            e.a = va ? *va : std::numeric_limits<double>::quiet_NaN();
            e.b = vb ? *vb : std::numeric_limits<double>::quiet_NaN();
        } else {
            e = compare_array(name, {*va}, {*vb}, tol.of(name));
            e.a = *va;
            e.b = *vb;
        }
        r.attributes.push_back(e);
    }
    std::set<std::string> arrays;
    for (const auto & [name, v] : x.arrays) arrays.insert(name);
    for (const auto & [name, v] : y.arrays) arrays.insert(name);
    for (const std::string & name : arrays) {
        const auto find = [&](const RestartArrays & list) -> const std::vector<double> * {
            for (const auto & [n, v] : list) {
                if (n == name) return &v;
            }
            return nullptr;
        };
        const std::vector<double> * va = find(x.arrays);
        const std::vector<double> * vb = find(y.arrays);
        if (!va || !vb) {
            ArrayResult e;
            e.name = name;
            e.s = Status::beyond;
            e.problem = va ? "only in A" : "only in B";
            r.arrays.push_back(e);
        } else {
            r.arrays.push_back(compare_array(name, *va, *vb, tol.of(name)));
        }
    }
    if (x.n_faces > 0 || y.n_faces > 0) {
        r.has_faces = true;
        const auto fa = read_faces(a.c);
        const auto fb = read_faces(b.c);
        if (fa.first != fb.first) {
            r.faces.name = "face records";
            r.faces.s = Status::beyond;
            r.faces.problem = fa.first.size() == fb.first.size() ? "of different faces"
                                                                 : count(fa.first.size()) + " faces in A, " +
                                                                       count(fb.first.size()) + " in B";
        } else {
            r.faces.name = "face records";
            r.faces.stats = compare(fa.second, fb.second, RestartFaces::WIDTH);
            r.faces.s = status(r.faces.stats, tol.all);
        }
    }
    return r;
}

/** @brief Which snapshot of each file to compare: by --step, --time or --snapshot, else the last. */
struct Selection {
    std::string option = "--snapshot", value = "last";
    double number = 0.0;
};

size_t pick_snapshot(const output::Contents & c, const Selection & pick, const char * which) {
    const std::string file = std::string(which) + " (" + c.path + ")";
    if (c.snapshots.empty()) throw Trouble(file + " holds no snapshot");
    const size_t n = c.snapshots.size();
    if (pick.option == "--step") {
        bool recorded = false;
        for (size_t i = 0; i < n; i++) {
            recorded = recorded || c.snapshots[i].step >= 0;
            if (static_cast<double>(c.snapshots[i].step) == pick.number) return i;
        }
        throw Trouble(file + (recorded ? " has no snapshot of step " + pick.value
                                       : " records no steps; --time or --snapshot selects its snapshot"));
    }
    if (pick.option == "--time") {
        size_t best = 0;
        for (size_t i = 1; i < n; i++) {
            if (std::abs(c.snapshots[i].t - pick.number) < std::abs(c.snapshots[best].t - pick.number)) best = i;
        }
        return best;
    }
    if (pick.value == "last") return n - 1;
    if (pick.number >= static_cast<double>(n)) {
        throw Trouble(file + " has " + count(n) + " snapshots, so no snapshot " + pick.value + " (they count from 0)");
    }
    return static_cast<size_t>(pick.number);
}

Side open_side(const std::string & path, const Selection & snapshot, const char * which) {
    Side s;
    if (output::is_mesh_file(path) || (s.c = output::open(path)).kind == output::Kind::mesh) {
        throw Trouble(path + " is a mesh file, which holds no field values; mallard diff compares snapshots, restart, "
                             "probe and CSV files");
    }
    s.snapshot = pick_snapshot(s.c, snapshot, which);
    s.run = parsed_run(s.c);
    return s;
}

Comparison compare_files(Side & a, Side & b, const cli::Args & args) {
    Comparison r;
    parse_tolerance(args.value_or("--tol", ""), r.tolerance);
    const bool remap = args.has("--remap");
    if (a.c.dim && b.c.dim && a.c.dim != b.c.dim) {
        throw Trouble("A (" + a.c.path + ") is " + std::to_string(a.c.dim) + "D and B (" + b.c.path + ") " +
                      std::to_string(b.c.dim) + "D");
    }

    std::vector<std::string> wanted;
    if (args.has("--fields")) {
        wanted = split(args.value_or("--fields", ""));
        for (const std::string & name : wanted) {
            const bool in_a = a.c.field(name) != nullptr, in_b = b.c.field(name) != nullptr;
            if (!in_a || !in_b) {
                throw Trouble("--fields names " + name + ", which " +
                              (in_a ? "B" : in_b ? "A" : "neither file") + " holds" + (in_a || in_b ? " not" : ""));
            }
        }
    } else {
        for (const auto & f : a.c.fields) {
            if (b.c.field(f.name)) {
                wanted.push_back(f.name);
            } else {
                r.only_a.push_back(f.name);
            }
        }
        for (const auto & f : b.c.fields) {
            if (!a.c.field(f.name)) r.only_b.push_back(f.name);
        }
    }
    if (wanted.empty()) {
        throw Trouble("A (" + a.c.path + ") and B (" + b.c.path + ") have no field in common: A holds " +
                      (a.c.fields.empty() ? "none" : count(a.c.fields.size())) + ", B " +
                      (b.c.fields.empty() ? "none" : count(b.c.fields.size())) + "; mallard info lists them");
    }
    for (const auto & [name, tol] : r.tolerance.field) {
        if (std::find(wanted.begin(), wanted.end(), name) == wanted.end() &&
            !(a.c.kind == output::Kind::restart && b.c.kind == output::Kind::restart)) {
            throw Trouble("--tol names " + name + ", which is not a field compared");
        }
    }

    for (const std::string & name : wanted) {
        const output::Field & fa = *a.c.field(name);
        const output::Field & fb = *b.c.field(name);
        FieldResult f;
        f.name = name;
        f.units = !fa.units.empty() ? fa.units : fb.units;
        f.location = fa.location;
        f.components = fa.components;
        f.tol = r.tolerance.of(name);
        if (!fa.units.empty() && !fb.units.empty() && fa.units != fb.units) {
            f.problem = "units differ: " + fa.units + " in A, " + fb.units + " in B";
        } else if (fa.components != fb.components) {
            f.problem = "components differ: " + std::to_string(fa.components) + " in A, " +
                        std::to_string(fb.components) + " in B";
        } else if (remap && fa.location != "cell") {
            f.problem = "not compared: --remap compares cell values, and this field's are per " + fa.location;
        }
        r.fields.push_back(std::move(f));
    }

    const double scale = std::max(std::abs(a.t()), std::abs(b.t()));
    const double dt = difference(a.t(), b.t());
    r.time = dt == 0.0 ? Status::same : dt <= r.tolerance.all * scale ? Status::within : Status::beyond;
    r.step = a.step() == b.step() ? Status::same : Status::within;

    const std::string sha_a = string_at(a.run, {"mesh", "sha256"});
    const std::string sha_b = string_at(b.run, {"mesh", "sha256"});
    read_files_notes(a.run, b.run, r.notes);
    std::map<std::string, std::vector<double>> remapped;
    if (remap) {
        for (Side * s : {&a, &b}) {
            if (!cell_values(s->c)) {
                throw Trouble(s->c.path + ": --remap needs cell values in the mesh's global cell order (restart files, "
                                          "VTKHDF and XDMF snapshots of the whole mesh), not those of " + s->c.format);
            }
        }
        find_mesh(a, args.value_or("--mesh-a", ""));
        if (!a.mesh) throw Trouble(mesh_needed(a, args, "--mesh-a"));
        find_mesh(b, args.value_or("--mesh-b", ""));
        if (!b.mesh) throw Trouble(mesh_needed(b, args, "--mesh-b"));
        r.how = "remap";
        r.remapped = true;
        remapped = remap_fields(a, b, r.fields, r.remap);
    } else {
        if (a.c.mesh.n_cells != b.c.mesh.n_cells && a.c.kind != output::Kind::table) {
            throw Trouble("A (" + a.c.path + ") has " + count(a.c.mesh.n_cells) + " " +
                          (a.c.kind == output::Kind::probe ? "elements" : "cells") + " and B (" + b.c.path + ") " +
                          count(b.c.mesh.n_cells) + ": different meshes; --remap compares B on A's mesh");
        }
        if (a.c.global_order != b.c.global_order) {
            const Side & other = a.c.global_order ? b : a;
            throw Trouble("the cells of " + other.c.path + " (" + other.c.format + ") are in the order of its own pieces " +
                          "or region, not the mesh's global cell order, so they do not match the other file's cell by "
                          "cell; compare files in global order (restart files, VTKHDF or XDMF snapshots of the whole "
                          "mesh), or two files of the same layout");
        }
        r.how = a.c.global_order ? "same_mesh" : "file_order";
        if (!sha_a.empty() && !sha_b.empty() && sha_a != sha_b) {
            r.notes.push_back("the run blocks of A and B record different mesh files (SHA-256) with the same cell "
                              "count; --remap compares them as different meshes");
        }
        if (!a.c.global_order) {
            r.notes.push_back("values compared in the files' own order, which neither file states to be the mesh's: "
                              "right for two files of the same layout (one rank, or the same partition)");
        }
    }

    for (FieldResult & f : r.fields) {
        if (!f.problem.empty()) {
            f.result = f.location != "cell" && remap ? Status::within : Status::beyond;
            continue;
        }
        const std::vector<double> va = output::read(a.c, f.name, a.snapshot);
        const std::vector<double> vb = remap ? remapped.at(f.name) : output::read(b.c, f.name, b.snapshot);
        if (va.size() != vb.size()) {
            throw Trouble(f.name + " has " + count(va.size() / f.components) + " values in A and " +
                          count(vb.size() / f.components) + " in B; --remap compares cell values on different meshes");
        }
        f.stats = compare(va, vb, f.components);
        f.result = status(f.stats, f.tol);
        if (f.stats.d.linf > 0.0) {
            const size_t i = f.stats.at * f.components;
            f.a_at.assign(va.begin() + static_cast<std::ptrdiff_t>(i), va.begin() + static_cast<std::ptrdiff_t>(i + f.components));
            f.b_at.assign(vb.begin() + static_cast<std::ptrdiff_t>(i), vb.begin() + static_cast<std::ptrdiff_t>(i + f.components));
        }
    }

    const bool locate = std::any_of(r.fields.begin(), r.fields.end(), [](const FieldResult & f) {
        return f.stats.d.linf > 0.0 && f.location == "cell";
    });
    if (locate && cell_values(a.c)) {
        find_mesh(a, args.value_or("--mesh-a", ""));
        if (a.mesh) {
            for (FieldResult & f : r.fields) {
                if (f.stats.d.linf <= 0.0 || f.location != "cell") continue;
                const uint64_t c = a.local[f.stats.at];
                f.has_x = true;
                for (size_t d = 0; d < N_DIM; d++) f.x[d] = static_cast<double>(a.mesh->h_cell_coords(c, d));
            }
        } else {
            r.notes.push_back("no cell coordinates for A: " + mesh_needed(a, args, "--mesh-a"));
        }
    } else if (locate) {
        r.notes.push_back("cell coordinates not shown: A's cells are not in the order of a mesh (" + a.c.format + ")");
    }

    if (a.c.kind == output::Kind::restart && b.c.kind == output::Kind::restart && !remap) {
        r.restart = true;
        r.state = compare_restart(a, b, r.tolerance);
    }

    r.result = worst(r.time, r.step);
    for (const FieldResult & f : r.fields) r.result = worst(r.result, f.result);
    if (r.restart) {
        for (const Row & row : r.state.header) r.result = worst(r.result, row.s);
        for (const ArrayResult & e : r.state.attributes) r.result = worst(r.result, e.s);
        for (const ArrayResult & e : r.state.arrays) r.result = worst(r.result, e.s);
        if (r.state.has_faces) r.result = worst(r.result, r.state.faces.s);
        if (!r.only_a.empty() || !r.only_b.empty()) r.result = Status::beyond;
    } else if (!r.only_a.empty() || !r.only_b.empty()) {
        r.result = worst(r.result, Status::within);
    }
    if (r.remapped) r.result = worst(r.result, Status::within);
    return r;
}

// ---------------------------------------------------------------------------
// Text

using Kind = report::Column::Kind;

/** @brief The status column's word for a comparison: ok identical, WARN within tolerance, ERROR beyond it. */
report::Status level(Status s) {
    return s == Status::same ? report::Status::OK : s == Status::within ? report::Status::WARNING : report::Status::ERROR;
}

std::string result_line(const Comparison & r) {
    std::string what;
    switch (r.result) {
        case Status::same:
            what = r.restart ? "identical, the same state bit for bit" : "identical, every value compared bit for bit";
            break;
        case Status::within: what = "within tolerance, differences but none beyond it"; break;
        case Status::beyond: {
            std::vector<std::string> beyond;
            for (const FieldResult & f : r.fields) {
                if (f.result == Status::beyond) beyond.push_back(f.name);
            }
            what = "beyond tolerance" + (beyond.empty() ? std::string() : ": " + list_text(beyond));
            break;
        }
    }
    return std::string("Result: ") + report::status_word(level(r.result)) + ", " + what +
           (r.result == Status::beyond ? " (exit status 1)" : " (exit status 0)");
}

std::string side_text(const Side & s) {
    const std::string kind = output::kind_name(s.c.kind);
    std::string out = s.c.path + " (" + (s.c.format == kind ? kind : kind + ", " + s.c.format);
    if (s.c.snapshots.size() > 1) out += ", snapshot " + std::to_string(s.snapshot) + " of " + count(s.c.snapshots.size());
    return out + ")";
}

std::string how_text(const Comparison & r, const Side & a) {
    const std::string n = count(a.c.mesh.n_cells);
    if (r.how == "remap") return "B remapped onto A's mesh (" + n + " cells), then cell by cell";
    if (a.c.kind == output::Kind::table) return "row by row";
    if (a.c.kind == output::Kind::probe) return "sample by sample (" + count(a.c.mesh.n_points) + " points)";
    if (r.how == "same_mesh") return "cell by cell, in the mesh's global cell order (" + n + " cells)";
    return "value by value, in the files' order";
}

std::string tolerance_text(const Tolerance & t) {
    std::string out = t.all == 0.0 ? "0 (values must be equal)" : real(t.all);
    for (const auto & [name, v] : t.field) out += ", " + name + " " + real(v);
    return out + "; on max |B - A| / max |A| per field";
}

void fields_section(report::Text & text, const Comparison & r) {
    std::string entries = "cell";
    if (!r.fields.empty()) entries = entry_name(r.fields.front().location);
    std::vector<report::Row> rows, relative_rows, located;
    bool vectors = false, problems = false;
    report::Status s = report::Status::OK;
    for (const FieldResult & f : r.fields) {
        const report::Status row = level(f.result);
        s = report::worse(s, row);
        vectors = vectors || f.components > 1;
        if (!f.problem.empty()) {
            problems = true;
            rows.push_back({row, {f.name, units_text(f.units), f.problem}});
            relative_rows.push_back({row, {f.name, "dimensionless", "-", "-", "-", real(f.tol)}});
            continue;
        }
        const Stats & st = f.stats;
        rows.push_back({row, {f.name, units_text(f.units), count(st.differing) + " of " + count(st.n), real(st.d.l1),
                              real(st.d.l2), real(st.d.linf)}});
        relative_rows.push_back({row, {f.name, "dimensionless", real(relative(st.d.l1, st.a.l1)),
                                       real(relative(st.d.l2, st.a.l2)), real(relative_max(st)), real(f.tol)}});
        if (st.d.linf > 0.0) {
            for (size_t k = 0; k < f.a_at.size(); k++) {
                // Enough digits to tell A from B
                const int digits = format("%.9g", f.a_at[k]) == format("%.9g", f.b_at[k]) ? 17 : 9;
                std::vector<std::string> cells = {f.a_at.size() > 1 ? f.name + " " + "xyz"[k] : f.name, count(st.at)};
                for (size_t d = 0; d < N_DIM; d++) cells.push_back(f.has_x ? format("%.6g", f.x[d]) : "-");
                cells.push_back(format("%.*g", digits, f.a_at[k]));
                cells.push_back(format("%.*g", digits, f.b_at[k]));
                cells.push_back(units_text(f.units));
                located.push_back({report::Status::NONE, std::move(cells)});
            }
        }
    }
    text.section("Fields", s);
    text.note("|B - A| per " + entries + ", in the field's units: L1, L2, Linf are its mean, root mean square and max "
              "over the " + entries + "s, not weighted by volume; differing: values not bitwise equal.");
    if (vectors) text.note("Fields with several components: |B - A| is the magnitude of the difference vector.");
    if (problems) text.note("Fields whose units or components differ are not compared.");
    text.table({"field", {"units", Kind::UNITS}, "differing " + entries + "s", {"L1", Kind::VALUE},
                {"L2", Kind::VALUE}, {"Linf", Kind::VALUE}},
               rows);

    text.section("Relative differences", s);
    text.note("A norm of B - A over the same norm of A; a field passes if its Linf is at most its tolerance.");
    text.table({"field", {"units", Kind::UNITS}, {"L1", Kind::VALUE}, {"L2", Kind::VALUE}, {"Linf", Kind::VALUE},
                {"tolerance on Linf", Kind::VALUE}},
               relative_rows);

    if (located.empty()) return;
    text.section("Largest differences");
    text.note("Per field, the " + entries + " of max |B - A| and both values there" +
              (vectors ? std::string(", a row per component of a vector") : std::string()) + ".");
    std::vector<report::Column> header = {"field", entries};
    for (size_t d = 0; d < N_DIM; d++) header.push_back(std::string(1, "xyz"[d]) + " [m]");
    header.insert(header.end(), {{"A", Kind::VALUE}, {"B", Kind::VALUE}, {"units", Kind::UNITS}});
    text.table(header, located, false);
    if (std::any_of(r.fields.begin(), r.fields.end(), [](const FieldResult & f) { return f.stats.d.linf > 0.0 && !f.has_x; })) {
        text.note("-: no cell coordinates (see the notes).");
    }
}

void restart_section(report::Text & text, const RestartResult & st) {
    report::Status s = report::Status::OK;
    std::vector<report::Row> header;
    for (const Row & row : st.header) {
        s = report::worse(s, level(row.s));
        header.push_back({level(row.s), {row.item, row.a, row.b}});
    }
    std::vector<report::Row> attributes;
    for (const ArrayResult & e : st.attributes) {
        s = report::worse(s, level(e.s));
        const double d = difference(e.a, e.b);
        attributes.push_back({level(e.s), {e.name, std::isnan(e.a) ? "-" : format("%.17g", e.a),
                                           std::isnan(e.b) ? "-" : format("%.17g", e.b),
                                           e.problem.empty() ? real(d) : e.problem}});
    }
    std::vector<ArrayResult> all = st.arrays;
    if (st.has_faces) all.push_back(st.faces);
    std::vector<report::Row> arrays;
    for (const ArrayResult & e : all) {
        s = report::worse(s, level(e.s));
        if (!e.problem.empty()) {
            arrays.push_back({level(e.s), {e.name, e.problem}});
        } else {
            arrays.push_back({level(e.s), {e.name, count(e.stats.differing) + " of " + count(e.stats.n),
                                           real(e.stats.d.linf), real(relative_max(e.stats))}});
        }
    }
    text.section("Restart state", s);
    text.note("Bit for bit; the run and snapshot blocks are not compared.");
    text.table({"header", "A", "B"}, header);
    if (!attributes.empty()) {
        text.blank();
        text.note("Attributes are the solver's scalar state; their units are not recorded.");
        text.table({"attribute", "A", "B", "|B - A|"}, attributes);
    }
    if (!arrays.empty()) {
        text.blank();
        text.note("Units not recorded; relative: max |B - A| over max |A|, dimensionless.");
        text.table({"array", "differing values", "Linf |B - A|", "Linf relative"}, arrays);
    }
}

report::Text diff_text(const Comparison & r, const Side & a, const Side & b, const bool color) {
    report::Text text(color);
    text.section("Diff", level(r.result));
    text.line(result_line(r));
    text.line("A: " + side_text(a));
    text.line("B: " + side_text(b));
    text.line("Compared: " + how_text(r, a));
    if (!a.mesh_from.empty()) text.line("Mesh of A: " + a.mesh_from);
    if (!b.mesh_from.empty()) text.line("Mesh of B: " + b.mesh_from);
    text.line("Tolerance: " + tolerance_text(r.tolerance));

    std::vector<report::Row> rows = {
        {level(r.time), {"t", seconds(a.t()), seconds(b.t()), real(difference(a.t(), b.t())), "s"}}};
    report::Status time = level(r.time);
    const auto step = [](int64_t s) { return s >= 0 ? std::to_string(s) : std::string("-"); };
    if (a.step() >= 0 || b.step() >= 0) {
        const bool both = a.step() >= 0 && b.step() >= 0;
        const report::Status row = both ? level(r.step) : report::Status::WARNING;
        time = report::worse(time, row);
        rows.push_back({row, {"step", step(a.step()), step(b.step()),
                              both ? std::to_string(std::abs(b.step() - a.step())) : "-", "steps"}});
    }
    text.section("Time", time);
    text.note("t compares with the default tolerance, relative to the larger |t|; steps that differ are a warning.");
    text.table({"quantity", {"A", Kind::VALUE}, {"B", Kind::VALUE}, {"|B - A|", Kind::VALUE}, {"units", Kind::UNITS}},
               rows);

    if (r.remapped) {
        const remap::Report & p = r.remap.report;
        text.section("Remap", report::Status::WARNING);
        text.note("Conservative, degree 2, limited. Remapped results differ from A by the remap's own error too, so a "
                  "remap is never identical. A cells outside B's mesh take the nearest B cell's value.");
        text.table({"B cells", "A cells", "integrals kept to", "A cells partly covered", "A cells outside B"},
                   {{report::Status::NONE, {count(p.n_source), count(p.n_target), format("%.1e", r.remap.kept),
                                            count(p.n_partial), count(p.n_nearest)}}},
                   false);
        text.note("integrals kept to: max |integral on A - on B| over the integral of |B|, dimensionless.");
    }

    fields_section(text, r);
    if (!r.only_a.empty() || !r.only_b.empty()) {
        text.section("Fields in one file only", r.restart ? report::Status::ERROR : report::Status::WARNING);
        text.note(r.restart ? "A different state." : "Not compared.");
        if (!r.only_a.empty()) text.line("Only in A: " + list_text(r.only_a));
        if (!r.only_b.empty()) text.line("Only in B: " + list_text(r.only_b));
    }
    if (r.restart) restart_section(text, r.state);
    if (!r.notes.empty()) {
        text.section("Notes");
        for (const std::string & n : r.notes) text.line(n);
    }
    text.blank();
    text.line(result_line(r));
    return text;
}

// ---------------------------------------------------------------------------
// JSON

std::string strings_json(const std::vector<std::string> & list) {
    std::string out = "[";
    for (const auto & s : list) out += (out.size() > 1 ? "," : "") + json::quote(s);
    return out + "]";
}

std::string numbers_json(const std::vector<double> & list) {
    std::string out = "[";
    for (size_t i = 0; i < list.size(); i++) out += (i ? "," : "") + json::number(list[i]);
    return out + "]";
}

const char * status_name(Status s) { return report::status_name(level(s)); }

std::string norms_json(const Norms & n) { return json::Object().add("l1", n.l1).add("l2", n.l2).add("linf", n.linf).str(); }

std::string stats_json(json::Object & o, const Stats & s) {
    o.add("values", s.n).add("differing", s.differing).raw("absolute", norms_json(s.d));
    o.raw("relative", norms_json({relative(s.d.l1, s.a.l1), relative(s.d.l2, s.a.l2), relative_max(s)}));
    return o.str();
}

std::string side_json(const Side & s) {
    const output::Snapshot & snap = s.c.snapshots[s.snapshot];
    return json::Object()
        .add("path", s.c.path)
        .add("kind", std::string(output::kind_name(s.c.kind)))
        .add("format", s.c.format)
        .add("snapshot", s.snapshot)
        .add("t", snap.t)
        .raw("step", snap.step >= 0 ? std::to_string(snap.step) : "null")
        .raw("mesh", s.mesh_from.empty() ? "null" : json::quote(s.mesh_from))
        .str();
}

std::string array_json(const ArrayResult & e) {
    json::Object o;
    o.add("name", e.name).add("status", std::string(status_name(e.s)));
    o.raw("problem", e.problem.empty() ? "null" : json::quote(e.problem));
    return stats_json(o, e.stats);
}

std::string comparison_json(const Comparison & r, const Side & a, const Side & b) {
    json::Object o;
    o.add("result", std::string(status_name(r.result))).add("exit_status", r.result == Status::beyond ? 1 : 0);
    o.raw("a", side_json(a)).raw("b", side_json(b)).add("compared", r.how);
    o.raw("tolerance", json::Object().add("default", r.tolerance.all).add_numbers("fields", {r.tolerance.field.begin(), r.tolerance.field.end()}).str());
    o.raw("time", json::Object()
                      .add("status", std::string(status_name(r.time)))
                      .add("a", a.t())
                      .add("b", b.t())
                      .add("difference", difference(a.t(), b.t()))
                      .raw("step_a", a.step() >= 0 ? std::to_string(a.step()) : "null")
                      .raw("step_b", b.step() >= 0 ? std::to_string(b.step()) : "null")
                      .str());
    std::string fields = "[";
    for (const FieldResult & f : r.fields) {
        json::Object e;
        e.add("name", f.name).add("units", f.units).add("components", f.components).add("location", f.location);
        e.add("status", std::string(status_name(f.result))).add("tolerance", f.tol);
        e.raw("problem", f.problem.empty() ? "null" : json::quote(f.problem));
        if (f.stats.d.linf > 0.0) {
            std::string x = "null";
            if (f.has_x) x = numbers_json(std::vector<double>(f.x.begin(), f.x.begin() + N_DIM));
            e.raw("max_at", json::Object()
                                .add("index", f.stats.at)
                                .raw("x", x)
                                .raw("a", numbers_json(f.a_at))
                                .raw("b", numbers_json(f.b_at))
                                .str());
        } else {
            e.raw("max_at", "null");
        }
        fields += (fields.size() > 1 ? "," : "") + stats_json(e, f.stats);
    }
    o.raw("fields", fields + "]");
    o.raw("only_in_a", strings_json(r.only_a)).raw("only_in_b", strings_json(r.only_b));
    if (r.restart) {
        std::string header = "[";
        for (const Row & row : r.state.header) {
            header += (header.size() > 1 ? "," : "") + json::Object()
                                                           .add("item", row.item)
                                                           .add("status", std::string(status_name(row.s)))
                                                           .add("a", row.a)
                                                           .add("b", row.b)
                                                           .str();
        }
        std::string attributes = "[", arrays = "[";
        for (const ArrayResult & e : r.state.attributes) {
            attributes += (attributes.size() > 1 ? "," : "") + json::Object()
                                                                   .add("name", e.name)
                                                                   .add("status", std::string(status_name(e.s)))
                                                                   .add("a", e.a)
                                                                   .add("b", e.b)
                                                                   .str();
        }
        for (const ArrayResult & e : r.state.arrays) arrays += (arrays.size() > 1 ? "," : "") + array_json(e);
        o.raw("restart", json::Object()
                             .raw("header", header + "]")
                             .raw("attributes", attributes + "]")
                             .raw("arrays", arrays + "]")
                             .raw("faces", r.state.has_faces ? array_json(r.state.faces) : "null")
                             .str());
    } else {
        o.raw("restart", "null");
    }
    if (r.remapped) {
        const remap::Report & p = r.remap.report;
        o.raw("remap", json::Object()
                           .add("degree", int(REMAP_DEGREE))
                           .add("source_cells", p.n_source)
                           .add("target_cells", p.n_target)
                           .add("integrals_kept_to", r.remap.kept)
                           .add("partly_covered", p.n_partial)
                           .add("outside", p.n_nearest)
                           .str());
    } else {
        o.raw("remap", "null");
    }
    o.raw("notes", strings_json(r.notes));
    return o.str();
}

} // namespace

int diff_main(const cli::Args & args) {
    int argc = args.argc;
    char ** argv = args.argv;
    comm::Session session(argc, argv);
    const cli::Command & command = *cli::command("diff");
    const std::string fmt = args.value_or("--format", "text");
    if (fmt != "text" && fmt != "json") return cli::usage_error(command, "--format takes text or json, not " + fmt);
    if (!args.has("--remap") && args.has("--mesh-b")) return cli::usage_error(command, "--mesh-b is for --remap");
    Tolerance tolerance;
    if (!parse_tolerance(args.value_or("--tol", ""), tolerance)) {
        return cli::usage_error(command, "--tol takes non-negative numbers, as 1e-6 or 1e-6,P=1e-3, not " +
                                             args.value_or("--tol", ""));
    }
    Selection snapshot;
    for (const char * option : {"--snapshot", "--step", "--time"}) {
        if (!args.has(option)) continue;
        if (args.has(snapshot.option) && snapshot.option != option) {
            return cli::usage_error(command, snapshot.option + " and " + option + " both select the snapshot; give one");
        }
        snapshot.option = option;
        snapshot.value = args.value_or(option, "");
        const bool whole = snapshot.option != "--time";
        if (!(snapshot.option == "--snapshot" && snapshot.value == "last") &&
            (!parse_number(snapshot.value, snapshot.number) || (whole && snapshot.number != std::floor(snapshot.number)))) {
            return cli::usage_error(command, snapshot.option + (whole ? " takes a whole number" : " takes a time in s") +
                                                 ", not " + snapshot.value);
        }
    }
    if (comm::size() > 1) {
        if (comm::is_root()) {
            logging::error(diagnostic_code::ONE_RANK, "mallard diff: runs on one rank, not " +
                                                          std::to_string(comm::size()) + "; start it without mpirun.");
        }
        return TROUBLE;
    }
    Kokkos::ScopeGuard kokkos(argc, argv);
    try {
        Side a = open_side(args.positional[0], snapshot, "A");
        Side b = open_side(args.positional[1], snapshot, "B");
        const Comparison r = compare_files(a, b, args);
        if (fmt == "json") {
            std::cout << comparison_json(r, a, b) << "\n";
        } else {
            diff_text(r, a, b, logging::color_stdout()).print();
        }
        const int status = r.result == Status::beyond ? DIFFERENT : 0;
        a.mesh.reset();
        b.mesh.reset();
        return status;
    } catch (const std::exception & e) {
        logging::error(diagnostic_code::DIFF, std::string("mallard diff: ") + e.what());
        return TROUBLE;
    }
}

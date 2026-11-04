/**
 * @file info_main.cpp
 * @brief `mallard info`: what a mesh, output or restart file, or a run directory, holds.
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 */

#include <array>
#include <cmath>
#include <exception>
#include <filesystem>
#include <fstream>
#include <initializer_list>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include <Kokkos_Core.hpp>

#include "comm.h"
#include "commands.h"
#include "json.h"
#include "log.h"
#include "mesh.h"
#include "mesh_quality.h"
#include "output_reader.h"

namespace fs = std::filesystem;

namespace {

using logging::count;
using logging::format;
using logging::item;

/** @brief A JSON value as text: strings as they are, numbers compactly, null as "-". */
std::string text(const json::Value * v) {
    if (!v) return "-";
    switch (v->type) {
        case json::Value::Type::STRING: return v->string;
        case json::Value::Type::NUMBER: return logging::real(v->number);
        case json::Value::Type::BOOL: return v->boolean ? "yes" : "no";
        case json::Value::Type::NUL: return "-";
        default: return "...";
    }
}

const json::Value * at(const json::Value & v, std::initializer_list<const char *> path) {
    const json::Value * p = &v;
    for (const char * key : path) {
        p = p->find(key);
        if (!p) return nullptr;
    }
    return p;
}

/** @brief "1 rank", "4 ranks". */
std::string plural(const json::Value * n, const std::string & noun) {
    return text(n) + " " + noun + (n && n->type == json::Value::Type::NUMBER && n->number == 1.0 ? "" : "s");
}

std::string list_text(const std::vector<std::string> & items, const char * sep) {
    std::string out;
    for (const auto & s : items) out += (out.empty() ? "" : sep) + s;
    return out;
}

/** @brief The input a provenance block records: input.resolved (version 2), else input.text as written. */
std::string recorded_input(const json::Value & p, bool & resolved) {
    resolved = false;
    if (const json::Value * r = at(p, {"input", "resolved"}); r && r->type == json::Value::Type::STRING) {
        resolved = true;
        return r->string;
    }
    if (const json::Value * t = at(p, {"input", "text"}); t && t->type == json::Value::Type::STRING) return t->string;
    return "";
}

/** @brief A file's provenance block, parsed; null if it has none. */
json::Value parsed_provenance(const output::Contents & c) {
    if (c.provenance.empty()) return {};
    try {
        return json::parse(c.provenance);
    } catch (const std::exception & e) {
        throw std::runtime_error(c.path + ": the provenance block is not valid JSON (" + e.what() + ")");
    }
}

void print_provenance(const json::Value & p) {
    logging::section("Provenance");
    std::string mallard = text(at(p, {"mallard", "version"}));
    if (const json::Value * sha = at(p, {"mallard", "git_sha"}); sha && !sha->is_null()) {
        const json::Value * dirty = at(p, {"mallard", "git_dirty"});
        mallard += " (" + sha->string.substr(0, 12) + (dirty && dirty->boolean ? ", dirty" : "") + ")";
    }
    item("Mallard", mallard + ", provenance version " + text(p.find("version")));
    item("Build", text(at(p, {"build", "dim"})) + "D, " + text(at(p, {"build", "precision"})) + ", " +
                      text(at(p, {"build", "build_type"})) + ", " + text(at(p, {"build", "backend"})) + ", MPI " +
                      text(at(p, {"build", "mpi"})) + ", HDF5 " + text(at(p, {"build", "hdf5"})));
    if (p.find("run")) {
        const json::Value * hosts = at(p, {"run", "hosts"});
        item("Run", text(at(p, {"run", "start_time"})) + ", " + plural(at(p, {"run", "ranks"}), "rank") +
                        (hosts ? " on " + plural(hosts, "host") : "") + ", " + text(at(p, {"run", "device"})));
        item("", "on " + text(at(p, {"run", "host"})) + " in " + text(at(p, {"run", "directory"})));
    }
    if (p.find("input")) {
        bool resolved = false;
        recorded_input(p, resolved);
        item("Input", text(at(p, {"input", "file"})) + ", sha256 " + text(at(p, {"input", "sha256"})));
        item("", std::string(resolved ? "resolved" : "as written") + "; mallard info --input prints it");
    }
    if (const json::Value * mesh = p.find("mesh")) {
        std::string m = text(mesh->find("type"));
        if (mesh->find("file")) m += ", " + text(mesh->find("file"));
        if (mesh->find("sha256")) m += ", sha256 " + text(mesh->find("sha256"));
        item("Mesh", m);
    }
    if (const json::Value * preset = p.find("preset"); preset && !preset->is_null()) {
        item("Preset", text(preset->find("name")));
    }
}

std::string time_text(const output::Contents & c) {
    const output::Snapshot & last = c.snapshots.back();
    std::string t = c.snapshots.size() == 1 ? "t = " + format("%.9g", last.t)
                                            : "t = " + format("%.9g", c.snapshots.front().t) + " to " + format("%.9g", last.t);
    if (last.step >= 0) t += ", step " + std::to_string(last.step) + (c.snapshots.size() > 1 ? " last" : "");
    return t;
}

void print_contents(const output::Contents & c) {
    using output::Kind;
    logging::section(c.path);
    std::string kind = output::kind_name(c.kind);
    if (c.snapshots.size() > 1) kind += ", " + count(c.snapshots.size()) + (c.kind == Kind::table ? " times" : " snapshots");
    item("Kind", kind);
    item("Format", c.format + (c.version >= 0 ? ", version " + std::to_string(c.version) : ""));
    std::vector<std::string> values;
    if (c.dim) values.push_back(std::to_string(c.dim) + "D");
    if (c.real_size) values.push_back(std::string(c.real_size == 8 ? "double" : "single") + " precision");
    if (c.compressed) values.emplace_back("compressed");
    if (!values.empty()) item("Values", list_text(values, ", "));
    const output::MeshSummary & m = c.mesh;
    if (c.kind == Kind::table) {
        item("Rows", count(m.n_points) + " per time");
    } else if (m.n_cells > 0 || m.n_points > 0) {
        std::vector<std::string> parts;
        if (m.n_cells > 0) parts.push_back(count(m.n_cells) + (c.kind == Kind::probe ? " elements" : " cells"));
        if (m.n_points > 0) parts.push_back(count(m.n_points) + " points");
        std::string line = list_text(parts, ", ");
        if (m.n_faces > 0) line += ", " + count(m.n_faces) + " faces";
        if (m.n_boundary_faces > 0) line += ", " + count(m.n_boundary_faces) + " boundary faces";
        item("Mesh", line);
        std::vector<std::string> types;
        for (const auto & [name, n] : m.cell_types) types.push_back(count(n) + " " + name);
        if (!types.empty()) item("", list_text(types, ", "));
    }
    if (m.has_bounds) {
        std::string box;
        for (size_t i = 0; i < size_t(c.dim ? c.dim : 3); i++) {
            box += (i ? " x " : "") + format("[%.6g, %.6g]", m.lo[i], m.hi[i]);
        }
        item("Bounds", box);
    }
    for (size_t z = 0; z < m.boundary_zones.size(); z++) {
        item(z ? "" : "Boundary zones", m.boundary_zones[z].first + ": " + count(m.boundary_zones[z].second) + " faces");
    }
    for (size_t z = 0; z < m.cell_zones.size(); z++) {
        item(z ? "" : "Cell zones", m.cell_zones[z].first + ": " + count(m.cell_zones[z].second) + " cells");
    }
    if (!c.mesh_file.empty()) item("Mesh file", c.mesh_file);
    if (!c.snapshots.empty() && c.kind != Kind::mesh) item("Time", time_text(c));
    if (c.kind == Kind::restart) {
        for (size_t a = 0; a < c.restart.attributes.size(); a++) {
            item(a ? "" : "Attributes", c.restart.attributes[a].first + " = " + format("%.9g", c.restart.attributes[a].second));
        }
        for (size_t a = 0; a < c.restart.arrays.size(); a++) {
            item(a ? "" : "Arrays", c.restart.arrays[a].first + ": " + count(c.restart.arrays[a].second.size()) + " values");
        }
        if (c.restart.n_faces > 0) item("Boundary states", count(c.restart.n_faces) + " characteristic boundary faces");
    }
    const std::vector<std::string> species = c.species();
    if (!species.empty()) item("Species", count(species.size()) + ": " + list_text(species, " "));
    for (const auto & note : c.notes) item("Note", note);

    if (!c.fields.empty()) {
        std::string where;
        if (c.kind == Kind::table) {
            where = "ranges of the last time";
        } else if (!c.snapshots.empty()) {
            where = "ranges at t = " + format("%.9g", c.snapshots.back().t);
        }
        logging::section("Fields", where);
        for (const auto & f : c.fields) {
            std::string line = f.units.empty() ? "-" : f.units;
            if (line.size() < 14) line.resize(14, ' ');
            std::vector<std::string> parts;
            if (!f.long_name.empty()) parts.push_back(f.long_name);
            if (f.components > 1) parts.push_back(std::to_string(f.components) + " components");
            if (!std::isnan(f.min)) {
                parts.push_back((f.components > 1 ? "magnitude " : "") + format("%.6g", f.min) + " to " +
                                format("%.6g", f.max));
            }
            item(f.name, line + " " + list_text(parts, ", "));
        }
    }
    const json::Value p = parsed_provenance(c);
    if (!p.is_null()) print_provenance(p);
}

std::string zones_json(const std::vector<std::pair<std::string, uint64_t>> & list) {
    json::Object o;
    for (const auto & [name, n] : list) o.add(name, n);
    return o.str();
}

std::string strings_json(const std::vector<std::string> & list) {
    std::string out = "[";
    for (const auto & s : list) out += (out.size() > 1 ? "," : "") + json::quote(s);
    return out + "]";
}

std::string contents_json(const output::Contents & c) {
    json::Object o;
    o.add("path", c.path).add("kind", std::string(output::kind_name(c.kind))).add("format", c.format);
    o.raw("version", c.version >= 0 ? std::to_string(c.version) : "null");
    o.raw("dim", c.dim ? std::to_string(c.dim) : "null");
    o.raw("precision", c.real_size == 8 ? "\"double\"" : c.real_size == 4 ? "\"single\"" : "null");
    o.add("compressed", c.compressed).add("global_order", c.global_order);
    const output::MeshSummary & m = c.mesh;
    json::Object mesh;
    mesh.add("cells", m.n_cells).add("points", m.n_points).add("faces", m.n_faces).add("boundary_faces", m.n_boundary_faces);
    mesh.raw("cell_types", zones_json(m.cell_types));
    if (m.has_bounds) {
        std::string box = "[";
        for (size_t i = 0; i < 3; i++) box += std::string(i ? "," : "") + "[" + json::number(m.lo[i]) + "," + json::number(m.hi[i]) + "]";
        mesh.raw("bounds", box + "]");
    } else {
        mesh.raw("bounds", "null");
    }
    mesh.raw("boundary_zones", zones_json(m.boundary_zones)).raw("cell_zones", zones_json(m.cell_zones));
    mesh.raw("file", c.mesh_file.empty() ? "null" : json::quote(c.mesh_file));
    o.raw("mesh", mesh.str());
    std::string snapshots = "[";
    for (const auto & s : c.snapshots) {
        json::Object e;
        e.add("t", s.t).raw("step", s.step >= 0 ? std::to_string(s.step) : "null").add("file", s.file).add("index", s.index);
        snapshots += (snapshots.size() > 1 ? "," : "") + e.str();
    }
    o.raw("snapshots", snapshots + "]");
    std::string fields = "[";
    for (const auto & f : c.fields) {
        json::Object e;
        e.add("name", f.name).add("components", f.components).add("units", f.units).add("long_name", f.long_name);
        e.add("location", f.location).add("min", f.min).add("max", f.max);
        fields += (fields.size() > 1 ? "," : "") + e.str();
    }
    o.raw("fields", fields + "]");
    o.raw("species", strings_json(c.species()));
    if (c.kind == output::Kind::restart) {
        std::vector<std::string> arrays;
        for (const auto & a : c.restart.arrays) arrays.push_back(a.first);
        o.raw("restart", json::Object()
                             .add_numbers("attributes", c.restart.attributes)
                             .raw("arrays", strings_json(arrays))
                             .add("boundary_faces", c.restart.n_faces)
                             .str());
    }
    o.raw("provenance", parsed_provenance(c).is_null() ? "null" : c.provenance);
    o.raw("notes", strings_json(c.notes));
    return o.str();
}

/** @brief The last start and end records of a run's mallard.jsonl, and its last record. */
struct Record {
    json::Value start, end, last;
    bool found = false;
};

Record run_record(const std::string & dir) {
    Record r;
    std::ifstream in(fs::path(dir) / "mallard.jsonl");
    std::string line;
    while (std::getline(in, line)) {
        try {
            json::Value v = json::parse(line);
            r.found = true;
            const std::string type = text(v.find("type"));
            if (type == "start") {
                r.start = v;
                r.end = json::Value{};
            } else if (type == "end") {
                r.end = v;
            }
            r.last = std::move(v);
        } catch (const std::exception &) {
            // A last line cut short by a run that was killed
        }
    }
    return r;
}

/** @brief The provenance of a run directory: its run record's, else that of its newest output file. */
json::Value run_provenance(const std::string & dir, const Record & record) {
    if (const json::Value * p = record.start.find("provenance")) return *p;
    json::Value newest;
    fs::file_time_type when{};
    for (const auto & f : output::run_files(dir)) {
        try {
            const output::Contents c = output::open(f);
            const auto t = fs::last_write_time(f);
            if (!c.provenance.empty() && (newest.is_null() || t > when)) {
                newest = parsed_provenance(c);
                when = t;
            }
        } catch (const std::exception &) {
            // mallard info DIR reports the files that do not open
        }
    }
    return newest;
}

int print_input(const json::Value & block, const std::string & what) {
    if (block.is_null()) {
        std::cerr << "mallard info: " << what << " has no provenance block (files of Mallard 0.9 and later have one), "
                  << "so no input.\n";
        return 1;
    }
    bool resolved = false;
    const std::string input = recorded_input(block, resolved);
    if (input.empty()) {
        std::cerr << "mallard info: the provenance of " << what << " records no input.\n";
        return 1;
    }
    std::cout << input;
    if (!input.ends_with('\n')) std::cout << "\n";
    if (!resolved) {
        std::cerr << "mallard info: provenance version " << text(block.find("version"))
                  << " records the input as written, not resolved (defaults, presets and units filled in); the "
                     "Mallard version that wrote it reads it the same way.\n";
    }
    return 0;
}

int run_directory(const std::string & dir, const cli::Args & args) {
    const Record record = run_record(dir);
    if (args.has("--input")) return print_input(run_provenance(dir, record), dir);
    std::vector<output::Contents> files;
    std::vector<std::pair<std::string, std::string>> errors;
    for (const auto & f : output::run_files(dir)) {
        try {
            files.push_back(output::open(f));
        } catch (const std::exception & e) {
            errors.emplace_back(f, e.what());
        }
    }
    if (args.has("--json")) {
        json::Object o;
        o.add("directory", dir);
        o.raw("start", record.start.type == json::Value::Type::OBJECT ? "true" : "false");
        o.raw("end", record.end.type == json::Value::Type::OBJECT ? "true" : "false");
        std::string list = "[";
        for (const auto & c : files) list += (list.size() > 1 ? "," : "") + contents_json(c);
        o.raw("files", list + "]");
        std::string bad = "[";
        for (const auto & [f, e] : errors) bad += (bad.size() > 1 ? "," : "") + json::Object().add("path", f).add("error", e).str();
        o.raw("errors", bad + "]");
        std::cout << o.str() << "\n";
        return errors.empty() ? 0 : 1;
    }
    logging::section("Run " + dir);
    if (!record.found) {
        item("Run record", "none (no mallard.jsonl)");
    } else {
        const json::Value & s = record.start;
        item("Started", text(at(s, {"provenance", "run", "start_time"})) + ", " +
                            plural(at(s, {"provenance", "run", "ranks"}), "rank") + ", Mallard " +
                            text(at(s, {"provenance", "mallard", "version"})));
        if (const json::Value * restart = s.find("restart")) {
            item("", "from " + text(restart->find("file")) + " (step " + text(restart->find("step")) + ")");
        }
        if (record.end.type == json::Value::Type::OBJECT) {
            const json::Value & e = record.end;
            const json::Value * wall = e.find("wall");
            item("Ended", text(e.find("reason")) + " at step " + text(e.find("step")) + ", t = " + text(e.find("t")) +
                              ", " + logging::duration(wall ? wall->number : 0.0) + " wall");
        } else {
            item("Ended", "no end record: still running, or stopped without one (last record at step " +
                              text(record.last.find("step")) + ")");
        }
    }
    logging::section("Output files", count(files.size()));
    for (const auto & c : files) {
        std::string line = std::string(output::kind_name(c.kind));
        if (c.format != line) line += ", " + c.format;
        if (!c.snapshots.empty() && c.kind != output::Kind::mesh) line += ", " + time_text(c);
        line += ", " + count(c.fields.size()) + (c.kind == output::Kind::table ? " columns" : " fields");
        item(fs::relative(c.path, dir).string(), line);
    }
    for (const auto & [f, e] : errors) item(fs::relative(f, dir).string(), logging::style(e, logging::Style::RED));
    return errors.empty() ? 0 : 1;
}

/** @brief The quality summary as JSON: per metric its max, mean, limit, cells above it and worst cells. */
std::string quality_json(const std::array<QualitySummary, MeshQuality::N_METRICS> & summary) {
    json::Object o;
    for (size_t m = 0; m < MeshQuality::N_METRICS; m++) {
        const QualitySummary & s = summary[m];
        std::string worst = "[";
        for (const QualityCell & c : s.worst) {
            std::string x = "[";
            for (size_t i = 0; i < N_DIM; i++) x += (i ? "," : "") + json::number(c.x[i]);
            worst += (worst.size() > 1 ? "," : "") +
                     json::Object().add("value", c.value).add("cell", c.cell).raw("x", x + "]").str();
        }
        o.raw(MeshQuality::FIELDS[m], json::Object()
                                          .add("max", s.max)
                                          .add("mean", s.mean)
                                          .add("limit", MeshQuality::LIMITS[m])
                                          .add("above_limit", s.n_over_limit)
                                          .raw("worst", worst + "]")
                                          .str());
    }
    return o.str();
}

/** @brief mallard info on a mesh file: read once, for its summary and, as a Mesh, its quality. */
int mesh_info(const std::string & path, const cli::Args & args, int argc, char ** argv) {
    if (args.has("--input")) {
        std::cerr << "mallard info: " << path << " is a mesh file, which records no input.\n";
        return 1;
    }
    const std::string quality_file = args.value_or("--quality", "");
    const MeshBlock block = output::read_mesh(path);
    output::Contents c = output::mesh_contents(path, block);
    Kokkos::ScopeGuard kokkos(argc, argv);
    auto mesh = std::make_shared<Mesh>();
    mesh->init_from_block(block);
    mesh->copy_host_to_device();
    c.mesh.n_faces = mesh->n_faces;
    const MeshQuality quality = mesh_quality(*mesh);
    const auto summary = summarize(*mesh, quality, 3);
    std::string written;
    if (!quality_file.empty()) {
        // The writer reports the file as a run's event line, which is not part of this report
        std::ostringstream quiet;
        std::streambuf * out = std::cout.rdbuf(quiet.rdbuf());
        try {
            written = write_quality(mesh, quality, quality_file);
        } catch (...) {
            std::cout.rdbuf(out);
            throw;
        }
        std::cout.rdbuf(out);
    }
    if (args.has("--json")) {
        std::string out = contents_json(c);
        out.pop_back();
        std::cout << out << ",\"quality\":" << quality_json(summary)
                  << ",\"quality_file\":" << (written.empty() ? "null" : json::quote(written)) << "}\n";
        return 0;
    }
    print_contents(c);
    logging::section("Mesh quality");
    print_quality(summary);
    if (!written.empty()) item("Quality fields", written);
    return 0;
}

} // namespace

int info_main(const cli::Args & args) {
    int argc = args.argc;
    char ** argv = args.argv;
    comm::Session session(argc, argv);
    const std::string & path = args.positional[0];
    try {
        std::error_code error;
        const bool mesh_file = output::is_mesh_file(path);
        if (!mesh_file && args.has("--quality")) {
            return cli::usage_error(*cli::command("info"), "--quality takes a mesh file, not " + path);
        }
        if (mesh_file) return mesh_info(path, args, argc, argv);
        if (fs::is_directory(path, error)) return run_directory(path, args);
        output::Contents c = output::open(path);
        if (args.has("--input")) return print_input(parsed_provenance(c), path);
        if (!c.snapshots.empty() && !c.fields.empty()) output::read_ranges(c, c.snapshots.size() - 1);
        if (args.has("--json")) {
            std::cout << contents_json(c) << "\n";
        } else {
            print_contents(c);
        }
    } catch (const std::exception & e) {
        std::cerr << "mallard info: " << e.what() << "\n";
        return 1;
    }
    return 0;
}

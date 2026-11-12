/**
 * @file info_main.cpp
 * @brief `mallard info`: what a mesh, output or restart file, or a run directory, holds.
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 */

#include <algorithm>
#include <array>
#include <cmath>
#include <exception>
#include <filesystem>
#include <fstream>
#include <initializer_list>
#include <iostream>
#include <map>
#include <memory>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include <Kokkos_Core.hpp>

#include "case_view.h"
#include "check.h"
#include "comm.h"
#include "commands.h"
#include "input.h"
#include "json.h"
#include "log.h"
#include "mesh.h"
#include "mesh_quality.h"
#include "output_reader.h"
#include "provenance.h"
#include "region_input.h"
#include "run_page.h"
#include "snapshot_block.h"
#include "diagnostic_code.h"

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

/** @brief The input a run block records: input.resolved (version 2 on), else input.text as written. */
std::string recorded_input(const json::Value & p, bool & resolved) {
    resolved = false;
    if (const json::Value * r = at(p, {"input", "resolved"}); r && r->type == json::Value::Type::STRING) {
        resolved = true;
        return r->string;
    }
    if (const json::Value * t = at(p, {"input", "text"}); t && t->type == json::Value::Type::STRING) return t->string;
    return "";
}

/** @brief A file's run block, parsed; null if it has none. */
json::Value parsed_run(const output::Contents & c) {
    if (c.run.empty()) return {};
    try {
        return json::parse(c.run);
    } catch (const std::exception & e) {
        throw std::runtime_error(c.path + ": the run block is not valid JSON (" + e.what() + ")");
    }
}

/** @brief "file, step N (run <id>)" of a restart_of object; "-" if null. */
std::string restart_of_text(const json::Value * r) {
    if (!r || r->type != json::Value::Type::OBJECT) return "no: a fresh start";
    const json::Value * id = r->find("run_id");
    return text(r->find("file")) + ", step " + text(r->find("step")) +
           (id && !id->is_null() ? " (run " + id->string + ")" : " (a run that recorded no run id)");
}

/** @brief "path, sha256 ..." of an entry of the block's files; a file a run starts from with its step and run. */
std::string read_file_text(const json::Value & f) {
    const json::Value * sha = f.find("sha256");
    std::string out = text(f.find("path"));
    if (const json::Value * step = f.find("step")) {
        const json::Value * id = f.find("run_id");
        out += ", step " + text(step) + (id && !id->is_null() ? " of run " + id->string : "");
    }
    if (sha && !sha->is_null()) return out + ", sha256 " + sha->string;
    return out + ", " + text(f.find("note")) + ", modified " + text(f.find("modified"));
}

void print_run(const json::Value & p) {
    logging::section("Run", "same for every file of the run");
    std::string mallard = text(at(p, {"mallard", "version"}));
    if (const json::Value * sha = at(p, {"mallard", "git_sha"}); sha && !sha->is_null()) {
        const json::Value * dirty = at(p, {"mallard", "git_dirty"});
        mallard += " (commit " + sha->string.substr(0, 12) + (dirty && dirty->boolean ? ", uncommitted changes" : "") + ")";
    }
    item("Mallard", mallard);
    const json::Value * schema = p.find("schema");
    item("Run block", "version " + text(p.find("version")) +
                          (schema && schema->string == provenance::LEGACY_SCHEMA ? ", named \"provenance\"" : ""));
    item("Build", text(at(p, {"build", "dim"})) + "D, " + text(at(p, {"build", "precision"})) + " precision, " +
                      text(at(p, {"build", "build_type"})) + ", " + text(at(p, {"build", "backend"})) + " backend, MPI " +
                      text(at(p, {"build", "mpi"})) + ", HDF5 " + text(at(p, {"build", "hdf5"})));
    if (p.find("run")) {
        const json::Value * hosts = at(p, {"run", "hosts"});
        item("Run id", text(at(p, {"run", "run_id"})));
        if (const json::Value * r = at(p, {"run", "restart_of"})) item("Restarted from", restart_of_text(r));
        item("Run started", text(at(p, {"run", "start_time"})));
        item("Ranks", plural(at(p, {"run", "ranks"}), "rank") + (hosts ? " on " + plural(hosts, "host") : ""));
        item("Device", text(at(p, {"run", "device"})));
        item("Host", text(at(p, {"run", "host"})));
        item("Directory", text(at(p, {"run", "directory"})));
    }
    if (p.find("input")) {
        bool resolved = false;
        recorded_input(p, resolved);
        item("Input file", text(at(p, {"input", "file"})));
        item("Input sha256", text(at(p, {"input", "sha256"})));
        item("Input recorded", std::string(resolved ? "resolved" : "as written") + "; mallard info --input prints it");
    }
    if (const json::Value * mesh = p.find("mesh")) {
        item("Mesh", text(mesh->find("type")));
        if (mesh->find("file")) item("Mesh file", text(mesh->find("file")));
        if (mesh->find("sha256")) item("Mesh sha256", text(mesh->find("sha256")));
    }
    if (const json::Value * files = p.find("files"); files && files->type == json::Value::Type::ARRAY) {
        for (const json::Value & f : files->array) item("Read " + text(f.find("kind")), read_file_text(f));
    }
    if (const json::Value * preset = p.find("preset"); preset && !preset->is_null()) {
        const json::Value * is_default = preset->find("default");
        const bool by_default = is_default && is_default->type == json::Value::Type::BOOL && is_default->boolean;
        item("Preset", text(preset->find("name")) + (by_default ? " (the default)" : ""));
    }
}

std::string seconds(double t) { return format("%.9g", t); }

/** @brief The snapshot block of the snapshot shown (the last), with units. */
void print_snapshot_block(const output::Contents & c) {
    if (c.snapshots.empty() || !c.snapshots.back().has_block) return;
    const output::Snapshot & s = c.snapshots.back();
    const snapshot_block::Block & b = s.block;
    logging::section("This snapshot", c.snapshots.size() > 1 ? "the last" : "");
    item("Step", b.step >= 0 ? std::to_string(b.step) : "-");
    item("Time", seconds(b.time) + " s");
    item("Reason", b.reason.empty() ? "-" : b.reason);
    item("Written at", b.written_at.empty() ? "-" : b.written_at);
    item("Run id", b.run_id.empty() ? "-" : b.run_id);
    if (b.restarted) {
        item("Restarted from", b.restart_of_file + ", step " + std::to_string(b.restart_of_step) +
                                   (b.restart_of_run_id.empty() ? " (a run that recorded no run id)"
                                                                : " (run " + b.restart_of_run_id + ")"));
    }
    item("Format", b.format + (b.format_version >= 0 ? ", version " + std::to_string(b.format_version) : ""));
    item("Solver precision", b.solver_precision.empty() ? "-" : b.solver_precision);
    if (!b.reconstruction.method.empty()) {
        std::string fallback;
        for (const std::string & m : b.reconstruction.fallback) fallback += (fallback.empty() ? "" : ", then ") + m;
        item("Point values", b.reconstruction.method + ", order " + std::to_string(b.reconstruction.order) +
                                 (fallback.empty() ? "" : "; falling back to " + fallback));
    }
    if (b.averaged) {
        item("Averaged over", "t = " + seconds(b.average_start) + " to " + seconds(b.average_end) + " s, " +
                                  (b.samples >= 0 ? count(static_cast<uint64_t>(b.samples), "sample")
                                                   : std::string("- samples")));
    }
    std::vector<std::vector<std::string>> rows;
    for (const snapshot_block::Field & f : b.fields) {
        const snapshot_block::Average & a = f.average;
        if (a.kind.empty()) continue;
        std::string what = a.kind + " of " + list_text(a.of, ", ");
        if (a.kind == "phase") {
            what += ", bin " + std::to_string(a.bin) + " of " + std::to_string(a.bins) +
                    (a.trigger.empty() ? " of the period " + seconds(a.period) + " s" : " of " + a.trigger + "'s cycle");
        }
        rows.push_back({f.name, what, a.weighting.empty() ? "-" : a.weighting,
                        "t = " + seconds(a.start) + " to " + seconds(a.end) + " s",
                        a.samples >= 0 ? count(static_cast<uint64_t>(a.samples)) : "-",
                        a.interval >= 0 ? std::to_string(a.interval) : "-"});
    }
    if (!rows.empty()) {
        logging::line("  Averaged fields: what each averages (Reynolds: unweighted by density), each sample weighted "
                      "by the time since the previous one.");
        logging::table({"field", "average", "weighting", "window", "samples", "every [steps]"}, rows);
    }
}

/** @brief The probe sampling block of a probe file: how its values are read, and where each located sample is. */
void print_probe_sampling(const output::Contents & c) {
    if (c.sampling.empty()) return;
    json::Value b;
    try {
        b = json::parse(c.sampling);
    } catch (const std::exception &) {
        return;
    }
    logging::section("Probe sampling", "where the samples are read");
    const json::Value * r = b.find("reconstruction");
    if (r && r->type == json::Value::Type::OBJECT) {
        std::string fallback;
        for (const json::Value & m : (*r)["fallback"].array) fallback += (fallback.empty() ? "" : ", then ") + m.string;
        item("Sampling", "point: " + (*r)["method"].string + ", order " +
                             std::to_string(static_cast<int>((*r)["order"].number)) +
                             (fallback.empty() ? "" : "; falling back to " + fallback));
    } else {
        item("Sampling", "cell: the value of the cell holding each sample");
    }
    std::string values;
    for (const json::Value & v : b["variables"].array) {
        values += (values.empty() ? "" : ", ") + v["name"].string + " " + v["value"].string;
    }
    item("Values", values.empty() ? "-" : values);
    const json::Value * clipped = b.find("clipped");
    std::string dropped;
    if (clipped && clipped->type == json::Value::Type::OBJECT) {
        dropped = count(static_cast<uint64_t>((*clipped)["dropped"].number)) + " of " +
                  count(static_cast<uint64_t>((*clipped)["of"].number)) + " samples outside the mesh dropped";
        for (const char * m : {"length", "area", "volume"}) {
            if (const json::Value * f = clipped->find(std::string(m) + "_fraction")) {
                dropped += " (" + format("%.3g", 100.0 * f->number) + "% of the " + m + ")";
            }
        }
    }
    const json::Value * points = b.find("points");
    if (!points || points->type != json::Value::Type::OBJECT) {
        item("Locations", "at the faces, cut cells or elements the probe is made of");
        if (!dropped.empty()) item("Clipped", dropped);
        return;
    }
    const json::Value & p = *points;
    const uint64_t n = static_cast<uint64_t>(p["n"].number);
    item("Located samples", count(n) + (dropped.empty() ? "" : "; " + dropped));
    item("Distance", "requested point to where it is read: max " + format("%.6g", p["max_distance"].number) +
                         " m, mean " + format("%.6g", p["mean_distance"].number) + " m");
    auto xyz = [](const json::Value & x) {
        std::string out;
        for (const json::Value & d : x.array) out += (out.empty() ? "" : ", ") + format("%.6g", d.number);
        return "(" + out + ")";
    };
    constexpr uint64_t ROWS = 12;
    std::vector<std::vector<std::string>> rows;
    for (uint64_t i = 0; i < n; i++) {
        if (n > ROWS && i == ROWS / 2) {
            rows.push_back({"...", count(n - ROWS) + " more; --format json lists every one"});
            i = n - ROWS / 2;
        }
        rows.push_back({std::to_string(i), xyz(p["requested"].array[i]), format("%.0f", p["cell"].array[i].number),
                        xyz(p["centroid"].array[i]), format("%.6g", p["distance"].array[i].number)});
    }
    logging::table({"sample", "requested [m]", "cell", "cell centroid [m]", "distance [m]"}, rows);
}

/** @brief "t = 0.002 s, step 2", or the range of several times. */
std::string time_text(const output::Contents & c) {
    const output::Snapshot & first = c.snapshots.front();
    const output::Snapshot & last = c.snapshots.back();
    if (c.snapshots.size() == 1) {
        return "t = " + seconds(last.t) + " s" + (last.step >= 0 ? ", step " + std::to_string(last.step) : "");
    }
    std::string t = "t = " + seconds(first.t) + " to " + seconds(last.t) + " s";
    if (first.step >= 0 && last.step >= 0) {
        t += ", steps " + std::to_string(first.step) + " to " + std::to_string(last.step);
    }
    return t;
}

std::string units_text(const std::string & units) {
    if (units.empty()) return "not recorded";
    return units == "1" ? "dimensionless" : units;
}


/** @brief Explains the "-" of a table's cells that the file does not record. */
void unrecorded_note(const std::vector<std::vector<std::string>> & rows) {
    for (const auto & r : rows) {
        if (std::find(r.begin(), r.end(), "-") != r.end()) {
            logging::line("  -: the file does not record it.");
            return;
        }
    }
}

void print_mesh(const output::Contents & c) {
    using output::Kind;
    const output::MeshSummary & m = c.mesh;
    const bool counts = c.kind != Kind::table && (m.n_cells > 0 || m.n_points > 0);
    if (!counts && c.kind != Kind::table && !m.has_bounds && m.boundary_zones.empty() && m.cell_zones.empty()) return;
    logging::section(c.kind == Kind::table ? "Rows" : c.kind == Kind::probe ? "Probe geometry" : "Mesh");
    if (c.kind == Kind::table) {
        logging::table({"rows per time"}, {{count(m.n_points)}});
    } else if (counts) {
        std::vector<std::string> header, row;
        const auto add = [&](const char * name, uint64_t n) {
            if (n == 0) return;
            header.emplace_back(name);
            row.push_back(count(n));
        };
        add(c.kind == Kind::probe ? "elements" : "cells", m.n_cells);
        add("points", m.n_points);
        add("faces", m.n_faces);
        add("boundary faces", m.n_boundary_faces);
        logging::table(header, {row});
    }
    if (!m.cell_types.empty()) {
        std::vector<std::vector<std::string>> rows;
        for (const auto & [name, n] : m.cell_types) rows.push_back({name, count(n)});
        logging::line();
        logging::table({"cell type", c.kind == Kind::probe ? "elements" : "cells"}, rows);
    }
    if (m.has_bounds) {
        std::vector<std::vector<std::string>> rows;
        for (size_t i = 0; i < size_t(c.dim ? c.dim : 3); i++) {
            rows.push_back({std::string(1, "xyz"[i]), format("%.6g", m.lo[i]), format("%.6g", m.hi[i])});
        }
        logging::line();
        logging::table({"axis", "min [m]", "max [m]"}, rows);
    }
    if (!m.boundary_zones.empty()) {
        std::vector<std::vector<std::string>> rows;
        for (const auto & [name, n] : m.boundary_zones) rows.push_back({name, count(n)});
        logging::line();
        logging::table({"boundary zone", "faces"}, rows);
    }
    if (!m.cell_zones.empty()) {
        std::vector<std::vector<std::string>> rows;
        for (const auto & [name, n] : m.cell_zones) rows.push_back({name, count(n)});
        logging::line();
        logging::table({"cell zone", "cells"}, rows);
    }
}

/** @brief The snapshots of a series, the middle ones elided past this many. */
constexpr size_t SNAPSHOT_ROWS = 12;

void print_snapshots(const output::Contents & c) {
    const size_t n = c.snapshots.size();
    logging::section("Snapshots", count(n, "snapshot"));
    bool steps = false;
    for (const auto & s : c.snapshots) steps = steps || s.step >= 0;
    std::vector<std::vector<std::string>> rows;
    const auto row = [&](size_t i) {
        const output::Snapshot & s = c.snapshots[i];
        std::vector<std::string> r = {std::to_string(i), seconds(s.t)};
        if (steps) r.push_back(s.step >= 0 ? std::to_string(s.step) : "not recorded");
        r.push_back(fs::path(s.file).filename().string() + (s.index ? ", index " + std::to_string(s.index) : ""));
        rows.push_back(std::move(r));
    };
    for (size_t i = 0; i < n; i++) {
        if (n > SNAPSHOT_ROWS && i == SNAPSHOT_ROWS / 2) {
            rows.push_back({"...", count(n - SNAPSHOT_ROWS) + " more; --format json lists every one"});
            i = n - SNAPSHOT_ROWS / 2;
        }
        row(i);
    }
    logging::table(steps ? std::vector<std::string>{"#", "t [s]", "step", "file"} : std::vector<std::string>{"#", "t [s]", "file"}, rows);
}

void print_fields(const output::Contents & c) {
    using output::Kind;
    std::string where;
    if (c.kind == Kind::table) {
        where = "ranges of the last time";
    } else if (!c.snapshots.empty()) {
        where = "ranges at t = " + seconds(c.snapshots.back().t) + " s" + (c.snapshots.size() > 1 ? ", the last" : "");
    }
    logging::section("Fields", where);
    const snapshot_block::Block * block =
        !c.snapshots.empty() && c.snapshots.back().has_block ? &c.snapshots.back().block : nullptr;
    bool vectors = false;
    std::vector<std::vector<std::string>> rows;
    for (const auto & f : c.fields) {
        vectors = vectors || f.components > 1;
        const bool range = !std::isnan(f.min);
        std::vector<std::string> row = {f.name, units_text(f.units), std::to_string(f.components),
                                        range ? format("%.6g", f.min) : "-", range ? format("%.6g", f.max) : "-"};
        if (block) {
            const snapshot_block::Field * stored = block->field(f.name);
            row.push_back(stored && !stored->value.empty() ? stored->value : "-");
            row.push_back(stored && !stored->stored.empty() ? stored->stored : "-");
        }
        row.push_back(f.long_name.empty() ? "-" : f.long_name);
        rows.push_back(std::move(row));
    }
    logging::table(block ? std::vector<std::string>{"field", "units", "components", "min", "max", "value", "stored",
                                                    "description"}
                         : std::vector<std::string>{"field", "units", "components", "min", "max", "description"},
                   rows);
    unrecorded_note(rows);
    if (vectors) logging::line("  min and max of fields with several components are of their magnitude.");
    if (block) {
        logging::line("  value: cell_average, the cell's average; cell_state, computed from the cell's averaged state; "
                      "point_value, the reconstruction's value at the sample.");
    }
}

void print_restart(const output::Contents & c) {
    const RestartHeader & r = c.restart;
    if (!r.attributes.empty()) {
        logging::section("Solver state", "units not recorded");
        std::vector<std::vector<std::string>> rows;
        for (const auto & [name, value] : r.attributes) rows.push_back({name, format("%.9g", value)});
        logging::table({"attribute", "value"}, rows);
    }
    if (!r.arrays.empty() || r.n_faces > 0) {
        logging::section("Restart arrays");
        std::vector<std::vector<std::string>> rows;
        for (const auto & [name, values] : r.arrays) rows.push_back({name, count(values.size())});
        if (r.n_faces > 0) rows.push_back({"characteristic boundary states", count(r.n_faces, "face")});
        logging::table({"array", "values"}, rows);
    }
}

void print_contents(const output::Contents & c) {
    using output::Kind;
    logging::section(c.path);
    item("Kind", output::kind_name(c.kind));
    item("Format", c.format + (c.version >= 0 ? ", version " + std::to_string(c.version) : ""));
    std::vector<std::string> values;
    if (c.dim) values.push_back(std::to_string(c.dim) + "D");
    if (c.real_size) values.push_back(std::string(c.real_size == 8 ? "double" : "single") + " precision");
    if (c.compressed) values.emplace_back("compressed");
    if (!values.empty()) item("Values", list_text(values, ", "));
    if (!c.snapshots.empty() && c.kind != Kind::mesh) {
        std::string times = time_text(c);
        if (c.snapshots.size() > 1) {
            times = count(c.snapshots.size()) + (c.kind == Kind::snapshot ? " snapshots, " : " times, ") + times;
        }
        item("Simulated time", times);
    }
    if (!c.mesh_file.empty()) item("Mesh file", c.mesh_file);
    const std::vector<std::string> species = c.species();
    if (!species.empty()) item("Species", count(species.size()) + ": " + list_text(species, " "));
    for (const auto & note : c.notes) item("Note", note);
    print_mesh(c);
    if (c.kind == Kind::snapshot && c.snapshots.size() > 1) print_snapshots(c);
    if (c.kind == Kind::restart) print_restart(c);
    if (!c.fields.empty()) print_fields(c);
    const json::Value p = parsed_run(c);
    if (!p.is_null()) print_run(p);
    print_snapshot_block(c);
    print_probe_sampling(c);
}

/** @brief A file an output record of the run log lists: as complete, of this size and SHA-256 (empty: not hashed). */
struct RecordedFile {
    uint64_t bytes = 0;
    std::string sha256;
};

/** @brief The last start and end records of a run's mallard.jsonl, its last record, and the files written. */
struct Record {
    json::Value start, end, last;
    std::map<std::string, RecordedFile> written;  ///< by path as recorded, the latest record of each
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
            } else if (type == "output") {
                if (const json::Value * files = v.find("files"); files && files->type == json::Value::Type::ARRAY) {
                    for (const json::Value & f : files->array) {
                        const json::Value * bytes = f.find("bytes");
                        r.written[text(f.find("path"))] = {bytes ? uint64_t(bytes->number) : 0, text(f.find("sha256"))};
                    }
                }
            }
            r.last = std::move(v);
        } catch (const std::exception &) {
            // A last line cut short by a run that was killed
        }
    }
    return r;
}

/** @brief The run block of a run directory: its run record's, else that of its output file of the latest run start. */
json::Value run_block(const std::string & dir, const Record & record) {
    if (const json::Value * p = record.start.find("provenance")) return *p;
    json::Value newest;
    std::string when;
    for (const auto & f : output::run_files(dir)) {
        try {
            const output::Contents c = output::open(f);
            if (c.run.empty()) continue;
            json::Value p = parsed_run(c);
            const std::string t = text(at(p, {"run", "start_time"}));
            if (newest.is_null() || t > when) {
                newest = std::move(p);
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
        logging::error(diagnostic_code::INFO, "mallard info: " + what +
                                                  " has no run block (files of Mallard 0.9 and later have one), so no "
                                                  "input.");
        return 1;
    }
    bool resolved = false;
    const std::string input = recorded_input(block, resolved);
    if (input.empty()) {
        logging::error(diagnostic_code::INFO, "mallard info: the run block of " + what + " records no input.");
        return 1;
    }
    std::cout << input;
    if (!input.ends_with('\n')) std::cout << "\n";
    if (!resolved) {
        std::cerr << "mallard info: run block version " << text(block.find("version"))
                  << " records the input as written, not resolved (defaults, presets and units filled in); the "
                     "Mallard version that wrote it reads it the same way.\n";
    }
    return 0;
}

/** @brief Files the run log records as written that are missing, or differ from their record. */
std::vector<std::pair<std::string, std::string>> check_records(const std::string & dir, const Record & record,
                                                               bool verify) {
    std::vector<std::pair<std::string, std::string>> problems;
    for (const auto & [path, recorded] : record.written) {
        const fs::path file = fs::path(path).is_absolute() ? fs::path(path) : fs::path(dir) / path;
        std::error_code error;
        const auto size = fs::file_size(file, error);
        if (error) {
            problems.emplace_back(path, "missing: removed or moved since the run wrote it");
        } else if (uint64_t(size) != recorded.bytes) {
            problems.emplace_back(path, count(uint64_t(size), "byte") + ", the run wrote " + count(recorded.bytes) +
                                            ": changed or cut short since");
        } else if (verify && !recorded.sha256.empty() && provenance::sha256_file(file.string()) != recorded.sha256) {
            problems.emplace_back(path, "its contents differ from what the run wrote (SHA-256)");
        }
    }
    return problems;
}

/** @brief Writes DIR/report.html from DIR/mallard.jsonl. */
int run_report(const std::string & dir) {
    const fs::path log = fs::path(dir) / "mallard.jsonl";
    if (!fs::exists(log)) {
        logging::error(diagnostic_code::INFO, "mallard info: " + dir +
                                                  " has no mallard.jsonl, which --html builds the run report from "
                                                  "([output] run_log = \"\" writes none; the run's own report.html, "
                                                  "[output] report, does not need it).");
        return 1;
    }
    const std::string page = (fs::path(dir) / "report.html").string();
    run_page::write(page, run_page::from_file(log.string()).html());
    std::cout << "Wrote " << page << "\n";
    return 0;
}

int run_directory(const std::string & dir, const cli::Args & args) {
    if (args.has("--html")) return run_report(dir);
    const Record record = run_record(dir);
    if (args.has("--input")) return print_input(run_block(dir, record), dir);
    std::vector<output::Contents> files;
    std::vector<std::pair<std::string, std::string>> errors;
    for (const auto & f : output::run_files(dir)) {
        try {
            files.push_back(output::open(f));
        } catch (const std::exception & e) {
            errors.emplace_back(f, e.what());
        }
    }
    const auto problems = check_records(dir, record, args.has("--verify"));
    if (args.value_or("--format", "text") == "json") {
        json::Object o;
        o.add("directory", dir);
        o.raw("start", record.start.type == json::Value::Type::OBJECT ? "true" : "false");
        o.raw("end", record.end.type == json::Value::Type::OBJECT ? "true" : "false");
        const json::Value * outcome = record.end.find("outcome");
        if (outcome) {
            o.add("outcome", outcome->string);
        } else {
            o.raw("outcome", "null");
        }
        std::string list = "[";
        for (const auto & c : files) list += (list.size() > 1 ? "," : "") + output::to_json(c);
        o.raw("files", list + "]");
        std::string bad = "[";
        for (const auto & [f, e] : errors) bad += (bad.size() > 1 ? "," : "") + json::Object().add("path", f).add("error", e).str();
        o.raw("errors", bad + "]");
        std::string differ = "[";
        for (const auto & [f, p] : problems) {
            differ += (differ.size() > 1 ? "," : "") + json::Object().add("path", f).add("problem", p).str();
        }
        o.add("recorded_files", static_cast<uint64_t>(record.written.size())).raw("differ_from_record", differ + "]");
        std::cout << o.str() << "\n";
        return errors.empty() && problems.empty() ? 0 : 1;
    }
    logging::section("Run " + dir);
    if (!record.found) {
        item("Run record", "none (no mallard.jsonl)");
    } else {
        const json::Value & s = record.start;
        item("Run record", "mallard.jsonl, the last run in it");
        item("Run id", text(s.find("run_id")));
        if (const json::Value * r = s.find("restart_of")) item("Restarted from", restart_of_text(r));
        item("Started", text(at(s, {"provenance", "run", "start_time"})));
        item("Ranks", plural(at(s, {"provenance", "run", "ranks"}), "rank"));
        item("Mallard", text(at(s, {"provenance", "mallard", "version"})));
        if (const json::Value * restart = s.find("restart"); restart && !s.find("restart_of")) {
            item("Restarted from", text(restart->find("file")) + ", step " + text(restart->find("step")));
        }
    }
    if (record.found) {
        std::cout << "\n" << run_page::from_file((fs::path(dir) / "mallard.jsonl").string()).text(logging::color_stdout());
    }
    logging::section("Output files", count(files.size(), "file"));
    std::vector<std::vector<std::string>> rows;
    for (const auto & c : files) {
        const bool timed = !c.snapshots.empty() && c.kind != output::Kind::mesh;
        const output::Snapshot * last = timed ? &c.snapshots.back() : nullptr;
        rows.push_back({fs::relative(c.path, dir).string(), output::kind_name(c.kind),
                        timed ? count(c.snapshots.size()) : "-", timed ? seconds(c.snapshots.front().t) : "-",
                        last ? seconds(last->t) : "-", last && last->step >= 0 ? std::to_string(last->step) : "-",
                        count(c.fields.size()), c.format});
    }
    logging::table({"file", "kind", "times", "first t [s]", "last t [s]", "last step", "fields", "format"}, rows);
    unrecorded_note(rows);
    if (!errors.empty()) {
        logging::section("Unreadable files", count(errors.size()));
        rows.clear();
        for (const auto & [f, e] : errors) rows.push_back({fs::relative(f, dir).string(), e});
        logging::table({"file", "error"}, rows);
    }
    if (record.found) {
        const std::string checked = args.has("--verify") ? "size and SHA-256" : "size";
        logging::section("Files against the run record",
                         count(record.written.size()) + " recorded as written, " + checked + " checked");
        if (problems.empty()) {
            item("Recorded files", record.written.empty() ? "none" : "all present, as written");
        } else {
            rows.clear();
            for (const auto & [f, p] : problems) rows.push_back({f, p});
            logging::table({"file", "problem"}, rows);
        }
    }
    return errors.empty() && problems.empty() ? 0 : 1;
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

void print_quality_table(const std::array<QualitySummary, MeshQuality::N_METRICS> & summary) {
    constexpr std::array<const char *, MeshQuality::N_METRICS> UNITS = {"deg", "dimensionless", "dimensionless",
                                                                         "dimensionless"};
    logging::section("Mesh quality", "over every cell");
    std::vector<std::vector<std::string>> rows, worst;
    for (size_t m = 0; m < MeshQuality::N_METRICS; m++) {
        const QualitySummary & s = summary[m];
        rows.push_back({MeshQuality::NAMES[m], UNITS[m], format("%.3g", s.max), format("%.3g", s.mean),
                        logging::real(MeshQuality::LIMITS[m]), count(s.n_over_limit)});
        for (const QualityCell & q : s.worst) {
            std::vector<std::string> row = {MeshQuality::NAMES[m], format("%.3g", q.value), count(q.cell)};
            for (size_t i = 0; i < N_DIM; i++) row.push_back(format("%.4g", q.x[i]));
            worst.push_back(std::move(row));
        }
    }
    logging::table({"metric", "units", "max", "mean", "limit", "cells over limit"}, rows);
    if (worst.empty()) return;
    logging::section("Worst cells", "per metric, worst first");
    std::vector<std::string> header = {"metric", "value", "cell"};
    for (size_t i = 0; i < N_DIM; i++) header.push_back(std::string(1, "xyz"[i]) + " [m]");
    logging::table(header, worst);
}

/** @brief mallard info on a mesh file: read once, for its summary and, as a Mesh, its quality. */
int mesh_info(const std::string & path, const cli::Args & args, int argc, char ** argv) {
    if (args.has("--input")) {
        logging::error(diagnostic_code::INFO, "mallard info: " + path + " is a mesh file, which records no input.");
        return 1;
    }
    const std::string quality_file = args.value_or("--mesh-fields", args.value_or("--quality", ""));
    if (args.has("--quality")) {
        logging::warning(diagnostic_code::OPTION_DEPRECATED, "mallard info --quality is deprecated: use --mesh-fields");
    }
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
    std::vector<std::string> view;
    if (const std::string dir = args.value_or("--view", ""); !dir.empty()) {
        if (!mesh->high_order_faces.empty()) mesh->apply_curved(curved::surfaces_of_mesh(*mesh, parse_input_string("")));
        view = case_view::write_view(dir, mesh_view(*mesh, region::MeshRegions(*mesh)).blocks,
                                     std::filesystem::path(path).filename().string());
    }
    if (args.value_or("--format", "text") == "json") {
        std::string out = output::to_json(c);
        out.pop_back();
        std::string files;
        for (const std::string & f : view) files += (files.empty() ? "" : ",") + json::quote(f);
        std::cout << out << ",\"quality\":" << quality_json(summary)
                  << ",\"mesh_fields_file\":" << (written.empty() ? "null" : json::quote(written))
                  << ",\"case_view\":[" << files << "]}\n";
        return 0;
    }
    print_contents(c);
    print_quality_table(summary);
    if (!written.empty()) item("Mesh fields", written);
    for (size_t k = 0; k < view.size(); k++) item(k == 0 ? "Case view" : "Case view image", view[k]);
    return 0;
}

} // namespace

int info_main(const cli::Args & args) {
    int argc = args.argc;
    char ** argv = args.argv;
    comm::Session session(argc, argv);
    const std::string & path = args.positional[0];
    if (const std::string f = args.value_or("--format", "text"); f != "text" && f != "json") {
        return cli::usage_error(*cli::command("info"), "--format takes text or json, not " + f);
    }
    try {
        std::error_code error;
        const bool mesh_file = output::is_mesh_file(path);
        for (const char * option : {"--mesh-fields", "--quality", "--view"}) {
            if (!mesh_file && args.has(option)) {
                return cli::usage_error(*cli::command("info"), std::string(option) + " takes a mesh file, not " + path);
            }
        }
        if (mesh_file) return mesh_info(path, args, argc, argv);
        if (fs::is_directory(path, error)) return run_directory(path, args);
        if (args.has("--html")) {
            return cli::usage_error(*cli::command("info"), "--html takes a run directory, not " + path);
        }
        output::Contents c = output::open(path);
        if (args.has("--input")) return print_input(parsed_run(c), path);
        if (!c.snapshots.empty() && !c.fields.empty()) output::read_ranges(c, c.snapshots.size() - 1);
        if (args.value_or("--format", "text") == "json") {
            std::cout << output::to_json(c) << "\n";
        } else {
            print_contents(c);
        }
    } catch (const std::exception & e) {
        logging::error(diagnostic_code::INFO, std::string("mallard info: ") + e.what());
        return 1;
    }
    return 0;
}

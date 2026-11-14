/**
 * @file pack_main.cpp
 * @brief `mallard pack`: one archive that reproduces a run, with a manifest of
 *        hashes, and `mallard pack --verify` that checks one.
 *
 * What goes in comes from the run record (mallard.jsonl): the last run's start
 * record (its run block: the input, the mesh, the files it read, the resolved
 * input) and its output records.
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 */

#include <algorithm>
#include <ctime>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <set>
#include <string>
#include <vector>

#include "archive.h"
#include "build_info.h"
#include "comm.h"
#include "commands.h"
#include "common_io.h"
#include "input.h"
#include "json.h"
#include "diagnostic_code.h"
#include "log.h"
#include "provenance.h"

namespace fs = std::filesystem;

namespace {

using logging::count;
using logging::item;

constexpr const char * SCHEMA = "mallard-pack";
constexpr int VERSION = 1;
constexpr const char * MANIFEST = "manifest.json";
constexpr const char * RUN_RECORD = "run.json";
constexpr const char * CASE = "case";
constexpr const char * OUTPUT = "output";

std::string text(const json::Value * v) {
    if (!v || v->is_null()) return "";
    if (v->type == json::Value::Type::STRING) return v->string;
    if (v->type == json::Value::Type::NUMBER) return json::number(v->number);
    return "";
}

/** @brief The last run of a run log: its start record (and its line), end record, output records and changes. */
struct Run {
    json::Value start, end;
    std::string start_line;
    std::vector<json::Value> outputs;
    std::vector<std::string> changes;  ///< `mallard ctl set` requests acted on
};

Run last_run(const std::string & log) {
    std::ifstream in(log);
    if (!in) {
        throw std::runtime_error("no run record " + log + ": mallard pack packs a run directory with the run record "
                                 "of Mallard 0.10 and later");
    }
    Run run;
    for (std::string line; std::getline(in, line);) {
        json::Value v;
        try {
            v = json::parse(line);
        } catch (const std::exception &) {
            continue;  // A last line cut short by a run that was killed
        }
        const std::string type = text(v.find("type"));
        if (type == "start") {
            run = Run{};
            run.start = v;
            run.start_line = line;
        } else if (type == "end") {
            run.end = v;
        } else if (type == "output" && text(v.find("run_id")) == text(run.start.find("run_id"))) {
            run.outputs.push_back(v);
        } else if (type == "event" && text(v.find("kind")) == "set") {
            run.changes.push_back("step " + text(v.find("step")) + ": " + text(v.find("text")));
        }
    }
    if (run.start.is_null()) throw std::runtime_error(log + " has no start record");
    return run;
}

/** @brief A file of the archive: from disk (source) or text, with what it is. */
struct Entry {
    std::string name;    ///< below the archive's top directory
    std::string source;  ///< on disk; empty for text
    std::string contents;
    std::string role;    ///< input, mesh, read, resolved_input, run_record, run_log, output
    std::string kind;    ///< of a file read (mechanism, restart, ...) or an output (vtu, restart, ...)
    std::string given;   ///< input, mesh, read: the path as the run gave it
    std::string recorded;  ///< SHA-256 the run record gives it, or ""
    archive::Member packed;
};

/** @brief How many leading ".." a relative path has once normal. */
size_t parents(const fs::path & normal) {
    size_t n = 0;
    for (const auto & part : normal) {
        if (part != "..") break;
        n++;
    }
    return n;
}

std::string json_list(const std::vector<std::string> & items) {
    std::string out = "[";
    for (const auto & s : items) out += (out.size() > 1 ? "," : "") + json::quote(s);
    return out + "]";
}

/**
 * @brief The files the resolved input names as the run's small outputs (forces, integrals, CSV probes, probe
 *        spectra), and the prefixes of its snapshot writers.
 */
std::vector<std::string> small_outputs(const std::string & resolved, std::set<std::string> & snapshot_writers) {
    std::vector<std::string> files;
    const Input input = parse_input_string(resolved, "the resolved input");
    const auto string_of = [](const Input & table, const char * key) {
        return table.is_table() && table.contains(key) && table.at(key).is_string() ? table.at(key).as_string()
                                                                                       : std::string();
    };
    const auto tables = [&](const char * key) {
        std::vector<Input> out;
        if (input.contains(key) && input.at(key).is_array()) {
            for (const Input & t : input.at(key).as_array()) out.push_back(t);
        } else if (input.contains(key) && input.at(key).is_table()) {
            out.push_back(input.at(key));
        }
        return out;
    };
    for (const Input & f : tables("forces")) files.push_back(string_of(f, "file"));
    for (const Input & i : tables("integrals")) files.push_back(string_of(i, "file"));
    for (const Input & p : tables("probes")) {
        if (string_of(p, "format") == "csv") {
            files.push_back(string_of(p, "file"));
            files.push_back(string_of(p, "reductions_file"));
        }
        if (p.contains("spectra")) {
            files.push_back(string_of(p.at("spectra"), "file"));
            files.push_back(string_of(p.at("spectra"), "reductions_file"));
        }
    }
    for (const Input & w : tables("write_data")) snapshot_writers.insert(string_of(w, "prefix"));
    std::erase(files, std::string());
    return files;
}

/** @brief The initialize.type and file of an input's text, or "" (an input this build cannot read). */
std::pair<std::string, std::string> initialize_of(const std::string & toml) {
    try {
        const Input input = parse_input_string(toml);
        if (!input.contains("initialize") || !input.at("initialize").is_table()) return {};
        const Input & init = input.at("initialize");
        const auto get = [&](const char * key) {
            return init.contains(key) && init.at(key).is_string() ? init.at(key).as_string() : std::string();
        };
        return {get("type"), get("file")};
    } catch (const std::exception &) {
        return {};
    }
}

std::string utc_now() {
    char text[32] = {};
    const std::time_t now = std::time(nullptr);
    std::strftime(text, sizeof(text), "%Y-%m-%dT%H:%M:%SZ", std::gmtime(&now));
    return text;
}

std::string archive_stem(const std::string & path) {
    std::string name = fs::path(path).filename().string();
    for (const char * ext : {".tar.gz", ".tgz", ".tar"}) {
        if (name.ends_with(ext)) return name.substr(0, name.size() - std::string(ext).size());
    }
    return name;
}

std::string manifest_json(const Run & run, const std::string & rerun_dir, const std::string & command,
                          const std::string & with, const std::vector<std::string> & notes,
                          const std::vector<Entry> & entries) {
    const json::Value & block = run.start["provenance"];
    json::Object mesh;
    mesh.add("type", text(block["mesh"].find("type")));
    if (block["mesh"].has("file")) mesh.add("file", text(block["mesh"].find("file")));
    json::Object rerun;
    rerun.add("directory", rerun_dir).add("command", command);
    std::string out = "{\n";
    const auto line = [&](const std::string & key, const std::string & value, bool last = false) {
        out += "  " + json::quote(key) + ": " + value + (last ? "\n" : ",\n");
    };
    line("schema", json::quote(SCHEMA));
    line("version", std::to_string(VERSION));
    line("created", json::quote(utc_now()));
    line("mallard", json::quote(mallard_version()));
    line("run_id", json::quote(text(run.start.find("run_id"))));
    line("rerun", rerun.str());
    line("ranks", text(block["run"].find("ranks")));
    line("mesh", mesh.str());
    line("with_outputs", json::quote(with));
    line("notes", json_list(notes));
    out += "  \"files\": [\n";
    for (size_t i = 0; i < entries.size(); i++) {
        const Entry & e = entries[i];
        json::Object f;
        f.add("path", e.packed.name).add("role", e.role);
        if (!e.kind.empty()) f.add("kind", e.kind);
        if (!e.given.empty()) f.add("given", e.given);
        f.add("bytes", e.packed.bytes).add("sha256", e.packed.sha256);
        out += "    " + f.str() + (i + 1 < entries.size() ? ",\n" : "\n");
    }
    out += "  ]\n}\n";
    return out;
}

int pack(const cli::Args & args) {
    const fs::path run_dir = args.positional[0];
    const std::string with = args.value_or("--with-outputs", "small");
    if (with != "small" && with != "all" && with != "none") {
        return cli::usage_error(*cli::command("pack"), "--with-outputs takes small, all or none, not " + with);
    }
    std::error_code error;
    if (!fs::is_directory(run_dir, error)) {
        throw std::runtime_error(run_dir.string() + " is not a run directory (to check an archive: mallard pack "
                                 "--verify " + run_dir.string() + ")");
    }
    const fs::path absolute_dir = fs::weakly_canonical(fs::absolute(run_dir), error);
    const std::string default_name =
        absolute_dir.filename().string() + (archive::have_gzip() ? ".tar.gz" : ".tar");
    const std::string out_path = args.value_or("--output", default_name);
    const Run run = last_run((run_dir / "mallard.jsonl").string());
    const json::Value & block = run.start["provenance"];
    const json::Value * input = block.find("input");
    if (!input || !input->find("text") || !input->find("resolved")) {
        throw std::runtime_error("the run block of " + (run_dir / "mallard.jsonl").string() +
                                 " records no resolved input: runs of Mallard 0.10 and later record it");
    }
    std::vector<std::string> notes;
    std::vector<Entry> cases;

    // What the run read: its input, mesh and other files, by the path it gave them
    std::string input_given = text(input->find("file"));
    if (input_given.empty() || fs::path(input_given).is_absolute()) {
        input_given = input_given.empty() ? "input.toml" : fs::path(input_given).filename().string();
    }
    cases.push_back({.name = "", .source = "", .contents = input->find("text")->string, .role = "input", .kind = "",
                     .given = input_given, .recorded = text(input->find("sha256")), .packed = {}});
    if (text(block["mesh"].find("type")) == "file") {
        cases.push_back({.name = "", .source = "", .contents = "", .role = "mesh", .kind = "",
                         .given = text(block["mesh"].find("file")), .recorded = text(block["mesh"].find("sha256")),
                         .packed = {}});
    }
    if (const json::Value * files = block.find("files")) {
        for (const json::Value & f : files->array) {
            cases.push_back({.name = "", .source = "", .contents = "", .role = "read", .kind = text(f.find("kind")),
                             .given = text(f.find("path")), .recorded = text(f.find("sha256")), .packed = {}});
        }
    }
    size_t depth = 0;
    for (const Entry & e : cases) {
        if (!fs::path(e.given).is_absolute()) depth = std::max(depth, parents(fs::path(e.given).lexically_normal()));
    }
    // The case keeps the layout of the files around the run directory, so the input's relative paths hold
    fs::path tail;
    {
        std::vector<fs::path> parts(absolute_dir.begin(), absolute_dir.end());
        if (depth + 1 > parts.size()) throw std::runtime_error("a file the run read is above the file system's root");
        for (size_t i = parts.size() - depth; i < parts.size(); i++) tail /= parts[i];
    }
    const fs::path rerun_dir = fs::path(CASE) / tail;
    std::vector<Entry> entries;
    std::set<std::string> case_names;
    for (Entry & e : cases) {
        const fs::path given(e.given);
        const bool text_only = e.role == "input";
        if (given.is_absolute()) {
            e.source = text_only ? "" : e.given;
            e.name = (fs::path("absolute") / given.relative_path()).generic_string();
            notes.push_back("The input names " + e.given + " by its absolute path: the rerun reads it there, not "
                            "from the archive (" + e.name + ").");
        } else {
            e.source = text_only ? "" : (run_dir / given).string();
            e.name = (rerun_dir / given).lexically_normal().generic_string();
        }
        if (!case_names.insert(e.name).second) continue;
        if (text_only) {
            entries.push_back(e);
        } else if (fs::is_directory(e.source, error)) {
            // A mesh directory (an OpenFOAM case): its files, the directory as the run's hash covers it
            if (!e.recorded.empty() && provenance::sha256_file(e.source) != e.recorded) {
                throw std::runtime_error("cannot pack " + e.source + ": it differs from the run's record of it "
                                         "(SHA-256): changed since the run read it");
            }
            std::vector<fs::path> files;
            for (const auto & f : fs::recursive_directory_iterator(e.source)) {
                if (f.is_regular_file()) files.push_back(fs::relative(f.path(), e.source));
            }
            std::sort(files.begin(), files.end());
            for (const fs::path & f : files) {
                Entry part = e;
                part.source = (fs::path(e.source) / f).string();
                part.name = (fs::path(e.name) / f).generic_string();
                part.recorded.clear();
                entries.push_back(part);
            }
        } else {
            entries.push_back(e);
        }
    }

    // The run's own record, and its outputs as its output records and resolved input name them
    entries.push_back({.name = RUN_RECORD, .source = "", .contents = run.start_line + "\n", .role = "run_record",
                       .kind = "", .given = "", .recorded = "", .packed = {}});
    entries.push_back({.name = "resolved.toml", .source = "", .contents = input->find("resolved")->string,
                       .role = "resolved_input", .kind = "", .given = "", .recorded = "", .packed = {}});
    std::set<std::string> output_names;
    const auto add_output = [&](const std::string & path, const std::string & role, const std::string & kind,
                                const std::string & recorded) {
        fs::path p(path);
        if (p.is_absolute()) {
            const fs::path relative = p.lexically_relative(absolute_dir);
            if (relative.empty() || parents(relative) > 0) {
                notes.push_back("Not packed: " + path + ", outside the run directory.");
                return;
            }
            p = relative;
        }
        p = p.lexically_normal();
        if (parents(p) > 0) {
            notes.push_back("Not packed: " + path + ", outside the run directory.");
            return;
        }
        const std::string name = (fs::path(OUTPUT) / p).generic_string();
        if (!output_names.insert(name).second || !fs::is_regular_file(run_dir / p, error)) return;
        entries.push_back({.name = name, .source = (run_dir / p).string(), .contents = "", .role = role,
                           .kind = kind, .given = "", .recorded = recorded, .packed = {}});
    };
    add_output("mallard.jsonl", "run_log", "", "");
    std::set<std::string> snapshot_writers;
    const std::vector<std::string> small = small_outputs(input->find("resolved")->string, snapshot_writers);
    if (with != "none") {
        for (const std::string & f : small) add_output(f, "output", "csv", "");
        for (const json::Value & record : run.outputs) {
            const bool snapshot = snapshot_writers.contains(text(record.find("writer")));
            if (snapshot && with != "all") continue;
            for (const json::Value & f : record["files"].array) {
                add_output(text(f.find("path")), "output", text(record.find("kind")), text(f.find("sha256")));
            }
        }
    }

    // How to rerun it: a restart given on the command line is not in the input
    std::string command = (text(block["run"].find("ranks")) == "1" ? "" : "mpirun -n " + text(block["run"].find("ranks")) + " ") +
                          std::string("mallard -i ") + input_given;
    if (const json::Value * restart = run.start.find("restart")) {
        const auto [type, file] = initialize_of(input->find("text")->string);
        if (type != "restart" || file != text(restart->find("file"))) command += " --restart " + text(restart->find("file"));
    }
    for (const std::string & change : run.changes) {
        notes.push_back("Changed while it ran (mallard ctl set), which a rerun does not repeat: " + change + ".");
    }
    if (run.end.is_null()) {
        notes.push_back("The run record has no end record: the run was still running, or was killed.");
    }

    const std::string top = archive_stem(out_path);
    const std::string partial = partial_name(out_path);
    try {
        archive::Writer writer(partial);
        for (Entry & e : entries) {
            const std::string name = top + "/" + e.name;
            e.packed = e.source.empty() ? writer.add_text(name, e.contents, std::time(nullptr))
                                        : writer.add_file(name, e.source);
            e.packed.name = e.name;
            if (!e.recorded.empty() && e.packed.sha256 != e.recorded) {
                throw std::runtime_error("cannot pack " + e.source + ": it differs from the run's record of it "
                                         "(SHA-256): changed since the run " +
                                         (e.role == "output" ? "wrote" : "read") + " it");
            }
        }
        const std::string manifest =
            manifest_json(run, rerun_dir.generic_string(), command, with, notes, entries);
        writer.add_text(top + "/" + MANIFEST, manifest, std::time(nullptr));
        writer.close();
        publish_file("the archive", out_path);
    } catch (...) {
        fs::remove(partial, error);
        throw;
    }

    uint64_t total = 0;
    std::map<std::string, size_t> by_role;
    for (const Entry & e : entries) {
        total += e.packed.bytes;
        by_role[e.role]++;
    }
    logging::section("Packed " + out_path, count(entries.size() + 1, "file"));
    item("Run", text(run.start.find("run_id")));
    item("Case", count(by_role["input"] + by_role["mesh"] + by_role["read"], "file") + ": the input, mesh and files "
                 "the run read");
    item("Outputs", count(by_role["output"] + by_role["run_log"], "file") + " (--with-outputs " + with + ")");
    item("Size", count(total, "byte") + " before compression");
    item("Rerun", "tar -xf " + fs::path(out_path).filename().string() + " && cd " + top + "/" +
                      rerun_dir.generic_string() + " && " + command);
    item("Verify", "mallard pack --verify " + out_path);
    for (const std::string & note : notes) logging::warning(diagnostic_code::PACK_RERUN, note);
    return 0;
}

int verify(const cli::Args & args) {
    const std::string path = args.positional[0];
    const auto depth_one = [](const std::string & name, const char * file) {
        const size_t slash = name.find('/');
        return slash != std::string::npos && name.substr(slash + 1) == file;
    };
    const std::vector<archive::Member> members = archive::read(
        path, [&](const std::string & name) { return depth_one(name, MANIFEST) || depth_one(name, RUN_RECORD); });
    const archive::Member * manifest = nullptr;
    const archive::Member * record = nullptr;
    for (const auto & m : members) {
        if (depth_one(m.name, MANIFEST)) manifest = &m;
    }
    if (!manifest) {
        logging::error(diagnostic_code::PACK,
                       "mallard pack: " + path + " has no " + MANIFEST + ": not an archive of mallard pack");
        return 1;
    }
    const std::string top = manifest->name.substr(0, manifest->name.find('/'));
    for (const auto & m : members) {
        if (m.name == top + "/" + RUN_RECORD) record = &m;
    }
    const json::Value m = json::parse(manifest->contents);
    if (text(m.find("schema")) != SCHEMA) {
        logging::error(diagnostic_code::PACK, "mallard pack: " + path + ": its manifest is not of mallard pack");
        return 1;
    }
    if (const json::Value * v = m.find("version"); !v || v->number > VERSION) {
        logging::error(diagnostic_code::PACK, "mallard pack: " + path + ": manifest version " + text(v) +
                                                  " is newer than this Mallard reads (" + std::to_string(VERSION) + ")");
        return 1;
    }
    std::map<std::string, const archive::Member *> by_name;
    for (const auto & member : members) by_name[member.name] = &member;
    std::vector<std::vector<std::string>> problems;
    std::set<std::string> listed = {manifest->name};
    std::map<std::string, std::pair<std::string, std::string>> given;  // input, mesh, read: path and SHA-256
    for (const json::Value & f : m["files"].array) {
        const std::string name = top + "/" + text(f.find("path"));
        listed.insert(name);
        const auto it = by_name.find(name);
        if (it == by_name.end()) {
            problems.push_back({text(f.find("path")), "missing from the archive"});
            continue;
        }
        if (double(it->second->bytes) != f["bytes"].number) {
            problems.push_back({text(f.find("path")), count(it->second->bytes, "byte") + ", the manifest says " +
                                                          text(f.find("bytes"))});
        } else if (it->second->sha256 != text(f.find("sha256"))) {
            problems.push_back({text(f.find("path")), "its SHA-256 differs from the manifest's"});
        }
        const std::string role = text(f.find("role"));
        if (role == "input" || role == "mesh" || role == "read") {
            const std::string key = role == "read" ? text(f.find("given")) : role;
            // A directory (an OpenFOAM mesh) is several files; the run hashed it as one
            if (given.contains(key)) {
                given[key].second.clear();
            } else {
                given[key] = {text(f.find("path")), it->second->sha256};
            }
        }
    }
    for (const auto & member : members) {
        if (!listed.contains(member.name)) problems.push_back({member.name.substr(top.size() + 1), "not in the manifest"});
    }
    // What the archive holds is what the run read, by the run block's own hashes
    size_t against_run = 0;
    if (record) {
        const json::Value block = json::parse(record->contents)["provenance"];
        const auto compare = [&](const std::string & key, const std::string & recorded) {
            const auto it = given.find(key);
            if (recorded.empty() || it == given.end() || it->second.second.empty()) return;
            against_run++;
            if (it->second.second != recorded) {
                problems.push_back({it->second.first, "differs from what the run read (its run block's SHA-256)"});
            }
        };
        compare("input", text(block["input"].find("sha256")));
        compare("mesh", text(block["mesh"].find("sha256")));
        if (const json::Value * files = block.find("files")) {
            for (const json::Value & f : files->array) compare(text(f.find("path")), text(f.find("sha256")));
        }
    } else {
        problems.push_back({RUN_RECORD, "missing: the run's start record"});
    }

    logging::section("Archive " + path, count(members.size(), "file"));
    item("Run", text(m.find("run_id")));
    item("Packed", text(m.find("created")) + ", by Mallard " + text(m.find("mallard")));
    item("Files", count(m["files"].array.size()) + " in the manifest, size and SHA-256 checked");
    item("Against the run", count(against_run, "file") + " checked against the SHA-256 the run recorded");
    item("Rerun", "cd " + top + "/" + text(m["rerun"].find("directory")) + " && " + text(m["rerun"].find("command")));
    for (const json::Value & note : m["notes"].array) item("Note", note.string);
    if (problems.empty()) {
        item("Result", "every file is as packed");
        return 0;
    }
    logging::section("Problems", count(problems.size()));
    logging::table({"file", "problem"}, problems);
    return 1;
}

} // namespace

int pack_main(const cli::Args & args) {
    int argc = args.argc;
    char ** argv = args.argv;
    if (args.has("--verify") && (args.has("--output") || args.has("--with-outputs"))) {
        return cli::usage_error(*cli::command("pack"), "--verify checks an archive; it takes no -o or --with-outputs");
    }
    comm::Session session(argc, argv);
    try {
        return args.has("--verify") ? verify(args) : pack(args);
    } catch (const std::exception & e) {
        logging::error(diagnostic_code::PACK, std::string("mallard pack: ") + e.what());
        return 1;
    }
}

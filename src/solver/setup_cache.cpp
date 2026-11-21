/**
 * @file setup_cache.cpp
 * @brief mallard session's stages of earlier setups: their records, and the solver and mesh they keep.
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 */

#include "setup_cache.h"

#include <algorithm>
#include <cstdio>
#include <filesystem>
#include <system_error>

#include "input_upgrade.h"
#include "mesh_block.h"
#include "numerics_preset.h"
#include "provenance.h"
#include "solver.h"

namespace {

using Path = StageRecord::Path;

/** @brief Calls f(path, table) for every table of v, v included, in a fixed order. */
template <typename F>
void each_table(const Input & v, Path & path, F && f) {
    if (v.is_table()) {
        f(path, v);
        for (const auto & [key, entry] : v.as_table().unobserved()) {
            path.push_back({.key = key, .index = 0, .element = false});
            each_table(entry, path, f);
            path.pop_back();
        }
    } else if (v.is_array()) {
        const auto & array = v.as_array();
        for (size_t i = 0; i < array.size(); i++) {
            path.push_back({.key = "", .index = i, .element = true});
            each_table(array[i], path, f);
            path.pop_back();
        }
    }
}

template <typename F>
void each_table(const Input & root, F && f) {
    Path path;
    each_table(root, path, f);
}

/** @brief The value at path, or nullptr. */
const Input * at(const Input & root, const Path & path) {
    const Input * v = &root;
    for (const StageRecord::Step & step : path) {
        if (step.element) {
            if (!v->is_array() || step.index >= v->as_array().size()) return nullptr;
            v = &v->as_array()[step.index];
        } else {
            if (!v->is_table()) return nullptr;
            const auto & entries = v->as_table().unobserved();
            const auto it = entries.find(step.key);
            if (it == entries.end()) return nullptr;
            v = &it->second;
        }
    }
    return v;
}

std::string sized(const std::string & text) { return std::to_string(text.size()) + ":" + text; }

/** @brief A value that is neither a table nor an array: its type and value, exactly. */
std::string scalar(const Input & v) {
    switch (v.type()) {
        case toml::value_t::boolean: return v.as_boolean() ? "b1" : "b0";
        case toml::value_t::integer: return "i" + std::to_string(v.as_integer());
        case toml::value_t::floating: {
            char text[64];
            std::snprintf(text, sizeof text, "%a", static_cast<double>(v.as_floating()));
            return std::string("f") + text;
        }
        case toml::value_t::string: return "s" + sized(v.as_string());
        default: return "o" + sized(toml::format(v));
    }
}

/**
 * @brief What a lookup sees of v, with its line (messages cite it): a table's keys, not their values, which its own
 *        lookups record; whole, the values too.
 */
std::string fingerprint(const Input & v, const bool whole) {
    const std::string line = "@" + std::to_string(line_of(v));
    if (v.is_table()) {
        std::string text = "T" + line + "{";
        for (const auto & [key, entry] : v.as_table().unobserved()) {
            text += sized(key);
            if (whole) text += "=" + fingerprint(entry, true) + ";";
        }
        return text + "}";
    }
    if (v.is_array()) {
        std::string text = "A" + line + "[";
        for (const Input & element : v.as_array()) text += fingerprint(element, whole) + ";";
        return text + "]";
    }
    return scalar(v) + line;
}

bool same(const Path & a, const Path & b) {
    return std::equal(a.begin(), a.end(), b.begin(), b.end(), [](const StageRecord::Step & x, const StageRecord::Step & y) {
        return x.element == y.element && x.key == y.key && x.index == y.index;
    });
}

std::string seen(const Input & table, const std::string & key) {
    const auto & entries = table.as_table().unobserved();
    const auto it = entries.find(key);
    return it == entries.end() ? "-" : fingerprint(it->second, false);
}

/** @brief A file's size and modification time, or empty if either is unknown. */
std::string stamp(const std::string & path) {
    namespace fs = std::filesystem;
    std::error_code e;
    const bool directory = fs::is_directory(path, e);
    if (e) return "";
    const uintmax_t bytes = directory ? 0 : fs::file_size(path, e);
    if (e) return "";
    const auto time = fs::last_write_time(path, e);
    if (e) return "";
    return std::to_string(bytes) + "@" + std::to_string(time.time_since_epoch().count());
}

bool lookups_empty(const StageRecord::Lookups & l) {
    return l.keys.empty() && l.uses.empty() && l.optional.empty() && !l.all && !l.inspected;
}

} // namespace

bool StageRecord::valid(const Input & input) const {
    for (const Table & t : tables) {
        const Input * table = at(input, t.path);
        if (!table || !table->is_table()) return false;
        for (const auto & [key, value] : t.seen) {
            if (seen(*table, key) != value) return false;
        }
        if (!t.whole.empty() && fingerprint(*table, true) != t.whole) return false;
    }
    for (const File & f : files) {
        if (f.stamp.empty() || stamp(f.path) != f.stamp) return false;
    }
    return true;
}

void StageRecord::replay(const Input & input) const {
    for (const Table & t : tables) {
        if (const Input * table = at(input, t.path)) table->as_table().merge_lookups(t.lookups);
    }
    for (const auto & [code, message] : warnings) logging::warning(code, message);
}

SetupCache::SetupCache(const MeshFileCache * mesh_files_in) : mesh_files(mesh_files_in) {}

SetupCache::~SetupCache() = default;

Solver & SetupCache::solver(const Input & input) {
    resuming = false;
    const auto valid = [&] {
        for (const auto & record : kept.records) {
            if (!record) return false;
        }
        Input prepared = unrecorded(input);
        try {
            input_upgrade::apply(prepared);
            numerics_preset::apply(prepared);
        } catch (const std::exception &) {
            return false;
        }
        for (const auto & record : kept.records) {
            if (!record->valid(prepared)) return false;
        }
        return true;
    };
    if (current && valid()) {
        resuming = true;
        // Parts of the setup may point into the input it read
        if (!setup_input) setup_input = std::make_unique<Input>(std::move(current->input));
        return *current;
    }
    current.reset();
    setup_input.reset();
    kept.records[PHYSICS].reset();
    kept.records[REST].reset();
    made = Kept();
    current = std::make_unique<Solver>();
    current->setup_cache = this;
    return *current;
}

void SetupCache::checked(const bool setup_done, const bool stepped) {
    // A reused setup stays as it was, unless the check took steps: only [run] failed if it did not complete
    if (resuming) {
        if (stepped) {
            current.reset();
            setup_input.reset();
        }
        return;
    }
    const bool keep = setup_done && !stepped;
    if (made.records[MESH] && made.mesh) {
        kept.mesh = std::move(made.mesh);
        kept.records[MESH] = std::move(made.records[MESH]);
    }
    if (keep && made.records[PHYSICS] && made.records[REST] && kept.records[MESH]) {
        kept.records[PHYSICS] = std::move(made.records[PHYSICS]);
        kept.records[REST] = std::move(made.records[REST]);
        kept.t = made.t;
        kept.step = made.step;
        kept.t_wall_setup = made.t_wall_setup;
        kept.setup_phases = std::move(made.setup_phases);
        kept.memory = made.memory;
    } else {
        current.reset();
    }
    made = Kept();
}

void SetupCache::clear() {
    current.reset();
    setup_input.reset();
    kept = Kept();
    made = Kept();
    resuming = false;
}

SetupStages::SetupStages(SetupCache * cache_in, Solver & solver_in)
    : cache(solver_in.is_distributed() ? nullptr : cache_in), solver(solver_in) {}

SetupStages::~SetupStages() {
    if (open) finish();
}

bool SetupStages::resume() {
    if (!cache) return false;
    if (!cache->resuming) {
        begin(SetupCache::MESH);
        return false;
    }
    const SetupCache::Kept & k = cache->kept;
    // With an error in [run], the setup stops after the physics
    const size_t last = solver.setup_errors.empty() ? SetupCache::REST : SetupCache::PHYSICS;
    for (size_t s = SetupCache::MESH; s <= last; s++) k.records[s]->replay(solver.input);
    solver.t = k.t;
    solver.step = k.step;
    solver.t_wall_setup = k.t_wall_setup;
    solver.run_health.setup_phases = k.setup_phases;
    memory::resume(k.memory);
    return true;
}

bool SetupStages::restore_mesh() {
    if (!cache) return false;
    const SetupCache::Kept & k = cache->kept;
    if (!k.mesh || !k.records[SetupCache::MESH] || !k.records[SetupCache::MESH]->valid(solver.input)) return false;
    k.records[SetupCache::MESH]->replay(solver.input);
    const MeshStage & m = *k.mesh;
    solver.mesh = m.mesh;
    solver.mesh_setup = m.mesh_setup;
    solver.n_cells_global = m.n_cells_global;
    solver.axisymmetric = m.axisymmetric;
    solver.partitioner = m.partitioner;
    solver.mesh_provenance = m.provenance;
    provenance::set_mesh(m.provenance[0], m.provenance[1], m.provenance[2]);
    memory::resume(m.memory);
    restored = true;
    return true;
}

void SetupStages::begin(const SetupCache::Stage stage) {
    open = stage;
    set_aside.clear();
    each_table(solver.input, [&](const Path & path, const Input & table) {
        set_aside.emplace_back(path, table.as_table().set_aside_lookups());
    });
    content = fingerprint(solver.input, true);
    files_before = provenance::read_paths().size();
    mesh_files_before = cache->mesh_files ? cache->mesh_files->files_asked().size() : 0;
    errors_before = solver.setup_errors.size();
    warnings = std::make_unique<logging::WarningCapture>();
}

StageRecord SetupStages::finish() {
    StageRecord record;
    each_table(solver.input, [&](const Path & path, const Input & table) {
        StageRecord::Lookups made = table.as_table().lookups_made();
        for (const auto & [p, aside] : set_aside) {
            if (!same(p, path)) continue;
            std::erase_if(made.optional, [&](const std::string & k) { return aside.optional.contains(k); });
            table.as_table().restore_lookups(aside);
            break;
        }
        if (lookups_empty(made)) return;
        StageRecord::Table t{.path = path, .lookups = made, .seen = {}, .whole = ""};
        for (const std::string & key : made.keys) t.seen.emplace_back(key, seen(table, key));
        if (made.all || made.inspected) t.whole = fingerprint(table, true);
        record.tables.push_back(std::move(t));
    });
    set_aside.clear();
    std::vector<std::string> paths = provenance::read_paths();
    if (cache->mesh_files) {
        const std::vector<std::string> & asked = cache->mesh_files->files_asked();
        paths.insert(paths.end(), asked.begin() + static_cast<std::ptrdiff_t>(mesh_files_before), asked.end());
    }
    for (size_t i = files_before; i < paths.size(); i++) record.files.push_back({paths[i], stamp(paths[i])});
    std::unique_ptr<logging::WarningCapture> capture = std::move(warnings);
    record.warnings = std::move(capture->warnings);
    capture.reset();
    for (const auto & [code, message] : record.warnings) logging::warning(code, message);
    open.reset();
    return record;
}

void SetupStages::end(const SetupCache::Stage stage) {
    if (!cache || open != stage) return;
    const std::string before = content;
    StageRecord record = finish();
    // A stage that failed, or changed the input, is not kept
    const bool keep = solver.setup_errors.size() == errors_before && fingerprint(solver.input, true) == before;
    SetupCache::Kept & made = cache->made;
    if (keep) made.records[stage] = std::move(record);
    if (keep && stage == SetupCache::MESH) {
        made.mesh = MeshStage{.mesh = solver.mesh,
                              .mesh_setup = solver.mesh_setup,
                              .n_cells_global = solver.n_cells_global,
                              .axisymmetric = solver.axisymmetric,
                              .partitioner = solver.partitioner,
                              .provenance = solver.mesh_provenance,
                              .memory = memory::counted()};
        if (!restored) cache->n_meshes++;
    }
    if (stage == SetupCache::PHYSICS) cache->n_setups++;
    if (keep && stage == SetupCache::REST) {
        made.t = solver.t;
        made.step = solver.step;
        made.t_wall_setup = solver.t_wall_setup;
        made.setup_phases = solver.run_health.setup_phases;
        made.memory = memory::counted();
    }
    if (stage != SetupCache::REST) begin(static_cast<SetupCache::Stage>(stage + 1));
}

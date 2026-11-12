/**
 * @file convergence_main.cpp
 * @brief `mallard post convergence`: the observed order of accuracy, the Richardson-extrapolated value and the
 *        grid convergence index of quantities of runs at several resolutions, and error norms of a field
 *        against an exact solution or the finest run. Runs are told apart by what they record (run ids, cell
 *        counts), never by their files' names.
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 */

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include <Kokkos_Core.hpp>

#include "cli.h"
#include "commands.h"
#include "convergence.h"
#include "expression.h"
#include "input.h"
#include "json.h"
#include "log.h"
#include "mesh.h"
#include "output_reader.h"
#include "overlap.h"
#include "parallel/comm.h"
#include "provenance.h"
#include "remap.h"
#include "report.h"
#include "diagnostic_code.h"

namespace fs = std::filesystem;

namespace {

using convergence::Behavior;
using logging::count;
using logging::format;
using report::Status;

const char * const NAME = "mallard post convergence";

/** @brief Quadrature degree of the cell averages of --exact: errors of order h^7, below a sixth-order scheme's. */
constexpr uint32_t EXACT_DEGREE = 6;

/** @brief Degree of the finest run's polynomials remapped onto the coarser runs without --exact: errors of order
 *         h_fine^6, below a sixth-order scheme's, and none where the coarser cells are unions of finer ones. */
constexpr uint8_t REFERENCE_DEGREE = remap::MAX_DEGREE;

/** @brief A quantity at most this fraction of its table's largest column on every run is zero to round-off. */
constexpr double ZERO = 1e-12;

/** @brief Round-off of runs in single precision: this many machine epsilons, where it exceeds the double thresholds. */
constexpr double SINGLE_EPSILONS = 100.0;

/** @brief Observed orders further than this from the design order are a warning. */
constexpr double ORDER_SLACK = 1.0;

/** @brief Orders of successive triplets differing by more than this fraction: not yet in the asymptotic range. */
constexpr double ORDER_DRIFT = 0.2;

struct UsageError : std::runtime_error {
    using std::runtime_error::runtime_error;
};

std::vector<std::string> split_list(const std::string & text) {
    std::vector<std::string> out;
    std::stringstream in(text);
    std::string item;
    while (std::getline(in, item, ',')) {
        const size_t a = item.find_first_not_of(" \t"), b = item.find_last_not_of(" \t");
        if (a != std::string::npos) out.push_back(item.substr(a, b - a + 1));
    }
    return out;
}

double parse_number(const std::string & text, const std::string & option) {
    try {
        size_t used = 0;
        const double x = std::stod(text, &used);
        if (used == text.size() && std::isfinite(x)) return x;
    } catch (const std::exception &) {
    }
    throw UsageError(option + " takes numbers, not " + text);
}

std::string number(const double x) { return std::isfinite(x) ? format("%.8g", x) : "-"; }
std::string order_text(const double p) { return std::isfinite(p) ? format("%.2f", p) : "-"; }
std::string short_number(const double x) { return format("%g", x); }
std::string percent(const double x) { return std::isfinite(x) ? format("%.3g%%", 100.0 * x) : "-"; }

bool same_time(const double a, const double b, const double tolerance) {
    return a == b || std::abs(a - b) <= tolerance * std::max(std::abs(a), std::abs(b));
}

const json::Value * at(const json::Value & v, std::initializer_list<const char *> path) {
    const json::Value * p = &v;
    for (const char * key : path) {
        p = p->find(key);
        if (!p) return nullptr;
    }
    return p;
}

std::string string_at(const json::Value & v, std::initializer_list<const char *> path) {
    const json::Value * p = at(v, path);
    return p && p->type == json::Value::Type::STRING ? p->string : "";
}

double number_at(const json::Value & v, std::initializer_list<const char *> path) {
    const json::Value * p = at(v, path);
    return p && p->type == json::Value::Type::NUMBER ? p->number : convergence::NaN;
}

// ---------------------------------------------------------------------------
// Runs

/** @brief A snapshot a run wrote, as its run record lists it. */
struct Written {
    std::string path;
    std::string kind;
    double t = 0.0;
};

/** @brief One resolution: a run directory, or a snapshot file. */
struct Run {
    std::string arg;
    std::string dir;                   ///< the run directory; empty for a file
    std::set<std::string> run_ids;     ///< of every part of the run (restarts continue it)
    std::string run_id;                ///< of its latest part
    json::Value block;                 ///< the run block
    int dim = 0;
    uint64_t cells = 0;
    double h = 0.0;
    std::vector<Written> written;
};

std::string resolve(const Run & run, const std::string & path) {
    if (path.empty() || fs::path(path).is_absolute()) return path;
    std::error_code error;
    if (!run.dir.empty() && fs::exists(fs::path(run.dir) / path, error)) return (fs::path(run.dir) / path).string();
    const std::string directory = string_at(run.block, {"run", "directory"});
    if (!directory.empty() && fs::exists(fs::path(directory) / path, error)) {
        return (fs::path(directory) / path).string();
    }
    return run.dir.empty() ? path : (fs::path(run.dir) / path).string();
}

Run open_directory(const std::string & dir) {
    Run run;
    run.arg = dir;
    run.dir = dir;
    std::ifstream in(fs::path(dir) / "mallard.jsonl");
    if (!in.good()) {
        throw std::runtime_error(dir + " has no run record (mallard.jsonl): give the directory a run ran in, with "
                                 "[output] run_log at its default");
    }
    std::string line;
    while (std::getline(in, line)) {
        json::Value v;
        try {
            v = json::parse(line);
        } catch (const std::exception &) {
            continue;  // A last line cut short by a run that was killed
        }
        const std::string type = string_at(v, {"type"});
        if (type == "start") {
            run.run_id = string_at(v, {"run_id"});
            if (!run.run_id.empty()) run.run_ids.insert(run.run_id);
            if (const json::Value * p = v.find("provenance")) run.block = *p;
        } else if (type == "progress") {
            const double cells = number_at(v, {"cells"});
            if (cells > 0.0) run.cells = static_cast<uint64_t>(cells);
        } else if (type == "output") {
            run.written.push_back({string_at(v, {"path"}), string_at(v, {"kind"}), number_at(v, {"t"})});
        }
    }
    if (run.run_ids.empty()) throw std::runtime_error(dir + "/mallard.jsonl has no start record of a run");
    if (run.cells == 0) {
        for (auto it = run.written.rbegin(); it != run.written.rend() && run.cells == 0; ++it) {
            if (it->kind == "vtp" || it->kind == "spectra") continue;
            try {
                run.cells = output::open(resolve(run, it->path)).mesh.n_cells;
            } catch (const std::exception &) {
            }
        }
    }
    if (run.cells == 0) {
        throw std::runtime_error(dir + ": its run record has no cell count (no progress row) and no snapshot of the "
                                 "mesh: run it until it reports progress once");
    }
    run.dim = static_cast<int>(number_at(run.block, {"build", "dim"}));
    return run;
}

Run open_file(const std::string & path) {
    const output::Contents c = output::open(path);
    if (c.kind != output::Kind::snapshot && c.kind != output::Kind::restart) {
        throw UsageError(path + " is a " + std::string(output::kind_name(c.kind)) +
                         " file: give run directories, or snapshots and restart files of the whole mesh");
    }
    Run run;
    run.arg = path;
    run.cells = c.mesh.n_cells;
    run.dim = c.dim;
    if (!c.run.empty()) run.block = json::parse(c.run);
    if (run.dim == 0) run.dim = static_cast<int>(number_at(run.block, {"build", "dim"}));
    for (const output::Snapshot & s : c.snapshots) {
        if (!s.run_id.empty()) run.run_ids.insert(s.run_id);
    }
    run.run_id = c.snapshots.empty() ? "" : c.snapshots.back().run_id;
    run.written.push_back({path, "file", c.snapshots.empty() ? 0.0 : c.snapshots.back().t});
    return run;
}

/** @brief The resolved input of a run's block (its input as written, for blocks before 0.9.1). */
std::string recorded_input(const Run & run) {
    std::string text = string_at(run.block, {"input", "resolved"});
    return text.empty() ? string_at(run.block, {"input", "text"}) : text;
}

/** @brief The order of accuracy the run's reconstruction is designed for; NaN where its input is not recorded. */
double design_order(const Run & run) {
    const std::string text = recorded_input(run);
    if (text.empty()) return convergence::NaN;
    const Input input = parse_input_string(text, run.arg + " (its run block)");
    std::string type = "FO";
    int order = 5;
    if (input.contains("numerics") && input.at("numerics").is_table() &&
        input.at("numerics").contains("face_reconstruction")) {
        const Input & f = input.at("numerics").at("face_reconstruction");
        type = find_value_or<std::string>(f, "type", "FO");
        order = find_value_or<int>(f, "order", 5);
    }
    if (type == "FO") return 1.0;
    if (type == "MUSCL") return 2.0;
    return order;
}

/** @brief The text of each table of a TOML input, by its header ("" for the keys before the first). */
std::map<std::string, std::string> sections(const std::string & text) {
    std::map<std::string, std::string> out;
    std::istringstream in(text);
    std::string line, header;
    std::map<std::string, int> repeats;
    while (std::getline(in, line)) {
        const size_t a = line.find_first_not_of(" \t");
        if (a != std::string::npos && line[a] == '[' && line.find('=') == std::string::npos) {
            const bool array = line.compare(a, 2, "[[") == 0;
            header = line.substr(a, line.find(array ? "]]" : "]", a) - a + (array ? 2 : 1));
            if (array) header += "#" + std::to_string(repeats[header]++);
            continue;
        }
        if (a == std::string::npos || line[a] == '#') continue;
        // Where a run writes its files is not part of the case
        const std::string key = line.substr(a, line.find_first_of(" =", a) - a);
        if (key == "file" || key == "reductions_file" || key == "prefix") continue;
        out[header] += line + "\n";
    }
    return out;
}

/** @brief The tables of the inputs, other than those that change with the resolution, where the runs differ. */
std::vector<std::string> differing_tables(const std::vector<Run> & runs) {
    const auto varies = [](const std::string & header) {
        for (const char * t : {"[mesh", "[run]", "[output]", "[[write_data]]", "[parallel]"}) {
            if (header.starts_with(t)) return true;
        }
        return false;
    };
    std::vector<std::map<std::string, std::string>> all;
    for (const Run & r : runs) {
        const std::string text = recorded_input(r);
        if (text.empty()) return {};
        all.push_back(sections(text));
    }
    std::set<std::string> headers;
    for (const auto & s : all) {
        for (const auto & [h, body] : s) headers.insert(h);
    }
    std::vector<std::string> out;
    for (const std::string & h : headers) {
        if (varies(h)) continue;
        for (const auto & s : all) {
            const auto it = s.find(h);
            const auto first = all[0].find(h);
            const bool same = (it == s.end()) == (first == all[0].end()) && (it == s.end() || it->second == first->second);
            if (!same) {
                std::string name = h.substr(0, h.find('#'));
                if (name.empty()) name = "the top-level keys";
                if (std::find(out.begin(), out.end(), name) == out.end()) out.push_back(name);
                break;
            }
        }
    }
    return out;
}

// ---------------------------------------------------------------------------
// Scalar quantities: the columns of the runs' tables

/** @brief How the value of a quantity is taken from its rows. */
struct Selection {
    enum class Kind { final, time, average } kind = Kind::final;
    double t = 0.0, t0 = 0.0, t1 = 0.0;
    double same = 1e-9;     ///< times this fraction apart are one time
    double window = 1e-12;  ///< rows this fraction outside the window of an average are in it
};

struct Value {
    double v = convergence::NaN;
    double scale = 0.0;           ///< the largest magnitude of its table's columns at that row
    double t = convergence::NaN;  ///< of the row taken (the end of the window of an average)
    std::string units;
    bool extracted = false;       ///< by mallard post extract, from the run's snapshots
};

std::vector<std::string> split_csv(const std::string & line) {
    std::vector<std::string> out;
    std::string cell;
    std::istringstream in(line);
    while (std::getline(in, cell, ',')) out.push_back(cell);
    return out;
}

/** @brief A snapshot a run read by its identity: the run that wrote it, its step and time. */
struct Source {
    std::string run_id;
    int64_t step = -1;
    double t = convergence::NaN;
};

/** @brief The snapshots each run that wrote a CSV table extracted its rows from (post extract), by run id. */
std::map<std::string, std::vector<Source>> table_sources(const std::string & path) {
    std::map<std::string, std::vector<Source>> out;
    std::ifstream in(path);
    for (std::string line; std::getline(in, line);) {
        if (line.empty() || line[0] != '#') continue;
        const std::string block = provenance::parse_comment(line);
        if (block.empty()) continue;
        json::Value v;
        try {
            v = json::parse(block);
        } catch (const std::exception &) {
            continue;
        }
        const json::Value * files = v.find("files");
        if (!files || files->type != json::Value::Type::ARRAY) continue;
        std::vector<Source> & sources = out[string_at(v, {"run", "run_id"})];
        for (const json::Value & f : files->array) {
            if (string_at(f, {"kind"}) != "snapshot") continue;
            const double step = number_at(f, {"step"});
            sources.push_back({string_at(f, {"run_id"}), std::isfinite(step) ? static_cast<int64_t>(step) : -1,
                               number_at(f, {"t"})});
        }
    }
    return out;
}

/**
 * @brief The quantities of one table of a run: label:column -> value. Only rows of the run's own parts count,
 *        and rows post extract wrote from the snapshots of those parts.
 */
void table_values(const Run & run, const output::Contents & c, const std::string & label, const Selection & sel,
                  std::map<std::string, Value> & out) {
    size_t t_column = c.fields.size();
    for (size_t i = 0; i < c.fields.size(); i++) {
        if (c.fields[i].name == "t") t_column = i;
    }
    if (t_column == c.fields.size() || c.snapshots.empty() || c.mesh.n_points != 1) return;
    const std::map<std::string, std::vector<Source>> sources = table_sources(c.path);
    bool extracted = false;
    const auto of_run = [&](const output::Snapshot & s) {
        if (run.run_ids.contains(s.run_id)) return true;
        const auto it = sources.find(s.run_id);
        if (it == sources.end()) return false;
        for (const Source & source : it->second) {
            if (run.run_ids.contains(source.run_id) && source.step == s.step && same_time(source.t, s.t, sel.same)) {
                extracted = true;
                return true;
            }
        }
        return false;
    };
    std::vector<size_t> rows;  // rows of the run, in time order: a rewound run's rows replace those they repeat
    std::vector<double> times;
    for (const output::Snapshot & s : c.snapshots) {
        if (!of_run(s)) continue;
        while (!times.empty() && times.back() >= s.t) {
            times.pop_back();
            rows.pop_back();
        }
        rows.push_back(s.index);
        times.push_back(s.t);
    }
    if (rows.empty()) return;
    std::vector<size_t> chosen;
    if (sel.kind == Selection::Kind::final) {
        chosen = {rows.size() - 1};
    } else if (sel.kind == Selection::Kind::time) {
        for (size_t k = 0; k < rows.size(); k++) {
            if (same_time(times[k], sel.t, sel.same)) chosen = {k};
        }
    } else {
        for (size_t k = 0; k < rows.size(); k++) {
            if (times[k] >= sel.t0 - sel.window * std::abs(sel.t0) && times[k] <= sel.t1 + sel.window * std::abs(sel.t1)) {
                chosen.push_back(k);
            }
        }
        if (chosen.size() < 2) return;
    }
    if (chosen.empty()) return;

    std::ifstream in(c.path);
    std::vector<std::vector<double>> data;
    std::string line;
    bool header = false;
    while (std::getline(in, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (line.empty() || line[0] == '#') continue;
        if (!header) {
            header = true;
            continue;
        }
        std::vector<double> row;
        for (const std::string & cell : split_csv(line)) {
            char * end = nullptr;
            const double v = std::strtod(cell.c_str(), &end);
            row.push_back(end != cell.c_str() ? v : convergence::NaN);
        }
        data.push_back(std::move(row));
    }
    const auto cell_of = [&](size_t k, size_t col) {
        const size_t r = rows[k];
        return r < data.size() && col < data[r].size() ? data[r][col] : convergence::NaN;
    };
    double scale = 0.0;
    for (size_t col = 0; col < c.fields.size(); col++) {
        if (col == t_column || c.fields[col].name == "step") continue;
        for (const size_t k : chosen) {
            if (std::isfinite(cell_of(k, col))) scale = std::max(scale, std::abs(cell_of(k, col)));
        }
    }
    for (size_t col = 0; col < c.fields.size(); col++) {
        const std::string & name = c.fields[col].name;
        if (name == "t" || name == "step") continue;
        const auto cell = [&](size_t k) { return cell_of(k, col); };
        Value value;
        value.units = c.fields[col].units;
        value.scale = scale;
        if (sel.kind == Selection::Kind::average) {
            double sum = 0.0;
            for (size_t i = 1; i < chosen.size(); i++) {
                sum += 0.5 * (cell(chosen[i]) + cell(chosen[i - 1])) * (times[chosen[i]] - times[chosen[i - 1]]);
            }
            value.v = sum / (times[chosen.back()] - times[chosen.front()]);
        } else {
            value.v = cell(chosen[0]);
        }
        value.t = times[chosen.back()];
        value.extracted = extracted;
        out[label + ":" + name] = value;
    }
}

/** @brief Every scalar quantity of a run directory's tables: forces, integrals, probe reductions, point probes. */
std::map<std::string, Value> scalar_values(const Run & run, const Selection & sel) {
    std::map<std::string, Value> out;
    for (const std::string & path : output::run_files(run.dir)) {
        if (fs::path(path).extension() != ".csv") continue;
        output::Contents c;
        try {
            c = output::open(path);
        } catch (const std::exception &) {
            continue;
        }
        if (c.kind != output::Kind::table && c.kind != output::Kind::probe) continue;
        std::string label = fs::relative(path, run.dir).string();
        label = label.substr(0, label.size() - 4);
        table_values(run, c, label, sel, out);
    }
    return out;
}

/** @brief Whether a requested name selects a quantity: its whole name, its column, or its table. */
bool selects(const std::string & request, const std::string & quantity) {
    if (request == quantity) return true;
    const size_t colon = quantity.rfind(':');
    return request == quantity.substr(colon + 1) || request == quantity.substr(0, colon);
}

// ---------------------------------------------------------------------------
// Error norms of a field against an exact solution or the finest run

struct Norms {
    double l1 = 0.0, l2 = 0.0, linf = 0.0;
    double t = 0.0;
};

/**
 * @brief The mesh a run's block records (its [mesh] and [[periodic]]), its files as they resolve from the run's
 *        directory, with the halo layers of remap polynomials of this degree.
 */
std::shared_ptr<Mesh> recorded_mesh(const Run & run, const uint8_t degree = 0) {
    const std::string text = recorded_input(run);
    if (text.empty()) throw std::runtime_error(run.arg + ": its run block records no input, so no mesh");
    const Input recorded = parse_input_string(text, run.arg + " (its run block)");
    if (!recorded.contains("mesh")) throw std::runtime_error(run.arg + ": its run block's input has no [mesh]");
    Input setup(Input::table_type{});
    Input mesh = recorded.at("mesh");
    if (find_value_or<std::string>(recorded, "mesh", "type", "file") == "file") {
        fs::path file = find_value_or<std::string>(recorded, "mesh", "filename", "mesh.msh");
        const std::string directory = string_at(run.block, {"run", "directory"});
        if (file.is_relative() && !directory.empty()) file = fs::path(directory) / file;
        std::error_code error;
        if (!fs::exists(file, error)) throw std::runtime_error(run.arg + ": its mesh file " + file.string() + " is not there");
        mesh.as_table()["filename"] = Input(file.string());
    }
    setup.as_table()["mesh"] = mesh;
    if (recorded.contains("periodic")) setup.as_table()["periodic"] = recorded.at("periodic");
    std::ostringstream sink;
    std::streambuf * const saved = std::cout.rdbuf(sink.rdbuf());
    struct Restore {
        std::streambuf * saved;
        ~Restore() { std::cout.rdbuf(saved); }
    } restore{saved};
    return remap::source_mesh(setup, degree);
}

/** @brief The field name and component of "NAME" or "NAME[k]". */
std::pair<std::string, uint32_t> field_component(const std::string & text) {
    const size_t open = text.find('[');
    if (open == std::string::npos) return {text, 0};
    if (text.back() != ']') throw UsageError("--field takes NAME or NAME[k], not " + text);
    const double k = parse_number(text.substr(open + 1, text.size() - open - 2), "--field's component");
    if (k < 0.0 || k != std::floor(k) || k > 2.0) throw UsageError("--field's component is 0, 1 or 2, not " + text);
    return {text.substr(0, open), static_cast<uint32_t>(k)};
}

/** @brief The values of one component of a run's field, in global cell order, at the snapshot of the study. */
struct FieldValues {
    std::vector<double> values;
    double t = 0.0;
    std::string file;
};

/** @brief The latest (or that of --time) of the whole-mesh snapshots of a run holding the field. */
FieldValues field_values(const Run & run, const std::string & field_text, const Selection & sel) {
    const auto [field, component] = field_component(field_text);
    output::Contents best;
    size_t best_index = 0;
    bool found = false;
    std::set<std::string> opened;
    for (const Written & w : run.written) {
        if (w.kind == "vtp" || w.kind == "spectra" || w.kind == "vtu") continue;
        const std::string path = resolve(run, w.path);
        if (!opened.insert(path).second) continue;
        output::Contents c;
        try {
            c = output::open(path);
        } catch (const std::exception &) {
            continue;
        }
        if (!c.global_order || !c.field(field)) continue;
        for (size_t k = 0; k < c.snapshots.size(); k++) {
            const output::Snapshot & s = c.snapshots[k];
            if (!s.run_id.empty() && !run.run_ids.empty() && !run.run_ids.contains(s.run_id)) continue;
            if (sel.kind == Selection::Kind::time && !same_time(s.t, sel.t, sel.same)) continue;
            if (!found || s.t > best.snapshots[best_index].t) {
                best = c;
                best_index = k;
                found = true;
            }
        }
    }
    if (!found) {
        throw std::runtime_error(run.arg + " has no snapshot of the whole mesh with " + field +
                                 (sel.kind == Selection::Kind::time ? " at t = " + number(sel.t) + " s" : "") +
                                 " (restart files, VTKHDF or XDMF output)");
    }
    const output::Field & f = *best.field(field);
    if (component >= f.components) {
        throw UsageError(field_text + ": " + field + " has " + count(f.components, "component"));
    }
    if (f.components > 1 && field_text.find('[') == std::string::npos) {
        throw UsageError(field + " is a vector: give one component, as " + field + "[0]");
    }
    const std::vector<double> all = output::read(best, field, best_index);
    FieldValues out;
    out.values.resize(all.size() / f.components);
    for (size_t c = 0; c < out.values.size(); c++) out.values[c] = all[c * f.components + component];
    out.t = best.snapshots[best_index].t;
    out.file = best.snapshots[best_index].file;
    return out;
}

bool recorded_axisymmetric(const Run & run) {
    const Input recorded = parse_input_string(recorded_input(run), run.arg + " (its run block)");
    return find_value_or<bool>(recorded, "physics", "axisymmetric", false);
}

void check_cells(const Run & run, const Mesh & mesh, const FieldValues & f) {
    const uint64_t n = mesh.n_global_cells > 0 ? mesh.n_global_cells : mesh.n_cells;
    if (n != f.values.size()) {
        throw std::runtime_error(run.arg + ": its recorded mesh has " + count(n, "cell") + ", its snapshot " +
                                 count(f.values.size()));
    }
}

/** @brief Adds a cell's difference d, of measure (weighted by r = y in axisymmetric runs) weight, to the norms. */
void accumulate(Norms & norms, double & measure, const double d, const double weight) {
    norms.l1 += d * weight;
    norms.l2 += d * d * weight;
    norms.linf = std::max(norms.linf, d);
    measure += weight;
}

void finish(Norms & norms, const double measure) {
    norms.l1 /= measure;
    norms.l2 = std::sqrt(norms.l2 / measure);
}

/** @brief The error norms of a run's field against the cell averages of the exact solution at its snapshot's time. */
Norms exact_norms(const Run & run, const std::string & field_text, const Expression & exact, const Selection & sel,
                  std::vector<std::string> & notes) {
    const FieldValues f = field_values(run, field_text, sel);
    const std::shared_ptr<Mesh> mesh = recorded_mesh(run);
    check_cells(run, *mesh, f);
    const bool axisymmetric = recorded_axisymmetric(run);
    Norms norms;
    norms.t = f.t;
    const overlap::Rule & rule = overlap::simplex_rule(EXACT_DEGREE + (axisymmetric ? 1u : 0u));
    double measure = 0.0;
    for (uint32_t c = 0; c < mesh->n_owned(); c++) {
        const overlap::Region region = overlap::region(overlap::cell_nodes(*mesh, c));
        double integral = 0.0, weight = 0.0;
        for (const overlap::Simplex & s : region.simplices) {
            const double v = overlap::volume(s);
            for (size_t q = 0; q < rule.points.size(); q++) {
                const overlap::Point x = overlap::map(s, rule.points[q]);
                const double r = axisymmetric ? x[1] : 1.0;
                const double z = N_DIM == 3 ? x[N_DIM - 1] : 0.0;
                integral += v * rule.weights[q] * r * exact(x[0], x[1], z, norms.t);
                weight += v * rule.weights[q] * r;
            }
        }
        accumulate(norms, measure, std::abs(f.values[mesh->h_global_cell(c)] - integral / weight), weight);
    }
    finish(norms, measure);
    notes.push_back(run.arg + ": " + field_component(field_text).first + " of " + f.file + " at t = " +
                    number(norms.t) + " s");
    return norms;
}

// ---------------------------------------------------------------------------
// Results

struct QuantityResult {
    std::string name, units;
    std::vector<double> values, times;            ///< per run, in the runs' order (coarse to fine)
    std::vector<convergence::Estimate> triplets;  ///< finest first: grids (1, 2, 3), then (2, 3, 4), ...
    bool zero = false;  ///< zero to round-off on every run
    bool extracted = false;  ///< of a table mallard post extract wrote from the runs' snapshots
    Status status = Status::OK;
    std::string verdict;

    std::string behavior() const { return zero ? "zero" : convergence::behavior_name(triplets[0].behavior); }
};

struct NormResult {
    std::string name;
    std::vector<double> errors;  ///< per run, coarse to fine
    std::vector<double> orders;  ///< of each pair of successive runs, coarse to fine
    double fitted = convergence::NaN;
    Status status = Status::OK;
    std::string verdict;
};

struct Study {
    std::vector<Run> runs;  ///< coarse to fine
    double design = convergence::NaN;
    std::string design_from;
    double safety = convergence::NaN;  ///< --safety-factor; NaN: Roache's
    double zero = ZERO;                         ///< of the coarsest precision of the runs
    double round_off = convergence::ROUND_OFF;  ///< of the coarsest precision of the runs
    std::vector<QuantityResult> quantities;
    std::string field, exact;  ///< exact empty: errors against the finest run
    std::vector<NormResult> norms;
    Status runs_status = Status::NONE;
    std::vector<std::string> run_notes, notes;
    Status status = Status::NONE;
};

/**
 * @brief The error norms of the field of every run but the finest against the finest run's, remapped
 *        conservatively onto its cells, coarse to fine.
 */
std::vector<Norms> finest_norms(Study & study, const Selection & sel) {
    const Run & finest = study.runs.back();
    const FieldValues reference = field_values(finest, study.field, sel);
    const std::shared_ptr<Mesh> fine_mesh = recorded_mesh(finest, REFERENCE_DEGREE);
    check_cells(finest, *fine_mesh, reference);
    const std::string field = field_component(study.field).first;
    remap::Options options;
    options.degree = REFERENCE_DEGREE;
    options.limit = false;
    options.axisymmetric = recorded_axisymmetric(finest);
    const overlap::Rule & rule = overlap::simplex_rule(options.axisymmetric ? 1u : 0u);
    std::vector<Norms> out;
    for (size_t i = 0; i + 1 < study.runs.size(); i++) {
        const Run & run = study.runs[i];
        const FieldValues f = field_values(run, study.field, sel);
        const std::shared_ptr<Mesh> mesh = recorded_mesh(run);
        check_cells(run, *mesh, f);
        if (!same_time(f.t, reference.t, sel.same)) {
            study.runs_status = report::worse(study.runs_status, Status::WARNING);
            study.run_notes.push_back(run.arg + "'s snapshot is at t = " + number(f.t) + " s and the finest run's at " +
                                      number(reference.t) + " s: compare them at one time with --time");
        }
        remap::Report report;
        const std::vector<double> on_run =
            remap::remap_global(fine_mesh, {reference.values}, {-1}, options, *mesh, report)[0];
        if (report.n_partial > 0 || report.n_nearest > 0) {
            study.runs_status = report::worse(study.runs_status, Status::WARNING);
            study.run_notes.push_back(finest.arg + "'s mesh covers " + count(report.n_partial + report.n_nearest) +
                                      " of " + run.arg + "'s cells only in part or not at all: the runs' domains differ");
        }
        Norms norms;
        norms.t = f.t;
        double measure = 0.0;
        for (uint32_t c = 0; c < mesh->n_owned(); c++) {
            const overlap::Region region = overlap::region(overlap::cell_nodes(*mesh, c));
            double weight = 0.0;
            for (const overlap::Simplex & s : region.simplices) {
                const double v = overlap::volume(s);
                for (size_t q = 0; q < rule.points.size(); q++) {
                    weight += v * rule.weights[q] * (options.axisymmetric ? overlap::map(s, rule.points[q])[1] : 1.0);
                }
            }
            const uint64_t g = mesh->h_global_cell(c);
            accumulate(norms, measure, std::abs(f.values[g] - on_run[g]), weight);
        }
        finish(norms, measure);
        study.notes.push_back(run.arg + ": " + field + " of " + f.file + " at t = " + number(f.t) + " s");
        out.push_back(norms);
    }
    study.notes.push_back(finest.arg + ": " + field + " of " + reference.file + " at t = " + number(reference.t) +
                          " s, the reference");
    return out;
}

std::string units_text(const std::string & units) {
    if (units.empty()) return "not recorded";
    return units == "1" ? "dimensionless" : units;
}

/** @brief The estimates of a quantity and its verdict in words. */
void estimate(QuantityResult & q, const Study & study) {
    const size_t n = study.runs.size();
    std::vector<double> h, phi;  // finest first
    for (size_t i = n; i-- > 0;) {
        h.push_back(study.runs[i].h);
        phi.push_back(q.values[i]);
    }
    const double fs3 = std::isfinite(study.safety) ? study.safety : convergence::SAFETY_THREE_GRIDS;
    const double fs2 = std::isfinite(study.safety) ? study.safety : convergence::SAFETY_TWO_GRIDS;
    if (n == 2) {
        q.triplets.push_back(convergence::two_grids({h[0], h[1]}, {phi[0], phi[1]}, study.design, fs2));
    } else {
        for (size_t i = 0; i + 2 < n; i++) {
            q.triplets.push_back(convergence::three_grids({h[i], h[i + 1], h[i + 2]}, {phi[i], phi[i + 1], phi[i + 2]}, fs3,
                                                          study.round_off));
        }
    }
    const convergence::Estimate & e = q.triplets[0];
    const std::string value = format("fine-grid value %s +/- %s (GCI, Fs = %.3g)", number(phi[0]).c_str(),
                                     percent(e.gci).c_str(), e.safety);
    const std::string design = std::isfinite(study.design) ? format(" (design %s)", short_number(study.design).c_str()) : "";
    std::vector<std::string> why;
    switch (e.behavior) {
        case Behavior::monotonic:
            q.status = Status::OK;
            q.verdict = "converging monotonically, order " + order_text(e.p) + design + ", extrapolated " +
                        number(e.extrapolated) + "; " + value;
            break;
        case Behavior::oscillatory:
            q.status = Status::WARNING;
            q.verdict = "oscillating between grids (R = " + format("%.3g", e.R) + "), so order " + order_text(e.p) +
                        " and the GCI are rough; " + value;
            break;
        case Behavior::divergent:
            q.status = Status::ERROR;
            q.verdict = "diverging: the change grows with refinement (R = " + format("%.3g", e.R) +
                        "), no order or GCI; check that the runs are one case, and refine further";
            break;
        case Behavior::unchanged:
            q.status = Status::OK;
            q.verdict = "the same on every grid, to round-off: grid independent";
            break;
        case Behavior::indeterminate:
            q.status = Status::WARNING;
            q.verdict = "the two finest grids agree to round-off and the coarser do not: no order follows; add a grid";
            break;
        case Behavior::assumed:
            q.status = Status::WARNING;
            q.verdict = format("2 grids: order assumed, not observed; Fs = %.3g: ", e.safety) + "with order " +
                        order_text(e.p) + ", extrapolated " + number(e.extrapolated) + "; " + value;
            break;
    }
    if (e.behavior == Behavior::monotonic || e.behavior == Behavior::oscillatory) {
        if (std::isfinite(study.design) && std::abs(e.p - study.design) > ORDER_SLACK) {
            q.status = report::worse(q.status, Status::WARNING);
            q.verdict += "; order " + order_text(e.p) + " is more than " + short_number(ORDER_SLACK) +
                         " from the design order " + short_number(study.design);
        }
        if (q.triplets.size() > 1 && std::isfinite(q.triplets[1].p) &&
            std::abs(q.triplets[1].p - e.p) > ORDER_DRIFT * std::abs(e.p)) {
            q.status = report::worse(q.status, Status::WARNING);
            q.verdict += "; the order is not yet steady (" + order_text(e.p) + " on the finest three grids, " +
                         order_text(q.triplets[1].p) + " on the next): the coarser grids are outside the asymptotic range";
        }
        if (!e.iterated) q.verdict += "; the order's iteration did not converge";
    }
}

void norm_verdict(NormResult & r, const Study & study) {
    const double p = r.orders.back();
    const std::string design = std::isfinite(study.design) ? format(" (design %s)", short_number(study.design).c_str()) : "";
    if (!(r.errors.back() < r.errors[r.errors.size() - 2])) {
        r.status = Status::ERROR;
        r.verdict = "the error does not decrease from the next-finest grid to the finest";
        return;
    }
    if (!std::isfinite(p)) {
        r.status = Status::ERROR;
        r.verdict = "no positive order fits the errors against the finest grid: refine further";
        return;
    }
    const bool finest = study.exact.empty();
    r.status = Status::OK;
    r.verdict = "order " + order_text(p) + (finest ? " from the two runs next to the finest" : " on the finest pair") +
                design;
    if (r.errors.size() > 2) {
        r.verdict += ", " + order_text(r.fitted) +
                     (finest ? " fitted over every run but the finest" : " fitted over every grid");
    }
    if (std::isfinite(study.design) && std::abs(p - study.design) > ORDER_SLACK) {
        r.status = Status::WARNING;
        r.verdict += "; more than " + short_number(ORDER_SLACK) + " from the design order";
    }
}

// ---------------------------------------------------------------------------
// Report

std::string ratio_text(const Run & r, const Run & fine) { return format("%.4g", r.h / fine.h); }

report::Text study_text(const Study & s, const bool color) {
    using Kind = report::Column::Kind;
    report::Text text(color);
    text.section("Runs", s.runs_status);
    std::vector<report::Row> rows;
    for (const Run & r : s.runs) {
        rows.push_back({Status::NONE, {r.arg, count(r.cells), ratio_text(r, s.runs.back()), r.run_id.empty() ? "-" : r.run_id}});
    }
    text.table({"run", "cells", "h / h_fine", "run id"}, rows, false);
    if (std::isfinite(s.design)) text.note("design order " + short_number(s.design) + ": " + s.design_from);
    for (const std::string & n : s.run_notes) text.line(n);

    if (!s.quantities.empty()) {
        Status worst = Status::NONE;
        for (const auto & q : s.quantities) worst = report::worse(worst, q.status);
        text.section("Grid convergence", worst);
        rows.clear();
        for (const QuantityResult & q : s.quantities) {
            const convergence::Estimate & e = q.triplets[0];
            const bool estimated = std::isfinite(e.extrapolated);
            rows.push_back({q.status,
                            {q.name, number(q.values.back()), estimated ? number(e.extrapolated) : "-", units_text(q.units),
                             order_text(e.p), estimated ? percent(e.gci) : "-", q.behavior()}});
        }
        text.table({"quantity", {"fine", Kind::VALUE}, {"extrapolated", Kind::VALUE}, {"units", Kind::UNITS}, "order",
                    "GCI", "behavior"},
                   rows);
        if (s.runs.size() > 3) {
            text.blank();
            text.note("order of each three successive grids, finest first:");
            rows.clear();
            for (const QuantityResult & q : s.quantities) {
                std::string orders;
                for (const auto & e : q.triplets) orders += (orders.empty() ? "" : ", ") + order_text(e.p);
                rows.push_back({Status::NONE, {q.name, orders}});
            }
            text.table({"quantity", "orders"}, rows, false);
        }
        text.blank();
        for (const QuantityResult & q : s.quantities) text.line(q.name + ": " + q.verdict);
        std::vector<std::string> extracted;
        for (const QuantityResult & q : s.quantities) {
            const std::string table = q.name.substr(0, q.name.rfind(':'));
            if (q.extracted && std::find(extracted.begin(), extracted.end(), table) == extracted.end()) {
                extracted.push_back(table);
            }
        }
        if (!extracted.empty()) {
            std::string list;
            for (const std::string & t : extracted) list += (list.empty() ? "" : ", ") + t;
            text.note("extracted by mallard post extract from the runs' snapshots: " + list);
        }
    }
    if (!s.norms.empty()) {
        Status worst = Status::NONE;
        for (const auto & r : s.norms) worst = report::worse(worst, r.status);
        const bool finest = s.exact.empty();
        text.section("Error of " + s.field + (finest ? " against the finest run" : " against the exact solution"), worst);
        if (finest) {
            text.note(s.field + " of " + s.runs.back().arg + ", remapped onto each coarser run's cells");
        } else {
            text.note(s.field + " = " + s.exact + ", cell averages");
        }
        std::vector<report::Column> header = {"norm"};
        for (size_t i = 0; i < s.norms[0].errors.size(); i++) header.emplace_back(count(s.runs[i].cells, "cell"));
        header.emplace_back("orders");
        rows.clear();
        for (const NormResult & r : s.norms) {
            report::Row row{r.status, {r.name}};
            for (const double e : r.errors) row.cells.push_back(format("%.4g", e));
            std::string orders;
            for (const double p : r.orders) orders += (orders.empty() ? "" : ", ") + order_text(p);
            row.cells.push_back(orders);
            rows.push_back(row);
        }
        text.table(header, rows);
        text.blank();
        for (const NormResult & r : s.norms) text.line(r.name + ": " + r.verdict);
    }
    if (!s.notes.empty()) {
        text.section("Notes");
        for (const std::string & n : s.notes) text.line(n);
    }
    return text;
}

std::string numbers_json(const std::vector<double> & list) {
    std::string out = "[";
    for (size_t i = 0; i < list.size(); i++) out += (i ? "," : "") + json::number(list[i]);
    return out + "]";
}

std::string strings_json(const std::vector<std::string> & list) {
    std::string out = "[";
    for (const auto & s : list) out += (out.size() > 1 ? "," : "") + json::quote(s);
    return out + "]";
}

std::string estimate_json(const convergence::Estimate & e) {
    return json::Object()
        .add("behavior", std::string(convergence::behavior_name(e.behavior)))
        .add("order", e.p)
        .add("r21", e.r21)
        .add("r32", e.r32)
        .add("R", e.R)
        .add("extrapolated", e.extrapolated)
        .add("e_a", e.e_a)
        .add("e_ext", e.e_ext)
        .add("gci", e.gci)
        .add("safety_factor", e.safety)
        .str();
}

std::string study_json(const Study & s, const int exit_status) {
    json::Object o;
    o.add("schema", std::string("mallard-convergence")).add("version", 1);
    o.add("status", std::string(report::status_name(s.status))).add("exit_status", exit_status);
    std::string runs = "[";
    for (const Run & r : s.runs) {
        runs += (runs.size() > 1 ? "," : "") + json::Object()
                                                   .add("run", r.arg)
                                                   .add("cells", r.cells)
                                                   .add("h", r.h)
                                                   .add("h_ratio", r.h / s.runs.back().h)
                                                   .raw("run_id", r.run_id.empty() ? "null" : json::quote(r.run_id))
                                                   .str();
    }
    o.raw("runs", runs + "]");
    o.add("design_order", s.design);
    o.raw("input_differences", strings_json(s.run_notes));
    std::string quantities = "[";
    for (const QuantityResult & q : s.quantities) {
        std::string triplets = "[";
        for (const auto & e : q.triplets) triplets += (triplets.size() > 1 ? "," : "") + estimate_json(e);
        json::Object e;
        e.add("name", q.name).add("units", q.units).add("status", std::string(report::status_name(q.status)));
        e.add("extracted", q.extracted);
        e.raw("values", numbers_json(q.values)).raw("t", numbers_json(q.times));
        e.add("behavior", q.behavior()).raw("estimate", estimate_json(q.triplets[0])).raw("triplets", triplets + "]").add("verdict", q.verdict);
        quantities += (quantities.size() > 1 ? "," : "") + e.str();
    }
    o.raw("quantities", quantities + "]");
    if (!s.norms.empty()) {
        std::string norms = "[";
        for (const NormResult & r : s.norms) {
            norms += (norms.size() > 1 ? "," : "") + json::Object()
                                                         .add("norm", r.name)
                                                         .add("status", std::string(report::status_name(r.status)))
                                                         .raw("errors", numbers_json(r.errors))
                                                         .raw("orders", numbers_json(r.orders))
                                                         .add("fitted_order", r.fitted)
                                                         .add("verdict", r.verdict)
                                                         .str();
        }
        o.raw("field", json::Object()
                           .add("name", s.field)
                           .raw("exact", s.exact.empty() ? "null" : json::quote(s.exact))
                           .add("reference", std::string(s.exact.empty() ? "finest" : "exact"))
                           .raw("norms", norms + "]")
                           .str());
    } else {
        o.raw("field", "null");
    }
    o.raw("notes", strings_json(s.notes));
    return o.str();
}

// ---------------------------------------------------------------------------

int convergence_study(const cli::Args & args) {
    const std::string fmt = args.value_or("--format", "text");
    if (fmt != "text" && fmt != "json") throw UsageError("--format takes text or json, not " + fmt);
    if (args.positional.size() < 2) throw UsageError("give two or more runs, at different resolutions");
    if (args.has("--field") && !args.has("--exact") && args.positional.size() < 3) {
        throw UsageError("--field without --exact needs three or more runs: the finest is the reference, and the "
                         "order follows from the errors of the others");
    }
    if (args.has("--exact") && !args.has("--field")) throw UsageError("--exact needs --field, the field it is of");
    if (args.has("--time") && args.has("--average")) throw UsageError("--time and --average both choose the values; give one");
    Selection sel;
    if (args.has("--time")) {
        sel.kind = Selection::Kind::time;
        sel.t = parse_number(args.value_or("--time", ""), "--time");
    } else if (args.has("--average")) {
        if (args.has("--field")) throw UsageError("--average is for quantities; a field's error is at one time (--time)");
        const std::string w = args.value_or("--average", "");
        const size_t colon = w.find(':');
        if (colon == std::string::npos) throw UsageError("--average takes T0:T1, not " + w);
        sel.kind = Selection::Kind::average;
        sel.t0 = parse_number(w.substr(0, colon), "--average");
        sel.t1 = parse_number(w.substr(colon + 1), "--average");
        if (!(sel.t1 > sel.t0)) throw UsageError("--average takes T0:T1 with T0 < T1, not " + w);
    }
    Study study;
    if (args.has("--safety-factor")) {
        study.safety = parse_number(args.value_or("--safety-factor", ""), "--safety-factor");
        if (!(study.safety >= 1.0)) throw UsageError("--safety-factor is at least 1");
    }
    std::vector<double> sizes;
    if (args.has("--sizes")) {
        for (const std::string & s : split_list(args.value_or("--sizes", ""))) {
            sizes.push_back(parse_number(s, "--sizes"));
            if (!(sizes.back() > 0.0)) throw UsageError("--sizes takes positive numbers");
        }
        if (sizes.size() != args.positional.size()) {
            throw UsageError("--sizes takes one size per run (" + std::to_string(args.positional.size()) + "), not " +
                             std::to_string(sizes.size()));
        }
    }

    for (const std::string & a : args.positional) {
        std::error_code error;
        if (fs::is_directory(a, error)) {
            study.runs.push_back(open_directory(a));
        } else if (!args.has("--field")) {
            throw UsageError(a + " is a file: give run directories (their tables hold the quantities), or "
                                 "snapshots with --field");
        } else {
            study.runs.push_back(open_file(a));
        }
    }
    for (size_t i = 0; i < study.runs.size(); i++) {
        Run & r = study.runs[i];
        if (r.dim != 2 && r.dim != 3) r.dim = N_DIM;
        if (r.dim != study.runs[0].dim) {
            throw std::runtime_error(study.runs[0].arg + " is " + std::to_string(study.runs[0].dim) + "D and " + r.arg +
                                     " " + std::to_string(r.dim) + "D");
        }
        r.h = sizes.empty() ? std::pow(1.0 / static_cast<double>(r.cells), 1.0 / r.dim) : sizes[i];
    }
    std::stable_sort(study.runs.begin(), study.runs.end(), [](const Run & a, const Run & b) { return a.h > b.h; });
    auto single = [](const Run & r) { return string_at(r.block, {"build", "precision"}) == "single"; };
    if (std::any_of(study.runs.begin(), study.runs.end(), single)) {
        const double epsilons = SINGLE_EPSILONS * static_cast<double>(std::numeric_limits<float>::epsilon());
        study.zero = std::max(ZERO, epsilons);
        study.round_off = std::max(convergence::ROUND_OFF, epsilons);
        sel.same = std::max(sel.same, epsilons);
        sel.window = std::max(sel.window, epsilons);
    }
    for (size_t i = 1; i < study.runs.size(); i++) {
        if (study.runs[i].h == study.runs[i - 1].h) {
            throw std::runtime_error(study.runs[i - 1].arg + " and " + study.runs[i].arg + " have the same " +
                                     (sizes.empty() ? "number of cells (" + count(study.runs[i].cells) + ")" : "size") +
                                     ": each run needs a resolution of its own");
        }
    }
    for (size_t i = 0; i < study.runs.size(); i++) {
        for (size_t j = 0; j < i; j++) {
            for (const std::string & id : study.runs[i].run_ids) {
                if (study.runs[j].run_ids.contains(id)) {
                    throw std::runtime_error(study.runs[j].arg + " and " + study.runs[i].arg + " hold the same run (" +
                                             id + ")");
                }
            }
        }
    }

    if (args.has("--order")) {
        study.design = parse_number(args.value_or("--order", ""), "--order");
        if (!(study.design > 0.0)) throw UsageError("--order is positive");
        study.design_from = "--order";
    } else {
        std::set<double> orders;
        for (const Run & r : study.runs) orders.insert(design_order(r));
        if (orders.size() == 1 && std::isfinite(*orders.begin())) {
            study.design = *orders.begin();
            study.design_from = "the runs' [numerics.face_reconstruction]";
        }
    }
    if (study.runs.size() == 2 && !std::isfinite(study.design) && !args.has("--field")) {
        throw std::runtime_error("two runs need an assumed order, and their inputs record none: give --order P");
    }
    const std::vector<std::string> differ = differing_tables(study.runs);
    if (!differ.empty()) {
        study.runs_status = Status::WARNING;
        std::string list;
        for (const auto & d : differ) list += (list.empty() ? "" : ", ") + d;
        study.run_notes.push_back("the runs' inputs differ in " + list +
                                  ": a convergence study changes only the resolution");
    }

    if (args.has("--field")) {
        study.field = args.value_or("--field", "");
        if (study.field.find(',') != std::string::npos) throw UsageError("--field takes one field, with its --exact");
        field_component(study.field);
        std::vector<Norms> norms;
        if (args.has("--exact")) {
            study.exact = args.value_or("--exact", "");
            const Expression exact("--exact", study.exact);
            for (const Run & r : study.runs) norms.push_back(exact_norms(r, study.field, exact, sel, study.notes));
        } else {
            norms = finest_norms(study, sel);
        }
        const double h_ref = study.exact.empty() ? study.runs.back().h : convergence::NaN;
        for (const char * name : {"L1", "L2", "Linf"}) {
            NormResult r;
            r.name = name;
            std::vector<double> h;
            for (size_t i = 0; i < norms.size(); i++) {
                const Norms & n = norms[i];
                r.errors.push_back(r.name == "L1" ? n.l1 : r.name == "L2" ? n.l2 : n.linf);
                h.push_back(study.runs[i].h);
                if (i == 0) continue;
                r.orders.push_back(std::isfinite(h_ref)
                                       ? convergence::reference_order(h[i - 1], h[i], h_ref, r.errors[i - 1], r.errors[i])
                                       : convergence::pair_order(h[i - 1], h[i], r.errors[i - 1], r.errors[i]));
            }
            r.fitted = std::isfinite(h_ref) ? convergence::fitted_reference_order(h, r.errors, h_ref)
                                            : convergence::fitted_order(h, r.errors);
            norm_verdict(r, study);
            study.norms.push_back(r);
        }
    } else {
        std::vector<std::map<std::string, Value>> values;
        for (const Run & r : study.runs) values.push_back(scalar_values(r, sel));
        std::vector<std::string> requests = split_list(args.value_or("--quantity", ""));
        std::set<std::string> matched;
        std::vector<std::string> missing;
        for (const auto & [name, first] : values[0]) {
            if (!requests.empty() &&
                std::none_of(requests.begin(), requests.end(), [&](const std::string & q) { return selects(q, name); })) {
                continue;
            }
            for (const std::string & q : requests) {
                if (selects(q, name)) matched.insert(q);
            }
            QuantityResult q;
            q.name = name;
            q.units = first.units;
            bool everywhere = true;
            for (size_t i = 0; i < values.size(); i++) {
                const auto it = values[i].find(name);
                if (it == values[i].end() || !std::isfinite(it->second.v)) {
                    everywhere = false;
                    break;
                }
                q.values.push_back(it->second.v);
                q.times.push_back(it->second.t);
                q.extracted = q.extracted || it->second.extracted;
                q.zero = (i == 0 || q.zero) && std::abs(it->second.v) <= study.zero * it->second.scale;
            }
            if (!everywhere) {
                missing.push_back(name);
                continue;
            }
            if (q.zero) {
                q.triplets.push_back({});
                q.triplets[0].behavior = Behavior::unchanged;
                q.verdict = format("zero to round-off: below %g of the largest column of its table on every run",
                                   study.zero);
            } else {
                estimate(q, study);
            }
            study.quantities.push_back(q);
        }
        for (const std::string & q : requests) {
            if (!matched.contains(q)) {
                throw std::runtime_error("no quantity " + q + " in " + study.runs[0].arg +
                                         (sel.kind == Selection::Kind::final ? "" : " at the times chosen") +
                                         ": quantities are TABLE:COLUMN of the run's CSV tables (forces, integrals, "
                                         "probes)");
            }
        }
        if (!missing.empty()) {
            std::string list;
            for (const auto & m : missing) list += (list.empty() ? "" : ", ") + m;
            study.notes.push_back("not in every run, so left out: " + list);
        }
        if (study.quantities.empty()) {
            throw std::runtime_error(
                "no quantity is in every run" +
                std::string(sel.kind == Selection::Kind::final ? "" : " at the times chosen") +
                ": the runs' directories hold no forces, integrals or probe tables of their runs ([[forces]], "
                "[integrals], [[probes]]); for a field, give --field");
        }
        if (sel.kind == Selection::Kind::final) {
            double lo = std::numeric_limits<double>::infinity(), hi = -lo;
            for (const QuantityResult & q : study.quantities) {
                for (const double t : q.times) {
                    lo = std::min(lo, t);
                    hi = std::max(hi, t);
                }
            }
            if (!same_time(lo, hi, sel.same)) {
                study.notes.push_back("the runs' last values are at different times (t = " + number(lo) + " to " +
                                      number(hi) + " s): right for a steady case; for an unsteady one, compare at "
                                      "one time with --time or over a window with --average");
            }
        }
    }
    if (std::isfinite(study.safety)) study.notes.push_back(format("factor of safety %.3g, from --safety-factor", study.safety));

    study.status = study.runs_status;
    for (const auto & q : study.quantities) study.status = report::worse(study.status, q.status);
    for (const auto & r : study.norms) study.status = report::worse(study.status, r.status);
    const int exit_status = study.status == Status::ERROR ? 1 : 0;
    if (fmt == "json") {
        std::cout << study_json(study, exit_status) << "\n";
    } else {
        study_text(study, logging::color_stdout()).print();
    }
    return exit_status;
}

} // namespace

int post_convergence(const cli::Args & args) {
    try {
        return convergence_study(args);
    } catch (const UsageError & e) {
        return cli::usage_error(*cli::command("post convergence"), e.what());
    } catch (const InputError & e) {
        logging::error(diagnostic_code::CONVERGENCE, std::string(NAME) + ": a run's input: " + e.what());
    } catch (const std::exception & e) {
        logging::error(diagnostic_code::CONVERGENCE, std::string(NAME) + ": " + e.what());
    }
    return 1;
}

int convergence_main(const cli::Args & args) {
    int argc = args.argc;
    char ** argv = args.argv;
    comm::Session session(argc, argv);
    if (comm::size() > 1) {
        if (comm::is_root()) {
            logging::error(diagnostic_code::ONE_RANK, std::string(NAME) + ": runs on one rank, not " +
                                                          std::to_string(comm::size()) + "; start it without mpirun.");
        }
        return 1;
    }
    Kokkos::ScopeGuard kokkos(argc, argv);
    return post_convergence(args);
}

/**
 * @file run_page.cpp
 * @brief The run report page: its model, fed record by record, and its HTML.
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 */

#include "run_page.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <utility>

#include "common_io.h"
#include "input.h"
#include "log.h"
#include "run_health.h"

namespace run_page {

namespace {

using logging::format;
using Type = json::Value::Type;

const char * const PLOT_JS =
#include "run_page_plot.inc"
    ;

const char * const STYLE = R"css(
:root{color-scheme:light dark;--bg:#fbfbfa;--fg:#1d1f21;--dim:#5f6368;--line:#d9dadc;--panel:#ffffff;
--accent:#2a6fdb;--band:rgba(42,111,219,.18);--ok:#1e7d3a;--warn:#a05a00;--error:#b3261e;--code:#f2f3f4}
@media (prefers-color-scheme:dark){:root{--bg:#16181b;--fg:#e6e7e8;--dim:#9aa0a6;--line:#33373c;--panel:#1d2024;
--accent:#7aa7f0;--band:rgba(122,167,240,.22);--ok:#6cc487;--warn:#e0a24a;--error:#f28b82;--code:#23272b}}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.45 system-ui,-apple-system,"Segoe UI",sans-serif}
main{max-width:1060px;margin:0 auto;padding:24px 16px 64px}
h1{font-size:1.6em;margin:0 0 4px}
h2{font-size:1.25em;margin:36px 0 10px;padding-bottom:4px;border-bottom:1px solid var(--line)}
h3{font-size:1.02em;margin:20px 0 6px}
.sub,.note,figcaption,.readout{color:var(--dim)}.note{font-size:.92em}
nav{margin:14px 0;font-size:.92em}nav a{margin-right:14px;color:var(--accent);text-decoration:none}
.banner{border:1px solid var(--line);border-left:6px solid var(--dim);background:var(--panel);padding:12px 16px;
margin:16px 0;border-radius:4px}
.banner.ok{border-left-color:var(--ok)}.banner.warning{border-left-color:var(--warn)}
.banner.error{border-left-color:var(--error)}
.banner strong{font-size:1.1em}
.status{font-weight:600;font-size:.85em;letter-spacing:.03em}
.status.ok{color:var(--ok)}.status.warning{color:var(--warn)}.status.error{color:var(--error)}
.scroll{overflow-x:auto}
table{border-collapse:collapse;margin:6px 0 10px;font-size:.92em}
th,td{text-align:left;padding:3px 12px 3px 0;vertical-align:top;border-bottom:1px solid var(--line)}
th{color:var(--dim);font-weight:500}td.num{text-align:right;font-variant-numeric:tabular-nums;white-space:nowrap}
table.kv th{width:12em}
code,pre{font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;font-size:.88em}
pre{background:var(--code);padding:10px;overflow-x:auto;border-radius:4px}
.plots{display:grid;grid-template-columns:repeat(auto-fill,minmax(320px,1fr));gap:14px}
figure{margin:0;background:var(--panel);border:1px solid var(--line);border-radius:4px;padding:8px}
figcaption{font-size:.9em;margin-bottom:2px}.readout{font-size:.8em;min-height:1.3em}
svg{width:100%;height:auto;display:block}
svg .grid{stroke:var(--line);stroke-width:1}svg .frame{fill:none;stroke:var(--line)}
svg .tick{fill:var(--dim);font-size:11px}svg .axis{fill:var(--fg);font-size:12px}
svg .line{fill:none;stroke:var(--accent);stroke-width:1.6}
svg .band{fill:var(--band);stroke:none}svg .cursor{stroke:var(--dim);stroke-dasharray:3 3}
details summary{cursor:pointer;color:var(--accent)}
@media print{nav,.plots,#plots-nojs{display:none}body{background:#fff;color:#000}}
)css";

std::string escape(std::string_view text) {
    std::string out;
    out.reserve(text.size());
    for (const char c : text) {
        switch (c) {
            case '&': out += "&amp;"; break;
            case '<': out += "&lt;"; break;
            case '>': out += "&gt;"; break;
            case '"': out += "&quot;"; break;
            default: out += c;
        }
    }
    return out;
}

const json::Value * at(const json::Value & v, std::initializer_list<const char *> path) {
    const json::Value * p = &v;
    for (const char * key : path) {
        p = p->find(key);
        if (!p) return nullptr;
    }
    return p;
}

double number_of(const json::Value * v, double fallback = 0.0) {
    return v && v->type == Type::NUMBER ? v->number : fallback;
}

std::string number_text(double x) {
    if (!std::isfinite(x)) return "-";
    if (x == std::floor(x) && std::fabs(x) < 1e15) return format("%.0f", x);
    return format("%.6g", x);
}

/** @brief A value as a table cell shows it: arrays of scalars as (a, b), objects as k: v pairs. */
std::string text_of(const json::Value * v) {
    if (!v) return "-";
    switch (v->type) {
        case Type::NUL: return "-";
        case Type::BOOL: return v->boolean ? "yes" : "no";
        case Type::NUMBER: return number_text(v->number);
        case Type::STRING: return v->string;
        case Type::ARRAY: {
            std::string out = "(";
            for (size_t i = 0; i < v->array.size(); i++) out += (i ? ", " : "") + text_of(&v->array[i]);
            return out + ")";
        }
        case Type::OBJECT: {
            std::string out;
            for (size_t i = 0; i < v->names.size(); i++) {
                out += (i ? ", " : "") + v->names[i] + ": " + text_of(&v->members[i]);
            }
            return out;
        }
    }
    return "-";
}

/** @brief Whether a value is false (a missing one is not). */
bool is_false(const json::Value * v) { return v && v->type == Type::BOOL && !v->boolean; }

std::string str(const json::Value * v) { return v && v->type == Type::STRING ? v->string : std::string(); }

std::string cell(const std::string & text) { return "<td>" + escape(text) + "</td>"; }
std::string num(const std::string & text) { return "<td class=\"num\">" + escape(text) + "</td>"; }

std::string head(const std::vector<std::string> & names) {
    std::string out = "<div class=\"scroll\"><table><tr>";
    for (const auto & n : names) out += "<th>" + escape(n) + "</th>";
    return out + "</tr>";
}
const char * const TABLE_END = "</table></div>\n";

/** @brief A table of label and value rows, those without a value left out. */
std::string key_values(const std::vector<std::pair<std::string, std::string>> & rows) {
    std::string out = "<div class=\"scroll\"><table class=\"kv\">";
    for (const auto & [k, v] : rows) {
        if (!v.empty() && v != "-") out += "<tr><th>" + escape(k) + "</th>" + cell(v) + "</tr>";
    }
    return out + TABLE_END;
}

std::string status_span(const char * status, const std::string & word) {
    return std::string("<span class=\"status ") + status + "\">" + escape(word) + "</span>";
}

/** @brief A unit as the page shows it: "-" for a dimensionless quantity ("1"). */
std::string shown(const std::string & unit) { return unit == "1" ? "-" : unit; }

/** @brief What a quantity of the progress record is, in words; empty if it has no description here. */
std::string description(const std::string & name) {
    static const std::map<std::string, std::string> TEXT = {
        {"dt", "time step"},
        {"cfl", "CFL number: the time step over the largest stable one"},
        {"min_rho", "lowest density over the cells"},
        {"min_p", "lowest pressure over the cells"},
        {"max_mach", "highest Mach number over the cells"},
        {"troubled_cells", "share of the cells TENO marks troubled (shocks, discontinuities)"},
        {"min_T", "lowest temperature over the cells"},
        {"max_T", "highest temperature over the cells"},
        {"max_sum_Y_error", "largest |sum of the mass fractions - 1| over the cells"},
        {"reacting_cells", "share of the cells whose chemistry is advanced"},
        {"max_chemistry_cost", "most chemistry integrator steps of a cell"},
        {"wall_per_step", "wall time per time step"},
        {"cells_per_second", "throughput: cell updates per second of time stepping"},
        {"memory_host", "host memory of the largest rank"},
        {"memory_device", "GPU memory in use on the fullest device"},
        {"kinetic_energy", "kinetic energy, integral of rho |u|^2 / 2"},
        {"enstrophy", "enstrophy, integral of rho |omega|^2 / 2"},
        {"dilatation_squared", "integral of (div u)^2"},
        {"pressure_dilatation", "pressure-dilatation work, integral of p div u"},
        {"velocity_squared", "integral of |u|^2"},
        {"vorticity_squared", "integral of |omega|^2"},
        {"density_squared", "integral of rho^2"},
        {"temperature", "integral of T"},
        {"temperature_squared", "integral of T^2"},
        {"ke_rate_convective", "kinetic energy rate of the convective fluxes"},
        {"ke_rate_viscous", "kinetic energy rate of the viscous fluxes"},
        {"ke_rate_sgs", "kinetic energy rate of the subgrid model"},
        {"eps_numerical", "numerical dissipation of kinetic energy"},
        {"pressure_work", "kinetic energy rate of the pressure work"},
        {"les_C", "dynamic LES model coefficient"},
        {"tf_beta", "dynamic flame efficiency coefficient"},
        {"Q", "heat flux of the modeled walls into the fluid"},
        {"mass", "total mass"},
        {"energy", "total energy"},
    };
    if (const auto it = TEXT.find(name); it != TEXT.end()) return it->second;
    if (name.size() > 2 && name[0] == 'F' && name.find('_') == 2) {
        return (name.ends_with("_pressure") ? std::string("pressure") : std::string("viscous")) + " force along " +
               name[1];
    }
    if (name.starts_with("momentum_")) return "total momentum along " + name.substr(9);
    return "";
}

/** @brief An axis label: the quantity and its units in brackets, "-" for none. */
std::string label(const std::string & name, const std::string & unit) {
    return name + " [" + (unit.empty() ? "units not recorded" : unit) + "]";
}

/** @brief What a group of plots shows, in words. */
std::string group_note(const std::string & group) {
    if (group == "Time step") return "The time step and the CFL number it was taken at.";
    if (group == "Extremes over the cells") {
        return "The lowest and highest values over the cells at each progress row: what fails first in a run that "
               "diverges.";
    }
    if (group == "Throughput and memory") return "The cost of each step and the memory in use.";
    if (group == "Integrals over the domain") {
        return "The [integrals] over the domain, as integrals.csv records them; per unit depth in 2D.";
    }
    if (group.starts_with("Forces on ")) {
        return "The force on the zone's walls, from the pressure and the viscous stress, as its [[forces]] file "
               "records it; per unit depth in 2D.";
    }
    if (group.starts_with("Probe ")) {
        return "The probe's latest reductions (or a one-point probe's values) at each progress row, as its CSV "
               "files record them; the files hold every write.";
    }
    if (group == "Conservation") {
        return "The conservation ledger's unexplained change of each total, relative to its scale (see Health).";
    }
    if (group == "Numerical repairs") return "How often each numerical repair has acted since the run started.";
    return "";
}

/** @brief What a plotted series is, in words: its quantity's description, or what its group makes of it. */
std::string caption(const Series & s) {
    if (s.group == "Conservation") return "unexplained change of the " + s.name + " total, relative to its scale";
    if (s.group == "Numerical repairs") return "times the repair " + s.name + " has acted";
    if (s.group.starts_with("Probe ")) {
        static const std::vector<std::pair<std::string, std::string>> REDUCTIONS = {
            {"integral_", "integral"}, {"mean_", "mean"},          {"favre_", "density-weighted (Favre) mean"},
            {"min_", "lowest value"},  {"max_", "highest value"}, {"rms_", "root mean square"}};
        for (const auto & [prefix, words] : REDUCTIONS) {
            if (s.name.starts_with(prefix)) return words + " of " + s.name.substr(prefix.size()) + " over the probe";
        }
        if (s.name.starts_with("moment_")) return "moment of " + s.name.substr(7) + " about moment_center";
        return s.name + " at the probe's point";
    }
    const std::string d = description(s.name);
    return d.empty() ? s.name : d;
}

/** @brief What a health activity counts, and why it matters. */
std::string activity_text(const std::string & name) {
    if (name == "limited_cells") {
        return "cells where the MUSCL limiter cuts the reconstructed gradient (smallest limiter below " +
               format("%.2g", health::LIMITED_BELOW) + "): shocks and steep gradients";
    }
    if (name == "troubled_cells") {
        return "cells TENO marks troubled and reconstructs from its smoother stencils: shocks";
    }
    if (name == "upwind_fraction") {
        return "the hybrid flux's mean upwind fraction: how much numerical dissipation it adds";
    }
    return "";
}

/** @brief A value of the resolved input by its dotted key, as TOML writes it, or empty. */
std::string input_value(const Input & input, const std::string & key) {
    const Input * v = &input;
    size_t begin = 0;
    while (begin <= key.size()) {
        const size_t dot = key.find('.', begin);
        const std::string part = key.substr(begin, dot == std::string::npos ? std::string::npos : dot - begin);
        if (!v->is_table()) return "";
        const auto & entries = v->as_table().entries();
        const auto it = entries.find(part);
        if (it == entries.end()) return "";
        v = &it->second;
        if (dot == std::string::npos) break;
        begin = dot + 1;
    }
    if (v->is_string()) return v->as_string();
    if (v->is_table() || v->is_array_of_tables()) return "";
    return toml::format(*v);
}

constexpr std::pair<const char *, const char *> SUMMARY_KEYS[] = {
    {"physics.type", "Physics"},
    {"physics.units", "Units"},
    {"physics.gamma", "Ratio of specific heats"},
    {"physics.gas", "Gas"},
    {"physics.mechanism", "Mechanism"},
    {"physics.transport", "Transport"},
    {"les.model", "LES model"},
    {"mesh.type", "Mesh"},
    {"mesh.filename", "Mesh file"},
    {"numerics.preset", "Numerics preset"},
    {"numerics.face_reconstruction.type", "Reconstruction"},
    {"numerics.riemann_solver", "Riemann solver"},
    {"numerics.time_integrator", "Time integrator"},
    {"run.cfl", "CFL"},
    {"run.dt", "Time step"},
    {"run.n_steps", "Steps"},
    {"run.t_stop", "Stop time"},
    {"run.t_wall_stop", "Wall time limit"},
    {"initialize.type", "Initial state"},
};

std::string seconds_text(double s) { return s > 0.0 ? logging::duration(s) : "-"; }

std::string bytes_text(double b) { return b >= 0.0 ? logging::si(b) + "B" : "-"; }

} // namespace

void Series::add(const uint64_t step, const double t, const double value) {
    if (!data.empty() && data.back().step == step) return;
    if (n_rows == 0) {
        lowest = highest = value;
    } else {
        lowest = std::min(lowest, value);
        highest = std::max(highest, value);
    }
    n_rows++;
    if (data.empty() || data.back().n >= per_bucket) {
        if (data.size() == BUCKETS) {
            std::vector<Bucket> merged;
            merged.reserve(BUCKETS);
            for (size_t i = 0; i + 1 < data.size(); i += 2) {
                Bucket m = data[i + 1];
                m.min = std::min(data[i].min, data[i + 1].min);
                m.max = std::max(data[i].max, data[i + 1].max);
                m.n = data[i].n + data[i + 1].n;
                merged.push_back(m);
            }
            data = std::move(merged);
            per_bucket *= 2;
        }
        if (data.empty() || data.back().n >= per_bucket) {
            data.push_back({step, t, value, value, value, 1});
            return;
        }
    }
    Bucket & b = data.back();
    b.step = step;
    b.t = t;
    b.last = value;
    b.min = std::min(b.min, value);
    b.max = std::max(b.max, value);
    b.n++;
}

void Series::truncate(const uint64_t step) {
    while (!data.empty() && data.back().step > step) data.pop_back();
    n_rows = 0;
    for (size_t i = 0; i < data.size(); i++) {
        lowest = i ? std::min(lowest, data[i].min) : data[i].min;
        highest = i ? std::max(highest, data[i].max) : data[i].max;
        n_rows += data[i].n;
    }
    // Rows after the restart start a bucket of their own
    if (!data.empty()) data.back().n = std::max(data.back().n, per_bucket);
}

void Kept::add(const json::Value & v) {
    n++;
    if (first.size() < KEPT) {
        first.push_back(v);
        return;
    }
    last.push_back(v);
    if (last.size() > KEPT) last.erase(last.begin());
}

std::string Page::unit(const std::string & group, const std::string & name, const std::string & member) const {
    const json::Value * units = start.find("units");
    if (!units) return "";
    if (!group.empty()) units = units->find(group);
    if (units && !member.empty()) units = units->find(name);
    const json::Value * u = units ? units->find(member.empty() ? name : member) : nullptr;
    return u && u->type == Type::STRING ? shown(u->string) : "";
}

Series & Page::series(const std::string & group, const std::string & name, const std::string & unit) {
    const std::string key = group + "\n" + name;
    if (const auto it = by_key.find(key); it != by_key.end()) return all[it->second];
    by_key[key] = all.size();
    all.push_back(Series{});
    Series & s = all.back();
    s.group = group;
    s.name = name;
    s.unit = unit;
    return s;
}

void Page::progress(const json::Value & r) {
    const uint64_t step = static_cast<uint64_t>(number_of(r.find("step")));
    const double t = number_of(r.find("t"));
    auto add = [&](const std::string & group, const std::string & name, const json::Value * v, uint64_t s, double time,
                   const std::string & unit) {
        if (v && v->type == Type::NUMBER && std::isfinite(v->number)) series(group, name, unit).add(s, time, v->number);
    };
    add("Time step", "dt", r.find("dt"), step, t, unit("", "dt"));
    add("Time step", "cfl", r.find("cfl"), step, t, unit("", "cfl"));
    if (const json::Value * m = r.find("monitors"); m && m->type == Type::OBJECT) {
        for (size_t i = 0; i < m->names.size(); i++) {
            add("Extremes over the cells", m->names[i], &m->members[i], step, t, unit("monitors", m->names[i]));
        }
    }
    for (const char * key : {"wall_per_step", "cells_per_second", "memory_host", "memory_device"}) {
        add("Throughput and memory", key, r.find(key), step, t, unit("", key));
    }
    for (size_t i = 0; i < r.names.size(); i++) {
        const std::string & key = r.names[i];
        const json::Value & v = r.members[i];
        const bool integrals = key == "integrals";
        if (v.type != Type::OBJECT || (!integrals && !key.starts_with("forces_"))) continue;
        const std::string group = integrals ? "Integrals over the domain" : "Forces on " + key.substr(7);
        const uint64_t s = static_cast<uint64_t>(number_of(v.find("step"), double(step)));
        const double time = number_of(v.find("t"), t);
        for (size_t j = 0; j < v.names.size(); j++) {
            if (v.names[j] != "step" && v.names[j] != "t") {
                add(group, v.names[j], &v.members[j], s, time, unit(key, v.names[j]));
            }
        }
    }
    if (const json::Value * probes = r.find("probes"); probes && probes->type == Type::OBJECT) {
        for (size_t i = 0; i < probes->names.size(); i++) {
            const json::Value & v = probes->members[i];
            if (v.type != Type::OBJECT) continue;
            const uint64_t s = static_cast<uint64_t>(number_of(v.find("step"), double(step)));
            const double time = number_of(v.find("t"), t);
            for (size_t j = 0; j < v.names.size(); j++) {
                if (v.names[j] != "step" && v.names[j] != "t") {
                    add("Probe " + probes->names[i], v.names[j], &v.members[j], s, time,
                        unit("probes", probes->names[i], v.names[j]));
                }
            }
        }
    }
    if (const json::Value * b = at(r, {"health", "balance_run"}); b && b->type == Type::OBJECT) {
        for (size_t i = 0; i < b->names.size(); i++) {
            if (is_false(b->members[i].find("checked"))) continue;
            add("Conservation", b->names[i], b->members[i].find("relative"), step, t, "-");
        }
    }
    if (const json::Value * rep = at(r, {"health", "repairs_run"}); rep && rep->type == Type::OBJECT) {
        for (size_t i = 0; i < rep->names.size(); i++) {
            add("Numerical repairs", rep->names[i], rep->members[i].find("activations"), step, t, "-");
        }
    }
    last_progress = r;
}

void Page::add(const json::Value & r) {
    n_records++;
    const std::string type = str(r.find("type"));
    if (type == "start") {
        start = r;
        end = json::Value{};
        divergence = json::Value{};
        nondimensional = has_resolved = false;
        configuration.clear();
        boundaries.clear();
        if (const json::Value * resolved = at(r, {"provenance", "input", "resolved"})) {
            try {
                const Input input = parse_input_string(resolved->string, "the resolved input");
                nondimensional = input_value(input, "physics.units") == "nondimensional";
                for (const auto & [key, name] : SUMMARY_KEYS) {
                    // -1: a stop condition not set
                    if (const std::string v = input_value(input, key); v != "-1" && v != "-1.0") {
                        configuration.emplace_back(name, v);
                    }
                }
                const auto & entries = input.as_table().entries();
                if (const auto it = entries.find("boundaries"); it != entries.end() && it->second.is_array()) {
                    for (const Input & b : it->second.as_array()) {
                        boundaries.emplace_back(input_value(b, "name"), input_value(b, "type"));
                    }
                }
                has_resolved = true;
            } catch (const std::exception &) {
                // A resolved input this build cannot read is shown as text only
            }
        }
        const json::Value * restart = r.find("restart");
        const uint64_t from = static_cast<uint64_t>(number_of(restart ? restart->find("step") : nullptr));
        if (restart && restart->find("rewound_from")) {
            for (Series & s : all) s.truncate(from);
        }
        Part part;
        part.run_id = str(r.find("run_id"));
        part.start_time = str(at(r, {"provenance", "run", "start_time"}));
        part.first_step = part.last_step = from;
        parts.push_back(part);
    } else if (type == "progress") {
        progress(r);
        if (!parts.empty()) {
            parts.back().last_step = static_cast<uint64_t>(number_of(r.find("step")));
        }
    } else if (type == "end") {
        end = r;
        if (!parts.empty()) {
            parts.back().outcome = str(r.find("outcome"));
            parts.back().last_step = static_cast<uint64_t>(number_of(r.find("step")));
        }
    } else if (type == "divergence") {
        divergence = r;
    } else if (type == "warning") {
        warnings.add(r);
    } else if (type == "error") {
        errors.add(r);
    } else if (type == "event") {
        events.add(r);
    } else if (type == "output") {
        Outputs & o = outputs[str(r.find("writer")) + " (" + str(r.find("kind")) + ")"];
        const uint64_t step = static_cast<uint64_t>(number_of(r.find("step")));
        if (o.n == 0) o.first_step = step;
        o.n++;
        o.last_step = step;
        o.bytes += static_cast<uint64_t>(number_of(r.find("bytes")));
        o.last_path = str(r.find("path"));
        recent_outputs.push_back(r);
        if (recent_outputs.size() > KEPT) recent_outputs.erase(recent_outputs.begin());
    }
}

report::Status Page::status() const {
    using report::Status;
    const std::string outcome = str(end.find("outcome"));
    const Status s = outcome.empty()                                        ? Status::NONE
                     : outcome == "finished" || outcome == "stopped"        ? Status::OK
                     : outcome == "signal" || outcome == "wall_time_limit" ? Status::WARNING
                                                                            : Status::ERROR;
    return s == Status::OK && (warnings.n || errors.n) ? Status::WARNING : s;
}

std::vector<CheckPart> Page::check() const {
    using report::Row;
    using report::Status;
    using Kind = report::Column::Kind;
    std::vector<CheckPart> out;
    const json::Value * c = start.find("check");
    if (!c || c->type != Type::OBJECT) return out;
    auto g3 = [](const json::Value * v) {
        return v && v->type == Type::NUMBER ? format("%.3g", v->number) : std::string("-");
    };
    auto point = [](const json::Value * x) {
        std::string text = "(";
        if (x && x->type == Type::ARRAY) {
            for (size_t i = 0; i < x->array.size(); i++) text += (i ? ", " : "") + format("%.4g", x->array[i].number);
        }
        return text + ") m";
    };

    if (const json::Value * q = c->find("mesh_quality"); q && q->type == Type::OBJECT) {
        CheckPart part{"Mesh quality",
                       "Over every cell, as mallard check measures it; a cell above the limit is questionable. Worst: "
                       "the cell with the largest value, by its global id and center."};
        part.header = {"metric", {"units", Kind::UNITS}, {"max", Kind::VALUE}, {"mean", Kind::VALUE},
                       {"limit", Kind::VALUE}, "cells over limit", "worst"};
        for (size_t i = 0; i < q->names.size(); i++) {
            const json::Value & m = q->members[i];
            const std::string name = str(m.find("name"));
            const Status row = number_of(m.find("above_limit")) > 0.0 ? Status::WARNING : Status::OK;
            part.status = report::worse(part.status, row);
            std::string worst = "-";
            if (const json::Value * w = m.find("worst"); w && w->type == Type::ARRAY && !w->array.empty()) {
                const json::Value & cell = w->array.front();
                worst = "cell " + text_of(cell.find("cell")) + " at " + point(cell.find("x"));
            }
            part.rows.push_back({row, {name, str(m.find("units")), g3(m.find("max")), g3(m.find("mean")),
                                       text_of(m.find("limit")), text_of(m.find("above_limit")), worst}});
        }
        out.push_back(std::move(part));
    }

    if (const json::Value * ts = c->find("time_step"); ts && ts->type == Type::OBJECT) {
        CheckPart part{"Time step", "The CFL number against the time integrator's stability limit for smooth "
                                    "flow: the requested one, and the largest at any progress row so far."};
        part.quantities = true;
        const std::string integrator = str(ts->find("integrator"));
        const json::Value * limit = ts->find("cfl_limit");
        const bool limited = limit && limit->type == Type::NUMBER;
        const std::string target = limited ? "<= " + format("%.4g", limit->number) : "";
        const std::string basis = limited ? integrator + "'s stability limit for smooth flow"
                                          : "no limit known for " + integrator;
        auto row = [&](const std::string & quantity, const double cfl, const std::string & what) {
            const Status s = !limited ? Status::NONE : cfl > limit->number ? Status::WARNING : Status::OK;
            part.status = report::worse(part.status, s);
            part.rows.push_back({s, {quantity, format("%.3g", cfl), "dimensionless", target, what}});
        };
        if (const json::Value * cfl = ts->find("cfl"); cfl && cfl->type == Type::NUMBER) {
            row("CFL number, run.cfl", cfl->number, basis);
        }
        if (const auto it = by_key.find("Time step\ncfl"); it != by_key.end() && all[it->second].rows() > 0) {
            row("CFL number, largest so far", all[it->second].largest(),
                "over " + logging::count(all[it->second].rows(), "progress row") + "; " + basis);
        }
        if (!part.rows.empty()) out.push_back(std::move(part));
    }

    if (const json::Value * res = c->find("resolution"); res && res->type == Type::OBJECT && !res->names.empty()) {
        CheckPart part{"Resolution", "Estimates from the initial state against the targets of the case's regime "
                                     "(DNS, or LES with [les]), as mallard check makes them; value -: not "
                                     "estimated, for the reason its basis gives."};
        part.quantities = true;
        for (size_t i = 0; i < res->names.size(); i++) {
            const json::Value & e = res->members[i];
            const std::string name = str(e.find("status"));
            const Status s = name == "ok" ? Status::OK : name == "warning" ? Status::WARNING
                             : name == "error"                             ? Status::ERROR
                                                                           : Status::NONE;
            part.status = report::worse(part.status, s);
            const json::Value * v = e.find("value");
            const bool known = v && v->type == Type::NUMBER;
            const json::Value * x = e.find("x");
            part.rows.push_back({s, {str(e.find("quantity")), known ? format("%.3g", v->number) : "-",
                                     known ? str(e.find("units")) : "", str(e.find("target")),
                                     str(e.find("basis")) + (x && x->type == Type::ARRAY ? ", at " + point(x) : "")}});
        }
        out.push_back(std::move(part));
    }

    if (const json::Value * m = c->find("memory"); m && m->type == Type::OBJECT) {
        CheckPart part{"Memory", "After setup, of the rank that uses the most."};
        part.quantities = true;
        part.status = Status::NONE;
        auto megabytes = [&](const char * key, const std::string & quantity, const std::string & basis) {
            if (const json::Value * b = m->find(key); b && b->type == Type::NUMBER) {
                part.rows.push_back({Status::NONE, {quantity, format("%.3g", b->number / 1e6), "MB", "", basis}});
            }
        };
        megabytes("host", "host memory", "resident set of the process");
        megabytes("host_peak", "host memory, peak", "during setup");
        const json::Value * total = m->find("device_total");
        megabytes("device", "device memory in use",
                  total && total->type == Type::NUMBER && total->number > 0.0
                      ? format("%.0f%% of the device's ", 100.0 * number_of(m->find("device")) / total->number) +
                            logging::si(total->number) + "B"
                      : "");
        if (!part.rows.empty()) out.push_back(std::move(part));
    }

    if (const json::Value * d = c->find("diagnostics"); d && d->type == Type::ARRAY && !d->array.empty()) {
        CheckPart part{"Diagnostics", "The check's warnings, with their codes."};
        part.header = {"code", "message, then the docs that explain it"};
        for (const json::Value & e : d->array) {
            const Status s = str(e.find("severity")) == "error" ? Status::ERROR : Status::WARNING;
            part.status = report::worse(part.status, s);
            part.rows.push_back({s, {str(e.find("code")), str(e.find("message")) + "\n" + str(e.find("docs"))}});
        }
        out.push_back(std::move(part));
    }
    return out;
}

std::string Page::text(const bool color) const {
    using report::Column;
    using report::Row;
    using report::Status;
    using Kind = Column::Kind;
    report::Text text(color);
    auto with_unit = [&](const json::Value * t) { return text_of(t) + (nondimensional ? "" : " s"); };

    text.section("Outcome", status());
    if (end.type == Type::OBJECT) {
        const std::string reason = str(end.find("reason"));
        text.line("Ended (" + str(end.find("outcome")) + "): " + (reason.empty() ? std::string() : reason + ", ") +
                  "at step " + text_of(end.find("step")) + ", t = " + with_unit(end.find("t")));
        for (const auto & [key, name] :
             {std::pair{"last_snapshot", "Last snapshot"}, {"last_restart", "Last restart"}}) {
            if (const json::Value * f = end.find(key)) {
                text.line(std::string(name) + ": " + str(f->find("path")) + ", step " + text_of(f->find("step")) +
                          ", t = " + with_unit(f->find("t")));
            }
        }
    } else {
        text.line("No end record: still running, or killed (SIGKILL, a lost node, a time limit without warning)");
        if (last_progress.type == Type::OBJECT) {
            text.line("Last progress at step " + text_of(last_progress.find("step")) + ", t = " +
                      with_unit(last_progress.find("t")) + ", " +
                      format("%.1f", 100.0 * number_of(last_progress.find("progress"))) + "% of the run");
        }
    }

    const json::Value * health = last_progress.find("health");
    std::vector<Row> repairs;
    if (const json::Value * rep = health ? health->find("repairs_run") : nullptr; rep && rep->type == Type::OBJECT) {
        for (size_t i = 0; i < rep->names.size(); i++) {
            const json::Value & r = rep->members[i];
            if (number_of(r.find("activations")) <= 0.0) continue;
            repairs.push_back(
                {Status::WARNING, {rep->names[i], text_of(r.find("activations")), text_of(r.find("cells"))}});
        }
    }
    Status health_status = health ? (repairs.empty() ? Status::OK : Status::WARNING) : Status::NONE;
    if (divergence.type == Type::OBJECT) health_status = Status::ERROR;
    text.section("Health", health_status);
    if (!health) {
        text.note("No health record: no progress row yet, or [output] health = false.");
    } else {
        if (const json::Value * m = health->find("margins_run"); m && m->type == Type::OBJECT && !m->names.empty()) {
            std::vector<Row> rows;
            for (size_t i = 0; i < m->names.size(); i++) {
                const json::Value & e = m->members[i];
                if (is_false(e.find("valid"))) continue;
                const std::string u = unit("margins", m->names[i]);
                rows.push_back({Status::NONE, {m->names[i], format("%.4g", number_of(e.find("value"))),
                                               u.empty() ? "-" : u, text_of(e.find("step")),
                                               text_of(e.find("x"))}});
            }
            text.note("Lowest values over the cells at any progress row so far, with where they occurred:");
            text.table({"quantity", {"value", Kind::VALUE}, {"units", Kind::UNITS}, "step", "at"}, rows, false);
        }
        if (const json::Value * a = health->find("activity"); a && a->type == Type::OBJECT && !a->names.empty()) {
            std::vector<Row> rows;
            std::string what;
            for (size_t i = 0; i < a->names.size(); i++) {
                const json::Value & e = a->members[i];
                rows.push_back({Status::NONE, {a->names[i], format("%.2f", 100.0 * number_of(e.find("all"))),
                                               format("%.2f", 100.0 * number_of(e.find("rank_min"))),
                                               format("%.2f", 100.0 * number_of(e.find("rank_max")))}});
                if (const std::string t = activity_text(a->names[i]); !t.empty()) what += "; " + a->names[i] + ": " + t;
            }
            text.blank();
            text.note("Activity at the last progress row, over the whole domain and on the least and most loaded "
                      "rank (the same in a serial run)" + what + ":");
            text.table({"activity", "whole domain [%]", "least loaded rank [%]", "most loaded rank [%]"}, rows, false);
        }
        text.blank();
        if (repairs.empty()) {
            text.line("Numerical repairs: none");
        } else {
            text.note("Numerical repairs since the run started, each a change of the solution to keep it admissible:");
            text.table({"repair", "times", "distinct cells"}, repairs);
        }
        if (const json::Value * b = health->find("balance_run"); b && b->type == Type::OBJECT && !b->names.empty()) {
            std::vector<Row> rows;
            for (size_t i = 0; i < b->names.size(); i++) {
                const json::Value & e = b->members[i];
                const bool ok = !is_false(e.find("checked"));
                rows.push_back({Status::NONE, {b->names[i], ok ? format("%.1e", number_of(e.find("relative"))) : "-",
                                               ok ? "-" : "", str(e.find("note"))}});
            }
            text.blank();
            text.note("Conservation: the change of each total that the boundaries, sources and repairs do not "
                      "explain, relative to the total's scale:");
            text.table({"total", {"unexplained change", Kind::VALUE}, {"units", Kind::UNITS}, "note"}, rows, false);
        }
    }

    const std::vector<CheckPart> checks = check();
    if (!checks.empty()) {
        Status worst = Status::OK;
        for (const CheckPart & part : checks) worst = report::worse(worst, part.status);
        text.section("Check at setup", worst);
        for (size_t i = 0; i < checks.size(); i++) {
            if (i) text.blank();
            text.note(checks[i].title + ": " + checks[i].note);
            if (checks[i].quantities) {
                text.quantities(checks[i].rows);
            } else {
                text.table(checks[i].header, checks[i].rows);
            }
        }
    }

    text.section("Cost");
    std::vector<Row> cost;
    auto quantity = [&](const std::string & name, const std::string & value, const std::string & unit,
                        const std::string & basis = "") {
        cost.push_back({Status::NONE, {name, value, unit, "", basis}});
    };
    if (end.type == Type::OBJECT) {
        const double wall = number_of(end.find("wall"));
        quantity("wall time", format("%.4g", wall), "s", "the whole run");
        for (const auto & [key, name] : {std::pair{"wall_setup", "setup"}, {"wall_stepping", "time stepping"},
                                         {"wall_chemistry", "chemistry"}, {"wall_diagnostics", "diagnostics"},
                                         {"wall_output", "output"}}) {
            if (const json::Value * v = end.find(key)) {
                quantity(name, format("%.4g", number_of(v)), "s",
                         wall > 0.0 ? format("%.1f%% of the wall time", 100.0 * number_of(v) / wall) : "");
            }
        }
        quantity("steps", text_of(end.find("steps")), "steps", "this run's");
        if (const json::Value * r = end.find("rejected_steps")) quantity("rejected steps", text_of(r), "steps");
        if (const json::Value * c = end.find("cells_per_second"); c && c->type == Type::NUMBER) {
            quantity("throughput", format("%.3g", c->number), "cells/s", "time stepping");
        }
    } else if (last_progress.type == Type::OBJECT) {
        quantity("wall time so far", format("%.4g", number_of(last_progress.find("wall"))), "s");
        if (const json::Value * eta = last_progress.find("eta"); eta && eta->type == Type::NUMBER) {
            quantity("remaining", format("%.4g", eta->number), "s", "estimate");
        }
    }
    if (const json::Value * peak = last_progress.find("memory_host_peak"); peak && peak->type == Type::NUMBER) {
        quantity("host memory, peak", format("%.3g", peak->number / 1e6), "MB", "the largest rank");
    }
    if (const json::Value * used = last_progress.find("memory_device"); used && used->type == Type::NUMBER) {
        quantity("device memory in use", format("%.3g", used->number / 1e6), "MB", "the fullest device");
    }
    if (cost.empty()) {
        text.note("No progress row yet.");
    } else {
        text.quantities(cost, false);
    }

    auto messages = [&](const Kept & kept, const char * title, const std::string & noun, Status bad) {
        text.section(title, kept.n ? bad : Status::OK);
        if (!kept.n) {
            text.line("none");
            return;
        }
        const bool cut = kept.n > kept.first.size() + kept.last.size();
        text.line(logging::count(kept.n, noun) +
                  (cut ? ", the first and last " + std::to_string(KEPT) + " (all in mallard.jsonl):" : ":"));
        for (const auto * list : {&kept.first, &kept.last}) {
            for (const json::Value & w : *list) {
                const std::string code = str(w.find("code"));
                text.line("- " + (code.empty() ? "" : code + ": ") + str(w.find("message")));
            }
        }
    };
    messages(warnings, "Warnings", "warning", Status::WARNING);
    if (errors.n) messages(errors, "Errors", "error", Status::ERROR);

    if (divergence.type == Type::OBJECT) {
        text.section("Divergence", Status::ERROR);
        text.line("At step " + text_of(divergence.find("step")) + ", t = " + with_unit(divergence.find("t")) + ": " +
                  text_of(divergence.find("invalid_cells")) + " invalid cells");
        const json::Value * causes = divergence.find("causes");
        if (causes && causes->type == Type::ARRAY && !causes->array.empty()) {
            const json::Value & c = causes->array.front();
            text.line("Most likely cause: " + str(c.find("title")));
            if (const std::string fix = text_of(c.find("fix")); fix != "-") text.line("Fix: " + fix);
        }
    }
    return text.str();
}

std::string Page::html() const {
    std::string h;
    h.reserve(1 << 18);
    const std::string input_file = str(at(start, {"provenance", "input", "file"}));
    const std::string run_id = str(start.find("run_id"));
    h += "<!doctype html>\n<html lang=\"en\">\n<head>\n<meta charset=\"utf-8\">\n"
         "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">\n<meta name=\"generator\" "
         "content=\"Mallard\">\n<title>Run report" +
         (input_file.empty() ? std::string() : ": " + escape(input_file)) + "</title>\n<style>" + STYLE +
         "</style>\n</head>\n<body>\n<main>\n";
    h += "<header><h1>Run report</h1><p class=\"sub\">" +
         escape(input_file.empty() ? std::string("Mallard run") : input_file) +
         (run_id.empty() ? std::string() : " &middot; run " + escape(run_id)) + "</p></header>\n";
    h += "<nav><a href=\"#outcome\">Outcome</a><a href=\"#run\">Run</a><a href=\"#configuration\">Configuration</a>"
         "<a href=\"#check\">Check</a><a href=\"#health\">Health</a><a href=\"#monitors\">Monitors</a><a href=\"#cost\">Cost</a>"
         "<a href=\"#output\">Output</a>";
    if (events.n) h += "<a href=\"#events\">Events</a>";
    if (divergence.type == Type::OBJECT) h += "<a href=\"#divergence\">Divergence</a>";
    h += "</nav>\n";

    // Outcome
    const std::string outcome = str(end.find("outcome"));
    const char * status = report::status_name(this->status());
    h += "<section id=\"outcome\"><div class=\"banner " + std::string(status) + "\">";
    if (end.type == Type::OBJECT) {
        std::string what = outcome;
        if (const std::string cause = str(end.find("cause")); !cause.empty()) what += " (" + cause + ")";
        h += "<strong>" + escape(what) + "</strong> at step " + escape(text_of(end.find("step"))) +
             ", t = " + escape(text_of(end.find("t"))) + (nondimensional ? "" : " s");
        if (const std::string reason = str(end.find("reason")); !reason.empty()) {
            h += "<p>" + escape(reason) + "</p>";
        }
        std::vector<std::pair<std::string, std::string>> rows;
        for (const auto & [key, name] :
             {std::pair{"last_snapshot", "Last snapshot"}, {"last_restart", "Last restart"}}) {
            if (const json::Value * f = end.find(key)) {
                rows.emplace_back(name, str(f->find("path")) + ", step " + text_of(f->find("step")) + ", t = " +
                                            text_of(f->find("t")));
            }
        }
        if (const std::string trigger = str(end.find("trigger")); !trigger.empty()) {
            rows.emplace_back("Trigger", trigger);
        }
        rows.emplace_back("Wall time", seconds_text(number_of(end.find("wall"))));
        rows.emplace_back("Warnings", warnings.n ? logging::count(warnings.n) : "none");
        h += key_values(rows);
    } else {
        h += "<strong>No end record</strong>: the run is still running, or was killed (SIGKILL, a lost node, a time "
             "limit without warning).";
        if (last_progress.type == Type::OBJECT) {
            h += " Last progress at step " + escape(text_of(last_progress.find("step"))) + ", t = " +
                 escape(text_of(last_progress.find("t"))) + ", " +
                 escape(format("%.1f", 100.0 * number_of(last_progress.find("progress")))) + "% of the run.";
        }
    }
    h += "</div></section>\n";

    // Run
    h += "<section id=\"run\"><h2>Run</h2>\n";
    const json::Value * p = start.find("provenance");
    if (!p) {
        h += "<p class=\"note\">No start record: the run log holds no run yet.</p>";
    } else {
        std::string mallard = str(at(*p, {"mallard", "version"}));
        if (const std::string sha = str(at(*p, {"mallard", "git_sha"})); !sha.empty()) {
            mallard += " (" + sha.substr(0, 12) + ")";
        }
        if (const json::Value * dirty = at(*p, {"mallard", "git_dirty"}); dirty && dirty->type == Type::BOOL) {
            if (dirty->boolean) mallard += ", uncommitted changes";
        }
        std::string build = text_of(at(*p, {"build", "dim"})) + "D, " + str(at(*p, {"build", "precision"})) + ", " +
                            str(at(*p, {"build", "backend"}));
        std::string restart_of;
        if (const json::Value * r = start.find("restart_of"); r && r->type == Type::OBJECT) {
            restart_of = str(r->find("file")) + ", step " + text_of(r->find("step")) + ", run " +
                         (str(r->find("run_id")).empty() ? std::string("before 0.10") : str(r->find("run_id")));
        }
        std::string input = input_file;
        if (const std::string sha = str(at(*p, {"input", "sha256"})); !sha.empty()) input += ", SHA-256 " + sha;
        std::string mesh = str(at(*p, {"mesh", "file"}));
        if (mesh.empty()) mesh = str(at(*p, {"mesh", "type"}));
        if (const std::string sha = str(at(*p, {"mesh", "sha256"})); !sha.empty()) mesh += ", SHA-256 " + sha;
        const json::Value * n_cells = last_progress.find("cells");
        const std::string cells = n_cells ? logging::count(uint64_t(number_of(n_cells))) : "";
        h += key_values({
            {"Run id", run_id},
            {"Restarted from", restart_of},
            {"Started", str(at(*p, {"run", "start_time"}))},
            {"Mallard", mallard},
            {"Build", build},
            {"Ranks", text_of(at(*p, {"run", "ranks"})) + " on " + text_of(at(*p, {"run", "hosts"})) + " host(s)"},
            {"Host", str(at(*p, {"run", "host"}))},
            {"Device", str(at(*p, {"run", "device"}))},
            {"Directory", str(at(*p, {"run", "directory"}))},
            {"Input", input},
            {"Mesh", mesh},
            {"Cells", cells},
        });
    }
    if (parts.size() > 1) {
        h += "<h3>Runs of this record</h3><p class=\"note\">A run continued from a restart file appends to the "
             "record: each part is one run.</p>";
        h += head({"run id", "started", "steps", "outcome"});
        for (const Part & part : parts) {
            h += "<tr>" + cell(part.run_id) + cell(part.start_time) +
                 num(logging::count(part.first_step) + " to " + logging::count(part.last_step)) +
                 cell(part.outcome.empty() ? "no end record" : part.outcome) + "</tr>";
        }
        h += TABLE_END;
    }
    h += "</section>\n";

    // Configuration
    h += "<section id=\"configuration\"><h2>Configuration</h2>\n";
    if (has_resolved) {
        h += key_values(configuration);
        if (!boundaries.empty()) {
            h += "<h3>Boundaries</h3>" + head({"zone", "type"});
            for (const auto & [zone, type] : boundaries) h += "<tr>" + cell(zone) + cell(type) + "</tr>";
            h += TABLE_END;
        }
    }
    if (const std::string resolved = str(at(start, {"provenance", "input", "resolved"})); !resolved.empty()) {
        h += "<details><summary>The resolved input: every key the run read, defaults filled in</summary><pre>" +
             escape(resolved) + "</pre></details>\n";
    } else if (start.type == Type::OBJECT) {
        h += "<p class=\"note\">The run block records no resolved input (runs before Mallard 0.9.1).</p>";
    }
    h += "</section>\n";

    // Check at setup
    if (const std::vector<CheckPart> checks = check(); !checks.empty()) {
        report::Status worst = report::Status::OK;
        for (const CheckPart & part : checks) worst = report::worse(worst, part.status);
        h += "<section id=\"check\"><h2>Check at setup " +
             status_span(report::status_name(worst), report::status_word(worst)) +
             "</h2>\n<p class=\"note\">What the setup measured of the mesh and the case, in the terms of "
             "<code>mallard check</code>, which also estimates the memory and cost of other partitions.</p>\n";
        for (const CheckPart & part : checks) {
            h += "<h3>" + escape(part.title) +
                 (part.status == report::Status::NONE
                      ? std::string()
                      : " " + status_span(report::status_name(part.status), report::status_word(part.status))) +
                 "</h3><p class=\"note\">" + escape(part.note) + "</p>";
            std::vector<std::string> names = {"status"};
            if (part.quantities) {
                names.insert(names.end(), {"quantity", "value", "units", "target", "basis"});
            } else {
                for (const report::Column & column : part.header) names.push_back(column.name);
            }
            h += head(names);
            for (const report::Row & row : part.rows) {
                h += "<tr><td>" + status_span(report::status_name(row.status), report::status_word(row.status)) +
                     "</td>";
                for (const std::string & text : row.cells) {
                    std::string shown_text = escape(text);
                    for (size_t at = 0; (at = shown_text.find('\n', at)) != std::string::npos;) {
                        shown_text.replace(at, 1, "<br>");
                    }
                    h += "<td>" + shown_text + "</td>";
                }
                h += "</tr>";
            }
            h += TABLE_END;
        }
        h += "</section>\n";
    }

    // Health
    h += "<section id=\"health\"><h2>Health</h2>\n";
    const json::Value * health = last_progress.find("health");
    if (!health) {
        h += "<p class=\"note\">No health record: no progress row yet, or <code>[output] health = false</code>.</p>";
    } else {
        if (const json::Value * m = health->find("margins_run"); m && m->type == Type::OBJECT && !m->names.empty()) {
            h += "<h3>Lowest values</h3><p class=\"note\">The lowest density, pressure and temperature over the "
                 "cells at any progress row so far, with the step and the cell center where each occurred. A value "
                 "falling toward zero is the usual first sign of a run about to fail.</p>" +
                 head({"quantity", "value", "units", "step", "at"});
            for (size_t i = 0; i < m->names.size(); i++) {
                const json::Value & e = m->members[i];
                if (is_false(e.find("valid"))) continue;
                h += "<tr>" + cell(m->names[i]) + num(text_of(e.find("value"))) +
                     cell(unit("margins", m->names[i])) + num(text_of(e.find("step"))) +
                     cell(text_of(e.find("x"))) + "</tr>";
            }
            h += TABLE_END;
        }
        if (const json::Value * a = health->find("activity"); a && a->type == Type::OBJECT && !a->names.empty()) {
            h += "<h3>Activity</h3><p class=\"note\">How much of the domain the shock-capturing numerics act on, "
                 "at the last progress row: over the whole domain and on the ranks with the least and the most of "
                 "it. Very different ranks mean an unbalanced load, since those cells cost more; in a serial run "
                 "the three columns match.</p>" +
                 head({"activity", "what is counted", "whole domain [%]", "least loaded rank [%]",
                       "most loaded rank [%]"});
            for (size_t i = 0; i < a->names.size(); i++) {
                const json::Value & s = a->members[i];
                h += "<tr>" + cell(a->names[i]) + cell(activity_text(a->names[i]));
                for (const char * k : {"all", "rank_min", "rank_max"}) {
                    h += num(format("%.2f", 100.0 * number_of(s.find(k))));
                }
                h += "</tr>";
            }
            h += TABLE_END;
        }
        if (const json::Value * rep = health->find("repairs_run"); rep && rep->type == Type::OBJECT) {
            std::string rows;
            for (size_t i = 0; i < rep->names.size(); i++) {
                const json::Value & r = rep->members[i];
                if (number_of(r.find("activations")) <= 0.0) continue;
                rows += "<tr>" + cell(rep->names[i]) + num(text_of(r.find("activations"))) +
                        num(text_of(r.find("cells"))) + "</tr>";
            }
            h += "<h3>Numerical repairs</h3>";
            h += rows.empty() ? "<p>" + status_span("ok", "ok") + " None since the run started.</p>"
                              : "<p>" + status_span("warning", "WARN") +
                                    " Repairs since the run started: each changed the solution of a cell to keep "
                                    "its density, pressure or mass fractions physical. A few are harmless; many, or "
                                    "a growing number, point at a time step or mesh problem.</p>" +
                                    head({"repair", "times", "distinct cells"}) + rows + TABLE_END;
        }
        if (const json::Value * b = health->find("balance_run"); b && b->type == Type::OBJECT && !b->names.empty()) {
            h += "<h3>Conservation</h3><p class=\"note\">The conservation ledger: the change of each total over "
                 "the run that the boundary fluxes, sources and repairs do not explain, relative to the total's "
                 "scale. A healthy run keeps it at round-off; Mallard warns above 1e5 machine epsilons, and a value "
                 "near that is a bug to report, not a setting to tune.</p>" +
                 head({"total", "unexplained change [-]", "note"});
            for (size_t i = 0; i < b->names.size(); i++) {
                const json::Value & e = b->members[i];
                const bool ok = !is_false(e.find("checked"));
                h += "<tr>" + cell(b->names[i]) + num(ok ? format("%.1e", number_of(e.find("relative"))) : "-") +
                     cell(str(e.find("note"))) + "</tr>";
            }
            h += TABLE_END;
        }
        if (const json::Value * d = health->find("element_drift")) {
            h += "<p>Elements: the largest relative change of an element's mass over a chemistry call, which "
                 "chemistry must keep at round-off: " +
                 escape(format("%.1e", number_of(d))) + "</p>";
        }
    }
    auto messages = [&](const Kept & kept, const char * title, const char * word, const char * cls) {
        h += "<h3>" + std::string(title) + "</h3>";
        if (!kept.n) {
            h += "<p>" + status_span("ok", "ok") + " None.</p>";
            return;
        }
        h += "<p>" + status_span(cls, word) + " " + logging::count(kept.n) +
             (kept.n > kept.first.size() + kept.last.size()
                  ? ", the first and last " + std::to_string(KEPT) + " below (all in mallard.jsonl)"
                  : std::string()) +
             ".</p><ul>";
        for (const auto * list : {&kept.first, &kept.last}) {
            for (const json::Value & w : *list) {
                const std::string code = str(w.find("code"));
                h += "<li>" + (code.empty() ? "" : "<code>" + escape(code) + "</code> ") + escape(str(w.find("message"))) +
                     "</li>";
            }
        }
        h += "</ul>";
    };
    messages(warnings, "Warnings", "WARN", "warning");
    if (errors.n) messages(errors, "Errors", "ERROR", "error");
    h += "</section>\n";

    // Monitors: the series as embedded data, plotted in the browser, and the last values as a table
    std::vector<std::string> groups;
    std::map<std::string, std::vector<const Series *>> plotted;
    size_t n_plotted = 0;
    for (const Series & s : all) {
        if (n_plotted >= MAX_PLOTTED || s.rows() == 0) continue;
        if (!plotted.contains(s.group)) groups.push_back(s.group);
        plotted[s.group].push_back(&s);
        n_plotted++;
    }
    h += "<section id=\"monitors\"><h2>Monitors</h2>\n<p class=\"note\">The quantities of every progress row "
         "against simulated time. A long run's rows are combined in up to " +
         std::to_string(BUCKETS) +
         " points per plot: the line goes through the last value of each point's rows, the shaded band spans their "
         "least and largest, so a short spike still shows. Hover over a plot to read its values.</p>\n"
         "<p id=\"plots-nojs\" class=\"note\">The plots need JavaScript; the table of last values below does "
         "not.</p>\n";
    std::string data = "{\"groups\":[";
    std::map<std::string, size_t> axis_of;  // series of the same rows share their times and steps
    std::vector<std::string> axes;
    for (size_t g = 0; g < groups.size(); g++) {
        h += "<h3>" + escape(groups[g]) + "</h3>";
        if (const std::string note = group_note(groups[g]); !note.empty()) {
            h += "<p class=\"note\">" + escape(note) + "</p>";
        }
        h += "<div class=\"plots\" id=\"plots-" + std::to_string(g) + "\"></div>\n";
        data += std::string(g ? "," : "") + "{\"title\":" + json::quote(groups[g]) + ",\"series\":[";
        const auto & list = plotted[groups[g]];
        for (size_t i = 0; i < list.size(); i++) {
            const Series & s = *list[i];
            const bool band = s.width() > 1;
            std::string ts, ys, lo, hi, steps;
            for (const Series::Bucket & b : s.buckets()) {
                const char * sep = ts.empty() ? "" : ",";
                ts += sep + format("%.6g", b.t);
                ys += sep + format("%.5g", b.last);
                steps += sep + std::to_string(b.step);
                if (band) {
                    lo += sep + format("%.5g", b.min);
                    hi += sep + format("%.5g", b.max);
                }
            }
            const std::string axis = "{\"t\":[" + ts + "],\"step\":[" + steps + "]}";
            const auto [it, added] = axis_of.emplace(axis, axes.size());
            if (added) axes.push_back(axis);
            data += std::string(i ? "," : "") + "{\"name\":" + json::quote(s.name) +
                    ",\"unit\":" + json::quote(s.unit) + ",\"label\":" + json::quote(label(s.name, s.unit)) +
                    ",\"title\":" + json::quote(caption(s)) +
                    ",\"axis\":" + std::to_string(it->second) + ",\"y\":[" + ys + "]" +
                    (band ? ",\"lo\":[" + lo + "],\"hi\":[" + hi + "]" : "") + "}";
        }
        data += "]}";
    }
    data += "],\"xlabel\":" + json::quote(label("t", unit("", "t").empty() ? "s" : unit("", "t"))) + ",\"axes\":[";
    for (size_t i = 0; i < axes.size(); i++) data += (i ? "," : "") + axes[i];
    data += "]}";
    if (all.empty()) h += "<p class=\"note\">No progress row yet.</p>";
    if (!all.empty()) {
        h += "<h3>Last values</h3><p class=\"note\">Each plotted quantity's value at the last progress row, and "
             "its least and largest over the run.</p>" +
             head({"group", "quantity", "what it is", "units", "last", "least", "largest", "rows"});
        for (const Series & s : all) {
            if (s.rows() == 0) continue;
            h += "<tr>" + cell(s.group) + cell(s.name) + cell(caption(s)) + cell(s.unit) +
                 num(format("%.6g", s.buckets().back().last)) +
                 num(format("%.6g", s.least())) + num(format("%.6g", s.largest())) + num(logging::count(s.rows())) +
                 "</tr>";
        }
        h += TABLE_END;
        if (all.size() > n_plotted) {
            h += "<p class=\"note\">" + std::to_string(MAX_PLOTTED) + " quantities plotted; the others are in this "
                 "table and in mallard.jsonl.</p>";
        }
    }
    h += "</section>\n";

    // Cost
    h += "<section id=\"cost\"><h2>Cost</h2>\n";
    if (end.type == Type::OBJECT) {
        const double wall = number_of(end.find("wall"));
        auto share = [&](const char * key) {
            const double s = number_of(end.find(key));
            return seconds_text(s) + (wall > 0.0 ? format(" (%.1f%%)", 100.0 * s / wall) : std::string());
        };
        std::vector<std::pair<std::string, std::string>> rows = {
            {"Wall time", seconds_text(wall)},
            {"Setup", share("wall_setup")},
            {"Time stepping", share("wall_stepping")},
            {"Chemistry", end.find("wall_chemistry") ? share("wall_chemistry") : ""},
            {"Diagnostics", share("wall_diagnostics")},
            {"Output", share("wall_output")},
            {"Steps", text_of(end.find("steps"))},
            {"Rejected steps", end.find("rejected_steps") ? text_of(end.find("rejected_steps")) : ""},
            {"Throughput", end.find("cells_per_second") && !end.find("cells_per_second")->is_null()
                               ? format("%.3g cells/s", number_of(end.find("cells_per_second")))
                               : ""},
        };
        h += key_values(rows);
    } else if (last_progress.type == Type::OBJECT) {
        h += key_values({{"Wall time so far", seconds_text(number_of(last_progress.find("wall")))},
                         {"Remaining (estimate)", seconds_text(number_of(last_progress.find("eta"), -1.0))}});
    }
    if (last_progress.type == Type::OBJECT) {
        h += key_values({{"Host memory, peak", bytes_text(number_of(last_progress.find("memory_host_peak"), -1.0))},
                         {"Device memory in use", bytes_text(number_of(last_progress.find("memory_device"), -1.0))},
                         {"Device memory", bytes_text(number_of(last_progress.find("memory_device_total"), -1.0))}});
    }
    h += "</section>\n";

    // Output
    h += "<section id=\"output\"><h2>Output</h2>\n";
    if (outputs.empty()) {
        h += "<p class=\"note\">No output recorded.</p>";
    } else {
        h += "<p class=\"note\">The output index of the run record: every snapshot completely written, by writer. "
             "<code>mallard info DIR</code> checks the files against it.</p>" +
             head({"writer (kind)", "snapshots", "first step", "last step", "size", "last"});
        for (const auto & [name, o] : outputs) {
            h += "<tr>" + cell(name) + num(logging::count(o.n)) + num(logging::count(o.first_step)) +
                 num(logging::count(o.last_step)) + num(bytes_text(double(o.bytes))) + cell(o.last_path) + "</tr>";
        }
        h += TABLE_END + std::string("<h3>Latest</h3>") + head({"step", "t", "kind", "reason", "size", "path"});
        for (auto it = recent_outputs.rbegin(); it != recent_outputs.rend(); ++it) {
            h += "<tr>" + num(text_of(it->find("step"))) + num(text_of(it->find("t"))) + cell(str(it->find("kind"))) +
                 cell(str(it->find("reason"))) + num(bytes_text(number_of(it->find("bytes")))) +
                 cell(str(it->find("path"))) + "</tr>";
        }
        h += TABLE_END;
    }
    h += "</section>\n";

    // Events
    if (events.n) {
        h += "<section id=\"events\"><h2>Events</h2><p class=\"note\">Requests acted on: run control and triggers" +
             (events.n > events.first.size() + events.last.size()
                  ? ", the first and last " + std::to_string(KEPT) + " of " + logging::count(events.n)
                  : std::string()) +
             ".</p>" + head({"step", "t", "kind", "what"});
        for (const auto * list : {&events.first, &events.last}) {
            for (const json::Value & e : *list) {
                h += "<tr>" + num(text_of(e.find("step"))) + num(text_of(e.find("t"))) + cell(str(e.find("kind"))) +
                     cell(str(e.find("text"))) + "</tr>";
            }
        }
        h += TABLE_END + std::string("</section>\n");
    }

    // Divergence
    if (divergence.type == Type::OBJECT) {
        const json::Value & d = divergence;
        h += "<section id=\"divergence\"><h2>Divergence " + status_span("error", "ERROR") + "</h2>\n";
        const json::Value * last_valid = d.find("last_valid");
        h += key_values({
            {"Step", text_of(d.find("step"))},
            {"Time", text_of(d.find("t"))},
            {"Invalid cells", text_of(d.find("invalid_cells"))},
            {"Failed first", text_of(d.find("failed"))},
            {"Extent", text_of(d.find("box"))},
            {"Last valid snapshot", last_valid && last_valid->type == Type::OBJECT
                                        ? str(last_valid->find("file")) + ", step " + text_of(last_valid->find("step"))
                                        : ""},
            {"Snapshot of the failure", text_of(d.find("snapshot"))},
        });
        const json::Value * causes = d.find("causes");
        if (causes && causes->type == Type::ARRAY && !causes->array.empty()) {
            h += "<h3>Likely causes, most likely first</h3>" + head({"cause", "score", "evidence", "fix"});
            for (const json::Value & c : causes->array) {
                h += "<tr>" + cell(str(c.find("title"))) + num(text_of(c.find("score"))) +
                     cell(text_of(c.find("evidence"))) + cell(text_of(c.find("fix"))) + "</tr>";
            }
            h += TABLE_END;
        }
        if (const json::Value * ts = d.find("time_step"); ts && ts->type == Type::OBJECT) {
            std::vector<std::pair<std::string, std::string>> rows;
            for (size_t i = 0; i < ts->names.size(); i++) rows.emplace_back(ts->names[i], text_of(&ts->members[i]));
            h += "<h3>Time step</h3>" + key_values(rows);
        }
        if (const json::Value * c = d.find("cell"); c && c->type == Type::OBJECT) {
            std::vector<std::pair<std::string, std::string>> rows;
            for (size_t i = 0; i < c->names.size(); i++) rows.emplace_back(c->names[i], text_of(&c->members[i]));
            h += "<h3>The first failed cell</h3>" + key_values(rows);
        }
        if (const json::Value * zones = d.find("zones"); zones && zones->type == Type::ARRAY && !zones->array.empty()) {
            h += "<h3>Boundary zones</h3>" + head({"zone", "type", "near", "compared"});
            for (const json::Value & z : zones->array) {
                std::string compared;
                for (const char * k : {"inadmissible", "ratio", "jump"}) {
                    if (const json::Value * v = z.find(k)) {
                        compared += (compared.empty() ? "" : ", ") + std::string(k) + " " + text_of(v);
                    }
                }
                h += "<tr>" + cell(str(z.find("name"))) + cell(str(z.find("type"))) + cell(text_of(z.find("near"))) +
                     cell(compared) + "</tr>";
            }
            h += TABLE_END;
        }
        if (const json::Value * hist = d.find("history"); hist && hist->type == Type::ARRAY && !hist->array.empty()) {
            std::vector<std::string> names;
            if (hist->array.front().type == Type::OBJECT) names = hist->array.front().names;
            h += "<h3>The last progress rows</h3>" + head(names);
            for (const json::Value & row : hist->array) {
                h += "<tr>";
                for (const std::string & n : names) h += num(text_of(row.find(n)));
                h += "</tr>";
            }
            h += TABLE_END;
        }
        h += "</section>\n";
    }

    // Embedded data: "</" cannot end the script element early
    std::string safe;
    safe.reserve(data.size());
    for (size_t i = 0; i < data.size(); i++) {
        if (data[i] == '<' && i + 1 < data.size() && data[i + 1] == '/') {
            safe += "<\\";
        } else {
            safe += data[i];
        }
    }
    h += "<script type=\"application/json\" id=\"mallard-series\">" + safe + "</script>\n<script>" + PLOT_JS +
         "</script>\n</main>\n</body>\n</html>\n";
    return h;
}

Page from_file(const std::string & run_log) {
    Page page;
    std::ifstream in(run_log);
    for (std::string line; std::getline(in, line);) {
        try {
            page.add(json::parse(line));
        } catch (const std::exception &) {
            // A last line cut short by a run that was killed
        }
    }
    return page;
}

void write(const std::string & path, const std::string & html) {
    create_parent_directories("the run report", path);
    const std::string partial = partial_name(path);
    {
        std::ofstream out(partial, std::ios::binary | std::ios::trunc);
        out << html;
        if (!out.good()) throw cannot_write("the run report", partial);
    }
    publish_file("the run report", path);
}

} // namespace run_page

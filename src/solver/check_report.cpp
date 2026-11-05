/**
 * @file check_report.cpp
 * @brief The report of mallard check, as text and as JSON, from the same CheckReport.
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 */

#include <algorithm>
#include <cctype>
#include <cmath>
#include <iostream>

#include "check.h"
#include "comm.h"
#include "json.h"
#include "log.h"
#include "report.h"

using report::Row;
using report::Status;

void CheckReport::error(const std::string & code, const std::string & message) {
    diagnostics.push_back({Status::ERROR, code, message});
}

void CheckReport::warning(const std::string & code, const std::string & message) {
    diagnostics.push_back({Status::WARNING, code, message});
}

std::vector<std::string> CheckReport::errors() const {
    std::vector<std::string> list;
    for (const CheckDiagnostic & d : diagnostics) {
        if (d.severity == Status::ERROR) list.push_back(d.message);
    }
    return list;
}

std::vector<std::string> CheckReport::warnings() const {
    std::vector<std::string> list;
    for (const CheckDiagnostic & d : diagnostics) {
        if (d.severity == Status::WARNING) list.push_back(d.message);
    }
    return list;
}

Status CheckReport::status() const {
    Status s = Status::OK;
    for (const CheckDiagnostic & d : diagnostics) s = report::worse(s, d.severity);
    return s;
}

std::string CheckDiagnostic::docs() const {
    std::string anchor = code;
    std::transform(anchor.begin(), anchor.end(), anchor.begin(), [](unsigned char c) { return std::tolower(c); });
    return "docs/guide/input_files.md#" + anchor;
}

namespace {

using logging::count;
using logging::format;

std::string g4(const double v) { return format("%.4g", v); }

/** @brief "3.96 GB", "208 B". */
std::string bytes(const double b) {
    std::string text = logging::si(b);
    if (!text.empty() && std::isalpha(static_cast<unsigned char>(text.back()))) {
        const char prefix = text.back();
        text.pop_back();
        return text + " " + prefix + "B";
    }
    return text + " B";
}

/** @brief A number of steps: whole above 1000 ("10,339"), else 4 significant digits. */
std::string steps(const double n) { return n >= 1000.0 ? count(static_cast<uint64_t>(std::llround(n))) : g4(n); }

std::string plural(const uint64_t n, const std::string & noun) {
    return count(n) + " " + noun + (n == 1 ? "" : "s");
}

template <size_t N>
std::string point(const std::array<double, N> & x) {
    std::string text = "(";
    for (size_t i = 0; i < N_DIM; i++) text += (i ? ", " : "") + g4(x[i]);
    return text + ") m";
}

std::string result_line(const CheckReport & r) {
    const size_t n_errors = r.errors().size(), n_warnings = r.warnings().size();
    return std::string("Result: ") + report::status_word(r.status()) + ", " + plural(n_errors, "error") + " and " +
           plural(n_warnings, "warning") + (n_errors > 0 ? " (exit status 1)" : " (exit status 0)");
}

size_t n_of(const CheckReport & r, const std::string & code) {
    return static_cast<size_t>(std::count_if(r.diagnostics.begin(), r.diagnostics.end(),
                                             [&](const CheckDiagnostic & d) { return d.code == code; }));
}

/** @brief A table of quantities: status, quantity, value, units, target, basis; columns empty in every row left out. */
void quantities(report::Text & text, const std::vector<Row> & rows) {
    const std::vector<std::string> header = {"quantity", "value", "units", "target", "basis"};
    std::vector<bool> used(header.size(), false);
    for (const Row & row : rows) {
        for (size_t c = 0; c < row.cells.size(); c++) used[c] = used[c] || !row.cells[c].empty();
    }
    std::vector<std::string> kept_header;
    for (size_t c = 0; c < header.size(); c++) {
        if (used[c]) kept_header.push_back(header[c]);
    }
    std::vector<Row> kept;
    for (const Row & row : rows) {
        Row k{row.status, {}};
        for (size_t c = 0; c < row.cells.size(); c++) {
            if (used[c]) k.cells.push_back(row.cells[c]);
        }
        kept.push_back(std::move(k));
    }
    text.table(kept_header, kept);
}

void overview(report::Text & text, const CheckReport & r) {
    text.section("Check", r.status());
    text.line(result_line(r));
    text.line("Input: " + (r.input_file.empty() ? std::string("the input") : r.input_file) +
              (r.setup_done ? ", set up on " + plural(uint64_t(r.ranks), "rank")
                            : ": the setup stopped on an error, so the report covers what it got to"));
    text.line("Units: SI (m, s, kg, K) wherever a column or a value names none; bytes with decimal prefixes "
              "(1 GB = 10^9 B)");
}

void input_section(report::Text & text, const CheckReport & r) {
    const size_t n_unused = n_of(r, check_code::KEY_UNUSED), n_deprecated = n_of(r, check_code::KEY_DEPRECATED);
    const bool stopped = n_of(r, check_code::INPUT) + n_of(r, check_code::SETUP) > 0;
    const Status setup = stopped ? Status::ERROR : Status::OK;
    std::vector<Row> rows;
    rows.push_back({setup, {"setup", stopped ? "stopped on an error: see " + std::string(n_of(r, check_code::INPUT)
                                                                                        ? check_code::INPUT
                                                                                        : check_code::SETUP)
                                             : "completed: mesh, partition, boundaries, numerics, initial state"}});
    rows.push_back({n_unused ? Status::ERROR : Status::OK,
                    {"unknown or unused keys",
                     n_unused ? plural(n_unused, "key") + ": see " + check_code::KEY_UNUSED
                              : (r.setup_done ? "none: every key is used" : "none found up to the error")}});
    rows.push_back({n_deprecated ? Status::WARNING : Status::OK,
                    {"deprecated keys",
                     n_deprecated ? plural(n_deprecated, "key") + ": see " + check_code::KEY_DEPRECATED : "none"}});
    Status s = setup;
    for (const Row & row : rows) s = report::worse(s, row.status);
    text.section("Input", s);
    text.table({"check", "result"}, rows);
}

void boundaries_section(report::Text & text, const CheckReport & r) {
    text.section("Boundaries", Status::OK);
    text.note("Every boundary face has a condition: " + plural(r.boundaries.size(), "zone") +
              " (periodic zones are paired, not listed).");
    if (r.boundaries.empty()) return;
    std::vector<Row> rows;
    for (const CheckBoundary & b : r.boundaries) {
        std::string details = b.details;
        if (!b.turbulence.empty()) details += (details.empty() ? "" : "\n") + ("inflow turbulence: " + b.turbulence);
        rows.push_back({Status::OK, {b.zone, b.condition, count(b.faces), details.empty() ? "-" : details}});
    }
    text.table({"zone", "condition", "faces", "details"}, rows);
}

std::string measure_text(const CheckRegion & g) {
    const std::string value = logging::real(g.measure);
    switch (g.dimension) {
        case 0: return value + " points";
        case 1: return value + " m";
        default: break;
    }
    if (g.dimension < N_DIM) return value + " m^2";
    if (N_DIM == 2) return value + (g.per_radian ? " m^3 per radian" : " m^2");
    return value + " m^3";
}

void regions_section(report::Text & text, const CheckReport & r) {
    if (r.regions.empty()) return;
    std::vector<Row> rows;
    Status s = Status::OK;
    for (const CheckRegion & g : r.regions) {
        static const char * kinds[4] = {"points", "curve", "surface", ""};
        const std::string kind = g.dimension == N_DIM ? "filled" : kinds[g.dimension];
        const std::string made = g.of_mesh                 ? "the mesh's"
                                 : g.dimension == N_DIM    ? "cell fractions"
                                 : g.exact                 ? "exact"
                                                           : "extracted, resolution " + std::to_string(g.resolution);
        const Status row = g.cells > 0 ? Status::OK : Status::ERROR;
        s = report::worse(s, row);
        rows.push_back({row, {g.name, kind, measure_text(g), count(g.cells), made}});
    }
    text.section("Regions", s);
    text.note("The mesh's own (the domain, its boundary and cell zones) and every region the case uses; cells: the "
              "cells each touches. A region of the case that touches no cell is an error.");
    text.table({"region", "kind", "measure", "cells", "made from"}, rows);
    if (!r.view_file.empty()) text.line("Case view (every region and zone, for ParaView): " + r.view_file);
}

void quality_section(report::Text & text, const CheckReport & r) {
    constexpr std::array<const char *, MeshQuality::N_METRICS> UNITS = {"deg", "dimensionless", "dimensionless",
                                                                         "dimensionless"};
    std::vector<Row> rows;
    std::vector<Row> worst;
    Status s = Status::OK;
    for (size_t m = 0; m < MeshQuality::N_METRICS; m++) {
        const QualitySummary & q = r.quality[m];
        const Status row = q.n_over_limit > 0 ? Status::WARNING : Status::OK;
        s = report::worse(s, row);
        rows.push_back({row, {MeshQuality::NAMES[m], UNITS[m], format("%.3g", q.max), format("%.3g", q.mean),
                              logging::real(MeshQuality::LIMITS[m]), count(q.n_over_limit)}});
        if (q.max < MeshQuality::NOTABLE[m]) continue;
        for (const QualityCell & c : q.worst) {
            std::vector<std::string> cells = {MeshQuality::NAMES[m], format("%.3g", c.value), count(c.cell)};
            for (size_t i = 0; i < N_DIM; i++) cells.push_back(g4(c.x[i]));
            worst.push_back({Status::NONE, cells});
        }
    }
    text.section("Mesh quality", s);
    text.note("Over every cell; a cell above the limit is questionable.");
    text.table({"metric", "units", "max", "mean", "limit", "cells over limit"}, rows);
    if (!worst.empty()) {
        text.blank();
        text.note("Worst cells of the metrics at or above their notable value (non-orthogonality 10 deg, skewness "
                  "0.05, aspect ratio 10, volume ratio 2), worst first:");
        std::vector<std::string> header = {"metric", "value", "cell (global id)"};
        for (size_t i = 0; i < N_DIM; i++) header.push_back(std::string(1, "xyz"[i]) + " [m]");
        text.table(header, worst, false);
    }
    if (!r.quality_file.empty()) text.line("Quality fields: " + r.quality_file);
}

void time_step_section(report::Text & text, const CheckReport & r) {
    const std::string basis = r.dt_limit == "CFL"      ? "run.cfl times the time step at CFL 1"
                              : r.dt_limit == "sponge" ? "the sponges' stable step, below run.cfl's"
                                                       : "run.dt";
    const bool limited = std::isfinite(r.cfl_limit);
    const Status cfl = !limited ? Status::NONE : r.cfl > r.cfl_limit ? Status::WARNING : Status::OK;
    std::vector<Row> rows = {
        {Status::NONE, {"time step dt", g4(r.dt), "s", "", basis}},
        {cfl,
         {"CFL number", format("%.3g", r.cfl), "dimensionless", limited ? "<= " + g4(r.cfl_limit) : "",
          limited ? r.integrator + "'s stability limit for smooth flow" : "no limit known for " + r.integrator}},
        {Status::NONE, {"time step at CFL 1", g4(r.dt_cfl1), "s", "", "the smallest over the cells"}},
        {Status::NONE, {"CFL-limiting cell", count(r.limiting_cell), "global id", "", "at " + point(r.limiting_x)}},
        {Status::NONE,
         {"cells within 10% of its step", count(r.n_region_cells), "cells", "",
          "in " + point(r.region_lo) + " to " + point(r.region_hi)}},
    };
    text.section("Time step", cfl == Status::NONE ? Status::OK : cfl);
    text.note("At the initial state, t = " + g4(r.t) + " s.");
    quantities(text, rows);
}

void resolution_section(report::Text & text, const CheckReport & r) {
    if (r.resolution.empty()) return;
    std::vector<Row> rows;
    Status s = Status::OK;
    for (const CheckEstimate & e : r.resolution) {
        s = report::worse(s, e.status);
        const bool known = !std::isnan(e.value);
        rows.push_back({e.status,
                        {e.quantity, known ? format("%.3g", e.value) : "-", known ? e.units : "", e.target,
                         e.basis + (e.located ? ", at " + point(e.x) : "")}});
    }
    text.section("Resolution", s);
    text.note("Estimates from the initial state, against the targets of the case's regime (DNS, or LES with "
              "[les]); value -: not estimated, for the reason its basis gives. See "
              "docs/guide/input_files.md#resolution.");
    quantities(text, rows);
}

void memory_section(report::Text & text, const CheckReport & r) {
    const double cells = static_cast<double>(std::max<uint64_t>(r.local_cells, 1));
    const bool target = r.target_ranks != r.ranks;
    Status s = Status::NONE;
    std::string device = "-";
    if (r.device_capacity > 0) {
        s = n_of(r, check_code::MEMORY_DEVICE) ? Status::ERROR : Status::OK;
        device = format("%.0f%% of %s", 100.0 * double(r.target_bytes_peak) / double(r.device_capacity),
                        bytes(double(r.device_capacity)).c_str());
    }
    std::vector<Row> rows = {{target ? Status::NONE : s,
                              {"this check", std::to_string(r.ranks), count(r.local_owned),
                               count(r.local_cells - r.local_owned), bytes(double(r.bytes)), bytes(double(r.bytes_peak)),
                               target ? "-" : device}}};
    if (target) {
        rows.push_back({s,
                        {"--ranks " + std::to_string(r.target_ranks), std::to_string(r.target_ranks),
                         count(r.target_owned), count(r.target_halo), bytes(double(r.target_bytes)),
                         bytes(double(r.target_bytes_peak)), device}});
    }
    text.section("Memory", s == Status::NONE ? Status::OK : s);
    text.note(std::string("Kokkos allocations in ") + (r.device_memory ? "device" : "host") +
              " memory per rank, of the rank with the largest peak (with --ranks from a serial check, scaled to the "
              "largest part the partitioner makes); peak: during the setup" +
              (r.steps_timed > 0 ? " and the timed steps" : "") + ".");
    text.table({"run", "ranks", "owned cells", "halo cells", "after setup", "peak", "peak of the device"}, rows);

    const std::string steady = r.steps_timed > 0 ? "at the steps' peak" : "after setup";
    std::vector<Row> categories;
    uint64_t total_setup = 0, total_steady = 0;
    for (size_t c = 0; c < memory::N_CATEGORIES; c++) {
        total_setup += r.setup_bytes[c];
        total_steady += r.steady_bytes[c];
        const double a = double(r.setup_bytes[c]), b = double(r.steady_bytes[c]);
        if (a < 0.0005 * double(r.bytes_peak) && b < 0.0005 * double(r.bytes_peak)) continue;
        const bool per_cell = memory::scales_with_cells(memory::Category(c));
        categories.push_back({Status::NONE,
                              {memory::category_name(memory::Category(c)), bytes(a), per_cell ? bytes(a / cells) : "-",
                               bytes(b), per_cell ? bytes(b / cells) : "-", per_cell ? "yes" : "no"}});
    }
    categories.push_back({Status::NONE,
                          {"total", bytes(double(total_setup)), "-",
                           bytes(double(total_steady)), "-", ""}});
    if (r.outside_kokkos > 0) {
        categories.push_back({Status::NONE,
                              {"outside Kokkos (measured)", "-", "-", bytes(double(r.outside_kokkos)), "-",
                               "no: CUDA's context and kernels' local memory, the most measured"}});
    }
    text.blank();
    text.note("By category, of the same rank; per cell: per owned and halo cell of the rank (" +
              count(r.local_cells) + "); categories below 0.05% of the peak are left out.");
    text.table({"category", "at the setup's peak", "per cell", steady, "per cell", "proportional to the cells"},
               categories, false);
}

void sizing_section(report::Text & text, const CheckReport & r) {
    if (r.sizing_name.empty()) return;
    const std::string unit = r.sizing_gpu ? "GPU" : "rank";
    const std::string units = unit + "s";
    std::vector<Row> rows;
    const Status fixed = n_of(r, check_code::SIZING_FIXED) ? Status::WARNING : Status::OK;
    rows.push_back({Status::NONE, {"memory per " + unit, bytes(double(r.sizing_memory)), "", "", r.sizing_name}});
    rows.push_back({Status::NONE,
                    {"kept free", format("%.0f%%", 100.0 * SIZING_MARGIN), "of the memory", "",
                     "fragmentation and allocations during the run"}});
    if (r.sizing_gpu) {
        rows.push_back({Status::NONE,
                        {"outside Kokkos per GPU", bytes(double(r.sizing_overhead)), "", "",
                         r.overhead_measured ? "measured on this GPU"
                                             : "typical: CUDA's context and kernels' local memory on an A100"}});
    }
    if (r.max_local_cells == 0) {
        rows.push_back(
            {fixed, {"most cells per " + unit, "0", "cells", "", "see " + std::string(check_code::SIZING_FIXED)}});
    } else {
        rows.push_back({fixed,
                        {"most cells per " + unit, count(r.max_local_cells), "owned and halo cells", "",
                         bytes(r.bytes_per_cell) + " per cell" +
                             (r.fixed_bytes > 0 ? " + " + bytes(double(r.fixed_bytes)) : "") +
                             (r.sizing_setup ? ", at the setup's peak" : ", in the steady state")}});
        if (r.min_ranks == 0) {
            rows.push_back({Status::WARNING,
                            {"fewest " + units, "-", units, "", "see " + std::string(check_code::SIZING_PARTS)}});
        } else {
            rows.push_back({Status::OK,
                            {"fewest " + units, std::to_string(r.min_ranks), units, "",
                             "largest part " + count(r.min_owned) + " owned + " + count(r.min_halo) +
                                 " halo cells, predicted peak " + bytes(double(r.min_peak)) +
                                 (r.min_partitioned ? " (from the partitioner)" : " (halo as in this check)")}});
            if (r.halo_ranks > r.min_ranks) {
                rows.push_back({Status::NONE,
                                {format("most %s with halo < %.0f%% of owned", units.c_str(),
                                        100.0 * SIZING_HALO_FRACTION),
                                 std::to_string(r.halo_ranks), units, "", "doubling from the fewest"}});
            } else if (r.min_partitioned && r.min_ranks > 1) {
                rows.push_back({Status::NONE,
                                {format("most %s with halo < %.0f%% of owned", units.c_str(),
                                        100.0 * SIZING_HALO_FRACTION),
                                 "none", units, "",
                                 "the halo is already above that on the fewest: a coarser mesh or larger " + units +
                                     " keep more of the work local"}});
            }
            if (r.throughput > 0.0 && r.dt > 0.0) {
                const int n = std::max(r.min_ranks, r.halo_ranks);
                rows.push_back({Status::NONE,
                                {"wall time per unit simulated time",
                                 logging::duration(double(r.n_cells) / (n * r.throughput * r.dt)), "", "",
                                 "on " + plural(uint64_t(n), unit) + " at " + logging::si(r.throughput) +
                                     " cells/s each"}});
            }
        }
    }
    Status s = Status::OK;
    for (const Row & row : rows) s = report::worse(s, row.status);
    text.section(r.sizing_gpu ? "GPU sizing" : "Rank sizing", s);
    text.note("The largest rank's predicted peak (the larger of the setup's and the steady state's, plus the memory "
              "outside Kokkos) against the memory per " + unit + ", for --memory or this build's GPU.");
    quantities(text, rows);
}

void cost_section(report::Text & text, const CheckReport & r) {
    text.section("Cost", Status::NONE);
    if (!(r.wall_per_time > 0.0)) {
        text.line("Not estimated: --throughput CELLS_PER_S or --steps N gives an estimate.");
        return;
    }
    const std::string hours = r.device_memory ? "GPU-hours" : "rank-hours";
    text.note("On " + plural(uint64_t(r.target_ranks), "rank") + " at " + logging::si(r.throughput) +
              " cells/s per rank (" +
              (r.throughput_measured ? "measured over " + plural(r.steps_timed, "step") : std::string("--throughput")) +
              "), at the initial time step; " + hours + ": ranks times wall hours.");
    std::vector<Row> rows;
    auto row = [&](const std::string & period, const double time) {
        rows.push_back({Status::NONE,
                        {period, g4(time), steps(time / r.dt), logging::duration(r.wall_per_time * time),
                         format("%.3g", r.wall_per_time * time * r.target_ranks / 3600.0)}});
    };
    row("per unit simulated time", 1.0);
    if (r.flow_through_time > 0.0) row("per flow-through time", r.flow_through_time);
    if (r.t_stop > 0.0) row("to run.t_stop", r.t_stop);
    if (r.n_steps > 0) row("for run.n_steps", double(r.n_steps) * r.dt);
    text.table({"period", "simulated time [s]", "steps", "wall time", hours}, rows, false);
}

void diagnostics_section(report::Text & text, const CheckReport & r) {
    text.section("Diagnostics", r.status());
    if (r.diagnostics.empty()) {
        text.line("None.");
        return;
    }
    std::vector<Row> rows;
    for (const Status severity : {Status::ERROR, Status::WARNING}) {
        for (const CheckDiagnostic & d : r.diagnostics) {
            if (d.severity == severity) rows.push_back({severity, {d.code, d.message + "\n" + d.docs()}});
        }
    }
    text.table({"code", "message, then the docs that explain it"}, rows);
}

std::string points_json(const double * x, const size_t n) {
    std::string out = "[";
    for (size_t i = 0; i < n; i++) out += (i ? "," : "") + json::number(x[i]);
    return out + "]";
}

template <size_t N>
std::string point_json(const std::array<double, N> & x) {
    return points_json(x.data(), N_DIM);
}

std::string bytes_json(const memory::Bytes & b) {
    json::Object o;
    for (size_t c = 0; c < memory::N_CATEGORIES; c++) o.add(memory::category_name(memory::Category(c)), b[c]);
    return o.str();
}

template <typename T, typename F>
std::string array_json(const std::vector<T> & list, F && f) {
    std::string out = "[";
    for (const T & item : list) out += (out.size() > 1 ? "," : "") + f(item);
    return out + "]";
}

} // namespace

std::string check_text(const CheckReport & r, const bool color) {
    report::Text text(color);
    overview(text, r);
    input_section(text, r);
    if (r.setup_done) boundaries_section(text, r);
    regions_section(text, r);
    if (r.has_quality) quality_section(text, r);
    if (r.dt > 0.0) time_step_section(text, r);
    resolution_section(text, r);
    if (r.bytes > 0) memory_section(text, r);
    sizing_section(text, r);
    if (r.setup_done) cost_section(text, r);
    diagnostics_section(text, r);
    text.blank();
    text.line(result_line(r));
    return text.str();
}

std::string check_json(const CheckReport & r) {
    json::Object o;
    o.add("version", 1);
    o.add("status", std::string(report::status_name(r.status())));
    o.add("errors", uint64_t(r.errors().size())).add("warnings", uint64_t(r.warnings().size()));
    o.add("input_file", r.input_file).add("setup_done", r.setup_done).add("dim", N_DIM);
    o.raw("diagnostics", array_json(r.diagnostics, [](const CheckDiagnostic & d) {
              return json::Object()
                  .add("severity", std::string(report::status_name(d.severity)))
                  .add("code", d.code)
                  .add("message", d.message)
                  .add("docs", d.docs())
                  .str();
          }));
    o.raw("unused_keys", array_json(r.unused, [](const Diagnostic & d) {
              return json::Object()
                  .add("key", d.key)
                  .raw("line", d.line > 0 ? std::to_string(d.line) : "null")
                  .raw("suggestion", d.suggestion.empty() ? "null" : json::quote(d.suggestion))
                  .str();
          }));
    o.raw("boundaries", array_json(r.boundaries, [](const CheckBoundary & b) {
              return json::Object()
                  .add("zone", b.zone)
                  .add("condition", b.condition)
                  .add("faces", b.faces)
                  .add("details", b.details)
                  .add("turbulence", b.turbulence)
                  .str();
          }));
    o.raw("regions", array_json(r.regions, [](const CheckRegion & g) {
              return json::Object()
                  .add("name", g.name)
                  .add("of_mesh", g.of_mesh)
                  .add("dimension", g.dimension)
                  .add("measure", g.measure)
                  .add("per_radian", g.per_radian)
                  .add("cells", g.cells)
                  .add("exact", g.exact)
                  .raw("resolution", g.exact ? "null" : std::to_string(g.resolution))
                  .str();
          }));
    o.raw("case_view", r.view_file.empty() ? "null" : json::quote(r.view_file));
    if (r.has_quality) {
        json::Object q;
        for (size_t m = 0; m < MeshQuality::N_METRICS; m++) {
            const QualitySummary & s = r.quality[m];
            q.raw(MeshQuality::FIELDS[m], json::Object()
                                              .add("max", s.max)
                                              .add("mean", s.mean)
                                              .add("limit", MeshQuality::LIMITS[m])
                                              .add("above_limit", s.n_over_limit)
                                              .raw("worst", array_json(s.worst, [](const QualityCell & c) {
                                                       return json::Object()
                                                           .add("value", c.value)
                                                           .add("cell", c.cell)
                                                           .raw("x", point_json(c.x))
                                                           .str();
                                                   }))
                                              .str());
        }
        o.raw("mesh_quality", q.str());
        o.raw("quality_file", r.quality_file.empty() ? "null" : json::quote(r.quality_file));
    } else {
        o.raw("mesh_quality", "null");
    }
    if (r.dt > 0.0) {
        o.raw("time_step", json::Object()
                               .add("t", r.t)
                               .add("dt", r.dt)
                               .add("dt_cfl1", r.dt_cfl1)
                               .add("cfl", r.cfl)
                               .add("cfl_limit", r.cfl_limit)
                               .add("integrator", r.integrator)
                               .add("set_by", r.dt_limit)
                               .add("limiting_cell", r.limiting_cell)
                               .raw("limiting_x", point_json(r.limiting_x))
                               .add("region_cells", r.n_region_cells)
                               .raw("region_lo", point_json(r.region_lo))
                               .raw("region_hi", point_json(r.region_hi))
                               .str());
    } else {
        o.raw("time_step", "null");
    }
    json::Object resolution;
    for (const CheckEstimate & e : r.resolution) {
        resolution.raw(e.id, json::Object()
                                 .add("quantity", e.quantity)
                                 .add("value", e.value)
                                 .add("units", e.units)
                                 .add("target", e.target)
                                 .add("status", std::string(report::status_name(e.status)))
                                 .add("basis", e.basis)
                                 .raw("x", e.located ? point_json(e.x) : "null")
                                 .str());
    }
    o.raw("resolution", resolution.str());
    if (r.bytes > 0) {
        o.raw("memory", json::Object()
                            .add("device_memory", r.device_memory)
                            .add("ranks", r.ranks)
                            .add("bytes", r.bytes)
                            .add("bytes_peak", r.bytes_peak)
                            .add("local_cells", r.local_cells)
                            .add("local_owned", r.local_owned)
                            .add("target_ranks", r.target_ranks)
                            .add("target_owned", r.target_owned)
                            .add("target_halo", r.target_halo)
                            .add("target_bytes", r.target_bytes)
                            .add("target_bytes_peak", r.target_bytes_peak)
                            .add("device_capacity", r.device_capacity)
                            .add("outside_kokkos", r.outside_kokkos)
                            .add("steps_timed", r.steps_timed)
                            .raw("at_setup_peak", bytes_json(r.setup_bytes))
                            .raw("steady", bytes_json(r.steady_bytes))
                            .str());
    } else {
        o.raw("memory", "null");
    }
    if (!r.sizing_name.empty()) {
        o.raw("sizing", json::Object()
                            .add("name", r.sizing_name)
                            .add("gpu", r.sizing_gpu)
                            .add("memory", r.sizing_memory)
                            .add("margin", SIZING_MARGIN)
                            .add("outside_kokkos", r.sizing_overhead)
                            .add("outside_kokkos_measured", r.overhead_measured)
                            .add("bytes_per_cell", r.bytes_per_cell)
                            .add("fixed_bytes", r.fixed_bytes)
                            .add("at_setup_peak", r.sizing_setup)
                            .add("max_local_cells", r.max_local_cells)
                            .add("min_ranks", r.min_ranks)
                            .add("min_owned", r.min_owned)
                            .add("min_halo", r.min_halo)
                            .add("min_peak", r.min_peak)
                            .add("min_partitioned", r.min_partitioned)
                            .add("halo_fraction", SIZING_HALO_FRACTION)
                            .add("halo_ranks", r.halo_ranks)
                            .str());
    } else {
        o.raw("sizing", "null");
    }
    if (r.setup_done) {
        o.raw("cost", json::Object()
                          .add("cells", r.n_cells)
                          .add("ranks", r.target_ranks)
                          .add("throughput", r.throughput)
                          .add("throughput_measured", r.throughput_measured)
                          .add("steps_timed", r.steps_timed)
                          .add("wall_per_time", r.wall_per_time)
                          .add("flow_through_time", r.flow_through_time)
                          .add("t_stop", r.t_stop)
                          .add("n_steps", r.n_steps)
                          .str());
    } else {
        o.raw("cost", "null");
    }
    return o.str();
}

void print_check(const CheckReport & r, const bool json) {
    if (json) {
        if (comm::is_root()) std::cout << check_json(r) << std::endl;
        return;
    }
    const std::string text = check_text(r, logging::color_stdout());
    size_t start = 0;
    while (start < text.size()) {
        const size_t end = text.find('\n', start);
        logging::line(std::string_view(text).substr(start, end - start));
        start = end + 1;
    }
}

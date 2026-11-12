/**
 * @file check_report.cpp
 * @brief The report of mallard check, as text and as JSON, from the same CheckReport.
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 */

#include <algorithm>
#include <cctype>
#include <cmath>
#include <initializer_list>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

#include "check.h"
#include "comm.h"
#include "json.h"
#include "log.h"
#include "report.h"

using report::Row;
using report::Status;
using Kind = report::Column::Kind;

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

std::string CheckDiagnostic::docs() const { return diagnostic_code::docs(code); }

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
/** @brief A duration as a bare value of three digits ("21.0") and its units: us, ms, s, min or h. */
std::pair<std::string, std::string> time_parts(const double s) {
    auto digits3 = [](const double v) { return format(v < 9.995 ? "%.2f" : (v < 99.95 ? "%.1f" : "%.0f"), v); };
    if (s < 0.9995e-3) return {digits3(s * 1.0e6), "us"};
    if (s < 0.9995) return {digits3(s * 1.0e3), "ms"};
    if (s < 59.95) return {digits3(s), "s"};
    if (s < 3597.0) return {digits3(s / 60.0), "min"};
    return {format("%.3g", s / 3600.0), "h"};
}

std::string steps(const double n) { return n >= 1000.0 ? count(static_cast<uint64_t>(std::llround(n))) : g4(n); }

template <size_t N>
std::string point(const std::array<double, N> & x) {
    std::string text = "(";
    for (size_t i = 0; i < N_DIM; i++) text += (i ? ", " : "") + g4(x[i]);
    return text + ") m";
}

std::string result_line(const CheckReport & r) {
    const size_t n_errors = r.errors().size(), n_warnings = r.warnings().size();
    return std::string("Result: ") + report::status_word(r.status()) + ", " + count(n_errors, "error") + " and " +
           count(n_warnings, "warning") + (n_errors > 0 ? " (exit status 1)" : " (exit status 0)");
}

size_t n_of(const CheckReport & r, const std::string & code) {
    return static_cast<size_t>(std::count_if(r.diagnostics.begin(), r.diagnostics.end(),
                                             [&](const CheckDiagnostic & d) { return d.code == code; }));
}


bool environment_only(const CheckReport & r) { return r.input_file.empty() && r.environment.checked; }

void overview(report::Text & text, const CheckReport & r) {
    text.section("Check", r.status());
    text.line(result_line(r));
    if (environment_only(r)) {
        text.line("Input: none, so the report covers the environment only (mallard check input.toml adds the case)");
    } else {
        text.line("Input: " + (r.input_file.empty() ? std::string("the input") : r.input_file) +
                  (r.setup_done ? ", set up on " + count(uint64_t(r.ranks), "rank")
                                : ": the setup stopped on an error, so the report covers what it got to"));
    }
    text.line("Units: SI (m, s, kg, K) wherever a column or a value names none; bytes with decimal prefixes "
              "(1 GB = 10^9 B)");
}

void input_section(report::Text & text, const CheckReport & r) {
    const size_t n_unused = n_of(r, diagnostic_code::KEY_UNUSED), n_deprecated = n_of(r, diagnostic_code::KEY_DEPRECATED);
    const bool stopped = n_of(r, diagnostic_code::INPUT) + n_of(r, diagnostic_code::SETUP) > 0;
    const Status setup = stopped ? Status::ERROR : Status::OK;
    std::vector<Row> rows;
    rows.push_back({setup, {"setup", stopped ? "stopped on an error: see " + std::string(n_of(r, diagnostic_code::INPUT)
                                                                                        ? diagnostic_code::INPUT
                                                                                        : diagnostic_code::SETUP)
                                             : "completed: mesh, partition, boundaries, numerics, initial state"}});
    rows.push_back({n_unused ? Status::ERROR : Status::OK,
                    {"unknown or unused keys",
                     n_unused ? count(n_unused, "key") + ": see " + diagnostic_code::KEY_UNUSED
                              : (r.setup_done ? "none: every key is used" : "none found up to the error")}});
    rows.push_back({n_deprecated ? Status::WARNING : Status::OK,
                    {"deprecated keys",
                     n_deprecated ? count(n_deprecated, "key") + ": see " + diagnostic_code::KEY_DEPRECATED : "none"}});
    Status s = setup;
    for (const Row & row : rows) s = report::worse(s, row.status);
    text.section("Input", s);
    text.table({"check", "result"}, rows);
}

void boundaries_section(report::Text & text, const CheckReport & r) {
    text.section("Boundaries", Status::OK);
    text.note("Every boundary face has a condition: " + count(r.boundaries.size(), "zone") +
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

std::string measure_units(const CheckRegion & g) {
    switch (g.dimension) {
        case 0: return "points";
        case 1: return "m";
        default: break;
    }
    if (g.dimension < N_DIM) return "m^2";
    if (N_DIM == 2) return g.per_radian ? "m^3 per radian" : "m^2";
    return "m^3";
}

std::string surface_units(const CheckRegion & g) {
    if (N_DIM == 2) return g.per_radian ? "m^2 per radian" : "m";
    return "m^2";
}

void regions_section(report::Text & text, const CheckReport & r) {
    if (r.regions.empty()) return;
    std::vector<Row> rows;
    Status s = Status::OK;
    for (const CheckRegion & g : r.regions) {
        static const char * kinds[4] = {"points", "curve", "surface", ""};
        const std::string kind = g.dimension == N_DIM ? "filled" : kinds[g.dimension];
        const std::string made = g.of_mesh                 ? "the mesh's"
                                 : g.solution              ? "the solution, during the run"
                                 : g.dimension == N_DIM    ? "cell fractions"
                                 : g.exact                 ? "exact"
                                                           : "extracted, resolution " + std::to_string(g.resolution);
        const Status row = g.cells > 0 || g.solution ? Status::OK : Status::ERROR;
        s = report::worse(s, row);
        rows.push_back({row, {g.name, kind, g.solution ? "-" : logging::real(g.measure), measure_units(g),
                              g.solution ? "-" : count(g.cells), made}});
    }
    text.section("Regions", s);
    text.note("The mesh's own (the domain, its boundary and cell zones) and every region the case uses; cells: the "
              "cells each touches. A region of the case that touches no cell is an error.");
    text.table({"region", "kind", {"measure", Kind::VALUE}, {"units", Kind::UNITS}, "cells", "made from"}, rows);
    // Filled regions of the case: their part in the domain against their own measures
    std::vector<Row> measures;
    for (const CheckRegion & g : r.regions) {
        if (g.of_mesh || g.dimension != N_DIM || g.solution) continue;
        const bool own = std::isfinite(g.own_measure);
        const std::string outside =
            own && g.own_measure > 0.0 ? logging::real(100.0 * std::max(0.0, 1.0 - g.measure / g.own_measure)) + "%" : "-";
        const std::string tolerance = !own ? "-" : g.own_tolerance == 0.0 ? "exact" : logging::real(g.own_tolerance);
        measures.push_back({Status::OK, {g.name, N_DIM == 2 && !g.per_radian ? "area" : "volume", logging::real(g.measure),
                                         own ? logging::real(g.own_measure) : "-", measure_units(g), tolerance, outside}});
        measures.push_back({Status::OK, {g.name, N_DIM == 2 && !g.per_radian ? "perimeter" : "surface area",
                                         logging::real(g.surface), own ? logging::real(g.own_surface) : "-",
                                         surface_units(g), tolerance, ""}});
    }
    if (!measures.empty()) {
        text.note("Filled regions: their part in the domain (its surface: theirs inside the domain and the domain's "
                  "boundary inside them) and their own measures, independent of the mesh (\"-\" for regions of the "
                  "mesh or unbounded ones), exact or within a relative tolerance; outside: the share outside the "
                  "domain.");
        text.table({"region", "quantity", {"in domain", Kind::VALUE}, {"own", Kind::VALUE}, {"units", Kind::UNITS},
                    "tolerance", "outside"},
                   measures);
    }
    if (!r.view_file.empty()) text.line("Case view (every region and zone, for ParaView): " + r.view_file);
    if (!r.view_images.empty()) {
        std::string images;
        for (const std::string & f : r.view_images) images += (images.empty() ? "" : ", ") + f;
        text.line("Case view images: " + images);
    }
}

void quality_section(report::Text & text, const CheckReport & r) {
    constexpr auto UNITS = MeshQuality::UNITS;
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
            std::vector<std::string> cells = {MeshQuality::NAMES[m], format("%.3g", c.value), UNITS[m],
                                              count(c.cell)};
            for (size_t i = 0; i < N_DIM; i++) cells.push_back(g4(c.x[i]));
            worst.push_back({Status::NONE, cells});
        }
    }
    text.section("Mesh quality", s);
    text.note("Over every cell; a cell above the limit is questionable.");
    text.table({"metric", {"units", Kind::UNITS}, {"max", Kind::VALUE}, {"mean", Kind::VALUE}, {"limit", Kind::VALUE},
                "cells over limit"},
               rows);
    if (!worst.empty()) {
        text.blank();
        text.note("Worst cells of the metrics at or above their notable value (non-orthogonality 10 deg, skewness "
                  "0.05, aspect ratio 10, volume ratio 2), worst first:");
        std::vector<report::Column> header = {"metric", {"value", Kind::VALUE}, {"units", Kind::UNITS},
                                              "cell (global id)"};
        for (size_t i = 0; i < N_DIM; i++) header.push_back(std::string(1, "xyz"[i]) + " [m]");
        text.table(header, worst, false);
    }
    if (!r.mesh_fields_file.empty()) text.line("Mesh and stencil fields: " + r.mesh_fields_file);
    if (!r.stencil_file.empty()) text.line("Stencils: " + r.stencil_file);
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
    text.quantities(rows);
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
    text.quantities(rows);
}

void memory_section(report::Text & text, const CheckReport & r) {
    const double cells = static_cast<double>(std::max<uint64_t>(r.local_cells, 1));
    const bool target = r.target_ranks != r.ranks;
    Status s = Status::NONE;
    std::string device = "-";
    if (r.device_capacity > 0) {
        s = n_of(r, diagnostic_code::MEMORY_DEVICE) ? Status::ERROR : Status::OK;
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
    const Status fixed = n_of(r, diagnostic_code::SIZING_FIXED) ? Status::WARNING : Status::OK;
    auto memory_row = [&](const std::string & quantity, const double b, const std::string & basis) {
        const std::string text_bytes = bytes(b);
        const size_t space = text_bytes.find(' ');
        rows.push_back({Status::NONE, {quantity, text_bytes.substr(0, space), text_bytes.substr(space + 1), "", basis}});
    };
    memory_row("memory per " + unit, double(r.sizing_memory), r.sizing_name);
    rows.push_back({Status::NONE,
                    {"kept free", format("%.0f", 100.0 * SIZING_MARGIN), "% of the memory", "",
                     "fragmentation and allocations during the run"}});
    if (r.sizing_gpu) {
        memory_row("outside Kokkos per GPU", double(r.sizing_overhead),
                         r.overhead_measured ? "measured on this GPU"
                                             : "typical: CUDA's context and kernels' local memory on an A100");
    }
    if (r.max_local_cells == 0) {
        rows.push_back(
            {fixed, {"most cells per " + unit, "0", "cells", "", "see " + std::string(diagnostic_code::SIZING_FIXED)}});
    } else {
        rows.push_back({fixed,
                        {"most cells per " + unit, count(r.max_local_cells), "owned and halo cells", "",
                         bytes(r.bytes_per_cell) + " per cell" +
                             (r.fixed_bytes > 0 ? " + " + bytes(double(r.fixed_bytes)) : "") +
                             (r.sizing_setup ? ", at the setup's peak" : ", in the steady state")}});
        if (r.min_ranks == 0) {
            rows.push_back({Status::WARNING,
                            {"fewest " + units, "-", units, "", "see " + std::string(diagnostic_code::SIZING_PARTS)}});
        } else {
            rows.push_back({Status::OK,
                            {"fewest " + units, std::to_string(r.min_ranks), units, "",
                             "largest part " + count(r.min_owned) + " owned + " + count(r.min_halo, "halo cell") +
                                 ", predicted peak " + bytes(double(r.min_peak)) +
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
                const auto [wall, wall_units] = time_parts(double(r.n_cells) / (n * r.throughput * r.dt));
                rows.push_back({Status::NONE,
                                {"wall time per unit simulated time", wall, wall_units, "",
                                 "on " + count(uint64_t(n), unit) + " at " + logging::si(r.throughput) +
                                     " cells/s each"}});
            }
        }
    }
    Status s = Status::OK;
    for (const Row & row : rows) s = report::worse(s, row.status);
    text.section(r.sizing_gpu ? "GPU sizing" : "Rank sizing", s);
    text.note("The largest rank's predicted peak (the larger of the setup's and the steady state's, plus the memory "
              "outside Kokkos) against the memory per " + unit + ", for --memory or this build's GPU.");
    text.quantities(rows);
}

void cost_section(report::Text & text, const CheckReport & r) {
    text.section("Cost", Status::NONE);
    if (!(r.wall_per_time > 0.0)) {
        text.line("Not estimated: --throughput CELLS_PER_S or --steps N gives an estimate.");
        return;
    }
    const std::string hours = r.device_memory ? "GPU-hours" : "rank-hours";
    text.note("On " + count(uint64_t(r.target_ranks), "rank") + " at " + logging::si(r.throughput) +
              " cells/s per rank (" +
              (r.throughput_measured ? "measured over " + count(r.steps_timed, "step") : std::string("--throughput")) +
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

/** @brief A member of a provenance block: a string as it is, a bool as yes or no, null as empty. */
std::string block_value(const json::Value & block, const std::string & key) {
    const json::Value * v = block.find(key);
    if (!v) return "";
    if (v->type == json::Value::Type::BOOL) return v->boolean ? "yes" : "no";
    if (v->type == json::Value::Type::NUMBER) return json::number(v->number);
    return v->string;
}

std::string version_or(const json::Value & dependencies, const std::string & key, const std::string & none) {
    const std::string v = block_value(dependencies, key);
    return v.empty() ? none : v;
}

std::string gpu_aware_library_text(int answer) {
    return answer < 0 ? "does not say" : answer ? "reads device memory" : "cannot read device memory";
}

void environment_section(report::Text & text, const CheckReport & r) {
    const CheckEnvironment & e = r.environment;
    const json::Value build = json::parse(e.build), deps = json::parse(e.dependencies);
    const auto status_of = [&](std::initializer_list<const char *> codes) {
        Status s = Status::OK;
        for (const CheckDiagnostic & d : r.diagnostics) {
            for (const char * code : codes) {
                if (d.code == code) s = report::worse(s, d.severity);
            }
        }
        return s;
    };
    std::string solvers;
    for (const int d : e.solvers) solvers += (solvers.empty() ? "mallard" : ", mallard") + std::to_string(d) + "d";
    const bool mpi = block_value(build, "mpi") == "yes";
    const bool gpu = block_value(build, "backend") != block_value(build, "host_backend");
    const bool openmp = block_value(build, "host_backend") == "OpenMP";
    const std::string hdf5 = block_value(deps, "hdf5");
    std::vector<Row> rows = {
        {Status::OK,
         {"Mallard", block_value(build, "dim") + "D, " + block_value(build, "precision") + " precision, " +
                         block_value(build, "build_type") + ", " + block_value(build, "compiler")}},
        {Status::OK, {"solvers installed", solvers.empty() ? "none beside this one" : solvers}},
        {Status::OK,
         {"Kokkos", block_value(build, "kokkos") + ", " + block_value(build, "backend") + " (host " +
                        block_value(build, "host_backend") + ")"}},
        {Status::OK,
         {"MPI", mpi ? version_or(deps, "mpi", "built") + "; " + count(uint64_t(e.ranks.size()), "rank") + " on " +
                           count(uint64_t(e.hosts), "host") + ", up to " + std::to_string(e.ranks_per_host) +
                           " per host"
                     : "not built"}},
        {status_of({diagnostic_code::ENV_GPU_AWARE_MPI, diagnostic_code::ENV_HOST_STAGING}),
         {"GPU-aware MPI", !mpi ? "not built"
                           : !gpu ? "not needed: the fields are in host memory"
                           : block_value(build, "gpu_aware_mpi") == "yes"
                               ? std::string("built; the MPI library ") + gpu_aware_library_text(e.gpu_aware_library) +
                                     "; " + (e.gpu_aware_used ? "used by this run" : "not used by this run")
                               : std::string("not built; the MPI library ") + gpu_aware_library_text(e.gpu_aware_library)}},
        {Status::OK,
         {"NCCL", block_value(build, "nccl") == "yes"
                      ? version_or(deps, "nccl", "built") +
                            (e.nccl_used ? "; halo exchanges use it" : "; halo exchanges do not use it in this run")
                      : "not built"}},
        {Status::OK,
         {"HDF5", hdf5.empty() ? "not built"
                               : hdf5 + (block_value(build, "hdf5_parallel") == "yes" ? ", parallel" : ", serial")}},
        {Status::OK, {"CGNS", version_or(deps, "cgns", "not built")}},
        {Status::OK, {"zlib", version_or(deps, "zlib", "not built")}},
        {Status::OK, {"KaMinPar", block_value(build, "kaminpar") == "yes" ? "built" : "not built"}},
        {status_of({diagnostic_code::ENV_PLACEMENT, diagnostic_code::ENV_THREAD_BINDING}),
         {"placement", status_of({diagnostic_code::ENV_PLACEMENT, diagnostic_code::ENV_THREAD_BINDING}) == Status::OK
                           ? "no problem found"
                           : std::string("see ") + diagnostic_code::ENV_PLACEMENT + " and " + diagnostic_code::ENV_THREAD_BINDING}},
    };
    Status s = Status::OK;
    for (const Row & row : rows) s = report::worse(s, row.status);
    text.section("Environment", s);
    text.table({"check", "result"}, rows);

    constexpr size_t MAX_RANKS = 16;
    std::vector<Row> ranks;
    for (size_t i = 0; i < e.ranks.size() && i < MAX_RANKS; i++) {
        const binding::Placement & p = e.ranks[i];
        ranks.push_back({Status::NONE,
                         {std::to_string(p.rank), p.host, i < e.devices.size() ? e.devices[i] : "",
                          p.gpu_index < 0 ? "-" : std::to_string(p.gpu_index) + " of " +
                                                      std::to_string(std::max<size_t>(p.gpus_visible.size(), 1)),
                          p.cpus.empty() ? "-" : binding::format_cpu_list(p.cpus), std::to_string(p.threads),
                          !openmp ? "-" : p.omp_bind ? "bound" : "free"}});
    }
    text.table({"rank", "host", "device", "GPU", "CPUs", "threads", "binding"}, ranks, false);
    if (e.ranks.size() > MAX_RANKS) text.note("... and " + count(e.ranks.size() - MAX_RANKS, "more rank"));
}

std::string environment_json(const CheckEnvironment & e) {
    if (!e.checked) return "null";
    json::Object o;
    o.raw("build", e.build).raw("dependencies", e.dependencies);
    o.raw("solvers", array_json(e.solvers, [](int d) { return std::to_string(d); }));
    o.add("ranks", uint64_t(e.ranks.size())).add("hosts", e.hosts).add("ranks_per_host", e.ranks_per_host);
    o.raw("gpu_aware_mpi", json::Object()
                               .raw("library", e.gpu_aware_library < 0 ? "null" : e.gpu_aware_library ? "true" : "false")
                               .add("used", e.gpu_aware_used)
                               .str());
    o.raw("nccl", json::Object().add("used", e.nccl_used).str());
    std::vector<size_t> index(e.ranks.size());
    for (size_t i = 0; i < index.size(); i++) index[i] = i;
    o.raw("placement", array_json(index, [&](size_t i) {
              const binding::Placement & p = e.ranks[i];
              const auto known = [](int v) { return v < 0 ? std::string("null") : std::to_string(v); };
              return json::Object()
                  .add("rank", p.rank)
                  .add("host", p.host)
                  .raw("device", i < e.devices.size() ? json::quote(e.devices[i]) : "null")
                  .add("threads", p.threads)
                  .raw("cpus", p.cpus.empty() ? "null" : json::quote(binding::format_cpu_list(p.cpus)))
                  .add("omp_bind", p.omp_bind)
                  .raw("gpu", p.gpu.empty() ? "null" : json::quote(p.gpu))
                  .raw("gpu_index", known(p.gpu_index))
                  .raw("gpu_numa", known(p.gpu_numa))
                  .add("gpus_visible", uint64_t(p.gpus_visible.size()))
                  .str();
          }));
    return o.str();
}

std::string diagnostics_json(const CheckReport & r) {
    return array_json(r.diagnostics, [](const CheckDiagnostic & d) {
        return json::Object()
            .add("severity", std::string(report::status_name(d.severity)))
            .add("code", d.code)
            .add("message", d.message)
            .add("docs", d.docs())
            .str();
    });
}

std::string quality_json(const CheckReport & r) {
    json::Object q;
    for (size_t m = 0; m < MeshQuality::N_METRICS; m++) {
        const QualitySummary & s = r.quality[m];
        q.raw(MeshQuality::FIELDS[m], json::Object()
                                          .add("name", std::string(MeshQuality::NAMES[m]))
                                          .add("max", s.max)
                                          .add("mean", s.mean)
                                          .add("units", std::string(MeshQuality::UNITS[m]))
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
    return q.str();
}

std::string resolution_json(const CheckReport & r) {
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
    return resolution.str();
}

} // namespace

std::string check_text(const CheckReport & r, const bool color) {
    report::Text text(color);
    overview(text, r);
    if (!environment_only(r)) {
        input_section(text, r);
        if (r.setup_done) boundaries_section(text, r);
        regions_section(text, r);
        if (r.has_quality) quality_section(text, r);
        if (r.dt > 0.0) time_step_section(text, r);
        resolution_section(text, r);
        if (r.bytes > 0) memory_section(text, r);
        sizing_section(text, r);
        if (r.setup_done) cost_section(text, r);
    }
    if (r.environment.checked) environment_section(text, r);
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
    o.raw("input_file", r.input_file.empty() ? "null" : json::quote(r.input_file))
        .add("setup_done", r.setup_done)
        .add("dim", N_DIM);
    o.raw("environment", environment_json(r.environment));
    o.raw("diagnostics", diagnostics_json(r));
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
                  .add("solution", g.solution)
                  .add("surface", g.surface)
                  .raw("own", std::isfinite(g.own_measure) ? json::Object()
                                                                 .add("measure", g.own_measure)
                                                                 .add("surface", g.own_surface)
                                                                 .add("tolerance", g.own_tolerance)
                                                                 .str()
                                                           : "null")
                  .str();
          }));
    o.raw("case_view", r.view_file.empty() ? "null" : json::quote(r.view_file));
    o.raw("case_view_images", array_json(r.view_images, [](const std::string & f) { return json::quote(f); }));
    if (r.has_quality) {
        o.raw("mesh_quality", quality_json(r));
        o.raw("mesh_fields_file", r.mesh_fields_file.empty() ? "null" : json::quote(r.mesh_fields_file));
        o.raw("stencil_file", r.stencil_file.empty() ? "null" : json::quote(r.stencil_file));
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
    o.raw("resolution", resolution_json(r));
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

json::Object check_summary(const CheckReport & r) {
    json::Object time_step;
    time_step.add("integrator", r.integrator).add("cfl_limit", r.cfl_limit).add("set_by", r.dt_limit);
    if (r.dt_limit == "CFL") time_step.add("cfl", r.cfl);
    return json::Object()
        .add("status", std::string(report::status_name(r.status())))
        .raw("diagnostics", diagnostics_json(r))
        .raw("mesh_quality", r.has_quality ? quality_json(r) : "null")
        .raw("time_step", time_step.str())
        .raw("resolution", resolution_json(r));
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

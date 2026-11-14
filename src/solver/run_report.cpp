/**
 * @file run_report.cpp
 * @brief The setup banner, the progress table, the summary and the health section of a run.
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 */

#include "run_report.h"

#include <cctype>
#include <cmath>
#include <utility>

#include "common.h"
#include "log.h"
#include "units.h"

using logging::format;
using report::Column;
using report::Field;
using report::Row;
using report::Status;

namespace {

/** @brief Wall time as h:mm:ss, rounded up to the second. */
std::string clock_time(const double s) {
    const auto total = static_cast<unsigned long long>(std::ceil(s));
    return format("%llu:%02llu:%02llu", total / 3600, (total / 60) % 60, total % 60);
}

std::string percent(const double fraction) { return format("%.2f", 100.0 * fraction); }

/** @brief Three significant digits with trailing zeros kept, so a column of times keeps its decimals. */
std::string milliseconds(const double seconds) {
    const double ms = 1.0e3 * seconds;
    return format(ms < 9.995 ? "%.2f" : ms < 99.95 ? "%.1f" : "%.0f", ms);
}

/** @brief The unit a table of wall times is in, from its largest: s, min or h. */
std::pair<const char *, double> wall_unit(const double largest) {
    if (largest < 120.0) return {"s", 1.0};
    if (largest < 7200.0) return {"min", 60.0};
    return {"h", 3600.0};
}

/** @brief Bytes as a bare value and its unit: {"8.14", "MB"}. */
std::pair<std::string, std::string> bytes_parts(const double b) {
    const std::string text = logging::si(b);
    if (!text.empty() && std::isalpha(static_cast<unsigned char>(text.back()))) {
        return {text.substr(0, text.size() - 1), std::string(1, text.back()) + "B"};
    }
    return {text, "B"};
}

/** @brief A value in seconds as a bare cell: "%.6g". */
std::string real(const double v) { return format("%.6g", v); }

} // namespace

void setup_text(report::Text & text, const std::vector<SetupSection> & sections) {
    for (const SetupSection & s : sections) {
        text.section(s.title);
        bool first = true;
        auto gap = [&] {
            if (!first) text.blank();
            first = false;
        };
        if (!s.settings.empty()) {
            gap();
            std::vector<Row> rows;
            for (const auto & [name, value] : s.settings) rows.push_back({Status::NONE, {name, value}});
            text.table({"setting", "value"}, rows, false);
        }
        if (!s.quantities.empty()) {
            gap();
            text.quantities(s.quantities, false);
        }
        for (const SetupSection::Table & t : s.tables) {
            if (t.rows.empty()) continue;
            gap();
            if (!t.note.empty()) text.note(t.note);
            text.table(t.columns, t.rows, false);
        }
    }
}

void SetupSection::add(const report::Summary & summary) {
    settings.insert(settings.end(), summary.settings.begin(), summary.settings.end());
    quantities.insert(quantities.end(), summary.quantities.begin(), summary.quantities.end());
}

SetupSection mesh_section(const MeshSetup & m) {
    using Kind = Column::Kind;
    SetupSection s{.title = "Mesh", .settings = m.settings};
    uint64_t n_cells = 0;
    std::string types;
    for (const auto & [type, n] : m.cells) {
        n_cells += n;
        types += (types.empty() ? "" : ", ") + logging::count(n) + " " + type;
    }
    s.quantities.push_back({Status::NONE, {"cells", logging::count(n_cells), "cells", "",
                                           m.cells.size() == 1 ? m.cells[0].first : types}});
    s.quantities.push_back({Status::NONE, {"faces", logging::count(m.faces), "faces", "", ""}});
    s.quantities.push_back({Status::NONE, {"boundary faces", logging::count(m.boundary_faces), "faces", "", ""}});

    SetupSection::Table ranges{.note = "", .columns = {"quantity", {"min", Kind::VALUE}, {"max", Kind::VALUE},
                                                       {"units", Kind::UNITS}, "basis"}, .rows = {}};
    const char * axes[] = {"x", "y", "z"};
    for (size_t d = 0; d < N_DIM; d++) {
        ranges.rows.push_back({Status::NONE, {std::string(axes[d]) + " extent", format("%.4g", m.lo[d]),
                                              format("%.4g", m.hi[d]), units::column("m"), "nodes"}});
    }
    ranges.rows.push_back({Status::NONE, {N_DIM == 2 ? "cell area" : "cell volume", format("%.3e", m.size_min),
                                          format("%.3e", m.size_max), units::column(N_DIM == 2 ? "m^2" : "m^3"),
                                          "ratio " + format("%.3g", m.size_min > 0.0 ? m.size_max / m.size_min : 0.0)}});
    if (m.partition) {
        const MeshSetup::Partition & p = *m.partition;
        s.settings.emplace_back("Partitioner", p.partitioner);
        s.settings.emplace_back("Halo exchange", p.exchange);
        const double mean = static_cast<double>(n_cells) / p.ranks;
        ranges.rows.push_back({Status::NONE, {"owned cells per rank", logging::count(p.owned_min),
                                              logging::count(p.owned_max), "cells",
                                              "imbalance " + format("%.3f", static_cast<double>(p.owned_max) / mean) +
                                                  " (largest / mean)"}});
        s.quantities.push_back({Status::NONE, {"ranks", std::to_string(p.ranks), "ranks", "", ""}});
        s.quantities.push_back({Status::NONE, {"halo layers", std::to_string(p.halo_layers), "layers", "", ""}});
        s.quantities.push_back({Status::NONE, {"halo cells", logging::count(p.halo_max), "cells", "",
                                               "most on a rank"}});
        s.quantities.push_back({Status::NONE, {"faces between ranks", logging::count(p.cut_faces), "faces", "", ""}});
        s.quantities.push_back({Status::NONE, {"halo layer-1 cells", logging::count(p.layer1_max), "cells", "",
                                               "most on a rank; " + logging::count(p.layer1_all) + " in all"}});
        s.quantities.push_back({Status::NONE, {"cells received per exchange", logging::count(p.received_max), "cells",
                                               "", "most on a rank; " + logging::count(p.received_all) + " in all"}});
    }
    s.tables.push_back(std::move(ranges));
    return s;
}

std::vector<Row> time_quantities(const TimeSetup & t) {
    std::vector<Row> rows;
    const std::string first = t.controlled ? "the first step; the step controller sets the others" : "every step";
    if (t.use_cfl) {
        rows.push_back({Status::NONE, {"CFL number", real(t.cfl), "-", "", first}});
    } else {
        rows.push_back({Status::NONE, {"time step", real(t.dt), units::column("s"), "", first}});
    }
    if (t.n_steps > 0) rows.push_back({Status::NONE, {"stop at step", logging::count(t.n_steps), "steps", "", ""}});
    if (t.t_stop > 0.0) {
        rows.push_back({Status::NONE, {"stop at time", real(t.t_stop), units::column("s"), "", "simulated time"}});
    }
    if (t.wall_stop > 0.0) {
        rows.push_back({Status::NONE, {"stop at wall time", real(t.wall_stop), "s", "", "of this run"}});
    }
    return rows;
}

SetupSection::Table boundary_table(const std::vector<CheckBoundary> & boundaries) {
    std::vector<Row> rows;
    for (const CheckBoundary & b : boundaries) {
        std::string details = b.details;
        if (!b.turbulence.empty()) details += (details.empty() ? "" : "\n") + ("turbulence: " + b.turbulence);
        rows.push_back({Status::NONE, {b.zone, b.condition, logging::count(b.faces), details.empty() ? "-" : details}});
    }
    return {.note = "", .columns = {"zone", "condition", "faces", "details"}, .rows = std::move(rows)};
}

SetupSection::Table output_table(const std::vector<OutputSummary> & outputs) {
    using Kind = Column::Kind;
    std::vector<Row> rows;
    for (const OutputSummary & o : outputs) {
        const bool steps = o.steps > 0;
        const bool timed = !steps && o.seconds > 0.0;
        rows.push_back({Status::NONE, {o.kind, o.file.empty() ? "-" : o.file,
                                       steps ? logging::count(o.steps) : timed ? real(o.seconds) : "-",
                                       steps ? "steps" : timed ? units::column("s") : "",
                                       o.contents.empty() ? "-" : o.contents}});
    }
    return {.note = "Intervals in steps, or in simulated time:",
            .columns = {"output", "file", {"every", Kind::VALUE}, {"units", Kind::UNITS}, "contents"},
            .rows = std::move(rows)};
}

report::Stream progress_stream(const ProgressColumns & c, const bool color) {
    using Kind = Column::Kind;
    std::vector<Field> fields = {
        {"step", "-", 11},
        {"t", units::column("s"), 10},
        {"dt", units::column("s"), 8},
        {"done", "%", 5},
        {"wall/step", "ms", 11},
        {"throughput", "cells/s", 8},
        {"ETA", "h:mm:ss", 9, Kind::TEXT},
        {"min rho", units::column("kg/m^3"), 10},
        {"min p", units::column("Pa"), 10},
        {"max Ma", "-", 7},
    };
    if (c.teno) fields.push_back({"troubled", "%", 6});
    if (c.mixture) {
        fields.push_back({"min T", "K", 7});
        fields.push_back({"max T", "K", 7});
        fields.push_back({"sum Y-1", "-", 9});
    }
    if (c.reacting) {
        fields.push_back({"reacting", "%", 6});
        fields.push_back({"chem/cell", "steps", 6});
    }
    fields.push_back({"written", "files", 0, Kind::TEXT});
    return report::Stream(std::move(fields), color);
}

std::vector<std::string> progress_cells(const ProgressColumns & c, const ProgressRow & r) {
    const bool timed = r.wall_per_step >= 0.0;
    std::vector<std::string> cells = {
        logging::count(r.step),
        format("%.4e", r.t),
        format("%.2e", r.dt),
        format("%.1f", 100.0 * r.done),
        timed ? format("%.2f", 1.0e3 * r.wall_per_step) : "-",
        timed ? format("%.2e", r.cells_per_second) : "-",
        r.done >= 1.0 ? clock_time(0.0) : r.eta >= 0.0 ? clock_time(r.eta) : "-",
        format("%.3e", r.min_rho),
        format("%.3e", r.min_p),
        format("%.3f", r.max_mach),
    };
    if (c.teno) cells.push_back(percent(r.troubled));
    if (c.mixture) {
        cells.push_back(format("%.1f", r.min_T));
        cells.push_back(format("%.1f", r.max_T));
        cells.push_back(format("%.2e", r.sum_Y_error));
    }
    if (c.reacting) {
        cells.push_back(percent(r.reacting));
        cells.push_back(format("%.0f", r.chemistry_steps));
    }
    cells.push_back(r.written.empty() ? "-" : r.written);
    return cells;
}

std::string progress_legend(const ProgressColumns & c) {
    std::string text = "done: of the run, by the stop that comes first; wall/step and throughput (cell updates per "
                       "second): time stepping since the previous row; ETA: wall time to the stop; min and max: over "
                       "the cells";
    if (c.teno) text += "; troubled: cells where TENO selects stencils (shocks)";
    if (c.mixture) text += "; sum Y-1: the largest |sum of the mass fractions - 1|";
    if (c.reacting) text += "; reacting: cells whose chemistry is advanced; chem/cell: most integrator steps of a cell";
    text += "; written: files since the previous row, named in mallard.jsonl and the summary (--verbose: a line each)";
    if (units::system() == units::System::NONDIMENSIONAL) text += ". Nondimensional case (physics.units): no units";
    return text + ".";
}

void summary_section(report::Text & text, const RunSummary & s) {
    text.section("Summary");
    text.line("Ended (" + s.outcome + "): " + s.stop + ", at step " + logging::count(s.step) + ", t = " +
              units::with(logging::real(s.t), "s"));

    const auto [unit, scale] = wall_unit(s.total);
    auto wall = [&](const double seconds) { return format("%.4g", seconds / scale); };
    auto of = [](const double part, const double whole) { return whole > 0.0 ? format("%.1f", 100.0 * part / whole) : "-"; };
    const bool chemistry = s.reacting && s.stepping > 0.0;
    std::vector<Row> phases = {
        {Status::NONE, {"setup", wall(s.setup), of(s.setup, s.total), ""}},
        {Status::NONE, {"time stepping", wall(s.stepping), of(s.stepping, s.total), "100.0"}},
    };
    if (chemistry) {
        phases.push_back({Status::NONE, {"  chemistry", wall(s.chemistry), of(s.chemistry, s.total),
                                         of(s.chemistry, s.stepping)}});
        if (s.chemistry_mean > 0.0) {
            const double waiting = s.chemistry_slowest - s.chemistry_mean;
            phases.push_back({Status::NONE, {"  waiting for the slowest chemistry rank", wall(waiting),
                                             of(waiting, s.total), of(waiting, s.stepping)}});
        }
    }
    phases.push_back({Status::NONE, {"diagnostics", wall(s.diagnostics), of(s.diagnostics, s.total), ""}});
    phases.push_back({Status::NONE, {"output", wall(s.output), of(s.output, s.total), ""}});
    phases.push_back({Status::NONE, {"total", wall(s.total), "100.0", ""}});
    std::vector<report::Column> header = {"phase", "wall time [" + std::string(unit) + "]", "of the total [%]"};
    if (chemistry) {
        header.push_back("of time stepping [%]");
    } else {
        for (Row & row : phases) row.cells.pop_back();
    }
    text.table(header, phases, false);

    std::vector<Row> rows;
    if (s.steps > 0 && s.stepping > 0.0) {
        rows.push_back({Status::NONE, {"throughput", format("%.3g", double(s.n_cells * s.steps) / s.stepping),
                                       "cells/s", "", "over " + logging::count(s.steps, "step")}});
        rows.push_back({Status::NONE, {"wall time per step", milliseconds(s.stepping / double(s.steps)),
                                       "ms", "", "time stepping"}});
    }
    if (s.rejected_counted) {
        rows.push_back({Status::NONE, {"rejected steps", logging::count(s.rejected), "steps", "", "error control"}});
    }
    if (s.reacting && s.chemistry_mean > 0.0) {
        rows.push_back({Status::NONE, {"chemistry of the slowest rank", format("%.4g", s.chemistry_slowest), "s", "",
                                       "mean over the ranks " + format("%.4g", s.chemistry_mean) + " s"}});
        if (s.chemistry_sent >= 0.0) {
            rows.push_back({Status::NONE, {"integrations on other ranks", format("%.1f", 100.0 * s.chemistry_sent),
                                           "%", "", "chemistry load balancing"}});
        }
    }
    if (s.writers) {
        const auto [value, bytes_unit] = bytes_parts(s.bytes);
        rows.push_back({Status::NONE, {"written", value, bytes_unit, "",
                                       "in " + format("%.3g", s.writing) + " s of writing" +
                                           (s.background ? ", in the background" : "")}});
        // Waits under a millisecond are only the queue's locking
        rows.push_back({Status::NONE, {"hashed for the run log", format("%.3g", s.hashing), "s", "",
                                       "off the main thread" +
                                           (s.hash_queue < 1e-3 ? std::string()
                                                                : ", the run waiting " + format("%.3g", s.hash_queue) +
                                                                      " s for it")}});
    }
    if (!rows.empty()) {
        text.blank();
        text.quantities(rows, false);
    }
    if (!s.files.empty()) {
        std::vector<Row> files;
        for (const RunSummary::Files & f : s.files) {
            files.push_back({Status::NONE, {f.kind, f.n > 0 ? logging::count(f.n) : "-", f.where}});
        }
        text.blank();
        text.note("Files (-: one file, appended to as the run goes):");
        text.table({"kind", "files", "index, or file"}, files, false);
    }
}

void health_section(report::Text & text, const health::Report & h) {
    text.section("Health");
    std::vector<Row> lowest;
    for (const health::Extremum & m : h.margins_run) {
        if (!m.valid) continue;
        const std::string unit = m.name == "rho" ? units::column("kg/m^3")
                                 : m.name == "p" ? units::column("Pa")
                                 : m.name == "T" ? units::column("K")
                                                 : "-";
        std::string at = "(";
        for (size_t i = 0; i < N_DIM; i++) at += (i ? ", " : "") + format("%.4g", m.x[i]);
        lowest.push_back({Status::NONE, {m.name, format("%.3g", m.value), unit, logging::count(m.step),
                                         units::with(at + ")", "m")}});
    }
    if (!lowest.empty()) {
        text.note("Lowest values at the progress rows, over the cells:");
        using Kind = Column::Kind;
        text.table({"quantity", {"value", Kind::VALUE}, {"units", Kind::UNITS}, "step", "at"}, lowest, false);
    }
    if (!h.activity.empty()) {
        std::vector<Row> rows;
        for (const health::Spread & a : h.activity) {
            rows.push_back({Status::NONE, {a.name, percent(a.all), percent(a.min), percent(a.max)}});
        }
        text.blank();
        text.note("Shares of the cells at the last progress row:");
        text.table({"activity", "all ranks [%]", "least rank [%]", "most rank [%]"}, rows, false);
    }
    if (!h.repairs_run.empty()) {
        std::vector<Row> rows;
        for (const health::Repair & r : h.repairs_run) {
            if (r.activations > 0) {
                rows.push_back({Status::NONE, {r.site, logging::count(r.activations), logging::count(r.cells)}});
            }
        }
        text.blank();
        if (rows.empty()) {
            text.line("Repairs: none");
        } else {
            text.note("Numerical repairs since the run started:");
            text.table({"repair", "times", "distinct cells"}, rows, false);
        }
    }
    std::vector<Row> balance;
    for (const health::Balance & b : h.balance_run) {
        if (b.checked) balance.push_back({Status::NONE, {b.name, format("%.1e", b.relative())}});
    }
    if (!balance.empty()) {
        text.blank();
        text.note("Conservation: the change of each total that the boundaries, sources and repairs do not explain, "
                  "relative to the total's scale:");
        text.table({"total", "unexplained change [-]"}, balance, false);
    }
    if (h.element_drift >= 0.0) {
        text.blank();
        text.line("Elements: largest relative change over a chemistry call " + format("%.1e", h.element_drift));
    }
    if (!h.timing.empty()) {
        std::vector<Row> rows;
        for (const health::Spread & t : h.timing) {
            rows.push_back({Status::NONE, {t.name, format("%.3g", 1.0e3 * t.max)}});
        }
        text.blank();
        text.table({"output", "per step, slowest rank [ms]"}, rows, false);
    }
}

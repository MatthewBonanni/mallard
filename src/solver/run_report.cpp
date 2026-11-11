/**
 * @file run_report.cpp
 * @brief The progress table, the summary and the health section of a run.
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

} // namespace

report::Stream progress_stream(const ProgressColumns & c, const bool color) {
    using Kind = Column::Kind;
    std::vector<Field> fields = {
        {"step", "-", 11},
        {"t", units::column("s"), 10},
        {"dt", units::column("s"), 8},
        {"done", "%", 5},
        {"wall/step", "ms", 9},
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
        timed ? milliseconds(r.wall_per_step) : "-",
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
                                       "cells/s", "", "over " + logging::count(s.steps) + " steps"}});
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

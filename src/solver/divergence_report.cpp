/**
 * @file divergence_report.cpp
 * @brief The likely causes of a divergence ranked from its evidence, and the report.
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 */

#include "divergence_report.h"

#include <algorithm>
#include <cmath>

#include "diagnostic_code.h"
#include "log.h"
#include "provenance.h"
#include "report.h"
#include "units.h"

namespace divergence {

const std::array<const char *, N_QUANTITIES> QUANTITY_NAMES = {"density", "pressure", "temperature", "mass fraction",
                                                               "velocity"};

double cfl_limit(const TimeIntegratorType type) {
    switch (type) {
        case TimeIntegratorType::FE: return 1.0;
        case TimeIntegratorType::SSPRK3: return 1.2;
        case TimeIntegratorType::RK4: return 1.35;
        default: return 2.2;
    }
}

namespace {

using logging::real;

constexpr std::array<const char *, N_QUANTITIES> QUANTITY_KEYS = {"density", "pressure", "temperature", "mass_fraction",
                                                                  "velocity"};

std::vector<double> pack_neighbors(const std::vector<uint64_t> & ids) {
    std::vector<double> out(Cell::MAX_NEIGHBORS, -1.0);
    for (size_t k = 0; k < std::min(ids.size(), Cell::MAX_NEIGHBORS); k++) out[k] = double(ids[k]);
    return out;
}

std::string point(const std::array<double, 3> & x) {
    std::string s = "(" + real(x[0]);
    for (int d = 1; d < N_DIM; d++) s += ", " + real(x[size_t(d)]);
    return s + ")";
}

std::string at(const std::array<double, 3> & x) { return units::with(point(x), "m"); }

/** @brief A column name with its units: "p [Pa]", or "p" in a nondimensional case. */
std::string named(const std::string & name, const std::string & si) {
    return units::system() == units::System::SI ? name + " [" + si + "]" : name;
}

/** @brief A value of a table: "%.4g", or "-" if not known. */
std::string value(const double v) { return std::isfinite(v) ? logging::format("%.4g", v) : "-"; }

std::string percent(double fraction) { return logging::format("%.0f%%", 100.0 * fraction); }

std::string cell_name(const Evidence & e, const Cell & c) {
    std::string s = "cell " + std::to_string(c.id) + " at " + at(c.x);
    if (e.n_ranks > 1) s += " on rank " + std::to_string(c.rank);
    return s;
}

/** @brief The cell's worst metric relative to its limit, its index, and whether it is over the limit. */
std::pair<double, size_t> worst_metric(const Cell & c) {
    double worst = 0.0;
    size_t m_worst = 0;
    for (size_t m = 0; m < MeshQuality::N_METRICS; m++) {
        if (!std::isfinite(c.quality[m])) continue;
        // Ratios are 1 for a perfect cell, angles and skewness 0
        const double base = m == 2 || m == 3 ? 1.0 : 0.0;
        const double r = (c.quality[m] - base) / (MeshQuality::LIMITS[m] - base);
        if (r > worst) worst = r, m_worst = m;
    }
    return {worst, m_worst};
}

bool notable(const Cell & c) {
    for (size_t m = 0; m < MeshQuality::N_METRICS; m++) {
        if (std::isfinite(c.quality[m]) && c.quality[m] > MeshQuality::NOTABLE[m]) return true;
    }
    return false;
}

std::string quality(const Cell & c) {
    std::string s;
    for (size_t m = 0; m < MeshQuality::N_METRICS; m++) {
        if (!std::isfinite(c.quality[m])) continue;
        s += (s.empty() ? "" : ", ") + std::string(MeshQuality::NAMES[m]) + " " +
             logging::format("%.3g", std::fabs(c.quality[m]) < 1e-9 ? 0.0 : c.quality[m]) +
             (std::string(MeshQuality::UNITS[m]) == "deg" ? " deg" : "");
    }
    return s.empty() ? "not known (a cell of another rank's halo)" : s;
}

/** @brief Whether the time step, not a few cells, is at fault: run.cfl itself is too large, or many cells were over. */
bool widespread_cfl(const Evidence & e) {
    if (e.use_cfl && !e.error_control && e.cfl_input > e.limit) return true;
    return e.n_over_limit > 4 && double(e.n_over_limit) > 0.1 * double(e.n_cells);
}

Cause cause(std::string kind, std::string title) {
    Cause c;
    c.kind = std::move(kind);
    c.title = std::move(title);
    return c;
}

Cause time_step(const Evidence & e) {
    Cause c = cause("time_step", "time step too large");
    if (!(e.dt > 0.0) || !(e.dt_cfl1 > 0.0)) return c;
    const double cfl = e.dt / e.dt_cfl1;
    const double fraction = e.n_cells ? double(e.n_over_limit) / double(e.n_cells) : 0.0;
    const bool widespread = widespread_cfl(e);
    const std::string limit = logging::format("%.3g", e.limit);
    if (cfl > e.limit) {
        c.score = widespread ? 0.9 : 0.55;
        c.evidence.push_back("the failing step ran at CFL " + logging::format("%.3g", cfl) +
                             (e.use_cfl && !e.error_control ? " (run.cfl = " + logging::format("%g", e.cfl_input) + ")"
                                                             : "") +
                             ", above " + limit + ", the largest stable value of " + e.integrator +
                             " for smooth flow");
        c.evidence.push_back(logging::count(e.n_over_limit) + " of " + logging::count(e.n_cells, "cell") + " (" +
                             percent(fraction) + ") were above it" +
                             (widespread ? "" : ": the excess is local, so look at those cells first"));
    } else if (cfl > 0.8 * e.limit) {
        c.score = 0.3;
        c.evidence.push_back("the failing step ran at CFL " + logging::format("%.3g", cfl) + ", close to " + limit +
                             ", the largest stable value of " + e.integrator + " for smooth flow");
    } else {
        c.score = 0.05;
        c.evidence.push_back("the failing step ran at CFL " + logging::format("%.3g", cfl) + ", within " + limit +
                             ", the largest stable value of " + e.integrator);
    }
    if (std::isfinite(e.focus.cfl)) {
        c.evidence.push_back("the local CFL number of " + cell_name(e, e.focus) + " was " +
                             logging::format("%.3g", e.focus.cfl));
    }
    // The guide's robust defaults: 1.0 for SSPRK3 and RK4, 2.0 for the integrators with larger limits
    const double target = e.limit < 1.2 ? 0.5 : e.limit < 2.0 ? 1.0 : 2.0;
    if (e.error_control) {
        c.fix = "tighten [numerics.error_control] tolerance";
    } else if (e.use_cfl) {
        const double lower = std::min(target, 0.5 * e.cfl_input);
        c.fix = "lower run.cfl to " + logging::format("%.2g", lower >= 0.1 ? std::floor(10.0 * lower) / 10.0 : lower);
    } else {
        c.fix = "lower run.dt below " + logging::format("%.3g", target * e.dt_cfl1) + " (CFL " +
                logging::format("%.2g", target) + " at the last valid state), or set run.cfl instead of run.dt";
    }
    return c;
}

Cause mesh(const Evidence & e) {
    Cause c = cause("mesh", "poor cells");
    std::vector<const Cell *> cells = {&e.focus};
    for (const Cell & n : e.neighbors) cells.push_back(&n);
    for (const Cell & n : e.others) cells.push_back(&n);
    double worst = 0.0;
    const Cell * poor = nullptr;
    std::vector<std::string> ids;
    for (const Cell * cell : cells) {
        const auto [r, m] = worst_metric(*cell);
        if (notable(*cell)) ids.push_back(std::to_string(cell->id));
        if (r > worst) worst = r, poor = cell;
    }
    if (!poor || ids.empty()) return c;
    const auto [r, m] = worst_metric(*poor);
    const bool over = r > 1.0;
    // Among the worst cells of the mesh
    const bool worst_of_mesh = poor->quality[m] >= 0.5 * e.mesh_max[m];
    const bool local_cfl = e.dt_cfl1 > 0.0 && e.dt / e.dt_cfl1 > e.limit && !widespread_cfl(e);
    c.score = over ? 0.8 : worst_of_mesh ? 0.5 : 0.2;
    if (local_cfl) c.score += 0.3;
    if (e.n_invalid <= 10) c.score += 0.1;
    c.score = std::min(c.score, 0.98);
    c.evidence.push_back(cell_name(e, *poor) + (poor->failed >= 0 ? ", which became invalid," : ", next to the failure,") +
                         " has " + std::string(MeshQuality::NAMES[m]) + " " + logging::format("%.3g", poor->quality[m]) +
                         " (questionable above " + logging::format("%g", MeshQuality::LIMITS[m]) + "; the mesh's worst is " +
                         logging::format("%.3g", e.mesh_max[m]) + ")");
    if (local_cfl) {
        c.evidence.push_back("only " + logging::count(e.n_over_limit, "cell") +
                             " exceeded the stable CFL number: the time step suits the rest of the mesh");
    }
    if (e.n_invalid <= 10) c.evidence.push_back("the failure is confined to " + logging::count(e.n_invalid, "cell"));
    std::sort(ids.begin(), ids.end());
    ids.erase(std::unique(ids.begin(), ids.end()), ids.end());
    std::string list;
    for (size_t k = 0; k < std::min<size_t>(ids.size(), 6); k++) list += (k ? ", " : "") + ids[k];
    c.fix = "improve cells " + list + " (mallard check input.toml --mesh-fields mesh writes the metrics for ParaView)" +
            (local_cfl ? ", or lower the time step to suit them" : "");
    return c;
}

Cause boundary(const Evidence & e, const Zone * zone) {
    Cause c = cause("boundary", "boundary condition");
    if (zone->near == 0 || e.n_invalid == 0) return c;
    const double fraction = double(zone->near) / double(e.n_invalid);
    c.title = "boundary condition of " + zone->name + " (" + zone->type + ")";
    c.score = 0.4 * fraction;
    const std::string compared = "p " + units::with(real(zone->p_ghost), "Pa") + " and rho " +
                                 units::with(real(zone->rho_ghost), "kg/m^3") + " outside, against p " +
                                 units::with(real(zone->p_cell), "Pa") + " and rho " +
                                 units::with(real(zone->rho_cell), "kg/m^3") + " in the cell next to it";
    if (zone->inadmissible) {
        c.score += 0.5;
        c.evidence.push_back("its exterior state is not a valid state: " + compared);
        c.fix = "check the values of " + zone->name +
                " (units, signs, and that its mass flow or pressure can be reached by the flow next to it)";
    } else if (zone->ratio > 10.0 || zone->jump > 3.0) {
        c.score += 0.5;
        c.evidence.push_back("its exterior state is far from the flow next to it: " + compared +
                             logging::format(" (ratio %.3g, velocity jump %.3g sound speeds)", zone->ratio, zone->jump));
        c.fix = "check the values of " + zone->name + " (units and signs), or start from a state closer to them";
    } else if (zone->ratio > 3.0 || zone->jump > 1.5) {
        c.score += 0.2;
        c.evidence.push_back("its exterior state differs from the flow next to it: " + compared);
        c.fix = "check the values of " + zone->name + " (units and signs)";
    } else if (zone->type == "outlet_extrapolation" || zone->type == "outlet_pressure") {
        c.fix = "check for flow entering through " + zone->name + "; a characteristic outlet (nscbc = true) or a "
                "longer domain lets disturbances leave";
    } else {
        c.fix = "check the condition and values of " + zone->name;
    }
    c.evidence.push_back(logging::count(zone->near) + " of the " + logging::count(e.n_invalid, "invalid cell") +
                         (e.n_invalid == 1 ? " is" : " are") + " within two cells of " + zone->name);
    return c;
}

/** @brief The zone whose condition is the likeliest cause. */
Cause boundary(const Evidence & e) {
    Cause best = cause("boundary", "boundary condition");
    for (const Zone & z : e.zones) {
        Cause c = boundary(e, &z);
        if (c.score > best.score) best = std::move(c);
    }
    return best;
}

Cause initial_state(const Evidence & e) {
    Cause c = cause("initial_state", e.from_restart ? "the state the run restarted from" : "initial state");
    const uint64_t steps = e.step - e.step_start;
    const std::string start = e.from_restart ? "the restart file's state" : "the initial state";
    const bool at_start = !e.has_valid || e.step_valid == e.step_start;
    if (e.n_inadmissible > 0 && at_start) {
        c.score = 0.95;
        c.evidence.push_back(start + " already has a non-positive density or pressure in " +
                             logging::count(e.n_inadmissible, "cell") + ", e.g. " + cell_name(e, e.inadmissible) + ": rho " +
                             units::with(real(e.inadmissible.rho), "kg/m^3") + ", p " +
                             units::with(real(e.inadmissible.p), "Pa"));
    } else if (steps <= 10) {
        c.score = 0.35;
        c.evidence.push_back("the run failed " + logging::count(steps, "step") +
                             " after its start, before the flow could develop");
    } else {
        return c;
    }
    double ratio = 1.0;
    for (const Cell & n : e.neighbors) {
        for (const auto & [a, b] : {std::make_pair(n.p, e.focus.p), std::make_pair(n.rho, e.focus.rho)}) {
            if (a > 0.0 && b > 0.0) ratio = std::max(ratio, std::max(a / b, b / a));
        }
    }
    if (ratio > 100.0 && c.score < 0.9) {
        c.score += 0.3;
        c.evidence.push_back(logging::format("the pressure or density of neighboring cells differ by a factor %.3g "
                                             "around ", ratio) + cell_name(e, e.focus));
    }
    c.fix = e.from_restart ? "check the restart file and what changed in the input since it was written"
                           : "check [initialize]: units, signs and the regions of its expressions (density, pressure "
                             "and temperature must be positive everywhere)";
    return c;
}

Cause chemistry(const Evidence & e) {
    Cause c = cause("chemistry", "chemistry stiffness");
    if (!e.reacting) return c;
    const bool thermo = e.focus.failed == TEMPERATURE || e.focus.failed == MASS_FRACTION ||
                        e.failed[TEMPERATURE] + e.failed[MASS_FRACTION] > e.n_invalid / 2;
    if (thermo) {
        c.score += 0.4;
        c.evidence.push_back(std::string("the first quantity to fail was the ") + QUANTITY_NAMES[size_t(e.focus.failed)]);
    }
    if (std::isfinite(e.focus.chemistry_cost) && e.chemistry_cost_mean > 0.0 &&
        e.focus.chemistry_cost > 10.0 * e.chemistry_cost_mean) {
        c.score += 0.2;
        c.evidence.push_back(logging::format("its cell needed %.0f chemistry sub-steps in the last step, against "
                                             "%.3g on average", e.focus.chemistry_cost, e.chemistry_cost_mean));
    }
    if (e.rejected_fraction > 0.1) {
        c.score += 0.2;
        c.evidence.push_back("the Rosenbrock integration rejected " + percent(e.rejected_fraction) +
                             " of its sub-steps since the last progress row");
    }
    c.fix = "lower run.cfl (the splitting error grows with it; detonations need about 0.35), or [chemistry] "
            "coupling = \"simpler\"";
    return c;
}

Cause numerics(const Evidence & e) {
    Cause c = cause("numerics", "the scheme's robustness");
    c.score = 0.15;
    if (!e.repairs.empty() && e.n_repaired > 0) {
        c.score += 0.3;
        std::string sites;
        for (const auto & s : e.repairs) sites += (sites.empty() ? "" : ", ") + s;
        c.evidence.push_back(logging::count(e.n_repaired) + " of the invalid cells were repaired (" + sites +
                             ") since the last progress row: the state was already out of bounds there");
    }
    if (e.teno && e.n_troubled > 0) {
        c.score += 0.1;
        c.evidence.push_back("TENO flagged " + logging::count(e.n_troubled) + " of the invalid cells as troubled");
    }
    if (e.limiter && e.n_limited > 0) {
        c.evidence.push_back("the MUSCL limiter was active in " + logging::count(e.n_limited) + " of the invalid cells");
    }
    if (c.evidence.empty()) c.evidence.push_back("no specific cause stands out; a more dissipative scheme often helps");
    c.fix = "a more dissipative scheme: numerics.preset = \"second_order\" (or \"first_order\")" +
            std::string(e.teno ? ", or bound_preserving = true for TENO" : "");
    return c;
}

json::Object cell_record(const Cell & c) {
    json::Object r;
    r.add("id", c.id).add("rank", c.rank);
    std::string x = "[";
    for (int d = 0; d < N_DIM; d++) x += (d ? "," : "") + json::number(c.x[size_t(d)]);
    r.raw("x", x + "]");
    if (c.failed >= 0) r.add("failed", std::string(QUANTITY_KEYS[size_t(c.failed)]));
    r.add("rho", c.rho).add("speed", c.speed).add("p", c.p).add("T", c.T).add("cfl", c.cfl);
    std::vector<std::pair<std::string, double>> q;
    for (size_t m = 0; m < MeshQuality::N_METRICS; m++) q.emplace_back(MeshQuality::FIELDS[m], c.quality[m]);
    r.add_numbers("quality", q);
    r.add("repaired", c.repaired).add("troubled", c.troubled);
    return r;
}

template <typename T, typename F>
std::string array(const std::vector<T> & items, F && f) {
    std::string s = "[";
    for (size_t k = 0; k < items.size(); k++) s += (k ? "," : "") + f(items[k]);
    return s + "]";
}

} // namespace

std::vector<double> Cell::pack() const {
    std::vector<double> r = {double(id), double(rank), x[0], x[1], x[2], double(failed), rho, speed, p, T, a, cfl};
    for (const double q : quality) r.push_back(q);
    r.insert(r.end(), {double(zone), double(layers), double(repaired), double(troubled), double(limited),
                       chemistry_cost, double(neighbors.size()), 0.0});
    const std::vector<double> n = pack_neighbors(neighbors);
    r.insert(r.end(), n.begin(), n.end());
    return r;
}

Cell Cell::unpack(const double * r) {
    Cell c;
    c.id = static_cast<uint64_t>(r[0]);
    c.rank = static_cast<int>(r[1]);
    c.x = {r[2], r[3], r[4]};
    c.failed = static_cast<int>(r[5]);
    c.rho = r[6], c.speed = r[7], c.p = r[8], c.T = r[9], c.a = r[10], c.cfl = r[11];
    for (size_t m = 0; m < MeshQuality::N_METRICS; m++) c.quality[m] = r[12 + m];
    c.zone = static_cast<int>(r[16]);
    c.layers = static_cast<int>(r[17]);
    c.repaired = r[18] != 0.0, c.troubled = r[19] != 0.0, c.limited = r[20] != 0.0;
    c.chemistry_cost = r[21];
    const size_t n = std::min(static_cast<size_t>(r[22]), MAX_NEIGHBORS);
    for (size_t k = 0; k < n; k++) c.neighbors.push_back(static_cast<uint64_t>(r[24 + k]));
    return c;
}

std::vector<Cause> rank_causes(const Evidence & e) {
    std::vector<Cause> causes = {initial_state(e), time_step(e), boundary(e), mesh(e), chemistry(e), numerics(e)};
    std::stable_sort(causes.begin(), causes.end(), [](const Cause & a, const Cause & b) { return a.score > b.score; });
    std::vector<Cause> kept;
    for (const Cause & c : causes) {
        if (kept.empty() || (c.score >= 0.25 && !c.evidence.empty())) kept.push_back(c);
    }
    return kept;
}

std::string text(const Evidence & e, const std::vector<Cause> & causes, const bool color) {
    using report::Column;
    using report::Row;
    using report::Status;
    report::Text text(color);
    text.section("Divergence report", Status::ERROR);
    text.note("Run " + provenance::run_id() + ".");
    text.note(std::string(units::system() == units::System::SI ? "SI units"
                                                               : "Nondimensional case (physics.units): no units") +
              ". Invalid: a non-finite value, or a non-positive density or pressure.");
    std::vector<Row> what = {{Status::ERROR, {"failing step", std::to_string(e.step), "-", "", ""}},
                             {Status::NONE, {"t", real(e.t), units::column("s"), "", ""}},
                             {Status::ERROR, {"invalid cells", logging::count(e.n_invalid), "cells", "",
                                              "of " + logging::count(e.n_cells)}}};
    for (size_t q = 0; q < N_QUANTITIES; q++) {
        if (e.failed[q] == 0) continue;
        what.push_back({Status::NONE, {std::string("  ") + QUANTITY_NAMES[q] + " failed first", logging::count(e.failed[q]),
                                       "cells", "", ""}});
    }
    if (e.has_valid) {
        what.push_back({Status::NONE, {"last valid step", std::to_string(e.step_valid), "-", "",
                                       "t = " + units::with(real(e.t_valid), "s") +
                                           (e.restart_file.empty() ? "" : ", in " + e.restart_file)}});
    }
    text.quantities(what);
    text.line("The invalid cells lie within " + point(e.lo) + " - " + at(e.hi) + ".");
    text.line("Closest to failing at the last valid state: " + cell_name(e, e.focus) +
              (e.focus.failed >= 0 ? std::string(" (its ") + QUANTITY_NAMES[size_t(e.focus.failed)] + " failed)" : "") +
              ".");
    if (!e.has_valid) {
        text.line("No valid state was kept" + std::string(e.step == e.step_start ? ": the run's first state is invalid."
                                                                                 : " ([run] keep_last_good = false)."));
    }

    text.section("Likely causes");
    text.note("Most likely first. The weight (0 to 1) ranks the causes of this failure; it is not a probability.");
    std::vector<Row> rows;
    for (size_t k = 0; k < causes.size(); k++) {
        const Cause & c = causes[k];
        std::string details;
        for (const auto & line : c.evidence) details += (details.empty() ? "" : "\n") + ("- " + line);
        if (!c.fix.empty()) details += (details.empty() ? "" : "\n") + ("Try: " + c.fix);
        rows.push_back({Status::NONE, {std::to_string(k + 1), c.title, logging::format("%.2f", c.score), details}});
    }
    text.table({"#", "cause", "weight", "evidence, and what to try"}, rows, false);

    text.section("Evidence");
    text.note(e.has_valid ? "The last valid state (step " + std::to_string(e.step_valid) + ")."
                          : "The failing state: no valid state was kept.");
    const bool ranks = e.n_ranks > 1;
    std::vector<Column> header = {"cell", "role"};
    if (ranks) header.emplace_back("rank");
    for (const auto & [name, si] : std::vector<std::pair<std::string, std::string>>{
             {"rho", "kg/m^3"}, {"|u|", "m/s"}, {"p", "Pa"}, {"T", "K"}}) {
        header.emplace_back(named(name, si));
    }
    header.emplace_back("CFL");
    header.emplace_back("at, quality, boundary and activity");
    std::vector<Row> cells;
    const auto add = [&](const Cell & c, const std::string & role) {
        std::string notes = at(c.x) + "\n" + quality(c);
        if (c.zone >= 0) {
            notes += "\nboundary " + e.zones[size_t(c.zone)].name + " (" + e.zones[size_t(c.zone)].type + ") " +
                     (c.layers == 0 ? "on one of its faces"
                                    : logging::count(uint64_t(c.layers), "cell") + " away");
        }
        std::string flags;
        if (c.repaired) flags += "repaired";
        if (c.troubled) flags += std::string(flags.empty() ? "" : ", ") + "TENO troubled";
        if (c.limited) flags += std::string(flags.empty() ? "" : ", ") + "limited";
        if (std::isfinite(c.chemistry_cost)) {
            flags += std::string(flags.empty() ? "" : ", ") + logging::format("%.0f chemistry sub-steps", c.chemistry_cost);
        }
        if (!flags.empty()) notes += "\n" + flags;
        std::vector<std::string> row = {std::to_string(c.id), role + (c.failed >= 0 ? ", invalid" : "")};
        if (ranks) row.push_back(std::to_string(c.rank));
        for (const double v : {c.rho, c.speed, c.p, c.T, c.cfl}) row.push_back(value(v));
        row.push_back(notes);
        cells.push_back({c.failed >= 0 ? Status::ERROR : Status::NONE, row});
    };
    add(e.focus, "closest");
    for (const Cell & n : e.neighbors) add(n, "neighbor");
    for (const Cell & n : e.others) {
        const bool listed = std::any_of(e.neighbors.begin(), e.neighbors.end(), [&](const Cell & m) { return m.id == n.id; });
        if (!listed) add(n, "other");
    }
    text.table(header, cells, false);
    text.note("closest: the invalid cell closest to failing at that state; neighbor: its face neighbors; other: "
              "the next invalid cells.");
    std::vector<Row> means = {{Status::NONE, {"mean density", real(e.rho_mean), units::column("kg/m^3"), "", ""}},
                              {Status::NONE, {"mean pressure", real(e.p_mean), units::column("Pa"), "", ""}}};
    if (e.n_inadmissible > 0) {
        means.push_back({Status::ERROR, {"cells with rho or p <= 0", logging::count(e.n_inadmissible), "cells", "",
                                         "e.g. " + cell_name(e, e.inadmissible)}});
    }
    text.quantities(means, e.n_inadmissible > 0);

    if (!e.history.empty()) {
        text.section("Progress rows before the failure");
        std::vector<Column> h = {"step", named("t", "s"), named("dt", "s"), "CFL", named("min rho", "kg/m^3"),
                                 named("min p", "Pa")};
        if (e.mixture) h.emplace_back(named("min T", "K"));
        std::vector<Row> history;
        for (const Sample & r : e.history) {
            std::vector<std::string> row = {std::to_string(r.step), logging::format("%.6g", r.t), value(r.dt),
                                            value(r.cfl), value(r.min_rho), value(r.min_p)};
            if (e.mixture) row.push_back(value(r.min_T));
            history.push_back({Status::NONE, row});
        }
        text.table(h, history, false);
    }

    text.section("Reproduce");
    if (!e.restart_file.empty()) {
        text.line("mallard -i <input> --restart " + e.restart_file);
        text.note("with [numerics] check_nan = true and [output] check_interval = 1 to follow it step by step");
    }
    if (!e.snapshot_file.empty()) {
        text.line(e.snapshot_file + ": the neighborhood of the invalid cells for ParaView (DIVERGED = 1 marks them; "
                  "the fields are the last valid state's, with the mesh quality)");
    }
    text.line("See " + diagnostic_code::docs(diagnostic_code::DIVERGED));
    return text.str();
}

json::Object record(const Evidence & e, const std::vector<Cause> & causes) {
    json::Object r;
    r.add("type", std::string("divergence")).add("run_id", provenance::run_id()).add("step", e.step).add("t", e.t);
    r.add("invalid_cells", e.n_invalid);
    std::vector<std::pair<std::string, double>> failed;
    for (size_t q = 0; q < N_QUANTITIES; q++) failed.emplace_back(QUANTITY_KEYS[q], double(e.failed[q]));
    r.add_numbers("failed", failed);
    r.raw("box", "[" + array(std::vector<double>(e.lo.begin(), e.lo.begin() + N_DIM), json::number) + "," +
                     array(std::vector<double>(e.hi.begin(), e.hi.begin() + N_DIM), json::number) + "]");
    if (e.has_valid) {
        json::Object v;
        v.add("step", e.step_valid).add("t", e.t_valid).add("file", e.restart_file);
        r.raw("last_valid", v.str());
    } else {
        r.raw("last_valid", "null");
    }
    r.raw("causes", array(causes, [](const Cause & c) {
        json::Object o;
        o.add("kind", c.kind).add("title", c.title).add("score", c.score).add("fix", c.fix);
        o.raw("evidence", array(c.evidence, [](const std::string & s) { return json::quote(s); }));
        return o.str();
    }));
    r.raw("cell", cell_record(e.focus).str());
    r.raw("neighbors", array(e.neighbors, [](const Cell & c) { return cell_record(c).str(); }));
    r.raw("others", array(e.others, [](const Cell & c) { return cell_record(c).str(); }));
    json::Object step;
    step.add("integrator", e.integrator).add("limit", e.limit).add("dt", e.dt).add("dt_cfl1", e.dt_cfl1);
    step.add("cfl", e.dt_cfl1 > 0.0 ? e.dt / e.dt_cfl1 : 0.0).add("cells_over_limit", e.n_over_limit);
    r.raw("time_step", step.str());
    r.raw("zones", array(e.zones, [](const Zone & z) {
        json::Object o;
        o.add("name", z.name).add("type", z.type).add("near", z.near);
        if (z.prescribed) o.add("inadmissible", z.inadmissible).add("ratio", z.ratio).add("jump", z.jump);
        return o.str();
    }));
    r.raw("history", array(e.history, [](const Sample & h) {
        json::Object o;
        o.add("step", h.step).add("t", h.t).add("dt", h.dt).add("cfl", h.cfl).add("min_rho", h.min_rho).add("min_p", h.min_p);
        if (std::isfinite(h.min_T)) o.add("min_T", h.min_T);
        return o.str();
    }));
    r.add("snapshot", e.snapshot_file);
    const bool si = units::system() == units::System::SI;
    json::Object u;
    for (const auto & [key, units] : std::vector<std::pair<const char *, const char *>>{
             {"t", "s"}, {"dt", "s"}, {"x", "m"}, {"rho", "kg m-3"}, {"speed", "m s-1"}, {"p", "Pa"}, {"T", "K"}}) {
        u.add(key, std::string(si ? units : "1"));
    }
    r.raw("units", u.str());
    return r;
}

} // namespace divergence

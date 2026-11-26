/**
 * @file health_events.cpp
 * @brief Run health events.
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 */

#include "health_events.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <string>

#include "diagnostic_code.h"
#include "json.h"
#include "log.h"

namespace health_events {

namespace {

double median(std::deque<double> v) {
    const auto mid = v.begin() + static_cast<std::ptrdiff_t>(v.size() / 2);
    std::nth_element(v.begin(), mid, v.end());
    return *mid;
}

void push(std::deque<double> & history, const double value) {
    if (!std::isfinite(value) || !(value > 0.0)) return;
    history.push_back(value);
    if (history.size() > HISTORY) history.pop_front();
}

std::string at(const uint64_t step) { return "step " + logging::count(step); }

Finding warning(const char * code, const uint64_t step, const double t, std::string message, const json::Object & values) {
    return {code, "", step, t, std::move(message), values.str()};
}

} // namespace

double job_end_from_environment() {
    const char * text = std::getenv(JOB_END_VARIABLE);
    if (!text || !*text) return 0.0;
    char * end = nullptr;
    const double value = std::strtod(text, &end);
    return end != text && std::isfinite(value) && value > 0.0 ? value : 0.0;
}

bool Monitor::Latch::trip() {
    hit = true;
    if (raised) return false;
    raised = true;
    return true;
}

void Monitor::Latch::end_row() {
    if (hit || !raised) {
        quiet = 0;
    } else if (++quiet >= REARM_ROWS) {
        raised = false;
        quiet = 0;
    }
    hit = false;
}

Monitor::Monitor(Settings settings, const uint64_t step, const uint64_t rejected)
    : config(std::move(settings)), row_step(step), row_rejected(rejected) {}

void Monitor::step(const uint64_t step, const double t, const double dt, const double dt_cfl1, std::vector<Finding> & out) {
    if (!(dt_cfl1 > 0.0)) return;
    const double cfl = dt / dt_cfl1;
    using logging::format;
    if (config.cfl_limit > 0.0 && !config.cfl_above_limit && cfl > config.cfl_limit * (1.0 + 1e-6) && cfl_limit.trip()) {
        out.push_back(warning(diagnostic_code::HEALTH_CFL_LIMIT, step, t,
                              format("CFL %.3g at ", cfl) + at(step) +
                                  format(" is above %.3g, the stability limit of %s for smooth flow", config.cfl_limit,
                                         config.integrator.c_str()),
                              json::Object().add("cfl", cfl).add("cfl_limit", config.cfl_limit).add("dt", dt).add(
                                  "dt_cfl1", dt_cfl1)));
    }
    if (cfls.size() >= MIN_HISTORY && cfl > CFL_SPIKE * cfl_median && cfl_spike.trip()) {
        out.push_back(warning(diagnostic_code::HEALTH_CFL_SPIKE, step, t,
                              format("CFL %.3g at ", cfl) + at(step) +
                                  format(" is %.3g times its median %.3g over the last %zu progress rows", cfl / cfl_median,
                                         cfl_median, cfls.size()),
                              json::Object().add("cfl", cfl).add("median", cfl_median).add("dt", dt).add("dt_cfl1", dt_cfl1)));
    }
    if (config.dt_follows_cfl && stable_dts.size() >= MIN_HISTORY && dt_cfl1 < DT_COLLAPSE * stable_dt_median &&
        dt_collapse.trip()) {
        out.push_back(warning(diagnostic_code::HEALTH_DT_COLLAPSE, step, t,
                              format("the stable time step %.3g s at ", dt_cfl1) + at(step) +
                                  format(" fell to %.3g of its median %.3g s over the last %zu progress rows",
                                         dt_cfl1 / stable_dt_median, stable_dt_median, stable_dts.size()),
                              json::Object().add("dt_cfl1", dt_cfl1).add("median", stable_dt_median).add("dt", dt)));
    }
}

void Monitor::row(const Row & r, std::vector<Finding> & out) {
    using logging::format;
    const uint64_t rejected = r.rejected - row_rejected, tried = r.step - row_step + rejected;
    if (rejected > 0 && static_cast<double>(rejected) > REJECTED_SHARE * static_cast<double>(tried) &&
        rejected_steps.trip()) {
        out.push_back(warning(diagnostic_code::HEALTH_REJECTED_STEPS, r.step, r.t,
                              logging::count(rejected) + " of the " + logging::count(tried) + " steps tried since " +
                                  at(row_step) + " were rejected by the error controller",
                              json::Object().add("rejected", rejected).add("tried", tried).add("since_step", row_step)));
    }
    for (Latch * latch : {&cfl_limit, &cfl_spike, &dt_collapse, &rejected_steps}) latch->end_row();
    row_step = r.step;
    row_rejected = r.rejected;
    push(cfls, r.cfl);
    push(stable_dts, r.dt_cfl1);
    if (!cfls.empty()) cfl_median = median(cfls);
    if (!stable_dts.empty()) stable_dt_median = median(stable_dts);

    if (config.steady_residual > 0.0 && !r.residuals.empty() && !converged) {
        auto largest = r.residuals.front();
        for (const auto & entry : r.residuals) {
            if (!(entry.second <= largest.second)) largest = entry;
        }
        json::Object values;
        for (const auto & [name, value] : r.residuals) values.add(name, value);
        values.add("steady_residual", config.steady_residual);
        if (largest.second <= config.steady_residual) {
            converged = true;
            out.push_back({nullptr, "converged", r.step, r.t,
                           format("every residual is below steady_residual = %.3g (largest: %s, %.3g)",
                                  config.steady_residual, largest.first.c_str(), largest.second),
                           values.str()});
        } else if (residual_mark < 0.0 || largest.second < 0.5 * residual_mark) {
            residual_mark = largest.second;
            residual_mark_step = r.step;
            stalled = false;
        } else if (!stalled && r.step - residual_mark_step >= config.stall_steps) {
            stalled = true;
            values.add("since_step", residual_mark_step).add("residual_since", residual_mark);
            out.push_back(warning(diagnostic_code::HEALTH_STALL, r.step, r.t,
                                  format("the residual of %s, %.3g, has not halved in ", largest.first.c_str(),
                                         largest.second) +
                                      logging::count(r.step - residual_mark_step) +
                                      format(" steps (%.3g at ", residual_mark) + at(residual_mark_step) +
                                      format("); the run is converged at %.3g", config.steady_residual),
                                  values));
        }
    }

    if (r.t_wall_stop > 0.0 && !wall_noted && r.wall >= WALL_NOTICE * r.t_wall_stop) {
        wall_noted = true;
        out.push_back({nullptr, "wall_time", r.step, r.t,
                       format("%.0f%% of t_wall_stop = ", 100.0 * WALL_NOTICE) + logging::duration(r.t_wall_stop) +
                           " used: the run stops in " + logging::duration(std::max(0.0, r.t_wall_stop - r.wall)),
                       json::Object().add("wall", r.wall).add("t_wall_stop", r.t_wall_stop).str()});
    }
    const bool stops_first = r.t_wall_stop > 0.0 && r.t_wall_stop <= config.job_end;
    if (config.job_end > 0.0 && !job_warned && !stops_first) {
        const double left = config.job_end - r.wall;
        const bool late = r.eta >= 0.0 && r.eta > left;
        if (late || left <= (1.0 - WALL_NOTICE) * config.job_end) {
            job_warned = true;
            json::Object values;
            values.add("job_left", std::max(0.0, left)).add_known("eta", r.eta);
            out.push_back(warning(
                diagnostic_code::HEALTH_JOB_TIME, r.step, r.t,
                "the job ends in " + logging::duration(std::max(0.0, left)) + " (" + JOB_END_VARIABLE + "), before the run" +
                    (r.eta >= 0.0 ? " (ETA " + logging::duration(r.eta) + ")" : std::string()) +
                    ": it will be killed without a final restart. Set t_wall_stop below it (mallard ctl set "
                    "t_wall_stop=...), or have Slurm signal the run before its limit (#SBATCH --signal=USR2@120)",
                values));
        }
    }

    for (const MonitorValues & m : r.monitored) {
        for (const auto & [name, value] : m.values) {
            if (std::isfinite(value) || !nonfinite.insert(m.name + "." + name).second) continue;
            out.push_back(warning(diagnostic_code::HEALTH_NONFINITE, r.step, r.t,
                                  "the monitored value " + name + " of " + m.name + " is " +
                                      (std::isnan(value) ? "NaN" : "infinite") + " (written at " + at(m.step) + ")",
                                  json::Object().add("monitor", m.name).add("value", name).add("step_written", m.step)));
        }
    }
}

} // namespace health_events

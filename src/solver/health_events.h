/**
 * @file health_events.h
 * @brief Run health events: warnings and events of the run log from values the host already has (each step's
 *        time step, the progress rows), with stable codes (docs/guide/output.md#run-health-events).
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 */

#ifndef HEALTH_EVENTS_H
#define HEALTH_EVENTS_H

#include <cstdint>
#include <deque>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "run_status.h"

namespace health_events {

inline constexpr size_t HISTORY = 20;           // progress rows a median is taken over
inline constexpr size_t MIN_HISTORY = 5;        // rows before spikes and collapses are judged
inline constexpr double CFL_SPIKE = 2.0;        // times the median CFL number
inline constexpr double DT_COLLAPSE = 0.1;      // of the median stable time step
inline constexpr double REJECTED_SHARE = 0.25;  // of the steps tried between two rows
inline constexpr int REARM_ROWS = 5;            // rows clear before a warning of a step can repeat
inline constexpr double WALL_NOTICE = 0.9;      // of t_wall_stop, and of the job's time
inline constexpr const char * JOB_END_VARIABLE = "SLURM_JOB_END_TIME";

/** @brief A warning (code set) or an event (kind set) of the run log. */
struct Finding {
    const char * code = nullptr;  // one of diagnostic_code::ALL, nullptr for an event
    std::string kind;             // the event's kind
    uint64_t step = 0;
    double t = 0.0;
    std::string message;
    std::string values;  // JSON object of the numbers it judged
};

struct Settings {
    std::string integrator;
    double cfl_limit = 0.0;         // the time integrator's for smooth flow, 0: none
    bool cfl_above_limit = false;   // run.cfl itself is above it, which W-CFL-LIMIT reported at setup
    bool dt_follows_cfl = false;    // dt is run.cfl times the stable time step
    double steady_residual = 0.0;   // [output] steady_residual, 0: none
    uint64_t stall_steps = 0;       // [output] stall_steps
    double job_end = 0.0;           // the scheduler's end of the job, on the solver's wall clock; 0: none
};

/** @brief A progress row as the host has it. */
struct Row {
    uint64_t step = 0;
    double t = 0.0;
    double dt_cfl1 = 0.0;
    double cfl = 0.0;
    double wall = 0.0;         // seconds since the solver started
    double eta = -1.0;         // seconds to the run's stop, -1: unknown
    double t_wall_stop = 0.0;  // seconds, 0 or less: none
    uint64_t rejected = 0;     // steps rejected by the error controller so far
    std::vector<std::pair<std::string, double>> residuals;  // empty: none at this row
    std::vector<MonitorValues> monitored;                   // the latest values of monitors and probes
};

/** @brief The scheduler's end of the job, from JOB_END_VARIABLE (Unix seconds); 0 if none. */
double job_end_from_environment();

class Monitor {
    public:
        Monitor() = default;
        /** @brief At a run's start, at step with rejected steps so far. */
        Monitor(Settings settings, uint64_t step, uint64_t rejected);

        /** @brief Each accepted step, at its end: the CFL number against its limit and median, the stable step's. */
        void step(uint64_t step, double t, double dt, double dt_cfl1, std::vector<Finding> & out);
        /** @brief Each progress row: rejected steps, the steady residual, the wall time, non-finite monitors. */
        void row(const Row & row, std::vector<Finding> & out);

        const Settings & settings() const { return config; }

    private:
        /** @brief Raised once, and again only after REARM_ROWS rows without its condition. */
        struct Latch {
            bool raised = false, hit = false;
            int quiet = 0;
            bool trip();
            void end_row();
        };

        Settings config;
        std::deque<double> cfls, stable_dts;
        double cfl_median = 0.0, stable_dt_median = 0.0;
        Latch cfl_limit, cfl_spike, dt_collapse, rejected_steps;
        uint64_t row_step = 0, row_rejected = 0;
        bool converged = false, stalled = false;
        double residual_mark = -1.0;
        uint64_t residual_mark_step = 0;
        bool wall_noted = false, job_warned = false;
        std::set<std::string> nonfinite;
};

} // namespace health_events

#endif // HEALTH_EVENTS_H

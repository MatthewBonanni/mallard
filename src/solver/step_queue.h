/**
 * @file step_queue.h
 * @author Matthew Bonanni (mbonanni001@gmail.com)
 * @brief Steps decided on the device: what each step's end does without the
 *        host (advance t and the step, retry a rejected step, stop for the
 *        host), and the record of each step that the host reads later.
 * @version 0.1
 * @date 2026-11-10
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 *
 * The host enqueues steps ahead of the device and reads their records after
 * they ran. A step after which the host must act (an output, a request, the
 * end of the run, a divergence, too many rejections) stops the device: the
 * steps enqueued after it do nothing, and the host acts at that step exactly as
 * it does when it waits for every step.
 *
 */

#ifndef STEP_QUEUE_H
#define STEP_QUEUE_H

#include <cstdint>
#include <limits>

#include <Kokkos_Core.hpp>

#include "common.h"

namespace step_queue {

/** @brief Rejections of one step before the run stops with an error (check_rejections). */
constexpr uint32_t MAX_TRIES = 50;

/** @brief What became of the step a launch took. */
enum Outcome : uint32_t {
    INACTIVE = 0,  // enqueued after a step the host acts on: did nothing
    ACCEPTED = 1,
    REJECTED = 2,  // by the step controller: the next launch takes the step again
    DIVERGED = 3,  // the state at the step's start is invalid
};

/** @brief Why the host must act after a step, as bits. */
enum Reason : uint32_t {
    REQUEST = 1,       // a run-control request reached the step's reduction
    STEP_DUE = 2,      // a step limit: output, n_steps, control files
    TIME_DUE = 4,      // a time limit: output by time, t_stop
    NONFINITE = 8,     // [numerics] check_nan found a non-finite value
    REJECTIONS = 16,   // MAX_TRIES rejections of the same step
    INVALID_DT = 32,   // an accepted step of dt <= 0
    DIVERGENCE = 64,
    CHEMISTRY = 128,   // the chemistry failed in some cell, or its balancing across ranks must grow
};

/**
 * @brief What the host decided at its last stop, for the steps until the next:
 *        it must act after the first step s, at time t, with s >= step_limit or
 *        t >= t_limit. Each of its conditions on a step's (s, t) is of one of
 *        these forms, so their minima are exact.
 */
struct Horizon {
    uint64_t step_limit = std::numeric_limits<uint64_t>::max();
    rtype t_limit = std::numeric_limits<rtype>::infinity();

    /** @brief The host acts at the next multiple of interval after step (interval > 0). */
    void at_multiple(uint64_t step, uint64_t interval) {
        if (interval == 0) return;
        const uint64_t next = (step / interval + 1) * interval;
        if (next < step_limit) step_limit = next;
    }
    void at_step(uint64_t s) {
        if (s < step_limit) step_limit = s;
    }
    /** @brief The host acts once t >= threshold. */
    void at_time(rtype threshold) {
        if (threshold < t_limit) t_limit = threshold;
    }
};

/** @brief Run health: minima of rho, p, T and the mass fractions with their cells, and sums of V rho, V p and V. */
struct Margins {
    static constexpr int N = 4;
    double v[N];
    int64_t at[N];
    double sum[3];
};

/** @brief A progress row's minima of rho and p, maximum Mach number and TENO-troubled cells. */
struct Diagnostics {
    rtype min_rho;
    rtype min_p;
    rtype max_mach;
    uint64_t n_troubled;
};

/** @brief A progress row's mixture minima: rho, p, -Mach, T, -T and -|sum Y - 1|. */
struct MixtureDiagnostics {
    rtype v[6];
};

/** @brief A progress row's chemistry: cells of the step's last call that needed it, sub-steps so far (taken and
 *         rejected) and the largest sub-step count of a cell. */
struct Chemistry {
    uint64_t last_active;
    uint64_t substeps[2];
    rtype max_cost;
};

/** @brief Run health: troubled cells, limited cells and the sum of upwind fractions. */
struct Activity {
    double v[3];
};

/** @brief Local sums and extrema of a progress row, before the reduction over the ranks. */
struct ProgressLocals {
    Diagnostics diagnostics;
    MixtureDiagnostics mixture;
    Chemistry chemistry;
    Margins margins;
    Activity activity;
    // The conservation ledger's boundary and source sums of the run
    double run_sum[2 * N_CONSERVATIVE];
};

/** @brief What a launch did, as the host reads it after the launch. */
struct Record {
    uint64_t seq;       // the launch's number
    uint32_t outcome;   // Outcome
    uint32_t reasons;   // Reason bits: the host acts after this step
    uint64_t step;      // at the step's start
    rtype t;            // at the step's start
    rtype t_end;        // t + dt, accepted steps
    rtype dt;
    rtype dt_cfl1;
    int32_t contributed;  // run-control requests of this rank in the step's reduction, as bits
    int32_t reduced;      // of every rank
    uint32_t tries;       // rejections of this step so far
    uint32_t progress;    // progress holds the step's progress row
    uint64_t chemistry_ns;  // of the step's chemistry calls, on the device's clock
    ProgressLocals progress_locals;
};

/** @brief Device-side state of the queue between launches. */
struct Loop {
    uint64_t seq;
    uint64_t step;
    uint32_t active;     // steps advance; 0 once the host must act
    uint32_t tries;
    uint32_t advanced;   // the current launch's step was accepted
    uint32_t progress;   // the current launch writes its progress row
    uint64_t check_interval;  // progress rows on the device every this many steps, 0: by the host
    Horizon horizon;
    double chemistry_failures;  // failed cells the host has seen, over the ranks
    uint64_t chemistry_start, chemistry_ns;  // of the current launch's chemistry calls
};

/** @brief The device's end of a step: inputs read after the step's kernels. */
struct StepEnd {
    bool valid;         // dt_cfl1 > 0 at the step's start
    bool controlled;
    bool accepted;      // by the step controller
    rtype dt;
    rtype t;
    int32_t reduced;    // requests over the ranks, as bits
};

/**
 * @brief The end of a step as the host takes it when it waits for every step
 *        (graph_step, then the run loop's checks), on the device: updates
 *        loop and fills the outcome, reasons and t_end of record.
 */
KOKKOS_INLINE_FUNCTION
void end_step(const StepEnd & in, Loop & loop, Record & record) {
    record.reasons = 0;
    record.t_end = in.t;
    loop.advanced = 0;
    if (!in.valid) {
        record.outcome = DIVERGED;
        record.reasons = DIVERGENCE;
    } else if (in.controlled && !in.accepted) {
        record.outcome = REJECTED;
        loop.tries++;
        if (loop.tries >= MAX_TRIES) record.reasons = REJECTIONS;
    } else {
        record.outcome = ACCEPTED;
        loop.tries = 0;
        if (!(in.dt > 0.0_r)) {
            record.reasons = INVALID_DT;
        } else {
            loop.advanced = 1;
            loop.step++;
            record.t_end = in.t + in.dt;
            if (in.reduced != 0) record.reasons |= REQUEST;
            if (loop.step >= loop.horizon.step_limit) record.reasons |= STEP_DUE;
            if (record.t_end >= loop.horizon.t_limit) record.reasons |= TIME_DUE;
        }
    }
    record.tries = loop.tries;
    if (record.reasons != 0) loop.active = 0;
    loop.progress = loop.advanced && loop.active && loop.check_interval > 0 && loop.step % loop.check_interval == 0;
}

} // namespace step_queue

#endif // STEP_QUEUE_H

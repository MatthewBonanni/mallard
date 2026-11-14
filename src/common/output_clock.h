/**
 * @file output_clock.h
 * @author Matthew Bonanni (mbonanni001@gmail.com)
 * @brief When a [[write_data]] output is due: the same arithmetic on the host
 *        (DataWriter) and in the step graph, which stages outputs.
 * @version 0.1
 * @date 2026-11-13
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 *
 */

#ifndef OUTPUT_CLOCK_H
#define OUTPUT_CLOCK_H

#include <cstdint>

#include <Kokkos_Core.hpp>

#include "common_typedef.h"

/** @brief A writer's schedule: every interval steps, else every time_interval of simulation time. */
struct OutputClock {
    uint64_t interval = 0;
    rtype time_interval = 0.0;
    uint64_t n_written = 0;  // by time: the next output is at n_written * time_interval

    // Each product and sum rounded on its own (volatile: never contracted), so that host and device agree bit for bit
    KOKKOS_INLINE_FUNCTION rtype next_time() const {
        if (interval > 0) return Kokkos::Experimental::infinity_v<rtype>;
        volatile rtype at = static_cast<rtype>(n_written) * time_interval;
        return at;
    }
    /** @brief By time: due once t is at least this; the tolerance absorbs round-off in accumulated time. */
    KOKKOS_INLINE_FUNCTION rtype due_time() const {
        volatile rtype tol = precision_tol(1e-9, 1e-5) * time_interval;
        return next_time() - tol;
    }
    KOKKOS_INLINE_FUNCTION bool due(const uint64_t step, const rtype t) const {
        return interval > 0 ? step % interval == 0 : t >= due_time();
    }
    /** @brief After an output at time t; by time, output times already passed are skipped (e.g. dt > time_interval). */
    KOKKOS_INLINE_FUNCTION void written(const rtype t) {
        if (interval > 0) {
            n_written++;
            return;
        }
        volatile rtype tol = precision_tol(1e-9, 1e-5) * time_interval;
        volatile rtype end = t + tol;
        while (next_time() <= end) n_written++;
    }
};

#endif // OUTPUT_CLOCK_H

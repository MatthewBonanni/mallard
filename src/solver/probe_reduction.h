/**
 * @file probe_reduction.h
 * @author Matthew Bonanni (mbonanni001@gmail.com)
 * @brief The arithmetic of probes' reductions, shared by the host's and the
 *        step graph's: per-sample terms summed exactly in fixed point
 *        (exact_sums.h), and the reduced values from those sums and the
 *        extrema, so both are the same bit for bit on any number of ranks.
 * @version 0.1
 * @date 2026-11-14
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 *
 */

#ifndef PROBE_REDUCTION_H
#define PROBE_REDUCTION_H

#include <cstdint>

#include <Kokkos_Core.hpp>

#include "common_math.h"

/** @brief A column of per-sample terms a probe's reductions sum: of its weight w and evaluated values q. */
struct ProbeTerm {
    enum Kind : uint32_t {
        WEIGHT,           // w
        WEIGHTED,         // w q[column]
        WEIGHTED_RHO,     // w q[rho] q[column]
        RHO,              // w q[rho]
        WEIGHTED_SQUARE,  // w q[column]^2
        MOMENT            // component `column` of (x - x_c) x w (q[vector[0]], q[vector[1]], q[vector[2]])
    };
    uint32_t kind = WEIGHT;
    uint32_t column = 0;
    uint32_t rho = 0;
    uint32_t vector[3] = {0, 0, 0};
    uint32_t n_dim = 0;  // of the vector's components
};

/** @brief A reduced value from the sums of the terms and the extrema of each evaluated variable. */
struct ProbeOutput {
    enum Kind : uint32_t {
        SUM,         // sums[a]
        RATIO,       // sums[a] / sums[b]
        ROOT_RATIO,  // sqrt(sums[a] / sums[b])
        MIN,         // lo[a]
        MAX          // hi[a]
    };
    uint32_t kind = SUM;
    uint32_t a = 0;
    uint32_t b = 0;
};

/**
 * @brief Term t of a sample of weight w, evaluated values q and position x relative to the moments' center
 *        (zero beyond N_DIM): each product rounded on its own, so host and device agree.
 */
template <typename Q>
KOKKOS_INLINE_FUNCTION double probe_term(const ProbeTerm & t, const double w, const Q * q, const double * x) {
    switch (t.kind) {
        case ProbeTerm::WEIGHT: return w;
        case ProbeTerm::WEIGHTED: return rounded_product(w, static_cast<double>(q[t.column]));
        case ProbeTerm::WEIGHTED_RHO:
            return rounded_product(rounded_product(w, static_cast<double>(q[t.rho])), static_cast<double>(q[t.column]));
        case ProbeTerm::RHO: return rounded_product(w, static_cast<double>(q[t.rho]));
        case ProbeTerm::WEIGHTED_SQUARE: {
            const double v = static_cast<double>(q[t.column]);
            return rounded_product(rounded_product(w, v), v);
        }
        case ProbeTerm::MOMENT: {
            double f[3] = {0.0, 0.0, 0.0};
            for (uint32_t d = 0; d < t.n_dim; d++) f[d] = rounded_product(w, static_cast<double>(q[t.vector[d]]));
            const uint32_t b = (t.column + 1) % 3, c = (t.column + 2) % 3;
            return rounded_product(x[b], f[c]) - rounded_product(x[c], f[b]);
        }
    }
    return 0.0;
}

/** @brief A reduced value; extrema of signed zeros are +0, whichever order found them. */
KOKKOS_INLINE_FUNCTION double probe_output(const ProbeOutput & o, const double * sums, const double * lo,
                                           const double * hi) {
    switch (o.kind) {
        case ProbeOutput::SUM: return sums[o.a];
        case ProbeOutput::RATIO: return sums[o.a] / sums[o.b];
        case ProbeOutput::ROOT_RATIO: return Kokkos::sqrt(sums[o.a] / sums[o.b]);
        case ProbeOutput::MIN: return lo[o.a] + 0.0;
        case ProbeOutput::MAX: return hi[o.a] + 0.0;
    }
    return 0.0;
}

#endif // PROBE_REDUCTION_H

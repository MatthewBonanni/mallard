/**
 * @file particle_injection.h
 * @author Matthew Bonanni (mbonanni001@gmail.com)
 * @brief Particle injectors ([[particles.injectors]]): parcels seeded during
 *        the run at a point, on boundary zones or in filled regions, with
 *        size distributions, from counter-based random numbers so that every
 *        rank count injects the same particles. Design:
 *        docs/design/particles.md.
 * @version 0.1
 * @date 2026-11-21
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 *
 */

#ifndef PARTICLE_INJECTION_H
#define PARTICLE_INJECTION_H

#include <array>
#include <cstdint>
#include <limits>
#include <string>
#include <utility>
#include <vector>

#include <Kokkos_Core.hpp>

#include "common.h"
#include "input.h"
#include "kexpr.h"
#include "particle_forces.h"
#include "pinned_mirrors.h"
#include "region.h"
#include "state.h"

class Mesh;
namespace region {
class Regions;
}
using RestartArrays = std::vector<std::pair<std::string, std::vector<double>>>;

namespace particles {

/** @brief Injectors of one run at most. */
inline constexpr uint32_t MAX_INJECTORS = 8;

/**
 * @brief Counter-based random numbers: draw n of the stream keyed by (seed,
 *        injector, index) is a hash of the four, so a particle's numbers do not
 *        depend on the rank, the thread or the order of kernels. SplitMix64's
 *        finalizer (Steele et al., 2014) chained over the key.
 */
struct Random {
    uint64_t key = 0;

    KOKKOS_INLINE_FUNCTION static uint64_t mix(uint64_t z) {
        z += 0x9e3779b97f4a7c15ull;
        z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ull;
        z = (z ^ (z >> 27)) * 0x94d049bb133111ebull;
        return z ^ (z >> 31);
    }

    KOKKOS_INLINE_FUNCTION Random(const uint64_t seed, const uint32_t injector, const uint64_t index)
        : key(mix(mix(mix(seed) ^ injector) ^ index)) {}

    /** @brief Draw n, uniform in [0, 1) with 53 random bits. */
    KOKKOS_INLINE_FUNCTION double uniform(const uint32_t n) const {
        return static_cast<double>(mix(key ^ mix(n)) >> 11) * 0x1.0p-53;
    }
};

/**
 * @brief Inverse transform sampling of a diameter from a tabulated number
 *        distribution: n nodes of ln d and of the cumulative distribution
 *        (cdf[0] = 0, cdf[n - 1] = 1), linear in ln d between nodes. One node
 *        holds the diameter itself, which every particle then has exactly.
 */
KOKKOS_INLINE_FUNCTION double sample_diameter(const double * log_d, const double * cdf, const uint32_t n, const double u) {
    if (n == 1) return log_d[0];
    uint32_t lo = 0, hi = n - 1;
    while (hi - lo > 1) {
        const uint32_t mid = (lo + hi) / 2;
        if (cdf[mid] <= u) {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    const double width = cdf[hi] - cdf[lo];
    const double f = width > 0.0 ? (u - cdf[lo]) / width : 0.0;
    return Kokkos::exp(log_d[lo] + f * (log_d[hi] - log_d[lo]));
}

/**
 * @brief A size distribution of [[particles.injectors]] diameter, as the table
 *        sample_diameter() reads, with its moments by number.
 */
struct SizeDistribution {
    std::vector<double> log_d, cdf;
    double mean_d = 0.0, mean_d2 = 0.0, mean_d3 = 0.0;  // by number
    double lo = 0.0, hi = 0.0;
    std::string summary;

    /** @brief The diameter of the uniform number u in [0, 1). */
    double sample(const double u) const {
        return sample_diameter(log_d.data(), cdf.data(), static_cast<uint32_t>(log_d.size()), u);
    }
};

/**
 * @brief The size distribution a diameter key gives: a number (every particle
 *        that diameter) or a table {distribution, basis, min, max and the
 *        distribution's parameters}. Distributions by mass are sampled by number.
 */
SizeDistribution read_size_distribution(const Input & holder, const std::string & key, const std::string & where);

/** @brief Where an injector puts its particles. */
enum class Site : uint8_t { POINT, ZONE, VOLUME };

/**
 * @brief The injector settings that may be expressions of (x, y, z, t), by their index in InjectorSpec::program:
 *        rate (of t only), the velocity's components, speed, the direction's components, cone_angle, temperature,
 *        weight, a diameter, and the parameters of a size distribution.
 */
namespace setting {
enum Index : uint32_t {
    RATE,
    SPEED,
    VELOCITY,
    DIRECTION = VELOCITY + N_DIM,
    CONE_ANGLE = DIRECTION + N_DIM,
    TEMPERATURE,
    WEIGHT,
    DIAMETER,
    SIZE,      // size, or median
    SPREAD,    // spread, or sigma
    EXPONENT,
    DIAMETER_MIN,
    DIAMETER_MAX,
    N_SETTINGS
};
} // namespace setting

/** @brief A size distribution whose parameters are expressions, sampled per parcel; TABLE: the tabulated one. */
enum class Law : uint8_t { TABLE, ROSIN_RAMMLER, LOG_NORMAL, NUKIYAMA_TANASAWA, UNIFORM };

/** @brief Quantiles the default bounds of a size distribution leave out at each end. */
inline constexpr double SIZE_TAIL = 1e-9;

/** @brief Blocks and cells per block of the grid in ln d that samples a size distribution of expressions. */
inline constexpr uint32_t LAW_BLOCKS = 32, LAW_CELLS = 32;

/**
 * @brief The density, in its own basis and up to a constant, of a size distribution at d: q = (size or median,
 *        spread or sigma, exponent, min, max). The same densities as read_size_distribution's.
 */
KOKKOS_INLINE_FUNCTION double law_density(const Law law, const double * q, const double d) {
    switch (law) {
        case Law::ROSIN_RAMMLER: {
            const double w = Kokkos::pow(d / q[0], q[1]);
            return q[1] / q[0] * Kokkos::pow(d / q[0], q[1] - 1.0) * Kokkos::exp(-w);
        }
        case Law::LOG_NORMAL: {
            const double z = Kokkos::log(d / q[0]) / q[1];
            return Kokkos::exp(-0.5 * z * z) / d;
        }
        case Law::NUKIYAMA_TANASAWA: return Kokkos::pow(d / q[0], q[2]) * Kokkos::exp(-Kokkos::pow(d / q[0], q[1]));
        case Law::UNIFORM: return d >= q[3] && d <= q[4] ? 1.0 : 0.0;
        default: return 0.0;
    }
}

/**
 * @brief The cumulative integral, by the trapezoidal rule, of g on the LAW_BLOCKS * LAW_CELLS cells of width h from
 *        a, at the ends of the blocks: sums[0] = 0, sums[LAW_BLOCKS] the total.
 */
template <typename G>
KOKKOS_INLINE_FUNCTION void block_sums(const G & g, const double a, const double h, double * sums) {
    sums[0] = 0.0;
    double previous = g(a);
    for (uint32_t b = 0; b < LAW_BLOCKS; b++) {
        double sum = 0.0;
        for (uint32_t k = 1; k <= LAW_CELLS; k++) {
            const double next = g(a + h * static_cast<double>(b * LAW_CELLS + k));
            sum += 0.5 * h * (previous + next);
            previous = next;
        }
        sums[b + 1] = sums[b] + sum;
    }
}

/**
 * @brief Where, in cells from a, the cumulative integral of g (block_sums) first reaches c: the cell's index plus
 *        the fraction of it, linear in the integral within the cell.
 */
template <typename G>
KOKKOS_INLINE_FUNCTION double reach(const G & g, const double a, const double h, const double * sums, const double c) {
    uint32_t b = 0;
    while (b + 1 < LAW_BLOCKS && sums[b + 1] < c) b++;
    double cumulative = sums[b];
    uint32_t i = b * LAW_CELLS;
    double previous = g(a + h * static_cast<double>(i));
    for (uint32_t k = 0; k < LAW_CELLS; k++, i++) {
        const double next = g(a + h * static_cast<double>(i + 1));
        const double step = 0.5 * h * (previous + next);
        if (cumulative + step >= c || k + 1 == LAW_CELLS) {
            const double f = step > 0.0 ? (c - cumulative) / step : 0.0;
            return static_cast<double>(i) + (f < 0.0 ? 0.0 : (f > 1.0 ? 1.0 : f));
        }
        cumulative += step;
        previous = next;
    }
    return static_cast<double>(i);
}

/**
 * @brief The diameter of the uniform number u in [0, 1) from a size distribution of parameters q (law_density),
 *        clipped to [q[3], q[4]] where has_min, has_max (else at the SIZE_TAIL quantiles in its basis), sampled by
 *        number: inverse transform sampling of the trapezoidal cumulative distribution in ln d, as
 *        read_size_distribution tabulates it, computed for this parcel. 0 if no diameters lie between the bounds.
 */
KOKKOS_INLINE_FUNCTION double sample_law(const Law law, const bool by_mass, const double * q, const bool has_min,
                                         const bool has_max, const double u) {
    const auto own = [&](const double s) {
        const double d = Kokkos::exp(s);
        return law_density(law, q, d) * d;
    };
    double sums[LAW_BLOCKS + 1];
    constexpr double n_cells = static_cast<double>(LAW_BLOCKS * LAW_CELLS);
    double lo = q[3], hi = q[4];
    if (law != Law::UNIFORM) {
        const double a = Kokkos::log(q[0]) - 18.0, h = 27.0 / n_cells;
        block_sums(own, a, h, sums);
        const double total = sums[LAW_BLOCKS];
        if (!(total > 0.0) || !Kokkos::isfinite(total)) return 0.0;
        const double first = Kokkos::floor(reach(own, a, h, sums, SIZE_TAIL * total));
        const double last = Kokkos::fmin(Kokkos::floor(reach(own, a, h, sums, (1.0 - SIZE_TAIL) * total)) + 1.0, n_cells);
        lo = has_min ? Kokkos::fmax(q[3], Kokkos::exp(a + h * first)) : Kokkos::exp(a + h * first);
        hi = has_max ? Kokkos::fmin(q[4], Kokkos::exp(a + h * last)) : Kokkos::exp(a + h * last);
    }
    if (!(hi > lo)) return 0.0;
    const double a = Kokkos::log(lo), h = (Kokkos::log(hi) - a) / n_cells;
    const auto number = [&](const double s) {
        const double d = Kokkos::exp(s);
        return law_density(law, q, d) * d / (by_mass ? d * d * d : 1.0);
    };
    block_sums(number, a, h, sums);
    const double total = sums[LAW_BLOCKS];
    if (!(total > 0.0) || !Kokkos::isfinite(total)) return 0.0;
    return Kokkos::exp(a + h * reach(number, a, h, sums, u * total));
}

/** @brief Rate injectors' panels per step, the deepest bisection of a panel and the evaluations of a step at most. */
inline constexpr uint32_t RATE_PANELS = 4, RATE_DEPTH = 30, RATE_EVALUATIONS = 2048;

/** @brief The error in parcels a rate injector's quadrature allows over a step: absolute, and relative. */
inline constexpr double RATE_TOLERANCE = 1e-7, RATE_RELATIVE_TOLERANCE = 1e-10;

/**
 * @brief The integral of rate(t) over [a, b]: adaptive Simpson quadrature with Richardson's correction on
 *        RATE_PANELS panels, each bisected where the halves' estimates differ by more than 15 times the
 *        interval's share of the tolerance (max(RATE_TOLERANCE, RATE_RELATIVE_TOLERANCE * |integral|)), at most
 *        RATE_DEPTH times and within RATE_EVALUATIONS evaluations. Deterministic: the same intervals in the same
 *        order on every rank. peak is the largest rate it evaluated.
 */
template <typename F>
KOKKOS_INLINE_FUNCTION double integrate_rate(const F & rate, const double a, const double b, double & peak) {
    struct Interval {
        double a, b, fa, fm, fb, whole, tolerance;
        uint32_t depth;
    };
    Interval stack[RATE_DEPTH + 2];
    uint32_t evaluations = 0;
    const auto f = [&](const double t) {
        const double v = rate(t);
        evaluations++;
        peak = Kokkos::fmax(peak, Kokkos::fabs(v));
        return v;
    };
    const double width = (b - a) / static_cast<double>(RATE_PANELS);
    double fa[RATE_PANELS + 1], fm[RATE_PANELS], estimate = 0.0;
    for (uint32_t k = 0; k <= RATE_PANELS; k++) fa[k] = f(k == RATE_PANELS ? b : a + width * static_cast<double>(k));
    for (uint32_t k = 0; k < RATE_PANELS; k++) {
        fm[k] = f(a + width * (static_cast<double>(k) + 0.5));
        estimate += width / 6.0 * (fa[k] + 4.0 * fm[k] + fa[k + 1]);
    }
    const double tolerance =
        Kokkos::fmax(RATE_TOLERANCE, RATE_RELATIVE_TOLERANCE * Kokkos::fabs(estimate)) / static_cast<double>(RATE_PANELS);
    double total = 0.0;
    for (uint32_t k = 0; k < RATE_PANELS; k++) {
        const double x0 = a + width * static_cast<double>(k), x1 = k + 1 == RATE_PANELS ? b : x0 + width;
        uint32_t n = 0;
        stack[n++] = {x0, x1, fa[k], fm[k], fa[k + 1], (x1 - x0) / 6.0 * (fa[k] + 4.0 * fm[k] + fa[k + 1]), tolerance, 0};
        while (n > 0) {
            const Interval in = stack[--n];
            const double m = 0.5 * (in.a + in.b);
            const double fl = f(0.5 * (in.a + m)), fr = f(0.5 * (m + in.b));
            const double left = (m - in.a) / 6.0 * (in.fa + 4.0 * fl + in.fm);
            const double right = (in.b - m) / 6.0 * (in.fm + 4.0 * fr + in.fb);
            const double delta = left + right - in.whole;
            if (in.depth >= RATE_DEPTH || evaluations >= RATE_EVALUATIONS || Kokkos::fabs(delta) <= 15.0 * in.tolerance) {
                total += left + right + delta / 15.0;
            } else {
                stack[n++] = {m, in.b, in.fm, fr, in.fb, right, 0.5 * in.tolerance, in.depth + 1};
                stack[n++] = {in.a, m, in.fa, fl, in.fm, left, 0.5 * in.tolerance, in.depth + 1};
            }
        }
    }
    return total;
}

/** @brief An injector's constants on the device. */
struct InjectorSpec {
    Site site = Site::POINT;
    uint8_t dimension = 0;     // of its simplices: 0 a point, N_DIM - 1 faces, N_DIM cells
    bool reject = false;       // points outside its region's shapes are drawn again
    bool along_normal = false; // the direction is the zone's inward normal at each face
    bool gas_velocity = false;     // solid particles start at the gas velocity where they enter
    bool gas_temperature = false;  // and at the gas temperature (with heat transfer)
    double rate = 0.0;         // parcels per second
    uint64_t count = 0;        // parcels released at once at start, in place of a rate
    double start = 0.0, end = 0.0;
    uint32_t entry_first = 0, entry_count = 0;
    uint32_t size_first = 0, size_count = 0;
    rtype speed = 0.0_r, cone = 0.0_r;
    rtype direction[N_DIM] = {};
    rtype weight = 1.0_r, temperature = 0.0_r;
    uint16_t program[setting::N_SETTINGS] = {};  // 1 + the index in Sampler::programs of a setting's expression, or 0
    double rate_scale = 1.0;  // parcels per unit of the rate's expression (mass_flow: 1 / (weight m))
    Law law = Law::TABLE;     // a size distribution of expressions, in place of the table
    bool by_mass = false;     // and its basis
};

/**
 * @brief What the draws of an injected particle choose, in order: its element
 *        (a face or cell over every rank), its diameter, its direction in the
 *        cone, then per attempt at a point in its element the simplex and the
 *        point in it.
 */
enum Draw : uint32_t { ENTRY = 0, DIAMETER = 1, CONE_POLAR = 2, CONE_AZIMUTH = 3, POINT = 8 };

/** @brief Attempts at a point inside an injector's region before the last one is kept. */
inline constexpr uint32_t MAX_ATTEMPTS = 1024;

/**
 * @brief The injectors' geometry and constants on the device, by value in
 *        kernels. Every rank holds the cumulative weights of every element
 *        (face or cell) of every injector in an order fixed by global ids, so
 *        each draws the same element for a particle; the rank owning it places
 *        the particle in it from its simplices.
 */
struct Sampler {
    InjectorSpec spec[MAX_INJECTORS];
    region::RegionView<> region[MAX_INJECTORS];
    uint32_t n = 0;
    uint64_t seed = 0;
    Kokkos::View<const double *> cdf;              // (entry): cumulative weight within its injector
    Kokkos::View<const int32_t *> local_of_entry;  // (entry): its local element, -1 on other ranks
    Kokkos::View<const uint32_t *> cell_of;        // (local element): the owned cell
    Kokkos::View<const rtype *[N_DIM]> normal_of;  // (local element): the inward normal (zones)
    Kokkos::View<const uint32_t *> simplex_offsets;  // (local element + 1)
    Kokkos::View<const double *> simplex_cdf;      // (simplex): cumulative measure within its element
    Kokkos::View<const double *[4][N_DIM]> vertices;  // (simplex, vertex, axis)
    Kokkos::View<const double *[N_DIM]> inside_of;    // (local element): a point in the region (rejection)
    Kokkos::View<const double *> size_log_d, size_cdf;
    kexpr::ProgramTable<> programs;          // the settings that are expressions
    Kokkos::View<uint64_t> fault;            // 1 + injector * N_SETTINGS + setting of the first invalid value, or 0
    Kokkos::View<double[5]> fault_at;        // its x, y, z, t and the value

    /** @brief Whether setting s of injector j is an expression. */
    KOKKOS_INLINE_FUNCTION bool given(const uint32_t j, const uint32_t s) const { return spec[j].program[s] != 0; }

    /** @brief Setting s of injector j at xt = (x, y, z, t). */
    KOKKOS_INLINE_FUNCTION double value(const uint32_t j, const uint32_t s, const double * xt) const {
        return programs(spec[j].program[s] - 1u, kexpr::ArrayVariables{xt});
    }

    /** @brief Record the first invalid value v of setting s of injector j, at xt; false. */
    KOKKOS_INLINE_FUNCTION bool invalid(const uint32_t j, const uint32_t s, const double * xt, const double v) const {
        const uint64_t key = 1u + uint64_t(j) * setting::N_SETTINGS + s;
        if (Kokkos::atomic_compare_exchange(&fault(), uint64_t(0), key) == 0) {
            for (int i = 0; i < 4; i++) fault_at(i) = xt[i];
            fault_at(4) = v;
        }
        return false;
    }

    /** @brief Setting s of injector j at xt into v, false (recorded) unless lo <= v (lo < v if open) and finite. */
    KOKKOS_INLINE_FUNCTION bool positive(const uint32_t j, const uint32_t s, const double * xt, double & v,
                                         const double lo = 0.0, const bool open = true) const {
        v = value(j, s, xt);
        if (!Kokkos::isfinite(v) || (open ? !(v > lo) : !(v >= lo))) return invalid(j, s, xt, v);
        return true;
    }

    /** @brief The rate of injector j at time t (0 where invalid, recorded). */
    KOKKOS_INLINE_FUNCTION double rate(const uint32_t j, const double t) const {
        const double xt[4] = {0.0, 0.0, 0.0, t};
        double v;
        return positive(j, setting::RATE, xt, v, 0.0, false) ? v * spec[j].rate_scale : 0.0;
    }

    /** @brief The element of injector j that draw ENTRY of r chooses, by weight. */
    KOKKOS_INLINE_FUNCTION uint32_t entry(const uint32_t j, const Random & r) const {
        const InjectorSpec & p = spec[j];
        const double target = r.uniform(ENTRY) * cdf(p.entry_first + p.entry_count - 1);
        uint32_t lo = 0, hi = p.entry_count - 1;
        while (lo < hi) {
            const uint32_t mid = (lo + hi) / 2;
            if (cdf(p.entry_first + mid) > target) {
                hi = mid;
            } else {
                lo = mid + 1;
            }
        }
        return p.entry_first + lo;
    }

    /** @brief A point uniform in local element e of injector j, from the draws of attempt a. */
    KOKKOS_INLINE_FUNCTION void point(const uint32_t j, const uint32_t e, const Random & r, const uint32_t a,
                                      double * x) const {
        const uint32_t first = simplex_offsets(e), last = simplex_offsets(e + 1) - 1;
        const uint32_t base = POINT + 4 * a;
        const double target = r.uniform(base) * simplex_cdf(last);
        uint32_t s = first;
        while (s < last && simplex_cdf(s) <= target) s++;
        // Barycentric weights from the spacings of k sorted uniforms
        const uint32_t k = spec[j].dimension;
        double u[4] = {0.0, 0.0, 0.0, 0.0};
        for (uint32_t i = 0; i < k; i++) u[i] = r.uniform(base + 1 + i);
        for (uint32_t i = 1; i < k; i++) {
            for (uint32_t m = i; m > 0 && u[m - 1] > u[m]; m--) {
                const double tmp = u[m];
                u[m] = u[m - 1];
                u[m - 1] = tmp;
            }
        }
        FOR_I_DIM x[i] = 0.0;
        double previous = 0.0;
        for (uint32_t v = 0; v <= k; v++) {
            const double next = v < k ? u[v] : 1.0;
            const double b = next - previous;
            previous = next;
            FOR_I_DIM x[i] += b * vertices(s, v, i);
        }
    }

    /** @brief The position, in local element e, of the particle of injector j with numbers r. */
    KOKKOS_INLINE_FUNCTION void place(const uint32_t j, const uint32_t e, const Random & r, double * x) const {
        for (uint32_t a = 0; a < MAX_ATTEMPTS; a++) {
            point(j, e, r, a, x);
            if (!spec[j].reject) return;
            const double q[3] = {x[0], x[1], N_DIM == 3 ? x[N_DIM - 1] : 0.0};
            if (region[j].distance(q) <= 0.0) return;
        }
        FOR_I_DIM x[i] = inside_of(e, i);
    }

    /**
     * @brief The velocity of the particle of injector j in local element e at xt = (x, y, z, t): its speed along a
     *        direction in the cone. False (recorded) if a setting that is an expression is invalid there.
     */
    KOKKOS_INLINE_FUNCTION bool velocity(const uint32_t j, const uint32_t e, const Random & r, const double * xt,
                                         rtype * v) const {
        const InjectorSpec & p = spec[j];
        double b[N_DIM];
        FOR_I_DIM b[i] = static_cast<double>(p.along_normal ? normal_of(e, i) : p.direction[i]);
        rtype speed = p.speed, cone_angle = p.cone;
        if (given(j, setting::VELOCITY) || given(j, setting::DIRECTION)) {
            const uint32_t first = given(j, setting::VELOCITY) ? setting::VELOCITY : setting::DIRECTION;
            double norm = 0.0;
            for (uint32_t i = 0; i < N_DIM; i++) {
                b[i] = value(j, first + i, xt);
                if (!Kokkos::isfinite(b[i])) return invalid(j, first + i, xt, b[i]);
                norm += b[i] * b[i];
            }
            norm = Kokkos::sqrt(norm);
            if (first == setting::DIRECTION && !(norm > 0.0)) return invalid(j, setting::DIRECTION, xt, norm);
            if (first == setting::VELOCITY) speed = static_cast<rtype>(norm);
            for (uint32_t i = 0; i < N_DIM; i++) b[i] = norm > 0.0 ? b[i] / norm : (i == 0 ? 1.0 : 0.0);
        }
        if (given(j, setting::SPEED)) {
            double s;
            if (!positive(j, setting::SPEED, xt, s, 0.0, false)) return false;
            speed = static_cast<rtype>(s);
        }
        if (given(j, setting::CONE_ANGLE)) {
            const double degrees = value(j, setting::CONE_ANGLE, xt);
            if (!(degrees >= 0.0 && degrees <= 90.0)) return invalid(j, setting::CONE_ANGLE, xt, degrees);
            cone_angle = static_cast<rtype>(degrees * Kokkos::numbers::pi_v<double> / 180.0);
        }
        if (cone_angle > 0.0_r) {
            const double cone = static_cast<double>(cone_angle);
            if constexpr (N_DIM == 2) {
                // Uniform in angle within the cone
                const double a = cone * (2.0 * r.uniform(CONE_POLAR) - 1.0);
                const double c = Kokkos::cos(a), s = Kokkos::sin(a);
                const double b0 = b[0];
                b[0] = c * b0 - s * b[1];
                b[1] = s * b0 + c * b[1];
            } else {
                // Uniform in solid angle within the cone about b, in the basis b, e1, e2 = b x e1
                const double a[3] = {b[0], b[1], b[N_DIM - 1]};
                const double cos_a = 1.0 - r.uniform(CONE_POLAR) * (1.0 - Kokkos::cos(cone));
                const double sin_a = Kokkos::sqrt(Kokkos::fmax(0.0, 1.0 - cos_a * cos_a));
                const double phi = 2.0 * Kokkos::numbers::pi_v<double> * r.uniform(CONE_AZIMUTH);
                int m = 0;
                for (int i = 1; i < 3; i++) {
                    if (Kokkos::fabs(a[i]) < Kokkos::fabs(a[m])) m = i;
                }
                double e1[3] = {0.0, 0.0, 0.0};
                e1[m] = 1.0;
                const double along = a[m];
                double norm = 0.0;
                for (int i = 0; i < 3; i++) {
                    e1[i] -= along * a[i];
                    norm += e1[i] * e1[i];
                }
                norm = Kokkos::sqrt(norm);
                for (int i = 0; i < 3; i++) e1[i] /= norm;
                const double e2[3] = {a[1] * e1[2] - a[2] * e1[1], a[2] * e1[0] - a[0] * e1[2], a[0] * e1[1] - a[1] * e1[0]};
                const double c = sin_a * Kokkos::cos(phi), s = sin_a * Kokkos::sin(phi);
                for (uint32_t i = 0; i < N_DIM; i++) b[i] = cos_a * a[i] + c * e1[i] + s * e2[i];
            }
        }
        FOR_I_DIM v[i] = static_cast<rtype>(static_cast<double>(speed) * b[i]);
        return true;
    }

    /**
     * @brief The diameter of the particle of injector j at xt = (x, y, z, t) into d: its diameter, or drawn from
     *        its size distribution. False (recorded) if a setting that is an expression is invalid there.
     */
    KOKKOS_INLINE_FUNCTION bool diameter(const uint32_t j, const Random & r, const double * xt, double & d) const {
        const InjectorSpec & p = spec[j];
        if (given(j, setting::DIAMETER)) return positive(j, setting::DIAMETER, xt, d);
        if (p.law == Law::TABLE) {
            d = sample_diameter(size_log_d.data() + p.size_first, size_cdf.data() + p.size_first, p.size_count,
                                r.uniform(Draw::DIAMETER));
            return true;
        }
        double q[5] = {1.0, 1.0, 0.0, 0.0, 0.0};
        if (p.law != Law::UNIFORM) {
            if (!positive(j, setting::SIZE, xt, q[0]) || !positive(j, setting::SPREAD, xt, q[1])) return false;
            if (p.law == Law::NUKIYAMA_TANASAWA && !positive(j, setting::EXPONENT, xt, q[2], -1.0)) return false;
        }
        if (given(j, setting::DIAMETER_MIN) && !positive(j, setting::DIAMETER_MIN, xt, q[3])) return false;
        if (given(j, setting::DIAMETER_MAX) && !positive(j, setting::DIAMETER_MAX, xt, q[4])) return false;
        d = sample_law(p.law, p.by_mass, q, given(j, setting::DIAMETER_MIN), given(j, setting::DIAMETER_MAX), r.uniform(Draw::DIAMETER));
        return d > 0.0 ? true : invalid(j, setting::DIAMETER, xt, d);
    }
};

/**
 * @brief The slots injected particles go into: Particles' arrays, and how a
 *        new particle's values start.
 */
struct InjectionTarget {
    Kokkos::View<uint64_t *> id;
    Kokkos::View<uint32_t *> cell;
    ParticleView x;
    uint32_t width = N_DIM;
    Kokkos::View<rtype **, Kokkos::LayoutRight> aux;
    Kokkos::View<const rtype *> initial_aux;
    Kokkos::View<rtype **, Kokkos::LayoutRight> props;  // (slot, k): diameter, weight (solid particles)
    bool solid = false;
    bool heated = false;
    Solid sphere;
    uint32_t first_quantity = N_DIM;
    Kokkos::View<uint8_t *> at_gas;  // (slot): SETTLE_VELOCITY | SETTLE_TEMPERATURE of the particles to take the gas's
};

/** @brief Bits of InjectionTarget::at_gas: the particle takes the gas velocity, temperature where it entered. */
inline constexpr uint8_t SETTLE_VELOCITY = 1, SETTLE_TEMPERATURE = 2;

/** @brief An injector as `mallard check` reports it, for drawing it and estimating its particles. */
struct InjectorInfo {
    std::string name;
    bool solid = false;
    std::string site;    // "point", "zone" or "region"
    std::string where;   // the point, the zones or the region, as given
    std::array<double, 3> center{};  // its point, or the centroid of its faces or volume
    std::array<std::array<double, 3>, 2> box{};
    double measure = 0.0;            // area (2D: length) of its faces, volume (2D: area) of its region
    bool along_normal = false;       // direction: the zones' inward normal
    bool gas_velocity = false;       // the gas velocity where each particle enters
    bool gas_temperature = false;    // the gas temperature there (with heat transfer)
    std::array<double, 3> direction{};
    double speed = 0.0, cone_angle = 0.0;  // m/s, degrees (half angle)
    double rate = 0.0;                     // parcels per second
    uint64_t count = 0;                    // parcels released at once at start, in place of a rate
    double mass = std::numeric_limits<double>::quiet_NaN();  // kg released at once (solid particles), if given
    double mass_flow = std::numeric_limits<double>::quiet_NaN();  // kg/s (solid particles)
    double start = 0.0, end = 0.0;
    double weight = 1.0;
    double mean_diameter = std::numeric_limits<double>::quiet_NaN();
    double sauter_diameter = std::numeric_limits<double>::quiet_NaN();
    double parcels = 0.0;  // expected over the run, from its start to t_stop (infinite without)
    std::vector<std::pair<std::string, std::string>> expressions;  // (key, text) of the settings that are expressions
};

/**
 * @brief The injectors of [[particles.injectors]]. After each accepted step,
 *        inject() adds to the particles the parcels each injector emitted over
 *        the step: its rate times the part of the step within [start, end),
 *        carrying the fraction of a parcel to the next step, or its count at
 *        once at the first step that reaches its start. Particle i of
 *        injector j draws everything from Random(seed, j, i), so the same
 *        particles enter on any rank count, with ids in order of injection.
 *        On the device, capturable in a step graph.
 */
class Injectors {
    public:
        /** @brief The defaults injectors take from [particles] and what the particles are. */
        struct Defaults {
            bool solid = false;
            bool heated = false;
            const SizeDistribution * diameter = nullptr;  // null: each injector gives its own
            rtype density = 0.0_r;
            rtype weight = 1.0_r;
            bool temperature_given = false;
            rtype temperature = 0.0_r;
        };

        /**
         * @brief Read [[particles.injectors]] of table and set up their geometry on this rank (collective when
         *        distributed).
         * @param global_cell The global id of each local cell.
         */
        void init(const Input & table, const Mesh & mesh, region::Regions & regions,
                  const std::vector<uint64_t> & global_cell, bool distributed, const Defaults & defaults);

        bool empty() const { return specs.empty(); }

        /** @brief Whether some injector's particles take the gas velocity or temperature where they enter. */
        bool at_gas() const {
            for (const InjectorSpec & p : specs) {
                if (p.gas_velocity || p.gas_temperature) return true;
            }
            return false;
        }
        uint32_t size() const { return static_cast<uint32_t>(specs.size()); }

        /** @brief Parcels the injectors emit from t0 to t1 together, an upper bound (each rounded up). */
        double parcels(double t0, double t1) const;

        /** @brief Parcels rate injector j emits from t0 to t1 (InputError where its rate is invalid). */
        double emitted(uint32_t j, double t0, double t1) const;

        /**
         * @brief Throw an InputError naming the injector, the key and the point of the first setting that was
         *        invalid where a particle entered (and the error of an expression that ran away), on every rank
         *        (collective when distributed).
         */
        void check_values() const;

        /** @brief Whether some injector emits for ever: no end. */
        bool endless() const;

        /**
         * @brief Continue from a restart file's PARTICLE_INJECTOR_<name> (parcels injected, fraction carried)
         *        and PARTICLE_NEXT_ID; an injector the file does not have starts afresh, and without
         *        PARTICLE_NEXT_ID ids continue from first_id.
         */
        void restore(const RestartArrays * restart, uint64_t first_id);

        /** @brief Scratch for slots of a rank of this capacity. */
        void reserve(uint32_t capacity);

        /**
         * @brief Inject from time t on, at the first call (the run's start), releasing there the count of each
         *        injector whose start it has reached.
         */
        void start(double t, const InjectionTarget & target);

        /** @brief Add the particles emitted up to the device time t_now (capturable). */
        void inject(const rtype * t_now, const InjectionTarget & target);

        /** @brief As inject(), at the host time t_now. */
        void inject(rtype t_now, const InjectionTarget & target);

        /** @brief Particles that found no free slot, on all ranks (collective when distributed). */
        uint64_t overflowed() const;

        /** @brief Enqueue copies of the injectors' state into pinned host mirrors (capturable). */
        void stage_output(PinnedMirrors & mirrors);

        /** @brief The state stage_output() staged: PARTICLE_NEXT_ID and PARTICLE_INJECTOR_<name> each. */
        RestartArrays restart_arrays() const;

        std::vector<std::string> summary() const;
        const std::vector<InjectorInfo> & info() const { return infos; }

    private:
        std::vector<InjectorSpec> specs;
        std::vector<std::string> names;
        std::vector<InjectorInfo> infos;
        std::vector<kexpr::Program> programs;  // the settings that are expressions
        std::vector<std::pair<std::string, std::size_t>> program_keys;  // and their keys and lines
        Sampler sampler;
        bool distributed = false;
        bool started = false;
        uint32_t n_local_entries = 0;

        Kokkos::View<uint64_t *> injected;   // (injector): parcels injected
        Kokkos::View<double *> carry;        // (injector): the fraction of a parcel emitted but not injected
        Kokkos::View<double> t_injected;     // time up to which every injector has injected
        Kokkos::View<uint64_t> next_id;
        Kokkos::View<uint64_t *> step_info;  // per step: offsets (n), firsts (n), total, first id, local, free
        Kokkos::View<uint64_t> n_overflow;
        Kokkos::View<uint32_t *> local_list;  // (k): the step's local particles
        Kokkos::View<uint32_t *> free_list;   // (k): free slots
        Kokkos::View<rtype> t_host;           // the host's time for inject(rtype)

        Kokkos::View<uint64_t *>::host_mirror_type h_injected;
        Kokkos::View<double *>::host_mirror_type h_carry;
        Kokkos::View<uint64_t>::host_mirror_type h_next_id;
};

} // namespace particles

#endif // PARTICLE_INJECTION_H

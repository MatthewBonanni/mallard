/**
 * @file particle_crossings.h
 * @author Matthew Bonanni (mbonanni001@gmail.com)
 * @brief Particle crossings ([[particles.crossings]]): a record of each
 *        particle that leaves through a boundary zone or crosses a surface;
 *        and particle fluxes ([[particles.fluxes]]): their counts per interval.
 * @version 0.1
 * @date 2026-11-21
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 *
 */

#ifndef PARTICLE_CROSSINGS_H
#define PARTICLE_CROSSINGS_H

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include <Kokkos_Core.hpp>

#include "common.h"
#include "particle_cloud.h"
#include "region.h"

namespace particles {

/** @brief Entries of [[particles.crossings]] (and of [[particles.fluxes]]), and surface pieces over all of them. */
inline constexpr uint32_t MAX_CROSSINGS = 8;
inline constexpr uint32_t MAX_ENTRIES = 2 * MAX_CROSSINGS;
inline constexpr uint32_t MAX_SURFACES = 8;
/** @brief Regions a surface piece is clipped to (inside or outside). */
inline constexpr uint32_t MAX_CLIPS = 2;

/** @brief A piece of a surface: where carrier's distance is zero, inside or outside each clip. */
struct Surface {
    region::RegionView<> carrier;
    region::RegionView<> clip[MAX_CLIPS];
    bool clip_inside[MAX_CLIPS] = {};
    uint32_t n_clips = 0;
    uint32_t entry = 0;
    double sign = 1.0;  // -1 where the piece faces into its carrier
};

/**
 * @brief The events of a step, appended on the device: the particle and its
 *        slot, the entry, the fraction of the step at which it crossed, its
 *        direction (+1 out of the surface's region, -1 into it, 0 leaving the
 *        domain) and where it crossed. The entries of tallied
 *        ([[particles.fluxes]]) count their events in tally(entry, 0 out or
 *        leaving, 1 in) instead: integers, the same on any rank count.
 */
struct CrossingEvents {
    Kokkos::View<uint32_t> count;
    Kokkos::View<uint64_t *> id;
    Kokkos::View<uint32_t *> slot, entry;
    Kokkos::View<double *> alpha;
    Kokkos::View<int8_t *> direction;
    Kokkos::View<double *[3]> x;
    Kokkos::View<uint64_t *[2], Kokkos::LayoutRight> tally;  // LayoutRight: the host reads it as (entry, direction) pairs
    uint32_t tallied = 0;

    KOKKOS_INLINE_FUNCTION void add(const uint64_t particle, const uint32_t s, const uint32_t e, const double a,
                                    const int8_t d, const double p[3]) const {
        if ((tallied >> e) & 1u) {
            Kokkos::atomic_add(&tally(e, d < 0 ? 1 : 0), uint64_t(1));
            return;
        }
        const uint32_t k = Kokkos::atomic_fetch_add(&count(), 1u);
        if (k >= slot.extent(0)) return;
        id(k) = particle;
        slot(k) = s;
        entry(k) = e;
        alpha(k) = a;
        direction(k) = d;
        for (int i = 0; i < 3; i++) x(k, i) = p[i];
    }
};

/**
 * @brief Records a particle leaving through face f of cell c (a boundary zone of
 *        the entries of mask) on its way from start to x, both in c's frame
 *        unless a periodic face lay between: where its path crosses the face's
 *        plane, else (a path through a periodic face or a rebound) the point
 *        of the plane nearest x, at the step's end.
 */
template <class Walker>
KOKKOS_INLINE_FUNCTION void record_exit(const CrossingEvents & events, const Walker & w, const uint32_t mask,
                                        const uint64_t particle, const uint32_t s, const uint32_t c, const uint32_t f,
                                        const rtype * x, const double * start) {
    rtype n[N_DIM], x0[N_DIM];
    FOR_I_DIM x0[i] = static_cast<rtype>(start[i]);
    const double d1 = static_cast<double>(w.distance(c, f, x, n));
    const double d0 = static_cast<double>(w.distance(c, f, x0, n));
    // A start on the plane within round-off crosses at the step's start
    const double along = d1 > d0 ? d0 / (d0 - d1) : -1.0;
    const bool path = along >= -0.5;
    const double alpha = path ? Kokkos::fmax(along, 0.0) : 1.0;
    double p[3] = {0.0, 0.0, 0.0};
    FOR_I_DIM {
        const double xi = static_cast<double>(x[i]);
        p[i] = path ? start[i] + alpha * (xi - start[i]) : xi - d1 * static_cast<double>(n[i]);
    }
    for (uint32_t e = 0; e < MAX_ENTRIES; e++) {
        if (mask & (1u << e)) events.add(particle, s, e, alpha, 0, p);
    }
}

/** @brief An entry of [[particles.crossings]] or [[particles.fluxes]], as the host writes it. */
struct Crossing {
    std::string name, file, region;
    bool exits = false;     // a boundary zone: the particles leave; else a surface they cross
    uint64_t interval = 0;  // [[particles.fluxes]]: steps between rows; 0 for a row per crossing
    double t_start = 0.0;   // [[particles.fluxes]]: the start of the interval of the next row
    std::vector<ParticleField> fields;
    std::vector<uint32_t> columns;  // particle variables of the fields, in order
    bool started = false;           // the run's provenance line is written
};

} // namespace particles

#endif // PARTICLE_CROSSINGS_H

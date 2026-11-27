/**
 * @file particle_trajectories.h
 * @author Matthew Bonanni (mbonanni001@gmail.com)
 * @brief Particle trajectories ([[particles.trajectories]]): the positions and
 *        fields of a subset of the particles, appended to a file on a schedule.
 * @version 0.1
 * @date 2026-11-21
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 *
 */

#ifndef PARTICLE_TRAJECTORIES_H
#define PARTICLE_TRAJECTORIES_H

#include <cstdint>
#include <string>
#include <vector>

#include "output_clock.h"
#include "particle_cloud.h"

namespace particles {

/** @brief Entries of [[particles.trajectories]]: a bit each in a particle's membership. */
inline constexpr uint32_t MAX_TRAJECTORIES = 8;

/**
 * @brief Whether particle id is in the fraction of the particles a trajectory
 *        keeps: a hash of the id alone, so the same on any rank count, and the
 *        particles of a smaller fraction are among those of a larger one.
 */
inline bool in_fraction(const uint64_t id, const double fraction) {
    uint64_t z = id + 0x9e3779b97f4a7c15ull;
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ull;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebull;
    z ^= z >> 31;
    return static_cast<double>(z >> 11) * 0x1.0p-53 < fraction;
}

/** @brief An entry of [[particles.trajectories]]. */
struct Trajectory {
    std::string name, file, region;
    double fraction = 1.0;
    OutputClock clock;
    uint32_t snapshot = 0;  // its staging (Particles::Snapshot)
    bool started = false;   // the run's provenance line is written
};

} // namespace particles

#endif // PARTICLE_TRAJECTORIES_H

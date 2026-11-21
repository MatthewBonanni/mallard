/**
 * @file particle_cloud.h
 * @author Matthew Bonanni (mbonanni001@gmail.com)
 * @brief The particles of a snapshot, as [[write_data]] writes them (particle_fields).
 * @version 0.1
 * @date 2026-11-21
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 *
 */

#ifndef PARTICLE_CLOUD_H
#define PARTICLE_CLOUD_H

#include <cstdint>
#include <string>
#include <vector>

/** @brief A field of particle snapshots: a particle variable or quantity, or the components of a vector. */
struct ParticleField {
    std::string name;
    uint32_t first = 0;         // its first column of ParticleCloud::values
    uint32_t n_components = 1;  // 1, or N_DIM for a vector
};

/** @brief A rank's particles of a snapshot, in no particular order: the ids identify them. */
struct ParticleCloud {
    std::vector<uint64_t> id;
    std::vector<double> x;       // 3 per particle, z = 0 in 2D
    std::vector<double> values;  // every field's columns per particle
};

#endif // PARTICLE_CLOUD_H

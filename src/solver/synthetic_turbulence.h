/**
 * @file synthetic_turbulence.h
 * @author Matthew Bonanni (mbonanni001@gmail.com)
 * @brief Divergence-free random velocity fields of a prescribed energy
 *        spectrum, as sums of Fourier modes, for initial states.
 * @version 0.1
 * @date 2026-11-04
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 *
 */

#ifndef SYNTHETIC_TURBULENCE_H
#define SYNTHETIC_TURBULENCE_H

#include <string>
#include <vector>

#include "input.h"

class Mesh;

namespace synthetic_turbulence {

/** @brief A Fourier mode of the velocity, a cos(k . x + phase), with a normal to k. */
struct Mode {
    double k[3];
    double a[3];
    double phase;
};

/** @brief The modes of a turbulence table, and its description for the run log. */
struct Field {
    std::vector<Mode> modes;
    std::string summary;
};

/**
 * @brief The field a turbulence table describes (docs/guide/initialization.md#synthetic-turbulence).
 *
 * The modes have the wavevectors of a box of the given period (default: the
 * mesh's bounding box), grouped into shells of width dk = 2 pi / (shortest
 * period) centered on j dk, up to k_max. Each shell holds its share of the
 * spectrum's energy, split evenly among at most modes_per_shell wavevectors,
 * chosen with the directions and phases by a hash of the seed and the
 * wavevector alone: the field does not depend on the ranks, and the modes of a
 * shell stay when k_max or other shells change. Collective (bounding box).
 *
 * @param table The turbulence table.
 * @param where Its key path, for messages.
 * @param mesh The mesh.
 */
Field read(const Input & table, const std::string & where, const Mesh & mesh);

} // namespace synthetic_turbulence

#endif // SYNTHETIC_TURBULENCE_H

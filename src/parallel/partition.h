/**
 * @file partition.h
 * @author Matthew Bonanni (mbonanni001@gmail.com)
 * @brief Cell partitioning for distributed runs.
 * @version 0.3
 * @date 2026-10-02
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 *
 */

#ifndef PARTITION_H
#define PARTITION_H

#include <array>
#include <cstdint>
#include <vector>

#include "common_typedef.h"

class Mesh;

/**
 * @brief Position along the Hilbert curve of a point in the box [lo, hi]
 *        (any dimension; 63 / N_DIM bits per axis).
 */
uint64_t hilbert_key(const std::array<double, N_DIM> & x, const std::array<double, N_DIM> & lo,
                     const std::array<double, N_DIM> & hi);

/**
 * @brief Owner rank of every cell: cells sorted along the Hilbert curve of their
 *        centroids, split into n_parts contiguous pieces of nearly equal size.
 */
std::vector<int> partition_hilbert(const Mesh & mesh, int n_parts);

#endif // PARTITION_H

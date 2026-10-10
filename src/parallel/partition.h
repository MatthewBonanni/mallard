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

class DistributedMesh;

/**
 * @brief Position along the Hilbert curve of a point in the box [lo, hi]
 *        (any dimension; 63 / N_DIM bits per axis).
 */
uint64_t hilbert_key(const std::array<double, N_DIM> & x, const std::array<double, N_DIM> & lo,
                     const std::array<double, N_DIM> & hi);

/**
 * @brief Owner rank of every block cell of a distributed mesh: cells sorted
 *        along the Hilbert curve of their vertex averages (ties by global id)
 *        by a distributed sample sort, split into n_parts contiguous pieces of
 *        nearly equal size (collective).
 */
std::vector<int> partition_hilbert(const DistributedMesh & mesh, int n_parts);

/**
 * @brief Parts per axis of the grid of n_parts blocks of a box with these
 *        extents that has the least total area between blocks.
 */
std::array<int, N_DIM> grid_shape(int n_parts, const std::array<double, N_DIM> & extent);

/**
 * @brief Owner rank of every block cell of a distributed mesh by multi-jagged
 *        coordinate partitioning (collective): the cells, keyed by the
 *        coordinates of their vertex averages (ties by global id), are split
 *        along x into equal slabs, each slab along y, and so on, with the
 *        number of pieces per axis from grid_shape() on their bounding box.
 *        A box of cuboid cells gives cuboid parts.
 */
std::vector<int> partition_multijagged(const DistributedMesh & mesh, int n_parts);

/**
 * @brief Owner rank of every block cell of a distributed mesh from dKaMinPar
 *        on its dual graph (collective). Requires Mallard_ENABLE_KAMINPAR.
 */
std::vector<int> partition_graph(const DistributedMesh & mesh, int n_parts);

/** @brief Whether this build has a graph partitioner. */
bool have_graph_partitioner();

#endif // PARTITION_H

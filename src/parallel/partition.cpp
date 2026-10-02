/**
 * @file partition.cpp
 * @author Matthew Bonanni (mbonanni001@gmail.com)
 * @brief Cell partitioning for distributed runs.
 * @version 0.3
 * @date 2026-10-02
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 *
 */

#include "partition.h"

#include <algorithm>
#include <limits>
#include <numeric>

#include "mesh.h"

uint64_t hilbert_key(const std::array<double, N_DIM> & x, const std::array<double, N_DIM> & lo,
                     const std::array<double, N_DIM> & hi) {
    constexpr int bits = 63 / N_DIM;
    constexpr uint32_t max_coord = (uint32_t(1) << bits) - 1;
    std::array<uint32_t, N_DIM> X;
    for (int d = 0; d < N_DIM; d++) {
        const double span = hi[d] - lo[d];
        const double t = span > 0.0 ? (x[d] - lo[d]) / span : 0.0;
        X[d] = static_cast<uint32_t>(std::clamp(t, 0.0, 1.0) * max_coord);
    }
    // Skilling, "Programming the Hilbert curve" (AIP Conf. Proc. 707, 2004): axes to transpose
    for (uint32_t Q = uint32_t(1) << (bits - 1); Q > 1; Q >>= 1) {
        const uint32_t P = Q - 1;
        for (int i = 0; i < N_DIM; i++) {
            if (X[i] & Q) {
                X[0] ^= P;
            } else {
                const uint32_t t = (X[0] ^ X[i]) & P;
                X[0] ^= t;
                X[i] ^= t;
            }
        }
    }
    for (int i = 1; i < N_DIM; i++) X[i] ^= X[i - 1];
    uint32_t t = 0;
    for (uint32_t Q = uint32_t(1) << (bits - 1); Q > 1; Q >>= 1) {
        if (X[N_DIM - 1] & Q) t ^= Q - 1;
    }
    for (int i = 0; i < N_DIM; i++) X[i] ^= t;
    // Interleave the transposed bits, most significant first
    uint64_t key = 0;
    for (int b = bits - 1; b >= 0; b--) {
        for (int i = 0; i < N_DIM; i++) key = (key << 1) | ((X[i] >> b) & 1u);
    }
    return key;
}

std::vector<int> partition_hilbert(const Mesh & mesh, int n_parts) {
    const uint32_t n = mesh.n_cells;
    std::array<double, N_DIM> lo, hi;
    lo.fill(std::numeric_limits<double>::max());
    hi.fill(std::numeric_limits<double>::lowest());
    for (uint32_t c = 0; c < n; c++) {
        for (int d = 0; d < N_DIM; d++) {
            lo[d] = std::min(lo[d], double(mesh.h_cell_coords(c, d)));
            hi[d] = std::max(hi[d], double(mesh.h_cell_coords(c, d)));
        }
    }
    std::vector<uint64_t> key(n);
    for (uint32_t c = 0; c < n; c++) {
        std::array<double, N_DIM> x;
        for (int d = 0; d < N_DIM; d++) x[d] = mesh.h_cell_coords(c, d);
        key[c] = hilbert_key(x, lo, hi);
    }
    std::vector<uint32_t> order(n);
    std::iota(order.begin(), order.end(), 0u);
    std::stable_sort(order.begin(), order.end(), [&](uint32_t a, uint32_t b) { return key[a] < key[b]; });
    std::vector<int> owner(n);
    for (uint32_t k = 0; k < n; k++) {
        owner[order[k]] = static_cast<int>((uint64_t(k) * n_parts) / n);
    }
    return owner;
}

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
#include <bit>
#include <limits>
#include <numeric>
#include <stdexcept>

#include "comm.h"
#include "distributed_mesh.h"

#ifdef Mallard_HAS_KAMINPAR
#include <dkaminpar.h>
#endif

uint64_t hilbert_key(const std::array<double, N_DIM> & x, const std::array<double, N_DIM> & lo,
                     const std::array<double, N_DIM> & hi) {
    constexpr int bits = 63 / N_DIM;
    constexpr uint32_t max_coord = (uint32_t(1) << bits) - 1;
    std::array<uint32_t, N_DIM> X;
    for (size_t d = 0; d < N_DIM; d++) {
        const double span = hi[d] - lo[d];
        const double t = span > 0.0 ? (x[d] - lo[d]) / span : 0.0;
        X[d] = static_cast<uint32_t>(std::clamp(t, 0.0, 1.0) * max_coord);
    }
    // Skilling, "Programming the Hilbert curve" (AIP Conf. Proc. 707, 2004): axes to transpose
    for (uint32_t Q = uint32_t(1) << (bits - 1); Q > 1; Q >>= 1) {
        const uint32_t P = Q - 1;
        for (size_t i = 0; i < N_DIM; i++) {
            if (X[i] & Q) {
                X[0] ^= P;
            } else {
                const uint32_t t = (X[0] ^ X[i]) & P;
                X[0] ^= t;
                X[i] ^= t;
            }
        }
    }
    for (size_t i = 1; i < N_DIM; i++) X[i] ^= X[i - 1];
    uint32_t t = 0;
    for (uint32_t Q = uint32_t(1) << (bits - 1); Q > 1; Q >>= 1) {
        if (X[N_DIM - 1] & Q) t ^= Q - 1;
    }
    for (size_t i = 0; i < N_DIM; i++) X[i] ^= t;
    // Interleave the transposed bits, most significant first
    uint64_t key = 0;
    for (int b = bits - 1; b >= 0; b--) {
        for (size_t i = 0; i < N_DIM; i++) key = (key << 1) | ((X[i] >> b) & 1u);
    }
    return key;
}


bool have_graph_partitioner() {
#ifdef Mallard_HAS_KAMINPAR
    return true;
#else
    return false;
#endif
}


std::vector<int> partition_hilbert(const DistributedMesh & mesh, int n_parts) {
    const size_t p = static_cast<size_t>(comm::size());
    const auto centers = mesh.block_cell_centers();
    std::array<double, N_DIM> lo, hi;
    lo.fill(std::numeric_limits<double>::max());
    hi.fill(std::numeric_limits<double>::lowest());
    for (const auto & x : centers) {
        for (size_t d = 0; d < N_DIM; d++) {
            lo[d] = std::min(lo[d], x[d]);
            hi[d] = std::max(hi[d], x[d]);
        }
    }
    lo = comm::allreduce(lo, comm::Op::MIN);
    hi = comm::allreduce(hi, comm::Op::MAX);
    using Item = std::pair<uint64_t, uint64_t>;  // (key, global id): unique, so splitters are exact
    std::vector<Item> items(centers.size());
    for (size_t c = 0; c < centers.size(); c++) items[c] = {hilbert_key(centers[c], lo, hi), mesh.first_cell() + c};
    std::sort(items.begin(), items.end());

    // Sample sort: evenly spaced samples from every rank pick p - 1 splitters.
    // Parts come from global positions, so splitters only balance the sort
    // itself; a bounded oversampling keeps the gathered samples O(p).
    constexpr size_t MAX_SAMPLES = 64;
    const size_t n_samples = std::min(p, MAX_SAMPLES);
    std::vector<uint64_t> samples;
    for (size_t k = 0; k < n_samples && !items.empty(); k++) {
        const Item & s = items[(items.size() * k) / n_samples];
        samples.push_back(s.first);
        samples.push_back(s.second);
    }
    const std::vector<uint64_t> all = comm::allgatherv(samples);
    std::vector<Item> sorted_samples;
    for (size_t i = 0; i < all.size(); i += 2) sorted_samples.push_back({all[i], all[i + 1]});
    std::sort(sorted_samples.begin(), sorted_samples.end());
    std::vector<Item> splitters;
    for (size_t k = 1; k < p && !sorted_samples.empty(); k++) {
        splitters.push_back(sorted_samples[(sorted_samples.size() * k) / p]);
    }
    std::vector<std::vector<uint64_t>> send(p);
    for (const Item & item : items) {
        const size_t dest = static_cast<size_t>(std::upper_bound(splitters.begin(), splitters.end(), item) - splitters.begin());
        send[dest].push_back(item.first);
        send[dest].push_back(item.second);
    }
    items.clear();
    for (const auto & from : comm::alltoallv(send)) {
        for (size_t i = 0; i < from.size(); i += 2) items.push_back({from[i], from[i + 1]});
    }
    std::sort(items.begin(), items.end());

    // Position along the curve -> part, sent to each cell's block rank
    const std::vector<uint64_t> counts = comm::allgatherv(std::vector<uint64_t>{items.size()});
    uint64_t first = 0;
    for (size_t r = 0; r < static_cast<size_t>(comm::rank()); r++) first += counts[r];
    const uint64_t n = mesh.n_global_cells();
    const auto & dist = mesh.cell_distribution();
    send.assign(p, {});
    for (size_t k = 0; k < items.size(); k++) {
        const uint64_t g = items[k].second;
        const size_t r = static_cast<size_t>(std::upper_bound(dist.begin(), dist.end(), g) - dist.begin() - 1);
        send[r].push_back(g);
        send[r].push_back(((first + k) * static_cast<uint64_t>(n_parts)) / n);
    }
    std::vector<int> owner(mesh.n_block_cells());
    for (const auto & from : comm::alltoallv(send)) {
        for (size_t i = 0; i < from.size(); i += 2) owner[from[i] - mesh.first_cell()] = static_cast<int>(from[i + 1]);
    }
    return owner;
}

std::array<int, N_DIM> grid_shape(int n_parts, const std::array<double, N_DIM> & extent) {
    std::array<int, N_DIM> best{}, shape{};
    double best_area = std::numeric_limits<double>::max();
    auto search = [&](auto & self, int d, int rest) -> void {
        if (d == N_DIM - 1) {
            shape[d] = rest;
            double area = 0.0;
            for (int i = 0; i < N_DIM; i++) {
                double face = 1.0;
                for (int j = 0; j < N_DIM; j++) face *= j == i ? 1.0 : extent[j];
                area += (shape[i] - 1) * face;
            }
            if (area < best_area) {
                best_area = area;
                best = shape;
            }
            return;
        }
        // Ties go to more pieces along earlier axes: cells numbered x first
        // then sit closer in memory to their neighbors across later axes
        for (int k = rest; k >= 1; k--) {
            if (rest % k != 0) continue;
            shape[d] = k;
            self(self, d + 1, rest / k);
        }
    };
    search(search, 0, n_parts);
    return best;
}

namespace {

/** @brief Unsigned key that sorts as the double does. */
uint64_t ordered_bits(double x) {
    if (x == 0.0) x = 0.0;  // -0.0 sorts with 0.0
    const uint64_t bits = std::bit_cast<uint64_t>(x);
    return (bits >> 63) ? ~bits : bits | (uint64_t(1) << 63);
}

} // namespace

std::vector<int> partition_multijagged(const DistributedMesh & mesh, int n_parts) {
    const auto centers = mesh.block_cell_centers();
    std::array<double, N_DIM> lo, hi, extent;
    lo.fill(std::numeric_limits<double>::max());
    hi.fill(std::numeric_limits<double>::lowest());
    for (const auto & x : centers) {
        for (int d = 0; d < N_DIM; d++) {
            lo[d] = std::min(lo[d], x[d]);
            hi[d] = std::max(hi[d], x[d]);
        }
    }
    lo = comm::allreduce(lo, comm::Op::MIN);
    hi = comm::allreduce(hi, comm::Op::MAX);
    for (int d = 0; d < N_DIM; d++) extent[d] = std::max(hi[d] - lo[d], 0.0);
    const std::array<int, N_DIM> shape = grid_shape(n_parts, extent);

    // Each axis splits every group of the previous axes into shape[d] pieces of
    // equal count, at keys (coordinate, global id) found by distributed bisection
    using Key = std::pair<uint64_t, uint64_t>;
    constexpr uint64_t MAX_KEY = std::numeric_limits<uint64_t>::max();
    std::vector<uint32_t> group(centers.size(), 0);
    uint32_t n_groups = 1;
    for (int d = 0; d < N_DIM; d++) {
        const int f = shape[d];
        if (f == 1) continue;
        std::vector<std::vector<Key>> keys(n_groups);
        for (size_t c = 0; c < centers.size(); c++) {
            keys[group[c]].push_back({ordered_bits(centers[c][d]), mesh.first_cell() + c});
        }
        std::vector<uint64_t> size(n_groups);
        for (uint32_t g = 0; g < n_groups; g++) {
            std::sort(keys[g].begin(), keys[g].end());
            size[g] = keys[g].size();
        }
        comm::allreduce(std::span<uint64_t>(size), comm::Op::SUM);
        auto count_le = [&](uint32_t g, const Key & k) {
            return uint64_t(std::upper_bound(keys[g].begin(), keys[g].end(), k) - keys[g].begin());
        };

        // Split s of group g is the key with exactly target[s] keys of the group below it
        const size_t n_splits = size_t(n_groups) * (f - 1);
        std::vector<uint64_t> target(n_splits);
        for (uint32_t g = 0; g < n_groups; g++) {
            for (int s = 1; s < f; s++) target[g * (f - 1) + s - 1] = size[g] * s / f;
        }
        std::vector<Key> split(n_splits);
        std::vector<uint64_t> a(n_splits), b(n_splits), counts(n_splits);
        // The smallest coordinate, then the smallest id at it, with more than target keys at or below
        for (int phase = 0; phase < 2; phase++) {
            for (size_t i = 0; i < n_splits; i++) {
                a[i] = 0;
                b[i] = phase == 0 ? MAX_KEY : mesh.n_global_cells() - 1;
            }
            while (true) {
                bool done = true;
                for (size_t i = 0; i < n_splits; i++) {
                    const uint64_t mid = a[i] + (b[i] - a[i]) / 2;
                    const Key k = phase == 0 ? Key{mid, MAX_KEY} : Key{split[i].first, mid};
                    counts[i] = count_le(i / (f - 1), k);
                    done = done && a[i] == b[i];
                }
                if (done) break;
                comm::allreduce(std::span<uint64_t>(counts), comm::Op::SUM);
                for (size_t i = 0; i < n_splits; i++) {
                    if (a[i] == b[i]) continue;
                    const uint64_t mid = a[i] + (b[i] - a[i]) / 2;
                    if (counts[i] > target[i]) {
                        b[i] = mid;
                    } else {
                        a[i] = mid + 1;
                    }
                }
            }
            for (size_t i = 0; i < n_splits; i++) (phase == 0 ? split[i].first : split[i].second) = a[i];
        }

        for (size_t c = 0; c < centers.size(); c++) {
            const Key k{ordered_bits(centers[c][d]), mesh.first_cell() + c};
            const auto first = split.begin() + size_t(group[c]) * (f - 1);
            group[c] = group[c] * f + uint32_t(std::upper_bound(first, first + (f - 1), k) - first);
        }
        n_groups *= f;
    }

    // Blocks numbered along a Hilbert curve, so that consecutive ranks (one
    // node's) hold a compact group of blocks
    std::array<double, N_DIM> grid_lo{}, grid_hi;
    grid_hi.fill(*std::max_element(shape.begin(), shape.end()));
    std::vector<std::pair<uint64_t, int>> blocks(n_parts);
    for (int p = 0; p < n_parts; p++) {
        std::array<double, N_DIM> center;
        for (int d = N_DIM - 1, rest = p; d >= 0; d--) {
            center[d] = rest % shape[d] + 0.5;
            rest /= shape[d];
        }
        blocks[p] = {hilbert_key(center, grid_lo, grid_hi), p};
    }
    std::sort(blocks.begin(), blocks.end());
    std::vector<int> rank_of_block(n_parts);
    for (int r = 0; r < n_parts; r++) rank_of_block[blocks[r].second] = r;
    std::vector<int> owner(group.size());
    for (size_t c = 0; c < group.size(); c++) owner[c] = rank_of_block[group[c]];
    return owner;
}

std::vector<int> partition_graph(const DistributedMesh & mesh, int n_parts) {
#ifdef Mallard_HAS_KAMINPAR
    using kaminpar::dist::GlobalEdgeID;
    using kaminpar::dist::GlobalNodeID;
    const auto & d = mesh.cell_distribution();
    std::vector<GlobalNodeID> vtxdist(d.begin(), d.end());
    std::vector<GlobalEdgeID> xadj(mesh.graph_offsets().begin(), mesh.graph_offsets().end());
    std::vector<GlobalNodeID> adjncy(mesh.graph_neighbors().begin(), mesh.graph_neighbors().end());
    // The same partition on every call, so per-rank caches (TENO stencils) can be reused
    kaminpar::dKaMinPar::reseed(0);
    kaminpar::dKaMinPar partitioner(comm::world(), 1, kaminpar::dist::create_default_context());
    partitioner.set_output_level(kaminpar::OutputLevel::QUIET);
    partitioner.copy_graph(vtxdist, xadj, adjncy);
    std::vector<kaminpar::dist::BlockID> blocks(mesh.n_block_cells());
    partitioner.compute_partition(static_cast<kaminpar::dist::BlockID>(n_parts), blocks);
    return std::vector<int>(blocks.begin(), blocks.end());
#else
    (void)mesh;
    (void)n_parts;
    throw std::runtime_error("This build has no graph partitioner (configure with Mallard_ENABLE_KAMINPAR=ON).");
#endif
}

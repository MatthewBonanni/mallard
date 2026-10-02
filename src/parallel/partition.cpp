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


bool have_graph_partitioner() {
#ifdef Mallard_HAS_KAMINPAR
    return true;
#else
    return false;
#endif
}


std::vector<int> partition_hilbert(const DistributedMesh & mesh, int n_parts) {
    const int p = comm::size();
    const auto centers = mesh.block_cell_centers();
    std::array<double, N_DIM> lo, hi;
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
    using Item = std::pair<uint64_t, uint64_t>;  // (key, global id): unique, so splitters are exact
    std::vector<Item> items(centers.size());
    for (size_t c = 0; c < centers.size(); c++) items[c] = {hilbert_key(centers[c], lo, hi), mesh.first_cell() + c};
    std::sort(items.begin(), items.end());

    // Sample sort: evenly spaced samples from every rank pick p - 1 splitters.
    // Parts come from global positions, so splitters only balance the sort
    // itself; a bounded oversampling keeps the gathered samples O(p).
    constexpr int MAX_SAMPLES = 64;
    const int n_samples = std::min(p, MAX_SAMPLES);
    std::vector<uint64_t> samples;
    for (int k = 0; k < n_samples && !items.empty(); k++) {
        const Item & s = items[(items.size() * k) / n_samples];
        samples.push_back(s.first);
        samples.push_back(s.second);
    }
    const std::vector<uint64_t> all = comm::allgatherv(samples);
    std::vector<Item> sorted_samples;
    for (size_t i = 0; i < all.size(); i += 2) sorted_samples.push_back({all[i], all[i + 1]});
    std::sort(sorted_samples.begin(), sorted_samples.end());
    std::vector<Item> splitters;
    for (int k = 1; k < p && !sorted_samples.empty(); k++) {
        splitters.push_back(sorted_samples[(sorted_samples.size() * k) / p]);
    }
    std::vector<std::vector<uint64_t>> send(p);
    for (const Item & item : items) {
        const int dest = std::upper_bound(splitters.begin(), splitters.end(), item) - splitters.begin();
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
    for (int r = 0; r < comm::rank(); r++) first += counts[r];
    const uint64_t n = mesh.n_global_cells();
    const auto & dist = mesh.cell_distribution();
    send.assign(p, {});
    for (size_t k = 0; k < items.size(); k++) {
        const uint64_t g = items[k].second;
        const int r = std::upper_bound(dist.begin(), dist.end(), g) - dist.begin() - 1;
        send[r].push_back(g);
        send[r].push_back(((first + k) * n_parts) / n);
    }
    std::vector<int> owner(mesh.n_block_cells());
    for (const auto & from : comm::alltoallv(send)) {
        for (size_t i = 0; i < from.size(); i += 2) owner[from[i] - mesh.first_cell()] = from[i + 1];
    }
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
    kaminpar::dKaMinPar partitioner(comm::world(), 1, kaminpar::dist::create_default_context());
    partitioner.set_output_level(kaminpar::OutputLevel::QUIET);
    partitioner.copy_graph(vtxdist, xadj, adjncy);
    std::vector<kaminpar::dist::BlockID> blocks(mesh.n_block_cells());
    partitioner.compute_partition(n_parts, blocks);
    return std::vector<int>(blocks.begin(), blocks.end());
#else
    (void)mesh;
    (void)n_parts;
    throw std::runtime_error("This build has no graph partitioner (configure with Mallard_ENABLE_KAMINPAR=ON).");
#endif
}

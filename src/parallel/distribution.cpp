/**
 * @file distribution.cpp
 * @author Matthew Bonanni (mbonanni001@gmail.com)
 * @brief A rank's share of a distributed mesh.
 * @version 0.3
 * @date 2026-10-02
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 *
 */

#include "distribution.h"

#include <stdexcept>
#include <string>
#include <unordered_map>

#include "comm.h"

void plan_halo_exchange(Distribution & dist, const std::vector<int> & halo_owner) {
    const int me = comm::rank();
    const int n_ranks = comm::size();
    std::unordered_map<uint64_t, uint32_t> local_of;
    for (uint32_t i = 0; i < dist.global_cell.size(); i++) local_of.emplace(dist.global_cell[i], i);
    std::vector<std::vector<uint64_t>> wanted(n_ranks);
    for (size_t i = dist.n_owned; i < dist.global_cell.size(); i++) {
        wanted[halo_owner[i - dist.n_owned]].push_back(dist.global_cell[i]);
    }
    const auto requested = comm::alltoallv(wanted);
    dist.neighbors.clear();
    dist.send_cells.clear();
    dist.recv_cells.clear();
    for (int r = 0; r < n_ranks; r++) {
        if (wanted[r].empty() && requested[r].empty()) continue;
        if (r == me) throw std::logic_error("plan_halo_exchange: a rank requested its own cells");
        dist.neighbors.push_back(r);
        std::vector<uint32_t> recv, send;
        for (uint64_t g : wanted[r]) recv.push_back(local_of.at(g));
        for (uint64_t g : requested[r]) {
            const auto it = local_of.find(g);
            if (it == local_of.end() || it->second >= dist.n_owned) {
                throw std::logic_error("plan_halo_exchange: rank " + std::to_string(r) +
                                       " requested a cell not owned here");
            }
            send.push_back(it->second);
        }
        dist.recv_cells.push_back(std::move(recv));
        dist.send_cells.push_back(std::move(send));
    }
}

void trim_halo_exchange(Distribution & dist, const std::vector<uint8_t> & needed) {
    // Both sides list a pair's cells in the same order, so a mask of the receiver's
    // list selects the same cells from the sender's
    std::vector<std::vector<uint64_t>> masks(comm::size());
    for (size_t n = 0; n < dist.neighbors.size(); n++) {
        for (const uint32_t c : dist.recv_cells[n]) masks[dist.neighbors[n]].push_back(needed[c]);
    }
    const auto keep = comm::alltoallv(masks);
    for (size_t n = 0; n < dist.neighbors.size(); n++) {
        const std::vector<uint64_t> & mask = keep[dist.neighbors[n]];
        if (mask.size() != dist.send_cells[n].size()) throw std::logic_error("trim_halo_exchange: lists differ");
        std::vector<uint32_t> send, recv;
        for (size_t k = 0; k < mask.size(); k++) {
            if (mask[k]) send.push_back(dist.send_cells[n][k]);
        }
        for (const uint32_t c : dist.recv_cells[n]) {
            if (needed[c]) recv.push_back(c);
        }
        dist.send_cells[n] = std::move(send);
        dist.recv_cells[n] = std::move(recv);
    }
}

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

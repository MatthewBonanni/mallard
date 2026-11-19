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

#include <algorithm>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "comm.h"

void plan_halo_exchange(Distribution & dist, const std::vector<int> & halo_owner) {
    const size_t me = static_cast<size_t>(comm::rank());
    const size_t n_ranks = static_cast<size_t>(comm::size());
    // Owned cells by global id
    std::vector<std::pair<uint64_t, uint32_t>> owned(dist.n_owned);
    for (uint32_t i = 0; i < dist.n_owned; i++) owned[i] = {dist.global_cell[i], i};
    std::sort(owned.begin(), owned.end());
    std::vector<std::vector<uint64_t>> wanted(n_ranks);
    std::vector<std::vector<uint32_t>> wanted_local(n_ranks);
    for (size_t i = dist.n_owned; i < dist.global_cell.size(); i++) {
        const size_t r = static_cast<size_t>(halo_owner[i - dist.n_owned]);
        wanted[r].push_back(dist.global_cell[i]);
        wanted_local[r].push_back(static_cast<uint32_t>(i));
    }
    const auto requested = comm::alltoallv(wanted);
    dist.neighbors.clear();
    dist.send_cells.clear();
    dist.recv_cells.clear();
    for (size_t r = 0; r < n_ranks; r++) {
        if (wanted[r].empty() && requested[r].empty()) continue;
        if (r == me) throw std::logic_error("plan_halo_exchange: a rank requested its own cells");
        dist.neighbors.push_back(static_cast<int>(r));
        std::vector<uint32_t> send;
        for (uint64_t g : requested[r]) {
            const auto it = std::lower_bound(owned.begin(), owned.end(), std::pair<uint64_t, uint32_t>{g, 0});
            if (it == owned.end() || it->first != g) {
                throw std::logic_error("plan_halo_exchange: rank " + std::to_string(r) +
                                       " requested a cell not owned here");
            }
            send.push_back(it->second);
        }
        dist.recv_cells.push_back(std::move(wanted_local[r]));
        dist.send_cells.push_back(std::move(send));
    }
}

void trim_halo_exchange(Distribution & dist, const std::vector<uint8_t> & needed) {
    // Both sides list a pair's cells in the same order, so a mask of the receiver's
    // list selects the same cells from the sender's
    std::vector<std::vector<uint64_t>> masks(static_cast<size_t>(comm::size()));
    for (size_t n = 0; n < dist.neighbors.size(); n++) {
        for (const uint32_t c : dist.recv_cells[n]) masks[static_cast<size_t>(dist.neighbors[n])].push_back(needed[c]);
    }
    const auto keep = comm::alltoallv(masks);
    for (size_t n = 0; n < dist.neighbors.size(); n++) {
        const std::vector<uint64_t> & mask = keep[static_cast<size_t>(dist.neighbors[n])];
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

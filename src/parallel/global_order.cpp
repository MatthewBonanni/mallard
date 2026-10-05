/**
 * @file global_order.cpp
 * @author Matthew Bonanni (mbonanni001@gmail.com)
 * @brief Moving per-item values to contiguous blocks in global-id order.
 * @version 0.5
 * @date 2026-10-04
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 *
 */

#include "global_order.h"

#include <algorithm>
#include <stdexcept>

#include "comm.h"
#include "mesh_block.h"

namespace {

/** @brief Rank whose block of n ids split p ways holds id g. */
int block_of(uint64_t g, uint64_t n, int p) {
    int r = static_cast<int>(g * uint64_t(p) / n);
    while (r > 0 && block_begin(n, r, p) > g) r--;
    while (r + 1 < p && block_begin(n, r + 1, p) <= g) r++;
    return r;
}

template <typename T>
std::vector<T> exchange(std::vector<std::vector<T>> && send, bool distributed) {
    if (!distributed) return std::move(send[0]);
    return comm::exchange(std::move(send)).data;
}

} // namespace

GlobalOrder::GlobalOrder(std::span<const uint64_t> ids, uint64_t n_global, bool distributed_in)
    : distributed(distributed_in), n_global_(n_global) {
    const int p = distributed ? comm::size() : 1, r = distributed ? comm::rank() : 0;
    first_ = block_begin(n_global, r, p);
    n_block = block_begin(n_global, r + 1, p) - first_;
    send_items.assign(p, {});
    std::vector<std::vector<uint64_t>> send(p);
    for (uint32_t i = 0; i < ids.size(); i++) {
        if (ids[i] >= n_global) throw std::logic_error("GlobalOrder: id out of range.");
        const int q = block_of(ids[i], n_global, p);
        send_items[q].push_back(i);
        send[q].push_back(ids[i]);
    }
    slot = exchange(std::move(send), distributed);
    for (uint64_t & s : slot) s -= first_;
}

template <typename T>
std::vector<T> GlobalOrder::gather(std::span<const T> values, uint32_t width) const {
    std::vector<std::vector<T>> send(send_items.size());
    for (size_t q = 0; q < send_items.size(); q++) {
        send[q].reserve(send_items[q].size() * width);
        for (uint32_t i : send_items[q]) {
            const T * item = values.data() + size_t(i) * width;
            send[q].insert(send[q].end(), item, item + width);
        }
    }
    const std::vector<T> received = exchange(std::move(send), distributed);
    std::vector<T> block(n_block * width, T(0));
    for (size_t k = 0; k < slot.size(); k++) {
        std::copy_n(received.data() + k * width, width, block.data() + slot[k] * width);
    }
    return block;
}

std::pair<std::vector<uint64_t>, std::vector<uint64_t>> GlobalOrder::gather_csr(
    std::span<const uint64_t> offsets, std::span<const uint64_t> values) const {
    // Each item as its length, then its values
    std::vector<std::vector<uint64_t>> send(send_items.size());
    for (size_t q = 0; q < send_items.size(); q++) {
        for (uint32_t i : send_items[q]) {
            send[q].push_back(offsets[i + 1] - offsets[i]);
            send[q].insert(send[q].end(), values.data() + offsets[i], values.data() + offsets[i + 1]);
        }
    }
    const std::vector<uint64_t> received = exchange(std::move(send), distributed);
    std::vector<uint64_t> length(n_block, 0), start(slot.size());
    for (size_t k = 0, at = 0; k < slot.size(); k++) {
        start[k] = at + 1;
        length[slot[k]] = received[at];
        at += 1 + received[at];
    }
    std::vector<uint64_t> block_offsets(n_block + 1, 0);
    for (uint64_t i = 0; i < n_block; i++) block_offsets[i + 1] = block_offsets[i] + length[i];
    std::vector<uint64_t> block_values(block_offsets.back());
    for (size_t k = 0; k < slot.size(); k++) {
        std::copy_n(received.data() + start[k], length[slot[k]], block_values.data() + block_offsets[slot[k]]);
    }
    return {std::move(block_offsets), std::move(block_values)};
}

template std::vector<double> GlobalOrder::gather(std::span<const double>, uint32_t) const;
template std::vector<float> GlobalOrder::gather(std::span<const float>, uint32_t) const;
template std::vector<uint64_t> GlobalOrder::gather(std::span<const uint64_t>, uint32_t) const;

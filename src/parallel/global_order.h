/**
 * @file global_order.h
 * @author Matthew Bonanni (mbonanni001@gmail.com)
 * @brief Moving per-item values from their owner ranks to contiguous blocks in
 *        global-id order, for writes at global offsets.
 * @version 0.5
 * @date 2026-10-04
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 *
 */

#ifndef GLOBAL_ORDER_H
#define GLOBAL_ORDER_H

#include <cstdint>
#include <span>
#include <utility>
#include <vector>

/**
 * @brief Items (cells, nodes) spread over the ranks, each with a global id in
 *        [0, n_global), regrouped so that rank r holds the ids
 *        [block_begin(n_global, r, p), block_begin(n_global, r + 1, p)) in
 *        order. An id may come from several ranks (shared nodes); it then
 *        takes any one of their values, so they should agree. Ids no rank
 *        sends are left zero. Without distributed, this rank holds every id
 *        and the regrouping is a local permutation (not collective).
 */
class GlobalOrder {
    public:
        GlobalOrder() = default;

        /** @brief Plan for the items with these global ids (collective if distributed). */
        GlobalOrder(std::span<const uint64_t> ids, uint64_t n_global, bool distributed);

        uint64_t n_global() const { return n_global_; }
        uint64_t first() const { return first_; }
        uint64_t size() const { return n_block; }

        /**
         * @brief This rank's block, width values per id, from the values
         *        (width per item) of the plan's items (collective if distributed).
         */
        template <typename T>
        std::vector<T> gather(std::span<const T> values, uint32_t width) const;

        /**
         * @brief Like gather, for items of different lengths (CSR offsets into
         *        values); returns the block's offsets (from 0) and values.
         */
        std::pair<std::vector<uint64_t>, std::vector<uint64_t>> gather_csr(std::span<const uint64_t> offsets,
                                                                           std::span<const uint64_t> values) const;

    private:
        bool distributed = false;
        uint64_t n_global_ = 0, first_ = 0, n_block = 0;
        std::vector<std::vector<uint32_t>> send_items;  // per rank: the items it gets, in order
        std::vector<uint64_t> slot;                     // per received item: its row in the block
};

#endif // GLOBAL_ORDER_H

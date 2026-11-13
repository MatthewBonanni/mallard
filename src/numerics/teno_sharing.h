/**
 * @file teno_sharing.h
 * @author Matthew Bonanni (mbonanni001@gmail.com)
 * @brief Sharing of equal pseudo-inverse tables between cells (host side).
 * @version 0.1
 * @date 2026-11-13
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 *
 */

#ifndef TENO_SHARING_H
#define TENO_SHARING_H

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <numeric>
#include <unordered_map>
#include <vector>

#include "common.h"

namespace teno {

/**
 * @brief Tables whose largest entries differ by at most this fraction of the
 *        largest are equal: round-off apart (about 1e-14 on lattices).
 */
constexpr double SHARE_TOLERANCE = 1.0e-10;

/**
 * @brief A stencil's pseudo-inverse (slot s, entry l at pinv[s width + l])
 *        in an order of its slots that does not depend on the cell
 *        numbering: within each segment (the whole stencil, or each face's
 *        sector stencil), slots sorted by their entries on a grid of 2^-25
 *        times the table's magnitude. Equal tables up to round-off then have
 *        the same order and the same hash, unless an entry straddles a grid
 *        line (they are then not shared, which costs memory only).
 */
struct CanonicalTable {
    std::vector<uint16_t> order;  // (canonical slot): the stencil's slot
    uint64_t hash = 0;
    double magnitude = 0.0;       // largest |entry|

    CanonicalTable() = default;

    CanonicalTable(const rtype * pinv, const uint32_t width, const uint16_t * segments, const uint32_t n_segments) {
        uint32_t n = 0;
        for (uint32_t k = 0; k < n_segments; k++) n += segments[k];
        for (size_t i = 0; i < size_t(n) * width; i++) magnitude = std::max(magnitude, std::abs(double(pinv[i])));
        int exponent = 0;
        std::frexp(magnitude, &exponent);
        const double grid = std::ldexp(1.0, exponent - 25);
        std::vector<int64_t> q(size_t(n) * width);
        for (size_t i = 0; i < q.size(); i++) q[i] = std::llround(double(pinv[i]) / grid);
        order.resize(n);
        std::iota(order.begin(), order.end(), uint16_t(0));
        add(exponent);
        uint32_t first = 0;
        for (uint32_t k = 0; k < n_segments; k++) {
            std::stable_sort(order.begin() + first, order.begin() + first + segments[k], [&](uint16_t a, uint16_t b) {
                return std::lexicographical_compare(q.begin() + a * width, q.begin() + (a + 1) * width,
                                                    q.begin() + b * width, q.begin() + (b + 1) * width);
            });
            add(segments[k]);
            first += segments[k];
        }
        for (const uint16_t s : order) {
            for (uint32_t l = 0; l < width; l++) add(q[size_t(s) * width + l]);
        }
    }

    private:
        template <typename T>
        void add(const T value) {
            const unsigned char * p = reinterpret_cast<const unsigned char *>(&value);
            for (size_t i = 0; i < sizeof(T); i++) {
                hash ^= p[i];
                hash *= 1099511628211ULL;
            }
        }
};

/**
 * @brief Distinct blocks of pseudo-inverse entries, each found again by the
 *        hashes of its lanes' canonical tables and checked entry by entry
 *        within SHARE_TOLERANCE of each lane's magnitude. A block is a slice
 *        of PackedStencils (one cell on the host): `slots` slots of `width`
 *        entries for each of its lanes, interleaved.
 */
class SharedBlocks {
    public:
        SharedBlocks(const uint8_t lane_shift, const uint32_t values_per_slot)
            : shift(lane_shift), width(values_per_slot) {}

        /** @brief The key of a block from its lanes' tables (lane_size: slots of each lane). */
        uint64_t key(const uint64_t slots, const uint16_t * lane_size, const CanonicalTable * lanes) const {
            uint64_t h = 1469598103934665603ULL ^ slots;
            for (uint32_t k = 0; k < (1u << shift); k++) {
                h = (h ^ lane_size[k]) * 1099511628211ULL;
                h = (h ^ (lane_size[k] ? lanes[k].hash : 0)) * 1099511628211ULL;
            }
            return h;
        }

        /** @brief First slot of a stored block with this key, the same sizes and these entries, or -1. */
        int64_t find(const uint64_t k, const uint64_t slots, const uint16_t * lane_size, const double * lane_magnitude,
                     const rtype * values) const {
            const auto range = index.equal_range(k);
            for (auto it = range.first; it != range.second; ++it) {
                if (matches(it->second, slots, lane_size, lane_magnitude, values)) return int64_t(blocks[it->second].start);
            }
            return -1;
        }

        /** @brief Whether stored block b has these sizes and entries. */
        bool matches(const size_t b, const uint64_t slots, const uint16_t * lane_size, const double * lane_magnitude,
                     const rtype * values) const {
            const Block & block = blocks[b];
            if (block.slots != slots || !std::equal(lane_size, lane_size + (1u << shift), block.lane_size.begin())) {
                return false;
            }
            const rtype * stored = data.data() + ((block.start * width) << shift);
            for (uint32_t lane = 0; lane < (1u << shift); lane++) {
                const double tolerance = SHARE_TOLERANCE * lane_magnitude[lane];
                for (size_t i = 0; i < size_t(lane_size[lane]) * width; i++) {
                    const size_t at = (i << shift) + lane;
                    if (!(std::abs(double(stored[at]) - double(values[at])) <= tolerance)) return false;
                }
            }
            return true;
        }

        /** @brief Store a block; its first slot. */
        uint64_t add(const uint64_t k, const uint64_t slots, const uint16_t * lane_size, const rtype * values) {
            const uint64_t at = n_slots;
            data.insert(data.end(), values, values + ((slots * width) << shift));
            n_slots += slots;
            index.emplace(k, blocks.size());
            blocks.push_back(Block{at, slots, std::vector<uint16_t>(lane_size, lane_size + (1u << shift))});
            return at;
        }

        /** @brief Index of the stored block with this key that a lookup tries first, or -1. */
        int64_t first(const uint64_t k) const {
            const auto it = index.find(k);
            return it == index.end() ? -1 : int64_t(it->second);
        }

        uint64_t start(const size_t b) const { return blocks[b].start; }

        /** @brief Slots stored, and their entries, interleaved by lane. */
        uint64_t slots() const { return n_slots; }
        size_t n_blocks() const { return blocks.size(); }
        std::vector<rtype> & values() { return data; }

    private:
        struct Block {
            uint64_t start;  // first slot
            uint64_t slots;
            std::vector<uint16_t> lane_size;
        };

        uint8_t shift;
        uint32_t width;
        uint64_t n_slots = 0;
        std::vector<Block> blocks;
        std::vector<rtype> data;
        std::unordered_multimap<uint64_t, size_t> index;
};

} // namespace teno

#endif // TENO_SHARING_H

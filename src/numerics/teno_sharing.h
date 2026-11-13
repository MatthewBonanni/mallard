/**
 * @file teno_sharing.h
 * @author Matthew Bonanni (mbonanni001@gmail.com)
 * @brief The store of TENO pseudo-inverse tables shared by cells of one
 *        stencil shape (host side, at setup).
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

#include <Kokkos_Core.hpp>

#include "comm.h"
#include "common.h"

namespace teno {

/**
 * @brief Tables are compared on a grid of 2^-SHARE_GRID_BITS times their
 *        magnitude (about 1e-9): the tables of one stencil shape differ by
 *        round-off (about 5e-14 on lattices), far below it.
 */
constexpr int SHARE_GRID_BITS = 30;

/**
 * @brief A stencil's pseudo-inverse (slot s, entry l at pinv[s width + l])
 *        on the grid, in an order of its slots that does not depend on the
 *        cell numbering: within each segment (the whole stencil, or each
 *        face's sector stencil), slots sorted by their entries on the grid.
 *        Two tables have the same shape if they are equal on the grid in that
 *        order. A function of the table alone, so of the cell, whatever the
 *        partition; tables of one shape with an entry on either side of a
 *        grid line are two shapes, which costs memory only.
 */
struct CanonicalTable {
    std::vector<uint16_t> order;     // (canonical slot): the stencil's slot
    std::vector<uint16_t> segments;  // slots of each segment
    int exponent = 0;                // of the table's largest |entry|, as frexp gives it
    uint64_t hash = 1469598103934665603ULL;

    CanonicalTable() = default;

    CanonicalTable(const rtype * pinv, const uint32_t width, std::vector<uint16_t> segment_sizes)
        : segments(std::move(segment_sizes)) {
        const uint32_t n = std::accumulate(segments.begin(), segments.end(), 0u);
        double magnitude = 0.0;
        for (size_t i = 0; i < size_t(n) * width; i++) magnitude = std::max(magnitude, std::abs(double(pinv[i])));
        std::frexp(magnitude, &exponent);
        std::vector<int64_t> q(size_t(n) * width);
        for (size_t i = 0; i < q.size(); i++) q[i] = on_grid(pinv[i], exponent);
        order.resize(n);
        std::iota(order.begin(), order.end(), uint16_t(0));
        add(exponent);
        uint32_t first = 0;
        for (const uint16_t size : segments) {
            std::stable_sort(order.begin() + first, order.begin() + first + size, [&](uint16_t a, uint16_t b) {
                return std::lexicographical_compare(q.begin() + a * width, q.begin() + (a + 1) * width,
                                                    q.begin() + b * width, q.begin() + (b + 1) * width);
            });
            add(size);
            first += size;
        }
        for (const uint16_t s : order) {
            for (uint32_t l = 0; l < width; l++) add(q[size_t(s) * width + l]);
        }
    }

    /** @brief An entry on the grid of a table of this exponent. */
    static int64_t on_grid(const rtype v, const int exp) { return std::llround(std::ldexp(double(v), SHARE_GRID_BITS - exp)); }

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
 * @brief One table per stencil shape: the cells of a shape share the table
 *        of its cell with the lowest global id, in canonical order, agreed
 *        over the ranks, so that a cell's table does not depend on the
 *        partition, the batches or the thread count. Cells are added in
 *        order; each gets its shape's index.
 */
class TableStore {
    public:
        explicit TableStore(const uint32_t values_per_slot) : width(values_per_slot) {}

        /**
         * @brief The shapes of n cells: tables[c] in canonical order (values
         *        ordered(c), canonical[c].order.size() slots), global ids gid[c].
         */
        template <typename Values>
        std::vector<uint32_t> add(const uint32_t n, const CanonicalTable * canonical, Values ordered, const uint64_t * gid) {
            std::vector<uint32_t> shape(n, 0);
            std::vector<int64_t> tried(n, -1);
            std::vector<uint8_t> found(n, 0);
            // Each cell tries the first shape of its hash, checked in parallel; the rest are searched in order
            for (uint32_t c = 0; c < n; c++) {
                const auto it = index.find(canonical[c].hash);
                if (it != index.end()) {
                    tried[c] = int64_t(it->second);
                    continue;
                }
                shape[c] = create(canonical[c], ordered(c), gid[c]);
                found[c] = 1;
            }
            Kokkos::parallel_for("teno_share", Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(0, n),
                                 [&](const uint32_t c) {
                if (!found[c] && same_shape(size_t(tried[c]), canonical[c], ordered(c))) {
                    shape[c] = uint32_t(tried[c]);
                    found[c] = 1;
                }
            });
            for (uint32_t c = 0; c < n; c++) {
                if (!found[c]) {
                    shape[c] = find(canonical[c], ordered(c));
                    if (shape[c] == NONE) {
                        shape[c] = create(canonical[c], ordered(c), gid[c]);
                        continue;
                    }
                }
                Shape & s = shapes[shape[c]];
                if (gid[c] < s.gid) {
                    s.gid = gid[c];
                    s.values.assign(ordered(c), ordered(c) + s.values.size());
                }
            }
            return shape;
        }

        /**
         * @brief The shape a caller keys by its own stencil-shape key (e.g. a
         *        structured block's offset pattern) instead of by equality of
         *        the tables, for a cell with table v (canonical order of t)
         *        and global id gid: created at its first cell, it takes the
         *        table of the lowest global id, and agree() matches it over
         *        the ranks by its key.
         */
        uint32_t add_keyed(const uint64_t key, const CanonicalTable & t, const rtype * v, const uint64_t gid) {
            uint32_t k = find_keyed(key);
            if (k == NONE) return create(t, v, gid, true, keyed_hash(key));
            Shape & s = shapes[k];
            if (gid < s.gid) {
                s.gid = gid;
                s.values.assign(v, v + s.values.size());
            }
            return k;
        }

        /** @brief The shape registered under a caller's key, or NONE. */
        uint32_t find_keyed(const uint64_t key) const {
            const auto it = keys.find(keyed_hash(key));
            return it == keys.end() ? NONE : it->second;
        }

        /**
         * @brief Every shape takes the table of its cell with the lowest
         *        global id over the ranks (collective).
         */
        void agree() {
            const int p = comm::size();
            if (p == 1) return;
            // The rank owning a hash learns every rank's lowest global id of it
            std::vector<std::vector<uint64_t>> report(static_cast<size_t>(p));
            for (const Shape & s : shapes) {
                auto & r = report[s.hash % uint64_t(p)];
                r.push_back(s.hash);
                r.push_back(s.gid);
            }
            const auto reports = comm::alltoallv(report);
            std::unordered_map<uint64_t, std::pair<uint64_t, uint64_t>> lowest;  // hash: (gid, rank)
            for (int r = 0; r < p; r++) {
                const auto & from = reports[size_t(r)];
                for (size_t k = 0; k < from.size(); k += 2) {
                    auto [it, added] = lowest.try_emplace(from[k], from[k + 1], uint64_t(r));
                    if (!added && from[k + 1] < it->second.first) it->second = {from[k + 1], uint64_t(r)};
                }
            }
            std::vector<std::vector<uint64_t>> answer(static_cast<size_t>(p));
            for (int r = 0; r < p; r++) {
                const auto & from = reports[size_t(r)];
                for (size_t k = 0; k < from.size(); k += 2) {
                    const auto & best = lowest.at(from[k]);
                    answer[size_t(r)].insert(answer[size_t(r)].end(), {from[k], best.first, best.second});
                }
            }
            const auto answers = comm::alltoallv(answer);
            // Ask the rank of each lowest global id that is not this rank's for its table
            std::unordered_map<uint64_t, std::pair<uint64_t, uint64_t>> best;  // hash: (gid, rank)
            for (const auto & from : answers) {
                for (size_t k = 0; k < from.size(); k += 3) best[from[k]] = {from[k + 1], from[k + 2]};
            }
            std::vector<std::vector<uint64_t>> request(static_cast<size_t>(p));
            std::vector<std::vector<size_t>> asked(static_cast<size_t>(p));
            for (size_t i = 0; i < shapes.size(); i++) {
                const auto & [g, r] = best.at(shapes[i].hash);
                if (g >= shapes[i].gid) continue;
                request[r].insert(request[r].end(), {shapes[i].hash, g});
                asked[r].push_back(i);
            }
            const auto requests = comm::alltoallv(request);
            std::unordered_multimap<uint64_t, size_t> by_hash;
            for (size_t i = 0; i < shapes.size(); i++) by_hash.emplace(shapes[i].hash, i);
            std::vector<std::vector<rtype>> reply(static_cast<size_t>(p));
            for (int r = 0; r < p; r++) {
                const auto & from = requests[size_t(r)];
                for (size_t k = 0; k < from.size(); k += 2) {
                    const auto range = by_hash.equal_range(from[k]);
                    for (auto it = range.first; it != range.second; ++it) {
                        if (shapes[it->second].gid != from[k + 1]) continue;
                        const auto & v = shapes[it->second].values;
                        reply[size_t(r)].insert(reply[size_t(r)].end(), v.begin(), v.end());
                        break;
                    }
                }
            }
            const auto replies = comm::alltoallv(reply);
            for (int r = 0; r < p; r++) {
                size_t at = 0;
                for (const size_t i : asked[size_t(r)]) {
                    Shape & s = shapes[i];
                    const rtype * v = replies[size_t(r)].data() + at;
                    at += s.values.size();
                    // A hash shared by two shapes keeps this rank's table
                    if (!(s.keyed || same_shape(i, s.canonical, v))) continue;
                    s.values.assign(v, v + s.values.size());
                    s.gid = best.at(s.hash).first;
                }
            }
        }

        /** @brief Slots of shape k's table, and its entries in canonical order. */
        uint16_t slots(const size_t k) const { return static_cast<uint16_t>(shapes[k].canonical.order.size()); }
        const std::vector<rtype> & values(const size_t k) const { return shapes[k].values; }
        /** @brief Global id of the cell whose table shape k takes. */
        uint64_t owner(const size_t k) const { return shapes[k].gid; }
        size_t size() const { return shapes.size(); }

        static constexpr uint32_t NONE = ~0u;

    private:
        struct Shape {
            uint64_t hash;
            uint64_t gid;
            CanonicalTable canonical;  // order unused; segments and exponent compared
            std::vector<rtype> values;
            bool keyed;                // by a caller's key, not by equality
        };

        static uint64_t keyed_hash(const uint64_t key) {
            uint64_t h = key + 0x9e3779b97f4a7c15ULL;
            h = (h ^ (h >> 30)) * 0xbf58476d1ce4e5b9ULL;
            h = (h ^ (h >> 27)) * 0x94d049bb133111ebULL;
            return h ^ (h >> 31);
        }

        uint32_t create(const CanonicalTable & t, const rtype * v, const uint64_t gid, const bool keyed = false,
                        const uint64_t hash = 0) {
            const size_t n = t.order.size() * width;
            CanonicalTable kept;
            kept.order.resize(t.order.size());
            kept.segments = t.segments;
            kept.exponent = t.exponent;
            kept.hash = t.hash;
            const uint64_t h = keyed ? hash : t.hash;
            kept.hash = h;
            shapes.push_back(Shape{h, gid, std::move(kept), std::vector<rtype>(v, v + n), keyed});
            const uint32_t k = uint32_t(shapes.size() - 1);
            if (keyed) {
                keys.emplace(h, k);
            } else {
                index.emplace(h, k);
            }
            return k;
        }

        uint32_t find(const CanonicalTable & t, const rtype * v) const {
            const auto range = index.equal_range(t.hash);
            for (auto it = range.first; it != range.second; ++it) {
                if (same_shape(it->second, t, v)) return uint32_t(it->second);
            }
            return NONE;
        }

        bool same_shape(const size_t k, const CanonicalTable & t, const rtype * v) const {
            const Shape & s = shapes[k];
            if (s.canonical.exponent != t.exponent || s.canonical.segments != t.segments) return false;
            for (size_t i = 0; i < s.values.size(); i++) {
                if (CanonicalTable::on_grid(s.values[i], t.exponent) != CanonicalTable::on_grid(v[i], t.exponent)) {
                    return false;
                }
            }
            return true;
        }

        uint32_t width;
        std::vector<Shape> shapes;
        std::unordered_multimap<uint64_t, size_t> index;  // equality hash: shapes
        std::unordered_map<uint64_t, uint32_t> keys;      // hash of a caller's key: shape
};

} // namespace teno

#endif // TENO_SHARING_H

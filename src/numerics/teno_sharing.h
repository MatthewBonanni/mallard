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
#include <map>
#include <numeric>
#include <type_traits>
#include <unordered_map>
#include <vector>

#include <Kokkos_Core.hpp>

#include "comm.h"
#include "common.h"
#include "gated.h"

namespace teno {

/**
 * @brief Tables are compared on a grid of 2^-SHARE_GRID_BITS times their
 *        magnitude (about 1e-9): the tables of one stencil shape differ by
 *        round-off (about 5e-14 on lattices), far below it.
 */
constexpr int SHARE_GRID_BITS = 30;

/** @brief An entry on the grid of a table of exponent exp (frexp's of its largest |entry|). */
template <typename T>
KOKKOS_INLINE_FUNCTION int64_t on_grid(const T v, const int exp, const int bits = SHARE_GRID_BITS) {
    return static_cast<int64_t>(Kokkos::round(Kokkos::ldexp(double(v), bits - exp)));
}

/** @brief The FNV-1a hash h extended by the bytes of value, lowest first. */
template <typename T>
KOKKOS_INLINE_FUNCTION uint64_t fnv(uint64_t h, const T value) {
    using U = std::make_unsigned_t<T>;
    const U u = static_cast<U>(value);
    for (size_t i = 0; i < sizeof(T); i++) {
        h ^= static_cast<uint64_t>((u >> (8 * i)) & U(0xff));
        h *= 1099511628211ULL;
    }
    return h;
}

/**
 * @brief Canonical order and hash of a table (entry i at v(i); slot s has
 *        entries s width .. (s + 1) width - 1), whose slots form n_segments
 *        segments of segments[k] slots: order(s) is set to the slot at
 *        canonical position s, exponent to frexp's of the largest |entry|.
 */
template <typename Entry, typename Order>
KOKKOS_INLINE_FUNCTION uint64_t canonicalize(const Entry & v, const uint32_t width, const uint16_t * segments,
                                             const int n_segments, const Order & order, int & exponent,
                                             const int bits = SHARE_GRID_BITS) {
    uint32_t n = 0;
    for (int k = 0; k < n_segments; k++) n += segments[k];
    double magnitude = 0.0;
    for (uint32_t i = 0; i < n * width; i++) {
        const double a = Kokkos::fabs(double(v(i)));
        magnitude = magnitude < a ? a : magnitude;
    }
    Kokkos::frexp(magnitude, &exponent);
    const int e = exponent;
    // Stable: an insertion sort by the slots' entries on the grid, lexicographically
    auto less = [&](const uint16_t a, const uint16_t b) {
        for (uint32_t l = 0; l < width; l++) {
            const int64_t qa = on_grid(v(a * width + l), e, bits), qb = on_grid(v(b * width + l), e, bits);
            if (qa != qb) return qa < qb;
        }
        return false;
    };
    uint64_t hash = fnv(1469598103934665603ULL, exponent);
    uint32_t first = 0;
    for (int k = 0; k < n_segments; k++) {
        const uint32_t end = first + segments[k];
        for (uint32_t s = first; s < end; s++) {
            const uint16_t x = static_cast<uint16_t>(s);
            uint32_t b = s;
            while (b > first && less(x, order(b - 1))) {
                order(b) = order(b - 1);
                b--;
            }
            order(b) = x;
        }
        hash = fnv(hash, segments[k]);
        first = end;
    }
    for (uint32_t s = 0; s < n; s++) {
        for (uint32_t l = 0; l < width; l++) hash = fnv(hash, on_grid(v(order(s) * width + l), e, bits));
    }
    return hash;
}

/** @brief Entries of a table in host memory, for canonicalize (a host lambda's call operator is host only). */
template <typename T>
struct HostEntries {
    const T * p;
    KOKKOS_INLINE_FUNCTION T operator()(const uint32_t i) const { return p[i]; }
};

/** @brief The slot order of a table in host memory, for canonicalize. */
struct HostOrder {
    uint16_t * p;
    KOKKOS_INLINE_FUNCTION uint16_t & operator()(const uint32_t s) const { return p[s]; }
};

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
    int bits = SHARE_GRID_BITS;      // the grid: 2^-bits of that magnitude
    uint64_t hash = 1469598103934665603ULL;

    CanonicalTable() = default;

    template <typename T>
    CanonicalTable(const T * pinv, const uint32_t width, std::vector<uint16_t> segment_sizes,
                   const int grid_bits = SHARE_GRID_BITS)
        : order(std::accumulate(segment_sizes.begin(), segment_sizes.end(), size_t(0))), segments(std::move(segment_sizes)),
          bits(grid_bits) {
        hash = canonicalize(HostEntries<T>{pinv}, width, segments.data(), int(segments.size()), HostOrder{order.data()},
                            exponent, bits);
    }
};

/**
 * @brief One table per stencil shape: the cells of a shape share the table
 *        of its cell with the lowest global id, in canonical order, agreed
 *        over the ranks, so that a cell's table does not depend on the
 *        partition, the batches or the thread count. Cells are added in
 *        order; each gets its shape's index. T: the type of the values.
 */
template <typename T = rtype>
class TableStore {
    public:
        explicit TableStore(const uint32_t values_per_slot) : width(values_per_slot) {}

        /** @brief A cell to compare with a shape of its exponent and segments (held: see add()). */
        struct Match {
            uint32_t cell;
            uint32_t shape;
            int64_t held;
        };

        /**
         * @brief The shapes of n cells: tables[c] in canonical order (values
         *        ordered(c), canonical[c].order.size() slots), global ids gid[c].
         */
        template <typename Values>
        std::vector<uint32_t> add(const uint32_t n, const CanonicalTable * canonical, Values ordered, const uint64_t * gid) {
            std::vector<std::pair<uint32_t, uint32_t>> taken;
            const auto shape = add(n, canonical, gid, [&](const std::vector<Match> & m) {
                std::vector<uint8_t> same(m.size(), 0);
                gated::parallel_for("teno_share", Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(0, m.size()),
                                     [&](const size_t i) {
                    const T * a = ordered(m[i].cell);
                    const T * b = m[i].held >= 0 ? ordered(uint32_t(m[i].held)) : shapes[m[i].shape].values.data();
                    const int e = canonical[m[i].cell].exponent, bits = canonical[m[i].cell].bits;
                    uint8_t equal = 1;
                    for (size_t k = 0; k < shapes[m[i].shape].values.size() && equal; k++) {
                        equal = on_grid(a[k], e, bits) == on_grid(b[k], e, bits);
                    }
                    same[i] = equal;
                });
                return same;
            }, taken);
            for (const auto & [k, c] : taken) set_values(k, ordered(c));
            return shape;
        }

        /**
         * @brief add() for tables the caller holds and compares:
         *        same(matches) tells, for each match, whether the cell's table
         *        is equal on the grid to the shape's, which is that of cell
         *        held of these n if held >= 0, else values(shape). taken: the
         *        shapes, in increasing order, whose table becomes that of a
         *        cell, which the caller then gives them with set_values().
         */
        template <typename Same>
        std::vector<uint32_t> add(const uint32_t n, const CanonicalTable * canonical, const uint64_t * gid, Same same,
                                  std::vector<std::pair<uint32_t, uint32_t>> & taken) {
            std::vector<uint32_t> shape(n, 0);
            std::vector<int64_t> tried(n, -1);
            std::vector<uint8_t> found(n, 0);
            const size_t first_new = shapes.size();
            std::vector<uint32_t> holder;  // (shape - first_new): the cell whose table it has
            auto held = [&](const size_t k) { return k >= first_new ? int64_t(holder[k - first_new]) : int64_t(-1); };
            auto make = [&](const uint32_t c) {
                holder.push_back(c);
                return create(canonical[c], nullptr, gid[c]);
            };
            // Each cell tries the first shape of its hash, compared at once; the rest are searched in order
            for (uint32_t c = 0; c < n; c++) {
                const auto it = index.find(canonical[c].hash);
                if (it != index.end()) {
                    tried[c] = int64_t(it->second);
                    continue;
                }
                shape[c] = make(c);
                found[c] = 1;
            }
            std::vector<Match> matches;
            for (uint32_t c = 0; c < n; c++) {
                if (!found[c] && same_layout(size_t(tried[c]), canonical[c])) {
                    matches.push_back({c, uint32_t(tried[c]), held(size_t(tried[c]))});
                }
            }
            const std::vector<uint8_t> equal = same(matches);
            for (size_t i = 0; i < matches.size(); i++) {
                if (!equal[i]) continue;
                shape[matches[i].cell] = matches[i].shape;
                found[matches[i].cell] = 1;
            }
            std::map<uint32_t, uint32_t> source;  // shape: the cell whose table it takes
            for (size_t k = first_new; k < shapes.size(); k++) source[uint32_t(k)] = holder[k - first_new];
            for (uint32_t c = 0; c < n; c++) {
                if (!found[c]) {
                    shape[c] = NONE;
                    const auto range = index.equal_range(canonical[c].hash);
                    for (auto it = range.first; it != range.second && shape[c] == NONE; ++it) {
                        if (same_layout(it->second, canonical[c]) && same({Match{c, uint32_t(it->second), held(it->second)}})[0]) {
                            shape[c] = uint32_t(it->second);
                        }
                    }
                    if (shape[c] == NONE) {
                        shape[c] = make(c);
                        source[shape[c]] = c;
                        continue;
                    }
                }
                Shape & s = shapes[shape[c]];
                if (gid[c] < s.gid) {
                    s.gid = gid[c];
                    source[shape[c]] = c;
                }
            }
            taken.assign(source.begin(), source.end());
            return shape;
        }

        /** @brief Shape k's table, in canonical order. */
        void set_values(const size_t k, const T * v) { shapes[k].values.assign(v, v + shapes[k].values.size()); }

        /**
         * @brief The shape a caller keys by its own stencil-shape key (e.g. a
         *        structured block's offset pattern) instead of by equality of
         *        the tables, for a cell with table v (canonical order of t)
         *        and global id gid: created at its first cell, it takes the
         *        table of the lowest global id, and agree() matches it over
         *        the ranks by its key.
         */
        uint32_t add_keyed(const uint64_t key, const CanonicalTable & t, const T * v, const uint64_t gid) {
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
            std::vector<std::vector<T>> reply(static_cast<size_t>(p));
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
                    const T * v = replies[size_t(r)].data() + at;
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
        const std::vector<T> & values(const size_t k) const { return shapes[k].values; }
        /** @brief Global id of the cell whose table shape k takes. */
        uint64_t owner(const size_t k) const { return shapes[k].gid; }
        size_t size() const { return shapes.size(); }

        static constexpr uint32_t NONE = ~0u;

    private:
        struct Shape {
            uint64_t hash;
            uint64_t gid;
            CanonicalTable canonical;  // order unused; segments and exponent compared
            std::vector<T> values;
            bool keyed;                // by a caller's key, not by equality
        };

        static uint64_t keyed_hash(const uint64_t key) {
            uint64_t h = key + 0x9e3779b97f4a7c15ULL;
            h = (h ^ (h >> 30)) * 0xbf58476d1ce4e5b9ULL;
            h = (h ^ (h >> 27)) * 0x94d049bb133111ebULL;
            return h ^ (h >> 31);
        }

        /** @brief A new shape of t's layout, whose table is v (null: set later). */
        uint32_t create(const CanonicalTable & t, const T * v, const uint64_t gid, const bool keyed = false,
                        const uint64_t hash = 0) {
            const size_t n = t.order.size() * width;
            CanonicalTable kept;
            kept.order.resize(t.order.size());
            kept.segments = t.segments;
            kept.exponent = t.exponent;
            kept.bits = t.bits;
            const uint64_t h = keyed ? hash : t.hash;
            kept.hash = h;
            shapes.push_back(Shape{h, gid, std::move(kept), v != nullptr ? std::vector<T>(v, v + n) : std::vector<T>(n), keyed});
            const uint32_t k = uint32_t(shapes.size() - 1);
            if (keyed) {
                keys.emplace(h, k);
            } else {
                index.emplace(h, k);
            }
            return k;
        }

        bool same_layout(const size_t k, const CanonicalTable & t) const {
            return shapes[k].canonical.exponent == t.exponent && shapes[k].canonical.bits == t.bits &&
                   shapes[k].canonical.segments == t.segments;
        }

        bool same_shape(const size_t k, const CanonicalTable & t, const T * v) const {
            if (!same_layout(k, t)) return false;
            const Shape & s = shapes[k];
            for (size_t i = 0; i < s.values.size(); i++) {
                if (on_grid(s.values[i], t.exponent, t.bits) != on_grid(v[i], t.exponent, t.bits)) return false;
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

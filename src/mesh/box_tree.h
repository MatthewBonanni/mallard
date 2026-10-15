/**
 * @file box_tree.h
 * @author Matthew Bonanni (mbonanni001@gmail.com)
 * @brief Bounding-box tree over a set of axis-aligned boxes (e.g. the cells of
 *        a mesh): overlap and nearest-point queries.
 * @version 0.1
 * @date 2026-10-15
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 *
 */

#ifndef BOX_TREE_H
#define BOX_TREE_H

#include <array>
#include <cstdint>
#include <limits>
#include <vector>

#include "common_typedef.h"

class Mesh;

/** @brief An axis-aligned box in N_DIM dimensions; empty until a point is added. */
struct Box {
    std::array<double, N_DIM> lo, hi;

    Box() {
        lo.fill(std::numeric_limits<double>::max());
        hi.fill(std::numeric_limits<double>::lowest());
    }

    void add(const double * x) {
        for (size_t d = 0; d < N_DIM; d++) {
            lo[d] = x[d] < lo[d] ? x[d] : lo[d];
            hi[d] = x[d] > hi[d] ? x[d] : hi[d];
        }
    }

    void add(const Box & b) {
        add(b.lo.data());
        add(b.hi.data());
    }

    bool empty() const { return lo[0] > hi[0]; }

    /** @brief Grow every side by pad. */
    void inflate(double pad) {
        for (size_t d = 0; d < N_DIM; d++) {
            lo[d] -= pad;
            hi[d] += pad;
        }
    }

    bool overlaps(const Box & b) const {
        for (size_t d = 0; d < N_DIM; d++) {
            if (b.hi[d] < lo[d] || b.lo[d] > hi[d]) return false;
        }
        return true;
    }

    bool contains(const double * x) const {
        for (size_t d = 0; d < N_DIM; d++) {
            if (x[d] < lo[d] || x[d] > hi[d]) return false;
        }
        return true;
    }

    /** @brief Squared distance from x to the box (0 inside). */
    double distance2(const double * x) const {
        double s = 0.0;
        for (size_t d = 0; d < N_DIM; d++) {
            const double e = x[d] < lo[d] ? lo[d] - x[d] : x[d] > hi[d] ? x[d] - hi[d] : 0.0;
            s += e * e;
        }
        return s;
    }
};

/**
 * @brief Binary tree of boxes, split at the median of the box centers along
 *        the longest side of each node: O(n log n) to build, O(log n + k)
 *        per query of k results. Item ids are positions in the boxes it was
 *        built from.
 */
class BoxTree {
    public:
        BoxTree() = default;
        explicit BoxTree(const std::vector<Box> & boxes);

        /** @brief Every item whose box overlaps b, in increasing id. */
        void overlapping(const Box & b, std::vector<uint32_t> & items) const;

        /**
         * @brief Every item whose box contains x, in increasing id (the
         *        candidates of a point location).
         */
        void containing(const double * x, std::vector<uint32_t> & items) const;

        /**
         * @brief Item minimizing distance2(item, x) as computed by the caller
         *        (e.g. to a cell's centroid), visiting only nodes whose boxes
         *        may hold a closer one; ties go to the lower id. Returns the id,
         *        or UINT32_MAX for an empty tree, and the distance in d2.
         */
        template <typename Distance>
        uint32_t nearest(const double * x, Distance && distance2, double & d2) const {
            uint32_t best = UINT32_MAX;
            d2 = std::numeric_limits<double>::infinity();
            if (nodes.empty()) return best;
            std::vector<uint32_t> stack{0};
            while (!stack.empty()) {
                const Node & node = nodes[stack.back()];
                stack.pop_back();
                if (node.box.distance2(x) > d2) continue;
                if (node.count > 0) {
                    for (uint32_t k = node.first; k < node.first + node.count; k++) {
                        const double d = distance2(order[k]);
                        if (d < d2 || (d == d2 && order[k] < best)) {
                            d2 = d;
                            best = order[k];
                        }
                    }
                    continue;
                }
                const Node & a = nodes[node.first], & b = nodes[node.first + 1];
                const bool a_first = a.box.distance2(x) <= b.box.distance2(x);
                stack.push_back(a_first ? node.first + 1 : node.first);
                stack.push_back(a_first ? node.first : node.first + 1);
            }
            return best;
        }

        /** @brief Boxes of the nodes at depth level (or leaves above it): a coarse cover of all items. */
        std::vector<Box> cover(uint32_t level) const;

        /** @brief Box of all items. */
        Box bounds() const { return nodes.empty() ? Box() : nodes[0].box; }

        size_t size() const { return order.size(); }

    private:
        struct Node {
            Box box;
            uint32_t first = 0;  // leaf: first slot of order; inner: first of its two children
            uint32_t count = 0;  // leaf: items; inner: 0
        };
        static constexpr uint32_t LEAF = 8;
        std::vector<Node> nodes;
        std::vector<uint32_t> order;
        std::vector<Box> item_boxes;
};

/** @brief Box of each of the first n cells of a mesh, from its nodes. */
std::vector<Box> cell_boxes(const Mesh & mesh, uint32_t n);

#endif // BOX_TREE_H

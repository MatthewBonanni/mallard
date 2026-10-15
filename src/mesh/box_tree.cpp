/**
 * @file box_tree.cpp
 * @author Matthew Bonanni (mbonanni001@gmail.com)
 * @brief Bounding-box tree implementation.
 * @version 0.1
 * @date 2026-10-15
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 *
 */

#include "box_tree.h"

#include <algorithm>
#include <numeric>

#include "mesh.h"

BoxTree::BoxTree(const std::vector<Box> & boxes) : item_boxes(boxes) {
    const uint32_t n = static_cast<uint32_t>(boxes.size());
    order.resize(n);
    std::iota(order.begin(), order.end(), 0u);
    if (n == 0) return;
    nodes.reserve(2 * (n / LEAF + 1));
    nodes.emplace_back();
    // (node, first, count) still to split
    struct Task {
        uint32_t node, first, count;
    };
    std::vector<Task> tasks{{0, 0, n}};
    auto center = [&](uint32_t i, size_t d) { return boxes[i].lo[d] + boxes[i].hi[d]; };
    while (!tasks.empty()) {
        const Task task = tasks.back();
        tasks.pop_back();
        Box box;
        for (uint32_t k = task.first; k < task.first + task.count; k++) box.add(boxes[order[k]]);
        nodes[task.node].box = box;
        if (task.count <= LEAF) {
            nodes[task.node].first = task.first;
            nodes[task.node].count = task.count;
            continue;
        }
        size_t axis = 0;
        for (size_t d = 1; d < N_DIM; d++) {
            if (box.hi[d] - box.lo[d] > box.hi[axis] - box.lo[axis]) axis = d;
        }
        const uint32_t half = task.count / 2;
        auto begin = order.begin() + task.first;
        std::nth_element(begin, begin + half, begin + task.count, [&](uint32_t a, uint32_t b) {
            const double ca = center(a, axis), cb = center(b, axis);
            return ca < cb || (ca == cb && a < b);
        });
        const uint32_t child = static_cast<uint32_t>(nodes.size());
        nodes.emplace_back();
        nodes.emplace_back();
        nodes[task.node].first = child;
        nodes[task.node].count = 0;
        tasks.push_back({child, task.first, half});
        tasks.push_back({child + 1, task.first + half, task.count - half});
    }
}

void BoxTree::overlapping(const Box & b, std::vector<uint32_t> & items) const {
    items.clear();
    if (nodes.empty()) return;
    std::vector<uint32_t> stack{0};
    while (!stack.empty()) {
        const Node & node = nodes[stack.back()];
        stack.pop_back();
        if (!node.box.overlaps(b)) continue;
        if (node.count == 0) {
            stack.push_back(node.first);
            stack.push_back(node.first + 1);
            continue;
        }
        for (uint32_t k = node.first; k < node.first + node.count; k++) {
            if (item_boxes[order[k]].overlaps(b)) items.push_back(order[k]);
        }
    }
    std::sort(items.begin(), items.end());
}

void BoxTree::containing(const double * x, std::vector<uint32_t> & items) const {
    Box point;
    point.add(x);
    overlapping(point, items);
}

std::vector<Box> BoxTree::cover(uint32_t level) const {
    std::vector<Box> out;
    if (nodes.empty()) return out;
    std::vector<std::pair<uint32_t, uint32_t>> stack{{0, 0}};
    while (!stack.empty()) {
        const auto [i, depth] = stack.back();
        stack.pop_back();
        if (depth == level || nodes[i].count > 0) {
            out.push_back(nodes[i].box);
            continue;
        }
        stack.push_back({nodes[i].first, depth + 1});
        stack.push_back({nodes[i].first + 1, depth + 1});
    }
    return out;
}

std::vector<Box> cell_boxes(const Mesh & mesh, uint32_t n) {
    std::vector<Box> boxes(n);
    for (uint32_t c = 0; c < n; c++) {
        for (uint32_t k = 0; k < mesh.h_n_nodes_of_cell(c); k++) {
            double x[N_DIM];
            for (size_t d = 0; d < N_DIM; d++) x[d] = double(mesh.h_node_coords(mesh.h_node_of_cell(c, k), d));
            boxes[c].add(x);
        }
    }
    return boxes;
}

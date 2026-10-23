/**
 * @file tabulated.h
 * @author Matthew Bonanni (mbonanni001@gmail.com)
 * @brief Tabulated data read from files: tables on tensor grids of one to
 *        three coordinates, interpolated, and scattered point clouds, by
 *        nearest points; evaluated with the same code on the host and in
 *        device kernels.
 * @version 0.1
 * @date 2026-10-23
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 *
 */

#ifndef TABULATED_H
#define TABULATED_H

#include <cstdint>
#include <string>
#include <vector>

#include <Kokkos_Core.hpp>

namespace tabulated {

/** @brief Coordinates of a data set, at most. */
constexpr uint32_t MAX_DIMS = 3;
/** @brief Neighbors of a point cloud's inverse-distance weighting, at most. */
constexpr uint32_t MAX_NEIGHBORS = 16;
/** @brief Depth of a point cloud's tree traversal stack. */
constexpr uint32_t MAX_DEPTH = 64;

/** @brief A data set: where its values, grid axes or tree are in the shared arrays. */
struct Set {
    uint8_t points = 0;         // 0: a table on a tensor grid; 1: scattered points
    uint8_t cubic = 0;          // tables: 4-point Lagrange interpolation along each axis, else linear
    uint32_t dims = 1;
    uint32_t n_fields = 0;
    uint32_t values = 0;        // node or point p, field f: values[values + p * n_fields + f]
    uint32_t axes[MAX_DIMS] = {0, 0, 0}, n_axis[MAX_DIMS] = {1, 1, 1};  // tables: grid axes in axes
    uint32_t nodes = 0;         // points: the tree's root in nodes
    uint32_t items = 0;         // points: first slot of the tree's item order
    uint32_t coords = 0;        // points: point i at coords[3 * (coords + i)]
    uint32_t neighbors = 1;     // points: nearest points weighted, 1 for the nearest alone
};

/** @brief A node of a point cloud's bounding-box tree (BoxTree::Node with three coordinates). */
struct TreeNode {
    double lo[3] = {0.0, 0.0, 0.0}, hi[3] = {0.0, 0.0, 0.0};
    uint32_t first = 0;  // leaf: first slot of the item order; inner: first of its two children
    uint32_t count = 0;  // leaf: items; inner: 0
};

namespace detail {

KOKKOS_INLINE_FUNCTION double clamp(const double v, const double lo, const double hi) {
    return v < lo ? lo : (v > hi ? hi : v);
}

/**
 * @brief Interpolation weights along one axis x[0, n) (increasing) at q,
 *        clamped to the axis: the m nodes from first and their weights w.
 *        Linear, or Lagrange on the 4 nearest nodes (cubic: exact for cubic
 *        polynomials); at a node, exactly that node's value.
 */
KOKKOS_INLINE_FUNCTION void axis_weights(const double * x, const uint32_t n, const double q, const bool cubic,
                                         uint32_t & first, uint32_t & m, double w[4]) {
    if (n == 1) {
        first = 0;
        m = 1;
        w[0] = 1.0;
        return;
    }
    const double qc = clamp(q, x[0], x[n - 1]);
    uint32_t lo = 0, hi = n - 1;  // x[lo] <= qc, and qc < x[hi] unless qc is the last node
    while (hi - lo > 1) {
        const uint32_t mid = lo + (hi - lo) / 2;
        if (x[mid] <= qc) {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    if (!cubic || n == 2) {
        first = lo;
        m = 2;
        const double t = (qc - x[lo]) / (x[lo + 1] - x[lo]);
        w[0] = 1.0 - t;
        w[1] = t;
        return;
    }
    m = n < 4 ? n : 4;
    first = lo >= 1 ? lo - 1 : 0;
    if (first + m > n) first = n - m;
    for (uint32_t j = 0; j < m; j++) {
        double v = 1.0;
        for (uint32_t k = 0; k < m; k++) {
            if (k != j) v *= (qc - x[first + k]) / (x[first + j] - x[first + k]);
        }
        w[j] = v;
    }
}

} // namespace detail

/** @brief Data sets in one memory space, copyable by value into kernels. */
template <typename MemorySpace = Kokkos::DefaultExecutionSpace::memory_space>
struct DataView {
    Kokkos::View<const Set *, MemorySpace> sets;
    Kokkos::View<const double *, MemorySpace> axes, values, coords;
    Kokkos::View<const TreeNode *, MemorySpace> nodes;
    Kokkos::View<const uint32_t *, MemorySpace> order;

    /** @brief Field f of data set s at the coordinates q[0, dims). */
    KOKKOS_INLINE_FUNCTION double operator()(const uint32_t s, const uint32_t f, const double * q) const {
        const Set & set = sets(s);
        return set.points ? nearest(set, f, q) : table(set, f, q);
    }

    KOKKOS_INLINE_FUNCTION double table(const Set & set, const uint32_t f, const double * q) const {
        uint32_t first[MAX_DIMS] = {0, 0, 0}, m[MAX_DIMS] = {1, 1, 1}, stride[MAX_DIMS] = {0, 0, 0};
        double w[MAX_DIMS][4] = {{1.0, 0.0, 0.0, 0.0}, {1.0, 0.0, 0.0, 0.0}, {1.0, 0.0, 0.0, 0.0}};
        for (uint32_t d = 0; d < set.dims; d++) {
            detail::axis_weights(axes.data() + set.axes[d], set.n_axis[d], q[d], set.cubic, first[d], m[d], w[d]);
        }
        uint32_t s = 1;
        for (uint32_t d = set.dims; d-- > 0;) {
            stride[d] = s;
            s *= set.n_axis[d];
        }
        double sum = 0.0;
        for (uint32_t a = 0; a < m[0]; a++) {
            for (uint32_t b = 0; b < m[1]; b++) {
                for (uint32_t c = 0; c < m[2]; c++) {
                    const uint32_t node = (first[0] + a) * stride[0] + (first[1] + b) * stride[1] +
                                          (first[2] + c) * stride[2];
                    sum += w[0][a] * w[1][b] * w[2][c] * values(set.values + node * set.n_fields + f);
                }
            }
        }
        return sum;
    }

    /**
     * @brief The value at the nearest point, or the inverse-distance-squared
     *        average of the set's neighbors nearest (exact at a data point).
     *        Ties go to the lower point index, so the result does not depend
     *        on the order of the search.
     */
    KOKKOS_INLINE_FUNCTION double nearest(const Set & set, const uint32_t f, const double * q) const {
        double best_d[MAX_NEIGHBORS];
        uint32_t best_i[MAX_NEIGHBORS];
        uint32_t found = 0;
        const uint32_t k = set.neighbors;
        auto box_distance = [&](const TreeNode & node) {
            double s = 0.0;
            for (uint32_t d = 0; d < set.dims; d++) {
                const double e = q[d] < node.lo[d] ? node.lo[d] - q[d] : (q[d] > node.hi[d] ? q[d] - node.hi[d] : 0.0);
                s += e * e;
            }
            return s;
        };
        uint32_t stack[MAX_DEPTH];
        uint32_t top = 0;
        stack[top++] = 0;
        while (top > 0) {
            const TreeNode & node = nodes(set.nodes + stack[--top]);
            if (found == k && box_distance(node) > best_d[k - 1]) continue;
            if (node.count > 0) {
                for (uint32_t j = node.first; j < node.first + node.count; j++) {
                    const uint32_t id = order(set.items + j);
                    const double * p = &coords(3 * (set.coords + id));
                    double d2 = 0.0;
                    for (uint32_t d = 0; d < set.dims; d++) d2 += (q[d] - p[d]) * (q[d] - p[d]);
                    if (found == k && (d2 > best_d[k - 1] || (d2 == best_d[k - 1] && id > best_i[k - 1]))) continue;
                    uint32_t slot = found < k ? found++ : k - 1;
                    while (slot > 0 && (d2 < best_d[slot - 1] || (d2 == best_d[slot - 1] && id < best_i[slot - 1]))) {
                        best_d[slot] = best_d[slot - 1];
                        best_i[slot] = best_i[slot - 1];
                        slot--;
                    }
                    best_d[slot] = d2;
                    best_i[slot] = id;
                }
                continue;
            }
            const TreeNode & a = nodes(set.nodes + node.first), & b = nodes(set.nodes + node.first + 1);
            const bool a_first = box_distance(a) <= box_distance(b);
            if (top + 2 > MAX_DEPTH) break;
            stack[top++] = a_first ? node.first + 1 : node.first;
            stack[top++] = a_first ? node.first : node.first + 1;
        }
        if (found == 0) return Kokkos::nan("");
        if (k == 1 || best_d[0] == 0.0) return values(set.values + best_i[0] * set.n_fields + f);
        double sum = 0.0, weight = 0.0;
        for (uint32_t j = 0; j < found; j++) {
            const double w = 1.0 / best_d[j];
            sum += w * values(set.values + best_i[j] * set.n_fields + f);
            weight += w;
        }
        return sum / weight;
    }
};

/**
 * @brief The columns of a CSV file: a header of names, then rows of numbers
 *        separated by commas (or spaces); lines starting with # are comments.
 */
struct Csv {
    std::vector<std::string> names;
    std::vector<std::vector<double>> columns;

    /** @brief Index of the named column, or -1. */
    int32_t column(const std::string & name) const;
};

/** @brief Read a CSV file; errors name the file and line. */
Csv read_csv(const std::string & file);

/** @brief Data sets on the host, gathered for the device. */
class Data {
    public:
        /**
         * @brief Add a table on a tensor grid: the rows, in any order, hold
         *        every combination of the distinct values of the coordinates
         *        exactly once.
         * @param coordinates [coordinate][row], one to three coordinates.
         * @param fields [field][row].
         * @param cubic Lagrange interpolation on 4 nodes per axis, else linear.
         * @param where Name for error messages.
         * @return The set's index.
         */
        uint32_t add_table(const std::vector<std::vector<double>> & coordinates,
                           const std::vector<std::vector<double>> & fields, bool cubic, const std::string & where);

        /**
         * @brief Add a point cloud, evaluated at its nearest point, or by
         *        inverse distance squared over its neighbors nearest points.
         * @param coordinates [coordinate][point], at most three coordinates.
         * @param fields [field][point].
         * @param neighbors Points averaged, 1 to MAX_NEIGHBORS.
         * @param where Name for error messages.
         * @return The set's index.
         */
        uint32_t add_points(const std::vector<std::vector<double>> & coordinates,
                            const std::vector<std::vector<double>> & fields, uint32_t neighbors,
                            const std::string & where);

        size_t size() const { return m_sets.size(); }
        const Set & set(const uint32_t s) const { return m_sets[s]; }

        /** @brief The sets in host memory (views of this object's vectors). */
        DataView<Kokkos::HostSpace> host() const;
        /** @brief The sets in the default device's memory. */
        DataView<> device() const;

    private:
        std::vector<Set> m_sets;
        std::vector<double> m_axes, m_values, m_coords;
        std::vector<TreeNode> m_nodes;
        std::vector<uint32_t> m_order;
};

} // namespace tabulated

#endif // TABULATED_H

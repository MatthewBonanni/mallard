/**
 * @file homogeneous_average.cpp
 * @brief Averages of cell fields over homogeneous directions.
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 */

#include "homogeneous_average.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <limits>
#include <map>
#include <numeric>
#include <span>
#include <stdexcept>

#include "comm.h"

namespace homogeneous {

namespace {

/** @brief Cluster index of each value: a sorted value joins its predecessor's cluster within tol. */
std::vector<uint32_t> clusters(const std::vector<double> & values, double tol) {
    std::vector<uint32_t> order(values.size());
    std::iota(order.begin(), order.end(), 0u);
    std::stable_sort(order.begin(), order.end(), [&](uint32_t a, uint32_t b) { return values[a] < values[b]; });
    std::vector<uint32_t> id(values.size());
    uint32_t current = 0;
    for (size_t k = 0; k < order.size(); k++) {
        if (k > 0 && values[order[k]] - values[order[k - 1]] > tol) current++;
        id[order[k]] = current;
    }
    return id;
}

/**
 * @brief Sums per row of points (coords[p * n_kept + d]) that agree in every kept direction:
 *        sums[p * width + j] added into their row's, which comes first in the result.
 */
std::vector<double> merge(const std::vector<double> & coords, const std::vector<double> & sums, size_t n_kept,
                          size_t width, const std::vector<double> & tol) {
    const size_t n = n_kept ? coords.size() / n_kept : 0;
    std::vector<std::vector<uint32_t>> ids(n_kept);
    for (size_t d = 0; d < n_kept; d++) {
        std::vector<double> v(n);
        for (size_t p = 0; p < n; p++) v[p] = coords[p * n_kept + d];
        ids[d] = clusters(v, tol[d]);
    }
    std::map<std::vector<uint32_t>, size_t> rows;
    std::vector<double> out;
    std::vector<uint32_t> key(n_kept);
    for (size_t p = 0; p < n; p++) {
        for (size_t d = 0; d < n_kept; d++) key[d] = ids[d][p];
        const auto [it, added] = rows.try_emplace(key, rows.size());
        if (added) out.resize(out.size() + width, 0.0);
        for (size_t j = 0; j < width; j++) out[it->second * width + j] += sums[p * width + j];
    }
    return out;
}

} // namespace

std::string Profile::csv() const {
    std::string out;
    for (const auto & a : axes) out += a + ",";
    out += "n_cells";
    for (const auto & n : names) out += "," + n;
    out += "\n";
    char buffer[32];
    for (size_t r = 0; r < rows(); r++) {
        for (size_t d = 0; d < axes.size(); d++) {
            std::snprintf(buffer, sizeof(buffer), "%.17g,", coords[r * axes.size() + d]);
            out += buffer;
        }
        out += std::to_string(n_cells[r]);
        for (size_t j = 0; j < names.size(); j++) {
            std::snprintf(buffer, sizeof(buffer), ",%.17g", values[r * names.size() + j]);
            out += buffer;
        }
        out += "\n";
    }
    return out;
}

Profile average(const std::vector<std::array<double, N_DIM>> & centers, const std::vector<double> & weights,
                const std::vector<double> & density, const std::vector<Column> & columns,
                const std::string & averaged, double rtol) {
    static constexpr const char * AXES[3] = {"x", "y", "z"};
    std::vector<size_t> kept;
    for (const char a : averaged) {
        if (a < 'x' || a >= 'x' + N_DIM) {
            throw std::runtime_error(std::string("--homogeneous: ") + a + " is not a direction of this " +
                                     std::to_string(N_DIM) + "D build (" + (N_DIM == 3 ? "x, y, z" : "x, y") + ")");
        }
    }
    for (size_t d = 0; d < N_DIM; d++) {
        if (averaged.find(static_cast<char>('x' + d)) == std::string::npos) kept.push_back(d);
    }
    if (kept.empty()) throw std::runtime_error("--homogeneous averages every direction: leave at least one");
    const size_t n_kept = kept.size(), n_cols = columns.size(), n_cells = weights.size();

    // Tolerances from the extent of the whole mesh, in each kept direction
    std::vector<double> extent(3 * n_kept, -std::numeric_limits<double>::infinity());
    for (size_t c = 0; c < n_cells; c++) {
        if (weights[c] <= 0.0) continue;
        for (size_t d = 0; d < n_kept; d++) {
            const double x = centers[c][kept[d]];
            extent[3 * d] = std::max(extent[3 * d], x);
            extent[3 * d + 1] = std::max(extent[3 * d + 1], -x);
            extent[3 * d + 2] = std::max(extent[3 * d + 2], std::abs(x));
        }
    }
    comm::allreduce(std::span<double>(extent), comm::Op::MAX);
    std::vector<double> tol(n_kept);
    for (size_t d = 0; d < n_kept; d++) {
        tol[d] = rtol * std::max({extent[3 * d] + extent[3 * d + 1], extent[3 * d + 2], 1e-300});
    }

    // Per cell: count, weight, Favre weight, weighted coordinates, weighted columns
    const size_t width = 3 + n_kept + n_cols;
    std::vector<double> coords, sums;
    for (size_t c = 0; c < n_cells; c++) {
        const double w = weights[c];
        if (w <= 0.0) continue;
        const double w_f = density.empty() ? 0.0 : w * density[c];
        for (size_t d = 0; d < n_kept; d++) coords.push_back(centers[c][kept[d]]);
        sums.insert(sums.end(), {1.0, w, w_f});
        for (size_t d = 0; d < n_kept; d++) sums.push_back(w * centers[c][kept[d]]);
        for (const Column & col : columns) {
            double x = col.values[c];
            if (col.mean_a >= 0) x += columns[static_cast<size_t>(col.mean_a)].values[c] *
                                      columns[static_cast<size_t>(col.mean_b)].values[c];
            sums.push_back((col.favre ? w_f : w) * x);
        }
    }
    const std::vector<double> local = merge(coords, sums, n_kept, width, tol);
    // Rows of every rank, merged again at their mean coordinates
    const size_t n_local = local.size() / width;
    std::vector<double> local_coords(n_local * n_kept);
    for (size_t r = 0; r < n_local; r++) {
        for (size_t d = 0; d < n_kept; d++) local_coords[r * n_kept + d] = local[r * width + 3 + d] / local[r * width + 1];
    }
    const std::vector<double> all_coords = comm::gatherv(local_coords), all = comm::gatherv(local);
    Profile profile;
    for (const size_t d : kept) profile.axes.emplace_back(AXES[d]);
    for (const Column & col : columns) profile.names.push_back(col.name);
    if (!comm::is_root()) return profile;
    const std::vector<double> rows = merge(all_coords, all, n_kept, width, tol);
    const size_t n_rows = rows.size() / width;
    std::vector<size_t> order(n_rows);
    std::iota(order.begin(), order.end(), size_t(0));
    const auto coord = [&](size_t r, size_t d) { return rows[r * width + 3 + d] / rows[r * width + 1]; };
    std::sort(order.begin(), order.end(), [&](size_t a, size_t b) {
        for (size_t d = 0; d < n_kept; d++) {
            if (coord(a, d) != coord(b, d)) return coord(a, d) < coord(b, d);
        }
        return false;
    });
    for (const size_t r : order) {
        const double * s = &rows[r * width];
        profile.n_cells.push_back(static_cast<uint64_t>(std::llround(s[0])));
        for (size_t d = 0; d < n_kept; d++) profile.coords.push_back(coord(r, d));
        const size_t first = profile.values.size();
        for (size_t j = 0; j < n_cols; j++) profile.values.push_back(s[3 + n_kept + j] / (columns[j].favre ? s[2] : s[1]));
        for (size_t j = 0; j < n_cols; j++) {
            const Column & col = columns[j];
            if (col.mean_a < 0) continue;
            profile.values[first + j] -= profile.values[first + static_cast<size_t>(col.mean_a)] *
                                         profile.values[first + static_cast<size_t>(col.mean_b)];
        }
    }
    return profile;
}

} // namespace homogeneous

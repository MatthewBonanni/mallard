/**
 * @file mesh_quality.cpp
 * @brief Cell quality metrics of a finite volume mesh.
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 */

#include "mesh_quality.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numbers>

#include "comm.h"

namespace {

enum Metric { NON_ORTHOGONALITY, SKEWNESS, ASPECT_RATIO, VOLUME_RATIO };

double norm(const std::array<double, N_DIM> & v) {
    double s = 0.0;
    for (const double x : v) s += x * x;
    return std::sqrt(s);
}

double dot(const std::array<double, N_DIM> & a, const std::array<double, N_DIM> & b) {
    double s = 0.0;
    for (size_t i = 0; i < N_DIM; i++) s += a[i] * b[i];
    return s;
}

} // namespace

MeshQuality mesh_quality(const Mesh & mesh) {
    const uint32_t n_owned = mesh.n_owned();
    MeshQuality q;
    for (auto & metric : q.cells) metric.assign(n_owned, 0.0);
    for (auto & ratio : {ASPECT_RATIO, VOLUME_RATIO}) std::fill(q.cells[ratio].begin(), q.cells[ratio].end(), 1.0);
    std::vector<double> d_min(n_owned, std::numeric_limits<double>::infinity()), d_max(n_owned, 0.0);

    for (uint32_t f = 0; f < mesh.n_faces; f++) {
        const int32_t c0 = mesh.h_cells_of_face(f, 0), c1 = mesh.h_cells_of_face(f, 1);
        const bool own0 = static_cast<uint32_t>(c0) < n_owned;
        const bool own1 = c1 >= 0 && static_cast<uint32_t>(c1) < n_owned;
        if (!own0 && !own1) continue;
        std::array<double, N_DIM> xf{}, n{}, x0{}, x1{};
        for (size_t i = 0; i < N_DIM; i++) {
            xf[i] = static_cast<double>(mesh.h_face_coords(f, i));
            n[i] = static_cast<double>(mesh.h_face_normals(f, i));
            x0[i] = static_cast<double>(mesh.h_cell_coords(c0, i));
        }
        const double n_norm = norm(n);
        if (n_norm == 0.0) continue;  // axis faces of axisymmetric meshes
        for (double & x : n) x /= n_norm;

        auto to_face = [&](int32_t c, const std::array<double, N_DIM> & xc) {
            std::array<rtype, N_DIM> point{};
            if (mesh.is_periodic()) {
                point = mesh.h_face_point(f, uint32_t(c));
            } else {
                for (size_t i = 0; i < N_DIM; i++) point[i] = mesh.h_face_coords(f, i);
            }
            std::array<double, N_DIM> r{};
            for (size_t i = 0; i < N_DIM; i++) r[i] = static_cast<double>(point[i]) - xc[i];
            return norm(r);
        };
        auto extent = [&](int32_t c, const std::array<double, N_DIM> & xc) {
            const double d = to_face(c, xc);
            d_min[uint32_t(c)] = std::min(d_min[uint32_t(c)], d);
            d_max[uint32_t(c)] = std::max(d_max[uint32_t(c)], d);
        };
        if (own0) extent(c0, x0);

        double non_orthogonality = 0.0, skewness = 0.0, volume_ratio = 1.0;
        if (c1 < 0) {
            std::array<double, N_DIM> d{};
            for (size_t i = 0; i < N_DIM; i++) d[i] = xf[i] - x0[i];
            const double length = norm(d);
            if (length > 0.0) non_orthogonality = std::acos(std::clamp(dot(n, d) / length, -1.0, 1.0));
        } else {
            std::array<rtype, N_DIM> c1_coords{}, x1_seen{};
            for (size_t i = 0; i < N_DIM; i++) c1_coords[i] = mesh.h_cell_coords(c1, i);
            x1_seen = c1_coords;
            if (mesh.is_periodic()) mesh.h_forward(mesh.h_face_shift(f), c1_coords.data(), x1_seen.data());
            for (size_t i = 0; i < N_DIM; i++) x1[i] = static_cast<double>(x1_seen[i]);
            if (own1) {
                std::array<double, N_DIM> x1_own{};
                for (size_t i = 0; i < N_DIM; i++) x1_own[i] = static_cast<double>(mesh.h_cell_coords(c1, i));
                extent(c1, x1_own);
            }
            std::array<double, N_DIM> d{};
            for (size_t i = 0; i < N_DIM; i++) d[i] = x1[i] - x0[i];
            const double length = norm(d);
            const double along = dot(n, d);
            non_orthogonality = std::acos(std::clamp(along / length, -1.0, 1.0));
            if (along > 0.0) {
                std::array<double, N_DIM> to_plane{};
                std::array<double, N_DIM> rel{};
                for (size_t i = 0; i < N_DIM; i++) rel[i] = xf[i] - x0[i];
                const double s = dot(n, rel) / along;
                for (size_t i = 0; i < N_DIM; i++) to_plane[i] = xf[i] - (x0[i] + s * d[i]);
                skewness = norm(to_plane) / length;
            }
            const double v0 = static_cast<double>(mesh.h_cell_volume(c0));
            const double v1 = static_cast<double>(mesh.h_cell_volume(c1));
            volume_ratio = std::max(v0, v1) / std::min(v0, v1);
        }
        non_orthogonality *= 180.0 / std::numbers::pi;
        for (const int32_t c : {c0, c1}) {
            if (c < 0 || static_cast<uint32_t>(c) >= n_owned) continue;
            const auto i = static_cast<uint32_t>(c);
            q.cells[NON_ORTHOGONALITY][i] = std::max(q.cells[NON_ORTHOGONALITY][i], non_orthogonality);
            q.cells[SKEWNESS][i] = std::max(q.cells[SKEWNESS][i], skewness);
            q.cells[VOLUME_RATIO][i] = std::max(q.cells[VOLUME_RATIO][i], volume_ratio);
        }
    }
    for (uint32_t c = 0; c < n_owned; c++) {
        if (d_min[c] > 0.0 && std::isfinite(d_min[c])) q.cells[ASPECT_RATIO][c] = d_max[c] / d_min[c];
    }
    return q;
}

std::array<QualitySummary, MeshQuality::N_METRICS> summarize(const Mesh & mesh, const MeshQuality & quality,
                                                              const size_t n_worst) {
    constexpr size_t RECORD = 2 + N_DIM;  // value, global cell, centroid
    std::array<QualitySummary, MeshQuality::N_METRICS> out;
    const uint32_t n_owned = mesh.n_owned();
    for (size_t m = 0; m < MeshQuality::N_METRICS; m++) {
        const std::vector<double> & values = quality.cells[m];
        QualitySummary & s = out[m];
        double sum = 0.0;
        uint64_t over = 0;
        for (const double v : values) {
            s.max = std::max(s.max, v);
            sum += v;
            over += v > MeshQuality::LIMITS[m];
        }
        s.max = comm::allreduce(s.max, comm::Op::MAX);
        const auto totals = comm::allreduce(std::array<double, 2>{sum, static_cast<double>(n_owned)}, comm::Op::SUM);
        s.mean = totals[1] > 0.0 ? totals[0] / totals[1] : 0.0;
        s.n_over_limit = comm::allreduce(over, comm::Op::SUM);

        std::vector<uint32_t> order(n_owned);
        for (uint32_t c = 0; c < n_owned; c++) order[c] = c;
        const size_t n_local = std::min<size_t>(n_worst, order.size());
        std::partial_sort(order.begin(), order.begin() + static_cast<std::ptrdiff_t>(n_local), order.end(),
                          [&](uint32_t a, uint32_t b) { return values[a] > values[b]; });
        std::vector<double> records;
        for (size_t k = 0; k < n_local; k++) {
            const uint32_t c = order[k];
            records.push_back(values[c]);
            records.push_back(static_cast<double>(mesh.h_global_cell(c)));
            for (size_t i = 0; i < N_DIM; i++) records.push_back(static_cast<double>(mesh.h_cell_coords(c, i)));
        }
        if (comm::size() > 1) records = comm::allgatherv(records);
        for (size_t r = 0; r + RECORD <= records.size(); r += RECORD) {
            QualityCell cell;
            cell.value = records[r];
            cell.cell = static_cast<uint64_t>(records[r + 1]);
            for (size_t i = 0; i < N_DIM; i++) cell.x[i] = records[r + 2 + i];
            s.worst.push_back(cell);
        }
        std::stable_sort(s.worst.begin(), s.worst.end(), [](const QualityCell & a, const QualityCell & b) {
            return a.value > b.value || (a.value == b.value && a.cell < b.cell);
        });
        if (s.worst.size() > n_worst) s.worst.resize(n_worst);
    }
    return out;
}

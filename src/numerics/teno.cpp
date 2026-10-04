/**
 * @file teno.cpp
 * @author Matthew Bonanni (mbonanni001@gmail.com)
 * @brief TENO-E reconstruction implementation.
 * @version 0.2
 * @date 2026-10-02
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 *
 */

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <unordered_map>
#include <string>
#include <utility>
#include <tuple>
#include <type_traits>
#include <stdexcept>
#include <vector>

#include <sys/mman.h>

#include <Kokkos_Core.hpp>

#include "comm.h"
#include "face_reconstruction.h"

#include "input.h"
#include "teno.h"

namespace {

// Relative tolerance for geometric tests on the mesh, whose coordinates are rtype
constexpr double GEOMETRY_TOL = precision_tol<double>(1e-10, 1e-5);

// Largest accepted Lebesgue constant of a central stencil's reconstruction at
// the cell's face quadrature points. Stencils on regular hexahedra, prisms and
// tetrahedra stay at 2-5; full-rank stencils that resolve a direction only
// through small centroid offsets reach tens to hundreds and amplify the
// truncation error alike.
constexpr double MAX_LEBESGUE = 10.0;

// Stencil candidates are ranked by distance in the metric of the local mesh
// spacing where its largest to smallest spacing ratio exceeds this. Regular
// tilings of cubes measure 1 (hexahedra), 2.13 (Kuhn tetrahedra), 1.73
// (prisms) and 1.57 (pyramids) from their cell shapes alone, and keep
// physical distance; a 2:1 stretch of Kuhn tetrahedra measures 3.15
constexpr double SPACING_ANISOTROPY = 2.5;

/**
 * @brief Gauss-Legendre nodes and weights on [-1, 1] (Newton iteration).
 */
void gauss_legendre(int n, std::vector<double> & x, std::vector<double> & w) {
    x.resize(n);
    w.resize(n);
    for (int i = 0; i < n; i++) {
        double z = std::cos(M_PI * (i + 0.75) / (n + 0.5));
        double dp = 0.0;
        for (int it = 0; it < 100; it++) {
            double p0 = 1.0, p1 = z;
            for (int k = 2; k <= n; k++) {
                const double p2 = ((2.0 * k - 1.0) * z * p1 - (k - 1.0) * p0) / k;
                p0 = p1;
                p1 = p2;
            }
            if (n == 1) { p1 = z; p0 = 1.0; }
            dp = n * (z * p1 - p0) / (z * z - 1.0);
            const double dz = p1 / dp;
            z -= dz;
            if (std::abs(dz) < 1e-15) break;
        }
        x[i] = z;
        w[i] = 2.0 / ((1.0 - z * z) * dp * dp);
    }
}

/**
 * @brief Collapsed (Duffy) Gauss quadrature on the reference triangle
 *        (0,0), (1,0), (0,1); weights sum to 1/2. Exact for total degree 2n-2.
 */
struct TriangleRule {
    std::vector<double> xi, eta, w;
    explicit TriangleRule(int n) {
        std::vector<double> g, gw;
        gauss_legendre(n, g, gw);
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                const double s = 0.5 * (g[i] + 1.0);
                const double t = 0.5 * (g[j] + 1.0);
                xi.push_back(s);
                eta.push_back(t * (1.0 - s));
                w.push_back(0.25 * gw[i] * gw[j] * (1.0 - s));
            }
        }
    }
};

/**
 * @brief Integrate f(xi, eta) over a polygon (given as vertices in the scaled
 *        frame) by fan triangulation.
 */
template <typename F>
void integrate_polygon(const std::vector<double> & px, const std::vector<double> & py,
                       const TriangleRule & rule, F && f) {
    for (size_t k = 1; k + 1 < px.size(); k++) {
        const double ax = px[k] - px[0], ay = py[k] - py[0];
        const double bx = px[k + 1] - px[0], by = py[k + 1] - py[0];
        const double det = std::abs(ax * by - ay * bx);
        for (size_t q = 0; q < rule.w.size(); q++) {
            const double x = px[0] + rule.xi[q] * ax + rule.eta[q] * bx;
            const double y = py[0] + rule.xi[q] * ay + rule.eta[q] * by;
            f(x, y, rule.w[q] * det);
        }
    }
}

/**
 * @brief Least-squares pseudo-inverse P = R^-1 Q^T (n x m) of a full-column-rank
 *        m x n matrix A (row-major) via Householder QR.
 * @param max_condition Largest accepted ratio of the diagonal entries of R
 *        after column equilibration (an estimate of the condition number).
 * @return False if A is too ill conditioned.
 */
bool pseudo_inverse(std::vector<double> A, int m, int n, std::vector<double> & P, double max_condition) {
    // Equilibrate columns so the rank test is independent of monomial scaling
    std::vector<double> col_scale(n, 0.0);
    for (int j = 0; j < n; j++) {
        for (int i = 0; i < m; i++) col_scale[j] += A[i * n + j] * A[i * n + j];
        col_scale[j] = std::sqrt(col_scale[j]);
        if (col_scale[j] == 0.0) return false;
    }
    if constexpr (N_DIM == 3) {
        // A column of round-off (e.g. the xy monomial over a stencil whose centroids
        // all lie on axis planes) would pass the rank test once equilibrated
        const double largest = *std::max_element(col_scale.begin(), col_scale.end());
        for (int j = 0; j < n; j++) {
            if (col_scale[j] < GEOMETRY_TOL * largest) return false;
        }
    }
    for (int j = 0; j < n; j++) {
        for (int i = 0; i < m; i++) A[i * n + j] /= col_scale[j];
    }
    // Reflections are applied a row at a time to all columns at once, which
    // reads both matrices contiguously and keeps each entry's sequence of
    // operations, so results do not depend on the loop order
    std::vector<double> QT(m * m, 0.0);
    for (int i = 0; i < m; i++) QT[i * m + i] = 1.0;
    std::vector<double> v(m), d(m);
    auto reflect = [&](double * M, const int cols, const int k, const double vnorm2) {
        std::fill(d.begin(), d.begin() + cols, 0.0);
        for (int i = k; i < m; i++) {
            const double * row = M + i * cols;
            for (int j = 0; j < cols; j++) d[j] += v[i] * row[j];
        }
        for (int j = 0; j < cols; j++) d[j] *= 2.0 / vnorm2;
        for (int i = k; i < m; i++) {
            double * row = M + i * cols;
            for (int j = 0; j < cols; j++) row[j] -= d[j] * v[i];
        }
    };
    double max_diag = 0.0;
    for (int k = 0; k < n; k++) {
        double norm = 0.0;
        for (int i = k; i < m; i++) norm += A[i * n + k] * A[i * n + k];
        norm = std::sqrt(norm);
        if (norm == 0.0) return false;
        const double alpha = (A[k * n + k] > 0.0) ? -norm : norm;
        for (int i = 0; i < m; i++) v[i] = (i < k) ? 0.0 : A[i * n + k];
        v[k] -= alpha;
        double vnorm2 = 0.0;
        for (int i = k; i < m; i++) vnorm2 += v[i] * v[i];
        if (vnorm2 == 0.0) continue;
        reflect(A.data(), n, k, vnorm2);
        reflect(QT.data(), m, k, vnorm2);
        max_diag = std::max(max_diag, std::abs(A[k * n + k]));
    }
    for (int k = 0; k < n; k++) {
        if (std::abs(A[k * n + k]) * max_condition < max_diag) return false;
    }
    P.assign(n * m, 0.0);
    for (int k = n - 1; k >= 0; k--) {
        double * row = &P[k * m];
        for (int j = 0; j < m; j++) row[j] = QT[k * m + j];
        for (int l = k + 1; l < n; l++) {
            const double a = A[k * n + l];
            for (int j = 0; j < m; j++) row[j] -= a * P[l * m + j];
        }
        for (int j = 0; j < m; j++) row[j] /= A[k * n + k];
    }
    for (int k = 0; k < n; k++) {
        for (int j = 0; j < m; j++) P[k * m + j] /= col_scale[k];
    }
    return true;
}

/**
 * @brief Lebesgue constant of the reconstruction U(x_q) = U_0 + sum_s c_qs (U_s - U_0),
 *        c_qs = psi(x_q)^T P[:, s]: max_q |1 - sum_s c_qs| + sum_s |c_qs|, the
 *        largest factor by which it can amplify cell averages.
 * @param psi Zero-mean basis at the evaluation points, row-major (point, l).
 * @param P Pseudo-inverse (n x m, row-major) of the stencil's least-squares system.
 */
double lebesgue_constant(const std::vector<double> & psi, int n, const std::vector<double> & P, int m) {
    double lambda = 0.0;
    std::vector<double> coef(m);
    for (size_t q = 0; q * n < psi.size(); q++) {
        std::fill(coef.begin(), coef.end(), 0.0);
        for (int l = 0; l < n; l++) {
            const double p = psi[q * n + l];
            for (int s = 0; s < m; s++) coef[s] += p * P[l * m + s];
        }
        double sum = 0.0, abs_sum = 0.0;
        for (int s = 0; s < m; s++) {
            const double c = coef[s];
            sum += c;
            abs_sum += std::abs(c);
        }
        lambda = std::max(lambda, std::abs(1.0 - sum) + abs_sum);
    }
    return lambda;
}

/** @brief Stable sort of items by key(item), evaluating each key once. */
template <typename T, typename K>
void sort_by_key(std::vector<T> & items, K && key) {
    std::vector<std::pair<double, uint32_t>> order(items.size());
    for (size_t k = 0; k < items.size(); k++) order[k] = {key(items[k]), static_cast<uint32_t>(k)};
    std::stable_sort(order.begin(), order.end(), [](const auto & a, const auto & b) { return a.first < b.first; });
    std::vector<T> sorted;
    sorted.reserve(items.size());
    for (const auto & o : order) sorted.push_back(items[o.second]);
    items.swap(sorted);
}

/** @brief Periodic lattice offset of a stencil entry's cell. */
using Lattice = std::array<int, 3>;

Lattice operator+(const Lattice & a, const std::array<int8_t, 3> & b) { return {a[0] + b[0], a[1] + b[1], a[2] + b[2]}; }

/**
 * @brief Cells reached by a stencil search with their lattice offsets: across
 *        periodic boundaries the search continues into translated copies of
 *        the mesh, so a cell can appear at several offsets.
 */
struct Visit {
    uint32_t cell;
    Lattice lattice;
    std::array<double, 3> t;  // translation of the cell's copy
    bool operator==(const Visit & o) const { return cell == o.cell && lattice == o.lattice; }
};

/** @brief Translation of a lattice offset. */
std::array<double, 3> lattice_translation(const Mesh & mesh, const Lattice & lattice) {
    std::array<double, 3> t = {0.0, 0.0, 0.0};
    for (size_t j = 0; j < mesh.periodic_translations.size(); j++) {
        FOR_I_DIM t[i] += lattice[j] * double(mesh.periodic_translations[j][i]);
    }
    return t;
}

/** @brief Vertex neighbors of a visited cell, at their lattice offsets. */
template <typename F>
void for_each_neighbor(const Mesh & mesh, const Visit & v, F && f) {
    for (uint32_t k = mesh.h_offsets_cells_of_cell(v.cell); k < mesh.h_offsets_cells_of_cell(v.cell + 1); k++) {
        const uint8_t s = mesh.h_cells_of_cell_shift(k);
        Visit nb{mesh.h_cells_of_cell(k), v.lattice + mesh.shift_lattice[s], {0.0, 0.0, 0.0}};
        nb.t = lattice_translation(mesh, nb.lattice);
        f(nb);
    }
}

/**
 * @brief Set of visits (cell and lattice offset) by open addressing; a linear
 *        search of the visited cells made the stencil search quadratic.
 */
class VisitSet {
    public:
        VisitSet() : keys(64, EMPTY) {}

        /** @brief Insert v; false if it was present. */
        bool insert(const Visit & v) {
            if (2 * (n + 1) > keys.size()) grow();
            return place(key(v));
        }

    private:
        static constexpr uint64_t EMPTY = ~uint64_t(0);
        static constexpr int LATTICE_BITS = 10;

        // The cell in the low 32 bits and each lattice offset in 10 bits: bits
        // 62 and 63 stay clear, so no key equals EMPTY
        static uint64_t key(const Visit & v) {
            constexpr int bound = 1 << (LATTICE_BITS - 1);
            uint64_t k = v.cell;
            for (int a = 0; a < 3; a++) {
                if (v.lattice[a] < -bound || v.lattice[a] >= bound) {
                    throw std::runtime_error("TENO: stencil search reached too many periodic copies of the mesh.");
                }
                k |= static_cast<uint64_t>(v.lattice[a] + bound) << (32 + LATTICE_BITS * a);
            }
            return k;
        }

        bool place(const uint64_t k) {
            const size_t mask = keys.size() - 1;
            for (size_t slot = ((k * 0x9E3779B97F4A7C15ULL) >> 32) & mask;; slot = (slot + 1) & mask) {
                if (keys[slot] == k) return false;
                if (keys[slot] == EMPTY) {
                    keys[slot] = k;
                    n++;
                    return true;
                }
            }
        }

        void grow() {
            std::vector<uint64_t> old(2 * keys.size(), EMPTY);
            old.swap(keys);
            n = 0;
            for (const uint64_t k : old) {
                if (k != EMPTY) place(k);
            }
        }

        std::vector<uint64_t> keys;
        size_t n = 0;
};

/**
 * @brief Vertex-neighbor layers around a cell, grown on demand and shared by
 *        the searches for its large and sector stencils.
 */
class Layers {
    public:
        Layers(const Mesh & m, const uint32_t i) : mesh(m) {
            cells.push_back(Visit{i, {0, 0, 0}, {0.0, 0.0, 0.0}});
            seen.insert(cells[0]);
        }

        /**
         * @brief Depth of the search that adds layers while at most n_min
         *        cells are gathered, up to max_layers or until none are left.
         * @param truncated Set once a layer is grown from a cell of the
         *        outermost halo layer, which misses neighbors on other ranks.
         */
        int reach(const size_t n_min, const int max_layers, bool & truncated) {
            int depth = 0;
            while (depth < max_layers && end[depth] <= n_min) {
                if (depth + 1 == static_cast<int>(end.size()) && !grow(truncated)) break;
                depth++;
            }
            return depth;
        }

        /** @brief Number of cells within depth layers, the first ones of cells. */
        size_t size(const int depth) const { return end[depth]; }

        std::vector<Visit> cells;

    private:
        bool grow(bool & truncated) {
            if (exhausted) return false;
            const size_t begin = end.size() > 1 ? end[end.size() - 2] : 0;
            for (size_t k = begin; k < end.back(); k++) {
                const Visit v = cells[k];
                if (v.cell >= mesh.n_complete()) truncated = true;
                for_each_neighbor(mesh, v, [&](const Visit & nb) {
                    if (seen.insert(nb)) cells.push_back(nb);
                });
            }
            if (cells.size() == end.back()) {
                exhausted = true;
                return false;
            }
            end.push_back(cells.size());
            return true;
        }

        const Mesh & mesh;
        VisitSet seen;
        std::vector<size_t> end = {1};  // (depth): cells within depth layers
        bool exhausted = false;
};

/**
 * @brief Eigenvalues w and orthonormal eigenvectors (columns of v) of a
 *        symmetric 3 x 3 matrix a (cyclic Jacobi).
 */
void symmetric_eigen(std::array<double, 9> a, double w[3], double v[9]) {
    for (int k = 0; k < 9; k++) v[k] = (k % 4 == 0) ? 1.0 : 0.0;
    for (int sweep = 0; sweep < 50; sweep++) {
        const double off = a[1] * a[1] + a[2] * a[2] + a[5] * a[5];
        if (off <= 1e-30 * (a[0] * a[0] + a[4] * a[4] + a[8] * a[8])) break;
        for (int p = 0; p < 2; p++) {
            for (int q = p + 1; q < 3; q++) {
                const double apq = a[3 * p + q];
                if (apq == 0.0) continue;
                const double theta = 0.5 * (a[3 * q + q] - a[3 * p + p]) / apq;
                const double t = (theta >= 0.0 ? 1.0 : -1.0) / (std::abs(theta) + std::sqrt(theta * theta + 1.0));
                const double c = 1.0 / std::sqrt(t * t + 1.0), s = t * c;
                for (int k = 0; k < 3; k++) {
                    const double akp = a[3 * k + p], akq = a[3 * k + q];
                    a[3 * k + p] = c * akp - s * akq;
                    a[3 * k + q] = s * akp + c * akq;
                }
                for (int k = 0; k < 3; k++) {
                    const double apk = a[3 * p + k], aqk = a[3 * q + k];
                    a[3 * p + k] = c * apk - s * aqk;
                    a[3 * q + k] = s * apk + c * aqk;
                }
                for (int k = 0; k < 3; k++) {
                    const double vkp = v[3 * k + p], vkq = v[3 * k + q];
                    v[3 * k + p] = c * vkp - s * vkq;
                    v[3 * k + q] = s * vkp + c * vkq;
                }
            }
        }
    }
    for (int k = 0; k < 3; k++) w[k] = a[4 * k];
}

/**
 * @brief Metric of unit determinant whose unit length is the mesh spacing in
 *        every direction, from the second moment m of the offsets of a cell's
 *        neighbors; the identity unless its spacing ratio exceeds
 *        SPACING_ANISOTROPY.
 */
std::array<double, 9> spacing_metric(const double m[9]) {
    const std::array<double, 9> identity = {1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0};
    double w[3], v[9];
    symmetric_eigen({m[0], m[1], m[2], m[3], m[4], m[5], m[6], m[7], m[8]}, w, v);
    if (!(std::min({w[0], w[1], w[2]}) > 0.0)) return identity;
    // Log spacings relative to their mean, and the largest log spacing ratio
    double l[3];
    for (int k = 0; k < 3; k++) l[k] = 0.5 * std::log(w[k]);
    const double mean = (l[0] + l[1] + l[2]) / 3.0;
    for (int k = 0; k < 3; k++) l[k] -= mean;
    const double spread = std::max({l[0], l[1], l[2]}) - std::min({l[0], l[1], l[2]});
    if (!(spread > std::log(SPACING_ANISOTROPY))) return identity;
    std::array<double, 9> metric = {};
    for (int k = 0; k < 3; k++) {
        const double e = std::exp(-2.0 * l[k]);
        for (int a = 0; a < 3; a++) {
            for (int b = 0; b < 3; b++) metric[3 * a + b] += v[3 * a + k] * e * v[3 * b + k];
        }
    }
    return metric;
}

/**
 * @brief Metric in which the stencil candidates of cell i are ranked: that of
 *        the second moment of its vertex neighbors' centroid offsets, completed
 *        across its boundary faces by the images of the cells at most as far
 *        from the boundary as the cell itself (see spacing_metric).
 */
std::array<double, 9> ranking_metric(const Mesh & mesh, const Kokkos::View<int32_t *>::host_mirror_type & face_bc,
                                     const Kokkos::View<BoundaryCondition *>::host_mirror_type & bcs, const uint32_t i) {
    using Point = std::array<double, 3>;
    double spread[9] = {};
    if constexpr (N_DIM == 3) {
        Point x0;
        for (int a = 0; a < 3; a++) x0[a] = double(mesh.h_cell_coords(i, a));
        const double h = std::cbrt(double(mesh.h_cell_volume(i)));
        std::vector<Point> offsets = {{0.0, 0.0, 0.0}};
        for_each_neighbor(mesh, Visit{i, {0, 0, 0}, {0.0, 0.0, 0.0}}, [&](const Visit & nb) {
            Point d;
            for (int a = 0; a < 3; a++) d[a] = double(mesh.h_cell_coords(nb.cell, a)) + nb.t[a] - x0[a];
            offsets.push_back(d);
        });
        for (uint32_t k = 0; k < mesh.h_n_faces_of_cell(i); k++) {
            const uint32_t f = mesh.h_face_of_cell(i, k);
            if (mesh.h_cells_of_face(f, 1) >= 0 || face_bc(f) < 0) continue;
            if (bcs(face_bc(f)).type == BoundaryType::PARTITION) continue;
            Point n;
            for (int a = 0; a < 3; a++) n[a] = double(mesh.h_face_normals(f, a)) / double(mesh.h_face_area(f));
            double d0 = 0.0;
            for (int a = 0; a < 3; a++) d0 += (double(mesh.h_face_coords(f, a)) - x0[a]) * n[a];
            const size_t n_offsets = offsets.size();
            for (size_t s = 0; s < n_offsets; s++) {
                double d = d0;
                for (int a = 0; a < 3; a++) d -= offsets[s][a] * n[a];
                if (d > d0 + GEOMETRY_TOL * h) continue;
                Point image;
                for (int a = 0; a < 3; a++) image[a] = offsets[s][a] + 2.0 * d * n[a];
                offsets.push_back(image);
            }
        }
        for (const Point & d : offsets) {
            for (int a = 0; a < 3; a++) {
                for (int b = 0; b < 3; b++) spread[3 * a + b] += d[a] * d[b];
            }
        }
    }
    return spacing_metric(spread);
}

// Cells per batch of the precomputation, bounding the unpacked per-cell
// tables held at once; a multiple of every slice size
constexpr uint32_t CHUNK_CELLS = 8192;

/** @brief Precomputed data of one reconstructed cell, before packing. */
struct CellTables {
    uint8_t gather_depth = 0;
    rtype scale = 0.0;
    std::vector<rtype> basis_mean;                  // (l)
    std::vector<rtype> si;                          // upper triangle of (l, m)
    std::vector<int32_t> large_cells, large_faces;  // (s)
    std::vector<rtype> large_pinv;                  // (s, l)
    std::array<uint16_t, teno::MAX_FACES> small_size = {};
    std::vector<int32_t> small_cells, small_faces;  // the faces' sector stencils one after another
    std::vector<rtype> small_pinv;                  // (s, l)
};

using IndexRow = std::vector<int32_t> CellTables::*;
using ValueRow = std::vector<rtype> CellTables::*;

template <typename T>
using HostUnmanaged = Kokkos::View<T *, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>>;

/** @brief Rows [c0, c0 + h.extent(0)) of a per-cell device array, from the host. */
template <typename View>
void upload_rows(const View & dev, const uint32_t c0, const typename View::host_mirror_type & h) {
    if (h.extent(0) == 0) return;
    auto tmp = Kokkos::create_mirror_view_and_copy(typename View::memory_space(), h);
    const std::pair<size_t, size_t> rows(c0, c0 + h.extent(0));
    if constexpr (View::rank() == 1) {
        Kokkos::deep_copy(Kokkos::subview(dev, rows), tmp);
    } else {
        Kokkos::deep_copy(Kokkos::subview(dev, rows, Kokkos::ALL()), tmp);
    }
}

/** @brief Host copy of rows [c0, c1) of a per-cell device array. */
template <typename View>
typename View::host_mirror_type download_rows(const View & dev, const uint32_t c0, const uint32_t c1) {
    using Host = typename View::host_mirror_type;
    const std::pair<size_t, size_t> rows(c0, c1);
    Host h;
    if constexpr (View::rank() == 1) {
        h = Host("teno_rows", c1 - c0);
    } else {
        h = Host("teno_rows", c1 - c0, dev.extent(1));
    }
    if (c1 == c0) return h;
    auto tmp = Kokkos::create_mirror_view(typename View::memory_space(), h);
    if constexpr (View::rank() == 1) {
        Kokkos::deep_copy(tmp, Kokkos::subview(dev, rows));
    } else {
        Kokkos::deep_copy(tmp, Kokkos::subview(dev, rows, Kokkos::ALL()));
    }
    Kokkos::deep_copy(h, tmp);
    return h;
}

/**
 * @brief Array in pages of its own, returned to the system as soon as it is
 *        released. Allocators may keep large freed blocks (macOS does), and
 *        the packed chunks, freed while the final arrays fill on the host,
 *        then doubled the peak memory of the setup.
 */
template <typename T>
class PageArray {
    public:
        PageArray() = default;
        PageArray(const size_t n, const T value) : n_(n) {
            if (n == 0) return;
            void * p = mmap(nullptr, n * sizeof(T), PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
            if (p == MAP_FAILED) throw std::bad_alloc();
            data_ = static_cast<T *>(p);
            std::fill(data_, data_ + n, value);
        }
        PageArray(PageArray && other) noexcept
            : data_(std::exchange(other.data_, nullptr)), n_(std::exchange(other.n_, 0)) {}
        PageArray & operator=(PageArray && other) noexcept {
            if (this != &other) {
                release();
                data_ = std::exchange(other.data_, nullptr);
                n_ = std::exchange(other.n_, 0);
            }
            return *this;
        }
        PageArray(const PageArray &) = delete;
        PageArray & operator=(const PageArray &) = delete;
        ~PageArray() { release(); }

        T * data() { return data_; }
        size_t size() const { return n_; }
        T & operator[](const size_t k) { return data_[k]; }

    private:
        void release() {
            if (data_ != nullptr) munmap(data_, n_ * sizeof(T));
            data_ = nullptr;
            n_ = 0;
        }

        T * data_ = nullptr;
        size_t n_ = 0;
};

/**
 * @brief Moves one stencil family (large or sector) of CellTables into a
 *        teno::PackedStencils and back. The device arrays can only be sized
 *        once every stencil is known, so chunks are packed on the host first.
 */
class PackedRows {
    public:
        PackedRows(uint8_t slice_shift, uint8_t values_per_slot, IndexRow cell_row, IndexRow face_row, ValueRow pinv_row)
            : shift(slice_shift), width(values_per_slot), cells(cell_row), faces(face_row), pinv(pinv_row) {}

        /** @brief Pack the stencils of cells [c0, c0 + tables.size()), which follow the previous chunk. */
        void add(const uint32_t c0, const std::vector<CellTables> & tables) {
            const uint32_t slice = 1u << shift;
            if (c0 % slice != 0 || c0 >> shift != slice_start.size() - 1) {
                throw std::logic_error("TENO: stencil chunks must be packed in order.");
            }
            const uint32_t n = tables.size();
            Chunk chunk;
            chunk.slot0 = slice_start.back();
            for (uint32_t a = 0; a < n; a += slice) {
                size_t largest = 0;
                for (uint32_t c = a; c < std::min(n, a + slice); c++) largest = std::max(largest, (tables[c].*cells).size());
                slice_start.push_back(slice_start.back() + largest);
            }
            const size_t n_slots = (slice_start.back() - chunk.slot0) << shift;
            chunk.cells = PageArray<int32_t>(n_slots, -1);
            chunk.faces = PageArray<int32_t>(n_slots, -1);
            chunk.pinv = PageArray<rtype>(n_slots * width, 0.0);
            Kokkos::parallel_for("teno_pack", Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(0, n),
                                 [&](const uint32_t c) {
                const CellTables & t = tables[c];
                const uint64_t start = slice_start[(c0 + c) >> shift] - chunk.slot0;
                const uint32_t lane = (c0 + c) & (slice - 1);
                for (size_t s = 0; s < (t.*cells).size(); s++) {
                    chunk.cells[((start + s) << shift) + lane] = (t.*cells)[s];
                    chunk.faces[((start + s) << shift) + lane] = (t.*faces)[s];
                    for (uint32_t l = 0; l < width; l++) {
                        chunk.pinv[(((start + s) * width + l) << shift) + lane] = (t.*pinv)[s * width + l];
                    }
                }
            });
            chunks.push_back(std::move(chunk));
        }

        /** @brief The packed stencils on the device; each host chunk is released once copied. */
        teno::PackedStencils finish(const std::string & label) {
            teno::PackedStencils out;
            out.shift = shift;
            out.width = width;
            const size_t n_slots = slice_start.back() << shift;
            out.cells = Kokkos::View<int32_t *>(Kokkos::view_alloc(Kokkos::WithoutInitializing, label), n_slots);
            out.faces = Kokkos::View<int32_t *>(Kokkos::view_alloc(Kokkos::WithoutInitializing, label + "_face"), n_slots);
            out.pinv = Kokkos::View<rtype *>(Kokkos::view_alloc(Kokkos::WithoutInitializing, label + "_pinv"),
                                             n_slots * width);
            for (Chunk & chunk : chunks) {
                const size_t b = chunk.slot0 << shift;
                const std::pair<size_t, size_t> slots(b, b + chunk.cells.size());
                const std::pair<size_t, size_t> values(b * width, (b + chunk.cells.size()) * width);
                Kokkos::deep_copy(Kokkos::subview(out.cells, slots),
                                  HostUnmanaged<int32_t>(chunk.cells.data(), chunk.cells.size()));
                Kokkos::deep_copy(Kokkos::subview(out.faces, slots),
                                  HostUnmanaged<int32_t>(chunk.faces.data(), chunk.faces.size()));
                Kokkos::deep_copy(Kokkos::subview(out.pinv, values),
                                  HostUnmanaged<rtype>(chunk.pinv.data(), chunk.pinv.size()));
                chunk = Chunk();
            }
            chunks.clear();
            out.slice_start = Kokkos::View<uint64_t *>(label + "_slice_start", slice_start.size());
            Kokkos::deep_copy(out.slice_start, HostUnmanaged<const uint64_t>(slice_start.data(), slice_start.size()));
            return out;
        }

        /**
         * @brief Stencils of cells [c0, c0 + tables.size()) from the device
         *        (h_slice_start: host copy of packed.slice_start; sizes: slots
         *        of each cell).
         */
        void download(const teno::PackedStencils & packed, const std::vector<uint64_t> & h_slice_start,
                      const uint32_t c0, const std::vector<uint16_t> & sizes, std::vector<CellTables> & tables) const {
            const uint8_t sh = packed.shift;
            const uint32_t slice = 1u << sh;
            const uint32_t c1 = c0 + tables.size();
            const uint64_t slot0 = h_slice_start[c0 >> sh];
            const uint64_t slot1 = h_slice_start[(c1 + slice - 1) >> sh];
            const std::pair<size_t, size_t> slots(slot0 << sh, slot1 << sh);
            const std::pair<size_t, size_t> values(slots.first * packed.width, slots.second * packed.width);
            std::vector<int32_t> h_cells(slots.second - slots.first), h_faces(h_cells.size());
            std::vector<rtype> h_pinv(values.second - values.first);
            Kokkos::deep_copy(HostUnmanaged<int32_t>(h_cells.data(), h_cells.size()), Kokkos::subview(packed.cells, slots));
            Kokkos::deep_copy(HostUnmanaged<int32_t>(h_faces.data(), h_faces.size()), Kokkos::subview(packed.faces, slots));
            Kokkos::deep_copy(HostUnmanaged<rtype>(h_pinv.data(), h_pinv.size()), Kokkos::subview(packed.pinv, values));
            for (uint32_t c = 0; c < tables.size(); c++) {
                CellTables & t = tables[c];
                const uint64_t start = h_slice_start[(c0 + c) >> sh] - slot0;
                const uint32_t lane = (c0 + c) & (slice - 1);
                (t.*cells).resize(sizes[c]);
                (t.*faces).resize(sizes[c]);
                (t.*pinv).resize(sizes[c] * packed.width);
                for (uint16_t s = 0; s < sizes[c]; s++) {
                    (t.*cells)[s] = h_cells[((start + s) << sh) + lane];
                    (t.*faces)[s] = h_faces[((start + s) << sh) + lane];
                    for (uint32_t l = 0; l < packed.width; l++) {
                        (t.*pinv)[s * packed.width + l] = h_pinv[(((start + s) * packed.width + l) << sh) + lane];
                    }
                }
            }
        }

    private:
        struct Chunk {
            uint64_t slot0 = 0;
            PageArray<int32_t> cells, faces;
            PageArray<rtype> pinv;
        };
        uint8_t shift;
        uint8_t width;
        IndexRow cells, faces;
        ValueRow pinv;
        std::vector<uint64_t> slice_start = {0};  // (slice + 1)
        std::vector<Chunk> chunks;
};

PackedRows large_rows(const TENO & scheme) {
    return PackedRows(scheme.slice_shift, scheme.n_dof_large, &CellTables::large_cells, &CellTables::large_faces,
                      &CellTables::large_pinv);
}

PackedRows small_rows(const TENO & scheme) {
    return PackedRows(scheme.slice_shift, teno::NK_SMALL, &CellTables::small_cells, &CellTables::small_faces,
                      &CellTables::small_pinv);
}

/** @brief Moves per-cell tables, chunk by chunk, into TENO's device arrays. */
class TableBuilder {
    public:
        TableBuilder(TENO & owner, const uint32_t n_cells)
            : scheme(owner), large(large_rows(owner)), small(small_rows(owner)) {
            const uint8_t nk = scheme.n_dof_large;
            scheme.scale = Kokkos::View<rtype *>("teno_scale", n_cells);
            scheme.basis_mean = Kokkos::View<rtype **>("teno_basis_mean", n_cells, nk);
            scheme.si_matrix = Kokkos::View<rtype **>("teno_si_matrix", n_cells, nk * (nk + 1) / 2);
            scheme.stencil_large_size = Kokkos::View<uint16_t *>("teno_stencil_large_size", n_cells);
            scheme.stencil_small_size = Kokkos::View<uint16_t **>("teno_stencil_small_size", n_cells, teno::MAX_FACES);
            scheme.gather_depth.assign(n_cells, 0);
        }

        /** @brief Tables of cells [c0, c0 + tables.size()), the chunk after the previous one. */
        void add(const uint32_t c0, const std::vector<CellTables> & tables) {
            const uint32_t n = tables.size();
            Kokkos::View<rtype *>::host_mirror_type h_scale("teno_scale_rows", n);
            Kokkos::View<rtype **>::host_mirror_type h_mean("teno_basis_mean_rows", n, scheme.basis_mean.extent(1));
            Kokkos::View<rtype **>::host_mirror_type h_si("teno_si_rows", n, scheme.si_matrix.extent(1));
            Kokkos::View<uint16_t *>::host_mirror_type h_large_size("teno_large_size_rows", n);
            Kokkos::View<uint16_t **>::host_mirror_type h_small_size("teno_small_size_rows", n, teno::MAX_FACES);
            Kokkos::parallel_for("teno_rows", Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(0, n),
                                 [&](const uint32_t c) {
                const CellTables & t = tables[c];
                scheme.gather_depth[c0 + c] = t.gather_depth;
                h_scale(c) = t.scale;
                for (size_t l = 0; l < t.basis_mean.size(); l++) h_mean(c, l) = t.basis_mean[l];
                for (size_t k = 0; k < t.si.size(); k++) h_si(c, k) = t.si[k];
                h_large_size(c) = t.large_cells.size();
                for (uint8_t k = 0; k < teno::MAX_FACES; k++) h_small_size(c, k) = t.small_size[k];
            });
            upload_rows(scheme.scale, c0, h_scale);
            upload_rows(scheme.basis_mean, c0, h_mean);
            upload_rows(scheme.si_matrix, c0, h_si);
            upload_rows(scheme.stencil_large_size, c0, h_large_size);
            upload_rows(scheme.stencil_small_size, c0, h_small_size);
            large.add(c0, tables);
            small.add(c0, tables);
        }

        void finish() {
            scheme.stencil_large = large.finish("teno_stencil_large");
            scheme.stencil_small = small.finish("teno_stencil_small");
        }

    private:
        TENO & scheme;
        PackedRows large, small;
};

/** @brief Host copy of the slice starts of packed stencils. */
std::vector<uint64_t> host_slice_start(const teno::PackedStencils & packed) {
    std::vector<uint64_t> h(packed.slice_start.extent(0));
    Kokkos::deep_copy(HostUnmanaged<uint64_t>(h.data(), h.size()), packed.slice_start);
    return h;
}

/** @brief Tables of the chunk of reconstructed cells starting at c0 from TENO's device arrays. */
void download_tables(const TENO & scheme, const std::vector<uint64_t> & large_slices,
                     const std::vector<uint64_t> & small_slices, const uint32_t c0, std::vector<CellTables> & tables) {
    const uint32_t c1 = c0 + tables.size();
    auto h_scale = download_rows(scheme.scale, c0, c1);
    auto h_mean = download_rows(scheme.basis_mean, c0, c1);
    auto h_si = download_rows(scheme.si_matrix, c0, c1);
    auto h_large_size = download_rows(scheme.stencil_large_size, c0, c1);
    auto h_small_size = download_rows(scheme.stencil_small_size, c0, c1);
    std::vector<uint16_t> n_large(tables.size()), n_small(tables.size(), 0);
    for (uint32_t c = 0; c < tables.size(); c++) {
        CellTables & t = tables[c];
        t.gather_depth = scheme.gather_depth[c0 + c];
        t.scale = h_scale(c);
        t.basis_mean.resize(h_mean.extent(1));
        for (size_t l = 0; l < t.basis_mean.size(); l++) t.basis_mean[l] = h_mean(c, l);
        t.si.resize(h_si.extent(1));
        for (size_t k = 0; k < t.si.size(); k++) t.si[k] = h_si(c, k);
        n_large[c] = h_large_size(c);
        for (uint8_t k = 0; k < teno::MAX_FACES; k++) {
            t.small_size[k] = h_small_size(c, k);
            n_small[c] += t.small_size[k];
        }
    }
    large_rows(scheme).download(scheme.stencil_large, large_slices, c0, n_large, tables);
    small_rows(scheme).download(scheme.stencil_small, small_slices, c0, n_small, tables);
}

/**
 * @brief Run precompute(i, tables, failed_large, invalid_small) over the
 *        reconstructed cells chunk by chunk, moving each chunk's tables to
 *        the device arrays. Returns the largest central stencil.
 */
template <typename F>
uint16_t precompute_in_chunks(TENO & scheme, const uint32_t n_reconstructed, F && precompute, uint32_t & n_failed_large,
                              uint32_t & n_invalid_small) {
    TableBuilder builder(scheme, n_reconstructed);
    std::vector<CellTables> chunk;
    uint16_t ns_used = 0;
    for (uint32_t c0 = 0; c0 < n_reconstructed; c0 += CHUNK_CELLS) {
        chunk.assign(std::min(CHUNK_CELLS, n_reconstructed - c0), CellTables());
        uint32_t failed = 0, invalid = 0;
        Kokkos::parallel_reduce("teno_precompute",
                                Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace, Kokkos::Schedule<Kokkos::Dynamic>>(
                                    c0, c0 + chunk.size()),
                                [&](const uint32_t i, uint32_t & failed_large, uint32_t & invalid_small) {
            precompute(i, chunk[i - c0], failed_large, invalid_small);
        }, failed, invalid);
        n_failed_large += failed;
        n_invalid_small += invalid;
        for (const CellTables & t : chunk) ns_used = std::max<uint16_t>(ns_used, t.large_cells.size());
        if (n_failed_large == 0) builder.add(c0, chunk);
    }
    if (n_failed_large == 0) builder.finish();
    return ns_used;
}

} // namespace

TENO::TENO() {
    type = FaceReconstructionType::TENO;
}

TENO::~TENO() {
    // Empty
}

void TENO::read_options(const toml::value & input) {
    const int order = toml::find_or<int>(input, "order", 5);
    // The small stencils are degree 2, so the central polynomial must be at least degree 2
    if (order < 3 || order > teno::MAX_DEGREE + 1) {
        throw std::runtime_error("TENO order must be between 3 and " +
                                 std::to_string(teno::MAX_DEGREE + 1) + " (use MUSCL for second order).");
    }
    degree = order - 1;
    n_dof_large = teno::n_dof(degree);
    stencil_factor = find_real_or(input, "stencil_factor", 2.0);
    n_stencil_large = static_cast<uint16_t>(std::ceil(stencil_factor * n_dof_large));
    n_stencil_small = toml::find_or<int>(input, "small_stencil_size", (N_DIM == 2) ? 10 : 2 * teno::NK_SMALL);
    sigma_threshold = find_real_or(input, "troubled_threshold", 1.0e-3);
    sigma_upper = find_real_or(input, "troubled_upper", 1.0e-2);
    C_T = find_real_or(input, "C_T", -1.0);
    characteristic = toml::find_or<bool>(input, "characteristic", true);
    max_condition = find_real_or(input, "max_condition", 1.0e8);
    bound_preserving = toml::find_or<bool>(input, "bound_preserving", false);
    cache_file = toml::find_or<std::string>(input, "cache_file", "");
    // One file per rank, made for this partition
    if (!cache_file.empty() && comm::size() > 1) {
        cache_file += ".r" + std::to_string(comm::rank()) + "-of-" + std::to_string(comm::size());
    }
}

void TENO::init(const toml::value & input) {
    read_options(input);
    const int order = degree + 1;
    const int n_gp = std::max(1, std::min<int>(teno::MAX_FACE_QUAD, (order + 1) / 2));
    quadrature_face = GaussLegendre(n_gp);
    if constexpr (N_DIM == 3) init_face_quadrature_3d(order);

    cache_loaded = !cache_file.empty() && load_cache();
    if (!cache_loaded) compute_stencils_and_matrices();
    allocate_scratch();
    largest_stencil = comm::allreduce(largest_stencil, comm::Op::MAX);
}

logging::Items TENO::summary() const {
    using logging::format;
    std::string stencils = format("large %d+1 (largest %u), small %d+1", static_cast<int>(n_stencil_large),
                                  largest_stencil, static_cast<int>(n_stencil_small));
    if (n_sector_unavailable > 0 && comm::size() == 1) {
        stencils += format(", %lld sector stencils cut by boundaries", static_cast<long long>(n_sector_unavailable));
    }
    logging::Items out = {
        {"Reconstruction", format("TENO, order %d (degree %d), %d face quadrature points", degree + 1, degree,
                                  static_cast<int>(n_face_quadrature_points()))},
        {"TENO stencils", stencils},
        {"TENO switch", "threshold " + logging::real(double(sigma_threshold)) + ", C_T " +
                            (C_T > 0.0_r ? logging::real(double(C_T)) : std::string("adaptive")) + ", characteristic " +
                            (characteristic ? "yes" : "no") + ", bound-preserving " +
                            (bound_preserving ? "yes" : "no")},
    };
    if (!cache_status.empty()) out.emplace_back("TENO cache", cache_status);
    return out;
}

uint8_t TENO::n_face_quadrature_points() const {
    if constexpr (N_DIM == 3) return face_quad_weights.extent(1);
    return quadrature_face.h_points.extent(0);
}

void TENO::compute_stencils_and_matrices() {
    if constexpr (N_DIM == 3) {
        compute_stencils_and_matrices_3d();
        return;
    }
    const uint32_t n_cells = mesh->n_cells;
    const uint8_t r = degree;
    const uint8_t nk = n_dof_large;
    const uint16_t ns = n_stencil_large;
    const uint16_t nss = n_stencil_small;
    const uint16_t ns_max = static_cast<uint16_t>(std::ceil(3.5 * nk));
    const uint16_t nss_max = 2 * nss;

    for (uint32_t i = 0; i < n_cells; i++) {
        if (mesh->h_n_faces_of_cell(i) > teno::MAX_FACES) {
            throw std::runtime_error("TENO supports cells with at most " +
                                     std::to_string(teno::MAX_FACES) + " faces.");
        }
    }

    auto h_face_bc = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), boundaries.face_bc);
    auto h_bcs = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), boundaries.bcs);

    const TriangleRule rule(r + 2);
    uint32_t n_failed_large = 0;
    uint32_t n_invalid_small = 0;

    auto precompute = [&](const uint32_t i, CellTables & out, uint32_t & failed_large, uint32_t & invalid_small) {
        const double x0 = double(mesh->h_cell_coords(i, 0));
        const double y0 = double(mesh->h_cell_coords(i, 1));
        const double h = std::sqrt(double(mesh->h_cell_volume(i)));
        out.scale = h;

        // Stencil entries are interior cells, or mirror images of interior cells
        // across a straight boundary segment (face >= 0) carrying the boundary
        // condition's ghost state. Mirrors give boundary cells full, centered
        // stencils instead of one-sided extrapolation, which is unstable at
        // inflow boundaries. Across periodic boundaries, cells are taken at
        // their translation (tx, ty); (mx, my) is a point of the mirror line.
        struct Entry {
            uint32_t cell;
            int32_t face;
            double x, y;
            double tx, ty;
            double mx, my;
        };
        auto mirror = [&](int32_t face, double mx, double my, double px, double py, double & qx, double & qy) {
            if (face < 0) {
                qx = px;
                qy = py;
                return;
            }
            const double nx = double(mesh->h_face_normals(face, 0)) / double(mesh->h_face_area(face));
            const double ny = double(mesh->h_face_normals(face, 1)) / double(mesh->h_face_area(face));
            const double d = (px - mx) * nx + (py - my) * ny;
            qx = px - 2.0 * d * nx;
            qy = py - 2.0 * d * ny;
        };

        // Mean of each monomial over a stencil entry, in the frame of target cell i
        auto monomial_means = [&](const Entry & e, uint8_t deg, std::vector<double> & means) {
            const uint32_t n_nodes = mesh->h_n_nodes_of_cell(e.cell);
            std::vector<double> px(n_nodes), py(n_nodes);
            for (uint32_t k = 0; k < n_nodes; k++) {
                const uint32_t node = mesh->h_node_of_cell(e.cell, k);
                double qx, qy;
                mirror(e.face, e.mx, e.my, double(mesh->h_node_coords(node, 0)) + e.tx, double(mesh->h_node_coords(node, 1)) + e.ty,
                       qx, qy);
                px[k] = (qx - x0) / h;
                py[k] = (qy - y0) / h;
            }
            const uint8_t n = teno::n_dof(deg);
            means.assign(n, 0.0);
            double area = 0.0;
            double phi[teno::MAX_NK];
            integrate_polygon(px, py, rule, [&](double x, double y, double w) {
                teno::monomials(deg, x, y, phi);
                for (uint8_t l = 0; l < n; l++) means[l] += w * phi[l];
                area += w;
            });
            for (uint8_t l = 0; l < n; l++) means[l] /= area;
        };

        std::vector<double> mean0;
        monomial_means(Entry{i, -1, x0, y0, 0.0, 0.0, 0.0, 0.0}, r, mean0);
        out.basis_mean.assign(mean0.begin(), mean0.end());

        // Whether a point lies strictly inside the copy of a cell translated by t
        auto point_in_cell = [&](const Visit & v, double px, double py) {
            const uint32_t c = v.cell;
            const uint32_t n = mesh->h_n_nodes_of_cell(c);
            px -= v.t[0];
            py -= v.t[1];
            int sign = 0;
            for (uint32_t k = 0; k < n; k++) {
                const uint32_t a = mesh->h_node_of_cell(c, k), b = mesh->h_node_of_cell(c, (k + 1) % n);
                const double cross = (double(mesh->h_node_coords(b, 0)) - double(mesh->h_node_coords(a, 0))) * (py - double(mesh->h_node_coords(a, 1))) -
                                     (double(mesh->h_node_coords(b, 1)) - double(mesh->h_node_coords(a, 1))) * (px - double(mesh->h_node_coords(a, 0)));
                const double tol = precision_tol<double>(1e-12, 1e-5) * h * h;
                const int s = (cross > tol) - (cross < -tol);
                if (s == 0) return false;  // On an edge: treat as outside (mirror of a boundary cell)
                if (sign == 0) sign = s;
                if (s != sign) return false;
            }
            return true;
        };

        // Candidates by vertex-neighbor layers plus their mirror images across
        // nearby boundary lines, sorted by distance
        int layers_used = 0;
        bool truncated = false;
        Layers layers(*mesh, i);
        int entries_depth = 0;
        std::vector<Entry> entries;
        auto gather = [&](size_t n_min, int max_layers) {
            const int depth = layers.reach(n_min, max_layers, truncated);
            layers_used = std::max(layers_used, depth);
            if (depth == entries_depth) return entries;
            entries_depth = depth;
            entries.clear();
            const std::vector<Visit> cells(layers.cells.begin(), layers.cells.begin() + layers.size(depth));
            // Straight boundary lines touched by the gathered cells. One line can
            // carry several conditions (e.g. inflow then wall), so each image takes
            // its state from the line's face nearest to it.
            struct LineFace {
                int32_t face;
                double x, y;  // centroid, translated with its cell
            };
            struct Line {
                double nx, ny;
                std::vector<LineFace> faces;
            };
            std::vector<Line> lines;
            for (const Visit & v : cells) {
                const uint32_t c = v.cell;
                for (uint32_t k = 0; k < mesh->h_n_faces_of_cell(c); k++) {
                    const uint32_t f = mesh->h_face_of_cell(c, k);
                    if (mesh->h_cells_of_face(f, 1) >= 0 || h_face_bc(f) < 0) continue;
                    if (h_bcs(h_face_bc(f)).type == BoundaryType::PARTITION) continue;
                    const double nx = double(mesh->h_face_normals(f, 0)) / double(mesh->h_face_area(f));
                    const double ny = double(mesh->h_face_normals(f, 1)) / double(mesh->h_face_area(f));
                    const LineFace lf{static_cast<int32_t>(f), double(mesh->h_face_coords(f, 0)) + v.t[0], double(mesh->h_face_coords(f, 1)) + v.t[1]};
                    Line * match = nullptr;
                    for (Line & line : lines) {
                        const LineFace & g = line.faces[0];
                        const double off = (lf.x - g.x) * line.nx + (lf.y - g.y) * line.ny;
                        if (std::abs(nx * line.nx + ny * line.ny - 1.0) < GEOMETRY_TOL && std::abs(off) < GEOMETRY_TOL * h) {
                            match = &line;
                            break;
                        }
                    }
                    if (match == nullptr) {
                        lines.push_back(Line{nx, ny, {}});
                        match = &lines.back();
                    }
                    const bool known = std::any_of(match->faces.begin(), match->faces.end(), [&](const LineFace & g) {
                        return g.face == lf.face && g.x == lf.x && g.y == lf.y;
                    });
                    if (!known) match->faces.push_back(lf);
                }
            }
            for (const Visit & v : cells) {
                const uint32_t c = v.cell;
                const double cx = double(mesh->h_cell_coords(c, 0)) + v.t[0], cy = double(mesh->h_cell_coords(c, 1)) + v.t[1];
                if (!(c == i && v.lattice == Lattice{0, 0, 0})) {
                    entries.push_back(Entry{c, -1, cx, cy, v.t[0], v.t[1], 0.0, 0.0});
                }
                for (const Line & line : lines) {
                    const LineFace & first = line.faces[0];
                    Entry e{c, first.face, 0.0, 0.0, v.t[0], v.t[1], first.x, first.y};
                    mirror(e.face, e.mx, e.my, cx, cy, e.x, e.y);
                    // Images that land inside the domain (non-convex boundaries) are not ghosts
                    bool inside = false;
                    for (const Visit & other : cells) {
                        if (point_in_cell(other, e.x, e.y)) {
                            inside = true;
                            break;
                        }
                    }
                    if (inside) continue;
                    // The ghost state comes from the line's face nearest to the image
                    double best = std::numeric_limits<double>::max();
                    for (const LineFace & lf : line.faces) {
                        const double dx = lf.x - 0.5 * (e.x + cx);
                        const double dy = lf.y - 0.5 * (e.y + cy);
                        if (dx * dx + dy * dy < best) {
                            best = dx * dx + dy * dy;
                            e.face = lf.face;
                            e.mx = lf.x;
                            e.my = lf.y;
                        }
                    }
                    entries.push_back(e);
                }
            }
            auto dist2 = [&](const Entry & e) {
                return (e.x - x0) * (e.x - x0) + (e.y - y0) * (e.y - y0);
            };
            sort_by_key(entries, dist2);
            return entries;
        };

        // Least-squares system rows: mean of psi_l over each stencil entry
        auto build_pinv = [&](const std::vector<Entry> & stencil, uint8_t deg, std::vector<double> & P) {
            const uint8_t n = teno::n_dof(deg);
            const int m = stencil.size();
            std::vector<double> A(m * n);
            std::vector<double> means;
            for (int s = 0; s < m; s++) {
                monomial_means(stencil[s], deg, means);
                for (uint8_t l = 0; l < n; l++) A[s * n + l] = means[l] - mean0[l];
            }
            return pseudo_inverse(A, m, n, P, double(max_condition));
        };

        // Zero-mean basis at the cell's face quadrature points
        std::vector<double> psi_faces;
        for (uint32_t k = 0; k < mesh->h_n_faces_of_cell(i); k++) {
            const uint32_t f = mesh->h_face_of_cell(i, k);
            const uint32_t na = mesh->h_node_of_face(f, 0), nb = mesh->h_node_of_face(f, 1);
            for (size_t q = 0; q < quadrature_face.h_points.extent(0); q++) {
                const double s_q = 0.5 * double(quadrature_face.h_points(q, 0));
                double xi[2], phi[teno::MAX_NK];
                for (int d = 0; d < 2; d++) {
                    const double x = double(mesh->h_face_coords(f, d)) +
                                     s_q * (double(mesh->h_node_coords(nb, d)) - double(mesh->h_node_coords(na, d)));
                    xi[d] = ((x - double(mesh->h_face_offset(f, i, d))) - (d == 0 ? x0 : y0)) / h;
                }
                teno::monomials(r, xi[0], xi[1], phi);
                for (uint8_t l = 0; l < nk; l++) psi_faces.push_back(phi[l] - mean0[l]);
            }
        }

        // Large central stencil, grown until the least-squares system has full
        // rank (anisotropic cells can have too few distinct rows/columns) and
        // is well conditioned
        std::vector<Entry> candidates = gather(ns_max, 64);
        std::vector<double> P;
        bool ok = false;
        // Never split a group of equidistant candidates: keeps stencils
        // independent of cell numbering, so mirror-symmetric meshes give
        // mirror-symmetric reconstructions
        auto dist2 = [&](const Entry & e) { return (e.x - x0) * (e.x - x0) + (e.y - y0) * (e.y - y0); };
        auto splits_tie = [&](const std::vector<Entry> & list, size_t n) {
            return n < list.size() && std::abs(dist2(list[n]) - dist2(list[n - 1])) < GEOMETRY_TOL * h * h;
        };
        // The smallest stencil within MAX_LEBESGUE, else the best conditioned one
        uint16_t n_used = ns;
        double best_lebesgue = std::numeric_limits<double>::max();
        for (uint16_t n_try = ns; n_try <= std::min<size_t>(ns_max, candidates.size()); n_try++) {
            if (splits_tie(candidates, n_try)) continue;
            std::vector<Entry> stencil(candidates.begin(), candidates.begin() + n_try);
            std::vector<double> P_try;
            if (!build_pinv(stencil, r, P_try)) continue;
            const double lebesgue = lebesgue_constant(psi_faces, nk, P_try, n_try);
            if (lebesgue < best_lebesgue) {
                best_lebesgue = lebesgue;
                n_used = n_try;
                P = std::move(P_try);
                ok = true;
            }
            if (lebesgue <= MAX_LEBESGUE) break;
        }
        if (!ok) {
            // A stencil cut off by the halo is retried once the halo is deep enough
            if (!truncated) failed_large++;
        } else {
            out.large_pinv.resize(n_used * nk);
            for (uint16_t s = 0; s < n_used; s++) {
                out.large_cells.push_back(candidates[s].cell);
                out.large_faces.push_back(candidates[s].face);
                for (uint8_t l = 0; l < nk; l++) out.large_pinv[s * nk + l] = P[l * n_used + s];
            }
        }

        // Small sector stencils, one per face
        std::vector<Entry> wide = gather(8 * nss, 6);
        out.gather_depth = layers_used;
        const uint32_t n_faces = mesh->h_n_faces_of_cell(i);
        for (uint32_t k = 0; k < n_faces; k++) {
            const uint32_t f = mesh->h_face_of_cell(i, k);
            const uint32_t na = mesh->h_node_of_face(f, 0);
            const uint32_t nb = mesh->h_node_of_face(f, 1);
            // The face's nodes are its cell 0's
            const double sx = double(mesh->h_face_offset(f, i, 0)), sy = double(mesh->h_face_offset(f, i, 1));
            const double ax = (double(mesh->h_node_coords(na, 0)) - sx) - x0, ay = (double(mesh->h_node_coords(na, 1)) - sy) - y0;
            const double bx = (double(mesh->h_node_coords(nb, 0)) - sx) - x0, by = (double(mesh->h_node_coords(nb, 1)) - sy) - y0;
            const double det = ax * by - ay * bx;
            std::vector<Entry> sector;
            for (const Entry & e : wide) {
                const double dx = e.x - x0;
                const double dy = e.y - y0;
                const double alpha = (dx * by - dy * bx) / det;
                const double beta = (ax * dy - ay * dx) / det;
                if (alpha >= -GEOMETRY_TOL && beta >= -GEOMETRY_TOL) sector.push_back(e);
            }
            size_t n_sector = nss;
            while (n_sector < nss_max && splits_tie(sector, n_sector)) n_sector++;
            if (sector.size() < nss || splits_tie(sector, n_sector)) {
                invalid_small++;
                continue;
            }
            sector.resize(n_sector);
            if (!build_pinv(sector, 2, P)) {
                invalid_small++;
                continue;
            }
            out.small_size[k] = n_sector;
            for (uint16_t s = 0; s < n_sector; s++) {
                out.small_cells.push_back(sector[s].cell);
                out.small_faces.push_back(sector[s].face);
                for (uint8_t l = 0; l < teno::NK_SMALL; l++) out.small_pinv.push_back(P[l * n_sector + s]);
            }
        }

        // Smoothness-indicator matrix: M_lm = sum_{1<=|beta|<=r} int D^beta phi_l D^beta phi_m
        {
            const uint32_t n_nodes = mesh->h_n_nodes_of_cell(i);
            std::vector<double> px(n_nodes), py(n_nodes);
            for (uint32_t k = 0; k < n_nodes; k++) {
                const uint32_t node = mesh->h_node_of_cell(i, k);
                px[k] = (double(mesh->h_node_coords(node, 0)) - x0) / h;
                py[k] = (double(mesh->h_node_coords(node, 1)) - y0) / h;
            }
            std::vector<double> M(nk * nk, 0.0);
            std::vector<double> d(nk);
            auto falling = [](int a, int p) {
                double c = 1.0;
                for (int t = 0; t < p; t++) c *= (a - t);
                return c;
            };
            integrate_polygon(px, py, rule, [&](double x, double y, double w) {
                for (int bp = 0; bp <= r; bp++) {
                    for (int bq = 0; bp + bq <= r; bq++) {
                        if (bp + bq == 0) continue;
                        for (uint8_t l = 0; l < nk; l++) {
                            uint8_t a, b;
                            teno::exponents(l, a, b);
                            d[l] = (a >= bp && b >= bq)
                                       ? falling(a, bp) * falling(b, bq) * std::pow(x, a - bp) * std::pow(y, b - bq)
                                       : 0.0;
                        }
                        for (uint8_t l = 0; l < nk; l++) {
                            for (uint8_t m = 0; m < nk; m++) M[l * nk + m] += w * d[l] * d[m];
                        }
                    }
                }
            });
            for (uint8_t l = 0; l < nk; l++) {
                for (uint8_t m = l; m < nk; m++) out.si.push_back(M[l * nk + m]);
            }
        }
    };
    // Outer halo cells are never reconstructed; their neighborhoods are cut off
    largest_stencil = precompute_in_chunks(*this, mesh->n_reconstructed(), precompute, n_failed_large, n_invalid_small);

    if (n_failed_large > 0) {
        throw std::runtime_error("TENO: could not build a full-rank large stencil for " +
                                 std::to_string(n_failed_large) + " cells (mesh too small for this order?).");
    }
    n_sector_unavailable = n_invalid_small;
}

namespace {

/**
 * @brief Collapsed (Stroud conical) Gauss quadrature on the reference
 *        tetrahedron (0,0,0), (1,0,0), (0,1,0), (0,0,1); weights sum to 1/6.
 */
struct TetRule {
    std::vector<double> x, y, z, w;
    explicit TetRule(int n) {
        std::vector<double> g, gw;
        gauss_legendre(n, g, gw);
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                for (int k = 0; k < n; k++) {
                    const double u = 0.5 * (g[i] + 1.0), v = 0.5 * (g[j] + 1.0), s = 0.5 * (g[k] + 1.0);
                    x.push_back(u);
                    y.push_back(v * (1.0 - u));
                    z.push_back(s * (1.0 - u) * (1.0 - v));
                    w.push_back(0.125 * gw[i] * gw[j] * gw[k] * (1.0 - u) * (1.0 - u) * (1.0 - v));
                }
            }
        }
    }
};

using Point3 = std::array<double, 3>;

/**
 * @brief Integrate f(point, weight) over a cell given by its node coordinates
 *        (Gmsh/VTK order, either orientation) via its tetrahedral decomposition.
 */
template <typename F>
void integrate_cell(const std::vector<Point3> & nodes, const TetRule & rule, F && f) {
    std::vector<std::array<Point3, 4>> tets;
    cell_tetrahedra(nodes, tets);
    for (const auto & t : tets) {
        double e[3][3];
        for (int a = 0; a < 3; a++) {
            for (int d = 0; d < 3; d++) e[a][d] = t[a + 1][d] - t[0][d];
        }
        const double det = std::abs(e[0][0] * (e[1][1] * e[2][2] - e[1][2] * e[2][1]) -
                                    e[0][1] * (e[1][0] * e[2][2] - e[1][2] * e[2][0]) +
                                    e[0][2] * (e[1][0] * e[2][1] - e[1][1] * e[2][0]));
        for (size_t q = 0; q < rule.w.size(); q++) {
            Point3 p;
            for (int d = 0; d < 3; d++) p[d] = t[0][d] + rule.x[q] * e[0][d] + rule.y[q] * e[1][d] + rule.z[q] * e[2][d];
            f(p, rule.w[q] * det);
        }
    }
}

// Moments of total degree below 2 * MAX_DEGREE - 1
constexpr int MAX_MOMENTS = (2 * teno::MAX_DEGREE - 1) * (2 * teno::MAX_DEGREE) * (2 * teno::MAX_DEGREE + 1) / 6;

/** @brief Add w x^a y^b z^c, a + b + c < NM, to the moment sums m, ordered by a, then b, then c. */
template <int NM>
void add_moments(const double w, const Point3 & x, double * m) {
    double px[NM], py[NM], pz[NM];
    px[0] = py[0] = pz[0] = 1.0;
    for (int k = 1; k < NM; k++) {
        px[k] = px[k - 1] * x[0];
        py[k] = py[k - 1] * x[1];
        pz[k] = pz[k - 1] * x[2];
    }
    int k = 0;
    for (int a = 0; a < NM; a++) {
        const double wa = w * px[a];
        for (int b = 0; a + b < NM; b++) {
            const double wab = wa * py[b];
            for (int c = 0; a + b + c < NM; c++) m[k++] += wab * pz[c];
        }
    }
}

double det3(const Point3 & a, const Point3 & b, const Point3 & c) {
    return a[0] * (b[1] * c[2] - b[2] * c[1]) - a[1] * (b[0] * c[2] - b[2] * c[0]) + a[2] * (b[0] * c[1] - b[1] * c[0]);
}

} // namespace

void TENO::compute_stencils_and_matrices_3d() {
    const uint32_t n_cells = mesh->n_cells;
    const uint8_t r = degree;
    const uint8_t nk = n_dof_large;
    const uint16_t ns = n_stencil_large;
    const uint16_t nss = n_stencil_small;
    // Shells of equidistant candidates are large on 3D lattices; leave room to finish one
    const uint16_t ns_max = static_cast<uint16_t>(std::ceil(3.5 * nk)) + 64;
    const uint16_t nss_max = 2 * nss;

    for (uint32_t i = 0; i < n_cells; i++) {
        if (mesh->h_n_faces_of_cell(i) > teno::MAX_FACES) {
            throw std::runtime_error("TENO supports cells with at most " +
                                     std::to_string(teno::MAX_FACES) + " faces.");
        }
    }

    auto h_face_bc = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), boundaries.face_bc);
    auto h_bcs = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), boundaries.bcs);
    auto h_face_quad_points = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), face_quad_points);
    auto h_face_quad_weights = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), face_quad_weights);

    // Collapsed Gauss with n points per direction is exact to degree 2n - 3 on a tet
    const TetRule rule((r + 4) / 2);
    const TetRule rule_si(r + 1);
    uint32_t n_failed_large = 0;
    uint32_t n_invalid_small = 0;

    auto node = [&](uint32_t n) {
        Point3 p;
        for (int d = 0; d < 3; d++) p[d] = double(mesh->h_node_coords(n, d));
        return p;
    };
    auto unit_normal = [&](uint32_t f) {
        Point3 n;
        for (int d = 0; d < 3; d++) n[d] = double(mesh->h_face_normals(f, d)) / double(mesh->h_face_area(f));
        return n;
    };

    // Central moments of every cell in its own scaled frame, mean of
    // ((x - x_c) / h_c)^a ((y - y_c) / h_c)^b ((z - z_c) / h_c)^c, so that
    // monomial means over unmirrored stencil entries follow by binomial
    // expansion instead of quadrature
    // Only total degrees below nm are stored
    const int nm = 2 * r - 1;
    std::vector<int> moment_slot(nm * nm * nm, -1);
    int n_moments = 0;
    for (int a = 0; a < nm; a++) {
        for (int b = 0; a + b < nm; b++) {
            for (int c = 0; a + b + c < nm; c++) moment_slot[(a * nm + b) * nm + c] = n_moments++;
        }
    }
    auto moment = [&](int a, int b, int c) { return moment_slot[(a * nm + b) * nm + c]; };
    std::vector<double> moments(static_cast<size_t>(n_cells) * n_moments, 0.0);
    Kokkos::parallel_for("teno_moments", Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(0, n_cells),
                         [&](const uint32_t c) {
        Point3 xc;
        for (int d = 0; d < 3; d++) xc[d] = double(mesh->h_cell_coords(c, d));
        const double hc = std::cbrt(double(mesh->h_cell_volume(c)));
        std::vector<Point3> p(mesh->h_n_nodes_of_cell(c));
        for (size_t k = 0; k < p.size(); k++) {
            const Point3 q = node(mesh->h_node_of_cell(c, k));
            for (int d = 0; d < 3; d++) p[k][d] = (q[d] - xc[d]) / hc;
        }
        double sums[MAX_MOMENTS] = {};
        double vol = 0.0;
        auto integrate = [&](auto nm_c) {
            integrate_cell(p, rule_si, [&](const Point3 & x, double w) {
                add_moments<decltype(nm_c)::value>(w, x, sums);
                vol += w;
            });
        };
        switch (nm) {
            case 3: integrate(std::integral_constant<int, 3>()); break;
            case 5: integrate(std::integral_constant<int, 5>()); break;
            case 7: integrate(std::integral_constant<int, 7>()); break;
            case 9: integrate(std::integral_constant<int, 9>()); break;
            default: break;
        }
        double * m = &moments[static_cast<size_t>(c) * n_moments];
        for (int k = 0; k < n_moments; k++) m[k] = sums[k] / vol;
    });
    std::vector<std::array<uint8_t, 3>> expo_all(nk);
    for (uint8_t l = 0; l < nk; l++) teno::exponents(l, expo_all[l][0], expo_all[l][1], expo_all[l][2]);
    double binom[12][12] = {};
    for (int n = 0; n < 12; n++) {
        binom[n][0] = 1.0;
        for (int k = 1; k <= n; k++) binom[n][k] = binom[n - 1][k - 1] + (k < n ? binom[n - 1][k] : 0.0);
    }

    auto precompute = [&](const uint32_t i, CellTables & out, uint32_t & failed_large, uint32_t & invalid_small) {
        Point3 x0;
        for (int d = 0; d < 3; d++) x0[d] = double(mesh->h_cell_coords(i, d));
        const double h = std::cbrt(double(mesh->h_cell_volume(i)));
        out.scale = h;

        // Stencil entries: interior cells, or mirror images of interior cells
        // across a planar boundary (face >= 0) carrying the boundary
        // condition's ghost state. Across periodic boundaries, cells are taken
        // at the translation t of their lattice offset; m is a point of the
        // mirror plane, from the copy of the face at lattice offset face_lattice.
        struct Entry {
            uint32_t cell;
            int32_t face;
            Point3 x;
            Lattice lattice;
            Point3 t;
            Lattice face_lattice;
            Point3 m;
        };
        auto mirror = [&](int32_t face, const Point3 & m, const Point3 & p) {
            if (face < 0) return p;
            const Point3 n = unit_normal(face);
            double d = 0.0;
            for (int k = 0; k < 3; k++) d += (p[k] - m[k]) * n[k];
            Point3 q;
            for (int k = 0; k < 3; k++) q[k] = p[k] - 2.0 * d * n[k];
            return q;
        };
        auto scaled_nodes = [&](const Entry & e) {
            std::vector<Point3> p(mesh->h_n_nodes_of_cell(e.cell));
            for (size_t k = 0; k < p.size(); k++) {
                Point3 x = node(mesh->h_node_of_cell(e.cell, k));
                for (int d = 0; d < 3; d++) x[d] += e.t[d];
                const Point3 q = mirror(e.face, e.m, x);
                for (int d = 0; d < 3; d++) p[k][d] = (q[d] - x0[d]) / h;
            }
            return p;
        };

        auto monomial_means = [&](const Entry & e, uint8_t deg, std::vector<double> & means) {
            const uint8_t n = teno::n_dof(deg);
            means.assign(n, 0.0);
            if (e.face < 0) {
                // ((x - x0) / h)^a = (d + s xi)^a with d = (x_c - x0) / h, s = h_c / h
                const double s = std::cbrt(double(mesh->h_cell_volume(e.cell))) / h;
                double dpow[3][12], spow[12];
                spow[0] = 1.0;
                for (int k = 1; k < nm; k++) spow[k] = spow[k - 1] * s;
                for (int d = 0; d < 3; d++) {
                    dpow[d][0] = 1.0;
                    const double dd = ((double(mesh->h_cell_coords(e.cell, d)) + e.t[d]) - x0[d]) / h;
                    for (int k = 1; k < nm; k++) dpow[d][k] = dpow[d][k - 1] * dd;
                }
                const double * m = &moments[static_cast<size_t>(e.cell) * n_moments];
                for (uint8_t l = 0; l < n; l++) {
                    const auto & ex = expo_all[l];
                    double sum = 0.0;
                    for (int ka = 0; ka <= ex[0]; ka++) {
                        const double ta = binom[ex[0]][ka] * dpow[0][ex[0] - ka];
                        for (int kb = 0; kb <= ex[1]; kb++) {
                            const double tb = ta * binom[ex[1]][kb] * dpow[1][ex[1] - kb];
                            for (int kc = 0; kc <= ex[2]; kc++) {
                                sum += tb * binom[ex[2]][kc] * dpow[2][ex[2] - kc] * spow[ka + kb + kc] *
                                       m[moment(ka, kb, kc)];
                            }
                        }
                    }
                    means[l] = sum;
                }
                return;
            }
            double vol = 0.0;
            double phi[teno::MAX_NK];
            integrate_cell(scaled_nodes(e), rule, [&](const Point3 & p, double w) {
                // teno::monomials from tables of powers, which repeat its products
                double px[teno::MAX_DEGREE + 1], py[teno::MAX_DEGREE + 1], pz[teno::MAX_DEGREE + 1];
                px[0] = py[0] = pz[0] = 1.0;
                for (uint8_t k = 1; k <= deg; k++) {
                    px[k] = px[k - 1] * p[0];
                    py[k] = py[k - 1] * p[1];
                    pz[k] = pz[k - 1] * p[2];
                }
                for (uint8_t l = 0; l < n; l++) phi[l] = px[expo_all[l][0]] * py[expo_all[l][1]] * pz[expo_all[l][2]];
                for (uint8_t l = 0; l < n; l++) means[l] += w * phi[l];
                vol += w;
            });
            for (uint8_t l = 0; l < n; l++) means[l] /= vol;
        };

        const Lattice zero = {0, 0, 0};
        const Point3 origin = {0.0, 0.0, 0.0};
        std::vector<double> mean0;
        monomial_means(Entry{i, -1, x0, zero, origin, zero, origin}, r, mean0);
        out.basis_mean.assign(mean0.begin(), mean0.end());

        // Whether a point lies strictly inside the copy of one of the cells
        // translated by its t (points on a face count as outside), from the
        // cells' face planes: centroids in the cell's frame and outward area
        // vectors, tabulated once per search
        struct FacePlane {
            Point3 x, n;
            double area;
        };
        std::vector<FacePlane> face_planes;
        std::vector<uint32_t> planes_begin;
        auto tabulate_face_planes = [&](const std::vector<Visit> & cells) {
            face_planes.clear();
            planes_begin.assign(1, 0);
            for (const Visit & v : cells) {
                const uint32_t c = v.cell;
                for (uint32_t k = 0; k < mesh->h_n_faces_of_cell(c); k++) {
                    const uint32_t f = mesh->h_face_of_cell(c, k);
                    const double sign = (mesh->h_cells_of_face(f, 0) == static_cast<int32_t>(c)) ? 1.0 : -1.0;
                    FacePlane fp;
                    for (int q = 0; q < 3; q++) {
                        fp.x[q] = double(mesh->h_face_coords(f, q)) - double(mesh->h_face_offset(f, c, q));
                        fp.n[q] = sign * double(mesh->h_face_normals(f, q));
                    }
                    fp.area = double(mesh->h_face_area(f));
                    face_planes.push_back(fp);
                }
                planes_begin.push_back(face_planes.size());
            }
        };
        const double inside_tol = -precision_tol<double>(1e-12, 1e-5) * h;
        auto inside_any = [&](const std::vector<Visit> & cells, const Point3 & p) {
            for (size_t c = 0; c < cells.size(); c++) {
                Point3 q;
                for (int a = 0; a < 3; a++) q[a] = p[a] - cells[c].t[a];
                bool inside = true;
                for (uint32_t k = planes_begin[c]; k < planes_begin[c + 1] && inside; k++) {
                    const FacePlane & fp = face_planes[k];
                    double d = 0.0;
                    for (int a = 0; a < 3; a++) d += (q[a] - fp.x[a]) * fp.n[a];
                    inside = !(d > inside_tol * fp.area);
                }
                if (inside) return true;
            }
            return false;
        };

        // On thin cells, physical distance takes the whole wall-normal column
        // first and leaves the other directions to small centroid offsets: a
        // fit whose face values amplify grid-scale vortical modes
        const std::array<double, 9> metric = ranking_metric(*mesh, h_face_bc, h_bcs, i);
        auto dist2 = [&](const Entry & e) {
            double s = 0.0;
            for (int a = 0; a < 3; a++) {
                for (int b = 0; b < 3; b++) s += (e.x[a] - x0[a]) * metric[3 * a + b] * (e.x[b] - x0[b]);
            }
            return s;
        };

        // Candidates by vertex-neighbor layers plus their mirror images across
        // nearby boundary planes, sorted by distance
        int layers_used = 0;
        bool truncated = false;
        Layers layers(*mesh, i);
        int entries_depth = 0;
        std::vector<Entry> entries;
        auto gather = [&](size_t n_min, int max_layers) {
            const int depth = layers.reach(n_min, max_layers, truncated);
            layers_used = std::max(layers_used, depth);
            if (depth == entries_depth) return entries;
            entries_depth = depth;
            entries.clear();
            const std::vector<Visit> cells(layers.cells.begin(), layers.cells.begin() + layers.size(depth));
            // Planar boundaries touched by the gathered cells; each image takes
            // its state from the plane's face nearest to it
            struct PlaneFace {
                int32_t face;
                Lattice lattice;
                Point3 x;  // centroid, translated with its cell
            };
            struct Plane {
                Point3 n;
                std::vector<PlaneFace> faces;
            };
            std::vector<Plane> planes;
            for (const Visit & v : cells) {
                const uint32_t c = v.cell;
                for (uint32_t k = 0; k < mesh->h_n_faces_of_cell(c); k++) {
                    const uint32_t f = mesh->h_face_of_cell(c, k);
                    if (mesh->h_cells_of_face(f, 1) >= 0 || h_face_bc(f) < 0) continue;
                    if (h_bcs(h_face_bc(f)).type == BoundaryType::PARTITION) continue;
                    const Point3 n = unit_normal(f);
                    PlaneFace pf{static_cast<int32_t>(f), v.lattice, {}};
                    for (int d = 0; d < 3; d++) pf.x[d] = double(mesh->h_face_coords(f, d)) + v.t[d];
                    Plane * match = nullptr;
                    for (Plane & plane : planes) {
                        const PlaneFace & g = plane.faces[0];
                        double off = 0.0, cos = 0.0;
                        for (int d = 0; d < 3; d++) {
                            off += (pf.x[d] - g.x[d]) * plane.n[d];
                            cos += n[d] * plane.n[d];
                        }
                        if (std::abs(cos - 1.0) < GEOMETRY_TOL && std::abs(off) < GEOMETRY_TOL * h) {
                            match = &plane;
                            break;
                        }
                    }
                    if (match == nullptr) {
                        planes.push_back(Plane{n, {}});
                        match = &planes.back();
                    }
                    const bool known = std::any_of(match->faces.begin(), match->faces.end(), [&](const PlaneFace & g) {
                        return g.face == pf.face && g.lattice == pf.lattice;
                    });
                    if (!known) match->faces.push_back(pf);
                }
            }
            if (!planes.empty()) tabulate_face_planes(cells);
            for (const Visit & v : cells) {
                const uint32_t c = v.cell;
                Point3 xc;
                for (int d = 0; d < 3; d++) xc[d] = double(mesh->h_cell_coords(c, d)) + v.t[d];
                if (!(c == i && v.lattice == zero)) entries.push_back(Entry{c, -1, xc, v.lattice, v.t, zero, origin});
                for (const Plane & plane : planes) {
                    const PlaneFace & first = plane.faces[0];
                    Entry e{c, first.face, mirror(first.face, first.x, xc), v.lattice, v.t, first.lattice, first.x};
                    if (inside_any(cells, e.x)) continue;
                    double best = std::numeric_limits<double>::max();
                    for (const PlaneFace & pf : plane.faces) {
                        double d2 = 0.0;
                        for (int d = 0; d < 3; d++) d2 += std::pow(pf.x[d] - 0.5 * (e.x[d] + xc[d]), 2);
                        if (d2 < best) {
                            best = d2;
                            e.face = pf.face;
                            e.face_lattice = pf.lattice;
                            e.m = pf.x;
                        }
                    }
                    entries.push_back(e);
                }
            }
            sort_by_key(entries, dist2);
            return entries;
        };

        // Means of all degree-r monomials per entry; lower degrees are a prefix
        using MeansKey = std::tuple<uint32_t, int32_t, Lattice, Lattice>;
        struct MeansHash {
            size_t operator()(const MeansKey & key) const {
                uint64_t x = std::get<0>(key) ^ (uint64_t(uint32_t(std::get<1>(key))) << 32);
                for (int a = 0; a < 3; a++) {
                    x = x * 0x9E3779B97F4A7C15ULL + uint32_t(std::get<2>(key)[a]);
                    x = x * 0x9E3779B97F4A7C15ULL + uint32_t(std::get<3>(key)[a]);
                }
                return x ^ (x >> 29);
            }
        };
        std::unordered_map<MeansKey, std::vector<double>, MeansHash> means_cache;
        auto build_pinv = [&](const std::vector<Entry> & stencil, uint8_t deg, std::vector<double> & P) {
            const uint8_t n = teno::n_dof(deg);
            const int m = stencil.size();
            std::vector<double> A(m * n);
            for (int s = 0; s < m; s++) {
                const Entry & e = stencil[s];
                const MeansKey key(e.cell, e.face, e.lattice, e.face_lattice);
                auto it = means_cache.find(key);
                if (it == means_cache.end()) {
                    std::vector<double> means;
                    monomial_means(stencil[s], r, means);
                    it = means_cache.emplace(key, std::move(means)).first;
                }
                for (uint8_t l = 0; l < n; l++) A[s * n + l] = it->second[l] - mean0[l];
            }
            return pseudo_inverse(A, m, n, P, double(max_condition));
        };
        auto splits_tie = [&](const std::vector<Entry> & list, size_t n) {
            return n < list.size() && std::abs(dist2(list[n]) - dist2(list[n - 1])) < GEOMETRY_TOL * h * h;
        };

        // Zero-mean basis at the cell's face quadrature points
        std::vector<double> psi_faces;
        for (uint32_t k = 0; k < mesh->h_n_faces_of_cell(i); k++) {
            const uint32_t f = mesh->h_face_of_cell(i, k);
            for (size_t q = 0; q < h_face_quad_weights.extent(1); q++) {
                if (h_face_quad_weights(f, q) == 0.0_r) continue;
                double xi[3], phi[teno::MAX_NK];
                for (int d = 0; d < 3; d++) {
                    xi[d] = ((double(h_face_quad_points(f, q, d)) - double(mesh->h_face_offset(f, i, d))) - x0[d]) / h;
                }
                teno::monomials(r, xi[0], xi[1], xi[2], phi);
                for (uint8_t l = 0; l < nk; l++) psi_faces.push_back(phi[l] - mean0[l]);
            }
        }

        // Large central stencil, grown until the least-squares system has full
        // rank and is well conditioned
        std::vector<Entry> candidates = gather(ns_max, 64);
        std::vector<double> P;
        bool ok = false;
        // The smallest stencil within MAX_LEBESGUE, else the best conditioned one
        uint16_t n_used = ns;
        double best_lebesgue = std::numeric_limits<double>::max();
        for (uint16_t n_try = ns; n_try <= std::min<size_t>(ns_max, candidates.size()); n_try++) {
            if (splits_tie(candidates, n_try)) continue;
            std::vector<Entry> stencil(candidates.begin(), candidates.begin() + n_try);
            std::vector<double> P_try;
            if (!build_pinv(stencil, r, P_try)) continue;
            const double lebesgue = lebesgue_constant(psi_faces, nk, P_try, n_try);
            if (lebesgue < best_lebesgue) {
                best_lebesgue = lebesgue;
                n_used = n_try;
                P = std::move(P_try);
                ok = true;
            }
            if (lebesgue <= MAX_LEBESGUE) break;
        }
        if (!ok) {
            // A stencil cut off by the halo is retried once the halo is deep enough
            if (!truncated) failed_large++;
        } else {
            out.large_pinv.resize(n_used * nk);
            for (uint16_t s = 0; s < n_used; s++) {
                out.large_cells.push_back(candidates[s].cell);
                out.large_faces.push_back(candidates[s].face);
                for (uint8_t l = 0; l < nk; l++) out.large_pinv[s * nk + l] = P[l * n_used + s];
            }
        }

        // Small sector stencils, one per face: entries whose direction from the
        // centroid lies in the cone spanned by the face's vertices
        std::vector<Entry> wide = gather(8 * nss, 6);
        out.gather_depth = layers_used;
        const uint32_t n_faces = mesh->h_n_faces_of_cell(i);
        for (uint32_t k = 0; k < n_faces; k++) {
            const uint32_t f = mesh->h_face_of_cell(i, k);
            // The face's nodes are its cell 0's
            std::vector<Point3> v(mesh->h_n_nodes_of_face(f));
            for (size_t a = 0; a < v.size(); a++) {
                const Point3 p = node(mesh->h_node_of_face(f, a));
                for (int d = 0; d < 3; d++) v[a][d] = (p[d] - double(mesh->h_face_offset(f, i, d))) - x0[d];
            }
            std::vector<double> dets(v.size());
            for (size_t a = 1; a + 1 < v.size(); a++) dets[a] = det3(v[0], v[a], v[a + 1]);
            std::vector<Entry> sector;
            for (const Entry & e : wide) {
                Point3 dx;
                for (int d = 0; d < 3; d++) dx[d] = e.x[d] - x0[d];
                bool in = false;
                for (size_t a = 1; a + 1 < v.size() && !in; a++) {
                    const double det = dets[a];
                    const double alpha = det3(dx, v[a], v[a + 1]) / det;
                    const double beta = det3(v[0], dx, v[a + 1]) / det;
                    const double gamma = det3(v[0], v[a], dx) / det;
                    in = alpha >= -GEOMETRY_TOL && beta >= -GEOMETRY_TOL && gamma >= -GEOMETRY_TOL;
                }
                if (in) sector.push_back(e);
            }
            size_t n_sector = nss;
            while (n_sector < nss_max && splits_tie(sector, n_sector)) n_sector++;
            if (sector.size() < nss || splits_tie(sector, n_sector)) {
                invalid_small++;
                continue;
            }
            sector.resize(n_sector);
            if (!build_pinv(sector, 2, P)) {
                invalid_small++;
                continue;
            }
            out.small_size[k] = n_sector;
            for (uint16_t s = 0; s < n_sector; s++) {
                out.small_cells.push_back(sector[s].cell);
                out.small_faces.push_back(sector[s].face);
                for (uint8_t l = 0; l < teno::NK_SMALL; l++) out.small_pinv.push_back(P[l * n_sector + s]);
            }
        }

        // Smoothness-indicator matrix: M_lm = sum_{1<=|beta|<=r} int D^beta phi_l D^beta phi_m
        {
            std::vector<double> M(nk * nk, 0.0);
            std::vector<double> d(nk);
            std::vector<std::array<uint8_t, 3>> expo(nk);
            for (uint8_t l = 0; l < nk; l++) teno::exponents(l, expo[l][0], expo[l][1], expo[l][2]);
            auto falling = [](int a, int p) {
                double c = 1.0;
                for (int t = 0; t < p; t++) c *= (a - t);
                return c;
            };
            // Integrals of monomial products over the cell are its central moments
            // (the scaled volume is 1)
            const double * mom = &moments[static_cast<size_t>(i) * n_moments];
            for (int b0 = 0; b0 <= r; b0++) {
                for (int b1 = 0; b0 + b1 <= r; b1++) {
                    for (int b2 = 0; b0 + b1 + b2 <= r; b2++) {
                        if (b0 + b1 + b2 == 0) continue;
                        for (uint8_t l = 0; l < nk; l++) {
                            const auto & e = expo[l];
                            d[l] = (e[0] >= b0 && e[1] >= b1 && e[2] >= b2)
                                       ? falling(e[0], b0) * falling(e[1], b1) * falling(e[2], b2)
                                       : 0.0;
                        }
                        for (uint8_t l = 0; l < nk; l++) {
                            if (d[l] == 0.0) continue;
                            for (uint8_t m = 0; m < nk; m++) {
                                if (d[m] == 0.0) continue;
                                const auto & el = expo[l];
                                const auto & em = expo[m];
                                M[l * nk + m] += d[l] * d[m] *
                                                 mom[moment(el[0] + em[0] - 2 * b0, el[1] + em[1] - 2 * b1,
                                                            el[2] + em[2] - 2 * b2)];
                            }
                        }
                    }
                }
            }
            for (uint8_t l = 0; l < nk; l++) {
                for (uint8_t m = l; m < nk; m++) out.si.push_back(M[l * nk + m]);
            }
        }
    };
    // Outer halo cells are never reconstructed; their neighborhoods are cut off
    largest_stencil = precompute_in_chunks(*this, mesh->n_reconstructed(), precompute, n_failed_large, n_invalid_small);

    if (n_failed_large > 0) {
        throw std::runtime_error("TENO: could not build a full-rank large stencil for " +
                                 std::to_string(n_failed_large) + " cells (mesh too small for this order?).");
    }
    n_sector_unavailable = n_invalid_small;
}

/**
 * @brief Per-cell TENO-E reconstruction of degree DEG to the quadrature points
 *        of all of the cell's faces. SmoothPass finishes the cells below the
 *        troubled threshold and queues the others with their central
 *        coefficients. The troubled passes then work on that queue only, so
 *        the register-heavy path does not slow the smooth cells, and split it
 *        finely, so that the few troubled cells along shocks still fill the
 *        device: sector-stencil coefficients per (cell, sector), stencil
 *        selection per (cell, face, characteristic variable), the projection
 *        back to conservative variables per (cell, face), and admissibility
 *        per cell. Each pass loops over its share of the queue, whose length
 *        never has to be read back to the host.
 */
template <uint8_t DEG, bool PRIM>
struct TENOFunctor {
    static constexpr uint8_t NK = teno::n_dof(DEG);
    struct SmoothPass {};
    struct TroubledSectorPass {};
    struct TroubledSelectPass {};
    struct TroubledProjectPass {};
    struct TroubledFinishPass {};
    struct GradientPass {};

    rtype sigma_threshold;
    rtype sigma_upper;
    rtype C_T;
    bool characteristic;
    bool bound_preserving;
    rtype gamma;

    Kokkos::View<uint32_t *> offsets_faces_of_cell;
    Kokkos::View<uint32_t *> faces_of_cell;
    Kokkos::View<int32_t *[2]> cells_of_face;
    Kokkos::View<uint32_t *> offsets_nodes_of_face;
    Kokkos::View<uint32_t *> nodes_of_face;
    Kokkos::View<rtype *[N_DIM]> node_coords;
    Kokkos::View<rtype *[N_DIM]> cell_coords;
    Kokkos::View<rtype *[N_DIM]> face_coords;
    Kokkos::View<rtype *[N_DIM]> face_normals;
    Kokkos::View<rtype *[N_DIM]> shifts;
    Kokkos::View<uint8_t *> face_shift;
    Kokkos::View<rtype **> quad_points;
    Kokkos::View<rtype ***> face_quad_points;   // 3D: (face, q, dim)
    Kokkos::View<rtype **> face_quad_weights;   // 3D: (face, q), zero on padding points
    BoundaryData boundaries;

    Kokkos::View<rtype *> scale;
    Kokkos::View<rtype **> basis_mean;
    Kokkos::View<uint16_t *> stencil_large_size;
    teno::PackedStencils stencil_large;
    Kokkos::View<uint16_t **> stencil_small_size;
    teno::PackedStencils stencil_small;
    Kokkos::View<rtype **> si_matrix;
    Kokkos::View<rtype *> sigma_out;
    Kokkos::View<rtype ***> coeffs;           // (cell, l, var): central coefficients of queued cells
    Kokkos::View<rtype ****> small_coeffs;    // (cell, sector, l, var): sector coefficients of queued cells
    Kokkos::View<uint32_t *> troubled_cells;  // queue of troubled cells
    Kokkos::View<uint32_t> n_troubled;

    Kokkos::View<rtype *[N_CONSERVATIVE]> W;
    Kokkos::View<rtype **[2][N_CONSERVATIVE]> face_solution;
    Kokkos::View<uint32_t *> cells;  // cells to reconstruct; empty for [0, n)
    Kokkos::View<rtype *[N_CONSERVATIVE][N_DIM]> gradients;  // GradientPass output
    // PRIM (gas mixtures): reconstruct W itself, see TENO::set_mixture
    Kokkos::View<rtype *, Kokkos::LayoutStride> cell_gamma;
    Kokkos::View<rtype *> cell_molar_mass;
    Kokkos::View<uint8_t **> selection;

    /**
     * @brief State of a stencil entry: cell c, or its mirror across boundary face f.
     */
    KOKKOS_INLINE_FUNCTION
    void entry_W(const int32_t c, const int32_t f, rtype * W_e) const {
        FOR_I_CONSERVATIVE W_e[i] = W(c, i);
        if (f >= 0) {
            rtype n[N_DIM], n_vec[N_DIM];
            FOR_I_DIM n_vec[i] = face_normals(f, i);
            unit<N_DIM>(n_vec, n);
            rtype W_c[N_CONSERVATIVE];
            FOR_I_CONSERVATIVE W_c[i] = W_e[i];
            rtype d = 0.0;
            FOR_I_DIM d += (face_coords(f, i) - cell_coords(c, i)) * n[i];
            boundaries.ghost_W_at(f, W_c, n, 2.0_r * Kokkos::fabs(d), W_e);
            const BoundaryCondition & bc = boundaries.bcs(boundaries.face_bc(f));
            if (bc.type == BoundaryType::FARFIELD) {
                // The characteristic state holds at the face; outside lies the free stream
                FOR_I_CONSERVATIVE W_e[i] = bc.data[i];
            }
        }
    }

    KOKKOS_INLINE_FUNCTION
    void entry_conservatives(const int32_t c, const int32_t f, rtype * U) const {
        if constexpr (PRIM) {
            entry_W(c, f, U);
        } else {
            rtype W_e[N_CONSERVATIVE];
            entry_W(c, f, W_e);
            U[0] = W_e[0];
            FOR_I_DIM U[1 + i] = W_e[0] * W_e[1 + i];
            U[N_DIM + 1] = W_e[N_DIM + 1] / (gamma - 1.0_r) + 0.5_r * W_e[0] * dot<N_DIM>(W_e + 1, W_e + 1);
        }
    }

    KOKKOS_INLINE_FUNCTION
    void conservatives(const int32_t c, rtype * U) const {
        if constexpr (PRIM) {
            FOR_I_CONSERVATIVE U[i] = W(c, i);
        } else {
            rtype W_c[N_CONSERVATIVE];
            FOR_I_CONSERVATIVE W_c[i] = W(c, i);
            U[0] = W_c[0];
            FOR_I_DIM U[1 + i] = W_c[0] * W_c[1 + i];
            U[N_DIM + 1] = W_c[N_DIM + 1] / (gamma - 1.0_r) + 0.5_r * W_c[0] * dot<N_DIM>(W_c + 1, W_c + 1);
        }
    }

    KOKKOS_INLINE_FUNCTION
    void to_primitives(const rtype * U, rtype * Wq) const {
        if constexpr (PRIM) {
            FOR_I_CONSERVATIVE Wq[i] = U[i];
        } else {
            Wq[0] = U[0];
            FOR_I_DIM Wq[1 + i] = U[1 + i] / U[0];
            Wq[N_DIM + 1] = (gamma - 1.0_r) * (U[N_DIM + 1] - 0.5_r * U[0] * dot<N_DIM>(Wq + 1, Wq + 1));
        }
    }

    KOKKOS_INLINE_FUNCTION
    uint8_t n_quad() const {
        if constexpr (N_DIM == 2) {
            return quad_points.extent(0);
        } else {
            return face_quad_weights.extent(1);
        }
    }

    /**
     * @brief Monomials (minus their cell means) at quadrature point q of face f,
     *        in the scaled frame of cell i_cell. False for 3D padding points.
     */
    KOKKOS_INLINE_FUNCTION
    bool face_point_basis(const uint32_t f, const uint8_t q, const uint32_t i_cell, rtype * psi) const {
        const rtype h = scale(i_cell);
        // The face's points are in its cell 0's frame
        const uint8_t s = (cells_of_face(f, 1) == static_cast<int32_t>(i_cell)) ? face_shift(f) : 0;
        if constexpr (N_DIM == 2) {
            const rtype xc = cell_coords(i_cell, 0), yc = cell_coords(i_cell, 1);
            const uint32_t node_0 = nodes_of_face(offsets_nodes_of_face(f));
            const uint32_t node_1 = nodes_of_face(offsets_nodes_of_face(f) + 1);
            const rtype s_q = 0.5_r * quad_points(q, 0);
            const rtype x = face_coords(f, 0) + s_q * (node_coords(node_1, 0) - node_coords(node_0, 0));
            const rtype y = face_coords(f, 1) + s_q * (node_coords(node_1, 1) - node_coords(node_0, 1));
            teno::monomials(DEG, ((x - shifts(s, 0)) - xc) / h, ((y - shifts(s, 1)) - yc) / h, psi);
        } else {
            if (face_quad_weights(f, q) == 0.0_r) return false;
            rtype xi[N_DIM];
            FOR_I_DIM xi[i] = ((face_quad_points(f, q, i) - shifts(s, i)) - cell_coords(i_cell, i)) / h;
            teno::monomials(DEG, xi, psi);
        }
        for (uint8_t l = 0; l < NK; l++) psi[l] -= basis_mean(i_cell, l);
        return true;
    }

    /**
     * @brief Gradients of W at the centroid from the central (large-stencil)
     *        polynomial of the conservative variables, whose linear monomials
     *        carry the first derivatives at the centroid.
     */
    KOKKOS_INLINE_FUNCTION
    void operator()(GradientPass, const uint32_t i_cell) const {
        rtype U0[N_CONSERVATIVE];
        conservatives(i_cell, U0);
        rtype dU[N_CONSERVATIVE][N_DIM] = {};
        const rtype inv_h = 1.0_r / scale(i_cell);
        const teno::PackedStencils::Row stencil = stencil_large.row(i_cell);
        for (uint16_t s = 0; s < stencil_large_size(i_cell); s++) {
            rtype U[N_CONSERVATIVE];
            entry_conservatives(stencil.cell(s), stencil.face(s), U);
            FOR_I_DIM {
                const rtype P = stencil.pinv<NK>(s, i) * inv_h;
                for (uint8_t v = 0; v < N_CONSERVATIVE; v++) dU[v][i] += P * (U[v] - U0[v]);
            }
        }
        if constexpr (PRIM) {
            FOR_I_CONSERVATIVE for (uint8_t d = 0; d < N_DIM; d++) gradients(i_cell, i, d) = dU[i][d];
        } else {
            const rtype rho = W(i_cell, 0);
            rtype u[N_DIM];
            FOR_I_DIM u[i] = W(i_cell, 1 + i);
            FOR_I_DIM {
                gradients(i_cell, 0, i) = dU[0][i];
                rtype work = dU[N_DIM + 1][i] - 0.5_r * dot<N_DIM>(u, u) * dU[0][i];
                for (uint8_t k = 0; k < N_DIM; k++) {
                    const rtype du = (dU[1 + k][i] - u[k] * dU[0][i]) / rho;
                    gradients(i_cell, 1 + k, i) = du;
                    work -= rho * u[k] * du;
                }
                gradients(i_cell, N_DIM + 1, i) = (gamma - 1.0_r) * work;
            }
        }
    }

    KOKKOS_INLINE_FUNCTION
    void operator()(SmoothPass, const uint32_t idx) const {
        const uint32_t i_cell = cells.extent(0) ? cells(idx) : idx;
        rtype U0[N_CONSERVATIVE], W0[N_CONSERVATIVE];
        conservatives(i_cell, U0);
        FOR_I_CONSERVATIVE W0[i] = W(i_cell, i);

        const uint16_t ns = stencil_large_size(i_cell);
        // One pass over the central stencil: large-stencil coefficients (conservative
        // variables) and the troubled-cell measure, the variance of the relative
        // density jumps (Welford's update)
        rtype aK[NK][N_CONSERVATIVE] = {};
        rtype g_mean = 0.0, g_m2 = 0.0;
        // Mixtures: also the relative jumps of the molar mass, which show
        // species interfaces at constant density
        rtype m_mean = 0.0, m_m2 = 0.0;
        const teno::PackedStencils::Row stencil = stencil_large.row(i_cell);
        for (uint16_t s = 0; s < ns; s++) {
            rtype U[N_CONSERVATIVE];
            entry_conservatives(stencil.cell(s), stencil.face(s), U);
            const rtype g = Kokkos::fabs(U[0] - W0[0]) / W0[0];
            const rtype delta = g - g_mean;
            g_mean += delta / (s + 1);
            g_m2 += delta * (g - g_mean);
            if constexpr (PRIM) {
                const rtype M0 = cell_molar_mass(i_cell);
                const rtype m = Kokkos::fabs(cell_molar_mass(stencil.cell(s)) - M0) / M0;
                const rtype delta_m = m - m_mean;
                m_mean += delta_m / (s + 1);
                m_m2 += delta_m * (m - m_mean);
            }
            for (uint8_t l = 0; l < NK; l++) {
                const rtype P = stencil.pinv<NK>(s, l);
                FOR_I_CONSERVATIVE aK[l][i] += P * (U[i] - U0[i]);
            }
        }
        const rtype sigma = PRIM ? Kokkos::fmax(g_m2, m_m2) / ns : g_m2 / ns;
        sigma_out(i_cell) = sigma;
        if (sigma >= sigma_threshold) {
            for (uint8_t l = 0; l < NK; l++) {
                FOR_I_CONSERVATIVE coeffs(i_cell, l, i) = aK[l][i];
            }
            troubled_cells(Kokkos::atomic_fetch_add(&n_troubled(), 1u)) = i_cell;
            return;
        }

        const uint8_t n_quad = this->n_quad();
        const uint32_t f_begin = offsets_faces_of_cell(i_cell);
        const uint8_t n_faces = offsets_faces_of_cell(i_cell + 1) - f_begin;
        bool admissible = true;
        for (uint8_t k = 0; k < n_faces; k++) {
            const uint32_t f = faces_of_cell(f_begin + k);
            const uint8_t side = (cells_of_face(f, 0) == static_cast<int32_t>(i_cell)) ? 0 : 1;
            for (uint8_t q = 0; q < n_quad; q++) {
                rtype psi[NK];
                if (!face_point_basis(f, q, i_cell, psi)) continue;
                rtype U_f[N_CONSERVATIVE];
                FOR_I_CONSERVATIVE U_f[i] = U0[i];
                for (uint8_t l = 0; l < NK; l++) {
                    FOR_I_CONSERVATIVE U_f[i] += aK[l][i] * psi[l];
                }
                rtype Wq[N_CONSERVATIVE];
                to_primitives(U_f, Wq);
                admissible = admissible && (Wq[0] > 0.0_r) && (Wq[N_DIM + 1] > 0.0_r) && Kokkos::isfinite(Wq[N_DIM + 1]);
                FOR_I_CONSERVATIVE face_solution(f, q, side, i) = Wq[i];
            }
        }
        if (!admissible) {
            for (uint8_t k = 0; k < n_faces; k++) {
                const uint32_t f = faces_of_cell(f_begin + k);
                const uint8_t side = (cells_of_face(f, 0) == static_cast<int32_t>(i_cell)) ? 0 : 1;
                for (uint8_t q = 0; q < n_quad; q++) {
                    FOR_I_CONSERVATIVE face_solution(f, q, side, i) = W0[i];
                }
            }
        }
    }

    KOKKOS_INLINE_FUNCTION
    bool is_padding(const uint32_t f, const uint8_t q) const {
        if constexpr (N_DIM == 2) {
            return false;
        } else {
            return face_quad_weights(f, q) == 0.0_r;
        }
    }

    KOKKOS_INLINE_FUNCTION
    uint8_t side_of(const uint32_t f, const uint32_t i_cell) const {
        return (cells_of_face(f, 0) == static_cast<int32_t>(i_cell)) ? 0 : 1;
    }

    /**
     * @brief Left and right eigenvectors for the characteristic projection on
     *        face f of cell i_cell (identity if not characteristic).
     */
    KOKKOS_INLINE_FUNCTION
    void face_eigenvectors(const uint32_t f, const uint32_t i_cell, const rtype * W0, rtype L[][N_CONSERVATIVE],
                           rtype R[][N_CONSERVATIVE]) const {
        if (characteristic) {
            const int32_t c0 = cells_of_face(f, 0), c1 = cells_of_face(f, 1);
            const int32_t nb = (c0 == static_cast<int32_t>(i_cell)) ? c1 : c0;
            rtype W_avg[N_CONSERVATIVE];
            FOR_I_CONSERVATIVE W_avg[i] = (nb >= 0) ? 0.5_r * (W0[i] + W(nb, i)) : W0[i];
            rtype n[N_DIM], n_vec[N_DIM];
            FOR_I_DIM n_vec[i] = face_normals(f, i);
            unit<N_DIM>(n_vec, n);
            if constexpr (PRIM) {
                const rtype g = (nb >= 0) ? 0.5_r * (cell_gamma(i_cell) + cell_gamma(nb)) : cell_gamma(i_cell);
                teno::primitive_eigenvectors(W_avg, Kokkos::sqrt(g * W_avg[N_DIM + 1] / W_avg[0]), n, L, R);
            } else {
                teno::eigenvectors(W_avg, n, gamma, L, R);
            }
        } else {
            FOR_I_CONSERVATIVE {
                for (uint8_t m = 0; m < N_CONSERVATIVE; m++) {
                    L[i][m] = (i == m) ? 1.0 : 0.0;
                    R[i][m] = L[i][m];
                }
            }
        }
    }

    /** @brief Component var of L a. */
    KOKKOS_INLINE_FUNCTION
    static rtype project(const rtype L[][N_CONSERVATIVE], const uint8_t var, const rtype * a) {
        rtype c = 0.0;
        for (uint8_t m = 0; m < N_CONSERVATIVE; m++) c += L[var][m] * a[m];
        return c;
    }

    /** @brief Smoothness indicator c^T S c of the N coefficients c of cell i_cell. */
    template <uint8_t N>
    KOKKOS_INLINE_FUNCTION
    rtype smoothness(const rtype * c, const uint32_t i_cell) const {
        rtype si = 0.0;
        for (uint8_t l = 0; l < N; l++) {
            rtype row = 0.0;
            for (uint8_t m = 0; m < N; m++) {
                row += si_matrix(i_cell, (l <= m) ? teno::upper_index(l, m, NK) : teno::upper_index(m, l, NK)) * c[m];
            }
            si += c[l] * row;
        }
        return si;
    }

    /** @brief Coefficients of the sector stencil s of queued cell j. */
    KOKKOS_INLINE_FUNCTION
    void troubled_sector(const uint32_t j, const uint8_t s) const {
        const uint32_t i_cell = troubled_cells(j);
        const uint32_t f_begin = offsets_faces_of_cell(i_cell);
        if (s >= offsets_faces_of_cell(i_cell + 1) - f_begin) return;
        const uint16_t n_small = stencil_small_size(i_cell, s);
        if (n_small == 0) return;
        rtype U0[N_CONSERVATIVE];
        conservatives(i_cell, U0);
        rtype aS[teno::NK_SMALL][N_CONSERVATIVE] = {};
        // The cell's sector stencils are stored one after another
        uint16_t start = 0;
        for (uint8_t t = 0; t < s; t++) start += stencil_small_size(i_cell, t);
        const teno::PackedStencils::Row stencil = stencil_small.row(i_cell);
        for (uint16_t e = 0; e < n_small; e++) {
            rtype U[N_CONSERVATIVE];
            entry_conservatives(stencil.cell(start + e), stencil.face(start + e), U);
            for (uint8_t l = 0; l < teno::NK_SMALL; l++) {
                const rtype P = stencil.pinv<teno::NK_SMALL>(start + e, l);
                FOR_I_CONSERVATIVE aS[l][i] += P * (U[i] - U0[i]);
            }
        }
        for (uint8_t l = 0; l < teno::NK_SMALL; l++) {
            FOR_I_CONSERVATIVE small_coeffs(i_cell, s, l, i) = aS[l][i];
        }
    }

    /**
     * @brief Stencil selection for characteristic variable var on face k of
     *        queued cell j, and that variable at the face's quadrature points,
     *        left in face_solution(f, q, side, var).
     */
    KOKKOS_INLINE_FUNCTION
    void troubled_select(const uint32_t j, const uint8_t k, const uint8_t var) const {
        const uint32_t i_cell = troubled_cells(j);
        const uint32_t f_begin = offsets_faces_of_cell(i_cell);
        const uint8_t n_faces = offsets_faces_of_cell(i_cell + 1) - f_begin;
        if (k >= n_faces) return;
        constexpr rtype eps = 1.0e-12;
        rtype U0[N_CONSERVATIVE], W0[N_CONSERVATIVE];
        conservatives(i_cell, U0);
        FOR_I_CONSERVATIVE W0[i] = W(i_cell, i);
        const uint32_t f = faces_of_cell(f_begin + k);
        rtype L[N_CONSERVATIVE][N_CONSERVATIVE], R[N_CONSERVATIVE][N_CONSERVATIVE];
        face_eigenvectors(f, i_cell, W0, L, R);

        const rtype c0_char = project(L, var, U0);
        rtype cK[NK];
        for (uint8_t l = 0; l < NK; l++) {
            rtype a[N_CONSERVATIVE];
            FOR_I_CONSERVATIVE a[i] = coeffs(i_cell, l, i);
            cK[l] = project(L, var, a);
        }
        bool valid[teno::MAX_FACES] = {};
        rtype cS[teno::MAX_FACES][teno::NK_SMALL];
        for (uint8_t s = 0; s < n_faces; s++) {
            valid[s] = stencil_small_size(i_cell, s) > 0;
            if (!valid[s]) continue;
            for (uint8_t l = 0; l < teno::NK_SMALL; l++) {
                rtype a[N_CONSERVATIVE];
                FOR_I_CONSERVATIVE a[i] = small_coeffs(i_cell, s, l, i);
                cS[s][l] = project(L, var, a);
            }
        }

        // gamma_k = 1 / (SI_k + eps)^6, normalized by the largest one so that
        // the weights cannot overflow (even in single precision)
        const rtype sigma = sigma_out(i_cell);
        const rtype cutoff = (C_T > 0.0_r) ? C_T : teno::adaptive_CT(sigma, sigma_threshold, sigma_upper);
        const rtype si_K = smoothness<NK>(cK, i_cell) + eps;
        rtype si_small[teno::MAX_FACES] = {};
        rtype si_min = si_K;
        for (uint8_t s = 0; s < n_faces; s++) {
            if (!valid[s]) continue;
            si_small[s] = smoothness<teno::NK_SMALL>(cS[s], i_cell) + eps;
            si_min = Kokkos::fmin(si_min, si_small[s]);
        }
        const rtype gK = Kokkos::pow(si_min / si_K, 6.0_r);
        rtype g_small[teno::MAX_FACES] = {};
        rtype sum_small = 0.0;
        for (uint8_t s = 0; s < n_faces; s++) {
            if (!valid[s]) continue;
            g_small[s] = Kokkos::pow(si_min / si_small[s], 6.0_r);
            sum_small += g_small[s];
        }
        const bool use_large = (sum_small == 0.0_r) || (gK / (gK + sum_small) >= cutoff);
        rtype w_small[teno::MAX_FACES] = {};
        if (!use_large) {
            rtype n_kept = 0.0;
            for (uint8_t s = 0; s < n_faces; s++) {
                const bool keep = valid[s] && (g_small[s] / sum_small >= cutoff);
                w_small[s] = keep ? 1.0 : 0.0;
                n_kept += w_small[s];
            }
            for (uint8_t s = 0; s < n_faces; s++) w_small[s] /= n_kept;
        }

        const uint8_t side = side_of(f, i_cell);
        if constexpr (PRIM) {
            if (var == 1) {
                uint8_t mask = 0;
                for (uint8_t s = 0; s < n_faces; s++) mask |= (w_small[s] > 0.0_r) ? (1u << s) : 0u;
                selection(f, side) = use_large ? TENO::SELECT_LARGE : mask;
            }
        }
        const uint8_t n_quad = this->n_quad();
        for (uint8_t q = 0; q < n_quad; q++) {
            rtype psi[NK];
            if (!face_point_basis(f, q, i_cell, psi)) continue;
            rtype v_char = c0_char;
            if (use_large) {
                for (uint8_t l = 0; l < NK; l++) v_char += cK[l] * psi[l];
            } else {
                for (uint8_t s = 0; s < n_faces; s++) {
                    if (!valid[s] || w_small[s] == 0.0_r) continue;
                    rtype p_s = 0.0;
                    for (uint8_t l = 0; l < teno::NK_SMALL; l++) p_s += cS[s][l] * psi[l];
                    v_char += w_small[s] * p_s;
                }
            }
            face_solution(f, q, side, var) = v_char;
        }
    }

    /**
     * @brief Conservative states at the quadrature points of face k of queued
     *        cell j from the characteristic ones that troubled_select left.
     */
    KOKKOS_INLINE_FUNCTION
    void troubled_project(const uint32_t j, const uint8_t k) const {
        const uint32_t i_cell = troubled_cells(j);
        const uint32_t f_begin = offsets_faces_of_cell(i_cell);
        if (k >= offsets_faces_of_cell(i_cell + 1) - f_begin) return;
        rtype W0[N_CONSERVATIVE];
        FOR_I_CONSERVATIVE W0[i] = W(i_cell, i);
        const uint32_t f = faces_of_cell(f_begin + k);
        rtype L[N_CONSERVATIVE][N_CONSERVATIVE], R[N_CONSERVATIVE][N_CONSERVATIVE];
        face_eigenvectors(f, i_cell, W0, L, R);
        const uint8_t side = side_of(f, i_cell);
        const uint8_t n_quad = this->n_quad();
        for (uint8_t q = 0; q < n_quad; q++) {
            if (is_padding(f, q)) continue;
            rtype v_char[N_CONSERVATIVE];
            FOR_I_CONSERVATIVE v_char[i] = face_solution(f, q, side, i);
            FOR_I_CONSERVATIVE {
                rtype U_f = 0.0;
                for (uint8_t m = 0; m < N_CONSERVATIVE; m++) U_f += R[i][m] * v_char[m];
                face_solution(f, q, side, i) = U_f;
            }
        }
    }

    uint32_t stride = 0;  // threads of each troubled pass, which loop over the queue

    KOKKOS_INLINE_FUNCTION
    void operator()(TroubledSectorPass, const uint32_t t) const {
        const uint32_t n = n_troubled() * teno::MAX_FACES;
        for (uint32_t idx = t; idx < n; idx += stride) troubled_sector(idx / teno::MAX_FACES, idx % teno::MAX_FACES);
    }

    KOKKOS_INLINE_FUNCTION
    void operator()(TroubledSelectPass, const uint32_t t) const {
        constexpr uint32_t per_cell = teno::MAX_FACES * N_CONSERVATIVE;
        const uint32_t n = n_troubled() * per_cell;
        for (uint32_t idx = t; idx < n; idx += stride) {
            const uint32_t r = idx % per_cell;
            troubled_select(idx / per_cell, r / N_CONSERVATIVE, r % N_CONSERVATIVE);
        }
    }

    KOKKOS_INLINE_FUNCTION
    void operator()(TroubledProjectPass, const uint32_t t) const {
        const uint32_t n = n_troubled() * teno::MAX_FACES;
        for (uint32_t idx = t; idx < n; idx += stride) troubled_project(idx / teno::MAX_FACES, idx % teno::MAX_FACES);
    }

    /**
     * @brief Conservative face state left by troubled_project; the cell mean
     *        on padding points.
     */
    KOKKOS_INLINE_FUNCTION
    void troubled_face_U(const uint32_t f, const uint8_t q, const uint8_t side, const rtype * U0, rtype * U) const {
        if (is_padding(f, q)) {
            FOR_I_CONSERVATIVE U[i] = U0[i];
        } else {
            FOR_I_CONSERVATIVE U[i] = face_solution(f, q, side, i);
        }
    }

    /**
     * @brief Admissibility check and optional bound-preserving limiting of
     *        queued cell j over all its faces, then the primitive face states.
     */
    KOKKOS_INLINE_FUNCTION
    void operator()(TroubledFinishPass, const uint32_t t) const {
        for (uint32_t j = t; j < n_troubled(); j += stride) troubled_finish(j);
    }

    KOKKOS_INLINE_FUNCTION
    void troubled_finish(const uint32_t j) const {
        const uint32_t i_cell = troubled_cells(j);
        rtype U0[N_CONSERVATIVE], W0[N_CONSERVATIVE];
        conservatives(i_cell, U0);
        FOR_I_CONSERVATIVE W0[i] = W(i_cell, i);
        const uint8_t n_quad = this->n_quad();
        const uint32_t f_begin = offsets_faces_of_cell(i_cell);
        const uint8_t n_faces = offsets_faces_of_cell(i_cell + 1) - f_begin;

        bool admissible = true;
        for (uint8_t k = 0; k < n_faces; k++) {
            const uint32_t f = faces_of_cell(f_begin + k);
            const uint8_t side = side_of(f, i_cell);
            for (uint8_t q = 0; q < n_quad; q++) {
                if (is_padding(f, q)) continue;
                rtype U_f[N_CONSERVATIVE], Wq[N_CONSERVATIVE];
                troubled_face_U(f, q, side, U0, U_f);
                to_primitives(U_f, Wq);
                admissible = admissible && (Wq[0] > 0.0_r) && (Wq[N_DIM + 1] > 0.0_r) && Kokkos::isfinite(Wq[N_DIM + 1]);
            }
        }

        rtype theta = 1.0;
        if (bound_preserving) {
            // Scale the high-order deviation so density and pressure at every face
            // point stay within the range of the cell and its face neighbors
            constexpr uint8_t P = N_DIM + 1;
            rtype lo[2] = {W0[0], W0[P]}, hi[2] = {W0[0], W0[P]};
            for (uint8_t k = 0; k < n_faces; k++) {
                const uint32_t f = faces_of_cell(f_begin + k);
                const int32_t c0 = cells_of_face(f, 0), c1 = cells_of_face(f, 1);
                const int32_t nb = (c0 == static_cast<int32_t>(i_cell)) ? c1 : c0;
                if (nb < 0) continue;
                lo[0] = Kokkos::fmin(lo[0], W(nb, 0));
                hi[0] = Kokkos::fmax(hi[0], W(nb, 0));
                lo[1] = Kokkos::fmin(lo[1], W(nb, P));
                hi[1] = Kokkos::fmax(hi[1], W(nb, P));
            }
            for (uint8_t k = 0; k < n_faces; k++) {
                const uint32_t f = faces_of_cell(f_begin + k);
                const uint8_t side = side_of(f, i_cell);
                for (uint8_t q = 0; q < n_quad; q++) {
                    if (is_padding(f, q)) continue;
                    rtype U_f[N_CONSERVATIVE], Wq[N_CONSERVATIVE];
                    troubled_face_U(f, q, side, U0, U_f);
                    to_primitives(U_f, Wq);
                    const rtype vals[2] = {Wq[0], Wq[P]};
                    const rtype centers[2] = {W0[0], W0[P]};
                    for (uint8_t v = 0; v < 2; v++) {
                        const rtype d = vals[v] - centers[v];
                        if (d > 0.0_r) theta = Kokkos::fmin(theta, (hi[v] - centers[v]) / d);
                        if (d < 0.0_r) theta = Kokkos::fmin(theta, (lo[v] - centers[v]) / d);
                    }
                }
            }
            if (theta < 1.0_r) {
                theta = Kokkos::fmax(0.0_r, theta);
                admissible = true;
                for (uint8_t k = 0; k < n_faces; k++) {
                    const uint32_t f = faces_of_cell(f_begin + k);
                    const uint8_t side = side_of(f, i_cell);
                    for (uint8_t q = 0; q < n_quad; q++) {
                        rtype U_face[N_CONSERVATIVE], U_f[N_CONSERVATIVE], Wq[N_CONSERVATIVE];
                        troubled_face_U(f, q, side, U0, U_face);
                        FOR_I_CONSERVATIVE U_f[i] = U0[i] + theta * (U_face[i] - U0[i]);
                        to_primitives(U_f, Wq);
                        admissible = admissible && (Wq[0] > 0.0_r) && (Wq[P] > 0.0_r);
                    }
                }
            }
        }
        const bool limited = theta < 1.0_r;

        for (uint8_t k = 0; k < n_faces; k++) {
            const uint32_t f = faces_of_cell(f_begin + k);
            const uint8_t side = side_of(f, i_cell);
            for (uint8_t q = 0; q < n_quad; q++) {
                rtype Wq[N_CONSERVATIVE];
                if (!admissible) {
                    FOR_I_CONSERVATIVE Wq[i] = W0[i];
                } else {
                    rtype U_f[N_CONSERVATIVE];
                    troubled_face_U(f, q, side, U0, U_f);
                    if (limited) {
                        FOR_I_CONSERVATIVE U_f[i] = U0[i] + theta * (U_f[i] - U0[i]);
                        to_primitives(U_f, Wq);
                    } else if (is_padding(f, q)) {
                        FOR_I_CONSERVATIVE Wq[i] = W0[i];
                    } else {
                        to_primitives(U_f, Wq);
                    }
                }
                FOR_I_CONSERVATIVE face_solution(f, q, side, i) = Wq[i];
            }
        }
    }
};

// Enough threads to fill a GPU, few enough that those finding no work cost little
constexpr uint32_t TROUBLED_THREADS = 1u << 18;

template <uint8_t DEG, bool PRIM>
void TENO::launch_reconstruction(const Kokkos::DefaultExecutionSpace & exec,
                                 Kokkos::View<rtype *[N_CONSERVATIVE]> solution,
                                 Kokkos::View<rtype **[2][N_CONSERVATIVE]> face_solution,
                                 Kokkos::View<uint32_t *> cells, bool troubled_pass) {
    using Functor = TENOFunctor<DEG, PRIM>;
    Functor functor{sigma_threshold, sigma_upper, C_T, characteristic, bound_preserving, boundaries.gamma,
                    mesh->offsets_faces_of_cell, mesh->faces_of_cell, mesh->cells_of_face,
                    mesh->offsets_nodes_of_face, mesh->nodes_of_face, mesh->node_coords,
                    mesh->cell_coords, mesh->face_coords, mesh->face_normals, mesh->shifts, mesh->face_shift,
                    quadrature_face.points, face_quad_points, face_quad_weights, boundaries,
                    scale, basis_mean, stencil_large_size, stencil_large, stencil_small_size, stencil_small,
                    si_matrix, troubled, troubled_coeffs, troubled_small_coeffs, troubled_cells, n_troubled,
                    solution, face_solution, cells, {}, cell_gamma, cell_molar_mass, selection};
    using Dynamic = Kokkos::Schedule<Kokkos::Dynamic>;
    using Space = Kokkos::DefaultExecutionSpace;
    if (!troubled_pass) {
        const uint32_t n = cells.extent(0) ? cells.extent(0) : mesh->n_reconstructed();
        Kokkos::parallel_for("teno_smooth", Kokkos::RangePolicy<Space, typename Functor::SmoothPass, Dynamic>(exec, 0, n),
                             functor);
        return;
    }
    // Each pass loops over the queue with as many threads as it could need,
    // up to TROUBLED_THREADS, so the queue length stays on the device
    const uint32_t n = mesh->n_reconstructed();
    auto launch = [&](const char * label, auto tag, uint32_t per_cell) {
        functor.stride = std::min<uint64_t>(TROUBLED_THREADS, uint64_t(n) * per_cell);
        if (functor.stride == 0) return;
        Kokkos::parallel_for(label, Kokkos::RangePolicy<Space, decltype(tag), Dynamic>(exec, 0, functor.stride),
                             functor);
    };
    launch("teno_troubled_sectors", typename Functor::TroubledSectorPass{}, teno::MAX_FACES);
    launch("teno_troubled_select", typename Functor::TroubledSelectPass{}, teno::MAX_FACES * N_CONSERVATIVE);
    launch("teno_troubled_project", typename Functor::TroubledProjectPass{}, teno::MAX_FACES);
    launch("teno_troubled_finish", typename Functor::TroubledFinishPass{}, 1);
    Kokkos::deep_copy(exec, n_troubled, 0u);
}

template <uint8_t DEG, bool PRIM>
void TENO::launch_gradients(Kokkos::View<rtype *[N_CONSERVATIVE]> solution,
                            Kokkos::View<rtype *[N_CONSERVATIVE][N_DIM]> gradients, const uint32_t n_cells) {
    using Functor = TENOFunctor<DEG, PRIM>;
    Functor functor{sigma_threshold, sigma_upper, C_T, characteristic, bound_preserving, boundaries.gamma,
                    mesh->offsets_faces_of_cell, mesh->faces_of_cell, mesh->cells_of_face,
                    mesh->offsets_nodes_of_face, mesh->nodes_of_face, mesh->node_coords,
                    mesh->cell_coords, mesh->face_coords, mesh->face_normals, mesh->shifts, mesh->face_shift,
                    quadrature_face.points, face_quad_points, face_quad_weights, boundaries,
                    scale, basis_mean, stencil_large_size, stencil_large, stencil_small_size, stencil_small,
                    si_matrix, troubled, troubled_coeffs, troubled_small_coeffs, troubled_cells, n_troubled,
                    solution, {}, {}, gradients, cell_gamma, cell_molar_mass, selection};
    Kokkos::parallel_for("teno_gradients", Kokkos::RangePolicy<typename Functor::GradientPass>(0, n_cells), functor);
}

void TENO::set_mixture(Kokkos::View<rtype *, Kokkos::LayoutStride> gamma, Kokkos::View<rtype *> molar_mass) {
    primitive = true;
    cell_gamma = gamma;
    cell_molar_mass = molar_mass;
    selection = Kokkos::View<uint8_t **>("teno_selection", mesh->n_faces, 2);
}

bool TENO::cell_gradients(Kokkos::View<rtype *[N_CONSERVATIVE]> solution,
                          Kokkos::View<rtype *[N_CONSERVATIVE][N_DIM]> gradients, const uint32_t n_cells) {
    switch (degree * 2 + primitive) {
        case 4: launch_gradients<2, false>(solution, gradients, n_cells); break;
        case 6: launch_gradients<3, false>(solution, gradients, n_cells); break;
        case 8: launch_gradients<4, false>(solution, gradients, n_cells); break;
        case 10: launch_gradients<5, false>(solution, gradients, n_cells); break;
        case 5: launch_gradients<2, true>(solution, gradients, n_cells); break;
        case 7: launch_gradients<3, true>(solution, gradients, n_cells); break;
        case 9: launch_gradients<4, true>(solution, gradients, n_cells); break;
        case 11: launch_gradients<5, true>(solution, gradients, n_cells); break;
        default: throw std::runtime_error("TENO: unsupported degree.");
    }
    return true;
}

void TENO::calc_face_values(Kokkos::View<rtype *[N_CONSERVATIVE]> solution,
                            Kokkos::View<rtype **[2][N_CONSERVATIVE]> face_solution) {
    calc_cell_face_values(Kokkos::DefaultExecutionSpace(), solution, face_solution, Kokkos::View<uint32_t *>());
    finish_cell_face_values(solution, face_solution);
}

std::vector<uint32_t> TENO::cells_independent_of_halo(uint32_t n_owned) const {
    // Stencil entries (mirrored ones too) and the face neighbors used for the
    // characteristic projection and the bounds
    const auto large_slices = host_slice_start(stencil_large);
    const auto small_slices = host_slice_start(stencil_small);
    std::vector<uint32_t> cells;
    std::vector<CellTables> chunk;
    for (uint32_t c0 = 0; c0 < n_owned; c0 += CHUNK_CELLS) {
        chunk.assign(std::min(CHUNK_CELLS, static_cast<uint32_t>(scale.extent(0)) - c0), CellTables());
        download_tables(*this, large_slices, small_slices, c0, chunk);
        for (uint32_t c = c0; c < std::min(n_owned, c0 + CHUNK_CELLS); c++) {
            const CellTables & t = chunk[c - c0];
            auto owned = [&](int32_t cell) { return cell < static_cast<int32_t>(n_owned); };
            bool independent = std::all_of(t.large_cells.begin(), t.large_cells.end(), owned) &&
                               std::all_of(t.small_cells.begin(), t.small_cells.end(), owned);
            for (uint32_t k = 0; k < mesh->h_n_faces_of_cell(c); k++) {
                const uint32_t f = mesh->h_face_of_cell(c, k);
                for (uint8_t side = 0; side < 2; side++) independent = independent && owned(mesh->h_cells_of_face(f, side));
            }
            if (independent) cells.push_back(c);
        }
    }
    return cells;
}

void TENO::calc_cell_face_values(const Kokkos::DefaultExecutionSpace & exec,
                                 Kokkos::View<rtype *[N_CONSERVATIVE]> solution,
                                 Kokkos::View<rtype **[2][N_CONSERVATIVE]> face_solution,
                                 Kokkos::View<uint32_t *> cells) {
    dispatch(exec, solution, face_solution, cells, false);
}

void TENO::finish_cell_face_values(Kokkos::View<rtype *[N_CONSERVATIVE]> solution,
                                   Kokkos::View<rtype **[2][N_CONSERVATIVE]> face_solution) {
    dispatch(Kokkos::DefaultExecutionSpace(), solution, face_solution, Kokkos::View<uint32_t *>(), true);
}

void TENO::dispatch(const Kokkos::DefaultExecutionSpace & exec, Kokkos::View<rtype *[N_CONSERVATIVE]> solution,
                    Kokkos::View<rtype **[2][N_CONSERVATIVE]> face_solution, Kokkos::View<uint32_t *> cells,
                    bool troubled_pass) {
    switch (degree * 2 + primitive) {
        case 4: launch_reconstruction<2, false>(exec, solution, face_solution, cells, troubled_pass); break;
        case 6: launch_reconstruction<3, false>(exec, solution, face_solution, cells, troubled_pass); break;
        case 8: launch_reconstruction<4, false>(exec, solution, face_solution, cells, troubled_pass); break;
        case 10: launch_reconstruction<5, false>(exec, solution, face_solution, cells, troubled_pass); break;
        case 5: launch_reconstruction<2, true>(exec, solution, face_solution, cells, troubled_pass); break;
        case 7: launch_reconstruction<3, true>(exec, solution, face_solution, cells, troubled_pass); break;
        case 9: launch_reconstruction<4, true>(exec, solution, face_solution, cells, troubled_pass); break;
        case 11: launch_reconstruction<5, true>(exec, solution, face_solution, cells, troubled_pass); break;
        default: throw std::runtime_error("TENO: unsupported degree.");
    }
}

namespace {

// Version 3 stores each reconstructed cell's tables at their actual stencil sizes; version 4 gathers
// candidates by interior cells, version 5 sorts them in the mesh-spacing metric, version 6 follows
// the round-off-accurate 2D cell centroids
constexpr char TENO_CACHE_MAGIC[16] = "MALLARD-TENO-6";
constexpr char TENO_CACHE_FAMILY[] = "MALLARD-TENO-";

struct Fnv1a {
    uint64_t h = 1469598103934665603ULL;
    void add(const void * data, size_t n) {
        const unsigned char * p = static_cast<const unsigned char *>(data);
        for (size_t i = 0; i < n; i++) {
            h ^= p[i];
            h *= 1099511628211ULL;
        }
    }
    template <typename T>
    void add(const T & value) { add(&value, sizeof(T)); }
};

struct CacheHeader {
    char magic[sizeof(TENO_CACHE_MAGIC)] = {};
    uint64_t options_key = 0;
    uint64_t cache_key = 0;
    uint8_t halo_layers = 0;
    uint32_t n_reconstructed = 0;
};

template <typename T>
void put(std::vector<char> & buf, const T * data, size_t n) {
    const char * p = reinterpret_cast<const char *>(data);
    buf.insert(buf.end(), p, p + n * sizeof(T));
}

template <typename T>
bool get(std::istream & in, T * data, size_t n) {
    in.read(reinterpret_cast<char *>(data), n * sizeof(T));
    return in.good();
}

void write_header(std::ostream & out, const CacheHeader & h) {
    std::vector<char> buf;
    put(buf, TENO_CACHE_MAGIC, sizeof(TENO_CACHE_MAGIC));
    put(buf, &h.options_key, 1);
    put(buf, &h.cache_key, 1);
    put(buf, &h.halo_layers, 1);
    put(buf, &h.n_reconstructed, 1);
    out.write(buf.data(), buf.size());
}

bool read_header(std::istream & in, CacheHeader & h) {
    return get(in, h.magic, sizeof(h.magic)) && get(in, &h.options_key, 1) && get(in, &h.cache_key, 1) &&
           get(in, &h.halo_layers, 1) && get(in, &h.n_reconstructed, 1);
}

/** @brief Append one cell's record: its fixed-size data, then its stencils at their actual sizes. */
void serialize(const CellTables & t, std::vector<char> & buf) {
    const uint16_t n_large = t.large_cells.size();
    put(buf, &t.gather_depth, 1);
    put(buf, &t.scale, 1);
    put(buf, &n_large, 1);
    put(buf, t.small_size.data(), t.small_size.size());
    put(buf, t.basis_mean.data(), t.basis_mean.size());
    put(buf, t.si.data(), t.si.size());
    put(buf, t.large_cells.data(), n_large);
    put(buf, t.large_faces.data(), n_large);
    put(buf, t.large_pinv.data(), t.large_pinv.size());
    put(buf, t.small_cells.data(), t.small_cells.size());
    put(buf, t.small_faces.data(), t.small_faces.size());
    put(buf, t.small_pinv.data(), t.small_pinv.size());
}

bool deserialize(std::istream & in, const uint8_t nk, CellTables & t) {
    uint16_t n_large = 0;
    if (!get(in, &t.gather_depth, 1) || !get(in, &t.scale, 1) || !get(in, &n_large, 1) ||
        !get(in, t.small_size.data(), t.small_size.size())) {
        return false;
    }
    size_t n_small = 0;
    for (uint16_t n : t.small_size) n_small += n;
    t.basis_mean.resize(nk);
    t.si.resize(nk * (nk + 1) / 2);
    t.large_cells.resize(n_large);
    t.large_faces.resize(n_large);
    t.large_pinv.resize(size_t(n_large) * nk);
    t.small_cells.resize(n_small);
    t.small_faces.resize(n_small);
    t.small_pinv.resize(n_small * teno::NK_SMALL);
    return get(in, t.basis_mean.data(), t.basis_mean.size()) && get(in, t.si.data(), t.si.size()) &&
           get(in, t.large_cells.data(), n_large) && get(in, t.large_faces.data(), n_large) &&
           get(in, t.large_pinv.data(), t.large_pinv.size()) && get(in, t.small_cells.data(), n_small) &&
           get(in, t.small_faces.data(), n_small) && get(in, t.small_pinv.data(), t.small_pinv.size());
}

} // namespace

void TENO::allocate_scratch() {
    const uint32_t n_reconstructed = mesh->n_reconstructed();
    troubled = Kokkos::View<rtype *>("teno_sigma", mesh->n_cells);
    troubled_coeffs = Kokkos::View<rtype ***>("teno_troubled_coeffs", n_reconstructed, n_dof_large, N_CONSERVATIVE);
    troubled_small_coeffs = Kokkos::View<rtype ****>("teno_troubled_small_coeffs", n_reconstructed, teno::MAX_FACES,
                                                     teno::NK_SMALL, N_CONSERVATIVE);
    troubled_cells = Kokkos::View<uint32_t *>("teno_troubled_cells", n_reconstructed);
    n_troubled = Kokkos::View<uint32_t>("teno_n_troubled");
}

std::array<double, 9> TENO::stencil_metric(const uint32_t i) const {
    auto h_face_bc = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), boundaries.face_bc);
    auto h_bcs = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), boundaries.bcs);
    return ranking_metric(*mesh, h_face_bc, h_bcs, i);
}

TENO::Stencils TENO::large_stencils() const {
    Stencils out;
    out.offsets.push_back(0);
    const uint32_t n_reconstructed = scale.extent(0);
    const auto large_slices = host_slice_start(stencil_large);
    const auto small_slices = host_slice_start(stencil_small);
    std::vector<CellTables> chunk;
    for (uint32_t c0 = 0; c0 < n_reconstructed; c0 += CHUNK_CELLS) {
        chunk.assign(std::min(CHUNK_CELLS, n_reconstructed - c0), CellTables());
        download_tables(*this, large_slices, small_slices, c0, chunk);
        for (const CellTables & t : chunk) {
            out.cells.insert(out.cells.end(), t.large_cells.begin(), t.large_cells.end());
            out.faces.insert(out.faces.end(), t.large_faces.begin(), t.large_faces.end());
            out.offsets.push_back(out.cells.size());
        }
    }
    return out;
}

uint64_t TENO::options_key() const {
    Fnv1a hash;
    hash.add(sizeof(rtype));
    hash.add(degree);
    hash.add(n_stencil_small);
    hash.add(stencil_factor);
    hash.add(max_condition);
    hash.add(comm::size());
    hash.add(comm::rank());
    return hash.h;
}

uint64_t TENO::cache_key() const {
    Fnv1a hash;
    hash.add(options_key());
    hash.add(mesh->n_cells);
    hash.add(mesh->n_faces);
    hash.add(mesh->n_reconstructed());
    hash.add(mesh->n_complete());
    hash.add(mesh->h_node_coords.data(), mesh->h_node_coords.span() * sizeof(rtype));
    hash.add(mesh->h_offsets_nodes_of_cell.data(), mesh->h_offsets_nodes_of_cell.span() * sizeof(uint32_t));
    hash.add(mesh->h_nodes_of_cell.data(), mesh->h_nodes_of_cell.span() * sizeof(uint32_t));
    hash.add(mesh->h_faces_of_cell.data(), mesh->h_faces_of_cell.span() * sizeof(uint32_t));
    // A rank's part of a distributed mesh: which global cells it holds, in which order
    hash.add(mesh->h_global_cell_id.data(), mesh->h_global_cell_id.size() * sizeof(uint64_t));
    auto h_face_bc = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), boundaries.face_bc);
    hash.add(h_face_bc.data(), h_face_bc.span() * sizeof(int32_t));
    auto h_bcs = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), boundaries.bcs);
    for (size_t b = 0; b < h_bcs.extent(0); b++) hash.add(h_bcs(b).type);
    if (mesh->is_periodic()) {
        hash.add(mesh->h_shifts.data(), mesh->h_shifts.span() * sizeof(rtype));
        hash.add(mesh->h_face_shift.data(), mesh->h_face_shift.span() * sizeof(uint8_t));
    }
    return hash.h;
}

uint8_t TENO::cached_halo_layers(const toml::value & input) {
    TENO teno;
    teno.read_options(input);
    if (teno.cache_file.empty()) return 0;
    std::ifstream in(teno.cache_file, std::ios::binary);
    CacheHeader header;
    if (!read_header(in, header) || std::string(header.magic) != TENO_CACHE_MAGIC ||
        header.options_key != teno.options_key()) {
        return 0;
    }
    return header.halo_layers;
}

void TENO::save_cache(const uint8_t halo_layers) {
    if (cache_file.empty() || cache_loaded) return;
    // Written aside and renamed, so an interrupted run never leaves a truncated cache
    const std::string partial = cache_file + ".partial";
    std::ofstream out(partial, std::ios::binary);
    if (!out.good()) {
        logging::warning("TENO: could not write stencil cache " + cache_file + ".");
        return;
    }
    const uint32_t n_reconstructed = scale.extent(0);
    CacheHeader header;
    header.options_key = options_key();
    header.cache_key = cache_key();
    header.halo_layers = halo_layers;
    header.n_reconstructed = n_reconstructed;
    write_header(out, header);
    const auto large_slices = host_slice_start(stencil_large);
    const auto small_slices = host_slice_start(stencil_small);
    std::vector<CellTables> chunk;
    std::vector<char> buf;
    for (uint32_t c0 = 0; c0 < n_reconstructed; c0 += CHUNK_CELLS) {
        chunk.assign(std::min(CHUNK_CELLS, n_reconstructed - c0), CellTables());
        download_tables(*this, large_slices, small_slices, c0, chunk);
        buf.clear();
        for (const CellTables & t : chunk) serialize(t, buf);
        out.write(buf.data(), buf.size());
    }
    out.close();
    std::error_code error;
    if (out.good()) std::filesystem::rename(partial, cache_file, error);
    if (!out.good() || error) {
        std::filesystem::remove(partial, error);
        logging::warning("TENO: could not write stencil cache " + cache_file + ".");
        return;
    }
    cache_status += (cache_status.empty() ? "" : ", ") +
                    logging::format("written to %s (%.0f MB)", cache_file.c_str(),
                                    static_cast<double>(std::filesystem::file_size(cache_file, error)) / 1e6);
}

bool TENO::load_cache() {
    std::ifstream in(cache_file, std::ios::binary);
    if (!in.good()) return false;
    CacheHeader header;
    const bool read = read_header(in, header);
    const std::string magic(header.magic, strnlen(header.magic, sizeof(header.magic)));
    if (!read || magic != TENO_CACHE_MAGIC) {
        cache_status = cache_file + (magic.rfind(TENO_CACHE_FAMILY, 0) == 0 ? " has an older format, recomputed"
                                                                            : " is not a TENO cache, recomputed");
        return false;
    }
    const uint32_t n_reconstructed = mesh->n_reconstructed();
    if (header.options_key != options_key() || header.n_reconstructed != n_reconstructed ||
        header.cache_key != cache_key()) {
        cache_status = cache_file + " does not match this case, recomputed";
        return false;
    }
    TableBuilder builder(*this, n_reconstructed);
    std::vector<CellTables> chunk;
    bool ok = true;
    for (uint32_t c0 = 0; c0 < n_reconstructed && ok; c0 += CHUNK_CELLS) {
        chunk.assign(std::min(CHUNK_CELLS, n_reconstructed - c0), CellTables());
        for (CellTables & t : chunk) {
            ok = ok && deserialize(in, n_dof_large, t);
            largest_stencil = std::max<uint32_t>(largest_stencil, t.large_cells.size());
        }
        if (ok) builder.add(c0, chunk);
    }
    if (!ok) {
        cache_status = cache_file + " is truncated, recomputed";
        return false;
    }
    builder.finish();
    cache_status = "loaded from " + cache_file;
    return true;
}

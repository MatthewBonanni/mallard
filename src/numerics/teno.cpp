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
#include <memory>
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
#include "growing_buffer.h"

#include "input.h"
#include "launch_bounds.h"
#include "teno.h"
#include "teno_setup.h"

namespace {

using teno_setup::GEOMETRY_TOL;
using teno_setup::MAX_LEBESGUE_2D;

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
    // Householder reflections H_k = I - beta_k v_k v_k^T, applied a row at a
    // time to all columns of A at once (contiguous reads)
    std::vector<double> V(n * m, 0.0), beta(n, 0.0), d(n);
    double max_diag = 0.0;
    for (int k = 0; k < n; k++) {
        double norm = 0.0;
        for (int i = k; i < m; i++) norm += A[i * n + k] * A[i * n + k];
        norm = std::sqrt(norm);
        if (norm == 0.0) return false;
        const double alpha = (A[k * n + k] > 0.0) ? -norm : norm;
        double * v = &V[k * m];
        for (int i = k; i < m; i++) v[i] = A[i * n + k];
        v[k] -= alpha;
        double vnorm2 = 0.0;
        for (int i = k; i < m; i++) vnorm2 += v[i] * v[i];
        if (vnorm2 == 0.0) continue;
        beta[k] = 2.0 / vnorm2;
        std::fill(d.begin(), d.end(), 0.0);
        for (int i = k; i < m; i++) {
            for (int j = 0; j < n; j++) d[j] += v[i] * A[i * n + j];
        }
        for (int j = 0; j < n; j++) d[j] *= beta[k];
        for (int i = k; i < m; i++) {
            for (int j = 0; j < n; j++) A[i * n + j] -= d[j] * v[i];
        }
        max_diag = std::max(max_diag, std::abs(A[k * n + k]));
    }
    for (int k = 0; k < n; k++) {
        if (std::abs(A[k * n + k]) * max_condition < max_diag) return false;
    }
    // Row r < n of Q^T = H_{n-1} ... H_0 is e_r^T H_r ... H_0, since H_k leaves
    // e_r unchanged for k > r: only the n rows the pseudo-inverse needs, built
    // as the columns of X (m x n) with H_k applied to columns r >= k at once
    std::vector<double> X(m * n, 0.0);
    for (int r = 0; r < n; r++) X[r * n + r] = 1.0;
    for (int k = n - 1; k >= 0; k--) {
        if (beta[k] == 0.0) continue;
        const double * v = &V[k * m];
        std::fill(d.begin(), d.end(), 0.0);
        for (int i = k; i < m; i++) {
            for (int r = k; r < n; r++) d[r] += X[i * n + r] * v[i];
        }
        for (int r = k; r < n; r++) d[r] *= beta[k];
        for (int i = k; i < m; i++) {
            for (int r = k; r < n; r++) X[i * n + r] -= d[r] * v[i];
        }
    }
    P.assign(n * m, 0.0);
    for (int r = 0; r < n; r++) {
        for (int j = 0; j < m; j++) P[r * m + j] = X[j * n + r];
    }
    for (int k = n - 1; k >= 0; k--) {
        double * row = &P[k * m];
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
    // Ties in position order: the order of a stable sort, without its buffer
    std::sort(order.begin(), order.end(), [](const auto & a, const auto & b) {
        return a.first < b.first || (!(b.first < a.first) && a.second < b.second);
    });
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
        VisitSet() : keys(1024, EMPTY) {}

        /** @brief Insert v; false if it was present. */
        bool insert(const Visit & v) {
            if (2 * (n + 1) > keys.size()) grow();
            return place(key(v));
        }

    private:
        static constexpr uint64_t EMPTY = ~uint64_t(0);
        static constexpr int LATTICE_BITS = 10;

        // The cell in the low 32 bits and each lattice offset in 10 bits (each
        // layer moves it by at most 1 and the searches stop at 64 layers): bits
        // 62 and 63 stay clear, so no key equals EMPTY
        static uint64_t key(const Visit & v) {
            constexpr int bound = 1 << (LATTICE_BITS - 1);
            uint64_t k = v.cell;
            for (int a = 0; a < 3; a++) k |= static_cast<uint64_t>(v.lattice[a] + bound) << (32 + LATTICE_BITS * a);
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

// Cells per batch of the precomputation, bounding the unpacked per-cell
// tables held at once; a multiple of every slice size
constexpr uint32_t CHUNK_CELLS = 8192;

/** @brief Precomputed data of one reconstructed cell, before packing. */
struct CellTables {
    uint8_t gather_depth = 0;
    rtype scale = 0.0;
    std::vector<rtype> basis_mean;                  // (l)
    std::vector<rtype> si;                          // 2D: upper triangle of (l, m)
    std::vector<double> moments;                    // 3D: central moments
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

template <typename T>
using DeviceUnmanaged = Kokkos::View<T *, Kokkos::MemoryTraits<Kokkos::Unmanaged>>;

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
            // Fresh anonymous pages are zero
            if (value != T(0)) std::fill(data_, data_ + n, value);
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
 *        once every stencil is known: on CUDA, each packed chunk goes straight
 *        into device memory that grows in place (stream()); elsewhere chunks
 *        stay packed on the host until finish().
 */
class PackedRows {
    public:
        PackedRows(uint8_t slice_shift, uint8_t values_per_slot, IndexRow cell_row, IndexRow face_row, ValueRow pinv_row,
                   bool single_values)
            : shift(slice_shift), width(values_per_slot), single(single_values), cells(cell_row), faces(face_row),
              pinv(pinv_row) {}

        /**
         * @brief Upload each chunk as it is added, for n_cells cells of at
         *        most max_slots_per_cell stencil entries each.
         */
        void stream(const uint32_t n_cells, const size_t max_slots_per_cell) {
            const size_t max_slots = ((size_t(n_cells) >> shift) + 1) * max_slots_per_cell << shift;
            memory = {std::make_shared<GrowingBuffer>(max_slots * sizeof(int32_t)),
                      std::make_shared<GrowingBuffer>(max_slots * sizeof(int32_t)),
                      std::make_shared<GrowingBuffer>(max_slots * width * sizeof(rtype))};
        }

        /** @brief The device memory that the stencils of finish() live in, if streamed. */
        const std::vector<std::shared_ptr<GrowingBuffer>> & device_memory() const { return memory; }

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
            chunk.pinv = PageArray<rtype>(teno::PackedStencils::pinv_elements(n_slots * width, single), 0.0);
            Kokkos::parallel_for("teno_pack", Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(0, n),
                                 [&](const uint32_t c) {
                const CellTables & t = tables[c];
                const uint64_t start = slice_start[(c0 + c) >> shift] - chunk.slot0;
                const uint32_t lane = (c0 + c) & (slice - 1);
                for (size_t s = 0; s < (t.*cells).size(); s++) {
                    chunk.cells[((start + s) << shift) + lane] = (t.*cells)[s];
                    chunk.faces[((start + s) << shift) + lane] = (t.*faces)[s];
                    for (uint32_t l = 0; l < width; l++) {
                        put_value(chunk.pinv.data(), (((start + s) * width + l) << shift) + lane, (t.*pinv)[s * width + l]);
                    }
                }
            });
            if (memory.empty()) {
                chunks.push_back(std::move(chunk));
                return;
            }
            const size_t b = chunk.slot0 << shift, n_chunk = chunk.cells.size();
            memory[0]->ensure((b + n_chunk) * sizeof(int32_t));
            memory[1]->ensure((b + n_chunk) * sizeof(int32_t));
            memory[2]->ensure((b + n_chunk) * width * value_bytes());
            Kokkos::deep_copy(DeviceUnmanaged<int32_t>(static_cast<int32_t *>(memory[0]->data()) + b, n_chunk),
                              HostUnmanaged<int32_t>(chunk.cells.data(), n_chunk));
            Kokkos::deep_copy(DeviceUnmanaged<int32_t>(static_cast<int32_t *>(memory[1]->data()) + b, n_chunk),
                              HostUnmanaged<int32_t>(chunk.faces.data(), n_chunk));
            Kokkos::deep_copy(DeviceUnmanaged<char>(static_cast<char *>(memory[2]->data()) + b * width * value_bytes(),
                                                    n_chunk * width * value_bytes()),
                              HostUnmanaged<char>(reinterpret_cast<char *>(chunk.pinv.data()), n_chunk * width * value_bytes()));
        }

        /**
         * @brief Pack the stencils of cells [c0, c0 + n), which follow the
         *        previous chunk, from slots 0..n of a batch on the device:
         *        sizes(t) entries of cells, faces and pinv for slot t.
         */
        void add_batch(const uint32_t c0, const uint32_t n, const Kokkos::View<uint16_t *> & sizes,
                       const teno_setup::SlotArray<int32_t> & batch_cells, const teno_setup::SlotArray<int32_t> & batch_faces,
                       const teno_setup::SlotArray<rtype> & batch_pinv) {
            const uint32_t slice = 1u << shift;
            if (c0 % slice != 0 || c0 >> shift != slice_start.size() - 1) {
                throw std::logic_error("TENO: stencil chunks must be packed in order.");
            }
            auto h_sizes = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),
                                                               Kokkos::subview(sizes, std::make_pair(0u, n)));
            Chunk chunk;
            chunk.slot0 = slice_start.back();
            const size_t first_slice = slice_start.size() - 1;
            for (uint32_t a = 0; a < n; a += slice) {
                size_t largest = 0;
                for (uint32_t c = a; c < std::min(n, a + slice); c++) largest = std::max<size_t>(largest, h_sizes(c));
                slice_start.push_back(slice_start.back() + largest);
            }
            const uint32_t n_slices = slice_start.size() - 1 - first_slice;
            uint64_t longest = 0;
            for (size_t sl = first_slice; sl + 1 < slice_start.size(); sl++) {
                longest = std::max(longest, slice_start[sl + 1] - slice_start[sl]);
            }
            Kokkos::View<uint64_t *> starts("teno_pack_starts", n_slices + 1);
            Kokkos::deep_copy(starts, HostUnmanaged<const uint64_t>(slice_start.data() + first_slice, n_slices + 1));
            const size_t n_slots = (slice_start.back() - chunk.slot0) << shift;
            const size_t b = chunk.slot0 << shift;
            int32_t * dest_cells;
            int32_t * dest_faces;
            char * dest_pinv;
            Kokkos::View<int32_t *> tmp_cells, tmp_faces;
            Kokkos::View<rtype *> tmp_pinv;
            constexpr bool host_device =
                Kokkos::SpaceAccessibility<Kokkos::HostSpace, Kokkos::DefaultExecutionSpace::memory_space>::accessible;
            if (!memory.empty()) {
                memory[0]->ensure((b + n_slots) * sizeof(int32_t));
                memory[1]->ensure((b + n_slots) * sizeof(int32_t));
                memory[2]->ensure((b + n_slots) * width * value_bytes());
                dest_cells = static_cast<int32_t *>(memory[0]->data()) + b;
                dest_faces = static_cast<int32_t *>(memory[1]->data()) + b;
                dest_pinv = static_cast<char *>(memory[2]->data()) + b * width * value_bytes();
            } else if (host_device) {
                chunk.cells = PageArray<int32_t>(n_slots, 0);
                chunk.faces = PageArray<int32_t>(n_slots, 0);
                chunk.pinv = PageArray<rtype>(teno::PackedStencils::pinv_elements(n_slots * width, single), 0.0);
                dest_cells = chunk.cells.data();
                dest_faces = chunk.faces.data();
                dest_pinv = reinterpret_cast<char *>(chunk.pinv.data());
            } else {
                tmp_cells = Kokkos::View<int32_t *>(Kokkos::view_alloc(Kokkos::WithoutInitializing, "teno_pack_cells"), n_slots);
                tmp_faces = Kokkos::View<int32_t *>(Kokkos::view_alloc(Kokkos::WithoutInitializing, "teno_pack_faces"), n_slots);
                tmp_pinv = Kokkos::View<rtype *>(Kokkos::view_alloc(Kokkos::WithoutInitializing, "teno_pack_pinv"),
                                                 teno::PackedStencils::pinv_elements(n_slots * width, single));
                dest_cells = tmp_cells.data();
                dest_faces = tmp_faces.data();
                dest_pinv = reinterpret_cast<char *>(tmp_pinv.data());
            }
            const uint8_t sh = shift;
            const uint8_t w = width;
            const bool sg = single;
            // A thread per (cell, slot), so that small batches still fill the device
            using Range2 = Kokkos::MDRangePolicy<Kokkos::Rank<2>>;
            Kokkos::parallel_for("teno_pack_batch", Range2({0, 0}, {int64_t(uint64_t(n_slices) << sh), int64_t(longest)}),
                                 KOKKOS_LAMBDA(const int64_t t, const int64_t s) {
                const uint32_t sl = t >> sh;
                const uint32_t lane = t & ((1u << sh) - 1);
                const uint64_t start = starts(sl) - starts(0);
                const uint64_t len = starts(sl + 1) - starts(sl);
                const uint32_t size = t < n ? sizes(t) : 0;
                const auto lc = batch_cells.lane(t < n ? t : 0);
                const auto lf = batch_faces.lane(t < n ? t : 0);
                const auto lp = batch_pinv.lane(t < n ? t : 0);
                if (uint64_t(s) < len) {
                    const uint64_t k = ((start + s) << sh) + lane;
                    dest_cells[k] = s < size ? lc[s] : -1;
                    dest_faces[k] = s < size ? lf[s] : -1;
                    for (uint32_t l = 0; l < w; l++) {
                        const uint64_t kp = (((start + s) * w + l) << sh) + lane;
                        const rtype v = s < size ? lp[s * w + l] : rtype(0);
                        if (sg) {
                            reinterpret_cast<float *>(dest_pinv)[kp] = static_cast<float>(v);
                        } else {
                            reinterpret_cast<rtype *>(dest_pinv)[kp] = v;
                        }
                    }
                }
            });
            Kokkos::fence();
            if (!memory.empty()) return;
            if (!host_device) {
                chunk.cells = PageArray<int32_t>(n_slots, 0);
                chunk.faces = PageArray<int32_t>(n_slots, 0);
                chunk.pinv = PageArray<rtype>(tmp_pinv.extent(0), 0.0);
                Kokkos::deep_copy(HostUnmanaged<int32_t>(chunk.cells.data(), n_slots), tmp_cells);
                Kokkos::deep_copy(HostUnmanaged<int32_t>(chunk.faces.data(), n_slots), tmp_faces);
                Kokkos::deep_copy(HostUnmanaged<rtype>(chunk.pinv.data(), tmp_pinv.extent(0)), tmp_pinv);
            }
            chunks.push_back(std::move(chunk));
        }

        /** @brief The packed stencils on the device; each host chunk is released once copied. */
        teno::PackedStencils finish(const std::string & label) {
            teno::PackedStencils out;
            out.shift = shift;
            out.width = width;
            out.single = single;
            const size_t n_slots = slice_start.back() << shift;
            const size_t n_pinv = teno::PackedStencils::pinv_elements(n_slots * width, single);
            if (!memory.empty()) {
                memory[0]->ensure(n_slots * sizeof(int32_t));
                memory[1]->ensure(n_slots * sizeof(int32_t));
                memory[2]->ensure(n_pinv * sizeof(rtype));
                out.cells = Kokkos::View<int32_t *>(static_cast<int32_t *>(memory[0]->data()), n_slots);
                out.faces = Kokkos::View<int32_t *>(static_cast<int32_t *>(memory[1]->data()), n_slots);
                out.pinv = Kokkos::View<rtype *>(static_cast<rtype *>(memory[2]->data()), n_pinv);
                out.slice_start = Kokkos::View<uint64_t *>(label + "_slice_start", slice_start.size());
                Kokkos::deep_copy(out.slice_start, HostUnmanaged<const uint64_t>(slice_start.data(), slice_start.size()));
                return out;
            }
            out.cells = Kokkos::View<int32_t *>(Kokkos::view_alloc(Kokkos::WithoutInitializing, label), n_slots);
            out.faces = Kokkos::View<int32_t *>(Kokkos::view_alloc(Kokkos::WithoutInitializing, label + "_face"), n_slots);
            out.pinv = Kokkos::View<rtype *>(Kokkos::view_alloc(Kokkos::WithoutInitializing, label + "_pinv"), n_pinv);
            for (Chunk & chunk : chunks) {
                const size_t b = chunk.slot0 << shift;
                const std::pair<size_t, size_t> slots(b, b + chunk.cells.size());
                Kokkos::deep_copy(Kokkos::subview(out.cells, slots),
                                  HostUnmanaged<int32_t>(chunk.cells.data(), chunk.cells.size()));
                Kokkos::deep_copy(Kokkos::subview(out.faces, slots),
                                  HostUnmanaged<int32_t>(chunk.faces.data(), chunk.faces.size()));
                const size_t n_bytes = chunk.cells.size() * width * value_bytes();
                Kokkos::deep_copy(DeviceUnmanaged<char>(reinterpret_cast<char *>(out.pinv.data()) + b * width * value_bytes(),
                                                        n_bytes),
                                  HostUnmanaged<char>(reinterpret_cast<char *>(chunk.pinv.data()), n_bytes));
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
            const size_t vb = packed.single ? sizeof(float) : sizeof(rtype);
            const size_t n_values = (slots.second - slots.first) * packed.width;
            std::vector<int32_t> h_cells(slots.second - slots.first), h_faces(h_cells.size());
            std::vector<char> h_pinv_bytes(n_values * vb);
            Kokkos::deep_copy(HostUnmanaged<int32_t>(h_cells.data(), h_cells.size()), Kokkos::subview(packed.cells, slots));
            Kokkos::deep_copy(HostUnmanaged<int32_t>(h_faces.data(), h_faces.size()), Kokkos::subview(packed.faces, slots));
            if (n_values > 0) {
                Kokkos::deep_copy(HostUnmanaged<char>(h_pinv_bytes.data(), h_pinv_bytes.size()),
                                  DeviceUnmanaged<char>(reinterpret_cast<char *>(packed.pinv.data()) +
                                                            slots.first * packed.width * vb,
                                                        h_pinv_bytes.size()));
            }
            auto h_pinv = [&](const size_t k) {
                return packed.single ? rtype(reinterpret_cast<const float *>(h_pinv_bytes.data())[k])
                                     : reinterpret_cast<const rtype *>(h_pinv_bytes.data())[k];
            };
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
                        (t.*pinv)[s * packed.width + l] = h_pinv((((start + s) * packed.width + l) << sh) + lane);
                    }
                }
            }
        }

    private:
        struct Chunk {
            uint64_t slot0 = 0;
            PageArray<int32_t> cells, faces;
            PageArray<rtype> pinv;  // floats when single
        };

        size_t value_bytes() const { return single ? sizeof(float) : sizeof(rtype); }

        void put_value(rtype * p, const size_t k, const rtype v) const {
            if (single) {
                reinterpret_cast<float *>(p)[k] = static_cast<float>(v);
            } else {
                p[k] = v;
            }
        }

        uint8_t shift;
        uint8_t width;
        bool single;  // pseudo-inverses stored in single precision
        IndexRow cells, faces;
        ValueRow pinv;
        std::vector<uint64_t> slice_start = {0};  // (slice + 1)
        std::vector<Chunk> chunks;
        std::vector<std::shared_ptr<GrowingBuffer>> memory;  // streamed: cells, faces, pinv
};

/**
 * @brief Per-cell rows of cells(j) from slot j of a batch, j < n: scale, basis
 *        means, smoothness-indicator matrix and stencil sizes.
 */
void store_cell_rows(TENO & scheme, const Kokkos::View<uint32_t *> & cells, const teno_setup::TableBatch & batch,
                     const uint32_t n) {
    auto scale = scheme.scale;
    auto basis_mean = scheme.basis_mean;
    auto si = scheme.si_matrix;
    auto large_size = scheme.stencil_large_size;
    auto small_size = scheme.stencil_small_size;
    Kokkos::parallel_for("teno_store_rows", Kokkos::RangePolicy<>(0, n), KOKKOS_LAMBDA(const uint32_t j) {
        const uint32_t c = cells(j);
        scale(c) = batch.scale.lane(j)[0];
        const auto mean = batch.basis_mean.lane(j);
        for (size_t l = 0; l < basis_mean.extent(1); l++) basis_mean(c, l) = mean[l];
        const auto s = batch.si.lane(j);
        for (size_t k = 0; k < si.extent(1); k++) si(c, k) = s[k];
        large_size(c) = batch.large_size(j);
        for (int k = 0; k < teno::MAX_FACES; k++) small_size(c, k) = batch.small_size(j, k);
    });
    auto h_depth = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),
                                                       Kokkos::subview(batch.gather_depth, std::make_pair(0u, n)));
    auto h_cells = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), Kokkos::subview(cells, std::make_pair(0u, n)));
    for (uint32_t j = 0; j < n; j++) scheme.gather_depth[h_cells(j)] = h_depth(j);
}

/**
 * @brief Packed stencils in which the cells with slot_of(c) >= 0 take their
 *        stencils (sizes(slot) entries) from that slot of a batch and every
 *        other cell keeps its own (old_size(c) entries). Slices are sized as
 *        a packing from scratch would size them.
 */
template <typename OldSize>
teno::PackedStencils repack(const teno::PackedStencils & old, const uint32_t n_cells, OldSize old_size,
                            const Kokkos::View<int32_t *> & slot_of, const Kokkos::View<uint16_t *> & sizes,
                            const teno_setup::SlotArray<int32_t> & batch_cells,
                            const teno_setup::SlotArray<int32_t> & batch_faces, const teno_setup::SlotArray<rtype> & batch_pinv,
                            const std::string & label) {
    const uint8_t sh = old.shift;
    const uint8_t w = old.width;
    const uint32_t n_slices = (n_cells + (1u << sh) - 1) >> sh;
    Kokkos::View<uint64_t *> start(label + "_slice_start", n_slices + 1);
    Kokkos::parallel_for("teno_repack_sizes", Kokkos::RangePolicy<>(0, n_slices), KOKKOS_LAMBDA(const uint32_t sl) {
        uint64_t largest = 0;
        for (uint32_t c = sl << sh; c < ((sl + 1) << sh) && c < n_cells; c++) {
            const int32_t j = slot_of(c);
            const uint64_t size = j >= 0 ? sizes(j) : old_size(c);
            largest = size > largest ? size : largest;
        }
        start(sl) = largest;
    });
    Kokkos::parallel_scan("teno_repack_scan", Kokkos::RangePolicy<>(0, n_slices + 1),
                          KOKKOS_LAMBDA(const uint32_t sl, uint64_t & sum, const bool final) {
        const uint64_t size = sl < n_slices ? start(sl) : 0;
        if (final) start(sl) = sum;
        sum += size;
    });
    uint64_t total = 0;
    Kokkos::deep_copy(total, Kokkos::subview(start, n_slices));
    teno::PackedStencils out;
    out.shift = sh;
    out.width = w;
    out.single = old.single;
    out.slice_start = start;
    out.cells = Kokkos::View<int32_t *>(Kokkos::view_alloc(Kokkos::WithoutInitializing, label), total << sh);
    out.faces = Kokkos::View<int32_t *>(Kokkos::view_alloc(Kokkos::WithoutInitializing, label + "_face"), total << sh);
    out.pinv = Kokkos::View<rtype *>(Kokkos::view_alloc(Kokkos::WithoutInitializing, label + "_pinv"),
                                     teno::PackedStencils::pinv_elements((total << sh) * w, old.single));
    const bool sg = old.single;
    auto new_cells = out.cells;
    auto new_faces = out.faces;
    auto new_pinv = out.pinv;
    auto old_start = old.slice_start;
    auto old_cells = old.cells;
    auto old_faces = old.faces;
    auto old_pinv = old.pinv;
    Kokkos::parallel_for("teno_repack", Kokkos::RangePolicy<>(0, size_t(n_slices) << sh), KOKKOS_LAMBDA(const uint32_t c) {
        const uint32_t sl = c >> sh;
        const uint32_t lane = c & ((1u << sh) - 1);
        const uint64_t s0 = start(sl);
        const uint64_t len = start(sl + 1) - s0;
        const int32_t j = c < n_cells ? slot_of(c) : -1;
        const uint64_t size = c >= n_cells ? 0 : (j >= 0 ? sizes(j) : old_size(c));
        const uint64_t o0 = old_start(sl);
        const auto lc = batch_cells.lane(j >= 0 ? j : 0);
        const auto lf = batch_faces.lane(j >= 0 ? j : 0);
        const auto lp = batch_pinv.lane(j >= 0 ? j : 0);
        for (uint64_t s = 0; s < len; s++) {
            const uint64_t k = ((s0 + s) << sh) + lane;
            const uint64_t ko = ((o0 + s) << sh) + lane;
            if (s >= size) {
                new_cells(k) = -1;
                new_faces(k) = -1;
            } else {
                new_cells(k) = j >= 0 ? lc[s] : old_cells(ko);
                new_faces(k) = j >= 0 ? lf[s] : old_faces(ko);
            }
            for (uint32_t l = 0; l < w; l++) {
                const uint64_t kp = (((s0 + s) * w + l) << sh) + lane;
                const uint64_t ko_p = (((o0 + s) * w + l) << sh) + lane;
                if (sg) {
                    const float old_v = reinterpret_cast<const float *>(old_pinv.data())[ko_p];
                    reinterpret_cast<float *>(new_pinv.data())[kp] =
                        s >= size ? 0.0f : (j >= 0 ? static_cast<float>(lp[s * w + l]) : old_v);
                } else {
                    new_pinv(kp) = s >= size ? rtype(0) : (j >= 0 ? lp[s * w + l] : old_pinv(ko_p));
                }
            }
        }
    });
    Kokkos::fence();
    return out;
}

/**
 * @brief Replace the tables of cells(j), j < n (distinct reconstructed cells),
 *        by slot j of a batch, keeping every other cell's.
 */
void merge_tables(TENO & scheme, const Kokkos::View<uint32_t *> & cells, const teno_setup::TableBatch & batch,
                  const uint32_t n) {
    const uint32_t n_cells = scheme.scale.extent(0);
    Kokkos::View<int32_t *> slot_of("teno_slot_of", n_cells);
    Kokkos::deep_copy(slot_of, -1);
    Kokkos::parallel_for("teno_slot_of", Kokkos::RangePolicy<>(0, n),
                         KOKKOS_LAMBDA(const uint32_t j) { slot_of(cells(j)) = j; });
    auto large_size = scheme.stencil_large_size;
    auto small_size = scheme.stencil_small_size;
    teno::PackedStencils large = repack(
        scheme.stencil_large, n_cells, KOKKOS_LAMBDA(const uint32_t c) { return uint64_t(large_size(c)); }, slot_of,
        batch.large_size, batch.large_cells, batch.large_faces, batch.large_pinv, "teno_stencil_large");
    teno::PackedStencils small = repack(
        scheme.stencil_small, n_cells,
        KOKKOS_LAMBDA(const uint32_t c) {
            uint64_t total = 0;
            for (int k = 0; k < teno::MAX_FACES; k++) total += small_size(c, k);
            return total;
        },
        slot_of, batch.small_total, batch.small_cells, batch.small_faces, batch.small_pinv, "teno_stencil_small");
    scheme.stencil_large = teno::PackedStencils();
    scheme.stencil_small = teno::PackedStencils();
    scheme.table_memory.clear();
    scheme.stencil_large = large;
    scheme.stencil_small = small;
    store_cell_rows(scheme, cells, batch, n);
    Kokkos::fence();
}

/** @brief Batch (in the device's layout, on the host) holding the tables of a chunk of cells. */
teno_setup::TableBatchT<Kokkos::HostSpace> host_batch(const std::vector<CellTables> & tables, const uint8_t nk) {
    uint16_t max_large = 1, max_small = 1;
    for (const CellTables & t : tables) {
        max_large = std::max<uint16_t>(max_large, t.large_cells.size());
        max_small = std::max<uint16_t>(max_small, t.small_cells.size());
    }
    const uint32_t n = tables.size();
    teno_setup::TableBatchT<Kokkos::HostSpace> b(n, nk, max_large, max_small, teno_setup::batch_interleaved());
    for (uint32_t j = 0; j < n; j++) {
        const CellTables & t = tables[j];
        b.status(j) = 0;
        b.gather_depth(j) = t.gather_depth;
        b.large_size(j) = t.large_cells.size();
        b.small_total(j) = t.small_cells.size();
        for (int k = 0; k < teno::MAX_FACES; k++) b.small_size(j, k) = t.small_size[k];
        b.scale.lane(j)[0] = t.scale;
        for (size_t l = 0; l < t.basis_mean.size(); l++) b.basis_mean.lane(j)[l] = t.basis_mean[l];
        for (size_t k = 0; k < t.si.size(); k++) b.si.lane(j)[k] = t.si[k];
        for (size_t s = 0; s < t.large_cells.size(); s++) {
            b.large_cells.lane(j)[s] = t.large_cells[s];
            b.large_faces.lane(j)[s] = t.large_faces[s];
        }
        for (size_t k = 0; k < t.large_pinv.size(); k++) b.large_pinv.lane(j)[k] = t.large_pinv[k];
        for (size_t s = 0; s < t.small_cells.size(); s++) {
            b.small_cells.lane(j)[s] = t.small_cells[s];
            b.small_faces.lane(j)[s] = t.small_faces[s];
        }
        for (size_t k = 0; k < t.small_pinv.size(); k++) b.small_pinv.lane(j)[k] = t.small_pinv[k];
    }
    return b;
}

PackedRows large_rows(const TENO & scheme) {
    return PackedRows(scheme.slice_shift, scheme.n_dof_large, &CellTables::large_cells, &CellTables::large_faces,
                      &CellTables::large_pinv, scheme.single_tables);
}

PackedRows small_rows(const TENO & scheme) {
    return PackedRows(scheme.slice_shift, teno::NK_SMALL, &CellTables::small_cells, &CellTables::small_faces,
                      &CellTables::small_pinv, scheme.single_tables);
}

/** @brief Moves per-cell tables, chunk by chunk, into TENO's device arrays. */
class TableBuilder {
    public:
        TableBuilder(TENO & owner, const uint32_t n_cells)
            : scheme(owner), large(large_rows(owner)), small(small_rows(owner)) {
            const uint8_t nk = scheme.n_dof_large;
            scheme.scale = Kokkos::View<rtype *>("teno_scale", n_cells);
            scheme.basis_mean = Kokkos::View<rtype **>("teno_basis_mean", n_cells, nk);
            // 3D: the setup hands over its moments; tables from the host (cache) bring theirs
            if constexpr (N_DIM == 2) scheme.si_matrix = Kokkos::View<rtype **>("teno_si_matrix", n_cells, nk * (nk + 1) / 2);
            if constexpr (N_DIM == 3) scheme.moments = teno::Moments();
            scheme.stencil_large_size = Kokkos::View<uint16_t *>("teno_stencil_large_size", n_cells);
            scheme.stencil_small_size = Kokkos::View<uint16_t **>("teno_stencil_small_size", n_cells, teno::MAX_FACES);
            scheme.gather_depth.assign(n_cells, 0);
            if (GrowingBuffer::supported()) {
                // The searches cap central stencils at 3.5 nk + 64 entries and sectors
                // at twice small_stencil_size; the reservation is address space only
                large.stream(n_cells, 4 * size_t(nk) + 64);
                small.stream(n_cells, teno::MAX_FACES * 2 * size_t(scheme.n_stencil_small));
            }
        }

        /** @brief Tables of cells [c0, c0 + tables.size()), the chunk after the previous one. */
        void add(const uint32_t c0, const std::vector<CellTables> & tables) {
            const uint32_t n = tables.size();
            Kokkos::View<rtype *>::host_mirror_type h_scale("teno_scale_rows", n);
            Kokkos::View<rtype **>::host_mirror_type h_mean("teno_basis_mean_rows", n, scheme.basis_mean.extent(1));
            Kokkos::View<rtype **>::host_mirror_type h_si("teno_si_rows", n, scheme.si_matrix.extent(1));
            if constexpr (N_DIM == 3) {
                if (!scheme.moments.is_allocated()) {
                    scheme.moments = teno::Moments("teno_moments", scheme.scale.extent(0), teno_setup::n_moments(scheme.degree));
                }
            }
            teno::Moments::host_mirror_type h_moments("teno_moments_rows", n, scheme.moments.extent(1));
            Kokkos::View<uint16_t *>::host_mirror_type h_large_size("teno_large_size_rows", n);
            Kokkos::View<uint16_t **>::host_mirror_type h_small_size("teno_small_size_rows", n, teno::MAX_FACES);
            Kokkos::parallel_for("teno_rows", Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(0, n),
                                 [&](const uint32_t c) {
                const CellTables & t = tables[c];
                scheme.gather_depth[c0 + c] = t.gather_depth;
                h_scale(c) = t.scale;
                for (size_t l = 0; l < t.basis_mean.size(); l++) h_mean(c, l) = t.basis_mean[l];
                for (size_t k = 0; k < t.si.size(); k++) h_si(c, k) = t.si[k];
                for (size_t k = 0; k < t.moments.size(); k++) h_moments(c, k) = t.moments[k];
                h_large_size(c) = t.large_cells.size();
                for (uint8_t k = 0; k < teno::MAX_FACES; k++) h_small_size(c, k) = t.small_size[k];
            });
            upload_rows(scheme.scale, c0, h_scale);
            upload_rows(scheme.basis_mean, c0, h_mean);
            if constexpr (N_DIM == 2) {
                upload_rows(scheme.si_matrix, c0, h_si);
            } else {
                upload_rows(scheme.moments, c0, h_moments);
            }
            upload_rows(scheme.stencil_large_size, c0, h_large_size);
            upload_rows(scheme.stencil_small_size, c0, h_small_size);
            large.add(c0, tables);
            small.add(c0, tables);
        }

        /** @brief Tables of cells [c0, c0 + n) = cells(0..n), the chunk after the previous one, from a batch. */
        void add_batch(const uint32_t c0, const Kokkos::View<uint32_t *> & cells, const teno_setup::TableBatch & batch,
                       const uint32_t n) {
            store_cell_rows(scheme, cells, batch, n);
            large.add_batch(c0, n, batch.large_size, batch.large_cells, batch.large_faces, batch.large_pinv);
            small.add_batch(c0, n, batch.small_total, batch.small_cells, batch.small_faces, batch.small_pinv);
        }

        void finish() {
            scheme.stencil_large = large.finish("teno_stencil_large");
            scheme.stencil_small = small.finish("teno_stencil_small");
            scheme.table_memory = large.device_memory();
            scheme.table_memory.insert(scheme.table_memory.end(), small.device_memory().begin(),
                                       small.device_memory().end());
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
    Kokkos::View<rtype **>::host_mirror_type h_si;
    teno::Moments::host_mirror_type h_moments;
    if constexpr (N_DIM == 2) {
        h_si = download_rows(scheme.si_matrix, c0, c1);
    } else {
        h_moments = download_rows(scheme.moments, c0, c1);
    }
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
        t.moments.resize(h_moments.extent(1));
        for (size_t k = 0; k < t.moments.size(); k++) t.moments[k] = h_moments(c, k);
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

/**
 * @brief Run precompute(i, tables, failed_large, invalid_small) over the given
 *        cells chunk by chunk, replacing their tables and keeping the others'.
 */
template <typename F>
void rebuild_in_chunks(TENO & scheme, const std::vector<uint32_t> & cells, F && precompute, uint32_t & n_failed_large,
                       uint32_t & n_invalid_small) {
    std::vector<CellTables> chunk;
    for (size_t c0 = 0; c0 < cells.size(); c0 += CHUNK_CELLS) {
        const uint32_t n = std::min<size_t>(CHUNK_CELLS, cells.size() - c0);
        chunk.assign(n, CellTables());
        uint32_t failed = 0, invalid = 0;
        Kokkos::parallel_reduce("teno_precompute",
                                Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace, Kokkos::Schedule<Kokkos::Dynamic>>(0, n),
                                [&](const uint32_t j, uint32_t & failed_large, uint32_t & invalid_small) {
            precompute(cells[c0 + j], chunk[j], failed_large, invalid_small);
        }, failed, invalid);
        n_failed_large += failed;
        n_invalid_small += invalid;
        if (n_failed_large > 0) continue;
        Kokkos::View<uint32_t *> d_cells("teno_rebuilt_cells", n);
        Kokkos::deep_copy(d_cells, HostUnmanaged<const uint32_t>(cells.data() + c0, n));
        merge_tables(scheme, d_cells, teno_setup::device_copy(host_batch(chunk, scheme.n_dof_large)), n);
    }
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
    troubled_capacity = find_real_or(input, "troubled_capacity", troubled_capacity);
    if (!(troubled_capacity > 0.0_r && troubled_capacity <= 1.0_r)) {
        throw std::runtime_error("TENO: troubled_capacity must be in (0, 1].");
    }
    cache_file = toml::find_or<std::string>(input, "cache_file", "");
    single_tables = toml::find_or<bool>(input, "single_precision_tables", false);
    if (single_tables && N_DIM != 3) throw std::runtime_error("TENO: single_precision_tables is for 3D runs.");
    // The cache then holds the tables as the runs use them
    cache_single = toml::find_or<bool>(input, "cache_single_precision", false) || single_tables;
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
    if constexpr (N_DIM == 3) {
        init_face_quadrature_3d(order);
        si_terms = teno_setup::smoothness_terms(degree);
    }

    if (mesh->axisymmetric) {
        // Exact to degree >= the reconstruction's, for the geometric source
        const TriangleRule rule((degree + 3) / 2);
        cell_rule = Kokkos::View<rtype *[3]>("teno_cell_rule", rule.w.size());
        auto h_rule = Kokkos::create_mirror_view(cell_rule);
        for (size_t q = 0; q < rule.w.size(); q++) {
            h_rule(q, 0) = rule.xi[q];
            h_rule(q, 1) = rule.eta[q];
            h_rule(q, 2) = rule.w[q];
        }
        Kokkos::deep_copy(cell_rule, h_rule);
    }

    cache_loaded = !cache_file.empty() && load_cache();
    if (!cache_loaded) {
        compute_stencils_and_matrices();
        // Runs that write the cache use the tables as they will be read back
        if (single_tables || (cache_single && !cache_file.empty())) round_to_single();
    }
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
    if (!setup_status.empty()) out.emplace_back("TENO setup", setup_status);
    if (!cache_status.empty()) out.emplace_back("TENO cache", cache_status);
    return out;
}

uint8_t TENO::n_face_quadrature_points() const {
    if constexpr (N_DIM == 3) return face_quad_weights.extent(1);
    return quadrature_face.h_points.extent(0);
}

void TENO::compute_stencils_and_matrices(const std::vector<uint32_t> * subset) {
    if constexpr (N_DIM == 3) {
        compute_stencils_and_matrices_3d(subset);
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
    const bool axisymmetric = mesh->axisymmetric;
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
                // Axisymmetric cell averages weight by the radius; |r| makes the
                // mirror image of a cell across the axis carry its mirrored average
                if (axisymmetric) w *= std::abs(y0 + h * y);
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
        // The smallest stencil within MAX_LEBESGUE_2D, else the best conditioned one
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
            if (lebesgue <= MAX_LEBESGUE_2D) break;
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
                // Entries past nss_max + 1 cannot change the stencil or the tie tests
                if (sector.size() > nss_max) break;
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
    Kokkos::Timer timer;
    if (subset == nullptr) {
        largest_stencil = precompute_in_chunks(*this, mesh->n_reconstructed(), precompute, n_failed_large, n_invalid_small);
    } else {
        rebuild_in_chunks(*this, *subset, precompute, n_failed_large, n_invalid_small);
    }
    setup_status = logging::format("host, %.2f s", timer.seconds());

    if (n_failed_large > 0) {
        throw std::runtime_error("TENO: could not build a full-rank large stencil for " +
                                 std::to_string(n_failed_large) + " cells (mesh too small for this order?).");
    }
    n_sector_unavailable = n_invalid_small;
}

namespace {

/** @brief cells(t) = c0 + t for t < n. */
void fill_range(const Kokkos::View<uint32_t *> & cells, const uint32_t c0, const uint32_t n) {
    Kokkos::parallel_for("teno_setup_range", Kokkos::RangePolicy<>(0, n),
                         KOKKOS_LAMBDA(const uint32_t t) { cells(t) = c0 + t; });
}

/** @brief Failed central stencils, left-out sector stencils and the largest central stencil of slots 0..n. */
void batch_counts(const teno_setup::TableBatch & out, const uint32_t n, uint32_t & failed, uint32_t & invalid,
                  uint16_t & largest) {
    Kokkos::parallel_reduce("teno_setup_counts", Kokkos::RangePolicy<>(0, n),
                            KOKKOS_LAMBDA(const uint32_t t, uint32_t & f, uint32_t & v, uint16_t & m) {
        f += out.failed_large(t);
        v += out.invalid_small(t);
        m = out.large_size(t) > m ? out.large_size(t) : m;
    }, failed, invalid, Kokkos::Max<uint16_t>(largest));
}

} // namespace

void TENO::compute_stencils_and_matrices_3d(const std::vector<uint32_t> * subset) {
    const uint32_t n_cells = mesh->n_cells;
    for (uint32_t i = 0; i < n_cells; i++) {
        if (mesh->h_n_faces_of_cell(i) > teno::MAX_FACES) {
            throw std::runtime_error("TENO supports cells with at most " +
                                     std::to_string(teno::MAX_FACES) + " faces.");
        }
    }
    Kokkos::Timer timer;
    teno_setup::Options options;
    options.degree = degree;
    options.nk = n_dof_large;
    options.ns = n_stencil_large;
    options.nss = n_stencil_small;
    options.max_condition = double(max_condition);
    options.batch_cells = setup_batch_cells;
    options.host_only = setup_on_host;
    options.cells = subset;
    teno_setup::Setup3DHandle setup(*mesh, boundaries, face_quad_points, face_quad_weights, options);

    // Cells per batch: whole slices of the packed stencils
    const uint32_t slice = 1u << slice_shift;
    const uint32_t n_target = subset ? static_cast<uint32_t>(subset->size()) : mesh->n_reconstructed();
    uint32_t batch = std::max(setup.batch_cells() / slice * slice, slice);
    batch = std::min(batch, (n_target + slice - 1) / slice * slice);
    batch = std::max(batch, slice);
    teno_setup::TableBatch out(batch, n_dof_large, setup.max_large(), setup.max_small(), teno_setup::batch_interleaved());
    Kokkos::View<uint32_t *> cells("teno_setup_batch_cells", batch);
    std::unique_ptr<TableBuilder> builder;
    if (subset == nullptr) builder = std::make_unique<TableBuilder>(*this, n_target);
    uint32_t n_failed_large = 0, n_invalid_small = 0;
    uint16_t largest = 0;
    double t_pack = 0.0;
    for (uint32_t c0 = 0; c0 < n_target; c0 += batch) {
        const uint32_t n = std::min(batch, n_target - c0);
        if (subset == nullptr) {
            fill_range(cells, c0, n);
        } else {
            Kokkos::deep_copy(Kokkos::subview(cells, std::make_pair(0u, n)),
                              HostUnmanaged<const uint32_t>(subset->data() + c0, n));
        }
        setup.compute(cells, n, out);
        uint32_t failed = 0, invalid = 0;
        uint16_t batch_largest = 0;
        batch_counts(out, n, failed, invalid, batch_largest);
        n_failed_large += failed;
        n_invalid_small += invalid;
        largest = std::max(largest, batch_largest);
        if (n_failed_large > 0) continue;
        Kokkos::Timer pack_timer;
        if (subset == nullptr) {
            builder->add_batch(c0, cells, out, n);
        } else {
            merge_tables(*this, cells, out, n);
        }
        t_pack += pack_timer.seconds();
    }
    if (subset == nullptr) {
        if (n_failed_large == 0) builder->finish();
        largest_stencil = largest;
    }
    // Every cell's moments are a function of its geometry alone
    if (n_failed_large == 0) moments = setup.moments(scale.extent(0));
    Kokkos::fence();
    const teno_setup::Timings & t = setup.timings();
    setup_status = logging::format("%s, %.2f s (geometry %.2f s, tables %.2f s, packing %.2f s)",
                                   setup_on_host ? "host" : Kokkos::DefaultExecutionSpace::name(), timer.seconds(),
                                   t.geometry, t.tables, t_pack);
    if (t.n_host_cells > 0) {
        setup_status += logging::format(", %llu cells redone on the host in %.2f s",
                                        static_cast<unsigned long long>(t.n_host_cells), t.host_cells);
    }
    setup_producer = setup_on_host ? "host" : Kokkos::DefaultExecutionSpace::name();

    if (n_failed_large > 0) {
        throw std::runtime_error("TENO: could not build a full-rank large stencil for " +
                                 std::to_string(n_failed_large) + " cells (mesh too small for this order?).");
    }
    if (subset == nullptr) n_sector_unavailable = n_invalid_small;
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
template <uint8_t DEG, bool PRIM, typename TABLE = rtype>
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
    Kokkos::View<rtype **> si_matrix;         // 2D
    teno::Moments moments;                    // 3D, with si_terms: the smoothness matrices
    teno::SmoothnessTerms si_terms;
    Kokkos::View<rtype *> sigma_out;
    Kokkos::View<rtype ***> coeffs;           // (queue slot, l, var): central coefficients of queued cells
    Kokkos::View<rtype ****> small_coeffs;    // (queue slot, sector, l, var): sector coefficients of queued cells
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
        const auto stencil = stencil_large.template row<TABLE>(i_cell);
        for (uint16_t s = 0; s < stencil_large_size(i_cell); s++) {
            rtype U[N_CONSERVATIVE];
            entry_conservatives(stencil.cell(s), stencil.face(s), U);
            FOR_I_DIM {
                const rtype P = stencil.template pinv<NK>(s, i) * inv_h;
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
        const auto stencil = stencil_large.template row<TABLE>(i_cell);
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
                const rtype P = stencil.template pinv<NK>(s, l);
                FOR_I_CONSERVATIVE aK[l][i] += P * (U[i] - U0[i]);
            }
        }
        const rtype sigma = PRIM ? Kokkos::fmax(g_m2, m_m2) / ns : g_m2 / ns;
        sigma_out(i_cell) = sigma;
        if (sigma >= sigma_threshold) {
            const uint32_t j = Kokkos::atomic_fetch_add(&n_troubled(), 1u);
            troubled_cells(j) = i_cell;
            if (j < coeffs.extent(0)) {
                for (uint8_t l = 0; l < NK; l++) {
                    FOR_I_CONSERVATIVE coeffs(j, l, i) = aK[l][i];
                }
            }
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

    // SmoothTeamPass (GPUs): SmoothPass with N_CONSERVATIVE threads per cell,
    // so that the central coefficients fit in registers. Over the stencil each
    // thread accumulates a chunk of them; the coefficients then go through
    // team scratch to the face points, which the cell's threads share out.
    // The arithmetic is SmoothPass's, with compile-time chunk bounds as its
    // loop indices: runtime ones change which multiplies the compiler fuses
    // into FMAs, and with them the results' last bits
    struct SmoothTeamPass {};
    static constexpr uint32_t TEAM_CELLS = 32;
    // Measured on A100s: faster than SmoothPass only when its coefficients spill
    static constexpr bool USE_TEAM = N_DIM == 3 && DEG >= 4;
    static constexpr uint8_t PARTS = N_CONSERVATIVE;  // threads per cell
    static constexpr uint32_t TEAM_SIZE = TEAM_CELLS * PARTS;
    static constexpr uint8_t CHUNK = (NK + PARTS - 1) / PARTS;
    static constexpr size_t TEAM_SCRATCH = sizeof(rtype) * NK * N_CONSERVATIVE * TEAM_CELLS + 2 * sizeof(int) * TEAM_CELLS;
    uint32_t n_smooth = 0;  // cells to reconstruct when cells is empty

    template <typename F>
    KOKKOS_INLINE_FUNCTION
    static void with_chunk(const uint8_t c, F && f) {
        switch (c) {
            case 0: f(std::integral_constant<uint8_t, 0>()); break;
            case 1: if constexpr (PARTS > 1) f(std::integral_constant<uint8_t, 1>()); break;
            case 2: if constexpr (PARTS > 2) f(std::integral_constant<uint8_t, 2>()); break;
            case 3: if constexpr (PARTS > 3) f(std::integral_constant<uint8_t, 3>()); break;
            default: if constexpr (PARTS > 4) f(std::integral_constant<uint8_t, 4>()); break;
        }
    }

    /**
     * @brief Chunk C of the central coefficients of cell i_cell, into aK_team
     *        (l, var). With QUEUE, chunk 0 also finds sigma, queues the cell if
     *        troubled and leaves in smooth_team[lane] -1 if smooth, else its
     *        place in the queue.
     */
    template <uint8_t C, bool QUEUE = true>
    KOKKOS_INLINE_FUNCTION
    void smooth_coefficients(const uint32_t i_cell, const rtype * U0, rtype * aK_team, int * smooth_team,
                             const uint32_t lane) const {
        constexpr uint8_t L0 = C * CHUNK;
        constexpr uint8_t L1 = (L0 + CHUNK < NK) ? L0 + CHUNK : NK;
        const uint16_t ns = stencil_large_size(i_cell);
        rtype aK[CHUNK][N_CONSERVATIVE] = {};
        rtype g_mean = 0.0, g_m2 = 0.0;
        rtype m_mean = 0.0, m_m2 = 0.0;
        const auto stencil = stencil_large.template row<TABLE>(i_cell);
        for (uint16_t s = 0; s < ns; s++) {
            rtype U[N_CONSERVATIVE];
            entry_conservatives(stencil.cell(s), stencil.face(s), U);
            if constexpr (C == 0 && QUEUE) {
                const rtype W0_rho = W(i_cell, 0);
                const rtype g = Kokkos::fabs(U[0] - W0_rho) / W0_rho;
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
            }
            for (uint8_t l = L0; l < L1; l++) {
                const rtype P = stencil.template pinv<NK>(s, l);
                FOR_I_CONSERVATIVE aK[l - L0][i] += P * (U[i] - U0[i]);
            }
        }
        for (uint8_t l = L0; l < L1; l++) {
            FOR_I_CONSERVATIVE aK_team[(l * N_CONSERVATIVE + i) * TEAM_CELLS + lane] = aK[l - L0][i];
        }
        if constexpr (C == 0 && QUEUE) {
            const rtype sigma = PRIM ? Kokkos::fmax(g_m2, m_m2) / ns : g_m2 / ns;
            sigma_out(i_cell) = sigma;
            const bool smooth = sigma < sigma_threshold;
            const uint32_t j = smooth ? 0 : Kokkos::atomic_fetch_add(&n_troubled(), 1u);
            if (!smooth) troubled_cells(j) = i_cell;
            smooth_team[lane] = smooth ? -1 : static_cast<int>(j);
        }
    }

    /** @brief Thread part's chunk of the central coefficients in aK_team (lane), into coeffs(slot). */
    KOKKOS_INLINE_FUNCTION
    void store_coefficients(const uint8_t part, const uint32_t slot, const rtype * aK_team, const uint32_t lane) const {
        with_chunk(part, [&](auto C) {
            constexpr uint8_t L0 = decltype(C)::value * CHUNK;
            constexpr uint8_t L1 = (L0 + CHUNK < NK) ? L0 + CHUNK : NK;
            for (uint8_t l = L0; l < L1; l++) {
                FOR_I_CONSERVATIVE coeffs(slot, l, i) = aK_team[(l * N_CONSERVATIVE + i) * TEAM_CELLS + lane];
            }
        });
    }

    template <typename Member>
    KOKKOS_INLINE_FUNCTION
    void operator()(SmoothTeamPass, const Member & team) const {
        const uint32_t lane = team.team_rank() % TEAM_CELLS;
        const uint8_t part = team.team_rank() / TEAM_CELLS;
        const uint32_t idx = team.league_rank() * TEAM_CELLS + lane;
        const uint32_t n = cells.extent(0) ? static_cast<uint32_t>(cells.extent(0)) : n_smooth;
        rtype * aK = static_cast<rtype *>(team.team_scratch(0).get_shmem(sizeof(rtype) * NK * N_CONSERVATIVE * TEAM_CELLS));
        int * smooth_team = static_cast<int *>(team.team_scratch(0).get_shmem(sizeof(int) * TEAM_CELLS));
        int * admissible_team = static_cast<int *>(team.team_scratch(0).get_shmem(sizeof(int) * TEAM_CELLS));

        const bool active = idx < n;
        const uint32_t i_cell = active ? (cells.extent(0) ? cells(idx) : idx) : 0;
        rtype U0[N_CONSERVATIVE];
        if (part == 0) {
            smooth_team[lane] = 0;
            admissible_team[lane] = 1;
        }
        team.team_barrier();
        if (active) {
            conservatives(i_cell, U0);
            with_chunk(part, [&](auto C) {
                smooth_coefficients<decltype(C)::value>(i_cell, U0, aK, smooth_team, lane);
            });
        }
        team.team_barrier();
        const int slot = smooth_team[lane];
        const bool smooth = slot < 0;
        if (active && !smooth && slot < static_cast<int>(coeffs.extent(0))) {
            with_chunk(part, [&](auto C) {
                constexpr uint8_t L0 = decltype(C)::value * CHUNK;
                constexpr uint8_t L1 = (L0 + CHUNK < NK) ? L0 + CHUNK : NK;
                for (uint8_t l = L0; l < L1; l++) {
                    FOR_I_CONSERVATIVE coeffs(slot, l, i) = aK[(l * N_CONSERVATIVE + i) * TEAM_CELLS + lane];
                }
            });
        }

        // The cell's face points, shared out among its threads
        const uint8_t n_quad = this->n_quad();
        const uint32_t f_begin = smooth ? offsets_faces_of_cell(i_cell) : 0;
        const uint32_t n_points = smooth ? (offsets_faces_of_cell(i_cell + 1) - f_begin) * n_quad : 0;
        bool admissible = true;
        for (uint32_t p = part; p < n_points; p += PARTS) {
            const uint32_t f = faces_of_cell(f_begin + p / n_quad);
            const uint8_t q = p % n_quad;
            rtype psi[NK];
            if (!face_point_basis(f, q, i_cell, psi)) continue;
            rtype U_f[N_CONSERVATIVE];
            FOR_I_CONSERVATIVE U_f[i] = U0[i];
            for (uint8_t l = 0; l < NK; l++) {
                FOR_I_CONSERVATIVE U_f[i] += aK[(l * N_CONSERVATIVE + i) * TEAM_CELLS + lane] * psi[l];
            }
            rtype Wq[N_CONSERVATIVE];
            to_primitives(U_f, Wq);
            admissible = admissible && (Wq[0] > 0.0_r) && (Wq[N_DIM + 1] > 0.0_r) && Kokkos::isfinite(Wq[N_DIM + 1]);
            const uint8_t side = side_of(f, i_cell);
            FOR_I_CONSERVATIVE face_solution(f, q, side, i) = Wq[i];
        }
        if (!admissible) admissible_team[lane] = 0;
        team.team_barrier();
        if (!admissible_team[lane]) {
            rtype W0[N_CONSERVATIVE];
            FOR_I_CONSERVATIVE W0[i] = W(i_cell, i);
            for (uint32_t p = part; p < n_points; p += PARTS) {
                const uint32_t f = faces_of_cell(f_begin + p / n_quad);
                const uint8_t side = side_of(f, i_cell);
                FOR_I_CONSERVATIVE face_solution(f, p % n_quad, side, i) = W0[i];
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

    /**
     * @brief Smoothness indicator c^T S c of the N coefficients c, for S with
     *        entries S(upper_index(l, m)).
     */
    template <uint8_t N, typename Matrix>
    KOKKOS_INLINE_FUNCTION
    static rtype smoothness(const rtype * c, const Matrix & S) {
        rtype si = 0.0;
        for (uint8_t l = 0; l < N; l++) {
            rtype row = 0.0;
            for (uint8_t m = 0; m < N; m++) {
                row += S((l <= m) ? teno::upper_index(l, m, NK) : teno::upper_index(m, l, NK)) * c[m];
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
        const auto stencil = stencil_small.template row<TABLE>(i_cell);
        for (uint16_t e = 0; e < n_small; e++) {
            rtype U[N_CONSERVATIVE];
            entry_conservatives(stencil.cell(start + e), stencil.face(start + e), U);
            for (uint8_t l = 0; l < teno::NK_SMALL; l++) {
                const rtype P = stencil.template pinv<teno::NK_SMALL>(start + e, l);
                FOR_I_CONSERVATIVE aS[l][i] += P * (U[i] - U0[i]);
            }
        }
        for (uint8_t l = 0; l < teno::NK_SMALL; l++) {
            FOR_I_CONSERVATIVE small_coeffs(j - round_begin, s, l, i) = aS[l][i];
        }
    }

    /**
     * @brief Stencil selection for characteristic variable var on face k of
     *        queued cell j, and that variable at the face's quadrature points,
     *        left in face_solution(f, q, side, var); S(e) is entry e of the
     *        cell's smoothness matrix.
     */
    template <typename Matrix>
    KOKKOS_INLINE_FUNCTION
    void troubled_select(const uint32_t j, const uint8_t k, const uint8_t var, const Matrix & S) const {
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
            FOR_I_CONSERVATIVE a[i] = coeffs(j - round_begin, l, i);
            cK[l] = project(L, var, a);
        }
        bool valid[teno::MAX_FACES] = {};
        rtype cS[teno::MAX_FACES][teno::NK_SMALL];
        for (uint8_t s = 0; s < n_faces; s++) {
            valid[s] = stencil_small_size(i_cell, s) > 0;
            if (!valid[s]) continue;
            for (uint8_t l = 0; l < teno::NK_SMALL; l++) {
                rtype a[N_CONSERVATIVE];
                FOR_I_CONSERVATIVE a[i] = small_coeffs(j - round_begin, s, l, i);
                cS[s][l] = project(L, var, a);
            }
        }

        // gamma_k = 1 / (SI_k + eps)^6, normalized by the largest one so that
        // the weights cannot overflow (even in single precision)
        const rtype sigma = sigma_out(i_cell);
        const rtype cutoff = (C_T > 0.0_r) ? C_T : teno::adaptive_CT(sigma, sigma_threshold, sigma_upper);
        const rtype si_K = smoothness<NK>(cK, S) + eps;
        rtype si_small[teno::MAX_FACES] = {};
        rtype si_min = si_K;
        for (uint8_t s = 0; s < n_faces; s++) {
            if (!valid[s]) continue;
            si_small[s] = smoothness<teno::NK_SMALL>(cS[s], S) + eps;
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
    // The troubled passes work in rounds on the queued cells [round_begin,
    // round_begin + coeffs.extent(0)), which hold the slots of the scratch
    uint32_t round_begin = 0;

    /** @brief End of this round's part of the queue. */
    KOKKOS_INLINE_FUNCTION
    uint32_t round_end() const {
        const uint32_t n = n_troubled();
        const uint32_t end = round_begin + static_cast<uint32_t>(coeffs.extent(0));
        return n < end ? n : end;
    }

    /** @brief Items of this round, per_cell per queued cell. */
    KOKKOS_INLINE_FUNCTION
    uint32_t round_items(const uint32_t per_cell) const {
        const uint32_t end = round_end();
        return end > round_begin ? (end - round_begin) * per_cell : 0;
    }

    // Central coefficients of the queued cells after the first round, which
    // the smooth pass did not keep: TroubledCentralPass with SmoothPass's
    // arithmetic, TroubledCentralTeamPass with SmoothTeamPass's
    struct TroubledCentralPass {};
    struct TroubledCentralTeamPass {};

    KOKKOS_INLINE_FUNCTION
    void operator()(TroubledCentralPass, const uint32_t t) const {
        const uint32_t n = round_items(1);
        for (uint32_t idx = t; idx < n; idx += stride) {
            const uint32_t i_cell = troubled_cells(round_begin + idx);
            rtype U0[N_CONSERVATIVE];
            conservatives(i_cell, U0);
            rtype aK[NK][N_CONSERVATIVE] = {};
            const auto stencil = stencil_large.template row<TABLE>(i_cell);
            for (uint16_t s = 0; s < stencil_large_size(i_cell); s++) {
                rtype U[N_CONSERVATIVE];
                entry_conservatives(stencil.cell(s), stencil.face(s), U);
                for (uint8_t l = 0; l < NK; l++) {
                    const rtype P = stencil.template pinv<NK>(s, l);
                    FOR_I_CONSERVATIVE aK[l][i] += P * (U[i] - U0[i]);
                }
            }
            for (uint8_t l = 0; l < NK; l++) {
                FOR_I_CONSERVATIVE coeffs(idx, l, i) = aK[l][i];
            }
        }
    }

    template <typename Member>
    KOKKOS_INLINE_FUNCTION
    void operator()(TroubledCentralTeamPass, const Member & team) const {
        const uint32_t lane = team.team_rank() % TEAM_CELLS;
        const uint8_t part = team.team_rank() / TEAM_CELLS;
        rtype * aK = static_cast<rtype *>(team.team_scratch(0).get_shmem(sizeof(rtype) * NK * N_CONSERVATIVE * TEAM_CELLS));
        const uint32_t n = round_items(1);
        for (uint32_t base = team.league_rank() * TEAM_CELLS; base < n; base += team.league_size() * TEAM_CELLS) {
            const uint32_t idx = base + lane;
            if (idx < n) {
                const uint32_t i_cell = troubled_cells(round_begin + idx);
                rtype U0[N_CONSERVATIVE];
                conservatives(i_cell, U0);
                with_chunk(part, [&](auto C) {
                    smooth_coefficients<decltype(C)::value, false>(i_cell, U0, aK, nullptr, lane);
                });
                store_coefficients(part, idx, aK, lane);
            }
            team.team_barrier();
        }
    }

    // SourcePass (axisymmetric runs)
    struct SourcePass {};
    Kokkos::View<uint32_t *> offsets_nodes_of_cell{};
    Kokkos::View<uint32_t *> nodes_of_cell{};
    Kokkos::View<rtype *[3]> cell_rule{};              // (q, [xi, eta, w]) on the reference triangle
    Kokkos::View<rtype *, Kokkos::LayoutStride> mu{};  // (cell), empty if inviscid
    Kokkos::View<rtype *> source{};

    /**
     * @brief Geometric source int (p - tau_thetatheta) dA of a smooth cell
     *        from its central polynomial at the points of cell_rule on a fan
     *        of triangles. Troubled cells, and cells whose polynomial is not
     *        admissible at a point, keep the source they have.
     */
    KOKKOS_INLINE_FUNCTION
    void operator()(SourcePass, const uint32_t i_cell) const {
        if constexpr (N_DIM == 2) {
            constexpr uint8_t E = N_DIM + 1;
            if (sigma_out(i_cell) >= sigma_threshold) return;
            rtype U0[N_CONSERVATIVE];
            conservatives(i_cell, U0);
            rtype aK[NK][N_CONSERVATIVE] = {};
            const auto stencil = stencil_large.template row<TABLE>(i_cell);
            for (uint16_t s = 0; s < stencil_large_size(i_cell); s++) {
                rtype U[N_CONSERVATIVE];
                entry_conservatives(stencil.cell(s), stencil.face(s), U);
                for (uint8_t l = 0; l < NK; l++) {
                    const rtype P = stencil.template pinv<NK>(s, l);
                    FOR_I_CONSERVATIVE aK[l][i] += P * (U[i] - U0[i]);
                }
            }
            const rtype h = scale(i_cell);
            const rtype xc = cell_coords(i_cell, 0), yc = cell_coords(i_cell, 1);
            const bool viscous = mu.extent(0) > 0;
            const uint32_t begin = offsets_nodes_of_cell(i_cell);
            const uint32_t n_nodes = offsets_nodes_of_cell(i_cell + 1) - begin;
            const uint32_t n0 = nodes_of_cell(begin);
            const rtype x0 = node_coords(n0, 0), y0 = node_coords(n0, 1);
            rtype sum = 0.0_r;
            for (uint32_t k = 1; k + 1 < n_nodes; k++) {
                const uint32_t na = nodes_of_cell(begin + k), nb = nodes_of_cell(begin + k + 1);
                const rtype ax = node_coords(na, 0) - x0, ay = node_coords(na, 1) - y0;
                const rtype bx = node_coords(nb, 0) - x0, by = node_coords(nb, 1) - y0;
                const rtype det = Kokkos::fabs(ax * by - ay * bx);
                for (uint32_t q = 0; q < cell_rule.extent(0); q++) {
                    const rtype x = x0 + cell_rule(q, 0) * ax + cell_rule(q, 1) * bx;
                    const rtype y = y0 + cell_rule(q, 0) * ay + cell_rule(q, 1) * by;
                    const rtype xi = (x - xc) / h, eta = (y - yc) / h;
                    rtype psi[NK];
                    teno::monomials(DEG, xi, eta, psi);
                    rtype U[N_CONSERVATIVE];
                    FOR_I_CONSERVATIVE U[i] = U0[i];
                    for (uint8_t l = 0; l < NK; l++) {
                        FOR_I_CONSERVATIVE U[i] += aK[l][i] * (psi[l] - basis_mean(i_cell, l));
                    }
                    rtype Wq[N_CONSERVATIVE];
                    to_primitives(U, Wq);
                    if (!(Wq[0] > 0.0_r && Wq[E] > 0.0_r)) return;
                    rtype p_eff = Wq[E];
                    if (viscous) {
                        rtype d_xi[NK], d_eta[NK];
                        teno::monomial_gradients(DEG, xi, eta, d_xi, d_eta);
                        rtype dU[N_CONSERVATIVE][N_DIM] = {};
                        for (uint8_t l = 0; l < NK; l++) {
                            FOR_I_CONSERVATIVE {
                                dU[i][0] += aK[l][i] * d_xi[l];
                                dU[i][1] += aK[l][i] * d_eta[l];
                            }
                        }
                        // du_x/dx and du_r/dr, from the conservative variables unless PRIM
                        rtype g00 = dU[1][0] / h, g11 = dU[2][1] / h;
                        if constexpr (!PRIM) {
                            g00 = (dU[1][0] - Wq[1] * dU[0][0]) / (Wq[0] * h);
                            g11 = (dU[2][1] - Wq[2] * dU[0][1]) / (Wq[0] * h);
                        }
                        const rtype hoop = Wq[2] / y;
                        p_eff -= mu(i_cell) * (2.0_r * hoop - 2.0_r / 3.0_r * (g00 + g11 + hoop));
                    }
                    sum += cell_rule(q, 2) * det * p_eff;
                }
            }
            source(i_cell) = sum;
        }
    }

    KOKKOS_INLINE_FUNCTION
    void operator()(TroubledSectorPass, const uint32_t t) const {
        const uint32_t n = round_items(teno::MAX_FACES);
        for (uint32_t idx = t; idx < n; idx += stride) {
            troubled_sector(round_begin + idx / teno::MAX_FACES, idx % teno::MAX_FACES);
        }
    }

    KOKKOS_INLINE_FUNCTION
    void operator()(TroubledSelectPass, const uint32_t t) const {
        constexpr uint32_t per_cell = teno::MAX_FACES * N_CONSERVATIVE;
        const uint32_t n = round_items(per_cell);
        for (uint32_t idx = t; idx < n; idx += stride) {
            const uint32_t r = idx % per_cell;
            const uint32_t j = round_begin + idx / per_cell;
            const uint32_t i_cell = troubled_cells(j);
            troubled_select(j, r / N_CONSERVATIVE, r % N_CONSERVATIVE, [&](const uint32_t e) { return si_matrix(i_cell, e); });
        }
    }

    // TroubledSelectTeamPass (3D): TroubledSelectPass with one queued cell per
    // team, whose threads first form the cell's smoothness matrix from its
    // moments in team scratch
    struct TroubledSelectTeamPass {};
    static constexpr uint32_t SELECT_TEAM = 32;
    static constexpr uint32_t N_SI = uint32_t(NK) * (NK + 1) / 2;
    static constexpr uint32_t N_MOMENTS = uint32_t(2 * DEG - 1) * (2 * DEG) * (2 * DEG + 1) / 6;
    static constexpr uint32_t SI_GROUP = 4;  // matrix entries per thread at a time
    static constexpr size_t SELECT_SCRATCH = sizeof(rtype) * N_SI + sizeof(double) * N_MOMENTS + 16;

    template <typename Member>
    KOKKOS_INLINE_FUNCTION
    void operator()(TroubledSelectTeamPass, const Member & team) const {
        rtype * S = static_cast<rtype *>(team.team_scratch(0).get_shmem(sizeof(rtype) * N_SI));
        double * mom = static_cast<double *>(team.team_scratch(0).get_shmem(sizeof(double) * N_MOMENTS));
        const uint32_t end = round_end();
        for (uint32_t j = round_begin + team.league_rank(); j < end; j += team.league_size()) {
            const uint32_t i_cell = troubled_cells(j);
            Kokkos::parallel_for(Kokkos::TeamThreadRange(team, N_MOMENTS), [&](const uint32_t k) { mom[k] = moments(i_cell, k); });
            team.team_barrier();
            Kokkos::parallel_for(Kokkos::TeamThreadRange(team, (N_SI + SI_GROUP - 1) / SI_GROUP), [&](const uint32_t g) {
                si_terms.entries<SI_GROUP>(g * SI_GROUP, N_SI, mom, S);
            });
            team.team_barrier();
            Kokkos::parallel_for(Kokkos::TeamThreadRange(team, uint32_t(teno::MAX_FACES) * N_CONSERVATIVE),
                                 [&](const uint32_t r) {
                troubled_select(j, r / N_CONSERVATIVE, r % N_CONSERVATIVE, [&](const uint32_t e) { return S[e]; });
            });
            team.team_barrier();
        }
    }

    KOKKOS_INLINE_FUNCTION
    void operator()(TroubledProjectPass, const uint32_t t) const {
        const uint32_t n = round_items(teno::MAX_FACES);
        for (uint32_t idx = t; idx < n; idx += stride) {
            troubled_project(round_begin + idx / teno::MAX_FACES, idx % teno::MAX_FACES);
        }
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
        const uint32_t n = round_items(1);
        for (uint32_t idx = t; idx < n; idx += stride) troubled_finish(round_begin + idx);
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

template <uint8_t DEG, bool PRIM, typename TABLE>
void TENO::launch_reconstruction(const Kokkos::DefaultExecutionSpace & exec,
                                 Kokkos::View<rtype *[N_CONSERVATIVE]> solution,
                                 Kokkos::View<rtype **[2][N_CONSERVATIVE]> face_solution,
                                 Kokkos::View<uint32_t *> cells, bool troubled_pass) {
    using Functor = TENOFunctor<DEG, PRIM, TABLE>;
    Functor functor{sigma_threshold, sigma_upper, C_T, characteristic, bound_preserving, boundaries.gamma,
                    mesh->offsets_faces_of_cell, mesh->faces_of_cell, mesh->cells_of_face,
                    mesh->offsets_nodes_of_face, mesh->nodes_of_face, mesh->node_coords,
                    mesh->cell_coords, mesh->face_coords, mesh->face_normals, mesh->shifts, mesh->face_shift,
                    quadrature_face.points, face_quad_points, face_quad_weights, boundaries,
                    scale, basis_mean, stencil_large_size, stencil_large, stencil_small_size, stencil_small,
                    si_matrix, moments, si_terms, troubled, troubled_coeffs, troubled_small_coeffs, troubled_cells,
                    n_troubled,
                    solution, face_solution, cells, {}, cell_gamma, cell_molar_mass, selection};
    using Dynamic = Kokkos::Schedule<Kokkos::Dynamic>;
    using Space = Kokkos::DefaultExecutionSpace;
    if (!troubled_pass) {
        const uint32_t n = cells.extent(0) ? cells.extent(0) : mesh->n_reconstructed();
        if constexpr (!Functor::USE_TEAM || Kokkos::SpaceAccessibility<Kokkos::HostSpace, Space::memory_space>::accessible) {
            Kokkos::parallel_for("teno_smooth",
                                 Kokkos::RangePolicy<Space, typename Functor::SmoothPass, Dynamic, HeavyBounds>(exec, 0, n),
                                 functor);
        } else {
            if (n == 0) return;
            functor.n_smooth = n;
            using Policy = Kokkos::TeamPolicy<Space, typename Functor::SmoothTeamPass,
                                              Kokkos::LaunchBounds<Functor::TEAM_SIZE, 2>>;
            const Policy policy = Policy(exec, (n + Functor::TEAM_CELLS - 1) / Functor::TEAM_CELLS, Functor::TEAM_SIZE)
                                      .set_scratch_size(0, Kokkos::PerTeam(Functor::TEAM_SCRATCH));
            Kokkos::parallel_for("teno_smooth", policy, functor);
        }
        return;
    }
    // Each pass loops over its round of the queue with as many threads as it
    // could need, up to TROUBLED_THREADS, so the queue length stays on the
    // device; rounds past the queue's end find no work
    const uint32_t n = mesh->n_reconstructed();
    const uint32_t capacity = troubled_coeffs.extent(0);
    if (n > 0 && capacity > 0) {
        auto launch = [&](const char * label, auto tag, uint32_t per_cell) {
            functor.stride = std::min<uint64_t>(TROUBLED_THREADS, uint64_t(capacity) * per_cell);
            Kokkos::parallel_for(label, Kokkos::RangePolicy<Space, decltype(tag), Dynamic, HeavyBounds>(exec, 0, functor.stride),
                                 functor);
        };
        constexpr bool host = Kokkos::SpaceAccessibility<Kokkos::HostSpace, Space::memory_space>::accessible;
        for (uint32_t begin = 0; begin < n; begin += capacity) {
            functor.round_begin = begin;
            if (begin > 0) {
                if constexpr (Functor::USE_TEAM && !host) {
                    using Policy = Kokkos::TeamPolicy<Space, typename Functor::TroubledCentralTeamPass,
                                                      Kokkos::LaunchBounds<Functor::TEAM_SIZE, 2>>;
                    const uint32_t league = std::min<uint32_t>((capacity + Functor::TEAM_CELLS - 1) / Functor::TEAM_CELLS,
                                                               TROUBLED_THREADS / Functor::TEAM_SIZE);
                    Kokkos::parallel_for("teno_troubled_central",
                                         Policy(exec, league, Functor::TEAM_SIZE)
                                             .set_scratch_size(0, Kokkos::PerTeam(Functor::TEAM_SCRATCH)),
                                         functor);
                } else {
                    launch("teno_troubled_central", typename Functor::TroubledCentralPass{}, 1);
                }
            }
            launch("teno_troubled_sectors", typename Functor::TroubledSectorPass{}, teno::MAX_FACES);
            if constexpr (N_DIM == 2) {
                launch("teno_troubled_select", typename Functor::TroubledSelectPass{}, teno::MAX_FACES * N_CONSERVATIVE);
            } else {
                const uint32_t team_size = host ? 1 : Functor::SELECT_TEAM;
                const uint32_t league = std::min<uint32_t>(capacity, TROUBLED_THREADS / team_size);
                using Policy = Kokkos::TeamPolicy<Space, typename Functor::TroubledSelectTeamPass, Dynamic>;
                Kokkos::parallel_for("teno_troubled_select",
                                     Policy(exec, league, team_size)
                                         .set_scratch_size(0, Kokkos::PerTeam(Functor::SELECT_SCRATCH)),
                                     functor);
            }
            launch("teno_troubled_project", typename Functor::TroubledProjectPass{}, teno::MAX_FACES);
            launch("teno_troubled_finish", typename Functor::TroubledFinishPass{}, 1);
        }
    }
    Kokkos::deep_copy(exec, n_troubled, 0u);
}

template <uint8_t DEG, bool PRIM, typename TABLE>
void TENO::launch_gradients(Kokkos::View<rtype *[N_CONSERVATIVE]> solution,
                            Kokkos::View<rtype *[N_CONSERVATIVE][N_DIM]> gradients, const uint32_t n_cells) {
    using Functor = TENOFunctor<DEG, PRIM, TABLE>;
    Functor functor{sigma_threshold, sigma_upper, C_T, characteristic, bound_preserving, boundaries.gamma,
                    mesh->offsets_faces_of_cell, mesh->faces_of_cell, mesh->cells_of_face,
                    mesh->offsets_nodes_of_face, mesh->nodes_of_face, mesh->node_coords,
                    mesh->cell_coords, mesh->face_coords, mesh->face_normals, mesh->shifts, mesh->face_shift,
                    quadrature_face.points, face_quad_points, face_quad_weights, boundaries,
                    scale, basis_mean, stencil_large_size, stencil_large, stencil_small_size, stencil_small,
                    si_matrix, moments, si_terms, troubled, troubled_coeffs, troubled_small_coeffs, troubled_cells,
                    n_troubled,
                    solution, {}, {}, gradients, cell_gamma, cell_molar_mass, selection};
    Kokkos::parallel_for("teno_gradients", HeavyRange<typename Functor::GradientPass>(0, n_cells), functor);
}

template <uint8_t DEG, bool PRIM, typename TABLE>
void TENO::launch_source(Kokkos::View<rtype *[N_CONSERVATIVE]> solution, Kokkos::View<rtype *, Kokkos::LayoutStride> mu,
                         Kokkos::View<rtype *> source, const uint32_t n_cells) {
    using Functor = TENOFunctor<DEG, PRIM, TABLE>;
    Functor functor{sigma_threshold, sigma_upper, C_T, characteristic, bound_preserving, boundaries.gamma,
                    mesh->offsets_faces_of_cell, mesh->faces_of_cell, mesh->cells_of_face,
                    mesh->offsets_nodes_of_face, mesh->nodes_of_face, mesh->node_coords,
                    mesh->cell_coords, mesh->face_coords, mesh->face_normals, mesh->shifts, mesh->face_shift,
                    quadrature_face.points, face_quad_points, face_quad_weights, boundaries,
                    scale, basis_mean, stencil_large_size, stencil_large, stencil_small_size, stencil_small,
                    si_matrix, moments, si_terms, troubled, troubled_coeffs, troubled_small_coeffs, troubled_cells,
                    n_troubled,
                    solution, {}, {}, {}, cell_gamma, cell_molar_mass, selection};
    functor.offsets_nodes_of_cell = mesh->offsets_nodes_of_cell;
    functor.nodes_of_cell = mesh->nodes_of_cell;
    functor.cell_rule = cell_rule;
    functor.mu = mu;
    functor.source = source;
    Kokkos::parallel_for("teno_axisymmetric_source", Kokkos::RangePolicy<typename Functor::SourcePass>(0, n_cells),
                         functor);
}

void TENO::axisymmetric_source(Kokkos::View<rtype *[N_CONSERVATIVE]> solution,
                               Kokkos::View<rtype *, Kokkos::LayoutStride> mu, Kokkos::View<rtype *> source,
                               const uint32_t n_cells) {
    if (!mesh->axisymmetric) return;
    switch (degree * 2 + primitive) {
        case 4: launch_source<2, false, rtype>(solution, mu, source, n_cells); break;
        case 6: launch_source<3, false, rtype>(solution, mu, source, n_cells); break;
        case 8: launch_source<4, false, rtype>(solution, mu, source, n_cells); break;
        case 10: launch_source<5, false, rtype>(solution, mu, source, n_cells); break;
        case 5: launch_source<2, true, rtype>(solution, mu, source, n_cells); break;
        case 7: launch_source<3, true, rtype>(solution, mu, source, n_cells); break;
        case 9: launch_source<4, true, rtype>(solution, mu, source, n_cells); break;
        case 11: launch_source<5, true, rtype>(solution, mu, source, n_cells); break;
        default: throw std::runtime_error("TENO: unsupported degree.");
    }
}

void TENO::set_mixture(Kokkos::View<rtype *, Kokkos::LayoutStride> gamma, Kokkos::View<rtype *> molar_mass) {
    if (single_tables) throw std::runtime_error("TENO: single_precision_tables is not available with gas mixtures.");
    primitive = true;
    cell_gamma = gamma;
    cell_molar_mass = molar_mass;
    selection = Kokkos::View<uint8_t **>("teno_selection", mesh->n_faces, 2);
}

bool TENO::cell_gradients(Kokkos::View<rtype *[N_CONSERVATIVE]> solution,
                          Kokkos::View<rtype *[N_CONSERVATIVE][N_DIM]> gradients, const uint32_t n_cells) {
#if N_DIM == 3
    if (single_tables) {
        switch (degree) {
            case 2: launch_gradients<2, false, float>(solution, gradients, n_cells); break;
            case 3: launch_gradients<3, false, float>(solution, gradients, n_cells); break;
            case 4: launch_gradients<4, false, float>(solution, gradients, n_cells); break;
            case 5: launch_gradients<5, false, float>(solution, gradients, n_cells); break;
            default: throw std::runtime_error("TENO: unsupported degree.");
        }
        return true;
    }
#endif
    switch (degree * 2 + primitive) {
        case 4: launch_gradients<2, false, rtype>(solution, gradients, n_cells); break;
        case 6: launch_gradients<3, false, rtype>(solution, gradients, n_cells); break;
        case 8: launch_gradients<4, false, rtype>(solution, gradients, n_cells); break;
        case 10: launch_gradients<5, false, rtype>(solution, gradients, n_cells); break;
        case 5: launch_gradients<2, true, rtype>(solution, gradients, n_cells); break;
        case 7: launch_gradients<3, true, rtype>(solution, gradients, n_cells); break;
        case 9: launch_gradients<4, true, rtype>(solution, gradients, n_cells); break;
        case 11: launch_gradients<5, true, rtype>(solution, gradients, n_cells); break;
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
#if N_DIM == 3
    if (single_tables) {
        switch (degree) {
            case 2: launch_reconstruction<2, false, float>(exec, solution, face_solution, cells, troubled_pass); break;
            case 3: launch_reconstruction<3, false, float>(exec, solution, face_solution, cells, troubled_pass); break;
            case 4: launch_reconstruction<4, false, float>(exec, solution, face_solution, cells, troubled_pass); break;
            case 5: launch_reconstruction<5, false, float>(exec, solution, face_solution, cells, troubled_pass); break;
            default: throw std::runtime_error("TENO: unsupported degree.");
        }
        return;
    }
#endif
    switch (degree * 2 + primitive) {
        case 4: launch_reconstruction<2, false, rtype>(exec, solution, face_solution, cells, troubled_pass); break;
        case 6: launch_reconstruction<3, false, rtype>(exec, solution, face_solution, cells, troubled_pass); break;
        case 8: launch_reconstruction<4, false, rtype>(exec, solution, face_solution, cells, troubled_pass); break;
        case 10: launch_reconstruction<5, false, rtype>(exec, solution, face_solution, cells, troubled_pass); break;
        case 5: launch_reconstruction<2, true, rtype>(exec, solution, face_solution, cells, troubled_pass); break;
        case 7: launch_reconstruction<3, true, rtype>(exec, solution, face_solution, cells, troubled_pass); break;
        case 9: launch_reconstruction<4, true, rtype>(exec, solution, face_solution, cells, troubled_pass); break;
        case 11: launch_reconstruction<5, true, rtype>(exec, solution, face_solution, cells, troubled_pass); break;
        default: throw std::runtime_error("TENO: unsupported degree.");
    }
}

namespace {

// Version 3 stores each reconstructed cell's tables at their actual stencil sizes; version 4 gathers
// candidates by interior cells, version 5 sorts them in the mesh-spacing metric, version 6 follows
// the round-off-accurate 2D cell centroids, version 7 can hold the pseudo-inverses and
// smoothness-indicator matrices in single precision and has tables from the faster setup (#140),
// which differ from version 6 at round-off; version 8 bounds 3D Lebesgue constants by 4 (#126, #229);
// version 9 records where the tables were set up (the execution space, or "host"); version 10 holds
// each 3D cell's moments in place of its smoothness-indicator matrix
constexpr char TENO_CACHE_MAGIC[16] = "MALLARD-TENO-10";
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
    char producer[16] = {};
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
    put(buf, h.producer, sizeof(h.producer));
    put(buf, &h.options_key, 1);
    put(buf, &h.cache_key, 1);
    put(buf, &h.halo_layers, 1);
    put(buf, &h.n_reconstructed, 1);
    out.write(buf.data(), buf.size());
}

bool read_header(std::istream & in, CacheHeader & h) {
    if (!get(in, h.magic, sizeof(h.magic))) return false;
    // Older formats have no producer
    if (std::string(h.magic, strnlen(h.magic, sizeof(h.magic))) != TENO_CACHE_MAGIC) return true;
    return get(in, h.producer, sizeof(h.producer)) && get(in, &h.options_key, 1) && get(in, &h.cache_key, 1) &&
           get(in, &h.halo_layers, 1) && get(in, &h.n_reconstructed, 1);
}

/** @brief Round values to single precision in place. */
template <typename T>
void round_values_to_single(const Kokkos::View<T *> & values) {
    Kokkos::parallel_for("teno_round_to_single", values.extent(0), KOKKOS_LAMBDA(const size_t k) {
        values(k) = static_cast<T>(static_cast<float>(values(k)));
    });
}

/** @brief Append values, in single precision if single. */
template <typename T>
void put_values(std::vector<char> & buf, const std::vector<T> & values, const bool single) {
    if (!single) {
        put(buf, values.data(), values.size());
        return;
    }
    std::vector<float> narrow(values.begin(), values.end());
    put(buf, narrow.data(), narrow.size());
}

/** @brief Read values.size() values, stored in single precision if single. */
template <typename T>
bool get_values(std::istream & in, std::vector<T> & values, const bool single) {
    if (!single) return get(in, values.data(), values.size());
    std::vector<float> narrow(values.size());
    if (!get(in, narrow.data(), narrow.size())) return false;
    std::copy(narrow.begin(), narrow.end(), values.begin());
    return true;
}

/**
 * @brief Append one cell's record: its fixed-size data, then its stencils at
 *        their actual sizes; single stores the pseudo-inverses and the
 *        smoothness-indicator matrix in single precision.
 */
void serialize(const CellTables & t, const bool single, std::vector<char> & buf) {
    const uint16_t n_large = t.large_cells.size();
    put(buf, &t.gather_depth, 1);
    put(buf, &t.scale, 1);
    put(buf, &n_large, 1);
    put(buf, t.small_size.data(), t.small_size.size());
    put(buf, t.basis_mean.data(), t.basis_mean.size());
    put_values(buf, t.si, single);
    put_values(buf, t.moments, single);
    put(buf, t.large_cells.data(), n_large);
    put(buf, t.large_faces.data(), n_large);
    put_values(buf, t.large_pinv, single);
    put(buf, t.small_cells.data(), t.small_cells.size());
    put(buf, t.small_faces.data(), t.small_faces.size());
    put_values(buf, t.small_pinv, single);
}

bool deserialize(std::istream & in, const uint8_t nk, const uint16_t n_moments, const bool single, CellTables & t) {
    uint16_t n_large = 0;
    if (!get(in, &t.gather_depth, 1) || !get(in, &t.scale, 1) || !get(in, &n_large, 1) ||
        !get(in, t.small_size.data(), t.small_size.size())) {
        return false;
    }
    size_t n_small = 0;
    for (uint16_t n : t.small_size) n_small += n;
    t.basis_mean.resize(nk);
    t.si.resize(N_DIM == 2 ? nk * (nk + 1) / 2 : 0);
    t.moments.resize(N_DIM == 3 ? n_moments : 0);
    t.large_cells.resize(n_large);
    t.large_faces.resize(n_large);
    t.large_pinv.resize(size_t(n_large) * nk);
    t.small_cells.resize(n_small);
    t.small_faces.resize(n_small);
    t.small_pinv.resize(n_small * teno::NK_SMALL);
    return get(in, t.basis_mean.data(), t.basis_mean.size()) && get_values(in, t.si, single) &&
           get_values(in, t.moments, single) &&
           get(in, t.large_cells.data(), n_large) && get(in, t.large_faces.data(), n_large) &&
           get_values(in, t.large_pinv, single) && get(in, t.small_cells.data(), n_small) &&
           get(in, t.small_faces.data(), n_small) && get_values(in, t.small_pinv, single);
}

} // namespace

void TENO::round_to_single() {
    if (!stencil_large.single) round_values_to_single(stencil_large.pinv);
    if (!stencil_small.single) round_values_to_single(stencil_small.pinv);
    if constexpr (N_DIM == 2) {
        round_values_to_single(Kokkos::View<rtype *>(si_matrix.data(), si_matrix.span()));
    } else {
        round_values_to_single(Kokkos::View<double *>(moments.data(), moments.span()));
    }
    Kokkos::fence();
}

void TENO::allocate_scratch() {
    const uint32_t n_reconstructed = mesh->n_reconstructed();
    troubled = Kokkos::View<rtype *>("teno_sigma", mesh->n_cells);
    const uint32_t capacity = std::min<uint32_t>(
        n_reconstructed, static_cast<uint32_t>(std::ceil(double(troubled_capacity) * double(n_reconstructed))));
    troubled_coeffs = Kokkos::View<rtype ***>("teno_troubled_coeffs", capacity, n_dof_large, N_CONSERVATIVE);
    troubled_small_coeffs = Kokkos::View<rtype ****>("teno_troubled_small_coeffs", capacity, teno::MAX_FACES,
                                                     teno::NK_SMALL, N_CONSERVATIVE);
    troubled_cells = Kokkos::View<uint32_t *>("teno_troubled_cells", n_reconstructed);
    n_troubled = Kokkos::View<uint32_t>("teno_n_troubled");
}

std::array<double, 9> TENO::stencil_metric(const uint32_t i) const {
    auto h_face_bc = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), boundaries.face_bc);
    auto h_bcs = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), boundaries.bcs);
    return teno_setup::ranking_metric(*mesh, h_face_bc, h_bcs, i);
}

void TENO::rebuild_cells(const std::vector<uint32_t> & cells) {
    std::vector<uint32_t> sorted(cells);
    std::sort(sorted.begin(), sorted.end());
    sorted.erase(std::unique(sorted.begin(), sorted.end()), sorted.end());
    if (sorted.empty()) return;
    if (sorted.back() >= scale.extent(0)) throw std::logic_error("TENO: only reconstructed cells have tables.");
    // The face quadrature follows the mesh's geometry
    if constexpr (N_DIM == 3) init_face_quadrature_3d(degree + 1);
    compute_stencils_and_matrices(&sorted);
    if (single_tables || (cache_single && !cache_file.empty())) round_to_single();
    uint16_t largest = 0;
    auto sizes = stencil_large_size;
    Kokkos::parallel_reduce("teno_largest", Kokkos::RangePolicy<>(0, sizes.extent(0)),
                            KOKKOS_LAMBDA(const uint32_t c, uint16_t & m) { m = sizes(c) > m ? sizes(c) : m; },
                            Kokkos::Max<uint16_t>(largest));
    largest_stencil = largest;
    cache_loaded = false;
}

std::vector<uint32_t> TENO::cells_within_reach(const std::vector<uint32_t> & changed) const {
    const uint32_t n_rec = scale.extent(0);
    uint8_t max_depth = 0;
    for (uint32_t c = 0; c < n_rec; c++) max_depth = std::max(max_depth, gather_depth[c]);
    constexpr uint8_t FAR = std::numeric_limits<uint8_t>::max();
    std::vector<uint8_t> distance(mesh->n_cells, FAR);
    std::vector<uint32_t> frontier, next;
    for (const uint32_t c : changed) {
        if (distance[c] == FAR) frontier.push_back(c);
        distance[c] = 0;
    }
    for (uint8_t d = 1; d <= max_depth && !frontier.empty(); d++) {
        next.clear();
        for (const uint32_t c : frontier) {
            for (uint32_t k = mesh->h_offsets_cells_of_cell(c); k < mesh->h_offsets_cells_of_cell(c + 1); k++) {
                const uint32_t nb = mesh->h_cells_of_cell(k);
                if (distance[nb] != FAR) continue;
                distance[nb] = d;
                next.push_back(nb);
            }
        }
        frontier.swap(next);
    }
    std::vector<uint32_t> out;
    for (uint32_t c = 0; c < n_rec; c++) {
        if (distance[c] <= gather_depth[c]) out.push_back(c);
    }
    return out;
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

std::vector<uint8_t> TENO::stencil_cells() const {
    std::vector<uint8_t> used(mesh->n_cells, 0);
    const uint32_t n_reconstructed = scale.extent(0);
    const auto large_slices = host_slice_start(stencil_large);
    const auto small_slices = host_slice_start(stencil_small);
    std::vector<CellTables> chunk;
    for (uint32_t c0 = 0; c0 < n_reconstructed; c0 += CHUNK_CELLS) {
        chunk.assign(std::min(CHUNK_CELLS, n_reconstructed - c0), CellTables());
        download_tables(*this, large_slices, small_slices, c0, chunk);
        for (const CellTables & t : chunk) {
            for (const int32_t c : t.large_cells) used[c] = 1;
            for (const int32_t c : t.small_cells) used[c] = 1;
        }
    }
    return used;
}

uint64_t TENO::options_key() const {
    Fnv1a hash;
    hash.add(sizeof(rtype));
    hash.add(degree);
    hash.add(n_stencil_small);
    hash.add(stencil_factor);
    hash.add(max_condition);
    hash.add(cache_single);
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
    // Axisymmetric meshes have r-weighted moments and revolved centroids
    if (mesh->axisymmetric) hash.add(uint8_t(1));
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
    std::strncpy(header.producer, setup_producer.c_str(), sizeof(header.producer) - 1);
    write_header(out, header);
    const auto large_slices = host_slice_start(stencil_large);
    const auto small_slices = host_slice_start(stencil_small);
    std::vector<CellTables> chunk;
    std::vector<char> buf;
    for (uint32_t c0 = 0; c0 < n_reconstructed; c0 += CHUNK_CELLS) {
        chunk.assign(std::min(CHUNK_CELLS, n_reconstructed - c0), CellTables());
        download_tables(*this, large_slices, small_slices, c0, chunk);
        buf.clear();
        for (const CellTables & t : chunk) serialize(t, cache_single, buf);
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
    const uint16_t n_moments = N_DIM == 3 ? teno_setup::n_moments(degree) : 0;
    std::vector<CellTables> chunk;
    bool ok = true;
    for (uint32_t c0 = 0; c0 < n_reconstructed && ok; c0 += CHUNK_CELLS) {
        chunk.assign(std::min(CHUNK_CELLS, n_reconstructed - c0), CellTables());
        for (CellTables & t : chunk) {
            ok = ok && deserialize(in, n_dof_large, n_moments, cache_single, t);
            largest_stencil = std::max<uint32_t>(largest_stencil, t.large_cells.size());
        }
        if (ok) builder.add(c0, chunk);
    }
    if (!ok) {
        cache_status = cache_file + " is truncated, recomputed";
        return false;
    }
    builder.finish();
    cache_status = "loaded from " + cache_file + " (set up on " +
                   std::string(header.producer, strnlen(header.producer, sizeof(header.producer))) + ")";
    setup_producer = std::string(header.producer, strnlen(header.producer, sizeof(header.producer)));
    return true;
}

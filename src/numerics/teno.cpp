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
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <memory>
#include <numeric>
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
#include "diagnostic_code.h"
#include "face_reconstruction.h"
#include "gated.h"
#include "growing_buffer.h"
#include "view_entries.h"

#include "input.h"
#include "launch_bounds.h"
#include "log.h"
#include "periodic.h"
#include "scratch_arena.h"
#include "teno_sharing.h"
#include "teno.h"
#include "structured_blocks.h"
#include "teno_setup.h"
#include "teno_shapes.h"

namespace {

/**
 * @brief Gauss-Legendre nodes and weights on [-1, 1] (Newton iteration).
 */
void gauss_legendre(int n, std::vector<double> & x, std::vector<double> & w) {
    x.resize(static_cast<size_t>(n));
    w.resize(static_cast<size_t>(n));
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
        x[static_cast<size_t>(i)] = z;
        w[static_cast<size_t>(i)] = 2.0 / ((1.0 - z * z) * dp * dp);
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
        for (size_t i = 0; i < g.size(); i++) {
            for (size_t j = 0; j < g.size(); j++) {
                const double s = 0.5 * (g[i] + 1.0);
                const double t = 0.5 * (g[j] + 1.0);
                xi.push_back(s);
                eta.push_back(t * (1.0 - s));
                w.push_back(0.25 * gw[i] * gw[j] * (1.0 - s));
            }
        }
    }
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
    // Rows of no columns: dev may be unallocated (no sector stencils)
    if (h.size() == 0) return;
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
    if (h.size() == 0) return h;
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
 * @brief Host copy of the slice starts, images and pseudo-inverse indices of
 *        packed stencils, and, if cells share their pseudo-inverses, of those.
 */
struct HostIndex {
    std::vector<uint64_t> slice_start;
    std::vector<uint64_t> pinv_index;
    std::vector<int32_t> images;
    std::vector<char> pinv;  // shared: packed.pinv
};

HostIndex host_index(const teno::PackedStencils & packed) {
    HostIndex h{std::vector<uint64_t>(packed.slice_start.extent(0)), std::vector<uint64_t>(packed.pinv_index.extent(0)),
                std::vector<int32_t>(packed.images.extent(0)), {}};
    Kokkos::deep_copy(HostUnmanaged<uint64_t>(h.slice_start.data(), h.slice_start.size()), packed.slice_start);
    Kokkos::deep_copy(HostUnmanaged<uint64_t>(h.pinv_index.data(), h.pinv_index.size()), packed.pinv_index);
    Kokkos::deep_copy(HostUnmanaged<int32_t>(h.images.data(), h.images.size()), packed.images);
    if (packed.shared) {
        h.pinv.resize(packed.pinv.extent(0) * sizeof(rtype));
        Kokkos::deep_copy(HostUnmanaged<char>(h.pinv.data(), h.pinv.size()),
                          DeviceUnmanaged<char>(reinterpret_cast<char *>(packed.pinv.data()), h.pinv.size()));
    }
    return h;
}

/** @brief Index of each cell's pseudo-inverses in packed stencils that share none (slice starts given). */
Kokkos::View<uint64_t *> own_pinv_index(const std::vector<uint64_t> & slice_start, const uint8_t shift,
                                        const uint32_t n_cells, const std::string & label) {
    std::vector<uint64_t> index(n_cells);
    for (uint32_t c = 0; c < n_cells; c++) index[c] = (slice_start[c >> shift] << shift) | (c & ((1u << shift) - 1));
    Kokkos::View<uint64_t *> out(Kokkos::view_alloc(Kokkos::WithoutInitializing, label), n_cells);
    Kokkos::deep_copy(out, HostUnmanaged<const uint64_t>(index.data(), index.size()));
    return out;
}

/**
 * @brief Moves one stencil family (large or sector) of CellTables into a
 *        teno::PackedStencils and back. The device arrays can only be sized
 *        once every stencil is known: on CUDA, each packed chunk goes straight
 *        into device memory that grows in place (stream()); elsewhere chunks
 *        stay packed on the host until finish().
 */
class PackedRows {
    public:
        /**
         * @brief With share_values, the cells of a stencil shape share one
         *        table (teno::TableStore), that of the cell with the lowest
         *        global id (global_ids, by local cell; null: the local index);
         *        sectors: the stencils are each face's sector stencil, one
         *        after another.
         */
        PackedRows(uint8_t slice_shift, uint8_t values_per_slot, IndexRow cell_row, IndexRow face_row, ValueRow pinv_row,
                   bool single_values, bool share_values, const uint64_t * global_ids, bool sectors)
            : shift(slice_shift), width(values_per_slot), single(single_values), share(share_values), gid(global_ids),
              by_face(sectors), cells(cell_row), faces(face_row), pinv(pinv_row), store(values_per_slot) {}

        /**
         * @brief Upload each chunk as it is added, for n_cells cells of at
         *        most max_slots_per_cell stencil entries each.
         */
        void stream(const uint32_t n_cells, const size_t max_slots_per_cell, const std::string & label) {
            const size_t max_slots = ((size_t(n_cells) >> shift) + 1) * max_slots_per_cell << shift;
            memory = {std::make_shared<GrowingBuffer>(max_slots * sizeof(int32_t), label),
                      std::make_shared<GrowingBuffer>(max_slots * width * sizeof(rtype), label + "_pinv")};
        }

        /** @brief The device memory that the stencils of finish() live in, if streamed. */
        const std::vector<std::shared_ptr<GrowingBuffer>> & device_memory() const { return memory; }

        /** @brief Pack the stencils of cells [c0, c0 + tables.size()), which follow the previous chunk. */
        void add(const uint32_t c0, const std::vector<CellTables> & tables) {
            n_added += static_cast<uint32_t>(tables.size());
            if (share) {
                add_shared(c0, tables);
                return;
            }
            const uint32_t slice = 1u << shift;
            if (c0 % slice != 0 || c0 >> shift != slice_start.size() - 1) {
                throw std::logic_error("TENO: stencil chunks must be packed in order.");
            }
            const uint32_t n = static_cast<uint32_t>(tables.size());
            Chunk chunk;
            chunk.slot0 = slice_start.back();
            for (uint32_t a = 0; a < n; a += slice) {
                size_t largest = 0;
                for (uint32_t c = a; c < std::min(n, a + slice); c++) largest = std::max(largest, (tables[c].*cells).size());
                slice_start.push_back(slice_start.back() + largest);
            }
            // Images are numbered in cell order, then slot order
            std::vector<uint64_t> first_image(n + 1, images.size() / 2);
            for (uint32_t c = 0; c < n; c++) {
                const auto & f = tables[c].*faces;
                first_image[c + 1] = first_image[c] + static_cast<uint64_t>(std::count_if(f.begin(), f.end(),
                                                                                     [](int32_t x) { return x != -1; }));
            }
            images.resize(2 * first_image[n]);
            const size_t n_slots = (slice_start.back() - chunk.slot0) << shift;
            chunk.cells = PageArray<int32_t>(n_slots, 0);
            chunk.pinv = PageArray<rtype>(teno::PackedStencils::pinv_elements(n_slots * width, single), 0.0);
            gated::parallel_for("teno_pack", Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(0, n),
                                 [&](const uint32_t c) {
                const CellTables & t = tables[c];
                const uint64_t start = slice_start[(c0 + c) >> shift] - chunk.slot0;
                const uint32_t lane = (c0 + c) & (slice - 1);
                uint64_t image = first_image[c];
                for (size_t s = 0; s < (t.*cells).size(); s++) {
                    const int32_t cell = (t.*cells)[s], face = (t.*faces)[s];
                    int32_t entry = cell;
                    if (face != -1) {
                        images[2 * image] = cell;
                        images[2 * image + 1] = face;
                        entry = teno::PackedStencils::image_entry(image++);
                    }
                    chunk.cells[((start + s) << shift) + lane] = entry;
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
            memory[1]->ensure((b + n_chunk) * width * value_bytes());
            Kokkos::deep_copy(DeviceUnmanaged<int32_t>(static_cast<int32_t *>(memory[0]->data()) + b, n_chunk),
                              HostUnmanaged<int32_t>(chunk.cells.data(), n_chunk));
            Kokkos::deep_copy(DeviceUnmanaged<char>(static_cast<char *>(memory[1]->data()) + b * width * value_bytes(),
                                                    n_chunk * width * value_bytes()),
                              HostUnmanaged<char>(reinterpret_cast<char *>(chunk.pinv.data()), n_chunk * width * value_bytes()));
        }

        /**
         * @brief Pack the stencils of cells [c0, c0 + n), which follow the
         *        previous chunk, from slots 0..n of a batch on the device:
         *        sizes(t) entries of cells, faces and pinv for slot t
         *        (sectors: segment_sizes(t, face) for each face). Shared:
         *        slot t's pseudo-inverse is bitwise that of slot same(t) <= t,
         *        which has its own (empty: every slot has its own).
         */
        void add_batch(const uint32_t c0, const uint32_t n, const Kokkos::View<uint16_t *> & sizes,
                       const teno_setup::TableBatch::V<uint16_t **> & segment_sizes,
                       const teno_setup::SlotArray<int32_t> & batch_cells, const teno_setup::SlotArray<int32_t> & batch_faces,
                       const teno_setup::SlotArray<rtype> & batch_pinv, const std::vector<uint32_t> & same = {}) {
            const uint32_t slice = 1u << shift;
            if (c0 % slice != 0 || c0 >> shift != slice_start.size() - 1) {
                throw std::logic_error("TENO: stencil chunks must be packed in order.");
            }
            n_added += n;
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
            const uint32_t n_slices = static_cast<uint32_t>(slice_start.size() - 1 - first_slice);
            uint64_t longest = 0;
            for (size_t sl = first_slice; sl + 1 < slice_start.size(); sl++) {
                longest = std::max(longest, slice_start[sl + 1] - slice_start[sl]);
            }
            Kokkos::View<uint64_t *> starts("teno_pack_starts", n_slices + 1);
            Kokkos::deep_copy(starts, HostUnmanaged<const uint64_t>(slice_start.data() + first_slice, n_slices + 1));
            const size_t n_slots = (slice_start.back() - chunk.slot0) << shift;
            const size_t b = chunk.slot0 << shift;
            // Shared: each cell's slots in canonical order (row(t) of order), its pseudo-inverse in the store
            Kokkos::View<uint16_t **, Kokkos::LayoutLeft> order;
            Kokkos::View<uint32_t *> row;
            if (share) order = share_batch(c0, n, sizes, segment_sizes, batch_pinv, same, row);
            int32_t * dest_cells;
            char * dest_pinv = nullptr;
            Kokkos::View<int32_t *> tmp_cells;
            Kokkos::View<rtype *> tmp_pinv;
            constexpr bool host_device =
                Kokkos::SpaceAccessibility<Kokkos::HostSpace, Kokkos::DefaultExecutionSpace::memory_space>::accessible;
            if (!memory.empty()) {
                memory[0]->ensure((b + n_slots) * sizeof(int32_t));
                memory[1]->ensure((b + n_slots) * width * value_bytes());
                dest_cells = static_cast<int32_t *>(memory[0]->data()) + b;
                dest_pinv = static_cast<char *>(memory[1]->data()) + b * width * value_bytes();
            } else if (host_device) {
                chunk.cells = PageArray<int32_t>(n_slots, 0);
                dest_cells = chunk.cells.data();
                if (!share) {
                    chunk.pinv = PageArray<rtype>(teno::PackedStencils::pinv_elements(n_slots * width, single), 0.0);
                    dest_pinv = reinterpret_cast<char *>(chunk.pinv.data());
                }
            } else {
                tmp_cells = Kokkos::View<int32_t *>(Kokkos::view_alloc(Kokkos::WithoutInitializing, "teno_pack_cells"), n_slots);
                dest_cells = tmp_cells.data();
                if (!share) {
                    tmp_pinv = Kokkos::View<rtype *>(Kokkos::view_alloc(Kokkos::WithoutInitializing, "teno_pack_pinv"),
                                                     teno::PackedStencils::pinv_elements(n_slots * width, single));
                    dest_pinv = reinterpret_cast<char *>(tmp_pinv.data());
                }
            }
            const uint8_t sh = shift;
            const uint8_t w = width;
            const bool sg = single;
            const bool permuted = share;
            // Images are numbered in cell order, then slot order (shared: canonical order)
            Kokkos::View<uint64_t *> first_image("teno_pack_first_image", n + 1);
            gated::parallel_scan("teno_pack_count_images", Kokkos::RangePolicy<>(0, n + 1),
                                  KOKKOS_LAMBDA(const uint32_t t, uint64_t & sum, const bool final) {
                if (final) first_image(t) = sum;
                if (t < n) {
                    const auto lf = batch_faces.lane(t);
                    for (uint32_t s = 0; s < sizes(t); s++) sum += lf[s] != -1 ? 1u : 0u;
                }
            });
            uint64_t n_new = 0;
            Kokkos::deep_copy(n_new, Kokkos::subview(first_image, n));
            Kokkos::View<int32_t *> new_images(Kokkos::view_alloc(Kokkos::WithoutInitializing, "teno_pack_images"),
                                               2 * n_new);
            const uint64_t image0 = images.size() / 2;
            gated::parallel_for("teno_pack_entries", Kokkos::RangePolicy<>(0, uint64_t(n_slices) << sh),
                                 KOKKOS_LAMBDA(const uint64_t t) {
                const uint32_t sl = static_cast<uint32_t>(t >> sh);
                const uint32_t lane = static_cast<uint32_t>(t & ((1u << sh) - 1));
                const uint64_t start = starts(sl) - starts(0);
                const uint64_t len = starts(sl + 1) - starts(sl);
                const uint32_t size = t < n ? sizes(t) : 0;
                const auto lc = batch_cells.lane(t < n ? t : 0);
                const auto lf = batch_faces.lane(t < n ? t : 0);
                uint64_t image = t < n ? first_image(t) : 0;
                for (uint64_t s = 0; s < len; s++) {
                    int32_t entry = 0;
                    if (s < size) {
                        const uint64_t from = permuted ? order(row(t), s) : s;
                        entry = lc[from];
                        if (lf[from] != -1) {
                            new_images(2 * image) = lc[from];
                            new_images(2 * image + 1) = lf[from];
                            entry = teno::PackedStencils::image_entry(image0 + image++);
                        }
                    }
                    dest_cells[((start + s) << sh) + lane] = entry;
                }
            });
            // A thread per (cell, slot), so that small batches still fill the device
            using Range2 = Kokkos::MDRangePolicy<Kokkos::Rank<2>>;
            if (!share) gated::parallel_for("teno_pack_batch", Range2({0, 0}, {int64_t(uint64_t(n_slices) << sh), int64_t(longest)}),
                                 KOKKOS_LAMBDA(const int64_t t_index, const int64_t s_index) {
                const uint64_t t = static_cast<uint64_t>(t_index), s = static_cast<uint64_t>(s_index);
                const uint32_t sl = static_cast<uint32_t>(t >> sh);
                const uint32_t lane = static_cast<uint32_t>(t & ((1u << sh) - 1));
                const uint64_t start = starts(sl) - starts(0);
                const uint64_t len = starts(sl + 1) - starts(sl);
                const uint32_t size = t < n ? sizes(t) : 0;
                const auto lp = batch_pinv.lane(t < n ? t : 0);
                if (s < len) {
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
            images.resize(images.size() + 2 * n_new);
            Kokkos::deep_copy(HostUnmanaged<int32_t>(images.data() + 2 * image0, 2 * n_new), new_images);
            if (!memory.empty()) return;
            if (!host_device) {
                chunk.cells = PageArray<int32_t>(n_slots, 0);
                Kokkos::deep_copy(HostUnmanaged<int32_t>(chunk.cells.data(), n_slots), tmp_cells);
                if (!share) {
                    chunk.pinv = PageArray<rtype>(tmp_pinv.extent(0), 0.0);
                    Kokkos::deep_copy(HostUnmanaged<rtype>(chunk.pinv.data(), tmp_pinv.extent(0)), tmp_pinv);
                }
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
            out.slice_start = Kokkos::View<uint64_t *>(label + "_slice_start", slice_start.size());
            Kokkos::deep_copy(out.slice_start, HostUnmanaged<const uint64_t>(slice_start.data(), slice_start.size()));
            out.images = Kokkos::View<int32_t *>(Kokkos::view_alloc(Kokkos::WithoutInitializing, label + "_images"),
                                                 images.size());
            Kokkos::deep_copy(out.images, HostUnmanaged<const int32_t>(images.data(), images.size()));
            images = std::vector<int32_t>();
            if (share) {
                finish_shared(out, label);
                out.cells = Kokkos::View<int32_t *>(Kokkos::view_alloc(Kokkos::WithoutInitializing, label), n_slots);
                for (Chunk & chunk : chunks) {
                    const size_t b = chunk.slot0 << shift;
                    Kokkos::deep_copy(Kokkos::subview(out.cells, std::make_pair(b, b + chunk.cells.size())),
                                      HostUnmanaged<int32_t>(chunk.cells.data(), chunk.cells.size()));
                    chunk = Chunk();
                }
                chunks.clear();
                return out;
            }
            out.pinv_index = own_pinv_index(slice_start, shift, n_added, label + "_pinv_index");
            if (!memory.empty()) {
                memory[0]->ensure(n_slots * sizeof(int32_t));
                memory[1]->ensure(n_pinv * sizeof(rtype));
                out.cells = Kokkos::View<int32_t *>(static_cast<int32_t *>(memory[0]->data()), n_slots);
                out.pinv = Kokkos::View<rtype *>(static_cast<rtype *>(memory[1]->data()), n_pinv);
                return out;
            }
            out.cells = Kokkos::View<int32_t *>(Kokkos::view_alloc(Kokkos::WithoutInitializing, label), n_slots);
            out.pinv = Kokkos::View<rtype *>(Kokkos::view_alloc(Kokkos::WithoutInitializing, label + "_pinv"), n_pinv);
            for (Chunk & chunk : chunks) {
                const size_t b = chunk.slot0 << shift;
                const std::pair<size_t, size_t> slots(b, b + chunk.cells.size());
                Kokkos::deep_copy(Kokkos::subview(out.cells, slots),
                                  HostUnmanaged<int32_t>(chunk.cells.data(), chunk.cells.size()));
                const size_t n_bytes = chunk.cells.size() * width * value_bytes();
                Kokkos::deep_copy(DeviceUnmanaged<char>(reinterpret_cast<char *>(out.pinv.data()) + b * width * value_bytes(),
                                                        n_bytes),
                                  HostUnmanaged<char>(reinterpret_cast<char *>(chunk.pinv.data()), n_bytes));
                chunk = Chunk();
            }
            chunks.clear();
            return out;
        }

        /**
         * @brief Stencils of cells [c0, c0 + tables.size()) from the device
         *        (h_slice_start: host copy of packed.slice_start; h_images: of
         *        packed.images; sizes: slots of each cell).
         */
        void download(const teno::PackedStencils & packed, const HostIndex & index, const uint32_t c0,
                      const std::vector<uint16_t> & sizes, std::vector<CellTables> & tables) const {
            const std::vector<uint64_t> & h_slice_start = index.slice_start;
            const std::vector<int32_t> & h_images = index.images;
            const uint8_t sh = packed.shift;
            const uint32_t slice = 1u << sh;
            const uint32_t c1 = c0 + static_cast<uint32_t>(tables.size());
            const uint64_t slot0 = h_slice_start[c0 >> sh];
            const uint64_t slot1 = h_slice_start[(c1 + slice - 1) >> sh];
            const std::pair<size_t, size_t> slots(slot0 << sh, slot1 << sh);
            const size_t vb = packed.single ? sizeof(float) : sizeof(rtype);
            const size_t n_values = (slots.second - slots.first) * packed.width;
            std::vector<int32_t> h_cells(slots.second - slots.first);
            const bool shared_values = packed.shared;
            std::vector<char> h_pinv_bytes(shared_values ? 0 : n_values * vb);
            Kokkos::deep_copy(HostUnmanaged<int32_t>(h_cells.data(), h_cells.size()), Kokkos::subview(packed.cells, slots));
            if (!shared_values && n_values > 0) {
                Kokkos::deep_copy(HostUnmanaged<char>(h_pinv_bytes.data(), h_pinv_bytes.size()),
                                  DeviceUnmanaged<char>(reinterpret_cast<char *>(packed.pinv.data()) +
                                                            slots.first * packed.width * vb,
                                                        h_pinv_bytes.size()));
            }
            const char * pinv_bytes = shared_values ? index.pinv.data() : h_pinv_bytes.data();
            auto h_pinv = [&](const size_t k) {
                return packed.single ? rtype(reinterpret_cast<const float *>(pinv_bytes)[k])
                                     : reinterpret_cast<const rtype *>(pinv_bytes)[k];
            };
            for (uint32_t c = 0; c < tables.size(); c++) {
                CellTables & t = tables[c];
                const uint64_t start = h_slice_start[(c0 + c) >> sh] - slot0;
                const uint64_t own = index.pinv_index[c0 + c];
                const uint64_t start_pinv = shared_values ? own >> sh : start;
                const uint32_t lane = (c0 + c) & (slice - 1);
                const uint64_t lane_pinv = shared_values ? own & (slice - 1) : lane;
                (t.*cells).resize(sizes[c]);
                (t.*faces).resize(sizes[c]);
                (t.*pinv).resize(sizes[c] * packed.width);
                for (uint16_t s = 0; s < sizes[c]; s++) {
                    const int32_t e = h_cells[((start + s) << sh) + lane];
                    const size_t image = e >= 0 ? 0 : static_cast<size_t>(-1 - e);
                    (t.*cells)[s] = e >= 0 ? e : h_images[2 * image];
                    (t.*faces)[s] = e >= 0 ? -1 : h_images[2 * image + 1];
                    for (uint32_t l = 0; l < packed.width; l++) {
                        (t.*pinv)[s * packed.width + l] = h_pinv((((start_pinv + s) * packed.width + l) << sh) + lane_pinv);
                    }
                }
            }
        }

        /**
         * @brief For each match, whether the table of row cell (slot
         *        slot(cell) of a batch, in order(cell, s)) is equal on the grid
         *        to that of row held or, if held < 0, of the shape in the pool.
         */
        std::vector<uint8_t> share_compare(const std::vector<teno::TableStore<>::Match> & matches,
                                           const Kokkos::View<uint16_t *> & sizes, const Kokkos::View<uint32_t *> & slot,
                                           const Kokkos::View<int32_t *> & exponent,
                                           const Kokkos::View<uint16_t **, Kokkos::LayoutLeft> & order,
                                           const teno_setup::SlotArray<rtype> & batch_pinv) const {
            const uint32_t w = width;
            const auto pool_values = pool;
            const size_t m = matches.size();
            std::vector<uint8_t> out(m, 0);
            if (m == 0) return out;
            Kokkos::View<int64_t **, Kokkos::HostSpace> h_pairs("teno_share_pairs", m, 2);
            for (size_t i = 0; i < m; i++) {
                h_pairs(i, 0) = matches[i].cell;
                h_pairs(i, 1) = matches[i].held >= 0 ? -1 - matches[i].held : int64_t(pool_at[matches[i].shape]);
            }
            auto pairs = Kokkos::create_mirror_view_and_copy(Kokkos::DefaultExecutionSpace::memory_space(), h_pairs);
            Kokkos::View<uint8_t *> equal("teno_share_equal", m);
            gated::parallel_for("teno_share_compare", Kokkos::RangePolicy<>(0, m), KOKKOS_LAMBDA(const size_t i) {
                const uint32_t r = static_cast<uint32_t>(pairs(i, 0));
                const int64_t ref = pairs(i, 1);
                const uint32_t h = ref < 0 ? static_cast<uint32_t>(-1 - ref) : r;
                const int e = exponent(r);
                const auto a = batch_pinv.lane(slot(r));
                const auto b = batch_pinv.lane(slot(h));
                uint8_t eq = 1;
                for (uint32_t s = 0; s < sizes(slot(r)) && eq; s++) {
                    for (uint32_t l = 0; l < w && eq; l++) {
                        const rtype x = a[order(r, s) * w + l];
                        const rtype y = ref < 0 ? b[order(h, s) * w + l] : pool_values(uint64_t(ref) + s * w + l);
                        eq = teno::on_grid(x, e) == teno::on_grid(y, e);
                    }
                }
                equal(i) = eq;
            });
            auto h_equal = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), equal);
            for (size_t i = 0; i < m; i++) out[i] = h_equal(i);
            return out;
        }

        /**
         * @brief add_batch() with share: the canonical order of the stencils
         *        of slots 0..n, slot t's in row(t) of the result, canonicalized
         *        and compared on the device, and their shapes in the store,
         *        which copies to the host only the tables that its shapes take.
         *        Slots that have the table of another (same) are not ranked
         *        again: the store sees each table once, with the lowest global
         *        id of its slots, which gives the shapes of ranking every slot.
         */
        Kokkos::View<uint16_t **, Kokkos::LayoutLeft> share_batch(const uint32_t c0, const uint32_t n,
                                                                 const Kokkos::View<uint16_t *> & sizes,
                                                                 const teno_setup::TableBatch::V<uint16_t **> & segment_sizes,
                                                                 const teno_setup::SlotArray<rtype> & batch_pinv,
                                                                 const std::vector<uint32_t> & same,
                                                                 Kokkos::View<uint32_t *> & row) {
            const uint32_t w = width;
            const bool sectors = by_face;
            auto h_sizes = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), Kokkos::subview(sizes, std::make_pair(0u, n)));
            // Rows: the slots that have their own table, in slot order
            std::vector<uint32_t> h_row(n), h_slot;
            std::vector<uint64_t> ids;
            if (!same.empty()) {
                for (uint32_t t = 0; t < n; t++) {
                    const uint64_t id = gid != nullptr ? gid[c0 + t] : uint64_t(c0 + t);
                    if (same[t] > t || same[same[t]] != same[t]) {
                        throw std::logic_error("TENO: a slot shares the table of a later slot or of one without its own.");
                    }
                    if (same[t] == t) {
                        h_row[t] = static_cast<uint32_t>(h_slot.size());
                        h_slot.push_back(t);
                        ids.push_back(id);
                        continue;
                    }
                    h_row[t] = h_row[same[t]];
                    ids[h_row[t]] = std::min(ids[h_row[t]], id);
                }
            } else {
                h_slot.resize(n);
                std::iota(h_slot.begin(), h_slot.end(), 0u);
                h_row = h_slot;
                for (uint32_t t = 0; t < n; t++) ids.push_back(gid != nullptr ? gid[c0 + t] : uint64_t(c0 + t));
            }
            const uint32_t n_rows = static_cast<uint32_t>(h_slot.size());
            row = Kokkos::View<uint32_t *>(Kokkos::view_alloc(Kokkos::WithoutInitializing, "teno_share_row"), n);
            Kokkos::deep_copy(row, HostUnmanaged<const uint32_t>(h_row.data(), n));
            Kokkos::View<uint32_t *> slot(Kokkos::view_alloc(Kokkos::WithoutInitializing, "teno_share_slot"), n_rows);
            Kokkos::deep_copy(slot, HostUnmanaged<const uint32_t>(h_slot.data(), n_rows));
            size_t most = 0;
            for (uint32_t t = 0; t < n; t++) most = std::max<size_t>(most, h_sizes(t));
            Kokkos::View<uint16_t **, Kokkos::LayoutLeft> order(Kokkos::view_alloc(Kokkos::WithoutInitializing, "teno_share_order"),
                                                                n_rows, most);
            Kokkos::View<uint64_t *> hash(Kokkos::view_alloc(Kokkos::WithoutInitializing, "teno_share_hash"), n_rows);
            Kokkos::View<int32_t *> exponent(Kokkos::view_alloc(Kokkos::WithoutInitializing, "teno_share_exponent"), n_rows);
            gated::parallel_for("teno_share_canonical", Kokkos::RangePolicy<>(0, n_rows), KOKKOS_LAMBDA(const uint32_t r) {
                const uint32_t t = slot(r);
                uint16_t segments[teno::MAX_FACES];
                int n_segments = 1;
                if (sectors) {
                    for (int k = 0; k < teno::MAX_FACES; k++) segments[k] = segment_sizes(t, k);
                    n_segments = teno::MAX_FACES;
                } else {
                    segments[0] = sizes(t);
                }
                const auto lp = batch_pinv.lane(t);
                int e = 0;
                hash(r) = teno::canonicalize([&](const uint32_t i) { return lp[i]; }, w, segments, n_segments,
                                             [&](const uint32_t s) -> uint16_t & { return order(r, s); }, e);
                exponent(r) = e;
            });
            auto h_hash = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), hash);
            auto h_exponent = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), exponent);
            typename teno_setup::TableBatch::V<uint16_t **>::host_mirror_type h_segments;
            if (sectors) {
                h_segments = Kokkos::create_mirror_view_and_copy(
                    Kokkos::HostSpace(), Kokkos::subview(segment_sizes, std::make_pair(0u, n), Kokkos::ALL));
            }
            std::vector<teno::CanonicalTable> canonical(n_rows);
            for (uint32_t r = 0; r < n_rows; r++) {
                const uint32_t t = h_slot[r];
                teno::CanonicalTable & c = canonical[r];
                c.order.resize(h_sizes(t));
                if (sectors) {
                    c.segments.resize(teno::MAX_FACES);
                    for (int k = 0; k < teno::MAX_FACES; k++) c.segments[size_t(k)] = h_segments(t, k);
                } else {
                    c.segments = {static_cast<uint16_t>(h_sizes(t))};
                }
                c.exponent = h_exponent(r);
                c.hash = h_hash(r);
            }
            auto compare = [&](const std::vector<teno::TableStore<>::Match> & matches) {
                return share_compare(matches, sizes, slot, exponent, order, batch_pinv);
            };
            const size_t first_new = store.size();
            std::vector<std::pair<uint32_t, uint32_t>> taken;
            const std::vector<uint32_t> shapes = store.add(n_rows, canonical.data(), ids.data(), compare, taken);
            for (uint32_t t = 0; t < n; t++) cell_shape.push_back(shapes[h_row[t]]);
            // The tables the shapes take, in canonical order; new shapes (the last of taken) also join the pool
            const size_t n_taken = taken.size();
            Kokkos::View<uint64_t **, Kokkos::HostSpace> h_take("teno_share_take", n_taken + 1, 2);
            size_t first_pooled = n_taken;
            for (size_t i = 0; i < n_taken; i++) {
                h_take(i, 0) = taken[i].second;
                h_take(i + 1, 1) = h_take(i, 1) + uint64_t(h_sizes(h_slot[taken[i].second])) * w;
                if (taken[i].first >= first_new && first_pooled == n_taken) first_pooled = i;
            }
            auto take = Kokkos::create_mirror_view_and_copy(Kokkos::DefaultExecutionSpace::memory_space(), h_take);
            Kokkos::View<rtype *> rows(Kokkos::view_alloc(Kokkos::WithoutInitializing, "teno_share_rows"), h_take(n_taken, 1));
            gated::parallel_for("teno_share_take", Kokkos::RangePolicy<>(0, n_taken), KOKKOS_LAMBDA(const size_t i) {
                const uint32_t r = static_cast<uint32_t>(take(i, 0));
                const uint32_t t = slot(r);
                const auto lp = batch_pinv.lane(t);
                for (uint32_t s = 0; s < sizes(t); s++) {
                    for (uint32_t l = 0; l < w; l++) rows(take(i, 1) + s * w + l) = lp[order(r, s) * w + l];
                }
            });
            std::vector<rtype> h_rows(rows.extent(0));
            Kokkos::deep_copy(HostUnmanaged<rtype>(h_rows.data(), h_rows.size()), rows);
            for (size_t i = 0; i < n_taken; i++) store.set_values(taken[i].first, h_rows.data() + h_take(i, 1));
            pool_at.resize(store.size(), 0);
            const size_t n_pooled = h_take(n_taken, 1) - h_take(first_pooled, 1);
            if (n_pooled > 0) {
                if (pool_used + n_pooled > pool.extent(0)) {
                    Kokkos::View<rtype *> grown(Kokkos::view_alloc(Kokkos::WithoutInitializing, "teno_share_pool"),
                                                std::max(2 * pool.extent(0), pool_used + n_pooled));
                    const std::pair<size_t, size_t> used(0, pool_used);
                    Kokkos::deep_copy(Kokkos::subview(grown, used), Kokkos::subview(pool, used));
                    pool = grown;
                }
                Kokkos::deep_copy(Kokkos::subview(pool, std::make_pair(pool_used, pool_used + n_pooled)),
                                  Kokkos::subview(rows, std::make_pair(size_t(h_take(first_pooled, 1)), size_t(h_take(n_taken, 1)))));
                for (size_t i = first_pooled; i < n_taken; i++) pool_at[taken[i].first] = pool_used + h_take(i, 1) - h_take(first_pooled, 1);
                pool_used += n_pooled;
            }
            return order;
        }

    private:
        struct Chunk {
            uint64_t slot0 = 0;
            PageArray<int32_t> cells;
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

        /** @brief Slots of each segment of t's stencil (one segment, or each face's sector stencil). */
        std::vector<uint16_t> segments(const CellTables & t) const {
            if (by_face) return std::vector<uint16_t>(t.small_size.begin(), t.small_size.end());
            return {static_cast<uint16_t>((t.*cells).size())};
        }

        /** @brief add() with the cells of a stencil shape sharing its table, their slots in canonical order. */
        void add_shared(const uint32_t c0, const std::vector<CellTables> & tables) {
            const uint32_t slice = 1u << shift;
            if (c0 % slice != 0 || c0 >> shift != slice_start.size() - 1) {
                throw std::logic_error("TENO: stencil chunks must be packed in order.");
            }
            const uint32_t n = static_cast<uint32_t>(tables.size());
            const size_t first_slice = slice_start.size() - 1;
            for (uint32_t a = 0; a < n; a += slice) {
                size_t largest = 0;
                for (uint32_t c = a; c < std::min(n, a + slice); c++) largest = std::max(largest, (tables[c].*cells).size());
                slice_start.push_back(slice_start.back() + largest);
            }
            // Cells of one stencil shape (e.g. from a cache) hold bitwise-equal tables: each table is ranked
            // once, with the lowest global id of its cells, which gives the shapes of ranking every cell
            using Bits = std::conditional_t<sizeof(rtype) == 8, uint64_t, uint32_t>;
            std::vector<uint64_t> bits(n);
            gated::parallel_for("teno_pack_bits", Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(0, n),
                                 [&](const uint32_t c) {
                const CellTables & t = tables[c];
                uint64_t h = (t.*pinv).size();
                for (const uint16_t k : segments(t)) h = (h ^ k) * 0x9e3779b97f4a7c15ULL;
                for (const rtype v : t.*pinv) h = ((h ^ std::bit_cast<Bits>(v)) * 0x9e3779b97f4a7c15ULL) ^ (h >> 29);
                bits[c] = h;
            });
            auto same_bits = [&](const uint32_t a, const uint32_t b) {
                const auto & x = tables[a].*pinv;
                const auto & y = tables[b].*pinv;
                return x.size() == y.size() && segments(tables[a]) == segments(tables[b]) &&
                       std::memcmp(x.data(), y.data(), x.size() * sizeof(rtype)) == 0;
            };
            constexpr uint32_t NO_ROW = ~0u;
            std::vector<uint32_t> row(n), first;  // (cell): its row; (row): its first cell
            std::vector<uint64_t> ids;
            std::unordered_multimap<uint64_t, uint32_t> seen;  // bits: row
            for (uint32_t c = 0; c < n; c++) {
                const uint64_t id = gid != nullptr ? gid[c0 + c] : uint64_t(c0 + c);
                uint32_t r = NO_ROW;
                const auto range = seen.equal_range(bits[c]);
                for (auto it = range.first; it != range.second && r == NO_ROW; ++it) {
                    if (same_bits(first[it->second], c)) r = it->second;
                }
                if (r == NO_ROW) {
                    r = static_cast<uint32_t>(first.size());
                    seen.emplace(bits[c], r);
                    first.push_back(c);
                    ids.push_back(id);
                }
                row[c] = r;
                ids[r] = std::min(ids[r], id);
            }
            const uint32_t n_rows = static_cast<uint32_t>(first.size());
            std::vector<teno::CanonicalTable> canonical(n_rows);
            std::vector<uint64_t> first_value(n_rows + 1, 0);
            for (uint32_t r = 0; r < n_rows; r++) first_value[r + 1] = first_value[r] + (tables[first[r]].*pinv).size();
            std::vector<rtype> ordered(first_value[n_rows]);
            gated::parallel_for("teno_pack_canonical", Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(0, n_rows),
                                 [&](const uint32_t r) {
                const CellTables & t = tables[first[r]];
                canonical[r] = teno::CanonicalTable((t.*pinv).data(), width, segments(t));
                for (size_t s = 0; s < canonical[r].order.size(); s++) {
                    const size_t from = canonical[r].order[s];
                    for (uint32_t l = 0; l < width; l++) ordered[first_value[r] + s * width + l] = (t.*pinv)[from * width + l];
                }
            });
            const std::vector<uint32_t> shapes =
                store.add(n_rows, canonical.data(), [&](uint32_t r) { return ordered.data() + first_value[r]; }, ids.data());
            ordered = std::vector<rtype>();
            for (uint32_t c = 0; c < n; c++) cell_shape.push_back(shapes[row[c]]);
            std::vector<uint64_t> first_image(n + 1, images.size() / 2);
            for (uint32_t c = 0; c < n; c++) {
                const auto & f = tables[c].*faces;
                first_image[c + 1] = first_image[c] + static_cast<uint64_t>(std::count_if(f.begin(), f.end(),
                                                                                     [](int32_t x) { return x != -1; }));
            }
            images.resize(2 * first_image[n]);
            Chunk chunk;
            chunk.slot0 = slice_start[first_slice];
            chunk.cells = PageArray<int32_t>((slice_start.back() - chunk.slot0) << shift, 0);
            // Slots in canonical order; images are numbered in cell order, then in that order
            gated::parallel_for("teno_pack", Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(0, n),
                                 [&](const uint32_t c) {
                const CellTables & t = tables[c];
                const uint64_t start = slice_start[(c0 + c) >> shift] - chunk.slot0;
                const uint32_t lane = (c0 + c) & (slice - 1);
                uint64_t image = first_image[c];
                for (size_t s = 0; s < (t.*cells).size(); s++) {
                    const uint16_t from = canonical[row[c]].order[s];
                    const int32_t cell = (t.*cells)[from], face = (t.*faces)[from];
                    int32_t entry = cell;
                    if (face != -1) {
                        images[2 * image] = cell;
                        images[2 * image + 1] = face;
                        entry = teno::PackedStencils::image_entry(image++);
                    }
                    chunk.cells[((start + s) << shift) + lane] = entry;
                }
            });
            chunks.push_back(std::move(chunk));
        }

        /**
         * @brief The shapes' tables, agreed over the ranks, packed like the
         *        stencils of cells: shape k is lane k mod 2^shift of block
         *        k >> shift, which a mesh of distinct tables packs exactly as
         *        its cells.
         */
        void finish_shared(teno::PackedStencils & out, const std::string & label) {
            pool = Kokkos::View<rtype *>();
            pool_at = std::vector<uint64_t>();
            store.agree();
            const uint32_t lanes = 1u << shift;
            const size_t n_shapes = store.size();
            const size_t n_blocks = (n_shapes + lanes - 1) >> shift;
            std::vector<uint64_t> block_start(n_blocks + 1, 0);
            for (size_t b = 0; b < n_blocks; b++) {
                uint64_t largest = 0;
                for (size_t k = b << shift; k < std::min(n_shapes, (b + 1) << shift); k++) {
                    largest = std::max<uint64_t>(largest, store.slots(k));
                }
                block_start[b + 1] = block_start[b] + largest;
            }
            std::vector<rtype> values(teno::PackedStencils::pinv_elements((block_start[n_blocks] * width) << shift, single), 0.0);
            gated::parallel_for("teno_pack_shapes", Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(0, n_shapes),
                                 [&](const size_t k) {
                const std::vector<rtype> & v = store.values(k);
                const uint64_t start = block_start[k >> shift];
                const size_t lane = k & (lanes - 1);
                for (size_t i = 0; i < v.size(); i++) put_value(values.data(), (((start * width) + i) << shift) + lane, v[i]);
            });
            out.shared = true;
            out.pinv = Kokkos::View<rtype *>(Kokkos::view_alloc(Kokkos::WithoutInitializing, label + "_pinv"), values.size());
            Kokkos::deep_copy(out.pinv, HostUnmanaged<const rtype>(values.data(), values.size()));
            auto shape_index = [&](const size_t k) { return (block_start[k >> shift] << shift) | (k & (lanes - 1)); };
            out.shape_owner = Kokkos::View<uint64_t *, Kokkos::HostSpace>(label + "_shape_owner", 2 * n_shapes);
            for (size_t k = 0; k < n_shapes; k++) {
                out.shape_owner(2 * k) = shape_index(k);
                out.shape_owner(2 * k + 1) = store.owner(k);
            }
            std::vector<uint64_t> index(cell_shape.size());
            for (size_t c = 0; c < index.size(); c++) index[c] = shape_index(cell_shape[c]);
            out.pinv_index = Kokkos::View<uint64_t *>(Kokkos::view_alloc(Kokkos::WithoutInitializing, label + "_pinv_index"),
                                                      index.size());
            Kokkos::deep_copy(out.pinv_index, HostUnmanaged<const uint64_t>(index.data(), index.size()));
        }

        uint8_t shift;
        uint8_t width;
        bool single;  // pseudo-inverses stored in single precision
        bool share;
        const uint64_t * gid;  // shared: (local cell), or null
        bool by_face;
        IndexRow cells, faces;
        ValueRow pinv;
        std::vector<uint64_t> slice_start = {0};  // (slice + 1)
        uint32_t n_added = 0;
        teno::TableStore<> store;                 // shared: a table per stencil shape
        Kokkos::View<rtype *> pool;               // shared, while adding: a table of each shape (canonical order)
        size_t pool_used = 0;
        std::vector<uint64_t> pool_at;            // (shape): its table in the pool
        std::vector<uint32_t> cell_shape;         // shared: (cell)
        std::vector<int32_t> images;              // (2 image + {cell, face field})
        std::vector<Chunk> chunks;
        std::vector<std::shared_ptr<GrowingBuffer>> memory;  // streamed: cells, pinv
};

/**
 * @brief Table rows rows(j) from slot j of a batch, j < n: scale, basis
 *        means, smoothness-indicator matrix and stencil sizes.
 */
void store_cell_rows(TENO & scheme, const Kokkos::View<uint32_t *> & rows, const teno_setup::TableBatch & batch,
                     const uint32_t n) {
    auto scale = scheme.scale;
    auto basis_mean = scheme.basis_mean;
    auto si = scheme.si_matrix;
    auto large_size = scheme.stencil_large_size;
    auto small_size = scheme.stencil_small_size;
    gated::parallel_for("teno_store_rows", Kokkos::RangePolicy<>(0, n), KOKKOS_LAMBDA(const uint32_t j) {
        const uint32_t c = rows(j);
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
    auto h_rows = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), Kokkos::subview(rows, std::make_pair(0u, n)));
    for (uint32_t j = 0; j < n; j++) scheme.gather_depth[h_rows(j)] = h_depth(j);
}

PackedRows large_rows(const TENO & scheme);
PackedRows small_rows(const TENO & scheme);
void download_tables(const TENO & scheme, const HostIndex & large_index, const HostIndex & small_index, uint32_t c0,
                     std::vector<CellTables> & tables);

/** @brief Pack every cell's stencils again, sharing the pseudo-inverses that are equal. */
void share_again(TENO & scheme) {
    const uint32_t n_cells = static_cast<uint32_t>(scheme.scale.extent(0));
    const HostIndex large_index = host_index(scheme.stencil_large);
    const HostIndex small_index = host_index(scheme.stencil_small);
    PackedRows large = large_rows(scheme), small = small_rows(scheme);
    std::vector<CellTables> chunk;
    for (uint32_t c0 = 0; c0 < n_cells; c0 += CHUNK_CELLS) {
        chunk.assign(std::min(CHUNK_CELLS, n_cells - c0), CellTables());
        download_tables(scheme, large_index, small_index, c0, chunk);
        large.add(c0, chunk);
        small.add(c0, chunk);
    }
    scheme.stencil_large = large.finish("teno_stencil_large");
    scheme.stencil_small = small.finish("teno_stencil_small");
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
    gated::parallel_for("teno_repack_sizes", Kokkos::RangePolicy<>(0, n_slices), KOKKOS_LAMBDA(const uint32_t sl) {
        uint64_t largest = 0;
        for (uint32_t c = sl << sh; c < ((sl + 1) << sh) && c < n_cells; c++) {
            const int32_t j = slot_of(c);
            const uint64_t size = j >= 0 ? sizes(j) : old_size(c);
            largest = size > largest ? size : largest;
        }
        start(sl) = largest;
    });
    gated::parallel_scan("teno_repack_scan", Kokkos::RangePolicy<>(0, n_slices + 1),
                          KOKKOS_LAMBDA(const uint32_t sl, uint64_t & sum, const bool final) {
        const uint64_t size = sl < n_slices ? start(sl) : 0;
        if (final) start(sl) = sum;
        sum += size;
    });
    // Images are renumbered in cell order, then slot order, as a packing from scratch numbers them
    auto old_start = old.slice_start;
    auto old_pinv_index = old.pinv_index;
    auto old_cells = old.cells;
    auto old_images = old.images;
    auto old_pinv = old.pinv;
    Kokkos::View<uint64_t *> first_image("teno_pack_first_image", n_cells + 1);
    gated::parallel_scan("teno_repack_count_images", Kokkos::RangePolicy<>(0, n_cells + 1),
                          KOKKOS_LAMBDA(const uint32_t c, uint64_t & sum, const bool final) {
        if (final) first_image(c) = sum;
        if (c == n_cells) return;
        const int32_t j = slot_of(c);
        if (j >= 0) {
            const auto lf = batch_faces.lane(static_cast<size_t>(j));
            for (uint32_t s = 0; s < sizes(j); s++) sum += lf[s] != -1 ? 1u : 0u;
        } else {
            const uint64_t o0 = old_start(c >> sh);
            const uint32_t lane = c & ((1u << sh) - 1);
            for (uint64_t s = 0; s < old_size(c); s++) sum += old_cells(((o0 + s) << sh) + lane) < 0 ? 1u : 0u;
        }
    });
    uint64_t total = 0, n_images = 0;
    Kokkos::deep_copy(total, Kokkos::subview(start, n_slices));
    Kokkos::deep_copy(n_images, Kokkos::subview(first_image, n_cells));
    teno::PackedStencils out;
    out.shift = sh;
    out.width = w;
    out.single = old.single;
    out.slice_start = start;
    out.pinv_index = Kokkos::View<uint64_t *>(Kokkos::view_alloc(Kokkos::WithoutInitializing, label + "_pinv_index"), n_cells);
    auto new_index = out.pinv_index;
    out.cells = Kokkos::View<int32_t *>(Kokkos::view_alloc(Kokkos::WithoutInitializing, label), total << sh);
    out.images = Kokkos::View<int32_t *>(Kokkos::view_alloc(Kokkos::WithoutInitializing, label + "_images"), 2 * n_images);
    out.pinv = Kokkos::View<rtype *>(Kokkos::view_alloc(Kokkos::WithoutInitializing, label + "_pinv"),
                                     teno::PackedStencils::pinv_elements((total << sh) * w, old.single));
    const bool sg = old.single;
    auto new_cells = out.cells;
    auto new_images = out.images;
    auto new_pinv = out.pinv;
    gated::parallel_for("teno_repack", Kokkos::RangePolicy<>(0, size_t(n_slices) << sh), KOKKOS_LAMBDA(const uint32_t c) {
        const uint32_t sl = c >> sh;
        const uint32_t lane = c & ((1u << sh) - 1);
        const uint64_t s0 = start(sl);
        const uint64_t len = start(sl + 1) - s0;
        const int32_t j = c < n_cells ? slot_of(c) : -1;
        const uint64_t size = c >= n_cells ? 0 : (j >= 0 ? sizes(j) : old_size(c));
        const uint64_t o0 = old_start(sl);
        const uint64_t t_old = c < n_cells ? old_pinv_index(c) : 0;
        const uint64_t o0_pinv = t_old >> sh;
        const uint64_t lane_pinv = t_old & ((1u << sh) - 1);
        if (c < n_cells) new_index(c) = (s0 << sh) | lane;
        const auto lc = batch_cells.lane(j >= 0 ? static_cast<size_t>(j) : 0);
        const auto lf = batch_faces.lane(j >= 0 ? static_cast<size_t>(j) : 0);
        const auto lp = batch_pinv.lane(j >= 0 ? static_cast<size_t>(j) : 0);
        uint64_t image = c < n_cells ? first_image(c) : 0;
        for (uint64_t s = 0; s < len; s++) {
            const uint64_t k = ((s0 + s) << sh) + lane;
            const uint64_t ko = ((o0 + s) << sh) + lane;
            int32_t entry = 0;
            if (s < size) {
                int32_t cell, face;
                if (j >= 0) {
                    cell = lc[s];
                    face = lf[s];
                } else {
                    const int32_t e = old_cells(ko);
                    cell = e >= 0 ? e : old_images(2 * (-1 - e));
                    face = e >= 0 ? -1 : old_images(2 * (-1 - e) + 1);
                }
                entry = cell;
                if (face != -1) {
                    new_images(2 * image) = cell;
                    new_images(2 * image + 1) = face;
                    entry = teno::PackedStencils::image_entry(image++);
                }
            }
            new_cells(k) = entry;
            for (uint32_t l = 0; l < w; l++) {
                const uint64_t kp = (((s0 + s) * w + l) << sh) + lane;
                const uint64_t ko_p = (((o0_pinv + s) * w + l) << sh) + lane_pinv;
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
 * @brief Replace the tables of rows rows(j), j < n (distinct), by slot j of a
 *        batch, keeping every other row's.
 */
void merge_tables(TENO & scheme, const Kokkos::View<uint32_t *> & rows, const teno_setup::TableBatch & batch,
                  const uint32_t n) {
    const uint32_t n_cells = static_cast<uint32_t>(scheme.scale.extent(0));
    Kokkos::View<int32_t *> slot_of("teno_slot_of", n_cells);
    Kokkos::deep_copy(slot_of, -1);
    gated::parallel_for("teno_slot_of", Kokkos::RangePolicy<>(0, n),
                         KOKKOS_LAMBDA(const uint32_t j) { slot_of(rows(j)) = static_cast<int32_t>(j); });
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
    store_cell_rows(scheme, rows, batch, n);
    Kokkos::fence();
}

PackedRows large_rows(const TENO & scheme) {
    return PackedRows(scheme.slice_shift, scheme.n_dof_large, &CellTables::large_cells, &CellTables::large_faces,
                      &CellTables::large_pinv, scheme.single_tables, scheme.share_tables,
                      scheme.global_cell_ids(), false);
}

PackedRows small_rows(const TENO & scheme) {
    return PackedRows(scheme.slice_shift, teno::NK_SMALL, &CellTables::small_cells, &CellTables::small_faces,
                      &CellTables::small_pinv, scheme.single_tables, scheme.share_tables,
                      scheme.global_cell_ids(), true);
}

/** @brief Host copy of the rows of cells [c0, c1) of an array with a row per cell shape. */
template <typename View>
typename View::host_mirror_type download_shape_rows(const View & dev, const Kokkos::View<uint32_t *> & shape,
                                                    const uint32_t c0, const uint32_t c1) {
    View rows(Kokkos::view_alloc(Kokkos::WithoutInitializing, "teno_shape_rows"), c1 - c0, dev.extent(1));
    gated::parallel_for("teno_shape_rows", Kokkos::RangePolicy<>(0, c1 - c0), KOKKOS_LAMBDA(const uint32_t j) {
        for (size_t k = 0; k < rows.extent(1); k++) rows(j, k) = dev(shape(c0 + j), k);
    });
    return download_rows(rows, 0, c1 - c0);
}

/** @brief An array with a row per cell shape as one with a row per cell. */
template <typename View>
void expand_shape_rows(View & dev, const Kokkos::View<uint32_t *> & shape, const std::string & label) {
    if (dev.extent(1) == 0) return;
    View rows(Kokkos::view_alloc(Kokkos::WithoutInitializing, label), shape.extent(0), dev.extent(1));
    const View from = dev;
    gated::parallel_for("teno_expand_rows", Kokkos::RangePolicy<>(0, shape.extent(0)), KOKKOS_LAMBDA(const uint32_t c) {
        for (size_t k = 0; k < rows.extent(1); k++) rows(c, k) = from(shape(c), k);
    });
    dev = rows;
}

/** @brief Every cell its own cell shape. */
void own_cell_shapes(TENO & scheme) {
    scheme.cell_shape = Kokkos::View<uint32_t *>(Kokkos::view_alloc(Kokkos::WithoutInitializing, "teno_cell_shape"),
                                                 scheme.scale.extent(0));
    auto shape = scheme.cell_shape;
    gated::parallel_for("teno_own_cell_shapes", Kokkos::RangePolicy<>(0, shape.extent(0)),
                        KOKKOS_LAMBDA(const uint32_t c) { shape(c) = c; });
    scheme.cell_shape_owner.clear();
}

/** @brief Rows of basis_mean and si_matrix or moments per cell, from rows per cell shape. */
void expand_cell_rows(TENO & scheme) {
    expand_shape_rows(scheme.basis_mean, scheme.cell_shape, "teno_basis_mean");
    if constexpr (N_DIM == 2) {
        expand_shape_rows(scheme.si_matrix, scheme.cell_shape, "teno_si_matrix");
    } else {
        expand_shape_rows(scheme.moments, scheme.cell_shape, "teno_moments");
    }
    own_cell_shapes(scheme);
}

/** @brief A cell's row of basis means and smoothness-indicator matrix (2D) or moments (3D), in double. */
struct CellRow {
    Kokkos::View<rtype **> basis_mean, si_matrix;
    teno::Moments moments;
    uint32_t nb;

    KOKKOS_INLINE_FUNCTION double operator()(const uint32_t c, const uint32_t i) const {
        if (i < nb) return double(basis_mean(c, i));
        if constexpr (N_DIM == 2) {
            return double(si_matrix(c, i - nb));
        } else {
            return moments(c, i - nb);
        }
    }
};

/** @brief For each (cell, other cell, exponent), whether their rows are equal on the grid of that exponent. */
std::vector<uint8_t> equal_cell_rows(const CellRow & row, const uint32_t width, const std::vector<std::array<int64_t, 3>> & pairs) {
    const size_t m = pairs.size();
    std::vector<uint8_t> out(m, 0);
    if (m == 0) return out;
    Kokkos::View<int64_t **, Kokkos::HostSpace> h_pairs("teno_cell_rows_pairs", m, 3);
    for (size_t i = 0; i < m; i++) {
        for (size_t k = 0; k < 3; k++) h_pairs(i, k) = pairs[i][k];
    }
    auto d_pairs = Kokkos::create_mirror_view_and_copy(Kokkos::DefaultExecutionSpace::memory_space(), h_pairs);
    Kokkos::View<uint8_t *> equal("teno_cell_rows_equal", m);
    gated::parallel_for("teno_cell_rows_compare", Kokkos::RangePolicy<>(0, m), KOKKOS_LAMBDA(const size_t i) {
        const uint32_t a = static_cast<uint32_t>(d_pairs(i, 0)), b = static_cast<uint32_t>(d_pairs(i, 1));
        const int e = static_cast<int>(d_pairs(i, 2));
        uint8_t eq = 1;
        for (uint32_t k = 0; k < width && eq; k++) eq = teno::on_grid(row(a, k), e) == teno::on_grid(row(b, k), e);
        equal(i) = eq;
    });
    auto h_equal = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), equal);
    for (size_t i = 0; i < m; i++) out[i] = h_equal(i);
    return out;
}

/** @brief Host copy of the rows of cells, one after another. */
std::vector<double> cell_rows(const CellRow & row, const uint32_t width, const std::vector<uint32_t> & cells) {
    std::vector<double> out(cells.size() * width);
    if (cells.empty()) return out;
    Kokkos::View<uint32_t *> d_cells(Kokkos::view_alloc(Kokkos::WithoutInitializing, "teno_cell_rows_cells"), cells.size());
    Kokkos::deep_copy(d_cells, HostUnmanaged<const uint32_t>(cells.data(), cells.size()));
    Kokkos::View<double *> rows(Kokkos::view_alloc(Kokkos::WithoutInitializing, "teno_cell_rows"), out.size());
    gated::parallel_for("teno_cell_rows_take", Kokkos::RangePolicy<>(0, cells.size()), KOKKOS_LAMBDA(const size_t i) {
        for (uint32_t k = 0; k < width; k++) rows(i * width + k) = row(d_cells(i), k);
    });
    Kokkos::deep_copy(HostUnmanaged<double>(out.data(), out.size()), rows);
    return out;
}

/**
 * @brief From rows of basis_mean and si_matrix or moments per cell, with
 *        share_tables, one row per cell shape: the cells whose rows are equal
 *        on the grid of teno::CanonicalTable share that of the one with the
 *        lowest global id, agreed over the ranks (teno::TableStore, in
 *        double); without, every cell keeps its own.
 */
void share_cell_rows(TENO & scheme) {
    const uint32_t n_cells = static_cast<uint32_t>(scheme.scale.extent(0));
    if (!scheme.share_tables) {
        own_cell_shapes(scheme);
        return;
    }
    const size_t nb = scheme.basis_mean.extent(1);
    const size_t ns = N_DIM == 2 ? scheme.si_matrix.extent(1) : scheme.moments.extent(1);
    const uint32_t width = static_cast<uint32_t>(nb + ns);
    const uint64_t * gid = scheme.global_cell_ids();
    const CellRow row{scheme.basis_mean, scheme.si_matrix, scheme.moments, static_cast<uint32_t>(nb)};
    teno::TableStore<double> store(width);
    std::vector<uint32_t> shape_of;
    shape_of.reserve(n_cells);
    std::vector<uint32_t> held;  // (shape): a cell of it
    Kokkos::View<uint64_t *> hash(Kokkos::view_alloc(Kokkos::WithoutInitializing, "teno_cell_rows_hash"), CHUNK_CELLS);
    Kokkos::View<int32_t *> exponent(Kokkos::view_alloc(Kokkos::WithoutInitializing, "teno_cell_rows_exponent"), CHUNK_CELLS);
    for (uint32_t c0 = 0; c0 < n_cells; c0 += CHUNK_CELLS) {
        const uint32_t n = std::min(n_cells - c0, CHUNK_CELLS);
        gated::parallel_for("teno_cell_rows_canonical", Kokkos::RangePolicy<>(0, n), KOKKOS_LAMBDA(const uint32_t j) {
            const uint16_t one = 1;
            uint16_t order = 0;
            int e = 0;
            hash(j) = teno::canonicalize([&](const uint32_t i) { return row(c0 + j, i); }, width, &one, 1,
                                         [&](const uint32_t) -> uint16_t & { return order; }, e);
            exponent(j) = e;
        });
        const std::pair<uint32_t, uint32_t> chunk(0, n);
        auto h_hash = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), Kokkos::subview(hash, chunk));
        auto h_exponent = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), Kokkos::subview(exponent, chunk));
        std::vector<teno::CanonicalTable> canonical(n);
        std::vector<uint64_t> ids(n);
        for (uint32_t j = 0; j < n; j++) {
            canonical[j].order.resize(1);
            canonical[j].segments = {1};
            canonical[j].exponent = h_exponent(j);
            canonical[j].hash = h_hash(j);
            ids[j] = gid != nullptr ? gid[c0 + j] : uint64_t(c0 + j);
        }
        auto same = [&](const std::vector<teno::TableStore<double>::Match> & matches) {
            std::vector<std::array<int64_t, 3>> pairs(matches.size());
            for (size_t i = 0; i < matches.size(); i++) {
                const auto & m = matches[i];
                pairs[i] = {c0 + m.cell, m.held >= 0 ? c0 + m.held : int64_t(held[m.shape]), canonical[m.cell].exponent};
            }
            return equal_cell_rows(row, width, pairs);
        };
        std::vector<std::pair<uint32_t, uint32_t>> taken;
        const std::vector<uint32_t> shapes = store.add(n, canonical.data(), ids.data(), same, taken);
        shape_of.insert(shape_of.end(), shapes.begin(), shapes.end());
        std::vector<uint32_t> cells(taken.size());
        for (size_t i = 0; i < taken.size(); i++) cells[i] = c0 + taken[i].second;
        const std::vector<double> rows = cell_rows(row, width, cells);
        held.resize(store.size());
        for (size_t i = 0; i < taken.size(); i++) {
            store.set_values(taken[i].first, rows.data() + i * width);
            held[taken[i].first] = cells[i];
        }
    }
    store.agree();
    const size_t n_shapes = store.size();
    Kokkos::View<rtype **>::host_mirror_type h_mean("teno_basis_mean_rows", n_shapes, nb);
    Kokkos::View<rtype **>::host_mirror_type h_si("teno_si_rows", N_DIM == 2 ? n_shapes : 0, ns);
    teno::Moments::host_mirror_type h_moments("teno_moments_rows", N_DIM == 3 ? n_shapes : 0, ns);
    scheme.cell_shape_owner.resize(n_shapes);
    for (size_t k = 0; k < n_shapes; k++) {
        const std::vector<double> & v = store.values(k);
        for (size_t l = 0; l < nb; l++) h_mean(k, l) = rtype(v[l]);
        for (size_t j = 0; j < ns; j++) {
            if constexpr (N_DIM == 2) {
                h_si(k, j) = rtype(v[nb + j]);
            } else {
                h_moments(k, j) = v[nb + j];
            }
        }
        scheme.cell_shape_owner[k] = store.owner(k);
    }
    scheme.basis_mean = Kokkos::View<rtype **>(Kokkos::view_alloc(Kokkos::WithoutInitializing, "teno_basis_mean"), n_shapes, nb);
    Kokkos::deep_copy(scheme.basis_mean, h_mean);
    if (ns > 0) {
        if constexpr (N_DIM == 2) {
            scheme.si_matrix = Kokkos::View<rtype **>(Kokkos::view_alloc(Kokkos::WithoutInitializing, "teno_si_matrix"), n_shapes, ns);
            Kokkos::deep_copy(scheme.si_matrix, h_si);
        } else {
            scheme.moments = teno::Moments(Kokkos::view_alloc(Kokkos::WithoutInitializing, "teno_moments"), n_shapes, ns);
            Kokkos::deep_copy(scheme.moments, h_moments);
        }
    }
    scheme.cell_shape = Kokkos::View<uint32_t *>(Kokkos::view_alloc(Kokkos::WithoutInitializing, "teno_cell_shape"), n_cells);
    Kokkos::deep_copy(scheme.cell_shape, HostUnmanaged<const uint32_t>(shape_of.data(), shape_of.size()));
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
            if constexpr (N_DIM == 2) {
                scheme.si_matrix = Kokkos::View<rtype **>("teno_si_matrix", n_cells,
                                                          scheme.sector_stencils ? nk * (nk + 1) / 2 : 0);
            }
            if constexpr (N_DIM == 3) scheme.moments = teno::Moments();
            scheme.stencil_large_size = Kokkos::View<uint16_t *>("teno_stencil_large_size", n_cells);
            scheme.stencil_small_size = Kokkos::View<uint16_t **>("teno_stencil_small_size", n_cells, teno::MAX_FACES);
            scheme.gather_depth.assign(n_cells, 0);
            if (GrowingBuffer::supported() && !scheme.share_tables) {
                // The searches cap central stencils at 3.5 nk + 64 entries and sectors
                // at twice small_stencil_size; the reservation is address space only
                large.stream(n_cells, 4 * size_t(nk) + 64, "teno_stencil_large");
                small.stream(n_cells, teno::MAX_FACES * 2 * size_t(scheme.n_stencil_small), "teno_stencil_small");
            }
        }

        /** @brief Tables of cells [c0, c0 + tables.size()), the chunk after the previous one. */
        void add(const uint32_t c0, const std::vector<CellTables> & tables) {
            const uint32_t n = static_cast<uint32_t>(tables.size());
            Kokkos::View<rtype *>::host_mirror_type h_scale("teno_scale_rows", n);
            Kokkos::View<rtype **>::host_mirror_type h_mean("teno_basis_mean_rows", n, scheme.basis_mean.extent(1));
            Kokkos::View<rtype **>::host_mirror_type h_si("teno_si_rows", n, scheme.si_matrix.extent(1));
            if constexpr (N_DIM == 3) {
                if (scheme.sector_stencils && !scheme.moments.is_allocated()) {
                    scheme.moments = teno::Moments("teno_moments", scheme.scale.extent(0), teno_setup::n_moments(scheme.degree));
                }
            }
            teno::Moments::host_mirror_type h_moments("teno_moments_rows", n, scheme.moments.extent(1));
            Kokkos::View<uint16_t *>::host_mirror_type h_large_size("teno_large_size_rows", n);
            Kokkos::View<uint16_t **>::host_mirror_type h_small_size("teno_small_size_rows", n, teno::MAX_FACES);
            gated::parallel_for("teno_rows", Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(0, n),
                                 [&](const uint32_t c) {
                const CellTables & t = tables[c];
                scheme.gather_depth[c0 + c] = t.gather_depth;
                h_scale(c) = t.scale;
                for (size_t l = 0; l < t.basis_mean.size(); l++) h_mean(c, l) = t.basis_mean[l];
                for (size_t k = 0; k < h_si.extent(1); k++) h_si(c, k) = t.si[k];
                for (size_t k = 0; k < h_moments.extent(1); k++) h_moments(c, k) = t.moments[k];
                h_large_size(c) = static_cast<uint16_t>(t.large_cells.size());
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

        /**
         * @brief Tables of rows [c0, c0 + n) = rows(0..n), the chunk after
         *        the previous one, from a batch (same: see PackedRows::add_batch()).
         */
        void add_batch(const uint32_t c0, const Kokkos::View<uint32_t *> & rows, const teno_setup::TableBatch & batch,
                       const uint32_t n, const std::vector<uint32_t> & same = {}) {
            store_cell_rows(scheme, rows, batch, n);
            large.add_batch(c0, n, batch.large_size, {}, batch.large_cells, batch.large_faces, batch.large_pinv, same);
            small.add_batch(c0, n, batch.small_total, batch.small_size, batch.small_cells, batch.small_faces, batch.small_pinv,
                            same);
        }

        void finish() {
            scheme.stencil_large = large.finish("teno_stencil_large");
            scheme.stencil_small = small.finish("teno_stencil_small");
            share_cell_rows(scheme);
            scheme.table_memory = large.device_memory();
            scheme.table_memory.insert(scheme.table_memory.end(), small.device_memory().begin(),
                                       small.device_memory().end());
        }

    private:
        TENO & scheme;
        PackedRows large, small;
};

/** @brief Slot j of from into slot to_slot[j] of to. */
void scatter_slots(const teno_setup::TableBatch & from, const std::vector<uint32_t> & to_slot, const teno_setup::TableBatch & to) {
    const uint32_t n = static_cast<uint32_t>(to_slot.size());
    if (n == 0) return;
    Kokkos::View<uint32_t *> slot(Kokkos::view_alloc(Kokkos::WithoutInitializing, "teno_shape_slots"), n);
    Kokkos::deep_copy(slot, HostUnmanaged<const uint32_t>(to_slot.data(), n));
    const size_t nk = from.nk;
    gated::parallel_for("teno_shape_scatter", Kokkos::RangePolicy<>(0, n), KOKKOS_LAMBDA(const uint32_t j) {
        const uint32_t k = slot(j);
        to.status(k) = from.status(j);
        to.gather_depth(k) = from.gather_depth(j);
        to.failed_large(k) = from.failed_large(j);
        to.invalid_small(k) = from.invalid_small(j);
        to.large_size(k) = from.large_size(j);
        to.small_total(k) = from.small_total(j);
        for (int f = 0; f < teno::MAX_FACES; f++) to.small_size(k, f) = from.small_size(j, f);
        auto copy = [&](const auto & a, const auto & b, const size_t count) {
            const auto x = a.lane(j);
            const auto y = b.lane(k);
            for (size_t i = 0; i < count; i++) y[i] = x[i];
        };
        const size_t n_large = from.large_size(j), n_small = from.small_total(j);
        copy(from.scale, to.scale, 1);
        copy(from.basis_mean, to.basis_mean, from.basis_mean.cap);
        copy(from.si, to.si, from.si.cap);
        copy(from.large_cells, to.large_cells, n_large);
        copy(from.large_faces, to.large_faces, n_large);
        copy(from.large_pinv, to.large_pinv, n_large * nk);
        copy(from.small_cells, to.small_cells, n_small);
        copy(from.small_faces, to.small_faces, n_small);
        copy(from.small_pinv, to.small_pinv, n_small * teno::NK_SMALL);
    });
}

/**
 * @brief With shared tables, the setup fits one cell of each stencil shape
 *        (teno::ShapeKeys): the cell with the lowest global id of each key,
 *        over the ranks. The other cells of the key take its stencil, mapped
 *        to their own cells by position in the key's order, its
 *        pseudo-inverses, basis means and smoothness matrix; their scale is
 *        their own. Cells without a key, alone in theirs, whose key's
 *        representative has a stencil the key does not describe, or whose
 *        neighborhood does not match the representative's, are fitted. Its
 *        cells are table rows [0, n_direct_rows); the extra rows are fitted.
 */
class KeyedSetup {
    public:
        KeyedSetup(const TENO & scheme, const Mesh & mesh, const BoundaryData & boundaries,
                   teno_setup::SetupHandle & setup_in, const Kokkos::View<uint32_t *> & cells, teno_setup::TableBatch & out)
            : setup(setup_in), keys(mesh, boundaries, setup_in.max_large(), scheme.n_stencil_small, out.slots),
              bad("teno_shape_bad", out.slots), forced(scheme.setup_refit_cells) {
            Kokkos::Timer timer;
            const uint32_t n = scheme.n_direct_rows;
            const uint32_t n_rows = scheme.n_rows();
            const teno::ShapeGroups groups = teno::ShapeGroups::build(keys.keys(n), scheme.global_cell_ids());
            std::vector<std::vector<uint64_t>> records(groups.groups.size());
            std::vector<uint32_t> reps, rep_group;
            for (size_t k = 0; k < groups.groups.size(); k++) {
                const auto & g = groups.groups[k];
                if (g.single || g.rep_cell == teno::ShapeGroups::NONE) continue;
                reps.push_back(g.rep_cell);
                rep_group.push_back(uint32_t(k));
            }
            const uint32_t batch = static_cast<uint32_t>(cells.extent(0));
            Kokkos::View<uint32_t *> size("teno_shape_sizes", batch);
            for (size_t r0 = 0; r0 < reps.size(); r0 += batch) {
                const uint32_t nb = static_cast<uint32_t>(std::min<size_t>(batch, reps.size() - r0));
                Kokkos::deep_copy(Kokkos::subview(cells, std::make_pair(0u, nb)),
                                  HostUnmanaged<const uint32_t>(reps.data() + r0, nb));
                setup.compute(cells, nb, out);
                keys.encode(cells, nb, out, size);
                std::vector<std::vector<uint64_t>> chunk = teno::ShapeKeys::records(out, nb, size);
                for (uint32_t j = 0; j < nb; j++) records[rep_group[r0 + j]] = std::move(chunk[j]);
            }
            groups.exchange(records);
            host_group.assign(n_rows, teno::ShapeGroups::NONE);
            for (uint32_t c = 0; c < n; c++) {
                const uint32_t k = groups.group[c];
                if (k != teno::ShapeGroups::NONE && !groups.groups[k].single && !records[k].empty() && records[k][0] == 1) {
                    host_group[c] = k;
                }
            }
            group = Kokkos::View<uint32_t *>(Kokkos::view_alloc(Kokkos::WithoutInitializing, "teno_shape_groups"), n_rows);
            Kokkos::deep_copy(group, HostUnmanaged<const uint32_t>(host_group.data(), n_rows));
            table = teno::ShapeKeys::upload(records);
            n_representatives = static_cast<uint32_t>(reps.size());
            seconds = timer.seconds();
        }

        /** @brief Rows of [c0, c0 + n) that are fitted. */
        std::vector<uint32_t> to_fit(const uint32_t c0, const uint32_t n) const {
            std::vector<uint32_t> fit;
            for (uint32_t c = c0; c < c0 + n; c++) {
                if (host_group[c] == teno::ShapeGroups::NONE) fit.push_back(c);
            }
            return fit;
        }

        /**
         * @brief Makes out the batch of rows [c0, c0 + n), row c0 + j in
         *        slot j: the fitted ones (fit, in slots 0.. of out) move to
         *        their slots, and the others take their keys' records. Sets
         *        same() for the batch.
         * @return The cells fitted because they do not match their records.
         */
        uint32_t batch(const uint32_t c0, const uint32_t n, const std::vector<uint32_t> & fit,
                       const Kokkos::View<uint32_t *> & cells, teno_setup::TableBatch & out) {
            same_slot.clear();
            if (fit.size() == n) return 0;
            const uint32_t n_fit = static_cast<uint32_t>(fit.size());
            std::vector<uint32_t> fit_slot(n_fit);
            for (size_t j = 0; j < n_fit; j++) fit_slot[j] = fit[j] - c0;
            teno_setup::TableBatch fitted;
            if (n_fit > 0) {
                fitted = teno_setup::TableBatch(n_fit, out.nk, out.max_large, out.max_small, out.large_pinv.interleaved);
                std::vector<uint32_t> slots(n_fit);
                std::iota(slots.begin(), slots.end(), 0u);
                scatter_slots(out, slots, fitted);
            }
            keys.decode(c0, n, group, table, out, bad);
            for (const uint32_t c : forced) {
                if (c >= c0 && c < c0 + n && host_group[c] != teno::ShapeGroups::NONE) {
                    Kokkos::deep_copy(Kokkos::subview(bad, c - c0), uint8_t(1));
                }
            }
            scatter_slots(fitted, fit_slot, out);
            uint32_t n_bad = 0;
            const auto flags = bad;
            gated::parallel_reduce("teno_shape_bad", Kokkos::RangePolicy<>(0, n),
                                    KOKKOS_LAMBDA(const uint32_t t, uint32_t & sum) { sum += flags(t); }, n_bad);
            std::vector<uint8_t> refit(n, 0);
            if (n_bad > 0) {
                auto h_bad = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), bad);
                for (uint32_t t = 0; t < n; t++) refit[t] = h_bad(t);
            }
            same_slot = teno::ShapeGroups::first_slots(host_group.data() + c0, refit.data(), n);
            if (n_bad == 0) return 0;
            std::vector<uint32_t> redo, redo_slot;
            for (uint32_t t = 0; t < n; t++) {
                if (refit[t] == 0) continue;
                redo.push_back(c0 + t);
                redo_slot.push_back(t);
            }
            Kokkos::deep_copy(Kokkos::subview(cells, std::make_pair(0u, n_bad)), HostUnmanaged<const uint32_t>(redo.data(), n_bad));
            teno_setup::TableBatch extra(n_bad, out.nk, out.max_large, out.max_small, out.large_pinv.interleaved);
            setup.compute(cells, n_bad, extra);
            scatter_slots(extra, redo_slot, out);
            return n_bad;
        }

        /** @brief (slot): the first slot of the batch with its key's tables (see PackedRows::add_batch()). */
        const std::vector<uint32_t> & same() const { return same_slot; }

        double seconds = 0.0;  // keys and representatives
        uint32_t n_representatives = 0;

    private:
        teno_setup::SetupHandle & setup;
        teno::ShapeKeys keys;
        std::vector<uint32_t> host_group;    // (row): its group if it takes the group's record, else NONE
        Kokkos::View<uint32_t *> group;      // (row): the same, on the device
        teno::ShapeKeys::Records table;
        Kokkos::View<uint8_t *> bad;         // (slot)
        std::vector<uint32_t> same_slot;  // (slot): see same()
        std::vector<uint32_t> forced;  // cells refitted although they match their records (tests)
};

/** @brief Tables of the chunk of reconstructed cells starting at c0 from TENO's device arrays. */
void download_tables(const TENO & scheme, const HostIndex & large_index, const HostIndex & small_index, const uint32_t c0,
                     std::vector<CellTables> & tables) {
    const uint32_t c1 = c0 + static_cast<uint32_t>(tables.size());
    auto h_scale = download_rows(scheme.scale, c0, c1);
    auto h_mean = download_shape_rows(scheme.basis_mean, scheme.cell_shape, c0, c1);
    Kokkos::View<rtype **>::host_mirror_type h_si;
    teno::Moments::host_mirror_type h_moments;
    if constexpr (N_DIM == 2) {
        h_si = download_shape_rows(scheme.si_matrix, scheme.cell_shape, c0, c1);
    } else {
        h_moments = download_shape_rows(scheme.moments, scheme.cell_shape, c0, c1);
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
    large_rows(scheme).download(scheme.stencil_large, large_index, c0, n_large, tables);
    small_rows(scheme).download(scheme.stencil_small, small_index, c0, n_small, tables);
}

} // namespace

TENO::TENO() {
    type = FaceReconstructionType::TENO;
}

TENO::~TENO() {
    // Empty
}

void TENO::read_options(const Input & input) {
    const int order = find_value_or<int>(input, "order", 5);
    // The small stencils are degree 2, so the central polynomial must be at least degree 2
    if (order < 3 || order > teno::MAX_DEGREE + 1) {
        throw std::runtime_error("TENO order must be between 3 and " +
                                 std::to_string(teno::MAX_DEGREE + 1) + " (use MUSCL for second order).");
    }
    degree = static_cast<uint8_t>(order - 1);
    n_dof_large = teno::n_dof(degree);
    stencil_factor = find_real_or(input, "stencil_factor", dim::none, 2.0);
    n_stencil_large = static_cast<uint16_t>(std::ceil(stencil_factor * n_dof_large));
    n_stencil_small = static_cast<uint16_t>(
        find_value_or<int>(input, "small_stencil_size", (N_DIM == 2) ? 10 : 2 * teno::NK_SMALL));
    sigma_threshold = find_real_or(input, "troubled_threshold", dim::none, 1.0e-3_r);
    sigma_upper = find_real_or(input, "troubled_upper", dim::none, 1.0e-2_r);
    C_T = find_real_or(input, "C_T", dim::none, -1.0);
    characteristic = find_value_or<bool>(input, "characteristic", true);
    max_condition = find_real_or(input, "max_condition", dim::none, 1.0e8);
    bound_preserving = find_value_or<bool>(input, "bound_preserving", false);
    troubled_capacity = find_real_or(input, "troubled_capacity", dim::none, troubled_capacity);
    if (!(troubled_capacity > 0.0_r && troubled_capacity <= 1.0_r)) {
        throw std::runtime_error("TENO: troubled_capacity must be in (0, 1].");
    }
    curved_mirrors = find_value_or<bool>(input, "curved_mirrors", false);
    const int wall_degree = find_value_or<int>(input, "curved_wall_degree", 3);
    if (wall_degree < 2 || wall_degree > teno::MAX_DEGREE) {
        throw std::runtime_error("TENO curved_wall_degree must be between 2 and " + std::to_string(teno::MAX_DEGREE) + ".");
    }
    curved_wall_degree = uint8_t(wall_degree);
    cache_file = find_value_or<std::string>(input, "cache_file", "");
    // Single-precision builds store their tables so anyway
    single_tables = find_value_or<bool>(input, "single_precision_tables", false) && sizeof(rtype) > sizeof(float);
    if (single_tables && N_DIM != 3) throw std::runtime_error("TENO: single_precision_tables is for 3D runs.");
    // The cache then holds the tables as the runs use them
    cache_single = find_value_or<bool>(input, "cache_single_precision", false) || single_tables;
    // One file per rank, made for this partition
    if (!cache_file.empty() && comm::size() > 1) {
        cache_file += ".r" + std::to_string(comm::rank()) + "-of-" + std::to_string(comm::size());
    }
}

void TENO::init_curved_source_rules() {
    const curved::Geometry & geometry = *mesh->curved_geometry;
    std::vector<uint32_t> start(mesh->n_cells + 1, 0);
    std::vector<curved::Vec3> points, x;
    std::vector<double> weights, w;
    for (uint32_t c = 0; c < mesh->n_cells; c++) {
        if (geometry.cell_is_curved(c)) {
            geometry.cell_rule(c, degree + 1, x, w);
            points.insert(points.end(), x.begin(), x.end());
            weights.insert(weights.end(), w.begin(), w.end());
        }
        start[c + 1] = uint32_t(weights.size());
    }
    curved_rule_start = Kokkos::View<uint32_t *>("teno_curved_rule_start", start.size());
    Kokkos::deep_copy(curved_rule_start, Kokkos::View<const uint32_t *, Kokkos::HostSpace, Kokkos::MemoryUnmanaged>(
                                             start.data(), start.size()));
    curved_rule = Kokkos::View<rtype *[3]>("teno_curved_rule", weights.size());
    auto h_rule = Kokkos::create_mirror_view(curved_rule);
    for (size_t q = 0; q < weights.size(); q++) {
        h_rule(q, 0) = rtype(points[q][0]);
        h_rule(q, 1) = rtype(points[q][1]);
        h_rule(q, 2) = rtype(weights[q]);
    }
    Kokkos::deep_copy(curved_rule, h_rule);
}

void TENO::init(const Input & input) {
    read_options(input);
    const int order = degree + 1;
    const int n_gp = std::max(1, std::min<int>(teno::MAX_FACE_QUAD, (order + 1) / 2));
    quadrature_face = GaussLegendre(static_cast<uint8_t>(n_gp));
    if constexpr (N_DIM == 3) {
        init_face_quadrature_3d(static_cast<uint8_t>(order));
        if (sector_stencils) si_terms = teno_setup::smoothness_terms(degree);
    }
    init_face_quadrature_2d_curved();
    if (N_DIM == 2 || !mesh->curved_geometry) {
        face_nodes = FaceNodes("teno_face_nodes", mesh->n_faces);
        auto h_face_nodes = Kokkos::create_mirror_view(face_nodes);
        for (uint32_t f = 0; f < h_face_nodes.extent(0); f++) {
            for (uint32_t k = 0; k < h_face_nodes.extent(1); k++) h_face_nodes(f, k) = host_face_node(f, k);
        }
        Kokkos::deep_copy(face_nodes, h_face_nodes);
        node_coords = NodeCoords("teno_node_coords", mesh->node_coords.extent(0));
        Kokkos::deep_copy(node_coords, mesh->node_coords);
        if (mesh->axisymmetric) {
            offsets_nodes_of_cell = mesh->offsets_nodes_of_cell;
            nodes_of_cell = mesh->nodes_of_cell;
        }
    }

    if (mesh->axisymmetric) {
        // Exact to degree >= the reconstruction's, for the geometric source
        const TriangleRule rule((degree + 3) / 2);
        cell_rule = Kokkos::View<rtype *[3]>("teno_cell_rule", rule.w.size());
        auto h_rule = Kokkos::create_mirror_view(cell_rule);
        for (size_t q = 0; q < rule.w.size(); q++) {
            h_rule(q, 0) = static_cast<rtype>(rule.xi[q]);
            h_rule(q, 1) = static_cast<rtype>(rule.eta[q]);
            h_rule(q, 2) = static_cast<rtype>(rule.w[q]);
        }
        Kokkos::deep_copy(cell_rule, h_rule);
        if (mesh->curved_geometry) init_curved_source_rules();
    }

    if (mesh->is_rotational()) {
        std::vector<int32_t> faces;
        for (uint32_t f = 0; f < mesh->n_faces; f++) {
            if (mesh->h_cells_of_face(f, 1) < 0) faces.push_back(int32_t(f));
        }
        if (int64_t(faces.size()) > teno::MAX_ROTATED_BOUNDARY_FACES) {
            throw std::runtime_error("TENO with rotational periodicity supports at most " +
                                     std::to_string(teno::MAX_ROTATED_BOUNDARY_FACES) + " boundary faces per rank.");
        }
        boundary_faces = Kokkos::View<int32_t *>("teno_boundary_faces", faces.size());
        Kokkos::deep_copy(boundary_faces, Kokkos::View<const int32_t *, Kokkos::HostSpace>(faces.data(), faces.size()));
        entry_rotations = Kokkos::View<rtype *[N_DIM][N_DIM]>("teno_entry_rotations", teno::TURN_SPAN);
        auto h_rotations = Kokkos::create_mirror_view(entry_rotations);
        for (int row = 0; row < teno::TURN_SPAN; row++) {
            const auto Q = rotation_matrix(mesh->periodic_rotation.axis,
                                           (row - teno::TURN_BIAS) * mesh->periodic_rotation.angle);
            FOR_I_DIM {
                for (uint8_t j = 0; j < N_DIM; j++) h_rotations(row, i, j) = static_cast<rtype>(Q[i][j]);
            }
        }
        Kokkos::deep_copy(entry_rotations, h_rotations);
    }
    choose_rows();
    cache_loaded = !cache_file.empty() && load_cache();
    if (!cache_loaded) {
        compute_stencils_and_matrices();
        // Runs that write the cache use the tables as they will be read back
        if (single_tables || (cache_single && !cache_file.empty())) round_to_single();
    }
    release_planar_face_points();
    allocate_scratch();
    largest_stencil = comm::allreduce(largest_stencil, comm::Op::MAX);
    find_block_stencils();
}

void TENO::release_planar_face_points() {
    // The kernels place the points of planar faces from their nodes; only the setup reads them
    if (N_DIM == 3 && !mesh->curved_geometry) face_quad_points = {};
}

void TENO::choose_rows() {
    n_direct_rows = partition_faces_exchanged ? mesh->n_owned() : mesh->n_reconstructed();
    extra_cells = partition_faces_exchanged ? halo_image_cells() : std::vector<uint32_t>();
    extra_cells_device = Kokkos::View<uint32_t *>("teno_extra_cells", extra_cells.size());
    Kokkos::deep_copy(extra_cells_device, HostUnmanaged<const uint32_t>(extra_cells.data(), extra_cells.size()));
    row_global_ids.clear();
    if (!extra_cells.empty() && !mesh->h_global_cell_id.empty()) {
        row_global_ids.assign(mesh->h_global_cell_id.begin(), mesh->h_global_cell_id.begin() + n_direct_rows);
        for (const uint32_t c : extra_cells) row_global_ids.push_back(mesh->h_global_cell_id[c]);
    }
}

int64_t TENO::row_of_cell(const uint32_t c) const {
    if (c < n_direct_rows) return c;
    const auto it = std::lower_bound(extra_cells.begin(), extra_cells.end(), c);
    if (it == extra_cells.end() || *it != c) return -1;
    return int64_t(n_direct_rows) + (it - extra_cells.begin());
}

report::Summary TENO::summary() const {
    using logging::count;
    report::Summary s;
    s.setting("Reconstruction", logging::format("TENO, order %d (degree %d)", degree + 1, degree));
    s.setting("TENO switch", std::string("characteristic ") + (characteristic ? "yes" : "no") + ", bound-preserving " +
                                 (bound_preserving ? "yes" : "no") + (C_T > 0.0_r ? "" : ", C_T adaptive"));
    if (!setup_status.empty()) s.setting("TENO setup", setup_status);
    if (!cache_status.empty()) s.setting("TENO cache", cache_status);
    if (share_tables) {
        const auto slots = [](const teno::PackedStencils & p) { 
            return std::make_pair(p.pinv.extent(0), teno::PackedStencils::pinv_elements(p.cells.extent(0) * p.width, p.single));
        };
        const auto [large_kept, large_all] = slots(stencil_large);
        const auto [small_kept, small_all] = slots(stencil_small);
        s.quantity("TENO shared tables", logging::format("%.1f", 100.0 * double(large_kept + small_kept) /
                                                                     double(std::max<size_t>(1, large_all + small_all))),
                   "%", "pseudo-inverse entries stored, of one per cell");
    }
    s.quantity("TENO face quadrature points", count(n_face_quadrature_points()), "points", "per face");
    s.quantity("TENO large stencil", count(n_stencil_large + 1u), "cells", "the cell and its neighbors");
    s.quantity("TENO largest stencil", count(largest_stencil), "cells", "");
    s.quantity("TENO small stencil", count(n_stencil_small + 1u), "cells", "the cell and its neighbors");
    if (n_sector_unavailable > 0 && comm::size() == 1) {
        s.quantity("TENO sector stencils cut", count(static_cast<uint64_t>(n_sector_unavailable)), "stencils",
                   "by boundaries");
    }
    s.quantity("TENO troubled_threshold", logging::real(double(sigma_threshold)), "-", "switch");
    if (C_T > 0.0_r) s.quantity("TENO C_T", logging::real(double(C_T)), "-", "cutoff");
    return s;
}

uint8_t TENO::n_face_quadrature_points() const {
    if constexpr (N_DIM == 3) return static_cast<uint8_t>(face_quad_weights.extent(1));
    return static_cast<uint8_t>(quadrature_face.h_points.extent(0));
}

namespace {

/** @brief cells(t) = c0 + t for t < n. */
void fill_range(const Kokkos::View<uint32_t *> & cells, const uint32_t c0, const uint32_t n) {
    gated::parallel_for("teno_setup_range", Kokkos::RangePolicy<>(0, n),
                         KOKKOS_LAMBDA(const uint32_t t) { cells(t) = c0 + t; });
}

/** @brief Failed central stencils, left-out sector stencils and the largest central stencil of slots 0..n. */
void batch_counts(const teno_setup::TableBatch & out, const uint32_t n, uint32_t & failed, uint32_t & invalid,
                  uint16_t & largest) {
    gated::parallel_reduce("teno_setup_counts", Kokkos::RangePolicy<>(0, n),
                            KOKKOS_LAMBDA(const uint32_t t, uint32_t & f, uint32_t & v, uint16_t & m) {
        f += out.failed_large(t);
        v += out.invalid_small(t);
        m = out.large_size(t) > m ? out.large_size(t) : m;
    }, failed, invalid, Kokkos::Max<uint16_t>(largest));
}

} // namespace

void TENO::compute_stencils_and_matrices(const std::vector<uint32_t> * subset) {
    const uint32_t n_cells = mesh->n_cells;
    for (uint32_t i = 0; i < n_cells; i++) {
        if (mesh->h_n_faces_of_cell(i) > teno::MAX_FACES) {
            throw std::runtime_error("TENO supports cells with at most " +
                                     std::to_string(teno::MAX_FACES) + " faces.");
        }
    }
    teno_setup::Options options;
    options.degree = degree;
    options.nk = n_dof_large;
    options.ns = n_stencil_large;
    options.nss = n_stencil_small;
    options.max_condition = double(max_condition);
    options.batch_cells = setup_batch_cells;
    options.host_only = setup_on_host;
    // The rows to set up (subset, or all) and their cells, unless both are [0, n_reconstructed)
    const uint32_t n_target = subset ? static_cast<uint32_t>(subset->size()) : n_rows();
    std::vector<uint32_t> rows, setup_cells;
    if (subset != nullptr || !extra_cells.empty() || n_target != mesh->n_reconstructed()) {
        rows = subset ? *subset : std::vector<uint32_t>(n_target);
        if (subset == nullptr) std::iota(rows.begin(), rows.end(), 0u);
        for (const uint32_t r : rows) setup_cells.push_back(cell_of_row(r));
        options.cells = &setup_cells;
    }
    options.curved_mirrors = curved_mirrors;
    options.curved_wall_degree = curved_wall_degree;
    options.wall_ns = static_cast<uint16_t>(std::ceil(stencil_factor * teno::n_dof(curved_wall_degree)));
    options.stencil_factor = double(stencil_factor);
    options.sector_stencils = sector_stencils;
    for (size_t q = 0; q < quadrature_face.h_points.extent(0); q++) options.face_points.push_back(double(quadrature_face.h_points(q, 0)));
    auto handle = std::make_unique<teno_setup::SetupHandle>(*mesh, boundaries, face_quad_points, face_quad_weights, options);
    teno_setup::SetupHandle & setup = *handle;

    // Cells per batch: whole slices of the packed stencils
    const uint32_t slice = 1u << slice_shift;
    const uint32_t batch = std::max(setup.batch_cells() / slice * slice, slice);
    teno_setup::TableBatch out(batch, n_dof_large, setup.max_large(), setup.max_small(), teno_setup::batch_interleaved());
    Kokkos::View<uint32_t *> cells("teno_setup_batch_cells", batch);
    // Where the batch's tables go: its cells' rows, which differ from the cells only with extra_cells
    const bool rows_are_cells = extra_cells.empty();
    Kokkos::View<uint32_t *> batch_rows = rows_are_cells ? cells : Kokkos::View<uint32_t *>("teno_setup_batch_rows", batch);
    std::unique_ptr<TableBuilder> builder;
    if (subset == nullptr) builder = std::make_unique<TableBuilder>(*this, n_target);
    std::unique_ptr<KeyedSetup> keyed;
    if (subset == nullptr) n_cells_fitted = n_target;
    if (subset == nullptr && share_tables && teno::ShapeKeys::supported(*mesh)) {
        keyed = std::make_unique<KeyedSetup>(*this, *mesh, boundaries, setup, cells, out);
        n_cells_fitted = keyed->n_representatives;
    }
    uint32_t n_failed_large = 0, n_invalid_small = 0;
    uint16_t largest = 0;
    double t_pack = 0.0;
    for (uint32_t c0 = 0; c0 < n_target; c0 += batch) {
        const uint32_t n = std::min(batch, n_target - c0);
        const auto first = std::make_pair(0u, n);
        if (keyed) {
            const std::vector<uint32_t> fit = keyed->to_fit(c0, n);
            const uint32_t n_fit = static_cast<uint32_t>(fit.size());
            if (n_fit > 0) {
                std::vector<uint32_t> fit_cells(n_fit);
                for (uint32_t j = 0; j < n_fit; j++) fit_cells[j] = cell_of_row(fit[j]);
                Kokkos::deep_copy(Kokkos::subview(cells, std::make_pair(0u, n_fit)),
                                  HostUnmanaged<const uint32_t>(fit_cells.data(), n_fit));
                setup.compute(cells, n_fit, out);
            }
            Kokkos::Timer pack_timer;
            n_cells_fitted += n_fit + keyed->batch(c0, n, fit, cells, out);
            uint32_t failed = 0, invalid = 0;
            uint16_t batch_largest = 0;
            batch_counts(out, n, failed, invalid, batch_largest);
            n_failed_large += failed;
            n_invalid_small += invalid;
            largest = std::max(largest, batch_largest);
            if (n_failed_large > 0) continue;
            fill_range(batch_rows, c0, n);
            builder->add_batch(c0, batch_rows, out, n, keyed->same());
            t_pack += pack_timer.seconds();
            continue;
        }
        if (setup_cells.empty()) {
            fill_range(cells, c0, n);
        } else {
            Kokkos::deep_copy(Kokkos::subview(cells, first), HostUnmanaged<const uint32_t>(setup_cells.data() + c0, n));
            if (!rows_are_cells) {
                Kokkos::deep_copy(Kokkos::subview(batch_rows, first), HostUnmanaged<const uint32_t>(rows.data() + c0, n));
            }
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
            builder->add_batch(c0, batch_rows, out, n);
        } else {
            merge_tables(*this, batch_rows, out, n);
        }
        t_pack += pack_timer.seconds();
    }
    // Every cell's moments are a function of its geometry alone
    if (N_DIM == 3 && n_failed_large == 0 && sector_stencils) moments = setup.moments(n_direct_rows, extra_cells_device);
    Kokkos::fence();
    const teno_setup::Timings t = setup.timings();
    // The packed tables are allocated after the setup's workspace is freed, so that they do not add to its peak
    out = teno_setup::TableBatch();
    cells = Kokkos::View<uint32_t *>();
    batch_rows = Kokkos::View<uint32_t *>();
    handle.reset();
    const double t_shapes = keyed ? keyed->seconds : -1.0;
    keyed.reset();
    if (subset == nullptr) {
        if (n_failed_large == 0) builder->finish();
        largest_stencil = largest;
    } else {
        if (share_tables) share_again(*this);
        share_cell_rows(*this);
    }
    setup_status = logging::format("on %s; of Setup's numerics time: geometry %.2f s, tables %.2f s, packing %.2f s",
                                   setup_on_host ? "the host" : Kokkos::DefaultExecutionSpace::name(), t.geometry,
                                   t.tables, t_pack);
    if (t_shapes >= 0.0) {
        const auto [fitted, of] = comm::allreduce(std::array<uint64_t, 2>{n_cells_fitted, n_target}, comm::Op::SUM);
        setup_status += logging::format(", stencil shapes %.2f s (%s fitted of %s)", t_shapes,
                                        logging::count(fitted).c_str(), logging::count(of).c_str());
    }
    if (t.n_host_cells > 0) {
        setup_status += logging::format(", %llu cells redone on the host in %.2f s",
                                        static_cast<unsigned long long>(t.n_host_cells), t.host_cells);
    }
    setup_producer = setup_on_host ? "host" : Kokkos::DefaultExecutionSpace::name();

    if (n_failed_large > 0) {
        throw std::runtime_error("TENO: could not build a full-rank large stencil for " +
                                 logging::count(n_failed_large, "cell") + " (mesh too small for this order?).");
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
    // SmoothPass that also writes point_gradients, a kernel of its own that leaves SmoothPass's registers alone
    struct SmoothGradientPass {};
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
    FaceNodes face_nodes;
    NodeCoords node_coords;
    Kokkos::View<rtype *[N_DIM]> cell_coords;
    Kokkos::View<rtype *[N_DIM]> face_coords;
    Kokkos::View<rtype *[N_DIM]> face_normals;
    PeriodicMaps periodic;
    Kokkos::View<uint8_t *> face_shift;
    Kokkos::View<rtype **> quad_points;
    Kokkos::View<rtype ***> face_quad_points;   // 3D: (face, q, dim), curved meshes
    FaceWeights<> face_quad_weights;            // 3D: (face, q), zero on padding points
    Kokkos::View<rtype *[4]> face_rule;         // 3D: planar faces' points (planar_face_point)
    BoundaryData boundaries;

    Kokkos::View<rtype *> scale;
    Kokkos::View<uint32_t *> cell_shape;      // (cell): its row of basis_mean and si_matrix or moments
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
    IndexSet cells;  // cells to reconstruct
    // GradientPass output; when allocated, the smooth passes also write their cells' gradients
    Kokkos::View<rtype *[N_CONSERVATIVE][N_DIM]> gradients;
    // PRIM (gas mixtures): reconstruct W itself, see TENO::set_mixture
    Kokkos::View<rtype *, Kokkos::LayoutStride> cell_gamma;
    Kokkos::View<rtype *> cell_molar_mass;
    Kokkos::View<uint8_t **> selection;
    // Rotational periodicity: TENO::entry_rotations
    Kokkos::View<rtype *[N_DIM][N_DIM]> entry_rotations;
    Kokkos::View<int32_t *> boundary_faces;
    // TENO::point_gradients: when allocated, the smooth passes also write there the gradients of their cells'
    // central polynomials at the face points
    FacePointGradients point_gradients{};
    teno::WallFits wall_fits{};
    struct WallFitPass {};

    /** @brief factor[d] = h^-d (d >= 1) of cell i_cell: its central coefficients of degree d in the unscaled frame over the scaled. */
    KOKKOS_INLINE_FUNCTION
    void unscaled_factors(const uint32_t i_cell, rtype (&factor)[DEG + 1]) const {
        const rtype inv_h = 1.0_r / scale(i_cell);
        factor[0] = 1.0_r;
        factor[1] = inv_h;
        for (uint8_t d = 2; d <= DEG; d++) factor[d] = factor[d - 1] * inv_h;
    }

    /**
     * @brief point_gradients of cell i_cell's side of face f at its point q, p in the cell's frame, with face state
     *        W_q, from the central coefficients a(l, var) (scaled frame) and unscaled_factors.
     */
    template <typename A>
    KOKKOS_INLINE_FUNCTION
    void store_point_gradient(const uint32_t i_cell, const uint32_t f, const uint8_t q, const uint8_t side,
                              const rtype * p, const rtype * W_q, A && a, const rtype (&factor)[DEG + 1]) const {
        rtype r[N_DIM], dU[N_CONSERVATIVE][N_DIM];
        FOR_I_DIM r[i] = p[i] - cell_coords(i_cell, i);
        FacePointGradients::conservative_gradient<DEG>(
            r, [&](const uint32_t l, const uint8_t d, const uint8_t k) { return a(l, k) * factor[d]; }, dU);
        if constexpr (PRIM) {
            // Gas mixtures reconstruct W; MixturePointGradientsFunctor adds the composition
            point_gradients.store_primitive(f, q, side, W_q, dU);
        } else {
            rtype v[FacePointGradients::NQ], g[FacePointGradients::NQ][N_DIM];
            point_gradients.primitive_value(W_q, v);
            point_gradients.primitive_gradient(W_q, dU, g);
            point_gradients.store(f, q, side, v, g);
        }
    }

    /** @brief Marks cell i_cell's sides of point_gradients unusable: TENO selects stencils there, or its face values are the mean. */
    KOKKOS_INLINE_FUNCTION
    void mark_point_gradients(const uint32_t i_cell) const {
        for (uint32_t k = offsets_faces_of_cell(i_cell); k < offsets_faces_of_cell(i_cell + 1); k++) {
            const uint32_t f = faces_of_cell(k);
            point_gradients.mark(f, side_of(f, i_cell));
        }
    }

    /**
     * @brief Wall fit k (TENO::wall_fits) of a cell found smooth: its values and gradients at the cell's face
     *        points replace those of its central polynomial, unless a face point is inadmissible, which marks
     *        the cell's sides unusable instead.
     */
    KOKKOS_INLINE_FUNCTION
    void operator()(WallFitPass, const uint32_t k) const {
        if constexpr (!PRIM) {
            constexpr uint8_t E = N_DIM + 1;
            const uint32_t i_cell = wall_fits.cells(k);
            const uint32_t f_begin = offsets_faces_of_cell(i_cell);
            const uint8_t n_faces = static_cast<uint8_t>(offsets_faces_of_cell(i_cell + 1) - f_begin);
            const uint32_t f0 = faces_of_cell(f_begin);
            if (!point_gradients.usable(f0, side_of(f0, i_cell))) return;
            const BoundaryCondition & bc = boundaries.bcs(wall_fits.bc(k));
            rtype u_w[N_DIM];
            FOR_I_DIM u_w[i] = bc.data[1 + i];
            const bool isothermal = bc.type == BoundaryType::WALL_ISOTHERMAL;
            const rtype e_w =
                isothermal ? point_gradients.R / (gamma - 1.0_r) * bc.data[0] + 0.5_r * dot<N_DIM>(u_w, u_w) : 0.0_r;
            // V = [rho, rho (u - u_w), rho (E - e_w)]: the wall points' rows ask for V = 0
            auto to_V = [&](rtype * x) {
                FOR_I_DIM x[1 + i] -= u_w[i] * x[0];
                if (isothermal) x[E] -= e_w * x[0];
            };
            rtype U0[N_CONSERVATIVE], V0[N_CONSERVATIVE];
            conservatives(i_cell, U0);
            FOR_I_CONSERVATIVE V0[i] = U0[i];
            to_V(V0);
            rtype a[NK][N_CONSERVATIVE] = {};
            for (uint16_t s = 0; s < wall_fits.size(k); s++) {
                rtype dV[N_CONSERVATIVE];
                conservatives(wall_fits.entries(k, s), dV);
                FOR_I_CONSERVATIVE dV[i] -= U0[i];
                to_V(dV);
                for (uint8_t l = 0; l < NK; l++) {
                    const rtype P_free = wall_fits.free_pinv(k, l, s), P_wall = wall_fits.wall_pinv(k, l, s);
                    a[l][0] = fmadd(P_free, dV[0], a[l][0]);
                    FOR_I_DIM a[l][1 + i] = fmadd(P_wall, dV[1 + i], a[l][1 + i]);
                    a[l][E] = fmadd(isothermal ? P_wall : P_free, dV[E], a[l][E]);
                }
            }
            for (uint8_t l = 0; l < NK; l++) {
                const rtype sum = wall_fits.wall_sum(k, l);
                FOR_I_DIM a[l][1 + i] = fmadd(u_w[i], a[l][0], a[l][1 + i] - sum * V0[1 + i]);
                if (isothermal) a[l][E] = fmadd(e_w, a[l][0], a[l][E] - sum * V0[E]);
            }
            const rtype h = wall_fits.scale(k);
            rtype factor[DEG + 1];
            factor[0] = 1.0_r;
            for (uint8_t d = 1; d <= DEG; d++) factor[d] = factor[d - 1] / h;
            const uint8_t n_quad = this->n_quad();
            auto point_state = [&](const rtype * p, rtype * Wq) {
                rtype xi[N_DIM], psi[NK], U_f[N_CONSERVATIVE];
                FOR_I_DIM xi[i] = (p[i] - cell_coords(i_cell, i)) / h;
                teno::monomials(DEG, xi, psi);
                FOR_I_CONSERVATIVE U_f[i] = U0[i];
                for (uint8_t l = 0; l < NK; l++) {
                    const rtype psi_l = psi[l] - wall_fits.mean(k, l);
                    FOR_I_CONSERVATIVE U_f[i] = fmadd(a[l][i], psi_l, U_f[i]);
                }
                to_primitives(U_f, Wq);
                return Wq[0] > 0.0_r && Wq[E] > 0.0_r && Kokkos::isfinite(Wq[E]);
            };
            for (uint8_t j = 0; j < n_faces; j++) {
                const uint32_t f = faces_of_cell(f_begin + j);
                const PlanarFace nodes = planar_nodes(f);
                for (uint8_t q = 0; q < n_quad; q++) {
                    rtype p[N_DIM], Wq[N_CONSERVATIVE];
                    if (!face_point(f, q, i_cell, nodes, p)) continue;
                    if (!point_state(p, Wq)) {
                        mark_point_gradients(i_cell);
                        return;
                    }
                }
            }
            for (uint8_t j = 0; j < n_faces; j++) {
                const uint32_t f = faces_of_cell(f_begin + j);
                const uint8_t side = side_of(f, i_cell);
                const PlanarFace nodes = planar_nodes(f);
                for (uint8_t q = 0; q < n_quad; q++) {
                    rtype p[N_DIM], Wq[N_CONSERVATIVE];
                    if (!face_point(f, q, i_cell, nodes, p)) continue;
                    point_state(p, Wq);
                    store_point_gradient(i_cell, f, q, side, p, Wq,
                                         [&](const uint32_t l, const uint8_t i) { return a[l][i]; }, factor);
                }
            }
        }
    }

    /** @brief The velocity of W_e by row `row` of entry_rotations. */
    KOKKOS_INLINE_FUNCTION
    void turn(const int32_t row, rtype * W_e) const {
        rtype u[N_DIM];
        FOR_I_DIM {
            u[i] = 0.0_r;
            for (uint8_t j = 0; j < N_DIM; j++) u[i] = fmadd(entry_rotations(row, i, j), W_e[1 + j], u[i]);
        }
        FOR_I_DIM W_e[1 + i] = u[i];
    }

    /**
     * @brief State of a stencil entry: cell c, or its mirror across boundary
     *        face f, turned into the reconstructed cell's frame if the face
     *        field `code` says so (teno::entry_code).
     */
    KOKKOS_INLINE_FUNCTION
    void entry_W(const int32_t c, const int32_t code, rtype * W_e) const {
        if (code < -1) {
            turned_entry_W(c, code, W_e);
            return;
        }
        FOR_I_CONSERVATIVE W_e[i] = W(c, i);
        if (code >= 0) ghost_W(c, code, W_e);
    }

    /** @brief The ghost of state W_e (cell c's, in face f's frame) across boundary face f, in place. */
    KOKKOS_INLINE_FUNCTION
    void ghost_W(const int32_t c, const int32_t f, rtype * W_e) const {
        rtype n[N_DIM], n_vec[N_DIM];
        FOR_I_DIM n_vec[i] = face_normals(f, i);
        unit<N_DIM>(n_vec, n);
        rtype W_c[N_CONSERVATIVE];
        FOR_I_CONSERVATIVE W_c[i] = W_e[i];
        rtype d = 0.0;
        FOR_I_DIM d += (face_coords(f, i) - cell_coords(c, i)) * n[i];
        boundaries.ghost_W_at(static_cast<uint32_t>(f), W_c, n, 2.0_r * Kokkos::fabs(d), W_e);
        const BoundaryCondition & bc = boundaries.bcs(boundaries.face_bc(f));
        if (bc.type == BoundaryType::FARFIELD) {
            // The characteristic state holds at the face; outside lies the free stream
            FOR_I_CONSERVATIVE W_e[i] = bc.data[i];
        }
    }

    /** @brief entry_W of an entry reached across a periodic rotation. */
    KOKKOS_FUNCTION
    void turned_entry_W(const int32_t c, const int32_t code, rtype * W_e) const {
        FOR_I_CONSERVATIVE W_e[i] = W(c, i);
        int32_t row, rel_row;
        const int32_t f = teno::entry_face(code, boundary_faces, row, rel_row);
        if (f >= 0) {
            // The ghost is taken in the face's frame
            if (rel_row >= 0) turn(rel_row, W_e);
            ghost_W(c, f, W_e);
        }
        if (row >= 0) turn(row, W_e);
    }

    KOKKOS_INLINE_FUNCTION
    void entry_conservatives(const int32_t c, const int32_t f, rtype * U) const {
        if constexpr (PRIM) {
            entry_W(c, f, U);
        } else {
            rtype W_e[N_CONSERVATIVE];
            entry_W(c, f, W_e);
            conservatives_of(W_e, U);
        }
    }

    /** @brief entry_conservatives, with W_e holding W(c) already. */
    KOKKOS_INLINE_FUNCTION
    void entry_conservatives_loaded(const int32_t c, const int32_t f, rtype * W_e, rtype * U) const {
        if (f < -1) {
            entry_conservatives(c, f, U);
            return;
        }
        if (f >= 0) ghost_W(c, f, W_e);
        if constexpr (PRIM) {
            FOR_I_CONSERVATIVE U[i] = W_e[i];
        } else {
            conservatives_of(W_e, U);
        }
    }

    KOKKOS_INLINE_FUNCTION
    void conservatives_of(const rtype * W_e, rtype * U) const {
        U[0] = W_e[0];
        FOR_I_DIM U[1 + i] = W_e[0] * W_e[1 + i];
        U[N_DIM + 1] = fmadd(0.5_r * W_e[0], fma_dot<N_DIM>(W_e + 1, W_e + 1), W_e[N_DIM + 1] / (gamma - 1.0_r));
    }

    KOKKOS_INLINE_FUNCTION
    void conservatives(const uint32_t c, rtype * U) const {
        if constexpr (PRIM) {
            FOR_I_CONSERVATIVE U[i] = W(c, i);
        } else {
            rtype W_c[N_CONSERVATIVE];
            FOR_I_CONSERVATIVE W_c[i] = W(c, i);
            conservatives_of(W_c, U);
        }
    }

    KOKKOS_INLINE_FUNCTION
    void to_primitives(const rtype * U, rtype * Wq) const {
        if constexpr (PRIM) {
            FOR_I_CONSERVATIVE Wq[i] = U[i];
        } else {
            Wq[0] = U[0];
            FOR_I_DIM Wq[1 + i] = U[1 + i] / U[0];
            Wq[N_DIM + 1] = (gamma - 1.0_r) * fmadd(-0.5_r * U[0], fma_dot<N_DIM>(Wq + 1, Wq + 1), U[N_DIM + 1]);
        }
    }

    KOKKOS_INLINE_FUNCTION
    uint8_t n_quad() const {
        if constexpr (N_DIM == 2) {
            return static_cast<uint8_t>(quad_points.extent(0));
        } else {
            return static_cast<uint8_t>(face_quad_weights.extent(1));
        }
    }

    /** @brief 3D meshes without curved faces: the nodes of face f, which face_point_basis places its points from. */
    KOKKOS_INLINE_FUNCTION
    PlanarFace planar_nodes(const uint32_t f) const {
        if constexpr (N_DIM == 2) {
            return {};
        } else {
            return face_quad_points.extent(0) > 0 ? PlanarFace() : planar_face(face_nodes, node_coords, f);
        }
    }

    /**
     * @brief Monomials (minus their cell means) at quadrature point q of face f
     *        (nodes: planar_nodes(f)), in the scaled frame of cell i_cell. False
     *        for 3D padding points.
     */
    KOKKOS_INLINE_FUNCTION
    bool face_point_basis(const uint32_t f, const uint8_t q, const uint32_t i_cell, const PlanarFace & nodes,
                          rtype * psi) const {
        rtype p[N_DIM];
        if (!face_point(f, q, i_cell, nodes, p)) return false;
        point_basis(i_cell, p, psi);
        return true;
    }

    /** @brief Monomials (minus their cell means) at point p of cell i_cell's frame, in its scaled frame. */
    KOKKOS_INLINE_FUNCTION
    void point_basis(const uint32_t i_cell, const rtype * p, rtype * psi) const {
        const rtype h = scale(i_cell);
        rtype xi[N_DIM];
        FOR_I_DIM xi[i] = (p[i] - cell_coords(i_cell, i)) / h;
        teno::monomials(DEG, xi, psi);
        const uint32_t row = cell_shape(i_cell);
        for (uint8_t l = 0; l < NK; l++) psi[l] -= basis_mean(row, l);
    }

    /** @brief Quadrature point q of face f (nodes: planar_nodes(f)) in cell i_cell's frame. False for 3D padding points. */
    KOKKOS_INLINE_FUNCTION
    bool face_point(const uint32_t f, const uint8_t q, const uint32_t i_cell, const PlanarFace & nodes,
                    rtype * p) const {
        // The face's points are in its cell 0's frame
        const uint8_t s = (cells_of_face(f, 1) == static_cast<int32_t>(i_cell)) ? face_shift(f) : 0;
        rtype x[N_DIM];
        if constexpr (N_DIM == 2) {
            if (face_quad_points.extent(0) > 0) {
                // Curved meshes: the faces' own points
                FOR_I_DIM x[i] = face_quad_points(f, q, i);
            } else {
                const uint32_t node_0 = face_nodes(f, 0);
                const uint32_t node_1 = face_nodes(f, 1);
                const rtype s_q = 0.5_r * quad_points(q, 0);
                FOR_I_DIM x[i] = face_coords(f, i) + s_q * (node_coords(node_1, i) - node_coords(node_0, i));
            }
        } else {
            if (face_quad_weights(f, q) == 0.0_r) return false;
            if (face_quad_points.extent(0) > 0) {
                FOR_I_DIM x[i] = face_quad_points(f, q, i);
            } else {
                planar_face_point(nodes, face_rule, q, x);
            }
        }
        periodic.backward(s, x, p);
        return true;
    }

    /**
     * @brief gradients(i_cell) of W from dU, the gradients at the centroid of the central polynomial of the
     *        conservative variables (of W itself with PRIM), whose linear monomials carry them.
     */
    KOKKOS_INLINE_FUNCTION
    void store_gradients(const uint32_t i_cell, const rtype (&dU)[N_CONSERVATIVE][N_DIM]) const {
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

    /** @brief Gradients of W at the centroid from the central (large-stencil) polynomial. */
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
                for (uint8_t v = 0; v < N_CONSERVATIVE; v++) dU[v][i] = fmadd(P, U[v] - U0[v], dU[v][i]);
            }
        }
        store_gradients(i_cell, dU);
    }

    // PointPass (probes, transmissive boundaries)
    struct PointPass {};
    Kokkos::View<uint32_t *> point_cells{};
    uint32_t n_direct_rows = 0;            // TENO::n_direct_rows
    Kokkos::View<uint32_t *> extra_cells{};  // TENO::extra_cells
    Kokkos::View<rtype *[N_DIM]> point_offsets{};  // from the cell centroid, point_cells.extent(0) groups
    Kokkos::View<rtype *[N_CONSERVATIVE]> point_W{};

    /**
     * @brief W at the points of a cell from its central polynomial, if the
     *        cell is smooth by SmoothPass's measure and the value admissible.
     */
    KOKKOS_INLINE_FUNCTION
    void operator()(PointPass, const uint32_t k) const {
        const uint32_t i_cell = point_cells(k);
        uint32_t i_row = i_cell;
        if (i_cell >= n_direct_rows) {
            uint32_t lo = 0, hi = static_cast<uint32_t>(extra_cells.extent(0));
            while (hi - lo > 1) {
                const uint32_t mid = (lo + hi) / 2;
                (extra_cells(mid) <= i_cell ? lo : hi) = mid;
            }
            KOKKOS_ASSERT(extra_cells(lo) == i_cell);
            i_row = n_direct_rows + lo;
        }
        rtype U0[N_CONSERVATIVE];
        conservatives(i_cell, U0);
        const rtype rho0 = W(i_cell, 0);
        const uint16_t ns = stencil_large_size(i_row);
        rtype aK[NK][N_CONSERVATIVE] = {};
        rtype g_mean = 0.0, g_m2 = 0.0, m_mean = 0.0, m_m2 = 0.0;
        const auto stencil = stencil_large.template row<TABLE>(i_row);
        for (uint16_t s = 0; s < ns; s++) {
            rtype U[N_CONSERVATIVE];
            entry_conservatives(stencil.cell(s), stencil.face(s), U);
            const rtype g = Kokkos::fabs(U[0] - rho0) / rho0;
            const rtype delta = g - g_mean;
            g_mean += delta / (s + 1);
            g_m2 = fmadd(delta, g - g_mean, g_m2);
            if constexpr (PRIM) {
                const rtype M0 = cell_molar_mass(i_cell);
                const rtype m = Kokkos::fabs(cell_molar_mass(stencil.cell(s)) - M0) / M0;
                const rtype delta_m = m - m_mean;
                m_mean += delta_m / (s + 1);
                m_m2 = fmadd(delta_m, m - m_mean, m_m2);
            }
            for (uint8_t l = 0; l < NK; l++) {
                const rtype P = stencil.template pinv<NK>(s, l);
                FOR_I_CONSERVATIVE aK[l][i] = fmadd(P, U[i] - U0[i], aK[l][i]);
            }
        }
        const rtype sigma = PRIM ? Kokkos::fmax(g_m2, m_m2) / ns : g_m2 / ns;
        if (sigma >= sigma_threshold) return;
        const rtype h = scale(i_row);
        const uint32_t row = cell_shape(i_row);
        const uint32_t m = static_cast<uint32_t>(point_offsets.extent(0) / point_cells.extent(0));
        for (uint32_t p = k * m; p < (k + 1) * m; p++) {
            rtype xi[N_DIM];
            FOR_I_DIM xi[i] = point_offsets(p, i) / h;
            rtype psi[NK];
            teno::monomials(DEG, xi, psi);
            rtype U[N_CONSERVATIVE];
            FOR_I_CONSERVATIVE U[i] = U0[i];
            for (uint8_t l = 0; l < NK; l++) {
                FOR_I_CONSERVATIVE U[i] = fmadd(aK[l][i], psi[l] - basis_mean(row, l), U[i]);
            }
            rtype Wq[N_CONSERVATIVE];
            to_primitives(U, Wq);
            if (!(Wq[0] > 0.0_r && Wq[N_DIM + 1] > 0.0_r && Kokkos::isfinite(Wq[N_DIM + 1]))) continue;
            FOR_I_CONSERVATIVE point_W(p, i) = Wq[i];
        }
    }

    // Block shapes of the cells of structured blocks: their central stencils by offsets
    teno::BlockStencils block{};

    KOKKOS_INLINE_FUNCTION
    void operator()(SmoothPass, const uint32_t idx) const {
        smooth_cell<false>(idx);
    }

    KOKKOS_INLINE_FUNCTION
    void operator()(SmoothGradientPass, const uint32_t idx) const {
        smooth_cell<true>(idx);
    }

    template <bool GRADIENTS>
    KOKKOS_INLINE_FUNCTION
    void smooth_cell(const uint32_t idx) const {
        const uint32_t i_cell = cells(idx);
        const uint32_t k = block.shape_of(i_cell);
        if (k == teno::BlockStencils::NONE) {
            smooth<GRADIENTS>(i_cell, stencil_large.template row<TABLE>(i_cell));
        } else {
            smooth<GRADIENTS>(i_cell, stencil_large.template block_row<TABLE>(i_cell, block.offsets_of(k)));
        }
    }

    template <bool GRADIENTS, typename Row>
    KOKKOS_INLINE_FUNCTION
    void smooth(const uint32_t i_cell, const Row & stencil) const {
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
        for (uint16_t s = 0; s < ns; s++) {
            rtype U[N_CONSERVATIVE];
            entry_conservatives(stencil.cell(s), stencil.face(s), U);
            const rtype g = Kokkos::fabs(U[0] - W0[0]) / W0[0];
            const rtype delta = g - g_mean;
            g_mean += delta / (s + 1);
            g_m2 = fmadd(delta, g - g_mean, g_m2);
            if constexpr (PRIM) {
                const rtype M0 = cell_molar_mass(i_cell);
                const rtype m = Kokkos::fabs(cell_molar_mass(stencil.cell(s)) - M0) / M0;
                const rtype delta_m = m - m_mean;
                m_mean += delta_m / (s + 1);
                m_m2 = fmadd(delta_m, m - m_mean, m_m2);
            }
            for (uint8_t l = 0; l < NK; l++) {
                const rtype P = stencil.template pinv<NK>(s, l);
                FOR_I_CONSERVATIVE aK[l][i] = fmadd(P, U[i] - U0[i], aK[l][i]);
            }
        }
        const rtype sigma = PRIM ? Kokkos::fmax(g_m2, m_m2) / ns : g_m2 / ns;
        sigma_out(i_cell) = sigma;
        if (sigma >= sigma_threshold) {
            if constexpr (GRADIENTS) mark_point_gradients(i_cell);
            const uint32_t j = Kokkos::atomic_fetch_add(&n_troubled(), 1u);
            troubled_cells(j) = i_cell;
            if (j < coeffs.extent(0)) {
                for (uint8_t l = 0; l < NK; l++) {
                    FOR_I_CONSERVATIVE coeffs(j, l, i) = aK[l][i];
                }
            }
            return;
        }
        if (gradients.extent(0)) {
            rtype dU[N_CONSERVATIVE][N_DIM];
            const rtype inv_h = 1.0_r / scale(i_cell);
            FOR_I_CONSERVATIVE for (uint8_t d = 0; d < N_DIM; d++) dU[i][d] = aK[d][i] * inv_h;
            store_gradients(i_cell, dU);
        }

        const uint8_t n_quad = this->n_quad();
        const uint32_t f_begin = offsets_faces_of_cell(i_cell);
        const uint8_t n_faces = static_cast<uint8_t>(offsets_faces_of_cell(i_cell + 1) - f_begin);
        rtype factor[DEG + 1];
        if constexpr (GRADIENTS) unscaled_factors(i_cell, factor);
        bool admissible = true;
        for (uint8_t k = 0; k < n_faces; k++) {
            const uint32_t f = faces_of_cell(f_begin + k);
            const uint8_t side = (cells_of_face(f, 0) == static_cast<int32_t>(i_cell)) ? 0 : 1;
            const PlanarFace nodes = planar_nodes(f);
            for (uint8_t q = 0; q < n_quad; q++) {
                rtype p[N_DIM], psi[NK];
                if (!face_point(f, q, i_cell, nodes, p)) continue;
                point_basis(i_cell, p, psi);
                rtype U_f[N_CONSERVATIVE];
                FOR_I_CONSERVATIVE U_f[i] = U0[i];
                for (uint8_t l = 0; l < NK; l++) {
                    FOR_I_CONSERVATIVE U_f[i] = fmadd(aK[l][i], psi[l], U_f[i]);
                }
                rtype Wq[N_CONSERVATIVE];
                to_primitives(U_f, Wq);
                admissible = admissible && (Wq[0] > 0.0_r) && (Wq[N_DIM + 1] > 0.0_r) && Kokkos::isfinite(Wq[N_DIM + 1]);
                FOR_I_CONSERVATIVE face_solution(f, q, side, i) = Wq[i];
                if constexpr (GRADIENTS) {
                    store_point_gradient(i_cell, f, q, side, p, Wq, [&](const uint32_t l, const uint8_t i) { return aK[l][i]; },
                                         factor);
                }
            }
        }
        if (!admissible) {
            // The viscous flux would pair the polynomial's gradients with the cell mean
            if constexpr (GRADIENTS) mark_point_gradients(i_cell);
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
    // The arithmetic is SmoothPass's, operation for operation (see fmadd)
    struct SmoothTeamPass {};
    // SmoothTeamPass that also writes point_gradients, a kernel of its own that leaves SmoothTeamPass's registers alone
    struct SmoothTeamGradientPass {};
    static constexpr uint32_t TEAM_CELLS = 32;
    // Measured on A100s: faster than SmoothPass only when its coefficients spill
    static constexpr bool USE_TEAM = N_DIM == 3 && DEG >= 4;
    static constexpr uint8_t PARTS = N_CONSERVATIVE;  // threads per cell
    static constexpr uint32_t TEAM_SIZE = TEAM_CELLS * PARTS;
    static constexpr uint8_t CHUNK = (NK + PARTS - 1) / PARTS;
    static constexpr size_t TEAM_SCRATCH = sizeof(rtype) * NK * N_CONSERVATIVE * TEAM_CELLS + 2 * sizeof(int) * TEAM_CELLS;
    // Stencil entries per round of staged_coefficients, a multiple of PARTS. At
    // order 6 rounds of 2 * PARTS would leave room for one team per SM only
    static constexpr uint8_t STAGE = DEG <= 4 ? 2 * PARTS : PARTS;
    static constexpr size_t SMOOTH_SCRATCH =
        TEAM_SCRATCH + sizeof(rtype) * STAGE * N_CONSERVATIVE * TEAM_CELLS + sizeof(int) * TEAM_CELLS;

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
     *        (l, var)
     */
    template <uint8_t C>
    KOKKOS_INLINE_FUNCTION
    void smooth_coefficients(const uint32_t i_cell, const rtype * U0, rtype * aK_team, const uint32_t lane) const {
        constexpr uint8_t L0 = C * CHUNK;
        constexpr uint8_t L1 = (L0 + CHUNK < NK) ? L0 + CHUNK : NK;
        const uint16_t ns = stencil_large_size(i_cell);
        rtype aK[CHUNK][N_CONSERVATIVE] = {};
        const auto stencil = stencil_large.template row<TABLE>(i_cell);
        for (uint16_t s = 0; s < ns; s++) {
            rtype U[N_CONSERVATIVE];
            entry_conservatives(stencil.cell(s), stencil.face(s), U);
            for (uint8_t l = L0; l < L1; l++) {
                const rtype P = stencil.template pinv<NK>(s, l);
                FOR_I_CONSERVATIVE aK[l - L0][i] = fmadd(P, U[i] - U0[i], aK[l - L0][i]);
            }
        }
        for (uint8_t l = L0; l < L1; l++) {
            FOR_I_CONSERVATIVE aK_team[(l * N_CONSERVATIVE + i) * TEAM_CELLS + lane] = aK[l - L0][i];
        }
    }

    /**
     * @brief Chunk C of the central coefficients of the team's cells, with
     *        the stencil entries' states staged in team scratch: in rounds of STAGE
     *        entries, each of a cell's threads converts every PARTS-th entry,
     *        then every thread accumulates its chunk over the round's entries
     *        in order. Every thread of the team calls it.
     */
    template <uint8_t C, typename Member, typename Row>
    KOKKOS_INLINE_FUNCTION
    void staged_coefficients(const Member & team, const bool active, const uint32_t i_cell, const Row & stencil,
                             const uint8_t part, const rtype * U0, rtype * aK_team, rtype * U_team, int * ns_team,
                             int * smooth_team, const uint32_t lane) const {
        constexpr uint8_t L0 = C * CHUNK;
        constexpr uint8_t L1 = (L0 + CHUNK < NK) ? L0 + CHUNK : NK;
        constexpr uint32_t PER = STAGE / PARTS;
        if (part == 0) ns_team[lane] = active ? stencil_large_size(i_cell) : 0;
        team.team_barrier();
        int ns_max = 0;
        for (uint32_t k = 0; k < TEAM_CELLS; k++) ns_max = Kokkos::max(ns_max, ns_team[k]);
        const uint16_t ns = static_cast<uint16_t>(ns_team[lane]);
        rtype aK[CHUNK][N_CONSERVATIVE] = {};
        rtype g_mean = 0.0, g_m2 = 0.0;
        rtype m_mean = 0.0, m_m2 = 0.0;
        for (uint16_t s0 = 0; s0 < ns_max; s0 = static_cast<uint16_t>(s0 + STAGE)) {
            // Every load of the thread's entries first, so that they are in flight together
            int32_t c[PER], f[PER];
            rtype W_e[PER][N_CONSERVATIVE];
            for (uint32_t j = 0; j < PER; j++) {
                const uint32_t s = s0 + part + j * PARTS;
                c[j] = s < ns ? stencil.cell(s) : -1;
                f[j] = s < ns ? stencil.face(s) : -1;
            }
            for (uint32_t j = 0; j < PER; j++) {
                if (c[j] >= 0) view_entries::load_row<N_CONSERVATIVE>(W, static_cast<uint32_t>(c[j]), W_e[j]);
            }
            for (uint32_t j = 0; j < PER; j++) {
                if (c[j] < 0) continue;
                rtype U[N_CONSERVATIVE];
                entry_conservatives_loaded(c[j], f[j], W_e[j], U);
                const uint32_t e = part + j * PARTS;
                FOR_I_CONSERVATIVE U_team[(e * N_CONSERVATIVE + i) * TEAM_CELLS + lane] = U[i];
            }
            team.team_barrier();
            const uint16_t s1 = Kokkos::min<uint16_t>(ns, static_cast<uint16_t>(s0 + STAGE));
            for (uint16_t s = s0; s < s1; s++) {
                rtype U[N_CONSERVATIVE];
                FOR_I_CONSERVATIVE U[i] = U_team[((s - s0) * N_CONSERVATIVE + i) * TEAM_CELLS + lane];
                if constexpr (C == 0) {
                    const rtype W0_rho = W(i_cell, 0);
                    const rtype g = Kokkos::fabs(U[0] - W0_rho) / W0_rho;
                    const rtype delta = g - g_mean;
                    g_mean += delta / (s + 1);
                    g_m2 = fmadd(delta, g - g_mean, g_m2);
                    if constexpr (PRIM) {
                        const rtype M0 = cell_molar_mass(i_cell);
                        const rtype m = Kokkos::fabs(cell_molar_mass(stencil.cell(s)) - M0) / M0;
                        const rtype delta_m = m - m_mean;
                        m_mean += delta_m / (s + 1);
                        m_m2 = fmadd(delta_m, m - m_mean, m_m2);
                    }
                }
                for (uint8_t l = L0; l < L1; l++) {
                    const rtype P = stencil.template pinv<NK>(s, l);
                    FOR_I_CONSERVATIVE aK[l - L0][i] = fmadd(P, U[i] - U0[i], aK[l - L0][i]);
                }
            }
            team.team_barrier();
        }
        if (!active) return;
        for (uint8_t l = L0; l < L1; l++) {
            FOR_I_CONSERVATIVE aK_team[(l * N_CONSERVATIVE + i) * TEAM_CELLS + lane] = aK[l - L0][i];
        }
        if constexpr (C == 0) {
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
        smooth_team<false>(team);
    }

    template <typename Member>
    KOKKOS_INLINE_FUNCTION
    void operator()(SmoothTeamGradientPass, const Member & team) const {
        smooth_team<true>(team);
    }

    template <bool GRADIENTS, typename Member>
    KOKKOS_INLINE_FUNCTION
    void smooth_team(const Member & team) const {
        const uint32_t lane = static_cast<uint32_t>(team.team_rank()) % TEAM_CELLS;
        const uint8_t part = static_cast<uint8_t>(static_cast<uint32_t>(team.team_rank()) / TEAM_CELLS);
        const uint32_t idx = static_cast<uint32_t>(team.league_rank()) * TEAM_CELLS + lane;
        const uint32_t n = cells.size();
        rtype * aK = static_cast<rtype *>(team.team_scratch(0).get_shmem(sizeof(rtype) * NK * N_CONSERVATIVE * TEAM_CELLS));
        int * smooth_team = static_cast<int *>(team.team_scratch(0).get_shmem(sizeof(int) * TEAM_CELLS));
        int * admissible_team = static_cast<int *>(team.team_scratch(0).get_shmem(sizeof(int) * TEAM_CELLS));

        const bool active = idx < n;
        const uint32_t i_cell = active ? cells(idx) : 0;
        const uint32_t k = active ? block.shape_of(i_cell) : teno::BlockStencils::NONE;
        rtype U0[N_CONSERVATIVE];
        if (part == 0) {
            smooth_team[lane] = 0;
            admissible_team[lane] = 1;
        }
        team.team_barrier();
        const auto stencil =
            stencil_large.template mixed_row<TABLE>(i_cell, k == teno::BlockStencils::NONE ? nullptr : block.offsets_of(k));
        rtype * U_team = static_cast<rtype *>(team.team_scratch(0).get_shmem(sizeof(rtype) * STAGE * N_CONSERVATIVE * TEAM_CELLS));
        int * ns_team = static_cast<int *>(team.team_scratch(0).get_shmem(sizeof(int) * TEAM_CELLS));
        if (active) conservatives(i_cell, U0);
        with_chunk(part, [&](auto C) {
            staged_coefficients<decltype(C)::value>(team, active, i_cell, stencil, part, U0, aK, U_team, ns_team,
                                                    smooth_team, lane);
        });
        team.team_barrier();
        const int slot = smooth_team[lane];
        const bool smooth = slot < 0;
        if constexpr (GRADIENTS) {
            if (active && !smooth && part == 0) mark_point_gradients(i_cell);
        }
        if (active && !smooth && slot < static_cast<int>(coeffs.extent(0))) {
            with_chunk(part, [&](auto C) {
                constexpr uint8_t L0 = decltype(C)::value * CHUNK;
                constexpr uint8_t L1 = (L0 + CHUNK < NK) ? L0 + CHUNK : NK;
                for (uint8_t l = L0; l < L1; l++) {
                    FOR_I_CONSERVATIVE coeffs(slot, l, i) = aK[(l * N_CONSERVATIVE + i) * TEAM_CELLS + lane];
                }
            });
        }

        if (active && smooth && part == 0 && gradients.extent(0)) {
            rtype dU[N_CONSERVATIVE][N_DIM];
            const rtype inv_h = 1.0_r / scale(i_cell);
            FOR_I_CONSERVATIVE for (uint8_t d = 0; d < N_DIM; d++) {
                dU[i][d] = aK[(d * N_CONSERVATIVE + i) * TEAM_CELLS + lane] * inv_h;
            }
            store_gradients(i_cell, dU);
        }

        // The cell's face points, shared out among its threads in consecutive
        // runs, so that each thread gathers the nodes of only the faces it spans
        const uint8_t n_quad = this->n_quad();
        const uint32_t f_begin = smooth ? offsets_faces_of_cell(i_cell) : 0;
        const uint32_t n_points = smooth ? (offsets_faces_of_cell(i_cell + 1) - f_begin) * n_quad : 0;
        const uint32_t run = (n_points + PARTS - 1) / PARTS;
        const uint32_t p_end = Kokkos::min(n_points, (part + 1u) * run);
        bool admissible = true;
        uint32_t face = 0;
        PlanarFace nodes;
        for (uint32_t p = part * run; p < p_end; p++) {
            const uint8_t q = p % n_quad;
            if (q == 0 || p == part * run) {
                face = faces_of_cell(f_begin + p / n_quad);
                nodes = planar_nodes(face);
            }
            const uint32_t f = face;
            rtype psi[NK];
            if (!face_point_basis(f, q, i_cell, nodes, psi)) continue;
            rtype U_f[N_CONSERVATIVE];
            FOR_I_CONSERVATIVE U_f[i] = U0[i];
            for (uint8_t l = 0; l < NK; l++) {
                FOR_I_CONSERVATIVE U_f[i] = fmadd(aK[(l * N_CONSERVATIVE + i) * TEAM_CELLS + lane], psi[l], U_f[i]);
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
            if constexpr (GRADIENTS) {
                if (part == 0) mark_point_gradients(i_cell);
            }
            rtype W0[N_CONSERVATIVE];
            FOR_I_CONSERVATIVE W0[i] = W(i_cell, i);
            for (uint32_t p = part; p < n_points; p += PARTS) {
                const uint32_t f = faces_of_cell(f_begin + p / n_quad);
                const uint8_t side = side_of(f, i_cell);
                FOR_I_CONSERVATIVE face_solution(f, p % n_quad, side, i) = W0[i];
            }
        } else if (GRADIENTS && n_points > 0) {
            rtype factor[DEG + 1];
            unscaled_factors(i_cell, factor);
            for (uint32_t p = part; p < n_points; p += PARTS) {
                const uint32_t f = faces_of_cell(f_begin + p / n_quad);
                const uint8_t q = p % n_quad;
                rtype x[N_DIM];
                if (!face_point(f, q, i_cell, planar_nodes(f), x)) continue;
                const uint8_t side = side_of(f, i_cell);
                rtype Wq[N_CONSERVATIVE];
                FOR_I_CONSERVATIVE Wq[i] = face_solution(f, q, side, i);
                store_point_gradient(
                    i_cell, f, q, side, x, Wq,
                    [&](const uint32_t l, const uint8_t i) { return aK[(l * N_CONSERVATIVE + i) * TEAM_CELLS + lane]; }, factor);
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
            rtype W_nb[N_CONSERVATIVE], W_avg[N_CONSERVATIVE];
            FOR_I_CONSERVATIVE W_nb[i] = (nb >= 0) ? W(nb, i) : W0[i];
            rtype n[N_DIM], n_vec[N_DIM];
            FOR_I_DIM n_vec[i] = face_normals(f, i);
            if (nb >= 0 && periodic.rotated()) {
                // The neighbor's state and the normal in i_cell's frame
                const bool side_1 = nb == c0;
                periodic.rotate_state(face_shift(f), W_nb, side_1);
                if (side_1) periodic.unrotate(face_shift(f), n_vec);
            }
            FOR_I_CONSERVATIVE W_avg[i] = (nb >= 0) ? 0.5_r * (W0[i] + W_nb[i]) : W0[i];
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
        for (uint8_t m = 0; m < N_CONSERVATIVE; m++) c = fmadd(L[var][m], a[m], c);
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
                row = fmadd(S((l <= m) ? teno::upper_index(l, m, NK) : teno::upper_index(m, l, NK)), c[m], row);
            }
            si = fmadd(c[l], row, si);
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
                FOR_I_CONSERVATIVE aS[l][i] = fmadd(P, U[i] - U0[i], aS[l][i]);
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
        const uint8_t n_faces = static_cast<uint8_t>(offsets_faces_of_cell(i_cell + 1) - f_begin);
        if (k >= n_faces) return;
        constexpr rtype eps = 1.0e-12_r;
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
        rtype si_K = smoothness<NK>(cK, S) + eps;
        rtype si_small[teno::MAX_FACES] = {};
        for (uint8_t s = 0; s < n_faces; s++) {
            if (valid[s]) si_small[s] = smoothness<teno::NK_SMALL>(cS[s], S) + eps;
        }
        // The two shear waves of a 3D face select together, on the sum of their
        // indicators, which does not depend on the choice of tangents
        if (N_DIM == 3 && characteristic && var >= 3) {
            const uint8_t other = static_cast<uint8_t>(7 - var);
            rtype c[NK];
            for (uint8_t l = 0; l < NK; l++) {
                rtype a[N_CONSERVATIVE];
                FOR_I_CONSERVATIVE a[i] = coeffs(j - round_begin, l, i);
                c[l] = project(L, other, a);
            }
            si_K += smoothness<NK>(c, S);
            for (uint8_t s = 0; s < n_faces; s++) {
                if (!valid[s]) continue;
                for (uint8_t l = 0; l < teno::NK_SMALL; l++) {
                    rtype a[N_CONSERVATIVE];
                    FOR_I_CONSERVATIVE a[i] = small_coeffs(j - round_begin, s, l, i);
                    c[l] = project(L, other, a);
                }
                si_small[s] += smoothness<teno::NK_SMALL>(c, S);
            }
        }
        rtype si_min = si_K;
        for (uint8_t s = 0; s < n_faces; s++) {
            if (valid[s]) si_min = Kokkos::fmin(si_min, si_small[s]);
        }
        const rtype gK = Kokkos::pow(si_min / si_K, 6.0_r);
        rtype g_small[teno::MAX_FACES] = {};
        rtype sum_small = 0.0;
        for (uint8_t s = 0; s < n_faces; s++) {
            if (!valid[s]) continue;
            g_small[s] = Kokkos::pow(si_min / si_small[s], 6.0_r);
            sum_small += g_small[s];
        }
        bool use_large = (sum_small == 0.0_r) || (gK / (gK + sum_small) >= cutoff);
        if constexpr (DEG == 3) {
            // A face whose own sector is the smoothest candidate while another is cut looks away from a
            // discontinuity in or next to the cell; the cubic passes the cutoff there but overshoots (#977).
            // Mirror-image sectors tie to round-off, and both of their faces must decide alike
            constexpr rtype tie = (sizeof(rtype) == sizeof(double)) ? 1.0e-8_r : 1.0e-4_r;
            if (use_large && valid[k] && si_small[k] <= si_min * (1.0_r + tie)) {
                for (uint8_t s = 0; s < n_faces; s++) {
                    if (valid[s] && g_small[s] / sum_small < cutoff) use_large = false;
                }
            }
        }
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
                for (uint8_t s = 0; s < n_faces; s++) mask = static_cast<uint8_t>(mask | ((w_small[s] > 0.0_r) ? (1u << s) : 0u));
                selection(f, side) = use_large ? TENO::SELECT_LARGE : mask;
            }
        }
        const uint8_t n_quad = this->n_quad();
        const PlanarFace nodes = planar_nodes(f);
        for (uint8_t q = 0; q < n_quad; q++) {
            rtype psi[NK];
            if (!face_point_basis(f, q, i_cell, nodes, psi)) continue;
            rtype v_char = c0_char;
            if (use_large) {
                for (uint8_t l = 0; l < NK; l++) v_char = fmadd(cK[l], psi[l], v_char);
            } else {
                for (uint8_t s = 0; s < n_faces; s++) {
                    if (!valid[s] || w_small[s] == 0.0_r) continue;
                    rtype p_s = 0.0;
                    for (uint8_t l = 0; l < teno::NK_SMALL; l++) p_s = fmadd(cS[s][l], psi[l], p_s);
                    v_char = fmadd(w_small[s], p_s, v_char);
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
                for (uint8_t m = 0; m < N_CONSERVATIVE; m++) U_f = fmadd(R[i][m], v_char[m], U_f);
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

    // Central coefficients of the queued cells of every round, when there can be
    // more than one (TENO::launch_reconstruction): TroubledCentralPass with
    // SmoothPass's arithmetic, TroubledCentralTeamPass with SmoothTeamPass's
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
                    FOR_I_CONSERVATIVE aK[l][i] = fmadd(P, U[i] - U0[i], aK[l][i]);
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
        const uint32_t lane = static_cast<uint32_t>(team.team_rank()) % TEAM_CELLS;
        const uint8_t part = static_cast<uint8_t>(static_cast<uint32_t>(team.team_rank()) / TEAM_CELLS);
        rtype * aK = static_cast<rtype *>(team.team_scratch(0).get_shmem(sizeof(rtype) * NK * N_CONSERVATIVE * TEAM_CELLS));
        const uint32_t n = round_items(1);
        for (uint32_t base = static_cast<uint32_t>(team.league_rank()) * TEAM_CELLS; base < n;
             base += static_cast<uint32_t>(team.league_size()) * TEAM_CELLS) {
            const uint32_t idx = base + lane;
            if (idx < n) {
                const uint32_t i_cell = troubled_cells(round_begin + idx);
                rtype U0[N_CONSERVATIVE];
                conservatives(i_cell, U0);
                with_chunk(part, [&](auto C) {
                    smooth_coefficients<decltype(C)::value>(i_cell, U0, aK, lane);
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
    Kokkos::View<uint32_t *> curved_rule_start{};      // (cell + 1): curved cells' points in curved_rule, else empty
    Kokkos::View<rtype *[3]> curved_rule{};            // (q, [x, y, w]) of the curved cells' own rules
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
                    FOR_I_CONSERVATIVE aK[l][i] = fmadd(P, U[i] - U0[i], aK[l][i]);
                }
            }
            const rtype h = scale(i_cell);
            const uint32_t row = cell_shape(i_cell);
            const rtype xc = cell_coords(i_cell, 0), yc = cell_coords(i_cell, 1);
            const bool viscous = mu.extent(0) > 0;
            // p_eff of the polynomial at (x, y); false where it is not admissible
            auto p_eff_at = [&](const rtype x, const rtype y, rtype & p_eff) {
                const rtype xi = (x - xc) / h, eta = (y - yc) / h;
                rtype psi[NK];
                teno::monomials(DEG, xi, eta, psi);
                rtype U[N_CONSERVATIVE];
                FOR_I_CONSERVATIVE U[i] = U0[i];
                for (uint8_t l = 0; l < NK; l++) {
                    FOR_I_CONSERVATIVE U[i] = fmadd(aK[l][i], psi[l] - basis_mean(row, l), U[i]);
                }
                rtype Wq[N_CONSERVATIVE];
                to_primitives(U, Wq);
                if (!(Wq[0] > 0.0_r && Wq[E] > 0.0_r)) return false;
                p_eff = Wq[E];
                if (viscous) {
                    rtype d_xi[NK], d_eta[NK];
                    teno::monomial_gradients(DEG, xi, eta, d_xi, d_eta);
                    rtype dU[N_CONSERVATIVE][N_DIM] = {};
                    for (uint8_t l = 0; l < NK; l++) {
                        FOR_I_CONSERVATIVE {
                            dU[i][0] = fmadd(aK[l][i], d_xi[l], dU[i][0]);
                            dU[i][1] = fmadd(aK[l][i], d_eta[l], dU[i][1]);
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
                return true;
            };
            rtype sum = 0.0_r;
            if (curved_rule_start.extent(0) > 0 && curved_rule_start(i_cell + 1) > curved_rule_start(i_cell)) {
                for (uint32_t q = curved_rule_start(i_cell); q < curved_rule_start(i_cell + 1); q++) {
                    rtype p_eff;
                    if (!p_eff_at(curved_rule(q, 0), curved_rule(q, 1), p_eff)) return;
                    sum += curved_rule(q, 2) * p_eff;
                }
                source(i_cell) = sum;
                return;
            }
            const uint32_t begin = offsets_nodes_of_cell(i_cell);
            const uint32_t n_nodes = offsets_nodes_of_cell(i_cell + 1) - begin;
            const uint32_t n0 = nodes_of_cell(begin);
            const rtype x0 = node_coords(n0, 0), y0 = node_coords(n0, 1);
            for (uint32_t k = 1; k + 1 < n_nodes; k++) {
                const uint32_t na = nodes_of_cell(begin + k), nb = nodes_of_cell(begin + k + 1);
                const rtype ax = node_coords(na, 0) - x0, ay = node_coords(na, 1) - y0;
                const rtype bx = node_coords(nb, 0) - x0, by = node_coords(nb, 1) - y0;
                const rtype det = Kokkos::fabs(ax * by - ay * bx);
                for (uint32_t q = 0; q < cell_rule.extent(0); q++) {
                    const rtype x = x0 + cell_rule(q, 0) * ax + cell_rule(q, 1) * bx;
                    const rtype y = y0 + cell_rule(q, 0) * ay + cell_rule(q, 1) * by;
                    rtype p_eff;
                    if (!p_eff_at(x, y, p_eff)) return;
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
            troubled_sector(round_begin + idx / teno::MAX_FACES, static_cast<uint8_t>(idx % teno::MAX_FACES));
        }
    }

    KOKKOS_INLINE_FUNCTION
    void operator()(TroubledSelectPass, const uint32_t t) const {
        constexpr uint32_t per_cell = teno::MAX_FACES * N_CONSERVATIVE;
        const uint32_t n = round_items(per_cell);
        for (uint32_t idx = t; idx < n; idx += stride) {
            const uint32_t r = idx % per_cell;
            const uint32_t j = round_begin + idx / per_cell;
            const uint32_t row = cell_shape(troubled_cells(j));
            troubled_select(j, static_cast<uint8_t>(r / N_CONSERVATIVE), static_cast<uint8_t>(r % N_CONSERVATIVE),
                            [&](const uint32_t e) { return si_matrix(row, e); });
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
        const uint32_t league_rank = static_cast<uint32_t>(team.league_rank());
        const uint32_t league_size = static_cast<uint32_t>(team.league_size());
        for (uint32_t j = round_begin + league_rank; j < end; j += league_size) {
            const uint32_t row = cell_shape(troubled_cells(j));
            Kokkos::parallel_for(Kokkos::TeamThreadRange(team, N_MOMENTS), [&](const uint32_t k) { mom[k] = moments(row, k); });
            team.team_barrier();
            Kokkos::parallel_for(Kokkos::TeamThreadRange(team, (N_SI + SI_GROUP - 1) / SI_GROUP), [&](const uint32_t g) {
                si_terms.entries<SI_GROUP>(g * SI_GROUP, N_SI, mom, S);
            });
            team.team_barrier();
            Kokkos::parallel_for(Kokkos::TeamThreadRange(team, uint32_t(teno::MAX_FACES) * N_CONSERVATIVE),
                                 [&](const uint32_t r) {
                troubled_select(j, static_cast<uint8_t>(r / N_CONSERVATIVE), static_cast<uint8_t>(r % N_CONSERVATIVE),
                                [&](const uint32_t e) { return S[e]; });
            });
            team.team_barrier();
        }
    }

    KOKKOS_INLINE_FUNCTION
    void operator()(TroubledProjectPass, const uint32_t t) const {
        const uint32_t n = round_items(teno::MAX_FACES);
        for (uint32_t idx = t; idx < n; idx += stride) {
            troubled_project(round_begin + idx / teno::MAX_FACES, static_cast<uint8_t>(idx % teno::MAX_FACES));
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
        const uint8_t n_faces = static_cast<uint8_t>(offsets_faces_of_cell(i_cell + 1) - f_begin);

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
            // point stay within the range of the cell and its face neighbors,
            // widened by teno::BOUND_SLACK of the cell's value
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
            for (uint8_t v = 0; v < 2; v++) {
                const rtype slack = teno::BOUND_SLACK * Kokkos::fabs(v == 0 ? W0[0] : W0[P]);
                lo[v] -= slack;
                hi[v] += slack;
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
                                 const IndexSet & cells, bool troubled_pass) {
    using Functor = TENOFunctor<DEG, PRIM, TABLE>;
    Functor functor{sigma_threshold, sigma_upper, C_T, characteristic, bound_preserving, boundaries.gamma,
                    mesh->offsets_faces_of_cell, mesh->faces_of_cell, mesh->cells_of_face,
                    face_nodes, node_coords,
                    mesh->cell_coords, mesh->face_coords, mesh->face_normals, mesh->periodic_maps(), mesh->face_shift,
                    quadrature_face.points, face_quad_points, face_quad_weights, face_rule, boundaries,
                    scale, cell_shape, basis_mean, stencil_large_size, stencil_large, stencil_small_size, stencil_small,
                    si_matrix, moments, si_terms, troubled, troubled_coeffs, troubled_small_coeffs, troubled_cells,
                    n_troubled,
                    solution, face_solution, cells, smooth_gradients, cell_gamma, cell_molar_mass, selection,
                    entry_rotations, boundary_faces};
    functor.point_gradients = point_gradients;
    using Dynamic = Kokkos::Schedule<Kokkos::Dynamic>;
    using Space = Kokkos::DefaultExecutionSpace;
    // A cell's place in the queue follows the order of the atomic increments.
    // When the queue can take more than one round, every round recomputes the
    // central coefficients of its cells, so that none depend on whether the
    // smooth pass kept them: two kernels need not round the same arithmetic alike
    const uint32_t capacity = static_cast<uint32_t>(troubled_coeffs.extent(0));
    const bool rounds = capacity < n_rows();
    if (!troubled_pass) {
        functor.block = block_stencils;
        if (rounds) functor.coeffs = {};
        const uint32_t n = cells.size();
        constexpr bool host = Kokkos::SpaceAccessibility<Kokkos::HostSpace, Space::memory_space>::accessible;
        auto launch_team = [&](auto tag) {
            if (n == 0) return;
            using Policy = Kokkos::TeamPolicy<Space, decltype(tag), Kokkos::LaunchBounds<Functor::TEAM_SIZE, 2>>;
            const Policy policy = Policy(exec, static_cast<int>((n + Functor::TEAM_CELLS - 1) / Functor::TEAM_CELLS), Functor::TEAM_SIZE)
                                      .set_scratch_size(0, Kokkos::PerTeam(Functor::SMOOTH_SCRATCH));
            gated::parallel_for("teno_smooth", policy, functor);
        };
        if (point_gradients.on()) {
            // In 3D the team's threads share out the face points' gradients, whatever DEG
            if constexpr (!host && N_DIM == 3) {
                launch_team(typename Functor::SmoothTeamGradientPass{});
            } else {
                gated::parallel_for(
                    "teno_smooth",
                    Kokkos::RangePolicy<Space, typename Functor::SmoothGradientPass, Dynamic, HeavyBounds>(exec, 0, n),
                    functor);
            }
            return;
        }
        if constexpr (!Functor::USE_TEAM || host) {
            gated::parallel_for("teno_smooth",
                                 Kokkos::RangePolicy<Space, typename Functor::SmoothPass, Dynamic, HeavyBounds>(exec, 0, n),
                                 functor);
        } else {
            launch_team(typename Functor::SmoothTeamPass{});
        }
        return;
    }
    // Each pass loops over its round of the queue with as many threads as it
    // could need, up to TROUBLED_THREADS, so the queue length stays on the
    // device; rounds past the queue's end find no work
    const uint32_t n = n_rows();
    if (n > 0 && capacity > 0) {
        auto launch = [&](const char * label, auto tag, uint32_t per_cell) {
            functor.stride = static_cast<uint32_t>(std::min<uint64_t>(TROUBLED_THREADS, uint64_t(capacity) * per_cell));
            gated::parallel_for(label, Kokkos::RangePolicy<Space, decltype(tag), Dynamic, HeavyBounds>(exec, 0, functor.stride),
                                 functor);
        };
        constexpr bool host = Kokkos::SpaceAccessibility<Kokkos::HostSpace, Space::memory_space>::accessible;
        for (uint32_t begin = 0; begin < n; begin += capacity) {
            functor.round_begin = begin;
            if (rounds) {
                if constexpr (Functor::USE_TEAM && !host) {
                    using Policy = Kokkos::TeamPolicy<Space, typename Functor::TroubledCentralTeamPass,
                                                      Kokkos::LaunchBounds<Functor::TEAM_SIZE, 2>>;
                    const uint32_t league = std::min<uint32_t>((capacity + Functor::TEAM_CELLS - 1) / Functor::TEAM_CELLS,
                                                               TROUBLED_THREADS / Functor::TEAM_SIZE);
                    gated::parallel_for("teno_troubled_central",
                                         Policy(exec, static_cast<int>(league), Functor::TEAM_SIZE)
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
                gated::parallel_for("teno_troubled_select",
                                     Policy(exec, static_cast<int>(league), static_cast<int>(team_size))
                                         .set_scratch_size(0, Kokkos::PerTeam(Functor::SELECT_SCRATCH)),
                                     functor);
            }
            launch("teno_troubled_project", typename Functor::TroubledProjectPass{}, teno::MAX_FACES);
            launch("teno_troubled_finish", typename Functor::TroubledFinishPass{}, 1);
        }
    }
    gated::fill(exec, n_troubled, 0u);
    if constexpr (!PRIM) {
        if (point_gradients.on() && wall_fits.n_fits() > 0) {
            functor.wall_fits = wall_fits;
            gated::parallel_for("teno_wall_fits",
                                Kokkos::RangePolicy<Space, typename Functor::WallFitPass, HeavyBounds>(exec, 0, wall_fits.n_fits()),
                                functor);
        }
    }
}

template <uint8_t DEG, bool PRIM, typename TABLE>
void TENO::launch_gradients(Kokkos::View<rtype *[N_CONSERVATIVE]> solution,
                            Kokkos::View<rtype *[N_CONSERVATIVE][N_DIM]> gradients, const uint32_t n_cells) {
    using Functor = TENOFunctor<DEG, PRIM, TABLE>;
    Functor functor{sigma_threshold, sigma_upper, C_T, characteristic, bound_preserving, boundaries.gamma,
                    mesh->offsets_faces_of_cell, mesh->faces_of_cell, mesh->cells_of_face,
                    face_nodes, node_coords,
                    mesh->cell_coords, mesh->face_coords, mesh->face_normals, mesh->periodic_maps(), mesh->face_shift,
                    quadrature_face.points, face_quad_points, face_quad_weights, face_rule, boundaries,
                    scale, cell_shape, basis_mean, stencil_large_size, stencil_large, stencil_small_size, stencil_small,
                    si_matrix, moments, si_terms, troubled, troubled_coeffs, troubled_small_coeffs, troubled_cells,
                    n_troubled,
                    solution, {}, {}, gradients, cell_gamma, cell_molar_mass, selection, entry_rotations, boundary_faces};
    gated::parallel_for("teno_gradients", HeavyRange<typename Functor::GradientPass>(0, n_cells), functor);
}

template <uint8_t DEG, bool PRIM, typename TABLE>
void TENO::launch_points(Kokkos::View<rtype *[N_CONSERVATIVE]> solution, Kokkos::View<uint32_t *> cells,
                         Kokkos::View<rtype *[N_DIM]> offsets, Kokkos::View<rtype *[N_CONSERVATIVE]> out) {
    using Functor = TENOFunctor<DEG, PRIM, TABLE>;
    Functor functor{sigma_threshold, sigma_upper, C_T, characteristic, bound_preserving, boundaries.gamma,
                    mesh->offsets_faces_of_cell, mesh->faces_of_cell, mesh->cells_of_face,
                    face_nodes, node_coords,
                    mesh->cell_coords, mesh->face_coords, mesh->face_normals, mesh->periodic_maps(), mesh->face_shift,
                    quadrature_face.points, face_quad_points, face_quad_weights, face_rule, boundaries,
                    scale, cell_shape, basis_mean, stencil_large_size, stencil_large, stencil_small_size, stencil_small,
                    si_matrix, moments, si_terms, troubled, troubled_coeffs, troubled_small_coeffs, troubled_cells,
                    n_troubled,
                    solution, {}, {}, {}, cell_gamma, cell_molar_mass, selection, entry_rotations, boundary_faces};
    functor.point_cells = cells;
    functor.n_direct_rows = n_direct_rows;
    functor.extra_cells = extra_cells_device;
    functor.point_offsets = offsets;
    functor.point_W = out;
    gated::parallel_for("teno_point_values", Kokkos::RangePolicy<typename Functor::PointPass>(0, cells.extent(0)),
                         functor);
}

template <uint8_t DEG, bool PRIM, typename TABLE>
void TENO::launch_source(Kokkos::View<rtype *[N_CONSERVATIVE]> solution, Kokkos::View<rtype *, Kokkos::LayoutStride> mu,
                         Kokkos::View<rtype *> source, const uint32_t n_cells) {
    using Functor = TENOFunctor<DEG, PRIM, TABLE>;
    Functor functor{sigma_threshold, sigma_upper, C_T, characteristic, bound_preserving, boundaries.gamma,
                    mesh->offsets_faces_of_cell, mesh->faces_of_cell, mesh->cells_of_face,
                    face_nodes, node_coords,
                    mesh->cell_coords, mesh->face_coords, mesh->face_normals, mesh->periodic_maps(), mesh->face_shift,
                    quadrature_face.points, face_quad_points, face_quad_weights, face_rule, boundaries,
                    scale, cell_shape, basis_mean, stencil_large_size, stencil_large, stencil_small_size, stencil_small,
                    si_matrix, moments, si_terms, troubled, troubled_coeffs, troubled_small_coeffs, troubled_cells,
                    n_troubled,
                    solution, {}, {}, {}, cell_gamma, cell_molar_mass, selection, entry_rotations, boundary_faces};
    functor.offsets_nodes_of_cell = offsets_nodes_of_cell;
    functor.nodes_of_cell = nodes_of_cell;
    functor.cell_rule = cell_rule;
    functor.curved_rule_start = curved_rule_start;
    functor.curved_rule = curved_rule;
    functor.mu = mu;
    functor.source = source;
    gated::parallel_for("teno_axisymmetric_source", Kokkos::RangePolicy<typename Functor::SourcePass>(0, n_cells),
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
    selection = Kokkos::View<uint8_t **>("teno_selection", mesh->n_state_faces(), 2);
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

bool TENO::point_values(Kokkos::View<rtype *[N_CONSERVATIVE]> solution, Kokkos::View<uint32_t *> cells,
                        Kokkos::View<rtype *[N_DIM]> offsets, Kokkos::View<rtype *[N_CONSERVATIVE]> out) {
    if (cells.extent(0) == 0) return true;
#if N_DIM == 3
    if (single_tables) {
        switch (degree) {
            case 2: launch_points<2, false, float>(solution, cells, offsets, out); break;
            case 3: launch_points<3, false, float>(solution, cells, offsets, out); break;
            case 4: launch_points<4, false, float>(solution, cells, offsets, out); break;
            case 5: launch_points<5, false, float>(solution, cells, offsets, out); break;
            default: throw std::runtime_error("TENO: unsupported degree.");
        }
        return true;
    }
#endif
    switch (degree * 2 + primitive) {
        case 4: launch_points<2, false, rtype>(solution, cells, offsets, out); break;
        case 6: launch_points<3, false, rtype>(solution, cells, offsets, out); break;
        case 8: launch_points<4, false, rtype>(solution, cells, offsets, out); break;
        case 10: launch_points<5, false, rtype>(solution, cells, offsets, out); break;
        case 5: launch_points<2, true, rtype>(solution, cells, offsets, out); break;
        case 7: launch_points<3, true, rtype>(solution, cells, offsets, out); break;
        case 9: launch_points<4, true, rtype>(solution, cells, offsets, out); break;
        case 11: launch_points<5, true, rtype>(solution, cells, offsets, out); break;
        default: throw std::runtime_error("TENO: unsupported degree.");
    }
    return true;
}

void TENO::calc_face_values(Kokkos::View<rtype *[N_CONSERVATIVE]> solution,
                            Kokkos::View<rtype **[2][N_CONSERVATIVE]> face_solution) {
    calc_cell_face_values(Kokkos::DefaultExecutionSpace(), solution, face_solution, IndexSet{{}, 0, n_direct_rows});
    finish_cell_face_values(solution, face_solution);
}

std::vector<uint32_t> TENO::cells_independent_of_halo(uint32_t n_owned) const {
    // Stencil entries (mirrored ones too) and the face neighbors used for the
    // characteristic projection and the bounds
    const auto large_index = host_index(stencil_large);
    const auto small_index = host_index(stencil_small);
    std::vector<uint32_t> cells;
    std::vector<CellTables> chunk;
    for (uint32_t c0 = 0; c0 < n_owned; c0 += CHUNK_CELLS) {
        chunk.assign(std::min(CHUNK_CELLS, static_cast<uint32_t>(scale.extent(0)) - c0), CellTables());
        download_tables(*this, large_index, small_index, c0, chunk);
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
                                 const IndexSet & cells) {
    dispatch(exec, solution, face_solution, cells, false);
}

void TENO::finish_cell_face_values(Kokkos::View<rtype *[N_CONSERVATIVE]> solution,
                                   Kokkos::View<rtype **[2][N_CONSERVATIVE]> face_solution) {
    dispatch(Kokkos::DefaultExecutionSpace(), solution, face_solution, IndexSet(), true);
}

void TENO::discard_cell_face_values() {
    gated::fill(n_troubled, 0u);
}

void TENO::dispatch(const Kokkos::DefaultExecutionSpace & exec, Kokkos::View<rtype *[N_CONSERVATIVE]> solution,
                    Kokkos::View<rtype **[2][N_CONSERVATIVE]> face_solution, const IndexSet & cells,
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
// each 3D cell's moments in place of its smoothness-indicator matrix; version 11 ranks 2D candidates
// in the mesh-spacing metric and mirrors them only across the boundary faces beneath them (#382);
// version 12 weights the 2D smoothness-indicator matrices so that they do not depend on the axes (#436);
// version 14 has 2D tables from the device setup (#776), which differ from version 13's at round-off;
// version 15 mirrors 3D candidates only across the boundary faces beneath them (#443); version 16
// starts odd-degree stencils of pyramids from 2.5 times the monomials of the next degree (#236);
// version 21 keeps 2D mirror images no deeper than the body under their foot (#976); version 22 has
// tables for the owned cells and halo image cells only when partition faces are exchanged (#1040);
// version 24 fits curved_wall_degree in 2D stencils grown past their start to a curved-wall cell (#1202);
// version 25 stops 3D stencils grown past their start short of a curved-wall cell at the full degree, and fits
// curved_wall_degree only when no shorter stencil fits (#1202);
// version 26 has shared tables fitted once per stencil shape (#986), which differ from version 25's at round-off
constexpr char TENO_CACHE_MAGIC[16] = "MALLARD-TENO-26";
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
    // Element by element in index order, which the backend's layout does not change
    template <typename V>
    void add_view(const V & v) {
        for (size_t i = 0; i < v.extent(0); i++) {
            for (size_t j = 0; j < v.extent(1); j++) {
                for (size_t k = 0; k < v.extent(2); k++) add(v.access(i, j, k));
            }
        }
    }
};

struct CacheHeader {
    char magic[sizeof(TENO_CACHE_MAGIC)] = {};
    char producer[16] = {};
    uint64_t options_key = 0;
    uint64_t cache_key = 0;
    uint8_t halo_layers = 0;
    uint32_t n_table_rows = 0;
};

template <typename T>
void put(std::vector<char> & buf, const T * data, size_t n) {
    const char * p = reinterpret_cast<const char *>(data);
    buf.insert(buf.end(), p, p + n * sizeof(T));
}

template <typename T>
bool get(std::istream & in, T * data, size_t n) {
    in.read(reinterpret_cast<char *>(data), static_cast<std::streamsize>(n * sizeof(T)));
    return in.good();
}

void write_header(std::ostream & out, const CacheHeader & h) {
    std::vector<char> buf;
    put(buf, TENO_CACHE_MAGIC, sizeof(TENO_CACHE_MAGIC));
    put(buf, h.producer, sizeof(h.producer));
    put(buf, &h.options_key, 1);
    put(buf, &h.cache_key, 1);
    put(buf, &h.halo_layers, 1);
    put(buf, &h.n_table_rows, 1);
    out.write(buf.data(), static_cast<std::streamsize>(buf.size()));
}

bool read_header(std::istream & in, CacheHeader & h) {
    if (!get(in, h.magic, sizeof(h.magic))) return false;
    // Older formats have no producer
    if (std::string(h.magic, strnlen(h.magic, sizeof(h.magic))) != TENO_CACHE_MAGIC) return true;
    return get(in, h.producer, sizeof(h.producer)) && get(in, &h.options_key, 1) && get(in, &h.cache_key, 1) &&
           get(in, &h.halo_layers, 1) && get(in, &h.n_table_rows, 1);
}

/** @brief Round values to single precision in place. */
template <typename T>
void round_values_to_single(const Kokkos::View<T *> & values) {
    gated::parallel_for("teno_round_to_single", values.extent(0), KOKKOS_LAMBDA(const size_t k) {
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
    const uint16_t n_large = static_cast<uint16_t>(t.large_cells.size());
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
    t.si.resize(N_DIM == 2 ? static_cast<size_t>(nk * (nk + 1) / 2) : 0);
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
    const uint32_t n_table_rows = n_rows();
    troubled = Kokkos::View<rtype *>("teno_sigma", mesh->n_reconstructed());
    const uint32_t capacity = !sector_stencils ? 0u : std::min<uint32_t>(
        n_table_rows, static_cast<uint32_t>(std::ceil(double(troubled_capacity) * double(n_table_rows))));
    troubled_coeffs = memory::scratch_view<Kokkos::View<rtype ***>>(own_scratch, "teno_troubled_coeffs", capacity,
                                                                     n_dof_large, N_CONSERVATIVE);
    troubled_small_coeffs = memory::scratch_view<Kokkos::View<rtype ****>>(
        own_scratch, "teno_troubled_small_coeffs", capacity, teno::MAX_FACES, teno::NK_SMALL, N_CONSERVATIVE);
    troubled_cells = Kokkos::View<uint32_t *>("teno_troubled_cells", n_table_rows);
    n_troubled = Kokkos::View<uint32_t>("teno_n_troubled");
}

size_t TENO::scratch_bytes() const {
    using memory::ScratchArena;
    return ScratchArena::aligned(troubled_coeffs.span() * sizeof(rtype)) +
           ScratchArena::aligned(troubled_small_coeffs.span() * sizeof(rtype));
}

void TENO::place_scratch(void * storage) {
    if (troubled_coeffs.span() == 0) return;
    std::byte * const p = static_cast<std::byte *>(storage);
    const size_t small = memory::ScratchArena::aligned(troubled_coeffs.span() * sizeof(rtype));
    troubled_coeffs = Kokkos::View<rtype ***>(reinterpret_cast<rtype *>(p), troubled_coeffs.layout());
    troubled_small_coeffs = Kokkos::View<rtype ****>(reinterpret_cast<rtype *>(p + small), troubled_small_coeffs.layout());
}

std::array<double, 9> TENO::stencil_metric(const uint32_t i) const {
    auto h_face_bc = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), boundaries.face_bc);
    auto h_bcs = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), boundaries.bcs);
    return teno_setup::ranking_metric(*mesh, h_face_bc, h_bcs, i);
}

namespace {

/**
 * @brief Cells, not among the sorted rebuilt ones, to rebuild with them: a
 *        shared table or row is that of its shape's cell with the lowest
 *        global id (owner(k) of shape k), and the other cells of the shape
 *        hold it, not their own. When that cell is rebuilt, a full setup gives
 *        the shape the table of the lowest of the rest, so each rank rebuilds
 *        its lowest (collective).
 */
template <typename ShapeOf, typename Owner>
std::vector<uint32_t> owners_of_rebuilt(const size_t n_shapes, ShapeOf shape_of, Owner owner,
                                        const std::vector<uint32_t> & rebuilt, const uint64_t * gid, const uint32_t n_cells) {
    auto id = [&](const uint32_t c) { return gid != nullptr ? gid[c] : uint64_t(c); };
    std::vector<uint64_t> lost;
    for (const uint32_t c : n_shapes > 0 ? rebuilt : std::vector<uint32_t>()) {
        if (owner(shape_of(c)) == id(c)) lost.push_back(id(c));
    }
    lost = comm::allgatherv(lost);
    std::sort(lost.begin(), lost.end());
    std::unordered_map<size_t, uint32_t> lowest;  // shape: cell
    for (uint32_t c = 0; c < (lost.empty() ? 0 : n_cells); c++) {
        const size_t k = shape_of(c);
        if (std::binary_search(rebuilt.begin(), rebuilt.end(), c) || !std::binary_search(lost.begin(), lost.end(), owner(k))) {
            continue;
        }
        auto [it, added] = lowest.try_emplace(k, c);
        if (!added && id(c) < id(it->second)) it->second = c;
    }
    std::vector<uint32_t> out;
    for (const auto & kc : lowest) out.push_back(kc.second);
    return out;
}

/** @brief owners_of_rebuilt for the shared pseudo-inverses of p. */
std::vector<uint32_t> owners_of_rebuilt_tables(const teno::PackedStencils & p, const std::vector<uint32_t> & rebuilt,
                                               const uint64_t * gid, const uint32_t n_cells) {
    auto index = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), p.pinv_index);
    const size_t n_shapes = p.shape_owner.extent(0) / 2;
    auto shape_of = [&](const uint32_t c) {
        size_t lo = 0, hi = n_shapes;
        while (hi - lo > 1) {
            const size_t mid = (lo + hi) / 2;
            (p.shape_owner(2 * mid) <= index(c) ? lo : hi) = mid;
        }
        return lo;
    };
    return owners_of_rebuilt(n_shapes, shape_of, [&](const size_t k) { return p.shape_owner(2 * k + 1); }, rebuilt, gid,
                             n_cells);
}

} // namespace

void TENO::rebuild_cells(const std::vector<uint32_t> & cells) {
    std::vector<uint32_t> sorted;
    for (const uint32_t c : cells) {
        const int64_t r = row_of_cell(c);
        if (r < 0) throw std::logic_error("TENO: only reconstructed cells have tables.");
        sorted.push_back(static_cast<uint32_t>(r));
    }
    std::sort(sorted.begin(), sorted.end());
    sorted.erase(std::unique(sorted.begin(), sorted.end()), sorted.end());
    if (share_tables) {
        const uint32_t n_rec = static_cast<uint32_t>(scale.extent(0));
        std::vector<uint32_t> more = owners_of_rebuilt_tables(stencil_large, sorted, global_cell_ids(), n_rec);
        const auto small = owners_of_rebuilt_tables(stencil_small, sorted, global_cell_ids(), n_rec);
        more.insert(more.end(), small.begin(), small.end());
        auto shape = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), cell_shape);
        const auto rows = owners_of_rebuilt(
            cell_shape_owner.size(), [&](const uint32_t c) { return size_t(shape(c)); },
            [&](const size_t k) { return cell_shape_owner[k]; }, sorted, global_cell_ids(), n_rec);
        more.insert(more.end(), rows.begin(), rows.end());
        sorted.insert(sorted.end(), more.begin(), more.end());
        std::sort(sorted.begin(), sorted.end());
        sorted.erase(std::unique(sorted.begin(), sorted.end()), sorted.end());
    } else if (sorted.empty()) {
        return;
    }
    // The face quadrature follows the mesh's geometry
    if constexpr (N_DIM == 3) init_face_quadrature_3d(degree + 1);
    expand_cell_rows(*this);
    compute_stencils_and_matrices(&sorted);
    release_planar_face_points();
    if (single_tables || (cache_single && !cache_file.empty())) round_to_single();
    uint16_t largest = 0;
    auto sizes = stencil_large_size;
    gated::parallel_reduce("teno_largest", Kokkos::RangePolicy<>(0, sizes.extent(0)),
                            KOKKOS_LAMBDA(const uint32_t c, uint16_t & m) { m = sizes(c) > m ? sizes(c) : m; },
                            Kokkos::Max<uint16_t>(largest));
    largest_stencil = largest;
    cache_loaded = false;
    find_block_stencils();
}

namespace {

struct OffsetsHash {
    size_t operator()(const std::vector<int32_t> & v) const {
        uint64_t h = 1469598103934665603ULL;
        for (const int32_t x : v) h = (h ^ uint32_t(x)) * 1099511628211ULL;
        return size_t(h);
    }
};

} // namespace

void TENO::find_block_stencils() {
    block_stencils = teno::BlockStencils();
    n_block_cells = 0;
    n_block_shapes = 0;
    const uint32_t n_rec = static_cast<uint32_t>(scale.extent(0));
    const uint32_t n = std::min(n_direct_rows, structured_block_cells(mesh->structured_blocks));
    if (!block_path || n == 0) return;
    const teno::PackedStencils & p = stencil_large;
    const uint8_t sh = p.shift;
    const uint32_t n_slices = (n + (1u << sh) - 1) >> sh;
    std::vector<uint64_t> start(n_slices + 1);
    Kokkos::deep_copy(HostUnmanaged<uint64_t>(start.data(), start.size()),
                      Kokkos::subview(p.slice_start, std::make_pair(0u, n_slices + 1)));
    std::vector<int32_t> slots(start[n_slices] << sh);
    Kokkos::deep_copy(HostUnmanaged<int32_t>(slots.data(), slots.size()),
                      Kokkos::subview(p.cells, std::make_pair(size_t(0), slots.size())));
    std::vector<uint16_t> sizes(n);
    Kokkos::deep_copy(HostUnmanaged<uint16_t>(sizes.data(), n), Kokkos::subview(stencil_large_size, std::make_pair(0u, n)));
    std::vector<uint32_t> shape(n_rec, teno::BlockStencils::NONE);
    std::vector<std::vector<int32_t>> shapes;
    std::unordered_map<std::vector<int32_t>, uint32_t, OffsetsHash> index;
    std::vector<int32_t> offsets;
    for (uint32_t c = 0; c < n; c++) {
        const uint64_t first = (start[c >> sh] << sh) + (c & ((1u << sh) - 1));
        offsets.resize(sizes[c]);
        bool image = false;
        for (uint32_t s = 0; s < sizes[c] && !image; s++) {
            const int32_t e = slots[first + (uint64_t(s) << sh)];
            image = e < 0;
            offsets[s] = e - static_cast<int32_t>(c);
        }
        if (image) continue;
        const auto [it, fresh] = index.try_emplace(offsets, uint32_t(shapes.size()));
        if (fresh) shapes.push_back(offsets);
        shape[c] = it->second;
        n_block_cells++;
    }
    if (shapes.empty()) return;
    size_t stride = 0;
    for (const auto & o : shapes) stride = std::max(stride, o.size());
    std::vector<int32_t> table(shapes.size() * stride, 0);
    for (size_t k = 0; k < shapes.size(); k++) std::copy(shapes[k].begin(), shapes[k].end(), table.begin() + std::ptrdiff_t(k * stride));
    block_stencils.stride = static_cast<uint32_t>(stride);
    block_stencils.shape = Kokkos::View<uint32_t *>(Kokkos::view_alloc(Kokkos::WithoutInitializing, "teno_block_shape"), n_rec);
    Kokkos::deep_copy(block_stencils.shape, HostUnmanaged<const uint32_t>(shape.data(), shape.size()));
    block_stencils.offsets = Kokkos::View<int32_t *>(Kokkos::view_alloc(Kokkos::WithoutInitializing, "teno_block_offsets"),
                                                     table.size());
    Kokkos::deep_copy(block_stencils.offsets, HostUnmanaged<const int32_t>(table.data(), table.size()));
    n_block_shapes = static_cast<uint32_t>(shapes.size());
}

std::vector<uint32_t> TENO::cells_within_reach(const std::vector<uint32_t> & changed) const {
    const uint32_t n_rec = n_rows();
    uint8_t max_depth = 0;
    for (uint32_t c = 0; c < n_rec; c++) max_depth = std::max(max_depth, gather_depth[c]);
    constexpr uint8_t FAR = std::numeric_limits<uint8_t>::max();
    std::vector<uint8_t> distance(mesh->n_cells, FAR);
    std::vector<uint32_t> frontier, next;
    for (const uint32_t c : changed) {
        if (distance[c] == FAR) frontier.push_back(c);
        distance[c] = 0;
    }
    const VertexNeighbors & neighbors = mesh->vertex_neighbors();
    for (uint8_t d = 1; d <= max_depth && !frontier.empty(); d++) {
        next.clear();
        for (const uint32_t c : frontier) {
            for (uint32_t k = neighbors.h_offsets(c); k < neighbors.h_offsets(c + 1); k++) {
                const uint32_t nb = neighbors.h_cells(k);
                if (distance[nb] != FAR) continue;
                distance[nb] = d;
                next.push_back(nb);
            }
        }
        frontier.swap(next);
    }
    std::vector<uint32_t> out;
    for (uint32_t r = 0; r < n_rec; r++) {
        if (distance[cell_of_row(r)] <= gather_depth[r]) out.push_back(cell_of_row(r));
    }
    return out;
}

void TENO::for_each_stencil(const std::function<void(uint32_t, const CellStencils &)> & visit, const bool pinv) const {
    const uint32_t n_table_rows = static_cast<uint32_t>(scale.extent(0));
    const auto large_index = host_index(stencil_large);
    const auto small_index = host_index(stencil_small);
    std::vector<CellTables> chunk;
    CellStencils s;
    for (uint32_t c0 = 0; c0 < n_table_rows; c0 += CHUNK_CELLS) {
        chunk.assign(std::min(CHUNK_CELLS, n_table_rows - c0), CellTables());
        download_tables(*this, large_index, small_index, c0, chunk);
        for (uint32_t k = 0; k < chunk.size(); k++) {
            const CellTables & t = chunk[k];
            s.central = static_cast<uint16_t>(t.large_cells.size());
            s.sectors = t.small_size;
            s.cells = t.large_cells;
            s.cells.insert(s.cells.end(), t.small_cells.begin(), t.small_cells.end());
            s.faces = t.large_faces;
            s.faces.insert(s.faces.end(), t.small_faces.begin(), t.small_faces.end());
            s.pinv.clear();
            if (pinv) {
                s.pinv = t.large_pinv;
                s.pinv.insert(s.pinv.end(), t.small_pinv.begin(), t.small_pinv.end());
            }
            visit(cell_of_row(c0 + k), s);
        }
    }
}

namespace {

/** @brief Collapsed (Duffy) Gauss quadrature on the reference tetrahedron; weights sum to 1/6. Exact for degree 2n-3. */
struct TetrahedronRule {
    std::vector<std::array<double, 3>> x;
    std::vector<double> w;
    explicit TetrahedronRule(int n) {
        std::vector<double> g, gw;
        gauss_legendre(n, g, gw);
        for (size_t i = 0; i < g.size(); i++) {
            for (size_t j = 0; j < g.size(); j++) {
                for (size_t k = 0; k < g.size(); k++) {
                    const double u = 0.5 * (g[i] + 1.0), v = 0.5 * (g[j] + 1.0), t = 0.5 * (g[k] + 1.0);
                    x.push_back({u, v * (1.0 - u), t * (1.0 - u) * (1.0 - v)});
                    w.push_back(0.125 * gw[i] * gw[j] * gw[k] * (1.0 - u) * (1.0 - u) * (1.0 - v));
                }
            }
        }
    }
};

}  // namespace

void TENO::init_wall_fits() {
    wall_fits = {};
    fitted_walls = {};
    if (!point_gradients.on() || primitive) return;
    const Mesh & m = *mesh;
    const uint8_t nk = n_dof_large;
    const auto h_face_bc = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), boundaries.face_bc);
    const auto h_bcs = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), boundaries.bcs);
    Kokkos::View<int32_t *>::host_mirror_type h_state_index;
    if (boundaries.face_state_index.is_allocated()) {
        h_state_index = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), boundaries.face_state_index);
    }
    auto is_wall = [&](const int32_t f) {
        const int32_t b = h_face_bc(static_cast<size_t>(f));
        return b >= 0 && h_bcs(static_cast<size_t>(b)).is_wall();
    };
    const auto h_scale = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), scale);
    const auto h_shape = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), cell_shape);
    const auto h_mean = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), basis_mean);
    Kokkos::View<rtype ***>::host_mirror_type h_quad_points;
    FaceWeights<Kokkos::HostSpace> h_quad_weights;
    FaceNodes::host_mirror_type h_face_nodes;
    NodeCoords::host_mirror_type h_node_coords;
    Kokkos::View<rtype *[4]>::host_mirror_type h_face_rule;
    if constexpr (N_DIM == 3) {
        h_quad_points = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), face_quad_points);
        h_quad_weights = face_quad_weights.host();
        h_face_nodes = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), face_nodes);
        h_node_coords = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), node_coords);
        h_face_rule = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), face_rule);
    }
    const uint8_t n_quad = n_face_quadrature_points();

    // The image of x nearest to ref across the periodic translations
    std::vector<std::array<double, N_DIM>> images(1, std::array<double, N_DIM>{});
    for (const auto & t : m.periodic_translations) {
        const size_t n = images.size();
        for (const int sign : {-1, 1}) {
            for (size_t k = 0; k < n; k++) {
                std::array<double, N_DIM> v = images[k];
                FOR_I_DIM v[i] += sign * double(t[i]);
                images.push_back(v);
            }
        }
    }
    auto nearest = [&](const double * x, const double * ref, double * out) {
        double best = std::numeric_limits<double>::max();
        for (const auto & v : images) {
            double d2 = 0.0;
            FOR_I_DIM d2 += (x[i] + v[i] - ref[i]) * (x[i] + v[i] - ref[i]);
            if (d2 < best) {
                best = d2;
                FOR_I_DIM out[i] = x[i] + v[i];
            }
        }
    };
    auto cell_center = [&](const uint32_t c, double * x) { FOR_I_DIM x[i] = double(m.h_cell_coords(c, i)); };
    auto node = [&](const uint32_t n, double * x) { FOR_I_DIM x[i] = double(m.h_node_coords(n, i)); };

    // Means of the monomials of (x - x0) / h over cell j, placed nearest to x0, over its simplices (r-weighted in
    // axisymmetric runs, as TENO's cell averages)
    const int n_rule = degree / 2 + 3;
    const TriangleRule tri(n_rule);
    const TetrahedronRule tet(n_rule);
    auto cell_means = [&](const uint32_t j, const double * x0, const double h, double * means) {
        double xj[N_DIM], raw[N_DIM];
        cell_center(j, raw);
        nearest(raw, x0, xj);
        for (int l = 0; l < nk; l++) means[l] = 0.0;
        double volume = 0.0;
        auto add = [&](const double * p, double w) {
            if (m.axisymmetric) w *= std::abs(p[1]);
            volume += w;
            double xi[N_DIM], phi[teno::MAX_NK];
            FOR_I_DIM xi[i] = (p[i] - x0[i]) / h;
            teno::monomials(degree, xi, phi);
            for (int l = 0; l < nk; l++) means[l] += w * phi[l];
        };
        for (uint32_t k = 0; k < m.h_n_faces_of_cell(j); k++) {
            const uint32_t f = m.h_face_of_cell(j, k);
            const uint32_t n_nodes = m.h_n_nodes_of_face(f);
            std::vector<std::array<double, N_DIM>> v(n_nodes);
            for (uint32_t a = 0; a < n_nodes; a++) {
                double x[N_DIM];
                node(m.h_node_of_face(f, a), x);
                nearest(x, xj, v[a].data());
            }
            if constexpr (N_DIM == 2) {
                const double e1[2] = {v[0][0] - xj[0], v[0][1] - xj[1]}, e2[2] = {v[1][0] - xj[0], v[1][1] - xj[1]};
                const double area = 0.5 * std::abs(e1[0] * e2[1] - e1[1] * e2[0]);
                for (size_t q = 0; q < tri.w.size(); q++) {
                    const double p[2] = {xj[0] + tri.xi[q] * e1[0] + tri.eta[q] * e2[0],
                                         xj[1] + tri.xi[q] * e1[1] + tri.eta[q] * e2[1]};
                    add(p, 2.0 * area * tri.w[q]);
                }
            } else {
                double g[3] = {0.0, 0.0, 0.0};
                for (const auto & p : v) {
                    for (size_t i = 0; i < 3; i++) g[i] += p[i] / double(n_nodes);
                }
                for (uint32_t a = 0; a < n_nodes; a++) {
                    const auto & p1 = v[a];
                    const auto & p2 = v[(a + 1) % n_nodes];
                    double e[3][3];
                    for (size_t i = 0; i < 3; i++) {
                        e[0][i] = g[i] - xj[i];
                        e[1][i] = p1[i] - xj[i];
                        e[2][i] = p2[i] - xj[i];
                    }
                    const double vol = std::abs(e[0][0] * (e[1][1] * e[2][2] - e[1][2] * e[2][1]) -
                                                e[0][1] * (e[1][0] * e[2][2] - e[1][2] * e[2][0]) +
                                                e[0][2] * (e[1][0] * e[2][1] - e[1][1] * e[2][0])) /
                                       6.0;
                    for (size_t q = 0; q < tet.w.size(); q++) {
                        double p[3];
                        for (size_t i = 0; i < 3; i++) {
                            p[i] = xj[i] + tet.x[q][0] * e[0][i] + tet.x[q][1] * e[1][i] + tet.x[q][2] * e[2][i];
                        }
                        add(p, 6.0 * vol * tet.w[q]);
                    }
                }
            }
        }
        for (int l = 0; l < nk; l++) means[l] /= volume;
    };
    // Quadrature point q of boundary face f (in its cell's frame); false for padding points
    auto wall_point = [&](const uint32_t f, const uint8_t q, double * x) {
        if constexpr (N_DIM == 2) {
            const uint32_t a = m.h_node_of_face(f, 0), b = m.h_node_of_face(f, 1);
            const double s_q = 0.5 * double(quadrature_face.h_points(q, 0));
            FOR_I_DIM x[i] = double(m.h_face_coords(f, i)) +
                             s_q * (double(m.h_node_coords(b, i)) - double(m.h_node_coords(a, i)));
            return true;
        } else {
            if (h_quad_weights(f, q) == 0.0_r) return false;
            if (h_quad_points.extent(0) > 0) {
                FOR_I_DIM x[i] = double(h_quad_points(f, q, i));
            } else {
                // Planar faces: the points the device places from the nodes (planar_face_point)
                rtype x_q[N_DIM];
                planar_face_point(planar_face(h_face_nodes, h_node_coords, f), h_face_rule, q, x_q);
                FOR_I_DIM x[i] = double(x_q[i]);
            }
            return true;
        }
    };

    std::vector<uint32_t> cells, entries;
    std::vector<int32_t> bcs;
    std::vector<uint16_t> sizes;
    std::vector<rtype> scales, means, free_pinv, wall_pinv, wall_sum;
    std::vector<uint8_t> fitted(m.n_state_faces(), 0);
    uint16_t max_size = 0;
    std::vector<double> A, X;
    for_each_stencil([&](const uint32_t c, const CellStencils & st) {
        if (c >= n_direct_rows) return;
        std::vector<int32_t> walls;
        std::vector<uint32_t> own;
        bool usable = true;
        for (uint16_t s = 0; s < st.central; s++) {
            const int32_t j = st.cells[s], code = st.faces[s];
            if (code < -1) usable = false;
            if (code >= 0 && is_wall(code) && std::find(walls.begin(), walls.end(), code) == walls.end()) {
                walls.push_back(code);
            }
            if (code == -1 && j != static_cast<int32_t>(c)) own.push_back(static_cast<uint32_t>(j));
        }
        if (walls.empty() || !usable) return;
        const int32_t bc = h_face_bc(static_cast<size_t>(walls[0]));
        for (const int32_t f : walls) {
            usable = usable && h_face_bc(static_cast<size_t>(f)) == bc &&
                     (h_state_index.extent(0) == 0 || h_state_index(static_cast<size_t>(f)) < 0);
        }
        if (!usable) return;
        double x0[N_DIM];
        cell_center(c, x0);
        const double h = double(h_scale(c));
        double mean0[teno::MAX_NK];
        for (int l = 0; l < nk; l++) mean0[l] = double(h_mean(h_shape(c), l));
        const int n_own = static_cast<int>(own.size());
        std::vector<double> rows(size_t(n_own) * nk);
        for (int s = 0; s < n_own; s++) {
            double mj[teno::MAX_NK];
            cell_means(own[size_t(s)], x0, h, mj);
            for (int l = 0; l < nk; l++) rows[size_t(s) * nk + size_t(l)] = mj[l] - mean0[l];
        }
        int n_points = 0;
        for (const int32_t f : walls) {
            for (uint8_t q = 0; q < n_quad; q++) {
                double raw[N_DIM], x[N_DIM], xi[N_DIM], phi[teno::MAX_NK];
                if (!wall_point(static_cast<uint32_t>(f), q, raw)) continue;
                nearest(raw, x0, x);
                FOR_I_DIM xi[i] = (x[i] - x0[i]) / h;
                teno::monomials(degree, xi, phi);
                for (int l = 0; l < nk; l++) rows.push_back(phi[l] - mean0[l]);
                n_points++;
            }
        }
        // The first n_rows rows over the monomials of degree <= d, dropping degrees while the rows do not
        // resolve them: without its mirror images, the first layer of cells at a flat wall of quadrilaterals or
        // hexahedra sees one level of cells too few to fit the density to the full degree normal to the wall.
        // The velocity's and isothermal walls' temperature gradients take its gradient times their differences
        // from the wall's values, small there; adiabatic walls' temperature gradient loses the degree
        auto fit = [&](const int n_rows, std::vector<double> & X) {
            for (uint8_t d = degree; d >= 1; d--) {
                const int n_cols = teno::n_dof(d);
                if (n_rows < n_cols) continue;
                std::vector<double> A(size_t(n_rows) * size_t(n_cols)), Xd;
                for (int r = 0; r < n_rows; r++) {
                    for (int l = 0; l < n_cols; l++) A[size_t(r) * size_t(n_cols) + size_t(l)] = rows[size_t(r) * nk + size_t(l)];
                }
                if (!teno_setup::least_squares_inverse(A, n_rows, n_cols, max_condition, Xd)) continue;
                X.assign(size_t(n_rows) * nk, 0.0);
                for (int r = 0; r < n_rows; r++) {
                    for (int l = 0; l < n_cols; l++) X[size_t(r) * nk + size_t(l)] = Xd[size_t(r) * size_t(n_cols) + size_t(l)];
                }
                return d;
            }
            return uint8_t(0);
        };
        std::vector<double> X_free, X_wall;
        const uint8_t d_free = fit(n_own, X_free), d_wall = fit(n_own + n_points, X_wall);
        if (d_free < 1 || d_wall < 2) return;
        cells.push_back(c);
        bcs.push_back(bc);
        scales.push_back(rtype(h));
        sizes.push_back(static_cast<uint16_t>(n_own));
        max_size = std::max(max_size, static_cast<uint16_t>(n_own));
        entries.insert(entries.end(), own.begin(), own.end());
        for (int l = 0; l < nk; l++) {
            means.push_back(rtype(mean0[l]));
            double sum = 0.0;
            for (int s = n_own; s < n_own + n_points; s++) sum += X_wall[size_t(s) * nk + size_t(l)];
            wall_sum.push_back(rtype(sum));
        }
        for (int s = 0; s < n_own; s++) {
            for (int l = 0; l < nk; l++) {
                free_pinv.push_back(rtype(X_free[size_t(s) * nk + size_t(l)]));
                wall_pinv.push_back(rtype(X_wall[size_t(s) * nk + size_t(l)]));
            }
        }
        for (uint32_t k = 0; k < m.h_n_faces_of_cell(c); k++) {
            const uint32_t f = m.h_face_of_cell(c, k);
            if (m.h_cells_of_face(f, 1) < 0 && is_wall(static_cast<int32_t>(f))) fitted[f] = 1;
        }
    }, false);

    const size_t n = cells.size();
    {
        wall_fits.cells = Kokkos::View<uint32_t *>("wall_fit_cells", n);
        wall_fits.bc = Kokkos::View<int32_t *>("wall_fit_bc", n);
        wall_fits.scale = Kokkos::View<rtype *>("wall_fit_scale", n);
        wall_fits.size = Kokkos::View<uint16_t *>("wall_fit_size", n);
        wall_fits.entries = Kokkos::View<uint32_t **>("wall_fit_entries", n, max_size);
        wall_fits.mean = Kokkos::View<rtype **>("wall_fit_mean", n, nk);
        wall_fits.free_pinv = Kokkos::View<rtype ***>("wall_fit_free_pinv", n, nk, max_size);
        wall_fits.wall_pinv = Kokkos::View<rtype ***>("wall_fit_wall_pinv", n, nk, max_size);
        wall_fits.wall_sum = Kokkos::View<rtype **>("wall_fit_wall_sum", n, nk);
        auto h_cells = Kokkos::create_mirror_view(wall_fits.cells);
        auto h_bc = Kokkos::create_mirror_view(wall_fits.bc);
        auto h_fit_scale = Kokkos::create_mirror_view(wall_fits.scale);
        auto h_size = Kokkos::create_mirror_view(wall_fits.size);
        auto h_entries = Kokkos::create_mirror_view(wall_fits.entries);
        auto h_fit_mean = Kokkos::create_mirror_view(wall_fits.mean);
        auto h_free = Kokkos::create_mirror_view(wall_fits.free_pinv);
        auto h_wall = Kokkos::create_mirror_view(wall_fits.wall_pinv);
        auto h_sum = Kokkos::create_mirror_view(wall_fits.wall_sum);
        size_t e = 0;
        for (size_t k = 0; k < n; k++) {
            h_cells(k) = cells[k];
            h_bc(k) = bcs[k];
            h_fit_scale(k) = scales[k];
            h_size(k) = sizes[k];
            for (uint16_t s = 0; s < sizes[k]; s++) {
                h_entries(k, s) = entries[e + s];
                for (uint8_t l = 0; l < nk; l++) {
                    h_free(k, l, s) = free_pinv[(e + s) * nk + l];
                    h_wall(k, l, s) = wall_pinv[(e + s) * nk + l];
                }
            }
            e += sizes[k];
            for (uint8_t l = 0; l < nk; l++) {
                h_fit_mean(k, l) = means[k * nk + l];
                h_sum(k, l) = wall_sum[k * nk + l];
            }
        }
        Kokkos::deep_copy(wall_fits.cells, h_cells);
        Kokkos::deep_copy(wall_fits.bc, h_bc);
        Kokkos::deep_copy(wall_fits.scale, h_fit_scale);
        Kokkos::deep_copy(wall_fits.size, h_size);
        Kokkos::deep_copy(wall_fits.entries, h_entries);
        Kokkos::deep_copy(wall_fits.mean, h_fit_mean);
        Kokkos::deep_copy(wall_fits.free_pinv, h_free);
        Kokkos::deep_copy(wall_fits.wall_pinv, h_wall);
        Kokkos::deep_copy(wall_fits.wall_sum, h_sum);
    }
    fitted_walls = Kokkos::View<uint8_t *>("fitted_walls", fitted.size());
    Kokkos::deep_copy(fitted_walls, Kokkos::View<const uint8_t *, Kokkos::HostSpace>(fitted.data(), fitted.size()));
}

TENO::Stencils TENO::large_stencils() const {
    Stencils out;
    out.offsets.push_back(0);
    for_each_stencil([&](uint32_t, const CellStencils & s) {
        out.cells.insert(out.cells.end(), s.cells.begin(), s.cells.begin() + s.central);
        out.faces.insert(out.faces.end(), s.faces.begin(), s.faces.begin() + s.central);
        out.offsets.push_back(out.cells.size());
    }, false);
    return out;
}

std::vector<uint8_t> TENO::stencil_cells(const std::vector<uint8_t> & cells) const {
    std::vector<uint8_t> used(mesh->n_cells, 0);
    for_each_stencil([&](uint32_t cell, const CellStencils & s) {
        if (!cells[cell]) return;
        for (const int32_t c : s.cells) used[static_cast<size_t>(c)] = 1;
    }, false);
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
    // A cache of shared tables holds each shape's table in canonical order; one of per-cell tables, which
    // sharing turns into the same, loads either way
    if (!share_tables) hash.add(uint8_t(1));
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
    // The cells with tables, when they are not [0, n_reconstructed)
    if (n_rows() != mesh->n_reconstructed() || !extra_cells.empty()) {
        hash.add(n_direct_rows);
        hash.add(extra_cells.data(), extra_cells.size() * sizeof(uint32_t));
    }
    hash.add_view(mesh->h_node_coords);
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
    // Curved cells have other centroids, volumes and moments, and curved faces other points
    if (mesh->curved_geometry) {
        hash.add(uint8_t(2));
        hash.add(curved_mirrors);
        hash.add(curved_wall_degree);
        hash.add_view(mesh->h_cell_coords);
        hash.add(mesh->h_cell_volume.data(), mesh->h_cell_volume.span() * sizeof(rtype));
        auto h_points = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), face_quad_points);
        hash.add_view(h_points);
    }
    if (mesh->is_periodic()) {
        hash.add_view(mesh->h_shifts);
        hash.add(mesh->h_face_shift.data(), mesh->h_face_shift.span() * sizeof(uint8_t));
        if (mesh->is_rotational()) hash.add_view(mesh->h_shift_rotations);
    }
    return hash.h;
}

uint8_t TENO::cached_halo_layers(const Input & input) {
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
    if (cache_file.empty() || cache_loaded || !sector_stencils) return;
    // Written aside and renamed, so an interrupted run never leaves a truncated cache
    const std::string partial = cache_file + ".partial";
    std::ofstream out(partial, std::ios::binary);
    if (!out.good()) {
        logging::warning(diagnostic_code::TENO_CACHE, "TENO: could not write stencil cache " + cache_file + ".");
        return;
    }
    const uint32_t n_table_rows = static_cast<uint32_t>(scale.extent(0));
    CacheHeader header;
    header.options_key = options_key();
    header.cache_key = cache_key();
    header.halo_layers = halo_layers;
    header.n_table_rows = n_table_rows;
    std::strncpy(header.producer, setup_producer.c_str(), sizeof(header.producer) - 1);
    write_header(out, header);
    const auto large_index = host_index(stencil_large);
    const auto small_index = host_index(stencil_small);
    std::vector<CellTables> chunk;
    std::vector<char> buf;
    for (uint32_t c0 = 0; c0 < n_table_rows; c0 += CHUNK_CELLS) {
        chunk.assign(std::min(CHUNK_CELLS, n_table_rows - c0), CellTables());
        download_tables(*this, large_index, small_index, c0, chunk);
        buf.clear();
        for (const CellTables & t : chunk) serialize(t, cache_single, buf);
        out.write(buf.data(), static_cast<std::streamsize>(buf.size()));
    }
    out.close();
    std::error_code error;
    if (out.good()) std::filesystem::rename(partial, cache_file, error);
    if (!out.good() || error) {
        std::filesystem::remove(partial, error);
        logging::warning(diagnostic_code::TENO_CACHE, "TENO: could not write stencil cache " + cache_file + ".");
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
    const uint32_t n_table_rows = n_rows();
    if (header.options_key != options_key() || header.n_table_rows != n_table_rows ||
        header.cache_key != cache_key()) {
        cache_status = cache_file + " does not match this case, recomputed";
        return false;
    }
    TableBuilder builder(*this, n_table_rows);
    const uint16_t n_moments = N_DIM == 3 ? teno_setup::n_moments(degree) : 0;
    std::vector<CellTables> chunk;
    bool ok = true;
    for (uint32_t c0 = 0; c0 < n_table_rows && ok; c0 += CHUNK_CELLS) {
        chunk.assign(std::min(CHUNK_CELLS, n_table_rows - c0), CellTables());
        for (CellTables & t : chunk) {
            ok = ok && deserialize(in, n_dof_large, n_moments, cache_single, t);
            if (!sector_stencils) {
                t.small_cells.clear();
                t.small_faces.clear();
                t.small_pinv.clear();
                t.small_size = {};
            }
            largest_stencil = std::max(largest_stencil, static_cast<uint32_t>(t.large_cells.size()));
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

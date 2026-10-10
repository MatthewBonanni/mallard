/**
 * @file teno_setup.h
 * @author Matthew Bonanni (mbonanni001@gmail.com)
 * @brief TENO-E table setup in 3D, per cell on the device (internal to the TENO implementation).
 * @version 0.1
 * @date 2026-10-09
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 *
 */

#ifndef TENO_SETUP_H
#define TENO_SETUP_H

#include <array>
#include <memory>
#include <string>
#include <vector>

#include <Kokkos_Core.hpp>

#include "boundary.h"
#include "mesh.h"
#include "teno.h"

namespace teno_setup {

// Relative tolerance for geometric tests on the mesh, whose coordinates are rtype
constexpr double GEOMETRY_TOL = precision_tol<double>(1e-10, 1e-5);

// Largest accepted Lebesgue constant of a central stencil's reconstruction at
// the cell's face quadrature points. Full-rank stencils that resolve a
// direction only through small centroid offsets reach tens to hundreds and
// amplify the truncation error alike.
constexpr double MAX_LEBESGUE_2D = 10.0;
// In 3D a bound of 10 still accepts near-degenerate stencils wherever the ties
// of a lattice are broken (mirror walls of prism, tetrahedral and mixed
// tilings, seams, jittered meshes): their face values follow the cells across
// the face, and the linearized scheme grows (#126, #229). Lattices are stable
// up to about 10 only because their tie-complete shells skip those sizes.
constexpr double MAX_LEBESGUE_3D = 4.0;
// Where no stencil reaches it (order 6 on tetrahedra and pyramids: 4.3-5), the
// smallest stencil within this factor of the best one found; the best alone is
// often the largest stencil, and markedly less accurate
constexpr double LEBESGUE_SLACK = 1.25;

/** @brief Element k of one cell's array in a batch: interleaved across cells on GPUs. */
template <typename T>
struct Lane {
    T * p;
    size_t stride;

    KOKKOS_INLINE_FUNCTION
    T & operator[](const size_t k) const { return p[k * stride]; }

    KOKKOS_INLINE_FUNCTION
    Lane at(const size_t k) const { return Lane{p + k * stride, stride}; }
};

/**
 * @brief An array of cap elements for each of slots cells. On GPUs element k
 *        of every cell is contiguous, so that a warp working on consecutive
 *        cells reads consecutive words; on the host each cell's array is.
 */
template <typename T, class Space = Kokkos::DefaultExecutionSpace::memory_space>
struct SlotArray {
    Kokkos::View<T *, Kokkos::LayoutRight, Space> data;
    size_t cap = 0;
    size_t slots = 0;
    bool interleaved = false;

    SlotArray() = default;
    SlotArray(const std::string & label, const size_t cap_in, const size_t slots_in, const bool interleaved_in)
        : data(Kokkos::view_alloc(Kokkos::WithoutInitializing, label), cap_in * slots_in), cap(cap_in),
          slots(slots_in), interleaved(interleaved_in) {}

    KOKKOS_INLINE_FUNCTION
    Lane<T> lane(const size_t slot) const {
        return interleaved ? Lane<T>{data.data() + slot, slots} : Lane<T>{data.data() + slot * cap, 1};
    }
};

/**
 * @brief Tables of a batch of cells before packing, one slot per cell, at the
 *        largest stencil sizes the searches allow.
 */
template <class Space>
struct TableBatchT {
    template <typename T>
    using V = Kokkos::View<T, Kokkos::LayoutRight, Space>;
    uint32_t slots = 0;
    uint8_t nk = 0;
    uint16_t max_large = 0;   // central stencil entries per cell
    uint16_t max_small = 0;   // sector stencil entries per cell, all faces
    V<uint8_t *> status;         // (slot): OK, or why the cell must be redone on the host
    V<uint8_t *> gather_depth;   // (slot)
    V<uint8_t *> failed_large;   // (slot): no central stencil, and the search was not cut by the halo
    V<uint8_t *> invalid_small;  // (slot): sector stencils left out
    V<uint16_t *> large_size;    // (slot)
    V<uint16_t **> small_size;   // (slot, face)
    V<uint16_t *> small_total;   // (slot)
    SlotArray<rtype, Space> scale, basis_mean, si;
    SlotArray<int32_t, Space> large_cells, large_faces, small_cells, small_faces;  // sectors one after another
    SlotArray<rtype, Space> large_pinv, small_pinv;  // (s * width + l)

    TableBatchT() = default;
    TableBatchT(const uint32_t n, const uint8_t nk_in, const uint16_t max_large_in, const uint16_t max_small_in,
                const bool interleaved)
        : slots(n), nk(nk_in), max_large(max_large_in), max_small(max_small_in),
          status("teno_batch_status", n), gather_depth("teno_batch_depth", n), failed_large("teno_batch_failed", n),
          invalid_small("teno_batch_invalid", n), large_size("teno_batch_large_size", n),
          small_size("teno_batch_small_size", n, teno::MAX_FACES), small_total("teno_batch_small_total", n),
          scale("teno_batch_scale", 1, n, interleaved), basis_mean("teno_batch_mean", nk_in, n, interleaved),
          si("teno_batch_si", size_t(nk_in) * (nk_in + 1) / 2, n, interleaved),
          large_cells("teno_batch_large_cells", max_large_in, n, interleaved),
          large_faces("teno_batch_large_faces", max_large_in, n, interleaved),
          small_cells("teno_batch_small_cells", max_small_in, n, interleaved),
          small_faces("teno_batch_small_faces", max_small_in, n, interleaved),
          large_pinv("teno_batch_large_pinv", size_t(max_large_in) * nk_in, n, interleaved),
          small_pinv("teno_batch_small_pinv", size_t(max_small_in) * teno::NK_SMALL, n, interleaved) {}
};

using TableBatch = TableBatchT<Kokkos::DefaultExecutionSpace::memory_space>;

/** @brief Whether batches interleave their cells (GPUs). */
bool batch_interleaved();

/** @brief The device copy of a batch on the host. */
TableBatch device_copy(const TableBatchT<Kokkos::HostSpace> & h);

/** @brief Why a cell's device setup must be redone on the host. */
enum Status : uint8_t { OK = 0, OUT_OF_SCRATCH = 1 };

/** @brief Options and sizes of the 3D setup. */
struct Options {
    uint8_t degree = 4;
    uint8_t nk = 0;
    uint16_t ns = 0;         // central stencil size to start from
    uint16_t nss = 0;        // sector stencil size
    double max_condition = 1e8;
    uint32_t batch_cells = 0;  // cells per device batch; 0 picks one from the device memory
    bool host_only = false;    // run every cell on the host (tests)
    const std::vector<uint32_t> * cells = nullptr;  // the cells that will be set up, if not all reconstructed ones
    bool curved_mirrors = false;  // mirror images across curved boundary faces (their planes) too
    uint8_t curved_wall_degree = 3;  // largest central degree of stencils reaching a curved wall without mirrors
    uint16_t wall_ns = 0;            // central stencil size to start from at that degree
};

/** @brief Timings of the last setup, in seconds. */
struct Timings {
    double geometry = 0.0;  // per-cell scales, metrics, moments and bounding boxes
    double tables = 0.0;    // stencils, pseudo-inverses and smoothness indicators
    double host_cells = 0.0;  // cells redone on the host
    uint64_t n_host_cells = 0;
};

class Setup3D;

/**
 * @brief The 3D TENO-E precomputation of a mesh: the geometry it needs (cell
 *        moments, bounding boxes and ranking metrics), computed once, and the
 *        tables of any batch of cells. Every cell's tables are a function of
 *        the mesh alone, computed with the same sequence of floating-point
 *        operations on the device and on the host, so they do not depend on
 *        the batch, the thread count or where they ran.
 */
class Setup3DHandle {
    public:
        Setup3DHandle(const Mesh & mesh, const BoundaryData & boundaries, Kokkos::View<rtype ***> face_quad_points,
                      Kokkos::View<rtype **> face_quad_weights, const Options & options);
        ~Setup3DHandle();

        /**
         * @brief Tables of cells[0..n) into slots 0..n of out (out.slots >= n);
         *        cells must be reconstructed cells.
         */
        void compute(const Kokkos::View<uint32_t *> & cells, uint32_t n, TableBatch & out);

        /** @brief Cells per batch that fit the device. */
        uint32_t batch_cells() const;

        uint16_t max_large() const;
        uint16_t max_small() const;
        const Timings & timings() const;

    private:
        std::unique_ptr<Setup3D> impl;
};

/**
 * @brief Metric (3 x 3, row-major) in which the stencil candidates of cell i
 *        are ranked by distance: the identity unless the spacing of its vertex
 *        neighbors (completed across its boundary faces by mirror images) is
 *        markedly anisotropic.
 */
std::array<double, 9> ranking_metric(const Mesh & mesh, const Kokkos::View<int32_t *>::host_mirror_type & face_bc,
                                     const Kokkos::View<BoundaryCondition *>::host_mirror_type & bcs, uint32_t i);

}  // namespace teno_setup

#endif // TENO_SETUP_H

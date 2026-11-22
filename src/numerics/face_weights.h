/**
 * @file face_weights.h
 * @brief Per-face quadrature weights stored once per distinct row.
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 */

#ifndef FACE_WEIGHTS_H
#define FACE_WEIGHTS_H

#include <cstdint>
#include <functional>
#include <string>

#include <Kokkos_Core.hpp>

#include "common_typedef.h"

/**
 * @brief The quadrature weights of every face, (face, q), read as a View of
 *        that shape. Faces whose rows are bitwise equal share one row of
 *        `table`, which `rows` names per face: the faces of a structured block
 *        have a few rows per direction and plane, triangles one. Values are
 *        exactly the per-face ones, so nothing reading them changes.
 */
template <typename MemorySpace = Kokkos::DefaultExecutionSpace::memory_space>
struct FaceWeights {
    template <typename T>
    using V = Kokkos::View<T, Kokkos::DefaultExecutionSpace::array_layout, MemorySpace>;
    V<uint32_t *> rows;  ///< (face): its row of table; unallocated when every face has its own
    V<rtype **> table;   ///< (row, q)
    uint32_t n_faces = 0;

    /** @brief Weight q of face f. */
    KOKKOS_INLINE_FUNCTION
    rtype operator()(const uint32_t f, const uint32_t q) const { return table(rows.data() ? rows(f) : f, q); }

    /** @brief Extents as the (face, q) View they stand for: faces, then points per face. */
    template <typename I>
    KOKKOS_INLINE_FUNCTION size_t extent(const I r) const {
        return r == 0 ? size_t(n_faces) : size_t(table.extent(1));
    }

    bool is_allocated() const { return table.is_allocated(); }

    /** @brief A copy in host memory (the same Views in host builds). */
    FaceWeights<Kokkos::HostSpace> host() const {
        // A mirror of unallocated rows is an allocated, empty View, which operator() would index.
        FaceWeights<Kokkos::HostSpace> out;
        if (rows.is_allocated()) out.rows = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), rows);
        out.table = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), table);
        out.n_faces = n_faces;
        return out;
    }
};

/**
 * @brief Weights of n_faces faces with n_q points each, row_of(f, w) writing
 *        face f's into w, stored once per distinct row (bitwise). Rows are
 *        numbered by their first face. When no two faces share a row, or
 *        share is false, the table is per face and rows is empty.
 * @param label Label of the table; rows take label + "_rows".
 */
FaceWeights<> share_face_weights(const std::string & label, uint32_t n_faces, uint32_t n_q,
                                 const std::function<void(uint32_t, rtype *)> & row_of, bool share = true);

#endif // FACE_WEIGHTS_H

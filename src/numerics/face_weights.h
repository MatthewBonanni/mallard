/**
 * @file face_weights.h
 * @brief Per-face data (quadrature weights, normals, areas) stored once per
 *        distinct row.
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 */

#ifndef FACE_WEIGHTS_H
#define FACE_WEIGHTS_H

#include <cstdint>
#include <functional>
#include <string>
#include <vector>

#include <Kokkos_Core.hpp>

#include "common_typedef.h"

/**
 * @brief A value or row of values per face, (face, k), read as a View of that
 *        shape: quadrature weights (face, q), normals (face, d), areas (face).
 *        Faces whose rows are bitwise equal share one row of `table`, which
 *        `rows` names per face: the faces of a structured block have a few
 *        rows per direction and plane. Values are exactly the per-face ones, so
 *        nothing reading them changes. With `rows` empty, `table` is per face
 *        and can be written (the mesh's setup does, before share_face_rows).
 */
template <typename MemorySpace = Kokkos::DefaultExecutionSpace::memory_space>
struct FaceRows {
    template <typename T>
    using V = Kokkos::View<T, Kokkos::DefaultExecutionSpace::array_layout, MemorySpace>;
    V<uint32_t *> rows;  ///< (face): its row of table; empty when every face has its own
    V<rtype **> table;   ///< (row, q)
    uint32_t n_faces = 0;

    /** @brief Row of face f in table. */
    KOKKOS_INLINE_FUNCTION
    uint32_t row(const uint32_t f) const { return rows.data() ? rows(f) : f; }

    /** @brief Value k of face f. */
    KOKKOS_INLINE_FUNCTION
    rtype operator()(const uint32_t f, const uint32_t k) const { return table(row(f), k); }

    /** @brief The value of face f, for one value per face. */
    KOKKOS_INLINE_FUNCTION
    rtype operator()(const uint32_t f) const { return table(row(f), 0); }

    /** @brief Extents as the (face, k) View they stand for: faces, then values per face. */
    template <typename I>
    KOKKOS_INLINE_FUNCTION size_t extent(const I r) const {
        return r == 0 ? size_t(n_faces) : size_t(table.extent(1));
    }

    bool is_allocated() const { return table.is_allocated(); }

    bool is_shared() const { return rows.is_allocated(); }

    /** @brief A copy in host memory (the same Views in host builds). */
    FaceRows<Kokkos::HostSpace> host() const {
        return {Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), rows),
                Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), table), n_faces};
    }
};

/** @brief n_faces rows of width values each, one per face (writable through table), labeled label. */
inline FaceRows<> per_face_rows(const std::string & label, const uint32_t n_faces, const uint32_t width) {
    return {FaceRows<>::V<uint32_t *>(), FaceRows<>::V<rtype **>(label, n_faces, width), n_faces};
}

/** @brief Quadrature weights of the faces, (face, q), zero on padding points. */
template <typename MemorySpace = Kokkos::DefaultExecutionSpace::memory_space>
using FaceWeights = FaceRows<MemorySpace>;

/**
 * @brief Per-face rows of n_faces faces for several tables that share one row
 *        index: row_of(f, v) writes face f's values of every table one after
 *        the other (widths[0] values, then widths[1], ...). Faces share a row
 *        when all their values are bitwise equal; rows are numbered by their
 *        first face. When no two faces share a row, or share is false, every
 *        table is per face and rows is empty.
 * @param labels Labels of the tables; rows_label that of the row index.
 */
std::vector<FaceRows<>> share_face_rows(const std::string & rows_label, const std::vector<std::string> & labels,
                                        const std::vector<uint32_t> & widths, uint32_t n_faces,
                                        const std::function<void(uint32_t, rtype *)> & row_of, bool share = true);

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

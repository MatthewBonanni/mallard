/**
 * @file periodic_maps.h
 * @author Matthew Bonanni (mbonanni001@gmail.com)
 * @brief Rigid maps between the frames of cells joined across periodic boundaries.
 * @version 0.1
 * @date 2026-10-15
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 *
 */

#ifndef PERIODIC_MAPS_H
#define PERIODIC_MAPS_H

#include <Kokkos_Core.hpp>

#include "common_typedef.h"

/**
 * @brief The rigid map of every periodic shift row (Mesh::shifts): row s takes
 *        a point x in the frame of a neighbor to Q_s x + t_s in the frame of
 *        the cell that sees it, and a vector v to Q_s v. Without rotational
 *        pairs, rotations is empty and Q_s is the identity, so points move by
 *        x + t_s exactly as before and vectors stay.
 */
struct PeriodicMaps {
    Kokkos::View<rtype *[N_DIM]> shifts;
    Kokkos::View<rtype *[N_DIM][N_DIM]> rotations;

    KOKKOS_INLINE_FUNCTION
    bool rotated() const { return rotations.extent(0) > 0; }

    /** @brief v <- Q_s v. */
    KOKKOS_INLINE_FUNCTION
    void rotate(const uint8_t s, rtype * v) const {
        rtype r[N_DIM];
        FOR_I_DIM {
            r[i] = 0.0_r;
            for (uint8_t j = 0; j < N_DIM; j++) r[i] += rotations(s, i, j) * v[j];
        }
        FOR_I_DIM v[i] = r[i];
    }

    /** @brief v <- Q_s^T v. */
    KOKKOS_INLINE_FUNCTION
    void unrotate(const uint8_t s, rtype * v) const {
        rtype r[N_DIM];
        FOR_I_DIM {
            r[i] = 0.0_r;
            for (uint8_t j = 0; j < N_DIM; j++) r[i] += rotations(s, j, i) * v[j];
        }
        FOR_I_DIM v[i] = r[i];
    }

    /** @brief y = Q_s x + t_s: a neighbor's point in the frame of the cell that sees it. */
    KOKKOS_INLINE_FUNCTION
    void forward(const uint8_t s, const rtype * x, rtype * y) const {
        FOR_I_DIM y[i] = x[i];
        if (rotated()) rotate(s, y);
        FOR_I_DIM y[i] = y[i] + shifts(s, i);
    }

    /** @brief y = Q_s^T (x - t_s): the inverse of forward. */
    KOKKOS_INLINE_FUNCTION
    void backward(const uint8_t s, const rtype * x, rtype * y) const {
        FOR_I_DIM y[i] = x[i] - shifts(s, i);
        if (rotated()) unrotate(s, y);
    }

    /** @brief The velocity (or momentum) of a state vector u, components [1, N_DIM], by Q_s or Q_s^T. */
    KOKKOS_INLINE_FUNCTION
    void rotate_state(const uint8_t s, rtype * u, const bool inverse = false) const {
        if (inverse) {
            unrotate(s, u + 1);
        } else {
            rotate(s, u + 1);
        }
    }

    /**
     * @brief Gradients g[k][d] of n variables by Q_s (or Q_s^T): every row is
     *        a vector, and rows [v0, v0 + N_DIM) are the components of a vector
     *        (none if v0 < 0), so that block turns as Q G Q^T.
     */
    KOKKOS_INLINE_FUNCTION
    void rotate_gradients(const uint8_t s, rtype (*g)[N_DIM], const uint8_t n, const int v0,
                          const bool inverse = false) const {
        for (uint8_t k = 0; k < n; k++) {
            if (inverse) {
                unrotate(s, g[k]);
            } else {
                rotate(s, g[k]);
            }
        }
        if (v0 < 0) return;
        for (uint8_t d = 0; d < N_DIM; d++) {
            rtype column[N_DIM];
            FOR_I_DIM column[i] = g[v0 + i][d];
            if (inverse) {
                unrotate(s, column);
            } else {
                rotate(s, column);
            }
            FOR_I_DIM g[v0 + i][d] = column[i];
        }
    }
};

#endif // PERIODIC_MAPS_H

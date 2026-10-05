/**
 * @file characteristic.h
 * @author Matthew Bonanni (mbonanni001@gmail.com)
 * @brief Transverse terms of characteristic (NSCBC) boundaries.
 * @version 0.1
 * @date 2026-10-04
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 *
 */

#ifndef CHARACTERISTIC_H
#define CHARACTERISTIC_H

#include <Kokkos_Core.hpp>

#include "boundary.h"
#include "common.h"
#include "gradient.h"

/**
 * @brief Pieces of the transverse terms of the incoming acoustic wave at
 *        every characteristic face (Yoo & Im 2007; Lodato, Domingo & Vervisch
 *        2008): [u_t . grad p, rho div_t u_t, rho u_t . grad u_n], combined
 *        by BoundaryData::characteristic_W with the face's sound speed c into
 *        T- = u_t . grad p + c^2 rho div_t u_t - c rho u_t . grad u_n.
 *
 * The tangential derivatives are a least-squares fit, in the plane of the
 * face, of the states of the cells of the neighboring characteristic faces
 * (char_neighbors), as finite-difference NSCBC differentiates along the
 * boundary. Cell gradients would pick up the normal variation of waves
 * crossing the boundary, and on triangles the grid-scale odd-even pattern of
 * the velocity, as a transverse divergence, which reflects acoustic waves;
 * reconstructed face states (TENO) make the correction unstable.
 */
struct CharacteristicTransverseFunctor {
    BoundaryData boundaries;
    Kokkos::View<rtype *[N_DIM]> face_coords;
    Kokkos::View<rtype *[N_DIM]> face_normals;
    Kokkos::View<rtype *> face_area;
    Kokkos::View<int32_t *[2]> cells_of_face;
    Kokkos::View<rtype *[N_CONSERVATIVE]> W_cells;

    /** @brief State of the cell of boundary face f. */
    KOKKOS_INLINE_FUNCTION
    void face_state(const uint32_t f, rtype * W) const {
        FOR_I_CONSERVATIVE W[i] = W_cells(cells_of_face(f, 0), i);
    }

    KOKKOS_INLINE_FUNCTION
    void operator()(const uint32_t k) const {
        constexpr uint8_t E = N_DIM + 1;
        const uint32_t f = boundaries.char_faces(k);
        rtype n[N_DIM], n_vec[N_DIM];
        FOR_I_DIM n_vec[i] = face_normals(f, i);
        unit<N_DIM>(n_vec, n);
        rtype W_f[N_CONSERVATIVE];
        face_state(f, W_f);

        // Neighbors in the plane, and a point along the normal that sets the normal derivatives to zero
        LSQSystem lsq;
        const rtype ell = (N_DIM == 2) ? face_area(f) : Kokkos::sqrt(face_area(f));
        rtype dx_n[N_DIM];
        FOR_I_DIM dx_n[i] = ell * n[i];
        lsq.add(dx_n, W_f, W_f);
        const uint32_t n_neighbors = boundaries.char_offsets(k + 1) - boundaries.char_offsets(k);
        if (n_neighbors + 1 < N_DIM) {
            for (uint8_t m = 0; m < 3; m++) boundaries.char_transverse(k, m) = 0.0_r;
            return;
        }
        for (uint32_t j = boundaries.char_offsets(k); j < boundaries.char_offsets(k + 1); j++) {
            const uint32_t g = boundaries.char_faces(boundaries.char_neighbors(j));
            rtype dx[N_DIM], W_g[N_CONSERVATIVE];
            FOR_I_DIM dx[i] = face_coords(g, i) - face_coords(f, i);
            const rtype dx_n_g = dot<N_DIM>(dx, n);
            FOR_I_DIM dx[i] -= dx_n_g * n[i];
            face_state(g, W_g);
            lsq.add(dx, W_f, W_g);
        }
        rtype g[N_CONSERVATIVE][N_DIM];
        lsq.solve(g);

        const rtype u_n = dot<N_DIM>(W_f + 1, n);
        rtype u_t[N_DIM];
        FOR_I_DIM u_t[i] = W_f[1 + i] - u_n * n[i];
        rtype div_t = 0.0_r, u_t_grad_u_n = 0.0_r;
        FOR_I_DIM {
            div_t += g[1 + i][i];
            for (uint8_t j = 0; j < N_DIM; j++) {
                div_t -= n[i] * g[1 + i][j] * n[j];
                u_t_grad_u_n += n[i] * g[1 + i][j] * u_t[j];
            }
        }
        boundaries.char_transverse(k, 0) = dot<N_DIM>(u_t, g[E]);
        boundaries.char_transverse(k, 1) = W_f[0] * div_t;
        boundaries.char_transverse(k, 2) = W_f[0] * u_t_grad_u_n;
    }
};

#endif // CHARACTERISTIC_H

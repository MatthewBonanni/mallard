/**
 * @file gradient.h
 * @author Matthew Bonanni (mbonanni001@gmail.com)
 * @brief Cell gradient reconstruction.
 * @version 0.1
 * @date 2026-10-02
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 *
 */

#ifndef GRADIENT_H
#define GRADIENT_H

#include <Kokkos_Core.hpp>

#include "common.h"
#include "boundary.h"

/**
 * @brief Weighted least-squares gradient of W = [rho, u_x, u_y, p] over
 *        face neighbors, with boundary ghost states placed at the mirror image
 *        of the cell centroid across the boundary face.
 *
 * Exact for linear fields on any mesh where the neighbor offsets span 2D.
 */
struct LSQGradientFunctor {
    Kokkos::View<uint32_t *> offsets_faces_of_cell;
    Kokkos::View<uint32_t *> faces_of_cell;
    Kokkos::View<int32_t *[2]> cells_of_face;
    Kokkos::View<rtype *[N_DIM]> cell_coords;
    Kokkos::View<rtype *[N_DIM]> face_coords;
    Kokkos::View<rtype *[N_DIM]> face_normals;
    BoundaryData boundaries;
    Kokkos::View<rtype *[N_CONSERVATIVE]> W;
    Kokkos::View<rtype *[N_CONSERVATIVE][N_DIM]> gradients;

    /**
     * @brief Offset to and state of the neighbor across face i_face.
     */
    KOKKOS_INLINE_FUNCTION
    void neighbor(const uint32_t i_cell, const uint32_t i_face,
                  const rtype * W_i, rtype * dx, rtype * W_j) const {
        const int32_t c0 = cells_of_face(i_face, 0);
        const int32_t c1 = cells_of_face(i_face, 1);
        if (c1 >= 0) {
            const int32_t j = (c0 == (int32_t)i_cell) ? c1 : c0;
            FOR_I_DIM dx[i] = cell_coords(j, i) - cell_coords(i_cell, i);
            FOR_I_CONSERVATIVE W_j[i] = W(j, i);
        } else {
            rtype n[N_DIM];
            const rtype n_vec[N_DIM] = {face_normals(i_face, 0), face_normals(i_face, 1)};
            unit<N_DIM>(n_vec, n);
            rtype d = 0.0;
            FOR_I_DIM d += (face_coords(i_face, i) - cell_coords(i_cell, i)) * n[i];
            FOR_I_DIM dx[i] = 2.0 * d * n[i];
            boundaries.ghost_W(i_face, W_i, n, W_j);
        }
    }

    KOKKOS_INLINE_FUNCTION
    void operator()(const uint32_t i_cell) const {
        rtype W_i[N_CONSERVATIVE];
        FOR_I_CONSERVATIVE W_i[i] = W(i_cell, i);
        rtype M[3] = {0.0, 0.0, 0.0};
        rtype b[N_CONSERVATIVE][N_DIM] = {};
        for (uint32_t k = offsets_faces_of_cell(i_cell); k < offsets_faces_of_cell(i_cell + 1); k++) {
            rtype dx[N_DIM], W_j[N_CONSERVATIVE];
            neighbor(i_cell, faces_of_cell(k), W_i, dx, W_j);
            const rtype w = 1.0 / (dx[0] * dx[0] + dx[1] * dx[1]);
            M[0] += w * dx[0] * dx[0];
            M[1] += w * dx[0] * dx[1];
            M[2] += w * dx[1] * dx[1];
            FOR_I_CONSERVATIVE {
                const rtype dW = W_j[i] - W_i[i];
                b[i][0] += w * dx[0] * dW;
                b[i][1] += w * dx[1] * dW;
            }
        }
        const rtype inv_det = 1.0 / (M[0] * M[2] - M[1] * M[1]);
        FOR_I_CONSERVATIVE {
            gradients(i_cell, i, 0) = inv_det * ( M[2] * b[i][0] - M[1] * b[i][1]);
            gradients(i_cell, i, 1) = inv_det * (-M[1] * b[i][0] + M[0] * b[i][1]);
        }
    }
};

#endif // GRADIENT_H

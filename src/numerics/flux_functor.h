/**
 * @file flux_functor.h
 * @author Matthew Bonanni (mbonanni001@gmail.com)
 * @brief Convective flux functor.
 * @version 0.2
 * @date 2024-11-26
 *
 * @copyright Copyright (c) 2024 Matthew Bonanni
 *
 */

#ifndef FLUX_FUNCTOR_H
#define FLUX_FUNCTOR_H

#include <Kokkos_Core.hpp>

#include "common.h"
#include "boundary.h"

/**
 * @brief Integrates the convective flux over every face into face_flux, the
 *        rate of change it causes in cells_of_face(:, 0) (cells_of_face(:, 1)
 *        receives its negative). Boundary faces use the boundary ghost state
 *        as the right state.
 *
 * Face normals point from cells_of_face(:, 0) to cells_of_face(:, 1), i.e.
 * out of the domain on boundary faces.
 */
template <typename T_riemann_solver>
struct ConvectiveFluxFunctor {
    Kokkos::View<rtype *[N_DIM]> normals;
    Kokkos::View<rtype *> face_area;
    Kokkos::View<int32_t *[2]> cells_of_face;
    Kokkos::View<rtype *> quad_weights;
    Kokkos::View<rtype **> face_weights;  // 3D: (face, q), zero on padding points
    Kokkos::View<rtype **[2][N_CONSERVATIVE]> face_solution;
    BoundaryData boundaries;
    Kokkos::View<rtype *[N_CONSERVATIVE]> W_cells;
    Kokkos::View<rtype *[N_CONSERVATIVE]> face_flux;
    rtype gamma;

    KOKKOS_INLINE_FUNCTION
    void operator()(const uint32_t i_face) const {
        uint8_t n_quad;
        if constexpr (N_DIM == 2) {
            n_quad = quad_weights.extent(0);
        } else {
            n_quad = face_weights.extent(1);
        }
        const int32_t c1 = cells_of_face(i_face, 1);
        rtype n_unit[N_DIM];
        rtype n_vec[N_DIM];
        FOR_I_DIM n_vec[i] = normals(i_face, i);
        unit<N_DIM>(n_vec, n_unit);

        rtype flux[N_CONSERVATIVE] = {};
        for (uint8_t i_quad = 0; i_quad < n_quad; i_quad++) {
            rtype w_q;
            if constexpr (N_DIM == 2) {
                w_q = quad_weights(i_quad);
            } else {
                w_q = face_weights(i_face, i_quad);
                if (w_q == 0.0) continue;
            }
            rtype W_l[N_CONSERVATIVE], W_r[N_CONSERVATIVE], flux_q[N_CONSERVATIVE];
            FOR_I_CONSERVATIVE W_l[i] = face_solution(i_face, i_quad, 0, i);
            if (c1 >= 0) {
                FOR_I_CONSERVATIVE W_r[i] = face_solution(i_face, i_quad, 1, i);
            } else {
                boundaries.exterior_W(i_face, i_quad, n_quad, W_l, n_unit, W_cells, face_solution, W_r);
            }
            T_riemann_solver::calc_flux(flux_q, n_unit, W_l, W_r, gamma);
            FOR_I_CONSERVATIVE flux[i] += w_q * flux_q[i];
        }

        // Weights sum to 2 (Gauss-Legendre on [-1, 1] in 2D)
        const rtype scale = 0.5 * face_area(i_face);
        FOR_I_CONSERVATIVE face_flux(i_face, i) = -scale * flux[i];
    }
};

/**
 * @brief Sums the face fluxes of each cell into its RHS in faces_of_cell
 *        order, so that the result does not depend on thread scheduling.
 */
struct FaceFluxSumFunctor {
    Kokkos::View<uint32_t *> offsets_faces_of_cell;
    Kokkos::View<uint32_t *> faces_of_cell;
    Kokkos::View<int32_t *[2]> cells_of_face;
    Kokkos::View<rtype *[N_CONSERVATIVE]> face_flux;
    Kokkos::View<rtype *[N_CONSERVATIVE]> rhs;

    KOKKOS_INLINE_FUNCTION
    void operator()(const uint32_t i_cell) const {
        rtype sum[N_CONSERVATIVE] = {};
        for (uint32_t k = offsets_faces_of_cell(i_cell); k < offsets_faces_of_cell(i_cell + 1); k++) {
            const uint32_t i_face = faces_of_cell(k);
            const rtype sign = (cells_of_face(i_face, 0) == static_cast<int32_t>(i_cell)) ? 1.0 : -1.0;
            FOR_I_CONSERVATIVE sum[i] += sign * face_flux(i_face, i);
        }
        FOR_I_CONSERVATIVE rhs(i_cell, i) = sum[i];
    }
};

#endif // FLUX_FUNCTOR_H

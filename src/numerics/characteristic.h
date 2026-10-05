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
 *        2008): [u_t . grad p, rho div_tangential u_t, rho u_t . grad u_n], combined
 *        by BoundaryData::characteristic_W with the face's sound speed c into
 *        T- = u_t . grad p + c^2 rho div_tangential u_t - c rho u_t . grad u_n.
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
        rtype div_tangential = 0.0_r, u_t_grad_u_n = 0.0_r;
        FOR_I_DIM {
            div_tangential += g[1 + i][i];
            for (uint8_t j = 0; j < N_DIM; j++) {
                div_tangential -= n[i] * g[1 + i][j] * n[j];
                u_t_grad_u_n += n[i] * g[1 + i][j] * u_t[j];
            }
        }
        boundaries.char_transverse(k, 0) = dot<N_DIM>(u_t, g[E]);
        boundaries.char_transverse(k, 1) = W_f[0] * div_tangential;
        boundaries.char_transverse(k, 2) = W_f[0] * u_t_grad_u_n;
    }
};

/**
 * @brief Advances the incoming acoustic wave w- = p - rho c u_n of every
 *        characteristic face by one forward-Euler step of the LODI relation
 *        (Poinsot & Lele 1992) at the face, with the transverse terms relaxed
 *        by beta (Lodato, Domingo & Vervisch 2008):
 *          dw-/dt = -K (p - p_t) - beta T-             (outlets)
 *          dw-/dt = K rho c (u_n - u_n,t) - beta T-    (inlets)
 *        with K = sigma c (1 - M^2) / L (K dt capped at 1). The face keeps its
 *        pressure and normal velocity (char_state), which are continuous at
 *        contacts, and the outgoing wave comes from the reconstructed interior
 *        state averaged over the face. Faces of supersonic outflow follow the
 *        interior; with initialize, the face starts from the interior state.
 */
struct CharacteristicStateFunctor {
    BoundaryData boundaries;
    Kokkos::View<rtype *[N_DIM]> face_normals;
    Kokkos::View<rtype **[2][N_CONSERVATIVE]> face_solution;
    Kokkos::View<rtype **[2][2]> face_thermo;  // Gas mixtures: [gamma, e0]; empty for a single gas
    Kokkos::View<rtype *> quad_weights;        // 2D
    Kokkos::View<rtype **> face_weights;       // 3D: (face, q)
    rtype dt;
    bool initialize;

    KOKKOS_INLINE_FUNCTION
    void operator()(const uint32_t k) const {
        using Relax = BoundaryCondition::Relax;
        constexpr uint8_t E = N_DIM + 1;
        const uint32_t f = boundaries.char_faces(k);
        const BoundaryCondition & bc = boundaries.bcs(boundaries.face_bc(f));
        uint8_t n_quad;
        if constexpr (N_DIM == 2) {
            n_quad = static_cast<uint8_t>(quad_weights.extent(0));
        } else {
            n_quad = static_cast<uint8_t>(face_weights.extent(1));
        }
        rtype W[N_CONSERVATIVE] = {};
        rtype gamma = 0.0_r, w_sum = 0.0_r;
        for (uint8_t q = 0; q < n_quad; q++) {
            rtype w;
            if constexpr (N_DIM == 2) {
                w = quad_weights(q);
            } else {
                w = face_weights(f, q);
            }
            if (w == 0.0_r) continue;
            w_sum += w;
            FOR_I_CONSERVATIVE W[i] += w * face_solution(f, q, 0, i);
            gamma += w * (face_thermo.extent(0) > 0 ? face_thermo(f, q, 0, 0) : boundaries.gamma);
        }
        FOR_I_CONSERVATIVE W[i] /= w_sum;
        gamma /= w_sum;
        rtype n[N_DIM], n_vec[N_DIM];
        FOR_I_DIM n_vec[i] = face_normals(f, i);
        unit<N_DIM>(n_vec, n);

        const rtype c2 = gamma * W[E] / W[0];
        const rtype c = Kokkos::sqrt(c2);
        const rtype Z = W[0] * c;
        const rtype u_n = dot<N_DIM>(W + 1, n);
        const rtype w_out = W[E] + Z * u_n;
        if (initialize || u_n >= c) {
            boundaries.char_state(k, 0) = W[E];
            boundaries.char_state(k, 1) = u_n;
            if (u_n >= c) return;
        }
        const bool inlet = bc.type == BoundaryType::NSCBC_INLET;
        if (inlet && u_n <= -c) {
            boundaries.char_state(k, 0) = bc.data[E];
            boundaries.char_state(k, 1) = dot<N_DIM>(bc.data + 1, n);
            return;
        }
        rtype w_in = boundaries.char_state(k, 0) - Z * boundaries.char_state(k, 1);
        const rtype p_b = 0.5_r * (w_out + w_in);
        const rtype u_b = 0.5_r * (w_out - w_in) / Z;
        const rtype M2 = dot<N_DIM>(W + 1, W + 1) / c2;
        const rtype K = Kokkos::fmin(bc.relax[Relax::ACOUSTIC] * c * Kokkos::fmax(1.0_r - M2, 0.0_r), 1.0_r / dt);
        const rtype relaxation = inlet ? K * Z * (u_b - dot<N_DIM>(bc.data + 1, n)) : -K * (p_b - bc.data[E]);
        const rtype beta = (bc.relax[Relax::BETA] < 0.0_r) ? Kokkos::fmin(Kokkos::sqrt(M2), 1.0_r)
                                                            : bc.relax[Relax::BETA];
        const rtype T_in = boundaries.char_transverse(k, 0) + c2 * boundaries.char_transverse(k, 1) -
                           c * boundaries.char_transverse(k, 2);
        w_in += dt * (relaxation - beta * T_in);
        boundaries.char_state(k, 0) = 0.5_r * (w_out + w_in);
        boundaries.char_state(k, 1) = 0.5_r * (w_out - w_in) / Z;
    }
};

#endif // CHARACTERISTIC_H

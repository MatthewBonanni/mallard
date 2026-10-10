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
#include "riemann_solver.h"

/**
 * @brief Low-Mach correction of Thornber et al. (J. Comput. Phys. 227, 2008):
 *        scales the jump of the reconstructed velocity across a face by
 *        z = min(1, max(M_L, M_R, M_cut)), keeping its mean. Upwind fluxes
 *        otherwise damp velocity jumps at the sound speed, a dissipation that
 *        does not vanish as M -> 0; with the correction it scales with the flow
 *        speed. The cutoff M_cut keeps some acoustic damping in gas nearly at
 *        rest, like the cutoff Mach number of preconditioned all-speed schemes
 *        (Weiss & Smith, AIAA J. 33(11), 1995). Supersonic faces (z = 1) are
 *        untouched.
 */
KOKKOS_INLINE_FUNCTION
void low_mach_correction(rtype * W_l, rtype * W_r, const rtype gamma, const rtype M_cut) {
    const rtype M_l2 = dot<N_DIM>(W_l + 1, W_l + 1) * W_l[0] / (gamma * W_l[N_DIM + 1]);
    const rtype M_r2 = dot<N_DIM>(W_r + 1, W_r + 1) * W_r[0] / (gamma * W_r[N_DIM + 1]);
    const rtype z = Kokkos::fmin(1.0_r, Kokkos::fmax(M_cut, Kokkos::sqrt(Kokkos::fmax(M_l2, M_r2))));
    FOR_I_DIM {
        const rtype mean = 0.5_r * (W_l[1 + i] + W_r[1 + i]);
        const rtype half_jump = 0.5_r * (W_l[1 + i] - W_r[1 + i]);
        W_l[1 + i] = mean + z * half_jump;
        W_r[1 + i] = mean - z * half_jump;
    }
}

/**
 * @brief Upwind fraction of a face of the hybrid convective flux: the larger of
 *        its cells' (boundary faces other than walls and symmetry planes, whose
 *        mirror states make the central flux exact, take the Riemann flux).
 */
KOKKOS_INLINE_FUNCTION
rtype face_upwind_fraction(const Kokkos::View<rtype *> & upwind, const BoundaryData & boundaries,
                           const int32_t c0, const int32_t c1, const uint32_t i_face) {
    if (c1 >= 0) return Kokkos::fmax(upwind(c0), upwind(c1));
    const BoundaryCondition & bc = boundaries.bcs(boundaries.face_bc(i_face));
    return (bc.is_wall() || bc.type == BoundaryType::SYMMETRY) ? upwind(c0) : 1.0_r;
}

/**
 * @brief Upwind fraction of each cell for the hybrid convective flux: 1 where
 *        the Ducros sensor restricted to compressions (Ducros et al. 1999;
 *        Bhagatwala & Lele 2009), theta = (div u)^2 / ((div u)^2 + |omega|^2)
 *        for div u < 0, exceeds the threshold (shocks), else the floor.
 *        gradients(c, 1 + k, i) = d u_k / d x_i.
 */
struct UpwindSensorFunctor {
    Kokkos::View<rtype *[N_CONSERVATIVE][N_DIM]> gradients;
    Kokkos::View<rtype *> upwind;
    rtype threshold;
    rtype floor;

    KOKKOS_INLINE_FUNCTION
    void operator()(const uint32_t c) const {
        rtype div = 0.0_r;
        FOR_I_DIM div += gradients(c, 1 + i, i);
        rtype omega2;
        if constexpr (N_DIM == 2) {
            const rtype w = gradients(c, 2, 0) - gradients(c, 1, 1);
            omega2 = w * w;
        } else {
            omega2 = 0.0_r;
            for (uint8_t k = 0; k < 3; k++) {
                const uint8_t a = static_cast<uint8_t>((k + 1) % 3), b = static_cast<uint8_t>((k + 2) % 3);
                const rtype w = gradients(c, 1 + b, a) - gradients(c, 1 + a, b);
                omega2 += w * w;
            }
        }
        const rtype div2 = div * div;
        const bool shock = div < 0.0_r && div2 > threshold * (div2 + omega2);
        upwind(c) = shock ? 1.0_r : floor;
    }
};

/**
 * @brief Integrates the convective flux over every face into face_flux, the
 *        rate of change it causes in cells_of_face(:, 0) (cells_of_face(:, 1)
 *        receives its negative). Boundary faces use the boundary ghost state
 *        as the right state. Characteristic boundaries take the low-Mach
 *        correction like interior faces: without it, the different
 *        dissipation of the boundary cell's two sides holds the outlet
 *        pressure of a steady shear flow away from its target.
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
    Kokkos::View<rtype **> face_weights;  // (face, q): 3D, zero on padding points; 2D axisymmetric, Gauss weight times r; else empty
    Kokkos::View<rtype **[2][N_CONSERVATIVE]> face_solution;
    BoundaryData boundaries;
    Kokkos::View<rtype *[N_CONSERVATIVE]> W_cells;
    Kokkos::View<rtype *[N_CONSERVATIVE]> face_flux;
    rtype gamma;
    rtype low_mach_cutoff;  // 1 disables the low-Mach correction
    Kokkos::View<rtype *> upwind;  // hybrid flux: upwind fraction of each cell (UpwindSensorFunctor), else empty

    KOKKOS_INLINE_FUNCTION
    void operator()(const uint32_t i_face) const {
        uint8_t n_quad;
        if constexpr (N_DIM == 2) {
            n_quad = static_cast<uint8_t>(quad_weights.extent(0));
        } else {
            n_quad = static_cast<uint8_t>(face_weights.extent(1));
        }
        const int32_t c1 = cells_of_face(i_face, 1);
        rtype n_unit[N_DIM];
        rtype n_vec[N_DIM];
        FOR_I_DIM n_vec[i] = normals(i_face, i);
        unit<N_DIM>(n_vec, n_unit);

        const bool hybrid = upwind.extent(0) > 0;
        const rtype phi =
            hybrid ? face_upwind_fraction(upwind, boundaries, cells_of_face(i_face, 0), c1, i_face) : 1.0_r;

        rtype flux[N_CONSERVATIVE] = {};
        for (uint8_t i_quad = 0; i_quad < n_quad; i_quad++) {
            rtype w_q;
            if constexpr (N_DIM == 2) {
                w_q = face_weights.extent(0) ? face_weights(i_face, i_quad) : quad_weights(i_quad);
            } else {
                w_q = face_weights(i_face, i_quad);
                if (w_q == 0.0_r) continue;
            }
            rtype W_l[N_CONSERVATIVE], W_r[N_CONSERVATIVE], flux_q[N_CONSERVATIVE];
            FOR_I_CONSERVATIVE W_l[i] = face_solution(i_face, i_quad, 0, i);
            if (c1 >= 0) {
                FOR_I_CONSERVATIVE W_r[i] = face_solution(i_face, i_quad, 1, i);
            } else {
                boundaries.exterior_W(i_face, i_quad, n_quad, W_l, n_unit, W_cells, face_solution, W_r);
            }
            rtype central[N_CONSERVATIVE] = {};
            if (hybrid) riemann::KEEP::calc_flux(central, n_unit, W_l, W_r, gamma);
            if (phi > 0.0_r) {
                if (low_mach_cutoff < 1.0_r &&
                    (c1 >= 0 || boundaries.bcs(boundaries.face_bc(i_face)).is_characteristic())) {
                    low_mach_correction(W_l, W_r, gamma, low_mach_cutoff);
                }
                T_riemann_solver::calc_flux(flux_q, n_unit, W_l, W_r, gamma);
                if (hybrid) FOR_I_CONSERVATIVE flux_q[i] = central[i] + phi * (flux_q[i] - central[i]);
            } else {
                FOR_I_CONSERVATIVE flux_q[i] = central[i];
            }
            FOR_I_CONSERVATIVE flux[i] += w_q * flux_q[i];
        }

        // Weights sum to 2 (Gauss-Legendre on [-1, 1] in 2D)
        const rtype scale = 0.5_r * face_area(i_face);
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

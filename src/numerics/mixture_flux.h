/**
 * @file mixture_flux.h
 * @author Matthew Bonanni (mbonanni001@gmail.com)
 * @brief Convective fluxes of a gas mixture: the flow block through the
 *        Riemann solvers, the species by mass-flux upwinding.
 * @version 0.3
 * @date 2026-10-02
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 *
 */

#ifndef MIXTURE_FLUX_H
#define MIXTURE_FLUX_H

#include <Kokkos_Core.hpp>

#include "boundary.h"
#include "common.h"
#include "riemann_solver.h"
#include "scalar_reconstruction.h"
#include "state.h"

/**
 * @brief The low-Mach correction of flux_functor.h with each side's own
 *        ratio of specific heats in its Mach number.
 */
KOKKOS_INLINE_FUNCTION
void low_mach_correction(rtype * W_l, rtype * W_r, const rtype gamma_l, const rtype gamma_r, const rtype M_cut) {
    const rtype M_l2 = dot<N_DIM>(W_l + 1, W_l + 1) * W_l[0] / (gamma_l * W_l[N_DIM + 1]);
    const rtype M_r2 = dot<N_DIM>(W_r + 1, W_r + 1) * W_r[0] / (gamma_r * W_r[N_DIM + 1]);
    const rtype z = Kokkos::fmin(1.0_r, Kokkos::fmax(M_cut, Kokkos::sqrt(Kokkos::fmax(M_l2, M_r2))));
    FOR_I_DIM {
        const rtype mean = 0.5_r * (W_l[1 + i] + W_r[1 + i]);
        const rtype half_jump = 0.5_r * (W_l[1 + i] - W_r[1 + i]);
        W_l[1 + i] = mean + z * half_jump;
        W_r[1 + i] = mean - z * half_jump;
    }
}

/**
 * @brief Flow-block flux of a gas mixture over every face, as
 *        ConvectiveFluxFunctor, with the face thermodynamics [gamma, e0] of
 *        each side. Also stores the mass flux per unit area at each
 *        quadrature point, positive from cells_of_face(:, 0) to (:, 1), for
 *        the species fluxes.
 */
template <typename T_riemann_solver>
struct MixtureFluxFunctor {
    Kokkos::View<rtype *[N_DIM]> normals;
    Kokkos::View<rtype *> face_area;
    Kokkos::View<int32_t *[2]> cells_of_face;
    Kokkos::View<rtype *> quad_weights;
    Kokkos::View<rtype **> face_weights;  // 3D: (face, q), zero on padding points
    Kokkos::View<rtype **[2][N_CONSERVATIVE]> face_solution;
    Kokkos::View<rtype **[2][2]> face_thermo;
    BoundaryData boundaries;
    Kokkos::View<rtype *[N_CONSERVATIVE]> W_cells;
    Kokkos::View<rtype **, Kokkos::LayoutStride> cell_thermo;  // (cell, [gamma, e0])
    Kokkos::View<rtype *[N_CONSERVATIVE]> face_flux;
    Kokkos::View<rtype **> face_mdot;  // (face, q)
    rtype low_mach_cutoff;

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

        rtype flux[N_CONSERVATIVE] = {};
        for (uint8_t i_quad = 0; i_quad < n_quad; i_quad++) {
            rtype w_q;
            if constexpr (N_DIM == 2) {
                w_q = quad_weights(i_quad);
            } else {
                w_q = face_weights(i_face, i_quad);
                if (w_q == 0.0_r) {
                    face_mdot(i_face, i_quad) = 0.0_r;
                    continue;
                }
            }
            rtype W_l[N_CONSERVATIVE], W_r[N_CONSERVATIVE], flux_q[N_CONSERVATIVE];
            rtype th_l[2], th_r[2];
            FOR_I_CONSERVATIVE W_l[i] = face_solution(i_face, i_quad, 0, i);
            th_l[0] = face_thermo(i_face, i_quad, 0, 0);
            th_l[1] = face_thermo(i_face, i_quad, 0, 1);
            if (c1 >= 0) {
                FOR_I_CONSERVATIVE W_r[i] = face_solution(i_face, i_quad, 1, i);
                th_r[0] = face_thermo(i_face, i_quad, 1, 0);
                th_r[1] = face_thermo(i_face, i_quad, 1, 1);
                if (low_mach_cutoff < 1.0_r) low_mach_correction(W_l, W_r, th_l[0], th_r[0], low_mach_cutoff);
            } else {
                boundaries.exterior_mixture(i_face, i_quad, n_quad, W_l, th_l, n_unit, W_cells, face_solution,
                                            face_thermo, cell_thermo, W_r, th_r);
            }
            T_riemann_solver::calc_flux(flux_q, n_unit, W_l, W_r, riemann::SideThermo{th_l[0], th_l[1]},
                                        riemann::SideThermo{th_r[0], th_r[1]});
            face_mdot(i_face, i_quad) = flux_q[0];
            FOR_I_CONSERVATIVE flux[i] += w_q * flux_q[i];
        }

        // Weights sum to 2 (Gauss-Legendre on [-1, 1] in 2D)
        const rtype scale = 0.5_r * face_area(i_face);
        FOR_I_CONSERVATIVE face_flux(i_face, i) = -scale * flux[i];
    }
};

/**
 * @brief Species fluxes by mass-flux upwinding (Larrouturou 1991),
 *        F_k = max(mdot, 0) Y_k^L + min(mdot, 0) Y_k^R, split by the side
 *        each part comes from: each cell writes, for each of its faces, the
 *        mass of each species leaving it through that face,
 *        slot(face, side, k) = A/2 sum_q w_q max(mdot_out, 0) Y_k(q), with
 *        Y_k from its own reconstruction; boundary faces also get the inflow
 *        from the exterior state in slot(face, 1, k). No two threads write the
 *        same slot.
 *
 * With face mass fractions in [0, 1] summing to one, the species fluxes sum
 * to the mass flux and keep rho Y_k non-negative under the flow's CFL limit.
 */
struct SpeciesSlotFunctor {
    Kokkos::View<uint32_t *> offsets_faces_of_cell;
    Kokkos::View<uint32_t *> faces_of_cell;
    Kokkos::View<rtype *> face_area;
    Kokkos::View<rtype *> quad_weights;   // 2D
    Kokkos::View<rtype **> face_weights;  // 3D: (face, q)
    Kokkos::View<rtype **> face_mdot;
    ScalarFaceValues values;
    BoundaryData boundaries;
    Kokkos::View<rtype ***, Kokkos::LayoutRight> slots;  // (face, side, k)
    uint32_t n_species;

    /** @brief Mass fraction k beyond boundary face f of cell c (r: offset of c to the face). */
    KOKKOS_INLINE_FUNCTION
    rtype exterior_Y(const uint32_t c, const uint32_t f, const rtype * r, const uint32_t k) const {
        const int32_t image_face = boundaries.face_image_face(f);
        const int32_t image = boundaries.face_image(f);
        if (image_face >= 0) {
            rtype r_image[N_DIM];
            values.offset(image, image_face, boundaries.face_image_side(f), r_image);
            return values.value(image, k, r_image);
        }
        if (image >= 0) return values.scalars(image, k);
        const int32_t i_bc = boundaries.face_bc(f);
        if (boundaries.bcs(i_bc).type == BoundaryType::UPT) return boundaries.bc_Y(i_bc, k);
        return values.value(c, k, r);
    }

    KOKKOS_INLINE_FUNCTION
    void operator()(const uint32_t c) const {
        for (uint32_t i = offsets_faces_of_cell(c); i < offsets_faces_of_cell(c + 1); i++) {
            const uint32_t f = faces_of_cell(i);
            const uint8_t side = (values.cells_of_face(f, 0) == static_cast<int32_t>(c)) ? 0 : 1;
            const bool boundary = values.cells_of_face(f, 1) < 0;
            rtype r[N_DIM];
            values.offset(c, f, side, r);
            rtype out = 0.0_r, in = 0.0_r;  // A/2 sum_q w_q max(+-mdot_out, 0)
            uint8_t n_quad;
            if constexpr (N_DIM == 2) {
                n_quad = static_cast<uint8_t>(quad_weights.extent(0));
            } else {
                n_quad = static_cast<uint8_t>(face_weights.extent(1));
            }
            for (uint8_t q = 0; q < n_quad; q++) {
                rtype w_q;
                if constexpr (N_DIM == 2) {
                    w_q = quad_weights(q);
                } else {
                    w_q = face_weights(f, q);
                }
                const rtype m_out = side == 0 ? face_mdot(f, q) : -face_mdot(f, q);
                out += w_q * Kokkos::fmax(m_out, 0.0_r);
                in += w_q * Kokkos::fmax(-m_out, 0.0_r);
            }
            const rtype scale = 0.5_r * face_area(f);
            // First-order and MUSCL reconstructions have one point per face
            for (uint32_t k = 0; k < n_species; k++) {
                slots(f, side, k) = scale * out * values.value(c, k, r);
                if (boundary) slots(f, 1, k) = scale * in * exterior_Y(c, f, r, k);
            }
        }
    }
};

/**
 * @brief Species RHS per unit volume: the inflow minus the outflow slots of
 *        each face of the cell, in faces_of_cell order.
 */
struct SpeciesSumFunctor {
    Kokkos::View<uint32_t *> offsets_faces_of_cell;
    Kokkos::View<uint32_t *> faces_of_cell;
    Kokkos::View<int32_t *[2]> cells_of_face;
    Kokkos::View<rtype *> cell_volume;
    Kokkos::View<rtype ***, Kokkos::LayoutRight> slots;
    SpeciesView rhs;
    uint32_t n_species;

    KOKKOS_INLINE_FUNCTION
    void operator()(const uint32_t c) const {
        const rtype inv_V = 1.0_r / cell_volume(c);
        for (uint32_t k = 0; k < n_species; k++) {
            rtype sum = 0.0_r;
            for (uint32_t i = offsets_faces_of_cell(c); i < offsets_faces_of_cell(c + 1); i++) {
                const uint32_t f = faces_of_cell(i);
                const uint8_t side = (cells_of_face(f, 0) == static_cast<int32_t>(c)) ? 0 : 1;
                sum += slots(f, 1 - side, k) - slots(f, side, k);
            }
            rhs(c, k) = sum * inv_V;
        }
    }
};

#endif // MIXTURE_FLUX_H

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
#include "flux_functor.h"
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
    Kokkos::View<rtype **> face_weights;  // (face, q): 3D, zero on padding points; 2D axisymmetric, Gauss weight times r; else empty
    Kokkos::View<rtype **[2][N_CONSERVATIVE]> face_solution;
    Kokkos::View<rtype **[2][2]> face_thermo;
    BoundaryData boundaries;
    Kokkos::View<rtype *[N_CONSERVATIVE]> W_cells;
    Kokkos::View<rtype **, Kokkos::LayoutStride> cell_thermo;  // (cell, [gamma, e0])
    Kokkos::View<rtype *[N_CONSERVATIVE]> face_flux;
    Kokkos::View<rtype **> face_mdot;  // (face, q)
    rtype low_mach_cutoff;
    Kokkos::View<rtype *> upwind;  // hybrid flux: upwind fraction of each cell, else empty
    Kokkos::View<rtype ***> quad_normals;  // curved meshes: (face, q, dim) unit normals; else empty

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
            rtype n_q[N_DIM];
            const rtype * n_point = quad_normal(quad_normals, i_face, i_quad, n_unit, n_q);
            rtype w_q;
            if constexpr (N_DIM == 2) {
                w_q = face_weights.extent(0) ? face_weights(i_face, i_quad) : quad_weights(i_quad);
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
            } else {
                boundaries.exterior_mixture(i_face, i_quad, n_quad, W_l, th_l, n_point, W_cells, face_solution,
                                            face_thermo, cell_thermo, W_r, th_r);
            }
            const riemann::SideThermo side_l{th_l[0], th_l[1]}, side_r{th_r[0], th_r[1]};
            rtype central[N_CONSERVATIVE] = {};
            if (hybrid) riemann::KEEP::calc_flux(central, n_point, W_l, W_r, side_l, side_r);
            if (phi > 0.0_r) {
                if (low_mach_cutoff < 1.0_r &&
                    (c1 >= 0 || boundaries.bcs(boundaries.face_bc(i_face)).is_characteristic())) {
                    low_mach_correction(W_l, W_r, th_l[0], th_r[0], low_mach_cutoff);
                }
                T_riemann_solver::calc_flux(flux_q, n_point, W_l, W_r, side_l, side_r);
                if (hybrid) FOR_I_CONSERVATIVE flux_q[i] = central[i] + phi * (flux_q[i] - central[i]);
            } else {
                FOR_I_CONSERVATIVE flux_q[i] = central[i];
            }
            face_mdot(i_face, i_quad) = flux_q[0];
            FOR_I_CONSERVATIVE flux[i] += w_q * flux_q[i];
        }

        // Weights sum to 2 (Gauss-Legendre on [-1, 1] in 2D)
        const rtype scale = 0.5_r * face_area(i_face);
        FOR_I_CONSERVATIVE face_flux(i_face, i) = -scale * flux[i];
    }
};

/**
 * @brief Double-flux variant of MixtureFluxFunctor (Abgrall & Karni 2001;
 *        Billet & Abgrall 2003; Ma, Lv & Ihme 2017): each cell's energy
 *        update uses a flux computed with its own frozen [gamma, e0] on both
 *        sides of the face, so pressure and velocity stay exactly uniform
 *        across contacts between different gases. Mass and momentum use the
 *        mean of the two sides' fluxes, which stays conservative and equals
 *        both at such contacts: both estimate the wave speeds with each
 *        side's own [gamma, e0], which makes their mass fluxes there agree
 *        for every solver, HLL and Rusanov included. face_flux holds side 0's energy flux,
 *        face_energy_1 side 1's (both as the rate of change of side 0).
 */
template <typename T_riemann_solver>
struct MixtureDoubleFluxFunctor {
    Kokkos::View<rtype *[N_DIM]> normals;
    Kokkos::View<rtype *> face_area;
    Kokkos::View<int32_t *[2]> cells_of_face;
    Kokkos::View<rtype *> quad_weights;
    Kokkos::View<rtype **> face_weights;
    Kokkos::View<rtype **[2][N_CONSERVATIVE]> face_solution;
    BoundaryData boundaries;
    Kokkos::View<rtype *[N_CONSERVATIVE]> W_cells;
    Kokkos::View<rtype *[2]> frozen;  // (cell, [gamma, e0])
    Kokkos::View<rtype **[2][2]> face_thermo;
    Kokkos::View<rtype **, Kokkos::LayoutStride> cell_thermo;
    Kokkos::View<rtype *[N_CONSERVATIVE]> face_flux;
    Kokkos::View<rtype *> face_energy_1;
    Kokkos::View<rtype **> face_mdot;
    rtype low_mach_cutoff;
    Kokkos::View<rtype *> upwind;  // hybrid flux: upwind fraction of each cell, else empty
    Kokkos::View<rtype ***> quad_normals;  // curved meshes: (face, q, dim) unit normals; else empty

    KOKKOS_INLINE_FUNCTION
    void operator()(const uint32_t i_face) const {
        constexpr uint8_t E = N_DIM + 1;
        uint8_t n_quad;
        if constexpr (N_DIM == 2) {
            n_quad = static_cast<uint8_t>(quad_weights.extent(0));
        } else {
            n_quad = static_cast<uint8_t>(face_weights.extent(1));
        }
        const int32_t c0 = cells_of_face(i_face, 0);
        const int32_t c1 = cells_of_face(i_face, 1);
        const riemann::SideThermo th_0{frozen(c0, 0), frozen(c0, 1)};
        const riemann::SideThermo th_1 = c1 >= 0 ? riemann::SideThermo{frozen(c1, 0), frozen(c1, 1)} : th_0;
        rtype n_unit[N_DIM];
        rtype n_vec[N_DIM];
        FOR_I_DIM n_vec[i] = normals(i_face, i);
        unit<N_DIM>(n_vec, n_unit);
        const bool hybrid = upwind.extent(0) > 0;
        const rtype phi = hybrid ? face_upwind_fraction(upwind, boundaries, c0, c1, i_face) : 1.0_r;

        rtype flux[N_CONSERVATIVE] = {};
        rtype energy_1 = 0.0_r;
        for (uint8_t i_quad = 0; i_quad < n_quad; i_quad++) {
            rtype n_q[N_DIM];
            const rtype * n_point = quad_normal(quad_normals, i_face, i_quad, n_unit, n_q);
            rtype w_q;
            if constexpr (N_DIM == 2) {
                w_q = face_weights.extent(0) ? face_weights(i_face, i_quad) : quad_weights(i_quad);
            } else {
                w_q = face_weights(i_face, i_quad);
                if (w_q == 0.0_r) {
                    face_mdot(i_face, i_quad) = 0.0_r;
                    continue;
                }
            }
            rtype W_l[N_CONSERVATIVE], W_r[N_CONSERVATIVE];
            FOR_I_CONSERVATIVE W_l[i] = face_solution(i_face, i_quad, 0, i);
            if (c1 >= 0) {
                FOR_I_CONSERVATIVE W_r[i] = face_solution(i_face, i_quad, 1, i);
            } else {
                const rtype th_i[2] = {th_0.gamma, th_0.e0};
                rtype th_g[2];
                boundaries.exterior_mixture(i_face, i_quad, n_quad, W_l, th_i, n_point, W_cells, face_solution,
                                            face_thermo, cell_thermo, W_r, th_g);
            }
            // Each side's central flux with its own frozen thermodynamics on both sides
            rtype K_0[N_CONSERVATIVE] = {}, K_1[N_CONSERVATIVE] = {};
            if (hybrid) {
                riemann::KEEP::calc_flux(K_0, n_point, W_l, W_r, th_0, th_0);
                riemann::KEEP::calc_flux(K_1, n_point, W_l, W_r, th_1, th_1);
            }
            if (low_mach_cutoff < 1.0_r) {
                if (c1 >= 0) {
                    low_mach_correction(W_l, W_r, th_0.gamma, th_1.gamma, low_mach_cutoff);
                } else if (boundaries.bcs(boundaries.face_bc(i_face)).is_characteristic()) {
                    low_mach_correction(W_l, W_r, th_0.gamma, th_0.gamma, low_mach_cutoff);
                }
            }
            rtype F_0[N_CONSERVATIVE], F_1[N_CONSERVATIVE];
            T_riemann_solver::calc_flux(F_0, n_point, W_l, W_r, th_0, riemann::SideThermo(th_0, th_1));
            if (c1 >= 0) {
                T_riemann_solver::calc_flux(F_1, n_point, W_l, W_r, riemann::SideThermo(th_1, th_0), th_1);
            } else {
                FOR_I_CONSERVATIVE F_1[i] = F_0[i];
            }
            if (hybrid) {
                FOR_I_CONSERVATIVE {
                    F_0[i] = K_0[i] + phi * (F_0[i] - K_0[i]);
                    F_1[i] = K_1[i] + phi * (F_1[i] - K_1[i]);
                }
            }
            const rtype mdot = 0.5_r * (F_0[0] + F_1[0]);
            face_mdot(i_face, i_quad) = mdot;
            flux[0] += w_q * mdot;
            FOR_I_DIM flux[1 + i] += w_q * 0.5_r * (F_0[1 + i] + F_1[1 + i]);
            flux[E] += w_q * F_0[E];
            energy_1 += w_q * F_1[E];
        }

        const rtype scale = 0.5_r * face_area(i_face);
        FOR_I_CONSERVATIVE face_flux(i_face, i) = -scale * flux[i];
        face_energy_1(i_face) = -scale * energy_1;
    }
};

/**
 * @brief FaceFluxSumFunctor for double flux: the energy of a face's side-1
 *        cell comes from face_energy_1.
 */
struct DoubleFluxSumFunctor {
    Kokkos::View<uint32_t *> offsets_faces_of_cell;
    Kokkos::View<uint32_t *> faces_of_cell;
    Kokkos::View<int32_t *[2]> cells_of_face;
    Kokkos::View<rtype *[N_CONSERVATIVE]> face_flux;
    Kokkos::View<rtype *> face_energy_1;
    Kokkos::View<rtype *[N_CONSERVATIVE]> rhs;

    KOKKOS_INLINE_FUNCTION
    void operator()(const uint32_t i_cell) const {
        constexpr uint8_t E = N_DIM + 1;
        rtype sum[N_CONSERVATIVE] = {};
        for (uint32_t k = offsets_faces_of_cell(i_cell); k < offsets_faces_of_cell(i_cell + 1); k++) {
            const uint32_t i_face = faces_of_cell(k);
            if (cells_of_face(i_face, 0) == static_cast<int32_t>(i_cell)) {
                FOR_I_CONSERVATIVE sum[i] += face_flux(i_face, i);
            } else {
                for (uint8_t i = 0; i < E; i++) sum[i] -= face_flux(i_face, i);
                sum[E] -= face_energy_1(i_face);
            }
        }
        FOR_I_CONSERVATIVE rhs(i_cell, i) = sum[i];
    }
};

/**
 * @brief Species fluxes by mass-flux upwinding (Larrouturou 1991),
 *        F_k = max(mdot, 0) Y_k^L + min(mdot, 0) Y_k^R, split by the side
 *        each part comes from: each cell writes, for each of its faces, the
 *        mass of each species leaving it through that face,
 *        slot(face, side, k) = A/2 sum_q w_q max(mdot_out, 0) Y_k(q), with
 *        Y_k from its own reconstruction (the evaluator Eval, see
 *        ScalarFaceValues); boundary faces also get the inflow from the
 *        exterior state in slot(face, 1, k). No two threads write the same slot.
 *
 * With face mass fractions in [0, 1] summing to one, the species fluxes sum
 * to the mass flux and keep rho Y_k non-negative under the flow's CFL limit.
 */
template <typename Eval>
struct SpeciesSlotFunctor {
    static constexpr uint8_t MAX_Q = teno::MAX_FACE_QUAD;

    Kokkos::View<uint32_t *> offsets_faces_of_cell;
    Kokkos::View<uint32_t *> faces_of_cell;
    Kokkos::View<int32_t *[2]> cells_of_face;
    Kokkos::View<rtype *> face_area;
    Kokkos::View<rtype *> quad_weights;   // 2D
    Kokkos::View<rtype **> face_weights;  // (face, q): 3D; 2D axisymmetric (see MixtureFluxFunctor); else empty
    Kokkos::View<rtype **> face_mdot;
    Eval values;
    BoundaryData boundaries;
    Kokkos::View<rtype ***, Kokkos::LayoutRight> slots;  // (face, side, k)
    uint32_t n_species;

    KOKKOS_INLINE_FUNCTION
    uint8_t n_quad() const {
        if constexpr (N_DIM == 2) {
            return static_cast<uint8_t>(quad_weights.extent(0));
        } else {
            return static_cast<uint8_t>(face_weights.extent(1));
        }
    }

    /**
     * @brief Mass fraction k beyond boundary face f (local face k_face of
     *        cell c, whose own values there are interior) at each face point.
     */
    KOKKOS_INLINE_FUNCTION
    void exterior_Y(const uint32_t f, const uint32_t k, const rtype * interior, rtype * out) const {
        const uint8_t nq = n_quad();
        const int32_t i_bc = boundaries.face_bc(f);
        const BoundaryType type = boundaries.bcs(i_bc).type;
        const int32_t image_face = boundaries.face_image_face(f);
        const int32_t image = boundaries.face_image(f);
        if (type == BoundaryType::NSCBC_INLET) {
            for (uint8_t q = 0; q < nq; q++) out[q] = boundaries.bc_Y(i_bc, k);
            return;
        }
        if (image_face >= 0) {
            uint8_t k_image = 0;
            while (faces_of_cell(offsets_faces_of_cell(image) + k_image) != static_cast<uint32_t>(image_face)) k_image++;
            rtype image_values[MAX_Q];
            values.face_values(image, k, k_image, image_values);
            for (uint8_t q = 0; q < nq; q++) {
                uint8_t q_image;
                if constexpr (N_DIM == 2) {
                    q_image = boundaries.face_image_flip(f) ? nq - 1 - q : q;
                } else {
                    q_image = boundaries.face_image_quad(f, q);
                }
                out[q] = image_values[q_image];
            }
            return;
        }
        for (uint8_t q = 0; q < nq; q++) {
            if (image >= 0) {
                out[q] = values.scalars(image, k);
            } else if (type == BoundaryType::UPT) {
                out[q] = boundaries.bc_Y(i_bc, k);
            } else {
                out[q] = interior[q];
            }
        }
    }

    KOKKOS_INLINE_FUNCTION
    void operator()(const uint32_t c) const {
        const uint32_t begin = offsets_faces_of_cell(c);
        const uint8_t n_faces = static_cast<uint8_t>(offsets_faces_of_cell(c + 1) - begin);
        const uint8_t nq = n_quad();
        // A/2 w_q max(+-mdot_out, 0) at each point of each face
        rtype w_out[teno::MAX_FACES][MAX_Q], w_in[teno::MAX_FACES][MAX_Q];
        for (uint8_t i = 0; i < n_faces; i++) {
            const uint32_t f = faces_of_cell(begin + i);
            const bool side_0 = cells_of_face(f, 0) == static_cast<int32_t>(c);
            const rtype scale = 0.5_r * face_area(f);
            for (uint8_t q = 0; q < nq; q++) {
                rtype w_q;
                if constexpr (N_DIM == 2) {
                    w_q = face_weights.extent(0) ? face_weights(f, q) : quad_weights(q);
                } else {
                    w_q = face_weights(f, q);
                }
                const rtype m_out = side_0 ? face_mdot(f, q) : -face_mdot(f, q);
                w_out[i][q] = scale * w_q * Kokkos::fmax(m_out, 0.0_r);
                w_in[i][q] = scale * w_q * Kokkos::fmax(-m_out, 0.0_r);
            }
        }
        for (uint32_t k = 0; k < n_species; k++) {
            FacePointValues Y[teno::MAX_FACES];
            values.cell_values(c, k, Y);
            for (uint8_t i = 0; i < n_faces; i++) {
                const uint32_t f = faces_of_cell(begin + i);
                const uint8_t side = cells_of_face(f, 0) == static_cast<int32_t>(c) ? 0 : 1;
                rtype out = 0.0_r;
                for (uint8_t q = 0; q < nq; q++) out += w_out[i][q] * Y[i][q];
                slots(f, side, k) = out;
                if (cells_of_face(f, 1) < 0) {
                    rtype Y_ext[MAX_Q];
                    exterior_Y(f, k, Y[i], Y_ext);
                    rtype in = 0.0_r;
                    for (uint8_t q = 0; q < nq; q++) in += w_in[i][q] * Y_ext[q];
                    slots(f, 1, k) = in;
                }
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

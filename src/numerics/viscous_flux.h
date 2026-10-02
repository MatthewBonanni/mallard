/**
 * @file viscous_flux.h
 * @author Matthew Bonanni (mbonanni001@gmail.com)
 * @brief Viscous (diffusive) flux functor for the Navier-Stokes equations.
 * @version 0.1
 * @date 2026-10-02
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 *
 */

#ifndef VISCOUS_FLUX_H
#define VISCOUS_FLUX_H

#include <Kokkos_Core.hpp>

#include "common.h"
#include "boundary.h"
#include "physics.h"

/**
 * @brief Integrates the viscous stress and heat flux over every face (one-point
 *        rule) and scatters them to the adjacent cells.
 *
 * Face gradients of velocity and temperature average the two cell gradients
 * and correct them along the face normal so that their component along the
 * line between the cell centroids equals the direct difference, which
 * suppresses odd-even decoupling and stays accurate on non-orthogonal
 * (e.g. triangular) meshes where that line is not aligned with the normal. Walls
 * use a one-sided difference between the cell centroid and the wall with the
 * wall velocity and, for isothermal walls, the wall temperature; adiabatic
 * walls carry no heat flux and heat-flux walls carry the prescribed one.
 * Symmetry faces carry no shear stress and no heat flux; transmissive and
 * outflow faces use zero normal derivatives.
 */
struct ViscousFluxFunctor {
    Kokkos::View<rtype *[N_DIM]> normals;
    Kokkos::View<rtype *> face_area;
    Kokkos::View<rtype *[N_DIM]> face_coords;
    Kokkos::View<rtype *[N_DIM]> cell_coords;
    Kokkos::View<int32_t *[2]> cells_of_face;
    Kokkos::View<rtype *[N_CONSERVATIVE]> W;
    Kokkos::View<rtype *[N_CONSERVATIVE][N_DIM]> gradients;
    BoundaryData boundaries;
    Kokkos::View<rtype *[N_CONSERVATIVE]> rhs;
    Euler physics;

    /**
     * @brief Velocity and temperature of cell c, and their gradients
     *        [u, v, T][x, y], from W = [rho, u, v, p] and its gradient.
     */
    KOKKOS_INLINE_FUNCTION
    void cell_state(const int32_t c, rtype * q, rtype g[3][N_DIM]) const {
        const rtype rho = W(c, 0), p = W(c, 3);
        q[0] = W(c, 1);
        q[1] = W(c, 2);
        q[2] = p / (rho * physics.R);
        FOR_I_DIM {
            g[0][i] = gradients(c, 1, i);
            g[1][i] = gradients(c, 2, i);
            g[2][i] = (gradients(c, 3, i) - physics.R * q[2] * gradients(c, 0, i)) / (rho * physics.R);
        }
    }

    KOKKOS_INLINE_FUNCTION
    void operator()(const uint32_t i_face) const {
        const int32_t c0 = cells_of_face(i_face, 0);
        const int32_t c1 = cells_of_face(i_face, 1);
        rtype n[N_DIM];
        const rtype n_vec[N_DIM] = {normals(i_face, 0), normals(i_face, 1)};
        unit<N_DIM>(n_vec, n);

        rtype q0[3], g0[3][N_DIM];
        cell_state(c0, q0, g0);
        rtype q_f[3], g_f[3][N_DIM];
        bool heat_flux_given = false;
        rtype heat_flux = 0.0;   // Into the domain
        bool symmetry = false;

        if (c1 >= 0) {
            rtype q1[3], g1[3][N_DIM];
            cell_state(c1, q1, g1);
            rtype d[N_DIM];
            FOR_I_DIM d[i] = cell_coords(c1, i) - cell_coords(c0, i);
            const rtype d_n = d[0] * n[0] + d[1] * n[1];
            for (uint8_t k = 0; k < 3; k++) {
                q_f[k] = 0.5 * (q0[k] + q1[k]);
                FOR_I_DIM g_f[k][i] = 0.5 * (g0[k][i] + g1[k][i]);
                const rtype correction = ((q1[k] - q0[k]) - (g_f[k][0] * d[0] + g_f[k][1] * d[1])) / d_n;
                FOR_I_DIM g_f[k][i] += correction * n[i];
            }
        } else {
            const BoundaryCondition & bc = boundaries.bcs(boundaries.face_bc(i_face));
            for (uint8_t k = 0; k < 3; k++) {
                q_f[k] = q0[k];
                FOR_I_DIM g_f[k][i] = g0[k][i];
            }
            if (bc.is_wall()) {
                rtype dn = 0.0;
                FOR_I_DIM dn += (face_coords(i_face, i) - cell_coords(c0, i)) * n[i];
                q_f[0] = bc.data[1];
                q_f[1] = bc.data[2];
                const uint8_t n_set = (bc.type == BoundaryType::WALL_ISOTHERMAL) ? 3 : 2;
                if (bc.type == BoundaryType::WALL_ISOTHERMAL) q_f[2] = bc.data[0];
                for (uint8_t k = 0; k < n_set; k++) {
                    const rtype correction = (q_f[k] - q0[k]) / dn - (g_f[k][0] * n[0] + g_f[k][1] * n[1]);
                    FOR_I_DIM g_f[k][i] += correction * n[i];
                }
                if (bc.type != BoundaryType::WALL_ISOTHERMAL) {
                    heat_flux_given = true;
                    heat_flux = (bc.type == BoundaryType::WALL_HEAT_FLUX) ? bc.data[3] : 0.0;
                }
            } else if (bc.type == BoundaryType::SYMMETRY) {
                symmetry = true;
            } else if (bc.type == BoundaryType::EXTRAPOLATION || bc.type == BoundaryType::P_OUT ||
                       bc.type == BoundaryType::P_OUT_AVERAGE || bc.type == BoundaryType::FARFIELD) {
                // Zero normal derivatives across transmissive and outflow boundaries
                for (uint8_t k = 0; k < 3; k++) {
                    const rtype g_n = g_f[k][0] * n[0] + g_f[k][1] * n[1];
                    FOR_I_DIM g_f[k][i] -= g_n * n[i];
                }
            }
        }

        const rtype mu = physics.viscosity(q_f[2]);
        const rtype kappa = physics.conductivity(mu);
        const rtype div = g_f[0][0] + g_f[1][1];
        const rtype txx = mu * (2.0 * g_f[0][0] - 2.0 / 3.0 * div);
        const rtype tyy = mu * (2.0 * g_f[1][1] - 2.0 / 3.0 * div);
        const rtype txy = mu * (g_f[0][1] + g_f[1][0]);
        rtype tau_n[N_DIM] = {txx * n[0] + txy * n[1], txy * n[0] + tyy * n[1]};
        rtype q_n = kappa * (g_f[2][0] * n[0] + g_f[2][1] * n[1]);
        if (symmetry) {
            // Keep only the normal stress; the normal velocity vanishes on the plane,
            // so the normal stress does no work
            const rtype tau_nn = tau_n[0] * n[0] + tau_n[1] * n[1];
            FOR_I_DIM tau_n[i] = tau_nn * n[i];
            const rtype u_n = q_f[0] * n[0] + q_f[1] * n[1];
            q_f[0] -= u_n * n[0];
            q_f[1] -= u_n * n[1];
            q_n = 0.0;
        }
        if (heat_flux_given) {
            q_n = heat_flux;
        }

        rtype flux[N_CONSERVATIVE];
        flux[0] = 0.0;
        flux[1] = tau_n[0];
        flux[2] = tau_n[1];
        flux[3] = q_f[0] * tau_n[0] + q_f[1] * tau_n[1] + q_n;

        const rtype A = face_area(i_face);
        FOR_I_CONSERVATIVE {
            Kokkos::atomic_add(&rhs(c0, i), A * flux[i]);
            if (c1 >= 0) Kokkos::atomic_add(&rhs(c1, i), -A * flux[i]);
        }
    }
};

#endif // VISCOUS_FLUX_H

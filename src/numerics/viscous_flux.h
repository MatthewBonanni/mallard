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
 * @brief Viscous traction tau . n of a Newtonian fluid under Stokes' hypothesis,
 *        tau = mu (grad u + grad u^T - 2/3 div u I), with g[k][i] = d u_k / d x_i.
 *        Axisymmetric flows add the hoop part u_r / r of the divergence.
 */
KOKKOS_INLINE_FUNCTION
void viscous_traction(const rtype mu, const rtype g[][N_DIM], const rtype * n, rtype * tau_n,
                      const rtype hoop_divergence = 0.0_r) {
    rtype div = 0.0;
    if constexpr (N_DIM == 2) {
        div = g[0][0] + g[1][1];
    } else {
        div = g[0][0] + g[1][1] + g[2][2];
    }
    if (hoop_divergence != 0.0_r) div += hoop_divergence;
    rtype tau[N_DIM][N_DIM];
    FOR_I_DIM {
        for (uint8_t j = 0; j < N_DIM; j++) tau[i][j] = mu * (g[i][j] + g[j][i]);
        tau[i][i] = mu * (2.0_r * g[i][i] - 2.0_r / 3.0_r * div);
    }
    FOR_I_DIM tau_n[i] = dot<N_DIM>(tau[i], n);
}

/**
 * @brief u_r / r at radius r for velocity u and gradient g (g[k][i] = d u_k /
 *        d x_i), with r = y: its limit d u_r / d r on the axis.
 */
KOKKOS_INLINE_FUNCTION
rtype hoop_divergence(const rtype r, const rtype * u, const rtype g[][N_DIM]) {
    return r > 0.0_r ? u[1] / r : g[1][1];
}

/**
 * @brief Axisymmetric meshes: jump from cell c0 to c1 of the cells' r-weighted
 *        variance along d (Mesh::cell_covariance), divided by |d|^2.
 */
KOKKOS_INLINE_FUNCTION
rtype variance_jump(const Kokkos::View<rtype *[3]> & covariance, const int32_t c0, const int32_t c1,
                    const rtype * d) {
    if constexpr (N_DIM == 2) {
        auto var = [&](const int32_t c) {
            return covariance(c, 0) * d[0] * d[0] + 2.0_r * covariance(c, 1) * d[0] * d[1] +
                   covariance(c, 2) * d[1] * d[1];
        };
        return (var(c1) - var(c0)) / dot<N_DIM>(d, d);
    } else {
        (void)covariance;
        (void)c0;
        (void)c1;
        (void)d;
        return 0.0_r;
    }
}

/**
 * @brief Part of the jump of two cell averages that comes from the cells'
 *        different spreads (variance jump dvar along d, see variance_jump): an
 *        average exceeds the value at the center by q_ss var / 2, with the
 *        second derivative q_ss along d from the cell gradients g0, g1.
 */
KOKKOS_INLINE_FUNCTION
rtype mean_offset(const rtype * g0, const rtype * g1, const rtype * d, const rtype dvar) {
    rtype dg[N_DIM];
    FOR_I_DIM dg[i] = g1[i] - g0[i];
    return 0.5_r * dot<N_DIM>(dg, d) / dot<N_DIM>(d, d) * dvar;
}

/**
 * @brief Fraction along d (from the center of cell c0 to that of its
 *        neighbor) of the projection of face f's center.
 */
KOKKOS_INLINE_FUNCTION
rtype face_fraction(const Kokkos::View<rtype *[N_DIM]> & face_coords, const Kokkos::View<rtype *[N_DIM]> & cell_coords,
                    const uint32_t f, const int32_t c0, const rtype * d) {
    rtype x_f[N_DIM];
    FOR_I_DIM x_f[i] = face_coords(f, i) - cell_coords(c0, i);
    return dot<N_DIM>(x_f, d) / dot<N_DIM>(d, d);
}

/**
 * @brief Integrates the viscous stress and heat flux over every face (one-point
 *        rule) and adds them to face_flux (see ConvectiveFluxFunctor).
 *
 * Face gradients of velocity and temperature average the two cell gradients
 * and correct them along the face normal so that their component along the
 * line between the cell centroids equals the direct difference, which
 * suppresses odd-even decoupling and stays accurate on non-orthogonal
 * (e.g. triangular) meshes where that line is not aligned with the normal. Walls
 * use a one-sided difference between the cell centroid and the wall with the
 * wall velocity and, for isothermal walls, the wall temperature; adiabatic
 * walls carry no heat flux and heat-flux walls carry the prescribed one.
 * Symmetry faces carry no shear stress and no heat flux. Transmissive faces
 * take the values and gradients of their image face (see
 * BoundaryData::exterior_W), which is what an interior face sees for a flow
 * that does not vary normal to the boundary; using the boundary cell's own
 * gradient instead is off by O(h) on triangles, whose centroids are offset
 * along the face. Transmissive faces without an image and outflow faces use
 * zero normal derivatives.
 */
struct ViscousFluxFunctor {
    Kokkos::View<rtype *[N_DIM]> normals;
    Kokkos::View<rtype *> face_area;
    Kokkos::View<rtype *[N_DIM]> face_coords;
    Kokkos::View<rtype *[N_DIM]> cell_coords;
    Kokkos::View<int32_t *[2]> cells_of_face;
    Kokkos::View<rtype *[N_DIM]> shifts;
    Kokkos::View<uint8_t *> face_shift;
    Kokkos::View<rtype *[N_CONSERVATIVE]> W;
    Kokkos::View<rtype *[N_CONSERVATIVE][N_DIM]> gradients;
    BoundaryData boundaries;
    Kokkos::View<rtype *[N_CONSERVATIVE]> face_flux;
    Euler physics;
    bool axisymmetric = false;            // face_area is then the revolved area
    Kokkos::View<rtype *[3]> covariance;  // axisymmetric: Mesh::cell_covariance

    static constexpr uint8_t NQ = N_DIM + 1;  // [u, T]

    /**
     * @brief Velocity and temperature q = [u, T] of cell c, and their
     *        gradients g[k][i] = d q_k / d x_i, from W = [rho, u, p] and its
     *        gradient.
     */
    KOKKOS_INLINE_FUNCTION
    void cell_state(const int32_t c, rtype * q, rtype g[NQ][N_DIM]) const {
        const rtype rho = W(c, 0), p = W(c, N_DIM + 1);
        for (uint8_t k = 0; k < N_DIM; k++) q[k] = W(c, 1 + k);
        q[N_DIM] = p / (rho * physics.R);
        FOR_I_DIM {
            for (uint8_t k = 0; k < N_DIM; k++) g[k][i] = gradients(c, 1 + k, i);
            g[N_DIM][i] = (gradients(c, N_DIM + 1, i) - physics.R * q[N_DIM] * gradients(c, 0, i)) / (rho * physics.R);
        }
    }

    /**
     * @brief Values and gradients q_f, g_f on interior face i_face.
     */
    KOKKOS_INLINE_FUNCTION
    void interior_face(const uint32_t i_face, rtype * q_f, rtype g_f[NQ][N_DIM]) const {
        const int32_t c0 = cells_of_face(i_face, 0);
        const int32_t c1 = cells_of_face(i_face, 1);
        rtype n[N_DIM];
        rtype n_vec[N_DIM];
        FOR_I_DIM n_vec[i] = normals(i_face, i);
        unit<N_DIM>(n_vec, n);
        rtype q0[NQ], g0[NQ][N_DIM], q1[NQ], g1[NQ][N_DIM];
        cell_state(c0, q0, g0);
        cell_state(c1, q1, g1);
        rtype d[N_DIM];
        const uint8_t s = face_shift(i_face);
        FOR_I_DIM d[i] = (cell_coords(c1, i) + shifts(s, i)) - cell_coords(c0, i);
        const rtype d_n = dot<N_DIM>(d, n);
        // Axisymmetric cell centers are not symmetric about the faces: interpolate
        // to the face, at fraction t along the line between them
        const rtype t = axisymmetric ? face_fraction(face_coords, cell_coords, i_face, c0, d) : 0.5_r;
        const rtype dvar = axisymmetric ? variance_jump(covariance, c0, c1, d) : 0.0_r;
        for (uint8_t k = 0; k < NQ; k++) {
            q_f[k] = 0.5_r * (q0[k] + q1[k]);
            FOR_I_DIM g_f[k][i] = 0.5_r * (g0[k][i] + g1[k][i]);
            rtype dq = q1[k] - q0[k];
            if (axisymmetric) dq -= mean_offset(g0[k], g1[k], d, dvar);
            const rtype correction = (dq - dot<N_DIM>(g_f[k], d)) / d_n;
            if (axisymmetric) {
                q_f[k] = q0[k] + t * (q1[k] - q0[k]);
                FOR_I_DIM g_f[k][i] = g0[k][i] + t * (g1[k][i] - g0[k][i]);
            }
            FOR_I_DIM g_f[k][i] += correction * n[i];
        }
    }


    KOKKOS_INLINE_FUNCTION
    void operator()(const uint32_t i_face) const {
        const int32_t c0 = cells_of_face(i_face, 0);
        const int32_t c1 = cells_of_face(i_face, 1);
        rtype n[N_DIM];
        rtype n_vec[N_DIM];
        FOR_I_DIM n_vec[i] = normals(i_face, i);
        unit<N_DIM>(n_vec, n);

        rtype q0[NQ], g0[NQ][N_DIM];
        cell_state(c0, q0, g0);
        rtype q_f[NQ], g_f[NQ][N_DIM];
        bool heat_flux_given = false;
        rtype heat_flux = 0.0;   // Into the domain
        bool symmetry = false;

        if (c1 >= 0) {
            interior_face(i_face, q_f, g_f);
        } else {
            const BoundaryCondition & bc = boundaries.bcs(boundaries.face_bc(i_face));
            for (uint8_t k = 0; k < NQ; k++) {
                q_f[k] = q0[k];
                FOR_I_DIM g_f[k][i] = g0[k][i];
            }
            if (bc.is_wall()) {
                rtype dn = 0.0;
                FOR_I_DIM dn += (face_coords(i_face, i) - cell_coords(c0, i)) * n[i];
                for (uint8_t k = 0; k < N_DIM; k++) q_f[k] = bc.data[1 + k];
                const uint8_t n_set = (bc.type == BoundaryType::WALL_ISOTHERMAL) ? NQ : N_DIM;
                if (bc.type == BoundaryType::WALL_ISOTHERMAL) q_f[N_DIM] = bc.data[0];
                for (uint8_t k = 0; k < n_set; k++) {
                    const rtype correction = (q_f[k] - q0[k]) / dn - dot<N_DIM>(g_f[k], n);
                    FOR_I_DIM g_f[k][i] += correction * n[i];
                }
                if (bc.type != BoundaryType::WALL_ISOTHERMAL) {
                    heat_flux_given = true;
                    heat_flux = (bc.type == BoundaryType::WALL_HEAT_FLUX) ? bc.data[N_DIM + 1] : 0.0_r;
                }
            } else if (bc.type == BoundaryType::SYMMETRY) {
                symmetry = true;
            } else if (boundaries.face_image_face(i_face) >= 0) {
                interior_face(boundaries.face_image_face(i_face), q_f, g_f);
            } else if (bc.type == BoundaryType::EXTRAPOLATION || bc.type == BoundaryType::P_OUT ||
                       bc.type == BoundaryType::P_OUT_AVERAGE || bc.type == BoundaryType::FARFIELD ||
                       bc.type == BoundaryType::PARTITION) {
                // Zero normal derivatives across transmissive and outflow boundaries
                for (uint8_t k = 0; k < NQ; k++) {
                    const rtype g_n = dot<N_DIM>(g_f[k], n);
                    FOR_I_DIM g_f[k][i] -= g_n * n[i];
                }
            }
        }

        const rtype mu = physics.viscosity(q_f[N_DIM]);
        const rtype kappa = physics.conductivity(mu);
        rtype tau_n[N_DIM];
        viscous_traction(mu, g_f, n, tau_n, axisymmetric ? hoop_divergence(face_coords(i_face, 1), q_f, g_f) : 0.0_r);
        rtype q_n = kappa * dot<N_DIM>(g_f[N_DIM], n);
        if (symmetry) {
            // Keep only the normal stress; the normal velocity vanishes on the plane,
            // so the normal stress does no work
            const rtype tau_nn = dot<N_DIM>(tau_n, n);
            FOR_I_DIM tau_n[i] = tau_nn * n[i];
            const rtype u_n = dot<N_DIM>(q_f, n);
            FOR_I_DIM q_f[i] -= u_n * n[i];
            q_n = 0.0;
        }
        if (heat_flux_given) {
            q_n = heat_flux;
        }

        rtype flux[N_CONSERVATIVE];
        flux[0] = 0.0;
        FOR_I_DIM flux[1 + i] = tau_n[i];
        flux[N_DIM + 1] = dot<N_DIM>(q_f, tau_n) + q_n;

        const rtype A = face_area(i_face);
        FOR_I_CONSERVATIVE face_flux(i_face, i) += A * flux[i];
    }
};

#endif // VISCOUS_FLUX_H

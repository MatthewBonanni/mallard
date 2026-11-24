/**
 * @file mixture_point_gradients.h
 * @author Matthew Bonanni (mbonanni001@gmail.com)
 * @brief Temperature and mole fractions at the face points of a gas mixture
 *        from the cells' TENO polynomials (viscous_gradients = "reconstruction").
 * @version 0.1
 * @date 2026-11-23
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 *
 */

#ifndef MIXTURE_POINT_GRADIENTS_H
#define MIXTURE_POINT_GRADIENTS_H

#include <Kokkos_Core.hpp>

#include "face_point_gradients.h"
#include "teno_scalars.h"
#include "thermo.h"

/**
 * @brief Per cell with usable point_gradients (TENO wrote W = [rho, u, p]
 *        and its gradients there, see FacePointGradients::store_primitive):
 *        the mass fractions' central polynomials (TENOScalarValues' large
 *        coefficients, scaled by the cell's theta as its face values) give
 *        Y_k and grad Y_k at each face point, and with them
 *        R = R_u sum_k Y_k / W_k, T = p / (rho R) and
 *        grad T = T (grad p / p - grad rho / rho - grad R / R) in place of p,
 *        X_k = Y_k W / W_k and dX_k/dn along the face's unit normal.
 */
template <uint8_t DEG>
struct MixturePointGradientsFunctor {
    static constexpr uint8_t NK = teno::n_dof(DEG);
    static constexpr uint8_t NQ = FacePointGradients::NQ;

    TENOScalarValues<DEG> values;
    FacePointGradients point_gradients;
    chemistry::ThermoTable<> thermo;
    Kokkos::View<rtype *[N_DIM]> normals;

    KOKKOS_INLINE_FUNCTION
    void operator()(const uint32_t c) const {
        const uint32_t f_begin = values.offsets_faces_of_cell(c);
        const uint8_t n_faces = values.n_faces(c);
        auto side_of = [&](const uint32_t f) -> uint8_t { return values.cells_of_face(f, 0) == static_cast<int32_t>(c) ? 0 : 1; };
        const uint32_t f0 = values.faces_of_cell(f_begin);
        if (!point_gradients.usable(f0, side_of(f0))) return;
        const uint32_t ns = values.n_species;
        const uint8_t n_quad = values.n_quad();
        rtype factor[DEG + 1];
        factor[0] = 1.0_r;
        for (uint8_t d = 1; d <= DEG; d++) factor[d] = factor[d - 1] / values.scale(c);
        const rtype theta = values.theta.extent(0) ? values.theta(c) : 1.0_r;
        // sum_k Y_k / W_k and its gradient at each point
        rtype sum[teno::MAX_FACES][teno::MAX_FACE_QUAD] = {}, d_sum[teno::MAX_FACES][teno::MAX_FACE_QUAD][N_DIM] = {};
        for (uint32_t k0 = 0; k0 < ns; k0 += N_CONSERVATIVE) {
            const uint32_t n = ns - k0 < N_CONSERVATIVE ? ns - k0 : N_CONSERVATIVE;
            rtype a[N_CONSERVATIVE][NK] = {};
            for (uint32_t i = 0; i < n; i++) values.large_coefficients(c, k0 + i, a[i]);
            for (uint8_t k = 0; k < n_faces; k++) {
                const uint32_t f = values.faces_of_cell(f_begin + k);
                const uint8_t side = side_of(f);
                rtype n_f[N_DIM], n_vec[N_DIM];
                FOR_I_DIM n_vec[i] = normals(f, i);
                unit<N_DIM>(n_vec, n_f);
                const PlanarFace nodes = values.planar_nodes(f);
                for (uint8_t q = 0; q < n_quad; q++) {
                    rtype p[N_DIM], psi[NK], r[N_DIM], dY[N_CONSERVATIVE][N_DIM];
                    if (!values.face_point_basis(f, q, c, nodes, psi)) continue;
                    values.face_point(f, q, c, nodes, p);
                    FOR_I_DIM r[i] = p[i] - values.cell_coords(c, i);
                    FacePointGradients::conservative_gradient<DEG>(
                        r, [&](const uint32_t l, const uint8_t d, const uint8_t i) { return a[i][l] * factor[d]; }, dY);
                    for (uint32_t m = 0; m < n; m++) {
                        rtype dev = 0.0_r;
                        for (uint8_t l = 0; l < NK; l++) dev += a[m][l] * psi[l];
                        const uint32_t s = k0 + m;
                        const rtype Y = values.scalars(c, s) + theta * dev;
                        rtype g[N_DIM];
                        FOR_I_DIM g[i] = theta * dY[m][i];
                        point_gradients.at(f, q, side, FacePointGradients::species(s)) = Y;
                        point_gradients.at(f, q, side, FacePointGradients::species(ns + s)) = dot<N_DIM>(g, n_f);
                        const rtype inv_W = static_cast<rtype>(thermo.inv_W(s));
                        sum[k][q] += Y * inv_W;
                        FOR_I_DIM d_sum[k][q][i] += g[i] * inv_W;
                    }
                }
            }
        }
        for (uint8_t k = 0; k < n_faces; k++) {
            const uint32_t f = values.faces_of_cell(f_begin + k);
            const uint8_t side = side_of(f);
            rtype n_f[N_DIM], n_vec[N_DIM];
            FOR_I_DIM n_vec[i] = normals(f, i);
            unit<N_DIM>(n_vec, n_f);
            const PlanarFace nodes = values.planar_nodes(f);
            for (uint8_t q = 0; q < n_quad; q++) {
                rtype p_q[N_DIM];
                if (!values.face_point(f, q, c, nodes, p_q)) continue;
                rtype v[NQ], g[NQ][N_DIM], d_rho[N_DIM];
                point_gradients.load(f, q, side, v, g);
                const rtype rho = point_gradients.at(f, q, side, FacePointGradients::DENSITY);
                FOR_I_DIM d_rho[i] = point_gradients.at(f, q, side, FacePointGradients::DENSITY + 1 + i);
                const rtype p = v[N_DIM];
                const rtype R = static_cast<rtype>(chemistry::GAS_CONSTANT) * sum[k][q];
                const rtype T = p / (rho * R);
                v[N_DIM] = T;
                FOR_I_DIM g[N_DIM][i] = T * (g[N_DIM][i] / p - d_rho[i] / rho - d_sum[k][q][i] / sum[k][q]);
                point_gradients.store(f, q, side, v, g);
                // X_k = Y_k W / W_k with W = 1 / sum
                const rtype W = 1.0_r / sum[k][q];
                const rtype dW_dn = -W * W * dot<N_DIM>(d_sum[k][q], n_f);
                for (uint32_t s = 0; s < ns; s++) {
                    rtype & X = point_gradients.at(f, q, side, FacePointGradients::species(s));
                    rtype & dX_dn = point_gradients.at(f, q, side, FacePointGradients::species(ns + s));
                    const rtype inv_W = static_cast<rtype>(thermo.inv_W(s));
                    const rtype Y = X;
                    X = Y * W * inv_W;
                    dX_dn = inv_W * (W * dX_dn + Y * dW_dn);
                }
            }
        }
    }
};

#endif // MIXTURE_POINT_GRADIENTS_H

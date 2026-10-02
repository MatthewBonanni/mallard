/**
 * @file teno.h
 * @author Matthew Bonanni (mbonanni001@gmail.com)
 * @brief Device helpers for the TENO-E reconstruction.
 * @version 0.2
 * @date 2026-10-02
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 *
 */

#ifndef TENO_H
#define TENO_H

#include <Kokkos_Core.hpp>

#include "common.h"

namespace teno {

constexpr uint8_t MAX_DEGREE = 5;
constexpr uint8_t MAX_NK = (MAX_DEGREE + 1) * (MAX_DEGREE + 2) / 2 - 1;  // non-constant dofs
constexpr uint8_t NK_SMALL = 5;                                           // degree-2 dofs
constexpr uint8_t MAX_FACES = 4;
constexpr uint8_t MAX_FACE_QUAD = 4;

/**
 * @brief Number of non-constant 2D monomials of total degree <= r.
 */
KOKKOS_INLINE_FUNCTION
constexpr uint8_t n_dof(const uint8_t r) {
    return (r + 1) * (r + 2) / 2 - 1;
}

/**
 * @brief Exponents (a, b) of the l-th monomial xi^a eta^b, ordered by total
 *        degree: (1,0), (0,1), (2,0), (1,1), (0,2), (3,0), ...
 */
KOKKOS_INLINE_FUNCTION
void exponents(const uint8_t l, uint8_t & a, uint8_t & b) {
    uint8_t d = 1;
    uint8_t first = 0;
    while (l >= first + d + 1) {
        first += d + 1;
        d++;
    }
    a = d - (l - first);
    b = l - first;
}

KOKKOS_INLINE_FUNCTION
rtype ipow(const rtype x, const uint8_t n) {
    rtype result = 1.0;
    for (uint8_t i = 0; i < n; i++) result *= x;
    return result;
}

/**
 * @brief Evaluate all monomials of degree 1..r at (xi, eta).
 */
KOKKOS_INLINE_FUNCTION
void monomials(const uint8_t r, const rtype xi, const rtype eta, rtype * phi) {
    const uint8_t nk = n_dof(r);
    for (uint8_t l = 0; l < nk; l++) {
        uint8_t a, b;
        exponents(l, a, b);
        phi[l] = ipow(xi, a) * ipow(eta, b);
    }
}

/**
 * @brief Left (L) and right (R) eigenvectors of the 2D Euler flux Jacobian
 *        in direction n for conservative variables, at state W = [rho, u, v, p].
 *        Characteristic order: u_n - a, u_n (entropy), u_n + a, u_n (shear).
 */
KOKKOS_INLINE_FUNCTION
void eigenvectors(const rtype * W, const rtype * n, const rtype gamma,
                  rtype L[N_CONSERVATIVE][N_CONSERVATIVE],
                  rtype R[N_CONSERVATIVE][N_CONSERVATIVE]) {
    const rtype u = W[1], v = W[2];
    const rtype a = Kokkos::sqrt(gamma * W[3] / W[0]);
    const rtype q2 = u * u + v * v;
    const rtype H = a * a / (gamma - 1.0) + 0.5 * q2;
    const rtype qn = u * n[0] + v * n[1];
    const rtype qt = -u * n[1] + v * n[0];
    const rtype b1 = (gamma - 1.0) / (a * a);
    const rtype b2 = 0.5 * b1 * q2;

    R[0][0] = 1.0;           R[0][1] = 1.0;      R[0][2] = 1.0;           R[0][3] = 0.0;
    R[1][0] = u - a * n[0];  R[1][1] = u;        R[1][2] = u + a * n[0];  R[1][3] = -n[1];
    R[2][0] = v - a * n[1];  R[2][1] = v;        R[2][2] = v + a * n[1];  R[2][3] = n[0];
    R[3][0] = H - a * qn;    R[3][1] = 0.5 * q2; R[3][2] = H + a * qn;    R[3][3] = qt;

    L[0][0] = 0.5 * (b2 + qn / a); L[0][1] = 0.5 * (-b1 * u - n[0] / a);
    L[0][2] = 0.5 * (-b1 * v - n[1] / a); L[0][3] = 0.5 * b1;
    L[1][0] = 1.0 - b2; L[1][1] = b1 * u; L[1][2] = b1 * v; L[1][3] = -b1;
    L[2][0] = 0.5 * (b2 - qn / a); L[2][1] = 0.5 * (-b1 * u + n[0] / a);
    L[2][2] = 0.5 * (-b1 * v + n[1] / a); L[2][3] = 0.5 * b1;
    L[3][0] = -qt; L[3][1] = -n[1]; L[3][2] = n[0]; L[3][3] = 0.0;
}

/**
 * @brief Adaptive cutoff C_T from the troubled-cell measure sigma
 *        (Liang, Shyy & Fu 2025): 1e-10 near sigma_L, 1e-6 above sigma_U.
 */
KOKKOS_INLINE_FUNCTION
rtype adaptive_CT(const rtype sigma, const rtype sigma_L, const rtype sigma_U) {
    const rtype m = (sigma >= sigma_U) ? 1.0 : Kokkos::fmin(1.0, Kokkos::fmax(0.0, (sigma - sigma_L) / (sigma_U - sigma_L)));
    const rtype g = (1.0 - m) * (1.0 - m) * (1.0 + 2.0 * m);
    const rtype psi = 10.0 - 4.0 * (1.0 - g);
    return Kokkos::pow(10.0, -Kokkos::floor(psi));
}

} // namespace teno

#endif // TENO_H

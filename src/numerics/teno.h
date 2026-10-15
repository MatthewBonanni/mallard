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

/**
 * @brief Number of non-constant N_DIM-variate monomials of total degree <= r.
 */
KOKKOS_INLINE_FUNCTION
constexpr uint8_t n_dof(const uint8_t r) {
    if constexpr (N_DIM == 2) {
        return static_cast<uint8_t>((r + 1) * (r + 2) / 2 - 1);
    } else {
        return static_cast<uint8_t>((r + 1) * (r + 2) * (r + 3) / 6 - 1);
    }
}

constexpr uint8_t MAX_NK = n_dof(MAX_DEGREE);     // non-constant dofs
constexpr uint8_t NK_SMALL = n_dof(2);            // degree-2 dofs
constexpr uint8_t MAX_FACES = (N_DIM == 2) ? 4 : 6;
constexpr uint8_t MAX_FACE_QUAD = (N_DIM == 2) ? 4 : 9;

/**
 * @brief Index of entry (l, m), l <= m, of a symmetric n x n matrix stored as
 *        its upper triangle, row by row.
 */
KOKKOS_INLINE_FUNCTION
constexpr uint16_t upper_index(const uint8_t l, const uint8_t m, const uint8_t n) {
    return static_cast<uint16_t>(l * n - l * (l - 1) / 2 + (m - l));
}

/**
 * @brief log2 of the cells per slice of PackedStencils: 32 on GPUs, so that a
 *        warp reads consecutive words, and 1 on the host, where each cell's
 *        stencil is then contiguous.
 */
constexpr uint8_t SLICE_SHIFT =
    Kokkos::SpaceAccessibility<Kokkos::DefaultExecutionSpace, Kokkos::HostSpace>::accessible ? 0 : 5;

/**
 * @brief Per-cell stencils of different sizes, stored without padding to the
 *        largest one in the mesh. Slot s of a cell's stencil holds the stencil
 *        cell, its mirror boundary face (or -1) and `width` pseudo-inverse
 *        entries. Consecutive cells form slices of 2^shift cells; a slice is
 *        padded to its largest stencil and interleaves its cells' slots.
 */
struct PackedStencils {
    Kokkos::View<uint64_t *> slice_start;  // (slice): first slot of the slice
    Kokkos::View<int32_t *> cells;
    Kokkos::View<int32_t *> faces;
    Kokkos::View<rtype *> pinv;
    uint8_t shift = SLICE_SHIFT;
    uint8_t width = 0;

    /** @brief One cell's stencil, resolved once per cell. */
    struct Row {
        const rtype * pinv_;
        const int32_t * cells_;
        const int32_t * faces_;
        uint8_t shift;

        KOKKOS_INLINE_FUNCTION
        int32_t cell(const uint32_t s) const { return cells_[s << shift]; }

        KOKKOS_INLINE_FUNCTION
        int32_t face(const uint32_t s) const { return faces_[s << shift]; }

        /** @brief Entry l of slot s, for pseudo-inverses of WIDTH entries per slot. */
        template <uint8_t WIDTH>
        KOKKOS_INLINE_FUNCTION
        rtype pinv(const uint32_t s, const uint32_t l) const { return pinv_[(s * WIDTH + l) << shift]; }
    };

    // The arrays are single allocations whose base pointers reach the kernels
    // as parameters: pointers loaded from device memory (e.g. a table of
    // separately allocated chunks) compile to generic loads, which made the
    // reconstruction up to 30% slower
    KOKKOS_INLINE_FUNCTION
    Row row(const uint32_t c) const {
        const uint64_t start = slice_start(c >> shift);
        const uint32_t lane = c & ((1u << shift) - 1);
        return Row{pinv.data() + ((start * width) << shift) + lane, cells.data() + (start << shift) + lane,
                   faces.data() + (start << shift) + lane, shift};
    }
};

/**
 * @brief Exponents (a, b) of the l-th monomial xi^a eta^b, ordered by total
 *        degree: (1,0), (0,1), (2,0), (1,1), (0,2), (3,0), ...
 */
KOKKOS_INLINE_FUNCTION
void exponents(const uint8_t l, uint8_t & a, uint8_t & b) {
    uint8_t d = 1;
    uint8_t first = 0;
    while (l >= first + d + 1) {
        first = static_cast<uint8_t>(first + d + 1);
        d++;
    }
    a = static_cast<uint8_t>(d - (l - first));
    b = static_cast<uint8_t>(l - first);
}

/**
 * @brief Exponents (a, b, c) of the l-th trivariate monomial xi^a eta^b zeta^c,
 *        ordered by total degree, then by decreasing a, then decreasing b:
 *        (1,0,0), (0,1,0), (0,0,1), (2,0,0), (1,1,0), (1,0,1), (0,2,0), ...
 */
KOKKOS_INLINE_FUNCTION
void exponents(const uint8_t l, uint8_t & a, uint8_t & b, uint8_t & c) {
    uint8_t idx = 0;
    for (uint8_t d = 1;; d++) {
        for (int i = d; i >= 0; i--) {
            for (int j = d - i; j >= 0; j--) {
                if (idx == l) {
                    a = static_cast<uint8_t>(i);
                    b = static_cast<uint8_t>(j);
                    c = static_cast<uint8_t>(d - i - j);
                    return;
                }
                idx++;
            }
        }
    }
}

template <typename T> KOKKOS_INLINE_FUNCTION
T ipow(const T x, const uint8_t n) {
    T result = 1;
    for (uint8_t i = 0; i < n; i++) result *= x;
    return result;
}

/**
 * @brief Evaluate all monomials of degree 1..r at (xi, eta).
 */
template <typename T> KOKKOS_INLINE_FUNCTION
void monomials(const uint8_t r, const T xi, const T eta, T * phi) {
    const uint8_t nk = n_dof(r);
    for (uint8_t l = 0; l < nk; l++) {
        uint8_t a, b;
        exponents(l, a, b);
        phi[l] = ipow(xi, a) * ipow(eta, b);
    }
}

/**
 * @brief Derivatives with respect to xi and eta of all monomials of degree
 *        1..r at (xi, eta).
 */
template <typename T> KOKKOS_INLINE_FUNCTION
void monomial_gradients(const uint8_t r, const T xi, const T eta, T * d_xi, T * d_eta) {
    const uint8_t nk = n_dof(r);
    for (uint8_t l = 0; l < nk; l++) {
        uint8_t a, b;
        exponents(l, a, b);
        d_xi[l] = a > 0 ? T(a) * ipow(xi, a - 1) * ipow(eta, b) : T(0);
        d_eta[l] = b > 0 ? T(b) * ipow(xi, a) * ipow(eta, b - 1) : T(0);
    }
}

/**
 * @brief Evaluate all trivariate monomials of degree 1..r at (xi, eta, zeta).
 */
template <typename T> KOKKOS_INLINE_FUNCTION
void monomials(const uint8_t r, const T xi, const T eta, const T zeta, T * phi) {
    uint8_t l = 0;
    for (uint8_t d = 1; d <= r; d++) {
        for (int i = d; i >= 0; i--) {
            for (int j = d - i; j >= 0; j--) {
                phi[l++] = ipow(xi, static_cast<uint8_t>(i)) * ipow(eta, static_cast<uint8_t>(j)) *
                           ipow(zeta, static_cast<uint8_t>(d - i - j));
            }
        }
    }
}

/**
 * @brief Monomials at a point x (N_DIM coordinates, already scaled).
 */
template <typename T> KOKKOS_INLINE_FUNCTION
void monomials(const uint8_t r, const T * x, T * phi) {
    if constexpr (N_DIM == 2) {
        monomials(r, x[0], x[1], phi);
    } else {
        monomials(r, x[0], x[1], x[2], phi);
    }
}

/**
 * @brief Left (L) and right (R) eigenvectors of the Euler flux Jacobian in
 *        direction n for conservative variables, at state W = [rho, u, p].
 *        Characteristic order: u_n - a, u_n (entropy), u_n + a, then one
 *        u_n (shear) wave per tangent of tangent_basis(n).
 */
KOKKOS_INLINE_FUNCTION
void eigenvectors(const rtype * W, const rtype * n, const rtype gamma,
                  rtype L[N_CONSERVATIVE][N_CONSERVATIVE],
                  rtype R[N_CONSERVATIVE][N_CONSERVATIVE]) {
    constexpr uint8_t E = N_DIM + 1;
    const rtype * u = W + 1;
    const rtype a = Kokkos::sqrt(gamma * W[E] / W[0]);
    const rtype q2 = dot<N_DIM>(u, u);
    const rtype H = a * a / (gamma - 1.0_r) + 0.5_r * q2;
    const rtype qn = dot<N_DIM>(u, n);
    rtype t[N_DIM - 1][N_DIM];
    tangent_basis(n, t[0], t[N_DIM - 2]);
    const rtype b1 = (gamma - 1.0_r) / (a * a);
    const rtype b2 = 0.5_r * b1 * q2;

    R[0][0] = 1.0;        R[0][1] = 1.0;      R[0][2] = 1.0;
    R[E][0] = H - a * qn; R[E][1] = 0.5_r * q2; R[E][2] = H + a * qn;
    L[0][0] = 0.5_r * (b2 + qn / a); L[0][E] = 0.5_r * b1;
    L[1][0] = 1.0_r - b2;            L[1][E] = -b1;
    L[2][0] = 0.5_r * (b2 - qn / a); L[2][E] = 0.5_r * b1;
    FOR_I_DIM {
        R[1 + i][0] = u[i] - a * n[i];
        R[1 + i][1] = u[i];
        R[1 + i][2] = u[i] + a * n[i];
        L[0][1 + i] = 0.5_r * (-b1 * u[i] - n[i] / a);
        L[1][1 + i] = b1 * u[i];
        L[2][1 + i] = 0.5_r * (-b1 * u[i] + n[i] / a);
    }
    for (uint8_t k = 0; k < N_DIM - 1; k++) {
        const uint8_t c = 3 + k;
        R[0][c] = 0.0;
        R[E][c] = dot<N_DIM>(u, t[k]);
        L[c][0] = -R[E][c];
        L[c][E] = 0.0;
        FOR_I_DIM {
            R[1 + i][c] = t[k][i];
            L[c][1 + i] = t[k][i];
        }
    }
}

/**
 * @brief Left (L) and right (R) eigenvectors of the Euler equations in
 *        primitive variables W = [rho, u, p] in direction n, for sound speed
 *        a: valid for any equation of state. Same characteristic order as
 *        eigenvectors(): u_n - a, u_n (entropy, here the density at fixed p
 *        and u), u_n + a, then the shear waves.
 */
KOKKOS_INLINE_FUNCTION
void primitive_eigenvectors(const rtype * W, const rtype a, const rtype * n,
                            rtype L[N_CONSERVATIVE][N_CONSERVATIVE],
                            rtype R[N_CONSERVATIVE][N_CONSERVATIVE]) {
    constexpr uint8_t E = N_DIM + 1;
    const rtype rho = W[0];
    rtype t[N_DIM - 1][N_DIM];
    tangent_basis(n, t[0], t[N_DIM - 2]);
    for (uint8_t r = 0; r < N_CONSERVATIVE; r++) {
        for (uint8_t c = 0; c < N_CONSERVATIVE; c++) {
            L[r][c] = 0.0_r;
            R[r][c] = 0.0_r;
        }
    }
    // Acoustic waves r = [1, -+a n / rho, a^2], entropy r = [1, 0, 0], shear r = [0, t, 0]
    R[0][0] = 1.0_r;
    R[0][1] = 1.0_r;
    R[0][2] = 1.0_r;
    R[E][0] = a * a;
    R[E][2] = a * a;
    L[0][E] = 0.5_r / (a * a);
    L[2][E] = 0.5_r / (a * a);
    L[1][0] = 1.0_r;
    L[1][E] = -1.0_r / (a * a);
    FOR_I_DIM {
        R[1 + i][0] = -a * n[i] / rho;
        R[1 + i][2] = a * n[i] / rho;
        L[0][1 + i] = -0.5_r * rho * n[i] / a;
        L[2][1 + i] = 0.5_r * rho * n[i] / a;
    }
    for (uint8_t k = 0; k < N_DIM - 1; k++) {
        FOR_I_DIM {
            R[1 + i][3 + k] = t[k][i];
            L[3 + k][1 + i] = t[k][i];
        }
    }
}

/**
 * @brief Adaptive cutoff C_T from the troubled-cell measure sigma
 *        (Liang, Shyy & Fu 2025): 1e-10 near sigma_L, 1e-6 above sigma_U.
 */
KOKKOS_INLINE_FUNCTION
rtype adaptive_CT(const rtype sigma, const rtype sigma_L, const rtype sigma_U) {
    const rtype m = (sigma >= sigma_U) ? 1.0_r : Kokkos::fmin(1.0_r, Kokkos::fmax(0.0_r, (sigma - sigma_L) / (sigma_U - sigma_L)));
    const rtype g = (1.0_r - m) * (1.0_r - m) * (1.0_r + 2.0_r * m);
    const rtype psi = 10.0_r - 4.0_r * (1.0_r - g);
    return Kokkos::pow(10.0_r, -Kokkos::floor(psi));
}

} // namespace teno

#endif // TENO_H

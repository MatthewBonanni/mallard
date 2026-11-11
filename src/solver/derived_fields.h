/**
 * @file derived_fields.h
 * @brief Derived output fields of a flow state (MACH, Q_CRITERION, LAMBDA2,
 *        SCHLIEREN) from its cell values and gradients, for the in-run
 *        writers and for any path that holds a loaded state.
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 */

#ifndef DERIVED_FIELDS_H
#define DERIVED_FIELDS_H

#include <cstdint>

#include <Kokkos_Core.hpp>

#include "common_typedef.h"

namespace derived {

/** @brief The derived fields, in the order of their columns. */
enum Field : uint8_t { MACH, Q_CRITERION, LAMBDA2, SCHLIEREN, N_FIELDS };

/** @brief Output names of the derived fields, indexed by Field. */
inline constexpr const char * NAMES[N_FIELDS] = {"MACH", "Q_CRITERION", "LAMBDA2", "SCHLIEREN"};

/**
 * @brief A velocity gradient as a 3x3 tensor, a[k][i] = d u_k / d x_i. A 2D
 *        flow has no third velocity component nor z variation, except the
 *        hoop strain a[2][2] = u_r / r of axisymmetric runs.
 */
struct Tensor {
    rtype a[3][3];
};

/**
 * @brief The velocity gradient of cell c from the gradients g of W = (rho, u, p),
 *        with the hoop strain when radius_coords (axisymmetric runs) is not empty.
 */
KOKKOS_INLINE_FUNCTION
Tensor velocity_gradient(const Kokkos::View<rtype *[N_CONSERVATIVE][N_DIM]> & g,
                         const Kokkos::View<rtype *[N_CONSERVATIVE]> & W,
                         const Kokkos::View<rtype *[N_DIM]> & radius_coords, const uint32_t c) {
    Tensor t{};
    for (uint8_t k = 0; k < N_DIM; k++) {
        FOR_I_DIM t.a[k][i] = g(c, 1 + k, i);
    }
    if (radius_coords.extent(0) > 0) t.a[2][2] = W(c, 2) / radius_coords(c, 1);
    return t;
}

/** @brief Q = (|Omega|^2 - |S|^2) / 2 = -a_ki a_ik / 2 of the velocity gradient a. */
KOKKOS_INLINE_FUNCTION
rtype q_criterion(const Tensor & g) {
    rtype gg = 0.0;
    for (int i = 0; i < 3; i++) {
        for (int k = 0; k < 3; k++) gg += g.a[k][i] * g.a[i][k];
    }
    return -0.5_r * gg;
}

/**
 * @brief Eigenvalues of the symmetric 3x3 matrix m (overwritten) by cyclic
 *        Jacobi rotations, in no particular order.
 */
KOKKOS_INLINE_FUNCTION
void symmetric_eigenvalues(rtype m[3][3], rtype ev[3]) {
    constexpr int pairs[3][2] = {{0, 1}, {0, 2}, {1, 2}};
    constexpr rtype eps = Kokkos::epsilon_v<rtype>;
    // Scaled to entries of order 1, so that squares neither underflow nor overflow
    rtype scale = 0.0;
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) scale = Kokkos::fmax(scale, Kokkos::fabs(m[i][j]));
    }
    for (int i = 0; i < 3; i++) ev[i] = 0.0;
    if (!(scale > 0.0_r)) return;
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) m[i][j] /= scale;
    }
    for (int sweep = 0; sweep < 32; sweep++) {
        const rtype off = m[0][1] * m[0][1] + m[0][2] * m[0][2] + m[1][2] * m[1][2];
        const rtype diag = m[0][0] * m[0][0] + m[1][1] * m[1][1] + m[2][2] * m[2][2];
        if (!(off > eps * eps * diag)) break;
        for (const auto & pq : pairs) {
            const int p = pq[0], q = pq[1], r = 3 - p - q;
            const rtype apq = m[p][q];
            if (apq == 0.0_r) continue;
            const rtype theta = (m[q][q] - m[p][p]) / (2.0_r * apq);
            // tan of the rotation angle, the smaller root of t^2 + 2 theta t - 1 = 0
            const rtype t = Kokkos::fabs(theta) > 1.0_r / eps
                                ? 0.5_r / theta
                                : Kokkos::copysign(1.0_r, theta) /
                                      (Kokkos::fabs(theta) + Kokkos::sqrt(theta * theta + 1.0_r));
            const rtype c = 1.0_r / Kokkos::sqrt(t * t + 1.0_r), s = t * c;
            m[p][p] -= t * apq;
            m[q][q] += t * apq;
            m[p][q] = m[q][p] = 0.0_r;
            const rtype arp = m[r][p], arq = m[r][q];
            m[r][p] = m[p][r] = c * arp - s * arq;
            m[r][q] = m[q][r] = s * arp + c * arq;
        }
    }
    for (int i = 0; i < 3; i++) ev[i] = scale * m[i][i];
}

/**
 * @brief lambda2 (Jeong & Hussain 1995): the middle eigenvalue of
 *        S^2 + Omega^2 = (a a + a^T a^T) / 2 of the velocity gradient a,
 *        negative inside a vortex core.
 */
KOKKOS_INLINE_FUNCTION
rtype lambda2(const Tensor & g) {
    rtype m[3][3];
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            rtype sum = 0.0;
            for (int k = 0; k < 3; k++) sum += g.a[i][k] * g.a[k][j] + g.a[k][i] * g.a[j][k];
            m[i][j] = 0.5_r * sum;
        }
    }
    rtype ev[3];
    symmetric_eigenvalues(m, ev);
    return ev[0] + ev[1] + ev[2] - Kokkos::fmax(ev[0], Kokkos::fmax(ev[1], ev[2])) -
           Kokkos::fmin(ev[0], Kokkos::fmin(ev[1], ev[2]));
}

/** @brief Which derived fields are computed, and the output column of each. */
struct Selection {
    int32_t column[N_FIELDS] = {-1, -1, -1, -1};
    uint32_t n_columns = 0;

    /** @brief Compute field f, in the next column. */
    void add(Field f) {
        if (column[f] < 0) column[f] = static_cast<int32_t>(n_columns++);
    }
    /** @brief Whether any field is computed. */
    bool any() const { return n_columns > 0; }
};

/** @brief The cell values a derived field is computed from. */
struct Inputs {
    Kokkos::View<rtype *[N_CONSERVATIVE]> W;                  ///< rho, u, p of each cell
    Kokkos::View<rtype *[N_CONSERVATIVE][N_DIM]> gradients;   ///< d W_k / d x_i
    rtype gamma = 1.4_r;                                      ///< ratio of specific heats, without cell_gamma
    Kokkos::View<rtype *, Kokkos::LayoutStride> cell_gamma;   ///< mixtures: each cell's (frozen) gamma
    Kokkos::View<rtype *[N_DIM]> radius_coords;               ///< axisymmetric runs: cell centers, radius second
};

/**
 * @brief The selected fields of cells [0, n_cells) into the columns of out.
 *        MACH = |u| / sqrt(gamma p / rho), Q_CRITERION and LAMBDA2 of the
 *        velocity gradient, SCHLIEREN = |grad rho|.
 */
void compute(const Selection & selection, const Inputs & inputs, uint32_t n_cells, Kokkos::View<rtype **> out);

} // namespace derived

#endif // DERIVED_FIELDS_H

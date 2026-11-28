/**
 * @file face_point_gradients.h
 * @author Matthew Bonanni (mbonanni001@gmail.com)
 * @brief Gradients of velocity and temperature at the face quadrature points
 *        from the cells' central TENO polynomials (viscous_gradients =
 *        "reconstruction").
 * @version 0.1
 * @date 2026-11-23
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 *
 */

#ifndef FACE_POINT_GRADIENTS_H
#define FACE_POINT_GRADIENTS_H

#include <cstdint>
#include <utility>

#include <Kokkos_Core.hpp>

#include "common_math.h"
#include "common_typedef.h"

namespace face_point_gradients {

/** @brief Exponents of non-constant monomial l of degree <= DEG, in TENO's order (teno::monomials). */
template <uint8_t DEG>
struct Monomial {
    uint8_t e[3] = {0, 0, 0};
    uint8_t degree = 0;
};

template <uint8_t DEG>
KOKKOS_INLINE_FUNCTION constexpr Monomial<DEG> monomial(const uint32_t l) {
    uint32_t k = 0;
    for (uint8_t d = 1; d <= DEG; d++) {
        if constexpr (N_DIM == 2) {
            for (uint8_t b = 0; b <= d; b++) {
                if (k++ == l) return {{static_cast<uint8_t>(d - b), b, 0}, d};
            }
        } else {
            for (int a = d; a >= 0; a--) {
                for (int b = d - a; b >= 0; b--) {
                    if (k++ == l) {
                        return {{static_cast<uint8_t>(a), static_cast<uint8_t>(b), static_cast<uint8_t>(d - a - b)}, d};
                    }
                }
            }
        }
    }
    return {};
}

}  // namespace face_point_gradients

/**
 * @brief At every face quadrature point, the gradients of u and T of each
 *        side's central polynomial, which TENO's smooth passes evaluate while
 *        they hold the coefficients and ViscousFluxFunctor reads.
 *        values(face, slot, side, var) packs the NQ x N_DIM values of each
 *        point into slots of N_CONSERVATIVE, so that a FaceExchange sends a
 *        face side as it sends face states. NaN in a side's first value marks
 *        a cell without a usable polynomial: troubled, or with face values
 *        that fall back to its mean.
 */
struct FacePointGradients {
    static constexpr uint8_t NQ = N_DIM + 1;  // [u, T]
    static constexpr uint8_t PER_POINT = NQ * N_DIM;

    Kokkos::View<rtype **[2][N_CONSERVATIVE]> values;
    rtype R = 0.0_r;
    rtype gamma = 0.0_r;

    static uint8_t n_slots(const uint8_t n_quad) {
        return static_cast<uint8_t>((uint32_t(n_quad) * PER_POINT + N_CONSERVATIVE - 1) / N_CONSERVATIVE);
    }

    KOKKOS_INLINE_FUNCTION
    bool on() const { return values.extent(0) > 0; }

    KOKKOS_INLINE_FUNCTION
    bool usable(const uint32_t f, const uint8_t side) const { return !Kokkos::isnan(values(f, 0, side, 0)); }

    KOKKOS_INLINE_FUNCTION
    void mark(const uint32_t f, const uint8_t side) const {
        values(f, 0, side, 0) = Kokkos::Experimental::quiet_NaN_v<rtype>;
    }

    KOKKOS_INLINE_FUNCTION
    void load(const uint32_t f, const uint8_t q, const uint8_t side, rtype g[NQ][N_DIM]) const {
        const uint32_t j0 = uint32_t(q) * PER_POINT;
        for (uint8_t k = 0; k < NQ; k++) {
            FOR_I_DIM {
                const uint32_t j = j0 + k * N_DIM + i;
                g[k][i] = values(f, j / N_CONSERVATIVE, side, j % N_CONSERVATIVE);
            }
        }
    }

    KOKKOS_INLINE_FUNCTION
    void store(const uint32_t f, const uint8_t q, const uint8_t side, const rtype g[NQ][N_DIM]) const {
        const uint32_t j0 = uint32_t(q) * PER_POINT;
        for (uint8_t k = 0; k < NQ; k++) {
            FOR_I_DIM {
                const uint32_t j = j0 + k * N_DIM + i;
                values(f, j / N_CONSERVATIVE, side, j % N_CONSERVATIVE) = g[k][i];
            }
        }
    }

    /**
     * @brief dU[v][j] = d U_v / d x_j at offset r from the cell center of the
     *        polynomial of degree DEG with coefficients b(l, degree, v) of the
     *        monomials of r, in TENO's order. The monomials are unrolled so
     *        that the powers of r stay in registers.
     */
    template <uint8_t DEG, typename B>
    KOKKOS_INLINE_FUNCTION static void conservative_gradient(const rtype * r, B && b,
                                                             rtype dU[N_CONSERVATIVE][N_DIM]) {
        rtype p[N_DIM][DEG + 1];
        FOR_I_DIM {
            p[i][0] = 1.0_r;
            for (uint8_t k = 1; k <= DEG; k++) p[i][k] = p[i][k - 1] * r[i];
        }
        FOR_I_CONSERVATIVE for (uint8_t j = 0; j < N_DIM; j++) dU[i][j] = 0.0_r;
        constexpr uint32_t NK = (N_DIM == 2) ? (DEG + 1) * (DEG + 2) / 2 - 1 : (DEG + 1) * (DEG + 2) * (DEG + 3) / 6 - 1;
        [&]<uint32_t... L>(std::integer_sequence<uint32_t, L...>) {
            (add_monomial<DEG, L>(p, b, dU), ...);
        }(std::make_integer_sequence<uint32_t, NK>{});
    }

    /**
     * @brief Gradients g[k][i] = d q_k / d x_i of q = [u, T] at face state W
     *        = [rho, u, p], from the gradients dU of the conservative
     *        variables there.
     */
    KOKKOS_INLINE_FUNCTION
    void primitive_gradient(const rtype * W, const rtype dU[N_CONSERVATIVE][N_DIM], rtype g[NQ][N_DIM]) const {
        const rtype rho = W[0];
        const rtype * u = W + 1;
        const rtype T = W[N_DIM + 1] / (rho * R);
        const rtype u2 = dot<N_DIM>(u, u);
        FOR_I_DIM {
            rtype dp = dU[N_DIM + 1][i] - 0.5_r * u2 * dU[0][i];
            for (uint8_t k = 0; k < N_DIM; k++) {
                g[k][i] = (dU[1 + k][i] - u[k] * dU[0][i]) / rho;
                dp -= rho * u[k] * g[k][i];
            }
            dp *= gamma - 1.0_r;
            g[N_DIM][i] = (dp - R * T * dU[0][i]) / (rho * R);
        }
    }

    private:
        // Terms of zero derivative are left out: they add +-0 to sums that start at +0, which changes no bit
        template <uint8_t DEG, uint32_t L, typename B>
        KOKKOS_INLINE_FUNCTION static void add_monomial(const rtype (&p)[N_DIM][DEG + 1], B && b,
                                                        rtype dU[N_CONSERVATIVE][N_DIM]) {
            constexpr face_point_gradients::Monomial<DEG> m = face_point_gradients::monomial<DEG>(L);
            rtype dphi[N_DIM] = {};
            FOR_I_DIM {
                if (m.e[i] == 0) continue;
                rtype d = static_cast<rtype>(m.e[i]) * p[i][m.e[i] - 1];
                for (uint8_t j = 0; j < N_DIM; j++) {
                    if (j != i) d *= p[j][m.e[j]];
                }
                dphi[i] = d;
            }
            FOR_I_CONSERVATIVE {
                const rtype c = b(L, m.degree, i);
                for (uint8_t j = 0; j < N_DIM; j++) {
                    if (m.e[j] != 0) dU[i][j] += c * dphi[j];
                }
            }
        }
};

#endif // FACE_POINT_GRADIENTS_H

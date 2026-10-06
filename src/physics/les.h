/**
 * @file les.h
 * @author Matthew Bonanni (mbonanni001@gmail.com)
 * @brief Subgrid-scale eddy-viscosity models of large-eddy simulation.
 * @version 0.1
 * @date 2026-10-05
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 *
 */

#ifndef LES_H
#define LES_H

#include <string>
#include <unordered_map>

#include <Kokkos_Core.hpp>
#include <toml.hpp>

#include "common.h"
#include "log.h"

enum class SGSModel : uint8_t { SMAGORINSKY, WALE, VREMAN, SIGMA };

static const std::unordered_map<std::string, SGSModel> SGS_MODELS = {
    {"smagorinsky", SGSModel::SMAGORINSKY},
    {"wale", SGSModel::WALE},
    {"vreman", SGSModel::VREMAN},
    {"sigma", SGSModel::SIGMA},
};

/**
 * @brief Eddy viscosity nu_t = C^2 Delta^2 D(g) (Vreman: c Delta^2 D(g)) of the
 *        resolved velocity gradient g[i][j] = d u_i / d x_j, with turbulent
 *        Prandtl and Schmidt numbers for the SGS heat and species fluxes (see
 *        docs/design/les.md). 2D gradients are those of a 3D flow with
 *        u_z = d/dz = 0. Plain aggregate, captured by value in kernels.
 */
struct LES {
    SGSModel model = SGSModel::SIGMA;
    rtype C = 1.35;
    rtype Pr_t = 0.9;
    rtype Sc_t = 0.9;

    /** @brief The [les] table of an input (see in_input). */
    static LES from_input(const toml::value & input);

    static bool in_input(const toml::value & input) { return input.contains("les"); }

    /** @brief Model constant by default: Lilly's C_s, Nicoud & Ducros's C_w, Vreman's c, Nicoud et al.'s C_sigma. */
    static rtype default_constant(SGSModel model);

    logging::Items summary() const;

    /** @brief D(g) of the model, the eddy viscosity for unit constant and filter width. */
    KOKKOS_INLINE_FUNCTION
    static double operator_of(const SGSModel model, const double g[3][3]) {
        switch (model) {
            case SGSModel::SMAGORINSKY:
                return smagorinsky(g);
            case SGSModel::WALE:
                return wale(g);
            case SGSModel::VREMAN:
                return vreman(g);
            default:
                return sigma(g);
        }
    }

    /** @brief Kinematic eddy viscosity for the gradient g[i][j] = d u_i / d x_j and filter width delta. */
    KOKKOS_INLINE_FUNCTION
    rtype nu_t(const double g[3][3], const rtype delta) const {
        const double c = static_cast<double>(C);
        const double coefficient = model == SGSModel::VREMAN ? c : c * c;
        const double d = static_cast<double>(delta);
        return static_cast<rtype>(coefficient * d * d * operator_of(model, g));
    }

    KOKKOS_INLINE_FUNCTION
    static double strain_squared(const double g[3][3]) {
        double s = 0.0;
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                const double s_ij = 0.5 * (g[i][j] + g[j][i]);
                s += s_ij * s_ij;
            }
        }
        return s;
    }

    /** @brief sqrt(2 S:S). */
    KOKKOS_INLINE_FUNCTION
    static double smagorinsky(const double g[3][3]) { return Kokkos::sqrt(2.0 * strain_squared(g)); }

    /** @brief (S^d:S^d)^(3/2) / ((S:S)^(5/2) + (S^d:S^d)^(5/4)), S^d the traceless symmetric part of g^2. */
    KOKKOS_INLINE_FUNCTION
    static double wale(const double g[3][3]) {
        double g2[3][3];
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                g2[i][j] = 0.0;
                for (int k = 0; k < 3; k++) g2[i][j] += g[i][k] * g[k][j];
            }
        }
        const double trace = (g2[0][0] + g2[1][1] + g2[2][2]) / 3.0;
        double sd = 0.0;
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                const double s = 0.5 * (g2[i][j] + g2[j][i]) - (i == j ? trace : 0.0);
                sd += s * s;
            }
        }
        const double ss = strain_squared(g);
        const double denominator = Kokkos::pow(ss, 2.5) + Kokkos::pow(sd, 1.25);
        return denominator > 0.0 ? Kokkos::pow(sd, 1.5) / denominator : 0.0;
    }

    /** @brief sqrt(B / (a:a)) with a_ij = d u_j / d x_i, b = a^T a and B its second invariant. */
    KOKKOS_INLINE_FUNCTION
    static double vreman(const double g[3][3]) {
        double b[3][3], aa = 0.0;
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                // b_ij = sum_m a_mi a_mj = sum_m g_im g_jm
                b[i][j] = 0.0;
                for (int m = 0; m < 3; m++) b[i][j] += g[i][m] * g[j][m];
                aa += g[i][j] * g[i][j];
            }
        }
        const double B = b[0][0] * b[1][1] - b[0][1] * b[0][1] + b[0][0] * b[2][2] - b[0][2] * b[0][2] +
                         b[1][1] * b[2][2] - b[1][2] * b[1][2];
        return aa > 0.0 ? Kokkos::sqrt(Kokkos::fmax(B, 0.0) / aa) : 0.0;
    }

    /**
     * @brief Singular values s1 >= s2 >= s3 >= 0 of g, the square roots of the
     *        eigenvalues l1 >= l2 >= l3 of G = g^T g. l1 comes from the
     *        trigonometric solution of the characteristic cubic (Nicoud et al.
     *        2011, appendix); l2 and l3 from l2 + l3 = (I2 - l2 l3) / l1 and
     *        l2 l3 = det(g)^2 / l1 with the principal minors I2 of G, so that
     *        they keep their relative accuracy when small (s3 = 0 exactly for
     *        planar gradients, s2 = s3 = 0 for pure shear).
     */
    KOKKOS_INLINE_FUNCTION
    static void singular_values(const double g[3][3], double s[3]) {
        double G[3][3];
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                G[i][j] = 0.0;
                for (int k = 0; k < 3; k++) G[i][j] += g[k][i] * g[k][j];
            }
        }
        const double I1 = G[0][0] + G[1][1] + G[2][2];
        const double I2 = G[0][0] * G[1][1] - G[0][1] * G[0][1] + G[0][0] * G[2][2] - G[0][2] * G[0][2] +
                          G[1][1] * G[2][2] - G[1][2] * G[1][2];
        const double det = g[0][0] * (g[1][1] * g[2][2] - g[1][2] * g[2][1]) -
                           g[0][1] * (g[1][0] * g[2][2] - g[1][2] * g[2][0]) +
                           g[0][2] * (g[1][0] * g[2][1] - g[1][1] * g[2][0]);
        const double I3 = det * det;
        double l1 = I1 / 3.0;
        const double a1 = I1 * I1 / 9.0 - I2 / 3.0;
        if (a1 > 0.0) {
            const double a2 = I1 * I1 * I1 / 27.0 - I1 * I2 / 6.0 + I3 / 2.0;
            const double a3 = Kokkos::acos(Kokkos::fmin(1.0, Kokkos::fmax(-1.0, a2 / Kokkos::pow(a1, 1.5)))) / 3.0;
            l1 += 2.0 * Kokkos::sqrt(a1) * Kokkos::cos(a3);
        }
        s[0] = s[1] = s[2] = 0.0;
        if (!(l1 > 0.0)) return;
        const double product = I3 / l1;
        const double sum = Kokkos::fmax((I2 - product) / l1, 0.0);
        const double l2 = 0.5 * (sum + Kokkos::sqrt(Kokkos::fmax(sum * sum - 4.0 * product, 0.0)));
        const double l3 = l2 > 0.0 ? product / l2 : 0.0;
        s[0] = Kokkos::sqrt(l1);
        s[1] = Kokkos::sqrt(Kokkos::fmin(l2, l1));
        s[2] = Kokkos::sqrt(Kokkos::fmin(l3, l2));
    }

    /** @brief s3 (s1 - s2)(s2 - s3) / s1^2 of the singular values s of g. */
    KOKKOS_INLINE_FUNCTION
    static double sigma(const double g[3][3]) {
        double s[3];
        singular_values(g, s);
        if (!(s[0] > 0.0)) return 0.0;
        return Kokkos::fmax(s[2] * (s[0] - s[1]) * (s[1] - s[2]) / (s[0] * s[0]), 0.0);
    }
};

#endif // LES_H

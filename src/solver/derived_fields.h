/**
 * @file derived_fields.h
 * @brief Derived output fields of a flow state (derived_catalog.h) from its
 *        cell values and gradients, for the in-run writers and mallard post
 *        derive, which runs them on a state read from a file.
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 */

#ifndef DERIVED_FIELDS_H
#define DERIVED_FIELDS_H

#include <cstdint>

#include <Kokkos_Core.hpp>

#include "common_typedef.h"
#include "derived_catalog.h"
#include "gradient.h"
#include "input.h"
#include "physics.h"
#include "thermo.h"

class MixtureModel;

namespace derived {

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

/** @brief The divergence of the velocity, the trace of its gradient a (with the hoop strain of axisymmetric runs). */
KOKKOS_INLINE_FUNCTION
rtype dilatation(const Tensor & g) { return g.a[0][0] + g.a[1][1] + g.a[2][2]; }

/** @brief The vorticity w = curl u of the velocity gradient a[k][i] = d u_k / d x_i. */
KOKKOS_INLINE_FUNCTION
void vorticity(const Tensor & g, rtype w[3]) {
    w[0] = g.a[2][1] - g.a[1][2];
    w[1] = g.a[0][2] - g.a[2][0];
    w[2] = g.a[1][0] - g.a[0][1];
}

/** @brief Enstrophy |w|^2 / 2 of the velocity gradient a. */
KOKKOS_INLINE_FUNCTION
rtype enstrophy(const Tensor & g) {
    rtype w[3];
    vorticity(g, w);
    return 0.5_r * (w[0] * w[0] + w[1] * w[1] + w[2] * w[2]);
}

/** @brief Strain-rate magnitude |S| = sqrt(2 S_ij S_ij), S the symmetric part of the velocity gradient a. */
KOKKOS_INLINE_FUNCTION
rtype strain_rate(const Tensor & g) {
    rtype ss = 0.0;
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            const rtype s = 0.5_r * (g.a[i][j] + g.a[j][i]);
            ss += s * s;
        }
    }
    return Kokkos::sqrt(2.0_r * ss);
}

/**
 * @brief Takeno flame index grad Y_F . grad Y_O / (|grad Y_F| |grad Y_O|):
 *        +1 where fuel and oxidizer gradients align (premixed), -1 where
 *        they oppose (non-premixed), 0 where either vanishes.
 */
KOKKOS_INLINE_FUNCTION
rtype flame_index(const rtype * g_fuel, const rtype * g_oxidizer) {
    rtype dot = 0.0, ff = 0.0, oo = 0.0;
    FOR_I_DIM {
        dot += g_fuel[i] * g_oxidizer[i];
        ff += g_fuel[i] * g_fuel[i];
        oo += g_oxidizer[i] * g_oxidizer[i];
    }
    const rtype norm = Kokkos::sqrt(ff) * Kokkos::sqrt(oo);
    return norm > 0.0_r ? dot / norm : 0.0_r;
}

/** @brief Which derived fields are computed, the output column of each, and what they need together. */
struct Selection {
    int32_t column[N_FIELDS];
    uint32_t n_columns = 0;
    uint16_t needs = 0;  ///< need:: bits of the fields selected

    Selection() {
        for (int32_t & c : column) c = -1;
    }
    /** @brief Compute field f, in the next column. */
    void add(Field f) {
        if (column[f] >= 0) return;
        column[f] = static_cast<int32_t>(n_columns++);
        needs = static_cast<uint16_t>(needs | FIELDS[f].needs);
    }
    /** @brief Whether any field is computed. */
    bool any() const { return n_columns > 0; }
    /** @brief Whether the selected fields need all of the need:: bits n. */
    bool need(uint16_t n) const { return (needs & n) != 0; }
};

/** @brief The slots of the mass-fraction combinations whose gradients FLAME_INDEX and SCALAR_DISSIPATION read. */
enum Scalar : uint8_t { FUEL, OXIDIZER, Z, N_SCALARS };

/**
 * @brief What the fields take from the input, beyond the state: the reference
 *        of CP, the perfect gas's entropy datum, and for mixtures the streams
 *        of the mixture fraction, the flame index's species and the progress
 *        variable. Read once at setup by from_input.
 */
struct Settings {
    // Perfect gas: ENTROPY = cp ln(T / T_ref) - R ln(p / p_ref), zero at [physics] p_ref, T_ref
    rtype cp = 0.0, R = 0.0, T_ref = 1.0, p_ref = 1.0;
    // CP = (p - p_inf) / q_inf, of [output] reference
    bool reference = false;
    rtype p_inf = 0.0, q_inf = 1.0;
    // Mixtures: weights (slot, species) of the Scalar combinations of mass fractions; Z adds z_offset
    Kokkos::View<rtype **, Kokkos::LayoutRight> scalar_weights;
    bool streams = false;
    rtype z_offset = 0.0;
    int32_t fuel_species = -1, oxidizer_species = -1;
    // PROGRESS_VARIABLE: sum of progress_weights Y (species), or (T - T_unburnt) / (T_burnt - T_unburnt)
    enum class Progress : uint8_t { NONE, SPECIES, TEMPERATURE } progress = Progress::NONE;
    Kokkos::View<rtype *> progress_weights;
    rtype T_unburnt = 0.0, T_burnt = 1.0;
    bool mixture = false, viscous = false;

    /**
     * @brief From [output] reference, [output.mixture] and, for a perfect gas,
     *        [physics] p_ref and T_ref; mixture is null for a perfect gas.
     */
    static Settings from_input(const Input & input, const Euler & physics, const MixtureModel * mixture,
                               bool viscous);

    /** @brief Whether this case can compute field f. */
    bool available(Field f) const;
    /**
     * @brief Throws the InputError of a [[write_data]] request for field f that
     *        this case cannot compute: what it needs and how to give it.
     */
    void require(Field f, const Input & input) const;
};

/** @brief Scratch of the mass-fraction gradients (allocated if the selection needs them). */
struct Workspace {
    Kokkos::View<rtype **, Kokkos::LayoutRight> values;      ///< (cell, Scalar)
    Kokkos::View<rtype ***, Kokkos::LayoutRight> gradients;  ///< (cell, Scalar, dimension)

    Workspace() = default;
    /** @brief Allocated for n_cells cells. */
    explicit Workspace(uint32_t n_cells);
};

/**
 * @brief The Scalar combinations of each cell's mass fractions Y (cells
 *        [0, n_cells), owned and halo) and their gradients in the owned cells
 *        [0, n_owned), with the stencil and weights of the least-squares cell
 *        gradient: the boundary ghost takes the prescribed composition bc_Y of
 *        inlet_state conditions, the cell's own elsewhere.
 */
void scalar_gradients(const Settings & settings, const Kokkos::View<rtype **, Kokkos::LayoutRight> & Y,
                      const LSQVertexGradientFunctor & stencil, const Kokkos::View<rtype **, Kokkos::LayoutRight> & bc_Y,
                      uint32_t n_cells, uint32_t n_owned, Workspace & workspace);

/** @brief The cell values a derived field is computed from. */
struct Inputs {
    Kokkos::View<rtype *[N_CONSERVATIVE]> W{};                ///< rho, u, p of each cell
    Kokkos::View<rtype *[N_CONSERVATIVE][N_DIM]> gradients{}; ///< d W_k / d x_i
    rtype gamma = 1.4_r;                                      ///< ratio of specific heats, without cell_gamma
    Kokkos::View<rtype *, Kokkos::LayoutStride> cell_gamma{}; ///< mixtures: each cell's (frozen) gamma
    Kokkos::View<rtype *[N_DIM]> radius_coords{};             ///< axisymmetric runs: cell centers, radius second
    Kokkos::View<rtype *[N_PRIMITIVE]> primitives{};          ///< u, p, T, h of each cell
    Kokkos::View<rtype **, Kokkos::LayoutRight> Y{};          ///< mixtures: (cell, [Y_1 .. Y_Ns, ...])
    chemistry::ThermoTable<> thermo{};                        ///< mixtures: species thermodynamics
    Kokkos::View<rtype *[3]> transport{};                     ///< viscous mixtures: (cell, [mu, lambda, -])
    Kokkos::View<rtype ***, Kokkos::LayoutRight> scalar_gradients{};  ///< Workspace::gradients
};

/**
 * @brief The selected fields of cells [0, n_cells) into the columns of out,
 *        as docs/guide/output.md#derived-fields defines them.
 */
void compute(const Selection & selection, const Settings & settings, const Inputs & inputs, uint32_t n_cells,
             Kokkos::View<rtype **> out);

} // namespace derived

#endif // DERIVED_FIELDS_H

/**
 * @file positivity.h
 * @author Matthew Bonanni (mbonanni001@gmail.com)
 * @brief Positivity preservation: Zhang-Shu scaling of reconstructed face
 *        states and the a-posteriori order fallback (MOOD) of each stage.
 * @version 0.9
 * @date 2026-10-15
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 *
 */

#ifndef POSITIVITY_H
#define POSITIVITY_H

#include <Kokkos_Core.hpp>

#include "common.h"
#include "mixture.h"
#include "mixture_flux.h"
#include "scalar_reconstruction.h"
#include "state.h"

namespace positivity {

/**
 * @brief Margin of the face-state limiter: rho and p at least this fraction
 *        of the cell's. Far above round-off on purpose: in cold gas, where p
 *        is a small difference of the total and kinetic energies, face
 *        pressures of 1e-12 p_c are noise, and on the Noh problem (TENO5, RHLL)
 *        they drive the pressure to zero within a few steps.
 */
constexpr rtype EPSILON = 1.0e-6_r;

/** @brief Fallback levels: 0 the scheme, 1 first order, 2 first order with the Rusanov flux. */
constexpr uint8_t MAX_LEVEL = 2;

/**
 * @brief (E - rho e0) rho - |m|^2 / 2 - k rho, which has the sign of
 *        p - (gamma - 1) k for rho > 0: concave in U, so its zero along a
 *        segment from an admissible state is bracketed and unique.
 */
KOKKOS_INLINE_FUNCTION
rtype pressure_residual(const rtype * U, const rtype e0, const rtype k) {
    return (U[N_DIM + 1] - U[0] * e0) * U[0] - 0.5_r * dot<N_DIM>(U + 1, U + 1) - k * U[0];
}

/** @brief Conservatives of W with the cell's frozen gamma and e0 (e0 = 0 for a perfect gas). */
KOKKOS_INLINE_FUNCTION
void to_conservatives(const rtype * W, const rtype gamma, const rtype e0, rtype * U) {
    U[0] = W[0];
    FOR_I_DIM U[1 + i] = W[0] * W[1 + i];
    U[N_DIM + 1] = W[N_DIM + 1] / (gamma - 1.0_r) + W[0] * e0 + 0.5_r * W[0] * dot<N_DIM>(W + 1, W + 1);
}

KOKKOS_INLINE_FUNCTION
void to_primitives(const rtype * U, const rtype gamma, const rtype e0, rtype * W) {
    W[0] = U[0];
    FOR_I_DIM W[1 + i] = U[1 + i] / U[0];
    W[N_DIM + 1] = (gamma - 1.0_r) * (U[N_DIM + 1] - U[0] * e0 - 0.5_r * U[0] * dot<N_DIM>(W + 1, W + 1));
}

/**
 * @brief Zhang-Shu scaling of the reconstructed face states of each cell
 *        (Zhang & Shu 2010; per cell as Ching et al. 2025): the largest
 *        theta in [0, 1] such that U_c + theta (U_q - U_c) has rho >= eps rho_c
 *        and p >= eps p_c at every face point q of the cell, first for density
 *        (linear), then for pressure (concave, by bisection), applied to all of
 *        the cell's face states. Conservatives use the cell's gamma and e0
 *        (mixtures), whose pressure positivity is that of the temperature.
 *        Cells whose states are admissible are left untouched, bit for bit;
 *        non-finite states give theta = 0 (the cell mean).
 */
struct FaceLimiterFunctor {
    Kokkos::View<uint32_t *> offsets_faces_of_cell;
    Kokkos::View<uint32_t *> faces_of_cell;
    Kokkos::View<int32_t *[2]> cells_of_face;
    Kokkos::View<rtype **[2][N_CONSERVATIVE]> face_solution;
    Kokkos::View<rtype **> face_weights;  // 3D: (face, q), zero on padding points; else empty
    Kokkos::View<rtype *[N_CONSERVATIVE]> W;
    ScalarView scalars;  // mixtures: (cell, [Y, gamma, e0]); else empty
    rtype gamma;
    uint32_t n_species;
    uint8_t n_quad;
    uint32_t n_owned;
    Kokkos::View<uint64_t> n_limited;  // owned cells limited

    KOKKOS_INLINE_FUNCTION
    bool padding(const uint32_t f, const uint8_t q) const {
        return face_weights.extent(0) > 0 && face_weights(f, q) == 0.0_r;
    }

    KOKKOS_INLINE_FUNCTION
    void operator()(const uint32_t c) const {
        constexpr uint8_t P = N_DIM + 1;
        rtype W0[N_CONSERVATIVE], U0[N_CONSERVATIVE];
        FOR_I_CONSERVATIVE W0[i] = W(c, i);
        if (!(W0[0] > 0.0_r && W0[P] > 0.0_r)) return;
        const rtype g = scalars.extent(0) ? scalars(c, n_species) : gamma;
        const rtype e0 = scalars.extent(0) ? scalars(c, n_species + 1) : 0.0_r;
        to_conservatives(W0, g, e0, U0);
        const rtype eps_rho = EPSILON * W0[0];
        const rtype eps_p = EPSILON * W0[P];
        const uint32_t begin = offsets_faces_of_cell(c);
        const uint32_t end = offsets_faces_of_cell(c + 1);

        rtype theta = 1.0_r;
        bool admissible = true;
        for (uint32_t j = begin; j < end; j++) {
            const uint32_t f = faces_of_cell(j);
            const uint8_t side = cells_of_face(f, 0) == static_cast<int32_t>(c) ? 0 : 1;
            for (uint8_t q = 0; q < n_quad; q++) {
                if (padding(f, q)) continue;
                rtype Wq[N_CONSERVATIVE], U[N_CONSERVATIVE];
                FOR_I_CONSERVATIVE Wq[i] = face_solution(f, q, side, i);
                to_conservatives(Wq, g, e0, U);
                bool finite = true;
                FOR_I_CONSERVATIVE finite = finite && Kokkos::isfinite(U[i]);
                if (!finite) {
                    theta = 0.0_r;
                    admissible = false;
                    continue;
                }
                if (U[0] < eps_rho) theta = Kokkos::fmin(theta, (U0[0] - eps_rho) / (U0[0] - U[0]));
                admissible = admissible && U[0] >= eps_rho && Wq[P] >= eps_p;
            }
        }
        if (admissible) return;

        // Pressure, on the density-limited states
        const rtype k = eps_p / (g - 1.0_r);
        rtype theta_p = 1.0_r;
        if (theta > 0.0_r) {
            for (uint32_t j = begin; j < end; j++) {
                const uint32_t f = faces_of_cell(j);
                const uint8_t side = cells_of_face(f, 0) == static_cast<int32_t>(c) ? 0 : 1;
                for (uint8_t q = 0; q < n_quad; q++) {
                    if (padding(f, q)) continue;
                    rtype Wq[N_CONSERVATIVE], U[N_CONSERVATIVE], dU[N_CONSERVATIVE];
                    FOR_I_CONSERVATIVE Wq[i] = face_solution(f, q, side, i);
                    to_conservatives(Wq, g, e0, U);
                    FOR_I_CONSERVATIVE dU[i] = theta * (U[i] - U0[i]);
                    FOR_I_CONSERVATIVE U[i] = U0[i] + dU[i];
                    if (pressure_residual(U, e0, k) >= 0.0_r) continue;
                    rtype lo = 0.0_r, hi = 1.0_r;
                    for (int it = 0; it < 60; it++) {
                        const rtype mid = 0.5_r * (lo + hi);
                        FOR_I_CONSERVATIVE U[i] = U0[i] + mid * dU[i];
                        if (pressure_residual(U, e0, k) >= 0.0_r) {
                            lo = mid;
                        } else {
                            hi = mid;
                        }
                    }
                    theta_p = Kokkos::fmin(theta_p, lo);
                }
            }
        }
        theta *= theta_p;
        if (theta >= 1.0_r) return;
        if (c < n_owned) Kokkos::atomic_inc(&n_limited());

        for (uint32_t j = begin; j < end; j++) {
            const uint32_t f = faces_of_cell(j);
            const uint8_t side = cells_of_face(f, 0) == static_cast<int32_t>(c) ? 0 : 1;
            for (uint8_t q = 0; q < n_quad; q++) {
                rtype Wq[N_CONSERVATIVE], U[N_CONSERVATIVE];
                if (theta > 0.0_r && !padding(f, q)) {
                    FOR_I_CONSERVATIVE Wq[i] = face_solution(f, q, side, i);
                    to_conservatives(Wq, g, e0, U);
                    FOR_I_CONSERVATIVE U[i] = U0[i] + theta * (U[i] - U0[i]);
                    to_primitives(U, g, e0, Wq);
                } else {
                    FOR_I_CONSERVATIVE Wq[i] = W0[i];
                }
                FOR_I_CONSERVATIVE face_solution(f, q, side, i) = Wq[i];
            }
        }
    }
};

/** @brief Mass fractions of a candidate state, rho Y_k + dt R_k over rho + dt R_rho. */
struct CandidateFractions {
    SpeciesView rhoY, rhs;
    uint32_t c;
    rtype dt;
    double inv_rho;
    KOKKOS_INLINE_FUNCTION double operator()(const uint32_t k) const {
        return static_cast<double>(rhoY(c, k) + dt * rhs(c, k)) * inv_rho;
    }
};

/**
 * @brief Detection step of the order fallback (MOOD; Clain, Diot & Loubère
 *        2011, with the relaxed criteria of Farmakis, Tsoutsanis & Nogueira
 *        2020 and, for mixtures, Tsoutsanis et al. 2023): the forward-Euler
 *        candidate U + dt R of each owned cell, which every stage of FE and
 *        SSPRK3 is a convex combination of, must be
 *        - physically admissible (PAD): finite, rho > 0, p > 0 (mixtures: the
 *          temperature extrapolated linearly below the fits' range positive,
 *          e > e(T_low) - cv(T_low) T_low), and no partial density below
 *          min(0, its current value);
 *        - optionally, within the relaxed discrete maximum principle (DMP) of
 *          its face neighborhood in rho and p: [min - delta, max + delta],
 *          delta = max(dmp_floor max(|min|, |max|), dmp_range (max - min)).
 *        A cell that fails goes up one level (DMP: only from level 0). Sums
 *        the cells that changed.
 */
struct CandidateFunctor {
    Kokkos::View<uint32_t *> offsets_faces_of_cell;
    Kokkos::View<uint32_t *> faces_of_cell;
    Kokkos::View<int32_t *[2]> cells_of_face;
    StateView U;
    StateView rhs;
    SpeciesView rhoY;
    SpeciesView rhs_Y;
    Kokkos::View<rtype *[N_CONSERVATIVE]> W;  // stage input, for the DMP
    Kokkos::View<rtype *> T_seed;             // mixtures
    Kokkos::View<uint8_t *> level;
    Mixture gas;
    bool mixture;
    rtype gamma;
    rtype dt;
    bool dmp;
    rtype dmp_floor;
    rtype dmp_range;
    bool count_failures = false;  // only count the cells whose candidate is not admissible

    /** @brief rho and p of the candidate; false if inadmissible. */
    KOKKOS_INLINE_FUNCTION
    bool admissible(const uint32_t c, rtype & rho, rtype & p) const {
        constexpr uint8_t E = N_DIM + 1;
        rtype Uc[N_CONSERVATIVE];
        bool finite = true;
        FOR_I_CONSERVATIVE {
            Uc[i] = U(c, i) + dt * rhs(c, i);
            finite = finite && Kokkos::isfinite(Uc[i]);
        }
        rho = Uc[0];
        if (!finite || !(rho > 0.0_r)) return false;
        const rtype kinetic = 0.5_r * dot<N_DIM>(Uc + 1, Uc + 1) / rho;
        if (!mixture) {
            p = (gamma - 1.0_r) * (Uc[E] - kinetic);
            return p > 0.0_r;
        }
        for (uint32_t k = 0; k < gas.n_species; k++) {
            const rtype value = rhoY(c, k) + dt * rhs_Y(c, k);
            if (!Kokkos::isfinite(value) || value < Kokkos::fmin(0.0_r, rhoY(c, k))) return false;
        }
        const CandidateFractions y{rhoY, rhs_Y, c, dt, 1.0 / static_cast<double>(rho)};
        const double e = static_cast<double>(Uc[E] - kinetic) / static_cast<double>(rho);
        double e_low, cv_low;
        gas.thermo.e_cv(gas.thermo.T_low, y, e_low, cv_low);
        if (!(e > e_low - cv_low * gas.thermo.T_low)) return false;
        if (dmp) {
            const double T = gas.thermo.T_from_e(e, y, static_cast<double>(T_seed(c)));
            p = static_cast<rtype>(static_cast<double>(rho) * gas.thermo.gas_constant(y) * T);
        }
        return true;
    }

    KOKKOS_INLINE_FUNCTION
    bool within_dmp(const uint32_t c, const rtype rho, const rtype p) const {
        constexpr uint8_t P = N_DIM + 1;
        rtype lo[2] = {W(c, 0), W(c, P)}, hi[2] = {W(c, 0), W(c, P)};
        for (uint32_t j = offsets_faces_of_cell(c); j < offsets_faces_of_cell(c + 1); j++) {
            const uint32_t f = faces_of_cell(j);
            const int32_t c0 = cells_of_face(f, 0), c1 = cells_of_face(f, 1);
            const int32_t nb = c0 == static_cast<int32_t>(c) ? c1 : c0;
            if (nb < 0) continue;
            lo[0] = Kokkos::fmin(lo[0], W(nb, 0));
            hi[0] = Kokkos::fmax(hi[0], W(nb, 0));
            lo[1] = Kokkos::fmin(lo[1], W(nb, P));
            hi[1] = Kokkos::fmax(hi[1], W(nb, P));
        }
        const rtype values[2] = {rho, p};
        for (int v = 0; v < 2; v++) {
            const rtype delta = Kokkos::fmax(dmp_floor * Kokkos::fmax(Kokkos::fabs(lo[v]), Kokkos::fabs(hi[v])),
                                             dmp_range * (hi[v] - lo[v]));
            if (values[v] < lo[v] - delta || values[v] > hi[v] + delta) return false;
        }
        return true;
    }

    KOKKOS_INLINE_FUNCTION
    void operator()(const uint32_t c, uint32_t & n_changed) const {
        const uint8_t lv = level(c);
        if (count_failures) {
            rtype rho, p;
            if (!admissible(c, rho, p)) n_changed++;
            return;
        }
        if (lv >= MAX_LEVEL) return;
        rtype rho = 0.0_r, p = 0.0_r;
        bool ok = admissible(c, rho, p);
        if (ok && dmp && lv == 0) ok = within_dmp(c, rho, p);
        if (!ok) {
            level(c) = static_cast<uint8_t>(lv + 1);
            n_changed++;
        }
    }
};

/**
 * @brief Faces whose level, the larger of their cells', went up: marks them
 *        changed and writes first-order states (and, for mixtures, the cells'
 *        gamma and e0). Only faces of owned cells; the others are left as
 *        they are.
 */
struct FallbackFacesFunctor {
    Kokkos::View<int32_t *[2]> cells_of_face;
    Kokkos::View<uint8_t *> level;
    Kokkos::View<uint8_t *> face_level;
    Kokkos::View<uint8_t *> changed;
    Kokkos::View<rtype **[2][N_CONSERVATIVE]> face_solution;
    Kokkos::View<rtype **[2][2]> face_thermo;  // mixtures, else empty
    Kokkos::View<rtype *[N_CONSERVATIVE]> W;
    ScalarView scalars;  // mixtures, else empty
    uint32_t n_owned;
    uint32_t n_species;

    KOKKOS_INLINE_FUNCTION
    void operator()(const uint32_t f) const {
        const int32_t c0 = cells_of_face(f, 0), c1 = cells_of_face(f, 1);
        changed(f) = 0;
        const bool owned = static_cast<uint32_t>(c0) < n_owned || (c1 >= 0 && static_cast<uint32_t>(c1) < n_owned);
        if (!owned) return;
        uint8_t lv = level(c0);
        if (c1 >= 0 && level(c1) > lv) lv = level(c1);
        if (lv == face_level(f)) return;
        face_level(f) = lv;
        changed(f) = 1;
        const uint32_t nq = static_cast<uint32_t>(face_solution.extent(1));
        for (uint8_t s = 0; s < 2; s++) {
            const int32_t c = s ? c1 : c0;
            if (c < 0) continue;
            for (uint32_t q = 0; q < nq; q++) {
                FOR_I_CONSERVATIVE face_solution(f, q, s, i) = W(c, i);
                if (face_thermo.extent(0) > 0) {
                    face_thermo(f, q, s, 0) = scalars(c, n_species);
                    face_thermo(f, q, s, 1) = scalars(c, n_species + 1);
                }
            }
        }
    }
};

/** @brief No species: single gases. */
struct NoSlots {
    KOKKOS_INLINE_FUNCTION void operator()(const uint32_t) const {}
};

/** @brief No viscous flux: inviscid runs. */
struct NoViscous {
    KOKKOS_INLINE_FUNCTION void operator()(const uint32_t) const {}
};

/**
 * @brief Species slots of a fallback face (see SpeciesSlotFunctor) from the
 *        cells' own mass fractions, the first-order reconstruction: each
 *        side's outflow and, on boundary faces, the inflow of the exterior
 *        state, after the convective flux has set face_mdot.
 */
struct FirstOrderSlots {
    SpeciesSlotFunctor<ScalarFaceValues> slot;  // its values evaluate cell values (no gradients)

    KOKKOS_INLINE_FUNCTION
    void operator()(const uint32_t f) const {
        constexpr uint8_t MAX_Q = teno::MAX_FACE_QUAD;
        const uint8_t nq = slot.n_quad();
        const rtype scale = 0.5_r * slot.face_area(f);
        rtype w_out[2][MAX_Q] = {}, w_in[MAX_Q] = {};
        for (uint8_t q = 0; q < nq; q++) {
            rtype w_q;
            if constexpr (N_DIM == 2) {
                w_q = slot.face_weights.extent(0) ? slot.face_weights(f, q) : slot.quad_weights(q);
            } else {
                w_q = slot.face_weights(f, q);
            }
            const rtype m = slot.face_mdot(f, q);
            w_out[0][q] = scale * w_q * Kokkos::fmax(m, 0.0_r);
            w_out[1][q] = scale * w_q * Kokkos::fmax(-m, 0.0_r);
            w_in[q] = w_out[1][q];
        }
        const int32_t c1 = slot.cells_of_face(f, 1);
        for (uint32_t k = 0; k < slot.n_species; k++) {
            for (uint8_t s = 0; s < 2; s++) {
                const int32_t c = s ? c1 : slot.cells_of_face(f, 0);
                if (c < 0) continue;
                const rtype Y = slot.values.scalars(c, k);
                rtype out = 0.0_r;
                for (uint8_t q = 0; q < nq; q++) out += w_out[s][q] * Y;
                slot.slots(f, s, k) = out;
            }
            if (c1 < 0) {
                rtype interior[MAX_Q], Y_ext[MAX_Q];
                for (uint8_t q = 0; q < nq; q++) interior[q] = slot.values.scalars(slot.cells_of_face(f, 0), k);
                slot.exterior_Y(f, k, interior, Y_ext);
                rtype in = 0.0_r;
                for (uint8_t q = 0; q < nq; q++) in += w_in[q] * Y_ext[q];
                slot.slots(f, 1, k) = in;
            }
        }
    }
};

/**
 * @brief Recomputes the fluxes of the changed faces at one level: the
 *        convective flux (first-order states, no low-Mach correction or
 *        central blend), the species slots, then the viscous flux, which adds
 *        to both.
 */
template <typename Convective, typename Slots, typename Viscous>
struct FallbackFluxFunctor {
    Convective convective;
    Slots slots;
    Viscous viscous;
    bool is_viscous;
    Kokkos::View<uint8_t *> face_level;
    Kokkos::View<uint8_t *> changed;
    uint8_t level;

    KOKKOS_INLINE_FUNCTION
    void operator()(const uint32_t f) const {
        if (!changed(f) || face_level(f) != level) return;
        convective(f);
        slots(f);
        if (is_viscous) viscous(f);
    }
};

/** @brief Cell levels to and from the flow block of a state, for the halo exchange. */
struct LevelPackFunctor {
    struct Pack {};
    struct Unpack {};
    Kokkos::View<uint8_t *> level;
    StateView packed;

    KOKKOS_INLINE_FUNCTION void operator()(Pack, const uint32_t c) const { packed(c, 0) = static_cast<rtype>(level(c)); }
    KOKKOS_INLINE_FUNCTION void operator()(Unpack, const uint32_t c) const {
        level(c) = static_cast<uint8_t>(packed(c, 0));
    }
};

/** @brief Owned cells at level 1 and at level 2. */
struct LevelCountFunctor {
    Kokkos::View<uint8_t *> level;

    struct value_type {
        uint64_t n[2];
    };

    KOKKOS_INLINE_FUNCTION void init(value_type & v) const { v.n[0] = v.n[1] = 0; }
    KOKKOS_INLINE_FUNCTION void join(value_type & dst, const value_type & src) const {
        dst.n[0] += src.n[0];
        dst.n[1] += src.n[1];
    }
    KOKKOS_INLINE_FUNCTION void operator()(const uint32_t c, value_type & v) const {
        if (level(c) > 0) v.n[level(c) - 1]++;
    }
};

} // namespace positivity

#endif // POSITIVITY_H

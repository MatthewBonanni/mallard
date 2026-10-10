/**
 * @file kinetics.h
 * @author Matthew Bonanni (mbonanni001@gmail.com)
 * @brief Device tables of a mechanism's reactions: rates of progress,
 *        production rates and their analytical derivatives.
 * @version 0.3
 * @date 2026-10-03
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 *
 */

#ifndef CHEMISTRY_KINETICS_H
#define CHEMISTRY_KINETICS_H

#include <cmath>
#include <cstdint>
#include <vector>

#include <Kokkos_Core.hpp>

#include "lanes.h"
#include "mechanism.h"

namespace chemistry {

/**
 * @brief Per-reaction derivatives of the rates of progress, in one block of
 *        KineticsTable::derivatives_size() doubles: d q / dT at fixed
 *        concentrations; d q / d C of each forward (reactant) and reverse
 *        (product) mass-action term; d q / d[M] of third-body and falloff
 *        reactions; and the d q / d C_j shared by every species through the
 *        pressure of PLOG and Chebyshev reactions.
 */
struct ReactionDerivatives {
    double * dq_dT;
    double * d_forward;
    double * d_reverse;
    double * dq_dM;
    double * dq_uniform;

    KOKKOS_INLINE_FUNCTION
    static ReactionDerivatives at(double * work, const uint32_t n_reactions, const uint32_t n_forward,
                                  const uint32_t n_reverse) {
        ReactionDerivatives d;
        d.dq_dT = work;
        d.d_forward = d.dq_dT + n_reactions;
        d.d_reverse = d.d_forward + n_forward;
        d.dq_dM = d.d_reverse + n_reverse;
        d.dq_uniform = d.dq_dM + n_reactions;
        return d;
    }
};

/**
 * @brief Reactions of a mechanism as flat device tables (compressed rows per
 *        reaction), in MemorySpace. All quantities SI with kmol, in double
 *        precision in every build. Concentrations C_k = rho Y_k / W_k.
 *
 * Rates follow Cantera: k = A T^b exp(-Ea / RT); three-body reactions
 * multiply both directions by [M] = sum_k eff_k C_k; falloff reactions use
 * k = k_inf Pr / (1 + Pr) F with Pr = k_0 [M] / k_inf and F = 1 (Lindemann),
 * Troe or SRI; PLOG reactions interpolate ln k linearly in ln p between
 * the bracketing pressures (constant outside them), summing the rates given
 * at one pressure; Chebyshev reactions take log10 k as a series in reduced
 * 1/T and log10 p, with p = C_total R T; reversible reactions take k_r = k_f / K_c with
 * K_c = exp(-sum_k nu_k g_k / RT) (p_atm / RT)^(sum_k nu_k).
 */
template <typename MemorySpace = Kokkos::DefaultExecutionSpace::memory_space>
struct KineticsTable {
    template <typename T>
    using View1 = Kokkos::View<T *, MemorySpace>;

    uint32_t n_species = 0;
    uint32_t n_reactions = 0;
    View1<uint8_t> type;          // ReactionType
    View1<uint8_t> falloff;       // FalloffType
    View1<uint8_t> reversible;
    Kokkos::View<double *[3], Kokkos::LayoutRight, MemorySpace> rate;  // A, b, Ea/R (falloff: high pressure)
    Kokkos::View<double *[3], Kokkos::LayoutRight, MemorySpace> low;   // falloff: low pressure
    Kokkos::View<double *[5], Kokkos::LayoutRight, MemorySpace> falloff_params;
    View1<uint32_t> forward_offset;  // (n_reactions + 1)
    View1<uint32_t> forward_species;
    View1<double> forward_order;
    View1<uint32_t> reverse_offset;  // products and their coefficients (the reverse orders)
    View1<uint32_t> reverse_species;
    View1<double> reverse_order;
    View1<uint32_t> net_offset;      // nonzero net coefficients nu'' - nu'
    View1<uint32_t> net_species;
    View1<double> net_nu;
    View1<uint32_t> species_offset;  // (n_species + 1): the same coefficients by species, by reaction index
    View1<uint32_t> species_reaction;
    View1<double> species_nu;
    View1<uint32_t> chunk_offset;  // (n_species + 1): chunks of each species' coefficients, for lanes
    View1<uint32_t> chunk_end;     // (chunk): end of its range in the species' coefficients
    // Jacobian entries d omega_k / d C_j with their terms (per-reaction derivative and coefficient), for lanes
    View1<uint32_t> entry_row, entry_column;
    View1<uint32_t> entry_offset;  // (entries + 1)
    View1<uint32_t> term_source;   // index into the ReactionDerivatives block
    View1<double> term_coefficient;
    View1<double> delta_nu;          // sum of net coefficients
    View1<uint32_t> efficiency_offset;
    View1<uint32_t> efficiency_species;
    View1<double> efficiency_extra;  // efficiency minus the default
    View1<double> default_efficiency;
    View1<uint32_t> plog_offset;       // (n_reactions + 1): pressure levels of reaction i
    View1<double> plog_ln_p;           // (level)
    View1<uint32_t> plog_rate_offset;  // (level + 1): rates summed at a level
    Kokkos::View<double *[3], Kokkos::LayoutRight, MemorySpace> plog_rate;  // A, b, Ea/R
    View1<uint32_t> chebyshev_offset;  // (n_reactions + 1): coefficients of reaction i, n_T x n_p row-major
    View1<uint32_t> chebyshev_n_p;
    View1<double> chebyshev;
    Kokkos::View<double *[4], Kokkos::LayoutRight, MemorySpace> chebyshev_range;  // 1/T_min, 1/T_max, ln p_min, ln p_max

    /**
     * @brief Concentration [kmol/m^3] below which orders 0 < n < 1 follow
     *        C_reg^n x ((2 - n) + (n - 1) x), x = C / C_reg (linear in C
     *        for x < 0): C^1 at C_reg, with a bounded slope at C = 0 where
     *        n C^(n - 1) diverges.
     */
    static constexpr double C_REG = 1e-12;
    double C_reg = C_REG;  // the concentration used (chemistry.C_reg)

    /** @brief Factors C^order of a reaction's products that the derivatives reuse. */
    static constexpr uint32_t MAX_KEPT = 4;

    /** @brief C^order: repeated products for integer orders, pow of max(C, 0) otherwise. */
    KOKKOS_INLINE_FUNCTION
    static double plain_power(const double C, const double order) {
        const double whole = Kokkos::floor(order);
        if (whole == order && order >= 0.0 && order <= 4.0) {
            double p = 1.0;
            for (int i = 0; i < static_cast<int>(order); i++) p *= C;
            return p;
        }
        return Kokkos::pow(Kokkos::fmax(C, 0.0), order);
    }

    /** @brief Mass-action factor C^order, regularized below C_reg for 0 < order < 1. */
    KOKKOS_INLINE_FUNCTION
    static double power(const double C, const double order, const double C_reg = C_REG) {
        if (order > 0.0 && order < 1.0 && C < C_reg) {
            const double x = C / C_reg;
            return Kokkos::pow(C_reg, order) * x * ((2.0 - order) + (order - 1.0) * Kokkos::fmax(x, 0.0));
        }
        return plain_power(C, order);
    }

    /** @brief d power(C, order) / dC. */
    KOKKOS_INLINE_FUNCTION
    static double power_derivative(const double C, const double order, const double C_reg = C_REG) {
        if (order == 0.0) return 0.0;
        if (order > 0.0 && order < 1.0 && C < C_reg) {
            const double x = C / C_reg;
            return Kokkos::pow(C_reg, order - 1.0) * ((2.0 - order) + 2.0 * (order - 1.0) * Kokkos::fmax(x, 0.0));
        }
        return order * plain_power(C, order - 1.0);
    }

    KOKKOS_INLINE_FUNCTION
    static double arrhenius(const double A, const double b, const double Ea_R, const double log_T, const double inv_T) {
        return A * Kokkos::exp(b * log_T - Ea_R * inv_T);
    }

    /** @brief Sum of the PLOG rates at pressure level l and its d ln k / dT. */
    KOKKOS_INLINE_FUNCTION
    void plog_level(const uint32_t l, const double log_T, const double inv_T, double & k, double & dlnk_dT) const {
        k = 0.0;
        double weighted = 0.0;
        for (uint32_t e = plog_rate_offset(l); e < plog_rate_offset(l + 1); e++) {
            const double k_e = arrhenius(plog_rate(e, 0), plog_rate(e, 1), plog_rate(e, 2), log_T, inv_T);
            k += k_e;
            weighted += k_e * (plog_rate(e, 1) + plog_rate(e, 2) * inv_T) * inv_T;
        }
        dlnk_dT = k != 0.0 ? weighted / k : 0.0;
    }

    /**
     * @brief Rate constant of a PLOG or Chebyshev reaction at T and ln p,
     *        with d ln k / dT at fixed p and d ln k / d ln p.
     */
    KOKKOS_INLINE_FUNCTION
    void pressure_rate(const uint32_t i, const double log_T, const double inv_T, const double ln_p, double & k,
                       double & dlnk_dT, double & dlnk_dlnp) const {
        constexpr double LN10 = 2.302585092994045684;
        dlnk_dlnp = 0.0;
        if (type(i) == static_cast<uint8_t>(ReactionType::PLOG)) {
            const uint32_t first = plog_offset(i), last = plog_offset(i + 1) - 1;
            if (ln_p <= plog_ln_p(first) || first == last) {
                plog_level(first, log_T, inv_T, k, dlnk_dT);
                return;
            }
            if (ln_p >= plog_ln_p(last)) {
                plog_level(last, log_T, inv_T, k, dlnk_dT);
                return;
            }
            uint32_t l = first;
            while (ln_p >= plog_ln_p(l + 1)) l++;
            double k1, d1, k2, d2;
            plog_level(l, log_T, inv_T, k1, d1);
            plog_level(l + 1, log_T, inv_T, k2, d2);
            const double span = plog_ln_p(l + 1) - plog_ln_p(l);
            const double w = (ln_p - plog_ln_p(l)) / span;
            const double ln_k1 = Kokkos::log(k1), ln_k2 = Kokkos::log(k2);
            k = Kokkos::exp(ln_k1 + w * (ln_k2 - ln_k1));
            dlnk_dT = d1 + w * (d2 - d1);
            dlnk_dlnp = (ln_k2 - ln_k1) / span;
            return;
        }
        // Chebyshev: series in Tr = (2/T - 1/T_min - 1/T_max) / (1/T_max - 1/T_min) and
        // pr = (2 ln p - ln p_min - ln p_max) / (ln p_max - ln p_min), with the recurrences
        // phi_{n+1} = 2 x phi_n - phi_{n-1} and phi'_{n+1} = 2 phi_n + 2 x phi'_n - phi'_{n-1}
        const double a_T = chebyshev_range(i, 0), b_T = chebyshev_range(i, 1);
        const double a_p = chebyshev_range(i, 2), b_p = chebyshev_range(i, 3);
        const double Tr = (2.0 * inv_T - a_T - b_T) / (b_T - a_T);
        const double pr = (2.0 * ln_p - a_p - b_p) / (b_p - a_p);
        const uint32_t n_p = chebyshev_n_p(i), first = chebyshev_offset(i);
        const uint32_t n_T = (chebyshev_offset(i + 1) - first) / n_p;
        double log_k = 0.0, dlogk_dTr = 0.0, dlogk_dpr = 0.0;
        double phi_T = 1.0, phi_T_prev = 0.0, d_T = 0.0, d_T_prev = 0.0;
        for (uint32_t t = 0; t < n_T; t++) {
            double sum = 0.0, dsum = 0.0;
            double phi_p = 1.0, phi_p_prev = 0.0, d_p = 0.0, d_p_prev = 0.0;
            for (uint32_t q = 0; q < n_p; q++) {
                const double alpha = chebyshev(first + t * n_p + q);
                sum += alpha * phi_p;
                dsum += alpha * d_p;
                const double phi_next = q == 0 ? pr : 2.0 * pr * phi_p - phi_p_prev;
                const double d_next = q == 0 ? 1.0 : 2.0 * phi_p + 2.0 * pr * d_p - d_p_prev;
                phi_p_prev = phi_p;
                d_p_prev = d_p;
                phi_p = phi_next;
                d_p = d_next;
            }
            log_k += phi_T * sum;
            dlogk_dTr += d_T * sum;
            dlogk_dpr += phi_T * dsum;
            const double phi_next = t == 0 ? Tr : 2.0 * Tr * phi_T - phi_T_prev;
            const double d_next = t == 0 ? 1.0 : 2.0 * phi_T + 2.0 * Tr * d_T - d_T_prev;
            phi_T_prev = phi_T;
            d_T_prev = d_T;
            phi_T = phi_next;
            d_T = d_next;
        }
        k = Kokkos::pow(10.0, log_k);
        dlnk_dT = LN10 * dlogk_dTr * (-2.0 * inv_T * inv_T / (b_T - a_T));
        dlnk_dlnp = LN10 * dlogk_dpr * 2.0 / (b_p - a_p);
    }

    /** @brief Third-body concentration of reaction i. */
    KOKKOS_INLINE_FUNCTION
    double third_body(const uint32_t i, const double * C, const double C_total) const {
        double M = default_efficiency(i) * C_total;
        for (uint32_t e = efficiency_offset(i); e < efficiency_offset(i + 1); e++) {
            M += efficiency_extra(e) * C[efficiency_species(e)];
        }
        return M;
    }

    /**
     * @brief Falloff blending F(Pr, T) and its logarithmic derivatives
     *        d ln F / d ln Pr and d ln F / dT at fixed Pr.
     */
    KOKKOS_INLINE_FUNCTION
    void falloff_function(const uint32_t i, const double T, const double Pr, double & F, double & dlnF_dlnPr,
                          double & dlnF_dT) const {
        constexpr double SMALL = 1e-300;
        constexpr double LN10 = 2.302585092994045684;
        F = 1.0;
        dlnF_dlnPr = 0.0;
        dlnF_dT = 0.0;
        const double L = Kokkos::log10(Kokkos::fmax(Pr, SMALL));
        if (falloff(i) == static_cast<uint8_t>(FalloffType::TROE)) {
            const double A = falloff_params(i, 0), T3 = falloff_params(i, 1), T1 = falloff_params(i, 2),
                         T2 = falloff_params(i, 3);
            const double e3 = Kokkos::fabs(T3) > SMALL ? Kokkos::exp(-T / T3) : 0.0;
            const double e1 = Kokkos::fabs(T1) > SMALL ? Kokkos::exp(-T / T1) : 0.0;
            const double e2 = T2 != 0.0 ? Kokkos::exp(-T2 / T) : 0.0;
            const double Fcent = (1.0 - A) * e3 + A * e1 + e2;
            double dFcent_dT = e2 * T2 / (T * T);
            if (Kokkos::fabs(T3) > SMALL) dFcent_dT -= (1.0 - A) * e3 / T3;
            if (Kokkos::fabs(T1) > SMALL) dFcent_dT -= A * e1 / T1;
            const double Lc = Kokkos::log10(Kokkos::fmax(Fcent, SMALL));
            const double c = -0.4 - 0.67 * Lc;
            const double n = 0.75 - 1.27 * Lc;
            const double x = L + c;
            const double D = n - 0.14 * x;
            const double f1 = x / D;
            const double g = 1.0 / (1.0 + f1 * f1);
            const double logF = Lc * g;
            F = Kokkos::pow(10.0, logF);
            // d f1 / dL at fixed Lc, and d f1 / dLc at fixed L
            const double df1_dL = n / (D * D);
            const double df1_dLc = (-0.67 * D - x * (-1.27 + 0.14 * 0.67)) / (D * D);
            const double dlogF_dL = -Lc * 2.0 * f1 * df1_dL * g * g;
            const double dlogF_dLc = g - Lc * 2.0 * f1 * df1_dLc * g * g;
            dlnF_dlnPr = dlogF_dL;  // d log10 F / d log10 Pr = d ln F / d ln Pr
            const double dLc_dT = Fcent > SMALL ? dFcent_dT / (Fcent * LN10) : 0.0;
            dlnF_dT = LN10 * dlogF_dLc * dLc_dT;
        } else if (falloff(i) == static_cast<uint8_t>(FalloffType::SRI)) {
            const double a = falloff_params(i, 0), b = falloff_params(i, 1), c = falloff_params(i, 2),
                         d = falloff_params(i, 3), e = falloff_params(i, 4);
            const double X = 1.0 / (1.0 + L * L);
            const double ea = a * Kokkos::exp(-b / T), ec = Kokkos::exp(-T / c);
            const double Z = ea + ec;
            F = d * Kokkos::pow(Z, X) * Kokkos::pow(T, e);
            dlnF_dlnPr = Kokkos::log(Z) * (-2.0 * L * X * X) / LN10;  // dX/dL with L = log10 Pr
            dlnF_dT = X * (ea * b / (T * T) - ec / c) / Z + e / T;
        }
    }

    /** @brief Doubles of ReactionDerivatives storage. */
    KOKKOS_INLINE_FUNCTION uint32_t derivatives_size() const {
        return 3 * n_reactions + static_cast<uint32_t>(forward_species.extent(0)) +
               static_cast<uint32_t>(reverse_species.extent(0));
    }

    /**
     * @brief Rates of progress q_i [kmol/(m^3 s)] of every reaction (one per
     *        lane) and, with derivatives, what the Jacobian needs.
     * @param C Concentrations (n_species), C_total their sum.
     * @param g_RT, h_RT Species g / RT and h / RT at T (h_RT only with derivatives).
     * @param q Rates of progress (n_reactions).
     * @param d Null, or the derivatives (laid out by ReactionDerivatives::at).
     */
    template <typename Lanes>
    KOKKOS_INLINE_FUNCTION void rates_of_progress(const Lanes & lanes, const double T, const double * C,
                                                  const double C_total, const double * g_RT, const double * h_RT,
                                                  double * q, const ReactionDerivatives * d) const {
        const double log_T = Kokkos::log(T), inv_T = 1.0 / T;
        const double log_c0 = Kokkos::log(ONE_ATM / (GAS_CONSTANT * T));
        lanes.for_each(n_reactions, [&](const uint32_t i) {
            double kf = arrhenius(rate(i, 0), rate(i, 1), rate(i, 2), log_T, inv_T);
            double dlnkf_dT = (rate(i, 1) + rate(i, 2) * inv_T) * inv_T;
            double dlnk_dlnp = 0.0, dk_dM = 0.0;
            const uint8_t kind = type(i);
            if (kind == static_cast<uint8_t>(ReactionType::PLOG) ||
                kind == static_cast<uint8_t>(ReactionType::CHEBYSHEV)) {
                // p = C_total R T, so d ln p / dT = 1 / T at fixed concentrations
                pressure_rate(i, log_T, inv_T, Kokkos::log(C_total * GAS_CONSTANT * T), kf, dlnkf_dT, dlnk_dlnp);
                dlnkf_dT += dlnk_dlnp * inv_T;
            } else if (kind == static_cast<uint8_t>(ReactionType::THREE_BODY)) {
                const double M = third_body(i, C, C_total);
                dk_dM = kf;
                kf *= M;
            } else if (kind == static_cast<uint8_t>(ReactionType::FALLOFF)) {
                const double k0 = arrhenius(low(i, 0), low(i, 1), low(i, 2), log_T, inv_T);
                const double dlnk0_dT = (low(i, 1) + low(i, 2) * inv_T) * inv_T;
                const double M = third_body(i, C, C_total);
                const double Pr = kf > 0.0 ? k0 * M / kf : 0.0;
                double F, dlnF_dlnPr, dlnF_dT;
                falloff_function(i, T, Pr, F, dlnF_dlnPr, dlnF_dT);
                // ln k = ln k_inf + ln Pr - ln(1 + Pr) + ln F, with d ln Pr / dT = dlnk0 - dlnkinf
                const double dlnPr_dT = dlnk0_dT - dlnkf_dT;
                dlnkf_dT += (1.0 / (1.0 + Pr) + dlnF_dlnPr) * dlnPr_dT + dlnF_dT;
                // k = k_inf Pr / (1 + Pr) F; dk/dM = k0 F [1 / (1 + Pr)^2 + d ln F / d ln Pr / (1 + Pr)]
                dk_dM = k0 * F / (1.0 + Pr) * (1.0 / (1.0 + Pr) + dlnF_dlnPr);
                kf *= Pr / (1.0 + Pr) * F;
            }
            // The factors C^order, kept for the derivatives (the first MAX_KEPT of each product)
            double kept_f[MAX_KEPT] = {}, kept_r[MAX_KEPT] = {};
            double prod_f = 1.0;
            for (uint32_t j = forward_offset(i); j < forward_offset(i + 1); j++) {
                const double factor = power(C[forward_species(j)], forward_order(j), C_reg);
                if (j - forward_offset(i) < MAX_KEPT) kept_f[j - forward_offset(i)] = factor;
                prod_f *= factor;
            }
            double kr = 0.0, prod_r = 0.0, dlnKc_dT = 0.0;
            if (reversible(i)) {
                double sum_g = 0.0, sum_h = 0.0;
                for (uint32_t j = net_offset(i); j < net_offset(i + 1); j++) {
                    sum_g += net_nu(j) * g_RT[net_species(j)];
                    if (d) sum_h += net_nu(j) * h_RT[net_species(j)];
                }
                const double ln_Kc = -sum_g + delta_nu(i) * log_c0;
                const double inv_Kc = Kokkos::exp(-ln_Kc);
                kr = kf * inv_Kc;
                prod_r = inv_Kc;
                for (uint32_t j = reverse_offset(i); j < reverse_offset(i + 1); j++) {
                    const double factor = power(C[reverse_species(j)], reverse_order(j), C_reg);
                    if (j - reverse_offset(i) < MAX_KEPT) kept_r[j - reverse_offset(i)] = factor;
                    prod_r *= factor;
                }
                // d(g/RT)/dT = -h/(R T^2), d ln(p_atm / RT) / dT = -1/T
                dlnKc_dT = (sum_h - delta_nu(i)) * inv_T;
            }
            const double fwd = kf * prod_f, rev = kf * prod_r;
            q[i] = fwd - rev;
            if (!d) return;
            d->dq_dT[i] = fwd * dlnkf_dT - rev * (dlnkf_dT - dlnKc_dT);
            // Mass-action terms: kf d(prod C^o)/dC_j and kr d(prod C^nu'')/dC_j
            for (uint32_t a = forward_offset(i); a < forward_offset(i + 1); a++) {
                double dd = kf * power_derivative(C[forward_species(a)], forward_order(a), C_reg);
                for (uint32_t b = forward_offset(i); b < forward_offset(i + 1); b++) {
                    if (b == a) continue;
                    dd *= b - forward_offset(i) < MAX_KEPT ? kept_f[b - forward_offset(i)]
                                                           : power(C[forward_species(b)], forward_order(b), C_reg);
                }
                d->d_forward[a] = dd;
            }
            for (uint32_t a = reverse_offset(i); a < reverse_offset(i + 1); a++) {
                double dd = 0.0;
                if (reversible(i)) {
                    dd = kr * power_derivative(C[reverse_species(a)], reverse_order(a), C_reg);
                    for (uint32_t b = reverse_offset(i); b < reverse_offset(i + 1); b++) {
                        if (b == a) continue;
                        dd *= b - reverse_offset(i) < MAX_KEPT ? kept_r[b - reverse_offset(i)]
                                                               : power(C[reverse_species(b)], reverse_order(b), C_reg);
                    }
                }
                d->d_reverse[a] = dd;
            }
            // Through [M]: d q / d C_j = dq_dM eff_j; through p: d q / d C_j = q (d ln k / d ln p) / C_total
            d->dq_dM[i] = dk_dM * (prod_f - prod_r);
            d->dq_uniform[i] = dlnk_dlnp != 0.0 && C_total > 0.0 ? q[i] * dlnk_dlnp / C_total : 0.0;
        });
        lanes.sync();
    }

    /** @brief Rates of progress of one thread, without derivatives. */
    KOKKOS_INLINE_FUNCTION
    void rates_of_progress(const double T, const double * C, const double * g_RT, double * q) const {
        double C_total = 0.0;
        for (uint32_t k = 0; k < n_species; k++) C_total += C[k];
        rates_of_progress(SerialLanes(), T, C, C_total, g_RT, nullptr, q, nullptr);
    }

    /**
     * @brief Net production rates omega_k = sum_i nu_ki q_i [kmol/(m^3 s)],
     *        one species per lane, summed by reaction index (or the same
     *        sums of any per-reaction quantity, e.g. dq_dT).
     */
    template <typename Lanes>
    KOKKOS_INLINE_FUNCTION void production_rates(const Lanes & lanes, const double * q, double * omega,
                                                 double * partial = nullptr) const {
        if constexpr (Lanes::parallel) {
            if (partial) {
                // Chunks of a few reactions per lane, then each species' chunks in order
                lanes.for_each(static_cast<uint32_t>(chunk_end.extent(0)), [&](const uint32_t c) {
                    const uint32_t begin = c == 0 ? 0 : chunk_end(c - 1);
                    double sum = 0.0;
                    for (uint32_t e = begin; e < chunk_end(c); e++) sum += species_nu(e) * q[species_reaction(e)];
                    partial[c] = sum;
                });
                lanes.sync();
                lanes.for_each(n_species, [&](const uint32_t k) {
                    double sum = 0.0;
                    for (uint32_t c = chunk_offset(k); c < chunk_offset(k + 1); c++) sum += partial[c];
                    omega[k] = sum;
                });
                lanes.sync();
                return;
            }
        }
        lanes.for_each(n_species, [&](const uint32_t k) {
            double sum = 0.0;
            for (uint32_t e = species_offset(k); e < species_offset(k + 1); e++) {
                sum += species_nu(e) * q[species_reaction(e)];
            }
            omega[k] = sum;
        });
        lanes.sync();
    }

    KOKKOS_INLINE_FUNCTION
    void production_rates(const double * q, double * omega) const { production_rates(SerialLanes(), q, omega); }

    /**
     * @brief The structural entries of the Jacobian of the production rates
     *        (d omega_k / d C_j at fixed T, without the part shared by all
     *        columns), values[e] for entry e = (entry_row(e), entry_column(e)),
     *        one entry per lane; and that shared part of each row in all_columns.
     */
    template <typename Lanes>
    KOKKOS_INLINE_FUNCTION void production_jacobian_entries(const Lanes & lanes, const ReactionDerivatives & d,
                                                            double * values, double * all_columns) const {
        lanes.for_each(n_species, [&](const uint32_t k) {
            double uniform = 0.0;
            for (uint32_t e = species_offset(k); e < species_offset(k + 1); e++) {
                const uint32_t i = species_reaction(e);
                uniform += species_nu(e) * (d.dq_dM[i] * default_efficiency(i) + d.dq_uniform[i]);
            }
            all_columns[k] = uniform;
        });
        const double * block = d.dq_dT;
        lanes.for_each(static_cast<uint32_t>(entry_row.extent(0)), [&](const uint32_t e) {
            double sum = 0.0;
            for (uint32_t t = entry_offset(e); t < entry_offset(e + 1); t++) {
                sum += term_coefficient(t) * block[term_source(t)];
            }
            values[e] = sum;
        });
        lanes.sync();
    }

    /**
     * @brief Jacobian of the production rates with respect to the
     *        concentrations at fixed T, one row per lane:
     *        J[dense_index(stride, k, j)] = d omega_k / d C_j for j < n_species
     *        (by rows, or by columns for lanes with Lanes::column_major).
     * @param d Derivatives from rates_of_progress at the same state.
     * @param all_columns If not null, the part of row k that is the same in
     *        every column (third bodies at their default efficiency, PLOG
     *        and Chebyshev pressures) goes to all_columns[k] instead of J.
     */
    template <typename Lanes>
    KOKKOS_INLINE_FUNCTION void production_jacobian(const Lanes & lanes, const ReactionDerivatives & d, double * J,
                                                    const uint32_t stride, double * all_columns = nullptr) const {
        const uint32_t n = n_species;
        constexpr bool CM = Lanes::column_major;
        if constexpr (Lanes::parallel) {
            // Rows start from the part shared by all columns; then each lane adds whole entries, so
            // that the lanes' loads balance (a row's reactions do not)
            lanes.for_each(n, [&](const uint32_t k) {
                double uniform = 0.0;
                for (uint32_t e = species_offset(k); e < species_offset(k + 1); e++) {
                    const uint32_t i = species_reaction(e);
                    uniform += species_nu(e) * (d.dq_dM[i] * default_efficiency(i) + d.dq_uniform[i]);
                }
                if (all_columns) all_columns[k] = uniform;
                const double base = all_columns ? 0.0 : uniform;
                for (uint32_t j = 0; j < n; j++) J[dense_index<CM>(stride, k, j)] = base;
            });
            lanes.sync();
            const double * block = d.dq_dT;
            lanes.for_each(static_cast<uint32_t>(entry_row.extent(0)), [&](const uint32_t e) {
                double sum = 0.0;
                for (uint32_t t = entry_offset(e); t < entry_offset(e + 1); t++) {
                    sum += term_coefficient(t) * block[term_source(t)];
                }
                J[dense_index<CM>(stride, entry_row(e), entry_column(e))] += sum;
            });
            lanes.sync();
            return;
        }
        lanes.for_each(n, [&](const uint32_t k) {
            auto row = [&](const uint32_t j) -> double & { return J[dense_index<CM>(stride, k, j)]; };
            for (uint32_t j = 0; j < n; j++) row(j) = 0.0;
            double uniform = 0.0;
            for (uint32_t e = species_offset(k); e < species_offset(k + 1); e++) {
                const uint32_t i = species_reaction(e);
                const double nu = species_nu(e);
                for (uint32_t a = forward_offset(i); a < forward_offset(i + 1); a++) {
                    row(forward_species(a)) += nu * d.d_forward[a];
                }
                for (uint32_t a = reverse_offset(i); a < reverse_offset(i + 1); a++) {
                    row(reverse_species(a)) -= nu * d.d_reverse[a];
                }
                const double all = nu * (d.dq_dM[i] * default_efficiency(i) + d.dq_uniform[i]);
                if (all_columns) {
                    uniform += all;
                } else if (all != 0.0) {
                    for (uint32_t j = 0; j < n; j++) row(j) += all;
                }
                if (d.dq_dM[i] != 0.0) {
                    for (uint32_t x = efficiency_offset(i); x < efficiency_offset(i + 1); x++) {
                        row(efficiency_species(x)) += nu * d.dq_dM[i] * efficiency_extra(x);
                    }
                }
            }
            if (all_columns) all_columns[k] = uniform;
        });
        lanes.sync();
    }
};

/**
 * @brief Copy a mechanism's reactions to MemorySpace.
 * @param C_reg Concentration of the regularization of orders 0 < n < 1 (KineticsTable::C_REG by default).
 */
template <typename MemorySpace = Kokkos::DefaultExecutionSpace::memory_space>
KineticsTable<MemorySpace> make_kinetics_table(const Mechanism & mechanism,
                                               const double C_reg = KineticsTable<MemorySpace>::C_REG) {
    using Table = KineticsTable<MemorySpace>;
    Table t;
    t.C_reg = C_reg;
    t.n_species = mechanism.n_species();
    t.n_reactions = static_cast<uint32_t>(mechanism.reactions.size());
    const uint32_t nr = t.n_reactions;
    auto u32 = [](size_t n) { return static_cast<uint32_t>(n); };
    std::vector<uint8_t> type(nr), falloff(nr), reversible(nr);
    std::vector<double> rate(3 * nr), low(3 * nr), params(5 * nr), delta_nu(nr), default_eff(nr);
    std::vector<uint32_t> f_off{0}, r_off{0}, n_off{0}, e_off{0}, f_sp, r_sp, n_sp, e_sp;
    std::vector<double> f_ord, r_ord, n_nu, e_extra;
    std::vector<uint32_t> plog_off{0}, plog_rate_off{0}, cheb_off{0}, cheb_n_p(nr, 1);
    std::vector<double> plog_ln_p, plog_rate, cheb, cheb_range(4 * nr, 0.0);
    for (uint32_t i = 0; i < nr; i++) {
        const Reaction & r = mechanism.reactions[i];
        for (size_t e = 0; e < r.plog.size(); e++) {
            if (e == 0 || r.plog[e].first != r.plog[e - 1].first) {
                if (e > 0) plog_rate_off.push_back(u32(plog_rate.size() / 3));
                plog_ln_p.push_back(std::log(r.plog[e].first));
            }
            plog_rate.insert(plog_rate.end(), {r.plog[e].second.A, r.plog[e].second.b, r.plog[e].second.Ea_R});
        }
        if (!r.plog.empty()) plog_rate_off.push_back(u32(plog_rate.size() / 3));
        plog_off.push_back(u32(plog_ln_p.size()));
        if (r.type == ReactionType::CHEBYSHEV) {
            cheb.insert(cheb.end(), r.chebyshev.begin(), r.chebyshev.end());
            cheb_n_p[i] = r.chebyshev_n_p;
            cheb_range[4 * i] = 1.0 / r.chebyshev_range[0];
            cheb_range[4 * i + 1] = 1.0 / r.chebyshev_range[1];
            cheb_range[4 * i + 2] = std::log(r.chebyshev_range[2]);
            cheb_range[4 * i + 3] = std::log(r.chebyshev_range[3]);
        }
        cheb_off.push_back(u32(cheb.size()));
        type[i] = static_cast<uint8_t>(r.type);
        falloff[i] = static_cast<uint8_t>(r.falloff);
        reversible[i] = r.reversible;
        const Arrhenius arr[2] = {r.rate, r.low};
        for (int a = 0; a < 2; a++) {
            std::vector<double> & dst = a ? low : rate;
            dst[3 * i] = arr[a].A;
            dst[3 * i + 1] = arr[a].b;
            dst[3 * i + 2] = arr[a].Ea_R;
        }
        for (uint32_t p = 0; p < 5; p++) params[5 * i + p] = r.falloff_params[p];
        for (const auto & [k, order] : r.orders) {
            f_sp.push_back(k);
            f_ord.push_back(order);
        }
        f_off.push_back(u32(f_sp.size()));
        for (const auto & [k, nu] : r.products) {
            r_sp.push_back(k);
            r_ord.push_back(nu);
        }
        r_off.push_back(u32(r_sp.size()));
        std::vector<double> net(t.n_species, 0.0);
        for (const auto & [k, nu] : r.products) net[k] += nu;
        for (const auto & [k, nu] : r.reactants) net[k] -= nu;
        double sum = 0.0;
        for (uint32_t k = 0; k < t.n_species; k++) {
            if (net[k] == 0.0) continue;
            n_sp.push_back(k);
            n_nu.push_back(net[k]);
            sum += net[k];
        }
        n_off.push_back(u32(n_sp.size()));
        delta_nu[i] = sum;
        default_eff[i] = r.default_efficiency;
        for (const auto & [k, eff] : r.efficiencies) {
            e_sp.push_back(k);
            e_extra.push_back(eff - r.default_efficiency);
        }
        e_off.push_back(u32(e_sp.size()));
    }
    auto copy = [](const auto & v, const char * label) {
        using T = typename std::decay_t<decltype(v)>::value_type;
        Kokkos::View<T *, MemorySpace> d(label, v.size());
        auto h = Kokkos::create_mirror_view(d);
        for (size_t a = 0; a < v.size(); a++) h(a) = v[a];
        Kokkos::deep_copy(d, h);
        return d;
    };
    auto copy2 = [&](const std::vector<double> & v, auto & dst, const char * label) {
        using V = std::decay_t<decltype(dst)>;
        constexpr size_t W = V::static_extent(1);
        dst = V(label, v.size() / W);
        auto h = Kokkos::create_mirror_view(dst);
        for (size_t a = 0; a < v.size() / W; a++) {
            for (size_t b = 0; b < W; b++) h(a, b) = v[a * W + b];
        }
        Kokkos::deep_copy(dst, h);
    };
    t.type = copy(type, "kinetics_type");
    t.falloff = copy(falloff, "kinetics_falloff");
    t.reversible = copy(reversible, "kinetics_reversible");
    copy2(rate, t.rate, "kinetics_rate");
    copy2(low, t.low, "kinetics_low");
    copy2(params, t.falloff_params, "kinetics_falloff_params");
    t.forward_offset = copy(f_off, "kinetics_forward_offset");
    t.forward_species = copy(f_sp, "kinetics_forward_species");
    t.forward_order = copy(f_ord, "kinetics_forward_order");
    t.reverse_offset = copy(r_off, "kinetics_reverse_offset");
    t.reverse_species = copy(r_sp, "kinetics_reverse_species");
    t.reverse_order = copy(r_ord, "kinetics_reverse_order");
    t.net_offset = copy(n_off, "kinetics_net_offset");
    t.net_species = copy(n_sp, "kinetics_net_species");
    t.net_nu = copy(n_nu, "kinetics_net_nu");
    // Net coefficients by species, in reaction order
    std::vector<uint32_t> s_off(t.n_species + 1, 0), s_re(n_sp.size());
    std::vector<double> s_nu(n_sp.size());
    for (uint32_t k : n_sp) s_off[k + 1]++;
    for (uint32_t k = 0; k < t.n_species; k++) s_off[k + 1] += s_off[k];
    std::vector<uint32_t> fill(s_off.begin(), s_off.end() - 1);
    for (uint32_t i = 0; i < nr; i++) {
        for (uint32_t j = n_off[i]; j < n_off[i + 1]; j++) {
            s_re[fill[n_sp[j]]] = i;
            s_nu[fill[n_sp[j]]++] = n_nu[j];
        }
    }
    // Chunks of at most 4 coefficients of each species
    {
        std::vector<uint32_t> offsets{0}, ends;
        for (uint32_t k = 0; k < t.n_species; k++) {
            for (uint32_t e = s_off[k]; e < s_off[k + 1]; e += 4) ends.push_back(std::min(e + 4, s_off[k + 1]));
            offsets.push_back(static_cast<uint32_t>(ends.size()));
        }
        t.chunk_offset = copy(offsets, "kinetics_chunk_offset");
        t.chunk_end = copy(ends, "kinetics_chunk_end");
    }
    // Jacobian entries and their terms, in the order of production_jacobian's row loops
    {
        const uint32_t nf = static_cast<uint32_t>(f_sp.size()), nrv = static_cast<uint32_t>(r_sp.size());
        const uint32_t off_forward = nr, off_reverse = nr + nf, off_dM = nr + nf + nrv;
        std::vector<std::vector<std::pair<uint32_t, double>>> terms(static_cast<size_t>(t.n_species) * t.n_species);
        for (uint32_t k = 0; k < t.n_species; k++) {
            for (uint32_t e = s_off[k]; e < s_off[k + 1]; e++) {
                const uint32_t i = s_re[e];
                const double nu = s_nu[e];
                auto & row = terms;
                for (uint32_t a = f_off[i]; a < f_off[i + 1]; a++) {
                    row[k * t.n_species + f_sp[a]].emplace_back(off_forward + a, nu);
                }
                if (reversible[i]) {
                    for (uint32_t a = r_off[i]; a < r_off[i + 1]; a++) {
                        row[k * t.n_species + r_sp[a]].emplace_back(off_reverse + a, -nu);
                    }
                }
                for (uint32_t x = e_off[i]; x < e_off[i + 1]; x++) {
                    row[k * t.n_species + e_sp[x]].emplace_back(off_dM + i, nu * e_extra[x]);
                }
            }
        }
        std::vector<uint32_t> rows, columns, offsets{0}, sources;
        std::vector<double> coefficients;
        for (uint32_t k = 0; k < t.n_species; k++) {
            for (uint32_t j = 0; j < t.n_species; j++) {
                const auto & list = terms[k * t.n_species + j];
                if (list.empty()) continue;
                rows.push_back(k);
                columns.push_back(j);
                for (const auto & [source, coefficient] : list) {
                    sources.push_back(source);
                    coefficients.push_back(coefficient);
                }
                offsets.push_back(static_cast<uint32_t>(sources.size()));
            }
        }
        t.entry_row = copy(rows, "kinetics_entry_row");
        t.entry_column = copy(columns, "kinetics_entry_column");
        t.entry_offset = copy(offsets, "kinetics_entry_offset");
        t.term_source = copy(sources, "kinetics_term_source");
        t.term_coefficient = copy(coefficients, "kinetics_term_coefficient");
    }
    t.species_offset = copy(s_off, "kinetics_species_offset");
    t.species_reaction = copy(s_re, "kinetics_species_reaction");
    t.species_nu = copy(s_nu, "kinetics_species_nu");
    t.delta_nu = copy(delta_nu, "kinetics_delta_nu");
    t.efficiency_offset = copy(e_off, "kinetics_efficiency_offset");
    t.efficiency_species = copy(e_sp, "kinetics_efficiency_species");
    t.efficiency_extra = copy(e_extra, "kinetics_efficiency_extra");
    t.default_efficiency = copy(default_eff, "kinetics_default_efficiency");
    t.plog_offset = copy(plog_off, "kinetics_plog_offset");
    t.plog_ln_p = copy(plog_ln_p, "kinetics_plog_ln_p");
    t.plog_rate_offset = copy(plog_rate_off, "kinetics_plog_rate_offset");
    copy2(plog_rate, t.plog_rate, "kinetics_plog_rate");
    t.chebyshev_offset = copy(cheb_off, "kinetics_chebyshev_offset");
    t.chebyshev_n_p = copy(cheb_n_p, "kinetics_chebyshev_n_p");
    t.chebyshev = copy(cheb, "kinetics_chebyshev");
    copy2(cheb_range, t.chebyshev_range, "kinetics_chebyshev_range");
    return t;
}

} // namespace chemistry

#endif // CHEMISTRY_KINETICS_H

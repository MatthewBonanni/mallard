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

#include "mechanism.h"

namespace chemistry {

/**
 * @brief Reactions of a mechanism as flat device tables (compressed rows per
 *        reaction), in MemorySpace. All quantities SI with kmol, in double
 *        precision in every build. Concentrations C_k = rho Y_k / W_k.
 *
 * Rates follow Cantera: k = A T^b exp(-Ea / RT); three-body reactions
 * multiply both directions by [M] = sum_k eff_k C_k; falloff reactions use
 * k = k_inf Pr / (1 + Pr) F with Pr = k_0 [M] / k_inf and F = 1 (Lindemann),
 * Troe or SRI; reversible reactions take k_r = k_f / K_c with
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
    View1<double> delta_nu;          // sum of net coefficients
    View1<uint32_t> efficiency_offset;
    View1<uint32_t> efficiency_species;
    View1<double> efficiency_extra;  // efficiency minus the default
    View1<double> default_efficiency;

    /** @brief C^order: repeated products for integer orders, pow of max(C, 0) otherwise. */
    KOKKOS_INLINE_FUNCTION
    static double power(const double C, const double order) {
        const double whole = Kokkos::floor(order);
        if (whole == order && order >= 0.0 && order <= 4.0) {
            double p = 1.0;
            for (int i = 0; i < static_cast<int>(order); i++) p *= C;
            return p;
        }
        return Kokkos::pow(Kokkos::fmax(C, 0.0), order);
    }

    /** @brief d(C^order)/dC. */
    KOKKOS_INLINE_FUNCTION
    static double power_derivative(const double C, const double order) {
        if (order == 0.0) return 0.0;
        return order * power(C, order - 1.0);
    }

    KOKKOS_INLINE_FUNCTION
    static double arrhenius(const double A, const double b, const double Ea_R, const double log_T, const double inv_T) {
        return A * Kokkos::exp(b * log_T - Ea_R * inv_T);
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

    /**
     * @brief Rates of progress q_i [kmol/(m^3 s)] and, if dq_dT is not null,
     *        their temperature derivatives at fixed concentrations.
     * @param C Concentrations (n_species).
     * @param g_RT, h_RT Species g / RT and h / RT at T (h_RT only for dq_dT).
     * @param q Rates of progress (n_reactions).
     * @param kf, kr Forward and reverse rate constants including third-body
     *        and falloff factors (n_reactions), for the Jacobian; may be null.
     */
    KOKKOS_INLINE_FUNCTION
    void rates_of_progress(const double T, const double * C, const double * g_RT, const double * h_RT, double * q,
                           double * dq_dT, double * kf_out, double * kr_out) const {
        const double log_T = Kokkos::log(T), inv_T = 1.0 / T;
        const double log_c0 = Kokkos::log(ONE_ATM / (GAS_CONSTANT * T));
        double C_total = 0.0;
        for (uint32_t k = 0; k < n_species; k++) C_total += C[k];
        for (uint32_t i = 0; i < n_reactions; i++) {
            double kf = arrhenius(rate(i, 0), rate(i, 1), rate(i, 2), log_T, inv_T);
            double dlnkf_dT = (rate(i, 1) + rate(i, 2) * inv_T) * inv_T;
            if (type(i) == static_cast<uint8_t>(ReactionType::THREE_BODY)) {
                kf *= third_body(i, C, C_total);
            } else if (type(i) == static_cast<uint8_t>(ReactionType::FALLOFF)) {
                const double k0 = arrhenius(low(i, 0), low(i, 1), low(i, 2), log_T, inv_T);
                const double dlnk0_dT = (low(i, 1) + low(i, 2) * inv_T) * inv_T;
                const double M = third_body(i, C, C_total);
                const double Pr = kf > 0.0 ? k0 * M / kf : 0.0;
                double F, dlnF_dlnPr, dlnF_dT;
                falloff_function(i, T, Pr, F, dlnF_dlnPr, dlnF_dT);
                // ln k = ln k_inf + ln Pr - ln(1 + Pr) + ln F, with d ln Pr / dT = dlnk0 - dlnkinf
                const double dlnPr_dT = dlnk0_dT - dlnkf_dT;
                dlnkf_dT += (1.0 / (1.0 + Pr) + dlnF_dlnPr) * dlnPr_dT + dlnF_dT;
                kf *= Pr / (1.0 + Pr) * F;
            }
            double fwd = kf;
            for (uint32_t j = forward_offset(i); j < forward_offset(i + 1); j++) {
                fwd *= power(C[forward_species(j)], forward_order(j));
            }
            double kr = 0.0, rev = 0.0, dlnKc_dT = 0.0;
            if (reversible(i)) {
                double sum_g = 0.0, sum_h = 0.0;
                for (uint32_t j = net_offset(i); j < net_offset(i + 1); j++) {
                    sum_g += net_nu(j) * g_RT[net_species(j)];
                    if (dq_dT) sum_h += net_nu(j) * h_RT[net_species(j)];
                }
                const double ln_Kc = -sum_g + delta_nu(i) * log_c0;
                kr = kf * Kokkos::exp(-ln_Kc);
                rev = kr;
                for (uint32_t j = reverse_offset(i); j < reverse_offset(i + 1); j++) {
                    rev *= power(C[reverse_species(j)], reverse_order(j));
                }
                // d(g/RT)/dT = -h/(R T^2), d ln(p_atm / RT) / dT = -1/T
                dlnKc_dT = (sum_h - delta_nu(i)) * inv_T;
            }
            q[i] = fwd - rev;
            if (dq_dT) dq_dT[i] = fwd * dlnkf_dT - rev * (dlnkf_dT - dlnKc_dT);
            if (kf_out) kf_out[i] = kf;
            if (kr_out) kr_out[i] = kr;
        }
    }

    /** @brief Net production rates omega_k = sum_i nu_ki q_i [kmol/(m^3 s)]. */
    KOKKOS_INLINE_FUNCTION
    void production_rates(const double * q, double * omega) const {
        for (uint32_t k = 0; k < n_species; k++) omega[k] = 0.0;
        for (uint32_t i = 0; i < n_reactions; i++) {
            for (uint32_t j = net_offset(i); j < net_offset(i + 1); j++) omega[net_species(j)] += net_nu(j) * q[i];
        }
    }

    /**
     * @brief Jacobian of the production rates with respect to the
     *        concentrations at fixed T: dw_dC[k * n_species + j] = d omega_k / d C_j.
     * @param kf, kr Rate constants from rates_of_progress at the same state.
     */
    KOKKOS_INLINE_FUNCTION
    void production_jacobian(const double T, const double * C, const double * kf, const double * kr,
                             double * dw_dC) const {
        const uint32_t n = n_species;
        for (uint32_t a = 0; a < n * n; a++) dw_dC[a] = 0.0;
        const double log_T = Kokkos::log(T), inv_T = 1.0 / T;
        double C_total = 0.0;
        for (uint32_t k = 0; k < n; k++) C_total += C[k];
        for (uint32_t i = 0; i < n_reactions; i++) {
            const uint32_t net_begin = net_offset(i), net_end = net_offset(i + 1);
            auto add = [&](const uint32_t j, const double dq) {
                for (uint32_t m = net_begin; m < net_end; m++) dw_dC[net_species(m) * n + j] += net_nu(m) * dq;
            };
            // Mass-action terms: kf d(prod C^o)/dC_j - kr d(prod C^nu'')/dC_j
            for (uint32_t a = forward_offset(i); a < forward_offset(i + 1); a++) {
                double d = kf[i] * power_derivative(C[forward_species(a)], forward_order(a));
                for (uint32_t b = forward_offset(i); b < forward_offset(i + 1); b++) {
                    if (b != a) d *= power(C[forward_species(b)], forward_order(b));
                }
                add(forward_species(a), d);
            }
            if (reversible(i)) {
                for (uint32_t a = reverse_offset(i); a < reverse_offset(i + 1); a++) {
                    double d = kr[i] * power_derivative(C[reverse_species(a)], reverse_order(a));
                    for (uint32_t b = reverse_offset(i); b < reverse_offset(i + 1); b++) {
                        if (b != a) d *= power(C[reverse_species(b)], reverse_order(b));
                    }
                    add(reverse_species(a), -d);
                }
            }
            if (type(i) == static_cast<uint8_t>(ReactionType::ELEMENTARY)) continue;
            // Third-body terms: d q / d C_j = (d ln k / d M) q eff_j, with q = k (prod_f - prod_r / Kc)
            double prod_f = 1.0, prod_r = 0.0;
            for (uint32_t a = forward_offset(i); a < forward_offset(i + 1); a++) {
                prod_f *= power(C[forward_species(a)], forward_order(a));
            }
            if (reversible(i)) {
                prod_r = kf[i] > 0.0 ? kr[i] / kf[i] : 0.0;
                for (uint32_t a = reverse_offset(i); a < reverse_offset(i + 1); a++) {
                    prod_r *= power(C[reverse_species(a)], reverse_order(a));
                }
            }
            const double M = third_body(i, C, C_total);
            double dk_dM;
            if (type(i) == static_cast<uint8_t>(ReactionType::THREE_BODY)) {
                dk_dM = M != 0.0 ? kf[i] / M : arrhenius(rate(i, 0), rate(i, 1), rate(i, 2), log_T, inv_T);
            } else {
                const double k_inf = arrhenius(rate(i, 0), rate(i, 1), rate(i, 2), log_T, inv_T);
                const double k0 = arrhenius(low(i, 0), low(i, 1), low(i, 2), log_T, inv_T);
                const double Pr = k_inf > 0.0 ? k0 * M / k_inf : 0.0;
                double F, dlnF_dlnPr, dlnF_dT;
                falloff_function(i, T, Pr, F, dlnF_dlnPr, dlnF_dT);
                // k = k_inf Pr / (1 + Pr) F; dk/dM = k0 F [1 / (1 + Pr)^2 + Pr / (1 + Pr) d ln F / d ln Pr / Pr]
                dk_dM = k0 * F / (1.0 + Pr) * (1.0 / (1.0 + Pr) + dlnF_dlnPr);
            }
            const double dq_dM = dk_dM * (prod_f - prod_r);
            if (dq_dM == 0.0) continue;
            const double base = default_efficiency(i);
            if (base != 0.0) {
                for (uint32_t j = 0; j < n; j++) add(j, base * dq_dM);
            }
            for (uint32_t e = efficiency_offset(i); e < efficiency_offset(i + 1); e++) {
                add(efficiency_species(e), efficiency_extra(e) * dq_dM);
            }
        }
    }
};

/** @brief Copy a mechanism's reactions to MemorySpace. */
template <typename MemorySpace = Kokkos::DefaultExecutionSpace::memory_space>
KineticsTable<MemorySpace> make_kinetics_table(const Mechanism & mechanism) {
    using Table = KineticsTable<MemorySpace>;
    Table t;
    t.n_species = static_cast<uint32_t>(mechanism.n_species());
    t.n_reactions = static_cast<uint32_t>(mechanism.reactions.size());
    const uint32_t nr = t.n_reactions;
    std::vector<uint8_t> type(nr), falloff(nr), reversible(nr);
    std::vector<double> rate(3 * nr), low(3 * nr), params(5 * nr), delta_nu(nr), default_eff(nr);
    std::vector<uint32_t> f_off{0}, r_off{0}, n_off{0}, e_off{0}, f_sp, r_sp, n_sp, e_sp;
    std::vector<double> f_ord, r_ord, n_nu, e_extra;
    for (uint32_t i = 0; i < nr; i++) {
        const Reaction & r = mechanism.reactions[i];
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
        for (int p = 0; p < 5; p++) params[5 * i + p] = r.falloff_params[p];
        for (const auto & [k, order] : r.orders) {
            f_sp.push_back(k);
            f_ord.push_back(order);
        }
        f_off.push_back(f_sp.size());
        for (const auto & [k, nu] : r.products) {
            r_sp.push_back(k);
            r_ord.push_back(nu);
        }
        r_off.push_back(r_sp.size());
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
        n_off.push_back(n_sp.size());
        delta_nu[i] = sum;
        default_eff[i] = r.default_efficiency;
        for (const auto & [k, eff] : r.efficiencies) {
            e_sp.push_back(k);
            e_extra.push_back(eff - r.default_efficiency);
        }
        e_off.push_back(e_sp.size());
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
    t.delta_nu = copy(delta_nu, "kinetics_delta_nu");
    t.efficiency_offset = copy(e_off, "kinetics_efficiency_offset");
    t.efficiency_species = copy(e_sp, "kinetics_efficiency_species");
    t.efficiency_extra = copy(e_extra, "kinetics_efficiency_extra");
    t.default_efficiency = copy(default_eff, "kinetics_default_efficiency");
    return t;
}

} // namespace chemistry

#endif // CHEMISTRY_KINETICS_H

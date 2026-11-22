/**
 * @file thermo.h
 * @author Matthew Bonanni (mbonanni001@gmail.com)
 * @brief Device tables of species thermodynamics and thermally perfect
 *        mixture properties.
 * @version 0.3
 * @date 2026-10-02
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 *
 */

#ifndef CHEMISTRY_THERMO_H
#define CHEMISTRY_THERMO_H

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <sstream>
#include <stdexcept>

#include <Kokkos_Core.hpp>

#include "mechanism.h"

namespace chemistry {

/** @brief Mass fractions held in a contiguous array. */
struct MassFractions {
    const double * Y;
    KOKKOS_INLINE_FUNCTION double operator()(const uint32_t k) const { return Y[k]; }
};

/**
 * @brief Species thermodynamics of a mechanism on the device, in double
 *        precision in every build: per species its NASA-9 form ranges
 *        (SpeciesThermo), and the ideal-gas mixture properties per unit mass.
 *
 * A plain aggregate of Views in MemorySpace (the device by default, or the
 * host for setup), captured by value in kernels. Mixture functions take the
 * mass fractions as any callable y(k).
 */
template <typename MemorySpace = Kokkos::DefaultExecutionSpace::memory_space>
struct ThermoTable {
    uint32_t n_species = 0;
    Kokkos::View<double *, MemorySpace> inv_W;                           // kmol/kg
    Kokkos::View<uint32_t *, MemorySpace> range_offset;                  // (n_species + 1): ranges of species k
    Kokkos::View<double *, MemorySpace> range_upper;                     // (range): upper bound, unused for a species' last range
    Kokkos::View<double *[9], Kokkos::LayoutRight, MemorySpace> coeffs;  // (range, coefficient)
    double T_low = 1.0;                          // bracket of T(e)
    double T_high = 1.0e5;

    // Fits on either side of a range bound need not meet there; blending them
    // over a band keeps h, s and e continuous in T, so T(e) has a solution
    static constexpr double BLEND_WIDTH = 1.0;  // K

    /** @brief Powers of T shared by all species. */
    struct Powers {
        double T, inv_T, inv_T2, log_T;
    };

    KOKKOS_INLINE_FUNCTION
    static Powers powers(const double T) {
        const double inv_T = 1.0 / T;
        return {T, inv_T, inv_T * inv_T, Kokkos::log(T)};
    }

    KOKKOS_INLINE_FUNCTION
    uint32_t range(const uint32_t k, const double T) const {
        uint32_t r = range_offset(k);
        const uint32_t last = range_offset(k + 1) - 1;
        while (r < last && T >= range_upper(r)) r++;
        return r;
    }

    /**
     * @brief Whether T lies in the band of BLEND_WIDTH below a range bound of
     *        species k, where the polynomials of the lower range r and of the
     *        range above blend, and x in (0, 1] across the band.
     */
    KOKKOS_INLINE_FUNCTION
    bool blends(const uint32_t k, const double T, uint32_t & r, double & x) const {
        r = range(k, T);
        if (r + 1 == range_offset(k + 1) || !(T > range_upper(r) - BLEND_WIDTH)) return false;
        x = 1.0 - (range_upper(r) - T) / BLEND_WIDTH;
        return true;
    }

    KOKKOS_INLINE_FUNCTION
    double cp_R_range(const uint32_t r, const Powers & p) const {
        const double T = p.T;
        return coeffs(r, 0) * p.inv_T2 + coeffs(r, 1) * p.inv_T +
               (coeffs(r, 2) + T * (coeffs(r, 3) + T * (coeffs(r, 4) + T * (coeffs(r, 5) + T * coeffs(r, 6)))));
    }

    KOKKOS_INLINE_FUNCTION
    double dcp_R_dT_range(const uint32_t r, const Powers & p) const {
        const double T = p.T;
        return -2.0 * coeffs(r, 0) * p.inv_T2 * p.inv_T - coeffs(r, 1) * p.inv_T2 +
               (coeffs(r, 3) + T * (2.0 * coeffs(r, 4) + T * (3.0 * coeffs(r, 5) + T * 4.0 * coeffs(r, 6))));
    }

    KOKKOS_INLINE_FUNCTION
    double h_RT_range(const uint32_t r, const Powers & p) const {
        const double T = p.T;
        return -coeffs(r, 0) * p.inv_T2 + coeffs(r, 1) * p.log_T * p.inv_T +
               (coeffs(r, 2) + T * (coeffs(r, 3) / 2.0 + T * (coeffs(r, 4) / 3.0 +
                                    T * (coeffs(r, 5) / 4.0 + T * coeffs(r, 6) / 5.0)))) +
               coeffs(r, 7) * p.inv_T;
    }

    KOKKOS_INLINE_FUNCTION
    double s_R_range(const uint32_t r, const Powers & p) const {
        const double T = p.T;
        return -0.5 * coeffs(r, 0) * p.inv_T2 - coeffs(r, 1) * p.inv_T + coeffs(r, 2) * p.log_T +
               T * (coeffs(r, 3) + T * (coeffs(r, 4) / 2.0 + T * (coeffs(r, 5) / 3.0 + T * coeffs(r, 6) / 4.0))) +
               coeffs(r, 8);
    }

    // In a band the enthalpy is H = (1 - w) H_r + w H_{r+1} with the smoothstep
    // w = x^2 (3 - 2x), and cp = dH/dT, continuous across the band's edges

    /** @brief cp_k / R. */
    KOKKOS_INLINE_FUNCTION
    double cp_R(const uint32_t k, const Powers & p) const {
        uint32_t r;
        double x;
        if (!blends(k, p.T, r, x)) return cp_R_range(r, p);
        const double w = x * x * (3.0 - 2.0 * x), dw_dT = 6.0 * x * (1.0 - x) / BLEND_WIDTH;
        return (1.0 - w) * cp_R_range(r, p) + w * cp_R_range(r + 1, p) +
               dw_dT * p.T * (h_RT_range(r + 1, p) - h_RT_range(r, p));
    }

    /** @brief d(cp_k / R) / dT. */
    KOKKOS_INLINE_FUNCTION
    double dcp_R_dT(const uint32_t k, const Powers & p) const {
        uint32_t r;
        double x;
        if (!blends(k, p.T, r, x)) return dcp_R_dT_range(r, p);
        const double w = x * x * (3.0 - 2.0 * x), dw_dT = 6.0 * x * (1.0 - x) / BLEND_WIDTH;
        const double d2w_dT2 = 6.0 * (1.0 - 2.0 * x) / (BLEND_WIDTH * BLEND_WIDTH);
        return (1.0 - w) * dcp_R_dT_range(r, p) + w * dcp_R_dT_range(r + 1, p) +
               2.0 * dw_dT * (cp_R_range(r + 1, p) - cp_R_range(r, p)) +
               d2w_dT2 * p.T * (h_RT_range(r + 1, p) - h_RT_range(r, p));
    }

    /** @brief h_k / (R T), molar enthalpy including the formation enthalpy. */
    KOKKOS_INLINE_FUNCTION
    double h_RT(const uint32_t k, const Powers & p) const {
        uint32_t r;
        double x;
        if (!blends(k, p.T, r, x)) return h_RT_range(r, p);
        const double w = x * x * (3.0 - 2.0 * x);
        return (1.0 - w) * h_RT_range(r, p) + w * h_RT_range(r + 1, p);
    }

    /** @brief s_k / R at the standard pressure, blended as h so that g is continuous. */
    KOKKOS_INLINE_FUNCTION
    double s_R(const uint32_t k, const Powers & p) const {
        uint32_t r;
        double x;
        if (!blends(k, p.T, r, x)) return s_R_range(r, p);
        const double w = x * x * (3.0 - 2.0 * x);
        return (1.0 - w) * s_R_range(r, p) + w * s_R_range(r + 1, p);
    }

    /** @brief Mixture gas constant R_u sum_k Y_k / W_k [J/(kg K)]. */
    template <typename F_Y>
    KOKKOS_INLINE_FUNCTION double gas_constant(const F_Y & y) const {
        double sum = 0.0;
        for (uint32_t k = 0; k < n_species; k++) sum += y(k) * inv_W(k);
        return GAS_CONSTANT * sum;
    }

    /** @brief Mixture cp [J/(kg K)]. */
    template <typename F_Y>
    KOKKOS_INLINE_FUNCTION double cp_mass(const double T, const F_Y & y) const {
        const Powers p = powers(T);
        double sum = 0.0;
        for (uint32_t k = 0; k < n_species; k++) sum += y(k) * inv_W(k) * cp_R(k, p);
        return GAS_CONSTANT * sum;
    }

    /** @brief Mixture enthalpy [J/kg]. */
    template <typename F_Y>
    KOKKOS_INLINE_FUNCTION double h_mass(const double T, const F_Y & y) const {
        const Powers p = powers(T);
        double sum = 0.0;
        for (uint32_t k = 0; k < n_species; k++) sum += y(k) * inv_W(k) * h_RT(k, p);
        return GAS_CONSTANT * T * sum;
    }

    /**
     * @brief Mixture entropy at the standard pressure without the entropy of
     *        mixing [J/(kg K)]: at fixed composition, s(T, p) differs from it
     *        by a function of p alone.
     */
    template <typename F_Y>
    KOKKOS_INLINE_FUNCTION double s0_mass(const double T, const F_Y & y) const {
        const Powers p = powers(T);
        double sum = 0.0;
        for (uint32_t k = 0; k < n_species; k++) sum += y(k) * inv_W(k) * s_R(k, p);
        return GAS_CONSTANT * sum;
    }

    /** @brief Mixture internal energy e [J/kg] and cv [J/(kg K)] at T. */
    template <typename F_Y>
    KOKKOS_INLINE_FUNCTION void e_cv(const double T, const F_Y & y, double & e, double & cv) const {
        const Powers p = powers(T);
        double sum_e = 0.0, sum_cv = 0.0;
        for (uint32_t k = 0; k < n_species; k++) {
            const double n = y(k) * inv_W(k);
            sum_e += n * (h_RT(k, p) - 1.0);
            sum_cv += n * (cp_R(k, p) - 1.0);
        }
        e = GAS_CONSTANT * T * sum_e;
        cv = GAS_CONSTANT * sum_cv;
    }

    /**
     * @brief Temperature at which the mixture has internal energy e: Newton
     *        from T_guess (300 K if outside the bracket), safeguarded by
     *        bisection within [T_low, T_high], until the update is below 1e-10 T.
     */
    template <typename F_Y>
    KOKKOS_INLINE_FUNCTION double T_from_e(const double e, const F_Y & y, const double T_guess) const {
        double a = T_low, b = T_high;
        double T = (T_guess > a && T_guess < b) ? T_guess : Kokkos::fmin(Kokkos::fmax(300.0, a), b);
        for (int it = 0; it < 200; it++) {
            double e_T, cv;
            e_cv(T, y, e_T, cv);
            const double f = e_T - e;
            if (f > 0.0) {
                b = T;
            } else {
                a = T;
            }
            double T_new = T - f / cv;
            if (!(T_new >= a && T_new <= b)) T_new = 0.5 * (a + b);
            if (Kokkos::fabs(T_new - T) <= 1e-10 * T) return T_new;
            T = T_new;
        }
        return T;
    }
};

/** @brief Copy a mechanism's species thermodynamics to MemorySpace. */
template <typename MemorySpace = Kokkos::DefaultExecutionSpace::memory_space>
ThermoTable<MemorySpace> make_thermo_table(const Mechanism & mechanism) {
    ThermoTable<MemorySpace> table;
    const uint32_t n = mechanism.n_species();
    table.n_species = n;
    uint32_t n_ranges = 0;
    for (const auto & sp : mechanism.species) n_ranges += static_cast<uint32_t>(sp.thermo.coeffs.size());

    table.inv_W = Kokkos::View<double *, MemorySpace>("thermo_inv_W", n);
    table.range_offset = Kokkos::View<uint32_t *, MemorySpace>("thermo_range_offset", n + 1);
    table.range_upper = Kokkos::View<double *, MemorySpace>("thermo_range_upper", n_ranges);
    table.coeffs = Kokkos::View<double *[9], Kokkos::LayoutRight, MemorySpace>("thermo_coeffs", n_ranges);
    auto h_inv_W = Kokkos::create_mirror_view(table.inv_W);
    auto h_offset = Kokkos::create_mirror_view(table.range_offset);
    auto h_upper = Kokkos::create_mirror_view(table.range_upper);
    auto h_coeffs = Kokkos::create_mirror_view(table.coeffs);

    // Common range of the fits, where every species is fitted
    double T_min = 0.0, T_max = std::numeric_limits<double>::infinity();
    uint32_t r = 0;
    for (uint32_t k = 0; k < n; k++) {
        const Species & sp = mechanism.species[k];
        h_inv_W(k) = 1.0 / sp.molecular_weight;
        h_offset(k) = r;
        for (size_t i = 0; i < sp.thermo.coeffs.size(); i++, r++) {
            h_upper(r) = sp.thermo.T_bounds[i + 1];
            for (size_t j = 0; j < 9; j++) h_coeffs(r, j) = sp.thermo.coeffs[i][j];
        }
        T_min = std::max(T_min, sp.thermo.T_bounds.front());
        T_max = std::min(T_max, sp.thermo.T_bounds.back());
    }
    h_offset(n) = r;

    // The blend adds dw/dT (h_{r+1} - h_r) to cp across its band: refuse fits
    // whose ranges are narrower than the band or whose jump would halve cp
    ThermoTable<Kokkos::HostSpace> host;
    host.coeffs = h_coeffs;
    for (uint32_t k = 0; k < n; k++) {
        const Species & sp = mechanism.species[k];
        for (size_t i = 0; i < sp.thermo.coeffs.size(); i++) {
            std::ostringstream error;
            error << "Species " << sp.name << ": ";
            if (sp.thermo.coeffs.size() > 1 && !(sp.thermo.T_bounds[i + 1] - sp.thermo.T_bounds[i] > table.BLEND_WIDTH)) {
                error << "a temperature range is narrower than the " << table.BLEND_WIDTH << " K blend between fits.";
                throw std::runtime_error(error.str());
            }
            if (i == 0) continue;
            const uint32_t lower = h_offset(k) + static_cast<uint32_t>(i) - 1;
            const double T = sp.thermo.T_bounds[i];
            const auto p = ThermoTable<Kokkos::HostSpace>::powers(T);
            const double jump = T * (host.h_RT_range(lower + 1, p) - host.h_RT_range(lower, p));
            const double cp = std::min(host.cp_R_range(lower, p), host.cp_R_range(lower + 1, p));
            if (!(1.5 * std::abs(jump) / table.BLEND_WIDTH < 0.5 * cp)) {
                error << "h jumps by " << jump * GAS_CONSTANT * 1e-3 << " J/mol at " << T
                      << " K between its fits, more than the " << table.BLEND_WIDTH
                      << " K blend absorbs; refit the species so its ranges meet.";
                throw std::runtime_error(error.str());
            }
        }
    }
    // T(e) may extrapolate somewhat past the common range; constant cp has no bounds
    table.T_low = std::max(0.5 * T_min, 1.0);
    table.T_high = std::isfinite(T_max) ? 2.0 * T_max : 1.0e5;

    Kokkos::deep_copy(table.inv_W, h_inv_W);
    Kokkos::deep_copy(table.range_offset, h_offset);
    Kokkos::deep_copy(table.range_upper, h_upper);
    Kokkos::deep_copy(table.coeffs, h_coeffs);
    return table;
}

} // namespace chemistry

#endif // CHEMISTRY_THERMO_H

/**
 * @file transport.h
 * @author Matthew Bonanni (mbonanni001@gmail.com)
 * @brief Species transport fits and mixture-averaged transport properties
 *        on the device.
 * @version 0.3
 * @date 2026-10-03
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 *
 */

#ifndef CHEMISTRY_TRANSPORT_H
#define CHEMISTRY_TRANSPORT_H

#include <array>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include <Kokkos_Core.hpp>

#include "mechanism.h"

namespace chemistry {

enum class TransportModel {
    MIXTURE_AVERAGED,
    UNITY_LEWIS,
    CONSTANT_LEWIS,
};

/**
 * @brief Pure-species transport fits of a mechanism, as Cantera computes
 *        them for its mixture-averaged model: polynomials of degree 4 in
 *        ln T of sqrt(mu_k / sqrt(T)) (viscosity), lambda_k / sqrt(T)
 *        (conductivity) and p D_kj / T^1.5 (binary diffusion coefficients),
 *        fitted over the mechanism's common thermo temperature range from
 *        Lennard-Jones collision integrals (Monchick & Mason) with Cantera's
 *        polar corrections and Parker's rotational relaxation.
 */
struct TransportFits {
    std::vector<std::array<double, 5>> viscosity;     // per species
    std::vector<std::array<double, 5>> conductivity;  // per species
    std::vector<std::array<double, 5>> diffusion;     // per pair (k, j >= k), k-major
};

/**
 * @brief Fit the transport properties of every species of a mechanism.
 * @throws std::runtime_error naming a species without gas transport data.
 */
TransportFits fit_transport(const Mechanism & mechanism);

/**
 * @brief Pair data of Cantera's mixture-averaged thermal diffusion: its fit
 *        of the reduced collision integral C* as a polynomial of degree 8 in
 *        ln T* (Monchick & Mason's table, with the fits in the reduced dipole
 *        moment), and the pair's well depth epsilon_kj [J], per pair (k, j >= k),
 *        k-major.
 */
struct ThermalDiffusionFits {
    std::vector<std::array<double, 9>> cstar;
    std::vector<double> well_depth;
};

/**
 * @brief Fit the thermal diffusion data of a mechanism.
 * @throws std::runtime_error naming a species without gas transport data.
 */
ThermalDiffusionFits fit_thermal_diffusion(const Mechanism & mechanism);

/**
 * @brief Transport properties of a mechanism's mixtures on the device, in
 *        double precision in every build: Wilke's viscosity, the conductivity
 *        of Mathur, Tondon & Saxena, and the species diffusion coefficients
 *        D_k of the model (Cantera's "mixture-averaged" and
 *        "unity-Lewis-number" models, and constant Lewis numbers), to be
 *        used with mole-fraction gradients and a correction velocity; with
 *        the mixture-averaged model optionally also Cantera's thermal
 *        diffusion coefficients (Soret effect).
 *
 * A plain aggregate of Views in MemorySpace, captured by value in kernels.
 */
template <typename MemorySpace = Kokkos::DefaultExecutionSpace::memory_space>
struct TransportTable {
    uint32_t n_species = 0;
    TransportModel model = TransportModel::MIXTURE_AVERAGED;
    Kokkos::View<double *, MemorySpace> W;                                // kg/kmol
    Kokkos::View<double *, MemorySpace> inv_Lewis;                        // constant Lewis numbers
    Kokkos::View<double *[5], Kokkos::LayoutRight, MemorySpace> visc;     // (k, coefficient)
    Kokkos::View<double *[5], Kokkos::LayoutRight, MemorySpace> cond;
    Kokkos::View<double **[5], Kokkos::LayoutRight, MemorySpace> diff;   // (k, j, coefficient), symmetric
    Kokkos::View<double **, Kokkos::LayoutRight, MemorySpace> wilke_w;   // (W_j / W_k)^(1/4)
    Kokkos::View<double **, Kokkos::LayoutRight, MemorySpace> wilke_d;   // 1 / sqrt(8 (1 + W_k / W_j))
    bool thermal_diffusion = false;
    Kokkos::View<double **[9], Kokkos::LayoutRight, MemorySpace> cstar;  // (k, j, coefficient), symmetric
    Kokkos::View<double **, Kokkos::LayoutRight, MemorySpace> kb_eps;    // k_B / epsilon_kj [1/K]

    static constexpr double TINY = 1.0e-20;  // smallest mole fraction in the mixing rules, as in Cantera

    KOKKOS_INLINE_FUNCTION
    static double poly(const double * c, const double L) {
        return c[0] + L * (c[1] + L * (c[2] + L * (c[3] + L * c[4])));
    }

    KOKKOS_INLINE_FUNCTION
    double species_viscosity(const uint32_t k, const double T) const {
        const double s = Kokkos::sqrt(Kokkos::sqrt(T)) * poly(&visc(k, 0), Kokkos::log(T));
        return s * s;
    }

    KOKKOS_INLINE_FUNCTION
    double species_conductivity(const uint32_t k, const double T) const {
        return Kokkos::sqrt(T) * poly(&cond(k, 0), Kokkos::log(T));
    }

    /** @brief Binary diffusion coefficient D_kj [m^2/s] at T and p. */
    KOKKOS_INLINE_FUNCTION
    double binary_diffusion(const uint32_t k, const uint32_t j, const double T, const double p) const {
        return T * Kokkos::sqrt(T) * poly(&diff(k, j, 0), Kokkos::log(T)) / p;
    }

    /**
     * @brief Mixture viscosity mu [Pa s], conductivity lambda [W/(m K)] and
     *        species diffusion coefficients D [m^2/s] (n_species values, also
     *        used as work memory) at temperature T, pressure p, density rho
     *        and cp [J/(kg K)], for mass fractions y(k). As Cantera, mole
     *        fractions below 1e-20 count as 1e-20 in the mixing rules.
     *
     * With DT (n_species values; mixture-averaged model with
     * thermal_diffusion only), also Cantera's mixture-averaged thermal
     * diffusion coefficients D^T_k [kg/(m s)] (Chapman & Cowling's model as
     * Zirwes & Kronenburg give it), for the fluxes -D^T_k grad T / T; they
     * sum to zero.
     */
    template <typename F_Y>
    KOKKOS_INLINE_FUNCTION void properties(const double T, const double p, const double rho, const double cp,
                                           const F_Y & y, double & mu, double & lambda, double * D,
                                           double * DT = nullptr) const {
        const double L = Kokkos::log(T);
        const double sqrt_T = Kokkos::sqrt(T);
        const double t14 = Kokkos::sqrt(sqrt_T);
        // Small negative mass fractions (round-off, or the overshoots of explicit
        // diffusion) count as zero
        auto Y = [&](const uint32_t k) { return Kokkos::fmax(0.0, y(k)); };
        double n = 0.0;
        for (uint32_t k = 0; k < n_species; k++) n += Y(k) / W(k);
        const double inv_n = 1.0 / n;  // mean molar mass
        auto X = [&](const uint32_t k) { return Kokkos::fmax(TINY, Y(k) / W(k) * inv_n); };
        // sqrt(mu_k) in D until the diffusion coefficients
        for (uint32_t k = 0; k < n_species; k++) D[k] = t14 * poly(&visc(k, 0), L);
        mu = 0.0;
        for (uint32_t k = 0; k < n_species; k++) {
            double sum = 0.0;
            for (uint32_t j = 0; j < n_species; j++) {
                const double f = 1.0 + D[k] / D[j] * wilke_w(k, j);
                sum += f * f * wilke_d(k, j) * X(j);
            }
            mu += X(k) * D[k] * D[k] / sum;
        }
        if (DT) {
            // a_k = (15/4) (mu_k / W_k) / (1 + 1.065 sum_(j != k) X_j Phi_kj / X_k), held in DT
            for (uint32_t k = 0; k < n_species; k++) {
                if (Y(k) < TINY) {
                    DT[k] = 0.0;
                    continue;
                }
                double sum = 0.0;
                for (uint32_t j = 0; j < n_species; j++) {
                    if (j == k) continue;
                    const double f = 1.0 + D[k] / D[j] * wilke_w(k, j);
                    sum += X(j) * (f * f * wilke_d(k, j));
                }
                DT[k] = (15.0 / 4.0) * D[k] * D[k] / W(k) / (1.0 + 1.065 * sum / X(k));
            }
        }
        double sum1 = 0.0, sum2 = 0.0;
        for (uint32_t k = 0; k < n_species; k++) {
            const double lambda_k = sqrt_T * poly(&cond(k, 0), L);
            sum1 += X(k) * lambda_k;
            sum2 += X(k) / lambda_k;
        }
        lambda = 0.5 * (sum1 + 1.0 / sum2);
        if (model != TransportModel::MIXTURE_AVERAGED) {
            const double alpha = lambda / (rho * cp);
            for (uint32_t k = 0; k < n_species; k++) D[k] = alpha * inv_Lewis(k);
            return;
        }
        const double T15_p = T * sqrt_T / p;
        if (n_species == 1) {
            D[0] = T15_p * poly(&diff(0, 0, 0), L);
            if (DT) DT[0] = 0.0;
            return;
        }
        if (DT) {
            // sum_(j != k) (1.2 C*_kj - 1) / (D_kj (W_k + W_j)) (Y_k a_j - Y_j a_k), held in D
            for (uint32_t k = 0; k < n_species; k++) D[k] = 0.0;
            for (uint32_t k = 0; k + 1 < n_species; k++) {
                for (uint32_t j = k + 1; j < n_species; j++) {
                    const double log_ts = Kokkos::log(T * kb_eps(k, j));
                    const double * c = &cstar(k, j, 0);
                    double C = c[8];
                    for (int m = 7; m >= 0; m--) C = C * log_ts + c[m];
                    const double D_kj = T15_p * poly(&diff(k, j, 0), L);
                    const double f = (1.2 * C - 1.0) / D_kj / (W(k) + W(j));
                    const double s = f * (Y(k) * DT[j] - Y(j) * DT[k]);
                    D[k] += s;
                    D[j] -= s;
                }
            }
        }
        // D_k = (1 - Y_k) / sum_(j != k) X_j / D_kj, with 1 - Y_k summed from the
        // other species: Cantera's (W - X_k W_k) / (W sum ...) cancels to round-off
        // noise over a vanishing denominator where species k is nearly pure
        for (uint32_t k = 0; k < n_species; k++) {
            double sum = 0.0, others = 0.0;
            for (uint32_t j = 0; j < n_species; j++) {
                if (j == k) continue;
                sum += X(j) / poly(&diff(j, k, 0), L);
                others += Y(j);
            }
            const double D_k = others * T15_p / sum;
            if (DT) DT[k] = D[k] * D_k * W(k) * inv_n;
            D[k] = D_k;
        }
        if (DT) {
            double norm = 0.0;
            for (uint32_t k = 0; k < n_species; k++) norm += DT[k];
            for (uint32_t k = 0; k < n_species; k++) DT[k] -= Y(k) * norm;
        }
    }
};

/**
 * @brief The transport table of a mechanism for a model.
 * @param lewis Constant Lewis numbers per species (CONSTANT_LEWIS only); empty: 1.
 * @param thermal_diffusion Also the data of the thermal diffusion coefficients (MIXTURE_AVERAGED only).
 */
template <typename MemorySpace = Kokkos::DefaultExecutionSpace::memory_space>
TransportTable<MemorySpace> make_transport_table(const Mechanism & mechanism, const TransportModel model,
                                                 const std::vector<double> & lewis = {},
                                                 const bool thermal_diffusion = false) {
    const TransportFits fits = fit_transport(mechanism);
    TransportTable<MemorySpace> table;
    const uint32_t n = static_cast<uint32_t>(mechanism.n_species());
    table.n_species = n;
    table.model = model;
    table.W = Kokkos::View<double *, MemorySpace>("transport_W", n);
    table.inv_Lewis = Kokkos::View<double *, MemorySpace>("transport_inv_Lewis", n);
    table.visc = Kokkos::View<double *[5], Kokkos::LayoutRight, MemorySpace>("transport_visc", n);
    table.cond = Kokkos::View<double *[5], Kokkos::LayoutRight, MemorySpace>("transport_cond", n);
    table.diff = Kokkos::View<double **[5], Kokkos::LayoutRight, MemorySpace>("transport_diff", n, n);
    table.wilke_w = Kokkos::View<double **, Kokkos::LayoutRight, MemorySpace>("transport_wilke_w", n, n);
    table.wilke_d = Kokkos::View<double **, Kokkos::LayoutRight, MemorySpace>("transport_wilke_d", n, n);
    auto h_W = Kokkos::create_mirror_view(table.W);
    auto h_inv_Le = Kokkos::create_mirror_view(table.inv_Lewis);
    auto h_visc = Kokkos::create_mirror_view(table.visc);
    auto h_cond = Kokkos::create_mirror_view(table.cond);
    auto h_diff = Kokkos::create_mirror_view(table.diff);
    auto h_ww = Kokkos::create_mirror_view(table.wilke_w);
    auto h_wd = Kokkos::create_mirror_view(table.wilke_d);
    size_t pair = 0;
    for (uint32_t k = 0; k < n; k++) {
        h_W(k) = mechanism.species[k].molecular_weight;
        h_inv_Le(k) = (model == TransportModel::CONSTANT_LEWIS && !lewis.empty()) ? 1.0 / lewis[k] : 1.0;
        for (int c = 0; c < 5; c++) {
            h_visc(k, c) = fits.viscosity[k][c];
            h_cond(k, c) = fits.conductivity[k][c];
        }
        for (uint32_t j = k; j < n; j++, pair++) {
            for (int c = 0; c < 5; c++) {
                h_diff(k, j, c) = fits.diffusion[pair][c];
                h_diff(j, k, c) = fits.diffusion[pair][c];
            }
        }
    }
    for (uint32_t k = 0; k < n; k++) {
        for (uint32_t j = 0; j < n; j++) {
            h_ww(k, j) = std::sqrt(std::sqrt(h_W(j) / h_W(k)));
            h_wd(k, j) = 1.0 / (std::sqrt(8.0) * std::sqrt(1.0 + h_W(k) / h_W(j)));
        }
    }
    Kokkos::deep_copy(table.W, h_W);
    Kokkos::deep_copy(table.inv_Lewis, h_inv_Le);
    Kokkos::deep_copy(table.visc, h_visc);
    Kokkos::deep_copy(table.cond, h_cond);
    Kokkos::deep_copy(table.diff, h_diff);
    Kokkos::deep_copy(table.wilke_w, h_ww);
    Kokkos::deep_copy(table.wilke_d, h_wd);
    if (thermal_diffusion) {
        if (model != TransportModel::MIXTURE_AVERAGED) {
            throw std::invalid_argument("Thermal diffusion needs the mixture-averaged transport model.");
        }
        const ThermalDiffusionFits td = fit_thermal_diffusion(mechanism);
        table.thermal_diffusion = true;
        table.cstar = Kokkos::View<double **[9], Kokkos::LayoutRight, MemorySpace>("transport_cstar", n, n);
        table.kb_eps = Kokkos::View<double **, Kokkos::LayoutRight, MemorySpace>("transport_kb_eps", n, n);
        auto h_cstar = Kokkos::create_mirror_view(table.cstar);
        auto h_kb_eps = Kokkos::create_mirror_view(table.kb_eps);
        size_t p = 0;
        for (uint32_t k = 0; k < n; k++) {
            for (uint32_t j = k; j < n; j++, p++) {
                for (int c = 0; c < 9; c++) {
                    h_cstar(k, j, c) = td.cstar[p][c];
                    h_cstar(j, k, c) = td.cstar[p][c];
                }
                h_kb_eps(k, j) = h_kb_eps(j, k) = BOLTZMANN / td.well_depth[p];
            }
        }
        Kokkos::deep_copy(table.cstar, h_cstar);
        Kokkos::deep_copy(table.kb_eps, h_kb_eps);
    }
    return table;
}

} // namespace chemistry

#endif // CHEMISTRY_TRANSPORT_H

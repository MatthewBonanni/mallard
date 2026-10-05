/**
 * @file pasr.h
 * @author Matthew Bonanni (mbonanni001@gmail.com)
 * @brief Partially stirred reactor (PaSR) closure of the filtered reaction
 *        rates (docs/design/les.md, section 6).
 * @version 0.1
 * @date 2026-10-05
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 *
 */

#ifndef PASR_H
#define PASR_H

#include <Kokkos_Core.hpp>
#include <toml.hpp>

#include "log.h"

/**
 * @brief Each cell's rates are those of its filtered state times the reacting
 *        fraction kappa = tau_c / (tau_c + tau_mix) (Golovitchev & Chomiak;
 *        Sabelnikov & Fureby, Combust. Flame 160, 2013), with the chemical
 *        time tau_c = rho cp T / |q| of the heat release rate q and the mixing
 *        time tau_mix = C_mix Delta^2 / (nu + nu_t) of molecular and SGS
 *        diffusion across the cell, so kappa -> 1 as the mesh resolves the
 *        mixing (Delta -> 0, the DNS limit) and kappa -> tau_c / tau_mix,
 *        a mixing-limited rate, where the chemistry is fast. Plain aggregate,
 *        captured by value in kernels.
 */
struct PartiallyStirredReactor {
    double C_mix = 1.0;

    /** @brief [les.combustion] of an input with model = "pasr". */
    static PartiallyStirredReactor from_input(const toml::value & table);

    logging::Items summary() const;

    /** @brief kappa of a cell: rho_cp_T = rho cp T, q the heat release rate, nu and nu_t kinematic viscosities. */
    KOKKOS_INLINE_FUNCTION
    double fraction(const double rho_cp_T, const double q, const double delta, const double nu,
                    const double nu_t) const {
        const double tau_mix = C_mix * delta * delta / (nu + nu_t);
        return rho_cp_T / (rho_cp_T + Kokkos::fabs(q) * tau_mix);
    }
};

#endif // PASR_H

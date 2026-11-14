/**
 * @file derived_catalog.h
 * @brief The catalog of derived output fields: each field's name, units,
 *        description and what a case needs for it, read by the in-run
 *        writers, mallard post derive and mallard list fields.
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 */

#ifndef DERIVED_CATALOG_H
#define DERIVED_CATALOG_H

#include <cstdint>

#include "units.h"

namespace derived {

/** @brief The derived fields; their columns follow the order they are requested in. */
enum Field : uint8_t {
    MACH,
    Q_CRITERION,
    LAMBDA2,
    SCHLIEREN,
    P0,
    T0,
    H0,
    SOUND_SPEED,
    ENTROPY,
    DILATATION,
    HELICITY,
    ENSTROPHY,
    STRAIN_RATE,
    CP,
    MIXTURE_FRACTION,
    PROGRESS_VARIABLE,
    FLAME_INDEX,
    SCALAR_DISSIPATION,
    N_FIELDS
};

/** @brief What a field needs, as bits. */
namespace need {
inline constexpr uint16_t GRADIENTS = 1u << 0;          ///< the cell gradients of W = (rho, u, p)
inline constexpr uint16_t SPECIES_GRADIENTS = 1u << 1;  ///< least-squares gradients of mass fractions
inline constexpr uint16_t THREE_D = 1u << 2;            ///< a 3D build
inline constexpr uint16_t MIXTURE = 1u << 3;            ///< a gas mixture
inline constexpr uint16_t VISCOUS = 1u << 4;            ///< navier_stokes (a diffusivity)
inline constexpr uint16_t REFERENCE = 1u << 5;          ///< [output] reference
inline constexpr uint16_t STREAMS = 1u << 6;            ///< [output.mixture] fuel and oxidizer
inline constexpr uint16_t SPECIES_PAIR = 1u << 7;       ///< a fuel and an oxidizer species
inline constexpr uint16_t PROGRESS = 1u << 8;           ///< [output.mixture] progress
} // namespace need

/** @brief A derived field's entry in the catalog. */
struct Definition {
    const char * name;
    Dimension dimension;
    const char * long_name;
    const char * group;  ///< the mallard list fields group
    uint16_t needs;      ///< need:: bits
    const char * needs_text;  ///< what a case needs, as mallard list fields shows it; "" if every case has it
};

namespace detail {
inline constexpr Dimension rate_squared = dim::of(0, 0, -2);
inline constexpr const char * DERIVED = "derived";
inline constexpr const char * MIXTURE = "mixture";
} // namespace detail

/** @brief The catalog, indexed by Field. */
inline constexpr Definition FIELDS[N_FIELDS] = {
    {"MACH", dim::none, "Mach number", detail::DERIVED, 0, ""},
    {"Q_CRITERION", detail::rate_squared, "Q-criterion", detail::DERIVED, need::GRADIENTS, ""},
    {"LAMBDA2", detail::rate_squared, "lambda2 vortex criterion", detail::DERIVED, need::GRADIENTS, ""},
    {"SCHLIEREN", dim::of(1, -4, 0), "density gradient magnitude", detail::DERIVED, need::GRADIENTS, ""},
    {"P0", dim::pressure, "stagnation pressure", detail::DERIVED, 0, ""},
    {"T0", dim::temperature, "stagnation temperature", detail::DERIVED, 0, ""},
    {"H0", dim::velocity_squared, "stagnation (total) specific enthalpy", detail::DERIVED, 0, ""},
    {"SOUND_SPEED", dim::velocity, "speed of sound", detail::DERIVED, 0, ""},
    {"ENTROPY", dim::of(0, 2, -2, -1), "specific entropy", detail::DERIVED, 0, ""},
    {"DILATATION", dim::rate, "dilatation (velocity divergence)", detail::DERIVED, need::GRADIENTS, ""},
    {"HELICITY", dim::acceleration, "helicity density", detail::DERIVED, need::GRADIENTS | need::THREE_D,
     "a 3D build (in 2D the vorticity is normal to the velocity)"},
    {"ENSTROPHY", detail::rate_squared, "enstrophy", detail::DERIVED, need::GRADIENTS, ""},
    {"STRAIN_RATE", dim::rate, "strain-rate magnitude", detail::DERIVED, need::GRADIENTS, ""},
    {"CP", dim::none, "pressure coefficient", detail::DERIVED, need::REFERENCE, "[output] reference = {rho, u, p}"},
    {"MIXTURE_FRACTION", dim::none, "Bilger mixture fraction", detail::MIXTURE, need::MIXTURE | need::STREAMS,
     "a mixture with [output.mixture] fuel and oxidizer"},
    {"PROGRESS_VARIABLE", dim::none, "progress variable", detail::MIXTURE, need::MIXTURE | need::PROGRESS,
     "a mixture with [output.mixture] progress"},
    {"FLAME_INDEX", dim::none, "Takeno flame index", detail::MIXTURE,
     need::MIXTURE | need::SPECIES_PAIR | need::SPECIES_GRADIENTS,
     "a mixture with [output.mixture] fuel and oxidizer, or fuel_species and oxidizer_species"},
    {"SCALAR_DISSIPATION", dim::rate, "scalar dissipation rate of the mixture fraction", detail::MIXTURE,
     need::MIXTURE | need::VISCOUS | need::STREAMS | need::SPECIES_GRADIENTS,
     "a mixture with navier_stokes and [output.mixture] fuel and oxidizer"},
};

/** @brief The field of a name, or N_FIELDS if no derived field has it. */
constexpr Field find(const char * name) {
    for (uint8_t f = 0; f < N_FIELDS; f++) {
        const char * a = FIELDS[f].name;
        const char * b = name;
        while (*a && *a == *b) {
            a++;
            b++;
        }
        if (*a == *b) return static_cast<Field>(f);
    }
    return N_FIELDS;
}

} // namespace derived

#endif // DERIVED_CATALOG_H

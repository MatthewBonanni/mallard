/**
 * @file probe_fluxes.h
 * @author Matthew Bonanni (mbonanni001@gmail.com)
 * @brief The scheme's viscous flux functors, from which boundary probes take
 *        their diffusive fluxes.
 * @version 0.1
 * @date 2026-11-04
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 *
 */

#ifndef PROBE_FLUXES_H
#define PROBE_FLUXES_H

#include "mixture_viscous_flux.h"
#include "viscous_flux.h"

/** @brief The molecular viscous flux of the run: of a single gas, or of a mixture (the other is empty). */
struct ProbeFluxes {
    ViscousFluxFunctor viscous = {};
    MixtureViscousFluxFunctor mixture = {};
};

#endif // PROBE_FLUXES_H

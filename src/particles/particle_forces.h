/**
 * @file particle_forces.h
 * @author Matthew Bonanni (mbonanni001@gmail.com)
 * @brief Forces on solid particles: drag correlations, gravity and buoyancy,
 *        and the stiff limit of the velocity's relaxation. Design:
 *        docs/design/particles.md.
 * @version 0.1
 * @date 2026-11-20
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 *
 */

#ifndef PARTICLE_FORCES_H
#define PARTICLE_FORCES_H

#include <cstdint>

#include <Kokkos_Core.hpp>

#include "common.h"
#include "physics.h"

namespace particles {

/** @brief Drag correlation of a sphere: f = C_D Re / 24, the drag over Stokes drag. */
enum class Drag : uint8_t { STOKES, SCHILLER_NAUMANN };

/** @brief The drag correction f(Re) = C_D Re / 24 of the correlation at the particle Reynolds number Re. */
KOKKOS_INLINE_FUNCTION rtype drag_factor(const Drag drag, const rtype Re) {
    switch (drag) {
        case Drag::SCHILLER_NAUMANN:
            // C_D = 24 / Re (1 + 0.15 Re^0.687) below Re = 1000, 0.44 above
            return Re < 1000.0_r ? 1.0_r + 0.15_r * Kokkos::pow(Re, 0.687_r) : 0.44_r * Re / 24.0_r;
        default: return 1.0_r;
    }
}

/**
 * @brief Solid spheres of one diameter and density in a single gas: the
 *        right-hand side of dx/dt and dv/dt from the gas state at the particle.
 *
 * The velocity relaxes to the gas velocity in tau = rho_p d^2 / (18 mu f). A
 * step dt <= tau resolves it: dx/dt = v, dv/dt = (u - v) / tau + g (1 - rho /
 * rho_p), at the integrator's order. A particle with tau < dt is in its stiff
 * limit, where an explicit integrator would be unstable: it moves at its
 * equilibrium velocity u + tau g' (first order in tau), and its velocity
 * relaxes to that velocity in one step. The flow's step never depends on tau.
 */
struct Solid {
    rtype diameter = 0.0;
    rtype density = 0.0;
    Drag drag = Drag::STOKES;
    rtype gravity[N_DIM] = {};
    Euler gas;

    /** @brief Relaxation time of the velocity at gas state W = [rho, u, p] and velocity v. */
    KOKKOS_INLINE_FUNCTION rtype relaxation_time(const rtype * W, const rtype * v) const {
        const rtype T = W[N_DIM + 1] / (W[0] * gas.R);
        const rtype mu = gas.viscosity(T);
        rtype slip2 = 0.0_r;
        FOR_I_DIM slip2 += (W[1 + i] - v[i]) * (W[1 + i] - v[i]);
        const rtype Re = W[0] * Kokkos::sqrt(slip2) * diameter / mu;
        return density * diameter * diameter / (18.0_r * mu * drag_factor(drag, Re));
    }

    /**
     * @brief dx/dt and dv/dt of a particle at velocity v in gas W, for a step dt.
     * @return Whether the particle is in its stiff limit (tau < dt).
     */
    KOKKOS_INLINE_FUNCTION bool rates(const rtype * W, const rtype * v, const rtype dt, rtype * dxdt, rtype * dvdt) const {
        const rtype tau = relaxation_time(W, v);
        const rtype buoyancy = 1.0_r - W[0] / density;
        const bool stiff = tau < dt;
        FOR_I_DIM {
            const rtype g = gravity[i] * buoyancy;
            if (stiff) {
                const rtype v_eq = W[1 + i] + tau * g;
                dxdt[i] = v_eq;
                dvdt[i] = (v_eq - v[i]) / dt;
            } else {
                dxdt[i] = v[i];
                dvdt[i] = (W[1 + i] - v[i]) / tau + g;
            }
        }
        return stiff;
    }
};

} // namespace particles

#endif // PARTICLE_FORCES_H

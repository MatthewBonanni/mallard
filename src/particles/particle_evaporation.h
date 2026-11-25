/**
 * @file particle_evaporation.h
 * @author Matthew Bonanni (mbonanni001@gmail.com)
 * @brief Evaporating droplets of one liquid in a gas mixture: the liquid's
 *        properties, the gas film at the surface, the d^2-law and
 *        Abramzon-Sirignano evaporation rates, and the droplet's rates of
 *        mass, momentum and energy. Design: docs/design/particles.md.
 * @version 0.1
 * @date 2026-11-21
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 *
 */

#ifndef PARTICLE_EVAPORATION_H
#define PARTICLE_EVAPORATION_H

#include <cstdint>

#include <Kokkos_Core.hpp>

#include "common.h"
#include "mixture.h"
#include "particle_forces.h"

namespace particles {

/** @brief Evaporation model: the classical d^2 law, or Abramzon and Sirignano's film model. */
enum class Evaporation : uint8_t { D2_LAW, ABRAMZON_SIRIGNANO };

/**
 * @brief A liquid of one component whose vapor is a species of the mechanism.
 *
 * Constant density and specific heat. The liquid's enthalpy is
 * h_l(T) = h_v(T_b) - L_b + c_l (T - T_b), from the vapor's enthalpy h_v of the
 * mechanism (formation enthalpy included) and the latent heat L_b at the
 * boiling temperature T_b at p_b, so that the latent heat at T,
 * L(T) = h_v(T) - h_l(T), follows Kirchhoff's law and the gas and the liquid
 * share one energy reference. The saturation pressure is Clausius-Clapeyron's
 * through (T_b, p_b) with L_b, or Antoine's log10(p / Pa) = A - B / (T / K + C).
 */
struct Liquid {
    uint32_t vapor = 0;            // species index
    double density = 0.0;
    double specific_heat = 0.0;
    double latent_heat = 0.0;      // at the boiling temperature
    double boiling_temperature = 0.0;
    double boiling_pressure = 101325.0;
    double critical_temperature = 0.0;
    double critical_pressure = 0.0;
    double R_vapor = 0.0;          // gas constant of the vapor
    double h_ref = 0.0;            // h_l(T_b)
    bool antoine = false;
    double A = 0.0, B = 0.0, C = 0.0;

    /** @brief The saturation pressure at T. */
    KOKKOS_INLINE_FUNCTION double saturation_pressure(const double T) const {
        if (antoine) return Kokkos::pow(10.0, A - B / (T + C));
        return boiling_pressure * Kokkos::exp(latent_heat / R_vapor * (1.0 / boiling_temperature - 1.0 / T));
    }

    /**
     * @brief The boiling temperature at pressure p (where the saturation pressure is p), at most the
     *        critical temperature, where the saturation curve ends.
     */
    KOKKOS_INLINE_FUNCTION double boiling_temperature_at(const double p) const {
        double T;
        if (antoine) {
            const double denominator = A - Kokkos::log10(p);
            T = denominator > 0.0 ? B / denominator - C : critical_temperature;
        } else {
            const double inverse = 1.0 / boiling_temperature - R_vapor / latent_heat * Kokkos::log(p / boiling_pressure);
            T = inverse > 0.0 ? 1.0 / inverse : critical_temperature;
        }
        return Kokkos::fmin(T, critical_temperature);
    }

    /** @brief The sensible enthalpy h_l(T) - h_l(T_b). */
    KOKKOS_INLINE_FUNCTION double sensible(const double T) const { return specific_heat * (T - boiling_temperature); }
};

/** @brief F(B) = (1 + B)^0.7 ln(1 + B) / B of Abramzon and Sirignano, the film thickening by the Stefan flow. */
KOKKOS_INLINE_FUNCTION double film_correction(const double B) {
    if (Kokkos::fabs(B) < 1e-6) return 1.0 + 0.2 * B;
    return Kokkos::pow(1.0 + B, 0.7) * Kokkos::log1p(B) / B;
}

/** @brief ln(1 + B) / B, 1 at B = 0. */
KOKKOS_INLINE_FUNCTION double log_ratio(const double B) {
    if (Kokkos::fabs(B) < 1e-6) return 1.0 - 0.5 * B;
    return Kokkos::log1p(B) / B;
}

/**
 * @brief Mass fractions of the film: the vapor's Y_f, the other species those of the gas scaled to 1 - Y_f.
 */
struct FilmFractions {
    const rtype * Y;   // of the gas
    uint32_t vapor;
    double Y_vapor;    // in the film
    double scale;      // (1 - Y_vapor) / (1 - Y_gas,vapor)

    KOKKOS_INLINE_FUNCTION double operator()(const uint32_t k) const {
        return k == vapor ? Y_vapor : Kokkos::fmax(0.0, static_cast<double>(Y[k])) * scale;
    }
};

/** @brief Mass fractions of the gas, negative round-off counted as 0. */
struct GasFractions {
    const rtype * Y;
    KOKKOS_INLINE_FUNCTION double operator()(const uint32_t k) const { return Kokkos::fmax(0.0, static_cast<double>(Y[k])); }
};

/** @brief What the gas does to a droplet: its motion, the evaporation rate and the heat it receives. */
struct Transfer {
    rtype tau;          // velocity relaxation
    rtype a[N_DIM];     // acceleration by the other forces
    double T_gas;       // gas temperature
    double mdot;        // evaporation rate (mass per time leaving the droplet; negative: condensation)
    double Q;           // heat from the gas at the surface
    double latent;      // latent heat at the surface temperature
    double h_vapor;     // h_v(T_s) - h_l(T_b): enthalpy of the vapor leaving, from the liquid's reference
    double heat_up;     // the droplet's thermal relaxation time without evaporation
};

/**
 * @brief Droplets of one liquid in a viscous gas mixture: from the gas state at
 *        the droplet, its drag and forces (Solid::motion), its evaporation and
 *        heating, and the rates of its position, momentum, energy and mass.
 *
 * A droplet's block values after its position are its momentum P = M v, its
 * energy E = M (c_l (T_p - T_b) + |v|^2 / 2) and its mass M: the conserved
 * quantities, so that the gas, which receives the opposite of their stage
 * rates, and the droplets together conserve mass, momentum and energy to
 * round-off, whatever the integrator. The droplet is isothermal (infinite
 * liquid conductivity). Properties of the gas film at the surface follow the
 * 1/3 rule: T_f = T_s + (T - T_s) / 3, Y_f = Y_s + (Y - Y_s) / 3, Y_s from the
 * saturation pressure at the surface temperature T_s by Raoult's law of a pure
 * liquid. T_s is the droplet's temperature up to the boiling temperature at
 * the gas pressure; above it the droplet boils: the heat it receives
 * evaporates it, and its superheat relaxes to the boiling temperature at its
 * heating rate.
 */
struct Droplet {
    Solid motion;  // density of the liquid, drag, gravity and forces; per-droplet diameter
    Liquid liquid;
    Evaporation model = Evaporation::ABRAMZON_SIRIGNANO;
    Mixture gas;
    double minimum_mass = 0.0;  // below it a droplet is inert until it is removed after the step

    /** @brief Block values after the position: momentum, energy, mass. */
    enum Value : uint32_t { MOMENTUM = 0, ENERGY = N_DIM, MASS = N_DIM + 1, N_VALUES = N_DIM + 2 };

    /**
     * @brief The values the first stage of a step keeps: those of Solid::Start (v, u, a, tau, dt), then the
     *        temperature, its linearized equilibrium and relaxation time (0: explicit), and the mass.
     */
    enum Start : uint32_t { V = Solid::V, U = Solid::U, A = Solid::A, TAU = Solid::TAU, DT = Solid::DT,
                            TP = 3 * N_DIM + 2, T_EQ = 3 * N_DIM + 3, TAU_T = 3 * N_DIM + 4, M0 = 3 * N_DIM + 5,
                            N_START = 3 * N_DIM + 6 };

    /** @brief The mass of a droplet of diameter d. */
    KOKKOS_INLINE_FUNCTION double mass_of(const double d) const {
        return liquid.density * Kokkos::numbers::pi_v<double> / 6.0 * d * d * d;
    }

    /** @brief The diameter of a droplet of mass M (0 if M is not positive). */
    KOKKOS_INLINE_FUNCTION double diameter_of(const double M) const {
        return M > 0.0 ? Kokkos::cbrt(6.0 * M / (Kokkos::numbers::pi_v<double> * liquid.density)) : 0.0;
    }

    /** @brief The velocity of a droplet whose block values from its momentum on are y (0 without mass). */
    KOKKOS_INLINE_FUNCTION void velocity(const rtype * y, rtype * v) const {
        const rtype M = y[MASS];
        FOR_I_DIM v[i] = M > 0.0_r ? y[MOMENTUM + i] / M : 0.0_r;
    }

    /** @brief The temperature of a droplet from its block values y (the boiling temperature without mass). */
    KOKKOS_INLINE_FUNCTION double temperature(const rtype * y) const {
        const double M = static_cast<double>(y[MASS]);
        if (!(M > 0.0)) return liquid.boiling_temperature;
        double v2 = 0.0;
        FOR_I_DIM v2 += static_cast<double>(y[MOMENTUM + i]) * static_cast<double>(y[MOMENTUM + i]);
        const double e = static_cast<double>(y[ENERGY]) / M - 0.5 * v2 / (M * M);
        return liquid.boiling_temperature + e / liquid.specific_heat;
    }

    /** @brief The block values after the position of a droplet of mass M, velocity v and temperature T. */
    KOKKOS_INLINE_FUNCTION void values(const double M, const rtype * v, const double T, rtype * y) const {
        double v2 = 0.0;
        FOR_I_DIM {
            y[MOMENTUM + i] = static_cast<rtype>(M * static_cast<double>(v[i]));
            v2 += static_cast<double>(v[i]) * static_cast<double>(v[i]);
        }
        y[ENERGY] = static_cast<rtype>(M * (liquid.sensible(T) + 0.5 * v2));
        y[MASS] = static_cast<rtype>(M);
    }

    /** @brief The gas temperature p / (rho R) at gas state W with mass fractions Y. */
    KOKKOS_INLINE_FUNCTION double gas_temperature(const rtype * W, const rtype * Y) const {
        return static_cast<double>(W[N_DIM + 1]) / (static_cast<double>(W[0]) * gas.thermo.gas_constant(GasFractions{Y}));
    }

    /**
     * @brief The motion, evaporation and heating of a droplet of mass M (positive), velocity v and temperature Tp
     *        in gas W with gradients G and mass fractions Y.
     * @param work n_species values of work memory.
     */
    KOKKOS_INLINE_FUNCTION Transfer transfer(const rtype * W, const rtype * G, const rtype * Y, const rtype * v,
                                             const double Tp, const double M, double * work) const {
        const double pi = Kokkos::numbers::pi_v<double>;
        const double rho = static_cast<double>(W[0]), p = static_cast<double>(W[N_DIM + 1]);
        const GasFractions y_gas{Y};
        const uint32_t k_v = liquid.vapor;
        const double R_gas = gas.thermo.gas_constant(y_gas);
        Transfer t;
        t.T_gas = p / (rho * R_gas);
        const double d = diameter_of(M);
        const double T_boil = liquid.boiling_temperature_at(p);
        const bool boiling = Tp >= T_boil;
        const double Ts = Kokkos::fmax(Kokkos::fmin(Tp, T_boil), gas.thermo.T_low);
        // Surface: the vapor at its saturation pressure (Raoult's law of a pure liquid), the gas's other species
        const double X_s = Kokkos::fmin(liquid.saturation_pressure(Ts) / p, 1.0);
        const double Y_v = Kokkos::fmin(y_gas(k_v), 1.0);
        double n_dry = 0.0;
        for (uint32_t k = 0; k < gas.n_species; k++) {
            if (k != k_v) n_dry += y_gas(k) * gas.thermo.inv_W(k);
        }
        const double W_v = 1.0 / gas.thermo.inv_W(k_v);
        const double W_dry = n_dry > 0.0 ? (1.0 - Y_v) / n_dry : W_v;
        double Y_s = X_s * W_v / (X_s * W_v + (1.0 - X_s) * W_dry);
        const bool saturated = boiling || !(Y_s < 1.0 - 1e-12);
        if (saturated) Y_s = 1.0;
        // The film, by the 1/3 rule
        const double T_f = Ts + (t.T_gas - Ts) / 3.0;
        const double Y_f = Y_s + (Y_v - Y_s) / 3.0;
        const FilmFractions y_film{Y, k_v, Y_f, 1.0 - Y_v > 1e-12 ? (1.0 - Y_f) / (1.0 - Y_v) : 0.0};
        const double R_f = gas.thermo.gas_constant(y_film);
        const double rho_f = p / (R_f * T_f);
        const double cp_f = gas.thermo.cp_mass(T_f, y_film);
        double mu_f, k_f;
        gas.transport.properties(T_f, p, rho_f, cp_f, y_film, mu_f, k_f, work);
        const double D_f = work[k_v];
        const auto Tf_powers = chemistry::ThermoTable<>::powers(T_f);
        const double cp_v = chemistry::GAS_CONSTANT * gas.thermo.inv_W(k_v) * gas.thermo.cp_R(k_v, Tf_powers);
        const auto Ts_powers = chemistry::ThermoTable<>::powers(Ts);
        const double h_v = chemistry::GAS_CONSTANT * Ts * gas.thermo.inv_W(k_v) * gas.thermo.h_RT(k_v, Ts_powers);
        t.h_vapor = h_v - liquid.h_ref;
        t.latent = t.h_vapor - liquid.sensible(Ts);
        // Drag and forces, in the film's viscosity
        const double cp_gas = gas.thermo.cp_mass(t.T_gas, y_gas);
        const Solid::Response r = motion.motion(W, G, v, static_cast<rtype>(Tp), static_cast<rtype>(t.T_gas),
                                         static_cast<rtype>(mu_f), static_cast<rtype>(cp_gas / (cp_gas - R_gas)),
                                         static_cast<rtype>(d));
        t.tau = r.tau;
        FOR_I_DIM t.a[i] = r.a[i];
        // Ranz-Marshall, then the film corrections
        const double Re = static_cast<double>(Solid::reynolds(W, v, static_cast<rtype>(mu_f), static_cast<rtype>(d)));
        const double Pr = mu_f * cp_f / k_f, Sc = mu_f / (rho_f * D_f);
        const double Sh0 = 2.0 + 0.6 * Kokkos::sqrt(Re) * Kokkos::cbrt(Sc);
        const double Nu0 = 2.0 + 0.6 * Kokkos::sqrt(Re) * Kokkos::cbrt(Pr);
        const double dT = t.T_gas - Ts;
        double Nu = Nu0, G_T = 1.0;
        if (saturated) {
            // Boiling: the heat received evaporates the droplet, and its superheat relaxes at its heating rate
            t.Q = pi * d * k_f * Nu0 * dT;
            t.heat_up = liquid.density * liquid.specific_heat * d * d / (6.0 * k_f * Nu0);
            const double superheat = M * liquid.specific_heat * (Tp - Ts) / t.heat_up;
            t.mdot = Kokkos::fmax(t.Q + superheat, 0.0) / t.latent;
            return t;
        }
        const double B_M = (Y_s - Y_v) / (1.0 - Y_s);
        if (model == Evaporation::D2_LAW) {
            t.mdot = pi * d * rho_f * D_f * Sh0 * Kokkos::log1p(B_M);
        } else {
            const double Sh = 2.0 + (Sh0 - 2.0) / film_correction(B_M);
            t.mdot = pi * d * rho_f * D_f * Sh * Kokkos::log1p(B_M);
            const double Le = k_f / (rho_f * cp_f * D_f);
            double B_T = B_M;
            for (int it = 0; it < 8; it++) {
                Nu = 2.0 + (Nu0 - 2.0) / film_correction(B_T);
                const double phi = cp_v / cp_f * Sh / Nu / Le;
                B_T = Kokkos::pow(1.0 + B_M, phi) - 1.0;
            }
            Nu = 2.0 + (Nu0 - 2.0) / film_correction(B_T);
            G_T = log_ratio(B_T);
        }
        t.Q = pi * d * k_f * Nu * G_T * dT;
        t.heat_up = liquid.density * liquid.specific_heat * d * d / (6.0 * k_f * Nu * G_T);
        return t;
    }

    /** @brief Values per droplet that the first stage of a step keeps. */
    KOKKOS_INLINE_FUNCTION static constexpr uint32_t n_start() { return N_START; }

    /**
     * @brief The rates of a stage: dx/dt into rate[0, N_DIM), then dP/dt, dE/dt and dM/dt, of a droplet whose
     *        block values after its position are y, in gas W with gradients G and mass fractions Y, for a step dt.
     *        The first stage of a step writes start, which every stage reads. The velocity relaxes as a solid
     *        particle's (Solid::rates, explicit or exponential by dt / tau); the temperature relaxes toward its
     *        equilibrium of heating and evaporative cooling, linearized at the step's start, exponentially where
     *        dt is above its relaxation time. The evaporation rate is at most the step's starting mass over dt,
     *        so that no droplet's mass turns negative: a droplet that would evaporate within the step
     *        evaporates at that rate, at its temperature.
     * @param work n_species values of work memory.
     */
    KOKKOS_INLINE_FUNCTION void rates(const rtype * W, const rtype * G, const rtype * Y, const rtype * y,
                                      const rtype dt, const bool first, rtype * start, rtype * rate,
                                      double * work) const {
        const double M = static_cast<double>(y[MASS]);
        const rtype * u = W + 1;
        if (!(M >= minimum_mass) || !(M > 0.0)) {
            FOR_I_DIM rate[i] = u[i];
            for (uint32_t k = 0; k < N_VALUES; k++) rate[N_DIM + k] = 0.0_r;
            if (first) {
                for (uint32_t k = 0; k < N_START; k++) start[k] = 0.0_r;
                start[DT] = dt;
            }
            return;
        }
        rtype v[N_DIM];
        velocity(y, v);
        const double cp = liquid.specific_heat;
        const double Tp = temperature(y);
        // Properties of a droplet driven past its critical temperature are those at it
        const double T_eval = Kokkos::fmin(Tp, liquid.critical_temperature);
        const Transfer t = transfer(W, G, Y, v, T_eval, M, work);
        if (first) {
            FOR_I_DIM {
                start[V + i] = v[i];
                start[U + i] = u[i];
                start[A + i] = t.a[i];
            }
            start[TAU] = t.tau;
            start[DT] = dt;
            start[TP] = static_cast<rtype>(Tp);
            start[M0] = static_cast<rtype>(M);
            // dT/dt = f(T), linearized by a backward difference (toward the boiling side's opposite)
            const double f0 = heating(t, M, static_cast<double>(dt)) / (M * cp);
            double tau_T = 0.0;
            const double delta = 1e-6 * T_eval;
            const Transfer cooler = transfer(W, G, Y, v, T_eval - delta, M, work);
            const double f1 = heating(cooler, M, static_cast<double>(dt)) / (M * cp);
            const double J = (f0 - f1) / delta;
            if (J < 0.0) tau_T = -1.0 / J;
            start[TAU_T] = static_cast<rtype>(tau_T);
            start[T_EQ] = static_cast<rtype>(Tp + tau_T * f0);
        }
        const double M_start = static_cast<double>(start[M0]);
        const double mdot = evaporation(t, M_start, static_cast<double>(dt));
        const double f = heating(t, M_start, static_cast<double>(dt)) / (M * cp);
        // Velocity, as a solid particle's
        rtype dv[N_DIM];
        const rtype h = dt / start[TAU];
        if (h > EXPONENTIAL_ABOVE) {
            const Exponential e(h);
            FOR_I_DIM {
                const rtype v_eq = u[i] + start[TAU] * t.a[i];
                const rtype v_eq0 = start[U + i] + start[TAU] * start[A + i], v0 = start[V + i];
                rate[i] = v_eq0 + e.carry * (v0 - v_eq0) + e.drift * (v_eq - v_eq0);
                dv[i] = (e.decay * (v_eq0 - v0) + e.forcing * (v_eq - v_eq0)) / dt;
            }
        } else {
            FOR_I_DIM {
                rate[i] = v[i];
                dv[i] = (u[i] - v[i]) / t.tau + t.a[i];
            }
        }
        // Temperature
        double dT = f;
        const double tau_T = static_cast<double>(start[TAU_T]);
        if (tau_T > 0.0 && static_cast<double>(dt) > static_cast<double>(EXPONENTIAL_ABOVE) * tau_T) {
            const Exponential e(static_cast<rtype>(static_cast<double>(dt) / tau_T));
            const double T_eq0 = static_cast<double>(start[T_EQ]), T0 = static_cast<double>(start[TP]);
            const double T_eq = Tp + tau_T * f;
            dT = (static_cast<double>(e.decay) * (T_eq0 - T0) + static_cast<double>(e.forcing) * (T_eq - T_eq0)) /
                 static_cast<double>(dt);
        }
        // The conserved values: dM = -mdot, dP = M dv + dM v, dE = dM e + M c dT + v . dP - dM |v|^2
        const double dM = -mdot;
        double v2 = 0.0, v_dP = 0.0;
        FOR_I_DIM {
            const double dP = M * static_cast<double>(dv[i]) + dM * static_cast<double>(v[i]);
            rate[N_DIM + MOMENTUM + i] = static_cast<rtype>(dP);
            v2 += static_cast<double>(v[i]) * static_cast<double>(v[i]);
            v_dP += static_cast<double>(v[i]) * dP;
        }
        const double e = liquid.sensible(Tp) + 0.5 * v2;
        rate[N_DIM + ENERGY] = static_cast<rtype>(dM * e + M * cp * dT + v_dP - dM * v2);
        rate[N_DIM + MASS] = static_cast<rtype>(dM);
    }

    /** @brief The evaporation rate of t, at most M_start / dt (dt > 0). */
    KOKKOS_INLINE_FUNCTION static double evaporation(const Transfer & t, const double M_start, const double dt) {
        return dt > 0.0 ? Kokkos::fmin(t.mdot, M_start / dt) : t.mdot;
    }

    /**
     * @brief The heat into the liquid: what the gas gives less the latent heat of the evaporation. A droplet
     *        that evaporation() limits evaporates within the step at its temperature.
     */
    KOKKOS_INLINE_FUNCTION static double heating(const Transfer & t, const double M_start, const double dt) {
        if (dt > 0.0 && t.mdot > M_start / dt) return 0.0;
        return t.Q - t.mdot * t.latent;
    }

    /**
     * @brief What a droplet with block values y after its position and stage rates rate (of rates()) gives
     *        the gas per unit time, out[0, N_CONSERVATIVE): the opposite of its changes of mass, momentum less
     *        gravity's, and energy (from the gas's reference: E + M h_l(T_b)) less gravity's work. The vapor's
     *        mass is out[0].
     */
    KOKKOS_INLINE_FUNCTION void exchange(const rtype * y, const rtype * rate, rtype * out) const {
        const double M = static_cast<double>(y[MASS]);
        const double dM = static_cast<double>(rate[N_DIM + MASS]);
        double gravity_work = 0.0;
        out[0] = static_cast<rtype>(-dM);
        FOR_I_DIM {
            out[1 + i] = static_cast<rtype>(-(static_cast<double>(rate[N_DIM + MOMENTUM + i]) -
                                              M * static_cast<double>(motion.gravity[i])));
            gravity_work += static_cast<double>(motion.gravity[i]) * static_cast<double>(y[MOMENTUM + i]);
        }
        out[N_DIM + 1] = static_cast<rtype>(-(static_cast<double>(rate[N_DIM + ENERGY]) + dM * liquid.h_ref - gravity_work));
    }

    /** @brief What the gas receives when a droplet with block values y is removed: its mass, momentum and energy. */
    KOKKOS_INLINE_FUNCTION void remains(const rtype * y, double * out) const {
        const double M = static_cast<double>(y[MASS]);
        out[0] = M;
        FOR_I_DIM out[1 + i] = static_cast<double>(y[MOMENTUM + i]);
        out[N_DIM + 1] = static_cast<double>(y[ENERGY]) + M * liquid.h_ref;
    }
};

} // namespace particles

#endif // PARTICLE_EVAPORATION_H

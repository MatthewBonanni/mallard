/**
 * @file physics.h
 * @author Matthew Bonanni (mbonanni001@gmail.com)
 * @brief Physics model declaration.
 * @version 0.2
 * @date 2023-12-27
 *
 * @copyright Copyright (c) 2023 Matthew Bonanni
 *
 */

#ifndef PHYSICS_H
#define PHYSICS_H

#include <string>
#include <unordered_map>

#include <Kokkos_Core.hpp>
#include <toml.hpp>

#include "common.h"

enum class PhysicsType {
    EULER,
    NAVIER_STOKES
};

static const std::unordered_map<std::string, PhysicsType> PHYSICS_TYPES = {
    {"euler", PhysicsType::EULER},
    {"navier_stokes", PhysicsType::NAVIER_STOKES}
};

static const std::unordered_map<PhysicsType, std::string> PHYSICS_NAMES = {
    {PhysicsType::EULER, "euler"},
    {PhysicsType::NAVIER_STOKES, "navier_stokes"}
};

enum class ViscosityModel {
    NONE,
    CONSTANT,
    SUTHERLAND
};

/**
 * @brief Calorically perfect gas, inviscid (Euler) or viscous and heat
 *        conducting (Navier-Stokes, Stokes hypothesis, constant Prandtl number).
 *
 * Plain aggregate so it can be captured by value in device kernels.
 * Primitive layout: [u_x, u_y, (u_z,) p, T, h] (see PRIMITIVE_NAMES).
 * Reconstruction layout ("W"): [rho, u_x, u_y, (u_z,) p].
 */
struct Euler {
    rtype gamma = 1.4;
    rtype R = 287.0;
    rtype cp = 1004.5;
    rtype cv = 717.5;
    ViscosityModel viscosity_model = ViscosityModel::NONE;
    rtype mu_ref = 0.0;      // Viscosity (constant) or at T_mu_ref (Sutherland)
    rtype T_mu_ref = 273.15;
    rtype S_mu = 110.4;      // Sutherland temperature
    rtype Pr = 0.72;

    /**
     * @brief Construct from gamma and a reference state (p_ref = rho_ref * R * T_ref).
     */
    static Euler from_reference(rtype gamma, rtype p_ref, rtype T_ref, rtype rho_ref);

    /**
     * @brief Construct from the [physics] table of an input file.
     */
    static Euler from_input(const toml::value & input);

    void print() const;

    KOKKOS_INLINE_FUNCTION
    PhysicsType get_type() const {
        return viscosity_model == ViscosityModel::NONE ? PhysicsType::EULER : PhysicsType::NAVIER_STOKES;
    }

    KOKKOS_INLINE_FUNCTION
    bool is_viscous() const { return viscosity_model != ViscosityModel::NONE; }

    /**
     * @brief Dynamic viscosity at temperature T.
     */
    KOKKOS_INLINE_FUNCTION
    rtype viscosity(const rtype T) const {
        switch (viscosity_model) {
            case ViscosityModel::CONSTANT:
                return mu_ref;
            case ViscosityModel::SUTHERLAND:
                return mu_ref * Kokkos::pow(T / T_mu_ref, 1.5) * (T_mu_ref + S_mu) / (T + S_mu);
            default:
                return 0.0;
        }
    }

    /**
     * @brief Thermal conductivity for dynamic viscosity mu.
     */
    KOKKOS_INLINE_FUNCTION
    rtype conductivity(const rtype mu) const { return mu * cp / Pr; }

    KOKKOS_INLINE_FUNCTION
    rtype get_gamma() const { return gamma; }

    KOKKOS_INLINE_FUNCTION
    rtype get_energy_from_temperature(const rtype T) const { return cv * T; }

    KOKKOS_INLINE_FUNCTION
    rtype get_temperature_from_energy(const rtype e) const { return e / cv; }

    KOKKOS_INLINE_FUNCTION
    rtype get_density_from_pressure_temperature(const rtype p, const rtype T) const {
        return p / (R * T);
    }

    KOKKOS_INLINE_FUNCTION
    rtype get_temperature_from_density_pressure(const rtype rho, const rtype p) const {
        return p / (rho * R);
    }

    KOKKOS_INLINE_FUNCTION
    rtype get_pressure_from_density_temperature(const rtype rho, const rtype T) const {
        return rho * R * T;
    }

    KOKKOS_INLINE_FUNCTION
    rtype get_pressure_from_density_energy(const rtype rho, const rtype e) const {
        return (gamma - 1.0) * rho * e;
    }

    KOKKOS_INLINE_FUNCTION
    rtype get_sound_speed_from_pressure_density(const rtype p, const rtype rho) const {
        return Kokkos::sqrt(gamma * p / rho);
    }

    /**
     * @brief Conservatives [rho, rho u, rho E] -> primitives [u, p, T, h].
     */
    KOKKOS_INLINE_FUNCTION
    void compute_primitives_from_conservatives(rtype * primitives,
                                               const rtype * conservatives) const {
        const rtype rho = conservatives[0];
        rtype u[N_DIM];
        FOR_I_DIM u[i] = conservatives[1 + i] / rho;
        const rtype e = conservatives[N_DIM + 1] / rho - 0.5 * dot<N_DIM>(u, u);
        const rtype p = get_pressure_from_density_energy(rho, e);
        FOR_I_DIM primitives[i] = u[i];
        primitives[N_DIM] = p;
        primitives[N_DIM + 1] = get_temperature_from_energy(e);
        primitives[N_DIM + 2] = e + p / rho;
    }

    /**
     * @brief Conservatives -> W = [rho, u, p].
     */
    KOKKOS_INLINE_FUNCTION
    void compute_W_from_conservatives(rtype * W, const rtype * conservatives) const {
        const rtype rho = conservatives[0];
        W[0] = rho;
        FOR_I_DIM W[1 + i] = conservatives[1 + i] / rho;
        W[N_DIM + 1] = (gamma - 1.0) * (conservatives[N_DIM + 1] - 0.5 * rho * dot<N_DIM>(W + 1, W + 1));
    }

    /**
     * @brief W = [rho, u, p] -> conservatives.
     */
    KOKKOS_INLINE_FUNCTION
    void compute_conservatives_from_W(rtype * conservatives, const rtype * W) const {
        conservatives[0] = W[0];
        FOR_I_DIM conservatives[1 + i] = W[0] * W[1 + i];
        conservatives[N_DIM + 1] = W[N_DIM + 1] / (gamma - 1.0) + 0.5 * W[0] * dot<N_DIM>(W + 1, W + 1);
    }
};

#endif // PHYSICS_H

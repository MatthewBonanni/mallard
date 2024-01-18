/**
 * @file physics.h
 * @author Matthew Bonanni (mbonanni001@gmail.com)
 * @brief Physics class declaration.
 * @version 0.1
 * @date 2023-12-27
 * 
 * @copyright Copyright (c) 2023 Matthew Bonanni
 * 
 */

#ifndef PHYSICS_H
#define PHYSICS_H

#include <unordered_map>
#include <string>

#include <Kokkos_Core.hpp>
#include <toml++/toml.h>

#include "common.h"

enum class PhysicsType {
    EULER
};

static const std::unordered_map<std::string, PhysicsType> PHYSICS_TYPES = {
    {"euler", PhysicsType::EULER}
};

static const std::unordered_map<PhysicsType, std::string> PHYSICS_NAMES = {
    {PhysicsType::EULER, "euler"}
};

class Physics {
    public:
        /**
         * @brief Construct a new Physics object.
         */
        Physics();

        /**
         * @brief Destroy the Physics object.
         */
        virtual ~Physics();

        /**
         * @brief Initialize the physics.
         * @param input Input file
         */
        virtual void init(const toml::table & input);

        /**
         * @brief Print the physics.
         */
        virtual void print() const;

        /**
         * @brief Get gamma.
         * @return Gamma
         */
        KOKKOS_INLINE_FUNCTION
        virtual rtype get_gamma() const = 0;

        /**
         * @brief Get energy from temperature.
         * @param T Temperature
         * @return Energy
         */
        KOKKOS_INLINE_FUNCTION
        virtual rtype get_energy_from_temperature(const rtype & T) const = 0;

        /**
         * @brief Get temperature from energy.
         * @param e Energy
         * @return Temperature
         */
        KOKKOS_INLINE_FUNCTION
        virtual rtype get_temperature_from_energy(const rtype & e) const = 0;

        /**
         * @brief Get density from pressure and temperature.
         * @param p Pressure
         * @param T Temperature
         * @return Density
         */
        KOKKOS_INLINE_FUNCTION
        virtual rtype get_density_from_pressure_temperature(const rtype & p,
                                                            const rtype & T) const = 0;
        
        /**
         * @brief Get temperature from density and pressure.
         * @param rho Density
         * @param p Pressure
         * @return Temperature
         */
        KOKKOS_INLINE_FUNCTION
        virtual rtype get_temperature_from_density_pressure(const rtype & rho,
                                                            const rtype & p) const = 0;
        
        /**
         * @brief Get pressure from density and temperature.
         * @param rho Density
         * @param T Temperature
         * @return Pressure
         */
        KOKKOS_INLINE_FUNCTION
        virtual rtype get_pressure_from_density_temperature(const rtype & rho,
                                                            const rtype & T) const = 0;
        
        /**
         * @brief Get sound speed from pressure and density.
         * @param p Pressure
         * @param rho Density
         */
        KOKKOS_INLINE_FUNCTION
        virtual rtype get_sound_speed_from_pressure_density(const rtype & p,
                                                            const rtype & rho) const = 0;
        
        /**
         * @brief Compute primitive variables from conservative variables.
         * @param primitives Primitive variables
         * @param conservatives Conservative variables
         */
        KOKKOS_INLINE_FUNCTION
        virtual void compute_primitives_from_conservatives(Primitives & primitives,
                                                           const State & conservatives) const = 0;

        /**
         * @brief Calculate the diffusive flux
         */
        virtual void calc_diffusive_flux(State & flux) = 0;
    protected:
        PhysicsType type;
    private:
};

class Euler : public Physics {
    public:
        /**
         * @brief Construct a new Euler object
         */
        Euler();

        /**
         * @brief Destroy the Euler object
         */
        ~Euler();

        /**
         * @brief Initialize the physics.
         * @param input Input file.
         */
        void init(const toml::table & input) override;

        /**
         * @brief Initialize the physics manually.
         * @param gamma Gamma
         * @param p_ref Reference pressure
         * @param T_ref Reference temperature
         * @param rho_ref Reference density
         */
        void init(const rtype & gamma,
                  const rtype & p_ref,
                  const rtype & T_ref,
                  const rtype & rho_ref);

        /**
         * @brief Print the physics.
         */
        void print() const override;

        /**
         * @brief Get gamma
         * @return gamma
         */
        KOKKOS_INLINE_FUNCTION
        rtype get_gamma() const override { return gamma; }

        /**
         * @brief Get energy from temperature.
         * @param T Temperature
         * @return Energy
         */
        KOKKOS_INLINE_FUNCTION
        rtype get_energy_from_temperature(const rtype & T) const override;

        /**
         * @brief Get temperature from energy.
         * @param e Energy
         * @return Temperature
         */
        KOKKOS_INLINE_FUNCTION
        rtype get_temperature_from_energy(const rtype & e) const override;

        /**
         * @brief Get density from pressure and temperature.
         * @param p Pressure
         * @param T Temperature
         * @return Density
         */
        KOKKOS_INLINE_FUNCTION
        rtype get_density_from_pressure_temperature(const rtype & p,
                                                    const rtype & T) const override;
        
        /**
         * @brief Get temperature from density and pressure.
         * @param rho Density
         * @param p Pressure
         * @return Temperature
         */
        KOKKOS_INLINE_FUNCTION
        rtype get_temperature_from_density_pressure(const rtype & rho,
                                                    const rtype & p) const override;
        
        /**
         * @brief Get pressure from density and temperature.
         * @param rho Density
         * @param T Temperature
         * @return Pressure
         */
        KOKKOS_INLINE_FUNCTION
        rtype get_pressure_from_density_temperature(const rtype & rho,
                                                    const rtype & T) const override;
        
        /**
         * @brief Get sound speed from pressure and density.
         * @param p Pressure
         * @param rho Density
         */
        KOKKOS_INLINE_FUNCTION
        rtype get_sound_speed_from_pressure_density(const rtype & p,
                                                    const rtype & rho) const override;

        /**
         * @brief Compute primitive variables from conservative variables.
         * @param primitives Primitive variables
         * @param conservatives Conservative variables
         */
        KOKKOS_INLINE_FUNCTION
        void compute_primitives_from_conservatives(Primitives & primitives,
                                                   const State & conservatives) const override;

        /**
         * @brief Calculate the diffusive flux
         */
        void calc_diffusive_flux(State & flux) override;
    protected:
    private:
        void set_R_cp_cv();

        rtype gamma;
        rtype p_ref, T_ref, rho_ref;
        rtype R, cp, cv;
};

#endif // PHYSICS_H
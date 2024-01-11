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

#include "common.h"

#include <toml++/toml.h>

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
        virtual double get_gamma() const = 0;

        /**
         * @brief Get energy from temperature.
         * @param T Temperature
         * @return Energy
         */
        virtual double get_energy_from_temperature(const double & T) const = 0;

        /**
         * @brief Get temperature from energy.
         * @param e Energy
         * @return Temperature
         */
        virtual double get_temperature_from_energy(const double & e) const = 0;

        /**
         * @brief Get density from pressure and temperature.
         * @param p Pressure
         * @param T Temperature
         * @return Density
         */
        virtual double get_density_from_pressure_temperature(const double & p,
                                                             const double & T) const = 0;
        
        /**
         * @brief Get sound speed from pressure and density.
         * @param p Pressure
         * @param rho Density
         */
        virtual double get_sound_speed_from_pressure_density(const double & p,
                                                             const double & rho) const = 0;
        
        /**
         * @brief Compute primitive variables from conservative variables.
         * @param primitives Primitive variables
         * @param conservatives Conservative variables
         */
        virtual void compute_primitives_from_conservatives(Primitives & primitives,
                                                           const State & conservatives) const = 0;

        /**
         * @brief Calculate the Euler flux
         * @param flux Flux vector
         * @param n_unit Unit normal vector
         * @param rho_l Left density
         * @param u_l Left velocity
         * @param p_l Left pressure
         * @param gamma_l Left gamma
         * @param h_l Left enthalpy
         * @param rho_r Right density
         * @param u_r Right velocity
         * @param p_r Right pressure
         * @param gamma_r Right gamma
         * @param h_r Right enthalpy
         */
        void calc_euler_flux(State & flux, const NVector & n_unit,
                             const double rho_l, const NVector & u_l,
                             const double p_l, const double gamma_l, const double h_l,
                             const double rho_r, const NVector & u_r,
                             const double p_r, const double gamma_r, const double h_r);

        /**
         * @brief Calculate the Euler flux (alias for above)
         * @param flux Flux vector
         * @param n_unit Unit normal vector
         * @param rho_l Left density
         * @param rho_r Right density
         * @param primitives_l Left primitive variables
         * @param primitives_r Right primitive variables
         */
        void calc_euler_flux(State & flux, const NVector & n_unit,
                             const double rho_l, const double rho_r,
                             const Primitives & primitives_l,
                             const Primitives & primitives_r);

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
        void init(const double & gamma,
                  const double & p_ref,
                  const double & T_ref,
                  const double & rho_ref);

        /**
         * @brief Print the physics.
         */
        void print() const override;

        /**
         * @brief Get gamma
         * @return gamma
         */
        double get_gamma() const override { return gamma; }

        /**
         * @brief Get energy from temperature.
         * @param T Temperature
         * @return Energy
         */
        double get_energy_from_temperature(const double & T) const override;

        /**
         * @brief Get temperature from energy.
         * @param e Energy
         * @return Temperature
         */
        double get_temperature_from_energy(const double & e) const override;

        /**
         * @brief Get density from pressure and temperature.
         * @param p Pressure
         * @param T Temperature
         * @return Density
         */
        double get_density_from_pressure_temperature(const double & p,
                                                     const double & T) const override;
        
        /**
         * @brief Get sound speed from pressure and density.
         * @param p Pressure
         * @param rho Density
         */
        double get_sound_speed_from_pressure_density(const double & p,
                                                     const double & rho) const override;

        /**
         * @brief Compute primitive variables from conservative variables.
         * @param primitives Primitive variables
         * @param conservatives Conservative variables
         */
        void compute_primitives_from_conservatives(Primitives & primitives,
                                                   const State & conservatives) const override;

        /**
         * @brief Calculate the diffusive flux
         */
        void calc_diffusive_flux(State & flux) override;
    protected:
    private:
        void set_R_cp_cv();

        double gamma;
        double p_ref, T_ref, rho_ref;
        double R, cp, cv;
};

#endif // PHYSICS_H
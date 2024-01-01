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

#include "common/common.h"

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
         */
        void calc_euler_flux(State & flux, const NVector & n_vec,
                             const double rho_l, const NVector & u_l,
                             const double p_l, const double gamma_l, const double H_l,
                             const double rho_r, const NVector & u_r,
                             const double p_r, const double gamma_r, const double H_r);

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
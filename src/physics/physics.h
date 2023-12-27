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
         * @brief Construct a new Physics object
         */
        Physics();

        /**
         * @brief Destroy the Physics object
         */
        virtual ~Physics();

        /**
         * @brief Initialize the physics.
         * @param input Input file.
         */
        virtual void init(const toml::table & input);

        /**
         * @brief Print the physics.
         */
        virtual void print() const;

        /**
         * @brief Get gamma
         * @return gamma
         */
        virtual double get_gamma() const = 0;

        /**
         * @brief Calculate the Euler flux
         */
        void calc_euler_flux(State & flux, const std::array<double, 2> & n_vec,
                             const double rho_l, const std::array<double, 2> & u_l,
                             const double p_l, const double gamma_l, const double H_l,
                             const double rho_r, const std::array<double, 2> & u_r,
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
         * @brief Calculate the diffusive flux
         */
        void calc_diffusive_flux(State & flux) override;
    protected:
        double gamma;
    private:
};

#endif // PHYSICS_H
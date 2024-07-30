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
#include <toml.hpp>

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
         * @param input TOML input data.
         */
        virtual void init(const toml::value & input);

        /**
         * @brief Print the physics.
         */
        virtual void print() const;

        /**
         * @brief Get the physics type.
         * @return Physics type
         */
        KOKKOS_INLINE_FUNCTION
        PhysicsType get_type() const { return type; }

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
         * @brief Get pressure from density and energy.
         * @param rho Density
         * @param e Energy
         * @return Pressure
         */
        KOKKOS_INLINE_FUNCTION
        virtual rtype get_pressure_from_density_energy(const rtype & rho,
                                                       const rtype & e) const = 0;
        
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
         * @param primitives Pointer to primitive variable array
         * @param conservatives Pointer to conservative variable array
         */
        KOKKOS_INLINE_FUNCTION
        virtual void compute_primitives_from_conservatives(rtype * primitives,
                                                           const rtype * conservatives) const = 0;
        
        /**
         * @brief Copy data from host to device.
         */
        virtual void copy_host_to_device();

        /**
         * @brief Copy data from device to host.
         */
        virtual void copy_device_to_host();
    protected:
        PhysicsType type;
        Kokkos::View<rtype [2]> p_bounds;
        Kokkos::View<rtype [2]>::HostMirror h_p_bounds;
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
         * @param input TOML input data.
         */
        void init(const toml::value & input) override;

        /**
         * @brief Initialize the physics manually.
         * FOR TESTING PURPOSES ONLY
         * @param p_min Minimum pressure
         * @param p_max Maximum pressure
         * @param gamma Gamma
         * @param p_ref Reference pressure
         * @param T_ref Reference temperature
         * @param rho_ref Reference density
         */
        void init(rtype p_min, rtype p_max, rtype gamma,
                  rtype p_ref, rtype T_ref, rtype rho_ref);

        /**
         * @brief Print the physics.
         */
        void print() const override;

        /**
         * @brief Get gamma
         * @return gamma
         */
        KOKKOS_INLINE_FUNCTION
        rtype get_gamma() const override { return constants(i_gamma); }

        /**
         * @brief Get gamma - host version
         * @return gamma
         */
        rtype get_h_gamma() const { return h_constants(i_gamma); }

        /**
         * @brief Get R
         * @return R
         */
        KOKKOS_INLINE_FUNCTION
        rtype get_R() const { return constants(i_R); }

        /**
         * @brief Get R - host version
         * @return R
         */
        rtype get_h_R() const { return h_constants(i_R); }

        /**
         * @brief Get Cp
         * @return Cp
         */
        KOKKOS_INLINE_FUNCTION
        rtype get_Cp() const { return constants(i_cp); }

        /**
         * @brief Get Cp - host version
         * @return Cp
         */
        rtype get_h_Cp() const { return h_constants(i_cp); }

        /**
         * @brief Get Cv
         * @return Cv
         */
        KOKKOS_INLINE_FUNCTION
        rtype get_Cv() const { return constants(i_cv); }

        /**
         * @brief Get Cv - host version
         * @return Cv
         */
        rtype get_h_Cv() const { return h_constants(i_cv); }

        /**
         * @brief Get energy from temperature.
         * @param T Temperature
         * @return Energy
         */
        KOKKOS_INLINE_FUNCTION
        rtype get_energy_from_temperature(const rtype & T) const override;

        /**
         * @brief Get energy from temperature - host version.
         * @param T Temperature
         * @return Energy
         */
        rtype h_get_energy_from_temperature(const rtype & T) const {
            return get_h_Cv() * T;
        }

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
         * @brief Get density from pressure and temperature - host version.
         * @param p Pressure
         * @param T Temperature
         * @return Density
         */
        rtype h_get_density_from_pressure_temperature(const rtype & p,
                                                      const rtype & T) const {
            return p / (T * get_h_R());
        }
        
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
         * @brief Get pressure from density and energy.
         * @param rho Density
         * @param e Energy
         * @return Pressure
         */
        KOKKOS_INLINE_FUNCTION
        rtype get_pressure_from_density_energy(const rtype & rho,
                                               const rtype & e) const override;
        
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
         * @param primitives Pointer to primitive variable array
         * @param conservatives Pointer to conservative variable array
         */
        KOKKOS_INLINE_FUNCTION
        void compute_primitives_from_conservatives(rtype * primitives,
                                                   const rtype * conservatives) const override;
        
        /**
         * @brief Copy data from host to device.
         */
        void copy_host_to_device() override;

        /**
         * @brief Copy data from device to host.
         */
        void copy_device_to_host() override;
    protected:
    private:
        void set_R_cp_cv();

        Kokkos::View<rtype [7]> constants;
        Kokkos::View<rtype [7]>::HostMirror h_constants;

        u_int8_t i_gamma = 0;
        u_int8_t i_p_ref = 1;
        u_int8_t i_T_ref = 2;
        u_int8_t i_rho_ref = 3;
        u_int8_t i_R = 4;
        u_int8_t i_cp = 5;
        u_int8_t i_cv = 6;
};

rtype Euler::get_energy_from_temperature(const rtype & T) const {
    return get_Cv() * T;
}

rtype Euler::get_temperature_from_energy(const rtype & e) const {
    return e / get_Cv();
}

rtype Euler::get_density_from_pressure_temperature(const rtype & p,
                                                   const rtype & T) const {
    return p / (T * get_R());
}

rtype Euler::get_temperature_from_density_pressure(const rtype & rho,
                                                   const rtype & p) const {
    return p / (rho * get_R());
}

rtype Euler::get_pressure_from_density_temperature(const rtype & rho,
                                                   const rtype & T) const {
    return rho * get_R() * T;
}

rtype Euler::get_pressure_from_density_energy(const rtype & rho,
                                              const rtype & e) const {
    return Kokkos::fmax(p_bounds(0), Kokkos::fmin(p_bounds(1), (get_gamma() - 1.0) * rho * e));
}

rtype Euler::get_sound_speed_from_pressure_density(const rtype & p,
                                                   const rtype & rho) const {
    return Kokkos::sqrt(get_gamma() * p / rho);
}

void Euler::compute_primitives_from_conservatives(rtype * primitives,
                                                  const rtype * conservatives) const {
    rtype rho = conservatives[0];
    rtype u[N_DIM] = {conservatives[1] / rho,
                      conservatives[2] / rho};
    rtype E = conservatives[3] / rho;
    rtype e = E - 0.5 * dot<N_DIM>(u, u);
    rtype p = get_pressure_from_density_energy(rho, e);
    rtype T = get_temperature_from_energy(e);
    rtype h = e + p / rho;
    primitives[0] = u[0];
    primitives[1] = u[1];
    primitives[2] = p;
    primitives[3] = T;
    primitives[4] = h;
}

#endif // PHYSICS_H
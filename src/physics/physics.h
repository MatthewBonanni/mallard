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

// Common non-template base class
class PhysicsBase {
public:
    virtual ~PhysicsBase() = default;
    virtual void init(const toml::value & input) = 0;
    virtual void print() const = 0;
    virtual PhysicsType get_type() const = 0;
    virtual rtype get_gamma() const = 0;
    virtual rtype get_energy_from_temperature(const rtype & T) const = 0;
    virtual rtype h_get_energy_from_temperature(const rtype & T) const = 0;
    virtual rtype get_temperature_from_energy(const rtype & e) const = 0;
    virtual rtype h_get_temperature_from_energy(const rtype & e) const = 0;
    virtual rtype get_density_from_pressure_temperature(const rtype & p,
                                                        const rtype & T) const = 0;
    virtual rtype h_get_density_from_pressure_temperature(const rtype & p,
                                                          const rtype & T) const = 0;
    virtual rtype get_temperature_from_density_pressure(const rtype & rho,
                                                        const rtype & p) const = 0;
    virtual rtype h_get_temperature_from_density_pressure(const rtype & rho,
                                                          const rtype & p) const = 0;
    virtual rtype get_pressure_from_density_temperature(const rtype & rho,
                                                        const rtype & T) const = 0;
    virtual rtype h_get_pressure_from_density_temperature(const rtype & rho,
                                                          const rtype & T) const = 0;
    virtual rtype get_pressure_from_density_energy(const rtype & rho,
                                                   const rtype & e) const = 0;
    virtual rtype h_get_pressure_from_density_energy(const rtype & rho,
                                                     const rtype & e) const = 0;
    virtual rtype get_sound_speed_from_pressure_density(const rtype & p,
                                                        const rtype & rho) const = 0;
    virtual rtype h_get_sound_speed_from_pressure_density(const rtype & p,
                                                          const rtype & rho) const = 0;
    virtual void compute_primitives_from_conservatives(rtype * primitives,
                                                       const rtype * conservatives) const = 0;
    virtual void h_compute_primitives_from_conservatives(rtype * primitives,
                                                         const rtype * conservatives) const = 0;
    virtual void copy_host_to_device() = 0;
    virtual void copy_device_to_host() = 0;
};

// CRTP base class
template<typename Derived>
class Physics {
    public:
        /**
         * @brief Construct a new Physics Base object.
         */
        Physics() {}

        /**
         * @brief Destroy the Physics Base object.
         */
        virtual ~Physics() {}

        /**
         * @brief Initialize the physics.
         * @param input TOML input data.
         */
        virtual void init(const toml::value & input) = 0;

        /**
         * @brief Print the physics.
         */
        virtual void print() const = 0;

        /**
         * @brief Get the type of the physics.
         * @return Physics type
         */
        KOKKOS_INLINE_FUNCTION
        constexpr PhysicsType get_type() const { 
            return static_cast<const Derived*>(this)->get_type_impl(); 
        }

        /**
         * @brief Get gamma.
         * @return Gamma
         */
        KOKKOS_INLINE_FUNCTION
        rtype get_gamma() const {
            return static_cast<const Derived*>(this)->get_gamma_impl();
        }

        /**
         * @brief Get energy from temperature.
         * @param T Temperature
         * @return Energy
         */
        KOKKOS_INLINE_FUNCTION
        rtype get_energy_from_temperature(const rtype & T) const {
            return static_cast<const Derived*>(this)->get_energy_from_temperature_impl(T);
        }

        /**
         * @brief Get energy from temperature - host version.
         * @param T Temperature
         * @return Energy
         */
        rtype h_get_energy_from_temperature(const rtype & T) const {
            return static_cast<const Derived*>(this)->h_get_energy_from_temperature_impl(T);
        }

        /**
         * @brief Get temperature from energy.
         * @param e Energy
         * @return Temperature
         */
        KOKKOS_INLINE_FUNCTION
        rtype get_temperature_from_energy(const rtype & e) const {
            return static_cast<const Derived*>(this)->get_temperature_from_energy_impl(e);
        }

        /**
         * @brief Get temperature from energy - host version.
         * @param e Energy
         * @return Temperature
         */
        rtype h_get_temperature_from_energy(const rtype & e) const {
            return static_cast<const Derived*>(this)->h_get_temperature_from_energy_impl(e);
        }

        /**
         * @brief Get density from pressure and temperature.
         * @param p Pressure
         * @param T Temperature
         * @return Density
         */
        KOKKOS_INLINE_FUNCTION
        rtype get_density_from_pressure_temperature(const rtype & p,
                                                    const rtype & T) const {
            return static_cast<const Derived*>(this)->get_density_from_pressure_temperature_impl(p, T);
        }

        /**
         * @brief Get density from pressure and temperature - host version.
         * @param p Pressure
         * @param T Temperature
         * @return Density
         */
        rtype h_get_density_from_pressure_temperature(const rtype & p,
                                                      const rtype & T) const {
            return static_cast<const Derived*>(this)->h_get_density_from_pressure_temperature_impl(p, T);
        }

        /**
         * @brief Get temperature from density and pressure.
         * @param rho Density
         * @param p Pressure
         * @return Temperature
         */
        KOKKOS_INLINE_FUNCTION
        rtype get_temperature_from_density_pressure(const rtype & rho,
                                                    const rtype & p) const {
            return static_cast<const Derived*>(this)->get_temperature_from_density_pressure_impl(rho, p);
        }

        /**
         * @brief Get temperature from density and pressure - host version.
         * @param rho Density
         * @param p Pressure
         * @return Temperature
         */
        rtype h_get_temperature_from_density_pressure(const rtype & rho,
                                                      const rtype & p) const {
            return static_cast<const Derived*>(this)->h_get_temperature_from_density_pressure_impl(rho, p);
        }

        /**
         * @brief Get pressure from density and temperature.
         * @param rho Density
         * @param T Temperature
         * @return Pressure
         */
        KOKKOS_INLINE_FUNCTION
        rtype get_pressure_from_density_temperature(const rtype & rho,
                                                    const rtype & T) const {
            return static_cast<const Derived*>(this)->get_pressure_from_density_temperature_impl(rho, T);
        }

        /**
         * @brief Get pressure from density and temperature - host version.
         * @param rho Density
         * @param T Temperature
         * @return Pressure
         */
        rtype h_get_pressure_from_density_temperature(const rtype & rho,
                                                      const rtype & T) const {
            return static_cast<const Derived*>(this)->h_get_pressure_from_density_temperature_impl(rho, T);
        }

        /**
         * @brief Get pressure from density and energy.
         * @param rho Density
         * @param e Energy
         * @return Pressure
         */
        KOKKOS_INLINE_FUNCTION
        rtype get_pressure_from_density_energy(const rtype & rho,
                                               const rtype & e) const {
            return static_cast<const Derived*>(this)->get_pressure_from_density_energy_impl(rho, e);
        }

        /**
         * @brief Get pressure from density and energy - host version.
         * @param rho Density
         * @param e Energy
         * @return Pressure
         */
        rtype h_get_pressure_from_density_energy(const rtype & rho,
                                                 const rtype & e) const {
            return static_cast<const Derived*>(this)->h_get_pressure_from_density_energy_impl(rho, e);
        }

        /**
         * @brief Get sound speed from pressure and density.
         * @param p Pressure
         * @param rho Density
         */
        KOKKOS_INLINE_FUNCTION
        rtype get_sound_speed_from_pressure_density(const rtype & p,
                                                    const rtype & rho) const {
            return static_cast<const Derived*>(this)->get_sound_speed_from_pressure_density_impl(p, rho);
        }

        /**
         * @brief Get sound speed from pressure and density - host version.
         * @param p Pressure
         * @param rho Density
         */
        rtype h_get_sound_speed_from_pressure_density(const rtype & p,
                                                      const rtype & rho) const {
            return static_cast<const Derived*>(this)->h_get_sound_speed_from_pressure_density_impl(p, rho);
        }

        /**
         * @brief Compute primitive variables from conservative variables.
         * @param primitives Pointer to primitive variable array
         * @param conservatives Pointer to conservative variable array
         */
        KOKKOS_INLINE_FUNCTION
        void compute_primitives_from_conservatives(rtype * primitives,
                                                   const rtype * conservatives) const {
            static_cast<const Derived*>(this)->compute_primitives_from_conservatives_impl(primitives, conservatives);
        }

        /**
         * @brief Compute primitive variables from conservative variables - host version.
         * @param primitives Pointer to primitive variable array
         * @param conservatives Pointer to conservative variable array
         */
        void h_compute_primitives_from_conservatives(rtype * primitives,
                                                     const rtype * conservatives) const {
            static_cast<const Derived*>(this)->h_compute_primitives_from_conservatives_impl(primitives, conservatives);
        }

        /**
         * @brief Copy data from host to device.
         */
        virtual void copy_host_to_device() = 0;

        /**
         * @brief Copy data from device to host.
         */
        virtual void copy_device_to_host() = 0;
    protected:
    private:
};

// Euler physics class
class Euler : public Physics<Euler> {
    public:
        /**
         * @brief Construct a new Euler object.
         */
        Euler();

        /**
         * @brief Destroy the Euler object.
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
         * @brief Get the type of the physics.
         * @return Physics type
         */
        KOKKOS_INLINE_FUNCTION
        constexpr PhysicsType get_type_impl() const {
            return PhysicsType::EULER;
        }

        /**
         * @brief Get gamma.
         * @return Gamma
         */
        KOKKOS_INLINE_FUNCTION
        rtype get_gamma_impl() const {
            return constants(i_gamma);
        }

        /**
         * @brief Get gamma - host version.
         * @return Gamma
         */
        rtype get_h_gamma() const {
            return h_constants(i_gamma);
        }

        /**
         * @brief Get R.
         * @return R
         */
        KOKKOS_INLINE_FUNCTION
        rtype get_R() const {
            return constants(i_R);
        }

        /**
         * @brief Get R - host version.
         * @return R
         */
        rtype get_h_R() const {
            return h_constants(i_R);
        }

        /**
         * @brief Get Cp.
         * @return Cp
         */
        KOKKOS_INLINE_FUNCTION
        rtype get_Cp() const {
            return constants(i_cp);
        }

        /**
         * @brief Get Cp - host version.
         * @return Cp
         */
        rtype get_h_Cp() const {
            return h_constants(i_cp);
        }

        /**
         * @brief Get Cv.
         * @return Cv
         */
        KOKKOS_INLINE_FUNCTION
        rtype get_Cv() const {
            return constants(i_cv);
        }

        /**
         * @brief Get Cv - host version.
         * @return Cv
         */
        rtype get_h_Cv() const {
            return h_constants(i_cv);
        }

        /**
         * @brief Get minimum pressure.
         * @return Minimum pressure
         */
        KOKKOS_INLINE_FUNCTION
        rtype get_p_min() const {
            return constants(i_p_min);
        }

        /**
         * @brief Get maximum pressure.
         * @return Maximum pressure
         */
        KOKKOS_INLINE_FUNCTION
        rtype get_p_max() const {
            return constants(i_p_max);
        }

        /**
         * @brief Get energy from temperature.
         * @param T Temperature
         * @return Energy
         */
        KOKKOS_INLINE_FUNCTION
        rtype get_energy_from_temperature_impl(const rtype & T) const;

        /**
         * @brief Get energy from temperature - host version.
         * @param T Temperature
         * @return Energy
         */
        rtype h_get_energy_from_temperature_impl(const rtype & T) const;

        /**
         * @brief Get temperature from energy.
         * @param e Energy
         * @return Temperature
         */
        KOKKOS_INLINE_FUNCTION
        rtype get_temperature_from_energy_impl(const rtype & e) const;

        /**
         * @brief Get temperature from energy - host version.
         * @param e Energy
         * @return Temperature
         */
        rtype h_get_temperature_from_energy_impl(const rtype & e) const;

        /**
         * @brief Get density from pressure and temperature.
         * @param p Pressure
         * @param T Temperature
         * @return Density
         */
        KOKKOS_INLINE_FUNCTION
        rtype get_density_from_pressure_temperature_impl(const rtype & p,
                                                         const rtype & T) const;
        
        /**
         * @brief Get density from pressure and temperature - host version.
         * @param p Pressure
         * @param T Temperature
         * @return Density
         */
        rtype h_get_density_from_pressure_temperature_impl(const rtype & p,
                                                           const rtype & T) const;

        /**
         * @brief Get temperature from density and pressure.
         * @param rho Density
         * @param p Pressure
         * @return Temperature
         */
        KOKKOS_INLINE_FUNCTION
        rtype get_temperature_from_density_pressure_impl(const rtype & rho,
                                                         const rtype & p) const;
        
        /**
         * @brief Get temperature from density and pressure - host version.
         * @param rho Density
         * @param p Pressure
         * @return Temperature
         */
        rtype h_get_temperature_from_density_pressure_impl(const rtype & rho,
                                                           const rtype & p) const;

        /**
         * @brief Get pressure from density and temperature.
         * @param rho Density
         * @param T Temperature
         * @return Pressure
         */
        KOKKOS_INLINE_FUNCTION
        rtype get_pressure_from_density_temperature_impl(const rtype & rho,
                                                         const rtype & T) const;
        
        /**
         * @brief Get pressure from density and temperature - host version.
         * @param rho Density
         * @param T Temperature
         * @return Pressure
         */
        rtype h_get_pressure_from_density_temperature_impl(const rtype & rho,
                                                           const rtype & T) const;

        /**
         * @brief Get pressure from density and energy.
         * @param rho Density
         * @param e Energy
         * @return Pressure
         */
        KOKKOS_INLINE_FUNCTION
        rtype get_pressure_from_density_energy_impl(const rtype & rho,
                                                    const rtype & e) const;
        
        /**
         * @brief Get pressure from density and energy - host version.
         * @param rho Density
         * @param e Energy
         * @return Pressure
         */
        rtype h_get_pressure_from_density_energy_impl(const rtype & rho,
                                                      const rtype & e) const;

        /**
         * @brief Get sound speed from pressure and density.
         * @param p Pressure
         * @param rho Density
         */
        KOKKOS_INLINE_FUNCTION
        rtype get_sound_speed_from_pressure_density_impl(const rtype & p,
                                                         const rtype & rho) const;
        
        /**
         * @brief Get sound speed from pressure and density - host version.
         * @param p Pressure
         * @param rho Density
         */
        rtype h_get_sound_speed_from_pressure_density_impl(const rtype & p,
                                                           const rtype & rho) const;

        /**
         * @brief Compute primitive variables from conservative variables.
         * @param primitives Pointer to primitive variable array
         * @param conservatives Pointer to conservative variable array
         */
        KOKKOS_INLINE_FUNCTION
        void compute_primitives_from_conservatives_impl(rtype * primitives,
                                                        const rtype * conservatives) const;

        /**
         * @brief Compute primitive variables from conservative variables - host version.
         * @param primitives Pointer to primitive variable array
         * @param conservatives Pointer to conservative variable array
         */
        void h_compute_primitives_from_conservatives_impl(rtype * primitives,
                                                          const rtype * conservatives) const;

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

        Kokkos::View<rtype [9]> constants;
        Kokkos::View<rtype [9]>::HostMirror h_constants;

        uint8_t i_gamma = 0;
        uint8_t i_p_ref = 1;
        uint8_t i_T_ref = 2;
        uint8_t i_rho_ref = 3;
        uint8_t i_R = 4;
        uint8_t i_cp = 5;
        uint8_t i_cv = 6;
        uint8_t i_p_min = 7;
        uint8_t i_p_max = 8;
};

// Type-erased wrapper
class PhysicsWrapper : public PhysicsBase {
private:
    struct Concept {
        virtual ~Concept() = default;
        virtual void init(const toml::value & input) = 0;
        virtual void print() const = 0;
        virtual PhysicsType get_type() const = 0;
        virtual rtype get_gamma() const = 0;
        virtual rtype get_energy_from_temperature(const rtype & T) const = 0;
        virtual rtype h_get_energy_from_temperature(const rtype & T) const = 0;
        virtual rtype get_temperature_from_energy(const rtype & e) const = 0;
        virtual rtype h_get_temperature_from_energy(const rtype & e) const = 0;
        virtual rtype get_density_from_pressure_temperature(const rtype & p,
                                                            const rtype & T) const = 0;
        virtual rtype h_get_density_from_pressure_temperature(const rtype & p,
                                                              const rtype & T) const = 0;
        virtual rtype get_temperature_from_density_pressure(const rtype & rho,
                                                            const rtype & p) const = 0;
        virtual rtype h_get_temperature_from_density_pressure(const rtype & rho,
                                                              const rtype & p) const = 0;
        virtual rtype get_pressure_from_density_temperature(const rtype & rho,
                                                            const rtype & T) const = 0;
        virtual rtype h_get_pressure_from_density_temperature(const rtype & rho,
                                                              const rtype & T) const = 0;
        virtual rtype get_pressure_from_density_energy(const rtype & rho,
                                                       const rtype & e) const = 0;
        virtual rtype h_get_pressure_from_density_energy(const rtype & rho,
                                                         const rtype & e) const = 0;
        virtual rtype get_sound_speed_from_pressure_density(const rtype & p,
                                                            const rtype & rho) const = 0;
        virtual rtype h_get_sound_speed_from_pressure_density(const rtype & p,
                                                              const rtype & rho) const = 0;
        virtual void compute_primitives_from_conservatives(rtype * primitives,
                                                           const rtype * conservatives) const = 0;
        virtual void h_compute_primitives_from_conservatives(rtype * primitives,
                                                             const rtype * conservatives) const = 0;
        virtual void copy_host_to_device() = 0;
        virtual void copy_device_to_host() = 0;
    };

    template<typename T_physics>
    struct Model : Concept {
        T_physics physics;
        Model(T_physics p) : physics(std::move(p)) {}
        void init(const toml::value & input) override {
            physics.init(input);
        }
        void print() const override {
            physics.print();
        }
        PhysicsType get_type() const override {
            return physics.get_type();
        }
        rtype get_gamma() const override {
            return physics.get_gamma();
        }
        rtype get_energy_from_temperature(const rtype & T) const override {
            return physics.get_energy_from_temperature(T);
        }
        rtype h_get_energy_from_temperature(const rtype & T) const override {
            return physics.h_get_energy_from_temperature(T);
        }
        rtype get_temperature_from_energy(const rtype & e) const override {
            return physics.get_temperature_from_energy(e);
        }
        rtype h_get_temperature_from_energy(const rtype & e) const override {
            return physics.h_get_temperature_from_energy(e);
        }
        rtype get_density_from_pressure_temperature(const rtype & p,
                                                    const rtype & T) const override {
            return physics.get_density_from_pressure_temperature(p, T);
        }
        rtype h_get_density_from_pressure_temperature(const rtype & p,
                                                      const rtype & T) const override {
            return physics.h_get_density_from_pressure_temperature(p, T);
        }
        rtype get_temperature_from_density_pressure(const rtype & rho,
                                                    const rtype & p) const override {
            return physics.get_temperature_from_density_pressure(rho, p);
        }
        rtype h_get_temperature_from_density_pressure(const rtype & rho,
                                                      const rtype & p) const override {
            return physics.h_get_temperature_from_density_pressure(rho, p);
        }
        rtype get_pressure_from_density_temperature(const rtype & rho,
                                                    const rtype & T) const override {
            return physics.get_pressure_from_density_temperature(rho, T);
        }
        rtype h_get_pressure_from_density_temperature(const rtype & rho,
                                                      const rtype & T) const override {
            return physics.h_get_pressure_from_density_temperature(rho, T);
        }
        rtype get_pressure_from_density_energy(const rtype & rho,
                                               const rtype & e) const override {
            return physics.get_pressure_from_density_energy(rho, e);
        }
        rtype h_get_pressure_from_density_energy(const rtype & rho,
                                                 const rtype & e) const override {
            return physics.h_get_pressure_from_density_energy(rho, e);
        }
        rtype get_sound_speed_from_pressure_density(const rtype & p,
                                                    const rtype & rho) const override {
            return physics.get_sound_speed_from_pressure_density(p, rho);
        }
        rtype h_get_sound_speed_from_pressure_density(const rtype & p,
                                                      const rtype & rho) const override {
            return physics.h_get_sound_speed_from_pressure_density(p, rho);
        }
        void compute_primitives_from_conservatives(rtype * primitives,
                                                   const rtype * conservatives) const override {
            physics.compute_primitives_from_conservatives(primitives, conservatives);
        }
        void h_compute_primitives_from_conservatives(rtype * primitives,
                                                     const rtype * conservatives) const override {
            physics.h_compute_primitives_from_conservatives(primitives, conservatives);
        }
        void copy_host_to_device() override {
            physics.copy_host_to_device();
        }
        void copy_device_to_host() override {
            physics.copy_device_to_host();
        }
    };

    std::unique_ptr<Concept> pimpl;

public:
    template<typename T>
    PhysicsWrapper(T physics) : pimpl(std::make_unique<Model<T>>(std::move(physics))) {}

    void init(const toml::value & input) override {
        pimpl->init(input);
    }
    void print() const override {
        pimpl->print();
    }
    PhysicsType get_type() const override {
        return pimpl->get_type();
    }
    rtype get_gamma() const override {
        return pimpl->get_gamma();
    }
    rtype get_energy_from_temperature(const rtype & T) const override {
        return pimpl->get_energy_from_temperature(T);
    }
    rtype h_get_energy_from_temperature(const rtype & T) const override {
        return pimpl->h_get_energy_from_temperature(T);
    }
    rtype get_temperature_from_energy(const rtype & e) const override {
        return pimpl->get_temperature_from_energy(e);
    }
    rtype h_get_temperature_from_energy(const rtype & e) const override {
        return pimpl->h_get_temperature_from_energy(e);
    }
    rtype get_density_from_pressure_temperature(const rtype & p,
                                                const rtype & T) const override {
        return pimpl->get_density_from_pressure_temperature(p, T);
    }
    rtype h_get_density_from_pressure_temperature(const rtype & p,
                                                  const rtype & T) const override {
        return pimpl->h_get_density_from_pressure_temperature(p, T);
    }
    rtype get_temperature_from_density_pressure(const rtype & rho,
                                                const rtype & p) const override {
        return pimpl->get_temperature_from_density_pressure(rho, p);
    }
    rtype h_get_temperature_from_density_pressure(const rtype & rho,
                                                  const rtype & p) const override {
        return pimpl->h_get_temperature_from_density_pressure(rho, p);
    }
    rtype get_pressure_from_density_temperature(const rtype & rho,
                                                const rtype & T) const override {
        return pimpl->get_pressure_from_density_temperature(rho, T);
    }
    rtype h_get_pressure_from_density_temperature(const rtype & rho,
                                                  const rtype & T) const override {
        return pimpl->h_get_pressure_from_density_temperature(rho, T);
    }
    rtype get_pressure_from_density_energy(const rtype & rho,
                                           const rtype & e) const override {
        return pimpl->get_pressure_from_density_energy(rho, e);
    }
    rtype h_get_pressure_from_density_energy(const rtype & rho,
                                             const rtype & e) const override {
        return pimpl->h_get_pressure_from_density_energy(rho, e);
    }
    rtype get_sound_speed_from_pressure_density(const rtype & p,
                                                const rtype & rho) const override {
        return pimpl->get_sound_speed_from_pressure_density(p, rho);
    }
    rtype h_get_sound_speed_from_pressure_density(const rtype & p,
                                                  const rtype & rho) const override {
        return pimpl->h_get_sound_speed_from_pressure_density(p, rho);
    }
    void compute_primitives_from_conservatives(rtype * primitives,
                                               const rtype * conservatives) const override {
        pimpl->compute_primitives_from_conservatives(primitives, conservatives);
    }
    void h_compute_primitives_from_conservatives(rtype * primitives,
                                                 const rtype * conservatives) const override {
        pimpl->h_compute_primitives_from_conservatives(primitives, conservatives);
    }
    void copy_host_to_device() override {
        pimpl->copy_host_to_device();
    }
    void copy_device_to_host() override {
        pimpl->copy_device_to_host();
    }

    template<typename T>
    T* get_as() {
        auto model = dynamic_cast<Model<T>*>(pimpl.get());
        return model ? &model->physics : nullptr;
    }
};

rtype Euler::get_energy_from_temperature_impl(const rtype & T) const {
    return get_Cv() * T;
}

rtype Euler::get_temperature_from_energy_impl(const rtype & e) const {
    return e / get_Cv();
}

rtype Euler::get_density_from_pressure_temperature_impl(const rtype & p,
                                                        const rtype & T) const {
    return p / (get_R() * T);
}

rtype Euler::get_temperature_from_density_pressure_impl(const rtype & rho,
                                                        const rtype & p) const {
    return p / (rho * get_R());
}

rtype Euler::get_pressure_from_density_temperature_impl(const rtype & rho,
                                                        const rtype & T) const {
    return rho * get_R() * T;
}

rtype Euler::get_pressure_from_density_energy_impl(const rtype & rho,
                                                   const rtype & e) const {
    return Kokkos::fmax(get_p_min(), Kokkos::fmin(get_p_max(), (get_gamma() - 1.0) * rho * e));
}

rtype Euler::get_sound_speed_from_pressure_density_impl(const rtype & p,
                                                        const rtype & rho) const {
    return Kokkos::sqrt(get_gamma() * p / rho);
}

void Euler::compute_primitives_from_conservatives_impl(rtype * primitives,
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
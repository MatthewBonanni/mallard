/**
 * @file boundary_wall_adiabatic.h
 * @author Matthew Bonanni (mbonanni001@gmail.com)
 * @brief Adiabatic wall boundary condition class declaration.
 * @version 0.1
 * @date 2023-12-20
 * 
 * @copyright Copyright (c) 2023 Matthew Bonanni
 * 
 */

#ifndef BOUNDARY_WALL_ADIABATIC_H
#define BOUNDARY_WALL_ADIABATIC_H

#include "boundary.h"

#include <Kokkos_Core.hpp>

#include "flux_functor.h"

class BoundaryWallAdiabatic : public Boundary {
    public:
        /**
         * @brief Construct a new BoundaryWallAdiabatic object
         */
        BoundaryWallAdiabatic();

        /**
         * @brief Destroy the BoundaryWallAdiabatic object
         */
        ~BoundaryWallAdiabatic();

        /**
         * @brief Print the boundary.
         */
        void print() override;

        /**
         * @brief Initialize the boundary.
         * @param input Pointer to the TOML input.
         */
        void init(const toml::value & input) override;

        /**
         * @brief Apply the boundary condition.
         * @param face_solution Face solution.
         * @param rhs Right hand side.
         */
        void apply(Kokkos::View<rtype **[2][N_CONSERVATIVE]> face_solution,
                   Kokkos::View<rtype *[N_CONSERVATIVE]> rhs) override;
    protected:
    private:
        /**
         * @brief Launch the flux functor.
         */
        template <typename T_physics, typename T_riemann_solver>
        void launch_flux_functor(Kokkos::View<rtype **[2][N_CONSERVATIVE]> face_solution,
                                 Kokkos::View<rtype *[N_CONSERVATIVE]> rhs);

        template <typename T_physics, typename T_riemann_solver>
        class WallAdiabaticFluxFunctor : public BaseFluxFunctor<WallAdiabaticFluxFunctor<T_physics, T_riemann_solver>,
                                                                T_physics,
                                                                T_riemann_solver> {
            public:
                using BaseFluxFunctor<WallAdiabaticFluxFunctor, T_physics, T_riemann_solver>::BaseFluxFunctor;

                KOKKOS_INLINE_FUNCTION
                void call_impl(const u_int32_t i_face_local) const;
        };
};

#endif // BOUNDARY_WALL_ADIABATIC_H
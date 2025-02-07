/**
 * @file boundary_symmetry.h
 * @author Matthew Bonanni (mbonanni001@gmail.com)
 * @brief Symmetry boundary condition class definition.
 * @version 0.1
 * @date 2024-01-04
 * 
 * @copyright Copyright (c) 2024 Matthew Bonanni
 * 
 */

#ifndef BOUNDARY_SYMMETRY_H
#define BOUNDARY_SYMMETRY_H

#include "boundary.h"

#include <Kokkos_Core.hpp>

#include "flux_functor.h"

class BoundarySymmetry : public Boundary {
    public:
        /**
         * @brief Construct a new BoundarySymmetry object
         */
        BoundarySymmetry();

        /**
         * @brief Destroy the BoundarySymmetry object
         */
        ~BoundarySymmetry();

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
        class SymmetryFluxFunctor : public BaseFluxFunctor<SymmetryFluxFunctor<T_physics, T_riemann_solver>,
                                                           T_physics,
                                                           T_riemann_solver> {
            public:
                using BaseFluxFunctor<SymmetryFluxFunctor, T_physics, T_riemann_solver>::BaseFluxFunctor;

                KOKKOS_INLINE_FUNCTION
                void calc_lr_states_impl(const uint32_t i_face,
                                         const uint8_t i_quad,
                                         rtype * conservatives_l,
                                         rtype * conservatives_r,
                                         rtype * primitives_l,
                                         rtype * primitives_r) const;
        };
};

#endif // BOUNDARY_SYMMETRY_H
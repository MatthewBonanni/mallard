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
         * @param face_solution Pointer to the face solution vector.
         * @param rhs Pointer to the right-hand side vector.
         */
        void apply(Kokkos::View<rtype *[2][N_CONSERVATIVE]> face_solution,
                   Kokkos::View<rtype *[N_CONSERVATIVE]> rhs) override;
    protected:
    private:
};

#endif // BOUNDARY_WALL_ADIABATIC_H
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
         * @param face_solution Pointer to the face solution vector.
         * @param rhs Pointer to the right-hand side vector.
         */
        void apply(view_3d * face_solution,
                   view_2d * rhs) override;
    protected:
    private:
};

#endif // BOUNDARY_SYMMETRY_H
/**
 * @file boundary_extrapolation.h
 * @author Matthew Bonanni (mbonanni001@gmail.com)
 * @brief Extrapolation boundary condition class declaration.
 * @version 0.1
 * @date 2024-01-18
 * 
 * @copyright Copyright (c) 2024 Matthew Bonanni
 * 
 */

#ifndef BOUNDARY_EXTRAPOLATION_H
#define BOUNDARY_EXTRAPOLATION_H

#include "boundary.h"

class BoundaryExtrapolation : public Boundary {
    public:
        /**
         * @brief Construct a new BoundaryExtrapolation object
         */
        BoundaryExtrapolation();

        /**
         * @brief Destroy the BoundaryExtrapolation object
         */
        ~BoundaryExtrapolation();

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

#endif // BOUNDARY_EXTRAPOLATION_H
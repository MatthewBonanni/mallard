/**
 * @file boundary_p_out.h
 * @author Matthew Bonanni (mbonanni001@gmail.com)
 * @brief Outflow pressure boundary condition class declaration.
 * @version 0.1
 * @date 2023-12-20
 * 
 * @copyright Copyright (c) 2023 Matthew Bonanni
 * 
 */

#ifndef BOUNDARY_P_OUT_H
#define BOUNDARY_P_OUT_H

#include "boundary.h"

class BoundaryPOut : public Boundary {
    public:
        /**
         * @brief Construct a new BoundaryPOut object
         */
        BoundaryPOut();

        /**
         * @brief Destroy the BoundaryPOut object
         */
        ~BoundaryPOut();

        /**
         * @brief Print the boundary.
         */
        void print() override;

        /**
         * @brief Initialize the boundary.
         * @param input Pointer to the TOML input.
         */
        void init(const toml::table & input) override;

        /**
         * @brief Apply the boundary condition.
         * @param face_solution Pointer to the face solution vector.
         * @param rhs Pointer to the right-hand side vector.
         */
        void apply(FaceStateVector * face_solution,
                   StateVector * rhs) override;
    protected:
    private:
        double p_bc;
};

#endif // BOUNDARY_P_OUT_H
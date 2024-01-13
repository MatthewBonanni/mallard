/**
 * @file boundary_upt.h
 * @author Matthew Bonanni (mbonanni001@gmail.com)
 * @brief UPT boundary condition class declaration.
 * @version 0.1
 * @date 2023-12-20
 * 
 * @copyright Copyright (c) 2023 Matthew Bonanni
 * 
 */

#ifndef BOUNDARY_UPT_H
#define BOUNDARY_UPT_H

#include "boundary.h"

class BoundaryUPT : public Boundary {
    public:
        /**
         * @brief Construct a new BoundaryUPT object
         */
        BoundaryUPT();

        /**
         * @brief Destroy the BoundaryUPT object
         */
        ~BoundaryUPT();

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
        void apply(view_3d * face_solution,
                   view_2d * rhs) override;
    protected:
    private:
        // Input
        NVector u_bc;
        rtype p_bc;
        rtype T_bc;

        // Dependent
        rtype rho_bc;
        rtype e_bc;
        rtype h_bc;
        Primitives primitives_bc;
};

#endif // BOUNDARY_UPT_H
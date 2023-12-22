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
         * @brief Initialize the boundary.
         * @param input Pointer to the TOML input.
         */
        void init(const toml::table& input) override;
    protected:
    private:
        std::array<double, 2> u;
        double p;
        double T;
};

#endif // BOUNDARY_UPT_H
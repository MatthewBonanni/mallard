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
    protected:
    private:
};

#endif // BOUNDARY_UPT_H
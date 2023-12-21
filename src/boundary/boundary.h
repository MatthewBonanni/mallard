/**
 * @file boundary.h
 * @author Matthew Bonanni (mbonanni001@gmail.com)
 * @brief Boundary class declaration.
 * @version 0.1
 * @date 2023-12-20
 * 
 * @copyright Copyright (c) 2023
 * 
 */

#ifndef BOUNDARY_H
#define BOUNDARY_H

#include "mesh/zone.h"

class Boundary {
    public:
        /**
         * @brief Construct a new Boundary object
         */
        Boundary();

        /**
         * @brief Destroy the Boundary object
         */
        ~Boundary();
    protected:
    private:
        FaceZone zone;
};

#endif // BOUNDARY_H
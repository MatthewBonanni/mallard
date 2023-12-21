/**
 * @file boundary.h
 * @author Matthew Bonanni (mbonanni001@gmail.com)
 * @brief Boundary class declaration.
 * @version 0.1
 * @date 2023-12-20
 * 
 * @copyright Copyright (c) 2023 Matthew Bonanni
 * 
 */

#ifndef BOUNDARY_H
#define BOUNDARY_H

#include "mesh/zone.h"

enum class BoundaryType {
    WALL_ADIABATIC = 1,
    UPT = 2,
    P_OUT = 3
};

static const std::unordered_map<std::string, BoundaryType> BOUNDARY_TYPES = {
    {"wall_adiabatic", BoundaryType::WALL_ADIABATIC},
    {"upt", BoundaryType::UPT},
    {"p_out", BoundaryType::P_OUT}
};

static const std::unordered_map<BoundaryType, std::string> BOUNDARY_NAMES = {
    {BoundaryType::WALL_ADIABATIC, "wall_adiabatic"},
    {BoundaryType::UPT, "upt"},
    {BoundaryType::P_OUT, "p_out"}
};

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

        /**
         * @brief Set the zone.
         * @param zone Pointer to the zone.
         */
        void set_zone(FaceZone * zone);
    protected:
    private:
        FaceZone * zone;
};

#endif // BOUNDARY_H
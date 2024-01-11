/**
 * @file zone.h
 * @author Matthew Bonanni (mbonanni001@gmail.com)
 * @brief Zone class declarations.
 * @version 0.1
 * @date 2023-12-20
 * 
 * @copyright Copyright (c) 2023 Matthew Bonanni
 * 
 */

#ifndef ZONES_H
#define ZONES_H

#include <string>
#include <vector>

#include "common.h"

enum class FaceZoneType {
    UNKNOWN = -1,
    BOUNDARY = 1,
    INTERNAL = 2
};

enum class CellZoneType {
    UNKNOWN = -1,
    FLUID = 1
};

class Zone {
    public:
        /**
         * @brief Construct a new Zone object
         */
        Zone();

        /**
         * @brief Destroy the Zone object
         */
        ~Zone();

        /**
         * @brief Get the name of the zone.
         * @return Name of the zone.
         */
        std::string get_name() const;

        /**
         * @brief Set the name of the zone.
         * @param name Name of the zone.
         */
        void set_name(const std::string& name);
    protected:
    private:
        std::string name;
};

class FaceZone : public Zone {
    public:
        /**
         * @brief Construct a new FaceZone object
         */
        FaceZone();

        /**
         * @brief Destroy the FaceZone object
         */
        ~FaceZone();

        /**
         * @brief Get the number of faces in the zone.
         * @return Number of faces in the zone.
         */
        int n_faces() const;

        /**
         * @brief Get a pointer to the vector of faces of the zone.
         * @return Faces of the zone.
         */
        std::vector<int> * faces();

        /**
         * @brief Get the type of the zone.
         * @return Type of the zone.
         */
        FaceZoneType get_type() const;

        /**
         * @brief Set the type of the zone.
         * @param type Type of the zone.
         */
        void set_type(FaceZoneType type);
    protected:
    private:
        std::vector<int> m_faces;
        FaceZoneType type;
};

class CellZone : public Zone {
    public:
        /**
         * @brief Construct a new CellZone object
         */
        CellZone();

        /**
         * @brief Destroy the CellZone object
         */
        ~CellZone();

        /**
         * @brief Get the number of cells in the zone.
         * @return Number of cells in the zone.
         */
        int n_cells() const;

        /**
         * @brief Get the cells of the zone.
         * @return Cells of the zone.
         */
        std::vector<int> cells() const;

        /**
         * @brief Get the type of the zone.
         * @return Type of the zone.
         */
        CellZoneType type() const;
    protected:
    private:
        std::vector<int> m_cells;
        CellZoneType m_type;
};

#endif // ZONES_H
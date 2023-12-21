/**
 * @file zone.h
 * @author Matthew Bonanni (mbonanni001@gmail.com)
 * @brief Zone class declarations.
 * @version 0.1
 * @date 2023-12-20
 * 
 * @copyright Copyright (c) 2023
 * 
 */

#ifndef ZONES_H
#define ZONES_H

#include <string>
#include <vector>

enum class FaceZoneKind {
    UNKNOWN = -1,
    BOUNDARY = 1,
    INTERNAL = 2
};

enum class CellZoneKind {
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
         * @brief Get the index of the zone.
         * @return Index of the zone.
         */
        int get_index() const;

        /**
         * @brief Get the name of the zone.
         * @return Name of the zone.
         */
        std::string get_name() const;
    protected:
    private:
        int index;
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
         * @brief Get the faces of the zone.
         * @return Faces of the zone.
         */
        std::vector<int> faces() const;

        /**
         * @brief Get the kind of the zone.
         * @return Kind of the zone.
         */
        FaceZoneKind kind() const;
    protected:
    private:
        std::vector<int> m_faces;
        FaceZoneKind m_kind;
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
         * @brief Get the cells of the zone.
         * @return Cells of the zone.
         */
        std::vector<int> cells() const;

        /**
         * @brief Get the kind of the zone.
         * @return Kind of the zone.
         */
        CellZoneKind kind() const;
    protected:
    private:
        std::vector<int> m_cells;
        CellZoneKind m_kind;
};

#endif // ZONES_H
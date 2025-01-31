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
    BOUNDARY,
    INTERIOR
};

enum class CellZoneType {
    FLUID
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
        u_int32_t n_faces() const;

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

        /**
         * @brief Copy the faces from the host to the device.
         */
        void copy_host_to_device();

        /**
         * @brief Copy the faces from the device to the host.
         */
        void copy_device_to_host();

        Kokkos::View<u_int32_t *> faces;
        Kokkos::View<u_int32_t *>::HostMirror h_faces;
    protected:
    private:
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
        u_int32_t n_cells() const;

        /**
         * @brief Get the type of the zone.
         * @return Type of the zone.
         */
        CellZoneType type() const;

        /**
         * @brief Copy the cells from the host to the device.
         */
        void copy_host_to_device();

        /**
         * @brief Copy the cells from the device to the host.
         */
        void copy_device_to_host();

        Kokkos::View<u_int32_t *> cells;
        Kokkos::View<u_int32_t *>::HostMirror h_cells;
    protected:
    private:
        CellZoneType m_type;
};

#endif // ZONES_H
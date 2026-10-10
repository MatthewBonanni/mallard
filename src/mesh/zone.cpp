/**
 * @file zone.cpp
 * @author Matthew Bonanni (mbonanni001@gmail.com)
 * @brief Zone class implementations.
 * @version 0.1
 * @date 2023-12-20
 * 
 * @copyright Copyright (c) 2023 Matthew Bonanni
 * 
 */

#include "zone.h"

Zone::Zone() {
    // Empty
}

Zone::~Zone() {
    // Empty
}

std::string Zone::get_name() const {
    return name;
}

void Zone::set_name(const std::string& name_in) {
    this->name = name_in;
}

FaceZone::FaceZone() {
    // Empty
}

FaceZone::~FaceZone() {
    // Empty
}

uint32_t FaceZone::n_faces() const {
    return static_cast<uint32_t>(faces.extent(0));
}

FaceZoneType FaceZone::get_type() const {
    return type;
}

void FaceZone::set_type(FaceZoneType type_in) {
    this->type = type_in;
}

void FaceZone::copy_host_to_device() {
    Kokkos::deep_copy(faces, h_faces);
}

void FaceZone::copy_device_to_host() {
    Kokkos::deep_copy(h_faces, faces);
}

CellZone::CellZone() {
    // Empty
}

CellZone::~CellZone() {
    // Empty
}

uint32_t CellZone::n_cells() const {
    return static_cast<uint32_t>(cells.extent(0));
}

CellZoneType CellZone::type() const {
    return m_type;
}

void CellZone::copy_host_to_device() {
    Kokkos::deep_copy(cells, h_cells);
}

void CellZone::copy_device_to_host() {
    Kokkos::deep_copy(h_cells, cells);
}
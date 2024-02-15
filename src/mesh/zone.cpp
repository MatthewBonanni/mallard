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

void Zone::set_name(const std::string& name) {
    this->name = name;
}

FaceZone::FaceZone() {
    // Empty
}

FaceZone::~FaceZone() {
    // Empty
}

u_int32_t FaceZone::n_faces() const {
    return m_faces.size();
}

std::vector<u_int32_t> * FaceZone::faces() {
    return &m_faces;
}

FaceZoneType FaceZone::get_type() const {
    return type;
}

void FaceZone::set_type(FaceZoneType type) {
    this->type = type;
}

CellZone::CellZone() {
    // Empty
}

CellZone::~CellZone() {
    // Empty
}

u_int32_t CellZone::n_cells() const {
    return m_cells.size();
}

std::vector<u_int32_t> CellZone::cells() const {
    return m_cells;
}

CellZoneType CellZone::type() const {
    return m_type;
}
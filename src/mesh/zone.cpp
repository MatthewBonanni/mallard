/**
 * @file zone.cpp
 * @author Matthew Bonanni (mbonanni001@gmail.com)
 * @brief Zone class implementations.
 * @version 0.1
 * @date 2023-12-20
 * 
 * @copyright Copyright (c) 2023
 * 
 */

#include "zone.h"

Zone::Zone() {
    // Empty
}

Zone::~Zone() {
    // Empty
}

int Zone::get_index() const {
    return index;
}

std::string Zone::get_name() const {
    return name;
}

FaceZone::FaceZone() {
    // Empty
}

FaceZone::~FaceZone() {
    // Empty
}

std::vector<int> FaceZone::faces() const {
    return m_faces;
}

FaceZoneKind FaceZone::kind() const {
    return m_kind;
}

CellZone::CellZone() {
    // Empty
}

CellZone::~CellZone() {
    // Empty
}

std::vector<int> CellZone::cells() const {
    return m_cells;
}

CellZoneKind CellZone::kind() const {
    return m_kind;
}
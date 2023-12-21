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

std::vector<int> * FaceZone::faces() {
    return &m_faces;
}

FaceZoneKind FaceZone::get_kind() const {
    return kind;
}

void FaceZone::set_kind(FaceZoneKind kind) {
    this->kind = kind;
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
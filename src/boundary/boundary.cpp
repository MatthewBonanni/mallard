/**
 * @file boundary.cpp
 * @author Matthew Bonanni (mbonanni001@gmail.com)
 * @brief Boundary class implementation.
 * @version 0.1
 * @date 2023-12-20
 * 
 * @copyright Copyright (c) 2023 Matthew Bonanni
 * 
 */

#include "boundary.h"

Boundary::Boundary() {
    // Empty
}

Boundary::~Boundary() {
    // Empty
}

void Boundary::set_zone(FaceZone * zone) {
    this->zone = zone;
}

void Boundary::init(const toml::table& input) {
    throw std::runtime_error("Boundary::init() not implemented.");
}
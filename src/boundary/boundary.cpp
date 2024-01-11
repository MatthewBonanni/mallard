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

#include <iostream>
#include <utility>

#include "common.h"

Boundary::Boundary() {
    // Empty
}

Boundary::~Boundary() {
    std::cout << "Destroying boundary: " << zone->get_name() << std::endl;
}

void Boundary::set_zone(FaceZone * zone) {
    this->zone = zone;
}

void Boundary::set_mesh(std::shared_ptr<Mesh> mesh) {
    this->mesh = mesh;
}

void Boundary::set_physics(std::shared_ptr<Physics> physics) {
    this->physics = physics;
}

void Boundary::print() {
    std::cout << LOG_SEPARATOR << std::endl;
    std::cout << "Boundary: " << zone->get_name() << std::endl;
    std::cout << "Type: " << BOUNDARY_NAMES.at(type) << std::endl;
}

void Boundary::init(const toml::table & input) {
    throw std::runtime_error("Boundary::init() not implemented.");
}

void Boundary::apply(FaceStateVector * face_solution,
                     StateVector * rhs) {
    throw std::runtime_error("Boundary::apply() not implemented.");
}